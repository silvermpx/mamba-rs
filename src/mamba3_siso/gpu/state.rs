//! Mamba-3 SISO shared state types: dimensions, saved activations,
//! reusable scratch buffers, and target-network scratch.
//!
//! Split from the former 2313-line `mamba3_gpu.rs` (task #381). The
//! structs and constructors live here; `forward.rs` and `backward.rs`
//! contain the launch-orchestration code.

use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use std::sync::Arc;

pub(crate) type CUptr = cudarc::driver::sys::CUdeviceptr;

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

/// Collected Mamba-3 dimensions for GPU launches.
#[derive(Debug, Clone, Copy)]
pub struct GpuMamba3Dims {
    pub batch: usize,
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub nheads: usize,
    pub headdim: usize,
    pub ngroups: usize,
    pub in_proj_dim: usize,
    pub seq_len: usize,
    pub mamba_input_dim: usize,
    pub n_layers: usize,
    pub n_angles: usize,
    pub a_floor: f32,
    pub is_outproj_norm: bool,
    /// false = sequential SSM (m3_burnin_fwd/m3_backward_seq), faster at T<=64.
    /// true = parallel chunked scan (10-kernel pipeline), faster at T>64.
    pub use_parallel_scan: bool,
}

impl GpuMamba3Dims {
    /// batch * seq_len.
    pub fn bt(&self) -> usize {
        self.batch * self.seq_len
    }

    /// Chunk size for parallel scan (matches Mamba-3 reference implementation
    /// `state-spaces/mamba/mamba_ssm/modules/mamba3.py`).
    pub fn chunk_size(&self) -> usize {
        CHUNK_SIZE
    }

    /// Number of chunks for parallel scan.
    pub fn n_chunks(&self) -> usize {
        self.seq_len.div_ceil(self.chunk_size())
    }
}

/// Chunk size for the Mamba-3 SISO chunked SSD parallel scan. Single source
/// of truth — every site that allocates `[B * n_chunks * nh * chunk_size]`
/// scratch must reference this constant rather than re-spelling `64`.
pub const CHUNK_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// Saved activations (per layer)
// ---------------------------------------------------------------------------

/// Per-layer saved activations for backward.
pub struct GpuMamba3LayerActs {
    // F1: RMSNorm
    pub residual: GpuBuffer,  // [B*T*d_model]
    pub rms_vals: GpuBuffer,  // [B*T]
    pub post_norm: GpuBuffer, // [B*T*d_model]
    // F3: Split
    pub z: GpuBuffer,          // [B*T*d_inner]
    pub x: GpuBuffer,          // [B*T*d_inner]
    pub b_raw: GpuBuffer,      // [B*T*ng*ds]
    pub c_raw: GpuBuffer,      // [B*T*ng*ds]
    pub dd_dt_raw: GpuBuffer,  // [B*T*nh] -- saved for backward
    pub dd_a_raw: GpuBuffer,   // [B*T*nh] -- saved for backward
    pub trap_raw: GpuBuffer,   // [B*T*nh] -- saved for backward
    pub dt: GpuBuffer,         // [B*T*nh] -- post-softplus
    pub a_val: GpuBuffer,      // [B*T*nh] -- -softplus(dd_A), clamped
    pub trap: GpuBuffer,       // [B*T*nh] -- post-sigmoid
    pub angles_raw: GpuBuffer, // [B*T*n_angles]
    // F4: BCNorm + bias + RoPE
    pub b_normed: GpuBuffer, // [B*T*ng*ds]
    pub c_normed: GpuBuffer, // [B*T*ng*ds]
    pub b_rms: GpuBuffer,    // [B*T*ng]
    pub c_rms: GpuBuffer,    // [B*T*ng]
    pub b_biased: GpuBuffer, // [B*T*nh*ds]
    pub c_biased: GpuBuffer, // [B*T*nh*ds]
    pub k: GpuBuffer,        // [B*T*nh*ds] -- post-RoPE B
    pub q: GpuBuffer,        // [B*T*nh*ds] -- post-RoPE C
    // F5: Angle accumulation
    pub angle_cumsum: GpuBuffer, // [B*T*nh*n_angles]
    pub alpha: GpuBuffer,        // [B*T*nh]
    pub beta: GpuBuffer,         // [B*T*nh]
    pub gamma: GpuBuffer,        // [B*T*nh]
    // F6: SSM
    pub h_saved: GpuBuffer, // [B*(T+1)*d_inner*ds] -- transposed layout (sequential)
    pub k_prev_saved: GpuBuffer, // [B*T*nh*ds] -- saved k_state at each t (sequential)
    pub v_prev_saved: GpuBuffer, // [B*T*nh*hd] -- saved v_state at each t (sequential)
    pub y: GpuBuffer,       // [B*T*d_inner]
    // F6 parallel: saved intermediates (skip recompute in backward)
    pub da_cumsum_saved: GpuBuffer,    // [B*T*nh] -- dA cumulative sum
    pub k_scaled_saved: GpuBuffer,     // [B*T*nh*ds] -- prescaled K
    pub scale_saved: GpuBuffer,        // [B*T*nh]
    pub gamma_saved: GpuBuffer,        // [B*T*nh]
    pub qk_dot_saved: GpuBuffer,       // [B*T*nh]
    pub chunk_states_saved: GpuBuffer, // [B*nc*nh*hd*ds] -- entering chunk states
    // F7: RMSNormGated
    pub gated_rms_vals: GpuBuffer, // [B*T*nheads] — one rstd per group (group_size=headdim)
    pub gated: GpuBuffer,          // [B*T*d_inner]
}

/// Backbone activations (all layers + input proj + norm_f).
pub struct GpuMamba3BackboneActs {
    pub input_proj_inputs: GpuBuffer,  // [B*T*mamba_input_dim]
    pub input_proj_outputs: GpuBuffer, // [B*T*d_model]
    pub layers: Vec<GpuMamba3LayerActs>,
    pub norm_f_input: GpuBuffer, // [B*T*d_model]
    pub norm_f_rms: GpuBuffer,   // [B*T]
}

// ---------------------------------------------------------------------------
// Scratch buffers
// ---------------------------------------------------------------------------

/// Reusable scratch for Mamba-3 forward + backward.
pub struct GpuMamba3Scratch {
    // Forward
    pub proj_flat: GpuBuffer, // [B*T*in_proj_dim]
    pub out_flat: GpuBuffer,  // [B*T*d_model]

    // Backward
    pub d_gated: GpuBuffer,       // [B*T*d_inner]
    pub d_y: GpuBuffer,           // [B*T*d_inner]
    pub d_z: GpuBuffer,           // [B*T*d_inner]
    pub d_norm_gate_w: GpuBuffer, // [B*T*d_inner]
    pub d_x: GpuBuffer,           // [B*T*d_inner]
    pub d_k: GpuBuffer,           // [B*T*nh*ds]
    pub d_q: GpuBuffer,           // [B*T*nh*ds]
    pub d_proj: GpuBuffer,        // [B*T*in_proj_dim]
    pub d_norm: GpuBuffer,        // [B*T*d_model]
    pub d_pre_norm: GpuBuffer,    // [B*T*d_model]

    // SSM backward reduction
    pub d_d_local: GpuBuffer, // [B*d_inner] -- per-thread D grad
    pub d_alpha: GpuBuffer,   // [B*T*nh] -- gradient of alpha
    pub d_beta: GpuBuffer,    // [B*T*nh] -- gradient of beta
    pub d_gamma: GpuBuffer,   // [B*T*nh] -- gradient of gamma

    // B5: ABG backward + angle_dt backward
    pub d_dt_angle: GpuBuffer,   // [B*T*nh] -- angle contribution to d_dt
    pub d_angles_raw: GpuBuffer, // [B*T*n_angles] -- from angle_dt_bwd
    pub d_dd_dt: GpuBuffer,      // [B*T*nh] -- d_dd_dt_raw from m3_abg_bwd
    pub d_dd_a: GpuBuffer,       // [B*T*nh] -- d_dd_a_raw from m3_abg_bwd
    pub d_trap_raw: GpuBuffer,   // [B*T*nh] -- d_trap_raw from m3_abg_bwd

    // B4: RoPE backward + bias backward + BCNorm backward
    pub d_b_pre_rope: GpuBuffer,   // [B*T*nh*ds] -- from rope_bwd
    pub d_c_pre_rope: GpuBuffer,   // [B*T*nh*ds] -- from rope_bwd
    pub d_angle_cumsum: GpuBuffer, // [B*T*nh*n_angles] -- from rope_bwd
    pub d_b_normed: GpuBuffer,     // [B*T*ng*ds] -- head->group reduced
    pub d_c_normed: GpuBuffer,     // [B*T*ng*ds] -- head->group reduced
    pub d_b_raw: GpuBuffer,        // [B*T*ng*ds] -- from bcnorm_bwd
    pub d_c_raw: GpuBuffer,        // [B*T*ng*ds] -- from bcnorm_bwd
    pub d_b_norm_w: GpuBuffer,     // [B*T*ng*ds] -- per-block dw (reduce later)
    pub d_c_norm_w: GpuBuffer,     // [B*T*ng*ds] -- per-block dw (reduce later)

    // Discarded dx for input_proj backward
    pub d_input_proj_dx: GpuBuffer, // [B*T*mamba_input_dim]

    // Chunked parallel scan buffers
    pub da_cumsum: GpuBuffer,     // [B * n_chunks * nheads * chunk_size]
    pub da_cs_sum: GpuBuffer,     // [B * nheads * n_chunks] -- per-chunk decay sum
    pub chunk_states: GpuBuffer,  // [B * n_chunks * nheads * headdim * d_state]
    pub final_states: GpuBuffer,  // [B * nheads * headdim * d_state]
    pub d_da_cumsum: GpuBuffer,   // backward
    pub d_prev_states: GpuBuffer, // backward
    // Parallel backward intermediate buffers (Steps 2-6)
    pub d_scale: GpuBuffer,     // [B * T * nh] -- dScale from dqktheta
    pub d_gamma_par: GpuBuffer, // [B * T * nh] -- dGamma from dqktheta
    pub d_qk_dot: GpuBuffer,    // [B * T * nh] -- dQK_dot from dqkv

    // -- Rule-B axis-0 reduction partials (Phase 2.7.5 M3 determinism fix) --
    /// Scratch for per-sample partials produced by Rule-B backward kernels.
    /// Reduced via `reduce_sum_axis0` to produce deterministic cross-batch
    /// accumulators (replaces the previous atomicAdd accumulators).
    ///
    /// Sized to fit the largest consumer:
    ///   - rmsnorm_bwd partials: `B*T * d_model`
    ///   - angle_dt_bwd partials (2 tensors): `nh * B*T*na + na * B*T*nh = 2 * nh * B*T * na`
    ///   - m3_dqkv dD_partials: `B * nh`
    ///
    /// The buffer is reused across kernels; only one consumer writes at a time.
    pub axis0_partials: GpuBuffer,
}

/// Layer pointers for persistent state (ssm + k_state + v_state + angle_state).
pub struct Mamba3LayerPtrs {
    pub ssm_state: CUptr,   // &mut [B * nh * hd * ds]
    pub k_state: CUptr,     // &mut [B * nh * ds]
    pub v_state: CUptr,     // &mut [B * nh * hd]
    pub angle_state: CUptr, // &mut [B * nh * n_angles]
}

/// Target network scratch (forward only, no backward).
pub struct GpuMamba3TargetScratch {
    pub proj_flat: GpuBuffer,     // [B*T*in_proj_dim]
    pub z: GpuBuffer,             // [B*T*d_inner]
    pub x: GpuBuffer,             // [B*T*d_inner]
    pub b_raw: GpuBuffer,         // [B*T*ng*ds]
    pub c_raw: GpuBuffer,         // [B*T*ng*ds]
    pub dd_dt_raw: GpuBuffer,     // [B*T*nh]
    pub dd_a_raw: GpuBuffer,      // [B*T*nh]
    pub trap_raw: GpuBuffer,      // [B*T*nh]
    pub dt: GpuBuffer,            // [B*T*nh]
    pub a_val: GpuBuffer,         // [B*T*nh]
    pub trap: GpuBuffer,          // [B*T*nh]
    pub angles_raw: GpuBuffer,    // [B*T*n_angles]
    pub b_normed: GpuBuffer,      // [B*T*ng*ds]
    pub c_normed: GpuBuffer,      // [B*T*ng*ds]
    pub b_rms: GpuBuffer,         // [B*T*ng]
    pub c_rms: GpuBuffer,         // [B*T*ng]
    pub b_biased: GpuBuffer,      // [B*T*nh*ds]
    pub c_biased: GpuBuffer,      // [B*T*nh*ds]
    pub k: GpuBuffer,             // [B*T*nh*ds]
    pub q: GpuBuffer,             // [B*T*nh*ds]
    pub angle_cumsum: GpuBuffer,  // [B*T*nh*n_angles]
    pub alpha: GpuBuffer,         // [B*T*nh]
    pub beta: GpuBuffer,          // [B*T*nh]
    pub gamma: GpuBuffer,         // [B*T*nh]
    pub y: GpuBuffer,             // [B*T*d_inner]
    pub gated: GpuBuffer,         // [B*T*d_inner]
    pub temporal_work: GpuBuffer, // [B*T*d_model] — working temporal (input_proj output → layers → norm_f)
    pub out_flat: GpuBuffer,      // [B*T*d_model]
    pub residual: GpuBuffer,      // [B*T*d_model]
    pub rms_discard: GpuBuffer,   // [B*T*nheads] — per-group rstd (discarded in target)
    // Persistent state for target burn-in
    pub ssm_states: GpuBuffer,   // [B*n_layers*nh*hd*ds]
    pub k_states: GpuBuffer,     // [B*n_layers*nh*ds]
    pub v_states: GpuBuffer,     // [B*n_layers*nh*hd]
    pub angle_states: GpuBuffer, // [B*n_layers*nh*n_angles]
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl GpuMamba3LayerActs {
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        bt: usize,
        dims: &GpuMamba3Dims,
    ) -> Result<Self, String> {
        let dm = dims.d_model;
        let di = dims.d_inner;
        let ds = dims.d_state;
        let nh = dims.nheads;
        let hd = dims.headdim;
        let ng = dims.ngroups;
        let b = dims.batch;
        let t = dims.seq_len;
        let na = dims.n_angles;
        let nc = dims.n_chunks();
        let cs = dims.chunk_size();
        Ok(Self {
            residual: GpuBuffer::zeros(stream, bt * dm)?,
            rms_vals: GpuBuffer::zeros(stream, bt)?,
            post_norm: GpuBuffer::zeros(stream, bt * dm)?,
            z: GpuBuffer::zeros(stream, bt * di)?,
            x: GpuBuffer::zeros(stream, bt * di)?,
            b_raw: GpuBuffer::zeros(stream, bt * ng * ds)?,
            c_raw: GpuBuffer::zeros(stream, bt * ng * ds)?,
            dd_dt_raw: GpuBuffer::zeros(stream, bt * nh)?,
            dd_a_raw: GpuBuffer::zeros(stream, bt * nh)?,
            trap_raw: GpuBuffer::zeros(stream, bt * nh)?,
            dt: GpuBuffer::zeros(stream, bt * nh)?,
            a_val: GpuBuffer::zeros(stream, bt * nh)?,
            trap: GpuBuffer::zeros(stream, bt * nh)?,
            angles_raw: GpuBuffer::zeros(stream, bt * na.max(1))?,
            b_normed: GpuBuffer::zeros(stream, bt * ng * ds)?,
            c_normed: GpuBuffer::zeros(stream, bt * ng * ds)?,
            b_rms: GpuBuffer::zeros(stream, bt * ng)?,
            c_rms: GpuBuffer::zeros(stream, bt * ng)?,
            b_biased: GpuBuffer::zeros(stream, bt * nh * ds)?,
            c_biased: GpuBuffer::zeros(stream, bt * nh * ds)?,
            k: GpuBuffer::zeros(stream, bt * nh * ds)?,
            q: GpuBuffer::zeros(stream, bt * nh * ds)?,
            angle_cumsum: GpuBuffer::zeros(stream, bt * nh * na.max(1))?,
            alpha: GpuBuffer::zeros(stream, bt * nh)?,
            beta: GpuBuffer::zeros(stream, bt * nh)?,
            gamma: GpuBuffer::zeros(stream, bt * nh)?,
            h_saved: GpuBuffer::zeros(stream, b * (t + 1) * di * ds)?,
            k_prev_saved: GpuBuffer::zeros(stream, bt * nh * ds)?,
            v_prev_saved: GpuBuffer::zeros(stream, bt * nh * hd)?,
            y: GpuBuffer::zeros(stream, bt * di)?,
            da_cumsum_saved: GpuBuffer::zeros(stream, b * nc * nh * cs)?,
            k_scaled_saved: GpuBuffer::zeros(stream, bt * nh * ds)?,
            scale_saved: GpuBuffer::zeros(stream, bt * nh)?,
            gamma_saved: GpuBuffer::zeros(stream, bt * nh)?,
            qk_dot_saved: GpuBuffer::zeros(stream, bt * nh)?,
            chunk_states_saved: GpuBuffer::zeros(stream, b * nc * nh * hd * ds)?,
            gated_rms_vals: GpuBuffer::zeros(stream, bt * nh)?,
            gated: GpuBuffer::zeros(stream, bt * di)?,
        })
    }
}

impl GpuMamba3BackboneActs {
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMamba3Dims,
    ) -> Result<Self, String> {
        let bt = dims.bt();
        Ok(Self {
            input_proj_inputs: GpuBuffer::zeros(stream, bt * dims.mamba_input_dim)?,
            input_proj_outputs: GpuBuffer::zeros(stream, bt * dims.d_model)?,
            layers: (0..dims.n_layers)
                .map(|_| GpuMamba3LayerActs::new(stream, bt, dims))
                .collect::<Result<_, _>>()?,
            norm_f_input: GpuBuffer::zeros(stream, bt * dims.d_model)?,
            norm_f_rms: GpuBuffer::zeros(stream, bt)?,
        })
    }
}

impl GpuMamba3Scratch {
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMamba3Dims,
    ) -> Result<Self, String> {
        let bt = dims.bt();
        let di = dims.d_inner;
        let dm = dims.d_model;
        let ds = dims.d_state;
        let nh = dims.nheads;
        let hd = dims.headdim;
        let ip = dims.in_proj_dim;
        let b = dims.batch;
        Ok(Self {
            proj_flat: GpuBuffer::zeros(stream, bt * ip)?,
            out_flat: GpuBuffer::zeros(stream, bt * dm)?,
            d_gated: GpuBuffer::zeros(stream, bt * di)?,
            d_y: GpuBuffer::zeros(stream, bt * di)?,
            d_z: GpuBuffer::zeros(stream, bt * di)?,
            d_norm_gate_w: GpuBuffer::zeros(stream, bt * di)?,
            d_x: GpuBuffer::zeros(stream, bt * di)?,
            d_k: GpuBuffer::zeros(stream, bt * nh * ds)?,
            d_q: GpuBuffer::zeros(stream, bt * nh * ds)?,
            d_proj: GpuBuffer::zeros(stream, bt * ip)?,
            d_norm: GpuBuffer::zeros(stream, bt * dm)?,
            d_pre_norm: GpuBuffer::zeros(stream, bt * dm)?,
            d_d_local: GpuBuffer::zeros(stream, b * di)?,
            d_alpha: GpuBuffer::zeros(stream, bt * nh)?,
            d_beta: GpuBuffer::zeros(stream, bt * nh)?,
            d_gamma: GpuBuffer::zeros(stream, bt * nh)?,
            d_dt_angle: GpuBuffer::zeros(stream, bt * nh)?,
            d_angles_raw: GpuBuffer::zeros(stream, bt * dims.n_angles.max(1))?,
            d_dd_dt: GpuBuffer::zeros(stream, bt * nh)?,
            d_dd_a: GpuBuffer::zeros(stream, bt * nh)?,
            d_trap_raw: GpuBuffer::zeros(stream, bt * nh)?,
            d_b_pre_rope: GpuBuffer::zeros(stream, bt * nh * ds)?,
            d_c_pre_rope: GpuBuffer::zeros(stream, bt * nh * ds)?,
            d_angle_cumsum: GpuBuffer::zeros(stream, bt * nh * dims.n_angles.max(1))?,
            d_b_normed: GpuBuffer::zeros(stream, bt * dims.ngroups * ds)?,
            d_c_normed: GpuBuffer::zeros(stream, bt * dims.ngroups * ds)?,
            d_b_raw: GpuBuffer::zeros(stream, bt * dims.ngroups * ds)?,
            d_c_raw: GpuBuffer::zeros(stream, bt * dims.ngroups * ds)?,
            d_b_norm_w: GpuBuffer::zeros(stream, bt * dims.ngroups * ds)?,
            d_c_norm_w: GpuBuffer::zeros(stream, bt * dims.ngroups * ds)?,
            d_input_proj_dx: GpuBuffer::zeros(stream, bt * dims.mamba_input_dim)?,
            da_cumsum: {
                let cs = dims.chunk_size();
                let nc = dims.n_chunks();
                GpuBuffer::zeros(stream, b * nc * nh * cs)?
            },
            da_cs_sum: {
                let nc = dims.n_chunks();
                GpuBuffer::zeros(stream, b * nh * nc)?
            },
            chunk_states: {
                let nc = dims.n_chunks();
                GpuBuffer::zeros(stream, b * nc * nh * hd * ds)?
            },
            final_states: GpuBuffer::zeros(stream, b * nh * hd * ds)?,
            d_da_cumsum: {
                let cs = dims.chunk_size();
                let nc = dims.n_chunks();
                GpuBuffer::zeros(stream, b * nc * nh * cs)?
            },
            d_prev_states: {
                let nc = dims.n_chunks();
                GpuBuffer::zeros(stream, b * nc * nh * hd * ds)?
            },
            d_scale: GpuBuffer::zeros(stream, bt * nh)?,
            d_gamma_par: GpuBuffer::zeros(stream, bt * nh)?,
            d_qk_dot: GpuBuffer::zeros(stream, bt * nh)?,
            // Rule-B axis-0 partials — sized to fit largest M3 consumer.
            axis0_partials: {
                let na = dims.n_angles.max(1);
                let angle_dt_sz = 2 * nh * bt * na;
                let rmsnorm_sz = bt * dm;
                let d_d_sz = b * nh;
                GpuBuffer::zeros(stream, angle_dt_sz.max(rmsnorm_sz).max(d_d_sz))?
            },
        })
    }
}

impl GpuMamba3TargetScratch {
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMamba3Dims,
    ) -> Result<Self, String> {
        let bt = dims.bt();
        let di = dims.d_inner;
        let dm = dims.d_model;
        let ds = dims.d_state;
        let nh = dims.nheads;
        let hd = dims.headdim;
        let ng = dims.ngroups;
        let ip = dims.in_proj_dim;
        let b = dims.batch;
        let nl = dims.n_layers;
        let na = dims.n_angles.max(1);
        Ok(Self {
            proj_flat: GpuBuffer::zeros(stream, bt * ip)?,
            z: GpuBuffer::zeros(stream, bt * di)?,
            x: GpuBuffer::zeros(stream, bt * di)?,
            b_raw: GpuBuffer::zeros(stream, bt * ng * ds)?,
            c_raw: GpuBuffer::zeros(stream, bt * ng * ds)?,
            dd_dt_raw: GpuBuffer::zeros(stream, bt * nh)?,
            dd_a_raw: GpuBuffer::zeros(stream, bt * nh)?,
            trap_raw: GpuBuffer::zeros(stream, bt * nh)?,
            dt: GpuBuffer::zeros(stream, bt * nh)?,
            a_val: GpuBuffer::zeros(stream, bt * nh)?,
            trap: GpuBuffer::zeros(stream, bt * nh)?,
            angles_raw: GpuBuffer::zeros(stream, bt * na)?,
            b_normed: GpuBuffer::zeros(stream, bt * ng * ds)?,
            c_normed: GpuBuffer::zeros(stream, bt * ng * ds)?,
            b_rms: GpuBuffer::zeros(stream, bt * ng)?,
            c_rms: GpuBuffer::zeros(stream, bt * ng)?,
            b_biased: GpuBuffer::zeros(stream, bt * nh * ds)?,
            c_biased: GpuBuffer::zeros(stream, bt * nh * ds)?,
            k: GpuBuffer::zeros(stream, bt * nh * ds)?,
            q: GpuBuffer::zeros(stream, bt * nh * ds)?,
            // `.max(1)` — keep the sentinel allocation when RoPE is disabled
            // (na==0) so kernels that unconditionally index into the buffer
            // don't touch a zero-length allocation. Mirrors the guard on
            // f32 `GpuMamba3LayerActs::angle_cumsum`.
            angle_cumsum: GpuBuffer::zeros(stream, bt * nh * na.max(1))?,
            alpha: GpuBuffer::zeros(stream, bt * nh)?,
            beta: GpuBuffer::zeros(stream, bt * nh)?,
            gamma: GpuBuffer::zeros(stream, bt * nh)?,
            y: GpuBuffer::zeros(stream, bt * di)?,
            gated: GpuBuffer::zeros(stream, bt * di)?,
            temporal_work: GpuBuffer::zeros(stream, bt * dm)?,
            out_flat: GpuBuffer::zeros(stream, bt * dm)?,
            residual: GpuBuffer::zeros(stream, bt * dm)?,
            rms_discard: GpuBuffer::zeros(stream, bt * nh)?,
            ssm_states: GpuBuffer::zeros(stream, b * nl * nh * hd * ds)?,
            k_states: GpuBuffer::zeros(stream, b * nl * nh * ds)?,
            v_states: GpuBuffer::zeros(stream, b * nl * nh * hd)?,
            angle_states: GpuBuffer::zeros(stream, b * nl * nh * na.max(1))?,
        })
    }
}
