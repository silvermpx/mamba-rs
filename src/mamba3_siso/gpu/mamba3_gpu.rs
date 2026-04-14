//! GPU forward and backward pipeline for Mamba-3 SISO layers.
//!
//! Mamba-3 SISO differs from Mamba-2 SSD:
//! - No conv1d (removed entirely)
//! - Input-dependent A (via dd_A in split, not fixed a_log)
//! - BCNorm + bias + RoPE on B and C
//! - Angle accumulation (sequential per head)
//! - Trapezoidal recurrence (alpha/beta/gamma) instead of exponential (da_exp)
//! - Additional persistent state: k_state, v_state, angle_state
//!
//! 7-phase forward (2 SGEMM, vs Mamba-2's 3 -- no conv1d proj):
//!   F1: RMSNorm
//!   F2: in_proj SGEMM [B*T, d_model -> in_proj_dim]
//!   F3: m3_split (8-way + fused activations)
//!   F4: BCNorm + bias + RoPE on B,C
//!   F5: angle_dt accumulation -> trapezoidal coefficients (alpha/beta/gamma)
//!   F6: SSM recurrence (trapezoidal)
//!   F7: RMSNormGated(y, z) or y*SiLU(z)
//!   F8: out_proj SGEMM + residual
//!
//! Source: CPU reference: mamba3_siso/cpu/forward.rs
//! Paper: Lahoti et al. "Mamba-3: SISO" (ICLR 2026)

use super::kernels::Mamba3Kernels;
use super::weights::{
    GpuMamba3Grads, GpuMamba3LayerGrads, GpuMamba3LayerWeights, GpuMamba3Weights,
};
use crate::mamba_ssm::gpu::blas::{gpu_sgemm_backward_grad_raw, gpu_sgemm_forward_raw};
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::launch::{grid_1d, grid_norm};
use cudarc::driver::PushKernelArg;
use std::sync::Arc;

type CUptr = cudarc::driver::sys::CUdeviceptr;

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

    /// Chunk size for parallel scan (64, matching Mamba-3 reference implementation).
    pub fn chunk_size(&self) -> usize {
        64
    }

    /// Number of chunks for parallel scan.
    pub fn n_chunks(&self) -> usize {
        self.seq_len.div_ceil(self.chunk_size())
    }
}

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
            // Parallel scan intermediates (saved to skip recompute in backward)
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
            // B5: ABG backward + angle_dt backward
            d_dt_angle: GpuBuffer::zeros(stream, bt * nh)?,
            d_angles_raw: GpuBuffer::zeros(stream, bt * dims.n_angles.max(1))?,
            d_dd_dt: GpuBuffer::zeros(stream, bt * nh)?,
            d_dd_a: GpuBuffer::zeros(stream, bt * nh)?,
            d_trap_raw: GpuBuffer::zeros(stream, bt * nh)?,
            // B4: RoPE backward + bias backward + BCNorm backward
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
            angle_cumsum: GpuBuffer::zeros(stream, bt * nh * na)?,
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
            angle_states: GpuBuffer::zeros(stream, b * nl * nh * na)?,
        })
    }
}

// ---------------------------------------------------------------------------
// GPU Forward -- single layer (sequential mode)
// ---------------------------------------------------------------------------

/// Mamba-3 SISO single-layer GPU forward (8-phase pipeline, sequential SSM).
pub fn gpu_forward_mamba3_layer(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    temporal: &mut GpuBuffer, // [B*T*d_model] in/out
    acts: &mut GpuMamba3LayerActs,
    lw: &GpuMamba3LayerWeights,
    layer_ptrs: &Mamba3LayerPtrs,
    scratch: &mut GpuMamba3Scratch,
    dims: &GpuMamba3Dims,
) -> Result<(), String> {
    let bt = dims.bt();
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let nh = dims.nheads;
    let hd = dims.headdim;
    let ng = dims.ngroups;
    let ip = dims.in_proj_dim;
    let na = dims.n_angles;

    // F1: RMSNorm -- save residual, compute post_norm
    acts.residual.copy_from_raw(temporal, &ctx.stream)?;
    {
        let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_fwd);
        builder.arg(acts.post_norm.inner_mut());
        builder.arg(acts.rms_vals.inner_mut());
        builder.arg(temporal.inner());
        let nw_ptr = lw.norm_weight.raw_ptr(&ctx.stream);
        builder.arg(&nw_ptr);
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let eps: f32 = 1e-5;
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        builder.arg(&eps);
        unsafe { builder.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("rmsnorm_fwd m3 F1: {:?}", e))?;
    }

    // F2: in_proj SGEMM [B*T, d_model -> in_proj_dim]
    gpu_sgemm_forward_raw(
        ctx,
        &mut scratch.proj_flat,
        &acts.post_norm,
        lw.in_proj_w.raw_ptr(&ctx.stream),
        None,
        (bt, dm, ip),
    )?;

    // F3: m3_split -- 8-way split + fused activations
    {
        let n_i = bt as i32;
        let di_i = di as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let nh_i = nh as i32;
        let na_i = na as i32;
        let db_ptr = lw.dt_bias.raw_ptr(&ctx.stream);
        let mut builder = ctx.stream.launch_builder(&m3k.m3_split);
        builder.arg(acts.z.inner_mut());
        builder.arg(acts.x.inner_mut());
        builder.arg(acts.b_raw.inner_mut());
        builder.arg(acts.c_raw.inner_mut());
        builder.arg(acts.dt.inner_mut());
        builder.arg(acts.a_val.inner_mut());
        builder.arg(acts.trap.inner_mut());
        builder.arg(acts.angles_raw.inner_mut());
        builder.arg(acts.dd_dt_raw.inner_mut());
        builder.arg(acts.dd_a_raw.inner_mut());
        builder.arg(acts.trap_raw.inner_mut());
        builder.arg(scratch.proj_flat.inner());
        builder.arg(&db_ptr);
        builder.arg(&dims.a_floor);
        builder.arg(&n_i);
        builder.arg(&di_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        builder.arg(&nh_i);
        builder.arg(&na_i);
        unsafe { builder.launch(grid_1d(bt * ip)) }.map_err(|e| format!("m3_split F3: {:?}", e))?;
    }

    // F4a: BCNorm on B
    {
        let bn_ptr = lw.b_norm_weight.raw_ptr(&ctx.stream);
        let n_i = bt as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((bt * ng) as u32, 1, 1),
            block_dim: (ds as u32, 1, 1),
            shared_mem_bytes: ds as u32 * 4,
        };
        let mut builder = ctx.stream.launch_builder(&m3k.bcnorm_fwd);
        builder.arg(acts.b_normed.inner_mut());
        builder.arg(acts.b_rms.inner_mut());
        builder.arg(acts.b_raw.inner());
        builder.arg(&bn_ptr);
        builder.arg(&n_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("bcnorm_fwd B F4a: {:?}", e))?;
    }

    // F4b: BCNorm on C
    {
        let cn_ptr = lw.c_norm_weight.raw_ptr(&ctx.stream);
        let n_i = bt as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((bt * ng) as u32, 1, 1),
            block_dim: (ds as u32, 1, 1),
            shared_mem_bytes: ds as u32 * 4,
        };
        let mut builder = ctx.stream.launch_builder(&m3k.bcnorm_fwd);
        builder.arg(acts.c_normed.inner_mut());
        builder.arg(acts.c_rms.inner_mut());
        builder.arg(acts.c_raw.inner());
        builder.arg(&cn_ptr);
        builder.arg(&n_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("bcnorm_fwd C F4b: {:?}", e))?;
    }

    // F4c: Bias add on B (expand groups -> heads)
    {
        let bb_ptr = lw.b_bias.raw_ptr(&ctx.stream);
        let n_i = bt as i32;
        let nh_i = nh as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.bc_bias_add);
        builder.arg(acts.b_biased.inner_mut());
        builder.arg(acts.b_normed.inner());
        builder.arg(&bb_ptr);
        builder.arg(&n_i);
        builder.arg(&nh_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(bt * nh * ds)) }
            .map_err(|e| format!("bc_bias_add B F4c: {:?}", e))?;
    }

    // F4d: Bias add on C
    {
        let cb_ptr = lw.c_bias.raw_ptr(&ctx.stream);
        let n_i = bt as i32;
        let nh_i = nh as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.bc_bias_add);
        builder.arg(acts.c_biased.inner_mut());
        builder.arg(acts.c_normed.inner());
        builder.arg(&cb_ptr);
        builder.arg(&n_i);
        builder.arg(&nh_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(bt * nh * ds)) }
            .map_err(|e| format!("bc_bias_add C F4d: {:?}", e))?;
    }

    // F5: angle_dt sequential accumulation (B envs, T timesteps each)
    // Must accumulate sequentially per batch element for correct state carry.
    if na > 0 {
        let b_i = (bt / dims.seq_len) as i32;
        let t_i = dims.seq_len as i32;
        let nh_i = nh as i32;
        let na_i = na as i32;
        let angle_st = layer_ptrs.angle_state;
        let mut builder = ctx.stream.launch_builder(&m3k.m3_angle_dt_fwd_seq);
        builder.arg(acts.angle_cumsum.inner_mut());
        builder.arg(&angle_st);
        builder.arg(acts.angles_raw.inner());
        builder.arg(acts.dt.inner());
        builder.arg(&b_i);
        builder.arg(&t_i);
        builder.arg(&nh_i);
        builder.arg(&na_i);
        let grid = cudarc::driver::LaunchConfig {
            grid_dim: (
                (bt / dims.seq_len) as u32,
                (nh * na).div_ceil(256) as u32,
                1,
            ),
            block_dim: (256.min((nh * na) as u32), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(grid) }.map_err(|e| format!("angle_dt_fwd_seq F5: {:?}", e))?;
    }

    // F4e+f: RoPE on B->K and C->Q (AFTER angle_cumsum is computed)
    if na > 0 {
        let n_i = bt as i32;
        let nh_i = nh as i32;
        let ds_i = ds as i32;
        let na_i = na as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.rope_fwd);
        builder.arg(acts.k.inner_mut());
        builder.arg(acts.q.inner_mut());
        builder.arg(acts.b_biased.inner());
        builder.arg(acts.c_biased.inner());
        builder.arg(acts.angle_cumsum.inner());
        builder.arg(&n_i);
        builder.arg(&nh_i);
        builder.arg(&ds_i);
        builder.arg(&na_i);
        unsafe { builder.launch(grid_1d(bt * nh * ds)) }
            .map_err(|e| format!("rope_fwd F4ef: {:?}", e))?;
    } else {
        acts.k.copy_from_raw(&acts.b_biased, &ctx.stream)?;
        acts.q.copy_from_raw(&acts.c_biased, &ctx.stream)?;
    }

    // F5b: Compute alpha, beta, gamma from dt, a_val, trap
    // alpha = exp(a_val * dt), beta = alpha * dt * (1-trap), gamma = trap * dt
    {
        let n_total = (bt * nh) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.m3_compute_abg);
        builder.arg(acts.alpha.inner_mut());
        builder.arg(acts.beta.inner_mut());
        builder.arg(acts.gamma.inner_mut());
        builder.arg(acts.dt.inner());
        builder.arg(acts.a_val.inner());
        builder.arg(acts.trap.inner());
        builder.arg(&n_total);
        unsafe { builder.launch(grid_1d(bt * nh)) }
            .map_err(|e| format!("m3_compute_abg F5b: {:?}", e))?;
    }

    // F6: SSM forward
    if dims.use_parallel_scan {
        // Parallel chunked scan (5-kernel pipeline) — better for T>64
        let dp_ptr = lw.d_param.raw_ptr(&ctx.stream);
        let nh_i = nh as i32;
        let hd_i = hd as i32;
        let ds_i = ds as i32;
        let t_i = dims.seq_len as i32;
        let cs = dims.chunk_size() as i32;
        let nc = dims.n_chunks();
        let b_i = dims.batch as i32;

        {
            let n_total = (bt * nh) as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.elementwise_mul);
            builder.arg(scratch.d_alpha.inner_mut());
            builder.arg(acts.a_val.inner());
            builder.arg(acts.dt.inner());
            builder.arg(&n_total);
            unsafe { builder.launch(grid_1d(bt * nh)) }
                .map_err(|e| format!("adt compute F6: {:?}", e))?;
        }
        {
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: ((dims.batch * nc) as u32, nh as u32, 1),
                block_dim: (dims.chunk_size() as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.m3_preprocess_chunks);
            builder.arg(scratch.d_q.inner_mut());
            builder.arg(scratch.d_beta.inner_mut());
            builder.arg(scratch.d_gamma.inner_mut());
            builder.arg(scratch.d_dd_dt.inner_mut());
            builder.arg(acts.k.inner());
            builder.arg(acts.q.inner());
            builder.arg(acts.dt.inner());
            builder.arg(acts.trap.inner());
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(&ds_i);
            builder.arg(&cs);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("m3_preprocess_chunks F6 K1: {:?}", e))?;
        }
        {
            let block_x = nh.min(256) as u32;
            let grid_z = nh.div_ceil(block_x as usize) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dims.batch as u32, nc as u32, grid_z),
                block_dim: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.m3_da_cumsum);
            builder.arg(scratch.da_cumsum.inner_mut());
            builder.arg(scratch.d_alpha.inner());
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(&cs);
            unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_dA_cumsum F6 K2: {:?}", e))?;
        }
        {
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: ((dims.batch * nc) as u32, nh as u32, 1),
                block_dim: (hd as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.m3_chunk_state_fwd);
            builder.arg(scratch.chunk_states.inner_mut());
            builder.arg(acts.x.inner());
            builder.arg(scratch.d_q.inner());
            builder.arg(scratch.da_cumsum.inner());
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(&hd_i);
            builder.arg(&ds_i);
            builder.arg(&cs);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("m3_chunk_state_fwd F6 K3: {:?}", e))?;
        }
        {
            let dim = hd * ds;
            let block_x = dim.min(256) as u32;
            let grid_z = dim.div_ceil(block_x as usize) as u32;
            let nc_i = nc as i32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dims.batch as u32, nh as u32, grid_z),
                block_dim: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.m3_state_passing_fwd);
            builder.arg(scratch.chunk_states.inner_mut());
            builder.arg(scratch.final_states.inner_mut());
            builder.arg(scratch.da_cumsum.inner());
            builder.arg(&b_i);
            builder.arg(&nc_i);
            builder.arg(&nh_i);
            builder.arg(&hd_i);
            builder.arg(&ds_i);
            builder.arg(&cs);
            builder.arg(&t_i);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("m3_state_passing_fwd F6 K4: {:?}", e))?;
        }
        {
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: ((dims.batch * nc) as u32, nh as u32, 1),
                block_dim: (hd as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.m3_chunk_scan_fwd);
            builder.arg(acts.y.inner_mut());
            builder.arg(acts.x.inner());
            builder.arg(acts.q.inner());
            builder.arg(scratch.d_q.inner());
            builder.arg(scratch.d_beta.inner());
            builder.arg(scratch.da_cumsum.inner());
            builder.arg(scratch.chunk_states.inner());
            builder.arg(&dp_ptr);
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(&hd_i);
            builder.arg(&ds_i);
            builder.arg(&cs);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("m3_chunk_scan_fwd F6 K5: {:?}", e))?;
        }

        // Save intermediates for backward (skip recompute: eliminates 6 kernel launches)
        // D2D copies are async and cheap (~0.01ms each on same stream)
        acts.da_cumsum_saved
            .copy_from_raw(&scratch.da_cumsum, &ctx.stream)?;
        acts.k_scaled_saved
            .copy_from_raw(&scratch.d_q, &ctx.stream)?; // K_scaled stored in d_q
        acts.scale_saved
            .copy_from_raw(&scratch.d_gamma, &ctx.stream)?; // scale stored in d_gamma
        acts.gamma_saved
            .copy_from_raw(&scratch.d_dd_dt, &ctx.stream)?; // gamma stored in d_dd_dt
        acts.qk_dot_saved
            .copy_from_raw(&scratch.d_beta, &ctx.stream)?; // qk_dot stored in d_beta
        acts.chunk_states_saved
            .copy_from_raw(&scratch.chunk_states, &ctx.stream)?;

        // Writeback persistent SSM/K/V state (parallel path does not mutate layer_ptrs
        // in-place like the sequential m3_burnin_fwd kernel does).
        {
            let block_x = hd.max(ds) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dims.batch as u32, nh as u32, 1),
                block_dim: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.m3_writeback_parallel_states);
            builder.arg(&layer_ptrs.ssm_state);
            builder.arg(&layer_ptrs.k_state);
            builder.arg(&layer_ptrs.v_state);
            builder.arg(scratch.final_states.inner());
            builder.arg(acts.k.inner());
            builder.arg(acts.x.inner());
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(&hd_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("m3_writeback_parallel_states F6: {:?}", e))?;
        }
    } else {
        // Sequential SSM (1 kernel) — faster for T<=64
        let dp_ptr = lw.d_param.raw_ptr(&ctx.stream);
        let b_i = dims.batch as i32;
        let t_i = dims.seq_len as i32;
        let nh_i = nh as i32;
        let hd_i = hd as i32;
        let ds_i = ds as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (dims.batch as u32, nh as u32, 1),
            block_dim: (hd as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = ctx.stream.launch_builder(&m3k.m3_burnin_fwd);
        builder.arg(&layer_ptrs.ssm_state);
        builder.arg(&layer_ptrs.k_state);
        builder.arg(&layer_ptrs.v_state);
        builder.arg(acts.y.inner_mut());
        builder.arg(acts.h_saved.inner_mut());
        builder.arg(acts.k_prev_saved.inner_mut());
        builder.arg(acts.v_prev_saved.inner_mut());
        builder.arg(acts.x.inner());
        builder.arg(acts.k.inner());
        builder.arg(acts.q.inner());
        builder.arg(acts.alpha.inner());
        builder.arg(acts.beta.inner());
        builder.arg(acts.gamma.inner());
        builder.arg(&dp_ptr);
        builder.arg(&b_i);
        builder.arg(&t_i);
        builder.arg(&nh_i);
        builder.arg(&hd_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_burnin_fwd F6 seq: {:?}", e))?;
    }

    // F7: Output gating
    // Default (is_outproj_norm=false): gated = y * SiLU(z)
    // With norm (is_outproj_norm=true): gated = RMSNorm(y) * weight * SiLU(z)
    if dims.is_outproj_norm {
        assert!(
            di <= 1024,
            "d_inner ({di}) exceeds rmsnorm_gated shared memory limit (1024)"
        );
        let nw_ptr = lw.norm_gate_weight.raw_ptr(&ctx.stream);
        let bt_i = bt as i32;
        let di_i = di as i32;
        let hd_i = dims.headdim as i32;
        let grid = cudarc::driver::LaunchConfig {
            grid_dim: (bt as u32, 1, 1),
            block_dim: (di as u32, 1, 1),
            shared_mem_bytes: (di * std::mem::size_of::<f32>()) as u32,
        };
        let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_gated_fwd);
        builder.arg(acts.gated.inner_mut());
        builder.arg(acts.gated_rms_vals.inner_mut());
        builder.arg(acts.y.inner());
        builder.arg(acts.z.inner());
        builder.arg(&nw_ptr);
        builder.arg(&bt_i);
        builder.arg(&di_i);
        builder.arg(&hd_i);
        unsafe { builder.launch(grid) }.map_err(|e| format!("rmsnorm_gated_fwd m3 F7: {:?}", e))?;
    } else {
        // Plain gating: gated[i] = y[i] * silu(z[i])
        let n = (bt * di) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.silu_gate_fwd);
        builder.arg(acts.gated.inner_mut());
        builder.arg(acts.y.inner());
        builder.arg(acts.z.inner());
        builder.arg(&n);
        unsafe { builder.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("silu_gate_fwd m3 F7: {:?}", e))?;
    }

    // F8: out_proj SGEMM + residual
    gpu_sgemm_forward_raw(
        ctx,
        &mut scratch.out_flat,
        &acts.gated,
        lw.out_proj_w.raw_ptr(&ctx.stream),
        None,
        (bt, di, dm),
    )?;

    // Residual: temporal = out_flat + residual
    {
        let ne = (bt * dm) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.residual_add);
        builder.arg(temporal.inner_mut());
        builder.arg(scratch.out_flat.inner());
        builder.arg(acts.residual.inner());
        builder.arg(&ne);
        unsafe { builder.launch(grid_1d(bt * dm)) }
            .map_err(|e| format!("residual_add m3 F8: {:?}", e))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// GPU Forward -- full backbone
// ---------------------------------------------------------------------------

/// Mamba-3 SISO full backbone forward (input proj + N layers + norm_f).
pub fn gpu_forward_mamba3_backbone(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    temporal: &mut GpuBuffer,
    acts: &mut GpuMamba3BackboneActs,
    mamba_w: &GpuMamba3Weights,
    mamba_input: &GpuBuffer,
    ssm_states: &mut GpuBuffer,
    k_states: &mut GpuBuffer,
    v_states: &mut GpuBuffer,
    angle_states: &mut GpuBuffer,
    scratch: &mut GpuMamba3Scratch,
    dims: &GpuMamba3Dims,
) -> Result<(), String> {
    let bt = dims.bt();
    let dm = dims.d_model;
    let ds = dims.d_state;
    let nh = dims.nheads;
    let hd = dims.headdim;
    let na = dims.n_angles.max(1);

    // Save input projection inputs
    acts.input_proj_inputs
        .copy_from_raw(mamba_input, &ctx.stream)?;

    // Input projection SGEMM: [B*T, mamba_input_dim] -> [B*T, d_model]
    gpu_sgemm_forward_raw(
        ctx,
        temporal,
        mamba_input,
        mamba_w.input_proj_w.raw_ptr(&ctx.stream),
        Some(mamba_w.input_proj_b.raw_ptr(&ctx.stream)),
        (bt, dims.mamba_input_dim, dm),
    )?;

    acts.input_proj_outputs
        .copy_from_raw(temporal, &ctx.stream)?;

    // N layers
    let f32_sz = std::mem::size_of::<f32>() as u64;
    let ssm_base = ssm_states.raw_ptr(&ctx.stream);
    let k_base = k_states.raw_ptr(&ctx.stream);
    let v_base = v_states.raw_ptr(&ctx.stream);
    let a_base = angle_states.raw_ptr(&ctx.stream);

    for l in 0..dims.n_layers {
        let ssm_off = dims.batch * l * nh * hd * ds;
        let k_off = dims.batch * l * nh * ds;
        let v_off = dims.batch * l * nh * hd;
        let a_off = dims.batch * l * nh * na;

        let layer_ptrs = Mamba3LayerPtrs {
            ssm_state: ssm_base + ssm_off as u64 * f32_sz,
            k_state: k_base + k_off as u64 * f32_sz,
            v_state: v_base + v_off as u64 * f32_sz,
            angle_state: a_base + a_off as u64 * f32_sz,
        };

        gpu_forward_mamba3_layer(
            ctx,
            m3k,
            temporal,
            &mut acts.layers[l],
            &mamba_w.layers[l],
            &layer_ptrs,
            scratch,
            dims,
        )?;
    }

    // Final RMSNorm (norm_f)
    acts.norm_f_input.copy_from_raw(temporal, &ctx.stream)?;
    {
        let nf_ptr = mamba_w.norm_f_weight.raw_ptr(&ctx.stream);
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let eps: f32 = 1e-5;
        let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_fwd);
        builder.arg(temporal.inner_mut());
        builder.arg(acts.norm_f_rms.inner_mut());
        builder.arg(acts.norm_f_input.inner());
        builder.arg(&nf_ptr);
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        builder.arg(&eps);
        unsafe { builder.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("rmsnorm_fwd norm_f m3: {:?}", e))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// GPU Backward -- single layer (sequential mode)
// ---------------------------------------------------------------------------

/// Mamba-3 SISO single-layer GPU backward (sequential SSM backward).
pub fn gpu_backward_mamba3_layer(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    d_temporal: &mut GpuBuffer, // [B*T*d_model] in/out gradient
    acts: &GpuMamba3LayerActs,
    lw: &GpuMamba3LayerWeights,
    lg: &GpuMamba3LayerGrads,
    scratch: &mut GpuMamba3Scratch,
    dims: &GpuMamba3Dims,
) -> Result<(), String> {
    let bt = dims.bt();
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let nh = dims.nheads;
    let hd = dims.headdim;
    let ng = dims.ngroups;
    let ip = dims.in_proj_dim;
    let na = dims.n_angles;
    let b = dims.batch as i32;
    let t = dims.seq_len as i32;

    // B8: out_proj backward: d_gated = d_temporal @ out_proj_w^T, d_out_proj_w += gated^T @ d_temporal
    gpu_sgemm_backward_grad_raw(
        ctx,
        &mut scratch.d_gated,
        (&lg.out_proj_w, None),
        d_temporal,
        &acts.gated,
        lw.out_proj_w.raw_ptr(&ctx.stream),
        (bt, di, dm),
    )?;

    // B7: Output gating backward (must match F7 forward branch)
    if dims.is_outproj_norm {
        // RMSNormGated backward
        assert!(
            di <= 1024,
            "d_inner ({di}) exceeds rmsnorm_gated shared memory limit"
        );
        let nw_ptr = lw.norm_gate_weight.raw_ptr(&ctx.stream);
        let bt_i = bt as i32;
        let di_i = di as i32;
        let hd_i = dims.headdim as i32;
        let grid = cudarc::driver::LaunchConfig {
            grid_dim: (bt as u32, 1, 1),
            block_dim: (di as u32, 1, 1),
            shared_mem_bytes: (di * std::mem::size_of::<f32>()) as u32,
        };
        let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_gated_bwd);
        builder.arg(scratch.d_y.inner_mut());
        builder.arg(scratch.d_z.inner_mut());
        builder.arg(scratch.d_norm_gate_w.inner_mut());
        builder.arg(scratch.d_gated.inner());
        builder.arg(acts.y.inner());
        builder.arg(acts.z.inner());
        builder.arg(&nw_ptr);
        builder.arg(acts.gated_rms_vals.inner());
        builder.arg(&bt_i);
        builder.arg(&di_i);
        builder.arg(&hd_i);
        unsafe { builder.launch(grid) }.map_err(|e| format!("rmsnorm_gated_bwd m3 B7: {:?}", e))?;
        // Accumulate d_norm_gate_w
        {
            let n_i = di as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
            let dst = lg.norm_gate_weight.ptr();
            builder.arg(&dst);
            builder.arg(scratch.d_norm_gate_w.inner());
            let bt_i = bt as i32;
            builder.arg(&bt_i);
            builder.arg(&n_i);
            unsafe { builder.launch(grid_1d(di)) }
                .map_err(|e| format!("colsum d_norm_gate_w m3: {:?}", e))?;
        }
    } else {
        // Plain silu_gate backward: d_y, d_z from d_gated, y, z
        let n = (bt * di) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.silu_gate_bwd);
        builder.arg(scratch.d_y.inner_mut());
        builder.arg(scratch.d_z.inner_mut());
        builder.arg(scratch.d_gated.inner());
        builder.arg(acts.y.inner());
        builder.arg(acts.z.inner());
        builder.arg(&n);
        unsafe { builder.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("silu_gate_bwd m3 B7: {:?}", e))?;
    }

    // B6: SSM backward
    // Zero atomicAdd targets before backward kernel (both paths use atomicAdd)
    scratch.d_k.zero(&ctx.stream)?;
    scratch.d_q.zero(&ctx.stream)?;
    scratch.d_alpha.zero(&ctx.stream)?;
    scratch.d_beta.zero(&ctx.stream)?;
    scratch.d_gamma.zero(&ctx.stream)?;
    scratch.d_d_local.zero(&ctx.stream)?;

    if !dims.use_parallel_scan {
        // Sequential BPTT (1 kernel) — faster for T<=64
        let dp_ptr = lw.d_param.raw_ptr(&ctx.stream);
        let b_i = dims.batch as i32;
        let t_i = dims.seq_len as i32;
        let nh_i = nh as i32;
        let hd_i = hd as i32;
        let ds_i = ds as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (dims.batch as u32, nh as u32, 1),
            block_dim: (hd as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = ctx.stream.launch_builder(&m3k.m3_backward_seq);
        builder.arg(acts.h_saved.inner());
        builder.arg(acts.k_prev_saved.inner());
        builder.arg(acts.v_prev_saved.inner());
        builder.arg(acts.x.inner());
        builder.arg(acts.k.inner());
        builder.arg(acts.q.inner());
        builder.arg(acts.alpha.inner());
        builder.arg(acts.beta.inner());
        builder.arg(acts.gamma.inner());
        builder.arg(&dp_ptr);
        builder.arg(scratch.d_y.inner()); // d_y from B7
        builder.arg(scratch.d_x.inner_mut());
        builder.arg(scratch.d_k.inner_mut());
        builder.arg(scratch.d_q.inner_mut());
        builder.arg(scratch.d_alpha.inner_mut());
        builder.arg(scratch.d_beta.inner_mut());
        builder.arg(scratch.d_gamma.inner_mut());
        builder.arg(scratch.d_d_local.inner_mut());
        builder.arg(&b_i);
        builder.arg(&t_i);
        builder.arg(&nh_i);
        builder.arg(&hd_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_backward_seq B6: {:?}", e))?;

        // RoPE backward: d_k/d_q are post-RoPE gradients. Apply inverse rotation
        // to get d_b_pre_rope/d_c_pre_rope (pre-RoPE gradients for B4).
        // Also computes d_angle_cumsum for B5a angle_dt_bwd.
        if na > 0 {
            let n_i = bt as i32;
            let nh_i = nh as i32;
            let ds_i = ds as i32;
            let na_i = na as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.rope_bwd);
            builder.arg(scratch.d_b_pre_rope.inner_mut()); // d_k_pre_rope
            builder.arg(scratch.d_c_pre_rope.inner_mut()); // d_q_pre_rope
            builder.arg(scratch.d_angle_cumsum.inner_mut());
            builder.arg(scratch.d_k.inner()); // d_k (post-RoPE)
            builder.arg(scratch.d_q.inner()); // d_q (post-RoPE)
            builder.arg(acts.b_biased.inner()); // k_pre_rope
            builder.arg(acts.c_biased.inner()); // q_pre_rope
            builder.arg(acts.angle_cumsum.inner());
            builder.arg(&n_i);
            builder.arg(&nh_i);
            builder.arg(&ds_i);
            builder.arg(&na_i);
            unsafe { builder.launch(grid_1d(bt * nh * ds)) }
                .map_err(|e| format!("rope_bwd seq B6: {:?}", e))?;
        } else {
            scratch
                .d_b_pre_rope
                .copy_from_raw(&scratch.d_k, &ctx.stream)?;
            scratch
                .d_c_pre_rope
                .copy_from_raw(&scratch.d_q, &ctx.stream)?;
        }

        // B/C bias gradients: colsum d_b_pre_rope [B*T*nh*ds] → lg.b_bias [nh*ds]
        // (parallel path does this via atomicAdd in m3_dqktheta)
        {
            let d_bb_ptr = lg.b_bias.ptr();
            let bt_i = bt as i32;
            let nhds_i = (nh * ds) as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
            builder.arg(&d_bb_ptr);
            builder.arg(scratch.d_b_pre_rope.inner());
            builder.arg(&bt_i);
            builder.arg(&nhds_i);
            unsafe { builder.launch(grid_1d(nh * ds)) }
                .map_err(|e| format!("colsum d_b_bias seq: {:?}", e))?;
        }
        {
            let d_cb_ptr = lg.c_bias.ptr();
            let bt_i = bt as i32;
            let nhds_i = (nh * ds) as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
            builder.arg(&d_cb_ptr);
            builder.arg(scratch.d_c_pre_rope.inner());
            builder.arg(&bt_i);
            builder.arg(&nhds_i);
            unsafe { builder.launch(grid_1d(nh * ds)) }
                .map_err(|e| format!("colsum d_c_bias seq: {:?}", e))?;
        }

        // d_D: reduce d_d_local[B*d_inner] -> lg.d_param[nh] via m3_reduce_d_D
        // (colsum would write nh*hd into nh buffer — buffer overrun!)
        {
            let d_dp = lg.d_param.ptr();
            let b_i2 = dims.batch as i32;
            let nh_i = nh as i32;
            let hd_i = hd as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.m3_reduce_d_d);
            builder.arg(&d_dp);
            builder.arg(scratch.d_d_local.inner());
            builder.arg(&b_i2);
            builder.arg(&nh_i);
            builder.arg(&hd_i);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (nh as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_reduce_d_D seq B6: {:?}", e))?;
        }
    } else {
        // =============================================================
        // Chunked parallel backward: 7-step pipeline matching Python
        // =============================================================
        // R1: Recompute adt, dA_cumsum, K_scaled, qk_dot, scale, gamma
        // R2: Recompute chunk_states + state_passing (entering states)
        // R3: Extract da_cs_sum
        // S1: m3_dqkv -- dQ_mid, dK_mid, dV, dADT, dQK_dot, dD
        // S2: m3_dqktheta -- inverse rotary + dScale + dGamma + bias grads
        // S3: m3_ddt_dtrap -- dScale/dGamma -> dDT, dTrap
        // (m3_final_grads runs after B5a angle_dt_bwd)
        // =============================================================
        let dp_ptr = lw.d_param.raw_ptr(&ctx.stream);
        let nh_i = nh as i32;
        let hd_i = hd as i32;
        let ds_i = ds as i32;
        let t_i = dims.seq_len as i32;
        let cs = dims.chunk_size() as i32;
        let nc = dims.n_chunks();
        let b_i = dims.batch as i32;
        let na = dims.n_angles;

        // Load saved intermediates from forward (replaces R1+R2 recompute: 6 fewer kernel launches)
        scratch
            .da_cumsum
            .copy_from_raw(&acts.da_cumsum_saved, &ctx.stream)?;
        scratch
            .d_b_pre_rope
            .copy_from_raw(&acts.k_scaled_saved, &ctx.stream)?; // K_scaled
        scratch
            .d_scale
            .copy_from_raw(&acts.scale_saved, &ctx.stream)?;
        scratch
            .d_gamma_par
            .copy_from_raw(&acts.gamma_saved, &ctx.stream)?;
        scratch
            .d_qk_dot
            .copy_from_raw(&acts.qk_dot_saved, &ctx.stream)?;
        scratch
            .chunk_states
            .copy_from_raw(&acts.chunk_states_saved, &ctx.stream)?;

        // R3: Extract da_cs_sum (per-chunk cumsum totals)
        {
            let block_x = nh.min(256) as u32;
            let grid_z = nh.div_ceil(block_x as usize) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dims.batch as u32, nc as u32, grid_z),
                block_dim: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.m3_extract_da_cs_sum);
            builder.arg(scratch.da_cs_sum.inner_mut());
            builder.arg(scratch.da_cumsum.inner());
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(&cs);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("m3_extract_da_cs_sum B6: {:?}", e))?;
        }

        // Zero output buffers for m3_dqkv (uses atomicAdd for some outputs)
        {
            let zero: f32 = 0.0;
            for (buf, sz, label) in [
                (&mut scratch.d_x, bt * di, "d_x"),
                (&mut scratch.d_k, bt * nh * ds, "d_k"),
                (&mut scratch.d_q, bt * nh * ds, "d_q"),
            ] {
                let ne = sz as i32;
                let mut builder = ctx.stream.launch_builder(&m3k.fill_scalar);
                builder.arg(buf.inner_mut());
                builder.arg(&zero);
                builder.arg(&ne);
                unsafe { builder.launch(grid_1d(sz)) }
                    .map_err(|e| format!("zero {label} B6 par: {:?}", e))?;
            }
        }

        // S1: m3_dqkv -- main backward kernel
        // Outputs: dQ_mid, dK_mid, dV, dADT, dQK_dot, dD
        // dADT computed directly via reverse cumsum trick (no separate cumsum_bwd)
        {
            // Shared memory: q[CS*ds] + k[CS*ds] + v[CS*hd] + dO[CS*hd]
            //              + da_cs[CS] + qk[CS] + ssm[hd*ds]
            let cs_u = dims.chunk_size();
            let smem = (cs_u * ds + cs_u * ds + cs_u * hd + cs_u * hd + cs_u + cs_u + hd * ds) * 4;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (nh as u32, dims.batch as u32, 1),
                block_dim: (hd as u32, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.m3_dqkv);
            builder.arg(scratch.d_q.inner_mut()); // dQ_mid [B*T*nh*ds]
            builder.arg(scratch.d_k.inner_mut()); // dK_mid [B*T*nh*ds]
            builder.arg(scratch.d_x.inner_mut()); // dV = d_x [B*T*d_inner]
            builder.arg(scratch.d_alpha.inner_mut()); // dADT [B*T*nh]
            builder.arg(scratch.d_beta.inner_mut()); // dQK_dot [B*T*nh]
            let d_dp_ptr = lg.d_param.ptr();
            builder.arg(&d_dp_ptr); // dD [nh] atomicAdd directly into grad
            builder.arg(acts.q.inner()); // Q_rot (post bias+RoPE)
            builder.arg(scratch.d_b_pre_rope.inner()); // K_scaled
            builder.arg(acts.x.inner()); // V = x [B*T*d_inner]
            builder.arg(scratch.da_cumsum.inner()); // DA_CS
            builder.arg(scratch.da_cs_sum.inner()); // DA_CS_SUM
            builder.arg(scratch.d_qk_dot.inner()); // QK_dot_in (from R1c)
            builder.arg(scratch.chunk_states.inner()); // SSM_States (entering)
            builder.arg(scratch.d_y.inner()); // dO = d_y
            builder.arg(&dp_ptr); // D_param [nh]
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(&hd_i);
            builder.arg(&ds_i);
            builder.arg(&cs);
            unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_dqkv B6 S1: {:?}", e))?;
        }

        // Zero d_angle_cumsum before m3_dqktheta (kernel does direct writes only for
        // RoPE pair indices; non-RoPE indices would retain stale values)
        if na > 0 {
            let n_ac = (dims.batch * dims.seq_len * nh * na) as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.fill_scalar);
            builder.arg(scratch.d_angle_cumsum.inner_mut());
            let zero: f32 = 0.0;
            builder.arg(&zero);
            builder.arg(&n_ac);
            unsafe { builder.launch(grid_1d(n_ac as usize)) }
                .map_err(|e| format!("zero d_angle_cumsum: {:?}", e))?;
        }

        // S2: m3_dqktheta -- inverse rotary, dScale, dGamma, bias grads
        {
            let na_i = na as i32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: ((dims.batch * nc) as u32, nh as u32, 1),
                block_dim: (dims.chunk_size() as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let d_cb_ptr = lg.c_bias.ptr();
            let d_bb_ptr = lg.b_bias.ptr();
            let mut builder = ctx.stream.launch_builder(&m3k.m3_dqktheta);
            // Outputs
            builder.arg(scratch.d_c_pre_rope.inner_mut()); // dQ_pre [B*T*nh*ds]
            builder.arg(scratch.d_b_pre_rope.inner_mut()); // dK_pre [B*T*nh*ds]
            builder.arg(scratch.d_angle_cumsum.inner_mut()); // dAngles_cumsum
            // d_scale/d_gamma_par: both output (dScale/dGamma) and input (Scale/Gamma from R1c)
            // Use raw_ptr for input to avoid borrow conflict
            let scale_in_ptr = scratch.d_scale.raw_ptr(&ctx.stream);
            let gamma_in_ptr = scratch.d_gamma_par.raw_ptr(&ctx.stream);
            builder.arg(scratch.d_scale.inner_mut()); // dScale [B*T*nh]
            builder.arg(scratch.d_gamma_par.inner_mut()); // dGamma [B*T*nh]
            builder.arg(&d_cb_ptr); // dQ_bias [nh*ds] atomicAdd
            builder.arg(&d_bb_ptr); // dK_bias [nh*ds] atomicAdd
            // Inputs
            builder.arg(acts.c_biased.inner()); // Q_raw (pre-RoPE, post-bias)
            builder.arg(acts.b_biased.inner()); // K_raw
            builder.arg(&scale_in_ptr); // Scale_in (from R1c, raw ptr)
            builder.arg(&gamma_in_ptr); // Gamma_in (from R1c, raw ptr)
            builder.arg(acts.angle_cumsum.inner()); // Angles
            builder.arg(scratch.d_q.inner()); // dQ_mid from m3_dqkv
            builder.arg(scratch.d_k.inner()); // dK_mid from m3_dqkv
            builder.arg(scratch.d_beta.inner()); // dQK_dot from m3_dqkv
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(&ds_i);
            builder.arg(&na_i);
            builder.arg(&cs);
            unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_dqktheta B6 S2: {:?}", e))?;
        }

        // Copy dQ_pre/dK_pre to d_q/d_k (these are the final pre-bias grads)
        scratch
            .d_q
            .copy_from_raw(&scratch.d_c_pre_rope, &ctx.stream)?;
        scratch
            .d_k
            .copy_from_raw(&scratch.d_b_pre_rope, &ctx.stream)?;

        // S3: m3_ddt_dtrap -- dScale/dGamma -> dDT, dTrap
        // Grid: ceil(B*T*nh/256), Block: 256 -- one thread per (b,t,h)
        {
            let mut builder = ctx.stream.launch_builder(&m3k.m3_ddt_dtrap);
            builder.arg(scratch.d_gamma.inner_mut()); // dDT -> d_gamma temp [B*T*nh]
            builder.arg(scratch.d_trap_raw.inner_mut()); // dTrap_presig [B*T*nh]
            builder.arg(scratch.d_scale.inner()); // dScale
            builder.arg(scratch.d_gamma_par.inner()); // dGamma
            builder.arg(acts.dt.inner()); // DT (post-softplus)
            builder.arg(acts.trap.inner()); // Trap (post-sigmoid)
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            unsafe { builder.launch(grid_1d(bt * nh)) }
                .map_err(|e| format!("m3_ddt_dtrap B6 S3: {:?}", e))?;
        }

        // S5: m3_final_grads -- combine dDT + dADT + d_dt_angle -> d_dd_dt, d_dd_a
        // d_dt_angle is computed later in B5a, so we need to defer this.
        // Instead, store dDT and dADT for now, and run m3_final_grads after B5a.
        // We'll stash dDT in d_dd_dt and dADT in d_alpha for now.
        // d_trap_raw is already correct from S3.

        // d_D: m3_dqkv writes d_D[nh] via atomicAdd directly into lg.d_param.
        // No reduction step needed (all batches already accumulated).

        // Copy dQ/dK through to d_b_pre_rope/d_c_pre_rope for downstream
        // (these are the pre-bias gradients needed by B4)
        scratch
            .d_b_pre_rope
            .copy_from_raw(&scratch.d_k, &ctx.stream)?;
        scratch
            .d_c_pre_rope
            .copy_from_raw(&scratch.d_q, &ctx.stream)?;

        // Note: d_alpha now holds dADT, d_dd_dt holds dDT from trapezoidal path.
        // d_trap_raw holds dTrap. These are the correct gradient paths.
        // d_beta is dQK_dot (no longer needed after dqktheta consumed it).
        // d_gamma is discarded (was used as temp).
    }

    // B6-reduce: d_D -- parallel writes d_D directly via atomicAdd in m3_dqkv (no reduction needed)

    // B4a: RoPE backward -- m3_dqktheta already did inverse rotary and computed
    // d_b_pre_rope, d_c_pre_rope, d_angle_cumsum. No separate step needed.

    // =================================================================
    // B5a: angle_dt_bwd -- d_angle_cumsum -> d_angles_raw + d_dt_angle
    // =================================================================
    // Zero output buffers (angle_dt_bwd uses atomicAdd for d_angles_raw)
    if na > 0 {
        {
            let ne = (bt * na) as i32;
            let zero: f32 = 0.0;
            let mut builder = ctx.stream.launch_builder(&m3k.fill_scalar);
            builder.arg(scratch.d_angles_raw.inner_mut());
            builder.arg(&zero);
            builder.arg(&ne);
            unsafe { builder.launch(grid_1d(bt * na)) }
                .map_err(|e| format!("fill d_angles_raw B5a: {:?}", e))?;
        }
        {
            let ne = (bt * nh) as i32;
            let zero: f32 = 0.0;
            let mut builder = ctx.stream.launch_builder(&m3k.fill_scalar);
            builder.arg(scratch.d_dt_angle.inner_mut());
            builder.arg(&zero);
            builder.arg(&ne);
            unsafe { builder.launch(grid_1d(bt * nh)) }
                .map_err(|e| format!("fill d_dt_angle B5a: {:?}", e))?;
        }

        let t_i = t;
        let nh_i = nh as i32;
        let na_i = na as i32;
        let b_i = b;
        let mut builder = ctx.stream.launch_builder(&m3k.m3_angle_dt_bwd_seq);
        builder.arg(scratch.d_angles_raw.inner_mut()); // 1: d_angles_raw (atomicAdd)
        builder.arg(scratch.d_dt_angle.inner_mut()); // 2: d_dt_angle
        builder.arg(scratch.d_angle_cumsum.inner()); // 3: d_angle_cumsum
        builder.arg(acts.angles_raw.inner()); // 4: angles_raw (saved)
        builder.arg(acts.dt.inner()); // 5: dt (post-softplus)
        builder.arg(&b_i); // 6: B
        builder.arg(&t_i); // 7: T
        builder.arg(&nh_i); // 8: nh
        builder.arg(&na_i); // 9: n_angles
        let grid = cudarc::driver::LaunchConfig {
            grid_dim: (dims.batch as u32, (nh * na).div_ceil(256) as u32, 1),
            block_dim: (256.min((nh * na) as u32), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(grid) }.map_err(|e| format!("m3_angle_dt_bwd_seq B5a: {:?}", e))?;
    } else {
        // No angles: zero d_dt_angle
        let ne = (bt * nh) as i32;
        let zero: f32 = 0.0;
        let mut builder = ctx.stream.launch_builder(&m3k.fill_scalar);
        builder.arg(scratch.d_dt_angle.inner_mut());
        builder.arg(&zero);
        builder.arg(&ne);
        unsafe { builder.launch(grid_1d(bt * nh)) }
            .map_err(|e| format!("fill d_dt_angle no-angle B5a: {:?}", e))?;
    }

    // =================================================================
    // B5b: Gradient chain rules for dd_dt, dd_a, trap
    // =================================================================
    // m3_final_grads combines dDT + dADT + d_dt_angle
    // d_dd_dt currently holds dDT from m3_ddt_dtrap
    // d_alpha currently holds dADT from m3_dqkv
    // d_dt_angle holds angle contribution from B5a
    // d_trap_raw already has the correct pre-sigmoid trap gradient from S3
    if !dims.use_parallel_scan {
        // Sequential mode: use m3_abg_bwd which takes d_alpha/d_beta/d_gamma directly
        // and computes d_dd_dt, d_dd_a, d_trap_raw with full chain rule.
        // This replaces the manual dADT conversion + m3_final_grads.
        let n_total = (bt * nh) as i32;
        let nh_i = nh as i32;
        let dtb_ptr = lw.dt_bias.raw_ptr(&ctx.stream);
        let mut builder = ctx.stream.launch_builder(&m3k.m3_abg_bwd);
        builder.arg(scratch.d_dd_dt.inner_mut());
        builder.arg(scratch.d_dd_a.inner_mut());
        builder.arg(scratch.d_trap_raw.inner_mut());
        builder.arg(scratch.d_alpha.inner());
        builder.arg(scratch.d_beta.inner());
        builder.arg(scratch.d_gamma.inner());
        builder.arg(scratch.d_dt_angle.inner());
        builder.arg(acts.dt.inner());
        builder.arg(acts.a_val.inner());
        builder.arg(acts.alpha.inner());
        builder.arg(acts.dd_dt_raw.inner());
        builder.arg(acts.dd_a_raw.inner());
        builder.arg(acts.trap_raw.inner());
        builder.arg(&dtb_ptr);
        builder.arg(&dims.a_floor);
        builder.arg(&n_total);
        builder.arg(&nh_i);
        unsafe { builder.launch(grid_1d(bt * nh)) }
            .map_err(|e| format!("m3_abg_bwd seq B5b: {:?}", e))?;
    } else {
        // Parallel mode: m3_final_grads takes dADT + dDT intermediates
        let n_total = (bt * nh) as i32;
        let nh_i = nh as i32;
        let dtb_ptr = lw.dt_bias.raw_ptr(&ctx.stream);
        let mut builder = ctx.stream.launch_builder(&m3k.m3_final_grads);
        builder.arg(scratch.d_dd_dt.inner_mut());
        builder.arg(scratch.d_dd_a.inner_mut());
        builder.arg(scratch.d_alpha.inner()); // dADT from m3_dqkv
        builder.arg(scratch.d_gamma.inner()); // dDT from m3_ddt_dtrap (stored in d_gamma)
        builder.arg(scratch.d_dt_angle.inner()); // dDT_angle from angle_dt_bwd
        builder.arg(acts.a_val.inner()); // a_val (clamped)
        builder.arg(acts.dt.inner()); // DT (post-softplus)
        builder.arg(acts.dd_dt_raw.inner()); // dd_dt_raw_saved (NO bias)
        builder.arg(acts.dd_a_raw.inner()); // dd_a_raw_saved
        builder.arg(&dtb_ptr); // dt_bias[nh]
        builder.arg(&dims.a_floor); // a_floor
        builder.arg(&n_total); // N = B*T*nh
        builder.arg(&nh_i); // nh
        unsafe { builder.launch(grid_1d(bt * nh)) }
            .map_err(|e| format!("m3_final_grads B5b: {:?}", e))?;
    }

    // B5b-accumulate: dt_bias gradient = sum of d_dd_dt across B*T samples per head
    // d_dd_dt already contains d_dt_total * sp_deriv_dt, which is the same as d_dt_bias
    {
        let d_dtb_ptr = lg.dt_bias.ptr();
        let bt_i = bt as i32;
        let nh_i = nh as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        builder.arg(&d_dtb_ptr);
        builder.arg(scratch.d_dd_dt.inner());
        builder.arg(&bt_i);
        builder.arg(&nh_i);
        unsafe { builder.launch(grid_1d(nh)) }
            .map_err(|e| format!("colsum d_dt_bias B5b: {:?}", e))?;
    }

    // =================================================================
    // B4b: bc_bias_add_bwd -- reduce heads -> groups
    // =================================================================
    // Parallel: m3_dqktheta already atomicAdd'd bias grads. Only need head->group reduce.

    // Reduce d_b_pre_rope from [B*T*nh*ds] -> d_b_normed [B*T*ng*ds] (sum heads per group)
    {
        let n_i = bt as i32;
        let nh_i = nh as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.bc_bias_add_bwd);
        builder.arg(scratch.d_b_normed.inner_mut());
        builder.arg(scratch.d_b_pre_rope.inner());
        builder.arg(&n_i);
        builder.arg(&nh_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(bt * ng * ds)) }
            .map_err(|e| format!("bc_bias_add_bwd B B4b: {:?}", e))?;
    }
    // Same for C
    {
        let n_i = bt as i32;
        let nh_i = nh as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.bc_bias_add_bwd);
        builder.arg(scratch.d_c_normed.inner_mut());
        builder.arg(scratch.d_c_pre_rope.inner());
        builder.arg(&n_i);
        builder.arg(&nh_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(bt * ng * ds)) }
            .map_err(|e| format!("bc_bias_add_bwd C B4b: {:?}", e))?;
    }

    // =================================================================
    // B4c: BCNorm backward for B and C
    //   d_b_normed -> d_b_raw + d_b_norm_weight contribution
    //   d_c_normed -> d_c_raw + d_c_norm_weight contribution
    // =================================================================
    // BCNorm backward for B
    {
        let bn_ptr = lw.b_norm_weight.raw_ptr(&ctx.stream);
        let n_i = bt as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((bt * ng) as u32, 1, 1),
            block_dim: (ds as u32, 1, 1),
            shared_mem_bytes: ds as u32 * 4,
        };
        let mut builder = ctx.stream.launch_builder(&m3k.bcnorm_bwd);
        builder.arg(scratch.d_b_raw.inner_mut()); // d_B (output)
        builder.arg(scratch.d_b_norm_w.inner_mut()); // d_weight per-block
        builder.arg(scratch.d_b_normed.inner()); // d_out (upstream)
        builder.arg(acts.b_raw.inner()); // B_raw (saved input)
        builder.arg(acts.b_rms.inner()); // rms_val (saved)
        builder.arg(&bn_ptr); // weight
        builder.arg(&n_i); // N
        builder.arg(&ng_i); // ng
        builder.arg(&ds_i); // ds
        unsafe { builder.launch(cfg) }.map_err(|e| format!("bcnorm_bwd B B4c: {:?}", e))?;
    }
    // Reduce d_b_norm_w: sum [B*T*ng, ds] -> [ds] and accumulate into lg.b_norm_weight
    {
        let d_bnw_ptr = lg.b_norm_weight.ptr();
        let rows = (bt * ng) as i32;
        let ds_i = ds as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        builder.arg(&d_bnw_ptr);
        builder.arg(scratch.d_b_norm_w.inner());
        builder.arg(&rows);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(ds)) }
            .map_err(|e| format!("colsum d_b_norm_w B4c: {:?}", e))?;
    }

    // BCNorm backward for C
    {
        let cn_ptr = lw.c_norm_weight.raw_ptr(&ctx.stream);
        let n_i = bt as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((bt * ng) as u32, 1, 1),
            block_dim: (ds as u32, 1, 1),
            shared_mem_bytes: ds as u32 * 4,
        };
        let mut builder = ctx.stream.launch_builder(&m3k.bcnorm_bwd);
        builder.arg(scratch.d_c_raw.inner_mut()); // d_C (output)
        builder.arg(scratch.d_c_norm_w.inner_mut()); // d_weight per-block
        builder.arg(scratch.d_c_normed.inner()); // d_out (upstream)
        builder.arg(acts.c_raw.inner()); // C_raw (saved input)
        builder.arg(acts.c_rms.inner()); // rms_val (saved)
        builder.arg(&cn_ptr); // weight
        builder.arg(&n_i); // N
        builder.arg(&ng_i); // ng
        builder.arg(&ds_i); // ds
        unsafe { builder.launch(cfg) }.map_err(|e| format!("bcnorm_bwd C B4c: {:?}", e))?;
    }
    // Reduce d_c_norm_w: sum [B*T*ng, ds] -> [ds] and accumulate into lg.c_norm_weight
    {
        let d_cnw_ptr = lg.c_norm_weight.ptr();
        let rows = (bt * ng) as i32;
        let ds_i = ds as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        builder.arg(&d_cnw_ptr);
        builder.arg(scratch.d_c_norm_w.inner());
        builder.arg(&rows);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(ds)) }
            .map_err(|e| format!("colsum d_c_norm_w B4c: {:?}", e))?;
    }

    // =================================================================
    // B3: Assemble d_proj from 8 gradient components
    //   [d_z | d_x | d_b_raw | d_c_raw | d_dd_dt | d_dd_a | d_trap | d_angles]
    // =================================================================
    {
        let n_i = bt as i32;
        let di_i = di as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let nh_i = nh as i32;
        let na_i = na as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.m3_split_bwd);
        builder.arg(scratch.d_proj.inner_mut()); // 1: d_proj (output)
        builder.arg(scratch.d_z.inner()); // 2: d_z (from B7)
        builder.arg(scratch.d_x.inner()); // 3: d_x (from B6)
        builder.arg(scratch.d_b_raw.inner()); // 4: d_B_raw (from B4c)
        builder.arg(scratch.d_c_raw.inner()); // 5: d_C_raw (from B4c)
        builder.arg(scratch.d_dd_dt.inner()); // 6: d_dd_dt (from B5b)
        builder.arg(scratch.d_dd_a.inner()); // 7: d_dd_a (from B5b)
        builder.arg(scratch.d_trap_raw.inner()); // 8: d_trap (from B5b)
        builder.arg(scratch.d_angles_raw.inner()); // 9: d_angles (from B5a)
        builder.arg(&n_i); // 10: N
        builder.arg(&di_i); // 11: di
        builder.arg(&ng_i); // 12: ng
        builder.arg(&ds_i); // 13: ds
        builder.arg(&nh_i); // 14: nh
        builder.arg(&na_i); // 15: n_angles
        unsafe { builder.launch(grid_1d(bt * ip)) }
            .map_err(|e| format!("m3_split_bwd B3: {:?}", e))?;
    }

    // B2: in_proj backward: d_post_norm = d_proj @ in_proj_w^T, d_in_proj_w += post_norm^T @ d_proj
    gpu_sgemm_backward_grad_raw(
        ctx,
        &mut scratch.d_norm,
        (&lg.in_proj_w, None),
        &scratch.d_proj,
        &acts.post_norm,
        lw.in_proj_w.raw_ptr(&ctx.stream),
        (bt, dm, ip),
    )?;

    // B1: RMSNorm backward
    // rmsnorm_backward(dx, d_scale, dy, x, scale, rms_saved, batch, dim)
    {
        let nw_ptr = lw.norm_weight.raw_ptr(&ctx.stream);
        let d_nw_ptr = lg.norm_weight.ptr();
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_bwd);
        builder.arg(scratch.d_pre_norm.inner_mut()); // dx
        builder.arg(&d_nw_ptr); // d_scale
        builder.arg(scratch.d_norm.inner()); // dy
        builder.arg(acts.residual.inner()); // x (saved input)
        builder.arg(&nw_ptr); // scale
        builder.arg(acts.rms_vals.inner()); // rms_saved
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        unsafe { builder.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("rmsnorm_bwd m3 B1: {:?}", e))?;
    }

    // Residual pass-through: d_temporal += d_pre_norm
    {
        let ne = (bt * dm) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.vec_add_inplace);
        builder.arg(d_temporal.inner_mut());
        builder.arg(scratch.d_pre_norm.inner());
        builder.arg(&ne);
        unsafe { builder.launch(grid_1d(bt * dm)) }
            .map_err(|e| format!("vec_add d_temporal m3: {:?}", e))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// GPU Backward -- full backbone
// ---------------------------------------------------------------------------

/// Mamba-3 SISO full backbone backward.
pub fn gpu_backward_mamba3_backbone(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    d_temporal: &mut GpuBuffer, // [B*T*d_model] gradient from downstream
    acts: &GpuMamba3BackboneActs,
    mamba_w: &GpuMamba3Weights,
    grads: &GpuMamba3Grads,
    scratch: &mut GpuMamba3Scratch,
    dims: &GpuMamba3Dims,
) -> Result<(), String> {
    let bt = dims.bt();
    let dm = dims.d_model;

    // norm_f backward: dx goes into scratch.d_norm, then copy to d_temporal
    // rmsnorm_backward(dx, d_scale, dy, x, scale, rms_saved, batch, dim)
    {
        let nf_ptr = mamba_w.norm_f_weight.raw_ptr(&ctx.stream);
        let d_nf_ptr = grads.norm_f_weight.ptr();
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_bwd);
        builder.arg(scratch.d_norm.inner_mut()); // dx
        builder.arg(&d_nf_ptr); // d_scale
        builder.arg(d_temporal.inner()); // dy
        builder.arg(acts.norm_f_input.inner()); // x
        builder.arg(&nf_ptr); // scale
        builder.arg(acts.norm_f_rms.inner()); // rms_saved
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        unsafe { builder.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("rmsnorm_bwd norm_f m3: {:?}", e))?;
    }
    // Copy d_norm -> d_temporal for layer backward chain
    d_temporal.copy_from_raw(&scratch.d_norm, &ctx.stream)?;

    // Backward through layers (reverse order)
    for l in (0..dims.n_layers).rev() {
        gpu_backward_mamba3_layer(
            ctx,
            m3k,
            d_temporal,
            &acts.layers[l],
            &mamba_w.layers[l],
            &grads.layers[l],
            scratch,
            dims,
        )?;
    }

    // input_proj backward
    gpu_sgemm_backward_grad_raw(
        ctx,
        &mut scratch.d_input_proj_dx,
        (&grads.input_proj_w, Some(&grads.input_proj_b)),
        d_temporal,
        &acts.input_proj_inputs,
        mamba_w.input_proj_w.raw_ptr(&ctx.stream),
        (bt, dims.mamba_input_dim, dm),
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// GPU Forward -- target burn-in (forward only, no activations saved)
// ---------------------------------------------------------------------------

/// Mamba-3 target burn-in forward (no activations saved, used for Bellman target).
///
/// Output: `temporal` receives `[B * d_model]` — the LAST timestep's representation.
/// Internally uses `tgt.temporal_work` as `[B*T*d_model]` working buffer.
pub fn gpu_forward_mamba3_target_burnin(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    temporal: &mut GpuBuffer, // [B * d_model] — output (last timestep only)
    mamba_w: &GpuMamba3Weights,
    mamba_input: &GpuBuffer,
    tgt: &mut GpuMamba3TargetScratch,
    dims: &GpuMamba3Dims,
) -> Result<(), String> {
    let bt = dims.bt();
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let nh = dims.nheads;
    let hd = dims.headdim;
    let ng = dims.ngroups;
    let ip = dims.in_proj_dim;
    let na = dims.n_angles.max(1);
    let b = dims.batch as i32;
    let t = dims.seq_len as i32;
    let f32_sz = std::mem::size_of::<f32>() as u64;

    // Zero per-layer states before burn-in (cold zero state, matching M1 target)
    tgt.ssm_states.zero(&ctx.stream)?;
    tgt.k_states.zero(&ctx.stream)?;
    tgt.v_states.zero(&ctx.stream)?;
    tgt.angle_states.zero(&ctx.stream)?;

    // Input proj → temporal_work [B*T*d_model]
    gpu_sgemm_forward_raw(
        ctx,
        &mut tgt.temporal_work,
        mamba_input,
        mamba_w.input_proj_w.raw_ptr(&ctx.stream),
        Some(mamba_w.input_proj_b.raw_ptr(&ctx.stream)),
        (bt, dims.mamba_input_dim, dm),
    )?;

    for l in 0..dims.n_layers {
        let lw = &mamba_w.layers[l];

        // Residual
        tgt.residual
            .copy_from_raw(&tgt.temporal_work, &ctx.stream)?;

        // RMSNorm
        {
            let bt_i = bt as i32;
            let dm_i = dm as i32;
            let eps: f32 = 1e-5;
            let nw_ptr = lw.norm_weight.raw_ptr(&ctx.stream);
            let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_fwd);
            builder.arg(tgt.out_flat.inner_mut()); // reuse as temp
            builder.arg(tgt.rms_discard.inner_mut());
            builder.arg(tgt.residual.inner());
            builder.arg(&nw_ptr);
            builder.arg(&bt_i);
            builder.arg(&dm_i);
            builder.arg(&eps);
            unsafe { builder.launch(grid_norm(bt, dm)) }
                .map_err(|e| format!("rmsnorm_fwd m3 tgt L{l}: {:?}", e))?;
        }

        // in_proj
        gpu_sgemm_forward_raw(
            ctx,
            &mut tgt.proj_flat,
            &tgt.out_flat,
            lw.in_proj_w.raw_ptr(&ctx.stream),
            None,
            (bt, dm, ip),
        )?;

        // m3_split
        {
            let n_i = bt as i32;
            let di_i = di as i32;
            let ng_i = ng as i32;
            let ds_i = ds as i32;
            let nh_i = nh as i32;
            let na_i = na as i32;
            let db_ptr = lw.dt_bias.raw_ptr(&ctx.stream);
            let mut builder = ctx.stream.launch_builder(&m3k.m3_split);
            builder.arg(tgt.z.inner_mut());
            builder.arg(tgt.x.inner_mut());
            builder.arg(tgt.b_raw.inner_mut());
            builder.arg(tgt.c_raw.inner_mut());
            builder.arg(tgt.dt.inner_mut());
            builder.arg(tgt.a_val.inner_mut());
            builder.arg(tgt.trap.inner_mut());
            builder.arg(tgt.angles_raw.inner_mut());
            builder.arg(tgt.dd_dt_raw.inner_mut());
            builder.arg(tgt.dd_a_raw.inner_mut());
            builder.arg(tgt.trap_raw.inner_mut());
            builder.arg(tgt.proj_flat.inner());
            builder.arg(&db_ptr);
            builder.arg(&dims.a_floor);
            builder.arg(&n_i);
            builder.arg(&di_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            builder.arg(&nh_i);
            builder.arg(&na_i);
            unsafe { builder.launch(grid_1d(bt * ip)) }
                .map_err(|e| format!("m3_split tgt L{l}: {:?}", e))?;
        }

        // BCNorm B
        {
            let bn_ptr = lw.b_norm_weight.raw_ptr(&ctx.stream);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: ((bt * ng) as u32, 1, 1),
                block_dim: (ds as u32, 1, 1),
                shared_mem_bytes: ds as u32 * 4,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.bcnorm_fwd);
            builder.arg(tgt.b_normed.inner_mut());
            builder.arg(tgt.b_rms.inner_mut());
            builder.arg(tgt.b_raw.inner());
            builder.arg(&bn_ptr);
            let n_i = bt as i32;
            let ng_i = ng as i32;
            let ds_i = ds as i32;
            builder.arg(&n_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("bcnorm_fwd B tgt L{l}: {:?}", e))?;
        }

        // BCNorm C
        {
            let cn_ptr = lw.c_norm_weight.raw_ptr(&ctx.stream);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: ((bt * ng) as u32, 1, 1),
                block_dim: (ds as u32, 1, 1),
                shared_mem_bytes: ds as u32 * 4,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.bcnorm_fwd);
            builder.arg(tgt.c_normed.inner_mut());
            builder.arg(tgt.c_rms.inner_mut());
            builder.arg(tgt.c_raw.inner());
            builder.arg(&cn_ptr);
            let n_i = bt as i32;
            let ng_i = ng as i32;
            let ds_i = ds as i32;
            builder.arg(&n_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("bcnorm_fwd C tgt L{l}: {:?}", e))?;
        }

        // Bias add B, C
        {
            let bb_ptr = lw.b_bias.raw_ptr(&ctx.stream);
            let n_i = bt as i32;
            let nh_i = nh as i32;
            let ng_i = ng as i32;
            let ds_i = ds as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.bc_bias_add);
            builder.arg(tgt.b_biased.inner_mut());
            builder.arg(tgt.b_normed.inner());
            builder.arg(&bb_ptr);
            builder.arg(&n_i);
            builder.arg(&nh_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(grid_1d(bt * nh * ds)) }
                .map_err(|e| format!("bc_bias_add B tgt L{l}: {:?}", e))?;
        }
        {
            let cb_ptr = lw.c_bias.raw_ptr(&ctx.stream);
            let n_i = bt as i32;
            let nh_i = nh as i32;
            let ng_i = ng as i32;
            let ds_i = ds as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.bc_bias_add);
            builder.arg(tgt.c_biased.inner_mut());
            builder.arg(tgt.c_normed.inner());
            builder.arg(&cb_ptr);
            builder.arg(&n_i);
            builder.arg(&nh_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(grid_1d(bt * nh * ds)) }
                .map_err(|e| format!("bc_bias_add C tgt L{l}: {:?}", e))?;
        }

        // RoPE (copy if no angles)
        if dims.n_angles > 0 {
            // angle_dt sequential accumulation (B envs, T timesteps)
            {
                let a_off = dims.batch * l * nh * na;
                let angle_st = tgt.angle_states.raw_ptr(&ctx.stream) + a_off as u64 * f32_sz;
                let b_i = dims.batch as i32;
                let t_i = dims.seq_len as i32;
                let nh_i = nh as i32;
                let na_i = na as i32;
                let mut builder = ctx.stream.launch_builder(&m3k.m3_angle_dt_fwd_seq);
                builder.arg(tgt.angle_cumsum.inner_mut());
                builder.arg(&angle_st);
                builder.arg(tgt.angles_raw.inner());
                builder.arg(tgt.dt.inner());
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&nh_i);
                builder.arg(&na_i);
                let grid = cudarc::driver::LaunchConfig {
                    grid_dim: (dims.batch as u32, (nh * na).div_ceil(256) as u32, 1),
                    block_dim: (256.min((nh * na) as u32), 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { builder.launch(grid) }
                    .map_err(|e| format!("angle_dt_fwd_seq tgt L{l}: {:?}", e))?;
            }

            // RoPE B->K + C->Q simultaneously
            {
                let n_i = bt as i32;
                let nh_i = nh as i32;
                let ds_i = ds as i32;
                let na_i = na as i32;
                let mut builder = ctx.stream.launch_builder(&m3k.rope_fwd);
                builder.arg(tgt.k.inner_mut()); // 1: B_rotated
                builder.arg(tgt.q.inner_mut()); // 2: C_rotated
                builder.arg(tgt.b_biased.inner()); // 3: B_biased
                builder.arg(tgt.c_biased.inner()); // 4: C_biased
                builder.arg(tgt.angle_cumsum.inner()); // 5: angle_cumsum
                builder.arg(&n_i);
                builder.arg(&nh_i);
                builder.arg(&ds_i);
                builder.arg(&na_i);
                unsafe { builder.launch(grid_1d(bt * nh * ds)) }
                    .map_err(|e| format!("rope_fwd tgt L{l}: {:?}", e))?;
            }
        } else {
            tgt.k.copy_from_raw(&tgt.b_biased, &ctx.stream)?;
            tgt.q.copy_from_raw(&tgt.c_biased, &ctx.stream)?;
        }

        // Compute alpha/beta/gamma for target
        {
            let n_total = (bt * nh) as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.m3_compute_abg);
            builder.arg(tgt.alpha.inner_mut());
            builder.arg(tgt.beta.inner_mut());
            builder.arg(tgt.gamma.inner_mut());
            builder.arg(tgt.dt.inner());
            builder.arg(tgt.a_val.inner());
            builder.arg(tgt.trap.inner());
            builder.arg(&n_total);
            unsafe { builder.launch(grid_1d(bt * nh)) }
                .map_err(|e| format!("m3_compute_abg tgt L{l}: {:?}", e))?;
        }

        // SSM forward (nosave variant)
        // Kernel: ssm_state, k_state, v_state, y_out, x, k, q,
        //   alpha, beta, gamma, D, batch, T, nh, hd, ds
        {
            let ssm_off = dims.batch * l * nh * hd * ds;
            let k_off = dims.batch * l * nh * ds;
            let v_off = dims.batch * l * nh * hd;
            let ssm_ptr = tgt.ssm_states.raw_ptr(&ctx.stream) + ssm_off as u64 * f32_sz;
            let k_ptr = tgt.k_states.raw_ptr(&ctx.stream) + k_off as u64 * f32_sz;
            let v_ptr = tgt.v_states.raw_ptr(&ctx.stream) + v_off as u64 * f32_sz;
            let dp_ptr = lw.d_param.raw_ptr(&ctx.stream);
            let nh_i = nh as i32;
            let hd_i = hd as i32;
            let ds_i = ds as i32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dims.batch as u32, nh as u32, 1),
                block_dim: (hd as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.m3_burnin_fwd_nosave);
            builder.arg(&ssm_ptr); // 1: ssm_state
            builder.arg(&k_ptr); // 2: k_state
            builder.arg(&v_ptr); // 3: v_state
            builder.arg(tgt.y.inner_mut()); // 4: y_out
            builder.arg(tgt.x.inner()); // 5: x_flat
            builder.arg(tgt.k.inner()); // 6: k_flat
            builder.arg(tgt.q.inner()); // 7: q_flat
            builder.arg(tgt.alpha.inner()); // 8: alpha_flat
            builder.arg(tgt.beta.inner()); // 9: beta_flat
            builder.arg(tgt.gamma.inner()); // 10: gamma_flat
            builder.arg(&dp_ptr); // 11: D
            builder.arg(&b); // 12: batch
            builder.arg(&t); // 13: T
            builder.arg(&nh_i); // 14: nh
            builder.arg(&hd_i); // 15: hd
            builder.arg(&ds_i); // 16: ds
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("m3_burnin_fwd_nosave tgt L{l}: {:?}", e))?;
        }

        // Output gating (match online forward F7 branch)
        if dims.is_outproj_norm {
            assert!(
                di <= 1024,
                "d_inner ({di}) exceeds rmsnorm_gated shared memory limit"
            );
            let nw_ptr = lw.norm_gate_weight.raw_ptr(&ctx.stream);
            let bt_i = bt as i32;
            let di_i = di as i32;
            let hd_i = dims.headdim as i32;
            let grid = cudarc::driver::LaunchConfig {
                grid_dim: (bt as u32, 1, 1),
                block_dim: (di as u32, 1, 1),
                shared_mem_bytes: (di * std::mem::size_of::<f32>()) as u32,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_gated_fwd);
            builder.arg(tgt.gated.inner_mut());
            builder.arg(tgt.rms_discard.inner_mut());
            builder.arg(tgt.y.inner());
            builder.arg(tgt.z.inner());
            builder.arg(&nw_ptr);
            builder.arg(&bt_i);
            builder.arg(&di_i);
            builder.arg(&hd_i);
            unsafe { builder.launch(grid) }
                .map_err(|e| format!("rmsnorm_gated_fwd m3 tgt L{l}: {:?}", e))?;
        } else {
            let n = (bt * di) as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.silu_gate_fwd);
            builder.arg(tgt.gated.inner_mut());
            builder.arg(tgt.y.inner());
            builder.arg(tgt.z.inner());
            builder.arg(&n);
            unsafe { builder.launch(grid_1d(bt * di)) }
                .map_err(|e| format!("silu_gate_fwd m3 tgt L{l}: {:?}", e))?;
        }

        // out_proj + residual
        gpu_sgemm_forward_raw(
            ctx,
            &mut tgt.out_flat,
            &tgt.gated,
            lw.out_proj_w.raw_ptr(&ctx.stream),
            None,
            (bt, di, dm),
        )?;
        {
            let ne = (bt * dm) as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.residual_add);
            builder.arg(tgt.temporal_work.inner_mut());
            builder.arg(tgt.out_flat.inner());
            builder.arg(tgt.residual.inner());
            builder.arg(&ne);
            unsafe { builder.launch(grid_1d(bt * dm)) }
                .map_err(|e| format!("residual_add m3 tgt L{l}: {:?}", e))?;
        }
    }

    // Final RMSNorm (norm_f) — output still in temporal_work [B*T*dm]
    {
        tgt.residual
            .copy_from_raw(&tgt.temporal_work, &ctx.stream)?;
        let nf_ptr = mamba_w.norm_f_weight.raw_ptr(&ctx.stream);
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let eps: f32 = 1e-5;
        let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_fwd);
        builder.arg(tgt.temporal_work.inner_mut());
        builder.arg(tgt.rms_discard.inner_mut());
        builder.arg(tgt.residual.inner());
        builder.arg(&nf_ptr);
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        builder.arg(&eps);
        unsafe { builder.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("rmsnorm_fwd norm_f m3 tgt: {:?}", e))?;
    }

    // Gather last timestep: temporal_work[B*T*dm] → temporal[B*dm]
    {
        let b_i = dims.batch as i32;
        let t_i = dims.seq_len as i32;
        let dm_i = dm as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.gather_last_timestep);
        builder.arg(temporal.inner_mut());
        builder.arg(tgt.temporal_work.inner());
        builder.arg(&b_i);
        builder.arg(&t_i);
        builder.arg(&dm_i);
        unsafe { builder.launch(grid_1d(dims.batch * dm)) }
            .map_err(|e| format!("gather_last_timestep m3 tgt: {:?}", e))?;
    }

    Ok(())
}
