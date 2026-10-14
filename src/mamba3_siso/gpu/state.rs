//! Mamba-3 SISO shared state types: dimensions, saved activations,
//! reusable scratch buffers, and target-network scratch.
//!
//! Split from the former 2313-line `mamba3_gpu.rs`. The
//! structs and constructors live here; `forward.rs` and `backward.rs`
//! contain the launch-orchestration code.

use super::kernels::Mamba3Kernels;
use crate::gpu::buffers::{GpuBuffer, GpuByteBuffer};
use crate::gpu::context::GpuCtx;
use std::sync::Arc;

pub(crate) type CUptr = cudarc::driver::sys::CUdeviceptr;

/// The four persistent Mamba-3 state buffers (SSM + K + V + RoPE angle),
/// grouped so call sites pass one bundle instead of four `&mut GpuBuffer`.
pub struct GpuMamba3StateBufs<'a> {
    /// SSM hidden state `[B * n_layers * nh * hd * ds]`.
    pub ssm: &'a mut GpuBuffer,
    /// K state `[B * n_layers * nh * ds]`.
    pub k: &'a mut GpuBuffer,
    /// V state `[B * n_layers * nh * hd]`.
    pub v: &'a mut GpuBuffer,
    /// RoPE angle accumulator `[B * n_layers * nh * n_angles]`.
    pub angle: &'a mut GpuBuffer,
}

impl GpuMamba3StateBufs<'_> {
    /// Reborrow the bundle so it can be handed to a callee without giving
    /// up the caller's exclusive borrows.
    pub fn reborrow(&mut self) -> GpuMamba3StateBufs<'_> {
        GpuMamba3StateBufs {
            ssm: self.ssm,
            k: self.k,
            v: self.v,
            angle: self.angle,
        }
    }
}

/// Execution context shared by every Mamba-3 GPU launch: CUDA context
/// (stream + cuBLAS), compiled M3 kernel handles, and launch dimensions.
#[derive(Clone, Copy)]
pub struct M3Exec<'a> {
    pub ctx: &'a GpuCtx,
    pub kernels: &'a Mamba3Kernels,
    pub dims: &'a GpuMamba3Dims,
    /// The sequences packed into the rows (see [`crate::varlen`]), i32
    /// `[B, T]` on the device; 0 for one sequence a row.
    pub seq_idx: CUptr,
}

impl M3Exec<'_> {
    /// The SSM kernels keep `d_state` floats per thread up to the capacity
    /// they were compiled with, and past it every one of them returns
    /// without writing. A pair that does not fit is refused before any
    /// launch.
    pub fn check_state_cap(&self) -> Result<(), String> {
        if self.dims.d_state > self.kernels.state_cap {
            return Err(format!(
                "d_state {} exceeds the state capacity {} these Mamba-3 kernels were \
                 compiled for (Mamba3Kernels::compile_with_state_cap)",
                self.dims.d_state, self.kernels.state_cap
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

/// Collected Mamba-3 dimensions for GPU launches.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Epsilon of the block norms (`Mamba3Config::rms_norm_eps`); it rides
    /// checkpoint metadata: a different eps is a different model.
    pub rms_norm_eps: f32,
    /// Epsilon of the mixer's B/C norms and output-gate norm
    /// (`Mamba3Config::ssm_norm_eps`).
    pub ssm_norm_eps: f32,
    /// The kind of the block norms (`Mamba3Config::block_norm`).
    pub block_norm: crate::config::BlockNorm,
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

    /// Refuse dimensions past the 32-bit range the SISO kernels take their
    /// scalar arguments in: the batch, the window, the rows they make and
    /// the row widths. Element counts and offsets are 64-bit in the kernels'
    /// 64-bit builds, which [`Self::index_width`] selects, so a tensor's size
    /// is bounded by memory alone.
    pub fn check_int_args(&self) -> Result<(), String> {
        let widths = crate::mamba3_siso::route::IntWidths {
            batch: self.batch,
            seq_len: self.seq_len,
            ngroups: self.ngroups,
            d_model: self.d_model,
            input_dim: self.mamba_input_dim,
            in_proj_dim: self.in_proj_dim,
            nheads: self.nheads,
            d_state: self.d_state,
            d_inner: self.d_inner,
        };
        crate::gpu::launch::check_int_args(&widths.args())
    }

    /// The index width of every SISO launch at these dimensions: the int
    /// builds while every tensor the kernels index fits the 32-bit range,
    /// the 64-bit builds past it. The sequential tape counts only on the
    /// sequential path: the chunked path never allocates it (a one-element
    /// sentinel stands in) and never indexes it.
    pub fn index_width(&self) -> crate::gpu::launch::IndexWidth {
        let bt = self.bt();
        let nh = self.nheads;
        let ds = self.d_state;
        let sequential_tape = if self.use_parallel_scan {
            0
        } else {
            self.batch * (self.seq_len + 1) * self.d_inner * ds
        };
        crate::gpu::launch::IndexWidth::covering(&[
            sequential_tape,
            // Chunk states [B, n_chunks, nh, hd, ds] and decay sums
            // [B, n_chunks, nh, chunk_size].
            self.batch * self.n_chunks() * self.d_inner * ds,
            self.batch * self.n_chunks() * nh * CHUNK_SIZE,
            // Rows of the projection, the inner channels and the model.
            bt * self.in_proj_dim,
            bt * self.d_inner,
            bt * self.d_model,
            // B and C lanes per head and per group, and the angles.
            bt * nh * ds,
            bt * self.ngroups * ds,
            bt * nh * self.n_angles,
            // The angle gradients' partials, both halves in one buffer.
            2 * bt * nh * self.n_angles,
        ])
    }
}

/// Chunk size for the Mamba-3 SISO chunked SSD parallel scan, the route's
/// `CHUNKED_CHUNK_SIZE` so its shared-memory bound and the kernels agree.
/// Every site that allocates `[B * n_chunks * nh * chunk_size]` scratch
/// must reference this constant rather than re-spelling `64`.
pub const CHUNK_SIZE: usize = crate::mamba3_siso::route::CHUNKED_CHUNK_SIZE;

#[cfg(test)]
mod index_width_tests {
    use super::GpuMamba3Dims;
    use crate::gpu::launch::IndexWidth;

    fn dims(batch: usize, seq_len: usize) -> GpuMamba3Dims {
        GpuMamba3Dims {
            batch,
            d_model: 384,
            d_inner: 768,
            d_state: 64,
            nheads: 48,
            headdim: 16,
            ngroups: 1,
            in_proj_dim: 1700,
            seq_len,
            mamba_input_dim: 384,
            n_layers: 24,
            n_angles: 16,
            a_floor: 1e-4,
            is_outproj_norm: false,
            rms_norm_eps: 1e-5,
            ssm_norm_eps: 1e-5,
            block_norm: crate::config::BlockNorm::Rms,
            use_parallel_scan: true,
        }
    }

    /// Realistic shapes take the int builds; a window whose tensors pass the
    /// 32-bit range takes the 64-bit builds instead of being refused, and
    /// only the scalar arguments the kernels take as `int` are bounded.
    #[test]
    fn a_long_window_takes_the_64_bit_builds() {
        let short = dims(2, 4621);
        short.check_int_args().unwrap();
        assert_eq!(short.index_width(), IndexWidth::I32);
        let long = dims(2, 25_000_000);
        long.check_int_args().unwrap();
        assert_eq!(long.index_width(), IndexWidth::I64);
        let err = dims(2, 1 << 30).check_int_args().unwrap_err();
        assert!(err.starts_with("batch*seq_len "), "{err}");
    }

    /// A window whose only range past the 32-bit one is the sequential tape
    /// keeps the int builds on the chunked path, which has no such tape, and
    /// takes the 64-bit builds on the sequential path, which does.
    #[test]
    fn the_sequential_tape_counts_only_on_the_sequential_path() {
        let mut d = GpuMamba3Dims {
            d_inner: 1536,
            d_state: 128,
            nheads: 24,
            headdim: 64,
            in_proj_dim: 3436,
            n_angles: 4,
            ..dims(1, 12_000)
        };
        assert_eq!(d.index_width(), IndexWidth::I32);
        d.use_parallel_scan = false;
        assert_eq!(d.index_width(), IndexWidth::I64);
    }
}

// ---------------------------------------------------------------------------
// Saved activations (per layer)
// ---------------------------------------------------------------------------

/// Per-layer saved activations for backward.
pub struct GpuMamba3LayerActs {
    // F1: the block norm
    pub residual: GpuBuffer, // [B*T*d_model]
    /// The block norm's statistics, `BlockNorm::stats_per_row` per row.
    pub rms_vals: GpuBuffer, // [B*T*stats]
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
    /// The carried K and V the window entered with, which its first step's
    /// trapezoidal fold read and the state pass then overwrote: the fold's
    /// gradient needs them. `[B*nh*ds]` and `[B*nh*hd]` on the chunked
    /// path; the sequential tapes hold them there (one-element sentinels).
    pub k_enter: GpuBuffer,
    pub v_enter: GpuBuffer,
    // F7: RMSNormGated
    pub gated_rms_vals: GpuBuffer, // [B*T*nheads] — one rstd per group (group_size=headdim)
    pub gated: GpuBuffer,          // [B*T*d_inner]
}

/// One layer's gated MLP saves for the backward.
pub struct GpuMamba3MlpActs {
    pub x_in: GpuBuffer, // [B*T*d_model] the rows entering the MLP
    pub post: GpuBuffer, // [B*T*d_model] the rows after its norm
    pub rms: GpuBuffer,  // [B*T*stats] its norm's statistics
    pub up: GpuBuffer,   // [B*T*2*hidden] value columns, then gate
    pub act: GpuBuffer,  // [B*T*hidden]
}

/// The gated MLP's saves for every layer, for a model with one.
pub struct GpuMamba3MlpBufs {
    pub hidden: usize,
    pub layers: Vec<GpuMamba3MlpActs>,
}

impl GpuMamba3MlpBufs {
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMamba3Dims,
        hidden: usize,
    ) -> Result<Self, String> {
        let (bt, dm) = (dims.bt(), dims.d_model);
        let stats = bt * dims.block_norm.stats_per_row();
        let z = |n: usize| GpuBuffer::zeros(stream, n);
        Ok(Self {
            hidden,
            layers: (0..dims.n_layers)
                .map(|_| {
                    Ok(GpuMamba3MlpActs {
                        x_in: z(bt * dm)?,
                        post: z(bt * dm)?,
                        rms: z(stats)?,
                        up: z(bt * 2 * hidden)?,
                        act: z(bt * hidden)?,
                    })
                })
                .collect::<Result<_, String>>()?,
        })
    }
}

/// Backbone activations (all layers + input proj + norm_f).
pub struct GpuMamba3BackboneActs {
    pub input_proj_inputs: GpuBuffer, // [B*T*mamba_input_dim]
    pub layers: Vec<GpuMamba3LayerActs>,
    pub norm_f_input: GpuBuffer, // [B*T*d_model]
    pub norm_f_rms: GpuBuffer,   // [B*T*stats]
    /// The gated MLP's buffers; `None` for a model without one.
    pub mlp: Option<GpuMamba3MlpBufs>,
}

// ---------------------------------------------------------------------------
// Scratch buffers
// ---------------------------------------------------------------------------

/// Reusable scratch for Mamba-3 forward + backward.
pub struct GpuMamba3Scratch {
    // Forward
    pub proj_flat: GpuBuffer, // [B*T*in_proj_dim]
    pub out_flat: GpuBuffer,  // [B*T*d_model]
    /// fp64 staging for the chunk-parallel angle accumulation:
    /// per-chunk raw delta sums and entering carries,
    /// `[B * n_chunks * nh * n_angles]` doubles each.
    pub angle_chunk_sums: GpuByteBuffer,
    pub angle_chunk_carries: GpuByteBuffer,

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
    pub da_cs_sum: GpuBuffer, // [B * nheads * n_chunks] -- per-chunk decay sum
    /// Per-chunk B terms of the reverse d_state recurrence
    /// (m3_dqkv_state_terms output). [B * n_chunks * nh * hd * ds]
    pub dstate_terms: GpuBuffer,
    /// Entering d_state per chunk (m3_dstate_passing_bwd output).
    /// [B * n_chunks * nh * hd * ds]
    pub dstate_enter: GpuBuffer,
    /// The gradient of the state the window entered with
    /// (m3_dstate_passing_bwd's last fold). [B * nh * hd * ds]
    pub dstate_seed: GpuBuffer,
    /// The gradient of the entering trapezoidal fold's coefficient
    /// (m3_enter_fold_bwd), the scale gradient of the step before the
    /// window's first. [B * nh]
    pub d_scale_enter: GpuBuffer,
    /// The chain each chunk's saved state adds to dADT, for the tiled
    /// backward. [B * n_chunks * nh]
    pub dm_chain: GpuBuffer,
    // Parallel backward intermediate buffers (Steps 2-6)
    pub d_scale: GpuBuffer,     // [B * T * nh] -- dScale from dqktheta
    pub d_gamma_par: GpuBuffer, // [B * T * nh] -- dGamma from dqktheta

    // -- Rule-B axis-0 reduction partials (determinism fix) --
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
    /// The gated MLP backward's gradients of its activation `[B*T*hidden]`,
    /// up projection `[B*T*2*hidden]` and normed rows `[B*T*d_model]`;
    /// empty unless [`Self::with_mlp`] sized them.
    pub mlp_d_act: GpuBuffer,
    pub mlp_d_up: GpuBuffer,
    pub mlp_d_post: GpuBuffer,
}

/// Layer pointers for persistent state (ssm + k_state + v_state + angle_state).
#[derive(Clone, Copy)]
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
    /// Statistics nobody reads back: the gated norm's per-group rstd and
    /// the block norms', `[B*T*max(nheads, stats)]`.
    pub rms_discard: GpuBuffer,
    /// The gated MLP's up projection `[B*T*2*hidden]` and activation
    /// `[B*T*hidden]`, empty unless [`Self::with_mlp`] sized them.
    pub mlp_up: GpuBuffer,
    pub mlp_act: GpuBuffer,
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
            rms_vals: GpuBuffer::zeros(stream, bt * dims.block_norm.stats_per_row())?,
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
            // The three sequential-scan tapes are consumed only by the
            // sequential forward/backward pair. On the chunked parallel
            // path they are never read or written, so they shrink to a
            // 1-element sentinel there — at large d_state the h tape
            // alone is B*(T+1)*d_inner*d_state floats of dead VRAM.
            h_saved: GpuBuffer::zeros(
                stream,
                if dims.use_parallel_scan {
                    1
                } else {
                    b * (t + 1) * di * ds
                },
            )?,
            k_prev_saved: GpuBuffer::zeros(
                stream,
                if dims.use_parallel_scan {
                    1
                } else {
                    bt * nh * ds
                },
            )?,
            v_prev_saved: GpuBuffer::zeros(
                stream,
                if dims.use_parallel_scan {
                    1
                } else {
                    bt * nh * hd
                },
            )?,
            k_enter: GpuBuffer::zeros(
                stream,
                if dims.use_parallel_scan {
                    b * nh * ds
                } else {
                    1
                },
            )?,
            v_enter: GpuBuffer::zeros(
                stream,
                if dims.use_parallel_scan {
                    b * nh * hd
                } else {
                    1
                },
            )?,
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
            layers: (0..dims.n_layers)
                .map(|_| GpuMamba3LayerActs::new(stream, bt, dims))
                .collect::<Result<_, _>>()?,
            norm_f_input: GpuBuffer::zeros(stream, bt * dims.d_model)?,
            norm_f_rms: GpuBuffer::zeros(stream, bt * dims.block_norm.stats_per_row())?,
            mlp: None,
        })
    }

    /// The same activations with the gated MLP's buffers, for a model whose
    /// MLP is `hidden` wide (0 leaves them out).
    pub fn with_mlp(
        mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMamba3Dims,
        hidden: usize,
    ) -> Result<Self, String> {
        self.mlp = if hidden > 0 {
            Some(GpuMamba3MlpBufs::new(stream, dims, hidden)?)
        } else {
            None
        };
        Ok(self)
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
        let angle_stage_bytes =
            b * dims.n_chunks() * nh * dims.n_angles.max(1) * std::mem::size_of::<f64>();
        Ok(Self {
            proj_flat: GpuBuffer::zeros(stream, bt * ip)?,
            out_flat: GpuBuffer::zeros(stream, bt * dm)?,
            angle_chunk_sums: GpuByteBuffer::zeros(stream, angle_stage_bytes)?,
            angle_chunk_carries: GpuByteBuffer::zeros(stream, angle_stage_bytes)?,
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
            da_cs_sum: {
                let nc = dims.n_chunks();
                GpuBuffer::zeros(stream, b * nh * nc)?
            },
            dstate_terms: {
                let nc = dims.n_chunks();
                GpuBuffer::zeros(stream, b * nc * nh * hd * ds)?
            },
            dstate_enter: {
                let nc = dims.n_chunks();
                GpuBuffer::zeros(stream, b * nc * nh * hd * ds)?
            },
            dstate_seed: GpuBuffer::zeros(stream, b * nh * hd * ds)?,
            d_scale_enter: GpuBuffer::zeros(stream, b * nh)?,
            dm_chain: GpuBuffer::zeros(stream, b * dims.n_chunks() * nh)?,
            d_scale: GpuBuffer::zeros(stream, bt * nh)?,
            d_gamma_par: GpuBuffer::zeros(stream, bt * nh)?,
            // Rule-B axis-0 partials — sized to fit largest M3 consumer.
            axis0_partials: {
                let na = dims.n_angles.max(1);
                let angle_dt_sz = 2 * nh * bt * na;
                let rmsnorm_sz = bt * dm;
                let d_d_sz = b * dims.n_chunks() * nh;
                GpuBuffer::zeros(stream, angle_dt_sz.max(rmsnorm_sz).max(d_d_sz))?
            },
            mlp_d_act: GpuBuffer::zeros(stream, 0)?,
            mlp_d_up: GpuBuffer::zeros(stream, 0)?,
            mlp_d_post: GpuBuffer::zeros(stream, 0)?,
        })
    }

    /// The same scratch with the gated MLP backward's buffers, for a model
    /// whose MLP is `hidden` wide.
    pub fn with_mlp(
        mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMamba3Dims,
        hidden: usize,
    ) -> Result<Self, String> {
        let bt = dims.bt();
        self.mlp_d_act = GpuBuffer::zeros(stream, bt * hidden)?;
        self.mlp_d_up = GpuBuffer::zeros(stream, bt * 2 * hidden)?;
        self.mlp_d_post = GpuBuffer::zeros(stream, bt * dims.d_model)?;
        Ok(self)
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
            rms_discard: GpuBuffer::zeros(stream, bt * nh.max(dims.block_norm.stats_per_row()))?,
            mlp_up: GpuBuffer::zeros(stream, 0)?,
            mlp_act: GpuBuffer::zeros(stream, 0)?,
            ssm_states: GpuBuffer::zeros(stream, b * nl * nh * hd * ds)?,
            k_states: GpuBuffer::zeros(stream, b * nl * nh * ds)?,
            v_states: GpuBuffer::zeros(stream, b * nl * nh * hd)?,
            angle_states: GpuBuffer::zeros(stream, b * nl * nh * na.max(1))?,
        })
    }

    /// The same scratch with the gated MLP's buffers, for a model whose MLP
    /// is `hidden` wide.
    pub fn with_mlp(
        mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMamba3Dims,
        hidden: usize,
    ) -> Result<Self, String> {
        self.mlp_up = GpuBuffer::zeros(stream, dims.bt() * 2 * hidden)?;
        self.mlp_act = GpuBuffer::zeros(stream, dims.bt() * hidden)?;
        Ok(self)
    }
}

/// Typed twins for the prefill's per-layer activation path (the bf16/f16
/// serve lane). Mirrors the trainer's `GpuMamba3LayerMixedActs` dtype map
/// exactly: what the trainer keeps typed is typed here, what it keeps f32
/// (dt/a_val/trap/angles/rms/residual/states and everything from `norm_f`
/// on) stays in the f32 `GpuMamba3TargetScratch`, untouched. A SEPARATE
/// struct on purpose: `GpuMamba3TargetScratch` is shared with the no-save
/// f32 forward and must not grow dtype branches.
pub struct Mamba3PrefillTypedScratch {
    pub dtype: crate::gpu::dtype::WeightDtype,
    /// Typed staging of `mamba_input` for the non-identity input_proj
    /// (cast_f32_to_* before the typed GEMM).
    pub input_cast: crate::gpu::buffers::DtypedBuf, // [B*T*input_dim]
    pub post_norm: crate::gpu::buffers::DtypedBuf, // [B*T*d_model]
    pub proj_flat: crate::gpu::buffers::DtypedBuf, // [B*T*in_proj_dim]
    pub x: crate::gpu::buffers::DtypedBuf,         // [B*T*d_inner]
    pub b_normed: crate::gpu::buffers::DtypedBuf,  // [B*T*ng*ds]
    pub c_normed: crate::gpu::buffers::DtypedBuf,  // [B*T*ng*ds]
    pub k: crate::gpu::buffers::DtypedBuf,         // [B*T*nh*ds]
    pub q: crate::gpu::buffers::DtypedBuf,         // [B*T*nh*ds]
    pub k_scaled: crate::gpu::buffers::DtypedBuf,  // [B*T*nh*ds]
    pub y: crate::gpu::buffers::DtypedBuf,         // [B*T*d_inner]
    pub gated: crate::gpu::buffers::DtypedBuf,     // [B*T*d_inner]
    pub out_flat: crate::gpu::buffers::DtypedBuf,  // [B*T*d_model]
}

impl Mamba3PrefillTypedScratch {
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMamba3Dims,
        dtype: crate::gpu::dtype::WeightDtype,
    ) -> Result<Self, String> {
        use crate::gpu::buffers::DtypedBuf;
        let bt = dims.bt();
        let dm = dims.d_model;
        let di = dims.d_inner;
        let ds = dims.d_state;
        let nh = dims.nheads;
        let ng = dims.ngroups;
        let ip = dims.in_proj_dim;
        Ok(Self {
            dtype,
            input_cast: DtypedBuf::zeros(stream, bt * dims.mamba_input_dim, dtype)?,
            post_norm: DtypedBuf::zeros(stream, bt * dm, dtype)?,
            proj_flat: DtypedBuf::zeros(stream, bt * ip, dtype)?,
            x: DtypedBuf::zeros(stream, bt * di, dtype)?,
            b_normed: DtypedBuf::zeros(stream, bt * ng * ds, dtype)?,
            c_normed: DtypedBuf::zeros(stream, bt * ng * ds, dtype)?,
            k: DtypedBuf::zeros(stream, bt * nh * ds, dtype)?,
            q: DtypedBuf::zeros(stream, bt * nh * ds, dtype)?,
            k_scaled: DtypedBuf::zeros(stream, bt * nh * ds, dtype)?,
            y: DtypedBuf::zeros(stream, bt * di, dtype)?,
            gated: DtypedBuf::zeros(stream, bt * di, dtype)?,
            out_flat: DtypedBuf::zeros(stream, bt * dm, dtype)?,
        })
    }
}
