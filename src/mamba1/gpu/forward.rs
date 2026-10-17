//! GPU Mamba SSM forward/backward — mirrors CPU mamba/optimized.rs.
//!
//! Key difference from CPU: batches across ALL B samples simultaneously.
//! - SGEMM calls use batch=B*T (all samples, all timesteps)
//! - SSM recurrence + conv1d: sequential across T, parallel across B*d_inner
//!
//! ## Forward pipeline (per layer):
//! F1: RmsNorm → F2: in_proj SGEMM → F3: split+SiLU(gate)
//! → F4a: conv1d burnin+SiLU → F4b: x_proj SGEMM → F4c: dt_proj+softplus
//! → F4d: SSM burnin forward → F4e: gating → F5: out_proj SGEMM → F6: residual
//!
//! ## Backward pipeline (per layer):
//! B1: out_proj bwd → B2: gating bwd → B3: SSM BPTT + reductions
//! → B4: softplus bwd + dt_proj bwd → B5: x_proj bwd → B6: SiLU+conv1d bwd
//! → B7: in_proj bwd → B8: RmsNorm bwd + residual
//!
//! Source: CPU reference in train/forward.rs

use super::backward::GpuMambaTargetScratch;
use super::bc_gather::BcGather;
use super::mixer_norms::launch_bcdt_norm;
use super::weights::{GpuMambaTrainLayerWeights, GpuMambaTrainWeights};
use crate::bidi::{opens_pair, reads_reversed};
use crate::gpu::bidi::{WindowRows, reverse_rows, zero_bytes};
use crate::gpu::blas::gpu_gemm_bi_forward_raw;
use crate::gpu::block_norm;
use crate::gpu::buffers::GpuBuffer;
use crate::gpu::context::GpuCtx;
use crate::gpu::dtype::WeightDtype;
use crate::gpu::launch::{
    grid_1d_cover, grid_1d_wide, grid_parallel_scan, scan_fwd_index_width,
    scan_fwd_nosave_index_width, seq_scan_index_width, y_runs, yz_runs,
};
use cudarc::driver::PushKernelArg;
use std::sync::Arc;

/// Sequence length threshold for switching from sequential SSM to parallel prefix scan.
///
/// The sequential kernel (one thread per (b,d) pair, looping over T) wins only on
/// the shortest sequences; the parallel prefix scan kernel walks O(T / 128) steps per
/// thread and, with its chunked backward, is the faster training route from T=128 on.
/// Measured on RTX 6000 Ada, whole training step with graph replay: at T=64 (d128,
/// B16) sequential 1.89 ms against parallel 2.04; at T=128 (d256, B16) 7.62 against
/// 7.58; at T=256 (d768, B8) 18.8 against 13.6 in bf16 and 30.6 against 24.6 in f32;
/// at T=256 (d1536, B4) 12.2 against 9.6. The threshold sits at 64.
///
/// The parallel scan launches (batch * d_inner) blocks of 128 threads with shared memory,
/// vs the sequential kernel which launches ceil(batch * d_inner / 256) blocks of 256 threads.
pub const PARALLEL_SCAN_THRESHOLD: usize = crate::config::ScanMode::PARALLEL_SCAN_THRESHOLD;

// ---------------------------------------------------------------------------
// GPU Mamba dimensions (mirrors CPU MambaDims)
// ---------------------------------------------------------------------------

/// Collected Mamba dimensions for GPU functions.
///
/// Mirrors CPU `MambaDims` from ops/dims.rs. Adds `batch` for GPU-parallel indexing.
#[derive(Debug, Clone, Copy)]
pub struct GpuMambaDims {
    /// Number of samples in the batch.
    pub batch: usize,
    /// Model dimension (features projected to this size).
    pub d_model: usize,
    /// Expanded inner dimension (`expand * d_model`).
    pub d_inner: usize,
    /// SSM state dimension (memory capacity per channel).
    pub d_state: usize,
    /// Convolution kernel width.
    pub d_conv: usize,
    /// Delta projection bottleneck rank (`ceil(d_model / 16)`).
    pub dt_rank: usize,
    /// x_proj output size: `dt_rank + 2 * d_state`.
    pub xdbl_dim: usize,
    /// Sequence length (timesteps per sample).
    pub seq_len: usize,
    /// Dimension of external input fed into input_proj.
    pub mamba_input_dim: usize,
    /// Number of stacked Mamba layers.
    pub n_layers: usize,
    /// GPU SSM scan dispatch: Sequential and Parallel as named at every
    /// d_state, Auto as [`crate::config::ScanMode::use_parallel`] resolves
    /// it.
    pub scan_mode: crate::config::ScanMode,
    /// Epsilon of the block norms (layer norms + norm_f). 1e-5 unless the
    /// checkpoint specifies otherwise (FalconMamba: 1e-6).
    pub rms_norm_eps: f32,
    /// The kind of the block norms (see [`crate::MambaConfig::block_norm`]).
    pub block_norm: crate::config::BlockNorm,
    /// FalconMamba's dt, B and C norm epsilon, `None` for Mamba-1 (see
    /// [`crate::MambaConfig::bcdt_rms_eps`]).
    pub bcdt_rms_eps: Option<f32>,
    /// Layers in forward and reversed pairs (see
    /// [`crate::MambaConfig::bidirectional`]).
    pub bidirectional: bool,
}

impl GpuMambaDims {
    /// Refuse dims these kernels cannot run. The SSM kernels keep
    /// `d_state` floats per thread up to the capacity the context compiled
    /// them with: past it every launch would return without writing, so
    /// nothing is launched at all. The conv needs at least one tap, and
    /// `xdbl_dim` must be the x_proj row the kernels read.
    pub fn check_for(&self, ctx: &GpuCtx) -> Result<(), String> {
        if self.d_state > ctx.state_cap() {
            return Err(format!(
                "d_state {} exceeds the state capacity {} this context compiled \
                 the Mamba-1 kernels for (GpuCtx::new_with_state_cap)",
                self.d_state,
                ctx.state_cap()
            ));
        }
        if self.d_conv == 0 {
            return Err("the Mamba-1 conv needs at least one tap (d_conv 0)".into());
        }
        if self.xdbl_dim != self.dt_rank + 2 * self.d_state {
            return Err(format!(
                "xdbl_dim {} is not dt_rank {} + 2 * d_state {}",
                self.xdbl_dim, self.dt_rank, self.d_state
            ));
        }
        self.check_int_args()
    }

    /// Refuse dims whose 32-bit kernel arguments would wrap: the kernels
    /// index every tensor with 64 bits, but take the dimensions, the row
    /// count of a window and its row widths as `int`s, the list
    /// [`crate::mamba1::route::admit`] checks.
    pub fn check_int_args(&self) -> Result<(), String> {
        let widths = crate::mamba1::route::IntWidths {
            batch: self.batch,
            seq_len: self.seq_len,
            d_model: self.d_model,
            input_dim: self.mamba_input_dim,
            d_inner: self.d_inner,
            xdbl_dim: self.xdbl_dim,
            d_conv: self.d_conv,
            d_state: self.d_state,
        };
        crate::gpu::launch::check_int_args(&widths.args())
    }

    /// `B * T` — total number of (sample, timestep) pairs.
    pub fn bt(&self) -> usize {
        self.batch * self.seq_len
    }
}

// ---------------------------------------------------------------------------
// GPU Mamba activation storage
// ---------------------------------------------------------------------------

/// Saved activations for one Mamba layer on GPU.
///
/// All tensors indexed as `[B * T * dim]` unless noted.
/// Needed by the backward pass to compute gradients.
pub struct GpuMambaLayerActs {
    // -- F1: the block norm --
    /// Pre-norm input saved for residual add `[B*T*d_model]`.
    pub residual: GpuBuffer,
    /// The block norm's statistics per (b,t) pair, `BlockNorm::stats_per_row`
    /// each `[B*T*stats]`.
    pub rms_vals: GpuBuffer,
    /// Post-norm output `[B*T*d_model]`.
    pub post_norm: GpuBuffer,

    // -- F2: in_proj output --
    /// The in_proj output `[B*T*2*d_inner]`, saved whole: the conv reads
    /// its x half and the gating its gate half through a row stride, and
    /// the backward reconstructs the conv windows from the x half, so
    /// nothing splits it.
    pub proj: GpuBuffer,

    // -- F4a: Conv1d + SiLU --
    /// Only the CARRY-IN window per (b, d) survives as conv tape
    /// `[B*d_inner*d_conv]`; later windows are reconstructed from the x
    /// half of `proj` in the backward.
    pub conv_states: GpuBuffer,
    /// Post-SiLU conv output, SSM input `[B*T*d_inner]`.
    pub u: GpuBuffer,

    // -- F4b-c: x_proj + dt_proj --
    /// x_proj output (dt_raw, B, C concatenated) `[B*T*xdbl_dim]`.
    pub xdbl: GpuBuffer,
    /// Delta before softplus `[B*T*d_inner]`.
    pub delta_raw: GpuBuffer,
    /// Delta after softplus `[B*T*d_inner]`.
    pub delta: GpuBuffer,

    // -- F4d: SSM --
    /// Hidden state saved BEFORE each step. T+1 entries per sample:
    /// index 0 = initial state, index t+1 = state after step t.
    /// Layout: `[B * (T+1) * d_inner * d_state]`.
    pub h_saved: GpuBuffer,
    /// SSM output before gating `[B*T*d_inner]`.
    pub y: GpuBuffer,

    // -- F4e: Gating --
    /// Gated output `y * gate_silu` `[B*T*d_inner]`.
    pub gated: GpuBuffer,
    /// FalconMamba's dt, B and C norm scales `[B*T*3]`; empty for Mamba-1.
    pub bcdt_rstd: GpuBuffer,
}

/// Full Mamba backbone activations on GPU.
pub struct GpuMambaBackboneActs {
    /// Saved input to input_proj: `[B*T*mamba_input_dim]`.
    pub input_proj_inputs: GpuBuffer,
    /// Per-layer activations.
    pub layers: Vec<GpuMambaLayerActs>,
    /// Saved pre-norm_f input for backward: `[B*T*d_model]`.
    pub norm_f_input: GpuBuffer,
    /// Saved norm_f statistics per sample for its backward: `[B*T*stats]`.
    pub norm_f_rms: GpuBuffer,
    /// The gated MLP's saves, for a model with one (see [`Self::with_mlp`]).
    pub mlp: Option<super::mlp::GpuMambaMlpBufs>,
}

impl GpuMambaBackboneActs {
    /// The same activations with the gated MLP's saves, for a model whose
    /// MLP is `hidden` wide (0 leaves them out).
    pub fn with_mlp(
        mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMambaDims,
        hidden: usize,
    ) -> Result<Self, String> {
        self.mlp = (hidden > 0)
            .then(|| {
                super::mlp::GpuMambaMlpBufs::new(
                    stream,
                    (dims.bt(), dims.d_model, dims.n_layers),
                    (hidden, dims.block_norm),
                )
            })
            .transpose()?;
        Ok(self)
    }

    /// Allocate all activation save buffers for Mamba forward/backward.
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMambaDims,
    ) -> Result<Self, String> {
        let batch = dims.batch;
        let seq_len = dims.seq_len;
        let d_model = dims.d_model;
        let d_inner = dims.d_inner;
        let d_state = dims.d_state;
        let d_conv = dims.d_conv;
        let dt_rank = dims.dt_rank;
        let n_layers = dims.n_layers;
        let mamba_input_dim = dims.mamba_input_dim;
        let bt = batch * seq_len;
        let xdbl_dim = dt_rank + 2 * d_state;
        let stats = bt * dims.block_norm.stats_per_row();

        let layers = (0..n_layers)
            .map(|_| {
                Ok(GpuMambaLayerActs {
                    // F1: the block norm
                    residual: GpuBuffer::zeros(stream, bt * d_model)?,
                    rms_vals: GpuBuffer::zeros(stream, stats)?,
                    post_norm: GpuBuffer::zeros(stream, bt * d_model)?,
                    // F2: in_proj output
                    proj: GpuBuffer::zeros(stream, bt * 2 * d_inner)?,
                    // F4a: Conv1d + SiLU
                    conv_states: GpuBuffer::zeros(stream, batch * d_inner * d_conv)?,
                    u: GpuBuffer::zeros(stream, bt * d_inner)?,
                    // F4b-c: x_proj + dt_proj
                    xdbl: GpuBuffer::zeros(stream, bt * xdbl_dim)?,
                    delta_raw: GpuBuffer::zeros(stream, bt * d_inner)?,
                    delta: GpuBuffer::zeros(stream, bt * d_inner)?,
                    // F4d: SSM — the full tape has T+1 entries; the
                    // parallel route's slim tape keeps only the
                    // per-chunk (run_a, run_b, h_entry) rows.
                    h_saved: GpuBuffer::zeros(
                        stream,
                        if dims.scan_mode.use_parallel(seq_len, d_state)
                            && crate::gpu::launch::scan_tape_slim()
                        {
                            crate::gpu::launch::scan_tape_len(batch, seq_len, d_inner, d_state)
                        } else {
                            batch * (seq_len + 1) * d_inner * d_state
                        },
                    )?,
                    y: GpuBuffer::zeros(stream, bt * d_inner)?,
                    // F4e: Gating
                    gated: GpuBuffer::zeros(stream, bt * d_inner)?,
                    bcdt_rstd: GpuBuffer::zeros(
                        stream,
                        if dims.bcdt_rms_eps.is_some() {
                            bt * 3
                        } else {
                            0
                        },
                    )?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(Self {
            input_proj_inputs: GpuBuffer::zeros(stream, bt * mamba_input_dim)?,
            layers,
            norm_f_input: GpuBuffer::zeros(stream, bt * d_model)?,
            norm_f_rms: GpuBuffer::zeros(stream, stats)?,
            mlp: None,
        })
    }
}

// ---------------------------------------------------------------------------
// GPU Mamba scratch buffers
// ---------------------------------------------------------------------------

/// Scratch buffers for GPU Mamba forward/backward.
///
/// Allocated once at training start, reused every gradient step.
pub struct GpuMambaScratch {
    /// Dimensions this scratch was allocated for.
    pub dims: GpuMambaDims,
    // -- Forward scratch --
    /// Out-proj output `[B*T*d_model]`.
    pub out_flat: GpuBuffer,
    /// A bidirectional pair's rows `[B*T*d_model]`: in the forward the
    /// stream after the pair's first mixer, in the backward d loss / d (the
    /// pair's output) held aside for the first layer (one element without
    /// pairs).
    pub pair: GpuBuffer,
    /// Gathered dt portion of xdbl for dt_proj SGEMM `[B*T*dt_rank]`.
    /// Separate from backward `d_dt_input` to avoid forward/backward collision.
    pub dt_gather_buf: GpuBuffer,

    // -- Backward scratch --
    /// Gradient of gated output `[B*T*d_inner]`.
    pub d_gated: GpuBuffer,
    /// Gradient of SSM output y `[B*T*d_inner]`.
    pub d_y: GpuBuffer,
    /// Gradient of discretized delta `[B*T*d_inner]`.
    pub d_delta: GpuBuffer,
    /// Gradient of raw delta (pre-softplus) `[B*T*d_inner]`.
    pub d_delta_raw: GpuBuffer,
    /// Gradient of SSM input u `[B*T*d_inner]`.
    pub d_u: GpuBuffer,
    /// Gradient of u from x_proj path `[B*T*d_inner]`.
    pub d_u_xproj: GpuBuffer,
    /// Gradient of x_proj output `[B*T*xdbl_dim]`.
    pub d_xdbl: GpuBuffer,
    /// Gradient of in_proj output `[B*T*2*d_inner]`; the conv and gating
    /// backwards write their halves in place.
    pub d_proj: GpuBuffer,
    /// Gradient of post-norm activations `[B*T*d_model]`.
    pub d_norm: GpuBuffer,
    /// Backward dx output from dt_proj `[B*T*dt_rank]`.
    pub d_dt_input: GpuBuffer,
    /// Gathered dt portion of xdbl for dt_proj backward x_saved `[B*T*dt_rank]`.
    /// Separate from `d_dt_input` to avoid aliasing in SGEMM backward.
    pub dt_xdbl_buf: GpuBuffer,

    // -- SSM backward reduction buffers --
    /// Per-(b,t,d,n) local dB before reduction `[B*T*d_inner*d_state]`.
    pub d_b_local: GpuBuffer,
    /// Per-(b,t,d,n) local dC before reduction `[B*T*d_inner*d_state]`.
    pub d_c_local: GpuBuffer,
    /// Per-channel dD accumulated across (b,t) `[B*d_inner]`.
    pub d_d_local: GpuBuffer,
    /// Per-channel d_a_log accumulated across (b,t) `[B*d_inner*d_state]`.
    pub d_a_log_local: GpuBuffer,
    /// Reduced dB `[B*T*d_state]`.
    pub d_b_reduced: GpuBuffer,
    /// Reduced dC `[B*T*d_state]`.
    pub d_c_reduced: GpuBuffer,

    // -- Input projection backward --
    /// Discarded dx for input_proj backward (input embedding detached) `[B*T*mamba_input_dim]`.
    pub d_input_proj_dx: GpuBuffer,

    // -- Rule-B axis-0 reduction partials (determinism fix) --
    /// Scratch for per-sample partials produced by Rule-B backward kernels
    /// (rmsnorm_bwd d_scale, the conv backward's tap and bias partials). Reduced via
    /// `reduce_sum_axis0` to produce deterministic cross-batch accumulators
    /// (replaces the previous atomicAdd accumulators).
    ///
    /// Sized to the max of the two consumer requirements:
    ///   - rmsnorm_bwd partials: `B*T * d_model`
    ///   - conv1d_bwd partials: `B * d_inner * d_conv + B * d_inner = B * d_inner * (d_conv + 1)`
    pub axis0_partials: GpuBuffer,
    /// The conv backward's pre-activation gradients `[B*T*d_inner]` for a
    /// conv wider than the tiled kernels hold (one element otherwise).
    pub conv_dco: GpuBuffer,
    /// The gated MLP backward's buffers, for a model with one (see
    /// [`Self::with_mlp`]).
    pub mlp: Option<super::mlp::GpuMambaMlpBwdBufs>,
}

impl GpuMambaScratch {
    /// The same scratch with the gated MLP backward's buffers, for a model
    /// whose MLP is `hidden` wide (0 leaves them out).
    pub fn with_mlp(
        mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        hidden: usize,
    ) -> Result<Self, String> {
        let dims = self.dims;
        self.mlp = (hidden > 0)
            .then(|| super::mlp::GpuMambaMlpBwdBufs::new(stream, (dims.bt(), dims.d_model), hidden))
            .transpose()?;
        Ok(self)
    }

    /// Allocate all scratch buffers for Mamba forward/backward.
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMambaDims,
    ) -> Result<Self, String> {
        let batch = dims.batch;
        let d_model = dims.d_model;
        let d_inner = dims.d_inner;
        let d_state = dims.d_state;
        let dt_rank = dims.dt_rank;
        let mamba_input_dim = dims.mamba_input_dim;
        let bt = batch * dims.seq_len;
        let xdbl_dim = dt_rank + 2 * d_state;
        // Fold-depth dB/dC partials (see the mixed scratch note).
        let bc_rows = if dims.scan_mode.use_parallel(dims.seq_len, d_state)
            && d_inner.is_multiple_of(crate::gpu::launch::SCAN_BWD_DGROUP)
        {
            d_inner / crate::gpu::launch::SCAN_BWD_DGROUP
        } else {
            d_inner
        };

        Ok(Self {
            dims: *dims,
            // Forward scratch
            out_flat: GpuBuffer::zeros(stream, bt * d_model)?,
            pair: GpuBuffer::zeros(stream, if dims.bidirectional { bt * d_model } else { 1 })?,
            dt_gather_buf: GpuBuffer::zeros(stream, bt * dt_rank)?,
            // Backward scratch
            d_gated: GpuBuffer::zeros(stream, bt * d_inner)?,
            d_y: GpuBuffer::zeros(stream, bt * d_inner)?,
            d_delta: GpuBuffer::zeros(stream, bt * d_inner)?,
            d_delta_raw: GpuBuffer::zeros(stream, bt * d_inner)?,
            d_u: GpuBuffer::zeros(stream, bt * d_inner)?,
            d_u_xproj: GpuBuffer::zeros(stream, bt * d_inner)?,
            d_xdbl: GpuBuffer::zeros(stream, bt * xdbl_dim)?,
            d_proj: GpuBuffer::zeros(stream, bt * 2 * d_inner)?,
            d_norm: GpuBuffer::zeros(stream, bt * d_model)?,
            d_dt_input: GpuBuffer::zeros(stream, bt * dt_rank)?,
            dt_xdbl_buf: GpuBuffer::zeros(stream, bt * dt_rank)?,
            // SSM backward per-thread buffers
            // Fold-depth dB/dC partials (see the mixed scratch note).
            d_b_local: GpuBuffer::zeros(stream, bt * bc_rows * d_state)?,
            d_c_local: GpuBuffer::zeros(stream, bt * bc_rows * d_state)?,
            d_d_local: GpuBuffer::zeros(stream, batch * d_inner)?,
            // The fold backward writes ONE partial row per chunk
            // ([b * n_chunks, di * ds], walk order); the sequential
            // route uses the first [b, di * ds] prefix as before.
            d_a_log_local: GpuBuffer::zeros(
                stream,
                batch
                    * dims.seq_len.div_ceil(crate::gpu::launch::SCAN_CHUNK).max(1)
                    * d_inner
                    * d_state,
            )?,
            d_b_reduced: GpuBuffer::zeros(stream, bt * d_state)?,
            d_c_reduced: GpuBuffer::zeros(stream, bt * d_state)?,
            // Discarded dx for input_proj backward
            d_input_proj_dx: GpuBuffer::zeros(stream, bt * mamba_input_dim)?,
            // Rule-B axis-0 partials scratch — sized to fit largest consumer.
            // rmsnorm_bwd needs [B*T * d_model]; conv1d_bwd needs
            // [B * d_inner * d_conv] + [B * d_inner] stacked.
            axis0_partials: GpuBuffer::zeros(
                stream,
                std::cmp::max(
                    bt * d_model,
                    batch * dims.seq_len.div_ceil(128) * d_inner * (dims.d_conv + 1),
                ),
            )?,
            conv_dco: GpuBuffer::zeros(
                stream,
                super::conv::conv_dco_len(bt, d_inner, dims.d_conv),
            )?,
            mlp: None,
        })
    }
}

// ---------------------------------------------------------------------------
// GPU Mamba forward
// ---------------------------------------------------------------------------

/// GPU Mamba single-layer forward: 10-step pipeline.
///
/// Mirrors CPU `forward_mamba_layer_batched` but batches across B*T for SGEMM.
/// Conv1d and SSM use burnin kernels (sequential across T, parallel across B*d_inner).
///
/// Mirrors CPU `forward_mamba_layer_batched` from train/forward.rs.
/// Per-layer state pointers — raw device pointers into flat buffers at layer offset.
/// Created by backbone from flat conv_states/ssm_states/a_neg_all buffers.
/// Matches official Mamba pattern: single flat allocation, per-layer kernel invocation with offset.
pub struct MambaLayerPtrs {
    pub conv_state: cudarc::driver::sys::CUdeviceptr, // &mut [B * d_inner * d_conv]
    pub ssm_state: cudarc::driver::sys::CUdeviceptr,  // &mut [B * d_inner * d_state]
    pub a_neg: cudarc::driver::sys::CUdeviceptr,      // &[d_inner * d_state]
}

pub fn gpu_forward_mamba_layer(
    ctx: &GpuCtx,
    stream_out: cudarc::driver::sys::CUdeviceptr,
    acts: &mut GpuMambaLayerActs,
    lw: &GpuMambaTrainLayerWeights,
    layer_ptrs: &MambaLayerPtrs,
    scratch: &mut GpuMambaScratch,
) -> Result<(), String> {
    layer_forward(ctx, stream_out, (acts, lw, layer_ptrs), scratch, None)
}

/// [`gpu_forward_mamba_layer`]; `pair_base`, set for the second layer of a
/// bidirectional pair, is the stream after the pair's first mixer: the
/// layer ran on the reversed window, so its output turns back and joins
/// those rows instead of its own input.
fn layer_forward(
    ctx: &GpuCtx,
    stream_out: cudarc::driver::sys::CUdeviceptr,
    (acts, lw, layer_ptrs): (
        &mut GpuMambaLayerActs,
        &GpuMambaTrainLayerWeights,
        &MambaLayerPtrs,
    ),
    scratch: &mut GpuMambaScratch,
    pair_base: Option<cudarc::driver::sys::CUdeviceptr>,
) -> Result<(), String> {
    let dims = scratch.dims;
    dims.check_for(ctx)?;
    let bt = dims.bt();
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dt_rank = dims.dt_rank;
    let xdbl_dim = dims.xdbl_dim;
    let b = dims.batch;
    let t = dims.seq_len;
    let d_conv = dims.d_conv;

    // ===================================================================
    // F1: the block norm — compute post_norm from this layer's residual
    // slot
    // ===================================================================
    // The residual stream value already lives in acts.residual: the
    // PREVIOUS layer's residual_add wrote it there directly (layer 0 is
    // seeded by the driver). The old per-layer temporal→residual D2D
    // copy is gone — same next-layer-residual plumbing the mixed lane
    // ships (forward_mixed.rs).
    block_norm::forward(
        ctx,
        WeightDtype::F32,
        (
            acts.post_norm.cached_ptr(),
            acts.rms_vals.cached_ptr(),
            acts.residual.cached_ptr(),
        ),
        lw.norm(),
        (bt, dm, dims.rms_norm_eps),
        "block norm mamba",
    )?;

    // ===================================================================
    // F2: Batch in_proj -- ONE cuBLAS SGEMM
    //     [B*T, d_model] -> [B*T, 2*d_inner]
    // ===================================================================
    // The output stays whole: the conv and the gating read their halves
    // through a row stride, so no split pass runs.
    gpu_gemm_bi_forward_raw(
        ctx,
        &mut acts.proj,
        &acts.post_norm,
        lw.in_proj_w.cached_ptr(),
        lw.in_proj_b_ptr(),
        (bt, dm, 2 * di),
    )?;

    // ===================================================================
    // F4a: Conv1d burnin + fused SiLU
    // ===================================================================
    // Tiled kernel, typed f32 instantiation. Argument order:
    // (u_out, state, conv_states_out, x_branch, weight, bias,
    //  batch, T, d_inner, d_conv, x_stride). Grid (b*di, T tiles) instead
    // of a serial walk over T; x is the first half of the in_proj output.
    super::conv::conv_forward(
        ctx,
        crate::gpu::dtype::WeightDtype::F32,
        &super::conv::ConvForward {
            u: acts.u.cached_ptr(),
            u_t: (0, 0),
            state: layer_ptrs.conv_state,
            saved: acts.conv_states.cached_ptr(),
            x: (acts.proj.cached_ptr(), 2 * di),
            weight: lw.conv1d_weight.cached_ptr(),
            bias: lw.conv1d_bias_ptr(),
        },
        (b, t, di, d_conv),
        format_args!("training"),
    )?;

    // ===================================================================
    // F4b: x_proj -- ONE cuBLAS SGEMM
    //      [B*T, d_inner] -> [B*T, xdbl_dim]
    // ===================================================================
    gpu_gemm_bi_forward_raw(
        ctx,
        &mut acts.xdbl,
        &acts.u,
        lw.x_proj_w.cached_ptr(),
        None,
        (bt, di, xdbl_dim),
    )?;
    // FalconMamba's dt, B and C norms, the scales kept for the backward.
    if let Some(eps) = dims.bcdt_rms_eps {
        launch_bcdt_norm(
            ctx,
            (acts.xdbl.cached_ptr(), acts.bcdt_rstd.cached_ptr()),
            crate::gpu::dtype::WeightDtype::F32,
            (bt, dims.dt_rank, dims.d_state),
            eps,
        )?;
    }

    // ===================================================================
    // F4c: dt_proj + softplus
    //      dt portion of xdbl [B*T, dt_rank] -> delta_raw [B*T, d_inner]
    // ===================================================================
    // Gather dt portion: xdbl[:, 0..dt_rank] → dt_gather_buf
    {
        let bt_i = bt as i32;
        let xdbl_i = xdbl_dim as i32;
        let dt_i = dt_rank as i32;
        let offset: i32 = 0;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.gather_cols);
        builder.arg(scratch.dt_gather_buf.inner_mut());
        builder.arg(acts.xdbl.inner());
        builder.arg(&bt_i);
        builder.arg(&xdbl_i);
        builder.arg(&dt_i);
        builder.arg(&offset);
        unsafe { builder.launch(grid_1d_wide(bt * dt_rank)) }
            .map_err(|e| format!("gather_cols dt mamba: {:?}", e))?;
    }

    gpu_gemm_bi_forward_raw(
        ctx,
        &mut acts.delta_raw,
        &scratch.dt_gather_buf,
        lw.dt_proj_w.cached_ptr(),
        Some(lw.dt_proj_b.cached_ptr()),
        (bt, dt_rank, di),
    )?;

    // Softplus is fused into the scan kernels below: they read delta_raw,
    // apply softplus inline (same value the deleted copy pass stored) and
    // write the post-softplus save the backward replays from.

    // ===================================================================
    // F4d: SSM burnin forward
    // ===================================================================
    // B/C are stored in xdbl at offsets [dt_rank..dt_rank+d_state] and [dt_rank+d_state..xdbl_dim]
    // We need raw pointers to these sub-slices.
    // For ssm_burnin_forward: B[batch*T*d_state], C[batch*T*d_state]
    // xdbl layout: [B*T, dt_rank + d_state + d_state]
    // B starts at offset dt_rank, C at dt_rank + d_state
    // Use gather_cols to extract B and C into scratch buffers.
    // Actually, ssm_burnin_forward expects contiguous B[batch*T, d_state] and C[batch*T, d_state].
    // We need to gather them from xdbl columns.
    //
    // For now, use a lightweight approach: launch gather_cols for B and C into d_b_reduced/d_c_reduced
    // (reusing scratch buffers that have the right size [B*T*d_state]).
    // B and C from xdbl, t-major on the parallel route so the scan's
    // per-(d, n) lane reads contiguous t-runs.
    BcGather {
        dst_b: scratch.d_b_reduced.cached_ptr(),
        dst_c: scratch.d_c_reduced.cached_ptr(),
        src: acts.xdbl.cached_ptr(),
        batch: b,
        seq: t,
        src_stride: xdbl_dim,
        d_state: ds,
        b_offset: dt_rank,
    }
    .launch(ctx, WeightDtype::F32, dims.scan_mode.use_parallel(t, ds))
    .map_err(|e| format!("fwd mamba: {e}"))?;

    // SSM burnin forward: the parallel prefix scan or the sequential
    // kernel, as the scan mode resolves for this window and d_state. Both
    // kernels have identical interfaces and produce identical results.
    {
        let b_i = b as i32;
        let t_i = t as i32;
        let di_i = di as i32;
        let ds_i = ds as i32;

        if dims.scan_mode.use_parallel(t, ds) {
            // Parallel prefix scan: O(T / NTHREADS) per thread instead of O(T).
            // Grid: (batch, d_inner) — one block per (b, d) pair.
            let tape_p = acts.h_saved.cached_ptr();
            let slim = crate::gpu::launch::scan_tape_slim();
            let slim_i: i32 = i32::from(slim);
            let width = scan_fwd_index_width(b, t, di, ds, slim);
            for (y_first, cfg) in y_runs(grid_parallel_scan(b, di, ds)) {
                let mut builder = ctx
                    .stream
                    .launch_builder(ctx.kernels.ssm_parallel_fwd.get(width));
                builder.arg(&layer_ptrs.ssm_state);
                builder.arg(acts.y.inner_mut());
                builder.arg(acts.h_saved.inner_mut());
                builder.arg(acts.delta_raw.inner());
                builder.arg(acts.delta.inner_mut());
                builder.arg(acts.u.inner());
                builder.arg(scratch.d_b_reduced.inner());
                builder.arg(scratch.d_c_reduced.inner());
                builder.arg(&layer_ptrs.a_neg);
                let dp_ptr = lw.d_param.cached_ptr();
                builder.arg(&dp_ptr);
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&di_i);
                builder.arg(&ds_i);
                builder.arg(&tape_p);
                builder.arg(&slim_i);
                builder.arg(&y_first);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| format!("ssm_parallel_fwd mamba: {:?}", e))?;
            }
        } else {
            // Sequential kernel: each thread handles one (b, d) pair over all T steps.
            let width = seq_scan_index_width(b, t, di, ds, di, true);
            let mut builder = ctx
                .stream
                .launch_builder(&ctx.kernels.ssm_burnin_fwd_typed.get(width).f32);
            builder.arg(&layer_ptrs.ssm_state);
            builder.arg(acts.y.inner_mut());
            builder.arg(acts.h_saved.inner_mut());
            builder.arg(acts.delta_raw.inner());
            builder.arg(acts.delta.inner_mut());
            builder.arg(acts.u.inner());
            builder.arg(scratch.d_b_reduced.inner());
            builder.arg(scratch.d_c_reduced.inner());
            builder.arg(&layer_ptrs.a_neg);
            let dp_ptr = lw.d_param.cached_ptr();
            builder.arg(&dp_ptr);
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&di_i);
            builder.arg(&ds_i);
            let lanes_i = crate::gpu::launch::seq_lanes(ds) as i32;
            builder.arg(&lanes_i);
            unsafe { builder.launch(crate::gpu::launch::grid_seq(b, di, ds)?) }
                .map_err(|e| format!("ssm_burnin_fwd mamba: {:?}", e))?;
        }
    }

    // ===================================================================
    // F4e: Gating — gated = y * SiLU(gate), the gate read from the second
    // half of the in_proj output through the row stride.
    // ===================================================================
    {
        let n = (bt * di) as i64;
        let di_i = di as i32;
        let gate_stride = (2 * di) as i32;
        let gate_off = di as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.gate_mul_silu);
        builder.arg(acts.gated.inner_mut());
        builder.arg(acts.y.inner());
        builder.arg(acts.proj.inner());
        builder.arg(&n);
        builder.arg(&di_i);
        builder.arg(&gate_stride);
        builder.arg(&gate_off);
        unsafe { builder.launch(grid_1d_wide(bt * di)) }
            .map_err(|e| format!("gate_mul_silu mamba: {:?}", e))?;
    }

    // ===================================================================
    // F5: Batch out_proj — ONE cuBLAS SGEMM
    //     [B*T, d_inner] -> [B*T, d_model]
    // ===================================================================
    gpu_gemm_bi_forward_raw(
        ctx,
        &mut scratch.out_flat,
        &acts.gated,
        lw.out_proj_w.cached_ptr(),
        lw.out_proj_b_ptr(),
        (bt, di, dm),
    )?;

    // ===================================================================
    // F6: Residual add — write the NEXT layer's residual slot (or
    // norm_f_input for the last layer) directly, no temporal round trip
    // ===================================================================
    let residual = match pair_base {
        Some(base) => {
            let rows = WindowRows {
                ptr: scratch.out_flat.cached_ptr(),
                elem_bytes: 4,
                batch: dims.batch,
                seq_len: dims.seq_len,
                width: dm,
            };
            reverse_rows(ctx, rows, 0)?;
            base
        }
        None => acts.residual.cached_ptr(),
    };
    crate::gpu::launch::launch_elems(
        &ctx.stream,
        &ctx.kernels.residual_add,
        &[stream_out, residual, scratch.out_flat.cached_ptr()],
        bt * dm,
        format_args!("residual_add mamba"),
    )?;

    Ok(())
}

/// GPU Mamba backbone forward: input_proj + N layers.
///
/// Mirrors CPU `forward_mamba_backbone_batched` from train/forward.rs.
/// Bundled GPU recurrent state for backbone forward/backward.
pub struct GpuRecurrentState {
    pub conv_states: GpuBuffer,
    pub ssm_states: GpuBuffer,
    pub a_neg_all: GpuBuffer,
}

/// CPU-side snapshot of the carried recurrence (conv + SSM state) for
/// TBPTT-style window handoff and checkpointed resume. `a_neg_all` is
/// deliberately NOT part of the blob — it is derived from `a_log` and
/// refreshed after every optimizer step, so importing a stale copy
/// would desynchronize it from the weights.
#[derive(Clone, Debug, PartialEq)]
pub struct RecurrentStateBlob {
    pub conv_states: Vec<f32>,
    pub ssm_states: Vec<f32>,
}

impl GpuRecurrentState {
    /// Download the carried recurrence to CPU.
    pub fn export_state(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<RecurrentStateBlob, String> {
        Ok(RecurrentStateBlob {
            conv_states: self.conv_states.to_cpu(stream)?,
            ssm_states: self.ssm_states.to_cpu(stream)?,
        })
    }

    /// Upload a previously exported recurrence. Errs on a length
    /// mismatch — the blob belongs to a different shape.
    pub fn import_state(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        blob: &RecurrentStateBlob,
    ) -> Result<(), String> {
        self.import_parts(stream, &blob.conv_states, &blob.ssm_states)
    }

    /// [`Self::import_state`] from borrowed conv and SSM values.
    pub(crate) fn import_parts(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        conv_states: &[f32],
        ssm_states: &[f32],
    ) -> Result<(), String> {
        if conv_states.len() != self.conv_states.len() || ssm_states.len() != self.ssm_states.len()
        {
            return Err(format!(
                "recurrent state mismatch: blob conv/ssm = {}/{} elements, \
                 state = {}/{} — the blob belongs to a different shape",
                conv_states.len(),
                ssm_states.len(),
                self.conv_states.len(),
                self.ssm_states.len()
            ));
        }
        self.conv_states.upload(stream, conv_states)?;
        self.ssm_states.upload(stream, ssm_states)?;
        Ok(())
    }
}

pub fn gpu_forward_mamba_backbone(
    ctx: &GpuCtx,
    temporal: &mut GpuBuffer,
    acts: &mut GpuMambaBackboneActs,
    mamba_w: &GpuMambaTrainWeights,
    mamba_input: &GpuBuffer,
    state: &mut GpuRecurrentState,
    scratch: &mut GpuMambaScratch,
) -> Result<(), String> {
    let dims = scratch.dims;
    dims.check_for(ctx)?;
    let bt = dims.bt();
    super::mlp::check_width(
        mamba_w.mlp.first().map(|m| m.hidden()),
        acts.mlp.as_ref().map(|m| m.hidden),
        "activations",
    )?;

    if mamba_w.input_proj_w.is_empty() {
        // A model without an input projection (the official checkpoints,
        // whose embedding feeds the first norm) takes its input as the rows.
        temporal.copy_from(mamba_input, &ctx.stream)?;
    } else {
        // Save input for backward
        acts.input_proj_inputs.copy_from(mamba_input, &ctx.stream)?;

        // Batched input_proj: [B*T, mamba_input_dim] -> [B*T, d_model]
        gpu_gemm_bi_forward_raw(
            ctx,
            temporal,
            mamba_input,
            mamba_w.input_proj_w.cached_ptr(),
            Some(mamba_w.input_proj_b.cached_ptr()),
            (bt, dims.mamba_input_dim, dims.d_model),
        )?;
    }

    // Seed the residual chain: layer 0's residual slot gets the
    // input_proj output ONCE; from there every layer's residual_add
    // writes the next layer's slot directly (mixed-lane plumbing —
    // kills the per-layer temporal→residual copy).
    acts.layers[0].residual.copy_from(temporal, &ctx.stream)?;

    // Mamba layers — per-layer offset into flat state buffers
    // Matches official Mamba pattern: single flat allocation, per-layer kernel invocation.
    // Same pattern as CPU train/forward.rs
    let conv_per_layer = dims.batch * dims.d_inner * dims.d_conv;
    let ssm_per_layer = dims.batch * dims.d_inner * dims.d_state;
    let a_neg_per_layer = dims.d_inner * dims.d_state;
    let bidi = dims.bidirectional;
    // A bidirectional model reads whole sequences: every window starts
    // from the empty state.
    if bidi {
        for buf in [&state.conv_states, &state.ssm_states] {
            zero_bytes(ctx, buf.cached_ptr(), buf.len() * 4)?;
        }
    }
    let pair_rows = scratch.pair.cached_ptr();

    for layer_idx in 0..dims.n_layers {
        // Compute per-layer pointers. cached_ptr — no SyncOnDrop, CUDA Graph safe.
        let conv_base = state.conv_states.raw_ptr(&ctx.stream);
        let ssm_base = state.ssm_states.raw_ptr(&ctx.stream);
        let aneg_base = state.a_neg_all.raw_ptr(&ctx.stream);
        let f32_sz = std::mem::size_of::<f32>() as u64;
        let layer_ptrs = MambaLayerPtrs {
            conv_state: conv_base + (layer_idx * conv_per_layer) as u64 * f32_sz,
            ssm_state: ssm_base + (layer_idx * ssm_per_layer) as u64 * f32_sz,
            a_neg: aneg_base + (layer_idx * a_neg_per_layer) as u64 * f32_sz,
        };

        let stream_out = if layer_idx + 1 < dims.n_layers {
            acts.layers[layer_idx + 1].residual.cached_ptr()
        } else {
            acts.norm_f_input.cached_ptr()
        };
        // A bidirectional pair: the first mixer's sum is the pair's
        // working rows; the second runs on its reversed input, the pair's
        // input the first layer's slot holds, and adds onto them. The
        // pair's MLPs follow both mixers, the first MLP's output the
        // second's input.
        let reversed = reads_reversed(bidi, layer_idx);
        if reversed {
            let (before, rest) = acts.layers.split_at_mut(layer_idx);
            rest[0]
                .residual
                .copy_from(&before[layer_idx - 1].residual, &ctx.stream)?;
            let rows = WindowRows {
                ptr: rest[0].residual.cached_ptr(),
                elem_bytes: 4,
                batch: dims.batch,
                seq_len: dims.seq_len,
                width: dims.d_model,
            };
            reverse_rows(ctx, rows, 0)?;
        }
        // With an MLP the mixer's sum lands in the MLP's input rows, and the
        // MLP writes the layer's output.
        let mixer_out = if opens_pair(bidi, layer_idx) {
            pair_rows
        } else {
            let first_mlp = if reversed { layer_idx - 1 } else { layer_idx };
            match &acts.mlp {
                Some(m) => m.layers[first_mlp].x_in.cached_ptr(),
                None => stream_out,
            }
        };
        // Split-borrow: the layer body needs &mut acts.layers[layer_idx]
        // while mixer_out points at a DIFFERENT slot (next layer /
        // norm_f_input / the MLP's input) — raw pointer, no aliasing of the
        // borrowed acts.
        layer_forward(
            ctx,
            mixer_out,
            (
                &mut acts.layers[layer_idx],
                &mamba_w.layers[layer_idx],
                &layer_ptrs,
            ),
            scratch,
            reversed.then_some(pair_rows),
        )?;
        if let Some(m) = acts.mlp.as_mut() {
            let shape = (bt, dims.d_model, dims.rms_norm_eps);
            if reversed {
                let into_second = m.layers[layer_idx].x_in.cached_ptr();
                super::mlp::forward_f32(
                    ctx,
                    into_second,
                    (&mut m.layers[layer_idx - 1], &mamba_w.mlp[layer_idx - 1]),
                    &mut scratch.out_flat,
                    shape,
                )?;
            }
            if !opens_pair(bidi, layer_idx) {
                super::mlp::forward_f32(
                    ctx,
                    stream_out,
                    (&mut m.layers[layer_idx], &mamba_w.mlp[layer_idx]),
                    &mut scratch.out_flat,
                    shape,
                )?;
            }
        }
        // All kernels for this layer have been launched on stream
    }

    // The final norm (norm_f) after all Mamba layers. The last layer's
    // residual_add already wrote acts.norm_f_input (the backward's
    // pre-norm save) — no copy needed; the normed output lands in
    // `temporal`, which stays the driver's output buffer.
    block_norm::forward(
        ctx,
        WeightDtype::F32,
        (
            temporal.cached_ptr(),
            acts.norm_f_rms.cached_ptr(),
            acts.norm_f_input.cached_ptr(),
        ),
        mamba_w.norm_f(),
        (bt, dims.d_model, dims.rms_norm_eps),
        "norm_f",
    )
}

// ---------------------------------------------------------------------------
// GPU Mamba backward
// ---------------------------------------------------------------------------

/// GPU Mamba single-layer backward: 8-phase pipeline.
///
/// Mirrors CPU `backward_mamba_layer_batched` but batches across B*T for SGEMM.
/// SSM BPTT uses `ssm_backward_local` kernel + 4 reduction kernels.
///
/// On entry, `d_temporal` contains upstream gradient. On exit, gradient w.r.t. layer input.
pub fn gpu_forward_mamba_target_burnin(
    ctx: &GpuCtx,
    target_temporal: &mut GpuBuffer, // [B * d_model] — output (last timestep)
    ip_out_flat: &GpuBuffer,         // [B * T * d_model] — pre-batched input_proj
    target_w: &GpuMambaTrainWeights,
    a_neg_all: &GpuBuffer,
    scratch: &mut GpuMambaTargetScratch,
) -> Result<(), String> {
    let dims = scratch.dims;
    dims.check_for(ctx)?;
    if dims.bidirectional {
        return Err("the target burn-in reads its layers one way: \
                    a bidirectional model has no target network forward"
            .into());
    }
    let seq_len = dims.seq_len;
    let b = dims.batch;
    let bt = b * seq_len;
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dt_rank = dims.dt_rank;
    let xdbl_dim = dims.xdbl_dim;
    let d_conv = dims.d_conv;
    let t = seq_len;
    // The t-major route, as in the prefill chain.
    let ts = super::backward::tmajor_stride(&dims);
    let tmajor_in = ts > 0 && scratch.u_t.len() >= b * di * ts;
    let ts_i = if tmajor_in { ts as i32 } else { 0 };

    // Zero per-layer conv/SSM states once before layer loop
    scratch.conv_states.zero(&ctx.stream)?;
    scratch.ssm_states.zero(&ctx.stream)?;

    // Working temporal buffer: start with ip_out_flat content, overwrite per layer.
    // Copy ip_out_flat → out_flat as the initial temporal [B*T*d_model].
    scratch.out_flat.copy_from(ip_out_flat, &ctx.stream)?;

    // Per-layer state offsets
    let conv_per_layer = b * di * d_conv;
    let ssm_per_layer = b * di * ds;
    let a_neg_per_layer = di * ds;

    for layer_idx in 0..dims.n_layers {
        let lw = &target_w.layers[layer_idx];

        // Compute per-layer state pointers
        let conv_base = scratch.conv_states.raw_ptr(&ctx.stream);
        let ssm_base = scratch.ssm_states.raw_ptr(&ctx.stream);
        let aneg_base = a_neg_all.raw_ptr(&ctx.stream);
        let f32_sz = std::mem::size_of::<f32>() as u64;
        let conv_ptr = conv_base + (layer_idx * conv_per_layer) as u64 * f32_sz;
        let ssm_ptr = ssm_base + (layer_idx * ssm_per_layer) as u64 * f32_sz;
        let a_neg_ptr = aneg_base + (layer_idx * a_neg_per_layer) as u64 * f32_sz;

        // === F1: the block norm [B*T] ===
        scratch.residual.copy_from(&scratch.out_flat, &ctx.stream)?;
        block_norm::forward(
            ctx,
            WeightDtype::F32,
            (
                scratch.out_flat.cached_ptr(),
                scratch.rms_discard.cached_ptr(),
                scratch.residual.cached_ptr(),
            ),
            lw.norm(),
            (bt, dm, dims.rms_norm_eps),
            "block norm target",
        )?;

        // === F2: in_proj SGEMM [B*T, dm] -> [B*T, 2*di] ===
        gpu_gemm_bi_forward_raw(
            ctx,
            &mut scratch.proj_flat,
            &scratch.out_flat,
            lw.in_proj_w.cached_ptr(),
            lw.in_proj_b_ptr(),
            (bt, dm, 2 * di),
        )?;

        // No split: the conv reads the x half of the in_proj output
        // through the row stride and the scan gates its own store from
        // the gate half, as the prefill chain does.

        // === F4a: conv1d burnin nosave + fused SiLU, T-tiled ===
        super::conv::conv_forward(
            ctx,
            crate::gpu::dtype::WeightDtype::F32,
            &super::conv::ConvForward {
                u: scratch.u.cached_ptr(),
                u_t: if tmajor_in {
                    (scratch.u_t.cached_ptr(), ts)
                } else {
                    (0, 0)
                },
                state: conv_ptr,
                saved: 0,
                x: (scratch.proj_flat.cached_ptr(), 2 * di),
                weight: lw.conv1d_weight.cached_ptr(),
                bias: lw.conv1d_bias_ptr(),
            },
            (b, t, di, d_conv),
            format_args!("target L{layer_idx}"),
        )?;

        // === F4b: x_proj SGEMM [B*T, di] -> [B*T, xdbl_dim], and
        // FalconMamba's dt, B and C norms ===
        gpu_gemm_bi_forward_raw(
            ctx,
            &mut scratch.xdbl,
            &scratch.u,
            lw.x_proj_w.cached_ptr(),
            None,
            (bt, di, xdbl_dim),
        )?;
        if let Some(eps) = dims.bcdt_rms_eps {
            launch_bcdt_norm(
                ctx,
                (scratch.xdbl.cached_ptr(), 0),
                crate::gpu::dtype::WeightDtype::F32,
                (bt, dims.dt_rank, dims.d_state),
                eps,
            )?;
        }

        // === F4c: gather dt + dt_proj + softplus ===
        {
            let bt_i = bt as i32;
            let xdbl_i = xdbl_dim as i32;
            let dt_i = dt_rank as i32;
            let offset: i32 = 0;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.gather_cols);
            builder.arg(scratch.dt_gather.inner_mut());
            builder.arg(scratch.xdbl.inner());
            builder.arg(&bt_i);
            builder.arg(&xdbl_i);
            builder.arg(&dt_i);
            builder.arg(&offset);
            unsafe { builder.launch(grid_1d_wide(bt * dt_rank)) }
                .map_err(|e| format!("gather dt target L{layer_idx}: {:?}", e))?;
        }
        gpu_gemm_bi_forward_raw(
            ctx,
            &mut scratch.delta,
            &scratch.dt_gather,
            lw.dt_proj_w.cached_ptr(),
            Some(lw.dt_proj_b.cached_ptr()),
            (bt, dt_rank, di),
        )?;
        if tmajor_in {
            let t_i = t as i32;
            let di_i = di as i32;
            for (y_first, z_first, cfg) in yz_runs(crate::gpu::launch::grid_tmajor_tiles(b, t, di))
            {
                let mut builder = ctx
                    .stream
                    .launch_builder(&ctx.kernels.softplus_fwd_tmajor.f32);
                builder.arg(scratch.u.inner_mut());
                builder.arg(scratch.delta.inner());
                builder.arg(&t_i);
                builder.arg(&di_i);
                builder.arg(&ts_i);
                builder.arg(&y_first);
                builder.arg(&z_first);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| format!("softplus_tmajor target L{layer_idx}: {:?}", e))?;
            }
        } else {
            let n = (bt * di) as i64;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.softplus_fwd);
            builder.arg(scratch.delta.inner_mut());
            builder.arg(&n);
            unsafe { builder.launch(grid_1d_wide(bt * di)) }
                .map_err(|e| format!("softplus target L{layer_idx}: {:?}", e))?;
        }

        // === F4d: gather B/C + SSM burnin nosave ===
        BcGather {
            dst_b: scratch.b_gathered.cached_ptr(),
            dst_c: scratch.c_gathered.cached_ptr(),
            src: scratch.xdbl.cached_ptr(),
            batch: b,
            seq: t,
            src_stride: xdbl_dim,
            d_state: ds,
            b_offset: dt_rank,
        }
        .launch(ctx, WeightDtype::F32, dims.scan_mode.use_parallel(t, ds))
        .map_err(|e| format!("target L{layer_idx}: {e}"))?;
        // SSM burnin nosave: the parallel or the sequential scan, as the
        // scan mode resolves for this window and d_state.
        {
            let b_i = b as i32;
            let t_i = t as i32;
            let di_i = di as i32;
            let ds_i = ds as i32;

            // The scan multiplies the gate into its own store, reading the
            // gate half of the in_proj output through the row stride, so
            // the gated output lands directly and no gating pass runs.
            let gate_stride = (2 * di) as i32;
            if dims.scan_mode.use_parallel(t, ds) {
                let width = scan_fwd_nosave_index_width(b, t, di, ds, 2 * di, tmajor_in);
                for (y_first, cfg) in y_runs(grid_parallel_scan(b, di, ds)) {
                    let mut builder = ctx
                        .stream
                        .launch_builder(ctx.kernels.ssm_parallel_fwd_nosave.get(width));
                    let (delta_in, u_in, y_out) = if tmajor_in {
                        (&scratch.u, &scratch.u_t, &mut scratch.delta)
                    } else {
                        (&scratch.delta, &scratch.u, &mut scratch.gated)
                    };
                    builder.arg(&ssm_ptr);
                    builder.arg(y_out.inner_mut());
                    builder.arg(delta_in.inner());
                    builder.arg(u_in.inner());
                    builder.arg(scratch.b_gathered.inner());
                    builder.arg(scratch.c_gathered.inner());
                    builder.arg(&a_neg_ptr);
                    let dp_ptr = lw.d_param.cached_ptr();
                    builder.arg(&dp_ptr);
                    builder.arg(scratch.proj_flat.inner());
                    builder.arg(&gate_stride);
                    builder.arg(&b_i);
                    builder.arg(&t_i);
                    builder.arg(&di_i);
                    builder.arg(&ds_i);
                    builder.arg(&ts_i);
                    builder.arg(&y_first);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| format!("ssm_parallel_nosave target L{layer_idx}: {:?}", e))?;
                }
                if tmajor_in {
                    for (y_first, z_first, cfg) in
                        yz_runs(crate::gpu::launch::grid_tmajor_tiles(b, t, di))
                    {
                        let mut builder =
                            ctx.stream.launch_builder(&ctx.kernels.gate_from_tmajor.f32);
                        builder.arg(scratch.gated.inner_mut());
                        builder.arg(scratch.delta.inner());
                        builder.arg(scratch.proj_flat.inner());
                        builder.arg(&t_i);
                        builder.arg(&di_i);
                        builder.arg(&ts_i);
                        builder.arg(&gate_stride);
                        builder.arg(&y_first);
                        builder.arg(&z_first);
                        unsafe { builder.launch(cfg) }.map_err(|e| {
                            format!("gate_from_tmajor target L{layer_idx}: {:?}", e)
                        })?;
                    }
                }
            } else {
                let width = seq_scan_index_width(b, t, di, ds, 2 * di, false);
                let mut builder = ctx
                    .stream
                    .launch_builder(&ctx.kernels.ssm_burnin_nosave_typed.get(width).f32);
                builder.arg(&ssm_ptr);
                builder.arg(scratch.gated.inner_mut());
                builder.arg(scratch.delta.inner());
                builder.arg(scratch.u.inner());
                builder.arg(scratch.b_gathered.inner());
                builder.arg(scratch.c_gathered.inner());
                builder.arg(&a_neg_ptr);
                let dp_ptr = lw.d_param.cached_ptr();
                builder.arg(&dp_ptr);
                builder.arg(scratch.proj_flat.inner());
                builder.arg(&gate_stride);
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&di_i);
                builder.arg(&ds_i);
                let lanes_i = crate::gpu::launch::seq_lanes(ds) as i32;
                builder.arg(&lanes_i);
                unsafe { builder.launch(crate::gpu::launch::grid_seq(b, di, ds)?) }
                    .map_err(|e| format!("ssm_nosave target L{layer_idx}: {:?}", e))?;
            }
        }

        // === F5: out_proj SGEMM [B*T, di] -> [B*T, dm] ===
        gpu_gemm_bi_forward_raw(
            ctx,
            &mut scratch.out_flat,
            &scratch.gated,
            lw.out_proj_w.cached_ptr(),
            lw.out_proj_b_ptr(),
            (bt, di, dm),
        )?;

        // === F6: residual add — temporal = residual + out_proj ===
        // out_flat already has out_proj result. Add residual in-place.
        crate::gpu::launch::launch_elems(
            &ctx.stream,
            &ctx.kernels.vec_add_inplace,
            &[scratch.out_flat.cached_ptr(), scratch.residual.cached_ptr()],
            bt * dm,
            format_args!("residual target L{layer_idx}"),
        )?;

        // The gated MLP after the mixer over out_flat's rows, the residual
        // buffer, dead until the next layer saves into it, taking the norm
        // and the down projection.
        if let Some(m) = target_w.mlp.get(layer_idx) {
            let rms = scratch.rms_discard.cached_ptr();
            let bufs =
                super::mlp::branch_bufs(scratch.mlp.as_mut(), (bt, m.hidden()), "target scratch")?;
            super::mlp::target_f32(
                ctx,
                m,
                (&mut scratch.out_flat, &mut scratch.residual, rms),
                bufs,
                (bt, dm, dims.rms_norm_eps),
            )?;
        }
    }

    // The final norm (norm_f) after all Mamba layers — applied to all B*T
    // timesteps, the residual buffer its input, rms_discard its statistics.
    scratch.residual.copy_from(&scratch.out_flat, &ctx.stream)?;
    block_norm::forward(
        ctx,
        WeightDtype::F32,
        (
            scratch.out_flat.cached_ptr(),
            scratch.rms_discard.cached_ptr(),
            scratch.residual.cached_ptr(),
        ),
        target_w.norm_f(),
        (bt, dm, dims.rms_norm_eps),
        "norm_f target",
    )?;

    // Extract last timestep: out_flat[b*(T-1)+..] → target_temporal[b*dm]
    {
        let b_i = b as i32;
        let t_i = t as i32;
        let dm_i = dm as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.gather_last_timestep);
        builder.arg(target_temporal.inner_mut());
        builder.arg(scratch.out_flat.inner());
        builder.arg(&b_i);
        builder.arg(&t_i);
        builder.arg(&dm_i);
        unsafe { builder.launch(grid_1d_cover(b * dm)?) }
            .map_err(|e| format!("gather_last target: {:?}", e))?;
    }

    Ok(())
}
