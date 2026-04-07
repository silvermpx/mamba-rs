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
use super::blas::gpu_sgemm_forward_raw;
use super::buffers::GpuBuffer;
use super::context::GpuCtx;
use super::launch::{grid_1d, grid_norm, grid_parallel_scan};
use super::weights::{GpuMambaTrainLayerWeights, GpuMambaTrainWeights};
use cudarc::driver::PushKernelArg;
use std::sync::Arc;

/// Sequence length threshold for switching from sequential SSM to parallel prefix scan.
///
/// For T <= 256, the sequential kernel (one thread per (b,d) pair, looping over T) is
/// faster due to lower overhead and better register utilization. For T > 256, the
/// parallel prefix scan kernel provides O(T / 128) depth per thread, giving significant
/// speedups on long sequences (e.g., 3-4x at T=1024, 6-8x at T=2048).
///
/// The parallel scan launches (batch * d_inner) blocks of 128 threads with shared memory,
/// vs the sequential kernel which launches ceil(batch * d_inner / 256) blocks of 256 threads.
/// The crossover point depends on GPU occupancy; 256 is a conservative safe choice.
const PARALLEL_SCAN_THRESHOLD: usize = 256;

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
}

impl GpuMambaDims {
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
    // -- F1: RmsNorm --
    /// Pre-norm input saved for residual add `[B*T*d_model]`.
    pub residual: GpuBuffer,
    /// RMS scalar per (b,t) pair `[B*T]`.
    pub rms_vals: GpuBuffer,
    /// Post-norm output `[B*T*d_model]`.
    pub post_norm: GpuBuffer,

    // -- F3: Split + gate --
    /// Gate branch before SiLU `[B*T*d_inner]`.
    pub gate_pre_silu: GpuBuffer,
    /// Gate branch after SiLU `[B*T*d_inner]`.
    pub gate_post_silu: GpuBuffer,

    // -- F4a: Conv1d + SiLU --
    /// Conv1d state saved after each step `[B*T*d_inner*d_conv]`.
    pub conv_states: GpuBuffer,
    /// Pre-SiLU conv output `[B*T*d_inner]`.
    pub post_conv: GpuBuffer,
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
    /// Discretization exponential `exp(delta * A)` `[B*T*d_inner*d_state]`.
    pub da_exp: GpuBuffer,
    /// SSM output before gating `[B*T*d_inner]`.
    pub y: GpuBuffer,

    // -- F4e: Gating --
    /// Gated output `y * gate_silu` `[B*T*d_inner]`.
    pub gated: GpuBuffer,
}

/// Full Mamba backbone activations on GPU.
pub struct GpuMambaBackboneActs {
    /// Saved input to input_proj: `[B*T*mamba_input_dim]`.
    pub input_proj_inputs: GpuBuffer,
    /// Saved output of input_proj: `[B*T*d_model]`.
    pub input_proj_outputs: GpuBuffer,
    /// Per-layer activations.
    pub layers: Vec<GpuMambaLayerActs>,
    /// Saved pre-norm_f input for backward: `[B*T*d_model]`.
    pub norm_f_input: GpuBuffer,
    /// Saved RMS values per sample for norm_f backward: `[B*T]`.
    pub norm_f_rms: GpuBuffer,
}

impl GpuMambaBackboneActs {
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

        let layers = (0..n_layers)
            .map(|_| {
                Ok(GpuMambaLayerActs {
                    // F1: RmsNorm
                    residual: GpuBuffer::zeros(stream, bt * d_model)?,
                    rms_vals: GpuBuffer::zeros(stream, bt)?,
                    post_norm: GpuBuffer::zeros(stream, bt * d_model)?,
                    // F3: Split + gate
                    gate_pre_silu: GpuBuffer::zeros(stream, bt * d_inner)?,
                    gate_post_silu: GpuBuffer::zeros(stream, bt * d_inner)?,
                    // F4a: Conv1d + SiLU
                    conv_states: GpuBuffer::zeros(stream, bt * d_inner * d_conv)?,
                    post_conv: GpuBuffer::zeros(stream, bt * d_inner)?,
                    u: GpuBuffer::zeros(stream, bt * d_inner)?,
                    // F4b-c: x_proj + dt_proj
                    xdbl: GpuBuffer::zeros(stream, bt * xdbl_dim)?,
                    delta_raw: GpuBuffer::zeros(stream, bt * d_inner)?,
                    delta: GpuBuffer::zeros(stream, bt * d_inner)?,
                    // F4d: SSM — h_saved has T+1 entries
                    h_saved: GpuBuffer::zeros(stream, batch * (seq_len + 1) * d_inner * d_state)?,
                    da_exp: GpuBuffer::zeros(stream, bt * d_inner * d_state)?,
                    y: GpuBuffer::zeros(stream, bt * d_inner)?,
                    // F4e: Gating
                    gated: GpuBuffer::zeros(stream, bt * d_inner)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(Self {
            input_proj_inputs: GpuBuffer::zeros(stream, bt * mamba_input_dim)?,
            input_proj_outputs: GpuBuffer::zeros(stream, bt * d_model)?,
            layers,
            norm_f_input: GpuBuffer::zeros(stream, bt * d_model)?,
            norm_f_rms: GpuBuffer::zeros(stream, bt)?,
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
    /// In-proj output (x + gate concatenated) `[B*T * 2*d_inner]`.
    pub proj_flat: GpuBuffer,
    /// x_branch after split `[B*T*d_inner]`.
    pub x_branch: GpuBuffer,
    /// Out-proj output `[B*T*d_model]`.
    pub out_flat: GpuBuffer,
    /// Gathered dt portion of xdbl for dt_proj SGEMM `[B*T*dt_rank]`.
    /// Separate from backward `d_dt_input` to avoid forward/backward collision.
    pub dt_gather_buf: GpuBuffer,

    // -- Backward scratch --
    /// Gradient of gated output `[B*T*d_inner]`.
    pub d_gated: GpuBuffer,
    /// Gradient of SSM output y `[B*T*d_inner]`.
    pub d_y: GpuBuffer,
    /// Gradient of gate branch `[B*T*d_inner]`.
    pub d_gate: GpuBuffer,
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
    /// Gradient of x_branch `[B*T*d_inner]`.
    pub d_x_branch: GpuBuffer,
    /// Gradient of in_proj output `[B*T*2*d_inner]`.
    pub d_proj: GpuBuffer,
    /// Gradient of post-norm activations `[B*T*d_model]`.
    pub d_norm: GpuBuffer,
    /// Gradient of pre-norm input `[B*T*d_model]`.
    pub d_pre_norm: GpuBuffer,
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
}

impl GpuMambaScratch {
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

        Ok(Self {
            dims: *dims,
            // Forward scratch
            proj_flat: GpuBuffer::zeros(stream, bt * 2 * d_inner)?,
            x_branch: GpuBuffer::zeros(stream, bt * d_inner)?,
            out_flat: GpuBuffer::zeros(stream, bt * d_model)?,
            dt_gather_buf: GpuBuffer::zeros(stream, bt * dt_rank)?,
            // Backward scratch
            d_gated: GpuBuffer::zeros(stream, bt * d_inner)?,
            d_y: GpuBuffer::zeros(stream, bt * d_inner)?,
            d_gate: GpuBuffer::zeros(stream, bt * d_inner)?,
            d_delta: GpuBuffer::zeros(stream, bt * d_inner)?,
            d_delta_raw: GpuBuffer::zeros(stream, bt * d_inner)?,
            d_u: GpuBuffer::zeros(stream, bt * d_inner)?,
            d_u_xproj: GpuBuffer::zeros(stream, bt * d_inner)?,
            d_xdbl: GpuBuffer::zeros(stream, bt * xdbl_dim)?,
            d_x_branch: GpuBuffer::zeros(stream, bt * d_inner)?,
            d_proj: GpuBuffer::zeros(stream, bt * 2 * d_inner)?,
            d_norm: GpuBuffer::zeros(stream, bt * d_model)?,
            d_pre_norm: GpuBuffer::zeros(stream, bt * d_model)?,
            d_dt_input: GpuBuffer::zeros(stream, bt * dt_rank)?,
            dt_xdbl_buf: GpuBuffer::zeros(stream, bt * dt_rank)?,
            // SSM backward per-thread buffers
            d_b_local: GpuBuffer::zeros(stream, bt * d_inner * d_state)?,
            d_c_local: GpuBuffer::zeros(stream, bt * d_inner * d_state)?,
            d_d_local: GpuBuffer::zeros(stream, batch * d_inner)?,
            d_a_log_local: GpuBuffer::zeros(stream, batch * d_inner * d_state)?,
            d_b_reduced: GpuBuffer::zeros(stream, bt * d_state)?,
            d_c_reduced: GpuBuffer::zeros(stream, bt * d_state)?,
            // Discarded dx for input_proj backward
            d_input_proj_dx: GpuBuffer::zeros(stream, bt * mamba_input_dim)?,
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
    temporal: &mut GpuBuffer,
    acts: &mut GpuMambaLayerActs,
    lw: &GpuMambaTrainLayerWeights,
    layer_ptrs: &MambaLayerPtrs,
    scratch: &mut GpuMambaScratch,
) -> Result<(), String> {
    let dims = scratch.dims;
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
    // F1: RmsNorm — save residual, compute post_norm
    // ===================================================================
    // Save temporal → acts.residual before normalization
    acts.residual.copy_from(temporal, &ctx.stream)?;

    // rmsnorm_forward(y, rms_out, x, scale, batch, dim, eps)
    {
        let batch_i = bt as i32;
        let dim_i = dm as i32;
        let eps: f32 = 1e-5;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.rmsnorm_fwd);
        builder.arg(acts.post_norm.inner_mut());
        builder.arg(acts.rms_vals.inner_mut());
        builder.arg(temporal.inner());
        let nw_ptr = lw.norm_weight.cached_ptr();
        builder.arg(&nw_ptr);
        builder.arg(&batch_i);
        builder.arg(&dim_i);
        builder.arg(&eps);
        unsafe { builder.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("rmsnorm_fwd mamba: {:?}", e))?;
    }

    // ===================================================================
    // F2: Batch in_proj -- ONE cuBLAS SGEMM
    //     [B*T, d_model] -> [B*T, 2*d_inner]
    // ===================================================================
    gpu_sgemm_forward_raw(
        ctx,
        &mut scratch.proj_flat,
        &acts.post_norm,
        lw.in_proj_w.cached_ptr(),
        None,
        (bt, dm, 2 * di),
    )?;

    // ===================================================================
    // F3: Split x/gate + SiLU(gate)
    // ===================================================================
    // split_gate_silu(x_branch, gate_pre_silu, gate_post_silu, proj, batch, d_inner)
    {
        let batch_i = bt as i32;
        let di_i = di as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.split_gate_silu);
        builder.arg(scratch.x_branch.inner_mut());
        builder.arg(acts.gate_pre_silu.inner_mut());
        builder.arg(acts.gate_post_silu.inner_mut());
        builder.arg(scratch.proj_flat.inner());
        builder.arg(&batch_i);
        builder.arg(&di_i);
        unsafe { builder.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("split_gate_silu mamba: {:?}", e))?;
    }

    // ===================================================================
    // F4a: Conv1d burnin + fused SiLU
    // ===================================================================
    // conv1d_burnin_forward(u_out, post_conv_out, conv_states_out, state, x_branch, weight, bias,
    //                       batch, T, d_inner, d_conv)
    {
        let b_i = b as i32;
        let t_i = t as i32;
        let di_i = di as i32;
        let dc_i = d_conv as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.conv1d_burnin_fwd);
        builder.arg(acts.u.inner_mut());
        builder.arg(acts.post_conv.inner_mut());
        builder.arg(acts.conv_states.inner_mut());
        builder.arg(&layer_ptrs.conv_state); // raw ptr at layer offset
        builder.arg(scratch.x_branch.inner());
        let cw_ptr = lw.conv1d_weight.cached_ptr();
        let cb_ptr = lw.conv1d_bias.cached_ptr();
        builder.arg(&cw_ptr);
        builder.arg(&cb_ptr);
        builder.arg(&b_i);
        builder.arg(&t_i);
        builder.arg(&di_i);
        builder.arg(&dc_i);
        unsafe { builder.launch(grid_1d(b * di)) }
            .map_err(|e| format!("conv1d_burnin_fwd mamba: {:?}", e))?;
    }

    // ===================================================================
    // F4b: x_proj -- ONE cuBLAS SGEMM
    //      [B*T, d_inner] -> [B*T, xdbl_dim]
    // ===================================================================
    gpu_sgemm_forward_raw(
        ctx,
        &mut acts.xdbl,
        &acts.u,
        lw.x_proj_w.cached_ptr(),
        None,
        (bt, di, xdbl_dim),
    )?;

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
        unsafe { builder.launch(grid_1d(bt * dt_rank)) }
            .map_err(|e| format!("gather_cols dt mamba: {:?}", e))?;
    }

    gpu_sgemm_forward_raw(
        ctx,
        &mut acts.delta_raw,
        &scratch.dt_gather_buf,
        lw.dt_proj_w.cached_ptr(),
        Some(lw.dt_proj_b.cached_ptr()),
        (bt, dt_rank, di),
    )?;

    // Softplus: delta = softplus(delta_raw) — fused copy+softplus, no memcpy
    {
        let n = (bt * di) as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.softplus_copy);
        builder.arg(acts.delta.inner_mut());
        builder.arg(acts.delta_raw.inner());
        builder.arg(&n);
        unsafe { builder.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("softplus_copy mamba: {:?}", e))?;
    }

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
    // Fused gather B+C from xdbl (saves 1 kernel launch vs 2 separate gather_cols)
    {
        let bt_i = bt as i32;
        let xdbl_i = xdbl_dim as i32;
        let ds_i = ds as i32;
        let b_offset = dt_rank as i32;
        let c_offset = (dt_rank + ds) as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.gather_bc_cols);
        builder.arg(scratch.d_b_reduced.inner_mut());
        builder.arg(scratch.d_c_reduced.inner_mut());
        builder.arg(acts.xdbl.inner());
        builder.arg(&bt_i);
        builder.arg(&xdbl_i);
        builder.arg(&ds_i);
        builder.arg(&b_offset);
        builder.arg(&c_offset);
        unsafe { builder.launch(grid_1d(bt * ds)) }
            .map_err(|e| format!("gather_bc_cols fwd mamba: {:?}", e))?;
    }

    // SSM burnin forward: use parallel prefix scan for long sequences,
    // sequential kernel for short ones. Threshold: T > 256.
    // Both kernels have identical interfaces and produce identical results.
    // EXCEPTION: sequential kernel silently fails for d_state > 64 (register
    // array limit), so we force the parallel scan path which handles d_state
    // up to MAX_DSTATE=256 via its outer loop.
    {
        let b_i = b as i32;
        let t_i = t as i32;
        let di_i = di as i32;
        let ds_i = ds as i32;

        if t > PARALLEL_SCAN_THRESHOLD || ds > 64 {
            // Parallel prefix scan: O(T / NTHREADS) per thread instead of O(T).
            // Grid: (batch, d_inner) — one block per (b, d) pair.
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.ssm_parallel_fwd);
            builder.arg(&layer_ptrs.ssm_state);
            builder.arg(acts.y.inner_mut());
            builder.arg(acts.h_saved.inner_mut());
            builder.arg(acts.da_exp.inner_mut());
            builder.arg(acts.delta.inner());
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
            unsafe { builder.launch(grid_parallel_scan(b, di)) }
                .map_err(|e| format!("ssm_parallel_fwd mamba: {:?}", e))?;
        } else {
            // Sequential kernel: each thread handles one (b, d) pair over all T steps.
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.ssm_burnin_fwd);
            builder.arg(&layer_ptrs.ssm_state);
            builder.arg(acts.y.inner_mut());
            builder.arg(acts.h_saved.inner_mut());
            builder.arg(acts.da_exp.inner_mut());
            builder.arg(acts.delta.inner());
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
            unsafe { builder.launch(grid_1d(b * di)) }
                .map_err(|e| format!("ssm_burnin_fwd mamba: {:?}", e))?;
        }
    }

    // ===================================================================
    // F4e: Gating — gated = y * gate_post_silu
    // ===================================================================
    {
        let n = (bt * di) as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.elementwise_mul);
        builder.arg(acts.gated.inner_mut());
        builder.arg(acts.y.inner());
        builder.arg(acts.gate_post_silu.inner());
        builder.arg(&n);
        unsafe { builder.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("elementwise_mul gating mamba: {:?}", e))?;
    }

    // ===================================================================
    // F5: Batch out_proj — ONE cuBLAS SGEMM
    //     [B*T, d_inner] -> [B*T, d_model]
    // ===================================================================
    gpu_sgemm_forward_raw(
        ctx,
        &mut scratch.out_flat,
        &acts.gated,
        lw.out_proj_w.cached_ptr(),
        None,
        (bt, di, dm),
    )?;

    // ===================================================================
    // F6: Residual add — temporal = residual + out_flat (fused, no memcpy)
    // ===================================================================
    {
        let n = (bt * dm) as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.residual_add);
        builder.arg(temporal.inner_mut());
        builder.arg(acts.residual.inner());
        builder.arg(scratch.out_flat.inner());
        builder.arg(&n);
        unsafe { builder.launch(grid_1d(bt * dm)) }
            .map_err(|e| format!("residual_add mamba: {:?}", e))?;
    }

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
    let bt = dims.bt();

    // Save input for backward
    acts.input_proj_inputs.copy_from(mamba_input, &ctx.stream)?;

    // Batched input_proj: [B*T, mamba_input_dim] -> [B*T, d_model]
    gpu_sgemm_forward_raw(
        ctx,
        temporal,
        mamba_input,
        mamba_w.input_proj_w.cached_ptr(),
        Some(mamba_w.input_proj_b.cached_ptr()),
        (bt, dims.mamba_input_dim, dims.d_model),
    )?;

    // Save output for backward
    acts.input_proj_outputs.copy_from(temporal, &ctx.stream)?;

    // Mamba layers — per-layer offset into flat state buffers
    // Matches official Mamba pattern: single flat allocation, per-layer kernel invocation.
    // Same pattern as CPU train/forward.rs
    let conv_per_layer = dims.batch * dims.d_inner * dims.d_conv;
    let ssm_per_layer = dims.batch * dims.d_inner * dims.d_state;
    let a_neg_per_layer = dims.d_inner * dims.d_state;

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

        gpu_forward_mamba_layer(
            ctx,
            temporal,
            &mut acts.layers[layer_idx],
            &mamba_w.layers[layer_idx],
            &layer_ptrs,
            scratch,
        )?;
        // All kernels for this layer have been launched on stream
    }

    // Final RmsNorm (norm_f) after all Mamba layers
    {
        let bt_i = bt as i32;
        let dm_i = dims.d_model as i32;
        let eps: f32 = 1e-5;
        // Save pre-norm input for backward
        acts.norm_f_input.copy_from(temporal, &ctx.stream)?;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.rmsnorm_fwd);
        builder.arg(temporal.inner_mut()); // normed output (in-place)
        builder.arg(acts.norm_f_rms.inner_mut()); // rms scalars (saved for backward)
        builder.arg(acts.norm_f_input.inner()); // input (saved copy)
        let nf_ptr = mamba_w.norm_f_weight.cached_ptr();
        builder.arg(&nf_ptr);
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        builder.arg(&eps);
        unsafe { builder.launch(grid_norm(bt, dims.d_model)) }
            .map_err(|e| format!("rmsnorm_fwd norm_f: {:?}", e))?;
    }

    Ok(())
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
    let dims = &scratch.dims;
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

        // === F1: RmsNorm [B*T] ===
        scratch.residual.copy_from(&scratch.out_flat, &ctx.stream)?;
        {
            let bt_i = bt as i32;
            let dm_i = dm as i32;
            let eps: f32 = 1e-5;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.rmsnorm_fwd);
            builder.arg(scratch.out_flat.inner_mut()); // normed output
            builder.arg(scratch.rms_discard.inner_mut()); // rms scalars (discarded)
            builder.arg(scratch.residual.inner()); // input
            let nw_ptr = lw.norm_weight.cached_ptr();
            builder.arg(&nw_ptr);
            builder.arg(&bt_i);
            builder.arg(&dm_i);
            builder.arg(&eps);
            unsafe { builder.launch(grid_norm(bt, dm)) }
                .map_err(|e| format!("rmsnorm target L{layer_idx}: {:?}", e))?;
        }

        // === F2: in_proj SGEMM [B*T, dm] -> [B*T, 2*di] ===
        gpu_sgemm_forward_raw(
            ctx,
            &mut scratch.proj_flat,
            &scratch.out_flat,
            lw.in_proj_w.cached_ptr(),
            None,
            (bt, dm, 2 * di),
        )?;

        // === F3: split x + SiLU(gate) [B*T] ===
        {
            let bt_i = bt as i32;
            let di_i = di as i32;
            let gs_raw = scratch.gate_silu.raw_ptr(&ctx.stream);
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.split_gate_silu);
            builder.arg(scratch.x_branch.inner_mut());
            builder.arg(scratch.gate_silu.inner_mut()); // gate_pre_silu (discarded)
            builder.arg(&gs_raw); // gate_post_silu writes to same buffer
            builder.arg(scratch.proj_flat.inner());
            builder.arg(&bt_i);
            builder.arg(&di_i);
            unsafe { builder.launch(grid_1d(bt * di)) }
                .map_err(|e| format!("split_gate target L{layer_idx}: {:?}", e))?;
        }

        // === F4a: conv1d burnin nosave + fused SiLU [all T, parallel B*d_inner] ===
        {
            let b_i = b as i32;
            let t_i = t as i32;
            let di_i = di as i32;
            let dc_i = d_conv as i32;
            let mut builder = ctx
                .stream
                .launch_builder(&ctx.kernels.conv1d_burnin_fwd_nosave);
            builder.arg(scratch.u.inner_mut()); // post-SiLU output [B*T*di]
            builder.arg(&conv_ptr); // per-layer state
            builder.arg(scratch.x_branch.inner());
            let cw_ptr = lw.conv1d_weight.cached_ptr();
            let cb_ptr = lw.conv1d_bias.cached_ptr();
            builder.arg(&cw_ptr);
            builder.arg(&cb_ptr);
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&di_i);
            builder.arg(&dc_i);
            unsafe { builder.launch(grid_1d(b * di)) }
                .map_err(|e| format!("conv1d_nosave target L{layer_idx}: {:?}", e))?;
        }

        // === F4b: x_proj SGEMM [B*T, di] -> [B*T, xdbl_dim] ===
        gpu_sgemm_forward_raw(
            ctx,
            &mut scratch.xdbl,
            &scratch.u,
            lw.x_proj_w.cached_ptr(),
            None,
            (bt, di, xdbl_dim),
        )?;

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
            unsafe { builder.launch(grid_1d(bt * dt_rank)) }
                .map_err(|e| format!("gather dt target L{layer_idx}: {:?}", e))?;
        }
        gpu_sgemm_forward_raw(
            ctx,
            &mut scratch.delta,
            &scratch.dt_gather,
            lw.dt_proj_w.cached_ptr(),
            Some(lw.dt_proj_b.cached_ptr()),
            (bt, dt_rank, di),
        )?;
        {
            let n = (bt * di) as i32;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.softplus_fwd);
            builder.arg(scratch.delta.inner_mut());
            builder.arg(&n);
            unsafe { builder.launch(grid_1d(bt * di)) }
                .map_err(|e| format!("softplus target L{layer_idx}: {:?}", e))?;
        }

        // === F4d: gather B/C + SSM burnin nosave ===
        {
            let bt_i = bt as i32;
            let xdbl_i = xdbl_dim as i32;
            let ds_i = ds as i32;
            let b_offset = dt_rank as i32;
            let c_offset = (dt_rank + ds) as i32;
            // Fused gather B+C from xdbl (saves 1 kernel launch)
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.gather_bc_cols);
            builder.arg(scratch.b_gathered.inner_mut());
            builder.arg(scratch.c_gathered.inner_mut());
            builder.arg(scratch.xdbl.inner());
            builder.arg(&bt_i);
            builder.arg(&xdbl_i);
            builder.arg(&ds_i);
            builder.arg(&b_offset);
            builder.arg(&c_offset);
            unsafe { builder.launch(grid_1d(bt * ds)) }
                .map_err(|e| format!("gather_bc_cols target L{layer_idx}: {:?}", e))?;
        }
        // SSM burnin nosave: parallel scan for T > threshold, sequential otherwise.
        // EXCEPTION: sequential kernel silently fails for d_state > 64 (register
        // array limit), so we force the parallel scan path which handles d_state
        // up to MAX_DSTATE=256 via its outer loop.
        {
            let b_i = b as i32;
            let t_i = t as i32;
            let di_i = di as i32;
            let ds_i = ds as i32;

            if t > PARALLEL_SCAN_THRESHOLD || ds > 64 {
                let mut builder = ctx
                    .stream
                    .launch_builder(&ctx.kernels.ssm_parallel_fwd_nosave);
                builder.arg(&ssm_ptr);
                builder.arg(scratch.y.inner_mut());
                builder.arg(scratch.delta.inner());
                builder.arg(scratch.u.inner());
                builder.arg(scratch.b_gathered.inner());
                builder.arg(scratch.c_gathered.inner());
                builder.arg(&a_neg_ptr);
                let dp_ptr = lw.d_param.cached_ptr();
                builder.arg(&dp_ptr);
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&di_i);
                builder.arg(&ds_i);
                unsafe { builder.launch(grid_parallel_scan(b, di)) }
                    .map_err(|e| format!("ssm_parallel_nosave target L{layer_idx}: {:?}", e))?;
            } else {
                let mut builder = ctx
                    .stream
                    .launch_builder(&ctx.kernels.ssm_burnin_fwd_nosave);
                builder.arg(&ssm_ptr);
                builder.arg(scratch.y.inner_mut());
                builder.arg(scratch.delta.inner());
                builder.arg(scratch.u.inner());
                builder.arg(scratch.b_gathered.inner());
                builder.arg(scratch.c_gathered.inner());
                builder.arg(&a_neg_ptr);
                let dp_ptr = lw.d_param.cached_ptr();
                builder.arg(&dp_ptr);
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&di_i);
                builder.arg(&ds_i);
                unsafe { builder.launch(grid_1d(b * di)) }
                    .map_err(|e| format!("ssm_nosave target L{layer_idx}: {:?}", e))?;
            }
        }

        // === F4e: gating [B*T] — y * gate_silu ===
        {
            let n = (bt * di) as i32;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.elementwise_mul);
            builder.arg(scratch.gated.inner_mut());
            builder.arg(scratch.y.inner());
            builder.arg(scratch.gate_silu.inner());
            builder.arg(&n);
            unsafe { builder.launch(grid_1d(bt * di)) }
                .map_err(|e| format!("gating target L{layer_idx}: {:?}", e))?;
        }

        // === F5: out_proj SGEMM [B*T, di] -> [B*T, dm] ===
        gpu_sgemm_forward_raw(
            ctx,
            &mut scratch.out_flat,
            &scratch.gated,
            lw.out_proj_w.cached_ptr(),
            None,
            (bt, di, dm),
        )?;

        // === F6: residual add — temporal = residual + out_proj ===
        // out_flat already has out_proj result. Add residual in-place.
        {
            let n = (bt * dm) as i32;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.vec_add_inplace);
            builder.arg(scratch.out_flat.inner_mut());
            builder.arg(scratch.residual.inner());
            builder.arg(&n);
            unsafe { builder.launch(grid_1d(bt * dm)) }
                .map_err(|e| format!("residual target L{layer_idx}: {:?}", e))?;
        }
    }

    // Final RmsNorm (norm_f) after all Mamba layers — applied to all B*T timesteps
    {
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let eps: f32 = 1e-5;
        // Use residual as temp input buffer, rms_discard for rms scalars
        scratch.residual.copy_from(&scratch.out_flat, &ctx.stream)?;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.rmsnorm_fwd);
        builder.arg(scratch.out_flat.inner_mut()); // normed output
        builder.arg(scratch.rms_discard.inner_mut()); // rms (discarded)
        builder.arg(scratch.residual.inner()); // input
        let tnf_ptr = target_w.norm_f_weight.cached_ptr();
        builder.arg(&tnf_ptr);
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        builder.arg(&eps);
        unsafe { builder.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("rmsnorm_fwd norm_f target: {:?}", e))?;
    }

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
        unsafe { builder.launch(grid_1d(b * dm)) }
            .map_err(|e| format!("gather_last target: {:?}", e))?;
    }

    Ok(())
}
