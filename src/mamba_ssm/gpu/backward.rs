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

use super::blas::gpu_sgemm_backward_grad_raw;
use super::buffers::GpuBuffer;
use super::context::GpuCtx;
use super::forward::{GpuMambaBackboneActs, GpuMambaLayerActs, GpuMambaScratch};
use super::launch::{grid_1d, grid_norm};
use super::weights::{
    GpuMambaGrads, GpuMambaLayerGrads, GpuMambaTrainLayerWeights, GpuMambaTrainWeights,
};
use cudarc::driver::PushKernelArg;
use std::sync::Arc;

// ---------------------------------------------------------------------------

pub fn gpu_backward_mamba_layer(
    ctx: &GpuCtx,
    d_temporal: &mut GpuBuffer,
    d_lw: &GpuMambaLayerGrads,
    acts: &GpuMambaLayerActs,
    lw: &GpuMambaTrainLayerWeights,
    a_neg_ptr: cudarc::driver::sys::CUdeviceptr,
    scratch: &mut GpuMambaScratch,
) -> Result<(), String> {
    let dims = scratch.dims; // Copy (GpuMambaDims is Copy)
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
    // B1: Batch out_proj backward
    // ===================================================================
    gpu_sgemm_backward_grad_raw(
        ctx,
        &mut scratch.d_gated,
        (&d_lw.out_proj_w, None),
        d_temporal,
        &acts.gated,
        lw.out_proj_w.cached_ptr(),
        (bt, di, dm),
    )?;

    // ===================================================================
    // B2: Gating backward
    // ===================================================================
    // gating_backward(d_y, d_gate_pre, d_gated, y, gate_pre, gate_post, n)
    {
        let n = (bt * di) as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.gating_backward);
        builder.arg(scratch.d_y.inner_mut());
        builder.arg(scratch.d_gate.inner_mut());
        builder.arg(scratch.d_gated.inner());
        builder.arg(acts.y.inner());
        builder.arg(acts.gate_pre_silu.inner());
        builder.arg(acts.gate_post_silu.inner());
        builder.arg(&n);
        unsafe { builder.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("gating_backward mamba: {:?}", e))?;
    }

    // ===================================================================
    // B3: SSM BPTT + reductions
    // ===================================================================
    // Gather B and C from xdbl for ssm_backward_local
    {
        let bt_i = bt as i32;
        let xdbl_i = xdbl_dim as i32;
        let ds_i = ds as i32;
        let b_offset = dt_rank as i32;
        let c_offset = (dt_rank + ds) as i32;
        // Fused gather B+C from xdbl (saves 1 kernel launch)
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
            .map_err(|e| format!("gather_bc_cols bwd mamba: {:?}", e))?;
    }

    // C6 fix: zero d_a_log_local before SSM backward (kernel uses += across T)
    scratch.d_a_log_local.zero(&ctx.stream)?;

    // ssm_backward_local(h_saved, delta_saved, u_saved, B_saved, C_saved, a_neg, D,
    //   dy, d_delta, d_u, d_B_local, d_C_local, d_D_local, d_a_log_local,
    //   batch, T, d_inner, d_state)
    {
        let b_i = b as i32;
        let t_i = t as i32;
        let di_i = di as i32;
        let ds_i = ds as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.ssm_backward_local);
        builder.arg(acts.h_saved.inner());
        builder.arg(acts.delta.inner());
        builder.arg(acts.u.inner());
        builder.arg(scratch.d_b_reduced.inner()); // B_saved
        builder.arg(scratch.d_c_reduced.inner()); // C_saved
        builder.arg(&a_neg_ptr); // raw ptr at layer offset
        let dp_ptr = lw.d_param.cached_ptr();
        builder.arg(&dp_ptr);
        builder.arg(scratch.d_y.inner()); // dy (from gating backward)
        builder.arg(scratch.d_delta.inner_mut());
        builder.arg(scratch.d_u.inner_mut());
        builder.arg(scratch.d_b_local.inner_mut());
        builder.arg(scratch.d_c_local.inner_mut());
        builder.arg(scratch.d_d_local.inner_mut());
        builder.arg(scratch.d_a_log_local.inner_mut());
        builder.arg(&b_i);
        builder.arg(&t_i);
        builder.arg(&di_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(b * di)) }
            .map_err(|e| format!("ssm_backward_local mamba: {:?}", e))?;
    }

    // Reductions: sum per-sample gradients across batch/d_inner
    // CRITICAL: zero reduction targets — they held gathered B/C values from ssm_backward_local
    // and reduction kernels use += accumulation (would add stale B/C values to gradients)
    scratch.d_b_reduced.zero(&ctx.stream)?;
    scratch.d_c_reduced.zero(&ctx.stream)?;
    {
        let b_i = b as i32;
        let t_i = t as i32;
        let di_i = di as i32;
        let ds_i = ds as i32;
        // d_B reduction
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.ssm_reduce_d_b);
        builder.arg(scratch.d_b_reduced.inner_mut());
        builder.arg(scratch.d_b_local.inner());
        builder.arg(&b_i);
        builder.arg(&t_i);
        builder.arg(&di_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(bt * ds)) }
            .map_err(|e| format!("ssm_reduce_d_B mamba: {:?}", e))?;
        // d_C reduction
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.ssm_reduce_d_c);
        builder.arg(scratch.d_c_reduced.inner_mut());
        builder.arg(scratch.d_c_local.inner());
        builder.arg(&b_i);
        builder.arg(&t_i);
        builder.arg(&di_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(bt * ds)) }
            .map_err(|e| format!("ssm_reduce_d_C mamba: {:?}", e))?;
        // d_D reduction
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.ssm_reduce_d_d);
        let _p = d_lw.d_param.ptr();
        builder.arg(&_p);
        builder.arg(scratch.d_d_local.inner());
        builder.arg(&b_i);
        builder.arg(&di_i);
        unsafe { builder.launch(grid_1d(di)) }
            .map_err(|e| format!("ssm_reduce_d_D mamba: {:?}", e))?;
        // d_a_log reduction
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.ssm_reduce_d_a_log);
        let _p = d_lw.a_log.ptr();
        builder.arg(&_p);
        builder.arg(scratch.d_a_log_local.inner());
        builder.arg(&b_i);
        builder.arg(&di_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(di * ds)) }
            .map_err(|e| format!("ssm_reduce_d_a_log mamba: {:?}", e))?;
    }

    // Pack d_b_reduced, d_c_reduced into d_xdbl at the right offsets
    // Zero d_xdbl first (dt portion will be filled by scatter_add_cols)
    scratch.d_xdbl.zero(&ctx.stream)?;
    {
        let bt_i = bt as i32;
        let xdbl_i = xdbl_dim as i32;
        let ds_i = ds as i32;
        let b_offset = dt_rank as i32;
        let c_offset = (dt_rank + ds) as i32;
        // Scatter d_B into d_xdbl at offset dt_rank
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.scatter_add_cols);
        builder.arg(scratch.d_xdbl.inner_mut());
        builder.arg(scratch.d_b_reduced.inner());
        builder.arg(&bt_i);
        builder.arg(&xdbl_i);
        builder.arg(&ds_i);
        builder.arg(&b_offset);
        unsafe { builder.launch(grid_1d(bt * ds)) }
            .map_err(|e| format!("scatter d_B mamba: {:?}", e))?;
        // Scatter d_C into d_xdbl at offset dt_rank + d_state
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.scatter_add_cols);
        builder.arg(scratch.d_xdbl.inner_mut());
        builder.arg(scratch.d_c_reduced.inner());
        builder.arg(&bt_i);
        builder.arg(&xdbl_i);
        builder.arg(&ds_i);
        builder.arg(&c_offset);
        unsafe { builder.launch(grid_1d(bt * ds)) }
            .map_err(|e| format!("scatter d_C mamba: {:?}", e))?;
    }

    // ===================================================================
    // B4: Softplus backward + dt_proj backward
    // ===================================================================
    // softplus_backward(dx, x_saved, dy, n)
    {
        let n = (bt * di) as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.softplus_bwd);
        builder.arg(scratch.d_delta_raw.inner_mut()); // dx output
        builder.arg(acts.delta_raw.inner()); // x_saved (pre-softplus activation)
        builder.arg(scratch.d_delta.inner()); // dy (upstream gradient from SSM)
        builder.arg(&n);
        unsafe { builder.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("softplus_bwd mamba: {:?}", e))?;
    }

    // Gather xdbl dt portion for x_saved (separate buffer from dx output)
    {
        let bt_i = bt as i32;
        let xdbl_i = xdbl_dim as i32;
        let dt_i = dt_rank as i32;
        let offset: i32 = 0;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.gather_cols);
        builder.arg(scratch.dt_xdbl_buf.inner_mut());
        builder.arg(acts.xdbl.inner());
        builder.arg(&bt_i);
        builder.arg(&xdbl_i);
        builder.arg(&dt_i);
        builder.arg(&offset);
        unsafe { builder.launch(grid_1d(bt * dt_rank)) }
            .map_err(|e| format!("gather dt x_saved mamba: {:?}", e))?;
    }

    gpu_sgemm_backward_grad_raw(
        ctx,
        &mut scratch.d_dt_input, // dx [B*T*dt_rank]
        (&d_lw.dt_proj_w, Some(&d_lw.dt_proj_b)),
        &scratch.d_delta_raw, // dy [B*T*d_inner]
        &scratch.dt_xdbl_buf, // x_saved [B*T*dt_rank]
        lw.dt_proj_w.cached_ptr(),
        (bt, dt_rank, di),
    )?;

    // Scatter-add d_dt_input into d_xdbl at offset 0
    {
        let bt_i = bt as i32;
        let xdbl_i = xdbl_dim as i32;
        let dt_i = dt_rank as i32;
        let offset: i32 = 0;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.scatter_add_cols);
        builder.arg(scratch.d_xdbl.inner_mut());
        builder.arg(scratch.d_dt_input.inner());
        builder.arg(&bt_i);
        builder.arg(&xdbl_i);
        builder.arg(&dt_i);
        builder.arg(&offset);
        unsafe { builder.launch(grid_1d(bt * dt_rank)) }
            .map_err(|e| format!("scatter dt bwd mamba: {:?}", e))?;
    }

    // ===================================================================
    // B5: x_proj backward
    // ===================================================================
    gpu_sgemm_backward_grad_raw(
        ctx,
        &mut scratch.d_u_xproj,
        (&d_lw.x_proj_w, None),
        &scratch.d_xdbl,
        &acts.u,
        lw.x_proj_w.cached_ptr(),
        (bt, di, xdbl_dim),
    )?;

    // Accumulate d_u += d_u_xproj
    {
        let n = (bt * di) as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.vec_add_inplace);
        builder.arg(scratch.d_u.inner_mut());
        builder.arg(scratch.d_u_xproj.inner());
        builder.arg(&n);
        unsafe { builder.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("vec_add d_u mamba: {:?}", e))?;
    }

    // ===================================================================
    // B6: SiLU backward + Conv1d backward (fused burnin kernel)
    // ===================================================================
    // conv1d_burnin_backward(d_x_branch, d_weight, d_bias, d_u, post_conv, conv_states, weight,
    //                        batch, T, d_inner, d_conv)
    {
        let b_i = b as i32;
        let t_i = t as i32;
        let di_i = di as i32;
        let dc_i = d_conv as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.conv1d_burnin_bwd);
        builder.arg(scratch.d_x_branch.inner_mut());
        let _p = d_lw.conv1d_weight.ptr();
        builder.arg(&_p);
        let _p = d_lw.conv1d_bias.ptr();
        builder.arg(&_p);
        builder.arg(scratch.d_u.inner());
        builder.arg(acts.post_conv.inner());
        builder.arg(acts.conv_states.inner());
        let cw_ptr = lw.conv1d_weight.cached_ptr();
        builder.arg(&cw_ptr);
        builder.arg(&b_i);
        builder.arg(&t_i);
        builder.arg(&di_i);
        builder.arg(&dc_i);
        unsafe { builder.launch(grid_1d(b * di)) }
            .map_err(|e| format!("conv1d_burnin_bwd mamba: {:?}", e))?;
    }

    // ===================================================================
    // B7: Batch in_proj backward
    // ===================================================================
    // Concat d_x_branch || d_gate -> d_proj
    {
        let bt_i = bt as i32;
        let di_i = di as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.concat_halves);
        builder.arg(scratch.d_proj.inner_mut());
        builder.arg(scratch.d_x_branch.inner());
        builder.arg(scratch.d_gate.inner());
        builder.arg(&bt_i);
        builder.arg(&di_i);
        unsafe { builder.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("concat_halves bwd mamba: {:?}", e))?;
    }

    gpu_sgemm_backward_grad_raw(
        ctx,
        &mut scratch.d_norm,
        (&d_lw.in_proj_w, None),
        &scratch.d_proj,
        &acts.post_norm,
        lw.in_proj_w.cached_ptr(),
        (bt, dm, 2 * di),
    )?;

    // ===================================================================
    // B8: RmsNorm backward + residual
    // ===================================================================
    // rmsnorm_backward(dx, d_scale, dy, x, scale, rms_saved, batch, dim)
    {
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let nw_ptr = lw.norm_weight.cached_ptr();
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.rmsnorm_bwd);
        builder.arg(scratch.d_pre_norm.inner_mut());
        let _p = d_lw.norm_weight.ptr();
        builder.arg(&_p);
        builder.arg(scratch.d_norm.inner());
        builder.arg(acts.residual.inner()); // x = input before norm
        builder.arg(&nw_ptr);
        builder.arg(acts.rms_vals.inner());
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        unsafe { builder.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("rmsnorm_bwd mamba: {:?}", e))?;
    }

    // Residual: d_temporal = d_temporal + d_pre_norm
    // d_temporal already has the upstream gradient. Add rmsnorm backward.
    {
        let n = (bt * dm) as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.vec_add_inplace);
        builder.arg(d_temporal.inner_mut());
        builder.arg(scratch.d_pre_norm.inner());
        builder.arg(&n);
        unsafe { builder.launch(grid_1d(bt * dm)) }
            .map_err(|e| format!("vec_add residual bwd mamba: {:?}", e))?;
    }

    Ok(())
}

/// GPU Mamba backbone backward: layers in reverse + input_proj backward.
///
/// Mirrors CPU `backward_mamba_backbone_batched` from train/forward.rs.
pub fn gpu_backward_mamba_backbone(
    ctx: &GpuCtx,
    d_temporal: &mut GpuBuffer,
    d_mamba: &GpuMambaGrads,
    acts: &GpuMambaBackboneActs,
    mamba_w: &GpuMambaTrainWeights,
    a_neg_all: &GpuBuffer,
    scratch: &mut GpuMambaScratch,
) -> Result<(), String> {
    let dims = scratch.dims; // Copy (GpuMambaDims is Copy)
    let bt = dims.bt();

    // norm_f backward — before reverse layer loop
    {
        let bt_i = bt as i32;
        let dm_i = dims.d_model as i32;
        // rmsnorm_backward(dx, d_scale, dy, x, scale, rms_saved, batch, dim)
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.rmsnorm_bwd);
        builder.arg(scratch.d_norm.inner_mut()); // dx (temp [B*T*d_model], will copy back)
        let _p = d_mamba.norm_f_weight.ptr();
        builder.arg(&_p); // d_scale (accumulated)
        builder.arg(d_temporal.inner()); // dy (upstream gradient)
        builder.arg(acts.norm_f_input.inner()); // saved pre-norm input
        let nf_ptr = mamba_w.norm_f_weight.cached_ptr();
        builder.arg(&nf_ptr); // scale
        builder.arg(acts.norm_f_rms.inner()); // saved rms
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        unsafe { builder.launch(grid_norm(bt, dims.d_model)) }
            .map_err(|e| format!("rmsnorm_bwd norm_f: {:?}", e))?;
        // Copy dx back into d_temporal for downstream layer backward
        d_temporal.copy_from(&scratch.d_norm, &ctx.stream)?;
    }

    // Mamba layers in reverse — per-layer a_neg offset
    let a_neg_per_layer = dims.d_inner * dims.d_state;
    for layer_idx in (0..dims.n_layers).rev() {
        let base = a_neg_all.raw_ptr(&ctx.stream);
        let a_neg_ptr = base + (layer_idx * a_neg_per_layer * std::mem::size_of::<f32>()) as u64;

        gpu_backward_mamba_layer(
            ctx,
            d_temporal,
            &d_mamba.layers[layer_idx],
            &acts.layers[layer_idx],
            &mamba_w.layers[layer_idx],
            a_neg_ptr,
            scratch,
        )?;
    }

    // Input projection backward (dx discarded — input embedding detached)
    gpu_sgemm_backward_grad_raw(
        ctx,
        &mut scratch.d_input_proj_dx,
        (&d_mamba.input_proj_w, Some(&d_mamba.input_proj_b)),
        d_temporal,
        &acts.input_proj_inputs,
        mamba_w.input_proj_w.cached_ptr(),
        (bt, dims.mamba_input_dim, dims.d_model),
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// GPU Mamba target forward (batched B*T, nosave kernels)
// ---------------------------------------------------------------------------

/// Scratch buffers for GPU Mamba target forward (batched B*T pipeline).
///
/// Uses nosave burnin kernels (no h_saved/da_exp/conv_states writes).
/// Per-layer conv/SSM state via flat buffers with layer offsets (matches online).
///
/// Uses batched B*T pipeline instead of per-sample step-by-step approach.
pub struct GpuMambaTargetScratch {
    // Batched B*T scratch (reusable from online forward scratch)
    pub proj_flat: GpuBuffer,   // [B*T*2*d_inner]
    pub x_branch: GpuBuffer,    // [B*T*d_inner]
    pub gate_silu: GpuBuffer,   // [B*T*d_inner] gate after SiLU
    pub u: GpuBuffer,           // [B*T*d_inner]
    pub xdbl: GpuBuffer,        // [B*T*xdbl_dim]
    pub dt_gather: GpuBuffer,   // [B*T*dt_rank]
    pub delta: GpuBuffer,       // [B*T*d_inner]
    pub y: GpuBuffer,           // [B*T*d_inner]
    pub gated: GpuBuffer,       // [B*T*d_inner]
    pub out_flat: GpuBuffer,    // [B*T*d_model]
    pub residual: GpuBuffer,    // [B*T*d_model] (saved before RmsNorm)
    pub rms_discard: GpuBuffer, // [B*T] (RmsNorm scalars, discarded)
    pub b_gathered: GpuBuffer,  // [B*T*d_state]
    pub c_gathered: GpuBuffer,  // [B*T*d_state]
    // Per-layer state (flat, with layer offsets — same pattern as online)
    pub conv_states: GpuBuffer, // [B*n_layers*d_inner*d_conv]
    pub ssm_states: GpuBuffer,  // [B*n_layers*d_inner*d_state]
    /// Dimensions this scratch was allocated for.
    pub dims: super::forward::GpuMambaDims,
}

impl GpuMambaTargetScratch {
    /// Allocate scratch buffers for batched Mamba target forward.
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &super::forward::GpuMambaDims,
    ) -> Result<Self, String> {
        let batch = dims.batch;
        let d_model = dims.d_model;
        let d_inner = dims.d_inner;
        let d_state = dims.d_state;
        let d_conv = dims.d_conv;
        let dt_rank = dims.dt_rank;
        let n_layers = dims.n_layers;
        let bt = batch * dims.seq_len;
        let xdbl_dim = dt_rank + 2 * d_state;

        Ok(Self {
            proj_flat: GpuBuffer::zeros(stream, bt * 2 * d_inner)?,
            x_branch: GpuBuffer::zeros(stream, bt * d_inner)?,
            gate_silu: GpuBuffer::zeros(stream, bt * d_inner)?,
            u: GpuBuffer::zeros(stream, bt * d_inner)?,
            xdbl: GpuBuffer::zeros(stream, bt * xdbl_dim)?,
            dt_gather: GpuBuffer::zeros(stream, bt * dt_rank)?,
            delta: GpuBuffer::zeros(stream, bt * d_inner)?,
            y: GpuBuffer::zeros(stream, bt * d_inner)?,
            gated: GpuBuffer::zeros(stream, bt * d_inner)?,
            out_flat: GpuBuffer::zeros(stream, bt * d_model)?,
            residual: GpuBuffer::zeros(stream, bt * d_model)?,
            rms_discard: GpuBuffer::zeros(stream, bt)?,
            b_gathered: GpuBuffer::zeros(stream, bt * d_state)?,
            c_gathered: GpuBuffer::zeros(stream, bt * d_state)?,
            conv_states: GpuBuffer::zeros(stream, batch * n_layers * d_inner * d_conv)?,
            ssm_states: GpuBuffer::zeros(stream, batch * n_layers * d_inner * d_state)?,
            dims: *dims,
        })
    }
}
