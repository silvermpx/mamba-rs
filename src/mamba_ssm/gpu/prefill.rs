//! Parallel GPU prefill for LLM prompt processing.
//!
//! Processes T tokens of a prompt in a single forward pass using the
//! `ssm_burnin_forward_nosave` / `ssm_parallel_scan_fwd_nosave` kernels,
//! producing the final hidden state in `GpuInferenceState` + last-timestep
//! temporal output. Avoids the O(T) kernel-launch overhead of step-by-step
//! prefill — 10-40× speedup for prompts with T > 128.
//!
//! Mirrors `gpu_forward_mamba_target_burnin` but uses the inference-path
//! flat `GpuMambaWeights` (not training `GpuMambaTrainWeights`).

use super::backward::GpuMambaTargetScratch;
use super::blas::{gpu_gemm_forward_dispatch, gpu_sgemm_forward_raw};
use super::buffers::GpuBuffer;
use super::context::GpuCtx;
use super::forward::{GpuMambaDims, PARALLEL_SCAN_THRESHOLD};
use super::inference::GpuInferenceState;
use super::launch::{grid_1d, grid_norm, grid_parallel_scan};
use super::weights::{MambaLayerWeightsView, MambaWeightsView};
use cudarc::driver::PushKernelArg;

/// Prefill a batched prompt sequence through the Mamba backbone in one call.
///
/// Inputs:
/// - `target_temporal`: `[B * d_model]` — output (last timestep hidden state, f32)
/// - `ip_out_flat`: `[B * T * d_model]` — pre-embedded prompt tokens (batch × time × d_model)
/// - `weights`: flat-buffer inference weights (M1)
/// - `state`: persistent inference state (conv + SSM) — updated in-place
/// - `a_neg_all`: precomputed `-exp(a_log)` for all layers `[n_layers * d_inner * d_state]`
/// - `scratch`: batched B*T working buffers
///
/// Inputs bundle for `gpu_forward_inference_prefill`.
pub struct PrefillInputs<'a, W: MambaWeightsView> {
    pub ip_out_flat: &'a GpuBuffer,
    pub weights: &'a W,
    pub a_neg_all: &'a GpuBuffer,
}

/// After this call, `state` holds the recurrent state at position T, and
/// `target_temporal` holds the pre-lm_head hidden state for token T (last).
/// Follow with normal `step()` calls to continue decoding.
pub fn gpu_forward_inference_prefill<W: MambaWeightsView>(
    ctx: &GpuCtx,
    target_temporal: &mut GpuBuffer,
    inputs: PrefillInputs<'_, W>,
    state: &mut GpuInferenceState,
    scratch: &mut GpuMambaTargetScratch,
) -> Result<(), String> {
    let PrefillInputs { ip_out_flat, weights, a_neg_all } = inputs;
    let dims: GpuMambaDims = scratch.dims;
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

    // State is the persistent inference state — do NOT zero. We build on top.
    // Working temporal: start with the input embeddings for all T timesteps.
    scratch.out_flat.copy_from(ip_out_flat, &ctx.stream)?;

    let f32_sz = std::mem::size_of::<f32>() as u64;

    for layer_idx in 0..weights.n_layers() {
        let lw = weights.layer(layer_idx);

        // Per-layer state pointers into the inference state (NOT scratch.conv_states).
        // inference state layout: conv[n_layers][batch * d_inner * d_conv],
        //                         ssm[n_layers][batch * d_inner * d_state].
        let conv_per_layer = b * di * d_conv;
        let ssm_per_layer = b * di * ds;
        let conv_ptr = state.conv.cached_ptr() + (layer_idx * conv_per_layer) as u64 * f32_sz;
        let ssm_ptr = state.ssm.cached_ptr() + (layer_idx * ssm_per_layer) as u64 * f32_sz;
        let a_neg_ptr = a_neg_all.cached_ptr() + (layer_idx * di * ds) as u64 * f32_sz;

        // F1: RmsNorm [B*T] ← save residual first
        scratch.residual.copy_from(&scratch.out_flat, &ctx.stream)?;
        {
            let bt_i = bt as i32;
            let dm_i = dm as i32;
            let eps: f32 = 1e-5;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.rmsnorm_fwd);
            builder.arg(scratch.out_flat.inner_mut());
            builder.arg(scratch.rms_discard.inner_mut());
            builder.arg(scratch.residual.inner());
            let nw = lw.norm_weight();
            builder.arg(&nw);
            builder.arg(&bt_i);
            builder.arg(&dm_i);
            builder.arg(&eps);
            unsafe { builder.launch(grid_norm(bt, dm)) }
                .map_err(|e| format!("rmsnorm prefill L{layer_idx}: {e:?}"))?;
        }

        // F2: in_proj GEMM [B*T, dm] → [B*T, 2*di] (dtype-dispatched)
        let (ipw, ipw_dt) = lw.in_proj_w();
        gpu_gemm_forward_dispatch(
            ctx,
            &mut scratch.proj_flat,
            &scratch.out_flat,
            ipw,
            ipw_dt,
            None,
            (bt, dm, 2 * di),
        )?;

        // F3: split x + SiLU(gate) [B*T]
        {
            let bt_i = bt as i32;
            let di_i = di as i32;
            let gs_raw = scratch.gate_silu.cached_ptr();
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.split_gate_silu);
            builder.arg(scratch.x_branch.inner_mut());
            builder.arg(scratch.gate_silu.inner_mut());
            builder.arg(&gs_raw);
            builder.arg(scratch.proj_flat.inner());
            builder.arg(&bt_i);
            builder.arg(&di_i);
            unsafe { builder.launch(grid_1d(bt * di)) }
                .map_err(|e| format!("split_gate prefill L{layer_idx}: {e:?}"))?;
        }

        // F4a: conv1d burnin nosave + fused SiLU [all T, parallel B*d_inner]
        {
            let b_i = b as i32;
            let t_i = t as i32;
            let di_i = di as i32;
            let dc_i = d_conv as i32;
            let mut builder = ctx
                .stream
                .launch_builder(&ctx.kernels.conv1d_burnin_fwd_nosave);
            builder.arg(scratch.u.inner_mut());
            builder.arg(&conv_ptr); // INFERENCE STATE conv — persistent
            builder.arg(scratch.x_branch.inner());
            let cw = lw.conv1d_weight();
            let cb = lw.conv1d_bias();
            builder.arg(&cw);
            builder.arg(&cb);
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&di_i);
            builder.arg(&dc_i);
            unsafe { builder.launch(grid_1d(b * di)) }
                .map_err(|e| format!("conv1d_nosave prefill L{layer_idx}: {e:?}"))?;
        }

        // F4b: x_proj GEMM [B*T, di] → [B*T, xdbl_dim]
        let (xpw, xpw_dt) = lw.x_proj_w();
        gpu_gemm_forward_dispatch(
            ctx,
            &mut scratch.xdbl,
            &scratch.u,
            xpw,
            xpw_dt,
            None,
            (bt, di, xdbl_dim),
        )?;

        // F4c: gather dt + dt_proj + softplus
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
                .map_err(|e| format!("gather dt prefill L{layer_idx}: {e:?}"))?;
        }
        let (dpw, dpw_dt) = lw.dt_proj_w();
        gpu_gemm_forward_dispatch(
            ctx,
            &mut scratch.delta,
            &scratch.dt_gather,
            dpw,
            dpw_dt,
            Some(lw.dt_proj_b()),
            (bt, dt_rank, di),
        )?;
        {
            let n = (bt * di) as i32;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.softplus_fwd);
            builder.arg(scratch.delta.inner_mut());
            builder.arg(&n);
            unsafe { builder.launch(grid_1d(bt * di)) }
                .map_err(|e| format!("softplus prefill L{layer_idx}: {e:?}"))?;
        }

        // F4d: gather B/C + SSM burnin nosave (sequential for T ≤ 256, parallel otherwise)
        {
            let bt_i = bt as i32;
            let xdbl_i = xdbl_dim as i32;
            let ds_i = ds as i32;
            let b_offset = dt_rank as i32;
            let c_offset = (dt_rank + ds) as i32;
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
                .map_err(|e| format!("gather_bc prefill L{layer_idx}: {e:?}"))?;
        }
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
                let dp = lw.d_param();
                builder.arg(&dp);
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&di_i);
                builder.arg(&ds_i);
                unsafe { builder.launch(grid_parallel_scan(b, di)) }
                    .map_err(|e| format!("ssm_parallel prefill L{layer_idx}: {e:?}"))?;
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
                let dp = lw.d_param();
                builder.arg(&dp);
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&di_i);
                builder.arg(&ds_i);
                unsafe { builder.launch(grid_1d(b * di)) }
                    .map_err(|e| format!("ssm_nosave prefill L{layer_idx}: {e:?}"))?;
            }
        }

        // F4e: gating — y * gate_silu
        {
            let n = (bt * di) as i32;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.elementwise_mul);
            builder.arg(scratch.gated.inner_mut());
            builder.arg(scratch.y.inner());
            builder.arg(scratch.gate_silu.inner());
            builder.arg(&n);
            unsafe { builder.launch(grid_1d(bt * di)) }
                .map_err(|e| format!("gating prefill L{layer_idx}: {e:?}"))?;
        }

        // F5: out_proj GEMM [B*T, di] → [B*T, dm]
        let (opw, opw_dt) = lw.out_proj_w();
        gpu_gemm_forward_dispatch(
            ctx,
            &mut scratch.out_flat,
            &scratch.gated,
            opw,
            opw_dt,
            None,
            (bt, di, dm),
        )?;

        // F6: residual add (in-place on out_flat)
        {
            let n = (bt * dm) as i32;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.vec_add_inplace);
            builder.arg(scratch.out_flat.inner_mut());
            builder.arg(scratch.residual.inner());
            builder.arg(&n);
            unsafe { builder.launch(grid_1d(bt * dm)) }
                .map_err(|e| format!("residual prefill L{layer_idx}: {e:?}"))?;
        }
    }

    // Final RmsNorm (norm_f) over all B*T timesteps
    {
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let eps: f32 = 1e-5;
        scratch.residual.copy_from(&scratch.out_flat, &ctx.stream)?;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.rmsnorm_fwd);
        builder.arg(scratch.out_flat.inner_mut());
        builder.arg(scratch.rms_discard.inner_mut());
        builder.arg(scratch.residual.inner());
        let nfw = weights.norm_f_weight();
        builder.arg(&nfw);
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        builder.arg(&eps);
        unsafe { builder.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("norm_f prefill: {e:?}"))?;
    }

    // Extract last timestep into target_temporal [B * dm]
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
            .map_err(|e| format!("gather_last prefill: {e:?}"))?;
    }

    // Avoid unused warning on the identity_proj-unused sgemm import
    let _ = gpu_sgemm_forward_raw;

    Ok(())
}
