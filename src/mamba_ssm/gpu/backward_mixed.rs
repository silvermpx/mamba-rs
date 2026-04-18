//! Mixed-precision (bf16/f16) backward for Mamba SSM. Mirrors the f32 twin
//! [`super::backward::gpu_backward_mamba_backbone`] step-by-step, dispatching
//! typed CUDA kernels and typed cuBLAS GemmEx where applicable.
//!
//! ## Precision invariants (enforced throughout)
//!
//! - **Residual stream `d_temporal` stays f32** across all layers. The per-
//!   layer `rmsnorm_backward_f32in_typed` reads typed `dy` and f32 `x`
//!   (residual) and writes f32 `dx` directly into the residual gradient —
//!   no extra cast round-trip. Matches PyTorch AMP `residual_in_fp32=True`
//!   and state-spaces/mamba `residual = residual.to(torch.float32)`.
//! - **BPTT state (`h_saved`, `conv_states`, `da_exp`) stays f32**. The
//!   `ssm_backward_local_typed` kernel promotes typed `delta/u/B/C` on load
//!   but keeps all scan variables in f32.
//! - **T-length accumulators (`d_D_local`, `d_a_log_local`) stay f32**.
//! - **Weight gradients stay f32** and accumulate via `atomicAdd` (elementwise
//!   kernels) or `beta=1.0` cuBLAS GemmEx (`gpu_sgemm_backward_dw_grad_typed`).
//! - **Compute type `CUBLAS_COMPUTE_32F_PEDANTIC`** on every GemmEx —
//!   intentional divergence from PyTorch's TF32 default; see 61325b3 for the
//!   mamba-1.4b regression that proved TF32 on top of bf16 loses stability.
//!
//! ## Per-layer sequence (reverse of `forward_mixed` layer pipeline)
//!
//! The forward residual split-add is: `next_res = cur_res + out_flat_typed`,
//! implemented by `residual_add_f32_typed`. On backward, `d_next_res` enters
//! as `d_temporal` (f32) and flows along BOTH paths simultaneously:
//!
//! - **Branch path**: `d_out_flat_typed = cast(d_next_res, typed)`. This
//!   feeds out_proj backward.
//! - **Residual path**: `d_cur_res = d_next_res` (identity). We keep the
//!   accumulator in `d_temporal` and add the in-layer rmsnorm-backward
//!   result to it at B8, giving the final `d_cur_res` for layer L-1.
//!
//! Backward steps, in order:
//! - **B0** (cast): `d_temporal` (f32) → `temporal_typed` (typed).
//! - **B1** out_proj backward: typed dW + typed dX → `d_gated`.
//! - **B2** gating_bwd_typed: `d_gated` → `d_y`, `d_gate_pre`.
//! - **B3** SSM BPTT + typed/f32 reductions → `d_delta`, `d_u`, f32 master
//!   grads for `d_param`/`a_log`, typed `d_b/c_reduced` cast+scattered into
//!   typed `d_xdbl`.
//! - **B4** softplus_bwd_typed + dt_proj backward (typed dW + typed dX +
//!   typed bias reduction).
//! - **B5** x_proj backward (typed dW + typed dX → `d_u_xproj`), then
//!   `d_u += d_u_xproj` via `scatter_add_cols_typed` with `offset=0`.
//! - **B6** conv1d_burnin_bwd_typed → `d_x_branch` + f32 `d_conv_w/b`
//!   master grads (atomicAdd).
//! - **B7** concat(`d_x_branch`, `d_gate`) → `d_proj`; in_proj backward
//!   (typed dW + typed dX → `d_norm`).
//! - **B8** rmsnorm_bwd_f32in_typed: typed `d_norm` + f32 `residual` → f32
//!   `d_pre_norm` + f32 `d_norm_weight` master grad. Then `d_temporal +=
//!   d_pre_norm` (f32 `vec_add_inplace`).
//!
//! Backbone: norm_f bwd (pure f32) → reverse layer loop → input_proj bwd
//! (identity guard: only non-identity path is currently unsupported).

use super::blas::{TypedPtr, gpu_gemm_ex_backward_dx_typed, gpu_sgemm_backward_dw_grad_typed};
use super::buffers::GpuBuffer;
use super::context::GpuCtx;
use super::dtype::WeightDtype;
use super::forward_mixed::{
    GpuMambaBackboneMixedActs, GpuMambaLayerMixedActs, GpuMambaMixedTrainScratch,
};
use super::launch::{grid_1d, grid_norm};
use super::weights::{
    GpuMambaGrads, GpuMambaLayerGrads, GpuMambaMixedLayerWeights, GpuMambaMixedWeights,
};
use cudarc::driver::PushKernelArg;

// ---------------------------------------------------------------------------

/// Per-layer mixed backward. Matches `gpu_backward_mamba_layer` step-by-step.
#[allow(clippy::too_many_arguments)]
pub fn gpu_backward_mamba_layer_mixed(
    ctx: &GpuCtx,
    d_temporal: &mut GpuBuffer,
    d_lw: &GpuMambaLayerGrads,
    acts: &GpuMambaLayerMixedActs,
    lw: &GpuMambaMixedLayerWeights,
    a_neg_ptr: cudarc::driver::sys::CUdeviceptr,
    scratch: &mut GpuMambaMixedTrainScratch,
    dtype: WeightDtype,
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
    let k = &ctx.kernels;

    // ─── B0: Cast d_temporal (f32) → temporal_typed (typed) ──────────
    // Forward: residual_add_f32_typed(next_res_f32, cur_res_f32, out_flat_typed)
    // → next_res = cur_res + out_flat. Backward: d_out_flat_typed = cast(
    // d_next_res, typed). Since d_temporal holds d_next_res entering this
    // layer, we cast it to typed via `vec_add_inplace_typed` on a zeroed
    // typed destination (f32_src → typed_dst = 0 + f32).
    scratch.temporal_typed.zero(&ctx.stream)?;
    {
        let n = (bt * dm) as i32;
        let mut bld = ctx
            .stream
            .launch_builder(k.vec_add_inplace_typed.get(dtype));
        let a = scratch.temporal_typed.cached_ptr();
        let b_p = d_temporal.cached_ptr();
        bld.arg(&a);
        bld.arg(&b_p);
        bld.arg(&n);
        unsafe { bld.launch(grid_1d(bt * dm)) }
            .map_err(|e| format!("cast d_temporal→typed: {e:?}"))?;
    }

    // ─── B1: out_proj backward — typed dW + typed dX ─────────────────
    // dW[di, dm] += gated^T[di, bt] @ d_out_flat[bt, dm]
    gpu_sgemm_backward_dw_grad_typed(
        ctx,
        &d_lw.out_proj_w,
        TypedPtr {
            ptr: scratch.temporal_typed.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: acts.gated.cached_ptr(),
            dtype,
        },
        bt,
        di,
        dm,
    )?;
    // dX[bt, di] = d_out_flat[bt, dm] @ out_proj_w^T[dm, di]
    gpu_gemm_ex_backward_dx_typed(
        ctx,
        TypedPtr {
            ptr: scratch.d_gated.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: scratch.temporal_typed.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: lw.out_proj_w.ptr(),
            dtype,
        },
        bt,
        di,
        dm,
    )?;

    // ─── B2: gating_bwd_typed ────────────────────────────────────────
    {
        let n = (bt * di) as i32;
        let mut bld = ctx.stream.launch_builder(k.gating_bwd_typed.get(dtype));
        let dy = scratch.d_y.cached_ptr();
        let dg = scratch.d_gate.cached_ptr();
        let dgin = scratch.d_gated.cached_ptr();
        let y = acts.y.cached_ptr();
        let gp = acts.gate_pre_silu.cached_ptr();
        let gs = acts.gate_post_silu.cached_ptr();
        bld.arg(&dy);
        bld.arg(&dg);
        bld.arg(&dgin);
        bld.arg(&y);
        bld.arg(&gp);
        bld.arg(&gs);
        bld.arg(&n);
        unsafe { bld.launch(grid_1d(bt * di)) }.map_err(|e| format!("gating_bwd_typed: {e:?}"))?;
    }

    // ─── B3: SSM BPTT + reductions ───────────────────────────────────
    // Gather typed xdbl[:, dt_rank..] and xdbl[:, dt_rank+ds..] into typed
    // b_buf/c_buf — `ssm_backward_local_typed` reads these as B_saved/C_saved.
    {
        let bt_i = bt as i32;
        let xdbl_i = xdbl_dim as i32;
        let ds_i = ds as i32;
        let b_off = dt_rank as i32;
        let c_off = (dt_rank + ds) as i32;
        let mut bld = ctx.stream.launch_builder(k.gather_bc_cols_typed.get(dtype));
        let bb = scratch.b_buf.cached_ptr();
        let cb = scratch.c_buf.cached_ptr();
        let xd = acts.xdbl.cached_ptr();
        bld.arg(&bb);
        bld.arg(&cb);
        bld.arg(&xd);
        bld.arg(&bt_i);
        bld.arg(&xdbl_i);
        bld.arg(&ds_i);
        bld.arg(&b_off);
        bld.arg(&c_off);
        unsafe { bld.launch(grid_1d(bt * ds)) }
            .map_err(|e| format!("gather_bc_cols_typed bwd: {e:?}"))?;
    }

    // Zero T-length accumulator before SSM backward (kernel uses += over T).
    scratch.d_a_log_local.zero(&ctx.stream)?;

    // SSM backward: parallel reverse-scan when T > PARALLEL_SCAN_THRESHOLD
    // (Step 8e — wires the typed parallel bwd kernel into production) or
    // d_state > 64 (sequential kernel cap from register array size).
    // Otherwise sequential ssm_backward_local_typed.
    {
        let b_i = b as i32;
        let t_i = t as i32;
        let di_i = di as i32;
        let ds_i = ds as i32;
        let h_p = acts.h_saved.cached_ptr();
        let delta_p = acts.delta.cached_ptr();
        let u_p = acts.u.cached_ptr();
        let b_p = scratch.b_buf.cached_ptr();
        let c_p = scratch.c_buf.cached_ptr();
        let dp = lw.d_param.ptr();
        let dy = scratch.d_y.cached_ptr();
        let dd = scratch.d_delta.cached_ptr();
        let du = scratch.d_u.cached_ptr();
        let dbl = scratch.d_b_local.cached_ptr();
        let dcl = scratch.d_c_local.cached_ptr();
        let ddd = scratch.d_d_local.cached_ptr();
        let da = scratch.d_a_log_local.cached_ptr();

        if t > super::forward::PARALLEL_SCAN_THRESHOLD || ds > 64 {
            // Parallel reverse-scan typed bwd (Step 8e).
            // Signature: h_saved, delta, u, B, C, a_neg, D, dy, d_delta,
            //   d_u, d_B_local, d_C_local, d_D_local, d_a_log_local,
            //   batch, T, d_inner, d_state.
            let mut bld = ctx
                .stream
                .launch_builder(k.ssm_parallel_bwd_typed.get(dtype));
            bld.arg(&h_p);
            bld.arg(&delta_p);
            bld.arg(&u_p);
            bld.arg(&b_p);
            bld.arg(&c_p);
            bld.arg(&a_neg_ptr);
            bld.arg(&dp);
            bld.arg(&dy);
            bld.arg(&dd);
            bld.arg(&du);
            bld.arg(&dbl);
            bld.arg(&dcl);
            bld.arg(&ddd);
            bld.arg(&da);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&di_i);
            bld.arg(&ds_i);
            unsafe {
                bld.launch(super::launch::grid_parallel_scan_bwd(
                    b,
                    di,
                    dtype.size_bytes(),
                ))
            }
            .map_err(|e| format!("ssm_parallel_bwd_typed: {e:?}"))?;
        } else {
            // Sequential typed bwd.
            let mut bld = ctx
                .stream
                .launch_builder(k.ssm_backward_local_typed.get(dtype));
            bld.arg(&h_p);
            bld.arg(&delta_p);
            bld.arg(&u_p);
            bld.arg(&b_p);
            bld.arg(&c_p);
            bld.arg(&a_neg_ptr);
            bld.arg(&dp);
            bld.arg(&dy);
            bld.arg(&dd);
            bld.arg(&du);
            bld.arg(&dbl);
            bld.arg(&dcl);
            bld.arg(&ddd);
            bld.arg(&da);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&di_i);
            bld.arg(&ds_i);
            unsafe { bld.launch(grid_1d(b * di)) }
                .map_err(|e| format!("ssm_backward_local_typed: {e:?}"))?;
        }
    }

    // Reductions: zero f32 targets; typed → f32 for d_b/c, f32 → f32 for
    // d_d/d_a_log.
    scratch.d_b_reduced.zero(&ctx.stream)?;
    scratch.d_c_reduced.zero(&ctx.stream)?;
    {
        let b_i = b as i32;
        let t_i = t as i32;
        let di_i = di as i32;
        let ds_i = ds as i32;
        // d_B reducer — typed in → f32 out.
        let reduce_db = match dtype {
            WeightDtype::Bf16 => &k.ssm_reduce_d_b_bf16,
            WeightDtype::F16 => &k.ssm_reduce_d_b_f16,
            WeightDtype::F32 => &k.ssm_reduce_d_b,
        };
        let mut bld = ctx.stream.launch_builder(reduce_db);
        bld.arg(scratch.d_b_reduced.inner_mut());
        let src = scratch.d_b_local.cached_ptr();
        bld.arg(&src);
        bld.arg(&b_i);
        bld.arg(&t_i);
        bld.arg(&di_i);
        bld.arg(&ds_i);
        unsafe { bld.launch(grid_1d(bt * ds)) }
            .map_err(|e| format!("ssm_reduce_d_b typed: {e:?}"))?;
        // d_C reducer.
        let reduce_dc = match dtype {
            WeightDtype::Bf16 => &k.ssm_reduce_d_c_bf16,
            WeightDtype::F16 => &k.ssm_reduce_d_c_f16,
            WeightDtype::F32 => &k.ssm_reduce_d_c,
        };
        let mut bld = ctx.stream.launch_builder(reduce_dc);
        bld.arg(scratch.d_c_reduced.inner_mut());
        let src = scratch.d_c_local.cached_ptr();
        bld.arg(&src);
        bld.arg(&b_i);
        bld.arg(&t_i);
        bld.arg(&di_i);
        bld.arg(&ds_i);
        unsafe { bld.launch(grid_1d(bt * ds)) }
            .map_err(|e| format!("ssm_reduce_d_c typed: {e:?}"))?;
        // d_D reducer — f32 in → f32 out.
        let mut bld = ctx.stream.launch_builder(&k.ssm_reduce_d_d);
        let p = d_lw.d_param.ptr();
        bld.arg(&p);
        bld.arg(scratch.d_d_local.inner());
        bld.arg(&b_i);
        bld.arg(&di_i);
        unsafe { bld.launch(grid_1d(di)) }.map_err(|e| format!("ssm_reduce_d_d: {e:?}"))?;
        // d_a_log reducer — f32 in → f32 out.
        let mut bld = ctx.stream.launch_builder(&k.ssm_reduce_d_a_log);
        let p = d_lw.a_log.ptr();
        bld.arg(&p);
        bld.arg(scratch.d_a_log_local.inner());
        bld.arg(&b_i);
        bld.arg(&di_i);
        bld.arg(&ds_i);
        unsafe { bld.launch(grid_1d(di * ds)) }
            .map_err(|e| format!("ssm_reduce_d_a_log: {e:?}"))?;
    }

    // Assemble typed d_xdbl: zero, then scatter d_b/c_reduced (f32) into
    // typed slots via typed staging (b_buf/c_buf). dt slot filled later by
    // scatter of d_dt_input after the dt_proj backward (so zero the whole
    // buffer once up-front).
    scratch.d_xdbl.zero(&ctx.stream)?;
    {
        let n_bc = (bt * ds) as i32;
        // Re-use b_buf as typed stage for d_b_reduced (typed = 0 + f32_src).
        scratch.b_buf.zero(&ctx.stream)?;
        let mut bld = ctx
            .stream
            .launch_builder(k.vec_add_inplace_typed.get(dtype));
        let dst = scratch.b_buf.cached_ptr();
        let src = scratch.d_b_reduced.cached_ptr();
        bld.arg(&dst);
        bld.arg(&src);
        bld.arg(&n_bc);
        unsafe { bld.launch(grid_1d(bt * ds)) }
            .map_err(|e| format!("cast d_b_reduced→typed: {e:?}"))?;
        scratch.c_buf.zero(&ctx.stream)?;
        let mut bld = ctx
            .stream
            .launch_builder(k.vec_add_inplace_typed.get(dtype));
        let dst = scratch.c_buf.cached_ptr();
        let src = scratch.d_c_reduced.cached_ptr();
        bld.arg(&dst);
        bld.arg(&src);
        bld.arg(&n_bc);
        unsafe { bld.launch(grid_1d(bt * ds)) }
            .map_err(|e| format!("cast d_c_reduced→typed: {e:?}"))?;
        // Scatter typed b_buf → d_xdbl[:, dt_rank..dt_rank+ds].
        let bt_i = bt as i32;
        let xdbl_i = xdbl_dim as i32;
        let ds_i = ds as i32;
        let b_off = dt_rank as i32;
        let c_off = (dt_rank + ds) as i32;
        let mut bld = ctx
            .stream
            .launch_builder(k.scatter_add_cols_typed.get(dtype));
        let dst = scratch.d_xdbl.cached_ptr();
        let src = scratch.b_buf.cached_ptr();
        bld.arg(&dst);
        bld.arg(&src);
        bld.arg(&bt_i);
        bld.arg(&xdbl_i);
        bld.arg(&ds_i);
        bld.arg(&b_off);
        unsafe { bld.launch(grid_1d(bt * ds)) }.map_err(|e| format!("scatter d_b typed: {e:?}"))?;
        let mut bld = ctx
            .stream
            .launch_builder(k.scatter_add_cols_typed.get(dtype));
        let dst = scratch.d_xdbl.cached_ptr();
        let src = scratch.c_buf.cached_ptr();
        bld.arg(&dst);
        bld.arg(&src);
        bld.arg(&bt_i);
        bld.arg(&xdbl_i);
        bld.arg(&ds_i);
        bld.arg(&c_off);
        unsafe { bld.launch(grid_1d(bt * ds)) }.map_err(|e| format!("scatter d_c typed: {e:?}"))?;
    }

    // ─── B4: softplus backward + dt_proj backward ────────────────────
    {
        let n = (bt * di) as i32;
        let mut bld = ctx.stream.launch_builder(k.softplus_bwd_typed.get(dtype));
        let dx = scratch.d_delta_raw.cached_ptr();
        let xs = acts.delta_raw.cached_ptr();
        let dy = scratch.d_delta.cached_ptr();
        bld.arg(&dx);
        bld.arg(&xs);
        bld.arg(&dy);
        bld.arg(&n);
        unsafe { bld.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("softplus_bwd_typed: {e:?}"))?;
    }
    // Gather typed xdbl[:, 0..dt_rank] → typed dt_xdbl_buf (x_saved for dt_proj dW).
    {
        let bt_i = bt as i32;
        let xdbl_i = xdbl_dim as i32;
        let dt_i = dt_rank as i32;
        let offset: i32 = 0;
        let mut bld = ctx.stream.launch_builder(k.gather_cols_typed.get(dtype));
        let dst = scratch.dt_xdbl_buf.cached_ptr();
        let src = acts.xdbl.cached_ptr();
        bld.arg(&dst);
        bld.arg(&src);
        bld.arg(&bt_i);
        bld.arg(&xdbl_i);
        bld.arg(&dt_i);
        bld.arg(&offset);
        unsafe { bld.launch(grid_1d(bt * dt_rank)) }
            .map_err(|e| format!("gather_cols dt bwd typed: {e:?}"))?;
    }
    // dt_proj dW — typed GemmEx accumulator on f32 master.
    gpu_sgemm_backward_dw_grad_typed(
        ctx,
        &d_lw.dt_proj_w,
        TypedPtr {
            ptr: scratch.d_delta_raw.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: scratch.dt_xdbl_buf.cached_ptr(),
            dtype,
        },
        bt,
        dt_rank,
        di,
    )?;
    // dt_proj dX — typed GemmEx.
    gpu_gemm_ex_backward_dx_typed(
        ctx,
        TypedPtr {
            ptr: scratch.d_dt_input.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: scratch.d_delta_raw.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: lw.dt_proj_w.ptr(),
            dtype,
        },
        bt,
        dt_rank,
        di,
    )?;
    // dt_proj bias grad — typed reduction of d_delta_raw over (b, t) into
    // f32 d_dt_proj_b master grad.
    {
        let bt_i = bt as i32;
        let di_i = di as i32;
        let mut bld = ctx.stream.launch_builder(k.reduce_bias_typed.get(dtype));
        let db = d_lw.dt_proj_b.ptr();
        let dy = scratch.d_delta_raw.cached_ptr();
        bld.arg(&db);
        bld.arg(&dy);
        bld.arg(&bt_i);
        bld.arg(&di_i);
        // One block per bias index, threads sum over bt; shared mem sized by
        // threads-per-block (grid_1d spawns 256 threads/block).
        let threads = 256u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (di as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: (threads as usize * std::mem::size_of::<f32>()) as u32,
        };
        unsafe { bld.launch(cfg) }.map_err(|e| format!("reduce_bias dt_proj: {e:?}"))?;
    }
    // Scatter d_dt_input (typed) into d_xdbl[:, 0..dt_rank].
    {
        let bt_i = bt as i32;
        let xdbl_i = xdbl_dim as i32;
        let dt_i = dt_rank as i32;
        let offset: i32 = 0;
        let mut bld = ctx
            .stream
            .launch_builder(k.scatter_add_cols_typed.get(dtype));
        let dst = scratch.d_xdbl.cached_ptr();
        let src = scratch.d_dt_input.cached_ptr();
        bld.arg(&dst);
        bld.arg(&src);
        bld.arg(&bt_i);
        bld.arg(&xdbl_i);
        bld.arg(&dt_i);
        bld.arg(&offset);
        unsafe { bld.launch(grid_1d(bt * dt_rank)) }
            .map_err(|e| format!("scatter dt typed bwd: {e:?}"))?;
    }

    // ─── B5: x_proj backward ─────────────────────────────────────────
    // dW — typed GemmEx.
    gpu_sgemm_backward_dw_grad_typed(
        ctx,
        &d_lw.x_proj_w,
        TypedPtr {
            ptr: scratch.d_xdbl.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: acts.u.cached_ptr(),
            dtype,
        },
        bt,
        di,
        xdbl_dim,
    )?;
    // dX — typed GemmEx into d_u_xproj.
    gpu_gemm_ex_backward_dx_typed(
        ctx,
        TypedPtr {
            ptr: scratch.d_u_xproj.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: scratch.d_xdbl.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: lw.x_proj_w.ptr(),
            dtype,
        },
        bt,
        di,
        xdbl_dim,
    )?;
    // Accumulate d_u += d_u_xproj (typed += typed). `scatter_add_cols_typed`
    // with `batch=bt, dst_stride=di, src_dim=di, offset=0` is exactly this.
    {
        let bt_i = bt as i32;
        let di_i_dst = di as i32;
        let di_i_src = di as i32;
        let offset: i32 = 0;
        let mut bld = ctx
            .stream
            .launch_builder(k.scatter_add_cols_typed.get(dtype));
        let dst = scratch.d_u.cached_ptr();
        let src = scratch.d_u_xproj.cached_ptr();
        bld.arg(&dst);
        bld.arg(&src);
        bld.arg(&bt_i);
        bld.arg(&di_i_dst);
        bld.arg(&di_i_src);
        bld.arg(&offset);
        unsafe { bld.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("d_u += d_u_xproj typed: {e:?}"))?;
    }

    // ─── B6: Conv1d burnin backward ──────────────────────────────────
    // Rule B (no atomicAdd): two-stage launch.
    // Stage 1: conv1d_burnin_bwd_typed writes typed d_x_branch + per-(b,d)
    //          f32 partials into axis0_partials split as [weight | bias].
    // Stage 2: two reduce_sum_axis0 launches reduce across B → f32 master
    //          grads in d_lw.conv1d_weight / d_lw.conv1d_bias.
    {
        let b_i = b as i32;
        let t_i = t as i32;
        let di_i = di as i32;
        let dc_i = d_conv as i32;
        let weight_partials_elems = b * di * d_conv;
        let bias_offset_bytes = (weight_partials_elems * std::mem::size_of::<f32>()) as u64;
        let axis0_base = scratch.axis0_partials.cached_ptr();
        let wp_ptr = axis0_base;
        let bp_ptr = axis0_base + bias_offset_bytes;
        // Stage 1: per-(b,d) partials.
        {
            let mut bld = ctx
                .stream
                .launch_builder(k.conv1d_burnin_bwd_typed.get(dtype));
            let dxb = scratch.d_x_branch.cached_ptr();
            let du = scratch.d_u.cached_ptr();
            let pc = acts.post_conv.cached_ptr();
            let cs = acts.conv_states.cached_ptr();
            let w = lw.conv1d_weight.ptr();
            bld.arg(&dxb);
            bld.arg(&wp_ptr); // d_weight_partials
            bld.arg(&bp_ptr); // d_bias_partials
            bld.arg(&du);
            bld.arg(&pc);
            bld.arg(&cs);
            bld.arg(&w);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&di_i);
            bld.arg(&dc_i);
            unsafe { bld.launch(grid_1d(b * di)) }
                .map_err(|e| format!("conv1d_burnin_bwd_typed partial: {e:?}"))?;
        }
        // Stage 2a: reduce weight partials [B, di*d_conv] → d_lw.conv1d_weight.
        {
            let block_dim = (b as u32).next_power_of_two().clamp(32, 256);
            let accumulate_i: i32 = 1;
            let dim_w = (di * d_conv) as i32;
            let p = d_lw.conv1d_weight.ptr();
            let mut bld = ctx.stream.launch_builder(&ctx.kernels.reduce_sum_axis0);
            bld.arg(&p);
            bld.arg(&wp_ptr);
            bld.arg(&b_i);
            bld.arg(&dim_w);
            bld.arg(&accumulate_i);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: ((di * d_conv) as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
            };
            unsafe { bld.launch(cfg) }
                .map_err(|e| format!("conv1d_burnin_bwd_typed weight final: {e:?}"))?;
        }
        // Stage 2b: reduce bias partials [B, di] → d_lw.conv1d_bias.
        {
            let block_dim = (b as u32).next_power_of_two().clamp(32, 256);
            let accumulate_i: i32 = 1;
            let p = d_lw.conv1d_bias.ptr();
            let mut bld = ctx.stream.launch_builder(&ctx.kernels.reduce_sum_axis0);
            bld.arg(&p);
            bld.arg(&bp_ptr);
            bld.arg(&b_i);
            bld.arg(&di_i);
            bld.arg(&accumulate_i);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (di as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
            };
            unsafe { bld.launch(cfg) }
                .map_err(|e| format!("conv1d_burnin_bwd_typed bias final: {e:?}"))?;
        }
    }

    // ─── B7: in_proj backward ────────────────────────────────────────
    // Concat(d_x_branch, d_gate) → d_proj [bt, 2*di] — layout matches
    // forward's split_gate_silu which reads proj[b, 0..di] for x_branch and
    // proj[b, di..2*di] for gate.
    {
        let bt_i = bt as i32;
        let di_i = di as i32;
        let mut bld = ctx.stream.launch_builder(k.concat_halves_typed.get(dtype));
        let dst = scratch.d_proj.cached_ptr();
        let fh = scratch.d_x_branch.cached_ptr();
        let sh = scratch.d_gate.cached_ptr();
        bld.arg(&dst);
        bld.arg(&fh);
        bld.arg(&sh);
        bld.arg(&bt_i);
        bld.arg(&di_i);
        unsafe { bld.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("concat_halves_typed bwd: {e:?}"))?;
    }
    // in_proj dW — typed GemmEx.
    gpu_sgemm_backward_dw_grad_typed(
        ctx,
        &d_lw.in_proj_w,
        TypedPtr {
            ptr: scratch.d_proj.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: acts.post_norm.cached_ptr(),
            dtype,
        },
        bt,
        dm,
        2 * di,
    )?;
    // in_proj dX — typed GemmEx into d_norm (feeds rmsnorm_bwd_f32in as dy).
    gpu_gemm_ex_backward_dx_typed(
        ctx,
        TypedPtr {
            ptr: scratch.d_norm.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: scratch.d_proj.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: lw.in_proj_w.ptr(),
            dtype,
        },
        bt,
        dm,
        2 * di,
    )?;

    // ─── B8: RmsNorm backward (typed dy + f32 x → f32 dx) + residual ─
    // Rule B (no atomicAdd): two-stage launch.
    // Stage 1: rmsnorm_bwd variant writes f32 d_pre_norm + per-sample per-dim
    //          f32 partials into axis0_partials[bt * dm].
    // Stage 2: reduce_sum_axis0 reduces across bt → d_lw.norm_weight (f32
    //          master grad, accumulate=1).
    // Dispatch: bf16/f16 → dual-dtype f32in kernel; f32 → pure-f32 rmsnorm_bwd.
    let rmsnorm_bwd = match dtype {
        WeightDtype::F32 => &k.rmsnorm_bwd,
        WeightDtype::Bf16 | WeightDtype::F16 => k.rmsnorm_bwd_f32in_typed.get(dtype),
    };
    {
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let axis0_ptr = scratch.axis0_partials.cached_ptr();
        // Stage 1
        {
            let mut bld = ctx.stream.launch_builder(rmsnorm_bwd);
            let dx = scratch.d_pre_norm.cached_ptr();
            let dy = scratch.d_norm.cached_ptr();
            let x = acts.residual.cached_ptr();
            let sc = lw.norm_weight.ptr();
            let rms = acts.rms_vals.cached_ptr();
            bld.arg(&dx);
            bld.arg(&axis0_ptr); // d_scale_partials
            bld.arg(&dy);
            bld.arg(&x);
            bld.arg(&sc);
            bld.arg(&rms);
            bld.arg(&bt_i);
            bld.arg(&dm_i);
            unsafe { bld.launch(grid_norm(bt, dm)) }
                .map_err(|e| format!("rmsnorm_bwd_f32in_typed partial: {e:?}"))?;
        }
        // Stage 2
        {
            let block_dim = (bt as u32).next_power_of_two().clamp(32, 256);
            let accumulate_i: i32 = 1;
            let p = d_lw.norm_weight.ptr();
            let mut bld = ctx.stream.launch_builder(&ctx.kernels.reduce_sum_axis0);
            bld.arg(&p);
            bld.arg(&axis0_ptr);
            bld.arg(&bt_i);
            bld.arg(&dm_i);
            bld.arg(&accumulate_i);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dm as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
            };
            unsafe { bld.launch(cfg) }
                .map_err(|e| format!("rmsnorm_bwd_f32in_typed final: {e:?}"))?;
        }
    }
    // Residual: d_temporal (f32) += d_pre_norm (f32).
    {
        let n = (bt * dm) as i32;
        let mut bld = ctx.stream.launch_builder(&k.vec_add_inplace);
        bld.arg(d_temporal.inner_mut());
        bld.arg(scratch.d_pre_norm.inner());
        bld.arg(&n);
        unsafe { bld.launch(grid_1d(bt * dm)) }
            .map_err(|e| format!("vec_add residual bwd mixed: {e:?}"))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------

/// Full-backbone mixed-precision backward. Mirrors
/// [`super::backward::gpu_backward_mamba_backbone`]:
/// norm_f bwd (pure f32) → reverse layer loop → input_proj bwd (identity).
///
/// **IMPORTANT**: weight gradients in `d_mamba` are **accumulated** via
/// `beta=1.0` / `atomicAdd`. The caller MUST call
/// [`GpuMambaGrads::zero`] before each training step if the buffer is reused
/// across iterations — otherwise gradients from step N−1 pollute step N and
/// the optimizer sees doubled updates.
pub fn gpu_backward_mamba_backbone_mixed(
    ctx: &GpuCtx,
    d_temporal: &mut GpuBuffer,
    d_mamba: &GpuMambaGrads,
    acts: &GpuMambaBackboneMixedActs,
    mamba_w: &GpuMambaMixedWeights,
    a_neg_all: &GpuBuffer,
    scratch: &mut GpuMambaMixedTrainScratch,
) -> Result<(), String> {
    let dims = scratch.dims;
    let dtype = acts.dtype;
    let bt = dims.bt();

    // norm_f backward — always f32 (dy=f32 d_temporal, x=f32 norm_f_input,
    // dx=f32). Result copied back into d_temporal.
    // Rule B (no atomicAdd): two-stage launch for d_scale accumulator.
    {
        let bt_i = bt as i32;
        let dm_i = dims.d_model as i32;
        let axis0_ptr = scratch.axis0_partials.cached_ptr();
        // Stage 1
        {
            let mut bld = ctx.stream.launch_builder(&ctx.kernels.rmsnorm_bwd);
            bld.arg(scratch.d_pre_norm.inner_mut());
            bld.arg(&axis0_ptr); // d_scale_partials
            bld.arg(d_temporal.inner());
            bld.arg(acts.norm_f_input.inner());
            let nf = mamba_w.norm_f_weight.ptr();
            bld.arg(&nf);
            bld.arg(acts.norm_f_rms.inner());
            bld.arg(&bt_i);
            bld.arg(&dm_i);
            unsafe { bld.launch(grid_norm(bt, dims.d_model)) }
                .map_err(|e| format!("rmsnorm_bwd norm_f mixed partial: {e:?}"))?;
        }
        // Stage 2
        {
            let block_dim = (bt as u32).next_power_of_two().clamp(32, 256);
            let accumulate_i: i32 = 1;
            let p = d_mamba.norm_f_weight.ptr();
            let mut bld = ctx.stream.launch_builder(&ctx.kernels.reduce_sum_axis0);
            bld.arg(&p);
            bld.arg(&axis0_ptr);
            bld.arg(&bt_i);
            bld.arg(&dm_i);
            bld.arg(&accumulate_i);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dims.d_model as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
            };
            unsafe { bld.launch(cfg) }
                .map_err(|e| format!("rmsnorm_bwd norm_f mixed final: {e:?}"))?;
        }
        d_temporal.copy_from(&scratch.d_pre_norm, &ctx.stream)?;
    }

    // Layers in reverse with per-layer a_neg offset.
    let a_neg_per_layer = dims.d_inner * dims.d_state;
    for layer_idx in (0..dims.n_layers).rev() {
        let base = a_neg_all.raw_ptr(&ctx.stream);
        let a_neg_ptr = base + (layer_idx * a_neg_per_layer * std::mem::size_of::<f32>()) as u64;
        gpu_backward_mamba_layer_mixed(
            ctx,
            d_temporal,
            &d_mamba.layers[layer_idx],
            &acts.layers[layer_idx],
            &mamba_w.layers[layer_idx],
            a_neg_ptr,
            scratch,
            dtype,
        )?;
    }

    // input_proj backward — identity-proj path only (matches forward_mixed).
    if mamba_w.input_proj_w.len_elems() != 0 {
        return Err(
            "mixed backward: non-identity input_proj not yet supported (matches forward_mixed)"
                .into(),
        );
    }
    Ok(())
}
