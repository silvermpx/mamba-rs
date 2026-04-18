//! Step 10 — Mamba-3 SISO mixed-precision (bf16/f16) backward training
//! pipeline. Mirrors [`super::mamba3_gpu::gpu_backward_mamba3_layer`] +
//! `_backbone` with typed activation I/O routed through the typed kernel
//! variants from Steps 9a/9b/9c/9d.
//!
//! ## Production-config scope
//! - `dims.use_parallel_scan = true` (chunked SSM bwd via Steps 9b + 9d)
//! - `dims.is_outproj_norm = true` (RMSNormGated via Step 9c)
//! - `dims.n_angles > 0` (RoPE via Step 9a)
//!
//! Non-production configs return `Err(...)` — fall back to the f32 path
//! [`super::mamba3_gpu::gpu_backward_mamba3_backbone`].
//!
//! ## Precision invariants
//! - **Activations**: typed (bf16/f16) — read via `cached_ptr()` from
//!   `GpuMamba3LayerMixedActs` and written to typed scratch buffers in
//!   `GpuMamba3MixedScratch`.
//! - **Master grads** (in `GpuMamba3LayerGrads`): all stay f32 — atomicAdd
//!   master-grad invariant per audit.
//! - **Activation grads on the wire**: typed where they cross typed-kernel
//!   boundaries (d_gated, d_y, d_z, d_b_normed, d_c_normed, d_b_raw,
//!   d_c_raw, d_proj, d_post_norm, d_temporal_typed); otherwise f32 in
//!   the existing `GpuMamba3Scratch`.

use crate::mamba_ssm::gpu::blas::{
    TypedPtr, gpu_gemm_ex_backward_dx_typed, gpu_sgemm_backward_dw_grad_typed,
};
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::launch::{grid_1d, grid_norm};
use crate::mamba3_siso::gpu::forward_mixed::{
    GpuMamba3BackboneMixedActs, GpuMamba3LayerMixedActs, GpuMamba3MixedScratch,
};
use crate::mamba3_siso::gpu::kernels::Mamba3Kernels;
use crate::mamba3_siso::gpu::mamba3_gpu::{GpuMamba3Dims, GpuMamba3Scratch};
use crate::mamba3_siso::gpu::weights::{
    GpuMamba3Grads, GpuMamba3LayerGrads, GpuMamba3MixedLayerWeights,
};
use crate::mamba3_siso::gpu::weights_mixed_train::GpuMamba3TrainMixedWeights;

use cudarc::driver::{LaunchConfig, PushKernelArg};

#[inline]
fn tp(ptr: cudarc::driver::sys::CUdeviceptr, dtype: WeightDtype) -> TypedPtr {
    TypedPtr { ptr, dtype }
}

/// Mamba-3 SISO full backbone mixed-precision backward.
///
/// **IMPORTANT**: weight gradients in `grads` are **accumulated** (`beta=1.0`
/// on the dW GEMMs). Caller MUST call [`GpuMamba3Grads::zero`] before each
/// training step.
#[allow(clippy::too_many_arguments)]
pub fn gpu_backward_mamba3_backbone_mixed(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    d_temporal: &mut GpuBuffer,
    acts: &GpuMamba3BackboneMixedActs,
    mamba_w: &GpuMamba3TrainMixedWeights,
    grads: &GpuMamba3Grads,
    f32_scratch: &mut GpuMamba3Scratch,
    mixed_scratch: &mut GpuMamba3MixedScratch,
    dims: &GpuMamba3Dims,
) -> Result<(), String> {
    let bt = dims.bt();
    let dm = dims.d_model;
    let dtype = acts.dtype;

    // norm_f bwd (f32, residual stays f32) — Phase 2.7.5 Rule B.
    {
        let nf_ptr = mamba_w.master.norm_f_weight.raw_ptr(&ctx.stream);
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let axis0_ptr = f32_scratch.axis0_partials.cached_ptr();
        {
            let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_bwd);
            builder.arg(f32_scratch.d_norm.inner_mut());
            builder.arg(&axis0_ptr); // d_scale_partials [bt*dm]
            builder.arg(d_temporal.inner());
            builder.arg(acts.norm_f_input.inner());
            builder.arg(&nf_ptr);
            builder.arg(acts.norm_f_rms.inner());
            builder.arg(&bt_i);
            builder.arg(&dm_i);
            unsafe { builder.launch(grid_norm(bt, dm)) }
                .map_err(|e| format!("rmsnorm_bwd norm_f m3 mixed stage1: {:?}", e))?;
        }
        {
            let block_dim = (bt as u32).next_power_of_two().clamp(32, 256);
            let accumulate_i: i32 = 1;
            let d_nf_ptr = grads.norm_f_weight.ptr();
            let mut builder = ctx.stream.launch_builder(&m3k.reduce_sum_axis0);
            builder.arg(&d_nf_ptr);
            builder.arg(&axis0_ptr);
            builder.arg(&bt_i);
            builder.arg(&dm_i);
            builder.arg(&accumulate_i);
            let cfg = LaunchConfig {
                grid_dim: (dm as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("rmsnorm_bwd norm_f m3 mixed reduce: {:?}", e))?;
        }
    }
    d_temporal.copy_from_raw(&f32_scratch.d_norm, &ctx.stream)?;

    // Per-layer bwd in REVERSE order.
    for l in (0..dims.n_layers).rev() {
        gpu_backward_mamba3_layer_mixed(
            ctx,
            m3k,
            d_temporal,
            &acts.layers[l],
            &mamba_w.compute.layers[l],
            &mamba_w.master.layers[l],
            &grads.layers[l],
            f32_scratch,
            mixed_scratch,
            dims,
            dtype,
        )?;
    }

    // input_proj bwd skipped when identity (production config).
    if mamba_w.compute.input_proj_w.len_elems() > 0 {
        return Err("m3_mixed bwd: non-identity input_proj bwd not yet wired — \
             run with cpu.input_proj_w.clear() (identity branch)"
            .into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gpu_backward_mamba3_layer_mixed(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    d_temporal: &mut GpuBuffer,
    acts: &GpuMamba3LayerMixedActs,
    lw: &GpuMamba3MixedLayerWeights,
    lw_master: &crate::mamba3_siso::gpu::weights::GpuMamba3LayerWeights,
    lg: &GpuMamba3LayerGrads,
    sc: &mut GpuMamba3Scratch,
    msc: &mut GpuMamba3MixedScratch,
    dims: &GpuMamba3Dims,
    dtype: WeightDtype,
) -> Result<(), String> {
    if !dims.use_parallel_scan {
        return Err("m3_mixed bwd: sequential SSM bwd not yet supported".into());
    }
    if !dims.is_outproj_norm {
        return Err("m3_mixed bwd: silu_gate (no norm) not yet supported".into());
    }

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
    let bt_i = bt as i32;
    let dm_i = dm as i32;
    let di_i = di as i32;
    let ds_i = ds as i32;
    let nh_i = nh as i32;
    let hd_i = hd as i32;
    let ng_i = ng as i32;

    // ----------------------------------------------------------------
    // B8: out_proj backward.
    //   d_gated = d_temporal @ out_proj_w^T
    //   d_out_proj_w += gated^T @ d_temporal
    //
    // Both GEMMs typed: d_temporal needs to be staged typed. We use the
    // mixed-scratch d_temporal_typed (cast f32 → typed via cast kernel).
    // ----------------------------------------------------------------
    cast_f32_to_typed(ctx, m3k, &mut msc.d_temporal_typed, d_temporal, bt * dm)?;

    // dW: d_out_proj_w += gated^T @ d_temporal_typed (typed inputs, f32 grad out).
    gpu_sgemm_backward_dw_grad_typed(
        ctx,
        &lg.out_proj_w,
        tp(msc.d_temporal_typed.cached_ptr(), dtype),
        tp(acts.gated.cached_ptr(), dtype),
        bt,
        di,
        dm,
    )?;
    // dX: d_gated_typed = d_temporal_typed @ out_proj_w^T (typed weights).
    gpu_gemm_ex_backward_dx_typed(
        ctx,
        tp(msc.d_gated_typed.cached_ptr(), dtype),
        tp(msc.d_temporal_typed.cached_ptr(), dtype),
        tp(lw.out_proj_w.ptr(), dtype),
        bt,
        di,
        dm,
    )?;

    // ----------------------------------------------------------------
    // B7: RMSNormGated backward (Step 9c typed kernel).
    // ----------------------------------------------------------------
    {
        let nw_ptr = lw.norm_gate_weight.ptr();
        let grid = LaunchConfig {
            grid_dim: (bt as u32, 1, 1),
            block_dim: (di as u32, 1, 1),
            shared_mem_bytes: (di * std::mem::size_of::<f32>()) as u32,
        };
        let mut builder = ctx
            .stream
            .launch_builder(m3k.rmsnorm_gated_bwd_typed.get(dtype));
        let dyp = msc.d_y_typed.cached_ptr();
        let dzp = msc.d_z_typed.cached_ptr();
        let dgp = msc.d_gated_typed.cached_ptr();
        let yp = acts.y.cached_ptr();
        let zp = acts.z.cached_ptr();
        builder.arg(&dyp);
        builder.arg(&dzp);
        builder.arg(sc.d_norm_gate_w.inner_mut()); // f32 master grad accumulator
        builder.arg(&dgp);
        builder.arg(&yp);
        builder.arg(&zp);
        builder.arg(&nw_ptr);
        builder.arg(acts.gated_rms_vals.inner());
        builder.arg(&bt_i);
        builder.arg(&di_i);
        builder.arg(&hd_i);
        unsafe { builder.launch(grid) }
            .map_err(|e| format!("rmsnorm_gated_bwd typed B7: {:?}", e))?;
    }
    // Reduce d_norm_gate_w → lg.norm_gate_weight
    {
        let n_i = di as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        let dst = lg.norm_gate_weight.ptr();
        builder.arg(&dst);
        builder.arg(sc.d_norm_gate_w.inner());
        builder.arg(&bt_i);
        builder.arg(&n_i);
        unsafe { builder.launch(grid_1d(di)) }
            .map_err(|e| format!("colsum d_norm_gate_w mixed: {:?}", e))?;
    }

    // ----------------------------------------------------------------
    // B6: Chunked parallel SSM backward (Steps 9b + 9d typed).
    //   Reuses saved intermediates from forward (k_scaled_saved, etc.).
    // ----------------------------------------------------------------
    let dp_ptr = lw_master.d_param.raw_ptr(&ctx.stream);
    let cs = dims.chunk_size() as i32;
    let nc = dims.n_chunks();
    let cs_u = dims.chunk_size();

    // Load saved intermediates from typed/f32 fwd save into f32 scratch
    // (the parallel bwd kernels read these from f32 buffers).
    sc.da_cumsum
        .copy_from_raw(&acts.da_cumsum_saved, &ctx.stream)?;
    sc.d_scale.copy_from_raw(&acts.scale_saved, &ctx.stream)?;
    sc.d_gamma_par
        .copy_from_raw(&acts.gamma_saved, &ctx.stream)?;
    sc.d_qk_dot.copy_from_raw(&acts.qk_dot_saved, &ctx.stream)?;
    sc.chunk_states
        .copy_from_raw(&acts.chunk_states_saved, &ctx.stream)?;

    // Extract per-chunk dA cumsum end values.
    {
        let block_x = nh.min(256) as u32;
        let grid_z = nh.div_ceil(block_x as usize) as u32;
        let cfg = LaunchConfig {
            grid_dim: (dims.batch as u32, nc as u32, grid_z),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = ctx.stream.launch_builder(&m3k.m3_extract_da_cs_sum);
        builder.arg(sc.da_cs_sum.inner_mut());
        builder.arg(sc.da_cumsum.inner());
        builder.arg(&b);
        builder.arg(&t);
        builder.arg(&nh_i);
        builder.arg(&cs);
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("m3_extract_da_cs_sum mixed: {:?}", e))?;
    }

    // Zero output buffers for m3_dqkv_typed.
    {
        let zero: f32 = 0.0;
        for (buf, sz, label) in [
            (&mut sc.d_x, bt * di, "d_x"),
            (&mut sc.d_k, bt * nh * ds, "d_k"),
            (&mut sc.d_q, bt * nh * ds, "d_q"),
        ] {
            let ne = sz as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.fill_scalar);
            builder.arg(buf.inner_mut());
            builder.arg(&zero);
            builder.arg(&ne);
            unsafe { builder.launch(grid_1d(sz)) }
                .map_err(|e| format!("zero {label} mixed: {:?}", e))?;
        }
    }

    // m3_dqkv typed — Phase 2.7.5 Rule B: dD_partials[B*nh] via axis0_partials,
    // followed by reduce_sum_axis0 → lg.d_param[nh] (accumulate=1).
    {
        let smem = (cs_u * ds + cs_u * ds + cs_u * hd + cs_u * hd + cs_u + cs_u + hd * ds) * 4;
        let cfg = LaunchConfig {
            grid_dim: (nh as u32, dims.batch as u32, 1),
            block_dim: (hd as u32, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        let mut builder = ctx.stream.launch_builder(m3k.m3_dqkv_typed.get(dtype));
        builder.arg(sc.d_q.inner_mut()); // dQ_mid
        builder.arg(sc.d_k.inner_mut()); // dK_mid
        builder.arg(sc.d_x.inner_mut()); // dV
        builder.arg(sc.d_alpha.inner_mut()); // dADT
        builder.arg(sc.d_beta.inner_mut()); // dQK_dot
        builder.arg(sc.axis0_partials.inner_mut()); // dD_partials [B*nh] (Phase 2.7.5)
        let q_p = acts.q.cached_ptr(); // typed Q_rot
        let ks_p = acts.k_scaled_saved.cached_ptr(); // typed K_scaled
        let v_p = acts.x.cached_ptr(); // typed V = x
        let dy_p = msc.d_y_typed.cached_ptr(); // typed dO = d_y
        builder.arg(&q_p);
        builder.arg(&ks_p);
        builder.arg(&v_p);
        builder.arg(sc.da_cumsum.inner());
        builder.arg(sc.da_cs_sum.inner());
        builder.arg(sc.d_qk_dot.inner());
        builder.arg(sc.chunk_states.inner());
        builder.arg(&dy_p);
        builder.arg(&dp_ptr);
        builder.arg(&b);
        builder.arg(&t);
        builder.arg(&nh_i);
        builder.arg(&hd_i);
        builder.arg(&ds_i);
        builder.arg(&cs);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_dqkv_typed B6: {:?}", e))?;
    }
    // Stage 2: reduce dD_partials[B, nh] → lg.d_param[nh] (accumulate=1).
    {
        let block_dim = (dims.batch as u32).next_power_of_two().clamp(32, 256);
        let accumulate_i: i32 = 1;
        let d_dp_ptr = lg.d_param.ptr();
        let mut rb = ctx.stream.launch_builder(&m3k.reduce_sum_axis0);
        rb.arg(&d_dp_ptr);
        rb.arg(sc.axis0_partials.inner());
        rb.arg(&b);
        rb.arg(&nh_i);
        rb.arg(&accumulate_i);
        let red_cfg = LaunchConfig {
            grid_dim: (nh as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
        };
        unsafe { rb.launch(red_cfg) }.map_err(|e| format!("m3_dqkv_typed dD reduce: {:?}", e))?;
    }

    // Zero d_angle_cumsum before m3_dqktheta.
    if na > 0 {
        let n_ac = (dims.batch * dims.seq_len * nh * na) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.fill_scalar);
        builder.arg(sc.d_angle_cumsum.inner_mut());
        let zero: f32 = 0.0;
        builder.arg(&zero);
        builder.arg(&n_ac);
        unsafe { builder.launch(grid_1d(n_ac as usize)) }
            .map_err(|e| format!("zero d_angle_cumsum mixed: {:?}", e))?;
    }

    // m3_dqktheta typed — Phase 2.7.5: dQ_bias/dK_bias removed from args
    // (caller does colsum_accumulate on dQ_pre/dK_pre scratch below).
    {
        let na_i = na as i32;
        let cfg = LaunchConfig {
            grid_dim: ((dims.batch * nc) as u32, nh as u32, 1),
            block_dim: (cs_u as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = ctx.stream.launch_builder(m3k.m3_dqktheta_typed.get(dtype));
        builder.arg(sc.d_c_pre_rope.inner_mut());
        builder.arg(sc.d_b_pre_rope.inner_mut());
        builder.arg(sc.d_angle_cumsum.inner_mut());
        let scale_in_ptr = sc.d_scale.raw_ptr(&ctx.stream);
        let gamma_in_ptr = sc.d_gamma_par.raw_ptr(&ctx.stream);
        builder.arg(sc.d_scale.inner_mut());
        builder.arg(sc.d_gamma_par.inner_mut());
        let cb_p = acts.c_biased.cached_ptr(); // typed Q_raw
        let bb_p = acts.b_biased.cached_ptr(); // typed K_raw
        builder.arg(&cb_p);
        builder.arg(&bb_p);
        builder.arg(&scale_in_ptr);
        builder.arg(&gamma_in_ptr);
        builder.arg(acts.angle_cumsum.inner());
        builder.arg(sc.d_q.inner());
        builder.arg(sc.d_k.inner());
        builder.arg(sc.d_beta.inner());
        builder.arg(&b);
        builder.arg(&t);
        builder.arg(&nh_i);
        builder.arg(&ds_i);
        builder.arg(&na_i);
        builder.arg(&cs);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_dqktheta_typed B6: {:?}", e))?;
    }
    // Phase 2.7.5: colsum dQ_pre / dK_pre → c_bias / b_bias (deterministic).
    {
        let d_cb_ptr = lg.c_bias.ptr();
        let nhds_i = (nh * ds) as i32;
        let mut cb = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        cb.arg(&d_cb_ptr);
        cb.arg(sc.d_c_pre_rope.inner());
        cb.arg(&bt_i);
        cb.arg(&nhds_i);
        unsafe { cb.launch(grid_1d(nh * ds)) }
            .map_err(|e| format!("colsum d_c_bias mixed: {:?}", e))?;
    }
    {
        let d_bb_ptr = lg.b_bias.ptr();
        let nhds_i = (nh * ds) as i32;
        let mut cb = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        cb.arg(&d_bb_ptr);
        cb.arg(sc.d_b_pre_rope.inner());
        cb.arg(&bt_i);
        cb.arg(&nhds_i);
        unsafe { cb.launch(grid_1d(nh * ds)) }
            .map_err(|e| format!("colsum d_b_bias mixed: {:?}", e))?;
    }

    // Copy dQ_pre/dK_pre into d_q/d_k for downstream consumption (matches
    // f32 path semantics — d_q/d_k are the post-bias-removal grads).
    sc.d_q.copy_from_raw(&sc.d_c_pre_rope, &ctx.stream)?;
    sc.d_k.copy_from_raw(&sc.d_b_pre_rope, &ctx.stream)?;

    // S3: m3_ddt_dtrap (pure f32) — produces dDT, dTrap.
    {
        let mut builder = ctx.stream.launch_builder(&m3k.m3_ddt_dtrap);
        builder.arg(sc.d_gamma.inner_mut()); // dDT → d_gamma scratch
        builder.arg(sc.d_trap_raw.inner_mut());
        builder.arg(sc.d_scale.inner());
        builder.arg(sc.d_gamma_par.inner());
        builder.arg(acts.dt.inner());
        builder.arg(acts.trap.inner());
        builder.arg(&b);
        builder.arg(&t);
        builder.arg(&nh_i);
        unsafe { builder.launch(grid_1d(bt * nh)) }
            .map_err(|e| format!("m3_ddt_dtrap mixed: {:?}", e))?;
    }

    // Re-copy d_b_pre_rope/d_c_pre_rope from d_k/d_q for B4 head→group reduce.
    sc.d_b_pre_rope.copy_from_raw(&sc.d_k, &ctx.stream)?;
    sc.d_c_pre_rope.copy_from_raw(&sc.d_q, &ctx.stream)?;

    // ----------------------------------------------------------------
    // B5a: angle_dt_bwd — Phase 2.7.5 Rule B (pure f32, no atomicAdd).
    // Stage 1: kernel writes contrib_angles + contrib_dt into axis0_partials.
    // Stage 2a/b: reduce_sum_axis0 → d_angles_raw / d_dt_angle.
    // ----------------------------------------------------------------
    if na > 0 {
        let na_i = na as i32;
        let btna = (bt * na) as i32;
        let btnh = (bt * nh) as i32;
        let contrib_angles_elems = nh * bt * na;
        let contrib_dt_offset_bytes = (contrib_angles_elems * std::mem::size_of::<f32>()) as u64;
        let contrib_angles_ptr = sc.axis0_partials.cached_ptr();
        let contrib_dt_ptr = sc.axis0_partials.cached_ptr() + contrib_dt_offset_bytes;
        // Stage 1
        {
            let mut builder = ctx.stream.launch_builder(&m3k.m3_angle_dt_bwd_seq);
            builder.arg(&contrib_angles_ptr);
            builder.arg(&contrib_dt_ptr);
            builder.arg(sc.d_angle_cumsum.inner());
            builder.arg(acts.angles_raw.inner());
            builder.arg(acts.dt.inner());
            builder.arg(&b);
            builder.arg(&t);
            builder.arg(&nh_i);
            builder.arg(&na_i);
            let grid = LaunchConfig {
                grid_dim: (dims.batch as u32, (nh * na).div_ceil(256) as u32, 1),
                block_dim: (256.min((nh * na) as u32), 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(grid) }
                .map_err(|e| format!("m3_angle_dt_bwd_seq mixed stage1: {:?}", e))?;
        }
        // Stage 2a: reduce nh → d_angles_raw[B*T*na]
        {
            let block_dim = (nh as u32).next_power_of_two().clamp(32, 256);
            let accumulate_i: i32 = 0;
            let mut builder = ctx.stream.launch_builder(&m3k.reduce_sum_axis0);
            builder.arg(sc.d_angles_raw.inner_mut());
            builder.arg(&contrib_angles_ptr);
            builder.arg(&nh_i);
            builder.arg(&btna);
            builder.arg(&accumulate_i);
            let cfg = LaunchConfig {
                grid_dim: ((bt * na) as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("angle_dt_bwd mixed reduce angles: {:?}", e))?;
        }
        // Stage 2b: reduce na → d_dt_angle[B*T*nh]
        {
            let block_dim = (na as u32).next_power_of_two().clamp(32, 256);
            let accumulate_i: i32 = 0;
            let mut builder = ctx.stream.launch_builder(&m3k.reduce_sum_axis0);
            builder.arg(sc.d_dt_angle.inner_mut());
            builder.arg(&contrib_dt_ptr);
            builder.arg(&na_i);
            builder.arg(&btnh);
            builder.arg(&accumulate_i);
            let cfg = LaunchConfig {
                grid_dim: ((bt * nh) as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("angle_dt_bwd mixed reduce dt: {:?}", e))?;
        }
    } else {
        let ne = (bt * nh) as i32;
        let zero: f32 = 0.0;
        let mut builder = ctx.stream.launch_builder(&m3k.fill_scalar);
        builder.arg(sc.d_dt_angle.inner_mut());
        builder.arg(&zero);
        builder.arg(&ne);
        unsafe { builder.launch(grid_1d(bt * nh)) }
            .map_err(|e| format!("zero d_dt_angle mixed: {:?}", e))?;
    }

    // ----------------------------------------------------------------
    // B5b: m3_final_grads (pure f32) — combine dADT + dDT + dDT_angle.
    // ----------------------------------------------------------------
    {
        let n_total = (bt * nh) as i32;
        let dtb_ptr = lw_master.dt_bias.raw_ptr(&ctx.stream);
        let mut builder = ctx.stream.launch_builder(&m3k.m3_final_grads);
        builder.arg(sc.d_dd_dt.inner_mut());
        builder.arg(sc.d_dd_a.inner_mut());
        builder.arg(sc.d_alpha.inner());
        builder.arg(sc.d_gamma.inner());
        builder.arg(sc.d_dt_angle.inner());
        builder.arg(acts.a_val.inner());
        builder.arg(acts.dt.inner());
        builder.arg(acts.dd_dt_raw.inner());
        builder.arg(acts.dd_a_raw.inner());
        builder.arg(&dtb_ptr);
        builder.arg(&dims.a_floor);
        builder.arg(&n_total);
        builder.arg(&nh_i);
        unsafe { builder.launch(grid_1d(bt * nh)) }
            .map_err(|e| format!("m3_final_grads mixed: {:?}", e))?;
    }

    // dt_bias gradient = colsum of d_dd_dt.
    {
        let d_dtb_ptr = lg.dt_bias.ptr();
        let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        builder.arg(&d_dtb_ptr);
        builder.arg(sc.d_dd_dt.inner());
        builder.arg(&bt_i);
        builder.arg(&nh_i);
        unsafe { builder.launch(grid_1d(nh)) }
            .map_err(|e| format!("colsum d_dt_bias mixed: {:?}", e))?;
    }

    // ----------------------------------------------------------------
    // B4b: bc_bias_add_bwd — head→group reduce.
    //
    // We use the f32 kernel (not the typed variant): bc_bias_add_bwd_typed
    // would require both input AND output to be typed, but the upstream
    // d_b_pre_rope is f32 (output of m3_dqktheta which writes f32 master
    // grads), and we need d_b_normed staged for the next typed kernel
    // (bcnorm_bwd_typed). Casting f32→typed→f32 here adds 2 kernel
    // launches per side for no precision benefit. The f32 reduce is also
    // cheap (head→group sum, no math). v0.3 cleanup: route m3_dqktheta
    // output via typed scratch so this whole section can become typed.
    {
        let mut builder = ctx.stream.launch_builder(&m3k.bc_bias_add_bwd);
        builder.arg(sc.d_b_normed.inner_mut());
        builder.arg(sc.d_b_pre_rope.inner());
        builder.arg(&bt_i);
        builder.arg(&nh_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(bt * ng * ds)) }
            .map_err(|e| format!("bc_bias_add_bwd B mixed: {:?}", e))?;
    }
    {
        let mut builder = ctx.stream.launch_builder(&m3k.bc_bias_add_bwd);
        builder.arg(sc.d_c_normed.inner_mut());
        builder.arg(sc.d_c_pre_rope.inner());
        builder.arg(&bt_i);
        builder.arg(&nh_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(bt * ng * ds)) }
            .map_err(|e| format!("bc_bias_add_bwd C mixed: {:?}", e))?;
    }

    // ----------------------------------------------------------------
    // B4c: bcnorm_bwd_typed — typed d_out + typed d_B output.
    // ----------------------------------------------------------------
    // Cast d_b_normed (f32) to typed for the typed bcnorm_bwd input.
    cast_f32_to_typed(
        ctx,
        m3k,
        &mut msc.d_b_normed_typed,
        &sc.d_b_normed,
        bt * ng * ds,
    )?;
    cast_f32_to_typed(
        ctx,
        m3k,
        &mut msc.d_c_normed_typed,
        &sc.d_c_normed,
        bt * ng * ds,
    )?;

    // BCNorm bwd for B (typed).
    {
        let bn_ptr = lw.b_norm_weight.ptr();
        let n_i = bt as i32;
        let cfg = LaunchConfig {
            grid_dim: ((bt * ng) as u32, 1, 1),
            block_dim: (ds as u32, 1, 1),
            shared_mem_bytes: ds as u32 * 4,
        };
        let mut builder = ctx.stream.launch_builder(m3k.bcnorm_bwd_typed.get(dtype));
        let dbp = msc.d_b_raw_typed.cached_ptr(); // typed output d_B
        let donp = msc.d_b_normed_typed.cached_ptr(); // typed d_out
        let brp = acts.b_raw.cached_ptr(); // typed B_raw
        builder.arg(&dbp);
        builder.arg(sc.d_b_norm_w.inner_mut());
        builder.arg(&donp);
        builder.arg(&brp);
        builder.arg(acts.b_rms.inner());
        builder.arg(&bn_ptr);
        builder.arg(&n_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("bcnorm_bwd_typed B B4c: {:?}", e))?;
    }
    // Reduce d_b_norm_w → lg.b_norm_weight.
    {
        let d_bnw_ptr = lg.b_norm_weight.ptr();
        let rows = (bt * ng) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        builder.arg(&d_bnw_ptr);
        builder.arg(sc.d_b_norm_w.inner());
        builder.arg(&rows);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(ds)) }
            .map_err(|e| format!("colsum d_b_norm_w mixed: {:?}", e))?;
    }

    // BCNorm bwd for C (typed).
    {
        let cn_ptr = lw.c_norm_weight.ptr();
        let n_i = bt as i32;
        let cfg = LaunchConfig {
            grid_dim: ((bt * ng) as u32, 1, 1),
            block_dim: (ds as u32, 1, 1),
            shared_mem_bytes: ds as u32 * 4,
        };
        let mut builder = ctx.stream.launch_builder(m3k.bcnorm_bwd_typed.get(dtype));
        let dcp = msc.d_c_raw_typed.cached_ptr();
        let donp = msc.d_c_normed_typed.cached_ptr();
        let crp = acts.c_raw.cached_ptr();
        builder.arg(&dcp);
        builder.arg(sc.d_c_norm_w.inner_mut());
        builder.arg(&donp);
        builder.arg(&crp);
        builder.arg(acts.c_rms.inner());
        builder.arg(&cn_ptr);
        builder.arg(&n_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("bcnorm_bwd_typed C B4c: {:?}", e))?;
    }
    {
        let d_cnw_ptr = lg.c_norm_weight.ptr();
        let rows = (bt * ng) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        builder.arg(&d_cnw_ptr);
        builder.arg(sc.d_c_norm_w.inner());
        builder.arg(&rows);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(ds)) }
            .map_err(|e| format!("colsum d_c_norm_w mixed: {:?}", e))?;
    }

    // ----------------------------------------------------------------
    // B3: m3_split_bwd_typed — assemble typed d_proj from 8 grads.
    // typed: d_z, d_x, d_b_raw, d_c_raw + f32: dd_dt/a/trap/angles
    // ----------------------------------------------------------------
    // d_x is f32 from m3_dqkv. Cast to typed for split_bwd.
    // Wait — d_z is the typed d_z from B7 (rmsnorm_gated_bwd_typed wrote
    // typed d_z_typed). d_x is f32 from m3_dqkv (master grad). For
    // split_bwd we need typed d_x. Cast.
    // Reuse d_y_typed as scratch for typed d_x (it's bt*di typed, same size).
    cast_f32_to_typed(ctx, m3k, &mut msc.d_y_typed, &sc.d_x, bt * di)?;
    {
        let na_i = na as i32;
        let mut builder = ctx.stream.launch_builder(m3k.m3_split_bwd_typed.get(dtype));
        let dprojp = msc.d_proj_typed.cached_ptr();
        let dzp = msc.d_z_typed.cached_ptr();
        let dxp_typed = msc.d_y_typed.cached_ptr(); // typed d_x via temp
        let dbrp = msc.d_b_raw_typed.cached_ptr();
        let dcrp = msc.d_c_raw_typed.cached_ptr();
        builder.arg(&dprojp);
        builder.arg(&dzp);
        builder.arg(&dxp_typed);
        builder.arg(&dbrp);
        builder.arg(&dcrp);
        builder.arg(sc.d_dd_dt.inner());
        builder.arg(sc.d_dd_a.inner());
        builder.arg(sc.d_trap_raw.inner());
        builder.arg(sc.d_angles_raw.inner());
        builder.arg(&bt_i);
        builder.arg(&di_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        builder.arg(&nh_i);
        builder.arg(&na_i);
        unsafe { builder.launch(grid_1d(bt * ip)) }
            .map_err(|e| format!("m3_split_bwd_typed B3: {:?}", e))?;
    }

    // ----------------------------------------------------------------
    // B2: in_proj backward (typed GEMMs).
    //   d_post_norm_typed = d_proj_typed @ in_proj_w^T
    //   d_in_proj_w += post_norm^T @ d_proj_typed
    // ----------------------------------------------------------------
    gpu_sgemm_backward_dw_grad_typed(
        ctx,
        &lg.in_proj_w,
        tp(msc.d_proj_typed.cached_ptr(), dtype),
        tp(acts.post_norm.cached_ptr(), dtype),
        bt,
        dm,
        ip,
    )?;
    gpu_gemm_ex_backward_dx_typed(
        ctx,
        tp(msc.d_post_norm_typed.cached_ptr(), dtype),
        tp(msc.d_proj_typed.cached_ptr(), dtype),
        tp(lw.in_proj_w.ptr(), dtype),
        bt,
        dm,
        ip,
    )?;

    // ----------------------------------------------------------------
    // B1: rmsnorm_bwd (f32-only kernel).
    // Cast typed d_post_norm → f32 d_norm scratch, then call f32 rmsnorm_bwd.
    // ----------------------------------------------------------------
    cast_typed_to_f32(ctx, m3k, &mut sc.d_norm, &msc.d_post_norm_typed, bt * dm)?;
    // Phase 2.7.5 Rule B two-stage:
    {
        let nw_ptr = lw_master.norm_weight.raw_ptr(&ctx.stream);
        let axis0_ptr = sc.axis0_partials.cached_ptr();
        {
            let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_bwd);
            builder.arg(sc.d_pre_norm.inner_mut());
            builder.arg(&axis0_ptr); // d_scale_partials [bt*dm]
            builder.arg(sc.d_norm.inner());
            builder.arg(acts.residual.inner());
            builder.arg(&nw_ptr);
            builder.arg(acts.rms_vals.inner());
            builder.arg(&bt_i);
            builder.arg(&dm_i);
            unsafe { builder.launch(grid_norm(bt, dm)) }
                .map_err(|e| format!("rmsnorm_bwd m3 B1 mixed stage1: {:?}", e))?;
        }
        {
            let block_dim = (bt as u32).next_power_of_two().clamp(32, 256);
            let accumulate_i: i32 = 1;
            let d_nw_ptr = lg.norm_weight.ptr();
            let mut builder = ctx.stream.launch_builder(&m3k.reduce_sum_axis0);
            builder.arg(&d_nw_ptr);
            builder.arg(&axis0_ptr);
            builder.arg(&bt_i);
            builder.arg(&dm_i);
            builder.arg(&accumulate_i);
            let cfg = LaunchConfig {
                grid_dim: (dm as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("rmsnorm_bwd m3 B1 mixed reduce: {:?}", e))?;
        }
    }

    // Residual: d_temporal += d_pre_norm.
    {
        let ne = (bt * dm) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.vec_add_inplace);
        builder.arg(d_temporal.inner_mut());
        builder.arg(sc.d_pre_norm.inner());
        builder.arg(&ne);
        unsafe { builder.launch(grid_1d(bt * dm)) }
            .map_err(|e| format!("vec_add d_temporal mixed: {:?}", e))?;
    }

    Ok(())
}

// ----------------------------------------------------------------
// dtype cast helpers
// ----------------------------------------------------------------

fn cast_f32_to_typed(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    dst: &mut crate::mamba_ssm::gpu::buffers::DtypedBuf,
    src: &GpuBuffer,
    n: usize,
) -> Result<(), String> {
    use crate::mamba_ssm::gpu::dtype::WeightDtype as WD;
    let n_i = n as i32;
    let kernel = match dst.dtype() {
        WD::F32 => {
            // Identity D2D copy.
            let bytes = n * 4;
            let stream = ctx.stream.cu_stream();
            unsafe {
                let r = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dst.cached_ptr(),
                    src.cached_ptr(),
                    bytes,
                    stream,
                );
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("cast f32→f32 D2D: {r:?}"));
                }
            }
            return Ok(());
        }
        WD::Bf16 => &m3k.cast_f32_to_bf16,
        WD::F16 => &m3k.cast_f32_to_f16,
    };
    let mut builder = ctx.stream.launch_builder(kernel);
    let dstp = dst.cached_ptr();
    let srcp = src.cached_ptr();
    builder.arg(&dstp);
    builder.arg(&srcp);
    builder.arg(&n_i);
    unsafe { builder.launch(grid_1d(n)) }.map_err(|e| format!("cast_f32_to_typed: {:?}", e))?;
    Ok(())
}

fn cast_typed_to_f32(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    dst: &mut GpuBuffer,
    src: &crate::mamba_ssm::gpu::buffers::DtypedBuf,
    n: usize,
) -> Result<(), String> {
    use crate::mamba_ssm::gpu::dtype::WeightDtype as WD;
    let n_i = n as i32;
    let kernel = match src.dtype() {
        WD::F32 => {
            let bytes = n * 4;
            let stream = ctx.stream.cu_stream();
            unsafe {
                let r = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dst.cached_ptr(),
                    src.cached_ptr(),
                    bytes,
                    stream,
                );
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("cast f32→f32 D2D: {r:?}"));
                }
            }
            return Ok(());
        }
        WD::Bf16 => &m3k.cast_bf16_to_f32,
        WD::F16 => &m3k.cast_f16_to_f32,
    };
    let mut builder = ctx.stream.launch_builder(kernel);
    let dstp = dst.cached_ptr();
    let srcp = src.cached_ptr();
    builder.arg(&dstp);
    builder.arg(&srcp);
    builder.arg(&n_i);
    unsafe { builder.launch(grid_1d(n)) }.map_err(|e| format!("cast_typed_to_f32: {:?}", e))?;
    Ok(())
}
