//! Mamba-3 SISO mixed-precision (bf16/f16) backward training
//! pipeline. Mirrors [`super::mamba3_gpu::gpu_backward_mamba3_layer`] +
//! `_backbone` with typed activation I/O routed through the typed kernel
//! variants from Steps 9a/9b/9c/9d.
//!
//! ## Production-config scope
//! - `dims.use_parallel_scan = true` (chunked SSM bwd via Steps 9b + 9d)
//! - `dims.is_outproj_norm = true` (the typed gated-norm backward)
//! - `dims.n_angles > 0` (the typed RoPE backward)
//!
//! Non-production configs return `Err(...)` — fall back to the f32 path
//! [`super::mamba3_gpu::gpu_backward_mamba3_backbone`].
//!
//! ## Precision invariants
//! - **Activations**: typed (bf16/f16) — read via `cached_ptr()` from
//!   `GpuMamba3LayerMixedActs` and written to typed scratch buffers in
//!   `GpuMamba3MixedScratch`.
//! - **Master grads** (in `GpuMamba3LayerGrads`): all stay f32, the
//!   master-gradient invariant.
//! - **Activation grads on the wire**: typed where they cross typed-kernel
//!   boundaries (d_gated, d_y, d_z, d_b_normed, d_c_normed, d_b_raw,
//!   d_c_raw, d_proj, d_post_norm, d_temporal_typed); otherwise f32 in
//!   the existing `GpuMamba3Scratch`.

use crate::mamba_ssm::gpu::blas::{
    TypedPtr, gpu_gemm_bi_backward_dw_grad_typed, gpu_gemm_ex_backward_dx_typed,
};
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::launch::{grid_1d, grid_norm};
use crate::mamba3_siso::gpu::forward_mixed::{
    GpuMamba3BackboneMixedActs, GpuMamba3LayerMixedActs, GpuMamba3MixedScratch,
};
use crate::mamba3_siso::gpu::kernels::Mamba3Kernels;
use crate::mamba3_siso::gpu::mamba3_gpu::GpuMamba3Scratch;
use crate::mamba3_siso::gpu::state::M3Exec;
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
pub fn gpu_backward_mamba3_backbone_mixed(
    exec: &M3Exec<'_>,
    d_temporal: &mut GpuBuffer,
    acts: &GpuMamba3BackboneMixedActs,
    mamba_w: &GpuMamba3TrainMixedWeights,
    grads: &GpuMamba3Grads,
    f32_scratch: &mut GpuMamba3Scratch,
    mixed_scratch: &mut GpuMamba3MixedScratch,
) -> Result<(), String> {
    let M3Exec {
        ctx,
        kernels: m3k,
        dims,
    } = *exec;
    let bt = dims.bt();
    let dm = dims.d_model;
    let dtype = acts.dtype;

    // norm_f bwd (f32, residual stays f32) — no-atomics partials.
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
            // Shared-kernel ABI: rmsnorm_backward grew an `accumulate`
            // arg (M1 residual fold); the M3 sites keep the
            // plain-store behavior.
            let accumulate_dx: i32 = 0;
            builder.arg(&accumulate_dx);
            let no_mirror: cudarc::driver::sys::CUdeviceptr = 0;
            builder.arg(&no_mirror);
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
            exec,
            d_temporal,
            &M3MixedLayerBwd {
                acts: &acts.layers[l],
                lw: &mamba_w.compute.layers[l],
                lw_master: &mamba_w.master.layers[l],
                lg: &grads.layers[l],
            },
            f32_scratch,
            mixed_scratch,
            dtype,
        )?;
    }

    // input_proj backward — identity leaves it untouched; non-identity
    // computes dW/db only. The dX below the projection is deliberately
    // discarded: the input is data, nothing trainable sits under it.
    if mamba_w.compute.input_proj_w.len_elems() > 0 {
        let mid = dims.mamba_input_dim;
        // d_temporal now holds the f32 gradient w.r.t. the input_proj
        // OUTPUT (every layer's dX has been folded back into it). Cast it
        // to the compute dtype, reusing the saved typed OUTPUTS buffer —
        // its forward value has no remaining consumer at this point.
        let dy_ptr = acts.input_proj_outputs.cached_ptr();
        {
            let n = (bt * dm) as i32;
            let cast = match dtype {
                WeightDtype::Bf16 => &m3k.cast_f32_to_bf16,
                WeightDtype::F16 => &m3k.cast_f32_to_f16,
                WeightDtype::F32 => {
                    return Err("m3_mixed backward: unexpected f32 compute dtype".into());
                }
            };
            let src = d_temporal.cached_ptr();
            let mut bld = ctx.stream.launch_builder(cast);
            bld.arg(&dy_ptr);
            bld.arg(&src);
            bld.arg(&n);
            unsafe { bld.launch(grid_1d(bt * dm)) }
                .map_err(|e| format!("m3_mixed input_proj dY cast: {e:?}"))?;
        }
        // db: typed accumulating reduction of dY over (b, t) into the f32
        // grad slice (deterministic — one block per bias index, no atomics).
        {
            let bt_i = bt as i32;
            let dm_i = dm as i32;
            let mut bld = ctx
                .stream
                .launch_builder(ctx.kernels.reduce_bias_typed.get(dtype));
            let db = grads.input_proj_b.ptr();
            bld.arg(&db);
            bld.arg(&dy_ptr);
            bld.arg(&bt_i);
            bld.arg(&dm_i);
            let threads = 256u32;
            let cfg = LaunchConfig {
                grid_dim: (dm as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: (threads as usize * std::mem::size_of::<f32>()) as u32,
            };
            unsafe { bld.launch(cfg) }
                .map_err(|e| format!("m3_mixed reduce_bias input_proj: {e:?}"))?;
        }
        // dW: saved typed inputs^T @ dY, accumulated into the f32 grad slice.
        gpu_gemm_bi_backward_dw_grad_typed(
            ctx,
            &grads.input_proj_w,
            TypedPtr { ptr: dy_ptr, dtype },
            TypedPtr {
                ptr: acts.input_proj_inputs.cached_ptr(),
                dtype,
            },
            bt,
            mid,
            dm,
        )?;
    }
    Ok(())
}

/// Per-layer bundle for the M3 mixed backward: saved activations, compute
/// + master weights, and the layer's master-grad slots.
#[derive(Clone, Copy)]
struct M3MixedLayerBwd<'a> {
    acts: &'a GpuMamba3LayerMixedActs,
    lw: &'a GpuMamba3MixedLayerWeights,
    lw_master: &'a crate::mamba3_siso::gpu::weights::GpuMamba3LayerWeights,
    lg: &'a GpuMamba3LayerGrads,
}

fn gpu_backward_mamba3_layer_mixed(
    exec: &M3Exec<'_>,
    d_temporal: &mut GpuBuffer,
    layer: &M3MixedLayerBwd<'_>,
    sc: &mut GpuMamba3Scratch,
    msc: &mut GpuMamba3MixedScratch,
    dtype: WeightDtype,
) -> Result<(), String> {
    let M3Exec {
        ctx,
        kernels: m3k,
        dims,
    } = *exec;
    let M3MixedLayerBwd {
        acts,
        lw,
        lw_master,
        lg,
    } = *layer;
    if !dims.use_parallel_scan {
        // Deliberate non-goal, not a gap: a sequential mixed backward
        // needs the full per-timestep h tape (B*(T+1)*d_inner*d_state
        // floats — CHUNK_SIZE times the chunked tape), and the chunked
        // path is bitwise-deterministic and faster at every measured T.
        // Revisit only if one of these appears: (a) a model whose
        // recurrence cannot be chunk-decomposed, (b) a debugging need
        // for per-step gradient taps, (c) a VRAM regime where the
        // chunked saves no longer fit but the sequential tape somehow
        // would (it will not at current shapes).
        return Err(
            "m3_mixed bwd: sequential SSM bwd is unsupported by design; \
             use the chunked parallel scan (ScanMode::Auto)"
                .into(),
        );
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
    gpu_gemm_bi_backward_dw_grad_typed(
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
    // B7: gate backward — gated-RMSNorm or plain SiLU gate, matching
    // the forward's output-stage choice.
    // ----------------------------------------------------------------
    if dims.is_outproj_norm {
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
    } else {
        // Plain SiLU gate: no norm weight to accumulate; d_y and d_z
        // come straight from the gate derivative. The kernel shares the
        // f32 twin's factored d_silu form so all three precisions stay
        // in lockstep with the CPU reference.
        let n = bt * di;
        let n_i = n as i32;
        let mut builder = ctx
            .stream
            .launch_builder(m3k.silu_gate_bwd_typed.get(dtype));
        let dyp = msc.d_y_typed.cached_ptr();
        let dzp = msc.d_z_typed.cached_ptr();
        let dgp = msc.d_gated_typed.cached_ptr();
        let yp = acts.y.cached_ptr();
        let zp = acts.z.cached_ptr();
        builder.arg(&dyp);
        builder.arg(&dzp);
        builder.arg(&dgp);
        builder.arg(&yp);
        builder.arg(&zp);
        builder.arg(&n_i);
        unsafe { builder.launch(grid_1d(n)) }
            .map_err(|e| format!("silu_gate_bwd typed B7: {:?}", e))?;
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

    // m3_dqkv typed — no-atomics partials rule: dD_partials[B*nh] via axis0_partials,
    // followed by reduce_sum_axis0 → lg.d_param[nh] (accumulate=1).
    {
        // Two chunk-by-state operand tiles, V/dO tiles, two per-step
        // lanes, the da/qk lanes, and TWO head-state tiles (true
        // states + d_state staging for the warp-parallel dADT).
        let legacy_floats = 2 * cs_u * ds + 2 * cs_u * hd + 4 * cs_u + 2 * hd * ds;
        let mats_floats = legacy_floats + cs_u * (cs_u - 1) * 3 / 2 + 2 * cs_u;
        // Consumer GPUs cap the per-block dynamic-smem opt-in near 99 KB.
        // One head per block: pair matrices when the tile
        // fits, legacy inline dots otherwise (bit-identical, slower).
        let cap_floats = 99 * 1024 / 4;
        let (per_head_floats, use_pair_mats): (usize, i32) = if mats_floats <= cap_floats {
            (mats_floats, 1)
        } else {
            (legacy_floats, 0)
        };
        let smem = per_head_floats * 4;
        if smem > 99 * 1024 {
            return Err(format!(
                "m3_dqkv shared-memory tile {} B exceeds the 99 KB opt-in \
                     (CS={cs_u} hd={hd} ds={ds}) even without the pair matrices",
                smem
            ));
        }
        // t-split: blockDim.y lanes stride the per-timestep
        // loops. Fixed launch geometry (never data-shaped); any T_SPLIT
        // yields identical bits since each output keeps one owning lane
        // with the same inner order.
        let nc = dims.n_chunks();
        // Reverse d_state decomposition (see the f32 lane).
        {
            let st_cfg = LaunchConfig {
                grid_dim: (nh as u32, dims.batch as u32, nc as u32),
                block_dim: (hd as u32, 4, 1),
                shared_mem_bytes: ((cs_u * ds + cs_u * hd + cs_u) * std::mem::size_of::<f32>())
                    as u32,
            };
            let mut sb = ctx
                .stream
                .launch_builder(m3k.m3_dqkv_state_terms_typed.get(dtype));
            let q_p = acts.q.cached_ptr();
            let dy_p = msc.d_y_typed.cached_ptr();
            sb.arg(sc.dstate_terms.inner_mut());
            sb.arg(&q_p);
            sb.arg(sc.da_cumsum.inner());
            sb.arg(&dy_p);
            sb.arg(&b);
            sb.arg(&t);
            sb.arg(&nh_i);
            sb.arg(&hd_i);
            sb.arg(&ds_i);
            sb.arg(&cs);
            unsafe { sb.launch(st_cfg) }
                .map_err(|e| format!("m3_dqkv_state_terms typed: {:?}", e))?;

            let nc_i = nc as i32;
            let mut pb = ctx.stream.launch_builder(&m3k.m3_dstate_passing_bwd);
            pb.arg(sc.dstate_enter.inner_mut());
            pb.arg(sc.dstate_terms.inner());
            pb.arg(sc.da_cs_sum.inner());
            pb.arg(&b);
            pb.arg(&nh_i);
            pb.arg(&hd_i);
            pb.arg(&ds_i);
            pb.arg(&nc_i);
            unsafe { pb.launch(grid_1d(dims.batch * nh * hd * ds)) }
                .map_err(|e| format!("m3_dstate_passing_bwd typed: {:?}", e))?;
        }
        let t_split = 16.min(1024 / hd.max(1)).max(1) as u32;
        let cfg = LaunchConfig {
            grid_dim: (nh as u32, dims.batch as u32, nc as u32),
            block_dim: (hd as u32, t_split, 1),
            shared_mem_bytes: smem as u32,
        };
        let mut builder = ctx.stream.launch_builder(m3k.m3_dqkv_typed.get(dtype));
        builder.arg(sc.d_q.inner_mut()); // dQ_mid
        builder.arg(sc.d_k.inner_mut()); // dK_mid
        builder.arg(sc.d_x.inner_mut()); // dV
        builder.arg(sc.d_alpha.inner_mut()); // dADT
        builder.arg(sc.d_beta.inner_mut()); // dQK_dot
        builder.arg(sc.axis0_partials.inner_mut()); // dD_partials [B*nh] (no-atomics partials)
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
        builder.arg(sc.dstate_enter.inner());
        builder.arg(&b);
        builder.arg(&t);
        builder.arg(&nh_i);
        builder.arg(&hd_i);
        builder.arg(&ds_i);
        builder.arg(&cs);
        builder.arg(&use_pair_mats);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_dqkv_typed B6: {:?}", e))?;
    }
    // Stage 2: reduce dD_partials[B*nc, nh] → lg.d_param[nh] (accumulate=1).
    {
        let rows = dims.batch * dims.n_chunks();
        let rows_i = rows as i32;
        let block_dim = (rows as u32).next_power_of_two().clamp(32, 256);
        let accumulate_i: i32 = 1;
        let d_dp_ptr = lg.d_param.ptr();
        let mut rb = ctx.stream.launch_builder(&m3k.reduce_sum_axis0);
        rb.arg(&d_dp_ptr);
        rb.arg(sc.axis0_partials.inner());
        rb.arg(&rows_i);
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

    // m3_dqktheta typed — dQ_bias/dK_bias removed from args
    // (caller does colsum_accumulate on dQ_pre/dK_pre scratch below).
    {
        let na_i = na as i32;
        // Six [CS][ds] staging tiles (4 inputs + 2 outputs) for the
        // coalesced I/O path; values identical, bits identical.
        // Large-d_state tiles overflow the 48 KB no-opt-in dynamic
        // limit — fall back to the direct-global path (staging = 0).
        let dqkt_smem = 6 * cs_u * ds * 4;
        let dqkt_staging: i32 = i32::from(dqkt_smem <= 48 * 1024);
        let cfg = LaunchConfig {
            grid_dim: ((dims.batch * nc) as u32, nh as u32, 1),
            block_dim: (cs_u as u32, 1, 1),
            shared_mem_bytes: if dqkt_staging == 1 {
                dqkt_smem as u32
            } else {
                0
            },
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
        builder.arg(&dqkt_staging);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_dqktheta_typed B6: {:?}", e))?;
    }
    // colsum dQ_pre / dK_pre → c_bias / b_bias (deterministic).
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

    // (the d_q/d_k identity round trip is gone — m3_ddt_dtrap
    // touches neither pair and every downstream consumer reads
    // d_b/c_pre_rope directly; the f32 twin dropped its copy pair in the
    // dead-work pass.)

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

    // ----------------------------------------------------------------
    // B5a: angle_dt_bwd — no-atomics partials rule (pure f32, no atomicAdd).
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
    gpu_gemm_bi_backward_dw_grad_typed(
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
    // no-atomics partials rule two-stage:
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
            // Shared-kernel ABI: rmsnorm_backward grew an `accumulate`
            // arg (M1 residual fold); the M3 sites keep the
            // plain-store behavior.
            let accumulate_dx: i32 = 0;
            builder.arg(&accumulate_dx);
            let no_mirror: cudarc::driver::sys::CUdeviceptr = 0;
            builder.arg(&no_mirror);
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
