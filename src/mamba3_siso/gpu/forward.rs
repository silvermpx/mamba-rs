//! Mamba-3 SISO GPU **forward** pass (training + target burn-in).
//!
//! Split from the former 2313-line `mamba3_gpu.rs` (task #381). 8-phase
//! pipeline:
//!   F1: RMSNorm
//!   F2: in_proj SGEMM
//!   F3: m3_split (8-way + fused activations)
//!   F4: BCNorm + bias + RoPE on B,C
//!   F5: angle_dt accumulation → trapezoidal (alpha/beta/gamma)
//!   F6: SSM recurrence (sequential or chunked parallel)
//!   F7: RMSNormGated(y, z) or y*SiLU(z)
//!   F8: out_proj SGEMM + residual
//!
//! For mixed-precision (bf16/f16) forward see [`super::forward_mixed`].

use super::kernels::Mamba3Kernels;
use super::state::{
    GpuMamba3BackboneActs, GpuMamba3Dims, GpuMamba3LayerActs, GpuMamba3Scratch,
    GpuMamba3TargetScratch, Mamba3LayerPtrs,
};
use super::weights::{GpuMamba3LayerWeights, GpuMamba3Weights};
use crate::mamba_ssm::gpu::blas::gpu_sgemm_forward_raw;
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::launch::{grid_1d, grid_norm};
use cudarc::driver::PushKernelArg;

/// Mamba-3 SISO single-layer GPU forward (8-phase pipeline).
#[allow(clippy::too_many_arguments)]
pub fn gpu_forward_mamba3_layer(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    temporal: &mut GpuBuffer,
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

    // F1: RMSNorm
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

    // F2: in_proj SGEMM
    gpu_sgemm_forward_raw(
        ctx,
        &mut scratch.proj_flat,
        &acts.post_norm,
        lw.in_proj_w.raw_ptr(&ctx.stream),
        None,
        (bt, dm, ip),
    )?;

    // F3: m3_split
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

    // F4a: BCNorm B
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
    // F4b: BCNorm C
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
    // F4c: Bias B
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
    // F4d: Bias C
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

    // F5: angle_dt sequential accumulation
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

    // F4e+f: RoPE on B->K and C->Q
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

    // F5b: alpha/beta/gamma
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

    // F6: SSM
    if dims.use_parallel_scan {
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

        acts.da_cumsum_saved
            .copy_from_raw(&scratch.da_cumsum, &ctx.stream)?;
        acts.k_scaled_saved
            .copy_from_raw(&scratch.d_q, &ctx.stream)?;
        acts.scale_saved
            .copy_from_raw(&scratch.d_gamma, &ctx.stream)?;
        acts.gamma_saved
            .copy_from_raw(&scratch.d_dd_dt, &ctx.stream)?;
        acts.qk_dot_saved
            .copy_from_raw(&scratch.d_beta, &ctx.stream)?;
        acts.chunk_states_saved
            .copy_from_raw(&scratch.chunk_states, &ctx.stream)?;

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
        let n = (bt * di) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.silu_gate_fwd);
        builder.arg(acts.gated.inner_mut());
        builder.arg(acts.y.inner());
        builder.arg(acts.z.inner());
        builder.arg(&n);
        unsafe { builder.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("silu_gate_fwd m3 F7: {:?}", e))?;
    }

    // F8: out_proj + residual
    gpu_sgemm_forward_raw(
        ctx,
        &mut scratch.out_flat,
        &acts.gated,
        lw.out_proj_w.raw_ptr(&ctx.stream),
        None,
        (bt, di, dm),
    )?;
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

/// Mamba-3 SISO full backbone forward (input proj + N layers + norm_f).
///
/// State semantics: with `dims.use_parallel_scan` the window is STATELESS —
/// the chunked kernels do not consume entering SSM/K/V state (chunk 0 always
/// starts from zero), so all four state buffers (including the RoPE angle
/// accumulator) are zeroed up front to keep the semantics consistent. Final
/// window states are still written back for inspection. State continuity
/// across calls is only supported by the sequential path
/// (`use_parallel_scan = false`, `m3_burnin_fwd`).
#[allow(clippy::too_many_arguments)]
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

    if dims.use_parallel_scan {
        // The chunked SSM kernels ignore entering SSM/K/V state, but the
        // angle kernel WOULD carry the persistent accumulator — a hybrid
        // that matches neither stateless-window nor full-continuity
        // semantics. Zero all four so every parallel window is cleanly
        // stateless. (Async memsets — CUDA Graph capture safe.)
        ssm_states.zero(&ctx.stream)?;
        k_states.zero(&ctx.stream)?;
        v_states.zero(&ctx.stream)?;
        angle_states.zero(&ctx.stream)?;
    }

    acts.input_proj_inputs
        .copy_from_raw(mamba_input, &ctx.stream)?;
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

/// Mamba-3 target burn-in forward (no activations saved, used for Bellman target).
///
/// Output: `temporal` receives `[B * d_model]` — the LAST timestep's representation.
/// Internally uses `tgt.temporal_work` as `[B*T*d_model]` working buffer.
#[allow(clippy::too_many_arguments)]
pub fn gpu_forward_mamba3_target_burnin(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    temporal: &mut GpuBuffer,
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

    tgt.ssm_states.zero(&ctx.stream)?;
    tgt.k_states.zero(&ctx.stream)?;
    tgt.v_states.zero(&ctx.stream)?;
    tgt.angle_states.zero(&ctx.stream)?;

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
        tgt.residual
            .copy_from_raw(&tgt.temporal_work, &ctx.stream)?;
        {
            let bt_i = bt as i32;
            let dm_i = dm as i32;
            let eps: f32 = 1e-5;
            let nw_ptr = lw.norm_weight.raw_ptr(&ctx.stream);
            let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_fwd);
            builder.arg(tgt.out_flat.inner_mut());
            builder.arg(tgt.rms_discard.inner_mut());
            builder.arg(tgt.residual.inner());
            builder.arg(&nw_ptr);
            builder.arg(&bt_i);
            builder.arg(&dm_i);
            builder.arg(&eps);
            unsafe { builder.launch(grid_norm(bt, dm)) }
                .map_err(|e| format!("rmsnorm_fwd m3 tgt L{l}: {:?}", e))?;
        }
        gpu_sgemm_forward_raw(
            ctx,
            &mut tgt.proj_flat,
            &tgt.out_flat,
            lw.in_proj_w.raw_ptr(&ctx.stream),
            None,
            (bt, dm, ip),
        )?;
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
        if dims.n_angles > 0 {
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
            {
                let n_i = bt as i32;
                let nh_i = nh as i32;
                let ds_i = ds as i32;
                let na_i = na as i32;
                let mut builder = ctx.stream.launch_builder(&m3k.rope_fwd);
                builder.arg(tgt.k.inner_mut());
                builder.arg(tgt.q.inner_mut());
                builder.arg(tgt.b_biased.inner());
                builder.arg(tgt.c_biased.inner());
                builder.arg(tgt.angle_cumsum.inner());
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
            builder.arg(&ssm_ptr);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(tgt.y.inner_mut());
            builder.arg(tgt.x.inner());
            builder.arg(tgt.k.inner());
            builder.arg(tgt.q.inner());
            builder.arg(tgt.alpha.inner());
            builder.arg(tgt.beta.inner());
            builder.arg(tgt.gamma.inner());
            builder.arg(&dp_ptr);
            builder.arg(&b);
            builder.arg(&t);
            builder.arg(&nh_i);
            builder.arg(&hd_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("m3_burnin_fwd_nosave tgt L{l}: {:?}", e))?;
        }
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
