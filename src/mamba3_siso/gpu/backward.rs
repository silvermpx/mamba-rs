//! Mamba-3 SISO GPU **backward** training pass (f32 master grads).
//!
//! Split from the former 2313-line `mamba3_gpu.rs`. Mirrors
//! the forward layout but in reverse order: B8 → B7 → B6 → B5 → B4 → B3
//! → B2 → B1 + residual add.
//!
//! For mixed-precision (bf16/f16) backward see [`super::backward_mixed`].

use super::state::{
    GpuMamba3BackboneActs, GpuMamba3LayerActs, GpuMamba3MlpActs, GpuMamba3Scratch, M3Exec,
};
use super::weights::{
    GpuMamba3Grads, GpuMamba3LayerGrads, GpuMamba3LayerWeights, GpuMamba3MlpGrads,
    GpuMamba3MlpWeights, GpuMamba3Weights,
};
use crate::gpu::bidi::{WindowRows, reverse_rows};
use crate::gpu::blas::gpu_gemm_bi_backward_grad_raw;
use crate::gpu::block_norm;
use crate::gpu::buffers::{GpuBuffer, GradSlice};
use crate::gpu::dtype::WeightDtype;
use crate::gpu::launch::{grid_1d_cover, grid_colsum, y_runs, yz_runs};
use cudarc::driver::PushKernelArg;

/// Mamba-3 SISO single-layer GPU backward (sequential SSM backward).
/// The operands of the entering fold's gradient.
pub(super) struct EnterFold<'a> {
    pub d_scale_enter: &'a mut GpuBuffer,
    pub seed_grad: &'a GpuBuffer,
    pub k_enter: &'a GpuBuffer,
    pub v_enter: &'a GpuBuffer,
}

/// The gradient of the trapezoidal fold the window's entering state took,
/// per sequence and head (`m3_enter_fold_bwd`): `m3_ddt_dtrap` adds it to
/// the first step's dt and trap gradients. Zero for a window that entered
/// with zero K and V.
pub(super) fn launch_enter_fold_bwd(exec: &M3Exec<'_>, op: EnterFold<'_>) -> Result<(), String> {
    let M3Exec {
        ctx,
        kernels: m3k,
        dims,
        ..
    } = *exec;
    let rows = u32::try_from(dims.batch * dims.nheads)
        .map_err(|_| "m3_enter_fold_bwd: batch * heads exceeds the grid".to_string())?;
    let (b, nh) = (dims.batch as i32, dims.nheads as i32);
    let (hd, ds) = (dims.headdim as i32, dims.d_state as i32);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = ctx
        .stream
        .launch_builder(m3k.m3_enter_fold_bwd.get(dims.index_width()));
    builder.arg(op.d_scale_enter.inner_mut());
    builder.arg(op.seed_grad.inner());
    builder.arg(op.k_enter.inner());
    builder.arg(op.v_enter.inner());
    builder.arg(&b);
    builder.arg(&nh);
    builder.arg(&hd);
    builder.arg(&ds);
    unsafe { builder.launch(cfg) }
        .map(|_| ())
        .map_err(|e| format!("m3_enter_fold_bwd: {e:?}"))
}

pub fn gpu_backward_mamba3_layer(
    exec: &M3Exec<'_>,
    d_temporal: &mut GpuBuffer, // [B*T*d_model] in/out gradient
    acts: &GpuMamba3LayerActs,
    lw: &GpuMamba3LayerWeights,
    lg: &GpuMamba3LayerGrads,
    scratch: &mut GpuMamba3Scratch,
) -> Result<(), String> {
    let into = d_temporal.cached_ptr();
    layer_backward(exec, (d_temporal, into), (acts, lw, lg), scratch)
}

/// [`gpu_backward_mamba3_layer`] with the norm's input gradient added into
/// the rows at `into` rather than `d_temporal`: the first layer of a
/// bidirectional pair reads d loss / d (the pair's output) and adds into
/// the gradient the second layer gathered.
fn layer_backward(
    exec: &M3Exec<'_>,
    (d_temporal, into): (&mut GpuBuffer, u64),
    (acts, lw, lg): (
        &GpuMamba3LayerActs,
        &GpuMamba3LayerWeights,
        &GpuMamba3LayerGrads,
    ),
    scratch: &mut GpuMamba3Scratch,
) -> Result<(), String> {
    exec.check_state_cap()?;
    let M3Exec {
        ctx,
        kernels: m3k,
        dims,
        seq_idx,
    } = *exec;
    let bt = dims.bt();
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let nh = dims.nheads;
    let hd = dims.headdim;
    let ng = dims.ngroups;
    let ip = dims.in_proj_dim;
    let na = dims.n_angles;
    let width = dims.index_width();
    let b = dims.batch as i32;
    let t = dims.seq_len as i32;

    // B8: out_proj backward: d_gated = d_temporal @ out_proj_w^T,
    //     d_out_proj_w += gated^T @ d_temporal.
    gpu_gemm_bi_backward_grad_raw(
        ctx,
        &mut scratch.d_gated,
        (&lg.out_proj_w, None),
        d_temporal,
        &acts.gated,
        lw.out_proj_w.raw_ptr(&ctx.stream),
        (bt, di, dm),
    )?;

    // B7: Output gating backward (matches F7 forward branch).
    if dims.is_outproj_norm {
        let nw_ptr = lw.norm_gate_weight.raw_ptr(&ctx.stream);
        let bt_i = bt as i32;
        let di_i = di as i32;
        let hd_i = dims.headdim as i32;
        let grid = super::kernels::gated_norm_cfg(bt, di, dims.headdim);
        for (y_first, cfg) in y_runs(grid) {
            let mut builder = ctx.stream.launch_builder(m3k.rmsnorm_gated_bwd.get(width));
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
            builder.arg(&y_first);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("rmsnorm_gated_bwd m3 B7: {:?}", e))?;
        }
        {
            let n_i = di as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
            let dst = lg.norm_gate_weight.ptr();
            builder.arg(&dst);
            builder.arg(scratch.d_norm_gate_w.inner());
            let bt_i = bt as i32;
            builder.arg(&bt_i);
            builder.arg(&n_i);
            unsafe { builder.launch(grid_colsum(di)) }
                .map_err(|e| format!("colsum d_norm_gate_w m3: {:?}", e))?;
        }
    } else {
        let n = crate::gpu::launch::WidthCount::new(bt * di, width)?;
        let mut builder = ctx.stream.launch_builder(m3k.silu_gate_bwd.get(width));
        builder.arg(scratch.d_y.inner_mut());
        builder.arg(scratch.d_z.inner_mut());
        builder.arg(scratch.d_gated.inner());
        builder.arg(acts.y.inner());
        builder.arg(acts.z.inner());
        n.push(&mut builder);
        unsafe { builder.launch(grid_1d_cover(bt * di)?) }
            .map_err(|e| format!("silu_gate_bwd m3 B7: {:?}", e))?;
    }

    // B6: SSM backward. The parallel branch re-zeros d_k/d_q itself
    // (fill_scalar over d_x/d_k/d_q) and never touches d_d_local — zero
    // only what each branch actually consumes.
    scratch.d_alpha.zero(&ctx.stream)?;
    scratch.d_beta.zero(&ctx.stream)?;
    scratch.d_gamma.zero(&ctx.stream)?;
    if !dims.use_parallel_scan {
        scratch.d_k.zero(&ctx.stream)?;
        scratch.d_q.zero(&ctx.stream)?;
        scratch.d_d_local.zero(&ctx.stream)?;
    }

    if !dims.use_parallel_scan {
        let dp_ptr = lw.d_param.raw_ptr(&ctx.stream);
        let b_i = dims.batch as i32;
        let t_i = dims.seq_len as i32;
        let nh_i = nh as i32;
        let hd_i = hd as i32;
        let ds_i = ds as i32;
        let (kernel, cfg) = m3k.sequential_backward(
            WeightDtype::F32,
            super::kernels::SequentialShape {
                batch: dims.batch,
                nheads: nh,
                headdim: hd,
                d_state: ds,
            },
            width,
        );
        for (y_first, cfg) in y_runs(cfg) {
            let mut builder = ctx.stream.launch_builder(kernel);
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
            builder.arg(scratch.d_y.inner());
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
            builder.arg(&y_first);
            builder.arg(&seq_idx);
            unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_backward_seq B6: {:?}", e))?;
        }

        if na > 0 {
            // ORDERING INVARIANT (important): `rope_bwd` writes
            // `d_angle_cumsum` with a plain store. It MUST precede
            // `m3_angle_dt_bwd_seq` (launched below), which READS the
            // buffer and emits Rule-B contribution tensors (post-refactor:
            // no atomicAdd anywhere in this path). If the order were
            // reversed, the angle backward would read stale values. Both
            // kernels target the same stream, so serialization is
            // automatic; only the launch order in this function matters.
            let n_i = bt as i32;
            let nh_i = nh as i32;
            let ds_i = ds as i32;
            let na_i = na as i32;
            let mut builder = ctx.stream.launch_builder(m3k.rope_bwd.get(width));
            builder.arg(scratch.d_b_pre_rope.inner_mut());
            builder.arg(scratch.d_c_pre_rope.inner_mut());
            builder.arg(scratch.d_angle_cumsum.inner_mut());
            builder.arg(scratch.d_k.inner());
            builder.arg(scratch.d_q.inner());
            builder.arg(acts.b_biased.inner());
            builder.arg(acts.c_biased.inner());
            builder.arg(acts.angle_cumsum.inner());
            builder.arg(&n_i);
            builder.arg(&nh_i);
            builder.arg(&ds_i);
            builder.arg(&na_i);
            unsafe { builder.launch(grid_1d_cover(bt * nh * ds)?) }
                .map_err(|e| format!("rope_bwd seq B6: {:?}", e))?;
        } else {
            scratch
                .d_b_pre_rope
                .copy_from_raw(&scratch.d_k, &ctx.stream)?;
            scratch
                .d_c_pre_rope
                .copy_from_raw(&scratch.d_q, &ctx.stream)?;
        }

        {
            let d_bb_ptr = lg.b_bias.ptr();
            let bt_i = bt as i32;
            let nhds_i = (nh * ds) as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
            builder.arg(&d_bb_ptr);
            builder.arg(scratch.d_b_pre_rope.inner());
            builder.arg(&bt_i);
            builder.arg(&nhds_i);
            unsafe { builder.launch(grid_colsum(nh * ds)) }
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
            unsafe { builder.launch(grid_colsum(nh * ds)) }
                .map_err(|e| format!("colsum d_c_bias seq: {:?}", e))?;
        }

        {
            let d_dp = lg.d_param.ptr();
            let b_i2 = dims.batch as i32;
            let nh_i = nh as i32;
            let hd_i = hd as i32;
            let mut builder = ctx.stream.launch_builder(m3k.m3_reduce_d_d.get(width));
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
        // Chunked parallel backward.
        let dp_ptr = lw.d_param.raw_ptr(&ctx.stream);
        let nh_i = nh as i32;
        let hd_i = hd as i32;
        let ds_i = ds as i32;
        let t_i = dims.seq_len as i32;
        let cs = dims.chunk_size() as i32;
        let nc = dims.n_chunks();
        let b_i = dims.batch as i32;
        let na = dims.n_angles;

        // The forward saved da_cumsum, k_scaled, scale, gamma, qk_dot and
        // the chunk states in the layer acts; the kernels read them from
        // there, and the scale and gamma gradients get their own scratch.

        {
            let block_x = nh.min(256) as u32;
            let grid_z = nh.div_ceil(block_x as usize) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dims.batch as u32, nc as u32, grid_z),
                block_dim: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            for (y_first, z_first, cfg) in yz_runs(cfg) {
                let mut builder = ctx
                    .stream
                    .launch_builder(m3k.m3_extract_da_cs_sum.get(width));
                builder.arg(scratch.da_cs_sum.inner_mut());
                builder.arg(acts.da_cumsum_saved.inner());
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&nh_i);
                builder.arg(&cs);
                builder.arg(&y_first);
                builder.arg(&z_first);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| format!("m3_extract_da_cs_sum B6: {:?}", e))?;
            }
        }

        // m3_dqkv assigns every element of d_q, d_k and d_x it owns, so the
        // three buffers need no zeroing first.
        // No-atomics partials rule: dD_partials[B*nh] via axis0_partials,
        // followed by reduce_sum_axis0 → lg.d_param[nh] (accumulate=1 across
        // layers since GpuMamba3Grads::zero runs once per step).
        let cs_u = dims.chunk_size();
        if crate::mamba3_siso::route::chunked_backward_tiled(hd, ds) {
            let io = super::dqkv_tiled::DqkvTiledIo {
                d_q: scratch.d_q.cached_ptr(),
                d_k: scratch.d_k.cached_ptr(),
                d_v: scratch.d_x.cached_ptr(),
                d_adt: scratch.d_alpha.cached_ptr(),
                d_qk_dot: scratch.d_beta.cached_ptr(),
                d_partials: scratch.axis0_partials.cached_ptr(),
                q_rot: acts.q.cached_ptr(),
                k_scaled: acts.k_scaled_saved.cached_ptr(),
                v: acts.x.cached_ptr(),
                da_cs: acts.da_cumsum_saved.cached_ptr(),
                da_cs_sum: scratch.da_cs_sum.cached_ptr(),
                qk_dot: acts.qk_dot_saved.cached_ptr(),
                chunk_states: acts.chunk_states_saved.cached_ptr(),
                d_o: scratch.d_y.cached_ptr(),
                d_param: dp_ptr,
                state_terms: scratch.dstate_terms.cached_ptr(),
                enter_state: scratch.dstate_enter.cached_ptr(),
                seed_grad: scratch.dstate_seed.cached_ptr(),
                dm_chain: scratch.dm_chain.cached_ptr(),
                seq_idx,
            };
            let shape = super::dqkv_tiled::DqkvShape {
                batch: dims.batch,
                seq_len: dims.seq_len,
                nheads: nh,
                headdim: hd,
                d_state: ds,
                chunk: cs_u,
            };
            super::dqkv_tiled::launch(&ctx.stream, m3k, WeightDtype::F32, shape, &io, width)?;
        } else {
            // Two chunk-by-state operand tiles and the V/dO tiles, each row
            // padded by one float so a warp reading consecutive rows spreads
            // over the banks; two per-step lanes, the da/qk lanes, and TWO
            // head-state tiles (true states + d_state staging for the
            // warp-parallel dADT). The kernel lays its tiles out the same way,
            // and the route bounds the same tile, so a shape it admits runs.
            let legacy_floats = crate::mamba3_siso::route::chunked_backward_smem_bytes(hd, ds) / 4;
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
            let nc = dims.n_chunks();
            // Reverse d_state decomposition: per-chunk B terms, then the
            // serial per-lane fold into each chunk's entering state.
            {
                let st_cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (nh as u32, dims.batch as u32, nc as u32),
                    block_dim: (hd as u32, 4, 1),
                    shared_mem_bytes: ((cs_u * ds + cs_u * hd + cs_u) * std::mem::size_of::<f32>())
                        as u32,
                };
                for (y_first, z_first, cfg) in yz_runs(st_cfg) {
                    let mut sb = ctx.stream.launch_builder(
                        m3k.m3_dqkv_state_terms_typed
                            .get(width)
                            .get(crate::gpu::dtype::WeightDtype::F32),
                    );
                    sb.arg(scratch.dstate_terms.inner_mut());
                    sb.arg(acts.q.inner());
                    sb.arg(acts.da_cumsum_saved.inner());
                    sb.arg(scratch.d_y.inner());
                    sb.arg(&b_i);
                    sb.arg(&t_i);
                    sb.arg(&nh_i);
                    sb.arg(&hd_i);
                    sb.arg(&ds_i);
                    sb.arg(&cs);
                    sb.arg(&y_first);
                    sb.arg(&z_first);
                    sb.arg(&seq_idx);
                    unsafe { sb.launch(cfg) }
                        .map_err(|e| format!("m3_dqkv_state_terms: {:?}", e))?;
                }

                let nc_i = nc as i32;
                let mut pb = ctx
                    .stream
                    .launch_builder(m3k.m3_dstate_passing_bwd.get(width));
                pb.arg(scratch.dstate_enter.inner_mut());
                pb.arg(scratch.dstate_terms.inner());
                pb.arg(scratch.da_cs_sum.inner());
                pb.arg(&b_i);
                pb.arg(&nh_i);
                pb.arg(&hd_i);
                pb.arg(&ds_i);
                pb.arg(&nc_i);
                pb.arg(scratch.dstate_seed.inner_mut());
                pb.arg(&t_i);
                pb.arg(&cs);
                pb.arg(&seq_idx);
                unsafe { pb.launch(grid_1d_cover(dims.batch * nh * hd * ds)?) }
                    .map_err(|e| format!("m3_dstate_passing_bwd: {:?}", e))?;
            }
            // t-split: blockDim.y lanes stride the per-timestep
            // loops. Inference launch geometry (never data-shaped); any T_SPLIT
            // yields identical bits since each output keeps one owning lane
            // with the same inner order. Chunks ride grid z.
            let t_split = 16.min(1024 / hd.max(1)).max(1) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (nh as u32, dims.batch as u32, nc as u32),
                block_dim: (hd as u32, t_split, 1),
                shared_mem_bytes: smem as u32,
            };
            crate::mamba3_siso::route::require_warp(hd, "m3_dqkv")?;
            let dqkv = m3k.dqkv_for_shape(
                WeightDtype::F32,
                super::transport::DqkvRoute {
                    state: ds,
                    head: hd,
                    chunk: cs_u,
                    time: dims.seq_len,
                    pairs: use_pair_mats == 1,
                },
                width,
            );
            for (y_first, z_first, cfg) in yz_runs(cfg) {
                let mut builder = ctx.stream.launch_builder(dqkv);
                builder.arg(scratch.d_q.inner_mut());
                builder.arg(scratch.d_k.inner_mut());
                builder.arg(scratch.d_x.inner_mut());
                builder.arg(scratch.d_alpha.inner_mut());
                builder.arg(scratch.d_beta.inner_mut());
                // dD_partials [B*nh] — reduced after kernel.
                builder.arg(scratch.axis0_partials.inner_mut());
                builder.arg(acts.q.inner());
                builder.arg(acts.k_scaled_saved.inner());
                builder.arg(acts.x.inner());
                builder.arg(acts.da_cumsum_saved.inner());
                builder.arg(scratch.da_cs_sum.inner());
                builder.arg(acts.qk_dot_saved.inner());
                builder.arg(acts.chunk_states_saved.inner());
                builder.arg(scratch.d_y.inner());
                builder.arg(&dp_ptr);
                builder.arg(scratch.dstate_enter.inner());
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&nh_i);
                builder.arg(&hd_i);
                builder.arg(&ds_i);
                builder.arg(&cs);
                builder.arg(&use_pair_mats);
                builder.arg(&y_first);
                builder.arg(&z_first);
                builder.arg(&seq_idx);
                unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_dqkv B6 S1: {:?}", e))?;
            }
        }
        // Stage 2: reduce dD_partials[B*nc, nh] → lg.d_param[nh]
        // (accumulate=1 across layers; GpuMamba3Grads::zero runs once per
        // training step).
        {
            let rows = dims.batch * dims.n_chunks();
            let rows_i = rows as i32;
            let block_dim = (rows as u32).next_power_of_two().clamp(32, 256);
            let accumulate_i: i32 = 1;
            let d_dp_ptr = lg.d_param.ptr();
            let mut rb = ctx.stream.launch_builder(&m3k.reduce_sum_axis0);
            rb.arg(&d_dp_ptr);
            rb.arg(scratch.axis0_partials.inner());
            rb.arg(&rows_i);
            rb.arg(&nh_i);
            rb.arg(&accumulate_i);
            let red_cfg = cudarc::driver::LaunchConfig {
                grid_dim: (nh as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
            };
            unsafe { rb.launch(red_cfg) }
                .map_err(|e| format!("m3_dqkv dD reduce B6 S1: {:?}", e))?;
        }

        if na > 0 {
            crate::gpu::launch::launch_fill(
                &ctx.stream,
                &m3k.fill_scalar,
                (scratch.d_angle_cumsum.cached_ptr(), 0.0),
                dims.batch * dims.seq_len * nh * na,
                format_args!("zero d_angle_cumsum"),
            )?;
        }

        // m3_dqktheta — dQ_bias/dK_bias removed from kernel args
        // (caller does colsum_accumulate on dQ_pre/dK_pre scratch below).
        {
            let na_i = na as i32;
            // Six [CS][ds] staging tiles (4 inputs + 2 outputs) for the
            // coalesced I/O path; values identical, bits identical.
            // Large-d_state tiles overflow the 48 KB no-opt-in dynamic
            // limit — fall back to the direct-global path (staging = 0).
            let dqkt_smem = 6 * dims.chunk_size() * ds * 4;
            let dqkt_staging: i32 = i32::from(dqkt_smem <= 48 * 1024);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: ((dims.batch * nc) as u32, nh as u32, 1),
                block_dim: (dims.chunk_size() as u32, 1, 1),
                shared_mem_bytes: if dqkt_staging == 1 {
                    dqkt_smem as u32
                } else {
                    0
                },
            };
            for (y_first, cfg) in y_runs(cfg) {
                let mut builder = ctx.stream.launch_builder(m3k.m3_dqktheta.get(width));
                builder.arg(scratch.d_c_pre_rope.inner_mut());
                builder.arg(scratch.d_b_pre_rope.inner_mut());
                builder.arg(scratch.d_angle_cumsum.inner_mut());
                let scale_in_ptr = acts.scale_saved.raw_ptr(&ctx.stream);
                let gamma_in_ptr = acts.gamma_saved.raw_ptr(&ctx.stream);
                builder.arg(scratch.d_scale.inner_mut());
                builder.arg(scratch.d_gamma_par.inner_mut());
                builder.arg(acts.c_biased.inner());
                builder.arg(acts.b_biased.inner());
                builder.arg(&scale_in_ptr);
                builder.arg(&gamma_in_ptr);
                builder.arg(acts.angle_cumsum.inner());
                builder.arg(scratch.d_q.inner());
                builder.arg(scratch.d_k.inner());
                builder.arg(scratch.d_beta.inner());
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&nh_i);
                builder.arg(&ds_i);
                builder.arg(&na_i);
                builder.arg(&cs);
                builder.arg(&dqkt_staging);
                builder.arg(&y_first);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| format!("m3_dqktheta B6 S2: {:?}", e))?;
            }
        }
        // colsum dQ_pre / dK_pre → c_bias / b_bias (deterministic).
        {
            let d_cb_ptr = lg.c_bias.ptr();
            let bt_i = bt as i32;
            let nhds_i = (nh * ds) as i32;
            let mut cb = ctx.stream.launch_builder(&m3k.colsum_accumulate);
            cb.arg(&d_cb_ptr);
            cb.arg(scratch.d_c_pre_rope.inner());
            cb.arg(&bt_i);
            cb.arg(&nhds_i);
            unsafe { cb.launch(grid_colsum(nh * ds)) }
                .map_err(|e| format!("colsum d_c_bias par: {:?}", e))?;
        }
        {
            let d_bb_ptr = lg.b_bias.ptr();
            let bt_i = bt as i32;
            let nhds_i = (nh * ds) as i32;
            let mut cb = ctx.stream.launch_builder(&m3k.colsum_accumulate);
            cb.arg(&d_bb_ptr);
            cb.arg(scratch.d_b_pre_rope.inner());
            cb.arg(&bt_i);
            cb.arg(&nhds_i);
            unsafe { cb.launch(grid_colsum(nh * ds)) }
                .map_err(|e| format!("colsum d_b_bias par: {:?}", e))?;
        }

        // (dead identity round trip removed: d_q/d_k used to shuttle the
        // pre_rope grads around m3_ddt_dtrap, whose argument list touches
        // neither pair — consumers read d_b/c_pre_rope directly.)
        launch_enter_fold_bwd(
            exec,
            EnterFold {
                d_scale_enter: &mut scratch.d_scale_enter,
                seed_grad: &scratch.dstate_seed,
                k_enter: &acts.k_enter,
                v_enter: &acts.v_enter,
            },
        )?;
        {
            let mut builder = ctx.stream.launch_builder(m3k.m3_ddt_dtrap.get(width));
            builder.arg(scratch.d_gamma.inner_mut());
            builder.arg(scratch.d_trap_raw.inner_mut());
            builder.arg(scratch.d_scale.inner());
            builder.arg(scratch.d_gamma_par.inner());
            builder.arg(acts.dt.inner());
            builder.arg(acts.trap.inner());
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(scratch.d_scale_enter.inner());
            builder.arg(&seq_idx);
            unsafe { builder.launch(grid_1d_cover(bt * nh)?) }
                .map_err(|e| format!("m3_ddt_dtrap B6 S3: {:?}", e))?;
        }
    }

    // B5a: angle_dt_bwd — no-atomics partials rule (no atomicAdd).
    // Stage 1: kernel writes contrib_angles[nh, B*T*na] + contrib_dt[na, B*T*nh]
    //          into split axis0_partials scratch.
    // Stage 2a: reduce_sum_axis0(d_angles_raw, contrib_angles, nh, B*T*na, 0).
    // Stage 2b: reduce_sum_axis0(d_dt_angle, contrib_dt, na, B*T*nh, 0).
    if na > 0 {
        let t_i = t;
        let nh_i = nh as i32;
        let na_i = na as i32;
        let b_i = b;
        let contrib_angles_elems = nh * bt * na;
        let contrib_dt_offset_bytes = (contrib_angles_elems * std::mem::size_of::<f32>()) as u64;
        let contrib_angles_ptr = scratch.axis0_partials.cached_ptr();
        let contrib_dt_ptr = scratch.axis0_partials.cached_ptr() + contrib_dt_offset_bytes;
        // Stage 1
        {
            let grid = m3k.angle_backward_cfg(dims.batch, dims.seq_len, nh, na);
            for (y_first, cfg) in y_runs(grid) {
                let mut builder = ctx
                    .stream
                    .launch_builder(m3k.m3_angle_dt_bwd_seq.get(width));
                builder.arg(&contrib_angles_ptr);
                builder.arg(&contrib_dt_ptr);
                builder.arg(scratch.d_angle_cumsum.inner());
                builder.arg(acts.angles_raw.inner());
                builder.arg(acts.dt.inner());
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&nh_i);
                builder.arg(&na_i);
                builder.arg(&y_first);
                builder.arg(&seq_idx);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| format!("m3_angle_dt_bwd_seq B5a stage1: {:?}", e))?;
            }
        }
        // Stage 2a: reduce nh → d_angles_raw[B*T*na]
        m3k.launch_axis0_sum(
            &ctx.stream,
            super::transport::Axis0Sum {
                out: scratch.d_angles_raw.cached_ptr(),
                partials: contrib_angles_ptr,
                rows: nh,
                columns: bt * na,
                accumulate: false,
            },
            width,
        )
        .map_err(|e| format!("angle_dt_bwd B5a reduce angles: {e}"))?;
        // Stage 2b: reduce na → d_dt_angle[B*T*nh]
        m3k.launch_axis0_sum(
            &ctx.stream,
            super::transport::Axis0Sum {
                out: scratch.d_dt_angle.cached_ptr(),
                partials: contrib_dt_ptr,
                rows: na,
                columns: bt * nh,
                accumulate: false,
            },
            width,
        )
        .map_err(|e| format!("angle_dt_bwd B5a reduce dt: {e}"))?;
    } else {
        crate::gpu::launch::launch_fill(
            &ctx.stream,
            &m3k.fill_scalar,
            (scratch.d_dt_angle.cached_ptr(), 0.0),
            bt * nh,
            format_args!("fill d_dt_angle no-angle B5a"),
        )?;
    }

    // B5b: m3_abg_bwd (sequential) or m3_final_grads (parallel).
    if !dims.use_parallel_scan {
        let n_total = crate::gpu::launch::WidthCount::new(bt * nh, width)?;
        let nh_i = nh as i32;
        let dtb_ptr = lw.dt_bias.raw_ptr(&ctx.stream);
        let mut builder = ctx.stream.launch_builder(m3k.m3_abg_bwd.get(width));
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
        n_total.push(&mut builder);
        builder.arg(&nh_i);
        let t_seq = dims.seq_len as i32;
        builder.arg(&t_seq);
        builder.arg(&seq_idx);
        unsafe { builder.launch(grid_1d_cover(bt * nh)?) }
            .map_err(|e| format!("m3_abg_bwd seq B5b: {:?}", e))?;
    } else {
        let n_total = crate::gpu::launch::WidthCount::new(bt * nh, width)?;
        let nh_i = nh as i32;
        let dtb_ptr = lw.dt_bias.raw_ptr(&ctx.stream);
        let mut builder = ctx.stream.launch_builder(m3k.m3_final_grads.get(width));
        builder.arg(scratch.d_dd_dt.inner_mut());
        builder.arg(scratch.d_dd_a.inner_mut());
        builder.arg(scratch.d_alpha.inner());
        builder.arg(scratch.d_gamma.inner());
        builder.arg(scratch.d_dt_angle.inner());
        builder.arg(acts.a_val.inner());
        builder.arg(acts.dt.inner());
        builder.arg(acts.dd_dt_raw.inner());
        builder.arg(acts.dd_a_raw.inner());
        builder.arg(&dtb_ptr);
        builder.arg(&dims.a_floor);
        n_total.push(&mut builder);
        builder.arg(&nh_i);
        unsafe { builder.launch(grid_1d_cover(bt * nh)?) }
            .map_err(|e| format!("m3_final_grads B5b: {:?}", e))?;
    }

    {
        let d_dtb_ptr = lg.dt_bias.ptr();
        let bt_i = bt as i32;
        let nh_i = nh as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        builder.arg(&d_dtb_ptr);
        builder.arg(scratch.d_dd_dt.inner());
        builder.arg(&bt_i);
        builder.arg(&nh_i);
        unsafe { builder.launch(grid_colsum(nh)) }
            .map_err(|e| format!("colsum d_dt_bias B5b: {:?}", e))?;
    }

    // B4b: bc_bias_add_bwd — reduce heads → groups.
    {
        let n_i = bt as i32;
        let nh_i = nh as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let mut builder = ctx.stream.launch_builder(m3k.bc_bias_add_bwd.get(width));
        builder.arg(scratch.d_b_normed.inner_mut());
        builder.arg(scratch.d_b_pre_rope.inner());
        builder.arg(&n_i);
        builder.arg(&nh_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d_cover(bt * ng * ds)?) }
            .map_err(|e| format!("bc_bias_add_bwd B B4b: {:?}", e))?;
    }
    {
        let n_i = bt as i32;
        let nh_i = nh as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let mut builder = ctx.stream.launch_builder(m3k.bc_bias_add_bwd.get(width));
        builder.arg(scratch.d_c_normed.inner_mut());
        builder.arg(scratch.d_c_pre_rope.inner());
        builder.arg(&n_i);
        builder.arg(&nh_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d_cover(bt * ng * ds)?) }
            .map_err(|e| format!("bc_bias_add_bwd C B4b: {:?}", e))?;
    }

    // B4c: BCNorm backward for B and C.
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
        let mut builder = ctx.stream.launch_builder(m3k.bcnorm_bwd.get(width));
        builder.arg(scratch.d_b_raw.inner_mut());
        builder.arg(scratch.d_b_norm_w.inner_mut());
        builder.arg(scratch.d_b_normed.inner());
        builder.arg(acts.b_raw.inner());
        builder.arg(acts.b_rms.inner());
        builder.arg(&bn_ptr);
        builder.arg(&n_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("bcnorm_bwd B B4c: {:?}", e))?;
    }
    {
        let d_bnw_ptr = lg.b_norm_weight.ptr();
        let rows = (bt * ng) as i32;
        let ds_i = ds as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        builder.arg(&d_bnw_ptr);
        builder.arg(scratch.d_b_norm_w.inner());
        builder.arg(&rows);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_colsum(ds)) }
            .map_err(|e| format!("colsum d_b_norm_w B4c: {:?}", e))?;
    }
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
        let mut builder = ctx.stream.launch_builder(m3k.bcnorm_bwd.get(width));
        builder.arg(scratch.d_c_raw.inner_mut());
        builder.arg(scratch.d_c_norm_w.inner_mut());
        builder.arg(scratch.d_c_normed.inner());
        builder.arg(acts.c_raw.inner());
        builder.arg(acts.c_rms.inner());
        builder.arg(&cn_ptr);
        builder.arg(&n_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("bcnorm_bwd C B4c: {:?}", e))?;
    }
    {
        let d_cnw_ptr = lg.c_norm_weight.ptr();
        let rows = (bt * ng) as i32;
        let ds_i = ds as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        builder.arg(&d_cnw_ptr);
        builder.arg(scratch.d_c_norm_w.inner());
        builder.arg(&rows);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_colsum(ds)) }
            .map_err(|e| format!("colsum d_c_norm_w B4c: {:?}", e))?;
    }

    // B3: m3_split_bwd — assemble d_proj from 8 grad components.
    {
        let n_i = bt as i32;
        let di_i = di as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let nh_i = nh as i32;
        let na_i = na as i32;
        let mut builder = ctx.stream.launch_builder(m3k.m3_split_bwd.get(width));
        builder.arg(scratch.d_proj.inner_mut());
        builder.arg(scratch.d_z.inner());
        builder.arg(scratch.d_x.inner());
        builder.arg(scratch.d_b_raw.inner());
        builder.arg(scratch.d_c_raw.inner());
        builder.arg(scratch.d_dd_dt.inner());
        builder.arg(scratch.d_dd_a.inner());
        builder.arg(scratch.d_trap_raw.inner());
        builder.arg(scratch.d_angles_raw.inner());
        builder.arg(&n_i);
        builder.arg(&di_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        builder.arg(&nh_i);
        builder.arg(&na_i);
        unsafe { builder.launch(grid_1d_cover(bt * ip)?) }
            .map_err(|e| format!("m3_split_bwd B3: {:?}", e))?;
    }

    // B2: in_proj backward.
    gpu_gemm_bi_backward_grad_raw(
        ctx,
        &mut scratch.d_norm,
        (&lg.in_proj_w, None),
        &scratch.d_proj,
        &acts.post_norm,
        lw.in_proj_w.raw_ptr(&ctx.stream),
        (bt, dm, ip),
    )?;

    // B1: the block norm's backward — no-atomics partials rule.
    // Stage 1 writes per-sample per-dim scale partials to axis0_partials
    // (a LayerNorm adds its bias gradient first); the input gradient is a
    // plain store. Stage 2: reduce_sum_axis0 → lg.norm_weight (accumulate=1).
    {
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let axis0_ptr = scratch.axis0_partials.cached_ptr();
        block_norm::backward(
            ctx,
            WeightDtype::F32,
            (
                scratch.d_pre_norm.cached_ptr(),
                axis0_ptr,
                scratch.d_norm.cached_ptr(),
                acts.residual.cached_ptr(),
                acts.rms_vals.cached_ptr(),
            ),
            (lw.norm(), lg.norm_bias.as_ref().map_or(0, GradSlice::ptr)),
            (bt, dm),
            (false, 0),
            "block norm bwd m3 B1",
        )?;
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
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dm as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("rmsnorm_bwd m3 B1 reduce: {:?}", e))?;
        }
    }

    // Residual pass-through: into += d_pre_norm.
    crate::gpu::launch::launch_elems(
        &ctx.stream,
        &m3k.vec_add_inplace,
        &[into, scratch.d_pre_norm.cached_ptr()],
        bt * dm,
        format_args!("vec_add d_temporal m3"),
    )?;

    Ok(())
}

/// Mamba-3 SISO full backbone backward.
///
/// **IMPORTANT**: weight gradients in `grads` are **accumulated** (`beta=1.0`
/// on the dW GEMMs). Caller MUST call [`GpuMamba3Grads::zero`] before each
/// training step.
pub fn gpu_backward_mamba3_backbone(
    exec: &M3Exec<'_>,
    d_temporal: &mut GpuBuffer,
    acts: &GpuMamba3BackboneActs,
    mamba_w: &GpuMamba3Weights,
    grads: &GpuMamba3Grads,
    scratch: &mut GpuMamba3Scratch,
) -> Result<(), String> {
    exec.check_state_cap()?;
    super::forward::check_mlp_buffers(mamba_w, acts.mlp.as_ref().map(|m| m.hidden))?;
    let M3Exec {
        ctx,
        kernels: m3k,
        dims,
        ..
    } = *exec;
    let bt = dims.bt();
    let dm = dims.d_model;

    // norm_f bwd — no-atomics partials rule. Without the final norm the
    // output's gradient already is the residual stream's.
    let final_norm = mamba_w.final_norm();
    if !final_norm.is_none() {
        block_norm::backward(
            ctx,
            WeightDtype::F32,
            (
                scratch.d_norm.cached_ptr(),
                scratch.axis0_partials.cached_ptr(),
                d_temporal.cached_ptr(),
                acts.norm_f_input.cached_ptr(),
                acts.norm_f_rms.cached_ptr(),
            ),
            (
                final_norm,
                grads.norm_f_bias.as_ref().map_or(0, GradSlice::ptr),
            ),
            (bt, dm),
            (false, 0),
            "norm_f bwd m3",
        )?;
        // Stage 2: reduce axis0_partials[bt, dm] → grads.norm_f_weight[dm] (accumulate=1).
        {
            let bt_i = bt as i32;
            let dm_i = dm as i32;
            let block_dim = (bt as u32).next_power_of_two().clamp(32, 256);
            let accumulate_i: i32 = 1;
            let d_nf_ptr = grads.norm_f_weight.ptr();
            let axis0_ptr = scratch.axis0_partials.cached_ptr();
            let mut builder = ctx.stream.launch_builder(&m3k.reduce_sum_axis0);
            builder.arg(&d_nf_ptr);
            builder.arg(&axis0_ptr);
            builder.arg(&bt_i);
            builder.arg(&dm_i);
            builder.arg(&accumulate_i);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dm as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("rmsnorm_bwd norm_f m3 reduce: {:?}", e))?;
        }
        d_temporal.copy_from_raw(&scratch.d_norm, &ctx.stream)?;
    }

    let mlp_backward = |l: usize, d: &mut GpuBuffer, scratch: &mut GpuMamba3Scratch| {
        // The MLP came after the mixer, so its backward comes first.
        match acts.mlp.as_ref() {
            Some(m) => {
                let layer = MlpBwd {
                    acts: &m.layers[l],
                    w: &mamba_w.mlp[l],
                    g: &grads.mlp[l],
                };
                gpu_backward_mamba3_mlp(exec, d, &layer, scratch)
            }
            None => Ok(()),
        }
    };
    let layer = |l: usize| (&acts.layers[l], &mamba_w.layers[l], &grads.layers[l]);
    if dims.bidirectional {
        // A pair from its MLPs down: both mixers take d loss / d (the
        // pair's output), the second, which ran on the reversed window,
        // reversed, its norm's gradient turned back; the first adds its own
        // on top: (d_out + c_reversed) + c_forward.
        let window = |ptr| WindowRows {
            ptr,
            elem_bytes: 4,
            batch: dims.batch,
            seq_len: dims.seq_len,
            width: dm,
        };
        for l in (0..dims.n_layers).step_by(2).rev() {
            mlp_backward(l + 1, d_temporal, scratch)?;
            mlp_backward(l, d_temporal, scratch)?;
            scratch.pair.copy_from_raw(d_temporal, &ctx.stream)?;
            let d = d_temporal.cached_ptr();
            reverse_rows(ctx, window(d), exec.seq_idx)?;
            layer_backward(exec, (d_temporal, d), layer(l + 1), scratch)?;
            reverse_rows(ctx, window(d), exec.seq_idx)?;
            // The first layer reads the held gradient as its own and adds
            // into the gathered one; the two trade places back after it.
            std::mem::swap(d_temporal, &mut scratch.pair);
            let gathered = scratch.pair.cached_ptr();
            let first = layer_backward(exec, (d_temporal, gathered), layer(l), scratch);
            std::mem::swap(d_temporal, &mut scratch.pair);
            first?;
        }
    } else {
        for l in (0..dims.n_layers).rev() {
            mlp_backward(l, d_temporal, scratch)?;
            let into = d_temporal.cached_ptr();
            layer_backward(exec, (d_temporal, into), layer(l), scratch)?;
        }
    }

    // input_proj bwd. Without a projection there is nothing to train.
    if !mamba_w.input_proj_w.is_empty() {
        gpu_gemm_bi_backward_grad_raw(
            ctx,
            &mut scratch.d_input_proj_dx,
            (&grads.input_proj_w, Some(&grads.input_proj_b)),
            d_temporal,
            &acts.input_proj_inputs,
            mamba_w.input_proj_w.raw_ptr(&ctx.stream),
            (bt, dims.mamba_input_dim, dm),
        )?;
    }

    Ok(())
}

/// One layer's gated MLP as the backward reads it.
struct MlpBwd<'a> {
    acts: &'a GpuMamba3MlpActs,
    w: &'a GpuMamba3MlpWeights,
    g: &'a GpuMamba3MlpGrads,
}

/// The backward of one layer's gated MLP: `d_temporal` holds the gradient
/// of the layer's output and leaves holding the gradient of the MLP's
/// input, the residual included. The down projection, the gate, the up
/// projection, then the norm, whose input gradient the norm kernel adds
/// into `d_temporal` itself.
fn gpu_backward_mamba3_mlp(
    exec: &M3Exec<'_>,
    d_temporal: &mut GpuBuffer,
    layer: &MlpBwd<'_>,
    scratch: &mut GpuMamba3Scratch,
) -> Result<(), String> {
    let M3Exec {
        ctx,
        kernels: m3k,
        dims,
        ..
    } = *exec;
    let MlpBwd { acts, w, g } = *layer;
    let (bt, dm, h) = (dims.bt(), dims.d_model, w.hidden());
    if scratch.mlp_d_act.len() != bt * h {
        return Err(format!(
            "the scratch has no buffers for the {h}-wide gated MLP (build it with `with_mlp`)"
        ));
    }
    let (bt_i, dm_i, h_i) = (bt as i32, dm as i32, h as i32);
    gpu_gemm_bi_backward_grad_raw(
        ctx,
        &mut scratch.mlp_d_act,
        (&g.fc2_w, None),
        d_temporal,
        &acts.act,
        w.fc2_w.raw_ptr(&ctx.stream),
        (bt, h, dm),
    )?;
    let mut gate = ctx.stream.launch_builder(&m3k.mlp_gate_bwd.f32);
    gate.arg(scratch.mlp_d_up.inner_mut());
    gate.arg(scratch.mlp_d_act.inner());
    gate.arg(acts.up.inner());
    gate.arg(&bt_i);
    gate.arg(&h_i);
    unsafe { gate.launch(grid_1d_cover(bt * h)?) }
        .map_err(|e| format!("MLP gate bwd m3: {e:?}"))?;
    gpu_gemm_bi_backward_grad_raw(
        ctx,
        &mut scratch.mlp_d_post,
        (&g.fc1_w, None),
        &scratch.mlp_d_up,
        &acts.post,
        w.fc1_w.raw_ptr(&ctx.stream),
        (bt, dm, 2 * h),
    )?;
    let partials = scratch.axis0_partials.cached_ptr();
    // Accumulate: the norm's input gradient adds into the residual's.
    block_norm::backward(
        ctx,
        WeightDtype::F32,
        (
            d_temporal.cached_ptr(),
            partials,
            scratch.mlp_d_post.cached_ptr(),
            acts.x_in.cached_ptr(),
            acts.rms.cached_ptr(),
        ),
        (w.norm(), g.norm_bias.as_ref().map_or(0, GradSlice::ptr)),
        (bt, dm),
        (true, 0),
        "MLP norm bwd m3",
    )?;
    let block = (bt as u32).next_power_of_two().clamp(32, 256);
    let d_nw = g.norm_weight.ptr();
    let reduce_acc: i32 = 1;
    let mut reduce = ctx.stream.launch_builder(&m3k.reduce_sum_axis0);
    reduce.arg(&d_nw);
    reduce.arg(&partials);
    reduce.arg(&bt_i);
    reduce.arg(&dm_i);
    reduce.arg(&reduce_acc);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (dm as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 4,
    };
    unsafe { reduce.launch(cfg) }.map_err(|e| format!("MLP norm reduce m3: {e:?}"))?;
    Ok(())
}
