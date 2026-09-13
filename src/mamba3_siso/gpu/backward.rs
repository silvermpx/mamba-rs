//! Mamba-3 SISO GPU **backward** training pass (f32 master grads).
//!
//! Split from the former 2313-line `mamba3_gpu.rs`. Mirrors
//! the forward layout but in reverse order: B8 → B7 → B6 → B5 → B4 → B3
//! → B2 → B1 + residual add.
//!
//! For mixed-precision (bf16/f16) backward see [`super::backward_mixed`].

use super::state::{GpuMamba3BackboneActs, GpuMamba3LayerActs, GpuMamba3Scratch, M3Exec};
use super::weights::{
    GpuMamba3Grads, GpuMamba3LayerGrads, GpuMamba3LayerWeights, GpuMamba3Weights,
};
use crate::mamba_ssm::gpu::blas::gpu_gemm_bi_backward_grad_raw;
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::launch::{grid_1d, grid_colsum, grid_norm};
use cudarc::driver::PushKernelArg;

/// Mamba-3 SISO single-layer GPU backward (sequential SSM backward).
pub fn gpu_backward_mamba3_layer(
    exec: &M3Exec<'_>,
    d_temporal: &mut GpuBuffer, // [B*T*d_model] in/out gradient
    acts: &GpuMamba3LayerActs,
    lw: &GpuMamba3LayerWeights,
    lg: &GpuMamba3LayerGrads,
    scratch: &mut GpuMamba3Scratch,
) -> Result<(), String> {
    let M3Exec {
        ctx,
        kernels: m3k,
        dims,
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
        let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_gated_bwd);
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
        unsafe { builder.launch(grid) }.map_err(|e| format!("rmsnorm_gated_bwd m3 B7: {:?}", e))?;
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
        let n = (bt * di) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.silu_gate_bwd);
        builder.arg(scratch.d_y.inner_mut());
        builder.arg(scratch.d_z.inner_mut());
        builder.arg(scratch.d_gated.inner());
        builder.arg(acts.y.inner());
        builder.arg(acts.z.inner());
        builder.arg(&n);
        unsafe { builder.launch(grid_1d(bt * di)) }
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
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (dims.batch as u32, nh as u32, 1),
            block_dim: (hd as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = ctx.stream.launch_builder(m3k.backward_seq_for_state(ds));
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
        unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_backward_seq B6: {:?}", e))?;

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
            let mut builder = ctx.stream.launch_builder(&m3k.rope_bwd);
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
            unsafe { builder.launch(grid_1d(bt * nh * ds)) }
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
            let mut builder = ctx.stream.launch_builder(&m3k.m3_reduce_d_d);
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
            let mut builder = ctx.stream.launch_builder(&m3k.m3_extract_da_cs_sum);
            builder.arg(scratch.da_cs_sum.inner_mut());
            builder.arg(acts.da_cumsum_saved.inner());
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(&cs);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("m3_extract_da_cs_sum B6: {:?}", e))?;
        }

        // m3_dqkv assigns every element of d_q, d_k and d_x it owns, so the
        // three buffers need no zeroing first.
        // No-atomics partials rule: dD_partials[B*nh] via axis0_partials,
        // followed by reduce_sum_axis0 → lg.d_param[nh] (accumulate=1 across
        // layers since GpuMamba3Grads::zero runs once per step).
        {
            let cs_u = dims.chunk_size();
            // Two chunk-by-state operand tiles and the V/dO tiles, each row
            // padded by one float so a warp reading consecutive rows spreads
            // over the banks; two per-step lanes, the da/qk lanes, and TWO
            // head-state tiles (true states + d_state staging for the
            // warp-parallel dADT). The kernel lays its tiles out the same way.
            let legacy_floats = 2 * cs_u * (ds + 1) + 2 * cs_u * (hd + 1) + 4 * cs_u + 2 * hd * ds;
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
                let mut sb = ctx.stream.launch_builder(
                    m3k.m3_dqkv_state_terms_typed
                        .get(crate::mamba_ssm::gpu::dtype::WeightDtype::F32),
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
                unsafe { sb.launch(st_cfg) }
                    .map_err(|e| format!("m3_dqkv_state_terms: {:?}", e))?;

                let nc_i = nc as i32;
                let mut pb = ctx.stream.launch_builder(&m3k.m3_dstate_passing_bwd);
                pb.arg(scratch.dstate_enter.inner_mut());
                pb.arg(scratch.dstate_terms.inner());
                pb.arg(scratch.da_cs_sum.inner());
                pb.arg(&b_i);
                pb.arg(&nh_i);
                pb.arg(&hd_i);
                pb.arg(&ds_i);
                pb.arg(&nc_i);
                unsafe { pb.launch(grid_1d(dims.batch * nh * hd * ds)) }
                    .map_err(|e| format!("m3_dstate_passing_bwd: {:?}", e))?;
            }
            // t-split: blockDim.y lanes stride the per-timestep
            // loops. Fixed launch geometry (never data-shaped); any T_SPLIT
            // yields identical bits since each output keeps one owning lane
            // with the same inner order. Chunks ride grid z.
            let t_split = 16.min(1024 / hd.max(1)).max(1) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (nh as u32, dims.batch as u32, nc as u32),
                block_dim: (hd as u32, t_split, 1),
                shared_mem_bytes: smem as u32,
            };
            let dqkv = m3k.dqkv_for_shape(
                WeightDtype::F32,
                ds,
                hd,
                cs_u,
                dims.seq_len,
                use_pair_mats == 1,
            );
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
            unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_dqkv B6 S1: {:?}", e))?;
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
            let n_ac = (dims.batch * dims.seq_len * nh * na) as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.fill_scalar);
            builder.arg(scratch.d_angle_cumsum.inner_mut());
            let zero: f32 = 0.0;
            builder.arg(&zero);
            builder.arg(&n_ac);
            unsafe { builder.launch(grid_1d(n_ac as usize)) }
                .map_err(|e| format!("zero d_angle_cumsum: {:?}", e))?;
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
            let mut builder = ctx.stream.launch_builder(&m3k.m3_dqktheta);
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
            unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_dqktheta B6 S2: {:?}", e))?;
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
        {
            let mut builder = ctx.stream.launch_builder(&m3k.m3_ddt_dtrap);
            builder.arg(scratch.d_gamma.inner_mut());
            builder.arg(scratch.d_trap_raw.inner_mut());
            builder.arg(scratch.d_scale.inner());
            builder.arg(scratch.d_gamma_par.inner());
            builder.arg(acts.dt.inner());
            builder.arg(acts.trap.inner());
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            unsafe { builder.launch(grid_1d(bt * nh)) }
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
        let btna = (bt * na) as i32;
        let btnh = (bt * nh) as i32;
        let contrib_angles_elems = nh * bt * na;
        let contrib_dt_offset_bytes = (contrib_angles_elems * std::mem::size_of::<f32>()) as u64;
        let contrib_angles_ptr = scratch.axis0_partials.cached_ptr();
        let contrib_dt_ptr = scratch.axis0_partials.cached_ptr() + contrib_dt_offset_bytes;
        // Stage 1
        {
            let mut builder = ctx.stream.launch_builder(&m3k.m3_angle_dt_bwd_seq);
            builder.arg(&contrib_angles_ptr);
            builder.arg(&contrib_dt_ptr);
            builder.arg(scratch.d_angle_cumsum.inner());
            builder.arg(acts.angles_raw.inner());
            builder.arg(acts.dt.inner());
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(&na_i);
            let grid = m3k.angle_backward_cfg(dims.batch, dims.seq_len, nh, na);
            unsafe { builder.launch(grid) }
                .map_err(|e| format!("m3_angle_dt_bwd_seq B5a stage1: {:?}", e))?;
        }
        // Stage 2a: reduce nh → d_angles_raw[B*T*na]
        {
            let accumulate_i: i32 = 0;
            let (reduction, cfg) = m3k.axis0_reduction(nh, bt * na);
            let mut builder = ctx.stream.launch_builder(reduction);
            builder.arg(scratch.d_angles_raw.inner_mut());
            builder.arg(&contrib_angles_ptr);
            builder.arg(&nh_i);
            builder.arg(&btna);
            builder.arg(&accumulate_i);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("angle_dt_bwd B5a reduce angles: {:?}", e))?;
        }
        // Stage 2b: reduce na → d_dt_angle[B*T*nh]
        {
            let accumulate_i: i32 = 0;
            let (reduction, cfg) = m3k.axis0_reduction(na, bt * nh);
            let mut builder = ctx.stream.launch_builder(reduction);
            builder.arg(scratch.d_dt_angle.inner_mut());
            builder.arg(&contrib_dt_ptr);
            builder.arg(&na_i);
            builder.arg(&btnh);
            builder.arg(&accumulate_i);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("angle_dt_bwd B5a reduce dt: {:?}", e))?;
        }
    } else {
        let ne = (bt * nh) as i32;
        let zero: f32 = 0.0;
        let mut builder = ctx.stream.launch_builder(&m3k.fill_scalar);
        builder.arg(scratch.d_dt_angle.inner_mut());
        builder.arg(&zero);
        builder.arg(&ne);
        unsafe { builder.launch(grid_1d(bt * nh)) }
            .map_err(|e| format!("fill d_dt_angle no-angle B5a: {:?}", e))?;
    }

    // B5b: m3_abg_bwd (sequential) or m3_final_grads (parallel).
    if !dims.use_parallel_scan {
        let n_total = (bt * nh) as i32;
        let nh_i = nh as i32;
        let dtb_ptr = lw.dt_bias.raw_ptr(&ctx.stream);
        let mut builder = ctx.stream.launch_builder(&m3k.m3_abg_bwd);
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
        builder.arg(&n_total);
        builder.arg(&nh_i);
        unsafe { builder.launch(grid_1d(bt * nh)) }
            .map_err(|e| format!("m3_abg_bwd seq B5b: {:?}", e))?;
    } else {
        let n_total = (bt * nh) as i32;
        let nh_i = nh as i32;
        let dtb_ptr = lw.dt_bias.raw_ptr(&ctx.stream);
        let mut builder = ctx.stream.launch_builder(&m3k.m3_final_grads);
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
        builder.arg(&n_total);
        builder.arg(&nh_i);
        unsafe { builder.launch(grid_1d(bt * nh)) }
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
        let mut builder = ctx.stream.launch_builder(&m3k.bc_bias_add_bwd);
        builder.arg(scratch.d_b_normed.inner_mut());
        builder.arg(scratch.d_b_pre_rope.inner());
        builder.arg(&n_i);
        builder.arg(&nh_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(bt * ng * ds)) }
            .map_err(|e| format!("bc_bias_add_bwd B B4b: {:?}", e))?;
    }
    {
        let n_i = bt as i32;
        let nh_i = nh as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.bc_bias_add_bwd);
        builder.arg(scratch.d_c_normed.inner_mut());
        builder.arg(scratch.d_c_pre_rope.inner());
        builder.arg(&n_i);
        builder.arg(&nh_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(grid_1d(bt * ng * ds)) }
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
        let mut builder = ctx.stream.launch_builder(&m3k.bcnorm_bwd);
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
        let mut builder = ctx.stream.launch_builder(&m3k.bcnorm_bwd);
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
        let mut builder = ctx.stream.launch_builder(&m3k.m3_split_bwd);
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
        unsafe { builder.launch(grid_1d(bt * ip)) }
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

    // B1: RMSNorm backward — no-atomics partials rule.
    // Stage 1: rmsnorm_bwd writes per-sample per-dim partials to axis0_partials.
    // Stage 2: reduce_sum_axis0 → lg.norm_weight (accumulate=1).
    {
        let nw_ptr = lw.norm_weight.raw_ptr(&ctx.stream);
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let axis0_ptr = scratch.axis0_partials.cached_ptr();
        {
            let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_bwd);
            builder.arg(scratch.d_pre_norm.inner_mut());
            builder.arg(&axis0_ptr); // d_scale_partials [bt*dm]
            builder.arg(scratch.d_norm.inner());
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
                .map_err(|e| format!("rmsnorm_bwd m3 B1 stage1: {:?}", e))?;
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
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dm as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("rmsnorm_bwd m3 B1 reduce: {:?}", e))?;
        }
    }

    // Residual pass-through: d_temporal += d_pre_norm.
    {
        let ne = (bt * dm) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.vec_add_inplace);
        builder.arg(d_temporal.inner_mut());
        builder.arg(scratch.d_pre_norm.inner());
        builder.arg(&ne);
        unsafe { builder.launch(grid_1d(bt * dm)) }
            .map_err(|e| format!("vec_add d_temporal m3: {:?}", e))?;
    }

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
    let M3Exec {
        ctx,
        kernels: m3k,
        dims,
    } = *exec;
    let bt = dims.bt();
    let dm = dims.d_model;

    // norm_f bwd — no-atomics partials rule.
    {
        let nf_ptr = mamba_w.norm_f_weight.raw_ptr(&ctx.stream);
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let axis0_ptr = scratch.axis0_partials.cached_ptr();
        let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_bwd);
        builder.arg(scratch.d_norm.inner_mut());
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
            .map_err(|e| format!("rmsnorm_bwd norm_f m3 stage1: {:?}", e))?;
    }
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

    for l in (0..dims.n_layers).rev() {
        gpu_backward_mamba3_layer(
            exec,
            d_temporal,
            &acts.layers[l],
            &mamba_w.layers[l],
            &grads.layers[l],
            scratch,
        )?;
    }

    // input_proj bwd.
    gpu_gemm_bi_backward_grad_raw(
        ctx,
        &mut scratch.d_input_proj_dx,
        (&grads.input_proj_w, Some(&grads.input_proj_b)),
        d_temporal,
        &acts.input_proj_inputs,
        mamba_w.input_proj_w.raw_ptr(&ctx.stream),
        (bt, dims.mamba_input_dim, dm),
    )?;

    Ok(())
}
