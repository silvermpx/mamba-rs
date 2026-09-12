//! Mixed-precision (bf16/f16) activation arena for Mamba-3 training.
//!
//! Mirrors the f32 [`super::mamba3_gpu::GpuMamba3LayerActs`] layout with
//! per-tensor dtype decisions matching PyTorch AMP / state-spaces/mamba
//! conventions:
//!
//! - **f32**: residual stream (residual), BPTT state (h_saved,
//!   k_prev_saved, v_prev_saved, chunk_states_saved), reduction saves
//!   (rms_vals, b_rms, c_rms, gated_rms_vals, angle_cumsum,
//!   da_cumsum_saved, scale_saved, gamma_saved, qk_dot_saved), small
//!   T-length coefficients (dt, a_val, trap), plus
//!   `norm_f_input` + `norm_f_rms`.
//! - **typed**: activation I/O at kernel boundaries — post_norm, z, x,
//!   b_raw, c_raw, b_normed, c_normed, b_biased, c_biased, k, q, y,
//!   gated, dd_dt_raw, dd_a_raw, trap_raw, angles_raw, k_scaled_saved,
//!   plus `input_proj_inputs` and `input_proj_outputs`.
//!
//! This scaffolding carries only the storage + allocation. The actual
//! forward wiring reads the compute weights from
//! [`super::weights_mixed_train::GpuMamba3TrainMixedWeights`] and writes
//! into these acts.

use std::sync::Arc;

use cudarc::driver::{CudaStream, PushKernelArg};

use crate::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use crate::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::launch::{grid_1d, grid_norm};
use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::gpu::state::{CHUNK_SIZE, GpuMamba3StateBufs, M3Exec, Mamba3LayerPtrs};
use crate::mamba3_siso::gpu::weights_mixed_train::GpuMamba3TrainMixedWeights;

/// Per-layer saved activations for M3 mixed-precision backward.
///
/// Field dtype decisions are documented inline; they match the precision
/// invariants (residual f32, BPTT f32, reduction saves f32,
/// activation I/O typed).
pub struct GpuMamba3LayerMixedActs {
    // F1: RMSNorm
    /// f32 residual input (pre-norm) — feeds layer output residual add too.
    pub residual: GpuBuffer,
    /// f32 RMS scalar per (b,t) pair.
    pub rms_vals: GpuBuffer,
    /// typed post-norm output → in_proj input.
    pub post_norm: DtypedBuf,

    // F3: Split (after in_proj)
    /// typed — z / x / b_raw / c_raw ARE typed per `m3_split_typed` kernel
    /// signature (activations branch of the 8-way split).
    pub z: DtypedBuf,
    pub x: DtypedBuf,
    pub b_raw: DtypedBuf,
    pub c_raw: DtypedBuf,
    /// f32 — raw (pre-nonlinearity) dt / A / trap scores and RoPE angles
    /// are written as **f32** by `m3_split_typed` (see
    /// `DEFINE_M3_SPLIT` in kernels/mamba3_ops.cu — only z/x/B/C are
    /// typed, the coefficients stay f32 for backward precision).
    pub dd_dt_raw: GpuBuffer,
    pub dd_a_raw: GpuBuffer,
    pub trap_raw: GpuBuffer,
    pub angles_raw: GpuBuffer,

    /// f32 — post-softplus dt (small T-length coefficient, precision-
    /// sensitive).
    pub dt: GpuBuffer,
    /// f32 — -softplus(dd_A), clamped.
    pub a_val: GpuBuffer,
    /// f32 — post-sigmoid trap.
    pub trap: GpuBuffer,

    // F4: BCNorm + bias + RoPE
    pub b_normed: DtypedBuf,
    pub c_normed: DtypedBuf,
    /// f32 — per-group RMS saves.
    pub b_rms: GpuBuffer,
    pub c_rms: GpuBuffer,
    pub b_biased: DtypedBuf,
    pub c_biased: DtypedBuf,
    /// typed — post-RoPE B (key).
    pub k: DtypedBuf,
    /// typed — post-RoPE C (query).
    pub q: DtypedBuf,

    // F5: Angle accumulation
    /// f32 — cumulative sum of angles (precision-sensitive).
    pub angle_cumsum: GpuBuffer,

    // F6: SSM
    /// f32 — hidden state BPTT save [B*(T+1)*d_inner*ds].
    pub h_saved: GpuBuffer,
    /// f32 — k_state saved per t.
    pub k_prev_saved: GpuBuffer,
    /// f32 — v_state saved per t.
    pub v_prev_saved: GpuBuffer,
    /// typed — SSM output before gating.
    pub y: DtypedBuf,

    // F6 parallel: saved intermediates (skip recompute in backward)
    pub da_cumsum_saved: GpuBuffer,
    /// typed — prescaled K (typed because it feeds a typed kernel in bwd).
    pub k_scaled_saved: DtypedBuf,
    pub scale_saved: GpuBuffer,
    pub gamma_saved: GpuBuffer,
    pub qk_dot_saved: GpuBuffer,
    /// f32 — chunk states (BPTT-class recurrent state).
    pub chunk_states_saved: GpuBuffer,

    // F7: RMSNormGated
    /// f32 — one rstd per head.
    pub gated_rms_vals: GpuBuffer,
    /// typed — gated output → out_proj input.
    pub gated: DtypedBuf,
}

/// Full M3 backbone mixed-precision activations.
pub struct GpuMamba3BackboneMixedActs {
    /// typed — input to input_proj.
    pub input_proj_inputs: DtypedBuf,
    /// typed — output of input_proj.
    pub input_proj_outputs: DtypedBuf,
    pub layers: Vec<GpuMamba3LayerMixedActs>,
    /// f32 — pre-norm_f input (residual stream).
    pub norm_f_input: GpuBuffer,
    /// f32 — RMS values per sample for norm_f backward.
    pub norm_f_rms: GpuBuffer,
    pub dtype: WeightDtype,
}

impl GpuMamba3BackboneMixedActs {
    /// Allocate all save buffers. Must be called once per trainer instance;
    /// re-used forward-pass to forward-pass (zero alloc on hot path, CUDA
    /// Graph capture safe).
    /// `use_parallel_scan` MUST match the flag the forward/backward run
    /// with (`GpuMamba3Dims::use_parallel_scan`): it decides whether the
    /// sequential-scan tapes get their full size or a 1-element sentinel,
    /// and a mismatch means the sequential kernels write out of bounds.
    pub fn new(
        stream: &Arc<CudaStream>,
        cfg: &Mamba3Config,
        batch: usize,
        seq_len: usize,
        input_dim: usize,
        dtype: WeightDtype,
        use_parallel_scan: bool,
    ) -> Result<Self, String> {
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let nh = cfg.nheads();
        let ng = cfg.ngroups;
        let na = cfg.num_rope_angles();
        let hd = cfg.headdim;
        let bt = batch * seq_len;
        // Match f32 GpuMamba3LayerActs: chunks sized by ceil(seq_len / chunk_size).
        let nc_max = seq_len.div_ceil(CHUNK_SIZE);
        // da_cumsum_saved is [B * nc * nh * chunk_size] per f32 layout.
        let da_cs_len = batch * nc_max * nh * CHUNK_SIZE;

        let layers = (0..cfg.n_layers)
            .map(|_| {
                Ok(GpuMamba3LayerMixedActs {
                    residual: GpuBuffer::zeros(stream, bt * dm)?,
                    rms_vals: GpuBuffer::zeros(stream, bt)?,
                    post_norm: DtypedBuf::zeros(stream, bt * dm, dtype)?,
                    z: DtypedBuf::zeros(stream, bt * di, dtype)?,
                    x: DtypedBuf::zeros(stream, bt * di, dtype)?,
                    b_raw: DtypedBuf::zeros(stream, bt * ng * ds, dtype)?,
                    c_raw: DtypedBuf::zeros(stream, bt * ng * ds, dtype)?,
                    dd_dt_raw: GpuBuffer::zeros(stream, bt * nh)?,
                    dd_a_raw: GpuBuffer::zeros(stream, bt * nh)?,
                    trap_raw: GpuBuffer::zeros(stream, bt * nh)?,
                    // `.max(1)` matches the f32 path — keeps a 1-elem sentinel
                    // allocation when RoPE is disabled (num_rope_angles == 0)
                    // so kernels that unconditionally index the buffer don't
                    // touch a zero-length allocation. Config validate()
                    // rejects `rope_fraction` other than 0.5 / 1.0 today,
                    // but the guard keeps this path symmetric with f32.
                    angles_raw: GpuBuffer::zeros(stream, bt * na.max(1))?,
                    dt: GpuBuffer::zeros(stream, bt * nh)?,
                    a_val: GpuBuffer::zeros(stream, bt * nh)?,
                    trap: GpuBuffer::zeros(stream, bt * nh)?,
                    b_normed: DtypedBuf::zeros(stream, bt * ng * ds, dtype)?,
                    c_normed: DtypedBuf::zeros(stream, bt * ng * ds, dtype)?,
                    b_rms: GpuBuffer::zeros(stream, bt * ng)?,
                    c_rms: GpuBuffer::zeros(stream, bt * ng)?,
                    b_biased: DtypedBuf::zeros(stream, bt * nh * ds, dtype)?,
                    c_biased: DtypedBuf::zeros(stream, bt * nh * ds, dtype)?,
                    k: DtypedBuf::zeros(stream, bt * nh * ds, dtype)?,
                    q: DtypedBuf::zeros(stream, bt * nh * ds, dtype)?,
                    // `.max(1)` — defense in depth, same rationale as
                    // `angles_raw` above and the f32 `angle_cumsum` path.
                    angle_cumsum: GpuBuffer::zeros(stream, bt * nh * na.max(1))?,
                    // Sequential-scan tapes, consumed only by the
                    // sequential forward branch (the mixed backward
                    // supports the chunked path only). On the chunked
                    // path they shrink to 1-element sentinels — the h
                    // tape alone is B*(T+1)*d_inner*d_state floats of
                    // dead VRAM otherwise.
                    h_saved: GpuBuffer::zeros(
                        stream,
                        if use_parallel_scan {
                            1
                        } else {
                            batch * (seq_len + 1) * di * ds
                        },
                    )?,
                    k_prev_saved: GpuBuffer::zeros(
                        stream,
                        if use_parallel_scan { 1 } else { bt * nh * ds },
                    )?,
                    v_prev_saved: GpuBuffer::zeros(
                        stream,
                        if use_parallel_scan { 1 } else { bt * nh * hd },
                    )?,
                    y: DtypedBuf::zeros(stream, bt * di, dtype)?,
                    da_cumsum_saved: GpuBuffer::zeros(stream, da_cs_len)?,
                    k_scaled_saved: DtypedBuf::zeros(stream, bt * nh * ds, dtype)?,
                    scale_saved: GpuBuffer::zeros(stream, bt * nh)?,
                    gamma_saved: GpuBuffer::zeros(stream, bt * nh)?,
                    qk_dot_saved: GpuBuffer::zeros(stream, bt * nh)?,
                    chunk_states_saved: GpuBuffer::zeros(stream, batch * nc_max * nh * hd * ds)?,
                    gated_rms_vals: GpuBuffer::zeros(stream, bt * nh)?,
                    gated: DtypedBuf::zeros(stream, bt * di, dtype)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let s = Self {
            input_proj_inputs: DtypedBuf::zeros(stream, bt * input_dim, dtype)?,
            input_proj_outputs: DtypedBuf::zeros(stream, bt * dm, dtype)?,
            layers,
            norm_f_input: GpuBuffer::zeros(stream, bt * dm)?,
            norm_f_rms: GpuBuffer::zeros(stream, bt)?,
            dtype,
        };

        // Race-fix invariant (a950648): callers may immediately upload from
        // host via default-stream HtoD; sync first.
        stream
            .synchronize()
            .map_err(|e| format!("sync after m3 mixed acts alloc: {e:?}"))?;

        Ok(s)
    }
}

// ════════════════════════════════════════════════════════════════════════
// Mixed-precision training forward — sequential AND chunked SSM.
// ════════════════════════════════════════════════════════════════════════

/// Per-layer forward in mixed precision. Mirrors
/// [`super::mamba3_gpu::gpu_forward_mamba3_layer`] step-by-step, dispatching
/// typed kernels for activation I/O while preserving f32 residual, BPTT
/// state, and small coefficient paths (see field-level comments on
/// `GpuMamba3LayerMixedActs`).
///
/// **Scope**: BOTH scan routes dispatch here (`dims.use_parallel_scan`
/// selects sequential vs chunked — the chunked typed kernels shipped after
/// this header was first written). The remaining asymmetry lives in the
/// BACKWARD: mixed sequential SSM backward is a documented non-goal (see
/// `backward_mixed.rs`), so mixed TRAINING must run the chunked route.
pub fn gpu_forward_mamba3_layer_mixed(
    exec: &M3Exec<'_>,
    stream_out: cudarc::driver::sys::CUdeviceptr,
    acts: &mut GpuMamba3LayerMixedActs,
    w: &crate::mamba3_siso::gpu::weights::GpuMamba3MixedLayerWeights,
    layer_ptrs: &Mamba3LayerPtrs,
    scratch: &mut GpuMamba3MixedScratch,
    dtype: WeightDtype,
) -> Result<(), String> {
    let M3Exec {
        ctx,
        kernels: m3k,
        dims,
    } = *exec;
    let Mamba3LayerPtrs {
        ssm_state,
        k_state,
        v_state,
        angle_state,
    } = *layer_ptrs;
    let GpuMamba3MixedScratch {
        proj_flat: proj_flat_scratch,
        out_flat: out_flat_scratch,
        alpha: alpha_scratch,
        beta: beta_scratch,
        gamma: gamma_scratch,
        adt_temp: adt_temp_scratch,
        final_states: final_states_scratch,
        ..
    } = scratch;
    let bt = dims.bt();
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let nh = dims.nheads;
    let hd = dims.headdim;
    let ng = dims.ngroups;
    let ip = dims.in_proj_dim;
    let na = dims.n_angles;

    // F1: rmsnorm_fwd_f32in_typed - f32 residual -> typed post_norm, f32
    // rms_vals. The residual stream already lives in acts.residual: the
    // previous layer's residual_add wrote it there directly (layer 0 is
    // seeded by the driver) - the per-layer temporal round trip is gone.
    {
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let eps: f32 = dims.rms_norm_eps;
        let mut bld = ctx
            .stream
            .launch_builder(m3k.rmsnorm_fwd_f32in_typed.get(dtype));
        let pn = acts.post_norm.cached_ptr();
        let rms = acts.rms_vals.cached_ptr();
        let x = acts.residual.cached_ptr();
        let nw = w.norm_weight.ptr();
        bld.arg(&pn);
        bld.arg(&rms);
        bld.arg(&x);
        bld.arg(&nw);
        bld.arg(&bt_i);
        bld.arg(&dm_i);
        bld.arg(&eps);
        unsafe { bld.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("m3_mixed F1 rmsnorm: {e:?}"))?;
    }

    // F2: in_proj GEMM typed [B*T, dm] -> [B*T, in_proj_dim].
    gpu_gemm_typed_forward_raw(
        ctx,
        TypedPtr {
            ptr: proj_flat_scratch.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: acts.post_norm.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: w.in_proj_w.ptr(),
            dtype,
        },
        None,
        (bt, dm, ip),
    )?;

    // F3: m3_split_typed — 8-way split + fused softplus/sigmoid.
    // z/x/b_raw/c_raw typed; dt/a_val/trap/angles_raw/dd_dt_raw/dd_a_raw/
    // trap_raw all f32 (per kernel signature).
    {
        let n_i = bt as i32;
        let di_i = di as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let nh_i = nh as i32;
        let na_i = na as i32;
        let db_ptr = w.dt_bias.ptr();
        let mut bld = ctx.stream.launch_builder(m3k.m3_split_typed.get(dtype));
        let z = acts.z.cached_ptr();
        let x = acts.x.cached_ptr();
        let br = acts.b_raw.cached_ptr();
        let cr = acts.c_raw.cached_ptr();
        let dt = acts.dt.cached_ptr();
        let av = acts.a_val.cached_ptr();
        let tr = acts.trap.cached_ptr();
        let ar = acts.angles_raw.cached_ptr();
        let dd_dt = acts.dd_dt_raw.cached_ptr();
        let dd_a = acts.dd_a_raw.cached_ptr();
        let tr_r = acts.trap_raw.cached_ptr();
        let proj = proj_flat_scratch.cached_ptr();
        bld.arg(&z);
        bld.arg(&x);
        bld.arg(&br);
        bld.arg(&cr);
        bld.arg(&dt);
        bld.arg(&av);
        bld.arg(&tr);
        bld.arg(&ar);
        bld.arg(&dd_dt);
        bld.arg(&dd_a);
        bld.arg(&tr_r);
        bld.arg(&proj);
        bld.arg(&db_ptr);
        bld.arg(&dims.a_floor);
        bld.arg(&n_i);
        bld.arg(&di_i);
        bld.arg(&ng_i);
        bld.arg(&ds_i);
        bld.arg(&nh_i);
        bld.arg(&na_i);
        unsafe { bld.launch(grid_1d(bt * ip)) }.map_err(|e| format!("m3_mixed F3 split: {e:?}"))?;
    }

    // F4a/b: bcnorm_fwd_bc_typed (fused B+C) — typed B_raw/C_raw →
    // typed B_normed/C_normed + f32 B_rms/C_rms.
    {
        let n_i = bt as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((bt * ng) as u32, 2, 1),
            block_dim: (ds as u32, 1, 1),
            shared_mem_bytes: ds as u32 * 4,
        };
        // Kernel signature (grouped by type, NOT interleaved B/C):
        //   B_normed, C_normed, B_rms, C_rms, B_raw, C_raw, B_weight, C_weight
        let mut bld = ctx
            .stream
            .launch_builder(m3k.bcnorm_fwd_bc_typed.get(dtype));
        let bn = acts.b_normed.cached_ptr();
        let cn = acts.c_normed.cached_ptr();
        let brms = acts.b_rms.cached_ptr();
        let crms = acts.c_rms.cached_ptr();
        let br = acts.b_raw.cached_ptr();
        let cr = acts.c_raw.cached_ptr();
        let bnw = w.b_norm_weight.ptr();
        let cnw = w.c_norm_weight.ptr();
        bld.arg(&bn);
        bld.arg(&cn);
        bld.arg(&brms);
        bld.arg(&crms);
        bld.arg(&br);
        bld.arg(&cr);
        bld.arg(&bnw);
        bld.arg(&cnw);
        bld.arg(&n_i);
        bld.arg(&ng_i);
        bld.arg(&ds_i);
        let eps_g5: f32 = dims.rms_norm_eps;
        bld.arg(&eps_g5);
        let src_stride = (ng * ds) as i32;
        bld.arg(&src_stride);
        unsafe { bld.launch(cfg) }.map_err(|e| format!("m3_mixed F4ab bcnorm_bc: {e:?}"))?;
    }

    // F5: angle accumulation (chunk-parallel, f32 args; see
    // gpu_angle_chunked_fwd).
    if na > 0 {
        // SAFETY: layer state and activations are disjoint allocations on
        // this context and remain live for the enclosing forward pass.
        unsafe {
            crate::mamba3_siso::gpu::forward::gpu_angle_chunked_fwd(
                ctx,
                m3k,
                crate::mamba3_siso::gpu::forward::AngleChunkedFwd {
                    angle_cumsum: &mut acts.angle_cumsum,
                    angle_state_ptr: angle_state,
                    angles_raw: &acts.angles_raw,
                    dt: &acts.dt,
                    sums: &scratch.angle_chunk_sums,
                    carries: &scratch.angle_chunk_carries,
                    shape: crate::mamba3_siso::gpu::forward::AngleChunkedShape {
                        batch: bt / dims.seq_len,
                        seq_len: dims.seq_len,
                        heads: nh,
                        angles: na,
                    },
                },
            )
        }?;
    }

    // F4c-f fused: typed bias add (B + C) + RoPE in one launch (the
    // typed stores keep the exact round-trip contract of the replaced
    // pair: biased rounds through FROM_F, the rotation consumes the
    // round-tripped values). With n_angles == 0 the kernel passes
    // through, replacing the old typed copy branch too.
    {
        let n_i = bt as i32;
        let nh_i = nh as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let na_i = na as i32;
        let mut bld = ctx
            .stream
            .launch_builder(m3k.m3_bias_rope_fwd_typed.get(dtype));
        let bb = acts.b_biased.cached_ptr();
        let cb = acts.c_biased.cached_ptr();
        let k = acts.k.cached_ptr();
        let q = acts.q.cached_ptr();
        let bn = acts.b_normed.cached_ptr();
        let cn = acts.c_normed.cached_ptr();
        let bb_p = w.b_bias.ptr();
        let cb_p = w.c_bias.ptr();
        let ac = acts.angle_cumsum.cached_ptr();
        bld.arg(&bb);
        bld.arg(&cb);
        bld.arg(&k);
        bld.arg(&q);
        bld.arg(&bn);
        bld.arg(&cn);
        bld.arg(&bb_p);
        bld.arg(&cb_p);
        bld.arg(&ac);
        bld.arg(&n_i);
        bld.arg(&nh_i);
        bld.arg(&ng_i);
        bld.arg(&ds_i);
        bld.arg(&na_i);
        // No angle state on this lane: the saved angles are read.
        let no_angle: cudarc::driver::sys::CUdeviceptr = 0;
        bld.arg(&no_angle);
        bld.arg(&no_angle);
        bld.arg(&no_angle);
        unsafe { bld.launch(grid_1d(bt * nh * ds)) }
            .map_err(|e| format!("m3_mixed F4 bias_rope: {e:?}"))?;
    }

    // F5b: m3_compute_abg, all f32. Only the sequential burn-in forward
    // below reads alpha/beta/gamma; the chunked pipeline derives its own
    // coefficients and the mixed backward never runs the sequential
    // branch, so nothing is saved for it.
    if !dims.use_parallel_scan {
        let n_total = (bt * nh) as i32;
        let mut bld = ctx.stream.launch_builder(&m3k.m3_compute_abg);
        bld.arg(alpha_scratch.inner_mut());
        bld.arg(beta_scratch.inner_mut());
        bld.arg(gamma_scratch.inner_mut());
        bld.arg(acts.dt.inner());
        bld.arg(acts.a_val.inner());
        bld.arg(acts.trap.inner());
        bld.arg(&n_total);
        unsafe { bld.launch(grid_1d(bt * nh)) }.map_err(|e| format!("m3_mixed F5b abg: {e:?}"))?;
    }

    // F6: SSM forward — sequential burnin OR chunked parallel scan.
    if dims.use_parallel_scan {
        // Chunked parallel scan path. Mirrors gpu_forward_mamba3_layer's
        // parallel branch but with typed I/O at the kernel boundaries.
        // 6-kernel pipeline writes DIRECTLY into acts saved buffers
        // (k_scaled_saved, scale/gamma/qk_dot_saved, chunk_states_saved,
        // da_cumsum_saved) — no separate scratch→acts copy step needed.
        let dp_ptr = w.d_param.ptr();
        let b_i = dims.batch as i32;
        let t_i = dims.seq_len as i32;
        let nh_i = nh as i32;
        let hd_i = hd as i32;
        let ds_i = ds as i32;
        let cs = dims.chunk_size() as i32;
        let nc = dims.n_chunks();

        // Kernel 0: adt = a_val · dt → adt_temp_scratch.
        {
            let (kern, count) = super::forward::adt_multiply_launch(
                m3k,
                bt * nh,
                &[
                    adt_temp_scratch.cached_ptr(),
                    acts.a_val.cached_ptr(),
                    acts.dt.cached_ptr(),
                ],
            );
            let n_total = count as i32;
            let mut bld = ctx.stream.launch_builder(kern);
            bld.arg(adt_temp_scratch.inner_mut());
            bld.arg(acts.a_val.inner());
            bld.arg(acts.dt.inner());
            bld.arg(&n_total);
            unsafe { bld.launch(grid_1d(count)) }.map_err(|e| format!("m3_mixed F6 adt: {e:?}"))?;
        }

        // K2: m3_dA_cumsum (pure f32) — adt → da_cumsum_saved.
        {
            let block_x = nh.min(256) as u32;
            let grid_z = nh.div_ceil(block_x as usize) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dims.batch as u32, nc as u32, grid_z),
                block_dim: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut bld = ctx.stream.launch_builder(&m3k.m3_da_cumsum);
            bld.arg(acts.da_cumsum_saved.inner_mut());
            bld.arg(adt_temp_scratch.inner());
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&nh_i);
            bld.arg(&cs);
            unsafe { bld.launch(cfg) }.map_err(|e| format!("m3_mixed F6 K2 da_cumsum: {e:?}"))?;
        }

        // K1+K3 fused (preprocess + chunk_state) when the shape allows:
        // K_scaled crosses the seam in shared memory instead of L2. K2
        // (dA_cumsum) moved ahead - it depends only on adt.
        if let Some(fcfg) =
            super::kernels::chunk_fused_cfg(dims.batch, nc, nh, hd, ds, dims.chunk_size())
        {
            let mut bld = ctx
                .stream
                .launch_builder(m3k.m3_chunk_pre_state_fused_typed.get(dtype));
            let ks = acts.k_scaled_saved.cached_ptr();
            let kp = acts.k.cached_ptr();
            let qp = acts.q.cached_ptr();
            let xp = acts.x.cached_ptr();
            bld.arg(&ks);
            bld.arg(acts.qk_dot_saved.inner_mut());
            bld.arg(acts.scale_saved.inner_mut());
            bld.arg(acts.gamma_saved.inner_mut());
            bld.arg(acts.chunk_states_saved.inner_mut());
            bld.arg(&kp);
            bld.arg(&qp);
            bld.arg(acts.dt.inner());
            bld.arg(acts.trap.inner());
            bld.arg(&xp);
            bld.arg(acts.da_cumsum_saved.inner());
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&nh_i);
            bld.arg(&hd_i);
            bld.arg(&ds_i);
            bld.arg(&cs);
            unsafe { bld.launch(fcfg) }
                .map_err(|e| format!("m3_mixed F6 fused pre+state: {e:?}"))?;
        } else {
            // K1: m3_preprocess_chunks_typed — typed K/Q + f32 DT/trap →
            // typed K_scaled (acts.k_scaled_saved) + f32 qk_dot/scale/gamma
            // (acts.qk_dot_saved, scale_saved, gamma_saved).
            {
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((dims.batch * nc) as u32, nh as u32, 1),
                    block_dim: (cs as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut bld = ctx
                    .stream
                    .launch_builder(m3k.m3_preprocess_chunks_typed.get(dtype));
                let ks = acts.k_scaled_saved.cached_ptr();
                let kp = acts.k.cached_ptr();
                let qp = acts.q.cached_ptr();
                bld.arg(&ks);
                bld.arg(acts.qk_dot_saved.inner_mut());
                bld.arg(acts.scale_saved.inner_mut());
                bld.arg(acts.gamma_saved.inner_mut());
                bld.arg(&kp);
                bld.arg(&qp);
                bld.arg(acts.dt.inner());
                bld.arg(acts.trap.inner());
                bld.arg(&b_i);
                bld.arg(&t_i);
                bld.arg(&nh_i);
                bld.arg(&ds_i);
                bld.arg(&cs);
                unsafe { bld.launch(cfg) }
                    .map_err(|e| format!("m3_mixed F6 K1 preprocess: {e:?}"))?;
            }

            // K3: m3_chunk_state_fwd_typed — typed x + typed K_scaled →
            // f32 chunk_states (acts.chunk_states_saved, mutated in place
            // by K4 into the entering states).
            {
                let cfg =
                    super::kernels::chunk_state_cfg(dims.batch, nc, nh, hd, ds, dims.chunk_size());
                let mut bld = ctx
                    .stream
                    .launch_builder(m3k.m3_chunk_state_fwd_typed.get(dtype));
                let xp = acts.x.cached_ptr();
                let ks = acts.k_scaled_saved.cached_ptr();
                bld.arg(acts.chunk_states_saved.inner_mut());
                bld.arg(&xp);
                bld.arg(&ks);
                bld.arg(acts.da_cumsum_saved.inner());
                bld.arg(&b_i);
                bld.arg(&t_i);
                bld.arg(&nh_i);
                bld.arg(&hd_i);
                bld.arg(&ds_i);
                bld.arg(&cs);
                unsafe { bld.launch(cfg) }
                    .map_err(|e| format!("m3_mixed F6 K3 chunk_state_fwd: {e:?}"))?;
            }
        }
        // K4: m3_state_passing_fwd (pure f32) — in-place mutate
        // chunk_states + write final_states.
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
            let mut bld = ctx.stream.launch_builder(&m3k.m3_state_passing_fwd);
            bld.arg(acts.chunk_states_saved.inner_mut());
            bld.arg(final_states_scratch.inner_mut());
            bld.arg(acts.da_cumsum_saved.inner());
            // Training forward: stateless window — chunk 0 enters at zero.
            let init_states_null: crate::mamba3_siso::gpu::state::CUptr = 0;
            bld.arg(&init_states_null);
            bld.arg(&b_i);
            bld.arg(&nc_i);
            bld.arg(&nh_i);
            bld.arg(&hd_i);
            bld.arg(&ds_i);
            bld.arg(&cs);
            bld.arg(&t_i);
            unsafe { bld.launch(cfg) }
                .map_err(|e| format!("m3_mixed F6 K4 state_passing: {e:?}"))?;
        }

        // K5 reads the entering states K4 left in acts.chunk_states_saved,
        // which is also what the backward needs.

        // K5: m3_chunk_scan_fwd_typed — typed y_out, x, q, K_scaled +
        // f32 qk_dot/da_cumsum/prev_states/D.
        {
            let (coop, cfg) =
                super::kernels::chunk_scan_cfg(dims.batch, nc, nh, hd, ds, dims.chunk_size());
            let kern = if coop {
                m3k.m3_chunk_scan_fwd_coop_typed.get(dtype)
            } else {
                m3k.m3_chunk_scan_fwd_typed.get(dtype)
            };
            let mut bld = ctx.stream.launch_builder(kern);
            let yp = acts.y.cached_ptr();
            let xp = acts.x.cached_ptr();
            let qp = acts.q.cached_ptr();
            let ks = acts.k_scaled_saved.cached_ptr();
            bld.arg(&yp);
            bld.arg(&xp);
            bld.arg(&qp);
            bld.arg(&ks);
            bld.arg(acts.qk_dot_saved.inner());
            bld.arg(acts.da_cumsum_saved.inner());
            bld.arg(acts.chunk_states_saved.inner());
            bld.arg(&dp_ptr);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&nh_i);
            bld.arg(&hd_i);
            bld.arg(&ds_i);
            bld.arg(&cs);
            unsafe { bld.launch(cfg) }
                .map_err(|e| format!("m3_mixed F6 K5 chunk_scan_fwd: {e:?}"))?;
        }

        // K6: m3_writeback_parallel_states_typed — final_states +
        // typed k/x → persistent f32 ssm_state/k_state/v_state.
        {
            let block_x = hd.max(ds) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dims.batch as u32, nh as u32, 1),
                block_dim: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut bld = ctx
                .stream
                .launch_builder(m3k.m3_writeback_parallel_states_typed.get(dtype));
            let kp = acts.k.cached_ptr();
            let xp = acts.x.cached_ptr();
            bld.arg(&ssm_state);
            bld.arg(&k_state);
            bld.arg(&v_state);
            bld.arg(final_states_scratch.inner());
            bld.arg(&kp);
            bld.arg(&xp);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&nh_i);
            bld.arg(&hd_i);
            bld.arg(&ds_i);
            unsafe { bld.launch(cfg) }.map_err(|e| format!("m3_mixed F6 K6 writeback: {e:?}"))?;
        }
    } else {
        // Sequential SSM burnin typed.
        let dp_ptr = w.d_param.ptr();
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
        let kernel = match dtype {
            WeightDtype::F32 => m3k.burnin_fwd_for_state(ds),
            WeightDtype::Bf16 => &m3k.m3_burnin_fwd_typed_bf16,
            WeightDtype::F16 => &m3k.m3_burnin_fwd_typed_f16,
        };
        let mut bld = ctx.stream.launch_builder(kernel);
        bld.arg(&ssm_state);
        bld.arg(&k_state);
        bld.arg(&v_state);
        let y = acts.y.cached_ptr();
        bld.arg(&y);
        bld.arg(acts.h_saved.inner_mut());
        bld.arg(acts.k_prev_saved.inner_mut());
        bld.arg(acts.v_prev_saved.inner_mut());
        let x = acts.x.cached_ptr();
        let k = acts.k.cached_ptr();
        let q = acts.q.cached_ptr();
        bld.arg(&x);
        bld.arg(&k);
        bld.arg(&q);
        bld.arg(alpha_scratch.inner());
        bld.arg(beta_scratch.inner());
        bld.arg(gamma_scratch.inner());
        bld.arg(&dp_ptr);
        bld.arg(&b_i);
        bld.arg(&t_i);
        bld.arg(&nh_i);
        bld.arg(&hd_i);
        bld.arg(&ds_i);
        unsafe { bld.launch(cfg) }.map_err(|e| format!("m3_mixed F6 burnin: {e:?}"))?;
    }

    // F7: gating — silu_gate_fwd_typed or rmsnorm_gated_fwd_typed.
    if dims.is_outproj_norm {
        assert!(di <= 1024, "d_inner {di} > 1024 rmsnorm_gated shm limit");
        let nw_ptr = w.norm_gate_weight.ptr();
        let bt_i = bt as i32;
        let di_i = di as i32;
        let hd_i = hd as i32;
        let grid = cudarc::driver::LaunchConfig {
            grid_dim: (bt as u32, 1, 1),
            block_dim: (di as u32, 1, 1),
            shared_mem_bytes: (di * std::mem::size_of::<f32>()) as u32,
        };
        let mut bld = ctx
            .stream
            .launch_builder(m3k.rmsnorm_gated_fwd_typed.get(dtype));
        let g = acts.gated.cached_ptr();
        let gr = acts.gated_rms_vals.cached_ptr();
        let y = acts.y.cached_ptr();
        let z = acts.z.cached_ptr();
        bld.arg(&g);
        bld.arg(&gr);
        bld.arg(&y);
        bld.arg(&z);
        bld.arg(&nw_ptr);
        bld.arg(&bt_i);
        bld.arg(&di_i);
        bld.arg(&hd_i);
        let eps_g5: f32 = dims.rms_norm_eps;
        bld.arg(&eps_g5);
        let z_stride = di as i32;
        bld.arg(&z_stride);
        unsafe { bld.launch(grid) }.map_err(|e| format!("m3_mixed F7 rmsnorm_gated: {e:?}"))?;
    } else {
        let n = (bt * di) as i32;
        let mut bld = ctx
            .stream
            .launch_builder(m3k.silu_gate_fwd_typed.get(dtype));
        let g = acts.gated.cached_ptr();
        let y = acts.y.cached_ptr();
        let z = acts.z.cached_ptr();
        bld.arg(&g);
        bld.arg(&y);
        bld.arg(&z);
        bld.arg(&n);
        let di_i = di as i32;
        bld.arg(&di_i);
        bld.arg(&di_i);
        unsafe { bld.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("m3_mixed F7 silu_gate: {e:?}"))?;
    }

    // F8: out_proj GEMM typed + residual_add_f32_typed (typed → f32 residual).
    gpu_gemm_typed_forward_raw(
        ctx,
        TypedPtr {
            ptr: out_flat_scratch.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: acts.gated.cached_ptr(),
            dtype,
        },
        TypedPtr {
            ptr: w.out_proj_w.ptr(),
            dtype,
        },
        None,
        (bt, di, dm),
    )?;
    {
        let n = (bt * dm) as i32;
        let mut bld = ctx
            .stream
            .launch_builder(m3k.residual_add_f32_typed.get(dtype));
        let dst = stream_out;
        let a = acts.residual.cached_ptr();
        let b = out_flat_scratch.cached_ptr();
        bld.arg(&dst);
        bld.arg(&a);
        bld.arg(&b);
        bld.arg(&n);
        unsafe { bld.launch(grid_1d(bt * dm)) }
            .map_err(|e| format!("m3_mixed F8 residual_add: {e:?}"))?;
    }

    Ok(())
}

/// Scratch for M3 mixed forward. `proj_flat`/`out_flat` typed; alpha/beta/
/// gamma stay f32 (precision-critical coefficients). Pre-allocated at
/// trainer construction per CUDA Graph zero-alloc invariant.
pub struct GpuMamba3MixedScratch {
    pub proj_flat: DtypedBuf,
    pub out_flat: DtypedBuf,
    pub alpha: GpuBuffer,
    pub beta: GpuBuffer,
    pub gamma: GpuBuffer,
    /// fp64 staging for the chunk-parallel angle accumulation
    /// (`[B * n_chunks * nh * n_angles]` doubles each).
    pub angle_chunk_sums: crate::mamba_ssm::gpu::buffers::GpuByteBuffer,
    pub angle_chunk_carries: crate::mamba_ssm::gpu::buffers::GpuByteBuffer,

    // Backward-only typed staging buffers (activation grads on the
    // wire match activation storage dtype per AMP precision invariant).
    /// typed `[B*T*d_model]` — d_temporal staged for typed dW GEMM at out_proj.
    pub d_temporal_typed: DtypedBuf,
    /// typed `[B*T*d_inner]` — gradient of `gated` (out_proj_bwd dX → input to
    /// rmsnorm_gated_bwd_typed).
    pub d_gated_typed: DtypedBuf,
    /// typed `[B*T*d_inner]` — gradient of `y` from rmsnorm_gated_bwd_typed.
    pub d_y_typed: DtypedBuf,
    /// typed `[B*T*d_inner]` — gradient of `z` from rmsnorm_gated_bwd_typed.
    pub d_z_typed: DtypedBuf,
    /// typed `[B*T*ng*ds]` — gradient of `b_normed` (head→group reduce).
    pub d_b_normed_typed: DtypedBuf,
    /// typed `[B*T*ng*ds]` — gradient of `c_normed`.
    pub d_c_normed_typed: DtypedBuf,
    /// typed `[B*T*ng*ds]` — gradient of `b_raw` (BCNorm bwd output).
    pub d_b_raw_typed: DtypedBuf,
    /// typed `[B*T*ng*ds]` — gradient of `c_raw`.
    pub d_c_raw_typed: DtypedBuf,
    /// typed `[B*T*ip]` — gradient of `proj_flat` (split_bwd output → in_proj dY).
    pub d_proj_typed: DtypedBuf,
    /// typed `[B*T*d_model]` — gradient of `post_norm` (in_proj dX → rmsnorm_bwd dy).
    pub d_post_norm_typed: DtypedBuf,

    // F6 chunked-forward intermediates (added so use_parallel_scan=true works
    // in the mixed forward; backward path uses the saved versions in acts).
    /// f32 `[B*nh]` — adt = a_val · dt, fed into m3_dA_cumsum.
    pub adt_temp: GpuBuffer,
    /// f32 [B * nh * hd * ds] — output of m3_state_passing_fwd, fed to
    /// m3_writeback_parallel_states for persistent SSM state writeback.
    pub final_states: GpuBuffer,

    pub dtype: WeightDtype,
}

impl GpuMamba3MixedScratch {
    pub fn new(
        stream: &Arc<CudaStream>,
        cfg: &Mamba3Config,
        batch: usize,
        seq_len: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let bt = batch * seq_len;
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ng = cfg.ngroups;
        let ds = cfg.d_state;
        let nh = cfg.nheads();
        let hd = cfg.headdim;
        let ip = cfg.in_proj_out_dim();
        let n_chunks_max = seq_len.div_ceil(CHUNK_SIZE);
        let s = Self {
            proj_flat: DtypedBuf::zeros(stream, bt * ip, dtype)?,
            out_flat: DtypedBuf::zeros(stream, bt * dm, dtype)?,
            alpha: GpuBuffer::zeros(stream, bt * nh)?,
            beta: GpuBuffer::zeros(stream, bt * nh)?,
            gamma: GpuBuffer::zeros(stream, bt * nh)?,
            d_temporal_typed: DtypedBuf::zeros(stream, bt * dm, dtype)?,
            d_gated_typed: DtypedBuf::zeros(stream, bt * di, dtype)?,
            d_y_typed: DtypedBuf::zeros(stream, bt * di, dtype)?,
            d_z_typed: DtypedBuf::zeros(stream, bt * di, dtype)?,
            d_b_normed_typed: DtypedBuf::zeros(stream, bt * ng * ds, dtype)?,
            d_c_normed_typed: DtypedBuf::zeros(stream, bt * ng * ds, dtype)?,
            d_b_raw_typed: DtypedBuf::zeros(stream, bt * ng * ds, dtype)?,
            d_c_raw_typed: DtypedBuf::zeros(stream, bt * ng * ds, dtype)?,
            d_proj_typed: DtypedBuf::zeros(stream, bt * ip, dtype)?,
            d_post_norm_typed: DtypedBuf::zeros(stream, bt * dm, dtype)?,
            adt_temp: GpuBuffer::zeros(stream, bt * nh)?,
            final_states: GpuBuffer::zeros(stream, batch * nh * hd * ds)?,
            angle_chunk_sums: crate::mamba_ssm::gpu::buffers::GpuByteBuffer::zeros(
                stream,
                batch
                    * n_chunks_max
                    * nh
                    * cfg.num_rope_angles().max(1)
                    * std::mem::size_of::<f64>(),
            )?,
            angle_chunk_carries: crate::mamba_ssm::gpu::buffers::GpuByteBuffer::zeros(
                stream,
                batch
                    * n_chunks_max
                    * nh
                    * cfg.num_rope_angles().max(1)
                    * std::mem::size_of::<f64>(),
            )?,
            dtype,
        };
        stream
            .synchronize()
            .map_err(|e| format!("sync after m3 mixed scratch alloc: {e:?}"))?;
        Ok(s)
    }
}

/// Full backbone M3 forward in mixed precision. Mirrors
/// [`super::mamba3_gpu::gpu_forward_mamba3_backbone`].
pub fn gpu_forward_mamba3_backbone_mixed(
    exec: &M3Exec<'_>,
    temporal_f32: &mut GpuBuffer,
    acts: &mut GpuMamba3BackboneMixedActs,
    w: &GpuMamba3TrainMixedWeights,
    mamba_input: &GpuBuffer,
    states: GpuMamba3StateBufs<'_>,
    scratch: &mut GpuMamba3MixedScratch,
) -> Result<(), String> {
    let M3Exec {
        ctx,
        kernels: m3k,
        dims,
    } = *exec;
    let dtype = acts.dtype;
    let bt = dims.bt();
    let dm = dims.d_model;
    let ds = dims.d_state;
    let nh = dims.nheads;
    let hd = dims.headdim;
    let na = dims.n_angles.max(1);

    // The acts constructor sizes the sequential-scan tapes from its own
    // use_parallel_scan flag; the launch branches on the dims flag. A
    // mismatch means the sequential kernels would write past 1-element
    // sentinels — refuse loudly instead.
    if !dims.use_parallel_scan {
        let need = dims.batch * (dims.seq_len + 1) * dims.d_inner * ds;
        if acts.layers.first().is_some_and(|l| l.h_saved.len() < need) {
            return Err(
                "m3_mixed forward: dims request the sequential scan but the acts \
                 were allocated for the chunked path (sentinel tapes) — construct \
                 the acts with the same use_parallel_scan flag the forward runs with"
                    .into(),
            );
        }
    }

    if dims.use_parallel_scan {
        // Stateless window: the chunked kernels ignore entering SSM/K/V
        // state; zero the angle accumulator too so all four states behave
        // identically (see gpu_forward_mamba3_backbone).
        states.ssm.zero(&ctx.stream)?;
        states.k.zero(&ctx.stream)?;
        states.v.zero(&ctx.stream)?;
        states.angle.zero(&ctx.stream)?;
    }

    // input_proj: if identity (len_elems==0), D2D copy mamba_input →
    // temporal. Otherwise typed GEMM. Matches M1 pattern.
    if w.compute.input_proj_w.len_elems() == 0 {
        let bytes = bt * dm * 4;
        let stream = ctx.stream.cu_stream();
        unsafe {
            let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                temporal_f32.cached_ptr(),
                mamba_input.cached_ptr(),
                bytes,
                stream,
            );
            if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(format!("m3_mixed identity input_proj D2D: {res:?}"));
            }
        }
    } else {
        // Non-identity input_proj — the trainable input-embedding path.
        // Three launches, mirroring the Mamba-1 mixed pipeline: (1) cast
        // the f32 input to the compute dtype, SAVED in acts (the backward
        // dW consumes it); (2) typed GEMM with the f32 bias into the saved
        // typed outputs; (3) upcast the outputs to seed the f32 residual
        // stream (norm math and every residual carry stay f32).
        let mid = dims.mamba_input_dim;
        {
            let n = (bt * mid) as i32;
            let cast = match dtype {
                WeightDtype::Bf16 => &m3k.cast_f32_to_bf16,
                WeightDtype::F16 => &m3k.cast_f32_to_f16,
                WeightDtype::F32 => {
                    return Err("m3_mixed forward: unexpected f32 compute dtype".into());
                }
            };
            let dst = acts.input_proj_inputs.cached_ptr();
            let src = mamba_input.cached_ptr();
            let mut bld = ctx.stream.launch_builder(cast);
            bld.arg(&dst);
            bld.arg(&src);
            bld.arg(&n);
            unsafe { bld.launch(grid_1d(bt * mid)) }
                .map_err(|e| format!("m3_mixed input_proj input cast: {e:?}"))?;
        }
        gpu_gemm_typed_forward_raw(
            ctx,
            TypedPtr {
                ptr: acts.input_proj_outputs.cached_ptr(),
                dtype,
            },
            TypedPtr {
                ptr: acts.input_proj_inputs.cached_ptr(),
                dtype,
            },
            TypedPtr {
                ptr: w.compute.input_proj_w.ptr(),
                dtype,
            },
            Some(w.compute.input_proj_b.ptr()),
            (bt, mid, dm),
        )?;
        {
            let n = (bt * dm) as i32;
            let cast = match dtype {
                WeightDtype::Bf16 => &m3k.cast_bf16_to_f32,
                WeightDtype::F16 => &m3k.cast_f16_to_f32,
                WeightDtype::F32 => {
                    return Err("m3_mixed forward: unexpected f32 compute dtype".into());
                }
            };
            let dst = temporal_f32.cached_ptr();
            let src = acts.input_proj_outputs.cached_ptr();
            let mut bld = ctx.stream.launch_builder(cast);
            bld.arg(&dst);
            bld.arg(&src);
            bld.arg(&n);
            unsafe { bld.launch(grid_1d(bt * dm)) }
                .map_err(|e| format!("m3_mixed input_proj residual upcast: {e:?}"))?;
        }
    }

    // Seed the residual chain: layer 0's slot gets the input_proj output
    // once; from there every layer's residual_add writes the next slot
    // (the LAST layer writes temporal_f32 back - the public contract).
    acts.layers[0]
        .residual
        .copy_from_raw(temporal_f32, &ctx.stream)?;

    // Mamba layers in forward order — per-layer offset into flat state buffers.
    let f32_sz = std::mem::size_of::<f32>() as u64;
    let ssm_base = states.ssm.cached_ptr();
    let k_base = states.k.cached_ptr();
    let v_base = states.v.cached_ptr();
    let a_base = states.angle.cached_ptr();

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
        let stream_out = if l + 1 < dims.n_layers {
            acts.layers[l + 1].residual.cached_ptr()
        } else {
            // The LAST layer's residual_add lands in the pre-norm save the
            // backward replays from; norm_f below writes the public output.
            acts.norm_f_input.cached_ptr()
        };
        gpu_forward_mamba3_layer_mixed(
            exec,
            stream_out,
            &mut acts.layers[l],
            &w.compute.layers[l],
            &layer_ptrs,
            scratch,
            dtype,
        )?;
    }

    // norm_f — the f32 rmsnorm from the saved pre-norm residual into the
    // caller's f32 temporal: the head consumes the POST-norm stream, the
    // exact mirror of the f32 lane (norm_f_weight is f32 in the mixed
    // arena). The backward's rmsnorm_bwd replays from norm_f_input +
    // norm_f_rms saved here.
    {
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let eps: f32 = dims.rms_norm_eps;
        let mut bld = ctx.stream.launch_builder(&m3k.rmsnorm_fwd);
        let rms = acts.norm_f_rms.cached_ptr();
        let x = acts.norm_f_input.cached_ptr();
        let nw = w.compute.norm_f_weight.ptr();
        bld.arg(temporal_f32.inner_mut());
        bld.arg(&rms);
        bld.arg(&x);
        bld.arg(&nw);
        bld.arg(&bt_i);
        bld.arg(&dm_i);
        bld.arg(&eps);
        unsafe { bld.launch(grid_norm(bt, dm)) }.map_err(|e| format!("m3_mixed norm_f: {e:?}"))?;
    }

    Ok(())
}
