//! per-kernel parity tests for the 2 HIGHEST-RISK typed M3
//! "final grad" kernels: `m3_dqkv`, `m3_dqktheta`. Each typed variant is
//! compared against its f32 oracle on identical random inputs.
//!
//! These are the most complex kernels in the M3 backward pipeline — the
//! `m3_dqkv` kernel holds register-resident d_state and multiple smem tiles
//! (Q/K/V/dO/ssm_states) across all chunks in reverse order. `m3_dqktheta`
//! threads per-timestep inverse-RoPE + bias accumulation.
//!
//! Typed I/O is limited to saved activations (Q_rot, K_scaled, V_in, dO for
//! dqkv; Q_raw, K_raw for dqktheta). All gradient outputs stay f32.
//!
//! The other 2 "final grad" kernels (`m3_ddt_dtrap`, `m3_final_grads`) are
//! pure-f32 (scalar grad math on f32 arrays) — no typed variant to test.

#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;
#[path = "common/frozen_kernels.rs"]
mod frozen_kernels;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::sys::CUfunction_attribute_enum as CUfunction_attribute;
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use frozen_kernels::frozen_module;
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::{IndexWidth, grid_1d_cover, y_runs, yz_runs};
use mamba_rs::mamba3_siso::gpu::kernels::{Mamba3Kernels, angle_backward_cfg};

/// The dQK-theta stage as it was released before the row-streaming stage
/// replaced it: f32, the bf16 and f16 forms, and the half transport form.
const FROZEN_DQKTHETA: &str = include_str!("fixtures/m3_dqktheta_reference.cu");

/// The angle gradients' reverse sum as it was released before the
/// batched-load kernel replaced it.
const FROZEN_ANGLE_BACKWARD: &str = include_str!("fixtures/m3_angle_backward_reference.cu");

// Config: realistic-ish head dims, 2 chunks with partial last chunk coverage.
const B: usize = 2;
const T: usize = 10;
const NH: usize = 2;
const HD: usize = 4;
const DS: usize = 8;
const CS: usize = 8;
const N_CHUNKS: usize = T.div_ceil(CS);
const D_INNER: usize = NH * HD;
const N_ANGLES: usize = 2;

fn det_rand(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & 0xFFFF) as f32 / 65536.0 - 0.5
        })
        .collect()
}

fn cos_norm(a: &[f32], b: &[f32]) -> (f32, f32) {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    let cos = if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        1.0
    };
    let ratio = if na > 0.0 {
        ((nb.sqrt()) / (na.sqrt())) as f32
    } else {
        1.0
    };
    (cos, ratio)
}

fn assert_close(label: &str, a: &[f32], b: &[f32], cos_min: f32, norm_tol: f32) {
    let (cos, ratio) = cos_norm(a, b);
    eprintln!("  {label}: cos={cos:.6} norm={ratio:.4} n={}", a.len());
    assert!(cos >= cos_min, "{label}: cos {cos} < {cos_min}");
    assert!(
        (ratio - 1.0).abs() <= norm_tol,
        "{label}: norm {ratio} outside [1 ± {norm_tol}]"
    );
}

fn upload_typed(ctx: &GpuCtx, data: &[f32], dtype: WeightDtype) -> DtypedBuf {
    let buf = DtypedBuf::zeros(&ctx.stream, data.len(), dtype).unwrap();
    ctx.stream.synchronize().unwrap();
    buf.upload_f32(&ctx.stream, data).unwrap();
    ctx.stream.synchronize().unwrap();
    buf
}

fn upload_f32(ctx: &GpuCtx, data: &[f32]) -> GpuBuffer {
    let mut b = GpuBuffer::zeros(&ctx.stream, data.len()).unwrap();
    ctx.stream.synchronize().unwrap();
    b.upload(&ctx.stream, data).unwrap();
    ctx.stream.synchronize().unwrap();
    b
}

fn download_f32(ctx: &GpuCtx, buf: &GpuBuffer, n: usize) -> Vec<f32> {
    let mut v = vec![0f32; n];
    buf.download(&ctx.stream, &mut v).unwrap();
    ctx.stream.synchronize().unwrap();
    v
}

/// Launches an `m3_dqktheta` build `f` over `rows` rows of `ds` states and
/// `na` angles: `ptrs` are its five outputs, then its eight inputs.
fn launch_dqktheta(
    ctx: &GpuCtx,
    f: &CudaFunction,
    ptrs: &[CUdeviceptr; 13],
    (rows, ds, na): (usize, usize, usize),
) {
    let mut bld = ctx.stream.launch_builder(f);
    for p in ptrs {
        bld.arg(p);
    }
    let dims = [rows, ds, na].map(|v| v as i32);
    for d in &dims {
        bld.arg(d);
    }
    unsafe { bld.launch(grid_1d_cover(rows).unwrap()) }.unwrap();
}

fn make_m3k(ctx: &GpuCtx) -> Mamba3Kernels {
    Mamba3Kernels::compile(ctx.stream.context(), arch::arch0()).unwrap()
}

fn tolerances(dtype: WeightDtype) -> (f32, f32) {
    match dtype {
        WeightDtype::Bf16 => (0.99, 0.05),
        WeightDtype::F16 => (0.999, 0.02),
        WeightDtype::F32 | WeightDtype::Tf32 => unreachable!(),
    }
}

// ─── m3_dqkv ───────────────────────────────────────────────────────────
//
// 6 outputs: dQ_mid, dK_mid, dV, dADT, dQK_dot_out, dD_partials.
// dD output is now per-(b,h) partials of length B*nh instead of
// [nh] atomicAdd accumulator. Callers reduce across B via reduce_sum_axis0.

fn check_dqkv(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);

    let q_vals = det_rand(B * T * NH * DS, 0x9B01);
    let ks_vals = det_rand(B * T * NH * DS, 0x9B02);
    let v_vals = det_rand(B * T * D_INNER, 0x9B03);
    let dcs_vals: Vec<f32> = (0..B * N_CHUNKS * NH * CS)
        .map(|i| -0.015 * (i as f32 % 4.0 + 1.0))
        .collect();
    // dA_cs_sum = cumsum end for each (b, chunk, h). Use sum of 2 adjacent slots.
    let mut dcs_sum_vals = vec![0f32; B * N_CHUNKS * NH];
    for b in 0..B {
        for c in 0..N_CHUNKS {
            for h in 0..NH {
                let base = ((b * N_CHUNKS + c) * NH + h) * CS;
                let chunk_len = (CS).min(T - c * CS);
                dcs_sum_vals[(b * N_CHUNKS + c) * NH + h] = dcs_vals[base + chunk_len - 1];
            }
        }
    }
    let qk_vals = det_rand(B * T * NH, 0x9B04);
    let ssm_vals = det_rand(B * N_CHUNKS * NH * HD * DS, 0x9B05);
    let do_vals = det_rand(B * T * D_INNER, 0x9B06);
    let d_vals: Vec<f32> = (0..NH).map(|i| 0.5 + 0.1 * (i as f32)).collect();

    let q_f32 = upload_f32(&ctx, &q_vals);
    let ks_f32 = upload_f32(&ctx, &ks_vals);
    let v_f32 = upload_f32(&ctx, &v_vals);
    let dcs_buf = upload_f32(&ctx, &dcs_vals);
    let dcs_sum_buf = upload_f32(&ctx, &dcs_sum_vals);
    let qk_buf = upload_f32(&ctx, &qk_vals);
    let ssm_buf = upload_f32(&ctx, &ssm_vals);
    let do_f32 = upload_f32(&ctx, &do_vals);
    let d_buf = upload_f32(&ctx, &d_vals);

    let n_q = B * T * NH * DS;
    let n_v = B * T * D_INNER;
    let n_th = B * T * NH;
    // dD output is per-(b,h) partials length B*NH.
    let n_d = B * N_CHUNKS * NH;

    // f32 oracle.
    let dq_ref = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dk_ref = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dv_ref = GpuBuffer::zeros(&ctx.stream, n_v).unwrap();
    let dadt_ref = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dqk_ref = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dd_ref = GpuBuffer::zeros(&ctx.stream, n_d).unwrap();
    ctx.stream.synchronize().unwrap();

    // Shared memory size: q_sm + k_sm + v_sm + do_sm + da_cs_sm + qk_sm + ssm_sm.
    // Must mirror the launch-site formula (two operand tiles, V/dO tiles,
    // per-step lanes, da/qk lanes, and TWO head-state tiles for the
    // warp-parallel decay-gradient section).
    // Includes the strict-upper-triangle pair matrices plus the
    // decay triangle and the two per-step exp lanes.
    // The four time-major tiles carry one float of padding per row, the
    // same sizing the production launchers use.
    let smem_floats = CS * (DS + 1) * 2
        + CS * (HD + 1) * 2
        + CS * 4
        + HD * DS * 2
        + CS * (CS - 1) * 3 / 2
        + CS * 2;
    let smem_bytes = (smem_floats * 4) as u32;
    let use_mats_i: i32 = 1;
    // t-split contract: one head per block, blockDim.y = t-split lanes;
    // chunks ride grid z with the entering state supplied per chunk.
    let enter = GpuBuffer::zeros(&ctx.stream, B * N_CHUNKS * NH * HD * DS).unwrap();
    ctx.stream.synchronize().unwrap();
    let cfg = LaunchConfig {
        grid_dim: (NH as u32, B as u32, N_CHUNKS as u32),
        block_dim: (HD as u32, 16, 1),
        shared_mem_bytes: smem_bytes,
    };

    let (bi, ti, nhi, hdi, dsi, csi) = (
        B as i32, T as i32, NH as i32, HD as i32, DS as i32, CS as i32,
    );

    for (y_first, z_first, cfg) in yz_runs(cfg) {
        let mut bld = ctx.stream.launch_builder(&m3k.m3_dqkv.i32);
        let args = [
            dq_ref.cached_ptr(),
            dk_ref.cached_ptr(),
            dv_ref.cached_ptr(),
            dadt_ref.cached_ptr(),
            dqk_ref.cached_ptr(),
            dd_ref.cached_ptr(),
            q_f32.cached_ptr(),
            ks_f32.cached_ptr(),
            v_f32.cached_ptr(),
            dcs_buf.cached_ptr(),
            dcs_sum_buf.cached_ptr(),
            qk_buf.cached_ptr(),
            ssm_buf.cached_ptr(),
            do_f32.cached_ptr(),
            d_buf.cached_ptr(),
            enter.cached_ptr(),
        ];
        for a in &args {
            bld.arg(a);
        }
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&nhi);
        bld.arg(&hdi);
        bld.arg(&dsi);
        bld.arg(&csi);
        bld.arg(&use_mats_i);
        bld.arg(&y_first);
        bld.arg(&z_first);
        bld.arg(&0u64);
        unsafe { bld.launch(cfg) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();

    let dq_ref_v = download_f32(&ctx, &dq_ref, n_q);
    let dk_ref_v = download_f32(&ctx, &dk_ref, n_q);
    let dv_ref_v = download_f32(&ctx, &dv_ref, n_v);
    let dadt_ref_v = download_f32(&ctx, &dadt_ref, n_th);
    let dqk_ref_v = download_f32(&ctx, &dqk_ref, n_th);
    let dd_ref_v = download_f32(&ctx, &dd_ref, n_d);

    // Typed variant.
    let q_t = upload_typed(&ctx, &q_vals, dtype);
    let ks_t = upload_typed(&ctx, &ks_vals, dtype);
    let v_t = upload_typed(&ctx, &v_vals, dtype);
    let do_t = upload_typed(&ctx, &do_vals, dtype);

    let dq_t = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dk_t = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dv_t = GpuBuffer::zeros(&ctx.stream, n_v).unwrap();
    let dadt_t = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dqk_t = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dd_t = GpuBuffer::zeros(&ctx.stream, n_d).unwrap();
    ctx.stream.synchronize().unwrap();

    for (y_first, z_first, cfg) in yz_runs(cfg) {
        let mut bld = ctx.stream.launch_builder(m3k.m3_dqkv_typed.i32.get(dtype));
        let args = [
            dq_t.cached_ptr(),
            dk_t.cached_ptr(),
            dv_t.cached_ptr(),
            dadt_t.cached_ptr(),
            dqk_t.cached_ptr(),
            dd_t.cached_ptr(),
            q_t.cached_ptr(),
            ks_t.cached_ptr(),
            v_t.cached_ptr(),
            dcs_buf.cached_ptr(),
            dcs_sum_buf.cached_ptr(),
            qk_buf.cached_ptr(),
            ssm_buf.cached_ptr(),
            do_t.cached_ptr(),
            d_buf.cached_ptr(),
            enter.cached_ptr(),
        ];
        for a in &args {
            bld.arg(a);
        }
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&nhi);
        bld.arg(&hdi);
        bld.arg(&dsi);
        bld.arg(&csi);
        bld.arg(&use_mats_i);
        bld.arg(&y_first);
        bld.arg(&z_first);
        bld.arg(&0u64);
        unsafe { bld.launch(cfg) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();

    let dq_t_v = download_f32(&ctx, &dq_t, n_q);
    let dk_t_v = download_f32(&ctx, &dk_t, n_q);
    let dv_t_v = download_f32(&ctx, &dv_t, n_v);
    let dadt_t_v = download_f32(&ctx, &dadt_t, n_th);
    let dqk_t_v = download_f32(&ctx, &dqk_t, n_th);
    let dd_t_v = download_f32(&ctx, &dd_t, n_d);

    eprintln!("m3_dqkv {dtype:?}:");
    let (cos_min, norm_tol) = tolerances(dtype);
    assert_close("dQ_mid", &dq_ref_v, &dq_t_v, cos_min, norm_tol);
    assert_close("dK_mid", &dk_ref_v, &dk_t_v, cos_min, norm_tol);
    assert_close("dV", &dv_ref_v, &dv_t_v, cos_min, norm_tol);
    assert_close("dADT", &dadt_ref_v, &dadt_t_v, cos_min, norm_tol);
    assert_close("dQK_dot", &dqk_ref_v, &dqk_t_v, cos_min, norm_tol);
    assert_close("dD", &dd_ref_v, &dd_t_v, cos_min, norm_tol);
}

#[test]
fn m3_dqkv_bf16() {
    check_dqkv(WeightDtype::Bf16);
}
#[test]
fn m3_dqkv_f16() {
    check_dqkv(WeightDtype::F16);
}

// ─── m3_dqktheta ───────────────────────────────────────────────────────
//
// kernel now writes 5 outputs (dQ_pre, dK_pre, dAngles_cumsum,
// dScale, dGamma). dQ_bias / dK_bias are produced by a caller-side
// colsum_accumulate over dQ_pre / dK_pre respectively — the f32 oracle vs
// typed comparison still covers them since both paths produce the same
// dQ_pre / dK_pre and run the same colsum on top.

fn check_dqktheta(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);

    let q_raw_vals = det_rand(B * T * NH * DS, 0x9B11);
    let k_raw_vals = det_rand(B * T * NH * DS, 0x9B12);
    let scale_vals: Vec<f32> = (0..B * T * NH)
        .map(|i| 0.3 + 0.01 * (i as f32 % 10.0))
        .collect();
    let gamma_vals: Vec<f32> = (0..B * T * NH)
        .map(|i| 0.2 + 0.015 * (i as f32 % 7.0))
        .collect();
    let angle_vals: Vec<f32> = (0..B * T * NH * N_ANGLES)
        .map(|i| 0.1 * (i as f32).sin())
        .collect();
    let dq_mid_vals = det_rand(B * T * NH * DS, 0x9B13);
    let dk_mid_vals = det_rand(B * T * NH * DS, 0x9B14);
    let dqk_vals = det_rand(B * T * NH, 0x9B15);

    let q_raw_f32 = upload_f32(&ctx, &q_raw_vals);
    let k_raw_f32 = upload_f32(&ctx, &k_raw_vals);
    let scale_buf = upload_f32(&ctx, &scale_vals);
    let gamma_buf = upload_f32(&ctx, &gamma_vals);
    let angle_buf = upload_f32(&ctx, &angle_vals);
    let dq_mid_buf = upload_f32(&ctx, &dq_mid_vals);
    let dk_mid_buf = upload_f32(&ctx, &dk_mid_vals);
    let dqk_buf = upload_f32(&ctx, &dqk_vals);

    let n_q = B * T * NH * DS;
    let n_ang = B * T * NH * N_ANGLES;
    let n_th = B * T * NH;
    let n_bias = NH * DS;

    // f32 oracle.
    let dqpre_ref = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dkpre_ref = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dang_ref = GpuBuffer::zeros(&ctx.stream, n_ang).unwrap();
    let dscale_ref = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dgamma_ref = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dqb_ref = GpuBuffer::zeros(&ctx.stream, n_bias).unwrap();
    let dkb_ref = GpuBuffer::zeros(&ctx.stream, n_bias).unwrap();
    ctx.stream.synchronize().unwrap();

    let dims = (B * T * NH, DS, N_ANGLES);
    launch_dqktheta(
        &ctx,
        &m3k.m3_dqktheta.i32,
        &[
            dqpre_ref.cached_ptr(),
            dkpre_ref.cached_ptr(),
            dang_ref.cached_ptr(),
            dscale_ref.cached_ptr(),
            dgamma_ref.cached_ptr(),
            q_raw_f32.cached_ptr(),
            k_raw_f32.cached_ptr(),
            scale_buf.cached_ptr(),
            gamma_buf.cached_ptr(),
            angle_buf.cached_ptr(),
            dq_mid_buf.cached_ptr(),
            dk_mid_buf.cached_ptr(),
            dqk_buf.cached_ptr(),
        ],
        dims,
    );
    ctx.stream.synchronize().unwrap();

    // produce dQ_bias/dK_bias from dQ_pre/dK_pre via colsum_accumulate.
    let cs_grid = LaunchConfig {
        grid_dim: (((NH * DS) as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    {
        let rows = (B * T) as i32;
        let cols = (NH * DS) as i32;
        let mut bld = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        let dqb_p = dqb_ref.cached_ptr();
        bld.arg(&dqb_p);
        bld.arg(dqpre_ref.inner());
        bld.arg(&rows);
        bld.arg(&cols);
        unsafe { bld.launch(cs_grid) }.unwrap();
    }
    {
        let rows = (B * T) as i32;
        let cols = (NH * DS) as i32;
        let mut bld = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        let dkb_p = dkb_ref.cached_ptr();
        bld.arg(&dkb_p);
        bld.arg(dkpre_ref.inner());
        bld.arg(&rows);
        bld.arg(&cols);
        unsafe { bld.launch(cs_grid) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();

    let dqpre_ref_v = download_f32(&ctx, &dqpre_ref, n_q);
    let dkpre_ref_v = download_f32(&ctx, &dkpre_ref, n_q);
    let dang_ref_v = download_f32(&ctx, &dang_ref, n_ang);
    let dscale_ref_v = download_f32(&ctx, &dscale_ref, n_th);
    let dgamma_ref_v = download_f32(&ctx, &dgamma_ref, n_th);
    let dqb_ref_v = download_f32(&ctx, &dqb_ref, n_bias);
    let dkb_ref_v = download_f32(&ctx, &dkb_ref, n_bias);

    // Typed variant.
    let q_raw_t = upload_typed(&ctx, &q_raw_vals, dtype);
    let k_raw_t = upload_typed(&ctx, &k_raw_vals, dtype);

    let dqpre_t = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dkpre_t = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dang_t = GpuBuffer::zeros(&ctx.stream, n_ang).unwrap();
    let dscale_t = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dgamma_t = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dqb_t = GpuBuffer::zeros(&ctx.stream, n_bias).unwrap();
    let dkb_t = GpuBuffer::zeros(&ctx.stream, n_bias).unwrap();
    ctx.stream.synchronize().unwrap();

    launch_dqktheta(
        &ctx,
        m3k.m3_dqktheta_typed.i32.get(dtype),
        &[
            dqpre_t.cached_ptr(),
            dkpre_t.cached_ptr(),
            dang_t.cached_ptr(),
            dscale_t.cached_ptr(),
            dgamma_t.cached_ptr(),
            q_raw_t.cached_ptr(),
            k_raw_t.cached_ptr(),
            scale_buf.cached_ptr(),
            gamma_buf.cached_ptr(),
            angle_buf.cached_ptr(),
            dq_mid_buf.cached_ptr(),
            dk_mid_buf.cached_ptr(),
            dqk_buf.cached_ptr(),
        ],
        dims,
    );
    ctx.stream.synchronize().unwrap();

    // typed path also colsums dQ_pre/dK_pre to produce bias grads.
    {
        let rows = (B * T) as i32;
        let cols = (NH * DS) as i32;
        let mut bld = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        let dqb_p = dqb_t.cached_ptr();
        bld.arg(&dqb_p);
        bld.arg(dqpre_t.inner());
        bld.arg(&rows);
        bld.arg(&cols);
        unsafe { bld.launch(cs_grid) }.unwrap();
    }
    {
        let rows = (B * T) as i32;
        let cols = (NH * DS) as i32;
        let mut bld = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        let dkb_p = dkb_t.cached_ptr();
        bld.arg(&dkb_p);
        bld.arg(dkpre_t.inner());
        bld.arg(&rows);
        bld.arg(&cols);
        unsafe { bld.launch(cs_grid) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();

    let dqpre_t_v = download_f32(&ctx, &dqpre_t, n_q);
    let dkpre_t_v = download_f32(&ctx, &dkpre_t, n_q);
    let dang_t_v = download_f32(&ctx, &dang_t, n_ang);
    let dscale_t_v = download_f32(&ctx, &dscale_t, n_th);
    let dgamma_t_v = download_f32(&ctx, &dgamma_t, n_th);
    let dqb_t_v = download_f32(&ctx, &dqb_t, n_bias);
    let dkb_t_v = download_f32(&ctx, &dkb_t, n_bias);

    eprintln!("m3_dqktheta {dtype:?}:");
    let (cos_min, norm_tol) = tolerances(dtype);
    assert_close("dQ_pre", &dqpre_ref_v, &dqpre_t_v, cos_min, norm_tol);
    assert_close("dK_pre", &dkpre_ref_v, &dkpre_t_v, cos_min, norm_tol);
    assert_close("dAngles", &dang_ref_v, &dang_t_v, cos_min, norm_tol);
    assert_close("dScale", &dscale_ref_v, &dscale_t_v, cos_min, norm_tol);
    assert_close("dGamma", &dgamma_ref_v, &dgamma_t_v, cos_min, norm_tol);
    assert_close("dQ_bias", &dqb_ref_v, &dqb_t_v, cos_min, norm_tol);
    assert_close("dK_bias", &dkb_ref_v, &dkb_t_v, cos_min, norm_tol);
}

#[test]
fn m3_dqktheta_bf16() {
    check_dqktheta(WeightDtype::Bf16);
}
#[test]
fn m3_dqktheta_f16() {
    check_dqktheta(WeightDtype::F16);
}

/// `m3_dqktheta` walks each row in registers with the arithmetic its
/// released whole-row form compiled to, so it gives that form's bits: the
/// f32 kernel and the bf16 and f16 ones, staged and direct, and the half
/// transport form that served sm_89, against both index-width builds; at
/// state widths that do and do not fill whole quads, with no angles, with
/// angles over part of the state and over all of it, and on rows that sit
/// one value off a quad's alignment, where every load is element-wise; and
/// at angles far enough out that the sine and cosine reduce them the slow
/// way.
#[test]
fn m3_dqktheta_keeps_the_bits_of_its_released_form() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let cap = 64;
    let m3k =
        Mamba3Kernels::compile_with_state_cap(ctx.stream.context(), arch::arch0(), cap).unwrap();
    let frozen = frozen_module(ctx.stream.context(), arch::arch0(), cap, FROZEN_DQKTHETA);
    let (b, t, nh, cs) = (2usize, 150usize, 3usize, 64usize);
    let rows = b * t * nh;
    let mut cases = 0;
    for (ds, na, lead, spread) in [
        (16usize, 4usize, 0usize, 1.0f32),
        (16, 8, 1, 1.0),
        (16, 4, 0, 4.0e5),
        (32, 16, 0, 1.0),
        (33, 8, 0, 1.0),
        (33, 16, 0, 1.0),
        (48, 12, 0, 1.0),
        (64, 16, 0, 1.0),
        (64, 32, 1, 1.0),
        (64, 0, 0, 1.0),
        (6, 3, 0, 1.0),
        (5, 2, 0, 1.0),
        (2, 1, 0, 1.0),
        (1, 0, 0, 1.0),
    ] {
        let n_q = rows * ds;
        let n_ang = rows * na.max(1);
        let q_vals = det_rand(n_q, 0xD701);
        let k_vals = det_rand(n_q, 0xD702);
        let scale = upload_f32(&ctx, &det_rand(rows, 0xD703));
        let gamma = upload_f32(&ctx, &det_rand(rows, 0xD704));
        let angle_vals: Vec<f32> = det_rand(n_ang, 0xD705).iter().map(|a| a * spread).collect();
        let angles = upload_f32(&ctx, &angle_vals);
        // `lead` values ahead of every row buffer put the rows off a quad's
        // alignment.
        let padded = |v: Vec<f32>| [vec![0.0; lead], v].concat();
        let dq_mid = upload_f32(&ctx, &padded(det_rand(n_q, 0xD706)));
        let dk_mid = upload_f32(&ctx, &padded(det_rand(n_q, 0xD707)));
        let dqk = upload_f32(&ctx, &det_rand(rows, 0xD708));
        for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
            let q = upload_typed(&ctx, &padded(q_vals.clone()), dtype);
            let k = upload_typed(&ctx, &padded(k_vals.clone()), dtype);
            let (act, f32_lead) = ((lead * dtype.size_bytes()) as u64, (lead * 4) as u64);
            let run = |launch: &dyn Fn(&[CUdeviceptr; 13]), poison: f32| {
                // Poisoned outputs, so an element a form does not write
                // cannot agree with one it does.
                let outs = [n_q + lead, n_q + lead, n_ang, rows, rows]
                    .map(|n| upload_f32(&ctx, &vec![poison; n]));
                launch(&[
                    outs[0].cached_ptr() + f32_lead,
                    outs[1].cached_ptr() + f32_lead,
                    outs[2].cached_ptr(),
                    outs[3].cached_ptr(),
                    outs[4].cached_ptr(),
                    q.cached_ptr() + act,
                    k.cached_ptr() + act,
                    scale.cached_ptr(),
                    gamma.cached_ptr(),
                    angles.cached_ptr(),
                    dq_mid.cached_ptr() + f32_lead,
                    dk_mid.cached_ptr() + f32_lead,
                    dqk.cached_ptr(),
                ]);
                ctx.stream.synchronize().unwrap();
                let mut got: Vec<Vec<f32>> = outs
                    .iter()
                    .zip([n_q + lead, n_q + lead, n_ang, rows, rows])
                    .map(|(buf, n)| download_f32(&ctx, buf, n))
                    .collect();
                got[0].drain(..lead);
                got[1].drain(..lead);
                // Without angles no form writes an angle gradient.
                if na == 0 {
                    got[2].clear();
                }
                got
            };
            // The released form, a block a chunk of a head, its rows through
            // `tiles` staged tiles or (0) straight from global memory.
            let released = |name: &str, tiles: usize, poison: f32| {
                let f = frozen.load_function(name).unwrap();
                let smem = tiles * cs * ds * 4;
                f.set_attribute(
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem as i32,
                )
                .unwrap();
                let cfg = LaunchConfig {
                    grid_dim: ((b * t.div_ceil(cs)) as u32, nh as u32, 1),
                    block_dim: (cs as u32, 1, 1),
                    shared_mem_bytes: smem as u32,
                };
                let dims = [b, t, nh, ds, na, cs].map(|v| v as i32);
                let staging = i32::from(tiles > 0);
                let y_first = 0i32;
                run(
                    &|ptrs: &[CUdeviceptr; 13]| {
                        let mut bld = ctx.stream.launch_builder(&f);
                        for p in ptrs {
                            bld.arg(p);
                        }
                        for d in &dims {
                            bld.arg(d);
                        }
                        bld.arg(&staging);
                        bld.arg(&y_first);
                        unsafe { bld.launch(cfg) }.unwrap();
                    },
                    poison,
                )
            };
            let suffix = match dtype {
                WeightDtype::Bf16 => "_bf16",
                WeightDtype::F16 => "_f16",
                _ => "",
            };
            let name = format!("m3_reference_dqktheta{suffix}");
            let want = released(&name, 0, 8192.0);
            let mut forms = vec![(format!("{name} staged"), released(&name, 6, -8192.0))];
            if dtype != WeightDtype::F32 {
                let transport = format!("m3_reference_dqktheta_transport{suffix}");
                forms.push((transport.clone(), released(&transport, 4, -4096.0)));
            }
            for width in [IndexWidth::I32, IndexWidth::I64] {
                let f = m3k.m3_dqktheta_typed.get(width).get(dtype);
                let got = run(
                    &|ptrs: &[CUdeviceptr; 13]| launch_dqktheta(&ctx, f, ptrs, (rows, ds, na)),
                    -2048.0,
                );
                forms.push((format!("{width:?} build"), got));
            }
            let outputs = ["dQ_pre", "dK_pre", "dAngles", "dScale", "dGamma"];
            for (form, got) in &forms {
                for ((w, g), output) in want.iter().zip(got).zip(outputs) {
                    for (i, (a, b)) in w.iter().zip(g).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{dtype:?} ds {ds} na {na} lead {lead} spread {spread}, {form}: {output}[{i}]"
                        );
                    }
                }
            }
            cases += 1;
        }
    }
    eprintln!("m3_dqktheta: the released form's bits on {cases} cases");
}

/// The angle gradients' reverse sum takes the loads of several steps at once
/// and adds them one step at a time, in the order and with the rounding of
/// its released per-step form, so it gives that form's bits: in both index
/// widths, at windows that are and are not whole batches of steps, with one
/// sequence a row and with rows packing several, from one column to many,
/// and with raw angles on both sides of the tangent's polynomial range.
#[test]
fn m3_angle_backward_keeps_the_bits_of_its_released_form() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);
    let frozen = frozen_module(
        ctx.stream.context(),
        arch::arch0(),
        64,
        FROZEN_ANGLE_BACKWARD,
    );
    let released = frozen
        .load_function("m3_reference_angle_dt_bwd_seq")
        .unwrap();
    let mut cases = 0;
    for (b, t, nh, na, packed) in [
        (1usize, 1usize, 1usize, 1usize, false),
        (2, 15, 3, 2, false),
        (2, 16, 3, 2, true),
        (2, 17, 3, 2, true),
        (3, 100, 48, 4, true),
        (4, 513, 24, 32, false),
        (1, 1300, 48, 4, false),
        (2, 33, 5, 7, true),
    ] {
        let (bt, total) = (b * t, nh * na);
        let d_cumsum = upload_f32(&ctx, &det_rand(bt * total, 0xE101));
        let raw: Vec<f32> = det_rand(bt * na, 0xE102).iter().map(|v| v * 6.0).collect();
        let raw = upload_f32(&ctx, &raw);
        let dt: Vec<f32> = det_rand(bt * nh, 0xE103).iter().map(|v| v + 0.55).collect();
        let dt = upload_f32(&ctx, &dt);
        // Sequences of 7 steps a row, their ids as the kernel reads them.
        let ids = upload_f32(
            &ctx,
            &(0..bt)
                .map(|i| f32::from_bits(((i % t) / 7) as u32))
                .collect::<Vec<_>>(),
        );
        let seq_idx = if packed { ids.cached_ptr() } else { 0 };
        let run = |f: &CudaFunction, cfg: LaunchConfig, poison: f32| {
            let outs = [
                upload_f32(&ctx, &vec![poison; total * bt]),
                upload_f32(&ctx, &vec![poison; total * bt]),
            ];
            let dims = [b, t, nh, na].map(|v| v as i32);
            for (y_first, cfg) in y_runs(cfg) {
                let mut bld = ctx.stream.launch_builder(f);
                let ptrs = [
                    outs[0].cached_ptr(),
                    outs[1].cached_ptr(),
                    d_cumsum.cached_ptr(),
                    raw.cached_ptr(),
                    dt.cached_ptr(),
                ];
                for p in &ptrs {
                    bld.arg(p);
                }
                for d in &dims {
                    bld.arg(d);
                }
                bld.arg(&y_first);
                bld.arg(&seq_idx);
                unsafe { bld.launch(cfg) }.unwrap();
            }
            ctx.stream.synchronize().unwrap();
            outs.map(|buf| download_f32(&ctx, &buf, total * bt))
        };
        let want = run(
            &released,
            LaunchConfig {
                grid_dim: (b as u32, total.div_ceil(256) as u32, 1),
                block_dim: (total.min(256) as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            8192.0,
        );
        for width in [IndexWidth::I32, IndexWidth::I64] {
            let got = run(
                m3k.m3_angle_dt_bwd_seq.get(width),
                angle_backward_cfg(b, nh, na),
                -8192.0,
            );
            for ((w, g), output) in want
                .iter()
                .zip(&got)
                .zip(["dAngles partials", "dDt partials"])
            {
                for (i, (x, y)) in w.iter().zip(g).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "B {b} T {t} nh {nh} na {na} packed {packed} {width:?}: {output}[{i}]"
                    );
                }
            }
        }
        cases += 1;
    }
    eprintln!("m3_angle_dt_bwd_seq: the released form's bits on {cases} cases");
}

// ─── isolated production-shape bench ─────────────────────────────────────
//
// Times m3_dqkv (production tier ladder), m3_dqktheta and one
// colsum_accumulate standalone at the production shape (B=8 T=1300
// d_model 384 -> nh=48 hd=16 ds=16 CS=64) so the M3 kernel ledger stays
// measurable without a profiler on the vast box.
//
// Run: cargo test --release --features cuda --test m3_final_grads_unit_parity \
//        m3_kernels_isolated_bench -- --ignored --nocapture

// ─── m3_dqktheta output bit-hash (production shape) ─────────────────────
//
// Same role as m3_dqkv_output_hash: the bit gate for data-movement work
// on m3_dqktheta (no run-digest instrument covers it).

#[test]
#[ignore]
fn m3_dqktheta_output_hash() {
    const CB: usize = 8;
    const CT: usize = 1300;
    const CNH: usize = 48;
    const CDS: usize = 16;
    const CNA: usize = 4;

    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k =
        Mamba3Kernels::compile_with_state_cap(ctx.stream.context(), arch::arch0(), CDS.max(16))
            .unwrap();

    let n_q = CB * CT * CNH * CDS;
    let n_th = CB * CT * CNH;
    let n_ang = CB * CT * CNH * CNA;

    let q_raw = upload_f32(&ctx, &det_rand(n_q, 0xB001));
    let k_raw = upload_f32(&ctx, &det_rand(n_q, 0xB002));
    let scale_buf = upload_f32(&ctx, &det_rand(n_th, 0xB003));
    let gamma_buf = upload_f32(&ctx, &det_rand(n_th, 0xB004));
    let angle_buf = upload_f32(&ctx, &det_rand(n_ang, 0xB005));
    let dq_mid = upload_f32(&ctx, &det_rand(n_q, 0xB006));
    let dk_mid = upload_f32(&ctx, &det_rand(n_q, 0xB007));
    let dqk_buf = upload_f32(&ctx, &det_rand(n_th, 0xB008));

    let dqpre = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dkpre = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dang = GpuBuffer::zeros(&ctx.stream, n_ang).unwrap();
    let dscale = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dgamma = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    ctx.stream.synchronize().unwrap();

    launch_dqktheta(
        &ctx,
        &m3k.m3_dqktheta.i32,
        &[
            dqpre.cached_ptr(),
            dkpre.cached_ptr(),
            dang.cached_ptr(),
            dscale.cached_ptr(),
            dgamma.cached_ptr(),
            q_raw.cached_ptr(),
            k_raw.cached_ptr(),
            scale_buf.cached_ptr(),
            gamma_buf.cached_ptr(),
            angle_buf.cached_ptr(),
            dq_mid.cached_ptr(),
            dk_mid.cached_ptr(),
            dqk_buf.cached_ptr(),
        ],
        (CB * CT * CNH, CDS, CNA),
    );
    ctx.stream.synchronize().unwrap();

    let fnv = |v: &[f32]| -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for x in v {
            for b in x.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    };
    eprintln!(
        "HASH dQ_pre  {:016x}",
        fnv(&download_f32(&ctx, &dqpre, n_q))
    );
    eprintln!(
        "HASH dK_pre  {:016x}",
        fnv(&download_f32(&ctx, &dkpre, n_q))
    );
    eprintln!(
        "HASH dAngles {:016x}",
        fnv(&download_f32(&ctx, &dang, n_ang))
    );
    eprintln!(
        "HASH dScale  {:016x}",
        fnv(&download_f32(&ctx, &dscale, n_th))
    );
    eprintln!(
        "HASH dGamma  {:016x}",
        fnv(&download_f32(&ctx, &dgamma, n_th))
    );
}

// ─── m3_dqkv output bit-hash (production shape) ──────────────────────────
//
// Prints an FNV-1a hash of every m3_dqkv output at the production shape on
// deterministic inputs. There is no M3 run-digest instrument, so this is
// the bit gate for lane-redistribution work on this kernel: record the
// hashes before a change, compare after.

#[test]
#[ignore]
fn m3_dqkv_output_hash() {
    const CB: usize = 8;
    const CT: usize = 1300;
    const CNH: usize = 48;
    const CHD: usize = 16;
    const CDS: usize = 16;
    const CCS: usize = 64;
    let n_chunks = CT.div_ceil(CCS);
    let d_inner = CNH * CHD;

    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    // Production JIT injects the 16-granular state cap; the default-64
    // helper would inflate the per-thread register arrays and overstate
    // these kernels vs the real trainer.
    let m3k =
        Mamba3Kernels::compile_with_state_cap(ctx.stream.context(), arch::arch0(), CDS.max(16))
            .unwrap();

    let n_q = CB * CT * CNH * CDS;
    let n_v = CB * CT * d_inner;
    let n_th = CB * CT * CNH;

    let q_f32 = upload_f32(&ctx, &det_rand(n_q, 0xA001));
    let ks_f32 = upload_f32(&ctx, &det_rand(n_q, 0xA002));
    let v_f32 = upload_f32(&ctx, &det_rand(n_v, 0xA003));
    let do_f32 = upload_f32(&ctx, &det_rand(n_v, 0xA004));
    let dcs_buf = upload_f32(&ctx, &det_rand(CB * n_chunks * CNH * CCS, 0xA005));
    let dcs_sum_buf = upload_f32(&ctx, &det_rand(CB * n_chunks * CNH, 0xA006));
    let qk_buf = upload_f32(&ctx, &det_rand(n_th, 0xA007));
    let ssm_buf = upload_f32(&ctx, &det_rand(CB * n_chunks * CNH * CHD * CDS, 0xA008));
    let d_buf = upload_f32(&ctx, &det_rand(CNH, 0xA009));

    let dq = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dk = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dv = GpuBuffer::zeros(&ctx.stream, n_v).unwrap();
    let dadt = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dqk = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dd = GpuBuffer::zeros(&ctx.stream, CB * n_chunks * CNH).unwrap();
    ctx.stream.synchronize().unwrap();

    // The four time-major tiles carry one float of padding per row, the
    // same sizing the production launchers use.
    let legacy_floats = 2 * CCS * (CDS + 1) + 2 * CCS * (CHD + 1) + 4 * CCS + 2 * CHD * CDS;
    let mats_floats = legacy_floats + CCS * (CCS - 1) * 3 / 2 + 2 * CCS;
    let cap_floats = 99 * 1024 / 4;
    let (per_head_floats, use_mats_i): (usize, i32) = if mats_floats <= cap_floats {
        (mats_floats, 1)
    } else {
        (legacy_floats, 0)
    };
    let t_split = 16u32.min((1024 / CHD) as u32);
    let enter = GpuBuffer::zeros(&ctx.stream, CB * n_chunks * CNH * CHD * CDS).unwrap();
    ctx.stream.synchronize().unwrap();
    let cfg = LaunchConfig {
        grid_dim: (CNH as u32, CB as u32, n_chunks as u32),
        block_dim: (CHD as u32, t_split, 1),
        shared_mem_bytes: (per_head_floats * 4) as u32,
    };

    let (bi, ti, nhi, hdi, dsi, csi) = (
        CB as i32, CT as i32, CNH as i32, CHD as i32, CDS as i32, CCS as i32,
    );
    for (y_first, z_first, cfg) in yz_runs(cfg) {
        let mut bld = ctx.stream.launch_builder(&m3k.m3_dqkv.i32);
        let args = [
            dq.cached_ptr(),
            dk.cached_ptr(),
            dv.cached_ptr(),
            dadt.cached_ptr(),
            dqk.cached_ptr(),
            dd.cached_ptr(),
            q_f32.cached_ptr(),
            ks_f32.cached_ptr(),
            v_f32.cached_ptr(),
            dcs_buf.cached_ptr(),
            dcs_sum_buf.cached_ptr(),
            qk_buf.cached_ptr(),
            ssm_buf.cached_ptr(),
            do_f32.cached_ptr(),
            d_buf.cached_ptr(),
            enter.cached_ptr(),
        ];
        for a in &args {
            bld.arg(a);
        }
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&nhi);
        bld.arg(&hdi);
        bld.arg(&dsi);
        bld.arg(&csi);
        bld.arg(&use_mats_i);
        bld.arg(&y_first);
        bld.arg(&z_first);
        bld.arg(&0u64);
        unsafe { bld.launch(cfg) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();

    let fnv = |v: &[f32]| -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for x in v {
            for b in x.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    };
    eprintln!(
        "tier: t_split={t_split} use_pair_mats={use_mats_i} smem={}B",
        per_head_floats * 4
    );
    eprintln!("HASH dQ_mid  {:016x}", fnv(&download_f32(&ctx, &dq, n_q)));
    eprintln!("HASH dK_mid  {:016x}", fnv(&download_f32(&ctx, &dk, n_q)));
    eprintln!("HASH dV      {:016x}", fnv(&download_f32(&ctx, &dv, n_v)));
    eprintln!(
        "HASH dADT    {:016x}",
        fnv(&download_f32(&ctx, &dadt, n_th))
    );
    eprintln!("HASH dQK_dot {:016x}", fnv(&download_f32(&ctx, &dqk, n_th)));
    eprintln!(
        "HASH dD      {:016x}",
        fnv(&download_f32(&ctx, &dd, CB * n_chunks * CNH))
    );
}
