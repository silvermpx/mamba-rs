//! Step 9b — per-kernel parity tests for the 2 HIGHEST-RISK typed M3
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

use cudarc::driver::{LaunchConfig, PushKernelArg};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;

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

fn make_m3k(ctx: &GpuCtx) -> Mamba3Kernels {
    Mamba3Kernels::compile(ctx.stream.context(), "sm_89").unwrap()
}

fn tolerances(dtype: WeightDtype) -> (f32, f32) {
    match dtype {
        WeightDtype::Bf16 => (0.99, 0.05),
        WeightDtype::F16 => (0.999, 0.02),
        WeightDtype::F32 => unreachable!(),
    }
}

// ─── m3_dqkv ───────────────────────────────────────────────────────────
//
// 6 outputs: dQ_mid, dK_mid, dV, dADT, dQK_dot_out, dD_out.
// dV is written (non-atomic); dD_out uses atomicAdd across B; others stored.

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
                dcs_sum_vals[(b * N_CHUNKS + c) * NH + h] =
                    dcs_vals[base + chunk_len - 1];
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
    let n_d = NH;

    // f32 oracle.
    let dq_ref = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dk_ref = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dv_ref = GpuBuffer::zeros(&ctx.stream, n_v).unwrap();
    let dadt_ref = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dqk_ref = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dd_ref = GpuBuffer::zeros(&ctx.stream, n_d).unwrap();
    ctx.stream.synchronize().unwrap();

    // Shared memory size: q_sm + k_sm + v_sm + do_sm + da_cs_sm + qk_sm + ssm_sm.
    let smem_floats = CS * DS * 2 + CS * HD * 2 + CS * 2 + HD * DS;
    let smem_bytes = (smem_floats * 4) as u32;
    let cfg = LaunchConfig {
        grid_dim: (NH as u32, B as u32, 1),
        block_dim: (HD as u32, 1, 1),
        shared_mem_bytes: smem_bytes,
    };

    let (bi, ti, nhi, hdi, dsi, csi) = (
        B as i32,
        T as i32,
        NH as i32,
        HD as i32,
        DS as i32,
        CS as i32,
    );

    let mut bld = ctx.stream.launch_builder(&m3k.m3_dqkv);
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
    unsafe { bld.launch(cfg) }.unwrap();
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

    let mut bld = ctx.stream.launch_builder(m3k.m3_dqkv_typed.get(dtype));
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
    unsafe { bld.launch(cfg) }.unwrap();
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
// 7 outputs: dQ_pre, dK_pre, dAngles_cumsum, dScale, dGamma, dQ_bias, dK_bias.

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

    let cfg = LaunchConfig {
        grid_dim: ((B * N_CHUNKS) as u32, NH as u32, 1),
        block_dim: (CS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (bi, ti, nhi, dsi, nai, csi) = (
        B as i32,
        T as i32,
        NH as i32,
        DS as i32,
        N_ANGLES as i32,
        CS as i32,
    );

    let mut bld = ctx.stream.launch_builder(&m3k.m3_dqktheta);
    let args = [
        dqpre_ref.cached_ptr(),
        dkpre_ref.cached_ptr(),
        dang_ref.cached_ptr(),
        dscale_ref.cached_ptr(),
        dgamma_ref.cached_ptr(),
        dqb_ref.cached_ptr(),
        dkb_ref.cached_ptr(),
        q_raw_f32.cached_ptr(),
        k_raw_f32.cached_ptr(),
        scale_buf.cached_ptr(),
        gamma_buf.cached_ptr(),
        angle_buf.cached_ptr(),
        dq_mid_buf.cached_ptr(),
        dk_mid_buf.cached_ptr(),
        dqk_buf.cached_ptr(),
    ];
    for a in &args {
        bld.arg(a);
    }
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&nhi);
    bld.arg(&dsi);
    bld.arg(&nai);
    bld.arg(&csi);
    unsafe { bld.launch(cfg) }.unwrap();
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

    let mut bld = ctx.stream.launch_builder(m3k.m3_dqktheta_typed.get(dtype));
    let args = [
        dqpre_t.cached_ptr(),
        dkpre_t.cached_ptr(),
        dang_t.cached_ptr(),
        dscale_t.cached_ptr(),
        dgamma_t.cached_ptr(),
        dqb_t.cached_ptr(),
        dkb_t.cached_ptr(),
        q_raw_t.cached_ptr(),
        k_raw_t.cached_ptr(),
        scale_buf.cached_ptr(),
        gamma_buf.cached_ptr(),
        angle_buf.cached_ptr(),
        dq_mid_buf.cached_ptr(),
        dk_mid_buf.cached_ptr(),
        dqk_buf.cached_ptr(),
    ];
    for a in &args {
        bld.arg(a);
    }
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&nhi);
    bld.arg(&dsi);
    bld.arg(&nai);
    bld.arg(&csi);
    unsafe { bld.launch(cfg) }.unwrap();
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
