//! Step 8c — per-kernel parity tests for the 4 new typed M3 chunked parallel
//! forward kernels: `m3_preprocess_chunks`, `m3_chunk_state_fwd`,
//! `m3_writeback_parallel_states`, `m3_chunk_scan_fwd`. Each typed variant is
//! compared against its f32 oracle on identical random inputs.
//!
//! The 2 scan-state kernels (`m3_dA_cumsum`, `m3_state_passing_fwd`) stay
//! pure-f32 per the Tri Dao invariant (compounding O(T) prefix arithmetic
//! must not be stored in half precision) — they are NOT tested here because
//! there is no typed variant to validate.

#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;

// Small config — big enough for 2 chunks, small enough to stay fast.
const B: usize = 2;
const T: usize = 8;
const NH: usize = 2;
const HD: usize = 4;
const DS: usize = 4;
const CHUNK_SIZE: usize = 4;
const N_CHUNKS: usize = T.div_ceil(CHUNK_SIZE);
const D_INNER: usize = NH * HD;

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

fn download_typed(ctx: &GpuCtx, buf: &DtypedBuf) -> Vec<f32> {
    let mut out = vec![0f32; buf.len_elems()];
    buf.download_f32(&ctx.stream, &mut out).unwrap();
    ctx.stream.synchronize().unwrap();
    out
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

// ─── m3_preprocess_chunks ──────────────────────────────────────────────

fn check_preprocess_chunks(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);

    // Shared random inputs (same values for both f32 and typed runs).
    let k_vals = det_rand(B * T * NH * DS, 0x8C01);
    let q_vals = det_rand(B * T * NH * DS, 0x8C02);
    // DT must be positive (post-softplus), trap_sig in (0,1) (post-sigmoid).
    let dt_vals: Vec<f32> = (0..B * T * NH)
        .map(|i| 0.1 + 0.01 * (i as f32 % 5.0))
        .collect();
    let trap_vals: Vec<f32> = (0..B * T * NH)
        .map(|i| 0.3 + 0.02 * (i as f32 % 7.0))
        .collect();

    let k_f32 = upload_f32(&ctx, &k_vals);
    let q_f32 = upload_f32(&ctx, &q_vals);
    let dt_buf = upload_f32(&ctx, &dt_vals);
    let trap_buf = upload_f32(&ctx, &trap_vals);

    let ks_n = B * T * NH * DS;
    let th_n = B * T * NH;

    // f32 oracle.
    let ks_f32 = GpuBuffer::zeros(&ctx.stream, ks_n).unwrap();
    let qk_f32 = GpuBuffer::zeros(&ctx.stream, th_n).unwrap();
    let scale_f32 = GpuBuffer::zeros(&ctx.stream, th_n).unwrap();
    let gamma_f32 = GpuBuffer::zeros(&ctx.stream, th_n).unwrap();
    ctx.stream.synchronize().unwrap();

    let cfg = LaunchConfig {
        grid_dim: ((B * N_CHUNKS) as u32, NH as u32, 1),
        block_dim: (CHUNK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (bi, ti, nhi, dsi, csi) = (B as i32, T as i32, NH as i32, DS as i32, CHUNK_SIZE as i32);
    let mut bld = ctx.stream.launch_builder(&m3k.m3_preprocess_chunks);
    let (a, b, c, d, e, f, g, h) = (
        ks_f32.cached_ptr(),
        qk_f32.cached_ptr(),
        scale_f32.cached_ptr(),
        gamma_f32.cached_ptr(),
        k_f32.cached_ptr(),
        q_f32.cached_ptr(),
        dt_buf.cached_ptr(),
        trap_buf.cached_ptr(),
    );
    bld.arg(&a);
    bld.arg(&b);
    bld.arg(&c);
    bld.arg(&d);
    bld.arg(&e);
    bld.arg(&f);
    bld.arg(&g);
    bld.arg(&h);
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&nhi);
    bld.arg(&dsi);
    bld.arg(&csi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();

    let ks_ref = download_f32(&ctx, &ks_f32, ks_n);
    let qk_ref = download_f32(&ctx, &qk_f32, th_n);
    let scale_ref = download_f32(&ctx, &scale_f32, th_n);
    let gamma_ref = download_f32(&ctx, &gamma_f32, th_n);

    // Typed variant.
    let k_t = upload_typed(&ctx, &k_vals, dtype);
    let q_t = upload_typed(&ctx, &q_vals, dtype);
    let ks_t = DtypedBuf::zeros(&ctx.stream, ks_n, dtype).unwrap();
    let qk_t = GpuBuffer::zeros(&ctx.stream, th_n).unwrap();
    let scale_t = GpuBuffer::zeros(&ctx.stream, th_n).unwrap();
    let gamma_t = GpuBuffer::zeros(&ctx.stream, th_n).unwrap();
    ctx.stream.synchronize().unwrap();

    let mut bld = ctx
        .stream
        .launch_builder(m3k.m3_preprocess_chunks_typed.get(dtype));
    let (a, b, c, d, e, f, g, h) = (
        ks_t.cached_ptr(),
        qk_t.cached_ptr(),
        scale_t.cached_ptr(),
        gamma_t.cached_ptr(),
        k_t.cached_ptr(),
        q_t.cached_ptr(),
        dt_buf.cached_ptr(),
        trap_buf.cached_ptr(),
    );
    bld.arg(&a);
    bld.arg(&b);
    bld.arg(&c);
    bld.arg(&d);
    bld.arg(&e);
    bld.arg(&f);
    bld.arg(&g);
    bld.arg(&h);
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&nhi);
    bld.arg(&dsi);
    bld.arg(&csi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();

    let ks_got = download_typed(&ctx, &ks_t);
    let qk_got = download_f32(&ctx, &qk_t, th_n);
    let scale_got = download_f32(&ctx, &scale_t, th_n);
    let gamma_got = download_f32(&ctx, &gamma_t, th_n);

    eprintln!("m3_preprocess_chunks {dtype:?}:");
    let (cos_min, norm_tol) = tolerances(dtype);
    // scale/gamma derive from f32 DT/trap_sig only → bit-exact.
    assert_close("scale (f32)", &scale_ref, &scale_got, 1.0 - 1e-6, 1e-4);
    assert_close("gamma (f32)", &gamma_ref, &gamma_got, 1.0 - 1e-6, 1e-4);
    // qk_dot = Σ(Q*K) * gamma, reduced from typed Q/K → dtype-bounded drift.
    assert_close("qk_dot", &qk_ref, &qk_got, cos_min, norm_tol);
    // K_scaled goes through bf16/f16 round-trip.
    assert_close("K_scaled", &ks_ref, &ks_got, cos_min, norm_tol);
}

#[test]
fn m3_preprocess_chunks_bf16() {
    check_preprocess_chunks(WeightDtype::Bf16);
}
#[test]
fn m3_preprocess_chunks_f16() {
    check_preprocess_chunks(WeightDtype::F16);
}

// ─── m3_chunk_state_fwd ────────────────────────────────────────────────

fn check_chunk_state_fwd(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);

    let x_vals = det_rand(B * T * D_INNER, 0x8C11);
    let ks_vals = det_rand(B * T * NH * DS, 0x8C12);
    // dA_cumsum values — negative (decay)
    let da_vals: Vec<f32> = (0..B * N_CHUNKS * NH * CHUNK_SIZE)
        .map(|i| -0.02 * (i as f32 % 4.0 + 1.0))
        .collect();

    let x_f32 = upload_f32(&ctx, &x_vals);
    let ks_f32_buf = upload_f32(&ctx, &ks_vals);
    let da_buf = upload_f32(&ctx, &da_vals);

    let states_n = B * N_CHUNKS * NH * HD * DS;

    // f32 oracle.
    let states_f32 = GpuBuffer::zeros(&ctx.stream, states_n).unwrap();
    ctx.stream.synchronize().unwrap();
    let cfg = LaunchConfig {
        grid_dim: ((B * N_CHUNKS) as u32, NH as u32, 1),
        block_dim: (HD as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (bi, ti, nhi, hdi, dsi, csi) = (
        B as i32,
        T as i32,
        NH as i32,
        HD as i32,
        DS as i32,
        CHUNK_SIZE as i32,
    );
    let mut bld = ctx.stream.launch_builder(&m3k.m3_chunk_state_fwd);
    let (a, b, c, d) = (
        states_f32.cached_ptr(),
        x_f32.cached_ptr(),
        ks_f32_buf.cached_ptr(),
        da_buf.cached_ptr(),
    );
    bld.arg(&a);
    bld.arg(&b);
    bld.arg(&c);
    bld.arg(&d);
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&nhi);
    bld.arg(&hdi);
    bld.arg(&dsi);
    bld.arg(&csi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let ref_states = download_f32(&ctx, &states_f32, states_n);

    // Typed variant.
    let x_t = upload_typed(&ctx, &x_vals, dtype);
    let ks_t = upload_typed(&ctx, &ks_vals, dtype);
    let states_t = GpuBuffer::zeros(&ctx.stream, states_n).unwrap();
    ctx.stream.synchronize().unwrap();
    let mut bld = ctx
        .stream
        .launch_builder(m3k.m3_chunk_state_fwd_typed.get(dtype));
    let (a, b, c, d) = (
        states_t.cached_ptr(),
        x_t.cached_ptr(),
        ks_t.cached_ptr(),
        da_buf.cached_ptr(),
    );
    bld.arg(&a);
    bld.arg(&b);
    bld.arg(&c);
    bld.arg(&d);
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&nhi);
    bld.arg(&hdi);
    bld.arg(&dsi);
    bld.arg(&csi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let got_states = download_f32(&ctx, &states_t, states_n);

    eprintln!("m3_chunk_state_fwd {dtype:?}:");
    let (cos_min, norm_tol) = tolerances(dtype);
    assert_close("states (f32)", &ref_states, &got_states, cos_min, norm_tol);
}

#[test]
fn m3_chunk_state_fwd_bf16() {
    check_chunk_state_fwd(WeightDtype::Bf16);
}
#[test]
fn m3_chunk_state_fwd_f16() {
    check_chunk_state_fwd(WeightDtype::F16);
}

// ─── m3_writeback_parallel_states ──────────────────────────────────────

fn check_writeback_parallel_states(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);

    let final_vals = det_rand(B * NH * HD * DS, 0x8C21);
    let k_vals = det_rand(B * T * NH * DS, 0x8C22);
    let x_vals = det_rand(B * T * D_INNER, 0x8C23);

    let final_f32 = upload_f32(&ctx, &final_vals);
    let k_f32 = upload_f32(&ctx, &k_vals);
    let x_f32 = upload_f32(&ctx, &x_vals);

    let ssm_n = B * NH * HD * DS;
    let k_n = B * NH * DS;
    let v_n = B * NH * HD;

    // f32 oracle.
    let ssm_ref = GpuBuffer::zeros(&ctx.stream, ssm_n).unwrap();
    let k_ref_out = GpuBuffer::zeros(&ctx.stream, k_n).unwrap();
    let v_ref_out = GpuBuffer::zeros(&ctx.stream, v_n).unwrap();
    ctx.stream.synchronize().unwrap();
    let cfg = LaunchConfig {
        grid_dim: (B as u32, NH as u32, 1),
        block_dim: (HD.max(DS) as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (bi, ti, nhi, hdi, dsi) = (B as i32, T as i32, NH as i32, HD as i32, DS as i32);
    let mut bld = ctx.stream.launch_builder(&m3k.m3_writeback_parallel_states);
    let (a, b, c, d, e, f) = (
        ssm_ref.cached_ptr(),
        k_ref_out.cached_ptr(),
        v_ref_out.cached_ptr(),
        final_f32.cached_ptr(),
        k_f32.cached_ptr(),
        x_f32.cached_ptr(),
    );
    bld.arg(&a);
    bld.arg(&b);
    bld.arg(&c);
    bld.arg(&d);
    bld.arg(&e);
    bld.arg(&f);
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&nhi);
    bld.arg(&hdi);
    bld.arg(&dsi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let ssm_ref_v = download_f32(&ctx, &ssm_ref, ssm_n);
    let k_ref_v = download_f32(&ctx, &k_ref_out, k_n);
    let v_ref_v = download_f32(&ctx, &v_ref_out, v_n);

    // Typed variant: typed k_flat/x_flat.
    let k_t = upload_typed(&ctx, &k_vals, dtype);
    let x_t = upload_typed(&ctx, &x_vals, dtype);
    let ssm_got = GpuBuffer::zeros(&ctx.stream, ssm_n).unwrap();
    let k_got_out = GpuBuffer::zeros(&ctx.stream, k_n).unwrap();
    let v_got_out = GpuBuffer::zeros(&ctx.stream, v_n).unwrap();
    ctx.stream.synchronize().unwrap();
    let mut bld = ctx
        .stream
        .launch_builder(m3k.m3_writeback_parallel_states_typed.get(dtype));
    let (a, b, c, d, e, f) = (
        ssm_got.cached_ptr(),
        k_got_out.cached_ptr(),
        v_got_out.cached_ptr(),
        final_f32.cached_ptr(),
        k_t.cached_ptr(),
        x_t.cached_ptr(),
    );
    bld.arg(&a);
    bld.arg(&b);
    bld.arg(&c);
    bld.arg(&d);
    bld.arg(&e);
    bld.arg(&f);
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&nhi);
    bld.arg(&hdi);
    bld.arg(&dsi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let ssm_got_v = download_f32(&ctx, &ssm_got, ssm_n);
    let k_got_v = download_f32(&ctx, &k_got_out, k_n);
    let v_got_v = download_f32(&ctx, &v_got_out, v_n);

    eprintln!("m3_writeback_parallel_states {dtype:?}:");
    let (cos_min, norm_tol) = tolerances(dtype);
    // ssm_state is copied f32→f32 (no dtype conversion) → bit-exact.
    assert_close("ssm_state", &ssm_ref_v, &ssm_got_v, 1.0 - 1e-6, 1e-4);
    assert_close("k_state", &k_ref_v, &k_got_v, cos_min, norm_tol);
    assert_close("v_state", &v_ref_v, &v_got_v, cos_min, norm_tol);
}

#[test]
fn m3_writeback_parallel_states_bf16() {
    check_writeback_parallel_states(WeightDtype::Bf16);
}
#[test]
fn m3_writeback_parallel_states_f16() {
    check_writeback_parallel_states(WeightDtype::F16);
}

// ─── m3_chunk_scan_fwd ─────────────────────────────────────────────────

fn check_chunk_scan_fwd(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);

    let x_vals = det_rand(B * T * D_INNER, 0x8C31);
    let q_vals = det_rand(B * T * NH * DS, 0x8C32);
    let ks_vals = det_rand(B * T * NH * DS, 0x8C33);
    let qk_vals = det_rand(B * T * NH, 0x8C34);
    let da_vals: Vec<f32> = (0..B * N_CHUNKS * NH * CHUNK_SIZE)
        .map(|i| -0.02 * (i as f32 % 4.0 + 1.0))
        .collect();
    let prev_vals = det_rand(B * N_CHUNKS * NH * HD * DS, 0x8C35);
    let d_vals: Vec<f32> = (0..NH).map(|i| 0.5 + 0.1 * (i as f32)).collect();

    let x_f32 = upload_f32(&ctx, &x_vals);
    let q_f32 = upload_f32(&ctx, &q_vals);
    let ks_f32_buf = upload_f32(&ctx, &ks_vals);
    let qk_buf = upload_f32(&ctx, &qk_vals);
    let da_buf = upload_f32(&ctx, &da_vals);
    let prev_buf = upload_f32(&ctx, &prev_vals);
    let d_buf = upload_f32(&ctx, &d_vals);

    let y_n = B * T * D_INNER;

    // f32 oracle.
    let y_f32 = GpuBuffer::zeros(&ctx.stream, y_n).unwrap();
    ctx.stream.synchronize().unwrap();
    let cfg = LaunchConfig {
        grid_dim: ((B * N_CHUNKS) as u32, NH as u32, 1),
        block_dim: (HD as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (bi, ti, nhi, hdi, dsi, csi) = (
        B as i32,
        T as i32,
        NH as i32,
        HD as i32,
        DS as i32,
        CHUNK_SIZE as i32,
    );
    let mut bld = ctx.stream.launch_builder(&m3k.m3_chunk_scan_fwd);
    let (a, b, c, d, e, f, g, h) = (
        y_f32.cached_ptr(),
        x_f32.cached_ptr(),
        q_f32.cached_ptr(),
        ks_f32_buf.cached_ptr(),
        qk_buf.cached_ptr(),
        da_buf.cached_ptr(),
        prev_buf.cached_ptr(),
        d_buf.cached_ptr(),
    );
    bld.arg(&a);
    bld.arg(&b);
    bld.arg(&c);
    bld.arg(&d);
    bld.arg(&e);
    bld.arg(&f);
    bld.arg(&g);
    bld.arg(&h);
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&nhi);
    bld.arg(&hdi);
    bld.arg(&dsi);
    bld.arg(&csi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let y_ref = download_f32(&ctx, &y_f32, y_n);

    // Typed variant.
    let x_t = upload_typed(&ctx, &x_vals, dtype);
    let q_t = upload_typed(&ctx, &q_vals, dtype);
    let ks_t = upload_typed(&ctx, &ks_vals, dtype);
    let y_t = DtypedBuf::zeros(&ctx.stream, y_n, dtype).unwrap();
    ctx.stream.synchronize().unwrap();
    let mut bld = ctx
        .stream
        .launch_builder(m3k.m3_chunk_scan_fwd_typed.get(dtype));
    let (a, b, c, d, e, f, g, h) = (
        y_t.cached_ptr(),
        x_t.cached_ptr(),
        q_t.cached_ptr(),
        ks_t.cached_ptr(),
        qk_buf.cached_ptr(),
        da_buf.cached_ptr(),
        prev_buf.cached_ptr(),
        d_buf.cached_ptr(),
    );
    bld.arg(&a);
    bld.arg(&b);
    bld.arg(&c);
    bld.arg(&d);
    bld.arg(&e);
    bld.arg(&f);
    bld.arg(&g);
    bld.arg(&h);
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&nhi);
    bld.arg(&hdi);
    bld.arg(&dsi);
    bld.arg(&csi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let y_got = download_typed(&ctx, &y_t);

    eprintln!("m3_chunk_scan_fwd {dtype:?}:");
    let (cos_min, norm_tol) = tolerances(dtype);
    assert_close("y_out", &y_ref, &y_got, cos_min, norm_tol);
}

#[test]
fn m3_chunk_scan_fwd_bf16() {
    check_chunk_scan_fwd(WeightDtype::Bf16);
}
#[test]
fn m3_chunk_scan_fwd_f16() {
    check_chunk_scan_fwd(WeightDtype::F16);
}
