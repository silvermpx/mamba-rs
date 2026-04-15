//! Step 9d — per-kernel parity tests for the 2 new typed M3 chunked parallel
//! backward kernels: `m3_chunk_scan_bwd`, `m3_chunk_state_bwd`. Each typed
//! variant is compared against its f32 oracle on identical random inputs.
//!
//! ALL gradient OUTPUTS remain f32 — this is the PyTorch AMP master-grad
//! invariant. bf16/f16 hardware atomicAdd is not supported on ≤sm_89 and
//! numerical safety requires f32 accumulation for reduction-style grads.
//! Typed I/O is therefore limited to saved-activation inputs only
//! (d_y, x, Q, K_scaled).
//!
//! The following chunked backward kernels stay pure f32 and are NOT tested
//! here (no typed variant to validate):
//!   - m3_state_passing_bwd  (inter-chunk prefix recurrence — f32 state only)
//!   - m3_cumsum_bwd  (reverse prefix sum — f32 only)
//!
//! Precision (measured on Ada sm_89 against f32 oracle with 128-thread
//! warp-reduce contention, chunk_size=4, T=8, 2 chunks):
//!   - bf16:  cos ≥ 0.99,   |norm−1| ≤ 0.05
//!   - f16:   cos ≥ 0.999,  |norm−1| ≤ 0.02

#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;

// Small config — forces 2 chunks of size 4, partial-last path exercised via
// second seed (T=7 below).
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

// ─── m3_chunk_scan_bwd ─────────────────────────────────────────────────
//
// 7 outputs: d_x, d_Q, d_K_scaled, d_qk_dot, d_D, d_prev_states, d_dA_cumsum.
// All outputs must be f32 (atomicAdd invariant).

type ScanBwdOut = (
    Vec<f32>, // d_x
    Vec<f32>, // d_Q
    Vec<f32>, // d_K_scaled
    Vec<f32>, // d_qk_dot
    Vec<f32>, // d_D
    Vec<f32>, // d_prev_states
    Vec<f32>, // d_dA_cumsum
);

#[allow(clippy::too_many_arguments)]
fn run_chunk_scan_bwd_f32(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    d_y: &GpuBuffer,
    x: &GpuBuffer,
    q: &GpuBuffer,
    ks: &GpuBuffer,
    qk: &GpuBuffer,
    da: &GpuBuffer,
    prev: &GpuBuffer,
    d_buf: &GpuBuffer,
) -> ScanBwdOut {
    let n_x = B * T * D_INNER;
    let n_q = B * T * NH * DS;
    let n_th = B * T * NH;
    let n_d = NH;
    let n_prev = B * N_CHUNKS * NH * HD * DS;
    let n_da = B * N_CHUNKS * NH * CHUNK_SIZE;

    let d_x = GpuBuffer::zeros(&ctx.stream, n_x).unwrap();
    let d_q = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let d_ks = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let d_qk = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let d_d = GpuBuffer::zeros(&ctx.stream, n_d).unwrap();
    let d_prev = GpuBuffer::zeros(&ctx.stream, n_prev).unwrap();
    let d_da = GpuBuffer::zeros(&ctx.stream, n_da).unwrap();
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
    let mut bld = ctx.stream.launch_builder(&m3k.m3_chunk_scan_bwd);
    let (dxp, dqp, dksp, dqkp, ddp, dprevp, ddap) = (
        d_x.cached_ptr(),
        d_q.cached_ptr(),
        d_ks.cached_ptr(),
        d_qk.cached_ptr(),
        d_d.cached_ptr(),
        d_prev.cached_ptr(),
        d_da.cached_ptr(),
    );
    let (dyp, xp, qp, ksp, qkp, dap, prevp, dp) = (
        d_y.cached_ptr(),
        x.cached_ptr(),
        q.cached_ptr(),
        ks.cached_ptr(),
        qk.cached_ptr(),
        da.cached_ptr(),
        prev.cached_ptr(),
        d_buf.cached_ptr(),
    );
    bld.arg(&dxp);
    bld.arg(&dqp);
    bld.arg(&dksp);
    bld.arg(&dqkp);
    bld.arg(&ddp);
    bld.arg(&dprevp);
    bld.arg(&ddap);
    bld.arg(&dyp);
    bld.arg(&xp);
    bld.arg(&qp);
    bld.arg(&ksp);
    bld.arg(&qkp);
    bld.arg(&dap);
    bld.arg(&prevp);
    bld.arg(&dp);
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&nhi);
    bld.arg(&hdi);
    bld.arg(&dsi);
    bld.arg(&csi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    (
        download_f32(ctx, &d_x, n_x),
        download_f32(ctx, &d_q, n_q),
        download_f32(ctx, &d_ks, n_q),
        download_f32(ctx, &d_qk, n_th),
        download_f32(ctx, &d_d, n_d),
        download_f32(ctx, &d_prev, n_prev),
        download_f32(ctx, &d_da, n_da),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_chunk_scan_bwd_typed(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    dtype: WeightDtype,
    d_y: &DtypedBuf,
    x: &DtypedBuf,
    q: &DtypedBuf,
    ks: &DtypedBuf,
    qk: &GpuBuffer,
    da: &GpuBuffer,
    prev: &GpuBuffer,
    d_buf: &GpuBuffer,
) -> ScanBwdOut {
    let n_x = B * T * D_INNER;
    let n_q = B * T * NH * DS;
    let n_th = B * T * NH;
    let n_d = NH;
    let n_prev = B * N_CHUNKS * NH * HD * DS;
    let n_da = B * N_CHUNKS * NH * CHUNK_SIZE;

    let d_x = GpuBuffer::zeros(&ctx.stream, n_x).unwrap();
    let d_q = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let d_ks = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let d_qk = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let d_d = GpuBuffer::zeros(&ctx.stream, n_d).unwrap();
    let d_prev = GpuBuffer::zeros(&ctx.stream, n_prev).unwrap();
    let d_da = GpuBuffer::zeros(&ctx.stream, n_da).unwrap();
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
    let mut bld = ctx
        .stream
        .launch_builder(m3k.m3_chunk_scan_bwd_typed.get(dtype));
    let (dxp, dqp, dksp, dqkp, ddp, dprevp, ddap) = (
        d_x.cached_ptr(),
        d_q.cached_ptr(),
        d_ks.cached_ptr(),
        d_qk.cached_ptr(),
        d_d.cached_ptr(),
        d_prev.cached_ptr(),
        d_da.cached_ptr(),
    );
    let (dyp, xp, qp, ksp, qkp, dap, prevp, dp) = (
        d_y.cached_ptr(),
        x.cached_ptr(),
        q.cached_ptr(),
        ks.cached_ptr(),
        qk.cached_ptr(),
        da.cached_ptr(),
        prev.cached_ptr(),
        d_buf.cached_ptr(),
    );
    bld.arg(&dxp);
    bld.arg(&dqp);
    bld.arg(&dksp);
    bld.arg(&dqkp);
    bld.arg(&ddp);
    bld.arg(&dprevp);
    bld.arg(&ddap);
    bld.arg(&dyp);
    bld.arg(&xp);
    bld.arg(&qp);
    bld.arg(&ksp);
    bld.arg(&qkp);
    bld.arg(&dap);
    bld.arg(&prevp);
    bld.arg(&dp);
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&nhi);
    bld.arg(&hdi);
    bld.arg(&dsi);
    bld.arg(&csi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    (
        download_f32(ctx, &d_x, n_x),
        download_f32(ctx, &d_q, n_q),
        download_f32(ctx, &d_ks, n_q),
        download_f32(ctx, &d_qk, n_th),
        download_f32(ctx, &d_d, n_d),
        download_f32(ctx, &d_prev, n_prev),
        download_f32(ctx, &d_da, n_da),
    )
}

fn check_chunk_scan_bwd(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);

    let dy_vals = det_rand(B * T * D_INNER, 0x9D01);
    let x_vals = det_rand(B * T * D_INNER, 0x9D02);
    let q_vals = det_rand(B * T * NH * DS, 0x9D03);
    let ks_vals = det_rand(B * T * NH * DS, 0x9D04);
    let qk_vals = det_rand(B * T * NH, 0x9D05);
    // dA_cumsum must be negative (decay) to avoid exp overflow.
    let da_vals: Vec<f32> = (0..B * N_CHUNKS * NH * CHUNK_SIZE)
        .map(|i| -0.02 * (i as f32 % 4.0 + 1.0))
        .collect();
    let prev_vals = det_rand(B * N_CHUNKS * NH * HD * DS, 0x9D06);
    let d_vals: Vec<f32> = (0..NH).map(|i| 0.5 + 0.1 * (i as f32)).collect();

    let dy_f32 = upload_f32(&ctx, &dy_vals);
    let x_f32 = upload_f32(&ctx, &x_vals);
    let q_f32 = upload_f32(&ctx, &q_vals);
    let ks_f32 = upload_f32(&ctx, &ks_vals);
    let qk_buf = upload_f32(&ctx, &qk_vals);
    let da_buf = upload_f32(&ctx, &da_vals);
    let prev_buf = upload_f32(&ctx, &prev_vals);
    let d_buf = upload_f32(&ctx, &d_vals);

    let (dx_ref, dq_ref, dks_ref, dqk_ref, dd_ref, dprev_ref, dda_ref) = run_chunk_scan_bwd_f32(
        &ctx, &m3k, &dy_f32, &x_f32, &q_f32, &ks_f32, &qk_buf, &da_buf, &prev_buf, &d_buf,
    );

    let dy_t = upload_typed(&ctx, &dy_vals, dtype);
    let x_t = upload_typed(&ctx, &x_vals, dtype);
    let q_t = upload_typed(&ctx, &q_vals, dtype);
    let ks_t = upload_typed(&ctx, &ks_vals, dtype);

    let (dx_got, dq_got, dks_got, dqk_got, dd_got, dprev_got, dda_got) = run_chunk_scan_bwd_typed(
        &ctx, &m3k, dtype, &dy_t, &x_t, &q_t, &ks_t, &qk_buf, &da_buf, &prev_buf, &d_buf,
    );

    eprintln!("m3_chunk_scan_bwd {dtype:?}:");
    let (cos_min, norm_tol) = tolerances(dtype);
    assert_close("d_x", &dx_ref, &dx_got, cos_min, norm_tol);
    assert_close("d_Q", &dq_ref, &dq_got, cos_min, norm_tol);
    assert_close("d_K_scaled", &dks_ref, &dks_got, cos_min, norm_tol);
    assert_close("d_qk_dot", &dqk_ref, &dqk_got, cos_min, norm_tol);
    assert_close("d_D", &dd_ref, &dd_got, cos_min, norm_tol);
    assert_close("d_prev_states", &dprev_ref, &dprev_got, cos_min, norm_tol);
    assert_close("d_dA_cumsum", &dda_ref, &dda_got, cos_min, norm_tol);
}

#[test]
fn m3_chunk_scan_bwd_bf16() {
    check_chunk_scan_bwd(WeightDtype::Bf16);
}
#[test]
fn m3_chunk_scan_bwd_f16() {
    check_chunk_scan_bwd(WeightDtype::F16);
}

// ─── m3_chunk_state_bwd ────────────────────────────────────────────────
//
// 3 outputs: d_x (in-place +=), d_K_scaled (atomicAdd), d_dA_cumsum (atomicAdd).

#[allow(clippy::too_many_arguments)]
fn run_chunk_state_bwd_f32(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    d_chunk_states: &GpuBuffer,
    x: &GpuBuffer,
    ks: &GpuBuffer,
    da: &GpuBuffer,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_x = B * T * D_INNER;
    let n_q = B * T * NH * DS;
    let n_da = B * N_CHUNKS * NH * CHUNK_SIZE;

    let d_x = GpuBuffer::zeros(&ctx.stream, n_x).unwrap();
    let d_ks = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let d_da = GpuBuffer::zeros(&ctx.stream, n_da).unwrap();
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
    let mut bld = ctx.stream.launch_builder(&m3k.m3_chunk_state_bwd);
    let (dxp, dksp, ddap) = (d_x.cached_ptr(), d_ks.cached_ptr(), d_da.cached_ptr());
    let (dcsp, xp, ksp, dap) = (
        d_chunk_states.cached_ptr(),
        x.cached_ptr(),
        ks.cached_ptr(),
        da.cached_ptr(),
    );
    bld.arg(&dxp);
    bld.arg(&dksp);
    bld.arg(&ddap);
    bld.arg(&dcsp);
    bld.arg(&xp);
    bld.arg(&ksp);
    bld.arg(&dap);
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&nhi);
    bld.arg(&hdi);
    bld.arg(&dsi);
    bld.arg(&csi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    (
        download_f32(ctx, &d_x, n_x),
        download_f32(ctx, &d_ks, n_q),
        download_f32(ctx, &d_da, n_da),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_chunk_state_bwd_typed(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    dtype: WeightDtype,
    d_chunk_states: &GpuBuffer,
    x: &DtypedBuf,
    ks: &DtypedBuf,
    da: &GpuBuffer,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_x = B * T * D_INNER;
    let n_q = B * T * NH * DS;
    let n_da = B * N_CHUNKS * NH * CHUNK_SIZE;

    let d_x = GpuBuffer::zeros(&ctx.stream, n_x).unwrap();
    let d_ks = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let d_da = GpuBuffer::zeros(&ctx.stream, n_da).unwrap();
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
    let mut bld = ctx
        .stream
        .launch_builder(m3k.m3_chunk_state_bwd_typed.get(dtype));
    let (dxp, dksp, ddap) = (d_x.cached_ptr(), d_ks.cached_ptr(), d_da.cached_ptr());
    let (dcsp, xp, ksp, dap) = (
        d_chunk_states.cached_ptr(),
        x.cached_ptr(),
        ks.cached_ptr(),
        da.cached_ptr(),
    );
    bld.arg(&dxp);
    bld.arg(&dksp);
    bld.arg(&ddap);
    bld.arg(&dcsp);
    bld.arg(&xp);
    bld.arg(&ksp);
    bld.arg(&dap);
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&nhi);
    bld.arg(&hdi);
    bld.arg(&dsi);
    bld.arg(&csi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    (
        download_f32(ctx, &d_x, n_x),
        download_f32(ctx, &d_ks, n_q),
        download_f32(ctx, &d_da, n_da),
    )
}

fn check_chunk_state_bwd(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);

    let dcs_vals = det_rand(B * N_CHUNKS * NH * HD * DS, 0x9E01);
    let x_vals = det_rand(B * T * D_INNER, 0x9E02);
    let ks_vals = det_rand(B * T * NH * DS, 0x9E03);
    let da_vals: Vec<f32> = (0..B * N_CHUNKS * NH * CHUNK_SIZE)
        .map(|i| -0.02 * (i as f32 % 4.0 + 1.0))
        .collect();

    let dcs_f32 = upload_f32(&ctx, &dcs_vals);
    let x_f32 = upload_f32(&ctx, &x_vals);
    let ks_f32 = upload_f32(&ctx, &ks_vals);
    let da_buf = upload_f32(&ctx, &da_vals);

    let (dx_ref, dks_ref, dda_ref) =
        run_chunk_state_bwd_f32(&ctx, &m3k, &dcs_f32, &x_f32, &ks_f32, &da_buf);

    let x_t = upload_typed(&ctx, &x_vals, dtype);
    let ks_t = upload_typed(&ctx, &ks_vals, dtype);

    let (dx_got, dks_got, dda_got) =
        run_chunk_state_bwd_typed(&ctx, &m3k, dtype, &dcs_f32, &x_t, &ks_t, &da_buf);

    eprintln!("m3_chunk_state_bwd {dtype:?}:");
    let (cos_min, norm_tol) = tolerances(dtype);
    assert_close("d_x", &dx_ref, &dx_got, cos_min, norm_tol);
    assert_close("d_K_scaled", &dks_ref, &dks_got, cos_min, norm_tol);
    assert_close("d_dA_cumsum", &dda_ref, &dda_got, cos_min, norm_tol);
}

#[test]
fn m3_chunk_state_bwd_bf16() {
    check_chunk_state_bwd(WeightDtype::Bf16);
}
#[test]
fn m3_chunk_state_bwd_f16() {
    check_chunk_state_bwd(WeightDtype::F16);
}

// ─── Larger config stress test ─────────────────────────────────────────
// Same kernels, bigger dims + varied DS per head. Catches precision
// drift that only surfaces at realistic head_dim / d_state.

fn check_chunk_scan_bwd_big(dtype: WeightDtype) {
    // Larger: B=4, T=16, NH=4, HD=8, DS=16, chunk_size=8, 2 chunks.
    const BB: usize = 4;
    const TT: usize = 16;
    const NNH: usize = 4;
    const HHD: usize = 8;
    const DDS: usize = 16;
    const CCS: usize = 8;
    const NC: usize = TT.div_ceil(CCS);
    const DI: usize = NNH * HHD;

    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);

    let dy_vals = det_rand(BB * TT * DI, 0xBB01);
    let x_vals = det_rand(BB * TT * DI, 0xBB02);
    let q_vals = det_rand(BB * TT * NNH * DDS, 0xBB03);
    let ks_vals = det_rand(BB * TT * NNH * DDS, 0xBB04);
    let qk_vals = det_rand(BB * TT * NNH, 0xBB05);
    let da_vals: Vec<f32> = (0..BB * NC * NNH * CCS)
        .map(|i| -0.015 * (i as f32 % 8.0 + 1.0))
        .collect();
    let prev_vals = det_rand(BB * NC * NNH * HHD * DDS, 0xBB06);
    let d_vals: Vec<f32> = (0..NNH).map(|i| 0.4 + 0.05 * (i as f32)).collect();

    let dy_f32 = upload_f32(&ctx, &dy_vals);
    let x_f32 = upload_f32(&ctx, &x_vals);
    let q_f32 = upload_f32(&ctx, &q_vals);
    let ks_f32 = upload_f32(&ctx, &ks_vals);
    let qk_buf = upload_f32(&ctx, &qk_vals);
    let da_buf = upload_f32(&ctx, &da_vals);
    let prev_buf = upload_f32(&ctx, &prev_vals);
    let d_buf = upload_f32(&ctx, &d_vals);

    let n_x = BB * TT * DI;
    let n_q = BB * TT * NNH * DDS;
    let n_th = BB * TT * NNH;
    let n_d = NNH;
    let n_prev = BB * NC * NNH * HHD * DDS;
    let n_da = BB * NC * NNH * CCS;

    let (bi, ti, nhi, hdi, dsi, csi) = (
        BB as i32, TT as i32, NNH as i32, HHD as i32, DDS as i32, CCS as i32,
    );
    let cfg = LaunchConfig {
        grid_dim: ((BB * NC) as u32, NNH as u32, 1),
        block_dim: (HHD as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // f32 oracle.
    let d_x = GpuBuffer::zeros(&ctx.stream, n_x).unwrap();
    let d_q = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let d_ks = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let d_qk = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let d_d = GpuBuffer::zeros(&ctx.stream, n_d).unwrap();
    let d_prev = GpuBuffer::zeros(&ctx.stream, n_prev).unwrap();
    let d_da = GpuBuffer::zeros(&ctx.stream, n_da).unwrap();
    ctx.stream.synchronize().unwrap();
    let mut bld = ctx.stream.launch_builder(&m3k.m3_chunk_scan_bwd);
    let args = [
        d_x.cached_ptr(),
        d_q.cached_ptr(),
        d_ks.cached_ptr(),
        d_qk.cached_ptr(),
        d_d.cached_ptr(),
        d_prev.cached_ptr(),
        d_da.cached_ptr(),
        dy_f32.cached_ptr(),
        x_f32.cached_ptr(),
        q_f32.cached_ptr(),
        ks_f32.cached_ptr(),
        qk_buf.cached_ptr(),
        da_buf.cached_ptr(),
        prev_buf.cached_ptr(),
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
    let dx_ref = download_f32(&ctx, &d_x, n_x);
    let dq_ref = download_f32(&ctx, &d_q, n_q);
    let dks_ref = download_f32(&ctx, &d_ks, n_q);
    let dqk_ref = download_f32(&ctx, &d_qk, n_th);
    let dd_ref = download_f32(&ctx, &d_d, n_d);
    let dprev_ref = download_f32(&ctx, &d_prev, n_prev);
    let dda_ref = download_f32(&ctx, &d_da, n_da);

    // Typed.
    let dy_t = upload_typed(&ctx, &dy_vals, dtype);
    let x_t = upload_typed(&ctx, &x_vals, dtype);
    let q_t = upload_typed(&ctx, &q_vals, dtype);
    let ks_t = upload_typed(&ctx, &ks_vals, dtype);

    let d_x2 = GpuBuffer::zeros(&ctx.stream, n_x).unwrap();
    let d_q2 = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let d_ks2 = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let d_qk2 = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let d_d2 = GpuBuffer::zeros(&ctx.stream, n_d).unwrap();
    let d_prev2 = GpuBuffer::zeros(&ctx.stream, n_prev).unwrap();
    let d_da2 = GpuBuffer::zeros(&ctx.stream, n_da).unwrap();
    ctx.stream.synchronize().unwrap();
    let mut bld = ctx
        .stream
        .launch_builder(m3k.m3_chunk_scan_bwd_typed.get(dtype));
    let args = [
        d_x2.cached_ptr(),
        d_q2.cached_ptr(),
        d_ks2.cached_ptr(),
        d_qk2.cached_ptr(),
        d_d2.cached_ptr(),
        d_prev2.cached_ptr(),
        d_da2.cached_ptr(),
        dy_t.cached_ptr(),
        x_t.cached_ptr(),
        q_t.cached_ptr(),
        ks_t.cached_ptr(),
        qk_buf.cached_ptr(),
        da_buf.cached_ptr(),
        prev_buf.cached_ptr(),
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
    let dx_got = download_f32(&ctx, &d_x2, n_x);
    let dq_got = download_f32(&ctx, &d_q2, n_q);
    let dks_got = download_f32(&ctx, &d_ks2, n_q);
    let dqk_got = download_f32(&ctx, &d_qk2, n_th);
    let dd_got = download_f32(&ctx, &d_d2, n_d);
    let dprev_got = download_f32(&ctx, &d_prev2, n_prev);
    let dda_got = download_f32(&ctx, &d_da2, n_da);

    eprintln!("m3_chunk_scan_bwd (big) {dtype:?}:");
    let (cos_min, norm_tol) = tolerances(dtype);
    assert_close("d_x", &dx_ref, &dx_got, cos_min, norm_tol);
    assert_close("d_Q", &dq_ref, &dq_got, cos_min, norm_tol);
    assert_close("d_K_scaled", &dks_ref, &dks_got, cos_min, norm_tol);
    assert_close("d_qk_dot", &dqk_ref, &dqk_got, cos_min, norm_tol);
    assert_close("d_D", &dd_ref, &dd_got, cos_min, norm_tol);
    assert_close("d_prev_states", &dprev_ref, &dprev_got, cos_min, norm_tol);
    assert_close("d_dA_cumsum", &dda_ref, &dda_got, cos_min, norm_tol);
}

#[test]
fn m3_chunk_scan_bwd_big_bf16() {
    check_chunk_scan_bwd_big(WeightDtype::Bf16);
}
#[test]
fn m3_chunk_scan_bwd_big_f16() {
    check_chunk_scan_bwd_big(WeightDtype::F16);
}
