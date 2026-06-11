//! Step 8e — parity test for the new `ssm_parallel_scan_bwd` against the
//! existing sequential `ssm_backward_local`. Validates the f32 + bf16 + f16
//! instantiations of the typed macro DEFINE_SSM_PARALLEL_SCAN_BWD.

#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::kernels::MambaKernels;
use mamba_rs::mamba_ssm::gpu::launch::{grid_1d, grid_parallel_scan_bwd};

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

fn upload_f32(ctx: &GpuCtx, data: &[f32]) -> GpuBuffer {
    let mut b = GpuBuffer::zeros(&ctx.stream, data.len()).unwrap();
    ctx.stream.synchronize().unwrap();
    b.upload(&ctx.stream, data).unwrap();
    ctx.stream.synchronize().unwrap();
    b
}

fn upload_typed(ctx: &GpuCtx, data: &[f32], dtype: WeightDtype) -> DtypedBuf {
    let buf = DtypedBuf::zeros(&ctx.stream, data.len(), dtype).unwrap();
    ctx.stream.synchronize().unwrap();
    buf.upload_f32(&ctx.stream, data).unwrap();
    ctx.stream.synchronize().unwrap();
    buf
}

fn download_f32(ctx: &GpuCtx, buf: &GpuBuffer, n: usize) -> Vec<f32> {
    let mut v = vec![0f32; n];
    buf.download(&ctx.stream, &mut v).unwrap();
    ctx.stream.synchronize().unwrap();
    v
}

fn download_typed(ctx: &GpuCtx, buf: &DtypedBuf) -> Vec<f32> {
    let mut out = vec![0f32; buf.len_elems()];
    buf.download_f32(&ctx.stream, &mut out).unwrap();
    ctx.stream.synchronize().unwrap();
    out
}

fn make_ctx() -> (GpuCtx, MambaKernels) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let kernels = MambaKernels::compile(ctx.stream.context(), "sm_89").unwrap();
    (ctx, kernels)
}

type BwdOuts = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

struct BwdInputs {
    b: usize,
    t: usize,
    di: usize,
    ds: usize,
    h_saved: Vec<f32>,
    delta: Vec<f32>,
    u: Vec<f32>,
    b_buf: Vec<f32>,
    c_buf: Vec<f32>,
    a_neg: Vec<f32>,
    d_param: Vec<f32>,
    dy: Vec<f32>,
}

fn make_inputs(b: usize, t: usize, di: usize, ds: usize) -> BwdInputs {
    // h_saved must be a plausible state — random is fine for parity (we're
    // not testing math vs CPU oracle, just kernel-vs-kernel match).
    BwdInputs {
        b,
        t,
        di,
        ds,
        h_saved: det_rand(b * (t + 1) * di * ds, 0xA1),
        delta: det_rand(b * t * di, 0xA2)
            .into_iter()
            .map(|x| 0.1 + 0.05 * (x + 0.5))
            .collect(),
        u: det_rand(b * t * di, 0xA3),
        b_buf: det_rand(b * t * ds, 0xA4),
        c_buf: det_rand(b * t * ds, 0xA5),
        a_neg: (0..di * ds).map(|i| -0.1 - 0.005 * (i as f32)).collect(),
        d_param: (0..di).map(|i| 0.5 + 0.05 * (i as f32)).collect(),
        dy: det_rand(b * t * di, 0xA6),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_seq_f32(ctx: &GpuCtx, k: &MambaKernels, inp: &BwdInputs) -> BwdOuts {
    let (b, t, di, ds) = (inp.b, inp.t, inp.di, inp.ds);
    let h_saved = upload_f32(ctx, &inp.h_saved);
    let delta = upload_f32(ctx, &inp.delta);
    let u = upload_f32(ctx, &inp.u);
    let b_buf = upload_f32(ctx, &inp.b_buf);
    let c_buf = upload_f32(ctx, &inp.c_buf);
    let a_neg = upload_f32(ctx, &inp.a_neg);
    let d_param = upload_f32(ctx, &inp.d_param);
    let dy = upload_f32(ctx, &inp.dy);

    let n_btdi = b * t * di;
    let n_btds = b * t * di * ds;
    let n_bdi = b * di;
    let n_bdids = b * di * ds;

    let mut d_delta = GpuBuffer::zeros(&ctx.stream, n_btdi).unwrap();
    let mut d_u = GpuBuffer::zeros(&ctx.stream, n_btdi).unwrap();
    let mut d_b_local = GpuBuffer::zeros(&ctx.stream, n_btds).unwrap();
    let mut d_c_local = GpuBuffer::zeros(&ctx.stream, n_btds).unwrap();
    let mut d_d_local = GpuBuffer::zeros(&ctx.stream, n_bdi).unwrap();
    let mut d_a_log_local = GpuBuffer::zeros(&ctx.stream, n_bdids).unwrap();
    ctx.stream.synchronize().unwrap();

    let bi = b as i32;
    let ti = t as i32;
    let di_i = di as i32;
    let ds_i = ds as i32;
    let mut bld = ctx.stream.launch_builder(&k.ssm_backward_local);
    let h = h_saved.cached_ptr();
    let dl = delta.cached_ptr();
    let uu = u.cached_ptr();
    let bb = b_buf.cached_ptr();
    let cc = c_buf.cached_ptr();
    let aa = a_neg.cached_ptr();
    let dp = d_param.cached_ptr();
    let dyp = dy.cached_ptr();
    let ddl = d_delta.cached_ptr();
    let dup = d_u.cached_ptr();
    let dbl = d_b_local.cached_ptr();
    let dcl = d_c_local.cached_ptr();
    let ddd = d_d_local.cached_ptr();
    let dal = d_a_log_local.cached_ptr();
    bld.arg(&h);
    bld.arg(&dl);
    bld.arg(&uu);
    bld.arg(&bb);
    bld.arg(&cc);
    bld.arg(&aa);
    bld.arg(&dp);
    bld.arg(&dyp);
    bld.arg(&ddl);
    bld.arg(&dup);
    bld.arg(&dbl);
    bld.arg(&dcl);
    bld.arg(&ddd);
    bld.arg(&dal);
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&di_i);
    bld.arg(&ds_i);
    unsafe { bld.launch(grid_1d(b * di)) }.unwrap();
    ctx.stream.synchronize().unwrap();

    // Suppress unused warnings on the input mut bindings.
    let _ = &mut d_delta;
    let _ = &mut d_u;
    let _ = &mut d_b_local;
    let _ = &mut d_c_local;
    let _ = &mut d_d_local;
    let _ = &mut d_a_log_local;
    (
        download_f32(ctx, &d_delta, n_btdi),
        download_f32(ctx, &d_u, n_btdi),
        download_f32(ctx, &d_b_local, n_btds),
        download_f32(ctx, &d_c_local, n_btds),
        download_f32(ctx, &d_d_local, n_bdi),
        download_f32(ctx, &d_a_log_local, n_bdids),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_par_typed(ctx: &GpuCtx, k: &MambaKernels, inp: &BwdInputs, dtype: WeightDtype) -> BwdOuts {
    let (b, t, di, ds) = (inp.b, inp.t, inp.di, inp.ds);
    let h_saved = upload_f32(ctx, &inp.h_saved);
    let delta = upload_typed(ctx, &inp.delta, dtype);
    let u = upload_typed(ctx, &inp.u, dtype);
    let b_buf = upload_typed(ctx, &inp.b_buf, dtype);
    let c_buf = upload_typed(ctx, &inp.c_buf, dtype);
    let a_neg = upload_f32(ctx, &inp.a_neg);
    let d_param = upload_f32(ctx, &inp.d_param);
    let dy = upload_typed(ctx, &inp.dy, dtype);

    let n_btdi = b * t * di;
    let n_btds = b * t * di * ds;
    let n_bdi = b * di;
    let n_bdids = b * di * ds;

    let d_delta = DtypedBuf::zeros(&ctx.stream, n_btdi, dtype).unwrap();
    let d_u = DtypedBuf::zeros(&ctx.stream, n_btdi, dtype).unwrap();
    let d_b_local = DtypedBuf::zeros(&ctx.stream, n_btds, dtype).unwrap();
    let d_c_local = DtypedBuf::zeros(&ctx.stream, n_btds, dtype).unwrap();
    let d_d_local = GpuBuffer::zeros(&ctx.stream, n_bdi).unwrap();
    let d_a_log_local = GpuBuffer::zeros(&ctx.stream, n_bdids).unwrap();
    ctx.stream.synchronize().unwrap();

    let bi = b as i32;
    let ti = t as i32;
    let di_i = di as i32;
    let ds_i = ds as i32;
    let mut bld = ctx
        .stream
        .launch_builder(k.ssm_parallel_bwd_typed.get(dtype));
    let h = h_saved.cached_ptr();
    let dl = delta.cached_ptr();
    let uu = u.cached_ptr();
    let bb = b_buf.cached_ptr();
    let cc = c_buf.cached_ptr();
    let aa = a_neg.cached_ptr();
    let dp = d_param.cached_ptr();
    let dyp = dy.cached_ptr();
    let ddl = d_delta.cached_ptr();
    let dup = d_u.cached_ptr();
    let dbl = d_b_local.cached_ptr();
    let dcl = d_c_local.cached_ptr();
    let ddd = d_d_local.cached_ptr();
    let dal = d_a_log_local.cached_ptr();
    bld.arg(&h);
    bld.arg(&dl);
    bld.arg(&uu);
    bld.arg(&bb);
    bld.arg(&cc);
    bld.arg(&aa);
    bld.arg(&dp);
    bld.arg(&dyp);
    bld.arg(&ddl);
    bld.arg(&dup);
    bld.arg(&dbl);
    bld.arg(&dcl);
    bld.arg(&ddd);
    bld.arg(&dal);
    bld.arg(&bi);
    bld.arg(&ti);
    bld.arg(&di_i);
    bld.arg(&ds_i);
    let cfg: LaunchConfig = grid_parallel_scan_bwd(b, di);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    (
        download_typed(ctx, &d_delta),
        download_typed(ctx, &d_u),
        download_typed(ctx, &d_b_local),
        download_typed(ctx, &d_c_local),
        download_f32(ctx, &d_d_local, n_bdi),
        download_f32(ctx, &d_a_log_local, n_bdids),
    )
}

fn check_parity(b: usize, t: usize, di: usize, ds: usize, dtype: WeightDtype) {
    let (ctx, k) = make_ctx();
    let inp = make_inputs(b, t, di, ds);
    let (dd_seq, du_seq, dbl_seq, dcl_seq, ddd_seq, dal_seq) = run_seq_f32(&ctx, &k, &inp);
    let (dd_par, du_par, dbl_par, dcl_par, ddd_par, dal_par) = run_par_typed(&ctx, &k, &inp, dtype);

    eprintln!("ssm_parallel_scan_bwd ({dtype:?}, B={b} T={t} di={di} ds={ds}):");
    let (cos_min, norm_tol) = match dtype {
        WeightDtype::F32 => (1.0 - 1e-4, 0.02),
        WeightDtype::Bf16 => (0.99, 0.05),
        WeightDtype::F16 => (0.999, 0.02),
    };
    assert_close("d_delta", &dd_seq, &dd_par, cos_min, norm_tol);
    assert_close("d_u", &du_seq, &du_par, cos_min, norm_tol);
    assert_close("d_B_local", &dbl_seq, &dbl_par, cos_min, norm_tol);
    assert_close("d_C_local", &dcl_seq, &dcl_par, cos_min, norm_tol);
    assert_close("d_D_local", &ddd_seq, &ddd_par, cos_min, norm_tol);
    assert_close("d_a_log_local", &dal_seq, &dal_par, cos_min, norm_tol);
}

// Small case: T=32 (single chunk), small ds, single batch.
#[test]
fn parity_small_f32() {
    check_parity(1, 32, 4, 4, WeightDtype::F32);
}

#[test]
fn parity_small_bf16() {
    check_parity(1, 32, 4, 4, WeightDtype::Bf16);
}

#[test]
fn parity_small_f16() {
    check_parity(1, 32, 4, 4, WeightDtype::F16);
}

// Multi-chunk case: T=2048 forces 2 chunks (CHUNK_SIZE=1024).
#[test]
fn parity_multi_chunk_f32() {
    check_parity(1, 2048, 4, 8, WeightDtype::F32);
}

#[test]
fn parity_multi_chunk_bf16() {
    check_parity(1, 2048, 4, 8, WeightDtype::Bf16);
}

// Partial-last-chunk case: T=1500.
#[test]
fn parity_partial_last_chunk_f32() {
    check_parity(1, 1500, 4, 8, WeightDtype::F32);
}

// Multi-batch case.
#[test]
fn parity_multi_batch_f32() {
    check_parity(2, 256, 8, 16, WeightDtype::F32);
}

// 3-chunk regression case for the audit-found postfix double-count bug
// (kernels/mamba_ssm_parallel.cu:1349-1354). With the bug, the last 8
// timesteps of every chunk except the very last would silently corrupt
// d_delta/d_u/d_B/d_a_log when n_chunks ≥ 2. CHUNK_SIZE = NTHREADS *
// NITEMS = 128 * 8 = 1024 → T = 3072 forces 3 chunks. The bug would
// fire on 2 of those (chunks 0 and 1, i.e. the earlier-in-time ones).
// Regression introduced post-Step 8e by audit Agent 2 (M1 deep), fixed
// in the same patch by setting next_a=1.0, next_b=0.0 for the last
// thread (identity for the exclusive-next-thread compose).
#[test]
fn parity_three_chunks_postfix_regression_f32() {
    check_parity(1, 3072, 4, 8, WeightDtype::F32);
}

#[test]
fn parity_three_chunks_postfix_regression_bf16() {
    check_parity(1, 3072, 4, 8, WeightDtype::Bf16);
}
