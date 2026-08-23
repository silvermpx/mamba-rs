//! parity test for the new `ssm_parallel_scan_bwd` against the
//! existing sequential `ssm_backward_local`. Validates the f32 + bf16 + f16
//! instantiations of the typed macro DEFINE_SSM_PARALLEL_SCAN_BWD.

#![cfg(feature = "cuda")]

mod common;

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
    let kernels = MambaKernels::compile(ctx.stream.context(), common::bench::arch0()).unwrap();
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

/// S2 T-major tape adapters: the parallel kernels read h_saved as
/// [b][d][n][t+1] and write their dB/dC locals as [b][n][d][t]; the
/// sequential reference keeps the historical layouts. Values are
/// identical — these permutations let the two kernels share one logical
/// input and one comparison space.
fn h_to_tmajor(h: &[f32], b: usize, t: usize, di: usize, ds: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; h.len()];
    for bb in 0..b {
        for t1 in 0..=t {
            for d in 0..di {
                for n in 0..ds {
                    let old = ((bb * (t + 1) + t1) * di + d) * ds + n;
                    let new = ((bb * di + d) * ds + n) * (t + 1) + t1;
                    out[new] = h[old];
                }
            }
        }
    }
    out
}

fn locals_from_tmajor(x: &[f32], b: usize, t: usize, di: usize, ds: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for bb in 0..b {
        for t1 in 0..t {
            for d in 0..di {
                for n in 0..ds {
                    let new = ((bb * ds + n) * di + d) * t + t1;
                    let old = ((bb * t + t1) * di + d) * ds + n;
                    out[old] = x[new];
                }
            }
        }
    }
    out
}

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

/// [b][t][n] -> [b][n][t] for the parallel kernels' T-major B/C reads
/// (the production gather writes this layout on the parallel route).
fn bc_to_tmajor(src: &[f32], b: usize, t: usize, ds: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    for bb in 0..b {
        for tt in 0..t {
            for n in 0..ds {
                out[(bb * ds + n) * t + tt] = src[(bb * t + tt) * ds + n];
            }
        }
    }
    out
}

fn run_par_typed(ctx: &GpuCtx, k: &MambaKernels, inp: &BwdInputs, dtype: WeightDtype) -> BwdOuts {
    let (b, t, di, ds) = (inp.b, inp.t, inp.di, inp.ds);
    let h_tmajor = h_to_tmajor(&inp.h_saved, inp.b, inp.t, inp.di, inp.ds);
    let h_saved = upload_f32(ctx, &h_tmajor);
    let delta = upload_typed(ctx, &inp.delta, dtype);
    let u = upload_typed(ctx, &inp.u, dtype);
    let b_buf = upload_typed(ctx, &bc_to_tmajor(&inp.b_buf, b, t, ds), dtype);
    let c_buf = upload_typed(ctx, &bc_to_tmajor(&inp.c_buf, b, t, ds), dtype);
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
    // Tape args: this test feeds a handcrafted FULL h tape, so slim = 0
    // (the tape pointer is unused on that path).
    let slim0: i32 = 0;
    bld.arg(&h);
    bld.arg(&slim0);
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
    let (dd_par, du_par, dbl_par_t, dcl_par_t, ddd_par, dal_par) =
        run_par_typed(&ctx, &k, &inp, dtype);
    let dbl_par = locals_from_tmajor(&dbl_par_t, b, t, di, ds);
    let dcl_par = locals_from_tmajor(&dcl_par_t, b, t, di, ds);

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
// Regression introduced by the pre-release audit, fixed
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

// ---------------------------------------------------------------------
// Production-route parity: fold kernel + slim tape. The pre-0.6.4 file
// drove only the ungrouped kernel with a handcrafted full tape — the
// route the trainer never takes at campaign shapes. Here the parallel
// forward produces the REAL h tape (full for the reference, slim for the
// device replay), the fold backward consumes it, and the sequential
// reference judges every output — dB/dC compared at the fold depth by
// group-summing the reference.
// ---------------------------------------------------------------------

fn h_from_tmajor(h_t: &[f32], b: usize, t: usize, di: usize, ds: usize) -> Vec<f32> {
    let mut out = vec![0f32; b * (t + 1) * di * ds];
    for bb in 0..b {
        for d in 0..di {
            for n in 0..ds {
                for tt in 0..=t {
                    let src = ((bb * di + d) * ds + n) * (t + 1) + tt;
                    let dst = ((bb * (t + 1) + tt) * di + d) * ds + n;
                    out[dst] = h_t[src];
                }
            }
        }
    }
    out
}

fn check_fold_slim_parity(b: usize, t: usize, di: usize, ds: usize, dtype: WeightDtype) {
    check_fold_parity(b, t, di, ds, dtype, true)
}

fn check_fold_parity(b: usize, t: usize, di: usize, ds: usize, dtype: WeightDtype, slim: bool) {
    use mamba_rs::mamba_ssm::gpu::launch::{
        SCAN_BWD_DGROUP, grid_parallel_scan_bwd_fold, grid_parallel_scan_typed, scan_tape_len,
    };
    assert_eq!(
        di % SCAN_BWD_DGROUP,
        0,
        "fold parity needs di divisible by G"
    );
    let (ctx, k) = make_ctx();
    let inp = make_inputs(b, t, di, ds);

    // Device forward, both tape modes, on the SAME inputs.
    let mut h = GpuBuffer::zeros(&ctx.stream, b * di * ds).unwrap();
    let y = DtypedBuf::zeros(&ctx.stream, b * t * di, dtype).unwrap();
    let delta = upload_typed(&ctx, &inp.delta, dtype);
    let u = upload_typed(&ctx, &inp.u, dtype);
    let b_buf = upload_typed(&ctx, &bc_to_tmajor(&inp.b_buf, b, t, ds), dtype);
    let c_buf = upload_typed(&ctx, &bc_to_tmajor(&inp.c_buf, b, t, ds), dtype);
    let a_neg = upload_f32(&ctx, &inp.a_neg);
    let d_param = upload_f32(&ctx, &inp.d_param);
    let h_saved = GpuBuffer::zeros(&ctx.stream, b * (t + 1) * di * ds).unwrap();
    let tape = GpuBuffer::zeros(&ctx.stream, scan_tape_len(b, t, di, ds)).unwrap();
    ctx.stream.synchronize().unwrap();

    let bi = b as i32;
    let ti = t as i32;
    let di_i = di as i32;
    let ds_i = ds as i32;
    let run_fwd = |slim_i: i32, tp: u64, hp: u64| {
        let mut bld = ctx
            .stream
            .launch_builder(k.ssm_parallel_fwd_typed.get(dtype));
        let yp = y.cached_ptr();
        let hs = h_saved.cached_ptr();
        let dl = delta.cached_ptr();
        let uu = u.cached_ptr();
        let bbp = b_buf.cached_ptr();
        let ccp = c_buf.cached_ptr();
        let aa = a_neg.cached_ptr();
        let dp = d_param.cached_ptr();
        bld.arg(&hp);
        bld.arg(&yp);
        bld.arg(&hs);
        bld.arg(&dl);
        bld.arg(&uu);
        bld.arg(&bbp);
        bld.arg(&ccp);
        bld.arg(&aa);
        bld.arg(&dp);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&di_i);
        bld.arg(&ds_i);
        bld.arg(&tp);
        bld.arg(&slim_i);
        unsafe { bld.launch(grid_parallel_scan_typed(b, di, dtype.size_bytes())) }.unwrap();
        ctx.stream.synchronize().unwrap();
    };
    // Full-tape forward FIRST and the reference h captured BEFORE the
    // slim forward runs - the slim pass clobbers h_saved, and a reference
    // read after it judges the (correct) replay against corrupted h. That
    // ordering bug in this very harness once read as a production defect.
    run_fwd(0, h_saved.cached_ptr(), h.cached_ptr());
    let h_dev = download_f32(&ctx, &h_saved, b * (t + 1) * di * ds);
    if slim {
        // The forward leaves its final running state in `h`; a second
        // pass starting from h_T encodes a shifted trajectory into the
        // tape, and the replay then faithfully reproduces the WRONG h -
        // exactly the harness defect this comment guards against.
        h.zero(&ctx.stream).unwrap();
        ctx.stream.synchronize().unwrap();
        run_fwd(1, tape.cached_ptr(), h.cached_ptr());
    }
    let mut inp_ref = make_inputs(b, t, di, ds);
    inp_ref.h_saved = h_from_tmajor(&h_dev, b, t, di, ds);
    let (dd_seq, du_seq, dbl_seq, dcl_seq, ddd_seq, dal_seq) = run_seq_f32(&ctx, &k, &inp_ref);

    // Fold backward on the slim tape.
    let g = SCAN_BWD_DGROUP;
    let dy = upload_typed(&ctx, &inp.dy, dtype);
    let d_delta = DtypedBuf::zeros(&ctx.stream, b * t * di, dtype).unwrap();
    let d_u = DtypedBuf::zeros(&ctx.stream, b * t * di, dtype).unwrap();
    let d_b_fold = DtypedBuf::zeros(&ctx.stream, b * t * (di / g) * ds, dtype).unwrap();
    let d_c_fold = DtypedBuf::zeros(&ctx.stream, b * t * (di / g) * ds, dtype).unwrap();
    let d_d_local = GpuBuffer::zeros(&ctx.stream, b * di).unwrap();
    let d_a_log_local = GpuBuffer::zeros(&ctx.stream, b * di * ds).unwrap();
    ctx.stream.synchronize().unwrap();
    {
        let mut bld = ctx
            .stream
            .launch_builder(k.ssm_parallel_bwd_fold_typed.get(dtype));
        let hs = h_saved.cached_ptr();
        let tp = tape.cached_ptr();
        let dl = delta.cached_ptr();
        let uu = u.cached_ptr();
        let bbp = b_buf.cached_ptr();
        let ccp = c_buf.cached_ptr();
        let aa = a_neg.cached_ptr();
        let dp = d_param.cached_ptr();
        let dyp = dy.cached_ptr();
        let ddl = d_delta.cached_ptr();
        let dup = d_u.cached_ptr();
        let dbl = d_b_fold.cached_ptr();
        let dcl = d_c_fold.cached_ptr();
        let ddd = d_d_local.cached_ptr();
        let dal = d_a_log_local.cached_ptr();
        bld.arg(&hs);
        bld.arg(&dl);
        bld.arg(&uu);
        bld.arg(&bbp);
        bld.arg(&ccp);
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
        let slim_i: i32 = i32::from(slim);
        let tape_arg = if slim { tp } else { hs };
        bld.arg(&tape_arg);
        bld.arg(&slim_i);
        unsafe { bld.launch(grid_parallel_scan_bwd_fold(b, di, ds, dtype.size_bytes())) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();

    let dd_par = download_typed(&ctx, &d_delta);
    let du_par = download_typed(&ctx, &d_u);
    let dbl_par = locals_from_tmajor(&download_typed(&ctx, &d_b_fold), b, t, di / g, ds);
    let dcl_par = locals_from_tmajor(&download_typed(&ctx, &d_c_fold), b, t, di / g, ds);
    let ddd_par = download_f32(&ctx, &d_d_local, b * di);
    let dal_par = download_f32(&ctx, &d_a_log_local, b * di * ds);

    // Group-sum the reference dB/dC to the fold depth.
    let group_sum = |x: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; b * t * (di / g) * ds];
        for bb in 0..b {
            for tt in 0..t {
                for d in 0..di {
                    for n in 0..ds {
                        let src = ((bb * t + tt) * di + d) * ds + n;
                        let dst = ((bb * t + tt) * (di / g) + d / g) * ds + n;
                        out[dst] += x[src];
                    }
                }
            }
        }
        out
    };

    eprintln!("ssm_parallel_bwd_fold+slim ({dtype:?}, B={b} T={t} di={di} ds={ds}):");
    let (cos_min, norm_tol) = match dtype {
        WeightDtype::F32 => (1.0 - 1e-4, 0.02),
        WeightDtype::Bf16 => (0.99, 0.05),
        WeightDtype::F16 => (0.999, 0.02),
    };
    assert_close("d_delta", &dd_seq, &dd_par, cos_min, norm_tol);
    assert_close("d_u", &du_seq, &du_par, cos_min, norm_tol);
    assert_close(
        "d_B_fold",
        &group_sum(&dbl_seq),
        &dbl_par,
        cos_min,
        norm_tol,
    );
    assert_close(
        "d_C_fold",
        &group_sum(&dcl_seq),
        &dcl_par,
        cos_min,
        norm_tol,
    );
    assert_close("d_D_local", &ddd_seq, &ddd_par, cos_min, norm_tol);
    assert_close("d_a_log_local", &dal_seq, &dal_par, cos_min, norm_tol);
}

#[test]
fn parity_fold_slim_f32() {
    check_fold_slim_parity(1, 2048, 4, 8, WeightDtype::F32);
}

// Bisection arms for the multi-chunk f32 divergence: a single chunk
// isolates the chunk-boundary replay, and T=1300 matches the campaign
// chunk count.
#[test]
fn parity_fold_slim_f32_single_chunk() {
    check_fold_slim_parity(1, 512, 4, 8, WeightDtype::F32);
}

#[test]
fn parity_fold_fulltape_f32_single_chunk() {
    check_fold_parity(1, 512, 4, 8, WeightDtype::F32, false);
}

#[test]
fn parity_fold_slim_f32_campaign_chunks() {
    check_fold_slim_parity(1, 1300, 4, 8, WeightDtype::F32);
}

#[test]
fn parity_fold_slim_bf16() {
    check_fold_slim_parity(2, 1300, 8, 16, WeightDtype::Bf16);
}

/// Diagnostic (manual): where exactly the fold's slim-replay d_C diverges
/// from its own full-tape run — positions, not norms.
#[test]
#[ignore = "diagnostic printer"]
fn diag_fold_slim_vs_full_positions() {
    use mamba_rs::mamba_ssm::gpu::launch::SCAN_BWD_DGROUP;
    let (b, t, di, ds) = (1usize, 512usize, 4usize, 8usize);
    let dtype = WeightDtype::F32;
    let g = SCAN_BWD_DGROUP;
    let run = |slim: bool, fold: bool| -> Vec<f32> {
        // Reuse the parity harness by rebuilding the rig each time (same
        // seeds -> identical inputs), returning d_C at fold depth.
        let (ctx, k) = make_ctx();
        let inp = make_inputs(b, t, di, ds);
        let h = GpuBuffer::zeros(&ctx.stream, b * di * ds).unwrap();
        let y = DtypedBuf::zeros(&ctx.stream, b * t * di, dtype).unwrap();
        let delta = upload_typed(&ctx, &inp.delta, dtype);
        let u = upload_typed(&ctx, &inp.u, dtype);
        let b_buf = upload_typed(&ctx, &bc_to_tmajor(&inp.b_buf, b, t, ds), dtype);
        let c_buf = upload_typed(&ctx, &bc_to_tmajor(&inp.c_buf, b, t, ds), dtype);
        let a_neg = upload_f32(&ctx, &inp.a_neg);
        let d_param = upload_f32(&ctx, &inp.d_param);
        let h_saved = GpuBuffer::zeros(&ctx.stream, b * (t + 1) * di * ds).unwrap();
        let tape = GpuBuffer::zeros(
            &ctx.stream,
            mamba_rs::mamba_ssm::gpu::launch::scan_tape_len(b, t, di, ds),
        )
        .unwrap();
        ctx.stream.synchronize().unwrap();
        let bi = b as i32;
        let ti = t as i32;
        let di_i = di as i32;
        let ds_i = ds as i32;
        // ONE forward, matching the backward's tape mode - running both
        // would let the second overwrite state the first arm depends on.
        let fwd_modes: [(i32, u64); 1] = if slim {
            [(1i32, tape.cached_ptr())]
        } else {
            [(0i32, h_saved.cached_ptr())]
        };
        for (sl, tp) in fwd_modes {
            let mut bld = ctx
                .stream
                .launch_builder(k.ssm_parallel_fwd_typed.get(dtype));
            let hp = h.cached_ptr();
            let yp = y.cached_ptr();
            let hs = h_saved.cached_ptr();
            let dl = delta.cached_ptr();
            let uu = u.cached_ptr();
            let bbp = b_buf.cached_ptr();
            let ccp = c_buf.cached_ptr();
            let aa = a_neg.cached_ptr();
            let dp = d_param.cached_ptr();
            bld.arg(&hp);
            bld.arg(&yp);
            bld.arg(&hs);
            bld.arg(&dl);
            bld.arg(&uu);
            bld.arg(&bbp);
            bld.arg(&ccp);
            bld.arg(&aa);
            bld.arg(&dp);
            bld.arg(&bi);
            bld.arg(&ti);
            bld.arg(&di_i);
            bld.arg(&ds_i);
            bld.arg(&tp);
            bld.arg(&sl);
            unsafe {
                bld.launch(mamba_rs::mamba_ssm::gpu::launch::grid_parallel_scan_typed(
                    b,
                    di,
                    dtype.size_bytes(),
                ))
            }
            .unwrap();
        }
        let dy = upload_typed(&ctx, &inp.dy, dtype);
        let d_delta = DtypedBuf::zeros(&ctx.stream, b * t * di, dtype).unwrap();
        let d_u = DtypedBuf::zeros(&ctx.stream, b * t * di, dtype).unwrap();
        let out_rows = if fold { di / g } else { di };
        let d_b_fold = DtypedBuf::zeros(&ctx.stream, b * t * out_rows * ds, dtype).unwrap();
        let d_c_fold = DtypedBuf::zeros(&ctx.stream, b * t * out_rows * ds, dtype).unwrap();
        let d_d_local = GpuBuffer::zeros(&ctx.stream, b * di).unwrap();
        let d_a_log_local = GpuBuffer::zeros(&ctx.stream, b * di * ds).unwrap();
        ctx.stream.synchronize().unwrap();
        let mut bld = ctx.stream.launch_builder(if fold {
            k.ssm_parallel_bwd_fold_typed.get(dtype)
        } else {
            k.ssm_parallel_bwd_typed.get(dtype)
        });
        let hs = h_saved.cached_ptr();
        let tp = tape.cached_ptr();
        let dl = delta.cached_ptr();
        let uu = u.cached_ptr();
        let bbp = b_buf.cached_ptr();
        let ccp = c_buf.cached_ptr();
        let aa = a_neg.cached_ptr();
        let dp = d_param.cached_ptr();
        let dyp = dy.cached_ptr();
        let ddl = d_delta.cached_ptr();
        let dup = d_u.cached_ptr();
        let dbl = d_b_fold.cached_ptr();
        let dcl = d_c_fold.cached_ptr();
        let ddd = d_d_local.cached_ptr();
        let dal = d_a_log_local.cached_ptr();
        bld.arg(&hs);
        bld.arg(&dl);
        bld.arg(&uu);
        bld.arg(&bbp);
        bld.arg(&ccp);
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
        let sl: i32 = i32::from(slim);
        let tpa = if slim { tp } else { hs };
        bld.arg(&tpa);
        bld.arg(&sl);
        let cfg = if fold {
            mamba_rs::mamba_ssm::gpu::launch::grid_parallel_scan_bwd_fold(
                b,
                di,
                ds,
                dtype.size_bytes(),
            )
        } else {
            grid_parallel_scan_bwd(b, di)
        };
        unsafe { bld.launch(cfg) }.unwrap();
        ctx.stream.synchronize().unwrap();
        download_typed(&ctx, &d_c_fold)
    };
    for (label, fold) in [("ungrouped", false), ("fold", true)] {
        let full = run(false, fold);
        let slim = run(true, fold);
        let n_mis = full
            .iter()
            .zip(&slim)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max_err = full
            .iter()
            .zip(&slim)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "ROUTE {label}: slim-vs-full d_C mismatched elems {n_mis}/{} max_abs {max_err:9.6}",
            full.len()
        );
    }
    let full = run(false, true);
    let slim = run(true, true);
    let groups = di / g;
    let mut per_t = vec![0f32; t];
    let mut worst: Vec<(f32, usize, usize, usize)> = Vec::new();
    for n in 0..ds {
        for gr in 0..groups {
            for tt in 0..t {
                let idx = ((n) * groups + gr) * t + tt;
                let e = (full[idx] - slim[idx]).abs();
                per_t[tt] += e;
                worst.push((e, tt, n, gr));
            }
        }
    }
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    eprintln!("worst abs diffs (err, t, n, group):");
    for w in &worst[..12] {
        eprintln!("  {:9.5} t={} n={} g={}", w.0, w.1, w.2, w.3);
    }
    let t_nonzero: Vec<usize> = (0..t).filter(|&tt| per_t[tt] > 1e-6).collect();
    eprintln!(
        "t positions with error: {} of {t}; first {:?} last {:?}",
        t_nonzero.len(),
        &t_nonzero.iter().take(8).collect::<Vec<_>>(),
        &t_nonzero.iter().rev().take(8).collect::<Vec<_>>()
    );
}
