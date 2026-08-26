//! Typed (bf16/f16) sgemm_bi — bit parity vs the f32 triad.
//!
//! Contract under test: a typed GEMM is BIT-IDENTICAL to "quantize inputs
//! to the 16-bit dtype, upcast to f32, run the f32 kernel, RNE-downcast the
//! output" — because the typed kernels keep f32 smem/accumulation and the
//! exact FMA chains of their f32 twins, upcasting only at the load site.
//!
//! Two layers:
//! - stage-2 native buckets (NN gemv / ultra-thin / narrow / narrow-small,
//!   TN gemv / narrow with f32 dW, NT gemv / narrow) — direct dispatcher
//!   calls;
//! - full-coverage `bi_sgemm_*_typed` blas entries — uncovered shapes route
//!   through the upcast → f32 sgemm_bi → RNE-downcast fallback, which must
//!   satisfy the SAME bit contract.

#![cfg(feature = "cuda")]

use cudarc::driver::PushKernelArg;
use half::{bf16, f16};
use mamba_rs::mamba_ssm::gpu::blas::{
    TypedPtr, bi_sgemm_backward_dw_typed, bi_sgemm_backward_dx_typed, bi_sgemm_forward_typed,
};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad;
use std::mem::size_of;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * scale
        })
        .collect()
}

/// Quantize f32 host values to the dtype's representable grid (RNE), back
/// as exact f32 — these are the values both the typed and the f32 reference
/// paths consume.
fn quantize(v: &[f32], dt: WeightDtype) -> Vec<f32> {
    match dt {
        WeightDtype::Bf16 => v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect(),
        WeightDtype::F16 => v.iter().map(|&x| f16::from_f32(x).to_f32()).collect(),
        WeightDtype::F32 => v.to_vec(),
    }
}

/// Host RNE downcast of the f32 reference output to the dtype grid.
fn downcast_grid(v: &[f32], dt: WeightDtype) -> Vec<f32> {
    quantize(v, dt)
}

struct Ctx {
    ctx: GpuCtx,
}

impl Ctx {
    fn new() -> Self {
        let device = GpuDevice::new(0).expect("gpu");
        let ctx = GpuCtx::new(&device).expect("ctx");
        // M-13 (GEMM-map audit): pin the tier explicitly - a shell with
        // MAMBA_RS_BI_TENSOR_CORES exported used to silently reroute this
        // suite through the TC kernels, testing a different contract.
        ctx.set_batch_invariant(true);
        ctx.set_bi_tensor_cores(false);
        ctx.set_fast_gemm(false);
        Self { ctx }
    }

    fn f32_buf(&self, data: &[f32]) -> GpuBuffer {
        let mut b = GpuBuffer::zeros(&self.ctx.stream, data.len()).unwrap();
        b.upload(&self.ctx.stream, data).unwrap();
        b
    }

    fn typed_buf(&self, data: &[f32], dt: WeightDtype) -> DtypedBuf {
        let b = DtypedBuf::zeros(&self.ctx.stream, data.len(), dt).unwrap();
        b.upload_f32(&self.ctx.stream, data).unwrap();
        b
    }
}

fn assert_bits(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}: bit mismatch at {i}: got {g:?} want {w:?}"
        );
    }
}

/// Forward parity: typed dispatch vs f32 dispatch on quantized inputs.
/// `full` routes through the blas-layer `bi_sgemm_forward_typed` (native
/// buckets + upcast fallback); otherwise the stage-2 dispatcher is called
/// directly and the shape must be natively covered.
fn check_forward(
    t: &Ctx,
    dt: WeightDtype,
    dims: (usize, usize, usize),
    with_bias: bool,
    full: bool,
) {
    let (m, k, n) = dims;
    let label = format!("fwd {dt:?} M{m} K{k} N{n} bias={with_bias} full={full}");
    let qx = quantize(&det(m * k, 11, 1.0), dt);
    let qw = quantize(&det(k * n, 22, 0.5), dt);
    let bias = det(n, 33, 0.25);

    // f32 reference on the SAME quantized values.
    let x32 = t.f32_buf(&qx);
    let w32 = t.f32_buf(&qw);
    let b32 = t.f32_buf(&bias);
    let mut y32 = GpuBuffer::zeros(&t.ctx.stream, m * n).unwrap();
    let bias_ptr = if with_bias { b32.cached_ptr() } else { 0 };
    gemm_bi_triad::sgemm_bi_forward(
        &t.ctx.stream,
        &t.ctx.kernels,
        &mut y32,
        &x32,
        w32.cached_ptr(),
        bias_ptr,
        (m, k, n),
    )
    .unwrap();
    t.ctx.stream.synchronize().unwrap();
    let ref_f32 = y32.to_cpu(&t.ctx.stream).unwrap();

    // Typed path.
    let xt = t.typed_buf(&qx, dt);
    let wt = t.typed_buf(&qw, dt);
    let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
    let (ytp, xtp, wtp) = (
        TypedPtr {
            ptr: yt.cached_ptr(),
            dtype: dt,
        },
        TypedPtr {
            ptr: xt.cached_ptr(),
            dtype: dt,
        },
        TypedPtr {
            ptr: wt.cached_ptr(),
            dtype: dt,
        },
    );
    if full {
        bi_sgemm_forward_typed(&t.ctx, ytp, xtp, wtp, bias_ptr, (m, k, n)).unwrap();
    } else {
        gemm_bi_triad::sgemm_bi_forward_typed(
            &t.ctx.stream,
            &t.ctx.kernels,
            ytp,
            xtp,
            wtp,
            bias_ptr,
            (m, k, n),
        )
        .unwrap();
    }
    t.ctx.stream.synchronize().unwrap();
    let mut got = vec![0.0f32; m * n];
    yt.download_f32(&t.ctx.stream, &mut got).unwrap();

    assert_bits(&label, &got, &downcast_grid(&ref_f32, dt));
}

/// dW parity: both paths accumulate into f32 — direct bit compare.
fn check_dw(t: &Ctx, dt: WeightDtype, dims: (usize, usize, usize), full: bool) {
    let (m, k, n) = dims;
    let label = format!("dw {dt:?} M{m} K{k} N{n} full={full}");
    let qx = quantize(&det(m * k, 44, 1.0), dt);
    let qdy = quantize(&det(m * n, 55, 0.5), dt);

    let x32 = t.f32_buf(&qx);
    let dy32 = t.f32_buf(&qdy);
    let dw32 = GpuBuffer::zeros(&t.ctx.stream, k * n).unwrap();
    gemm_bi_triad::sgemm_bi_backward_dw(
        &t.ctx.stream,
        &t.ctx.kernels,
        dw32.cached_ptr(),
        &dy32,
        &x32,
        (m, k, n),
    )
    .unwrap();
    t.ctx.stream.synchronize().unwrap();
    let ref_dw = dw32.to_cpu(&t.ctx.stream).unwrap();

    let xt = t.typed_buf(&qx, dt);
    let dyt = t.typed_buf(&qdy, dt);
    let dwt = GpuBuffer::zeros(&t.ctx.stream, k * n).unwrap();
    let (dytp, xtp) = (
        TypedPtr {
            ptr: dyt.cached_ptr(),
            dtype: dt,
        },
        TypedPtr {
            ptr: xt.cached_ptr(),
            dtype: dt,
        },
    );
    if full {
        bi_sgemm_backward_dw_typed(&t.ctx, dwt.cached_ptr(), dytp, xtp, (m, k, n)).unwrap();
    } else {
        gemm_bi_triad::sgemm_bi_backward_dw_typed(
            &t.ctx.stream,
            &t.ctx.kernels,
            dwt.cached_ptr(),
            dytp,
            xtp,
            (m, k, n),
        )
        .unwrap();
    }
    t.ctx.stream.synchronize().unwrap();
    let got = dwt.to_cpu(&t.ctx.stream).unwrap();

    assert_bits(&label, &got, &ref_dw);
}

/// dX parity: typed output vs RNE-downcast f32 reference.
fn check_dx(t: &Ctx, dt: WeightDtype, dims: (usize, usize, usize), full: bool) {
    let (m, k, n) = dims;
    let label = format!("dx {dt:?} M{m} K{k} N{n} full={full}");
    let qdy = quantize(&det(m * n, 66, 0.5), dt);
    let qw = quantize(&det(k * n, 77, 0.5), dt);

    let dy32 = t.f32_buf(&qdy);
    let w32 = t.f32_buf(&qw);
    let mut dx32 = GpuBuffer::zeros(&t.ctx.stream, m * k).unwrap();
    gemm_bi_triad::sgemm_bi_backward_dx(
        &t.ctx.stream,
        &t.ctx.kernels,
        &mut dx32,
        &dy32,
        w32.cached_ptr(),
        (m, k, n),
    )
    .unwrap();
    t.ctx.stream.synchronize().unwrap();
    let ref_dx = dx32.to_cpu(&t.ctx.stream).unwrap();

    let dyt = t.typed_buf(&qdy, dt);
    let wt = t.typed_buf(&qw, dt);
    let dxt = DtypedBuf::zeros(&t.ctx.stream, m * k, dt).unwrap();
    let (dxtp, dytp, wtp) = (
        TypedPtr {
            ptr: dxt.cached_ptr(),
            dtype: dt,
        },
        TypedPtr {
            ptr: dyt.cached_ptr(),
            dtype: dt,
        },
        TypedPtr {
            ptr: wt.cached_ptr(),
            dtype: dt,
        },
    );
    if full {
        bi_sgemm_backward_dx_typed(&t.ctx, dxtp, dytp, wtp, (m, k, n)).unwrap();
    } else {
        gemm_bi_triad::sgemm_bi_backward_dx_typed(
            &t.ctx.stream,
            &t.ctx.kernels,
            dxtp,
            dytp,
            wtp,
            (m, k, n),
        )
        .unwrap();
    }
    t.ctx.stream.synchronize().unwrap();
    let mut got = vec![0.0f32; m * k];
    dxt.download_f32(&t.ctx.stream, &mut got).unwrap();

    assert_bits(&label, &got, &downcast_grid(&ref_dx, dt));
}

#[test]
fn typed_stage2_buckets_bit_match_f32_triad() {
    let t = Ctx::new();
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        // Forward: gemv (N=1), ultra-thin (M<32), narrow-small (M<=64),
        // narrow (M>64), each with and without bias.
        check_forward(&t, dt, (64, 384, 1), true, false);
        check_forward(&t, dt, (64, 384, 1), false, false);
        check_forward(&t, dt, (8, 384, 512), true, false);
        check_forward(&t, dt, (32, 96, 80), true, false);
        check_forward(&t, dt, (256, 96, 80), true, false);
        check_forward(&t, dt, (256, 96, 80), false, false);
        // K-tail (K % 16 != 0) inside narrow buckets.
        check_forward(&t, dt, (32, 100, 80), true, false);

        // dW: gemv (N=1) + narrow.
        check_dw(&t, dt, (64, 384, 1), false);
        check_dw(&t, dt, (256, 96, 80), false);
        check_dw(&t, dt, (250, 100, 80), false);

        // dX: gemv (N=1) + narrow.
        check_dx(&t, dt, (64, 384, 1), false);
        check_dx(&t, dt, (256, 96, 80), false);
        check_dx(&t, dt, (250, 100, 80), false);
    }
}

/// Full-coverage entries: shapes WITHOUT a native typed bucket must take
/// the upcast → f32 sgemm_bi → RNE-downcast fallback and still satisfy the
/// bit contract. Shapes mirror real training GEMMs (M = B·T, layer dims).
#[test]
fn typed_full_coverage_bit_match_f32_triad() {
    let t = Ctx::new();
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        // Forward: split-K32 (M 32..1024, K%32==0, N 128..2048), Big
        // (M>=128), K-tail inside Big, and a dt_proj-like tiny-K shape.
        check_forward(&t, dt, (64, 384, 512), true, true);
        check_forward(&t, dt, (256, 384, 512), true, true);
        check_forward(&t, dt, (256, 384, 512), false, true);
        check_forward(&t, dt, (2048, 768, 512), true, true);
        check_forward(&t, dt, (256, 100, 512), true, true);
        check_forward(&t, dt, (256, 8, 256), true, true);
        // Wide-N / wide-K gap-fill (audit blocker #2): Mamba-1 in_proj at
        // micro-batch — M ∈ [32,128) with N = 2·d_inner > 2048, and M < 32
        // with K = d_model > 2048 (d_model = 2560 class).
        check_forward(&t, dt, (64, 768, 3072), true, true);
        check_forward(&t, dt, (16, 2560, 5120), true, true);

        // dW: TN big/split-M shapes (dW stays f32 — direct bit compare).
        check_dw(&t, dt, (256, 384, 512), true);
        check_dw(&t, dt, (2048, 768, 512), true);
        check_dw(&t, dt, (250, 100, 512), true);
        check_dw(&t, dt, (256, 8, 256), true);
        check_dw(&t, dt, (64, 768, 3072), true);

        // dX: NT big/split-N shapes.
        check_dx(&t, dt, (256, 384, 512), true);
        check_dx(&t, dt, (2048, 768, 512), true);
        check_dx(&t, dt, (250, 100, 512), true);
        check_dx(&t, dt, (256, 8, 256), true);
        check_dx(&t, dt, (64, 768, 3072), true);
    }
}

/// Gate-drift sweep: the typed dispatch decides native-Big vs fallback via
/// predicates that MIRROR the f32 cascade (`nn/tn/nt_routes_to_big`). If a
/// predicate ever disagrees with the real cascade, the typed path runs a
/// different bucket than the f32 reference and the bits diverge — this
/// sweep walks the M/N gate boundaries (slim 512, splitk 1024/2048,
/// gap-fill 128, M_SLIM_FORCE 512) to catch exactly that.
#[test]
fn typed_gate_boundary_sweep_bit_match() {
    let t = Ctx::new();
    let dt = WeightDtype::Bf16;
    let k = 384usize;
    for m in [128usize, 256, 511, 512, 513, 1024, 1025, 2048] {
        for n in [128usize, 512, 513, 516, 2048, 2052, 3072] {
            check_forward(&t, dt, (m, k, n), true, true);
            check_dw(&t, dt, (m, k, n), true);
            check_dx(&t, dt, (m, k, n), true);
        }
    }
    // K-axis boundaries at a Big-routed (M, N).
    for kk in [96usize, 100, 768, 2560] {
        check_forward(&t, dt, (512, kk, 3072), true, true);
        check_dw(&t, dt, (512, kk, 3072), true);
        check_dx(&t, dt, (512, kk, 3072), true);
    }
}

/// Measures the upcast-fallback tax on Big training shapes: typed entry
/// (upcast → f32 kernel → downcast) vs the bare f32 kernel on pre-upcast
/// operands. The delta is the ceiling on what native typed Big kernels
/// (stage 3) could recover with a SLOWER-than-cp.async staging scheme.
#[test]
#[ignore] // wall-clock benchmark — run explicitly on a quiet GPU
fn bench_upcast_fallback_tax() {
    use std::time::Instant;
    let t = Ctx::new();
    let dt = WeightDtype::Bf16;
    // (M, K, N) Big-bucket training shapes.
    for (m, k, n) in [
        (2048usize, 768usize, 512usize),
        (2048, 768, 3072),
        (4096, 1536, 3072),
        (256, 384, 512),
    ] {
        let qx = quantize(&det(m * k, 11, 1.0), dt);
        let qw = quantize(&det(k * n, 22, 0.5), dt);
        let x32 = t.f32_buf(&qx);
        let w32 = t.f32_buf(&qw);
        let mut y32 = GpuBuffer::zeros(&t.ctx.stream, m * n).unwrap();
        let xt = t.typed_buf(&qx, dt);
        let wt = t.typed_buf(&qw, dt);
        let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();

        let iters = 50;
        // Bare f32 kernel (operands already f32).
        for _ in 0..3 {
            gemm_bi_triad::sgemm_bi_forward(
                &t.ctx.stream,
                &t.ctx.kernels,
                &mut y32,
                &x32,
                w32.cached_ptr(),
                0,
                (m, k, n),
            )
            .unwrap();
        }
        t.ctx.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            gemm_bi_triad::sgemm_bi_forward(
                &t.ctx.stream,
                &t.ctx.kernels,
                &mut y32,
                &x32,
                w32.cached_ptr(),
                0,
                (m, k, n),
            )
            .unwrap();
        }
        t.ctx.stream.synchronize().unwrap();
        let f32_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // Typed entry: upcast → same f32 kernel → RNE downcast.
        let run_typed = || {
            bi_sgemm_forward_typed(
                &t.ctx,
                TypedPtr {
                    ptr: yt.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: xt.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: wt.cached_ptr(),
                    dtype: dt,
                },
                0,
                (m, k, n),
            )
            .unwrap();
        };
        for _ in 0..3 {
            run_typed();
        }
        t.ctx.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            run_typed();
        }
        t.ctx.stream.synchronize().unwrap();
        let typed_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        eprintln!(
            "[M{m} K{k} N{n}] f32 kernel {f32_us:.1} us | typed fallback {typed_us:.1} us | \
             cast tax {:.1} us ({:.1}%)",
            typed_us - f32_us,
            (typed_us - f32_us) / f32_us * 100.0
        );
    }
}

/// Typed forward is batch-invariant WITHIN a dispatch bucket: row 0 of Y is
/// bit-identical across batch sizes that route to the same bucket, exactly
/// like the f32 triad (see `nn_forward_is_batch_invariant_within_bucket`).
/// Native bucket pair: M=1 vs M=16 (ultra-thin). Fallback pair: M=64 vs
/// M=256 (upcast → f32 split-K32 → RNE downcast; elementwise casts are
/// trivially batch-invariant, so the f32 kernel's invariance carries over).
#[test]
fn typed_forward_is_batch_invariant_within_bucket() {
    let t = Ctx::new();
    let (k, n) = (384usize, 512usize);
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        let row = quantize(&det(k, 42, 1.0), dt);
        let w_host = quantize(&det(k * n, 43, 0.5), dt);
        let wt = t.typed_buf(&w_host, dt);

        let run = |m: usize| -> Vec<f32> {
            // Row 0 = `row`; other rows vary (must not affect row 0).
            let mut x_host = quantize(&det(m * k, 100 + m as u32, 1.0), dt);
            x_host[..k].copy_from_slice(&row);
            let xt = t.typed_buf(&x_host, dt);
            let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
            bi_sgemm_forward_typed(
                &t.ctx,
                TypedPtr {
                    ptr: yt.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: xt.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: wt.cached_ptr(),
                    dtype: dt,
                },
                0,
                (m, k, n),
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            let mut out = vec![0.0f32; m * n];
            yt.download_f32(&t.ctx.stream, &mut out).unwrap();
            out[..n].to_vec()
        };

        // Native ultra-thin bucket: M in [1, 32).
        let y1 = run(1);
        let y16 = run(16);
        // Fallback split-K32 bucket: M in [32, 1024], K % 32 == 0.
        let y64 = run(64);
        let y256 = run(256);
        for (i, (&a, &b)) in y1.iter().zip(&y16).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{dt:?} ultra-thin bucket variance at col {i}: M=1 {a:?} vs M=16 {b:?}"
            );
        }
        for (i, (&a, &b)) in y64.iter().zip(&y256).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{dt:?} split-K fallback bucket variance at col {i}: M=64 {a:?} vs M=256 {b:?}"
            );
        }
    }
}

/// Classifier patch-embed (input_proj) GEMM shapes: fwd NN
/// [B*T=18468, P^2=1024, d_model=384] and its dW TN twin, through the
/// full-coverage entries the production path uses. Proves bucket/fallback
/// coverage under the bit contract for the vision-training GEMMs.
#[test]
fn typed_classifier_input_proj_shapes_bit_match() {
    let t = Ctx::new();
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        check_forward(&t, dt, (18468, 1024, 384), true, true);
        check_dw(&t, dt, (18468, 1024, 384), true);
        // Tiny-M + K-tail combo (the capture-regression trainer shape):
        // bt=4, input_dim=200, d_model=32.
        check_forward(&t, dt, (4, 200, 32), true, true);
        check_dw(&t, dt, (4, 200, 32), true);
    }
}

const TC_REPEATS: usize = 20;
const GUARD: f32 = -7.0;

fn assert_exact<T: Copy + std::fmt::Debug + Eq>(label: &str, got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
        assert_eq!(got, want, "{label}: mismatch at element {index}");
    }
}

#[derive(Clone, Copy, Debug)]
enum NnSchedule {
    Tile128,
    Tile64,
    Thin16,
}

#[derive(Clone, Copy, Debug)]
enum BackwardSchedule {
    Tile128,
    Tile64,
}

struct TypedSubview {
    storage: DtypedBuf,
    offset: usize,
    rows: usize,
    cols: usize,
    stride: usize,
    dtype: WeightDtype,
}

impl TypedSubview {
    fn new(
        t: &Ctx,
        logical: &[f32],
        rows: usize,
        cols: usize,
        stride: usize,
        offset: usize,
        dtype: WeightDtype,
    ) -> Self {
        assert_eq!(logical.len(), rows * cols);
        assert!(stride >= cols);
        let mut host = vec![GUARD; offset + rows * stride + 8];
        for row in 0..rows {
            host[offset + row * stride..offset + row * stride + cols]
                .copy_from_slice(&logical[row * cols..(row + 1) * cols]);
        }
        Self {
            storage: t.typed_buf(&host, dtype),
            offset,
            rows,
            cols,
            stride,
            dtype,
        }
    }

    fn ptr(&self) -> u64 {
        self.storage.cached_ptr() + (self.offset * self.dtype.size_bytes()) as u64
    }

    fn upload_logical(&self, t: &Ctx, logical: &[f32]) {
        assert_eq!(logical.len(), self.rows * self.cols);
        let mut host = vec![GUARD; self.offset + self.rows * self.stride + 8];
        for row in 0..self.rows {
            host[self.offset + row * self.stride..self.offset + row * self.stride + self.cols]
                .copy_from_slice(&logical[row * self.cols..(row + 1) * self.cols]);
        }
        self.storage.upload_f32(&t.ctx.stream, &host).unwrap();
    }

    fn logical_bits(&self, t: &Ctx) -> Vec<u16> {
        let mut host = vec![0.0; self.storage.len_elems()];
        self.storage.download_f32(&t.ctx.stream, &mut host).unwrap();
        let bits = |value: f32| match self.dtype {
            WeightDtype::Bf16 => bf16::from_f32(value).to_bits(),
            WeightDtype::F16 => f16::from_f32(value).to_bits(),
            WeightDtype::F32 => unreachable!("TC typed output must be 16-bit"),
        };
        let guard = bits(GUARD);
        let mut logical = Vec::with_capacity(self.rows * self.cols);
        for (index, &value) in host.iter().enumerate() {
            let in_row = index
                .checked_sub(self.offset)
                .map(|i| (i / self.stride, i % self.stride));
            if let Some((row, col)) =
                in_row.filter(|(row, col)| *row < self.rows && *col < self.cols)
            {
                logical.push(bits(value));
                assert_eq!(logical.len(), row * self.cols + col + 1);
            } else {
                assert_eq!(bits(value), guard, "typed guard changed at element {index}");
            }
        }
        logical
    }
}

struct F32Subview {
    storage: GpuBuffer,
    offset: usize,
    rows: usize,
    cols: usize,
    stride: usize,
}

impl F32Subview {
    fn new(
        t: &Ctx,
        logical: &[f32],
        rows: usize,
        cols: usize,
        stride: usize,
        offset: usize,
    ) -> Self {
        assert_eq!(logical.len(), rows * cols);
        assert!(stride >= cols);
        let mut host = vec![GUARD; offset + rows * stride + 8];
        for row in 0..rows {
            host[offset + row * stride..offset + row * stride + cols]
                .copy_from_slice(&logical[row * cols..(row + 1) * cols]);
        }
        Self {
            storage: t.f32_buf(&host),
            offset,
            rows,
            cols,
            stride,
        }
    }

    fn ptr(&self) -> u64 {
        self.storage.cached_ptr() + (self.offset * size_of::<f32>()) as u64
    }

    fn upload_logical(&mut self, t: &Ctx, logical: &[f32]) {
        assert_eq!(logical.len(), self.rows * self.cols);
        let mut host = vec![GUARD; self.offset + self.rows * self.stride + 8];
        for row in 0..self.rows {
            host[self.offset + row * self.stride..self.offset + row * self.stride + self.cols]
                .copy_from_slice(&logical[row * self.cols..(row + 1) * self.cols]);
        }
        self.storage.upload(&t.ctx.stream, &host).unwrap();
    }

    fn logical_bits(&self, t: &Ctx) -> Vec<u32> {
        let host = self.storage.to_cpu(&t.ctx.stream).unwrap();
        let mut logical = Vec::with_capacity(self.rows * self.cols);
        for (index, &value) in host.iter().enumerate() {
            let in_row = index
                .checked_sub(self.offset)
                .map(|i| (i / self.stride, i % self.stride));
            if let Some((row, col)) =
                in_row.filter(|(row, col)| *row < self.rows && *col < self.cols)
            {
                logical.push(value.to_bits());
                assert_eq!(logical.len(), row * self.cols + col + 1);
            } else {
                assert_eq!(
                    value.to_bits(),
                    GUARD.to_bits(),
                    "f32 guard changed at element {index}"
                );
            }
        }
        logical
    }
}

fn launch_tc_nn(
    t: &Ctx,
    schedule: NnSchedule,
    dtype: WeightDtype,
    c: u64,
    a: u64,
    b: u64,
    dims: (usize, usize, usize),
    strides: (usize, usize, usize),
    beta: f32,
) -> Result<(), String> {
    let (m, k, n) = dims;
    let (lda, ldb, ldc) = strides;
    let (function, bm, bn, threads, shared_mem_bytes) = match schedule {
        NnSchedule::Tile128 => (&t.ctx.kernels.sgemm_nn_tc_typed, 128, 128, 256, 71_680),
        NnSchedule::Tile64 => (&t.ctx.kernels.sgemm_nn_tc64_typed, 64, 64, 128, 0),
        NnSchedule::Thin16 => (&t.ctx.kernels.sgemm_nn_tc16_typed, 16, 32, 128, 0),
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            m.div_ceil(bm).checked_mul(n.div_ceil(bn)).unwrap() as u32,
            1,
            1,
        ),
        block_dim: (threads, 1, 1),
        shared_mem_bytes,
    };
    let bias = 0u64;
    let alpha = 1.0f32;
    let (m, n, k, lda, ldb, ldc) = (
        m as i32, n as i32, k as i32, lda as i32, ldb as i32, ldc as i32,
    );
    let mut launch = t.ctx.stream.launch_builder(function.get(dtype));
    launch.arg(&c);
    launch.arg(&a);
    launch.arg(&b);
    launch.arg(&bias);
    launch.arg(&alpha);
    launch.arg(&beta);
    launch.arg(&m);
    launch.arg(&n);
    launch.arg(&k);
    launch.arg(&lda);
    launch.arg(&ldb);
    launch.arg(&ldc);
    unsafe { launch.launch(cfg) }.map_err(|error| format!("{schedule:?} NN launch: {error:?}"))?;
    t.ctx
        .stream
        .synchronize()
        .map_err(|error| format!("{schedule:?} NN synchronize: {error:?}"))
}

fn launch_tc_tn(
    t: &Ctx,
    schedule: BackwardSchedule,
    dtype: WeightDtype,
    c: u64,
    a: u64,
    b: u64,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (m, k, n) = dims;
    let (function, edge, threads, shared_mem_bytes) = match schedule {
        BackwardSchedule::Tile128 => (&t.ctx.kernels.sgemm_tn_tc_typed, 128, 256, 69_632),
        BackwardSchedule::Tile64 => (&t.ctx.kernels.sgemm_tn_tc64_typed, 64, 128, 0),
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            k.div_ceil(edge).checked_mul(n.div_ceil(edge)).unwrap() as u32,
            1,
            1,
        ),
        block_dim: (threads, 1, 1),
        shared_mem_bytes,
    };
    let alpha = 1.0f32;
    let (m, k, n) = (m as i32, k as i32, n as i32);
    let mut launch = t.ctx.stream.launch_builder(function.get(dtype));
    launch.arg(&c);
    launch.arg(&a);
    launch.arg(&b);
    launch.arg(&alpha);
    launch.arg(&m);
    launch.arg(&k);
    launch.arg(&n);
    unsafe { launch.launch(cfg) }.map_err(|error| format!("{schedule:?} TN launch: {error:?}"))?;
    t.ctx
        .stream
        .synchronize()
        .map_err(|error| format!("{schedule:?} TN synchronize: {error:?}"))
}

fn launch_tc_nt(
    t: &Ctx,
    schedule: BackwardSchedule,
    dtype: WeightDtype,
    c: u64,
    a: u64,
    b: u64,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (m, k, n) = dims;
    let (function, edge, threads, shared_mem_bytes) = match schedule {
        BackwardSchedule::Tile128 => (&t.ctx.kernels.sgemm_nt_tc_typed, 128, 256, 73_728),
        BackwardSchedule::Tile64 => (&t.ctx.kernels.sgemm_nt_tc64_typed, 64, 128, 0),
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            m.div_ceil(edge).checked_mul(k.div_ceil(edge)).unwrap() as u32,
            1,
            1,
        ),
        block_dim: (threads, 1, 1),
        shared_mem_bytes,
    };
    let alpha = 1.0f32;
    let (m, n, k) = (m as i32, n as i32, k as i32);
    let mut launch = t.ctx.stream.launch_builder(function.get(dtype));
    launch.arg(&c);
    launch.arg(&a);
    launch.arg(&b);
    launch.arg(&alpha);
    launch.arg(&m);
    launch.arg(&n);
    launch.arg(&k);
    unsafe { launch.launch(cfg) }.map_err(|error| format!("{schedule:?} NT launch: {error:?}"))?;
    t.ctx
        .stream
        .synchronize()
        .map_err(|error| format!("{schedule:?} NT synchronize: {error:?}"))
}

#[test]
fn tc128_packed_epilogues_match_scalar_fallback_bytes() {
    let t = Ctx::new();
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let nn_dims = (129usize, 65usize, 129usize);
        let nn_strides = (72usize, 136usize, 130usize);
        let nn_a = TypedSubview::new(
            &t,
            &det(nn_dims.0 * nn_dims.1, 801, 0.5),
            nn_dims.0,
            nn_dims.1,
            nn_strides.0,
            0,
            dtype,
        );
        let nn_b = TypedSubview::new(
            &t,
            &det(nn_dims.1 * nn_dims.2, 802, 0.5),
            nn_dims.1,
            nn_dims.2,
            nn_strides.1,
            0,
            dtype,
        );
        let zero_nn = vec![0.0; nn_dims.0 * nn_dims.2];
        let nn_aligned =
            TypedSubview::new(&t, &zero_nn, nn_dims.0, nn_dims.2, nn_strides.2, 0, dtype);
        launch_tc_nn(
            &t,
            NnSchedule::Tile128,
            dtype,
            nn_aligned.ptr(),
            nn_a.ptr(),
            nn_b.ptr(),
            nn_dims,
            nn_strides,
            0.0,
        )
        .unwrap();
        let nn_want = nn_aligned.logical_bits(&t);

        for (offset, ldc) in [(1usize, 130usize), (0, 131)] {
            let nn_scalar =
                TypedSubview::new(&t, &zero_nn, nn_dims.0, nn_dims.2, ldc, offset, dtype);
            for repeat in 0..TC_REPEATS {
                nn_scalar.upload_logical(&t, &zero_nn);
                launch_tc_nn(
                    &t,
                    NnSchedule::Tile128,
                    dtype,
                    nn_scalar.ptr(),
                    nn_a.ptr(),
                    nn_b.ptr(),
                    nn_dims,
                    (nn_strides.0, nn_strides.1, ldc),
                    0.0,
                )
                .unwrap_or_else(|error| {
                    panic!("{dtype:?} NN offset={offset} ldc={ldc} repeat={repeat}: {error}")
                });
                assert_exact(
                    &format!("{dtype:?} NN offset={offset} ldc={ldc} repeat={repeat}"),
                    &nn_scalar.logical_bits(&t),
                    &nn_want,
                );
            }
        }

        let nn_initial = det(nn_dims.0 * nn_dims.2, 803, 0.125);
        let nn_beta_aligned = TypedSubview::new(
            &t,
            &nn_initial,
            nn_dims.0,
            nn_dims.2,
            nn_strides.2,
            0,
            dtype,
        );
        launch_tc_nn(
            &t,
            NnSchedule::Tile128,
            dtype,
            nn_beta_aligned.ptr(),
            nn_a.ptr(),
            nn_b.ptr(),
            nn_dims,
            nn_strides,
            0.5,
        )
        .unwrap();
        let nn_beta_want = nn_beta_aligned.logical_bits(&t);
        let nn_beta_scalar = TypedSubview::new(
            &t,
            &nn_initial,
            nn_dims.0,
            nn_dims.2,
            nn_strides.2,
            1,
            dtype,
        );
        for repeat in 0..TC_REPEATS {
            nn_beta_scalar.upload_logical(&t, &nn_initial);
            launch_tc_nn(
                &t,
                NnSchedule::Tile128,
                dtype,
                nn_beta_scalar.ptr(),
                nn_a.ptr(),
                nn_b.ptr(),
                nn_dims,
                nn_strides,
                0.5,
            )
            .unwrap_or_else(|error| panic!("{dtype:?} NN beta repeat={repeat}: {error}"));
            assert_exact(
                &format!("{dtype:?} NN beta offset=1 repeat={repeat}"),
                &nn_beta_scalar.logical_bits(&t),
                &nn_beta_want,
            );
        }

        let tn_dims = (65usize, 128usize, 128usize);
        let tn_a = TypedSubview::new(
            &t,
            &det(tn_dims.0 * tn_dims.1, 811, 0.5),
            tn_dims.0,
            tn_dims.1,
            tn_dims.1,
            0,
            dtype,
        );
        let tn_b = TypedSubview::new(
            &t,
            &det(tn_dims.0 * tn_dims.2, 812, 0.5),
            tn_dims.0,
            tn_dims.2,
            tn_dims.2,
            0,
            dtype,
        );
        let tn_initial = det(tn_dims.1 * tn_dims.2, 813, 0.125);
        let tn_aligned = F32Subview::new(&t, &tn_initial, tn_dims.1, tn_dims.2, tn_dims.2, 0);
        launch_tc_tn(
            &t,
            BackwardSchedule::Tile128,
            dtype,
            tn_aligned.ptr(),
            tn_a.ptr(),
            tn_b.ptr(),
            tn_dims,
        )
        .unwrap();
        let tn_want = tn_aligned.logical_bits(&t);
        let mut tn_scalar = F32Subview::new(&t, &tn_initial, tn_dims.1, tn_dims.2, tn_dims.2, 1);
        for repeat in 0..TC_REPEATS {
            tn_scalar.upload_logical(&t, &tn_initial);
            launch_tc_tn(
                &t,
                BackwardSchedule::Tile128,
                dtype,
                tn_scalar.ptr(),
                tn_a.ptr(),
                tn_b.ptr(),
                tn_dims,
            )
            .unwrap_or_else(|error| panic!("{dtype:?} TN offset=1 repeat={repeat}: {error}"));
            assert_exact(
                &format!("{dtype:?} TN offset=1 repeat={repeat}"),
                &tn_scalar.logical_bits(&t),
                &tn_want,
            );
        }

        let nt_dims = (129usize, 128usize, 72usize);
        let nt_a = TypedSubview::new(
            &t,
            &det(nt_dims.0 * nt_dims.2, 821, 0.5),
            nt_dims.0,
            nt_dims.2,
            nt_dims.2,
            0,
            dtype,
        );
        let nt_b = TypedSubview::new(
            &t,
            &det(nt_dims.1 * nt_dims.2, 822, 0.5),
            nt_dims.1,
            nt_dims.2,
            nt_dims.2,
            0,
            dtype,
        );
        let zero_nt = vec![0.0; nt_dims.0 * nt_dims.1];
        let nt_aligned = TypedSubview::new(&t, &zero_nt, nt_dims.0, nt_dims.1, nt_dims.1, 0, dtype);
        launch_tc_nt(
            &t,
            BackwardSchedule::Tile128,
            dtype,
            nt_aligned.ptr(),
            nt_a.ptr(),
            nt_b.ptr(),
            nt_dims,
        )
        .unwrap();
        let nt_want = nt_aligned.logical_bits(&t);
        let nt_scalar = TypedSubview::new(&t, &zero_nt, nt_dims.0, nt_dims.1, nt_dims.1, 1, dtype);
        for repeat in 0..TC_REPEATS {
            nt_scalar.upload_logical(&t, &zero_nt);
            launch_tc_nt(
                &t,
                BackwardSchedule::Tile128,
                dtype,
                nt_scalar.ptr(),
                nt_a.ptr(),
                nt_b.ptr(),
                nt_dims,
            )
            .unwrap_or_else(|error| panic!("{dtype:?} NT offset=1 repeat={repeat}: {error}"));
            assert_exact(
                &format!("{dtype:?} NT offset=1 repeat={repeat}"),
                &nt_scalar.logical_bits(&t),
                &nt_want,
            );
        }
    }
}

#[test]
fn tc_cp_async_misaligned_operands_match_scalar_stage_bytes() {
    let t = Ctx::new();
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for schedule in [NnSchedule::Tile128, NnSchedule::Tile64, NnSchedule::Thin16] {
            let (dims, strides) = match schedule {
                NnSchedule::Tile128 | NnSchedule::Tile64 => ((129, 65, 129), (72, 136, 130)),
                NnSchedule::Thin16 => ((17, 9, 33), (16, 40, 34)),
            };
            let a_data = det(dims.0 * dims.1, 831, 0.5);
            let b_data = det(dims.1 * dims.2, 832, 0.5);
            let zero = vec![0.0; dims.0 * dims.2];
            let a_aligned = TypedSubview::new(&t, &a_data, dims.0, dims.1, strides.0, 0, dtype);
            let b_aligned = TypedSubview::new(&t, &b_data, dims.1, dims.2, strides.1, 0, dtype);
            let c_reference = TypedSubview::new(&t, &zero, dims.0, dims.2, strides.2, 0, dtype);
            launch_tc_nn(
                &t,
                schedule,
                dtype,
                c_reference.ptr(),
                a_aligned.ptr(),
                b_aligned.ptr(),
                dims,
                strides,
                0.0,
            )
            .unwrap();
            let want = c_reference.logical_bits(&t);

            for (a_offset, b_offset) in [(1usize, 0usize), (0, 1)] {
                let a = TypedSubview::new(&t, &a_data, dims.0, dims.1, strides.0, a_offset, dtype);
                let b = TypedSubview::new(&t, &b_data, dims.1, dims.2, strides.1, b_offset, dtype);
                let c = TypedSubview::new(&t, &zero, dims.0, dims.2, strides.2, 0, dtype);
                for repeat in 0..TC_REPEATS {
                    c.upload_logical(&t, &zero);
                    launch_tc_nn(&t, schedule, dtype, c.ptr(), a.ptr(), b.ptr(), dims, strides, 0.0)
                        .unwrap_or_else(|error| panic!(
                            "{dtype:?} {schedule:?} A+{a_offset} B+{b_offset} repeat={repeat}: {error}"
                        ));
                    assert_exact(
                        &format!(
                            "{dtype:?} {schedule:?} A+{a_offset} B+{b_offset} repeat={repeat}"
                        ),
                        &c.logical_bits(&t),
                        &want,
                    );
                }
            }
        }

        let tn_dims = (65usize, 128usize, 128usize);
        let tn_a_data = det(tn_dims.0 * tn_dims.1, 841, 0.5);
        let tn_b_data = det(tn_dims.0 * tn_dims.2, 842, 0.5);
        let tn_initial = det(tn_dims.1 * tn_dims.2, 843, 0.125);
        for schedule in [BackwardSchedule::Tile128, BackwardSchedule::Tile64] {
            let a_aligned =
                TypedSubview::new(&t, &tn_a_data, tn_dims.0, tn_dims.1, tn_dims.1, 0, dtype);
            let b_aligned =
                TypedSubview::new(&t, &tn_b_data, tn_dims.0, tn_dims.2, tn_dims.2, 0, dtype);
            let reference = F32Subview::new(&t, &tn_initial, tn_dims.1, tn_dims.2, tn_dims.2, 0);
            launch_tc_tn(
                &t,
                schedule,
                dtype,
                reference.ptr(),
                a_aligned.ptr(),
                b_aligned.ptr(),
                tn_dims,
            )
            .unwrap();
            let want = reference.logical_bits(&t);
            for (a_offset, b_offset) in [(1usize, 0usize), (0, 1)] {
                let a = TypedSubview::new(
                    &t, &tn_a_data, tn_dims.0, tn_dims.1, tn_dims.1, a_offset, dtype,
                );
                let b = TypedSubview::new(
                    &t, &tn_b_data, tn_dims.0, tn_dims.2, tn_dims.2, b_offset, dtype,
                );
                let mut c = F32Subview::new(&t, &tn_initial, tn_dims.1, tn_dims.2, tn_dims.2, 0);
                for repeat in 0..TC_REPEATS {
                    c.upload_logical(&t, &tn_initial);
                    launch_tc_tn(&t, schedule, dtype, c.ptr(), a.ptr(), b.ptr(), tn_dims)
                        .unwrap_or_else(|error| panic!(
                            "{dtype:?} {schedule:?} TN A+{a_offset} B+{b_offset} repeat={repeat}: {error}"
                        ));
                    assert_exact(
                        &format!(
                            "{dtype:?} {schedule:?} TN A+{a_offset} B+{b_offset} repeat={repeat}"
                        ),
                        &c.logical_bits(&t),
                        &want,
                    );
                }
            }
        }

        let nt_dims = (129usize, 129usize, 72usize);
        let nt_a_data = det(nt_dims.0 * nt_dims.2, 851, 0.5);
        let nt_b_data = det(nt_dims.1 * nt_dims.2, 852, 0.5);
        let zero = vec![0.0; nt_dims.0 * nt_dims.1];
        for schedule in [BackwardSchedule::Tile128, BackwardSchedule::Tile64] {
            let a_aligned =
                TypedSubview::new(&t, &nt_a_data, nt_dims.0, nt_dims.2, nt_dims.2, 0, dtype);
            let b_aligned =
                TypedSubview::new(&t, &nt_b_data, nt_dims.1, nt_dims.2, nt_dims.2, 0, dtype);
            let reference = TypedSubview::new(&t, &zero, nt_dims.0, nt_dims.1, nt_dims.1, 0, dtype);
            launch_tc_nt(
                &t,
                schedule,
                dtype,
                reference.ptr(),
                a_aligned.ptr(),
                b_aligned.ptr(),
                nt_dims,
            )
            .unwrap();
            let want = reference.logical_bits(&t);
            for (a_offset, b_offset) in [(1usize, 0usize), (0, 1)] {
                let a = TypedSubview::new(
                    &t, &nt_a_data, nt_dims.0, nt_dims.2, nt_dims.2, a_offset, dtype,
                );
                let b = TypedSubview::new(
                    &t, &nt_b_data, nt_dims.1, nt_dims.2, nt_dims.2, b_offset, dtype,
                );
                let c = TypedSubview::new(&t, &zero, nt_dims.0, nt_dims.1, nt_dims.1, 0, dtype);
                for repeat in 0..TC_REPEATS {
                    c.upload_logical(&t, &zero);
                    launch_tc_nt(&t, schedule, dtype, c.ptr(), a.ptr(), b.ptr(), nt_dims)
                        .unwrap_or_else(|error| panic!(
                            "{dtype:?} {schedule:?} NT A+{a_offset} B+{b_offset} repeat={repeat}: {error}"
                        ));
                    assert_exact(
                        &format!(
                            "{dtype:?} {schedule:?} NT A+{a_offset} B+{b_offset} repeat={repeat}"
                        ),
                        &c.logical_bits(&t),
                        &want,
                    );
                }
            }
        }
    }
}

#[test]
fn tc_source_centralizes_async_copy_and_avoids_type_punned_stores() {
    let source = include_str!("../kernels/gemm_bi_triad.cu");
    let tc_source = source
        .split_once("#define TC_BM")
        .expect("tensor-core section marker")
        .1;

    assert_eq!(
        tc_source.matches("cp.async.ca.shared.global").count(),
        1,
        "tensor-core async copies must go through sgb_cp_async_16_zfill"
    );
    assert!(!tc_source.contains("*(unsigned *)&C"));
    assert!(!tc_source.contains("float2 *dst"));
}
