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

use half::{bf16, f16};
use mamba_rs::mamba_ssm::gpu::blas::{
    TypedPtr, bi_sgemm_backward_dw_typed, bi_sgemm_backward_dx_typed, bi_sgemm_forward_typed,
};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::sgemm_bi;

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
    sgemm_bi::sgemm_bi_forward(
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
        sgemm_bi::sgemm_bi_forward_typed(
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
    sgemm_bi::sgemm_bi_backward_dw(
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
        sgemm_bi::sgemm_bi_backward_dw_typed(
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
    sgemm_bi::sgemm_bi_backward_dx(
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
        sgemm_bi::sgemm_bi_backward_dx_typed(
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
