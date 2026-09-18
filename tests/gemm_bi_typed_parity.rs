//! Typed (bf16/f16) gemm_bi — bit parity vs the f32 triad.
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
//! - full-coverage `gemm_bi_*_typed` blas entries — uncovered shapes route
//!   through the upcast → f32 gemm_bi → RNE-downcast fallback, which must
//!   satisfy the SAME bit contract.

#![cfg(feature = "cuda")]

use cudarc::driver::PushKernelArg;
use half::{bf16, f16};
use mamba_rs::mamba_ssm::gpu::blas::{
    TypedPtr, gemm_bi_backward_dw_typed, gemm_bi_backward_dx_typed, gemm_bi_forward_typed,
};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad;
use std::mem::size_of;

#[path = "support/triad_half_tile_screen.rs"]
mod triad_half_tile_screen;

use mamba_rs::mamba_ssm::gpu::GemmMode;
use triad_half_tile_screen::FixedHalfS3Params;

unsafe impl cudarc::driver::DeviceRepr for FixedHalfS3Params {}

fn cuda_braced_scope_after<'a>(source: &'a str, marker: &str) -> &'a str {
    let marker_start = source.find(marker).expect("CUDA scope marker");
    let marker_source = &source[marker_start..];
    let open = marker_source.find('{').expect("CUDA scope opening brace");
    let mut depth = 0usize;
    for (offset, byte) in marker_source[open..].bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &marker_source[..open + offset + 1];
                }
            }
            _ => {}
        }
    }
    panic!("CUDA scope closing brace");
}

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
        WeightDtype::F32 | WeightDtype::Tf32 => v.to_vec(),
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
        // Pin the tier explicitly: a shell with
        // MAMBA_RS_BI_TENSOR_CORES exported used to silently reroute this
        // suite through the TC kernels, testing a different contract.
        ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        ctx.route_controls().set_tensor_cores(false);
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
/// `full` routes through the blas-layer `gemm_bi_forward_typed` (native
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
    gemm_bi_triad::gemm_bi_forward(
        &t.ctx,
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
        gemm_bi_forward_typed(&t.ctx, ytp, xtp, wtp, bias_ptr, (m, k, n)).unwrap();
    } else {
        gemm_bi_triad::gemm_bi_forward_typed_native(
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
    gemm_bi_triad::gemm_bi_backward_dw(&t.ctx, dw32.cached_ptr(), &dy32, &x32, (m, k, n)).unwrap();
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
        gemm_bi_backward_dw_typed(&t.ctx, dwt.cached_ptr(), dytp, xtp, (m, k, n)).unwrap();
    } else {
        gemm_bi_triad::gemm_bi_backward_dw_typed_native(
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
    gemm_bi_triad::gemm_bi_backward_dx(&t.ctx, &mut dx32, &dy32, w32.cached_ptr(), (m, k, n))
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
        gemm_bi_backward_dx_typed(&t.ctx, dxtp, dytp, wtp, (m, k, n)).unwrap();
    } else {
        gemm_bi_triad::gemm_bi_backward_dx_typed_native(
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
/// the upcast → f32 gemm_bi → RNE-downcast fallback and still satisfy the
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
        // Wide-N and wide-K shapes: Mamba-1 in_proj at
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

#[test]
fn fixed_family_keeps_uncovered_typed_scalar_fallbacks() {
    let t = Ctx::new();
    t.ctx.route_controls().set_family(BiGemmFamily::Inference);
    let dims = (64, 128, 128);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        check_forward(&t, dtype, dims, true, true);
        check_dw(&t, dtype, dims, true);
        check_dx(&t, dtype, dims, true);
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
            gemm_bi_forward_typed(
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
            WeightDtype::F32 | WeightDtype::Tf32 => unreachable!("TC typed output must be 16-bit"),
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

fn tc_nn_operands(dtype: WeightDtype, c: u64, a: u64, b: u64) -> gemm_bi_triad::TcFwdOperands {
    gemm_bi_triad::TcFwdOperands {
        y: TypedPtr { ptr: c, dtype },
        x: TypedPtr { ptr: a, dtype },
        w: TypedPtr { ptr: b, dtype },
        bias_ptr: 0,
    }
}

fn enqueue_tc_nn(
    t: &Ctx,
    schedule: NnSchedule,
    operands: &gemm_bi_triad::TcFwdOperands,
    dims: (usize, usize, usize),
    strides: (usize, usize, usize),
    beta: f32,
) -> Result<(), String> {
    let (m, k, n) = dims;
    let (lda, ldb, ldc) = strides;
    let (function, bm, bn, threads, shared_mem_bytes) = match schedule {
        NnSchedule::Tile128 => (&t.ctx.kernels.gemm_bi_nn_tc_typed, 128, 128, 256, 71_680),
        NnSchedule::Tile64 => (&t.ctx.kernels.gemm_bi_nn_tc64_typed, 64, 64, 128, 0),
        NnSchedule::Thin16 => (&t.ctx.kernels.gemm_bi_nn_tc16_typed, 16, 32, 128, 0),
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
    let alpha = 1.0f32;
    let (m, n, k, lda, ldb, ldc) = (
        m as i32, n as i32, k as i32, lda as i32, ldb as i32, ldc as i32,
    );
    let mut launch = t.ctx.stream.launch_builder(function.get(operands.y.dtype));
    launch.arg(&operands.y.ptr);
    launch.arg(&operands.x.ptr);
    launch.arg(&operands.w.ptr);
    launch.arg(&operands.bias_ptr);
    launch.arg(&alpha);
    launch.arg(&beta);
    launch.arg(&m);
    launch.arg(&n);
    launch.arg(&k);
    launch.arg(&lda);
    launch.arg(&ldb);
    launch.arg(&ldc);
    unsafe { launch.launch(cfg) }
        .map(|_| ())
        .map_err(|error| format!("{schedule:?} NN launch: {error:?}"))
}

fn launch_tc_nn(
    t: &Ctx,
    schedule: NnSchedule,
    operands: &gemm_bi_triad::TcFwdOperands,
    dims: (usize, usize, usize),
    strides: (usize, usize, usize),
    beta: f32,
) -> Result<(), String> {
    enqueue_tc_nn(t, schedule, operands, dims, strides, beta)?;
    t.ctx
        .stream
        .synchronize()
        .map_err(|error| format!("{schedule:?} NN synchronize: {error:?}"))
}

fn enqueue_tc_tn(
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
        BackwardSchedule::Tile128 => (&t.ctx.kernels.gemm_bi_tn_tc_typed, 128, 256, 69_632),
        BackwardSchedule::Tile64 => (&t.ctx.kernels.gemm_bi_tn_tc64_typed, 64, 128, 0),
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
    unsafe { launch.launch(cfg) }
        .map(|_| ())
        .map_err(|error| format!("{schedule:?} TN launch: {error:?}"))
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
    enqueue_tc_tn(t, schedule, dtype, c, a, b, dims)?;
    t.ctx
        .stream
        .synchronize()
        .map_err(|error| format!("{schedule:?} TN synchronize: {error:?}"))
}

fn launch_scalar_tn_slim(
    t: &Ctx,
    c: u64,
    a: u64,
    b: u64,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (m, k, n) = dims;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (
            k.div_ceil(128).checked_mul(n.div_ceil(64)).unwrap() as u32,
            1,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let alpha = 1.0f32;
    let (m, k, n) = (m as i32, k as i32, n as i32);
    let mut launch = t.ctx.stream.launch_builder(&t.ctx.kernels.gemm_bi_tn_slim);
    launch.arg(&c);
    launch.arg(&a);
    launch.arg(&b);
    launch.arg(&alpha);
    launch.arg(&m);
    launch.arg(&k);
    launch.arg(&n);
    unsafe { launch.launch(config) }
        .map_err(|error| format!("scalar TN slim launch: {error:?}"))?;
    t.ctx
        .stream
        .synchronize()
        .map_err(|error| format!("scalar TN slim synchronize: {error:?}"))
}

/// The three device pointers of one NT tensor-core launch.
#[derive(Clone, Copy)]
struct NtOperands {
    c: u64,
    a: u64,
    b: u64,
}

fn enqueue_tc_nt(
    t: &Ctx,
    schedule: BackwardSchedule,
    dtype: WeightDtype,
    c: u64,
    a: u64,
    b: u64,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    enqueue_tc_nt_with_alpha(t, schedule, dtype, NtOperands { c, a, b }, dims, 1.0)
}

fn enqueue_tc_nt_with_alpha(
    t: &Ctx,
    schedule: BackwardSchedule,
    dtype: WeightDtype,
    operands: NtOperands,
    dims: (usize, usize, usize),
    alpha: f32,
) -> Result<(), String> {
    let NtOperands { c, a, b } = operands;
    let (m, k, n) = dims;
    let (function, edge, threads, shared_mem_bytes) = match schedule {
        BackwardSchedule::Tile128 => (&t.ctx.kernels.gemm_bi_nt_tc_typed, 128, 256, 73_728),
        BackwardSchedule::Tile64 => (&t.ctx.kernels.gemm_bi_nt_tc64_typed, 64, 128, 0),
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
    let (m, n, k) = (m as i32, n as i32, k as i32);
    let mut launch = t.ctx.stream.launch_builder(function.get(dtype));
    launch.arg(&c);
    launch.arg(&a);
    launch.arg(&b);
    launch.arg(&alpha);
    launch.arg(&m);
    launch.arg(&n);
    launch.arg(&k);
    unsafe { launch.launch(cfg) }
        .map(|_| ())
        .map_err(|error| format!("{schedule:?} NT launch: {error:?}"))
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
    enqueue_tc_nt(t, schedule, dtype, c, a, b, dims)?;
    t.ctx
        .stream
        .synchronize()
        .map_err(|error| format!("{schedule:?} NT synchronize: {error:?}"))
}

const ADA_HALF_NT_D768_OUT: (usize, usize, usize) = (2_048, 1_536, 768);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdaHalfNtCell {
    D768Out,
    D768In,
    Prism,
}

impl AdaHalfNtCell {
    const fn dims(self) -> (usize, usize, usize) {
        match self {
            Self::D768Out => ADA_HALF_NT_D768_OUT,
            Self::D768In => (2_048, 768, 3_072),
            Self::Prism => (4_621, 384, 1_928),
        }
    }

    fn fixed_s3_grid(self) -> u32 {
        let (m, k_out, _) = self.dims();
        m.div_ceil(128).checked_mul(k_out.div_ceil(128)).unwrap() as u32
    }
}

#[test]
fn ada_half_nt_cells_freeze_storage_strides_and_launch_grids() {
    for (cell, expected) in [
        (
            AdaHalfNtCell::D768Out,
            (
                (2_048, 1_536, 768),
                (1_572_864, 1_179_648, 3_145_728),
                (768, 768, 1_536),
                192,
                768,
            ),
        ),
        (
            AdaHalfNtCell::D768In,
            (
                (2_048, 768, 3_072),
                (6_291_456, 2_359_296, 1_572_864),
                (3_072, 3_072, 768),
                96,
                384,
            ),
        ),
        (
            AdaHalfNtCell::Prism,
            (
                (4_621, 384, 1_928),
                (8_909_288, 740_352, 1_774_464),
                (1_928, 1_928, 384),
                111,
                438,
            ),
        ),
    ] {
        let (m, k_out, reduction) = cell.dims();
        let lengths = (m * reduction, k_out * reduction, m * k_out);
        let strides = (reduction, reduction, k_out);
        let current_grid = m.div_ceil(64) * k_out.div_ceil(64);
        assert_eq!(
            (
                cell.dims(),
                lengths,
                strides,
                cell.fixed_s3_grid(),
                current_grid,
            ),
            expected
        );
    }
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
            &tc_nn_operands(dtype, nn_aligned.ptr(), nn_a.ptr(), nn_b.ptr()),
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
                    &tc_nn_operands(dtype, nn_scalar.ptr(), nn_a.ptr(), nn_b.ptr()),
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
            &tc_nn_operands(dtype, nn_beta_aligned.ptr(), nn_a.ptr(), nn_b.ptr()),
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
                &tc_nn_operands(dtype, nn_beta_scalar.ptr(), nn_a.ptr(), nn_b.ptr()),
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
fn scalar_tn_slim_output_subview_matches_the_aligned_result() {
    let t = Ctx::new();
    let dims = (65usize, 128usize, 128usize);
    let a = F32Subview::new(
        &t,
        &det(dims.0 * dims.1, 815, 0.5),
        dims.0,
        dims.1,
        dims.1,
        0,
    );
    let b = F32Subview::new(
        &t,
        &det(dims.0 * dims.2, 816, 0.5),
        dims.0,
        dims.2,
        dims.2,
        0,
    );
    let initial = det(dims.1 * dims.2, 817, 0.125);
    let aligned = F32Subview::new(&t, &initial, dims.1, dims.2, dims.2, 0);
    launch_scalar_tn_slim(&t, aligned.ptr(), a.ptr(), b.ptr(), dims).unwrap();
    let expected = aligned.logical_bits(&t);

    let shifted = F32Subview::new(&t, &initial, dims.1, dims.2, dims.2, 1);
    launch_scalar_tn_slim(&t, shifted.ptr(), a.ptr(), b.ptr(), dims).unwrap();
    assert_exact(
        "scalar TN slim output offset=1",
        &shifted.logical_bits(&t),
        &expected,
    );
}

#[test]
fn tc128_tn_odd_width_output_subviews_match_exactly() {
    let t = Ctx::new();
    let dims = (65usize, 128usize, 129usize);

    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let a = TypedSubview::new(
            &t,
            &det(dims.0 * dims.1, 823, 0.5),
            dims.0,
            dims.1,
            dims.1,
            0,
            dtype,
        );
        let b = TypedSubview::new(
            &t,
            &det(dims.0 * dims.2, 824, 0.5),
            dims.0,
            dims.2,
            dims.2,
            0,
            dtype,
        );
        let initial = det(dims.1 * dims.2, 825, 0.125);
        let mut aligned = F32Subview::new(&t, &initial, dims.1, dims.2, dims.2, 0);
        let mut shifted = F32Subview::new(&t, &initial, dims.1, dims.2, dims.2, 1);

        for repeat in 0..TC_REPEATS {
            aligned.upload_logical(&t, &initial);
            shifted.upload_logical(&t, &initial);
            launch_tc_tn(
                &t,
                BackwardSchedule::Tile128,
                dtype,
                aligned.ptr(),
                a.ptr(),
                b.ptr(),
                dims,
            )
            .unwrap_or_else(|error| panic!("{dtype:?} odd-N aligned repeat={repeat}: {error}"));
            launch_tc_tn(
                &t,
                BackwardSchedule::Tile128,
                dtype,
                shifted.ptr(),
                a.ptr(),
                b.ptr(),
                dims,
            )
            .unwrap_or_else(|error| panic!("{dtype:?} odd-N shifted repeat={repeat}: {error}"));
            assert_exact(
                &format!("{dtype:?} TN odd-N shifted repeat={repeat}"),
                &shifted.logical_bits(&t),
                &aligned.logical_bits(&t),
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
                &tc_nn_operands(dtype, c_reference.ptr(), a_aligned.ptr(), b_aligned.ptr()),
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
                    launch_tc_nn(
                        &t,
                        schedule,
                        &tc_nn_operands(dtype, c.ptr(), a.ptr(), b.ptr()),
                        dims,
                        strides,
                        0.0,
                    )
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
    let mma_source = include_str!("../kernels/gemm_bi_triad/mma16.cuh");
    let sm80_source = include_str!("../kernels/gemm_bi_triad/sm80/mma.cu");
    let (typed_sm80, tf32_sm80) = sm80_source
        .split_once("struct Sm80Tf32KernelParams")
        .expect("SM80 typed/TF32 source boundary");
    let source = [
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        mma_source,
        sm80_source,
    ]
    .concat();
    let typed_helper = cuda_braced_scope_after(mma_source, "void cp_async_16_zfill(");
    let typed_l2_helper = cuda_braced_scope_after(mma_source, "void cp_async_16_zfill_l2(");
    let tf32_stage = cuda_braced_scope_after(tf32_sm80, "void tf32_stage_async(");

    assert_eq!(
        mma_source.matches("void cp_async_16_zfill(").count(),
        1,
        "the shared typed async-copy helper must have one definition"
    );
    assert_eq!(
        typed_helper.matches("cp.async.ca.shared.global").count(),
        1,
        "the shared typed helper must own one cache-all cp.async opcode"
    );
    assert!(typed_helper.contains("[%0], [%1], 16, %2"));
    assert_eq!(
        mma_source.matches("void cp_async_16_zfill_l2(").count(),
        1,
        "the L2-only typed async-copy helper must have one definition"
    );
    assert_eq!(
        typed_l2_helper.matches("cp.async.cg.shared.global").count(),
        1,
        "the L2-only typed helper must own one cp.async opcode"
    );
    assert!(typed_l2_helper.contains("[%0], [%1], 16, %2"));
    assert!(mma_source.contains("return valid_bytes == 0 ? base : base + valid_offset;"));
    assert_eq!(
        typed_sm80.matches("cp.async.ca.shared.global").count(),
        0,
        "typed SM80 kernels must not bypass cp_async_16_zfill"
    );
    assert_eq!(
        typed_sm80.matches("cp.async.cg.shared.global").count(),
        0,
        "typed SM80 kernels must not bypass the named async-copy helpers"
    );
    let typed_copies = typed_sm80.matches("cp_async_16_zfill(").count()
        + typed_sm80.matches("cp_async_16_zfill_l2(").count();
    assert!(typed_copies > 0, "typed SM80 kernels must use async copies");
    assert_eq!(
        typed_copies,
        typed_sm80.matches("cp_async_source(").count(),
        "every typed async copy must select an in-allocation source"
    );
    assert_eq!(
        typed_copies,
        typed_sm80.matches("_bytes == 0 ? 0").count(),
        "every typed async copy must clamp its zero-byte integer offset"
    );
    assert_eq!(
        tf32_sm80
            .matches("void gemm_bi_tf32_cp_async_4_zfill(")
            .count(),
        0,
        "portable TF32 staging must not retain the scalar-width async path"
    );
    // The narrow-stride TF32 staging path issues four 4-byte copies from
    // one named helper; nothing else in the portable TF32 source may carry
    // the scalar-width opcode.
    assert_eq!(
        tf32_sm80.matches("void tf32_cp_async_4x4_zfill(").count(),
        1,
        "portable TF32 staging must own exactly one narrow-stride helper"
    );
    assert_eq!(
        tf32_sm80.matches("cp.async.ca.shared.global").count(),
        1,
        "portable TF32 staging must use only the named 16-byte helper"
    );
    let tf32_copy_helper = cuda_braced_scope_after(tf32_sm80, "void tf32_cp_async_16_zfill(");
    assert_eq!(
        tf32_sm80.matches("void tf32_cp_async_16_zfill(").count(),
        1,
        "portable TF32 staging must centralize its tile-aware cache policy"
    );
    assert!(tf32_copy_helper.contains("if constexpr (BM == 16)"));
    assert!(tf32_copy_helper.contains("cp_async_16_zfill("));
    assert!(tf32_copy_helper.contains("cp_async_16_zfill_l2("));
    // The staging loop routes every copy through the stride-aware wrapper,
    // which picks the 16-byte helper or the narrow 4x4 helper per call.
    let tf32_wide_copies = tf32_stage.matches("tf32_cp_async_zfill<").count();
    assert_eq!(tf32_wide_copies, 4);
    let tf32_wrapper = cuda_braced_scope_after(tf32_sm80, "void tf32_cp_async_zfill(");
    assert_eq!(
        tf32_wrapper.matches("tf32_cp_async_16_zfill<BM>(").count(),
        2
    );
    assert_eq!(tf32_wrapper.matches("tf32_cp_async_4x4_zfill(").count(), 2);
    let tf32_copies = tf32_wide_copies;
    assert_eq!(
        tf32_copies,
        tf32_stage
            .matches("long long safe_offset = _bytes == 0 ? 0 : valid_offset;")
            .count(),
        "every TF32 async copy must clamp its zero-byte integer offset"
    );
    assert_eq!(
        tf32_copies,
        tf32_stage.matches("cp_async_source(").count(),
        "every TF32 async copy must select an in-allocation source"
    );
    for line in tf32_stage
        .lines()
        .filter(|line| line.contains("cp_async_source("))
    {
        assert!(
            line.contains("safe_offset, _bytes"),
            "TF32 source selection bypasses its safe offset: {line}"
        );
    }
    for line in tf32_stage
        .lines()
        .filter(|line| line.contains("tf32_cp_async_zfill<"))
    {
        assert!(
            line.contains("dst, src, _bytes"),
            "TF32 async-copy call bypasses its selected source: {line}"
        );
    }
    let tf32_narrow_helper = cuda_braced_scope_after(tf32_sm80, "void tf32_cp_async_4x4_zfill(");
    assert_eq!(
        source.matches("cp.async.ca.shared.global").count(),
        typed_helper.matches("cp.async.ca.shared.global").count()
            + tf32_narrow_helper
                .matches("cp.async.ca.shared.global")
                .count(),
        "only the named helpers may own cache-all cp.async opcodes"
    );
    assert_eq!(
        source.matches("cp.async.cg.shared.global").count(),
        typed_l2_helper.matches("cp.async.cg.shared.global").count(),
        "only the L2-only typed helper may own an L2-cached cp.async opcode"
    );
    assert!(!source.contains("*(unsigned *)&C"));
    assert!(!source.contains("float2 *dst"));
}

#[test]
fn typed_native_api_names_are_distinct_from_full_policy_entries() {
    let launch = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/launch.rs");
    let blas = include_str!("../src/mamba_ssm/gpu/blas.rs");
    for operation in ["forward", "backward_dw", "backward_dx"] {
        let full = format!("pub fn gemm_bi_{operation}_typed(");
        let native = format!("pub fn gemm_bi_{operation}_typed_native(");
        assert!(blas.contains(&full), "missing full-policy API {full}");
        assert!(!blas.contains(&native), "BLAS layer exports {native}");
        assert!(launch.contains(&native), "missing native-only API {native}");
        assert!(
            !launch.contains(&full),
            "native dispatcher duplicates full-policy API {full}"
        );
    }
}

#[test]
fn tn_rect128x64_source_contract_is_forced_tn_ca_one_bank_bk32_s3() {
    let source = include_str!("../kernels/gemm_bi_triad/sm80/mma.cu");
    let marker = "#define GEMM_BI_TN_RECT_BM";
    let candidate = &source[source.find(marker).expect("TN Rect128x64 source marker")..];
    let candidate = candidate
        .split_once("// NT (dX) staging")
        .expect("Tile64 NT source boundary")
        .0;

    for required in [
        "#define GEMM_BI_TN_RECT_BM 128",
        "#define GEMM_BI_TN_RECT_BN 64",
        "#define GEMM_BI_TN_RECT_BK 32",
        "#define GEMM_BI_TN_RECT_STAGES 3",
        "#define GEMM_BI_TN_RECT_THREADS 256",
        "__launch_bounds__(256, 2)",
        "Xs[GEMM_BI_TN_RECT_STAGES][GEMM_BI_TN_RECT_BK][GEMM_BI_TN_RECT_LDX]",
        "Ys[GEMM_BI_TN_RECT_STAGES][GEMM_BI_TN_RECT_BK][GEMM_BI_TN_RECT_LDY]",
        "cp_async_16_zfill(_dst, _src, _bytes)",
        "cp.async.wait_group 1",
        "int macro64_tiles = (M_red - 1) / 64 + 1",
        "int k32_tiles = 2 * macro64_tiles",
        "unsigned a_frag[2][4]",
        "unsigned b_frag[4][2]",
        "GEMM_BI_DEFINE_GEMM_BI_TN_TC128X64(bf16",
        "GEMM_BI_DEFINE_GEMM_BI_TN_TC128X64(f16",
        "#undef GEMM_BI_DEFINE_GEMM_BI_TN_TC128X64",
        "#undef GEMM_BI_TN_RECT_BM",
    ] {
        assert!(
            candidate.contains(required),
            "TN Rect128x64 source is missing {required:?}"
        );
    }
    for forbidden in [
        "cp_async_16_zfill_l2",
        "cp.async.cg",
        "float2",
        "accumulate_float2_or_scalar",
        "atomic",
        "split",
        "REDUX",
        "nn_tc128x64",
        "nt_tc128x64",
        "a_frag_next",
        "b_frag_next",
    ] {
        assert!(
            !candidate.contains(forbidden),
            "TN Rect128x64 source contains forbidden {forbidden:?}"
        );
    }
}

#[test]
fn tc128_output_pointer_formation_is_column_guarded() {
    let source = [
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        include_str!("../kernels/gemm_bi_triad/mma16.cuh"),
        include_str!("../kernels/gemm_bi_triad/sm80/mma.cu"),
    ]
    .concat();
    let tc128_source = source
        .split_once("#define GEMM_BI_TC64_BM")
        .expect("Tile64 macro boundary")
        .0;

    assert_eq!(
        tc128_source.matches("output_start_if_valid(").count(),
        4,
        "the helper definition and all three TC128 epilogues must use the guarded output start"
    );
    assert!(
        !tc128_source.contains("= &C["),
        "TC128 epilogues must not form output pointers before validating c0"
    );
    let helper = tc128_source
        .split_once("T* output_start_if_valid(")
        .expect("guarded output helper")
        .1
        .split_once("__device__ __forceinline__ int cp_async_valid_elems")
        .expect("next device helper")
        .0;
    let guard = helper
        .find("if (column >= extent) return nullptr;")
        .expect("column guard");
    let pointer = helper
        .find("return base + row_offset + column;")
        .expect("output pointer expression");
    assert!(
        guard < pointer,
        "the column guard must precede pointer formation"
    );
}
