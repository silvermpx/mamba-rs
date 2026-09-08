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

use cudarc::driver::{CudaFunction, CudaGraph, PushKernelArg, sys};
use half::{bf16, f16};
use mamba_rs::mamba_ssm::gpu::blas::{
    TypedPtr, gemm_bi_backward_dw_typed, gemm_bi_backward_dx_typed, gemm_bi_forward_typed,
};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad;
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use sha2::{Digest as _, Sha256};
use std::mem::size_of;

mod common;
#[path = "support/triad_half_nn_n64_source.rs"]
mod triad_half_nn_n64_source;
#[path = "support/triad_half_nt_compact_source.rs"]
mod triad_half_nt_compact_source;
#[path = "support/triad_half_nt_s3_source.rs"]
mod triad_half_nt_s3_source;
#[path = "support/triad_half_tile_screen.rs"]
mod triad_half_tile_screen;
#[path = "support/triad_half_tn_compact_source.rs"]
mod triad_half_tn_compact_source;
#[path = "support/triad_half_tn_microtile_source.rs"]
mod triad_half_tn_microtile_source;
#[path = "support/triad_half_tn_s4_source.rs"]
mod triad_half_tn_s4_source;

use common::gpu_quiet::QuietGpu;
use triad_half_tile_screen::{
    AdaHalfArm, BracketOrder, FixedHalfS3Params, candidate_over_reference,
    percentile as ada_percentile, retain_decision,
};

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

    fn new_ada() -> Result<Self, String> {
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (8, 9) || identity.multiprocessor_count != 142 {
            return Err(format!(
                "typed NN tile discovery requires CC8.9/142 SM, found {:?}/{} SM",
                identity.compute_capability, identity.multiprocessor_count
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_tensor_cores(true);
        ctx.set_fast_gemm(false);
        Ok(Self { ctx })
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
    gemm_bi_triad::gemm_bi_backward_dw(
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
    gemm_bi_triad::gemm_bi_backward_dx(
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

#[test]
fn fixed_family_keeps_uncovered_typed_scalar_fallbacks() {
    let t = Ctx::new();
    t.ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
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
            gemm_bi_triad::gemm_bi_forward(
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
            gemm_bi_triad::gemm_bi_forward(
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

const ADA_HALF_D768_IN: (usize, usize, usize) = (2_048, 768, 3_072);
const ADA_HALF_WINDOWS: usize = 7;
const ADA_HALF_WARMUPS: usize = 4;

#[derive(Clone, Copy)]
enum AdaHalfPath {
    Eager,
    Graph,
}

impl AdaHalfPath {
    const fn name(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Graph => "graph",
        }
    }
}

fn ada_half_symbol(arm: AdaHalfArm, dtype: WeightDtype) -> &'static str {
    match (arm, dtype) {
        (AdaHalfArm::PortableTc64, WeightDtype::Bf16) => "gemm_bi_nn_tc64_bf16",
        (AdaHalfArm::PortableTc64, WeightDtype::F16) => "gemm_bi_nn_tc64_f16",
        (AdaHalfArm::PortableTc128, WeightDtype::Bf16) => "gemm_bi_nn_tc_bf16",
        (AdaHalfArm::PortableTc128, WeightDtype::F16) => "gemm_bi_nn_tc_f16",
        (AdaHalfArm::FixedSm89Tc128S3, WeightDtype::Bf16) => {
            "gemm_bi_nn_fixed_sm89_tc128_s3_v1_bf16"
        }
        (AdaHalfArm::FixedSm89Tc128S3, WeightDtype::F16) => "gemm_bi_nn_fixed_sm89_tc128_s3_v1_f16",
        (_, WeightDtype::F32) => panic!("Ada half discovery requires a half dtype"),
    }
}

fn ada_half_function(
    t: &Ctx,
    arm: AdaHalfArm,
    dtype: WeightDtype,
) -> Result<&CudaFunction, String> {
    match arm {
        AdaHalfArm::PortableTc64 => Ok(t.ctx.kernels.gemm_bi_nn_tc64_typed.get(dtype)),
        AdaHalfArm::PortableTc128 => Ok(t.ctx.kernels.gemm_bi_nn_tc_typed.get(dtype)),
        AdaHalfArm::FixedSm89Tc128S3 => t
            .ctx
            .kernels
            .fixed_sm89_half_s3
            .as_ref()
            .map(|holder| holder.get(dtype))
            .ok_or_else(|| {
                t.ctx
                    .kernels
                    .fixed_sm89_half_s3_rejection
                    .clone()
                    .unwrap_or_else(|| "Fixed SM89 half S3 holder is unavailable".into())
            }),
    }
}

fn ada_half_values(len: usize, dtype: WeightDtype, mut state: u32) -> Vec<f32> {
    assert_ne!(state, 0);
    (0..len)
        .map(|index| {
            let special = match (dtype, index) {
                (WeightDtype::Bf16, 0) => Some(bf16::from_bits(0x0000).to_f32()),
                (WeightDtype::Bf16, 1) => Some(bf16::from_bits(0x8000).to_f32()),
                (WeightDtype::Bf16, 2) => Some(bf16::from_bits(0x0001).to_f32()),
                (WeightDtype::Bf16, 3) => Some(bf16::from_bits(0x8001).to_f32()),
                (WeightDtype::F16, 0) => Some(f16::from_bits(0x0000).to_f32()),
                (WeightDtype::F16, 1) => Some(f16::from_bits(0x8000).to_f32()),
                (WeightDtype::F16, 2) => Some(f16::from_bits(0x0001).to_f32()),
                (WeightDtype::F16, 3) => Some(f16::from_bits(0x8001).to_f32()),
                _ => None,
            };
            special.unwrap_or_else(|| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let raw = (state & 0x1fff) as i32 - 4096;
                let value = raw as f32 / 8192.0;
                match dtype {
                    WeightDtype::Bf16 => bf16::from_f32(value).to_f32(),
                    WeightDtype::F16 => f16::from_f32(value).to_f32(),
                    WeightDtype::F32 => unreachable!("Ada half discovery requires a half dtype"),
                }
            })
        })
        .collect()
}

struct AdaHalfFixture {
    a: TypedSubview,
    b: TypedSubview,
    candidate: TypedSubview,
    reference: TypedSubview,
    fast: TypedSubview,
    a_values: Vec<f32>,
    b_values: Vec<f32>,
    output_seed: Vec<f32>,
    a_bits: Vec<u16>,
    b_bits: Vec<u16>,
    dtype: WeightDtype,
    dims: (usize, usize, usize),
}

impl AdaHalfFixture {
    fn new(t: &Ctx, dtype: WeightDtype, dims: (usize, usize, usize)) -> Self {
        Self::new_with_guard(t, dtype, dims, 8)
    }

    fn new_with_guard(
        t: &Ctx,
        dtype: WeightDtype,
        dims: (usize, usize, usize),
        aligned_guard_offset: usize,
    ) -> Self {
        let (m, k, n) = dims;
        let a_values = ada_half_values(m * k, dtype, 0xa89a_2001);
        let b_values = ada_half_values(k * n, dtype, 0xb89a_2002);
        let output_seed = ada_half_values(m * n, dtype, 0xc89a_2003);
        assert_eq!(aligned_guard_offset % 8, 0);
        let a = TypedSubview::new(t, &a_values, m, k, k, aligned_guard_offset, dtype);
        let b = TypedSubview::new(t, &b_values, k, n, n, aligned_guard_offset, dtype);
        let candidate = TypedSubview::new(t, &output_seed, m, n, n, aligned_guard_offset, dtype);
        let reference = TypedSubview::new(t, &output_seed, m, n, n, aligned_guard_offset, dtype);
        let fast = TypedSubview::new(t, &output_seed, m, n, n, aligned_guard_offset, dtype);
        let a_bits = a.logical_bits(t);
        let b_bits = b.logical_bits(t);
        Self {
            a,
            b,
            candidate,
            reference,
            fast,
            a_values,
            b_values,
            output_seed,
            a_bits,
            b_bits,
            dtype,
            dims,
        }
    }

    fn output(&self, arm: AdaHalfArm) -> &TypedSubview {
        match arm {
            AdaHalfArm::PortableTc128 => &self.reference,
            AdaHalfArm::PortableTc64 | AdaHalfArm::FixedSm89Tc128S3 => &self.candidate,
        }
    }

    fn operands(&self, arm: AdaHalfArm) -> gemm_bi_triad::TcFwdOperands {
        tc_nn_operands(
            self.dtype,
            self.output(arm).ptr(),
            self.a.ptr(),
            self.b.ptr(),
        )
    }

    fn reset_and_validate_inputs(&self, t: &Ctx, arm: AdaHalfArm) -> Result<(), String> {
        self.a.upload_logical(t, &self.a_values);
        self.b.upload_logical(t, &self.b_values);
        self.output(arm).upload_logical(t, &self.output_seed);
        t.ctx
            .stream
            .synchronize()
            .map_err(|error| format!("Ada half reset sync: {error:?}"))?;
        if self.a.logical_bits(t) != self.a_bits || self.b.logical_bits(t) != self.b_bits {
            return Err("Ada half input bits or guards changed".into());
        }
        Ok(())
    }

    fn validate_inputs(&self, t: &Ctx) -> Result<(), String> {
        if self.a.logical_bits(t) != self.a_bits || self.b.logical_bits(t) != self.b_bits {
            return Err("Ada half input bits or guards changed after launch".into());
        }
        Ok(())
    }

    fn reset_fast(&self, t: &Ctx) -> Result<(), String> {
        self.a.upload_logical(t, &self.a_values);
        self.b.upload_logical(t, &self.b_values);
        self.fast.upload_logical(t, &self.output_seed);
        t.ctx
            .stream
            .synchronize()
            .map_err(|error| format!("Ada half Fast reset sync: {error:?}"))?;
        self.validate_inputs(t)
    }
}

fn enqueue_ada_half(t: &Ctx, fixture: &AdaHalfFixture, arm: AdaHalfArm) -> Result<(), String> {
    let operands = fixture.operands(arm);
    let (m, k, n) = fixture.dims;
    match arm {
        AdaHalfArm::PortableTc64 => enqueue_tc_nn(
            t,
            NnSchedule::Tile64,
            &operands,
            fixture.dims,
            (k, n, n),
            0.0,
        ),
        AdaHalfArm::PortableTc128 => enqueue_tc_nn(
            t,
            NnSchedule::Tile128,
            &operands,
            fixture.dims,
            (k, n, n),
            0.0,
        ),
        AdaHalfArm::FixedSm89Tc128S3 => {
            let function = ada_half_function(t, arm, fixture.dtype)?;
            let bias = 0_u64;
            let params = FixedHalfS3Params {
                alpha: 1.0,
                beta: 0.0,
                m: m as i32,
                n: n as i32,
                k: k as i32,
                lda: k as i32,
                ldb: n as i32,
                ldc: n as i32,
            };
            let mut launch = t.ctx.stream.launch_builder(function);
            launch
                .arg(&operands.y.ptr)
                .arg(&operands.x.ptr)
                .arg(&operands.w.ptr)
                .arg(&bias)
                .arg(&params);
            unsafe {
                launch.launch(cudarc::driver::LaunchConfig {
                    grid_dim: (m.div_ceil(128) as u32 * n.div_ceil(128) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 98_304,
                })
            }
            .map(|_| ())
            .map_err(|error| format!("Fixed SM89 half S3 NN launch: {error:?}"))
        }
    }
}

fn ada_half_resource_gate(t: &Ctx, arm: AdaHalfArm, dtype: WeightDtype) -> Result<(), String> {
    let function = ada_half_function(t, arm, dtype)?;
    let (threads, expected_static, dynamic_shared, required_occupancy) = arm.resources();
    let registers = function
        .num_regs()
        .map_err(|error| format!("Ada half registers: {error:?}"))?;
    let local = function
        .local_size_bytes()
        .map_err(|error| format!("Ada half local bytes: {error:?}"))?;
    let static_shared = function
        .shared_size_bytes()
        .map_err(|error| format!("Ada half static shared: {error:?}"))?;
    let max_threads = function
        .max_threads_per_block()
        .map_err(|error| format!("Ada half max threads: {error:?}"))?;
    let occupancy = function
        .occupancy_max_active_blocks_per_multiprocessor(threads, dynamic_shared, None)
        .map_err(|error| format!("Ada half occupancy: {error:?}"))?;
    let symbol = ada_half_symbol(arm, dtype);
    println!(
        "{{\"schema\":\"MambaBiHalfNnTileAdaDiscoveryResourceV1\",\"dtype\":\"{dtype:?}\",\"symbol\":\"{symbol}\",\"threads\":{threads},\"registers\":{registers},\"local_bytes\":{local},\"static_shared_bytes\":{static_shared},\"dynamic_shared_bytes\":{dynamic_shared},\"max_threads\":{max_threads},\"occupancy\":{occupancy},\"required_occupancy\":{required_occupancy}}}"
    );
    if registers <= 0
        || local != 0
        || static_shared != expected_static
        || max_threads < threads as i32
        || occupancy < required_occupancy
    {
        return Err(format!(
            "Ada half {symbol} resource floor failed: regs={registers} local={local} static={static_shared}/{expected_static} max_threads={max_threads}/{threads} occupancy={occupancy}/{required_occupancy}"
        ));
    }
    Ok(())
}

fn capture_ada_half(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    arm: AdaHalfArm,
) -> Result<CudaGraph, String> {
    unsafe { capture_into_graph(&t.ctx.stream, || enqueue_ada_half(t, fixture, arm)) }
}

fn validate_single_node_graph(graph: &CudaGraph, label: &str) -> Result<(), String> {
    let mut count = 0_usize;
    let result =
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) };
    if result != sys::CUresult::CUDA_SUCCESS || count != 1 {
        return Err(format!(
            "Ada half {label} graph must contain exactly one kernel node: result={result:?} count={count}"
        ));
    }
    Ok(())
}

fn validate_nonempty_graph(graph: &CudaGraph, label: &str) -> Result<(), String> {
    let mut count = 0_usize;
    let result =
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) };
    if result != sys::CUresult::CUDA_SUCCESS || count == 0 {
        return Err(format!(
            "Ada half {label} graph must be nonempty: result={result:?} count={count}"
        ));
    }
    Ok(())
}

fn launch_ada_half_path(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    arm: AdaHalfArm,
    graph: &CudaGraph,
    path: AdaHalfPath,
) -> Result<(), String> {
    match path {
        AdaHalfPath::Eager => enqueue_ada_half(t, fixture, arm),
        AdaHalfPath::Graph => graph
            .launch()
            .map_err(|error| format!("Ada half graph launch: {error:?}")),
    }
}

fn observe_ada_half_many(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    arm: AdaHalfArm,
    graph: &CudaGraph,
    path: AdaHalfPath,
    gemms: usize,
    expected: Option<&[u16]>,
) -> Result<(f64, Vec<u16>), String> {
    fixture.reset_and_validate_inputs(t, arm)?;
    let start = t
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("Ada half timing start: {error:?}"))?;
    for _ in 0..gemms {
        launch_ada_half_path(t, fixture, arm, graph, path)?;
    }
    let end = t
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("Ada half timing end: {error:?}"))?;
    let elapsed_us = f64::from(
        start
            .elapsed_ms(&end)
            .map_err(|error| format!("Ada half timing: {error:?}"))?,
    ) * 1_000.0
        / gemms as f64;
    if !elapsed_us.is_finite() || elapsed_us <= 0.0 {
        return Err(format!("invalid Ada half timing sample {elapsed_us}"));
    }
    let bits = fixture.output(arm).logical_bits(t);
    if let Some(expected) = expected
        && bits != expected
    {
        let mismatch = bits
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
            .unwrap_or(bits.len());
        return Err(format!(
            "Ada half {} {} differs from forced TC128 at {mismatch}",
            arm.name(),
            path.name()
        ));
    }
    fixture.validate_inputs(t)?;
    Ok((elapsed_us, bits))
}

fn observe_ada_half(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    arm: AdaHalfArm,
    graph: &CudaGraph,
    path: AdaHalfPath,
    expected: Option<&[u16]>,
) -> Result<(f64, Vec<u16>), String> {
    observe_ada_half_many(t, fixture, arm, graph, path, 1, expected)
}

fn ada_half_screen_stratum(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    candidate: AdaHalfArm,
    candidate_graph: &CudaGraph,
    reference_graph: &CudaGraph,
    expected: &[u16],
    path: AdaHalfPath,
    order: BracketOrder,
) -> Result<[f64; 2], String> {
    for _ in 0..ADA_HALF_WARMUPS {
        observe_ada_half(t, fixture, candidate, candidate_graph, path, Some(expected))?;
        observe_ada_half(
            t,
            fixture,
            AdaHalfArm::PortableTc128,
            reference_graph,
            path,
            Some(expected),
        )?;
    }
    let arms = match order {
        BracketOrder::Abba => [
            (candidate, candidate_graph),
            (AdaHalfArm::PortableTc128, reference_graph),
            (AdaHalfArm::PortableTc128, reference_graph),
            (candidate, candidate_graph),
        ],
        BracketOrder::Baab => [
            (AdaHalfArm::PortableTc128, reference_graph),
            (candidate, candidate_graph),
            (candidate, candidate_graph),
            (AdaHalfArm::PortableTc128, reference_graph),
        ],
    };
    let mut observations = Vec::with_capacity(ADA_HALF_WINDOWS);
    let mut ratios = Vec::with_capacity(ADA_HALF_WINDOWS);
    for _ in 0..ADA_HALF_WINDOWS {
        let mut raw = [0.0; 4];
        for (index, (arm, graph)) in arms.into_iter().enumerate() {
            raw[index] = observe_ada_half(t, fixture, arm, graph, path, Some(expected))?.0;
        }
        ratios.push(candidate_over_reference(raw, order));
        observations.push(raw);
    }
    let p50 = ada_percentile(&ratios, 0.50).ok_or("invalid Ada half p50")?;
    let p95 = ada_percentile(&ratios, 0.95).ok_or("invalid Ada half p95")?;
    let order_name = match order {
        BracketOrder::Abba => "ABBA",
        BracketOrder::Baab => "BAAB",
    };
    let observations_json = format!(
        "[{}]",
        observations
            .iter()
            .map(|raw| format!("[{:.9},{:.9},{:.9},{:.9}]", raw[0], raw[1], raw[2], raw[3]))
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "{{\"schema\":\"MambaBiHalfNnTileAdaDiscoveryScreenV1\",\"dtype\":\"{:?}\",\"cell\":\"d768_in_proj\",\"shape\":[2048,768,3072],\"candidate\":\"{}\",\"candidate_symbol\":\"{}\",\"comparator\":\"forced_tc128\",\"comparator_symbol\":\"{}\",\"path\":\"{}\",\"order\":\"{order_name}\",\"windows\":{ADA_HALF_WINDOWS},\"warmups_per_arm\":{ADA_HALF_WARMUPS},\"logical_gemms_per_observation\":1,\"raw_observations_us\":{observations_json},\"ratio_direction\":\"candidate_over_forced_tc128\",\"ratio_p50\":{p50:.9},\"ratio_p95\":{p95:.9}}}",
        fixture.dtype,
        candidate.name(),
        ada_half_symbol(candidate, fixture.dtype),
        ada_half_symbol(AdaHalfArm::PortableTc128, fixture.dtype),
        path.name(),
    );
    Ok([p50, p95])
}

fn run_ada_half_nn_d768_in_screen(candidate: AdaHalfArm, cohort: &str) -> Result<(), String> {
    if candidate == AdaHalfArm::PortableTc128 {
        return Err("Ada half discovery candidate cannot be its own TC128 comparator".into());
    }
    assert!(
        !cfg!(debug_assertions),
        "Ada half discovery requires --release"
    );
    let quiet = QuietGpu::for_cuda_ordinal(0)?;
    let _pre = quiet.require_pre_context(&format!("{cohort}/pre-context"))?;
    let t = Ctx::new_ada()?;
    let _cohort = quiet.require_cohort(&format!("{cohort}/cohort"))?;
    let mut decisions = Vec::new();
    for dtype in [WeightDtype::F16, WeightDtype::Bf16] {
        ada_half_resource_gate(&t, candidate, dtype)?;
        ada_half_resource_gate(&t, AdaHalfArm::PortableTc128, dtype)?;
        let fixture = AdaHalfFixture::new(&t, dtype, ADA_HALF_D768_IN);
        let candidate_graph = capture_ada_half(&t, &fixture, candidate)?;
        let reference_graph = capture_ada_half(&t, &fixture, AdaHalfArm::PortableTc128)?;
        validate_single_node_graph(&candidate_graph, candidate.name())?;
        validate_single_node_graph(&reference_graph, "TC128")?;

        let reference = observe_ada_half(
            &t,
            &fixture,
            AdaHalfArm::PortableTc128,
            &reference_graph,
            AdaHalfPath::Eager,
            None,
        )?
        .1;
        for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
            for repeat in 0..2 {
                for (arm, graph) in [
                    (AdaHalfArm::PortableTc128, &reference_graph),
                    (candidate, &candidate_graph),
                ] {
                    let bits =
                        observe_ada_half(&t, &fixture, arm, graph, path, Some(&reference))?.1;
                    let schedule = match arm {
                        AdaHalfArm::PortableTc64 => "Tile64",
                        AdaHalfArm::PortableTc128 => "Tile128",
                        AdaHalfArm::FixedSm89Tc128S3 => "FixedSm89Tc128S3",
                    };
                    println!(
                        "{{\"schema\":\"MambaBiHalfNnTileAdaDiscoveryBitsV1\",\"dtype\":\"{dtype:?}\",\"schedule\":\"{schedule}\",\"path\":\"{}\",\"repeat\":{repeat},\"words\":{}}}",
                        path.name(),
                        bits.len(),
                    );
                }
            }
        }

        let mut strata = Vec::with_capacity(4);
        for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
            for order in [BracketOrder::Abba, BracketOrder::Baab] {
                strata.push(ada_half_screen_stratum(
                    &t,
                    &fixture,
                    candidate,
                    &candidate_graph,
                    &reference_graph,
                    &reference,
                    path,
                    order,
                )?);
            }
        }
        let retain = retain_decision(&strata);
        println!(
            "{{\"schema\":\"MambaBiHalfNnTileAdaDiscoveryDecisionV1\",\"dtype\":\"{dtype:?}\",\"cell\":\"d768_in_proj\",\"shape\":[2048,768,3072],\"candidate\":\"{}\",\"comparator\":\"forced_tc128\",\"strata_order\":[\"eager/ABBA\",\"eager/BAAB\",\"graph/ABBA\",\"graph/BAAB\"],\"strata\":{:?},\"ratio_direction\":\"candidate_over_forced_tc128\",\"retain\":{retain},\"decision\":\"{}\",\"promotion\":false}}",
            candidate.name(),
            strata,
            if retain {
                "advance_to_full_qualification"
            } else {
                "stop_no_retry"
            },
        );
        decisions.push(retain);
    }
    drop(t);
    quiet.verify_post_cohort(&format!("{cohort}/post"))?;
    if decisions != [true, true] {
        return Err(format!(
            "one or more Ada half {} dtype screens failed <0.99: {decisions:?}",
            candidate.name()
        ));
    }
    Ok(())
}

fn enqueue_ada_half_nn_fast_with_algo(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    algo: cudarc::cublas::sys::cublasGemmAlgo_t,
) -> Result<(), String> {
    use cudarc::cublas::{result, sys};
    let (m, k, n) = fixture.dims;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    unsafe {
        result::gemm_ex(
            *t.ctx.blas.handle(),
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            (&alpha as *const f32).cast(),
            fixture.b.ptr() as *const _,
            fixture.dtype.cuda_data_type(),
            n as i32,
            fixture.a.ptr() as *const _,
            fixture.dtype.cuda_data_type(),
            k as i32,
            (&beta as *const f32).cast(),
            fixture.fast.ptr() as *mut _,
            fixture.dtype.cuda_data_type(),
            n as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            algo,
        )
    }
    .map_err(|error| format!("Ada homogeneous-half Fast NN GEMM: {error:?}"))
}

fn enqueue_ada_half_nn_fast(t: &Ctx, fixture: &AdaHalfFixture) -> Result<(), String> {
    enqueue_ada_half_nn_fast_with_algo(
        t,
        fixture,
        cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )
}

fn capture_ada_half_nn_fast(t: &Ctx, fixture: &AdaHalfFixture) -> Result<CudaGraph, String> {
    unsafe { capture_into_graph(&t.ctx.stream, || enqueue_ada_half_nn_fast(t, fixture)) }
}

fn capture_ada_half_nn_fast_with_algo(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    algo: cudarc::cublas::sys::cublasGemmAlgo_t,
) -> Result<CudaGraph, String> {
    unsafe {
        capture_into_graph(&t.ctx.stream, || {
            enqueue_ada_half_nn_fast_with_algo(t, fixture, algo)
        })
    }
}

fn observe_ada_half_nn_fast_many(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    graph: &CudaGraph,
    path: AdaHalfPath,
    gemms: usize,
    expected: Option<&[u16]>,
) -> Result<(f64, Vec<u16>), String> {
    fixture.reset_fast(t)?;
    let start = t
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("Ada half Fast timing start: {error:?}"))?;
    for _ in 0..gemms {
        match path {
            AdaHalfPath::Eager => enqueue_ada_half_nn_fast(t, fixture)?,
            AdaHalfPath::Graph => graph
                .launch()
                .map_err(|error| format!("Ada half Fast graph launch: {error:?}"))?,
        }
    }
    let end = t
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("Ada half Fast timing end: {error:?}"))?;
    let elapsed_us = f64::from(
        start
            .elapsed_ms(&end)
            .map_err(|error| format!("Ada half Fast timing: {error:?}"))?,
    ) * 1_000.0
        / gemms as f64;
    if !elapsed_us.is_finite() || elapsed_us <= 0.0 {
        return Err(format!("invalid Ada half Fast timing {elapsed_us}"));
    }
    let bits = fixture.fast.logical_bits(t);
    if let Some(expected) = expected
        && bits != expected
    {
        let mismatch = bits
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
            .unwrap_or(bits.len());
        return Err(format!("Ada half Fast self-repeat differs at {mismatch}"));
    }
    fixture.validate_inputs(t)?;
    Ok((elapsed_us, bits))
}

fn observe_ada_half_nn_fast(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    graph: &CudaGraph,
    path: AdaHalfPath,
    expected: Option<&[u16]>,
) -> Result<(f64, Vec<u16>), String> {
    observe_ada_half_nn_fast_many(t, fixture, graph, path, 1, expected)
}

fn ada_half_s3_fast_stratum(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    candidate_graph: &CudaGraph,
    fast_graph: &CudaGraph,
    candidate_bits: &[u16],
    fast_bits: &[u16],
    cell: &str,
    path: AdaHalfPath,
    order: BracketOrder,
) -> Result<[f64; 2], String> {
    for _ in 0..ADA_HALF_WARMUPS {
        observe_ada_half(
            t,
            fixture,
            AdaHalfArm::FixedSm89Tc128S3,
            candidate_graph,
            path,
            Some(candidate_bits),
        )?;
        observe_ada_half_nn_fast(t, fixture, fast_graph, path, Some(fast_bits))?;
    }
    let candidate_first = matches!(order, BracketOrder::Abba);
    let arms = if candidate_first {
        [true, false, false, true]
    } else {
        [false, true, true, false]
    };
    let mut raw_windows = Vec::with_capacity(ADA_HALF_WINDOWS);
    let mut ratios = Vec::with_capacity(ADA_HALF_WINDOWS);
    for _ in 0..ADA_HALF_WINDOWS {
        let mut raw = [0.0; 4];
        for (index, candidate) in arms.into_iter().enumerate() {
            raw[index] = if candidate {
                observe_ada_half(
                    t,
                    fixture,
                    AdaHalfArm::FixedSm89Tc128S3,
                    candidate_graph,
                    path,
                    Some(candidate_bits),
                )?
                .0
            } else {
                observe_ada_half_nn_fast(t, fixture, fast_graph, path, Some(fast_bits))?.0
            };
        }
        ratios.push(candidate_over_reference(raw, order));
        raw_windows.push(raw);
    }
    let p50 = ada_percentile(&ratios, 0.5).ok_or("invalid S3/Fast p50")?;
    let p95 = ada_percentile(&ratios, 0.95).ok_or("invalid S3/Fast p95")?;
    let order = if candidate_first { "ABBA" } else { "BAAB" };
    let raw = raw_windows
        .iter()
        .map(|row| format!("[{:.9},{:.9},{:.9},{:.9}]", row[0], row[1], row[2], row[3]))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{{\"schema\":\"MambaBiHalfNnS3FastScreenV1\",\"dtype\":\"{:?}\",\"cell\":\"{cell}\",\"shape\":[{},{},{}],\"candidate\":\"fixed_sm89_tc128_s3\",\"candidate_symbol\":\"{}\",\"comparator\":\"native_half_fast\",\"path\":\"{}\",\"order\":\"{order}\",\"windows\":{ADA_HALF_WINDOWS},\"logical_gemms_per_observation\":1,\"raw_observations_us\":[{raw}],\"ratio_direction\":\"candidate_over_fast\",\"ratio_p50\":{p50:.9},\"ratio_p95\":{p95:.9}}}",
        fixture.dtype,
        fixture.dims.0,
        fixture.dims.1,
        fixture.dims.2,
        ada_half_symbol(AdaHalfArm::FixedSm89Tc128S3, fixture.dtype),
        path.name(),
    );
    Ok([p50, p95])
}

fn run_ada_half_nn_s3_shape_fast_batch() -> Result<(), String> {
    assert!(
        !cfg!(debug_assertions),
        "Ada half S3/Fast requires --release"
    );
    let quiet = QuietGpu::for_cuda_ordinal(0)?;
    let _pre = quiet.require_pre_context("half-nn-s3-fast/pre-context")?;
    let t = Ctx::new_ada()?;
    let compiler = t.ctx.kernels.compiler_identity();
    if compiler.nvrtc_version != (13, 2) {
        return Err(format!(
            "S3/Fast requires CUDA13.2, found {:?}",
            compiler.nvrtc_version
        ));
    }
    let _cohort = quiet.require_cohort("half-nn-s3-fast/cohort")?;
    let cells = [
        ("d768_in_proj", ADA_HALF_D768_IN, AdaHalfArm::PortableTc128),
        ("d128_out_proj", (1_024, 256, 128), AdaHalfArm::PortableTc64),
        (
            "d768_out_proj",
            (2_048, 1_536, 768),
            AdaHalfArm::PortableTc128,
        ),
        (
            "prism_in_proj",
            (4_621, 384, 1_928),
            AdaHalfArm::PortableTc128,
        ),
    ];
    for (cell, dims, current) in cells {
        for dtype in [WeightDtype::F16, WeightDtype::Bf16] {
            if cell == "d768_in_proj" && dtype == WeightDtype::Bf16 {
                continue;
            }
            ada_half_resource_gate(&t, AdaHalfArm::FixedSm89Tc128S3, dtype)?;
            ada_half_resource_gate(&t, current, dtype)?;
            let fixture = AdaHalfFixture::new(&t, dtype, dims);
            let candidate_graph = capture_ada_half(&t, &fixture, AdaHalfArm::FixedSm89Tc128S3)?;
            let current_graph = capture_ada_half(&t, &fixture, current)?;
            fixture.reset_fast(&t)?;
            enqueue_ada_half_nn_fast(&t, &fixture)?;
            t.ctx
                .stream
                .synchronize()
                .map_err(|error| format!("Ada half Fast warmup: {error:?}"))?;
            let fast_graph = capture_ada_half_nn_fast(&t, &fixture)?;
            validate_single_node_graph(&candidate_graph, "Fixed S3")?;
            validate_single_node_graph(&current_graph, "current TC")?;
            validate_nonempty_graph(&fast_graph, "native-half Fast")?;

            let current_bits = observe_ada_half(
                &t,
                &fixture,
                current,
                &current_graph,
                AdaHalfPath::Eager,
                None,
            )?
            .1;
            for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                for repeat in 0..2 {
                    for (arm, graph, role) in [
                        (current, &current_graph, "forced_current"),
                        (
                            AdaHalfArm::FixedSm89Tc128S3,
                            &candidate_graph,
                            "fixed_s3_candidate",
                        ),
                    ] {
                        observe_ada_half(&t, &fixture, arm, graph, path, Some(&current_bits))?;
                        println!(
                            "{{\"schema\":\"MambaBiHalfNnS3CurrentBitsV1\",\"dtype\":\"{dtype:?}\",\"cell\":\"{cell}\",\"shape\":[{},{},{}],\"role\":\"{role}\",\"schedule\":\"{}\",\"path\":\"{}\",\"repeat\":{repeat},\"words\":{}}}",
                            dims.0,
                            dims.1,
                            dims.2,
                            arm.name(),
                            path.name(),
                            current_bits.len(),
                        );
                    }
                }
            }
            let fast_bits =
                observe_ada_half_nn_fast(&t, &fixture, &fast_graph, AdaHalfPath::Eager, None)?.1;
            let fast_value = |word| match dtype {
                WeightDtype::F16 => f16::from_bits(word).to_f32(),
                WeightDtype::Bf16 => bf16::from_bits(word).to_f32(),
                WeightDtype::F32 => unreachable!(),
            };
            let fast_is_finite = fast_bits.iter().all(|&word| fast_value(word).is_finite());
            let fast_is_nonzero = fast_bits.iter().any(|&word| fast_value(word) != 0.0);
            if !fast_is_finite || !fast_is_nonzero {
                return Err(format!(
                    "{dtype:?} {cell} Fast output must be finite and nonzero"
                ));
            }
            for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                for _ in 0..2 {
                    observe_ada_half_nn_fast(&t, &fixture, &fast_graph, path, Some(&fast_bits))?;
                }
            }
            let mut strata = Vec::with_capacity(4);
            for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                for order in [BracketOrder::Abba, BracketOrder::Baab] {
                    strata.push(ada_half_s3_fast_stratum(
                        &t,
                        &fixture,
                        &candidate_graph,
                        &fast_graph,
                        &current_bits,
                        &fast_bits,
                        cell,
                        path,
                        order,
                    )?);
                }
            }
            let retain = retain_decision(&strata);
            println!(
                "{{\"schema\":\"MambaBiHalfNnS3FastDecisionV1\",\"dtype\":\"{dtype:?}\",\"cell\":\"{cell}\",\"shape\":[{},{},{}],\"strata\":{:?},\"strata_order\":[\"eager/ABBA\",\"eager/BAAB\",\"graph/ABBA\",\"graph/BAAB\"],\"retain\":{retain},\"decision\":\"{}\",\"promotion\":false}}",
                dims.0,
                dims.1,
                dims.2,
                strata,
                if retain { "advance" } else { "stop_no_retry" },
            );
        }
    }
    drop(t);
    quiet.verify_post_cohort("half-nn-s3-fast/post").map(|_| ())
}

#[test]
#[ignore = "requires exclusive Ada CC8.9 CUDA13.2; Fixed S3 NN shape/Fast discovery"]
fn ada_half_nn_fixed_s3_shape_fast_batch_discovery_once7() -> Result<(), String> {
    run_ada_half_nn_s3_shape_fast_batch()
}

fn observe_ada_half_nn_graph_window(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    graph: &CudaGraph,
    candidate: bool,
    expected: Option<&[u16]>,
) -> Result<(f64, Vec<u16>), String> {
    const GEMMS: usize = 5;
    if candidate {
        fixture.reset_and_validate_inputs(t, AdaHalfArm::FixedSm89Tc128S3)?;
    } else {
        fixture.reset_fast(t)?;
    }
    let start = t
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("NN denominator diagnostic start: {error:?}"))?;
    for _ in 0..GEMMS {
        graph
            .launch()
            .map_err(|error| format!("NN denominator diagnostic graph: {error:?}"))?;
    }
    let end = t
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("NN denominator diagnostic end: {error:?}"))?;
    let elapsed_us = f64::from(
        start
            .elapsed_ms(&end)
            .map_err(|error| format!("NN denominator diagnostic timing: {error:?}"))?,
    ) * 1_000.0
        / GEMMS as f64;
    if !elapsed_us.is_finite() || elapsed_us <= 0.0 {
        return Err(format!("invalid NN denominator diagnostic {elapsed_us}"));
    }
    let bits = if candidate {
        fixture.output(AdaHalfArm::FixedSm89Tc128S3).logical_bits(t)
    } else {
        fixture.fast.logical_bits(t)
    };
    if let Some(expected) = expected
        && bits != expected
    {
        return Err("NN denominator diagnostic output was not repeatable".into());
    }
    fixture.validate_inputs(t)?;
    Ok((elapsed_us, bits))
}

fn run_ada_half_nn_fast_denominator_diagnostic() -> Result<(), String> {
    use cudarc::cublas::sys::cublasGemmAlgo_t;
    assert!(
        !cfg!(debug_assertions),
        "NN denominator diagnostic requires --release"
    );
    let quiet = QuietGpu::for_cuda_ordinal(0)?;
    let _pre = quiet.require_pre_context("half-nn-fast-diagnostic/pre-context")?;
    let t = Ctx::new_ada()?;
    let compiler = t.ctx.kernels.compiler_identity();
    if compiler.nvrtc_version != (13, 2) {
        return Err(format!(
            "NN denominator diagnostic requires CUDA13.2, found {:?}",
            compiler.nvrtc_version
        ));
    }
    let _cohort = quiet.require_cohort("half-nn-fast-diagnostic/cohort")?;
    for guard_offset in [8usize, 128] {
        let fixture =
            AdaHalfFixture::new_with_guard(&t, WeightDtype::F16, (2_048, 1_536, 768), guard_offset);
        let candidate_graph = capture_ada_half(&t, &fixture, AdaHalfArm::FixedSm89Tc128S3)?;
        validate_single_node_graph(&candidate_graph, "diagnostic Fixed S3")?;
        for (algo_name, algo) in [
            ("default", cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT),
            (
                "default_tensor_op",
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            ),
        ] {
            fixture.reset_fast(&t)?;
            enqueue_ada_half_nn_fast_with_algo(&t, &fixture, algo)?;
            t.ctx
                .stream
                .synchronize()
                .map_err(|error| format!("NN denominator diagnostic warmup: {error:?}"))?;
            let fast_graph = capture_ada_half_nn_fast_with_algo(&t, &fixture, algo)?;
            validate_nonempty_graph(&fast_graph, algo_name)?;
            let candidate_bits =
                observe_ada_half_nn_graph_window(&t, &fixture, &candidate_graph, true, None)?.1;
            let fast_bits =
                observe_ada_half_nn_graph_window(&t, &fixture, &fast_graph, false, None)?.1;
            let fast_is_finite = fast_bits
                .iter()
                .all(|&word| f16::from_bits(word).to_f32().is_finite());
            let fast_is_nonzero = fast_bits
                .iter()
                .any(|&word| f16::from_bits(word).to_f32() != 0.0);
            if !fast_is_finite || !fast_is_nonzero {
                return Err("NN denominator Fast output is non-finite or all-zero".into());
            }
            let mut candidate_us = [0.0; 3];
            let mut fast_us = [0.0; 3];
            for sample in 0..3 {
                let candidate_first = sample % 2 == 0;
                for run_candidate in [candidate_first, !candidate_first] {
                    let value = observe_ada_half_nn_graph_window(
                        &t,
                        &fixture,
                        if run_candidate {
                            &candidate_graph
                        } else {
                            &fast_graph
                        },
                        run_candidate,
                        Some(if run_candidate {
                            &candidate_bits
                        } else {
                            &fast_bits
                        }),
                    )?
                    .0;
                    if run_candidate {
                        candidate_us[sample] = value;
                    } else {
                        fast_us[sample] = value;
                    }
                }
            }
            let ratios = [
                candidate_us[0] / fast_us[0],
                candidate_us[1] / fast_us[1],
                candidate_us[2] / fast_us[2],
            ];
            println!(
                "{{\"schema\":\"MambaBiHalfNnFastDenominatorDiagnosticV1\",\"dtype\":\"F16\",\"cell\":\"d768_out_proj\",\"shape\":[2048,1536,768],\"guard_half_elements\":{guard_offset},\"pointer_mod_256\":{{\"a\":{},\"b\":{},\"candidate\":{},\"fast\":{}}},\"cublas_algo\":\"{algo_name}\",\"path\":\"graph\",\"gemms_per_sample\":5,\"samples\":3,\"candidate_us\":{:?},\"fast_us\":{:?},\"candidate_over_fast\":{:?},\"diagnostic_only\":true,\"promotion\":false}}",
                fixture.a.ptr() % 256,
                fixture.b.ptr() % 256,
                fixture.candidate.ptr() % 256,
                fixture.fast.ptr() % 256,
                candidate_us,
                fast_us,
                ratios,
            );
        }
    }
    drop(t);
    quiet
        .verify_post_cohort("half-nn-fast-diagnostic/post")
        .map(|_| ())
}

#[test]
#[ignore = "requires exclusive Ada CC8.9 CUDA13.2; NN Fast denominator diagnostic"]
fn ada_half_nn_d768_out_fast_denominator_diagnostic() -> Result<(), String> {
    run_ada_half_nn_fast_denominator_diagnostic()
}

#[derive(Clone, Copy)]
enum AdaHalfNnN64Variant {
    D768In,
    D768Out,
}

impl AdaHalfNnN64Variant {
    const fn name(self) -> &'static str {
        match self {
            Self::D768In => "fixed_n64_m64n64_bk64_s3",
            Self::D768Out => "fixed_n64_m128n64_bk64_s2",
        }
    }

    const fn dims(self) -> (usize, usize, usize) {
        match self {
            Self::D768In => (2_048, 768, 3_072),
            Self::D768Out => (2_048, 1_536, 768),
        }
    }

    const fn grid(self) -> u32 {
        match self {
            Self::D768In => 1_536,
            Self::D768Out => 192,
        }
    }
}

struct AdaHalfNnN64Candidate {
    in_bf16: CudaFunction,
    in_f16: CudaFunction,
    out_bf16: CudaFunction,
    out_f16: CudaFunction,
    source_sha256: String,
}

impl AdaHalfNnN64Candidate {
    fn function(&self, variant: AdaHalfNnN64Variant, dtype: WeightDtype) -> &CudaFunction {
        match (variant, dtype) {
            (AdaHalfNnN64Variant::D768In, WeightDtype::Bf16) => &self.in_bf16,
            (AdaHalfNnN64Variant::D768In, WeightDtype::F16) => &self.in_f16,
            (AdaHalfNnN64Variant::D768Out, WeightDtype::Bf16) => &self.out_bf16,
            (AdaHalfNnN64Variant::D768Out, WeightDtype::F16) => &self.out_f16,
            (_, WeightDtype::F32) => panic!("N64 candidate requires a half dtype"),
        }
    }
}

fn compile_ada_half_nn_n64_candidate(t: &Ctx) -> Result<AdaHalfNnN64Candidate, String> {
    let transformed = triad_half_nn_n64_source::compose_source(
        include_str!("../kernels/gemm_bi_fixed/sm89_half_n64.cu"),
        include_str!("../kernels/gemm_bi_fixed/sm89_half_swizzle.cu"),
    )?;
    let common = include_str!("../kernels/gemm_bi_fixed/common.cuh")
        .lines()
        .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
        .collect::<Vec<_>>()
        .join("\n");
    let source = [
        include_str!("../kernels/_typed_prelude.cuh"),
        &common,
        &transformed,
    ]
    .join("\n");
    let source_sha256 = format!("{:x}", Sha256::digest(source.as_bytes()));
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        source,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_89"),
            options: vec![
                "--fmad=true".into(),
                "--extra-device-vectorization".into(),
                "-DNDEBUG".into(),
                "-DGEMM_BI_GROUP_M=16".into(),
                "-DMAMBA_RS_STATE_CAP=256".into(),
                "--frandom-seed=1295072049".into(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        },
    )
    .map_err(|error| format!("compile half NN N64 candidates: {error:?}"))?;
    let module = t
        .ctx
        .stream
        .context()
        .load_module(ptx)
        .map_err(|error| format!("load half NN N64 module: {error:?}"))?;
    let load = |prefix: &str, suffix: &str| {
        let symbol = format!("{prefix}{suffix}");
        module
            .load_function(&symbol)
            .map_err(|error| format!("load {symbol}: {error:?}"))
    };
    Ok(AdaHalfNnN64Candidate {
        in_bf16: load(triad_half_nn_n64_source::IN_PREFIX, "bf16")?,
        in_f16: load(triad_half_nn_n64_source::IN_PREFIX, "f16")?,
        out_bf16: load(triad_half_nn_n64_source::OUT_PREFIX, "bf16")?,
        out_f16: load(triad_half_nn_n64_source::OUT_PREFIX, "f16")?,
        source_sha256,
    })
}

fn gate_ada_half_nn_n64_resources(
    candidate: &AdaHalfNnN64Candidate,
    variant: AdaHalfNnN64Variant,
    dtype: WeightDtype,
) -> Result<(), String> {
    let function = candidate.function(variant, dtype);
    let registers = function
        .num_regs()
        .map_err(|error| format!("N64 registers: {error:?}"))?;
    let local = function
        .local_size_bytes()
        .map_err(|error| format!("N64 local: {error:?}"))?;
    let static_shared = function
        .shared_size_bytes()
        .map_err(|error| format!("N64 static: {error:?}"))?;
    let max_threads = function
        .max_threads_per_block()
        .map_err(|error| format!("N64 max threads: {error:?}"))?;
    let occupancy = function
        .occupancy_max_active_blocks_per_multiprocessor(128, 49_152, None)
        .map_err(|error| format!("N64 occupancy: {error:?}"))?;
    println!(
        "{{\"schema\":\"MambaBiHalfNnN64ResourceV1\",\"variant\":\"{}\",\"dtype\":\"{dtype:?}\",\"source_sha256\":\"{}\",\"threads\":128,\"registers\":{registers},\"local_bytes\":{local},\"static_shared_bytes\":{static_shared},\"dynamic_shared_bytes\":49152,\"max_threads\":{max_threads},\"occupancy\":{occupancy},\"required_occupancy\":2}}",
        variant.name(),
        candidate.source_sha256,
    );
    if registers <= 0 || local != 0 || static_shared != 0 || max_threads < 128 || occupancy < 2 {
        return Err(format!(
            "half NN N64 resource gate failed: regs={registers} local={local} static={static_shared}/0 max_threads={max_threads}/128 occupancy={occupancy}/2"
        ));
    }
    Ok(())
}

fn enqueue_ada_half_nn_n64(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    candidate: &AdaHalfNnN64Candidate,
    variant: AdaHalfNnN64Variant,
) -> Result<(), String> {
    let (m, k, n) = variant.dims();
    let output = fixture.candidate.ptr();
    let a = fixture.a.ptr();
    let b = fixture.b.ptr();
    let bias = 0_u64;
    let params = FixedHalfS3Params {
        alpha: 1.0,
        beta: 0.0,
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda: k as i32,
        ldb: n as i32,
        ldc: n as i32,
    };
    let mut launch = t
        .ctx
        .stream
        .launch_builder(candidate.function(variant, fixture.dtype));
    launch.arg(&output).arg(&a).arg(&b).arg(&bias).arg(&params);
    unsafe {
        launch.launch(cudarc::driver::LaunchConfig {
            grid_dim: (variant.grid(), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 49_152,
        })
    }
    .map(|_| ())
    .map_err(|error| format!("half NN {} launch: {error:?}", variant.name()))
}

fn capture_ada_half_nn_n64(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    candidate: &AdaHalfNnN64Candidate,
    variant: AdaHalfNnN64Variant,
) -> Result<CudaGraph, String> {
    unsafe {
        capture_into_graph(&t.ctx.stream, || {
            enqueue_ada_half_nn_n64(t, fixture, candidate, variant)
        })
    }
}

#[derive(Clone, Copy)]
enum AdaHalfNnAlignedCandidate<'a> {
    FixedS3,
    N64(&'a AdaHalfNnN64Candidate, AdaHalfNnN64Variant),
}

impl AdaHalfNnAlignedCandidate<'_> {
    const fn name(self) -> &'static str {
        match self {
            Self::FixedS3 => "fixed_sm89_tc128_s3",
            Self::N64(_, variant) => variant.name(),
        }
    }
}

fn observe_ada_half_nn_aligned_candidate(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    graph: &CudaGraph,
    candidate: AdaHalfNnAlignedCandidate<'_>,
    path: AdaHalfPath,
    gemms: usize,
    expected: &[u16],
) -> Result<f64, String> {
    match candidate {
        AdaHalfNnAlignedCandidate::FixedS3 => observe_ada_half_many(
            t,
            fixture,
            AdaHalfArm::FixedSm89Tc128S3,
            graph,
            path,
            gemms,
            Some(expected),
        )
        .map(|value| value.0),
        AdaHalfNnAlignedCandidate::N64(n64, variant) => {
            fixture.reset_and_validate_inputs(t, AdaHalfArm::FixedSm89Tc128S3)?;
            let start = t
                .ctx
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| format!("N64 timing start: {error:?}"))?;
            for _ in 0..gemms {
                match path {
                    AdaHalfPath::Eager => enqueue_ada_half_nn_n64(t, fixture, n64, variant)?,
                    AdaHalfPath::Graph => graph
                        .launch()
                        .map_err(|error| format!("N64 graph launch: {error:?}"))?,
                }
            }
            let end = t
                .ctx
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| format!("N64 timing end: {error:?}"))?;
            let elapsed_us = f64::from(
                start
                    .elapsed_ms(&end)
                    .map_err(|error| format!("N64 timing: {error:?}"))?,
            ) * 1_000.0
                / gemms as f64;
            if !elapsed_us.is_finite() || elapsed_us <= 0.0 {
                return Err(format!("invalid N64 timing {elapsed_us}"));
            }
            let bits = fixture.candidate.logical_bits(t);
            if bits != expected {
                let mismatch = bits
                    .iter()
                    .zip(expected)
                    .position(|(actual, expected)| actual != expected)
                    .unwrap_or(bits.len());
                return Err(format!("N64 {} differs at word {mismatch}", variant.name()));
            }
            fixture.validate_inputs(t)?;
            Ok(elapsed_us)
        }
    }
}

#[derive(Clone, Copy)]
enum AdaHalfNnAlignedComparator {
    ForcedTc128,
    Fast,
}

impl AdaHalfNnAlignedComparator {
    const fn name(self) -> &'static str {
        match self {
            Self::ForcedTc128 => "forced_tc128",
            Self::Fast => "native_half_fast",
        }
    }
}

fn observe_ada_half_nn_aligned_comparator(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    current_graph: &CudaGraph,
    fast_graph: &CudaGraph,
    comparator: AdaHalfNnAlignedComparator,
    path: AdaHalfPath,
    expected: &[u16],
) -> Result<f64, String> {
    const GEMMS: usize = 20;
    match comparator {
        AdaHalfNnAlignedComparator::ForcedTc128 => observe_ada_half_many(
            t,
            fixture,
            AdaHalfArm::PortableTc128,
            current_graph,
            path,
            GEMMS,
            Some(expected),
        )
        .map(|value| value.0),
        AdaHalfNnAlignedComparator::Fast => {
            observe_ada_half_nn_fast_many(t, fixture, fast_graph, path, GEMMS, Some(expected))
                .map(|value| value.0)
        }
    }
}

fn screen_ada_half_nn_aligned_pair(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    candidate_graph: &CudaGraph,
    candidate: AdaHalfNnAlignedCandidate<'_>,
    current_graph: &CudaGraph,
    fast_graph: &CudaGraph,
    candidate_bits: &[u16],
    comparator_bits: &[u16],
    comparator: AdaHalfNnAlignedComparator,
    cell: &str,
    path: AdaHalfPath,
    order: BracketOrder,
) -> Result<[f64; 2], String> {
    const GEMMS: usize = 20;
    for _ in 0..ADA_HALF_WARMUPS {
        observe_ada_half_nn_aligned_candidate(
            t,
            fixture,
            candidate_graph,
            candidate,
            path,
            GEMMS,
            candidate_bits,
        )?;
        observe_ada_half_nn_aligned_comparator(
            t,
            fixture,
            current_graph,
            fast_graph,
            comparator,
            path,
            comparator_bits,
        )?;
    }
    let candidate_first = matches!(order, BracketOrder::Abba);
    let arms = if candidate_first {
        [true, false, false, true]
    } else {
        [false, true, true, false]
    };
    let mut raw_windows = Vec::with_capacity(ADA_HALF_WINDOWS);
    let mut ratios = Vec::with_capacity(ADA_HALF_WINDOWS);
    for _ in 0..ADA_HALF_WINDOWS {
        let mut raw = [0.0; 4];
        for (index, run_candidate) in arms.into_iter().enumerate() {
            raw[index] = if run_candidate {
                observe_ada_half_nn_aligned_candidate(
                    t,
                    fixture,
                    candidate_graph,
                    candidate,
                    path,
                    GEMMS,
                    candidate_bits,
                )?
            } else {
                observe_ada_half_nn_aligned_comparator(
                    t,
                    fixture,
                    current_graph,
                    fast_graph,
                    comparator,
                    path,
                    comparator_bits,
                )?
            };
        }
        ratios.push(candidate_over_reference(raw, order));
        raw_windows.push(raw);
    }
    let p50 = ada_percentile(&ratios, 0.5).ok_or("invalid aligned NN p50")?;
    let p95 = ada_percentile(&ratios, 0.95).ok_or("invalid aligned NN p95")?;
    let order = if candidate_first { "ABBA" } else { "BAAB" };
    let raw = raw_windows
        .iter()
        .map(|row| format!("[{:.9},{:.9},{:.9},{:.9}]", row[0], row[1], row[2], row[3]))
        .collect::<Vec<_>>()
        .join(",");
    let schema = match candidate {
        AdaHalfNnAlignedCandidate::FixedS3 => "MambaBiHalfNnS3AlignedScreenV1",
        AdaHalfNnAlignedCandidate::N64(_, _) => "MambaBiHalfNnN64AlignedScreenV1",
    };
    println!(
        "{{\"schema\":\"{schema}\",\"dtype\":\"{:?}\",\"cell\":\"{cell}\",\"shape\":[{},{},{}],\"candidate\":\"{}\",\"comparator\":\"{}\",\"path\":\"{}\",\"order\":\"{order}\",\"windows\":{ADA_HALF_WINDOWS},\"logical_gemms_per_observation\":{GEMMS},\"raw_observations_us\":[{raw}],\"ratio_direction\":\"candidate_over_comparator\",\"ratio_p50\":{p50:.9},\"ratio_p95\":{p95:.9}}}",
        fixture.dtype,
        fixture.dims.0,
        fixture.dims.1,
        fixture.dims.2,
        candidate.name(),
        comparator.name(),
        path.name(),
    );
    Ok([p50, p95])
}

fn run_ada_half_nn_s3_aligned_cells(
    cells: &[(&str, (usize, usize, usize), WeightDtype)],
    quiet_label: &str,
) -> Result<(), String> {
    assert!(
        !cfg!(debug_assertions),
        "aligned NN S3 confirmation requires --release"
    );
    let quiet = QuietGpu::for_cuda_ordinal(0)?;
    let _pre = quiet.require_pre_context(&format!("{quiet_label}/pre-context"))?;
    let t = Ctx::new_ada()?;
    let compiler = t.ctx.kernels.compiler_identity();
    if compiler.nvrtc_version != (13, 2) {
        return Err(format!(
            "aligned NN S3 confirmation requires CUDA13.2, found {:?}",
            compiler.nvrtc_version
        ));
    }
    let _cohort = quiet.require_cohort(&format!("{quiet_label}/cohort"))?;
    for &(cell, dims, dtype) in cells {
        ada_half_resource_gate(&t, AdaHalfArm::FixedSm89Tc128S3, dtype)?;
        ada_half_resource_gate(&t, AdaHalfArm::PortableTc128, dtype)?;
        let fixture = AdaHalfFixture::new_with_guard(&t, dtype, dims, 128);
        let candidate_graph = capture_ada_half(&t, &fixture, AdaHalfArm::FixedSm89Tc128S3)?;
        let current_graph = capture_ada_half(&t, &fixture, AdaHalfArm::PortableTc128)?;
        fixture.reset_fast(&t)?;
        enqueue_ada_half_nn_fast(&t, &fixture)?;
        t.ctx
            .stream
            .synchronize()
            .map_err(|error| format!("aligned NN Fast warmup: {error:?}"))?;
        let fast_graph = capture_ada_half_nn_fast(&t, &fixture)?;
        validate_single_node_graph(&candidate_graph, "aligned Fixed S3")?;
        validate_single_node_graph(&current_graph, "aligned forced TC128")?;
        validate_nonempty_graph(&fast_graph, "aligned native-half Fast")?;

        let current_bits = observe_ada_half_many(
            &t,
            &fixture,
            AdaHalfArm::PortableTc128,
            &current_graph,
            AdaHalfPath::Eager,
            20,
            None,
        )?
        .1;
        for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
            for arm in [AdaHalfArm::PortableTc128, AdaHalfArm::FixedSm89Tc128S3] {
                observe_ada_half_many(
                    &t,
                    &fixture,
                    arm,
                    if arm == AdaHalfArm::PortableTc128 {
                        &current_graph
                    } else {
                        &candidate_graph
                    },
                    path,
                    20,
                    Some(&current_bits),
                )?;
            }
        }
        let fast_bits =
            observe_ada_half_nn_fast_many(&t, &fixture, &fast_graph, AdaHalfPath::Eager, 20, None)?
                .1;
        let fast_value = |word| match dtype {
            WeightDtype::F16 => f16::from_bits(word).to_f32(),
            WeightDtype::Bf16 => bf16::from_bits(word).to_f32(),
            WeightDtype::F32 => unreachable!(),
        };
        if !fast_bits.iter().all(|&word| fast_value(word).is_finite())
            || !fast_bits.iter().any(|&word| fast_value(word) != 0.0)
        {
            return Err(format!("aligned {dtype:?} {cell} Fast is invalid"));
        }
        observe_ada_half_nn_fast_many(
            &t,
            &fixture,
            &fast_graph,
            AdaHalfPath::Graph,
            20,
            Some(&fast_bits),
        )?;

        let mut decisions = Vec::with_capacity(2);
        for comparator in [
            AdaHalfNnAlignedComparator::ForcedTc128,
            AdaHalfNnAlignedComparator::Fast,
        ] {
            let comparator_bits = match comparator {
                AdaHalfNnAlignedComparator::ForcedTc128 => &current_bits,
                AdaHalfNnAlignedComparator::Fast => &fast_bits,
            };
            let mut strata = Vec::with_capacity(4);
            for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                for order in [BracketOrder::Abba, BracketOrder::Baab] {
                    strata.push(screen_ada_half_nn_aligned_pair(
                        &t,
                        &fixture,
                        &candidate_graph,
                        AdaHalfNnAlignedCandidate::FixedS3,
                        &current_graph,
                        &fast_graph,
                        &current_bits,
                        comparator_bits,
                        comparator,
                        cell,
                        path,
                        order,
                    )?);
                }
            }
            decisions.push((comparator.name(), retain_decision(&strata), strata));
        }
        println!(
            "{{\"schema\":\"MambaBiHalfNnS3AlignedDecisionV1\",\"dtype\":\"{dtype:?}\",\"cell\":\"{cell}\",\"shape\":[{},{},{}],\"guard_half_elements\":128,\"pointer_mod_256\":{{\"a\":{},\"b\":{},\"candidate\":{},\"current\":{},\"fast\":{}}},\"comparators\":[{{\"name\":\"{}\",\"retain\":{},\"strata\":{:?}}},{{\"name\":\"{}\",\"retain\":{},\"strata\":{:?}}}],\"promotion\":false}}",
            dims.0,
            dims.1,
            dims.2,
            fixture.a.ptr() % 256,
            fixture.b.ptr() % 256,
            fixture.candidate.ptr() % 256,
            fixture.reference.ptr() % 256,
            fixture.fast.ptr() % 256,
            decisions[0].0,
            decisions[0].1,
            decisions[0].2,
            decisions[1].0,
            decisions[1].1,
            decisions[1].2,
        );
    }
    drop(t);
    quiet
        .verify_post_cohort(&format!("{quiet_label}/post"))
        .map(|_| ())
}

fn run_ada_half_nn_s3_aligned_confirmation() -> Result<(), String> {
    run_ada_half_nn_s3_aligned_cells(
        &[
            ("d768_out_proj", (2_048, 1_536, 768), WeightDtype::F16),
            ("d768_out_proj", (2_048, 1_536, 768), WeightDtype::Bf16),
            ("prism_in_proj", (4_621, 384, 1_928), WeightDtype::F16),
            ("prism_in_proj", (4_621, 384, 1_928), WeightDtype::Bf16),
        ],
        "half-nn-s3-aligned",
    )
}

#[test]
#[ignore = "requires exclusive Ada CC8.9 CUDA13.2; aligned Fixed S3 NN confirmation"]
fn ada_half_nn_fixed_s3_aligned_four_cell_confirmation_once7() -> Result<(), String> {
    run_ada_half_nn_s3_aligned_confirmation()
}

#[test]
#[ignore = "requires exclusive Ada CC8.9 CUDA13.2; aligned BF16 Fixed S3 NN d768-in"]
fn ada_half_nn_fixed_s3_aligned_bf16_d768_in_confirmation_once7() -> Result<(), String> {
    run_ada_half_nn_s3_aligned_cells(
        &[("d768_in_proj", (2_048, 768, 3_072), WeightDtype::Bf16)],
        "half-nn-s3-aligned-bf16-d768-in",
    )
}

#[test]
#[ignore = "requires exclusive Ada CC8.9 CUDA13.2; aligned F16 Fixed S3 NN d768-in"]
fn ada_half_nn_fixed_s3_aligned_f16_d768_in_confirmation_once7() -> Result<(), String> {
    run_ada_half_nn_s3_aligned_cells(
        &[("d768_in_proj", (2_048, 768, 3_072), WeightDtype::F16)],
        "half-nn-s3-aligned-f16-d768-in",
    )
}

fn run_ada_half_nn_n64_batch() -> Result<(), String> {
    assert!(
        !cfg!(debug_assertions),
        "half NN N64 discovery requires --release"
    );
    let quiet = QuietGpu::for_cuda_ordinal(0)?;
    let _pre = quiet.require_pre_context("half-nn-n64/pre-context")?;
    let t = Ctx::new_ada()?;
    let compiler = t.ctx.kernels.compiler_identity();
    if compiler.nvrtc_version != (13, 2) {
        return Err(format!(
            "half NN N64 requires CUDA13.2, found {:?}",
            compiler.nvrtc_version
        ));
    }
    let n64 = compile_ada_half_nn_n64_candidate(&t)?;
    let _cohort = quiet.require_cohort("half-nn-n64/cohort")?;
    for (cell, variant) in [
        ("d768_in_proj", AdaHalfNnN64Variant::D768In),
        ("d768_out_proj", AdaHalfNnN64Variant::D768Out),
    ] {
        for dtype in [WeightDtype::F16, WeightDtype::Bf16] {
            gate_ada_half_nn_n64_resources(&n64, variant, dtype)?;
            ada_half_resource_gate(&t, AdaHalfArm::PortableTc128, dtype)?;
            let dims = variant.dims();
            let fixture = AdaHalfFixture::new_with_guard(&t, dtype, dims, 128);
            let candidate_graph = capture_ada_half_nn_n64(&t, &fixture, &n64, variant)?;
            let current_graph = capture_ada_half(&t, &fixture, AdaHalfArm::PortableTc128)?;
            fixture.reset_fast(&t)?;
            enqueue_ada_half_nn_fast(&t, &fixture)?;
            t.ctx
                .stream
                .synchronize()
                .map_err(|error| format!("N64 Fast warmup: {error:?}"))?;
            let fast_graph = capture_ada_half_nn_fast(&t, &fixture)?;
            validate_single_node_graph(&candidate_graph, variant.name())?;
            validate_single_node_graph(&current_graph, "N64 forced TC128")?;
            validate_nonempty_graph(&fast_graph, "N64 native-half Fast")?;

            let current_bits = observe_ada_half_many(
                &t,
                &fixture,
                AdaHalfArm::PortableTc128,
                &current_graph,
                AdaHalfPath::Eager,
                20,
                None,
            )?
            .1;
            let candidate_ref = AdaHalfNnAlignedCandidate::N64(&n64, variant);
            for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                observe_ada_half_many(
                    &t,
                    &fixture,
                    AdaHalfArm::PortableTc128,
                    &current_graph,
                    path,
                    20,
                    Some(&current_bits),
                )?;
                observe_ada_half_nn_aligned_candidate(
                    &t,
                    &fixture,
                    &candidate_graph,
                    candidate_ref,
                    path,
                    20,
                    &current_bits,
                )?;
            }
            let fast_bits = observe_ada_half_nn_fast_many(
                &t,
                &fixture,
                &fast_graph,
                AdaHalfPath::Eager,
                20,
                None,
            )?
            .1;
            let fast_value = |word| match dtype {
                WeightDtype::F16 => f16::from_bits(word).to_f32(),
                WeightDtype::Bf16 => bf16::from_bits(word).to_f32(),
                WeightDtype::F32 => unreachable!(),
            };
            if !fast_bits.iter().all(|&word| fast_value(word).is_finite())
                || !fast_bits.iter().any(|&word| fast_value(word) != 0.0)
            {
                return Err(format!("N64 {dtype:?} {cell} Fast is invalid"));
            }
            observe_ada_half_nn_fast_many(
                &t,
                &fixture,
                &fast_graph,
                AdaHalfPath::Graph,
                20,
                Some(&fast_bits),
            )?;

            let mut decisions = Vec::with_capacity(2);
            for comparator in [
                AdaHalfNnAlignedComparator::ForcedTc128,
                AdaHalfNnAlignedComparator::Fast,
            ] {
                let comparator_bits = match comparator {
                    AdaHalfNnAlignedComparator::ForcedTc128 => &current_bits,
                    AdaHalfNnAlignedComparator::Fast => &fast_bits,
                };
                let mut strata = Vec::with_capacity(4);
                for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                    for order in [BracketOrder::Abba, BracketOrder::Baab] {
                        strata.push(screen_ada_half_nn_aligned_pair(
                            &t,
                            &fixture,
                            &candidate_graph,
                            candidate_ref,
                            &current_graph,
                            &fast_graph,
                            &current_bits,
                            comparator_bits,
                            comparator,
                            cell,
                            path,
                            order,
                        )?);
                    }
                }
                decisions.push((comparator.name(), retain_decision(&strata), strata));
            }
            println!(
                "{{\"schema\":\"MambaBiHalfNnN64AlignedDecisionV1\",\"dtype\":\"{dtype:?}\",\"cell\":\"{cell}\",\"shape\":[{},{},{}],\"variant\":\"{}\",\"guard_half_elements\":128,\"comparators\":[{{\"name\":\"{}\",\"retain\":{},\"strata\":{:?}}},{{\"name\":\"{}\",\"retain\":{},\"strata\":{:?}}}],\"promotion\":false}}",
                dims.0,
                dims.1,
                dims.2,
                variant.name(),
                decisions[0].0,
                decisions[0].1,
                decisions[0].2,
                decisions[1].0,
                decisions[1].1,
                decisions[1].2,
            );
        }
    }
    drop(t);
    quiet.verify_post_cohort("half-nn-n64/post").map(|_| ())
}

#[test]
#[ignore = "requires exclusive Ada CC8.9 CUDA13.2; two-shape half NN N64 discovery"]
fn ada_half_nn_n64_two_shape_vs_current_and_fast_discovery_once7() -> Result<(), String> {
    run_ada_half_nn_n64_batch()
}

#[test]
#[ignore = "requires an exclusive quiet CC8.9/142-SM Ada GPU; discovery only"]
fn ada_half_nn_d768_in_tc64_vs_tc128_discovery_once7() -> Result<(), String> {
    run_ada_half_nn_d768_in_screen(AdaHalfArm::PortableTc64, "typed-nn-d768-in-ada")
}

#[test]
#[ignore = "requires an exclusive quiet CC8.9/142-SM Ada GPU; Fixed S3 reuse discovery only"]
fn ada_half_nn_d768_in_fixed_s3_vs_tc128_discovery_once7() -> Result<(), String> {
    run_ada_half_nn_d768_in_screen(
        AdaHalfArm::FixedSm89Tc128S3,
        "typed-nn-d768-in-fixed-s3-ada",
    )
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

const ADA_HALF_TN_D128_IN: (usize, usize, usize) = (1_024, 128, 512);
const ADA_HALF_TN_D128_OUT: (usize, usize, usize) = (1_024, 256, 128);
const ADA_HALF_TN_OBSERVATION_GEMMS: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdaHalfTnArm {
    Candidate,
    CurrentTc64,
    Fast,
}

impl AdaHalfTnArm {
    const fn name(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::CurrentTc64 => "current_tc64",
            Self::Fast => "native_half_fast",
        }
    }
}

struct AdaHalfTnCandidate {
    kind: AdaHalfTnCandidateKind,
    bf16: CudaFunction,
    f16: CudaFunction,
    source_sha256: String,
}

#[derive(Clone, Copy)]
enum AdaHalfTnCandidateKind {
    Microtile(triad_half_tn_microtile_source::Microtile),
    Rect128x64,
    Tc64Bk32S4,
    Tc64Bk64S2Compact,
}

impl AdaHalfTnCandidate {
    fn name(&self) -> &'static str {
        match self.kind {
            AdaHalfTnCandidateKind::Microtile(
                triad_half_tn_microtile_source::Microtile::M32N32,
            ) => "m32n32_bk32_s3",
            AdaHalfTnCandidateKind::Microtile(
                triad_half_tn_microtile_source::Microtile::M16N32,
            ) => "m16n32_bk32_s3",
            AdaHalfTnCandidateKind::Rect128x64 => "loaded_rect128x64_bk32_s3",
            AdaHalfTnCandidateKind::Tc64Bk32S4 => "tc64_bk32_s4",
            AdaHalfTnCandidateKind::Tc64Bk64S2Compact => "tc64_bk64_s2_compact_xor",
        }
    }

    fn symbol(&self, dtype: WeightDtype) -> String {
        use triad_half_tn_microtile_source::Microtile;
        match (self.kind, dtype) {
            (AdaHalfTnCandidateKind::Microtile(Microtile::M32N32), WeightDtype::Bf16) => {
                "gemm_bi_tn_test_m32n32_sm80_mma_half_v1_bf16".into()
            }
            (AdaHalfTnCandidateKind::Microtile(Microtile::M32N32), WeightDtype::F16) => {
                "gemm_bi_tn_test_m32n32_sm80_mma_half_v1_f16".into()
            }
            (AdaHalfTnCandidateKind::Microtile(Microtile::M16N32), WeightDtype::Bf16) => {
                "gemm_bi_tn_test_m16n32_sm80_mma_half_v1_bf16".into()
            }
            (AdaHalfTnCandidateKind::Microtile(Microtile::M16N32), WeightDtype::F16) => {
                "gemm_bi_tn_test_m16n32_sm80_mma_half_v1_f16".into()
            }
            (AdaHalfTnCandidateKind::Rect128x64, WeightDtype::Bf16) => {
                "gemm_bi_tn_tc128x64_bf16".into()
            }
            (AdaHalfTnCandidateKind::Rect128x64, WeightDtype::F16) => {
                "gemm_bi_tn_tc128x64_f16".into()
            }
            (AdaHalfTnCandidateKind::Tc64Bk32S4, WeightDtype::Bf16) => {
                triad_half_tn_s4_source::BF16_SYMBOL.into()
            }
            (AdaHalfTnCandidateKind::Tc64Bk32S4, WeightDtype::F16) => {
                triad_half_tn_s4_source::F16_SYMBOL.into()
            }
            (AdaHalfTnCandidateKind::Tc64Bk64S2Compact, WeightDtype::Bf16) => {
                format!("{}bf16", triad_half_tn_compact_source::SYMBOL_PREFIX)
            }
            (AdaHalfTnCandidateKind::Tc64Bk64S2Compact, WeightDtype::F16) => {
                format!("{}f16", triad_half_tn_compact_source::SYMBOL_PREFIX)
            }
            (_, WeightDtype::F32) => panic!("TN microtile requires a half dtype"),
        }
    }

    fn function(&self, dtype: WeightDtype) -> &CudaFunction {
        match dtype {
            WeightDtype::Bf16 => &self.bf16,
            WeightDtype::F16 => &self.f16,
            WeightDtype::F32 => panic!("TN microtile requires a half dtype"),
        }
    }

    const fn geometry(&self) -> (usize, usize, u32, i32) {
        match self.kind {
            AdaHalfTnCandidateKind::Microtile(
                triad_half_tn_microtile_source::Microtile::M32N32,
            ) => (32, 32, 128, 15_360),
            AdaHalfTnCandidateKind::Microtile(
                triad_half_tn_microtile_source::Microtile::M16N32,
            ) => (16, 32, 64, 12_288),
            AdaHalfTnCandidateKind::Rect128x64 => (128, 64, 256, 39_936),
            AdaHalfTnCandidateKind::Tc64Bk32S4 => (64, 64, 128, 36_864),
            AdaHalfTnCandidateKind::Tc64Bk64S2Compact => (64, 64, 128, 32_768),
        }
    }

    const fn schema_name(&self) -> &'static str {
        match self.kind {
            AdaHalfTnCandidateKind::Microtile(_) => "Microtile",
            AdaHalfTnCandidateKind::Rect128x64 => "Rect128x64",
            AdaHalfTnCandidateKind::Tc64Bk32S4 => "Tc64Bk32S4",
            AdaHalfTnCandidateKind::Tc64Bk64S2Compact => "Tc64Bk64S2Compact",
        }
    }

    const fn required_occupancy(&self) -> u32 {
        match self.kind {
            AdaHalfTnCandidateKind::Tc64Bk64S2Compact => 3,
            _ => 1,
        }
    }
}

fn compile_ada_half_tn_candidate(
    t: &Ctx,
    tile: triad_half_tn_microtile_source::Microtile,
) -> Result<AdaHalfTnCandidate, String> {
    let production = include_str!("../kernels/gemm_bi_triad/sm80.cu");
    let transformed = triad_half_tn_microtile_source::candidate_source(production, tile)?;
    let source = [
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        include_str!("../kernels/gemm_bi_triad/mma16.cuh"),
        &transformed,
    ]
    .iter()
    .map(|part| {
        part.lines()
            .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
            .collect::<Vec<_>>()
            .join("\n")
    })
    .collect::<Vec<_>>()
    .join("\n");
    let source_sha256 = format!("{:x}", Sha256::digest(source.as_bytes()));
    let options = cudarc::nvrtc::CompileOptions {
        arch: Some("compute_89"),
        options: vec![
            "--fmad=true".into(),
            "--extra-device-vectorization".into(),
            "-DNDEBUG".into(),
            "-DGEMM_BI_GROUP_M=16".into(),
            "-DMAMBA_RS_STATE_CAP=256".into(),
            "--frandom-seed=1295072049".into(),
        ],
        include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(source, options)
        .map_err(|error| format!("compile {tile:?} half TN candidate: {error:?}"))?;
    let module = t
        .ctx
        .stream
        .context()
        .load_module(ptx)
        .map_err(|error| format!("load {tile:?} half TN candidate: {error:?}"))?;
    let prefix = match tile {
        triad_half_tn_microtile_source::Microtile::M32N32 => {
            triad_half_tn_microtile_source::M32N32_SYMBOL_PREFIX
        }
        triad_half_tn_microtile_source::Microtile::M16N32 => {
            triad_half_tn_microtile_source::M16N32_SYMBOL_PREFIX
        }
    };
    let bf16 = module
        .load_function(&format!("{prefix}bf16"))
        .map_err(|error| format!("load {tile:?} BF16 TN symbol: {error:?}"))?;
    let f16 = module
        .load_function(&format!("{prefix}f16"))
        .map_err(|error| format!("load {tile:?} F16 TN symbol: {error:?}"))?;
    Ok(AdaHalfTnCandidate {
        kind: AdaHalfTnCandidateKind::Microtile(tile),
        bf16,
        f16,
        source_sha256,
    })
}

fn loaded_ada_half_tn_rect_candidate(t: &Ctx) -> AdaHalfTnCandidate {
    AdaHalfTnCandidate {
        kind: AdaHalfTnCandidateKind::Rect128x64,
        bf16: t
            .ctx
            .kernels
            .gemm_bi_tn_tc128x64_typed
            .get(WeightDtype::Bf16)
            .clone(),
        f16: t
            .ctx
            .kernels
            .gemm_bi_tn_tc128x64_typed
            .get(WeightDtype::F16)
            .clone(),
        source_sha256: format!(
            "{:x}",
            Sha256::digest(include_str!("../kernels/gemm_bi_triad/sm80.cu").as_bytes())
        ),
    }
}

fn compile_ada_half_tn_isolated_candidate(
    t: &Ctx,
    kind: AdaHalfTnCandidateKind,
    transformed: String,
    bf16_symbol: &str,
    f16_symbol: &str,
) -> Result<AdaHalfTnCandidate, String> {
    let source = [
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        include_str!("../kernels/gemm_bi_triad/mma16.cuh"),
        &transformed,
    ]
    .iter()
    .map(|part| {
        part.lines()
            .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
            .collect::<Vec<_>>()
            .join("\n")
    })
    .collect::<Vec<_>>()
    .join("\n");
    let source_sha256 = format!("{:x}", Sha256::digest(source.as_bytes()));
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        source,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_89"),
            options: vec![
                "--fmad=true".into(),
                "--extra-device-vectorization".into(),
                "-DNDEBUG".into(),
                "-DGEMM_BI_GROUP_M=16".into(),
                "-DMAMBA_RS_STATE_CAP=256".into(),
                "--frandom-seed=1295072049".into(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        },
    )
    .map_err(|error| format!("compile half TN isolated candidate: {error:?}"))?;
    let module = t
        .ctx
        .stream
        .context()
        .load_module(ptx)
        .map_err(|error| format!("load half TN isolated candidate module: {error:?}"))?;
    Ok(AdaHalfTnCandidate {
        kind,
        bf16: module
            .load_function(bf16_symbol)
            .map_err(|error| format!("load half TN isolated BF16: {error:?}"))?,
        f16: module
            .load_function(f16_symbol)
            .map_err(|error| format!("load half TN isolated F16: {error:?}"))?,
        source_sha256,
    })
}

fn compile_ada_half_tn_s4_candidate(t: &Ctx) -> Result<AdaHalfTnCandidate, String> {
    let transformed =
        triad_half_tn_s4_source::compose_source(include_str!("../kernels/gemm_bi_triad/sm80.cu"))?;
    compile_ada_half_tn_isolated_candidate(
        t,
        AdaHalfTnCandidateKind::Tc64Bk32S4,
        transformed,
        triad_half_tn_s4_source::BF16_SYMBOL,
        triad_half_tn_s4_source::F16_SYMBOL,
    )
}

fn compile_ada_half_tn_compact_candidate(t: &Ctx) -> Result<AdaHalfTnCandidate, String> {
    let transformed = triad_half_tn_compact_source::candidate_source(include_str!(
        "../kernels/gemm_bi_triad/sm80.cu"
    ))?;
    let bf16_symbol = format!("{}bf16", triad_half_tn_compact_source::SYMBOL_PREFIX);
    let f16_symbol = format!("{}f16", triad_half_tn_compact_source::SYMBOL_PREFIX);
    compile_ada_half_tn_isolated_candidate(
        t,
        AdaHalfTnCandidateKind::Tc64Bk64S2Compact,
        transformed,
        &bf16_symbol,
        &f16_symbol,
    )
}

struct AdaHalfTnFixture {
    a: TypedSubview,
    b: TypedSubview,
    candidate: F32Subview,
    current: F32Subview,
    fast: F32Subview,
    a_values: Vec<f32>,
    b_values: Vec<f32>,
    output_seed: Vec<f32>,
    a_bits: Vec<u16>,
    b_bits: Vec<u16>,
    dims: (usize, usize, usize),
    dtype: WeightDtype,
}

impl AdaHalfTnFixture {
    fn new(t: &Ctx, dtype: WeightDtype, dims: (usize, usize, usize)) -> Self {
        Self::new_with_guards(t, dtype, dims, 8, 4)
    }

    fn new_with_guards(
        t: &Ctx,
        dtype: WeightDtype,
        dims: (usize, usize, usize),
        half_guard: usize,
        f32_guard: usize,
    ) -> Self {
        let (m, k, n) = dims;
        let a_values = ada_half_values(m * k, dtype, 0xa89a_7101 ^ k as u32);
        let b_values = ada_half_values(m * n, dtype, 0xb89a_7102 ^ n as u32);
        let mut output_seed = det(k * n, 0xc89a_7103 ^ (k * n) as u32, 0.03125);
        output_seed[0] = 0.0;
        output_seed[1] = -0.0;
        assert_eq!(half_guard % 8, 0);
        assert_eq!(f32_guard % 4, 0);
        let a = TypedSubview::new(t, &a_values, m, k, k, half_guard, dtype);
        let b = TypedSubview::new(t, &b_values, m, n, n, half_guard, dtype);
        let candidate = F32Subview::new(t, &output_seed, k, n, n, f32_guard);
        let current = F32Subview::new(t, &output_seed, k, n, n, f32_guard);
        let fast = F32Subview::new(t, &output_seed, k, n, n, f32_guard);
        let a_bits = a.logical_bits(t);
        let b_bits = b.logical_bits(t);
        Self {
            a,
            b,
            candidate,
            current,
            fast,
            a_values,
            b_values,
            output_seed,
            a_bits,
            b_bits,
            dims,
            dtype,
        }
    }

    fn output(&self, arm: AdaHalfTnArm) -> &F32Subview {
        match arm {
            AdaHalfTnArm::Candidate => &self.candidate,
            AdaHalfTnArm::CurrentTc64 => &self.current,
            AdaHalfTnArm::Fast => &self.fast,
        }
    }

    fn reset(&mut self, t: &Ctx, arm: AdaHalfTnArm) -> Result<(), String> {
        self.a.upload_logical(t, &self.a_values);
        self.b.upload_logical(t, &self.b_values);
        let seed = &self.output_seed;
        match arm {
            AdaHalfTnArm::Candidate => self.candidate.upload_logical(t, seed),
            AdaHalfTnArm::CurrentTc64 => self.current.upload_logical(t, seed),
            AdaHalfTnArm::Fast => self.fast.upload_logical(t, seed),
        }
        t.ctx
            .stream
            .synchronize()
            .map_err(|error| format!("half TN reset synchronize: {error:?}"))
    }

    fn validate_inputs(&self, t: &Ctx) -> Result<(), String> {
        if self.a.logical_bits(t) != self.a_bits || self.b.logical_bits(t) != self.b_bits {
            return Err("half TN input words or guards changed".into());
        }
        Ok(())
    }
}

fn enqueue_ada_half_tn_candidate(
    t: &Ctx,
    fixture: &AdaHalfTnFixture,
    candidate: &AdaHalfTnCandidate,
) -> Result<(), String> {
    let (m, k, n) = fixture.dims;
    let (bm, bn, threads, _) = candidate.geometry();
    let alpha = 1.0f32;
    let (m, k, n) = (m as i32, k as i32, n as i32);
    let output = fixture.candidate.ptr();
    let a = fixture.a.ptr();
    let b = fixture.b.ptr();
    let mut launch = t
        .ctx
        .stream
        .launch_builder(candidate.function(fixture.dtype));
    launch
        .arg(&output)
        .arg(&a)
        .arg(&b)
        .arg(&alpha)
        .arg(&m)
        .arg(&k)
        .arg(&n);
    unsafe {
        launch.launch(cudarc::driver::LaunchConfig {
            grid_dim: (
                (k as usize).div_ceil(bm) as u32 * (n as usize).div_ceil(bn) as u32,
                1,
                1,
            ),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| format!("launch {} half TN candidate: {error:?}", candidate.name()))
}

fn enqueue_ada_half_tn_fast(t: &Ctx, fixture: &AdaHalfTnFixture) -> Result<(), String> {
    use cudarc::cublas::{result, sys};
    let (m, k, n) = fixture.dims;
    let alpha = 1.0f32;
    let beta = 1.0f32;
    unsafe {
        result::gemm_ex(
            *t.ctx.blas.handle(),
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_T,
            n as i32,
            k as i32,
            m as i32,
            (&alpha as *const f32).cast(),
            fixture.b.ptr() as *const _,
            fixture.dtype.cuda_data_type(),
            n as i32,
            fixture.a.ptr() as *const _,
            fixture.dtype.cuda_data_type(),
            k as i32,
            (&beta as *const f32).cast(),
            fixture.fast.ptr() as *mut _,
            sys::cudaDataType::CUDA_R_32F,
            n as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
    .map_err(|error| format!("native-half Fast TN GEMM: {error:?}"))
}

fn enqueue_ada_half_tn_arm(
    t: &Ctx,
    fixture: &AdaHalfTnFixture,
    candidate: &AdaHalfTnCandidate,
    arm: AdaHalfTnArm,
) -> Result<(), String> {
    match arm {
        AdaHalfTnArm::Candidate => enqueue_ada_half_tn_candidate(t, fixture, candidate),
        AdaHalfTnArm::CurrentTc64 => enqueue_tc_tn(
            t,
            BackwardSchedule::Tile64,
            fixture.dtype,
            fixture.current.ptr(),
            fixture.a.ptr(),
            fixture.b.ptr(),
            fixture.dims,
        ),
        AdaHalfTnArm::Fast => enqueue_ada_half_tn_fast(t, fixture),
    }
}

fn capture_ada_half_tn_arm(
    t: &Ctx,
    fixture: &AdaHalfTnFixture,
    candidate: &AdaHalfTnCandidate,
    arm: AdaHalfTnArm,
) -> Result<CudaGraph, String> {
    unsafe {
        capture_into_graph(&t.ctx.stream, || {
            enqueue_ada_half_tn_arm(t, fixture, candidate, arm)
        })
    }
}

fn gate_ada_half_tn_resources(
    candidate: &AdaHalfTnCandidate,
    dtype: WeightDtype,
) -> Result<(), String> {
    let function = candidate.function(dtype);
    let (_, _, threads, expected_static) = candidate.geometry();
    let registers = function
        .num_regs()
        .map_err(|error| format!("half TN registers: {error:?}"))?;
    let local = function
        .local_size_bytes()
        .map_err(|error| format!("half TN local bytes: {error:?}"))?;
    let static_shared = function
        .shared_size_bytes()
        .map_err(|error| format!("half TN static shared: {error:?}"))?;
    let max_threads = function
        .max_threads_per_block()
        .map_err(|error| format!("half TN max threads: {error:?}"))?;
    let occupancy = function
        .occupancy_max_active_blocks_per_multiprocessor(threads, 0, None)
        .map_err(|error| format!("half TN occupancy: {error:?}"))?;
    let required_occupancy = candidate.required_occupancy();
    let schema = candidate.schema_name();
    let source_key = match candidate.kind {
        AdaHalfTnCandidateKind::Microtile(_)
        | AdaHalfTnCandidateKind::Tc64Bk32S4
        | AdaHalfTnCandidateKind::Tc64Bk64S2Compact => "source_sha256",
        AdaHalfTnCandidateKind::Rect128x64 => "source_fragment_sha256",
    };
    println!(
        "{{\"schema\":\"MambaBiHalfTn{schema}ResourceV1\",\"candidate\":\"{}\",\"dtype\":\"{dtype:?}\",\"symbol\":\"{}\",\"{source_key}\":\"{}\",\"threads\":{threads},\"registers\":{registers},\"local_bytes\":{local},\"static_shared_bytes\":{static_shared},\"dynamic_shared_bytes\":0,\"max_threads\":{max_threads},\"occupancy\":{occupancy},\"required_occupancy\":{required_occupancy}}}",
        candidate.name(),
        candidate.symbol(dtype),
        candidate.source_sha256,
    );
    if registers <= 0
        || local != 0
        || static_shared != expected_static
        || max_threads < threads as i32
        || occupancy < required_occupancy
    {
        return Err(format!(
            "half TN {} resource gate failed: regs={registers} local={local} static={static_shared}/{expected_static} max_threads={max_threads}/{threads} occupancy={occupancy}/{required_occupancy}",
            candidate.name()
        ));
    }
    Ok(())
}

fn observe_ada_half_tn(
    t: &Ctx,
    fixture: &mut AdaHalfTnFixture,
    candidate: &AdaHalfTnCandidate,
    graphs: &[CudaGraph; 3],
    arm: AdaHalfTnArm,
    path: AdaHalfPath,
    gemms: usize,
    expected: Option<&[u32]>,
) -> Result<(f64, Vec<u32>), String> {
    fixture.reset(t, arm)?;
    let start = t
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("half TN timing start: {error:?}"))?;
    for _ in 0..gemms {
        match path {
            AdaHalfPath::Eager => enqueue_ada_half_tn_arm(t, fixture, candidate, arm)?,
            AdaHalfPath::Graph => graphs[arm as usize]
                .launch()
                .map_err(|error| format!("half TN graph launch: {error:?}"))?,
        }
    }
    let end = t
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("half TN timing end: {error:?}"))?;
    let elapsed_us = f64::from(
        start
            .elapsed_ms(&end)
            .map_err(|error| format!("half TN timing: {error:?}"))?,
    ) * 1_000.0
        / gemms as f64;
    if !elapsed_us.is_finite() || elapsed_us <= 0.0 {
        return Err(format!("invalid half TN timing {elapsed_us}"));
    }
    let bits = fixture.output(arm).logical_bits(t);
    if let Some(expected) = expected
        && bits != expected
    {
        let mismatch = bits
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
            .unwrap_or(bits.len());
        return Err(format!(
            "half TN {} {} differs from TC64 at word {mismatch}",
            candidate.name(),
            arm.name()
        ));
    }
    fixture.validate_inputs(t)?;
    Ok((elapsed_us, bits))
}

fn screen_ada_half_tn_pair(
    t: &Ctx,
    fixture: &mut AdaHalfTnFixture,
    candidate: &AdaHalfTnCandidate,
    graphs: &[CudaGraph; 3],
    comparator: AdaHalfTnArm,
    candidate_expected: &[u32],
    comparator_expected: &[u32],
    path: AdaHalfPath,
    order: BracketOrder,
) -> Result<[f64; 2], String> {
    for _ in 0..ADA_HALF_WARMUPS {
        observe_ada_half_tn(
            t,
            fixture,
            candidate,
            graphs,
            AdaHalfTnArm::Candidate,
            path,
            ADA_HALF_TN_OBSERVATION_GEMMS,
            Some(candidate_expected),
        )?;
        observe_ada_half_tn(
            t,
            fixture,
            candidate,
            graphs,
            comparator,
            path,
            ADA_HALF_TN_OBSERVATION_GEMMS,
            Some(comparator_expected),
        )?;
    }
    let arms = match order {
        BracketOrder::Abba => [
            AdaHalfTnArm::Candidate,
            comparator,
            comparator,
            AdaHalfTnArm::Candidate,
        ],
        BracketOrder::Baab => [
            comparator,
            AdaHalfTnArm::Candidate,
            AdaHalfTnArm::Candidate,
            comparator,
        ],
    };
    let mut raw_windows = Vec::with_capacity(ADA_HALF_WINDOWS);
    let mut ratios = Vec::with_capacity(ADA_HALF_WINDOWS);
    for _ in 0..ADA_HALF_WINDOWS {
        let mut raw = [0.0; 4];
        for (index, arm) in arms.into_iter().enumerate() {
            let expected = match arm {
                AdaHalfTnArm::Candidate => candidate_expected,
                _ => comparator_expected,
            };
            raw[index] = observe_ada_half_tn(
                t,
                fixture,
                candidate,
                graphs,
                arm,
                path,
                ADA_HALF_TN_OBSERVATION_GEMMS,
                Some(expected),
            )?
            .0;
        }
        ratios.push(candidate_over_reference(raw, order));
        raw_windows.push(raw);
    }
    let p50 = ada_percentile(&ratios, 0.5).ok_or("invalid half TN p50")?;
    let p95 = ada_percentile(&ratios, 0.95).ok_or("invalid half TN p95")?;
    let order = match order {
        BracketOrder::Abba => "ABBA",
        BracketOrder::Baab => "BAAB",
    };
    let raw = raw_windows
        .iter()
        .map(|row| format!("[{:.9},{:.9},{:.9},{:.9}]", row[0], row[1], row[2], row[3]))
        .collect::<Vec<_>>()
        .join(",");
    let schema = candidate.schema_name();
    println!(
        "{{\"schema\":\"MambaBiHalfTn{schema}ScreenV1\",\"candidate\":\"{}\",\"dtype\":\"{:?}\",\"shape\":[{},{},{}],\"comparator\":\"{}\",\"path\":\"{}\",\"order\":\"{order}\",\"windows\":{ADA_HALF_WINDOWS},\"logical_gemms_per_observation\":{ADA_HALF_TN_OBSERVATION_GEMMS},\"raw_observations_us\":[{raw}],\"ratio_direction\":\"candidate_over_comparator\",\"ratio_p50\":{p50:.9},\"ratio_p95\":{p95:.9}}}",
        candidate.name(),
        fixture.dtype,
        fixture.dims.0,
        fixture.dims.1,
        fixture.dims.2,
        comparator.name(),
        path.name(),
    );
    Ok([p50, p95])
}

fn run_ada_half_tn_candidate_cells(
    t: &Ctx,
    candidates: &[AdaHalfTnCandidate],
    cells: &[(&str, (usize, usize, usize))],
    aligned_buffers: bool,
) -> Result<(), String> {
    for candidate in candidates {
        for &(cell, dims) in cells {
            for dtype in [WeightDtype::F16, WeightDtype::Bf16] {
                gate_ada_half_tn_resources(candidate, dtype)?;
                let mut fixture = if aligned_buffers {
                    AdaHalfTnFixture::new_with_guards(t, dtype, dims, 128, 64)
                } else {
                    AdaHalfTnFixture::new(t, dtype, dims)
                };
                for arm in [
                    AdaHalfTnArm::Candidate,
                    AdaHalfTnArm::CurrentTc64,
                    AdaHalfTnArm::Fast,
                ] {
                    fixture.reset(t, arm)?;
                    enqueue_ada_half_tn_arm(t, &fixture, candidate, arm)?;
                    t.ctx
                        .stream
                        .synchronize()
                        .map_err(|error| format!("half TN graph warmup: {error:?}"))?;
                }
                let graphs = [
                    capture_ada_half_tn_arm(t, &fixture, candidate, AdaHalfTnArm::Candidate)?,
                    capture_ada_half_tn_arm(t, &fixture, candidate, AdaHalfTnArm::CurrentTc64)?,
                    capture_ada_half_tn_arm(t, &fixture, candidate, AdaHalfTnArm::Fast)?,
                ];
                validate_single_node_graph(&graphs[0], "candidate")?;
                validate_single_node_graph(&graphs[1], "current_tc64")?;
                validate_nonempty_graph(&graphs[2], "native_half_fast")?;
                let fast_bits = observe_ada_half_tn(
                    t,
                    &mut fixture,
                    candidate,
                    &graphs,
                    AdaHalfTnArm::Fast,
                    AdaHalfPath::Eager,
                    1,
                    None,
                )?
                .1;
                if !fast_bits
                    .iter()
                    .all(|&word| f32::from_bits(word).is_finite())
                    || !fast_bits.iter().any(|&word| f32::from_bits(word) != 0.0)
                {
                    return Err(format!(
                        "{} {dtype:?} {cell} Fast output is non-finite or all-zero",
                        candidate.name()
                    ));
                }
                for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                    observe_ada_half_tn(
                        t,
                        &mut fixture,
                        candidate,
                        &graphs,
                        AdaHalfTnArm::Fast,
                        path,
                        1,
                        Some(&fast_bits),
                    )?;
                }
                let expected = observe_ada_half_tn(
                    t,
                    &mut fixture,
                    candidate,
                    &graphs,
                    AdaHalfTnArm::CurrentTc64,
                    AdaHalfPath::Eager,
                    1,
                    None,
                )?
                .1;
                let schema = candidate.schema_name();
                for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                    for repeat in 0..2 {
                        for arm in [AdaHalfTnArm::CurrentTc64, AdaHalfTnArm::Candidate] {
                            observe_ada_half_tn(
                                t,
                                &mut fixture,
                                candidate,
                                &graphs,
                                arm,
                                path,
                                1,
                                Some(&expected),
                            )?;
                            println!(
                                "{{\"schema\":\"MambaBiHalfTn{schema}BitsV1\",\"candidate\":\"{}\",\"dtype\":\"{dtype:?}\",\"cell\":\"{cell}\",\"arm\":\"{}\",\"path\":\"{}\",\"repeat\":{repeat},\"words\":{}}}",
                                candidate.name(),
                                arm.name(),
                                path.name(),
                                expected.len(),
                            );
                        }
                    }
                }
                let candidate_timing_bits = observe_ada_half_tn(
                    t,
                    &mut fixture,
                    candidate,
                    &graphs,
                    AdaHalfTnArm::Candidate,
                    AdaHalfPath::Eager,
                    ADA_HALF_TN_OBSERVATION_GEMMS,
                    None,
                )?
                .1;
                observe_ada_half_tn(
                    t,
                    &mut fixture,
                    candidate,
                    &graphs,
                    AdaHalfTnArm::Candidate,
                    AdaHalfPath::Graph,
                    ADA_HALF_TN_OBSERVATION_GEMMS,
                    Some(&candidate_timing_bits),
                )?;
                let fast_timing_bits = observe_ada_half_tn(
                    t,
                    &mut fixture,
                    candidate,
                    &graphs,
                    AdaHalfTnArm::Fast,
                    AdaHalfPath::Eager,
                    ADA_HALF_TN_OBSERVATION_GEMMS,
                    None,
                )?
                .1;
                observe_ada_half_tn(
                    t,
                    &mut fixture,
                    candidate,
                    &graphs,
                    AdaHalfTnArm::Fast,
                    AdaHalfPath::Graph,
                    ADA_HALF_TN_OBSERVATION_GEMMS,
                    Some(&fast_timing_bits),
                )?;
                let mut current_diagnostics = Vec::with_capacity(2);
                for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                    let candidate_us = observe_ada_half_tn(
                        t,
                        &mut fixture,
                        candidate,
                        &graphs,
                        AdaHalfTnArm::Candidate,
                        path,
                        ADA_HALF_TN_OBSERVATION_GEMMS,
                        None,
                    )?
                    .0;
                    let current_us = observe_ada_half_tn(
                        t,
                        &mut fixture,
                        candidate,
                        &graphs,
                        AdaHalfTnArm::CurrentTc64,
                        path,
                        ADA_HALF_TN_OBSERVATION_GEMMS,
                        None,
                    )?
                    .0;
                    current_diagnostics.push(candidate_us / current_us);
                }
                let mut fast_strata = Vec::with_capacity(4);
                for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                    for order in [BracketOrder::Abba, BracketOrder::Baab] {
                        fast_strata.push(screen_ada_half_tn_pair(
                            t,
                            &mut fixture,
                            candidate,
                            &graphs,
                            AdaHalfTnArm::Fast,
                            &candidate_timing_bits,
                            &fast_timing_bits,
                            path,
                            order,
                        )?);
                    }
                }
                let retain = retain_decision(&fast_strata);
                println!(
                    "{{\"schema\":\"MambaBiHalfTn{schema}DecisionV1\",\"candidate\":\"{}\",\"dtype\":\"{dtype:?}\",\"cell\":\"{cell}\",\"shape\":[{},{},{}],\"current_diagnostic_candidate_over_current\":{:?},\"current_diagnostic_order\":[\"eager\",\"graph\"],\"fast_strata\":{:?},\"strata_order\":[\"eager/ABBA\",\"eager/BAAB\",\"graph/ABBA\",\"graph/BAAB\"],\"retain\":{retain},\"decision\":\"{}\",\"promotion\":false}}",
                    candidate.name(),
                    dims.0,
                    dims.1,
                    dims.2,
                    current_diagnostics,
                    fast_strata,
                    if retain { "advance" } else { "stop_no_retry" },
                );
            }
        }
    }
    Ok(())
}

fn run_ada_half_tn_microtile_batch() -> Result<(), String> {
    assert!(
        !cfg!(debug_assertions),
        "half TN discovery requires --release"
    );
    let quiet = QuietGpu::for_cuda_ordinal(0)?;
    let _pre = quiet.require_pre_context("half-tn-microtile/pre-context")?;
    let t = Ctx::new_ada()?;
    let compiler = t.ctx.kernels.compiler_identity();
    if compiler.nvrtc_version != (13, 2) {
        return Err(format!(
            "half TN discovery requires CUDA 13.2, found {:?}",
            compiler.nvrtc_version
        ));
    }
    let candidates = [
        compile_ada_half_tn_candidate(&t, triad_half_tn_microtile_source::Microtile::M32N32)?,
        compile_ada_half_tn_candidate(&t, triad_half_tn_microtile_source::Microtile::M16N32)?,
    ];
    let _cohort = quiet.require_cohort("half-tn-microtile/cohort")?;
    for candidate in &candidates {
        for (cell, dims) in [
            ("d128_in_proj", ADA_HALF_TN_D128_IN),
            ("d128_out_proj", ADA_HALF_TN_D128_OUT),
        ] {
            for dtype in [WeightDtype::F16, WeightDtype::Bf16] {
                gate_ada_half_tn_resources(candidate, dtype)?;
                let mut fixture = AdaHalfTnFixture::new(&t, dtype, dims);
                for arm in [
                    AdaHalfTnArm::Candidate,
                    AdaHalfTnArm::CurrentTc64,
                    AdaHalfTnArm::Fast,
                ] {
                    fixture.reset(&t, arm)?;
                    enqueue_ada_half_tn_arm(&t, &fixture, candidate, arm)?;
                    t.ctx
                        .stream
                        .synchronize()
                        .map_err(|error| format!("half TN graph warmup: {error:?}"))?;
                }
                let graphs = [
                    capture_ada_half_tn_arm(&t, &fixture, candidate, AdaHalfTnArm::Candidate)?,
                    capture_ada_half_tn_arm(&t, &fixture, candidate, AdaHalfTnArm::CurrentTc64)?,
                    capture_ada_half_tn_arm(&t, &fixture, candidate, AdaHalfTnArm::Fast)?,
                ];
                validate_single_node_graph(&graphs[0], "candidate")?;
                validate_single_node_graph(&graphs[1], "current_tc64")?;
                validate_nonempty_graph(&graphs[2], "native_half_fast")?;
                let fast_bits = observe_ada_half_tn(
                    &t,
                    &mut fixture,
                    candidate,
                    &graphs,
                    AdaHalfTnArm::Fast,
                    AdaHalfPath::Eager,
                    1,
                    None,
                )?
                .1;
                if !fast_bits
                    .iter()
                    .all(|&word| f32::from_bits(word).is_finite())
                    || !fast_bits.iter().any(|&word| f32::from_bits(word) != 0.0)
                {
                    return Err(format!(
                        "{} {dtype:?} {cell} Fast output is non-finite or all-zero",
                        candidate.name()
                    ));
                }
                for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                    observe_ada_half_tn(
                        &t,
                        &mut fixture,
                        candidate,
                        &graphs,
                        AdaHalfTnArm::Fast,
                        path,
                        1,
                        Some(&fast_bits),
                    )?;
                }
                let expected = observe_ada_half_tn(
                    &t,
                    &mut fixture,
                    candidate,
                    &graphs,
                    AdaHalfTnArm::CurrentTc64,
                    AdaHalfPath::Eager,
                    1,
                    None,
                )?
                .1;
                for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                    for repeat in 0..2 {
                        for arm in [AdaHalfTnArm::CurrentTc64, AdaHalfTnArm::Candidate] {
                            observe_ada_half_tn(
                                &t,
                                &mut fixture,
                                candidate,
                                &graphs,
                                arm,
                                path,
                                1,
                                Some(&expected),
                            )?;
                            println!(
                                "{{\"schema\":\"MambaBiHalfTnMicrotileBitsV1\",\"candidate\":\"{}\",\"dtype\":\"{dtype:?}\",\"cell\":\"{cell}\",\"arm\":\"{}\",\"path\":\"{}\",\"repeat\":{repeat},\"words\":{}}}",
                                candidate.name(),
                                arm.name(),
                                path.name(),
                                expected.len(),
                            );
                        }
                    }
                }
                let candidate_timing_bits = observe_ada_half_tn(
                    &t,
                    &mut fixture,
                    candidate,
                    &graphs,
                    AdaHalfTnArm::Candidate,
                    AdaHalfPath::Eager,
                    ADA_HALF_TN_OBSERVATION_GEMMS,
                    None,
                )?
                .1;
                observe_ada_half_tn(
                    &t,
                    &mut fixture,
                    candidate,
                    &graphs,
                    AdaHalfTnArm::Candidate,
                    AdaHalfPath::Graph,
                    ADA_HALF_TN_OBSERVATION_GEMMS,
                    Some(&candidate_timing_bits),
                )?;
                let fast_timing_bits = observe_ada_half_tn(
                    &t,
                    &mut fixture,
                    candidate,
                    &graphs,
                    AdaHalfTnArm::Fast,
                    AdaHalfPath::Eager,
                    ADA_HALF_TN_OBSERVATION_GEMMS,
                    None,
                )?
                .1;
                observe_ada_half_tn(
                    &t,
                    &mut fixture,
                    candidate,
                    &graphs,
                    AdaHalfTnArm::Fast,
                    AdaHalfPath::Graph,
                    ADA_HALF_TN_OBSERVATION_GEMMS,
                    Some(&fast_timing_bits),
                )?;
                let mut current_diagnostics = Vec::with_capacity(2);
                for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                    let candidate_us = observe_ada_half_tn(
                        &t,
                        &mut fixture,
                        candidate,
                        &graphs,
                        AdaHalfTnArm::Candidate,
                        path,
                        ADA_HALF_TN_OBSERVATION_GEMMS,
                        None,
                    )?
                    .0;
                    let current_us = observe_ada_half_tn(
                        &t,
                        &mut fixture,
                        candidate,
                        &graphs,
                        AdaHalfTnArm::CurrentTc64,
                        path,
                        ADA_HALF_TN_OBSERVATION_GEMMS,
                        None,
                    )?
                    .0;
                    current_diagnostics.push(candidate_us / current_us);
                }
                let mut fast_strata = Vec::with_capacity(4);
                for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                    for order in [BracketOrder::Abba, BracketOrder::Baab] {
                        fast_strata.push(screen_ada_half_tn_pair(
                            &t,
                            &mut fixture,
                            candidate,
                            &graphs,
                            AdaHalfTnArm::Fast,
                            &candidate_timing_bits,
                            &fast_timing_bits,
                            path,
                            order,
                        )?);
                    }
                }
                let retain = retain_decision(&fast_strata);
                println!(
                    "{{\"schema\":\"MambaBiHalfTnMicrotileDecisionV1\",\"candidate\":\"{}\",\"dtype\":\"{dtype:?}\",\"cell\":\"{cell}\",\"shape\":[{},{},{}],\"current_diagnostic_candidate_over_current\":{:?},\"current_diagnostic_order\":[\"eager\",\"graph\"],\"fast_strata\":{:?},\"strata_order\":[\"eager/ABBA\",\"eager/BAAB\",\"graph/ABBA\",\"graph/BAAB\"],\"retain\":{retain},\"decision\":\"{}\",\"promotion\":false}}",
                    candidate.name(),
                    dims.0,
                    dims.1,
                    dims.2,
                    current_diagnostics,
                    fast_strata,
                    if retain { "advance" } else { "stop_no_retry" },
                );
            }
        }
    }
    drop(t);
    quiet
        .verify_post_cohort("half-tn-microtile/post")
        .map(|_| ())
}

#[test]
#[ignore = "requires exclusive Ada CC8.9 CUDA13.2; two-arm half TN microtile discovery"]
fn ada_half_tn_d128_microtiles_vs_current_and_fast_discovery_once7() -> Result<(), String> {
    run_ada_half_tn_microtile_batch()
}

fn run_ada_half_tn_rect128x64_batch() -> Result<(), String> {
    assert!(
        !cfg!(debug_assertions),
        "half TN Rect128x64 discovery requires --release"
    );
    let quiet = QuietGpu::for_cuda_ordinal(0)?;
    let _pre = quiet.require_pre_context("half-tn-rect128x64/pre-context")?;
    let t = Ctx::new_ada()?;
    let compiler = t.ctx.kernels.compiler_identity();
    if compiler.nvrtc_version != (13, 2) {
        return Err(format!(
            "half TN Rect128x64 requires CUDA13.2, found {:?}",
            compiler.nvrtc_version
        ));
    }
    let candidate = loaded_ada_half_tn_rect_candidate(&t);
    let _cohort = quiet.require_cohort("half-tn-rect128x64/cohort")?;
    run_ada_half_tn_candidate_cells(
        &t,
        &[candidate],
        &[
            ("d768_in_proj", (2_048, 768, 3_072)),
            ("d768_out_proj", (2_048, 1_536, 768)),
            ("prism_in_proj", (4_621, 384, 1_928)),
        ],
        true,
    )?;
    drop(t);
    quiet
        .verify_post_cohort("half-tn-rect128x64/post")
        .map(|_| ())
}

#[test]
#[ignore = "requires exclusive Ada CC8.9 CUDA13.2; loaded half TN Rect128x64 discovery"]
fn ada_half_tn_rect128x64_three_cell_vs_current_and_fast_discovery_once7() -> Result<(), String> {
    run_ada_half_tn_rect128x64_batch()
}

fn run_ada_half_tn_tc64_bk32_s4_batch() -> Result<(), String> {
    assert!(
        !cfg!(debug_assertions),
        "half TN TC64/BK32/S4 discovery requires --release"
    );
    let quiet = QuietGpu::for_cuda_ordinal(0)?;
    let _pre = quiet.require_pre_context("half-tn-tc64-bk32-s4/pre-context")?;
    let t = Ctx::new_ada()?;
    let compiler = t.ctx.kernels.compiler_identity();
    if compiler.nvrtc_version != (13, 2) {
        return Err(format!(
            "half TN TC64/BK32/S4 requires CUDA13.2, found {:?}",
            compiler.nvrtc_version
        ));
    }
    let candidate = compile_ada_half_tn_s4_candidate(&t)?;
    let _cohort = quiet.require_cohort("half-tn-tc64-bk32-s4/cohort")?;
    run_ada_half_tn_candidate_cells(
        &t,
        &[candidate],
        &[
            ("d768_in_proj", (2_048, 768, 3_072)),
            ("d768_out_proj", (2_048, 1_536, 768)),
            ("prism_in_proj", (4_621, 384, 1_928)),
        ],
        true,
    )?;
    drop(t);
    quiet
        .verify_post_cohort("half-tn-tc64-bk32-s4/post")
        .map(|_| ())
}

#[test]
#[ignore = "requires exclusive Ada CC8.9 CUDA13.2; half TN TC64/BK32/S4 discovery"]
fn ada_half_tn_tc64_bk32_s4_three_cell_vs_current_and_fast_discovery_once7() -> Result<(), String> {
    run_ada_half_tn_tc64_bk32_s4_batch()
}

fn run_ada_half_tn_tc64_bk64_s2_compact_batch() -> Result<(), String> {
    assert!(
        !cfg!(debug_assertions),
        "half TN compact BK64/S2 discovery requires --release"
    );
    let quiet = QuietGpu::for_cuda_ordinal(0)?;
    let _pre = quiet.require_pre_context("half-tn-compact-bk64-s2/pre-context")?;
    let t = Ctx::new_ada()?;
    let compiler = t.ctx.kernels.compiler_identity();
    if compiler.nvrtc_version != (13, 2) {
        return Err(format!(
            "half TN compact BK64/S2 requires CUDA13.2, found {:?}",
            compiler.nvrtc_version
        ));
    }
    let candidate = compile_ada_half_tn_compact_candidate(&t)?;
    let _cohort = quiet.require_cohort("half-tn-compact-bk64-s2/cohort")?;
    run_ada_half_tn_candidate_cells(
        &t,
        &[candidate],
        &[
            ("d768_in_proj", (2_048, 768, 3_072)),
            ("d768_out_proj", (2_048, 1_536, 768)),
            ("prism_in_proj", (4_621, 384, 1_928)),
        ],
        true,
    )?;
    drop(t);
    quiet
        .verify_post_cohort("half-tn-compact-bk64-s2/post")
        .map(|_| ())
}

#[test]
#[ignore = "requires exclusive Ada CC8.9 CUDA13.2; half TN compact BK64/S2 discovery"]
fn ada_half_tn_tc64_bk64_s2_compact_three_cell_vs_current_and_fast_discovery_once7()
-> Result<(), String> {
    run_ada_half_tn_tc64_bk64_s2_compact_batch()
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

fn enqueue_tc_nt(
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
enum AdaHalfNtArm {
    Candidate,
    CurrentTc64,
    Fast,
}

impl AdaHalfNtArm {
    const fn name(self, candidate: AdaHalfNtCandidateKind) -> &'static str {
        match self {
            Self::Candidate => candidate.name(),
            Self::CurrentTc64 => "forced_tc64",
            Self::Fast => "native_half_fast",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdaHalfNtCandidateKind {
    Bk32S3,
    CompactBk64S2,
    LoadedTc128,
}

impl AdaHalfNtCandidateKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Bk32S3 => "tc64_bk32_s3",
            Self::CompactBk64S2 => "tc64_bk64_s2_compact_xor",
            Self::LoadedTc128 => "loaded_tc128_bk64_s2",
        }
    }

    const fn resource_schema(self) -> &'static str {
        match self {
            Self::Bk32S3 => "MambaBiHalfNtS3ResourceV1",
            Self::CompactBk64S2 => "MambaBiHalfNtCompactResourceV1",
            Self::LoadedTc128 => "MambaBiHalfNtLoadedTc128ResourceV1",
        }
    }

    const fn bits_schema(self) -> &'static str {
        match self {
            Self::Bk32S3 => "MambaBiHalfNtS3BitsV1",
            Self::CompactBk64S2 => "MambaBiHalfNtCompactBitsV1",
            Self::LoadedTc128 => "MambaBiHalfNtLoadedTc128BitsV1",
        }
    }

    const fn screen_schema(self) -> &'static str {
        match self {
            Self::Bk32S3 => "MambaBiHalfNtS3FastScreenV1",
            Self::CompactBk64S2 => "MambaBiHalfNtCompactFastScreenV1",
            Self::LoadedTc128 => "MambaBiHalfNtLoadedTc128ScreenV1",
        }
    }

    const fn decision_schema(self) -> &'static str {
        match self {
            Self::Bk32S3 => "MambaBiHalfNtS3FastDecisionV1",
            Self::CompactBk64S2 => "MambaBiHalfNtCompactFastDecisionV1",
            Self::LoadedTc128 => "MambaBiHalfNtLoadedTc128DecisionV1",
        }
    }

    const fn static_shared_bytes(self) -> i32 {
        match self {
            Self::Bk32S3 => 30_720,
            Self::CompactBk64S2 => 32_768,
            Self::LoadedTc128 => 0,
        }
    }

    const fn launch_resources(self) -> (u32, usize, u32) {
        match self {
            Self::Bk32S3 | Self::CompactBk64S2 => (128, 0, 3),
            Self::LoadedTc128 => (256, 73_728, 1),
        }
    }
}

struct AdaHalfNtS3Candidate {
    kind: AdaHalfNtCandidateKind,
    bf16: CudaFunction,
    f16: CudaFunction,
    source_sha256: String,
}

impl AdaHalfNtS3Candidate {
    fn function(&self, dtype: WeightDtype) -> &CudaFunction {
        match dtype {
            WeightDtype::Bf16 => &self.bf16,
            WeightDtype::F16 => &self.f16,
            WeightDtype::F32 => panic!("NT S3 candidate requires a half dtype"),
        }
    }
}

fn compile_ada_half_nt_candidate(
    t: &Ctx,
    kind: AdaHalfNtCandidateKind,
) -> Result<AdaHalfNtS3Candidate, String> {
    let production = include_str!("../kernels/gemm_bi_triad/sm80.cu");
    let (transformed, symbol_prefix) = match kind {
        AdaHalfNtCandidateKind::Bk32S3 => (
            triad_half_nt_s3_source::candidate_source(production)?,
            triad_half_nt_s3_source::SYMBOL_PREFIX,
        ),
        AdaHalfNtCandidateKind::CompactBk64S2 => (
            triad_half_nt_compact_source::candidate_source(production)?,
            triad_half_nt_compact_source::SYMBOL_PREFIX,
        ),
        AdaHalfNtCandidateKind::LoadedTc128 => {
            return Err("loaded NT TC128 does not require NVRTC composition".into());
        }
    };
    let source = [
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        include_str!("../kernels/gemm_bi_triad/mma16.cuh"),
        &transformed,
    ]
    .iter()
    .map(|part| {
        part.lines()
            .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
            .collect::<Vec<_>>()
            .join("\n")
    })
    .collect::<Vec<_>>()
    .join("\n");
    let source_sha256 = format!("{:x}", Sha256::digest(source.as_bytes()));
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        source,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_89"),
            options: vec![
                "--fmad=true".into(),
                "--extra-device-vectorization".into(),
                "-DNDEBUG".into(),
                "-DGEMM_BI_GROUP_M=16".into(),
                "-DMAMBA_RS_STATE_CAP=256".into(),
                "--frandom-seed=1295072049".into(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        },
    )
    .map_err(|error| format!("compile half NT {} candidate: {error:?}", kind.name()))?;
    let module = t
        .ctx
        .stream
        .context()
        .load_module(ptx)
        .map_err(|error| format!("load half NT {} module: {error:?}", kind.name()))?;
    let load = |suffix| {
        let symbol = format!("{symbol_prefix}{suffix}");
        module
            .load_function(&symbol)
            .map_err(|error| format!("load {symbol}: {error:?}"))
    };
    Ok(AdaHalfNtS3Candidate {
        kind,
        bf16: load("bf16")?,
        f16: load("f16")?,
        source_sha256,
    })
}

fn loaded_ada_half_nt_tc128_candidate(t: &Ctx) -> AdaHalfNtS3Candidate {
    AdaHalfNtS3Candidate {
        kind: AdaHalfNtCandidateKind::LoadedTc128,
        bf16: t
            .ctx
            .kernels
            .gemm_bi_nt_tc_typed
            .get(WeightDtype::Bf16)
            .clone(),
        f16: t
            .ctx
            .kernels
            .gemm_bi_nt_tc_typed
            .get(WeightDtype::F16)
            .clone(),
        source_sha256: format!(
            "{:x}",
            Sha256::digest(include_str!("../kernels/gemm_bi_triad/sm80.cu").as_bytes())
        ),
    }
}

struct AdaHalfNtS3Fixture {
    a: TypedSubview,
    b: TypedSubview,
    outputs: [TypedSubview; 3],
    a_values: Vec<f32>,
    b_values: Vec<f32>,
    output_seed: Vec<f32>,
    a_bits: Vec<u16>,
    b_bits: Vec<u16>,
    dtype: WeightDtype,
}

impl AdaHalfNtS3Fixture {
    fn new(t: &Ctx, dtype: WeightDtype) -> Self {
        Self::new_with_guard(t, dtype, 8)
    }

    fn new_with_guard(t: &Ctx, dtype: WeightDtype, guard_offset: usize) -> Self {
        let (m, k, n) = ADA_HALF_NT_D768_OUT;
        let a_values = ada_half_values(m * n, dtype, 0xa89a_8201);
        let b_values = ada_half_values(k * n, dtype, 0xb89a_8202);
        let output_seed = ada_half_values(m * k, dtype, 0xc89a_8203);
        let a = TypedSubview::new(t, &a_values, m, n, n, guard_offset, dtype);
        let b = TypedSubview::new(t, &b_values, k, n, n, guard_offset, dtype);
        let make_output = || TypedSubview::new(t, &output_seed, m, k, k, guard_offset, dtype);
        let outputs = [make_output(), make_output(), make_output()];
        let a_bits = a.logical_bits(t);
        let b_bits = b.logical_bits(t);
        Self {
            a,
            b,
            outputs,
            a_values,
            b_values,
            output_seed,
            a_bits,
            b_bits,
            dtype,
        }
    }

    fn output(&self, arm: AdaHalfNtArm) -> &TypedSubview {
        &self.outputs[arm as usize]
    }

    fn reset(&self, t: &Ctx, arm: AdaHalfNtArm) -> Result<(), String> {
        self.a.upload_logical(t, &self.a_values);
        self.b.upload_logical(t, &self.b_values);
        self.output(arm).upload_logical(t, &self.output_seed);
        t.ctx
            .stream
            .synchronize()
            .map_err(|error| format!("half NT reset: {error:?}"))?;
        self.validate_inputs(t)
    }

    fn validate_inputs(&self, t: &Ctx) -> Result<(), String> {
        if self.a.logical_bits(t) != self.a_bits || self.b.logical_bits(t) != self.b_bits {
            return Err("half NT input words or guards changed".into());
        }
        Ok(())
    }
}

fn enqueue_ada_half_nt_s3_arm(
    t: &Ctx,
    fixture: &AdaHalfNtS3Fixture,
    candidate: &AdaHalfNtS3Candidate,
    arm: AdaHalfNtArm,
) -> Result<(), String> {
    let (m, k, n) = ADA_HALF_NT_D768_OUT;
    match arm {
        AdaHalfNtArm::Candidate if candidate.kind == AdaHalfNtCandidateKind::LoadedTc128 => {
            enqueue_tc_nt(
                t,
                BackwardSchedule::Tile128,
                fixture.dtype,
                fixture.output(arm).ptr(),
                fixture.a.ptr(),
                fixture.b.ptr(),
                ADA_HALF_NT_D768_OUT,
            )
        }
        AdaHalfNtArm::Candidate => {
            let output = fixture.output(arm).ptr();
            let a = fixture.a.ptr();
            let b = fixture.b.ptr();
            let alpha = 1.0f32;
            let (m, n, k) = (m as i32, n as i32, k as i32);
            let mut launch = t
                .ctx
                .stream
                .launch_builder(candidate.function(fixture.dtype));
            launch
                .arg(&output)
                .arg(&a)
                .arg(&b)
                .arg(&alpha)
                .arg(&m)
                .arg(&n)
                .arg(&k);
            unsafe {
                launch.launch(cudarc::driver::LaunchConfig {
                    grid_dim: (32 * 24, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|error| format!("half NT {} launch: {error:?}", candidate.kind.name()))
        }
        AdaHalfNtArm::CurrentTc64 => enqueue_tc_nt(
            t,
            BackwardSchedule::Tile64,
            fixture.dtype,
            fixture.output(arm).ptr(),
            fixture.a.ptr(),
            fixture.b.ptr(),
            ADA_HALF_NT_D768_OUT,
        ),
        AdaHalfNtArm::Fast => {
            use cudarc::cublas::{result, sys as blas_sys};
            let alpha = 1.0f32;
            let beta = 0.0f32;
            unsafe {
                result::gemm_ex(
                    *t.ctx.blas.handle(),
                    blas_sys::cublasOperation_t::CUBLAS_OP_T,
                    blas_sys::cublasOperation_t::CUBLAS_OP_N,
                    k as i32,
                    m as i32,
                    n as i32,
                    (&alpha as *const f32).cast(),
                    fixture.b.ptr() as *const _,
                    fixture.dtype.cuda_data_type(),
                    n as i32,
                    fixture.a.ptr() as *const _,
                    fixture.dtype.cuda_data_type(),
                    n as i32,
                    (&beta as *const f32).cast(),
                    fixture.output(arm).ptr() as *mut _,
                    fixture.dtype.cuda_data_type(),
                    k as i32,
                    blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    blas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
            }
            .map_err(|error| format!("native-half Fast NT GEMM: {error:?}"))
        }
    }
}

fn gate_ada_half_nt_s3_resources(
    candidate: &AdaHalfNtS3Candidate,
    dtype: WeightDtype,
) -> Result<(), String> {
    let function = candidate.function(dtype);
    let (threads, dynamic_shared, required_occupancy) = candidate.kind.launch_resources();
    let registers = function
        .num_regs()
        .map_err(|error| format!("NT S3 registers: {error:?}"))?;
    let local = function
        .local_size_bytes()
        .map_err(|error| format!("NT S3 local: {error:?}"))?;
    let static_shared = function
        .shared_size_bytes()
        .map_err(|error| format!("NT S3 static: {error:?}"))?;
    let max_threads = function
        .max_threads_per_block()
        .map_err(|error| format!("NT S3 max threads: {error:?}"))?;
    let occupancy = function
        .occupancy_max_active_blocks_per_multiprocessor(threads, dynamic_shared, None)
        .map_err(|error| format!("NT S3 occupancy: {error:?}"))?;
    let schema = candidate.kind.resource_schema();
    let expected_static = candidate.kind.static_shared_bytes();
    let source_key = if candidate.kind == AdaHalfNtCandidateKind::LoadedTc128 {
        "source_fragment_sha256"
    } else {
        "source_sha256"
    };
    println!(
        "{{\"schema\":\"{schema}\",\"dtype\":\"{dtype:?}\",\"candidate\":\"{}\",\"{source_key}\":\"{}\",\"threads\":{threads},\"registers\":{registers},\"local_bytes\":{local},\"static_shared_bytes\":{static_shared},\"dynamic_shared_bytes\":{dynamic_shared},\"max_threads\":{max_threads},\"occupancy\":{occupancy},\"required_occupancy\":{required_occupancy}}}",
        candidate.kind.name(),
        candidate.source_sha256,
    );
    if registers <= 0
        || local != 0
        || static_shared != expected_static
        || max_threads < threads as i32
        || occupancy < required_occupancy
    {
        return Err(format!(
            "half NT {} resource gate failed: regs={registers} local={local} static={static_shared}/{expected_static} dynamic={dynamic_shared} max_threads={max_threads}/{threads} occupancy={occupancy}/{required_occupancy}",
            candidate.kind.name(),
        ));
    }
    Ok(())
}

fn capture_ada_half_nt_s3_arm(
    t: &Ctx,
    fixture: &AdaHalfNtS3Fixture,
    candidate: &AdaHalfNtS3Candidate,
    arm: AdaHalfNtArm,
) -> Result<CudaGraph, String> {
    unsafe {
        capture_into_graph(&t.ctx.stream, || {
            enqueue_ada_half_nt_s3_arm(t, fixture, candidate, arm)
        })
    }
}

fn observe_ada_half_nt_s3(
    t: &Ctx,
    fixture: &AdaHalfNtS3Fixture,
    candidate: &AdaHalfNtS3Candidate,
    graphs: &[CudaGraph; 3],
    arm: AdaHalfNtArm,
    path: AdaHalfPath,
    gemms: usize,
    expected: Option<&[u16]>,
) -> Result<(f64, Vec<u16>), String> {
    fixture.reset(t, arm)?;
    let start = t
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("half NT timing start: {error:?}"))?;
    for _ in 0..gemms {
        match path {
            AdaHalfPath::Eager => enqueue_ada_half_nt_s3_arm(t, fixture, candidate, arm)?,
            AdaHalfPath::Graph => graphs[arm as usize]
                .launch()
                .map_err(|error| format!("half NT graph launch: {error:?}"))?,
        }
    }
    let end = t
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("half NT timing end: {error:?}"))?;
    let elapsed_us = f64::from(
        start
            .elapsed_ms(&end)
            .map_err(|error| format!("half NT timing: {error:?}"))?,
    ) * 1_000.0
        / gemms as f64;
    if !elapsed_us.is_finite() || elapsed_us <= 0.0 {
        return Err(format!("invalid half NT timing {elapsed_us}"));
    }
    let bits = fixture.output(arm).logical_bits(t);
    if let Some(expected) = expected
        && bits != expected
    {
        let mismatch = bits
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
            .unwrap_or(bits.len());
        return Err(format!(
            "half NT {} {} differs at word {mismatch}",
            arm.name(candidate.kind),
            path.name()
        ));
    }
    fixture.validate_inputs(t)?;
    Ok((elapsed_us, bits))
}

fn screen_ada_half_nt_s3_pair(
    t: &Ctx,
    fixture: &AdaHalfNtS3Fixture,
    candidate: &AdaHalfNtS3Candidate,
    graphs: &[CudaGraph; 3],
    candidate_bits: &[u16],
    comparator: AdaHalfNtArm,
    comparator_bits: &[u16],
    path: AdaHalfPath,
    order: BracketOrder,
) -> Result<[f64; 2], String> {
    const GEMMS: usize = 20;
    for _ in 0..ADA_HALF_WARMUPS {
        observe_ada_half_nt_s3(
            t,
            fixture,
            candidate,
            graphs,
            AdaHalfNtArm::Candidate,
            path,
            GEMMS,
            Some(candidate_bits),
        )?;
        observe_ada_half_nt_s3(
            t,
            fixture,
            candidate,
            graphs,
            comparator,
            path,
            GEMMS,
            Some(comparator_bits),
        )?;
    }
    let arms = match order {
        BracketOrder::Abba => [
            AdaHalfNtArm::Candidate,
            comparator,
            comparator,
            AdaHalfNtArm::Candidate,
        ],
        BracketOrder::Baab => [
            comparator,
            AdaHalfNtArm::Candidate,
            AdaHalfNtArm::Candidate,
            comparator,
        ],
    };
    let mut raw_windows = Vec::with_capacity(ADA_HALF_WINDOWS);
    let mut ratios = Vec::with_capacity(ADA_HALF_WINDOWS);
    for _ in 0..ADA_HALF_WINDOWS {
        let mut raw = [0.0; 4];
        for (index, arm) in arms.into_iter().enumerate() {
            let expected = if arm == AdaHalfNtArm::Candidate {
                candidate_bits
            } else {
                comparator_bits
            };
            raw[index] = observe_ada_half_nt_s3(
                t,
                fixture,
                candidate,
                graphs,
                arm,
                path,
                GEMMS,
                Some(expected),
            )?
            .0;
        }
        ratios.push(candidate_over_reference(raw, order));
        raw_windows.push(raw);
    }
    let p50 = ada_percentile(&ratios, 0.5).ok_or("invalid half NT candidate p50")?;
    let p95 = ada_percentile(&ratios, 0.95).ok_or("invalid half NT candidate p95")?;
    let order = match order {
        BracketOrder::Abba => "ABBA",
        BracketOrder::Baab => "BAAB",
    };
    let raw = raw_windows
        .iter()
        .map(|row| format!("[{:.9},{:.9},{:.9},{:.9}]", row[0], row[1], row[2], row[3]))
        .collect::<Vec<_>>()
        .join(",");
    let schema = candidate.kind.screen_schema();
    println!(
        "{{\"schema\":\"{schema}\",\"dtype\":\"{:?}\",\"cell\":\"d768_out_proj\",\"shape\":[2048,1536,768],\"candidate\":\"{}\",\"comparator\":\"{}\",\"path\":\"{}\",\"order\":\"{order}\",\"windows\":{ADA_HALF_WINDOWS},\"logical_gemms_per_observation\":{GEMMS},\"raw_observations_us\":[{raw}],\"ratio_direction\":\"candidate_over_comparator\",\"ratio_p50\":{p50:.9},\"ratio_p95\":{p95:.9}}}",
        fixture.dtype,
        candidate.kind.name(),
        comparator.name(candidate.kind),
        path.name(),
    );
    Ok([p50, p95])
}

fn run_ada_half_nt_d768_out_batch(
    kind: AdaHalfNtCandidateKind,
    guard_offset: usize,
    comparators: &[AdaHalfNtArm],
) -> Result<(), String> {
    assert!(
        !cfg!(debug_assertions),
        "half NT candidate requires --release"
    );
    let quiet = QuietGpu::for_cuda_ordinal(0)?;
    let _pre = quiet.require_pre_context("half-nt-candidate/pre-context")?;
    let t = Ctx::new_ada()?;
    let compiler = t.ctx.kernels.compiler_identity();
    if compiler.nvrtc_version != (13, 2) {
        return Err(format!(
            "half NT candidate requires CUDA13.2, found {:?}",
            compiler.nvrtc_version
        ));
    }
    let candidate = if kind == AdaHalfNtCandidateKind::LoadedTc128 {
        loaded_ada_half_nt_tc128_candidate(&t)
    } else {
        compile_ada_half_nt_candidate(&t, kind)?
    };
    let _cohort = quiet.require_cohort("half-nt-candidate/cohort")?;
    for dtype in [WeightDtype::F16, WeightDtype::Bf16] {
        gate_ada_half_nt_s3_resources(&candidate, dtype)?;
        let fixture = AdaHalfNtS3Fixture::new_with_guard(&t, dtype, guard_offset);
        if kind == AdaHalfNtCandidateKind::LoadedTc128
            && (fixture.a.ptr() % 256 != 0
                || fixture.b.ptr() % 256 != 0
                || fixture.outputs.iter().any(|output| output.ptr() % 256 != 0))
        {
            return Err(format!(
                "{dtype:?} loaded NT TC128 timing fixture is not 256-byte aligned"
            ));
        }
        for arm in [
            AdaHalfNtArm::Candidate,
            AdaHalfNtArm::CurrentTc64,
            AdaHalfNtArm::Fast,
        ] {
            fixture.reset(&t, arm)?;
            enqueue_ada_half_nt_s3_arm(&t, &fixture, &candidate, arm)?;
            t.ctx
                .stream
                .synchronize()
                .map_err(|error| format!("half NT graph warmup: {error:?}"))?;
        }
        let graphs = [
            capture_ada_half_nt_s3_arm(&t, &fixture, &candidate, AdaHalfNtArm::Candidate)?,
            capture_ada_half_nt_s3_arm(&t, &fixture, &candidate, AdaHalfNtArm::CurrentTc64)?,
            capture_ada_half_nt_s3_arm(&t, &fixture, &candidate, AdaHalfNtArm::Fast)?,
        ];
        validate_single_node_graph(&graphs[0], candidate.kind.name())?;
        validate_single_node_graph(&graphs[1], "half NT forced TC64")?;
        validate_nonempty_graph(&graphs[2], "half NT native Fast")?;

        let current_bits = observe_ada_half_nt_s3(
            &t,
            &fixture,
            &candidate,
            &graphs,
            AdaHalfNtArm::CurrentTc64,
            AdaHalfPath::Eager,
            1,
            None,
        )?
        .1;
        for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
            for repeat in 0..2 {
                for arm in [AdaHalfNtArm::CurrentTc64, AdaHalfNtArm::Candidate] {
                    observe_ada_half_nt_s3(
                        &t,
                        &fixture,
                        &candidate,
                        &graphs,
                        arm,
                        path,
                        1,
                        Some(&current_bits),
                    )?;
                    println!(
                        "{{\"schema\":\"{}\",\"dtype\":\"{dtype:?}\",\"candidate\":\"{}\",\"arm\":\"{}\",\"path\":\"{}\",\"repeat\":{repeat},\"words\":{}}}",
                        candidate.kind.bits_schema(),
                        candidate.kind.name(),
                        arm.name(candidate.kind),
                        path.name(),
                        current_bits.len(),
                    );
                }
            }
        }
        let fast_bits = observe_ada_half_nt_s3(
            &t,
            &fixture,
            &candidate,
            &graphs,
            AdaHalfNtArm::Fast,
            AdaHalfPath::Eager,
            1,
            None,
        )?
        .1;
        let fast_value = |word| match dtype {
            WeightDtype::F16 => f16::from_bits(word).to_f32(),
            WeightDtype::Bf16 => bf16::from_bits(word).to_f32(),
            WeightDtype::F32 => unreachable!(),
        };
        if !fast_bits.iter().all(|&word| fast_value(word).is_finite())
            || !fast_bits.iter().any(|&word| fast_value(word) != 0.0)
        {
            return Err(format!("{dtype:?} half NT Fast is non-finite or all-zero"));
        }
        for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
            for _ in 0..2 {
                observe_ada_half_nt_s3(
                    &t,
                    &fixture,
                    &candidate,
                    &graphs,
                    AdaHalfNtArm::Fast,
                    path,
                    1,
                    Some(&fast_bits),
                )?;
            }
        }
        for &comparator in comparators {
            if !matches!(comparator, AdaHalfNtArm::CurrentTc64 | AdaHalfNtArm::Fast) {
                return Err("half NT comparator must be current TC64 or native Fast".into());
            }
            let comparator_bits = match comparator {
                AdaHalfNtArm::CurrentTc64 => &current_bits,
                AdaHalfNtArm::Fast => &fast_bits,
                AdaHalfNtArm::Candidate => unreachable!(),
            };
            let mut strata = Vec::with_capacity(4);
            for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
                for order in [BracketOrder::Abba, BracketOrder::Baab] {
                    strata.push(screen_ada_half_nt_s3_pair(
                        &t,
                        &fixture,
                        &candidate,
                        &graphs,
                        &current_bits,
                        comparator,
                        comparator_bits,
                        path,
                        order,
                    )?);
                }
            }
            let retain = retain_decision(&strata);
            println!(
                "{{\"schema\":\"{}\",\"dtype\":\"{dtype:?}\",\"cell\":\"d768_out_proj\",\"shape\":[2048,1536,768],\"candidate\":\"{}\",\"comparator\":\"{}\",\"strata\":{:?},\"strata_order\":[\"eager/ABBA\",\"eager/BAAB\",\"graph/ABBA\",\"graph/BAAB\"],\"retain\":{retain},\"decision\":\"{}\",\"promotion\":false}}",
                candidate.kind.decision_schema(),
                candidate.kind.name(),
                comparator.name(candidate.kind),
                strata,
                if retain { "advance" } else { "stop_no_retry" },
            );
        }
    }
    drop(t);
    quiet
        .verify_post_cohort("half-nt-candidate/post")
        .map(|_| ())
}

#[test]
#[ignore = "requires exclusive Ada CC8.9 CUDA13.2; half NT BK32/S3 discovery"]
fn ada_half_nt_d768_out_bk32_s3_vs_current_and_fast_discovery_once7() -> Result<(), String> {
    run_ada_half_nt_d768_out_batch(AdaHalfNtCandidateKind::Bk32S3, 8, &[AdaHalfNtArm::Fast])
}

#[test]
#[ignore = "requires exclusive Ada CC8.9 CUDA13.2; half NT compact BK64/S2 discovery"]
fn ada_half_nt_d768_out_compact_bk64_s2_vs_current_and_fast_discovery_once7() -> Result<(), String>
{
    run_ada_half_nt_d768_out_batch(
        AdaHalfNtCandidateKind::CompactBk64S2,
        128,
        &[AdaHalfNtArm::Fast],
    )
}

#[test]
#[ignore = "requires exclusive Ada CC8.9 CUDA13.2; loaded half NT TC128 d768-out"]
fn ada_half_nt_d768_out_loaded_tc128_vs_current_and_fast_discovery_once7() -> Result<(), String> {
    run_ada_half_nt_d768_out_batch(
        AdaHalfNtCandidateKind::LoadedTc128,
        128,
        &[AdaHalfNtArm::CurrentTc64, AdaHalfNtArm::Fast],
    )
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
    let sm80_source = include_str!("../kernels/gemm_bi_triad/sm80.cu");
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
    let typed_helper = cuda_braced_scope_after(mma_source, "void gemm_bi_cp_async_16_zfill(");
    let typed_l2_helper = cuda_braced_scope_after(mma_source, "void gemm_bi_cp_async_16_zfill_l2(");
    let tf32_stage = cuda_braced_scope_after(tf32_sm80, "void gemm_bi_tf32_stage_async(");

    assert_eq!(
        mma_source
            .matches("void gemm_bi_cp_async_16_zfill(")
            .count(),
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
        mma_source
            .matches("void gemm_bi_cp_async_16_zfill_l2(")
            .count(),
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
        "typed SM80 kernels must not bypass gemm_bi_cp_async_16_zfill"
    );
    assert_eq!(
        typed_sm80.matches("cp.async.cg.shared.global").count(),
        0,
        "typed SM80 kernels must not bypass the named async-copy helpers"
    );
    let typed_copies = typed_sm80.matches("gemm_bi_cp_async_16_zfill(").count()
        + typed_sm80.matches("gemm_bi_cp_async_16_zfill_l2(").count();
    assert!(typed_copies > 0, "typed SM80 kernels must use async copies");
    assert_eq!(
        typed_copies,
        typed_sm80.matches("gemm_bi_cp_async_source(").count(),
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
        tf32_sm80
            .matches("void gemm_bi_tf32_cp_async_4x4_zfill(")
            .count(),
        1,
        "portable TF32 staging must own exactly one narrow-stride helper"
    );
    assert_eq!(
        tf32_sm80.matches("cp.async.ca.shared.global").count(),
        1,
        "portable TF32 staging must use only the named 16-byte helper"
    );
    let tf32_copy_helper =
        cuda_braced_scope_after(tf32_sm80, "void gemm_bi_tf32_cp_async_16_zfill(");
    assert_eq!(
        tf32_sm80
            .matches("void gemm_bi_tf32_cp_async_16_zfill(")
            .count(),
        1,
        "portable TF32 staging must centralize its tile-aware cache policy"
    );
    assert!(tf32_copy_helper.contains("if constexpr (BM == 16)"));
    assert!(tf32_copy_helper.contains("gemm_bi_cp_async_16_zfill("));
    assert!(tf32_copy_helper.contains("gemm_bi_cp_async_16_zfill_l2("));
    // The staging loop routes every copy through the stride-aware wrapper,
    // which picks the 16-byte helper or the narrow 4x4 helper per call.
    let tf32_wide_copies = tf32_stage.matches("gemm_bi_tf32_cp_async_zfill<").count();
    assert_eq!(tf32_wide_copies, 4);
    let tf32_wrapper = cuda_braced_scope_after(tf32_sm80, "void gemm_bi_tf32_cp_async_zfill(");
    assert_eq!(
        tf32_wrapper
            .matches("gemm_bi_tf32_cp_async_16_zfill<BM>(")
            .count(),
        2
    );
    assert_eq!(
        tf32_wrapper
            .matches("gemm_bi_tf32_cp_async_4x4_zfill(")
            .count(),
        2
    );
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
        tf32_stage.matches("gemm_bi_cp_async_source(").count(),
        "every TF32 async copy must select an in-allocation source"
    );
    for line in tf32_stage
        .lines()
        .filter(|line| line.contains("gemm_bi_cp_async_source("))
    {
        assert!(
            line.contains("safe_offset, _bytes"),
            "TF32 source selection bypasses its safe offset: {line}"
        );
    }
    for line in tf32_stage
        .lines()
        .filter(|line| line.contains("gemm_bi_tf32_cp_async_zfill<"))
    {
        assert!(
            line.contains("dst, src, _bytes"),
            "TF32 async-copy call bypasses its selected source: {line}"
        );
    }
    let tf32_narrow_helper =
        cuda_braced_scope_after(tf32_sm80, "void gemm_bi_tf32_cp_async_4x4_zfill(");
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
    let source = include_str!("../kernels/gemm_bi_triad/sm80.cu");
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
        "gemm_bi_cp_async_16_zfill(_dst, _src, _bytes)",
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
        "gemm_bi_cp_async_16_zfill_l2",
        "cp.async.cg",
        "float2",
        "gemm_bi_accumulate_float2_or_scalar",
        "atomic",
        "split",
        "REDUX",
        "gemm_bi_nn_tc128x64",
        "gemm_bi_nt_tc128x64",
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
        include_str!("../kernels/gemm_bi_triad/sm80.cu"),
    ]
    .concat();
    let tc128_source = source
        .split_once("#define GEMM_BI_TC64_BM")
        .expect("Tile64 macro boundary")
        .0;

    assert_eq!(
        tc128_source
            .matches("gemm_bi_output_start_if_valid(")
            .count(),
        4,
        "the helper definition and all three TC128 epilogues must use the guarded output start"
    );
    assert!(
        !tc128_source.contains("= &C["),
        "TC128 epilogues must not form output pointers before validating c0"
    );
    let helper = tc128_source
        .split_once("T* gemm_bi_output_start_if_valid(")
        .expect("guarded output helper")
        .1
        .split_once("__device__ __forceinline__ int gemm_bi_cp_async_valid_elems")
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
