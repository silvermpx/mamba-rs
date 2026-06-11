//! Stage 5 tensor-core tier (`bi_tensor_cores`) — contract tests.
//!
//! The TC NN forward (`sgemm_bi_nn_tc_*`, mma.sync.m16n8k16 + f32
//! accumulate) is a SEPARATE numeric contract from the scalar triad: its
//! reduction tree differs from the ascending-K FMA chain, so outputs do not
//! bit-match the scalar kernels. What it MUST satisfy:
//!   1. correctness — close to the f32 reference on quantized inputs
//!      (cosine; a fragment-layout bug shows up as garbage, not noise);
//!   2. run-to-run determinism — bit-identical across launches;
//!   3. STRICT batch invariance — row 0 bit-identical across ALL M (each
//!      element's K-reduction lives in one warp, independent of grid).

#![cfg(feature = "cuda")]

use half::{bf16, f16};
use mamba_rs::mamba_ssm::gpu::blas::TypedPtr;
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

fn quantize(v: &[f32], dt: WeightDtype) -> Vec<f32> {
    match dt {
        WeightDtype::Bf16 => v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect(),
        WeightDtype::F16 => v.iter().map(|&x| f16::from_f32(x).to_f32()).collect(),
        WeightDtype::F32 => v.to_vec(),
    }
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

    fn typed_buf(&self, data: &[f32], dt: WeightDtype) -> DtypedBuf {
        let b = DtypedBuf::zeros(&self.ctx.stream, data.len(), dt).unwrap();
        b.upload_f32(&self.ctx.stream, data).unwrap();
        b
    }

    fn f32_buf(&self, data: &[f32]) -> GpuBuffer {
        let mut b = GpuBuffer::zeros(&self.ctx.stream, data.len()).unwrap();
        b.upload(&self.ctx.stream, data).unwrap();
        b
    }
}

/// Launch the TC forward on quantized inputs; returns Y upcast to f32.
fn run_tc(
    t: &Ctx,
    dt: WeightDtype,
    dims: (usize, usize, usize),
    qx: &[f32],
    qw: &[f32],
    bias: Option<&GpuBuffer>,
) -> Vec<f32> {
    let (m, _k, n) = dims;
    let xt = t.typed_buf(qx, dt);
    let wt = t.typed_buf(qw, dt);
    let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
    sgemm_bi::sgemm_bi_forward_tc(
        &t.ctx.stream,
        &t.ctx.kernels,
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
        bias.map_or(0, |b| b.cached_ptr()),
        dims,
    )
    .unwrap();
    t.ctx.stream.synchronize().unwrap();
    let mut out = vec![0.0f32; m * n];
    yt.download_f32(&t.ctx.stream, &mut out).unwrap();
    out
}

fn cos_sim(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

#[test]
fn tc_forward_matches_f32_reference_loosely() {
    let t = Ctx::new();
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        for (m, k, n) in [
            (256usize, 384usize, 512usize),
            (256, 100, 512),  // K-tail (K % 32 != 0)
            (300, 768, 3072), // M/N-tails
            (2048, 768, 3072),
        ] {
            let qx = quantize(&det(m * k, 11, 1.0), dt);
            let qw = quantize(&det(k * n, 22, 0.5), dt);
            let bias = det(n, 33, 0.25);
            let b32 = t.f32_buf(&bias);

            // f32 reference on the SAME quantized values.
            let x32 = t.f32_buf(&qx);
            let w32 = t.f32_buf(&qw);
            let mut y32 = GpuBuffer::zeros(&t.ctx.stream, m * n).unwrap();
            sgemm_bi::sgemm_bi_forward(
                &t.ctx.stream,
                &t.ctx.kernels,
                &mut y32,
                &x32,
                w32.cached_ptr(),
                b32.cached_ptr(),
                (m, k, n),
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            let reference = y32.to_cpu(&t.ctx.stream).unwrap();

            let got = run_tc(&t, dt, (m, k, n), &qx, &qw, Some(&b32));
            let cos = cos_sim(&got, &reference);
            eprintln!("TC {dt:?} M{m} K{k} N{n}: cos vs f32 = {cos:.9}");
            assert!(
                cos > 0.9999,
                "TC {dt:?} M{m} K{k} N{n}: cos {cos} — fragment layout or staging bug"
            );
        }
    }
}

#[test]
fn tc_forward_is_deterministic_and_all_m_batch_invariant() {
    let t = Ctx::new();
    let (k, n) = (768usize, 3072usize);
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        let row = quantize(&det(k, 42, 1.0), dt);
        let qw = quantize(&det(k * n, 43, 0.5), dt);

        let run = |m: usize, seed: u32| -> Vec<f32> {
            let mut qx = quantize(&det(m * k, seed, 1.0), dt);
            qx[..k].copy_from_slice(&row);
            run_tc(&t, dt, (m, k, n), &qx, &qw, None)[..n].to_vec()
        };

        // Run-to-run determinism at fixed M.
        let a = run(256, 100);
        let b = run(256, 100);
        for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{dt:?} TC nondeterminism at col {i}"
            );
        }

        // STRICT all-M batch invariance of row 0 (other rows differ).
        let y128 = run(128, 200);
        let y512 = run(512, 300);
        let y2048 = run(2048, 400);
        for (i, (&x, &y)) in y128.iter().zip(&y512).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{dt:?} TC batch-variance at col {i}: M=128 vs M=512"
            );
        }
        for (i, (&x, &y)) in y128.iter().zip(&y2048).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{dt:?} TC batch-variance at col {i}: M=128 vs M=2048"
            );
        }
    }
}

#[test]
#[ignore] // wall-clock benchmark — run explicitly on a quiet GPU
fn bench_tc_vs_scalar_paths() {
    use mamba_rs::mamba_ssm::gpu::blas::bi_sgemm_forward_typed;
    use std::time::Instant;
    let t = Ctx::new();
    let dt = WeightDtype::Bf16;
    for (m, k, n) in [
        (2048usize, 768usize, 3072usize),
        (4096, 1536, 3072),
        (2048, 768, 512),
    ] {
        let qx = quantize(&det(m * k, 11, 1.0), dt);
        let qw = quantize(&det(k * n, 22, 0.5), dt);
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

        let iters = 50;
        let time_path = |label: &str, f: &dyn Fn()| -> f64 {
            for _ in 0..3 {
                f();
            }
            t.ctx.stream.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                f();
            }
            t.ctx.stream.synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            eprintln!("  {label}: {us:.1} us");
            us
        };

        eprintln!("[M{m} K{k} N{n}]");
        let scalar = time_path("scalar bi (native/fallback)", &|| {
            bi_sgemm_forward_typed(&t.ctx, ytp, xtp, wtp, 0, (m, k, n)).unwrap();
        });
        let tc = time_path("tensor-core bi", &|| {
            sgemm_bi::sgemm_bi_forward_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                ytp,
                xtp,
                wtp,
                0,
                (m, k, n),
            )
            .unwrap();
        });
        eprintln!("  TC speedup vs scalar bi: {:.2}x", scalar / tc);
    }
}
