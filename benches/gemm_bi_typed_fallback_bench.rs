//! Wall clock of the typed upcast fallback against the native typed
//! buckets: the tax a shape pays when it leaves the covered set and routes
//! through upcast, f32 gemm_bi and RNE-downcast. The bit contract of both
//! paths is tests/gemm_bi_typed_parity.rs.

#![cfg(feature = "cuda")]

use half::{bf16, f16};
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gemm_bi_forward_typed};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad;

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
/// Measures the upcast-fallback tax on Big training shapes: typed entry
/// (upcast → f32 kernel → downcast) vs the bare f32 kernel on pre-upcast
/// operands. The delta is the ceiling on what native typed Big kernels
/// could recover with a slower-than-cp.async staging scheme.
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

// Run every instrument, or only the ones named on the command line:
// `cargo bench --bench <target> --features cuda -- <name> [<name> ...]`.
fn main() {
    let selected: Vec<String> = std::env::args().skip(1).collect();
    let run = |name: &str| selected.is_empty() || selected.iter().any(|s| s == name);
    if run("bench_upcast_fallback_tax") {
        bench_upcast_fallback_tax();
    }
}
