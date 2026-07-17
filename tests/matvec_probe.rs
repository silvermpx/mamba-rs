//! Bisection probe: the batch-invariant matvec route of
//! gpu_gemm_typed_forward_raw at the trainable-input_proj shapes.

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;

fn probe(ctx: &GpuCtx, m: usize, k: usize, n: usize) -> Result<(), String> {
    let dt = WeightDtype::Bf16;
    let x = DtypedBuf::zeros(&ctx.stream, m * k, dt)?;
    let w = DtypedBuf::zeros(&ctx.stream, k * n, dt)?;
    let c = DtypedBuf::zeros(&ctx.stream, m * n, dt)?;
    let bias = GpuBuffer::zeros(&ctx.stream, n)?;
    gpu_gemm_typed_forward_raw(
        ctx,
        TypedPtr {
            ptr: c.cached_ptr(),
            dtype: dt,
        },
        TypedPtr {
            ptr: x.cached_ptr(),
            dtype: dt,
        },
        TypedPtr {
            ptr: w.cached_ptr(),
            dtype: dt,
        },
        Some(bias.cached_ptr()),
        (m, k, n),
    )?;
    ctx.stream
        .synchronize()
        .map_err(|e| format!("sync after ({m},{k},{n}): {e:?}"))
}

fn probe_one(m: usize, k: usize, n: usize) {
    let dev = GpuDevice::new(0).expect("dev");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    ctx.set_batch_invariant(true);
    probe(&ctx, m, k, n).unwrap_or_else(|e| panic!("shape ({m},{k},{n}): {e}"));
}

#[test]
fn bi_matvec_k20() {
    probe_one(4, 20, 32);
}
#[test]
fn bi_matvec_k200() {
    probe_one(4, 200, 32);
}
#[test]
fn bi_matvec_k192() {
    probe_one(4, 192, 32);
}
#[test]
fn bi_matvec_k64() {
    probe_one(4, 64, 32);
}
#[test]
fn bi_matvec_k768() {
    probe_one(4, 768, 32);
}
