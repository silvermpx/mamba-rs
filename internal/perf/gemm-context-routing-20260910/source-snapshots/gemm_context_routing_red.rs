#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::blas::{gpu_gemm_bi_forward_ptr, gpu_gemm_bi_tied_lm_head_raw};
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GemmMode, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp;

const BATCH: usize = 3;
const D_MODEL: usize = 37;
const VOCAB_PADDED: usize = 96;

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_triad_public_forward_ptr_records_nn_route() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic)
        .expect("select deterministic GEMM mode");
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);

    let x = GpuBuffer::zeros(&ctx.stream, BATCH * D_MODEL).expect("allocate X[3,37]");
    let w = GpuBuffer::zeros(&ctx.stream, D_MODEL * VOCAB_PADDED).expect("allocate W[37,96]");
    let mut y = GpuBuffer::zeros(&ctx.stream, BATCH * VOCAB_PADDED).expect("allocate Y[3,96]");

    let trace = ctx
        .record_eager_gemm_trace(|| {
            gpu_gemm_bi_forward_ptr(
                &ctx,
                &mut y,
                x.cached_ptr(),
                w.cached_ptr(),
                None,
                (BATCH, D_MODEL, VOCAB_PADDED),
            )
        })
        .expect("deterministic Triad raw NN forward");

    assert!(!trace.routes().is_empty(), "raw NN recorded no route");
    assert!(
        trace.routes().iter().all(|route| {
            route.op == ResolvedGemmOp::Nn
                && route.shape == (BATCH, D_MODEL, VOCAB_PADDED)
                && route.strides == (D_MODEL, VOCAB_PADDED, VOCAB_PADDED)
        }),
        "raw NN recorded unexpected routes: {:?}",
        trace.routes()
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_triad_tied_f32_raw_records_nt_route_and_strides() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic)
        .expect("select deterministic GEMM mode");
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);

    let hidden = GpuBuffer::zeros(&ctx.stream, BATCH * D_MODEL).expect("allocate hidden[3,37]");
    let embedding =
        GpuBuffer::zeros(&ctx.stream, VOCAB_PADDED * D_MODEL).expect("allocate embedding[96,37]");
    let logits =
        GpuBuffer::zeros(&ctx.stream, BATCH * VOCAB_PADDED).expect("allocate logits[3,96]");

    let trace = ctx
        .record_eager_gemm_trace(|| {
            gpu_gemm_bi_tied_lm_head_raw(
                &ctx,
                logits.cached_ptr(),
                hidden.cached_ptr(),
                embedding.cached_ptr(),
                BATCH,
                D_MODEL,
                VOCAB_PADDED,
            )
        })
        .expect("deterministic Triad tied F32 NT forward");

    assert!(!trace.routes().is_empty(), "tied F32 NT recorded no route");
    assert!(
        trace.routes().iter().all(|route| {
            route.op == ResolvedGemmOp::Nt
                && route.shape == (BATCH, VOCAB_PADDED, D_MODEL)
                && route.strides == (D_MODEL, D_MODEL, VOCAB_PADDED)
        }),
        "tied F32 NT recorded unexpected routes: {:?}",
        trace.routes()
    );
}
