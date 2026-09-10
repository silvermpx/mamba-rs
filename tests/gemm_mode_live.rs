#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::context::{GemmMode, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;

#[test]
#[ignore = "needs a CUDA device"]
fn gpu_mode_change_rejects_capture_but_allows_same_mode_noop() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");

    let error = match unsafe {
        capture_into_graph(&ctx.stream, || {
            ctx.set_gemm_mode(GemmMode::Deterministic)?;
            ctx.set_gemm_mode(GemmMode::CublasFast)
        })
    } {
        Ok(_) => panic!("a different mode during capture unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.contains("cannot change GEMM mode"), "{error}");
    assert!(error.contains("capture"), "{error}");
    assert_eq!(ctx.gemm_mode(), GemmMode::Deterministic);
}

#[test]
#[ignore = "needs a CUDA device"]
fn gpu_mode_change_is_rejected_during_route_recording() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");

    let error = ctx
        .record_eager_gemm_trace(|| ctx.set_gemm_mode(GemmMode::CublasPedantic))
        .expect_err("a different mode during route recording must fail");
    assert!(error.contains("GEMM route recording is active"), "{error}");
    assert_eq!(ctx.gemm_mode(), GemmMode::Deterministic);
}
