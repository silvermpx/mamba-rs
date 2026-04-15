//! Step 13 — dynamic loss scaler integration tests on GPU.
//!
//! The CPU state-machine tests live inline in `loss_scaler.rs::tests`. This
//! file exercises the GPU-side helpers (`check_inf_nan_gpu`, `scale_grads_gpu`)
//! plus an end-to-end "scaled backward → check → unscale → step" simulation
//! against a synthetic gradient buffer.

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::kernels::MambaKernels;
use mamba_rs::mamba_ssm::gpu::loss_scaler::{
    DynamicLossScaler, OverflowFlag, check_inf_nan_gpu, scale_grads_gpu,
};

fn make_ctx() -> (GpuCtx, MambaKernels) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let kernels = MambaKernels::compile(ctx.stream.context(), "sm_89").unwrap();
    (ctx, kernels)
}

fn upload(ctx: &GpuCtx, data: &[f32]) -> GpuBuffer {
    let mut b = GpuBuffer::zeros(&ctx.stream, data.len()).unwrap();
    ctx.stream.synchronize().unwrap();
    b.upload(&ctx.stream, data).unwrap();
    ctx.stream.synchronize().unwrap();
    b
}

#[test]
fn check_inf_nan_clean_buffer() {
    let (ctx, k) = make_ctx();
    let grads = upload(&ctx, &[0.1, -0.2, 1e-6, 1e6, -1e6, 0.0]);
    let mut flag = OverflowFlag::new(&ctx.stream).unwrap();
    flag.zero(&ctx.stream).unwrap();
    check_inf_nan_gpu(&ctx, &k, &mut flag, &grads).unwrap();
    assert_eq!(flag.read(&ctx.stream).unwrap(), 0, "no overflow expected");
}

#[test]
fn check_inf_nan_with_inf() {
    let (ctx, k) = make_ctx();
    let grads = upload(&ctx, &[0.1, f32::INFINITY, 1.0]);
    let mut flag = OverflowFlag::new(&ctx.stream).unwrap();
    flag.zero(&ctx.stream).unwrap();
    check_inf_nan_gpu(&ctx, &k, &mut flag, &grads).unwrap();
    assert_ne!(flag.read(&ctx.stream).unwrap(), 0, "+inf must trigger");
}

#[test]
fn check_inf_nan_with_neg_inf() {
    let (ctx, k) = make_ctx();
    let grads = upload(&ctx, &[1.0, f32::NEG_INFINITY, 2.0]);
    let mut flag = OverflowFlag::new(&ctx.stream).unwrap();
    flag.zero(&ctx.stream).unwrap();
    check_inf_nan_gpu(&ctx, &k, &mut flag, &grads).unwrap();
    assert_ne!(flag.read(&ctx.stream).unwrap(), 0, "-inf must trigger");
}

#[test]
fn check_inf_nan_with_nan() {
    let (ctx, k) = make_ctx();
    let grads = upload(&ctx, &[1.0, f32::NAN, 2.0]);
    let mut flag = OverflowFlag::new(&ctx.stream).unwrap();
    flag.zero(&ctx.stream).unwrap();
    check_inf_nan_gpu(&ctx, &k, &mut flag, &grads).unwrap();
    assert_ne!(flag.read(&ctx.stream).unwrap(), 0, "NaN must trigger");
}

#[test]
fn check_inf_nan_accumulates_across_buffers() {
    let (ctx, k) = make_ctx();
    let clean = upload(&ctx, &[1.0, 2.0, 3.0]);
    let dirty = upload(&ctx, &[1.0, f32::INFINITY, 3.0]);
    let mut flag = OverflowFlag::new(&ctx.stream).unwrap();
    flag.zero(&ctx.stream).unwrap();
    check_inf_nan_gpu(&ctx, &k, &mut flag, &clean).unwrap();
    assert_eq!(flag.read(&ctx.stream).unwrap(), 0, "clean buffer alone");
    check_inf_nan_gpu(&ctx, &k, &mut flag, &dirty).unwrap();
    assert_ne!(
        flag.read(&ctx.stream).unwrap(),
        0,
        "dirty after clean → flag set (atomicOr accumulates)"
    );
}

#[test]
fn scale_grads_in_place() {
    let (ctx, k) = make_ctx();
    let mut grads = upload(&ctx, &[1.0, 2.0, -3.0, 0.5]);
    scale_grads_gpu(&ctx, &k, &mut grads, 1.0 / 4.0).unwrap();
    let mut out = vec![0f32; 4];
    grads.download(&ctx.stream, &mut out).unwrap();
    ctx.stream.synchronize().unwrap();
    assert_eq!(out, [0.25, 0.5, -0.75, 0.125]);
}

#[test]
fn scale_grads_zero_length_noop() {
    let (ctx, k) = make_ctx();
    let mut grads = GpuBuffer::zeros(&ctx.stream, 0).unwrap();
    // Should not crash on empty input.
    scale_grads_gpu(&ctx, &k, &mut grads, 2.0).unwrap();
    let mut flag = OverflowFlag::new(&ctx.stream).unwrap();
    flag.zero(&ctx.stream).unwrap();
    check_inf_nan_gpu(&ctx, &k, &mut flag, &grads).unwrap();
    assert_eq!(flag.read(&ctx.stream).unwrap(), 0);
}

/// End-to-end simulation: scale loss → simulated grads → check → unscale.
#[test]
fn full_amp_cycle_simulation() {
    let (ctx, k) = make_ctx();
    let mut scaler = DynamicLossScaler::new().with_init_scale(8.0);

    // Iter 1: clean grads at scale=8 → expect no overflow, unscale to true vals.
    let true_grads = vec![0.1, -0.2, 0.3, 1e-3];
    let scaled: Vec<f32> = true_grads.iter().map(|x| x * scaler.scale()).collect();
    let mut buf = upload(&ctx, &scaled);

    let mut flag = OverflowFlag::new(&ctx.stream).unwrap();
    flag.zero(&ctx.stream).unwrap();
    check_inf_nan_gpu(&ctx, &k, &mut flag, &buf).unwrap();
    let overflow = flag.read(&ctx.stream).unwrap() != 0;
    assert!(!overflow, "clean grads should not overflow");

    scale_grads_gpu(&ctx, &k, &mut buf, 1.0 / scaler.scale()).unwrap();
    let mut got = vec![0f32; 4];
    buf.download(&ctx.stream, &mut got).unwrap();
    ctx.stream.synchronize().unwrap();
    for (a, b) in got.iter().zip(&true_grads) {
        assert!((a - b).abs() < 1e-6, "unscale: {} vs {}", a, b);
    }
    scaler.update(false);
    assert_eq!(scaler.scale(), 8.0); // not yet at growth interval

    // Iter 2: simulate overflow (one grad is inf).
    let bad_grads = vec![1.0, f32::INFINITY, 3.0];
    let buf = upload(&ctx, &bad_grads);
    flag.zero(&ctx.stream).unwrap();
    check_inf_nan_gpu(&ctx, &k, &mut flag, &buf).unwrap();
    let overflow = flag.read(&ctx.stream).unwrap() != 0;
    assert!(overflow, "inf must overflow");
    // Caller skips optimizer step; only update the scaler.
    scaler.update(true);
    assert_eq!(scaler.scale(), 4.0, "scale halved after overflow");

    // bad_grads is left as-is in `buf` because we skipped unscale — verify.
    let mut got = vec![0f32; 3];
    buf.download(&ctx.stream, &mut got).unwrap();
    ctx.stream.synchronize().unwrap();
    assert_eq!(got, [1.0, f32::INFINITY, 3.0]);
}
