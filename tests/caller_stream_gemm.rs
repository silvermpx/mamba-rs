//! A program that owns its CUDA context, stream and buffers runs its GEMMs
//! through mamba-rs on them: the forward, the input gradient and the
//! accumulated weight gradient on raw device pointers give the bits a
//! context with its own stream and buffers gives, under the exact and the
//! TF32 policy.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use mamba_rs::gpu::GemmMode;
use mamba_rs::gpu::blas::{
    gpu_gemm_f32_backward_dw_ptrs, gpu_gemm_f32_backward_dx_ptrs, gpu_gemm_f32_forward_ptrs,
};
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::{F32TriadPolicy, GpuCtx};
use mamba_rs::gpu::device::GpuDevice;

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as u32 as f32 / u32::MAX as f32) * 0.5 - 0.25
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

struct Operands {
    x: Vec<f32>,
    w: Vec<f32>,
    bias: Vec<f32>,
    dy: Vec<f32>,
    dw0: Vec<f32>,
}

impl Operands {
    fn new((b, k, n): (usize, usize, usize), seed: u64) -> Self {
        Self {
            x: synth(b * k, seed),
            w: synth(k * n, seed + 1),
            bias: synth(n, seed + 2),
            dy: synth(b * n, seed + 3),
            dw0: synth(k * n, seed + 4),
        }
    }
}

/// The three products on buffers the caller allocated with cudarc.
fn caller_products(
    ctx: &GpuCtx,
    stream: &Arc<CudaStream>,
    dims: (usize, usize, usize),
    ops: &Operands,
) -> [Vec<f32>; 3] {
    let (b, k, n) = dims;
    let x: CudaSlice<f32> = stream.clone_htod(&ops.x).unwrap();
    let w: CudaSlice<f32> = stream.clone_htod(&ops.w).unwrap();
    let bias: CudaSlice<f32> = stream.clone_htod(&ops.bias).unwrap();
    let dy: CudaSlice<f32> = stream.clone_htod(&ops.dy).unwrap();
    let dw: CudaSlice<f32> = stream.clone_htod(&ops.dw0).unwrap();
    let y: CudaSlice<f32> = stream.alloc_zeros(b * n).unwrap();
    let dx: CudaSlice<f32> = stream.alloc_zeros(b * k).unwrap();
    {
        let (xp, _g1) = x.device_ptr(stream);
        let (wp, _g2) = w.device_ptr(stream);
        let (bp, _g3) = bias.device_ptr(stream);
        let (dyp, _g4) = dy.device_ptr(stream);
        let (dwp, _g5) = dw.device_ptr(stream);
        let (yp, _g6) = y.device_ptr(stream);
        let (dxp, _g7) = dx.device_ptr(stream);
        unsafe {
            gpu_gemm_f32_forward_ptrs(ctx, yp, xp, wp, Some(bp), dims).unwrap();
            gpu_gemm_f32_backward_dx_ptrs(ctx, dxp, dyp, wp, dims).unwrap();
            gpu_gemm_f32_backward_dw_ptrs(ctx, dwp, dyp, xp, dims).unwrap();
        }
    }
    stream.synchronize().unwrap();
    [
        stream.clone_dtoh(&y).unwrap(),
        stream.clone_dtoh(&dx).unwrap(),
        stream.clone_dtoh(&dw).unwrap(),
    ]
}

/// The same products through a context that owns its stream and buffers.
fn owned_products(ctx: &GpuCtx, dims: (usize, usize, usize), ops: &Operands) -> [Vec<f32>; 3] {
    let (b, k, n) = dims;
    let upload = |values: &[f32]| {
        let mut buffer = GpuBuffer::zeros(&ctx.stream, values.len()).unwrap();
        buffer.upload(&ctx.stream, values).unwrap();
        buffer
    };
    let x = upload(&ops.x);
    let w = upload(&ops.w);
    let bias = upload(&ops.bias);
    let dy = upload(&ops.dy);
    let dw = upload(&ops.dw0);
    let y = GpuBuffer::zeros(&ctx.stream, b * n).unwrap();
    let dx = GpuBuffer::zeros(&ctx.stream, b * k).unwrap();
    unsafe {
        gpu_gemm_f32_forward_ptrs(
            ctx,
            y.cached_ptr(),
            x.cached_ptr(),
            w.cached_ptr(),
            Some(bias.cached_ptr()),
            dims,
        )
        .unwrap();
        gpu_gemm_f32_backward_dx_ptrs(ctx, dx.cached_ptr(), dy.cached_ptr(), w.cached_ptr(), dims)
            .unwrap();
        gpu_gemm_f32_backward_dw_ptrs(ctx, dw.cached_ptr(), dy.cached_ptr(), x.cached_ptr(), dims)
            .unwrap();
    }
    ctx.stream.synchronize().unwrap();
    [
        y.to_cpu(&ctx.stream).unwrap(),
        dx.to_cpu(&ctx.stream).unwrap(),
        dw.to_cpu(&ctx.stream).unwrap(),
    ]
}

#[test]
fn caller_stream_gemms_give_the_owned_context_bits() {
    let Ok(raw) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let stream = raw.new_stream().unwrap();
    let device = GpuDevice::from_context(raw.clone()).unwrap();
    let caller = GpuCtx::new_on_stream(&device, stream.clone(), 64, GemmMode::Deterministic)
        .expect("a context on the caller's stream");
    let owned_device = GpuDevice::new(0).unwrap();
    let owned = GpuCtx::new(&owned_device).unwrap();
    for policy in [
        F32TriadPolicy::ExactScalarFma,
        F32TriadPolicy::AllowDeterministicTf32,
    ] {
        caller.route_controls().set_f32_policy(policy);
        owned.route_controls().set_f32_policy(policy);
        for (index, dims) in [
            (7, 64, 96),
            (128, 384, 384),
            (513, 96, 40),
            (1024, 512, 128),
        ]
        .into_iter()
        .enumerate()
        {
            let ops = Operands::new(dims, 0x5eed_0000 + index as u64 * 16);
            let got = caller_products(&caller, &stream, dims, &ops);
            let want = owned_products(&owned, dims, &ops);
            for (label, (g, w)) in ["forward", "input gradient", "weight gradient"]
                .iter()
                .zip(got.iter().zip(want.iter()))
            {
                assert_eq!(bits(g), bits(w), "{policy:?} {dims:?} {label}");
            }
        }
    }
}
