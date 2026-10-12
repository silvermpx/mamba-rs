//! A gradient arena past `i32::MAX` elements, as a multi-billion parameter
//! model trains with: the clip norm and the f16 overflow check see every
//! element. Before the counts were 64-bit, the norm of such an arena came
//! out 0 (no clip, no spike skip) and the overflow check panicked.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::CudaStream;
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::grad_clip::{alloc_partials, clip_grads_device};
use mamba_rs::gpu::loss_scaler::{OverflowFlag, check_inf_nan_gpu};

/// Set `n` floats at `ptr` to `value` on the device.
fn fill(stream: &Arc<CudaStream>, ptr: u64, value: f32, n: usize) {
    let r = unsafe {
        cudarc::driver::sys::cuMemsetD32Async(ptr, value.to_bits(), n, stream.cu_stream())
    };
    assert_eq!(r, cudarc::driver::sys::CUresult::CUDA_SUCCESS, "memset");
}

#[test]
#[ignore = "allocates 8 GiB of device memory"]
fn an_arena_past_i32_max_is_measured_and_checked_whole() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let n = i32::MAX as usize + 4097;
    let mut grads = GpuBuffer::zeros(&ctx.stream, n).expect("an 8 GiB arena");
    fill(&ctx.stream, grads.cached_ptr(), 1.0, n);

    // Every element is 1.0, so the norm is sqrt(n): the f64 sum of ones is
    // exact, the cast to f32 the only rounding.
    let mut partials = alloc_partials(&ctx.stream).expect("partials");
    let mut scratch = GpuBuffer::zeros(&ctx.stream, 2).expect("scratch");
    let norm = clip_grads_device(&ctx, &mut grads, &mut partials, &mut scratch, f32::INFINITY)
        .expect("norm");
    assert_eq!(
        norm,
        (n as f64).sqrt() as f32,
        "the norm of the whole arena"
    );

    // One non-finite value past i32::MAX: the overflow check finds it.
    let mut flag = OverflowFlag::new(&ctx.stream).expect("flag");
    flag.zero(&ctx.stream).expect("zero flag");
    check_inf_nan_gpu(&ctx, &ctx.kernels, &mut flag, &grads).expect("clean check");
    assert_eq!(flag.read(&ctx.stream).expect("read flag"), 0);
    let last = grads.cached_ptr() + ((n - 1) * std::mem::size_of::<f32>()) as u64;
    fill(&ctx.stream, last, f32::INFINITY, 1);
    check_inf_nan_gpu(&ctx, &ctx.kernels, &mut flag, &grads).expect("check");
    assert_eq!(
        flag.read(&ctx.stream).expect("read flag"),
        1,
        "the last element"
    );
}
