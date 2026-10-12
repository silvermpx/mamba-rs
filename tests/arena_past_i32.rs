//! Windows past `i32::MAX` elements. A gradient arena as a multi-billion
//! parameter model trains with: the clip norm and the f16 overflow check see
//! every element. Before the counts were 64-bit, the norm of such an arena
//! came out 0 (no clip, no spike skip) and the overflow check panicked. And
//! the Mamba-1 element kernels and the last-row gather over a window of that
//! many elements: each one is processed once, those past 2^31 included.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaStream, PushKernelArg};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::grad_clip::{alloc_partials, clip_grads_device};
use mamba_rs::gpu::launch::{grid_1d_cover, grid_1d_wide};
use mamba_rs::gpu::loss_scaler::{OverflowFlag, check_inf_nan_gpu};

/// Set `n` floats at `ptr` to `value` on the device.
fn fill(stream: &Arc<CudaStream>, ptr: u64, value: f32, n: usize) {
    let r = unsafe {
        cudarc::driver::sys::cuMemsetD32Async(ptr, value.to_bits(), n, stream.cu_stream())
    };
    assert_eq!(r, cudarc::driver::sys::CUresult::CUDA_SUCCESS, "memset");
}

/// Set `n` 16-bit values at `ptr` to `bits` on the device.
fn fill16(stream: &Arc<CudaStream>, ptr: u64, bits: u16, n: usize) {
    let r = unsafe { cudarc::driver::sys::cuMemsetD16Async(ptr, bits, n, stream.cu_stream()) };
    assert_eq!(r, cudarc::driver::sys::CUresult::CUDA_SUCCESS, "memset");
}

/// The 16-bit values at the given element positions of `ptr`.
fn read16(stream: &Arc<CudaStream>, ptr: u64, at: &[usize]) -> Vec<u16> {
    stream.synchronize().expect("sync");
    at.iter()
        .map(|&i| {
            let mut v = 0u16;
            let r = unsafe {
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    (&mut v as *mut u16).cast(),
                    ptr + (i * 2) as u64,
                    2,
                )
            };
            assert_eq!(r, cudarc::driver::sys::CUresult::CUDA_SUCCESS, "read {i}");
            v
        })
        .collect()
}

/// bf16 0.5 and 2.0.
const HALF: u16 = 0x3f00;
const TWO: u16 = 0x4000;

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

/// The Mamba-1 element kernels over a window of `i32::MAX + 4097` bf16
/// elements: the in-place softplus strides over the whole count, so every
/// element, the last included, is softplus(0.5) and none is applied twice;
/// the column gather over rows of two elements forms the offsets past 2^31
/// in 64 bits, so the elements there read their own source elements and not
/// the ones 2^31 back.
#[test]
#[ignore = "allocates 8 GiB of device memory"]
fn m1_element_kernels_run_every_row_past_i32_max() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let s = &ctx.stream;
    let n = i32::MAX as usize + 4097;
    let at = [
        0,
        1,
        i32::MAX as usize - 1,
        i32::MAX as usize,
        i32::MAX as usize + 1,
        n - 1,
    ];
    let n_i = n as i64;

    let x = DtypedBuf::zeros(s, n, WeightDtype::Bf16).expect("4 GiB of bf16");
    fill16(s, x.cached_ptr(), HALF, n);
    let xp = x.cached_ptr();
    let mut bld = s.launch_builder(ctx.kernels.softplus_fwd_typed.get(WeightDtype::Bf16));
    bld.arg(&xp);
    bld.arg(&n_i);
    unsafe { bld.launch(grid_1d_wide(n)) }.expect("softplus");
    let got = read16(s, xp, &at);
    // softplus(0.5) = 0.974 rounds to bf16 0x3f79; twice applied it would be
    // softplus(0.974) = 1.30.
    assert!(
        got.iter().all(|&v| v == 0x3f79),
        "softplus once everywhere: {got:04x?}"
    );
    drop(x);

    let rows = (n / 2) as i32;
    let two = 2i32;
    let zero = 0i32;
    let src = DtypedBuf::zeros(s, n, WeightDtype::Bf16).expect("4 GiB of bf16");
    let dst = DtypedBuf::zeros(s, n, WeightDtype::Bf16).expect("4 GiB of bf16");
    let (sp, dp) = (src.cached_ptr(), dst.cached_ptr());
    let low = i32::MAX as usize + 1;
    fill16(s, sp, HALF, low);
    fill16(s, sp + (low * 2) as u64, TWO, n - low);
    let mut bld = s.launch_builder(ctx.kernels.gather_cols_typed.get(WeightDtype::Bf16));
    bld.arg(&dp);
    bld.arg(&sp);
    bld.arg(&rows);
    bld.arg(&two);
    bld.arg(&two);
    bld.arg(&zero);
    unsafe { bld.launch(grid_1d_wide(n)) }.expect("gather");
    let got = read16(s, dp, &at);
    let want: Vec<u16> = at
        .iter()
        .map(|&i| if i < low { HALF } else { TWO })
        .collect();
    assert_eq!(got, want, "every element reads its own source element");
}

/// The last-row gather over more than `i32::MAX` outputs, the rows of a
/// prompt batch that wide: every output reads its own row, those past 2^31
/// included.
#[test]
#[ignore = "allocates 8 GiB of device memory"]
fn the_last_row_gather_reaches_every_output_past_i32_max() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let s = &ctx.stream;
    // Two sequences of one step, rows of half the outputs each.
    let d = (i32::MAX as usize).div_ceil(2) + 2049;
    let n = 2 * d;
    let at = [
        0,
        d - 1,
        d,
        i32::MAX as usize - 1,
        i32::MAX as usize,
        i32::MAX as usize + 1,
        n - 1,
    ];
    let src = DtypedBuf::zeros(s, n, WeightDtype::Bf16).expect("4 GiB of bf16");
    let dst = DtypedBuf::zeros(s, n, WeightDtype::Bf16).expect("4 GiB of bf16");
    let (sp, dp) = (src.cached_ptr(), dst.cached_ptr());
    let low = i32::MAX as usize + 1;
    fill16(s, sp, HALF, low);
    fill16(s, sp + (low * 2) as u64, TWO, n - low);
    let (b, t, d_i) = (2i32, 1i32, d as i32);
    let mut bld = s.launch_builder(
        ctx.kernels
            .gather_last_timestep_typed
            .get(WeightDtype::Bf16),
    );
    bld.arg(&dp);
    bld.arg(&sp);
    bld.arg(&b);
    bld.arg(&t);
    bld.arg(&d_i);
    unsafe { bld.launch(grid_1d_cover(n).expect("grid")) }.expect("gather");
    let got = read16(s, dp, &at);
    let want: Vec<u16> = at
        .iter()
        .map(|&i| if i < low { HALF } else { TWO })
        .collect();
    assert_eq!(got, want, "every output reads its own row");
}
