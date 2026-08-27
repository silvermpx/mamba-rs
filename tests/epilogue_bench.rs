//! Epilogue-class measurement instruments at the production shape:
//!
//! - `m1_bandwidth_ceiling`: what a bare bt*d_inner bf16 copy costs in
//!   scalar / uint2 / uint4 shapes. This one reading prices the whole
//!   vectorization class - if the scalar copy already runs near the
//!   vector copies, elementwise vectorization is a minor item.
//! - `bench_epilogue_kernels_isolated`: the elementwise/cast/reduce
//!   kernels the training step launches, each timed alone, so any
//!   deletion or fusion proposal is priced against a measured cost
//!   instead of a residue.
//! - `m2_reduce_dim_sweep`: reduce_sum_axis0 across output widths -
//!   the column-strided reads LOOK 8-16x amplified but L2 may absorb
//!   the stride entirely at high residency; measure before touching.
//!
//!   cargo test --release --features cuda --test epilogue_bench -- --ignored --nocapture
#![cfg(feature = "cuda")]

mod common;

use common::bench::{bench_stamp, timed};
use cudarc::driver::PushKernelArg;
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::launch::grid_1d;

const B: usize = 8;
const T: usize = 1300;
const DM: usize = 384;
const DI: usize = 768;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn fbuf(ctx: &GpuCtx, n: usize, seed: u32) -> GpuBuffer {
    let mut b = GpuBuffer::zeros(&ctx.stream, n).unwrap();
    b.upload(&ctx.stream, &det(n, seed, 1.0)).unwrap();
    b
}

fn tbuf(ctx: &GpuCtx, n: usize, seed: u32, dt: WeightDtype) -> DtypedBuf {
    let b = DtypedBuf::zeros(&ctx.stream, n, dt).unwrap();
    b.upload_f32(&ctx.stream, &det(n, seed, 1.0)).unwrap();
    b
}

/// M-1: bandwidth ceiling for the elementwise class. Three copy kernels
/// compiled here (bench-only; the shipped sources stay clean), each
/// moving bt*d_inner bf16 = ~16 MB read + ~16 MB write.
#[test]
#[ignore = "measurement instrument"]
fn m1_bandwidth_ceiling() {
    let device = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&device).unwrap();
    let n = B * T * DI;
    assert_eq!(n % 8, 0, "uint4 copy needs n % 8 == 0 for bf16");

    let src_code = r#"
#include <cuda_bf16.h>
extern "C" __global__ void copy_scalar(__nv_bfloat16* dst, const __nv_bfloat16* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i];
}
extern "C" __global__ void copy_uint2(unsigned int* dst2, const unsigned int* src2, int n2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    dst2[i] = src2[i];
}
extern "C" __global__ void copy_uint4(uint4* dst4, const uint4* src4, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    dst4[i] = src4[i];
}
"#;
    let arch = GpuDevice::nvrtc_arch(device.compute_capability);
    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some(arch),
        include_paths: vec![format!("{cuda_home}/include")],
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(src_code, opts).unwrap();
    let module = device.context().load_module(ptx).unwrap();

    let src = tbuf(&ctx, n, 1, WeightDtype::Bf16);
    let dst = DtypedBuf::zeros(&ctx.stream, n, WeightDtype::Bf16).unwrap();
    let bytes = 2.0 * (n as f64) * 2.0; // read + write, 2 B/elem
    println!(
        "{}",
        bench_stamp(&device, &ctx, &format!("copy n={n} bf16"), "m1_ceiling", 0)
    );
    for (name, elems_per_thread) in [
        ("copy_scalar", 1usize),
        ("copy_uint2", 2),
        ("copy_uint4", 8),
    ] {
        let f = module.load_function(name).unwrap();
        let count = (n / elems_per_thread) as i32;
        let d = dst.cached_ptr();
        let s = src.cached_ptr();
        let ms = timed(&ctx, 200, || {
            let mut bld = ctx.stream.launch_builder(&f);
            bld.arg(&d);
            bld.arg(&s);
            bld.arg(&count);
            unsafe { bld.launch(grid_1d(count as usize)) }.unwrap();
        });
        let gbs = bytes / (ms / 1e3) / 1e9;
        println!("  {name:12} {ms:.4} ms  {gbs:.0} GB/s");
    }
}

/// M-0: the epilogue kernels the production step actually launches, each
/// timed alone at the production shape (bf16 typed variants where the
/// mixed lane uses them).
#[test]
#[ignore = "measurement instrument"]
fn bench_epilogue_kernels_isolated() {
    let device = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&device).unwrap();
    let k = &ctx.kernels;
    let dt = WeightDtype::Bf16;
    let bt = B * T;

    println!(
        "{}",
        bench_stamp(
            &device,
            &ctx,
            &format!("epilogue B{B} T{T} d{DM} di{DI}"),
            "m0_ledger",
            0
        )
    );

    // elementwise_mul_typed: gated = y * gate_post_silu  [bt*di]
    {
        let a = tbuf(&ctx, bt * DI, 2, dt);
        let b = tbuf(&ctx, bt * DI, 3, dt);
        let y = DtypedBuf::zeros(&ctx.stream, bt * DI, dt).unwrap();
        let n = (bt * DI) as i32;
        let (yp, ap, bp) = (y.cached_ptr(), a.cached_ptr(), b.cached_ptr());
        let f = k.elementwise_mul_typed.get(dt);
        let ms = timed(&ctx, 200, || {
            let mut bld = ctx.stream.launch_builder(f);
            bld.arg(&yp);
            bld.arg(&ap);
            bld.arg(&bp);
            bld.arg(&n);
            unsafe { bld.launch(grid_1d(bt * DI)) }.unwrap();
        });
        println!("  elementwise_mul_typed [bt*di]   {ms:.4} ms");
    }

    // softplus_copy_typed: delta = softplus(dt_raw)  [bt*di]
    {
        let a = tbuf(&ctx, bt * DI, 4, dt);
        let y = DtypedBuf::zeros(&ctx.stream, bt * DI, dt).unwrap();
        let n = (bt * DI) as i32;
        let (yp, ap) = (y.cached_ptr(), a.cached_ptr());
        let f = k.softplus_copy_typed.get(dt);
        let ms = timed(&ctx, 200, || {
            let mut bld = ctx.stream.launch_builder(f);
            bld.arg(&yp);
            bld.arg(&ap);
            bld.arg(&n);
            unsafe { bld.launch(grid_1d(bt * DI)) }.unwrap();
        });
        println!("  softplus_copy_typed  [bt*di]    {ms:.4} ms");
    }

    // concat_halves_typed: d_proj from (d_x_branch, d_gate)  [bt, 2di]
    {
        let a = tbuf(&ctx, bt * DI, 5, dt);
        let b = tbuf(&ctx, bt * DI, 6, dt);
        let y = DtypedBuf::zeros(&ctx.stream, bt * 2 * DI, dt).unwrap();
        let (bt_i, di_i) = (bt as i32, DI as i32);
        let (yp, ap, bp) = (y.cached_ptr(), a.cached_ptr(), b.cached_ptr());
        let f = k.concat_halves_typed.get(dt);
        let ms = timed(&ctx, 200, || {
            let mut bld = ctx.stream.launch_builder(f);
            bld.arg(&yp);
            bld.arg(&ap);
            bld.arg(&bp);
            bld.arg(&bt_i);
            bld.arg(&di_i);
            unsafe { bld.launch(grid_1d(bt * DI)) }.unwrap();
        });
        println!("  concat_halves_typed  [bt*2di]   {ms:.4} ms");
    }

    // vec_cast_zplus_typed: f32 -> typed  [bt*dm]
    {
        let a = fbuf(&ctx, bt * DM, 7);
        let y = DtypedBuf::zeros(&ctx.stream, bt * DM, dt).unwrap();
        let n = (bt * DM) as i32;
        let (yp, ap) = (y.cached_ptr(), a.cached_ptr());
        let f = k.vec_cast_zplus_typed.get(dt);
        let ms = timed(&ctx, 200, || {
            let mut bld = ctx.stream.launch_builder(f);
            bld.arg(&yp);
            bld.arg(&ap);
            bld.arg(&n);
            unsafe { bld.launch(grid_1d(bt * DM)) }.unwrap();
        });
        println!("  vec_cast_zplus_typed [bt*dm]    {ms:.4} ms");
    }

    // residual_add_typed: out = a + b  [bt*dm]
    {
        let a = tbuf(&ctx, bt * DM, 8, dt);
        let b = tbuf(&ctx, bt * DM, 9, dt);
        let y = DtypedBuf::zeros(&ctx.stream, bt * DM, dt).unwrap();
        let n = (bt * DM) as i32;
        let (yp, ap, bp) = (y.cached_ptr(), a.cached_ptr(), b.cached_ptr());
        let f = k.residual_add_typed.get(dt);
        let ms = timed(&ctx, 200, || {
            let mut bld = ctx.stream.launch_builder(f);
            bld.arg(&yp);
            bld.arg(&ap);
            bld.arg(&bp);
            bld.arg(&n);
            unsafe { bld.launch(grid_1d(bt * DM)) }.unwrap();
        });
        println!("  residual_add_typed   [bt*dm]    {ms:.4} ms");
    }
}

/// M-2: reduce_sum_axis0 across output widths. Rows = bt (the per-sample
/// partials layout the backward reduces).
#[test]
#[ignore = "measurement instrument"]
fn m2_reduce_dim_sweep() {
    let device = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&device).unwrap();
    let bt = B * T;
    println!(
        "{}",
        bench_stamp(&device, &ctx, &format!("reduce rows={bt}"), "m2_sweep", bt)
    );
    for dim in [64usize, 768, 4096] {
        let partials = fbuf(&ctx, bt * dim, 30 + dim as u32);
        let mut out = GpuBuffer::zeros(&ctx.stream, dim).unwrap();
        let (bt_i, dim_i, acc) = (bt as i32, dim as i32, 0i32);
        let block = (bt.next_power_of_two()).clamp(32, 256) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (dim as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };
        let ms = timed(&ctx, 100, || {
            let mut bld = ctx.stream.launch_builder(&ctx.kernels.reduce_sum_axis0);
            bld.arg(out.inner_mut());
            bld.arg(partials.inner());
            bld.arg(&bt_i);
            bld.arg(&dim_i);
            bld.arg(&acc);
            unsafe { bld.launch(cfg) }.unwrap();
        });
        let bytes = (bt * dim * 4) as f64;
        let gbs = bytes / (ms / 1e3) / 1e9;
        println!("  dim={dim:5}  {ms:.4} ms  {gbs:.0} GB/s effective");
    }
}
