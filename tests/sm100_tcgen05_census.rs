//! Datacenter-Blackwell qualification census (needs a CC 10.x device -
//! B200/B300).
//!
//! Decides the tcgen05 rung's contract class: launch the forced tcgen05
//! kernel and the mma.sync Tile128 twin on identical inputs and compare
//! bits. Equal bits admit tcgen05 into the existing arithmetic family as
//! a scheduling rung (no new goldens); differing bits make it a separate
//! family that must be route-pinned and cover all M itself. Either way
//! the sm_100/sm_103 dispatch cells stay empty until this has run green
//! on real hardware.
//!
//! The bias arm exercises the TMEM seed path (tcgen05.st before the
//! first MMA with enable-input-d), which no other rung shares.
#![cfg(feature = "cuda")]

use cudarc::driver::PushKernelArg;
use mamba_rs::mamba_ssm::gpu::blas::TypedPtr;
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    TcFwdOperands, TcTile, gemm_bi_forward_tc_with_tile,
};

fn exposing(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let base = ((s & 0xFFFFFF) as f32 / 16777216.0) * 0.4 - 0.2;
            match i % 4 {
                0 => 4096.0,
                1 => -4096.0,
                2 => base * 512.0,
                _ => base,
            }
        })
        .collect()
}

#[test]
#[ignore = "needs a CC 10.x device"]
fn tcgen05_vs_mma_sync_family_verdict() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let arch = GpuDevice::nvrtc_arch(dev.compute_capability);
    assert!(
        arch == "sm_100a" || arch == "sm_103a",
        "this census runs on datacenter Blackwell only (got {arch})"
    );
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let tcgen = ctx
        .kernels
        .gemm_bi_nn_sm100_typed
        .as_ref()
        .expect("datacenter Blackwell loads the tcgen05 rung");

    let dt = WeightDtype::Bf16;
    for (m, k, n, with_bias) in [
        (128usize, 384usize, 384usize, false),
        (128, 384, 384, true),
        (4621, 768, 2304, false),
        (4621, 768, 2304, true),
    ] {
        let a = DtypedBuf::zeros(&ctx.stream, m * k, dt).unwrap();
        a.upload_f32(&ctx.stream, &exposing(m * k, 7)).unwrap();
        let w = DtypedBuf::zeros(&ctx.stream, k * n, dt).unwrap();
        w.upload_f32(&ctx.stream, &exposing(k * n, 9)).unwrap();
        let bias_buf = GpuBuffer::from_cpu(&ctx.stream, &exposing(n, 11)).unwrap();
        let bias: u64 = if with_bias { bias_buf.cached_ptr() } else { 0 };
        let c_tc = DtypedBuf::zeros(&ctx.stream, m * n, dt).unwrap();
        let c_mma = DtypedBuf::zeros(&ctx.stream, m * n, dt).unwrap();

        // Forced tcgen05 launch: 128x128 output tiles, 2-stage staging.
        let grid = (m.div_ceil(128) * n.div_ceil(128)) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 65_536,
        };
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let ct = c_tc.cached_ptr();
        let ap = a.cached_ptr();
        let wp = w.cached_ptr();
        let mut b = ctx.stream.launch_builder(tcgen.get(dt));
        b.arg(&ct);
        b.arg(&ap);
        b.arg(&wp);
        b.arg(&bias);
        b.arg(&alpha);
        b.arg(&beta);
        b.arg(&mi);
        b.arg(&ni);
        b.arg(&ki);
        b.arg(&ki);
        b.arg(&ni);
        b.arg(&ni);
        unsafe { b.launch(cfg) }.expect("tcgen05 launch");

        gemm_bi_forward_tc_with_tile(
            &ctx.stream,
            &ctx.kernels,
            &TcFwdOperands {
                y: TypedPtr {
                    ptr: c_mma.cached_ptr(),
                    dtype: dt,
                },
                x: TypedPtr {
                    ptr: a.cached_ptr(),
                    dtype: dt,
                },
                w: TypedPtr {
                    ptr: w.cached_ptr(),
                    dtype: dt,
                },
                bias_ptr: bias,
            },
            (m, k, n),
            TcTile::Tile128,
        )
        .expect("mma.sync twin");
        ctx.stream.synchronize().unwrap();

        let mut ht = vec![0.0f32; m * n];
        c_tc.download_f32(&ctx.stream, &mut ht).unwrap();
        let mut hm = vec![0.0f32; m * n];
        c_mma.download_f32(&ctx.stream, &mut hm).unwrap();
        let diff = ht
            .iter()
            .zip(&hm)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        println!(
            "M{m} K{k} N{n} bias={with_bias}: tcgen05 vs mma.sync differing bits {diff}/{} -> {}",
            m * n,
            if diff == 0 {
                "SAME family (scheduling rung, no new goldens)"
            } else {
                "SEPARATE family (route-pinned, must cover all M)"
            }
        );
    }
}

/// Byte repeatability of the forced rung on its own: one hundred eager
/// repeats must hash identically before any dispatch cell may open.
#[test]
#[ignore = "needs a CC 10.x device"]
fn tcgen05_eager_repeatability() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let arch = GpuDevice::nvrtc_arch(dev.compute_capability);
    assert!(arch == "sm_100a" || arch == "sm_103a");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let tcgen = ctx.kernels.gemm_bi_nn_sm100_typed.as_ref().unwrap();

    let dt = WeightDtype::Bf16;
    let (m, k, n) = (515usize, 768usize, 640usize);
    let a = DtypedBuf::zeros(&ctx.stream, m * k, dt).unwrap();
    a.upload_f32(&ctx.stream, &exposing(m * k, 21)).unwrap();
    let w = DtypedBuf::zeros(&ctx.stream, k * n, dt).unwrap();
    w.upload_f32(&ctx.stream, &exposing(k * n, 23)).unwrap();
    let c = DtypedBuf::zeros(&ctx.stream, m * n, dt).unwrap();

    let grid = (m.div_ceil(128) * n.div_ceil(128)) as u32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 65_536,
    };
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let (mi, ni, ki) = (m as i32, n as i32, k as i32);
    let bias: u64 = 0;
    let mut reference: Option<Vec<u32>> = None;
    for round in 0..100 {
        let cp = c.cached_ptr();
        let ap = a.cached_ptr();
        let wp = w.cached_ptr();
        let mut b = ctx.stream.launch_builder(tcgen.get(dt));
        b.arg(&cp);
        b.arg(&ap);
        b.arg(&wp);
        b.arg(&bias);
        b.arg(&alpha);
        b.arg(&beta);
        b.arg(&mi);
        b.arg(&ni);
        b.arg(&ki);
        b.arg(&ki);
        b.arg(&ni);
        b.arg(&ni);
        unsafe { b.launch(cfg) }.expect("tcgen05 launch");
        ctx.stream.synchronize().unwrap();
        let mut h = vec![0.0f32; m * n];
        c.download_f32(&ctx.stream, &mut h).unwrap();
        let bits: Vec<u32> = h.iter().map(|v| v.to_bits()).collect();
        match &reference {
            None => reference = Some(bits),
            Some(r) => assert_eq!(r, &bits, "round {round} diverged"),
        }
    }
}
