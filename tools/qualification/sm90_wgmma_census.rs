//! Hopper qualification check (needs an sm_90a device - H100/H200).
//!
//! Decides the wgmma rung's contract class: launch the forced wgmma
//! kernel and the mma.sync Tile128 twin on identical inputs and compare
//! bits. Equal bits admit wgmma into the existing arithmetic family as a
//! scheduling rung (no new goldens); differing bits make it a separate
//! family that must be route-pinned and cover all M itself. Either way
//! the sm_90a dispatch cells stay empty until this has run green on real
//! hardware.
#![cfg(feature = "cuda")]

use cudarc::driver::PushKernelArg;
use mamba_rs::mamba_ssm::gpu::blas::TypedPtr;
use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
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
#[ignore = "needs an sm_90a device"]
fn wgmma_vs_mma_sync_family_verdict() {
    let dev = GpuDevice::new(0).expect("cuda device");
    assert_eq!(
        GpuDevice::nvrtc_arch(dev.compute_capability),
        "sm_90a",
        "this check runs on Hopper only"
    );
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let wgmma = ctx
        .kernels
        .gemm_bi_nn_sm90_typed
        .as_ref()
        .expect("sm_90a loads the wgmma rung");

    let dt = WeightDtype::Bf16;
    for (m, k, n) in [(128usize, 384usize, 384usize), (4621, 768, 2304)] {
        let a = DtypedBuf::zeros(&ctx.stream, m * k, dt).unwrap();
        a.upload_f32(&ctx.stream, &exposing(m * k, 7)).unwrap();
        let w = DtypedBuf::zeros(&ctx.stream, k * n, dt).unwrap();
        w.upload_f32(&ctx.stream, &exposing(k * n, 9)).unwrap();
        let c_wg = DtypedBuf::zeros(&ctx.stream, m * n, dt).unwrap();
        let c_mma = DtypedBuf::zeros(&ctx.stream, m * n, dt).unwrap();

        // Forced wgmma launch: 64x128 output tiles, 2-stage staging.
        let grid = (m.div_ceil(64) * n.div_ceil(128)) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 49_152,
        };
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let bias: u64 = 0;
        let cw = c_wg.cached_ptr();
        let ap = a.cached_ptr();
        let wp = w.cached_ptr();
        let mut b = ctx.stream.launch_builder(wgmma.get(dt));
        b.arg(&cw);
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
        unsafe { b.launch(cfg) }.expect("wgmma launch");

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
                bias_ptr: 0,
            },
            (m, k, n),
            TcTile::Tile128,
        )
        .expect("mma.sync twin");
        ctx.stream.synchronize().unwrap();

        let mut hw = vec![0.0f32; m * n];
        c_wg.download_f32(&ctx.stream, &mut hw).unwrap();
        let mut hm = vec![0.0f32; m * n];
        c_mma.download_f32(&ctx.stream, &mut hm).unwrap();
        let diff = hw
            .iter()
            .zip(&hm)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        println!(
            "M{m} K{k} N{n}: wgmma vs mma.sync differing bits {diff}/{} -> {}",
            m * n,
            if diff == 0 {
                "SAME family (scheduling rung, no new goldens)"
            } else {
                "SEPARATE family (route-pinned, must cover all M)"
            }
        );
    }
}
