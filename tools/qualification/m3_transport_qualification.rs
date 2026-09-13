//! Qualify compiled Mamba-3 transport selectors and their launch contracts.

#![cfg(feature = "cuda")]

#[path = "support/m3_transport/raw.rs"]
mod raw;

use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::gpu::Mamba3Kernels;

#[test]
#[ignore = "requires an Ada device and a qualified native compiler"]
fn compiled_transport_routes_keep_function_and_geometry_together() {
    let device = GpuDevice::new(0).unwrap();
    assert_eq!(device.context().compute_capability().unwrap(), (8, 9));
    assert_eq!(device.nvrtc_target(), "sm_89");
    for cap in [16, 32, 64] {
        let kernels =
            Mamba3Kernels::compile_with_state_cap(device.context(), device.nvrtc_target(), cap)
                .unwrap();
        for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
            let legacy = kernels.m3_dqkv_typed.get(dtype);
            assert!(!std::ptr::eq(
                kernels.dqkv_for_shape(dtype, 16, 16, 64, 1300, true),
                legacy,
            ));
            for (state, head, chunk, time, pairs) in [
                (8, 16, 64, 1300, true),
                (16, 8, 64, 1300, true),
                (16, 16, 32, 1300, true),
                (16, 16, 64, 127, true),
                (16, 16, 64, 1300, false),
            ] {
                assert!(std::ptr::eq(
                    kernels.dqkv_for_shape(dtype, state, head, chunk, time, pairs),
                    legacy,
                ));
            }

            let legacy = kernels.m3_dqktheta_typed.get(dtype);
            let (selected, tiles) = kernels.dqktheta_for_shape(dtype, 16, 64, 4);
            let optimized = cap == 16 && dtype != WeightDtype::F32;
            assert_eq!(!std::ptr::eq(selected, legacy), optimized);
            assert_eq!(tiles, if optimized { 4 } else { 6 });
            for (state, chunk, angles) in [(8, 64, 4), (16, 32, 4), (16, 64, 0)] {
                let (selected, tiles) = kernels.dqktheta_for_shape(dtype, state, chunk, angles);
                assert!(std::ptr::eq(selected, legacy));
                assert_eq!(tiles, 6);
            }
        }

        let (selected, cfg) = kernels.axis0_reduction(4, 499200);
        if cap == 16 {
            assert!(!std::ptr::eq(selected, &kernels.reduce_sum_axis0));
            assert_eq!(cfg.grid_dim, (62400, 1, 1));
            assert_eq!(cfg.block_dim, (8, 32, 1));
            assert_eq!(cfg.shared_mem_bytes, 1024);
            let (selected, cfg) = kernels.axis0_reduction(48, 41600);
            assert!(!std::ptr::eq(selected, &kernels.reduce_sum_axis0));
            assert_eq!(cfg.grid_dim, (10400, 1, 1));
            assert_eq!(cfg.block_dim, (4, 64, 1));
            assert_eq!(cfg.shared_mem_bytes, 1024);
        } else {
            assert!(std::ptr::eq(selected, &kernels.reduce_sum_axis0));
            assert_eq!(cfg.grid_dim, (499200, 1, 1));
            assert_eq!(cfg.block_dim, (32, 1, 1));
            assert_eq!(cfg.shared_mem_bytes, 128);
        }
        for (rows, columns, threads) in [(4, 4095, 32), (65, 499200, 128)] {
            let (selected, cfg) = kernels.axis0_reduction(rows, columns);
            assert!(std::ptr::eq(selected, &kernels.reduce_sum_axis0));
            assert_eq!(cfg.grid_dim, (columns as u32, 1, 1));
            assert_eq!(cfg.block_dim, (threads, 1, 1));
            assert_eq!(cfg.shared_mem_bytes, threads * 4);
        }
        let cfg = kernels.angle_backward_cfg(8, 1300, 48, 4);
        assert_eq!(cfg.grid_dim, (8, if cap == 16 { 6 } else { 1 }, 1));
        assert_eq!(cfg.block_dim, (if cap == 16 { 32 } else { 192 }, 1, 1));
        for (batch, time, heads, angles) in [(128, 1300, 48, 4), (8, 127, 48, 4)] {
            let cfg = kernels.angle_backward_cfg(batch, time, heads, angles);
            assert_eq!(cfg.grid_dim, (batch as u32, 1, 1));
            assert_eq!(cfg.block_dim, (192, 1, 1));
        }
        raw::qualify(&device, &kernels);
    }
}
