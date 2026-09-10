//! Device and kernel facts — no timing, no math. Prints the register /
//! local / shared footprint and the achieved occupancy of the production
//! kernels at their production launch geometry, plus the device limits
//! those numbers trade against. One run settles occupancy questions that
//! otherwise get argued from datasheets.
//!
//!   cargo test --release --features cuda --test device_facts -- --ignored --nocapture

#![cfg(feature = "cuda")]

#[path = "../../tests/common/arch.rs"]
mod arch;

use cudarc::driver::sys::CUdevice_attribute_enum as DevAttr;
use cudarc::driver::sys::CUfunction_attribute_enum as FnAttr;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::launch::{grid_parallel_scan_bwd, grid_parallel_scan_bwd_fold};
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;

#[test]
#[ignore = "manual fact sheet"]
fn device_and_kernel_facts() {
    let device = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new_with_state_cap(&device, 16).unwrap();
    let cuda = device.context();

    let dev_attr = |a: DevAttr| cuda.attribute(a).unwrap_or(-1);
    eprintln!(
        "DEVICE arch={} sm_count={} smem/SM={} smem/block-optin={} regs/SM={} l2={}",
        device.nvrtc_target(),
        dev_attr(DevAttr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        dev_attr(DevAttr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR),
        dev_attr(DevAttr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),
        dev_attr(DevAttr::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR),
        dev_attr(DevAttr::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE),
    );

    let (b, di) = (8usize, 768usize);
    let dtype = WeightDtype::Bf16;
    let m3k = Mamba3Kernels::compile(ctx.stream.context(), arch::arch0()).unwrap();

    let fold_cfg = grid_parallel_scan_bwd_fold(b, di, 16, dtype.size_bytes());
    let plain_cfg = grid_parallel_scan_bwd(b, di);
    let facts: [(&str, &cudarc::driver::CudaFunction, u32, usize); 4] = [
        (
            "ssm_parallel_bwd_fold_bf16 (production)",
            ctx.kernels.ssm_parallel_bwd_fold_typed.get(dtype),
            fold_cfg.block_dim.0 * fold_cfg.block_dim.1 * fold_cfg.block_dim.2,
            fold_cfg.shared_mem_bytes as usize,
        ),
        (
            "ssm_parallel_bwd_bf16 (legacy route)",
            ctx.kernels.ssm_parallel_bwd_typed.get(dtype),
            plain_cfg.block_dim.0 * plain_cfg.block_dim.1 * plain_cfg.block_dim.2,
            plain_cfg.shared_mem_bytes as usize,
        ),
        (
            "ssm_reduce_d_BC_tmajor_bf16",
            ctx.kernels.ssm_reduce_d_bc_tmajor_typed.get(dtype),
            256,
            0,
        ),
        (
            "m3_chunk_scan_fwd_bf16",
            m3k.m3_chunk_scan_fwd_typed.get(dtype),
            256,
            0,
        ),
    ];
    for (name, f, block, smem) in facts {
        let regs = f
            .get_attribute(FnAttr::CU_FUNC_ATTRIBUTE_NUM_REGS)
            .unwrap_or(-1);
        let local = f
            .get_attribute(FnAttr::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
            .unwrap_or(-1);
        let static_smem = f
            .get_attribute(FnAttr::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
            .unwrap_or(-1);
        let occ = f
            .occupancy_max_active_blocks_per_multiprocessor(block, smem, None)
            .unwrap_or(0);
        eprintln!(
            "KERNEL {name}: regs={regs} local_bytes={local} static_smem={static_smem} \
             dyn_smem={smem} block={block} occupancy_blocks_per_sm={occ}"
        );
    }
}
