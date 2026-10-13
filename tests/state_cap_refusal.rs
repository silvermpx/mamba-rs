//! Kernels compiled for a smaller state than the dims ask refuse the run.
//!
//! The SSM kernels keep the state in per-thread arrays sized at compile
//! time, and past that size every one of them returns without writing. The
//! low-level entry points of Mamba-1 and Mamba-3 take kernels and dims
//! separately, so a pair that does not fit is refused at the entry, with an
//! error naming both sizes, instead of producing zeros.
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use mamba_rs::config::ScanMode;
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::mamba1::gpu::forward::GpuMambaDims;
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
use mamba_rs::mamba3_siso::gpu::state::{GpuMamba3Dims, M3Exec};

fn m1_dims(d_state: usize, d_conv: usize) -> GpuMambaDims {
    let dt_rank = 2;
    GpuMambaDims {
        batch: 1,
        d_model: 32,
        d_inner: 64,
        d_state,
        d_conv,
        dt_rank,
        xdbl_dim: dt_rank + 2 * d_state,
        seq_len: 8,
        mamba_input_dim: 32,
        n_layers: 1,
        scan_mode: ScanMode::Sequential,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
    }
}

#[test]
fn m1_dims_past_the_compiled_capacities_are_refused() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new_with_state_cap(&device, 16).expect("context at 16 states");
    m1_dims(16, 4).check_for(&ctx).expect("16 states fit");
    let err = m1_dims(32, 4).check_for(&ctx).unwrap_err();
    assert!(err.contains("d_state 32") && err.contains("16"), "{err}");
    // Any conv width runs; a conv needs a tap.
    m1_dims(16, 9).check_for(&ctx).expect("nine taps run");
    let err = m1_dims(16, 0).check_for(&ctx).unwrap_err();
    assert!(err.contains("d_conv 0"), "{err}");
    let mut skewed = m1_dims(16, 4);
    skewed.xdbl_dim += 1;
    assert!(skewed.check_for(&ctx).is_err());
}

#[test]
fn m3_dims_past_the_compiled_capacity_are_refused() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let kernels = Mamba3Kernels::compile_with_state_cap(ctx.stream.context(), arch::arch0(), 16)
        .expect("kernels at 16 states");
    let dims = |d_state: usize| GpuMamba3Dims {
        batch: 1,
        d_model: 32,
        d_inner: 64,
        d_state,
        nheads: 8,
        headdim: 8,
        ngroups: 1,
        in_proj_dim: 2 * 64 + 2 * d_state + 3 * 8 + d_state / 4,
        seq_len: 8,
        mamba_input_dim: 32,
        n_layers: 1,
        n_angles: d_state / 4,
        a_floor: 1e-4,
        rms_norm_eps: 1e-5,
        ssm_norm_eps: 1e-5,
        is_outproj_norm: false,
        use_parallel_scan: true,
    };
    let fits = dims(16);
    M3Exec {
        ctx: &ctx,
        kernels: &kernels,
        dims: &fits,
    }
    .check_state_cap()
    .expect("16 states fit");
    let too_wide = dims(32);
    let err = M3Exec {
        ctx: &ctx,
        kernels: &kernels,
        dims: &too_wide,
    }
    .check_state_cap()
    .unwrap_err();
    assert!(err.contains("d_state 32") && err.contains("16"), "{err}");
}
