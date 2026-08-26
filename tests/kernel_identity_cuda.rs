#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{ArtifactKind, ModuleKind};

#[test]
fn repeated_nvrtc_compiles_have_the_same_identity() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let first = GpuCtx::new(&device).expect("first context");
    let second = GpuCtx::new(&device).expect("second context");

    assert_eq!(
        first.kernels.compiler_identity(),
        second.kernels.compiler_identity()
    );
    let first_artifacts = first.kernels.artifact_set_identity();
    let second_artifacts = second.kernels.artifact_set_identity();
    assert_eq!(first_artifacts, second_artifacts);
    assert_eq!(first_artifacts.module_count, 1);
    assert_eq!(
        first_artifacts.legacy_combined.module_kind,
        ModuleKind::LegacyCombined
    );
    assert_eq!(
        first_artifacts.legacy_combined.artifact_kind,
        ArtifactKind::Ptx
    );
}
