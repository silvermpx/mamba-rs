#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{ArtifactKind, ModuleKind};

#[cfg(target_os = "linux")]
struct CacheEnvGuard(Option<std::ffi::OsString>);

#[cfg(target_os = "linux")]
impl Drop for CacheEnvGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(value) = self.0.take() {
                std::env::set_var("MAMBA_RS_KERNEL_CACHE", value);
            } else {
                std::env::remove_var("MAMBA_RS_KERNEL_CACHE");
            }
        }
    }
}

#[test]
fn repeated_nvrtc_compiles_have_the_same_identity() {
    #[cfg(target_os = "linux")]
    let home = std::env::var_os("HOME").expect("HOME");
    #[cfg(target_os = "linux")]
    let home = std::fs::canonicalize(home).expect("canonical HOME");
    #[cfg(target_os = "linux")]
    let root = tempfile::Builder::new()
        .prefix("mamba-cache-test-")
        .tempdir_in(home)
        .expect("trusted cache root");
    #[cfg(target_os = "linux")]
    let _env_guard = CacheEnvGuard(std::env::var_os("MAMBA_RS_KERNEL_CACHE"));
    #[cfg(target_os = "linux")]
    unsafe {
        std::env::set_var("MAMBA_RS_KERNEL_CACHE", root.path().join("cache"));
    }

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
    #[cfg(target_os = "linux")]
    assert!(first.kernels.compiler_identity().nvrtc_library_known);
    #[cfg(target_os = "linux")]
    let entries: Vec<_> = std::fs::read_dir(root.path().join("cache"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .filter(|name| name.to_string_lossy().starts_with("mamba-kernels-v1-"))
        .collect();
    #[cfg(target_os = "linux")]
    assert_eq!(entries.len(), 1, "one envelope entry per compile key");
}
