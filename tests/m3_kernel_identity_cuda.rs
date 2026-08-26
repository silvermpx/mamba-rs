#![cfg(all(feature = "cuda", target_os = "linux"))]

use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{ArtifactKind, ModuleKind};
use mamba_rs::mamba3_siso::gpu::Mamba3Kernels;

struct CacheEnvGuard(Option<std::ffi::OsString>);

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
fn repeated_m3_nvrtc_compiles_have_the_same_identity() {
    let home = std::env::var_os("HOME").expect("HOME");
    let home = std::fs::canonicalize(home).expect("canonical HOME");
    let root = tempfile::Builder::new()
        .prefix("mamba3-cache-test-")
        .tempdir_in(home)
        .expect("trusted cache root");
    let cache = root.path().join("cache");
    let _env_guard = CacheEnvGuard(std::env::var_os("MAMBA_RS_KERNEL_CACHE"));
    unsafe {
        std::env::set_var("MAMBA_RS_KERNEL_CACHE", &cache);
    }
    let legacy = cache.join("mamba3-kernels-deadbeef.ptx");
    std::fs::create_dir_all(&cache).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    std::fs::write(&legacy, b"legacy raw cache stays untouched").unwrap();

    let device = GpuDevice::new(0).expect("CUDA device");
    let first =
        Mamba3Kernels::compile(device.context(), device.nvrtc_target()).expect("first M3 compile");
    let second =
        Mamba3Kernels::compile(device.context(), device.nvrtc_target()).expect("second M3 compile");

    assert_eq!(first.compiler_identity(), second.compiler_identity());
    assert!(first.compiler_identity().nvrtc_library_known);
    assert_eq!(first.artifact_identity(), second.artifact_identity());
    assert_eq!(
        first.artifact_identity().module_kind,
        ModuleKind::Mamba3Combined
    );
    assert_eq!(first.artifact_identity().artifact_kind, ArtifactKind::Ptx);

    let entries: Vec<_> = std::fs::read_dir(&cache)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .filter(|name| name.to_string_lossy().starts_with("mamba3-kernels-v2-"))
        .collect();
    assert_eq!(entries.len(), 1, "one envelope entry per compile key");
    assert_eq!(
        std::fs::read(legacy).unwrap(),
        b"legacy raw cache stays untouched"
    );
}
