#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{ArtifactKind, CacheEnvelope, ModuleKind};

#[cfg(target_os = "linux")]
struct CacheEnvGuard(Option<std::ffi::OsString>);

#[cfg(target_os = "linux")]
struct CudaCacheEnvGuard(Option<std::ffi::OsString>);

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

#[cfg(target_os = "linux")]
impl Drop for CudaCacheEnvGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(value) = self.0.take() {
                std::env::set_var("CUDA_CACHE_DISABLE", value);
            } else {
                std::env::remove_var("CUDA_CACHE_DISABLE");
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn one_cache_entry(cache: &std::path::Path, prefix: &str) -> std::path::PathBuf {
    let entries: Vec<_> = std::fs::read_dir(cache)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(prefix)
        })
        .collect();
    assert_eq!(entries.len(), 1, "one envelope entry per compile key");
    entries.into_iter().next().unwrap()
}

#[cfg(target_os = "linux")]
fn envelope_payload(path: &std::path::Path) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    let key = bytes[18..50].try_into().unwrap();
    CacheEnvelope::decode(key, ArtifactKind::Ptx, &bytes)
        .unwrap()
        .payload
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
    let _cuda_cache_guard = CudaCacheEnvGuard(std::env::var_os("CUDA_CACHE_DISABLE"));
    #[cfg(target_os = "linux")]
    unsafe {
        std::env::set_var("MAMBA_RS_KERNEL_CACHE", root.path().join("cache"));
        std::env::set_var("CUDA_CACHE_DISABLE", "1");
    }

    let device = GpuDevice::new(0).expect("CUDA device");
    let first = GpuCtx::new(&device).expect("first context");
    #[cfg(target_os = "linux")]
    let cached_entry = { one_cache_entry(&root.path().join("cache"), "mamba-kernels-v1-") };
    #[cfg(target_os = "linux")]
    let cached_before = std::fs::metadata(&cached_entry).unwrap();
    let second = GpuCtx::new(&device).expect("second context");
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::MetadataExt;

        let cached_after = std::fs::metadata(&cached_entry).unwrap();
        assert_eq!(
            cached_before.ino(),
            cached_after.ino(),
            "cache hit replaced the entry"
        );
        assert_eq!(
            cached_before.modified().unwrap(),
            cached_after.modified().unwrap(),
            "cache hit rewrote the entry"
        );
    }

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
    if first.kernels.compiler_identity().nvrtc_version >= (12, 9) {
        let first_payload = envelope_payload(&cached_entry);
        let second_cache = root.path().join("second-cache");
        unsafe {
            std::env::set_var("MAMBA_RS_KERNEL_CACHE", &second_cache);
        }
        let third = GpuCtx::new(&device).expect("independent cold context");
        let second_entry = one_cache_entry(&second_cache, "mamba-kernels-v1-");
        let second_payload = envelope_payload(&second_entry);
        assert_eq!(
            first_payload, second_payload,
            "two cold NVRTC compiles produced different canonical PTX"
        );
        assert_eq!(
            first.kernels.artifact_set_identity(),
            third.kernels.artifact_set_identity()
        );
    }
}
