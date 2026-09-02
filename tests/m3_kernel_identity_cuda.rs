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

fn one_cache_entry(cache: &std::path::Path) -> std::path::PathBuf {
    let entries: Vec<_> = std::fs::read_dir(cache)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("mamba3-kernels-v2-")
        })
        .collect();
    assert_eq!(entries.len(), 1, "one envelope entry per compile key");
    entries.into_iter().next().unwrap()
}

#[test]
fn m3_cache_hit_preserves_the_exact_artifact_identity() {
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
    let cached_entry = one_cache_entry(&cache);
    let cached_before = std::fs::metadata(&cached_entry).unwrap();
    let second =
        Mamba3Kernels::compile(device.context(), device.nvrtc_target()).expect("second M3 compile");
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

    assert_eq!(first.compiler_identity(), second.compiler_identity());
    assert!(first.compiler_identity().nvrtc_library_known);
    assert_eq!(first.artifact_identity(), second.artifact_identity());
    assert_eq!(
        first.artifact_identity().module_kind,
        ModuleKind::Mamba3Combined
    );
    assert_eq!(first.artifact_identity().artifact_kind, ArtifactKind::Ptx);

    assert_eq!(
        std::fs::read(legacy).unwrap(),
        b"legacy raw cache stays untouched"
    );
}
