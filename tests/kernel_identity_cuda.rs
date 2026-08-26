#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
#[cfg(target_os = "linux")]
use mamba_rs::mamba_ssm::gpu::kernel_identity::{ArtifactIdentity, CacheEnvelope, FramedSha256};
use mamba_rs::mamba_ssm::gpu::kernel_identity::{ArtifactKind, ModuleKind};

#[cfg(target_os = "linux")]
const SCALAR_ENTRIES: &[&str] = &[
    "sgemm_bi_nn",
    "sgemm_bi_tn",
    "sgemm_bi_tn_splitm_partial",
    "sgemm_bi_splitm_reduce",
    "sgemm_bi_nn_splitk_big_partial",
    "sgemm_bi_nt",
    "sgemm_bi_nt_splitn_big_partial",
    "sgemm_bi_nn_slim",
    "sgemm_bi_nn_splitk_slim_partial",
    "sgemm_bi_tn_slim",
    "sgemm_bi_nt_slim",
    "sgemm_bi_nn_ultra_thin",
    "sgemm_bi_nn_gemv",
    "sgemm_bi_tn_gemv",
    "sgemm_bi_nt_gemv",
    "sgemm_bi_nn_narrow",
    "sgemm_bi_nn_narrow_small",
    "sgemm_bi_tn_narrow",
    "sgemm_bi_tn_narrow_splitm_partial",
    "sgemm_bi_nt_narrow",
    "sgemm_bi_nn_splitk32_partial",
    "sgemm_bi_splitk_reduce",
    "sgemm_bi_dx_col_gemv",
    "sgemm_transpose_f32_2d",
    "sgemm_bi_nn_gemv_bf16",
    "sgemm_bi_nn_gemv_f16",
    "sgemm_bi_tn_gemv_bf16",
    "sgemm_bi_tn_gemv_f16",
    "sgemm_bi_nt_gemv_bf16",
    "sgemm_bi_nt_gemv_f16",
    "sgemm_bi_nn_ultra_thin_bf16",
    "sgemm_bi_nn_ultra_thin_f16",
    "sgemm_bi_nn_narrow_bf16",
    "sgemm_bi_nn_narrow_f16",
    "sgemm_bi_nn_narrow_small_bf16",
    "sgemm_bi_nn_narrow_small_f16",
    "sgemm_bi_tn_narrow_bf16",
    "sgemm_bi_tn_narrow_f16",
    "sgemm_bi_nt_narrow_bf16",
    "sgemm_bi_nt_narrow_f16",
    "sgemm_bi_nn_big_bf16",
    "sgemm_bi_nn_big_f16",
    "sgemm_bi_tn_big_bf16",
    "sgemm_bi_tn_big_f16",
    "sgemm_bi_nt_big_bf16",
    "sgemm_bi_nt_big_f16",
];

#[cfg(target_os = "linux")]
const SM80_ENTRIES: &[&str] = &[
    "sgemm_bi_nn_tc_bf16",
    "sgemm_bi_nn_tc_f16",
    "sgemm_bi_tn_tc_bf16",
    "sgemm_bi_tn_tc_f16",
    "sgemm_bi_nt_tc_bf16",
    "sgemm_bi_nt_tc_f16",
    "sgemm_bi_nn_tc64_bf16",
    "sgemm_bi_nn_tc64_f16",
    "sgemm_bi_nn_tc16_bf16",
    "sgemm_bi_nn_tc16_f16",
    "sgemm_bi_tn_tc64_bf16",
    "sgemm_bi_tn_tc64_f16",
    "sgemm_bi_nt_tc64_bf16",
    "sgemm_bi_nt_tc64_f16",
];

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
fn cache_entries(cache: &std::path::Path, prefix: &str) -> Vec<std::path::PathBuf> {
    let mut entries: Vec<_> = std::fs::read_dir(cache)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(prefix)
        })
        .collect();
    entries.sort();
    assert_eq!(entries.len(), 3, "one envelope per active CUDA module");
    entries
}

#[cfg(target_os = "linux")]
fn envelope_payload(path: &std::path::Path) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    let key = bytes[18..50].try_into().unwrap();
    CacheEnvelope::decode(key, ArtifactKind::Ptx, &bytes)
        .unwrap()
        .payload
}

#[cfg(target_os = "linux")]
fn artifact_payload(entries: &[std::path::PathBuf], artifact: ArtifactIdentity) -> Vec<u8> {
    envelope_payload(&artifact_cache_entry(entries, artifact))
}

#[cfg(target_os = "linux")]
fn artifact_cache_entry(
    entries: &[std::path::PathBuf],
    artifact: ArtifactIdentity,
) -> std::path::PathBuf {
    entries
        .iter()
        .find(|path| FramedSha256::bytes(&envelope_payload(path)) == artifact.artifact_digest)
        .cloned()
        .unwrap_or_else(|| panic!("cache payload for {:?} not found", artifact.module_kind))
}

#[cfg(target_os = "linux")]
fn ptx_entries(payload: &[u8]) -> std::collections::BTreeSet<String> {
    let source = std::str::from_utf8(payload).expect("canonical PTX is UTF-8");
    source
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let rest = line
                .strip_prefix(".visible .entry ")
                .or_else(|| line.strip_prefix(".entry "))?;
            rest.split(['(', ' ', '\t'])
                .next()
                .filter(|name| !name.is_empty())
                .map(str::to_string)
        })
        .collect()
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
    let first_artifacts = first.kernels.artifact_set_identity();
    #[cfg(target_os = "linux")]
    let cached_entries = cache_entries(&root.path().join("cache"), "mamba-kernels-v1-");
    #[cfg(target_os = "linux")]
    let cached_before: Vec<_> = cached_entries
        .iter()
        .map(|path| (path.clone(), std::fs::metadata(path).unwrap()))
        .collect();
    let second = GpuCtx::new(&device).expect("second context");
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::MetadataExt;

        for (path, before) in &cached_before {
            let after = std::fs::metadata(path).unwrap();
            assert_eq!(before.ino(), after.ino(), "cache hit replaced {path:?}");
            assert_eq!(
                before.modified().unwrap(),
                after.modified().unwrap(),
                "cache hit rewrote {path:?}"
            );
        }
    }

    assert_eq!(
        first.kernels.compiler_identity(),
        second.kernels.compiler_identity()
    );
    let second_artifacts = second.kernels.artifact_set_identity();
    assert_eq!(first_artifacts, second_artifacts);
    assert_eq!(first_artifacts.module_count, 3);
    assert_eq!(first_artifacts.fixed.module_kind, ModuleKind::Fixed);
    assert_eq!(first_artifacts.fixed.artifact_kind, ArtifactKind::Ptx);
    assert_eq!(
        first_artifacts.triad_scalar.module_kind,
        ModuleKind::TriadScalar
    );
    assert_eq!(
        first_artifacts.triad_sm80.module_kind,
        ModuleKind::TriadSm80
    );
    assert_eq!(first_artifacts.specialized, None);
    assert_eq!(
        first_artifacts.fixed.compile_key,
        first.kernels.compiler_identity().invocation_digest
    );
    assert_eq!(
        first_artifacts.triad_scalar.compile_key,
        first.kernels.scalar_compiler_identity().invocation_digest
    );
    assert_eq!(
        first_artifacts.triad_sm80.compile_key,
        first.kernels.sm80_compiler_identity().invocation_digest
    );
    #[cfg(target_os = "linux")]
    {
        let fixed_entries = ptx_entries(&artifact_payload(&cached_entries, first_artifacts.fixed));
        assert!(
            fixed_entries.iter().all(|name| !name.starts_with("sgemm_")),
            "Fixed unexpectedly exports triad symbols: {fixed_entries:?}"
        );

        let scalar_entries = ptx_entries(&artifact_payload(
            &cached_entries,
            first_artifacts.triad_scalar,
        ));
        let expected_scalar: std::collections::BTreeSet<_> = SCALAR_ENTRIES
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        assert_eq!(scalar_entries, expected_scalar);

        let sm80_entries = ptx_entries(&artifact_payload(
            &cached_entries,
            first_artifacts.triad_sm80,
        ));
        let expected_sm80: std::collections::BTreeSet<_> = SM80_ENTRIES
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        assert_eq!(sm80_entries, expected_sm80);
    }
    #[cfg(target_os = "linux")]
    assert!(first.kernels.compiler_identity().nvrtc_library_known);
    #[cfg(target_os = "linux")]
    if first.kernels.compiler_identity().nvrtc_version >= (12, 9) {
        let second_cache = root.path().join("second-cache");
        unsafe {
            std::env::set_var("MAMBA_RS_KERNEL_CACHE", &second_cache);
        }
        let third = GpuCtx::new(&device).expect("independent cold context");
        let second_entries = cache_entries(&second_cache, "mamba-kernels-v1-");
        for artifact in [
            first_artifacts.fixed,
            first_artifacts.triad_scalar,
            first_artifacts.triad_sm80,
        ] {
            assert_eq!(
                artifact_payload(&cached_entries, artifact),
                artifact_payload(&second_entries, artifact),
                "two cold NVRTC compiles produced different {:?} PTX",
                artifact.module_kind
            );
        }
        assert_eq!(
            first.kernels.artifact_set_identity(),
            third.kernels.artifact_set_identity()
        );
    }

    #[cfg(target_os = "linux")]
    {
        unsafe {
            std::env::set_var("MAMBA_RS_KERNEL_CACHE", root.path().join("cache"));
        }
        let scalar_entry = artifact_cache_entry(&cached_entries, first_artifacts.triad_scalar);
        let mut scalar_payload = artifact_payload(&cached_entries, first_artifacts.triad_scalar);
        let needle = b".entry sgemm_bi_nn(";
        let replacement = b".entry sgemm_bi_nx(";
        let offsets: Vec<_> = scalar_payload
            .windows(needle.len())
            .enumerate()
            .filter_map(|(offset, value)| (value == needle).then_some(offset))
            .collect();
        assert_eq!(offsets.len(), 1, "one Scalar sgemm_bi_nn PTX entry");
        let offset = offsets[0];
        scalar_payload[offset..offset + needle.len()].copy_from_slice(replacement);
        std::fs::write(
            scalar_entry,
            CacheEnvelope::encode(
                first_artifacts.triad_scalar.compile_key,
                ArtifactKind::Ptx,
                &scalar_payload,
            ),
        )
        .expect("rewrite valid Scalar cache envelope");

        let error = match GpuCtx::new(&device) {
            Ok(_) => panic!("missing owned Scalar symbol unexpectedly fell back"),
            Err(error) => error,
        };
        assert!(error.contains("TriadScalar"), "{error}");
        assert!(error.contains("sgemm_bi_nn"), "{error}");
    }
}
