//! One NVRTC module composed from kernel sources, cached under its own
//! module kind: the identity, cache and header-closure flow every family
//! module beyond the Fixed and Mamba-3 ones shares.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaModule};

use super::kernel_identity as ki;

/// A loaded module and its identities.
pub(crate) struct ComposedModule {
    pub module: Arc<CudaModule>,
    pub compiler_identity: ki::CompilerIdentity,
    pub artifact_identity: ki::ArtifactIdentity,
}

/// What names one composed module: its sources in order (the typed prelude
/// first; each source's own include of it is stripped), its kind, its
/// cache file prefix, the NVRTC seed and any extra options.
pub(crate) struct ComposedSpec<'a> {
    pub sources: &'a [&'a str],
    pub kind: ki::ModuleKind,
    pub cache_prefix: &'a str,
    pub seed: &'a str,
    pub label: &'a str,
}

/// Compile (or load from the cache) a composed module for `arch`.
pub(crate) fn compile_composed(
    ctx: &Arc<CudaContext>,
    arch: &'static str,
    spec: &ComposedSpec<'_>,
) -> Result<ComposedModule, String> {
    let sources = spec.sources;
    let combined: String = sources
        .iter()
        .map(|s| {
            s.lines()
                .filter(|l| !l.trim().starts_with("#include \"_typed_prelude.cuh\""))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let (nv_major, nv_minor) = crate::mamba_ssm::gpu::kernels::nvrtc_version();
    let mut option_strings = vec![
        "--fmad=true".to_string(),
        "--extra-device-vectorization".to_string(),
        "-DNDEBUG".to_string(),
    ];
    option_strings.extend(ki::deterministic_nvrtc_options(
        (nv_major, nv_minor),
        spec.seed,
    ));
    let include_paths = crate::mamba_ssm::gpu::kernels::cuda_include_paths();
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some(arch),
        options: option_strings.clone(),
        include_paths: include_paths.clone(),
        ..Default::default()
    };
    let nvrtc_library_domain = ki::nvrtc_library_domain();
    let header_manifest = ki::header_manifest(combined.as_bytes(), &include_paths);
    let mut argv = vec![format!("--gpu-architecture={arch}").into_bytes()];
    argv.extend(option_strings.iter().map(|v| v.as_bytes().to_vec()));
    let key_material = ki::CompileKeyMaterial {
        module_kind: spec.kind,
        source: combined.as_bytes().to_vec(),
        target: arch.as_bytes().to_vec(),
        argv,
        header_manifest: header_manifest.clone(),
        nvrtc_version: (nv_major, nv_minor),
        nvrtc_library_domain: nvrtc_library_domain.clone(),
        output_kind: ki::ArtifactKind::Ptx,
        composer_revision: ki::COMPOSER_REVISION,
        compiler_revision: ki::COMPILER_REVISION,
        numeric_abi_revision: ki::NUMERIC_ABI_REVISION,
        schedule_revision: ki::SCHEDULE_REVISION,
    };
    let invocation_digest = key_material.invocation_digest();
    let cache_key = key_material.digest();
    let cache_path = cache_key.and_then(|key| {
        crate::mamba_ssm::gpu::kernels::kernel_cache_dir()
            .map(|dir| dir.join(format!("{}{}.bin", spec.cache_prefix, ki::digest_hex(&key))))
    });

    let mut loaded = None;
    if let (Some(path), Some(key)) = (&cache_path, cache_key)
        && let Some(hit) = ki::read_cache(path, key, ki::ArtifactKind::Ptx)
        && let Ok(source) = ki::canonical_ptx_from_cache(hit.payload)
        && let Ok(module) = ctx.load_module(cudarc::nvrtc::Ptx::from_src(source))
        && ki::cache_hit_header_closure_is_current(
            combined.as_bytes(),
            &include_paths,
            &header_manifest,
        )
        && nvrtc_library_domain
            .as_deref()
            .is_some_and(ki::nvrtc_library_domain_is_current)
    {
        loaded = Some((module, hit.artifact_digest));
    }
    let (module, artifact_digest) = match loaded {
        Some(value) => value,
        None => {
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(&combined, opts).map_err(|e| {
                format!(
                    "NVRTC {} compile failed: {}",
                    spec.label,
                    format!("{e:?}").replace("\\n", "\n")
                )
            })?;
            let image = ptx
                .as_bytes()
                .ok_or_else(|| format!("NVRTC returned {} PTX without a raw image", spec.label))?;
            let ptx_source = ki::canonical_ptx_image(image)?;
            if !ki::header_manifest_is_current(
                combined.as_bytes(),
                &include_paths,
                &header_manifest,
            ) {
                return Err("CUDA headers changed during the Mamba-2 compilation; retry".into());
            }
            if let Some(domain) = nvrtc_library_domain.as_deref()
                && !ki::nvrtc_library_domain_is_current(domain)
            {
                return Err(format!(
                    "NVRTC libraries changed during the {} compilation; retry",
                    spec.label
                ));
            }
            let artifact_digest = ki::FramedSha256::bytes(ptx_source.as_bytes());
            if let (Some(path), Some(key)) = (&cache_path, cache_key) {
                ki::publish_cache(path, key, ki::ArtifactKind::Ptx, ptx_source.as_bytes());
            }
            let module = ctx
                .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
                .map_err(|e| format!("{} module load failed: {e:?}", spec.label))?;
            (module, artifact_digest)
        }
    };
    let compiler_identity = ki::CompilerIdentity {
        source_digest: ki::FramedSha256::bytes(combined.as_bytes()),
        invocation_digest,
        header_manifest_digest: ki::FramedSha256::new(b"cuda-header-manifest.v1")
            .optional(b"manifest", header_manifest.as_deref())
            .finish(),
        target: ki::CudaTarget::new(arch)?,
        nvrtc_version: (nv_major, nv_minor),
        nvrtc_library_domain: ki::FramedSha256::new(b"nvrtc-library-set-identity.v2")
            .optional(b"domain", nvrtc_library_domain.as_deref())
            .finish(),
        nvrtc_library_known: nvrtc_library_domain.is_some(),
        output_kind: ki::ArtifactKind::Ptx,
        composer_revision: ki::COMPOSER_REVISION,
        compiler_revision: ki::COMPILER_REVISION,
        numeric_abi_revision: ki::NUMERIC_ABI_REVISION,
        schedule_revision: ki::SCHEDULE_REVISION,
    };
    let artifact_identity = ki::ArtifactIdentity {
        module_kind: spec.kind,
        artifact_kind: ki::ArtifactKind::Ptx,
        compile_key: invocation_digest,
        artifact_digest,
    };
    Ok(ComposedModule {
        module,
        compiler_identity,
        artifact_identity,
    })
}
