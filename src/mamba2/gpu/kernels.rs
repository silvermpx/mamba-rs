//! The Mamba-2 CUDA module: `kernels/mamba2_ops.cu` after the typed
//! prelude, compiled with NVRTC and cached under its own module kind, so
//! adding it moves no identity of the Fixed or Mamba-3 modules. The norms
//! and the residual adds come from the Fixed module through the context.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig};

use crate::mamba_ssm::gpu::kernel_identity as ki;
use crate::mamba_ssm::gpu::kernels::TypedKernel;

/// The compiled Mamba-2 kernels.
pub struct Mamba2Kernels {
    /// The decode step of every head, by activation dtype.
    pub step_fwd: TypedKernel,
    /// The grouped gated RMSNorm.
    pub gated_norm_fwd: TypedKernel,
    /// The gate without a norm.
    pub gate_only_fwd: TypedKernel,
    /// The mixer's inner MLP gate.
    pub mlp_gate_fwd: TypedKernel,
    /// The prefill: conv over the window and its state, dt with the chunk
    /// cumsum, the CB products, the chunk states, the state passing and
    /// the chunk scan.
    pub conv_prefill: TypedKernel,
    /// The decode's conv, the prefill conv's chain at one token.
    pub conv_step: TypedKernel,
    pub conv_state: TypedKernel,
    pub dt_cumsum: TypedKernel,
    pub bmm_chunk: TypedKernel,
    pub chunk_state: TypedKernel,
    pub state_passing: CudaFunction,
    pub chunk_scan: TypedKernel,
    pub compiler_identity: ki::CompilerIdentity,
    pub artifact_identity: ki::ArtifactIdentity,
}

/// The threads of one decode-step block: one warp, one channel each. Its
/// shared state tile is `32 * (d_state + 1)` floats, 33 KB at d_state 256.
pub const STEP_THREADS: u32 = 32;

/// Launch geometry of [`Mamba2Kernels::step_fwd`].
pub fn step_cfg(batch: usize, nh: usize, hd: usize, ds: usize) -> LaunchConfig {
    let rows = (STEP_THREADS as usize).min(hd);
    LaunchConfig {
        grid_dim: (
            batch as u32,
            nh as u32,
            hd.div_ceil(STEP_THREADS as usize) as u32,
        ),
        block_dim: (STEP_THREADS, 1, 1),
        shared_mem_bytes: ((2 * ds + 4 + rows * (ds + 1)) * 4) as u32,
    }
}

/// Tokens per SSD chunk.
pub const CHUNK: usize = 64;

/// Launch geometry of the gated norm: one block per (row, group), its
/// group and per-head sums staged in shared memory.
pub fn gated_norm_cfg(rows: usize, d_ssm: usize, ng: usize, hd: usize) -> LaunchConfig {
    let gs = d_ssm / ng;
    LaunchConfig {
        grid_dim: (rows as u32, ng as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: ((gs + gs / hd) * 4) as u32,
    }
}

impl Mamba2Kernels {
    /// Compile (or load from the cache) the Mamba-2 module for `arch`.
    pub fn compile(ctx: &Arc<CudaContext>, arch: &'static str) -> Result<Self, String> {
        let sources = [
            include_str!("../../../kernels/_typed_prelude.cuh"),
            include_str!("../../../kernels/mamba2_ops.cu"),
            include_str!("../../../kernels/mamba2_ssd.cu"),
        ];
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
            "2405210601",
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
            module_kind: ki::ModuleKind::Mamba2Combined,
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
                .map(|dir| dir.join(format!("mamba2-kernels-v1-{}.bin", ki::digest_hex(&key))))
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
                        "NVRTC Mamba-2 compile failed: {}",
                        format!("{e:?}").replace("\\n", "\n")
                    )
                })?;
                let image = ptx
                    .as_bytes()
                    .ok_or("NVRTC returned Mamba-2 PTX without a raw image")?;
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
                    return Err(
                        "NVRTC libraries changed during the Mamba-2 compilation; retry".into(),
                    );
                }
                let artifact_digest = ki::FramedSha256::bytes(ptx_source.as_bytes());
                if let (Some(path), Some(key)) = (&cache_path, cache_key) {
                    ki::publish_cache(path, key, ki::ArtifactKind::Ptx, ptx_source.as_bytes());
                }
                let module = ctx
                    .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
                    .map_err(|e| format!("Mamba-2 module load failed: {e:?}"))?;
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
            module_kind: ki::ModuleKind::Mamba2Combined,
            artifact_kind: ki::ArtifactKind::Ptx,
            compile_key: invocation_digest,
            artifact_digest,
        };
        let get = |name: &str| -> Result<CudaFunction, String> {
            module
                .load_function(name)
                .map_err(|e| format!("Mamba-2 kernel '{name}' not found: {e:?}"))
        };
        let typed = |base: &str| -> Result<TypedKernel, String> {
            Ok(TypedKernel {
                f32: get(base)?,
                bf16: get(&format!("{base}_bf16"))?,
                f16: get(&format!("{base}_f16"))?,
            })
        };
        Ok(Self {
            step_fwd: typed("m2_step_fwd")?,
            gated_norm_fwd: typed("m2_gated_norm_fwd")?,
            gate_only_fwd: typed("m2_gate_only_fwd")?,
            mlp_gate_fwd: typed("m2_mlp_gate_fwd")?,
            conv_prefill: typed("m2_conv_prefill")?,
            conv_step: typed("m2_conv_step")?,
            conv_state: typed("m2_conv_state")?,
            dt_cumsum: typed("m2_dt_cumsum")?,
            bmm_chunk: typed("m2_bmm_chunk")?,
            chunk_state: typed("m2_chunk_state")?,
            state_passing: get("m2_state_passing")?,
            chunk_scan: typed("m2_chunk_scan")?,
            compiler_identity,
            artifact_identity,
        })
    }
}
