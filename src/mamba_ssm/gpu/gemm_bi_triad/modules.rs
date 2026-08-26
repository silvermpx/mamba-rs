use std::sync::Arc;
use std::{collections::HashMap, sync::Mutex};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream};

use crate::mamba_ssm::gpu::kernel_identity::{
    ArtifactIdentity, ArtifactKind, CompilerIdentity, CudaTarget, FramedSha256, ModuleKind,
};

use super::super::kernels::{
    CudaModuleAnchors, HalfKernel, cuda_include_paths, kernel_cache_dir, nvrtc_version,
};

pub(crate) struct CompileModuleRequest<'a> {
    pub ctx: &'a Arc<CudaContext>,
    pub arch: &'static str,
    pub state_cap: usize,
    pub module_kind: ModuleKind,
}

pub(crate) struct CompiledModule {
    pub module: Arc<CudaModule>,
    pub compiler_identity: CompilerIdentity,
    pub artifact_identity: ArtifactIdentity,
}

pub(crate) fn compile_module(request: CompileModuleRequest<'_>) -> Result<CompiledModule, String> {
    validate_module_target(request.module_kind, request.arch)?;
    let combined = compose_module_source(request.module_kind)?;
    let group_m = match request.arch {
        "sm_80" | "sm_86" | "sm_87" => 8,
        _ => 16,
    };
    let nvrtc = nvrtc_version();
    let mut option_strings = vec![
        "--fmad=true".to_string(),
        "--extra-device-vectorization".to_string(),
        "-DNDEBUG".to_string(),
        format!("-DSGB_GROUP_M={group_m}"),
        format!("-DMAMBA_RS_STATE_CAP={}", request.state_cap),
    ];
    option_strings.extend(
        crate::mamba_ssm::gpu::kernel_identity::deterministic_nvrtc_options(nvrtc, "1295072049"),
    );
    let include_paths = cuda_include_paths();
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some(request.arch),
        options: option_strings.clone(),
        include_paths: include_paths.clone(),
        ..Default::default()
    };
    let nvrtc_library_domain = crate::mamba_ssm::gpu::kernel_identity::nvrtc_library_domain();
    let header_manifest = crate::mamba_ssm::gpu::kernel_identity::header_manifest(
        combined.as_bytes(),
        &include_paths,
    );
    let mut argv: Vec<Vec<u8>> = include_paths
        .iter()
        .map(|path| format!("--include-path={path}").into_bytes())
        .collect();
    argv.push(format!("--gpu-architecture={}", request.arch).into_bytes());
    argv.extend(option_strings.iter().map(|value| value.as_bytes().to_vec()));
    let key_material = crate::mamba_ssm::gpu::kernel_identity::CompileKeyMaterial {
        module_kind: request.module_kind,
        source: combined.as_bytes().to_vec(),
        target: request.arch.as_bytes().to_vec(),
        argv,
        include_roots: include_paths
            .iter()
            .map(|value| value.as_bytes().to_vec())
            .collect(),
        header_manifest: header_manifest.clone(),
        nvrtc_version: nvrtc,
        nvrtc_library_domain: nvrtc_library_domain.clone(),
        output_kind: ArtifactKind::Ptx,
        composer_revision: crate::mamba_ssm::gpu::kernel_identity::COMPOSER_REVISION,
        compiler_revision: crate::mamba_ssm::gpu::kernel_identity::COMPILER_REVISION,
        numeric_abi_revision: crate::mamba_ssm::gpu::kernel_identity::NUMERIC_ABI_REVISION,
        schedule_revision: crate::mamba_ssm::gpu::kernel_identity::SCHEDULE_REVISION,
    };
    let invocation_digest = key_material.invocation_digest();
    let cache_key = key_material.digest();
    let cache_path = cache_key.and_then(|key| {
        kernel_cache_dir().map(|directory| {
            directory.join(format!(
                "mamba-kernels-v1-{}.bin",
                crate::mamba_ssm::gpu::kernel_identity::digest_hex(&key)
            ))
        })
    });

    let mut loaded = None;
    if let (Some(path), Some(key)) = (&cache_path, cache_key)
        && let Some(hit) =
            crate::mamba_ssm::gpu::kernel_identity::read_cache(path, key, ArtifactKind::Ptx)
        && let Ok(src) =
            crate::mamba_ssm::gpu::kernel_identity::canonical_ptx_from_cache(hit.payload)
        && validate_specialized_ptx(request.module_kind, &src).is_ok()
        && let Ok(module) = request.ctx.load_module(cudarc::nvrtc::Ptx::from_src(src))
        && crate::mamba_ssm::gpu::kernel_identity::cache_hit_header_closure_is_current(
            combined.as_bytes(),
            &include_paths,
            &header_manifest,
        )
        && nvrtc_library_domain
            .as_deref()
            .is_some_and(crate::mamba_ssm::gpu::kernel_identity::nvrtc_library_domain_is_current)
    {
        loaded = Some((module, hit.artifact_digest));
    }

    let (module, artifact_digest) = match loaded {
        Some(value) => value,
        None => {
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(&combined, opts).map_err(|error| {
                format!(
                    "{:?} NVRTC compile failed: {}",
                    request.module_kind,
                    format!("{error:?}").replace("\\n", "\n")
                )
            })?;
            let ptx_image = ptx
                .as_bytes()
                .ok_or_else(|| format!("{:?} NVRTC returned no PTX image", request.module_kind))?;
            let ptx_source =
                crate::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(ptx_image)?;
            validate_specialized_ptx(request.module_kind, &ptx_source)?;
            if !crate::mamba_ssm::gpu::kernel_identity::header_manifest_is_current(
                combined.as_bytes(),
                &include_paths,
                &header_manifest,
            ) {
                return Err(format!(
                    "{:?} CUDA headers changed during compilation; retry initialization",
                    request.module_kind
                ));
            }
            if let Some(domain) = nvrtc_library_domain.as_deref()
                && !crate::mamba_ssm::gpu::kernel_identity::nvrtc_library_domain_is_current(domain)
            {
                return Err(format!(
                    "{:?} NVRTC libraries changed during compilation; retry initialization",
                    request.module_kind
                ));
            }
            let artifact_digest = FramedSha256::bytes(ptx_source.as_bytes());
            if let (Some(path), Some(key)) = (&cache_path, cache_key) {
                crate::mamba_ssm::gpu::kernel_identity::publish_cache(
                    path,
                    key,
                    ArtifactKind::Ptx,
                    ptx_source.as_bytes(),
                );
            }
            let module = request
                .ctx
                .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
                .map_err(|error| {
                    format!("{:?} module load failed: {error:?}", request.module_kind)
                })?;
            (module, artifact_digest)
        }
    };

    let compiler_identity = CompilerIdentity {
        source_digest: FramedSha256::bytes(combined.as_bytes()),
        invocation_digest,
        header_manifest_digest: FramedSha256::new(b"cuda-header-manifest.v1")
            .optional(b"manifest", header_manifest.as_deref())
            .finish(),
        target: CudaTarget::new(request.arch)?,
        nvrtc_version: nvrtc,
        nvrtc_library_domain: FramedSha256::new(b"nvrtc-library-set-identity.v2")
            .optional(b"domain", nvrtc_library_domain.as_deref())
            .finish(),
        nvrtc_library_known: nvrtc_library_domain.is_some(),
        output_kind: ArtifactKind::Ptx,
        composer_revision: crate::mamba_ssm::gpu::kernel_identity::COMPOSER_REVISION,
        compiler_revision: crate::mamba_ssm::gpu::kernel_identity::COMPILER_REVISION,
        numeric_abi_revision: crate::mamba_ssm::gpu::kernel_identity::NUMERIC_ABI_REVISION,
        schedule_revision: crate::mamba_ssm::gpu::kernel_identity::SCHEDULE_REVISION,
    };
    Ok(CompiledModule {
        module,
        compiler_identity,
        artifact_identity: ArtifactIdentity {
            module_kind: request.module_kind,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: invocation_digest,
            artifact_digest,
        },
    })
}

fn validate_module_target(kind: ModuleKind, arch: &str) -> Result<(), String> {
    if kind == ModuleKind::TriadSm90a && arch != "sm_90a" {
        return Err(format!(
            "TriadSm90a requires exact target sm_90a, got {arch}"
        ));
    }
    Ok(())
}

fn validate_specialized_ptx(kind: ModuleKind, ptx: &str) -> Result<(), String> {
    if kind != ModuleKind::TriadSm90a {
        return Ok(());
    }
    let target = ptx
        .lines()
        .find_map(|line| line.trim().strip_prefix(".target "))
        .ok_or_else(|| "TriadSm90a PTX has no target directive".to_string())?;
    if target.split(',').next().map(str::trim) != Some("sm_90a") {
        return Err(format!(
            "TriadSm90a PTX target is {target}, expected sm_90a"
        ));
    }
    for &symbol in SM90A_SYMBOLS {
        let marker = format!(".entry {symbol}(");
        if ptx.matches(&marker).count() != 1 {
            return Err(format!("TriadSm90a PTX must contain one entry {symbol}"));
        }
    }
    for instruction in [
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
        "mbarrier.arrive.expect_tx",
        "mbarrier.try_wait.parity",
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16",
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16",
        "wgmma.fence.sync.aligned",
        "wgmma.commit_group.sync.aligned",
        "wgmma.wait_group.sync.aligned",
        "setmaxnreg.dec.sync.aligned.u32",
        "setmaxnreg.inc.sync.aligned.u32",
    ] {
        if !ptx.contains(instruction) {
            return Err(format!("TriadSm90a PTX is missing {instruction}"));
        }
    }
    if ptx.split_ascii_whitespace().any(|token| {
        token.starts_with("atom.")
            || token.starts_with("red.")
            || token.starts_with("atom::")
            || token.starts_with("red::")
    }) {
        return Err("TriadSm90a PTX contains a numeric atomic or reduction instruction".into());
    }
    Ok(())
}

struct SourceFragment {
    logical_name: &'static str,
    source: &'static str,
    allowed_quoted_includes: &'static [&'static str],
}

const TYPED_PRELUDE: SourceFragment = SourceFragment {
    logical_name: "kernels/_typed_prelude.cuh",
    source: include_str!("../../../../kernels/_typed_prelude.cuh"),
    allowed_quoted_includes: &[],
};

const FIXED_SOURCE_FRAGMENTS: &[SourceFragment] = &[
    TYPED_PRELUDE,
    SourceFragment {
        logical_name: "kernels/mamba_ssm.cu",
        source: include_str!("../../../../kernels/mamba_ssm.cu"),
        allowed_quoted_includes: &["_typed_prelude.cuh"],
    },
    SourceFragment {
        logical_name: "kernels/mamba_ssm_parallel.cu",
        source: include_str!("../../../../kernels/mamba_ssm_parallel.cu"),
        allowed_quoted_includes: &["_typed_prelude.cuh"],
    },
    SourceFragment {
        logical_name: "kernels/conv1d.cu",
        source: include_str!("../../../../kernels/conv1d.cu"),
        allowed_quoted_includes: &["_typed_prelude.cuh"],
    },
    SourceFragment {
        logical_name: "kernels/activations.cu",
        source: include_str!("../../../../kernels/activations.cu"),
        allowed_quoted_includes: &["_typed_prelude.cuh"],
    },
    SourceFragment {
        logical_name: "kernels/norms.cu",
        source: include_str!("../../../../kernels/norms.cu"),
        allowed_quoted_includes: &["_typed_prelude.cuh"],
    },
    SourceFragment {
        logical_name: "kernels/elementwise.cu",
        source: include_str!("../../../../kernels/elementwise.cu"),
        allowed_quoted_includes: &["_typed_prelude.cuh"],
    },
    SourceFragment {
        logical_name: "kernels/loss_scaler.cu",
        source: include_str!("../../../../kernels/loss_scaler.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/grad_clip.cu",
        source: include_str!("../../../../kernels/grad_clip.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/adamw.cu",
        source: include_str!("../../../../kernels/adamw.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_fixed/common.cuh",
        source: include_str!("../../../../kernels/gemm_bi_fixed/common.cuh"),
        allowed_quoted_includes: &["_typed_prelude.cuh"],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_fixed/ffma.cuh",
        source: include_str!("../../../../kernels/gemm_bi_fixed/ffma.cuh"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_fixed/wmma_legacy.cuh",
        source: include_str!("../../../../kernels/gemm_bi_fixed/wmma_legacy.cuh"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_fixed/matvec.cuh",
        source: include_str!("../../../../kernels/gemm_bi_fixed/matvec.cuh"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_fixed/mma16.cuh",
        source: include_str!("../../../../kernels/gemm_bi_fixed/mma16.cuh"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_fixed/sm90_wgmma.cuh",
        source: include_str!("../../../../kernels/gemm_bi_fixed/sm90_wgmma.cuh"),
        allowed_quoted_includes: &[],
    },
];

const TRIAD_CONTRACT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_triad/contract.cuh",
    source: include_str!("../../../../kernels/gemm_bi_triad/contract.cuh"),
    allowed_quoted_includes: &[],
};

const TRIAD_COMMON: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_triad/common.cuh",
    source: include_str!("../../../../kernels/gemm_bi_triad/common.cuh"),
    allowed_quoted_includes: &[],
};

const TRIAD_EPILOGUE: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_triad/epilogue.cuh",
    source: include_str!("../../../../kernels/gemm_bi_triad/epilogue.cuh"),
    allowed_quoted_includes: &[],
};

const SCALAR_SOURCE_FRAGMENTS: &[SourceFragment] = &[
    TYPED_PRELUDE,
    TRIAD_CONTRACT,
    TRIAD_COMMON,
    TRIAD_EPILOGUE,
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/scalar.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/scalar.cu"),
        allowed_quoted_includes: &[],
    },
];

const SM80_SOURCE_FRAGMENTS: &[SourceFragment] = &[
    TYPED_PRELUDE,
    TRIAD_CONTRACT,
    TRIAD_COMMON,
    TRIAD_EPILOGUE,
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/mma16.cuh",
        source: include_str!("../../../../kernels/gemm_bi_triad/mma16.cuh"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/sm80.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/sm80.cu"),
        allowed_quoted_includes: &[],
    },
];

const SM90A_SOURCE_FRAGMENTS: &[SourceFragment] = &[
    TYPED_PRELUDE,
    TRIAD_CONTRACT,
    TRIAD_COMMON,
    TRIAD_EPILOGUE,
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/sm90a.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/sm90a.cu"),
        allowed_quoted_includes: &[],
    },
];

pub(super) const SCALAR_SYMBOLS: &[&str] = &[
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

pub(super) const SM80_SYMBOLS: &[&str] = &[
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

pub const SM90A_SYMBOLS: &[&str] = &[
    "sgemm_bi_nn_sm90a_wgmma_wg1_bf16",
    "sgemm_bi_nn_sm90a_wgmma_wg1_f16",
    "sgemm_bi_tn_sm90a_wgmma_wg1_bf16",
    "sgemm_bi_tn_sm90a_wgmma_wg1_f16",
    "sgemm_bi_nt_sm90a_wgmma_wg1_bf16",
    "sgemm_bi_nt_sm90a_wgmma_wg1_f16",
    "sgemm_bi_nn_sm90a_wgmma_wg2_bf16",
    "sgemm_bi_nn_sm90a_wgmma_wg2_f16",
    "sgemm_bi_tn_sm90a_wgmma_wg2_bf16",
    "sgemm_bi_tn_sm90a_wgmma_wg2_f16",
    "sgemm_bi_nt_sm90a_wgmma_wg2_bf16",
    "sgemm_bi_nt_sm90a_wgmma_wg2_f16",
];

fn module_fragments(kind: ModuleKind) -> Result<&'static [SourceFragment], String> {
    match kind {
        ModuleKind::Fixed => Ok(FIXED_SOURCE_FRAGMENTS),
        ModuleKind::TriadScalar => Ok(SCALAR_SOURCE_FRAGMENTS),
        ModuleKind::TriadSm80 => Ok(SM80_SOURCE_FRAGMENTS),
        ModuleKind::TriadSm90a => Ok(SM90A_SOURCE_FRAGMENTS),
        _ => Err(format!("no source fragments for {kind:?}")),
    }
}

fn compose_module_source(kind: ModuleKind) -> Result<String, String> {
    compose_fragments(module_fragments(kind)?)
}

fn compose_fragments(fragments: &[SourceFragment]) -> Result<String, String> {
    let mut composed = String::new();
    for fragment in fragments {
        validate_fragment(fragment)?;
        composed.push_str("#line 1 \"");
        composed.push_str(fragment.logical_name);
        composed.push_str("\"\n");
        append_without_local_includes(&mut composed, fragment)?;
        if !composed.ends_with('\n') {
            composed.push('\n');
        }
    }
    Ok(composed)
}

fn validate_fragment(fragment: &SourceFragment) -> Result<(), String> {
    if fragment.logical_name.contains('"') || fragment.logical_name.contains('\n') {
        return Err(format!(
            "invalid logical source name {}",
            fragment.logical_name
        ));
    }
    if fragment.logical_name.starts_with("kernels/gemm_bi_triad/")
        && fragment.logical_name.ends_with(".cuh")
        && fragment.source.contains("extern \"C\" __global__")
    {
        return Err(format!(
            "triad header {} exports a global kernel",
            fragment.logical_name
        ));
    }
    Ok(())
}

fn append_without_local_includes(
    output: &mut String,
    fragment: &SourceFragment,
) -> Result<(), String> {
    let mut physical = fragment.source.split_inclusive('\n').peekable();
    while let Some(first) = physical.next() {
        let mut raw = first.to_owned();
        let mut logical = first.strip_suffix('\n').unwrap_or(first).to_owned();
        while logical.ends_with('\\') {
            logical.pop();
            let Some(next) = physical.next() else {
                return Err(format!(
                    "{} ends inside a preprocessor continuation",
                    fragment.logical_name
                ));
            };
            raw.push_str(next);
            logical.push_str(next.strip_suffix('\n').unwrap_or(next));
        }

        match include_directive(&logical).map_err(|reason| {
            format!(
                "{} contains invalid include directive: {reason}",
                fragment.logical_name
            )
        })? {
            Some(IncludeDirective::Quoted(target)) => {
                if target.ends_with(".cu") {
                    return Err(format!(
                        "{} includes forbidden CUDA source {target}",
                        fragment.logical_name
                    ));
                }
                if !fragment.allowed_quoted_includes.contains(&target.as_str()) {
                    return Err(format!(
                        "{} contains unlisted quoted include {target}",
                        fragment.logical_name
                    ));
                }
            }
            Some(IncludeDirective::Angle(target)) => {
                if target.ends_with(".cu") {
                    return Err(format!(
                        "{} includes forbidden CUDA source {target}",
                        fragment.logical_name
                    ));
                }
                output.push_str(&raw);
            }
            None => output.push_str(&raw),
        }
    }
    Ok(())
}

enum IncludeDirective {
    Quoted(String),
    Angle(String),
}

fn include_directive(logical_line: &str) -> Result<Option<IncludeDirective>, &'static str> {
    let bytes = logical_line.as_bytes();
    let Some(mut cursor) = skip_space_and_comments(bytes, 0) else {
        return Ok(None);
    };
    if bytes.get(cursor) == Some(&b'#') {
        cursor += 1;
    } else if bytes.get(cursor..cursor + 2) == Some(b"%:") {
        cursor += 2;
    } else {
        return Ok(None);
    }
    cursor = skip_space_and_comments(bytes, cursor).ok_or("unterminated comment after '#'")?;
    let start = cursor;
    while bytes.get(cursor).is_some_and(u8::is_ascii_alphabetic) {
        cursor += 1;
    }
    let directive = &bytes[start..cursor];
    if directive == b"import" {
        return Err("include-like import directive is forbidden");
    }
    if directive != b"include" {
        return Ok(None);
    }
    cursor = skip_space_and_comments(bytes, cursor).ok_or("unterminated comment after include")?;
    match bytes.get(cursor) {
        Some(b'"') => {
            cursor += 1;
            let target_start = cursor;
            while let Some(byte) = bytes.get(cursor) {
                match byte {
                    b'"' => {
                        if !include_trailing_is_trivia(bytes, cursor + 1) {
                            return Err("trailing tokens after quoted include target");
                        }
                        let target = String::from_utf8(bytes[target_start..cursor].to_vec())
                            .map_err(|_| "quoted include target is not UTF-8")?;
                        return Ok(Some(IncludeDirective::Quoted(target)));
                    }
                    b'\\' => return Err("escaped quoted include target"),
                    _ => cursor += 1,
                }
            }
            Err("unterminated quoted include target")
        }
        Some(b'<') => {
            cursor += 1;
            let target_start = cursor;
            while let Some(byte) = bytes.get(cursor) {
                if *byte == b'>' {
                    if !include_trailing_is_trivia(bytes, cursor + 1) {
                        return Err("trailing tokens after angle include target");
                    }
                    let target = String::from_utf8(bytes[target_start..cursor].to_vec())
                        .map_err(|_| "angle include target is not UTF-8")?;
                    return Ok(Some(IncludeDirective::Angle(target)));
                }
                cursor += 1;
            }
            Err("unterminated angle include target")
        }
        Some(_) => Err("macro include target"),
        None => Err("missing include target"),
    }
}

fn include_trailing_is_trivia(bytes: &[u8], mut cursor: usize) -> bool {
    loop {
        while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        if cursor == bytes.len() || bytes.get(cursor..cursor + 2) == Some(b"//") {
            return true;
        }
        if bytes.get(cursor..cursor + 2) != Some(b"/*") {
            return false;
        }
        let Some(remainder) = bytes.get(cursor + 2..) else {
            return false;
        };
        let Some(end) = remainder.windows(2).position(|window| window == b"*/") else {
            return false;
        };
        cursor += end + 4;
    }
}

fn skip_space_and_comments(bytes: &[u8], mut cursor: usize) -> Option<usize> {
    loop {
        while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        if bytes.get(cursor..cursor + 2) != Some(b"/*") {
            return Some(cursor);
        }
        let remainder = bytes.get(cursor + 2..)?;
        let end = remainder.windows(2).position(|window| window == b"*/")?;
        cursor += end + 4;
    }
}

fn resolve_owned_symbol<T>(
    symbol: &str,
    scalar_get: impl FnOnce(&str) -> Result<T, String>,
    sm80_get: impl FnOnce(&str) -> Result<T, String>,
) -> Result<T, String> {
    if SCALAR_SYMBOLS.contains(&symbol) {
        scalar_get(symbol).map_err(|error| format!("TriadScalar symbol {symbol}: {error}"))
    } else if SM80_SYMBOLS.contains(&symbol) {
        sm80_get(symbol).map_err(|error| format!("TriadSm80 symbol {symbol}: {error}"))
    } else {
        Err(format!("no triad module owns symbol {symbol}"))
    }
}

/// Loaded deterministic training GEMM triad.
///
/// Scalar and Tensor Core handles remain in separate CUDA modules. Keeping the
/// modules and their identities beside the handles makes a route impossible to
/// outlive, or be replayed against, a different compiled artifact.
type Sm90aMapCacheKey = (
    [super::contract::Sm90aTensorMapKey; 2],
    [super::contract::Sm90aAllocationIdentity; 2],
    super::contract::Sm90aOp,
    u8,
    super::contract::Sm90aShape,
);
type Sm90aMapCache = Mutex<HashMap<Sm90aMapCacheKey, super::contract::Sm90aPreparedTensorMaps>>;

pub struct GemmBiKernels {
    _modules: CudaModuleAnchors,
    context_handle: usize,
    scalar_compiler_identity: CompilerIdentity,
    sm80_compiler_identity: CompilerIdentity,
    sm90a_compiler_identity: Option<CompilerIdentity>,
    artifact_set_identity: crate::mamba_ssm::gpu::kernel_identity::ArtifactSetIdentity,
    sm90a_functions: HashMap<&'static str, CudaFunction>,
    sm90a_tensor_maps: Sm90aMapCache,

    pub sgemm_nn: CudaFunction,
    pub sgemm_tn: CudaFunction,
    pub sgemm_nt: CudaFunction,
    pub sgemm_nn_slim: CudaFunction,
    pub sgemm_tn_slim: CudaFunction,
    pub sgemm_nt_slim: CudaFunction,
    pub sgemm_nn_ultra_thin: CudaFunction,
    pub sgemm_nn_gemv: CudaFunction,
    pub sgemm_tn_gemv: CudaFunction,
    pub sgemm_nt_gemv: CudaFunction,
    pub sgemm_nn_narrow: CudaFunction,
    pub sgemm_nn_narrow_small: CudaFunction,
    pub sgemm_tn_narrow: CudaFunction,
    pub sgemm_tn_narrow_splitm_partial: CudaFunction,
    pub sgemm_nt_narrow: CudaFunction,
    pub sgemm_nn_splitk32_partial: CudaFunction,
    pub sgemm_splitk_reduce: CudaFunction,
    pub sgemm_tn_splitm_partial: CudaFunction,
    pub sgemm_splitm_reduce: CudaFunction,
    pub sgemm_nn_splitk_big_partial: CudaFunction,
    pub sgemm_nt_splitn_big_partial: CudaFunction,
    pub sgemm_nn_splitk_slim_partial: CudaFunction,
    pub sgemm_transpose_f32_2d: CudaFunction,
    pub sgemm_dx_col_gemv: CudaFunction,

    pub sgemm_nn_gemv_typed: HalfKernel,
    pub sgemm_tn_gemv_typed: HalfKernel,
    pub sgemm_nt_gemv_typed: HalfKernel,
    pub sgemm_nn_ultra_thin_typed: HalfKernel,
    pub sgemm_nn_narrow_typed: HalfKernel,
    pub sgemm_nn_narrow_small_typed: HalfKernel,
    pub sgemm_tn_narrow_typed: HalfKernel,
    pub sgemm_nt_narrow_typed: HalfKernel,
    pub sgemm_nn_big_typed: HalfKernel,
    pub sgemm_tn_big_typed: HalfKernel,
    pub sgemm_nt_big_typed: HalfKernel,
    pub sgemm_nn_tc_typed: HalfKernel,
    pub sgemm_tn_tc_typed: HalfKernel,
    pub sgemm_nt_tc_typed: HalfKernel,
    pub sgemm_nn_tc64_typed: HalfKernel,
    pub sgemm_nn_tc16_typed: HalfKernel,
    pub sgemm_tn_tc64_typed: HalfKernel,
    pub sgemm_nt_tc64_typed: HalfKernel,

    splitk_scratch: std::sync::OnceLock<CudaSlice<f32>>,
    transpose_scratch: std::sync::OnceLock<CudaSlice<f32>>,
}

impl GemmBiKernels {
    pub(crate) fn load(
        context_handle: usize,
        fixed_artifact: ArtifactIdentity,
        scalar: CompiledModule,
        sm80: CompiledModule,
        sm90a: Option<CompiledModule>,
    ) -> Result<Self, String> {
        let sm90a = sm90a.and_then(|module| {
            load_sm90a_functions(&module)
                .ok()
                .map(|functions| (module, functions))
        });
        let mut artifacts = vec![
            fixed_artifact,
            scalar.artifact_identity,
            sm80.artifact_identity,
        ];
        if let Some((module, _)) = sm90a.as_ref() {
            artifacts.push(module.artifact_identity);
        }
        let artifact_set_identity =
            crate::mamba_ssm::gpu::kernel_identity::build_artifact_set(&artifacts)?;
        let load = |name: &str| load_owned_function(name, &scalar.module, &sm80.module);
        let load_half = |base: &str| load_owned_half(base, &scalar.module, &sm80.module);
        let load_half_dynsmem = |base: &str, bytes: i32| {
            let kernel = load_half(base)?;
            set_half_dynamic_shared(&kernel, base, bytes)?;
            Ok::<HalfKernel, String>(kernel)
        };

        let sgemm_nn = load("sgemm_bi_nn")?;
        set_dynamic_shared(&sgemm_nn, "sgemm_bi_nn", 34 * 1024)?;
        let sgemm_tn = load("sgemm_bi_tn")?;
        set_dynamic_shared(&sgemm_tn, "sgemm_bi_tn", 34 * 1024)?;
        let sgemm_nt = load("sgemm_bi_nt")?;
        set_dynamic_shared(&sgemm_nt, "sgemm_bi_nt", 34 * 1024)?;
        let sgemm_nn_splitk_big_partial = load("sgemm_bi_nn_splitk_big_partial")?;
        set_dynamic_shared(
            &sgemm_nn_splitk_big_partial,
            "sgemm_bi_nn_splitk_big_partial",
            34 * 1024,
        )?;
        let sgemm_nt_splitn_big_partial = load("sgemm_bi_nt_splitn_big_partial")?;
        set_dynamic_shared(
            &sgemm_nt_splitn_big_partial,
            "sgemm_bi_nt_splitn_big_partial",
            34 * 1024,
        )?;

        let sm90a_functions = sm90a
            .as_ref()
            .map(|(_, functions)| functions.clone())
            .unwrap_or_default();
        let mut anchors = vec![scalar.module.clone(), sm80.module.clone()];
        if let Some((module, _)) = sm90a.as_ref() {
            anchors.push(module.module.clone());
        }

        Ok(Self {
            _modules: CudaModuleAnchors::new(anchors),
            context_handle,
            scalar_compiler_identity: scalar.compiler_identity,
            sm80_compiler_identity: sm80.compiler_identity,
            sm90a_compiler_identity: sm90a.as_ref().map(|(module, _)| module.compiler_identity),
            artifact_set_identity,
            sm90a_functions,
            sm90a_tensor_maps: Mutex::new(HashMap::new()),
            sgemm_nn,
            sgemm_tn,
            sgemm_nt,
            sgemm_nn_slim: load("sgemm_bi_nn_slim")?,
            sgemm_tn_slim: load("sgemm_bi_tn_slim")?,
            sgemm_nt_slim: load("sgemm_bi_nt_slim")?,
            sgemm_nn_ultra_thin: load("sgemm_bi_nn_ultra_thin")?,
            sgemm_nn_gemv: load("sgemm_bi_nn_gemv")?,
            sgemm_tn_gemv: load("sgemm_bi_tn_gemv")?,
            sgemm_nt_gemv: load("sgemm_bi_nt_gemv")?,
            sgemm_nn_narrow: load("sgemm_bi_nn_narrow")?,
            sgemm_nn_narrow_small: load("sgemm_bi_nn_narrow_small")?,
            sgemm_tn_narrow: load("sgemm_bi_tn_narrow")?,
            sgemm_tn_narrow_splitm_partial: load("sgemm_bi_tn_narrow_splitm_partial")?,
            sgemm_nt_narrow: load("sgemm_bi_nt_narrow")?,
            sgemm_nn_splitk32_partial: load("sgemm_bi_nn_splitk32_partial")?,
            sgemm_splitk_reduce: load("sgemm_bi_splitk_reduce")?,
            sgemm_tn_splitm_partial: load("sgemm_bi_tn_splitm_partial")?,
            sgemm_splitm_reduce: load("sgemm_bi_splitm_reduce")?,
            sgemm_nn_splitk_big_partial,
            sgemm_nt_splitn_big_partial,
            sgemm_nn_splitk_slim_partial: load("sgemm_bi_nn_splitk_slim_partial")?,
            sgemm_transpose_f32_2d: load("sgemm_transpose_f32_2d")?,
            sgemm_dx_col_gemv: load("sgemm_bi_dx_col_gemv")?,
            sgemm_nn_gemv_typed: load_half("sgemm_bi_nn_gemv")?,
            sgemm_tn_gemv_typed: load_half("sgemm_bi_tn_gemv")?,
            sgemm_nt_gemv_typed: load_half("sgemm_bi_nt_gemv")?,
            sgemm_nn_ultra_thin_typed: load_half("sgemm_bi_nn_ultra_thin")?,
            sgemm_nn_narrow_typed: load_half("sgemm_bi_nn_narrow")?,
            sgemm_nn_narrow_small_typed: load_half("sgemm_bi_nn_narrow_small")?,
            sgemm_tn_narrow_typed: load_half("sgemm_bi_tn_narrow")?,
            sgemm_nt_narrow_typed: load_half("sgemm_bi_nt_narrow")?,
            sgemm_nn_big_typed: load_half_dynsmem("sgemm_bi_nn_big", 34 * 1024)?,
            sgemm_tn_big_typed: load_half_dynsmem("sgemm_bi_tn_big", 34 * 1024)?,
            sgemm_nt_big_typed: load_half_dynsmem("sgemm_bi_nt_big", 34 * 1024)?,
            sgemm_nn_tc_typed: load_half_dynsmem("sgemm_bi_nn_tc", 75_776)?,
            sgemm_tn_tc_typed: load_half_dynsmem("sgemm_bi_tn_tc", 75_776)?,
            sgemm_nt_tc_typed: load_half_dynsmem("sgemm_bi_nt_tc", 75_776)?,
            sgemm_nn_tc64_typed: load_half("sgemm_bi_nn_tc64")?,
            sgemm_nn_tc16_typed: load_half("sgemm_bi_nn_tc16")?,
            sgemm_tn_tc64_typed: load_half("sgemm_bi_tn_tc64")?,
            sgemm_nt_tc64_typed: load_half("sgemm_bi_nt_tc64")?,
            splitk_scratch: std::sync::OnceLock::new(),
            transpose_scratch: std::sync::OnceLock::new(),
        })
    }

    pub fn artifact_set_identity(
        &self,
    ) -> crate::mamba_ssm::gpu::kernel_identity::ArtifactSetIdentity {
        self.artifact_set_identity
    }

    pub fn scalar_compiler_identity(&self) -> CompilerIdentity {
        self.scalar_compiler_identity
    }

    pub fn sm80_compiler_identity(&self) -> CompilerIdentity {
        self.sm80_compiler_identity
    }

    pub fn sm90a_compiler_identity(&self) -> Option<CompilerIdentity> {
        self.sm90a_compiler_identity
    }

    pub fn has_sm90a_wgmma(&self) -> bool {
        self.sm90a_functions.len() == SM90A_SYMBOLS.len()
    }

    pub(super) fn context_handle(&self) -> usize {
        self.context_handle
    }

    pub(super) fn sm90a_function(&self, symbol: &str) -> Option<&CudaFunction> {
        self.sm90a_functions.get(symbol)
    }

    pub(super) fn prepare_sm90a_tensor_maps(
        &self,
        request: super::contract::Sm90aMapRequest,
        keys: [super::contract::Sm90aTensorMapKey; 2],
        allocations: [super::contract::Sm90aAllocationIdentity; 2],
        capturing: bool,
        binding: super::contract::Sm90aMapBinding,
    ) -> Result<super::contract::Sm90aPreparedTensorMaps, String> {
        if binding.context_handle != self.context_handle
            || Some(binding.compiler) != self.sm90a_compiler_identity
            || Some(binding.artifact) != self.artifact_set_identity.specialized
        {
            return Err("SM90a tensor-map binding does not match its CUDA module context".into());
        }
        let mut cache = self
            .sm90a_tensor_maps
            .lock()
            .map_err(|_| "SM90a tensor-map cache is poisoned".to_string())?;
        let dtype = match request.dtype {
            super::super::dtype::WeightDtype::F32 => 0,
            super::super::dtype::WeightDtype::F16 => 1,
            super::super::dtype::WeightDtype::Bf16 => 2,
        };
        let cache_key = (keys, allocations, request.op, dtype, request.shape);
        cache.retain(|(cached_keys, cached_allocations, _, _, _), _| {
            *cached_keys != keys || *cached_allocations == allocations
        });
        if let Some(maps) = cache.get(&cache_key) {
            return Ok(*maps);
        }
        if capturing {
            return Err("SM90a tensor-map cache miss during graph capture".into());
        }
        let maps = super::contract::encode_sm90a_tensor_maps(keys, request, binding, allocations)?;
        cache.insert(cache_key, maps);
        Ok(maps)
    }

    pub fn splitk_scratch_buf(&self, stream: &Arc<CudaStream>) -> Result<&CudaSlice<f32>, String> {
        if self.splitk_scratch.get().is_none() {
            let buffer = stream
                .alloc_zeros::<f32>(1 << 23)
                .map_err(|error| format!("splitk_scratch alloc: {error:?}"))?;
            let _ = self.splitk_scratch.set(buffer);
        }
        self.splitk_scratch
            .get()
            .ok_or_else(|| "splitk_scratch cell empty after init".to_string())
    }

    pub fn transpose_scratch_buf(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<&CudaSlice<f32>, String> {
        if self.transpose_scratch.get().is_none() {
            let buffer = stream
                .alloc_zeros::<f32>(1 << 22)
                .map_err(|error| format!("transpose_scratch alloc: {error:?}"))?;
            let _ = self.transpose_scratch.set(buffer);
        }
        self.transpose_scratch
            .get()
            .ok_or_else(|| "transpose_scratch cell empty after init".to_string())
    }
}

fn load_owned_function(
    name: &str,
    scalar: &Arc<CudaModule>,
    sm80: &Arc<CudaModule>,
) -> Result<CudaFunction, String> {
    resolve_owned_symbol(
        name,
        |symbol| load_function(scalar, ModuleKind::TriadScalar, symbol),
        |symbol| load_function(sm80, ModuleKind::TriadSm80, symbol),
    )
}

fn load_function(
    module: &Arc<CudaModule>,
    kind: ModuleKind,
    name: &str,
) -> Result<CudaFunction, String> {
    module
        .load_function(name)
        .map_err(|error| format!("{kind:?} kernel {name} not found: {error:?}"))
}

fn load_sm90a_functions(
    module: &CompiledModule,
) -> Result<HashMap<&'static str, CudaFunction>, String> {
    if module.artifact_identity.module_kind != ModuleKind::TriadSm90a
        || module.compiler_identity.target.as_str() != "sm_90a"
    {
        return Err("specialized triad module is not exact-target TriadSm90a".into());
    }
    let mut functions = HashMap::new();
    for &symbol in SM90A_SYMBOLS {
        let function = load_function(&module.module, ModuleKind::TriadSm90a, symbol)?;
        set_dynamic_shared(
            &function,
            symbol,
            super::contract::SM90A_DYNAMIC_SHARED_BYTES as i32,
        )?;
        if function
            .local_size_bytes()
            .map_err(|error| format!("query {symbol} local memory: {error:?}"))?
            != 0
        {
            return Err(format!("{symbol} spills to local memory"));
        }
        let wg2 = symbol.contains("_wg2_");
        let threads = if wg2 { 256 } else { 128 };
        if function
            .max_threads_per_block()
            .map_err(|error| format!("query {symbol} max threads: {error:?}"))?
            < threads
        {
            return Err(format!("{symbol} cannot launch {threads} threads"));
        }
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(
                threads as u32,
                super::contract::SM90A_DYNAMIC_SHARED_BYTES as usize,
                None,
            )
            .map_err(|error| format!("query {symbol} occupancy: {error:?}"))?;
        if occupancy < if wg2 { 1 } else { 3 } {
            return Err(format!(
                "{symbol} occupancy {occupancy} misses its schedule gate"
            ));
        }
        if !wg2
            && function
                .occupancy_available_dynamic_smem_per_block(3, 128)
                .map_err(|error| format!("query {symbol} shared-memory capacity: {error:?}"))?
                < super::contract::SM90A_DYNAMIC_SHARED_BYTES as usize
        {
            return Err(format!("{symbol} cannot sustain three 73984-byte CTAs"));
        }
        functions.insert(symbol, function);
    }
    Ok(functions)
}

fn load_owned_half(
    base: &str,
    scalar: &Arc<CudaModule>,
    sm80: &Arc<CudaModule>,
) -> Result<HalfKernel, String> {
    Ok(HalfKernel {
        bf16: load_owned_function(&format!("{base}_bf16"), scalar, sm80)?,
        f16: load_owned_function(&format!("{base}_f16"), scalar, sm80)?,
    })
}

fn set_dynamic_shared(function: &CudaFunction, name: &str, bytes: i32) -> Result<(), String> {
    function
        .set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes,
        )
        .map_err(|error| format!("set MAX_DYNAMIC_SHARED for {name}: {error:?}"))
}

fn set_half_dynamic_shared(kernel: &HalfKernel, name: &str, bytes: i32) -> Result<(), String> {
    set_dynamic_shared(&kernel.bf16, name, bytes)?;
    set_dynamic_shared(&kernel.f16, name, bytes)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::mamba_ssm::gpu::kernel_identity::ModuleKind;

    use super::{
        SCALAR_SYMBOLS as PRODUCTION_SCALAR_SYMBOLS, SM80_SYMBOLS as PRODUCTION_SM80_SYMBOLS,
        SM90A_SYMBOLS, SourceFragment, compose_fragments, compose_module_source,
        resolve_owned_symbol, validate_module_target,
    };

    const FIXED_FRAGMENTS: &[&str] = &[
        "kernels/_typed_prelude.cuh",
        "kernels/mamba_ssm.cu",
        "kernels/mamba_ssm_parallel.cu",
        "kernels/conv1d.cu",
        "kernels/activations.cu",
        "kernels/norms.cu",
        "kernels/elementwise.cu",
        "kernels/loss_scaler.cu",
        "kernels/grad_clip.cu",
        "kernels/adamw.cu",
        "kernels/gemm_bi_fixed/common.cuh",
        "kernels/gemm_bi_fixed/ffma.cuh",
        "kernels/gemm_bi_fixed/wmma_legacy.cuh",
        "kernels/gemm_bi_fixed/matvec.cuh",
        "kernels/gemm_bi_fixed/mma16.cuh",
        "kernels/gemm_bi_fixed/sm90_wgmma.cuh",
    ];

    const SCALAR_FRAGMENTS: &[&str] = &[
        "kernels/_typed_prelude.cuh",
        "kernels/gemm_bi_triad/contract.cuh",
        "kernels/gemm_bi_triad/common.cuh",
        "kernels/gemm_bi_triad/epilogue.cuh",
        "kernels/gemm_bi_triad/scalar.cu",
    ];

    const SM80_FRAGMENTS: &[&str] = &[
        "kernels/_typed_prelude.cuh",
        "kernels/gemm_bi_triad/contract.cuh",
        "kernels/gemm_bi_triad/common.cuh",
        "kernels/gemm_bi_triad/epilogue.cuh",
        "kernels/gemm_bi_triad/mma16.cuh",
        "kernels/gemm_bi_triad/sm80.cu",
    ];

    const SM90A_FRAGMENTS: &[&str] = &[
        "kernels/_typed_prelude.cuh",
        "kernels/gemm_bi_triad/contract.cuh",
        "kernels/gemm_bi_triad/common.cuh",
        "kernels/gemm_bi_triad/epilogue.cuh",
        "kernels/gemm_bi_triad/sm90a.cu",
    ];

    const SCALAR_SYMBOLS: &[&str] = &[
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

    const SM80_SYMBOLS: &[&str] = &[
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

    fn assert_composition(kind: ModuleKind, expected_names: &[&str]) {
        let first = compose_module_source(kind).unwrap();
        let second = compose_module_source(kind).unwrap();
        assert_eq!(first, second, "{kind:?} composition changed between calls");
        assert!(first.ends_with('\n'), "{kind:?} has no terminal newline");
        assert!(!first.contains(env!("CARGO_MANIFEST_DIR")));
        let boundaries: Vec<_> = first
            .lines()
            .filter_map(|line| line.strip_prefix("#line 1 \"")?.strip_suffix('"'))
            .collect();
        assert_eq!(boundaries, expected_names, "{kind:?} source boundaries");
    }

    #[test]
    fn module_sources_have_exact_deterministic_boundaries() {
        assert_composition(ModuleKind::Fixed, FIXED_FRAGMENTS);
        assert_composition(ModuleKind::TriadScalar, SCALAR_FRAGMENTS);
        assert_composition(ModuleKind::TriadSm80, SM80_FRAGMENTS);
        assert_composition(ModuleKind::TriadSm90a, SM90A_FRAGMENTS);

        assert!(
            !std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("kernels/gemm_bi_triad.cu")
                .exists(),
            "obsolete root triad monolith must not return"
        );
        assert!(
            !compose_module_source(ModuleKind::Fixed)
                .unwrap()
                .contains("sgemm_bi_")
        );
    }

    #[test]
    fn quoted_include_allowlist_is_exact_and_fail_closed() {
        let accepted = compose_fragments(&[SourceFragment {
            logical_name: "kernels/synthetic.cu",
            source: "#include \"common.cuh\"\n#include <cuda_fp16.h>\nint value;\n",
            allowed_quoted_includes: &["common.cuh"],
        }])
        .unwrap();
        assert_eq!(
            accepted,
            "#line 1 \"kernels/synthetic.cu\"\n#include <cuda_fp16.h>\nint value;\n"
        );

        for source in [
            "#include \"forbidden.cuh\"\n",
            "# include \"forbidden.cuh\"\n",
            "#/**/include/**/\"forbidden.cuh\"\n",
            "%:include \"forbidden.cuh\"\n",
            "#inc\\\nlude \"forbidden.cuh\"\n",
        ] {
            let error = compose_fragments(&[SourceFragment {
                logical_name: "kernels/synthetic.cu",
                source,
                allowed_quoted_includes: &["common.cuh"],
            }])
            .expect_err("unlisted quoted include must fail closed");
            assert!(error.contains("kernels/synthetic.cu"), "{error}");
            assert!(error.contains("forbidden.cuh"), "{error}");
        }

        let error = compose_fragments(&[SourceFragment {
            logical_name: "kernels/synthetic.cu",
            source: "#include \"nested.cu\"\n",
            allowed_quoted_includes: &["nested.cu"],
        }])
        .expect_err("a CUDA source must not include another CUDA source");
        assert!(error.contains("nested.cu"), "{error}");

        for source in [
            "#define HEADER \"forbidden.cuh\"\n#include HEADER\n",
            "#include <nested.cu>\n",
            "#include \"unterminated.cuh\n",
            "#include \"escaped\\\".cuh\"\n",
            "#import \"forbidden.cuh\"\n",
            "%:import \"forbidden.cuh\"\n",
            "#include_next \"forbidden.cuh\"\n",
            "%:include_next \"forbidden.cuh\"\n",
        ] {
            let error = compose_fragments(&[SourceFragment {
                logical_name: "kernels/synthetic.cu",
                source,
                allowed_quoted_includes: &[],
            }])
            .expect_err("an unclassifiable or CUDA-source include must fail closed");
            assert!(error.contains("kernels/synthetic.cu"), "{error}");
            assert!(error.contains("include"), "{error}");
        }

        let error = compose_fragments(&[SourceFragment {
            logical_name: "kernels/synthetic.cu",
            source: "#include \"common.cuh\" garbage\n",
            allowed_quoted_includes: &["common.cuh"],
        }])
        .expect_err("trailing include tokens must fail closed");
        assert!(error.contains("include"), "{error}");

        let error = compose_fragments(&[SourceFragment {
            logical_name: "kernels/gemm_bi_triad/synthetic.cuh",
            source: "extern \"C\" __global__ void forbidden() {}\n",
            allowed_quoted_includes: &[],
        }])
        .expect_err("a triad header must not export a global kernel");
        assert!(error.contains("synthetic.cuh"), "{error}");

        let lookalikes = "// #include \"forbidden.cuh\"\n\
            const char* text = \"#include \\\"forbidden.cuh\\\"\";\n\
            #include <cuda_bf16.h>\n";
        let composed = compose_fragments(&[SourceFragment {
            logical_name: "kernels/lookalikes.cu",
            source: lookalikes,
            allowed_quoted_includes: &[],
        }])
        .unwrap();
        assert!(composed.contains("#include <cuda_bf16.h>"));
        assert!(composed.contains("const char* text"));
    }

    #[test]
    fn triad_symbol_inventories_are_exact_disjoint_and_owned() {
        assert_eq!(PRODUCTION_SCALAR_SYMBOLS, SCALAR_SYMBOLS);
        assert_eq!(PRODUCTION_SM80_SYMBOLS, SM80_SYMBOLS);

        let scalar: BTreeSet<_> = SCALAR_SYMBOLS.iter().copied().collect();
        let sm80: BTreeSet<_> = SM80_SYMBOLS.iter().copied().collect();
        assert_eq!(scalar.len(), 46);
        assert_eq!(sm80.len(), 14);
        assert_eq!(SM90A_SYMBOLS.len(), 12);
        assert!(scalar.is_disjoint(&sm80));
        assert!(SM90A_SYMBOLS.iter().all(|symbol| !scalar.contains(symbol)));
        assert!(SM90A_SYMBOLS.iter().all(|symbol| !sm80.contains(symbol)));
        assert_eq!(scalar.union(&sm80).count(), 60);
        assert!(
            scalar
                .union(&sm80)
                .copied()
                .all(|name| { name.starts_with("sgemm_bi_") || name == "sgemm_transpose_f32_2d" })
        );
    }

    #[test]
    fn sm90a_module_requires_the_exact_architecture_target() {
        validate_module_target(ModuleKind::TriadSm90a, "sm_90a").unwrap();
        for target in ["sm_80", "sm_89", "sm_90", "sm_100a", "sm_120"] {
            let error = validate_module_target(ModuleKind::TriadSm90a, target).unwrap_err();
            assert!(error.contains("exact target sm_90a"), "{error}");
        }
        validate_module_target(ModuleKind::TriadSm80, "sm_90a").unwrap();
    }

    #[test]
    fn owned_symbol_resolution_never_falls_back_to_fixed() {
        let scalar_calls = std::cell::Cell::new(0);
        let sm80_calls = std::cell::Cell::new(0);
        let value: usize = resolve_owned_symbol(
            "sgemm_bi_nn",
            |name| {
                scalar_calls.set(scalar_calls.get() + 1);
                Ok(name.len())
            },
            |_| panic!("scalar symbol consulted the SM80 module"),
        )
        .unwrap();
        assert_eq!(value, "sgemm_bi_nn".len());
        assert_eq!(scalar_calls.get(), 1);
        assert_eq!(sm80_calls.get(), 0);

        let missing: Result<(), String> = resolve_owned_symbol(
            "sgemm_bi_nn_tc_bf16",
            |_| panic!("SM80 symbol consulted the scalar module"),
            |name| {
                sm80_calls.set(sm80_calls.get() + 1);
                Err(format!("absent {name}"))
            },
        );
        let error = missing.expect_err("missing owned symbol must abort initialization");
        assert!(error.contains("TriadSm80"), "{error}");
        assert!(error.contains("sgemm_bi_nn_tc_bf16"), "{error}");
        assert_eq!(sm80_calls.get(), 1);
    }
}
