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

pub(crate) struct QualifiedSpecializedModule {
    module: CompiledModule,
    functions: HashMap<&'static str, CudaFunction>,
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
        && validate_specialized_ptx(request.module_kind, request.arch, &src).is_ok()
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
            validate_specialized_ptx(request.module_kind, request.arch, &ptx_source)?;
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

pub(crate) fn compile_sm100_optional(
    ctx: &Arc<CudaContext>,
    state_cap: usize,
    device_cc: (i32, i32),
) -> Option<QualifiedSpecializedModule> {
    select_sm100_candidate(
        super::dispatch::sm100_target_candidates(device_cc),
        |candidate| probe_sm100_target(ctx, candidate),
        |candidate| {
            compile_module(CompileModuleRequest {
                ctx,
                arch: candidate.nvrtc_arch,
                state_cap,
                module_kind: ModuleKind::TriadSm100,
            })
        },
        qualify_specialized_module,
    )
}

fn select_sm100_candidate<T, U>(
    candidates: &[super::contract::Sm100TargetCandidate],
    mut probe: impl FnMut(super::contract::Sm100TargetCandidate) -> Result<(), String>,
    mut compile: impl FnMut(super::contract::Sm100TargetCandidate) -> Result<T, String>,
    mut qualify: impl FnMut(T) -> Result<U, String>,
) -> Option<U> {
    for &candidate in candidates {
        if probe(candidate).is_err() {
            continue;
        }
        let Ok(compiled) = compile(candidate) else {
            continue;
        };
        if let Ok(qualified) = qualify(compiled) {
            return Some(qualified);
        }
    }
    None
}

const SM100_PROBE_SOURCE: &str = r#"
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

#if __CUDACC_VER_MAJOR__ >= 13
struct alignas(128) CUtensorMap {
#else
struct alignas(64) CUtensorMap {
#endif
    unsigned long long opaque[16];
};

static_assert(sizeof(CUtensorMap) == 128, "unexpected tensor map ABI");

static __device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

static __device__ __forceinline__ unsigned long long tcgen_desc(
    const void* p, unsigned lbo, unsigned sbo) {
    unsigned long long d =
        (static_cast<unsigned long long>(smem_addr(p)) >> 4) & 0x3fffULL;
    d |= static_cast<unsigned long long>(lbo & 0x3fffU) << 16;
    d |= static_cast<unsigned long long>(sbo & 0x3fffU) << 32;
    d |= 1ULL << 46;
    d |= 2ULL << 61;
    return d;
}

static __device__ __forceinline__ void wait_barrier(
    unsigned bar, unsigned phase) {
    unsigned ready;
    do {
        asm volatile(
            "{ .reg .pred p; "
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
            "p, [%1], %2; "
            "selp.b32 %0, 1, 0, p; }"
            : "=r"(ready) : "r"(bar), "r"(phase) : "memory");
    } while (!ready);
}

static __device__ __forceinline__ void issue_mma(
    unsigned tmem, unsigned long long a, unsigned long long b,
    unsigned idesc, unsigned accumulate) {
    unsigned zero = 0;
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %8, 0; "
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "
        "{%4, %5, %6, %7}, p; }"
        :: "r"(tmem), "l"(a), "l"(b), "r"(idesc),
           "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(accumulate)
        : "memory");
}

extern "C" __global__ void tcgen05_probe(
    unsigned* out,
    const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b) {
    extern __shared__ __align__(1024) unsigned char smem[];
    __shared__ unsigned tmem_base;
    __shared__ __align__(8) unsigned long long barriers[2];

    unsigned full = smem_addr(&barriers[0]);
    unsigned done = smem_addr(&barriers[1]);
    unsigned warp = threadIdx.x >> 5;

    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"(full) : "memory");
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"(done) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp == 0) {
        unsigned dst = smem_addr(&tmem_base);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
            "[%0], %1;"
            :: "r"(dst), "r"(128U) : "memory");
    }
    __syncthreads();

    unsigned values[8] = {
        threadIdx.x, threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3,
        threadIdx.x + 4, threadIdx.x + 5, threadIdx.x + 6, threadIdx.x + 7,
    };
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], "
        "{%1, %2, %3, %4, %5, %6, %7, %8};"
        :: "r"(tmem_base), "r"(values[0]), "r"(values[1]), "r"(values[2]),
           "r"(values[3]), "r"(values[4]), "r"(values[5]), "r"(values[6]),
           "r"(values[7]) : "memory");
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned dst = smem_addr(smem + 1024);
        unsigned bar = full;
        unsigned long long a_map =
            reinterpret_cast<unsigned long long>(&map_a);
        unsigned long long b_map =
            reinterpret_cast<unsigned long long>(&map_b);
        int x = 0;
        int y = 0;
        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
            "_, [%0], 32768;"
            :: "r"(bar) : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile."
            "mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :: "r"(dst), "l"(a_map), "r"(x), "r"(y), "r"(bar) : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile."
            "mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :: "r"(dst + 16384), "l"(b_map), "r"(x), "r"(y), "r"(bar)
            : "memory");

        wait_barrier(full, 0);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

        unsigned long long a_k = tcgen_desc(smem + 1024, 1, 64);
        unsigned long long a_mn = tcgen_desc(smem + 1024, 512, 64);
        unsigned long long b_k = tcgen_desc(smem + 17408, 1, 64);
        unsigned long long b_mn64 = tcgen_desc(smem + 17408, 0, 64);
        unsigned long long b_mn128 = tcgen_desc(smem + 17408, 512, 64);

        issue_mma(tmem_base, a_k, b_k, 0x08100010U, 1);
        issue_mma(tmem_base, a_k, b_mn64, 0x08110010U, 1);
        issue_mma(tmem_base, a_mn, b_mn64, 0x08118010U, 1);
        issue_mma(tmem_base, a_k, b_k, 0x08200010U, 1);
        issue_mma(tmem_base, a_k, b_mn128, 0x08210010U, 1);
        issue_mma(tmem_base, a_mn, b_mn128, 0x08218010U, 1);
        issue_mma(tmem_base, a_k, b_k, 0x08100490U, 1);
        issue_mma(tmem_base, a_k, b_mn64, 0x08110490U, 1);
        issue_mma(tmem_base, a_mn, b_mn64, 0x08118490U, 1);
        issue_mma(tmem_base, a_k, b_k, 0x08200490U, 1);
        issue_mma(tmem_base, a_k, b_mn128, 0x08210490U, 1);
        issue_mma(tmem_base, a_mn, b_mn128, 0x08218490U, 1);

        asm volatile(
            "tcgen05.commit.cta_group::1."
            "mbarrier::arrive::one.shared::cluster.b64 [%0];"
            :: "r"(done) : "memory");
        wait_barrier(done, 0);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    }
    __syncthreads();

    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]),
          "=r"(values[3]), "=r"(values[4]), "=r"(values[5]),
          "=r"(values[6]), "=r"(values[7])
        : "r"(tmem_base) : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    out[threadIdx.x] = values[0];
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();

    if (warp == 0) {
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
            ::: "memory");
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_base), "r"(128U) : "memory");
    }
}

#endif
"#;

fn probe_sm100_target(
    ctx: &Arc<CudaContext>,
    candidate: super::contract::Sm100TargetCandidate,
) -> Result<(), String> {
    let options = cudarc::nvrtc::CompileOptions {
        arch: Some(candidate.nvrtc_arch),
        options: vec!["--fmad=true".to_string(), "-DNDEBUG".to_string()],
        ..Default::default()
    };
    let image =
        cudarc::nvrtc::compile_ptx_with_opts(SM100_PROBE_SOURCE, options).map_err(|error| {
            format!(
                "TriadSm100 target probe {} failed: {}",
                candidate.nvrtc_arch,
                format!("{error:?}").replace("\\n", "\n")
            )
        })?;
    let bytes = image
        .as_bytes()
        .ok_or_else(|| "TriadSm100 target probe returned no PTX image".to_string())?;
    let ptx = crate::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(bytes)?;
    validate_sm100_probe_ptx(candidate.nvrtc_arch, &ptx)?;
    let module = ctx
        .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
        .map_err(|error| format!("TriadSm100 target probe load failed: {error:?}"))?;
    module
        .load_function("tcgen05_probe")
        .map_err(|error| format!("TriadSm100 target probe symbol failed: {error:?}"))?;
    Ok(())
}

fn validate_module_target(kind: ModuleKind, arch: &str) -> Result<(), String> {
    if kind == ModuleKind::TriadSm90a && arch != "sm_90a" {
        return Err(format!(
            "TriadSm90a requires exact target sm_90a, got {arch}"
        ));
    }
    if kind == ModuleKind::TriadSm100 && sm100_target_for_arch(arch).is_none() {
        return Err(format!(
            "TriadSm100 requires compute_100f, compute_100a, compute_103f, or compute_103a, got {arch}"
        ));
    }
    Ok(())
}

fn validate_specialized_ptx(kind: ModuleKind, arch: &str, ptx: &str) -> Result<(), String> {
    match kind {
        ModuleKind::TriadSm90a => validate_sm90a_ptx(ptx),
        ModuleKind::TriadSm100 => validate_sm100_ptx(arch, ptx),
        _ => Ok(()),
    }
}

fn ptx_target(ptx: &str) -> Result<&str, String> {
    ptx.lines()
        .find_map(|line| line.trim().strip_prefix(".target "))
        .and_then(|target| target.split(',').next().map(str::trim))
        .ok_or_else(|| "specialized PTX has no target directive".to_string())
}

fn validate_sm90a_ptx(ptx: &str) -> Result<(), String> {
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

fn sm100_target_for_arch(arch: &str) -> Option<super::contract::Sm100TargetCandidate> {
    [(10, 0), (10, 3)]
        .into_iter()
        .flat_map(super::dispatch::sm100_target_candidates)
        .copied()
        .find(|target| target.nvrtc_arch == arch)
}

fn validate_sm100_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let candidate = sm100_target_for_arch(arch)
        .ok_or_else(|| format!("TriadSm100 has no target candidate for {arch}"))?;
    let actual = ptx_target(ptx)?;
    if actual != candidate.ptx_target {
        return Err(format!(
            "TriadSm100 PTX target is {actual}, expected {}",
            candidate.ptx_target
        ));
    }
    for spec in super::contract::SM100_KERNEL_SPECS {
        let marker = format!(".entry {}(", spec.symbol);
        if ptx.matches(&marker).count() != 1 {
            return Err(format!(
                "TriadSm100 PTX must contain one entry {}",
                spec.symbol
            ));
        }
    }
    validate_sm100_feature_instructions(ptx)
}

fn validate_sm100_probe_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let candidate = sm100_target_for_arch(arch)
        .ok_or_else(|| format!("TriadSm100 has no target candidate for {arch}"))?;
    let actual = ptx_target(ptx)?;
    if actual != candidate.ptx_target {
        return Err(format!(
            "TriadSm100 probe PTX target is {actual}, expected {}",
            candidate.ptx_target
        ));
    }
    if ptx.matches(".entry tcgen05_probe(").count() != 1 {
        return Err("TriadSm100 probe PTX must contain one entry tcgen05_probe".into());
    }
    validate_sm100_feature_instructions(ptx)
}

fn validate_sm100_feature_instructions(ptx: &str) -> Result<(), String> {
    for instruction in [
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
        "mbarrier.arrive.expect_tx",
        "mbarrier.try_wait.parity",
        "tcgen05.alloc.cta_group::1",
        "tcgen05.relinquish_alloc_permit.cta_group::1",
        "tcgen05.dealloc.cta_group::1",
        "tcgen05.mma.cta_group::1.kind::f16",
        "tcgen05.commit.cta_group::1",
        "tcgen05.fence::before_thread_sync",
        "tcgen05.fence::after_thread_sync",
        "tcgen05.ld.sync.aligned.32x32b.x8.b32",
        "tcgen05.wait::ld.sync.aligned",
        "tcgen05.st.sync.aligned.32x32b.x8.b32",
        "tcgen05.wait::st.sync.aligned",
    ] {
        if !ptx.contains(instruction) {
            return Err(format!("TriadSm100 PTX is missing {instruction}"));
        }
    }
    if ptx.split_ascii_whitespace().any(|token| {
        token.starts_with("atom.")
            || token.starts_with("red.")
            || token.starts_with("atom::")
            || token.starts_with("red::")
            || token.starts_with("tcgen05.ld.red")
            || token.starts_with("wgmma.")
            || token.contains("cta_group::2")
            || token.contains("multicast")
    }) {
        return Err("TriadSm100 PTX contains a forbidden instruction family".into());
    }
    if ptx.split_ascii_whitespace().any(|token| {
        token == "call"
            || token.starts_with("call.")
            || token == ".callprototype"
            || token == ".calltargets"
    }) {
        return Err("TriadSm100 PTX contains a device call instruction".into());
    }
    for symbol in [
        "cudaLaunchDevice",
        "cudaGetParameterBuffer",
        "cudaDeviceSynchronize",
        "__cudaPushCallConfiguration",
        "__cudaPopCallConfiguration",
        "malloc",
        "free",
        "operator new",
        "operator delete",
    ] {
        if ptx.contains(symbol) {
            return Err(format!(
                "TriadSm100 PTX contains forbidden device-runtime symbol {symbol}"
            ));
        }
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

const SM100_SOURCE_FRAGMENTS: &[SourceFragment] = &[
    TYPED_PRELUDE,
    TRIAD_CONTRACT,
    TRIAD_COMMON,
    TRIAD_EPILOGUE,
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/sm100.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/sm100.cu"),
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
        ModuleKind::TriadSm100 => Ok(SM100_SOURCE_FRAGMENTS),
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
type Sm100MapCacheKey = (
    [super::contract::Sm90aTensorMapKey; 2],
    [super::contract::Sm90aAllocationIdentity; 2],
    super::contract::Sm100TensorOrigins,
    super::contract::Sm100Op,
    u8,
    super::contract::Sm100Tile,
    super::contract::Sm100Shape,
);
type Sm100MapCache = Mutex<HashMap<Sm100MapCacheKey, super::contract::Sm100PreparedTensorMaps>>;

pub(crate) fn qualify_specialized_module(
    module: CompiledModule,
) -> Result<QualifiedSpecializedModule, String> {
    let functions = match module.artifact_identity.module_kind {
        ModuleKind::TriadSm90a => load_sm90a_functions(&module),
        ModuleKind::TriadSm100 => load_sm100_functions(&module),
        kind => Err(format!("unsupported specialized triad module {kind:?}")),
    }?;
    Ok(QualifiedSpecializedModule { module, functions })
}

pub struct GemmBiKernels {
    _modules: CudaModuleAnchors,
    context_handle: usize,
    scalar_compiler_identity: CompilerIdentity,
    sm80_compiler_identity: CompilerIdentity,
    specialized_compiler_identity: Option<CompilerIdentity>,
    artifact_set_identity: crate::mamba_ssm::gpu::kernel_identity::ArtifactSetIdentity,
    specialized_functions: HashMap<&'static str, CudaFunction>,
    sm90a_tensor_maps: Sm90aMapCache,
    sm100_tensor_maps: Sm100MapCache,

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
        specialized: Option<QualifiedSpecializedModule>,
    ) -> Result<Self, String> {
        let mut artifacts = vec![
            fixed_artifact,
            scalar.artifact_identity,
            sm80.artifact_identity,
        ];
        if let Some(specialized) = specialized.as_ref() {
            artifacts.push(specialized.module.artifact_identity);
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

        let specialized_functions = specialized
            .as_ref()
            .map(|specialized| specialized.functions.clone())
            .unwrap_or_default();
        let mut anchors = vec![scalar.module.clone(), sm80.module.clone()];
        if let Some(specialized) = specialized.as_ref() {
            anchors.push(specialized.module.module.clone());
        }

        Ok(Self {
            _modules: CudaModuleAnchors::new(anchors),
            context_handle,
            scalar_compiler_identity: scalar.compiler_identity,
            sm80_compiler_identity: sm80.compiler_identity,
            specialized_compiler_identity: specialized
                .as_ref()
                .map(|specialized| specialized.module.compiler_identity),
            artifact_set_identity,
            specialized_functions,
            sm90a_tensor_maps: Mutex::new(HashMap::new()),
            sm100_tensor_maps: Mutex::new(HashMap::new()),
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
        self.artifact_set_identity
            .specialized
            .filter(|artifact| artifact.module_kind == ModuleKind::TriadSm90a)
            .and(self.specialized_compiler_identity)
    }

    pub fn sm100_compiler_identity(&self) -> Option<CompilerIdentity> {
        self.artifact_set_identity
            .specialized
            .filter(|artifact| artifact.module_kind == ModuleKind::TriadSm100)
            .and(self.specialized_compiler_identity)
    }

    pub fn sm100_target_candidate(&self) -> Option<super::contract::Sm100TargetCandidate> {
        self.sm100_compiler_identity()
            .and_then(|compiler| sm100_target_for_arch(compiler.target.as_str()))
    }

    pub fn has_sm90a_wgmma(&self) -> bool {
        self.sm90a_compiler_identity().is_some()
            && self.specialized_functions.len() == SM90A_SYMBOLS.len()
    }

    pub fn has_sm100_tcgen(&self) -> bool {
        self.sm100_compiler_identity().is_some()
            && self.specialized_functions.len() == super::contract::SM100_KERNEL_SPECS.len()
    }

    pub(super) fn context_handle(&self) -> usize {
        self.context_handle
    }

    pub(super) fn sm90a_function(&self, symbol: &str) -> Option<&CudaFunction> {
        self.has_sm90a_wgmma()
            .then(|| self.specialized_functions.get(symbol))
            .flatten()
    }

    pub(super) fn sm100_function(&self, symbol: &str) -> Option<&CudaFunction> {
        self.has_sm100_tcgen()
            .then(|| self.specialized_functions.get(symbol))
            .flatten()
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
            || Some(binding.compiler) != self.sm90a_compiler_identity()
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

    pub(super) fn prepare_sm100_tensor_maps(
        &self,
        request: super::contract::Sm100MapRequest,
        keys: [super::contract::Sm90aTensorMapKey; 2],
        allocations: [super::contract::Sm90aAllocationIdentity; 2],
        origins: super::contract::Sm100TensorOrigins,
        capturing: bool,
        binding: super::contract::Sm100MapBinding,
    ) -> Result<super::contract::Sm100PreparedTensorMaps, String> {
        if binding.context_handle != self.context_handle
            || Some(binding.compiler) != self.sm100_compiler_identity()
            || Some(binding.artifact) != self.artifact_set_identity.specialized
            || Some(binding.target) != self.sm100_target_candidate()
        {
            return Err("SM100 tensor-map binding does not match its CUDA module context".into());
        }
        let mut cache = self
            .sm100_tensor_maps
            .lock()
            .map_err(|_| "SM100 tensor-map cache is poisoned".to_string())?;
        let dtype = match request.dtype {
            super::super::dtype::WeightDtype::F32 => 0,
            super::super::dtype::WeightDtype::F16 => 1,
            super::super::dtype::WeightDtype::Bf16 => 2,
        };
        let cache_key = (
            keys,
            allocations,
            origins,
            request.op,
            dtype,
            request.tile,
            request.shape,
        );
        cache.retain(|(cached_keys, cached_allocations, _, _, _, _, _), _| {
            *cached_keys != keys || *cached_allocations == allocations
        });
        if let Some(maps) = cache.get(&cache_key) {
            return Ok(*maps);
        }
        if capturing {
            return Err("SM100 tensor-map cache miss during graph capture".into());
        }
        let maps = super::contract::encode_sm100_tensor_maps(
            keys,
            request,
            binding,
            allocations,
            origins,
        )?;
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

fn load_sm100_functions(
    module: &CompiledModule,
) -> Result<HashMap<&'static str, CudaFunction>, String> {
    if module.artifact_identity.module_kind != ModuleKind::TriadSm100
        || sm100_target_for_arch(module.compiler_identity.target.as_str()).is_none()
    {
        return Err("specialized triad module is not a valid TriadSm100 target".into());
    }
    let mut functions = HashMap::new();
    for spec in super::contract::SM100_KERNEL_SPECS {
        let function = load_function(&module.module, ModuleKind::TriadSm100, spec.symbol)?;
        let shared = i32::try_from(spec.dynamic_shared_bytes)
            .map_err(|_| format!("{} shared memory exceeds i32::MAX", spec.symbol))?;
        set_dynamic_shared(&function, spec.symbol, shared)?;
        if function
            .local_size_bytes()
            .map_err(|error| format!("query {} local memory: {error:?}", spec.symbol))?
            != 0
        {
            return Err(format!("{} spills to local memory", spec.symbol));
        }
        let threads = i32::try_from(spec.threads)
            .map_err(|_| format!("{} thread count exceeds i32::MAX", spec.symbol))?;
        if function
            .max_threads_per_block()
            .map_err(|error| format!("query {} max threads: {error:?}", spec.symbol))?
            < threads
        {
            return Err(format!(
                "{} cannot launch {} threads",
                spec.symbol, spec.threads
            ));
        }
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(
                spec.threads,
                spec.dynamic_shared_bytes as usize,
                None,
            )
            .map_err(|error| format!("query {} occupancy: {error:?}", spec.symbol))?;
        if occupancy < 1 {
            return Err(format!("{} has zero launch occupancy", spec.symbol));
        }
        functions.insert(spec.symbol, function);
    }
    if functions.len() != super::contract::SM100_KERNEL_SPECS.len() {
        return Err("TriadSm100 did not load its complete symbol inventory".into());
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
        SM90A_SYMBOLS, SM100_PROBE_SOURCE, SourceFragment, compose_fragments,
        compose_module_source, resolve_owned_symbol, select_sm100_candidate,
        validate_module_target, validate_sm100_probe_ptx, validate_sm100_ptx,
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

    const SM100_FRAGMENTS: &[&str] = &[
        "kernels/_typed_prelude.cuh",
        "kernels/gemm_bi_triad/contract.cuh",
        "kernels/gemm_bi_triad/common.cuh",
        "kernels/gemm_bi_triad/epilogue.cuh",
        "kernels/gemm_bi_triad/sm100.cu",
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
        assert_composition(ModuleKind::TriadSm100, SM100_FRAGMENTS);

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
        let sm90a: BTreeSet<_> = SM90A_SYMBOLS.iter().copied().collect();
        let sm100: BTreeSet<_> = super::super::contract::SM100_KERNEL_SPECS
            .iter()
            .map(|spec| spec.symbol)
            .collect();
        assert_eq!(sm90a.len(), 12);
        assert_eq!(sm100.len(), 72);
        assert!(scalar.is_disjoint(&sm80));
        assert!(scalar.is_disjoint(&sm90a));
        assert!(scalar.is_disjoint(&sm100));
        assert!(sm80.is_disjoint(&sm90a));
        assert!(sm80.is_disjoint(&sm100));
        assert!(sm90a.is_disjoint(&sm100));
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
    fn sm100_module_accepts_only_exact_feature_targets() {
        for target in [
            "compute_100f",
            "compute_100a",
            "compute_103f",
            "compute_103a",
        ] {
            validate_module_target(ModuleKind::TriadSm100, target).unwrap();
        }
        for target in [
            "compute_100",
            "compute_103",
            "sm_100a",
            "sm_103a",
            "compute_120f",
        ] {
            assert!(validate_module_target(ModuleKind::TriadSm100, target).is_err());
        }
    }

    #[test]
    fn sm100_probe_is_small_feature_complete_and_validated_fail_closed() {
        assert!(SM100_PROBE_SOURCE.contains("tcgen05_probe"));
        assert!(SM100_PROBE_SOURCE.contains("cp.async.bulk.tensor.2d"));
        assert!(SM100_PROBE_SOURCE.contains("tcgen05.mma.cta_group::1.kind::f16"));
        assert!(!SM100_PROBE_SOURCE.contains("sgemm_bi_nn_sm100"));

        let instructions = [
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
            "mbarrier.arrive.expect_tx",
            "mbarrier.try_wait.parity",
            "tcgen05.alloc.cta_group::1",
            "tcgen05.relinquish_alloc_permit.cta_group::1",
            "tcgen05.dealloc.cta_group::1",
            "tcgen05.mma.cta_group::1.kind::f16",
            "tcgen05.commit.cta_group::1",
            "tcgen05.fence::before_thread_sync",
            "tcgen05.fence::after_thread_sync",
            "tcgen05.ld.sync.aligned.32x32b.x8.b32",
            "tcgen05.wait::ld.sync.aligned",
            "tcgen05.st.sync.aligned.32x32b.x8.b32",
            "tcgen05.wait::st.sync.aligned",
        ];
        let valid = format!(
            ".version 9.0\n.target sm_100f\n.entry tcgen05_probe(\n{}\n",
            instructions.join("\n")
        );
        validate_sm100_probe_ptx("compute_100f", &valid).unwrap();

        assert!(validate_sm100_probe_ptx("compute_100a", &valid).is_err());
        assert!(
            validate_sm100_probe_ptx("compute_100f", &valid.replace("tcgen05_probe", "wrong"))
                .is_err()
        );
        assert!(
            validate_sm100_probe_ptx(
                "compute_100f",
                &valid.replace("tcgen05.wait::st.sync.aligned", "")
            )
            .is_err()
        );
        assert!(
            validate_sm100_probe_ptx("compute_100f", &format!("{valid}\natom.global.add.f32"))
                .is_err()
        );
    }

    #[test]
    fn sm100_probe_compiles_for_every_exact_feature_target() {
        for (requested, emitted) in [
            ("compute_100f", "sm_100f"),
            ("compute_100a", "sm_100a"),
            ("compute_103f", "sm_103f"),
            ("compute_103a", "sm_103a"),
        ] {
            let options = cudarc::nvrtc::CompileOptions {
                arch: Some(requested),
                options: vec!["--fmad=true".to_string(), "-DNDEBUG".to_string()],
                ..Default::default()
            };
            let image = cudarc::nvrtc::compile_ptx_with_opts(SM100_PROBE_SOURCE, options)
                .unwrap_or_else(|error| panic!("SM100 probe failed for {requested}: {error}"));
            let ptx = crate::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(
                image.as_bytes().expect("SM100 probe PTX image"),
            )
            .expect("SM100 probe PTX must be canonical UTF-8");
            assert!(
                ptx.lines()
                    .any(|line| line.trim() == format!(".target {emitted}"))
            );
            validate_sm100_probe_ptx(requested, &ptx).unwrap();
        }
    }

    #[test]
    fn sm100_candidate_fallback_survives_family_probe_failure() {
        let probes = std::cell::RefCell::new(Vec::new());
        let compiles = std::cell::RefCell::new(Vec::new());
        let qualified = std::cell::RefCell::new(Vec::new());
        let selected = select_sm100_candidate(
            super::super::dispatch::sm100_target_candidates((10, 0)),
            |candidate| {
                probes.borrow_mut().push(candidate.nvrtc_arch);
                if candidate.kind == super::super::contract::Sm100TargetKind::Family {
                    Err("injected family probe failure".into())
                } else {
                    Ok(())
                }
            },
            |candidate| {
                compiles.borrow_mut().push(candidate.nvrtc_arch);
                Ok(candidate)
            },
            |candidate| {
                qualified.borrow_mut().push(candidate.nvrtc_arch);
                Ok(candidate)
            },
        )
        .expect("exact candidate fallback");
        assert_eq!(selected.nvrtc_arch, "compute_100a");
        assert_eq!(*probes.borrow(), ["compute_100f", "compute_100a"]);
        assert_eq!(*compiles.borrow(), ["compute_100a"]);
        assert_eq!(*qualified.borrow(), ["compute_100a"]);
    }

    #[test]
    fn sm100_candidate_fallback_survives_family_compile_failure() {
        let compiles = std::cell::RefCell::new(Vec::new());
        let qualified = std::cell::RefCell::new(Vec::new());
        let selected = select_sm100_candidate(
            super::super::dispatch::sm100_target_candidates((10, 0)),
            |_| Ok(()),
            |candidate| {
                compiles.borrow_mut().push(candidate.nvrtc_arch);
                if candidate.kind == super::super::contract::Sm100TargetKind::Family {
                    Err("injected family compile failure".into())
                } else {
                    Ok(candidate)
                }
            },
            |candidate| {
                qualified.borrow_mut().push(candidate.nvrtc_arch);
                Ok(candidate)
            },
        )
        .expect("exact candidate fallback");
        assert_eq!(selected.nvrtc_arch, "compute_100a");
        assert_eq!(*compiles.borrow(), ["compute_100f", "compute_100a"]);
        assert_eq!(*qualified.borrow(), ["compute_100a"]);
    }

    #[test]
    fn sm100_candidate_fallback_survives_family_qualification_failure() {
        let qualified = std::cell::RefCell::new(Vec::new());
        let selected = select_sm100_candidate(
            super::super::dispatch::sm100_target_candidates((10, 0)),
            |_| Ok(()),
            Ok,
            |candidate| {
                qualified.borrow_mut().push(candidate.nvrtc_arch);
                if candidate.kind == super::super::contract::Sm100TargetKind::Family {
                    Err("injected family qualification failure".into())
                } else {
                    Ok(candidate)
                }
            },
        )
        .expect("exact candidate fallback");
        assert_eq!(selected.nvrtc_arch, "compute_100a");
        assert_eq!(*qualified.borrow(), ["compute_100f", "compute_100a"]);
    }

    #[test]
    fn sm100_production_ptx_validator_rejects_partial_or_mixed_artifacts() {
        let instructions = [
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
            "mbarrier.arrive.expect_tx",
            "mbarrier.try_wait.parity",
            "tcgen05.alloc.cta_group::1",
            "tcgen05.relinquish_alloc_permit.cta_group::1",
            "tcgen05.dealloc.cta_group::1",
            "tcgen05.mma.cta_group::1.kind::f16",
            "tcgen05.commit.cta_group::1",
            "tcgen05.fence::before_thread_sync",
            "tcgen05.fence::after_thread_sync",
            "tcgen05.ld.sync.aligned.32x32b.x8.b32",
            "tcgen05.wait::ld.sync.aligned",
            "tcgen05.st.sync.aligned.32x32b.x8.b32",
            "tcgen05.wait::st.sync.aligned",
        ];
        let mut valid = ".version 9.0\n.target sm_100f\n".to_string();
        for spec in super::super::contract::SM100_KERNEL_SPECS {
            valid.push_str(&format!(".entry {}(\n", spec.symbol));
        }
        valid.push_str(&instructions.join("\n"));
        validate_sm100_ptx("compute_100f", &valid).unwrap();

        assert!(validate_sm100_ptx("compute_100a", &valid).is_err());
        let first = super::super::contract::SM100_KERNEL_SPECS[0].symbol;
        assert!(
            validate_sm100_ptx(
                "compute_100f",
                &valid.replacen(&format!(".entry {first}("), ".entry missing(", 1),
            )
            .is_err()
        );
        assert!(
            validate_sm100_ptx(
                "compute_100f",
                &format!("{valid}\nwgmma.fence.sync.aligned")
            )
            .is_err()
        );
        for forbidden in [
            "call.uni (_), cudaLaunchDevice, ();",
            ".callprototype ()_();",
            ".extern .func malloc;",
            ".extern .func free;",
            ".extern .func cudaGetParameterBuffer;",
        ] {
            assert!(
                validate_sm100_ptx("compute_100f", &format!("{valid}\n{forbidden}")).is_err(),
                "validator accepted {forbidden}"
            );
        }
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
