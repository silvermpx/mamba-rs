#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GpuCtx, HalfTriadPolicy};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    PhysicalQualificationRequest, PhysicalQualificationRoute, QualifiedPhysicalLaunchEvidence,
    TcTile, presize_physical_qualification_suite, qualify_physical_launch,
};
#[cfg(target_os = "linux")]
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    SM80_TF32_ROUTE_SPECS, SM80_TF32_WIDE_ROUTE_SPECS, SM120_KERNEL_SPECS,
    SM120_STREAMK_KERNEL_SPECS, SM120_TF32_ROUTE_SPECS,
};
#[cfg(target_os = "linux")]
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    ArtifactIdentity, ArtifactSetIdentity, CacheEnvelope, FramedSha256, digest_hex,
};
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    ArtifactKind, ModuleKind, PhysicalLaunchKind, PolicyDtype, ResolvedGemmOp,
};

#[cfg(target_os = "linux")]
const SCALAR_ENTRIES: &[&str] = &[
    "gemm_bi_nn",
    "gemm_bi_nn_m64n64_bk16_s2_v1",
    "gemm_bi_nn_splitk32_m32n64_exact_v1",
    "gemm_bi_nn_prism_m64n64_bk16_s2_v1",
    "gemm_bi_nn_zero_reduction_v1",
    "gemm_bi_tn",
    "gemm_bi_tn_aligned",
    "gemm_bi_tn_zero_reduction_v1",
    "gemm_bi_tn_splitm_partial",
    "gemm_bi_tn_m16n16_bk16_s2_splitm16_v1",
    "gemm_bi_tn_splitm_partial_aligned",
    "gemm_bi_splitm_reduce",
    "gemm_bi_nt",
    "gemm_bi_nt_m2n16_bk64_splitk32_v1",
    "gemm_bi_nt_zero_reduction_v1",
    "gemm_bi_nn_slim",
    "gemm_bi_nn_splitk_slim_partial",
    "gemm_bi_tn_slim",
    "gemm_bi_nt_slim",
    "gemm_bi_nn_ultra_thin",
    "gemm_bi_nn_gemv",
    "gemm_bi_tn_gemv",
    "gemm_bi_nt_gemv",
    "gemm_bi_nn_narrow",
    "gemm_bi_nn_narrow_small",
    "gemm_bi_tn_narrow",
    "gemm_bi_tn_narrow_splitm_partial",
    "gemm_bi_tn_narrow_splitm_partial_aligned",
    "gemm_bi_nt_narrow",
    "gemm_bi_nn_splitk32_partial",
    "gemm_bi_splitk_reduce",
    "gemm_bi_dx_col_gemv",
    "gemm_bi_transpose_f32_2d",
    "gemm_bi_transpose_f32_32x16_d768_v1",
    "gemm_bi_nn_gemv_bf16",
    "gemm_bi_nn_gemv_f16",
    "gemm_bi_tn_gemv_bf16",
    "gemm_bi_tn_gemv_f16",
    "gemm_bi_nt_gemv_bf16",
    "gemm_bi_nt_gemv_f16",
    "gemm_bi_nn_ultra_thin_bf16",
    "gemm_bi_nn_ultra_thin_f16",
    "gemm_bi_nn_narrow_bf16",
    "gemm_bi_nn_narrow_f16",
    "gemm_bi_nn_narrow_small_bf16",
    "gemm_bi_nn_narrow_small_f16",
    "gemm_bi_tn_narrow_bf16",
    "gemm_bi_tn_narrow_f16",
    "gemm_bi_nt_narrow_bf16",
    "gemm_bi_nt_narrow_f16",
    "gemm_bi_nn_big_bf16",
    "gemm_bi_nn_big_f16",
    "gemm_bi_tn_big_bf16",
    "gemm_bi_tn_big_f16",
    "gemm_bi_nt_big_bf16",
    "gemm_bi_nt_big_f16",
];

#[cfg(target_os = "linux")]
const SM80_TYPED_ENTRIES: &[&str] = &[
    "gemm_bi_nn_tc_bf16",
    "gemm_bi_nn_tc_f16",
    "gemm_bi_tn_tc_bf16",
    "gemm_bi_tn_tc_f16",
    "gemm_bi_nt_tc_bf16",
    "gemm_bi_nt_tc_f16",
    "gemm_bi_nn_tc64_bf16",
    "gemm_bi_nn_tc64_f16",
    "gemm_bi_nn_tc16_bf16",
    "gemm_bi_nn_tc16_f16",
    "gemm_bi_tn_tc64_bf16",
    "gemm_bi_tn_tc64_f16",
    "gemm_bi_tn_tc128x64_bf16",
    "gemm_bi_tn_tc128x64_f16",
    "gemm_bi_nt_tc64_bf16",
    "gemm_bi_nt_tc64_f16",
];

/// Composed into the portable module on every sm80-family target except
/// CC 12.x, whose boards run the SM120 stream-K kernel instead.
#[cfg(target_os = "linux")]
const SM80_STREAMK_ENTRIES: &[&str] = &[
    "gemm_bi_tn_tc64_streamk_bf16",
    "gemm_bi_tn_tc64_streamk_f16",
];

#[cfg(target_os = "linux")]
fn sm80_target_composes_streamk(target: &str) -> bool {
    !matches!(target, "compute_120" | "compute_121" | "sm_120" | "sm_121")
}

#[cfg(target_os = "linux")]
const SM80_SPLITK_ENTRIES: &[&str] = &[
    "gemm_bi_nn_sm80_mma_tf32_splitk2_v1_m16n32_bk32_s4",
    "gemm_bi_nn_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4",
    "gemm_bi_nt_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s3",
    "gemm_bi_nt_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4",
    "gemm_bi_nt_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s3",
    "gemm_bi_nt_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s4",
];

/// The TN split-K family of the extension fragment, composed with the
/// stream-K kernels and the wide TF32 tile on the sm80-family targets.
#[cfg(target_os = "linux")]
const SM80_TN_SPLITK_ENTRIES: &[&str] = &[
    "gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m64n64_bk32_s2",
    "gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m64n64_bk32_s3",
    "gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s3",
    "gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s4",
];

#[cfg(target_os = "linux")]
/// The portable TF32 entries every Fixed PTX carries, whatever its target.
const FIXED_TF32_ENTRIES_PORTABLE: &[&str] = &[
    "gemm_bi_nn_tf32_v1_m128n64_bk32_s2",
    "gemm_bi_nn_tf32_v1_m128n64_bk32_s3",
    "gemm_bi_nn_tf32_v1_m64n64_bk32_s2",
    "gemm_bi_nn_tf32_v1_m64n64_bk32_s3",
    "gemm_bi_nn_tf32_v1_m16n32_bk32_s4",
];

/// The TMA entries the Fixed source compiles only for the 12.0 and 12.1
/// architectures; every other target's PTX must not carry them.
#[cfg(target_os = "linux")]
const FIXED_TF32_ENTRIES_SM120: &[&str] = &[
    "gemm_bi_nn_sm120_tma_tf32_v1_m128n64_bk32_s2",
    "gemm_bi_nn_sm120_tma_tf32_v1_m128n64_bk32_s3",
    "gemm_bi_nn_sm120_tma_tf32_v1_m64n128_bk32_s2",
    "gemm_bi_nn_sm120_tma_tf32_v1_m64n128_bk32_s3",
    "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_producer_warp",
    "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2",
    "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_pair_store",
];

/// The Fixed TF32 entries a PTX compiled for `target` carries: the portable
/// set on every architecture, plus the TMA set on the two SM120 targets the
/// kernel source admits.
#[cfg(target_os = "linux")]
fn expected_fixed_tf32_entries(target: &str) -> std::collections::BTreeSet<String> {
    let sm120 = matches!(target, "compute_120" | "compute_121");
    FIXED_TF32_ENTRIES_PORTABLE
        .iter()
        .chain(FIXED_TF32_ENTRIES_SM120.iter().filter(|_| sm120))
        .map(|name| (*name).to_string())
        .collect()
}

#[cfg(target_os = "linux")]
fn expected_scalar_entries() -> std::collections::BTreeSet<String> {
    SCALAR_ENTRIES
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}

#[cfg(target_os = "linux")]
fn expected_sm80_entries(target: &str) -> std::collections::BTreeSet<String> {
    SM80_TYPED_ENTRIES
        .iter()
        .chain(
            SM80_STREAMK_ENTRIES
                .iter()
                .filter(|_| sm80_target_composes_streamk(target)),
        )
        .chain(SM80_SPLITK_ENTRIES)
        .chain(
            SM80_TN_SPLITK_ENTRIES
                .iter()
                .filter(|_| sm80_target_composes_streamk(target)),
        )
        .map(|name| (*name).to_string())
        .chain(
            SM80_TF32_ROUTE_SPECS
                .iter()
                .chain(
                    SM80_TF32_WIDE_ROUTE_SPECS
                        .iter()
                        .filter(|_| sm80_target_composes_streamk(target)),
                )
                .map(|spec| spec.symbol.to_string()),
        )
        .collect()
}

#[cfg(target_os = "linux")]
#[test]
fn expected_cuda_module_fixtures_are_unique() {
    assert_eq!(expected_scalar_entries().len(), SCALAR_ENTRIES.len());
    assert_eq!(
        expected_sm80_entries("sm_89").len(),
        SM80_TYPED_ENTRIES.len()
            + SM80_STREAMK_ENTRIES.len()
            + SM80_SPLITK_ENTRIES.len()
            + SM80_TN_SPLITK_ENTRIES.len()
            + SM80_TF32_ROUTE_SPECS.len()
            + SM80_TF32_WIDE_ROUTE_SPECS.len()
    );
    assert_eq!(
        expected_sm80_entries("compute_120").len(),
        SM80_TYPED_ENTRIES.len() + SM80_SPLITK_ENTRIES.len() + SM80_TF32_ROUTE_SPECS.len()
    );
    assert_eq!(
        expected_fixed_tf32_entries("compute_120").len(),
        FIXED_TF32_ENTRIES_PORTABLE.len() + FIXED_TF32_ENTRIES_SM120.len()
    );
    assert_eq!(
        expected_fixed_tf32_entries("compute_89").len(),
        FIXED_TF32_ENTRIES_PORTABLE.len()
    );
    let specialized: std::collections::BTreeSet<_> = SM120_KERNEL_SPECS
        .iter()
        .chain(SM120_STREAMK_KERNEL_SPECS.iter())
        .map(|spec| spec.symbol)
        .chain(SM120_TF32_ROUTE_SPECS.iter().map(|spec| spec.symbol))
        .collect();
    assert_eq!(
        specialized.len(),
        SM120_KERNEL_SPECS.len() + SM120_STREAMK_KERNEL_SPECS.len() + SM120_TF32_ROUTE_SPECS.len()
    );
}

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
fn cache_entries(
    cache: &std::path::Path,
    prefix: &str,
    expected_count: usize,
) -> Vec<std::path::PathBuf> {
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
    assert_eq!(
        entries.len(),
        expected_count,
        "one envelope per active CUDA module"
    );
    entries
}

#[cfg(target_os = "linux")]
fn active_artifacts(artifacts: ArtifactSetIdentity) -> Vec<ArtifactIdentity> {
    let expected_count = 3
        + usize::from(artifacts.specialized.is_some())
        + usize::from(artifacts.sm89_half.is_some())
        + usize::from(artifacts.sm89_exact_f32.is_some())
        + usize::from(artifacts.sm89_exact_f32_d128.is_some())
        + usize::from(artifacts.sm89_tf32_joint.is_some());
    assert_eq!(
        usize::from(artifacts.module_count),
        expected_count,
        "artifact-set count must match specialized-module presence"
    );
    let mut active = vec![
        artifacts.fixed,
        artifacts.triad_scalar,
        artifacts.triad_sm80,
    ];
    if let Some(specialized) = artifacts.specialized {
        active.push(specialized);
    }
    if let Some(sm89_half) = artifacts.sm89_half {
        active.push(sm89_half);
    }
    if let Some(sm89_exact_f32) = artifacts.sm89_exact_f32 {
        active.push(sm89_exact_f32);
    }
    if let Some(sm89_exact_f32_d128) = artifacts.sm89_exact_f32_d128 {
        active.push(sm89_exact_f32_d128);
    }
    if let Some(sm89_tf32_joint) = artifacts.sm89_tf32_joint {
        active.push(sm89_tf32_joint);
    }
    let kinds: std::collections::HashSet<_> =
        active.iter().map(|artifact| artifact.module_kind).collect();
    let digests: std::collections::HashSet<_> = active
        .iter()
        .map(|artifact| artifact.artifact_digest)
        .collect();
    assert_eq!(
        kinds.len(),
        expected_count,
        "active module kind is duplicated"
    );
    assert_eq!(
        digests.len(),
        expected_count,
        "active module artifact is duplicated"
    );
    active
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
        .unwrap_or_else(|| {
            let seen = entries
                .iter()
                .map(|path| {
                    format!(
                        "{}={}",
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        digest_hex(&FramedSha256::bytes(&envelope_payload(path)))
                    )
                })
                .collect::<Vec<_>>();
            panic!(
                "cache payload for {:?} not found: expected artifact digest {} under compile key {}; cache holds {seen:?}",
                artifact.module_kind,
                digest_hex(&artifact.artifact_digest),
                digest_hex(&artifact.compile_key)
            )
        })
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

struct HalfTraceFixture {
    op: ResolvedGemmOp,
    dtype: WeightDtype,
    dims: (usize, usize, usize),
    forced_tile: Option<TcTile>,
    tensor_cores: bool,
}

fn record_half_fixture(
    ctx: &GpuCtx,
    fixture: HalfTraceFixture,
) -> Result<QualifiedPhysicalLaunchEvidence, String> {
    let route = if let Some(tile) = fixture.forced_tile {
        PhysicalQualificationRoute::HalfForced {
            dtype: fixture.dtype,
            tile,
        }
    } else {
        PhysicalQualificationRoute::HalfPolicy {
            dtype: fixture.dtype,
            tensor_cores: fixture.tensor_cores,
            half_policy: HalfTriadPolicy::TiledParityV1,
        }
    };
    let request = PhysicalQualificationRequest::contiguous(fixture.op, fixture.dims, route);
    let qualified = qualify_physical_launch(ctx, request)?;
    Ok(qualified.evidence().clone())
}

fn half_trace_context() -> GpuCtx {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    let scratch_requests = [
        PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nn,
            (128, 128, 128),
            PhysicalQualificationRoute::HalfPolicy {
                dtype: WeightDtype::Bf16,
                tensor_cores: true,
                half_policy: HalfTriadPolicy::TiledParityV1,
            },
        ),
        PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nn,
            (64, 384, 512),
            PhysicalQualificationRoute::HalfPolicy {
                dtype: WeightDtype::Bf16,
                tensor_cores: false,
                half_policy: HalfTriadPolicy::TiledParityV1,
            },
        ),
        PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Tn,
            (64, 384, 512),
            PhysicalQualificationRoute::HalfPolicy {
                dtype: WeightDtype::Bf16,
                tensor_cores: false,
                half_policy: HalfTriadPolicy::TiledParityV1,
            },
        ),
        PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nt,
            (64, 384, 512),
            PhysicalQualificationRoute::HalfPolicy {
                dtype: WeightDtype::Bf16,
                tensor_cores: false,
                half_policy: HalfTriadPolicy::TiledParityV1,
            },
        ),
    ];
    presize_physical_qualification_suite(&ctx, &scratch_requests)
        .expect("pre-size half qualification fixtures");
    ctx
}

#[test]
#[ignore = "requires a CUDA device"]
fn half_physical_trace_records_native_scalar_and_tensor_core_routes() {
    let ctx = half_trace_context();
    for (dtype, suffix) in [(WeightDtype::Bf16, "_bf16"), (WeightDtype::F16, "_f16")] {
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            let scalar = record_half_fixture(
                &ctx,
                HalfTraceFixture {
                    op,
                    dtype,
                    dims: (64, 96, 80),
                    forced_tile: None,
                    tensor_cores: false,
                },
            )
            .unwrap();
            assert_eq!(scalar.nodes().len(), 1);
            assert_eq!(scalar.nodes()[0].kind, PhysicalLaunchKind::Gemm);
            assert_eq!(scalar.nodes()[0].logical_op, op);
            assert!(scalar.nodes()[0].symbol.ends_with(suffix));
            assert_eq!(scalar.nodes()[0].module_kind, ModuleKind::TriadScalar);

            let tensor_core = record_half_fixture(
                &ctx,
                HalfTraceFixture {
                    op,
                    dtype,
                    dims: (128, 128, 128),
                    forced_tile: None,
                    tensor_cores: true,
                },
            )
            .unwrap();
            assert_eq!(tensor_core.nodes().len(), 1);
            assert_eq!(tensor_core.nodes()[0].kind, PhysicalLaunchKind::Gemm);
            assert_eq!(tensor_core.nodes()[0].logical_op, op);
            assert!(tensor_core.nodes()[0].symbol.ends_with(suffix));
            assert_eq!(tensor_core.nodes()[0].module_kind, ModuleKind::TriadSm80);
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn half_physical_trace_records_complete_upcast_routes() {
    let ctx = half_trace_context();
    for (dtype, upcast, downcast, logical_dtype) in [
        (
            WeightDtype::Bf16,
            "cast_bf16_to_f32",
            "cast_f32_to_bf16",
            PolicyDtype::Bf16,
        ),
        (
            WeightDtype::F16,
            "cast_f16_to_f32",
            "cast_f32_to_f16",
            PolicyDtype::F16,
        ),
    ] {
        for (op, expected_gemms, has_downcast) in [
            (
                ResolvedGemmOp::Nn,
                &["gemm_bi_nn_splitk32_partial", "gemm_bi_splitk_reduce"][..],
                true,
            ),
            (ResolvedGemmOp::Tn, &["gemm_bi_tn_slim"][..], false),
            (
                ResolvedGemmOp::Nt,
                &[
                    "gemm_bi_transpose_f32_2d",
                    "gemm_bi_nn_splitk32_partial",
                    "gemm_bi_splitk_reduce",
                ][..],
                true,
            ),
        ] {
            let expected_count = 2 + expected_gemms.len() + usize::from(has_downcast);
            let fallback = record_half_fixture(
                &ctx,
                HalfTraceFixture {
                    op,
                    dtype,
                    dims: (64, 384, 512),
                    forced_tile: None,
                    tensor_cores: false,
                },
            )
            .unwrap();
            let nodes = fallback.nodes();
            assert_eq!(nodes.len(), expected_count, "{op:?} {dtype:?}");
            assert_eq!(nodes[0].kind, PhysicalLaunchKind::InputUpcast);
            assert_eq!(nodes[0].symbol, upcast);
            assert_eq!(nodes[1].kind, PhysicalLaunchKind::InputUpcast);
            assert_eq!(nodes[1].symbol, upcast);
            assert_eq!(
                nodes[2..2 + expected_gemms.len()]
                    .iter()
                    .map(|node| node.symbol)
                    .collect::<Vec<_>>(),
                expected_gemms,
                "{op:?} {dtype:?}"
            );
            assert!(
                nodes[2..2 + expected_gemms.len()]
                    .iter()
                    .all(|node| node.kind == PhysicalLaunchKind::Gemm)
            );
            if has_downcast {
                assert_eq!(
                    nodes.last().unwrap().kind,
                    PhysicalLaunchKind::OutputDowncast
                );
                assert_eq!(nodes.last().unwrap().symbol, downcast);
            }
            assert!(nodes.iter().all(|node| node.logical_dtype == logical_dtype));
            assert!(
                nodes
                    .iter()
                    .filter(|node| node.kind != PhysicalLaunchKind::Gemm)
                    .all(|node| node.module_kind == ModuleKind::Fixed)
            );
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn half_qualification_forced_tiles_normalize_and_restore_policy() {
    let ctx = half_trace_context();
    for (op, tile, expected_extent, symbol) in [
        (
            ResolvedGemmOp::Nn,
            TcTile::Tile64,
            (64, 64),
            "gemm_bi_nn_tc64_bf16",
        ),
        (
            ResolvedGemmOp::Nn,
            TcTile::Tile128,
            (128, 128),
            "gemm_bi_nn_tc_bf16",
        ),
        (
            ResolvedGemmOp::Tn,
            TcTile::Tile64,
            (64, 64),
            "gemm_bi_tn_tc64_bf16",
        ),
        (
            ResolvedGemmOp::Tn,
            TcTile::Tile128,
            (128, 128),
            "gemm_bi_tn_tc_bf16",
        ),
        (
            ResolvedGemmOp::Nt,
            TcTile::Tile64,
            (64, 64),
            "gemm_bi_nt_tc64_bf16",
        ),
        (
            ResolvedGemmOp::Nt,
            TcTile::Tile128,
            (128, 128),
            "gemm_bi_nt_tc_bf16",
        ),
    ] {
        ctx.set_bi_tensor_cores(false);
        let trace = record_half_fixture(
            &ctx,
            HalfTraceFixture {
                op,
                dtype: WeightDtype::Bf16,
                dims: (128, 128, 128),
                forced_tile: Some(tile),
                tensor_cores: false,
            },
        )
        .unwrap();
        assert_eq!(trace.single_launch_tile(), Some(expected_extent));
        assert_eq!(trace.single_launch_symbol(), Some(symbol));
        assert!(!ctx.bi_tensor_cores(), "forced route must restore policy");
    }

    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_gemm_mode(GemmMode::CublasPedantic).unwrap();
    let trace = record_half_fixture(
        &ctx,
        HalfTraceFixture {
            op: ResolvedGemmOp::Nn,
            dtype: WeightDtype::Bf16,
            dims: (64, 96, 80),
            forced_tile: None,
            tensor_cores: false,
        },
    )
    .expect("qualification must normalize its private live policy");
    assert_eq!(trace.launch_count(), 1);
    assert_eq!(ctx.bi_gemm_family(), BiGemmFamily::Inference);
    assert!(!ctx.batch_invariant());
}

fn assert_exact_n64_resources(function: &cudarc::driver::CudaFunction) {
    assert_eq!(function.local_size_bytes().unwrap(), 0);
    assert_eq!(function.shared_size_bytes().unwrap(), 32_768);
    assert!((1..=160).contains(&function.num_regs().unwrap()));
    assert!(function.max_threads_per_block().unwrap() >= 128);
    assert!(
        function
            .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
            .unwrap()
            >= 3
    );
    assert_eq!(
        function.get_attribute(cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT).unwrap(),
        100
    );
}

fn assert_fixed_exact_n64_optional_holders(ctx: &GpuCtx, compute_capability: (u32, u32)) {
    let ada_holder = ctx.kernels.fixed_sm89_f32_n64_copyplan.as_ref();
    let ada_reason = ctx.kernels.fixed_sm89_f32_n64_copyplan_rejection.as_ref();
    if ctx.kernels.compiler_identity().target.as_str() != "sm_89" {
        assert!(
            ada_holder.is_none(),
            "Ada-only exact kernel loaded on another target"
        );
        assert!(ada_reason.is_some(), "non-Ada rejection reason was lost");
    } else {
        let function =
            ada_holder.unwrap_or_else(|| panic!("Ada exact N64 missing: {ada_reason:?}"));
        assert!(ada_reason.is_none());
        assert_exact_n64_resources(function);
    }

    let sm120_holder = ctx.kernels.fixed_sm120_f32_n64_copyplan.as_ref();
    let sm120_reason = ctx.kernels.fixed_sm120_f32_n64_copyplan_rejection.as_ref();
    let owns_sm120 = ctx.kernels.compiler_identity().target.as_str() == "compute_120"
        && compute_capability == (12, 0);
    if owns_sm120 {
        let function =
            sm120_holder.unwrap_or_else(|| panic!("SM120 exact N64 missing: {sm120_reason:?}"));
        assert!(sm120_reason.is_none());
        assert_exact_n64_resources(function);
    } else {
        assert!(
            sm120_holder.is_none(),
            "SM120 exact kernel loaded on a foreign device"
        );
        assert!(
            sm120_reason.is_some(),
            "non-SM120 rejection reason was lost"
        );
    }

    let sliced_holder = ctx.kernels.fixed_sm120_f32_n64_sliced.as_ref();
    let sliced_reason = ctx.kernels.fixed_sm120_f32_n64_sliced_rejection.as_ref();
    let owns_sliced = ctx.kernels.compiler_identity().target.as_str() == "compute_120"
        && compute_capability == (12, 0);
    if owns_sliced {
        let function =
            sliced_holder.unwrap_or_else(|| panic!("SM120 sliced N64 missing: {sliced_reason:?}"));
        assert!(sliced_reason.is_none());
        assert_exact_n64_resources(function);
    } else {
        assert!(
            sliced_holder.is_none(),
            "SM120 sliced kernel loaded on a foreign device"
        );
        assert!(
            sliced_reason.is_some(),
            "non-SM120 rejection reason was lost"
        );
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
    let _cuda_cache_guard = CudaCacheEnvGuard(std::env::var_os("CUDA_CACHE_DISABLE"));
    #[cfg(target_os = "linux")]
    unsafe {
        std::env::set_var("MAMBA_RS_KERNEL_CACHE", root.path().join("cache"));
        std::env::set_var("CUDA_CACHE_DISABLE", "1");
    }

    let device = GpuDevice::new(0).expect("CUDA device");
    let first = GpuCtx::new(&device).expect("first context");
    assert_fixed_exact_n64_optional_holders(&first, device.compute_capability);
    let first_artifacts = first.kernels.artifact_set_identity();
    #[cfg(target_os = "linux")]
    let active_artifacts = active_artifacts(first_artifacts);
    #[cfg(target_os = "linux")]
    let cached_entries = cache_entries(
        &root.path().join("cache"),
        "mamba-kernels-v1-",
        active_artifacts.len(),
    );
    #[cfg(target_os = "linux")]
    let cached_before: Vec<_> = cached_entries
        .iter()
        .map(|path| (path.clone(), std::fs::metadata(path).unwrap()))
        .collect();
    #[cfg(target_os = "linux")]
    let cached_payloads: Vec<_> = active_artifacts
        .iter()
        .map(|artifact| (*artifact, artifact_payload(&cached_entries, *artifact)))
        .collect();
    let second = GpuCtx::new(&device).expect("second context");
    assert_fixed_exact_n64_optional_holders(&second, device.compute_capability);
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
        let exact_n64_entries: std::collections::BTreeSet<_> = fixed_entries
            .iter()
            .filter(|name| name.starts_with("gemm_bi_nn_fixed_sm89_f32_n64_copyplan"))
            .map(String::as_str)
            .collect();
        let expected_exact_n64 = if first.kernels.compiler_identity().target.as_str() == "sm_89" {
            std::collections::BTreeSet::from(["gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1"])
        } else {
            std::collections::BTreeSet::new()
        };
        assert_eq!(
            exact_n64_entries, expected_exact_n64,
            "cached exact N64 inventory"
        );
        let sm120_exact_n64_entries: std::collections::BTreeSet<_> = fixed_entries
            .iter()
            .filter(|name| name.starts_with("gemm_bi_nn_fixed_sm120_f32_n64_copyplan"))
            .map(String::as_str)
            .collect();
        let expected_sm120_exact_n64 =
            if first.kernels.compiler_identity().target.as_str() == "compute_120" {
                std::collections::BTreeSet::from(["gemm_bi_nn_fixed_sm120_f32_n64_copyplan_v1"])
            } else {
                std::collections::BTreeSet::new()
            };
        assert_eq!(
            sm120_exact_n64_entries, expected_sm120_exact_n64,
            "cached SM120 exact N64 inventory"
        );
        let sm120_sliced_entries: std::collections::BTreeSet<_> = fixed_entries
            .iter()
            .filter(|name| name.starts_with("gemm_bi_nn_fixed_sm120_f32_n64_sliced"))
            .map(String::as_str)
            .collect();
        let expected_sm120_sliced =
            if first.kernels.compiler_identity().target.as_str() == "compute_120" {
                std::collections::BTreeSet::from(["gemm_bi_nn_fixed_sm120_f32_n64_sliced_v1"])
            } else {
                std::collections::BTreeSet::new()
            };
        assert_eq!(
            sm120_sliced_entries, expected_sm120_sliced,
            "cached SM120 sliced N64 inventory"
        );
        assert!(
            fixed_entries.iter().all(|name| !name.starts_with("sgemm_")),
            "Fixed unexpectedly exports legacy sgemm symbols: {fixed_entries:?}"
        );
        let fixed_tf32_entries: std::collections::BTreeSet<_> = fixed_entries
            .iter()
            .filter(|name| name.contains("_tf32_v1_"))
            .cloned()
            .collect();
        let expected_fixed_tf32 =
            expected_fixed_tf32_entries(first.kernels.compiler_identity().target.as_str());
        assert_eq!(fixed_tf32_entries, expected_fixed_tf32);

        let scalar_entries = ptx_entries(&artifact_payload(
            &cached_entries,
            first_artifacts.triad_scalar,
        ));
        let expected_scalar = expected_scalar_entries();
        assert_eq!(
            SCALAR_ENTRIES.len(),
            expected_scalar.len(),
            "scalar fixture contains duplicate entries"
        );
        assert_eq!(scalar_entries, expected_scalar);

        let sm80_entries = ptx_entries(&artifact_payload(
            &cached_entries,
            first_artifacts.triad_sm80,
        ));
        let sm80_target = first.kernels.compiler_identity().target;
        let expected_sm80 = expected_sm80_entries(sm80_target.as_str());
        assert_eq!(
            expected_sm80.len(),
            SM80_TYPED_ENTRIES.len()
                + if sm80_target_composes_streamk(sm80_target.as_str()) {
                    SM80_STREAMK_ENTRIES.len()
                        + SM80_TF32_WIDE_ROUTE_SPECS.len()
                        + SM80_TN_SPLITK_ENTRIES.len()
                } else {
                    0
                }
                + SM80_SPLITK_ENTRIES.len()
                + SM80_TF32_ROUTE_SPECS.len(),
            "SM80 fixture contains duplicate entries"
        );
        assert_eq!(sm80_entries, expected_sm80);

        if let Some(specialized) = first_artifacts.specialized
            && specialized.module_kind == ModuleKind::TriadSm120
        {
            let specialized_entries = ptx_entries(&artifact_payload(&cached_entries, specialized));
            let expected_specialized: std::collections::BTreeSet<_> = SM120_KERNEL_SPECS
                .iter()
                .chain(SM120_STREAMK_KERNEL_SPECS.iter())
                .map(|spec| spec.symbol.to_string())
                .chain(
                    SM120_TF32_ROUTE_SPECS
                        .iter()
                        .map(|spec| spec.symbol.to_string()),
                )
                .collect();
            assert_eq!(specialized_entries, expected_specialized);
        }
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
        let second_entries =
            cache_entries(&second_cache, "mamba-kernels-v1-", active_artifacts.len());
        for artifact in &active_artifacts {
            assert_eq!(
                artifact_payload(&cached_entries, *artifact),
                artifact_payload(&second_entries, *artifact),
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
        let original_scalar_payload = cached_payloads
            .iter()
            .find_map(|(artifact, payload)| {
                (artifact.module_kind == ModuleKind::TriadScalar).then(|| payload.clone())
            })
            .expect("original Scalar cache payload");
        let mut scalar_payload = original_scalar_payload.clone();
        let needle = b".entry gemm_bi_nn(";
        let replacement = b".entry gemm_bi_nx(";
        let offsets: Vec<_> = scalar_payload
            .windows(needle.len())
            .enumerate()
            .filter_map(|(offset, value)| (value == needle).then_some(offset))
            .collect();
        assert_eq!(offsets.len(), 1, "one Scalar gemm_bi_nn PTX entry");
        let offset = offsets[0];
        scalar_payload[offset..offset + needle.len()].copy_from_slice(replacement);
        std::fs::write(
            &scalar_entry,
            CacheEnvelope::encode(
                first_artifacts.triad_scalar.compile_key,
                ArtifactKind::Ptx,
                &scalar_payload,
            ),
        )
        .expect("rewrite valid Scalar cache envelope");

        let healed =
            GpuCtx::new(&device).expect("invalid cached PTX must recompile trusted source");
        assert_eq!(healed.kernels.artifact_set_identity(), first_artifacts);
        let repaired_entries = cache_entries(
            &root.path().join("cache"),
            "mamba-kernels-v1-",
            active_artifacts.len(),
        );
        let repaired_scalar_entry =
            artifact_cache_entry(&repaired_entries, first_artifacts.triad_scalar);
        let repaired_scalar_payload =
            artifact_payload(&repaired_entries, first_artifacts.triad_scalar);
        assert_eq!(repaired_scalar_payload, original_scalar_payload);
        assert_ne!(repaired_scalar_payload, scalar_payload);
        assert!(
            repaired_scalar_payload
                .windows(replacement.len())
                .all(|window| window != replacement),
            "repaired Scalar cache retained the poisoned symbol"
        );

        use std::os::unix::fs::MetadataExt;
        let original_scalar_metadata = cached_before
            .iter()
            .find_map(|(path, metadata)| (path == &scalar_entry).then_some(metadata))
            .expect("original Scalar cache metadata");
        let repaired_scalar_metadata = std::fs::metadata(&repaired_scalar_entry).unwrap();
        assert_ne!(
            repaired_scalar_metadata.ino(),
            original_scalar_metadata.ino(),
            "self-heal must atomically replace the poisoned Scalar envelope"
        );

        for artifact in &active_artifacts {
            if artifact.module_kind == ModuleKind::TriadScalar {
                continue;
            }
            let original_entry = artifact_cache_entry(&cached_entries, *artifact);
            let repaired_entry = artifact_cache_entry(&repaired_entries, *artifact);
            assert_eq!(original_entry, repaired_entry);
            let original_payload = cached_payloads
                .iter()
                .find_map(|(cached_artifact, payload)| {
                    (cached_artifact == artifact).then_some(payload)
                })
                .expect("original active cache payload");
            assert_eq!(
                &artifact_payload(&repaired_entries, *artifact),
                original_payload
            );
            let original_metadata = cached_before
                .iter()
                .find_map(|(path, metadata)| (path == &original_entry).then_some(metadata))
                .expect("original active cache metadata");
            let repaired_metadata = std::fs::metadata(&repaired_entry).unwrap();
            assert_eq!(repaired_metadata.ino(), original_metadata.ino());
            assert_eq!(
                repaired_metadata.modified().unwrap(),
                original_metadata.modified().unwrap()
            );
        }
    }
}
