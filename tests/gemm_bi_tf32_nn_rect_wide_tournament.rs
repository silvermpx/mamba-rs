use std::collections::BTreeSet;

#[cfg(feature = "cuda")]
mod common;

const DIMS: (usize, usize, usize) = (512, 3_072, 768);
const OFFICIAL_WINDOWS: usize = 101;
const MIN_PRODUCTION_P05_SPEEDUP: f64 = 1.20;
const MIN_PRODUCTION_P50_SPEEDUP: f64 = 1.25;
const PRODUCTION_PARITY_MIN_P05: f64 = 0.97;
const PRODUCTION_PARITY_MIN_P50: f64 = 0.985;
const PRODUCTION_PARITY_MAX_P50: f64 = 1.015;
const PRODUCTION_PARITY_MAX_P95: f64 = 1.03;
const DIRECT_GUARD_ELEMENTS: usize = 32;
const DIRECT_GUARD_BITS: u32 = 0x7fc1_5247;

fn guarded_active_bits(
    snapshot: &[f32],
    active_len: usize,
    expected_read_only: Option<&[f32]>,
    label: &str,
) -> Result<Vec<u32>, String> {
    let expected_len = active_len
        .checked_add(2 * DIRECT_GUARD_ELEMENTS)
        .ok_or_else(|| format!("{label} guarded extent overflow"))?;
    if snapshot.len() != expected_len {
        return Err(format!(
            "{label} guarded length changed: expected={expected_len} actual={}",
            snapshot.len()
        ));
    }
    for (side, guards) in [
        ("prefix", &snapshot[..DIRECT_GUARD_ELEMENTS]),
        ("suffix", &snapshot[DIRECT_GUARD_ELEMENTS + active_len..]),
    ] {
        if let Some((index, value)) = guards
            .iter()
            .enumerate()
            .find(|(_, value)| value.to_bits() != DIRECT_GUARD_BITS)
        {
            return Err(format!(
                "{label} {side} guard changed at {index}: {:08x}",
                value.to_bits()
            ));
        }
    }
    let active = &snapshot[DIRECT_GUARD_ELEMENTS..DIRECT_GUARD_ELEMENTS + active_len];
    if let Some(expected) = expected_read_only {
        if expected.len() != active_len {
            return Err(format!(
                "{label} read-only expectation length changed: expected={} active={active_len}",
                expected.len()
            ));
        }
        if let Some(index) = active
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            return Err(format!("{label} read-only active value changed at {index}"));
        }
    }
    Ok(active.iter().map(|value| value.to_bits()).collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CandidateSpec {
    symbol: &'static str,
    tile: (u32, u32),
    bk: u32,
    stages: u8,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    dynamic_shared_bytes: u32,
    minimum_occupancy: u32,
}

const CANDIDATES: [CandidateSpec; 5] = [
    CandidateSpec {
        symbol: "gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2",
        tile: (64, 64),
        bk: 32,
        stages: 2,
        grid: (96, 1, 1),
        block: (128, 1, 1),
        dynamic_shared_bytes: 32_896,
        minimum_occupancy: 3,
    },
    CandidateSpec {
        symbol: "gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2",
        tile: (64, 128),
        bk: 32,
        stages: 2,
        grid: (48, 1, 1),
        block: (256, 1, 1),
        dynamic_shared_bytes: 49_280,
        minimum_occupancy: 2,
    },
    CandidateSpec {
        symbol: "gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3",
        tile: (64, 128),
        bk: 32,
        stages: 3,
        grid: (48, 1, 1),
        block: (256, 1, 1),
        dynamic_shared_bytes: 73_856,
        minimum_occupancy: 1,
    },
    CandidateSpec {
        symbol: "gemm_bi_nn_sm120_tma_mma_tf32_v1_m128n64_bk32_s2",
        tile: (128, 64),
        bk: 32,
        stages: 2,
        grid: (48, 1, 1),
        block: (256, 1, 1),
        dynamic_shared_bytes: 49_280,
        minimum_occupancy: 2,
    },
    CandidateSpec {
        symbol: "gemm_bi_nn_sm120_tma_mma_tf32_v1_m128n64_bk32_s3",
        tile: (128, 64),
        bk: 32,
        stages: 3,
        grid: (48, 1, 1),
        block: (256, 1, 1),
        dynamic_shared_bytes: 73_856,
        minimum_occupancy: 1,
    },
];

const WINNER: CandidateSpec = CANDIDATES[0];

const fn candidate_specs() -> &'static [CandidateSpec; 5] {
    &CANDIDATES
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiteralTmaMapKey {
    operand: &'static str,
    global_dimensions: [u64; 2],
    outer_byte_stride: u64,
    box_dimensions: [u32; 2],
    base_offset_bytes: u64,
    format: &'static str,
}

fn literal_tma_map_keys(spec: CandidateSpec) -> Result<[LiteralTmaMapKey; 2], String> {
    let (m, k, n) = DIMS;
    let byte_stride = |elements: usize, name: &str| {
        u64::try_from(
            elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| format!("{name} byte stride overflow"))?,
        )
        .map_err(|_| format!("{name} byte stride exceeds u64"))
    };
    Ok([
        LiteralTmaMapKey {
            operand: "A",
            global_dimensions: [k as u64, m as u64],
            outer_byte_stride: byte_stride(k, "A")?,
            box_dimensions: [spec.bk, spec.tile.0],
            base_offset_bytes: 0,
            format: "uint32-v1",
        },
        LiteralTmaMapKey {
            operand: "B",
            global_dimensions: [n as u64, k as u64],
            outer_byte_stride: byte_stride(n, "B")?,
            box_dimensions: [spec.bk, spec.bk],
            base_offset_bytes: 0,
            format: "uint32-v1",
        },
    ])
}

fn literal_row_major_oracle() -> ([f32; 12], [f32; 8], [f32; 6]) {
    let a = [
        1.0_f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0,
    ];
    let b = [5.0_f32, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0];
    let mut output = [0.0_f32; 6];
    for row in 0..3 {
        for column in 0..2 {
            for reduction in 0..4 {
                output[row * 2 + column] = a[row * 4 + reduction]
                    .mul_add(b[reduction * 2 + column], output[row * 2 + column]);
            }
        }
    }
    (a, b, output)
}

fn row_major_geometry(dims: (usize, usize, usize)) -> Result<(usize, usize, usize), String> {
    let (m, k, n) = dims;
    if m == 0 || k == 0 || n == 0 {
        return Err(format!("NN dimensions must be positive: {dims:?}"));
    }
    m.checked_mul(n)
        .and_then(|_| m.checked_mul(k))
        .and_then(|_| k.checked_mul(n))
        .ok_or_else(|| format!("NN dimensions overflow: {dims:?}"))?;
    Ok((k, n, n))
}

fn cublas_geometry(dims: (usize, usize, usize)) -> Result<(i32, i32, i32, i32, i32, i32), String> {
    row_major_geometry(dims)?;
    let (m, k, n) = dims;
    let to_i32 = |value: usize, label: &str| {
        i32::try_from(value).map_err(|_| format!("cuBLAS {label} exceeds i32: {value}"))
    };
    Ok((
        to_i32(n, "m")?,
        to_i32(m, "n")?,
        to_i32(k, "k")?,
        to_i32(n, "lda")?,
        to_i32(k, "ldb")?,
        to_i32(n, "ldc")?,
    ))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CandidateIdentity {
    symbol: String,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    dynamic_shared_bytes: u32,
    arguments_digest: [u8; 32],
    request_identity_digest: [u8; 32],
    launch_digest: [u8; 32],
}

fn expected_identity(spec: CandidateSpec) -> CandidateIdentity {
    CandidateIdentity {
        symbol: spec.symbol.to_owned(),
        grid: spec.grid,
        block: spec.block,
        dynamic_shared_bytes: spec.dynamic_shared_bytes,
        arguments_digest: [1; 32],
        request_identity_digest: [2; 32],
        launch_digest: [3; 32],
    }
}

fn validate_candidate_identity(
    spec: CandidateSpec,
    identity: &CandidateIdentity,
) -> Result<(), String> {
    if identity.symbol != spec.symbol
        || identity.grid != spec.grid
        || identity.block != spec.block
        || identity.dynamic_shared_bytes != spec.dynamic_shared_bytes
    {
        return Err(format!(
            "candidate physical identity changed: spec={spec:?} actual={identity:?}"
        ));
    }
    for (label, digest) in [
        ("arguments", identity.arguments_digest),
        ("request identity", identity.request_identity_digest),
        ("launch", identity.launch_digest),
    ] {
        if digest == [0; 32] {
            return Err(format!("candidate {label} digest is zero"));
        }
    }
    if identity.arguments_digest == identity.request_identity_digest
        || identity.arguments_digest == identity.launch_digest
        || identity.request_identity_digest == identity.launch_digest
    {
        return Err("candidate physical identity digests are not domain-distinct".into());
    }
    Ok(())
}

fn percentile(values: &[f64], fraction: f64) -> Result<f64, String> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || !(fraction > 0.0 && fraction <= 1.0)
    {
        return Err("percentile requires positive finite samples and a unit fraction".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Ok(sorted[index])
}

fn validate_official_gate(windows: usize, p05: f64, p50: f64) -> Result<(), String> {
    if windows < OFFICIAL_WINDOWS {
        return Err(format!(
            "official timing requires at least {OFFICIAL_WINDOWS} windows, found {windows}"
        ));
    }
    if [p05, p50]
        .into_iter()
        .any(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(format!("invalid production speedups p05={p05} p50={p50}"));
    }
    if p05 < MIN_PRODUCTION_P05_SPEEDUP || p50 < MIN_PRODUCTION_P50_SPEEDUP {
        return Err(format!(
            "production speedup failed: p05={p05:.9} p50={p50:.9}"
        ));
    }
    Ok(())
}

fn validate_post_promotion_speedup(windows: usize, p05: f64, p50: f64) -> Result<(), String> {
    validate_official_gate(windows, p05, p50)
}

fn validate_post_promotion_parity(
    windows: usize,
    p05: f64,
    p50: f64,
    p95: f64,
) -> Result<(), String> {
    if windows < OFFICIAL_WINDOWS {
        return Err(format!(
            "post-promotion parity requires at least {OFFICIAL_WINDOWS} windows, found {windows}"
        ));
    }
    if [p05, p50, p95]
        .into_iter()
        .any(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(format!(
            "invalid post-promotion parity p05={p05} p50={p50} p95={p95}"
        ));
    }
    if p05 < PRODUCTION_PARITY_MIN_P05
        || !(PRODUCTION_PARITY_MIN_P50..=PRODUCTION_PARITY_MAX_P50).contains(&p50)
        || p95 > PRODUCTION_PARITY_MAX_P95
    {
        return Err(format!(
            "production/forced-winner parity failed: p05={p05:.9} p50={p50:.9} p95={p95:.9}"
        ));
    }
    Ok(())
}

#[test]
fn rect_wide_candidates_have_literal_row_major_geometry() {
    let candidates = candidate_specs();
    assert_eq!(candidates.len(), 5);
    assert_eq!(WINNER, candidates[0]);
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| {
                (
                    candidate.symbol,
                    candidate.tile,
                    candidate.stages,
                    candidate.grid,
                    candidate.block,
                    candidate.dynamic_shared_bytes,
                    candidate.minimum_occupancy,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (
                "gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2",
                (64, 64),
                2,
                (96, 1, 1),
                (128, 1, 1),
                32_896,
                3,
            ),
            (
                "gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2",
                (64, 128),
                2,
                (48, 1, 1),
                (256, 1, 1),
                49_280,
                2,
            ),
            (
                "gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3",
                (64, 128),
                3,
                (48, 1, 1),
                (256, 1, 1),
                73_856,
                1,
            ),
            (
                "gemm_bi_nn_sm120_tma_mma_tf32_v1_m128n64_bk32_s2",
                (128, 64),
                2,
                (48, 1, 1),
                (256, 1, 1),
                49_280,
                2,
            ),
            (
                "gemm_bi_nn_sm120_tma_mma_tf32_v1_m128n64_bk32_s3",
                (128, 64),
                3,
                (48, 1, 1),
                (256, 1, 1),
                73_856,
                1,
            ),
        ]
    );
    assert!(candidates.iter().all(|candidate| candidate.bk == 32));
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.symbol)
            .collect::<BTreeSet<_>>()
            .len(),
        candidates.len()
    );
}

#[test]
fn row_major_nn_and_cublas_geometry_are_not_transposed() {
    assert_eq!(row_major_geometry(DIMS).unwrap(), (3_072, 768, 768));
    assert_eq!(
        cublas_geometry(DIMS).unwrap(),
        (768, 512, 3_072, 768, 3_072, 768)
    );
    for invalid in [(0, 3_072, 768), (512, 0, 768), (512, 3_072, 0)] {
        assert!(row_major_geometry(invalid).is_err());
        assert!(cublas_geometry(invalid).is_err());
    }
}

#[test]
fn live_correctness_presizes_before_the_first_capture() {
    let source = include_str!("gemm_bi_tf32_nn_rect_wide_tournament.rs");
    let start = source
        .rfind("fn rect_wide_candidates_are_exact_and_graph_stable()")
        .expect("live correctness test");
    let end = source[start..]
        .find("fn screen_candidate(")
        .map(|offset| start + offset)
        .expect("screening helper after live correctness test");
    let body = &source[start..end];
    let presize = body
        .find("presize_tournament_context(&reference_ctx)?;")
        .expect("live correctness must pre-size its context");
    let first_capture = body
        .find("validate_exact_cohort_witness(&reference_ctx)?;")
        .expect("live correctness cohort witness");
    assert!(
        presize < first_capture,
        "live correctness must pre-size every qualification before the witness freezes capture capacity"
    );
}

#[test]
fn literal_tma_keys_and_small_nn_oracle_are_absolute() {
    for spec in candidate_specs() {
        let keys = literal_tma_map_keys(*spec).unwrap();
        assert_eq!(
            keys,
            [
                LiteralTmaMapKey {
                    operand: "A",
                    global_dimensions: [3_072, 512],
                    outer_byte_stride: 12_288,
                    box_dimensions: [32, spec.tile.0],
                    base_offset_bytes: 0,
                    format: "uint32-v1",
                },
                LiteralTmaMapKey {
                    operand: "B",
                    global_dimensions: [768, 3_072],
                    outer_byte_stride: 3_072,
                    box_dimensions: [32, 32],
                    base_offset_bytes: 0,
                    format: "uint32-v1",
                },
            ]
        );
    }
    let (a, b, actual) = literal_row_major_oracle();
    assert_eq!(a.len(), 3 * 4);
    assert_eq!(b.len(), 4 * 2);
    assert_eq!(
        actual.map(f32::to_bits),
        [5.0, 7.0, 22.0, 26.0, 51.0, 57.0].map(f32::to_bits)
    );
    assert_ne!(actual[1].to_bits(), 11.0_f32.to_bits());
}

#[test]
fn candidate_identity_rejects_every_physical_mutation() {
    let expected = expected_identity(candidate_specs()[0]);
    validate_candidate_identity(candidate_specs()[0], &expected).unwrap();
    for mutate in [
        |identity: &mut CandidateIdentity| identity.grid.0 += 1,
        |identity: &mut CandidateIdentity| identity.block.0 += 1,
        |identity: &mut CandidateIdentity| identity.dynamic_shared_bytes += 4,
        |identity: &mut CandidateIdentity| identity.arguments_digest = [0; 32],
        |identity: &mut CandidateIdentity| identity.request_identity_digest = [0; 32],
        |identity: &mut CandidateIdentity| identity.launch_digest = [0; 32],
        |identity: &mut CandidateIdentity| {
            identity.request_identity_digest = identity.arguments_digest
        },
        |identity: &mut CandidateIdentity| identity.launch_digest = identity.arguments_digest,
        |identity: &mut CandidateIdentity| {
            identity.launch_digest = identity.request_identity_digest
        },
    ] {
        let mut changed = expected.clone();
        mutate(&mut changed);
        assert!(validate_candidate_identity(candidate_specs()[0], &changed).is_err());
    }
    let mut changed = expected;
    changed.symbol.push_str("_wrong");
    assert!(validate_candidate_identity(candidate_specs()[0], &changed).is_err());
}

#[test]
fn performance_statistics_fail_closed() {
    let positive = (1..=101).map(|value| value as f64).collect::<Vec<_>>();
    assert_eq!(percentile(&positive, 0.05).unwrap(), 6.0);
    assert_eq!(percentile(&positive, 0.50).unwrap(), 51.0);
    assert!(percentile(&positive, 0.0).is_err());
    for values in [
        vec![],
        vec![0.0],
        vec![-1.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
    ] {
        assert!(percentile(&values, 0.50).is_err());
    }
    for windows in [0, 1, 20, 100] {
        assert!(validate_official_gate(windows, 1.20, 1.25).is_err());
    }
    assert!(validate_official_gate(101, 1.20, 1.25).is_ok());
    assert!(validate_official_gate(101, 1.199_999, 1.25).is_err());
    assert!(validate_official_gate(101, 1.20, 1.249_999).is_err());
    assert!(validate_official_gate(101, f64::NAN, 2.0).is_err());
}

#[test]
fn post_promotion_speedup_and_parity_gates_fail_closed() {
    assert!(validate_post_promotion_speedup(101, 1.20, 1.25).is_ok());
    assert!(validate_post_promotion_speedup(101, 1.199_999, 1.25).is_err());
    assert!(validate_post_promotion_speedup(101, 1.20, 1.249_999).is_err());
    assert!(validate_post_promotion_speedup(100, 2.0, 2.0).is_err());

    assert!(validate_post_promotion_parity(101, 0.97, 1.0, 1.03).is_ok());
    for (p05, p50, p95) in [
        (0.969_999, 1.0, 1.02),
        (0.99, 0.984_999, 1.02),
        (0.99, 1.015_001, 1.02),
        (0.99, 1.0, 1.030_001),
        (f64::NAN, 1.0, 1.01),
        (0.99, f64::INFINITY, 1.01),
        (0.99, 1.0, 0.0),
    ] {
        assert!(validate_post_promotion_parity(101, p05, p50, p95).is_err());
    }
    assert!(validate_post_promotion_parity(100, 0.99, 1.0, 1.01).is_err());
}

#[test]
fn guarded_snapshot_rejects_prefix_suffix_and_read_only_mutations() {
    let active = [1.0_f32, -2.0, 3.5];
    let mut snapshot =
        vec![f32::from_bits(DIRECT_GUARD_BITS); active.len() + 2 * DIRECT_GUARD_ELEMENTS];
    snapshot[DIRECT_GUARD_ELEMENTS..DIRECT_GUARD_ELEMENTS + active.len()].copy_from_slice(&active);
    assert_eq!(
        guarded_active_bits(&snapshot, active.len(), Some(&active), "A").unwrap(),
        active.map(f32::to_bits)
    );

    for index in [0, DIRECT_GUARD_ELEMENTS + active.len()] {
        let mut changed = snapshot.clone();
        changed[index] = 0.0;
        assert!(guarded_active_bits(&changed, active.len(), Some(&active), "A").is_err());
    }
    let mut changed = snapshot;
    changed[DIRECT_GUARD_ELEMENTS + 1] = 9.0;
    assert!(guarded_active_bits(&changed, active.len(), Some(&active), "A").is_err());

    assert!(guarded_active_bits(&[], active.len(), Some(&active), "A").is_err());
    assert!(
        guarded_active_bits(
            &vec![f32::from_bits(DIRECT_GUARD_BITS); 2 * DIRECT_GUARD_ELEMENTS],
            0,
            Some(&active),
            "A",
        )
        .is_err()
    );
}

#[test]
fn production_sources_own_every_candidate_and_a_bounded_abi() {
    let source = include_str!("../kernels/gemm_bi_triad/sm120.cu");
    for candidate in candidate_specs() {
        assert_eq!(source.matches(candidate.symbol).count(), 2);
    }
    let macro_body = source
        .split("#define SM120_DEFINE_TF32_KERNEL")
        .nth(1)
        .expect("production TF32 kernel macro");
    let signature = macro_body
        .split("void NAME(")
        .nth(1)
        .expect("production TF32 kernel entry signature")
        .split(") {")
        .next()
        .expect("production TF32 kernel signature");
    assert_eq!(
        signature.matches(',').count(),
        4,
        "TF32 ABI must remain five arguments"
    );
    for helper in [
        "sm120_tf32_entry",
        "sm120_tf32_kernel",
        "sm120_tf32_produce_stage",
        "sm120_tf32_issue_stage",
        "sm120_tf32_load_issue",
        "sm120_tf32_store",
        "sm120_init_barrier",
        "sm120_wait_barrier",
        "sm120_arrive_empty",
    ] {
        assert!(
            source.contains(helper),
            "actual candidate helper closure lost {helper}"
        );
    }
    assert!(
        !source.to_ascii_lowercase().contains("atomic")
            && !source.contains(" atom.")
            && !source.contains("\tatom.")
            && !source.contains(" red.")
            && !source.contains("\tred."),
        "production SM120 source contains an atomic operation"
    );

    let contract = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/contract.rs");
    for literal in [
        "ResolvedGemmOp::Nn => [",
        "box_dimensions: [spec.map_bk, spec.tile.0]",
        "box_dimensions: [spec.map_bk, spec.map_bk]",
        "Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(_) => Tf32TensorMapFormat::Uint32V1",
    ] {
        assert!(
            contract.contains(literal),
            "production TMA planner lost literal contract {literal}"
        );
    }
}

#[cfg(feature = "cuda")]
mod cuda_tournament {
    use super::*;
    use common::gpu_quiet::QuietGpu;
    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
        sys,
    };
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationF32Epilogue, PhysicalQualificationRequest, PhysicalQualificationRoute,
        QualifiedPhysicalLaunch, Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages,
        Tf32PortableTile, Tf32Sm120Route, Tf32Sm120Stages, Tf32Sm120Tile,
        presize_physical_qualification_suite, qualify_physical_launch, tf32_kernel_spec,
    };
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, FramedSha256, ModuleKind,
        NUMERIC_ABI_REVISION, ResolvedGemmOp, SCHEDULE_REVISION, digest_hex,
    };
    use std::ffi::c_void;
    use std::sync::Arc;

    const CORPUS_SALT: u64 = 0x5123_0727_6801;
    const SCREENING_WINDOWS: usize = 21;
    const TARGET_WINDOW_MS: f64 = 10.0;
    const MAX_WINDOW_ITERATIONS: usize = 16_384;
    const WINNER_MIN_P05: f64 = 1.00;
    const WINNER_MIN_P50: f64 = 1.01;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Path {
        Eager,
        Graph,
    }

    impl Path {
        const fn name(self) -> &'static str {
            match self {
                Self::Eager => "eager",
                Self::Graph => "graph",
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Order {
        Abba,
        Baab,
    }

    impl Order {
        const fn name(self) -> &'static str {
            match self {
                Self::Abba => "ABBA",
                Self::Baab => "BAAB",
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum QualificationArm {
        ScalarBaseline,
        Production,
        Forced(CandidateSpec),
    }

    impl QualificationArm {
        const fn name(self) -> &'static str {
            match self {
                Self::ScalarBaseline => "scalar-policy-baseline",
                Self::Production => "production-tag31",
                Self::Forced(spec) => spec.symbol,
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum PairGate {
        ScalarToProductionSpeedup,
        ProductionToWinnerParity,
        RunnerToWinner,
    }

    #[derive(Clone, Copy, Debug)]
    struct TimingStats {
        baseline_p50_us: f64,
        candidate_p50_us: f64,
        speedup_p05: f64,
        speedup_p50: f64,
        speedup_p95: f64,
    }

    fn epilogue() -> PhysicalQualificationF32Epilogue {
        PhysicalQualificationF32Epilogue::new(1.0, 0.0, false)
    }

    fn configure(device: &GpuDevice, policy: F32TriadPolicy) -> Result<GpuCtx, String> {
        let ctx = GpuCtx::new(device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_f32_triad_policy(policy);
        Ok(ctx)
    }

    fn candidate_route(spec: CandidateSpec) -> Tf32PhysicalRoute {
        let tile = match spec.tile {
            (64, 64) => Tf32Sm120Tile::M64N64,
            (64, 128) => Tf32Sm120Tile::M64N128,
            (128, 64) => Tf32Sm120Tile::M128N64,
            _ => unreachable!("host geometry gate admits only production SM120 TF32 tiles"),
        };
        let stages = match spec.stages {
            2 => Tf32Sm120Stages::S2,
            3 => Tf32Sm120Stages::S3,
            _ => unreachable!("host geometry gate admits only production SM120 TF32 stages"),
        };
        Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(Tf32Sm120Route { tile, stages })
    }

    fn literal_tma_map_digest(spec: CandidateSpec) -> Result<[u8; 32], String> {
        let keys = literal_tma_map_keys(spec)?;
        let mut digest = FramedSha256::new(b"tf32-nn-rect-wide-literal-tma-keys.v1")
            .required(b"logical-op", b"nn")
            .required(b"symbol", spec.symbol.as_bytes())
            .required(b"map-count", &(keys.len() as u64).to_le_bytes());
        for key in keys {
            digest = digest
                .required(b"operand", key.operand.as_bytes())
                .required(b"format", key.format.as_bytes())
                .required(
                    b"global-dimension-0",
                    &key.global_dimensions[0].to_le_bytes(),
                )
                .required(
                    b"global-dimension-1",
                    &key.global_dimensions[1].to_le_bytes(),
                )
                .required(b"outer-byte-stride", &key.outer_byte_stride.to_le_bytes())
                .required(b"box-dimension-0", &key.box_dimensions[0].to_le_bytes())
                .required(b"box-dimension-1", &key.box_dimensions[1].to_le_bytes())
                .required(b"base-offset-bytes", &key.base_offset_bytes.to_le_bytes());
        }
        Ok(digest.finish())
    }

    #[test]
    fn literal_tma_map_keys_have_frozen_route_digests() {
        let expected = [
            (
                CANDIDATES[0].symbol,
                "f2e5c9726eef9fbbc5827bbfc6c7144080d8508372af2d66c3460f5d07376df1",
            ),
            (
                CANDIDATES[1].symbol,
                "ec196d8d165a072745be9f4950430d518f2c41c82d90d7955aac0103a6fe1f1f",
            ),
            (
                CANDIDATES[2].symbol,
                "3951d5454a35641ce61cdf6659452a486708961de068e0de4be3effc72cfc97e",
            ),
            (
                CANDIDATES[3].symbol,
                "2ba767e5535f404c9e997ad02752029d5e88c766f2877ac78035db1000ae3a20",
            ),
            (
                CANDIDATES[4].symbol,
                "dfb1708a1ff4bd4231a46fbc6e44aa8cff05dbbd41d7f0db525091d341cc9ced",
            ),
        ];
        for (spec, (symbol, expected_hex)) in candidate_specs().iter().zip(expected) {
            assert_eq!(spec.symbol, symbol);
            assert_eq!(
                digest_hex(&literal_tma_map_digest(*spec).unwrap()),
                expected_hex
            );
        }
    }

    fn forced_request(spec: CandidateSpec) -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Nn,
            DIMS,
            PhysicalQualificationRoute::Tf32Forced(candidate_route(spec)),
            epilogue(),
        )
    }

    fn portable_reference_request() -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Nn,
            DIMS,
            PhysicalQualificationRoute::Tf32Forced(Tf32PhysicalRoute::MmaTf32RnaV1(
                Tf32PortableRoute {
                    tile: Tf32PortableTile::M128N64,
                    stages: Tf32PortableStages::S2,
                },
            )),
            epilogue(),
        )
    }

    fn production_request() -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Nn,
            DIMS,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::AllowDeterministicTf32V1),
            epilogue(),
        )
    }

    fn scalar_baseline_request() -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Nn,
            DIMS,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
            epilogue(),
        )
    }

    fn validate_exact_environment(ctx: &GpuCtx) -> Result<(), String> {
        let specialized = ctx
            .kernels
            .f32_triad_availability()
            .specialized
            .ok_or_else(|| {
                "rect-wide tournament requires the specialized TF32 module".to_string()
            })?;
        let artifact = specialized.artifact;
        let compiler = specialized.compiler;
        if specialized.module_kind != ModuleKind::TriadSm120
            || specialized.target.as_str() != "compute_120"
            || specialized.device.compute_capability != (12, 0)
            || specialized.device.multiprocessor_count != 170
            || specialized.device.target.as_str() != "sm_120"
            || specialized.device_caps.compute_capability != (12, 0)
            || specialized.device_caps.nvrtc_version != (13, 2)
            || specialized
                .device_caps
                .accepted_target
                .is_none_or(|target| target.as_str() != "compute_120")
            || specialized.device_caps.optin_shared_bytes != 101_376
            || !specialized.device_caps.tensor_map_access
            || compiler.target.as_str() != "compute_120"
            || compiler.nvrtc_version != (13, 2)
            || !compiler.nvrtc_library_known
            || compiler.nvrtc_library_domain == [0; 32]
            || compiler.source_digest == [0; 32]
            || compiler.invocation_digest == [0; 32]
            || compiler.header_manifest_digest == [0; 32]
            || compiler.output_kind != ArtifactKind::Ptx
            || compiler.composer_revision != COMPOSER_REVISION
            || compiler.compiler_revision != COMPILER_REVISION
            || compiler.numeric_abi_revision != NUMERIC_ABI_REVISION
            || compiler.schedule_revision != SCHEDULE_REVISION
            || artifact.module_kind != ModuleKind::TriadSm120
            || artifact.artifact_kind != compiler.output_kind
            || artifact.compile_key != compiler.invocation_digest
            || artifact.compile_key == [0; 32]
            || artifact.artifact_digest == [0; 32]
        {
            return Err(format!(
                "rect-wide tournament requires the exact CUDA 13.2 SM120/170-SM TF32 artifact: specialized={specialized:?}"
            ));
        }
        Ok(())
    }

    fn validate_release_build() -> Result<(), String> {
        if cfg!(debug_assertions) {
            Err("rect-wide official timing requires --release".into())
        } else {
            Ok(())
        }
    }

    fn production_sm120_source() -> String {
        let fragments = [
            (
                "kernels/_typed_prelude.cuh",
                include_str!("../kernels/_typed_prelude.cuh"),
            ),
            (
                "kernels/gemm_bi_triad/contract.cuh",
                include_str!("../kernels/gemm_bi_triad/contract.cuh"),
            ),
            (
                "kernels/gemm_bi_triad/common.cuh",
                include_str!("../kernels/gemm_bi_triad/common.cuh"),
            ),
            (
                "kernels/gemm_bi_triad/epilogue.cuh",
                include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
            ),
            (
                "kernels/gemm_bi_triad/sm120.cu",
                include_str!("../kernels/gemm_bi_triad/sm120.cu"),
            ),
        ];
        let mut source = String::new();
        for (logical_name, fragment) in fragments {
            source.push_str("#line 1 \"");
            source.push_str(logical_name);
            source.push_str("\"\n");
            source.push_str(fragment);
            if !source.ends_with('\n') {
                source.push('\n');
            }
        }
        source
    }

    fn compile_sm120_source() -> Result<String, String> {
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            options: vec![
                "--fmad=true".to_owned(),
                "--extra-device-vectorization".to_owned(),
                "-DNDEBUG".to_owned(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        cudarc::nvrtc::compile_ptx_with_opts(production_sm120_source(), options)
            .map(|ptx| ptx.to_src())
            .map_err(|error| format!("compile production SM120 TF32 source: {error:?}"))
    }

    struct ResourceModule {
        _module: Arc<CudaModule>,
        functions: Vec<(CandidateSpec, CudaFunction)>,
    }

    impl ResourceModule {
        fn function(&self, spec: CandidateSpec) -> Result<&CudaFunction, String> {
            self.functions
                .iter()
                .find_map(|(candidate, function)| (*candidate == spec).then_some(function))
                .ok_or_else(|| format!("direct resource module lost {}", spec.symbol))
        }
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct DirectSm120Params {
        a_x: i32,
        a_y: i32,
        b_x: i32,
        b_y: i32,
        alpha: f32,
        beta: f32,
        m: i32,
        k: i32,
        n: i32,
        ldc: i32,
    }

    unsafe impl DeviceRepr for DirectSm120Params {}

    const _: () = {
        assert!(std::mem::size_of::<DirectSm120Params>() == 40);
        assert!(std::mem::align_of::<DirectSm120Params>() == 4);
    };

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    struct DirectTensorMap(sys::CUtensorMap);

    unsafe impl DeviceRepr for DirectTensorMap {}

    const _: () = {
        assert!(std::mem::size_of::<DirectTensorMap>() == std::mem::size_of::<sys::CUtensorMap>());
        assert!(
            std::mem::align_of::<DirectTensorMap>() == std::mem::align_of::<sys::CUtensorMap>()
        );
    };

    #[derive(Clone, Copy)]
    struct DirectTensorMaps {
        a: DirectTensorMap,
        b: DirectTensorMap,
    }

    struct DirectGuardedBuffer {
        buffer: GpuBuffer,
        initial_active: Vec<f32>,
        active_len: usize,
    }

    impl DirectGuardedBuffer {
        fn new(stream: &Arc<CudaStream>, active: Vec<f32>) -> Result<Self, String> {
            let active_len = active.len();
            let total = active_len
                .checked_add(2 * DIRECT_GUARD_ELEMENTS)
                .ok_or_else(|| "direct guarded allocation extent overflow".to_string())?;
            let mut contents = vec![f32::from_bits(DIRECT_GUARD_BITS); total];
            contents[DIRECT_GUARD_ELEMENTS..DIRECT_GUARD_ELEMENTS + active_len]
                .copy_from_slice(&active);
            Ok(Self {
                buffer: GpuBuffer::from_cpu(stream, &contents)?,
                initial_active: active,
                active_len,
            })
        }

        fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
            let mut contents = vec![
                f32::from_bits(DIRECT_GUARD_BITS);
                self.active_len + 2 * DIRECT_GUARD_ELEMENTS
            ];
            contents[DIRECT_GUARD_ELEMENTS..DIRECT_GUARD_ELEMENTS + self.active_len]
                .copy_from_slice(&self.initial_active);
            self.buffer.upload(stream, &contents)
        }

        fn active_ptr(&self, stream: &Arc<CudaStream>, label: &str) -> Result<u64, String> {
            let pointer = self.buffer.raw_ptr_at(stream, DIRECT_GUARD_ELEMENTS);
            if pointer == 0 || !pointer.is_multiple_of(128) {
                return Err(format!(
                    "{label} active pointer must be nonzero and 128-byte aligned: {pointer:#x}"
                ));
            }
            Ok(pointer)
        }

        fn output_bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
            let snapshot = self.buffer.to_cpu(stream)?;
            guarded_active_bits(&snapshot, self.active_len, None, label)
        }

        fn validate_read_only(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
            let snapshot = self.buffer.to_cpu(stream)?;
            guarded_active_bits(
                &snapshot,
                self.active_len,
                Some(&self.initial_active),
                label,
            )?;
            Ok(())
        }
    }

    struct DirectFixture {
        a: DirectGuardedBuffer,
        b: DirectGuardedBuffer,
        output: DirectGuardedBuffer,
        maps: DirectTensorMaps,
        params: DirectSm120Params,
        config: LaunchConfig,
    }

    struct DirectRuntime {
        stream: Arc<CudaStream>,
        module: ResourceModule,
    }

    fn load_resource_module(device: &GpuDevice) -> Result<ResourceModule, String> {
        let ptx = compile_sm120_source()?;
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|error| format!("load production SM120 TF32 resource module: {error:?}"))?;
        let mut functions = Vec::with_capacity(candidate_specs().len());
        for spec in candidate_specs() {
            let function = module.load_function(spec.symbol).map_err(|error| {
                format!("load {} for resource validation: {error:?}", spec.symbol)
            })?;
            function
                .set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    spec.dynamic_shared_bytes as i32,
                )
                .map_err(|error| {
                    format!(
                        "set {} dynamic shared resource contract: {error:?}",
                        spec.symbol
                    )
                })?;
            functions.push((*spec, function));
        }
        Ok(ResourceModule {
            _module: module,
            functions,
        })
    }

    fn new_direct_runtime(device: &GpuDevice) -> Result<DirectRuntime, String> {
        Ok(DirectRuntime {
            stream: device.fork_stream()?,
            module: load_resource_module(device)?,
        })
    }

    fn direct_tensor_map(
        pointer: u64,
        width: usize,
        rows: usize,
        stride: usize,
        box_dimensions: [u32; 2],
        label: &str,
    ) -> Result<DirectTensorMap, String> {
        if pointer == 0 || !pointer.is_multiple_of(128) {
            return Err(format!(
                "{label} TMA base must be nonzero and 128-byte aligned: {pointer:#x}"
            ));
        }
        if width == 0 || rows == 0 || stride < width {
            return Err(format!(
                "{label} invalid TMA geometry width={width} rows={rows} stride={stride}"
            ));
        }
        let byte_stride = stride
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("{label} TMA byte stride overflow"))?;
        if !byte_stride.is_multiple_of(16) {
            return Err(format!(
                "{label} TMA byte stride must be a multiple of 16: {byte_stride}"
            ));
        }
        let dimensions = [
            u64::try_from(width).map_err(|_| format!("{label} width exceeds u64"))?,
            u64::try_from(rows).map_err(|_| format!("{label} rows exceed u64"))?,
        ];
        let global_strides = [
            u64::try_from(byte_stride).map_err(|_| format!("{label} byte stride exceeds u64"))?
        ];
        let element_strides = [1_u32, 1_u32];
        let mut raw = std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed();
        let result = unsafe {
            sys::cuTensorMapEncodeTiled(
                raw.as_mut_ptr(),
                sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
                2,
                pointer as usize as *mut c_void,
                dimensions.as_ptr(),
                global_strides.as_ptr(),
                box_dimensions.as_ptr(),
                element_strides.as_ptr(),
                sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("encode {label} direct tensor map: {result:?}"));
        }
        Ok(DirectTensorMap(unsafe { raw.assume_init() }))
    }

    fn new_direct_fixture(
        runtime: &DirectRuntime,
        spec: CandidateSpec,
        dims: (usize, usize, usize),
        strides: (usize, usize, usize),
        a_values: Vec<f32>,
        b_values: Vec<f32>,
        output_values: Vec<f32>,
    ) -> Result<DirectFixture, String> {
        let (m, k, n) = dims;
        let (lda, ldb, ldc) = strides;
        let expected_a = m
            .checked_mul(lda)
            .ok_or_else(|| "direct A extent overflow".to_string())?;
        let expected_b = k
            .checked_mul(ldb)
            .ok_or_else(|| "direct B extent overflow".to_string())?;
        let expected_output = m
            .checked_mul(ldc)
            .ok_or_else(|| "direct output extent overflow".to_string())?;
        if lda < k
            || ldb < n
            || ldc < n
            || a_values.len() != expected_a
            || b_values.len() != expected_b
            || output_values.len() != expected_output
        {
            return Err(format!(
                "invalid direct fixture dims={dims:?} strides={strides:?} lengths=({},{},{}) expected=({expected_a},{expected_b},{expected_output})",
                a_values.len(),
                b_values.len(),
                output_values.len()
            ));
        }
        let a = DirectGuardedBuffer::new(&runtime.stream, a_values)?;
        let b = DirectGuardedBuffer::new(&runtime.stream, b_values)?;
        let output = DirectGuardedBuffer::new(&runtime.stream, output_values)?;
        let a_pointer = a.active_ptr(&runtime.stream, "A")?;
        let b_pointer = b.active_ptr(&runtime.stream, "B")?;
        output.active_ptr(&runtime.stream, "output")?;
        let maps = DirectTensorMaps {
            a: direct_tensor_map(a_pointer, k, m, lda, [spec.bk, spec.tile.0], "A")?,
            b: direct_tensor_map(b_pointer, n, k, ldb, [spec.bk, spec.bk], "B")?,
        };
        let grid = m
            .div_ceil(spec.tile.0 as usize)
            .checked_mul(n.div_ceil(spec.tile.1 as usize))
            .ok_or_else(|| "direct grid overflow".to_string())?;
        let grid = u32::try_from(grid).map_err(|_| "direct grid exceeds u32".to_string())?;
        Ok(DirectFixture {
            a,
            b,
            output,
            maps,
            params: DirectSm120Params {
                a_x: 0,
                a_y: 0,
                b_x: 0,
                b_y: 0,
                alpha: 1.0,
                beta: 0.0,
                m: i32::try_from(m).map_err(|_| "direct M exceeds i32".to_string())?,
                k: i32::try_from(k).map_err(|_| "direct K exceeds i32".to_string())?,
                n: i32::try_from(n).map_err(|_| "direct N exceeds i32".to_string())?,
                ldc: i32::try_from(ldc).map_err(|_| "direct ldc exceeds i32".to_string())?,
            },
            config: LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: spec.block,
                shared_mem_bytes: spec.dynamic_shared_bytes,
            },
        })
    }

    fn launch_direct(
        runtime: &DirectRuntime,
        spec: CandidateSpec,
        fixture: &DirectFixture,
    ) -> Result<(), String> {
        let output = fixture.output.active_ptr(&runtime.stream, "output")?;
        let bias = 0_u64;
        let mut builder = runtime
            .stream
            .launch_builder(runtime.module.function(spec)?);
        builder.arg(&output);
        builder.arg(&fixture.maps.a);
        builder.arg(&fixture.maps.b);
        builder.arg(&bias);
        builder.arg(&fixture.params);
        unsafe { builder.launch(fixture.config) }
            .map(|_| ())
            .map_err(|error| format!("launch direct {}: {error:?}", spec.symbol))
    }

    fn capture_direct(
        runtime: &DirectRuntime,
        spec: CandidateSpec,
        fixture: &DirectFixture,
    ) -> Result<CudaGraph, String> {
        unsafe { capture_into_graph(&runtime.stream, || launch_direct(runtime, spec, fixture)) }
    }

    fn qualification_seeded_values(len: usize, salt: u64) -> Vec<f32> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ salt;
        (0..len)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let signed = ((state.wrapping_add(index as u64) % 4093) as i32) - 2046;
                signed as f32 / 1024.0
            })
            .collect()
    }

    fn validate_driver_resources(device: &GpuDevice) -> Result<(), String> {
        let resources = load_resource_module(device)?;
        for (spec, function) in &resources.functions {
            let registers = function
                .num_regs()
                .map_err(|error| format!("{} registers: {error:?}", spec.symbol))?;
            let local = function
                .local_size_bytes()
                .map_err(|error| format!("{} local bytes: {error:?}", spec.symbol))?;
            let static_shared = function
                .shared_size_bytes()
                .map_err(|error| format!("{} static shared bytes: {error:?}", spec.symbol))?;
            let max_threads = function
                .max_threads_per_block()
                .map_err(|error| format!("{} max threads: {error:?}", spec.symbol))?;
            let max_dynamic = function
                .get_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                )
                .map_err(|error| format!("{} max dynamic shared bytes: {error:?}", spec.symbol))?;
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(
                    spec.block.0 * spec.block.1 * spec.block.2,
                    spec.dynamic_shared_bytes as usize,
                    None,
                )
                .map_err(|error| format!("{} occupancy: {error:?}", spec.symbol))?;
            if registers <= 0
                || registers > 128
                || local != 0
                || static_shared != 0
                || max_threads < spec.block.0 as i32
                || max_dynamic < spec.dynamic_shared_bytes as i32
                || occupancy < spec.minimum_occupancy
            {
                return Err(format!(
                    "{} driver resource contract failed: regs={registers}/128 local={local}/0 static_shared={static_shared}/0 dynamic_shared={}/{max_dynamic} threads={}/{} occupancy={}/{}",
                    spec.symbol,
                    spec.dynamic_shared_bytes,
                    spec.block.0,
                    max_threads,
                    occupancy,
                    spec.minimum_occupancy
                ));
            }
            eprintln!(
                "tf32_nn_rect_wide resource symbol={} regs={} local_bytes={} static_shared_bytes={} dynamic_shared_bytes={} max_dynamic_shared_bytes={} threads={} max_threads={} occupancy={}",
                spec.symbol,
                registers,
                local,
                static_shared,
                spec.dynamic_shared_bytes,
                max_dynamic,
                spec.block.0,
                max_threads,
                occupancy
            );
        }
        Ok(())
    }

    fn metric_before(line: &str, suffix: &str) -> Option<u64> {
        let prefix = line.split_once(suffix)?.0;
        prefix
            .split(|byte: char| !byte.is_ascii_digit())
            .rfind(|field| !field.is_empty())?
            .parse()
            .ok()
    }

    fn validate_ptxas_entry(report: &str, spec: CandidateSpec) -> Result<(), String> {
        let marker = format!("Compiling entry function '{}'", spec.symbol);
        if report.matches(&marker).count() != 1 {
            return Err(format!(
                "{} requires exactly one ptxas entry record",
                spec.symbol
            ));
        }
        let block = report
            .split_once(&marker)
            .expect("count above proves the ptxas entry exists")
            .1
            .split("Compiling entry function '")
            .next()
            .expect("ptxas entry block");
        if !block.contains("0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads") {
            return Err(format!(
                "{} has nonzero or unreported stack/spills:\n{block}",
                spec.symbol
            ));
        }
        let usage = block
            .lines()
            .find(|line| line.contains("Used ") && line.contains(" registers"))
            .ok_or_else(|| format!("{} lost ptxas register usage", spec.symbol))?;
        let registers = metric_before(usage, " registers")
            .ok_or_else(|| format!("{} has malformed ptxas registers: {usage}", spec.symbol))?;
        let static_shared = metric_before(usage, " bytes smem").unwrap_or(0);
        if registers == 0 || registers > 128 || static_shared != 0 {
            return Err(format!(
                "{} ptxas resource contract failed: registers={registers}/128 static_shared={static_shared}/0",
                spec.symbol
            ));
        }
        Ok(())
    }

    fn validate_sass_has_no_atomics(sass: &str) -> Result<(), String> {
        for line in sass.lines() {
            let instruction = line
                .split_once("*/")
                .map(|(_, tail)| tail.trim_start())
                .unwrap_or_else(|| line.trim_start());
            if instruction.starts_with("ATOM") || instruction.starts_with("RED.") {
                return Err(format!(
                    "compiled SM120 candidate contains an atomic: {line}"
                ));
            }
        }
        Ok(())
    }

    fn validate_ptxas_resources_and_no_atomics() -> Result<(), String> {
        let ptx = compile_sm120_source()?;
        let directory =
            tempfile::tempdir().map_err(|error| format!("resource tempdir: {error}"))?;
        let ptx_path = directory.path().join("rect-wide.ptx");
        let cubin_path = directory.path().join("rect-wide.cubin");
        std::fs::write(&ptx_path, ptx).map_err(|error| format!("write resource PTX: {error}"))?;
        let tool = |name: &str| {
            std::env::var("CUDA_HOME")
                .map(|root| std::path::PathBuf::from(root).join("bin").join(name))
                .unwrap_or_else(|_| name.into())
        };
        let version = std::process::Command::new(tool("ptxas"))
            .arg("--version")
            .output()
            .map_err(|error| format!("launch ptxas --version: {error}"))?;
        let version_text = format!(
            "{}{}",
            String::from_utf8_lossy(&version.stdout),
            String::from_utf8_lossy(&version.stderr)
        );
        if !version.status.success() || !version_text.contains("release 13.2") {
            return Err(format!("requires exact ptxas 13.2, found {version_text}"));
        }
        let output = std::process::Command::new(tool("ptxas"))
            .args(["--verbose", "--gpu-name=sm_120"])
            .arg(&ptx_path)
            .arg("--output-file")
            .arg(&cubin_path)
            .output()
            .map_err(|error| format!("launch ptxas: {error}"))?;
        let report = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if !output.status.success() {
            return Err(format!("ptxas sm_120 failed:\n{report}"));
        }
        for spec in candidate_specs() {
            validate_ptxas_entry(&report, *spec)?;
        }
        let disassembly = std::process::Command::new(tool("nvdisasm"))
            .arg("--separate-functions")
            .arg(&cubin_path)
            .output()
            .map_err(|error| format!("launch nvdisasm: {error}"))?;
        if !disassembly.status.success() {
            return Err(format!(
                "nvdisasm failed: {}",
                String::from_utf8_lossy(&disassembly.stderr)
            ));
        }
        validate_sass_has_no_atomics(&String::from_utf8_lossy(&disassembly.stdout))?;
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA 13.2 NVRTC, ptxas, and nvdisasm but launches no GPU work"]
    fn rect_wide_candidates_meet_ptxas_resource_and_no_atomic_contract() -> Result<(), String> {
        validate_ptxas_resources_and_no_atomics()
    }

    #[test]
    #[ignore = "requires an exclusive CC12.0/170-SM CUDA 13.2 GPU"]
    fn rect_wide_candidates_meet_driver_resource_contract() -> Result<(), String> {
        let device = GpuDevice::new(0)?;
        let ctx = configure(&device, F32TriadPolicy::AllowDeterministicTf32V1)?;
        validate_exact_environment(&ctx)?;
        validate_driver_resources(&device)
    }

    fn exact_cohort_request() -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Nn,
            (2_048, 768, 3_072),
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::AllowDeterministicTf32V1),
            epilogue(),
        )
    }

    fn presize_tournament_context(ctx: &GpuCtx) -> Result<(), String> {
        let mut requests = Vec::with_capacity(candidate_specs().len() + 3);
        requests.push(exact_cohort_request());
        requests.push(scalar_baseline_request());
        requests.push(production_request());
        requests.extend(candidate_specs().iter().copied().map(forced_request));
        presize_physical_qualification_suite(ctx, &requests)
    }

    fn validate_exact_cohort_witness(ctx: &GpuCtx) -> Result<(), String> {
        validate_exact_environment(ctx)?;
        let request = exact_cohort_request();
        let qualified = qualify_physical_launch(ctx, request)?;
        let evidence = qualified.evidence();
        if evidence.launch_count() != 1
            || evidence.single_launch_symbol()
                != Some("gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2")
            || evidence.uniform_module_kind() != Some(ModuleKind::TriadSm120)
            || !evidence.eager_graph_equal()
        {
            return Err(format!(
                "exact CUDA 13.2 SM120 cohort witness changed: {:?}",
                evidence.nodes()
            ));
        }
        Ok(())
    }

    fn validate_candidate_evidence(
        spec: CandidateSpec,
        qualified: &QualifiedPhysicalLaunch<'_>,
    ) -> Result<CandidateIdentity, String> {
        let production_spec = tf32_kernel_spec(ResolvedGemmOp::Nn, candidate_route(spec))?;
        if production_spec.symbol != spec.symbol
            || production_spec.tile != spec.tile
            || production_spec.bk != spec.bk
            || production_spec.stages != spec.stages
            || production_spec.threads != spec.block.0
            || production_spec.dynamic_shared_bytes != spec.dynamic_shared_bytes
            || production_spec.module_kind != ModuleKind::TriadSm120
        {
            return Err(format!(
                "candidate no longer matches the production TF32 inventory: host={spec:?} production={production_spec:?}"
            ));
        }
        let evidence = qualified.evidence();
        if evidence.launch_count() != 1
            || !evidence.eager_graph_equal()
            || evidence.uniform_module_kind() != Some(ModuleKind::TriadSm120)
        {
            return Err(format!("candidate route evidence changed: {evidence:?}"));
        }
        let [node] = evidence.nodes() else {
            return Err("candidate must contain exactly one physical node".into());
        };
        if node.logical_op != ResolvedGemmOp::Nn
            || node.shape != DIMS
            || node.strides != (3_072, 768, 768)
            || node.tile != Some(spec.tile)
            || node.module_kind != ModuleKind::TriadSm120
        {
            return Err(format!("candidate logical identity changed: {node:?}"));
        }
        let identity = CandidateIdentity {
            symbol: node.symbol.to_owned(),
            grid: node.launch.grid_dim,
            block: node.launch.block_dim,
            dynamic_shared_bytes: node.launch.shared_mem_bytes,
            arguments_digest: node.launch.arguments_digest,
            request_identity_digest: evidence.request_identity_digest(),
            launch_digest: evidence.launch_digest(),
        };
        validate_candidate_identity(spec, &identity)?;
        Ok(identity)
    }

    fn run_repeated_bits(
        launch: &mut QualifiedPhysicalLaunch<'_>,
        ctx: &GpuCtx,
        salt: u64,
    ) -> Result<Vec<u32>, String> {
        launch.seed_f32_operands(ctx, salt)?;
        launch.measure_eager_window_ms(ctx, 1)?;
        let expected = launch.f32_output_bits(ctx)?;
        for repeat in 0..3 {
            launch.seed_f32_operands(ctx, salt)?;
            launch.measure_eager_window_ms(ctx, 1)?;
            if launch.f32_output_bits(ctx)? != expected {
                return Err(format!("eager repeat {repeat} changed output bits"));
            }
            launch.seed_f32_operands(ctx, salt)?;
            launch.measure_graph_window_ms(ctx, 1)?;
            if launch.f32_output_bits(ctx)? != expected {
                return Err(format!("graph repeat {repeat} changed output bits"));
            }
        }
        Ok(expected)
    }

    fn run_single_term_bits(
        launch: &mut QualifiedPhysicalLaunch<'_>,
        ctx: &GpuCtx,
    ) -> Result<Vec<u32>, String> {
        launch.seed_f32_nn_single_term_probe(ctx)?;
        launch.measure_eager_window_ms(ctx, 1)?;
        let eager = launch.f32_output_bits(ctx)?;
        launch.seed_f32_nn_single_term_probe(ctx)?;
        launch.measure_graph_window_ms(ctx, 1)?;
        let graph = launch.f32_output_bits(ctx)?;
        if eager != graph {
            return Err("single-term row-major probe differs between eager and graph".into());
        }
        Ok(eager)
    }

    fn validate_absolute_single_term_oracle(bits: &[u32]) -> Result<(), String> {
        let (m, _, n) = DIMS;
        if bits.len() != m * n {
            return Err(format!(
                "single-term output length changed: expected {} found {}",
                m * n,
                bits.len()
            ));
        }
        let mut checked = 0_usize;
        for row in 0..m {
            let a = ((row % 7) as i32 - 3) as f32 * 0.125;
            for column in 0..n {
                let b = ((column % 11) as i32 - 5) as f32 * 0.125;
                let expected = a.mul_add(b, 0.0);
                if expected != 0.0 {
                    let index = row * n + column;
                    if bits[index] != expected.to_bits() {
                        return Err(format!(
                            "absolute row-major TF32 oracle failed at row={row} column={column}: actual={:08x} expected={:08x}",
                            bits[index],
                            expected.to_bits()
                        ));
                    }
                    checked += 1;
                }
            }
        }
        if checked < m * n / 2 {
            return Err(format!(
                "absolute row-major TF32 oracle checked too few outputs: {checked}"
            ));
        }
        Ok(())
    }

    fn execute_direct_repeats(
        runtime: &DirectRuntime,
        spec: CandidateSpec,
        mut fixture: DirectFixture,
        expected: &[u32],
        label: &str,
    ) -> Result<(), String> {
        let expected_grid = if label == "large" {
            spec.grid
        } else {
            (1, 1, 1)
        };
        if fixture.config.grid_dim != expected_grid
            || fixture.config.block_dim != spec.block
            || fixture.config.shared_mem_bytes != spec.dynamic_shared_bytes
        {
            return Err(format!(
                "{} {label} direct launch geometry changed: {:?}",
                spec.symbol, fixture.config
            ));
        }
        let graph = capture_direct(runtime, spec, &fixture)?;
        for path in [Path::Eager, Path::Graph] {
            for repeat in 0..3 {
                fixture.output.reset(&runtime.stream)?;
                match path {
                    Path::Eager => launch_direct(runtime, spec, &fixture)?,
                    Path::Graph => graph.launch().map_err(|error| {
                        format!("launch direct {} graph: {error:?}", spec.symbol)
                    })?,
                }
                let actual = fixture
                    .output
                    .output_bits(&runtime.stream, &format!("{} {label} output", spec.symbol))?;
                if actual != expected {
                    let mismatch = actual
                        .iter()
                        .zip(expected)
                        .position(|(actual, expected)| actual != expected)
                        .unwrap_or(actual.len().min(expected.len()));
                    return Err(format!(
                        "{} {label} direct {} repeat {repeat} differs at {mismatch}",
                        spec.symbol,
                        path.name()
                    ));
                }
            }
        }
        fixture
            .a
            .validate_read_only(&runtime.stream, &format!("{} {label} A", spec.symbol))?;
        fixture
            .b
            .validate_read_only(&runtime.stream, &format!("{} {label} B", spec.symbol))?;
        runtime
            .stream
            .synchronize()
            .map_err(|error| format!("synchronize direct {} {label}: {error:?}", spec.symbol))?;
        Ok(())
    }

    fn run_direct_small_literal(
        runtime: &DirectRuntime,
        spec: CandidateSpec,
    ) -> Result<(), String> {
        let (a, dense_b, expected) = literal_row_major_oracle();
        let mut padded_b = vec![0.0_f32; 4 * 4];
        for row in 0..4 {
            padded_b[row * 4..row * 4 + 2].copy_from_slice(&dense_b[row * 2..row * 2 + 2]);
        }
        let fixture = new_direct_fixture(
            runtime,
            spec,
            (3, 4, 2),
            (4, 4, 2),
            a.to_vec(),
            padded_b,
            vec![0.0; 6],
        )?;
        execute_direct_repeats(runtime, spec, fixture, &expected.map(f32::to_bits), "small")
    }

    fn run_direct_large_guarded(
        runtime: &DirectRuntime,
        spec: CandidateSpec,
        expected: &[u32],
    ) -> Result<(), String> {
        let (m, k, n) = DIMS;
        let fixture = new_direct_fixture(
            runtime,
            spec,
            DIMS,
            (k, n, n),
            qualification_seeded_values(m * k, CORPUS_SALT ^ 0x2d),
            qualification_seeded_values(k * n, CORPUS_SALT ^ 0x67),
            qualification_seeded_values(m * n, CORPUS_SALT ^ 0x91),
        )?;
        execute_direct_repeats(runtime, spec, fixture, expected, "large")
    }

    fn measure(
        launch: &mut QualifiedPhysicalLaunch<'_>,
        ctx: &GpuCtx,
        path: Path,
        iterations: usize,
    ) -> Result<f64, String> {
        if iterations == 0 {
            return Err("timing iterations must be positive".into());
        }
        let elapsed_ms = match path {
            Path::Eager => launch.measure_eager_window_ms(ctx, iterations)?,
            Path::Graph => launch.measure_graph_window_ms(ctx, iterations)?,
        };
        let us = elapsed_ms * 1_000.0 / iterations as f64;
        if !us.is_finite() || us <= 0.0 {
            return Err(format!("invalid {path:?} timing sample {us}"));
        }
        Ok(us)
    }

    fn calibrate(
        launch: &mut QualifiedPhysicalLaunch<'_>,
        ctx: &GpuCtx,
        path: Path,
    ) -> Result<usize, String> {
        let pilot = measure(launch, ctx, path, 3)?;
        Ok(((TARGET_WINDOW_MS * 1_000.0 / pilot).round() as usize).clamp(3, MAX_WINDOW_ITERATIONS))
    }

    fn paired(
        baseline: (&GpuCtx, &mut QualifiedPhysicalLaunch<'_>),
        candidate: (&GpuCtx, &mut QualifiedPhysicalLaunch<'_>),
        path: Path,
        order: Order,
        windows: usize,
    ) -> Result<TimingStats, String> {
        if windows == 0 {
            return Err("paired timing requires positive windows".into());
        }
        let (baseline_ctx, baseline) = baseline;
        let (candidate_ctx, candidate) = candidate;
        baseline.seed_f32_operands(baseline_ctx, CORPUS_SALT)?;
        candidate.seed_f32_operands(candidate_ctx, CORPUS_SALT)?;
        let baseline_iterations = calibrate(baseline, baseline_ctx, path)?;
        let candidate_iterations = calibrate(candidate, candidate_ctx, path)?;
        let iterations = baseline_iterations.max(candidate_iterations);
        measure(baseline, baseline_ctx, path, iterations)?;
        measure(candidate, candidate_ctx, path, iterations)?;
        let mut baseline_samples = Vec::with_capacity(windows);
        let mut candidate_samples = Vec::with_capacity(windows);
        let mut speedups = Vec::with_capacity(windows);
        for _ in 0..windows {
            let (b0, b1, c0, c1) = match order {
                Order::Abba => {
                    let b0 = measure(baseline, baseline_ctx, path, iterations)?;
                    let c0 = measure(candidate, candidate_ctx, path, iterations)?;
                    let c1 = measure(candidate, candidate_ctx, path, iterations)?;
                    let b1 = measure(baseline, baseline_ctx, path, iterations)?;
                    (b0, b1, c0, c1)
                }
                Order::Baab => {
                    let c0 = measure(candidate, candidate_ctx, path, iterations)?;
                    let b0 = measure(baseline, baseline_ctx, path, iterations)?;
                    let b1 = measure(baseline, baseline_ctx, path, iterations)?;
                    let c1 = measure(candidate, candidate_ctx, path, iterations)?;
                    (b0, b1, c0, c1)
                }
            };
            let baseline_us = (b0 + b1) * 0.5;
            let candidate_us = (c0 + c1) * 0.5;
            let speedup = baseline_us / candidate_us;
            if [baseline_us, candidate_us, speedup]
                .into_iter()
                .any(|value| !value.is_finite() || value <= 0.0)
            {
                return Err("paired timing produced a non-finite or non-positive sample".into());
            }
            baseline_samples.push(baseline_us);
            candidate_samples.push(candidate_us);
            speedups.push(speedup);
        }
        Ok(TimingStats {
            baseline_p50_us: percentile(&baseline_samples, 0.50)?,
            candidate_p50_us: percentile(&candidate_samples, 0.50)?,
            speedup_p05: percentile(&speedups, 0.05)?,
            speedup_p50: percentile(&speedups, 0.50)?,
            speedup_p95: percentile(&speedups, 0.95)?,
        })
    }

    fn print_stats(label: &str, path: Path, order: Order, windows: usize, stats: TimingStats) {
        eprintln!(
            "tf32_nn_rect_wide pair={label} path={} order={} windows={} baseline_us_p50={:.6} candidate_us_p50={:.6} speedup_p05={:.9} speedup_p50={:.9} speedup_p95={:.9}",
            path.name(),
            order.name(),
            windows,
            stats.baseline_p50_us,
            stats.candidate_p50_us,
            stats.speedup_p05,
            stats.speedup_p50,
            stats.speedup_p95,
        );
    }

    fn qualify_candidate<'a>(
        ctx: &'a GpuCtx,
        spec: CandidateSpec,
    ) -> Result<QualifiedPhysicalLaunch<'a>, String> {
        let request = forced_request(spec);
        presize_physical_qualification_suite(ctx, &[request])?;
        let qualified = qualify_physical_launch(ctx, request)?;
        qualified.validate_timed_request(ctx, request)?;
        validate_candidate_evidence(spec, &qualified)?;
        Ok(qualified)
    }

    fn qualify_production<'a>(ctx: &'a GpuCtx) -> Result<QualifiedPhysicalLaunch<'a>, String> {
        let request = production_request();
        presize_physical_qualification_suite(ctx, &[request])?;
        let qualified = qualify_physical_launch(ctx, request)?;
        qualified.validate_timed_request(ctx, request)?;
        validate_candidate_evidence(WINNER, &qualified)?;
        let revision = qualified.evidence().route_identity().tuning_table_revision;
        if revision != 31 {
            return Err(format!(
                "rect-wide production route must use qualification revision 31, found {revision}"
            ));
        }
        Ok(qualified)
    }

    fn qualify_scalar_baseline<'a>(ctx: &'a GpuCtx) -> Result<QualifiedPhysicalLaunch<'a>, String> {
        let request = scalar_baseline_request();
        presize_physical_qualification_suite(ctx, &[request])?;
        let qualified = qualify_physical_launch(ctx, request)?;
        qualified.validate_timed_request(ctx, request)?;
        let evidence = qualified.evidence();
        if evidence.launch_count() != 1
            || evidence.uniform_module_kind() != Some(ModuleKind::TriadScalar)
            || !evidence.eager_graph_equal()
        {
            return Err(format!(
                "rect-wide scalar policy baseline identity changed: {:?}",
                evidence.nodes()
            ));
        }
        Ok(qualified)
    }

    #[test]
    #[ignore = "requires an exclusive CC12.0/170-SM CUDA 13.2 GPU"]
    fn rect_wide_candidates_are_exact_and_graph_stable() -> Result<(), String> {
        let device = GpuDevice::new(0)?;
        let reference_ctx = configure(&device, F32TriadPolicy::AllowDeterministicTf32V1)?;
        presize_tournament_context(&reference_ctx)?;
        validate_exact_cohort_witness(&reference_ctx)?;
        validate_ptxas_resources_and_no_atomics()?;
        validate_driver_resources(&device)?;
        validate_exact_environment(&reference_ctx)?;
        let (reference_bits, reference_orientation) = {
            let reference_request = portable_reference_request();
            let mut reference = qualify_physical_launch(&reference_ctx, reference_request)?;
            let reference_evidence = reference.evidence();
            if reference_evidence.launch_count() != 1
                || reference_evidence.uniform_module_kind() != Some(ModuleKind::TriadSm80)
                || !reference_evidence.eager_graph_equal()
            {
                return Err(format!(
                    "portable TF32 reference identity changed: {:?}",
                    reference_evidence.nodes()
                ));
            }
            (
                run_repeated_bits(&mut reference, &reference_ctx, CORPUS_SALT)?,
                run_single_term_bits(&mut reference, &reference_ctx)?,
            )
        };
        {
            let mut production = qualify_production(&reference_ctx)?;
            let actual = run_repeated_bits(&mut production, &reference_ctx, CORPUS_SALT)?;
            if actual != reference_bits {
                return Err(
                    "tag31 production differs bitwise from the portable TF32 reference".into(),
                );
            }
            let orientation = run_single_term_bits(&mut production, &reference_ctx)?;
            validate_absolute_single_term_oracle(&orientation)?;
            if orientation != reference_orientation {
                return Err(
                    "tag31 production failed the exact row-major single-term oracle".into(),
                );
            }
        }
        let direct = new_direct_runtime(&device)?;

        let mut identities = BTreeSet::new();
        for spec in candidate_specs() {
            run_direct_small_literal(&direct, *spec)?;
            run_direct_large_guarded(&direct, *spec, &reference_bits)?;
            let mut candidate = qualify_candidate(&reference_ctx, *spec)?;
            let identity = validate_candidate_evidence(*spec, &candidate)?;
            if !identities.insert(identity.arguments_digest) {
                return Err(format!("candidate argument digest collided: {identity:?}"));
            }
            let actual = run_repeated_bits(&mut candidate, &reference_ctx, CORPUS_SALT)?;
            if actual != reference_bits {
                return Err(format!(
                    "{} differs bitwise from the direct portable TF32 reference",
                    spec.symbol
                ));
            }
            let orientation = run_single_term_bits(&mut candidate, &reference_ctx)?;
            validate_absolute_single_term_oracle(&orientation)?;
            if orientation != reference_orientation {
                return Err(format!(
                    "{} failed the exact row-major single-term oracle",
                    spec.symbol
                ));
            }
        }
        Ok(())
    }

    fn screen_candidate(
        production_ctx: &GpuCtx,
        candidate_ctx: &GpuCtx,
        spec: CandidateSpec,
    ) -> Result<f64, String> {
        let mut production = qualify_production(production_ctx)?;
        let mut candidate = qualify_candidate(candidate_ctx, spec)?;
        let mut conservative = f64::INFINITY;
        for path in [Path::Eager, Path::Graph] {
            for order in [Order::Abba, Order::Baab] {
                let stats = paired(
                    (production_ctx, &mut production),
                    (candidate_ctx, &mut candidate),
                    path,
                    order,
                    SCREENING_WINDOWS,
                )?;
                print_stats(
                    &format!("production->{}-screen", spec.symbol),
                    path,
                    order,
                    SCREENING_WINDOWS,
                    stats,
                );
                conservative = conservative.min(stats.speedup_p50);
            }
        }
        Ok(conservative)
    }

    fn qualify_arm<'a>(
        ctx: &'a GpuCtx,
        arm: QualificationArm,
    ) -> Result<QualifiedPhysicalLaunch<'a>, String> {
        match arm {
            QualificationArm::ScalarBaseline => qualify_scalar_baseline(ctx),
            QualificationArm::Production => qualify_production(ctx),
            QualificationArm::Forced(spec) => qualify_candidate(ctx, spec),
        }
    }

    fn official_pair(
        baseline_ctx: &GpuCtx,
        candidate_ctx: &GpuCtx,
        baseline_arm: QualificationArm,
        candidate_arm: QualificationArm,
        gate: PairGate,
        quiet: &QuietGpu,
    ) -> Result<(), String> {
        quiet.require_cohort("tf32-nn-rect-wide-official-pair")?;
        let mut baseline = qualify_arm(baseline_ctx, baseline_arm)?;
        let mut candidate = qualify_arm(candidate_ctx, candidate_arm)?;
        for path in [Path::Eager, Path::Graph] {
            for order in [Order::Abba, Order::Baab] {
                let stats = paired(
                    (baseline_ctx, &mut baseline),
                    (candidate_ctx, &mut candidate),
                    path,
                    order,
                    OFFICIAL_WINDOWS,
                )?;
                print_stats(
                    &format!("{}->{}-official", baseline_arm.name(), candidate_arm.name()),
                    path,
                    order,
                    OFFICIAL_WINDOWS,
                    stats,
                );
                match gate {
                    PairGate::ScalarToProductionSpeedup => validate_post_promotion_speedup(
                        OFFICIAL_WINDOWS,
                        stats.speedup_p05,
                        stats.speedup_p50,
                    )?,
                    PairGate::ProductionToWinnerParity => validate_post_promotion_parity(
                        OFFICIAL_WINDOWS,
                        stats.speedup_p05,
                        stats.speedup_p50,
                        stats.speedup_p95,
                    )?,
                    PairGate::RunnerToWinner => {
                        if stats.speedup_p05 < WINNER_MIN_P05 || stats.speedup_p50 < WINNER_MIN_P50
                        {
                            return Err(format!(
                                "winner is not stable against runner-up: path={path:?} order={order:?} stats={stats:?}"
                            ));
                        }
                    }
                }
            }
        }
        quiet.verify_post_cohort("tf32-nn-rect-wide-official-pair")?;
        Ok(())
    }

    struct CublasBuffers {
        output: GpuBuffer,
        a: GpuBuffer,
        b: GpuBuffer,
    }

    fn seeded_values(len: usize, salt: u64) -> Vec<f32> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ salt;
        (0..len)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let signed = ((state.wrapping_add(index as u64) % 4093) as i32) - 2046;
                signed as f32 / 1024.0
            })
            .collect()
    }

    fn cublas_buffers(ctx: &GpuCtx) -> Result<CublasBuffers, String> {
        let (m, k, n) = DIMS;
        Ok(CublasBuffers {
            output: GpuBuffer::from_cpu(&ctx.stream, &seeded_values(m * n, CORPUS_SALT ^ 0x91))?,
            a: GpuBuffer::from_cpu(&ctx.stream, &seeded_values(m * k, CORPUS_SALT ^ 0x2d))?,
            b: GpuBuffer::from_cpu(&ctx.stream, &seeded_values(k * n, CORPUS_SALT ^ 0x67))?,
        })
    }

    fn launch_cublas_fast_nn(
        ctx: &GpuCtx,
        buffers: &CublasBuffers,
        dims: (usize, usize, usize),
    ) -> Result<(), String> {
        use cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32;
        use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;

        let (physical_m, physical_n, physical_k, lda, ldb, ldc) = cublas_geometry(dims)?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *ctx.blas.handle(),
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                physical_m,
                physical_n,
                physical_k,
                (&alpha as *const f32).cast::<c_void>(),
                buffers.b.raw_ptr(&ctx.stream) as *const c_void,
                WeightDtype::F32.cuda_data_type(),
                lda,
                buffers.a.raw_ptr(&ctx.stream) as *const c_void,
                WeightDtype::F32.cuda_data_type(),
                ldb,
                (&beta as *const f32).cast::<c_void>(),
                buffers.output.raw_ptr(&ctx.stream) as *mut c_void,
                WeightDtype::F32.cuda_data_type(),
                ldc,
                CUBLAS_COMPUTE_32F_FAST_TF32,
                cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|error| format!("cuBLAS-fast NN launch failed: {error:?}"))?;
        }
        Ok(())
    }

    fn validate_informational_cublas_orientation(ctx: &GpuCtx) -> Result<(), String> {
        let expected = [5.0_f32, 7.0, 22.0, 26.0, 51.0, 57.0];
        let a_host = [
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0,
        ];
        let b_host = [5.0_f32, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0];
        let buffers = CublasBuffers {
            output: GpuBuffer::from_cpu(&ctx.stream, &[0.0; 6])?,
            a: GpuBuffer::from_cpu(&ctx.stream, &a_host)?,
            b: GpuBuffer::from_cpu(&ctx.stream, &b_host)?,
        };
        for repeat in 0..3 {
            launch_cublas_fast_nn(ctx, &buffers, (3, 4, 2))?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("cuBLAS orientation sync: {error:?}"))?;
            let actual = buffers.output.to_cpu(&ctx.stream)?;
            if actual
                .iter()
                .zip(expected)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                return Err(format!(
                    "cuBLAS-fast row-major orientation repeat {repeat} changed: {actual:?}"
                ));
            }
            if buffers.a.to_cpu(&ctx.stream)? != a_host || buffers.b.to_cpu(&ctx.stream)? != b_host
            {
                return Err("cuBLAS-fast orientation probe modified an input".into());
            }
        }
        Ok(())
    }

    fn measure_cublas_fast(
        ctx: &GpuCtx,
        buffers: &CublasBuffers,
        iterations: usize,
    ) -> Result<f64, String> {
        if iterations == 0 {
            return Err("cuBLAS timing iterations must be positive".into());
        }
        let start = ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("cuBLAS start event: {error:?}"))?;
        for _ in 0..iterations {
            launch_cublas_fast_nn(ctx, buffers, DIMS)?;
        }
        let end = ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("cuBLAS end event: {error:?}"))?;
        let us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("cuBLAS elapsed event: {error:?}"))?,
        ) * 1_000.0
            / iterations as f64;
        if !us.is_finite() || us <= 0.0 {
            return Err(format!("invalid cuBLAS-fast timing sample {us}"));
        }
        Ok(us)
    }

    fn report_informational_cublas_fast(ctx: &GpuCtx, quiet: &QuietGpu) -> Result<(), String> {
        quiet.require_cohort("tf32-nn-rect-wide-cublas-fast")?;
        validate_informational_cublas_orientation(ctx)?;
        let buffers = cublas_buffers(ctx)?;
        let pilot = measure_cublas_fast(ctx, &buffers, 3)?;
        let iterations =
            ((TARGET_WINDOW_MS * 1_000.0 / pilot).round() as usize).clamp(3, MAX_WINDOW_ITERATIONS);
        measure_cublas_fast(ctx, &buffers, iterations)?;
        let mut samples = Vec::with_capacity(OFFICIAL_WINDOWS);
        for _ in 0..OFFICIAL_WINDOWS {
            samples.push(measure_cublas_fast(ctx, &buffers, iterations)?);
        }
        eprintln!(
            "tf32_nn_rect_wide arm=cublas_fast informational_only=true promotion_gate=false windows={} p05_us={:.6} p50_us={:.6} p95_us={:.6}",
            OFFICIAL_WINDOWS,
            percentile(&samples, 0.05)?,
            percentile(&samples, 0.50)?,
            percentile(&samples, 0.95)?,
        );
        quiet.verify_post_cohort("tf32-nn-rect-wide-cublas-fast")?;
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC12.0/170-SM CUDA 13.2 GPU"]
    fn rect_wide_tournament_abba_baab() -> Result<(), String> {
        validate_release_build()?;
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("tf32-nn-rect-wide-pre-context")?;
        let device = GpuDevice::new(0)?;
        let baseline_ctx = configure(&device, F32TriadPolicy::AllowDeterministicTf32V1)?;
        let candidate_ctx = configure(&device, F32TriadPolicy::AllowDeterministicTf32V1)?;
        presize_tournament_context(&baseline_ctx)?;
        presize_tournament_context(&candidate_ctx)?;
        validate_exact_cohort_witness(&baseline_ctx)?;
        validate_ptxas_resources_and_no_atomics()?;
        validate_driver_resources(&device)?;
        let mut ranking = Vec::new();
        for spec in candidate_specs() {
            quiet.require_cohort(spec.symbol)?;
            ranking.push((
                *spec,
                screen_candidate(&baseline_ctx, &candidate_ctx, *spec)?,
            ));
            quiet.verify_post_cohort(spec.symbol)?;
        }
        ranking.sort_by(|left, right| right.1.total_cmp(&left.1));
        let [(winner, winner_score), (runner_up, runner_up_score), ..] = ranking.as_slice() else {
            return Err("rect-wide tournament requires at least two candidates".into());
        };
        eprintln!(
            "tf32_nn_rect_wide screening winner={} score={:.9} runner_up={} score={:.9}",
            winner.symbol, winner_score, runner_up.symbol, runner_up_score
        );
        if *winner != WINNER {
            return Err(format!(
                "post-promotion screening winner changed: expected={} actual={}",
                WINNER.symbol, winner.symbol
            ));
        }
        official_pair(
            &baseline_ctx,
            &candidate_ctx,
            QualificationArm::ScalarBaseline,
            QualificationArm::Production,
            PairGate::ScalarToProductionSpeedup,
            &quiet,
        )?;
        official_pair(
            &baseline_ctx,
            &candidate_ctx,
            QualificationArm::Production,
            QualificationArm::Forced(WINNER),
            PairGate::ProductionToWinnerParity,
            &quiet,
        )?;
        official_pair(
            &baseline_ctx,
            &candidate_ctx,
            QualificationArm::Forced(*runner_up),
            QualificationArm::Forced(WINNER),
            PairGate::RunnerToWinner,
            &quiet,
        )?;
        if let Err(error) = report_informational_cublas_fast(&baseline_ctx, &quiet) {
            eprintln!(
                "tf32_nn_rect_wide arm=cublas_fast informational_only=true promotion_gate=false unavailable={error}"
            );
        }
        Ok(())
    }
}
