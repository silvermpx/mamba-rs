use std::collections::BTreeSet;

#[cfg(feature = "cuda")]
mod common;

const DIMS: (usize, usize, usize) = (256, 512, 384);
const OFFICIAL_WINDOWS: usize = 101;
const MIN_BASELINE_P05_SPEEDUP: f64 = 1.05;
const MIN_BASELINE_P50_SPEEDUP: f64 = 1.08;
const DIRECT_GUARD_ELEMENTS: usize = 32;
const DIRECT_GUARD_BITS: u32 = 0x7fc1_5247;

fn validate_portable_sm120_targets(targets: (&str, &str, &str, &str)) -> Result<(), String> {
    let expected = ("compute_120", "compute_120", "compute_120", "sm_120");
    if targets == expected {
        Ok(())
    } else {
        Err(format!(
            "portable SM120 target identity changed: expected={expected:?} actual={targets:?}"
        ))
    }
}

#[test]
fn portable_sm120_target_contract_rejects_old_and_single_field_mutations() {
    let production = ("compute_120", "compute_120", "compute_120", "sm_120");
    validate_portable_sm120_targets(production).unwrap();
    assert!(validate_portable_sm120_targets(("sm_120", "sm_120", "sm_120", "sm_120")).is_err());
    for changed in [
        ("sm_120", production.1, production.2, production.3),
        (production.0, "sm_120", production.2, production.3),
        (production.0, production.1, "sm_120", production.3),
        (production.0, production.1, production.2, "compute_120"),
    ] {
        assert!(validate_portable_sm120_targets(changed).is_err());
    }
}

#[test]
fn production_and_forced_eager_timers_use_distinct_validation_paths() {
    let launch = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/launch.rs");
    let qualification = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/qualification.rs");
    let cache_launch = launch
        .split_once("fn launch<Scalar>(")
        .and_then(|(_, tail)| tail.split_once("fn launch_observed"))
        .map(|(body, _)| body)
        .expect("prepared cache launch body");
    assert!(cache_launch.contains("enqueue_validated_prepared_f32_triad"));
    assert!(!cache_launch.contains("validate_prepared_f32_triad"));

    let public_launch = launch
        .split_once("unsafe fn launch_prepared_f32_triad(")
        .and_then(|(_, tail)| tail.split_once("unsafe fn enqueue_validated_prepared_f32_triad("))
        .map(|(body, _)| body)
        .expect("validated prepared launch body");
    let validation = public_launch
        .find("validate_prepared_f32_triad")
        .expect("forced launch validation");
    let enqueue = public_launch
        .find("enqueue_validated_prepared_f32_triad")
        .expect("forced launch enqueue");
    assert!(validation < enqueue);

    let eager_timer = qualification
        .split_once("pub fn measure_eager_window_ms(")
        .and_then(|(_, tail)| tail.split_once("fn seeded_qualification_values"))
        .map(|(body, _)| body)
        .expect("qualification eager timer body");
    assert!(eager_timer.contains("launch_f32_policy_eager"));
    assert!(eager_timer.contains("launch_prepared_f32_triad"));

    let prevalidated = qualification
        .split_once("pub fn measure_prevalidated_forced_eager_window_ms(")
        .and_then(|(_, tail)| tail.split_once("fn seeded_qualification_values"))
        .map(|(body, _)| body)
        .expect("prevalidated forced eager timer body");
    let validate = prevalidated
        .find("validate_prepared_f32_triad_for_timing")
        .expect("validation-only preflight before timing");
    let timed = prevalidated
        .find("measure_production_eager")
        .expect("prevalidated timed window");
    let enqueue = prevalidated
        .find("enqueue_validated_prepared_f32_triad")
        .expect("trusted enqueue inside timed window");
    assert!(validate < timed && timed < enqueue);
    assert!(!prevalidated.contains("launch_prepared_f32_triad"));
}

#[test]
fn parity_audits_graph_before_eager_and_compares_normalized_route_identity() {
    let source = include_str!("gemm_bi_tf32_tn_underfill_tournament.rs");
    let official = source
        .split_once("\n    fn official_pair(")
        .and_then(|(_, tail)| tail.split_once("\n    fn report_validated_forced_overhead("))
        .map(|(body, _)| body)
        .expect("official pair body");
    assert!(official.contains("[Path::Graph, Path::Eager]"));

    let correctness = source
        .split_once("\n    fn tn_underfill_candidates_are_exact_and_graph_stable(")
        .and_then(|(_, tail)| tail.split_once("\n    fn official_pair("))
        .map(|(body, _)| body)
        .expect("correctness gate body");
    assert!(correctness.contains("production_route_identity != forced_route_identity"));
}

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
    register_cap: u32,
    minimum_occupancy: u32,
    tma: bool,
}

const CANDIDATES: [CandidateSpec; 4] = [
    CandidateSpec {
        symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4",
        tile: (16, 32),
        bk: 32,
        stages: 4,
        grid: (384, 1, 1),
        block: (128, 1, 1),
        dynamic_shared_bytes: 32_768,
        register_cap: 96,
        minimum_occupancy: 3,
        tma: false,
    },
    CandidateSpec {
        symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m16n16_bk32_s4",
        tile: (16, 16),
        bk: 32,
        stages: 4,
        grid: (768, 1, 1),
        block: (64, 1, 1),
        dynamic_shared_bytes: 24_576,
        register_cap: 96,
        minimum_occupancy: 4,
        tma: false,
    },
    CandidateSpec {
        symbol: "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2",
        tile: (64, 64),
        bk: 32,
        stages: 2,
        grid: (48, 1, 1),
        block: (128, 1, 1),
        dynamic_shared_bytes: 32_896,
        register_cap: 128,
        minimum_occupancy: 2,
        tma: true,
    },
    CandidateSpec {
        symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m64n64_bk32_s3",
        tile: (64, 64),
        bk: 32,
        stages: 3,
        grid: (48, 1, 1),
        block: (128, 1, 1),
        dynamic_shared_bytes: 55_296,
        register_cap: 128,
        minimum_occupancy: 1,
        tma: false,
    },
];

const fn candidate_specs() -> &'static [CandidateSpec; 4] {
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
    if !spec.tma {
        return Err(format!("{} does not use tensor maps", spec.symbol));
    }
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
            box_dimensions: [spec.bk, spec.bk],
            base_offset_bytes: 0,
            format: "uint32-v1",
        },
        LiteralTmaMapKey {
            operand: "B",
            global_dimensions: [n as u64, m as u64],
            outer_byte_stride: byte_stride(n, "B")?,
            box_dimensions: [spec.bk, spec.bk],
            base_offset_bytes: 0,
            format: "uint32-v1",
        },
    ])
}

fn literal_tn_oracle() -> ([f32; 12], [f32; 6], [f32; 8], [f32; 8]) {
    let a = [
        1.0_f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0,
    ];
    let b = [5.0_f32, 7.0, 11.0, 13.0, 17.0, 19.0];
    let initial = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let expected = [6.0_f32, 9.0, 25.0, 30.0, 56.0, 63.0, 7.0, 8.0];
    (a, b, initial, expected)
}

fn tn_geometry(dims: (usize, usize, usize)) -> Result<(usize, usize, usize), String> {
    let (m, k, n) = dims;
    if m == 0 || k == 0 || n == 0 {
        return Err(format!("TN dimensions must be positive: {dims:?}"));
    }
    m.checked_mul(n)
        .and_then(|_| m.checked_mul(k))
        .and_then(|_| k.checked_mul(n))
        .ok_or_else(|| format!("TN dimensions overflow: {dims:?}"))?;
    Ok((k, n, n))
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
    if p05 < MIN_BASELINE_P05_SPEEDUP || p50 < MIN_BASELINE_P50_SPEEDUP {
        return Err(format!(
            "baseline speedup failed: p05={p05:.9} p50={p50:.9}"
        ));
    }
    Ok(())
}

#[test]
fn tn_underfill_candidates_have_literal_geometry() {
    let candidates = candidate_specs();
    assert_eq!(candidates.len(), 4);
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
                    candidate.register_cap,
                    candidate.minimum_occupancy,
                    candidate.tma,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (
                "gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4",
                (16, 32),
                4,
                (384, 1, 1),
                (128, 1, 1),
                32_768,
                96,
                3,
                false,
            ),
            (
                "gemm_bi_tn_sm80_mma_tf32_v1_m16n16_bk32_s4",
                (16, 16),
                4,
                (768, 1, 1),
                (64, 1, 1),
                24_576,
                96,
                4,
                false,
            ),
            (
                "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2",
                (64, 64),
                2,
                (48, 1, 1),
                (128, 1, 1),
                32_896,
                128,
                2,
                true,
            ),
            (
                "gemm_bi_tn_sm80_mma_tf32_v1_m64n64_bk32_s3",
                (64, 64),
                3,
                (48, 1, 1),
                (128, 1, 1),
                55_296,
                128,
                1,
                false,
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
fn row_major_tn_geometry_is_not_transposed() {
    assert_eq!(tn_geometry(DIMS).unwrap(), (512, 384, 384));
    for invalid in [(0, 512, 384), (256, 0, 384), (256, 512, 0)] {
        assert!(tn_geometry(invalid).is_err());
    }
}

#[test]
fn literal_tma_keys_and_small_tn_beta_one_oracle_are_absolute() {
    for spec in candidate_specs().iter().filter(|spec| spec.tma) {
        assert_eq!(
            literal_tma_map_keys(*spec).unwrap(),
            [
                LiteralTmaMapKey {
                    operand: "A",
                    global_dimensions: [512, 256],
                    outer_byte_stride: 2_048,
                    box_dimensions: [32, 32],
                    base_offset_bytes: 0,
                    format: "uint32-v1",
                },
                LiteralTmaMapKey {
                    operand: "B",
                    global_dimensions: [384, 256],
                    outer_byte_stride: 1_536,
                    box_dimensions: [32, 32],
                    base_offset_bytes: 0,
                    format: "uint32-v1",
                },
            ]
        );
    }
    for spec in candidate_specs().iter().filter(|spec| !spec.tma) {
        assert!(literal_tma_map_keys(*spec).is_err());
    }
    let (a, b, initial, expected) = literal_tn_oracle();
    assert_eq!(a.len(), 3 * 4);
    assert_eq!(b.len(), 3 * 2);
    assert_eq!(initial.len(), 4 * 2);
    assert_eq!(
        expected.map(f32::to_bits),
        [6.0, 9.0, 25.0, 30.0, 56.0, 63.0, 7.0, 8.0].map(f32::to_bits)
    );
    assert_ne!(initial.map(f32::to_bits), expected.map(f32::to_bits));
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
        assert!(validate_official_gate(windows, 1.05, 1.08).is_err());
    }
    assert!(validate_official_gate(101, 1.05, 1.08).is_ok());
    assert!(validate_official_gate(101, 1.049_999, 1.08).is_err());
    assert!(validate_official_gate(101, 1.05, 1.079_999).is_err());
    assert!(validate_official_gate(101, f64::NAN, 2.0).is_err());
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
    let sm80 = include_str!("../kernels/gemm_bi_triad/sm80.cu");
    let sm120 = include_str!("../kernels/gemm_bi_triad/sm120.cu");
    for candidate in candidate_specs() {
        let source = if candidate.tma { sm120 } else { sm80 };
        assert_eq!(source.matches(candidate.symbol).count(), 2);
        assert!(!candidate.symbol.contains("splitk"));
    }
    for (source, macro_name) in [
        (sm80, "#define GEMM_BI_TF32_DEFINE_KERNEL"),
        (sm120, "#define SM120_DEFINE_TF32_KERNEL"),
    ] {
        let signature = source
            .split(macro_name)
            .nth(1)
            .expect("production TF32 kernel macro")
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
    }

    let contract = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/contract.rs");
    for literal in [
        "ResolvedGemmOp::Tn => [",
        "box_dimensions: [spec.bk, spec.bk]",
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

    const CORPUS_SALT: u64 = 0x746e_7532_3536_3338;
    const TARGET_WINDOW_MS: f64 = 10.0;
    const MAX_WINDOW_ITERATIONS: usize = 16_384;
    const WINNER_MIN_P05: f64 = 1.00;
    const WINNER_MIN_P50: f64 = 1.01;
    const PARITY_MIN: f64 = 0.98;
    const PARITY_MAX: f64 = 1.02;

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

    #[derive(Clone, Copy, Debug)]
    struct TimingStats {
        baseline_p50_us: f64,
        candidate_p50_us: f64,
        speedup_p05: f64,
        speedup_p50: f64,
        speedup_p95: f64,
    }

    fn epilogue() -> PhysicalQualificationF32Epilogue {
        PhysicalQualificationF32Epilogue::new(1.0, 1.0, false)
    }

    fn configure(device: &GpuDevice, policy: F32TriadPolicy) -> Result<GpuCtx, String> {
        let ctx = GpuCtx::new(device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_f32_triad_policy(policy);
        Ok(ctx)
    }

    fn candidate_route(spec: CandidateSpec) -> Tf32PhysicalRoute {
        if spec.tma {
            return Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(Tf32Sm120Route {
                tile: Tf32Sm120Tile::M64N64,
                stages: Tf32Sm120Stages::S2,
            });
        }
        let tile = match spec.tile {
            (16, 32) => Tf32PortableTile::M16N32,
            (16, 16) => Tf32PortableTile::M16N16,
            (64, 64) => Tf32PortableTile::M64N64,
            _ => unreachable!("host geometry gate admits only retained portable TF32 tiles"),
        };
        let stages = match spec.stages {
            3 => Tf32PortableStages::S3,
            4 => Tf32PortableStages::S4,
            _ => unreachable!("host geometry gate admits only retained portable TF32 stages"),
        };
        Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute { tile, stages })
    }

    fn literal_tma_map_digest(spec: CandidateSpec) -> Result<[u8; 32], String> {
        let keys = literal_tma_map_keys(spec)?;
        let mut digest = FramedSha256::new(b"tf32-tn-underfill-literal-tma-keys.v1")
            .required(b"logical-op", b"tn")
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
        let spec = CANDIDATES[2];
        assert!(spec.tma);
        assert_eq!(
            digest_hex(&literal_tma_map_digest(spec).unwrap()),
            "16466b164d2220f73ee547f48044497e0868ec87d8bea6feb8f341f03cecc9c4"
        );
    }

    fn forced_request(spec: CandidateSpec) -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Tn,
            DIMS,
            PhysicalQualificationRoute::Tf32Forced(candidate_route(spec)),
            epilogue(),
        )
    }

    fn production_request() -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Tn,
            DIMS,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::AllowDeterministicTf32V1),
            epilogue(),
        )
    }

    #[derive(Clone, Copy)]
    enum Arm {
        Production,
        Forced(CandidateSpec),
    }

    impl Arm {
        fn label(self) -> &'static str {
            match self {
                Self::Production => "production-tag32",
                Self::Forced(spec) => spec.symbol,
            }
        }

        fn spec(self) -> CandidateSpec {
            match self {
                Self::Production => CANDIDATES[0],
                Self::Forced(spec) => spec,
            }
        }

        fn request(self) -> PhysicalQualificationRequest {
            match self {
                Self::Production => production_request(),
                Self::Forced(spec) => forced_request(spec),
            }
        }

        fn prevalidated_forced(self) -> bool {
            matches!(self, Self::Forced(_))
        }
    }

    #[derive(Clone, Copy)]
    enum TimingGate {
        HardBaseline,
        Parity,
        Winner,
    }

    fn portable_reference_request() -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Tn,
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

    fn validate_exact_environment(ctx: &GpuCtx) -> Result<(), String> {
        let availability = ctx.kernels.f32_triad_availability();
        let portable = availability.portable.ok_or_else(|| {
            "TN underfill tournament requires the portable TF32 module".to_string()
        })?;
        let specialized = availability.specialized.ok_or_else(|| {
            "TN underfill tournament requires the specialized TF32 module".to_string()
        })?;
        let portable_accepted = portable
            .device_caps
            .accepted_target
            .ok_or_else(|| "portable SM120 module has no accepted target".to_string())?;
        validate_portable_sm120_targets((
            portable.target.as_str(),
            portable.compiler.target.as_str(),
            portable_accepted.as_str(),
            portable.device.target.as_str(),
        ))?;
        for (qualified, kind, compiler_target) in [
            (portable, ModuleKind::TriadSm80, "compute_120"),
            (specialized, ModuleKind::TriadSm120, "compute_120"),
        ] {
            let artifact = qualified.artifact;
            let compiler = qualified.compiler;
            if qualified.module_kind != kind
                || qualified.target.as_str() != compiler_target
                || qualified.device.compute_capability != (12, 0)
                || qualified.device.multiprocessor_count != 170
                || qualified.device.target.as_str() != "sm_120"
                || qualified.device_caps.compute_capability != (12, 0)
                || qualified.device_caps.nvrtc_version != (13, 2)
                || qualified.device_caps.accepted_target != Some(qualified.target)
                || qualified.device_caps.optin_shared_bytes != 101_376
                || !qualified.device_caps.tensor_map_access
                || compiler.target.as_str() != compiler_target
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
                || artifact.module_kind != kind
                || artifact.artifact_kind != compiler.output_kind
                || artifact.compile_key != compiler.invocation_digest
                || artifact.compile_key == [0; 32]
                || artifact.artifact_digest == [0; 32]
            {
                return Err(format!(
                    "TN underfill tournament requires the exact CUDA 13.2 SM120/170-SM artifact: qualified={qualified:?}"
                ));
            }
        }
        Ok(())
    }

    fn validate_release_build() -> Result<(), String> {
        if cfg!(debug_assertions) {
            Err("tn-underfill official timing requires --release".into())
        } else {
            Ok(())
        }
    }

    fn compose_source(fragments: &[(&str, &str)]) -> String {
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

    fn production_sm80_source() -> String {
        compose_source(&[
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
                "kernels/gemm_bi_triad/mma16.cuh",
                include_str!("../kernels/gemm_bi_triad/mma16.cuh"),
            ),
            (
                "kernels/gemm_bi_triad/sm80.cu",
                include_str!("../kernels/gemm_bi_triad/sm80.cu"),
            ),
        ])
    }

    fn production_sm120_source() -> String {
        compose_source(&[
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
        ])
    }

    fn compile_source(source: String, target: &'static str, label: &str) -> Result<String, String> {
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some(target),
            options: vec![
                "--fmad=true".to_owned(),
                "--extra-device-vectorization".to_owned(),
                "-DNDEBUG".to_owned(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        cudarc::nvrtc::compile_ptx_with_opts(source, options)
            .map(|ptx| ptx.to_src())
            .map_err(|error| format!("compile production {label} TF32 source: {error:?}"))
    }

    fn compile_sm80_source(target: &'static str) -> Result<String, String> {
        compile_source(production_sm80_source(), target, "SM80 portable")
    }

    fn compile_sm120_source() -> Result<String, String> {
        compile_source(production_sm120_source(), "compute_120", "SM120")
    }

    struct ResourceModules {
        _modules: Vec<Arc<CudaModule>>,
        functions: Vec<(CandidateSpec, CudaFunction)>,
    }

    impl ResourceModules {
        fn function(&self, spec: CandidateSpec) -> Result<&CudaFunction, String> {
            self.functions
                .iter()
                .find_map(|(candidate, function)| (*candidate == spec).then_some(function))
                .ok_or_else(|| format!("direct resource module lost {}", spec.symbol))
        }
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct DirectSm80Params {
        alpha: f32,
        beta: f32,
        m: i32,
        k: i32,
        n: i32,
        lda: i32,
        ldb: i32,
        ldc: i32,
    }

    unsafe impl DeviceRepr for DirectSm80Params {}

    const _: () = {
        assert!(std::mem::size_of::<DirectSm80Params>() == 32);
        assert!(std::mem::align_of::<DirectSm80Params>() == 4);
    };

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
        maps: Option<DirectTensorMaps>,
        sm80_params: DirectSm80Params,
        sm120_params: DirectSm120Params,
        config: LaunchConfig,
    }

    struct DirectRuntime {
        stream: Arc<CudaStream>,
        modules: ResourceModules,
    }

    fn load_resource_modules(
        device: &GpuDevice,
        specs: &[CandidateSpec],
    ) -> Result<ResourceModules, String> {
        let portable_target = match device.compute_capability {
            (8, 9) => "sm_89",
            (12, 0) => "sm_120",
            other => {
                return Err(format!(
                    "unsupported TN underfill resource device {other:?}"
                ));
            }
        };
        let mut modules = Vec::new();
        let mut functions = Vec::with_capacity(specs.len());
        for (tma, ptx) in [
            (false, Some(compile_sm80_source(portable_target)?)),
            (
                true,
                specs
                    .iter()
                    .any(|spec| spec.tma)
                    .then(compile_sm120_source)
                    .transpose()?,
            ),
        ] {
            let Some(ptx) = ptx else { continue };
            let module = device
                .context()
                .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
                .map_err(|error| {
                    format!("load production TF32 resource module tma={tma}: {error:?}")
                })?;
            for spec in specs.iter().filter(|spec| spec.tma == tma) {
                let function = module.load_function(spec.symbol).map_err(|error| {
                    format!("load {} for resource validation: {error:?}", spec.symbol)
                })?;
                function
                    .set_attribute(
                        sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        spec.dynamic_shared_bytes as i32,
                    )
                    .map_err(|error| format!("set {} dynamic shared resource contract: {error:?}", spec.symbol))?;
                functions.push((*spec, function));
            }
            modules.push(module);
        }
        if functions.len() != specs.len() {
            return Err(format!(
                "resource module loaded {} of {} candidates",
                functions.len(),
                specs.len()
            ));
        }
        Ok(ResourceModules {
            _modules: modules,
            functions,
        })
    }

    fn new_direct_runtime(device: &GpuDevice) -> Result<DirectRuntime, String> {
        Ok(DirectRuntime {
            stream: device.fork_stream()?,
            modules: load_resource_modules(device, candidate_specs())?,
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
        let expected_b = m
            .checked_mul(ldb)
            .ok_or_else(|| "direct B extent overflow".to_string())?;
        let expected_output = k
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
        let maps = if spec.tma {
            Some(DirectTensorMaps {
                a: direct_tensor_map(a_pointer, k, m, lda, [spec.bk, spec.bk], "A")?,
                b: direct_tensor_map(b_pointer, n, m, ldb, [spec.bk, spec.bk], "B")?,
            })
        } else {
            None
        };
        let grid = k
            .div_ceil(spec.tile.0 as usize)
            .checked_mul(n.div_ceil(spec.tile.1 as usize))
            .ok_or_else(|| "direct grid overflow".to_string())?;
        let grid = u32::try_from(grid).map_err(|_| "direct grid exceeds u32".to_string())?;
        Ok(DirectFixture {
            a,
            b,
            output,
            maps,
            sm80_params: DirectSm80Params {
                alpha: 1.0,
                beta: 1.0,
                m: i32::try_from(m).map_err(|_| "direct M exceeds i32".to_string())?,
                k: i32::try_from(k).map_err(|_| "direct K exceeds i32".to_string())?,
                n: i32::try_from(n).map_err(|_| "direct N exceeds i32".to_string())?,
                lda: i32::try_from(lda).map_err(|_| "direct lda exceeds i32".to_string())?,
                ldb: i32::try_from(ldb).map_err(|_| "direct ldb exceeds i32".to_string())?,
                ldc: i32::try_from(ldc).map_err(|_| "direct ldc exceeds i32".to_string())?,
            },
            sm120_params: DirectSm120Params {
                a_x: 0,
                a_y: 0,
                b_x: 0,
                b_y: 0,
                alpha: 1.0,
                beta: 1.0,
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
        if spec.tma {
            let maps = fixture
                .maps
                .ok_or_else(|| format!("{} direct TMA maps are missing", spec.symbol))?;
            let mut builder = runtime
                .stream
                .launch_builder(runtime.modules.function(spec)?);
            builder.arg(&output);
            builder.arg(&maps.a);
            builder.arg(&maps.b);
            builder.arg(&bias);
            builder.arg(&fixture.sm120_params);
            unsafe { builder.launch(fixture.config) }
                .map(|_| ())
                .map_err(|error| format!("launch direct {}: {error:?}", spec.symbol))
        } else {
            let a = fixture.a.active_ptr(&runtime.stream, "A")?;
            let b = fixture.b.active_ptr(&runtime.stream, "B")?;
            let mut builder = runtime
                .stream
                .launch_builder(runtime.modules.function(spec)?);
            builder.arg(&output);
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&bias);
            builder.arg(&fixture.sm80_params);
            unsafe { builder.launch(fixture.config) }
                .map(|_| ())
                .map_err(|error| format!("launch direct {}: {error:?}", spec.symbol))
        }
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

    fn validate_driver_resources(
        device: &GpuDevice,
        specs: &[CandidateSpec],
    ) -> Result<(), String> {
        let resources = load_resource_modules(device, specs)?;
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
                || registers > spec.register_cap as i32
                || local != 0
                || static_shared != 0
                || max_threads < spec.block.0 as i32
                || max_dynamic < spec.dynamic_shared_bytes as i32
                || occupancy < spec.minimum_occupancy
            {
                return Err(format!(
                    "{} driver resource contract failed: regs={registers}/{} local={local}/0 static_shared={static_shared}/0 dynamic_shared={}/{max_dynamic} threads={}/{} occupancy={}/{}",
                    spec.symbol,
                    spec.register_cap,
                    spec.dynamic_shared_bytes,
                    spec.block.0,
                    max_threads,
                    occupancy,
                    spec.minimum_occupancy
                ));
            }
            eprintln!(
                "tf32_tn_underfill resource symbol={} regs={} local_bytes={} static_shared_bytes={} dynamic_shared_bytes={} max_dynamic_shared_bytes={} threads={} max_threads={} occupancy={}",
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
        if registers == 0 || registers > spec.register_cap as u64 || static_shared != 0 {
            return Err(format!(
                "{} ptxas resource contract failed: registers={registers}/{} static_shared={static_shared}/0",
                spec.symbol, spec.register_cap
            ));
        }
        Ok(())
    }

    fn line_offset(source: &str, mut predicate: impl FnMut(&str) -> bool) -> Option<usize> {
        let mut offset = 0;
        for line in source.split_inclusive('\n') {
            let content = line.trim_end_matches(['\n', '\r']);
            if predicate(content) {
                return Some(offset);
            }
            offset += line.len();
        }
        None
    }

    fn nvdisasm_symbol(line: &str) -> Option<&str> {
        let line = line
            .trim()
            .strip_prefix("//")
            .map(str::trim)
            .unwrap_or(line.trim());
        let symbol = line.strip_prefix("Function : ")?;
        (!symbol.is_empty() && !symbol.contains(char::is_whitespace)).then_some(symbol)
    }

    fn sass_entry<'a>(sass: &'a str, symbol: &str) -> Result<&'a str, String> {
        if let Some(start) = line_offset(sass, |line| nvdisasm_symbol(line) == Some(symbol)) {
            let tail = &sass[start..];
            let body_start = tail
                .find('\n')
                .map(|offset| offset + 1)
                .unwrap_or(tail.len());
            let end = line_offset(&tail[body_start..], |line| nvdisasm_symbol(line).is_some())
                .map(|offset| body_start + offset)
                .unwrap_or(tail.len());
            return Ok(&tail[..end]);
        }
        let label = format!("\n{symbol}:\n");
        let start = sass
            .find(&label)
            .map(|offset| offset + 1)
            .ok_or_else(|| format!("SASS omitted {symbol}"))?;
        let tail = &sass[start..];
        let end = tail
            .find("\n//--------------------- .text.")
            .or_else(|| tail.find("\n\t.section\t.text."))
            .unwrap_or(tail.len());
        Ok(&tail[..end])
    }

    fn validate_sass_has_no_atomics(sass: &str, spec: CandidateSpec) -> Result<(), String> {
        let entry = sass_entry(sass, spec.symbol)?;
        for line in entry.lines() {
            let instruction = line
                .split_once("*/")
                .map(|(_, tail)| tail.trim_start())
                .unwrap_or_else(|| line.trim_start());
            let opcode = instruction
                .split_whitespace()
                .find(|token| !token.starts_with('@'))
                .unwrap_or("");
            if opcode.starts_with("ATOM") || opcode.starts_with("RED.") {
                return Err(format!(
                    "compiled {} candidate contains an atomic: {line}",
                    spec.symbol
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn sass_atomic_gate_is_symbol_scoped_and_rejects_predicated_opcodes() {
        let spec = CANDIDATES[0];
        let unrelated = format!(
            "// Function : unrelated\n/*0000*/ ATOMG.E.ADD.F32 R0, [R2], R3;\n// Function : {}\n/*0010*/ FADD R0, R1, R2;\n",
            spec.symbol
        );
        validate_sass_has_no_atomics(&unrelated, spec).unwrap();
        let predicated = format!(
            "// Function : {}\n/*0000*/ @P0 ATOMG.E.ADD.F32 R0, [R2], R3;\n",
            spec.symbol
        );
        assert!(validate_sass_has_no_atomics(&predicated, spec).is_err());
    }

    fn validate_ptxas_case(
        label: &str,
        gpu: &str,
        ptx: String,
        specs: &[CandidateSpec],
    ) -> Result<(), String> {
        let directory =
            tempfile::tempdir().map_err(|error| format!("resource tempdir: {error}"))?;
        let ptx_path = directory.path().join(format!("{label}.ptx"));
        let cubin_path = directory.path().join(format!("{label}.cubin"));
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
            .arg("--verbose")
            .arg(format!("--gpu-name={gpu}"))
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
            return Err(format!("ptxas {gpu} failed:\n{report}"));
        }
        for spec in specs {
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
        let sass = String::from_utf8_lossy(&disassembly.stdout);
        for spec in specs {
            validate_sass_has_no_atomics(&sass, *spec)?;
        }
        Ok(())
    }

    fn validate_ptxas_resources_and_no_atomics() -> Result<(), String> {
        let mut nvrtc_major = 0;
        let mut nvrtc_minor = 0;
        let nvrtc_result =
            unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut nvrtc_major, &mut nvrtc_minor) };
        if nvrtc_result != cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS
            || (nvrtc_major, nvrtc_minor) != (13, 2)
        {
            return Err(format!(
                "requires exact NVRTC 13.2, found result={nvrtc_result:?} version={nvrtc_major}.{nvrtc_minor}"
            ));
        }
        let portable = candidate_specs()
            .iter()
            .copied()
            .filter(|spec| !spec.tma)
            .collect::<Vec<_>>();
        let specialized = candidate_specs()
            .iter()
            .copied()
            .filter(|spec| spec.tma)
            .collect::<Vec<_>>();
        validate_ptxas_case(
            "portable-sm89",
            "sm_89",
            compile_sm80_source("sm_89")?,
            &portable,
        )?;
        validate_ptxas_case(
            "portable-sm120",
            "sm_120",
            compile_sm80_source("sm_120")?,
            &portable,
        )?;
        validate_ptxas_case(
            "specialized-sm120",
            "sm_120",
            compile_sm120_source()?,
            &specialized,
        )
    }

    #[test]
    #[ignore = "requires CUDA 13.2 NVRTC, ptxas, and nvdisasm but launches no GPU work"]
    fn tn_underfill_candidates_meet_ptxas_resource_and_no_atomic_contract() -> Result<(), String> {
        validate_ptxas_resources_and_no_atomics()
    }

    #[test]
    #[ignore = "requires an exclusive CC8.9 CUDA 13.2 GPU"]
    fn tn_underfill_portable_candidates_meet_sm89_driver_resource_contract() -> Result<(), String> {
        let device = GpuDevice::new(0)?;
        let ctx = configure(&device, F32TriadPolicy::AllowDeterministicTf32V1)?;
        let portable = ctx
            .kernels
            .f32_triad_availability()
            .portable
            .ok_or_else(|| "SM89 portable TF32 module unavailable".to_string())?;
        let compiler = portable.compiler;
        let artifact = portable.artifact;
        if device.compute_capability != (8, 9)
            || portable.module_kind != ModuleKind::TriadSm80
            || portable.target.as_str() != "sm_89"
            || portable.device.compute_capability != (8, 9)
            || portable.device.target.as_str() != "sm_89"
            || portable.device_caps.compute_capability != (8, 9)
            || portable.device_caps.nvrtc_version != (13, 2)
            || portable.device_caps.accepted_target != Some(portable.target)
            || portable.device_caps.optin_shared_bytes == 0
            || compiler.target.as_str() != "sm_89"
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
            || artifact.module_kind != ModuleKind::TriadSm80
            || artifact.artifact_kind != compiler.output_kind
            || artifact.compile_key != compiler.invocation_digest
            || artifact.compile_key == [0; 32]
            || artifact.artifact_digest == [0; 32]
        {
            return Err(format!(
                "requires exact CUDA 13.2 SM89 portable artifact: {portable:?}"
            ));
        }
        let specs = candidate_specs()
            .iter()
            .copied()
            .filter(|spec| !spec.tma)
            .collect::<Vec<_>>();
        validate_driver_resources(&device, &specs)
    }

    #[test]
    #[ignore = "requires an exclusive CC12.0/170-SM CUDA 13.2 GPU"]
    fn tn_underfill_candidates_meet_sm120_driver_resource_contract() -> Result<(), String> {
        let device = GpuDevice::new(0)?;
        let ctx = configure(&device, F32TriadPolicy::AllowDeterministicTf32V1)?;
        validate_exact_environment(&ctx)?;
        validate_driver_resources(&device, candidate_specs())
    }

    fn validate_exact_cohort_witness(device: &GpuDevice) -> Result<(), String> {
        let ctx = configure(device, F32TriadPolicy::AllowDeterministicTf32V1)?;
        validate_exact_environment(&ctx)?;
        let request = PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Tn,
            DIMS,
            PhysicalQualificationRoute::Tf32Forced(Tf32PhysicalRoute::MmaTf32RnaV1(
                Tf32PortableRoute {
                    tile: Tf32PortableTile::M128N64,
                    stages: Tf32PortableStages::S2,
                },
            )),
            epilogue(),
        );
        let qualified = qualify_physical_launch(&ctx, request)?;
        let evidence = qualified.evidence();
        if evidence.launch_count() != 1
            || evidence.single_launch_symbol()
                != Some("gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s2")
            || evidence.uniform_module_kind() != Some(ModuleKind::TriadSm80)
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
        let module_kind = if spec.tma {
            ModuleKind::TriadSm120
        } else {
            ModuleKind::TriadSm80
        };
        let production_spec = tf32_kernel_spec(ResolvedGemmOp::Tn, candidate_route(spec))?;
        if production_spec.symbol != spec.symbol
            || production_spec.tile != spec.tile
            || production_spec.bk != spec.bk
            || production_spec.stages != spec.stages
            || production_spec.threads != spec.block.0
            || production_spec.dynamic_shared_bytes != spec.dynamic_shared_bytes
            || production_spec.module_kind != module_kind
        {
            return Err(format!(
                "candidate no longer matches the production TF32 inventory: host={spec:?} production={production_spec:?}"
            ));
        }
        let evidence = qualified.evidence();
        if evidence.launch_count() != 1
            || !evidence.eager_graph_equal()
            || evidence.uniform_module_kind() != Some(module_kind)
        {
            return Err(format!("candidate route evidence changed: {evidence:?}"));
        }
        let [node] = evidence.nodes() else {
            return Err("candidate must contain exactly one physical node".into());
        };
        if node.logical_op != ResolvedGemmOp::Tn
            || node.shape != DIMS
            || node.strides != (512, 384, 384)
            || node.tile != Some(spec.tile)
            || node.module_kind != module_kind
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
        let (a, dense_b, initial, expected) = literal_tn_oracle();
        let mut padded_b = vec![0.0_f32; 3 * 4];
        for row in 0..3 {
            padded_b[row * 4..row * 4 + 2].copy_from_slice(&dense_b[row * 2..row * 2 + 2]);
        }
        let fixture = new_direct_fixture(
            runtime,
            spec,
            (3, 4, 2),
            (4, 4, 2),
            a.to_vec(),
            padded_b,
            initial.to_vec(),
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
            qualification_seeded_values(m * n, CORPUS_SALT ^ 0x67),
            qualification_seeded_values(k * n, CORPUS_SALT ^ 0x91),
        )?;
        execute_direct_repeats(runtime, spec, fixture, expected, "large")
    }

    fn measure(
        launch: &mut QualifiedPhysicalLaunch<'_>,
        ctx: &GpuCtx,
        path: Path,
        iterations: usize,
        prevalidated_forced: bool,
    ) -> Result<f64, String> {
        if iterations == 0 {
            return Err("timing iterations must be positive".into());
        }
        let elapsed_ms = match (path, prevalidated_forced) {
            (Path::Eager, true) => {
                launch.measure_prevalidated_forced_eager_window_ms(ctx, iterations)?
            }
            (Path::Eager, false) => launch.measure_eager_window_ms(ctx, iterations)?,
            (Path::Graph, _) => launch.measure_graph_window_ms(ctx, iterations)?,
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
        prevalidated_forced: bool,
    ) -> Result<usize, String> {
        let pilot = measure(launch, ctx, path, 3, prevalidated_forced)?;
        Ok(((TARGET_WINDOW_MS * 1_000.0 / pilot).round() as usize).clamp(3, MAX_WINDOW_ITERATIONS))
    }

    fn paired(
        baseline: (&GpuCtx, &mut QualifiedPhysicalLaunch<'_>, bool),
        candidate: (&GpuCtx, &mut QualifiedPhysicalLaunch<'_>, bool),
        path: Path,
        order: Order,
        windows: usize,
    ) -> Result<TimingStats, String> {
        if windows == 0 {
            return Err("paired timing requires positive windows".into());
        }
        let (baseline_ctx, baseline, baseline_prevalidated) = baseline;
        let (candidate_ctx, candidate, candidate_prevalidated) = candidate;
        baseline.seed_f32_operands(baseline_ctx, CORPUS_SALT)?;
        candidate.seed_f32_operands(candidate_ctx, CORPUS_SALT)?;
        let baseline_iterations = calibrate(baseline, baseline_ctx, path, baseline_prevalidated)?;
        let candidate_iterations =
            calibrate(candidate, candidate_ctx, path, candidate_prevalidated)?;
        let iterations = baseline_iterations.max(candidate_iterations);
        measure(
            baseline,
            baseline_ctx,
            path,
            iterations,
            baseline_prevalidated,
        )?;
        measure(
            candidate,
            candidate_ctx,
            path,
            iterations,
            candidate_prevalidated,
        )?;
        let mut baseline_samples = Vec::with_capacity(windows);
        let mut candidate_samples = Vec::with_capacity(windows);
        let mut speedups = Vec::with_capacity(windows);
        for _ in 0..windows {
            let (b0, b1, c0, c1) = match order {
                Order::Abba => {
                    let b0 = measure(
                        baseline,
                        baseline_ctx,
                        path,
                        iterations,
                        baseline_prevalidated,
                    )?;
                    let c0 = measure(
                        candidate,
                        candidate_ctx,
                        path,
                        iterations,
                        candidate_prevalidated,
                    )?;
                    let c1 = measure(
                        candidate,
                        candidate_ctx,
                        path,
                        iterations,
                        candidate_prevalidated,
                    )?;
                    let b1 = measure(
                        baseline,
                        baseline_ctx,
                        path,
                        iterations,
                        baseline_prevalidated,
                    )?;
                    (b0, b1, c0, c1)
                }
                Order::Baab => {
                    let c0 = measure(
                        candidate,
                        candidate_ctx,
                        path,
                        iterations,
                        candidate_prevalidated,
                    )?;
                    let b0 = measure(
                        baseline,
                        baseline_ctx,
                        path,
                        iterations,
                        baseline_prevalidated,
                    )?;
                    let b1 = measure(
                        baseline,
                        baseline_ctx,
                        path,
                        iterations,
                        baseline_prevalidated,
                    )?;
                    let c1 = measure(
                        candidate,
                        candidate_ctx,
                        path,
                        iterations,
                        candidate_prevalidated,
                    )?;
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
            "tf32_tn_underfill pair={label} path={} order={} windows={} baseline_us_p50={:.6} candidate_us_p50={:.6} speedup_p05={:.9} speedup_p50={:.9} speedup_p95={:.9}",
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

    fn qualify_arm<'a>(ctx: &'a GpuCtx, arm: Arm) -> Result<QualifiedPhysicalLaunch<'a>, String> {
        let spec = arm.spec();
        let request = arm.request();
        presize_physical_qualification_suite(ctx, &[request])?;
        let qualified = qualify_physical_launch(ctx, request)?;
        qualified.validate_timed_request(ctx, request)?;
        validate_candidate_evidence(spec, &qualified)?;
        Ok(qualified)
    }

    fn qualify_candidate<'a>(
        ctx: &'a GpuCtx,
        spec: CandidateSpec,
    ) -> Result<QualifiedPhysicalLaunch<'a>, String> {
        qualify_arm(ctx, Arm::Forced(spec))
    }

    #[test]
    #[ignore = "requires an exclusive CC12.0/170-SM CUDA 13.2 GPU"]
    fn tn_underfill_candidates_are_exact_and_graph_stable() -> Result<(), String> {
        let device = GpuDevice::new(0)?;
        validate_exact_cohort_witness(&device)?;
        validate_ptxas_resources_and_no_atomics()?;
        validate_driver_resources(&device, candidate_specs())?;
        let reference_ctx = configure(&device, F32TriadPolicy::AllowDeterministicTf32V1)?;
        validate_exact_environment(&reference_ctx)?;
        let reference_request = portable_reference_request();
        presize_physical_qualification_suite(&reference_ctx, &[reference_request])?;
        let mut reference = qualify_physical_launch(&reference_ctx, reference_request)?;
        reference.validate_timed_request(&reference_ctx, reference_request)?;
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
        let reference_bits = run_repeated_bits(&mut reference, &reference_ctx, CORPUS_SALT)?;
        let direct = new_direct_runtime(&device)?;

        let mut identities = BTreeSet::new();
        for spec in candidate_specs() {
            run_direct_small_literal(&direct, *spec)?;
            run_direct_large_guarded(&direct, *spec, &reference_bits)?;
            let ctx = configure(&device, F32TriadPolicy::AllowDeterministicTf32V1)?;
            validate_exact_environment(&ctx)?;
            let mut candidate = qualify_candidate(&ctx, *spec)?;
            let identity = validate_candidate_evidence(*spec, &candidate)?;
            if !identities.insert(identity.arguments_digest) {
                return Err(format!("candidate argument digest collided: {identity:?}"));
            }
            let actual = run_repeated_bits(&mut candidate, &ctx, CORPUS_SALT)?;
            if actual != reference_bits {
                return Err(format!(
                    "{} differs bitwise from the direct portable TF32 reference",
                    spec.symbol
                ));
            }
        }
        let (production_identity, production_route_identity, production_bits) = {
            let ctx = configure(&device, F32TriadPolicy::AllowDeterministicTf32V1)?;
            validate_exact_environment(&ctx)?;
            let mut launch = qualify_arm(&ctx, Arm::Production)?;
            let identity = validate_candidate_evidence(CANDIDATES[0], &launch)?;
            let route_identity = *launch.evidence().route_identity();
            let bits = run_repeated_bits(&mut launch, &ctx, CORPUS_SALT)?;
            (identity, route_identity, bits)
        };
        let (forced_identity, forced_route_identity, forced_bits) = {
            let ctx = configure(&device, F32TriadPolicy::AllowDeterministicTf32V1)?;
            let mut launch = qualify_arm(&ctx, Arm::Forced(CANDIDATES[0]))?;
            let identity = validate_candidate_evidence(CANDIDATES[0], &launch)?;
            let route_identity = *launch.evidence().route_identity();
            let bits = run_repeated_bits(&mut launch, &ctx, CORPUS_SALT)?;
            (identity, route_identity, bits)
        };
        // Route identity binds policy, compiler, artifacts, targets, numeric
        // contracts, device capabilities, and revisions. Function identity is
        // the node symbol and module owner checked above. Raw pointer equality
        // is intentionally unavailable; the address-free argument digest binds
        // allocation generations and subviews instead.
        if production_route_identity != forced_route_identity
            || production_identity.symbol != forced_identity.symbol
            || production_identity.grid != forced_identity.grid
            || production_identity.block != forced_identity.block
            || production_identity.dynamic_shared_bytes != forced_identity.dynamic_shared_bytes
            || production_identity.arguments_digest != forced_identity.arguments_digest
        {
            return Err(format!(
                "production/forced winner physical identity differs: production={production_identity:?} forced={forced_identity:?}"
            ));
        }
        if production_bits != forced_bits || production_bits != reference_bits {
            return Err("production/forced winner bit parity changed".into());
        }
        Ok(())
    }

    fn official_pair(
        device: &GpuDevice,
        baseline_arm: Arm,
        candidate_arm: Arm,
        gate: TimingGate,
        quiet: &QuietGpu,
    ) -> Result<(), String> {
        quiet.require_cohort("tf32-tn-underfill-official-pair")?;
        let baseline_ctx = configure(device, F32TriadPolicy::AllowDeterministicTf32V1)?;
        let candidate_ctx = configure(device, F32TriadPolicy::AllowDeterministicTf32V1)?;
        let mut baseline = qualify_arm(&baseline_ctx, baseline_arm)?;
        let mut candidate = qualify_arm(&candidate_ctx, candidate_arm)?;
        for path in [Path::Graph, Path::Eager] {
            for order in [Order::Abba, Order::Baab] {
                let stats = paired(
                    (
                        &baseline_ctx,
                        &mut baseline,
                        baseline_arm.prevalidated_forced(),
                    ),
                    (
                        &candidate_ctx,
                        &mut candidate,
                        candidate_arm.prevalidated_forced(),
                    ),
                    path,
                    order,
                    OFFICIAL_WINDOWS,
                )?;
                print_stats(
                    &format!(
                        "{}->{}-official",
                        baseline_arm.label(),
                        candidate_arm.label()
                    ),
                    path,
                    order,
                    OFFICIAL_WINDOWS,
                    stats,
                );
                match gate {
                    TimingGate::HardBaseline => validate_official_gate(
                        OFFICIAL_WINDOWS,
                        stats.speedup_p05,
                        stats.speedup_p50,
                    )?,
                    TimingGate::Parity
                        if stats.speedup_p05 < PARITY_MIN || stats.speedup_p95 > PARITY_MAX =>
                    {
                        return Err(format!(
                            "production/forced winner parity failed: path={path:?} order={order:?} stats={stats:?}"
                        ));
                    }
                    TimingGate::Winner
                        if stats.speedup_p05 < WINNER_MIN_P05
                            || stats.speedup_p50 < WINNER_MIN_P50 =>
                    {
                        return Err(format!(
                            "winner is not stable against runner-up: path={path:?} order={order:?} stats={stats:?}"
                        ));
                    }
                    _ => {}
                }
            }
        }
        quiet.verify_post_cohort("tf32-tn-underfill-official-pair")?;
        Ok(())
    }

    fn report_validated_forced_overhead(
        device: &GpuDevice,
        quiet: &QuietGpu,
    ) -> Result<(), String> {
        quiet.require_cohort("tf32-tn-underfill-validated-overhead")?;
        let ctx = configure(device, F32TriadPolicy::AllowDeterministicTf32V1)?;
        let mut launch = qualify_arm(&ctx, Arm::Forced(CANDIDATES[0]))?;
        let validated_iterations = calibrate(&mut launch, &ctx, Path::Eager, false)?;
        let prevalidated_iterations = calibrate(&mut launch, &ctx, Path::Eager, true)?;
        let iterations = validated_iterations.max(prevalidated_iterations);
        let mut ratios = Vec::with_capacity(21);
        for window in 0_usize..21 {
            let (validated, prevalidated) = if window.is_multiple_of(2) {
                (
                    measure(&mut launch, &ctx, Path::Eager, iterations, false)?,
                    measure(&mut launch, &ctx, Path::Eager, iterations, true)?,
                )
            } else {
                let prevalidated = measure(&mut launch, &ctx, Path::Eager, iterations, true)?;
                let validated = measure(&mut launch, &ctx, Path::Eager, iterations, false)?;
                (validated, prevalidated)
            };
            ratios.push(validated / prevalidated);
        }
        eprintln!(
            "tf32_tn_underfill informational_validated_forced_overhead windows=21 ratio_p50={:.9}",
            percentile(&ratios, 0.50)?,
        );
        quiet
            .verify_post_cohort("tf32-tn-underfill-validated-overhead")
            .map(drop)
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC12.0/170-SM CUDA 13.2 GPU"]
    fn tn_underfill_tournament_abba_baab() -> Result<(), String> {
        validate_release_build()?;
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("tf32-tn-underfill-pre-context")?;
        let device = GpuDevice::new(0)?;
        validate_exact_cohort_witness(&device)?;
        validate_ptxas_resources_and_no_atomics()?;
        validate_driver_resources(&device, candidate_specs())?;
        report_validated_forced_overhead(&device, &quiet)?;
        official_pair(
            &device,
            Arm::Forced(CANDIDATES[3]),
            Arm::Production,
            TimingGate::HardBaseline,
            &quiet,
        )?;
        official_pair(
            &device,
            Arm::Production,
            Arm::Forced(CANDIDATES[0]),
            TimingGate::Parity,
            &quiet,
        )?;
        official_pair(
            &device,
            Arm::Forced(CANDIDATES[1]),
            Arm::Forced(CANDIDATES[0]),
            TimingGate::Winner,
            &quiet,
        )?;
        Ok(())
    }
}
