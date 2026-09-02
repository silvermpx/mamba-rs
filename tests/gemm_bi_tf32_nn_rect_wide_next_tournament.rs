use std::collections::BTreeSet;

#[cfg(feature = "cuda")]
mod common;

const DIMS: (usize, usize, usize) = (512, 3_072, 768);
const SCREENING_WINDOWS: usize = 21;
const OFFICIAL_WINDOWS: usize = 101;
const SCREENING_MIN_P05: f64 = 1.02;
const SCREENING_MIN_P50: f64 = 1.05;
const WINNER_MIN_P05: f64 = 1.00;
const WINNER_MIN_P50: f64 = 1.01;
const PARITY_MIN_P05: f64 = 0.97;
const PARITY_MAX_P05: f64 = 1.03;
const PARITY_MIN_P50: f64 = 0.98;
const PARITY_MAX_P50: f64 = 1.02;
const GUARD_ELEMENTS: usize = 32;
const GUARD_BITS: u32 = 0x7fc1_4e58;
const PRODUCTION_TUNING_TABLE_REVISION: u16 = 36;
const CUDA_SOURCE: &str = include_str!("gemm_bi_tf32_nn_rect_wide_next_tournament.cu");

fn validate_production_tuning_revision(actual: u16) -> Result<(), String> {
    (actual == PRODUCTION_TUNING_TABLE_REVISION)
        .then_some(())
        .ok_or_else(|| {
            format!(
                "production tag33 tuning revision changed: expected={} actual={actual}",
                PRODUCTION_TUNING_TABLE_REVISION
            )
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum CandidateKind {
    M80N32S2,
    M64N40S2,
    M80N32S3Producer,
    M80N32Bk64S2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CandidateSpec {
    kind: CandidateKind,
    symbol: &'static str,
    tile: (u32, u32),
    storage_n: u32,
    logical_bk: u32,
    map_bk: u32,
    stages: u8,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    dynamic_shared_bytes: u32,
    minimum_occupancy: u32,
    maximum_registers: u32,
    producer_warp: bool,
}

const CANDIDATES: [CandidateSpec; 4] = [
    CandidateSpec {
        kind: CandidateKind::M80N32S2,
        symbol: "gemm_bi_nn_sm120_tma_mma_tf32_exp_m80n32_bk32_s2_v1",
        tile: (80, 32),
        storage_n: 32,
        logical_bk: 32,
        map_bk: 32,
        stages: 2,
        grid: (168, 1, 1),
        block: (160, 1, 1),
        dynamic_shared_bytes: 28_800,
        minimum_occupancy: 3,
        maximum_registers: 80,
        producer_warp: false,
    },
    CandidateSpec {
        kind: CandidateKind::M64N40S2,
        symbol: "gemm_bi_nn_sm120_tma_mma_tf32_exp_m64n40_bk32_s2_v1",
        tile: (64, 40),
        storage_n: 64,
        logical_bk: 32,
        map_bk: 32,
        stages: 2,
        grid: (160, 1, 1),
        block: (128, 1, 1),
        dynamic_shared_bytes: 32_896,
        minimum_occupancy: 3,
        maximum_registers: 80,
        producer_warp: false,
    },
    CandidateSpec {
        kind: CandidateKind::M80N32S3Producer,
        symbol: "gemm_bi_nn_sm120_tma_mma_tf32_exp_m80n32_bk32_s3_producer_v1",
        tile: (80, 32),
        storage_n: 32,
        logical_bk: 32,
        map_bk: 32,
        stages: 3,
        grid: (168, 1, 1),
        block: (192, 1, 1),
        dynamic_shared_bytes: 43_136,
        minimum_occupancy: 2,
        maximum_registers: 80,
        producer_warp: true,
    },
    CandidateSpec {
        kind: CandidateKind::M80N32Bk64S2,
        symbol: "gemm_bi_nn_sm120_tma_mma_tf32_exp_m80n32_bk64_s2_dual32_v1",
        tile: (80, 32),
        storage_n: 32,
        logical_bk: 64,
        map_bk: 32,
        stages: 2,
        grid: (168, 1, 1),
        block: (160, 1, 1),
        dynamic_shared_bytes: 57_472,
        minimum_occupancy: 1,
        maximum_registers: 80,
        producer_warp: false,
    },
];

fn production_spec() -> CandidateSpec {
    CandidateSpec {
        symbol: "gemm_bi_nn_sm120_tma_mma_tf32_v1_m80n32_bk64_s2",
        ..CANDIDATES[3]
    }
}

fn validate_production_spec(spec: CandidateSpec) -> Result<(), String> {
    if spec != production_spec() {
        return Err(format!("tag33 production specification changed: {spec:?}"));
    }
    validate_spec(spec)
}

type LaunchGeometry = ((u32, u32, u32), (u32, u32, u32), u32);

fn launch_geometry(
    spec: CandidateSpec,
    dims: (usize, usize, usize),
) -> Result<LaunchGeometry, String> {
    let (m, _, n) = dims;
    let grid = m
        .div_ceil(spec.tile.0 as usize)
        .checked_mul(n.div_ceil(spec.tile.1 as usize))
        .ok_or_else(|| "launch grid overflow".to_string())?;
    Ok((
        (
            grid.try_into()
                .map_err(|_| "launch grid exceeds u32".to_string())?,
            1,
            1,
        ),
        spec.block,
        spec.dynamic_shared_bytes,
    ))
}

#[test]
fn rect_wide_production_fixture_is_exact_m80n32_bk64_and_rejects_old_launch_geometry() {
    let spec = production_spec();
    assert_eq!(
        spec.symbol,
        "gemm_bi_nn_sm120_tma_mma_tf32_v1_m80n32_bk64_s2"
    );
    assert_eq!(spec.tile, (80, 32));
    assert_eq!((spec.logical_bk, spec.map_bk, spec.stages), (64, 32, 2));
    assert_eq!(spec.block, (160, 1, 1));
    assert_eq!(spec.dynamic_shared_bytes, 57_472);
    assert_eq!(map_keys(spec)[0].box_dimensions, [32, 80]);
    assert_eq!(map_keys(spec)[1].box_dimensions, [32, 32]);
    assert_eq!(
        launch_geometry(spec, DIMS).unwrap(),
        ((168, 1, 1), (160, 1, 1), 57_472)
    );

    let mut old = spec;
    old.tile = (64, 64);
    old.block = (128, 1, 1);
    old.dynamic_shared_bytes = 32_896;
    assert_ne!(
        launch_geometry(old, DIMS).unwrap(),
        launch_geometry(spec, DIMS).unwrap()
    );
    assert!(validate_production_spec(old).is_err());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiteralMapKey {
    dimensions: [u64; 2],
    outer_byte_stride: u64,
    box_dimensions: [u32; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PhysicalIdentity {
    symbol: String,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    dynamic_shared_bytes: u32,
    arguments_digest: [u8; 32],
    launch_digest: [u8; 32],
    graph_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimingMode {
    Screening,
    Official,
}

fn candidate_specs() -> &'static [CandidateSpec; 4] {
    &CANDIDATES
}

fn map_keys(spec: CandidateSpec) -> [LiteralMapKey; 2] {
    let (m, k, n) = DIMS;
    [
        LiteralMapKey {
            dimensions: [k as u64, m as u64],
            outer_byte_stride: (k * std::mem::size_of::<f32>()) as u64,
            box_dimensions: [spec.map_bk, spec.tile.0],
        },
        LiteralMapKey {
            dimensions: [n as u64, k as u64],
            outer_byte_stride: (n * std::mem::size_of::<f32>()) as u64,
            box_dimensions: [spec.map_bk, spec.map_bk],
        },
    ]
}

fn validate_map_keys(spec: CandidateSpec, actual: &[LiteralMapKey; 2]) -> Result<(), String> {
    let expected = map_keys(spec);
    if actual != &expected {
        return Err(format!(
            "candidate tensor-map identity changed: expected={expected:?} actual={actual:?}"
        ));
    }
    Ok(())
}

fn validate_spec(spec: CandidateSpec) -> Result<(), String> {
    let (m, _, n) = DIMS;
    let grid = m
        .div_ceil(spec.tile.0 as usize)
        .checked_mul(n.div_ceil(spec.tile.1 as usize))
        .ok_or_else(|| "candidate grid overflow".to_string())?;
    let slabs = spec.logical_bk / spec.map_bk;
    let stage_bytes = (spec.tile.0 + spec.storage_n)
        .checked_mul(spec.map_bk)
        .and_then(|value| value.checked_mul(4))
        .and_then(|value| value.checked_mul(slabs))
        .ok_or_else(|| "candidate shared-memory size overflow".to_string())?;
    let dynamic = 128_u32
        .checked_add(
            stage_bytes
                .checked_mul(u32::from(spec.stages))
                .ok_or_else(|| "candidate staged shared-memory size overflow".to_string())?,
        )
        .ok_or_else(|| "candidate dynamic shared-memory size overflow".to_string())?;
    let compute_warps = spec.tile.0 / 16;
    let expected_threads = (compute_warps + u32::from(spec.producer_warp)) * 32;
    if spec.map_bk != 32
        || !matches!(spec.logical_bk, 32 | 64)
        || !spec.logical_bk.is_multiple_of(spec.map_bk)
        || spec.storage_n < spec.tile.1
        || !spec.storage_n.is_multiple_of(32)
        || spec.grid != (grid as u32, 1, 1)
        || spec.block != (expected_threads, 1, 1)
        || spec.dynamic_shared_bytes != dynamic
        || spec.minimum_occupancy == 0
        || spec.maximum_registers == 0
    {
        return Err(format!("invalid candidate specification: {spec:?}"));
    }
    Ok(())
}

fn expected_identity(spec: CandidateSpec) -> PhysicalIdentity {
    PhysicalIdentity {
        symbol: spec.symbol.to_owned(),
        grid: spec.grid,
        block: spec.block,
        dynamic_shared_bytes: spec.dynamic_shared_bytes,
        arguments_digest: [1; 32],
        launch_digest: [2; 32],
        graph_digest: [3; 32],
    }
}

fn validate_identity(spec: CandidateSpec, identity: &PhysicalIdentity) -> Result<(), String> {
    if identity.symbol != spec.symbol
        || identity.grid != spec.grid
        || identity.block != spec.block
        || identity.dynamic_shared_bytes != spec.dynamic_shared_bytes
    {
        return Err(format!("physical launch identity mismatch: {identity:?}"));
    }
    let digests = [
        identity.arguments_digest,
        identity.launch_digest,
        identity.graph_digest,
    ];
    if digests.contains(&[0; 32]) || digests.iter().collect::<BTreeSet<_>>().len() != digests.len()
    {
        return Err("physical launch digests must be nonzero and domain-distinct".into());
    }
    Ok(())
}

fn validate_eager_graph_identity(
    spec: CandidateSpec,
    eager: &PhysicalIdentity,
    graph: &PhysicalIdentity,
) -> Result<(), String> {
    validate_identity(spec, eager)?;
    validate_identity(spec, graph)?;
    if eager != graph {
        return Err(format!(
            "eager and captured graph physical identities differ: eager={eager:?} graph={graph:?}"
        ));
    }
    Ok(())
}

fn validate_cross_candidate_identity_uniqueness(
    identities: &[PhysicalIdentity],
) -> Result<(), String> {
    let mut launch_digests = BTreeSet::new();
    let mut graph_digests = BTreeSet::new();
    for identity in identities {
        if !launch_digests.insert(identity.launch_digest)
            || !graph_digests.insert(identity.graph_digest)
        {
            return Err(format!(
                "{} physical launch or graph digest collided",
                identity.symbol
            ));
        }
    }
    Ok(())
}

fn exceptional_single_term_cases() -> &'static [(&'static str, u32, u32); 8] {
    &[
        ("positive-zero", 0x0000_0000, 0x0000_0000),
        ("negative-zero", 0x8000_0000, 0x0000_0000),
        ("minimum-subnormal", 0x0000_0001, 0x0000_0000),
        ("maximum-subnormal", 0x007f_ffff, 0x0080_0000),
        ("positive-infinity", 0x7f80_0000, 0x7f80_0000),
        ("negative-infinity", 0xff80_0000, 0xff80_0000),
        ("quiet-nan", 0x7fc1_2345, 0x7fff_ffff),
        ("signaling-nan", 0x7f81_2345, 0x7fff_ffff),
    ]
}

fn guarded_active_bits(
    snapshot: &[f32],
    active_len: usize,
    read_only: Option<&[f32]>,
) -> Result<Vec<u32>, String> {
    let expected_len = active_len
        .checked_add(2 * GUARD_ELEMENTS)
        .ok_or_else(|| "guarded extent overflow".to_string())?;
    if snapshot.len() != expected_len || active_len == 0 {
        return Err("guarded snapshot length mismatch".into());
    }
    if snapshot[..GUARD_ELEMENTS]
        .iter()
        .chain(&snapshot[GUARD_ELEMENTS + active_len..])
        .any(|value| value.to_bits() != GUARD_BITS)
    {
        return Err("guarded snapshot red zone changed".into());
    }
    let active = &snapshot[GUARD_ELEMENTS..GUARD_ELEMENTS + active_len];
    if read_only.is_some_and(|expected| {
        expected.len() != active.len()
            || expected
                .iter()
                .zip(active)
                .any(|(expected, actual)| expected.to_bits() != actual.to_bits())
    }) {
        return Err("read-only active payload changed".into());
    }
    Ok(active.iter().map(|value| value.to_bits()).collect())
}

fn percentile(samples: &[f64], fraction: f64) -> Result<f64, String> {
    if samples.is_empty()
        || samples
            .iter()
            .any(|sample| !sample.is_finite() || *sample <= 0.0)
        || !(fraction > 0.0 && fraction <= 1.0)
    {
        return Err("percentile requires positive finite samples".into());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Ok(sorted[index])
}

fn cublas_geometry(dims: (usize, usize, usize)) -> Result<(i32, i32, i32, i32, i32, i32), String> {
    let (m, k, n) = dims;
    if m == 0 || k == 0 || n == 0 {
        return Err(format!("NN dimensions must be positive: {dims:?}"));
    }
    m.checked_mul(k)
        .and_then(|_| k.checked_mul(n))
        .and_then(|_| m.checked_mul(n))
        .ok_or_else(|| format!("NN dimensions overflow: {dims:?}"))?;
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

fn validate_timing_gate(
    mode: TimingMode,
    windows: usize,
    p05: f64,
    p50: f64,
) -> Result<(), String> {
    let (expected_windows, minimum_p05, minimum_p50) = match mode {
        TimingMode::Screening => (SCREENING_WINDOWS, SCREENING_MIN_P05, SCREENING_MIN_P50),
        TimingMode::Official => (OFFICIAL_WINDOWS, SCREENING_MIN_P05, SCREENING_MIN_P50),
    };
    if windows != expected_windows
        || [p05, p50]
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0)
        || p05 < minimum_p05
        || p50 < minimum_p50
    {
        return Err(format!(
            "timing gate failed: mode={mode:?} windows={windows} p05={p05} p50={p50}"
        ));
    }
    Ok(())
}

fn validate_parity_gate(p05: f64, p50: f64) -> Result<(), String> {
    if !p05.is_finite()
        || !p50.is_finite()
        || !(PARITY_MIN_P05..=PARITY_MAX_P05).contains(&p05)
        || !(PARITY_MIN_P50..=PARITY_MAX_P50).contains(&p50)
    {
        return Err(format!("production parity failed: p05={p05} p50={p50}"));
    }
    Ok(())
}

#[test]
fn ranked_candidates_have_exact_geometry_and_resources() {
    assert_eq!(candidate_specs().len(), 4);
    assert_eq!(
        candidate_specs().map(|spec| spec.kind),
        [
            CandidateKind::M80N32S2,
            CandidateKind::M64N40S2,
            CandidateKind::M80N32S3Producer,
            CandidateKind::M80N32Bk64S2,
        ]
    );
    for spec in candidate_specs() {
        validate_spec(*spec).unwrap();
    }
    assert_eq!(candidate_specs()[0].grid.0, 168);
    assert_eq!(
        candidate_specs().map(|spec| spec.minimum_occupancy),
        [3, 3, 2, 1]
    );
    assert_eq!(candidate_specs()[1].grid.0, 160);
    assert_eq!(candidate_specs()[2].block.0, 192);
    assert_eq!(candidate_specs()[3].logical_bk, 64);
    assert_eq!(candidate_specs()[3].map_bk, 32);
    assert_eq!(
        candidate_specs()
            .iter()
            .map(|spec| spec.symbol)
            .collect::<BTreeSet<_>>()
            .len(),
        4
    );
}

#[test]
fn production_witness_freezes_the_current_revision_and_rejects_the_previous() {
    assert_eq!(PRODUCTION_TUNING_TABLE_REVISION, 36);
    validate_production_tuning_revision(36).unwrap();
    assert!(validate_production_tuning_revision(35).is_err());
}

#[test]
fn tma_keys_are_literal_and_bk64_keeps_two_bk32_slabs() {
    for spec in candidate_specs() {
        let [a, b] = map_keys(*spec);
        assert_eq!(a.dimensions, [3_072, 512]);
        assert_eq!(a.outer_byte_stride, 12_288);
        assert_eq!(a.box_dimensions, [32, spec.tile.0]);
        assert_eq!(b.dimensions, [768, 3_072]);
        assert_eq!(b.outer_byte_stride, 3_072);
        assert_eq!(b.box_dimensions, [32, 32]);
        validate_map_keys(*spec, &[a, b]).unwrap();
    }
    let spec = candidate_specs()[0];
    for mutate in [
        |keys: &mut [LiteralMapKey; 2]| keys[0].dimensions[0] += 1,
        |keys: &mut [LiteralMapKey; 2]| keys[0].outer_byte_stride += 4,
        |keys: &mut [LiteralMapKey; 2]| keys[0].box_dimensions[1] += 16,
        |keys: &mut [LiteralMapKey; 2]| keys[1].dimensions[1] -= 1,
        |keys: &mut [LiteralMapKey; 2]| keys[1].box_dimensions[0] = 64,
    ] {
        let mut changed = map_keys(spec);
        mutate(&mut changed);
        assert!(validate_map_keys(spec, &changed).is_err());
    }
}

#[test]
fn geometry_and_identity_mutations_fail_closed() {
    let spec = candidate_specs()[0];
    for mutate in [
        |value: &mut CandidateSpec| value.grid.0 += 1,
        |value: &mut CandidateSpec| value.block.0 += 32,
        |value: &mut CandidateSpec| value.dynamic_shared_bytes += 128,
        |value: &mut CandidateSpec| value.map_bk = 64,
        |value: &mut CandidateSpec| value.storage_n = 48,
        |value: &mut CandidateSpec| value.minimum_occupancy = 0,
    ] {
        let mut changed = spec;
        mutate(&mut changed);
        assert!(validate_spec(changed).is_err());
    }
    let expected = expected_identity(spec);
    validate_identity(spec, &expected).unwrap();
    for mutate in [
        |value: &mut PhysicalIdentity| value.grid.0 += 1,
        |value: &mut PhysicalIdentity| value.block.0 += 1,
        |value: &mut PhysicalIdentity| value.dynamic_shared_bytes += 1,
        |value: &mut PhysicalIdentity| value.arguments_digest = [0; 32],
        |value: &mut PhysicalIdentity| value.launch_digest = value.arguments_digest,
        |value: &mut PhysicalIdentity| value.graph_digest = value.launch_digest,
    ] {
        let mut changed = expected.clone();
        mutate(&mut changed);
        assert!(validate_identity(spec, &changed).is_err());
    }

    let eager = expected_identity(spec);
    validate_eager_graph_identity(spec, &eager, &eager).unwrap();
    for mutate in [
        |value: &mut PhysicalIdentity| value.symbol.push_str("_wrong"),
        |value: &mut PhysicalIdentity| value.grid.0 += 1,
        |value: &mut PhysicalIdentity| value.block.0 += 1,
        |value: &mut PhysicalIdentity| value.dynamic_shared_bytes += 1,
        |value: &mut PhysicalIdentity| value.arguments_digest = [9; 32],
    ] {
        let mut changed = eager.clone();
        mutate(&mut changed);
        assert!(validate_eager_graph_identity(spec, &eager, &changed).is_err());
    }
}

#[test]
fn cross_candidate_identity_allows_argument_reuse_but_rejects_launch_or_graph_collisions() {
    let first_spec = candidate_specs()[0];
    let second_spec = candidate_specs()[2];
    let first = expected_identity(first_spec);
    let mut second = expected_identity(second_spec);
    second.arguments_digest = first.arguments_digest;
    second.launch_digest = [4; 32];
    second.graph_digest = [5; 32];

    validate_identity(first_spec, &first).unwrap();
    validate_identity(second_spec, &second).unwrap();
    validate_cross_candidate_identity_uniqueness(&[first.clone(), second.clone()]).unwrap();

    let mut launch_collision = second.clone();
    launch_collision.launch_digest = first.launch_digest;
    assert!(
        validate_cross_candidate_identity_uniqueness(&[first.clone(), launch_collision]).is_err()
    );

    let mut graph_collision = second;
    graph_collision.graph_digest = first.graph_digest;
    assert!(
        validate_cross_candidate_identity_uniqueness(&[first.clone(), graph_collision]).is_err()
    );
    assert!(validate_cross_candidate_identity_uniqueness(&[first.clone(), first]).is_err());
}

#[test]
fn isolated_exceptional_oracle_has_frozen_expected_bits() {
    assert_eq!(
        exceptional_single_term_cases(),
        &[
            ("positive-zero", 0x0000_0000, 0x0000_0000),
            ("negative-zero", 0x8000_0000, 0x0000_0000),
            ("minimum-subnormal", 0x0000_0001, 0x0000_0000),
            ("maximum-subnormal", 0x007f_ffff, 0x0080_0000),
            ("positive-infinity", 0x7f80_0000, 0x7f80_0000),
            ("negative-infinity", 0xff80_0000, 0xff80_0000),
            ("quiet-nan", 0x7fc1_2345, 0x7fff_ffff),
            ("signaling-nan", 0x7f81_2345, 0x7fff_ffff),
        ]
    );
}

#[test]
fn timing_and_guard_validators_reject_every_bad_value() {
    let good = (1..=101).map(|value| value as f64).collect::<Vec<_>>();
    assert_eq!(percentile(&good, 0.05).unwrap(), 6.0);
    for samples in [
        vec![],
        vec![0.0],
        vec![-1.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
    ] {
        assert!(percentile(&samples, 0.5).is_err());
    }
    assert!(validate_timing_gate(TimingMode::Screening, 21, 1.02, 1.05).is_ok());
    assert!(validate_timing_gate(TimingMode::Official, 101, 1.02, 1.05).is_ok());
    assert_eq!((WINNER_MIN_P05, WINNER_MIN_P50), (1.00, 1.01));
    assert!(validate_parity_gate(1.0, 1.0).is_ok());
    for (p05, p50) in [
        (0.96, 1.0),
        (1.04, 1.0),
        (1.0, 0.97),
        (1.0, 1.03),
        (f64::NAN, 1.0),
    ] {
        assert!(validate_parity_gate(p05, p50).is_err());
    }
    for windows in [0, 1, 20, 22, 100, 102] {
        assert!(validate_timing_gate(TimingMode::Screening, windows, 2.0, 2.0).is_err());
        assert!(validate_timing_gate(TimingMode::Official, windows, 2.0, 2.0).is_err());
    }
    for (p05, p50) in [
        (0.0, 2.0),
        (f64::NAN, 2.0),
        (2.0, f64::INFINITY),
        (1.019, 2.0),
        (2.0, 1.049),
    ] {
        assert!(validate_timing_gate(TimingMode::Screening, 21, p05, p50).is_err());
    }

    let active = [1.0_f32, -2.0, 3.0];
    let mut snapshot = vec![f32::from_bits(GUARD_BITS); active.len() + 2 * GUARD_ELEMENTS];
    snapshot[GUARD_ELEMENTS..GUARD_ELEMENTS + active.len()].copy_from_slice(&active);
    assert_eq!(
        guarded_active_bits(&snapshot, active.len(), Some(&active)).unwrap(),
        active.map(f32::to_bits)
    );
    for index in [0, GUARD_ELEMENTS + active.len()] {
        let mut changed = snapshot.clone();
        changed[index] = 0.0;
        assert!(guarded_active_bits(&changed, active.len(), Some(&active)).is_err());
    }
    assert_eq!(
        cublas_geometry(DIMS).unwrap(),
        (768, 512, 3_072, 768, 3_072, 768)
    );
    assert_eq!(cublas_geometry((3, 4, 2)).unwrap(), (2, 3, 4, 2, 4, 2));
    assert!(cublas_geometry((0, 4, 2)).is_err());
}

#[test]
fn actual_cuda_source_has_four_five_argument_entries_and_no_forbidden_route() {
    for spec in candidate_specs() {
        assert_eq!(CUDA_SOURCE.matches(spec.symbol).count(), 1);
    }
    assert!(!CUDA_SOURCE.contains("m64n128"));
    assert!(!CUDA_SOURCE.contains("m128n64"));
    assert!(!CUDA_SOURCE.contains("split_k"));
    assert!(!CUDA_SOURCE.contains("blockIdx.z"));
    assert!(!CUDA_SOURCE.to_ascii_lowercase().contains("atomic"));
    assert!(CUDA_SOURCE.contains("sm120_tf32_next_issue_stage"));
    assert!(CUDA_SOURCE.contains("sm120_tf32_next_produce_stage"));
    assert!(CUDA_SOURCE.contains("issue * 8"));
    assert!(CUDA_SOURCE.contains("slab * 32"));
    assert!(CUDA_SOURCE.contains("sm120_expect_transaction<stage_bytes>"));
    for entry in CUDA_SOURCE.split("extern \"C\" __global__").skip(1) {
        let signature = entry.split(") {").next().expect("entry signature");
        assert_eq!(
            signature.matches(',').count(),
            4,
            "entry ABI must have five arguments: {signature}"
        );
    }
}

#[test]
fn every_candidate_has_a_strict_no_spill_register_budget() {
    assert_eq!(
        candidate_specs()
            .iter()
            .map(|spec| (spec.kind, spec.maximum_registers))
            .collect::<Vec<_>>(),
        vec![
            (CandidateKind::M80N32S2, 80),
            (CandidateKind::M64N40S2, 80),
            (CandidateKind::M80N32S3Producer, 80),
            (CandidateKind::M80N32Bk64S2, 80),
        ]
    );
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
        presize_physical_qualification_suite, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, FramedSha256, ModuleKind,
        NUMERIC_ABI_REVISION, ResolvedGemmOp, SCHEDULE_REVISION,
    };
    use std::ffi::{CStr, c_void};
    use std::mem::size_of;
    use std::sync::Arc;

    const PRODUCTION_SYMBOL: &str = "gemm_bi_nn_sm120_tma_mma_tf32_v1_m80n32_bk64_s2";
    const TARGET_WINDOW_MS: f64 = 10.0;
    const MAX_WINDOW_ITERATIONS: usize = 16_384;
    const CORPUS_SALT: u64 = 0x8050_3240_6402;

    fn configure(device: &GpuDevice) -> Result<GpuCtx, String> {
        let ctx = GpuCtx::new(device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
        Ok(ctx)
    }

    fn validate_exact_environment(ctx: &GpuCtx) -> Result<(), String> {
        let specialized = ctx
            .kernels
            .f32_triad_availability()
            .specialized
            .ok_or_else(|| "next tournament requires the specialized TF32 module".to_string())?;
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
                "next tournament requires exact CUDA13.2 compute120 CC12.0/170-SM artifact: {specialized:?}"
            ));
        }
        Ok(())
    }

    fn production_request() -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Nn,
            DIMS,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::AllowDeterministicTf32V1),
            PhysicalQualificationF32Epilogue::new(1.0, 0.0, false),
        )
    }

    fn validate_production_witness(ctx: &GpuCtx) -> Result<(), String> {
        let request = production_request();
        presize_physical_qualification_suite(ctx, &[request])?;
        let qualified = qualify_physical_launch(ctx, request)?;
        qualified.validate_timed_request(ctx, request)?;
        let evidence = qualified.evidence();
        let [node] = evidence.nodes() else {
            return Err(format!(
                "tag33 production witness must contain one node: {:?}",
                evidence.nodes()
            ));
        };
        if validate_production_tuning_revision(evidence.route_identity().tuning_table_revision)
            .is_err()
            || !evidence.eager_graph_equal()
            || evidence.uniform_module_kind() != Some(ModuleKind::TriadSm120)
            || node.logical_op != ResolvedGemmOp::Nn
            || node.shape != DIMS
            || node.strides != (3_072, 768, 768)
            || node.symbol != PRODUCTION_SYMBOL
            || node.tile != Some((80, 32))
            || node.launch.grid_dim != (168, 1, 1)
            || node.launch.block_dim != (160, 1, 1)
            || node.launch.shared_mem_bytes != 57_472
            || node.launch.arguments_digest == [0; 32]
            || evidence.launch_digest() == [0; 32]
            || evidence.request_identity_digest() == [0; 32]
        {
            return Err(format!(
                "tag33 production physical identity changed: evidence={evidence:?} node={node:?}"
            ));
        }
        Ok(())
    }

    fn composed_source() -> String {
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
            (
                "tests/gemm_bi_tf32_nn_rect_wide_next_tournament.cu",
                CUDA_SOURCE,
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
        cudarc::nvrtc::compile_ptx_with_opts(composed_source(), options)
            .map(|ptx| ptx.to_src())
            .map_err(|error| format!("compile next SM120 TF32 tournament source: {error:?}"))
    }

    fn tool(name: &str) -> std::path::PathBuf {
        std::env::var("CUDA_HOME")
            .map(|root| std::path::PathBuf::from(root).join("bin").join(name))
            .unwrap_or_else(|_| name.into())
    }

    fn metric_before(line: &str, suffix: &str) -> Option<u64> {
        line.split_once(suffix)?
            .0
            .split(|character: char| !character.is_ascii_digit())
            .rfind(|field| !field.is_empty())?
            .parse()
            .ok()
    }

    fn validate_ptxas_entry(report: &str, spec: CandidateSpec) -> Result<(), String> {
        let marker = format!("Compiling entry function '{}'", spec.symbol);
        if report.matches(&marker).count() != 1 {
            return Err(format!("{} requires one ptxas entry", spec.symbol));
        }
        let block = report
            .split_once(&marker)
            .expect("entry count checked")
            .1
            .split("Compiling entry function '")
            .next()
            .expect("entry report block");
        if !block.contains("0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads") {
            return Err(format!(
                "{} has stack or spill traffic:\n{block}",
                spec.symbol
            ));
        }
        let usage = block
            .lines()
            .find(|line| line.contains("Used ") && line.contains(" registers"))
            .ok_or_else(|| format!("{} lost ptxas usage", spec.symbol))?;
        let registers = metric_before(usage, " registers")
            .ok_or_else(|| format!("{} malformed ptxas usage: {usage}", spec.symbol))?;
        let static_shared = metric_before(usage, " bytes smem").unwrap_or(0);
        if registers == 0 || registers > u64::from(spec.maximum_registers) || static_shared != 0 {
            return Err(format!(
                "{} ptxas budget failed: regs={registers}/{} static_shared={static_shared}/0",
                spec.symbol, spec.maximum_registers
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

    fn validate_sass_entry(sass: &str, spec: CandidateSpec) -> Result<(), String> {
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
                    "compiled {} contains a forbidden instruction: {line}",
                    spec.symbol
                ));
            }
        }
        Ok(())
    }

    fn validate_sass(sass: &str) -> Result<(), String> {
        validate_sass_entry(sass, production_spec())?;
        for spec in candidate_specs() {
            validate_sass_entry(sass, *spec)?;
        }
        Ok(())
    }

    #[test]
    fn sass_gate_is_symbol_scoped_and_rejects_predicated_atomics() {
        let spec = CANDIDATES[0];
        let unrelated = format!(
            "// Function : unrelated\n/*0000*/ ATOMG.E.ADD.F32 R0, [R2], R3;\n// Function : {}\n/*0010*/ FADD R0, R1, R2;\n",
            spec.symbol
        );
        validate_sass_entry(&unrelated, spec).unwrap();
        let predicated = format!(
            "// Function : {}\n/*0000*/ @P0 ATOMG.E.ADD.F32 R0, [R2], R3;\n",
            spec.symbol
        );
        assert!(validate_sass_entry(&predicated, spec).is_err());
        assert!(
            validate_sass_entry("// Function : unrelated\n/*0000*/ FADD R0, R1, R2;\n", spec)
                .is_err()
        );
    }

    fn validate_ptxas_resources_and_sass() -> Result<(), String> {
        let ptx = compile_sm120_source()?;
        let directory =
            tempfile::tempdir().map_err(|error| format!("resource tempdir: {error}"))?;
        let ptx_path = directory.path().join("next-rect-wide.ptx");
        let cubin_path = directory.path().join("next-rect-wide.cubin");
        std::fs::write(&ptx_path, ptx).map_err(|error| format!("write PTX: {error}"))?;
        let version = std::process::Command::new(tool("ptxas"))
            .arg("--version")
            .output()
            .map_err(|error| format!("ptxas --version: {error}"))?;
        let version_text = format!(
            "{}{}",
            String::from_utf8_lossy(&version.stdout),
            String::from_utf8_lossy(&version.stderr)
        );
        if !version.status.success() || !version_text.contains("release 13.2") {
            return Err(format!("requires exact ptxas 13.2: {version_text}"));
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
            return Err(format!("ptxas sm120 failed:\n{report}"));
        }
        validate_ptxas_entry(&report, production_spec())?;
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
        validate_sass(&String::from_utf8_lossy(&disassembly.stdout))
    }

    struct ResourceModule {
        _module: Arc<CudaModule>,
        production: CudaFunction,
        candidates: Vec<(CandidateSpec, CudaFunction)>,
    }

    impl ResourceModule {
        fn candidate(&self, spec: CandidateSpec) -> Result<&CudaFunction, String> {
            self.candidates
                .iter()
                .find_map(|(candidate, function)| (*candidate == spec).then_some(function))
                .ok_or_else(|| format!("loaded module lost {}", spec.symbol))
        }
    }

    fn load_resource_module(device: &GpuDevice) -> Result<ResourceModule, String> {
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(compile_sm120_source()?))
            .map_err(|error| format!("load next tournament module: {error:?}"))?;
        let production = module
            .load_function(PRODUCTION_SYMBOL)
            .map_err(|error| format!("load production reference: {error:?}"))?;
        let production_spec = production_spec();
        validate_production_spec(production_spec)?;
        production
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                production_spec.dynamic_shared_bytes as i32,
            )
            .map_err(|error| format!("set production dynamic shared: {error:?}"))?;
        let mut candidates = Vec::with_capacity(CANDIDATES.len());
        for spec in candidate_specs() {
            let function = module
                .load_function(spec.symbol)
                .map_err(|error| format!("load {}: {error:?}", spec.symbol))?;
            function
                .set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    spec.dynamic_shared_bytes as i32,
                )
                .map_err(|error| format!("set {} dynamic shared: {error:?}", spec.symbol))?;
            candidates.push((*spec, function));
        }
        Ok(ResourceModule {
            _module: module,
            production,
            candidates,
        })
    }

    fn validate_driver_resources(device: &GpuDevice) -> Result<(), String> {
        let resources = load_resource_module(device)?;
        validate_function_resources(&resources.production, production_spec())?;
        for (spec, function) in &resources.candidates {
            validate_function_resources(function, *spec)?;
        }
        Ok(())
    }

    fn validate_function_resources(
        function: &CudaFunction,
        spec: CandidateSpec,
    ) -> Result<(), String> {
        let registers = function
            .num_regs()
            .map_err(|error| format!("regs: {error:?}"))?;
        let local = function
            .local_size_bytes()
            .map_err(|error| format!("local: {error:?}"))?;
        let static_shared = function
            .shared_size_bytes()
            .map_err(|error| format!("smem: {error:?}"))?;
        let max_threads = function
            .max_threads_per_block()
            .map_err(|error| format!("threads: {error:?}"))?;
        let max_dynamic = function
            .get_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            )
            .map_err(|error| format!("dynamic shared: {error:?}"))?;
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(
                spec.block.0,
                spec.dynamic_shared_bytes as usize,
                None,
            )
            .map_err(|error| format!("occupancy: {error:?}"))?;
        if registers <= 0
            || registers > spec.maximum_registers as i32
            || local != 0
            || static_shared != 0
            || max_threads < spec.block.0 as i32
            || max_dynamic < spec.dynamic_shared_bytes as i32
            || occupancy < spec.minimum_occupancy
        {
            return Err(format!(
                "{} driver budget failed regs={registers}/{} local={local}/0 static={static_shared}/0 dynamic={}/{max_dynamic} threads={}/{} occupancy={}/{}",
                spec.symbol,
                spec.maximum_registers,
                spec.dynamic_shared_bytes,
                spec.block.0,
                max_threads,
                occupancy,
                spec.minimum_occupancy
            ));
        }
        eprintln!(
            "tf32_nn_rect_wide_next resource symbol={} regs={} local={} static_shared={} dynamic_shared={} threads={} occupancy={}",
            spec.symbol,
            registers,
            local,
            static_shared,
            spec.dynamic_shared_bytes,
            spec.block.0,
            occupancy
        );
        Ok(())
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct DirectParams {
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

    unsafe impl DeviceRepr for DirectParams {}

    const _: () = {
        assert!(std::mem::size_of::<DirectParams>() == 40);
        assert!(std::mem::align_of::<DirectParams>() == 4);
    };

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    struct DirectTensorMap(sys::CUtensorMap);

    unsafe impl DeviceRepr for DirectTensorMap {}

    #[derive(Clone, Copy)]
    struct DirectMaps {
        a: DirectTensorMap,
        b: DirectTensorMap,
    }

    struct GuardedBuffer {
        buffer: GpuBuffer,
        initial: Vec<f32>,
    }

    impl GuardedBuffer {
        fn new(stream: &Arc<CudaStream>, initial: Vec<f32>) -> Result<Self, String> {
            if initial.is_empty() {
                return Err("guarded GPU buffer cannot be empty".into());
            }
            let mut contents = vec![f32::from_bits(GUARD_BITS); initial.len() + 2 * GUARD_ELEMENTS];
            contents[GUARD_ELEMENTS..GUARD_ELEMENTS + initial.len()].copy_from_slice(&initial);
            Ok(Self {
                buffer: GpuBuffer::from_cpu(stream, &contents)?,
                initial,
            })
        }

        fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
            let mut contents =
                vec![f32::from_bits(GUARD_BITS); self.initial.len() + 2 * GUARD_ELEMENTS];
            contents[GUARD_ELEMENTS..GUARD_ELEMENTS + self.initial.len()]
                .copy_from_slice(&self.initial);
            self.buffer.upload(stream, &contents)
        }

        fn active_ptr(&self, stream: &Arc<CudaStream>, label: &str) -> Result<u64, String> {
            let pointer = self.buffer.raw_ptr_at(stream, GUARD_ELEMENTS);
            if pointer == 0 || !pointer.is_multiple_of(128) {
                return Err(format!(
                    "{label} pointer is not nonzero 128-byte aligned: {pointer:#x}"
                ));
            }
            Ok(pointer)
        }

        fn bits(&self, stream: &Arc<CudaStream>) -> Result<Vec<u32>, String> {
            guarded_active_bits(&self.buffer.to_cpu(stream)?, self.initial.len(), None)
        }

        fn validate_read_only(&self, stream: &Arc<CudaStream>) -> Result<(), String> {
            guarded_active_bits(
                &self.buffer.to_cpu(stream)?,
                self.initial.len(),
                Some(&self.initial),
            )?;
            Ok(())
        }
    }

    struct Fixture {
        a: GuardedBuffer,
        b: GuardedBuffer,
        production_output: GuardedBuffer,
        candidate_output: GuardedBuffer,
        production_maps: DirectMaps,
        candidate_maps: DirectMaps,
        params: DirectParams,
        production_config: LaunchConfig,
        candidate_config: LaunchConfig,
    }

    struct Runtime {
        stream: Arc<CudaStream>,
        module: ResourceModule,
    }

    fn direct_tensor_map(
        pointer: u64,
        width: usize,
        rows: usize,
        stride: usize,
        box_dimensions: [u32; 2],
        label: &str,
    ) -> Result<DirectTensorMap, String> {
        if pointer == 0 || !pointer.is_multiple_of(128) || width == 0 || rows == 0 || stride < width
        {
            return Err(format!("invalid {label} tensor-map input"));
        }
        let byte_stride = stride
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("{label} byte stride overflow"))?;
        if !byte_stride.is_multiple_of(16) {
            return Err(format!("{label} byte stride is not 16-byte aligned"));
        }
        let dimensions = [width as u64, rows as u64];
        let global_strides = [byte_stride as u64];
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
            return Err(format!("encode {label} tensor map: {result:?}"));
        }
        Ok(DirectTensorMap(unsafe { raw.assume_init() }))
    }

    fn new_runtime(device: &GpuDevice, stream: Arc<CudaStream>) -> Result<Runtime, String> {
        Ok(Runtime {
            stream,
            module: load_resource_module(device)?,
        })
    }

    fn new_fixture(
        runtime: &Runtime,
        spec: CandidateSpec,
        dims: (usize, usize, usize),
        strides: (usize, usize, usize),
        a_values: Vec<f32>,
        b_values: Vec<f32>,
        output_values: Vec<f32>,
    ) -> Result<Fixture, String> {
        let (m, k, n) = dims;
        let (lda, ldb, ldc) = strides;
        if lda < k
            || ldb < n
            || ldc < n
            || a_values.len() != m.checked_mul(lda).ok_or("A extent overflow")?
            || b_values.len() != k.checked_mul(ldb).ok_or("B extent overflow")?
            || output_values.len() != m.checked_mul(ldc).ok_or("C extent overflow")?
        {
            return Err("fixture geometry or allocation length mismatch".into());
        }
        let a = GuardedBuffer::new(&runtime.stream, a_values)?;
        let b = GuardedBuffer::new(&runtime.stream, b_values)?;
        let production_output = GuardedBuffer::new(&runtime.stream, output_values.clone())?;
        let candidate_output = GuardedBuffer::new(&runtime.stream, output_values)?;
        let a_pointer = a.active_ptr(&runtime.stream, "A")?;
        let b_pointer = b.active_ptr(&runtime.stream, "B")?;
        production_output.active_ptr(&runtime.stream, "production C")?;
        candidate_output.active_ptr(&runtime.stream, "candidate C")?;
        let production_spec = production_spec();
        validate_production_spec(production_spec)?;
        let production_keys = map_keys(production_spec);
        let production_maps = DirectMaps {
            a: direct_tensor_map(
                a_pointer,
                k,
                m,
                lda,
                production_keys[0].box_dimensions,
                "production A",
            )?,
            b: direct_tensor_map(
                b_pointer,
                n,
                k,
                ldb,
                production_keys[1].box_dimensions,
                "production B",
            )?,
        };
        let candidate_maps = DirectMaps {
            a: direct_tensor_map(
                a_pointer,
                k,
                m,
                lda,
                [spec.map_bk, spec.tile.0],
                "candidate A",
            )?,
            b: direct_tensor_map(
                b_pointer,
                n,
                k,
                ldb,
                [spec.map_bk, spec.map_bk],
                "candidate B",
            )?,
        };
        let candidate_grid = m
            .div_ceil(spec.tile.0 as usize)
            .checked_mul(n.div_ceil(spec.tile.1 as usize))
            .ok_or("candidate grid overflow")?;
        let (production_grid, production_block, production_shared) =
            launch_geometry(production_spec, dims)?;
        Ok(Fixture {
            a,
            b,
            production_output,
            candidate_output,
            production_maps,
            candidate_maps,
            params: DirectParams {
                a_x: 0,
                a_y: 0,
                b_x: 0,
                b_y: 0,
                alpha: 1.0,
                beta: 0.0,
                m: m.try_into().map_err(|_| "M exceeds i32")?,
                k: k.try_into().map_err(|_| "K exceeds i32")?,
                n: n.try_into().map_err(|_| "N exceeds i32")?,
                ldc: ldc.try_into().map_err(|_| "ldc exceeds i32")?,
            },
            production_config: LaunchConfig {
                grid_dim: production_grid,
                block_dim: production_block,
                shared_mem_bytes: production_shared,
            },
            candidate_config: LaunchConfig {
                grid_dim: (
                    candidate_grid
                        .try_into()
                        .map_err(|_| "candidate grid exceeds u32")?,
                    1,
                    1,
                ),
                block_dim: spec.block,
                shared_mem_bytes: spec.dynamic_shared_bytes,
            },
        })
    }

    fn launch_production(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        let output = fixture
            .production_output
            .active_ptr(&runtime.stream, "production C")?;
        let bias = 0_u64;
        let mut builder = runtime.stream.launch_builder(&runtime.module.production);
        builder.arg(&output);
        builder.arg(&fixture.production_maps.a);
        builder.arg(&fixture.production_maps.b);
        builder.arg(&bias);
        builder.arg(&fixture.params);
        unsafe { builder.launch(fixture.production_config) }
            .map(|_| ())
            .map_err(|error| format!("launch production reference: {error:?}"))
    }

    fn launch_candidate(
        runtime: &Runtime,
        spec: CandidateSpec,
        fixture: &Fixture,
    ) -> Result<(), String> {
        let output = fixture
            .candidate_output
            .active_ptr(&runtime.stream, "candidate C")?;
        let bias = 0_u64;
        let mut builder = runtime
            .stream
            .launch_builder(runtime.module.candidate(spec)?);
        builder.arg(&output);
        builder.arg(&fixture.candidate_maps.a);
        builder.arg(&fixture.candidate_maps.b);
        builder.arg(&bias);
        builder.arg(&fixture.params);
        unsafe { builder.launch(fixture.candidate_config) }
            .map(|_| ())
            .map_err(|error| format!("launch {}: {error:?}", spec.symbol))
    }

    fn capture_production(runtime: &Runtime, fixture: &Fixture) -> Result<CudaGraph, String> {
        unsafe { capture_into_graph(&runtime.stream, || launch_production(runtime, fixture)) }
    }

    fn capture_candidate(
        runtime: &Runtime,
        spec: CandidateSpec,
        fixture: &Fixture,
    ) -> Result<CudaGraph, String> {
        unsafe { capture_into_graph(&runtime.stream, || launch_candidate(runtime, spec, fixture)) }
    }

    fn synchronize(runtime: &Runtime, label: &str) -> Result<(), String> {
        runtime
            .stream
            .synchronize()
            .map_err(|error| format!("{label} synchronize: {error:?}"))
    }

    fn validate_outputs(
        runtime: &Runtime,
        fixture: &Fixture,
        label: &str,
    ) -> Result<Vec<u32>, String> {
        let production = fixture.production_output.bits(&runtime.stream)?;
        let candidate = fixture.candidate_output.bits(&runtime.stream)?;
        if production != candidate {
            let index = production
                .iter()
                .zip(&candidate)
                .position(|(left, right)| left != right)
                .unwrap_or(0);
            return Err(format!(
                "{label} differs from production at {index}: {:08x} != {:08x}",
                production[index], candidate[index]
            ));
        }
        fixture.a.validate_read_only(&runtime.stream)?;
        fixture.b.validate_read_only(&runtime.stream)?;
        Ok(candidate)
    }

    fn reset_outputs(runtime: &Runtime, fixture: &mut Fixture) -> Result<(), String> {
        fixture.production_output.reset(&runtime.stream)?;
        fixture.candidate_output.reset(&runtime.stream)
    }

    fn seeded_values(length: usize, salt: u64) -> Vec<f32> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ salt;
        (0..length)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let signed = ((state.wrapping_add(index as u64) % 4093) as i32) - 2046;
                signed as f32 / 1024.0
            })
            .collect()
    }

    fn exceptional_values(length: usize, salt: u64) -> Vec<f32> {
        let mut values = seeded_values(length, salt);
        for (index, bits) in [
            (0, 0x8000_0000),
            (1, 0x7f80_0000),
            (2, 0xff80_0000),
            (31, 0x7fc1_2345),
            (97, 0xffc2_3456),
        ] {
            if index < values.len() {
                values[index] = f32::from_bits(bits);
            }
        }
        values
    }

    fn bytes_of<T>(value: &T) -> &[u8] {
        unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
    }

    fn digest_arguments(arguments: &[&[u8]]) -> [u8; 32] {
        let mut digest = FramedSha256::new(b"tf32-next-physical-arguments.v2");
        for argument in arguments {
            digest = digest.required(b"argument", argument);
        }
        digest.finish()
    }

    fn physical_identity_from_parts(
        symbol: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        dynamic_shared_bytes: u32,
        arguments_digest: [u8; 32],
    ) -> PhysicalIdentity {
        let launch_digest = FramedSha256::new(b"tf32-next-launch.v2")
            .required(b"symbol", symbol.as_bytes())
            .required(b"grid-x", &grid.0.to_le_bytes())
            .required(b"grid-y", &grid.1.to_le_bytes())
            .required(b"grid-z", &grid.2.to_le_bytes())
            .required(b"block-x", &block.0.to_le_bytes())
            .required(b"block-y", &block.1.to_le_bytes())
            .required(b"block-z", &block.2.to_le_bytes())
            .required(b"shared", &dynamic_shared_bytes.to_le_bytes())
            .required(b"arguments", &arguments_digest)
            .finish();
        let graph_digest = FramedSha256::new(b"tf32-next-single-node-graph.v2")
            .required(b"launch", &launch_digest)
            .required(b"ordered-node-count", &1_u64.to_le_bytes())
            .finish();
        PhysicalIdentity {
            symbol: symbol.to_owned(),
            grid,
            block,
            dynamic_shared_bytes,
            arguments_digest,
            launch_digest,
            graph_digest,
        }
    }

    fn eager_candidate_identity(
        runtime: &Runtime,
        spec: CandidateSpec,
        fixture: &Fixture,
    ) -> Result<PhysicalIdentity, String> {
        let output = fixture
            .candidate_output
            .active_ptr(&runtime.stream, "identity C")?;
        let bias = 0_u64;
        let arguments_digest = digest_arguments(&[
            bytes_of(&output),
            bytes_of(&fixture.candidate_maps.a),
            bytes_of(&fixture.candidate_maps.b),
            bytes_of(&bias),
            bytes_of(&fixture.params),
        ]);
        Ok(physical_identity_from_parts(
            spec.symbol,
            fixture.candidate_config.grid_dim,
            fixture.candidate_config.block_dim,
            fixture.candidate_config.shared_mem_bytes,
            arguments_digest,
        ))
    }

    fn cuda_ok(result: sys::CUresult, label: &str) -> Result<(), String> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{label}: {result:?}"))
        }
    }

    fn graph_candidate_identity(graph: &CudaGraph) -> Result<PhysicalIdentity, String> {
        let raw = graph.cu_graph();
        let mut node_count = 0_usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, std::ptr::null_mut(), &mut node_count) },
            "query candidate graph node count",
        )?;
        if node_count != 1 {
            return Err(format!(
                "candidate graph has {node_count} nodes instead of one"
            ));
        }
        let mut edge_count = 0_usize;
        cuda_ok(
            unsafe {
                sys::cuGraphGetEdges_v2(
                    raw,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut edge_count,
                )
            },
            "query candidate graph edge count",
        )?;
        if edge_count != 0 {
            return Err(format!(
                "single-node candidate graph has {edge_count} unexpected edges"
            ));
        }
        let mut node = std::ptr::null_mut();
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, &mut node, &mut node_count) },
            "query candidate graph node",
        )?;
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        cuda_ok(
            unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
            "query candidate graph node type",
        )?;
        if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            return Err(format!(
                "candidate graph contains a non-kernel node: {kind:?}"
            ));
        }
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            "query candidate graph kernel parameters",
        )?;
        let mut name = std::ptr::null();
        cuda_ok(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "query candidate graph function name",
        )?;
        if name.is_null() || params.kernelParams.is_null() {
            return Err("candidate graph omitted its function name or packed arguments".into());
        }
        let symbol = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|error| format!("candidate graph function name is not UTF-8: {error}"))?;
        let argument_sizes = [
            size_of::<u64>(),
            size_of::<DirectTensorMap>(),
            size_of::<DirectTensorMap>(),
            size_of::<u64>(),
            size_of::<DirectParams>(),
        ];
        let mut arguments = Vec::with_capacity(argument_sizes.len());
        for (index, size) in argument_sizes.into_iter().enumerate() {
            let pointer = unsafe { *params.kernelParams.add(index) };
            if pointer.is_null() {
                return Err(format!("candidate graph argument {index} is null"));
            }
            arguments.push(unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), size) });
        }
        Ok(physical_identity_from_parts(
            symbol,
            (params.gridDimX, params.gridDimY, params.gridDimZ),
            (params.blockDimX, params.blockDimY, params.blockDimZ),
            params.sharedMemBytes,
            digest_arguments(&arguments),
        ))
    }

    fn run_small_oracle(runtime: &Runtime, spec: CandidateSpec) -> Result<(), String> {
        let a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let b = vec![
            5.0, 7.0, 0.0, 0.0, 11.0, 13.0, 0.0, 0.0, 17.0, 19.0, 0.0, 0.0, 23.0, 29.0, 0.0, 0.0,
        ];
        let expected = [
            5.0_f32, 7.0, 0.0, 0.0, 22.0, 26.0, 0.0, 0.0, 51.0, 57.0, 0.0, 0.0,
        ]
        .map(f32::to_bits);
        let mut fixture = new_fixture(runtime, spec, (3, 4, 2), (4, 4, 4), a, b, vec![0.0; 12])?;
        for repeat in 0..3 {
            reset_outputs(runtime, &mut fixture)?;
            launch_production(runtime, &fixture)?;
            launch_candidate(runtime, spec, &fixture)?;
            synchronize(runtime, "small eager")?;
            let actual = validate_outputs(runtime, &fixture, "small eager")?;
            if actual != expected {
                return Err(format!(
                    "{} small eager repeat {repeat} failed: {actual:?}",
                    spec.symbol
                ));
            }
        }
        let production_graph = capture_production(runtime, &fixture)?;
        let candidate_graph = capture_candidate(runtime, spec, &fixture)?;
        for repeat in 0..3 {
            reset_outputs(runtime, &mut fixture)?;
            production_graph
                .launch()
                .map_err(|error| format!("small production graph: {error:?}"))?;
            candidate_graph
                .launch()
                .map_err(|error| format!("small candidate graph: {error:?}"))?;
            synchronize(runtime, "small graph")?;
            let actual = validate_outputs(runtime, &fixture, "small graph")?;
            if actual != expected {
                return Err(format!(
                    "{} small graph repeat {repeat} failed: {actual:?}",
                    spec.symbol
                ));
            }
        }
        Ok(())
    }

    fn run_exceptional_oracle(runtime: &Runtime, spec: CandidateSpec) -> Result<(), String> {
        const STRIDE: usize = 4;
        let cases = exceptional_single_term_cases();
        let mut a = vec![0.0_f32; cases.len() * STRIDE];
        let mut expected = vec![0_u32; cases.len() * STRIDE];
        for (row, &(_, input_bits, expected_bits)) in cases.iter().enumerate() {
            a[row * STRIDE] = f32::from_bits(input_bits);
            expected[row * STRIDE] = expected_bits;
        }
        let mut b = vec![0.0_f32; STRIDE];
        b[0] = 1.0;
        let mut fixture = new_fixture(
            runtime,
            spec,
            (cases.len(), 1, 1),
            (STRIDE, STRIDE, STRIDE),
            a,
            b,
            vec![0.0; cases.len() * STRIDE],
        )?;
        for repeat in 0..3 {
            reset_outputs(runtime, &mut fixture)?;
            launch_production(runtime, &fixture)?;
            launch_candidate(runtime, spec, &fixture)?;
            synchronize(runtime, "exceptional eager")?;
            let actual = validate_outputs(runtime, &fixture, "exceptional eager")?;
            if actual != expected {
                return Err(format!(
                    "{} exceptional eager repeat {repeat} failed: actual={actual:08x?} expected={expected:08x?}",
                    spec.symbol
                ));
            }
        }
        let production_graph = capture_production(runtime, &fixture)?;
        let candidate_graph = capture_candidate(runtime, spec, &fixture)?;
        for repeat in 0..3 {
            reset_outputs(runtime, &mut fixture)?;
            production_graph
                .launch()
                .map_err(|error| format!("exceptional production graph: {error:?}"))?;
            candidate_graph
                .launch()
                .map_err(|error| format!("exceptional candidate graph: {error:?}"))?;
            synchronize(runtime, "exceptional graph")?;
            let actual = validate_outputs(runtime, &fixture, "exceptional graph")?;
            if actual != expected {
                return Err(format!(
                    "{} exceptional graph repeat {repeat} failed: actual={actual:08x?} expected={expected:08x?}",
                    spec.symbol
                ));
            }
        }
        Ok(())
    }

    fn run_large_exact(runtime: &Runtime, spec: CandidateSpec) -> Result<PhysicalIdentity, String> {
        let (m, k, n) = DIMS;
        let mut fixture = new_fixture(
            runtime,
            spec,
            DIMS,
            (k, n, n),
            exceptional_values(m * k, CORPUS_SALT ^ 0xa1),
            exceptional_values(k * n, CORPUS_SALT ^ 0xb2),
            seeded_values(m * n, CORPUS_SALT ^ 0xc3),
        )?;
        let eager_identity = eager_candidate_identity(runtime, spec, &fixture)?;
        validate_identity(spec, &eager_identity)?;
        let mut eager_bits = None;
        for repeat in 0..3 {
            reset_outputs(runtime, &mut fixture)?;
            launch_production(runtime, &fixture)?;
            launch_candidate(runtime, spec, &fixture)?;
            synchronize(runtime, "large eager")?;
            let actual = validate_outputs(runtime, &fixture, "large eager")?;
            if eager_bits
                .as_ref()
                .is_some_and(|expected| expected != &actual)
            {
                return Err(format!("{} eager repeat {repeat} changed", spec.symbol));
            }
            eager_bits = Some(actual);
        }
        let production_graph = capture_production(runtime, &fixture)?;
        let candidate_graph = capture_candidate(runtime, spec, &fixture)?;
        let graph_identity = graph_candidate_identity(&candidate_graph)?;
        validate_eager_graph_identity(spec, &eager_identity, &graph_identity)?;
        for repeat in 0..3 {
            reset_outputs(runtime, &mut fixture)?;
            production_graph
                .launch()
                .map_err(|error| format!("large production graph: {error:?}"))?;
            candidate_graph
                .launch()
                .map_err(|error| format!("large candidate graph: {error:?}"))?;
            synchronize(runtime, "large graph")?;
            let actual = validate_outputs(runtime, &fixture, "large graph")?;
            if eager_bits
                .as_ref()
                .is_none_or(|expected| expected != &actual)
            {
                return Err(format!("{} graph repeat {repeat} changed", spec.symbol));
            }
        }
        Ok(graph_identity)
    }

    #[test]
    #[ignore = "requires exclusive CC12.0/170-SM CUDA13.2 hardware"]
    fn next_candidates_are_guarded_exact_and_graph_stable() -> Result<(), String> {
        let device = GpuDevice::new(0)?;
        let ctx = configure(&device)?;
        validate_exact_environment(&ctx)?;
        validate_production_witness(&ctx)?;
        validate_ptxas_resources_and_sass()?;
        validate_driver_resources(&device)?;
        let runtime = new_runtime(&device, ctx.stream.clone())?;
        let mut identities = Vec::with_capacity(CANDIDATES.len());
        for spec in candidate_specs() {
            run_small_oracle(&runtime, *spec)?;
            run_exceptional_oracle(&runtime, *spec)?;
            identities.push(run_large_exact(&runtime, *spec)?);
        }
        validate_cross_candidate_identity_uniqueness(&identities)
    }

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
    enum Arm {
        Production,
        Candidate(CandidateSpec),
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum PairGate {
        Parity,
        WinnerSpeedup,
    }

    impl Arm {
        const fn name(self) -> &'static str {
            match self {
                Self::Production => "production-tag33",
                Self::Candidate(spec) => spec.symbol,
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

    struct TimedArm<'a> {
        runtime: &'a Runtime,
        fixture: Fixture,
        arm: Arm,
        graph: CudaGraph,
    }

    fn timing_fixture(runtime: &Runtime, spec: CandidateSpec) -> Result<Fixture, String> {
        let (m, k, n) = DIMS;
        new_fixture(
            runtime,
            spec,
            DIMS,
            (k, n, n),
            seeded_values(m * k, CORPUS_SALT ^ 0x11),
            seeded_values(k * n, CORPUS_SALT ^ 0x22),
            vec![0.0; m * n],
        )
    }

    fn new_timed_arm<'a>(
        runtime: &'a Runtime,
        arm: Arm,
        map_spec: CandidateSpec,
    ) -> Result<TimedArm<'a>, String> {
        let effective_map_spec = match arm {
            Arm::Production => map_spec,
            Arm::Candidate(spec) => spec,
        };
        let fixture = timing_fixture(runtime, effective_map_spec)?;
        let graph = match arm {
            Arm::Production => capture_production(runtime, &fixture)?,
            Arm::Candidate(spec) => capture_candidate(runtime, spec, &fixture)?,
        };
        Ok(TimedArm {
            runtime,
            fixture,
            arm,
            graph,
        })
    }

    fn launch_timed(arm: &TimedArm<'_>, path: Path) -> Result<(), String> {
        match path {
            Path::Eager => match arm.arm {
                Arm::Production => launch_production(arm.runtime, &arm.fixture),
                Arm::Candidate(spec) => launch_candidate(arm.runtime, spec, &arm.fixture),
            },
            Path::Graph => arm
                .graph
                .launch()
                .map_err(|error| format!("launch {} graph: {error:?}", arm.arm.name())),
        }
    }

    fn measure(arm: &TimedArm<'_>, path: Path, iterations: usize) -> Result<f64, String> {
        if iterations == 0 {
            return Err("timing iterations must be positive".into());
        }
        let start = arm
            .runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("{} start event: {error:?}", arm.arm.name()))?;
        for _ in 0..iterations {
            launch_timed(arm, path)?;
        }
        let end = arm
            .runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("{} end event: {error:?}", arm.arm.name()))?;
        let microseconds = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("{} elapsed event: {error:?}", arm.arm.name()))?,
        ) * 1_000.0
            / iterations as f64;
        if !microseconds.is_finite() || microseconds <= 0.0 {
            return Err(format!("{} invalid timing {microseconds}", arm.arm.name()));
        }
        Ok(microseconds)
    }

    fn paired(
        baseline: &TimedArm<'_>,
        candidate: &TimedArm<'_>,
        path: Path,
        order: Order,
        windows: usize,
    ) -> Result<TimingStats, String> {
        if !matches!(windows, SCREENING_WINDOWS | OFFICIAL_WINDOWS) {
            return Err(format!("paired timing rejects window count {windows}"));
        }
        for _ in 0..128 {
            launch_timed(baseline, path)?;
            launch_timed(candidate, path)?;
        }
        synchronize(baseline.runtime, "baseline warmup")?;
        synchronize(candidate.runtime, "candidate warmup")?;
        let pilot_baseline = measure(baseline, path, 3)?;
        let pilot_candidate = measure(candidate, path, 3)?;
        let pilot = pilot_baseline.max(pilot_candidate);
        let iterations =
            ((TARGET_WINDOW_MS * 1_000.0 / pilot).round() as usize).clamp(3, MAX_WINDOW_ITERATIONS);
        let mut baseline_samples = Vec::with_capacity(windows);
        let mut candidate_samples = Vec::with_capacity(windows);
        let mut speedups = Vec::with_capacity(windows);
        for _ in 0..windows {
            let (b0, b1, c0, c1) = match order {
                Order::Abba => {
                    let b0 = measure(baseline, path, iterations)?;
                    let c0 = measure(candidate, path, iterations)?;
                    let c1 = measure(candidate, path, iterations)?;
                    let b1 = measure(baseline, path, iterations)?;
                    (b0, b1, c0, c1)
                }
                Order::Baab => {
                    let c0 = measure(candidate, path, iterations)?;
                    let b0 = measure(baseline, path, iterations)?;
                    let b1 = measure(baseline, path, iterations)?;
                    let c1 = measure(candidate, path, iterations)?;
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
                return Err("paired timing produced an invalid sample".into());
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

    fn print_stats(
        baseline: Arm,
        candidate: Arm,
        path: Path,
        order: Order,
        windows: usize,
        stats: TimingStats,
    ) {
        eprintln!(
            "tf32_nn_rect_wide_next baseline={} candidate={} path={} order={} windows={} baseline_us_p50={:.6} candidate_us_p50={:.6} speedup_p05={:.9} speedup_p50={:.9} speedup_p95={:.9}",
            baseline.name(),
            candidate.name(),
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

    #[derive(Clone, Copy, Debug)]
    struct ScreenResult {
        spec: CandidateSpec,
        worst_p05: f64,
        worst_p50: f64,
    }

    fn screen_candidate(
        production_runtime: &Runtime,
        candidate_runtime: &Runtime,
        spec: CandidateSpec,
    ) -> Result<ScreenResult, String> {
        let production = new_timed_arm(production_runtime, Arm::Production, spec)?;
        let candidate = new_timed_arm(candidate_runtime, Arm::Candidate(spec), spec)?;
        let mut worst_p05 = f64::INFINITY;
        let mut worst_p50 = f64::INFINITY;
        for path in [Path::Eager, Path::Graph] {
            for order in [Order::Abba, Order::Baab] {
                let stats = paired(&production, &candidate, path, order, SCREENING_WINDOWS)?;
                print_stats(
                    Arm::Production,
                    Arm::Candidate(spec),
                    path,
                    order,
                    SCREENING_WINDOWS,
                    stats,
                );
                worst_p05 = worst_p05.min(stats.speedup_p05);
                worst_p50 = worst_p50.min(stats.speedup_p50);
            }
        }
        Ok(ScreenResult {
            spec,
            worst_p05,
            worst_p50,
        })
    }

    fn official_pair(
        baseline_runtime: &Runtime,
        candidate_runtime: &Runtime,
        baseline_arm: Arm,
        candidate_arm: Arm,
        map_spec: CandidateSpec,
        gate: PairGate,
        quiet: &QuietGpu,
    ) -> Result<(), String> {
        quiet.require_cohort("tf32-nn-rect-wide-next-official")?;
        let baseline = new_timed_arm(baseline_runtime, baseline_arm, map_spec)?;
        let candidate = new_timed_arm(candidate_runtime, candidate_arm, map_spec)?;
        for path in [Path::Eager, Path::Graph] {
            for order in [Order::Abba, Order::Baab] {
                let stats = paired(&baseline, &candidate, path, order, OFFICIAL_WINDOWS)?;
                print_stats(
                    baseline_arm,
                    candidate_arm,
                    path,
                    order,
                    OFFICIAL_WINDOWS,
                    stats,
                );
                match gate {
                    PairGate::Parity => {
                        validate_parity_gate(stats.speedup_p05, stats.speedup_p50)?;
                    }
                    PairGate::WinnerSpeedup
                        if stats.speedup_p05 < WINNER_MIN_P05
                            || stats.speedup_p50 < WINNER_MIN_P50 =>
                    {
                        return Err(format!(
                            "winner is unstable against runner: path={path:?} order={order:?} stats={stats:?}"
                        ));
                    }
                    PairGate::WinnerSpeedup => {}
                }
            }
        }
        quiet.verify_post_cohort("tf32-nn-rect-wide-next-official")?;
        Ok(())
    }

    struct CublasBuffers {
        output: GpuBuffer,
        a: GpuBuffer,
        b: GpuBuffer,
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
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("cuBLAS start event: {error:?}"))?;
        for _ in 0..iterations {
            launch_cublas_fast_nn(ctx, buffers, DIMS)?;
        }
        let end = ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("cuBLAS end event: {error:?}"))?;
        let microseconds = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("cuBLAS elapsed event: {error:?}"))?,
        ) * 1_000.0
            / iterations as f64;
        if !microseconds.is_finite() || microseconds <= 0.0 {
            return Err(format!("invalid cuBLAS-fast timing sample {microseconds}"));
        }
        Ok(microseconds)
    }

    fn report_informational_cublas_fast(ctx: &GpuCtx, quiet: &QuietGpu) -> Result<(), String> {
        quiet.require_cohort("tf32-nn-rect-wide-next-cublas-fast")?;
        validate_informational_cublas_orientation(ctx)?;
        let (m, k, n) = DIMS;
        let buffers = CublasBuffers {
            output: GpuBuffer::from_cpu(&ctx.stream, &vec![0.0; m * n])?,
            a: GpuBuffer::from_cpu(&ctx.stream, &seeded_values(m * k, CORPUS_SALT ^ 0x91))?,
            b: GpuBuffer::from_cpu(&ctx.stream, &seeded_values(k * n, CORPUS_SALT ^ 0x2d))?,
        };
        let pilot = measure_cublas_fast(ctx, &buffers, 3)?;
        let iterations =
            ((TARGET_WINDOW_MS * 1_000.0 / pilot).round() as usize).clamp(3, MAX_WINDOW_ITERATIONS);
        let mut samples = Vec::with_capacity(OFFICIAL_WINDOWS);
        for _ in 0..OFFICIAL_WINDOWS {
            samples.push(measure_cublas_fast(ctx, &buffers, iterations)?);
        }
        eprintln!(
            "tf32_nn_rect_wide_next arm=cublas_fast informational_only=true promotion_gate=false windows={} p05_us={:.6} p50_us={:.6} p95_us={:.6}",
            OFFICIAL_WINDOWS,
            percentile(&samples, 0.05)?,
            percentile(&samples, 0.50)?,
            percentile(&samples, 0.95)?,
        );
        quiet.verify_post_cohort("tf32-nn-rect-wide-next-cublas-fast")?;
        Ok(())
    }

    fn validate_release_build() -> Result<(), String> {
        if cfg!(debug_assertions) {
            Err("official next tournament requires --release".into())
        } else {
            Ok(())
        }
    }

    #[test]
    #[ignore = "requires exclusive quiet CC12.0/170-SM CUDA13.2 hardware"]
    fn next_tournament_screen21_and_official101_abba_baab() -> Result<(), String> {
        validate_release_build()?;
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("tf32-nn-rect-wide-next-pre-context")?;
        let device = GpuDevice::new(0)?;
        let production_ctx = configure(&device)?;
        let candidate_ctx = configure(&device)?;
        validate_exact_environment(&production_ctx)?;
        validate_exact_environment(&candidate_ctx)?;
        validate_production_witness(&production_ctx)?;
        validate_ptxas_resources_and_sass()?;
        validate_driver_resources(&device)?;
        let correctness_runtime = new_runtime(&device, production_ctx.stream.clone())?;
        for spec in candidate_specs() {
            run_small_oracle(&correctness_runtime, *spec)?;
            run_exceptional_oracle(&correctness_runtime, *spec)?;
            run_large_exact(&correctness_runtime, *spec)?;
        }
        let production_runtime = new_runtime(&device, production_ctx.stream.clone())?;
        let candidate_runtime = new_runtime(&device, candidate_ctx.stream.clone())?;
        let mut ranking = Vec::with_capacity(CANDIDATES.len());
        for spec in candidate_specs() {
            quiet.require_cohort(spec.symbol)?;
            ranking.push(screen_candidate(
                &production_runtime,
                &candidate_runtime,
                *spec,
            )?);
            quiet.verify_post_cohort(spec.symbol)?;
        }
        ranking.sort_by(|left, right| right.worst_p50.total_cmp(&left.worst_p50));
        let [winner, runner, ..] = ranking.as_slice() else {
            return Err("next tournament requires two ranked candidates".into());
        };
        if winner.spec != CANDIDATES[3] {
            return Err(format!(
                "post-promotion production identity lost the verified winner: {winner:?}"
            ));
        }
        validate_parity_gate(winner.worst_p05, winner.worst_p50)?;
        eprintln!(
            "tf32_nn_rect_wide_next screening_winner={} worst_p05={:.9} worst_p50={:.9} runner={}",
            winner.spec.symbol, winner.worst_p05, winner.worst_p50, runner.spec.symbol
        );
        official_pair(
            &production_runtime,
            &candidate_runtime,
            Arm::Production,
            Arm::Candidate(winner.spec),
            winner.spec,
            PairGate::Parity,
            &quiet,
        )?;
        official_pair(
            &production_runtime,
            &candidate_runtime,
            Arm::Candidate(runner.spec),
            Arm::Candidate(winner.spec),
            winner.spec,
            PairGate::WinnerSpeedup,
            &quiet,
        )?;
        if let Err(error) = report_informational_cublas_fast(&production_ctx, &quiet) {
            eprintln!(
                "tf32_nn_rect_wide_next arm=cublas_fast informational_only=true promotion_gate=false unavailable={error}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA13.2 NVRTC, ptxas, and nvdisasm; launches no GPU work"]
    fn next_candidates_have_strict_ptxas_and_sass_resources() -> Result<(), String> {
        validate_ptxas_resources_and_sass()
    }

    #[test]
    #[ignore = "requires exclusive CC12.0/170-SM CUDA13.2 hardware"]
    fn next_candidates_have_strict_driver_resources() -> Result<(), String> {
        let device = GpuDevice::new(0)?;
        let ctx = configure(&device)?;
        validate_exact_environment(&ctx)?;
        validate_driver_resources(&device)
    }
}
