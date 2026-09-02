use std::ffi::OsStr;

#[cfg(feature = "cuda")]
mod common;

const SCREEN_WINDOWS: usize = 21;
const OFFICIAL_WINDOWS: usize = 101;
#[cfg(feature = "cuda")]
const MIN_PRODUCTION_P05: f64 = 1.005;
#[cfg(feature = "cuda")]
const MIN_PRODUCTION_P50: f64 = 1.01;
#[cfg(feature = "cuda")]
const MIN_RUNNER_P05: f64 = 1.002;
#[cfg(feature = "cuda")]
const MIN_RUNNER_P50: f64 = 1.005;

fn validate_exact_candidate_dims(dims_i32: (i32, i32, i32)) -> Result<(), String> {
    if dims_i32 != (512, 2_048, 16) {
        return Err(format!(
            "direct NT skinny candidates require dims_i32=(512,2048,16), found {dims_i32:?}"
        ));
    }
    Ok(())
}

fn exact_windows(value: Option<&OsStr>, required: usize) -> Result<usize, String> {
    let Some(value) = value else {
        return Ok(required);
    };
    let value = value
        .to_str()
        .ok_or_else(|| "MAMBA_RS_NT_SKINNY_WINDOWS is not Unicode".to_string())?;
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid MAMBA_RS_NT_SKINNY_WINDOWS={value}: {error}"))?;
    if parsed != required {
        return Err(format!(
            "MAMBA_RS_NT_SKINNY_WINDOWS must equal {required}, found {parsed}"
        ));
    }
    Ok(parsed)
}

fn checked_percentile(values: &[f64], fraction: f64) -> Result<f64, String> {
    if values.is_empty()
        || !fraction.is_finite()
        || !(0.0..=1.0).contains(&fraction)
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("percentile requires finite positive samples and a [0,1] fraction".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Ok(sorted[index])
}

fn validate_speedup(
    windows: usize,
    p05: f64,
    p50: f64,
    p95: f64,
    minimum_p05: f64,
    minimum_p50: f64,
) -> Result<(), String> {
    if !matches!(windows, SCREEN_WINDOWS | OFFICIAL_WINDOWS)
        || [p05, p50, p95, minimum_p05, minimum_p50]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || p05 > p50
        || p50 > p95
        || p05 < minimum_p05
        || p50 < minimum_p50
    {
        return Err(format!(
            "speedup gate failed: windows={windows} p05={p05} p50={p50} p95={p95} required={minimum_p05}/{minimum_p50}"
        ));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn verify_post_cohort_even_on_error<T, U>(
    body: Result<T, String>,
    post: Result<U, String>,
) -> Result<T, String> {
    match (body, post) {
        (Ok(value), Ok(_)) => Ok(value),
        (Err(body), Ok(_)) => Err(body),
        (Ok(_), Err(post)) => Err(post),
        (Err(body), Err(post)) => Err(format!("{body}; quiet postflight: {post}")),
    }
}

fn splitk32_products(values: &[f32]) -> f32 {
    let mut reduced = 0.0_f32;
    for (chunk, values) in values.chunks(32).enumerate() {
        let partial = values
            .iter()
            .fold(0.0_f32, |sum, value| 1.0_f32.mul_add(*value, sum));
        reduced = if chunk == 0 {
            partial
        } else {
            reduced + partial
        };
    }
    reduced
}

#[test]
fn splitk32_tree_killer_corpus_freezes_chunk_order_and_boundaries() {
    let mut w1 = vec![0.0_f32; 2_048];
    w1[31] = 2.0_f32.powi(25);
    w1[32] = -2.0_f32.powi(25);
    w1[63] = 1.0;
    assert_eq!(splitk32_products(&w1).to_bits(), 0.0_f32.to_bits());

    let mut w2 = vec![0.0_f32; 2_048];
    w2[0] = 2.0_f32.powi(25);
    w2[32] = 1.0;
    w2[64] = -2.0_f32.powi(25);
    w2[96] = 1.0;
    assert_eq!(splitk32_products(&w2).to_bits(), 1.0_f32.to_bits());

    let negative_partial = (-f32::from_bits(1)).mul_add(0.25, 0.0);
    let mut w3 = negative_partial;
    for _ in 1..64 {
        w3 += negative_partial;
    }
    assert_eq!(w3.to_bits(), (-0.0_f32).to_bits());

    let mut w4 = vec![0.0_f32; 2_048];
    w4[31] = f32::from_bits(0x007f_ffff);
    w4[32] = f32::from_bits(1);
    assert_eq!(splitk32_products(&w4).to_bits(), 0x0080_0000);
    w4.fill(0.0);
    w4[2_047] = 1.0;
    assert_eq!(splitk32_products(&w4).to_bits(), 1.0_f32.to_bits());
}

#[test]
fn thin_cols_followup_contract_freezes_identity_offsets_paths_and_official_pair() {
    for required in [
        "const M8N4_GRID: (u32, u32, u32) = (256, 1, 1);",
        "const M8N4_SHARED: usize = 6_528;",
        "production active pointers must remain at allocation offset zero",
        "production_exceptional_bits",
        "production eager bits changed",
        "production graph bits changed",
        "old_graph_identity",
        "post_promotion_correctness_gate",
        "Path::Eager",
        "Path::Graph",
        "measure_candidate_graph",
    ] {
        assert!(TEST_SOURCE.contains(required), "missing {required}");
    }
    let official = TEST_SOURCE
        .rsplit_once("fn skinny_direct_official_101_windows()")
        .map(|(_, tail)| tail)
        .expect("official101 body");
    assert!(!official.contains("measure_all_then_rank("));
    assert!(!official.contains("selected_official_pair("));
    assert!(official.contains("baseline: Arm::Old3Node"));
    assert!(official.contains("baseline: Arm::M8Runner"));
}

#[test]
fn third_pass_contract_requires_raw_repeats_and_full_two_path_cohorts() {
    for required in [
        "exceptional_fixtures",
        "exceptional_k2047_signaling_nan",
        "sparse_fixture(runtime, case, a, b, alpha, false, Some(oracle))",
        "raw production eager bits changed",
        "raw production graph bits changed",
        "capture_raw_production_graph",
        "for path in [Path::Eager, Path::Graph]",
        "paired_path_performance",
        "candidate_graphs",
        "production_holder: Option",
        "path_balanced_us",
        "seed_f32_operands",
    ] {
        assert!(TEST_SOURCE.contains(required), "missing {required}");
    }
    let paired = TEST_SOURCE
        .rsplit_once("fn paired_performance(")
        .and_then(|(_, tail)| tail.split_once("fn require_sm120("))
        .map(|(body, _)| body)
        .expect("paired performance body");
    assert!(!paired.contains("timing probe"));
    assert!(paired.contains("for path in [Path::Eager, Path::Graph]"));
    assert!(paired.matches("validate_speedup(").count() >= 2);
}

#[test]
fn live_gpu_evidence_uses_independent_tree_killers_and_alpha_oracles() {
    for required in [
        "fn tree_killer_fixtures(",
        "TreeKiller::ChunkBoundary31_32_63",
        "TreeKiller::ReductionOrder64",
        "TreeKiller::NegativeZeroSeed",
        "TreeKiller::MinNormalAndEndpoint2047",
        "fn alpha_fixture(",
        "-0.75_f32",
        "production_correctness_gate(&runtime, &mut fixture, case, true)",
        "correctness_gate(&runtime, &kernels, &mut fixture, case)",
        "post_promotion_correctness_gate(&runtime, &kernels, &mut fixture, case)",
    ] {
        assert!(
            TEST_SOURCE.contains(required),
            "missing live GPU evidence: {required}"
        );
    }
    let correctness = TEST_SOURCE
        .rsplit_once("fn skinny_direct_correctness_resources_graph_and_redzones()")
        .map(|(_, body)| body)
        .expect("live correctness test");
    assert!(correctness.contains("for (case, mut fixture) in tree_killer_fixtures(&runtime)?"));
    assert!(correctness.contains("let (case, mut fixture) = alpha_fixture(&runtime)?"));
    assert!(!correctness.contains("exceptional.oracle = production_exceptional_bits"));
}

#[test]
fn sparse_oracles_preserve_alpha_zero_nonfinite_rows_and_partial_exit_sign() {
    let sparse = TEST_SOURCE
        .rsplit_once("fn sparse_fixture(")
        .and_then(|(_, tail)| tail.split_once("fn tree_killer_fixture("))
        .map(|(body, _)| body)
        .expect("sparse fixture body");
    assert!(sparse.contains("let default_bits = (alpha * 0.0_f32).to_bits();"));
    assert!(sparse.contains("vec![default_bits; m * k_out]"));

    let exceptional = TEST_SOURCE
        .rsplit_once("fn exceptional_fixtures(")
        .and_then(|(_, tail)| tail.split_once("fn alpha_fixture("))
        .map(|(body, _)| body)
        .expect("exceptional fixtures body");
    for required in [
        "for row in 0..m",
        "a[row * n + inner] = 1.0;",
        "oracle[row * k_out] = expected;",
    ] {
        assert!(
            exceptional.contains(required),
            "missing exceptional semantic gate: {required}"
        );
    }

    let negative_zero = TEST_SOURCE
        .rsplit_once("TreeKiller::NegativeZeroSeed =>")
        .and_then(|(_, tail)| tail.split_once("TreeKiller::MinNormalAndEndpoint2047"))
        .map(|(body, _)| body)
        .expect("negative-zero witness body");
    assert!(negative_zero.contains("chunk * 32 + 31"));
    assert!(negative_zero.contains("assert_eq!(inner, 2_047);"));
}

#[test]
fn nvidia_ffma_nan_oracles_are_canonical_not_payload_preserving() {
    let exceptional = TEST_SOURCE
        .rsplit_once("fn exceptional_fixtures(")
        .and_then(|(_, tail)| tail.split_once("fn alpha_fixture("))
        .map(|(body, _)| body)
        .expect("exceptional fixtures body");
    for (id, fields) in [
        (
            "exceptional_k64_quiet_nan",
            ["64", "0x7fc2_3456", "0x7fff_ffff"],
        ),
        (
            "exceptional_k2047_signaling_nan",
            ["2_047", "0x7f81_2345", "0x7fff_ffff"],
        ),
    ] {
        let tuple = exceptional
            .split_once(id)
            .and_then(|(_, tail)| tail.split_once(')'))
            .map(|(tuple, _)| tuple)
            .unwrap_or_else(|| panic!("missing NaN fixture {id}"));
        for field in fields {
            assert!(
                tuple.contains(field),
                "missing {field} in canonical NaN oracle {id}"
            );
        }
    }
}

#[test]
fn post_promotion_qualification_freezes_old_route_production_and_runner_pairs() {
    for required in [
        "const PRODUCTION_SYMBOL: &str = \"gemm_bi_nt_m2n16_bk64_splitk32_v1\";",
        "const OLD_TRANSPOSE_SYMBOL: &str = \"gemm_bi_transpose_f32_2d\";",
        "const OLD_PARTIAL_SYMBOL: &str = \"gemm_bi_nn_splitk32_partial\";",
        "const OLD_REDUCER_SYMBOL: &str = \"gemm_bi_splitk_reduce\";",
        "Old3Node,",
        "M8Runner,",
        "splitk_scratch: GpuBuffer",
        "transpose_scratch: GpuBuffer",
        "gemm_bi_nt_m2n16_bk64_splitk32_v1",
        "grid_dim: (256, 1, 1)",
        "block_dim: (64, 1, 1)",
        "shared_mem_bytes: 17_984",
        "baseline: Arm::Old3Node",
        "baseline: Arm::M8Runner",
        "candidate: Arm::Production",
        "old three-node bits differ from production",
    ] {
        assert!(
            TEST_SOURCE.contains(required),
            "post-promotion qualification lost {required}"
        );
    }
    let official = TEST_SOURCE
        .rsplit_once("fn skinny_direct_official_101_windows()")
        .map(|(_, tail)| tail)
        .expect("official101 body");
    assert!(!official.contains("measure_all_then_rank("));
    assert!(!official.contains("selected_official_pair("));
    assert_eq!(official.matches("candidate: Arm::Production").count(), 2);

    assert_eq!(CANDIDATE_SOURCE.matches("extern \"C\"").count(), 1);
    assert_eq!(CANDIDATE_SOURCE.matches("void gemm_bi_nt_thin_").count(), 1);
    let arms = TEST_SOURCE
        .rsplit_once("enum Arm {")
        .and_then(|(_, tail)| tail.split_once('}'))
        .map(|(body, _)| body)
        .expect("post-promotion arms");
    assert_eq!(
        arms.split(',')
            .map(str::trim)
            .filter(|arm| !arm.is_empty())
            .collect::<Vec<_>>(),
        ["Old3Node", "Production", "M8Runner"]
    );
}

#[test]
fn live_resource_contract_uses_measured_safe_ceiling_and_complete_census() {
    let loader = TEST_SOURCE
        .rsplit_once("fn load_kernels(")
        .and_then(|(_, tail)| tail.split_once("fn fixture_with_values("))
        .map(|(body, _)| body)
        .expect("kernel loader body");
    assert_eq!(loader.matches("register_cap: 96").count(), 1);
    assert!(!loader.contains("register_cap: 112"));
    assert!(!loader.contains("register_cap: 64"));

    let gate = TEST_SOURCE
        .rsplit_once("fn resource_gate(")
        .and_then(|(_, tail)| tail.split_once("fn production_request("))
        .map(|(body, _)| body)
        .expect("resource gate body");
    assert!(gate.contains("let mut observations"));
    assert!(gate.contains("let mut failures"));
    assert!(gate.contains("observations.push"));
    assert!(gate.contains("failures.push"));
}

#[test]
fn direct_candidates_reject_every_neighbor_shape_before_enqueue() {
    assert!(validate_exact_candidate_dims((512, 2_048, 16)).is_ok());
    for dims in [
        (511, 2_048, 16),
        (513, 2_048, 16),
        (512, 2_047, 16),
        (512, 2_049, 16),
        (512, 2_048, 15),
        (512, 2_048, 17),
    ] {
        assert!(
            validate_exact_candidate_dims(dims).is_err(),
            "accepted {dims:?}"
        );
    }
    let launch = TEST_SOURCE
        .rsplit_once("fn launch(")
        .and_then(|(_, tail)| tail.split_once("fn capture("))
        .map(|(body, _)| body)
        .expect("candidate launch body");
    let gate = launch
        .find("super::validate_exact_candidate_dims(fixture.dims_i32)?;")
        .expect("exact candidate gate");
    let enqueue = launch.find("launch_builder").expect("candidate enqueue");
    assert!(gate < enqueue);
    let capture = TEST_SOURCE
        .rsplit_once("fn capture(")
        .and_then(|(_, tail)| tail.split_once("fn bytes_of"))
        .map(|(body, _)| body)
        .expect("candidate capture body");
    let capture_gate = capture
        .find("super::validate_exact_candidate_dims(fixture.dims_i32)?;")
        .expect("pre-capture exact candidate gate");
    let begin_capture = capture
        .find("capture_into_graph")
        .expect("stream capture entry");
    assert!(capture_gate < begin_capture);
    let run = CANDIDATE_SOURCE
        .split_once("void run(")
        .and_then(|(_, tail)| tail.split_once("issue_stage(shared"))
        .map(|(body, _)| body)
        .expect("CUDA pre-enqueue gate");
    assert!(run.contains("m != 512 || n != 2048 || k_out != 16"));
}

#[test]
fn stage_loader_vectorizes_row_major_segments_with_padded_banks() {
    let issue = CANDIDATE_SOURCE
        .split_once("void issue_stage(")
        .and_then(|(_, tail)| tail.split_once("void run("))
        .map(|(body, _)| body)
        .expect("issue_stage body");
    for required in [
        "int local_row = linear / SEGMENTS;",
        "int segment = linear - local_row * SEGMENTS;",
        "int local_column = linear / SEGMENTS;",
        "int reduction = segment * COPY_FLOATS;",
        "local_row * ROW_STRIDE + reduction",
        "A_STAGE + local_column * ROW_STRIDE + reduction",
        "cp.async.ca.shared.global [%0], [%1], 16, %2",
        "global_reduction + COPY_FLOATS - 1 < dims.y",
    ] {
        assert!(
            issue.contains(required),
            "missing contiguous-K loader mapping: {required}"
        );
    }
    for required in [
        "static constexpr int COPY_FLOATS = 4;",
        "static constexpr int ROW_STRIDE = NT_THIN_BK + COPY_FLOATS;",
        "static constexpr int SEGMENTS = NT_THIN_BK / COPY_FLOATS;",
    ] {
        assert!(
            CANDIDATE_SOURCE.contains(required),
            "missing vector layout: {required}"
        );
    }
    assert!(!issue.contains(", 4, %2"));

    for extent in [4, 8] {
        let coordinates = (0..extent * 16)
            .map(|linear| (linear / 16, linear % 16))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(coordinates.len(), extent * 16);
        assert!(
            coordinates
                .iter()
                .all(|&(outer, segment)| { outer < extent && segment < 16 })
        );
    }
    assert_eq!(
        (0..8).map(|row| row * 68 % 32).collect::<Vec<_>>(),
        vec![0, 4, 8, 12, 16, 20, 24, 28]
    );
    for required in [
        "stage_a[local_row * A_STRIDE + dot]",
        "stage_b[local_column * B_STRIDE + dot]",
        "SkinnyNtM8N4SplitK32Kernel::SHARED_BYTES == 6528",
    ] {
        assert!(
            CANDIDATE_SOURCE.contains(required),
            "missing row-major contract: {required}"
        );
    }
}

#[test]
fn splitk32_killers_reject_competing_reduction_trees() {
    let mut w1 = vec![0.0_f32; 2_048];
    w1[31] = 2.0_f32.powi(25);
    w1[32] = -2.0_f32.powi(25);
    w1[63] = 1.0;
    let sequential = w1.iter().fold(0.0_f32, |sum, value| sum + value);
    assert_eq!(splitk32_products(&w1).to_bits(), 0.0_f32.to_bits());
    assert_eq!(sequential.to_bits(), 1.0_f32.to_bits());

    let mut w2 = vec![0.0_f32; 2_048];
    w2[0] = 2.0_f32.powi(25);
    w2[32] = 1.0;
    w2[64] = -2.0_f32.powi(25);
    w2[96] = 1.0;
    let partials = w2
        .chunks(32)
        .map(|chunk| chunk.iter().fold(0.0_f32, |sum, value| sum + value))
        .collect::<Vec<_>>();
    let reversed = partials
        .iter()
        .rev()
        .fold(0.0_f32, |sum, value| sum + value);
    let balanced = (partials[0] + partials[1]) + (partials[2] + partials[3]);
    assert_eq!(splitk32_products(&w2).to_bits(), 1.0_f32.to_bits());
    assert_eq!(reversed.to_bits(), 0.0_f32.to_bits());
    assert_eq!(balanced.to_bits(), 0.0_f32.to_bits());

    let negative_partial = (-f32::from_bits(1)).mul_add(0.25, 0.0);
    let seeded = (1..64).fold(negative_partial, |sum, _| sum + negative_partial);
    let plus_zero_seed = (0..64).fold(0.0_f32, |sum, _| sum + negative_partial);
    assert_eq!(seeded.to_bits(), (-0.0_f32).to_bits());
    assert_eq!(plus_zero_seed.to_bits(), 0.0_f32.to_bits());

    let mut w4 = vec![0.0_f32; 2_048];
    w4[31] = f32::from_bits(0x007f_ffff);
    w4[32] = f32::from_bits(1);
    assert_eq!(splitk32_products(&w4).to_bits(), 0x0080_0000);
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PhysicalIdentity {
    symbol: String,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared: u32,
    argument_count: usize,
    arguments_digest: [u8; 32],
}

fn validate_physical_identity(
    actual: &PhysicalIdentity,
    expected: &PhysicalIdentity,
) -> Result<(), String> {
    if actual != expected || actual.argument_count != 7 || actual.arguments_digest == [0; 32] {
        return Err(format!(
            "physical identity mismatch: actual={actual:?} expected={expected:?}"
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResourceContract {
    threads: u32,
    static_shared: usize,
    dynamic_shared: usize,
    register_cap: usize,
    minimum_blocks: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResourceObservation {
    threads: u32,
    static_shared: usize,
    dynamic_shared: usize,
    registers: usize,
    local_bytes: usize,
    active_blocks: u32,
}

fn validate_resources(
    actual: ResourceObservation,
    expected: ResourceContract,
) -> Result<(), String> {
    if actual.threads != expected.threads
        || actual.static_shared != expected.static_shared
        || actual.dynamic_shared != expected.dynamic_shared
        || actual.registers > expected.register_cap
        || actual.local_bytes != 0
        || actual.active_blocks < expected.minimum_blocks
    {
        return Err(format!(
            "resource contract failed: actual={actual:?} expected={expected:?}"
        ));
    }
    Ok(())
}

const THIN_M8N4: &str = "gemm_bi_nt_thin_m8n4_bk64_splitk32_exp_v1";
#[cfg(feature = "cuda")]
const PRODUCTION_SYMBOL: &str = "gemm_bi_nt_m2n16_bk64_splitk32_v1";
#[cfg(feature = "cuda")]
const OLD_TRANSPOSE_SYMBOL: &str = "gemm_bi_transpose_f32_2d";
#[cfg(feature = "cuda")]
const OLD_PARTIAL_SYMBOL: &str = "gemm_bi_nn_splitk32_partial";
#[cfg(feature = "cuda")]
const OLD_REDUCER_SYMBOL: &str = "gemm_bi_splitk_reduce";
const CANDIDATE_SOURCE: &str = include_str!("gemm_bi_scalar_nt_skinny_direct_experiment.cu");
const TEST_SOURCE: &str = include_str!("gemm_bi_scalar_nt_skinny_direct_experiment.rs");
#[cfg(feature = "cuda")]
const SCALAR_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/scalar.cu");

#[test]
fn timing_contract_is_fail_closed_at_every_boundary() {
    assert_eq!(exact_windows(None, SCREEN_WINDOWS).unwrap(), SCREEN_WINDOWS);
    assert_eq!(
        exact_windows(Some(OsStr::new("101")), OFFICIAL_WINDOWS).unwrap(),
        OFFICIAL_WINDOWS
    );
    for invalid in ["0", "20", "22", "100", "102", "bad"] {
        assert!(exact_windows(Some(OsStr::new(invalid)), SCREEN_WINDOWS).is_err());
    }
    let samples = (1..=OFFICIAL_WINDOWS)
        .map(|value| value as f64)
        .collect::<Vec<_>>();
    assert_eq!(checked_percentile(&samples, 0.05).unwrap(), 6.0);
    assert_eq!(checked_percentile(&samples, 0.50).unwrap(), 51.0);
    for invalid in [
        vec![],
        vec![0.0],
        vec![-1.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
    ] {
        assert!(checked_percentile(&invalid, 0.5).is_err());
    }
    assert!(validate_speedup(21, 1.005, 1.01, 1.2, 1.005, 1.01).is_ok());
    assert!(validate_speedup(21, 1.004_999, 1.01, 1.2, 1.005, 1.01).is_err());
    assert!(validate_speedup(21, 1.005, 1.009_999, 1.2, 1.005, 1.01).is_err());
    assert!(validate_speedup(20, 2.0, 2.0, 2.0, 1.005, 1.01).is_err());
    assert!(validate_speedup(21, f64::NAN, 2.0, 2.0, 1.005, 1.01).is_err());
}

#[test]
fn graph_policy_allows_launch_floor_without_weakening_eager_gates() {
    assert!(validate_speedup(21, 0.9996, 1.0001, 1.001, 0.9995, 1.0).is_ok());
    assert!(validate_speedup(21, 0.9996, 1.0001, 1.001, 1.005, 1.01).is_err());
    let runtime = TEST_SOURCE
        .rsplit_once("mod cuda_experiment {")
        .map(|(_, runtime)| runtime)
        .expect("CUDA experiment module");
    for required in [
        "struct Thresholds",
        "struct PathThresholds",
        "thresholds: PathThresholds",
        "Path::Eager => thresholds.eager",
        "Path::Graph => thresholds.graph",
        "p05: 0.9995",
        "p50: 1.0",
    ] {
        assert!(
            runtime.contains(required),
            "missing path policy: {required}"
        );
    }
}

#[test]
fn physical_and_resource_contracts_reject_every_single_field_mutation() {
    let identity = PhysicalIdentity {
        symbol: THIN_M8N4.into(),
        grid: (256, 1, 1),
        block: (32, 1, 1),
        shared: 6_528,
        argument_count: 7,
        arguments_digest: [7; 32],
    };
    assert!(validate_physical_identity(&identity, &identity).is_ok());
    for mutate in [
        |value: &mut PhysicalIdentity| value.symbol.push_str("_wrong"),
        |value: &mut PhysicalIdentity| value.grid.0 += 1,
        |value: &mut PhysicalIdentity| value.block.0 += 1,
        |value: &mut PhysicalIdentity| value.shared += 4,
        |value: &mut PhysicalIdentity| value.argument_count -= 1,
        |value: &mut PhysicalIdentity| value.arguments_digest = [0; 32],
    ] {
        let mut changed = identity.clone();
        mutate(&mut changed);
        assert!(validate_physical_identity(&identity, &changed).is_err());
    }

    let contract = ResourceContract {
        threads: 32,
        static_shared: 0,
        dynamic_shared: 6_528,
        register_cap: 96,
        minimum_blocks: 4,
    };
    let observation = ResourceObservation {
        threads: 32,
        static_shared: 0,
        dynamic_shared: 6_528,
        registers: 48,
        local_bytes: 0,
        active_blocks: 4,
    };
    assert!(validate_resources(observation, contract).is_ok());
    for mutate in [
        |value: &mut ResourceObservation| value.threads += 1,
        |value: &mut ResourceObservation| value.static_shared += 4,
        |value: &mut ResourceObservation| value.dynamic_shared += 4,
        |value: &mut ResourceObservation| value.registers = 97,
        |value: &mut ResourceObservation| value.local_bytes = 4,
        |value: &mut ResourceObservation| value.active_blocks = 3,
    ] {
        let mut changed = observation;
        mutate(&mut changed);
        assert!(validate_resources(changed, contract).is_err());
    }
}

#[test]
fn skinny_direct_candidates_are_isolated_single_owner_kernels() {
    let registry = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
    assert!(
        !registry.contains(THIN_M8N4),
        "{THIN_M8N4} escaped into production"
    );
    let marker = format!("void {THIN_M8N4}(");
    let (_, tail) = CANDIDATE_SOURCE
        .split_once(&marker)
        .unwrap_or_else(|| panic!("missing candidate {THIN_M8N4}"));
    let (parameters, _) = tail
        .split_once(") {")
        .unwrap_or_else(|| panic!("missing parameter list for {THIN_M8N4}"));
    assert_eq!(
        parameters.matches(',').count() + 1,
        7,
        "{THIN_M8N4} ABI drift"
    );
    for required in [
        "NT_THIN_BK 64",
        "cp.async.ca.shared.global",
        "cp.async.wait_group 0",
        "partial_0 = __fmaf_rn(",
        "partial_1 = __fmaf_rn(",
        "SkinnyNtM8N4SplitK32Kernel",
    ] {
        assert!(
            CANDIDATE_SOURCE.contains(required),
            "candidate lost {required}"
        );
    }
    for forbidden in ["atomic", "split_k", "mma.sync"] {
        assert!(
            !CANDIDATE_SOURCE.contains(forbidden),
            "candidate contains forbidden token {forbidden}"
        );
    }
}

#[test]
fn harness_requires_independent_exactness_graph_guard_resource_and_timing_evidence() {
    let module_marker = ["mod cuda_", "experiment {"].concat();
    let runtime = TEST_SOURCE
        .rsplit_once(&module_marker)
        .map(|(_, runtime)| runtime)
        .expect("CUDA experiment module");
    for required in [
        "rounding_sensitive_values",
        "nt_cpu_oracle",
        "mul_add",
        "CPU sequential bits differ",
        "capture_into_graph",
        "graph bits differ from eager",
        "validate_red_zones",
        "local_size_bytes",
        "num_regs",
        "occupancy_max_active_blocks_per_multiprocessor",
        "ABBA",
        "BAAB",
        "MAMBA_RS_NT_SKINNY_WINDOWS",
    ] {
        assert!(runtime.contains(required), "harness lost {required}");
    }
}

#[test]
fn thin_cols_post_promotion_contract_is_complete_and_production_anchored() {
    let marker = format!("void {THIN_M8N4}(");
    let (_, tail) = CANDIDATE_SOURCE
        .split_once(&marker)
        .unwrap_or_else(|| panic!("missing exact thin-cols runner {THIN_M8N4}"));
    let (parameters, _) = tail
        .split_once(") {")
        .unwrap_or_else(|| panic!("missing ABI for {THIN_M8N4}"));
    assert_eq!(parameters.matches(',').count() + 1, 7, "{THIN_M8N4} ABI");
    for required in [
        "SkinnyNtM8N4SplitK32Kernel",
        "partial_0",
        "partial_1",
        "reduced = partial_0;",
        "reduced = __fadd_rn(reduced, partial_1);",
        "__fmul_rn(alpha, reduced)",
    ] {
        assert!(CANDIDATE_SOURCE.contains(required), "missing {required}");
    }
    for required in [
        "PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1)",
        "gemm_bi_transpose_f32_2d",
        "gemm_bi_nn_splitk32_partial",
        "gemm_bi_splitk_reduce",
        "launch_old3",
        "old3_correctness_gate",
        "verify_post_cohort_even_on_error",
    ] {
        assert!(TEST_SOURCE.contains(required), "missing {required}");
    }
    for forbidden in ["--use_fast_math", ".ftz", "atomic", "split_k"] {
        assert!(
            !CANDIDATE_SOURCE.contains(forbidden),
            "forbidden {forbidden}"
        );
    }
    assert!(CANDIDATE_SOURCE.contains("int reduction_offset, int3 dims)"));
    let paired = TEST_SOURCE
        .rsplit_once("fn paired_performance(")
        .and_then(|(_, tail)| tail.split_once(") -> Result<(), String>"))
        .map(|(parameters, _)| parameters)
        .expect("paired_performance signature");
    assert!(paired.matches(',').count() < 7);
}

#[cfg(feature = "cuda")]
fn compose_cuda_source() -> String {
    [
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        SCALAR_SOURCE,
        CANDIDATE_SOURCE,
    ]
    .iter()
    .map(|source| {
        source
            .lines()
            .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
            .collect::<Vec<_>>()
            .join("\n")
    })
    .collect::<Vec<_>>()
    .join("\n")
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA NVRTC but does not launch GPU work"]
fn skinny_direct_candidates_compile_for_compute80() {
    let options = cudarc::nvrtc::CompileOptions {
        arch: Some("compute_80"),
        options: vec![
            "--fmad=true".to_owned(),
            "--extra-device-vectorization".to_owned(),
            "-DNDEBUG".to_owned(),
            "-DGEMM_BI_GROUP_M=8".to_owned(),
        ],
        include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(compose_cuda_source(), options)
        .expect("scalar NT skinny candidates must compile for compute_80")
        .to_src();
    assert!(ptx.contains(THIN_M8N4), "compute_80 PTX lost {THIN_M8N4}");
    assert!(ptx.contains("fma.rn.f32"));
}

#[cfg(feature = "cuda")]
mod cuda_experiment {
    use std::ffi::CStr;
    use std::mem::size_of;
    use std::sync::Arc;

    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, CudaStream, LaunchConfig, PushKernelArg, sys,
    };
    use mamba_rs::mamba_ssm::gpu::blas::gpu_gemm_bi_backward_dx_raw;
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationF32Epilogue, PhysicalQualificationRequest, PhysicalQualificationRoute,
        QualifiedPhysicalLaunch, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp;
    use sha2::{Digest as _, Sha256};

    use super::common::gpu_quiet::QuietGpu;

    use super::{
        MIN_PRODUCTION_P05, MIN_PRODUCTION_P50, MIN_RUNNER_P05, MIN_RUNNER_P50, OFFICIAL_WINDOWS,
        OLD_PARTIAL_SYMBOL, OLD_REDUCER_SYMBOL, OLD_TRANSPOSE_SYMBOL, PRODUCTION_SYMBOL,
        PhysicalIdentity, ResourceContract, ResourceObservation, SCREEN_WINDOWS, THIN_M8N4,
        checked_percentile, compose_cuda_source, exact_windows, validate_physical_identity,
        validate_resources, validate_speedup, verify_post_cohort_even_on_error,
    };

    const GUARD_ELEMENTS: usize = 64;
    const A_GUARD_BITS: u32 = 0x7fc0_a51d;
    const B_GUARD_BITS: u32 = 0x7fc0_b51d;
    const C_GUARD_BITS: u32 = 0x7fc0_c51d;
    const CORRECTNESS_REPEATS: usize = 3;
    const TARGET_WINDOW_US: f64 = 10_000.0;
    const M8N4_GRID: (u32, u32, u32) = (256, 1, 1);
    const M8N4_SHARED: usize = 6_528;

    #[derive(Clone, Copy)]
    struct Case {
        id: &'static str,
        dims: (usize, usize, usize),
    }

    const THIN_COLS: Case = Case {
        id: "thin_cols",
        dims: (512, 16, 2_048),
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        Old3Node,
        Production,
        M8Runner,
    }

    #[derive(Clone, Copy, Debug)]
    enum Path {
        Eager,
        Graph,
    }

    struct Runtime {
        device: GpuDevice,
        production: GpuCtx,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
    }

    struct Kernel {
        function: CudaFunction,
        config: LaunchConfig,
        symbol: &'static str,
        resources: ResourceContract,
    }

    struct Kernels {
        old_transpose: CudaFunction,
        old_partial: CudaFunction,
        old_reducer: CudaFunction,
        m8n4: Kernel,
    }

    struct GuardedBuffer {
        buffer: GpuBuffer,
        expected: Vec<f32>,
        active_offset: usize,
        active_len: usize,
        guard_bits: u32,
    }

    impl GuardedBuffer {
        fn input(
            stream: &Arc<CudaStream>,
            active: Vec<f32>,
            pointer_offset: usize,
            guard_bits: u32,
        ) -> Result<Self, String> {
            Self::new(stream, active, pointer_offset, guard_bits)
        }

        fn output(
            stream: &Arc<CudaStream>,
            active_len: usize,
            pointer_offset: usize,
        ) -> Result<Self, String> {
            Self::new(
                stream,
                vec![f32::from_bits(0x3e80_0001); active_len],
                pointer_offset,
                C_GUARD_BITS,
            )
        }

        fn suffix_only(
            stream: &Arc<CudaStream>,
            active: Vec<f32>,
            guard_bits: u32,
        ) -> Result<Self, String> {
            let active_len = active.len();
            let mut expected = active;
            expected.extend(std::iter::repeat_n(
                f32::from_bits(guard_bits),
                GUARD_ELEMENTS,
            ));
            Ok(Self {
                buffer: GpuBuffer::from_cpu(stream, &expected)?,
                expected,
                active_offset: 0,
                active_len,
                guard_bits,
            })
        }

        fn new(
            stream: &Arc<CudaStream>,
            active: Vec<f32>,
            pointer_offset: usize,
            guard_bits: u32,
        ) -> Result<Self, String> {
            let active_offset = GUARD_ELEMENTS
                .checked_add(pointer_offset)
                .ok_or_else(|| "guard offset overflows usize".to_string())?;
            let active_len = active.len();
            let total = active_offset
                .checked_add(active_len)
                .and_then(|value| value.checked_add(GUARD_ELEMENTS))
                .ok_or_else(|| "guarded allocation size overflows usize".to_string())?;
            let mut expected = vec![f32::from_bits(guard_bits); total];
            expected[active_offset..active_offset + active_len].copy_from_slice(&active);
            Ok(Self {
                buffer: GpuBuffer::from_cpu(stream, &expected)?,
                expected,
                active_offset,
                active_len,
                guard_bits,
            })
        }

        fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
            self.buffer.raw_ptr_at(stream, self.active_offset)
        }

        fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
            self.buffer.upload(stream, &self.expected)
        }

        fn validate_red_zones(
            &self,
            stream: &Arc<CudaStream>,
            label: &str,
        ) -> Result<Vec<f32>, String> {
            let actual = self.buffer.to_cpu(stream)?;
            for (index, value) in actual[..self.active_offset]
                .iter()
                .chain(&actual[self.active_offset + self.active_len..])
                .enumerate()
            {
                if value.to_bits() != self.guard_bits {
                    return Err(format!(
                        "{label} red zone changed at guard element {index}: 0x{:08x}",
                        value.to_bits()
                    ));
                }
            }
            Ok(actual)
        }

        fn active_bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
            let actual = self.validate_red_zones(stream, label)?;
            Ok(
                actual[self.active_offset..self.active_offset + self.active_len]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect(),
            )
        }

        fn validate_unchanged(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
            let actual = self.validate_red_zones(stream, label)?;
            if actual
                .iter()
                .zip(&self.expected)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                return Err(format!("{label} read-only input changed"));
            }
            Ok(())
        }
    }

    struct Fixture {
        a: GuardedBuffer,
        b: GuardedBuffer,
        production_a: GuardedBuffer,
        production_b: GuardedBuffer,
        production_output: GuardedBuffer,
        old_output: GuardedBuffer,
        m8n4_output: GuardedBuffer,
        transpose_scratch: GpuBuffer,
        splitk_scratch: GpuBuffer,
        oracle: Vec<u32>,
        dims_i32: (i32, i32, i32),
        alpha: f32,
    }

    fn rounding_sensitive_values(len: usize, mut state: u64) -> Vec<f32> {
        (0..len)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if index % 4_093 == 0 {
                    if index & 1 == 0 { 0.0 } else { -0.0 }
                } else {
                    let sign = ((state >> 63) as u32) << 31;
                    let exponent = (124 + ((state >> 29) as u32 % 7)) << 23;
                    let mantissa = ((state as u32) & 0x007f_ffff) | 1;
                    f32::from_bits(sign | exponent | mantissa)
                }
            })
            .collect()
    }

    fn nt_cpu_oracle(case: Case, a: &[f32], b: &[f32], alpha: f32) -> Vec<u32> {
        let (m, k_out, n) = case.dims;
        let mut output = Vec::with_capacity(m * k_out);
        for row in 0..m {
            for column in 0..k_out {
                let mut accumulator = 0.0f32;
                for chunk in 0..n.div_ceil(32) {
                    let mut partial = 0.0f32;
                    for inner in chunk * 32..((chunk + 1) * 32).min(n) {
                        partial = a[row * n + inner].mul_add(b[column * n + inner], partial);
                    }
                    accumulator = if chunk == 0 {
                        partial
                    } else {
                        accumulator + partial
                    };
                }
                output.push((alpha * accumulator).to_bits());
            }
        }
        output
    }

    fn compile_ptx(target: &'static str) -> Result<String, String> {
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some(target),
            options: vec![
                "--fmad=true".to_owned(),
                "--extra-device-vectorization".to_owned(),
                "-DNDEBUG".to_owned(),
                "-DGEMM_BI_GROUP_M=16".to_owned(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        cudarc::nvrtc::compile_ptx_with_opts(compose_cuda_source(), options)
            .map(|image| image.to_src())
            .map_err(|error| format!("compile scalar NT skinny experiment: {error:?}"))
    }

    fn new_runtime() -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        if device.compute_capability.0 < 8 {
            return Err(format!(
                "scalar NT skinny experiment requires SM80+, found {:?}",
                device.compute_capability
            ));
        }
        let ptx = compile_ptx(device.nvrtc_target())?;
        let production = GpuCtx::new(&device)?;
        production.set_batch_invariant(true);
        production.set_bi_gemm_family(BiGemmFamily::Triad);
        production.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        let stream = device.fork_stream()?;
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|error| format!("load scalar NT skinny module: {error:?}"))?;
        Ok(Runtime {
            device,
            production,
            stream,
            module,
        })
    }

    fn make_kernel(
        runtime: &Runtime,
        symbol: &'static str,
        tile: (usize, usize),
        resources: ResourceContract,
        case: Case,
    ) -> Result<Kernel, String> {
        let function = runtime
            .module
            .load_function(symbol)
            .map_err(|error| format!("load {symbol}: {error:?}"))?;
        if resources.dynamic_shared > 0 {
            function
                .set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    resources.dynamic_shared as i32,
                )
                .map_err(|error| format!("set shared-memory attribute for {symbol}: {error:?}"))?;
        }
        let blocks = case
            .dims
            .0
            .div_ceil(tile.0)
            .checked_mul(case.dims.1.div_ceil(tile.1))
            .ok_or_else(|| format!("{symbol} grid overflows usize"))?;
        Ok(Kernel {
            function,
            config: LaunchConfig {
                grid_dim: (
                    u32::try_from(blocks).map_err(|_| format!("{symbol} grid exceeds u32"))?,
                    1,
                    1,
                ),
                block_dim: (resources.threads, 1, 1),
                shared_mem_bytes: resources.dynamic_shared as u32,
            },
            symbol,
            resources,
        })
    }

    fn load_kernels(runtime: &Runtime, case: Case) -> Result<Kernels, String> {
        let kernels = Kernels {
            old_transpose: runtime
                .module
                .load_function(OLD_TRANSPOSE_SYMBOL)
                .map_err(|error| format!("load {OLD_TRANSPOSE_SYMBOL}: {error:?}"))?,
            old_partial: runtime
                .module
                .load_function(OLD_PARTIAL_SYMBOL)
                .map_err(|error| format!("load {OLD_PARTIAL_SYMBOL}: {error:?}"))?,
            old_reducer: runtime
                .module
                .load_function(OLD_REDUCER_SYMBOL)
                .map_err(|error| format!("load {OLD_REDUCER_SYMBOL}: {error:?}"))?,
            m8n4: make_kernel(
                runtime,
                THIN_M8N4,
                (8, 4),
                ResourceContract {
                    threads: 32,
                    static_shared: 0,
                    dynamic_shared: M8N4_SHARED,
                    register_cap: 96,
                    minimum_blocks: 4,
                },
                case,
            )?,
        };
        if kernels.m8n4.config.grid_dim != M8N4_GRID
            || kernels.m8n4.config.shared_mem_bytes as usize != M8N4_SHARED
        {
            return Err("M8N4 frozen launch identity drifted".into());
        }
        Ok(kernels)
    }

    fn fixture_with_values(
        runtime: &Runtime,
        case: Case,
        a_values: Vec<f32>,
        b_values: Vec<f32>,
        alpha: f32,
        unaligned: bool,
        oracle: Option<Vec<u32>>,
    ) -> Result<Fixture, String> {
        let (m, k_out, n) = case.dims;
        if a_values.len() != m * n || b_values.len() != k_out * n {
            return Err(format!("{} fixture input extent changed", case.id));
        }
        let oracle = oracle.unwrap_or_else(|| nt_cpu_oracle(case, &a_values, &b_values, alpha));
        let output_len = m * k_out;
        let offsets = if unaligned {
            (1, 2, 3, 1)
        } else {
            (0, 0, 0, 0)
        };
        Ok(Fixture {
            production_a: GuardedBuffer::suffix_only(
                &runtime.production.stream,
                a_values.clone(),
                A_GUARD_BITS,
            )?,
            production_b: GuardedBuffer::suffix_only(
                &runtime.production.stream,
                b_values.clone(),
                B_GUARD_BITS,
            )?,
            a: GuardedBuffer::input(&runtime.stream, a_values, offsets.0, A_GUARD_BITS)?,
            b: GuardedBuffer::input(&runtime.stream, b_values, offsets.1, B_GUARD_BITS)?,
            production_output: GuardedBuffer::suffix_only(
                &runtime.production.stream,
                vec![f32::from_bits(0x3e80_0001); output_len],
                C_GUARD_BITS,
            )?,
            old_output: GuardedBuffer::output(&runtime.stream, output_len, offsets.2)?,
            m8n4_output: GuardedBuffer::output(&runtime.stream, output_len, offsets.3)?,
            transpose_scratch: GpuBuffer::zeros(&runtime.stream, n * k_out)?,
            splitk_scratch: GpuBuffer::zeros(&runtime.stream, n.div_ceil(32) * m * k_out)?,
            oracle,
            dims_i32: (m as i32, n as i32, k_out as i32),
            alpha,
        })
    }

    fn new_fixture(runtime: &Runtime, case: Case, unaligned: bool) -> Result<Fixture, String> {
        let (m, k_out, n) = case.dims;
        fixture_with_values(
            runtime,
            case,
            rounding_sensitive_values(m * n, 0xa51d_0001),
            rounding_sensitive_values(k_out * n, 0xb51d_0002),
            1.0,
            unaligned,
            None,
        )
    }

    #[derive(Clone, Copy)]
    enum TreeKiller {
        ChunkBoundary31_32_63,
        ReductionOrder64,
        NegativeZeroSeed,
        MinNormalAndEndpoint2047,
    }

    fn sparse_fixture(
        runtime: &Runtime,
        id: &'static str,
        a_entries: &[(usize, f32)],
        b_entries: &[(usize, usize, f32)],
        expected: &[(usize, u32)],
        alpha: f32,
    ) -> Result<(Case, Fixture), String> {
        let case = Case { id, ..THIN_COLS };
        let (m, k_out, n) = case.dims;
        let mut a = vec![0.0; m * n];
        let mut b = vec![0.0; k_out * n];
        let default_bits = (alpha * 0.0_f32).to_bits();
        let mut oracle = vec![default_bits; m * k_out];
        for &(inner, value) in a_entries {
            a[inner] = value;
        }
        for &(column, inner, value) in b_entries {
            b[column * n + inner] = value;
        }
        for &(index, bits) in expected {
            oracle[index] = bits;
        }
        fixture_with_values(runtime, case, a, b, alpha, false, Some(oracle))
            .map(|fixture| (case, fixture))
    }

    fn tree_killer_fixture(
        runtime: &Runtime,
        witness: TreeKiller,
    ) -> Result<(Case, Fixture), String> {
        match witness {
            TreeKiller::ChunkBoundary31_32_63 => sparse_fixture(
                runtime,
                "splitk32_w1_boundaries_31_32_63",
                &[(31, 1.0), (32, 1.0), (63, 1.0)],
                &[
                    (0, 31, 2.0_f32.powi(25)),
                    (0, 32, -2.0_f32.powi(25)),
                    (0, 63, 1.0),
                ],
                &[(0, 0.0_f32.to_bits())],
                1.0,
            ),
            TreeKiller::ReductionOrder64 => sparse_fixture(
                runtime,
                "splitk32_w2_reduction_order_64",
                &[(0, 1.0), (32, 1.0), (64, 1.0), (96, 1.0)],
                &[
                    (0, 0, 2.0_f32.powi(25)),
                    (0, 32, 1.0),
                    (0, 64, -2.0_f32.powi(25)),
                    (0, 96, 1.0),
                ],
                &[(0, 1.0_f32.to_bits())],
                1.0,
            ),
            TreeKiller::NegativeZeroSeed => {
                let entries = (0..64)
                    .map(|chunk| (chunk * 32 + 31, -f32::from_bits(1)))
                    .collect::<Vec<_>>();
                let b_entries = (0..64)
                    .map(|chunk| (0, chunk * 32 + 31, 0.25))
                    .collect::<Vec<_>>();
                let inner = entries.last().expect("negative-zero chunks").0;
                assert_eq!(inner, 2_047);
                sparse_fixture(
                    runtime,
                    "splitk32_w3_negative_zero_seed",
                    &entries,
                    &b_entries,
                    &[(0, (-0.0_f32).to_bits())],
                    1.0,
                )
            }
            TreeKiller::MinNormalAndEndpoint2047 => sparse_fixture(
                runtime,
                "splitk32_w4_min_normal_endpoint_2047",
                &[(31, 1.0), (32, 1.0), (2_047, 1.0)],
                &[
                    (0, 31, f32::from_bits(0x007f_ffff)),
                    (0, 32, f32::from_bits(1)),
                    (1, 2_047, 1.0),
                ],
                &[(0, 0x0080_0000), (1, 1.0_f32.to_bits())],
                1.0,
            ),
        }
    }

    fn tree_killer_fixtures(runtime: &Runtime) -> Result<Vec<(Case, Fixture)>, String> {
        [
            TreeKiller::ChunkBoundary31_32_63,
            TreeKiller::ReductionOrder64,
            TreeKiller::NegativeZeroSeed,
            TreeKiller::MinNormalAndEndpoint2047,
        ]
        .into_iter()
        .map(|witness| tree_killer_fixture(runtime, witness))
        .collect()
    }

    fn exceptional_fixtures(runtime: &Runtime) -> Result<Vec<(Case, Fixture)>, String> {
        [
            ("exceptional_k0_min_subnormal", 0, 0x0000_0001, 0x0000_0001),
            (
                "exceptional_k31_max_subnormal",
                31,
                0x007f_ffff,
                0x007f_ffff,
            ),
            (
                "exceptional_k32_positive_infinity",
                32,
                0x7f80_0000,
                0x7f80_0000,
            ),
            (
                "exceptional_k63_negative_infinity",
                63,
                0xff80_0000,
                0xff80_0000,
            ),
            ("exceptional_k64_quiet_nan", 64, 0x7fc2_3456, 0x7fff_ffff),
            (
                "exceptional_k2047_signaling_nan",
                2_047,
                0x7f81_2345,
                0x7fff_ffff,
            ),
        ]
        .into_iter()
        .map(|(id, inner, input, expected)| {
            let case = Case { id, ..THIN_COLS };
            let (m, k_out, n) = case.dims;
            let mut a = vec![0.0; m * n];
            let mut b = vec![0.0; k_out * n];
            let mut oracle = vec![0.0_f32.to_bits(); m * k_out];
            b[inner] = f32::from_bits(input);
            for row in 0..m {
                a[row * n + inner] = 1.0;
                oracle[row * k_out] = expected;
            }
            fixture_with_values(runtime, case, a, b, 1.0, false, Some(oracle))
                .map(|fixture| (case, fixture))
        })
        .collect()
    }

    fn alpha_fixture(runtime: &Runtime) -> Result<(Case, Fixture), String> {
        sparse_fixture(
            runtime,
            "splitk32_final_alpha_negative_fraction",
            &[(0, 2.0)],
            &[(0, 0, 3.0)],
            &[(0, (-4.5_f32).to_bits())],
            -0.75_f32,
        )
    }

    fn kernel(kernels: &Kernels, arm: Arm) -> &Kernel {
        match arm {
            Arm::Old3Node | Arm::Production => {
                panic!("old and production routes do not use a candidate Kernel")
            }
            Arm::M8Runner => &kernels.m8n4,
        }
    }

    fn output(fixture: &Fixture, arm: Arm) -> &GuardedBuffer {
        match arm {
            Arm::Old3Node => &fixture.old_output,
            Arm::Production => &fixture.production_output,
            Arm::M8Runner => &fixture.m8n4_output,
        }
    }

    fn output_mut(fixture: &mut Fixture, arm: Arm) -> &mut GuardedBuffer {
        match arm {
            Arm::Old3Node => &mut fixture.old_output,
            Arm::Production => &mut fixture.production_output,
            Arm::M8Runner => &mut fixture.m8n4_output,
        }
    }

    fn launch_old3(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &mut Fixture,
    ) -> Result<(), String> {
        super::validate_exact_candidate_dims(fixture.dims_i32)?;
        let (m, n, k_out) = fixture.dims_i32;
        let transpose = fixture.transpose_scratch.raw_ptr(&runtime.stream);
        let partials = fixture.splitk_scratch.raw_ptr(&runtime.stream);
        let output = fixture.old_output.ptr(&runtime.stream);
        let a = fixture.a.ptr(&runtime.stream);
        let b = fixture.b.ptr(&runtime.stream);

        let rows = k_out;
        let columns = n;
        let mut transpose_builder = runtime.stream.launch_builder(&kernels.old_transpose);
        transpose_builder.arg(&transpose);
        transpose_builder.arg(&b);
        transpose_builder.arg(&rows);
        transpose_builder.arg(&columns);
        unsafe {
            transpose_builder.launch(LaunchConfig {
                grid_dim: (64, 1, 1),
                block_dim: (32, 32, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| format!("launch {OLD_TRANSPOSE_SYMBOL}: {error:?}"))?;

        let chunks = 64_i32;
        let lda = n;
        let mut partial_builder = runtime.stream.launch_builder(&kernels.old_partial);
        partial_builder.arg(&partials);
        partial_builder.arg(&a);
        partial_builder.arg(&transpose);
        partial_builder.arg(&m);
        partial_builder.arg(&k_out);
        partial_builder.arg(&chunks);
        partial_builder.arg(&lda);
        unsafe {
            partial_builder.launch(LaunchConfig {
                grid_dim: (1_024, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| format!("launch {OLD_PARTIAL_SYMBOL}: {error:?}"))?;

        let null_pointer = 0_u64;
        let zero = 0_i32;
        let mut reducer_builder = runtime.stream.launch_builder(&kernels.old_reducer);
        reducer_builder.arg(&output);
        reducer_builder.arg(&partials);
        reducer_builder.arg(&null_pointer);
        reducer_builder.arg(&null_pointer);
        reducer_builder.arg(&null_pointer);
        reducer_builder.arg(&fixture.alpha);
        reducer_builder.arg(&m);
        reducer_builder.arg(&k_out);
        reducer_builder.arg(&chunks);
        reducer_builder.arg(&n);
        reducer_builder.arg(&zero);
        reducer_builder.arg(&zero);
        unsafe {
            reducer_builder.launch(LaunchConfig {
                grid_dim: (32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| format!("launch {OLD_REDUCER_SYMBOL}: {error:?}"))
    }

    fn launch(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &mut Fixture,
        arm: Arm,
    ) -> Result<(), String> {
        if arm == Arm::Old3Node {
            return launch_old3(runtime, kernels, fixture);
        }
        if arm == Arm::Production {
            return Err("production denominator must use its qualified holder".into());
        }
        super::validate_exact_candidate_dims(fixture.dims_i32)?;
        let kernel = kernel(kernels, arm);
        let c = output(fixture, arm).ptr(&runtime.stream);
        let a = fixture.a.ptr(&runtime.stream);
        let b = fixture.b.ptr(&runtime.stream);
        let (m, n, k_out) = fixture.dims_i32;
        let mut builder = runtime.stream.launch_builder(&kernel.function);
        builder.arg(&c);
        builder.arg(&a);
        builder.arg(&b);
        builder.arg(&fixture.alpha);
        builder.arg(&m);
        builder.arg(&n);
        builder.arg(&k_out);
        unsafe { builder.launch(kernel.config) }
            .map(|_| ())
            .map_err(|error| format!("launch {}: {error:?}", kernel.symbol))
    }

    fn capture(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &mut Fixture,
        arm: Arm,
    ) -> Result<CudaGraph, String> {
        super::validate_exact_candidate_dims(fixture.dims_i32)?;
        unsafe { capture_into_graph(&runtime.stream, || launch(runtime, kernels, fixture, arm)) }
    }

    fn bytes_of<T>(value: &T) -> &[u8] {
        unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
    }

    fn digest_arguments(arguments: &[&[u8]]) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"nt-skinny-direct-arguments.v1");
        for argument in arguments {
            digest.update((argument.len() as u64).to_le_bytes());
            digest.update(argument);
        }
        digest.finalize().into()
    }

    fn eager_identity(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &Fixture,
        arm: Arm,
    ) -> PhysicalIdentity {
        let kernel = kernel(kernels, arm);
        let c = output(fixture, arm).ptr(&runtime.stream);
        let a = fixture.a.ptr(&runtime.stream);
        let b = fixture.b.ptr(&runtime.stream);
        let (m, n, k_out) = fixture.dims_i32;
        PhysicalIdentity {
            symbol: kernel.symbol.into(),
            grid: kernel.config.grid_dim,
            block: kernel.config.block_dim,
            shared: kernel.config.shared_mem_bytes,
            argument_count: 7,
            arguments_digest: digest_arguments(&[
                bytes_of(&c),
                bytes_of(&a),
                bytes_of(&b),
                bytes_of(&fixture.alpha),
                bytes_of(&m),
                bytes_of(&n),
                bytes_of(&k_out),
            ]),
        }
    }

    fn cuda_ok(result: sys::CUresult, label: &str) -> Result<(), String> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{label}: {result:?}"))
        }
    }

    unsafe fn graph_identity(graph: &CudaGraph) -> Result<PhysicalIdentity, String> {
        let raw = graph.cu_graph();
        let mut node_count = 0_usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, std::ptr::null_mut(), &mut node_count) },
            "query graph node count",
        )?;
        if node_count != 1 {
            return Err(format!(
                "skinny graph has {node_count} nodes instead of one"
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
            "query graph edge count",
        )?;
        if edge_count != 0 {
            return Err(format!("single-node skinny graph has {edge_count} edges"));
        }
        let mut node = std::ptr::null_mut();
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, &mut node, &mut node_count) },
            "query graph node",
        )?;
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        cuda_ok(
            unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
            "query graph node type",
        )?;
        if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            return Err(format!("skinny graph node is {kind:?}, not a kernel"));
        }
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            "query graph kernel parameters",
        )?;
        let mut name = std::ptr::null();
        cuda_ok(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "query graph function name",
        )?;
        if name.is_null() || params.kernelParams.is_null() {
            return Err("skinny graph omitted its function or packed arguments".into());
        }
        let symbol = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|error| format!("graph function name is not UTF-8: {error}"))?
            .to_owned();
        let sizes = [
            size_of::<u64>(),
            size_of::<u64>(),
            size_of::<u64>(),
            size_of::<f32>(),
            size_of::<i32>(),
            size_of::<i32>(),
            size_of::<i32>(),
        ];
        let mut arguments = Vec::with_capacity(sizes.len());
        for (index, size) in sizes.into_iter().enumerate() {
            let pointer = unsafe { *params.kernelParams.add(index) };
            if pointer.is_null() {
                return Err(format!("skinny graph argument {index} is null"));
            }
            arguments.push(unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), size) });
        }
        Ok(PhysicalIdentity {
            symbol,
            grid: (params.gridDimX, params.gridDimY, params.gridDimZ),
            block: (params.blockDimX, params.blockDimY, params.blockDimZ),
            shared: params.sharedMemBytes,
            argument_count: sizes.len(),
            arguments_digest: digest_arguments(&arguments),
        })
    }

    unsafe fn old_graph_identity(graph: &CudaGraph) -> Result<Vec<PhysicalIdentity>, String> {
        let raw = graph.cu_graph();
        let mut node_count = 0_usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, std::ptr::null_mut(), &mut node_count) },
            "query old graph node count",
        )?;
        if node_count != 3 {
            return Err(format!(
                "old route graph has {node_count} nodes instead of three"
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
            "query old graph edge count",
        )?;
        if edge_count != 2 {
            return Err(format!(
                "old route graph has {edge_count} edges instead of two"
            ));
        }
        let mut from = vec![std::ptr::null_mut(); edge_count];
        let mut to = vec![std::ptr::null_mut(); edge_count];
        let mut edge_data: Vec<sys::CUgraphEdgeData> = (0..edge_count)
            .map(|_| unsafe { std::mem::zeroed() })
            .collect();
        cuda_ok(
            unsafe {
                sys::cuGraphGetEdges_v2(
                    raw,
                    from.as_mut_ptr(),
                    to.as_mut_ptr(),
                    edge_data.as_mut_ptr(),
                    &mut edge_count,
                )
            },
            "query old graph edges",
        )?;
        if edge_data.iter().any(|edge| {
            edge.from_port != 0
                || edge.to_port != 0
                || edge.type_ != sys::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT as u8
                || edge.reserved != [0; 5]
        }) {
            return Err("old route graph edge descriptor changed".into());
        }
        let heads = from
            .iter()
            .copied()
            .filter(|node| !to.contains(node))
            .collect::<Vec<_>>();
        if heads.len() != 1 {
            return Err(format!("old route graph has {} chain heads", heads.len()));
        }
        let mut ordered = vec![heads[0]];
        for _ in 0..2 {
            let next = from
                .iter()
                .zip(&to)
                .filter_map(|(source, destination)| {
                    (*source == *ordered.last().unwrap()).then_some(*destination)
                })
                .collect::<Vec<_>>();
            if next.len() != 1 || ordered.contains(&next[0]) {
                return Err("old route graph is not one ordered chain".into());
            }
            ordered.push(next[0]);
        }

        let expected = [
            (OLD_TRANSPOSE_SYMBOL, (64, 1, 1), (32, 32, 1), 4_usize),
            (OLD_PARTIAL_SYMBOL, (1_024, 1, 1), (128, 1, 1), 7),
            (OLD_REDUCER_SYMBOL, (32, 1, 1), (256, 1, 1), 12),
        ];
        let mut identities = Vec::with_capacity(3);
        for (node, (expected_symbol, grid, block, argument_count)) in
            ordered.into_iter().zip(expected)
        {
            let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            cuda_ok(
                unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
                "query old graph node type",
            )?;
            if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
                return Err(format!("old route graph contains {kind:?}"));
            }
            let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
            cuda_ok(
                unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
                "query old graph kernel parameters",
            )?;
            let mut name = std::ptr::null();
            cuda_ok(
                unsafe { sys::cuFuncGetName(&mut name, params.func) },
                "query old graph function name",
            )?;
            if name.is_null() || params.kernelParams.is_null() {
                return Err("old route graph omitted function or packed arguments".into());
            }
            let symbol = unsafe { CStr::from_ptr(name) }
                .to_str()
                .map_err(|error| format!("old graph function name is not UTF-8: {error}"))?;
            if symbol != expected_symbol
                || (params.gridDimX, params.gridDimY, params.gridDimZ) != grid
                || (params.blockDimX, params.blockDimY, params.blockDimZ) != block
                || params.sharedMemBytes != 0
            {
                return Err(format!("old route graph node drifted at {expected_symbol}"));
            }
            let argument_sizes: &[usize] = match expected_symbol {
                OLD_TRANSPOSE_SYMBOL => &[8, 8, 4, 4],
                OLD_PARTIAL_SYMBOL => &[8, 8, 8, 4, 4, 4, 4],
                OLD_REDUCER_SYMBOL => &[8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4],
                _ => unreachable!(),
            };
            let mut arguments = Vec::with_capacity(argument_sizes.len());
            for (index, size) in argument_sizes.iter().copied().enumerate() {
                let pointer = unsafe { *params.kernelParams.add(index) };
                if pointer.is_null() {
                    return Err(format!("{expected_symbol} argument {index} is null"));
                }
                arguments.push(unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), size) });
            }
            let identity = PhysicalIdentity {
                symbol: symbol.to_owned(),
                grid,
                block,
                shared: 0,
                argument_count,
                arguments_digest: digest_arguments(&arguments),
            };
            if identity.arguments_digest == [0; 32] {
                return Err(format!("{expected_symbol} arguments digest is zero"));
            }
            identities.push(identity);
        }
        let distinct = identities
            .iter()
            .map(|identity| identity.arguments_digest)
            .collect::<std::collections::BTreeSet<_>>();
        if distinct.len() != identities.len() {
            return Err("old route graph argument digests collide".into());
        }
        Ok(identities)
    }

    fn execute_and_read(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &mut Fixture,
        arm: Arm,
        path: Path,
        graph: &CudaGraph,
    ) -> Result<Vec<u32>, String> {
        output_mut(fixture, arm).reset(&runtime.stream)?;
        match path {
            Path::Eager => launch(runtime, kernels, fixture, arm)?,
            Path::Graph => graph
                .launch()
                .map_err(|error| format!("launch {arm:?} graph: {error:?}"))?,
        }
        runtime
            .stream
            .synchronize()
            .map_err(|error| format!("synchronize {arm:?} {path:?}: {error:?}"))?;
        fixture.a.validate_unchanged(&runtime.stream, "A")?;
        fixture.b.validate_unchanged(&runtime.stream, "B")?;
        let label = match arm {
            Arm::Old3Node => "old three-node output",
            Arm::Production => "production output",
            _ => kernel(kernels, arm).symbol,
        };
        output(fixture, arm).active_bits(&runtime.stream, label)
    }

    fn correctness_gate(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &mut Fixture,
        case: Case,
    ) -> Result<(), String> {
        for arm in [Arm::M8Runner] {
            let graph = capture(runtime, kernels, fixture, arm)?;
            let eager_physical = eager_identity(runtime, kernels, fixture, arm);
            let graph_physical = unsafe { graph_identity(&graph) }?;
            validate_physical_identity(&graph_physical, &eager_physical)?;
            let mut repeated = None;
            for repeat in 0..CORRECTNESS_REPEATS {
                let eager = execute_and_read(runtime, kernels, fixture, arm, Path::Eager, &graph)?;
                if eager != fixture.oracle {
                    let index = eager
                        .iter()
                        .zip(&fixture.oracle)
                        .position(|(actual, expected)| actual != expected)
                        .unwrap_or(0);
                    return Err(format!(
                        "{} {arm:?} CPU sequential bits differ at element {index}: actual=0x{:08x} expected=0x{:08x}",
                        case.id, eager[index], fixture.oracle[index]
                    ));
                }
                let graph_bits =
                    execute_and_read(runtime, kernels, fixture, arm, Path::Graph, &graph)?;
                if graph_bits != eager {
                    return Err(format!(
                        "{} {arm:?} graph bits differ from eager at repeat {repeat}",
                        case.id
                    ));
                }
                if repeated.as_ref().is_some_and(|expected| expected != &eager) {
                    return Err(format!("{} {arm:?} repeated bits changed", case.id));
                }
                repeated.get_or_insert(eager);
            }
            runtime
                .stream
                .synchronize()
                .map_err(|error| format!("synchronize before graph drop: {error:?}"))?;
            drop(graph);
        }
        fixture.a.validate_unchanged(&runtime.stream, "A")?;
        fixture.b.validate_unchanged(&runtime.stream, "B")?;
        Ok(())
    }

    fn old3_correctness_gate(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &mut Fixture,
        case: Case,
    ) -> Result<Vec<u32>, String> {
        let graph = capture(runtime, kernels, fixture, Arm::Old3Node)?;
        unsafe { old_graph_identity(&graph) }?;
        let mut expected = None;
        for repeat in 0..CORRECTNESS_REPEATS {
            let eager = execute_and_read(
                runtime,
                kernels,
                fixture,
                Arm::Old3Node,
                Path::Eager,
                &graph,
            )?;
            if eager != fixture.oracle {
                return Err(format!(
                    "{} old three-node host-oracle bits differ",
                    case.id
                ));
            }
            let graph_bits = execute_and_read(
                runtime,
                kernels,
                fixture,
                Arm::Old3Node,
                Path::Graph,
                &graph,
            )?;
            if graph_bits != eager {
                return Err(format!(
                    "{} old three-node graph bits differ at repeat {repeat}",
                    case.id
                ));
            }
            if expected.as_ref().is_some_and(|expected| expected != &eager) {
                return Err(format!("{} old three-node repeated bits changed", case.id));
            }
            expected.get_or_insert(eager);
        }
        runtime
            .stream
            .synchronize()
            .map_err(|error| format!("synchronize before old graph drop: {error:?}"))?;
        drop(graph);
        fixture
            .a
            .validate_unchanged(&runtime.stream, "old three-node A")?;
        fixture
            .b
            .validate_unchanged(&runtime.stream, "old three-node B")?;
        Ok(expected.expect("old route executes at least once"))
    }

    fn launch_raw_production(
        runtime: &Runtime,
        fixture: &mut Fixture,
        case: Case,
    ) -> Result<(), String> {
        let b = fixture.production_b.ptr(&runtime.production.stream);
        gpu_gemm_bi_backward_dx_raw(
            &runtime.production,
            &mut fixture.production_output.buffer,
            &fixture.production_a.buffer,
            b,
            case.dims.0,
            case.dims.1,
            case.dims.2,
        )
    }

    fn production_snapshot(runtime: &Runtime, fixture: &Fixture) -> Result<Vec<u32>, String> {
        fixture
            .production_a
            .validate_unchanged(&runtime.production.stream, "production A")?;
        fixture
            .production_b
            .validate_unchanged(&runtime.production.stream, "production B")?;
        fixture
            .production_output
            .active_bits(&runtime.production.stream, "production C")
    }

    fn capture_raw_production_graph(
        runtime: &Runtime,
        fixture: &mut Fixture,
        case: Case,
    ) -> Result<CudaGraph, String> {
        unsafe {
            capture_into_graph(&runtime.production.stream, || {
                launch_raw_production(runtime, fixture, case)
            })
        }
    }

    fn production_correctness_gate(
        runtime: &Runtime,
        fixture: &mut Fixture,
        case: Case,
        compare_host_oracle: bool,
    ) -> Result<Vec<u32>, String> {
        if fixture.production_a.active_offset != 0 || fixture.production_output.active_offset != 0 {
            return Err("production active pointers must remain at allocation offset zero".into());
        }
        let mut expected = None;
        for _ in 0..3 {
            fixture
                .production_output
                .reset(&runtime.production.stream)?;
            launch_raw_production(runtime, fixture, case)?;
            runtime
                .production
                .stream
                .synchronize()
                .map_err(|error| format!("synchronize raw production eager: {error:?}"))?;
            let bits = production_snapshot(runtime, fixture)?;
            if expected.as_ref().is_some_and(|expected| expected != &bits) {
                return Err("raw production eager bits changed".into());
            }
            expected.get_or_insert(bits);
        }
        let graph = capture_raw_production_graph(runtime, fixture, case)?;
        for _ in 0..3 {
            fixture
                .production_output
                .reset(&runtime.production.stream)?;
            graph
                .launch()
                .map_err(|error| format!("launch raw production graph: {error:?}"))?;
            runtime
                .production
                .stream
                .synchronize()
                .map_err(|error| format!("synchronize raw production graph: {error:?}"))?;
            if production_snapshot(runtime, fixture)? != *expected.as_ref().unwrap() {
                return Err("raw production graph bits changed".into());
            }
        }
        let actual = expected.unwrap();
        if compare_host_oracle && actual != fixture.oracle {
            return Err(format!("{} public production bits differ", case.id));
        }
        Ok(actual)
    }

    fn post_promotion_correctness_gate(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &mut Fixture,
        case: Case,
    ) -> Result<(), String> {
        let production = production_correctness_gate(runtime, fixture, case, true)?;
        let old = old3_correctness_gate(runtime, kernels, fixture, case)?;
        if old != production {
            return Err(format!(
                "{} old three-node bits differ from production",
                case.id
            ));
        }
        correctness_gate(runtime, kernels, fixture, case)
    }

    fn resource_gate(kernels: &Kernels) -> Result<(), String> {
        let mut observations = Vec::new();
        {
            let arm = Arm::M8Runner;
            let kernel = kernel(kernels, arm);
            let local = kernel
                .function
                .local_size_bytes()
                .map_err(|error| format!("query {} local bytes: {error:?}", kernel.symbol))?
                as usize;
            let registers = kernel
                .function
                .num_regs()
                .map_err(|error| format!("query {} registers: {error:?}", kernel.symbol))?
                as usize;
            let static_shared = kernel
                .function
                .shared_size_bytes()
                .map_err(|error| format!("query {} static shared: {error:?}", kernel.symbol))?
                as usize;
            let occupancy = kernel
                .function
                .occupancy_max_active_blocks_per_multiprocessor(
                    kernel.config.block_dim.0,
                    kernel.resources.dynamic_shared,
                    None,
                )
                .map_err(|error| format!("query {} occupancy: {error:?}", kernel.symbol))?;
            eprintln!(
                "NT skinny {arm:?} symbol={} registers={registers} local_bytes={local} static_shared_bytes={static_shared} dynamic_shared_bytes={} active_blocks={occupancy}",
                kernel.symbol, kernel.resources.dynamic_shared
            );
            observations.push((
                arm,
                ResourceObservation {
                    threads: kernel.config.block_dim.0,
                    static_shared,
                    dynamic_shared: kernel.config.shared_mem_bytes as usize,
                    registers,
                    local_bytes: local,
                    active_blocks: occupancy,
                },
                kernel.resources,
            ));
        }
        let mut failures = Vec::new();
        for (arm, observation, contract) in observations {
            if let Err(error) = validate_resources(observation, contract) {
                failures.push(format!("{arm:?}: {error}"));
            }
        }
        if !failures.is_empty() {
            return Err(format!(
                "candidate resource census failed: {}",
                failures.join("; ")
            ));
        }
        Ok(())
    }

    fn production_request() -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Nt,
            THIN_COLS.dims,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
            PhysicalQualificationF32Epilogue::new(1.0, 0.0, false),
        )
    }

    fn production_holder(runtime: &Runtime) -> Result<QualifiedPhysicalLaunch<'_>, String> {
        let mut holder = qualify_physical_launch(&runtime.production, production_request())?;
        holder.seed_f32_operands(&runtime.production, 0x0051_1732_u64)?;
        let nodes = holder.evidence().nodes();
        let evidence = holder.evidence();
        let production_config = LaunchConfig {
            grid_dim: (256, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 17_984,
        };
        let symbols = nodes.iter().map(|node| node.symbol).collect::<Vec<_>>();
        let launches = nodes
            .iter()
            .map(|node| (node.launch.grid_dim, node.launch.block_dim, node.strides))
            .collect::<Vec<_>>();
        let argument_digests = nodes
            .iter()
            .map(|node| node.launch.arguments_digest)
            .collect::<std::collections::BTreeSet<_>>();
        if nodes.len() != 1
            || symbols != [PRODUCTION_SYMBOL]
            || launches != [((256, 1, 1), (64, 1, 1), (2_048, 2_048, 16))]
            || nodes[0].launch.grid_dim != production_config.grid_dim
            || nodes[0].launch.block_dim != production_config.block_dim
            || nodes[0].launch.shared_mem_bytes != production_config.shared_mem_bytes
            || nodes
                .iter()
                .any(|node| node.launch.arguments_digest == [0; 32])
            || argument_digests.len() != nodes.len()
            || evidence.evidence_scope() != "eager_preflight_same_launcher"
            || !evidence.eager_graph_equal()
            || evidence.launch_digest() == [0; 32]
            || nodes.iter().any(|node| {
                format!("{:?}", node.module_kind) != "TriadScalar"
                    || format!("{:?}", node.logical_dtype) != "F32"
                    || format!("{:?}", node.execution_dtype) != "F32"
                    || node.launch.shared_mem_bytes != 17_984
            })
        {
            return Err(format!(
                "actual production denominator drifted: symbols={symbols:?} launches={launches:?}"
            ));
        }
        Ok(holder)
    }

    fn production_eager3_graph3(runtime: &Runtime) -> Result<(), String> {
        let mut holder = production_holder(runtime)?;
        holder.measure_eager_window_ms(&runtime.production, 1)?;
        let expected = holder.f32_output_bits(&runtime.production)?;
        for _ in 0..3 {
            holder.measure_eager_window_ms(&runtime.production, 1)?;
            if holder.f32_output_bits(&runtime.production)? != expected {
                return Err("production eager bits changed".into());
            }
        }
        for _ in 0..3 {
            holder.measure_graph_window_ms(&runtime.production, 1)?;
            if holder.f32_output_bits(&runtime.production)? != expected {
                return Err("production graph bits changed".into());
            }
        }
        Ok(())
    }

    fn measure(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &mut Fixture,
        arm: Arm,
        iterations: usize,
    ) -> Result<f64, String> {
        let start = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record start: {error:?}"))?;
        for _ in 0..iterations {
            launch(runtime, kernels, fixture, arm)?;
        }
        let end = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record end: {error:?}"))?;
        let value = start
            .elapsed_ms(&end)
            .map(|milliseconds| f64::from(milliseconds) * 1_000.0 / iterations as f64)
            .map_err(|error| format!("measure {arm:?}: {error:?}"))?;
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("measure {arm:?} returned invalid {value}"));
        }
        Ok(value)
    }

    struct PairLaunches<'a> {
        production_holder: Option<QualifiedPhysicalLaunch<'a>>,
        baseline_graph: Option<CudaGraph>,
        candidate_graph: Option<CudaGraph>,
        baseline: Arm,
        candidate: Arm,
    }

    fn candidate_graphs<'a>(
        runtime: &'a Runtime,
        baseline_graph: Option<CudaGraph>,
        candidate_graph: Option<CudaGraph>,
        baseline: Arm,
        candidate: Arm,
        needs_production: bool,
    ) -> Result<PairLaunches<'a>, String> {
        Ok(PairLaunches {
            production_holder: needs_production
                .then(|| production_holder(runtime))
                .transpose()?,
            baseline_graph,
            candidate_graph,
            baseline,
            candidate,
        })
    }

    fn graph_for<'a>(launches: &'a PairLaunches<'_>, arm: Arm) -> Result<&'a CudaGraph, String> {
        if arm == launches.baseline {
            launches
                .baseline_graph
                .as_ref()
                .ok_or_else(|| format!("missing {arm:?} baseline graph"))
        } else if arm == launches.candidate {
            launches
                .candidate_graph
                .as_ref()
                .ok_or_else(|| format!("missing {arm:?} candidate graph"))
        } else {
            Err(format!("{arm:?} is outside the active pair"))
        }
    }

    fn measure_arm(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &mut Fixture,
        launches: &mut PairLaunches<'_>,
        arm: Arm,
        path: Path,
        iterations: usize,
    ) -> Result<f64, String> {
        if arm == Arm::Production {
            let holder = launches
                .production_holder
                .as_mut()
                .ok_or("production holder is absent from production pair")?;
            let elapsed = match path {
                Path::Eager => holder.measure_eager_window_ms(&runtime.production, iterations)?,
                Path::Graph => holder.measure_graph_window_ms(&runtime.production, iterations)?,
            };
            let value = elapsed * 1_000.0 / iterations as f64;
            return (value.is_finite() && value > 0.0)
                .then_some(value)
                .ok_or_else(|| format!("production measure returned invalid {value}"));
        }
        match path {
            Path::Eager => measure(runtime, kernels, fixture, arm, iterations),
            Path::Graph => measure_candidate_graph(runtime, graph_for(launches, arm)?, iterations),
        }
    }

    fn measure_candidate_graph(
        runtime: &Runtime,
        graph: &CudaGraph,
        iterations: usize,
    ) -> Result<f64, String> {
        let start = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record candidate graph start: {error:?}"))?;
        for _ in 0..iterations {
            graph
                .launch()
                .map_err(|error| format!("launch candidate graph: {error:?}"))?;
        }
        let end = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record candidate graph end: {error:?}"))?;
        start
            .elapsed_ms(&end)
            .map(|ms| f64::from(ms) * 1_000.0 / iterations as f64)
            .map_err(|error| format!("measure candidate graph: {error:?}"))
    }

    #[derive(Clone, Copy)]
    struct Thresholds {
        p05: f64,
        p50: f64,
    }

    #[derive(Clone, Copy)]
    struct PathThresholds {
        eager: Thresholds,
        graph: Thresholds,
    }

    impl PathThresholds {
        fn gated(eager_p05: f64, eager_p50: f64) -> Self {
            Self {
                eager: Thresholds {
                    p05: eager_p05,
                    p50: eager_p50,
                },
                graph: Thresholds {
                    p05: 0.9995,
                    p50: 1.0,
                },
            }
        }
    }

    #[derive(Clone, Copy)]
    struct PairSpec {
        baseline: Arm,
        candidate: Arm,
        windows: usize,
        thresholds: PathThresholds,
    }

    fn paired_performance(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &mut Fixture,
        case: Case,
        pair: PairSpec,
        quiet: &QuietGpu,
    ) -> Result<(), String> {
        let PairSpec {
            baseline,
            candidate,
            windows,
            thresholds,
        } = pair;
        if !matches!(windows, SCREEN_WINDOWS | OFFICIAL_WINDOWS) {
            return Err(format!("unsupported paired window count {windows}"));
        }
        let candidate_graph = (candidate != Arm::Production)
            .then(|| capture(runtime, kernels, fixture, candidate))
            .transpose()?;
        let baseline_graph = (baseline != Arm::Production)
            .then(|| capture(runtime, kernels, fixture, baseline))
            .transpose()?;
        let mut launches = candidate_graphs(
            runtime,
            baseline_graph,
            candidate_graph,
            baseline,
            candidate,
            baseline == Arm::Production || candidate == Arm::Production,
        )?;
        for path in [Path::Eager, Path::Graph] {
            let threshold = match path {
                Path::Eager => thresholds.eager,
                Path::Graph => thresholds.graph,
            };
            let label = format!(
                "nt-skinny/{}/{baseline:?}-to-{candidate:?}/{path:?}/{windows}",
                case.id
            );
            quiet.require_cohort(&label)?;
            let result = (|| {
                for _ in 0..64 {
                    measure_arm(runtime, kernels, fixture, &mut launches, baseline, path, 1)?;
                    measure_arm(runtime, kernels, fixture, &mut launches, candidate, path, 1)?;
                }
                runtime
                    .stream
                    .synchronize()
                    .map_err(|error| format!("synchronize warmup: {error:?}"))?;
                let baseline_probe =
                    measure_arm(runtime, kernels, fixture, &mut launches, baseline, path, 4)?;
                let candidate_probe =
                    measure_arm(runtime, kernels, fixture, &mut launches, candidate, path, 4)?;
                let iterations = (TARGET_WINDOW_US / baseline_probe.min(candidate_probe)).ceil();
                if !iterations.is_finite() || iterations <= 0.0 {
                    return Err(format!("invalid calibrated iteration count {iterations}"));
                }
                let iterations = (iterations as usize).clamp(1, 4_096);
                let mut abba_baseline = Vec::with_capacity(windows);
                let mut abba_candidate = Vec::with_capacity(windows);
                let mut abba_speedup = Vec::with_capacity(windows);
                let mut baab_baseline = Vec::with_capacity(windows);
                let mut baab_candidate = Vec::with_capacity(windows);
                let mut baab_speedup = Vec::with_capacity(windows);
                for _ in 0..windows {
                    let b0 = measure_arm(
                        runtime,
                        kernels,
                        fixture,
                        &mut launches,
                        baseline,
                        path,
                        iterations,
                    )?;
                    let c0 = measure_arm(
                        runtime,
                        kernels,
                        fixture,
                        &mut launches,
                        candidate,
                        path,
                        iterations,
                    )?;
                    let c1 = measure_arm(
                        runtime,
                        kernels,
                        fixture,
                        &mut launches,
                        candidate,
                        path,
                        iterations,
                    )?;
                    let b1 = measure_arm(
                        runtime,
                        kernels,
                        fixture,
                        &mut launches,
                        baseline,
                        path,
                        iterations,
                    )?;
                    let baseline_us = (b0 + b1) * 0.5;
                    let candidate_us = (c0 + c1) * 0.5;
                    abba_baseline.push(baseline_us);
                    abba_candidate.push(candidate_us);
                    abba_speedup.push(baseline_us / candidate_us);

                    let c0 = measure_arm(
                        runtime,
                        kernels,
                        fixture,
                        &mut launches,
                        candidate,
                        path,
                        iterations,
                    )?;
                    let b0 = measure_arm(
                        runtime,
                        kernels,
                        fixture,
                        &mut launches,
                        baseline,
                        path,
                        iterations,
                    )?;
                    let b1 = measure_arm(
                        runtime,
                        kernels,
                        fixture,
                        &mut launches,
                        baseline,
                        path,
                        iterations,
                    )?;
                    let c1 = measure_arm(
                        runtime,
                        kernels,
                        fixture,
                        &mut launches,
                        candidate,
                        path,
                        iterations,
                    )?;
                    let baseline_us = (b0 + b1) * 0.5;
                    let candidate_us = (c0 + c1) * 0.5;
                    baab_baseline.push(baseline_us);
                    baab_candidate.push(candidate_us);
                    baab_speedup.push(baseline_us / candidate_us);
                }
                runtime
                    .stream
                    .synchronize()
                    .map_err(|error| format!("synchronize timing: {error:?}"))?;

                let abba_p05 = checked_percentile(&abba_speedup, 0.05)?;
                let abba_p50 = checked_percentile(&abba_speedup, 0.50)?;
                let abba_p95 = checked_percentile(&abba_speedup, 0.95)?;
                let baab_p05 = checked_percentile(&baab_speedup, 0.05)?;
                let baab_p50 = checked_percentile(&baab_speedup, 0.50)?;
                let baab_p95 = checked_percentile(&baab_speedup, 0.95)?;
                if threshold.p05 > 0.0 || threshold.p50 > 0.0 {
                    validate_speedup(
                        windows,
                        abba_p05,
                        abba_p50,
                        abba_p95,
                        threshold.p05,
                        threshold.p50,
                    )?;
                    validate_speedup(
                        windows,
                        baab_p05,
                        baab_p50,
                        baab_p95,
                        threshold.p05,
                        threshold.p50,
                    )?;
                }
                let abba_baseline_p50 = checked_percentile(&abba_baseline, 0.50)?;
                let abba_candidate_p50 = checked_percentile(&abba_candidate, 0.50)?;
                let baab_baseline_p50 = checked_percentile(&baab_baseline, 0.50)?;
                let baab_candidate_p50 = checked_percentile(&baab_candidate, 0.50)?;
                eprintln!(
                    concat!(
                        "NT skinny {} baseline={:?} candidate={:?} windows={} iterations={} ",
                        "ABBA baseline_us={:.6} candidate_us={:.6} speedup_p05={:.9} speedup_p50={:.9} speedup_p95={:.9} ",
                        "BAAB baseline_us={:.6} candidate_us={:.6} speedup_p05={:.9} speedup_p50={:.9} speedup_p95={:.9}"
                    ),
                    case.id,
                    baseline,
                    candidate,
                    windows,
                    iterations,
                    abba_baseline_p50,
                    abba_candidate_p50,
                    abba_p05,
                    abba_p50,
                    abba_p95,
                    baab_baseline_p50,
                    baab_candidate_p50,
                    baab_p05,
                    baab_p50,
                    baab_p95,
                );
                Ok(())
            })();
            let postflight = quiet.verify_post_cohort(&label);
            verify_post_cohort_even_on_error(result, postflight)?;
        }
        Ok(())
    }

    fn require_sm120(runtime: &Runtime) -> Result<(), String> {
        let identity = runtime.device.identity();
        if identity.compute_capability != (12, 0) || identity.multiprocessor_count != 170 {
            return Err(format!(
                "timing requires CC 12.0 with 170 SMs, found CC {}.{} with {} SMs",
                identity.compute_capability.0,
                identity.compute_capability.1,
                identity.multiprocessor_count
            ));
        }
        let mut major = 0;
        let mut minor = 0;
        let status = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
        if status != cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS || (major, minor) != (13, 2) {
            return Err(format!(
                "timing requires NVRTC 13.2, found status={status:?} version={major}.{minor}"
            ));
        }
        if runtime.device.nvrtc_target() != "compute_120" {
            return Err(format!(
                "timing requires compute_120, found {}",
                runtime.device.nvrtc_target()
            ));
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive quiet SM80+ GPU"]
    fn skinny_direct_correctness_resources_graph_and_redzones() -> Result<(), String> {
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("nt-skinny-correctness-pre-context")?;
        let runtime = new_runtime()?;
        quiet.require_cohort("nt-skinny-correctness")?;
        let body = (|| {
            let case = THIN_COLS;
            production_eager3_graph3(&runtime)?;
            let kernels = load_kernels(&runtime, case)?;
            resource_gate(&kernels)?;
            let mut fixture = new_fixture(&runtime, case, false)?;
            post_promotion_correctness_gate(&runtime, &kernels, &mut fixture, case)?;
            for (case, mut fixture) in tree_killer_fixtures(&runtime)? {
                post_promotion_correctness_gate(&runtime, &kernels, &mut fixture, case)?;
            }
            for (case, mut fixture) in exceptional_fixtures(&runtime)? {
                post_promotion_correctness_gate(&runtime, &kernels, &mut fixture, case)?;
            }
            let (case, mut fixture) = alpha_fixture(&runtime)?;
            correctness_gate(&runtime, &kernels, &mut fixture, case)
        })();
        verify_post_cohort_even_on_error(body, quiet.verify_post_cohort("nt-skinny-correctness"))
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC12.0/170-SM CUDA13.2 GPU and exactly 21 windows"]
    fn skinny_direct_screening_21_windows() -> Result<(), String> {
        let windows = exact_windows(
            std::env::var_os("MAMBA_RS_NT_SKINNY_WINDOWS").as_deref(),
            SCREEN_WINDOWS,
        )?;
        if windows != SCREEN_WINDOWS {
            return Err("screening window contract changed".into());
        }
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("nt-skinny-screen-pre-context")?;
        let runtime = new_runtime()?;
        require_sm120(&runtime)?;
        let case = THIN_COLS;
        let kernels = load_kernels(&runtime, case)?;
        resource_gate(&kernels)?;
        let mut fixture = new_fixture(&runtime, case, false)?;
        paired_performance(
            &runtime,
            &kernels,
            &mut fixture,
            case,
            PairSpec {
                baseline: Arm::Old3Node,
                candidate: Arm::Production,
                windows: SCREEN_WINDOWS,
                thresholds: PathThresholds::gated(MIN_PRODUCTION_P05, MIN_PRODUCTION_P50),
            },
            &quiet,
        )?;
        paired_performance(
            &runtime,
            &kernels,
            &mut fixture,
            case,
            PairSpec {
                baseline: Arm::M8Runner,
                candidate: Arm::Production,
                windows: SCREEN_WINDOWS,
                thresholds: PathThresholds::gated(MIN_RUNNER_P05, MIN_RUNNER_P50),
            },
            &quiet,
        )?;
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC12.0/170-SM CUDA13.2 GPU and exactly 101 windows"]
    fn skinny_direct_official_101_windows() -> Result<(), String> {
        let windows = exact_windows(
            std::env::var_os("MAMBA_RS_NT_SKINNY_WINDOWS").as_deref(),
            OFFICIAL_WINDOWS,
        )?;
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("nt-skinny-official-pre-context")?;
        let runtime = new_runtime()?;
        require_sm120(&runtime)?;
        let case = THIN_COLS;
        let kernels = load_kernels(&runtime, case)?;
        resource_gate(&kernels)?;
        let mut fixture = new_fixture(&runtime, case, false)?;
        paired_performance(
            &runtime,
            &kernels,
            &mut fixture,
            case,
            PairSpec {
                baseline: Arm::Old3Node,
                candidate: Arm::Production,
                windows,
                thresholds: PathThresholds::gated(MIN_PRODUCTION_P05, MIN_PRODUCTION_P50),
            },
            &quiet,
        )?;
        paired_performance(
            &runtime,
            &kernels,
            &mut fixture,
            case,
            PairSpec {
                baseline: Arm::M8Runner,
                candidate: Arm::Production,
                windows,
                thresholds: PathThresholds::gated(MIN_RUNNER_P05, MIN_RUNNER_P50),
            },
            &quiet,
        )?;
        Ok(())
    }
}
