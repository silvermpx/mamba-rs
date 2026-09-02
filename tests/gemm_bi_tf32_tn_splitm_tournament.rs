//! Test-only tournament: SM120 deterministic TF32 TN routes that split the
//! reduction into a fixed number of partitions and combine them in a fixed
//! order.
//!
//! The production TN route owns one output tile per CTA and leaves most of a
//! 170-SM part idle on the small-output training cells. A split family fills
//! the machine at the price of a different, still deterministic, summation
//! tree, so the harness does not compare bits against production. It demands
//! the exact integer oracle, three identical eager repeats, three identical
//! graph replays that equal the eager bits, intact red zones and untouched
//! inputs, and an f64 reference distance no worse than a small multiple of the
//! production route's own distance on every cell. Nothing here registers a
//! production route.

#[cfg(feature = "cuda")]
mod common;

const CUDA_SOURCE: &str = include_str!("gemm_bi_tf32_tn_splitm_tournament.cu");
const PRODUCTION_SYMBOL: &str = "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair";
const PRODUCTION_TUNING_TABLE_REVISION: u16 = 36;
const SCREENING_WINDOWS: usize = 21;
const OFFICIAL_WINDOWS: usize = 101;
const SCREENING_MIN_P05: f64 = 1.02;
const SCREENING_MIN_P50: f64 = 1.05;
const WINNER_MIN_P05: f64 = 1.00;
const WINNER_MIN_P50: f64 = 1.01;
const GUARD_ELEMENTS: usize = 32;
const GUARD_BITS: u32 = 0x7fc1_4e58;
const REFERENCE_ERROR_FACTOR: f64 = 4.0;
const REFERENCE_ERROR_FLOOR: f64 = 1.0 / 1_048_576.0;

/// A TN cell: `m` is the reduction, the output is `[k][n]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    label: &'static str,
    dims: (usize, usize, usize),
}

const CELLS: [Cell; 3] = [
    Cell {
        label: "tn_d768_in_proj",
        dims: (2_048, 768, 3_072),
    },
    Cell {
        label: "tn_d768_out_proj",
        dims: (2_048, 1_536, 768),
    },
    Cell {
        label: "tn_prism_in_proj",
        dims: (4_621, 384, 1_928),
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CandidateSpec {
    symbol: &'static str,
    tile: (u32, u32),
    stages: u32,
    partitions: u32,
    block: (u32, u32, u32),
    dynamic_shared_bytes: u32,
    minimum_occupancy: u32,
    maximum_registers: u32,
}

const fn dynamic_shared_bytes(tile: (u32, u32), stages: u32) -> u32 {
    128 + stages * (tile.0 + tile.1) * 32 * 4
}

const fn spec(
    symbol: &'static str,
    tile: (u32, u32),
    stages: u32,
    partitions: u32,
    minimum_occupancy: u32,
) -> CandidateSpec {
    CandidateSpec {
        symbol,
        tile,
        stages,
        partitions,
        block: (tile.0 * tile.1 / 32, 1, 1),
        dynamic_shared_bytes: dynamic_shared_bytes(tile, stages),
        minimum_occupancy,
        maximum_registers: 128,
    }
}

const CANDIDATES: [CandidateSpec; 4] = [
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair_exp_splitm7_v1",
        (64, 128),
        4,
        7,
        1,
    ),
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2_pair_exp_splitm7_v1",
        (64, 128),
        2,
        7,
        2,
    ),
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair_exp_splitm5_v1",
        (64, 128),
        4,
        5,
        1,
    ),
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair_exp_splitm4_v1",
        (64, 128),
        4,
        4,
        1,
    ),
];

const fn production_spec() -> CandidateSpec {
    spec(PRODUCTION_SYMBOL, (64, 128), 4, 1, 1)
}

fn tn_tiles(dims: (usize, usize, usize), tile: (u32, u32)) -> Result<u32, String> {
    let (_, k, n) = dims;
    let rows = k.div_ceil(tile.0 as usize);
    let columns = n.div_ceil(tile.1 as usize);
    let cells = rows
        .checked_mul(columns)
        .ok_or_else(|| format!("TN grid overflow for {dims:?}"))?;
    u32::try_from(cells).map_err(|_| format!("TN grid exceeds u32 for {dims:?}"))
}

/// Fraction of the launch's last wave that does work, at one CTA slot per
/// SM per resident block. This is the quantity the split factor buys.
fn wave_efficiency(ctas: u32, per_sm: u32, sms: u32) -> f64 {
    let slots = f64::from(per_sm * sms);
    let waves = (f64::from(ctas) / slots).ceil();
    f64::from(ctas) / (waves * slots)
}

fn validate_spec(spec: CandidateSpec) -> Result<(), String> {
    let (rows, columns) = spec.tile;
    if !rows.is_multiple_of(32) || !columns.is_multiple_of(32) || rows == 0 || columns == 0 {
        return Err(format!("{} tile is not warp aligned", spec.symbol));
    }
    if !(2..=4).contains(&spec.stages) {
        return Err(format!("{} stage count is outside 2..=4", spec.symbol));
    }
    if spec.partitions == 0 || spec.partitions > 8 {
        return Err(format!("{} split factor is outside 1..=8", spec.symbol));
    }
    if spec.block != (rows * columns / 32, 1, 1) {
        return Err(format!(
            "{} block does not match its warp tiling",
            spec.symbol
        ));
    }
    if spec.dynamic_shared_bytes != dynamic_shared_bytes(spec.tile, spec.stages) {
        return Err(format!(
            "{} shared bytes drifted from the formula",
            spec.symbol
        ));
    }
    if spec.dynamic_shared_bytes > 101_376 {
        return Err(format!(
            "{} exceeds the SM120 opt-in shared limit",
            spec.symbol
        ));
    }
    if spec.minimum_occupancy == 0 || spec.maximum_registers == 0 {
        return Err(format!("{} has no resource floor", spec.symbol));
    }
    Ok(())
}

fn validate_candidate_symbol(spec: CandidateSpec) -> Result<(), String> {
    let suffix = format!("_exp_splitm{}_v1", spec.partitions);
    if !spec.symbol.starts_with("gemm_bi_tn_sm120_tma_mma_tf32_v1_")
        || !spec.symbol.ends_with(&suffix)
    {
        return Err(format!(
            "{} is not a test-only TN split candidate",
            spec.symbol
        ));
    }
    let definition = format!(
        "{}, {}, {}, {}, {})",
        spec.symbol, spec.tile.0, spec.tile.1, spec.stages, spec.partitions
    );
    if !CUDA_SOURCE.contains(&definition) {
        return Err(format!(
            "{} is not defined with its geometry by the tournament source",
            spec.symbol
        ));
    }
    Ok(())
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

fn screening_passes(p05: f64, p50: f64) -> bool {
    p05.is_finite() && p50.is_finite() && p05 >= SCREENING_MIN_P05 && p50 >= SCREENING_MIN_P50
}

fn validate_winner_gate(p05: f64, p50: f64) -> Result<(), String> {
    if !p05.is_finite() || !p50.is_finite() || p05 < WINNER_MIN_P05 || p50 < WINNER_MIN_P50 {
        return Err(format!("winner gate failed: p05={p05} p50={p50}"));
    }
    Ok(())
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

/// `cvt.rna.tf32.f32`: keep ten explicit mantissa bits, ties away from zero.
/// Infinities and NaNs pass through untouched, as the instruction leaves them.
fn to_tf32_rna(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x1000) & !0x1fff;
    f32::from_bits(rounded)
}

struct SmallOracle {
    dims: (usize, usize, usize),
    a: Vec<f32>,
    b: Vec<f32>,
    old: Vec<f32>,
    expected: Vec<u32>,
}

/// Small TN oracle on integers: TF32 represents every integer below 2048
/// exactly and the sums stay far below 2^24, so the exact product is the only
/// admissible answer for the production chain and for any split of it.
fn small_oracle() -> SmallOracle {
    let (m, k, n) = (67_usize, 8_usize, 8_usize);
    let a: Vec<f32> = (0..m * k)
        .map(|index| ((index * 7 + 3) % 23) as f32 - 11.0)
        .collect();
    let b: Vec<f32> = (0..m * n)
        .map(|index| ((index * 5 + 1) % 19) as f32 - 9.0)
        .collect();
    let old: Vec<f32> = (0..k * n)
        .map(|index| ((index * 3 + 2) % 17) as f32 - 8.0)
        .collect();
    let expected = (0..k * n)
        .map(|index| {
            let (row, column) = (index / n, index % n);
            let mut sum = old[index] as f64;
            for reduction in 0..m {
                sum += a[reduction * k + row] as f64 * b[reduction * n + column] as f64;
            }
            (sum as f32).to_bits()
        })
        .collect();
    SmallOracle {
        dims: (m, k, n),
        a,
        b,
        old,
        expected,
    }
}

/// f64 reference of the TN cell from TF32-rounded operands, in a cache
/// friendly order: one `A` element scales one contiguous `B` row.
fn reference_output(dims: (usize, usize, usize), a: &[f32], b: &[f32], old: &[f32]) -> Vec<f64> {
    let (m, k, n) = dims;
    let mut reference: Vec<f64> = old.iter().map(|value| f64::from(*value)).collect();
    let b_rounded: Vec<f64> = b
        .iter()
        .map(|value| f64::from(to_tf32_rna(*value)))
        .collect();
    for reduction in 0..m {
        let b_row = &b_rounded[reduction * n..(reduction + 1) * n];
        for row in 0..k {
            let scale = f64::from(to_tf32_rna(a[reduction * k + row]));
            let out = &mut reference[row * n..(row + 1) * n];
            for (target, source) in out.iter_mut().zip(b_row) {
                *target += scale * source;
            }
        }
    }
    reference
}

fn max_abs_error(actual_bits: &[u32], reference: &[f64]) -> Result<f64, String> {
    if actual_bits.len() != reference.len() {
        return Err("reference length mismatch".into());
    }
    let mut worst = 0.0_f64;
    for (bits, expected) in actual_bits.iter().zip(reference) {
        let actual = f64::from(f32::from_bits(*bits));
        if !actual.is_finite() {
            return Err(format!(
                "non-finite value {bits:#010x} against reference {expected}"
            ));
        }
        worst = worst.max((actual - expected).abs());
    }
    Ok(worst)
}

fn validate_reference_distance(
    label: &str,
    production_error: f64,
    candidate_error: f64,
    reference_scale: f64,
) -> Result<(), String> {
    let allowed =
        REFERENCE_ERROR_FACTOR * production_error + REFERENCE_ERROR_FLOOR * reference_scale;
    if !candidate_error.is_finite() || candidate_error > allowed {
        return Err(format!(
            "{label} reference distance {candidate_error} exceeds {allowed} (production {production_error})"
        ));
    }
    Ok(())
}

#[test]
fn split_factor_seven_fills_the_170_sm_part_on_every_cell() {
    let production = production_spec();
    for cell in CELLS {
        let tiles = tn_tiles(cell.dims, production.tile).unwrap();
        let single = wave_efficiency(tiles, 1, 170);
        let split = wave_efficiency(tiles * 7, 1, 170);
        assert!(single < 0.86, "{}: {single}", cell.label);
        assert!(split > 0.98, "{}: {split}", cell.label);
    }
    assert_eq!(tn_tiles(CELLS[2].dims, production.tile).unwrap(), 96);
    assert!((wave_efficiency(96, 1, 170) - 96.0 / 170.0).abs() < 1e-12);
}

#[test]
fn candidates_are_test_only_and_match_the_storage_formula() {
    for candidate in CANDIDATES {
        validate_spec(candidate).unwrap();
        validate_candidate_symbol(candidate).unwrap();
    }
    let production = production_spec();
    validate_spec(production).unwrap();
    assert_eq!(production.partitions, 1);
    assert_eq!(production.dynamic_shared_bytes, 98_432);
    assert!(validate_candidate_symbol(production).is_err());
    assert_eq!(
        CUDA_SOURCE
            .matches("SM120_DEFINE_TF32_TN_SPLITM_KERNEL(\n")
            .count(),
        CANDIDATES.len()
    );
    assert!(!CUDA_SOURCE.contains("SM120_DEFINE_TF32_KERNEL("));
    assert!(!CUDA_SOURCE.contains("SM120_DEFINE_TF32_PAIR_KERNEL("));
}

#[test]
fn candidate_source_keeps_a_fixed_reduction_order_and_no_floating_atomics() {
    for required in [
        "sm120_tf32_issue_stage<Op, M, N, Stages>(",
        "atomicInc(counters + blockIdx.x, Partitions - 1U)",
        "sum0 = __fadd_rn(sum0, value.x);",
        "__fmaf_rn(params.alpha, sum0, old.x)",
        "st.global.cg.v2.f32",
        "ld.global.cg.v2.f32",
        "__threadfence();",
        "storage + Stages * stage_bytes + 120",
    ] {
        assert!(
            CUDA_SOURCE.contains(required),
            "candidate source lost {required}"
        );
    }
    for forbidden in ["atomicAdd", "red.global.add", "__shfl", "atomicExch"] {
        assert!(
            !CUDA_SOURCE.contains(forbidden),
            "candidate source must not use {forbidden}"
        );
    }
}

#[test]
fn tf32_rounding_and_oracles_behave() {
    assert_eq!(to_tf32_rna(1.0).to_bits(), 1.0_f32.to_bits());
    assert_eq!(to_tf32_rna(f32::INFINITY), f32::INFINITY);
    assert!(to_tf32_rna(f32::NAN).is_nan());
    let one_plus_half_ulp = f32::from_bits(0x3f80_1000);
    assert_eq!(to_tf32_rna(one_plus_half_ulp).to_bits(), 0x3f80_2000);
    let one_plus_less = f32::from_bits(0x3f80_0fff);
    assert_eq!(to_tf32_rna(one_plus_less).to_bits(), 0x3f80_0000);
    assert_eq!(to_tf32_rna(-one_plus_half_ulp).to_bits(), 0xbf80_2000);
    let oracle = small_oracle();
    let (m, k, n) = oracle.dims;
    assert_eq!(oracle.expected.len(), k * n);
    assert!(
        m > 32,
        "the small oracle must span more than one reduction tile"
    );
    let reference = reference_output(oracle.dims, &oracle.a, &oracle.b, &oracle.old);
    assert_eq!(max_abs_error(&oracle.expected, &reference).unwrap(), 0.0);
    assert!(validate_reference_distance("x", 1e-3, 4e-3, 1.0).is_ok());
    assert!(validate_reference_distance("x", 1e-3, 4.1e-3, 1.0).is_err());
    assert!(validate_reference_distance("x", 0.0, 1e-8, 1.0).is_ok());
    assert!(percentile(&[1.0, f64::NAN], 0.5).is_err());
    assert!(validate_winner_gate(1.0, 1.009).is_err());
    assert!(!screening_passes(1.02, 1.049));
    let mut snapshot = vec![f32::from_bits(GUARD_BITS); 2 * GUARD_ELEMENTS + 4];
    snapshot[GUARD_ELEMENTS..GUARD_ELEMENTS + 4].fill(0.0);
    snapshot[GUARD_ELEMENTS] = 1.0;
    assert_eq!(
        guarded_active_bits(&snapshot, 4, None).unwrap(),
        [1.0_f32.to_bits(), 0, 0, 0]
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
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationF32Epilogue, PhysicalQualificationRequest, PhysicalQualificationRoute,
        presize_physical_qualification_suite, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, ModuleKind, NUMERIC_ABI_REVISION,
        ResolvedGemmOp, SCHEDULE_REVISION,
    };
    use std::collections::BTreeMap;
    use std::ffi::{CStr, c_void};
    use std::sync::Arc;

    const TARGET_WINDOW_MS: f64 = 10.0;
    const MAX_WINDOW_ITERATIONS: usize = 16_384;
    const CORPUS_SALT: u64 = 0x5b1d_7e40_2c93;
    const LABEL: &str = "tf32_tn_splitm";

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
            .ok_or_else(|| {
                "the split tournament requires the specialized TF32 module".to_string()
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
                "the split tournament requires an exact CUDA13.2 compute120 CC12.0/170-SM artifact: {specialized:?}"
            ));
        }
        Ok(())
    }

    fn production_request(cell: Cell) -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Tn,
            cell.dims,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::AllowDeterministicTf32V1),
            PhysicalQualificationF32Epilogue::new(1.0, 1.0, false),
        )
    }

    fn validate_production_witness(ctx: &GpuCtx) -> Result<(), String> {
        let requests: Vec<_> = CELLS.iter().map(|cell| production_request(*cell)).collect();
        presize_physical_qualification_suite(ctx, &requests)?;
        let production = production_spec();
        for (cell, request) in CELLS.iter().zip(requests) {
            let qualified = qualify_physical_launch(ctx, request)?;
            qualified.validate_timed_request(ctx, request)?;
            let evidence = qualified.evidence();
            let [node] = evidence.nodes() else {
                return Err(format!(
                    "{} production witness must contain one node: {:?}",
                    cell.label,
                    evidence.nodes()
                ));
            };
            let (m, k, n) = cell.dims;
            if evidence.route_identity().tuning_table_revision != PRODUCTION_TUNING_TABLE_REVISION
                || !evidence.eager_graph_equal()
                || evidence.uniform_module_kind() != Some(ModuleKind::TriadSm120)
                || node.logical_op != ResolvedGemmOp::Tn
                || node.shape != (m, k, n)
                || node.strides != (k, n, n)
                || node.symbol != PRODUCTION_SYMBOL
                || node.tile != Some(production.tile)
                || node.launch.grid_dim != (tn_tiles(cell.dims, production.tile)?, 1, 1)
                || node.launch.block_dim != production.block
                || node.launch.shared_mem_bytes != production.dynamic_shared_bytes
                || node.launch.arguments_digest == [0; 32]
                || evidence.launch_digest() == [0; 32]
                || evidence.request_identity_digest() == [0; 32]
            {
                return Err(format!(
                    "{} production physical identity changed: evidence={evidence:?} node={node:?}",
                    cell.label
                ));
            }
        }
        Ok(())
    }

    /// The tournament launches the pair kernel directly, so production
    /// dispatch admitting a different route on this box does not change the
    /// comparison; it is still worth knowing, so the witness is reported,
    /// never silently swallowed.
    fn report_production_dispatch(ctx: &GpuCtx) {
        match validate_production_witness(ctx) {
            Ok(()) => {
                eprintln!("{LABEL} production dispatch selects {PRODUCTION_SYMBOL} on every cell")
            }
            Err(error) => eprintln!("{LABEL} production dispatch witness: {error}"),
        }
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
            ("tests/gemm_bi_tf32_tn_splitm_tournament.cu", CUDA_SOURCE),
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
            .map_err(|error| format!("compile the SM120 TF32 split tournament source: {error:?}"))
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
        let allowed_static = 0;
        if registers == 0
            || registers > u64::from(spec.maximum_registers)
            || static_shared != allowed_static
        {
            return Err(format!(
                "{} ptxas budget failed: regs={registers}/{} static_shared={static_shared}/{allowed_static}",
                spec.symbol, spec.maximum_registers
            ));
        }
        eprintln!(
            "{LABEL} ptxas symbol={} regs={registers} static_shared={static_shared}",
            spec.symbol
        );
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

    /// A split candidate may carry exactly one integer completion atomic and
    /// nothing else; the production route carries none.
    fn validate_sass_entry(sass: &str, spec: CandidateSpec) -> Result<(), String> {
        let entry = sass_entry(sass, spec.symbol)?;
        let mut integer_atomics = 0_usize;
        for line in entry.lines() {
            let instruction = line
                .split_once("*/")
                .map(|(_, tail)| tail.trim_start())
                .unwrap_or_else(|| line.trim_start());
            let opcode = instruction
                .split_whitespace()
                .find(|token| !token.starts_with('@'))
                .unwrap_or("");
            if opcode.starts_with("RED.") {
                return Err(format!(
                    "compiled {} contains a reduction instruction: {line}",
                    spec.symbol
                ));
            }
            if opcode.starts_with("ATOM") {
                if opcode.contains(".F32") || opcode.contains(".F64") || opcode.contains(".F16") {
                    return Err(format!(
                        "compiled {} contains a floating atomic: {line}",
                        spec.symbol
                    ));
                }
                integer_atomics += 1;
            }
        }
        let allowed = usize::from(spec.partitions > 1);
        if integer_atomics != allowed {
            return Err(format!(
                "compiled {} has {integer_atomics} integer atomics, expected {allowed}",
                spec.symbol
            ));
        }
        eprintln!(
            "{LABEL} sass symbol={} integer_atomics={integer_atomics}",
            spec.symbol
        );
        Ok(())
    }

    fn validate_ptxas_resources_and_sass() -> Result<(), String> {
        let ptx = compile_sm120_source()?;
        let directory =
            tempfile::tempdir().map_err(|error| format!("resource tempdir: {error}"))?;
        let ptx_path = directory.path().join("tn-splitm.ptx");
        let cubin_path = directory.path().join("tn-splitm.cubin");
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
        for spec in CANDIDATES {
            validate_ptxas_entry(&report, spec)?;
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
        validate_sass_entry(&sass, production_spec())?;
        for spec in CANDIDATES {
            validate_sass_entry(&sass, spec)?;
        }
        Ok(())
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
            .map_err(|error| format!("load the split tournament module: {error:?}"))?;
        let production_spec = production_spec();
        let production = module
            .load_function(PRODUCTION_SYMBOL)
            .map_err(|error| format!("load production reference: {error:?}"))?;
        production
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                production_spec.dynamic_shared_bytes as i32,
            )
            .map_err(|error| format!("set production dynamic shared: {error:?}"))?;
        let mut candidates = Vec::with_capacity(CANDIDATES.len());
        for spec in CANDIDATES {
            let function = module
                .load_function(spec.symbol)
                .map_err(|error| format!("load {}: {error:?}", spec.symbol))?;
            function
                .set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    spec.dynamic_shared_bytes as i32,
                )
                .map_err(|error| format!("set {} dynamic shared: {error:?}", spec.symbol))?;
            candidates.push((spec, function));
        }
        Ok(ResourceModule {
            _module: module,
            production,
            candidates,
        })
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
        let allowed_static = 0;
        if registers <= 0
            || registers > spec.maximum_registers as i32
            || local != 0
            || static_shared != allowed_static
            || max_threads < spec.block.0 as i32
            || max_dynamic < spec.dynamic_shared_bytes as i32
            || occupancy < spec.minimum_occupancy
        {
            return Err(format!(
                "{} driver budget failed regs={registers}/{} local={local}/0 static={static_shared}/{allowed_static} dynamic={}/{max_dynamic} threads={}/{} occupancy={}/{}",
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
            "{LABEL} resource symbol={} regs={registers} local={local} static_shared={static_shared} dynamic_shared={} threads={} occupancy={occupancy}",
            spec.symbol, spec.dynamic_shared_bytes, spec.block.0
        );
        Ok(())
    }

    fn validate_driver_resources(device: &GpuDevice) -> Result<(), String> {
        let resources = load_resource_module(device)?;
        validate_function_resources(&resources.production, production_spec())?;
        for (spec, function) in &resources.candidates {
            validate_function_resources(function, *spec)?;
        }
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

    fn cuda_ok(result: sys::CUresult, label: &str) -> Result<(), String> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{label}: {result:?}"))
        }
    }

    /// The production 32x32 plane map: 128-byte rows, TMA 128B swizzle. Both
    /// arms read through the same maps; only the reduction ownership differs.
    fn plane_tensor_map(
        pointer: u64,
        width: usize,
        rows: usize,
        stride: usize,
        label: &str,
    ) -> Result<DirectTensorMap, String> {
        if pointer == 0 || !pointer.is_multiple_of(16) || width == 0 || rows == 0 || stride < width
        {
            return Err(format!("invalid {label} plane tensor-map input"));
        }
        let byte_stride = stride
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("{label} byte stride overflow"))?;
        if !byte_stride.is_multiple_of(16) {
            return Err(format!("{label} byte stride is not 16-byte aligned"));
        }
        let dimensions = [width as u64, rows as u64];
        let global_strides = [byte_stride as u64];
        let box_dimensions = [32_u32, 32_u32];
        let element_strides = [1_u32, 1_u32];
        let mut raw = std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed();
        cuda_ok(
            unsafe {
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
            },
            &format!("encode {label} plane tensor map"),
        )?;
        Ok(DirectTensorMap(unsafe { raw.assume_init() }))
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
        /// Partition slabs and the per-tile completion counters. The
        /// counters wrap to zero on the last arrival, so the buffer is
        /// uploaded once and never touched again.
        partial: GuardedBuffer,
        counters: GuardedBuffer,
        maps: DirectMaps,
        params: DirectParams,
        production_config: LaunchConfig,
        candidate_config: LaunchConfig,
    }

    struct Runtime {
        stream: Arc<CudaStream>,
        module: ResourceModule,
    }

    fn new_runtime(device: &GpuDevice, stream: Arc<CudaStream>) -> Result<Runtime, String> {
        Ok(Runtime {
            stream,
            module: load_resource_module(device)?,
        })
    }

    /// TN geometry: `A` is `[m][k]` with `lda`, `B` is `[m][n]` with `ldb`,
    /// the output is `[k][n]` with `ldc`, and the reduction runs over `m`.
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
            || b_values.len() != m.checked_mul(ldb).ok_or("B extent overflow")?
            || output_values.len() != k.checked_mul(ldc).ok_or("C extent overflow")?
        {
            return Err("fixture geometry or allocation length mismatch".into());
        }
        let tiles = tn_tiles(dims, spec.tile)?;
        let slab = k.checked_mul(n).ok_or("partial slab overflow")?;
        let partial_len = slab
            .checked_mul(spec.partitions as usize)
            .ok_or("partial extent overflow")?;
        let a = GuardedBuffer::new(&runtime.stream, a_values)?;
        let b = GuardedBuffer::new(&runtime.stream, b_values)?;
        let production_output = GuardedBuffer::new(&runtime.stream, output_values.clone())?;
        let candidate_output = GuardedBuffer::new(&runtime.stream, output_values)?;
        let partial = GuardedBuffer::new(&runtime.stream, vec![0.0; partial_len])?;
        let counters = GuardedBuffer::new(&runtime.stream, vec![0.0; tiles as usize])?;
        let a_pointer = a.active_ptr(&runtime.stream, "A")?;
        let b_pointer = b.active_ptr(&runtime.stream, "B")?;
        production_output.active_ptr(&runtime.stream, "production C")?;
        candidate_output.active_ptr(&runtime.stream, "candidate C")?;
        partial.active_ptr(&runtime.stream, "partial")?;
        counters.active_ptr(&runtime.stream, "counters")?;
        let production = production_spec();
        let maps = DirectMaps {
            a: plane_tensor_map(a_pointer, k, m, lda, "A")?,
            b: plane_tensor_map(b_pointer, n, m, ldb, "B")?,
        };
        Ok(Fixture {
            a,
            b,
            production_output,
            candidate_output,
            partial,
            counters,
            maps,
            params: DirectParams {
                a_x: 0,
                a_y: 0,
                b_x: 0,
                b_y: 0,
                alpha: 1.0,
                beta: 1.0,
                m: m.try_into().map_err(|_| "M exceeds i32")?,
                k: k.try_into().map_err(|_| "K exceeds i32")?,
                n: n.try_into().map_err(|_| "N exceeds i32")?,
                ldc: ldc.try_into().map_err(|_| "ldc exceeds i32")?,
            },
            production_config: LaunchConfig {
                grid_dim: (tn_tiles(dims, production.tile)?, 1, 1),
                block_dim: production.block,
                shared_mem_bytes: production.dynamic_shared_bytes,
            },
            candidate_config: LaunchConfig {
                grid_dim: (tiles, 1, spec.partitions),
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
        builder.arg(&fixture.maps.a);
        builder.arg(&fixture.maps.b);
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
        let partial = fixture.partial.active_ptr(&runtime.stream, "partial")?;
        let counters = fixture.counters.active_ptr(&runtime.stream, "counters")?;
        let bias = 0_u64;
        let mut builder = runtime
            .stream
            .launch_builder(runtime.module.candidate(spec)?);
        builder.arg(&output);
        builder.arg(&partial);
        builder.arg(&counters);
        builder.arg(&fixture.maps.a);
        builder.arg(&fixture.maps.b);
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

    fn graph_symbol(graph: &CudaGraph) -> Result<String, String> {
        let mut count = 0_usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) },
            "query graph node count",
        )?;
        if count != 1 {
            return Err(format!("captured graph has {count} nodes, expected one"));
        }
        let mut node = std::ptr::null_mut();
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count) },
            "query graph node",
        )?;
        let mut params = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            "query graph kernel parameters",
        )?;
        let mut name = std::ptr::null();
        cuda_ok(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "query graph function name",
        )?;
        if name.is_null() {
            return Err("graph function name is null".into());
        }
        unsafe { CStr::from_ptr(name) }
            .to_str()
            .map(str::to_owned)
            .map_err(|error| format!("graph function name is not UTF-8: {error}"))
    }

    struct ArmBits {
        production: Vec<u32>,
        candidate: Vec<u32>,
    }

    fn read_outputs(runtime: &Runtime, fixture: &Fixture) -> Result<ArmBits, String> {
        let production = fixture.production_output.bits(&runtime.stream)?;
        let candidate = fixture.candidate_output.bits(&runtime.stream)?;
        fixture.a.validate_read_only(&runtime.stream)?;
        fixture.b.validate_read_only(&runtime.stream)?;
        fixture.counters.validate_read_only(&runtime.stream)?;
        fixture.partial.bits(&runtime.stream)?;
        Ok(ArmBits {
            production,
            candidate,
        })
    }

    fn reset_outputs(runtime: &Runtime, fixture: &mut Fixture) -> Result<(), String> {
        fixture.production_output.reset(&runtime.stream)?;
        fixture.candidate_output.reset(&runtime.stream)
    }

    /// Three eager launches and three graph replays of both arms. The
    /// candidate must repeat its own bits exactly and its graph must equal its
    /// eager bits; the production bits are returned for the reference gate.
    fn run_deterministic(
        runtime: &Runtime,
        spec: CandidateSpec,
        fixture: &mut Fixture,
        label: &str,
    ) -> Result<ArmBits, String> {
        let mut first: Option<ArmBits> = None;
        for repeat in 0..3 {
            reset_outputs(runtime, fixture)?;
            launch_production(runtime, fixture)?;
            launch_candidate(runtime, spec, fixture)?;
            synchronize(runtime, label)?;
            let actual = read_outputs(runtime, fixture)?;
            if let Some(previous) = &first
                && (previous.production != actual.production
                    || previous.candidate != actual.candidate)
            {
                return Err(format!(
                    "{} {label} eager repeat {repeat} changed",
                    spec.symbol
                ));
            }
            first.get_or_insert(actual);
        }
        let production_graph = capture_production(runtime, fixture)?;
        let candidate_graph = capture_candidate(runtime, spec, fixture)?;
        if graph_symbol(&production_graph)? != PRODUCTION_SYMBOL
            || graph_symbol(&candidate_graph)? != spec.symbol
        {
            return Err(format!(
                "{} {label} captured an unexpected symbol",
                spec.symbol
            ));
        }
        let eager = first.ok_or_else(|| "deterministic run produced no bits".to_string())?;
        for repeat in 0..3 {
            reset_outputs(runtime, fixture)?;
            production_graph
                .launch()
                .map_err(|error| format!("{label} production graph: {error:?}"))?;
            candidate_graph
                .launch()
                .map_err(|error| format!("{label} candidate graph: {error:?}"))?;
            synchronize(runtime, label)?;
            let actual = read_outputs(runtime, fixture)?;
            if actual.production != eager.production || actual.candidate != eager.candidate {
                return Err(format!(
                    "{} {label} graph repeat {repeat} differs from eager",
                    spec.symbol
                ));
            }
        }
        Ok(eager)
    }

    fn run_small_oracle(runtime: &Runtime, spec: CandidateSpec) -> Result<(), String> {
        let oracle = small_oracle();
        let (_, k, n) = oracle.dims;
        let mut fixture = new_fixture(
            runtime,
            spec,
            oracle.dims,
            (k, n, n),
            oracle.a,
            oracle.b,
            oracle.old,
        )?;
        let bits = run_deterministic(runtime, spec, &mut fixture, "small")?;
        if bits.production != oracle.expected {
            return Err("production missed the integer oracle".into());
        }
        if bits.candidate != oracle.expected {
            return Err(format!("{} missed the integer oracle", spec.symbol));
        }
        Ok(())
    }

    struct CellReference {
        a: Vec<f32>,
        b: Vec<f32>,
        old: Vec<f32>,
        reference: Vec<f64>,
        scale: f64,
    }

    fn cell_reference(cell: Cell) -> CellReference {
        let (m, k, n) = cell.dims;
        let a = seeded_values(m * k, CORPUS_SALT ^ 0xa1);
        let b = seeded_values(m * n, CORPUS_SALT ^ 0xb2);
        let old = seeded_values(k * n, CORPUS_SALT ^ 0xc3);
        let reference = reference_output(cell.dims, &a, &b, &old);
        let scale = reference
            .iter()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()));
        CellReference {
            a,
            b,
            old,
            reference,
            scale,
        }
    }

    fn run_cell(
        runtime: &Runtime,
        spec: CandidateSpec,
        cell: Cell,
        reference: &CellReference,
    ) -> Result<(), String> {
        let (_, k, n) = cell.dims;
        let mut fixture = new_fixture(
            runtime,
            spec,
            cell.dims,
            (k, n, n),
            reference.a.clone(),
            reference.b.clone(),
            reference.old.clone(),
        )?;
        let bits = run_deterministic(runtime, spec, &mut fixture, cell.label)?;
        let production_error = max_abs_error(&bits.production, &reference.reference)?;
        let candidate_error = max_abs_error(&bits.candidate, &reference.reference)?;
        let identical = bits.production == bits.candidate;
        eprintln!(
            "{LABEL} reference cell={} candidate={} production_max_abs_error={production_error:.9e} candidate_max_abs_error={candidate_error:.9e} scale={:.6e} bits_identical_to_production={identical}",
            cell.label, spec.symbol, reference.scale
        );
        validate_reference_distance(
            &format!("{} {}", cell.label, spec.symbol),
            production_error,
            candidate_error,
            reference.scale,
        )
    }

    #[test]
    #[ignore = "requires exclusive CC12.0/170-SM CUDA13.2 hardware"]
    fn splitm_candidates_are_deterministic_and_within_reference_distance() -> Result<(), String> {
        let device = GpuDevice::new(0)?;
        let ctx = configure(&device)?;
        validate_exact_environment(&ctx)?;
        report_production_dispatch(&ctx);
        validate_ptxas_resources_and_sass()?;
        validate_driver_resources(&device)?;
        let runtime = new_runtime(&device, ctx.stream.clone())?;
        let references: BTreeMap<&str, CellReference> = CELLS
            .iter()
            .map(|cell| (cell.label, cell_reference(*cell)))
            .collect();
        for spec in CANDIDATES {
            run_small_oracle(&runtime, spec)?;
            for cell in CELLS {
                run_cell(&runtime, spec, cell, &references[cell.label])?;
            }
        }
        Ok(())
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
                Self::Abba => "abba",
                Self::Baab => "baab",
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
        candidate: Option<CandidateSpec>,
        graph: CudaGraph,
    }

    fn new_timed_arm<'a>(
        runtime: &'a Runtime,
        cell: Cell,
        spec: CandidateSpec,
        candidate: Option<CandidateSpec>,
    ) -> Result<TimedArm<'a>, String> {
        let (m, k, n) = cell.dims;
        let fixture = new_fixture(
            runtime,
            spec,
            cell.dims,
            (k, n, n),
            seeded_values(m * k, CORPUS_SALT ^ 0x11),
            seeded_values(m * n, CORPUS_SALT ^ 0x22),
            vec![0.0; k * n],
        )?;
        let graph = match candidate {
            None => capture_production(runtime, &fixture)?,
            Some(spec) => capture_candidate(runtime, spec, &fixture)?,
        };
        Ok(TimedArm {
            runtime,
            fixture,
            candidate,
            graph,
        })
    }

    fn arm_name(candidate: Option<CandidateSpec>) -> &'static str {
        candidate.map_or(PRODUCTION_SYMBOL, |spec| spec.symbol)
    }

    fn launch_timed(arm: &TimedArm<'_>, path: Path) -> Result<(), String> {
        match (path, arm.candidate) {
            (Path::Eager, None) => launch_production(arm.runtime, &arm.fixture),
            (Path::Eager, Some(spec)) => launch_candidate(arm.runtime, spec, &arm.fixture),
            (Path::Graph, _) => arm
                .graph
                .launch()
                .map_err(|error| format!("launch {} graph: {error:?}", arm_name(arm.candidate))),
        }
    }

    fn measure(arm: &TimedArm<'_>, path: Path, iterations: usize) -> Result<f64, String> {
        if iterations == 0 {
            return Err("timing iterations must be positive".into());
        }
        let name = arm_name(arm.candidate);
        let start = arm
            .runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("{name} start event: {error:?}"))?;
        for _ in 0..iterations {
            launch_timed(arm, path)?;
        }
        let end = arm
            .runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("{name} end event: {error:?}"))?;
        let microseconds = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("{name} elapsed event: {error:?}"))?,
        ) * 1_000.0
            / iterations as f64;
        if !microseconds.is_finite() || microseconds <= 0.0 {
            return Err(format!("{name} invalid timing {microseconds}"));
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
        cell: Cell,
        candidate: CandidateSpec,
        path: Path,
        order: Order,
        windows: usize,
        stats: TimingStats,
    ) {
        eprintln!(
            "{LABEL} cell={} baseline={PRODUCTION_SYMBOL} candidate={} path={} order={} windows={} baseline_us_p50={:.6} candidate_us_p50={:.6} speedup_p05={:.9} speedup_p50={:.9} speedup_p95={:.9}",
            cell.label,
            candidate.symbol,
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
        runtime: &Runtime,
        cell: Cell,
        spec: CandidateSpec,
    ) -> Result<ScreenResult, String> {
        let production = new_timed_arm(runtime, cell, spec, None)?;
        let candidate = new_timed_arm(runtime, cell, spec, Some(spec))?;
        let mut worst_p05 = f64::INFINITY;
        let mut worst_p50 = f64::INFINITY;
        for path in [Path::Eager, Path::Graph] {
            for order in [Order::Abba, Order::Baab] {
                let stats = paired(&production, &candidate, path, order, SCREENING_WINDOWS)?;
                print_stats(cell, spec, path, order, SCREENING_WINDOWS, stats);
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

    fn official(
        runtime: &Runtime,
        cell: Cell,
        spec: CandidateSpec,
        quiet: &QuietGpu,
    ) -> Result<(), String> {
        let cohort = format!("{LABEL}-official-{}", cell.label);
        quiet.require_cohort(&cohort)?;
        let production = new_timed_arm(runtime, cell, spec, None)?;
        let candidate = new_timed_arm(runtime, cell, spec, Some(spec))?;
        for path in [Path::Eager, Path::Graph] {
            for order in [Order::Abba, Order::Baab] {
                let stats = paired(&production, &candidate, path, order, OFFICIAL_WINDOWS)?;
                print_stats(cell, spec, path, order, OFFICIAL_WINDOWS, stats);
                validate_winner_gate(stats.speedup_p05, stats.speedup_p50).map_err(|error| {
                    format!(
                        "{} {} path={path:?} order={order:?}: {error}",
                        cell.label, spec.symbol
                    )
                })?;
            }
        }
        quiet.verify_post_cohort(&cohort)?;
        Ok(())
    }

    fn validate_release_build() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("timing requires a release build".into());
        }
        Ok(())
    }

    /// Screens every split factor on every cell with 21 windows, then confirms
    /// the best screened candidate per cell with the official 101-window run.
    #[test]
    #[ignore = "requires exclusive CC12.0/170-SM CUDA13.2 hardware"]
    fn splitm_tournament_screen21_and_official101_abba_baab() -> Result<(), String> {
        validate_release_build()?;
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context(&format!("{LABEL}-pre-context"))?;
        let device = GpuDevice::new(0)?;
        let ctx = configure(&device)?;
        validate_exact_environment(&ctx)?;
        report_production_dispatch(&ctx);
        validate_driver_resources(&device)?;
        let runtime = new_runtime(&device, ctx.stream.clone())?;
        let references: BTreeMap<&str, CellReference> = CELLS
            .iter()
            .map(|cell| (cell.label, cell_reference(*cell)))
            .collect();
        for spec in CANDIDATES {
            run_small_oracle(&runtime, spec)?;
            for cell in CELLS {
                run_cell(&runtime, spec, cell, &references[cell.label])?;
            }
        }
        let mut winners = Vec::new();
        for cell in CELLS {
            let cohort = format!("{LABEL}-screen-{}", cell.label);
            quiet.require_cohort(&cohort)?;
            let mut best: Option<ScreenResult> = None;
            for spec in CANDIDATES {
                let result = screen_candidate(&runtime, cell, spec)?;
                eprintln!(
                    "{LABEL} screen cell={} candidate={} worst_p05={:.9} worst_p50={:.9} passes={}",
                    cell.label,
                    spec.symbol,
                    result.worst_p05,
                    result.worst_p50,
                    screening_passes(result.worst_p05, result.worst_p50)
                );
                if screening_passes(result.worst_p05, result.worst_p50)
                    && best.is_none_or(|current| result.worst_p50 > current.worst_p50)
                {
                    best = Some(result);
                }
            }
            quiet.verify_post_cohort(&cohort)?;
            match best {
                Some(result) => winners.push((cell, result.spec)),
                None => eprintln!("{LABEL} screen cell={} winner=none", cell.label),
            }
        }
        for (cell, spec) in &winners {
            official(&runtime, *cell, *spec, &quiet)?;
            eprintln!(
                "{LABEL} official cell={} winner={}",
                cell.label, spec.symbol
            );
        }
        if winners.is_empty() {
            return Err("no split candidate passed screening on any cell".into());
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires ptxas and nvdisasm 13.2"]
    fn splitm_candidates_have_strict_ptxas_and_sass_resources() -> Result<(), String> {
        validate_ptxas_resources_and_sass()
    }

    #[test]
    #[ignore = "requires CC12.0 CUDA13.2 hardware"]
    fn splitm_candidates_have_strict_driver_resources() -> Result<(), String> {
        let device = GpuDevice::new(0)?;
        let ctx = configure(&device)?;
        validate_exact_environment(&ctx)?;
        validate_driver_resources(&device)
    }
}
