//! Test-only tournament: SM120 deterministic TF32 TN stream-K routes that deal
//! the (tile, k) work space to a fixed persistent grid and combine straddled
//! tiles in a fixed order.
//!
//! The production TN route owns one output tile per CTA and leaves part of a
//! 170-SM machine idle on every training cell whose tile count does not fill
//! its waves. A stream-K family fills the machine at the price of a different,
//! still deterministic, summation tree, so the harness does not compare bits
//! against production. It demands
//! the exact integer oracle, three identical eager repeats, three identical
//! graph replays that equal the eager bits, intact red zones and untouched
//! inputs, and an f64 reference distance no worse than a small multiple of the
//! production route's own distance on every cell. Nothing here registers a
//! production route.

#[cfg(feature = "cuda")]
mod common;

const CUDA_SOURCE: &str = include_str!("gemm_bi_tf32_tn_streamk_tournament.cu");
const PRODUCTION_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm120.cu");
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
    /// Persistent CTA count; the harness admits it only when the device keeps
    /// that many resident at once.
    grid: u32,
    block: (u32, u32, u32),
    /// Rows and columns of the tile one warp owns; every thread keeps
    /// `warp_tile.0 * warp_tile.1 / 32` accumulators.
    warp_tile: (u32, u32),
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
    warp_tile: (u32, u32),
    stages: u32,
    grid: u32,
    minimum_occupancy: u32,
) -> CandidateSpec {
    CandidateSpec {
        symbol,
        tile,
        stages,
        grid,
        block: ((tile.0 / warp_tile.0) * (tile.1 / warp_tile.1) * 32, 1, 1),
        warp_tile,
        dynamic_shared_bytes: dynamic_shared_bytes(tile, stages),
        minimum_occupancy,
        // Two resident CTAs share the 64K-register file, one has it alone;
        // the budget follows what the candidate must keep resident.
        maximum_registers: if minimum_occupancy >= 2 { 128 } else { 255 },
    }
}

impl CandidateSpec {
    const fn accumulators_per_thread(self) -> u32 {
        self.warp_tile.0 * self.warp_tile.1 / 32
    }
}

// The first two entries keep the first design as the yardstick; the rest
// are the continuous-pipeline design at the same grids.
// The production stream-K kernel runs as a candidate too, so the promoted
// build is measured in the same harness as the experiments.
const CANDIDATES: [CandidateSpec; 11] = [
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n128_bk32_s3_pair_exp_streamk_wide",
        (128, 128),
        (32, 64),
        3,
        170,
        1,
    ),
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n128_bk32_s2_pair_exp_streamk_wide",
        (128, 128),
        (32, 64),
        2,
        170,
        1,
    ),
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_streamk",
        (64, 128),
        (32, 32),
        3,
        170,
        1,
    ),
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair_exp_streamk_drained",
        (64, 128),
        (32, 32),
        4,
        170,
        1,
    ),
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_exp_streamk_flowing",
        (64, 128),
        (32, 32),
        3,
        96,
        1,
    ),
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_exp_streamk_flowing",
        (64, 128),
        (32, 32),
        3,
        128,
        1,
    ),
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2_pair_exp_streamk_drained",
        (64, 128),
        (32, 32),
        2,
        340,
        2,
    ),
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair_exp_streamk_flowing",
        (64, 128),
        (32, 32),
        4,
        170,
        1,
    ),
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_exp_streamk_flowing",
        (64, 128),
        (32, 32),
        3,
        170,
        1,
    ),
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2_pair_exp_streamk_flowing",
        (64, 128),
        (32, 32),
        2,
        170,
        2,
    ),
    spec(
        "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2_pair_exp_streamk_flowing",
        (64, 128),
        (32, 32),
        2,
        340,
        2,
    ),
];

const fn production_spec() -> CandidateSpec {
    spec(PRODUCTION_SYMBOL, (64, 128), (32, 32), 4, 0, 1)
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
    if spec.grid > 170 * spec.minimum_occupancy {
        return Err(format!(
            "{} grid exceeds what {} resident blocks per SM keep co-resident",
            spec.symbol, spec.minimum_occupancy
        ));
    }
    let (warp_rows, warp_columns) = spec.warp_tile;
    if warp_rows == 0
        || warp_columns == 0
        || !rows.is_multiple_of(warp_rows)
        || !columns.is_multiple_of(warp_columns)
        || !warp_rows.is_multiple_of(16)
        || !warp_columns.is_multiple_of(8)
    {
        return Err(format!("{} warp tile does not tile the block", spec.symbol));
    }
    if spec.block != ((rows / warp_rows) * (columns / warp_columns) * 32, 1, 1) {
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
    if !spec.symbol.starts_with("gemm_bi_tn_sm120_tma_mma_tf32_v1_")
        || !(spec.symbol.ends_with("_exp_streamk_drained")
            || spec.symbol.ends_with("_exp_streamk_flowing")
            || spec.symbol.ends_with("_exp_streamk_wide")
            || spec.symbol.ends_with("_pair_streamk"))
    {
        return Err(format!(
            "{} is not a test-only TN stream-K candidate",
            spec.symbol
        ));
    }
    let definition = format!(
        "{}, {}, {}, {}",
        spec.symbol, spec.tile.0, spec.tile.1, spec.stages
    );
    if !CUDA_SOURCE.contains(&definition) && !PRODUCTION_SOURCE.contains(&definition) {
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

/// Mirror of the device dealing formula: CTA ranges are contiguous, cover
/// the unit space exactly once, and differ by at most one unit.
fn streamk_range(units: u64, grid: u64, cta: u64) -> (u64, u64) {
    let base = units / grid;
    let remainder = units % grid;
    let first = cta * base + cta.min(remainder);
    (first, first + base + u64::from(cta < remainder))
}

#[test]
fn stream_k_dealing_covers_every_unit_once_and_balances_the_grid() {
    let production = production_spec();
    for cell in CELLS {
        let tiles = u64::from(tn_tiles(cell.dims, production.tile).unwrap());
        let k_tiles = (cell.dims.0 as u64).div_ceil(32);
        let units = tiles * k_tiles;
        for grid in [170_u64, 340] {
            let mut cursor = 0;
            let mut sizes = Vec::new();
            for cta in 0..grid {
                let (first, last) = streamk_range(units, grid, cta);
                assert_eq!(first, cursor, "{} grid {grid} cta {cta}", cell.label);
                cursor = last;
                sizes.push(last - first);
            }
            assert_eq!(cursor, units);
            assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
            let single = wave_efficiency(tiles as u32, 1, 170);
            assert!(single < 0.86, "{}: {single}", cell.label);
        }
    }
    assert_eq!(streamk_range(3, 170, 2), (2, 3));
    assert_eq!(streamk_range(3, 170, 3), (3, 3));
}

#[test]
fn candidates_are_test_only_and_match_the_storage_formula() {
    for candidate in CANDIDATES {
        validate_spec(candidate).unwrap();
        validate_candidate_symbol(candidate).unwrap();
    }
    let production = production_spec();
    validate_spec(production).unwrap();
    assert_eq!(production.grid, 0);
    assert_eq!(production.dynamic_shared_bytes, 98_432);
    assert!(validate_candidate_symbol(production).is_err());
    assert_eq!(
        CUDA_SOURCE
            .matches("SM120_DEFINE_TF32_TN_EXP_STREAMK_DRAINED_KERNEL(\n")
            .count(),
        3
    );
    assert_eq!(
        CUDA_SOURCE
            .matches("SM120_DEFINE_TF32_TN_EXP_STREAMK_FLOWING_KERNEL(\n")
            .count(),
        3
    );
    assert_eq!(
        CUDA_SOURCE
            .matches("SM120_DEFINE_TF32_TN_EXP_STREAMK_WIDE_KERNEL(\n")
            .count(),
        2
    );
    let mut identities: Vec<(&str, u32)> = CANDIDATES
        .iter()
        .map(|spec| (spec.symbol, spec.grid))
        .collect();
    identities.sort_unstable();
    identities.dedup();
    assert_eq!(identities.len(), CANDIDATES.len());
    assert!(!CUDA_SOURCE.contains("SM120_DEFINE_TF32_KERNEL("));
    assert!(!CUDA_SOURCE.contains("SM120_DEFINE_TF32_PAIR_KERNEL("));
}

#[test]
fn candidate_source_keeps_a_fixed_reduction_order_and_no_floating_atomics() {
    for required in [
        "sm120_tf32_issue_stage<Sm120Tn, M, N, Stages>(",
        "for (int source = first_cta; source < cta; ++source)",
        "__fadd_rn(accumulator[m_atom][n_atom][0], value.x)",
        "st.release.gpu.global.u32",
        "ld.acquire.gpu.global.u32",
        "st.global.cg.v4.f32",
        "ld.global.cg.v4.f32",
        "__threadfence();",
        "sm120_tf32_store_pair<Sm120Tn>(",
    ] {
        assert!(
            CUDA_SOURCE.contains(required),
            "candidate source lost {required}"
        );
    }
    for forbidden in [
        "atomicAdd",
        "atomicInc",
        "red.global.add",
        "__shfl",
        "atomicExch",
    ] {
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
    const LABEL: &str = "tf32_tn_streamk";

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
                format!(
                    "the stream-K tournament requires the specialized TF32 module: {}",
                    ctx.kernels
                        .specialized_tf32_rejection()
                        .unwrap_or("no rejection recorded")
                )
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
                "the stream-K tournament requires an exact CUDA13.2 compute120 CC12.0/170-SM artifact: {specialized:?}"
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
            ("tests/gemm_bi_tf32_tn_streamk_tournament.cu", CUDA_SOURCE),
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

    /// The traced build stamps %globaltimer around every segment; only the
    /// trace test loads it, every gate and timing runs the plain build.
    fn compile_sm120_source(trace: bool) -> Result<String, String> {
        let mut options = vec![
            "--fmad=true".to_owned(),
            "--extra-device-vectorization".to_owned(),
            "-DNDEBUG".to_owned(),
        ];
        if trace {
            options.push("-DSM120_EXP_STREAMK_TRACE=1".to_owned());
        }
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            options,
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        cudarc::nvrtc::compile_ptx_with_opts(composed_source(), options)
            .map(|ptx| ptx.to_src())
            .map_err(|error| {
                format!("compile the SM120 TF32 stream-K tournament source: {error:?}")
            })
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

    /// Stream-K coordinates through release/acquire flags, never through
    /// atomics; neither arm may carry an atomic or reduction instruction.
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
        if integer_atomics != 0 {
            return Err(format!(
                "compiled {} has {integer_atomics} integer atomics, expected none",
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
        let ptx = compile_sm120_source(false)?;
        let directory =
            tempfile::tempdir().map_err(|error| format!("resource tempdir: {error}"))?;
        let ptx_path = directory.path().join("tn-streamk.ptx");
        let cubin_path = directory.path().join("tn-streamk.cubin");
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

    fn load_resource_module(device: &GpuDevice, trace: bool) -> Result<ResourceModule, String> {
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(compile_sm120_source(trace)?))
            .map_err(|error| format!("load the stream-K tournament module: {error:?}"))?;
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
            "{LABEL} resource symbol={} regs={registers} local={local} static_shared={static_shared} dynamic_shared={} threads={} occupancy={occupancy}",
            spec.symbol, spec.dynamic_shared_bytes, spec.block.0
        );
        Ok(())
    }

    fn validate_driver_resources(device: &GpuDevice) -> Result<(), String> {
        let resources = load_resource_module(device, false)?;
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
        /// Two fragment-order slabs per CTA and two flags per CTA. Owners
        /// clear the flags they consumed, so the buffer is uploaded once.
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

    fn new_runtime(
        device: &GpuDevice,
        stream: Arc<CudaStream>,
        trace: bool,
    ) -> Result<Runtime, String> {
        Ok(Runtime {
            stream,
            module: load_resource_module(device, trace)?,
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
        let slab = (spec.block.0 as usize) * (spec.accumulators_per_thread() as usize);
        let slots = (spec.grid.max(1) as usize) * 2;
        let partial_len = slab
            .checked_mul(slots)
            .and_then(|slabs| slabs.checked_add(trace_words(spec.grid)))
            .ok_or("partial extent overflow")?;
        let a = GuardedBuffer::new(&runtime.stream, a_values)?;
        let b = GuardedBuffer::new(&runtime.stream, b_values)?;
        let production_output = GuardedBuffer::new(&runtime.stream, output_values.clone())?;
        let candidate_output = GuardedBuffer::new(&runtime.stream, output_values)?;
        let partial = GuardedBuffer::new(&runtime.stream, vec![0.0; partial_len])?;
        let counters = GuardedBuffer::new(&runtime.stream, vec![0.0; slots])?;
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
                grid_dim: (spec.grid.max(1), 1, 1),
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
    fn streamk_candidates_are_deterministic_and_within_reference_distance() -> Result<(), String> {
        let device = GpuDevice::new(0)?;
        let ctx = configure(&device)?;
        validate_exact_environment(&ctx)?;
        report_production_dispatch(&ctx);
        validate_ptxas_resources_and_sass()?;
        validate_driver_resources(&device)?;
        let runtime = new_runtime(&device, ctx.stream.clone(), false)?;
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
    fn streamk_tournament_screen21_and_official101_abba_baab() -> Result<(), String> {
        validate_release_build()?;
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context(&format!("{LABEL}-pre-context"))?;
        let device = GpuDevice::new(0)?;
        let ctx = configure(&device)?;
        validate_exact_environment(&ctx)?;
        report_production_dispatch(&ctx);
        validate_driver_resources(&device)?;
        let runtime = new_runtime(&device, ctx.stream.clone(), false)?;
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
            return Err("no stream-K candidate passed screening on any cell".into());
        }
        Ok(())
    }

    /// Eight 64-bit stamps per CTA, stored as f32 words behind the slabs.
    fn trace_words(grid: u32) -> usize {
        (grid.max(1) as usize) * 16
    }

    struct CtaTrace {
        entry: u64,
        first_begin: u64,
        first_end: u64,
        last_begin: u64,
        last_end: u64,
        fixup_end: u64,
        exit: u64,
        segments: u64,
        units_first: u64,
        units_last: u64,
    }

    fn read_traces(
        runtime: &Runtime,
        fixture: &Fixture,
        grid: u32,
    ) -> Result<Vec<CtaTrace>, String> {
        let bits = fixture.partial.bits(&runtime.stream)?;
        let words = trace_words(grid);
        if bits.len() < words {
            return Err("partial buffer is shorter than its trace tail".into());
        }
        let tail = &bits[bits.len() - words..];
        Ok((0..grid as usize)
            .map(|cta| {
                let word = |slot: usize| {
                    u64::from(tail[cta * 16 + slot * 2])
                        | (u64::from(tail[cta * 16 + slot * 2 + 1]) << 32)
                };
                let packed = word(7);
                CtaTrace {
                    entry: word(0),
                    first_begin: word(1),
                    first_end: word(2),
                    last_begin: word(3),
                    last_end: word(4),
                    fixup_end: word(5),
                    exit: word(6),
                    segments: packed >> 48,
                    units_first: (packed >> 24) & 0xff_ffff,
                    units_last: packed & 0xff_ffff,
                }
            })
            .collect())
    }

    fn median_us(values: &mut [f64]) -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        values.sort_by(f64::total_cmp);
        values[values.len() / 2] / 1_000.0
    }

    fn report_trace(cell: Cell, spec: CandidateSpec, traces: &[CtaTrace]) {
        let origin = traces.iter().map(|t| t.entry).min().unwrap_or(0);
        let span = traces
            .iter()
            .map(|t| t.exit)
            .max()
            .unwrap_or(0)
            .saturating_sub(origin);
        let skew = traces
            .iter()
            .map(|t| t.entry)
            .max()
            .unwrap_or(0)
            .saturating_sub(origin);
        let delta = |a: u64, b: u64| a.saturating_sub(b) as f64;
        let mut prologue: Vec<f64> = traces
            .iter()
            .map(|t| delta(t.first_begin, t.entry))
            .collect();
        let mut first_rate: Vec<f64> = traces
            .iter()
            .filter(|t| t.units_first > 0)
            .map(|t| delta(t.first_end, t.first_begin) / t.units_first as f64)
            .collect();
        let mut last_rate: Vec<f64> = traces
            .iter()
            .filter(|t| t.segments >= 2 && t.units_last > 0)
            .map(|t| delta(t.last_end, t.last_begin) / t.units_last as f64)
            .collect();
        let mut gap: Vec<f64> = traces
            .iter()
            .filter(|t| t.segments == 2)
            .map(|t| delta(t.last_begin, t.first_end))
            .collect();
        let mut fixup: Vec<f64> = traces
            .iter()
            .filter(|t| t.fixup_end != 0)
            .map(|t| delta(t.fixup_end, t.last_end))
            .collect();
        let mut tail: Vec<f64> = traces
            .iter()
            .map(|t| delta(t.exit, t.last_end.max(t.fixup_end)))
            .collect();
        let mut busy: Vec<f64> = traces.iter().map(|t| delta(t.exit, t.entry)).collect();
        let busy_max = traces
            .iter()
            .map(|t| t.exit.saturating_sub(t.entry))
            .max()
            .unwrap_or(0);
        let mut histogram = [0_usize; 4];
        for trace in traces {
            histogram[(trace.segments as usize).min(3)] += 1;
        }
        let owners = fixup.len();
        let mut slowest: Vec<(u64, usize)> = traces
            .iter()
            .enumerate()
            .map(|(cta, t)| (t.exit.saturating_sub(t.entry), cta))
            .collect();
        slowest.sort_unstable_by(|a, b| b.cmp(a));
        let stragglers: Vec<String> = slowest
            .iter()
            .take(4)
            .map(|(busy, cta)| {
                let t = &traces[*cta];
                format!(
                    "cta{cta}:{:.1}us/seg{}/{}+{}",
                    *busy as f64 / 1_000.0,
                    t.segments,
                    t.units_first,
                    t.units_last
                )
            })
            .collect();
        let busy_p90 = slowest[slowest.len() / 10].0 as f64 / 1_000.0;
        eprintln!(
            "{LABEL} stragglers cell={} candidate={} grid={} busy_p90_us={busy_p90:.2} {}",
            cell.label,
            spec.symbol,
            spec.grid,
            stragglers.join(" ")
        );
        eprintln!(
            "{LABEL} trace cell={} candidate={} grid={} span_us={:.2} entry_skew_us={:.2} busy_p50_us={:.2} busy_max_us={:.2} prologue_us={:.2} first_ns_per_unit={:.1} last_ns_per_unit={:.1} segment_gap_us={:.2} fixup_us={:.2} owners={owners} tail_us={:.2} segments_1_2_3={}/{}/{}",
            cell.label,
            spec.symbol,
            spec.grid,
            span as f64 / 1_000.0,
            skew as f64 / 1_000.0,
            median_us(&mut busy),
            busy_max as f64 / 1_000.0,
            median_us(&mut prologue),
            median_us(&mut first_rate) * 1_000.0,
            median_us(&mut last_rate) * 1_000.0,
            median_us(&mut gap),
            median_us(&mut fixup),
            median_us(&mut tail),
            histogram[1],
            histogram[2],
            histogram[3],
        );
    }

    #[test]
    #[ignore = "requires exclusive CC12.0/170-SM CUDA13.2 hardware"]
    fn streamk_segment_trace() -> Result<(), String> {
        let device = GpuDevice::new(0)?;
        let ctx = configure(&device)?;
        validate_exact_environment(&ctx)?;
        let runtime = new_runtime(&device, ctx.stream.clone(), true)?;
        for cell in CELLS {
            for spec in CANDIDATES {
                let (m, k, n) = cell.dims;
                let fixture = new_fixture(
                    &runtime,
                    spec,
                    cell.dims,
                    (k, n, n),
                    seeded_values(m * k, CORPUS_SALT ^ 0x11),
                    seeded_values(m * n, CORPUS_SALT ^ 0x22),
                    vec![0.0; k * n],
                )?;
                // Every launch overwrites the trace, so the readback holds the
                // last launch of a long back-to-back batch: the first few
                // milliseconds after an idle gap run below the settled clock.
                for _ in 0..1_024 {
                    launch_candidate(&runtime, spec, &fixture)?;
                }
                synchronize(&runtime, "trace batch")?;
                // One more launch between two events: the wall time of a launch
                // next to the span the stamps see inside it.
                let start = runtime
                    .stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(|error| format!("trace start event: {error:?}"))?;
                launch_candidate(&runtime, spec, &fixture)?;
                let end = runtime
                    .stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(|error| format!("trace end event: {error:?}"))?;
                synchronize(&runtime, "trace event launch")?;
                let event_us = f64::from(
                    start
                        .elapsed_ms(&end)
                        .map_err(|error| format!("trace elapsed: {error:?}"))?,
                ) * 1_000.0;
                let traces = read_traces(&runtime, &fixture, spec.grid)?;
                eprintln!(
                    "{LABEL} launch cell={} candidate={} grid={} event_us={event_us:.2}",
                    cell.label, spec.symbol, spec.grid
                );
                report_trace(cell, spec, &traces);
            }
        }
        Ok(())
    }

    /// The launch cost of an empty kernel at the candidates' launch shapes:
    /// what a back-to-back stream pays between two kernel bodies.
    #[test]
    #[ignore = "requires exclusive CC12.0/170-SM CUDA13.2 hardware"]
    fn streamk_launch_cost_probe() -> Result<(), String> {
        let device = GpuDevice::new(0)?;
        let ctx = configure(&device)?;
        validate_exact_environment(&ctx)?;
        let source = r#"
extern "C" __global__ void launch_probe_kernel(float* sink, int arm) {
    extern __shared__ float storage[];
    if (arm < 0) {
        storage[threadIdx.x] = 1.0f;
        __syncthreads();
        sink[blockIdx.x] = storage[(threadIdx.x + 1) % blockDim.x];
    }
}
"#;
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(source, options)
            .map_err(|error| format!("compile the launch probe: {error:?}"))?;
        let module = device
            .context()
            .load_module(ptx)
            .map_err(|error| format!("load the launch probe: {error:?}"))?;
        let function = module
            .load_function("launch_probe_kernel")
            .map_err(|error| format!("load the launch probe function: {error:?}"))?;
        function
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                98_432,
            )
            .map_err(|error| format!("set launch probe dynamic shared: {error:?}"))?;
        let sink = GuardedBuffer::new(&ctx.stream, vec![0.0; 512])?;
        let sink_ptr = sink.active_ptr(&ctx.stream, "sink")?;
        let arm = 0_i32;
        for (grid, shared) in [
            (170_u32, 0_u32),
            (170, 49_280),
            (170, 73_856),
            (170, 98_432),
            (96, 98_432),
            (288, 98_432),
            (340, 49_280),
        ] {
            let config = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared,
            };
            let launch = |stream: &Arc<CudaStream>| -> Result<(), String> {
                let mut builder = stream.launch_builder(&function);
                builder.arg(&sink_ptr);
                builder.arg(&arm);
                unsafe { builder.launch(config) }
                    .map(|_| ())
                    .map_err(|error| format!("launch probe: {error:?}"))
            };
            for _ in 0..256 {
                launch(&ctx.stream)?;
            }
            ctx.stream
                .synchronize()
                .map_err(|error| format!("probe warmup: {error:?}"))?;
            let iterations = 2_000;
            let start = ctx
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| format!("probe start event: {error:?}"))?;
            for _ in 0..iterations {
                launch(&ctx.stream)?;
            }
            let end = ctx
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| format!("probe end event: {error:?}"))?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("probe batch: {error:?}"))?;
            let per_launch_us = f64::from(
                start
                    .elapsed_ms(&end)
                    .map_err(|error| format!("probe elapsed: {error:?}"))?,
            ) * 1_000.0
                / iterations as f64;
            eprintln!(
                "{LABEL} launch-cost grid={grid} threads=256 dynamic_shared={shared} per_launch_us={per_launch_us:.2}"
            );
        }
        Ok(())
    }

    /// Replays one arm's graph for a few seconds so an external sampler can
    /// read the clocks and power the arm settles at.
    #[test]
    #[ignore = "requires exclusive CC12.0/170-SM CUDA13.2 hardware"]
    fn streamk_sustained_probe() -> Result<(), String> {
        let label = std::env::var("STREAMK_PROBE_CELL")
            .map_err(|_| "STREAMK_PROBE_CELL names the cell label".to_string())?;
        let arm = std::env::var("STREAMK_PROBE_ARM")
            .map_err(|_| "STREAMK_PROBE_ARM is production or a candidate index".to_string())?;
        let seconds: f64 = std::env::var("STREAMK_PROBE_SECONDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(4.0);
        let cell = CELLS
            .iter()
            .copied()
            .find(|cell| cell.label == label)
            .ok_or_else(|| format!("unknown cell {label}"))?;
        let candidate = if arm == "production" {
            None
        } else {
            let index: usize = arm
                .parse()
                .map_err(|_| format!("candidate index expected, got {arm}"))?;
            Some(
                *CANDIDATES
                    .get(index)
                    .ok_or_else(|| format!("candidate index {index} out of range"))?,
            )
        };
        let device = GpuDevice::new(0)?;
        let ctx = configure(&device)?;
        validate_exact_environment(&ctx)?;
        let runtime = new_runtime(&device, ctx.stream.clone(), false)?;
        let spec = candidate.unwrap_or(production_spec());
        let timed = new_timed_arm(&runtime, cell, spec, candidate)?;
        for _ in 0..64 {
            launch_timed(&timed, Path::Graph)?;
        }
        synchronize(&runtime, "probe warmup")?;
        let started = std::time::Instant::now();
        let mut launches = 0_usize;
        let mut total_us = 0.0;
        while started.elapsed().as_secs_f64() < seconds {
            total_us += measure(&timed, Path::Graph, 200)? * 200.0;
            launches += 200;
        }
        eprintln!(
            "{LABEL} sustained cell={} arm={} grid={} launches={launches} mean_us={:.3}",
            cell.label,
            arm_name(candidate),
            spec.grid,
            total_us / launches as f64
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires ptxas and nvdisasm 13.2"]
    fn streamk_candidates_have_strict_ptxas_and_sass_resources() -> Result<(), String> {
        validate_ptxas_resources_and_sass()
    }

    #[test]
    #[ignore = "requires CC12.0 CUDA13.2 hardware"]
    fn streamk_candidates_have_strict_driver_resources() -> Result<(), String> {
        let device = GpuDevice::new(0)?;
        let ctx = configure(&device)?;
        validate_exact_environment(&ctx)?;
        validate_driver_resources(&device)
    }
}
