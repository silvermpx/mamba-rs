use std::ffi::OsStr;
use std::path::{Path, PathBuf};

#[cfg(feature = "cuda")]
mod common;

const CUDA_SOURCE_FILE: &str = "gemm_bi_scalar_wide_microtile_experiment.cu";
const SCREEN_WINDOWS: usize = 21;
const OFFICIAL_WINDOWS: usize = 101;
const MIN_SPEEDUP_P05: f64 = 1.01;
const MIN_SPEEDUP_P50: f64 = 1.01;
const BK: usize = 16;
const OPERAND_SALT: u64 = 0x4d64_1280;
const ORACLE_SAMPLES: usize = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Nn,
    Tn,
    Nt,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Self::Nn => "nn",
            Self::Tn => "tn",
            Self::Nt => "nt",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DriverResourceContract {
    registers: i32,
    occupancy: u32,
}

/// One experimental kernel: its symbol, operation, block geometry, and the
/// resource pins the box has to reproduce before any timing counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct KernelSpec {
    symbol: &'static str,
    op: Op,
    tile: (usize, usize),
    threads: u32,
    ptxas_registers: u64,
    driver: DriverResourceContract,
}

const fn kernel(
    symbol: &'static str,
    op: Op,
    tile: (usize, usize),
    threads: u32,
    ptxas_registers: u64,
    registers: i32,
    occupancy: u32,
) -> KernelSpec {
    KernelSpec {
        symbol,
        op,
        tile,
        threads,
        ptxas_registers,
        driver: DriverResourceContract {
            registers,
            occupancy,
        },
    }
}

const KERNELS: [KernelSpec; 12] = [
    kernel(
        "gemm_bi_nn_m128n64_bk16_s2_micro8x8_exp",
        Op::Nn,
        (128, 64),
        128,
        168,
        168,
        3,
    ),
    kernel(
        "gemm_bi_nn_m64n128_bk16_s2_micro8x8_exp",
        Op::Nn,
        (64, 128),
        128,
        166,
        166,
        3,
    ),
    kernel(
        "gemm_bi_nn_m64n64_bk16_s2_micro8x8_exp",
        Op::Nn,
        (64, 64),
        64,
        168,
        168,
        5,
    ),
    kernel(
        "gemm_bi_tn_m128n64_bk16_s2_micro8x8_exp",
        Op::Tn,
        (128, 64),
        128,
        168,
        168,
        3,
    ),
    kernel(
        "gemm_bi_tn_m64n128_bk16_s2_micro8x8_exp",
        Op::Tn,
        (64, 128),
        128,
        166,
        166,
        3,
    ),
    kernel(
        "gemm_bi_tn_m64n64_bk16_s2_micro8x8_exp",
        Op::Tn,
        (64, 64),
        64,
        168,
        168,
        5,
    ),
    kernel(
        "gemm_bi_nt_m128n64_bk16_s2_micro8x8_exp",
        Op::Nt,
        (128, 64),
        128,
        168,
        168,
        3,
    ),
    kernel(
        "gemm_bi_nt_m64n128_bk16_s2_micro8x8_exp",
        Op::Nt,
        (64, 128),
        128,
        168,
        168,
        3,
    ),
    kernel(
        "gemm_bi_nt_m64n64_bk16_s2_micro8x8_exp",
        Op::Nt,
        (64, 64),
        64,
        168,
        168,
        5,
    ),
    kernel(
        "gemm_bi_nt_m128n64_bk16_s2_micro8x8_kvec_exp",
        Op::Nt,
        (128, 64),
        128,
        226,
        226,
        2,
    ),
    kernel(
        "gemm_bi_nt_m64n128_bk16_s2_micro8x8_kvec_exp",
        Op::Nt,
        (64, 128),
        128,
        254,
        254,
        2,
    ),
    kernel(
        "gemm_bi_nt_m64n64_bk16_s2_micro8x8_kvec_exp",
        Op::Nt,
        (64, 64),
        64,
        226,
        226,
        4,
    ),
];

/// A candidate arm on one cell: which kernel and how many reduction splits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ArmSpec {
    kernel: usize,
    splits: u32,
}

/// One hot cell in the performance-matrix convention `(m, k, n)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    name: &'static str,
    op: Op,
    dims: (usize, usize, usize),
    production_launches: u32,
    production_symbol: Option<&'static str>,
    arms: [ArmSpec; 3],
}

const fn arm(kernel: usize, splits: u32) -> ArmSpec {
    ArmSpec { kernel, splits }
}

const PRODUCTION_NN_SYMBOL: &str = "gemm_bi_nn_m64n64_bk16_s2_v1";

const CELLS: [Cell; 9] = [
    Cell {
        name: "nn/d768_in_proj",
        op: Op::Nn,
        dims: (2_048, 768, 3_072),
        production_launches: 1,
        production_symbol: Some(PRODUCTION_NN_SYMBOL),
        arms: [arm(0, 1), arm(1, 1), arm(2, 1)],
    },
    Cell {
        name: "nn/d768_out_proj",
        op: Op::Nn,
        dims: (2_048, 1_536, 768),
        production_launches: 1,
        production_symbol: Some(PRODUCTION_NN_SYMBOL),
        arms: [arm(0, 4), arm(1, 4), arm(2, 4)],
    },
    Cell {
        name: "nn/prism_in_proj",
        op: Op::Nn,
        dims: (4_621, 384, 1_928),
        production_launches: 1,
        production_symbol: Some(PRODUCTION_NN_SYMBOL),
        arms: [arm(0, 1), arm(1, 1), arm(2, 1)],
    },
    Cell {
        name: "tn/d768_in_proj",
        op: Op::Tn,
        dims: (2_048, 768, 3_072),
        production_launches: 2,
        production_symbol: None,
        arms: [arm(3, 1), arm(4, 1), arm(4, 3)],
    },
    Cell {
        name: "tn/d768_out_proj",
        op: Op::Tn,
        dims: (2_048, 1_536, 768),
        production_launches: 2,
        production_symbol: None,
        arms: [arm(3, 2), arm(4, 2), arm(5, 2)],
    },
    Cell {
        name: "tn/prism_in_proj",
        op: Op::Tn,
        dims: (4_621, 384, 1_928),
        production_launches: 2,
        production_symbol: None,
        arms: [arm(3, 5), arm(4, 5), arm(3, 3)],
    },
    Cell {
        name: "nt/d768_in_proj",
        op: Op::Nt,
        dims: (2_048, 768, 3_072),
        production_launches: 2,
        production_symbol: None,
        arms: [arm(9, 5), arm(7, 5), arm(6, 5)],
    },
    Cell {
        name: "nt/d768_out_proj",
        op: Op::Nt,
        dims: (2_048, 1_536, 768),
        production_launches: 2,
        production_symbol: None,
        arms: [arm(7, 2), arm(6, 2), arm(9, 2)],
    },
    Cell {
        name: "nt/prism_in_proj",
        op: Op::Nt,
        dims: (4_621, 384, 1_928),
        production_launches: 2,
        production_symbol: None,
        arms: [arm(9, 3), arm(10, 3), arm(7, 2)],
    },
];

/// Kernel-side `(M, N, K)` of a cell: NN reduces over k, TN over m (the
/// output is k x n), NT over n (the output is m x k).
fn kernel_dims(op: Op, dims: (usize, usize, usize)) -> (usize, usize, usize) {
    let (m, k, n) = dims;
    match op {
        Op::Nn => (m, n, k),
        Op::Tn => (k, n, m),
        Op::Nt => (m, k, n),
    }
}

/// Row-major extents `(rows, columns)` of the A and B operands as stored.
fn operand_extents(op: Op, dims: (usize, usize, usize)) -> ((usize, usize), (usize, usize)) {
    let (m, k, n) = dims;
    match op {
        Op::Nn => ((m, k), (k, n)),
        Op::Tn => ((m, k), (m, n)),
        Op::Nt => ((m, n), (k, n)),
    }
}

fn tile_count(kernel_dims: (usize, usize, usize), tile: (usize, usize)) -> usize {
    kernel_dims.0.div_ceil(tile.0) * kernel_dims.1.div_ceil(tile.1)
}

fn k_tiles(kernel_dims: (usize, usize, usize)) -> usize {
    kernel_dims.2.div_ceil(BK)
}

fn tiles_per_split(kernel_dims: (usize, usize, usize), splits: u32) -> usize {
    k_tiles(kernel_dims).div_ceil(splits as usize)
}

fn shared_mem_bytes(tile: (usize, usize)) -> u32 {
    (2 * (tile.0 * BK + BK * tile.1) * 4 + 2 * 8) as u32
}

/// Every split must own at least one k tile, otherwise the unit count lies
/// about the parallelism.
fn validate_arm(cell: &Cell, spec: ArmSpec) -> Result<(), String> {
    let kernel = KERNELS.get(spec.kernel).ok_or_else(|| {
        format!(
            "{} names kernel {} which does not exist",
            cell.name, spec.kernel
        )
    })?;
    if kernel.op != cell.op {
        return Err(format!(
            "{} names {} whose operation does not match",
            cell.name, kernel.symbol
        ));
    }
    if spec.splits == 0 {
        return Err(format!("{} has a zero split count", cell.name));
    }
    let dims = kernel_dims(cell.op, cell.dims);
    let per_split = tiles_per_split(dims, spec.splits);
    if per_split * (spec.splits as usize - 1) >= k_tiles(dims) {
        return Err(format!(
            "{} with {} splits leaves an empty split on {}",
            cell.name, spec.splits, kernel.symbol
        ));
    }
    Ok(())
}

/// Reproduces the holder's operand seeding so raw candidate launches read
/// exactly the operands the production route was timed and checked on.
fn seeded_qualification_values(len: usize, salt: u64) -> Vec<f32> {
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

/// The chain the kernel walks for output `(r, c)`: with one split the
/// ascending-k fma chain from zero; with S splits each contiguous range of
/// k tiles from zero, folded as ((0 + p0) + p1) + ... in that order.
fn cpu_chain_bits(
    op: Op,
    a: &[f32],
    b: &[f32],
    kernel_dims: (usize, usize, usize),
    r: usize,
    c: usize,
    splits: u32,
) -> u32 {
    let (m, n, k) = kernel_dims;
    let a_at = |kk: usize| match op {
        Op::Tn => a[kk * m + r],
        Op::Nn | Op::Nt => a[r * k + kk],
    };
    let b_at = |kk: usize| match op {
        Op::Nt => b[c * k + kk],
        Op::Nn | Op::Tn => b[kk * n + c],
    };
    if splits == 1 {
        let mut acc = 0.0_f32;
        for kk in 0..k {
            acc = a_at(kk).mul_add(b_at(kk), acc);
        }
        return acc.to_bits();
    }
    let per_split = tiles_per_split(kernel_dims, splits) * BK;
    let mut folded = 0.0_f32;
    for s in 0..splits as usize {
        let begin = (s * per_split).min(k);
        let end = ((s + 1) * per_split).min(k);
        let mut acc = 0.0_f32;
        for kk in begin..end {
            acc = a_at(kk).mul_add(b_at(kk), acc);
        }
        folded += acc;
    }
    folded.to_bits()
}

fn candidate_source_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(CUDA_SOURCE_FILE)
}

fn read_candidate_source() -> Result<String, String> {
    let path = candidate_source_path();
    std::fs::read_to_string(&path)
        .map_err(|error| format!("read candidate CUDA source {}: {error}", path.display()))
}

fn exact_windows(value: Option<&OsStr>, required: usize) -> Result<usize, String> {
    if !matches!(required, SCREEN_WINDOWS | OFFICIAL_WINDOWS) {
        return Err(format!("unsupported qualification window count {required}"));
    }
    let actual = match value {
        Some(value) => value
            .to_str()
            .ok_or_else(|| "window override is not UTF-8".to_string())?
            .parse::<usize>()
            .map_err(|error| format!("parse qualification window override: {error}"))?,
        None => required,
    };
    if actual != required {
        return Err(format!(
            "qualification requires exactly {required} windows, received {actual}"
        ));
    }
    Ok(actual)
}

fn percentile(values: &[f64], quantile: f64) -> Result<f64, String> {
    if values.is_empty() || !quantile.is_finite() || !(0.0..=1.0).contains(&quantile) {
        return Err("invalid percentile request".into());
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("percentile samples must be finite and positive".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * quantile).floor() as usize;
    Ok(sorted[index])
}

fn speedup_stats(windows: usize, ratios: &[f64]) -> Result<(f64, f64, f64), String> {
    if !matches!(windows, SCREEN_WINDOWS | OFFICIAL_WINDOWS) || ratios.len() != windows {
        return Err("speedup sample count is not a frozen qualification window".into());
    }
    Ok((
        percentile(ratios, 0.05)?,
        percentile(ratios, 0.50)?,
        percentile(ratios, 0.95)?,
    ))
}

fn validate_speedup(windows: usize, ratios: &[f64]) -> Result<(f64, f64, f64), String> {
    let stats = speedup_stats(windows, ratios)?;
    if stats.0 < MIN_SPEEDUP_P05 || stats.1 < MIN_SPEEDUP_P50 {
        return Err(format!(
            "candidate speedup failed: p05={:.9} p50={:.9} requires p05>={MIN_SPEEDUP_P05:.3} p50>={MIN_SPEEDUP_P50:.3}",
            stats.0, stats.1
        ));
    }
    Ok(stats)
}

fn validate_ptxas_register_observation(expected: u64, registers: &[u64]) -> Result<(), String> {
    if registers != [expected] {
        return Err(format!(
            "register census is {registers:?}, expected={expected}"
        ));
    }
    Ok(())
}

fn validate_driver_resource_observation(
    contract: DriverResourceContract,
    registers: i32,
    occupancy: u32,
    local: i32,
    static_shared: i32,
) -> Result<(), String> {
    if registers != contract.registers
        || occupancy != contract.occupancy
        || local != 0
        || static_shared != 0
    {
        return Err(format!(
            "registers={registers} expected_registers={} occupancy={occupancy} expected_occupancy={} local={local} static_shared={static_shared}",
            contract.registers, contract.occupancy
        ));
    }
    Ok(())
}

#[test]
fn candidate_source_is_exact_one_owner_and_never_enters_production() {
    let source = read_candidate_source().expect("candidate CUDA source must exist");
    for spec in KERNELS {
        assert_eq!(
            source
                .matches(&format!("WIDE_DEFINE_KERNEL({}, ", spec.symbol))
                .count(),
            1,
            "{} must be instantiated exactly once",
            spec.symbol
        );
        let line = source
            .lines()
            .find(|line| line.contains(&format!("WIDE_DEFINE_KERNEL({}, ", spec.symbol)))
            .expect("instantiation line");
        let fields = line
            .trim_end_matches(')')
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>();
        assert_eq!(fields.len(), 9, "{line}");
        assert_eq!(
            fields[1],
            format!("WIDE_OP_{}", spec.op.name().to_ascii_uppercase())
        );
        assert_eq!(fields[2], spec.tile.0.to_string());
        assert_eq!(fields[3], spec.tile.1.to_string());
        assert_eq!(fields[6], spec.threads.to_string());
        assert_eq!(fields[7], spec.driver.occupancy.to_string());
        assert_eq!(
            fields[8],
            if spec.symbol.contains("_kvec_") {
                "true"
            } else {
                "false"
            }
        );
    }
    for required in [
        "#define WIDE_BK 16",
        "#define WIDE_TM 8",
        "#define WIDE_TN 8",
        "#define WIDE_K_PIPE 2",
        "__fmaf_rn(",
        "__fadd_rn(",
        "cp.async.bulk.tensor.2d",
        "mbarrier.try_wait.parity",
        "st.release.gpu.global.u32",
        "ld.acquire.gpu.global.u32",
        "Keep this row-major result nest in ascending reduction order.",
    ] {
        assert!(source.contains(required), "candidate omitted {required}");
    }
    let lower = source.to_ascii_lowercase();
    for forbidden in ["atomic", "mma.sync", "--use_fast_math"] {
        assert!(!lower.contains(forbidden), "candidate contains {forbidden}");
    }

    let modules = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
    for spec in KERNELS {
        assert!(!modules.contains(spec.symbol));
    }
    let production = include_str!("../kernels/gemm_bi_triad/scalar_nn_m64n64.cu");
    assert_eq!(
        production
            .matches(&format!("void {PRODUCTION_NN_SYMBOL}("))
            .count(),
        1
    );
}

#[test]
fn cells_name_matching_kernels_and_nonempty_splits() {
    let mut names = Vec::new();
    for cell in CELLS {
        assert!(!names.contains(&cell.name));
        names.push(cell.name);
        assert_eq!(cell.production_symbol.is_some(), cell.op == Op::Nn);
        assert_eq!(
            cell.production_launches,
            if cell.op == Op::Nn { 1 } else { 2 }
        );
        for spec in cell.arms {
            validate_arm(&cell, spec).unwrap();
        }
        let dims = kernel_dims(cell.op, cell.dims);
        let (a, b) = operand_extents(cell.op, cell.dims);
        assert_eq!(a.0 * a.1 * b.0 * b.1 / (dims.2 * dims.2), dims.0 * dims.1);
    }
    let cell = CELLS[0];
    assert_eq!(kernel_dims(cell.op, cell.dims), (2_048, 3_072, 768));
    assert_eq!(kernel_dims(Op::Tn, cell.dims), (768, 3_072, 2_048));
    assert_eq!(kernel_dims(Op::Nt, cell.dims), (2_048, 768, 3_072));
    assert_eq!(tile_count(kernel_dims(cell.op, cell.dims), (128, 64)), 768);
    assert_eq!(tile_count(kernel_dims(cell.op, cell.dims), (64, 64)), 1_536);
    assert_eq!(shared_mem_bytes((128, 64)), 24_592);
    assert_eq!(shared_mem_bytes((64, 64)), 16_400);
    assert_eq!(tiles_per_split((2_048, 768, 1_536), 4), 24);
    let too_many = Cell {
        arms: [arm(0, 97), arm(0, 1), arm(0, 1)],
        ..CELLS[0]
    };
    assert!(validate_arm(&too_many, too_many.arms[0]).is_err());
    let wrong_op = Cell {
        arms: [arm(3, 1), arm(0, 1), arm(0, 1)],
        ..CELLS[0]
    };
    assert!(validate_arm(&wrong_op, wrong_op.arms[0]).is_err());
}

#[test]
fn cpu_chain_matches_a_hand_fold_and_the_seeding_is_stable() {
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [0.5_f32, 0.25, 0.125, 0.0625];
    let dims = (1, 1, 4);
    let single = f32::from_bits(cpu_chain_bits(Op::Nn, &a, &b, dims, 0, 0, 1));
    assert_eq!(single, ((0.5 + 0.5) + 0.375) + 0.25);
    let tn = f32::from_bits(cpu_chain_bits(Op::Tn, &a, &b, dims, 0, 0, 1));
    let nt = f32::from_bits(cpu_chain_bits(Op::Nt, &a, &b, dims, 0, 0, 1));
    assert_eq!(tn, single);
    assert_eq!(nt, single);
    let seeded = seeded_qualification_values(8, OPERAND_SALT ^ 0x2d);
    assert_eq!(seeded, seeded_qualification_values(8, OPERAND_SALT ^ 0x2d));
    assert_ne!(seeded, seeded_qualification_values(8, OPERAND_SALT ^ 0x67));
    assert!(seeded.iter().all(|value| value.abs() <= 2.0));
}

#[test]
fn resource_observations_are_strict_and_fail_closed() {
    assert!(validate_ptxas_register_observation(94, &[94]).is_ok());
    for invalid in [vec![], vec![0], vec![93], vec![95], vec![94, 94]] {
        assert!(validate_ptxas_register_observation(94, &invalid).is_err());
    }
    let driver = DriverResourceContract {
        registers: 96,
        occupancy: 2,
    };
    assert!(validate_driver_resource_observation(driver, 96, 2, 0, 0).is_ok());
    for invalid in [
        (95, 2, 0, 0),
        (97, 2, 0, 0),
        (96, 1, 0, 0),
        (96, 3, 0, 0),
        (96, 2, 1, 0),
        (96, 2, 0, 1),
    ] {
        assert!(
            validate_driver_resource_observation(
                driver, invalid.0, invalid.1, invalid.2, invalid.3
            )
            .is_err()
        );
    }
}

#[test]
fn windows_percentiles_and_promotion_thresholds_are_fail_closed() {
    assert_eq!(exact_windows(None, 21).unwrap(), 21);
    assert_eq!(exact_windows(Some(OsStr::new("101")), 101).unwrap(), 101);
    for invalid in ["0", "1", "20", "22", "100"] {
        assert!(exact_windows(Some(OsStr::new(invalid)), 21).is_err());
    }
    assert!(percentile(&[], 0.5).is_err());
    for invalid in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        assert!(percentile(&[1.0, invalid], 0.5).is_err());
    }
    assert!(validate_speedup(21, &[1.009; 21]).is_err());
    assert!(validate_speedup(21, &[1.01; 21]).is_ok());
    assert!(validate_speedup(101, &vec![1.02; 101]).is_ok());
    assert!(speedup_stats(21, &[0.9; 21]).is_ok());
    assert!(speedup_stats(21, &[0.9; 20]).is_err());
}

#[cfg(feature = "cuda")]
mod cuda_experiment {
    use std::ffi::{CStr, c_void};
    use std::process::Command;
    use std::sync::Arc;

    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, sys,
    };
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationRequest, PhysicalQualificationRoute, QualifiedPhysicalLaunch,
        presize_physical_qualification_suite, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{ModuleKind, ResolvedGemmOp};
    use sha2::{Digest as _, Sha256};

    use super::common::gpu_quiet::QuietGpu;
    use super::{
        ArmSpec, BK, CELLS, Cell, KERNELS, KernelSpec, OFFICIAL_WINDOWS, OPERAND_SALT,
        ORACLE_SAMPLES, Op, SCREEN_WINDOWS, cpu_chain_bits, exact_windows, kernel_dims,
        operand_extents, percentile, read_candidate_source, seeded_qualification_values,
        shared_mem_bytes, speedup_stats, tile_count, tiles_per_split, validate_arm,
        validate_driver_resource_observation, validate_ptxas_register_observation,
        validate_speedup,
    };

    const GUARD_ELEMENTS: usize = 64;
    const INPUT_GUARD_BITS: u32 = 0x7fc1_a128;
    const OUTPUT_GUARD_BITS: u32 = 0x7fc1_c128;
    const SCRATCH_GUARD_BITS: u32 = 0x7fc1_e128;
    const CORRECTNESS_REPEATS: usize = 3;
    const TARGET_WINDOW_US: f64 = 10_000.0;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum PathKind {
        Eager,
        Graph,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Order {
        Abba,
        Baab,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct KernelParams {
        alpha: f32,
        beta: f32,
        m: i32,
        n: i32,
        k: i32,
        ldc: i32,
        splits: i32,
        tiles_per_split: i32,
    }

    unsafe impl DeviceRepr for KernelParams {}

    const _: [(); 32] = [(); std::mem::size_of::<KernelParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<KernelParams>()];

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    struct DirectTensorMap(sys::CUtensorMap);

    unsafe impl DeviceRepr for DirectTensorMap {}

    const _: [(); 128] = [(); std::mem::size_of::<DirectTensorMap>()];

    /// One 2d f32 tensor map with zero fill past the edges, boxed the way the
    /// kernel names its k tile. The NT B map carries the 64-byte swizzle the
    /// kernel undoes when it reads its rows.
    fn encode_tensor_map(
        pointer: u64,
        inner: usize,
        outer: usize,
        stride: usize,
        box_dimensions: [u32; 2],
        swizzle_64b: bool,
        label: &str,
    ) -> Result<DirectTensorMap, String> {
        if pointer == 0 || !pointer.is_multiple_of(16) || inner == 0 || outer == 0 || stride < inner
        {
            return Err(format!("invalid {label} tensor-map input"));
        }
        let byte_stride = stride
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("{label} byte stride overflow"))?;
        if !byte_stride.is_multiple_of(16) {
            return Err(format!("{label} byte stride is not 16-byte aligned"));
        }
        let dimensions = [inner as u64, outer as u64];
        let global_strides = [byte_stride as u64];
        let element_strides = [1_u32, 1_u32];
        let swizzle = if swizzle_64b {
            sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B
        } else {
            sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE
        };
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
                    swizzle,
                    sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                    sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
                )
            },
            &format!("encode {label} tensor map"),
        )?;
        Ok(DirectTensorMap(unsafe { raw.assume_init() }))
    }

    struct Runtime {
        production_ctx: GpuCtx,
        stream: Arc<CudaStream>,
        kernels: Vec<CudaFunction>,
    }

    struct GuardedBuffer {
        buffer: GpuBuffer,
        expected: Vec<f32>,
        active_offset: usize,
        active_len: usize,
        guard_bits: u32,
    }

    impl GuardedBuffer {
        fn new(
            stream: &Arc<CudaStream>,
            active: Vec<f32>,
            guard_bits: u32,
        ) -> Result<Self, String> {
            let active_offset = GUARD_ELEMENTS;
            let active_len = active.len();
            let total = active_offset
                .checked_add(active_len)
                .and_then(|value| value.checked_add(GUARD_ELEMENTS))
                .ok_or_else(|| "guarded allocation extent overflow".to_string())?;
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

        fn snapshot(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<f32>, String> {
            let actual = self.buffer.to_cpu(stream)?;
            for (index, value) in actual[..self.active_offset]
                .iter()
                .chain(&actual[self.active_offset + self.active_len..])
                .enumerate()
            {
                if value.to_bits() != self.guard_bits {
                    return Err(format!(
                        "{label} red zone {index} changed to 0x{:08x}",
                        value.to_bits()
                    ));
                }
            }
            Ok(actual)
        }

        fn bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
            let actual = self.snapshot(stream, label)?;
            Ok(
                actual[self.active_offset..self.active_offset + self.active_len]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect(),
            )
        }

        fn unchanged(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
            let actual = self.snapshot(stream, label)?;
            if actual
                .iter()
                .zip(&self.expected)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                return Err(format!("{label} read-only data changed"));
            }
            Ok(())
        }
    }

    /// One candidate arm bound to a cell: its kernel, launch shape, tensor
    /// maps over the fixture operands, and parameters.
    struct BoundArm {
        spec: ArmSpec,
        kernel: KernelSpec,
        config: LaunchConfig,
        slab_floats: usize,
        params: KernelParams,
        a_map: DirectTensorMap,
        b_map: DirectTensorMap,
        flag_count: usize,
    }

    struct Fixture {
        cell: Cell,
        kernel_dims: (usize, usize, usize),
        a: GuardedBuffer,
        b: GuardedBuffer,
        output: GuardedBuffer,
        partials: GuardedBuffer,
        flags: GpuBuffer,
        flag_capacity: usize,
        a_host: Vec<f32>,
        b_host: Vec<f32>,
        arms: Vec<BoundArm>,
    }

    fn compose_source() -> Result<String, String> {
        let candidate = read_candidate_source()?;
        Ok([
            include_str!("../kernels/_typed_prelude.cuh"),
            include_str!("../kernels/gemm_bi_triad/contract.cuh"),
            include_str!("../kernels/gemm_bi_triad/common.cuh"),
            candidate.as_str(),
        ]
        .iter()
        .map(|source| {
            source
                .lines()
                .filter(|line| !line.trim().starts_with("#include \""))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n"))
    }

    fn compile_ptx(target: &'static str) -> Result<String, String> {
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
        cudarc::nvrtc::compile_ptx_with_opts(compose_source()?, options)
            .map(|image| image.to_src())
            .map_err(|error| format!("compile wide-microtile experiment: {error:?}"))
    }

    fn resolved_op(op: Op) -> ResolvedGemmOp {
        match op {
            Op::Nn => ResolvedGemmOp::Nn,
            Op::Tn => ResolvedGemmOp::Tn,
            Op::Nt => ResolvedGemmOp::Nt,
        }
    }

    fn production_request(cell: &Cell) -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous(
            resolved_op(cell.op),
            cell.dims,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
        )
    }

    fn new_runtime() -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (12, 0) || identity.multiprocessor_count != 170 {
            return Err(format!(
                "wide-microtile experiment requires CC12.0/170SM, found CC{}.{} with {} SMs",
                identity.compute_capability.0,
                identity.compute_capability.1,
                identity.multiprocessor_count
            ));
        }
        let production_ctx = GpuCtx::new(&device)?;
        production_ctx.set_batch_invariant(true);
        production_ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        production_ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        let requests = CELLS.iter().map(production_request).collect::<Vec<_>>();
        presize_physical_qualification_suite(&production_ctx, &requests)?;

        let ptx = compile_ptx(device.nvrtc_target())?;
        validate_ptx(&ptx)?;
        let stream = device.fork_stream()?;
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|error| format!("load wide-microtile experiment module: {error:?}"))?;
        let mut kernels = Vec::with_capacity(KERNELS.len());
        for spec in KERNELS {
            let function = module
                .load_function(spec.symbol)
                .map_err(|error| format!("load {}: {error:?}", spec.symbol))?;
            function
                .set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes(spec.tile) as i32,
                )
                .map_err(|error| format!("set {} dynamic shared: {error:?}", spec.symbol))?;
            kernels.push(function);
        }
        Ok(Runtime {
            production_ctx,
            stream,
            kernels,
        })
    }

    fn production_holder<'a>(
        runtime: &'a Runtime,
        cell: &Cell,
    ) -> Result<QualifiedPhysicalLaunch<'a>, String> {
        let mut holder =
            qualify_physical_launch(&runtime.production_ctx, production_request(cell))?;
        holder.seed_f32_operands(&runtime.production_ctx, OPERAND_SALT)?;
        let evidence = holder.evidence();
        let nodes = evidence.nodes();
        if evidence.launch_count() != cell.production_launches
            || nodes.len() != cell.production_launches as usize
            || nodes
                .iter()
                .any(|node| node.launch.arguments_digest == [0; 32])
            || evidence.launch_digest() == [0; 32]
            || !evidence.eager_graph_equal()
        {
            return Err(format!(
                "{} production physical identity drifted: evidence={evidence:?}",
                cell.name
            ));
        }
        if let Some(symbol) = cell.production_symbol {
            let (m, n, _) = kernel_dims(cell.op, cell.dims);
            let grid = tile_count((m, n, 0), (64, 64)) as u32;
            if nodes[0].symbol != symbol
                || nodes[0].module_kind != ModuleKind::TriadScalar
                || nodes[0].launch.grid_dim != (grid, 1, 1)
                || nodes[0].launch.block_dim != (128, 1, 1)
                || nodes[0].launch.shared_mem_bytes != 17_408
            {
                return Err(format!(
                    "{} production single-launch identity drifted: evidence={evidence:?}",
                    cell.name
                ));
            }
        }
        eprintln!(
            "scalar_wide_microtile production cell={} launches={} symbols={:?}",
            cell.name,
            evidence.launch_count(),
            nodes.iter().map(|node| node.symbol).collect::<Vec<_>>()
        );
        Ok(holder)
    }

    fn bind_arm(
        fixture_dims: (usize, usize, usize),
        cell: &Cell,
        spec: ArmSpec,
        a: u64,
        b: u64,
    ) -> Result<BoundArm, String> {
        validate_arm(cell, spec)?;
        let kernel = KERNELS[spec.kernel];
        let (m, n, k) = fixture_dims;
        let (bm, bn) = kernel.tile;
        let (a_map, b_map) = match cell.op {
            Op::Nn => (
                encode_tensor_map(a, k, m, k, [BK as u32, bm as u32], false, "A")?,
                encode_tensor_map(b, n, k, n, [bn as u32, BK as u32], false, "B")?,
            ),
            Op::Tn => (
                encode_tensor_map(a, m, k, m, [bm as u32, BK as u32], false, "A")?,
                encode_tensor_map(b, n, k, n, [bn as u32, BK as u32], false, "B")?,
            ),
            Op::Nt => (
                encode_tensor_map(a, k, m, k, [BK as u32, bm as u32], false, "A")?,
                encode_tensor_map(b, k, n, k, [BK as u32, bn as u32], true, "B")?,
            ),
        };
        let tiles = tile_count(fixture_dims, kernel.tile);
        let units = tiles * spec.splits as usize;
        let config = LaunchConfig {
            grid_dim: (units as u32, 1, 1),
            block_dim: (kernel.threads, 1, 1),
            shared_mem_bytes: shared_mem_bytes(kernel.tile),
        };
        Ok(BoundArm {
            spec,
            kernel,
            config,
            slab_floats: tiles * bm * bn * (spec.splits as usize - 1),
            params: KernelParams {
                alpha: 1.0,
                beta: 0.0,
                m: m as i32,
                n: n as i32,
                k: k as i32,
                ldc: n as i32,
                splits: spec.splits as i32,
                tiles_per_split: tiles_per_split(fixture_dims, spec.splits) as i32,
            },
            a_map,
            b_map,
            flag_count: tiles * (spec.splits as usize - 1),
        })
    }

    fn new_fixture(runtime: &Runtime, cell: &Cell) -> Result<Fixture, String> {
        let dims = kernel_dims(cell.op, cell.dims);
        let (a_extent, b_extent) = operand_extents(cell.op, cell.dims);
        let a_host = seeded_qualification_values(a_extent.0 * a_extent.1, OPERAND_SALT ^ 0x2d);
        let b_host = seeded_qualification_values(b_extent.0 * b_extent.1, OPERAND_SALT ^ 0x67);
        let a = GuardedBuffer::new(&runtime.stream, a_host.clone(), INPUT_GUARD_BITS)?;
        let b = GuardedBuffer::new(&runtime.stream, b_host.clone(), INPUT_GUARD_BITS)?;
        let output = GuardedBuffer::new(
            &runtime.stream,
            vec![0.0; dims.0 * dims.1],
            OUTPUT_GUARD_BITS,
        )?;
        let mut arms = Vec::with_capacity(cell.arms.len());
        for spec in cell.arms {
            arms.push(bind_arm(
                dims,
                cell,
                spec,
                a.ptr(&runtime.stream),
                b.ptr(&runtime.stream),
            )?);
        }
        let slab_capacity = arms
            .iter()
            .map(|arm| arm.slab_floats)
            .max()
            .unwrap_or(0)
            .max(4);
        let partials = GuardedBuffer::new(
            &runtime.stream,
            vec![0.0; slab_capacity],
            SCRATCH_GUARD_BITS,
        )?;
        let flag_capacity = arms
            .iter()
            .map(|arm| arm.flag_count)
            .max()
            .unwrap_or(0)
            .max(1);
        let flags = GpuBuffer::from_cpu(&runtime.stream, &vec![0.0_f32; flag_capacity])?;
        Ok(Fixture {
            cell: *cell,
            kernel_dims: dims,
            a,
            b,
            output,
            partials,
            flags,
            flag_capacity,
            a_host,
            b_host,
            arms,
        })
    }

    fn launch(runtime: &Runtime, fixture: &Fixture, index: usize) -> Result<(), String> {
        let arm = &fixture.arms[index];
        let output = fixture.output.ptr(&runtime.stream);
        let bias = 0_u64;
        let partials = fixture.partials.ptr(&runtime.stream);
        let flags = fixture.flags.raw_ptr_at(&runtime.stream, 0);
        if [output, partials, flags]
            .into_iter()
            .any(|pointer| pointer == 0 || pointer & 15 != 0)
        {
            return Err("wide-microtile launch requires non-null 16-byte-aligned buffers".into());
        }
        if arm.flag_count > fixture.flag_capacity {
            return Err("flag buffer is smaller than the unit count".into());
        }
        let mut builder = runtime
            .stream
            .launch_builder(&runtime.kernels[arm.spec.kernel]);
        builder.arg(&output);
        builder.arg(&bias);
        builder.arg(&partials);
        builder.arg(&flags);
        builder.arg(&arm.params);
        builder.arg(&arm.a_map);
        builder.arg(&arm.b_map);
        unsafe { builder.launch(arm.config) }
            .map(|_| ())
            .map_err(|error| format!("launch {}: {error:?}", arm.kernel.symbol))
    }

    fn capture(runtime: &Runtime, fixture: &Fixture, index: usize) -> Result<CudaGraph, String> {
        unsafe { capture_into_graph(&runtime.stream, || launch(runtime, fixture, index)) }
    }

    fn cuda_ok(result: sys::CUresult, operation: &str) -> Result<(), String> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{operation}: {result:?}"))
        }
    }

    fn graph_identity(graph: &CudaGraph, arm: &BoundArm) -> Result<[u8; 32], String> {
        let symbol = arm.kernel.symbol;
        let raw = graph.cu_graph();
        let mut count = 0_usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, std::ptr::null_mut(), &mut count) },
            "query graph nodes",
        )?;
        if count != 1 {
            return Err(format!("{symbol} graph has {count} nodes"));
        }
        let mut edges = 0_usize;
        cuda_ok(
            unsafe {
                sys::cuGraphGetEdges_v2(
                    raw,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut edges,
                )
            },
            "query graph edges",
        )?;
        if edges != 0 {
            return Err(format!("{symbol} graph has {edges} edges"));
        }
        let mut node = std::ptr::null_mut();
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, &mut node, &mut count) },
            "read graph node",
        )?;
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        cuda_ok(
            unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
            "read graph node type",
        )?;
        if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            return Err(format!("{symbol} graph node is {kind:?}"));
        }
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            "read graph kernel params",
        )?;
        let mut name = std::ptr::null();
        cuda_ok(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "read graph function name",
        )?;
        if name.is_null() || params.kernelParams.is_null() {
            return Err(format!("{symbol} graph omitted function or arguments"));
        }
        let graph_symbol = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|error| format!("graph symbol is not UTF-8: {error}"))?;
        let config = arm.config;
        if graph_symbol != symbol
            || (params.gridDimX, params.gridDimY, params.gridDimZ) != config.grid_dim
            || (params.blockDimX, params.blockDimY, params.blockDimZ) != config.block_dim
            || params.sharedMemBytes != config.shared_mem_bytes
        {
            return Err(format!("{symbol} graph physical identity drifted"));
        }
        let mut digest = Sha256::new();
        digest.update(b"scalar-wide-microtile-graph-args.v2");
        digest.update(symbol.as_bytes());
        for (index, size) in [8_usize, 8, 8, 8, 32, 128, 128].into_iter().enumerate() {
            let pointer = unsafe { *params.kernelParams.add(index) };
            if pointer.is_null() {
                return Err(format!("{symbol} graph argument {index} is null"));
            }
            digest.update((size as u64).to_le_bytes());
            digest.update(unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), size) });
        }
        let digest: [u8; 32] = digest.finalize().into();
        if digest == [0; 32] {
            return Err(format!("{symbol} graph argument digest is zero"));
        }
        Ok(digest)
    }

    /// The oracle sample: a fixed pseudo-random set of outputs plus the four
    /// corners, so every tile edge and every split boundary is exercised.
    fn oracle_sample(dims: (usize, usize, usize)) -> Vec<(usize, usize)> {
        let (m, n, _) = dims;
        let mut state = 0x5eed_0777_u64;
        let mut sample = Vec::with_capacity(ORACLE_SAMPLES + 4);
        for _ in 0..ORACLE_SAMPLES {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let r = ((state >> 33) as usize) % m;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let c = ((state >> 33) as usize) % n;
            sample.push((r, c));
        }
        sample.extend([(0, 0), (m - 1, n - 1), (m - 1, 0), (0, n - 1)]);
        sample
    }

    fn validate_oracle(fixture: &Fixture, arm: &BoundArm, bits: &[u32]) -> Result<(), String> {
        let (_, n, _) = fixture.kernel_dims;
        let mut mismatches = 0_usize;
        for (r, c) in oracle_sample(fixture.kernel_dims) {
            let expected = cpu_chain_bits(
                fixture.cell.op,
                &fixture.a_host,
                &fixture.b_host,
                fixture.kernel_dims,
                r,
                c,
                arm.spec.splits,
            );
            if bits[r * n + c] != expected {
                mismatches += 1;
            }
        }
        if mismatches != 0 {
            return Err(format!(
                "{} {} splits={} differs from the CPU chain in {mismatches} sampled outputs",
                fixture.cell.name, arm.kernel.symbol, arm.spec.splits
            ));
        }
        Ok(())
    }

    fn validate_correctness(
        runtime: &Runtime,
        fixture: &mut Fixture,
        production_bits: &[u32],
    ) -> Result<(), String> {
        let mut digests = Vec::with_capacity(fixture.arms.len());
        let mut graphs = Vec::with_capacity(fixture.arms.len());
        for index in 0..fixture.arms.len() {
            let graph = capture(runtime, fixture, index)?;
            let digest = graph_identity(&graph, &fixture.arms[index])?;
            if digests.contains(&digest) {
                return Err(format!(
                    "{} physical argument digests collided",
                    fixture.cell.name
                ));
            }
            digests.push(digest);
            graphs.push(graph);
        }
        for (index, graph) in graphs.iter().enumerate() {
            let mut reference: Option<Vec<u32>> = None;
            for path in [PathKind::Eager, PathKind::Graph] {
                for repeat in 0..CORRECTNESS_REPEATS {
                    fixture.output.reset(&runtime.stream)?;
                    match path {
                        PathKind::Eager => launch(runtime, fixture, index)?,
                        PathKind::Graph => graph.launch().map_err(|error| {
                            format!(
                                "launch {} graph: {error:?}",
                                fixture.arms[index].kernel.symbol
                            )
                        })?,
                    }
                    runtime.stream.synchronize().map_err(|error| {
                        format!(
                            "synchronize {}: {error:?}",
                            fixture.arms[index].kernel.symbol
                        )
                    })?;
                    let arm = &fixture.arms[index];
                    let actual = fixture.output.bits(&runtime.stream, arm.kernel.symbol)?;
                    match reference.as_ref() {
                        None => {
                            validate_oracle(fixture, arm, &actual)?;
                            let differing = actual
                                .iter()
                                .zip(production_bits)
                                .filter(|(actual, expected)| actual != expected)
                                .count();
                            let same_family = fixture.cell.op == Op::Nn && arm.spec.splits == 1;
                            if same_family && differing != 0 {
                                return Err(format!(
                                    "{} {} differs from production bits in {differing} elements",
                                    fixture.cell.name, arm.kernel.symbol
                                ));
                            }
                            eprintln!(
                                "scalar_wide_microtile family cell={} symbol={} splits={} production_bit_differences={differing} same_family={same_family}",
                                fixture.cell.name, arm.kernel.symbol, arm.spec.splits
                            );
                        }
                        Some(reference) => {
                            if actual != *reference {
                                return Err(format!(
                                    "{} {} {path:?} repeat {repeat} is not deterministic",
                                    fixture.cell.name, arm.kernel.symbol
                                ));
                            }
                        }
                    }
                    if reference.is_none() {
                        reference = Some(actual);
                    }
                    fixture.a.unchanged(&runtime.stream, "A")?;
                    fixture.b.unchanged(&runtime.stream, "B")?;
                    fixture.partials.snapshot(&runtime.stream, "partials")?;
                    let flags = fixture.flags.to_cpu(&runtime.stream)?;
                    if flags.iter().any(|flag| flag.to_bits() != 0) {
                        return Err(format!(
                            "{} {} left a flag raised",
                            fixture.cell.name, arm.kernel.symbol
                        ));
                    }
                }
            }
            eprintln!(
                "scalar_wide_microtile exact cell={} symbol={} splits={} units={} oracle=ok determinism=ok",
                fixture.cell.name,
                fixture.arms[index].kernel.symbol,
                fixture.arms[index].spec.splits,
                fixture.arms[index].config.grid_dim.0
            );
        }
        Ok(())
    }

    fn validate_driver_resources(runtime: &Runtime) -> Result<(), String> {
        let mut failures = Vec::new();
        for (spec, function) in KERNELS.iter().zip(&runtime.kernels) {
            let registers = function
                .num_regs()
                .map_err(|error| format!("{} registers: {error:?}", spec.symbol))?;
            let local = function
                .local_size_bytes()
                .map_err(|error| format!("{} local bytes: {error:?}", spec.symbol))?;
            let static_shared = function
                .shared_size_bytes()
                .map_err(|error| format!("{} static shared: {error:?}", spec.symbol))?;
            let shared = shared_mem_bytes(spec.tile);
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(spec.threads, shared as usize, None)
                .map_err(|error| format!("{} occupancy: {error:?}", spec.symbol))?;
            eprintln!(
                "scalar_wide_microtile resource symbol={} threads={} dynamic_shared={shared} static_shared={static_shared} registers={registers} local={local} occupancy={occupancy}",
                spec.symbol, spec.threads
            );
            if let Err(detail) = validate_driver_resource_observation(
                spec.driver,
                registers,
                occupancy,
                local,
                static_shared,
            ) {
                failures.push(format!(
                    "{} violates exact SM120 resource contract: {detail}",
                    spec.symbol
                ));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }

    fn ptx_entry<'a>(ptx: &'a str, symbol: &str) -> Result<&'a str, String> {
        let marker = format!(".entry {symbol}(");
        let (_, tail) = ptx
            .split_once(&marker)
            .ok_or_else(|| format!("PTX omitted {symbol}"))?;
        Ok(tail
            .split_once(".visible .entry ")
            .map_or(tail, |(entry, _)| entry))
    }

    fn validate_ptx(ptx: &str) -> Result<(), String> {
        for spec in KERNELS {
            let entry = ptx_entry(ptx, spec.symbol)?;
            let parameters = entry
                .split_once("\n)")
                .map(|(parameters, _)| parameters)
                .ok_or_else(|| format!("{} PTX parameter list is malformed", spec.symbol))?;
            if parameters.matches(".param").count() != 7
                || !entry.contains("fma.rn.f32")
                || !entry.contains("cp.async.bulk.tensor.2d")
            {
                return Err(format!("{} PTX ABI or FFMA contract changed", spec.symbol));
            }
            for token in [
                "atom.", "atom::", "red.", "red::", "redux.", "mma.", ".ftz", "call.",
            ] {
                if entry
                    .split_ascii_whitespace()
                    .any(|field| field.starts_with(token) || field.contains(".ftz"))
                {
                    return Err(format!("{} PTX contains forbidden {token}", spec.symbol));
                }
            }
        }
        Ok(())
    }

    fn metric_before(line: &str, suffix: &str) -> Option<u64> {
        line.split_once(suffix)?
            .0
            .split(|character: char| !character.is_ascii_digit())
            .rfind(|field| !field.is_empty())?
            .parse()
            .ok()
    }

    fn command_output(mut command: Command, label: &str) -> Result<std::process::Output, String> {
        let output = command
            .output()
            .map_err(|error| format!("run {label}: {error}"))?;
        if output.status.success() {
            Ok(output)
        } else {
            Err(format!(
                "{label} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
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

    fn opcode_census(entry: &str) -> String {
        let mut counts: Vec<(&str, usize)> = Vec::new();
        for line in entry.lines() {
            let Some((_, instruction)) = line.split_once("*/") else {
                continue;
            };
            let Some(mnemonic) = instruction
                .split_ascii_whitespace()
                .find(|field| !field.starts_with('@'))
            else {
                continue;
            };
            let mnemonic = mnemonic.split('.').next().unwrap_or(mnemonic);
            match counts.iter_mut().find(|(name, _)| *name == mnemonic) {
                Some((_, count)) => *count += 1,
                None => counts.push((mnemonic, 1)),
            }
        }
        counts.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(right.0)));
        counts
            .iter()
            .take(12)
            .map(|(name, count)| format!("{name}={count}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn validate_ptxas_and_sass(target: &str, ptx: &str) -> Result<(), String> {
        validate_ptx(ptx)?;
        let nonce = std::process::id();
        let stem = format!("mamba-wide-microtile-{target}-{nonce}");
        let directory = std::env::temp_dir();
        let ptx_path = directory.join(format!("{stem}.ptx"));
        let cubin_path = directory.join(format!("{stem}.cubin"));
        std::fs::write(&ptx_path, ptx).map_err(|error| format!("write PTX: {error}"))?;
        let output = command_output(
            {
                let mut command = Command::new("ptxas");
                command
                    .arg(format!("--gpu-name={target}"))
                    .arg("--verbose")
                    .arg(&ptx_path)
                    .arg("--output-file")
                    .arg(&cubin_path);
                command
            },
            "ptxas",
        );
        let result = (|| {
            let output = output?;
            let report = format!(
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let disassembly = command_output(
                {
                    let mut command = Command::new("nvdisasm");
                    command.arg(&cubin_path);
                    command
                },
                "nvdisasm",
            )?;
            let sass = String::from_utf8_lossy(&disassembly.stdout);
            let mut failures = Vec::new();
            for spec in KERNELS {
                let marker = format!("Compiling entry function '{}'", spec.symbol);
                let block = report
                    .split_once(&marker)
                    .and_then(|(_, tail)| tail.split("Compiling entry function '").next())
                    .ok_or_else(|| format!("ptxas omitted {} resource record", spec.symbol))?;
                let mut spills = Vec::new();
                for metric in [
                    " bytes stack frame",
                    " bytes spill stores",
                    " bytes spill loads",
                ] {
                    let values = block
                        .lines()
                        .filter_map(|line| metric_before(line, metric))
                        .collect::<Vec<_>>();
                    spills.push(values.clone());
                    if values != [0] {
                        failures.push(format!(
                            "{} {target} has nonzero or ambiguous {metric}: {values:?}",
                            spec.symbol
                        ));
                    }
                }
                let registers = block
                    .lines()
                    .filter_map(|line| metric_before(line, " registers"))
                    .collect::<Vec<_>>();
                let entry = sass_entry(&sass, spec.symbol)?;
                eprintln!(
                    "scalar_wide_microtile ptxas target={target} symbol={} registers={registers:?} stack_spill_stores_loads={spills:?} sass: {}",
                    spec.symbol,
                    opcode_census(entry)
                );
                if let Err(detail) =
                    validate_ptxas_register_observation(spec.ptxas_registers, &registers)
                {
                    failures.push(format!("{} {target} {detail}", spec.symbol));
                }
                if !entry.contains("FFMA") || !entry.contains("UTMALDG") {
                    failures.push(format!("{} SASS omitted FFMA or UTMALDG", spec.symbol));
                }
                for forbidden in [" LDL", " STL", " ATOM", " RED", " REDUX", " LDGSTS"] {
                    if entry.contains(forbidden) {
                        failures.push(format!("{} SASS contains {forbidden}", spec.symbol));
                    }
                }
            }
            if failures.is_empty() {
                Ok(())
            } else {
                Err(failures.join("; "))
            }
        })();
        let _ = std::fs::remove_file(ptx_path);
        let _ = std::fs::remove_file(cubin_path);
        result
    }

    struct TimingFixture {
        fixture: Fixture,
        graphs: Vec<CudaGraph>,
    }

    fn timing_fixture(runtime: &Runtime, cell: &Cell) -> Result<TimingFixture, String> {
        let fixture = new_fixture(runtime, cell)?;
        let mut graphs = Vec::with_capacity(fixture.arms.len());
        for index in 0..fixture.arms.len() {
            let graph = capture(runtime, &fixture, index)?;
            graph_identity(&graph, &fixture.arms[index])?;
            graphs.push(graph);
        }
        Ok(TimingFixture { fixture, graphs })
    }

    fn measure_candidate(
        runtime: &Runtime,
        timing: &TimingFixture,
        index: usize,
        path: PathKind,
        iterations: usize,
    ) -> Result<f64, String> {
        if iterations == 0 {
            return Err("timing iterations must be nonzero".into());
        }
        let symbol = timing.fixture.arms[index].kernel.symbol;
        let start = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record start event: {error:?}"))?;
        for _ in 0..iterations {
            match path {
                PathKind::Eager => launch(runtime, &timing.fixture, index)?,
                PathKind::Graph => timing.graphs[index]
                    .launch()
                    .map_err(|error| format!("launch {symbol} graph: {error:?}"))?,
            }
        }
        let end = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record end event: {error:?}"))?;
        let us = start
            .elapsed_ms(&end)
            .map(|milliseconds| f64::from(milliseconds) * 1_000.0 / iterations as f64)
            .map_err(|error| format!("measure {symbol}: {error:?}"))?;
        if us.is_finite() && us > 0.0 {
            Ok(us)
        } else {
            Err(format!("{symbol} returned invalid timing {us}"))
        }
    }

    fn measure_production(
        runtime: &Runtime,
        holder: &mut QualifiedPhysicalLaunch<'_>,
        path: PathKind,
        iterations: usize,
    ) -> Result<f64, String> {
        let ms = match path {
            PathKind::Eager => {
                holder.measure_eager_window_ms(&runtime.production_ctx, iterations)?
            }
            PathKind::Graph => {
                holder.measure_graph_window_ms(&runtime.production_ctx, iterations)?
            }
        };
        let us = ms * 1_000.0 / iterations as f64;
        if us.is_finite() && us > 0.0 {
            Ok(us)
        } else {
            Err(format!("production returned invalid timing {us}"))
        }
    }

    fn adjacent(
        runtime: &Runtime,
        holder: &mut QualifiedPhysicalLaunch<'_>,
        timing: &TimingFixture,
        index: usize,
        path: PathKind,
        order: Order,
        iterations: usize,
    ) -> Result<(f64, f64), String> {
        let sequence = match order {
            Order::Abba => [true, false, false, true],
            Order::Baab => [false, true, true, false],
        };
        let mut production = Vec::with_capacity(2);
        let mut candidate = Vec::with_capacity(2);
        for is_production in sequence {
            if is_production {
                production.push(measure_production(runtime, holder, path, iterations)?);
            } else {
                candidate.push(measure_candidate(runtime, timing, index, path, iterations)?);
            }
        }
        Ok((
            production.iter().sum::<f64>() / 2.0,
            candidate.iter().sum::<f64>() / 2.0,
        ))
    }

    /// One paired cohort for one arm on one cell. Reports both p50s and the
    /// speedup percentiles, then returns the speedup stats so the caller
    /// can decide which arm carries the cell.
    #[derive(Clone, Copy)]
    struct Cohort {
        index: usize,
        path: PathKind,
        order: Order,
        windows: usize,
    }

    fn paired(
        runtime: &Runtime,
        holder: &mut QualifiedPhysicalLaunch<'_>,
        timing: &TimingFixture,
        cohort: Cohort,
        quiet: &QuietGpu,
    ) -> Result<(f64, f64, f64), String> {
        let Cohort {
            index,
            path,
            order,
            windows,
        } = cohort;
        let cell = timing.fixture.cell;
        let arm = &timing.fixture.arms[index];
        let production_probe = measure_production(runtime, holder, path, 4)?;
        let candidate_probe = measure_candidate(runtime, timing, index, path, 4)?;
        let iterations = (TARGET_WINDOW_US / production_probe.min(candidate_probe)).ceil() as usize;
        if iterations == 0 {
            return Err("timing calibration returned zero iterations".into());
        }
        let label = format!(
            "scalar-wide-microtile/{}/{}/s{}/{path:?}/{order:?}/{windows}",
            cell.name, arm.kernel.symbol, arm.spec.splits
        );
        quiet.require_cohort(&label)?;
        let body = (|| {
            let mut production_us = Vec::with_capacity(windows);
            let mut candidate_us = Vec::with_capacity(windows);
            let mut ratios = Vec::with_capacity(windows);
            for _ in 0..windows {
                let (production, measured) =
                    adjacent(runtime, holder, timing, index, path, order, iterations)?;
                if !production.is_finite()
                    || production <= 0.0
                    || !measured.is_finite()
                    || measured <= 0.0
                {
                    return Err(format!(
                        "invalid paired sample production={production} candidate={measured}"
                    ));
                }
                production_us.push(production);
                candidate_us.push(measured);
                ratios.push(production / measured);
            }
            let stats = speedup_stats(windows, &ratios)?;
            let (m, n, k) = timing.fixture.kernel_dims;
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            let production_p50 = percentile(&production_us, 0.50)?;
            let candidate_p50 = percentile(&candidate_us, 0.50)?;
            eprintln!(
                "scalar_wide_microtile cell={} symbol={} splits={} path={path:?} order={order:?} windows={windows} production_us_p50={production_p50:.6} candidate_us_p50={candidate_p50:.6} production_tflops={:.2} candidate_tflops={:.2} speedup_p05={:.9} speedup_p50={:.9} speedup_p95={:.9}",
                cell.name,
                arm.kernel.symbol,
                arm.spec.splits,
                flops / production_p50 / 1.0e6,
                flops / candidate_p50 / 1.0e6,
                stats.0,
                stats.1,
                stats.2
            );
            Ok(stats)
        })();
        super::verify_post_cohort_even_on_error(body, quiet.verify_post_cohort(&label).map(drop))
    }

    fn run_timing(windows: usize) -> Result<(), String> {
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        let label = format!("scalar-wide-microtile-{windows}");
        quiet.require_pre_context(&label)?;
        let body = (|| {
            let runtime = new_runtime()?;
            validate_driver_resources(&runtime)?;
            let mut failures = Vec::new();
            for cell in CELLS.iter() {
                let mut holder = production_holder(&runtime, cell)?;
                holder.measure_graph_window_ms(&runtime.production_ctx, 1)?;
                holder.measure_eager_window_ms(&runtime.production_ctx, 1)?;
                let timing = timing_fixture(&runtime, cell)?;
                let mut best: Option<(usize, f64)> = None;
                for index in 0..timing.fixture.arms.len() {
                    let mut worst_p50 = f64::INFINITY;
                    for path in [PathKind::Eager, PathKind::Graph] {
                        for order in [Order::Abba, Order::Baab] {
                            let stats = paired(
                                &runtime,
                                &mut holder,
                                &timing,
                                Cohort {
                                    index,
                                    path,
                                    order,
                                    windows,
                                },
                                &quiet,
                            )?;
                            worst_p50 = worst_p50.min(stats.1);
                        }
                    }
                    if best.is_none_or(|(_, p50)| worst_p50 > p50) {
                        best = Some((index, worst_p50));
                    }
                }
                let (index, p50) = best.ok_or_else(|| "no arm was timed".to_string())?;
                let arm = &timing.fixture.arms[index];
                eprintln!(
                    "scalar_wide_microtile best cell={} symbol={} splits={} worst_cohort_speedup_p50={p50:.9}",
                    cell.name, arm.kernel.symbol, arm.spec.splits
                );
                if let Err(detail) = validate_speedup(windows, &vec![p50; windows]) {
                    failures.push(format!("{}: {detail}", cell.name));
                }
                drop(timing);
                drop(holder);
            }
            if failures.is_empty() {
                Ok(())
            } else {
                Err(failures.join("; "))
            }
        })();
        super::verify_post_cohort_even_on_error(body, quiet.verify_post_cohort(&label).map(drop))
    }

    #[test]
    #[ignore = "requires CUDA 13.2 NVRTC but launches no GPU work"]
    fn candidate_compiles_for_sm120_with_exact_ptx() -> Result<(), String> {
        validate_ptx(&compile_ptx("compute_120")?)
    }

    #[test]
    #[ignore = "requires CUDA 13.2 NVRTC, ptxas, and nvdisasm but launches no GPU work"]
    fn candidate_meets_sm120_ptxas_sass_and_resource_contracts() -> Result<(), String> {
        validate_ptxas_and_sass("sm_120", &compile_ptx("compute_120")?)
    }

    #[test]
    #[ignore = "requires an exclusive exact SM120 CUDA13.2 GPU"]
    fn candidate_is_exact_guarded_and_graph_stable() -> Result<(), String> {
        let runtime = new_runtime()?;
        validate_driver_resources(&runtime)?;
        for cell in CELLS.iter() {
            let mut holder = production_holder(&runtime, cell)?;
            holder.measure_eager_window_ms(&runtime.production_ctx, 1)?;
            let production_bits = holder.f32_output_bits(&runtime.production_ctx)?;
            let mut fixture = new_fixture(&runtime, cell)?;
            validate_correctness(&runtime, &mut fixture, &production_bits)?;
            drop(fixture);
            drop(holder);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive quiet exact SM120 CUDA13.2 GPU and release build"]
    fn candidate_screen_21_windows_abba_baab() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("timing requires --release".into());
        }
        let windows = exact_windows(
            std::env::var_os("MAMBA_RS_WIDE_MICROTILE_WINDOWS").as_deref(),
            SCREEN_WINDOWS,
        )?;
        run_timing(windows)
    }

    #[test]
    #[ignore = "requires an exclusive quiet exact SM120 CUDA13.2 GPU and release build"]
    fn candidate_official_101_windows_abba_baab() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("official timing requires --release".into());
        }
        let windows = exact_windows(
            std::env::var_os("MAMBA_RS_WIDE_MICROTILE_WINDOWS").as_deref(),
            OFFICIAL_WINDOWS,
        )?;
        run_timing(windows)
    }
}

fn verify_post_cohort_even_on_error<T>(
    body: Result<T, String>,
    postflight: Result<(), String>,
) -> Result<T, String> {
    match (body, postflight) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(body), Ok(())) => Err(body),
        (Ok(_), Err(postflight)) => Err(postflight),
        (Err(body), Err(postflight)) => Err(format!("{body}; quiet-GPU postflight: {postflight}")),
    }
}

#[test]
fn postflight_errors_are_never_hidden_by_body_results() {
    assert_eq!(verify_post_cohort_even_on_error(Ok(7), Ok(())), Ok(7));
    assert_eq!(
        verify_post_cohort_even_on_error::<()>(Err("body".into()), Ok(())).unwrap_err(),
        "body"
    );
    assert_eq!(
        verify_post_cohort_even_on_error(Ok(()), Err("post".into())).unwrap_err(),
        "post"
    );
    let both =
        verify_post_cohort_even_on_error::<()>(Err("body".into()), Err("post".into())).unwrap_err();
    assert!(both.contains("body") && both.contains("post"));
}
