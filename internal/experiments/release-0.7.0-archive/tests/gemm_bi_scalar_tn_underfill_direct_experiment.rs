use std::ffi::OsStr;
use std::path::{Path, PathBuf};

#[cfg(feature = "cuda")]
mod common;

const DIMS: (usize, usize, usize) = (256, 512, 384);
const SCREEN_WINDOWS: usize = 21;
const OFFICIAL_WINDOWS: usize = 101;
#[cfg(feature = "cuda")]
const EXCEPTIONAL_OUTPUT_DIGESTS: [[u8; 32]; 8] = [
    [
        67, 201, 29, 95, 111, 167, 125, 123, 168, 171, 90, 216, 199, 115, 129, 15, 53, 78, 144,
        195, 115, 112, 50, 253, 212, 247, 165, 90, 217, 38, 171, 247,
    ],
    [
        140, 82, 65, 140, 142, 187, 207, 60, 185, 236, 83, 151, 212, 129, 44, 223, 100, 228, 74, 8,
        25, 100, 200, 8, 196, 113, 155, 12, 199, 130, 134, 235,
    ],
    [
        76, 246, 220, 67, 15, 56, 72, 84, 48, 144, 66, 97, 87, 64, 194, 194, 170, 92, 213, 127,
        184, 160, 224, 78, 190, 216, 251, 174, 137, 197, 32, 203,
    ],
    [
        49, 99, 226, 16, 188, 0, 122, 218, 54, 231, 91, 76, 193, 194, 199, 242, 123, 6, 51, 97,
        114, 117, 172, 88, 91, 72, 52, 153, 126, 246, 42, 19,
    ],
    [
        49, 170, 190, 10, 68, 162, 227, 68, 18, 227, 41, 146, 209, 214, 120, 161, 35, 140, 25, 240,
        86, 57, 106, 87, 182, 35, 183, 245, 77, 107, 220, 144,
    ],
    [
        49, 170, 190, 10, 68, 162, 227, 68, 18, 227, 41, 146, 209, 214, 120, 161, 35, 140, 25, 240,
        86, 57, 106, 87, 182, 35, 183, 245, 77, 107, 220, 144,
    ],
    [
        34, 223, 212, 209, 192, 141, 57, 134, 155, 164, 26, 250, 173, 120, 232, 101, 134, 141, 185,
        41, 116, 80, 136, 109, 193, 232, 113, 242, 188, 32, 166, 169,
    ],
    [
        34, 223, 212, 209, 192, 141, 57, 134, 155, 164, 26, 250, 173, 120, 232, 101, 134, 141, 185,
        41, 116, 80, 136, 109, 193, 232, 113, 242, 188, 32, 166, 169,
    ],
];
#[cfg(feature = "cuda")]
const MIN_WINNER_EAGER_P05: f64 = 1.002;
#[cfg(feature = "cuda")]
const MIN_WINNER_EAGER_P50: f64 = 1.003;
#[cfg(feature = "cuda")]
const MIN_WINNER_GRAPH_P05: f64 = 0.995;
#[cfg(feature = "cuda")]
const MIN_WINNER_GRAPH_P50: f64 = 0.998;
const CUDA_SOURCE_FILE: &str = "gemm_bi_scalar_tn_underfill_direct_experiment.cu";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CandidateSpec {
    symbol: &'static str,
    tile: (u32, u32),
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    dynamic_shared_bytes: u32,
    register_cap: u32,
    minimum_occupancy: u32,
}

const M16N32: CandidateSpec = CandidateSpec {
    symbol: "gemm_bi_tn_underfill_m16n32_bk16_s2_splitm16_exp_v1",
    tile: (16, 32),
    grid: (384, 1, 1),
    block: (128, 1, 1),
    dynamic_shared_bytes: 6_656,
    register_cap: 112,
    minimum_occupancy: 4,
};

const M32N32: CandidateSpec = CandidateSpec {
    symbol: "gemm_bi_tn_underfill_m32n32_bk16_s2_splitm16_exp_v1",
    tile: (32, 32),
    grid: (16, 12, 1),
    block: (128, 1, 1),
    dynamic_shared_bytes: 8_192,
    register_cap: 112,
    minimum_occupancy: 4,
};

const M16N32_APAD0_CA: CandidateSpec = CandidateSpec {
    symbol: "gemm_bi_tn_underfill_m16n32_bk16_s2_splitm16_apad0_ca_exp_v1",
    tile: (16, 32),
    grid: (384, 1, 1),
    block: (128, 1, 1),
    dynamic_shared_bytes: 6_144,
    register_cap: 112,
    minimum_occupancy: 4,
};

const M16N32_APAD0_CG: CandidateSpec = CandidateSpec {
    symbol: "gemm_bi_tn_underfill_m16n32_bk16_s2_splitm16_apad0_cg_exp_v1",
    tile: (16, 32),
    grid: (384, 1, 1),
    block: (128, 1, 1),
    dynamic_shared_bytes: 6_144,
    register_cap: 112,
    minimum_occupancy: 4,
};

const M16N16: CandidateSpec = CandidateSpec {
    symbol: "gemm_bi_tn_underfill_m16n16_bk16_s2_splitm16_exp_v1",
    tile: (16, 16),
    grid: (768, 1, 1),
    block: (64, 1, 1),
    dynamic_shared_bytes: 4_096,
    register_cap: 112,
    minimum_occupancy: 4,
};

const M8N32: CandidateSpec = CandidateSpec {
    symbol: "gemm_bi_tn_underfill_m8n32_bk16_s2_splitm16_exp_v1",
    tile: (8, 32),
    grid: (768, 1, 1),
    block: (64, 1, 1),
    dynamic_shared_bytes: 5_120,
    register_cap: 112,
    minimum_occupancy: 4,
};

const INITIAL_WINNER: CandidateSpec = M16N32;
const INITIAL_RUNNER: CandidateSpec = M32N32;
const INITIAL_CANDIDATES: [CandidateSpec; 2] = [INITIAL_WINNER, INITIAL_RUNNER];
const NEXT_CHALLENGERS: [CandidateSpec; 4] = [M16N32_APAD0_CA, M16N32_APAD0_CG, M16N16, M8N32];
const CANDIDATES: [CandidateSpec; 6] = [
    M16N32,
    M32N32,
    M16N32_APAD0_CA,
    M16N32_APAD0_CG,
    M16N16,
    M8N32,
];
#[cfg(feature = "cuda")]
const OFFICIAL_WINNER: CandidateSpec = M16N16;
#[cfg(feature = "cuda")]
const OFFICIAL_RUNNER: CandidateSpec = M8N32;
#[cfg(feature = "cuda")]
const OFFICIAL_CANDIDATES: [CandidateSpec; 2] = [OFFICIAL_WINNER, OFFICIAL_RUNNER];

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

fn validate_exact_dims(dims: (i32, i32, i32)) -> Result<(), String> {
    if dims == (256, 512, 384) {
        Ok(())
    } else {
        Err(format!(
            "TN underfill direct candidates require (m,k,n)=(256,512,384), found {dims:?}"
        ))
    }
}

fn exact_windows(value: Option<&OsStr>, required: usize) -> Result<usize, String> {
    if !matches!(required, SCREEN_WINDOWS | OFFICIAL_WINDOWS) {
        return Err(format!("unsupported required window count {required}"));
    }
    let Some(value) = value else {
        return Ok(required);
    };
    let value = value
        .to_str()
        .ok_or_else(|| "MAMBA_RS_TN_UNDERFILL_DIRECT_WINDOWS is not Unicode".to_string())?;
    let parsed = value.parse::<usize>().map_err(|error| {
        format!("invalid MAMBA_RS_TN_UNDERFILL_DIRECT_WINDOWS={value}: {error}")
    })?;
    if parsed != required {
        return Err(format!(
            "MAMBA_RS_TN_UNDERFILL_DIRECT_WINDOWS must equal {required}, found {parsed}"
        ));
    }
    Ok(parsed)
}

fn percentile(values: &[f64], fraction: f64) -> Result<f64, String> {
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
    percentiles: (f64, f64, f64),
    minimums: (f64, f64),
) -> Result<(), String> {
    let (p05, p50, p95) = percentiles;
    let (minimum_p05, minimum_p50) = minimums;
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

fn entry_signature<'a>(source: &'a str, symbol: &str) -> Result<&'a str, String> {
    let marker = format!("void {symbol}(");
    let tail = source
        .split_once(&marker)
        .map(|(_, tail)| tail)
        .ok_or_else(|| format!("CUDA source omitted {symbol}"))?;
    tail.split_once(')')
        .map(|(signature, _)| signature)
        .ok_or_else(|| format!("CUDA source omitted {symbol} parameter terminator"))
}

fn ptx_entry<'a>(ptx: &'a str, symbol: &str) -> Result<&'a str, String> {
    let marker = format!(".visible .entry {symbol}(");
    let tail = ptx
        .split_once(&marker)
        .map(|(_, tail)| tail)
        .ok_or_else(|| format!("PTX omitted {symbol}"))?;
    let next = tail
        .find("\n.visible .entry ")
        .or_else(|| tail.find("\n.weak .entry "))
        .unwrap_or(tail.len());
    Ok(&tail[..next])
}

fn ptx_instruction_opcode(line: &str) -> Option<&str> {
    let mut tokens = line.split("//").next()?.split_whitespace();
    let mut opcode = tokens.next()?;
    while opcode.starts_with('@') || opcode.ends_with(':') {
        opcode = tokens.next()?;
    }
    if opcode.starts_with('.') || matches!(opcode, "{" | "}" | ")") {
        return None;
    }
    Some(opcode.trim_end_matches(';'))
}

fn validate_candidate_ptx(ptx: &str) -> Result<(), String> {
    for spec in CANDIDATES {
        let entry = ptx_entry(ptx, spec.symbol)?;
        let parameters = entry
            .split_once("\n)")
            .map(|(parameters, _)| parameters)
            .ok_or_else(|| format!("{} PTX parameter list is malformed", spec.symbol))?;
        let declarations = parameters
            .lines()
            .filter(|line| line.trim_start().starts_with(".param "))
            .collect::<Vec<_>>();
        if declarations.len() != 7
            || !declarations[..3].iter().all(|line| line.contains(".u64 "))
            || !declarations[3].contains(".f32 ")
            || !declarations[4..].iter().all(|line| line.contains(".u32 "))
        {
            return Err(format!(
                "{} has the wrong seven-argument PTX ABI",
                spec.symbol
            ));
        }
        for required in ["fma.rn.f32", "add.rn.f64", "mul.rn.f64"] {
            if !entry.contains(required) {
                return Err(format!("{} PTX omitted {required}", spec.symbol));
            }
        }
        for opcode in entry.lines().filter_map(ptx_instruction_opcode) {
            let opcode = opcode.to_ascii_lowercase();
            if opcode.starts_with("atom.")
                || opcode.starts_with("atom::")
                || opcode.starts_with("red.")
                || opcode.starts_with("red::")
                || opcode.starts_with("redux.")
                || opcode.starts_with("mma.")
                || opcode.contains(".ftz")
                || opcode.starts_with("call.")
            {
                return Err(format!(
                    "{} PTX contains forbidden opcode {opcode}",
                    spec.symbol
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResourceContract {
    threads: u32,
    static_shared: usize,
    dynamic_shared: usize,
    register_cap: usize,
    minimum_occupancy: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResourceObservation {
    threads: u32,
    static_shared: usize,
    dynamic_shared: usize,
    registers: usize,
    local_bytes: usize,
    occupancy: u32,
}

fn resource_contract(spec: CandidateSpec) -> ResourceContract {
    ResourceContract {
        threads: spec.block.0,
        static_shared: 0,
        dynamic_shared: spec.dynamic_shared_bytes as usize,
        register_cap: spec.register_cap as usize,
        minimum_occupancy: spec.minimum_occupancy,
    }
}

fn validate_resources(
    actual: ResourceObservation,
    expected: ResourceContract,
) -> Result<(), String> {
    if actual.threads != expected.threads
        || actual.static_shared != expected.static_shared
        || actual.dynamic_shared != expected.dynamic_shared
        || actual.registers == 0
        || actual.registers > expected.register_cap
        || actual.local_bytes != 0
        || actual.occupancy < expected.minimum_occupancy
    {
        return Err(format!(
            "resource contract failed: actual={actual:?} expected={expected:?}"
        ));
    }
    Ok(())
}

fn splitm16_value(x: &[f32], dy: &[f32], seed: f32, row: usize, column: usize) -> u32 {
    let mut sum = 0.0_f64;
    for chunk in 0..16 {
        let mut partial = 0.0_f32;
        for reduction in chunk * 16..(chunk + 1) * 16 {
            partial = x[reduction * DIMS.1 + row].mul_add(dy[reduction * DIMS.2 + column], partial);
        }
        if chunk == 0 {
            sum = f64::from(partial);
        } else {
            sum += f64::from(partial);
        }
    }
    (seed + (1.0_f64 * sum) as f32).to_bits()
}

fn splitm16_oracle(x: &[f32], dy: &[f32], seed: &[f32]) -> Result<Vec<u32>, String> {
    if x.len() != DIMS.0 * DIMS.1 || dy.len() != DIMS.0 * DIMS.2 || seed.len() != DIMS.1 * DIMS.2 {
        return Err("old two-node oracle received the wrong tensor extents".into());
    }
    let mut bits = Vec::with_capacity(seed.len());
    for row in 0..DIMS.1 {
        for column in 0..DIMS.2 {
            bits.push(splitm16_value(
                x,
                dy,
                seed[row * DIMS.2 + column],
                row,
                column,
            ));
        }
    }
    Ok(bits)
}

#[test]
fn tn_underfill_cuda_source_exports_six_exact_candidates() {
    let source = read_candidate_source().expect("candidate CUDA source must exist");
    for spec in CANDIDATES {
        assert_eq!(
            spec.grid.0 * spec.grid.1 * spec.grid.2,
            (DIMS.1 as u32 / spec.tile.0) * (DIMS.2 as u32 / spec.tile.1),
            "{} grid does not cover its exact tile census",
            spec.symbol
        );
        let signature = entry_signature(&source, spec.symbol).unwrap();
        assert_eq!(
            signature.matches(',').count() + 1,
            7,
            "{} ABI drifted",
            spec.symbol
        );
    }
    for required in [
        "m != MRed",
        "k != KOut",
        "n != N",
        "Chunks == 16",
        "M16N32::SharedBytes == 6656",
        "M32N32::SharedBytes == 8192",
        "M16N32APad0Ca::SharedBytes == 6144",
        "M16N32APad0Cg::SharedBytes == 6144",
        "M16N16::SharedBytes == 4096",
        "M8N32::SharedBytes == 5120",
        "__fmaf_rn",
        "__dadd_rn",
        "__dmul_rn",
        "cp.async.wait_group 0",
    ] {
        assert!(
            source.contains(required),
            "candidate source omitted {required}"
        );
    }
    for forbidden in ["atomic", "mma.sync", "--use_fast_math", "__fmul_rz"] {
        assert!(
            !source.to_ascii_lowercase().contains(forbidden),
            "candidate source contains forbidden {forbidden}"
        );
    }
    let modules = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
    for spec in CANDIDATES {
        assert!(
            !modules.contains(spec.symbol),
            "test candidate {} escaped into production inventory",
            spec.symbol
        );
    }
}

#[test]
fn harness_freezes_public_one_node_production_and_no_rescreen_official_timing() {
    let source = include_str!("gemm_bi_scalar_tn_underfill_direct_experiment.rs");
    for required in [
        "qualify_physical_launch",
        "production TN underfill identity drifted",
        "gemm_bi_tn_m16n16_bk16_s2_splitm16_v1",
        "gpu_gemm_bi_backward_dw_grad",
        "production raw active pointers must remain at allocation offset zero",
        "candidate_graph_identity",
        "PathKind::Eager",
        "PathKind::Graph",
        "Order::Abba",
        "Order::Baab",
        "verify_post_cohort_even_on_error",
        "validate_ptxas_and_sass",
    ] {
        assert!(source.contains(required), "harness omitted {required}");
    }
    let official = source
        .rsplit_once("fn tn_underfill_direct_official_101_windows()")
        .map(|(_, body)| body)
        .expect("official-101 test body");
    assert!(!official.contains("for spec in CANDIDATES"));
    assert!(official.contains("OFFICIAL_WINNER"));
    assert!(official.contains("OFFICIAL_RUNNER"));
    assert!(official.contains("load_kernels(&runtime, &OFFICIAL_CANDIDATES)"));
    assert!(official.contains("require_promoted_production(&runtime)"));
    assert!(official.contains("(Arm::Candidate(OFFICIAL_RUNNER), Arm::Production)"));
    assert!(official.contains("(Arm::Candidate(OFFICIAL_WINNER), Arm::Production)"));
    let runtime = source
        .rsplit_once("mod cuda_experiment {")
        .map(|(_, runtime)| runtime)
        .expect("CUDA experiment module");
    assert!(!runtime.contains("cudarc::cublas"));
}

#[test]
fn next_screen_covers_four_frozen_challengers_without_changing_the_official_pair() {
    let expected = [
        (
            "gemm_bi_tn_underfill_m16n32_bk16_s2_splitm16_apad0_ca_exp_v1",
            (16, 32),
            (384, 1, 1),
            (128, 1, 1),
            6_144,
            112,
            4,
        ),
        (
            "gemm_bi_tn_underfill_m16n32_bk16_s2_splitm16_apad0_cg_exp_v1",
            (16, 32),
            (384, 1, 1),
            (128, 1, 1),
            6_144,
            112,
            4,
        ),
        (
            "gemm_bi_tn_underfill_m16n16_bk16_s2_splitm16_exp_v1",
            (16, 16),
            (768, 1, 1),
            (64, 1, 1),
            4_096,
            112,
            4,
        ),
        (
            "gemm_bi_tn_underfill_m8n32_bk16_s2_splitm16_exp_v1",
            (8, 32),
            (768, 1, 1),
            (64, 1, 1),
            5_120,
            112,
            4,
        ),
    ];
    assert_eq!(INITIAL_CANDIDATES, [INITIAL_WINNER, INITIAL_RUNNER]);
    assert_eq!(NEXT_CHALLENGERS.len(), expected.len());
    assert_eq!(CANDIDATES.len(), INITIAL_CANDIDATES.len() + expected.len());
    assert_eq!(&CANDIDATES[..2], &INITIAL_CANDIDATES);
    assert_eq!(&CANDIDATES[2..], &NEXT_CHALLENGERS);
    for (spec, expected) in NEXT_CHALLENGERS.into_iter().zip(expected) {
        assert_eq!(
            (
                spec.symbol,
                spec.tile,
                spec.grid,
                spec.block,
                spec.dynamic_shared_bytes,
                spec.register_cap,
                spec.minimum_occupancy,
            ),
            expected
        );
    }
    let source = include_str!("gemm_bi_scalar_tn_underfill_direct_experiment.rs");
    assert!(source.contains("const OFFICIAL_WINNER: CandidateSpec = M16N16;"));
    assert!(source.contains("const OFFICIAL_RUNNER: CandidateSpec = M8N32;"));
    let next = source
        .rsplit_once("fn tn_underfill_direct_next_screen_21_windows()")
        .map(|(_, body)| body)
        .expect("next-screen test body");
    assert!(next.contains("run_informational_pair_matrix"));
    assert!(next.contains("(Arm::Candidate(challenger), Arm::Production)"));
    assert!(next.contains("for challenger in NEXT_CHALLENGERS"));
}

#[test]
fn exact_shape_window_percentile_and_speedup_gates_are_fail_closed() {
    validate_exact_dims((256, 512, 384)).unwrap();
    for neighbor in [
        (255, 512, 384),
        (257, 512, 384),
        (256, 511, 384),
        (256, 513, 384),
        (256, 512, 383),
        (256, 512, 385),
    ] {
        assert!(
            validate_exact_dims(neighbor).is_err(),
            "accepted {neighbor:?}"
        );
    }
    assert_eq!(exact_windows(None, SCREEN_WINDOWS).unwrap(), SCREEN_WINDOWS);
    assert_eq!(
        exact_windows(Some(OsStr::new("101")), OFFICIAL_WINDOWS).unwrap(),
        OFFICIAL_WINDOWS
    );
    for invalid in ["0", "1", "20", "22", "100", "102", "bad"] {
        assert!(exact_windows(Some(OsStr::new(invalid)), SCREEN_WINDOWS).is_err());
    }
    let values = (1..=101).map(|value| value as f64).collect::<Vec<_>>();
    assert_eq!(percentile(&values, 0.05).unwrap(), 6.0);
    assert_eq!(percentile(&values, 0.50).unwrap(), 51.0);
    for values in [
        vec![],
        vec![0.0],
        vec![-1.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
    ] {
        assert!(percentile(&values, 0.5).is_err());
    }
    validate_speedup(21, (1.02, 1.05, 1.2), (1.02, 1.05)).unwrap();
    assert!(validate_speedup(20, (2.0, 2.0, 2.0), (1.02, 1.05)).is_err());
    assert!(validate_speedup(21, (1.019, 1.05, 1.2), (1.02, 1.05)).is_err());
    assert!(validate_speedup(21, (1.02, 1.049, 1.2), (1.02, 1.05)).is_err());
    assert!(validate_speedup(21, (f64::NAN, 2.0, 2.0), (1.02, 1.05)).is_err());
}

#[test]
fn splitm16_oracle_freezes_mchunk16_f64_reducer_and_exception_classes() {
    let mut x = vec![0.0; DIMS.0 * DIMS.1];
    let mut dy = vec![0.0; DIMS.0 * DIMS.2];
    let mut seed = vec![0.0; DIMS.1 * DIMS.2];
    let witnesses = [
        (0, 0x0000_0001, 1.0_f32.to_bits()),
        (15, 0x007f_ffff, 1.0_f32.to_bits()),
        (16, 1.0_f32.to_bits(), f32::INFINITY.to_bits()),
        (31, (-1.0_f32).to_bits(), f32::INFINITY.to_bits()),
        (32, 0x7fc2_3456, 1.0_f32.to_bits()),
        (63, 0x7fa1_2345, 1.0_f32.to_bits()),
        (128, 0.0_f32.to_bits(), (-1.0_f32).to_bits()),
        (255, (-0.0_f32).to_bits(), 1.0_f32.to_bits()),
    ];
    for (column, (reduction, x_bits, dy_bits)) in witnesses.into_iter().enumerate() {
        x[reduction * DIMS.1] = f32::from_bits(x_bits);
        dy[reduction * DIMS.2 + column] = f32::from_bits(dy_bits);
    }
    seed[8] = -0.0;
    let expected = splitm16_oracle(&x, &dy, &seed).unwrap();
    for column in 0..witnesses.len() {
        assert_eq!(
            expected[column],
            splitm16_value(&x, &dy, seed[column], 0, column),
            "exception class {column}"
        );
    }

    x.fill(0.0);
    dy.fill(0.0);
    x[15 * DIMS.1] = 2.0_f32.powi(25);
    dy[15 * DIMS.2] = 1.0;
    x[16 * DIMS.1] = -2.0_f32.powi(25);
    dy[16 * DIMS.2] = 1.0;
    x[31 * DIMS.1] = 1.0;
    dy[31 * DIMS.2] = 1.0;
    let split_tree = splitm16_value(&x, &dy, 0.0, 0, 0);
    let direct = (0..DIMS.0)
        .fold(0.0_f32, |sum, reduction| {
            x[reduction * DIMS.1].mul_add(dy[reduction * DIMS.2], sum)
        })
        .to_bits();
    assert_ne!(split_tree, direct, "tree killer stopped discriminating");
}

#[test]
fn resource_contract_rejects_every_single_field_mutation() {
    for spec in CANDIDATES {
        let contract = resource_contract(spec);
        let valid = ResourceObservation {
            threads: contract.threads,
            static_shared: contract.static_shared,
            dynamic_shared: contract.dynamic_shared,
            registers: contract.register_cap,
            local_bytes: 0,
            occupancy: contract.minimum_occupancy,
        };
        validate_resources(valid, contract).unwrap();
        for mutate in [
            |value: &mut ResourceObservation| value.threads += 1,
            |value: &mut ResourceObservation| value.static_shared += 4,
            |value: &mut ResourceObservation| value.dynamic_shared += 4,
            |value: &mut ResourceObservation| value.registers += 1,
            |value: &mut ResourceObservation| value.local_bytes = 4,
            |value: &mut ResourceObservation| value.occupancy -= 1,
        ] {
            let mut changed = valid;
            mutate(&mut changed);
            assert!(validate_resources(changed, contract).is_err());
        }
    }
}

#[test]
fn ptx_contract_rejects_abi_arithmetic_and_nondeterministic_mutations() {
    let entry = |symbol: &str| {
        format!(
            ".visible .entry {symbol}(\n.param .u64 c,\n.param .u64 a,\n.param .u64 b,\n.param .f32 alpha,\n.param .u32 m,\n.param .u32 k,\n.param .u32 n\n)\n{{\nfma.rn.f32;\nadd.rn.f64;\nmul.rn.f64;\nret;\n}}\n"
        )
    };
    let valid = CANDIDATES
        .iter()
        .map(|spec| entry(spec.symbol))
        .collect::<String>();
    validate_candidate_ptx(&valid).unwrap();
    validate_candidate_ptx(&valid.replace("ret;", "ld.shared.u32;\nret;"))
        .expect("shared address-space spelling must not look like a reduction opcode");
    for (from, to) in [
        (".param .f32 alpha", ".param .u32 alpha"),
        ("fma.rn.f32", "mad.rn.f32"),
        ("add.rn.f64", "sub.rn.f64"),
        ("mul.rn.f64", "div.rn.f64"),
        ("ret;", "atom.global.add.f32; ret;"),
        ("ret;", "red.global.add.f32; ret;"),
        ("ret;", "redux.sync.add.u32; ret;"),
        ("ret;", "mma.sync.aligned; ret;"),
        ("ret;", "add.rn.ftz.f32; ret;"),
        ("ret;", "target: @p0 add.rn.ftz.f32; ret;"),
        ("ret;", "call.uni helper; ret;"),
    ] {
        let mutated = valid.replacen(from, to, 1);
        assert!(validate_candidate_ptx(&mutated).is_err(), "accepted {to}");
    }
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

#[cfg(feature = "cuda")]
mod cuda_experiment {
    use std::ffi::CStr;
    use std::sync::Arc;

    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, CudaStream, LaunchConfig, PushKernelArg, sys,
    };
    use mamba_rs::mamba_ssm::gpu::blas::gpu_gemm_bi_backward_dw_grad;
    use mamba_rs::mamba_ssm::gpu::buffers::{GpuBuffer, GradSlice};
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
        CANDIDATES, CandidateSpec, DIMS, EXCEPTIONAL_OUTPUT_DIGESTS, MIN_WINNER_EAGER_P05,
        MIN_WINNER_EAGER_P50, MIN_WINNER_GRAPH_P05, MIN_WINNER_GRAPH_P50, NEXT_CHALLENGERS,
        OFFICIAL_CANDIDATES, OFFICIAL_RUNNER, OFFICIAL_WINDOWS, OFFICIAL_WINNER,
        ResourceObservation, SCREEN_WINDOWS, exact_windows, percentile, read_candidate_source,
        resource_contract, splitm16_oracle, validate_candidate_ptx, validate_exact_dims,
        validate_resources, validate_speedup,
    };

    const GUARD_ELEMENTS: usize = 64;
    const X_GUARD: u32 = 0x7fc0_a271;
    const DY_GUARD: u32 = 0x7fc0_b271;
    const C_GUARD: u32 = 0x7fc0_c271;
    const TARGET_WINDOW_US: f64 = 10_000.0;
    const CORRECTNESS_REPEATS: usize = 3;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        Production,
        Candidate(CandidateSpec),
    }

    impl Arm {
        fn label(self) -> &'static str {
            match self {
                Self::Production => "production-m16n16",
                Self::Candidate(spec) => spec.symbol,
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum PathKind {
        Eager,
        Graph,
    }

    #[derive(Clone, Copy, Debug)]
    enum Order {
        Abba,
        Baab,
    }

    struct Runtime {
        production: GpuCtx,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        expects_promoted_production: bool,
    }

    struct Kernel {
        spec: CandidateSpec,
        function: CudaFunction,
    }

    struct Kernels {
        values: Vec<Kernel>,
    }

    impl Kernels {
        fn get(&self, spec: CandidateSpec) -> Result<&Kernel, String> {
            self.values
                .iter()
                .find(|kernel| kernel.spec == spec)
                .ok_or_else(|| format!("candidate kernel {} is missing", spec.symbol))
        }
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
            prefix: usize,
            guard_bits: u32,
        ) -> Result<Self, String> {
            let active_len = active.len();
            let total = prefix
                .checked_add(active_len)
                .and_then(|value| value.checked_add(GUARD_ELEMENTS))
                .ok_or_else(|| "guarded allocation extent overflow".to_string())?;
            let mut expected = vec![f32::from_bits(guard_bits); total];
            expected[prefix..prefix + active_len].copy_from_slice(&active);
            Ok(Self {
                buffer: GpuBuffer::from_cpu(stream, &expected)?,
                expected,
                active_offset: prefix,
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
                        "{label} guard {index} changed to 0x{:08x}",
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
                return Err(format!("{label} read-only payload changed"));
            }
            Ok(())
        }
    }

    struct Fixture {
        x: GuardedBuffer,
        dy: GuardedBuffer,
        outputs: Vec<(CandidateSpec, GuardedBuffer)>,
        expected: Vec<u32>,
    }

    struct ProductionFixture {
        x: GuardedBuffer,
        dy: GuardedBuffer,
        output: GuardedBuffer,
        expected: Vec<u32>,
    }

    fn compose_source() -> Result<String, String> {
        let candidate = read_candidate_source()?;
        Ok([
            include_str!("../kernels/_typed_prelude.cuh"),
            candidate.as_str(),
        ]
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
            .map_err(|error| format!("compile scalar TN underfill experiment: {error:?}"))
    }

    fn new_runtime() -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        if device.compute_capability.0 < 8 {
            return Err(format!(
                "TN underfill experiment requires SM80+, found {:?}",
                device.compute_capability
            ));
        }
        let ptx = compile_ptx(device.nvrtc_target())?;
        validate_candidate_ptx(&ptx)?;
        let mut nvrtc_major = 0;
        let mut nvrtc_minor = 0;
        let nvrtc_status =
            unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut nvrtc_major, &mut nvrtc_minor) };
        if nvrtc_status != cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("query NVRTC version: {nvrtc_status:?}"));
        }
        let expects_promoted_production = device.compute_capability == (12, 0)
            && device.multiprocessor_count() == 170
            && (nvrtc_major, nvrtc_minor) == (13, 2);
        let production = GpuCtx::new(&device)?;
        production.set_batch_invariant(true);
        production.set_bi_gemm_family(BiGemmFamily::Triad);
        production.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        presize_physical_qualification_suite(&production, &[production_request()])?;
        let stream = device.fork_stream()?;
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|error| format!("load scalar TN underfill module: {error:?}"))?;
        Ok(Runtime {
            production,
            stream,
            module,
            expects_promoted_production,
        })
    }

    fn load_kernels(runtime: &Runtime, candidates: &[CandidateSpec]) -> Result<Kernels, String> {
        let mut values = Vec::with_capacity(candidates.len());
        for spec in candidates.iter().copied() {
            let function = runtime
                .module
                .load_function(spec.symbol)
                .map_err(|error| format!("load {}: {error:?}", spec.symbol))?;
            function
                .set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    spec.dynamic_shared_bytes as i32,
                )
                .map_err(|error| format!("set {} dynamic shared: {error:?}", spec.symbol))?;
            values.push(Kernel { spec, function });
        }
        Ok(Kernels { values })
    }

    fn production_request() -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Tn,
            DIMS,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
        )
    }

    fn production_holder(runtime: &Runtime) -> Result<QualifiedPhysicalLaunch<'_>, String> {
        let mut holder = qualify_physical_launch(&runtime.production, production_request())?;
        holder.seed_f32_operands(&runtime.production, 0x544e_5546)?;
        let evidence = holder.evidence();
        let nodes = evidence.nodes();
        let symbols = nodes.iter().map(|node| node.symbol).collect::<Vec<_>>();
        let promoted = evidence.launch_count() == 1
            && symbols == ["gemm_bi_tn_m16n16_bk16_s2_splitm16_v1"]
            && nodes[0].module_kind == ModuleKind::TriadScalar
            && nodes[0].launch.grid_dim == (768, 1, 1)
            && nodes[0].launch.block_dim == (64, 1, 1)
            && nodes[0].launch.shared_mem_bytes == 4_096;
        let fallback = evidence.launch_count() == 2
            && symbols == ["gemm_bi_tn_splitm_partial_aligned", "gemm_bi_splitm_reduce"]
            && nodes
                .iter()
                .all(|node| node.module_kind == ModuleKind::TriadScalar)
            && nodes[0].launch.grid_dim == (12, 1, 16)
            && nodes[0].launch.block_dim == (256, 1, 1)
            && nodes[0].launch.shared_mem_bytes == 0
            && nodes[1].launch.grid_dim == (768, 1, 1)
            && nodes[1].launch.block_dim == (256, 1, 1)
            && nodes[1].launch.shared_mem_bytes == 0;
        let expected_route = if runtime.expects_promoted_production {
            promoted
        } else {
            fallback
        };
        if !expected_route
            || nodes
                .iter()
                .any(|node| node.launch.arguments_digest == [0; 32])
            || (fallback && nodes[0].launch.arguments_digest == nodes[1].launch.arguments_digest)
            || evidence.launch_digest() == [0; 32]
            || !evidence.eager_graph_equal()
        {
            return Err(format!(
                "production TN underfill identity drifted: {nodes:?}"
            ));
        }
        Ok(holder)
    }

    fn require_promoted_production(runtime: &Runtime) -> Result<(), String> {
        let holder = production_holder(runtime)?;
        let nodes = holder.evidence().nodes();
        if nodes.len() == 1 && nodes[0].symbol == "gemm_bi_tn_m16n16_bk16_s2_splitm16_v1" {
            Ok(())
        } else {
            Err(format!(
                "post-promotion timing requires the qualified one-node route, found {nodes:?}"
            ))
        }
    }

    fn random_values(len: usize, mut state: u64) -> Vec<f32> {
        (0..len)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if index % 4093 == 0 {
                    if index.is_multiple_of(2) { 0.0 } else { -0.0 }
                } else {
                    let signed = ((state % 4093) as i32) - 2046;
                    signed as f32 / 1024.0
                }
            })
            .collect()
    }

    fn make_values(kind: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut x = random_values(DIMS.0 * DIMS.1, 0x1357_2468 ^ kind as u64);
        let mut dy = random_values(DIMS.0 * DIMS.2, 0x9753_8642 ^ kind as u64);
        let mut seed = random_values(DIMS.1 * DIMS.2, 0x2468_1357 ^ kind as u64);
        if kind == 1 {
            x.fill(0.0);
            dy.fill(0.0);
            seed.fill(0.0);
            x[15 * DIMS.1] = 2.0_f32.powi(25);
            dy[15 * DIMS.2] = 1.0;
            x[16 * DIMS.1] = -2.0_f32.powi(25);
            dy[16 * DIMS.2] = 1.0;
            x[31 * DIMS.1] = 1.0;
            dy[31 * DIMS.2] = 1.0;
        } else if (2..10).contains(&kind) {
            x.fill(0.0);
            dy.fill(0.0);
            seed.fill(0.0);
            let case = [
                (0, 0x0000_0001, 1.0_f32.to_bits()),
                (15, 0x007f_ffff, 1.0_f32.to_bits()),
                (16, 1.0_f32.to_bits(), f32::INFINITY.to_bits()),
                (31, (-1.0_f32).to_bits(), f32::INFINITY.to_bits()),
                (32, 0x7fc2_3456, 1.0_f32.to_bits()),
                (63, 0x7fa1_2345, 1.0_f32.to_bits()),
                (128, 0.0_f32.to_bits(), (-1.0_f32).to_bits()),
                (255, (-0.0_f32).to_bits(), 1.0_f32.to_bits()),
            ][kind - 2];
            let (reduction, x_bits, dy_bits) = case;
            x[reduction * DIMS.1] = f32::from_bits(x_bits);
            dy[reduction * DIMS.2] = f32::from_bits(dy_bits);
            seed[0] = if kind == 9 { -0.0 } else { 0.0 };
        }
        (x, dy, seed)
    }

    fn candidate_fixture(runtime: &Runtime, kind: usize) -> Result<Fixture, String> {
        let (x, dy, seed) = make_values(kind);
        let expected = splitm16_oracle(&x, &dy, &seed)?;
        let outputs = CANDIDATES
            .iter()
            .copied()
            .map(|spec| {
                GuardedBuffer::new(&runtime.stream, seed.clone(), GUARD_ELEMENTS, C_GUARD)
                    .map(|output| (spec, output))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Fixture {
            x: GuardedBuffer::new(&runtime.stream, x, GUARD_ELEMENTS, X_GUARD)?,
            dy: GuardedBuffer::new(&runtime.stream, dy, GUARD_ELEMENTS, DY_GUARD)?,
            outputs,
            expected,
        })
    }

    fn production_fixture(runtime: &Runtime, kind: usize) -> Result<ProductionFixture, String> {
        let (x, dy, seed) = make_values(kind);
        let expected = splitm16_oracle(&x, &dy, &seed)?;
        Ok(ProductionFixture {
            x: GuardedBuffer::new(&runtime.production.stream, x, 0, X_GUARD)?,
            dy: GuardedBuffer::new(&runtime.production.stream, dy, 0, DY_GUARD)?,
            output: GuardedBuffer::new(&runtime.production.stream, seed, 0, C_GUARD)?,
            expected,
        })
    }

    fn candidate_output(fixture: &Fixture, spec: CandidateSpec) -> Result<&GuardedBuffer, String> {
        fixture
            .outputs
            .iter()
            .find_map(|(actual, output)| (*actual == spec).then_some(output))
            .ok_or_else(|| format!("output for {} is missing", spec.symbol))
    }

    fn candidate_output_mut(
        fixture: &mut Fixture,
        spec: CandidateSpec,
    ) -> Result<&mut GuardedBuffer, String> {
        fixture
            .outputs
            .iter_mut()
            .find_map(|(actual, output)| (*actual == spec).then_some(output))
            .ok_or_else(|| format!("output for {} is missing", spec.symbol))
    }

    fn launch_candidate(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &Fixture,
        spec: CandidateSpec,
    ) -> Result<(), String> {
        validate_exact_dims((DIMS.0 as i32, DIMS.1 as i32, DIMS.2 as i32))?;
        let kernel = kernels.get(spec)?;
        let output = candidate_output(fixture, spec)?.ptr(&runtime.stream);
        let x = fixture.x.ptr(&runtime.stream);
        let dy = fixture.dy.ptr(&runtime.stream);
        let alpha = 1.0_f32;
        let m = DIMS.0 as i32;
        let k = DIMS.1 as i32;
        let n = DIMS.2 as i32;
        let mut builder = runtime.stream.launch_builder(&kernel.function);
        builder.arg(&output);
        builder.arg(&x);
        builder.arg(&dy);
        builder.arg(&alpha);
        builder.arg(&m);
        builder.arg(&k);
        builder.arg(&n);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: spec.grid,
                block_dim: spec.block,
                shared_mem_bytes: spec.dynamic_shared_bytes,
            })
        }
        .map(|_| ())
        .map_err(|error| format!("launch {}: {error:?}", spec.symbol))
    }

    fn launch_production(runtime: &Runtime, fixture: &ProductionFixture) -> Result<(), String> {
        let output = fixture.output.ptr(&runtime.production.stream);
        gpu_gemm_bi_backward_dw_grad(
            &runtime.production,
            &GradSlice::from_raw(output, DIMS.1 * DIMS.2),
            &fixture.dy.buffer,
            &fixture.x.buffer,
            DIMS.0,
            DIMS.1,
            DIMS.2,
        )
    }

    fn capture_candidate(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &Fixture,
        spec: CandidateSpec,
    ) -> Result<CudaGraph, String> {
        unsafe {
            capture_into_graph(&runtime.stream, || {
                launch_candidate(runtime, kernels, fixture, spec)
            })
        }
    }

    fn capture_production(
        runtime: &Runtime,
        fixture: &ProductionFixture,
    ) -> Result<CudaGraph, String> {
        unsafe {
            capture_into_graph(&runtime.production.stream, || {
                launch_production(runtime, fixture)
            })
        }
    }

    fn cuda_ok(result: sys::CUresult, operation: &str) -> Result<(), String> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{operation}: {result:?}"))
        }
    }

    fn candidate_graph_identity(
        graph: &CudaGraph,
        spec: CandidateSpec,
    ) -> Result<[u8; 32], String> {
        let raw = graph.cu_graph();
        let mut count = 0_usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, std::ptr::null_mut(), &mut count) },
            "query candidate graph nodes",
        )?;
        if count != 1 {
            return Err(format!("{} graph has {count} nodes", spec.symbol));
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
            "query candidate graph edges",
        )?;
        if edges != 0 {
            return Err(format!("{} graph has {edges} edges", spec.symbol));
        }
        let mut node = std::ptr::null_mut();
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, &mut node, &mut count) },
            "read candidate graph node",
        )?;
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        cuda_ok(
            unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
            "read candidate graph node type",
        )?;
        if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            return Err(format!("{} graph node is {kind:?}", spec.symbol));
        }
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            "read candidate graph params",
        )?;
        let mut name = std::ptr::null();
        cuda_ok(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "read candidate graph symbol",
        )?;
        if name.is_null() || params.kernelParams.is_null() {
            return Err(format!(
                "{} graph omitted function or packed arguments",
                spec.symbol
            ));
        }
        let symbol = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|error| format!("graph symbol is not UTF-8: {error}"))?;
        if symbol != spec.symbol
            || (params.gridDimX, params.gridDimY, params.gridDimZ) != spec.grid
            || (params.blockDimX, params.blockDimY, params.blockDimZ) != spec.block
            || params.sharedMemBytes != spec.dynamic_shared_bytes
        {
            return Err(format!("{} graph geometry drifted", spec.symbol));
        }
        let sizes = [8_usize, 8, 8, 4, 4, 4, 4];
        let mut digest = Sha256::new();
        digest.update(b"scalar-tn-underfill-direct-graph-args.v1");
        for (index, size) in sizes.into_iter().enumerate() {
            let pointer = unsafe { *params.kernelParams.add(index) };
            if pointer.is_null() {
                return Err(format!("{} graph argument {index} is null", spec.symbol));
            }
            digest.update((size as u64).to_le_bytes());
            digest.update(unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), size) });
        }
        let digest: [u8; 32] = digest.finalize().into();
        if digest == [0; 32] {
            return Err(format!("{} graph argument digest is zero", spec.symbol));
        }
        Ok(digest)
    }

    fn validate_candidate_resources(kernels: &Kernels) -> Result<(), String> {
        let mut census = Vec::new();
        for kernel in &kernels.values {
            let spec = kernel.spec;
            let observation = ResourceObservation {
                threads: spec.block.0,
                static_shared: kernel
                    .function
                    .shared_size_bytes()
                    .map_err(|error| format!("{} static shared: {error:?}", spec.symbol))?
                    as usize,
                dynamic_shared: spec.dynamic_shared_bytes as usize,
                registers: kernel
                    .function
                    .num_regs()
                    .map_err(|error| format!("{} registers: {error:?}", spec.symbol))?
                    as usize,
                local_bytes: kernel
                    .function
                    .local_size_bytes()
                    .map_err(|error| format!("{} local: {error:?}", spec.symbol))?
                    as usize,
                occupancy: kernel
                    .function
                    .occupancy_max_active_blocks_per_multiprocessor(
                        spec.block.0,
                        spec.dynamic_shared_bytes as usize,
                        None,
                    )
                    .map_err(|error| format!("{} occupancy: {error:?}", spec.symbol))?,
            };
            eprintln!(
                "scalar_tn_underfill resource symbol={} observation={observation:?}",
                spec.symbol
            );
            census.push((spec, observation));
        }
        let failures = census
            .into_iter()
            .filter_map(|(spec, observation)| {
                validate_resources(observation, resource_contract(spec))
                    .err()
                    .map(|error| format!("{}: {error}", spec.symbol))
            })
            .collect::<Vec<_>>();
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }

    fn candidate_correctness(
        runtime: &Runtime,
        kernels: &Kernels,
        fixture: &mut Fixture,
        spec: CandidateSpec,
    ) -> Result<(), String> {
        let graph = capture_candidate(runtime, kernels, fixture, spec)?;
        candidate_graph_identity(&graph, spec)?;
        for path in [PathKind::Eager, PathKind::Graph] {
            for repeat in 0..CORRECTNESS_REPEATS {
                candidate_output_mut(fixture, spec)?.reset(&runtime.stream)?;
                match path {
                    PathKind::Eager => launch_candidate(runtime, kernels, fixture, spec)?,
                    PathKind::Graph => graph
                        .launch()
                        .map_err(|error| format!("launch {} graph: {error:?}", spec.symbol))?,
                }
                runtime
                    .stream
                    .synchronize()
                    .map_err(|error| format!("synchronize {}: {error:?}", spec.symbol))?;
                let actual = candidate_output(fixture, spec)?.bits(&runtime.stream, spec.symbol)?;
                if actual != fixture.expected {
                    return Err(format!(
                        "{} {path:?} repeat {repeat} differs from the frozen SplitM16 result",
                        spec.symbol
                    ));
                }
                fixture.x.unchanged(&runtime.stream, "candidate X")?;
                fixture.dy.unchanged(&runtime.stream, "candidate dY")?;
            }
        }
        Ok(())
    }

    fn production_reference(
        runtime: &Runtime,
        fixture: &mut ProductionFixture,
    ) -> Result<Vec<u32>, String> {
        fixture.output.reset(&runtime.production.stream)?;
        launch_production(runtime, fixture)?;
        runtime
            .production
            .stream
            .synchronize()
            .map_err(|error| format!("synchronize production reference: {error:?}"))?;
        fixture
            .x
            .unchanged(&runtime.production.stream, "production reference X")?;
        fixture
            .dy
            .unchanged(&runtime.production.stream, "production reference dY")?;
        fixture
            .output
            .bits(&runtime.production.stream, "production reference output")
    }

    fn output_bits_digest(bits: &[u32]) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"scalar-tn-underfill-exceptional-output.v1");
        digest.update((bits.len() as u64).to_le_bytes());
        for value in bits {
            digest.update(value.to_le_bytes());
        }
        digest.finalize().into()
    }

    fn production_correctness(
        runtime: &Runtime,
        fixture: &mut ProductionFixture,
    ) -> Result<(), String> {
        if fixture.x.active_offset != 0
            || fixture.dy.active_offset != 0
            || fixture.output.active_offset != 0
        {
            return Err(
                "production raw active pointers must remain at allocation offset zero".into(),
            );
        }
        let graph = capture_production(runtime, fixture)?;
        for path in [PathKind::Eager, PathKind::Graph] {
            for repeat in 0..CORRECTNESS_REPEATS {
                fixture.output.reset(&runtime.production.stream)?;
                match path {
                    PathKind::Eager => launch_production(runtime, fixture)?,
                    PathKind::Graph => graph
                        .launch()
                        .map_err(|error| format!("launch production graph: {error:?}"))?,
                }
                runtime
                    .production
                    .stream
                    .synchronize()
                    .map_err(|error| format!("synchronize production: {error:?}"))?;
                let actual = fixture
                    .output
                    .bits(&runtime.production.stream, "production output")?;
                if actual != fixture.expected {
                    return Err(format!(
                        "production {path:?} repeat {repeat} differs from the frozen SplitM16 result"
                    ));
                }
                fixture
                    .x
                    .unchanged(&runtime.production.stream, "production X")?;
                fixture
                    .dy
                    .unchanged(&runtime.production.stream, "production dY")?;
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

    fn validate_ptxas_report(report: &str, spec: CandidateSpec) -> Result<(), String> {
        let marker = format!("Compiling entry function '{}'", spec.symbol);
        if report.matches(&marker).count() != 1 {
            return Err(format!("{} lost its unique ptxas record", spec.symbol));
        }
        let block = report
            .split_once(&marker)
            .expect("record count proves marker exists")
            .1
            .split("Compiling entry function '")
            .next()
            .expect("ptxas entry block");
        if !block.contains("0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads") {
            return Err(format!(
                "{} has stack or spill traffic:\n{block}",
                spec.symbol
            ));
        }
        let usage = block
            .lines()
            .find(|line| line.contains("Used ") && line.contains(" registers"))
            .ok_or_else(|| format!("{} lost ptxas resource usage", spec.symbol))?;
        let registers = metric_before(usage, " registers")
            .ok_or_else(|| format!("{} malformed register usage: {usage}", spec.symbol))?;
        let static_shared = metric_before(usage, " bytes smem").unwrap_or(0);
        if registers == 0 || registers > spec.register_cap as u64 || static_shared != 0 {
            return Err(format!(
                "{} ptxas resources failed: registers={registers}/{} static_shared={static_shared}/0",
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
        let line = line.trim();
        let line = line.strip_prefix("//").map(str::trim).unwrap_or(line);
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

    fn validate_sass_no_atomics(sass: &str, spec: CandidateSpec) -> Result<(), String> {
        for line in sass_entry(sass, spec.symbol)?.lines() {
            let instruction = line
                .split_once("*/")
                .map(|(_, tail)| tail.trim_start())
                .unwrap_or_else(|| line.trim_start());
            let opcode = instruction
                .split_whitespace()
                .find(|token| !token.starts_with('@'))
                .unwrap_or("");
            if opcode.starts_with("ATOM") || opcode.starts_with("RED.") {
                return Err(format!("{} SASS contains atomic: {line}", spec.symbol));
            }
        }
        Ok(())
    }

    fn validate_ptxas_and_sass(gpu: &str, ptx: String) -> Result<(), String> {
        let directory =
            tempfile::tempdir().map_err(|error| format!("resource tempdir: {error}"))?;
        let ptx_path = directory.path().join(format!("tn-underfill-{gpu}.ptx"));
        let cubin_path = directory.path().join(format!("tn-underfill-{gpu}.cubin"));
        std::fs::write(&ptx_path, ptx).map_err(|error| format!("write PTX: {error}"))?;
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
            .args(["--verbose", &format!("--gpu-name={gpu}")])
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
        for spec in CANDIDATES {
            validate_ptxas_report(&report, spec)?;
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
        for spec in CANDIDATES {
            validate_sass_no_atomics(&sass, spec)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA 13.2 NVRTC but launches no GPU work"]
    fn tn_underfill_candidates_compile_for_compute80_and_ptx_is_exact() -> Result<(), String> {
        validate_candidate_ptx(&compile_ptx("compute_80")?)
    }

    #[test]
    #[ignore = "requires CUDA 13.2 NVRTC, ptxas, and nvdisasm but launches no GPU work"]
    fn tn_underfill_candidates_meet_sm80_sm120_ptxas_and_sass_contracts() -> Result<(), String> {
        validate_ptxas_and_sass("sm_80", compile_ptx("compute_80")?)?;
        validate_ptxas_and_sass("sm_120", compile_ptx("compute_120")?)
    }

    #[test]
    #[ignore = "requires an exclusive SM80+ CUDA GPU"]
    fn tn_underfill_candidates_are_exact_resource_bounded_and_graph_stable() -> Result<(), String> {
        let runtime = new_runtime()?;
        if runtime.expects_promoted_production {
            require_promoted_production(&runtime)?;
        }
        let kernels = load_kernels(&runtime, &CANDIDATES)?;
        validate_candidate_resources(&kernels)?;
        let mut holder = production_holder(&runtime)?;
        holder.measure_graph_window_ms(&runtime.production, 1)?;
        holder.measure_eager_window_ms(&runtime.production, 1)?;
        drop(holder);
        for kind in 0..10 {
            let mut production = production_fixture(&runtime, kind)?;
            let reference = production_reference(&runtime, &mut production)?;
            if kind >= 2 {
                let digest = output_bits_digest(&reference);
                if digest != EXCEPTIONAL_OUTPUT_DIGESTS[kind - 2] {
                    return Err(format!(
                        "exceptional production fixture {kind} differs from the independent frozen witness: {digest:?}"
                    ));
                }
            }
            if kind < 2 && reference != production.expected {
                return Err(format!(
                    "finite production fixture {kind} differs from the independent SplitM16 oracle"
                ));
            }
            production.expected.clone_from(&reference);
            production_correctness(&runtime, &mut production)?;
            let mut candidate = candidate_fixture(&runtime, kind)?;
            candidate.expected.clone_from(&reference);
            for spec in CANDIDATES {
                candidate_correctness(&runtime, &kernels, &mut candidate, spec)?;
            }
        }
        Ok(())
    }

    struct TimingFixture {
        candidate: Fixture,
        graphs: Vec<(CandidateSpec, CudaGraph)>,
    }

    impl TimingFixture {
        fn graph(&self, spec: CandidateSpec) -> Result<&CudaGraph, String> {
            self.graphs
                .iter()
                .find_map(|(actual, graph)| (*actual == spec).then_some(graph))
                .ok_or_else(|| format!("timing graph for {} is missing", spec.symbol))
        }
    }

    fn new_timing_fixture(runtime: &Runtime, kernels: &Kernels) -> Result<TimingFixture, String> {
        let candidate = candidate_fixture(runtime, 0)?;
        let graphs = kernels
            .values
            .iter()
            .map(|kernel| {
                let spec = kernel.spec;
                capture_candidate(runtime, kernels, &candidate, spec).map(|graph| (spec, graph))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (spec, graph) in &graphs {
            candidate_graph_identity(graph, *spec)?;
        }
        Ok(TimingFixture { candidate, graphs })
    }

    fn measure_candidate(
        runtime: &Runtime,
        kernels: &Kernels,
        timing: &TimingFixture,
        request: (Arm, PathKind, usize),
    ) -> Result<f64, String> {
        let (arm, path, iterations) = request;
        let Arm::Candidate(spec) = arm else {
            return Err("custom candidate timer received production".into());
        };
        let start = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("candidate start event: {error:?}"))?;
        for _ in 0..iterations {
            match path {
                PathKind::Eager => launch_candidate(runtime, kernels, &timing.candidate, spec)?,
                PathKind::Graph => timing
                    .graph(spec)?
                    .launch()
                    .map_err(|error| format!("launch {} graph: {error:?}", spec.symbol))?,
            }
        }
        let end = runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("candidate end event: {error:?}"))?;
        let us = start
            .elapsed_ms(&end)
            .map(|ms| f64::from(ms) * 1_000.0 / iterations as f64)
            .map_err(|error| format!("candidate elapsed: {error:?}"))?;
        if us.is_finite() && us > 0.0 {
            Ok(us)
        } else {
            Err(format!("{} returned invalid {us}", arm.label()))
        }
    }

    fn measure_arm(
        context: (&Runtime, &Kernels, &TimingFixture),
        production: &mut QualifiedPhysicalLaunch<'_>,
        request: (Arm, PathKind, usize),
    ) -> Result<f64, String> {
        let (runtime, kernels, timing) = context;
        let (arm, path, iterations) = request;
        if arm == Arm::Production {
            let ms = match path {
                PathKind::Eager => {
                    production.measure_eager_window_ms(&runtime.production, iterations)?
                }
                PathKind::Graph => {
                    production.measure_graph_window_ms(&runtime.production, iterations)?
                }
            };
            let us = ms * 1_000.0 / iterations as f64;
            return if us.is_finite() && us > 0.0 {
                Ok(us)
            } else {
                Err(format!("production returned invalid {us}"))
            };
        }
        measure_candidate(runtime, kernels, timing, request)
    }

    fn measure_adjacent(
        context: (&Runtime, &Kernels, &TimingFixture),
        holder: &mut QualifiedPhysicalLaunch<'_>,
        pair: (Arm, Arm),
        path_order: (PathKind, Order),
        iterations: usize,
    ) -> Result<(f64, f64), String> {
        let (baseline, candidate) = pair;
        let (path, order) = path_order;
        let sequence = match order {
            Order::Abba => [baseline, candidate, candidate, baseline],
            Order::Baab => [candidate, baseline, baseline, candidate],
        };
        let mut baseline_values = Vec::with_capacity(2);
        let mut candidate_values = Vec::with_capacity(2);
        for arm in sequence {
            let value = measure_arm(context, holder, (arm, path, iterations))?;
            if arm == baseline {
                baseline_values.push(value);
            } else {
                candidate_values.push(value);
            }
        }
        Ok((
            baseline_values.iter().sum::<f64>() / 2.0,
            candidate_values.iter().sum::<f64>() / 2.0,
        ))
    }

    fn paired(
        context: (&Runtime, &Kernels, &TimingFixture),
        pair: (Arm, Arm),
        path_order: (PathKind, Order),
        windows: usize,
        quiet: &QuietGpu,
        enforce_speedup: bool,
    ) -> Result<(), String> {
        let (runtime, _, _) = context;
        let (path, order) = path_order;
        let mut holder = production_holder(runtime)?;
        let baseline_probe = measure_arm(context, &mut holder, (pair.0, path, 4))?;
        let candidate_probe = measure_arm(context, &mut holder, (pair.1, path, 4))?;
        let iterations = (TARGET_WINDOW_US / baseline_probe.min(candidate_probe)).ceil() as usize;
        if iterations == 0 {
            return Err("timing calibration returned zero iterations".into());
        }
        let label = format!(
            "scalar-tn-underfill/{}-to-{}/{path:?}/{order:?}/{windows}",
            pair.0.label(),
            pair.1.label()
        );
        quiet.require_cohort(&label)?;
        let body = (|| {
            let mut baseline_us = Vec::with_capacity(windows);
            let mut candidate_us = Vec::with_capacity(windows);
            let mut ratios = Vec::with_capacity(windows);
            for _ in 0..windows {
                let (baseline, candidate) =
                    measure_adjacent(context, &mut holder, pair, path_order, iterations)?;
                if !baseline.is_finite()
                    || baseline <= 0.0
                    || !candidate.is_finite()
                    || candidate <= 0.0
                {
                    return Err(format!(
                        "invalid paired samples baseline={baseline} candidate={candidate}"
                    ));
                }
                baseline_us.push(baseline);
                candidate_us.push(candidate);
                ratios.push(baseline / candidate);
            }
            let stats = (
                percentile(&ratios, 0.05)?,
                percentile(&ratios, 0.50)?,
                percentile(&ratios, 0.95)?,
            );
            eprintln!(
                "scalar_tn_underfill pair={} path={path:?} order={order:?} windows={windows} baseline_us_p50={:.6} candidate_us_p50={:.6} speedup_p05={:.9} speedup_p50={:.9} speedup_p95={:.9}",
                label,
                percentile(&baseline_us, 0.50)?,
                percentile(&candidate_us, 0.50)?,
                stats.0,
                stats.1,
                stats.2
            );
            if !enforce_speedup {
                Ok(())
            } else {
                let minimums = if path == PathKind::Eager {
                    (MIN_WINNER_EAGER_P05, MIN_WINNER_EAGER_P50)
                } else {
                    (MIN_WINNER_GRAPH_P05, MIN_WINNER_GRAPH_P50)
                };
                validate_speedup(windows, stats, minimums)
            }
        })();
        super::verify_post_cohort_even_on_error(body, quiet.verify_post_cohort(&label).map(drop))
    }

    fn run_pair_matrix(
        context: (&Runtime, &Kernels, &TimingFixture),
        pair: (Arm, Arm),
        windows: usize,
        quiet: &QuietGpu,
    ) -> Result<(), String> {
        for path in [PathKind::Eager, PathKind::Graph] {
            for order in [Order::Abba, Order::Baab] {
                paired(context, pair, (path, order), windows, quiet, true)?;
            }
        }
        Ok(())
    }

    fn run_informational_pair_matrix(
        context: (&Runtime, &Kernels, &TimingFixture),
        pair: (Arm, Arm),
        windows: usize,
        quiet: &QuietGpu,
    ) -> Result<(), String> {
        for path in [PathKind::Eager, PathKind::Graph] {
            for order in [Order::Abba, Order::Baab] {
                paired(context, pair, (path, order), windows, quiet, false)?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive quiet SM80+ CUDA GPU"]
    fn tn_underfill_direct_post_promotion_screen_21_windows() -> Result<(), String> {
        let windows = exact_windows(
            std::env::var_os("MAMBA_RS_TN_UNDERFILL_DIRECT_WINDOWS").as_deref(),
            SCREEN_WINDOWS,
        )?;
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        let label = "scalar-tn-underfill-post-promotion-screen";
        quiet.require_pre_context(label)?;
        let body = (|| {
            let runtime = new_runtime()?;
            require_promoted_production(&runtime)?;
            let kernels = load_kernels(&runtime, &OFFICIAL_CANDIDATES)?;
            validate_candidate_resources(&kernels)?;
            let timing = new_timing_fixture(&runtime, &kernels)?;
            let context = (&runtime, &kernels, &timing);
            run_pair_matrix(
                context,
                (Arm::Candidate(OFFICIAL_RUNNER), Arm::Production),
                windows,
                &quiet,
            )?;
            run_informational_pair_matrix(
                context,
                (Arm::Candidate(OFFICIAL_WINNER), Arm::Production),
                windows,
                &quiet,
            )
        })();
        super::verify_post_cohort_even_on_error(body, quiet.verify_post_cohort(label).map(drop))
    }

    #[test]
    #[ignore = "requires an exclusive quiet SM80+ CUDA GPU"]
    fn tn_underfill_direct_next_screen_21_windows() -> Result<(), String> {
        let windows = exact_windows(
            std::env::var_os("MAMBA_RS_TN_UNDERFILL_DIRECT_WINDOWS").as_deref(),
            SCREEN_WINDOWS,
        )?;
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        let label = "scalar-tn-underfill-next-screen";
        quiet.require_pre_context(label)?;
        let body = (|| {
            let runtime = new_runtime()?;
            require_promoted_production(&runtime)?;
            let kernels = load_kernels(&runtime, &CANDIDATES)?;
            validate_candidate_resources(&kernels)?;
            let timing = new_timing_fixture(&runtime, &kernels)?;
            let context = (&runtime, &kernels, &timing);
            for challenger in NEXT_CHALLENGERS {
                run_informational_pair_matrix(
                    context,
                    (Arm::Candidate(challenger), Arm::Production),
                    windows,
                    &quiet,
                )?;
            }
            Ok(())
        })();
        super::verify_post_cohort_even_on_error(body, quiet.verify_post_cohort(label).map(drop))
    }

    #[test]
    #[ignore = "requires an exclusive quiet SM80+ CUDA GPU and a release build"]
    fn tn_underfill_direct_official_101_windows() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("official TN underfill timing requires --release".into());
        }
        let windows = exact_windows(
            std::env::var_os("MAMBA_RS_TN_UNDERFILL_DIRECT_WINDOWS").as_deref(),
            OFFICIAL_WINDOWS,
        )?;
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        let label = "scalar-tn-underfill-official";
        quiet.require_pre_context(label)?;
        let body = (|| {
            let runtime = new_runtime()?;
            require_promoted_production(&runtime)?;
            let kernels = load_kernels(&runtime, &OFFICIAL_CANDIDATES)?;
            validate_candidate_resources(&kernels)?;
            let timing = new_timing_fixture(&runtime, &kernels)?;
            let context = (&runtime, &kernels, &timing);
            run_pair_matrix(
                context,
                (Arm::Candidate(OFFICIAL_RUNNER), Arm::Production),
                windows,
                &quiet,
            )?;
            run_informational_pair_matrix(
                context,
                (Arm::Candidate(OFFICIAL_WINNER), Arm::Production),
                windows,
                &quiet,
            )
        })();
        super::verify_post_cohort_even_on_error(body, quiet.verify_post_cohort(label).map(drop))
    }
}
