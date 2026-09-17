//! Independent SM120 exact-N64 production qualification with current incumbent AUTO.
//! Observed/unlocked telemetry; no Ada qualification source or semantics are changed.

use super::{
    CStr, CudaGraph, DtypedBuf, F32TriadPolicy, GpuCtx, GpuDevice, InferenceFwdOperands,
    InferenceShape, InferenceTile, TypedPtr, WeightDtype, capture_into_graph, digest_hex,
    inference_forward_f32_legacy_baseline, inference_forward_with_tile, launch_sm120_exact_auto,
    percentile, sm120_exact_event_window_us, sm120_exact_vendor_launch,
};
use cudarc::driver::{CudaFunction, sys};
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::io::{BufWriter, Write};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Corpus {
    Representable,
    Nonrepresentable,
}

#[derive(Clone, Copy)]
struct ProofInput<'a> {
    a: &'a [f32],
    b: &'a [f32],
    bias: Option<f32>,
    k: usize,
    a_start: usize,
    b_start: usize,
    b_stride: usize,
}

#[derive(Debug)]
struct Proof {
    ordered: u32,
    exact: f64,
}

fn selection(value: Option<&str>, inventory: &[&str]) -> Result<Vec<usize>, String> {
    let Some(value) = value else {
        return Ok((0..inventory.len()).collect());
    };
    let mut selected = Vec::new();
    for name in value.split(',') {
        let index = inventory
            .iter()
            .position(|item| *item == name)
            .ok_or_else(|| format!("unknown/empty filter {name:?}; expected {inventory:?}"))?;
        if selected.contains(&index) {
            return Err(format!("duplicate filter {name:?}"));
        }
        selected.push(index);
    }
    Ok(selected)
}

fn window_count(value: Option<&str>) -> Result<usize, String> {
    match value.unwrap_or("101") {
        "21" => Ok(21),
        "101" => Ok(101),
        other => Err(format!(
            "windows must be 21 (screen) or 101 (admission), got {other:?}"
        )),
    }
}
fn slots(order: &str) -> Result<[usize; 4], String> {
    match order {
        "ABBA" => Ok([0, 1, 1, 0]),
        "BAAB" => Ok([1, 0, 0, 1]),
        _ => Err(format!("unknown paired order {order}")),
    }
}
fn paired_ratio(order: &str, raw: [f64; 4]) -> Result<f64, String> {
    if raw.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err("nonpositive/nonfinite event sample".into());
    }
    let mut sums = [0.0; 2];
    for (arm, value) in slots(order)?.into_iter().zip(raw) {
        sums[arm] += value * 0.5;
    }
    let ratio = sums[0] / sums[1];
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err("invalid paired ratio".into());
    }
    Ok(ratio)
}
fn seed(index: usize, salt: u64, corpus: Corpus) -> f32 {
    let mut x = (index as u64)
        .wrapping_add(1)
        .wrapping_mul(0x9e3779b97f4a7c15)
        ^ salt;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    match corpus {
        Corpus::Representable => ((x % 4093) as i32 - 2046) as f32 / 1024.0,
        Corpus::Nonrepresentable => f32::from_bits(0x3f000000 | (x as u32 & 0x007fffff) | 1),
    }
}
fn numeric_close(actual: f64, reference: f64) -> bool {
    actual.is_finite()
        && reference.is_finite()
        && (actual - reference).abs() <= 0.0002 * (1.0 + reference.abs())
}
fn dyadic_integer(value: f32) -> Result<i64, String> {
    let scaled = f64::from(value) * 1024.0;
    if !scaled.is_finite() || scaled.abs() > 2046.0 || scaled != scaled.trunc() {
        return Err("proof requires finite [-2046,2046]/1024 inputs".into());
    }
    Ok(scaled as i64)
}
fn order_proof(input: ProofInput<'_>) -> Result<Proof, String> {
    if input.k > (i32::MAX - 31) as usize || input.b_stride == 0 {
        return Err("invalid proof K/stride".into());
    }
    let mut numerator = 0i64;
    let mut ordered = 0.0f32;
    for k in 0..input.k {
        let ai = input
            .a_start
            .checked_add(k)
            .ok_or("proof A index overflow")?;
        let bi = k
            .checked_mul(input.b_stride)
            .and_then(|i| input.b_start.checked_add(i))
            .ok_or("proof B index overflow")?;
        let a = *input.a.get(ai).ok_or("proof A view out of bounds")?;
        let b = *input.b.get(bi).ok_or("proof B view out of bounds")?;
        let product = dyadic_integer(a)?
            .checked_mul(dyadic_integer(b)?)
            .ok_or("exact product overflow")?;
        numerator = numerator.checked_add(product).ok_or("exact sum overflow")?;
        ordered = a.mul_add(b, ordered);
    }
    for _ in input.k..input.k.div_ceil(32) * 32 {
        ordered = 0.0f32.mul_add(0.0, ordered);
    }
    ordered *= 1.0;
    if let Some(bias) = input.bias {
        numerator = numerator
            .checked_add(
                dyadic_integer(bias)?
                    .checked_mul(1024)
                    .ok_or("bias numerator overflow")?,
            )
            .ok_or("biased exact sum overflow")?;
        ordered += bias;
    }
    if numerator.unsigned_abs() > (1u64 << 53) {
        return Err("exact numerator exceeds lossless f64 integer range".into());
    }
    Ok(Proof {
        ordered: ordered.to_bits(),
        exact: numerator as f64 / 1048576.0,
    })
}
fn certificate(corpus: Corpus, words: [u32; 4], input: ProofInput<'_>) -> Result<Proof, String> {
    if corpus != Corpus::Representable {
        return Err("no exceptional numeric proof for this corpus".into());
    }
    let [candidate, old, reference, vendor] = words;
    let proof = order_proof(input)?;
    if candidate != old
        || candidate != proof.ordered
        || !f32::from_bits(candidate).is_finite()
        || !numeric_close(f64::from(f32::from_bits(reference)), proof.exact)
        || !numeric_close(f64::from(f32::from_bits(vendor)), proof.exact)
    {
        return Err(format!(
            "independent order proof failed: words={words:x?}, proof={proof:?}"
        ));
    }
    Ok(proof)
}

const CANDIDATE: &str = "nn_sm120_f32_n64_copyplan";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Candidate {
    CopyPlan,
    Sliced,
}

impl Candidate {
    fn parse(value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("copyplan") => Ok(Self::CopyPlan),
            Some("sliced") => Ok(Self::Sliced),
            Some(other) => Err(format!(
                "MAMBA_FIXED_SM120_EXACT_N64_CANDIDATE only accepts copyplan or sliced, got {other:?}"
            )),
        }
    }

    fn tile(self) -> InferenceTile {
        match self {
            Self::CopyPlan => InferenceTile::F32Sm120N64CopyPlan,
            Self::Sliced => InferenceTile::F32Sm120N64Sliced,
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Self::CopyPlan => CANDIDATE,
            Self::Sliced => "nn_sm120_f32_n64_sliced",
        }
    }
}

#[test]
fn exact_n64_candidate_cpu_selection_defaults_and_rejects_unknown_values() {
    assert_eq!(Candidate::parse(None).unwrap(), Candidate::CopyPlan);
    assert_eq!(
        Candidate::parse(Some("copyplan")).unwrap(),
        Candidate::CopyPlan
    );
    assert_eq!(Candidate::parse(Some("sliced")).unwrap(), Candidate::Sliced);
    for bad in [
        "",
        "auto",
        "CopyPlan",
        "SLICED",
        "sliced ",
        "copyplan,sliced",
    ] {
        assert!(
            Candidate::parse(Some(bad)).is_err(),
            "invalid candidate {bad:?}"
        );
    }
    assert_eq!(
        Candidate::CopyPlan.tile(),
        InferenceTile::F32Sm120N64CopyPlan
    );
    assert_eq!(Candidate::CopyPlan.symbol(), "nn_sm120_f32_n64_copyplan");
    assert_eq!(Candidate::Sliced.tile(), InferenceTile::F32Sm120N64Sliced);
    assert_eq!(Candidate::Sliced.symbol(), "nn_sm120_f32_n64_sliced");
}

fn candidate_own_admission(
    candidate: Candidate,
    actual_auto: InferenceTile,
    cell: &str,
    has_bias: bool,
    protocol: &Protocol,
    legacy_p95: &[f64],
    auto_p95: &[f64],
) -> bool {
    if protocol.diagnostic_fast {
        return false;
    }
    if candidate == Candidate::Sliced {
        // Sliced was withdrawn from AUTO after its production vendor gate.
        // A fresh force qualification must beat the retained actual copyplan
        // as well as Legacy; it may never excuse a sliced self-comparison.
        return cell == "hot_b"
            && !has_bias
            && matches!(
                actual_auto,
                InferenceTile::F32Sm120N64CopyPlan | InferenceTile::F32Sm120TmaFmaM128N64
            )
            && own_admission(
                cell,
                protocol.windows,
                &protocol.paths,
                legacy_p95,
                auto_p95,
            );
    }
    if protocol.auto_phase.as_deref() == Some("promoted") && actual_auto == candidate.tile() {
        admission(cell, protocol.windows, &protocol.paths, legacy_p95)
    } else {
        own_admission(
            cell,
            protocol.windows,
            &protocol.paths,
            legacy_p95,
            auto_p95,
        )
    }
}

#[test]
fn exact_n64_candidate_cpu_sliced_b0_requires_independent_auto_comparison() {
    let mut protocol = Protocol {
        diagnostic_fast: false,
        cells: vec![1],
        biases: vec![0],
        paths: vec![0, 1],
        windows: 101,
        auto_phase: Some("promoted".into()),
    };
    let win = [0.9; 4];
    let self_noise = [1.02; 4];
    assert!(!candidate_own_admission(
        Candidate::Sliced,
        InferenceTile::F32Sm120N64Sliced,
        "hot_b",
        false,
        &protocol,
        &win,
        &self_noise
    ));
    assert!(
        !candidate_own_admission(
            Candidate::Sliced,
            InferenceTile::F32Sm120N64Sliced,
            "hot_b",
            false,
            &protocol,
            &win,
            &win
        ),
        "unqualified sliced AUTO cannot certify itself"
    );
    assert!(candidate_own_admission(
        Candidate::Sliced,
        InferenceTile::F32Sm120N64CopyPlan,
        "hot_b",
        false,
        &protocol,
        &win,
        &win
    ));
    assert!(candidate_own_admission(
        Candidate::Sliced,
        InferenceTile::F32Sm120TmaFmaM128N64,
        "hot_b",
        false,
        &protocol,
        &win,
        &win
    ));
    assert!(!candidate_own_admission(
        Candidate::CopyPlan,
        InferenceTile::F32Sm120N64Sliced,
        "hot_b",
        false,
        &protocol,
        &win,
        &self_noise
    ));
    assert!(!candidate_own_admission(
        Candidate::Sliced,
        InferenceTile::F32Sm120N64CopyPlan,
        "hot_b",
        false,
        &protocol,
        &win,
        &self_noise
    ));
    assert!(candidate_own_admission(
        Candidate::CopyPlan,
        InferenceTile::F32Sm120N64CopyPlan,
        "hot_e",
        true,
        &protocol,
        &win,
        &self_noise
    ));
    for (cell, bias) in [("hot_b", true), ("hot_e", false), ("hot_a", false)] {
        assert!(!candidate_own_admission(
            Candidate::Sliced,
            InferenceTile::F32Sm120N64Sliced,
            cell,
            bias,
            &protocol,
            &win,
            &win
        ));
    }
    for bad in [vec![0.9; 3], vec![1.0; 4], vec![f64::NAN; 4], vec![0.0; 4]] {
        assert!(!candidate_own_admission(
            Candidate::Sliced,
            InferenceTile::F32Sm120N64Sliced,
            "hot_b",
            false,
            &protocol,
            &bad,
            &win
        ));
    }
    protocol.windows = 21;
    assert!(!candidate_own_admission(
        Candidate::Sliced,
        InferenceTile::F32Sm120N64Sliced,
        "hot_b",
        false,
        &protocol,
        &win,
        &win
    ));
    protocol.windows = 101;
    protocol.paths = vec![0];
    assert!(!candidate_own_admission(
        Candidate::Sliced,
        InferenceTile::F32Sm120N64Sliced,
        "hot_b",
        false,
        &protocol,
        &win,
        &win
    ));
    protocol.paths = vec![0, 1];
    protocol.auto_phase = None;
    assert!(!candidate_own_admission(
        Candidate::Sliced,
        InferenceTile::F32Sm120N64Sliced,
        "hot_b",
        false,
        &protocol,
        &win,
        &self_noise
    ));
}
const LEGACY: &str = "f32_f32_s2";
const N128: &str = "f32_f32_n128_s2";
const ORACLE: &str = "f32_f32";
const GUARD: usize = 32;
const CANARY: u32 = 0x419a0000; // 19.25, checked as raw words.

fn resource_contract(observed: [i32; 6]) -> Result<(), String> {
    let [local, regs, shared, threads, blocks, carveout] = observed;
    if local != 0
        || !(1..=160).contains(&regs)
        || shared != 32768
        || threads < 128
        || blocks < 3
        || carveout != 100
    {
        return Err(format!("candidate live resources declined: {observed:?}"));
    }
    Ok(())
}
fn admission(cell: &str, windows: usize, paths: &[usize], own_ratios: &[f64]) -> bool {
    matches!(cell, "hot_e" | "hot_b")
        && windows == 101
        && paths.len() == 2
        && paths.contains(&0)
        && paths.contains(&1)
        && own_ratios.len() == 4
        && own_ratios
            .iter()
            .all(|r| r.is_finite() && *r > 0.0 && *r < 1.0)
}

// Independent literals: neither phase may be inferred from the route returned
// by production or from the selected timing subset.
fn expected_auto(
    phase: &str,
    shape: InferenceShape,
    has_bias: bool,
    t256_loaded: bool,
) -> Result<InferenceTile, String> {
    let dims = (shape.m, shape.k, shape.n);
    match phase {
        "incumbent" => Ok(if matches!(dims, (4621, 384, 1928) | (4621, 768, 2304)) {
            InferenceTile::F32N128S2
        } else {
            InferenceTile::Legacy
        }),
        "promoted" => Ok(if dims == (4621, 384, 1928) && !has_bias {
            InferenceTile::F32Sm120TmaFmaM64N128
        } else if dims == (4621, 384, 1928) && t256_loaded {
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
        } else if dims == (4621, 384, 1928) {
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64
        } else if dims == (4621, 768, 2304) && !has_bias {
            InferenceTile::F32Sm120TmaFmaM128N64
        } else if dims == (4621, 768, 2304) && has_bias {
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64
        } else if dims == (4621, 1928, 384) && has_bias {
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96
        } else if matches!(dims, (2048, 2304, 768) | (2048, 768, 2304)) {
            InferenceTile::F32Sm120N64CopyPlan
        } else {
            InferenceTile::Legacy
        }),
        other => Err(format!(
            "MAMBA_FIXED_SM120_EXACT_N64_AUTO only accepts incumbent or promoted, got {other:?}"
        )),
    }
}
fn auto_phase(
    requested: Option<&str>,
    shape: InferenceShape,
    has_bias: bool,
    actual: InferenceTile,
    t256_loaded: bool,
) -> Result<(), String> {
    let phase = requested.unwrap_or("incumbent");
    let expected = expected_auto(phase, shape, has_bias, t256_loaded)?;
    if actual != expected {
        return Err(format!(
            "SM120 AUTO mismatch: shape={shape:?}, expected={expected:?}, actual={actual:?}"
        ));
    }
    Ok(())
}
fn own_admission(
    cell: &str,
    windows: usize,
    paths: &[usize],
    legacy_p95: &[f64],
    auto_p95: &[f64],
) -> bool {
    admission(cell, windows, paths, legacy_p95) && admission(cell, windows, paths, auto_p95)
}

fn complete_inventory(actual: &[String], expected: &[String]) -> Result<(), String> {
    let actual_set: std::collections::BTreeSet<_> = actual.iter().collect();
    let expected_set: std::collections::BTreeSet<_> = expected.iter().collect();
    if expected.is_empty()
        || actual_set.len() != actual.len()
        || expected_set.len() != expected.len()
        || actual_set != expected_set
    {
        return Err(format!(
            "incomplete/duplicate/foreign paired inventory: actual={actual:?}, expected={expected:?}"
        ));
    }
    Ok(())
}

#[test]
fn exact_n64_admission_cpu_completion_rejects_duplicate_missing_or_foreign_pair() {
    let expected = vec![
        "hot_e/0/eager/ABBA/explicit_legacy".into(),
        "hot_e/0/graph/BAAB/pedantic".into(),
    ];
    complete_inventory(&expected, &expected).unwrap();
    for actual in [
        vec![],
        vec![expected[0].clone()],
        vec![expected[0].clone(); 2],
        vec![expected[0].clone(), "wrong".into()],
    ] {
        assert!(complete_inventory(&actual, &expected).is_err());
    }
    assert!(complete_inventory(&[], &[]).is_err());
}

#[test]
fn exact_n64_admission_cpu_auto_phase_is_explicit_and_fail_closed() {
    for (shape, expected) in [
        (
            InferenceShape {
                m: 4621,
                k: 384,
                n: 1928,
            },
            InferenceTile::F32N128S2,
        ),
        (
            InferenceShape {
                m: 4621,
                k: 768,
                n: 2304,
            },
            InferenceTile::F32N128S2,
        ),
        (
            InferenceShape {
                m: 4621,
                k: 1928,
                n: 384,
            },
            InferenceTile::Legacy,
        ),
        (
            InferenceShape {
                m: 2048,
                k: 768,
                n: 2304,
            },
            InferenceTile::Legacy,
        ),
        (
            InferenceShape {
                m: 2048,
                k: 2304,
                n: 768,
            },
            InferenceTile::Legacy,
        ),
        (InferenceShape { m: 65, k: 1, n: 65 }, InferenceTile::Legacy),
        (
            InferenceShape {
                m: 4620,
                k: 768,
                n: 2304,
            },
            InferenceTile::Legacy,
        ),
        (
            InferenceShape {
                m: 4622,
                k: 768,
                n: 2304,
            },
            InferenceTile::Legacy,
        ),
    ] {
        auto_phase(None, shape, false, expected, true).unwrap();
        auto_phase(Some("incumbent"), shape, false, expected, true).unwrap();
        for wrong in [
            InferenceTile::Legacy,
            InferenceTile::F32N128S2,
            InferenceTile::F32Sm120N64CopyPlan,
            InferenceTile::F32Sm120TmaFmaM128N64,
            InferenceTile::F32Sm120TmaFmaM64N128,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM64N128,
        ] {
            if wrong != expected {
                assert!(auto_phase(None, shape, false, wrong, true).is_err());
            }
        }
        let promoted = expected_auto("promoted", shape, false, true).unwrap();
        auto_phase(Some("promoted"), shape, false, promoted, true).unwrap();
        for wrong in [
            InferenceTile::Legacy,
            InferenceTile::F32N128S2,
            InferenceTile::F32Sm120N64CopyPlan,
            InferenceTile::F32Sm120TmaFmaM128N64,
            InferenceTile::F32Sm120TmaFmaM64N128,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM64N128,
        ] {
            if wrong != promoted {
                assert!(auto_phase(Some("promoted"), shape, false, wrong, true).is_err());
            }
        }
        for invalid in ["", "auto", "legacy", "candidate", "incumbent ", "promoted "] {
            assert!(auto_phase(Some(invalid), shape, false, expected, true).is_err());
        }
    }
}

#[test]
fn exact_n64_admission_cpu_current_auto_and_legacy_must_both_win() {
    assert!(own_admission("hot_b", 101, &[0, 1], &[0.8; 4], &[0.9; 4]));
    assert!(!own_admission("hot_b", 101, &[0, 1], &[0.8; 4], &[1.01; 4]));
    assert!(!own_admission("hot_b", 101, &[0, 1], &[1.01; 4], &[0.8; 4]));
    assert!(!own_admission("hot_b", 21, &[0, 1], &[0.8; 4], &[0.8; 4]));
    assert!(!own_admission("hot_b", 101, &[0], &[0.8; 4], &[0.8; 4]));
    assert!(!own_admission("hot_b", 101, &[0, 1], &[0.8; 4], &[0.8; 3]));
    for value in [1.0, 0.0, f64::NAN, f64::INFINITY] {
        assert!(!own_admission(
            "hot_b",
            101,
            &[0, 1],
            &[0.8; 4],
            &[value; 4]
        ));
    }
}

#[test]
fn exact_n64_admission_cpu_resources_decline_unknown_spills_and_residency() {
    let good = [0, 135, 32768, 128, 3, 100];
    resource_contract(good).unwrap();
    for (index, value) in [
        (0, 1),
        (0, -1),
        (1, 0),
        (1, 161),
        (2, 32767),
        (3, 127),
        (4, 2),
        (5, -1),
        (5, 0),
    ] {
        let mut bad = good;
        bad[index] = value;
        assert!(resource_contract(bad).is_err());
    }
}

#[test]
fn exact_n64_admission_cpu_screen_partial_paths_and_controls_never_admit() {
    assert!(admission("hot_e", 101, &[0, 1], &[0.9; 4]));
    assert!(!admission("hot_e", 21, &[0, 1], &[0.9; 4]));
    assert!(!admission("hot_e", 101, &[0], &[0.9; 2]));
    for cell in ["hot_a", "hot_c", "hot_d"] {
        assert!(!admission(cell, 101, &[0, 1], &[0.9; 4]));
    }
    for ratios in [vec![], vec![0.9; 3], vec![1.0; 4], vec![f64::NAN; 4]] {
        assert!(!admission("hot_b", 101, &[0, 1], &ratios));
    }
}

#[derive(Clone, Debug, PartialEq)]
struct LaunchProof {
    symbol: String,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared: u32,
    pointers: [u64; 4],
    parameters: [u32; 8],
    abi: Vec<(usize, usize)>,
    terminal_rejected: bool,
}
fn launch_contract(actual: &LaunchProof, expected: &LaunchProof) -> Result<(), String> {
    if actual != expected || !actual.terminal_rejected {
        return Err(format!(
            "physical graph/ABI mismatch: actual={actual:?}, expected={expected:?}"
        ));
    }
    Ok(())
}
fn guard_words(words: &[u32], active: usize) -> Result<(), String> {
    if active.checked_add(2 * GUARD) != Some(words.len()) {
        return Err("guarded storage extent mismatch".into());
    }
    if words[..GUARD]
        .iter()
        .chain(&words[GUARD + active..])
        .any(|word| *word != CANARY)
    {
        return Err("leading/trailing canary corrupted".into());
    }
    Ok(())
}
fn raw_equal(actual: &[u32], expected: &[u32]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err("raw output extent mismatch".into());
    }
    if let Some((i, (actual, expected))) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        return Err(format!(
            "raw output mismatch index={i} actual={actual:#010x} expected={expected:#010x}"
        ));
    }
    Ok(())
}

#[test]
fn exact_n64_admission_cpu_graph_contract_rejects_wrong_real_launch() {
    let expected = LaunchProof {
        symbol: CANDIDATE.into(),
        grid: (384, 1, 1),
        block: (128, 1, 1),
        shared: 0,
        pointers: [0x1000, 0x2000, 0x3000, 0],
        parameters: [0x3f800000, 0, 2048, 768, 2304, 2304, 768, 768],
        abi: vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)],
        terminal_rejected: true,
    };
    launch_contract(&expected, &expected).unwrap();
    let mut bad = Vec::new();
    let mut x = expected.clone();
    x.symbol = LEGACY.into();
    bad.push(x);
    let mut x = expected.clone();
    x.grid.0 = 383;
    bad.push(x);
    let mut x = expected.clone();
    x.block.0 = 256;
    bad.push(x);
    let mut x = expected.clone();
    x.shared = 32768;
    bad.push(x);
    let mut x = expected.clone();
    x.parameters.swap(3, 4);
    bad.push(x);
    let mut x = expected.clone();
    x.pointers[0] += 4;
    bad.push(x);
    let mut x = expected.clone();
    x.abi[4].1 = 28;
    bad.push(x);
    let mut x = expected.clone();
    x.abi.push((64, 4));
    bad.push(x);
    let mut x = expected.clone();
    x.terminal_rejected = false;
    bad.push(x);
    for actual in bad {
        assert!(launch_contract(&actual, &expected).is_err());
    }
}

#[test]
fn exact_n64_admission_cpu_guards_and_raw_gate_reject_poison_and_single_bits() {
    let mut words = vec![CANARY; GUARD * 2 + 2];
    words[GUARD..GUARD + 2].copy_from_slice(&[0x40c00000; 2]);
    guard_words(&words, 2).unwrap();
    for index in [0, GUARD - 1, GUARD + 2, words.len() - 1] {
        let mut corrupt = words.clone();
        corrupt[index] ^= 1;
        assert!(guard_words(&corrupt, 2).is_err());
    }
    assert!(guard_words(&words[..words.len() - 1], 2).is_err());
    raw_equal(&[0x40c00000], &[0x40c00000]).unwrap();
    for bad in [0xa5a5a5a5, 0x40c00001] {
        assert!(raw_equal(&[bad], &[0x40c00000]).is_err());
    }
    assert!(raw_equal(&[], &[0]).is_err());
}

#[test]
fn exact_n64_admission_cpu_bias_preseed_and_reordered_chain_do_not_certify() {
    // At 32768, the /1024 bias is a half-ULP and pre-seeding loses it.
    // Cancellation returns to zero, so adding bias afterwards retains 2^-10.
    let a = vec![1.0; 65536];
    let b: Vec<_> = (0..65536)
        .map(|k| if k < 32768 { 1.0 } else { -1.0 })
        .collect();
    let input = ProofInput {
        a: &a,
        b: &b,
        bias: Some(1.0 / 1024.0),
        k: 65536,
        a_start: 0,
        b_start: 0,
        b_stride: 1,
    };
    let proof = order_proof(input).unwrap();
    let preseeded = a
        .iter()
        .zip(&b)
        .fold(1.0f32 / 1024.0, |acc, (&a, &b)| a.mul_add(b, acc));
    assert_eq!(proof.ordered, 0x3a800000);
    assert_eq!(preseeded.to_bits(), 0);
    assert!(certificate(Corpus::Representable, [0, 0, 0x3a800000, 0x3a800000], input).is_err());
    let (a, b) = cancellation_vectors();
    let input = ProofInput {
        a: &a,
        b: &b,
        bias: None,
        k: 1928,
        a_start: 0,
        b_start: 0,
        b_stride: 1,
    };
    let proof = order_proof(input).unwrap();
    let reverse = a
        .iter()
        .zip(&b)
        .rev()
        .fold(0.0f32, |acc, (&a, &b)| a.mul_add(b, acc));
    assert_ne!(reverse.to_bits(), proof.ordered);
    let exact = (proof.exact as f32).to_bits();
    assert!(
        certificate(
            Corpus::Representable,
            [reverse.to_bits(), reverse.to_bits(), exact, exact],
            input
        )
        .is_err()
    );
    // /1024 inputs have exactly representable products. A non-fused witness
    // necessarily leaves that bounded proof corpus; it must not gain a waiver.
    let a = [1.0000001f32];
    let b = [0.9999999f32];
    assert_ne!(
        a[0].mul_add(b[0], -1.0).to_bits(),
        (a[0] * b[0] - 1.0).to_bits()
    );
    assert!(
        order_proof(ProofInput {
            a: &a,
            b: &b,
            bias: Some(-1.0),
            k: 1,
            a_start: 0,
            b_start: 0,
            b_stride: 1
        })
        .is_err()
    );
}

#[test]
fn exact_n64_admission_cpu_filters_fail_closed() {
    assert_eq!(
        selection(Some("hot_e,hot_b"), &["hot_a", "hot_b", "hot_e"]).unwrap(),
        [2, 1]
    );
    for bad in ["", "hot_e,hot_e", "hot_e,", " hot_e", "hot_x"] {
        assert!(
            selection(Some(bad), &["hot_b", "hot_e"]).is_err(),
            "accepted {bad:?}"
        );
    }
}

#[test]
fn exact_n64_admission_cpu_windows_distinguish_screen_from_admission() {
    assert_eq!(window_count(Some("21")).unwrap(), 21);
    assert_eq!(window_count(None).unwrap(), 101);
    for bad in ["0", "1", "100", "102", "", " 101"] {
        assert!(window_count(Some(bad)).is_err());
    }
}

#[test]
fn exact_n64_admission_cpu_protocol_is_true_four_position_pairing() {
    assert_eq!(slots("ABBA").unwrap(), [0, 1, 1, 0]);
    assert_eq!(slots("BAAB").unwrap(), [1, 0, 0, 1]);
    assert!(slots("ABAB").is_err());
    assert_eq!(paired_ratio("ABBA", [2., 10., 6., 6.]).unwrap(), 0.5);
    assert_eq!(paired_ratio("BAAB", [10., 2., 6., 6.]).unwrap(), 0.5);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(paired_ratio("ABBA", [bad, 1., 1., 1.]).is_err());
    }
}

fn cancellation_vectors() -> (Vec<f32>, Vec<f32>) {
    (
        (0..1928)
            .map(|k| seed(549 * 1928 + k, 0x0adaa001, Corpus::Representable))
            .collect(),
        (0..1928)
            .map(|k| seed(k * 384 + 319, 0x0adab001, Corpus::Representable))
            .collect(),
    )
}

#[test]
fn exact_n64_admission_cpu_proof_reproduces_frozen_cancellation_bits() {
    let (a, b) = cancellation_vectors();
    let input = ProofInput {
        a: &a,
        b: &b,
        bias: None,
        k: 1928,
        a_start: 0,
        b_start: 0,
        b_stride: 1,
    };
    let proof = order_proof(input).unwrap();
    assert_eq!(proof.ordered, 0xbd2baf00);
    assert_eq!(proof.exact, -43710. / 1048576.);
    let words = [0xbd2baf00, 0xbd2baf00, 0xbd2abd00, 0xbd2abd00];
    certificate(Corpus::Representable, words, input).unwrap();
    for index in 0..4 {
        let mut wrong = words;
        wrong[index] = if index < 2 { wrong[index] ^ 1 } else { 0 };
        assert!(
            certificate(Corpus::Representable, wrong, input).is_err(),
            "accepted mutation at {index}"
        );
    }
    assert!(certificate(Corpus::Nonrepresentable, words, input).is_err());
}

#[test]
fn exact_n64_admission_cpu_proof_uses_actual_view_stride_and_post_bias() {
    let a = [99., 0.5, 0.25, 99.];
    let b = [99., 0.5, 99., 99., -0.5, 99.];
    let input = ProofInput {
        a: &a,
        b: &b,
        bias: Some(0.5),
        k: 2,
        a_start: 1,
        b_start: 1,
        b_stride: 3,
    };
    let proof = order_proof(input).unwrap();
    assert_eq!(proof.ordered, 0x3f200000); // 0.5*0.5 + 0.25*(-0.5) + 0.5 = 0.625.
    assert_eq!(proof.exact, 0.625);
    assert!(
        order_proof(ProofInput {
            a_start: 0,
            ..input
        })
        .is_err()
    );
    assert!(
        order_proof(ProofInput {
            b_stride: 2,
            ..input
        })
        .is_err()
    );
    assert!(
        order_proof(ProofInput {
            b_start: usize::MAX,
            ..input
        })
        .is_err()
    );
    assert_ne!(
        order_proof(ProofInput {
            bias: None,
            ..input
        })
        .unwrap()
        .ordered,
        proof.ordered
    );
    for k in [0, 31, 32, 33] {
        let a = vec![0.5; k];
        let b = vec![-0.5; k];
        let p = order_proof(ProofInput {
            a: &a,
            b: &b,
            bias: Some(0.5),
            k,
            a_start: 0,
            b_start: 0,
            b_stride: 1,
        })
        .unwrap();
        assert_eq!(p.ordered, (-0.25 * k as f32 + 0.5).to_bits());
        assert_eq!(p.exact, -0.25 * k as f64 + 0.5);
    }
}

#[test]
fn exact_n64_admission_cpu_proof_rejects_foreign_corpus_and_bounds() {
    for bad in [0.1, 2.0, f32::NAN, f32::INFINITY] {
        let a = [bad];
        let b = [1.0];
        assert!(
            order_proof(ProofInput {
                a: &a,
                b: &b,
                bias: None,
                k: 1,
                a_start: 0,
                b_start: 0,
                b_stride: 1
            })
            .is_err()
        );
    }
    assert!(
        order_proof(ProofInput {
            a: &[],
            b: &[],
            bias: None,
            k: i32::MAX as usize,
            a_start: 0,
            b_start: 0,
            b_stride: 1
        })
        .is_err()
    );
}

const SCHEMA: &str = "MambaBiFixedSm120ExactN64AdmissionV1";
const POISON: u32 = 0xa5a5a5a5;
const MISSING_LAUNCH_RED: &str = "SM120_EXACT_N64_MISSING_LAUNCH_RED";

fn poison_red_requested(value: Option<&str>) -> Result<bool, String> {
    match value {
        None => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "MAMBA_FIXED_SM120_EXACT_N64_POISON_RED only accepts 1, got {other:?}"
        )),
    }
}
fn single_term_output_gate(words: &[u32], expected: u32, red: bool) -> Result<(), String> {
    if red {
        if words.is_empty() || expected == POISON || words.iter().any(|word| *word != POISON) {
            return Err(
                "missing-launch switch did not leave the entire active output poisoned".into(),
            );
        }
        return Err(format!(
            "{MISSING_LAUNCH_RED} index=0 actual={POISON:#010x} expected={expected:#010x} all_active_words_poisoned={}",
            words.len()
        ));
    }
    if words.is_empty() {
        return Err("empty single-term output".into());
    }
    raw_equal(words, &vec![expected; words.len()])
}

#[test]
fn exact_n64_admission_cpu_missing_launch_red_requires_actual_poison() {
    assert!(!poison_red_requested(None).unwrap());
    assert!(poison_red_requested(Some("1")).unwrap());
    for bad in ["0", "", "true", " 1"] {
        assert!(poison_red_requested(Some(bad)).is_err());
    }
    single_term_output_gate(&[0x40c00000; 4], 0x40c00000, false).unwrap();
    let red = single_term_output_gate(&[POISON; 4], 0x40c00000, true).unwrap_err();
    assert!(red.contains(MISSING_LAUNCH_RED));
    for words in [vec![0x40c00000; 4], vec![POISON, 0x40c00000], vec![]] {
        let reason = single_term_output_gate(&words, 0x40c00000, true).unwrap_err();
        assert!(!reason.contains(MISSING_LAUNCH_RED));
    }
    assert!(
        !single_term_output_gate(&[POISON], 0x40c00000, false)
            .unwrap_err()
            .contains(MISSING_LAUNCH_RED)
    );
}
const ARMS: [&str; 6] = [
    "candidate",
    "current_auto",
    "explicit_legacy",
    "pedantic",
    "old_oracle",
    "reference",
];
const PATHS: [&str; 2] = ["eager", "graph"];
const ORDERS: [&str; 2] = ["ABBA", "BAAB"];
const CELLS: [(&str, InferenceShape); 5] = [
    (
        "hot_a",
        InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
    ),
    (
        "hot_b",
        InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
    ),
    (
        "hot_c",
        InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
    ),
    (
        "hot_d",
        InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
    ),
    (
        "hot_e",
        InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ),
];

fn quoted(value: &str) -> String {
    format!("\"{}\"", super::sm120_exact_json_escape(value))
}
fn env(name: &str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("{name}: {error}")),
    }
}
fn cuda(status: sys::CUresult, label: &str) -> Result<(), String> {
    if status == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{label}: {status:?}"))
    }
}
fn words_digest(words: &[u32]) -> String {
    let mut hash = Sha256::new();
    for word in words {
        hash.update(word.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

struct Evidence {
    candidate: Candidate,
    diagnostic_fast: bool,
    writer: BufWriter<std::fs::File>,
    digest: Sha256,
    lines: usize,
    pair_keys: Vec<String>,
    environment: EnvironmentIdentity,
}
impl Evidence {
    fn new(preflight: &str, candidate: Candidate, diagnostic_fast: bool) -> Result<Self, String> {
        let environment = environment_identity(preflight)?;
        let path = env("MAMBA_FIXED_SM120_EXACT_N64_JSONL")?
            .ok_or("set MAMBA_FIXED_SM120_EXACT_N64_JSONL to a new evidence file")?;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            candidate,
            diagnostic_fast,
            writer: BufWriter::new(file),
            digest: Sha256::new(),
            lines: 0,
            pair_keys: Vec::new(),
            environment,
        })
    }
    fn telemetry(&self, label: &str) -> Result<String, String> {
        let value = telemetry(label)?;
        environment_unchanged(&self.environment, &value)?;
        Ok(value)
    }
    fn emit(&mut self, fields: String) -> Result<(), String> {
        let line = format!(
            "{{\"schema\":\"{SCHEMA}\",\"candidate\":\"{:?}\",\"candidate_symbol\":{},\"diagnostic_fast\":{},\"admission_enabled\":{}, {fields}}}\n",
            self.candidate.tile(),
            quoted(self.candidate.symbol()),
            self.diagnostic_fast,
            !self.diagnostic_fast,
        );
        self.writer
            .write_all(line.as_bytes())
            .map_err(|e| e.to_string())?;
        self.writer.flush().map_err(|e| e.to_string())?;
        self.digest.update(line.as_bytes());
        self.lines += 1;
        print!("{line}");
        Ok(())
    }
    fn pair(&mut self, key: String, fields: String) -> Result<(), String> {
        if self.pair_keys.contains(&key) {
            return Err(format!("duplicate pair {key}"));
        }
        self.emit(fields)?;
        self.pair_keys.push(key);
        Ok(())
    }
    fn complete(mut self, records: usize, expected_keys: &[String]) -> Result<(), String> {
        complete_inventory(&self.pair_keys, expected_keys)?;
        let expected = expected_keys.len();
        if records != expected {
            return Err(format!("incomplete pair inventory: {records}/{expected}"));
        }
        let digest = format!("{:x}", self.digest.clone().finalize());
        self.emit(format!("\"record\":\"complete\",\"pair_records\":{records},\"expected_pair_records\":{expected},\"preceding_lines\":{},\"preceding_jsonl_sha256\":{},\"all_gates_passed\":true", self.lines, quoted(&digest)))?;
        self.writer.get_ref().sync_all().map_err(|e| e.to_string())
    }
}

struct Guarded {
    device: DtypedBuf,
    initial: Vec<f32>,
    active: usize,
}
impl Guarded {
    fn new(ctx: &GpuCtx, active: Vec<f32>) -> Result<Self, String> {
        let mut initial = vec![
            f32::from_bits(CANARY);
            active
                .len()
                .checked_add(2 * GUARD)
                .ok_or("storage length overflow")?
        ];
        initial[GUARD..GUARD + active.len()].copy_from_slice(&active);
        let device = DtypedBuf::zeros(&ctx.stream, initial.len(), WeightDtype::F32)?;
        device.upload_f32(&ctx.stream, &initial)?;
        Ok(Self {
            device,
            initial,
            active: active.len(),
        })
    }
    fn ptr(&self) -> u64 {
        self.device.cached_ptr() + (GUARD * 4) as u64
    }
    fn reset(&self, ctx: &GpuCtx) -> Result<(), String> {
        self.device.upload_f32(&ctx.stream, &self.initial)
    }
    fn all(&self, ctx: &GpuCtx) -> Result<Vec<u32>, String> {
        let mut host = vec![0.0; self.initial.len()];
        self.device.download_f32(&ctx.stream, &mut host)?;
        ctx.stream
            .synchronize()
            .map_err(|e| format!("guarded D2H synchronize: {e:?}"))?;
        let words: Vec<_> = host.into_iter().map(f32::to_bits).collect();
        guard_words(&words, self.active)?;
        Ok(words)
    }
    fn read(&self, ctx: &GpuCtx) -> Result<Vec<u32>, String> {
        let words = self.all(ctx)?;
        Ok(words[GUARD..GUARD + self.active].to_vec())
    }
    fn unchanged(&self, ctx: &GpuCtx) -> Result<(), String> {
        let actual = self.all(ctx)?;
        let expected: Vec<_> = self.initial.iter().map(|v| v.to_bits()).collect();
        raw_equal(&actual, &expected)
    }
}

struct Case {
    candidate: Candidate,
    shape: InferenceShape,
    corpus: Corpus,
    has_bias: bool,
    a: Guarded,
    b: Guarded,
    bias: Guarded,
    outputs: Vec<Guarded>,
    auto: Cell<Option<InferenceTile>>,
}
impl Case {
    fn new(
        ctx: &GpuCtx,
        shape: InferenceShape,
        corpus: Corpus,
        has_bias: bool,
        candidate: Candidate,
    ) -> Result<Self, String> {
        let a = Guarded::new(
            ctx,
            (0..shape.m * shape.k)
                .map(|i| seed(i, 0x0adaa001, corpus))
                .collect(),
        )?;
        let b = Guarded::new(
            ctx,
            (0..shape.k * shape.n)
                .map(|i| seed(i, 0x0adab001, corpus))
                .collect(),
        )?;
        let bias = Guarded::new(
            ctx,
            (0..shape.n).map(|i| seed(i, 0x0adab1a5, corpus)).collect(),
        )?;
        let outputs = (0..6)
            .map(|_| Guarded::new(ctx, vec![f32::from_bits(POISON); shape.m * shape.n]))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            candidate,
            shape,
            corpus,
            has_bias,
            a,
            b,
            bias,
            outputs,
            auto: Cell::new(None),
        })
    }
    fn operands(&self, arm: usize) -> InferenceFwdOperands {
        InferenceFwdOperands {
            c: TypedPtr {
                ptr: self.outputs[arm].ptr(),
                dtype: WeightDtype::F32,
            },
            x: TypedPtr {
                ptr: self.a.ptr(),
                dtype: WeightDtype::F32,
            },
            w: TypedPtr {
                ptr: self.b.ptr(),
                dtype: WeightDtype::F32,
            },
            bias_ptr: self.has_bias.then(|| self.bias.ptr()),
        }
    }
    fn launch(&self, ctx: &GpuCtx, arm: usize) -> Result<(), String> {
        let operands = self.operands(arm);
        match arm {
            0 => inference_forward_with_tile(ctx, operands, self.shape, self.candidate.tile()),
            1 => {
                let selected = launch_sm120_exact_auto(ctx, operands, self.shape);
                if self.auto.get().is_some_and(|previous| previous != selected) {
                    return Err("AUTO route drift".into());
                }
                self.auto.set(Some(selected));
                Ok(())
            }
            2 => inference_forward_with_tile(ctx, operands, self.shape, InferenceTile::Legacy),
            4 => inference_forward_f32_legacy_baseline(ctx, operands, self.shape),
            3 | 5 => {
                // Every invocation includes bias broadcast + beta1, or beta0
                // without bias. Reference has its own separately reset output.
                sm120_exact_vendor_launch(
                    ctx,
                    operands,
                    self.shape,
                    cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
                );
                Ok(())
            }
            _ => Err("unknown exact admission arm".into()),
        }
    }
    fn inputs(&self, ctx: &GpuCtx) -> Result<(), String> {
        self.a.unchanged(ctx)?;
        self.b.unchanged(ctx)?;
        self.bias.unchanged(ctx)?;
        // All six actual launch operands share these exact active pointers.
        for arm in 0..6 {
            let op = self.operands(arm);
            if op.x.ptr != self.a.ptr()
                || op.w.ptr != self.b.ptr()
                || op.bias_ptr != self.has_bias.then(|| self.bias.ptr())
            {
                return Err("common input pointer proof failed".into());
            }
        }
        Ok(())
    }
    fn proof(&self, index: usize) -> ProofInput<'_> {
        ProofInput {
            a: &self.a.initial,
            b: &self.b.initial,
            bias: self
                .has_bias
                .then(|| self.bias.initial[GUARD + index % self.shape.n]),
            k: self.shape.k,
            a_start: GUARD + (index / self.shape.n) * self.shape.k,
            b_start: GUARD + index % self.shape.n,
            b_stride: self.shape.n,
        }
    }
    fn read_all(&self, ctx: &GpuCtx) -> Result<Vec<Vec<u32>>, String> {
        self.inputs(ctx)?;
        self.outputs.iter().map(|buffer| buffer.read(ctx)).collect()
    }
}

// Read only; never apply the standalone harness's MaxShared preference to an incumbent.
fn resources(function: &CudaFunction, threads: u32) -> Result<[i32; 6], String> {
    let error = |e| format!("live function attributes: {e:?}");
    Ok([
        function.local_size_bytes().map_err(error)?,
        function.num_regs().map_err(error)?,
        function.shared_size_bytes().map_err(error)?,
        function.max_threads_per_block().map_err(error)?,
        i32::try_from(
            function
                .occupancy_max_active_blocks_per_multiprocessor(threads, 0, None)
                .map_err(error)?,
        )
        .map_err(|e| e.to_string())?,
        function
            .get_attribute(
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            )
            .map_err(error)?,
    ])
}
fn incumbent_carveouts(values: [i32; 3]) -> Result<(), String> {
    if values != [-1; 3] {
        return Err(format!(
            "SM120 incumbents must retain default carveouts: {values:?}"
        ));
    }
    Ok(())
}

#[test]
fn exact_n64_admission_cpu_incumbent_default_carveout_is_not_candidate_policy() {
    incumbent_carveouts([-1; 3]).unwrap();
    for index in 0..3 {
        for value in [0, 100] {
            let mut observed = [-1; 3];
            observed[index] = value;
            assert!(incumbent_carveouts(observed).is_err());
        }
    }
}

fn resource_snapshot(ctx: &GpuCtx, selected: Candidate) -> Result<[[i32; 6]; 4], String> {
    let (holder, rejection) = match selected {
        Candidate::CopyPlan => (
            &ctx.kernels.fixed_sm120_f32_n64_copyplan,
            &ctx.kernels.fixed_sm120_f32_n64_copyplan_rejection,
        ),
        Candidate::Sliced => (
            &ctx.kernels.fixed_sm120_f32_n64_sliced,
            &ctx.kernels.fixed_sm120_f32_n64_sliced_rejection,
        ),
    };
    let candidate = holder
        .as_ref()
        .ok_or_else(|| format!("{} not admitted: {rejection:?}", selected.symbol()))?;
    let snapshot = [
        resources(candidate, 128)?,
        resources(&ctx.kernels.gemm_bi_f32_f32_s2, 128)?,
        resources(&ctx.kernels.gemm_bi_f32_f32_n128_s2, 256)?,
        resources(&ctx.kernels.gemm_bi_f32_f32, 256)?,
    ];
    resource_contract(snapshot[0])?;
    incumbent_carveouts([snapshot[1][5], snapshot[2][5], snapshot[3][5]])?;
    Ok(snapshot)
}

fn abi_layout(function: sys::CUfunction, count: usize) -> Result<Vec<(usize, usize)>, String> {
    let mut layout = Vec::new();
    for index in 0..count {
        let (mut offset, mut size) = (0, 0);
        cuda(
            unsafe { sys::cuFuncGetParamInfo(function, index, &mut offset, &mut size) },
            "captured Driver parameter ABI",
        )?;
        layout.push((offset, size));
    }
    let (mut offset, mut size) = (0, 0);
    let terminal = unsafe { sys::cuFuncGetParamInfo(function, count, &mut offset, &mut size) };
    if terminal != sys::CUresult::CUDA_ERROR_INVALID_VALUE {
        return Err(format!("unexpected terminal parameter result {terminal:?}"));
    }
    Ok(layout)
}

fn captured<T: Copy>(params: &sys::CUDA_KERNEL_NODE_PARAMS_v2, index: usize) -> Result<T, String> {
    if params.kernelParams.is_null() || !params.extra.is_null() {
        return Err("own graph kernel must expose parameter storage".into());
    }
    // Graph remains alive, and the actual live ABI has been checked before these reads.
    let pointer = unsafe { *params.kernelParams.add(index) };
    if pointer.is_null() {
        return Err(format!("missing captured argument {index}"));
    }
    Ok(unsafe { std::ptr::read_unaligned(pointer.cast::<T>()) })
}
fn graph_nodes(graph: sys::CUgraph) -> Result<Vec<sys::CUgraphNode>, String> {
    let mut count = 0;
    cuda(
        unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut count) },
        "graph node count",
    )?;
    if count == 0 {
        return Err("empty captured workflow".into());
    }
    let mut nodes = vec![std::ptr::null_mut(); count];
    cuda(
        unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count) },
        "graph node inventory",
    )?;
    if count != nodes.len() {
        return Err("captured graph inventory drift".into());
    }
    Ok(nodes)
}
fn kernel_params(
    node: sys::CUgraphNode,
) -> Result<(String, sys::CUDA_KERNEL_NODE_PARAMS_v2), String> {
    let mut params = unsafe { std::mem::zeroed() };
    cuda(
        unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
        "Driver graph kernel params",
    )?;
    let mut name = std::ptr::null();
    cuda(
        unsafe { sys::cuFuncGetName(&mut name, params.func) },
        "Driver graph actual symbol",
    )?;
    if name.is_null() {
        return Err("null graph symbol".into());
    }
    let name = unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|e| e.to_string())?
        .to_owned();
    Ok((name, params))
}

#[cfg(test)]
fn own_kernel(
    arm: usize,
    auto: Option<InferenceTile>,
) -> Result<(&'static str, usize, u32), String> {
    own_kernel_for_candidate(arm, auto, Candidate::CopyPlan)
}

fn own_kernel_for_candidate(
    arm: usize,
    auto: Option<InferenceTile>,
    candidate: Candidate,
) -> Result<(&'static str, usize, u32), String> {
    match (arm, auto) {
        (0, _) => Ok((candidate.symbol(), 64, 128)),
        (1, Some(InferenceTile::F32Sm120N64CopyPlan)) => Ok((CANDIDATE, 64, 128)),
        (1, Some(InferenceTile::F32Sm120M128N64CopyPlanT256)) => {
            Ok(("nn_sm120_f32_n64_copyplan_m128n64_t256", 64, 256))
        }
        (1, Some(InferenceTile::F32Sm120N64CopyPlanT256)) => {
            Ok(("nn_sm120_f32_n64_copyplan_t256", 64, 256))
        }
        (1, Some(InferenceTile::F32Sm120N64Sliced)) => Ok(("nn_sm120_f32_n64_sliced", 64, 128)),
        (1, Some(InferenceTile::F32Sm120TmaFmaM128N64)) => {
            Ok(("nn_sm120_tma_fma_m128n64_bk16_s2", 64, 128))
        }
        (1, Some(InferenceTile::F32Sm120TmaFmaM64N128)) => {
            Ok(("nn_sm120_tma_fma_m64n128_bk16_s2", 128, 128))
        }
        (1, Some(InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64)) => {
            Ok(("nn_sm120_tma_fma_postbias_m128n64_bk16_s2", 64, 128))
        }
        (1, Some(InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64K4)) => {
            Ok(("nn_sm120_tma_fma_postbias_m128n64_bk16_s2_k4", 64, 128))
        }
        (1, Some(InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256)) => {
            Ok(("nn_sm120_tma_fma_postbias_m128n64_t256_bk16_s2", 64, 256))
        }
        (1, Some(InferenceTile::F32Sm120TmaFmaFixedNoBiasM128N64T256)) => {
            Ok(("nn_sm120_tma_fma_nobias_m128n64_t256_bk16_s2", 64, 256))
        }
        (1, Some(InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96)) => {
            Ok(("nn_sm120_tma_fma_postbias_m128n96_bk16_s2", 96, 256))
        }
        (1, Some(InferenceTile::F32Sm120TmaFmaFixedPostBiasM64N128)) => {
            Ok(("nn_sm120_tma_fma_postbias_m64n128_bk16_s2", 128, 128))
        }
        (1, Some(InferenceTile::F32N128S2)) => Ok((N128, 128, 256)),
        (1, Some(InferenceTile::Legacy)) | (2, _) => Ok((LEGACY, 64, 128)),
        (4, _) => Ok((ORACLE, 64, 256)),
        _ => Err("missing or unqualified SM120 own route".into()),
    }
}

#[test]
fn exact_n64_admission_cpu_copyplan_t256_force_mapping_preserves_auto() {
    assert_eq!(
        own_kernel(1, Some(InferenceTile::F32Sm120M128N64CopyPlanT256)).unwrap(),
        ("nn_sm120_f32_n64_copyplan_m128n64_t256", 64, 256)
    );
    assert_eq!(
        own_kernel(1, Some(InferenceTile::F32Sm120N64CopyPlanT256)).unwrap(),
        ("nn_sm120_f32_n64_copyplan_t256", 64, 256)
    );
    assert_eq!(
        expected_auto(
            "promoted",
            InferenceShape {
                m: 2048,
                k: 2304,
                n: 768
            },
            false,
            true
        )
        .unwrap(),
        InferenceTile::F32Sm120N64CopyPlan
    );
}

#[test]
fn exact_n64_admission_cpu_nobias_t256_force_mapping_is_distinct() {
    assert_eq!(
        own_kernel(1, Some(InferenceTile::F32Sm120TmaFmaFixedNoBiasM128N64T256)).unwrap(),
        ("nn_sm120_tma_fma_nobias_m128n64_t256_bk16_s2", 64, 256)
    );
    assert_eq!(
        expected_auto(
            "promoted",
            InferenceShape {
                m: 2048,
                k: 2304,
                n: 768
            },
            false,
            true
        )
        .unwrap(),
        InferenceTile::F32Sm120N64CopyPlan,
    );
}

#[test]
fn exact_n64_admission_cpu_n128_graph_geometry_is_not_legacy() {
    let (symbol, tile_n, threads) = own_kernel(1, Some(InferenceTile::F32N128S2)).unwrap();
    assert_eq!(symbol, "f32_f32_n128_s2");
    assert_eq!(tile_n, 128);
    assert_eq!(threads, 256);
    assert_eq!(4621usize.div_ceil(64) * 2304usize.div_ceil(tile_n), 1314);
    assert_eq!(
        own_kernel(2, Some(InferenceTile::F32N128S2)).unwrap(),
        (LEGACY, 64, 128)
    );
    assert_eq!(
        own_kernel(1, Some(InferenceTile::F32Sm120N64CopyPlan)).unwrap(),
        (CANDIDATE, 64, 128)
    );
    assert_eq!(
        own_kernel(1, Some(InferenceTile::F32Sm120TmaFmaM64N128)).unwrap(),
        ("nn_sm120_tma_fma_m64n128_bk16_s2", 128, 128)
    );
    assert_eq!(
        own_kernel(1, Some(InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64K4)).unwrap(),
        ("nn_sm120_tma_fma_postbias_m128n64_bk16_s2_k4", 64, 128)
    );
    assert!(own_kernel(1, None).is_err());
    assert_eq!(
        own_kernel(
            1,
            Some(InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256)
        )
        .unwrap(),
        ("nn_sm120_tma_fma_postbias_m128n64_t256_bk16_s2", 64, 256)
    );
    assert_eq!(
        own_kernel(1, Some(InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96)).unwrap(),
        ("nn_sm120_tma_fma_postbias_m128n96_bk16_s2", 96, 256)
    );
    let expected = LaunchProof {
        symbol: symbol.into(),
        grid: (1314, 1, 1),
        block: (256, 1, 1),
        shared: 0,
        pointers: [0x1000, 0x2000, 0x3000, 0],
        parameters: [0x3f800000, 0, 4621, 2304, 768, 768, 2304, 2304],
        abi: (0..4)
            .map(|i| (i * 8, 8))
            .chain((0..8).map(|i| (32 + i * 4, 4)))
            .collect(),
        terminal_rejected: true,
    };
    launch_contract(&expected, &expected).unwrap();
    for field in 0..8 {
        let mut wrong = expected.clone();
        match field {
            0 => wrong.symbol = LEGACY.into(),
            1 => wrong.grid.0 = 2628,
            2 => wrong.block.0 = 128,
            3 => wrong.abi = vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)],
            4 => wrong.parameters[1] = (-0.0f32).to_bits(),
            5 => wrong.parameters[6] += 1,
            6 => wrong.pointers[1] += 4,
            _ => wrong.terminal_rejected = false,
        }
        assert!(launch_contract(&wrong, &expected).is_err());
    }
}

#[test]
fn exact_n64_admission_cpu_a1_promotion_requires_loaded_t256_with_control_fallback() {
    let shape = InferenceShape {
        m: 4621,
        k: 384,
        n: 1928,
    };
    for (loaded, expected, rejected) in [
        (
            true,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64,
        ),
        (
            false,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256,
        ),
    ] {
        assert_eq!(
            expected_auto("promoted", shape, true, loaded).unwrap(),
            expected
        );
        auto_phase(Some("promoted"), shape, true, expected, loaded).unwrap();
        assert!(auto_phase(Some("promoted"), shape, true, rejected, loaded).is_err());
        let (symbol, tile_n, threads) = own_kernel(1, Some(expected)).unwrap();
        assert_eq!(tile_n, 64);
        assert_eq!(threads, if loaded { 256 } else { 128 });
        assert_eq!(
            symbol,
            if loaded {
                "nn_sm120_tma_fma_postbias_m128n64_t256_bk16_s2"
            } else {
                "nn_sm120_tma_fma_postbias_m128n64_bk16_s2"
            }
        );
    }
}

#[test]
fn exact_n64_sliced_cpu_promoted_route_retains_copyplan_and_graph_compact() {
    for (m, k, n, bias, want) in [
        (2048, 768, 2304, false, InferenceTile::F32Sm120N64CopyPlan),
        (2048, 768, 2304, true, InferenceTile::F32Sm120N64CopyPlan),
        (4621, 768, 2304, false, InferenceTile::F32Sm120TmaFmaM128N64),
        (
            4621,
            768,
            2304,
            true,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64,
        ),
        (4621, 1928, 384, false, InferenceTile::Legacy),
        (
            4621,
            1928,
            384,
            true,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96,
        ),
        (2048, 2304, 768, false, InferenceTile::F32Sm120N64CopyPlan),
        (2048, 2304, 768, true, InferenceTile::F32Sm120N64CopyPlan),
        (4621, 384, 1928, false, InferenceTile::F32Sm120TmaFmaM64N128),
        (
            4621,
            384,
            1928,
            true,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256,
        ),
        (4620, 768, 2304, false, InferenceTile::Legacy),
        (4622, 768, 2304, false, InferenceTile::Legacy),
    ] {
        assert_eq!(
            expected_auto("promoted", InferenceShape { m, k, n }, bias, true).unwrap(),
            want
        );
    }
    assert_eq!(
        own_kernel(1, Some(InferenceTile::F32Sm120N64Sliced)).unwrap(),
        ("nn_sm120_f32_n64_sliced", 64, 128)
    );
    // Explicit candidate stays old copyplan. A different AUTO is not a self comparison.
    assert_eq!(
        own_kernel(0, None).unwrap(),
        ("nn_sm120_f32_n64_copyplan", 64, 128)
    );
    assert_eq!(own_kernel(2, None).unwrap(), ("f32_f32_s2", 64, 128));
}

#[test]
fn exact_n64_candidate_cpu_graph_candidate_and_auto_are_independent() {
    for (candidate, symbol) in [
        (Candidate::CopyPlan, "nn_sm120_f32_n64_copyplan"),
        (Candidate::Sliced, "nn_sm120_f32_n64_sliced"),
    ] {
        assert_eq!(
            own_kernel_for_candidate(0, None, candidate).unwrap(),
            (symbol, 64, 128)
        );
        assert_eq!(
            own_kernel_for_candidate(1, Some(InferenceTile::F32Sm120N64Sliced), candidate).unwrap(),
            ("nn_sm120_f32_n64_sliced", 64, 128)
        );
        assert_eq!(
            own_kernel_for_candidate(1, Some(InferenceTile::F32Sm120N64CopyPlan), candidate)
                .unwrap(),
            ("nn_sm120_f32_n64_copyplan", 64, 128)
        );
        assert_eq!(
            own_kernel_for_candidate(2, None, candidate).unwrap(),
            ("f32_f32_s2", 64, 128)
        );
        assert!(own_kernel_for_candidate(1, None, candidate).is_err());
        assert!(own_kernel_for_candidate(1, Some(InferenceTile::Tf32M128S2), candidate).is_err());
    }
}

fn own_graph(graph: &CudaGraph, case: &Case, arm: usize) -> Result<String, String> {
    let nodes = graph_nodes(graph.cu_graph())?;
    if nodes.len() != 1 {
        return Err("own workflow must capture exactly one kernel".into());
    }
    let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
    cuda(
        unsafe { sys::cuGraphNodeGetType(nodes[0], &mut kind) },
        "own graph node kind",
    )?;
    if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
        return Err("non-kernel node in own graph".into());
    }
    let (symbol, params) = kernel_params(nodes[0])?;
    let (expected_symbol, tile_n, threads) =
        own_kernel_for_candidate(arm, case.auto.get(), case.candidate)?;
    let exact_tma_tile = match expected_symbol {
        "nn_sm120_tma_fma_m128n64_bk16_s2" => Some((128usize, 64usize)),
        "nn_sm120_tma_fma_m64n128_bk16_s2" => Some((64, 128)),
        "nn_sm120_tma_fma_postbias_m128n64_bk16_s2" => Some((128, 64)),
        "nn_sm120_tma_fma_postbias_m128n64_bk16_s2_k4" => Some((128, 64)),
        "nn_sm120_tma_fma_postbias_m128n64_t256_bk16_s2" => Some((128, 64)),
        "nn_sm120_tma_fma_nobias_m128n64_t256_bk16_s2" => Some((128, 64)),
        "nn_sm120_tma_fma_postbias_m128n96_bk16_s2" => Some((128, 96)),
        "nn_sm120_tma_fma_postbias_m64n128_bk16_s2" => Some((64, 128)),
        _ => None,
    };
    if let Some((tile_m, tile_n)) = exact_tma_tile {
        let expected_abi = vec![
            (0, 8),
            (8, 8),
            (16, 8),
            (128, 128),
            (256, 128),
            (384, 8),
            (392, 32),
        ];
        let layout = abi_layout(params.func, 7)?;
        if layout != expected_abi {
            return Err(format!("captured exact-TMA ABI differs: {layout:?}"));
        }
        let output: u64 = captured(&params, 0)?;
        let partial: u64 = captured(&params, 1)?;
        let flags: u64 = captured(&params, 2)?;
        let map_a: [u8; 128] = captured(&params, 3)?;
        let map_b: [u8; 128] = captured(&params, 4)?;
        let bias: u64 = captured(&params, 5)?;
        let values: [u32; 8] = captured(&params, 6)?;
        let s = case.shape;
        let ops = case.operands(arm);
        let expected_grid = ((s.m.div_ceil(tile_m) * s.n.div_ceil(tile_n)) as u32, 1, 1);
        let actual_grid = (params.gridDimX, params.gridDimY, params.gridDimZ);
        let actual_block = (params.blockDimX, params.blockDimY, params.blockDimZ);
        let expected_shared = if tile_n == 96 { 28_688 } else { 24_592 };
        let expected_values = [
            1.0f32.to_bits(),
            0,
            s.m as u32,
            s.n as u32,
            s.k as u32,
            s.n as u32,
            1,
            s.k.div_ceil(16) as u32,
        ];
        if symbol != expected_symbol
            || actual_grid != expected_grid
            || actual_block != (threads, 1, 1)
            || params.sharedMemBytes != expected_shared
            || output != ops.c.ptr
            || partial != 0
            || flags != 0
            || bias != ops.bias_ptr.unwrap_or(0)
            || map_a.iter().all(|byte| *byte == 0)
            || map_b.iter().all(|byte| *byte == 0)
            || values != expected_values
        {
            return Err(format!(
                "captured exact-TMA launch differs: symbol={symbol:?} grid={actual_grid:?} block={actual_block:?} shared={} output={output:#x} partial={partial:#x} flags={flags:#x} bias={bias:#x} values={values:?}",
                params.sharedMemBytes,
            ));
        }
        return Ok(format!(
            "{{\"symbol\":{},\"grid\":[{},{},{}],\"block\":[{},{},{}],\"dynamic_shared\":{},\"output\":{},\"partial\":{},\"flags\":{},\"bias\":{},\"parameter_words\":{:?},\"driver_abi\":{:?},\"terminal_probe\":true}}",
            quoted(&symbol),
            actual_grid.0,
            actual_grid.1,
            actual_grid.2,
            actual_block.0,
            actual_block.1,
            actual_block.2,
            params.sharedMemBytes,
            output,
            partial,
            flags,
            bias,
            values,
            layout,
        ));
    }
    let compact = expected_symbol == CANDIDATE
        || expected_symbol == "nn_sm120_f32_n64_copyplan_m128n64_t256"
        || expected_symbol == "nn_sm120_f32_n64_copyplan_t256"
        || expected_symbol == "nn_sm120_f32_n64_sliced";
    let expected_abi = if compact {
        vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]
    } else {
        (0..4)
            .map(|i| (i * 8, 8))
            .chain((0..8).map(|i| (32 + i * 4, 4)))
            .collect()
    };
    let layout = abi_layout(params.func, if compact { 5 } else { 12 })?;
    if layout != expected_abi {
        return Err(format!("captured ABI differs: {layout:?}"));
    }
    let pointers = [
        captured(&params, 0)?,
        captured(&params, 1)?,
        captured(&params, 2)?,
        captured(&params, 3)?,
    ];
    let values: [u32; 8] = if compact {
        captured(&params, 4)?
    } else {
        [
            captured(&params, 4)?,
            captured(&params, 5)?,
            captured(&params, 6)?,
            captured(&params, 7)?,
            captured(&params, 8)?,
            captured(&params, 9)?,
            captured(&params, 10)?,
            captured(&params, 11)?,
        ]
    };
    let s = case.shape;
    let ops = case.operands(arm);
    let expected = LaunchProof {
        symbol: expected_symbol.into(),
        grid: (
            (s.m.div_ceil(
                if expected_symbol == "nn_sm120_f32_n64_copyplan_m128n64_t256" {
                    128
                } else {
                    64
                },
            ) * s.n.div_ceil(tile_n)) as u32,
            1,
            1,
        ),
        block: (threads, 1, 1),
        shared: 0,
        pointers: [ops.c.ptr, ops.x.ptr, ops.w.ptr, ops.bias_ptr.unwrap_or(0)],
        parameters: [
            1.0f32.to_bits(),
            0,
            s.m as u32,
            s.n as u32,
            s.k as u32,
            s.k as u32,
            s.n as u32,
            s.n as u32,
        ],
        abi: expected_abi,
        terminal_rejected: true,
    };
    let actual = LaunchProof {
        symbol,
        grid: (params.gridDimX, params.gridDimY, params.gridDimZ),
        block: (params.blockDimX, params.blockDimY, params.blockDimZ),
        shared: params.sharedMemBytes,
        pointers,
        parameters: values,
        abi: layout,
        terminal_rejected: true,
    };
    launch_contract(&actual, &expected)?;
    let abi = actual
        .abi
        .iter()
        .map(|(offset, size)| format!("[{offset},{size}]"))
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        "{{\"symbol\":{},\"grid\":[{},{},{}],\"block\":[{},{},{}],\"dynamic_shared\":{},\"pointers\":{:?},\"parameter_words\":{:?},\"driver_abi\":[{abi}],\"terminal_probe\":true}}",
        quoted(&actual.symbol),
        actual.grid.0,
        actual.grid.1,
        actual.grid.2,
        actual.block.0,
        actual.block.1,
        actual.block.2,
        actual.shared,
        actual.pointers,
        actual.parameters
    ))
}

struct VendorGraph {
    json: String,
    kernels: usize,
    bias: usize,
}
// Driver API deliberately handles foreign cuBLAS functions; do not use the
// Runtime cudaGraphKernelNodeGetParams API (known invalid-device-function trap).
fn vendor_graph(
    graph: sys::CUgraph,
    case: &Case,
    arm: usize,
    depth: usize,
) -> Result<VendorGraph, String> {
    if depth > 8 {
        return Err("vendor child-graph depth exceeds bound".into());
    }
    let nodes = graph_nodes(graph)?;
    let (mut kernels, mut bias) = (0, 0);
    let mut rendered = Vec::new();
    for node in &nodes {
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        cuda(
            unsafe { sys::cuGraphNodeGetType(*node, &mut kind) },
            "vendor graph node kind",
        )?;
        if kind == sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            let (symbol, params) = kernel_params(*node)?;
            if [
                params.gridDimX,
                params.gridDimY,
                params.gridDimZ,
                params.blockDimX,
                params.blockDimY,
                params.blockDimZ,
            ]
            .contains(&0)
            {
                return Err("invalid vendor physical launch geometry".into());
            }
            if symbol == "bias_broadcast" {
                let expected = [(0, 8), (8, 8), (16, 4), (20, 4)];
                if abi_layout(params.func, 4)? != expected {
                    return Err("vendor bias ABI mismatch".into());
                }
                let s = case.shape;
                if !case.has_bias
                    || captured::<u64>(&params, 0)? != case.outputs[arm].ptr()
                    || captured::<u64>(&params, 1)? != case.bias.ptr()
                    || captured::<i32>(&params, 2)? != s.m as i32
                    || captured::<i32>(&params, 3)? != s.n as i32
                    || (params.blockDimX, params.blockDimY, params.blockDimZ) != (256, 1, 1)
                    || (params.gridDimX, params.gridDimY, params.gridDimZ)
                        != ((s.m * s.n).div_ceil(256) as u32, 1, 1)
                    || params.sharedMemBytes != 0
                {
                    return Err("incomplete/wrong vendor bias broadcast".into());
                }
                bias += 1;
            } else {
                kernels += 1;
            }
            rendered.push(format!("{{\"kind\":\"kernel\",\"symbol\":{},\"grid\":[{},{},{}],\"block\":[{},{},{}],\"dynamic_shared\":{}}}",
                quoted(&symbol),params.gridDimX,params.gridDimY,params.gridDimZ,params.blockDimX,params.blockDimY,params.blockDimZ,params.sharedMemBytes));
        } else if kind == sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_GRAPH {
            let mut child = std::ptr::null_mut();
            cuda(
                unsafe { sys::cuGraphChildGraphNodeGetGraph(*node, &mut child) },
                "vendor child graph",
            )?;
            let description = vendor_graph(child, case, arm, depth + 1)?;
            kernels += description.kernels;
            bias += description.bias;
            rendered.push(format!(
                "{{\"kind\":\"child_graph\",\"graph\":{}}}",
                description.json
            ));
        } else {
            // Preserve the installed implementation's actual auxiliary nodes.
            rendered.push(format!("{{\"kind\":{}}}", quoted(&format!("{kind:?}"))));
        }
    }
    let mut count = 0;
    cuda(
        unsafe {
            sys::cuGraphGetEdges_v2(
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut count,
            )
        },
        "vendor edge count",
    )?;
    let mut from = vec![std::ptr::null_mut(); count];
    let mut to = from.clone();
    let mut data = Vec::with_capacity(count);
    data.resize_with(count, || unsafe { std::mem::zeroed() });
    if count > 0 {
        cuda(
            unsafe {
                sys::cuGraphGetEdges_v2(
                    graph,
                    from.as_mut_ptr(),
                    to.as_mut_ptr(),
                    data.as_mut_ptr(),
                    &mut count,
                )
            },
            "vendor edges",
        )?;
    }
    let edges = from
        .iter()
        .zip(&to)
        .map(|(a, b)| {
            let a = nodes
                .iter()
                .position(|node| node == a)
                .ok_or("missing edge source")?;
            let b = nodes
                .iter()
                .position(|node| node == b)
                .ok_or("missing edge target")?;
            Ok(format!("[{a},{b}]"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(VendorGraph {
        json: format!(
            "{{\"nodes\":[{}],\"edges\":[{}]}}",
            rendered.join(","),
            edges.join(",")
        ),
        kernels,
        bias,
    })
}

#[derive(Default)]
struct NumericResult {
    exceptions: usize,
    worst_ordered_exact_error: f64,
}
fn check_outputs(
    case: &Case,
    words: &[Vec<u32>],
    evidence: &mut Evidence,
    phase: &str,
) -> Result<NumericResult, String> {
    if words.len() != 6 {
        return Err("missing actual output arm".into());
    }
    let n = case.shape.m * case.shape.n;
    if words.iter().any(|arm| arm.len() != n) {
        return Err("numeric active extent mismatch".into());
    }
    for arm in [0, 1, 2] {
        raw_equal(&words[arm], &words[4])
            .map_err(|e| format!("{} versus old oracle: {e}", ARMS[arm]))?;
    }
    let mut result = NumericResult::default();
    for (i, &candidate_bits) in words[0].iter().enumerate() {
        let reference = f64::from(f32::from_bits(words[5][i]));
        let vendor = f64::from(f32::from_bits(words[3][i]));
        if !numeric_close(vendor, reference) {
            return Err(format!(
                "timed PEDANTIC/reference numeric mismatch index={i}"
            ));
        }
        let ordered = f64::from(f32::from_bits(candidate_bits));
        if !numeric_close(ordered, reference) {
            let proof = certificate(
                case.corpus,
                [candidate_bits, words[4][i], words[5][i], words[3][i]],
                case.proof(i),
            )
            .map_err(|e| format!("numeric mismatch index={i}: {e}"))?;
            let error = (ordered - proof.exact).abs();
            result.exceptions += 1;
            result.worst_ordered_exact_error = result.worst_ordered_exact_error.max(error);
            evidence.emit(format!("\"record\":\"ordered_accuracy_exception\",\"phase\":{},\"m\":{},\"k\":{},\"n\":{},\"bias\":{},\"index\":{i},\"candidate_bits\":{},\"old_bits\":{},\"cpu_ordered_bits\":{},\"exact_dyadic\":{},\"reference\":{reference},\"timed_vendor\":{vendor},\"ordered_exact_error\":{error},\"unchanged_tolerance\":true",
                quoted(phase),case.shape.m,case.shape.k,case.shape.n,case.has_bias,words[0][i],words[4][i],proof.ordered,proof.exact))?;
        }
    }
    Ok(result)
}

fn verify(
    case: &Case,
    ctx: &GpuCtx,
    expected: Option<&[Vec<u32>]>,
    evidence: &mut Evidence,
    phase: &str,
) -> Result<(Vec<Vec<u32>>, NumericResult), String> {
    let words = case.read_all(ctx)?;
    if let Some(expected) = expected {
        if expected.len() != 6 {
            return Err("repeat oracle arm count".into());
        }
        for arm in 0..6 {
            raw_equal(&words[arm], &expected[arm])
                .map_err(|e| format!("{phase}/{} repeat: {e}", ARMS[arm]))?;
        }
    }
    let numeric = check_outputs(case, &words, evidence, phase)?;
    Ok((words, numeric))
}

fn capture_workflows(ctx: &GpuCtx, case: &Case) -> Result<(Vec<CudaGraph>, String), String> {
    let mut graphs = Vec::new();
    let mut physical = Vec::new();
    for (arm, label) in ARMS.iter().enumerate() {
        let graph = unsafe { capture_into_graph(&ctx.stream, || case.launch(ctx, arm)) }?;
        let descriptor = if matches!(arm, 3 | 5) {
            let description = vendor_graph(graph.cu_graph(), case, arm, 0)?;
            if description.kernels == 0 || description.bias != usize::from(case.has_bias) {
                return Err("vendor graph omitted GEMM or complete bias workflow".into());
            }
            description.json
        } else {
            own_graph(&graph, case, arm)?
        };
        physical.push(format!("{}:{descriptor}", quoted(label)));
        graphs.push(graph);
    }
    Ok((graphs, format!("{{{}}}", physical.join(","))))
}

fn replay_checked(
    case: &Case,
    ctx: &GpuCtx,
    graphs: &[CudaGraph],
    expected: &[Vec<u32>],
    evidence: &mut Evidence,
    phase: &str,
) -> Result<NumericResult, String> {
    let mut last = NumericResult::default();
    for replay in 0..2 {
        for (arm, graph) in graphs.iter().enumerate() {
            case.outputs[arm].reset(ctx)?;
            graph
                .launch()
                .map_err(|e| format!("{phase} captured replay: {e:?}"))?;
        }
        let (_, numeric) = verify(
            case,
            ctx,
            Some(expected),
            evidence,
            &format!("{phase}/poisoned_graph_{replay}"),
        )?;
        last = numeric;
    }
    Ok(last)
}

const TELEMETRY_POLICY: &str = "observed_unlocked_no_clock_or_power_mutation";

#[derive(Clone, Debug, PartialEq)]
struct EnvironmentIdentity {
    uuid: String,
    name: String,
    driver: String,
    power_limit: f64,
}
fn environment_identity(value: &str) -> Result<EnvironmentIdentity, String> {
    let fields: Vec<_> = value.trim().split(',').map(str::trim).collect();
    if fields.len() != 7
        || !fields[0].starts_with("GPU-")
        || fields[0].len() <= 4
        || fields[1].is_empty()
        || fields[2].is_empty()
        || !fields[6].starts_with('P')
        || fields[6].len() <= 1
        || !fields[6][1..].bytes().all(|b| b.is_ascii_digit())
    {
        return Err(format!("malformed observed SM120 telemetry: {value:?}"));
    }
    let positive = |index: usize| -> Result<f64, String> {
        let number = fields[index]
            .parse::<f64>()
            .map_err(|e| format!("SM120 telemetry field {index}: {e}"))?;
        if !number.is_finite() || number <= 0.0 {
            return Err(format!(
                "nonpositive/nonfinite SM120 telemetry field {index}"
            ));
        }
        Ok(number)
    };
    positive(3)?; // Observed SM clock; deliberately not pinned to Ada's value.
    let power_limit = positive(4)?;
    positive(5)?; // Temperature is recorded, not frozen.
    Ok(EnvironmentIdentity {
        uuid: fields[0].into(),
        name: fields[1].into(),
        driver: fields[2].into(),
        power_limit,
    })
}
fn environment_unchanged(initial: &EnvironmentIdentity, value: &str) -> Result<(), String> {
    let observed = environment_identity(value)?;
    if &observed != initial {
        return Err(format!(
            "SM120 UUID/name/driver/power-limit drift: initial={initial:?}, observed={observed:?}"
        ));
    }
    Ok(())
}
fn telemetry(label: &str) -> Result<String, String> {
    super::sm120_exact_environment_preflight(label)?;
    let snapshot = std::process::Command::new("nvidia-smi")
        .args([
            "--id=0",
            "--query-gpu=uuid,name,driver_version,clocks.sm,power.limit,temperature.gpu,pstate",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !snapshot.status.success() {
        return Err("SM120 exact admission nvidia-smi telemetry failed".into());
    }
    let value = String::from_utf8(snapshot.stdout).map_err(|e| e.to_string())?;
    environment_identity(&value)?;
    Ok(value.trim().to_owned())
}

#[test]
fn exact_n64_admission_cpu_observed_telemetry_freezes_board_not_clock() {
    let first = "GPU-test, NVIDIA GeForce RTX 5090, 595.58.03, 2500, 575.0, 60, P0";
    let identity = environment_identity(first).unwrap();
    environment_unchanged(
        &identity,
        "GPU-test, NVIDIA GeForce RTX 5090, 595.58.03, 2400, 575.00, 70, P2",
    )
    .unwrap();
    for changed in [
        first.replace("GPU-test", "GPU-other"),
        first.replace("RTX 5090", "RTX 5080"),
        first.replace("595.58.03", "595.58.04"),
        first.replace("575.0", "570.0"),
    ] {
        assert!(environment_unchanged(&identity, &changed).is_err());
    }
    for malformed in [
        first.replace("2500", "0"),
        first.replace("2500", "NaN"),
        first.replace("575.0", "inf"),
        first.replace("575.0", "-1"),
        first.replace("60", "NaN"),
        first.replace("P0", "unknown"),
        first.replace("GPU-test", "GPU-"),
        first.to_owned() + ", extra",
    ] {
        assert!(environment_identity(&malformed).is_err());
    }
}

struct Protocol {
    diagnostic_fast: bool,
    cells: Vec<usize>,
    biases: Vec<usize>,
    paths: Vec<usize>,
    windows: usize,
    auto_phase: Option<String>,
}

fn jobs(protocol: &Protocol) -> Vec<(usize, usize, Corpus, bool)> {
    let mut jobs = Vec::new();
    for index in 0..CELLS.len() {
        if protocol.diagnostic_fast && index != 1 {
            continue;
        }
        for &bias in &protocol.biases {
            for corpus in [Corpus::Representable, Corpus::Nonrepresentable] {
                jobs.push((index, bias, corpus, false));
            }
        }
    }
    for &index in &protocol.cells {
        for &bias in &protocol.biases {
            jobs.push((index, bias, Corpus::Representable, true));
        }
    }
    jobs
}

#[test]
fn exact_n64_admission_cpu_all_finite_controls_precede_any_timing() {
    let protocol = Protocol {
        diagnostic_fast: false,
        cells: vec![1],
        biases: vec![0],
        paths: vec![0, 1],
        windows: 21,
        auto_phase: None,
    };
    let jobs = jobs(&protocol);
    assert_eq!(jobs.iter().position(|job| job.3), Some(10));
    assert_eq!(jobs.len(), 11);
    assert_eq!(jobs[10], (1, 0, Corpus::Representable, true));
    for index in 0..5 {
        assert_eq!(jobs[index * 2], (index, 0, Corpus::Representable, false));
        assert_eq!(
            jobs[index * 2 + 1],
            (index, 0, Corpus::Nonrepresentable, false)
        );
    }
}

impl Protocol {
    fn from_env() -> Result<Self, String> {
        let auto_phase = env("MAMBA_FIXED_SM120_EXACT_N64_AUTO")?;
        if let Some(value) = auto_phase.as_deref() {
            expected_auto(value, InferenceShape { m: 1, k: 1, n: 1 }, false, true)?;
        }
        Ok(Self {
            diagnostic_fast: false,
            cells: selection(
                env("MAMBA_FIXED_SM120_EXACT_N64_CELLS")?.as_deref(),
                &CELLS.map(|cell| cell.0),
            )?,
            biases: selection(
                env("MAMBA_FIXED_SM120_EXACT_N64_BIAS")?.as_deref(),
                &["0", "1"],
            )?,
            paths: selection(env("MAMBA_FIXED_SM120_EXACT_N64_PATHS")?.as_deref(), &PATHS)?,
            windows: window_count(env("MAMBA_FIXED_SM120_EXACT_N64_WINDOWS")?.as_deref())?,
            auto_phase,
        })
    }
    fn expected_pairs(&self) -> Vec<String> {
        let mut keys = Vec::new();
        for &cell in &self.cells {
            for &bias in &self.biases {
                for &path in &self.paths {
                    for order in ORDERS {
                        for control in [1, 2, 3] {
                            keys.push(format!(
                                "{}/{}/{}/{}/{}",
                                CELLS[cell].0, bias, PATHS[path], order, ARMS[control]
                            ));
                        }
                    }
                }
            }
        }
        keys
    }
}

fn diagnostic_fast_requested(
    value: Option<&str>,
    candidate: Candidate,
    protocol: &Protocol,
    poison_red: bool,
) -> Result<bool, String> {
    match value {
        None => Ok(false),
        Some("1") if candidate == Candidate::Sliced
            && protocol.cells == [1] && protocol.biases == [0]
            && protocol.windows == 21 && protocol.paths.len() == 2
            && protocol.paths.contains(&0) && protocol.paths.contains(&1)
            && !poison_red => Ok(true),
        _ => Err("MAMBA_FIXED_SM120_EXACT_N64_DIAGNOSTIC_FAST requires exactly 1, candidate=sliced, cell=hot_b, bias=0, windows=21, both eager+graph paths, and no POISON_RED".into()),
    }
}

#[cfg(test)]
fn diagnostic_protocol() -> Protocol {
    Protocol {
        diagnostic_fast: false,
        cells: vec![1],
        biases: vec![0],
        paths: vec![0, 1],
        windows: 21,
        auto_phase: Some("promoted".into()),
    }
}

#[test]
fn exact_n64_diagnostic_cpu_parser_requires_explicit_b0_sliced_screen() {
    let good = diagnostic_protocol();
    assert!(!diagnostic_fast_requested(None, Candidate::CopyPlan, &good, false).unwrap());
    assert!(diagnostic_fast_requested(Some("1"), Candidate::Sliced, &good, false).unwrap());
    for bad in ["", "0", "true", " 1", "1 "] {
        assert!(diagnostic_fast_requested(Some(bad), Candidate::Sliced, &good, false).is_err());
    }
    assert!(diagnostic_fast_requested(Some("1"), Candidate::CopyPlan, &good, false).is_err());
    assert!(diagnostic_fast_requested(Some("1"), Candidate::Sliced, &good, true).is_err());
    for field in 0..7 {
        let mut bad = diagnostic_protocol();
        match field {
            0 => bad.cells = vec![4],
            1 => bad.cells = vec![1, 4],
            2 => bad.biases = vec![1],
            3 => bad.biases = vec![0, 1],
            4 => bad.windows = 101,
            5 => bad.paths = vec![0],
            _ => bad.paths = vec![1],
        }
        assert!(
            diagnostic_fast_requested(Some("1"), Candidate::Sliced, &bad, false).is_err(),
            "restricted field {field}"
        );
    }
}

#[test]
fn exact_n64_diagnostic_cpu_keeps_both_corpora_and_all_twelve_pairs() {
    let mut protocol = diagnostic_protocol();
    protocol.diagnostic_fast = true;
    assert_eq!(
        jobs(&protocol),
        vec![
            (1, 0, Corpus::Representable, false),
            (1, 0, Corpus::Nonrepresentable, false),
            (1, 0, Corpus::Representable, true)
        ]
    );
    let expected: Vec<String> = ["eager", "graph"]
        .into_iter()
        .flat_map(|path| {
            ["ABBA", "BAAB"].into_iter().flat_map(move |order| {
                ["current_auto", "explicit_legacy", "pedantic"]
                    .into_iter()
                    .map(move |control| format!("hot_b/0/{path}/{order}/{control}"))
            })
        })
        .collect();
    assert_eq!(protocol.expected_pairs(), expected);
}

#[test]
fn exact_n64_diagnostic_cpu_never_admits_even_if_windows_are_tampered() {
    let mut protocol = diagnostic_protocol();
    protocol.diagnostic_fast = true;
    protocol.windows = 101;
    for candidate in [Candidate::CopyPlan, Candidate::Sliced] {
        assert!(!candidate_own_admission(
            candidate,
            InferenceTile::F32Sm120N64CopyPlan,
            "hot_b",
            false,
            &protocol,
            &[0.5; 4],
            &[0.5; 4]
        ));
    }
}

fn input_manifest(case: &Case, ctx: &GpuCtx) -> Result<String, String> {
    case.inputs(ctx)?;
    let a = case.a.read(ctx)?;
    let b = case.b.read(ctx)?;
    let bias = case.bias.read(ctx)?;
    Ok(format!(
        "{{\"a_sha256\":{},\"b_sha256\":{},\"bias_sha256\":{},\"a_elements\":{},\"b_elements\":{},\"bias_elements\":{},\"shared_active_pointers_all_six_arms\":true,\"prefix_suffix_guard_words\":32,\"guarded_allocations\":9,\"guards_checked_even_when_bias_absent\":true}}",
        quoted(&words_digest(&a)),
        quoted(&words_digest(&b)),
        quoted(&words_digest(&bias)),
        a.len(),
        b.len(),
        bias.len()
    ))
}

fn identity_contract(
    cc: (u32, u32),
    sm_count: u32,
    nvrtc: (i32, i32),
    known_library: bool,
    target: &str,
) -> Result<(), String> {
    if cc != (12, 0)
        || sm_count != 170
        || nvrtc != (13, 2)
        || !known_library
        || target != "compute_120"
    {
        return Err(format!(
            "unqualified SM120 identity: cc={cc:?}, SMs={sm_count}, NVRTC={nvrtc:?}, known={known_library}, target={target}"
        ));
    }
    Ok(())
}
fn profile_request(enabled: Option<&str>, cc: Option<&str>) -> Result<(), String> {
    if enabled != Some("1") || cc != Some("12.0") {
        return Err(
            "set MAMBA_FIXED_SM120_EXACT_N64_ENABLE=1 and MAMBA_FIXED_SM120_EXACT_N64_CC=12.0"
                .into(),
        );
    }
    Ok(())
}

#[test]
fn exact_n64_admission_cpu_profile_identity_is_exact_and_explicit() {
    profile_request(Some("1"), Some("12.0")).unwrap();
    for (enabled, cc) in [
        (None, Some("12.0")),
        (Some("0"), Some("12.0")),
        (Some("1"), None),
        (Some("1"), Some("8.9")),
        (Some("1"), Some("12.1")),
        (Some("1"), Some("12.0 ")),
    ] {
        assert!(profile_request(enabled, cc).is_err());
    }
    identity_contract((12, 0), 170, (13, 2), true, "compute_120").unwrap();
    for (cc, sms, nvrtc, known, target) in [
        ((12, 1), 170, (13, 2), true, "compute_120"),
        ((8, 9), 170, (13, 2), true, "compute_120"),
        ((12, 0), 169, (13, 2), true, "compute_120"),
        ((12, 0), 171, (13, 2), true, "compute_120"),
        ((12, 0), 170, (12, 8), true, "compute_120"),
        ((12, 0), 170, (13, 0), true, "compute_120"),
        ((12, 0), 170, (13, 2), false, "compute_120"),
        ((12, 0), 170, (13, 2), true, "sm_120"),
    ] {
        assert!(identity_contract(cc, sms, nvrtc, known, target).is_err());
    }
}

fn identity(ctx: &GpuCtx, device: &GpuDevice) -> Result<String, String> {
    let compiler = ctx.kernels.compiler_identity();
    let artifacts = ctx.kernels.artifact_set_identity();
    let driver = device.identity().driver;
    identity_contract(
        device.compute_capability,
        device.multiprocessor_count(),
        compiler.nvrtc_version,
        compiler.nvrtc_library_known,
        compiler.target.as_str(),
    )?;
    if driver.build_sources == 0 {
        return Err(format!(
            "unqualified exact-N64 production identity: {compiler:?}, device={:?}",
            device.identity()
        ));
    }
    let mut version = 0;
    let status =
        unsafe { cudarc::cublas::sys::cublasGetVersion_v2(*ctx.blas.handle(), &mut version) };
    if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || version <= 0 {
        return Err("installed cuBLAS version query failed".into());
    }
    Ok(format!(
        concat!(
            "{{\"cc\":[12,0],\"sm_count\":170,\"nvrtc\":[13,2],\"nvrtc_library_known\":true,\"nvrtc_library_domain\":{},",
            "\"compiler_target\":\"compute_120\",\"state_capacity\":{},\"source_sha256\":{},\"invocation_sha256\":{},\"headers_sha256\":{},",
            "\"fixed_compile_key\":{},\"fixed_artifact\":{},\"aggregate_artifact\":{},\"driver_api\":{},\"driver_build_sources\":{},\"driver_build_digest\":{},",
            "\"composer_revision\":{},\"compiler_revision\":{},\"numeric_abi_revision\":{},\"schedule_revision\":{},\"tuning_revision\":{},\"cublas_version\":{version}}}"
        ),
        quoted(&digest_hex(&compiler.nvrtc_library_domain)),
        ctx.kernels.state_cap,
        quoted(&digest_hex(&compiler.source_digest)),
        quoted(&digest_hex(&compiler.invocation_digest)),
        quoted(&digest_hex(&compiler.header_manifest_digest)),
        quoted(&digest_hex(&artifacts.fixed.compile_key)),
        quoted(&digest_hex(&artifacts.fixed.artifact_digest)),
        quoted(&digest_hex(&artifacts.ordered_digest)),
        driver.api_version,
        driver.build_sources,
        quoted(&digest_hex(&driver.build_digest)),
        compiler.composer_revision,
        compiler.compiler_revision,
        compiler.numeric_abi_revision,
        compiler.schedule_revision,
        super::TUNING_TABLE_REVISION,
        version = version
    ))
}

fn single_term_preflight(
    ctx: &GpuCtx,
    evidence: &mut Evidence,
    poison_red: bool,
    candidate: Candidate,
) -> Result<(), String> {
    for has_bias in [false, true] {
        let mut case = Case::new(
            ctx,
            InferenceShape { m: 65, k: 1, n: 65 },
            Corpus::Representable,
            has_bias,
            candidate,
        )?;
        let a_len = case.a.active;
        let b_len = case.b.active;
        let bias_len = case.bias.active;
        case.a.initial[GUARD..GUARD + a_len].fill(2.0);
        case.b.initial[GUARD..GUARD + b_len].fill(3.0);
        case.bias.initial[GUARD..GUARD + bias_len].fill(-0.5);
        case.a.reset(ctx)?;
        case.b.reset(ctx)?;
        case.bias.reset(ctx)?;
        for _ in 0..128 {
            for arm in 0..6 {
                case.launch(ctx, arm)?;
            }
        }
        if case.auto.get() != Some(InferenceTile::Legacy) {
            return Err("small single-term AUTO must remain Legacy".into());
        }
        let expected = vec![
            if has_bias {
                5.5f32.to_bits()
            } else {
                6.0f32.to_bits()
            };
            65 * 65
        ];
        let words = case.read_all(ctx)?;
        for arm in &words {
            single_term_output_gate(arm, expected[0], false)?;
        }
        let (graphs, physical) = capture_workflows(ctx, &case)?;
        if poison_red {
            // Warm/capture/inspect the real candidate first. Its next launch is
            // deliberately absent after this reset; no graph is replayed here.
            case.outputs[0].reset(ctx)?;
            let poisoned = case.read_all(ctx)?;
            for arm in 1..6 {
                raw_equal(&poisoned[arm], &words[arm])?;
            }
            let reason = single_term_output_gate(&poisoned[0], expected[0], true)
                .err()
                .ok_or("missing-launch switch unexpectedly passed")?;
            if !reason.starts_with(MISSING_LAUNCH_RED) {
                return Err(reason);
            }
            evidence.emit(format!("\"record\":\"missing_launch_red\",\"marker\":{},\"reason\":{},\"physical\":{},\"all_other_controls_unchanged\":true,\"all_guards\":true,\"pair_records\":0,\"completion_emitted\":false",quoted(MISSING_LAUNCH_RED),quoted(&reason),physical))?;
            return Err(reason);
        }
        replay_checked(&case, ctx, &graphs, &words, evidence, "single_term")?;
        evidence.emit(format!("\"record\":\"single_term_orientation_gate\",\"m\":65,\"k\":1,\"n\":65,\"bias\":{has_bias},\"expected_bits\":{},\"all_six_outputs_raw_equal\":true,\"physical\":{physical}",expected[0]))?;
    }
    Ok(())
}

fn run_case(
    ctx: &GpuCtx,
    case: &Case,
    label: &str,
    protocol: &Protocol,
    evidence: &mut Evidence,
    initial_resources: [[i32; 6]; 4],
    timed: bool,
) -> Result<usize, String> {
    let s = case.shape;
    if case.candidate != evidence.candidate || protocol.diagnostic_fast != evidence.diagnostic_fast
    {
        return Err("case/evidence candidate or diagnostic-mode drift".into());
    }
    for arm in [5, 4, 0, 1, 2, 3] {
        case.outputs[arm].reset(ctx)?;
        case.launch(ctx, arm)?;
    }
    auto_phase(
        protocol.auto_phase.as_deref(),
        case.shape,
        case.has_bias,
        case.auto.get().ok_or("AUTO was not launched")?,
        ctx.kernels
            .fixed_sm120_fma_postbias
            .as_ref()
            .is_some_and(|kernels| kernels.m128n64_t256.is_some()),
    )?;
    let (expected, pre_numeric) = verify(case, ctx, None, evidence, "eager_first")?;
    for arm in 0..6 {
        case.outputs[arm].reset(ctx)?;
        case.launch(ctx, arm)?;
    }
    verify(case, ctx, Some(&expected), evidence, "eager_repeat")?;
    // Every complete arm is eagerly warmed before the graph being measured exists.
    for _ in 0..128 {
        for arm in 0..6 {
            case.launch(ctx, arm)?;
        }
    }
    ctx.stream
        .synchronize()
        .map_err(|e| format!("capture warmup synchronize: {e:?}"))?;
    let (graphs, physical) = capture_workflows(ctx, case)?;
    replay_checked(case, ctx, &graphs, &expected, evidence, "pre_timing")?;
    let inputs = input_manifest(case, ctx)?;
    let output_digests = expected
        .iter()
        .map(|words| quoted(&words_digest(words)))
        .collect::<Vec<_>>()
        .join(",");
    let selected = case.auto.get().ok_or("AUTO was not launched")?;
    evidence.emit(format!(concat!("\"record\":\"correctness_control\",\"cell\":{},\"m\":{},\"k\":{},\"n\":{},\"bias\":{},\"corpus\":{},",
        "\"auto_tile\":{},\"physical\":{},\"common_inputs\":{},\"output_sha256_by_arm\":[{}],\"arm_order\":[\"candidate\",\"current_auto\",\"explicit_legacy\",\"pedantic\",\"old_oracle\",\"reference\"],",
        "\"raw_bits\":true,\"all_guards\":true,\"eager_repeat\":true,\"poisoned_graph_replays\":2,\"ordered_numeric_exceptions\":{},\"worst_ordered_exact_error\":{}"),
        quoted(label),s.m,s.k,s.n,case.has_bias,quoted(&format!("{:?}",case.corpus)),quoted(&format!("{selected:?}")),physical,inputs,output_digests,
        pre_numeric.exceptions,pre_numeric.worst_ordered_exact_error))?;
    if resource_snapshot(ctx, case.candidate)? != initial_resources {
        return Err(
            "candidate/incumbent function attributes drifted during correctness gates".into(),
        );
    }
    if !timed {
        return Ok(0);
    }
    let mut records = 0;
    let mut legacy_p95 = Vec::new();
    let mut auto_p95 = Vec::new();
    let mut vendor_p95 = Vec::new();
    for &path in &protocol.paths {
        for order in ORDERS {
            for control in [1usize, 2, 3] {
                let launch = |arm: usize| -> Result<(), String> {
                    if path == 0 {
                        case.launch(ctx, arm)
                    } else {
                        graphs[arm]
                            .launch()
                            .map_err(|e| format!("timed graph replay: {e:?}"))
                    }
                };
                // Per-pair warmup is eager, outside events, even for the graph path.
                for _ in 0..128 {
                    case.launch(ctx, 0)?;
                    case.launch(ctx, control)?;
                }
                for arm in [0, control] {
                    case.outputs[arm].reset(ctx)?;
                    launch(arm)?;
                }
                verify(case, ctx, Some(&expected), evidence, "pair_pre_timing")?;
                let preflight = evidence.telemetry("SM120 exact N64 pair preflight")?;
                let mut raw = Vec::with_capacity(protocol.windows);
                let mut ratios = Vec::with_capacity(protocol.windows);
                for _ in 0..protocol.windows {
                    let mut positions = [0.0; 4];
                    for (position, slot) in slots(order)?.into_iter().enumerate() {
                        let arm = if slot == 0 { 0 } else { control };
                        positions[position] = sm120_exact_event_window_us(ctx, 20, || {
                            launch(arm).expect("exact-N64 complete timed workflow")
                        });
                    }
                    ratios.push(paired_ratio(order, positions)?);
                    raw.push(positions);
                }
                ctx.stream
                    .synchronize()
                    .map_err(|e| format!("post timing sync: {e:?}"))?;
                let (_, post_numeric) =
                    verify(case, ctx, Some(&expected), evidence, "pair_post_timing")?;
                // Re-poison/replay after timing as well as checking the timed outputs.
                replay_checked(case, ctx, &graphs, &expected, evidence, "pair_post_timing")?;
                if resource_snapshot(ctx, case.candidate)? != initial_resources {
                    return Err("incumbent/candidate attributes changed during timing".into());
                }
                let postflight = evidence.telemetry("SM120 exact N64 pair postflight")?;
                let mut sorted = ratios.clone();
                sorted.sort_by(f64::total_cmp);
                let p50 = percentile(&sorted, 0.5);
                let p95 = percentile(&sorted, 0.95);
                match control {
                    1 => auto_p95.push(p95),
                    2 => legacy_p95.push(p95),
                    3 => vendor_p95.push(p95),
                    _ => unreachable!(),
                }
                evidence.pair(format!("{label}/{}/{}/{order}/{}",usize::from(case.has_bias),PATHS[path],ARMS[control]),format!(concat!("\"record\":\"pair\",\"cell\":{},\"m\":{},\"k\":{},\"n\":{},\"bias\":{},\"corpus\":\"signed_representable\",",
                    "\"control\":{},\"actual_auto_tile\":{},\"path\":{},\"order\":{},\"windows\":{},\"eager_warmups_per_arm\":128,",
                    "\"operations_per_event_window\":20,\"captured_operations_per_graph\":1,\"graph_replays_per_event_window\":{},\"raw_position_arm_indices\":{:?},",
                    "\"raw_position_us_per_operation\":{:?},\"raw_window_ratios\":{:?},\"paired_ratio_p50\":{},\"paired_ratio_p95\":{},",
                    "\"vendor_compute\":\"CUBLAS_COMPUTE_32F_PEDANTIC\",\"vendor_algorithm\":\"CUBLAS_GEMM_DEFAULT\",\"vendor_bias_every_call\":{},\"vendor_beta\":{},",
                    "\"reference_separate_output_untimed\":true,\"numeric_tolerance\":\"0.0002*(1+abs(reference))\",\"pre_numeric_exceptions\":{},\"post_numeric_exceptions\":{},\"worst_ordered_exact_error\":{},",
                    "\"physical\":{},\"resources_candidate_legacy_n128_oracle\":{:?},\"resource_field_order\":[\"local_bytes\",\"registers\",\"static_shared\",\"max_threads\",\"active_blocks\",\"preferred_carveout\"],",
                    "\"incumbent_attributes_unchanged\":true,\"common_inputs\":{},\"all_raw_repeat_bits_pre_post\":true,\"all_guards_pre_post\":true,\"preflight\":{},\"postflight\":{},\"screen_only\":{}"),
                    quoted(label),s.m,s.k,s.n,case.has_bias,quoted(ARMS[control]),quoted(&format!("{selected:?}")),quoted(PATHS[path]),quoted(order),protocol.windows,
                    if path==1 {20} else {0},slots(order)?,raw,ratios,p50,p95,case.has_bias,usize::from(case.has_bias),pre_numeric.exceptions,post_numeric.exceptions,
                    pre_numeric.worst_ordered_exact_error.max(post_numeric.worst_ordered_exact_error),physical,initial_resources,inputs,quoted(&preflight),quoted(&postflight),protocol.windows!=101))?;
                records += 1;
            }
        }
    }
    let promoted = !protocol.diagnostic_fast && protocol.auto_phase.as_deref() == Some("promoted");
    let auto_is_candidate = selected == case.candidate.tile();
    let own_win = candidate_own_admission(
        case.candidate,
        selected,
        label,
        case.has_bias,
        protocol,
        &legacy_p95,
        &auto_p95,
    );
    let vendor_win = !protocol.diagnostic_fast
        && own_win
        && vendor_p95.len() == 4
        && vendor_p95.iter().all(|r| r.is_finite() && *r <= 1.0);
    evidence.emit(format!("\"record\":\"cell_summary\",\"cell\":{},\"bias\":{},\"windows\":{},\"legacy_paired_p95\":{:?},\"auto_paired_p95\":{:?},\"pedantic_paired_p95\":{:?},\"auto_is_candidate_self_comparison\":{auto_is_candidate},\"eligible_own_win\":{own_win},\"vendor_win\":{vendor_win},\"host_auto_not_changed\":{},\"host_auto_promoted\":{promoted}",
        quoted(label),case.has_bias,protocol.windows,legacy_p95,auto_p95,vendor_p95,!promoted))?;
    Ok(records)
}

pub(super) fn run() {
    run_checked().expect("SM120 exact N64 production paired admission failed");
}
fn run_checked() -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err("paired admission requires --release".into());
    }
    profile_request(
        env("MAMBA_FIXED_SM120_EXACT_N64_ENABLE")?.as_deref(),
        env("MAMBA_FIXED_SM120_EXACT_N64_CC")?.as_deref(),
    )?;
    // These legacy filters are not meaningful in this single-candidate protocol.
    for unsupported in [
        "MAMBA_FIXED_SM120_EXACT_N64_TILES",
        "MAMBA_FIXED_SM120_EXACT_N64_ROWS",
        "MAMBA_FIXED_ADA_VENDOR",
        "MAMBA_FIXED_ADA_ROWS",
        "MAMBA_FIXED_ADA_CELLS",
        "MAMBA_FIXED_ADA_BIAS",
        "MAMBA_FIXED_ADA_WINDOWS",
        "MAMBA_FIXED_VENDOR_TILES",
        "MAMBA_FIXED_VENDOR_PATHS",
        "MAMBA_FIXED_VENDOR_EXACT_CC",
        "MAMBA_FIXED_EXACT_N64_AUTO",
        "MAMBA_FIXED_EXACT_N64_JSONL",
        "MAMBA_FIXED_EXACT_N64_POISON_RED",
    ] {
        if env(unsupported)?.is_some() {
            return Err(format!(
                "unset unrelated {unsupported} for exact-N64 admission"
            ));
        }
    }
    let mut protocol = Protocol::from_env()?;
    let candidate = Candidate::parse(env("MAMBA_FIXED_SM120_EXACT_N64_CANDIDATE")?.as_deref())?;
    let poison_red =
        poison_red_requested(env("MAMBA_FIXED_SM120_EXACT_N64_POISON_RED")?.as_deref())?;
    protocol.diagnostic_fast = diagnostic_fast_requested(
        env("MAMBA_FIXED_SM120_EXACT_N64_DIAGNOSTIC_FAST")?.as_deref(),
        candidate,
        &protocol,
        poison_red,
    )?;
    let preflight = telemetry("SM120 exact N64 run preflight")?;
    let device = GpuDevice::new(0)?;
    if device.compute_capability != (12, 0) || device.multiprocessor_count() != 170 {
        return Err("SM120 paired qualification requires exact CC12.0/170 SMs".into());
    }
    let ctx = GpuCtx::new(&device)?;
    super::configure_sm120_exact_custom(&ctx, F32TriadPolicy::ExactScalarFma);
    let identity = identity(&ctx, &device)?;
    let initial_resources = resource_snapshot(&ctx, candidate)?;
    let mut evidence = Evidence::new(&preflight, candidate, protocol.diagnostic_fast)?;
    let auto_phase = protocol.auto_phase.as_deref().unwrap_or("incumbent");
    evidence.emit(format!(concat!("\"record\":\"manifest\",\"identity\":{},\"preflight\":{},\"resources_candidate_legacy_n128_oracle\":{:?},\"selected_cells\":{:?},\"selected_biases\":{:?},\"selected_paths\":{:?},\"windows\":{},",
        "\"telemetry_policy\":{},\"auto_phase\":{},\"support_source_sha256\":{},\"wrapper_source_sha256\":{},\"warmups\":128,\"event_window_operations\":20,\"all_five_hot_cells_both_finite_corpora_are_untimed_controls\":{},\"single_term_preflight_performed\":{},",
        "\"timed_corpus\":\"signed_representable\",\"source\":\"actual_loaded_production_NVRTC\",\"standalone_evidence_reused\":false,\"accuracy_exceptions_are_not_all_element_tolerance_success\":true"),
        identity,quoted(&preflight),initial_resources,protocol.cells,protocol.biases,protocol.paths,protocol.windows,
        quoted(TELEMETRY_POLICY),quoted(auto_phase),quoted(&format!("{:x}",Sha256::digest(include_bytes!("fixed_sm120_exact_n64_admission.rs")))),
        quoted(&format!("{:x}",Sha256::digest(include_bytes!("../gemm_bi_inference_sm120_performance.rs")))),!protocol.diagnostic_fast,!protocol.diagnostic_fast))?;
    if !protocol.diagnostic_fast {
        single_term_preflight(&ctx, &mut evidence, poison_red, candidate)?;
    }
    let mut records = 0;
    // Full admission retains all five controls; diagnostic-only mode retains
    // both B0 corpora and all B0 paired checks without admitting any route.
    for (index, bias, corpus, timed) in jobs(&protocol) {
        let (label, shape) = CELLS[index];
        let case = Case::new(&ctx, shape, corpus, bias == 1, candidate)?;
        records += run_case(
            &ctx,
            &case,
            label,
            &protocol,
            &mut evidence,
            initial_resources,
            timed,
        )?;
    }
    if resource_snapshot(&ctx, candidate)? != initial_resources {
        return Err("final candidate/incumbent attribute drift".into());
    }
    let postflight = evidence.telemetry("SM120 exact N64 run postflight")?;
    evidence.emit(format!("\"record\":\"final_gates\",\"postflight\":{},\"incumbent_attributes_unchanged\":true,\"all_inputs_outputs_guards_replays_checked\":true",quoted(&postflight)))?;
    evidence.complete(records, &protocol.expected_pairs())
}
