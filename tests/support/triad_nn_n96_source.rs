pub const FIXED_N96_SYMBOL: &str = "gemm_bi_nn_fixed_sm89_rna_tf32_v1_m128n96_bk32_s3";
pub const TRIAD_NN_N96_SYMBOL: &str = "gemm_bi_nn_triad_sm89_add_half_tf32_exp_m128n96_bk32_s3";

pub const FIXED_RNA_ROUND: &str = r#"__device__ __forceinline__ unsigned tf32n96_round(unsigned bits) {
    float value = __uint_as_float(bits);
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}"#;

pub const TRIAD_ADD_HALF_ROUND: &str = r#"__device__ __forceinline__ unsigned tf32n96_round(unsigned bits) {
    return bits + 0x1000U;
}"#;

fn replace_exactly_once(source: &str, old: &str, new: &str, label: &str) -> Result<String, String> {
    let count = source.matches(old).count();
    if count != 1 {
        return Err(format!(
            "Triad NN N96 {label} seam count changed: expected 1, found {count}"
        ));
    }
    Ok(source.replacen(old, new, 1))
}

pub fn compose_triad_nn_n96_source(fixed_source: &str) -> Result<String, String> {
    let source = replace_exactly_once(
        fixed_source,
        FIXED_RNA_ROUND,
        TRIAD_ADD_HALF_ROUND,
        "operand conversion",
    )?;
    let source = replace_exactly_once(
        &source,
        FIXED_N96_SYMBOL,
        TRIAD_NN_N96_SYMBOL,
        "export symbol",
    )?;
    if restore_fixed_n96_source(&source)? != fixed_source {
        return Err("Triad NN N96 transform does not restore the immutable Fixed source".into());
    }
    Ok(source)
}

pub fn restore_fixed_n96_source(candidate_source: &str) -> Result<String, String> {
    let source = replace_exactly_once(
        candidate_source,
        TRIAD_ADD_HALF_ROUND,
        FIXED_RNA_ROUND,
        "restored operand conversion",
    )?;
    replace_exactly_once(
        &source,
        TRIAD_NN_N96_SYMBOL,
        FIXED_N96_SYMBOL,
        "restored export symbol",
    )
}

pub const fn triad_add_half_ulp(bits: u32) -> u32 {
    bits.wrapping_add(0x1000)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BracketOrder {
    Abba,
    Baab,
}

impl BracketOrder {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Abba => "ABBA",
            Self::Baab => "BAAB",
        }
    }

    pub const fn candidate_slots(self) -> [bool; 4] {
        match self {
            Self::Abba => [false, true, true, false],
            Self::Baab => [true, false, false, true],
        }
    }
}

pub fn candidate_over_auto_ratio(
    order: BracketOrder,
    observations: [f64; 4],
) -> Result<f64, String> {
    if observations
        .iter()
        .any(|sample| !sample.is_finite() || *sample <= 0.0)
    {
        return Err(format!("invalid bracket observations: {observations:?}"));
    }
    let candidate_slots = order.candidate_slots();
    let mut candidate = 0.0;
    let mut auto = 0.0;
    for (index, sample) in observations.into_iter().enumerate() {
        if candidate_slots[index] {
            candidate += sample;
        } else {
            auto += sample;
        }
    }
    Ok(candidate / auto)
}

pub fn valid_finite_nonzero_f32_bits(words: &[u32]) -> bool {
    !words.is_empty()
        && words.iter().all(|word| f32::from_bits(*word).is_finite())
        && words.iter().any(|word| word & 0x7fff_ffff != 0)
}

pub fn compare_bits(left: &[u32], right: &[u32]) -> (usize, Option<usize>) {
    let first = left
        .iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())));
    let count = left
        .iter()
        .zip(right)
        .filter(|(left, right)| left != right)
        .count()
        + left.len().abs_diff(right.len());
    (count, first)
}

pub fn validate_public_auto_harness(source: &str) -> Result<(), String> {
    let public_auto = "TriadArm::ActualAuto => {\n                let auto_a";
    let public_entrypoint = "gpu_gemm_bi_forward_raw(";
    let typed_entrypoint = "TriadArm::ActualAuto => gpu_gemm_typed_forward_raw(";
    let graph_contract = r#"assert_single_tf32_graph(
            &auto_graph,
            TARGET.shape,
            CURRENT_WIDE_SYMBOL,"#;
    if source.matches(public_auto).count() != 1
        || source.matches(public_entrypoint).count() != 1
        || source.contains(typed_entrypoint)
        || source.matches(graph_contract).count() != 1
    {
        return Err("Triad NN N96 public AUTO entrypoint or graph contract changed".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXED_SOURCE: &str = include_str!("../../kernels/gemm_bi_fixed/tf32_rna_n96.cu");

    #[test]
    fn triad_nn_n96_transform_is_exact_and_reversible() {
        let transformed = compose_triad_nn_n96_source(FIXED_SOURCE).expect("transform Fixed N96");
        assert_eq!(transformed.matches(FIXED_N96_SYMBOL).count(), 0);
        assert_eq!(transformed.matches(TRIAD_NN_N96_SYMBOL).count(), 1);
        assert_eq!(transformed.matches(FIXED_RNA_ROUND).count(), 0);
        assert_eq!(transformed.matches(TRIAD_ADD_HALF_ROUND).count(), 1);
        assert_eq!(
            restore_fixed_n96_source(&transformed).unwrap(),
            FIXED_SOURCE
        );
    }

    #[test]
    fn triad_nn_n96_transform_rejects_missing_or_duplicate_seams() {
        assert!(compose_triad_nn_n96_source("").is_err());
        assert!(compose_triad_nn_n96_source(&format!("{FIXED_SOURCE}\n{FIXED_SOURCE}")).is_err());

        let transformed = compose_triad_nn_n96_source(FIXED_SOURCE).unwrap();
        assert!(restore_fixed_n96_source(&format!("{transformed}\n{transformed}")).is_err());
    }

    #[test]
    fn add_half_ulp_conversion_matches_current_wide_edge_bits() {
        for (bits, expected) in [
            (0x0000_0000, 0x0000_1000),
            (0x8000_0000, 0x8000_1000),
            (0x3f80_0000, 0x3f80_1000),
            (0x3f80_1000, 0x3f80_2000),
            (0x7f80_0000, 0x7f80_1000),
            (0x7f80_0001, 0x7f80_1001),
            (0x7fff_ffff, 0x8000_0fff),
            (0xffff_ffff, 0x0000_0fff),
        ] {
            assert_eq!(triad_add_half_ulp(bits), expected, "bits=0x{bits:08x}");
        }
    }

    #[test]
    fn bracket_order_and_candidate_over_auto_ratio_are_exact() {
        assert_eq!(
            BracketOrder::Abba.candidate_slots(),
            [false, true, true, false]
        );
        assert_eq!(
            BracketOrder::Baab.candidate_slots(),
            [true, false, false, true]
        );
        assert_eq!(
            candidate_over_auto_ratio(BracketOrder::Abba, [10.0, 4.0, 6.0, 10.0]).unwrap(),
            0.5
        );
        assert_eq!(
            candidate_over_auto_ratio(BracketOrder::Baab, [4.0, 10.0, 10.0, 6.0]).unwrap(),
            0.5
        );
        for invalid in [
            [0.0, 1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0],
            [f64::NAN, 1.0, 1.0, 1.0],
            [f64::INFINITY, 1.0, 1.0, 1.0],
        ] {
            assert!(candidate_over_auto_ratio(BracketOrder::Abba, invalid).is_err());
        }
    }

    #[test]
    fn fast_self_bits_must_be_finite_and_not_all_zero() {
        assert!(valid_finite_nonzero_f32_bits(&[
            0x0000_0000,
            0x8000_0000,
            0x3f80_0000,
        ]));
        assert!(!valid_finite_nonzero_f32_bits(&[]));
        assert!(!valid_finite_nonzero_f32_bits(&[0x0000_0000, 0x8000_0000,]));
        assert!(!valid_finite_nonzero_f32_bits(&[0x3f80_0000, 0x7f80_0000]));
        assert!(!valid_finite_nonzero_f32_bits(&[0x3f80_0000, 0x7fc0_1234]));
    }

    #[test]
    fn bit_difference_reports_first_word_and_total_count() {
        assert_eq!(compare_bits(&[1, 2, 3, 4], &[1, 9, 3, 8]), (2, Some(1)));
        assert_eq!(compare_bits(&[1, 2], &[1, 2]), (0, None));
        assert_eq!(compare_bits(&[1, 2, 3], &[1]), (2, Some(1)));
    }

    #[test]
    fn harness_uses_public_triad_auto_and_pins_its_graph_symbol() {
        let harness = include_str!("../gemm_bi_fixed_tf32_n96_discovery.rs");
        validate_public_auto_harness(harness).unwrap();
        let wrong = harness.replacen("gpu_gemm_bi_forward_raw(", "gpu_gemm_typed_forward_raw(", 1);
        assert!(validate_public_auto_harness(&wrong).is_err());
    }
}
