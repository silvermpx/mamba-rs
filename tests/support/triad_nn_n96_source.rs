pub const FIXED_N96_SYMBOL: &str = "nn_sm89_rna_tf32_m128n96_bk32_s3";
pub const TRIAD_NN_N96_SYMBOL: &str = "nn_triad_sm89_add_half_tf32_exp_m128n96_bk32_s3";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScreenTarget {
    pub cell: &'static str,
    pub shape: (usize, usize, usize),
}

pub const D768_IN_TARGET: ScreenTarget = ScreenTarget {
    cell: "d768_in_proj",
    shape: (2_048, 768, 3_072),
};

pub const PRISM_TARGET: ScreenTarget = ScreenTarget {
    cell: "prism_in_proj",
    shape: (4_621, 384, 1_928),
};

pub fn logical_alignment_guard_elements(
    alignment_bytes: usize,
    element_bytes: usize,
) -> Result<usize, String> {
    if alignment_bytes == 0 || element_bytes == 0 || !alignment_bytes.is_multiple_of(element_bytes)
    {
        return Err(format!(
            "invalid logical alignment {alignment_bytes}/{element_bytes}"
        ));
    }
    Ok(alignment_bytes / element_bytes)
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactBitScope {
    ForcedAddHalfFamily,
    TimedPublicAutoTarget,
}

pub fn exact_bits_match(
    scope: ExactBitScope,
    candidate: &[u32],
    current_wide: &[u32],
    actual_auto: &[u32],
) -> bool {
    candidate == current_wide
        && match scope {
            ExactBitScope::ForcedAddHalfFamily => true,
            ExactBitScope::TimedPublicAutoTarget => candidate == actual_auto,
        }
}

pub fn expected_nn_auto_grid(
    shape: (usize, usize, usize),
    tile: (usize, usize),
    zero_reduction: bool,
) -> Result<(u32, u32, u32), String> {
    if tile.0 == 0 || tile.1 == 0 {
        return Err("NN AUTO tile dimensions must be nonzero".into());
    }
    let blocks = if zero_reduction {
        shape
            .0
            .checked_mul(shape.2)
            .ok_or_else(|| "NN zero-reduction output extent overflows usize".to_owned())?
            .div_ceil(256)
    } else {
        shape
            .0
            .div_ceil(tile.0)
            .checked_mul(shape.2.div_ceil(tile.1))
            .ok_or_else(|| "NN tiled grid extent overflows usize".to_owned())?
    };
    Ok((
        u32::try_from(blocks).map_err(|_| "NN AUTO grid exceeds u32".to_owned())?,
        1,
        1,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXED_SOURCE: &str = include_str!("../../kernels/gemm_bi_inference/sm89/tf32_rna_n96.cu");

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
    fn focused_fallback_and_timed_target_have_distinct_exact_bit_scopes() {
        let forced = [1, 2, 3];
        let scalar_fallback = [1, 9, 3];
        assert!(exact_bits_match(
            ExactBitScope::ForcedAddHalfFamily,
            &forced,
            &forced,
            &scalar_fallback,
        ));
        assert!(!exact_bits_match(
            ExactBitScope::ForcedAddHalfFamily,
            &[1, 8, 3],
            &forced,
            &scalar_fallback,
        ));
        assert!(!exact_bits_match(
            ExactBitScope::TimedPublicAutoTarget,
            &forced,
            &forced,
            &scalar_fallback,
        ));
        assert!(exact_bits_match(
            ExactBitScope::TimedPublicAutoTarget,
            &forced,
            &forced,
            &forced,
        ));
    }

    #[test]
    fn zero_reduction_uses_linear_output_grid_without_relaxing_tiled_targets() {
        assert_eq!(
            expected_nn_auto_grid((129, 0, 100), (1, 1), true),
            Ok((51, 1, 1))
        );
        assert_eq!(
            expected_nn_auto_grid((129, 36, 100), (64, 32), false),
            Ok((12, 1, 1))
        );
        assert!(expected_nn_auto_grid((129, 0, 100), (0, 1), true).is_err());
    }

    #[test]
    fn missing_ada_nn_cells_and_aligned_guard_are_exact() {
        assert_eq!(D768_IN_TARGET.cell, "d768_in_proj");
        assert_eq!(D768_IN_TARGET.shape, (2_048, 768, 3_072));
        assert_eq!(PRISM_TARGET.cell, "prism_in_proj");
        assert_eq!(PRISM_TARGET.shape, (4_621, 384, 1_928));
        assert_eq!(logical_alignment_guard_elements(256, 4), Ok(64));
        assert!(logical_alignment_guard_elements(0, 4).is_err());
        assert!(logical_alignment_guard_elements(256, 0).is_err());
        assert!(logical_alignment_guard_elements(255, 4).is_err());
    }
}
