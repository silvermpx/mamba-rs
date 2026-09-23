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
    return __float_as_uint(fmaf(__uint_as_float(bits & 0xff800000U), 1.0f / 2048.0f, __uint_as_float(bits)));
}"#;

const COPY_PLAN_ADVANCE_SEAM: &str = "__device__ __forceinline__ void tf32n96_advance_plan(";
const KERNEL_BODY_SEAM: &str = "__device__ __forceinline__ void tf32n96_kernel(";

/// The copy of a stage whose K slab lies inside the reduction, ahead of the
/// plan's advance.
pub const TRIAD_WHOLE_SLAB_COPY: &str = r#"// A stage whose K slab lies inside the reduction: each chunk's length is
// fixed for the CTA (16 bytes inside the matrix, 0 past its rows or
// columns, which zero-fills without reading memory), so nothing is clamped
// per tile.
__device__ __forceinline__ void tf32n96_stage_slice_whole(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes, unsigned b_stage_bytes, int issue) {
    gbf_tf32_copy_cg(
        plan.a_destination[issue] + a_stage_bytes, plan.a_source[issue],
        plan.a_row_valid[issue] ? 16 : 0);
    if (issue < 3) {
        gbf_tf32_copy_cg(
            plan.b_destination[issue] + b_stage_bytes, plan.b_source[issue],
            plan.b_column_bytes[issue]);
    }
}

"#;

/// The rotated tile body, ahead of the kernel body.
pub const TRIAD_ROTATED_TILE: &str = r#"// The ring stages one tile of the main loop reads, fills and reads next,
// and the K slab of the stage it fills.
struct Tf32n96TileStages {
    const float* a_read;
    const float* b_read;
    const float* a_next;
    const float* b_next;
    unsigned write_a;
    unsigned write_b;
    long long b_slab_rows;
    int fill_k;
    bool fills;
    bool has_following;
};

// One quarter of the next stage's copies; `Whole` is the copy form of a
// stage whose slab lies inside the reduction.
template <bool Whole>
__device__ __forceinline__ void tf32n96_fill_slice(
    const Tf32n96CopyPlan& plan, const Tf32n96TileStages& stages, int reduction, int issue) {
    if (Whole) {
        tf32n96_stage_slice_whole(plan, stages.write_a, stages.write_b, issue);
    } else if (stages.fills) {
        tf32n96_stage_slice(
            plan, stages.write_a, stages.write_b, stages.fill_k, reduction, issue);
    }
}

// One tile of the main loop. The stage wait and barrier sit in step 2,
// after its mma, and step 3 loads the next tile's step-0 fragments, so a
// tile starts on its mma rather than on the barrier. Every read of the
// tile's stage is issued before that barrier, and the copies into the slot
// read two tiles back start only after the previous tile's barrier.
template <bool Whole>
__device__ __forceinline__ void tf32n96_tile(
    Tf32n96CopyPlan& plan, const Tf32n96FragmentOffsets& offsets,
    const Tf32n96TileStages& stages, int reduction, Tf32n96Fragments (&fragments)[2],
    float (&acc)[4][3][4]) {
#pragma unroll
    for (int step = 0; step < 3; ++step) {
        tf32n96_fill_slice<Whole>(plan, stages, reduction, step);
        if (step == 2) {
            tf32n96_fill_slice<Whole>(plan, stages, reduction, 3);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        tf32n96_load_fragments(
            stages.a_read, stages.b_read, step + 1, offsets, fragments[(step + 1) & 1]);
        tf32n96_mma(fragments[step & 1], acc);
    }
    if (stages.fills) tf32n96_advance_plan(plan, stages.b_slab_rows);
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();
    if (stages.has_following) {
        tf32n96_load_fragments(stages.a_next, stages.b_next, 0, offsets, fragments[0]);
    }
    tf32n96_mma(fragments[1], acc);
}

"#;

pub const FIXED_MAIN_LOOP: &str = r#"    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        unsigned next = tile + 2;
        bool has_next = next < tile_count;
        int write_stage = read_stage == 0 ? 2 : read_stage - 1;
        unsigned write_a_bytes = (unsigned)write_stage * 128U * 32U * 4U;
        unsigned write_b_bytes = (unsigned)write_stage * 32U * 96U * 4U;
        const float* a_read = a_stages + read_stage * 128 * 32;
        const float* b_read = b_stages + read_stage * 32 * 96;
        Tf32n96Fragments fragments[2];
        tf32n96_load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
                tf32n96_stage_slice(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 32U), params.k, issue);
            }
            if (issue < 3) {
                tf32n96_load_fragments(
                    a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
            }
            tf32n96_mma(fragments[issue & 1], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (has_next) tf32n96_advance_plan(plan, b_slab_rows);
        if (++read_stage == 3) read_stage = 0;
    }"#;

pub const TRIAD_ROTATED_MAIN_LOOP: &str = r#"    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();
    Tf32n96Fragments fragments[2];
    tf32n96_load_fragments(a_stages, b_stages, 0, offsets, fragments[0]);
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        unsigned fill = tile + 2;
        int write_stage = read_stage == 0 ? 2 : read_stage - 1;
        int next_stage = read_stage == 2 ? 0 : read_stage + 1;
        Tf32n96TileStages stages;
        stages.a_read = a_stages + read_stage * 128 * 32;
        stages.b_read = b_stages + read_stage * 32 * 96;
        stages.a_next = a_stages + next_stage * 128 * 32;
        stages.b_next = b_stages + next_stage * 32 * 96;
        stages.write_a = (unsigned)write_stage * 128U * 32U * 4U;
        stages.write_b = (unsigned)write_stage * 32U * 96U * 4U;
        stages.b_slab_rows = b_slab_rows;
        stages.fill_k = (int)(fill * 32U);
        stages.fills = fill < tile_count;
        stages.has_following = tile + 1 < tile_count;
        if (stages.fills && stages.fill_k + 32 <= params.k) {
            tf32n96_tile<true>(plan, offsets, stages, params.k, fragments, acc);
        } else {
            tf32n96_tile<false>(plan, offsets, stages, params.k, fragments, acc);
        }
        read_stage = next_stage;
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
        COPY_PLAN_ADVANCE_SEAM,
        &format!("{TRIAD_WHOLE_SLAB_COPY}{COPY_PLAN_ADVANCE_SEAM}"),
        "whole-slab copy",
    )?;
    let source = replace_exactly_once(
        &source,
        KERNEL_BODY_SEAM,
        &format!("{TRIAD_ROTATED_TILE}{KERNEL_BODY_SEAM}"),
        "rotated tile",
    )?;
    let source = replace_exactly_once(
        &source,
        FIXED_MAIN_LOOP,
        TRIAD_ROTATED_MAIN_LOOP,
        "main loop",
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
    let source = replace_exactly_once(
        &source,
        &format!("{TRIAD_WHOLE_SLAB_COPY}{COPY_PLAN_ADVANCE_SEAM}"),
        COPY_PLAN_ADVANCE_SEAM,
        "restored whole-slab copy",
    )?;
    let source = replace_exactly_once(
        &source,
        &format!("{TRIAD_ROTATED_TILE}{KERNEL_BODY_SEAM}"),
        KERNEL_BODY_SEAM,
        "restored rotated tile",
    )?;
    let source = replace_exactly_once(
        &source,
        TRIAD_ROTATED_MAIN_LOOP,
        FIXED_MAIN_LOOP,
        "restored main loop",
    )?;
    replace_exactly_once(
        &source,
        TRIAD_NN_N96_SYMBOL,
        FIXED_N96_SYMBOL,
        "restored export symbol",
    )
}

pub fn triad_add_half_ulp(bits: u32) -> u32 {
    let value = f32::from_bits(bits);
    f32::from_bits(bits & 0xff80_0000)
        .mul_add(1.0 / 2048.0, value)
        .to_bits()
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

    const FIXED_SOURCE: &str = include_str!("../../kernels/gemm_bi_inference/sm80/tf32_rna_n96.cu");

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
            (0x0000_0000, 0x0000_0000),
            (0x8000_0000, 0x8000_0000),
            (0x007f_ffff, 0x007f_ffff),
            (0x3f80_0000, 0x3f80_1000),
            (0x3f80_1000, 0x3f80_2000),
            (0x7f7f_ffff, 0x7f80_0000),
            (0x7f80_0000, 0x7f80_0000),
            (0xff80_0000, 0xff80_0000),
        ] {
            assert_eq!(triad_add_half_ulp(bits), expected, "bits=0x{bits:08x}");
        }
        for bits in [0x7f80_0001, 0x7f80_1000, 0x7fff_ffff, 0xffff_ffff] {
            assert!(
                f32::from_bits(triad_add_half_ulp(bits)).is_nan(),
                "NaN operand 0x{bits:08x} must stay a NaN"
            );
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
