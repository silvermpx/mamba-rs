#[path = "triad_half_nt_m64n192_s3_source.rs"]
mod parent;

pub const SYMBOL: &str = "gemm_bi_nt_test_fixed_s3_m64n192_issue2211_f16";
pub const RETAINED_SYMBOL: &str = parent::SYMBOL;
pub const BLOCK_THREADS: u32 = parent::BLOCK_THREADS;
pub const DYNAMIC_SHARED_BYTES: usize = parent::DYNAMIC_SHARED_BYTES;
pub const MAX_REGISTERS: i32 = parent::MAX_REGISTERS;
pub const REQUIRED_OCCUPANCY: u32 = parent::REQUIRED_OCCUPANCY;
pub const EXPECTED_HMMA: usize = 64;
pub const EXPECTED_LDGSTS: usize = 18;

const CURRENT_ISSUE_SCHEDULE: &str = r#"            if (refill) sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 0);
            sm89_fixed_half_swizzle::nt_m64n192_load_fragments(a_read, b_read, 1, offsets, fragments[1]);
            sm89_fixed_half_swizzle::nt_m64n192_consume_fragments<T>(fragments[0], acc);

            if (refill) sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 1);
            sm89_fixed_half_swizzle::nt_m64n192_load_fragments(a_read, b_read, 2, offsets, fragments[0]);
            sm89_fixed_half_swizzle::nt_m64n192_consume_fragments<T>(fragments[1], acc);

            if (refill) {
                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 2);
                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 3);
            }
            sm89_fixed_half_swizzle::nt_m64n192_load_fragments(a_read, b_read, 3, offsets, fragments[1]);
            sm89_fixed_half_swizzle::nt_m64n192_consume_fragments<T>(fragments[0], acc);

            if (refill) {
                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 4);
                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 5);
                asm volatile("cp.async.commit_group;\n" ::);
            }"#;

const ISSUE_2211_SCHEDULE: &str = r#"            if (refill) {
                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 0);
                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 1);
            }
            sm89_fixed_half_swizzle::nt_m64n192_load_fragments(a_read, b_read, 1, offsets, fragments[1]);
            sm89_fixed_half_swizzle::nt_m64n192_consume_fragments<T>(fragments[0], acc);

            if (refill) {
                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 2);
                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 3);
            }
            sm89_fixed_half_swizzle::nt_m64n192_load_fragments(a_read, b_read, 2, offsets, fragments[0]);
            sm89_fixed_half_swizzle::nt_m64n192_consume_fragments<T>(fragments[1], acc);

            if (refill)
                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 4);
            sm89_fixed_half_swizzle::nt_m64n192_load_fragments(a_read, b_read, 3, offsets, fragments[1]);
            sm89_fixed_half_swizzle::nt_m64n192_consume_fragments<T>(fragments[0], acc);

            if (refill) {
                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 5);
                asm volatile("cp.async.commit_group;\n" ::);
            }"#;

pub fn retained_source(swizzle: &str, s3: &str) -> Result<String, String> {
    parent::candidate_source(swizzle, s3)
}

pub fn candidate_source(swizzle: &str, s3: &str) -> Result<String, String> {
    let mut source = retained_source(swizzle, s3)?;
    replace_exact(&mut source, CURRENT_ISSUE_SCHEDULE, ISSUE_2211_SCHEDULE)?;
    replace_exact(&mut source, RETAINED_SYMBOL, SYMBOL)?;
    if !has_issue_pattern_2211(&source) {
        return Err("M64N192 issue2211 transform did not produce the frozen schedule".into());
    }
    Ok(source)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    replace_exact(&mut source, ISSUE_2211_SCHEDULE, CURRENT_ISSUE_SCHEDULE)?;
    replace_exact(&mut source, SYMBOL, RETAINED_SYMBOL)?;
    Ok(source)
}

pub fn has_issue_pattern_2211(source: &str) -> bool {
    source.matches(ISSUE_2211_SCHEDULE).count() == 1
        && source.matches(CURRENT_ISSUE_SCHEDULE).count() == 0
}

pub fn all_strata_pass(strata: &[[f64; 2]]) -> bool {
    strata.len() == 4
        && strata.iter().all(|[p50, p95]| {
            p50.is_finite()
                && p95.is_finite()
                && *p50 > 0.0
                && *p95 > 0.0
                && *p50 < 0.99
                && *p95 < 0.99
        })
}

fn replace_exact(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    let count = source.matches(before).count();
    if count != 1 {
        return Err(format!(
            "M64N192 issue2211 anchor expected once, observed {count}: {before:?}"
        ));
    }
    *source = source.replacen(before, after, 1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SWIZZLE: &str = include_str!("../../kernels/gemm_bi_fixed/sm89_half_swizzle.cu");
    const S3: &str = include_str!("../../kernels/gemm_bi_fixed/sm89_half_s3.cu");

    #[test]
    fn transform_changes_only_the_export_and_physical_copy_issue_schedule() {
        let retained = retained_source(SWIZZLE, S3).unwrap();
        let candidate = candidate_source(SWIZZLE, S3).unwrap();
        assert!(has_issue_pattern_2211(&candidate));
        assert_eq!(candidate.matches(SYMBOL).count(), 1);
        assert_eq!(candidate.matches(RETAINED_SYMBOL).count(), 0);
        assert_eq!(restore_retained_source(&candidate).unwrap(), retained);
    }

    #[test]
    fn schedule_moves_full_slice_four_earlier_and_leaves_partial_slice_five_last() {
        let candidate = candidate_source(SWIZZLE, S3).unwrap();
        let schedule = candidate.find(ISSUE_2211_SCHEDULE).unwrap();
        let body = &candidate[schedule..schedule + ISSUE_2211_SCHEDULE.len()];
        for slice in 0..6 {
            assert_eq!(body.matches(&format!("next_k, N, {slice})")).count(), 1);
        }
        let load1 = body.find("b_read, 1, offsets").unwrap();
        let load2 = body.find("b_read, 2, offsets").unwrap();
        let load3 = body.find("b_read, 3, offsets").unwrap();
        assert!(body.find("next_k, N, 1)").unwrap() < load1);
        assert!(body.find("next_k, N, 3)").unwrap() < load2);
        assert!(body.find("next_k, N, 4)").unwrap() < load3);
        assert!(body.find("next_k, N, 5)").unwrap() > load3);
    }

    #[test]
    fn arithmetic_geometry_and_async_copy_count_are_unchanged() {
        let retained = retained_source(SWIZZLE, S3).unwrap();
        let candidate = candidate_source(SWIZZLE, S3).unwrap();
        for anchor in [
            "mma.sync.aligned.m16n8k16.row.col.f32.",
            "gbf_store_pair_rne",
            "cp.async.cg.shared.global",
            "static constexpr int kSharedBytes = 98304;",
            "__launch_bounds__(384, 1)",
        ] {
            assert_eq!(
                candidate.matches(anchor).count(),
                retained.matches(anchor).count()
            );
        }
    }

    #[test]
    fn malformed_or_already_transformed_input_fails_closed() {
        assert!(candidate_source("", S3).is_err());
        assert!(candidate_source(SWIZZLE, "").is_err());
        let candidate = candidate_source(SWIZZLE, S3).unwrap();
        let mut duplicate = candidate.clone();
        duplicate.push_str(ISSUE_2211_SCHEDULE);
        assert!(restore_retained_source(&duplicate).is_err());
    }
}
