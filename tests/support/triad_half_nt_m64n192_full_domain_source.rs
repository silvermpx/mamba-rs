#[path = "triad_half_nt_m64n192_s3_source.rs"]
mod parent;

pub const SYMBOL: &str = "gemm_bi_nt_test_fixed_s3_m64n192_full_domain_f16";
pub const RETAINED_SYMBOL: &str = parent::SYMBOL;
pub const TARGET: (usize, usize, usize) = (2_048, 1_536, 768);
pub const TARGET_GRID: u32 = 256;
pub const BLOCK_THREADS: u32 = parent::BLOCK_THREADS;
pub const DYNAMIC_SHARED_BYTES: usize = parent::DYNAMIC_SHARED_BYTES;
pub const MAX_REGISTERS: i32 = parent::MAX_REGISTERS;
pub const REQUIRED_OCCUPANCY: u32 = parent::REQUIRED_OCCUPANCY;
pub const EXPECTED_HMMA: usize = 64;
pub const EXPECTED_LDGSTS: usize = 18;

const EXACT_EXPORT: &str = r#"
extern "C" __global__ __launch_bounds__(384, 1)
void gemm_bi_nt_test_fixed_s3_m64n192_full_domain_f16(
    __half* __restrict__ C, const __half* __restrict__ A,
    const __half* __restrict__ B, float alpha,
    int M, int N, int K_out) {
    (void)M;
    (void)N;
    (void)K_out;
    sm89_test_half_nt_m64n192_s3::kernel<__half>(
        C, A, B, alpha, 2048, 768, 1536);
}
"#;

pub fn retained_source(swizzle: &str, s3: &str) -> Result<String, String> {
    parent::candidate_source(swizzle, s3)
}

pub fn candidate_source(swizzle: &str, s3: &str) -> Result<String, String> {
    let retained = retained_source(swizzle, s3)?;
    require_count(&retained, RETAINED_SYMBOL, 1, "retained export")?;
    require_count(&retained, SYMBOL, 0, "preexisting exact export")?;
    Ok(format!("{retained}\n{EXACT_EXPORT}"))
}

pub fn restore_retained_source(candidate: &str, swizzle: &str, s3: &str) -> Result<String, String> {
    let retained = retained_source(swizzle, s3)?;
    let expected = format!("{retained}\n{EXACT_EXPORT}");
    if candidate != expected {
        return Err("M64N192 full-domain candidate differs from fail-closed transform".into());
    }
    Ok(retained)
}

pub const fn is_exact_target(m: usize, k_out: usize, reduction: usize) -> bool {
    m == TARGET.0 && k_out == TARGET.1 && reduction == TARGET.2
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

fn require_count(source: &str, needle: &str, expected: usize, label: &str) -> Result<(), String> {
    let actual = source.matches(needle).count();
    if actual != expected {
        return Err(format!(
            "M64N192 full-domain {label} expected {expected}, observed {actual}: {needle:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SWIZZLE: &str = include_str!("../../kernels/gemm_bi_fixed/sm89_half_swizzle.cu");
    const S3: &str = include_str!("../../kernels/gemm_bi_fixed/sm89_half_s3.cu");

    #[test]
    fn exact_export_inlines_literal_full_domain_into_the_measured_n192_body() {
        let candidate = candidate_source(SWIZZLE, S3).unwrap();
        assert_eq!(candidate.matches(&format!("void {SYMBOL}(")).count(), 1);
        assert_eq!(
            candidate
                .matches(&format!("void {RETAINED_SYMBOL}("))
                .count(),
            1
        );
        assert!(candidate.contains("C, A, B, alpha, 2048, 768, 1536);"));
        assert!(candidate.contains("(void)M;"));
        assert!(candidate.contains("(void)N;"));
        assert!(candidate.contains("(void)K_out;"));
    }

    #[test]
    fn exact_export_changes_no_measured_kernel_source_or_arithmetic() {
        let retained = retained_source(SWIZZLE, S3).unwrap();
        let candidate = candidate_source(SWIZZLE, S3).unwrap();
        assert_eq!(
            restore_retained_source(&candidate, SWIZZLE, S3).unwrap(),
            retained
        );
        assert!(candidate.starts_with(&retained));
        for anchor in [
            "cp.async.cg.shared.global",
            "cp.async.wait_group 1;",
            "mma.sync.aligned.m16n8k16.row.col.f32.",
            "gbf_store_pair_rne",
        ] {
            assert_eq!(
                candidate[..retained.len()].matches(anchor).count(),
                retained.matches(anchor).count()
            );
        }
    }

    #[test]
    fn target_is_fully_divisible_and_fills_the_n192_grid() {
        assert_eq!(TARGET.0 % 64, 0);
        assert_eq!(TARGET.1 % 192, 0);
        assert_eq!(TARGET.2 % 64, 0);
        assert_eq!(TARGET.0.div_ceil(64) * TARGET.1.div_ceil(192), 256);
        assert_eq!(TARGET_GRID, 256);
    }

    #[test]
    fn malformed_parent_fails_closed() {
        assert!(candidate_source("", S3).is_err());
        assert!(candidate_source(SWIZZLE, "").is_err());
    }
}
