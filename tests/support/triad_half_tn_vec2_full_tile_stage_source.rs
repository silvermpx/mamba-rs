#[path = "triad_half_tn_vec2_epilogue_source.rs"]
mod retained;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_tc64_bk64_s2_regpipe_vec2_full_tile_stage_";
pub const RETAINED_SYMBOL_PREFIX: &str = retained::SYMBOL_PREFIX;
pub const BLOCK_THREADS: u32 = 128;
pub const STATIC_SHARED_BYTES: i32 = 32_768;
pub const REQUIRED_OCCUPANCY: u32 = 3;
pub const REGISTER_CAP: i32 = 128;
pub const TEXT_RATIO_CAP: f64 = 1.20;
pub const EXPECTED_HMMA: usize = 32;
pub const EXPECTED_LDSM: usize = 24;
pub const TARGET: (usize, usize, usize) = (2_048, 768, 3_072);
pub const TARGET_GRID: u32 = 576;

pub fn retained_source(production: &str) -> Result<String, String> {
    retained::candidate_source(production)
}

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = retained_source(production)?;
    replace_exact(
        &mut source,
        SCALAR_STAGE_MARKER,
        &format!("{FULL_TILE_STAGE_MACRO}{SCALAR_STAGE_MARKER}"),
        "full-tile stage helper insertion",
    )?;
    replace_exact(
        &mut source,
        FAST_STAGE_END,
        &format!("{FAST_STAGE_END}{}", encode_macro_block(FULL_TILE_FLAG)),
        "target-only admission",
    )?;
    replace_exact(
        &mut source,
        RETAINED_PROLOGUE,
        FULL_TILE_PROLOGUE,
        "prologue selection",
    )?;
    replace_exact(
        &mut source,
        RETAINED_REFILL,
        FULL_TILE_REFILL,
        "refill selection",
    )?;
    replace_exact(
        &mut source,
        RETAINED_SYMBOL_PREFIX,
        SYMBOL_PREFIX,
        "candidate export",
    )?;
    Ok(source)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    replace_exact(
        &mut source,
        SYMBOL_PREFIX,
        RETAINED_SYMBOL_PREFIX,
        "restore export",
    )?;
    replace_exact(
        &mut source,
        FULL_TILE_REFILL,
        RETAINED_REFILL,
        "restore refill",
    )?;
    replace_exact(
        &mut source,
        FULL_TILE_PROLOGUE,
        RETAINED_PROLOGUE,
        "restore prologue",
    )?;
    replace_exact(
        &mut source,
        &format!("{FAST_STAGE_END}{}", encode_macro_block(FULL_TILE_FLAG)),
        FAST_STAGE_END,
        "restore target-only admission",
    )?;
    replace_exact(
        &mut source,
        &format!("{FULL_TILE_STAGE_MACRO}{SCALAR_STAGE_MARKER}"),
        SCALAR_STAGE_MARKER,
        "restore full-tile stage helper",
    )?;
    Ok(source)
}

pub const fn uses_full_tile_stage(reduction: usize, k_out: usize, n: usize) -> bool {
    reduction == TARGET.0 && k_out == TARGET.1 && n == TARGET.2
}

pub fn all_strata_below(strata: &[[f64; 2]], threshold: f64) -> bool {
    strata.len() == 4
        && threshold.is_finite()
        && threshold > 0.0
        && strata.iter().all(|[p50, p95]| {
            p50.is_finite()
                && p95.is_finite()
                && *p50 > 0.0
                && *p95 > 0.0
                && *p50 < threshold
                && *p95 < threshold
        })
}

const SCALAR_STAGE_MARKER: &str = "#define GEMM_BI_TC64_STAGE_TN_SCALAR";
const FAST_STAGE_END: &str =
    "                      ((K_out & 7) == 0) && ((N & 7) == 0);                    \\\n";
const FULL_TILE_FLAG: &str = r#"    bool full_tile_stage = fast_stage &&
                           M_red == 2048 && K_out == 768 && N == 3072;
"#;

const RETAINED_PROLOGUE: &str = r#"    if (fast_stage) {                                                          \
        GEMM_BI_TC64_STAGE_TN_ASYNC(0, 0);                                         \
    } else {                                                                   \
        GEMM_BI_TC64_STAGE_TN_SCALAR(0, 0, T_ACT, FROM_F);                         \
    }                                                                          \"#;

const FULL_TILE_PROLOGUE: &str = r#"    if (full_tile_stage) {                                                     \
        GEMM_BI_TC64_STAGE_TN_ASYNC_FULL(0, 0);                                    \
    } else if (fast_stage) {                                                    \
        GEMM_BI_TC64_STAGE_TN_ASYNC(0, 0);                                         \
    } else {                                                                   \
        GEMM_BI_TC64_STAGE_TN_SCALAR(0, 0, T_ACT, FROM_F);                         \
    }                                                                          \"#;

const RETAINED_REFILL: &str = r#"            if (fast_stage) {                                                  \
                GEMM_BI_TC64_STAGE_TN_ASYNC(read_buf ^ 1, (mt + 1) * GEMM_BI_TC64_BK); \
            } else {                                                           \
                GEMM_BI_TC64_STAGE_TN_SCALAR(read_buf ^ 1, (mt + 1) * GEMM_BI_TC64_BK, \
                                         T_ACT, FROM_F);                       \
            }                                                                  \"#;

const FULL_TILE_REFILL: &str = r#"            if (full_tile_stage) {                                             \
                GEMM_BI_TC64_STAGE_TN_ASYNC_FULL(                              \
                    read_buf ^ 1, (mt + 1) * GEMM_BI_TC64_BK);                 \
            } else if (fast_stage) {                                           \
                GEMM_BI_TC64_STAGE_TN_ASYNC(read_buf ^ 1, (mt + 1) * GEMM_BI_TC64_BK); \
            } else {                                                           \
                GEMM_BI_TC64_STAGE_TN_SCALAR(read_buf ^ 1, (mt + 1) * GEMM_BI_TC64_BK, \
                                         T_ACT, FROM_F);                       \
            }                                                                  \"#;

const FULL_TILE_STAGE_MACRO: &str = r#"#define GEMM_BI_TC64_STAGE_TN_ASYNC_FULL(buf, mIdx)                           \
    do {                                                                      \
        unsigned _xs =                                                        \
            Xs_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2); \
        unsigned _ys =                                                        \
            Ys_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2); \
        for (int _i = threadIdx.x;                                             \
             _i < GEMM_BI_TC64_BK * (GEMM_BI_TC64_BM / 8);                   \
             _i += GEMM_BI_TC64_THREADS) {                                    \
            int _r = _i / (GEMM_BI_TC64_BM / 8);                             \
            int _c = (_i % (GEMM_BI_TC64_BM / 8)) * 8;                       \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * GEMM_BI_TC64_BM + _c;                           \
            unsigned _dst = _xs +                                             \
                (unsigned)(GEMM_BI_HALF_TN_INDEX(_r, _c) * 2);                \
            const void* _src = A + (long long)_gm * K_out + _gk;             \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"         \
                         :: "r"(_dst), "l"(_src));                            \
        }                                                                     \
        for (int _i = threadIdx.x;                                             \
             _i < GEMM_BI_TC64_BK * (GEMM_BI_TC64_BN / 8);                   \
             _i += GEMM_BI_TC64_THREADS) {                                    \
            int _r = _i / (GEMM_BI_TC64_BN / 8);                             \
            int _c = (_i % (GEMM_BI_TC64_BN / 8)) * 8;                       \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_TC64_BN + _c;                           \
            unsigned _dst = _ys +                                             \
                (unsigned)(GEMM_BI_HALF_TN_INDEX(_r, _c) * 2);                \
            const void* _src = B + (long long)_gm * N + _gn;                 \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"         \
                         :: "r"(_dst), "l"(_src));                            \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                             \
    } while (0)

"#;

fn replace_exact(
    source: &mut String,
    before: &str,
    after: &str,
    label: &str,
) -> Result<(), String> {
    let actual = source.matches(before).count();
    if actual != 1 {
        return Err(format!(
            "half TN full-tile stage {label} seam expected 1, observed {actual}"
        ));
    }
    *source = source.replacen(before, after, 1);
    Ok(())
}

fn encode_macro_block(decoded: &str) -> String {
    decoded.lines().map(|line| format!("{line} \\\n")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn target_uses_direct_full_tile_stage_and_every_other_shape_keeps_fallback() {
        let source = candidate_source(PRODUCTION).unwrap();
        assert!(source.contains("bool full_tile_stage = fast_stage &&"));
        assert!(source.contains("M_red == 2048 && K_out == 768 && N == 3072"));
        assert_eq!(
            source.matches("GEMM_BI_TC64_STAGE_TN_ASYNC_FULL(").count(),
            3
        );
        assert!(source.contains("else if (fast_stage)"));
        assert!(source.contains("GEMM_BI_TC64_STAGE_TN_ASYNC(0, 0);"));
        assert!(
            source
                .contains("GEMM_BI_TC64_STAGE_TN_ASYNC(read_buf ^ 1, (mt + 1) * GEMM_BI_TC64_BK);")
        );
    }

    #[test]
    fn full_stage_removes_per_copy_extent_work_but_preserves_copy_ownership() {
        let source = candidate_source(PRODUCTION).unwrap();
        let start = source
            .find("#define GEMM_BI_TC64_STAGE_TN_ASYNC_FULL")
            .unwrap();
        let end = source[start..]
            .find("#define GEMM_BI_TC64_STAGE_TN_SCALAR")
            .map(|offset| start + offset)
            .unwrap();
        let full = &source[start..end];
        assert_eq!(
            full.matches("cp.async.ca.shared.global [%0], [%1], 16;")
                .count(),
            2
        );
        assert!(!full.contains("gemm_bi_cp_async_valid_elems"));
        assert!(!full.contains("gemm_bi_cp_async_source"));
        assert!(!full.contains("gemm_bi_cp_async_16_zfill"));
        assert_eq!(full.matches("GEMM_BI_HALF_TN_INDEX(_r, _c)").count(), 2);
        assert!(full.contains("A + (long long)_gm * K_out + _gk"));
        assert!(full.contains("B + (long long)_gm * N + _gn"));
    }

    #[test]
    fn admission_is_exact_target_only_including_each_tail_and_k0() {
        assert!(uses_full_tile_stage(2_048, 768, 3_072));
        assert!(!uses_full_tile_stage(2_047, 768, 3_072));
        assert!(!uses_full_tile_stage(2_048, 767, 3_072));
        assert!(!uses_full_tile_stage(2_048, 768, 3_071));
        assert!(!uses_full_tile_stage(67, 72, 72));
        assert!(!uses_full_tile_stage(67, 69, 71));
        assert!(!uses_full_tile_stage(64, 64, 64));
        assert!(!uses_full_tile_stage(0, 65, 67));
    }

    #[test]
    fn transform_is_reversible_and_keeps_math_epilogue_and_sync_byte_identical() {
        let retained = retained_source(PRODUCTION).unwrap();
        let candidate = candidate_source(PRODUCTION).unwrap();
        assert_eq!(restore_retained_source(&candidate).unwrap(), retained);
        for anchor in [
            "mma.sync.aligned.m16n8k16.row.col.f32.",
            "gemm_bi_accumulate_float2_or_scalar(",
            "cp.async.wait_group 0;",
            "__syncthreads();",
            "read_buf ^= 1;",
        ] {
            assert_eq!(
                candidate.matches(anchor).count(),
                retained.matches(anchor).count()
            );
        }
        assert_eq!(candidate.matches(SYMBOL_PREFIX).count(), 1);
    }

    #[test]
    fn malformed_or_ambiguous_parent_fails_closed() {
        assert!(candidate_source("").is_err());
        let duplicate = PRODUCTION.replacen(
            SCALAR_STAGE_MARKER,
            &format!("{SCALAR_STAGE_MARKER}\n{SCALAR_STAGE_MARKER}"),
            1,
        );
        assert!(candidate_source(&duplicate).is_err());
    }
}
