#[path = "triad_half_tn_vec2_epilogue_source.rs"]
mod retained;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_tc64_bk64_s2_regpipe_vec2_full_domain_";
pub const RETAINED_SYMBOL_PREFIX: &str = retained::SYMBOL_PREFIX;
pub const BLOCK_THREADS: u32 = 128;
pub const STATIC_SHARED_BYTES: i32 = 32_768;
pub const REQUIRED_OCCUPANCY: u32 = 3;
pub const REGISTER_CAP: i32 = 125;
pub const TEXT_RATIO_CAP: f64 = 1.05;
pub const EXPECTED_HMMA: usize = 32;
pub const EXPECTED_LDGSTS: usize = 24;
pub const TARGET: (usize, usize, usize) = (2_048, 768, 3_072);
pub const TARGET_GRID: (u32, u32, u32) = (48, 12, 1);
pub const RETAINED_TARGET_GRID: (u32, u32, u32) = (576, 1, 1);

pub fn retained_source(production: &str) -> Result<String, String> {
    retained::candidate_source(production)
}

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = retained_source(production)?;
    replace_exact(&mut source, RETAINED_MAPPER, FULL_DOMAIN_MAPPER, "mapper")?;
    replace_exact(
        &mut source,
        RETAINED_EPILOGUE,
        FULL_DOMAIN_EPILOGUE,
        "epilogue",
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
        FULL_DOMAIN_EPILOGUE,
        RETAINED_EPILOGUE,
        "restore epilogue",
    )?;
    replace_exact(
        &mut source,
        FULL_DOMAIN_MAPPER,
        RETAINED_MAPPER,
        "restore mapper",
    )?;
    Ok(source)
}

pub const fn tile_coordinate(block_x: u32, block_y: u32) -> (u32, u32) {
    (block_y, block_x)
}

pub const fn is_full_domain_shape(reduction: usize, k_out: usize, n: usize) -> bool {
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

const RETAINED_MAPPER: &str = r#"    int num_pid_n = (N + GEMM_BI_TC64_BN - 1) / GEMM_BI_TC64_BN;                       \
    int pid_m = blockIdx.x / num_pid_n;                                        \
    int pid_n = blockIdx.x % num_pid_n;                                        \"#;

const FULL_DOMAIN_MAPPER: &str = r#"    /* exact d768-in full-domain 2D tile ownership */                              \
    int pid_m = blockIdx.y;                                                     \
    int pid_n = blockIdx.x;                                                     \"#;

const RETAINED_EPILOGUE: &str = r#"    /* epilogue: paired f32 accumulate into dW, scalar at the N tail */         \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            int r0 = pid_m * GEMM_BI_TC64_BM + warpM + fm * 16 + g;                \
            int c0 = pid_n * GEMM_BI_TC64_BN + warpN + fn * 8 + 2 * t;             \
            _Pragma("unroll")                                                  \
            for (int half = 0; half < 2; ++half) {                             \
                int gr = r0 + half * 8;                                        \
                int gc = c0;                                                   \
                if (gr >= K_out || gc >= N) continue;                          \
                float* dst = C + (long long)gr * N + gc;                       \
                bool packed = gc + 1 < N && (((unsigned long long)dst & 7ull) == 0); \
                if (packed) {                                                  \
                    gemm_bi_accumulate_float2_or_scalar(                       \
                        dst, alpha * acc[fm][fn][2 * half],                    \
                        alpha * acc[fm][fn][2 * half + 1], true);              \
                } else {                                                       \
                    dst[0] += alpha * acc[fm][fn][2 * half];                   \
                    if (gc + 1 < N)                                            \
                        dst[1] += alpha * acc[fm][fn][2 * half + 1];           \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \"#;

const FULL_DOMAIN_EPILOGUE: &str = r#"    /* exact d768-in full-domain paired f32 accumulate into dW */               \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            int r0 = pid_m * GEMM_BI_TC64_BM + warpM + fm * 16 + g;                \
            int c0 = pid_n * GEMM_BI_TC64_BN + warpN + fn * 8 + 2 * t;             \
            _Pragma("unroll")                                                  \
            for (int half = 0; half < 2; ++half) {                             \
                int gr = r0 + half * 8;                                        \
                int gc = c0;                                                   \
                float* dst = C + (long long)gr * N + gc;                       \
                gemm_bi_accumulate_float2_or_scalar(                           \
                    dst, alpha * acc[fm][fn][2 * half],                        \
                    alpha * acc[fm][fn][2 * half + 1], true);                  \
            }                                                                  \
        }                                                                      \
    }                                                                          \"#;

fn replace_exact(
    source: &mut String,
    before: &str,
    after: &str,
    label: &str,
) -> Result<(), String> {
    let actual = source.matches(before).count();
    if actual != 1 {
        return Err(format!(
            "half TN full-domain {label} seam expected 1, observed {actual}"
        ));
    }
    *source = source.replacen(before, after, 1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn full_domain_source_has_2d_mapper_and_unpredicated_paired_epilogue() {
        let source = candidate_source(PRODUCTION).unwrap();
        assert!(source.contains("int pid_m = blockIdx.y;"));
        assert!(source.contains("int pid_n = blockIdx.x;"));
        assert!(!source.contains("int num_pid_n = (N + GEMM_BI_TC64_BN - 1)"));
        assert!(source.contains("gemm_bi_accumulate_float2_or_scalar("));
        assert!(source.contains("alpha * acc[fm][fn][2 * half]"));
        assert!(!source.contains("if (gr >= K_out || gc >= N) continue;"));
        assert!(!source.contains("bool packed = gc + 1 < N"));
    }

    #[test]
    fn target_grid_owns_every_tile_exactly_once() {
        let mut seen = [[false; 48]; 12];
        for y in 0..TARGET_GRID.1 {
            for x in 0..TARGET_GRID.0 {
                let (m, n) = tile_coordinate(x, y);
                assert!(m < 12 && n < 48);
                assert!(!seen[m as usize][n as usize]);
                seen[m as usize][n as usize] = true;
            }
        }
        assert!(seen.into_iter().flatten().all(|owned| owned));
    }

    #[test]
    fn admission_is_exact_target_only_and_rejects_tail_shapes() {
        assert!(is_full_domain_shape(2_048, 768, 3_072));
        assert!(!is_full_domain_shape(67, 72, 72));
        assert!(!is_full_domain_shape(67, 69, 71));
        assert!(!is_full_domain_shape(64, 64, 64));
        assert!(!is_full_domain_shape(0, 65, 67));
    }

    #[test]
    fn transform_is_reversible_and_keeps_mainloop_byte_identical() {
        let retained = retained_source(PRODUCTION).unwrap();
        let candidate = candidate_source(PRODUCTION).unwrap();
        assert_eq!(restore_retained_source(&candidate).unwrap(), retained);
        assert_eq!(
            candidate
                .matches("mma.sync.aligned.m16n8k16.row.col.f32.")
                .count(),
            retained
                .matches("mma.sync.aligned.m16n8k16.row.col.f32.")
                .count()
        );
        assert_eq!(
            candidate.matches("gemm_bi_cp_async_16(").count(),
            retained.matches("gemm_bi_cp_async_16(").count()
        );
        assert_eq!(
            candidate.matches("__syncthreads();").count(),
            retained.matches("__syncthreads();").count()
        );
    }

    #[test]
    fn malformed_parent_fails_closed() {
        assert!(candidate_source("").is_err());
    }
}
