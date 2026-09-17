#[path = "triad_half_tn_regpipe_source.rs"]
pub mod regpipe;

pub const SYMBOL_PREFIX: &str = "tn_test_tc64_bk64_s2_regpipe_vec2_";

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = regpipe::candidate_source(production)?;
    replace_exact(&mut source, OLD_EPILOGUE, NEW_EPILOGUE)?;
    replace_exact(&mut source, regpipe::SYMBOL_PREFIX, SYMBOL_PREFIX)?;
    Ok(source)
}

const OLD_EPILOGUE: &str = r#"    /* epilogue: f32 accumulate into dW */                                     \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            int r0 = pid_m * GEMM_BI_TC64_BM + warpM + fm * 16 + g;                \
            int c0 = pid_n * GEMM_BI_TC64_BN + warpN + fn * 8 + 2 * t;             \
            _Pragma("unroll")                                                  \
            for (int e = 0; e < 4; e++) {                                      \
                int gr = r0 + (e >= 2 ? 8 : 0);                                \
                int gc = c0 + (e & 1);                                         \
                if (gr >= K_out || gc >= N) continue;                          \
                C[(long long)gr * N + gc] += alpha * acc[fm][fn][e];           \
            }                                                                  \
        }                                                                      \
    }                                                                          \"#;

const NEW_EPILOGUE: &str = r#"    /* epilogue: paired f32 accumulate into dW, scalar at the N tail */         \
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
                    accumulate_float2_or_scalar(                       \
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

fn replace_exact(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    let actual = source.matches(before).count();
    if actual != 1 {
        return Err(format!(
            "TN vec2 epilogue anchor expected 1, observed {actual}"
        ));
    }
    *source = source.replacen(before, after, 1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80/mma.cu");

    #[test]
    fn target_epilogue_uses_aligned_pairs_and_scalar_tail() {
        let source = candidate_source(PRODUCTION).unwrap();
        assert!(source.contains("void tn_test_tc64_bk64_s2_regpipe_vec2_##SUFFIX"));
        assert!(source.contains("accumulate_float2_or_scalar("));
        assert!(
            source.contains("bool packed = gc + 1 < N && (((unsigned long long)dst & 7ull) == 0);")
        );
        assert!(source.contains("dst[0] += alpha * acc[fm][fn][2 * half];"));
        assert!(source.contains("dst[1] += alpha * acc[fm][fn][2 * half + 1];"));
        assert!(!source.contains(OLD_EPILOGUE));
    }

    #[test]
    fn reduction_pipeline_and_source_outside_epilogue_are_unchanged() {
        let incumbent = regpipe::candidate_source(PRODUCTION).unwrap();
        let actual = candidate_source(PRODUCTION).unwrap();
        assert_eq!(
            actual
                .matches("mma.sync.aligned.m16n8k16.row.col.f32.")
                .count(),
            incumbent
                .matches("mma.sync.aligned.m16n8k16.row.col.f32.")
                .count()
        );
        assert!(actual.contains("a_frag[(ks + 1) & 1][fm]"));
        assert!(actual.contains("a_frag[ks & 1][fm]"));
        let normalized = actual
            .replace(NEW_EPILOGUE, OLD_EPILOGUE)
            .replace(SYMBOL_PREFIX, regpipe::SYMBOL_PREFIX);
        assert_eq!(normalized, incumbent);
    }

    #[test]
    fn malformed_parent_fails_closed() {
        assert!(candidate_source("").is_err());
        let duplicate = format!("{PRODUCTION}\n#define GEMM_BI_TC64_STAGE_TN_ASYNC\n");
        assert!(candidate_source(&duplicate).is_err());
    }
}
