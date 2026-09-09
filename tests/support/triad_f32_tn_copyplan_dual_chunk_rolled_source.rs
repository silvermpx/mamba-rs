#[path = "triad_f32_tn_copyplan_dual_chunk_source.rs"]
mod retained;

pub const ROLLED_DUAL_RAW_SYMBOL: &str =
    "gemm_bi_tn_test_fixed_sm89_f32_n64_dual_chunk_rolled_raw_v1";
pub const ROLLED_DUAL_FUSED_SYMBOL: &str =
    "gemm_bi_tn_test_fixed_sm89_f32_n64_dual_chunk_rolled_fused_finalize_v1";

const RETAINED_BODY: &str = r#"    float acc0[8][4];
    float acc1[8][4];
    DUAL_RUN_CHAIN(acc0, a, b, k0);
    __syncthreads();
    const float* a1 = a + k0;
    const float* b1 = b + (long long)k0 * ldb;
    DUAL_RUN_CHAIN(acc1, a1, b1, k1);"#;

const ROLLED_BODY: &str = r#"    float acc[8][4];
    float partial0[8][4];
#pragma unroll 1
    for (int chain = 0; chain < 2; ++chain) {
        ROLLED_RUN_CHAIN(
            acc,
            chain == 0 ? a : a + k0,
            chain == 0 ? b : b + (long long)k0 * ldb,
            chain == 0 ? k0 : k1);
        if (chain == 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    partial0[i][j] = acc[i][j];
            }
            __syncthreads();
        }
    }"#;

fn replace_once(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    let count = source.matches(before).count();
    if count != 1 {
        return Err(format!(
            "rolled dual-chunk anchor {before:?}: expected1 observed{count}"
        ));
    }
    *source = source.replacen(before, after, 1);
    Ok(())
}

fn replace_exact_count(
    source: &mut String,
    before: &str,
    after: &str,
    expected: usize,
) -> Result<(), String> {
    let count = source.matches(before).count();
    if count != expected {
        return Err(format!(
            "rolled dual-chunk anchor {before:?}: expected{expected} observed{count}"
        ));
    }
    *source = source.replace(before, after);
    Ok(())
}

fn transform(
    mut source: String,
    retained_symbol: &str,
    rolled_symbol: &str,
) -> Result<String, String> {
    replace_once(&mut source, retained_symbol, rolled_symbol)?;
    replace_once(&mut source, RETAINED_BODY, ROLLED_BODY)?;
    replace_exact_count(&mut source, "DUAL_RUN_CHAIN", "ROLLED_RUN_CHAIN", 2)?;
    replace_once(
        &mut source,
        "plane0[idx] = acc0[i][j];\n            plane1[idx] = acc1[i][j];",
        "plane0[idx] = partial0[i][j];\n            plane1[idx] = acc[i][j];",
    )
    .or_else(|_| {
        replace_once(
            &mut source,
            "__dadd_rn((double)acc0[i][j], (double)acc1[i][j])",
            "__dadd_rn((double)partial0[i][j], (double)acc[i][j])",
        )
    })?;
    Ok(source)
}

fn reverse(
    mut source: String,
    retained_symbol: &str,
    rolled_symbol: &str,
) -> Result<String, String> {
    replace_once(&mut source, rolled_symbol, retained_symbol)?;
    replace_once(&mut source, ROLLED_BODY, RETAINED_BODY)?;
    replace_exact_count(&mut source, "ROLLED_RUN_CHAIN", "DUAL_RUN_CHAIN", 2)?;
    replace_once(
        &mut source,
        "plane0[idx] = partial0[i][j];\n            plane1[idx] = acc[i][j];",
        "plane0[idx] = acc0[i][j];\n            plane1[idx] = acc1[i][j];",
    )
    .or_else(|_| {
        replace_once(
            &mut source,
            "__dadd_rn((double)partial0[i][j], (double)acc[i][j])",
            "__dadd_rn((double)acc0[i][j], (double)acc1[i][j])",
        )
    })?;
    Ok(source)
}

pub fn compose_raw_source() -> Result<String, String> {
    transform(
        retained::compose_raw_source(),
        retained::DUAL_RAW_SYMBOL,
        ROLLED_DUAL_RAW_SYMBOL,
    )
}

pub fn compose_fused_source() -> Result<String, String> {
    transform(
        retained::compose_fused_source(),
        retained::DUAL_FUSED_SYMBOL,
        ROLLED_DUAL_FUSED_SYMBOL,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_rolled_contract(source: &str, symbol: &str) {
        assert_eq!(source.matches(symbol).count(), 1);
        assert_eq!(source.matches("#pragma unroll 1").count(), 1);
        assert_eq!(
            source
                .matches("for (int chain = 0; chain < 2; ++chain)")
                .count(),
            1
        );
        assert_eq!(
            source
                .matches("ROLLED_RUN_CHAIN(\n            acc,")
                .count(),
            1
        );
        assert!(source.contains("float partial0[8][4];"));
        assert!(source.contains("partial0[i][j] = acc[i][j];"));
        assert!(!source.contains("float acc0[8][4]"));
        assert!(!source.contains("float acc1[8][4]"));
    }

    #[test]
    fn raw_candidate_rolls_one_unchanged_mainloop_over_two_exact_chains() {
        let source = compose_raw_source().unwrap();
        assert_rolled_contract(&source, ROLLED_DUAL_RAW_SYMBOL);
        assert!(source.contains("plane0[idx] = partial0[i][j];"));
        assert!(source.contains("plane1[idx] = acc[i][j];"));
        assert_eq!(
            reverse(source, retained::DUAL_RAW_SYMBOL, ROLLED_DUAL_RAW_SYMBOL).unwrap(),
            retained::compose_raw_source()
        );
    }

    #[test]
    fn fused_candidate_keeps_the_exact_fp64_finalize_order() {
        let source = compose_fused_source().unwrap();
        assert_rolled_contract(&source, ROLLED_DUAL_FUSED_SYMBOL);
        let normalized = source.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(
            normalized.contains(
                "const double sum = __dadd_rn((double)partial0[i][j], (double)acc[i][j]);"
            )
        );
        assert!(source.contains("__dmul_rn((double)alpha, sum)"));
        assert!(source.contains("__double2float_rn"));
        assert!(source.contains("output[idx] = __fadd_rn(output[idx], update);"));
        assert_eq!(
            reverse(
                source,
                retained::DUAL_FUSED_SYMBOL,
                ROLLED_DUAL_FUSED_SYMBOL
            )
            .unwrap(),
            retained::compose_fused_source()
        );
    }
}
