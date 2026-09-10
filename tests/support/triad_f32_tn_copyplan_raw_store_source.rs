pub const PRODUCTION_SYMBOL: &str = "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1";
pub const RAW_SYMBOL: &str = "gemm_bi_tn_test_fixed_sm89_f32_n64_copyplan_raw_v1";

const PRELUDE: &str = include_str!("../../kernels/_typed_prelude.cuh");
const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_inference/sm89_f32_n64_copyplan.cu");

fn replace_once(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    let count = source.matches(before).count();
    if count != 1 {
        return Err(format!(
            "raw-store CopyPlan anchor {before:?}: expected1 observed{count}"
        ));
    }
    *source = source.replacen(before, after, 1);
    Ok(())
}

/// Test-only clone of the unchanged production CopyPlan mainloop. The sole
/// transformation removes the alpha=1 multiplication at the physical partial
/// store so NaN payload bits cannot be altered after the exact FMA chain.
pub fn compose_source() -> Result<String, String> {
    let mut body = PRODUCTION.to_owned();
    replace_once(&mut body, PRODUCTION_SYMBOL, RAW_SYMBOL)?;
    replace_once(
        &mut body,
        "float val0 = __fmul_rn(alpha, acc[i][j]);",
        "float val0 = acc[i][j];",
    )?;
    replace_once(
        &mut body,
        "float val1 = __fmul_rn(alpha, acc[i][j + 1]);",
        "float val1 = acc[i][j + 1];",
    )?;
    replace_once(
        &mut body,
        "float val = __fmul_rn(alpha, acc[i][j]);",
        "float val = acc[i][j];",
    )?;
    Ok(format!(
        "{PRELUDE}\n#define gbf_aligned16 gemm_bi_tn_test_aligned16\n__device__ __forceinline__ bool gemm_bi_tn_test_aligned16(const void* p) {{ return (reinterpret_cast<unsigned long long>(p) & 15ULL) == 0ULL; }}\n{body}\n#undef gbf_aligned16\n"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_is_symbol_and_raw_store_only() {
        let transformed = compose_source().unwrap();
        assert_eq!(transformed.matches(RAW_SYMBOL).count(), 1);
        assert!(!transformed.contains(PRODUCTION_SYMBOL));
        assert_eq!(transformed.matches("float val0 = acc[i][j];").count(), 1);
        assert_eq!(
            transformed.matches("float val1 = acc[i][j + 1];").count(),
            1
        );
        assert_eq!(transformed.matches("float val = acc[i][j];").count(), 1);
        assert_eq!(
            transformed
                .matches("__fmaf_rn(a_reg[i], b_reg[j], acc[i][j])")
                .count(),
            2
        );
    }
}
