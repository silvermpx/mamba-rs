#[path = "triad_half_nt_fixed_s3_source.rs"]
mod fixed_s3;

pub const SYMBOL_PREFIX: &str = "gemm_bi_nt_test_fixed_s3_direct_epilogue_";
const VECTOR_DISPATCH: &str = concat!(
    "    {\n",
    "        bool vector_output = (K_out & 7) == 0 && gbf_aligned16(C)\n",
    "            && pid_n * 128 + 128 <= K_out;\n",
    "        if (vector_output) {\n",
    "            sm89_fixed_half_swizzle::vector_epilogue(C, reinterpret_cast<float*>(sm89_fhs_shared), acc, alpha,\n",
    "                M, K_out, pid_m, pid_n, warpM, warpN);\n",
    "            return;\n",
    "        }\n",
    "    }\n"
);
const DIRECT_CALL: &str =
    "    sm89_fixed_half_swizzle::scalar_epilogue(C, acc, alpha, 0.0f, M, K_out, K_out,";

pub fn candidate_source(swizzle: &str, s3: &str) -> Result<String, String> {
    let source = fixed_s3::candidate_source(swizzle, s3)?;
    for anchor in [VECTOR_DISPATCH, DIRECT_CALL, fixed_s3::SYMBOL_PREFIX] {
        let count = source.matches(anchor).count();
        if count != 1 {
            return Err(format!(
                "NT direct epilogue source boundary changed: expected1, found{count}: {anchor:?}"
            ));
        }
    }
    Ok(source
        .replacen(VECTOR_DISPATCH, "", 1)
        .replace(fixed_s3::SYMBOL_PREFIX, SYMBOL_PREFIX))
}

#[cfg(test)]
mod tests {
    use super::*;
    const SWIZZLE: &str = include_str!("../../kernels/gemm_bi_inference/sm89_half_swizzle.cu");
    const S3: &str = include_str!("../../kernels/gemm_bi_inference/sm89_half_s3.cu");

    #[test]
    fn only_output_dispatch_and_export_change() {
        let base = fixed_s3::candidate_source(SWIZZLE, S3).unwrap();
        let source = candidate_source(SWIZZLE, S3).unwrap();
        assert_eq!(source.matches(SYMBOL_PREFIX).count(), 1);
        assert!(!source.contains(fixed_s3::SYMBOL_PREFIX));
        assert!(!source.contains(VECTOR_DISPATCH));
        assert_eq!(source.matches(DIRECT_CALL).count(), 1);
        assert!(source.contains("gbf_store_pair_rne(destination, first, second);"));
        assert!(source.contains("float first = __fmul_rn(alpha, acc[fm][fn][2 * half]);"));
        assert!(source.contains("float second = __fmul_rn(alpha, acc[fm][fn][2 * half + 1]);"));
        let restored = source
            .replace(SYMBOL_PREFIX, fixed_s3::SYMBOL_PREFIX)
            .replacen(DIRECT_CALL, &format!("{VECTOR_DISPATCH}{DIRECT_CALL}"), 1);
        assert_eq!(restored, base);
    }

    #[test]
    fn parent_layout_or_pipeline_drift_is_rejected() {
        assert!(candidate_source("", S3).is_err());
        assert!(candidate_source(SWIZZLE, "").is_err());
    }
}
