pub const SYMBOL: &str = "gemm_bi_tn_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2";

const HELPER: &str = include_str!("../gemm_bi_tf32_tn_compact_xor.cuh");

struct Transformation {
    label: &'static str,
    from: &'static str,
    to: &'static str,
}

const TRANSFORMATIONS: [Transformation; 8] = [
    Transformation {
        label: "TN compact storage",
        from: concat!(
            "    static constexpr int bk32 = 32;\n",
            "    static constexpr int ARows = Op == SgbTf32Tn ? bk32 : BM;\n",
            "    static constexpr int AStride = Op == SgbTf32Tn ? BM + 8 : 36;\n",
            "    static constexpr int BRows = Op == SgbTf32Nt ? BN : bk32;\n",
            "    static constexpr int BStride = Op == SgbTf32Nt ? 36\n",
            "        : (BN == 64 ? 72 : (BN == 32 ? 40 : 24));"
        ),
        to: concat!(
            "    static constexpr int bk32 = 32;\n",
            "    static constexpr bool tn_compact_eight_warp_s2 =\n",
            "        Op == SgbTf32Tn && BM == 128 && BN == 64 && Stages == 2;\n",
            "    static constexpr int ARows = Op == SgbTf32Tn ? bk32 : BM;\n",
            "    static constexpr int AStride = tn_compact_eight_warp_s2 ? BM\n",
            "        : (Op == SgbTf32Tn ? BM + 8 : 36);\n",
            "    static constexpr int BRows = Op == SgbTf32Nt ? BN : bk32;\n",
            "    static constexpr int BStride = tn_compact_eight_warp_s2 ? BN\n",
            "        : (Op == SgbTf32Nt ? 36\n",
            "        : (BN == 64 ? 72 : (BN == 32 ? 40 : 24)));"
        ),
    },
    Transformation {
        label: "TN compact A slot",
        from: concat!(
            "    if constexpr (Op == SgbTf32Tn) {\n",
            "        return storage->a[stage][reduction][row];\n",
            "    }\n",
            "    return storage->a[stage][row][reduction];"
        ),
        to: concat!(
            "    if constexpr (Op == SgbTf32Tn) {\n",
            "        if constexpr (BM == 128 && BN == 64 && Stages == 2) {\n",
            "            return storage->a[stage][reduction]\n",
            "                [gemm_bi_tf32_tn_test_compact_xor_axis(row, reduction)];\n",
            "        }\n",
            "        return storage->a[stage][reduction][row];\n",
            "    }\n",
            "    return storage->a[stage][row][reduction];"
        ),
    },
    Transformation {
        label: "TN compact B slot",
        from: concat!(
            "    if constexpr (Op == SgbTf32Nt) {\n",
            "        return storage->b[stage][column][reduction];\n",
            "    }\n",
            "    return storage->b[stage][reduction][column];"
        ),
        to: concat!(
            "    if constexpr (Op == SgbTf32Nt) {\n",
            "        return storage->b[stage][column][reduction];\n",
            "    }\n",
            "    if constexpr (Op == SgbTf32Tn && BM == 128 && BN == 64 && Stages == 2) {\n",
            "        return storage->b[stage][reduction]\n",
            "            [gemm_bi_tf32_tn_test_compact_xor_axis(column, reduction)];\n",
            "    }\n",
            "    return storage->b[stage][reduction][column];"
        ),
    },
    Transformation {
        label: "TN compact storage extent",
        from: "== 53248, \"TN M128N64 s2 storage\"",
        to: "== 49152, \"TN compact-eight-warp M128N64 s2 storage\"",
    },
    Transformation {
        label: "TN compact accumulator ownership",
        from: concat!(
            "__device__ __forceinline__ void gemm_bi_tf32_kernel(\n",
            "    float* output, const float* a, const float* b, const float* bias,\n",
            "    Sm80Tf32KernelParams params) {\n",
            "    constexpr int MAtoms = BM == 128 ? 4 : (BM == 64 ? 2 : 1);"
        ),
        to: concat!(
            "__device__ __forceinline__ void gemm_bi_tf32_kernel(\n",
            "    float* output, const float* a, const float* b, const float* bias,\n",
            "    Sm80Tf32KernelParams params) {\n",
            "    constexpr bool tn_compact_eight_warp_s2 =\n",
            "        Op == SgbTf32Tn && BM == 128 && BN == 64 && Stages == 2;\n",
            "    constexpr int MAtoms = tn_compact_eight_warp_s2 ? 2\n",
            "        : (BM == 128 ? 4 : (BM == 64 ? 2 : 1));"
        ),
    },
    Transformation {
        label: "TN compact compute and row ownership",
        from: concat!(
            "    bool compute = BM != 128 || warp < 4;\n",
            "    int warp_m = BM == 128 ? (warp >> 1) * 64\n",
            "        : (BM == 64 ? (warp >> 1) * 32 : 0);"
        ),
        to: concat!(
            "    bool compute = tn_compact_eight_warp_s2 || BM != 128 || warp < 4;\n",
            "    int warp_m = tn_compact_eight_warp_s2 ? (warp >> 1) * 32\n",
            "        : (BM == 128 ? (warp >> 1) * 64\n",
            "        : (BM == 64 ? (warp >> 1) * 32 : 0));"
        ),
    },
    Transformation {
        label: "TN compact target symbol",
        from: concat!(
            "GEMM_BI_TF32_DEFINE_KERNEL(",
            "gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s2, ",
            "SgbTf32Tn, 128, 64, 2, 256, 1)"
        ),
        to: concat!(
            "GEMM_BI_TF32_DEFINE_KERNEL(",
            "gemm_bi_tn_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2, ",
            "SgbTf32Tn, 128, 64, 2, 256, 1)"
        ),
    },
    Transformation {
        label: "TN compact target signature",
        from: "TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s2);",
        to: concat!(
            "TF32_ASSERT_KERNEL_SIGNATURE(",
            "gemm_bi_tn_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2);"
        ),
    },
];

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = production.to_owned();
    for transformation in TRANSFORMATIONS {
        replace_exact(
            &mut source,
            transformation.from,
            transformation.to,
            transformation.label,
        )?;
    }
    Ok(format!("{HELPER}\n{source}"))
}

fn replace_exact(source: &mut String, from: &str, to: &str, label: &str) -> Result<(), String> {
    let count = source.matches(from).count();
    if count != 1 {
        return Err(format!(
            "{label} source boundary count changed: expected 1, observed {count}"
        ));
    }
    *source = source.replacen(from, to, 1);
    Ok(())
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn candidate_is_exactly_tn_m128n64_s2_scoped() {
        let source = candidate_source(PRODUCTION).unwrap();
        assert!(source.contains("Op == SgbTf32Tn && BM == 128 && BN == 64 && Stages == 2"));
        assert!(source.contains("== 49152, \"TN compact-eight-warp M128N64 s2 storage\""));
        assert!(source.contains("gemm_bi_tf32_tn_test_compact_xor_axis(row, reduction)"));
        assert!(source.contains("gemm_bi_tf32_tn_test_compact_xor_axis(column, reduction)"));
        assert!(source.contains(&format!(
            "GEMM_BI_TF32_DEFINE_KERNEL({SYMBOL}, SgbTf32Tn, 128, 64, 2, 256, 1)"
        )));
        assert!(source.contains(&format!("TF32_ASSERT_KERNEL_SIGNATURE({SYMBOL});")));
        assert!(!source.contains(
            "TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s2);"
        ));
        assert!(
            source
                .contains("GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s2")
        );
    }

    #[test]
    fn candidate_fails_closed_when_target_symbol_count_drifts() {
        const TARGET: &str = concat!(
            "GEMM_BI_TF32_DEFINE_KERNEL(",
            "gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s2, ",
            "SgbTf32Tn, 128, 64, 2, 256, 1)"
        );
        let missing = PRODUCTION.replacen(TARGET, "", 1);
        let missing_error = candidate_source(&missing).unwrap_err();
        assert!(missing_error.contains("target symbol"), "{missing_error}");
        assert!(
            missing_error.contains("expected 1, observed 0"),
            "{missing_error}"
        );

        let duplicate = format!("{PRODUCTION}\n{TARGET}");
        let duplicate_error = candidate_source(&duplicate).unwrap_err();
        assert!(
            duplicate_error.contains("target symbol"),
            "{duplicate_error}"
        );
        assert!(
            duplicate_error.contains("expected 1, observed 2"),
            "{duplicate_error}"
        );
    }
}
