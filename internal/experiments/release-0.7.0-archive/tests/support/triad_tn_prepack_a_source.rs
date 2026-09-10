pub const GEMM_SYMBOL: &str = "gemm_bi_tn_test_prepack_a_sm80_mma_tf32_v1_m128n64_bk32_s3";
pub const PACK_SYMBOL: &str = "gemm_bi_tn_test_prepack_a_rna_tf32_v1";

fn replace_exact(
    source: &mut String,
    from: &str,
    to: &str,
    expected: usize,
    label: &str,
) -> Result<(), String> {
    let count = source.matches(from).count();
    if count != expected {
        return Err(format!(
            "{label} source boundary count changed: expected {expected}, observed {count}"
        ));
    }
    *source = source.replace(from, to);
    Ok(())
}

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = production.to_owned();
    const RNA_HELPER: &str = concat!(
        "__device__ __forceinline__ unsigned gemm_bi_tf32_rna(float value) {\n",
        "    unsigned result;\n",
        "    asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(result) : \"f\"(value));\n",
        "    return result;\n",
        "}\n"
    );
    let injected = format!(
        r#"{RNA_HELPER}

template <SgbTf32Op Op>
__device__ __forceinline__ unsigned gemm_bi_tf32_prepacked_a_word(float value) {{
    if constexpr (Op == SgbTf32Tn) return __float_as_uint(value);
    return gemm_bi_tf32_rna(value);
}}

extern "C" __global__ void {PACK_SYMBOL}(
    const float* input, unsigned* output, unsigned long long count) {{
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = gemm_bi_tf32_rna(input[index]);
}}
"#
    );
    replace_exact(&mut source, RNA_HELPER, &injected, 1, "RNA helper")?;
    for (from, to, label) in [
        (
            "gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread))",
            "gemm_bi_tf32_prepacked_a_word<Op>(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread))",
            "A fragment 0",
        ),
        (
            "gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread))",
            "gemm_bi_tf32_prepacked_a_word<Op>(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread))",
            "A fragment 1",
        ),
        (
            "gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread + 4))",
            "gemm_bi_tf32_prepacked_a_word<Op>(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread + 4))",
            "A fragment 2",
        ),
        (
            "gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread + 4))",
            "gemm_bi_tf32_prepacked_a_word<Op>(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread + 4))",
            "A fragment 3",
        ),
    ] {
        replace_exact(&mut source, from, to, 1, label)?;
    }
    let old = "gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s3";
    replace_exact(&mut source, old, GEMM_SYMBOL, 2, "TN target symbol")?;
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn prepack_adapter_changes_only_tn_a_conversion_and_target_export() {
        let source = candidate_source(PRODUCTION).unwrap();
        assert!(source.contains(PACK_SYMBOL));
        assert!(source.contains(GEMM_SYMBOL));
        assert!(
            !source.contains(
                "GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s3,"
            )
        );
        assert_eq!(
            source.matches("gemm_bi_tf32_prepacked_a_word<Op>").count(),
            4
        );
        assert_eq!(
            source
                .matches("gemm_bi_tf32_rna(gemm_bi_tf32_b_slot<Op>")
                .count(),
            2
        );
        assert!(source.contains("cvt.rna.tf32.f32"));
        assert!(source.contains("__float_as_uint(value)"));
    }

    #[test]
    fn prepack_adapter_rejects_changed_source_boundaries() {
        let error = candidate_source(&PRODUCTION.replacen("cvt.rna.tf32.f32", "cvt.rn.f32.f32", 1))
            .unwrap_err();
        assert!(error.contains("RNA helper"));
    }
}
