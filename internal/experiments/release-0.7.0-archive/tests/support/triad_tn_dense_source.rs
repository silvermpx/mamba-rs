pub const SYMBOL: &str = "gemm_bi_tn_test_dense_sm80_mma_tf32_v1_m128n64_bk32_s3";
const HELPER: &str = include_str!("../gemm_bi_tf32_tn_dense_copy.cuh");

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = production.to_owned();
    const STAGE: &str = concat!(
        "template <SgbTf32Op Op, int BM, int BN, int Stages,\n",
        "          bool NarrowA, bool NarrowB>\n",
        "__device__ __forceinline__ void gemm_bi_tf32_stage_async("
    );
    replace_exact(
        &mut source,
        STAGE,
        &format!("{HELPER}\n{STAGE}"),
        "helper insertion",
    )?;
    const BODY: &str = concat!(
        "    int reduction_extent = gemm_bi_tf32_reduction<Op>(problem.params);\n\n",
        "    if constexpr (Op == SgbTf32Tn) {\n",
        "        constexpr int RowChunks = BM / 4;"
    );
    const DENSE_BODY: &str = concat!(
        "    int reduction_extent = gemm_bi_tf32_reduction<Op>(problem.params);\n\n",
        "    if constexpr (Op == SgbTf32Tn && BM == 128 && BN == 64\n",
        "                  && Stages == 3 && !NarrowA && !NarrowB) {\n",
        "        if (gemm_bi_tf32_tn_test_dense_full_stage(problem, reduction_base)) {\n",
        "            gemm_bi_tf32_tn_test_dense_stage(storage, stage, problem, reduction_base);\n",
        "            asm volatile(\"cp.async.commit_group;\\n\" ::);\n",
        "            return;\n",
        "        }\n",
        "    }\n\n",
        "    if constexpr (Op == SgbTf32Tn) {\n",
        "        constexpr int RowChunks = BM / 4;"
    );
    replace_exact(&mut source, BODY, DENSE_BODY, "TN dense stage branch")?;
    let old = "gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s3";
    replace_exact(
        &mut source,
        &format!("GEMM_BI_TF32_DEFINE_KERNEL({old}, SgbTf32Tn, 128, 64, 3, 256, 1)"),
        &format!("GEMM_BI_TF32_DEFINE_KERNEL({SYMBOL}, SgbTf32Tn, 128, 64, 3, 256, 1)"),
        "target export",
    )?;
    replace_exact(
        &mut source,
        &format!("TF32_ASSERT_KERNEL_SIGNATURE({old});"),
        &format!("TF32_ASSERT_KERNEL_SIGNATURE({SYMBOL});"),
        "target signature",
    )?;
    Ok(source)
}

fn replace_exact(source: &mut String, from: &str, to: &str, label: &str) -> Result<(), String> {
    let count = source.matches(from).count();
    if count != 1 {
        return Err(format!("{label}: expected one boundary, observed {count}"));
    }
    *source = source.replacen(from, to, 1);
    Ok(())
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;
    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");
    const EXPORT: &str = "GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s3, SgbTf32Tn, 128, 64, 3, 256, 1)";

    #[test]
    fn dense_composer_binds_a_distinct_export_and_preserves_compute() {
        let source = candidate_source(PRODUCTION).unwrap();
        assert!(source.contains(&format!(
            "GEMM_BI_TF32_DEFINE_KERNEL({SYMBOL}, SgbTf32Tn, 128, 64, 3, 256, 1)"
        )));
        let start = "template <SgbTf32Op Op, int BM, int BN, int Stages, int MAtoms, int NAtoms>\n__device__ __forceinline__ void gemm_bi_tf32_compute_stage(";
        let original_compute = PRODUCTION.split_once(start).unwrap().1;
        let candidate_compute = source.split_once(start).unwrap().1;
        let end = "#define GEMM_BI_TF32_DEFINE_KERNEL";
        assert_eq!(
            original_compute.split_once(end).unwrap().0,
            candidate_compute.split_once(end).unwrap().0
        );
    }

    #[test]
    fn dense_composer_rejects_missing_or_duplicate_export() {
        assert!(candidate_source(&PRODUCTION.replacen(EXPORT, "", 1)).is_err());
        assert!(candidate_source(&format!("{PRODUCTION}\n{EXPORT}")).is_err());
    }
}
