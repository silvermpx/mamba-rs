//! The accepted compact eight-warp two-stage NT candidate, composed from
//! the immutable SM80 TF32 source by exact anchored replacements: the XOR
//! bank-conflict-free shared layout for both operands, the eight-warp
//! accumulator ownership and the two-stage storage extent. The library's
//! finalist regression composes it here, hands it to the A-only ldmatrix
//! adapter, and compares the result with the production finalist body.

const PRODUCTION_CUDA: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");
const CANDIDATE_CUDA: &str = include_str!("../gemm_bi_tf32_nt_compact_xor.cu");
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
pub fn candidate_source() -> Result<String, String> {
    let mut source = PRODUCTION_CUDA.to_owned();
    replace_exact(
        &mut source,
        concat!(
            "    static constexpr int bk32 = 32;\n",
            "    static constexpr int ARows = Op == SgbTf32Tn ? bk32 : BM;\n",
            "    static constexpr int AStride = Op == SgbTf32Tn ? BM + 8 : 36;\n",
            "    static constexpr int BRows = Op == SgbTf32Nt ? BN : bk32;\n",
            "    static constexpr int BStride = Op == SgbTf32Nt ? 36\n",
            "        : (BN == 64 ? 72 : (BN == 32 ? 40 : 24));"
        ),
        concat!(
            "    static constexpr int bk32 = 32;\n",
            "    static constexpr bool compact_eight_warp_s2 =\n",
            "        Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2;\n",
            "    static constexpr int ARows = Op == SgbTf32Tn ? bk32 : BM;\n",
            "    static constexpr int AStride = Op == SgbTf32Tn ? BM + 8\n",
            "        : (compact_eight_warp_s2 ? 32 : 36);\n",
            "    static constexpr int BRows = Op == SgbTf32Nt ? BN : bk32;\n",
            "    static constexpr int BStride = Op == SgbTf32Nt\n",
            "        ? (compact_eight_warp_s2 ? 32 : 36)\n",
            "        : (BN == 64 ? 72 : (BN == 32 ? 40 : 24));"
        ),
        1,
        "compact eight-warp S2 storage specialization",
    )?;
    replace_exact(
        &mut source,
        concat!(
            "    if constexpr (Op == SgbTf32Tn) {\n",
            "        return storage->a[stage][reduction][row];\n",
            "    }\n",
            "    return storage->a[stage][row][reduction];"
        ),
        concat!(
            "    if constexpr (Op == SgbTf32Tn) {\n",
            "        return storage->a[stage][reduction][row];\n",
            "    }\n",
            "    if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2) {\n",
            "        return storage->a[stage][row][gemm_bi_nt_test_compact_xor_k(row, reduction)];\n",
            "    }\n",
            "    return storage->a[stage][row][reduction];"
        ),
        1,
        "compact eight-warp S2 A slot",
    )?;
    replace_exact(
        &mut source,
        concat!(
            "    if constexpr (Op == SgbTf32Nt) {\n",
            "        return storage->b[stage][column][reduction];\n",
            "    }\n",
            "    return storage->b[stage][reduction][column];"
        ),
        concat!(
            "    if constexpr (Op == SgbTf32Nt) {\n",
            "        if constexpr (BM == 128 && BN == 64 && Stages == 2) {\n",
            "            return storage->b[stage][column][gemm_bi_nt_test_compact_xor_k(column, reduction)];\n",
            "        }\n",
            "        return storage->b[stage][column][reduction];\n",
            "    }\n",
            "    return storage->b[stage][reduction][column];"
        ),
        1,
        "compact eight-warp S2 B slot",
    )?;
    replace_exact(
        &mut source,
        "== 55296, \"NT M128N64 s2 storage\"",
        "== 49152, \"NT compact-eight-warp M128N64 s2 storage\"",
        1,
        "compact eight-warp S2 extent",
    )?;
    replace_exact(
        &mut source,
        concat!(
            "__device__ __forceinline__ void gemm_bi_tf32_kernel(\n",
            "    float* output, const float* a, const float* b, const float* bias,\n",
            "    Sm80Tf32KernelParams params) {\n",
            "    constexpr int MAtoms = BM == 128 ? 4 : (BM == 64 ? 2 : 1);"
        ),
        concat!(
            "__device__ __forceinline__ void gemm_bi_tf32_kernel(\n",
            "    float* output, const float* a, const float* b, const float* bias,\n",
            "    Sm80Tf32KernelParams params) {\n",
            "    constexpr bool compact_eight_warp_s2 =\n",
            "        Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2;\n",
            "    constexpr int MAtoms = compact_eight_warp_s2 ? 2\n",
            "        : (BM == 128 ? 4 : (BM == 64 ? 2 : 1));"
        ),
        1,
        "compact eight-warp S2 accumulator ownership",
    )?;
    replace_exact(
        &mut source,
        concat!(
            "    bool compute = BM != 128 || warp < 4;\n",
            "    int warp_m = BM == 128 ? (warp >> 1) * 64\n",
            "        : (BM == 64 ? (warp >> 1) * 32 : 0);"
        ),
        concat!(
            "    bool compute = compact_eight_warp_s2 || BM != 128 || warp < 4;\n",
            "    int warp_m = compact_eight_warp_s2 ? (warp >> 1) * 32\n",
            "        : (BM == 128 ? (warp >> 1) * 64\n",
            "        : (BM == 64 ? (warp >> 1) * 32 : 0));"
        ),
        1,
        "compact eight-warp S2 compute and row ownership",
    )?;
    replace_exact(
        &mut source,
        concat!(
            "GEMM_BI_TF32_DEFINE_KERNEL(",
            "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s2, ",
            "SgbTf32Nt, 128, 64, 2, 256, 1)"
        ),
        concat!(
            "GEMM_BI_TF32_DEFINE_KERNEL(",
            "gemm_bi_nt_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2, ",
            "SgbTf32Nt, 128, 64, 2, 256, 1)"
        ),
        1,
        "compact eight-warp S2 target symbol",
    )?;
    replace_exact(
        &mut source,
        "TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s2);",
        "TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nt_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2);",
        1,
        "compact eight-warp S2 target signature",
    )?;
    Ok(format!("{CANDIDATE_CUDA}\n{source}"))
}
