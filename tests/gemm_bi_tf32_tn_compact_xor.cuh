__device__ __forceinline__ int gemm_bi_tf32_tn_test_compact_xor_axis(
    int axis, int reduction) {
    return axis ^ ((reduction & 3) << 3);
}
