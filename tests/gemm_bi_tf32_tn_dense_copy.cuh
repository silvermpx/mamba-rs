// Test-only TN full-stage copy specialization. Arithmetic and staging layout
// are supplied by the unchanged production S3 pipeline.
__device__ __forceinline__ bool gemm_bi_tf32_tn_test_dense_full_stage(
    const SgbTf32Problem& problem, int reduction_base) {
    return problem.params.k >= 128 && problem.params.n >= 64
        && problem.params.m >= 32
        && problem.tile_row >= 0 && problem.tile_column >= 0
        && reduction_base >= 0
        && problem.tile_row <= problem.params.k - 128
        && problem.tile_column <= problem.params.n - 64
        && reduction_base <= problem.params.m - 32
        && gemm_bi_is_aligned_16(problem.a)
        && gemm_bi_is_aligned_16(problem.b)
        && (problem.params.lda & 3) == 0
        && (problem.params.ldb & 3) == 0;
}

__device__ __forceinline__ void gemm_bi_tf32_tn_test_dense_stage(
    SgbTf32Storage<SgbTf32Tn, 128, 64, 3>* storage, int stage,
    const SgbTf32Problem& problem, int reduction_base) {
    for (int linear = (int)threadIdx.x; linear < 32 * 32; linear += 256) {
        int reduction = linear >> 5;
        int row = (linear & 31) * 4;
        const float* source = problem.a
            + (long long)(reduction_base + reduction) * problem.params.lda
            + problem.tile_row + row;
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            &gemm_bi_tf32_a_slot<SgbTf32Tn>(storage, stage, row, reduction));
        gemm_bi_tf32_cp_async_16_zfill<128>(destination, source, 16);
    }
    for (int linear = (int)threadIdx.x; linear < 32 * 16; linear += 256) {
        int reduction = linear >> 4;
        int column = (linear & 15) * 4;
        const float* source = problem.b
            + (long long)(reduction_base + reduction) * problem.params.ldb
            + problem.tile_column + column;
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            &gemm_bi_tf32_b_slot<SgbTf32Tn>(storage, stage, reduction, column));
        gemm_bi_tf32_cp_async_16_zfill<128>(destination, source, 16);
    }
}
