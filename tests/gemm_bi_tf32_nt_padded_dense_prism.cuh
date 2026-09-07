// Test-only per-stage bounds dispatcher for the prism dense-copy arm.
// The frozen padded-dense helper and production generic staging are defined
// before this header is inserted into the transformed source.

__device__ __forceinline__ bool gemm_bi_tf32_nt_test_padded_dense_prism_target(
    const Sm80Tf32KernelParams& params) {
    return params.m == 4621 && params.k == 384 && params.n == 1928
        && params.lda == 1928 && params.ldb == 1928 && params.ldc == 384;
}

__device__ __forceinline__ bool gemm_bi_tf32_nt_test_padded_dense_prism_full_stage(
    const SgbTf32Problem& problem,
    int reduction_base) {
    return problem.tile_row + 128 <= problem.params.m
        && problem.tile_column + 64 <= problem.params.k
        && reduction_base + 32 <= problem.params.n;
}

__device__ __forceinline__ void gemm_bi_tf32_nt_test_padded_dense_prism_stage(
    SgbTf32Storage<SgbTf32Nt, 128, 64, 3>* storage,
    int stage,
    const SgbTf32Problem& problem,
    int reduction_base) {
    if (gemm_bi_tf32_nt_test_padded_dense_prism_full_stage(
            problem, reduction_base)) {
        gemm_bi_tf32_nt_test_padded_dense_stage(
            storage, stage, problem, reduction_base);
    } else {
        gemm_bi_tf32_stage_async<SgbTf32Nt, 128, 64, 3, false, false>(
            storage, stage, problem, reduction_base);
    }
}

template <int MAtoms, int NAtoms>
__device__ __forceinline__ void gemm_bi_tf32_nt_test_padded_dense_prism_mainloop(
    SgbTf32Storage<SgbTf32Nt, 128, 64, 3>* storage,
    const SgbTf32Problem& problem,
    unsigned tile_count,
    const SgbTf32ThreadPlan& thread_plan,
    float (&accumulators)[MAtoms][NAtoms][4]) {
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        if (tile < tile_count) {
            gemm_bi_tf32_nt_test_padded_dense_prism_stage(
                storage, (int)tile, problem, (int)(tile * 32U));
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        unsigned next = tile + 2;
        if (next < tile_count) {
            gemm_bi_tf32_nt_test_padded_dense_prism_stage(
                storage, (int)(next % 3), problem, (int)(next * 32U));
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
        if (thread_plan.compute) {
            gemm_bi_tf32_compute_stage<
                SgbTf32Nt, 128, 64, 3, MAtoms, NAtoms>(
                storage,
                (int)(tile % 3),
                thread_plan.warp_m,
                thread_plan.warp_n,
                thread_plan.group,
                thread_plan.thread,
                accumulators);
        }
        __syncthreads();
    }
}
