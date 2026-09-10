// Test-only dense full-tile staging for the padded-stride NT discovery arm.
// The transformed production source provides storage, slot, copy, compute,
// parameter, and thread-plan helpers before this header is inserted.

__device__ __forceinline__ bool gemm_bi_tf32_nt_test_padded_dense_target(
    const Sm80Tf32KernelParams& params) {
    return params.m == 2048 && params.k == 768 && params.n == 3072
        && params.lda == 3072 && params.ldb == 3072 && params.ldc == 768;
}

__device__ __forceinline__ void gemm_bi_tf32_nt_test_padded_dense_stage(
    SgbTf32Storage<SgbTf32Nt, 128, 64, 3>* storage,
    int stage,
    const SgbTf32Problem& problem,
    int reduction_base) {
    constexpr int Threads = 256;
    for (int linear = (int)threadIdx.x; linear < 128 * 8; linear += Threads) {
        int row = linear >> 3;
        int reduction = (linear & 7) * 4;
        const float* source = problem.a
            + (long long)(problem.tile_row + row) * problem.params.lda
            + reduction_base + reduction;
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            &gemm_bi_tf32_a_slot<SgbTf32Nt>(storage, stage, row, reduction));
        gemm_bi_tf32_cp_async_16_zfill<128>(destination, source, 16);
    }
    for (int linear = (int)threadIdx.x; linear < 64 * 8; linear += Threads) {
        int column = linear >> 3;
        int reduction = (linear & 7) * 4;
        const float* source = problem.b
            + (long long)(problem.tile_column + column) * problem.params.ldb
            + reduction_base + reduction;
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            &gemm_bi_tf32_b_slot<SgbTf32Nt>(storage, stage, reduction, column));
        gemm_bi_tf32_cp_async_16_zfill<128>(destination, source, 16);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int MAtoms, int NAtoms>
__device__ __forceinline__ void gemm_bi_tf32_nt_test_padded_dense_mainloop(
    SgbTf32Storage<SgbTf32Nt, 128, 64, 3>* storage,
    const SgbTf32Problem& problem,
    unsigned tile_count,
    const SgbTf32ThreadPlan& thread_plan,
    float (&accumulators)[MAtoms][NAtoms][4]) {
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        if (tile < tile_count) {
            gemm_bi_tf32_nt_test_padded_dense_stage(
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
            gemm_bi_tf32_nt_test_padded_dense_stage(
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
