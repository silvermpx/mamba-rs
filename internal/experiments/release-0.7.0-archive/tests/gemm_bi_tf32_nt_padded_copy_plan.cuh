// Test-only precomputed global-to-shared copy plan for the padded-stride NT
// discovery arm. The including transformed production source provides the
// storage, copy, compute, and parameter helpers used below.
struct SgbTf32NtTestPaddedCopyPlan {
    long long a_row_base[4];
    long long b_row_base[2];
    unsigned a_destination[4];
    unsigned b_destination[2];
    bool a_row_valid[4];
    bool b_row_valid[2];
    int reduction_offset;
};

__device__ __forceinline__ void gemm_bi_tf32_nt_test_make_padded_copy_plan(
    SgbTf32Storage<SgbTf32Nt, 128, 64, 3>* storage,
    const SgbTf32Problem& problem,
    SgbTf32NtTestPaddedCopyPlan& plan) {
    plan.reduction_offset = ((int)threadIdx.x & 7) * 4;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int row = linear >> 3;
        int global_row = problem.tile_row + row;
        plan.a_row_valid[slice] = global_row < problem.params.m;
        plan.a_row_base[slice] = plan.a_row_valid[slice]
            ? (long long)global_row * problem.params.lda
            : 0;
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(
            &gemm_bi_tf32_a_slot<SgbTf32Nt>(
                storage, 0, row, plan.reduction_offset));
    }
#pragma unroll
    for (int slice = 0; slice < 2; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int column = linear >> 3;
        int global_column = problem.tile_column + column;
        plan.b_row_valid[slice] = global_column < problem.params.k;
        plan.b_row_base[slice] = plan.b_row_valid[slice]
            ? (long long)global_column * problem.params.ldb
            : 0;
        plan.b_destination[slice] = (unsigned)__cvta_generic_to_shared(
            &gemm_bi_tf32_b_slot<SgbTf32Nt>(
                storage, 0, plan.reduction_offset, column));
    }
}

__device__ __forceinline__ void gemm_bi_tf32_nt_test_padded_copy_stage(
    const SgbTf32NtTestPaddedCopyPlan& plan,
    const SgbTf32Problem& problem,
    int stage,
    int reduction_base) {
    int remaining = problem.params.n - reduction_base - plan.reduction_offset;
    remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
    int full_bytes = remaining * 4;
    constexpr unsigned AStageBytes = 128U * 36U * 4U;
    constexpr unsigned BStageBytes = 64U * 36U * 4U;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int bytes = plan.a_row_valid[slice] ? full_bytes : 0;
        long long valid_offset = plan.a_row_base[slice]
            + reduction_base + plan.reduction_offset;
        const float* source = gemm_bi_cp_async_source(
            problem.a, bytes == 0 ? 0 : valid_offset, bytes);
        gemm_bi_tf32_cp_async_zfill<false, 128>(
            plan.a_destination[slice] + (unsigned)stage * AStageBytes,
            source,
            bytes);
    }
#pragma unroll
    for (int slice = 0; slice < 2; ++slice) {
        int bytes = plan.b_row_valid[slice] ? full_bytes : 0;
        long long valid_offset = plan.b_row_base[slice]
            + reduction_base + plan.reduction_offset;
        const float* source = gemm_bi_cp_async_source(
            problem.b, bytes == 0 ? 0 : valid_offset, bytes);
        gemm_bi_tf32_cp_async_zfill<false, 128>(
            plan.b_destination[slice] + (unsigned)stage * BStageBytes,
            source,
            bytes);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int MAtoms, int NAtoms>
__device__ __forceinline__ void gemm_bi_tf32_nt_test_padded_copy_plan_mainloop(
    SgbTf32Storage<SgbTf32Nt, 128, 64, 3>* storage,
    const SgbTf32Problem& problem,
    unsigned tile_count,
    const SgbTf32ThreadPlan& thread_plan,
    float (&accumulators)[MAtoms][NAtoms][4]) {
    SgbTf32NtTestPaddedCopyPlan plan;
    gemm_bi_tf32_nt_test_make_padded_copy_plan(storage, problem, plan);
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        if (tile < tile_count) {
            gemm_bi_tf32_nt_test_padded_copy_stage(
                plan, problem, (int)tile, (int)(tile * 32U));
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        unsigned next = tile + 2;
        if (next < tile_count) {
            gemm_bi_tf32_nt_test_padded_copy_stage(
                plan, problem, (int)(next % 3), (int)(next * 32U));
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
