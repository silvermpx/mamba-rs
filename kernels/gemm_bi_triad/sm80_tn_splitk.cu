// TN split-K family of the portable TF32 module. Composed after sm80.cu for
// every sm80-family target except CC 12.x (whose portable twin stays frozen
// against its TF32 cohort), so it reuses the shared storage, compute stage
// and partial-store helpers and adds only what the TN layout needs: a k-major
// A stage on the full-tile fast path, and a fixup that folds beta into the
// fused output the way the NN family does (there is no bias term for TN).
// Eight partitions on every candidate: the TN cells these serve carry a
// reduction of several thousand rows over a 36-288 tile output grid.

static_assert(sizeof(SgbTf32Storage<SgbTf32Tn, 32, 32, 3>) == 30720, "TN M32N32 s3 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Tn, 32, 32, 4>) == 40960, "TN M32N32 s4 storage");

template <SgbTf32Op Op, int BM, int BN, int Stages,
          bool NarrowA, bool NarrowB>
__device__ __forceinline__ void gemm_bi_tf32_tn_splitk_stage_async(
    SgbTf32Storage<Op, BM, BN, Stages>* storage, int stage,
    const SgbTf32Problem& problem, int reduction_base, bool full_output_tile) {
    constexpr int Threads = 128;
    static_assert(Op == SgbTf32Tn, "the TN split-K stage serves the k-major A layout");
    int reduction_extent = gemm_bi_tf32_reduction<Op>(problem.params);
    bool full_reduction_tile = reduction_extent >= 32
        && reduction_base <= reduction_extent - 32;
    if (!full_output_tile || !full_reduction_tile) {
        gemm_bi_tf32_stage_async<
            Op, BM, BN, Stages, NarrowA, NarrowB>(
            storage, stage, problem, reduction_base);
        return;
    }
    constexpr int RowChunks = BM / 4;
    for (int linear = (int)threadIdx.x;
         linear < 32 * RowChunks;
         linear += Threads) {
        int reduction = linear / RowChunks;
        int row = (linear - reduction * RowChunks) * 4;
        const float* source = problem.a
            + (long long)(reduction_base + reduction) * problem.params.lda
            + problem.tile_row + row;
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            &gemm_bi_tf32_a_slot<Op>(storage, stage, row, reduction));
        gemm_bi_tf32_cp_async_zfill<NarrowA, BM>(destination, source, 16);
    }
    for (int linear = (int)threadIdx.x;
         linear < 32 * (BN / 4);
         linear += Threads) {
        int reduction = linear / (BN / 4);
        int column = (linear - reduction * (BN / 4)) * 4;
        const float* source = problem.b
            + (long long)(reduction_base + reduction) * problem.params.ldb
            + problem.tile_column + column;
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            &gemm_bi_tf32_b_slot<Op>(storage, stage, reduction, column));
        gemm_bi_tf32_cp_async_zfill<NarrowB, BM>(destination, source, 16);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

template <SgbTf32Op Op, int BM, int BN, int Stages, int MAtoms, int NAtoms,
          bool NarrowA, bool NarrowB>
__device__ __forceinline__ void gemm_bi_tf32_tn_splitk_async_mainloop(
    SgbTf32Storage<Op, BM, BN, Stages>* storage,
    const SgbTf32Problem& problem, unsigned tile_begin, unsigned tile_count,
    bool full_output_tile, const SgbTf32ThreadPlan& thread_plan,
    float (&accumulators)[MAtoms][NAtoms][4]) {
#pragma unroll
    for (unsigned tile = 0; tile < Stages - 1; ++tile) {
        if (tile < tile_count) {
            gemm_bi_tf32_tn_splitk_stage_async<
                Op, BM, BN, Stages, NarrowA, NarrowB>(
                storage, static_cast<int>(tile), problem,
                static_cast<int>((tile_begin + tile) * 32U),
                full_output_tile);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(Stages - 2));
        __syncthreads();
        unsigned next = tile + Stages - 1;
        if (next < tile_count) {
            int write_stage = read_stage == 0 ? Stages - 1 : read_stage - 1;
            gemm_bi_tf32_tn_splitk_stage_async<
                Op, BM, BN, Stages, NarrowA, NarrowB>(
                storage, write_stage, problem,
                static_cast<int>((tile_begin + next) * 32U),
                full_output_tile);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
        if (thread_plan.compute) {
            gemm_bi_tf32_compute_stage<
                Op, BM, BN, Stages, MAtoms, NAtoms>(
                storage, read_stage, thread_plan.warp_m, thread_plan.warp_n,
                thread_plan.group, thread_plan.thread, accumulators);
        }
        __syncthreads();
        if (++read_stage == Stages) read_stage = 0;
    }
}

template <SgbTf32Op Op, int BM, int BN, int Stages, int Partitions>
__device__ __forceinline__ void gemm_bi_tf32_tn_splitk_fused_kernel(
    float* output, float* partial, unsigned* counters,
    const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    constexpr int MAtoms = BM == 128 ? 4 : (BM == 64 ? 2 : 1);
    constexpr int NAtoms = BM == 16 ? 1 : (BM == 32 ? 2 : 4);
    static_assert(Op == SgbTf32Tn, "the TN split-K kernel serves the TN layout");
    static_assert(
        (BM == 64 && BN == 64 && (Stages == 2 || Stages == 3) && Partitions == 8)
        || (BM == 32 && BN == 32 && (Stages == 3 || Stages == 4) && Partitions == 8),
        "unsupported TN split-K layout");
    int rows = gemm_bi_tf32_rows<Op>(params);
    int columns = gemm_bi_tf32_columns<Op>(params);
    int reduction = gemm_bi_tf32_reduction<Op>(params);
    assert(bias == nullptr);
    int partition = (int)blockIdx.z;
    unsigned full_tiles = (static_cast<unsigned>(reduction) + 31U) / 32U;
    unsigned tiles_per_partition = (full_tiles + Partitions - 1U) / Partitions;
    unsigned tile_begin = min(static_cast<unsigned>(partition) * tiles_per_partition, full_tiles);
    unsigned tile_end = min(tile_begin + tiles_per_partition, full_tiles);
    SgbTf32Problem problem = {
        partial, a, b, nullptr, params,
        (int)blockIdx.y * BM,
        (int)blockIdx.x * BN,
    };
    bool full_output_tile = problem.tile_row + BM <= rows
        && problem.tile_column + BN <= columns;
    extern __shared__ __align__(16) unsigned char gemm_bi_tf32_shared[];
    auto* storage = reinterpret_cast<SgbTf32Storage<Op, BM, BN, Stages>*>(
        gemm_bi_tf32_shared);

    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    bool compute = BM != 128 || warp < 4;
    int warp_m = BM == 128 ? (warp >> 1) * 64
        : (BM == 64 ? (warp >> 1) * 32
            : (BM == 32 ? (warp >> 1) * 16 : 0));
    int warp_n = BM == 16 ? warp * 8
        : (BM == 32 ? (warp & 1) * 16 : (warp & 1) * 32);
    int group = lane >> 2;
    int thread = lane & 3;
    float accumulators[MAtoms][NAtoms][4] = {};
    unsigned tile_count = tile_end - tile_begin;
    SgbTf32ThreadPlan thread_plan = {compute, warp_m, warp_n, group, thread};

    bool wide_a = gemm_bi_tf32_can_stage_a16(a, params);
    bool wide_b = gemm_bi_tf32_can_stage_b16(b, params);
    if (wide_a && wide_b) {
        gemm_bi_tf32_tn_splitk_async_mainloop<
            Op, BM, BN, Stages, MAtoms, NAtoms, false, false>(
            storage, problem, tile_begin, tile_count,
            full_output_tile, thread_plan, accumulators);
    } else if (gemm_bi_tf32_can_stage_async_4(a, b, params)) {
        if (wide_a) {
            gemm_bi_tf32_tn_splitk_async_mainloop<
                Op, BM, BN, Stages, MAtoms, NAtoms, false, true>(
                storage, problem, tile_begin, tile_count,
                full_output_tile, thread_plan, accumulators);
        } else if (wide_b) {
            gemm_bi_tf32_tn_splitk_async_mainloop<
                Op, BM, BN, Stages, MAtoms, NAtoms, true, false>(
                storage, problem, tile_begin, tile_count,
                full_output_tile, thread_plan, accumulators);
        } else {
            gemm_bi_tf32_tn_splitk_async_mainloop<
                Op, BM, BN, Stages, MAtoms, NAtoms, true, true>(
                storage, problem, tile_begin, tile_count,
                full_output_tile, thread_plan, accumulators);
        }
    } else {
        int stage = 0;
        for (unsigned tile = 0; tile < tile_count; ++tile) {
            gemm_bi_tf32_stage_scalar<Op>(
                storage, stage, problem,
                static_cast<int>((tile_begin + tile) * 32U));
            __syncthreads();
            if (compute) {
                gemm_bi_tf32_compute_stage<
                    Op, BM, BN, Stages, MAtoms, NAtoms>(
                    storage, stage, warp_m, warp_n,
                    group, thread, accumulators);
            }
            __syncthreads();
            if (++stage == Stages) stage = 0;
        }
    }

    if (compute) {
        long long partial_stride = (long long)rows * columns;
        float* partition_output = partial + (long long)partition * partial_stride;
        bool packed_output = full_output_tile && (columns & 1) == 0
            && gemm_bi_is_aligned_8(partition_output);
#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = problem.tile_row + warp_m + m_atom * 16
                        + group + half * 8;
                    int column = problem.tile_column + warp_n + n_atom * 8
                        + 2 * thread;
                    if (packed_output) {
                        float* destination = partition_output
                            + (long long)row * columns + column;
                        gemm_bi_tf32_partial_store_cg_v2(destination, make_float2(
                            accumulators[m_atom][n_atom][2 * half],
                            accumulators[m_atom][n_atom][2 * half + 1]));
                    } else if (row < rows) {
                        float* destination = partition_output
                            + (long long)row * columns + column;
#pragma unroll
                        for (int element = 0; element < 2; ++element) {
                            if (column + element < columns) {
                                gemm_bi_tf32_partial_store_cg(
                                    destination + element,
                                    accumulators[m_atom][n_atom][2 * half + element]);
                            }
                        }
                    }
                }
            }
        }
    }

    __threadfence();
    __syncthreads();
    __shared__ bool last_partition;
    if (threadIdx.x == 0) {
        unsigned tile = (unsigned)blockIdx.y * (unsigned)gridDim.x
            + (unsigned)blockIdx.x;
        last_partition = atomicInc(counters + tile, Partitions - 1U)
            == Partitions - 1U;
    }
    __syncthreads();
    if (!last_partition || !compute) return;

    long long partial_stride = (long long)rows * columns;
#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = problem.tile_row + warp_m + m_atom * 16
                    + group + half * 8;
                int column = problem.tile_column + warp_n + n_atom * 8
                    + 2 * thread;
                if (row >= rows || column >= columns) continue;
                long long index = (long long)row * columns + column;
                float* destination = output + (long long)row * params.ldc + column;
                bool has_second = column + 1 < columns;
                bool packed_partial = has_second && (columns & 1) == 0
                    && gemm_bi_is_aligned_8(partial + index);
                float2 p0;
                float2 p1;
                float2 p2;
                float2 p3;
                float2 p4;
                float2 p5;
                float2 p6;
                float2 p7;
                if (packed_partial) {
                    p0 = gemm_bi_tf32_partial_load_cg_v2(partial + index);
                    p1 = gemm_bi_tf32_partial_load_cg_v2(
                        partial + partial_stride + index);
                    if constexpr (Partitions >= 4) {
                        p2 = gemm_bi_tf32_partial_load_cg_v2(
                            partial + 2 * partial_stride + index);
                        p3 = gemm_bi_tf32_partial_load_cg_v2(
                            partial + 3 * partial_stride + index);
                    }
                    if constexpr (Partitions == 8) {
                        p4 = gemm_bi_tf32_partial_load_cg_v2(
                            partial + 4 * partial_stride + index);
                        p5 = gemm_bi_tf32_partial_load_cg_v2(
                            partial + 5 * partial_stride + index);
                        p6 = gemm_bi_tf32_partial_load_cg_v2(
                            partial + 6 * partial_stride + index);
                        p7 = gemm_bi_tf32_partial_load_cg_v2(
                            partial + 7 * partial_stride + index);
                    }
                } else {
                    p0.x = gemm_bi_tf32_partial_load_cg(partial + index);
                    p1.x = gemm_bi_tf32_partial_load_cg(
                        partial + partial_stride + index);
                    if constexpr (Partitions >= 4) {
                        p2.x = gemm_bi_tf32_partial_load_cg(
                            partial + 2 * partial_stride + index);
                        p3.x = gemm_bi_tf32_partial_load_cg(
                            partial + 3 * partial_stride + index);
                    }
                    if constexpr (Partitions == 8) {
                        p4.x = gemm_bi_tf32_partial_load_cg(
                            partial + 4 * partial_stride + index);
                        p5.x = gemm_bi_tf32_partial_load_cg(
                            partial + 5 * partial_stride + index);
                        p6.x = gemm_bi_tf32_partial_load_cg(
                            partial + 6 * partial_stride + index);
                        p7.x = gemm_bi_tf32_partial_load_cg(
                            partial + 7 * partial_stride + index);
                    }
                    if (has_second) {
                        p0.y = gemm_bi_tf32_partial_load_cg(partial + index + 1);
                        p1.y = gemm_bi_tf32_partial_load_cg(
                            partial + partial_stride + index + 1);
                        if constexpr (Partitions >= 4) {
                            p2.y = gemm_bi_tf32_partial_load_cg(
                                partial + 2 * partial_stride + index + 1);
                            p3.y = gemm_bi_tf32_partial_load_cg(
                                partial + 3 * partial_stride + index + 1);
                        }
                        if constexpr (Partitions == 8) {
                            p4.y = gemm_bi_tf32_partial_load_cg(
                                partial + 4 * partial_stride + index + 1);
                            p5.y = gemm_bi_tf32_partial_load_cg(
                                partial + 5 * partial_stride + index + 1);
                            p6.y = gemm_bi_tf32_partial_load_cg(
                                partial + 6 * partial_stride + index + 1);
                            p7.y = gemm_bi_tf32_partial_load_cg(
                                partial + 7 * partial_stride + index + 1);
                        }
                    }
                }
                float value0;
                if constexpr (Op == SgbTf32Nn) {
                    float bias0 = bias == nullptr ? 0.0f : bias[column];
                    float sum0 = bias0;
                    sum0 = __fadd_rn(sum0, p0.x);
                    sum0 = __fadd_rn(sum0, p1.x);
                    if constexpr (Partitions == 4) {
                        sum0 = __fadd_rn(sum0, p2.x);
                        sum0 = __fadd_rn(sum0, p3.x);
                    }
                    value0 = params.alpha == 1.0f
                        ? sum0 : __fmul_rn(params.alpha, sum0);
                    if (params.beta != 0.0f) {
                        value0 = __fmaf_rn(params.beta, destination[0], value0);
                    }
                } else {
                    float sum0 = __fadd_rn(p0.x, p1.x);
                    sum0 = __fadd_rn(sum0, p2.x);
                    sum0 = __fadd_rn(sum0, p3.x);
                    if constexpr (Partitions == 8) {
                        sum0 = __fadd_rn(sum0, p4.x);
                        sum0 = __fadd_rn(sum0, p5.x);
                        sum0 = __fadd_rn(sum0, p6.x);
                        sum0 = __fadd_rn(sum0, p7.x);
                    }
                    value0 = params.alpha == 1.0f
                        ? sum0 : __fmul_rn(params.alpha, sum0);
                    if (params.beta != 0.0f) {
                        value0 = __fmaf_rn(params.beta, destination[0], value0);
                    }
                }
                if (has_second) {
                    float value1;
                    if constexpr (Op == SgbTf32Nn) {
                        float sum1 = bias == nullptr ? 0.0f : bias[column + 1];
                        sum1 = __fadd_rn(sum1, p0.y);
                        sum1 = __fadd_rn(sum1, p1.y);
                        if constexpr (Partitions == 4) {
                            sum1 = __fadd_rn(sum1, p2.y);
                            sum1 = __fadd_rn(sum1, p3.y);
                        }
                        value1 = params.alpha == 1.0f
                            ? sum1 : __fmul_rn(params.alpha, sum1);
                        if (params.beta != 0.0f) {
                            value1 = __fmaf_rn(params.beta, destination[1], value1);
                        }
                    } else {
                        float sum1 = __fadd_rn(p0.y, p1.y);
                        sum1 = __fadd_rn(sum1, p2.y);
                        sum1 = __fadd_rn(sum1, p3.y);
                        if constexpr (Partitions == 8) {
                            sum1 = __fadd_rn(sum1, p4.y);
                            sum1 = __fadd_rn(sum1, p5.y);
                            sum1 = __fadd_rn(sum1, p6.y);
                            sum1 = __fadd_rn(sum1, p7.y);
                        }
                        value1 = params.alpha == 1.0f
                            ? sum1 : __fmul_rn(params.alpha, sum1);
                        if (params.beta != 0.0f) {
                            value1 = __fmaf_rn(params.beta, destination[1], value1);
                        }
                    }
                    if (gemm_bi_is_aligned_8(destination)) {
                        *reinterpret_cast<float2*>(destination) =
                            make_float2(value0, value1);
                    } else {
                        destination[0] = value0;
                        destination[1] = value1;
                    }
                } else {
                    destination[0] = value0;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 2)
void gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m64n64_bk32_s2(
    float* output, float* partial, unsigned* counters,
    const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    gemm_bi_tf32_tn_splitk_fused_kernel<SgbTf32Tn, 64, 64, 2, 8>(
        output, partial, counters, a, b, bias, params);
}

extern "C" __global__ __launch_bounds__(128, 1)
void gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m64n64_bk32_s3(
    float* output, float* partial, unsigned* counters,
    const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    gemm_bi_tf32_tn_splitk_fused_kernel<SgbTf32Tn, 64, 64, 3, 8>(
        output, partial, counters, a, b, bias, params);
}

extern "C" __global__ __launch_bounds__(128, 3)
void gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s3(
    float* output, float* partial, unsigned* counters,
    const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    gemm_bi_tf32_tn_splitk_fused_kernel<SgbTf32Tn, 32, 32, 3, 8>(
        output, partial, counters, a, b, bias, params);
}

extern "C" __global__ __launch_bounds__(128, 2)
void gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s4(
    float* output, float* partial, unsigned* counters,
    const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    gemm_bi_tf32_tn_splitk_fused_kernel<SgbTf32Tn, 32, 32, 4, 8>(
        output, partial, counters, a, b, bias, params);
}

#define TF32_ASSERT_SPLITK_KERNEL_SIGNATURE(NAME) \
    static_assert(SgbTf32SameType<decltype(&NAME), SgbTf32SplitKKernelSignature>::value, \
        "TF32 split-K kernel signature")
TF32_ASSERT_SPLITK_KERNEL_SIGNATURE(gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m64n64_bk32_s2);
TF32_ASSERT_SPLITK_KERNEL_SIGNATURE(gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m64n64_bk32_s3);
TF32_ASSERT_SPLITK_KERNEL_SIGNATURE(gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s3);
TF32_ASSERT_SPLITK_KERNEL_SIGNATURE(gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s4);
#undef TF32_ASSERT_SPLITK_KERNEL_SIGNATURE
