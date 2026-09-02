// Test-only SM120 TF32 pair-store experiment. Production registration is forbidden.
#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)

struct Sm120Tf32Pair {
    float first;
    float second;
};

template <int Op>
static __device__ __forceinline__ void sm120_tf32_pair_store_exp_v1(
    void* output, int row, int column, Sm120Tf32Pair accumulator,
    const float* bias, const Sm120KernelParams& params, bool full_tile) {
    if (full_tile) {
        float* destination = static_cast<float*>(output)
            + static_cast<long long>(row) * params.ldc + column;
        Sm120Tf32Pair old_output = {0.0f, 0.0f};
        if constexpr (Op == Sm120Tn) {
            float2 old_pair = *reinterpret_cast<const float2*>(destination);
            old_output.first = old_pair.x;
            old_output.second = old_pair.y;
        }
        float first = sm120_tf32_epilogue<Op>(
            accumulator.first, old_output.first, bias, column, params);
        float second = sm120_tf32_epilogue<Op>(
            accumulator.second, old_output.second, bias, column + 1, params);
        *reinterpret_cast<float2*>(destination) = make_float2(first, second);
        return;
    }
    sm120_tf32_store<Op>(
        output, row, column, accumulator.first, bias, params);
    sm120_tf32_store<Op>(
        output, row, column + 1, accumulator.second, bias, params);
}

template <int Op, int M, int N, int Stages, bool ProducerWarp = false>
static __device__ __forceinline__ void sm120_tf32_pair_kernel_exp_v1(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    assert(params.alpha == 1.0f || bias == nullptr);
    extern __shared__ __align__(1024) unsigned char storage[];
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    constexpr int compute_warps = Sm120Tf32Storage<M, N, Stages>::threads / 32;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    unsigned payload = shared;
    unsigned full_base = shared + Stages * stage_bytes;
    unsigned empty_base = full_base + 64;
    int rows = sm120_tf32_rows<Op>(params);
    int columns = sm120_tf32_columns<Op>(params);
    int column_tiles = 1 + (columns - 1) / N;
    int output_row = (int)blockIdx.x / column_tiles * M;
    int output_column = (int)blockIdx.x % column_tiles * N;
    int reduction = sm120_tf32_reduction<Op>(params);
    int tile_count = 1 + (reduction - 1) / 32;
    const Sm120Tf32StageContext stage_context = {
        &a_map, &b_map, &params, payload, full_base, output_row, output_column};
    int warp = (int)threadIdx.x >> 5;
    int compute_warp = ProducerWarp ? warp - 1 : warp;
    constexpr int warp_columns = N / 32;
    int warp_m = (compute_warp / warp_columns) * 32;
    int warp_n = (compute_warp % warp_columns) * 32;
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
    float accumulator[2][4][4];

    if (!ProducerWarp || warp != 0) {
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
                for (int element = 0; element < 4; ++element) {
                    accumulator[m_atom][n_atom][element] = 0.0f;
                }
            }
        }
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            sm120_init_barrier<1>(full_base + stage * 8);
            sm120_init_barrier<compute_warps>(empty_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if constexpr (ProducerWarp) {
        if (warp == 0) {
            if (lane == 0) {
#pragma unroll
                for (int tile = 0; tile < Stages; ++tile) {
                    if (tile < tile_count) {
                        sm120_tf32_produce_stage<Op, M, N, Stages>(
                            stage_context, tile);
                    }
                }
                for (int refill = Stages; refill < tile_count; ++refill) {
                    int consumed = refill - Stages;
                    int stage = consumed % Stages;
                    unsigned generation = (unsigned)(consumed / Stages);
                    sm120_wait_barrier(
                        empty_base + stage * 8, generation & 1U);
                    sm120_tf32_produce_stage<Op, M, N, Stages>(
                        stage_context, refill);
                }
            }
            return;
        }
    } else {
        if (warp == 0 && lane == 0) {
#pragma unroll
            for (int tile = 0; tile < Stages; ++tile) {
                if (tile < tile_count) {
                    sm120_tf32_produce_stage<Op, M, N, Stages>(
                        stage_context, tile);
                }
            }
        }
        if constexpr (Stages == 2) sm120_sync_warp();
    }

    for (int tile = 0; tile < tile_count; ++tile) {
        int stage = tile % Stages;
        unsigned generation = (unsigned)(tile / Stages);
        sm120_wait_barrier(full_base + stage * 8, generation & 1U);
        sm120_tf32_issue_stage<Op, M, N, Stages>(
            storage, stage, warp_m, warp_n, accumulator);
        if (lane == 0) {
            sm120_arrive_empty(empty_base + stage * 8);
        }
        if constexpr (!ProducerWarp) {
            if (warp == 0 && lane == 0) {
                int refill = tile + Stages;
                if (refill < tile_count) {
                    sm120_wait_barrier(empty_base + stage * 8, generation & 1U);
                    sm120_tf32_produce_stage<Op, M, N, Stages>(
                        stage_context, refill);
                }
            }
            if constexpr (Stages == 2) sm120_sync_warp();
        }
    }

    bool full_tile = output_row + M <= rows
        && output_column + N <= columns
        && (reinterpret_cast<unsigned long long>(output) & 7ULL) == 0ULL
        && (params.ldc & 1) == 0;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; element += 2) {
                int row = output_row + warp_m + m_atom * 16
                    + group + (element >= 2 ? 8 : 0);
                int column = output_column + warp_n + n_atom * 8
                    + 2 * thread;
                Sm120Tf32Pair pair = {
                    accumulator[m_atom][n_atom][element],
                    accumulator[m_atom][n_atom][element + 1]};
                sm120_tf32_pair_store_exp_v1<Op>(
                    output, row, column, pair, bias, params, full_tile);
            }
        }
    }
}

template <int Op, int M, int N, int Stages, bool ProducerWarp = false>
static __device__ __forceinline__ void sm120_tf32_pair_entry_exp_v1(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    if (sm120_tf32_reduction<Op>(params) == 0) {
        sm120_tf32_zero_reduction_epilogue<Op, M, N>(output, bias, params);
        return;
    }
    sm120_tf32_pair_kernel_exp_v1<Op, M, N, Stages, ProducerWarp>(
        output, a_map, b_map, bias, params);
}

template <int Op, int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_load_issue_w16_exp_v1(
    unsigned char* storage, int stage, int warp_m, int warp_n, int k8,
    unsigned (&a_fragments)[4], unsigned (&b_fragments)[4][2]) {
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
    int row = warp_m + group;
    a_fragments[0] = sm120_tf32_rna(
        sm120_tf32_load_a<Op, M, N, Stages>(storage, stage, row, k8 + thread));
    a_fragments[1] = sm120_tf32_rna(
        sm120_tf32_load_a<Op, M, N, Stages>(storage, stage, row + 8, k8 + thread));
    a_fragments[2] = sm120_tf32_rna(
        sm120_tf32_load_a<Op, M, N, Stages>(storage, stage, row, k8 + thread + 4));
    a_fragments[3] = sm120_tf32_rna(
        sm120_tf32_load_a<Op, M, N, Stages>(storage, stage, row + 8, k8 + thread + 4));
#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        b_fragments[n_atom][0] = sm120_tf32_rna(
            sm120_tf32_load_b<Op, M, N, Stages>(storage, stage, k8 + thread, column));
        b_fragments[n_atom][1] = sm120_tf32_rna(
            sm120_tf32_load_b<Op, M, N, Stages>(storage, stage, k8 + thread + 4, column));
    }
}

template <int Op, int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_issue_stage_w16_exp_v1(
    unsigned char* storage, int stage, int warp_m, int warp_n,
    float (&accumulator)[4][4]) {
    unsigned a_fragments[2][4];
    unsigned b_fragments[2][4][2];
    sm120_tf32_load_issue_w16_exp_v1<Op, M, N, Stages>(
        storage, stage, warp_m, warp_n, 0, a_fragments[0], b_fragments[0]);
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        int current = issue & 1;
        int next = current ^ 1;
        if (issue + 1 < 4) {
            sm120_tf32_load_issue_w16_exp_v1<Op, M, N, Stages>(storage, stage,
                warp_m, warp_n, issue * 8 + 8,
                a_fragments[next], b_fragments[next]);
        }
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
            sm120_tf32_mma_m16n8k8(
                accumulator[n_atom], a_fragments[current], b_fragments[current][n_atom]);
        }
    }
}

template <bool PairStore>
static __device__ __forceinline__ void sm120_tf32_tn_w16_kernel_exp_v1(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int Stages = 2;
    constexpr int warps = 8;
    assert(params.alpha == 1.0f || bias == nullptr);
    extern __shared__ __align__(1024) unsigned char storage[];
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    unsigned payload = shared;
    unsigned full_base = shared + Stages * stage_bytes;
    unsigned empty_base = full_base + 64;
    int rows = sm120_tf32_rows<Sm120Tn>(params);
    int columns = sm120_tf32_columns<Sm120Tn>(params);
    int column_tiles = 1 + (columns - 1) / N;
    int output_row = (int)blockIdx.x / column_tiles * M;
    int output_column = (int)blockIdx.x % column_tiles * N;
    int reduction = sm120_tf32_reduction<Sm120Tn>(params);
    int tile_count = 1 + (reduction - 1) / 32;
    const Sm120Tf32StageContext stage_context = {
        &a_map, &b_map, &params, payload, full_base, output_row, output_column};
    int warp = (int)threadIdx.x >> 5;
    int warp_m = (warp >> 1) * 16;
    int warp_n = (warp & 1) * 32;
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
    float accumulator[4][4];

#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
        for (int element = 0; element < 4; ++element) {
            accumulator[n_atom][element] = 0.0f;
        }
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            sm120_init_barrier<1>(full_base + stage * 8);
            sm120_init_barrier<warps>(empty_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp == 0 && lane == 0) {
#pragma unroll
        for (int tile = 0; tile < Stages; ++tile) {
            if (tile < tile_count) {
                sm120_tf32_produce_stage<Sm120Tn, M, N, Stages>(stage_context, tile);
            }
        }
    }
    sm120_sync_warp();

    for (int tile = 0; tile < tile_count; ++tile) {
        int stage = tile % Stages;
        unsigned generation = (unsigned)(tile / Stages);
        sm120_wait_barrier(full_base + stage * 8, generation & 1U);
        sm120_tf32_issue_stage_w16_exp_v1<Sm120Tn, M, N, Stages>(
            storage, stage, warp_m, warp_n, accumulator);
        if (lane == 0) {
            sm120_arrive_empty(empty_base + stage * 8);
        }
        if (warp == 0 && lane == 0) {
            int refill = tile + Stages;
            if (refill < tile_count) {
                sm120_wait_barrier(empty_base + stage * 8, generation & 1U);
                sm120_tf32_produce_stage<Sm120Tn, M, N, Stages>(
                    stage_context, refill);
            }
        }
        sm120_sync_warp();
    }

    bool full_tile = output_row + M <= rows
        && output_column + N <= columns
        && (reinterpret_cast<unsigned long long>(output) & 7ULL) == 0ULL
        && (params.ldc & 1) == 0;
#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        if constexpr (PairStore) {
#pragma unroll
            for (int element = 0; element < 4; element += 2) {
                int row = output_row + warp_m + group + (element >= 2 ? 8 : 0);
                int column = output_column + warp_n + n_atom * 8 + 2 * thread;
                Sm120Tf32Pair pair = {
                    accumulator[n_atom][element], accumulator[n_atom][element + 1]};
                sm120_tf32_pair_store_exp_v1<Sm120Tn>(
                    output, row, column, pair, bias, params, full_tile);
            }
        } else {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = output_row + warp_m + group + (element >= 2 ? 8 : 0);
                int column = output_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                sm120_tf32_store<Sm120Tn>(output, row, column,
                    accumulator[n_atom][element], bias, params);
            }
        }
    }
}

template <bool PairStore>
static __device__ __forceinline__ void sm120_tf32_tn_w16_entry_exp_v1(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    if (sm120_tf32_reduction<Sm120Tn>(params) == 0) {
        sm120_tf32_zero_reduction_epilogue<Sm120Tn, 64, 64>(
            output, bias, params);
        return;
    }
    sm120_tf32_tn_w16_kernel_exp_v1<PairStore>(
        output, a_map, b_map, bias, params);
}

static __device__ __forceinline__ void sm120_tf32_tma_copy_3d_exp_v1(
    unsigned destination, unsigned long long map, int x, int y, int z,
    unsigned barrier) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(destination), "l"(map), "r"(x), "r"(y), "r"(z), "r"(barrier)
        : "memory");
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_produce_stage_a3d_exp_v1(
    const Sm120Tf32StageContext& context, int tile) {
    constexpr int plane_bytes = 4096;
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned stage_index = (unsigned)(tile % Stages);
    unsigned stage = context.payload + stage_index * stage_bytes;
    unsigned barrier = context.full_base + stage_index * 8;
    unsigned a_destination = stage;
    unsigned b_destination = stage + M * 32 * 4;
    unsigned long long a_descriptor =
        reinterpret_cast<unsigned long long>(context.a_map);
    unsigned long long b_descriptor =
        reinterpret_cast<unsigned long long>(context.b_map);
    const Sm120KernelParams& params = *context.params;
    int reduction = tile * 32;
    sm120_expect_transaction<stage_bytes>(barrier);
    sm120_tf32_tma_copy_3d_exp_v1(a_destination, a_descriptor,
        context.output_row & 31, context.output_row >> 5, reduction, barrier);
#pragma unroll
    for (int plane = 0; plane < N / 32; ++plane) {
        sm120_tma_copy(b_destination + plane * plane_bytes, b_descriptor,
            context.output_column + plane * 32, reduction,
            params.b_x, params.b_y, barrier);
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ float sm120_tf32_tn_load_a3d_exp_v1(
    unsigned char* storage, int stage, int row, int reduction) {
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned stage_base = (unsigned)(stage * stage_bytes);
    unsigned logical_row = (unsigned)(reduction * (M / 32) + row / 32);
    unsigned element = (unsigned)(row & 31);
    unsigned offset = sm120_tf32_sw128_offset(stage_base, logical_row, element);
    return *reinterpret_cast<float*>(storage + offset);
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_load_issue_a3d_exp_v1(
    unsigned char* storage, int stage, int warp_m, int warp_n, int k8,
    unsigned (&a_fragments)[2][4], unsigned (&b_fragments)[4][2]) {
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
        int row = warp_m + m_atom * 16 + group;
        a_fragments[m_atom][0] = sm120_tf32_rna(
            sm120_tf32_tn_load_a3d_exp_v1<M, N, Stages>(
                storage, stage, row, k8 + thread));
        a_fragments[m_atom][1] = sm120_tf32_rna(
            sm120_tf32_tn_load_a3d_exp_v1<M, N, Stages>(
                storage, stage, row + 8, k8 + thread));
        a_fragments[m_atom][2] = sm120_tf32_rna(
            sm120_tf32_tn_load_a3d_exp_v1<M, N, Stages>(
                storage, stage, row, k8 + thread + 4));
        a_fragments[m_atom][3] = sm120_tf32_rna(
            sm120_tf32_tn_load_a3d_exp_v1<M, N, Stages>(
                storage, stage, row + 8, k8 + thread + 4));
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        b_fragments[n_atom][0] = sm120_tf32_rna(
            sm120_tf32_load_b<Sm120Tn, M, N, Stages>(
                storage, stage, k8 + thread, column));
        b_fragments[n_atom][1] = sm120_tf32_rna(
            sm120_tf32_load_b<Sm120Tn, M, N, Stages>(
                storage, stage, k8 + thread + 4, column));
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_issue_stage_a3d_exp_v1(
    unsigned char* storage, int stage, int warp_m, int warp_n,
    float (&accumulator)[2][4][4]) {
    unsigned a_fragments[2][2][4];
    unsigned b_fragments[2][4][2];
    sm120_tf32_tn_load_issue_a3d_exp_v1<M, N, Stages>(
        storage, stage, warp_m, warp_n, 0, a_fragments[0], b_fragments[0]);
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        int current = issue & 1;
        int next = current ^ 1;
        if (issue + 1 < 4) {
            sm120_tf32_tn_load_issue_a3d_exp_v1<M, N, Stages>(storage, stage,
                warp_m, warp_n, issue * 8 + 8,
                a_fragments[next], b_fragments[next]);
        }
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 4; ++n_atom) {
                sm120_tf32_mma_m16n8k8(accumulator[m_atom][n_atom],
                    a_fragments[current][m_atom], b_fragments[current][n_atom]);
            }
        }
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_kernel_a3d_exp_v1(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    constexpr int warps = Sm120Tf32Storage<M, N, Stages>::threads / 32;
    assert(params.alpha == 1.0f || bias == nullptr);
    extern __shared__ __align__(1024) unsigned char storage[];
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    unsigned payload = shared;
    unsigned full_base = shared + Stages * stage_bytes;
    unsigned empty_base = full_base + 64;
    int columns = sm120_tf32_columns<Sm120Tn>(params);
    int column_tiles = 1 + (columns - 1) / N;
    int output_row = (int)blockIdx.x / column_tiles * M;
    int output_column = (int)blockIdx.x % column_tiles * N;
    int tile_count = 1 + (sm120_tf32_reduction<Sm120Tn>(params) - 1) / 32;
    const Sm120Tf32StageContext stage_context = {
        &a_map, &b_map, &params, payload, full_base, output_row, output_column};
    int warp = (int)threadIdx.x >> 5;
    constexpr int warp_columns = N / 32;
    int warp_m = (warp / warp_columns) * 32;
    int warp_n = (warp % warp_columns) * 32;
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
    float accumulator[2][4][4] = {};

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            sm120_init_barrier<1>(full_base + stage * 8);
            sm120_init_barrier<warps>(empty_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp == 0 && lane == 0) {
#pragma unroll
        for (int tile = 0; tile < Stages; ++tile) {
            if (tile < tile_count) {
                sm120_tf32_tn_produce_stage_a3d_exp_v1<M, N, Stages>(
                    stage_context, tile);
            }
        }
    }
    if constexpr (Stages == 2) sm120_sync_warp();

    for (int tile = 0; tile < tile_count; ++tile) {
        int stage = tile % Stages;
        unsigned generation = (unsigned)(tile / Stages);
        sm120_wait_barrier(full_base + stage * 8, generation & 1U);
        sm120_tf32_tn_issue_stage_a3d_exp_v1<M, N, Stages>(
            storage, stage, warp_m, warp_n, accumulator);
        if (lane == 0) sm120_arrive_empty(empty_base + stage * 8);
        if (warp == 0 && lane == 0) {
            int refill = tile + Stages;
            if (refill < tile_count) {
                sm120_wait_barrier(empty_base + stage * 8, generation & 1U);
                sm120_tf32_tn_produce_stage_a3d_exp_v1<M, N, Stages>(
                    stage_context, refill);
            }
        }
        if constexpr (Stages == 2) sm120_sync_warp();
    }

#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = output_row + warp_m + m_atom * 16
                    + group + (element >= 2 ? 8 : 0);
                int column = output_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                sm120_tf32_store<Sm120Tn>(output, row, column,
                    accumulator[m_atom][n_atom][element], bias, params);
            }
        }
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_entry_a3d_exp_v1(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    if (sm120_tf32_reduction<Sm120Tn>(params) == 0) {
        sm120_tf32_zero_reduction_epilogue<Sm120Tn, M, N>(
            output, bias, params);
        return;
    }
    sm120_tf32_tn_kernel_a3d_exp_v1<M, N, Stages>(
        output, a_map, b_map, bias, params);
}

template <int M, int N, int Stages>
struct Sm120Tf32Bk64StorageExpV1 {
    static constexpr int threads = (M / 32) * (N / 32) * 32;
    static constexpr int stage_bytes = (M + N) * 64 * 4;
    static constexpr int dynamic_bytes = 128 + Stages * stage_bytes;
};

static_assert(Sm120Tf32Bk64StorageExpV1<64, 128, 2>::dynamic_bytes == 98432);

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_produce_stage_bk64_exp_v1(
    const Sm120Tf32StageContext& context, int tile) {
    constexpr int plane_bytes = 8192;
    constexpr int stage_bytes = Sm120Tf32Bk64StorageExpV1<M, N, Stages>::stage_bytes;
    unsigned stage_index = (unsigned)(tile % Stages);
    unsigned stage = context.payload + stage_index * stage_bytes;
    unsigned barrier = context.full_base + stage_index * 8;
    unsigned a_destination = stage;
    unsigned b_destination = stage + M * 64 * 4;
    unsigned long long a_descriptor =
        reinterpret_cast<unsigned long long>(context.a_map);
    unsigned long long b_descriptor =
        reinterpret_cast<unsigned long long>(context.b_map);
    const Sm120KernelParams& params = *context.params;
    int reduction = tile * 64;
    sm120_expect_transaction<stage_bytes>(barrier);
#pragma unroll
    for (int plane = 0; plane < M / 32; ++plane) {
        sm120_tma_copy(a_destination + plane * plane_bytes, a_descriptor,
            context.output_row + plane * 32, reduction,
            params.a_x, params.a_y, barrier);
    }
#pragma unroll
    for (int plane = 0; plane < N / 32; ++plane) {
        sm120_tma_copy(b_destination + plane * plane_bytes, b_descriptor,
            context.output_column + plane * 32, reduction,
            params.b_x, params.b_y, barrier);
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ float sm120_tf32_tn_load_a_bk64_exp_v1(
    unsigned char* storage, int stage, int row, int reduction) {
    constexpr int stage_bytes = Sm120Tf32Bk64StorageExpV1<M, N, Stages>::stage_bytes;
    unsigned plane = (unsigned)(row / 32) * 8192U;
    unsigned logical_row = (unsigned)reduction;
    unsigned element = (unsigned)(row & 31);
    unsigned offset = sm120_tf32_sw128_offset(plane, logical_row, element);
    return *reinterpret_cast<float*>(storage + stage * stage_bytes + offset);
}

template <int M, int N, int Stages>
static __device__ __forceinline__ float sm120_tf32_tn_load_b_bk64_exp_v1(
    unsigned char* storage, int stage, int reduction, int column) {
    constexpr int stage_bytes = Sm120Tf32Bk64StorageExpV1<M, N, Stages>::stage_bytes;
    unsigned base = M * 64 * 4;
    unsigned plane = base + (unsigned)(column / 32) * 8192U;
    unsigned logical_row = (unsigned)reduction;
    unsigned element = (unsigned)(column & 31);
    unsigned offset = sm120_tf32_sw128_offset(plane, logical_row, element);
    return *reinterpret_cast<float*>(storage + stage * stage_bytes + offset);
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_load_issue_bk64_exp_v1(
    unsigned char* storage, int stage, int warp_m, int warp_n, int k8,
    unsigned (&a_fragments)[2][4], unsigned (&b_fragments)[4][2]) {
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
        int row = warp_m + m_atom * 16 + group;
        a_fragments[m_atom][0] = sm120_tf32_rna(
            sm120_tf32_tn_load_a_bk64_exp_v1<M, N, Stages>(
                storage, stage, row, k8 + thread));
        a_fragments[m_atom][1] = sm120_tf32_rna(
            sm120_tf32_tn_load_a_bk64_exp_v1<M, N, Stages>(
                storage, stage, row + 8, k8 + thread));
        a_fragments[m_atom][2] = sm120_tf32_rna(
            sm120_tf32_tn_load_a_bk64_exp_v1<M, N, Stages>(
                storage, stage, row, k8 + thread + 4));
        a_fragments[m_atom][3] = sm120_tf32_rna(
            sm120_tf32_tn_load_a_bk64_exp_v1<M, N, Stages>(
                storage, stage, row + 8, k8 + thread + 4));
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        b_fragments[n_atom][0] = sm120_tf32_rna(
            sm120_tf32_tn_load_b_bk64_exp_v1<M, N, Stages>(
                storage, stage, k8 + thread, column));
        b_fragments[n_atom][1] = sm120_tf32_rna(
            sm120_tf32_tn_load_b_bk64_exp_v1<M, N, Stages>(
                storage, stage, k8 + thread + 4, column));
    }
}

template <int Issues, int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_issue_stage_bk64_exp_v1(
    unsigned char* storage, int stage, int warp_m, int warp_n,
    float (&accumulator)[2][4][4]) {
    unsigned a_fragments[2][2][4];
    unsigned b_fragments[2][4][2];
    sm120_tf32_tn_load_issue_bk64_exp_v1<M, N, Stages>(
        storage, stage, warp_m, warp_n, 0, a_fragments[0], b_fragments[0]);
#pragma unroll
    for (int issue = 0; issue < Issues; ++issue) {
        int current = issue & 1;
        int next = current ^ 1;
        if (issue + 1 < Issues) {
            sm120_tf32_tn_load_issue_bk64_exp_v1<M, N, Stages>(storage, stage,
                warp_m, warp_n, issue * 8 + 8,
                a_fragments[next], b_fragments[next]);
        }
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 4; ++n_atom) {
                sm120_tf32_mma_m16n8k8(accumulator[m_atom][n_atom],
                    a_fragments[current][m_atom], b_fragments[current][n_atom]);
            }
        }
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_kernel_bk64_exp_v1(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    constexpr int warps = Sm120Tf32Bk64StorageExpV1<M, N, Stages>::threads / 32;
    assert(params.alpha == 1.0f || bias == nullptr);
    extern __shared__ __align__(1024) unsigned char storage[];
    constexpr int stage_bytes = Sm120Tf32Bk64StorageExpV1<M, N, Stages>::stage_bytes;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    unsigned payload = shared;
    unsigned full_base = shared + Stages * stage_bytes;
    unsigned empty_base = full_base + 64;
    int rows = sm120_tf32_rows<Sm120Tn>(params);
    int columns = sm120_tf32_columns<Sm120Tn>(params);
    int column_tiles = 1 + (columns - 1) / N;
    int output_row = (int)blockIdx.x / column_tiles * M;
    int output_column = (int)blockIdx.x % column_tiles * N;
    int reduction = sm120_tf32_reduction<Sm120Tn>(params);
    int bk32_tiles = 1 + (reduction - 1) / 32;
    int tile_count = 1 + (reduction - 1) / 64;
    const Sm120Tf32StageContext stage_context = {
        &a_map, &b_map, &params, payload, full_base, output_row, output_column};
    int warp = (int)threadIdx.x >> 5;
    constexpr int warp_columns = N / 32;
    int warp_m = (warp / warp_columns) * 32;
    int warp_n = (warp % warp_columns) * 32;
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
    float accumulator[2][4][4] = {};

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            sm120_init_barrier<1>(full_base + stage * 8);
            sm120_init_barrier<warps>(empty_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp == 0 && lane == 0) {
#pragma unroll
        for (int tile = 0; tile < Stages; ++tile) {
            if (tile < tile_count) {
                sm120_tf32_tn_produce_stage_bk64_exp_v1<M, N, Stages>(
                    stage_context, tile);
            }
        }
    }
    if constexpr (Stages == 2) sm120_sync_warp();

    for (int tile = 0; tile < tile_count; ++tile) {
        int stage = tile % Stages;
        unsigned generation = (unsigned)(tile / Stages);
        sm120_wait_barrier(full_base + stage * 8, generation & 1U);
        if (tile * 2 + 1 < bk32_tiles) {
            sm120_tf32_tn_issue_stage_bk64_exp_v1<8, M, N, Stages>(
                storage, stage, warp_m, warp_n, accumulator);
        } else {
            sm120_tf32_tn_issue_stage_bk64_exp_v1<4, M, N, Stages>(
                storage, stage, warp_m, warp_n, accumulator);
        }
        if (lane == 0) sm120_arrive_empty(empty_base + stage * 8);
        if (warp == 0 && lane == 0) {
            int refill = tile + Stages;
            if (refill < tile_count) {
                sm120_wait_barrier(empty_base + stage * 8, generation & 1U);
                sm120_tf32_tn_produce_stage_bk64_exp_v1<M, N, Stages>(
                    stage_context, refill);
            }
        }
        if constexpr (Stages == 2) sm120_sync_warp();
    }

    bool full_tile = output_row + M <= rows
        && output_column + N <= columns
        && (reinterpret_cast<unsigned long long>(output) & 7ULL) == 0ULL
        && (params.ldc & 1) == 0;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; element += 2) {
                int row = output_row + warp_m + m_atom * 16
                    + group + (element >= 2 ? 8 : 0);
                int column = output_column + warp_n + n_atom * 8
                    + 2 * thread;
                Sm120Tf32Pair pair = {
                    accumulator[m_atom][n_atom][element],
                    accumulator[m_atom][n_atom][element + 1]};
                sm120_tf32_pair_store_exp_v1<Sm120Tn>(
                    output, row, column, pair, bias, params, full_tile);
            }
        }
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_entry_bk64_exp_v1(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    if (sm120_tf32_reduction<Sm120Tn>(params) == 0) {
        sm120_tf32_zero_reduction_epilogue<Sm120Tn, M, N>(
            output, bias, params);
        return;
    }
    sm120_tf32_tn_kernel_bk64_exp_v1<M, N, Stages>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(256)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s2_exp_pair_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_pair_entry_exp_v1<Sm120Tn, 128, 64, 2>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(256)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s3_exp_pair_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_pair_entry_exp_v1<Sm120Tn, 128, 64, 3>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(128)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2_exp_pair_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_pair_entry_exp_v1<Sm120Tn, 64, 64, 2>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(256)
void gemm_bi_nt_sm120_tma_mma_tf32_v1_m64n128_bk32_s2_exp_pair_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_pair_entry_exp_v1<Sm120Nt, 64, 128, 2>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(256)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2_exp_w16_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_tn_w16_entry_exp_v1<false>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(256)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2_exp_w16_pair_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_tn_w16_entry_exp_v1<true>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(128)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2_exp_a3d_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_tn_entry_a3d_exp_v1<64, 64, 2>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(256)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s4_exp_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_entry<Sm120Tn, 128, 64, 4>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(256)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_exp_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_entry<Sm120Tn, 64, 128, 4>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(256)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s4_exp_pair_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_pair_entry_exp_v1<Sm120Tn, 128, 64, 4>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(256)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_exp_pair_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_pair_entry_exp_v1<Sm120Tn, 64, 128, 4>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(288)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_exp_pair_producer_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_pair_entry_exp_v1<Sm120Tn, 64, 128, 4, true>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(256)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s4_exp_a3d_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_tn_entry_a3d_exp_v1<128, 64, 4>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(256)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk64_s2_exp_pair_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_tn_entry_bk64_exp_v1<64, 128, 2>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(192)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m96n64_bk32_s3_exp_pair_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_pair_entry_exp_v1<Sm120Tn, 96, 64, 3>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(192)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m96n64_bk32_s4_exp_pair_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_pair_entry_exp_v1<Sm120Tn, 96, 64, 4>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(192)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n96_bk32_s3_exp_pair_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_pair_entry_exp_v1<Sm120Tn, 64, 96, 3>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(192)
void gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n96_bk32_s4_exp_pair_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_pair_entry_exp_v1<Sm120Tn, 64, 96, 4>(
        output, a_map, b_map, bias, params);
}

#endif
