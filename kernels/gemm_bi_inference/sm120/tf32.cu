#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)

#if __CUDACC_VER_MAJOR__ >= 13
struct alignas(128) GbfTensorMap {
#else
struct alignas(64) GbfTensorMap {
#endif
    unsigned long long opaque[16];
};

static_assert(sizeof(GbfTensorMap) == 128, "Tensor-map ABI drift");

struct GbfSm120Tf32Params {
    int m;
    int k;
    int n;
    int ldc;
};

static_assert(sizeof(GbfSm120Tf32Params) == 16, "SM120 TF32 parameter ABI drift");
static_assert(__is_standard_layout(GbfSm120Tf32Params),
              "SM120 TF32 parameters must remain standard layout");

template <int M, int N, int Stages>
struct GbfSm120Tf32Storage {
    static constexpr int stage_bytes = (M + N) * 32 * 4;
    static constexpr int dynamic_bytes = 128 + Stages * stage_bytes;
};

template <int M, int N, int WarpN>
struct GbfSm120Tf32Warps {
    static constexpr int threads = (M / 32) * (N / WarpN) * 32;
};

static_assert(GbfSm120Tf32Storage<128, 64, 2>::dynamic_bytes == 49280);
static_assert(GbfSm120Tf32Storage<128, 64, 3>::dynamic_bytes == 73856);
static_assert(GbfSm120Tf32Storage<64, 128, 2>::dynamic_bytes == 49280);
static_assert(GbfSm120Tf32Storage<64, 128, 3>::dynamic_bytes == 73856);
static_assert(GbfSm120Tf32Storage<64, 64, 2>::dynamic_bytes == 32896);

template <int Arrivals>
__device__ __forceinline__ void gbf_sm120_init_barrier(unsigned barrier) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(barrier), "n"(Arrivals) : "memory");
}

__device__ __forceinline__ void gbf_sm120_wait_barrier(
    unsigned barrier, unsigned phase) {
    unsigned ready;
    do {
        asm volatile(
            "{ .reg .pred p; "
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
            "p, [%1], %2; "
            "selp.b32 %0, 1, 0, p; }"
            : "=r"(ready) : "r"(barrier), "r"(phase) : "memory");
    } while (!ready);
}

template <int Bytes>
__device__ __forceinline__ void gbf_sm120_expect_transaction(unsigned barrier) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
        "_, [%0], %1;"
        :: "r"(barrier), "n"(Bytes) : "memory");
}

__device__ __forceinline__ void gbf_sm120_arrive_empty(unsigned barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :: "r"(barrier) : "memory");
}

__device__ __forceinline__ void gbf_sm120_tma_copy(
    unsigned destination, unsigned long long map, int x, int y,
    unsigned barrier) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(destination), "l"(map), "r"(x), "r"(y), "r"(barrier)
        : "memory");
}

__device__ __forceinline__ unsigned gbf_sm120_swizzled_offset(
    unsigned plane_base, unsigned logical_row, unsigned element) {
    unsigned chunk = element / 4U;
    unsigned element_in_vector = element & 3;
    unsigned row_start = plane_base + logical_row * 128U;
    unsigned phase = (row_start / 128U) & 7U;
    unsigned physical_chunk = chunk ^ phase;
    return row_start + physical_chunk * 16U
        + element_in_vector * 4;
}

__device__ __forceinline__ unsigned gbf_sm120_tf32_rna(float value) {
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ void gbf_sm120_tf32_mma(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

template <int M, int N>
__device__ __forceinline__ void gbf_sm120_tf32_zero_reduction(
    float* output, const float* bias, const GbfSm120Tf32Params& params) {
    int column_tiles = (params.n + N - 1) / N;
    int output_row = (int)blockIdx.x / column_tiles * M;
    int output_column = (int)blockIdx.x % column_tiles * N;
    for (int linear = (int)threadIdx.x; linear < M * N; linear += (int)blockDim.x) {
        int row = output_row + linear / N;
        int column = output_column + linear % N;
        if (row < params.m && column < params.n) {
            output[(long long)row * params.ldc + column] =
                bias == nullptr ? 0.0f : bias[column];
        }
    }
}

struct GbfSm120StageContext {
    const GbfTensorMap* a_map;
    const GbfTensorMap* b_map;
    unsigned payload;
    unsigned full_base;
    int output_row;
    int output_column;
};

template <int M, int N, int Stages>
__device__ __forceinline__ void gbf_sm120_produce_stage(
    const GbfSm120StageContext& context, int tile) {
    constexpr int plane_bytes = 32 * 32 * 4;
    constexpr int stage_bytes = GbfSm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned stage_index = (unsigned)(tile % Stages);
    unsigned stage = context.payload + stage_index * stage_bytes;
    unsigned barrier = context.full_base + stage_index * 8;
    unsigned a_destination = stage;
    unsigned b_destination = stage + M * 32 * 4;
    unsigned long long a_descriptor =
        reinterpret_cast<unsigned long long>(context.a_map);
    unsigned long long b_descriptor =
        reinterpret_cast<unsigned long long>(context.b_map);
    int reduction = tile * 32;
    gbf_sm120_expect_transaction<GbfSm120Tf32Storage<M, N, Stages>::stage_bytes>(barrier);
    gbf_sm120_tma_copy(
        a_destination, a_descriptor, reduction, context.output_row, barrier);
#pragma unroll
    for (int plane = 0; plane < N / 32; ++plane) {
        gbf_sm120_tma_copy(
            b_destination + plane * plane_bytes, b_descriptor,
            context.output_column + plane * 32, reduction, barrier);
    }
}

template <int M, int N, int Stages>
__device__ __forceinline__ float gbf_sm120_load_a(
    unsigned char* storage, int stage, int row, int reduction) {
    constexpr int stage_bytes = GbfSm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned plane = (unsigned)(row / 32) * (32 * 32 * 4U);
    unsigned logical_row = (unsigned)(row & 31);
    unsigned offset = gbf_sm120_swizzled_offset(
        plane, logical_row, (unsigned)reduction);
    return *reinterpret_cast<float*>(storage + stage * stage_bytes + offset);
}

template <int M, int N, int Stages>
__device__ __forceinline__ float gbf_sm120_load_b(
    unsigned char* storage, int stage, int reduction, int column) {
    constexpr int stage_bytes = GbfSm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned base = M * 32 * 4;
    unsigned plane = base + (unsigned)(column / 32) * (32 * 32 * 4U);
    unsigned offset = gbf_sm120_swizzled_offset(
        plane, (unsigned)reduction, (unsigned)(column & 31));
    return *reinterpret_cast<float*>(storage + stage * stage_bytes + offset);
}

template <int M, int N, int Stages, int WarpN>
__device__ __forceinline__ void gbf_sm120_load_issue(
    unsigned char* storage, int stage, int warp_m, int warp_n,
    int k8, unsigned (&a_fragments)[2][4],
    unsigned (&b_fragments)[WarpN / 8][2]) {
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
        int row = warp_m + m_atom * 16 + group;
        a_fragments[m_atom][0] = gbf_sm120_tf32_rna(
            gbf_sm120_load_a<M, N, Stages>(storage, stage, row, k8 + thread));
        a_fragments[m_atom][1] = gbf_sm120_tf32_rna(
            gbf_sm120_load_a<M, N, Stages>(
                storage, stage, row + 8, k8 + thread));
        a_fragments[m_atom][2] = gbf_sm120_tf32_rna(
            gbf_sm120_load_a<M, N, Stages>(
                storage, stage, row, k8 + thread + 4));
        a_fragments[m_atom][3] = gbf_sm120_tf32_rna(
            gbf_sm120_load_a<M, N, Stages>(
                storage, stage, row + 8, k8 + thread + 4));
    }
#pragma unroll
    for (int n_atom = 0; n_atom < WarpN / 8; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        b_fragments[n_atom][0] = gbf_sm120_tf32_rna(
            gbf_sm120_load_b<M, N, Stages>(storage, stage, k8 + thread, column));
        b_fragments[n_atom][1] = gbf_sm120_tf32_rna(
            gbf_sm120_load_b<M, N, Stages>(
                storage, stage, k8 + thread + 4, column));
    }
}

template <int M, int N, int Stages, int WarpN>
__device__ __forceinline__ void gbf_sm120_issue_stage(
    unsigned char* storage, int stage, int warp_m, int warp_n,
    float (&accumulator)[2][WarpN / 8][4]) {
    unsigned a_fragments[2][2][4];
    unsigned b_fragments[2][WarpN / 8][2];
    gbf_sm120_load_issue<M, N, Stages, WarpN>(
        storage, stage, warp_m, warp_n, 0, a_fragments[0], b_fragments[0]);
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        int current = issue & 1;
        int next = current ^ 1;
        if (issue + 1 < 4) {
            gbf_sm120_load_issue<M, N, Stages, WarpN>(
                storage, stage, warp_m, warp_n, (issue + 1) * 8,
                a_fragments[next], b_fragments[next]);
        }
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < WarpN / 8; ++n_atom) {
                gbf_sm120_tf32_mma(
                    accumulator[m_atom][n_atom],
                    a_fragments[current][m_atom], b_fragments[current][n_atom]);
            }
        }
    }
}

template <int M, int N, int Stages, int WarpN, bool PairStore,
          bool ProducerWarp>
__device__ __forceinline__ void gbf_sm120_tf32_kernel(
    float* output, const GbfTensorMap& a_map, const GbfTensorMap& b_map,
    const float* bias, const GbfSm120Tf32Params& params) {
    extern __shared__ __align__(1024) unsigned char storage[];
    constexpr int stage_bytes = GbfSm120Tf32Storage<M, N, Stages>::stage_bytes;
    constexpr int compute_warps = GbfSm120Tf32Warps<M, N, WarpN>::threads / 32;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    unsigned payload = shared;
    unsigned full_base = shared + Stages * stage_bytes;
    unsigned empty_base = full_base + 64;
    int column_tiles = (params.n + N - 1) / N;
    int output_row = (int)blockIdx.x / column_tiles * M;
    int output_column = (int)blockIdx.x % column_tiles * N;
    int tile_count = (params.k + 31) / 32;
    const GbfSm120StageContext context = {
        &a_map, &b_map, payload, full_base, output_row, output_column};
    int warp = (int)threadIdx.x >> 5;
    int compute_warp = ProducerWarp ? warp - 1 : warp;
    constexpr int warp_columns = N / WarpN;
    int warp_m = (compute_warp / warp_columns) * 32;
    int warp_n = (compute_warp % warp_columns) * WarpN;
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
    float accumulator[2][WarpN / 8][4];

    if (!ProducerWarp || warp != 0) {
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < WarpN / 8; ++n_atom) {
#pragma unroll
                for (int element = 0; element < 4; ++element) {
                    int column = output_column + warp_n + n_atom * 8
                        + 2 * thread + (element & 1);
                    accumulator[m_atom][n_atom][element] =
                        column < params.n && bias != nullptr ? bias[column] : 0.0f;
                }
            }
        }
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            gbf_sm120_init_barrier<1>(full_base + stage * 8);
            gbf_sm120_init_barrier<compute_warps>(empty_base + stage * 8);
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
                        gbf_sm120_produce_stage<M, N, Stages>(context, tile);
                    }
                }
                for (int refill = Stages; refill < tile_count; ++refill) {
                    int consumed = refill - Stages;
                    int stage = consumed % Stages;
                    unsigned generation = (unsigned)(consumed / Stages);
                    gbf_sm120_wait_barrier(
                        empty_base + stage * 8, generation & 1U);
                    gbf_sm120_produce_stage<M, N, Stages>(context, refill);
                }
            }
            return;
        }
    } else {
        if (warp == 0 && lane == 0) {
#pragma unroll
            for (int tile = 0; tile < Stages; ++tile) {
                if (tile < tile_count) {
                    gbf_sm120_produce_stage<M, N, Stages>(context, tile);
                }
            }
        }
        if constexpr (Stages == 2) __syncwarp();
    }

    for (int tile = 0; tile < tile_count; ++tile) {
        int stage = tile % Stages;
        unsigned generation = (unsigned)(tile / Stages);
        gbf_sm120_wait_barrier(full_base + stage * 8, generation & 1U);
        gbf_sm120_issue_stage<M, N, Stages, WarpN>(
            storage, stage, warp_m, warp_n, accumulator);
        if (lane == 0) gbf_sm120_arrive_empty(empty_base + stage * 8);
        if constexpr (!ProducerWarp) {
            if (warp == 0 && lane == 0) {
                int refill = tile + Stages;
                if (refill < tile_count) {
                    gbf_sm120_wait_barrier(empty_base + stage * 8, generation & 1U);
                    gbf_sm120_produce_stage<M, N, Stages>(context, refill);
                }
            }
            if constexpr (Stages == 2) __syncwarp();
        }
    }

    if constexpr (PairStore) {
        bool pair_store_fast = params.m >= M
            && output_row <= params.m - M
            && params.n >= N
            && output_column <= params.n - N
            && (reinterpret_cast<unsigned long long>(output) & 7ULL) == 0
            && (params.ldc & 1) == 0;
        if (pair_store_fast) {
#pragma unroll
            for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
                for (int n_atom = 0; n_atom < WarpN / 8; ++n_atom) {
#pragma unroll
                    for (int element = 0; element < 4; element += 2) {
                        int row = output_row + warp_m + m_atom * 16
                            + group + (element >= 2 ? 8 : 0);
                        int column = output_column + warp_n + n_atom * 8
                            + 2 * thread + (element & 1);
                        float2 pair = {
                            accumulator[m_atom][n_atom][element],
                            accumulator[m_atom][n_atom][element + 1]};
                        *reinterpret_cast<float2*>(
                            output + (long long)row * params.ldc + column) = pair;
                    }
                }
            }
            return;
        }
    }

#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < WarpN / 8; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = output_row + warp_m + m_atom * 16
                    + group + (element >= 2 ? 8 : 0);
                int column = output_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                if (row < params.m && column < params.n) {
                    output[(long long)row * params.ldc + column] =
                        accumulator[m_atom][n_atom][element];
                }
            }
        }
    }
}

template <int M, int N, int Stages, int WarpN, bool PairStore,
          bool ProducerWarp>
__device__ __forceinline__ void gbf_sm120_tf32_entry(
    float* output, const GbfTensorMap& a_map, const GbfTensorMap& b_map,
    const float* bias, const GbfSm120Tf32Params& params) {
    if (params.k == 0) {
        gbf_sm120_tf32_zero_reduction<M, N>(output, bias, params);
        return;
    }
    gbf_sm120_tf32_kernel<M, N, Stages, WarpN, PairStore, ProducerWarp>(
        output, a_map, b_map, bias, params);
}

#define GBF_SM120_TF32_KERNEL(NAME, M, N, STAGES, WARP_N, PAIR_STORE, PRODUCER_WARP) \
extern "C" __global__ __launch_bounds__((M * N) / WARP_N                    \
                                         + (PRODUCER_WARP ? 32 : 0)) void NAME( \
    float* output, const __grid_constant__ GbfTensorMap a_map,                \
    const __grid_constant__ GbfTensorMap b_map, const float* bias,            \
    const __grid_constant__ GbfSm120Tf32Params params) {                      \
    gbf_sm120_tf32_entry<M, N, STAGES, WARP_N, PAIR_STORE, PRODUCER_WARP>(    \
        output, a_map, b_map, bias, params);                                  \
}

GBF_SM120_TF32_KERNEL(
    nn_sm120_tma_tf32_m128n64_bk32_s2, 128, 64, 2, 64, true, false)
GBF_SM120_TF32_KERNEL(
    nn_sm120_tma_tf32_m128n64_bk32_s3, 128, 64, 3, 32, false, false)
GBF_SM120_TF32_KERNEL(
    nn_sm120_tma_tf32_m64n128_bk32_s2, 64, 128, 2, 64, true, false)
GBF_SM120_TF32_KERNEL(
    nn_sm120_tma_tf32_m64n128_bk32_s3, 64, 128, 3, 32, false, false)
GBF_SM120_TF32_KERNEL(
    nn_sm120_tma_tf32_m64n64_bk32_s2_producer_warp,
    64, 64, 2, 32, true, true)
GBF_SM120_TF32_KERNEL(
    nn_sm120_tma_tf32_m64n64_bk32_s2, 64, 64, 2, 32, false, false)
GBF_SM120_TF32_KERNEL(
    nn_sm120_tma_tf32_m64n64_bk32_s2_pair_store,
    64, 64, 2, 32, true, false)

#undef GBF_SM120_TF32_KERNEL
#endif
