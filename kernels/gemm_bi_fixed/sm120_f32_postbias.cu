// Deterministic exact-F32 post-dot-bias TMA kernels for Fixed inference.
// This fragment is composed after sm120_tma.cu and reuses its SM120 map,
// barrier, and TMA helpers.
#if __CUDA_ARCH__ == 1200

// Exact-F32 Fixed inference specializations. These deliberately live in the
// Fixed module rather than sharing the Triad exact body: Fixed applies bias
// after the dot product, while the Triad one-split contract seeds the FMA
// chain with bias. The seven-argument entry ABI matches the measured SM120
// TMA variants even though one-split Fixed does not use partials or flags.
struct GbfSm120FmaParams {
    float alpha;
    float beta;
    int m;
    int n;
    int k;
    int ldc;
    int splits;
    int tiles_per_split;
};

static __device__ __forceinline__ bool gbf_sm120_fma_aligned16(const void* pointer) {
    return (reinterpret_cast<unsigned long long>(pointer) & 15ULL) == 0;
}

static_assert(sizeof(GbfSm120FmaParams) == 32,
              "Fixed SM120 exact-F32 parameter ABI drift");
static_assert(alignof(GbfSm120FmaParams) == 4,
              "Fixed SM120 exact-F32 parameter alignment drift");
static_assert(__is_standard_layout(GbfSm120FmaParams),
              "Fixed SM120 exact-F32 parameters must remain standard layout");

#define GBF_SM120_FMA_BK 16
#define GBF_SM120_FMA_GROUP_M 16
#define GBF_SM120_FMA_STAGES 2

template <int BM, int BN>
struct GbfSm120FmaStorage {
    static constexpr int a_stage = BM * GBF_SM120_FMA_BK;
    static constexpr int b_stage = GBF_SM120_FMA_BK * BN;
    static constexpr int stage_floats = a_stage + b_stage;
    static constexpr int stage_bytes = stage_floats * (int)sizeof(float);
    static constexpr int dynamic_bytes =
        GBF_SM120_FMA_STAGES * stage_bytes
        + GBF_SM120_FMA_STAGES * 8;
};

static_assert(GbfSm120FmaStorage<128, 64>::dynamic_bytes == 24592,
              "Fixed SM120 m128n64 exact-F32 shared size drift");
static_assert(GbfSm120FmaStorage<64, 128>::dynamic_bytes == 24592,
              "Fixed SM120 m64n128 exact-F32 shared size drift");
static_assert(GbfSm120FmaStorage<128, 96>::dynamic_bytes == 28688,
              "Fixed SM120 m128n96 exact-F32 shared size drift");

template <int BM, int BN, int WM, int WN, int TM, int TN, int THREADS>
static __device__ __forceinline__ void gbf_sm120_fma_fixed_postbias_body(
    float* __restrict__ output,
    const GbfSm120HalfTensorMap& a_map,
    const GbfSm120HalfTensorMap& b_map,
    const float* __restrict__ bias,
    const GbfSm120FmaParams& params) {
    constexpr int BK = GBF_SM120_FMA_BK;
    constexpr int STAGES = GBF_SM120_FMA_STAGES;
    constexpr int A_STAGE = GbfSm120FmaStorage<BM, BN>::a_stage;
    constexpr int STAGE_FLOATS = GbfSm120FmaStorage<BM, BN>::stage_floats;
    constexpr int STAGE_BYTES = GbfSm120FmaStorage<BM, BN>::stage_bytes;
    constexpr int LANE_COLUMNS = WN / TN;
    constexpr int LANE_ROWS = WM / TM;
    static_assert(THREADS / 32 == (BM / WM) * (BN / WN),
                  "Fixed SM120 exact-F32 warp grid does not cover the tile");
    static_assert(LANE_COLUMNS * LANE_ROWS == 32,
                  "Fixed SM120 exact-F32 lane grid does not cover the warp");
    static_assert(THREADS * TM * TN == BM * BN,
                  "Fixed SM120 exact-F32 microtiles do not cover the tile");
    static_assert(STAGE_BYTES % 1024 == 0,
                  "Fixed SM120 exact-F32 stages must remain aligned");

    extern __shared__ __align__(1024) float gbf_sm120_fma_smem[];
    float* stages = gbf_sm120_fma_smem;
    unsigned smem_base = __cvta_generic_to_shared(gbf_sm120_fma_smem);
    unsigned barrier_base = smem_base + STAGES * STAGE_BYTES;

    int tile_id = blockIdx.x;
    int num_pid_m = (params.m + BM - 1) / BM;
    int num_pid_n = (params.n + BN - 1) / BN;
    int num_pid_in_group = GBF_SM120_FMA_GROUP_M * num_pid_n;
    int group_id = tile_id / num_pid_in_group;
    int first_pid_m = group_id * GBF_SM120_FMA_GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, GBF_SM120_FMA_GROUP_M);
    int pid_m = first_pid_m
        + ((tile_id % num_pid_in_group) % group_size_m);
    int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int warp_row = warp / (BN / WN);
    int warp_column = warp % (BN / WN);
    int thread_column = lane % LANE_COLUMNS;
    int thread_row = lane / LANE_COLUMNS;

    auto local_row = [&](int i) -> int {
        return warp_row * WM + thread_row + i * LANE_ROWS;
    };
    auto local_column = [&](int c) -> int {
        return warp_column * WN + (c / 4) * (LANE_COLUMNS * 4)
            + thread_column * 4 + (c % 4);
    };

    float thread_results[TM * TN];
#pragma unroll
    for (int result = 0; result < TM * TN; ++result) {
        thread_results[result] = 0.0f;
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < STAGES; ++stage) {
            gbf_sm120_half_init_barrier<1>(barrier_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    auto produce = [&](int stage, int k_tile) {
        unsigned barrier = barrier_base + stage * 8;
        unsigned destination = smem_base + stage * STAGE_BYTES;
        unsigned b_destination = destination + A_STAGE * (int)sizeof(float);
        unsigned long long a_descriptor =
            reinterpret_cast<unsigned long long>(&a_map);
        unsigned long long b_descriptor =
            reinterpret_cast<unsigned long long>(&b_map);
        gbf_sm120_half_expect_transaction<STAGE_BYTES>(barrier);
        gbf_sm120_half_tma_copy(
            destination, a_descriptor, k_tile * BK, pid_m * BM,
            0, 0, barrier);
        gbf_sm120_half_tma_copy(
            b_destination, b_descriptor, pid_n * BN, k_tile * BK,
            0, 0, barrier);
    };

    int num_k_tiles = (params.k + BK - 1) / BK;
    int range = min(params.tiles_per_split, num_k_tiles);
    if (threadIdx.x == 0 && range > 0) {
        produce(0, 0);
    }
    for (int step = 0; step < range; ++step) {
        int stage = step % STAGES;
        gbf_sm120_half_wait_barrier(
            barrier_base + stage * 8,
            (unsigned)((step / STAGES) & 1));
        __syncthreads();
        int next = step + STAGES - 1;
        if (threadIdx.x == 0 && next < range) {
            produce(next % STAGES, next);
        }

        const float* a_stage = stages + stage * STAGE_FLOATS;
        const float* b_stage = a_stage + A_STAGE;
        float a_fragment[TM][4];
        float b_fragment[TN][4];

#pragma unroll
        for (int dot_index = 0; dot_index < BK; ++dot_index) {
            if (dot_index % 4 == 0) {
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    float4 value = *reinterpret_cast<const float4*>(
                        a_stage + local_row(i) * BK + dot_index);
                    a_fragment[i][0] = value.x;
                    a_fragment[i][1] = value.y;
                    a_fragment[i][2] = value.z;
                    a_fragment[i][3] = value.w;
                }
            }
#pragma unroll
            for (int j = 0; j < TN / 4; ++j) {
                float4 value = *reinterpret_cast<const float4*>(
                    b_stage + dot_index * BN + local_column(j * 4));
                b_fragment[j * 4 + 0][0] = value.x;
                b_fragment[j * 4 + 1][0] = value.y;
                b_fragment[j * 4 + 2][0] = value.z;
                b_fragment[j * 4 + 3][0] = value.w;
            }

            // This row-major result nest and the surrounding dot loop are the
            // Fixed scalar contract's ascending-k __fmaf_rn chain.
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                float a = a_fragment[i][dot_index % 4];
#pragma unroll
                for (int c = 0; c < TN; ++c) {
                    thread_results[i * TN + c] = __fmaf_rn(
                        a, b_fragment[c][0], thread_results[i * TN + c]);
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_row = pid_m * BM + local_row(i);
        if (global_row >= params.m) continue;
#pragma unroll
        for (int j = 0; j < TN / 4; ++j) {
            int global_column = pid_n * BN + local_column(j * 4);
            int idx = i * TN + j * 4;
            float* destination = output
                + (long long)global_row * params.ldc + global_column;
            if (global_column + 3 >= params.n || (params.ldc & 3) != 0
                || !gbf_sm120_fma_aligned16(output)) {
#pragma unroll
                for (int element = 0;
                     element < 4 && global_column + element < params.n;
                     ++element) {
                    float scaled = __fmul_rn(
                        params.alpha, thread_results[idx + element]);
                    destination[element] = __fadd_rn(
                        scaled, bias[global_column + element]);
                }
                continue;
            }

            float v0 = __fadd_rn(
                __fmul_rn(params.alpha, thread_results[idx + 0]),
                bias[global_column + 0]);
            float v1 = __fadd_rn(
                __fmul_rn(params.alpha, thread_results[idx + 1]),
                bias[global_column + 1]);
            float v2 = __fadd_rn(
                __fmul_rn(params.alpha, thread_results[idx + 2]),
                bias[global_column + 2]);
            float v3 = __fadd_rn(
                __fmul_rn(params.alpha, thread_results[idx + 3]),
                bias[global_column + 3]);
            float4 value = {v0, v1, v2, v3};
            reinterpret_cast<float4*>(destination)[0] = value;
        }
    }
}

#define GBF_SM120_FMA_DEFINE_FIXED_POSTBIAS(                                \
    NAME, BM, BN, WM, WN, TM, TN, THREADS, MIN_BLOCKS)                     \
    extern "C" __global__ __launch_bounds__(THREADS, MIN_BLOCKS)           \
    void NAME(                                                              \
        float* __restrict__ output,                                         \
        float* __restrict__ partials,                                       \
        unsigned* __restrict__ flags,                                       \
        const __grid_constant__ GbfSm120HalfTensorMap a_map,                \
        const __grid_constant__ GbfSm120HalfTensorMap b_map,                \
        const float* __restrict__ bias,                                     \
        const __grid_constant__ GbfSm120FmaParams params) {                 \
        (void)partials;                                                      \
        (void)flags;                                                         \
        assert(params.splits == 1);                                         \
        assert(params.beta == 0.0f);                                        \
        assert(bias != nullptr);                                            \
        gbf_sm120_fma_fixed_postbias_body<                                  \
            BM, BN, WM, WN, TM, TN, THREADS>(                              \
            output, a_map, b_map, bias, params);                            \
    }

GBF_SM120_FMA_DEFINE_FIXED_POSTBIAS(
    gemm_bi_nn_sm120_tma_fma_v1_fixed_postbias_m128n64_bk16_s2,
    128, 64, 32, 64, 8, 8, 128, 3)
GBF_SM120_FMA_DEFINE_FIXED_POSTBIAS(
    gemm_bi_nn_sm120_tma_fma_v1_fixed_postbias_m64n128_bk16_s2,
    64, 128, 64, 32, 8, 8, 128, 3)
GBF_SM120_FMA_DEFINE_FIXED_POSTBIAS(
    gemm_bi_nn_sm120_tma_fma_v1_fixed_postbias_m128n96_bk16_s2,
    128, 96, 32, 48, 4, 12, 256, 3)

// Force-only eight-warp twin of the fully-unrolled M128N64 control.
GBF_SM120_FMA_DEFINE_FIXED_POSTBIAS(
    gemm_bi_nn_sm120_tma_fma_v1_fixed_postbias_m128n64_t256_bk16_s2,
    128, 64, 32, 32, 4, 8, 256, 3)

// Force-only no-bias twin. Preserve the existing postbias control body.
template <int BM, int BN, int WM, int WN, int TM, int TN, int THREADS>
static __device__ __forceinline__ void gbf_sm120_fma_fixed_nobias_body(
    float* __restrict__ output,
    const GbfSm120HalfTensorMap& a_map,
    const GbfSm120HalfTensorMap& b_map,
    const GbfSm120FmaParams& params) {
    constexpr int BK = GBF_SM120_FMA_BK;
    constexpr int STAGES = GBF_SM120_FMA_STAGES;
    constexpr int A_STAGE = GbfSm120FmaStorage<BM, BN>::a_stage;
    constexpr int STAGE_FLOATS = GbfSm120FmaStorage<BM, BN>::stage_floats;
    constexpr int STAGE_BYTES = GbfSm120FmaStorage<BM, BN>::stage_bytes;
    constexpr int LANE_COLUMNS = WN / TN;
    constexpr int LANE_ROWS = WM / TM;
    static_assert(THREADS / 32 == (BM / WM) * (BN / WN),
                  "Fixed SM120 exact-F32 warp grid does not cover the tile");
    static_assert(LANE_COLUMNS * LANE_ROWS == 32,
                  "Fixed SM120 exact-F32 lane grid does not cover the warp");
    static_assert(THREADS * TM * TN == BM * BN,
                  "Fixed SM120 exact-F32 microtiles do not cover the tile");
    static_assert(STAGE_BYTES % 1024 == 0,
                  "Fixed SM120 exact-F32 stages must remain aligned");

    extern __shared__ __align__(1024) float gbf_sm120_fma_smem[];
    float* stages = gbf_sm120_fma_smem;
    unsigned smem_base = __cvta_generic_to_shared(gbf_sm120_fma_smem);
    unsigned barrier_base = smem_base + STAGES * STAGE_BYTES;

    int tile_id = blockIdx.x;
    int num_pid_m = (params.m + BM - 1) / BM;
    int num_pid_n = (params.n + BN - 1) / BN;
    int num_pid_in_group = GBF_SM120_FMA_GROUP_M * num_pid_n;
    int group_id = tile_id / num_pid_in_group;
    int first_pid_m = group_id * GBF_SM120_FMA_GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, GBF_SM120_FMA_GROUP_M);
    int pid_m = first_pid_m
        + ((tile_id % num_pid_in_group) % group_size_m);
    int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int warp_row = warp / (BN / WN);
    int warp_column = warp % (BN / WN);
    int thread_column = lane % LANE_COLUMNS;
    int thread_row = lane / LANE_COLUMNS;

    auto local_row = [&](int i) -> int {
        return warp_row * WM + thread_row + i * LANE_ROWS;
    };
    auto local_column = [&](int c) -> int {
        return warp_column * WN + (c / 4) * (LANE_COLUMNS * 4)
            + thread_column * 4 + (c % 4);
    };

    float thread_results[TM * TN];
#pragma unroll
    for (int result = 0; result < TM * TN; ++result) {
        thread_results[result] = 0.0f;
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < STAGES; ++stage) {
            gbf_sm120_half_init_barrier<1>(barrier_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    auto produce = [&](int stage, int k_tile) {
        unsigned barrier = barrier_base + stage * 8;
        unsigned destination = smem_base + stage * STAGE_BYTES;
        unsigned b_destination = destination + A_STAGE * (int)sizeof(float);
        unsigned long long a_descriptor =
            reinterpret_cast<unsigned long long>(&a_map);
        unsigned long long b_descriptor =
            reinterpret_cast<unsigned long long>(&b_map);
        gbf_sm120_half_expect_transaction<STAGE_BYTES>(barrier);
        gbf_sm120_half_tma_copy(
            destination, a_descriptor, k_tile * BK, pid_m * BM,
            0, 0, barrier);
        gbf_sm120_half_tma_copy(
            b_destination, b_descriptor, pid_n * BN, k_tile * BK,
            0, 0, barrier);
    };

    int num_k_tiles = (params.k + BK - 1) / BK;
    int range = min(params.tiles_per_split, num_k_tiles);
    if (threadIdx.x == 0 && range > 0) {
        produce(0, 0);
    }
    for (int step = 0; step < range; ++step) {
        int stage = step % STAGES;
        gbf_sm120_half_wait_barrier(
            barrier_base + stage * 8,
            (unsigned)((step / STAGES) & 1));
        __syncthreads();
        int next = step + STAGES - 1;
        if (threadIdx.x == 0 && next < range) {
            produce(next % STAGES, next);
        }

        const float* a_stage = stages + stage * STAGE_FLOATS;
        const float* b_stage = a_stage + A_STAGE;
        float a_fragment[TM][4];
        float b_fragment[TN][4];

#pragma unroll
        for (int dot_index = 0; dot_index < BK; ++dot_index) {
            if (dot_index % 4 == 0) {
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    float4 value = *reinterpret_cast<const float4*>(
                        a_stage + local_row(i) * BK + dot_index);
                    a_fragment[i][0] = value.x;
                    a_fragment[i][1] = value.y;
                    a_fragment[i][2] = value.z;
                    a_fragment[i][3] = value.w;
                }
            }
#pragma unroll
            for (int j = 0; j < TN / 4; ++j) {
                float4 value = *reinterpret_cast<const float4*>(
                    b_stage + dot_index * BN + local_column(j * 4));
                b_fragment[j * 4 + 0][0] = value.x;
                b_fragment[j * 4 + 1][0] = value.y;
                b_fragment[j * 4 + 2][0] = value.z;
                b_fragment[j * 4 + 3][0] = value.w;
            }

            // This row-major result nest and the surrounding dot loop are the
            // Fixed scalar contract's ascending-k __fmaf_rn chain.
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                float a = a_fragment[i][dot_index % 4];
#pragma unroll
                for (int c = 0; c < TN; ++c) {
                    thread_results[i * TN + c] = __fmaf_rn(
                        a, b_fragment[c][0], thread_results[i * TN + c]);
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_row = pid_m * BM + local_row(i);
        if (global_row >= params.m) continue;
#pragma unroll
        for (int j = 0; j < TN / 4; ++j) {
            int global_column = pid_n * BN + local_column(j * 4);
            int idx = i * TN + j * 4;
            float* destination = output
                + (long long)global_row * params.ldc + global_column;
            if (global_column + 3 >= params.n || (params.ldc & 3) != 0
                || !gbf_sm120_fma_aligned16(output)) {
#pragma unroll
                for (int element = 0;
                     element < 4 && global_column + element < params.n;
                     ++element) {
                    float scaled = __fmul_rn(
                        params.alpha, thread_results[idx + element]);
                    destination[element] = scaled;
                }
                continue;
            }

            float v0 = __fmul_rn(params.alpha, thread_results[idx + 0]);
            float v1 = __fmul_rn(params.alpha, thread_results[idx + 1]);
            float v2 = __fmul_rn(params.alpha, thread_results[idx + 2]);
            float v3 = __fmul_rn(params.alpha, thread_results[idx + 3]);
            float4 value = {v0, v1, v2, v3};
            reinterpret_cast<float4*>(destination)[0] = value;
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 3)
void gemm_bi_nn_sm120_tma_fma_v1_fixed_nobias_m128n64_t256_bk16_s2(
    float* __restrict__ output,
    float* __restrict__ partials,
    unsigned* __restrict__ flags,
    const __grid_constant__ GbfSm120HalfTensorMap a_map,
    const __grid_constant__ GbfSm120HalfTensorMap b_map,
    const float* __restrict__ bias,
    const __grid_constant__ GbfSm120FmaParams params) {
    (void)partials;
    (void)flags;
    assert(params.splits == 1);
    assert(params.beta == 0.0f);
    assert(bias == nullptr);
    gbf_sm120_fma_fixed_nobias_body<128, 64, 32, 32, 4, 8, 256>(
        output, a_map, b_map, params);
}

// Force-only chunk-by-four twin. Keep the control body above unchanged.
template <int BM, int BN, int WM, int WN, int TM, int TN, int THREADS>
static __device__ __forceinline__ void gbf_sm120_fma_fixed_postbias_k4_body(
    float* __restrict__ output,
    const GbfSm120HalfTensorMap& a_map,
    const GbfSm120HalfTensorMap& b_map,
    const float* __restrict__ bias,
    const GbfSm120FmaParams& params) {
    constexpr int BK = GBF_SM120_FMA_BK;
    constexpr int STAGES = GBF_SM120_FMA_STAGES;
    constexpr int A_STAGE = GbfSm120FmaStorage<BM, BN>::a_stage;
    constexpr int STAGE_FLOATS = GbfSm120FmaStorage<BM, BN>::stage_floats;
    constexpr int STAGE_BYTES = GbfSm120FmaStorage<BM, BN>::stage_bytes;
    constexpr int LANE_COLUMNS = WN / TN;
    constexpr int LANE_ROWS = WM / TM;
    static_assert(THREADS / 32 == (BM / WM) * (BN / WN),
                  "Fixed SM120 exact-F32 warp grid does not cover the tile");
    static_assert(LANE_COLUMNS * LANE_ROWS == 32,
                  "Fixed SM120 exact-F32 lane grid does not cover the warp");
    static_assert(THREADS * TM * TN == BM * BN,
                  "Fixed SM120 exact-F32 microtiles do not cover the tile");
    static_assert(STAGE_BYTES % 1024 == 0,
                  "Fixed SM120 exact-F32 stages must remain aligned");

    extern __shared__ __align__(1024) float gbf_sm120_fma_smem[];
    float* stages = gbf_sm120_fma_smem;
    unsigned smem_base = __cvta_generic_to_shared(gbf_sm120_fma_smem);
    unsigned barrier_base = smem_base + STAGES * STAGE_BYTES;

    int tile_id = blockIdx.x;
    int num_pid_m = (params.m + BM - 1) / BM;
    int num_pid_n = (params.n + BN - 1) / BN;
    int num_pid_in_group = GBF_SM120_FMA_GROUP_M * num_pid_n;
    int group_id = tile_id / num_pid_in_group;
    int first_pid_m = group_id * GBF_SM120_FMA_GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, GBF_SM120_FMA_GROUP_M);
    int pid_m = first_pid_m
        + ((tile_id % num_pid_in_group) % group_size_m);
    int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int warp_row = warp / (BN / WN);
    int warp_column = warp % (BN / WN);
    int thread_column = lane % LANE_COLUMNS;
    int thread_row = lane / LANE_COLUMNS;

    auto local_row = [&](int i) -> int {
        return warp_row * WM + thread_row + i * LANE_ROWS;
    };
    auto local_column = [&](int c) -> int {
        return warp_column * WN + (c / 4) * (LANE_COLUMNS * 4)
            + thread_column * 4 + (c % 4);
    };

    float thread_results[TM * TN];
#pragma unroll
    for (int result = 0; result < TM * TN; ++result) {
        thread_results[result] = 0.0f;
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < STAGES; ++stage) {
            gbf_sm120_half_init_barrier<1>(barrier_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    auto produce = [&](int stage, int k_tile) {
        unsigned barrier = barrier_base + stage * 8;
        unsigned destination = smem_base + stage * STAGE_BYTES;
        unsigned b_destination = destination + A_STAGE * (int)sizeof(float);
        unsigned long long a_descriptor =
            reinterpret_cast<unsigned long long>(&a_map);
        unsigned long long b_descriptor =
            reinterpret_cast<unsigned long long>(&b_map);
        gbf_sm120_half_expect_transaction<STAGE_BYTES>(barrier);
        gbf_sm120_half_tma_copy(
            destination, a_descriptor, k_tile * BK, pid_m * BM,
            0, 0, barrier);
        gbf_sm120_half_tma_copy(
            b_destination, b_descriptor, pid_n * BN, k_tile * BK,
            0, 0, barrier);
    };

    int num_k_tiles = (params.k + BK - 1) / BK;
    int range = min(params.tiles_per_split, num_k_tiles);
    if (threadIdx.x == 0 && range > 0) {
        produce(0, 0);
    }
    for (int step = 0; step < range; ++step) {
        int stage = step % STAGES;
        gbf_sm120_half_wait_barrier(
            barrier_base + stage * 8,
            (unsigned)((step / STAGES) & 1));
        __syncthreads();
        int next = step + STAGES - 1;
        if (threadIdx.x == 0 && next < range) {
            produce(next % STAGES, next);
        }

        const float* a_stage = stages + stage * STAGE_FLOATS;
        const float* b_stage = a_stage + A_STAGE;
        float a_fragment[TM][4];
        float b_fragment[TN][4];

        // Keep one four-value chunk in the hot body while retaining ascending k.
#pragma unroll 1
        for (int dot_base = 0; dot_base < BK; dot_base += 4) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                float4 value = *reinterpret_cast<const float4*>(
                    a_stage + local_row(i) * BK + dot_base);
                a_fragment[i][0] = value.x;
                a_fragment[i][1] = value.y;
                a_fragment[i][2] = value.z;
                a_fragment[i][3] = value.w;
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                int dot_index = dot_base + q;
#pragma unroll
                for (int j = 0; j < TN / 4; ++j) {
                    float4 value = *reinterpret_cast<const float4*>(
                        b_stage + dot_index * BN + local_column(j * 4));
                    b_fragment[j * 4 + 0][0] = value.x;
                    b_fragment[j * 4 + 1][0] = value.y;
                    b_fragment[j * 4 + 2][0] = value.z;
                    b_fragment[j * 4 + 3][0] = value.w;
                }
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    float a = a_fragment[i][q];
#pragma unroll
                    for (int c = 0; c < TN; ++c) {
                        thread_results[i * TN + c] = __fmaf_rn(
                            a, b_fragment[c][0], thread_results[i * TN + c]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_row = pid_m * BM + local_row(i);
        if (global_row >= params.m) continue;
#pragma unroll
        for (int j = 0; j < TN / 4; ++j) {
            int global_column = pid_n * BN + local_column(j * 4);
            int idx = i * TN + j * 4;
            float* destination = output
                + (long long)global_row * params.ldc + global_column;
            if (global_column + 3 >= params.n || (params.ldc & 3) != 0
                || !gbf_sm120_fma_aligned16(output)) {
#pragma unroll
                for (int element = 0;
                     element < 4 && global_column + element < params.n;
                     ++element) {
                    float scaled = __fmul_rn(
                        params.alpha, thread_results[idx + element]);
                    destination[element] = __fadd_rn(
                        scaled, bias[global_column + element]);
                }
                continue;
            }

            float v0 = __fadd_rn(
                __fmul_rn(params.alpha, thread_results[idx + 0]),
                bias[global_column + 0]);
            float v1 = __fadd_rn(
                __fmul_rn(params.alpha, thread_results[idx + 1]),
                bias[global_column + 1]);
            float v2 = __fadd_rn(
                __fmul_rn(params.alpha, thread_results[idx + 2]),
                bias[global_column + 2]);
            float v3 = __fadd_rn(
                __fmul_rn(params.alpha, thread_results[idx + 3]),
                bias[global_column + 3]);
            float4 value = {v0, v1, v2, v3};
            reinterpret_cast<float4*>(destination)[0] = value;
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 3)
void gemm_bi_nn_sm120_tma_fma_v1_fixed_postbias_m128n64_bk16_s2_k4(
    float* __restrict__ output,
    float* __restrict__ partials,
    unsigned* __restrict__ flags,
    const __grid_constant__ GbfSm120HalfTensorMap a_map,
    const __grid_constant__ GbfSm120HalfTensorMap b_map,
    const float* __restrict__ bias,
    const __grid_constant__ GbfSm120FmaParams params) {
    (void)partials;
    (void)flags;
    assert(params.splits == 1);
    assert(params.beta == 0.0f);
    assert(bias != nullptr);
    gbf_sm120_fma_fixed_postbias_k4_body<128, 64, 32, 64, 8, 8, 128>(
        output, a_map, b_map, bias, params);
}

#undef GBF_SM120_FMA_DEFINE_FIXED_POSTBIAS
#undef GBF_SM120_FMA_STAGES
#undef GBF_SM120_FMA_GROUP_M
#undef GBF_SM120_FMA_BK

#endif
