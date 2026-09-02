// Exact-F32 NN experiment: the production ascending-k FMA chain behind an
// 8x8 register microtile, fed by TMA, instantiated over three block tiles.
//
// Every output element still walks its reduction in ascending k with one
// __fmaf_rn per step starting from the bias (or zero), so the bits match
// the production 64x64 route bit for bit; only the work per thread, the
// block geometry and the way operands reach shared memory change.
//
// Shape of the schedule:
//   - A and B tiles arrive through cp.async.bulk.tensor, one 2d box each
//     per k tile, issued by a single thread and completed on an mbarrier.
//     The compute warps never touch the load/store pipe for operands, so
//     the FMA and shared-load streams run without the per-warp copy stalls
//     that held the cp.async schedule at 53 TFLOPS.
//   - A lands row-major ([m][k], 64-byte rows) and B lands [k][n]; each
//     thread reads its eight A rows as one float4 per row every four k
//     steps and its B columns as float4 along n every k step.
//   - Each thread owns rows thread_row + i * LANE_ROWS and column chunks
//     j * LANE_COLUMNS * 4 + thread_column * 4, so every float4 shared
//     load is bank-conflict free in both warp shapes without padding.
//   - Two shared stages; the barrier at the top of a k tile is what
//     retires the stage refilled during that tile.
// The driver's opaque tensor-map descriptor, laid out the way the SM120
// production source declares it; NVRTC does not ship cuda.h.
#if __CUDACC_VER_MAJOR__ >= 13
struct alignas(128) CUtensorMap {
#else
struct alignas(64) CUtensorMap {
#endif
    unsigned long long opaque[16];
};

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap size changed");

#ifndef WIDE_NN_BK
#define WIDE_NN_BK 16
#endif
#define WIDE_NN_TM 8
#define WIDE_NN_TN 8
#define WIDE_NN_WARP_SIZE 32
#define WIDE_NN_GROUP_M 16
#ifndef WIDE_NN_K_PIPE
#define WIDE_NN_K_PIPE 2
#endif

struct SgbNnWideParams {
    float alpha;
    float beta;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(SgbNnWideParams) == 32,
              "wide-microtile parameter ABI drift");
static_assert(alignof(SgbNnWideParams) == 4,
              "wide-microtile parameter alignment drift");
static_assert(__is_standard_layout(SgbNnWideParams),
              "wide-microtile parameters must remain standard layout");

template <int Arrivals>
static __device__ __forceinline__ void wide_nn_init_barrier(unsigned barrier) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(barrier), "n"(Arrivals) : "memory");
}

static __device__ __forceinline__ void wide_nn_wait_barrier(
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
static __device__ __forceinline__ void wide_nn_expect_transaction(
    unsigned barrier) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
        "_, [%0], %1;"
        :: "r"(barrier), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void wide_nn_tma_copy(
    unsigned destination, const CUtensorMap* map, int x, int y,
    unsigned barrier) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(destination), "l"(reinterpret_cast<unsigned long long>(map)),
           "r"(x), "r"(y), "r"(barrier)
        : "memory");
}

template <int BM, int BN, int WM, int WN, int THREADS>
__device__ __forceinline__ void wide_nn_microtile_body(
    float* __restrict__ C,
    const float* __restrict__ bias,
    const SgbNnWideParams& params,
    const CUtensorMap& a_map,
    const CUtensorMap& b_map
) {
    constexpr int BK = WIDE_NN_BK;
    constexpr int TM = WIDE_NN_TM;
    constexpr int TN = WIDE_NN_TN;
    constexpr int K_PIPE = WIDE_NN_K_PIPE;
    constexpr int A_ROW = BK;
    constexpr int B_ROW = BN;
    constexpr int A_STAGE = BM * A_ROW;
    constexpr int B_STAGE = BK * B_ROW;
    constexpr int STAGE_FLOATS = A_STAGE + B_STAGE;
    constexpr int STAGE_BYTES = STAGE_FLOATS * (int)sizeof(float);
    constexpr int WARPS = THREADS / WIDE_NN_WARP_SIZE;
    constexpr int LANE_COLUMNS = WN / TN;
    constexpr int LANE_ROWS = WM / TM;
    static_assert(WARPS == (BM / WM) * (BN / WN),
                  "warp grid does not cover the block tile");
    static_assert(LANE_COLUMNS * LANE_ROWS == WIDE_NN_WARP_SIZE,
                  "lane grid does not cover the warp tile");
    static_assert(BK % 4 == 0 && A_ROW % 4 == 0 && B_ROW % 4 == 0,
                  "shared rows must stay float4 aligned");
    static_assert(STAGE_BYTES % 128 == 0,
                  "TMA stages must stay 128-byte aligned");
    static_assert(TM % 4 == 0 && TN % 4 == 0,
                  "fragments must be whole float4 groups");
    static_assert(K_PIPE >= 2, "the pipeline needs a stage to refill");

    extern __shared__ __align__(128) float wide_nn_smem[];
    float* stages = wide_nn_smem;
    unsigned smem_base = __cvta_generic_to_shared(wide_nn_smem);
    unsigned barrier_base = smem_base + K_PIPE * STAGE_BYTES;

    int num_pid_m = (params.m + BM - 1) / BM;
    int num_pid_n = (params.n + BN - 1) / BN;
    int num_pid_in_group = WIDE_NN_GROUP_M * num_pid_n;
    int tile_id = blockIdx.x;
    int group_id = tile_id / num_pid_in_group;
    int first_pid_m = group_id * WIDE_NN_GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, WIDE_NN_GROUP_M);
    int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
    int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    int warp = threadIdx.x / WIDE_NN_WARP_SIZE;
    int lane = threadIdx.x % WIDE_NN_WARP_SIZE;
    int warp_row = warp / (BN / WN);
    int warp_column = warp % (BN / WN);
    int thread_column = lane % LANE_COLUMNS;
    int thread_row = lane / LANE_COLUMNS;
    int row_base = warp_row * WM + thread_row;
    int column_base = warp_column * WN + thread_column * 4;

    float threadResults[TM * TN];

    if (bias != nullptr) {
        #pragma unroll
        for (int j = 0; j < TN / 4; ++j) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                int global_column = pid_n * BN + column_base
                    + j * (LANE_COLUMNS * 4) + e;
                float bias_value =
                    global_column < params.n ? bias[global_column] : 0.0f;
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    threadResults[i * TN + j * 4 + e] = bias_value;
                }
            }
        }
    } else {
        #pragma unroll
        for (int result = 0; result < TM * TN; ++result) {
            threadResults[result] = 0.0f;
        }
    }

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < K_PIPE; ++stage) {
            wide_nn_init_barrier<1>(barrier_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    // Out-of-range rows and columns arrive as zeros from the tensor map,
    // which is exactly what the production loader zero-fills.
    auto produce = [&](int stage, int k_tile) {
        unsigned barrier = barrier_base + stage * 8;
        unsigned destination = smem_base + stage * STAGE_BYTES;
        wide_nn_expect_transaction<STAGE_BYTES>(barrier);
        wide_nn_tma_copy(destination, &a_map, k_tile * BK, pid_m * BM, barrier);
        wide_nn_tma_copy(destination + A_STAGE * (int)sizeof(float), &b_map,
                         pid_n * BN, k_tile * BK, barrier);
    };

    int num_k_tiles = (params.k + BK - 1) / BK;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < K_PIPE - 1; ++stage) {
            if (stage < num_k_tiles) {
                produce(stage, stage);
            }
        }
    }
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        int stage = tile % K_PIPE;
        wide_nn_wait_barrier(barrier_base + stage * 8,
                             (unsigned)((tile / K_PIPE) & 1));
        // The stage refilled below was read one iteration ago; this barrier
        // is what proves every thread has left it.
        __syncthreads();
        int next_tile = tile + K_PIPE - 1;
        if (threadIdx.x == 0 && next_tile < num_k_tiles) {
            produce(next_tile % K_PIPE, next_tile);
        }

        const float* As_read = stages + stage * STAGE_FLOATS;
        const float* Bs_read = As_read + A_STAGE;
        float a_fragment[TM][4];
        float b_fragment[TN];

        #pragma unroll
        for (int dot_index = 0; dot_index < BK; ++dot_index) {
            if (dot_index % 4 == 0) {
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    float4 value = *reinterpret_cast<const float4*>(
                        As_read + (row_base + i * LANE_ROWS) * A_ROW
                        + dot_index);
                    a_fragment[i][0] = value.x;
                    a_fragment[i][1] = value.y;
                    a_fragment[i][2] = value.z;
                    a_fragment[i][3] = value.w;
                }
            }
            #pragma unroll
            for (int j = 0; j < TN / 4; ++j) {
                float4 value = *reinterpret_cast<const float4*>(
                    Bs_read + dot_index * B_ROW + column_base
                    + j * (LANE_COLUMNS * 4));
                b_fragment[j * 4 + 0] = value.x;
                b_fragment[j * 4 + 1] = value.y;
                b_fragment[j * 4 + 2] = value.z;
                b_fragment[j * 4 + 3] = value.w;
            }

            // Keep this row-major result nest in ascending reduction order.
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                float a = a_fragment[i][dot_index % 4];
                #pragma unroll
                for (int c = 0; c < TN; ++c) {
                    threadResults[i * TN + c] =
                        __fmaf_rn(a, b_fragment[c], threadResults[i * TN + c]);
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_row = pid_m * BM + row_base + i * LANE_ROWS;
        if (global_row >= params.m) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < TN / 4; ++j) {
            int global_column = pid_n * BN + column_base + j * (LANE_COLUMNS * 4);
            int idx = i * TN + j * 4;
            float* destination =
                C + (long long)global_row * params.ldc + global_column;
            if (global_column + 3 >= params.n || (params.ldc & 3) != 0
                || !gemm_bi_is_aligned_16(C)) {
                #pragma unroll
                for (int element = 0;
                     element < 4 && global_column + element < params.n;
                     ++element) {
                    float value = params.alpha * threadResults[idx + element];
                    if (params.beta != 0.0f) {
                        value += params.beta * destination[element];
                    }
                    destination[element] = value;
                }
                continue;
            }

            float4 previous;
            if (params.beta != 0.0f) {
                previous = reinterpret_cast<float4*>(destination)[0];
            }
            float v0 = params.alpha * threadResults[idx + 0];
            float v1 = params.alpha * threadResults[idx + 1];
            float v2 = params.alpha * threadResults[idx + 2];
            float v3 = params.alpha * threadResults[idx + 3];
            if (params.beta != 0.0f) {
                v0 += params.beta * previous.x;
                v1 += params.beta * previous.y;
                v2 += params.beta * previous.z;
                v3 += params.beta * previous.w;
            }
            float4 output = {v0, v1, v2, v3};
            reinterpret_cast<float4*>(destination)[0] = output;
        }
    }
}

// 64x128 block tile, four warps side by side each owning 64x32.
extern "C" __global__ __launch_bounds__(128, 3)
void gemm_bi_nn_m64n128_bk16_s2_micro8x8_exp(
    float* __restrict__ C,
    const float* __restrict__ bias,
    const __grid_constant__ SgbNnWideParams params,
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map
) {
    assert(params.alpha == 1.0f || bias == nullptr);
    wide_nn_microtile_body<64, 128, 64, 32, 128>(C, bias, params, a_map, b_map);
}

// 128x128 block tile, eight warps in a 2x4 grid each owning 64x32.
extern "C" __global__ __launch_bounds__(256, 2)
void gemm_bi_nn_m128n128_bk16_s2_micro8x8_exp(
    float* __restrict__ C,
    const float* __restrict__ bias,
    const __grid_constant__ SgbNnWideParams params,
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map
) {
    assert(params.alpha == 1.0f || bias == nullptr);
    wide_nn_microtile_body<128, 128, 64, 32, 256>(C, bias, params, a_map, b_map);
}

// 128x64 block tile, four warps stacked each owning 32x64.
extern "C" __global__ __launch_bounds__(128, 3)
void gemm_bi_nn_m128n64_bk16_s2_micro8x8_exp(
    float* __restrict__ C,
    const float* __restrict__ bias,
    const __grid_constant__ SgbNnWideParams params,
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map
) {
    assert(params.alpha == 1.0f || bias == nullptr);
    wide_nn_microtile_body<128, 64, 32, 64, 128>(C, bias, params, a_map, b_map);
}

#undef WIDE_NN_K_PIPE
#undef WIDE_NN_GROUP_M
#undef WIDE_NN_WARP_SIZE
#undef WIDE_NN_TN
#undef WIDE_NN_TM
#undef WIDE_NN_BK
