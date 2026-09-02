// Exact-F32 NN experiment: the production ascending-k FMA chain behind an
// 8x8 register microtile, instantiated over three block tiles.
//
// Every output element still walks its reduction in ascending k with one
// __fmaf_rn per step starting from the bias (or zero), so the bits match
// the production 64x64 route bit for bit; only the work per thread and the
// block geometry change.
//
// Shape of the schedule:
//   - A stays row-major in shared memory ([m][k], padded to 80 bytes per
//     row) and arrives through 16-byte cp.async, so no copy scatters
//     4-byte elements. Each thread reads its eight rows as one float4 per
//     row every four k steps.
//   - B stays [k][n] and is read as float4 along n every k step.
//   - Each thread owns rows thread_row + i * LANE_ROWS and column chunks
//     j * LANE_COLUMNS * 4 + thread_column * 4, so every float4 shared
//     load is bank-conflict free in both warp shapes.
//   - The copies for the next k tile are spread across the unrolled k
//     loop instead of being issued in one burst after the barrier, which
//     keeps the FMA pipe fed while the memory pipe drains.
#ifndef WIDE_NN_BK
#define WIDE_NN_BK 16
#endif
#define WIDE_NN_TM 8
#define WIDE_NN_TN 8
#define WIDE_NN_WARP_SIZE 32
#define WIDE_NN_A_PAD 4
#define WIDE_NN_B_PAD 4
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

template <int BM, int BN, int WM, int WN, int THREADS>
__device__ __forceinline__ void wide_nn_microtile_body(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    const SgbNnWideParams& params
) {
    constexpr int BK = WIDE_NN_BK;
    constexpr int TM = WIDE_NN_TM;
    constexpr int TN = WIDE_NN_TN;
    constexpr int K_PIPE = WIDE_NN_K_PIPE;
    constexpr int A_ROW = BK + WIDE_NN_A_PAD;
    constexpr int B_ROW = BN + WIDE_NN_B_PAD;
    constexpr int A_STAGE = BM * A_ROW;
    constexpr int B_STAGE = BK * B_ROW;
    constexpr int WARPS = THREADS / WIDE_NN_WARP_SIZE;
    constexpr int LANE_COLUMNS = WN / TN;
    constexpr int LANE_ROWS = WM / TM;
    constexpr int A_COPY_ROWS = THREADS / (BK / 4);
    constexpr int A_COPIES = BM / A_COPY_ROWS;
    constexpr int B_COPY_ROWS = THREADS / (BN / 4);
    constexpr int B_COPIES = BK / B_COPY_ROWS;
    constexpr int COPIES = A_COPIES + B_COPIES;
    constexpr int COPY_STRIDE = BK / COPIES;
    static_assert(WARPS == (BM / WM) * (BN / WN),
                  "warp grid does not cover the block tile");
    static_assert(LANE_COLUMNS * LANE_ROWS == WIDE_NN_WARP_SIZE,
                  "lane grid does not cover the warp tile");
    static_assert(BM % A_COPY_ROWS == 0 && BK % B_COPY_ROWS == 0,
                  "copies do not cover the tiles");
    static_assert(COPIES <= BK && COPY_STRIDE >= 1,
                  "too many copies to spread over the k loop");
    static_assert(BK % 4 == 0 && A_ROW % 4 == 0 && B_ROW % 4 == 0,
                  "shared rows must stay float4 aligned");
    static_assert(TM % 4 == 0 && TN % 4 == 0,
                  "fragments must be whole float4 groups");

    extern __shared__ __align__(16) float smem[];
    float* As_buf = smem;
    float* Bs_buf = smem + K_PIPE * A_STAGE;

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
    int a_copy_row = threadIdx.x / (BK / 4);
    int a_copy_k = (threadIdx.x % (BK / 4)) * 4;
    int b_copy_row = threadIdx.x / (BN / 4);
    int b_copy_column = (threadIdx.x % (BN / 4)) * 4;
    bool a_vector = (params.lda & 3) == 0 && gemm_bi_is_aligned_16(A);
    bool b_vector = (params.ldb & 3) == 0 && gemm_bi_is_aligned_16(B);

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

    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    // One 16-byte copy of A per call; the source-size operand zero-fills
    // whatever lies past the k edge, and the element path covers strides
    // that are not float4 aligned.
    auto issue_a = [&](int stage, int bk_index, int copy) {
        int m_local = copy * A_COPY_ROWS + a_copy_row;
        int global_row = pid_m * BM + m_local;
        int global_column = bk_index + a_copy_k;
        unsigned destination = As_base
            + (stage * A_STAGE + m_local * A_ROW + a_copy_k)
                * (unsigned)sizeof(float);
        int remaining = params.k - global_column;
        bool valid = global_row < params.m && remaining > 0;
        if (a_vector) {
            int source_bytes = valid ? min(remaining, 4) * 4 : 0;
            const float* source = valid
                ? A + (long long)global_row * params.lda + global_column
                : A;
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                :: "r"(destination), "l"(source), "r"(source_bytes));
        } else {
            #pragma unroll
            for (int element = 0; element < 4; ++element) {
                bool element_valid = valid && element < remaining;
                const float* source = element_valid
                    ? A + (long long)global_row * params.lda
                        + global_column + element
                    : A;
                int source_bytes = element_valid ? 4 : 0;
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                    :: "r"(destination + element * (unsigned)sizeof(float)),
                       "l"(source), "r"(source_bytes));
            }
        }
    };

    auto issue_b = [&](int stage, int bk_index, int copy) {
        int k_local = copy * B_COPY_ROWS + b_copy_row;
        int global_row = bk_index + k_local;
        int global_column = pid_n * BN + b_copy_column;
        unsigned destination = Bs_base
            + (stage * B_STAGE + k_local * B_ROW + b_copy_column)
                * (unsigned)sizeof(float);
        int remaining = params.n - global_column;
        bool valid = global_row < params.k && remaining > 0;
        if (b_vector) {
            int source_bytes = valid ? min(remaining, 4) * 4 : 0;
            const float* source = valid
                ? B + (long long)global_row * params.ldb + global_column
                : B;
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                :: "r"(destination), "l"(source), "r"(source_bytes));
        } else {
            #pragma unroll
            for (int element = 0; element < 4; ++element) {
                bool element_valid = valid && element < remaining;
                const float* source = element_valid
                    ? B + (long long)global_row * params.ldb
                        + global_column + element
                    : B;
                int source_bytes = element_valid ? 4 : 0;
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                    :: "r"(destination + element * (unsigned)sizeof(float)),
                       "l"(source), "r"(source_bytes));
            }
        }
    };

    auto issue_copy = [&](int stage, int bk_index, int copy) {
        if (copy < A_COPIES) {
            issue_a(stage, bk_index, copy);
        } else {
            issue_b(stage, bk_index, copy - A_COPIES);
        }
    };

    int num_k_tiles = (params.k + BK - 1) / BK;
    #pragma unroll
    for (int stage = 0; stage < K_PIPE - 1; ++stage) {
        if (stage < num_k_tiles) {
            #pragma unroll
            for (int copy = 0; copy < COPIES; ++copy) {
                issue_copy(stage, stage * BK, copy);
            }
        }
        asm volatile("cp.async.commit_group;\n");
    }
    int read_stage = 0;
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(K_PIPE - 2));
        __syncthreads();

        // The stage written during this iteration was read one iteration
        // ago; every thread has passed the barrier above since then.
        int next_tile = tile + K_PIPE - 1;
        int write_stage = (read_stage + K_PIPE - 1) % K_PIPE;
        bool has_next = next_tile < num_k_tiles;
        int next_bk = next_tile * BK;
        const float* As_read = As_buf + read_stage * A_STAGE;
        const float* Bs_read = Bs_buf + read_stage * B_STAGE;
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
            if (dot_index % COPY_STRIDE == 0
                && dot_index / COPY_STRIDE < COPIES) {
                int copy = dot_index / COPY_STRIDE;
                if (has_next) {
                    issue_copy(write_stage, next_bk, copy);
                }
                if (copy == COPIES - 1) {
                    asm volatile("cp.async.commit_group;\n");
                }
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

        read_stage = (read_stage + 1) % K_PIPE;
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
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    SgbNnWideParams params
) {
    assert(params.alpha == 1.0f || bias == nullptr);
    wide_nn_microtile_body<64, 128, 64, 32, 128>(C, A, B, bias, params);
}

// 128x128 block tile, eight warps in a 2x4 grid each owning 64x32. Two
// blocks per SM would cap it at 128 registers and spill, so it runs alone.
extern "C" __global__ __launch_bounds__(256, 1)
void gemm_bi_nn_m128n128_bk16_s2_micro8x8_exp(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    SgbNnWideParams params
) {
    assert(params.alpha == 1.0f || bias == nullptr);
    wide_nn_microtile_body<128, 128, 64, 32, 256>(C, A, B, bias, params);
}

// 128x64 block tile, four warps stacked each owning 32x64.
extern "C" __global__ __launch_bounds__(128, 3)
void gemm_bi_nn_m128n64_bk16_s2_micro8x8_exp(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    SgbNnWideParams params
) {
    assert(params.alpha == 1.0f || bias == nullptr);
    wide_nn_microtile_body<128, 64, 32, 64, 128>(C, A, B, bias, params);
}

#undef WIDE_NN_K_PIPE
#undef WIDE_NN_GROUP_M
#undef WIDE_NN_B_PAD
#undef WIDE_NN_A_PAD
#undef WIDE_NN_WARP_SIZE
#undef WIDE_NN_TN
#undef WIDE_NN_TM
#undef WIDE_NN_BK
