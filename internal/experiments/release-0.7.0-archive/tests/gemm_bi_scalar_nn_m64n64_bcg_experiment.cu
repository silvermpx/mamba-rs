// Exact-F32 NN schedule for wide and deep projection cells.
#define BCG_NN_M64N64_BM 64
#define BCG_NN_M64N64_BN 64
#define BCG_NN_M64N64_BK 16
#define BCG_NN_M64N64_WM 32
#define BCG_NN_M64N64_WN 32
#define BCG_NN_M64N64_TM 4
#define BCG_NN_M64N64_TN 8
#define BCG_NN_M64N64_THREADS 128
#define BCG_NN_M64N64_WARP_SIZE 32
#define BCG_NN_M64N64_A_PAD 4
#define BCG_NN_M64N64_B_PAD 4
#define BCG_NN_M64N64_GROUP_M 16
#define BCG_NN_M64N64_B_ROW_STRIDE \
    (BCG_NN_M64N64_THREADS / (BCG_NN_M64N64_BN / 4))

struct SgbNnM64N64BcgParams {
    float alpha;
    float beta;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(SgbNnM64N64BcgParams) == 32,
              "M64N64 parameter ABI drift");
static_assert(alignof(SgbNnM64N64BcgParams) == 4,
              "M64N64 parameter alignment drift");
static_assert(__is_standard_layout(SgbNnM64N64BcgParams),
              "M64N64 parameters must remain standard layout");

extern "C" __global__ __launch_bounds__(BCG_NN_M64N64_THREADS, 3)
void gemm_bi_nn_m64n64_bk16_s2_bcg_exp_v1(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    SgbNnM64N64BcgParams params
) {
    assert(params.alpha == 1.0f || bias == nullptr);
    constexpr int K_PIPE = 2;
    constexpr int A_STAGE =
        BCG_NN_M64N64_BK * (BCG_NN_M64N64_BM + BCG_NN_M64N64_A_PAD);
    constexpr int B_STAGE =
        BCG_NN_M64N64_BK * (BCG_NN_M64N64_BN + BCG_NN_M64N64_B_PAD);
    constexpr int TOTAL_SMEM_BYTES =
        K_PIPE * (A_STAGE + B_STAGE) * (int)sizeof(float);
    static_assert(TOTAL_SMEM_BYTES == 17408,
                  "M64N64 shared memory changed");
    static_assert(BCG_NN_M64N64_THREADS / BCG_NN_M64N64_WARP_SIZE ==
                      (BCG_NN_M64N64_BM / BCG_NN_M64N64_WM) *
                          (BCG_NN_M64N64_BN / BCG_NN_M64N64_WN),
                  "M64N64 warp grid changed");
    static_assert(BCG_NN_M64N64_B_ROW_STRIDE == 8,
                  "M64N64 B loader changed");

    extern __shared__ __align__(16) float smem[];
    float* As_buf = smem;
    float* Bs_buf = smem + K_PIPE * A_STAGE;

    int num_pid_m = (params.m + BCG_NN_M64N64_BM - 1) / BCG_NN_M64N64_BM;
    int num_pid_n = (params.n + BCG_NN_M64N64_BN - 1) / BCG_NN_M64N64_BN;
    int num_pid_in_group = BCG_NN_M64N64_GROUP_M * num_pid_n;
    int tile_id = blockIdx.x;
    int group_id = tile_id / num_pid_in_group;
    int first_pid_m = group_id * BCG_NN_M64N64_GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, BCG_NN_M64N64_GROUP_M);
    int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
    int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    int warp = threadIdx.x / BCG_NN_M64N64_WARP_SIZE;
    int lane = threadIdx.x % BCG_NN_M64N64_WARP_SIZE;
    int warp_row = warp / (BCG_NN_M64N64_BN / BCG_NN_M64N64_WN);
    int warp_column = warp % (BCG_NN_M64N64_BN / BCG_NN_M64N64_WN);
    int thread_column = lane % (BCG_NN_M64N64_WN / BCG_NN_M64N64_TN);
    int thread_row = lane / (BCG_NN_M64N64_WN / BCG_NN_M64N64_TN);
    int inner_row_b = threadIdx.x / (BCG_NN_M64N64_BN / 4);
    int inner_column_b = threadIdx.x % (BCG_NN_M64N64_BN / 4);

    float reg_m[BCG_NN_M64N64_TM] = {0.0f};
    float reg_n[BCG_NN_M64N64_TN] = {0.0f};
    float threadResults[BCG_NN_M64N64_TM * BCG_NN_M64N64_TN];

    if (bias != nullptr) {
        #pragma unroll
        for (int result_column = 0;
             result_column < BCG_NN_M64N64_TN;
             ++result_column) {
            int global_column = pid_n * BCG_NN_M64N64_BN
                + warp_column * BCG_NN_M64N64_WN
                + thread_column * BCG_NN_M64N64_TN + result_column;
            float bias_value =
                global_column < params.n ? bias[global_column] : 0.0f;
            #pragma unroll
            for (int result_row = 0;
                 result_row < BCG_NN_M64N64_TM;
                 ++result_row) {
                threadResults[result_row * BCG_NN_M64N64_TN + result_column] =
                    bias_value;
            }
        }
    } else {
        #pragma unroll
        for (int result = 0;
             result < BCG_NN_M64N64_TM * BCG_NN_M64N64_TN;
             ++result) {
            threadResults[result] = 0.0f;
        }
    }

    float* C_warp = C
        + (pid_m * BCG_NN_M64N64_BM + warp_row * BCG_NN_M64N64_WM) * params.ldc
        + pid_n * BCG_NN_M64N64_BN + warp_column * BCG_NN_M64N64_WN;
    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    #define ISSUE_BCG_NN_M64N64_TILE(stage, bk_index) do {                              \
        {                                                                                \
            constexpr int WARPS =                                                       \
                BCG_NN_M64N64_THREADS / BCG_NN_M64N64_WARP_SIZE;                         \
            constexpr int M_ROWS_PER_WARP_INSTRUCTION =                                \
                BCG_NN_M64N64_WARP_SIZE / BCG_NN_M64N64_BK;                              \
            constexpr int M_ROWS_PER_WARP = BCG_NN_M64N64_BM / WARPS;                   \
            constexpr int INSTRUCTIONS_PER_WARP =                                      \
                M_ROWS_PER_WARP / M_ROWS_PER_WARP_INSTRUCTION;                         \
            int load_warp = threadIdx.x / BCG_NN_M64N64_WARP_SIZE;                      \
            int load_lane = threadIdx.x % BCG_NN_M64N64_WARP_SIZE;                      \
            int row_in_instruction = load_lane / BCG_NN_M64N64_BK;                      \
            int k_local = load_lane % BCG_NN_M64N64_BK;                                 \
            _Pragma("unroll")                                                         \
            for (int instruction = 0; instruction < INSTRUCTIONS_PER_WARP;             \
                 ++instruction) {                                                       \
                int m_local = load_warp * M_ROWS_PER_WARP                              \
                    + instruction * M_ROWS_PER_WARP_INSTRUCTION                        \
                    + row_in_instruction;                                               \
                int global_row = pid_m * BCG_NN_M64N64_BM + m_local;                    \
                int global_column = (bk_index) + k_local;                              \
                unsigned destination = As_base                                         \
                    + ((stage) * A_STAGE                                                \
                       + k_local * (BCG_NN_M64N64_BM + BCG_NN_M64N64_A_PAD)               \
                       + m_local) * (unsigned)sizeof(float);                            \
                bool valid =                                                           \
                    global_row < params.m && global_column < params.k;                  \
                const float* source = valid                                            \
                    ? A + (long long)global_row * params.lda + global_column            \
                    : A;                                                               \
                int source_bytes = valid ? 4 : 0;                                      \
                asm volatile(                                                          \
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"                   \
                    :: "r"(destination), "l"(source), "r"(source_bytes));              \
            }                                                                           \
        }                                                                                \
        for (int offset = 0;                                                            \
             offset + BCG_NN_M64N64_B_ROW_STRIDE <= BCG_NN_M64N64_BK;                    \
             offset += BCG_NN_M64N64_B_ROW_STRIDE) {                                     \
            int global_row = (bk_index) + inner_row_b + offset;                        \
            int global_column =                                                        \
                pid_n * BCG_NN_M64N64_BN + inner_column_b * 4;                          \
            unsigned destination = Bs_base                                             \
                + ((stage) * B_STAGE                                                    \
                   + (inner_row_b + offset) *                                           \
                         (BCG_NN_M64N64_BN + BCG_NN_M64N64_B_PAD)                         \
                   + inner_column_b * 4) * (unsigned)sizeof(float);                    \
            bool full = global_row < params.k                                          \
                && global_column + 3 < params.n                                        \
                && (params.ldb & 3) == 0 && gemm_bi_is_aligned_16(B);                  \
            if (full) {                                                                 \
                const float* source =                                                   \
                    B + (long long)global_row * params.ldb + global_column;             \
                asm volatile(                                                          \
                    "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"                  \
                    :: "r"(destination), "l"(source), "n"(16));                       \
            } else {                                                                    \
                _Pragma("unroll")                                                     \
                for (int element = 0; element < 4; ++element) {                        \
                    bool valid = global_row < params.k                                 \
                        && global_column + element < params.n;                          \
                    const float* source = valid                                        \
                        ? B + (long long)global_row * params.ldb                        \
                            + global_column + element                                  \
                        : B;                                                           \
                    int source_bytes = valid ? 4 : 0;                                  \
                    asm volatile(                                                      \
                        "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"               \
                        :: "r"(destination +                                           \
                                   element * (unsigned)sizeof(float)),                 \
                           "l"(source), "r"(source_bytes));                            \
                }                                                                       \
            }                                                                           \
        }                                                                               \
        asm volatile("cp.async.commit_group;\n");                                     \
    } while (0)

    int num_k_tiles = (params.k + BCG_NN_M64N64_BK - 1) / BCG_NN_M64N64_BK;
    ISSUE_BCG_NN_M64N64_TILE(0, 0);
    int read_stage = 0;
    int write_stage = 1;
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(K_PIPE - 2));
        __syncthreads();

        int next_tile = tile + 1;
        if (next_tile < num_k_tiles) {
            ISSUE_BCG_NN_M64N64_TILE(write_stage, next_tile * BCG_NN_M64N64_BK);
        }

        const float* As_read = As_buf + read_stage * A_STAGE;
        const float* Bs_read = Bs_buf + read_stage * B_STAGE;
        float reg_m_next[BCG_NN_M64N64_TM];
        float reg_n_next[BCG_NN_M64N64_TN];

        #pragma unroll
        for (int i = 0; i < BCG_NN_M64N64_TM; ++i) {
            reg_m[i] = As_read[warp_row * BCG_NN_M64N64_WM
                + thread_row * BCG_NN_M64N64_TM + i];
        }
        #pragma unroll
        for (int i = 0; i < BCG_NN_M64N64_TN; ++i) {
            reg_n[i] = Bs_read[warp_column * BCG_NN_M64N64_WN
                + thread_column * BCG_NN_M64N64_TN + i];
        }

        for (int dot_index = 0; dot_index < BCG_NN_M64N64_BK; ++dot_index) {
            if (dot_index + 1 < BCG_NN_M64N64_BK) {
                #pragma unroll
                for (int i = 0; i < BCG_NN_M64N64_TM; ++i) {
                    reg_m_next[i] =
                        As_read[(dot_index + 1) *
                                    (BCG_NN_M64N64_BM + BCG_NN_M64N64_A_PAD)
                            + warp_row * BCG_NN_M64N64_WM
                            + thread_row * BCG_NN_M64N64_TM + i];
                }
                #pragma unroll
                for (int i = 0; i < BCG_NN_M64N64_TN; ++i) {
                    reg_n_next[i] =
                        Bs_read[(dot_index + 1) *
                                    (BCG_NN_M64N64_BN + BCG_NN_M64N64_B_PAD)
                            + warp_column * BCG_NN_M64N64_WN
                            + thread_column * BCG_NN_M64N64_TN + i];
                }
            }

            // Keep this row-major result nest in ascending reduction order.
            #pragma unroll
            for (int result_row = 0;
                 result_row < BCG_NN_M64N64_TM;
                 ++result_row) {
                #pragma unroll
                for (int result_column = 0;
                     result_column < BCG_NN_M64N64_TN;
                     ++result_column) {
                    int idx = result_row * BCG_NN_M64N64_TN + result_column;
                    threadResults[idx] = __fmaf_rn(
                        reg_m[result_row],
                        reg_n[result_column],
                        threadResults[idx]);
                }
            }

            if (dot_index + 1 < BCG_NN_M64N64_BK) {
                #pragma unroll
                for (int i = 0; i < BCG_NN_M64N64_TM; ++i) {
                    reg_m[i] = reg_m_next[i];
                }
                #pragma unroll
                for (int i = 0; i < BCG_NN_M64N64_TN; ++i) {
                    reg_n[i] = reg_n_next[i];
                }
            }
        }

        read_stage = (read_stage + 1) % K_PIPE;
        write_stage = (write_stage + 1) % K_PIPE;
    }
    #undef ISSUE_BCG_NN_M64N64_TILE

    #pragma unroll
    for (int result_row = 0; result_row < BCG_NN_M64N64_TM; ++result_row) {
        int global_row = pid_m * BCG_NN_M64N64_BM
            + warp_row * BCG_NN_M64N64_WM
            + thread_row * BCG_NN_M64N64_TM + result_row;
        if (global_row >= params.m) {
            continue;
        }
        #pragma unroll
        for (int result_column = 0;
             result_column < BCG_NN_M64N64_TN;
             result_column += 4) {
            int global_column = pid_n * BCG_NN_M64N64_BN
                + warp_column * BCG_NN_M64N64_WN
                + thread_column * BCG_NN_M64N64_TN + result_column;
            int idx = result_row * BCG_NN_M64N64_TN + result_column;
            float* destination = &C_warp[
                (thread_row * BCG_NN_M64N64_TM + result_row) * params.ldc
                + thread_column * BCG_NN_M64N64_TN + result_column];
            if (global_column + 3 >= params.n || (params.ldc & 3) != 0
                || !gemm_bi_is_aligned_16(C)) {
                #pragma unroll
                for (int element = 0;
                     element < 4 && global_column + element < params.n;
                     ++element) {
                    float value =
                        params.alpha * threadResults[idx + element];
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

#undef BCG_NN_M64N64_B_ROW_STRIDE
#undef BCG_NN_M64N64_GROUP_M
#undef BCG_NN_M64N64_B_PAD
#undef BCG_NN_M64N64_A_PAD
#undef BCG_NN_M64N64_WARP_SIZE
#undef BCG_NN_M64N64_THREADS
#undef BCG_NN_M64N64_TN
#undef BCG_NN_M64N64_TM
#undef BCG_NN_M64N64_WN
#undef BCG_NN_M64N64_WM
#undef BCG_NN_M64N64_BK
#undef BCG_NN_M64N64_BN
#undef BCG_NN_M64N64_BM

