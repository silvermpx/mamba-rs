// Exact-shape NN schedule for the measured d768 input projection.
#define EXACTSHAPE_NN_M64N64_M 2048
#define EXACTSHAPE_NN_M64N64_N 3072
#define EXACTSHAPE_NN_M64N64_K 768
#define EXACTSHAPE_NN_M64N64_BM 64
#define EXACTSHAPE_NN_M64N64_BN 64
#define EXACTSHAPE_NN_M64N64_BK 16
#define EXACTSHAPE_NN_M64N64_WM 32
#define EXACTSHAPE_NN_M64N64_WN 32
#define EXACTSHAPE_NN_M64N64_TM 4
#define EXACTSHAPE_NN_M64N64_TN 8
#define EXACTSHAPE_NN_M64N64_THREADS 128
#define EXACTSHAPE_NN_M64N64_WARP_SIZE 32
#define EXACTSHAPE_NN_M64N64_A_PAD 4
#define EXACTSHAPE_NN_M64N64_B_PAD 4
#define EXACTSHAPE_NN_M64N64_GROUP_M 16
#define EXACTSHAPE_NN_M64N64_B_ROW_STRIDE \
    (EXACTSHAPE_NN_M64N64_THREADS / (EXACTSHAPE_NN_M64N64_BN / 4))

extern "C" __global__ __launch_bounds__(EXACTSHAPE_NN_M64N64_THREADS, 3)
void gemm_bi_nn_m64n64_bk16_s2_exactshape_exp_v1(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B
) {
    constexpr int K_PIPE = 2;
    constexpr int A_STAGE =
        EXACTSHAPE_NN_M64N64_BK *
        (EXACTSHAPE_NN_M64N64_BM + EXACTSHAPE_NN_M64N64_A_PAD);
    constexpr int B_STAGE =
        EXACTSHAPE_NN_M64N64_BK *
        (EXACTSHAPE_NN_M64N64_BN + EXACTSHAPE_NN_M64N64_B_PAD);
    constexpr int TOTAL_SMEM_BYTES =
        K_PIPE * (A_STAGE + B_STAGE) * (int)sizeof(float);
    constexpr int NUM_PID_N = EXACTSHAPE_NN_M64N64_N / EXACTSHAPE_NN_M64N64_BN;
    constexpr int TILES_PER_GROUP = EXACTSHAPE_NN_M64N64_GROUP_M * NUM_PID_N;
    constexpr int K_TILES = EXACTSHAPE_NN_M64N64_K / EXACTSHAPE_NN_M64N64_BK;
    static_assert(TOTAL_SMEM_BYTES == 17408,
                  "exact-shape M64N64 shared memory changed");
    static_assert(EXACTSHAPE_NN_M64N64_M % EXACTSHAPE_NN_M64N64_BM == 0,
                  "exact-shape M dimension changed");
    static_assert(EXACTSHAPE_NN_M64N64_N % EXACTSHAPE_NN_M64N64_BN == 0,
                  "exact-shape N dimension changed");
    static_assert(EXACTSHAPE_NN_M64N64_K % EXACTSHAPE_NN_M64N64_BK == 0,
                  "exact-shape K dimension changed");
    static_assert(NUM_PID_N == 48 && TILES_PER_GROUP == 768 && K_TILES == 48,
                  "exact-shape launch geometry changed");
    static_assert(EXACTSHAPE_NN_M64N64_THREADS /
                          EXACTSHAPE_NN_M64N64_WARP_SIZE ==
                      (EXACTSHAPE_NN_M64N64_BM / EXACTSHAPE_NN_M64N64_WM) *
                          (EXACTSHAPE_NN_M64N64_BN /
                           EXACTSHAPE_NN_M64N64_WN),
                  "exact-shape warp grid changed");
    static_assert(EXACTSHAPE_NN_M64N64_B_ROW_STRIDE == 8,
                  "exact-shape B loader changed");

    extern __shared__ __align__(16) float smem[];
    float* As_buf = smem;
    float* Bs_buf = smem + K_PIPE * A_STAGE;

    int tile_id = blockIdx.x;
    int group_id = tile_id / TILES_PER_GROUP;
    int tile_in_group = tile_id % TILES_PER_GROUP;
    int pid_m = group_id * EXACTSHAPE_NN_M64N64_GROUP_M
        + tile_in_group % EXACTSHAPE_NN_M64N64_GROUP_M;
    int pid_n = tile_in_group / EXACTSHAPE_NN_M64N64_GROUP_M;

    int warp = threadIdx.x / EXACTSHAPE_NN_M64N64_WARP_SIZE;
    int lane = threadIdx.x % EXACTSHAPE_NN_M64N64_WARP_SIZE;
    int warp_row = warp / (EXACTSHAPE_NN_M64N64_BN / EXACTSHAPE_NN_M64N64_WN);
    int warp_column = warp % (EXACTSHAPE_NN_M64N64_BN / EXACTSHAPE_NN_M64N64_WN);
    int thread_column = lane % (EXACTSHAPE_NN_M64N64_WN / EXACTSHAPE_NN_M64N64_TN);
    int thread_row = lane / (EXACTSHAPE_NN_M64N64_WN / EXACTSHAPE_NN_M64N64_TN);
    int inner_row_b = threadIdx.x / (EXACTSHAPE_NN_M64N64_BN / 4);
    int inner_column_b = threadIdx.x % (EXACTSHAPE_NN_M64N64_BN / 4);

    float reg_m[EXACTSHAPE_NN_M64N64_TM] = {0.0f};
    float reg_n[EXACTSHAPE_NN_M64N64_TN] = {0.0f};
    float thread_results[EXACTSHAPE_NN_M64N64_TM * EXACTSHAPE_NN_M64N64_TN];
    #pragma unroll
    for (int result = 0;
         result < EXACTSHAPE_NN_M64N64_TM * EXACTSHAPE_NN_M64N64_TN;
         ++result) {
        thread_results[result] = 0.0f;
    }

    float* C_warp = C
        + (long long)(pid_m * EXACTSHAPE_NN_M64N64_BM
                      + warp_row * EXACTSHAPE_NN_M64N64_WM) *
              EXACTSHAPE_NN_M64N64_N
        + pid_n * EXACTSHAPE_NN_M64N64_BN
        + warp_column * EXACTSHAPE_NN_M64N64_WN;
    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    #define ISSUE_EXACTSHAPE_NN_M64N64_TILE(stage, bk_index) do {                    \
        {                                                                            \
            constexpr int WARPS =                                                    \
                EXACTSHAPE_NN_M64N64_THREADS / EXACTSHAPE_NN_M64N64_WARP_SIZE;      \
            constexpr int M_ROWS_PER_WARP_INSTRUCTION =                              \
                EXACTSHAPE_NN_M64N64_WARP_SIZE / EXACTSHAPE_NN_M64N64_BK;           \
            constexpr int M_ROWS_PER_WARP = EXACTSHAPE_NN_M64N64_BM / WARPS;         \
            constexpr int INSTRUCTIONS_PER_WARP =                                    \
                M_ROWS_PER_WARP / M_ROWS_PER_WARP_INSTRUCTION;                       \
            int load_warp = threadIdx.x / EXACTSHAPE_NN_M64N64_WARP_SIZE;            \
            int load_lane = threadIdx.x % EXACTSHAPE_NN_M64N64_WARP_SIZE;            \
            int row_in_instruction = load_lane / EXACTSHAPE_NN_M64N64_BK;            \
            int k_local = load_lane % EXACTSHAPE_NN_M64N64_BK;                       \
            _Pragma("unroll")                                                       \
            for (int instruction = 0; instruction < INSTRUCTIONS_PER_WARP;           \
                 ++instruction) {                                                     \
                int m_local = load_warp * M_ROWS_PER_WARP                             \
                    + instruction * M_ROWS_PER_WARP_INSTRUCTION                       \
                    + row_in_instruction;                                             \
                int global_row = pid_m * EXACTSHAPE_NN_M64N64_BM + m_local;           \
                int global_column = (bk_index) + k_local;                             \
                unsigned destination = As_base                                       \
                    + ((stage) * A_STAGE                                              \
                       + k_local *                                                    \
                             (EXACTSHAPE_NN_M64N64_BM +                               \
                              EXACTSHAPE_NN_M64N64_A_PAD)                            \
                       + m_local) * (unsigned)sizeof(float);                          \
                const float* source = A                                               \
                    + (long long)global_row * EXACTSHAPE_NN_M64N64_K                 \
                    + global_column;                                                  \
                asm volatile(                                                        \
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"                 \
                    :: "r"(destination), "l"(source), "n"(4));                     \
            }                                                                         \
        }                                                                             \
        for (int offset = 0;                                                          \
             offset + EXACTSHAPE_NN_M64N64_B_ROW_STRIDE <=                           \
                 EXACTSHAPE_NN_M64N64_BK;                                             \
             offset += EXACTSHAPE_NN_M64N64_B_ROW_STRIDE) {                          \
            int global_row = (bk_index) + inner_row_b + offset;                       \
            int global_column =                                                       \
                pid_n * EXACTSHAPE_NN_M64N64_BN + inner_column_b * 4;                \
            unsigned destination = Bs_base                                            \
                + ((stage) * B_STAGE                                                  \
                   + (inner_row_b + offset) *                                         \
                         (EXACTSHAPE_NN_M64N64_BN + EXACTSHAPE_NN_M64N64_B_PAD)       \
                   + inner_column_b * 4) * (unsigned)sizeof(float);                   \
            const float* source = B                                                   \
                + (long long)global_row * EXACTSHAPE_NN_M64N64_N                    \
                + global_column;                                                      \
            asm volatile(                                                            \
                "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"                    \
                :: "r"(destination), "l"(source), "n"(16));                        \
        }                                                                             \
        asm volatile("cp.async.commit_group;\n");                                    \
    } while (0)

    ISSUE_EXACTSHAPE_NN_M64N64_TILE(0, 0);
    int read_stage = 0;
    int write_stage = 1;
    for (int tile = 0; tile < K_TILES; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(K_PIPE - 2));
        __syncthreads();

        int next_tile = tile + 1;
        if (next_tile < K_TILES) {
            ISSUE_EXACTSHAPE_NN_M64N64_TILE(
                write_stage,
                next_tile * EXACTSHAPE_NN_M64N64_BK);
        }

        const float* As_read = As_buf + read_stage * A_STAGE;
        const float* Bs_read = Bs_buf + read_stage * B_STAGE;
        float reg_m_next[EXACTSHAPE_NN_M64N64_TM];
        float reg_n_next[EXACTSHAPE_NN_M64N64_TN];

        #pragma unroll
        for (int i = 0; i < EXACTSHAPE_NN_M64N64_TM; ++i) {
            reg_m[i] = As_read[warp_row * EXACTSHAPE_NN_M64N64_WM
                + thread_row * EXACTSHAPE_NN_M64N64_TM + i];
        }
        #pragma unroll
        for (int i = 0; i < EXACTSHAPE_NN_M64N64_TN; ++i) {
            reg_n[i] = Bs_read[warp_column * EXACTSHAPE_NN_M64N64_WN
                + thread_column * EXACTSHAPE_NN_M64N64_TN + i];
        }

        for (int dot_index = 0;
             dot_index < EXACTSHAPE_NN_M64N64_BK;
             ++dot_index) {
            if (dot_index + 1 < EXACTSHAPE_NN_M64N64_BK) {
                #pragma unroll
                for (int i = 0; i < EXACTSHAPE_NN_M64N64_TM; ++i) {
                    reg_m_next[i] =
                        As_read[(dot_index + 1) *
                                    (EXACTSHAPE_NN_M64N64_BM +
                                     EXACTSHAPE_NN_M64N64_A_PAD)
                            + warp_row * EXACTSHAPE_NN_M64N64_WM
                            + thread_row * EXACTSHAPE_NN_M64N64_TM + i];
                }
                #pragma unroll
                for (int i = 0; i < EXACTSHAPE_NN_M64N64_TN; ++i) {
                    reg_n_next[i] =
                        Bs_read[(dot_index + 1) *
                                    (EXACTSHAPE_NN_M64N64_BN +
                                     EXACTSHAPE_NN_M64N64_B_PAD)
                            + warp_column * EXACTSHAPE_NN_M64N64_WN
                            + thread_column * EXACTSHAPE_NN_M64N64_TN + i];
                }
            }

            // Keep every output on the production ascending-K chain.
            #pragma unroll
            for (int result_row = 0;
                 result_row < EXACTSHAPE_NN_M64N64_TM;
                 ++result_row) {
                #pragma unroll
                for (int result_column = 0;
                     result_column < EXACTSHAPE_NN_M64N64_TN;
                     ++result_column) {
                    int result =
                        result_row * EXACTSHAPE_NN_M64N64_TN + result_column;
                    thread_results[result] = __fmaf_rn(
                        reg_m[result_row],
                        reg_n[result_column],
                        thread_results[result]);
                }
            }

            if (dot_index + 1 < EXACTSHAPE_NN_M64N64_BK) {
                #pragma unroll
                for (int i = 0; i < EXACTSHAPE_NN_M64N64_TM; ++i) {
                    reg_m[i] = reg_m_next[i];
                }
                #pragma unroll
                for (int i = 0; i < EXACTSHAPE_NN_M64N64_TN; ++i) {
                    reg_n[i] = reg_n_next[i];
                }
            }
        }

        read_stage = (read_stage + 1) % K_PIPE;
        write_stage = (write_stage + 1) % K_PIPE;
    }
    #undef ISSUE_EXACTSHAPE_NN_M64N64_TILE

    #pragma unroll
    for (int result_row = 0;
         result_row < EXACTSHAPE_NN_M64N64_TM;
         ++result_row) {
        #pragma unroll
        for (int result_column = 0;
             result_column < EXACTSHAPE_NN_M64N64_TN;
             result_column += 4) {
            int result =
                result_row * EXACTSHAPE_NN_M64N64_TN + result_column;
            float* destination = &C_warp[
                (thread_row * EXACTSHAPE_NN_M64N64_TM + result_row) *
                    EXACTSHAPE_NN_M64N64_N
                + thread_column * EXACTSHAPE_NN_M64N64_TN + result_column];
            float4 output = {
                thread_results[result + 0],
                thread_results[result + 1],
                thread_results[result + 2],
                thread_results[result + 3]
            };
            reinterpret_cast<float4*>(destination)[0] = output;
        }
    }
}

#undef EXACTSHAPE_NN_M64N64_B_ROW_STRIDE
#undef EXACTSHAPE_NN_M64N64_GROUP_M
#undef EXACTSHAPE_NN_M64N64_B_PAD
#undef EXACTSHAPE_NN_M64N64_A_PAD
#undef EXACTSHAPE_NN_M64N64_WARP_SIZE
#undef EXACTSHAPE_NN_M64N64_THREADS
#undef EXACTSHAPE_NN_M64N64_TN
#undef EXACTSHAPE_NN_M64N64_TM
#undef EXACTSHAPE_NN_M64N64_WN
#undef EXACTSHAPE_NN_M64N64_WM
#undef EXACTSHAPE_NN_M64N64_BK
#undef EXACTSHAPE_NN_M64N64_BN
#undef EXACTSHAPE_NN_M64N64_BM
#undef EXACTSHAPE_NN_M64N64_K
#undef EXACTSHAPE_NN_M64N64_N
#undef EXACTSHAPE_NN_M64N64_M
// Exact-F32 NN schedule specialized for the qualified operand contract.
#define QUALIFIED_NN_M64N64_BM 64
#define QUALIFIED_NN_M64N64_BN 64
#define QUALIFIED_NN_M64N64_BK 16
#define QUALIFIED_NN_M64N64_WM 32
#define QUALIFIED_NN_M64N64_WN 32
#define QUALIFIED_NN_M64N64_TM 4
#define QUALIFIED_NN_M64N64_TN 8
#define QUALIFIED_NN_M64N64_THREADS 128
#define QUALIFIED_NN_M64N64_WARP_SIZE 32
#define QUALIFIED_NN_M64N64_A_PAD 4
#define QUALIFIED_NN_M64N64_B_PAD 4
#define QUALIFIED_NN_M64N64_GROUP_M 16
#define QUALIFIED_NN_M64N64_B_ROW_STRIDE \
    (QUALIFIED_NN_M64N64_THREADS / (QUALIFIED_NN_M64N64_BN / 4))

struct QualifiedNnM64N64Params {
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(QualifiedNnM64N64Params) == 24,
              "M64N64 parameter ABI drift");
static_assert(alignof(QualifiedNnM64N64Params) == 4,
              "M64N64 parameter alignment drift");
static_assert(__is_standard_layout(QualifiedNnM64N64Params),
              "M64N64 parameters must remain standard layout");

__device__ __forceinline__ void qualified_nn_m64n64_body(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    QualifiedNnM64N64Params params,
    float* smem
) {
    constexpr int K_PIPE = 2;
    constexpr int A_STAGE =
        QUALIFIED_NN_M64N64_BK * (QUALIFIED_NN_M64N64_BM + QUALIFIED_NN_M64N64_A_PAD);
    constexpr int B_STAGE =
        QUALIFIED_NN_M64N64_BK * (QUALIFIED_NN_M64N64_BN + QUALIFIED_NN_M64N64_B_PAD);
    constexpr int TOTAL_SMEM_BYTES =
        K_PIPE * (A_STAGE + B_STAGE) * (int)sizeof(float);
    static_assert(TOTAL_SMEM_BYTES == 17408,
                  "M64N64 shared memory changed");
    static_assert(QUALIFIED_NN_M64N64_THREADS / QUALIFIED_NN_M64N64_WARP_SIZE ==
                      (QUALIFIED_NN_M64N64_BM / QUALIFIED_NN_M64N64_WM) *
                          (QUALIFIED_NN_M64N64_BN / QUALIFIED_NN_M64N64_WN),
                  "M64N64 warp grid changed");
    static_assert(QUALIFIED_NN_M64N64_B_ROW_STRIDE == 8,
                  "M64N64 B loader changed");

    float* As_buf = smem;
    float* Bs_buf = smem + K_PIPE * A_STAGE;

    int num_pid_m = (params.m + QUALIFIED_NN_M64N64_BM - 1) / QUALIFIED_NN_M64N64_BM;
    int num_pid_n = (params.n + QUALIFIED_NN_M64N64_BN - 1) / QUALIFIED_NN_M64N64_BN;
    int num_pid_in_group = QUALIFIED_NN_M64N64_GROUP_M * num_pid_n;
    int tile_id = blockIdx.x;
    int group_id = tile_id / num_pid_in_group;
    int first_pid_m = group_id * QUALIFIED_NN_M64N64_GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, QUALIFIED_NN_M64N64_GROUP_M);
    int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
    int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    int warp = threadIdx.x / QUALIFIED_NN_M64N64_WARP_SIZE;
    int lane = threadIdx.x % QUALIFIED_NN_M64N64_WARP_SIZE;
    int warp_row = warp / (QUALIFIED_NN_M64N64_BN / QUALIFIED_NN_M64N64_WN);
    int warp_column = warp % (QUALIFIED_NN_M64N64_BN / QUALIFIED_NN_M64N64_WN);
    int thread_column = lane % (QUALIFIED_NN_M64N64_WN / QUALIFIED_NN_M64N64_TN);
    int thread_row = lane / (QUALIFIED_NN_M64N64_WN / QUALIFIED_NN_M64N64_TN);
    int inner_row_b = threadIdx.x / (QUALIFIED_NN_M64N64_BN / 4);
    int inner_column_b = threadIdx.x % (QUALIFIED_NN_M64N64_BN / 4);

    float reg_m[QUALIFIED_NN_M64N64_TM] = {0.0f};
    float reg_n[QUALIFIED_NN_M64N64_TN] = {0.0f};
    float threadResults[QUALIFIED_NN_M64N64_TM * QUALIFIED_NN_M64N64_TN];

    #pragma unroll
    for (int result = 0;
         result < QUALIFIED_NN_M64N64_TM * QUALIFIED_NN_M64N64_TN;
         ++result) {
        threadResults[result] = 0.0f;
    }

    float* C_warp = C
        + (pid_m * QUALIFIED_NN_M64N64_BM + warp_row * QUALIFIED_NN_M64N64_WM) * params.ldc
        + pid_n * QUALIFIED_NN_M64N64_BN + warp_column * QUALIFIED_NN_M64N64_WN;
    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    #define ISSUE_QUALIFIED_NN_M64N64_TILE(stage, bk_index) do {                              \
        {                                                                                \
            constexpr int WARPS =                                                       \
                QUALIFIED_NN_M64N64_THREADS / QUALIFIED_NN_M64N64_WARP_SIZE;                         \
            constexpr int M_ROWS_PER_WARP_INSTRUCTION =                                \
                QUALIFIED_NN_M64N64_WARP_SIZE / QUALIFIED_NN_M64N64_BK;                              \
            constexpr int M_ROWS_PER_WARP = QUALIFIED_NN_M64N64_BM / WARPS;                   \
            constexpr int INSTRUCTIONS_PER_WARP =                                      \
                M_ROWS_PER_WARP / M_ROWS_PER_WARP_INSTRUCTION;                         \
            int load_warp = threadIdx.x / QUALIFIED_NN_M64N64_WARP_SIZE;                      \
            int load_lane = threadIdx.x % QUALIFIED_NN_M64N64_WARP_SIZE;                      \
            int row_in_instruction = load_lane / QUALIFIED_NN_M64N64_BK;                      \
            int k_local = load_lane % QUALIFIED_NN_M64N64_BK;                                 \
            _Pragma("unroll")                                                         \
            for (int instruction = 0; instruction < INSTRUCTIONS_PER_WARP;             \
                 ++instruction) {                                                       \
                int m_local = load_warp * M_ROWS_PER_WARP                              \
                    + instruction * M_ROWS_PER_WARP_INSTRUCTION                        \
                    + row_in_instruction;                                               \
                int global_row = pid_m * QUALIFIED_NN_M64N64_BM + m_local;                    \
                int global_column = (bk_index) + k_local;                              \
                unsigned destination = As_base                                         \
                    + ((stage) * A_STAGE                                                \
                       + k_local * (QUALIFIED_NN_M64N64_BM + QUALIFIED_NN_M64N64_A_PAD)               \
                       + m_local) * (unsigned)sizeof(float);                            \
                bool valid =                                                           \
                    global_row < params.m;                  \
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
             offset + QUALIFIED_NN_M64N64_B_ROW_STRIDE <= QUALIFIED_NN_M64N64_BK;                    \
             offset += QUALIFIED_NN_M64N64_B_ROW_STRIDE) {                                     \
            int global_row = (bk_index) + inner_row_b + offset;                        \
            int global_column =                                                        \
                pid_n * QUALIFIED_NN_M64N64_BN + inner_column_b * 4;                          \
            unsigned destination = Bs_base                                             \
                + ((stage) * B_STAGE                                                    \
                   + (inner_row_b + offset) *                                           \
                         (QUALIFIED_NN_M64N64_BN + QUALIFIED_NN_M64N64_B_PAD)                         \
                   + inner_column_b * 4) * (unsigned)sizeof(float);                    \
            bool full = global_column + 3 < params.n;                          \
            if (full) {                                                                 \
                const float* source =                                                   \
                    B + (long long)global_row * params.ldb + global_column;             \
                asm volatile(                                                          \
                    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"                  \
                    :: "r"(destination), "l"(source), "n"(16));                       \
            } else {                                                                    \
                _Pragma("unroll")                                                     \
                for (int element = 0; element < 4; ++element) {                        \
                    bool valid = global_column + element < params.n;           \
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

    int num_k_tiles = params.k / QUALIFIED_NN_M64N64_BK;
    ISSUE_QUALIFIED_NN_M64N64_TILE(0, 0);
    int read_stage = 0;
    int write_stage = 1;
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(K_PIPE - 2));
        __syncthreads();

        int next_tile = tile + 1;
        if (next_tile < num_k_tiles) {
            ISSUE_QUALIFIED_NN_M64N64_TILE(write_stage, next_tile * QUALIFIED_NN_M64N64_BK);
        }

        const float* As_read = As_buf + read_stage * A_STAGE;
        const float* Bs_read = Bs_buf + read_stage * B_STAGE;
        float reg_m_next[QUALIFIED_NN_M64N64_TM];
        float reg_n_next[QUALIFIED_NN_M64N64_TN];

        #pragma unroll
        for (int i = 0; i < QUALIFIED_NN_M64N64_TM; ++i) {
            reg_m[i] = As_read[warp_row * QUALIFIED_NN_M64N64_WM
                + thread_row * QUALIFIED_NN_M64N64_TM + i];
        }
        #pragma unroll
        for (int i = 0; i < QUALIFIED_NN_M64N64_TN; ++i) {
            reg_n[i] = Bs_read[warp_column * QUALIFIED_NN_M64N64_WN
                + thread_column * QUALIFIED_NN_M64N64_TN + i];
        }

        for (int dot_index = 0; dot_index < QUALIFIED_NN_M64N64_BK; ++dot_index) {
            if (dot_index + 1 < QUALIFIED_NN_M64N64_BK) {
                #pragma unroll
                for (int i = 0; i < QUALIFIED_NN_M64N64_TM; ++i) {
                    reg_m_next[i] =
                        As_read[(dot_index + 1) *
                                    (QUALIFIED_NN_M64N64_BM + QUALIFIED_NN_M64N64_A_PAD)
                            + warp_row * QUALIFIED_NN_M64N64_WM
                            + thread_row * QUALIFIED_NN_M64N64_TM + i];
                }
                #pragma unroll
                for (int i = 0; i < QUALIFIED_NN_M64N64_TN; ++i) {
                    reg_n_next[i] =
                        Bs_read[(dot_index + 1) *
                                    (QUALIFIED_NN_M64N64_BN + QUALIFIED_NN_M64N64_B_PAD)
                            + warp_column * QUALIFIED_NN_M64N64_WN
                            + thread_column * QUALIFIED_NN_M64N64_TN + i];
                }
            }

            // Keep this row-major result nest in ascending reduction order.
            #pragma unroll
            for (int result_row = 0;
                 result_row < QUALIFIED_NN_M64N64_TM;
                 ++result_row) {
                #pragma unroll
                for (int result_column = 0;
                     result_column < QUALIFIED_NN_M64N64_TN;
                     ++result_column) {
                    int idx = result_row * QUALIFIED_NN_M64N64_TN + result_column;
                    threadResults[idx] = __fmaf_rn(
                        reg_m[result_row],
                        reg_n[result_column],
                        threadResults[idx]);
                }
            }

            if (dot_index + 1 < QUALIFIED_NN_M64N64_BK) {
                #pragma unroll
                for (int i = 0; i < QUALIFIED_NN_M64N64_TM; ++i) {
                    reg_m[i] = reg_m_next[i];
                }
                #pragma unroll
                for (int i = 0; i < QUALIFIED_NN_M64N64_TN; ++i) {
                    reg_n[i] = reg_n_next[i];
                }
            }
        }

        read_stage = (read_stage + 1) % K_PIPE;
        write_stage = (write_stage + 1) % K_PIPE;
    }
    #undef ISSUE_QUALIFIED_NN_M64N64_TILE

    #pragma unroll
    for (int result_row = 0; result_row < QUALIFIED_NN_M64N64_TM; ++result_row) {
        int global_row = pid_m * QUALIFIED_NN_M64N64_BM
            + warp_row * QUALIFIED_NN_M64N64_WM
            + thread_row * QUALIFIED_NN_M64N64_TM + result_row;
        if (global_row >= params.m) {
            continue;
        }
        #pragma unroll
        for (int result_column = 0;
             result_column < QUALIFIED_NN_M64N64_TN;
             result_column += 4) {
            int global_column = pid_n * QUALIFIED_NN_M64N64_BN
                + warp_column * QUALIFIED_NN_M64N64_WN
                + thread_column * QUALIFIED_NN_M64N64_TN + result_column;
            int idx = result_row * QUALIFIED_NN_M64N64_TN + result_column;
            float* destination = &C_warp[
                (thread_row * QUALIFIED_NN_M64N64_TM + result_row) * params.ldc
                + thread_column * QUALIFIED_NN_M64N64_TN + result_column];
            if (global_column + 3 >= params.n) {
                #pragma unroll
                for (int element = 0;
                     element < 4 && global_column + element < params.n;
                     ++element) {
                    destination[element] = threadResults[idx + element];
                }
                continue;
            }

            float4 output = {
                threadResults[idx + 0],
                threadResults[idx + 1],
                threadResults[idx + 2],
                threadResults[idx + 3]
            };
            reinterpret_cast<float4*>(destination)[0] = output;
        }
    }
}


extern "C" __global__ __launch_bounds__(QUALIFIED_NN_M64N64_THREADS, 3)
void gemm_bi_nn_m64n64_bk16_s2_fixed_semantics_lb3_exp_v1(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    QualifiedNnM64N64Params params
) {
    extern __shared__ __align__(16) float smem[];
    qualified_nn_m64n64_body(C, A, B, params, smem);
}

extern "C" __global__ __launch_bounds__(QUALIFIED_NN_M64N64_THREADS, 4)
void gemm_bi_nn_m64n64_bk16_s2_fixed_semantics_lb4_exp_v1(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    QualifiedNnM64N64Params params
) {
    extern __shared__ __align__(16) float smem[];
    qualified_nn_m64n64_body(C, A, B, params, smem);
}

#undef QUALIFIED_NN_M64N64_B_ROW_STRIDE
#undef QUALIFIED_NN_M64N64_GROUP_M
#undef QUALIFIED_NN_M64N64_B_PAD
#undef QUALIFIED_NN_M64N64_A_PAD
#undef QUALIFIED_NN_M64N64_WARP_SIZE
#undef QUALIFIED_NN_M64N64_THREADS
#undef QUALIFIED_NN_M64N64_TN
#undef QUALIFIED_NN_M64N64_TM
#undef QUALIFIED_NN_M64N64_WN
#undef QUALIFIED_NN_M64N64_WM
#undef QUALIFIED_NN_M64N64_BK
#undef QUALIFIED_NN_M64N64_BN
#undef QUALIFIED_NN_M64N64_BM
