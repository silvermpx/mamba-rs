// Test-only exact-F32 NN Split-K candidate. Every CTA computes one 64x64
// output tile for one fixed 32-wide K partition. The two BK16 load stages
// feed one uninterrupted ascending 32-FFMA accumulator chain.
#define NN_SPLITK_M64_BM 64
#define NN_SPLITK_M64_BN 64
#define NN_SPLITK_M64_BK 16
#define NN_SPLITK_M64_WM 32
#define NN_SPLITK_M64_WN 32
#define NN_SPLITK_M64_TM 4
#define NN_SPLITK_M64_TN 8
#define NN_SPLITK_M64_THREADS 128
#define NN_SPLITK_M64_WARP_SIZE 32
#define NN_SPLITK_M64_A_PAD 4
#define NN_SPLITK_M64_B_PAD 4
#define NN_SPLITK_M64_K_STAGES 2
#define NN_SPLITK_M64_B_ROW_STRIDE \
    (NN_SPLITK_M64_THREADS / (NN_SPLITK_M64_BN / 4))

extern "C" __global__ __launch_bounds__(NN_SPLITK_M64_THREADS, 4)
void gemm_bi_nn_splitk32_m64n64_partial_exp(
    float* __restrict__ partial,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int M,
    int N,
    int K_CHUNKS,
    int lda
) {
    constexpr int A_STAGE =
        NN_SPLITK_M64_BK * (NN_SPLITK_M64_BM + NN_SPLITK_M64_A_PAD);
    constexpr int B_STAGE =
        NN_SPLITK_M64_BK * (NN_SPLITK_M64_BN + NN_SPLITK_M64_B_PAD);
    constexpr int SHARED_BYTES =
        NN_SPLITK_M64_K_STAGES * (A_STAGE + B_STAGE) * (int)sizeof(float);
    static_assert(SHARED_BYTES == 17408,
                  "M64N64 Split-K shared memory changed");
    static_assert(NN_SPLITK_M64_B_ROW_STRIDE == 8,
                  "M64N64 Split-K B loader changed");

    extern __shared__ __align__(16) float smem[];
    float* As = smem;
    float* Bs = smem + NN_SPLITK_M64_K_STAGES * A_STAGE;

    int num_pid_m = (M + NN_SPLITK_M64_BM - 1) / NN_SPLITK_M64_BM;
    int num_pid_n = (N + NN_SPLITK_M64_BN - 1) / NN_SPLITK_M64_BN;
    int total_mn = num_pid_m * num_pid_n;
    int pid_k = blockIdx.x / total_mn;
    int pid_mn = blockIdx.x % total_mn;
    if (pid_k >= K_CHUNKS) return;
    int pid_m = pid_mn / num_pid_n;
    int pid_n = pid_mn % num_pid_n;
    int k_offset = pid_k * NN_SPLITK_M64_K_STAGES * NN_SPLITK_M64_BK;

    int warp = threadIdx.x / NN_SPLITK_M64_WARP_SIZE;
    int lane = threadIdx.x % NN_SPLITK_M64_WARP_SIZE;
    int warp_row = warp / (NN_SPLITK_M64_BN / NN_SPLITK_M64_WN);
    int warp_col = warp % (NN_SPLITK_M64_BN / NN_SPLITK_M64_WN);
    int thread_col = lane % (NN_SPLITK_M64_WN / NN_SPLITK_M64_TN);
    int thread_row = lane / (NN_SPLITK_M64_WN / NN_SPLITK_M64_TN);
    int inner_row_b = threadIdx.x / (NN_SPLITK_M64_BN / 4);
    int inner_col_b = threadIdx.x % (NN_SPLITK_M64_BN / 4);

    float results[NN_SPLITK_M64_TM * NN_SPLITK_M64_TN] = {0.0f};
    float reg_m[NN_SPLITK_M64_TM];
    float reg_n[NN_SPLITK_M64_TN];
    float next_m[NN_SPLITK_M64_TM];
    float next_n[NN_SPLITK_M64_TN];
    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    #define ISSUE_NN_SPLITK_M64_STAGE(stage_index) do {                              \
        {                                                                             \
            constexpr int WARPS =                                                     \
                NN_SPLITK_M64_THREADS / NN_SPLITK_M64_WARP_SIZE;                      \
            constexpr int ROWS_PER_INSTRUCTION =                                      \
                NN_SPLITK_M64_WARP_SIZE / NN_SPLITK_M64_BK;                           \
            constexpr int ROWS_PER_WARP = NN_SPLITK_M64_BM / WARPS;                   \
            constexpr int INSTRUCTIONS = ROWS_PER_WARP / ROWS_PER_INSTRUCTION;        \
            int load_warp = threadIdx.x / NN_SPLITK_M64_WARP_SIZE;                    \
            int load_lane = threadIdx.x % NN_SPLITK_M64_WARP_SIZE;                    \
            int row_in_instruction = load_lane / NN_SPLITK_M64_BK;                    \
            int k_local = load_lane % NN_SPLITK_M64_BK;                               \
            _Pragma("unroll")                                                        \
            for (int instruction = 0; instruction < INSTRUCTIONS; ++instruction) {    \
                int m_local = load_warp * ROWS_PER_WARP                               \
                    + instruction * ROWS_PER_INSTRUCTION + row_in_instruction;        \
                int global_row = pid_m * NN_SPLITK_M64_BM + m_local;                  \
                int global_k = k_offset + (stage_index) * NN_SPLITK_M64_BK + k_local; \
                unsigned destination = As_base                                        \
                    + ((stage_index) * A_STAGE                                         \
                       + k_local * (NN_SPLITK_M64_BM + NN_SPLITK_M64_A_PAD)            \
                       + m_local) * (unsigned)sizeof(float);                           \
                bool valid = global_row < M;                                           \
                const float* source = valid                                            \
                    ? A + (long long)global_row * lda + global_k : A;                  \
                int source_bytes = valid ? 4 : 0;                                      \
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"         \
                             :: "r"(destination), "l"(source), "r"(source_bytes));    \
            }                                                                          \
        }                                                                              \
        for (int offset = 0; offset < NN_SPLITK_M64_BK;                               \
             offset += NN_SPLITK_M64_B_ROW_STRIDE) {                                  \
            int global_k = k_offset + (stage_index) * NN_SPLITK_M64_BK                \
                + inner_row_b + offset;                                                \
            int global_col = pid_n * NN_SPLITK_M64_BN + inner_col_b * 4;              \
            unsigned destination = Bs_base                                             \
                + ((stage_index) * B_STAGE                                             \
                   + (inner_row_b + offset) *                                          \
                         (NN_SPLITK_M64_BN + NN_SPLITK_M64_B_PAD)                      \
                   + inner_col_b * 4) * (unsigned)sizeof(float);                       \
            bool full = global_col + 3 < N && (N & 3) == 0                            \
                && gemm_bi_is_aligned_16(B);                                           \
            if (full) {                                                                \
                const float* source = B + (long long)global_k * N + global_col;         \
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"            \
                             :: "r"(destination), "l"(source));                       \
            } else {                                                                   \
                _Pragma("unroll")                                                     \
                for (int element = 0; element < 4; ++element) {                       \
                    bool valid = global_col + element < N;                             \
                    const float* source = valid                                        \
                        ? B + (long long)global_k * N + global_col + element : B;       \
                    int source_bytes = valid ? 4 : 0;                                  \
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"     \
                                 :: "r"(destination +                                  \
                                            element * (unsigned)sizeof(float)),        \
                                    "l"(source), "r"(source_bytes));                   \
                }                                                                      \
            }                                                                          \
        }                                                                              \
        asm volatile("cp.async.commit_group;\n");                                    \
    } while (0)

    ISSUE_NN_SPLITK_M64_STAGE(0);
    ISSUE_NN_SPLITK_M64_STAGE(1);
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    // The stage loop and dot loop jointly enumerate the partition as 0..31.
    #pragma unroll
    for (int stage = 0; stage < NN_SPLITK_M64_K_STAGES; ++stage) {
        const float* As_read = As + stage * A_STAGE;
        const float* Bs_read = Bs + stage * B_STAGE;
        #pragma unroll
        for (int i = 0; i < NN_SPLITK_M64_TM; ++i) {
            reg_m[i] = As_read[warp_row * NN_SPLITK_M64_WM
                + thread_row * NN_SPLITK_M64_TM + i];
        }
        #pragma unroll
        for (int i = 0; i < NN_SPLITK_M64_TN; ++i) {
            reg_n[i] = Bs_read[warp_col * NN_SPLITK_M64_WN
                + thread_col * NN_SPLITK_M64_TN + i];
        }
        #pragma unroll
        for (int dot = 0; dot < NN_SPLITK_M64_BK; ++dot) {
            if (dot + 1 < NN_SPLITK_M64_BK) {
                #pragma unroll
                for (int i = 0; i < NN_SPLITK_M64_TM; ++i) {
                    next_m[i] = As_read[(dot + 1) *
                        (NN_SPLITK_M64_BM + NN_SPLITK_M64_A_PAD)
                        + warp_row * NN_SPLITK_M64_WM
                        + thread_row * NN_SPLITK_M64_TM + i];
                }
                #pragma unroll
                for (int i = 0; i < NN_SPLITK_M64_TN; ++i) {
                    next_n[i] = Bs_read[(dot + 1) *
                        (NN_SPLITK_M64_BN + NN_SPLITK_M64_B_PAD)
                        + warp_col * NN_SPLITK_M64_WN
                        + thread_col * NN_SPLITK_M64_TN + i];
                }
            }
            #pragma unroll
            for (int rm = 0; rm < NN_SPLITK_M64_TM; ++rm) {
                #pragma unroll
                for (int rn = 0; rn < NN_SPLITK_M64_TN; ++rn) {
                    int index = rm * NN_SPLITK_M64_TN + rn;
                    results[index] = __fmaf_rn(reg_m[rm], reg_n[rn], results[index]);
                }
            }
            if (dot + 1 < NN_SPLITK_M64_BK) {
                #pragma unroll
                for (int i = 0; i < NN_SPLITK_M64_TM; ++i) reg_m[i] = next_m[i];
                #pragma unroll
                for (int i = 0; i < NN_SPLITK_M64_TN; ++i) reg_n[i] = next_n[i];
            }
        }
    }
    #undef ISSUE_NN_SPLITK_M64_STAGE

    float* partial_base = partial + (long long)pid_k * M * N;
    #pragma unroll
    for (int rm = 0; rm < NN_SPLITK_M64_TM; ++rm) {
        int global_row = pid_m * NN_SPLITK_M64_BM + warp_row * NN_SPLITK_M64_WM
            + thread_row * NN_SPLITK_M64_TM + rm;
        if (global_row >= M) continue;
        #pragma unroll
        for (int rn = 0; rn < NN_SPLITK_M64_TN; rn += 4) {
            int global_col = pid_n * NN_SPLITK_M64_BN + warp_col * NN_SPLITK_M64_WN
                + thread_col * NN_SPLITK_M64_TN + rn;
            int index = rm * NN_SPLITK_M64_TN + rn;
            float* destination = partial_base + (long long)global_row * N + global_col;
            if (global_col + 3 < N && (N & 3) == 0
                && gemm_bi_is_aligned_16(destination)) {
                float4 output = {
                    results[index], results[index + 1],
                    results[index + 2], results[index + 3]
                };
                reinterpret_cast<float4*>(destination)[0] = output;
            } else {
                #pragma unroll
                for (int element = 0; element < 4 && global_col + element < N; ++element) {
                    destination[element] = results[index + element];
                }
            }
        }
    }
}

#undef NN_SPLITK_M64_B_ROW_STRIDE
#undef NN_SPLITK_M64_K_STAGES
#undef NN_SPLITK_M64_B_PAD
#undef NN_SPLITK_M64_A_PAD
#undef NN_SPLITK_M64_WARP_SIZE
#undef NN_SPLITK_M64_THREADS
#undef NN_SPLITK_M64_TN
#undef NN_SPLITK_M64_TM
#undef NN_SPLITK_M64_WN
#undef NN_SPLITK_M64_WM
#undef NN_SPLITK_M64_BK
#undef NN_SPLITK_M64_BN
#undef NN_SPLITK_M64_BM
