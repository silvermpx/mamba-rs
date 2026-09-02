#include "_typed_prelude.cuh"

#define PRISM_EXPERIMENT_M 4621
#define PRISM_EXPERIMENT_N 384
#define PRISM_EXPERIMENT_K 1928
#define PRISM_EXPERIMENT_PADDED_K 1936
#define PRISM_EXPERIMENT_K_TILES 121
#define PRISM_EXPERIMENT_FULL_TILES 120

struct PrismExperimentParams {
    float alpha;
    float beta;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(PrismExperimentParams) == 32,
              "prism experiment parameter ABI changed");
static_assert(alignof(PrismExperimentParams) == 4,
              "prism experiment parameter alignment changed");
static_assert(__is_standard_layout(PrismExperimentParams),
              "prism experiment parameters must remain standard layout");

template <int BN, int THREADS>
__device__ __forceinline__ void prism_issue_full_tile(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* As_buf,
    float* Bs_buf,
    int stage,
    int bk_index
) {
    constexpr int BM = 64;
    constexpr int BK = 16;
    constexpr int A_PAD = 4;
    constexpr int B_PAD = 4;
    constexpr int A_STAGE = BK * (BM + A_PAD);
    constexpr int B_STAGE = BK * (BN + B_PAD);
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS = THREADS / WARP_SIZE;
    constexpr int M_ROWS_PER_WARP_INSTRUCTION = WARP_SIZE / BK;
    constexpr int M_ROWS_PER_WARP = BM / WARPS;
    constexpr int M_INSTRUCTIONS =
        M_ROWS_PER_WARP / M_ROWS_PER_WARP_INSTRUCTION;
    constexpr int B_VECTOR_COLUMNS = BN / 4;
    constexpr int B_ROW_STRIDE = THREADS / B_VECTOR_COLUMNS;

    int pid_m = blockIdx.x;
    int pid_n = blockIdx.y;
    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);
    int load_warp = threadIdx.x / WARP_SIZE;
    int load_lane = threadIdx.x % WARP_SIZE;
    int row_in_instruction = load_lane / BK;
    int k_local = load_lane % BK;

    #pragma unroll
    for (int instruction = 0; instruction < M_INSTRUCTIONS; ++instruction) {
        int m_local = load_warp * M_ROWS_PER_WARP
            + instruction * M_ROWS_PER_WARP_INSTRUCTION
            + row_in_instruction;
        int global_row = pid_m * BM + m_local;
        int global_column = bk_index + k_local;
        unsigned destination = As_base
            + (stage * A_STAGE + k_local * (BM + A_PAD) + m_local)
                * (unsigned)sizeof(float);
        bool valid = global_row < PRISM_EXPERIMENT_M;
        const float* source = valid
            ? A + (long long)global_row * PRISM_EXPERIMENT_K + global_column
            : A;
        int source_bytes = valid ? 4 : 0;
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
            :: "r"(destination), "l"(source), "r"(source_bytes));
    }

    int inner_row = threadIdx.x / B_VECTOR_COLUMNS;
    int inner_column = threadIdx.x % B_VECTOR_COLUMNS;
    #pragma unroll
    for (int offset = 0; offset < BK; offset += B_ROW_STRIDE) {
        int reduction_row = bk_index + inner_row + offset;
        int output_column = pid_n * BN + inner_column * 4;
        unsigned destination = Bs_base
            + (stage * B_STAGE
               + (inner_row + offset) * (BN + B_PAD)
               + inner_column * 4) * (unsigned)sizeof(float);
        const float* source = B
            + (long long)reduction_row * PRISM_EXPERIMENT_N
            + output_column;
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16, 16;\n"
            :: "r"(destination), "l"(source));
    }
    asm volatile("cp.async.commit_group;\n");
}

template <int BN, int THREADS>
__device__ __forceinline__ void prism_issue_tail_tile(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* As_buf,
    float* Bs_buf,
    int stage
) {
    constexpr int BM = 64;
    constexpr int BK = 16;
    constexpr int A_PAD = 4;
    constexpr int B_PAD = 4;
    constexpr int A_STAGE = BK * (BM + A_PAD);
    constexpr int B_STAGE = BK * (BN + B_PAD);
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS = THREADS / WARP_SIZE;
    constexpr int M_ROWS_PER_WARP_INSTRUCTION = WARP_SIZE / BK;
    constexpr int M_ROWS_PER_WARP = BM / WARPS;
    constexpr int M_INSTRUCTIONS =
        M_ROWS_PER_WARP / M_ROWS_PER_WARP_INSTRUCTION;
    constexpr int B_VECTOR_COLUMNS = BN / 4;
    constexpr int B_ROW_STRIDE = THREADS / B_VECTOR_COLUMNS;
    constexpr int TAIL = PRISM_EXPERIMENT_K - PRISM_EXPERIMENT_FULL_TILES * BK;

    static_assert(TAIL == 8, "prism reduction tail changed");
    int pid_m = blockIdx.x;
    int pid_n = blockIdx.y;
    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);
    int load_warp = threadIdx.x / WARP_SIZE;
    int load_lane = threadIdx.x % WARP_SIZE;
    int row_in_instruction = load_lane / BK;
    int k_local = load_lane % BK;

    #pragma unroll
    for (int instruction = 0; instruction < M_INSTRUCTIONS; ++instruction) {
        int m_local = load_warp * M_ROWS_PER_WARP
            + instruction * M_ROWS_PER_WARP_INSTRUCTION
            + row_in_instruction;
        int global_row = pid_m * BM + m_local;
        unsigned destination = As_base
            + (stage * A_STAGE + k_local * (BM + A_PAD) + m_local)
                * (unsigned)sizeof(float);
        bool valid = global_row < PRISM_EXPERIMENT_M && k_local < TAIL;
        const float* source = valid
            ? A + (long long)global_row * PRISM_EXPERIMENT_K
                + PRISM_EXPERIMENT_FULL_TILES * BK + k_local
            : A;
        int source_bytes = valid ? 4 : 0;
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
            :: "r"(destination), "l"(source), "r"(source_bytes));
    }

    int inner_row = threadIdx.x / B_VECTOR_COLUMNS;
    int inner_column = threadIdx.x % B_VECTOR_COLUMNS;
    #pragma unroll
    for (int offset = 0; offset < BK; offset += B_ROW_STRIDE) {
        int k_local_b = inner_row + offset;
        int output_column = pid_n * BN + inner_column * 4;
        unsigned destination = Bs_base
            + (stage * B_STAGE
               + k_local_b * (BN + B_PAD)
               + inner_column * 4) * (unsigned)sizeof(float);
        bool valid = k_local_b < TAIL;
        const float* source = valid
            ? B + (long long)(PRISM_EXPERIMENT_FULL_TILES * BK + k_local_b)
                * PRISM_EXPERIMENT_N + output_column
            : B;
        int source_bytes = valid ? 16 : 0;
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
            :: "r"(destination), "l"(source), "r"(source_bytes));
    }
    asm volatile("cp.async.commit_group;\n");
}

template <int BN>
__device__ __forceinline__ void prism_compute_tile(
    const float* As_read,
    const float* Bs_read,
    int warp_row,
    int warp_column,
    int thread_row,
    int thread_column,
    float (&thread_results)[32]
) {
    constexpr int BM = 64;
    constexpr int BK = 16;
    constexpr int WM = 32;
    constexpr int WN = 32;
    constexpr int TM = 4;
    constexpr int TN = 8;
    constexpr int A_PAD = 4;
    constexpr int B_PAD = 4;
    float reg_m[TM];
    float reg_n[TN];
    float next_m[TM];
    float next_n[TN];

    #pragma unroll
    for (int row = 0; row < TM; ++row) {
        reg_m[row] = As_read[warp_row * WM + thread_row * TM + row];
    }
    #pragma unroll
    for (int column = 0; column < TN; ++column) {
        reg_n[column] = Bs_read[warp_column * WN + thread_column * TN + column];
    }

    for (int dot_index = 0; dot_index < BK; ++dot_index) {
        if (dot_index + 1 < BK) {
            #pragma unroll
            for (int row = 0; row < TM; ++row) {
                next_m[row] = As_read[(dot_index + 1) * (BM + A_PAD)
                    + warp_row * WM + thread_row * TM + row];
            }
            #pragma unroll
            for (int column = 0; column < TN; ++column) {
                next_n[column] = Bs_read[(dot_index + 1) * (BN + B_PAD)
                    + warp_column * WN + thread_column * TN + column];
            }
        }

        // This loop nest is the numerical contract: every output advances in K order.
        #pragma unroll
        for (int row = 0; row < TM; ++row) {
            #pragma unroll
            for (int column = 0; column < TN; ++column) {
                int result = row * TN + column;
                thread_results[result] = __fmaf_rn(
                    reg_m[row], reg_n[column], thread_results[result]);
            }
        }

        if (dot_index + 1 < BK) {
            #pragma unroll
            for (int row = 0; row < TM; ++row) {
                reg_m[row] = next_m[row];
            }
            #pragma unroll
            for (int column = 0; column < TN; ++column) {
                reg_n[column] = next_n[column];
            }
        }
    }
}

template <int BN, int THREADS>
__device__ __forceinline__ void prism_exact_body(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* smem
) {
    constexpr int BM = 64;
    constexpr int BK = 16;
    constexpr int WM = 32;
    constexpr int WN = 32;
    constexpr int TM = 4;
    constexpr int TN = 8;
    constexpr int A_PAD = 4;
    constexpr int B_PAD = 4;
    constexpr int A_STAGE = BK * (BM + A_PAD);
    constexpr int B_STAGE = BK * (BN + B_PAD);
    constexpr int TOTAL_SMEM_BYTES = 2 * (A_STAGE + B_STAGE) * (int)sizeof(float);
    static_assert(BN != 64 || TOTAL_SMEM_BYTES == 17408,
                  "M64N64 shared memory changed");
    static_assert(BN != 128 || TOTAL_SMEM_BYTES == 25600,
                  "M64N128 shared memory changed");
    static_assert(PRISM_EXPERIMENT_PADDED_K == PRISM_EXPERIMENT_K_TILES * BK,
                  "padded reduction changed");
    static_assert(THREADS / 32 == (BM / WM) * (BN / WN),
                  "warp grid changed");

    float* As_buf = smem;
    float* Bs_buf = smem + 2 * A_STAGE;
    int pid_m = blockIdx.x;
    int pid_n = blockIdx.y;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int warp_row = warp / (BN / WN);
    int warp_column = warp % (BN / WN);
    int thread_column = lane % (WN / TN);
    int thread_row = lane / (WN / TN);
    float thread_results[TM * TN];

    #pragma unroll
    for (int result = 0; result < TM * TN; ++result) {
        thread_results[result] = 0.0f;
    }

    prism_issue_full_tile<BN, THREADS>(A, B, As_buf, Bs_buf, 0, 0);
    int read_stage = 0;
    int write_stage = 1;
    for (int tile = 0; tile < PRISM_EXPERIMENT_FULL_TILES; ++tile) {
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (tile + 1 < PRISM_EXPERIMENT_FULL_TILES) {
            prism_issue_full_tile<BN, THREADS>(
                A, B, As_buf, Bs_buf, write_stage, (tile + 1) * BK);
        } else {
            prism_issue_tail_tile<BN, THREADS>(
                A, B, As_buf, Bs_buf, write_stage);
        }
        prism_compute_tile<BN>(
            As_buf + read_stage * A_STAGE,
            Bs_buf + read_stage * B_STAGE,
            warp_row,
            warp_column,
            thread_row,
            thread_column,
            thread_results);
        read_stage ^= 1;
        write_stage ^= 1;
    }
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    prism_compute_tile<BN>(
        As_buf + read_stage * A_STAGE,
        Bs_buf + read_stage * B_STAGE,
        warp_row,
        warp_column,
        thread_row,
        thread_column,
        thread_results);

    #pragma unroll
    for (int result_row = 0; result_row < TM; ++result_row) {
        int global_row = pid_m * BM + warp_row * WM
            + thread_row * TM + result_row;
        if (global_row >= PRISM_EXPERIMENT_M) {
            continue;
        }
        #pragma unroll
        for (int result_column = 0; result_column < TN; result_column += 4) {
            int global_column = pid_n * BN + warp_column * WN
                + thread_column * TN + result_column;
            int result = result_row * TN + result_column;
            float* destination = C
                + (long long)global_row * PRISM_EXPERIMENT_N + global_column;
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

extern "C" __global__ __launch_bounds__(128, 5)
void gemm_bi_nn_prism_m64n64_bk16_s2_occ5_experiment_v1(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__,
    PrismExperimentParams
) {
    extern __shared__ __align__(16) float smem[];
    prism_exact_body<64, 128>(C, A, B, smem);
}

extern "C" __global__ __launch_bounds__(256, 2)
void gemm_bi_nn_prism_m64n128_bk16_s2_experiment_v1(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__,
    PrismExperimentParams
) {
    extern __shared__ __align__(16) float smem[];
    prism_exact_body<128, 256>(C, A, B, smem);
}

#undef PRISM_EXPERIMENT_FULL_TILES
#undef PRISM_EXPERIMENT_K_TILES
#undef PRISM_EXPERIMENT_PADDED_K
#undef PRISM_EXPERIMENT_K
#undef PRISM_EXPERIMENT_N
#undef PRISM_EXPERIMENT_M
