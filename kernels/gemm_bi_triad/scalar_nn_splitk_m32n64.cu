// Exact qualified specialization of the deterministic M32N64/BK32 Split-K
// partial. Its output layout and ascending 32-FFMA chain match the generic
// production partial consumed by gemm_bi_splitk_reduce.
#define NN_SPLITK_EXACT_M 128
#define NN_SPLITK_EXACT_K 8192
#define NN_SPLITK_EXACT_N 128
#define NN_SPLITK_EXACT_CHUNKS 256
#define NN_SPLITK_EXACT_BM 32
#define NN_SPLITK_EXACT_BN 64
#define NN_SPLITK_EXACT_BK 32
#define NN_SPLITK_EXACT_WM 16
#define NN_SPLITK_EXACT_WN 32
#define NN_SPLITK_EXACT_TM 4
#define NN_SPLITK_EXACT_TN 4
#define NN_SPLITK_EXACT_WARP_SIZE 32
#define NN_SPLITK_EXACT_A_PAD 4
#define NN_SPLITK_EXACT_B_PAD 4
#define NN_SPLITK_EXACT_THREADS 128

extern "C" __global__ __launch_bounds__(NN_SPLITK_EXACT_THREADS, 4)
void gemm_bi_nn_splitk32_m32n64_exact_v1(
    float* __restrict__ partial,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int M,
    int N,
    int K_CHUNKS,
    int lda
) {
    constexpr int EXACT_M = NN_SPLITK_EXACT_M;
    constexpr int EXACT_K = NN_SPLITK_EXACT_K;
    constexpr int EXACT_N = NN_SPLITK_EXACT_N;
    constexpr int EXACT_CHUNKS = NN_SPLITK_EXACT_CHUNKS;
    constexpr int M_TILES = EXACT_M / NN_SPLITK_EXACT_BM;
    constexpr int N_TILES = EXACT_N / NN_SPLITK_EXACT_BN;
    constexpr int MN_TILES = M_TILES * N_TILES;
    constexpr int TOTAL_BLOCKS = EXACT_CHUNKS * MN_TILES;
    static_assert(EXACT_M == 128, "exact M changed");
    static_assert(EXACT_K == 8192, "exact K changed");
    static_assert(EXACT_N == 128, "exact N changed");
    static_assert(EXACT_CHUNKS * NN_SPLITK_EXACT_BK == EXACT_K,
                  "exact chunk census changed");
    static_assert(TOTAL_BLOCKS == 2048, "exact grid changed");

    // The uniform guard precedes every asynchronous copy and barrier.
    if (!partial || !A || !B || M != EXACT_M || N != EXACT_N
        || K_CHUNKS != EXACT_CHUNKS || lda != EXACT_K
        || blockDim.x != NN_SPLITK_EXACT_THREADS || blockDim.y != 1
        || blockDim.z != 1 || gridDim.x != TOTAL_BLOCKS || gridDim.y != 1
        || gridDim.z != 1 || !gemm_bi_is_aligned_16(partial)
        || !gemm_bi_is_aligned_16(A) || !gemm_bi_is_aligned_16(B)) {
        return;
    }

    __shared__ float As[NN_SPLITK_EXACT_BK *
                        (NN_SPLITK_EXACT_BM + NN_SPLITK_EXACT_A_PAD)];
    __shared__ float Bs[NN_SPLITK_EXACT_BK *
                        (NN_SPLITK_EXACT_BN + NN_SPLITK_EXACT_B_PAD)];
    static_assert(sizeof(As) + sizeof(Bs) == 13312,
                  "exact partial shared memory changed");

    int pid_k = (int)blockIdx.x / MN_TILES;
    int pid_mn = (int)blockIdx.x - pid_k * MN_TILES;
    int pid_m = pid_mn % M_TILES;
    int pid_n = pid_mn / M_TILES;
    int k_offset = pid_k * NN_SPLITK_EXACT_BK;

    int warp = (int)threadIdx.x / NN_SPLITK_EXACT_WARP_SIZE;
    int warp_col = warp % (NN_SPLITK_EXACT_BN / NN_SPLITK_EXACT_WN);
    int warp_row = warp / (NN_SPLITK_EXACT_BN / NN_SPLITK_EXACT_WN);
    int lane = (int)threadIdx.x % NN_SPLITK_EXACT_WARP_SIZE;
    int thread_col = lane % (NN_SPLITK_EXACT_WN / NN_SPLITK_EXACT_TN);
    int thread_row = lane / (NN_SPLITK_EXACT_WN / NN_SPLITK_EXACT_TN);
    int inner_row_a = (int)threadIdx.x / (NN_SPLITK_EXACT_BK / 4);
    int inner_col_a = (int)threadIdx.x % (NN_SPLITK_EXACT_BK / 4);
    int inner_row_b = (int)threadIdx.x / (NN_SPLITK_EXACT_BN / 4);
    int inner_col_b = (int)threadIdx.x % (NN_SPLITK_EXACT_BN / 4);

    const float* A_block = A
        + (long long)pid_m * NN_SPLITK_EXACT_BM * EXACT_K + k_offset;
    const float* B_block = B
        + (long long)k_offset * EXACT_N + pid_n * NN_SPLITK_EXACT_BN;
    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

#pragma unroll
    for (int offset = 0; offset < NN_SPLITK_EXACT_BM; offset += 16) {
        const float* source = A_block
            + (long long)(inner_row_a + offset) * EXACT_K + inner_col_a * 4;
#pragma unroll
        for (int element = 0; element < 4; ++element) {
            unsigned destination = As_base
                + ((inner_col_a * 4 + element) *
                       (NN_SPLITK_EXACT_BM + NN_SPLITK_EXACT_A_PAD)
                   + inner_row_a + offset) * (unsigned)sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                         :: "r"(destination), "l"(source + element));
        }
    }

#pragma unroll
    for (int offset = 0; offset < NN_SPLITK_EXACT_BK; offset += 8) {
        const float* source = B_block
            + (long long)(inner_row_b + offset) * EXACT_N + inner_col_b * 4;
        unsigned destination = Bs_base
            + ((inner_row_b + offset) *
                   (NN_SPLITK_EXACT_BN + NN_SPLITK_EXACT_B_PAD)
               + inner_col_b * 4) * (unsigned)sizeof(float);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                     :: "r"(destination), "l"(source));
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    float results[NN_SPLITK_EXACT_TM * NN_SPLITK_EXACT_TN] = {0.0f};
    float reg_m[2][NN_SPLITK_EXACT_TM] = {{0.0f}};
    float reg_n[2][NN_SPLITK_EXACT_TN] = {{0.0f}};

#pragma unroll
    for (int element = 0; element < NN_SPLITK_EXACT_TM; ++element) {
        reg_m[0][element] = As[
            warp_row * NN_SPLITK_EXACT_WM
            + thread_row * NN_SPLITK_EXACT_TM + element];
    }
#pragma unroll
    for (int element = 0; element < NN_SPLITK_EXACT_TN; ++element) {
        reg_n[0][element] = Bs[
            warp_col * NN_SPLITK_EXACT_WN
            + thread_col * NN_SPLITK_EXACT_TN + element];
    }

#pragma unroll
    for (int dot = 0; dot < NN_SPLITK_EXACT_BK; ++dot) {
        int current = dot & 1;
        int next = current ^ 1;
        if (dot + 1 < NN_SPLITK_EXACT_BK) {
#pragma unroll
            for (int element = 0; element < NN_SPLITK_EXACT_TM; ++element) {
                reg_m[next][element] = As[
                    (dot + 1) *
                        (NN_SPLITK_EXACT_BM + NN_SPLITK_EXACT_A_PAD)
                    + warp_row * NN_SPLITK_EXACT_WM
                    + thread_row * NN_SPLITK_EXACT_TM + element];
            }
#pragma unroll
            for (int element = 0; element < NN_SPLITK_EXACT_TN; ++element) {
                reg_n[next][element] = Bs[
                    (dot + 1) *
                        (NN_SPLITK_EXACT_BN + NN_SPLITK_EXACT_B_PAD)
                    + warp_col * NN_SPLITK_EXACT_WN
                    + thread_col * NN_SPLITK_EXACT_TN + element];
            }
        }
#pragma unroll
        for (int row = 0; row < NN_SPLITK_EXACT_TM; ++row) {
#pragma unroll
            for (int column = 0; column < NN_SPLITK_EXACT_TN; ++column) {
                int index = row * NN_SPLITK_EXACT_TN + column;
                results[index] = __fmaf_rn(
                    reg_m[current][row], reg_n[current][column], results[index]);
            }
        }
    }

    float* partial_base = partial + (long long)pid_k * EXACT_M * EXACT_N;
#pragma unroll
    for (int row = 0; row < NN_SPLITK_EXACT_TM; ++row) {
        int global_row = pid_m * NN_SPLITK_EXACT_BM
            + warp_row * NN_SPLITK_EXACT_WM
            + thread_row * NN_SPLITK_EXACT_TM + row;
#pragma unroll
        for (int column = 0; column < NN_SPLITK_EXACT_TN; column += 4) {
            int global_column = pid_n * NN_SPLITK_EXACT_BN
                + warp_col * NN_SPLITK_EXACT_WN
                + thread_col * NN_SPLITK_EXACT_TN + column;
            int index = row * NN_SPLITK_EXACT_TN + column;
            float4 output = {
                results[index + 0], results[index + 1],
                results[index + 2], results[index + 3]
            };
            reinterpret_cast<float4*>(
                &partial_base[(long long)global_row * EXACT_N + global_column])[0] = output;
        }
    }
}

#undef NN_SPLITK_EXACT_B_PAD
#undef NN_SPLITK_EXACT_A_PAD
#undef NN_SPLITK_EXACT_WARP_SIZE
#undef NN_SPLITK_EXACT_THREADS
#undef NN_SPLITK_EXACT_TN
#undef NN_SPLITK_EXACT_TM
#undef NN_SPLITK_EXACT_WN
#undef NN_SPLITK_EXACT_WM
#undef NN_SPLITK_EXACT_BK
#undef NN_SPLITK_EXACT_BN
#undef NN_SPLITK_EXACT_BM
#undef NN_SPLITK_EXACT_CHUNKS
#undef NN_SPLITK_EXACT_N
#undef NN_SPLITK_EXACT_K
#undef NN_SPLITK_EXACT_M
