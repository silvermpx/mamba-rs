// Test-only exact-F32 NN Split-K candidate for (128, 8192, 128).
// Each CTA owns one 32-wide reduction partition and writes one unique partial.
#define NN_SPLITK_M32N128_BM 32
#define NN_SPLITK_M32N128_BN 128
#define NN_SPLITK_M32N128_BK 32
#define NN_SPLITK_M32N128_WM 16
#define NN_SPLITK_M32N128_WN 64
#define NN_SPLITK_M32N128_TM 4
#define NN_SPLITK_M32N128_TN 8
#define NN_SPLITK_M32N128_THREADS 128
#define NN_SPLITK_M32N128_WARP_SIZE 32
#define NN_SPLITK_M32N128_A_PAD 4
#define NN_SPLITK_M32N128_B_PAD 4

static __device__ __forceinline__ bool nn_splitk_m32n128_aligned_16(
    const void* pointer
) {
    return (reinterpret_cast<unsigned long long>(pointer) & 15ULL) == 0ULL;
}

extern "C" __global__ __launch_bounds__(NN_SPLITK_M32N128_THREADS, 3)
void gemm_bi_nn_splitk32_m32n128_bk32_exp_v1(
    float* __restrict__ partial,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int M,
    int N,
    int K_CHUNKS,
    int lda
) {
    constexpr int A_STRIDE = NN_SPLITK_M32N128_BM + NN_SPLITK_M32N128_A_PAD;
    constexpr int B_STRIDE = NN_SPLITK_M32N128_BN + NN_SPLITK_M32N128_B_PAD;
    constexpr int A_ELEMENTS = NN_SPLITK_M32N128_BK * A_STRIDE;
    constexpr int B_ELEMENTS = NN_SPLITK_M32N128_BK * B_STRIDE;
    constexpr int SHARED_BYTES = (A_ELEMENTS + B_ELEMENTS) * (int)sizeof(float);
    static_assert(SHARED_BYTES == 21504,
                  "M32N128 Split-K shared memory changed");
    static_assert(NN_SPLITK_M32N128_THREADS == 128,
                  "M32N128 Split-K block size changed");
    static_assert(
        (NN_SPLITK_M32N128_BM / NN_SPLITK_M32N128_WM) *
                (NN_SPLITK_M32N128_BN / NN_SPLITK_M32N128_WN) ==
            NN_SPLITK_M32N128_THREADS / NN_SPLITK_M32N128_WARP_SIZE,
        "M32N128 Split-K warp grid changed");

    extern __shared__ __align__(16) float shared[];
    float* As = shared;
    float* Bs = shared + A_ELEMENTS;

    if (M <= 0 || N <= 0 || K_CHUNKS <= 0 || lda <= 0) {
        return;
    }

    int num_pid_m = (M + NN_SPLITK_M32N128_BM - 1) / NN_SPLITK_M32N128_BM;
    int num_pid_n = (N + NN_SPLITK_M32N128_BN - 1) / NN_SPLITK_M32N128_BN;
    int total_mn = num_pid_m * num_pid_n;
    int pid_k = blockIdx.x / total_mn;
    int pid_mn = blockIdx.x % total_mn;
    if (pid_k >= K_CHUNKS) {
        return;
    }
    int pid_m = pid_mn / num_pid_n;
    int pid_n = pid_mn % num_pid_n;
    int k_offset = pid_k * NN_SPLITK_M32N128_BK;

    int warp = threadIdx.x / NN_SPLITK_M32N128_WARP_SIZE;
    int lane = threadIdx.x % NN_SPLITK_M32N128_WARP_SIZE;
    int warp_row = warp / (NN_SPLITK_M32N128_BN / NN_SPLITK_M32N128_WN);
    int warp_column = warp % (NN_SPLITK_M32N128_BN / NN_SPLITK_M32N128_WN);
    int thread_column = lane % (NN_SPLITK_M32N128_WN / NN_SPLITK_M32N128_TN);
    int thread_row = lane / (NN_SPLITK_M32N128_WN / NN_SPLITK_M32N128_TN);

    int a_segment = threadIdx.x % (NN_SPLITK_M32N128_BK / 4);
    int a_row_group = threadIdx.x / (NN_SPLITK_M32N128_BK / 4);
    int b_segment = threadIdx.x % (NN_SPLITK_M32N128_BN / 4);
    int b_k_group = threadIdx.x / (NN_SPLITK_M32N128_BN / 4);

    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    // A is transposed into shared memory so rows are conflict-free at fixed K.
    #pragma unroll
    for (int row_offset = 0; row_offset < NN_SPLITK_M32N128_BM;
         row_offset += 16) {
        int local_row = a_row_group + row_offset;
        int global_row = pid_m * NN_SPLITK_M32N128_BM + local_row;
        int local_k_base = a_segment * 4;
        const float* source_base = global_row < M
            ? A + (long long)global_row * lda + k_offset + local_k_base
            : A;
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            unsigned destination = As_base +
                ((local_k_base + element) * A_STRIDE + local_row) *
                    (unsigned)sizeof(float);
            int source_bytes = global_row < M ? 4 : 0;
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                :: "r"(destination), "l"(source_base + element),
                   "r"(source_bytes));
        }
    }

    // B rows and both source and destination segments are contiguous 16B spans.
    #pragma unroll
    for (int k_group = 0; k_group < NN_SPLITK_M32N128_BK; k_group += 4) {
        int local_k = b_k_group + k_group;
        int global_column =
            pid_n * NN_SPLITK_M32N128_BN + b_segment * 4;
        unsigned destination = Bs_base +
            (local_k * B_STRIDE + b_segment * 4) * (unsigned)sizeof(float);
        bool full = global_column + 3 < N && (N & 3) == 0 &&
            nn_splitk_m32n128_aligned_16(B);
        if (full) {
            const float* source = B +
                (long long)(k_offset + local_k) * N + global_column;
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(destination), "l"(source));
        } else {
            #pragma unroll
            for (int element = 0; element < 4; ++element) {
                bool valid = global_column + element < N;
                const float* source = valid
                    ? B + (long long)(k_offset + local_k) * N +
                          global_column + element
                    : B;
                int source_bytes = valid ? 4 : 0;
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                    :: "r"(destination + element * (unsigned)sizeof(float)),
                       "l"(source), "r"(source_bytes));
            }
        }
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    float results[NN_SPLITK_M32N128_TM * NN_SPLITK_M32N128_TN] = {0.0f};
    float reg_m[2][NN_SPLITK_M32N128_TM];
    float reg_n[2][NN_SPLITK_M32N128_TN];

    #pragma unroll
    for (int element = 0; element < NN_SPLITK_M32N128_TM; ++element) {
        reg_m[0][element] = As[
            warp_row * NN_SPLITK_M32N128_WM +
            thread_row * NN_SPLITK_M32N128_TM + element];
    }
    #pragma unroll
    for (int element = 0; element < NN_SPLITK_M32N128_TN; ++element) {
        reg_n[0][element] = Bs[
            warp_column * NN_SPLITK_M32N128_WN +
            thread_column * NN_SPLITK_M32N128_TN + element];
    }

    #pragma unroll
    for (int dot = 0; dot < NN_SPLITK_M32N128_BK; ++dot) {
        int current = dot & 1;
        int next = current ^ 1;
        if (dot + 1 < NN_SPLITK_M32N128_BK) {
            #pragma unroll
            for (int element = 0; element < NN_SPLITK_M32N128_TM; ++element) {
                reg_m[next][element] = As[
                    (dot + 1) * A_STRIDE +
                    warp_row * NN_SPLITK_M32N128_WM +
                    thread_row * NN_SPLITK_M32N128_TM + element];
            }
            #pragma unroll
            for (int element = 0; element < NN_SPLITK_M32N128_TN; ++element) {
                reg_n[next][element] = Bs[
                    (dot + 1) * B_STRIDE +
                    warp_column * NN_SPLITK_M32N128_WN +
                    thread_column * NN_SPLITK_M32N128_TN + element];
            }
        }

        // Keep every output on one ascending 32-step FFMA chain.
        #pragma unroll
        for (int row = 0; row < NN_SPLITK_M32N128_TM; ++row) {
            #pragma unroll
            for (int column = 0; column < NN_SPLITK_M32N128_TN; ++column) {
                int index = row * NN_SPLITK_M32N128_TN + column;
                results[index] = __fmaf_rn(
                    reg_m[current][row], reg_n[current][column], results[index]);
            }
        }
    }

    float* partial_base = partial + (long long)pid_k * M * N;
    #pragma unroll
    for (int row = 0; row < NN_SPLITK_M32N128_TM; ++row) {
        int global_row = pid_m * NN_SPLITK_M32N128_BM +
            warp_row * NN_SPLITK_M32N128_WM +
            thread_row * NN_SPLITK_M32N128_TM + row;
        if (global_row >= M) {
            continue;
        }
        #pragma unroll
        for (int column = 0; column < NN_SPLITK_M32N128_TN; column += 4) {
            int global_column = pid_n * NN_SPLITK_M32N128_BN +
                warp_column * NN_SPLITK_M32N128_WN +
                thread_column * NN_SPLITK_M32N128_TN + column;
            if (global_column >= N) {
                continue;
            }
            int index = row * NN_SPLITK_M32N128_TN + column;
            float* destination = partial_base +
                (long long)global_row * N + global_column;
            bool full = global_column + 3 < N && (N & 3) == 0 &&
                nn_splitk_m32n128_aligned_16(destination);
            if (full) {
                float4 packed = {
                    results[index],
                    results[index + 1],
                    results[index + 2],
                    results[index + 3]
                };
                *reinterpret_cast<float4*>(destination) = packed;
            } else {
                #pragma unroll
                for (int element = 0; element < 4; ++element) {
                    if (global_column + element < N) {
                        destination[element] = results[index + element];
                    }
                }
            }
        }
    }
}
