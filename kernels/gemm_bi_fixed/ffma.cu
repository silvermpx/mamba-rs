#define DEFINE_GEMM_BI_FFMA(NAME, T_IO, T_OUT, FROM_F_OUT, ZERO_IO)             \
extern "C" __global__ __launch_bounds__(THREADS, 2) void                        \
NAME(                                                                           \
    T_OUT* __restrict__ c,                                                      \
    const T_IO* __restrict__ a,                                                 \
    const T_IO* __restrict__ b,                                                 \
    const float* __restrict__ bias,                                             \
    float alpha, float beta,                                                    \
    int m, int n, int k,                                                        \
    int lda, int ldb, int ldc                                                   \
) {                                                                             \
    __shared__ T_IO smem_a[BLOCK_M * BLOCK_K];                                  \
    __shared__ T_IO smem_b[BLOCK_K * BLOCK_N];                                  \
                                                                                \
    int num_pid_m = (m + BLOCK_M - 1) / BLOCK_M;                                \
    int num_pid_n = (n + BLOCK_N - 1) / BLOCK_N;                                \
    int num_pid_in_group = GROUP_M * num_pid_n;                                 \
    int group_id = blockIdx.x / num_pid_in_group;                               \
    int first_pid_m = group_id * GROUP_M;                                       \
    int group_size_m = min(num_pid_m - first_pid_m, GROUP_M);                   \
    int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m); \
    int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;                 \
                                                                                \
    int row0 = pid_m * BLOCK_M;                                                 \
    int col0 = pid_n * BLOCK_N;                                                 \
                                                                                \
    int tx = threadIdx.x & 15;                                                  \
    int ty = threadIdx.x >> 4;                                                  \
    int row_base = row0 + ty * 4;                                               \
    int col_base = col0 + tx * 4;                                               \
                                                                                \
    float acc[4][4];                                                            \
    _Pragma("unroll")                                                           \
    for (int i = 0; i < 4; i++) {                                               \
        _Pragma("unroll")                                                       \
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;                           \
    }                                                                           \
                                                                                \
    for (int k_tile = 0; k_tile < k; k_tile += BLOCK_K) {                       \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < 8; i++) {                                           \
            int idx = i * THREADS + threadIdx.x;                                \
            int smem_r = idx / BLOCK_K;                                         \
            int smem_c = idx % BLOCK_K;                                         \
            int g_r = row0 + smem_r;                                            \
            int g_c = k_tile + smem_c;                                          \
            smem_a[smem_r * BLOCK_K + smem_c] =                                 \
                (g_r < m && g_c < k) ? a[g_r * lda + g_c] : ZERO_IO();          \
        }                                                                       \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < 8; i++) {                                           \
            int idx = i * THREADS + threadIdx.x;                                \
            int smem_r = idx / BLOCK_N;                                         \
            int smem_c = idx % BLOCK_N;                                         \
            int g_r = k_tile + smem_r;                                          \
            int g_c = col0 + smem_c;                                            \
            smem_b[smem_r * BLOCK_N + smem_c] =                                 \
                (g_r < k && g_c < n) ? b[g_r * ldb + g_c] : ZERO_IO();          \
        }                                                                       \
        __syncthreads();                                                        \
                                                                                \
        _Pragma("unroll")                                                       \
        for (int kk = 0; kk < BLOCK_K; kk++) {                                  \
            float a_reg[4];                                                     \
            float b_reg[4];                                                     \
            _Pragma("unroll")                                                   \
            for (int i = 0; i < 4; i++)                                         \
                a_reg[i] = to_f(smem_a[(ty * 4 + i) * BLOCK_K + kk]);           \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < 4; j++)                                         \
                b_reg[j] = to_f(smem_b[kk * BLOCK_N + tx * 4 + j]);             \
            _Pragma("unroll")                                                   \
            for (int i = 0; i < 4; i++) {                                       \
                _Pragma("unroll")                                               \
                for (int j = 0; j < 4; j++) acc[i][j] += a_reg[i] * b_reg[j];   \
            }                                                                   \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
                                                                                \
    _Pragma("unroll")                                                           \
    for (int i = 0; i < 4; i++) {                                               \
        int r = row_base + i;                                                   \
        if (r >= m) continue;                                                   \
        _Pragma("unroll")                                                       \
        for (int j = 0; j < 4; j++) {                                           \
            int col = col_base + j;                                             \
            if (col >= n) continue;                                             \
            float val = __fmul_rn(alpha, acc[i][j]);                    \
            if (bias != nullptr) val = __fadd_rn(val, bias[col]);       \
            if (beta != 0.0f)                                           \
                val = __fmaf_rn(beta, to_f(c[r * ldc + col]), val);     \
            c[r * ldc + col] = FROM_F_OUT(val);                                 \
        }                                                                       \
    }                                                                           \
}

// --- Tensor Core path (bf16 / f16) --------------------------------------
// Per-warp arrangement: 8 warps as 4 (along M) × 2 (along N).
//   warp_id = threadIdx.x / 32
//   warp_m  = warp_id / WARPS_N        (0..3)
//   warp_n  = warp_id % WARPS_N        (0..1)
// Each warp owns:
//   M rows: warp_m * 16  ..  warp_m * 16 + 16   (1 frag-M)
//   N cols: warp_n * 32  ..  warp_n * 32 + 32   (2 frag-N)
// K reduction: 2 inner iterations of 16 each.
//
// Smem A is [BLOCK_M, BLOCK_K] row-major; smem B is [BLOCK_K, BLOCK_N] row-major.
// `wmma::load_matrix_sync` reads with the given leading dimension; B is
// row-major from K's perspective so we use `wmma::row_major` and ldB=BLOCK_N.
