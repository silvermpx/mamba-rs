#define DEFINE_GEMM_BI_TC(NAME, T_IO, T_OUT, FROM_F_OUT, ZERO_IO)               \
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
    /* GROUP_M swizzle for L2 locality (vLLM recipe). */                        \
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
    int warp_id = threadIdx.x / 32;                                             \
    int warp_m = warp_id / WARPS_N;                                             \
    int warp_n = warp_id % WARPS_N;                                             \
                                                                                \
    /* f32 accumulator fragments — one per (warp_m row, warp_n col0/col1). */   \
    wmma::fragment<wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, float> acc_frag[WARP_FRAGS_N]; \
    _Pragma("unroll")                                                           \
    for (int j = 0; j < WARP_FRAGS_N; j++) wmma::fill_fragment(acc_frag[j], 0.0f); \
                                                                                \
    for (int k_tile = 0; k_tile < k; k_tile += BLOCK_K) {                       \
        /* Cooperatively load A tile [BLOCK_M, BLOCK_K] = 2048 elems / 256 threads = 8/thread */ \
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
        /* Cooperatively load B tile [BLOCK_K, BLOCK_N] = 2048 elems / 256 threads = 8/thread */ \
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
        /* Inner K loop: 2 frag-K iterations of 16 each. Tensor Core MMA.   */  \
        /* Reduction order is fixed (kk = 0, 1) and per-output independent  */  \
        /* of M — preserves cross-batch bit-identity.                       */  \
        _Pragma("unroll")                                                       \
        for (int kk = 0; kk < K_TILES; kk++) {                                  \
            wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K,              \
                T_IO, wmma::row_major> a_frag;                                  \
            wmma::load_matrix_sync(                                             \
                a_frag,                                                         \
                &smem_a[(warp_m * FRAG_M) * BLOCK_K + kk * FRAG_K],             \
                BLOCK_K);                                                       \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < WARP_FRAGS_N; j++) {                            \
                wmma::fragment<wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K,          \
                    T_IO, wmma::row_major> b_frag;                              \
                wmma::load_matrix_sync(                                         \
                    b_frag,                                                     \
                    &smem_b[(kk * FRAG_K) * BLOCK_N                             \
                            + warp_n * (FRAG_N * WARP_FRAGS_N) + j * FRAG_N],   \
                    BLOCK_N);                                                   \
                wmma::mma_sync(acc_frag[j], a_frag, b_frag, acc_frag[j]);       \
            }                                                                   \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
                                                                                \
    /* Epilogue. Stage f32 accumulator to a per-warp smem tile, then each   */  \
    /* thread does scalar (alpha, beta, bias) + cast and writes one element */  \
    /* of C. Reusing smem_a (>= 2048 f32 elements when sizeof(T_IO)>=2) for */  \
    /* the staging buffer; only valid when BLOCK_M*BLOCK_K*sizeof(T_IO) >=  */  \
    /* WARPS_PER_CTA * FRAG_M * (FRAG_N * WARP_FRAGS_N) * sizeof(float) =   */  \
    /* 8*16*32*4 = 16 KB. BLOCK_M*BLOCK_K*sizeof(bf16) = 64*32*2 = 4 KB —   */  \
    /* not enough. Use a dedicated f32 staging buffer instead.              */  \
    __shared__ float smem_acc[BLOCK_M * BLOCK_N];                               \
                                                                                \
    /* Each warp stores its 2 N-fragments into the per-warp slot.           */  \
    int warp_row0 = warp_m * FRAG_M;                                            \
    int warp_col0 = warp_n * (FRAG_N * WARP_FRAGS_N);                           \
    _Pragma("unroll")                                                           \
    for (int j = 0; j < WARP_FRAGS_N; j++) {                                    \
        wmma::store_matrix_sync(                                                \
            &smem_acc[warp_row0 * BLOCK_N + warp_col0 + j * FRAG_N],            \
            acc_frag[j],                                                        \
            BLOCK_N,                                                            \
            wmma::mem_row_major);                                               \
    }                                                                           \
    __syncthreads();                                                            \
                                                                                \
    /* Scalar epilogue: 4096 elems / 256 threads = 16 per thread.           */  \
    _Pragma("unroll")                                                           \
    for (int i = 0; i < 16; i++) {                                              \
        int idx = i * THREADS + threadIdx.x;                                    \
        int local_r = idx / BLOCK_N;                                            \
        int local_c = idx % BLOCK_N;                                            \
        int r = row0 + local_r;                                                 \
        int col = col0 + local_c;                                               \
        if (r >= m || col >= n) continue;                                       \
        float val = alpha * smem_acc[local_r * BLOCK_N + local_c];              \
        if (bias != nullptr) val += bias[col];                                  \
        if (beta != 0.0f) val += beta * to_f(c[r * ldc + col]);                 \
        c[r * ldc + col] = FROM_F_OUT(val);                                     \
    }                                                                           \
}

// Tensor-Core instantiations for half-precision paths (the regression source).
DEFINE_GEMM_BI_TC(gemm_bi_bf16_bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16, zero_bf16)
DEFINE_GEMM_BI_TC(gemm_bi_f16_f16,   __half,        __half,        from_f_f16,  zero_f16)
DEFINE_GEMM_BI_TC(gemm_bi_bf16_f32,  __nv_bfloat16, float,         from_f_f32,  zero_bf16)
DEFINE_GEMM_BI_TC(gemm_bi_f16_f32,   __half,        float,         from_f_f32,  zero_f16)

// f32 path stays on CUDA cores (Tensor Cores require fp16/bf16/tf32 inputs;
// converting f32→tf32 would lose 13 mantissa bits — not acceptable for the
// f32 training path that exists specifically because the user wants exact
// f32 math). cuBLAS f32 was never the regression source.
DEFINE_GEMM_BI_FFMA(gemm_bi_f32_f32, float, float, from_f_f32, zero_f32)
