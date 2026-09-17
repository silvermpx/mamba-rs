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

// Experimental exact-f32 mainloop. Every output keeps the legacy tile's
// ascending K order and FFMA chain; staging overlaps the current slab.
#define GBF_F32_THREADS 128
#define GBF_F32_S2_STAGE_ASYNC(BUF, K_TILE)                                    \
    do {                                                                       \
        unsigned _as = (unsigned)__cvta_generic_to_shared(&smem_a[(BUF)][0]);  \
        unsigned _bs = (unsigned)__cvta_generic_to_shared(&smem_b[(BUF)][0]);  \
        for (int _i = threadIdx.x; _i < BLOCK_M * (BLOCK_K / 4);               \
             _i += GBF_F32_THREADS) {                                          \
            int _r = _i / (BLOCK_K / 4);                                      \
            int _c = (_i % (BLOCK_K / 4)) * 4;                                \
            int _gr = row0 + _r;                                               \
            int _gc = (K_TILE) + _c;                                           \
            int _valid = (_gr < m) ? (k - _gc) : 0;                           \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);    \
            unsigned _dst = _as + (unsigned)((_r * BLOCK_K + _c) * 4);        \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&a[(long long)_gr * lda + _gc]                  \
                : (const void*)a;                                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));              \
        }                                                                      \
        for (int _i = threadIdx.x; _i < BLOCK_K * (BLOCK_N / 4);               \
             _i += GBF_F32_THREADS) {                                          \
            int _r = _i / (BLOCK_N / 4);                                      \
            int _c = (_i % (BLOCK_N / 4)) * 4;                                \
            int _gr = (K_TILE) + _r;                                           \
            int _gc = col0 + _c;                                               \
            int _valid = (_gr < k) ? (n - _gc) : 0;                           \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);    \
            unsigned _dst = _bs + (unsigned)((_r * BLOCK_N + _c) * 4);        \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&b[(long long)_gr * ldb + _gc]                  \
                : (const void*)b;                                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));              \
        }                                                                      \
        asm volatile("cp.async.commit_group;\n");                             \
    } while (0)

#define GBF_F32_S2_STAGE_SCALAR(BUF, K_TILE)                                   \
    do {                                                                       \
        for (int _i = threadIdx.x; _i < BLOCK_M * BLOCK_K;                    \
             _i += GBF_F32_THREADS) {                                          \
            int _r = _i / BLOCK_K;                                             \
            int _c = _i % BLOCK_K;                                             \
            int _gr = row0 + _r;                                               \
            int _gc = (K_TILE) + _c;                                           \
            smem_a[(BUF)][_i] = (_gr < m && _gc < k)                          \
                ? a[(long long)_gr * lda + _gc]                                \
                : 0.0f;                                                        \
        }                                                                      \
        for (int _i = threadIdx.x; _i < BLOCK_K * BLOCK_N;                    \
             _i += GBF_F32_THREADS) {                                          \
            int _r = _i / BLOCK_N;                                             \
            int _c = _i % BLOCK_N;                                             \
            int _gr = (K_TILE) + _r;                                           \
            int _gc = col0 + _c;                                               \
            smem_b[(BUF)][_i] = (_gr < k && _gc < n)                           \
                ? b[(long long)_gr * ldb + _gc]                                \
                : 0.0f;                                                        \
        }                                                                      \
    } while (0)

extern "C" __global__ __launch_bounds__(GBF_F32_THREADS, 2) void
f32_f32_s2(
    float* __restrict__ c,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float alpha, float beta,
    int m, int n, int k,
    int lda, int ldb, int ldc
) {
    __align__(16) __shared__ float smem_a[2][BLOCK_M * BLOCK_K];
    __align__(16) __shared__ float smem_b[2][BLOCK_K * BLOCK_N];

    int num_pid_m = (m + BLOCK_M - 1) / BLOCK_M;
    int num_pid_n = (n + BLOCK_N - 1) / BLOCK_N;
    int num_pid_in_group = GROUP_M * num_pid_n;
    int group_id = blockIdx.x / num_pid_in_group;
    int first_pid_m = group_id * GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, GROUP_M);
    int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m);
    int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;
    int row0 = pid_m * BLOCK_M;
    int col0 = pid_n * BLOCK_N;
    int tx = threadIdx.x & 15;
    int ty = threadIdx.x >> 4;
    int row_base = row0 + ty * 8;
    int col_base = col0 + tx * 4;

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    }

    int num_k_tiles = (k + BLOCK_K - 1) / BLOCK_K;
    bool fast_stage = gbf_aligned16(a) && gbf_aligned16(b)
                   && (lda & 3) == 0 && (ldb & 3) == 0;
    if (num_k_tiles > 0) {
        if (fast_stage) {
            GBF_F32_S2_STAGE_ASYNC(0, 0);
        } else {
            GBF_F32_S2_STAGE_SCALAR(0, 0);
        }
    }
    int read_buf = 0;
    for (int kt = 0; kt < num_k_tiles; kt++) {
        if (fast_stage) asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        int next_k = (kt + 1) * BLOCK_K;
        if (kt + 1 < num_k_tiles) {
            if (fast_stage) {
                GBF_F32_S2_STAGE_ASYNC(read_buf ^ 1, next_k);
            } else {
                GBF_F32_S2_STAGE_SCALAR(read_buf ^ 1, next_k);
            }
        }

#pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk++) {
            float a_reg[8];
            float b_reg[4];
#pragma unroll
            for (int i = 0; i < 8; i++)
                a_reg[i] = smem_a[read_buf][(ty * 8 + i) * BLOCK_K + kk];
#pragma unroll
            for (int j = 0; j < 4; j++)
                b_reg[j] = smem_b[read_buf][kk * BLOCK_N + tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] = __fmaf_rn(a_reg[i], b_reg[j], acc[i][j]);
            }
        }
        read_buf ^= 1;
    }

    bool pair_store_fast = row0 <= m - BLOCK_M
        && col0 <= n - BLOCK_N
        && (reinterpret_cast<unsigned long long>(c) & 7ULL) == 0
        && (ldc & 1) == 0;
    if (pair_store_fast) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int r = row_base + i;
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                int col = col_base + j;
                float val0 = __fmul_rn(alpha, acc[i][j]);
                float val1 = __fmul_rn(alpha, acc[i][j + 1]);
                if (bias != nullptr) {
                    val0 = __fadd_rn(val0, bias[col]);
                    val1 = __fadd_rn(val1, bias[col + 1]);
                }
                if (beta != 0.0f) {
                    val0 = __fmaf_rn(beta, c[(long long)r * ldc + col], val0);
                    val1 = __fmaf_rn(beta, c[(long long)r * ldc + col + 1], val1);
                }
                float2 pair = {val0, val1};
                *reinterpret_cast<float2*>(c + (long long)r * ldc + col) = pair;
            }
        }
        return;
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = row_base + i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col_base + j;
            if (col >= n) continue;
            float val = __fmul_rn(alpha, acc[i][j]);
            if (bias != nullptr) val = __fadd_rn(val, bias[col]);
            if (beta != 0.0f)
                val = __fmaf_rn(beta, c[(long long)r * ldc + col], val);
            c[(long long)r * ldc + col] = val;
        }
    }
}

#undef GBF_F32_S2_STAGE_SCALAR
#undef GBF_F32_S2_STAGE_ASYNC
#undef GBF_F32_THREADS

// BEGIN exact-f32 N128 S2 candidate
#define GBF_F32_N128_BM 64
#define GBF_F32_N128_BN 128
#define GBF_F32_N128_BK 32
#define GBF_F32_N128_STAGES 2
#define GBF_F32_N128_THREADS 256

#define GBF_F32_N128_STAGE_ASYNC(BUF, K_TILE)                                  \
    do {                                                                       \
        unsigned _as = (unsigned)__cvta_generic_to_shared(&smem_a[(BUF)][0]);  \
        unsigned _bs = (unsigned)__cvta_generic_to_shared(&smem_b[(BUF)][0]);  \
        for (int _i = threadIdx.x;                                             \
             _i < GBF_F32_N128_BM * (GBF_F32_N128_BK / 4);                    \
             _i += GBF_F32_N128_THREADS) {                                     \
            int _r = _i / (GBF_F32_N128_BK / 4);                              \
            int _c = (_i % (GBF_F32_N128_BK / 4)) * 4;                        \
            int _gr = row0 + _r;                                               \
            int _gc = (K_TILE) + _c;                                           \
            int _valid = (_gr < m) ? (k - _gc) : 0;                           \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);    \
            unsigned _dst = _as                                                \
                + (unsigned)((_r * GBF_F32_N128_BK + _c) * 4);                \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&a[(long long)_gr * lda + _gc]                  \
                : (const void*)a;                                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));              \
        }                                                                      \
        for (int _i = threadIdx.x;                                             \
             _i < GBF_F32_N128_BK * (GBF_F32_N128_BN / 4);                    \
             _i += GBF_F32_N128_THREADS) {                                     \
            int _r = _i / (GBF_F32_N128_BN / 4);                              \
            int _c = (_i % (GBF_F32_N128_BN / 4)) * 4;                        \
            int _gr = (K_TILE) + _r;                                           \
            int _gc = col0 + _c;                                               \
            int _valid = (_gr < k) ? (n - _gc) : 0;                           \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);    \
            unsigned _dst = _bs                                                \
                + (unsigned)((_r * GBF_F32_N128_BN + _c) * 4);                \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&b[(long long)_gr * ldb + _gc]                  \
                : (const void*)b;                                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));              \
        }                                                                      \
        asm volatile("cp.async.commit_group;\n");                             \
    } while (0)

#define GBF_F32_N128_STAGE_SCALAR(BUF, K_TILE)                                 \
    do {                                                                       \
        for (int _i = threadIdx.x;                                             \
             _i < GBF_F32_N128_BM * GBF_F32_N128_BK;                          \
             _i += GBF_F32_N128_THREADS) {                                     \
            int _r = _i / GBF_F32_N128_BK;                                    \
            int _c = _i % GBF_F32_N128_BK;                                    \
            int _gr = row0 + _r;                                               \
            int _gc = (K_TILE) + _c;                                           \
            smem_a[(BUF)][_i] = (_gr < m && _gc < k)                          \
                ? a[(long long)_gr * lda + _gc]                                \
                : 0.0f;                                                        \
        }                                                                      \
        for (int _i = threadIdx.x;                                             \
             _i < GBF_F32_N128_BK * GBF_F32_N128_BN;                          \
             _i += GBF_F32_N128_THREADS) {                                     \
            int _r = _i / GBF_F32_N128_BN;                                    \
            int _c = _i % GBF_F32_N128_BN;                                    \
            int _gr = (K_TILE) + _r;                                           \
            int _gc = col0 + _c;                                               \
            smem_b[(BUF)][_i] = (_gr < k && _gc < n)                          \
                ? b[(long long)_gr * ldb + _gc]                                \
                : 0.0f;                                                        \
        }                                                                      \
    } while (0)

extern "C" __global__ __launch_bounds__(GBF_F32_N128_THREADS, 2) void
f32_f32_n128_s2(
    float* __restrict__ c,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float alpha, float beta,
    int m, int n, int k,
    int lda, int ldb, int ldc
) {
    __align__(16) __shared__ float
        smem_a[GBF_F32_N128_STAGES][GBF_F32_N128_BM * GBF_F32_N128_BK];
    __align__(16) __shared__ float
        smem_b[GBF_F32_N128_STAGES][GBF_F32_N128_BK * GBF_F32_N128_BN];

    int num_pid_m = (m + GBF_F32_N128_BM - 1) / GBF_F32_N128_BM;
    int num_pid_n = (n + GBF_F32_N128_BN - 1) / GBF_F32_N128_BN;
    int num_pid_in_group = GROUP_M * num_pid_n;
    int group_id = blockIdx.x / num_pid_in_group;
    int first_pid_m = group_id * GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, GROUP_M);
    int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m);
    int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;
    int row0 = pid_m * GBF_F32_N128_BM;
    int col0 = pid_n * GBF_F32_N128_BN;
    int tx = threadIdx.x & 31;
    int ty = threadIdx.x >> 5;
    int row_base = row0 + ty * 8;
    int col_base = col0 + tx * 4;

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    }

    int num_k_tiles = (k + GBF_F32_N128_BK - 1) / GBF_F32_N128_BK;
    bool fast_stage = gbf_aligned16(a) && gbf_aligned16(b)
                   && (lda & 3) == 0 && (ldb & 3) == 0;
    if (num_k_tiles > 0) {
        if (fast_stage) {
            GBF_F32_N128_STAGE_ASYNC(0, 0);
        } else {
            GBF_F32_N128_STAGE_SCALAR(0, 0);
        }
    }
    int read_buf = 0;
    for (int kt = 0; kt < num_k_tiles; kt++) {
        if (fast_stage) asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        int next_k = (kt + 1) * GBF_F32_N128_BK;
        if (kt + 1 < num_k_tiles) {
            if (fast_stage) {
                GBF_F32_N128_STAGE_ASYNC(read_buf ^ 1, next_k);
            } else {
                GBF_F32_N128_STAGE_SCALAR(read_buf ^ 1, next_k);
            }
        }

#pragma unroll 8
        for (int kk = 0; kk < GBF_F32_N128_BK; kk++) {
            float a_reg[8];
            float b_reg[4];
#pragma unroll
            for (int i = 0; i < 8; i++)
                a_reg[i] = smem_a[read_buf][(ty * 8 + i) * GBF_F32_N128_BK + kk];
#pragma unroll
            for (int j = 0; j < 4; j++)
                b_reg[j] = smem_b[read_buf][kk * GBF_F32_N128_BN + tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] = __fmaf_rn(a_reg[i], b_reg[j], acc[i][j]);
            }
        }
        read_buf ^= 1;
    }

    bool pair_store_fast = row0 <= m - GBF_F32_N128_BM
        && col0 <= n - GBF_F32_N128_BN
        && (reinterpret_cast<unsigned long long>(c) & 7ULL) == 0
        && (ldc & 1) == 0;
    if (pair_store_fast) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int r = row_base + i;
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                int col = col_base + j;
                float val0 = __fmul_rn(alpha, acc[i][j]);
                float val1 = __fmul_rn(alpha, acc[i][j + 1]);
                if (bias != nullptr) {
                    val0 = __fadd_rn(val0, bias[col]);
                    val1 = __fadd_rn(val1, bias[col + 1]);
                }
                if (beta != 0.0f) {
                    val0 = __fmaf_rn(beta, c[(long long)r * ldc + col], val0);
                    val1 = __fmaf_rn(beta, c[(long long)r * ldc + col + 1], val1);
                }
                float2 pair = {val0, val1};
                *reinterpret_cast<float2*>(c + (long long)r * ldc + col) = pair;
            }
        }
        return;
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = row_base + i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col_base + j;
            if (col >= n) continue;
            float val = __fmul_rn(alpha, acc[i][j]);
            if (bias != nullptr) val = __fadd_rn(val, bias[col]);
            if (beta != 0.0f)
                val = __fmaf_rn(beta, c[(long long)r * ldc + col], val);
            c[(long long)r * ldc + col] = val;
        }
    }
}

#undef GBF_F32_N128_STAGE_SCALAR
#undef GBF_F32_N128_STAGE_ASYNC
#undef GBF_F32_N128_THREADS
#undef GBF_F32_N128_STAGES
#undef GBF_F32_N128_BK
#undef GBF_F32_N128_BN
#undef GBF_F32_N128_BM
// END exact-f32 N128 S2 candidate

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
