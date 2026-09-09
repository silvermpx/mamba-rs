/*
 * Sealed SM89 exact-F32 TN production source templates.
 *
 * The Rust owner extracts these three marked sections and expands only symbol
 * and epilogue placeholders. Keep the arithmetic bodies byte-derived from the
 * frozen Ada discovery winners.
 */

// SM89_EXACT_F32_DUAL_BEGIN
struct DualChunkParams {
    float alpha;
    int m, n, k0, k1, lda, ldb, ldc;
};
static_assert(sizeof(DualChunkParams) == 32, "dual-chunk parameter size");
static_assert(alignof(DualChunkParams) == 4, "dual-chunk parameter alignment");

#define DUAL_BM 64
#define DUAL_BN 64
#define DUAL_BK 32
#define DUAL_GROUP_M 8
#define DUAL_THREADS 128

__device__ __forceinline__ bool dual_aligned16(const void* pointer) {
    return (reinterpret_cast<unsigned long long>(pointer) & 15ULL) == 0ULL;
}

#define DUAL_GENERIC_ASYNC(BUF, K_TILE)                                      \
    do {                                                                      \
        unsigned _as =                                                        \
            (unsigned)__cvta_generic_to_shared(&smem_a[(BUF)][0]);            \
        unsigned _bs =                                                        \
            (unsigned)__cvta_generic_to_shared(&smem_b[(BUF)][0]);            \
        for (int _i = threadIdx.x; _i < DUAL_BM * (DUAL_BK / 4);             \
             _i += DUAL_THREADS) {                                            \
            int _r = _i / (DUAL_BK / 4);                                     \
            int _c = (_i % (DUAL_BK / 4)) * 4;                               \
            int _gr = row0 + _r;                                              \
            int _gc = (K_TILE) + _c;                                         \
            int _valid = (_gr < m) ? (chain_k - _gc) : 0;                    \
            int _bytes =                                                      \
                _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);            \
            unsigned _dst =                                                   \
                _as + (unsigned)((_r * DUAL_BK + _c) * 4);                   \
            const void* _src = _bytes > 0                                    \
                ? (const void*)&chain_a[(long long)_gr * lda + _gc]           \
                : (const void*)chain_a;                                       \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"  \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));             \
        }                                                                     \
        for (int _i = threadIdx.x; _i < DUAL_BK * (DUAL_BN / 4);             \
             _i += DUAL_THREADS) {                                            \
            int _r = _i / (DUAL_BN / 4);                                     \
            int _c = (_i % (DUAL_BN / 4)) * 4;                               \
            int _gr = (K_TILE) + _r;                                         \
            int _gc = col0 + _c;                                              \
            int _valid = (_gr < chain_k) ? (n - _gc) : 0;                    \
            int _bytes =                                                      \
                _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);            \
            unsigned _dst =                                                   \
                _bs + (unsigned)((_r * DUAL_BN + _c) * 4);                   \
            const void* _src = _bytes > 0                                    \
                ? (const void*)&chain_b[(long long)_gr * ldb + _gc]           \
                : (const void*)chain_b;                                       \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"  \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));             \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define DUAL_GENERIC_SCALAR(BUF, K_TILE)                                     \
    do {                                                                      \
        for (int _i = threadIdx.x; _i < DUAL_BM * DUAL_BK;                   \
             _i += DUAL_THREADS) {                                            \
            int _r = _i / DUAL_BK;                                            \
            int _c = _i % DUAL_BK;                                            \
            int _gr = row0 + _r;                                              \
            int _gc = (K_TILE) + _c;                                         \
            smem_a[(BUF)][_i] = (_gr < m && _gc < chain_k)                   \
                ? chain_a[(long long)_gr * lda + _gc]                         \
                : 0.0f;                                                       \
        }                                                                     \
        for (int _i = threadIdx.x; _i < DUAL_BK * DUAL_BN;                   \
             _i += DUAL_THREADS) {                                            \
            int _r = _i / DUAL_BN;                                            \
            int _c = _i % DUAL_BN;                                            \
            int _gr = (K_TILE) + _r;                                         \
            int _gc = col0 + _c;                                              \
            smem_b[(BUF)][_i] = (_gr < chain_k && _gc < n)                   \
                ? chain_b[(long long)_gr * ldb + _gc]                         \
                : 0.0f;                                                       \
        }                                                                     \
    } while (0)

#define DUAL_PLANNED_STAGE(BUF)                                               \
    do {                                                                      \
        unsigned _stage =                                                     \
            (unsigned)(BUF) * (DUAL_BM * DUAL_BK * 4);                        \
        _Pragma("unroll")                                                     \
        for (int _i = 0; _i < 4; ++_i) {                                     \
            unsigned _dst =                                                   \
                chain_a_destination + _stage + (unsigned)(_i * 2048);         \
            unsigned long long _address = chain_a_next                        \
                + (unsigned long long)_i * chain_a_vector_stride;             \
            unsigned long long _source = chain_a_bytes[_i] > 0                \
                ? _address                                                     \
                : reinterpret_cast<unsigned long long>(chain_a);              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"  \
                         :: "r"(_dst), "l"(_source),                         \
                            "r"(chain_a_bytes[_i]));                          \
        }                                                                     \
        _Pragma("unroll")                                                     \
        for (int _i = 0; _i < 4; ++_i) {                                     \
            unsigned _dst =                                                   \
                chain_b_destination + _stage + (unsigned)(_i * 2048);         \
            unsigned long long _address = chain_b_next                        \
                + (unsigned long long)_i * chain_b_vector_stride;             \
            unsigned long long _source = chain_b_bytes > 0                    \
                ? _address                                                     \
                : reinterpret_cast<unsigned long long>(chain_b);              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"  \
                         :: "r"(_dst), "l"(_source), "r"(chain_b_bytes));    \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define DUAL_RUN_CHAIN(ACC, A_PTR, B_PTR, K_LEN)                              \
    do {                                                                      \
        _Pragma("unroll")                                                     \
        for (int _i = 0; _i < 8; ++_i) {                                     \
            _Pragma("unroll")                                                 \
            for (int _j = 0; _j < 4; ++_j) (ACC)[_i][_j] = 0.0f;             \
        }                                                                     \
        const float* chain_a = (A_PTR);                                        \
        const float* chain_b = (B_PTR);                                        \
        const int chain_k = (K_LEN);                                           \
        const int chain_num_k_tiles = (chain_k + DUAL_BK - 1) / DUAL_BK;      \
        const bool chain_fast_stage = dual_aligned16(chain_a)                  \
            && dual_aligned16(chain_b) && (lda & 3) == 0 && (ldb & 3) == 0;   \
        if (chain_num_k_tiles > 0 && chain_fast_stage                          \
            && (chain_k & (DUAL_BK - 1)) == 0) {                              \
            const int chain_copy_a_row = tid / (DUAL_BK / 4);                 \
            const int chain_copy_a_col = (tid % (DUAL_BK / 4)) * 4;           \
            const int chain_copy_b_row = tid / (DUAL_BN / 4);                 \
            const int chain_copy_b_col =                                      \
                col0 + (tid % (DUAL_BN / 4)) * 4;                             \
            int chain_a_bytes[4];                                              \
            _Pragma("unroll")                                                 \
            for (int _i = 0; _i < 4; ++_i)                                   \
                chain_a_bytes[_i] =                                           \
                    row0 + chain_copy_a_row + _i * 16 < m ? 16 : 0;           \
            const int chain_remaining_b = n - chain_copy_b_col;               \
            const int chain_b_bytes = chain_remaining_b >= 4                  \
                ? 16                                                           \
                : (chain_remaining_b > 0 ? chain_remaining_b * 4 : 0);        \
            const unsigned chain_a_destination =                              \
                (unsigned)__cvta_generic_to_shared(&smem_a[0][0])             \
                + (unsigned)(tid * 16);                                       \
            const unsigned chain_b_destination =                              \
                (unsigned)__cvta_generic_to_shared(&smem_b[0][0])             \
                + (unsigned)(tid * 16);                                       \
            const unsigned long long chain_a_vector_stride =                  \
                (unsigned long long)lda * 64ULL;                              \
            const unsigned long long chain_b_vector_stride =                  \
                (unsigned long long)ldb * 32ULL;                              \
            const unsigned long long chain_b_slab_stride =                    \
                chain_b_vector_stride * 4ULL;                                 \
            unsigned long long chain_a_next =                                 \
                reinterpret_cast<unsigned long long>(chain_a)                 \
                + ((unsigned long long)(row0 + chain_copy_a_row)              \
                       * (unsigned long long)lda                               \
                   + (unsigned long long)chain_copy_a_col)                    \
                    * 4ULL;                                                    \
            unsigned long long chain_b_next =                                 \
                reinterpret_cast<unsigned long long>(chain_b)                 \
                + ((unsigned long long)chain_copy_b_row                        \
                       * (unsigned long long)ldb                               \
                   + (unsigned long long)chain_copy_b_col)                    \
                    * 4ULL;                                                    \
            DUAL_PLANNED_STAGE(0);                                             \
            chain_a_next += DUAL_BK * 4ULL;                                   \
            chain_b_next += chain_b_slab_stride;                              \
            int chain_read_buf = 0;                                            \
            for (int chain_kt = 0; chain_kt < chain_num_k_tiles; ++chain_kt) {\
                asm volatile("cp.async.wait_group 0;\n");                    \
                __syncthreads();                                               \
                if (chain_kt + 1 < chain_num_k_tiles) {                       \
                    DUAL_PLANNED_STAGE(chain_read_buf ^ 1);                    \
                    chain_a_next += DUAL_BK * 4ULL;                           \
                    chain_b_next += chain_b_slab_stride;                      \
                }                                                             \
                _Pragma("unroll")                                             \
                for (int kk = 0; kk < DUAL_BK; ++kk) {                        \
                    float a_reg[8];                                            \
                    float b_reg[4];                                            \
                    _Pragma("unroll")                                         \
                    for (int i = 0; i < 8; ++i)                               \
                        a_reg[i] = smem_a[chain_read_buf]                      \
                            [(ty * 8 + i) * DUAL_BK + kk];                     \
                    _Pragma("unroll")                                         \
                    for (int j = 0; j < 4; ++j)                               \
                        b_reg[j] = smem_b[chain_read_buf]                      \
                            [kk * DUAL_BN + tx * 4 + j];                       \
                    _Pragma("unroll")                                         \
                    for (int i = 0; i < 8; ++i) {                             \
                        _Pragma("unroll")                                     \
                        for (int j = 0; j < 4; ++j)                           \
                            (ACC)[i][j] = __fmaf_rn(                           \
                                a_reg[i], b_reg[j], (ACC)[i][j]);             \
                    }                                                         \
                }                                                             \
                chain_read_buf ^= 1;                                           \
            }                                                                 \
        } else {                                                              \
            if (chain_num_k_tiles > 0) {                                      \
                if (chain_fast_stage) DUAL_GENERIC_ASYNC(0, 0);               \
                else DUAL_GENERIC_SCALAR(0, 0);                               \
            }                                                                 \
            int chain_read_buf = 0;                                            \
            for (int chain_kt = 0; chain_kt < chain_num_k_tiles; ++chain_kt) {\
                if (chain_fast_stage)                                         \
                    asm volatile("cp.async.wait_group 0;\n");                \
                __syncthreads();                                               \
                const int chain_next_k = (chain_kt + 1) * DUAL_BK;            \
                if (chain_kt + 1 < chain_num_k_tiles) {                       \
                    if (chain_fast_stage)                                     \
                        DUAL_GENERIC_ASYNC(chain_read_buf ^ 1, chain_next_k);  \
                    else                                                      \
                        DUAL_GENERIC_SCALAR(chain_read_buf ^ 1, chain_next_k); \
                }                                                             \
                _Pragma("unroll")                                             \
                for (int kk = 0; kk < DUAL_BK; ++kk) {                        \
                    float a_reg[8];                                            \
                    float b_reg[4];                                            \
                    _Pragma("unroll")                                         \
                    for (int i = 0; i < 8; ++i)                               \
                        a_reg[i] = smem_a[chain_read_buf]                      \
                            [(ty * 8 + i) * DUAL_BK + kk];                     \
                    _Pragma("unroll")                                         \
                    for (int j = 0; j < 4; ++j)                               \
                        b_reg[j] = smem_b[chain_read_buf]                      \
                            [kk * DUAL_BN + tx * 4 + j];                       \
                    _Pragma("unroll")                                         \
                    for (int i = 0; i < 8; ++i) {                             \
                        _Pragma("unroll")                                     \
                        for (int j = 0; j < 4; ++j)                           \
                            (ACC)[i][j] = __fmaf_rn(                           \
                                a_reg[i], b_reg[j], (ACC)[i][j]);             \
                    }                                                         \
                }                                                             \
                chain_read_buf ^= 1;                                           \
            }                                                                 \
        }                                                                     \
    } while (0)

extern "C" __global__ __launch_bounds__(128, 3)
void __DUAL_SYMBOL__(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    DualChunkParams params
) {
    const float alpha = params.alpha;
    const int m = params.m, n = params.n;
    const int k0 = params.k0, k1 = params.k1;
    const int lda = params.lda, ldb = params.ldb, ldc = params.ldc;
    __align__(16) __shared__ float smem_a[2][DUAL_BM * DUAL_BK];
    __align__(16) __shared__ float smem_b[2][DUAL_BK * DUAL_BN];

    const int num_pid_m = (m + DUAL_BM - 1) / DUAL_BM;
    const int num_pid_n = (n + DUAL_BN - 1) / DUAL_BN;
    const int num_pid_in_group = DUAL_GROUP_M * num_pid_n;
    const int group_id = blockIdx.x / num_pid_in_group;
    const int first_pid_m = group_id * DUAL_GROUP_M;
    const int group_size_m = min(num_pid_m - first_pid_m, DUAL_GROUP_M);
    const int pid_m =
        first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m);
    const int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;
    const int row0 = pid_m * DUAL_BM;
    const int col0 = pid_n * DUAL_BN;
    const int tid = (int)threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int row_base = row0 + ty * 8;
    const int col_base = col0 + tx * 4;

    float acc0[8][4];
    float acc1[8][4];
    DUAL_RUN_CHAIN(acc0, a, b, k0);
    __syncthreads();
    const float* a1 = a + k0;
    const float* b1 = b + (long long)k0 * ldb;
    DUAL_RUN_CHAIN(acc1, a1, b1, k1);

__DUAL_EPILOGUE__
}

#undef DUAL_RUN_CHAIN
#undef DUAL_PLANNED_STAGE
#undef DUAL_GENERIC_SCALAR
#undef DUAL_GENERIC_ASYNC
#undef DUAL_THREADS
#undef DUAL_GROUP_M
#undef DUAL_BK
#undef DUAL_BN
#undef DUAL_BM
// SM89_EXACT_F32_DUAL_END

// SM89_EXACT_F32_FUSED_EPILOGUE_BEGIN
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = row_base + i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = col_base + j;
            if (col >= n) continue;
            const long long idx = (long long)r * ldc + col;
            const double sum = __dadd_rn((double)acc0[i][j], (double)acc1[i][j]);
            const float update = __double2float_rn(__dmul_rn((double)alpha, sum));
            output[idx] = __fadd_rn(output[idx], update);
        }
    }
// SM89_EXACT_F32_FUSED_EPILOGUE_END

// SM89_EXACT_F32_DIRECT_BEGIN
#define PRISM_DIRECT_TN_BM 64
#define PRISM_DIRECT_TN_BN 64
#define PRISM_DIRECT_TN_BK 16
#define PRISM_DIRECT_TN_THREADS 128
#define PRISM_DIRECT_TN_GROUP_M 6
#define PRISM_DIRECT_TN_A_STAGE (PRISM_DIRECT_TN_BK * PRISM_DIRECT_TN_BM)
#define PRISM_DIRECT_TN_B_STAGE (PRISM_DIRECT_TN_BK * PRISM_DIRECT_TN_BN)
#define PRISM_DIRECT_TN_SMEM_BYTES 16384
static_assert(
    2 * (PRISM_DIRECT_TN_A_STAGE + PRISM_DIRECT_TN_B_STAGE) * sizeof(float)
        == PRISM_DIRECT_TN_SMEM_BYTES,
    "direct Prism TN shared memory changed");

extern "C" __global__ __launch_bounds__(PRISM_DIRECT_TN_THREADS, 4)
void __SM89_EXACT_F32_DIRECT_SYMBOL__(
    float* __restrict__ partial,
    const float* __restrict__ x,
    const float* __restrict__ dy,
    int M_red, int K_out, int N, int M_CHUNK
) {
    if ((K_out & 3) != 0 || (N & 3) != 0
        || (reinterpret_cast<unsigned long long>(x) & 15ULL) != 0
        || (reinterpret_cast<unsigned long long>(dy) & 15ULL) != 0) return;

    const int fc = (int)blockIdx.z;
    const int m_begin = fc * M_CHUNK;
    const int m_end = min(m_begin + M_CHUNK, M_red);
    if (m_begin >= M_red) return;

    __align__(16) __shared__ float smem_a[2][PRISM_DIRECT_TN_A_STAGE];
    __align__(16) __shared__ float smem_b[2][PRISM_DIRECT_TN_B_STAGE];

    const int num_pid_m = (K_out + PRISM_DIRECT_TN_BM - 1) / PRISM_DIRECT_TN_BM;
    const int num_pid_n = (N + PRISM_DIRECT_TN_BN - 1) / PRISM_DIRECT_TN_BN;
    const int num_pid_in_group = PRISM_DIRECT_TN_GROUP_M * num_pid_n;
    const int group_id = (int)blockIdx.x / num_pid_in_group;
    const int first_pid_m = group_id * PRISM_DIRECT_TN_GROUP_M;
    const int group_size_m = min(num_pid_m - first_pid_m, PRISM_DIRECT_TN_GROUP_M);
    const int pid_m = first_pid_m
        + (((int)blockIdx.x % num_pid_in_group) % group_size_m);
    const int pid_n = ((int)blockIdx.x % num_pid_in_group) / group_size_m;
    const int row0 = pid_m * PRISM_DIRECT_TN_BM;
    const int col0 = pid_n * PRISM_DIRECT_TN_BN;
    const int tx = (int)threadIdx.x & 15;
    const int ty = (int)threadIdx.x >> 4;
    const int row_base = row0 + ty * 8;
    const int col_base = col0 + tx * 4;

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    }

    const unsigned a_base = (unsigned)__cvta_generic_to_shared(&smem_a[0][0]);
    const unsigned b_base = (unsigned)__cvta_generic_to_shared(&smem_b[0][0]);

#define PRISM_DIRECT_TN_ISSUE_STAGE(STAGE, K_TILE) do {                           \
        const unsigned _a_stage = a_base                                          \
            + (unsigned)(STAGE) * PRISM_DIRECT_TN_A_STAGE * (unsigned)sizeof(float); \
        const unsigned _b_stage = b_base                                          \
            + (unsigned)(STAGE) * PRISM_DIRECT_TN_B_STAGE * (unsigned)sizeof(float); \
        for (int _i = (int)threadIdx.x;                                           \
             _i < PRISM_DIRECT_TN_BK * (PRISM_DIRECT_TN_BM / 4);                 \
             _i += PRISM_DIRECT_TN_THREADS) {                                     \
            const int _kr = _i / (PRISM_DIRECT_TN_BM / 4);                       \
            const int _mc = (_i % (PRISM_DIRECT_TN_BM / 4)) * 4;                 \
            const int _global_r = m_begin + (K_TILE) + _kr;                      \
            const int _global_m = row0 + _mc;                                    \
            const int _remaining = K_out - _global_m;                            \
            const int _elements = _remaining >= 4 ? 4                            \
                : (_remaining > 0 ? _remaining : 0);                             \
            const int _bytes = _global_r < m_end ? _elements * 4 : 0;            \
            const unsigned _destination = _a_stage                               \
                + (unsigned)(_kr * PRISM_DIRECT_TN_BM + _mc)                     \
                    * (unsigned)sizeof(float);                                    \
            const float* _source = _bytes > 0                                    \
                ? x + (long long)_global_r * K_out + _global_m : x;              \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(_destination), "l"(_source), "r"(_bytes));     \
        }                                                                         \
        for (int _i = (int)threadIdx.x;                                           \
             _i < PRISM_DIRECT_TN_BK * (PRISM_DIRECT_TN_BN / 4);                 \
             _i += PRISM_DIRECT_TN_THREADS) {                                     \
            const int _kr = _i / (PRISM_DIRECT_TN_BN / 4);                       \
            const int _nc = (_i % (PRISM_DIRECT_TN_BN / 4)) * 4;                 \
            const int _global_r = m_begin + (K_TILE) + _kr;                      \
            const int _global_n = col0 + _nc;                                    \
            const int _remaining = N - _global_n;                                \
            const int _elements = _remaining >= 4 ? 4                            \
                : (_remaining > 0 ? _remaining : 0);                             \
            const int _bytes = _global_r < m_end ? _elements * 4 : 0;            \
            const unsigned _destination = _b_stage                               \
                + (unsigned)(_kr * PRISM_DIRECT_TN_BN + _nc)                     \
                    * (unsigned)sizeof(float);                                    \
            const float* _source = _bytes > 0                                    \
                ? dy + (long long)_global_r * N + _global_n : dy;                \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(_destination), "l"(_source), "r"(_bytes));     \
        }                                                                         \
        asm volatile("cp.async.commit_group;\n");                               \
    } while (0)

#define PRISM_DIRECT_TN_FMA_STEP(KK) do {                                         \
        float a_reg[8];                                                           \
        float b_reg[4];                                                           \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < 8; ++i)                                              \
            a_reg[i] = a_read[(KK) * PRISM_DIRECT_TN_BM + ty * 8 + i];           \
        _Pragma("unroll")                                                       \
        for (int j = 0; j < 4; ++j)                                              \
            b_reg[j] = b_read[(KK) * PRISM_DIRECT_TN_BN + tx * 4 + j];           \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < 8; ++i) {                                            \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < 4; ++j)                                         \
                acc[i][j] = __fmaf_rn(a_reg[i], b_reg[j], acc[i][j]);            \
        }                                                                         \
    } while (0)

    const int chunk_k = m_end - m_begin;
    const int num_k_tiles = (chunk_k + PRISM_DIRECT_TN_BK - 1) / PRISM_DIRECT_TN_BK;
    if (num_k_tiles > 0) PRISM_DIRECT_TN_ISSUE_STAGE(0, 0);
    int read_stage = 0;
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (tile + 1 < num_k_tiles) {
            PRISM_DIRECT_TN_ISSUE_STAGE(read_stage ^ 1, (tile + 1) * PRISM_DIRECT_TN_BK);
        }
        const float* a_read = &smem_a[read_stage][0];
        const float* b_read = &smem_b[read_stage][0];
#pragma unroll
        for (int kk = 0; kk < PRISM_DIRECT_TN_BK; ++kk) {
            PRISM_DIRECT_TN_FMA_STEP(kk);
        }
        read_stage ^= 1;
    }
#undef PRISM_DIRECT_TN_FMA_STEP
#undef PRISM_DIRECT_TN_ISSUE_STAGE

    float* partial_chunk = partial + (long long)fc * K_out * N;
    const bool vector_store = row0 <= K_out - PRISM_DIRECT_TN_BM
        && col0 <= N - PRISM_DIRECT_TN_BN
        && (reinterpret_cast<unsigned long long>(partial_chunk) & 15ULL) == 0;
    if (vector_store) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 value = {acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
            *reinterpret_cast<float4*>(
                partial_chunk + (long long)(row_base + i) * N + col_base) = value;
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row_base + i;
        if (row >= K_out) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int column = col_base + j;
            if (column < N) {
                partial_chunk[(long long)row * N + column] = acc[i][j];
            }
        }
    }
}

#undef PRISM_DIRECT_TN_BM
#undef PRISM_DIRECT_TN_BN
#undef PRISM_DIRECT_TN_BK
#undef PRISM_DIRECT_TN_THREADS
#undef PRISM_DIRECT_TN_GROUP_M
#undef PRISM_DIRECT_TN_A_STAGE
#undef PRISM_DIRECT_TN_B_STAGE
#undef PRISM_DIRECT_TN_SMEM_BYTES
// SM89_EXACT_F32_DIRECT_END
