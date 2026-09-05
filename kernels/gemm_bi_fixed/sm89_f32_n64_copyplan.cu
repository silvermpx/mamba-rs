// Ada-only exact F32 N64 copy-plan extension. The immutable standalone
// 713a575 candidate supplies the body; only names and compact ABI unpacking differ.
struct FixedSm89ExactF32Params {
    float alpha, beta;
    int m, n, k, lda, ldb, ldc;
};
static_assert(sizeof(FixedSm89ExactF32Params) == 32, "exact N64 parameter size");
static_assert(alignof(FixedSm89ExactF32Params) == 4, "exact N64 parameter alignment");
// mma16.cu has already undefined the incumbent tile macros at this boundary.
// Private constants preserve the incumbent geometry without leaking ambient macros.
#define SM89_EXACT_N64_CP_BM 64
#define SM89_EXACT_N64_CP_BN 64
#define SM89_EXACT_N64_CP_BK 32
#define SM89_EXACT_N64_CP_GROUP_M 8
// Production N64/BK32/S2 geometry, accumulator ownership, full ascending-K32
// FMA chain and epilogue are retained. The only specialization is producer
// planning for aligned full-K slabs. No tensor operations, split reduction,
// scratch or atomics. Actual registers/spills/residency require parent testing.
// Includes expect the unchanged production Fixed common.cuh helpers/constants.
static_assert(SM89_EXACT_N64_CP_BM == 64 && SM89_EXACT_N64_CP_BN == 64 && SM89_EXACT_N64_CP_BK == 32,
              "N64 copy-plan experiment requires the incumbent geometry");
#define SM89_EXACT_N64_CP_THREADS 128
#define SM89_EXACT_N64_CP_GENERIC_ASYNC(BUF, K_TILE)                                    \
    do {                                                                       \
        unsigned _as = (unsigned)__cvta_generic_to_shared(&smem_a[(BUF)][0]);  \
        unsigned _bs = (unsigned)__cvta_generic_to_shared(&smem_b[(BUF)][0]);  \
        for (int _i = threadIdx.x; _i < SM89_EXACT_N64_CP_BM * (SM89_EXACT_N64_CP_BK / 4);               \
             _i += SM89_EXACT_N64_CP_THREADS) {                                          \
            int _r = _i / (SM89_EXACT_N64_CP_BK / 4);                                      \
            int _c = (_i % (SM89_EXACT_N64_CP_BK / 4)) * 4;                                \
            int _gr = row0 + _r;                                               \
            int _gc = (K_TILE) + _c;                                           \
            int _valid = (_gr < m) ? (k - _gc) : 0;                           \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);    \
            unsigned _dst = _as + (unsigned)((_r * SM89_EXACT_N64_CP_BK + _c) * 4);        \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&a[(long long)_gr * lda + _gc]                  \
                : (const void*)a;                                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));              \
        }                                                                      \
        for (int _i = threadIdx.x; _i < SM89_EXACT_N64_CP_BK * (SM89_EXACT_N64_CP_BN / 4);               \
             _i += SM89_EXACT_N64_CP_THREADS) {                                          \
            int _r = _i / (SM89_EXACT_N64_CP_BN / 4);                                      \
            int _c = (_i % (SM89_EXACT_N64_CP_BN / 4)) * 4;                                \
            int _gr = (K_TILE) + _r;                                           \
            int _gc = col0 + _c;                                               \
            int _valid = (_gr < k) ? (n - _gc) : 0;                           \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);    \
            unsigned _dst = _bs + (unsigned)((_r * SM89_EXACT_N64_CP_BN + _c) * 4);        \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&b[(long long)_gr * ldb + _gc]                  \
                : (const void*)b;                                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));              \
        }                                                                      \
        asm volatile("cp.async.commit_group;\n");                             \
    } while (0)

#define SM89_EXACT_N64_CP_GENERIC_SCALAR(BUF, K_TILE)                                   \
    do {                                                                       \
        for (int _i = threadIdx.x; _i < SM89_EXACT_N64_CP_BM * SM89_EXACT_N64_CP_BK;                    \
             _i += SM89_EXACT_N64_CP_THREADS) {                                          \
            int _r = _i / SM89_EXACT_N64_CP_BK;                                             \
            int _c = _i % SM89_EXACT_N64_CP_BK;                                             \
            int _gr = row0 + _r;                                               \
            int _gc = (K_TILE) + _c;                                           \
            smem_a[(BUF)][_i] = (_gr < m && _gc < k)                          \
                ? a[(long long)_gr * lda + _gc]                                \
                : 0.0f;                                                        \
        }                                                                      \
        for (int _i = threadIdx.x; _i < SM89_EXACT_N64_CP_BK * SM89_EXACT_N64_CP_BN;                    \
             _i += SM89_EXACT_N64_CP_THREADS) {                                          \
            int _r = _i / SM89_EXACT_N64_CP_BN;                                             \
            int _c = _i % SM89_EXACT_N64_CP_BN;                                             \
            int _gr = (K_TILE) + _r;                                           \
            int _gc = col0 + _c;                                               \
            smem_b[(BUF)][_i] = (_gr < k && _gc < n)                           \
                ? b[(long long)_gr * ldb + _gc]                                \
                : 0.0f;                                                        \
        }                                                                      \
    } while (0)

// All plan variables are initialized once inside the full-K branch below.
// Shared/vector offsets and source lengths do not change with the K slab.
// Only two advancing 64-bit source bases are retained; each copy derives its
// own address from a common base plus a vector stride, then discards it.
// Invalid rows/columns select the original allocation base for a zero-byte
// copy, so no out-of-range pointer is submitted even to a zero-fill operation.
#define SM89_EXACT_N64_CP_PLANNED_STAGE(BUF)                                             \
    do {                                                                    \
        unsigned _stage = (unsigned)(BUF) * (SM89_EXACT_N64_CP_BM * SM89_EXACT_N64_CP_BK * 4);           \
        _Pragma("unroll")                                                    \
        for (int _i = 0; _i < 4; ++_i) {                                    \
            unsigned _dst = a_destination + _stage + (unsigned)(_i * 2048);  \
            unsigned long long _address = a_next +                           \
                (unsigned long long)_i * a_vector_stride;                   \
            unsigned long long _source = a_bytes[_i] > 0 ? _address          \
                : reinterpret_cast<unsigned long long>(a);                  \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"  \
                         :: "r"(_dst), "l"(_source), "r"(a_bytes[_i]));       \
        }                                                                   \
        _Pragma("unroll")                                                    \
        for (int _i = 0; _i < 4; ++_i) {                                    \
            unsigned _dst = b_destination + _stage + (unsigned)(_i * 2048);  \
            unsigned long long _address = b_next +                           \
                (unsigned long long)_i * b_vector_stride;                   \
            unsigned long long _source = b_bytes > 0 ? _address             \
                : reinterpret_cast<unsigned long long>(b);                  \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"  \
                         :: "r"(_dst), "l"(_source), "r"(b_bytes));           \
        }                                                                   \
        asm volatile("cp.async.commit_group;\n");                             \
    } while (0)

extern "C" __global__ __launch_bounds__(SM89_EXACT_N64_CP_THREADS, 2) void
gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1(
    float* __restrict__ c,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    FixedSm89ExactF32Params params
) {
    const float alpha = params.alpha, beta = params.beta;
    const int m = params.m, n = params.n, k = params.k;
    const int lda = params.lda, ldb = params.ldb, ldc = params.ldc;
    __align__(16) __shared__ float smem_a[2][SM89_EXACT_N64_CP_BM * SM89_EXACT_N64_CP_BK];
    __align__(16) __shared__ float smem_b[2][SM89_EXACT_N64_CP_BK * SM89_EXACT_N64_CP_BN];

    int num_pid_m = (m + SM89_EXACT_N64_CP_BM - 1) / SM89_EXACT_N64_CP_BM;
    int num_pid_n = (n + SM89_EXACT_N64_CP_BN - 1) / SM89_EXACT_N64_CP_BN;
    int num_pid_in_group = SM89_EXACT_N64_CP_GROUP_M * num_pid_n;
    int group_id = blockIdx.x / num_pid_in_group;
    int first_pid_m = group_id * SM89_EXACT_N64_CP_GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, SM89_EXACT_N64_CP_GROUP_M);
    int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m);
    int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;
    int row0 = pid_m * SM89_EXACT_N64_CP_BM;
    int col0 = pid_n * SM89_EXACT_N64_CP_BN;
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

    int num_k_tiles = (k + SM89_EXACT_N64_CP_BK - 1) / SM89_EXACT_N64_CP_BK;
    bool fast_stage = gbf_aligned16(a) && gbf_aligned16(b)
                   && (lda & 3) == 0 && (ldb & 3) == 0;
    // K=0 must never construct operand plans: null A/B are legal then.
    // Keep partial-K and unaligned inputs on the exact incumbent staging path.
    if (num_k_tiles > 0 && fast_stage && (k & (SM89_EXACT_N64_CP_BK - 1)) == 0) {
        const int tid = (int)threadIdx.x;
        const int copy_a_row = tid / (SM89_EXACT_N64_CP_BK / 4);
        const int copy_a_col = (tid % (SM89_EXACT_N64_CP_BK / 4)) * 4;
        const int copy_b_row = tid / (SM89_EXACT_N64_CP_BN / 4);
        const int copy_b_col = col0 + (tid % (SM89_EXACT_N64_CP_BN / 4)) * 4;
        int a_bytes[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            a_bytes[i] = row0 + copy_a_row + i * 16 < m ? 16 : 0;
        const int remaining_b = n - copy_b_col;
        const int b_bytes = remaining_b >= 4 ? 16 : (remaining_b > 0 ? remaining_b * 4 : 0);
        const unsigned a_destination =
            (unsigned)__cvta_generic_to_shared(&smem_a[0][0]) + (unsigned)(tid * 16);
        const unsigned b_destination =
            (unsigned)__cvta_generic_to_shared(&smem_b[0][0]) + (unsigned)(tid * 16);
        const unsigned long long a_vector_stride = (unsigned long long)lda * 64ULL;
        const unsigned long long b_vector_stride = (unsigned long long)ldb * 32ULL;
        const unsigned long long b_slab_stride = b_vector_stride * 4ULL;
        // Integer addresses avoid forming C++ pointers outside an allocation
        // for masked M/N lanes. SM89_EXACT_N64_CP_PLANNED_STAGE selects a/b for those lanes.
        unsigned long long a_next = reinterpret_cast<unsigned long long>(a)
            + ((unsigned long long)(row0 + copy_a_row) * (unsigned long long)lda
               + (unsigned long long)copy_a_col) * 4ULL;
        unsigned long long b_next = reinterpret_cast<unsigned long long>(b)
            + ((unsigned long long)copy_b_row * (unsigned long long)ldb
               + (unsigned long long)copy_b_col) * 4ULL;

        SM89_EXACT_N64_CP_PLANNED_STAGE(0);
        a_next += SM89_EXACT_N64_CP_BK * 4ULL;
        b_next += b_slab_stride;
        int read_buf = 0;
        for (int kt = 0; kt < num_k_tiles; ++kt) {
            asm volatile("cp.async.wait_group 0;\n");
            // Both cross-warp copy visibility and previous-stage consumption
            // must finish before a stage is read or reused.
            __syncthreads();
            if (kt + 1 < num_k_tiles) {
                SM89_EXACT_N64_CP_PLANNED_STAGE(read_buf ^ 1);
                a_next += SM89_EXACT_N64_CP_BK * 4ULL;
                b_next += b_slab_stride;
            }

    #pragma unroll
            for (int kk = 0; kk < SM89_EXACT_N64_CP_BK; kk++) {
                float a_reg[8];
                float b_reg[4];
    #pragma unroll
                for (int i = 0; i < 8; i++)
                    a_reg[i] = smem_a[read_buf][(ty * 8 + i) * SM89_EXACT_N64_CP_BK + kk];
    #pragma unroll
                for (int j = 0; j < 4; j++)
                    b_reg[j] = smem_b[read_buf][kk * SM89_EXACT_N64_CP_BN + tx * 4 + j];
    #pragma unroll
                for (int i = 0; i < 8; i++) {
    #pragma unroll
                    for (int j = 0; j < 4; j++)
                        acc[i][j] = __fmaf_rn(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
            read_buf ^= 1;
        }
    } else {
        if (num_k_tiles > 0) {
            if (fast_stage) {
                SM89_EXACT_N64_CP_GENERIC_ASYNC(0, 0);
            } else {
                SM89_EXACT_N64_CP_GENERIC_SCALAR(0, 0);
            }
        }
        int read_buf = 0;
        for (int kt = 0; kt < num_k_tiles; kt++) {
            if (fast_stage) asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            int next_k = (kt + 1) * SM89_EXACT_N64_CP_BK;
            if (kt + 1 < num_k_tiles) {
                if (fast_stage) {
                    SM89_EXACT_N64_CP_GENERIC_ASYNC(read_buf ^ 1, next_k);
                } else {
                    SM89_EXACT_N64_CP_GENERIC_SCALAR(read_buf ^ 1, next_k);
                }
            }

    #pragma unroll
            for (int kk = 0; kk < SM89_EXACT_N64_CP_BK; kk++) {
                float a_reg[8];
                float b_reg[4];
    #pragma unroll
                for (int i = 0; i < 8; i++)
                    a_reg[i] = smem_a[read_buf][(ty * 8 + i) * SM89_EXACT_N64_CP_BK + kk];
    #pragma unroll
                for (int j = 0; j < 4; j++)
                    b_reg[j] = smem_b[read_buf][kk * SM89_EXACT_N64_CP_BN + tx * 4 + j];
    #pragma unroll
                for (int i = 0; i < 8; i++) {
    #pragma unroll
                    for (int j = 0; j < 4; j++)
                        acc[i][j] = __fmaf_rn(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
            read_buf ^= 1;
        }
    }

    bool pair_store_fast = row0 <= m - SM89_EXACT_N64_CP_BM
        && col0 <= n - SM89_EXACT_N64_CP_BN
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

#undef SM89_EXACT_N64_CP_PLANNED_STAGE
#undef SM89_EXACT_N64_CP_GENERIC_SCALAR
#undef SM89_EXACT_N64_CP_GENERIC_ASYNC
#undef SM89_EXACT_N64_CP_THREADS

#undef SM89_EXACT_N64_CP_BM
#undef SM89_EXACT_N64_CP_BN
#undef SM89_EXACT_N64_CP_BK
#undef SM89_EXACT_N64_CP_GROUP_M
