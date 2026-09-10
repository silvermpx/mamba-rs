// SM120-only sliced next-stage exact F32 N64 extension.
// Frozen standalone 7f3499c24e61f3071c45cfed6dc7bb5fcce73e4761a9c5c68740559346db7133:
// only private names/constants and compact five-argument ABI unpacking differ.
struct FixedSm120SlicedF32Params {
    float alpha, beta;
    int m, n, k, lda, ldb, ldc;
};
static_assert(sizeof(FixedSm120SlicedF32Params) == 32, "sliced parameter size");
static_assert(alignof(FixedSm120SlicedF32Params) == 4, "sliced parameter alignment");
#define SM120_EXACT_N64_SLICED_BM 64
#define SM120_EXACT_N64_SLICED_BN 64
#define SM120_EXACT_N64_SLICED_BK 32
#define SM120_EXACT_N64_SLICED_GROUP_M 8
namespace fixed_sm120_n64_sliced {
__host__ __device__ constexpr int next_copy_slice(int kk) {
    return kk >= 0 && kk < 32 && (kk & 7) == 0 ? kk / 8 : -1;
}
__host__ __device__ constexpr bool full_mn_tile(int m, int n, int row0, int col0) {
    return m >= 64 && n >= 64 && row0 >= 0 && col0 >= 0
        && row0 <= m - 64 && col0 <= n - 64;
}
}
// Production N64/BK32/S2 geometry, accumulator ownership, full ascending-K32
// FMA chain and epilogue are retained. The only specialization is producer
// planning for aligned full-K slabs. No tensor operations, split reduction,
// scratch or atomics. Actual registers/spills/residency require parent testing.
// Includes expect the unchanged production Fixed common.cuh helpers/constants.
static_assert(SM120_EXACT_N64_SLICED_BM == 64 && SM120_EXACT_N64_SLICED_BN == 64 && SM120_EXACT_N64_SLICED_BK == 32,
              "N64 copy-plan experiment requires the incumbent geometry");
#define SM120_EXACT_N64_SLICED_THREADS 128
#define SM120_EXACT_N64_SLICED_GENERIC_ASYNC(BUF, K_TILE)                                    \
    do {                                                                       \
        unsigned _as = (unsigned)__cvta_generic_to_shared(&smem_a[(BUF)][0]);  \
        unsigned _bs = (unsigned)__cvta_generic_to_shared(&smem_b[(BUF)][0]);  \
        for (int _i = threadIdx.x; _i < SM120_EXACT_N64_SLICED_BM * (SM120_EXACT_N64_SLICED_BK / 4);               \
             _i += SM120_EXACT_N64_SLICED_THREADS) {                                          \
            int _r = _i / (SM120_EXACT_N64_SLICED_BK / 4);                                      \
            int _c = (_i % (SM120_EXACT_N64_SLICED_BK / 4)) * 4;                                \
            int _gr = row0 + _r;                                               \
            int _gc = (K_TILE) + _c;                                           \
            int _valid = (_gr < m) ? (k - _gc) : 0;                           \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);    \
            unsigned _dst = _as + (unsigned)((_r * SM120_EXACT_N64_SLICED_BK + _c) * 4);        \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&a[(long long)_gr * lda + _gc]                  \
                : (const void*)a;                                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));              \
        }                                                                      \
        for (int _i = threadIdx.x; _i < SM120_EXACT_N64_SLICED_BK * (SM120_EXACT_N64_SLICED_BN / 4);               \
             _i += SM120_EXACT_N64_SLICED_THREADS) {                                          \
            int _r = _i / (SM120_EXACT_N64_SLICED_BN / 4);                                      \
            int _c = (_i % (SM120_EXACT_N64_SLICED_BN / 4)) * 4;                                \
            int _gr = (K_TILE) + _r;                                           \
            int _gc = col0 + _c;                                               \
            int _valid = (_gr < k) ? (n - _gc) : 0;                           \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);    \
            unsigned _dst = _bs + (unsigned)((_r * SM120_EXACT_N64_SLICED_BN + _c) * 4);        \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&b[(long long)_gr * ldb + _gc]                  \
                : (const void*)b;                                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));              \
        }                                                                      \
        asm volatile("cp.async.commit_group;\n");                             \
    } while (0)

#define SM120_EXACT_N64_SLICED_GENERIC_SCALAR(BUF, K_TILE)                                   \
    do {                                                                       \
        for (int _i = threadIdx.x; _i < SM120_EXACT_N64_SLICED_BM * SM120_EXACT_N64_SLICED_BK;                    \
             _i += SM120_EXACT_N64_SLICED_THREADS) {                                          \
            int _r = _i / SM120_EXACT_N64_SLICED_BK;                                             \
            int _c = _i % SM120_EXACT_N64_SLICED_BK;                                             \
            int _gr = row0 + _r;                                               \
            int _gc = (K_TILE) + _c;                                           \
            smem_a[(BUF)][_i] = (_gr < m && _gc < k)                          \
                ? a[(long long)_gr * lda + _gc]                                \
                : 0.0f;                                                        \
        }                                                                      \
        for (int _i = threadIdx.x; _i < SM120_EXACT_N64_SLICED_BK * SM120_EXACT_N64_SLICED_BN;                    \
             _i += SM120_EXACT_N64_SLICED_THREADS) {                                          \
            int _r = _i / SM120_EXACT_N64_SLICED_BN;                                             \
            int _c = _i % SM120_EXACT_N64_SLICED_BN;                                             \
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
#define SM120_EXACT_N64_SLICED_PLANNED_STAGE(BUF)                                             \
    do {                                                                    \
        unsigned _stage = (unsigned)(BUF) * (SM120_EXACT_N64_SLICED_BM * SM120_EXACT_N64_SLICED_BK * 4);           \
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

// Full-M/N tiles only: every vector is in bounds, so omit src-size and masks.
// This is the ONLY new device-memory operation; destinations/order are v1's.
#define SM120_EXACT_N64_SLICED_UNMASKED_STAGE(BUF)                                            \
    do {                                                                    \
        unsigned _stage = (unsigned)(BUF) * (SM120_EXACT_N64_SLICED_BM * SM120_EXACT_N64_SLICED_BK * 4);           \
        _Pragma("unroll")                                                    \
        for (int _i = 0; _i < 4; ++_i) {                                    \
            unsigned _dst = a_destination + _stage + (unsigned)(_i * 2048);  \
            unsigned long long _address = a_next +                           \
                (unsigned long long)_i * a_vector_stride;                   \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"       \
                         :: "r"(_dst), "l"(_address));                       \
        }                                                                   \
        _Pragma("unroll")                                                    \
        for (int _i = 0; _i < 4; ++_i) {                                    \
            unsigned _dst = b_destination + _stage + (unsigned)(_i * 2048);  \
            unsigned long long _address = b_next +                           \
                (unsigned long long)_i * b_vector_stride;                   \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"       \
                         :: "r"(_dst), "l"(_address));                       \
        }                                                                   \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

// Only complete M/N tiles call this. One disjoint A/B pair per issue.
// Commit belongs to the caller after slice3; no wait/barrier is removed.
#define SM120_EXACT_N64_SLICED_NEXT_SLICE(BUF, I)                                          \
    do {                                                                  \
        unsigned _stage = (unsigned)(BUF) * (SM120_EXACT_N64_SLICED_BM * SM120_EXACT_N64_SLICED_BK * 4);         \
        unsigned _i = (unsigned)(I);                                       \
        unsigned _adst = a_destination + _stage + _i * 2048U;               \
        unsigned long long _asrc = a_next + (unsigned long long)_i * a_vector_stride; \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"         \
                     :: "r"(_adst), "l"(_asrc));                           \
        unsigned _bdst = b_destination + _stage + _i * 2048U;               \
        unsigned long long _bsrc = b_next + (unsigned long long)_i * b_vector_stride; \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"         \
                     :: "r"(_bdst), "l"(_bsrc));                           \
    } while (0)

extern "C" __global__ __launch_bounds__(SM120_EXACT_N64_SLICED_THREADS, 2) void
gemm_bi_nn_fixed_sm120_f32_n64_sliced_v1(
    float* __restrict__ c,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    FixedSm120SlicedF32Params params
) {
    const float alpha = params.alpha, beta = params.beta;
    const int m = params.m, n = params.n, k = params.k;
    const int lda = params.lda, ldb = params.ldb, ldc = params.ldc;
    __align__(16) __shared__ float smem_a[2][SM120_EXACT_N64_SLICED_BM * SM120_EXACT_N64_SLICED_BK];
    __align__(16) __shared__ float smem_b[2][SM120_EXACT_N64_SLICED_BK * SM120_EXACT_N64_SLICED_BN];

    int num_pid_m = (m + SM120_EXACT_N64_SLICED_BM - 1) / SM120_EXACT_N64_SLICED_BM;
    int num_pid_n = (n + SM120_EXACT_N64_SLICED_BN - 1) / SM120_EXACT_N64_SLICED_BN;
    int num_pid_in_group = SM120_EXACT_N64_SLICED_GROUP_M * num_pid_n;
    int group_id = blockIdx.x / num_pid_in_group;
    int first_pid_m = group_id * SM120_EXACT_N64_SLICED_GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, SM120_EXACT_N64_SLICED_GROUP_M);
    int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m);
    int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;
    int row0 = pid_m * SM120_EXACT_N64_SLICED_BM;
    int col0 = pid_n * SM120_EXACT_N64_SLICED_BN;
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

    int num_k_tiles = (k + SM120_EXACT_N64_SLICED_BK - 1) / SM120_EXACT_N64_SLICED_BK;
    bool fast_stage = gbf_aligned16(a) && gbf_aligned16(b)
                   && (lda & 3) == 0 && (ldb & 3) == 0;
    // K=0 must never construct operand plans: null A/B are legal then.
    // Keep partial-K and unaligned inputs on the exact incumbent staging path.
    if (num_k_tiles > 0 && fast_stage && (k & (SM120_EXACT_N64_SLICED_BK - 1)) == 0) {
        const int tid = (int)threadIdx.x;
        const int copy_a_row = tid / (SM120_EXACT_N64_SLICED_BK / 4);
        const int copy_a_col = (tid % (SM120_EXACT_N64_SLICED_BK / 4)) * 4;
        const int copy_b_row = tid / (SM120_EXACT_N64_SLICED_BN / 4);
        const int copy_b_col = col0 + (tid % (SM120_EXACT_N64_SLICED_BN / 4)) * 4;
        const unsigned a_destination =
            (unsigned)__cvta_generic_to_shared(&smem_a[0][0]) + (unsigned)(tid * 16);
        const unsigned b_destination =
            (unsigned)__cvta_generic_to_shared(&smem_b[0][0]) + (unsigned)(tid * 16);
        const unsigned long long a_vector_stride = (unsigned long long)lda * 64ULL;
        const unsigned long long b_vector_stride = (unsigned long long)ldb * 32ULL;
        const unsigned long long b_slab_stride = b_vector_stride * 4ULL;
        // Integer addresses avoid forming C++ pointers outside an allocation
        // for masked M/N lanes. SM120_EXACT_N64_SLICED_PLANNED_STAGE selects a/b for those lanes.
        unsigned long long a_next = reinterpret_cast<unsigned long long>(a)
            + ((unsigned long long)(row0 + copy_a_row) * (unsigned long long)lda
               + (unsigned long long)copy_a_col) * 4ULL;
        unsigned long long b_next = reinterpret_cast<unsigned long long>(b)
            + ((unsigned long long)copy_b_row * (unsigned long long)ldb
               + (unsigned long long)copy_b_col) * 4ULL;

        // Uniform CTA split. Only this branch can omit zero-fill operands.
        // The outer aligned/full-K admission remains exactly the v1 condition.
        if (fixed_sm120_n64_sliced::full_mn_tile(m, n, row0, col0)) {
            SM120_EXACT_N64_SLICED_UNMASKED_STAGE(0);
            a_next += SM120_EXACT_N64_SLICED_BK * 4ULL;
            b_next += b_slab_stride;
            int read_buf = 0;
            for (int kt = 0; kt < num_k_tiles; ++kt) {
                asm volatile("cp.async.wait_group 0;\n");
                // Both cross-warp copy visibility and previous-stage consumption
                // must finish before a stage is read or reused.
                __syncthreads();

        #pragma unroll
                for (int kk = 0; kk < SM120_EXACT_N64_SLICED_BK; kk++) {
                    const int next_slice = fixed_sm120_n64_sliced::next_copy_slice(kk);
                    if (kt + 1 < num_k_tiles && next_slice >= 0) {
                        SM120_EXACT_N64_SLICED_NEXT_SLICE(read_buf ^ 1, next_slice);
                        // The group contains all four disjoint A/B pairs.
                        if (next_slice == 3) asm volatile("cp.async.commit_group;\n");
                    }
                    float a_reg[8];
                    float b_reg[4];
        #pragma unroll
                    for (int i = 0; i < 8; i++)
                        a_reg[i] = smem_a[read_buf][(ty * 8 + i) * SM120_EXACT_N64_SLICED_BK + kk];
        #pragma unroll
                    for (int j = 0; j < 4; j++)
                        b_reg[j] = smem_b[read_buf][kk * SM120_EXACT_N64_SLICED_BN + tx * 4 + j];
        #pragma unroll
                    for (int i = 0; i < 8; i++) {
        #pragma unroll
                        for (int j = 0; j < 4; j++)
                            acc[i][j] = __fmaf_rn(a_reg[i], b_reg[j], acc[i][j]);
                    }
                }
                // All next-stage address uses are over before either base advances.
                if (kt + 1 < num_k_tiles) {
                    a_next += SM120_EXACT_N64_SLICED_BK * 4ULL;
                    b_next += b_slab_stride;
                }
                read_buf ^= 1;
            }
        } else {
            int a_bytes[4];
    #pragma unroll
            for (int i = 0; i < 4; ++i)
                a_bytes[i] = row0 + copy_a_row + i * 16 < m ? 16 : 0;
            const int remaining_b = n - copy_b_col;
            const int b_bytes = remaining_b >= 4 ? 16 : (remaining_b > 0 ? remaining_b * 4 : 0);
            SM120_EXACT_N64_SLICED_PLANNED_STAGE(0);
            a_next += SM120_EXACT_N64_SLICED_BK * 4ULL;
            b_next += b_slab_stride;
            int read_buf = 0;
            for (int kt = 0; kt < num_k_tiles; ++kt) {
                asm volatile("cp.async.wait_group 0;\n");
                // Both cross-warp copy visibility and previous-stage consumption
                // must finish before a stage is read or reused.
                __syncthreads();
                if (kt + 1 < num_k_tiles) {
                    SM120_EXACT_N64_SLICED_PLANNED_STAGE(read_buf ^ 1);
                    a_next += SM120_EXACT_N64_SLICED_BK * 4ULL;
                    b_next += b_slab_stride;
                }

        #pragma unroll
                for (int kk = 0; kk < SM120_EXACT_N64_SLICED_BK; kk++) {
                    float a_reg[8];
                    float b_reg[4];
        #pragma unroll
                    for (int i = 0; i < 8; i++)
                        a_reg[i] = smem_a[read_buf][(ty * 8 + i) * SM120_EXACT_N64_SLICED_BK + kk];
        #pragma unroll
                    for (int j = 0; j < 4; j++)
                        b_reg[j] = smem_b[read_buf][kk * SM120_EXACT_N64_SLICED_BN + tx * 4 + j];
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
    } else {
        if (num_k_tiles > 0) {
            if (fast_stage) {
                SM120_EXACT_N64_SLICED_GENERIC_ASYNC(0, 0);
            } else {
                SM120_EXACT_N64_SLICED_GENERIC_SCALAR(0, 0);
            }
        }
        int read_buf = 0;
        for (int kt = 0; kt < num_k_tiles; kt++) {
            if (fast_stage) asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            int next_k = (kt + 1) * SM120_EXACT_N64_SLICED_BK;
            if (kt + 1 < num_k_tiles) {
                if (fast_stage) {
                    SM120_EXACT_N64_SLICED_GENERIC_ASYNC(read_buf ^ 1, next_k);
                } else {
                    SM120_EXACT_N64_SLICED_GENERIC_SCALAR(read_buf ^ 1, next_k);
                }
            }

    #pragma unroll
            for (int kk = 0; kk < SM120_EXACT_N64_SLICED_BK; kk++) {
                float a_reg[8];
                float b_reg[4];
    #pragma unroll
                for (int i = 0; i < 8; i++)
                    a_reg[i] = smem_a[read_buf][(ty * 8 + i) * SM120_EXACT_N64_SLICED_BK + kk];
    #pragma unroll
                for (int j = 0; j < 4; j++)
                    b_reg[j] = smem_b[read_buf][kk * SM120_EXACT_N64_SLICED_BN + tx * 4 + j];
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

    bool pair_store_fast = row0 <= m - SM120_EXACT_N64_SLICED_BM
        && col0 <= n - SM120_EXACT_N64_SLICED_BN
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

#undef SM120_EXACT_N64_SLICED_PLANNED_STAGE
#undef SM120_EXACT_N64_SLICED_GENERIC_SCALAR
#undef SM120_EXACT_N64_SLICED_GENERIC_ASYNC
#undef SM120_EXACT_N64_SLICED_THREADS
#undef SM120_EXACT_N64_SLICED_UNMASKED_STAGE
#undef SM120_EXACT_N64_SLICED_NEXT_SLICE

#undef SM120_EXACT_N64_SLICED_BM
#undef SM120_EXACT_N64_SLICED_BN
#undef SM120_EXACT_N64_SLICED_BK
#undef SM120_EXACT_N64_SLICED_GROUP_M
