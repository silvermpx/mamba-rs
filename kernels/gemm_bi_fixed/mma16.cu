// ============================================================================
// The inference tile ladder - the fixed family's fast NN forward.
// ============================================================================
// Three tensor-core tiles, all mma.sync m16n8k16 over ascending 64-wide
// K-slabs with f32 accumulators: GBF128 (128x128, dynamic smem) for large
// grids, GBF64 (64x64, static smem) for mid-size and underfilled grids,
// GBF16 (16x32) for small-M decode and narrow-N shapes. The three tiles
// produce byte-identical output per element - the reduction chain an
// element sees depends only on its own row, column and the slab order,
// never on the tile geometry - which is what makes a shape-keyed tile
// pick legal without changing bits. A safety layer hardens address
// formation: misaligned typed subviews take the scalar staging path
// (same shared-memory bytes) instead of silently staging wrong data.

// ============================================================================
// GBF128: the 128x128 tensor-core tile.
// ============================================================================
// mma.sync.aligned.m16n8k16 with f32 accumulators. Fully deterministic
// (fixed K order, fixed fragment/tile assignment, no atomics, no split-K)
// and batch-invariant across all M: each output element's entire
// K-reduction lives in one warp, independent of gridDim and M.
//
// Staging:
//   - As[m][k] (row-major) AND Bs[k][n] (row-major, global layout) are both
//     16B-chunk contiguous -> 2-stage cp.async pipeline with 4-operand
//     zero-fill for tails (bit-exact vs scalar zero stores). B fragments
//     come from ldmatrix.x2.TRANS of the k-major tile (delivers the
//     col-major k16n8 fragment without a staging transpose).
//   - Pads keep every ldmatrix row chunk in a distinct 4-bank group:
//     A row stride 72 halves (36 words ≡ 4 mod 8), B row stride 136 halves
//     (68 words ≡ 4 mod 8). Row bases are 16B-aligned (144 B / 272 B).
//   - Scalar staging fallback (uniform branch) when lda/ldb % 8 != 0.
//   - Smem (BK=64): NN 71 680 B / TN 69 632 B / NT 73 728 B — beyond the
//     48 KB static cap, so all three use dynamic smem with the
//     MAX_DYNAMIC_SHARED_SIZE_BYTES opt-in set at module load
//     (kernels.rs); launch passes the exact per-kernel byte count.
//     BK=64 halves the wait_group/__syncthreads boundary count per CTA
//     vs BK=32 (the measured per-boundary cost dominated the gap to
//     cuBLAS-TC; see internal/tc-bk64-blueprint.md).
//
// Geometry: CTA 256 threads = 8 warps as 2x4; BM=BN=128 BK=64; warp tile
// 64x32 = 4 m-frags(16) x 4 n-frags(8); bias pre-seeded into the f32
// accumulators (alpha must be 1.0 with bias); one RNE downcast at store.
//
// Fragment thread maps (PTX ISA m16n8k16, 16-bit A/B, .row.col):
//   lane L: g = L>>2, t = L&3
//   A: a0={(g,2t),(g,2t+1)} a1={(g+8,..)} a2={(g,2t+8),..} a3={(g+8,2t+8),..}
//   B: b0={(2t,g),(2t+1,g)} b1={(2t+8,g),(2t+9,g)}
//   C: c0=(g,2t) c1=(g,2t+1) c2=(g+8,2t) c3=(g+8,2t+1)
// A x4: lanes 0-7/8-15/16-23/24-31 -> (rows 0-7,k0)/(rows 8-15,k0)/
// (rows 0-7,k0+8)/(rows 8-15,k0+8). B x2.trans: lanes 0-7/8-15 -> stored
// rows (k0..k0+7)/(k0+8..k0+15) at column n0; .trans delivers M^T fragments
// = the col-major b-frags.

#define GBF128_BM 128
#define GBF128_BN 128
#define GBF128_BK 64
#define GBF128_PAD_A 8
#define GBF128_PAD_B 8
#define GBF128_LDA (GBF128_BK + GBF128_PAD_A)
#define GBF128_LDB (GBF128_BN + GBF128_PAD_B)

// Issue one A+B tile into smem stage `buf` via 16B cp.async with zero-fill
// (fast path; requires lda%8==0 && ldb%8==0, checked by caller-side branch).
// A: 128 rows x 8 chunks; B: 64 rows x 16 chunks; 2048 cp.async / 256 thr.
#define GBF128_STAGE_ASYNC(buf, bkIdx)                                        \
    do {                                                                      \
        unsigned _as = As_sbase + (unsigned)((buf) * GBF128_BM * GBF128_LDA * 2);     \
        unsigned _bs = Bs_sbase + (unsigned)((buf) * GBF128_BK * GBF128_LDB * 2);     \
        for (int _i = threadIdx.x; _i < GBF128_BM * (GBF128_BK / 8); _i += 256) {     \
            int _m = _i / (GBF128_BK / 8);                                        \
            int _c = _i % (GBF128_BK / 8);                                        \
            int _k = _c * 8;                                                  \
            int _gr = pid_m * GBF128_BM + _m;                                     \
            int _gc = (bkIdx) + _k;                                           \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as + (unsigned)((_m * GBF128_LDA + _k) * 2);         \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&A[(long long)_gr * lda + _gc]                 \
                : (const void*)A;                                             \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBF128_BK * (GBF128_BN / 8); _i += 256) {     \
            int _k = _i / (GBF128_BN / 8);                                        \
            int _c = _i % (GBF128_BN / 8);                                        \
            int _n = _c * 8;                                                  \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBF128_BN + _n;                                     \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs + (unsigned)((_k * GBF128_LDB + _n) * 2);         \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&B[(long long)_gk * ldb + _gn]                 \
                : (const void*)B;                                             \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

// Scalar staging fallback for misaligned lda/ldb (rare; uniform branch).
#define GBF128_STAGE_SCALAR(buf, bkIdx, TT, FF)                               \
    do {                                                                      \
        TT* _Asw = &As[buf][0][0];                                            \
        TT* _Bsw = &Bs[buf][0][0];                                            \
        for (int _i = threadIdx.x; _i < GBF128_BM * GBF128_BK; _i += 256) {           \
            int _m = _i / GBF128_BK;                                              \
            int _k = _i % GBF128_BK;                                              \
            int _gr = pid_m * GBF128_BM + _m;                                     \
            int _gc = (bkIdx) + _k;                                           \
            _Asw[_m * GBF128_LDA + _k] = (_gr < M && _gc < K)                     \
                                         ? A[(long long)_gr * lda + _gc]      \
                                         : FF(0.0f);                          \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBF128_BK * GBF128_BN; _i += 256) {           \
            int _k = _i / GBF128_BN;                                              \
            int _n = _i % GBF128_BN;                                              \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBF128_BN + _n;                                     \
            _Bsw[_k * GBF128_LDB + _n] = (_gk < K && _gn < N)                     \
                                         ? B[(long long)_gk * ldb + _gn]      \
                                         : FF(0.0f);                          \
        }                                                                     \
    } while (0)

#define DEFINE_GEMM_BI_NN_TC128(SUFFIX, T_ACT, FROM_F, MMA_T)                    \
extern "C" __global__ __launch_bounds__(256, 1)                                \
void gemm_bi_nn_tc128_##SUFFIX(                                                  \
    T_ACT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    const float* __restrict__ bias,                                            \
    float alpha, float beta,                                                   \
    int M, int N, int K,                                                       \
    int lda, int ldb, int ldc                                                  \
) {                                                                            \
    assert(alpha == 1.0f || bias == nullptr);                                  \
    extern __shared__ __align__(16) unsigned char gbf128_dynsmem[];           \
    T_ACT (*As)[GBF128_BM][GBF128_LDA] =                                               \
        reinterpret_cast<T_ACT (*)[GBF128_BM][GBF128_LDA]>(gbf128_dynsmem);            \
    T_ACT (*Bs)[GBF128_BK][GBF128_LDB] = reinterpret_cast<T_ACT (*)[GBF128_BK][GBF128_LDB]>(   \
        gbf128_dynsmem + 2 * GBF128_BM * GBF128_LDA * (int)sizeof(T_ACT));             \
    int num_pid_n = (N + GBF128_BN - 1) / GBF128_BN;                                   \
    int pid_m = blockIdx.x / num_pid_n;                                        \
    int pid_n = blockIdx.x % num_pid_n;                                        \
    int warp = threadIdx.x / 32;                                               \
    int lane = threadIdx.x % 32;                                               \
    int warpRow = warp / 4;                                                    \
    int warpCol = warp % 4;                                                    \
    int warpM = warpRow * 64;                                                  \
    int warpN = warpCol * 32;                                                  \
    int g = lane >> 2;                                                         \
    int t = lane & 3;                                                          \
    int lm_r = lane & 7;                                                       \
    int lm_q = lane >> 3;                                                      \
    int lm_row_off = (lm_q & 1) ? 8 : 0;                                       \
    int lm_col_off = (lm_q & 2) ? 8 : 0;                                       \
    int lmb_row_off = (lm_q & 1) ? 8 : 0;                                      \
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(&As[0][0][0]);      \
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(&Bs[0][0][0]);      \
    bool fast_stage = ((lda & 7) == 0) && ((ldb & 7) == 0) &&                  \
                      gbf_aligned16(A) && gbf_aligned16(B);                    \
    float acc[4][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 4; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            float b0 = 0.0f, b1 = 0.0f;                                        \
            if (bias != nullptr) {                                             \
                int c0 = pid_n * GBF128_BN + warpN + fn * 8 + 2 * t;               \
                b0 = (c0 < N) ? bias[c0] : 0.0f;                               \
                b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                       \
            }                                                                  \
            acc[fm][fn][0] = b0;                                               \
            acc[fm][fn][1] = b1;                                               \
            acc[fm][fn][2] = b0;                                               \
            acc[fm][fn][3] = b1;                                               \
        }                                                                      \
    }                                                                          \
    int num_k_tiles = (K + GBF128_BK - 1) / GBF128_BK;                                 \
    if (fast_stage) {                                                          \
        GBF128_STAGE_ASYNC(0, 0);                                              \
    } else {                                                                   \
        GBF128_STAGE_SCALAR(0, 0, T_ACT, FROM_F);                              \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (kt + 1 < num_k_tiles) {                                            \
            if (fast_stage) {                                                  \
                GBF128_STAGE_ASYNC(read_buf ^ 1, (kt + 1) * GBF128_BK);            \
            } else {                                                           \
                GBF128_STAGE_SCALAR(read_buf ^ 1, (kt + 1) * GBF128_BK, T_ACT,     \
                                    FROM_F);                                   \
            }                                                                  \
        }                                                                      \
        unsigned As_rd = As_sbase + (unsigned)(read_buf * GBF128_BM * GBF128_LDA * 2); \
        unsigned Bs_rd = Bs_sbase + (unsigned)(read_buf * GBF128_BK * GBF128_LDB * 2); \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (GBF128_BK / 16); ks++) {                            \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[4][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 4; fm++) {                                   \
                int row = warpM + fm * 16 + lm_row_off + lm_r;                 \
                unsigned addr = As_rd +                                        \
                    (unsigned)((row * GBF128_LDA + k0 + lm_col_off) * 2);          \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "                \
                    "{%0,%1,%2,%3}, [%4];\n"                                   \
                    : "=r"(a_frag[fm][0]), "=r"(a_frag[fm][1]),                \
                      "=r"(a_frag[fm][2]), "=r"(a_frag[fm][3])                 \
                    : "r"(addr));                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int fn = 0; fn < 4; fn++) {                                   \
                int row = k0 + lmb_row_off + lm_r;                             \
                unsigned addr = Bs_rd +                                        \
                    (unsigned)((row * GBF128_LDB + warpN + fn * 8) * 2);           \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "          \
                    "{%0,%1}, [%2];\n"                                         \
                    : "=r"(b_frag[fn][0]), "=r"(b_frag[fn][1])                 \
                    : "r"(addr));                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 4; fm++) {                                   \
                _Pragma("unroll")                                              \
                for (int fn = 0; fn < 4; fn++) {                               \
                    asm volatile(                                              \
                        "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "."     \
                        MMA_T ".f32 "                                          \
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "              \
                        "{%0,%1,%2,%3};\n"                                     \
                        : "+f"(acc[fm][fn][0]), "+f"(acc[fm][fn][1]),          \
                          "+f"(acc[fm][fn][2]), "+f"(acc[fm][fn][3])           \
                        : "r"(a_frag[fm][0]), "r"(a_frag[fm][1]),              \
                          "r"(a_frag[fm][2]), "r"(a_frag[fm][3]),              \
                          "r"(b_frag[fn][0]), "r"(b_frag[fn][1]));             \
                }                                                              \
            }                                                                  \
        }                                                                      \
        read_buf ^= 1;                                                         \
    }                                                                          \
    _Pragma("unroll")                                                         \
    for (int fm = 0; fm < 4; fm++) {                                          \
        _Pragma("unroll")                                                     \
        for (int fn = 0; fn < 4; fn++) {                                      \
            int r0 = pid_m * GBF128_BM + warpM + fm * 16 + g;                     \
            int c0 = pid_n * GBF128_BN + warpN + fn * 8 + 2 * t;                  \
            /* c0 is always even (warpN, fn*8, 2t all even), so when          \
               ldc is even too the (c0, c0+1) pair is 4-byte aligned:         \
               pack both RNE results into one 32-bit store (half the          \
               store issue). Values identical to the scalar path. */          \
            _Pragma("unroll")                                                 \
            for (int half = 0; half < 2; half++) {                            \
                int gr = r0 + (half ? 8 : 0);                                 \
                if (gr >= M) continue;                                        \
                /* Explicit unfused multiply: a bare alpha*acc leaves   */ \
                /* ptxas free to contract it per target, which would    */ \
                /* make the epilogue bits architecture-dependent.       */ \
                float v0 = __fmul_rn(alpha, acc[fm][fn][2 * half]);                     \
                float v1 = __fmul_rn(alpha, acc[fm][fn][2 * half + 1]);                 \
                T_ACT* _dst = (c0 < N)                                        \
                    ? &C[(long long)gr * ldc + c0]                            \
                    : (T_ACT*)0;                                              \
                if (beta == 0.0f && (ldc & 1) == 0 && c0 + 1 < N &&           \
                    gbf_aligned4(_dst)) {                                     \
                    gbf_store_pair_rne(_dst, v0, v1);                         \
                } else {                                                      \
                    for (int e = 0; e < 2; e++) {                             \
                        int gc = c0 + e;                                      \
                        if (gc >= N) continue;                                \
                        float val = e ? v1 : v0;                              \
                        if (beta != 0.0f)                                     \
                            val += beta * to_f(C[(long long)gr * ldc + gc]);  \
                        C[(long long)gr * ldc + gc] = FROM_F(val);            \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

DEFINE_GEMM_BI_NN_TC128(bf16, __nv_bfloat16, from_f_bf16, "bf16")
DEFINE_GEMM_BI_NN_TC128(f16,  __half,        from_f_f16,  "f16")

// ============================================================================
// GBF64: the 64x64 tensor-core tile.
// ============================================================================
// Same numeric contract as GBF128, and bit-identical to it per output
// element: both walk K in ascending 64-wide slabs split into ascending
// m16n8k16 mma steps with the same 16B-chunk zero-fill for tails, so an
// element's f32 accumulator sees the exact same mma chain whichever tile
// the dispatcher picked. Do not change the slab width, the mma step
// order, or the tail zero-fill here without changing GBF128 and GBF16 in
// lockstep - the bit-identity is what keeps the tile pick free.
//
// Geometry: CTA 128 threads = 4 warps as 2x2; BM=BN=64, BK=64; warp tile
// 32x32 = 2 m-frags(16) x 4 n-frags(8). Same 2-stage cp.async staging (16B
// chunks, 4-operand zero-fill tails), same ldmatrix x4 / x2(.trans)
// fragment loads, same conflict-free pads (row strides ≡ 4 mod 8 words,
// 16B-aligned row bases): A-layout rows 40 halves (BK wide), B-layout rows
// 72 halves (BN wide). Static smem at BK=64: 36 864 B per kernel (< 48 KB,
// so the Tile64 family stays static while Tile128 goes dynamic).
// Why it wins on small shapes: a 128-tile CTA grid underfills the GPU
// (e.g. d128 in_proj dW = 4 CTAs on a 142-SM Ada); quartering the tile
// quadruples the CTA count at the same total FLOPs.
//
// Every constant below is section-local (GBF64_*).
// NEVER reference ambient tile names from other sections
// define from earlier sections inside this section.

#define GBF64_BM 64
#define GBF64_BN 64
#define GBF64_BK 64
#define GBF64_THREADS 128
#define GBF64_LDA (GBF64_BK + 8) /* 72 halves = 144 B rows, 36 words ≡ 4 mod 8 */
#define GBF64_LDB (GBF64_BN + 8) /* 72 halves = 144 B rows, 36 words ≡ 4 mod 8 */

// NN staging: A 64 rows x 4 chunks + B 32 rows x 8 chunks = 512 cp.async
// over 128 threads (4 per thread), 16B each with zero-fill tails.
#define GBF64_STAGE_ASYNC(buf, bkIdx)                                      \
    do {                                                                      \
        unsigned _as =                                                        \
            As_sbase + (unsigned)((buf) * GBF64_BM * GBF64_LDA * 2);    \
        unsigned _bs =                                                        \
            Bs_sbase + (unsigned)((buf) * GBF64_BK * GBF64_LDB * 2);    \
        for (int _i = threadIdx.x; _i < GBF64_BM * (GBF64_BK / 8);      \
             _i += GBF64_THREADS) {                                        \
            int _m = _i / (GBF64_BK / 8);                                  \
            int _k = (_i % (GBF64_BK / 8)) * 8;                            \
            int _gr = pid_m * GBF64_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as + (unsigned)((_m * GBF64_LDA + _k) * 2);   \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&A[(long long)_gr * lda + _gc]                 \
                : (const void*)A;                                             \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBF64_BK * (GBF64_BN / 8);      \
             _i += GBF64_THREADS) {                                        \
            int _k = _i / (GBF64_BN / 8);                                  \
            int _n = (_i % (GBF64_BN / 8)) * 8;                            \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBF64_BN + _n;                               \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs + (unsigned)((_k * GBF64_LDB + _n) * 2);   \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&B[(long long)_gk * ldb + _gn]                 \
                : (const void*)B;                                             \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

// Scalar staging fallback for misaligned lda/ldb (rare; uniform branch).
#define GBF64_STAGE_SCALAR(buf, bkIdx, TT, FF)                             \
    do {                                                                      \
        TT* _Asw = &As[buf][0][0];                                            \
        TT* _Bsw = &Bs[buf][0][0];                                            \
        for (int _i = threadIdx.x; _i < GBF64_BM * GBF64_BK;            \
             _i += GBF64_THREADS) {                                        \
            int _m = _i / GBF64_BK;                                        \
            int _k = _i % GBF64_BK;                                        \
            int _gr = pid_m * GBF64_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            _Asw[_m * GBF64_LDA + _k] = (_gr < M && _gc < K)               \
                                               ? A[(long long)_gr * lda + _gc]\
                                               : FF(0.0f);                    \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBF64_BK * GBF64_BN;            \
             _i += GBF64_THREADS) {                                        \
            int _k = _i / GBF64_BN;                                        \
            int _n = _i % GBF64_BN;                                        \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBF64_BN + _n;                               \
            _Bsw[_k * GBF64_LDB + _n] = (_gk < K && _gn < N)               \
                                               ? B[(long long)_gk * ldb + _gn]\
                                               : FF(0.0f);                    \
        }                                                                     \
    } while (0)

#define DEFINE_GEMM_BI_NN_TC64(SUFFIX, T_ACT, FROM_F, MMA_T)                  \
extern "C" __global__ __launch_bounds__(GBF64_THREADS, 1)                   \
void gemm_bi_nn_tc64_##SUFFIX(                                                \
    T_ACT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    const float* __restrict__ bias,                                            \
    float alpha, float beta,                                                   \
    int M, int N, int K,                                                       \
    int lda, int ldb, int ldc                                                  \
) {                                                                            \
    assert(alpha == 1.0f || bias == nullptr);                                  \
    __shared__ __align__(16) T_ACT As[2][GBF64_BM][GBF64_LDA];           \
    __shared__ __align__(16) T_ACT Bs[2][GBF64_BK][GBF64_LDB];           \
    int num_pid_n = (N + GBF64_BN - 1) / GBF64_BN;                       \
    int pid_m = blockIdx.x / num_pid_n;                                        \
    int pid_n = blockIdx.x % num_pid_n;                                        \
    int warp = threadIdx.x / 32;                                               \
    int lane = threadIdx.x % 32;                                               \
    int warpM = (warp / 2) * 32;                                               \
    int warpN = (warp % 2) * 32;                                               \
    int g = lane >> 2;                                                         \
    int t = lane & 3;                                                          \
    int lm_r = lane & 7;                                                       \
    int lm_q = lane >> 3;                                                      \
    int lm_row_off = (lm_q & 1) ? 8 : 0;                                       \
    int lm_col_off = (lm_q & 2) ? 8 : 0;                                       \
    int lmb_row_off = (lm_q & 1) ? 8 : 0;                                      \
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(&As[0][0][0]);      \
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(&Bs[0][0][0]);      \
    bool fast_stage = ((lda & 7) == 0) && ((ldb & 7) == 0) &&                  \
                      gbf_aligned16(A) && gbf_aligned16(B);                    \
    float acc[2][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            float b0 = 0.0f, b1 = 0.0f;                                        \
            if (bias != nullptr) {                                             \
                int c0 = pid_n * GBF64_BN + warpN + fn * 8 + 2 * t;         \
                b0 = (c0 < N) ? bias[c0] : 0.0f;                               \
                b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                       \
            }                                                                  \
            acc[fm][fn][0] = b0;                                               \
            acc[fm][fn][1] = b1;                                               \
            acc[fm][fn][2] = b0;                                               \
            acc[fm][fn][3] = b1;                                               \
        }                                                                      \
    }                                                                          \
    int num_k_tiles = (K + GBF64_BK - 1) / GBF64_BK;                     \
    if (fast_stage) {                                                          \
        GBF64_STAGE_ASYNC(0, 0);                                            \
    } else {                                                                   \
        GBF64_STAGE_SCALAR(0, 0, T_ACT, FROM_F);                            \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (kt + 1 < num_k_tiles) {                                            \
            if (fast_stage) {                                                  \
                GBF64_STAGE_ASYNC(read_buf ^ 1, (kt + 1) * GBF64_BK);    \
            } else {                                                           \
                GBF64_STAGE_SCALAR(read_buf ^ 1, (kt + 1) * GBF64_BK,    \
                                      T_ACT, FROM_F);                          \
            }                                                                  \
        }                                                                      \
        unsigned As_rd =                                                       \
            As_sbase + (unsigned)(read_buf * GBF64_BM * GBF64_LDA * 2);  \
        unsigned Bs_rd =                                                       \
            Bs_sbase + (unsigned)(read_buf * GBF64_BK * GBF64_LDB * 2);  \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (GBF64_BK / 16); ks++) {                            \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[2][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 2; fm++) {                                   \
                int row = warpM + fm * 16 + lm_row_off + lm_r;                 \
                unsigned addr = As_rd +                                        \
                    (unsigned)((row * GBF64_LDA + k0 + lm_col_off) * 2);    \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "                \
                    "{%0,%1,%2,%3}, [%4];\n"                                   \
                    : "=r"(a_frag[fm][0]), "=r"(a_frag[fm][1]),                \
                      "=r"(a_frag[fm][2]), "=r"(a_frag[fm][3])                 \
                    : "r"(addr));                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int fn = 0; fn < 4; fn++) {                                   \
                int row = k0 + lmb_row_off + lm_r;                             \
                unsigned addr = Bs_rd +                                        \
                    (unsigned)((row * GBF64_LDB + warpN + fn * 8) * 2);     \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "          \
                    "{%0,%1}, [%2];\n"                                         \
                    : "=r"(b_frag[fn][0]), "=r"(b_frag[fn][1])                 \
                    : "r"(addr));                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 2; fm++) {                                   \
                _Pragma("unroll")                                              \
                for (int fn = 0; fn < 4; fn++) {                               \
                    asm volatile(                                              \
                        "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "."     \
                        MMA_T ".f32 "                                          \
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "              \
                        "{%0,%1,%2,%3};\n"                                     \
                        : "+f"(acc[fm][fn][0]), "+f"(acc[fm][fn][1]),          \
                          "+f"(acc[fm][fn][2]), "+f"(acc[fm][fn][3])           \
                        : "r"(a_frag[fm][0]), "r"(a_frag[fm][1]),              \
                          "r"(a_frag[fm][2]), "r"(a_frag[fm][3]),              \
                          "r"(b_frag[fn][0]), "r"(b_frag[fn][1]));             \
                }                                                              \
            }                                                                  \
        }                                                                      \
        read_buf ^= 1;                                                         \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            int r0 = pid_m * GBF64_BM + warpM + fm * 16 + g;                \
            int c0 = pid_n * GBF64_BN + warpN + fn * 8 + 2 * t;             \
            _Pragma("unroll")                                                  \
            for (int e = 0; e < 4; e++) {                                      \
                int gr = r0 + (e >= 2 ? 8 : 0);                                \
                int gc = c0 + (e & 1);                                         \
                if (gr >= M || gc >= N) continue;                              \
                float val = __fmul_rn(alpha, acc[fm][fn][e]);                            \
                if (beta != 0.0f)                                              \
                    val += beta * to_f(C[(long long)gr * ldc + gc]);           \
                C[(long long)gr * ldc + gc] = FROM_F(val);                     \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

DEFINE_GEMM_BI_NN_TC64(bf16, __nv_bfloat16, from_f_bf16, "bf16")
DEFINE_GEMM_BI_NN_TC64(f16,  __half,        from_f_f16,  "f16")
// ── GBF16: the 16x32 thin tile (decode and narrow-N shapes) ──
//
// Same arithmetic contract as GBF64/GBF128 (ascending m16n8k16 K-slabs,
// f32 accumulators, bias pre-seeded, single RNE downcast), so it is
// byte-identical to them per element. What differs is scheduling: a
// 16-row tile stops wasting 3/4 of the MMA work at M<=16, BN=32 raises
// the CTA count at small M (decode needs CTAs for memory-level
// parallelism, not tile area), and a 4-deep cp.async pipeline keeps
// enough bytes in flight to approach the DRAM floor where a 2-stage
// pipeline stalls on latency.
//
// Pipeline bookkeeping: every iteration commits EXACTLY one group -
// a real stage or an empty commit once the K range is exhausted - so
// the in-flight count is uniform and `wait_group STAGES-2` always
// waits for precisely the stage the mainloop is about to read. Without
// the empty commits a short K (fewer tiles than stages) would make
// wait_group return before the data landed.
//
// Every constant below is section-local (GBF16_*).
#define GBF16_BM 16
#define GBF16_BN 32
#define GBF16_BK 64
#define GBF16_THREADS 128
#define GBF16_STAGES 4
#define GBF16_LDA (GBF16_BK + 8) /* 72 halves = 144 B rows */
#define GBF16_LDB (GBF16_BN + 8) /* 40 halves = 80 B rows, 20 words == 4 mod 8 */

#define GBF16_STAGE_ASYNC(buf, bkIdx)                                      \
    do {                                                                      \
        unsigned _as =                                                        \
            As_sbase + (unsigned)((buf) * GBF16_BM * GBF16_LDA * 2);    \
        unsigned _bs =                                                        \
            Bs_sbase + (unsigned)((buf) * GBF16_BK * GBF16_LDB * 2);    \
        for (int _i = threadIdx.x; _i < GBF16_BM * (GBF16_BK / 8);      \
             _i += GBF16_THREADS) {                                        \
            int _m = _i / (GBF16_BK / 8);                                  \
            int _k = (_i % (GBF16_BK / 8)) * 8;                            \
            int _gr = pid_m * GBF16_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as + (unsigned)((_m * GBF16_LDA + _k) * 2);   \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&A[(long long)_gr * lda + _gc]                 \
                : (const void*)A;                                             \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBF16_BK * (GBF16_BN / 8);      \
             _i += GBF16_THREADS) {                                        \
            int _k = _i / (GBF16_BN / 8);                                  \
            int _n = (_i % (GBF16_BN / 8)) * 8;                            \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBF16_BN + _n;                               \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs + (unsigned)((_k * GBF16_LDB + _n) * 2);   \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&B[(long long)_gk * ldb + _gn]                 \
                : (const void*)B;                                             \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define GBF16_STAGE_SCALAR(buf, bkIdx, TT, FF)                             \
    do {                                                                      \
        TT* _Asw = &As[buf][0][0];                                            \
        TT* _Bsw = &Bs[buf][0][0];                                            \
        for (int _i = threadIdx.x; _i < GBF16_BM * GBF16_BK;            \
             _i += GBF16_THREADS) {                                        \
            int _m = _i / GBF16_BK;                                        \
            int _k = _i % GBF16_BK;                                        \
            int _gr = pid_m * GBF16_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            _Asw[_m * GBF16_LDA + _k] = (_gr < M && _gc < K)               \
                                               ? A[(long long)_gr * lda + _gc]\
                                               : FF(0.0f);                    \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBF16_BK * GBF16_BN;            \
             _i += GBF16_THREADS) {                                        \
            int _k = _i / GBF16_BN;                                        \
            int _n = _i % GBF16_BN;                                        \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBF16_BN + _n;                               \
            _Bsw[_k * GBF16_LDB + _n] = (_gk < K && _gn < N)               \
                                               ? B[(long long)_gk * ldb + _gn]\
                                               : FF(0.0f);                    \
        }                                                                     \
    } while (0)

#define DEFINE_GEMM_BI_NN_TC16(SUFFIX, T_ACT, FROM_F, MMA_T)                  \
extern "C" __global__ __launch_bounds__(GBF16_THREADS, 3)                   \
void gemm_bi_nn_tc16_##SUFFIX(                                                \
    T_ACT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    const float* __restrict__ bias,                                            \
    float alpha, float beta,                                                   \
    int M, int N, int K,                                                       \
    int lda, int ldb, int ldc                                                  \
) {                                                                            \
    assert(alpha == 1.0f || bias == nullptr);                                  \
    __shared__ __align__(16)                                                   \
        T_ACT As[GBF16_STAGES][GBF16_BM][GBF16_LDA];                  \
    __shared__ __align__(16)                                                   \
        T_ACT Bs[GBF16_STAGES][GBF16_BK][GBF16_LDB];                  \
    int num_pid_n = (N + GBF16_BN - 1) / GBF16_BN;                       \
    int pid_m = blockIdx.x / num_pid_n;                                        \
    int pid_n = blockIdx.x % num_pid_n;                                        \
    int warp = threadIdx.x / 32;                                               \
    int lane = threadIdx.x % 32;                                               \
    int warpN = warp * 8;                                                      \
    int g = lane >> 2;                                                         \
    int t = lane & 3;                                                          \
    int lm_r = lane & 7;                                                       \
    int lm_q = lane >> 3;                                                      \
    int lm_row_off = (lm_q & 1) ? 8 : 0;                                       \
    int lm_col_off = (lm_q & 2) ? 8 : 0;                                       \
    int lmb_row_off = (lm_q & 1) ? 8 : 0;                                      \
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(&As[0][0][0]);      \
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(&Bs[0][0][0]);      \
    bool fast_stage = ((lda & 7) == 0) && ((ldb & 7) == 0) &&                  \
                      gbf_aligned16(A) && gbf_aligned16(B);                    \
    float acc[4];                                                              \
    {                                                                          \
        float b0 = 0.0f, b1 = 0.0f;                                            \
        if (bias != nullptr) {                                                 \
            int c0 = pid_n * GBF16_BN + warpN + 2 * t;                      \
            b0 = (c0 < N) ? bias[c0] : 0.0f;                                   \
            b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                           \
        }                                                                      \
        acc[0] = b0;                                                           \
        acc[1] = b1;                                                           \
        acc[2] = b0;                                                           \
        acc[3] = b1;                                                           \
    }                                                                          \
    int num_k_tiles = (K + GBF16_BK - 1) / GBF16_BK;                     \
    /* Prologue: STAGES-1 commit groups, real or empty - uniform count. */    \
    for (int p = 0; p < GBF16_STAGES - 1; p++) {                            \
        if (p < num_k_tiles) {                                                 \
            if (fast_stage) {                                                  \
                GBF16_STAGE_ASYNC(p, p * GBF16_BK);                      \
            } else {                                                           \
                GBF16_STAGE_SCALAR(p, p * GBF16_BK, T_ACT, FROM_F);      \
                asm volatile("cp.async.commit_group;\n");                     \
            }                                                                  \
        } else {                                                               \
            asm volatile("cp.async.commit_group;\n");                         \
        }                                                                      \
    }                                                                          \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                 \
        asm volatile("cp.async.wait_group %0;\n"                              \
                     :: "n"(GBF16_STAGES - 2));                             \
        __syncthreads();                                                       \
        int next = kt + GBF16_STAGES - 1;                                   \
        if (next < num_k_tiles) {                                              \
            int wbuf = next % GBF16_STAGES;                                 \
            if (fast_stage) {                                                  \
                GBF16_STAGE_ASYNC(wbuf, next * GBF16_BK);                \
            } else {                                                           \
                GBF16_STAGE_SCALAR(wbuf, next * GBF16_BK, T_ACT,         \
                                      FROM_F);                                 \
                asm volatile("cp.async.commit_group;\n");                     \
            }                                                                  \
        } else {                                                               \
            asm volatile("cp.async.commit_group;\n");                         \
        }                                                                      \
        int rbuf = kt % GBF16_STAGES;                                       \
        unsigned As_rd =                                                       \
            As_sbase + (unsigned)(rbuf * GBF16_BM * GBF16_LDA * 2);      \
        unsigned Bs_rd =                                                       \
            Bs_sbase + (unsigned)(rbuf * GBF16_BK * GBF16_LDB * 2);      \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (GBF16_BK / 16); ks++) {                      \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[4];                                                \
            unsigned b_frag[2];                                                \
            {                                                                  \
                int row = lm_row_off + lm_r;                                   \
                unsigned addr = As_rd +                                        \
                    (unsigned)((row * GBF16_LDA + k0 + lm_col_off) * 2);    \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "                \
                    "{%0,%1,%2,%3}, [%4];\n"                                  \
                    : "=r"(a_frag[0]), "=r"(a_frag[1]),                        \
                      "=r"(a_frag[2]), "=r"(a_frag[3])                         \
                    : "r"(addr));                                              \
            }                                                                  \
            {                                                                  \
                int row = k0 + lmb_row_off + lm_r;                             \
                unsigned addr = Bs_rd +                                        \
                    (unsigned)((row * GBF16_LDB + warpN) * 2);              \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "          \
                    "{%0,%1}, [%2];\n"                                        \
                    : "=r"(b_frag[0]), "=r"(b_frag[1])                         \
                    : "r"(addr));                                              \
            }                                                                  \
            asm volatile(                                                      \
                "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "."             \
                MMA_T ".f32 "                                                  \
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "                      \
                "{%0,%1,%2,%3};\n"                                            \
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])       \
                : "r"(a_frag[0]), "r"(a_frag[1]),                              \
                  "r"(a_frag[2]), "r"(a_frag[3]),                              \
                  "r"(b_frag[0]), "r"(b_frag[1]));                             \
        }                                                                      \
    }                                                                          \
    {                                                                          \
        int r0 = pid_m * GBF16_BM + g;                                      \
        int c0 = pid_n * GBF16_BN + warpN + 2 * t;                          \
        _Pragma("unroll")                                                      \
        for (int e = 0; e < 4; e++) {                                          \
            int gr = r0 + (e >= 2 ? 8 : 0);                                    \
            int gc = c0 + (e & 1);                                             \
            if (gr >= M || gc >= N) continue;                                  \
            float val = __fmul_rn(alpha, acc[e]);                                        \
            if (beta != 0.0f)                                                  \
                val += beta * to_f(C[(long long)gr * ldc + gc]);               \
            C[(long long)gr * ldc + gc] = FROM_F(val);                         \
        }                                                                      \
    }                                                                          \
}

DEFINE_GEMM_BI_NN_TC16(bf16, __nv_bfloat16, from_f_bf16, "bf16")
DEFINE_GEMM_BI_NN_TC16(f16,  __half,        from_f_f16,  "f16")

// GBF hygiene: everything section-local above is undefined so nothing can
// leak into gemm_bi_triad.cu, which concatenates AFTER this file.
#undef GBF128_BM
#undef GBF128_BN
#undef GBF128_BK
#undef GBF128_PAD_A
#undef GBF128_PAD_B
#undef GBF128_LDA
#undef GBF128_LDB
#undef GBF128_STAGE_ASYNC
#undef GBF128_STAGE_SCALAR
#undef DEFINE_GEMM_BI_NN_TC128
#undef GBF64_BM
#undef GBF64_BN
#undef GBF64_BK
#undef GBF64_THREADS
#undef GBF64_LDA
#undef GBF64_LDB
#undef GBF64_STAGE_ASYNC
#undef GBF64_STAGE_SCALAR
#undef DEFINE_GEMM_BI_NN_TC64
#undef GBF16_BM
#undef GBF16_BN
#undef GBF16_BK
#undef GBF16_THREADS
#undef GBF16_STAGES
#undef GBF16_LDA
#undef GBF16_LDB
#undef GBF16_STAGE_ASYNC
#undef GBF16_STAGE_SCALAR
#undef DEFINE_GEMM_BI_NN_TC16

// Ambient-define hygiene: this file is concatenated BEFORE gemm_bi_triad.cu
// in the single NVRTC blob. A tile macro leaking downstream would compile
// someone else's kernels with these constants - plausible code that never
// launches correctly. Undefine everything tile-local.
#undef BLOCK_M
#undef BLOCK_N
#undef BLOCK_K
#undef GROUP_M
#undef THREADS
#undef WARPS_PER_CTA
#undef WARPS_M
#undef WARPS_N
#undef FRAG_M
#undef FRAG_N
#undef FRAG_K
#undef WARP_FRAGS_N
#undef BLOCK_N_MV
#undef WARPS_PER_BLOCK
#undef THREADS_PER_BLOCK
