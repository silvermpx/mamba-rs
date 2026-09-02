#define GEMM_BI_TC128_BM 128
#define GEMM_BI_TC128_BN 128
#define GEMM_BI_TC128_BK 64
#define GEMM_BI_TC128_PAD_A 8
#define GEMM_BI_TC128_PAD_B 8
#define GEMM_BI_TC128_LDA (GEMM_BI_TC128_BK + GEMM_BI_TC128_PAD_A)
#define GEMM_BI_TC128_LDB (GEMM_BI_TC128_BN + GEMM_BI_TC128_PAD_B)

// Issue one A+B tile into smem stage `buf` via 16B cp.async with zero-fill
// (fast path; the caller proves 16-byte operand bases and row starts).
// A: 128 rows x 8 chunks; B: 64 rows x 16 chunks; 2048 cp.async / 256 thr.
#define GEMM_BI_TC128_STAGE_ASYNC(buf, bkIdx)                                        \
    do {                                                                      \
        unsigned _as = As_sbase + (unsigned)((buf) * GEMM_BI_TC128_BM * GEMM_BI_TC128_LDA * 2);     \
        unsigned _bs = Bs_sbase + (unsigned)((buf) * GEMM_BI_TC128_BK * GEMM_BI_TC128_LDB * 2);     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC128_BM * (GEMM_BI_TC128_BK / 8); _i += 256) {     \
            int _m = _i / (GEMM_BI_TC128_BK / 8);                                        \
            int _c = _i % (GEMM_BI_TC128_BK / 8);                                        \
            int _k = _c * 8;                                                  \
            int _gr = pid_m * GEMM_BI_TC128_BM + _m;                                     \
            int _gc = (bkIdx) + _k;                                           \
            int _elems = gemm_bi_cp_async_valid_elems(_gr < M, K, _gc);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _as + (unsigned)((_m * GEMM_BI_TC128_LDA + _k) * 2);         \
            long long _offset = _bytes == 0 ? 0 : (long long)_gr * lda + _gc; \
            const void* _src = gemm_bi_cp_async_source(A, _offset, _bytes);        \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC128_BK * (GEMM_BI_TC128_BN / 8); _i += 256) {     \
            int _k = _i / (GEMM_BI_TC128_BN / 8);                                        \
            int _c = _i % (GEMM_BI_TC128_BN / 8);                                        \
            int _n = _c * 8;                                                  \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GEMM_BI_TC128_BN + _n;                                     \
            int _elems = gemm_bi_cp_async_valid_elems(_gk < K, N, _gn);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _bs + (unsigned)((_k * GEMM_BI_TC128_LDB + _n) * 2);         \
            long long _offset = _bytes == 0 ? 0 : (long long)_gk * ldb + _gn; \
            const void* _src = gemm_bi_cp_async_source(B, _offset, _bytes);        \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

// Scalar staging fallback for misaligned lda/ldb (rare; uniform branch).
#define GEMM_BI_TC128_STAGE_SCALAR(buf, bkIdx, TT, FF)                               \
    do {                                                                      \
        TT* _Asw = &As[buf][0][0];                                            \
        TT* _Bsw = &Bs[buf][0][0];                                            \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC128_BM * GEMM_BI_TC128_BK; _i += 256) {           \
            int _m = _i / GEMM_BI_TC128_BK;                                              \
            int _k = _i % GEMM_BI_TC128_BK;                                              \
            int _gr = pid_m * GEMM_BI_TC128_BM + _m;                                     \
            int _gc = (bkIdx) + _k;                                           \
            _Asw[_m * GEMM_BI_TC128_LDA + _k] = (_gr < M && _gc < K)                     \
                                         ? A[(long long)_gr * lda + _gc]      \
                                         : FF(0.0f);                          \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC128_BK * GEMM_BI_TC128_BN; _i += 256) {           \
            int _k = _i / GEMM_BI_TC128_BN;                                              \
            int _n = _i % GEMM_BI_TC128_BN;                                              \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GEMM_BI_TC128_BN + _n;                                     \
            _Bsw[_k * GEMM_BI_TC128_LDB + _n] = (_gk < K && _gn < N)                     \
                                         ? B[(long long)_gk * ldb + _gn]      \
                                         : FF(0.0f);                          \
        }                                                                     \
    } while (0)

#define GEMM_BI_DEFINE_GEMM_BI_NN_TC128(SUFFIX, T_ACT, FROM_F, MMA_T)                    \
extern "C" __global__ __launch_bounds__(256, 1)                                \
void gemm_bi_nn_tc_##SUFFIX(                                                  \
    T_ACT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    const float* __restrict__ bias,                                            \
    float alpha, float beta,                                                   \
    int M, int N, int K,                                                       \
    int lda, int ldb, int ldc                                                  \
) {                                                                            \
    assert(alpha == 1.0f || bias == nullptr);                                  \
    extern __shared__ __align__(16) unsigned char gemm_bi_tc_dynsmem[];           \
    T_ACT (*As)[GEMM_BI_TC128_BM][GEMM_BI_TC128_LDA] =                                               \
        reinterpret_cast<T_ACT (*)[GEMM_BI_TC128_BM][GEMM_BI_TC128_LDA]>(gemm_bi_tc_dynsmem);            \
    T_ACT (*Bs)[GEMM_BI_TC128_BK][GEMM_BI_TC128_LDB] = reinterpret_cast<T_ACT (*)[GEMM_BI_TC128_BK][GEMM_BI_TC128_LDB]>(   \
        gemm_bi_tc_dynsmem + 2 * GEMM_BI_TC128_BM * GEMM_BI_TC128_LDA * (int)sizeof(T_ACT));             \
    int num_pid_n = (N + GEMM_BI_TC128_BN - 1) / GEMM_BI_TC128_BN;                                   \
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
    bool fast_stage = gemm_bi_is_aligned_16(A) && gemm_bi_is_aligned_16(B) &&         \
                      ((lda & 7) == 0) && ((ldb & 7) == 0) &&                  \
                      ((K & 7) == 0) && ((N & 7) == 0);                        \
    bool packed_epilogue = gemm_bi_is_aligned_4(C);                               \
    float acc[4][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 4; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            float b0 = 0.0f, b1 = 0.0f;                                        \
            if (bias != nullptr) {                                             \
                int c0 = pid_n * GEMM_BI_TC128_BN + warpN + fn * 8 + 2 * t;               \
                b0 = (c0 < N) ? bias[c0] : 0.0f;                               \
                b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                       \
            }                                                                  \
            acc[fm][fn][0] = b0;                                               \
            acc[fm][fn][1] = b1;                                               \
            acc[fm][fn][2] = b0;                                               \
            acc[fm][fn][3] = b1;                                               \
        }                                                                      \
    }                                                                          \
    int num_k_tiles = (K + GEMM_BI_TC128_BK - 1) / GEMM_BI_TC128_BK;                                 \
    if (fast_stage) {                                                          \
        GEMM_BI_TC128_STAGE_ASYNC(0, 0);                                              \
    } else {                                                                   \
        GEMM_BI_TC128_STAGE_SCALAR(0, 0, T_ACT, FROM_F);                              \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (kt + 1 < num_k_tiles) {                                            \
            if (fast_stage) {                                                  \
                GEMM_BI_TC128_STAGE_ASYNC(read_buf ^ 1, (kt + 1) * GEMM_BI_TC128_BK);            \
            } else {                                                           \
                GEMM_BI_TC128_STAGE_SCALAR(read_buf ^ 1, (kt + 1) * GEMM_BI_TC128_BK, T_ACT,     \
                                    FROM_F);                                   \
            }                                                                  \
        }                                                                      \
        unsigned As_rd = As_sbase + (unsigned)(read_buf * GEMM_BI_TC128_BM * GEMM_BI_TC128_LDA * 2); \
        unsigned Bs_rd = Bs_sbase + (unsigned)(read_buf * GEMM_BI_TC128_BK * GEMM_BI_TC128_LDB * 2); \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (GEMM_BI_TC128_BK / 16); ks++) {                            \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[4][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 4; fm++) {                                   \
                int row = warpM + fm * 16 + lm_row_off + lm_r;                 \
                unsigned addr = As_rd +                                        \
                    (unsigned)((row * GEMM_BI_TC128_LDA + k0 + lm_col_off) * 2);          \
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
                    (unsigned)((row * GEMM_BI_TC128_LDB + warpN + fn * 8) * 2);           \
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
            int r0 = pid_m * GEMM_BI_TC128_BM + warpM + fm * 16 + g;                     \
            int c0 = pid_n * GEMM_BI_TC128_BN + warpN + fn * 8 + 2 * t;                  \
            /* c0 is even. A 4-byte base and even ldc keep every row pair      \
               aligned; subviews and odd strides use the same scalar RNE. */   \
            _Pragma("unroll")                                                 \
            for (int half = 0; half < 2; half++) {                            \
                int gr = r0 + (half ? 8 : 0);                                 \
                if (gr >= M) continue;                                        \
                T_ACT* dst = gemm_bi_output_start_if_valid(                        \
                    C, (long long)gr * ldc, c0, N);                            \
                if (dst == nullptr) continue;                                 \
                float v0 = alpha * acc[fm][fn][2 * half];                     \
                float v1 = alpha * acc[fm][fn][2 * half + 1];                 \
                if (beta == 0.0f && packed_epilogue && c0 + 1 < N &&         \
                    gemm_bi_is_aligned_4(dst)) {                                   \
                    gemm_bi_store_pair_rne(dst, v0, v1);                           \
                } else {                                                      \
                    for (int e = 0; e < 2; e++) {                             \
                        int gc = c0 + e;                                      \
                        if (gc >= N) continue;                                \
                        float val = e ? v1 : v0;                              \
                        if (beta != 0.0f)                                     \
                            val += beta * to_f(C[(long long)gr * ldc + gc]);  \
                        dst[e] = FROM_F(val);                                  \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

GEMM_BI_DEFINE_GEMM_BI_NN_TC128(bf16, __nv_bfloat16, from_f_bf16, "bf16")
GEMM_BI_DEFINE_GEMM_BI_NN_TC128(f16,  __half,        from_f_f16,  "f16")

// ============================================================================
// Tensor-core backward twins: TN dW and NT dX (bi_tensor_cores tier).
// ============================================================================
// Same numeric contract class as gemm_bi_nn_tc_*: deterministic (fixed
// reduction order, fixed fragment/tile assignment, no atomics, no split),
// f32 mma accumulation. dW accumulates into the f32 master (+=, no
// downcast); dX is a typed RNE overwrite.
//
// TN (dW): C[K_out,N] += X^T[K_out,M] @ dY[M,N], reduction over M.
//   Xs[m][k_out] and dYs[m][n] staged in GLOBAL layout (cp.async 16B) —
//   the transposed A-fragments come from ldmatrix.x4.TRANS, dY B-fragments
//   from ldmatrix.x2.TRANS (stored rows are the reduction dim, exactly the
//   NN-B pattern).
// NT (dX): C[M,K_out] = dY[M,N] @ W^T[N,K_out], reduction over N.
//   dYs[m][n] (plain x4, the NN-A pattern) and Ws[k_out][n] (plain x2:
//   fragment k = n lives along the stored row) — both global-layout,
//   cp.async 16B.
//
// Smem strides keep ldmatrix row chunks in distinct 4-bank groups:
//   [.][136] rows: 68 words ≡ 4 (mod 8); [.][40] rows: 20 words ≡ 4 (mod 8).

// TN staging: Xs[m_local][k_out chunk], dYs[m_local][n chunk]; both rows
// are the M-reduction dim (GEMM_BI_TC128_BK rows per tile).
#define GEMM_BI_TC128_STAGE_TN_ASYNC(buf, mIdx)                                      \
    do {                                                                      \
        unsigned _xs = Xs_sbase + (unsigned)((buf) * GEMM_BI_TC128_BK * GEMM_BI_TC128_LDB * 2);     \
        unsigned _ys = Ys_sbase + (unsigned)((buf) * GEMM_BI_TC128_BK * GEMM_BI_TC128_LDB * 2);     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC128_BK * (GEMM_BI_TC128_BM / 8); _i += 256) {     \
            int _r = _i / (GEMM_BI_TC128_BM / 8);                                        \
            int _c = (_i % (GEMM_BI_TC128_BM / 8)) * 8;                                  \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * GEMM_BI_TC128_BM + _c;                                     \
            int _elems = gemm_bi_cp_async_valid_elems(_gm < M_red, K_out, _gk);   \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _xs + (unsigned)((_r * GEMM_BI_TC128_LDB + _c) * 2);         \
            long long _offset =                                                \
                _bytes == 0 ? 0 : (long long)_gm * K_out + _gk;               \
            const void* _src = gemm_bi_cp_async_source(A, _offset, _bytes);        \
            gemm_bi_cp_async_16_zfill_l2(_dst, _src, _bytes);                      \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC128_BK * (GEMM_BI_TC128_BN / 8); _i += 256) {     \
            int _r = _i / (GEMM_BI_TC128_BN / 8);                                        \
            int _c = (_i % (GEMM_BI_TC128_BN / 8)) * 8;                                  \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_TC128_BN + _c;                                     \
            int _elems = gemm_bi_cp_async_valid_elems(_gm < M_red, N, _gn);       \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ys + (unsigned)((_r * GEMM_BI_TC128_LDB + _c) * 2);         \
            long long _offset = _bytes == 0 ? 0 : (long long)_gm * N + _gn;   \
            const void* _src = gemm_bi_cp_async_source(B, _offset, _bytes);        \
            gemm_bi_cp_async_16_zfill_l2(_dst, _src, _bytes);                      \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define GEMM_BI_TC128_STAGE_TN_SCALAR(buf, mIdx, TT, FF)                             \
    do {                                                                      \
        TT* _xs = &Xs[buf][0][0];                                             \
        TT* _ys = &Ys[buf][0][0];                                             \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC128_BK * GEMM_BI_TC128_BM; _i += 256) {           \
            int _r = _i / GEMM_BI_TC128_BM;                                              \
            int _c = _i % GEMM_BI_TC128_BM;                                              \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * GEMM_BI_TC128_BM + _c;                                     \
            _xs[_r * GEMM_BI_TC128_LDB + _c] = (_gm < M_red && _gk < K_out)              \
                                        ? A[(long long)_gm * K_out + _gk]     \
                                        : FF(0.0f);                           \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC128_BK * GEMM_BI_TC128_BN; _i += 256) {           \
            int _r = _i / GEMM_BI_TC128_BN;                                              \
            int _c = _i % GEMM_BI_TC128_BN;                                              \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_TC128_BN + _c;                                     \
            _ys[_r * GEMM_BI_TC128_LDB + _c] = (_gm < M_red && _gn < N)                  \
                                        ? B[(long long)_gm * N + _gn]         \
                                        : FF(0.0f);                           \
        }                                                                     \
    } while (0)

#define GEMM_BI_DEFINE_GEMM_BI_TN_TC128(SUFFIX, T_ACT, FROM_F, MMA_T)                    \
extern "C" __global__ __launch_bounds__(256, 1)                                \
void gemm_bi_tn_tc_##SUFFIX(                                                  \
    float* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M_red, int K_out, int N                                                \
) {                                                                            \
    extern __shared__ __align__(16) unsigned char gemm_bi_tc_dynsmem[];           \
    T_ACT (*Xs)[GEMM_BI_TC128_BK][GEMM_BI_TC128_LDB] =                                               \
        reinterpret_cast<T_ACT (*)[GEMM_BI_TC128_BK][GEMM_BI_TC128_LDB]>(gemm_bi_tc_dynsmem);            \
    T_ACT (*Ys)[GEMM_BI_TC128_BK][GEMM_BI_TC128_LDB] = reinterpret_cast<T_ACT (*)[GEMM_BI_TC128_BK][GEMM_BI_TC128_LDB]>(   \
        gemm_bi_tc_dynsmem + 2 * GEMM_BI_TC128_BK * GEMM_BI_TC128_LDB * (int)sizeof(T_ACT));             \
    int num_pid_n = (N + GEMM_BI_TC128_BN - 1) / GEMM_BI_TC128_BN;                                   \
    int pid_m = blockIdx.x / num_pid_n;                                        \
    int pid_n = blockIdx.x % num_pid_n;                                        \
    int warp = threadIdx.x / 32;                                               \
    int lane = threadIdx.x % 32;                                               \
    int warpM = (warp / 4) * 64;                                               \
    int warpN = (warp % 4) * 32;                                               \
    int g = lane >> 2;                                                         \
    int t = lane & 3;                                                          \
    int lm_r = lane & 7;                                                       \
    int lm_q = lane >> 3;                                                      \
    /* A x4.trans quadrants: stored-row off (m) = (q&2)?8:0, col off (ko) =  */\
    /* (q&1)?8:0. B x2.trans: stored-row off (m) = (q&1)?8:0.                */\
    int lm_arow_off = (lm_q & 2) ? 8 : 0;                                      \
    int lm_acol_off = (lm_q & 1) ? 8 : 0;                                      \
    int lm_brow_off = (lm_q & 1) ? 8 : 0;                                      \
    unsigned Xs_sbase = (unsigned)__cvta_generic_to_shared(&Xs[0][0][0]);      \
    unsigned Ys_sbase = (unsigned)__cvta_generic_to_shared(&Ys[0][0][0]);      \
    bool fast_stage = gemm_bi_is_aligned_16(A) && gemm_bi_is_aligned_16(B) &&         \
                      ((K_out & 7) == 0) && ((N & 7) == 0);                    \
    bool packed_epilogue = gemm_bi_is_aligned_8(C) && ((N & 1) == 0);             \
    float acc[4][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 4; fm++)                                             \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++)                                         \
            _Pragma("unroll")                                                  \
            for (int e = 0; e < 4; e++) acc[fm][fn][e] = 0.0f;                 \
    int num_m_tiles = (M_red + GEMM_BI_TC128_BK - 1) / GEMM_BI_TC128_BK;                             \
    if (fast_stage) {                                                          \
        GEMM_BI_TC128_STAGE_TN_ASYNC(0, 0);                                           \
    } else {                                                                   \
        GEMM_BI_TC128_STAGE_TN_SCALAR(0, 0, T_ACT, FROM_F);                           \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int mt = 0; mt < num_m_tiles; mt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (mt + 1 < num_m_tiles) {                                            \
            if (fast_stage) {                                                  \
                GEMM_BI_TC128_STAGE_TN_ASYNC(read_buf ^ 1, (mt + 1) * GEMM_BI_TC128_BK);         \
            } else {                                                           \
                GEMM_BI_TC128_STAGE_TN_SCALAR(read_buf ^ 1, (mt + 1) * GEMM_BI_TC128_BK, T_ACT,  \
                                       FROM_F);                                \
            }                                                                  \
        }                                                                      \
        unsigned Xs_rd = Xs_sbase + (unsigned)(read_buf * GEMM_BI_TC128_BK * GEMM_BI_TC128_LDB * 2); \
        unsigned Ys_rd = Ys_sbase + (unsigned)(read_buf * GEMM_BI_TC128_BK * GEMM_BI_TC128_LDB * 2); \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (GEMM_BI_TC128_BK / 16); ks++) {                            \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[4][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 4; fm++) {                                   \
                int srow = k0 + lm_arow_off + lm_r;                            \
                int scol = warpM + fm * 16 + lm_acol_off;                      \
                unsigned addr =                                                \
                    Xs_rd + (unsigned)((srow * GEMM_BI_TC128_LDB + scol) * 2);            \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "          \
                    "{%0,%1,%2,%3}, [%4];\n"                                   \
                    : "=r"(a_frag[fm][0]), "=r"(a_frag[fm][1]),                \
                      "=r"(a_frag[fm][2]), "=r"(a_frag[fm][3])                 \
                    : "r"(addr));                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int fn = 0; fn < 4; fn++) {                                   \
                int srow = k0 + lm_brow_off + lm_r;                            \
                unsigned addr = Ys_rd +                                        \
                    (unsigned)((srow * GEMM_BI_TC128_LDB + warpN + fn * 8) * 2);          \
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
    /* A float2 RMW needs an 8-byte base and even N; otherwise both values     \
       take the same per-element load, multiply, add and scalar store. */      \
    _Pragma("unroll")                                                         \
    for (int fm = 0; fm < 4; fm++) {                                          \
        _Pragma("unroll")                                                     \
        for (int fn = 0; fn < 4; fn++) {                                      \
            int r0 = pid_m * GEMM_BI_TC128_BM + warpM + fm * 16 + g;                     \
            int c0 = pid_n * GEMM_BI_TC128_BN + warpN + fn * 8 + 2 * t;                  \
            _Pragma("unroll")                                                 \
            for (int half = 0; half < 2; half++) {                            \
                int gr = r0 + (half ? 8 : 0);                                 \
                if (gr >= K_out) continue;                                    \
                float* dst = gemm_bi_output_start_if_valid(                        \
                    C, (long long)gr * N, c0, N);                              \
                if (dst == nullptr) continue;                                 \
                if (c0 + 1 < N) {                                             \
                    float x = alpha * acc[fm][fn][2 * half];                   \
                    float y = alpha * acc[fm][fn][2 * half + 1];               \
                    bool packed = packed_epilogue && gemm_bi_is_aligned_8(dst);    \
                    gemm_bi_accumulate_float2_or_scalar(dst, x, y, packed);        \
                } else {                                                      \
                    for (int e = 0; e < 2; e++) {                             \
                        int gc = c0 + e;                                      \
                        if (gc >= N) continue;                                \
                        dst[e] += alpha * acc[fm][fn][2 * half + e];          \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

GEMM_BI_DEFINE_GEMM_BI_TN_TC128(bf16, __nv_bfloat16, from_f_bf16, "bf16")
GEMM_BI_DEFINE_GEMM_BI_TN_TC128(f16,  __half,        from_f_f16,  "f16")

// NT staging: dYs[m_local][n chunk] (output rows x reduction) and
// Ws[k_out_local][n chunk] (output cols x reduction).
#define GEMM_BI_TC128_STAGE_NT_ASYNC(buf, nIdx)                                      \
    do {                                                                      \
        unsigned _ys = Ys_sbase + (unsigned)((buf) * GEMM_BI_TC128_BM * GEMM_BI_TC128_LDA * 2);     \
        unsigned _ws = Ws_sbase + (unsigned)((buf) * GEMM_BI_TC128_BN * GEMM_BI_TC128_LDA * 2);     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC128_BM * (GEMM_BI_TC128_BK / 8); _i += 256) {     \
            int _m = _i / (GEMM_BI_TC128_BK / 8);                                        \
            int _c = (_i % (GEMM_BI_TC128_BK / 8)) * 8;                                  \
            int _gm = pid_m * GEMM_BI_TC128_BM + _m;                                     \
            int _gn = (nIdx) + _c;                                            \
            int _elems = gemm_bi_cp_async_valid_elems(_gm < M, N, _gn);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ys + (unsigned)((_m * GEMM_BI_TC128_LDA + _c) * 2);         \
            long long _offset = _bytes == 0 ? 0 : (long long)_gm * N + _gn;   \
            const void* _src = gemm_bi_cp_async_source(A, _offset, _bytes);        \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC128_BN * (GEMM_BI_TC128_BK / 8); _i += 256) {     \
            int _k = _i / (GEMM_BI_TC128_BK / 8);                                        \
            int _c = (_i % (GEMM_BI_TC128_BK / 8)) * 8;                                  \
            int _gk = pid_n * GEMM_BI_TC128_BN + _k;                                     \
            int _gn = (nIdx) + _c;                                            \
            int _elems = gemm_bi_cp_async_valid_elems(_gk < K_out, N, _gn);       \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ws + (unsigned)((_k * GEMM_BI_TC128_LDA + _c) * 2);         \
            long long _offset = _bytes == 0 ? 0 : (long long)_gk * N + _gn;   \
            const void* _src = gemm_bi_cp_async_source(B, _offset, _bytes);        \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define GEMM_BI_TC128_STAGE_NT_SCALAR(buf, nIdx, TT, FF)                             \
    do {                                                                      \
        TT* _ys = &Ys[buf][0][0];                                             \
        TT* _ws = &Ws[buf][0][0];                                             \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC128_BM * GEMM_BI_TC128_BK; _i += 256) {           \
            int _m = _i / GEMM_BI_TC128_BK;                                              \
            int _c = _i % GEMM_BI_TC128_BK;                                              \
            int _gm = pid_m * GEMM_BI_TC128_BM + _m;                                     \
            int _gn = (nIdx) + _c;                                            \
            _ys[_m * GEMM_BI_TC128_LDA + _c] = (_gm < M && _gn < N)                      \
                                        ? A[(long long)_gm * N + _gn]         \
                                        : FF(0.0f);                           \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC128_BN * GEMM_BI_TC128_BK; _i += 256) {           \
            int _k = _i / GEMM_BI_TC128_BK;                                              \
            int _c = _i % GEMM_BI_TC128_BK;                                              \
            int _gk = pid_n * GEMM_BI_TC128_BN + _k;                                     \
            int _gn = (nIdx) + _c;                                            \
            _ws[_k * GEMM_BI_TC128_LDA + _c] = (_gk < K_out && _gn < N)                  \
                                        ? B[(long long)_gk * N + _gn]         \
                                        : FF(0.0f);                           \
        }                                                                     \
    } while (0)

#define GEMM_BI_DEFINE_GEMM_BI_NT_TC128(SUFFIX, T_ACT, FROM_F, MMA_T)                    \
extern "C" __global__ __launch_bounds__(256, 1)                                \
void gemm_bi_nt_tc_##SUFFIX(                                                  \
    T_ACT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M, int N, int K_out                                                    \
) {                                                                            \
    extern __shared__ __align__(16) unsigned char gemm_bi_tc_dynsmem[];           \
    T_ACT (*Ys)[GEMM_BI_TC128_BM][GEMM_BI_TC128_LDA] =                                               \
        reinterpret_cast<T_ACT (*)[GEMM_BI_TC128_BM][GEMM_BI_TC128_LDA]>(gemm_bi_tc_dynsmem);            \
    T_ACT (*Ws)[GEMM_BI_TC128_BN][GEMM_BI_TC128_LDA] = reinterpret_cast<T_ACT (*)[GEMM_BI_TC128_BN][GEMM_BI_TC128_LDA]>(   \
        gemm_bi_tc_dynsmem + 2 * GEMM_BI_TC128_BM * GEMM_BI_TC128_LDA * (int)sizeof(T_ACT));             \
    int num_pid_n = (K_out + GEMM_BI_TC128_BN - 1) / GEMM_BI_TC128_BN;                               \
    int pid_m = blockIdx.x / num_pid_n;                                        \
    int pid_n = blockIdx.x % num_pid_n;                                        \
    int warp = threadIdx.x / 32;                                               \
    int lane = threadIdx.x % 32;                                               \
    int warpM = (warp / 4) * 64;                                               \
    int warpN = (warp % 4) * 32;                                               \
    int g = lane >> 2;                                                         \
    int t = lane & 3;                                                          \
    int lm_r = lane & 7;                                                       \
    int lm_q = lane >> 3;                                                      \
    int lm_row_off = (lm_q & 1) ? 8 : 0;                                       \
    int lm_col_off = (lm_q & 2) ? 8 : 0;                                       \
    int lmb_col_off = (lm_q & 1) ? 8 : 0;                                      \
    unsigned Ys_sbase = (unsigned)__cvta_generic_to_shared(&Ys[0][0][0]);      \
    unsigned Ws_sbase = (unsigned)__cvta_generic_to_shared(&Ws[0][0][0]);      \
    bool fast_stage = gemm_bi_is_aligned_16(A) && gemm_bi_is_aligned_16(B) &&         \
                      ((N & 7) == 0);                                          \
    bool packed_epilogue = gemm_bi_is_aligned_4(C) && ((K_out & 1) == 0);         \
    float acc[4][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 4; fm++)                                             \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++)                                         \
            _Pragma("unroll")                                                  \
            for (int e = 0; e < 4; e++) acc[fm][fn][e] = 0.0f;                 \
    int num_n_tiles = (N + GEMM_BI_TC128_BK - 1) / GEMM_BI_TC128_BK;                                 \
    if (fast_stage) {                                                          \
        GEMM_BI_TC128_STAGE_NT_ASYNC(0, 0);                                           \
    } else {                                                                   \
        GEMM_BI_TC128_STAGE_NT_SCALAR(0, 0, T_ACT, FROM_F);                           \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int nt = 0; nt < num_n_tiles; nt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (nt + 1 < num_n_tiles) {                                            \
            if (fast_stage) {                                                  \
                GEMM_BI_TC128_STAGE_NT_ASYNC(read_buf ^ 1, (nt + 1) * GEMM_BI_TC128_BK);         \
            } else {                                                           \
                GEMM_BI_TC128_STAGE_NT_SCALAR(read_buf ^ 1, (nt + 1) * GEMM_BI_TC128_BK, T_ACT,  \
                                       FROM_F);                                \
            }                                                                  \
        }                                                                      \
        unsigned Ys_rd = Ys_sbase + (unsigned)(read_buf * GEMM_BI_TC128_BM * GEMM_BI_TC128_LDA * 2); \
        unsigned Ws_rd = Ws_sbase + (unsigned)(read_buf * GEMM_BI_TC128_BN * GEMM_BI_TC128_LDA * 2); \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (GEMM_BI_TC128_BK / 16); ks++) {                            \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[4][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 4; fm++) {                                   \
                int row = warpM + fm * 16 + lm_row_off + lm_r;                 \
                unsigned addr = Ys_rd +                                        \
                    (unsigned)((row * GEMM_BI_TC128_LDA + k0 + lm_col_off) * 2);          \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "                \
                    "{%0,%1,%2,%3}, [%4];\n"                                   \
                    : "=r"(a_frag[fm][0]), "=r"(a_frag[fm][1]),                \
                      "=r"(a_frag[fm][2]), "=r"(a_frag[fm][3])                 \
                    : "r"(addr));                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int fn = 0; fn < 4; fn++) {                                   \
                int row = warpN + fn * 8 + lm_r;                               \
                unsigned addr = Ws_rd +                                        \
                    (unsigned)((row * GEMM_BI_TC128_LDA + k0 + lmb_col_off) * 2);         \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "                \
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
    /* A 4-byte base and even K_out keep each RNE pair aligned. A half-offset  \
       subview or tail uses identical scalar conversions. */                  \
    _Pragma("unroll")                                                         \
    for (int fm = 0; fm < 4; fm++) {                                          \
        _Pragma("unroll")                                                     \
        for (int fn = 0; fn < 4; fn++) {                                      \
            int r0 = pid_m * GEMM_BI_TC128_BM + warpM + fm * 16 + g;                     \
            int c0 = pid_n * GEMM_BI_TC128_BN + warpN + fn * 8 + 2 * t;                  \
            _Pragma("unroll")                                                 \
            for (int half = 0; half < 2; half++) {                            \
                int gr = r0 + (half ? 8 : 0);                                 \
                if (gr >= M) continue;                                        \
                T_ACT* dst = gemm_bi_output_start_if_valid(                        \
                    C, (long long)gr * K_out, c0, K_out);                      \
                if (dst == nullptr) continue;                                 \
                float v0 = alpha * acc[fm][fn][2 * half];                     \
                float v1 = alpha * acc[fm][fn][2 * half + 1];                 \
                if (packed_epilogue && c0 + 1 < K_out &&                     \
                    gemm_bi_is_aligned_4(dst)) {                                   \
                    gemm_bi_store_pair_rne(dst, v0, v1);                           \
                } else {                                                      \
                    for (int e = 0; e < 2; e++) {                             \
                        int gc = c0 + e;                                      \
                        if (gc >= K_out) continue;                            \
                        dst[e] = FROM_F(e ? v1 : v0);                          \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

GEMM_BI_DEFINE_GEMM_BI_NT_TC128(bf16, __nv_bfloat16, from_f_bf16, "bf16")
GEMM_BI_DEFINE_GEMM_BI_NT_TC128(f16,  __half,        from_f_f16,  "f16")

// ============================================================================
// 64x64 tensor-core twins for small shapes in the bi_tensor_cores tier.
// ============================================================================
// Same numeric contract class as the 128-tile TC kernels above — and one
// property stronger: BIT-IDENTICAL to them per output element. All three
// walk the reduction dim (K for NN, M for TN, N for NT) in ascending
// BK-wide slabs split into ascending m16n8k16 mma steps (BK lockstep across
// both families: 64), with the same
// 16B-chunk zero-fill for tails, so every output element's f32 accumulator
// sees the exact same mma chain regardless of which tile size the
// dispatcher picked. That bit-match is what makes the underfill-aware
// Tile64/Tile128 routing in gpu/gemm_bi.rs legal under the strict all-M
// invariance contract (tests/gemm_bi_tc.rs asserts the cross-tile
// bit-identity directly). Do NOT change the slab width, the ks order, or
// the tail zero-fill here without changing the 128-tile kernels in
// lockstep.
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
// Every constant below is section-local (GEMM_BI_TC64_*).
// NEVER reference GEMM_BI_TC128_BM/GEMM_BI_TC128_BN/GEMM_BI_TC128_BK/GEMM_BI_TC128_LDA/GEMM_BI_TC128_LDB or any other ambient
// define from earlier sections inside this section.

#define GEMM_BI_TC64_BM 64
#define GEMM_BI_TC64_BN 64
#define GEMM_BI_TC64_BK 64
#define GEMM_BI_TC64_THREADS 128
#define GEMM_BI_TC64_LDA (GEMM_BI_TC64_BK + 8) /* 72 halves = 144 B rows, 36 words ≡ 4 mod 8 */
#define GEMM_BI_TC64_LDB (GEMM_BI_TC64_BN + 8) /* 72 halves = 144 B rows, 36 words ≡ 4 mod 8 */

// NN staging: A 64 rows x 4 chunks + B 32 rows x 8 chunks = 512 cp.async
// over 128 threads (4 per thread), 16B each with zero-fill tails.
#define GEMM_BI_TC64_STAGE_ASYNC(buf, bkIdx)                                      \
    do {                                                                      \
        unsigned _as =                                                        \
            As_sbase + (unsigned)((buf) * GEMM_BI_TC64_BM * GEMM_BI_TC64_LDA * 2);    \
        unsigned _bs =                                                        \
            Bs_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2);    \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BM * (GEMM_BI_TC64_BK / 8);      \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _m = _i / (GEMM_BI_TC64_BK / 8);                                  \
            int _k = (_i % (GEMM_BI_TC64_BK / 8)) * 8;                            \
            int _gr = pid_m * GEMM_BI_TC64_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            int _elems = gemm_bi_cp_async_valid_elems(_gr < M, K, _gc);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _as + (unsigned)((_m * GEMM_BI_TC64_LDA + _k) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gr * lda + _gc; \
            const void* _src = gemm_bi_cp_async_source(A, _offset, _bytes);        \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BK * (GEMM_BI_TC64_BN / 8);      \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _k = _i / (GEMM_BI_TC64_BN / 8);                                  \
            int _n = (_i % (GEMM_BI_TC64_BN / 8)) * 8;                            \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GEMM_BI_TC64_BN + _n;                               \
            int _elems = gemm_bi_cp_async_valid_elems(_gk < K, N, _gn);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _bs + (unsigned)((_k * GEMM_BI_TC64_LDB + _n) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gk * ldb + _gn; \
            const void* _src = gemm_bi_cp_async_source(B, _offset, _bytes);        \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

// Scalar staging fallback for misaligned lda/ldb (rare; uniform branch).
#define GEMM_BI_TC64_STAGE_SCALAR(buf, bkIdx, TT, FF)                             \
    do {                                                                      \
        TT* _Asw = &As[buf][0][0];                                            \
        TT* _Bsw = &Bs[buf][0][0];                                            \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BM * GEMM_BI_TC64_BK;            \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _m = _i / GEMM_BI_TC64_BK;                                        \
            int _k = _i % GEMM_BI_TC64_BK;                                        \
            int _gr = pid_m * GEMM_BI_TC64_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            _Asw[_m * GEMM_BI_TC64_LDA + _k] = (_gr < M && _gc < K)               \
                                               ? A[(long long)_gr * lda + _gc]\
                                               : FF(0.0f);                    \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BK * GEMM_BI_TC64_BN;            \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _k = _i / GEMM_BI_TC64_BN;                                        \
            int _n = _i % GEMM_BI_TC64_BN;                                        \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GEMM_BI_TC64_BN + _n;                               \
            _Bsw[_k * GEMM_BI_TC64_LDB + _n] = (_gk < K && _gn < N)               \
                                               ? B[(long long)_gk * ldb + _gn]\
                                               : FF(0.0f);                    \
        }                                                                     \
    } while (0)

#define GEMM_BI_DEFINE_GEMM_BI_NN_TC64(SUFFIX, T_ACT, FROM_F, MMA_T)                  \
extern "C" __global__ __launch_bounds__(GEMM_BI_TC64_THREADS, 1)                   \
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
    __shared__ __align__(16) T_ACT As[2][GEMM_BI_TC64_BM][GEMM_BI_TC64_LDA];           \
    __shared__ __align__(16) T_ACT Bs[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB];           \
    int num_pid_n = (N + GEMM_BI_TC64_BN - 1) / GEMM_BI_TC64_BN;                       \
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
    bool fast_stage = gemm_bi_is_aligned_16(A) && gemm_bi_is_aligned_16(B) &&         \
                      ((lda & 7) == 0) && ((ldb & 7) == 0) &&                  \
                      ((K & 7) == 0) && ((N & 7) == 0);                        \
    float acc[2][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            float b0 = 0.0f, b1 = 0.0f;                                        \
            if (bias != nullptr) {                                             \
                int c0 = pid_n * GEMM_BI_TC64_BN + warpN + fn * 8 + 2 * t;         \
                b0 = (c0 < N) ? bias[c0] : 0.0f;                               \
                b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                       \
            }                                                                  \
            acc[fm][fn][0] = b0;                                               \
            acc[fm][fn][1] = b1;                                               \
            acc[fm][fn][2] = b0;                                               \
            acc[fm][fn][3] = b1;                                               \
        }                                                                      \
    }                                                                          \
    int num_k_tiles = (K + GEMM_BI_TC64_BK - 1) / GEMM_BI_TC64_BK;                     \
    if (fast_stage) {                                                          \
        GEMM_BI_TC64_STAGE_ASYNC(0, 0);                                            \
    } else {                                                                   \
        GEMM_BI_TC64_STAGE_SCALAR(0, 0, T_ACT, FROM_F);                            \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (kt + 1 < num_k_tiles) {                                            \
            if (fast_stage) {                                                  \
                GEMM_BI_TC64_STAGE_ASYNC(read_buf ^ 1, (kt + 1) * GEMM_BI_TC64_BK);    \
            } else {                                                           \
                GEMM_BI_TC64_STAGE_SCALAR(read_buf ^ 1, (kt + 1) * GEMM_BI_TC64_BK,    \
                                      T_ACT, FROM_F);                          \
            }                                                                  \
        }                                                                      \
        unsigned As_rd =                                                       \
            As_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BM * GEMM_BI_TC64_LDA * 2);  \
        unsigned Bs_rd =                                                       \
            Bs_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2);  \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (GEMM_BI_TC64_BK / 16); ks++) {                            \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[2][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 2; fm++) {                                   \
                int row = warpM + fm * 16 + lm_row_off + lm_r;                 \
                unsigned addr = As_rd +                                        \
                    (unsigned)((row * GEMM_BI_TC64_LDA + k0 + lm_col_off) * 2);    \
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
                    (unsigned)((row * GEMM_BI_TC64_LDB + warpN + fn * 8) * 2);     \
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
            int r0 = pid_m * GEMM_BI_TC64_BM + warpM + fm * 16 + g;                \
            int c0 = pid_n * GEMM_BI_TC64_BN + warpN + fn * 8 + 2 * t;             \
            _Pragma("unroll")                                                  \
            for (int e = 0; e < 4; e++) {                                      \
                int gr = r0 + (e >= 2 ? 8 : 0);                                \
                int gc = c0 + (e & 1);                                         \
                if (gr >= M || gc >= N) continue;                              \
                float val = alpha * acc[fm][fn][e];                            \
                if (beta != 0.0f)                                              \
                    val += beta * to_f(C[(long long)gr * ldc + gc]);           \
                C[(long long)gr * ldc + gc] = FROM_F(val);                     \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

GEMM_BI_DEFINE_GEMM_BI_NN_TC64(bf16, __nv_bfloat16, from_f_bf16, "bf16")
GEMM_BI_DEFINE_GEMM_BI_NN_TC64(f16,  __half,        from_f_f16,  "f16")

// ── Thin16 rung (R0 of the 0.6.10 ladder): 16x32x64, 4 warps, 4-stage ──
//
// The decode/thin-M rung of the bit-identical tile ladder. Same
// arithmetic contract as TC64/TC128 (ascending m16n8k16 K-slabs, f32
// accumulators, bias pre-seeded, single RNE downcast) - the census gate
// asserts byte-identity against Tile64. What differs is SCHEDULING: a
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
// Every constant below is section-local (GEMM_BI_TC16_*).
#define GEMM_BI_TC16_BM 16
#define GEMM_BI_TC16_BN 32
#define GEMM_BI_TC16_BK 64
#define GEMM_BI_TC16_THREADS 128
#define GEMM_BI_TC16_STAGES 4
#define GEMM_BI_TC16_LDA (GEMM_BI_TC16_BK + 8) /* 72 halves = 144 B rows */
#define GEMM_BI_TC16_LDB (GEMM_BI_TC16_BN + 8) /* 40 halves = 80 B rows, 20 words == 4 mod 8 */

#define GEMM_BI_TC16_STAGE_ASYNC(buf, bkIdx)                                      \
    do {                                                                      \
        unsigned _as =                                                        \
            As_sbase + (unsigned)((buf) * GEMM_BI_TC16_BM * GEMM_BI_TC16_LDA * 2);    \
        unsigned _bs =                                                        \
            Bs_sbase + (unsigned)((buf) * GEMM_BI_TC16_BK * GEMM_BI_TC16_LDB * 2);    \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC16_BM * (GEMM_BI_TC16_BK / 8);      \
             _i += GEMM_BI_TC16_THREADS) {                                        \
            int _m = _i / (GEMM_BI_TC16_BK / 8);                                  \
            int _k = (_i % (GEMM_BI_TC16_BK / 8)) * 8;                            \
            int _gr = pid_m * GEMM_BI_TC16_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            int _elems = gemm_bi_cp_async_valid_elems(_gr < M, K, _gc);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _as + (unsigned)((_m * GEMM_BI_TC16_LDA + _k) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gr * lda + _gc; \
            const void* _src = gemm_bi_cp_async_source(A, _offset, _bytes);        \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC16_BK * (GEMM_BI_TC16_BN / 8);      \
             _i += GEMM_BI_TC16_THREADS) {                                        \
            int _k = _i / (GEMM_BI_TC16_BN / 8);                                  \
            int _n = (_i % (GEMM_BI_TC16_BN / 8)) * 8;                            \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GEMM_BI_TC16_BN + _n;                               \
            int _elems = gemm_bi_cp_async_valid_elems(_gk < K, N, _gn);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _bs + (unsigned)((_k * GEMM_BI_TC16_LDB + _n) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gk * ldb + _gn; \
            const void* _src = gemm_bi_cp_async_source(B, _offset, _bytes);        \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define GEMM_BI_TC16_STAGE_SCALAR(buf, bkIdx, TT, FF)                             \
    do {                                                                      \
        TT* _Asw = &As[buf][0][0];                                            \
        TT* _Bsw = &Bs[buf][0][0];                                            \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC16_BM * GEMM_BI_TC16_BK;            \
             _i += GEMM_BI_TC16_THREADS) {                                        \
            int _m = _i / GEMM_BI_TC16_BK;                                        \
            int _k = _i % GEMM_BI_TC16_BK;                                        \
            int _gr = pid_m * GEMM_BI_TC16_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            _Asw[_m * GEMM_BI_TC16_LDA + _k] = (_gr < M && _gc < K)               \
                                               ? A[(long long)_gr * lda + _gc]\
                                               : FF(0.0f);                    \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC16_BK * GEMM_BI_TC16_BN;            \
             _i += GEMM_BI_TC16_THREADS) {                                        \
            int _k = _i / GEMM_BI_TC16_BN;                                        \
            int _n = _i % GEMM_BI_TC16_BN;                                        \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GEMM_BI_TC16_BN + _n;                               \
            _Bsw[_k * GEMM_BI_TC16_LDB + _n] = (_gk < K && _gn < N)               \
                                               ? B[(long long)_gk * ldb + _gn]\
                                               : FF(0.0f);                    \
        }                                                                     \
    } while (0)

#define GEMM_BI_DEFINE_GEMM_BI_NN_TC16(SUFFIX, T_ACT, FROM_F, MMA_T)                  \
extern "C" __global__ __launch_bounds__(GEMM_BI_TC16_THREADS, 3)                   \
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
        T_ACT As[GEMM_BI_TC16_STAGES][GEMM_BI_TC16_BM][GEMM_BI_TC16_LDA];                  \
    __shared__ __align__(16)                                                   \
        T_ACT Bs[GEMM_BI_TC16_STAGES][GEMM_BI_TC16_BK][GEMM_BI_TC16_LDB];                  \
    int num_pid_n = (N + GEMM_BI_TC16_BN - 1) / GEMM_BI_TC16_BN;                       \
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
    bool fast_stage = gemm_bi_is_aligned_16(A) && gemm_bi_is_aligned_16(B) &&         \
                      ((lda & 7) == 0) && ((ldb & 7) == 0) &&                  \
                      ((K & 7) == 0) && ((N & 7) == 0);                        \
    float acc[4];                                                              \
    {                                                                          \
        float b0 = 0.0f, b1 = 0.0f;                                            \
        if (bias != nullptr) {                                                 \
            int c0 = pid_n * GEMM_BI_TC16_BN + warpN + 2 * t;                      \
            b0 = (c0 < N) ? bias[c0] : 0.0f;                                   \
            b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                           \
        }                                                                      \
        acc[0] = b0;                                                           \
        acc[1] = b1;                                                           \
        acc[2] = b0;                                                           \
        acc[3] = b1;                                                           \
    }                                                                          \
    int num_k_tiles = (K + GEMM_BI_TC16_BK - 1) / GEMM_BI_TC16_BK;                     \
    /* Prologue: STAGES-1 commit groups, real or empty - uniform count. */    \
    for (int p = 0; p < GEMM_BI_TC16_STAGES - 1; p++) {                            \
        if (p < num_k_tiles) {                                                 \
            if (fast_stage) {                                                  \
                GEMM_BI_TC16_STAGE_ASYNC(p, p * GEMM_BI_TC16_BK);                      \
            } else {                                                           \
                GEMM_BI_TC16_STAGE_SCALAR(p, p * GEMM_BI_TC16_BK, T_ACT, FROM_F);      \
                asm volatile("cp.async.commit_group;\n");                     \
            }                                                                  \
        } else {                                                               \
            asm volatile("cp.async.commit_group;\n");                         \
        }                                                                      \
    }                                                                          \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                 \
        asm volatile("cp.async.wait_group %0;\n"                              \
                     :: "n"(GEMM_BI_TC16_STAGES - 2));                             \
        __syncthreads();                                                       \
        int next = kt + GEMM_BI_TC16_STAGES - 1;                                   \
        if (next < num_k_tiles) {                                              \
            int wbuf = next % GEMM_BI_TC16_STAGES;                                 \
            if (fast_stage) {                                                  \
                GEMM_BI_TC16_STAGE_ASYNC(wbuf, next * GEMM_BI_TC16_BK);                \
            } else {                                                           \
                GEMM_BI_TC16_STAGE_SCALAR(wbuf, next * GEMM_BI_TC16_BK, T_ACT,         \
                                      FROM_F);                                 \
                asm volatile("cp.async.commit_group;\n");                     \
            }                                                                  \
        } else {                                                               \
            asm volatile("cp.async.commit_group;\n");                         \
        }                                                                      \
        int rbuf = kt % GEMM_BI_TC16_STAGES;                                       \
        unsigned As_rd =                                                       \
            As_sbase + (unsigned)(rbuf * GEMM_BI_TC16_BM * GEMM_BI_TC16_LDA * 2);      \
        unsigned Bs_rd =                                                       \
            Bs_sbase + (unsigned)(rbuf * GEMM_BI_TC16_BK * GEMM_BI_TC16_LDB * 2);      \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (GEMM_BI_TC16_BK / 16); ks++) {                      \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[4];                                                \
            unsigned b_frag[2];                                                \
            {                                                                  \
                int row = lm_row_off + lm_r;                                   \
                unsigned addr = As_rd +                                        \
                    (unsigned)((row * GEMM_BI_TC16_LDA + k0 + lm_col_off) * 2);    \
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
                    (unsigned)((row * GEMM_BI_TC16_LDB + warpN) * 2);              \
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
        int r0 = pid_m * GEMM_BI_TC16_BM + g;                                      \
        int c0 = pid_n * GEMM_BI_TC16_BN + warpN + 2 * t;                          \
        _Pragma("unroll")                                                      \
        for (int e = 0; e < 4; e++) {                                          \
            int gr = r0 + (e >= 2 ? 8 : 0);                                    \
            int gc = c0 + (e & 1);                                             \
            if (gr >= M || gc >= N) continue;                                  \
            float val = alpha * acc[e];                                        \
            if (beta != 0.0f)                                                  \
                val += beta * to_f(C[(long long)gr * ldc + gc]);               \
            C[(long long)gr * ldc + gc] = FROM_F(val);                         \
        }                                                                      \
    }                                                                          \
}

GEMM_BI_DEFINE_GEMM_BI_NN_TC16(bf16, __nv_bfloat16, from_f_bf16, "bf16")
GEMM_BI_DEFINE_GEMM_BI_NN_TC16(f16,  __half,        from_f_f16,  "f16")

// TN (dW) staging: Xs[m_local][k_out chunk], dYs[m_local][n chunk]; both
// rows are the M-reduction dim (GEMM_BI_TC64_BK rows per tile, 64-wide rows).
#define GEMM_BI_TC64_STAGE_TN_ASYNC(buf, mIdx)                                    \
    do {                                                                      \
        unsigned _xs =                                                        \
            Xs_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2);    \
        unsigned _ys =                                                        \
            Ys_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2);    \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BK * (GEMM_BI_TC64_BM / 8);      \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _r = _i / (GEMM_BI_TC64_BM / 8);                                  \
            int _c = (_i % (GEMM_BI_TC64_BM / 8)) * 8;                            \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * GEMM_BI_TC64_BM + _c;                               \
            int _elems = gemm_bi_cp_async_valid_elems(_gm < M_red, K_out, _gk);   \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _xs + (unsigned)((_r * GEMM_BI_TC64_LDB + _c) * 2);   \
            long long _offset =                                                \
                _bytes == 0 ? 0 : (long long)_gm * K_out + _gk;               \
            const void* _src = gemm_bi_cp_async_source(A, _offset, _bytes);        \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BK * (GEMM_BI_TC64_BN / 8);      \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _r = _i / (GEMM_BI_TC64_BN / 8);                                  \
            int _c = (_i % (GEMM_BI_TC64_BN / 8)) * 8;                            \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_TC64_BN + _c;                               \
            int _elems = gemm_bi_cp_async_valid_elems(_gm < M_red, N, _gn);       \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ys + (unsigned)((_r * GEMM_BI_TC64_LDB + _c) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gm * N + _gn;   \
            const void* _src = gemm_bi_cp_async_source(B, _offset, _bytes);        \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define GEMM_BI_TC64_STAGE_TN_SCALAR(buf, mIdx, TT, FF)                           \
    do {                                                                      \
        TT* _xs = &Xs[buf][0][0];                                             \
        TT* _ys = &Ys[buf][0][0];                                             \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BK * GEMM_BI_TC64_BM;            \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _r = _i / GEMM_BI_TC64_BM;                                        \
            int _c = _i % GEMM_BI_TC64_BM;                                        \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * GEMM_BI_TC64_BM + _c;                               \
            _xs[_r * GEMM_BI_TC64_LDB + _c] = (_gm < M_red && _gk < K_out)        \
                                              ? A[(long long)_gm * K_out + _gk]\
                                              : FF(0.0f);                     \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BK * GEMM_BI_TC64_BN;            \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _r = _i / GEMM_BI_TC64_BN;                                        \
            int _c = _i % GEMM_BI_TC64_BN;                                        \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_TC64_BN + _c;                               \
            _ys[_r * GEMM_BI_TC64_LDB + _c] = (_gm < M_red && _gn < N)            \
                                              ? B[(long long)_gm * N + _gn]   \
                                              : FF(0.0f);                     \
        }                                                                     \
    } while (0)

#define GEMM_BI_DEFINE_GEMM_BI_TN_TC64(SUFFIX, T_ACT, FROM_F, MMA_T)                  \
extern "C" __global__ __launch_bounds__(GEMM_BI_TC64_THREADS, 1)                   \
void gemm_bi_tn_tc64_##SUFFIX(                                                \
    float* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M_red, int K_out, int N                                                \
) {                                                                            \
    __shared__ __align__(16) T_ACT Xs[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB];           \
    __shared__ __align__(16) T_ACT Ys[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB];           \
    int num_pid_n = (N + GEMM_BI_TC64_BN - 1) / GEMM_BI_TC64_BN;                       \
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
    /* A x4.trans quadrants: stored-row off (m) = (q&2)?8:0, col off (ko) =  */\
    /* (q&1)?8:0. B x2.trans: stored-row off (m) = (q&1)?8:0.                */\
    int lm_arow_off = (lm_q & 2) ? 8 : 0;                                      \
    int lm_acol_off = (lm_q & 1) ? 8 : 0;                                      \
    int lm_brow_off = (lm_q & 1) ? 8 : 0;                                      \
    unsigned Xs_sbase = (unsigned)__cvta_generic_to_shared(&Xs[0][0][0]);      \
    unsigned Ys_sbase = (unsigned)__cvta_generic_to_shared(&Ys[0][0][0]);      \
    bool fast_stage = gemm_bi_is_aligned_16(A) && gemm_bi_is_aligned_16(B) &&         \
                      ((K_out & 7) == 0) && ((N & 7) == 0);                    \
    float acc[2][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++)                                             \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++)                                         \
            _Pragma("unroll")                                                  \
            for (int e = 0; e < 4; e++) acc[fm][fn][e] = 0.0f;                 \
    int num_m_tiles = (M_red + GEMM_BI_TC64_BK - 1) / GEMM_BI_TC64_BK;                 \
    if (fast_stage) {                                                          \
        GEMM_BI_TC64_STAGE_TN_ASYNC(0, 0);                                         \
    } else {                                                                   \
        GEMM_BI_TC64_STAGE_TN_SCALAR(0, 0, T_ACT, FROM_F);                         \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int mt = 0; mt < num_m_tiles; mt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (mt + 1 < num_m_tiles) {                                            \
            if (fast_stage) {                                                  \
                GEMM_BI_TC64_STAGE_TN_ASYNC(read_buf ^ 1, (mt + 1) * GEMM_BI_TC64_BK); \
            } else {                                                           \
                GEMM_BI_TC64_STAGE_TN_SCALAR(read_buf ^ 1, (mt + 1) * GEMM_BI_TC64_BK, \
                                         T_ACT, FROM_F);                       \
            }                                                                  \
        }                                                                      \
        unsigned Xs_rd =                                                       \
            Xs_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2);  \
        unsigned Ys_rd =                                                       \
            Ys_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2);  \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (GEMM_BI_TC64_BK / 16); ks++) {                  \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[2][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 2; fm++) {                                   \
                int srow = k0 + lm_arow_off + lm_r;                            \
                int scol = warpM + fm * 16 + lm_acol_off;                      \
                unsigned addr =                                                \
                    Xs_rd + (unsigned)((srow * GEMM_BI_TC64_LDB + scol) * 2);   \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "          \
                    "{%0,%1,%2,%3}, [%4];\n"                                   \
                    : "=r"(a_frag[fm][0]), "=r"(a_frag[fm][1]),                \
                      "=r"(a_frag[fm][2]), "=r"(a_frag[fm][3])                 \
                    : "r"(addr));                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int fn = 0; fn < 4; fn++) {                                   \
                int srow = k0 + lm_brow_off + lm_r;                            \
                unsigned addr = Ys_rd +                                        \
                    (unsigned)((srow * GEMM_BI_TC64_LDB + warpN + fn * 8) * 2);\
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
    /* epilogue: f32 accumulate into dW */                                     \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            int r0 = pid_m * GEMM_BI_TC64_BM + warpM + fm * 16 + g;                \
            int c0 = pid_n * GEMM_BI_TC64_BN + warpN + fn * 8 + 2 * t;             \
            _Pragma("unroll")                                                  \
            for (int e = 0; e < 4; e++) {                                      \
                int gr = r0 + (e >= 2 ? 8 : 0);                                \
                int gc = c0 + (e & 1);                                         \
                if (gr >= K_out || gc >= N) continue;                          \
                C[(long long)gr * N + gc] += alpha * acc[fm][fn][e];           \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

GEMM_BI_DEFINE_GEMM_BI_TN_TC64(bf16, __nv_bfloat16, from_f_bf16, "bf16")
GEMM_BI_DEFINE_GEMM_BI_TN_TC64(f16,  __half,        from_f_f16,  "f16")

#define GEMM_BI_TN_RECT_BM 128
#define GEMM_BI_TN_RECT_BN 64
#define GEMM_BI_TN_RECT_BK 32
#define GEMM_BI_TN_RECT_STAGES 3
#define GEMM_BI_TN_RECT_THREADS 256
#define GEMM_BI_TN_RECT_LDX (GEMM_BI_TN_RECT_BM + 8)
#define GEMM_BI_TN_RECT_LDY (GEMM_BI_TN_RECT_BN + 8)

#define GEMM_BI_TN_RECT_STAGE_ASYNC(buf, mIdx)                                \
    do {                                                                      \
        unsigned _xs = Xs_sbase +                                             \
            (unsigned)((buf) * GEMM_BI_TN_RECT_BK * GEMM_BI_TN_RECT_LDX * 2); \
        unsigned _ys = Ys_sbase +                                             \
            (unsigned)((buf) * GEMM_BI_TN_RECT_BK * GEMM_BI_TN_RECT_LDY * 2); \
        for (int _i = threadIdx.x;                                            \
             _i < GEMM_BI_TN_RECT_BK * (GEMM_BI_TN_RECT_BM / 8);             \
             _i += GEMM_BI_TN_RECT_THREADS) {                                 \
            int _r = _i / (GEMM_BI_TN_RECT_BM / 8);                          \
            int _c = (_i % (GEMM_BI_TN_RECT_BM / 8)) * 8;                    \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * GEMM_BI_TN_RECT_BM + _c;                        \
            int _elems =                                                      \
                gemm_bi_cp_async_valid_elems(_gm < M_red, K_out, _gk);        \
            int _bytes = _elems * 2;                                          \
            unsigned _dst =                                                   \
                _xs + (unsigned)((_r * GEMM_BI_TN_RECT_LDX + _c) * 2);        \
            long long _offset =                                               \
                _bytes == 0 ? 0 : (long long)_gm * K_out + _gk;               \
            const void* _src = gemm_bi_cp_async_source(A, _offset, _bytes);   \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                    \
        }                                                                     \
        for (int _i = threadIdx.x;                                            \
             _i < GEMM_BI_TN_RECT_BK * (GEMM_BI_TN_RECT_BN / 8);             \
             _i += GEMM_BI_TN_RECT_THREADS) {                                 \
            int _r = _i / (GEMM_BI_TN_RECT_BN / 8);                          \
            int _c = (_i % (GEMM_BI_TN_RECT_BN / 8)) * 8;                    \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_TN_RECT_BN + _c;                        \
            int _elems = gemm_bi_cp_async_valid_elems(_gm < M_red, N, _gn);  \
            int _bytes = _elems * 2;                                          \
            unsigned _dst =                                                   \
                _ys + (unsigned)((_r * GEMM_BI_TN_RECT_LDY + _c) * 2);        \
            long long _offset =                                               \
                _bytes == 0 ? 0 : (long long)_gm * N + _gn;                   \
            const void* _src = gemm_bi_cp_async_source(B, _offset, _bytes);   \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                    \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                             \
    } while (0)

#define GEMM_BI_TN_RECT_STAGE_SCALAR(buf, mIdx, TT, FF)                       \
    do {                                                                      \
        TT* _xs = &Xs[buf][0][0];                                             \
        TT* _ys = &Ys[buf][0][0];                                             \
        for (int _i = threadIdx.x;                                            \
             _i < GEMM_BI_TN_RECT_BK * GEMM_BI_TN_RECT_BM;                   \
             _i += GEMM_BI_TN_RECT_THREADS) {                                 \
            int _r = _i / GEMM_BI_TN_RECT_BM;                                \
            int _c = _i % GEMM_BI_TN_RECT_BM;                                \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * GEMM_BI_TN_RECT_BM + _c;                        \
            _xs[_r * GEMM_BI_TN_RECT_LDX + _c] =                              \
                (_gm < M_red && _gk < K_out)                                  \
                    ? A[(long long)_gm * K_out + _gk]                         \
                    : FF(0.0f);                                               \
        }                                                                     \
        for (int _i = threadIdx.x;                                            \
             _i < GEMM_BI_TN_RECT_BK * GEMM_BI_TN_RECT_BN;                   \
             _i += GEMM_BI_TN_RECT_THREADS) {                                 \
            int _r = _i / GEMM_BI_TN_RECT_BN;                                \
            int _c = _i % GEMM_BI_TN_RECT_BN;                                \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_TN_RECT_BN + _c;                        \
            _ys[_r * GEMM_BI_TN_RECT_LDY + _c] =                              \
                (_gm < M_red && _gn < N)                                      \
                    ? B[(long long)_gm * N + _gn]                             \
                    : FF(0.0f);                                               \
        }                                                                     \
    } while (0)

#define GEMM_BI_TN_RECT_COMMIT_EMPTY()                                        \
    asm volatile("cp.async.commit_group;\n")

#define GEMM_BI_DEFINE_GEMM_BI_TN_TC128X64(SUFFIX, T_ACT, FROM_F, MMA_T)      \
extern "C" __global__ __launch_bounds__(256, 2)                              \
void gemm_bi_tn_tc128x64_##SUFFIX(                                            \
    float* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M_red, int K_out, int N                                                \
) {                                                                            \
    if (M_red <= 0 || K_out <= 0 || N <= 0) return;                            \
    __shared__ __align__(16) T_ACT                                             \
        Xs[GEMM_BI_TN_RECT_STAGES][GEMM_BI_TN_RECT_BK][GEMM_BI_TN_RECT_LDX]; \
    __shared__ __align__(16) T_ACT                                             \
        Ys[GEMM_BI_TN_RECT_STAGES][GEMM_BI_TN_RECT_BK][GEMM_BI_TN_RECT_LDY]; \
    int num_pid_n = (N + GEMM_BI_TN_RECT_BN - 1) / GEMM_BI_TN_RECT_BN;       \
    int pid_m = blockIdx.x / num_pid_n;                                        \
    int pid_n = blockIdx.x % num_pid_n;                                        \
    int warp = threadIdx.x >> 5;                                               \
    int lane = threadIdx.x & 31;                                               \
    int warpM = (warp >> 1) * 32;                                              \
    int warpN = (warp & 1) * 32;                                               \
    int g = lane >> 2;                                                         \
    int t = lane & 3;                                                          \
    int lm_r = lane & 7;                                                       \
    int lm_q = lane >> 3;                                                      \
    int lm_arow_off = (lm_q & 2) ? 8 : 0;                                     \
    int lm_acol_off = (lm_q & 1) ? 8 : 0;                                     \
    int lm_brow_off = (lm_q & 1) ? 8 : 0;                                     \
    unsigned Xs_sbase = (unsigned)__cvta_generic_to_shared(&Xs[0][0][0]);      \
    unsigned Ys_sbase = (unsigned)__cvta_generic_to_shared(&Ys[0][0][0]);      \
    bool fast_stage = gemm_bi_is_aligned_16(A) &&                              \
                      gemm_bi_is_aligned_16(B) &&                              \
                      ((K_out & 7) == 0) && ((N & 7) == 0);                    \
    float acc[2][4][4];                                                        \
    _Pragma("unroll")                                                         \
    for (int fm = 0; fm < 2; fm++)                                             \
        _Pragma("unroll")                                                     \
        for (int fn = 0; fn < 4; fn++)                                         \
            _Pragma("unroll")                                                 \
            for (int e = 0; e < 4; e++) acc[fm][fn][e] = 0.0f;                \
    int macro64_tiles = (M_red - 1) / 64 + 1;                                 \
    int k32_tiles = 2 * macro64_tiles;                                         \
    _Pragma("unroll")                                                         \
    for (int p = 0; p < 2; p++) {                                              \
        if (p < k32_tiles) {                                                   \
            if (fast_stage) {                                                  \
                GEMM_BI_TN_RECT_STAGE_ASYNC(p % GEMM_BI_TN_RECT_STAGES,       \
                                            p * GEMM_BI_TN_RECT_BK);           \
            } else {                                                           \
                GEMM_BI_TN_RECT_STAGE_SCALAR(p % GEMM_BI_TN_RECT_STAGES,      \
                                             p * GEMM_BI_TN_RECT_BK,           \
                                             T_ACT, FROM_F);                    \
                GEMM_BI_TN_RECT_COMMIT_EMPTY();                                \
            }                                                                  \
        } else {                                                               \
            GEMM_BI_TN_RECT_COMMIT_EMPTY();                                    \
        }                                                                      \
    }                                                                          \
    for (int kt = 0; kt < k32_tiles; kt++) {                                  \
        asm volatile("cp.async.wait_group 1;\n");                             \
        __syncthreads();                                                       \
        int next = kt + 2;                                                     \
        if (next < k32_tiles) {                                                \
            if (fast_stage) {                                                  \
                GEMM_BI_TN_RECT_STAGE_ASYNC(                                   \
                    next % GEMM_BI_TN_RECT_STAGES,                             \
                    next * GEMM_BI_TN_RECT_BK);                                \
            } else {                                                           \
                GEMM_BI_TN_RECT_STAGE_SCALAR(                                  \
                    next % GEMM_BI_TN_RECT_STAGES,                             \
                    next * GEMM_BI_TN_RECT_BK, T_ACT, FROM_F);                 \
                GEMM_BI_TN_RECT_COMMIT_EMPTY();                                \
            }                                                                  \
        } else {                                                               \
            GEMM_BI_TN_RECT_COMMIT_EMPTY();                                    \
        }                                                                      \
        int read_buf = kt % GEMM_BI_TN_RECT_STAGES;                           \
        unsigned Xs_rd = Xs_sbase +                                            \
            (unsigned)(read_buf * GEMM_BI_TN_RECT_BK *                        \
                       GEMM_BI_TN_RECT_LDX * 2);                               \
        unsigned Ys_rd = Ys_sbase +                                            \
            (unsigned)(read_buf * GEMM_BI_TN_RECT_BK *                        \
                       GEMM_BI_TN_RECT_LDY * 2);                               \
        _Pragma("unroll")                                                     \
        for (int ks = 0; ks < (GEMM_BI_TN_RECT_BK / 16); ks++) {              \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[2][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                 \
            for (int fm = 0; fm < 2; fm++) {                                  \
                int srow = k0 + lm_arow_off + lm_r;                            \
                int scol = warpM + fm * 16 + lm_acol_off;                      \
                unsigned addr = Xs_rd +                                        \
                    (unsigned)((srow * GEMM_BI_TN_RECT_LDX + scol) * 2);       \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "         \
                    "{%0,%1,%2,%3}, [%4];\n"                                  \
                    : "=r"(a_frag[fm][0]), "=r"(a_frag[fm][1]),             \
                      "=r"(a_frag[fm][2]), "=r"(a_frag[fm][3])              \
                    : "r"(addr));                                             \
            }                                                                  \
            _Pragma("unroll")                                                 \
            for (int fn = 0; fn < 4; fn++) {                                  \
                int srow = k0 + lm_brow_off + lm_r;                            \
                unsigned addr = Ys_rd +                                        \
                    (unsigned)((srow * GEMM_BI_TN_RECT_LDY +                   \
                                warpN + fn * 8) * 2);                          \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "         \
                    "{%0,%1}, [%2];\n"                                        \
                    : "=r"(b_frag[fn][0]), "=r"(b_frag[fn][1])              \
                    : "r"(addr));                                             \
            }                                                                  \
            _Pragma("unroll")                                                 \
            for (int fm = 0; fm < 2; fm++) {                                  \
                _Pragma("unroll")                                             \
                for (int fn = 0; fn < 4; fn++) {                              \
                    asm volatile(                                              \
                        "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "."  \
                        MMA_T ".f32 "                                         \
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "             \
                        "{%0,%1,%2,%3};\n"                                     \
                        : "+f"(acc[fm][fn][0]), "+f"(acc[fm][fn][1]),       \
                          "+f"(acc[fm][fn][2]), "+f"(acc[fm][fn][3])        \
                        : "r"(a_frag[fm][0]), "r"(a_frag[fm][1]),            \
                          "r"(a_frag[fm][2]), "r"(a_frag[fm][3]),            \
                          "r"(b_frag[fn][0]), "r"(b_frag[fn][1]));           \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    _Pragma("unroll")                                                         \
    for (int fm = 0; fm < 2; fm++) {                                          \
        _Pragma("unroll")                                                     \
        for (int fn = 0; fn < 4; fn++) {                                      \
            int r0 = pid_m * GEMM_BI_TN_RECT_BM + warpM + fm * 16 + g;       \
            int c0 = pid_n * GEMM_BI_TN_RECT_BN + warpN + fn * 8 + 2 * t;    \
            _Pragma("unroll")                                                 \
            for (int e = 0; e < 4; e++) {                                     \
                int gr = r0 + (e >= 2 ? 8 : 0);                               \
                int gc = c0 + (e & 1);                                        \
                if (gr >= K_out || gc >= N) continue;                         \
                C[(long long)gr * N + gc] += alpha * acc[fm][fn][e];          \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

GEMM_BI_DEFINE_GEMM_BI_TN_TC128X64(bf16, __nv_bfloat16, from_f_bf16, "bf16")
GEMM_BI_DEFINE_GEMM_BI_TN_TC128X64(f16, __half, from_f_f16, "f16")

#undef GEMM_BI_DEFINE_GEMM_BI_TN_TC128X64
#undef GEMM_BI_TN_RECT_COMMIT_EMPTY
#undef GEMM_BI_TN_RECT_STAGE_SCALAR
#undef GEMM_BI_TN_RECT_STAGE_ASYNC
#undef GEMM_BI_TN_RECT_LDY
#undef GEMM_BI_TN_RECT_LDX
#undef GEMM_BI_TN_RECT_THREADS
#undef GEMM_BI_TN_RECT_STAGES
#undef GEMM_BI_TN_RECT_BK
#undef GEMM_BI_TN_RECT_BN
#undef GEMM_BI_TN_RECT_BM

// NT (dX) staging: dYs[m_local][n chunk] (output rows x reduction) and
// Ws[k_out_local][n chunk] (output cols x reduction); rows are BK (32) wide.
#define GEMM_BI_TC64_STAGE_NT_ASYNC(buf, nIdx)                                    \
    do {                                                                      \
        unsigned _ys =                                                        \
            Ys_sbase + (unsigned)((buf) * GEMM_BI_TC64_BM * GEMM_BI_TC64_LDA * 2);    \
        unsigned _ws =                                                        \
            Ws_sbase + (unsigned)((buf) * GEMM_BI_TC64_BN * GEMM_BI_TC64_LDA * 2);    \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BM * (GEMM_BI_TC64_BK / 8);      \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _m = _i / (GEMM_BI_TC64_BK / 8);                                  \
            int _c = (_i % (GEMM_BI_TC64_BK / 8)) * 8;                            \
            int _gm = pid_m * GEMM_BI_TC64_BM + _m;                               \
            int _gn = (nIdx) + _c;                                            \
            int _elems = gemm_bi_cp_async_valid_elems(_gm < M, N, _gn);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ys + (unsigned)((_m * GEMM_BI_TC64_LDA + _c) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gm * N + _gn;   \
            const void* _src = gemm_bi_cp_async_source(A, _offset, _bytes);        \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BN * (GEMM_BI_TC64_BK / 8);      \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _k = _i / (GEMM_BI_TC64_BK / 8);                                  \
            int _c = (_i % (GEMM_BI_TC64_BK / 8)) * 8;                            \
            int _gk = pid_n * GEMM_BI_TC64_BN + _k;                               \
            int _gn = (nIdx) + _c;                                            \
            int _elems = gemm_bi_cp_async_valid_elems(_gk < K_out, N, _gn);       \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ws + (unsigned)((_k * GEMM_BI_TC64_LDA + _c) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gk * N + _gn;   \
            const void* _src = gemm_bi_cp_async_source(B, _offset, _bytes);        \
            gemm_bi_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define GEMM_BI_TC64_STAGE_NT_SCALAR(buf, nIdx, TT, FF)                           \
    do {                                                                      \
        TT* _ys = &Ys[buf][0][0];                                             \
        TT* _ws = &Ws[buf][0][0];                                             \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BM * GEMM_BI_TC64_BK;            \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _m = _i / GEMM_BI_TC64_BK;                                        \
            int _c = _i % GEMM_BI_TC64_BK;                                        \
            int _gm = pid_m * GEMM_BI_TC64_BM + _m;                               \
            int _gn = (nIdx) + _c;                                            \
            _ys[_m * GEMM_BI_TC64_LDA + _c] = (_gm < M && _gn < N)                \
                                              ? A[(long long)_gm * N + _gn]   \
                                              : FF(0.0f);                     \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BN * GEMM_BI_TC64_BK;            \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _k = _i / GEMM_BI_TC64_BK;                                        \
            int _c = _i % GEMM_BI_TC64_BK;                                        \
            int _gk = pid_n * GEMM_BI_TC64_BN + _k;                               \
            int _gn = (nIdx) + _c;                                            \
            _ws[_k * GEMM_BI_TC64_LDA + _c] = (_gk < K_out && _gn < N)            \
                                              ? B[(long long)_gk * N + _gn]   \
                                              : FF(0.0f);                     \
        }                                                                     \
    } while (0)

#define GEMM_BI_DEFINE_GEMM_BI_NT_TC64(SUFFIX, T_ACT, FROM_F, MMA_T)                  \
extern "C" __global__ __launch_bounds__(GEMM_BI_TC64_THREADS, 1)                   \
void gemm_bi_nt_tc64_##SUFFIX(                                                \
    T_ACT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M, int N, int K_out                                                    \
) {                                                                            \
    __shared__ __align__(16) T_ACT Ys[2][GEMM_BI_TC64_BM][GEMM_BI_TC64_LDA];           \
    __shared__ __align__(16) T_ACT Ws[2][GEMM_BI_TC64_BN][GEMM_BI_TC64_LDA];           \
    int num_pid_n = (K_out + GEMM_BI_TC64_BN - 1) / GEMM_BI_TC64_BN;                   \
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
    int lmb_col_off = (lm_q & 1) ? 8 : 0;                                      \
    unsigned Ys_sbase = (unsigned)__cvta_generic_to_shared(&Ys[0][0][0]);      \
    unsigned Ws_sbase = (unsigned)__cvta_generic_to_shared(&Ws[0][0][0]);      \
    bool fast_stage = gemm_bi_is_aligned_16(A) && gemm_bi_is_aligned_16(B) &&         \
                      ((N & 7) == 0);                                          \
    float acc[2][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++)                                             \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++)                                         \
            _Pragma("unroll")                                                  \
            for (int e = 0; e < 4; e++) acc[fm][fn][e] = 0.0f;                 \
    int num_n_tiles = (N + GEMM_BI_TC64_BK - 1) / GEMM_BI_TC64_BK;                     \
    if (fast_stage) {                                                          \
        GEMM_BI_TC64_STAGE_NT_ASYNC(0, 0);                                         \
    } else {                                                                   \
        GEMM_BI_TC64_STAGE_NT_SCALAR(0, 0, T_ACT, FROM_F);                         \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int nt = 0; nt < num_n_tiles; nt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (nt + 1 < num_n_tiles) {                                            \
            if (fast_stage) {                                                  \
                GEMM_BI_TC64_STAGE_NT_ASYNC(read_buf ^ 1, (nt + 1) * GEMM_BI_TC64_BK); \
            } else {                                                           \
                GEMM_BI_TC64_STAGE_NT_SCALAR(read_buf ^ 1, (nt + 1) * GEMM_BI_TC64_BK, \
                                         T_ACT, FROM_F);                       \
            }                                                                  \
        }                                                                      \
        unsigned Ys_rd =                                                       \
            Ys_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BM * GEMM_BI_TC64_LDA * 2);  \
        unsigned Ws_rd =                                                       \
            Ws_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BN * GEMM_BI_TC64_LDA * 2);  \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (GEMM_BI_TC64_BK / 16); ks++) {                  \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[2][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 2; fm++) {                                   \
                int row = warpM + fm * 16 + lm_row_off + lm_r;                 \
                unsigned addr = Ys_rd +                                        \
                    (unsigned)((row * GEMM_BI_TC64_LDA + k0 + lm_col_off) * 2);\
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "                \
                    "{%0,%1,%2,%3}, [%4];\n"                                   \
                    : "=r"(a_frag[fm][0]), "=r"(a_frag[fm][1]),                \
                      "=r"(a_frag[fm][2]), "=r"(a_frag[fm][3])                 \
                    : "r"(addr));                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int fn = 0; fn < 4; fn++) {                                   \
                int row = warpN + fn * 8 + lm_r;                               \
                unsigned addr = Ws_rd +                                        \
                    (unsigned)((row * GEMM_BI_TC64_LDA + k0 + lmb_col_off) * 2);\
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "                \
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
    /* epilogue: typed RNE overwrite of dX */                                  \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            int r0 = pid_m * GEMM_BI_TC64_BM + warpM + fm * 16 + g;                \
            int c0 = pid_n * GEMM_BI_TC64_BN + warpN + fn * 8 + 2 * t;             \
            _Pragma("unroll")                                                  \
            for (int e = 0; e < 4; e++) {                                      \
                int gr = r0 + (e >= 2 ? 8 : 0);                                \
                int gc = c0 + (e & 1);                                         \
                if (gr >= M || gc >= K_out) continue;                          \
                C[(long long)gr * K_out + gc] =                                \
                    FROM_F(alpha * acc[fm][fn][e]);                            \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

GEMM_BI_DEFINE_GEMM_BI_NT_TC64(bf16, __nv_bfloat16, from_f_bf16, "bf16")
GEMM_BI_DEFINE_GEMM_BI_NT_TC64(f16,  __half,        from_f_f16,  "f16")

#undef GEMM_BI_DEFINE_GEMM_BI_NN_TC128
#undef GEMM_BI_DEFINE_GEMM_BI_NN_TC16
#undef GEMM_BI_DEFINE_GEMM_BI_NN_TC64
#undef GEMM_BI_DEFINE_GEMM_BI_NT_TC128
#undef GEMM_BI_DEFINE_GEMM_BI_NT_TC64
#undef GEMM_BI_DEFINE_GEMM_BI_TN_TC128
#undef GEMM_BI_DEFINE_GEMM_BI_TN_TC64
#undef GEMM_BI_TC128_BK
#undef GEMM_BI_TC128_BM
#undef GEMM_BI_TC128_BN
#undef GEMM_BI_TC128_LDA
#undef GEMM_BI_TC128_LDB
#undef GEMM_BI_TC128_PAD_A
#undef GEMM_BI_TC128_PAD_B
#undef GEMM_BI_TC128_STAGE_ASYNC
#undef GEMM_BI_TC128_STAGE_NT_ASYNC
#undef GEMM_BI_TC128_STAGE_NT_SCALAR
#undef GEMM_BI_TC128_STAGE_SCALAR
#undef GEMM_BI_TC128_STAGE_TN_ASYNC
#undef GEMM_BI_TC128_STAGE_TN_SCALAR
#undef GEMM_BI_TC16_BK
#undef GEMM_BI_TC16_BM
#undef GEMM_BI_TC16_BN
#undef GEMM_BI_TC16_LDA
#undef GEMM_BI_TC16_LDB
#undef GEMM_BI_TC16_STAGES
#undef GEMM_BI_TC16_STAGE_ASYNC
#undef GEMM_BI_TC16_STAGE_SCALAR
#undef GEMM_BI_TC16_THREADS
#undef GEMM_BI_TC64_BK
#undef GEMM_BI_TC64_BM
#undef GEMM_BI_TC64_BN
#undef GEMM_BI_TC64_LDA
#undef GEMM_BI_TC64_LDB
#undef GEMM_BI_TC64_STAGE_ASYNC
#undef GEMM_BI_TC64_STAGE_NT_ASYNC
#undef GEMM_BI_TC64_STAGE_NT_SCALAR
#undef GEMM_BI_TC64_STAGE_SCALAR
#undef GEMM_BI_TC64_STAGE_TN_ASYNC
#undef GEMM_BI_TC64_STAGE_TN_SCALAR
#undef GEMM_BI_TC64_THREADS

// Deterministic TF32 uses a private F32 pipeline. The 16-bit paths above have
// different fragment, staging, and rounding contracts.
struct Sm80Tf32KernelParams {
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(Sm80Tf32KernelParams) == 32, "TF32 parameter ABI drift");
static_assert(alignof(Sm80Tf32KernelParams) == 4, "TF32 parameter alignment drift");
static_assert(__is_standard_layout(Sm80Tf32KernelParams),
              "TF32 parameters must remain standard layout");
// Eight ordered 4-byte fields in 32 bytes leave no internal or tail padding.
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->alpha) == 4,
              "TF32 alpha size changed");
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->beta) == 4,
              "TF32 beta size changed");
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->m) == 4,
              "TF32 M size changed");
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->k) == 4,
              "TF32 K size changed");
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->n) == 4,
              "TF32 N size changed");
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->lda) == 4,
              "TF32 A stride size changed");
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->ldb) == 4,
              "TF32 B stride size changed");
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->ldc) == 4,
              "TF32 output stride size changed");

enum SgbTf32Op { SgbTf32Nn, SgbTf32Tn, SgbTf32Nt };

template <SgbTf32Op Op, int BM, int BN, int Stages>
struct __align__(16) SgbTf32Storage {
    static constexpr int bk32 = 32;
    static constexpr int ARows = Op == SgbTf32Tn ? bk32 : BM;
    static constexpr int AStride = Op == SgbTf32Tn ? BM + 8 : 36;
    static constexpr int BRows = Op == SgbTf32Nt ? BN : bk32;
    static constexpr int BStride = Op == SgbTf32Nt ? 36
        : (BN == 64 ? 72 : (BN == 32 ? 40 : 24));
    float a[Stages][ARows][AStride];
    float b[Stages][BRows][BStride];
};

static_assert(sizeof(SgbTf32Storage<SgbTf32Nn, 128, 64, 2>) == 55296, "NN M128N64 s2 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nn, 128, 64, 3>) == 82944, "NN M128N64 s3 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nn, 64, 64, 2>) == 36864, "NN M64N64 s2 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nn, 64, 64, 3>) == 55296, "NN M64N64 s3 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nn, 16, 32, 4>) == 29696, "NN M16N32 s4 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nn, 16, 32, 3>) == 22272, "NN M16N32 s3 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nn, 16, 16, 4>) == 21504, "NN M16N16 s4 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Tn, 128, 64, 2>) == 53248, "TN M128N64 s2 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Tn, 128, 64, 3>) == 79872, "TN M128N64 s3 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Tn, 64, 64, 2>) == 36864, "TN M64N64 s2 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Tn, 64, 64, 3>) == 55296, "TN M64N64 s3 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Tn, 16, 32, 4>) == 32768, "TN M16N32 s4 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Tn, 16, 16, 4>) == 24576, "TN M16N16 s4 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nt, 128, 64, 2>) == 55296, "NT M128N64 s2 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nt, 128, 64, 3>) == 82944, "NT M128N64 s3 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nt, 64, 64, 2>) == 36864, "NT M64N64 s2 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nt, 64, 64, 3>) == 55296, "NT M64N64 s3 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nt, 16, 32, 4>) == 27648, "NT M16N32 s4 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nt, 16, 32, 3>) == 20736, "NT M16N32 s3 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nt, 32, 32, 3>) == 27648, "NT M32N32 s3 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nt, 32, 32, 4>) == 36864, "NT M32N32 s4 storage");
static_assert(sizeof(SgbTf32Storage<SgbTf32Nt, 16, 16, 4>) == 18432, "NT M16N16 s4 storage");

template <SgbTf32Op Op, int BM, int BN, int Stages>
__device__ __forceinline__ float& gemm_bi_tf32_a_slot(
    SgbTf32Storage<Op, BM, BN, Stages>* storage,
    int stage, int row, int reduction) {
    if constexpr (Op == SgbTf32Tn) {
        return storage->a[stage][reduction][row];
    }
    return storage->a[stage][row][reduction];
}

template <SgbTf32Op Op, int BM, int BN, int Stages>
__device__ __forceinline__ float& gemm_bi_tf32_b_slot(
    SgbTf32Storage<Op, BM, BN, Stages>* storage,
    int stage, int reduction, int column) {
    if constexpr (Op == SgbTf32Nt) {
        return storage->b[stage][column][reduction];
    }
    return storage->b[stage][reduction][column];
}

struct SgbTf32Problem {
    float* output;
    const float* a;
    const float* b;
    const float* bias;
    Sm80Tf32KernelParams params;
    int tile_row;
    int tile_column;
};

__device__ __forceinline__ unsigned gemm_bi_tf32_rna(float value) {
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ void gemm_bi_tf32_mma_m16n8k8(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

template <SgbTf32Op Op>
__device__ __forceinline__ int gemm_bi_tf32_rows(const Sm80Tf32KernelParams& p) {
    return Op == SgbTf32Tn ? p.k : p.m;
}

template <SgbTf32Op Op>
__device__ __forceinline__ int gemm_bi_tf32_columns(const Sm80Tf32KernelParams& p) {
    return Op == SgbTf32Nt ? p.k : p.n;
}

template <SgbTf32Op Op>
__device__ __forceinline__ int gemm_bi_tf32_reduction(const Sm80Tf32KernelParams& p) {
    return Op == SgbTf32Nn ? p.k : (Op == SgbTf32Tn ? p.m : p.n);
}

template <SgbTf32Op Op>
__device__ __forceinline__ float gemm_bi_tf32_read_a(
    const SgbTf32Problem& problem, int row, int reduction) {
    if constexpr (Op == SgbTf32Tn) {
        return problem.a[(long long)reduction * problem.params.lda + row];
    }
    return problem.a[(long long)row * problem.params.lda + reduction];
}

template <SgbTf32Op Op>
__device__ __forceinline__ float gemm_bi_tf32_read_b(
    const SgbTf32Problem& problem, int reduction, int column) {
    if constexpr (Op == SgbTf32Nt) {
        return problem.b[(long long)column * problem.params.ldb + reduction];
    }
    return problem.b[(long long)reduction * problem.params.ldb + column];
}

template <SgbTf32Op Op, int BM, int BN, int Stages>
__device__ __forceinline__ void gemm_bi_tf32_stage_scalar(
    SgbTf32Storage<Op, BM, BN, Stages>* storage, int stage,
    const SgbTf32Problem& problem, int reduction_base) {
    constexpr int Threads = BM == 128 ? 256 : (BM == 16 && BN == 16 ? 64 : 128);
    int rows = gemm_bi_tf32_rows<Op>(problem.params);
    int columns = gemm_bi_tf32_columns<Op>(problem.params);
    for (int linear = (int)threadIdx.x; linear < BM * 32; linear += Threads) {
        int row = linear >> 5;
        int reduction = linear & 31;
        int global_row = problem.tile_row + row;
        int global_reduction = reduction_base + reduction;
        float value = 0.0f;
        if (global_row < rows && global_reduction < gemm_bi_tf32_reduction<Op>(problem.params)) {
            value = gemm_bi_tf32_read_a<Op>(problem, global_row, global_reduction);
        }
        gemm_bi_tf32_a_slot<Op>(storage, stage, row, reduction) = value;
    }
    for (int linear = (int)threadIdx.x; linear < 32 * BN; linear += Threads) {
        int reduction = linear / BN;
        int column = linear - reduction * BN;
        int global_reduction = reduction_base + reduction;
        int global_column = problem.tile_column + column;
        float value = 0.0f;
        if (global_column < columns && global_reduction < gemm_bi_tf32_reduction<Op>(problem.params)) {
            value = gemm_bi_tf32_read_b<Op>(problem, global_reduction, global_column);
        }
        gemm_bi_tf32_b_slot<Op>(storage, stage, reduction, column) = value;
    }
}

template <int BM>
__device__ __forceinline__ void gemm_bi_tf32_cp_async_16_zfill(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    if constexpr (BM == 16) {
        gemm_bi_cp_async_16_zfill(shared_dst, global_src, valid_bytes);
    } else {
        gemm_bi_cp_async_16_zfill_l2(shared_dst, global_src, valid_bytes);
    }
}

__device__ __forceinline__ void gemm_bi_tf32_cp_async_4x4_zfill(
    unsigned shared_dst, const float* global_src, int valid_bytes) {
#pragma unroll
    for (int element = 0; element < 4; ++element) {
        int bytes = valid_bytes >= (element + 1) * 4 ? 4 : 0;
        const float* source = bytes == 0 ? global_src : global_src + element;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                     :: "r"(shared_dst + element * 4), "l"(source), "r"(bytes));
    }
}

template <bool Narrow, int BM>
__device__ __forceinline__ void gemm_bi_tf32_cp_async_zfill(
    unsigned shared_dst, const float* global_src, int valid_bytes) {
    if constexpr (!Narrow) {
        if (valid_bytes == 16) {
            gemm_bi_tf32_cp_async_16_zfill<BM>(shared_dst, global_src, valid_bytes);
        } else {
            gemm_bi_tf32_cp_async_4x4_zfill(shared_dst, global_src, valid_bytes);
        }
    } else if (valid_bytes == 16 && gemm_bi_is_aligned_16(global_src)) {
        gemm_bi_tf32_cp_async_16_zfill<BM>(shared_dst, global_src, valid_bytes);
    } else {
        gemm_bi_tf32_cp_async_4x4_zfill(shared_dst, global_src, valid_bytes);
    }
}

template <SgbTf32Op Op, int BM, int BN, int Stages,
          bool NarrowA, bool NarrowB>
__device__ __forceinline__ void gemm_bi_tf32_stage_async(
    SgbTf32Storage<Op, BM, BN, Stages>* storage, int stage,
    const SgbTf32Problem& problem, int reduction_base) {
    constexpr int Threads = BM == 128 ? 256 : (BM == 16 && BN == 16 ? 64 : 128);
    int rows = gemm_bi_tf32_rows<Op>(problem.params);
    int columns = gemm_bi_tf32_columns<Op>(problem.params);
    int reduction_extent = gemm_bi_tf32_reduction<Op>(problem.params);

    if constexpr (Op == SgbTf32Tn) {
        constexpr int RowChunks = BM / 4;
        for (int linear = (int)threadIdx.x;
             linear < 32 * RowChunks;
             linear += Threads) {
            int reduction = linear / RowChunks;
            int row = (linear - reduction * RowChunks) * 4;
            int global_row = problem.tile_row + row;
            int global_reduction = reduction_base + reduction;
            int valid = global_reduction < reduction_extent ? rows - global_row : 0;
            valid = valid < 0 ? 0 : (valid > 4 ? 4 : valid);
            int _bytes = valid * 4;
            long long valid_offset = (long long)global_reduction * problem.params.lda + global_row;
            long long safe_offset = _bytes == 0 ? 0 : valid_offset;
            const float* src = gemm_bi_cp_async_source(problem.a, safe_offset, _bytes);
            unsigned dst = (unsigned)__cvta_generic_to_shared(
                &gemm_bi_tf32_a_slot<Op>(storage, stage, row, reduction));
            gemm_bi_tf32_cp_async_zfill<NarrowA, BM>(dst, src, _bytes);
        }
    } else {
        for (int linear = (int)threadIdx.x; linear < BM * 8; linear += Threads) {
            int row = linear >> 3;
            int reduction = (linear & 7) * 4;
            int global_row = problem.tile_row + row;
            int global_reduction = reduction_base + reduction;
            int valid = global_row < rows ? reduction_extent - global_reduction : 0;
            valid = valid < 0 ? 0 : (valid > 4 ? 4 : valid);
            int _bytes = valid * 4;
            long long valid_offset = (long long)global_row * problem.params.lda + global_reduction;
            long long safe_offset = _bytes == 0 ? 0 : valid_offset;
            const float* src = gemm_bi_cp_async_source(problem.a, safe_offset, _bytes);
            unsigned dst = (unsigned)__cvta_generic_to_shared(
                &gemm_bi_tf32_a_slot<Op>(storage, stage, row, reduction));
            gemm_bi_tf32_cp_async_zfill<NarrowA, BM>(dst, src, _bytes);
        }
    }

    if constexpr (Op == SgbTf32Nt) {
        for (int linear = (int)threadIdx.x; linear < BN * 8; linear += Threads) {
            int column = linear >> 3;
            int reduction = (linear & 7) * 4;
            int global_column = problem.tile_column + column;
            int global_reduction = reduction_base + reduction;
            int valid = global_column < columns ? reduction_extent - global_reduction : 0;
            valid = valid < 0 ? 0 : (valid > 4 ? 4 : valid);
            int _bytes = valid * 4;
            long long valid_offset = (long long)global_column * problem.params.ldb + global_reduction;
            long long safe_offset = _bytes == 0 ? 0 : valid_offset;
            const float* src = gemm_bi_cp_async_source(problem.b, safe_offset, _bytes);
            unsigned dst = (unsigned)__cvta_generic_to_shared(
                &gemm_bi_tf32_b_slot<Op>(storage, stage, reduction, column));
            gemm_bi_tf32_cp_async_zfill<NarrowB, BM>(dst, src, _bytes);
        }
    } else {
        for (int linear = (int)threadIdx.x; linear < 32 * (BN / 4); linear += Threads) {
            int reduction = linear / (BN / 4);
            int column = (linear - reduction * (BN / 4)) * 4;
            int global_reduction = reduction_base + reduction;
            int global_column = problem.tile_column + column;
            int valid = global_reduction < reduction_extent ? columns - global_column : 0;
            valid = valid < 0 ? 0 : (valid > 4 ? 4 : valid);
            int _bytes = valid * 4;
            long long valid_offset = (long long)global_reduction * problem.params.ldb + global_column;
            long long safe_offset = _bytes == 0 ? 0 : valid_offset;
            const float* src = gemm_bi_cp_async_source(problem.b, safe_offset, _bytes);
            unsigned dst = (unsigned)__cvta_generic_to_shared(
                &gemm_bi_tf32_b_slot<Op>(storage, stage, reduction, column));
            gemm_bi_tf32_cp_async_zfill<NarrowB, BM>(dst, src, _bytes);
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

template <SgbTf32Op Op>
__device__ __forceinline__ float gemm_bi_tf32_epilogue(
    float accumulator, float old_output, const float* bias, int column,
    const Sm80Tf32KernelParams& params) {
    if constexpr (Op == SgbTf32Nn) {
        (void)bias;
        (void)column;
        float value = params.alpha == 1.0f
            ? accumulator
            : __fmul_rn(params.alpha, accumulator);
        if (params.beta == 0.0f) return value;
        return __fmaf_rn(params.beta, old_output, value);
    } else if constexpr (Op == SgbTf32Tn) {
        (void)bias;
        (void)column;
        return __fmaf_rn(params.alpha, accumulator, old_output);
    } else {
        (void)old_output;
        (void)bias;
        (void)column;
        return params.alpha == 1.0f
            ? accumulator
            : __fmul_rn(params.alpha, accumulator);
    }
}

template <SgbTf32Op Op>
__device__ __forceinline__ void gemm_bi_tf32_store(
    float* output, int row, int column, float accumulator,
    const float* bias, const Sm80Tf32KernelParams& params) {
    int rows = gemm_bi_tf32_rows<Op>(params);
    int columns = gemm_bi_tf32_columns<Op>(params);
    if (row >= rows || column >= columns) return;
    float* destination = output + (long long)row * params.ldc + column;
    float old_output = 0.0f;
    if constexpr (Op == SgbTf32Tn) {
        old_output = *destination;
    } else if constexpr (Op == SgbTf32Nn) {
        if (params.beta != 0.0f) old_output = *destination;
    }
    float value = gemm_bi_tf32_epilogue<Op>(accumulator, old_output, bias, column, params);
#line 2001 "mamba_tf32_k0_zero_store"
    *destination = value;
#line 1560 "sm80.cu"
}

template <SgbTf32Op Op, int BM, int BN>
__device__ __forceinline__ void gemm_bi_tf32_zero_reduction_epilogue(
    float* output, const float* bias, const Sm80Tf32KernelParams& params) {
    (void)&gemm_bi_tf32_epilogue<Op>;
    int rows = gemm_bi_tf32_rows<Op>(params);
    int columns = gemm_bi_tf32_columns<Op>(params);
    int tile_row = (int)blockIdx.x / ((columns + BN - 1) / BN) * BM;
    int tile_column = (int)blockIdx.x % ((columns + BN - 1) / BN) * BN;
    for (int linear = (int)threadIdx.x; linear < BM * BN; linear += (int)blockDim.x) {
        int row = tile_row + linear / BN;
        int column = tile_column + linear % BN;
        if (row < rows && column < columns) {
            float accumulator = Op == SgbTf32Nn && bias != nullptr ? bias[column] : 0.0f;
            gemm_bi_tf32_store<Op>(output, row, column, accumulator, bias, params);
        }
    }
}

__device__ __forceinline__ bool gemm_bi_tf32_can_stage_a16(
    const float* a, const Sm80Tf32KernelParams& params) {
    return gemm_bi_is_aligned_16(a) && (params.lda & 3) == 0;
}

__device__ __forceinline__ bool gemm_bi_tf32_can_stage_b16(
    const float* b, const Sm80Tf32KernelParams& params) {
    return gemm_bi_is_aligned_16(b) && (params.ldb & 3) == 0;
}

__device__ __forceinline__ bool gemm_bi_tf32_can_stage_async_4(
    const float* a, const float* b, const Sm80Tf32KernelParams& params) {
    (void)params;
    return gemm_bi_is_aligned_4(a) && gemm_bi_is_aligned_4(b);
}

template <SgbTf32Op Op, int BM, int BN, int Stages, int MAtoms, int NAtoms>
__device__ __forceinline__ void gemm_bi_tf32_compute_stage(
    SgbTf32Storage<Op, BM, BN, Stages>* storage, int stage,
    int warp_m, int warp_n, int group, int thread,
    float (&accumulators)[MAtoms][NAtoms][4]) {
    const int k_offsets[4] = {0, 8, 16, 24};
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        int k8 = k_offsets[issue];
        unsigned a_fragments[MAtoms][4];
        unsigned b_fragments[NAtoms][2];
#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
            int row = warp_m + m_atom * 16 + group;
            a_fragments[m_atom][0] =
                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread));
            a_fragments[m_atom][1] =
                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread));
            a_fragments[m_atom][2] =
                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread + 4));
            a_fragments[m_atom][3] =
                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread + 4));
        }
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
            int column = warp_n + n_atom * 8 + group;
            b_fragments[n_atom][0] =
                gemm_bi_tf32_rna(gemm_bi_tf32_b_slot<Op>(storage, stage, k8 + thread, column));
            b_fragments[n_atom][1] =
                gemm_bi_tf32_rna(gemm_bi_tf32_b_slot<Op>(storage, stage, k8 + thread + 4, column));
        }
#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
                gemm_bi_tf32_mma_m16n8k8(
                    accumulators[m_atom][n_atom],
                    a_fragments[m_atom], b_fragments[n_atom]);
            }
        }
    }
}

struct SgbTf32ThreadPlan {
    bool compute;
    int warp_m;
    int warp_n;
    int group;
    int thread;
};

template <SgbTf32Op Op, int BM, int BN, int Stages,
          int MAtoms, int NAtoms, bool NarrowA, bool NarrowB>
__device__ __forceinline__ void gemm_bi_tf32_async_mainloop(
    SgbTf32Storage<Op, BM, BN, Stages>* storage,
    const SgbTf32Problem& problem, unsigned tile_count,
    const SgbTf32ThreadPlan& thread_plan,
    float (&accumulators)[MAtoms][NAtoms][4]) {
#pragma unroll
    for (unsigned tile = 0; tile < Stages - 1; ++tile) {
        if (tile < tile_count) {
            gemm_bi_tf32_stage_async<Op, BM, BN, Stages, NarrowA, NarrowB>(
                storage, static_cast<int>(tile), problem,
                static_cast<int>(tile * 32U));
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(Stages - 2));
        __syncthreads();
        unsigned next = tile + Stages - 1;
        if (next < tile_count) {
            gemm_bi_tf32_stage_async<Op, BM, BN, Stages, NarrowA, NarrowB>(
                storage, static_cast<int>(next % Stages), problem,
                static_cast<int>(next * 32U));
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
        if (thread_plan.compute) {
            gemm_bi_tf32_compute_stage<Op, BM, BN, Stages, MAtoms, NAtoms>(
                storage, static_cast<int>(tile % Stages),
                thread_plan.warp_m, thread_plan.warp_n,
                thread_plan.group, thread_plan.thread, accumulators);
        }
        __syncthreads();
    }
}

template <SgbTf32Op Op, int BM, int BN, int Stages>
__device__ __forceinline__ void gemm_bi_tf32_kernel(
    float* output, const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    constexpr int MAtoms = BM == 128 ? 4 : (BM == 64 ? 2 : 1);
    constexpr int NAtoms = BM == 16 ? 1 : 4;
    int columns = gemm_bi_tf32_columns<Op>(params);
    int column_tiles = (columns + BN - 1) / BN;
    SgbTf32Problem problem = {
        output, a, b, bias, params,
        (int)blockIdx.x / column_tiles * BM,
        (int)blockIdx.x % column_tiles * BN,
    };
    extern __shared__ __align__(16) unsigned char gemm_bi_tf32_shared[];
    auto* storage = reinterpret_cast<SgbTf32Storage<Op, BM, BN, Stages>*>(gemm_bi_tf32_shared);

    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    bool compute = BM != 128 || warp < 4;
    int warp_m = BM == 128 ? (warp >> 1) * 64
        : (BM == 64 ? (warp >> 1) * 32 : 0);
    int warp_n = BM == 16 ? warp * 8 : (warp & 1) * 32;
    int group = lane >> 2;
    int thread = lane & 3;
    float accumulators[MAtoms][NAtoms][4];

#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = problem.tile_row + warp_m + m_atom * 16
                    + group + (element >= 2 ? 8 : 0);
                int column = problem.tile_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                float seed = 0.0f;
                if constexpr (Op == SgbTf32Nn) {
                    if (compute && row < params.m && column < params.n && bias != nullptr) {
                        seed = bias[column];
                    }
                }
                accumulators[m_atom][n_atom][element] = seed;
            }
        }
    }

    unsigned tile_count =
        (static_cast<unsigned>(gemm_bi_tf32_reduction<Op>(params)) + 31U) / 32U;
    SgbTf32ThreadPlan thread_plan = {compute, warp_m, warp_n, group, thread};
    bool wide_a = gemm_bi_tf32_can_stage_a16(a, params);
    bool wide_b = gemm_bi_tf32_can_stage_b16(b, params);
    if (wide_a && wide_b) {
        gemm_bi_tf32_async_mainloop<
            Op, BM, BN, Stages, MAtoms, NAtoms, false, false>(
            storage, problem, tile_count, thread_plan, accumulators);
    } else if (gemm_bi_tf32_can_stage_async_4(a, b, params)) {
        if (wide_a) {
            gemm_bi_tf32_async_mainloop<
                Op, BM, BN, Stages, MAtoms, NAtoms, false, true>(
                storage, problem, tile_count, thread_plan, accumulators);
        } else if (wide_b) {
            gemm_bi_tf32_async_mainloop<
                Op, BM, BN, Stages, MAtoms, NAtoms, true, false>(
                storage, problem, tile_count, thread_plan, accumulators);
        } else {
            gemm_bi_tf32_async_mainloop<
                Op, BM, BN, Stages, MAtoms, NAtoms, true, true>(
                storage, problem, tile_count, thread_plan, accumulators);
        }
    } else {
        for (unsigned tile = 0; tile < tile_count; ++tile) {
            int stage = static_cast<int>(tile % Stages);
            gemm_bi_tf32_stage_scalar<Op>(
                storage, stage, problem, static_cast<int>(tile * 32U));
            __syncthreads();
            if (compute) {
                gemm_bi_tf32_compute_stage<Op, BM, BN, Stages, MAtoms, NAtoms>(
                    storage, stage, warp_m, warp_n,
                    group, thread, accumulators);
            }
            __syncthreads();
        }
    }

    if (compute) {
#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
                for (int element = 0; element < 4; ++element) {
                    int row = problem.tile_row + warp_m + m_atom * 16
                        + group + (element >= 2 ? 8 : 0);
                    int column = problem.tile_column + warp_n + n_atom * 8
                        + 2 * thread + (element & 1);
                    gemm_bi_tf32_store<Op>(output, row, column,
                        accumulators[m_atom][n_atom][element], bias, params);
                }
            }
        }
    }
}

template <SgbTf32Op Op, int BM, int BN, int Stages>
__device__ __forceinline__ void gemm_bi_tf32_entry(
    float* output, const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    int reduction = gemm_bi_tf32_reduction<Op>(params);
#line 1001 "mamba_tf32_k0_guard"
    bool zero_reduction = reduction == 0;
#line 1002 "mamba_tf32_k0_branch"
    if (zero_reduction) {
        gemm_bi_tf32_zero_reduction_epilogue<Op, BM, BN>(output, bias, params);
        return;
    }
#line 1750 "sm80.cu"
    gemm_bi_tf32_kernel<Op, BM, BN, Stages>(output, a, b, bias, params);
}

template <SgbTf32Op Op, int BM, int BN, int Stages,
          bool NarrowA, bool NarrowB>
__device__ __forceinline__ void gemm_bi_tf32_splitk_stage_async(
    SgbTf32Storage<Op, BM, BN, Stages>* storage, int stage,
    const SgbTf32Problem& problem, int reduction_base, bool full_output_tile) {
    constexpr int Threads = 128;
    static_assert(Op == SgbTf32Nn || Op == SgbTf32Nt,
                  "split-K supports NN and NT layouts");
    int reduction_extent = gemm_bi_tf32_reduction<Op>(problem.params);
    bool full_reduction_tile = reduction_extent >= 32
        && reduction_base <= reduction_extent - 32;
    if (!full_output_tile || !full_reduction_tile) {
        gemm_bi_tf32_stage_async<
            Op, BM, BN, Stages, NarrowA, NarrowB>(
            storage, stage, problem, reduction_base);
        return;
    }
    for (int linear = (int)threadIdx.x;
         linear < BM * 8;
         linear += Threads) {
        int row = linear >> 3;
        int reduction = (linear & 7) * 4;
        const float* source = problem.a
            + (long long)(problem.tile_row + row) * problem.params.lda
            + reduction_base + reduction;
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            &gemm_bi_tf32_a_slot<Op>(storage, stage, row, reduction));
        gemm_bi_tf32_cp_async_zfill<NarrowA, BM>(destination, source, 16);
    }
    if constexpr (Op == SgbTf32Nt) {
        for (int linear = (int)threadIdx.x;
             linear < BN * 8;
             linear += Threads) {
            int column = linear >> 3;
            int reduction = (linear & 7) * 4;
            const float* source = problem.b
                + (long long)(problem.tile_column + column) * problem.params.ldb
                + reduction_base + reduction;
            unsigned destination = (unsigned)__cvta_generic_to_shared(
                &gemm_bi_tf32_b_slot<Op>(storage, stage, reduction, column));
            gemm_bi_tf32_cp_async_zfill<NarrowB, BM>(destination, source, 16);
        }
    } else {
        for (int linear = (int)threadIdx.x;
             linear < 32 * (BN / 4);
             linear += Threads) {
            int reduction = linear / (BN / 4);
            int column = (linear - reduction * (BN / 4)) * 4;
            const float* source = problem.b
                + (long long)(reduction_base + reduction) * problem.params.ldb
                + problem.tile_column + column;
            unsigned destination = (unsigned)__cvta_generic_to_shared(
                &gemm_bi_tf32_b_slot<Op>(storage, stage, reduction, column));
            gemm_bi_tf32_cp_async_zfill<NarrowB, BM>(destination, source, 16);
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

template <SgbTf32Op Op, int BM, int BN, int Stages, int MAtoms, int NAtoms,
          bool NarrowA, bool NarrowB>
__device__ __forceinline__ void gemm_bi_tf32_splitk_async_mainloop(
    SgbTf32Storage<Op, BM, BN, Stages>* storage,
    const SgbTf32Problem& problem, unsigned tile_begin, unsigned tile_count,
    bool full_output_tile, const SgbTf32ThreadPlan& thread_plan,
    float (&accumulators)[MAtoms][NAtoms][4]) {
#pragma unroll
    for (unsigned tile = 0; tile < Stages - 1; ++tile) {
        if (tile < tile_count) {
            gemm_bi_tf32_splitk_stage_async<
                Op, BM, BN, Stages, NarrowA, NarrowB>(
                storage, static_cast<int>(tile), problem,
                static_cast<int>((tile_begin + tile) * 32U),
                full_output_tile);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(Stages - 2));
        __syncthreads();
        unsigned next = tile + Stages - 1;
        if (next < tile_count) {
            int write_stage = read_stage == 0 ? Stages - 1 : read_stage - 1;
            gemm_bi_tf32_splitk_stage_async<
                Op, BM, BN, Stages, NarrowA, NarrowB>(
                storage, write_stage, problem,
                static_cast<int>((tile_begin + next) * 32U),
                full_output_tile);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
        if (thread_plan.compute) {
            gemm_bi_tf32_compute_stage<
                Op, BM, BN, Stages, MAtoms, NAtoms>(
                storage, read_stage, thread_plan.warp_m, thread_plan.warp_n,
                thread_plan.group, thread_plan.thread, accumulators);
        }
        __syncthreads();
        if (++read_stage == Stages) read_stage = 0;
    }
}

__device__ __forceinline__ void gemm_bi_tf32_partial_store_cg(
    float* address, float value) {
    asm volatile("st.global.cg.f32 [%0], %1;\n" ::
        "l"(address), "f"(value) : "memory");
}

__device__ __forceinline__ void gemm_bi_tf32_partial_store_cg_v2(
    float* address, float2 value) {
    asm volatile("st.global.cg.v2.f32 [%0], {%1, %2};\n" ::
        "l"(address), "f"(value.x), "f"(value.y) : "memory");
}

__device__ __forceinline__ float gemm_bi_tf32_partial_load_cg(
    const float* address) {
    float value;
    asm volatile("ld.global.cg.f32 %0, [%1];\n" :
        "=f"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ float2 gemm_bi_tf32_partial_load_cg_v2(
    const float* address) {
    float2 value;
    asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];\n" :
        "=f"(value.x), "=f"(value.y) : "l"(address) : "memory");
    return value;
}

template <SgbTf32Op Op, int BM, int BN, int Stages, int Partitions>
__device__ __forceinline__ void gemm_bi_tf32_splitk_fused_kernel(
    float* output, float* partial, unsigned* counters,
    const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    constexpr int MAtoms = BM == 128 ? 4 : (BM == 64 ? 2 : 1);
    constexpr int NAtoms = BM == 16 ? 1 : (BM == 32 ? 2 : 4);
    static_assert(Op == SgbTf32Nn || Op == SgbTf32Nt,
                  "split-K supports NN and NT layouts");
    static_assert(
        (Op == SgbTf32Nn && BM == 16 && BN == 32 && Stages == 4
         && (Partitions == 2 || Partitions == 4))
        || (Op == SgbTf32Nt && BM == 16 && BN == 32
            && (Stages == 3 || Stages == 4) && Partitions == 4)
        || (Op == SgbTf32Nt && BM == 32 && BN == 32
            && (Stages == 3 || Stages == 4) && Partitions == 8),
        "unsupported split-K layout");
    int rows = gemm_bi_tf32_rows<Op>(params);
    int columns = gemm_bi_tf32_columns<Op>(params);
    int reduction = gemm_bi_tf32_reduction<Op>(params);
    if constexpr (Op == SgbTf32Nt) {
        assert(bias == nullptr);
        assert(params.beta == 0.0f);
    }
    int partition = (int)blockIdx.z;
    unsigned full_tiles = (static_cast<unsigned>(reduction) + 31U) / 32U;
    unsigned tiles_per_partition = (full_tiles + Partitions - 1U) / Partitions;
    unsigned tile_begin = min(static_cast<unsigned>(partition) * tiles_per_partition, full_tiles);
    unsigned tile_end = min(tile_begin + tiles_per_partition, full_tiles);
    SgbTf32Problem problem = {
        partial, a, b, nullptr, params,
        (int)blockIdx.y * BM,
        (int)blockIdx.x * BN,
    };
    bool full_output_tile = problem.tile_row + BM <= rows
        && problem.tile_column + BN <= columns;
    extern __shared__ __align__(16) unsigned char gemm_bi_tf32_shared[];
    auto* storage = reinterpret_cast<SgbTf32Storage<Op, BM, BN, Stages>*>(
        gemm_bi_tf32_shared);

    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    bool compute = BM != 128 || warp < 4;
    int warp_m = BM == 128 ? (warp >> 1) * 64
        : (BM == 64 ? (warp >> 1) * 32
            : (BM == 32 ? (warp >> 1) * 16 : 0));
    int warp_n = BM == 16 ? warp * 8
        : (BM == 32 ? (warp & 1) * 16 : (warp & 1) * 32);
    int group = lane >> 2;
    int thread = lane & 3;
    float accumulators[MAtoms][NAtoms][4] = {};
    unsigned tile_count = tile_end - tile_begin;
    SgbTf32ThreadPlan thread_plan = {compute, warp_m, warp_n, group, thread};

    bool wide_a = gemm_bi_tf32_can_stage_a16(a, params);
    bool wide_b = gemm_bi_tf32_can_stage_b16(b, params);
    if (wide_a && wide_b) {
        gemm_bi_tf32_splitk_async_mainloop<
            Op, BM, BN, Stages, MAtoms, NAtoms, false, false>(
            storage, problem, tile_begin, tile_count,
            full_output_tile, thread_plan, accumulators);
    } else if (gemm_bi_tf32_can_stage_async_4(a, b, params)) {
        if (wide_a) {
            gemm_bi_tf32_splitk_async_mainloop<
                Op, BM, BN, Stages, MAtoms, NAtoms, false, true>(
                storage, problem, tile_begin, tile_count,
                full_output_tile, thread_plan, accumulators);
        } else if (wide_b) {
            gemm_bi_tf32_splitk_async_mainloop<
                Op, BM, BN, Stages, MAtoms, NAtoms, true, false>(
                storage, problem, tile_begin, tile_count,
                full_output_tile, thread_plan, accumulators);
        } else {
            gemm_bi_tf32_splitk_async_mainloop<
                Op, BM, BN, Stages, MAtoms, NAtoms, true, true>(
                storage, problem, tile_begin, tile_count,
                full_output_tile, thread_plan, accumulators);
        }
    } else {
        int stage = 0;
        for (unsigned tile = 0; tile < tile_count; ++tile) {
            gemm_bi_tf32_stage_scalar<Op>(
                storage, stage, problem,
                static_cast<int>((tile_begin + tile) * 32U));
            __syncthreads();
            if (compute) {
                gemm_bi_tf32_compute_stage<
                    Op, BM, BN, Stages, MAtoms, NAtoms>(
                    storage, stage, warp_m, warp_n,
                    group, thread, accumulators);
            }
            __syncthreads();
            if (++stage == Stages) stage = 0;
        }
    }

    if (compute) {
        long long partial_stride = (long long)rows * columns;
        float* partition_output = partial + (long long)partition * partial_stride;
        bool packed_output = full_output_tile && (columns & 1) == 0
            && gemm_bi_is_aligned_8(partition_output);
#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = problem.tile_row + warp_m + m_atom * 16
                        + group + half * 8;
                    int column = problem.tile_column + warp_n + n_atom * 8
                        + 2 * thread;
                    if (packed_output) {
                        float* destination = partition_output
                            + (long long)row * columns + column;
                        gemm_bi_tf32_partial_store_cg_v2(destination, make_float2(
                            accumulators[m_atom][n_atom][2 * half],
                            accumulators[m_atom][n_atom][2 * half + 1]));
                    } else if (row < rows) {
                        float* destination = partition_output
                            + (long long)row * columns + column;
#pragma unroll
                        for (int element = 0; element < 2; ++element) {
                            if (column + element < columns) {
                                gemm_bi_tf32_partial_store_cg(
                                    destination + element,
                                    accumulators[m_atom][n_atom][2 * half + element]);
                            }
                        }
                    }
                }
            }
        }
    }

    __threadfence();
    __syncthreads();
    __shared__ bool last_partition;
    if (threadIdx.x == 0) {
        unsigned tile = (unsigned)blockIdx.y * (unsigned)gridDim.x
            + (unsigned)blockIdx.x;
        last_partition = atomicInc(counters + tile, Partitions - 1U)
            == Partitions - 1U;
    }
    __syncthreads();
    if (!last_partition || !compute) return;

    long long partial_stride = (long long)rows * columns;
#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = problem.tile_row + warp_m + m_atom * 16
                    + group + half * 8;
                int column = problem.tile_column + warp_n + n_atom * 8
                    + 2 * thread;
                if (row >= rows || column >= columns) continue;
                long long index = (long long)row * columns + column;
                float* destination = output + (long long)row * params.ldc + column;
                bool has_second = column + 1 < columns;
                bool packed_partial = has_second && (columns & 1) == 0
                    && gemm_bi_is_aligned_8(partial + index);
                float2 p0;
                float2 p1;
                float2 p2;
                float2 p3;
                float2 p4;
                float2 p5;
                float2 p6;
                float2 p7;
                if (packed_partial) {
                    p0 = gemm_bi_tf32_partial_load_cg_v2(partial + index);
                    p1 = gemm_bi_tf32_partial_load_cg_v2(
                        partial + partial_stride + index);
                    if constexpr (Partitions >= 4) {
                        p2 = gemm_bi_tf32_partial_load_cg_v2(
                            partial + 2 * partial_stride + index);
                        p3 = gemm_bi_tf32_partial_load_cg_v2(
                            partial + 3 * partial_stride + index);
                    }
                    if constexpr (Partitions == 8) {
                        p4 = gemm_bi_tf32_partial_load_cg_v2(
                            partial + 4 * partial_stride + index);
                        p5 = gemm_bi_tf32_partial_load_cg_v2(
                            partial + 5 * partial_stride + index);
                        p6 = gemm_bi_tf32_partial_load_cg_v2(
                            partial + 6 * partial_stride + index);
                        p7 = gemm_bi_tf32_partial_load_cg_v2(
                            partial + 7 * partial_stride + index);
                    }
                } else {
                    p0.x = gemm_bi_tf32_partial_load_cg(partial + index);
                    p1.x = gemm_bi_tf32_partial_load_cg(
                        partial + partial_stride + index);
                    if constexpr (Partitions >= 4) {
                        p2.x = gemm_bi_tf32_partial_load_cg(
                            partial + 2 * partial_stride + index);
                        p3.x = gemm_bi_tf32_partial_load_cg(
                            partial + 3 * partial_stride + index);
                    }
                    if constexpr (Partitions == 8) {
                        p4.x = gemm_bi_tf32_partial_load_cg(
                            partial + 4 * partial_stride + index);
                        p5.x = gemm_bi_tf32_partial_load_cg(
                            partial + 5 * partial_stride + index);
                        p6.x = gemm_bi_tf32_partial_load_cg(
                            partial + 6 * partial_stride + index);
                        p7.x = gemm_bi_tf32_partial_load_cg(
                            partial + 7 * partial_stride + index);
                    }
                    if (has_second) {
                        p0.y = gemm_bi_tf32_partial_load_cg(partial + index + 1);
                        p1.y = gemm_bi_tf32_partial_load_cg(
                            partial + partial_stride + index + 1);
                        if constexpr (Partitions >= 4) {
                            p2.y = gemm_bi_tf32_partial_load_cg(
                                partial + 2 * partial_stride + index + 1);
                            p3.y = gemm_bi_tf32_partial_load_cg(
                                partial + 3 * partial_stride + index + 1);
                        }
                        if constexpr (Partitions == 8) {
                            p4.y = gemm_bi_tf32_partial_load_cg(
                                partial + 4 * partial_stride + index + 1);
                            p5.y = gemm_bi_tf32_partial_load_cg(
                                partial + 5 * partial_stride + index + 1);
                            p6.y = gemm_bi_tf32_partial_load_cg(
                                partial + 6 * partial_stride + index + 1);
                            p7.y = gemm_bi_tf32_partial_load_cg(
                                partial + 7 * partial_stride + index + 1);
                        }
                    }
                }
                float value0;
                if constexpr (Op == SgbTf32Nn) {
                    float bias0 = bias == nullptr ? 0.0f : bias[column];
                    float sum0 = bias0;
                    sum0 = __fadd_rn(sum0, p0.x);
                    sum0 = __fadd_rn(sum0, p1.x);
                    if constexpr (Partitions == 4) {
                        sum0 = __fadd_rn(sum0, p2.x);
                        sum0 = __fadd_rn(sum0, p3.x);
                    }
                    value0 = params.alpha == 1.0f
                        ? sum0 : __fmul_rn(params.alpha, sum0);
                    if (params.beta != 0.0f) {
                        value0 = __fmaf_rn(params.beta, destination[0], value0);
                    }
                } else {
                    float sum0 = __fadd_rn(p0.x, p1.x);
                    sum0 = __fadd_rn(sum0, p2.x);
                    sum0 = __fadd_rn(sum0, p3.x);
                    if constexpr (Partitions == 8) {
                        sum0 = __fadd_rn(sum0, p4.x);
                        sum0 = __fadd_rn(sum0, p5.x);
                        sum0 = __fadd_rn(sum0, p6.x);
                        sum0 = __fadd_rn(sum0, p7.x);
                    }
                    value0 = params.alpha == 1.0f
                        ? sum0 : __fmul_rn(params.alpha, sum0);
                }
                if (has_second) {
                    float value1;
                    if constexpr (Op == SgbTf32Nn) {
                        float sum1 = bias == nullptr ? 0.0f : bias[column + 1];
                        sum1 = __fadd_rn(sum1, p0.y);
                        sum1 = __fadd_rn(sum1, p1.y);
                        if constexpr (Partitions == 4) {
                            sum1 = __fadd_rn(sum1, p2.y);
                            sum1 = __fadd_rn(sum1, p3.y);
                        }
                        value1 = params.alpha == 1.0f
                            ? sum1 : __fmul_rn(params.alpha, sum1);
                        if (params.beta != 0.0f) {
                            value1 = __fmaf_rn(params.beta, destination[1], value1);
                        }
                    } else {
                        float sum1 = __fadd_rn(p0.y, p1.y);
                        sum1 = __fadd_rn(sum1, p2.y);
                        sum1 = __fadd_rn(sum1, p3.y);
                        if constexpr (Partitions == 8) {
                            sum1 = __fadd_rn(sum1, p4.y);
                            sum1 = __fadd_rn(sum1, p5.y);
                            sum1 = __fadd_rn(sum1, p6.y);
                            sum1 = __fadd_rn(sum1, p7.y);
                        }
                        value1 = params.alpha == 1.0f
                            ? sum1 : __fmul_rn(params.alpha, sum1);
                    }
                    if (gemm_bi_is_aligned_8(destination)) {
                        *reinterpret_cast<float2*>(destination) =
                            make_float2(value0, value1);
                    } else {
                        destination[0] = value0;
                        destination[1] = value1;
                    }
                } else {
                    destination[0] = value0;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 3)
void gemm_bi_nn_sm80_mma_tf32_splitk2_v1_m16n32_bk32_s4(
    float* output, float* partial, unsigned* counters,
    const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    gemm_bi_tf32_splitk_fused_kernel<SgbTf32Nn, 16, 32, 4, 2>(
        output, partial, counters, a, b, bias, params);
}

extern "C" __global__ __launch_bounds__(128, 3)
void gemm_bi_nn_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4(
    float* output, float* partial, unsigned* counters,
    const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    gemm_bi_tf32_splitk_fused_kernel<SgbTf32Nn, 16, 32, 4, 4>(
        output, partial, counters, a, b, bias, params);
}

extern "C" __global__ __launch_bounds__(128, 3)
void gemm_bi_nt_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s3(
    float* output, float* partial, unsigned* counters,
    const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    gemm_bi_tf32_splitk_fused_kernel<SgbTf32Nt, 16, 32, 3, 4>(
        output, partial, counters, a, b, bias, params);
}

extern "C" __global__ __launch_bounds__(128, 3)
void gemm_bi_nt_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4(
    float* output, float* partial, unsigned* counters,
    const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    gemm_bi_tf32_splitk_fused_kernel<SgbTf32Nt, 16, 32, 4, 4>(
        output, partial, counters, a, b, bias, params);
}

extern "C" __global__ __launch_bounds__(128, 3)
void gemm_bi_nt_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s3(
    float* output, float* partial, unsigned* counters,
    const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    gemm_bi_tf32_splitk_fused_kernel<SgbTf32Nt, 32, 32, 3, 8>(
        output, partial, counters, a, b, bias, params);
}

extern "C" __global__ __launch_bounds__(128, 2)
void gemm_bi_nt_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s4(
    float* output, float* partial, unsigned* counters,
    const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    gemm_bi_tf32_splitk_fused_kernel<SgbTf32Nt, 32, 32, 4, 8>(
        output, partial, counters, a, b, bias, params);
}

#define GEMM_BI_TF32_DEFINE_KERNEL(NAME, OP, BM, BN, STAGES, THREADS, MIN_BLOCKS) \
extern "C" __global__ __launch_bounds__(THREADS, MIN_BLOCKS) void NAME(       \
    float* output, const float* a, const float* b, const float* bias,          \
    Sm80Tf32KernelParams params) {                                             \
    gemm_bi_tf32_entry<OP, BM, BN, STAGES>(output, a, b, bias, params);            \
}

GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_nn_sm80_mma_tf32_v1_m128n64_bk32_s2, SgbTf32Nn, 128, 64, 2, 256, 1)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_nn_sm80_mma_tf32_v1_m128n64_bk32_s3, SgbTf32Nn, 128, 64, 3, 256, 1)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_nn_sm80_mma_tf32_v1_m64n64_bk32_s2, SgbTf32Nn, 64, 64, 2, 128, 1)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_nn_sm80_mma_tf32_v1_m64n64_bk32_s3, SgbTf32Nn, 64, 64, 3, 128, 1)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_nn_sm80_mma_tf32_v1_m16n32_bk32_s4, SgbTf32Nn, 16, 32, 4, 128, 3)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_nn_sm80_mma_tf32_v1_m16n16_bk32_s4, SgbTf32Nn, 16, 16, 4, 64, 4)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s2, SgbTf32Tn, 128, 64, 2, 256, 1)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s3, SgbTf32Tn, 128, 64, 3, 256, 1)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_tn_sm80_mma_tf32_v1_m64n64_bk32_s2, SgbTf32Tn, 64, 64, 2, 128, 1)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_tn_sm80_mma_tf32_v1_m64n64_bk32_s3, SgbTf32Tn, 64, 64, 3, 128, 1)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4, SgbTf32Tn, 16, 32, 4, 128, 3)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_tn_sm80_mma_tf32_v1_m16n16_bk32_s4, SgbTf32Tn, 16, 16, 4, 64, 4)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s2, SgbTf32Nt, 128, 64, 2, 256, 1)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3, SgbTf32Nt, 128, 64, 3, 256, 1)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_nt_sm80_mma_tf32_v1_m64n64_bk32_s2, SgbTf32Nt, 64, 64, 2, 128, 1)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_nt_sm80_mma_tf32_v1_m64n64_bk32_s3, SgbTf32Nt, 64, 64, 3, 128, 1)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_nt_sm80_mma_tf32_v1_m16n32_bk32_s4, SgbTf32Nt, 16, 32, 4, 128, 3)
GEMM_BI_TF32_DEFINE_KERNEL(gemm_bi_nt_sm80_mma_tf32_v1_m16n16_bk32_s4, SgbTf32Nt, 16, 16, 4, 64, 4)

template <typename A, typename B> struct SgbTf32SameType { static constexpr bool value = false; };
template <typename A> struct SgbTf32SameType<A, A> { static constexpr bool value = true; };
using SgbTf32KernelSignature = void (*)(
    float*, const float*, const float*, const float*, Sm80Tf32KernelParams);
#define TF32_ASSERT_KERNEL_SIGNATURE(NAME) \
    static_assert(SgbTf32SameType<decltype(&NAME), SgbTf32KernelSignature>::value, "TF32 kernel signature")

TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nn_sm80_mma_tf32_v1_m128n64_bk32_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nn_sm80_mma_tf32_v1_m128n64_bk32_s3);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nn_sm80_mma_tf32_v1_m64n64_bk32_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nn_sm80_mma_tf32_v1_m64n64_bk32_s3);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nn_sm80_mma_tf32_v1_m16n32_bk32_s4);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nn_sm80_mma_tf32_v1_m16n16_bk32_s4);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s3);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm80_mma_tf32_v1_m64n64_bk32_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm80_mma_tf32_v1_m64n64_bk32_s3);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm80_mma_tf32_v1_m16n16_bk32_s4);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nt_sm80_mma_tf32_v1_m64n64_bk32_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nt_sm80_mma_tf32_v1_m64n64_bk32_s3);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nt_sm80_mma_tf32_v1_m16n32_bk32_s4);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nt_sm80_mma_tf32_v1_m16n16_bk32_s4);
using SgbTf32SplitKKernelSignature = void (*)(
    float*, float*, unsigned*, const float*, const float*, const float*,
    Sm80Tf32KernelParams);
#define TF32_ASSERT_SPLITK_KERNEL_SIGNATURE(NAME) \
    static_assert(SgbTf32SameType<decltype(&NAME), SgbTf32SplitKKernelSignature>::value, \
        "TF32 split-K kernel signature")
TF32_ASSERT_SPLITK_KERNEL_SIGNATURE(gemm_bi_nn_sm80_mma_tf32_splitk2_v1_m16n32_bk32_s4);
TF32_ASSERT_SPLITK_KERNEL_SIGNATURE(gemm_bi_nn_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4);
TF32_ASSERT_SPLITK_KERNEL_SIGNATURE(gemm_bi_nt_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s3);
TF32_ASSERT_SPLITK_KERNEL_SIGNATURE(gemm_bi_nt_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4);
TF32_ASSERT_SPLITK_KERNEL_SIGNATURE(gemm_bi_nt_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s3);
TF32_ASSERT_SPLITK_KERNEL_SIGNATURE(gemm_bi_nt_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s4);

#undef TF32_ASSERT_SPLITK_KERNEL_SIGNATURE
#undef TF32_ASSERT_KERNEL_SIGNATURE
#undef GEMM_BI_TF32_DEFINE_KERNEL
