#define SGB_TC128_BM 128
#define SGB_TC128_BN 128
#define SGB_TC128_BK 64
#define SGB_TC128_PAD_A 8
#define SGB_TC128_PAD_B 8
#define SGB_TC128_LDA (SGB_TC128_BK + SGB_TC128_PAD_A)
#define SGB_TC128_LDB (SGB_TC128_BN + SGB_TC128_PAD_B)

// Issue one A+B tile into smem stage `buf` via 16B cp.async with zero-fill
// (fast path; the caller proves 16-byte operand bases and row starts).
// A: 128 rows x 8 chunks; B: 64 rows x 16 chunks; 2048 cp.async / 256 thr.
#define SGB_TC128_STAGE_ASYNC(buf, bkIdx)                                        \
    do {                                                                      \
        unsigned _as = As_sbase + (unsigned)((buf) * SGB_TC128_BM * SGB_TC128_LDA * 2);     \
        unsigned _bs = Bs_sbase + (unsigned)((buf) * SGB_TC128_BK * SGB_TC128_LDB * 2);     \
        for (int _i = threadIdx.x; _i < SGB_TC128_BM * (SGB_TC128_BK / 8); _i += 256) {     \
            int _m = _i / (SGB_TC128_BK / 8);                                        \
            int _c = _i % (SGB_TC128_BK / 8);                                        \
            int _k = _c * 8;                                                  \
            int _gr = pid_m * SGB_TC128_BM + _m;                                     \
            int _gc = (bkIdx) + _k;                                           \
            int _elems = sgb_cp_async_valid_elems(_gr < M, K, _gc);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _as + (unsigned)((_m * SGB_TC128_LDA + _k) * 2);         \
            long long _offset = _bytes == 0 ? 0 : (long long)_gr * lda + _gc; \
            const void* _src = sgb_cp_async_source(A, _offset, _bytes);        \
            sgb_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SGB_TC128_BK * (SGB_TC128_BN / 8); _i += 256) {     \
            int _k = _i / (SGB_TC128_BN / 8);                                        \
            int _c = _i % (SGB_TC128_BN / 8);                                        \
            int _n = _c * 8;                                                  \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * SGB_TC128_BN + _n;                                     \
            int _elems = sgb_cp_async_valid_elems(_gk < K, N, _gn);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _bs + (unsigned)((_k * SGB_TC128_LDB + _n) * 2);         \
            long long _offset = _bytes == 0 ? 0 : (long long)_gk * ldb + _gn; \
            const void* _src = sgb_cp_async_source(B, _offset, _bytes);        \
            sgb_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

// Scalar staging fallback for misaligned lda/ldb (rare; uniform branch).
#define SGB_TC128_STAGE_SCALAR(buf, bkIdx, TT, FF)                               \
    do {                                                                      \
        TT* _Asw = &As[buf][0][0];                                            \
        TT* _Bsw = &Bs[buf][0][0];                                            \
        for (int _i = threadIdx.x; _i < SGB_TC128_BM * SGB_TC128_BK; _i += 256) {           \
            int _m = _i / SGB_TC128_BK;                                              \
            int _k = _i % SGB_TC128_BK;                                              \
            int _gr = pid_m * SGB_TC128_BM + _m;                                     \
            int _gc = (bkIdx) + _k;                                           \
            _Asw[_m * SGB_TC128_LDA + _k] = (_gr < M && _gc < K)                     \
                                         ? A[(long long)_gr * lda + _gc]      \
                                         : FF(0.0f);                          \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SGB_TC128_BK * SGB_TC128_BN; _i += 256) {           \
            int _k = _i / SGB_TC128_BN;                                              \
            int _n = _i % SGB_TC128_BN;                                              \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * SGB_TC128_BN + _n;                                     \
            _Bsw[_k * SGB_TC128_LDB + _n] = (_gk < K && _gn < N)                     \
                                         ? B[(long long)_gk * ldb + _gn]      \
                                         : FF(0.0f);                          \
        }                                                                     \
    } while (0)

#define SGB_DEFINE_SGEMM_BI_NN_TC128(SUFFIX, T_ACT, FROM_F, MMA_T)                    \
extern "C" __global__ __launch_bounds__(256, 1)                                \
void sgemm_bi_nn_tc_##SUFFIX(                                                  \
    T_ACT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    const float* __restrict__ bias,                                            \
    float alpha, float beta,                                                   \
    int M, int N, int K,                                                       \
    int lda, int ldb, int ldc                                                  \
) {                                                                            \
    assert(alpha == 1.0f || bias == nullptr);                                  \
    extern __shared__ __align__(16) unsigned char sgb_tc_dynsmem[];           \
    T_ACT (*As)[SGB_TC128_BM][SGB_TC128_LDA] =                                               \
        reinterpret_cast<T_ACT (*)[SGB_TC128_BM][SGB_TC128_LDA]>(sgb_tc_dynsmem);            \
    T_ACT (*Bs)[SGB_TC128_BK][SGB_TC128_LDB] = reinterpret_cast<T_ACT (*)[SGB_TC128_BK][SGB_TC128_LDB]>(   \
        sgb_tc_dynsmem + 2 * SGB_TC128_BM * SGB_TC128_LDA * (int)sizeof(T_ACT));             \
    int num_pid_n = (N + SGB_TC128_BN - 1) / SGB_TC128_BN;                                   \
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
    bool fast_stage = sgb_is_aligned_16(A) && sgb_is_aligned_16(B) &&         \
                      ((lda & 7) == 0) && ((ldb & 7) == 0);                    \
    bool packed_epilogue = sgb_is_aligned_4(C) && ((ldc & 1) == 0);           \
    float acc[4][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 4; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            float b0 = 0.0f, b1 = 0.0f;                                        \
            if (bias != nullptr) {                                             \
                int c0 = pid_n * SGB_TC128_BN + warpN + fn * 8 + 2 * t;               \
                b0 = (c0 < N) ? bias[c0] : 0.0f;                               \
                b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                       \
            }                                                                  \
            acc[fm][fn][0] = b0;                                               \
            acc[fm][fn][1] = b1;                                               \
            acc[fm][fn][2] = b0;                                               \
            acc[fm][fn][3] = b1;                                               \
        }                                                                      \
    }                                                                          \
    int num_k_tiles = (K + SGB_TC128_BK - 1) / SGB_TC128_BK;                                 \
    if (fast_stage) {                                                          \
        SGB_TC128_STAGE_ASYNC(0, 0);                                              \
    } else {                                                                   \
        SGB_TC128_STAGE_SCALAR(0, 0, T_ACT, FROM_F);                              \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (kt + 1 < num_k_tiles) {                                            \
            if (fast_stage) {                                                  \
                SGB_TC128_STAGE_ASYNC(read_buf ^ 1, (kt + 1) * SGB_TC128_BK);            \
            } else {                                                           \
                SGB_TC128_STAGE_SCALAR(read_buf ^ 1, (kt + 1) * SGB_TC128_BK, T_ACT,     \
                                    FROM_F);                                   \
            }                                                                  \
        }                                                                      \
        unsigned As_rd = As_sbase + (unsigned)(read_buf * SGB_TC128_BM * SGB_TC128_LDA * 2); \
        unsigned Bs_rd = Bs_sbase + (unsigned)(read_buf * SGB_TC128_BK * SGB_TC128_LDB * 2); \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (SGB_TC128_BK / 16); ks++) {                            \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[4][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 4; fm++) {                                   \
                int row = warpM + fm * 16 + lm_row_off + lm_r;                 \
                unsigned addr = As_rd +                                        \
                    (unsigned)((row * SGB_TC128_LDA + k0 + lm_col_off) * 2);          \
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
                    (unsigned)((row * SGB_TC128_LDB + warpN + fn * 8) * 2);           \
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
            int r0 = pid_m * SGB_TC128_BM + warpM + fm * 16 + g;                     \
            int c0 = pid_n * SGB_TC128_BN + warpN + fn * 8 + 2 * t;                  \
            /* c0 is even. A 4-byte base and even ldc keep every row pair      \
               aligned; subviews and odd strides use the same scalar RNE. */   \
            _Pragma("unroll")                                                 \
            for (int half = 0; half < 2; half++) {                            \
                int gr = r0 + (half ? 8 : 0);                                 \
                if (gr >= M) continue;                                        \
                T_ACT* dst = sgb_output_start_if_valid(                        \
                    C, (long long)gr * ldc, c0, N);                            \
                if (dst == nullptr) continue;                                 \
                float v0 = alpha * acc[fm][fn][2 * half];                     \
                float v1 = alpha * acc[fm][fn][2 * half + 1];                 \
                if (beta == 0.0f && packed_epilogue && c0 + 1 < N &&         \
                    sgb_is_aligned_4(dst)) {                                   \
                    sgb_store_pair_rne(dst, v0, v1);                           \
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

SGB_DEFINE_SGEMM_BI_NN_TC128(bf16, __nv_bfloat16, from_f_bf16, "bf16")
SGB_DEFINE_SGEMM_BI_NN_TC128(f16,  __half,        from_f_f16,  "f16")

// ============================================================================
// Tensor-core backward twins: TN dW and NT dX (bi_tensor_cores tier).
// ============================================================================
// Same numeric contract class as sgemm_bi_nn_tc_*: deterministic (fixed
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
// are the M-reduction dim (SGB_TC128_BK rows per tile).
#define SGB_TC128_STAGE_TN_ASYNC(buf, mIdx)                                      \
    do {                                                                      \
        unsigned _xs = Xs_sbase + (unsigned)((buf) * SGB_TC128_BK * SGB_TC128_LDB * 2);     \
        unsigned _ys = Ys_sbase + (unsigned)((buf) * SGB_TC128_BK * SGB_TC128_LDB * 2);     \
        for (int _i = threadIdx.x; _i < SGB_TC128_BK * (SGB_TC128_BM / 8); _i += 256) {     \
            int _r = _i / (SGB_TC128_BM / 8);                                        \
            int _c = (_i % (SGB_TC128_BM / 8)) * 8;                                  \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * SGB_TC128_BM + _c;                                     \
            int _elems = sgb_cp_async_valid_elems(_gm < M_red, K_out, _gk);   \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _xs + (unsigned)((_r * SGB_TC128_LDB + _c) * 2);         \
            long long _offset =                                                \
                _bytes == 0 ? 0 : (long long)_gm * K_out + _gk;               \
            const void* _src = sgb_cp_async_source(A, _offset, _bytes);        \
            sgb_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SGB_TC128_BK * (SGB_TC128_BN / 8); _i += 256) {     \
            int _r = _i / (SGB_TC128_BN / 8);                                        \
            int _c = (_i % (SGB_TC128_BN / 8)) * 8;                                  \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * SGB_TC128_BN + _c;                                     \
            int _elems = sgb_cp_async_valid_elems(_gm < M_red, N, _gn);       \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ys + (unsigned)((_r * SGB_TC128_LDB + _c) * 2);         \
            long long _offset = _bytes == 0 ? 0 : (long long)_gm * N + _gn;   \
            const void* _src = sgb_cp_async_source(B, _offset, _bytes);        \
            sgb_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define SGB_TC128_STAGE_TN_SCALAR(buf, mIdx, TT, FF)                             \
    do {                                                                      \
        TT* _xs = &Xs[buf][0][0];                                             \
        TT* _ys = &Ys[buf][0][0];                                             \
        for (int _i = threadIdx.x; _i < SGB_TC128_BK * SGB_TC128_BM; _i += 256) {           \
            int _r = _i / SGB_TC128_BM;                                              \
            int _c = _i % SGB_TC128_BM;                                              \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * SGB_TC128_BM + _c;                                     \
            _xs[_r * SGB_TC128_LDB + _c] = (_gm < M_red && _gk < K_out)              \
                                        ? A[(long long)_gm * K_out + _gk]     \
                                        : FF(0.0f);                           \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SGB_TC128_BK * SGB_TC128_BN; _i += 256) {           \
            int _r = _i / SGB_TC128_BN;                                              \
            int _c = _i % SGB_TC128_BN;                                              \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * SGB_TC128_BN + _c;                                     \
            _ys[_r * SGB_TC128_LDB + _c] = (_gm < M_red && _gn < N)                  \
                                        ? B[(long long)_gm * N + _gn]         \
                                        : FF(0.0f);                           \
        }                                                                     \
    } while (0)

#define SGB_DEFINE_SGEMM_BI_TN_TC128(SUFFIX, T_ACT, FROM_F, MMA_T)                    \
extern "C" __global__ __launch_bounds__(256, 1)                                \
void sgemm_bi_tn_tc_##SUFFIX(                                                  \
    float* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M_red, int K_out, int N                                                \
) {                                                                            \
    extern __shared__ __align__(16) unsigned char sgb_tc_dynsmem[];           \
    T_ACT (*Xs)[SGB_TC128_BK][SGB_TC128_LDB] =                                               \
        reinterpret_cast<T_ACT (*)[SGB_TC128_BK][SGB_TC128_LDB]>(sgb_tc_dynsmem);            \
    T_ACT (*Ys)[SGB_TC128_BK][SGB_TC128_LDB] = reinterpret_cast<T_ACT (*)[SGB_TC128_BK][SGB_TC128_LDB]>(   \
        sgb_tc_dynsmem + 2 * SGB_TC128_BK * SGB_TC128_LDB * (int)sizeof(T_ACT));             \
    int num_pid_n = (N + SGB_TC128_BN - 1) / SGB_TC128_BN;                                   \
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
    bool fast_stage = sgb_is_aligned_16(A) && sgb_is_aligned_16(B) &&         \
                      ((K_out & 7) == 0) && ((N & 7) == 0);                    \
    bool packed_epilogue = sgb_is_aligned_8(C) && ((N & 1) == 0);             \
    float acc[4][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 4; fm++)                                             \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++)                                         \
            _Pragma("unroll")                                                  \
            for (int e = 0; e < 4; e++) acc[fm][fn][e] = 0.0f;                 \
    int num_m_tiles = (M_red + SGB_TC128_BK - 1) / SGB_TC128_BK;                             \
    if (fast_stage) {                                                          \
        SGB_TC128_STAGE_TN_ASYNC(0, 0);                                           \
    } else {                                                                   \
        SGB_TC128_STAGE_TN_SCALAR(0, 0, T_ACT, FROM_F);                           \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int mt = 0; mt < num_m_tiles; mt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (mt + 1 < num_m_tiles) {                                            \
            if (fast_stage) {                                                  \
                SGB_TC128_STAGE_TN_ASYNC(read_buf ^ 1, (mt + 1) * SGB_TC128_BK);         \
            } else {                                                           \
                SGB_TC128_STAGE_TN_SCALAR(read_buf ^ 1, (mt + 1) * SGB_TC128_BK, T_ACT,  \
                                       FROM_F);                                \
            }                                                                  \
        }                                                                      \
        unsigned Xs_rd = Xs_sbase + (unsigned)(read_buf * SGB_TC128_BK * SGB_TC128_LDB * 2); \
        unsigned Ys_rd = Ys_sbase + (unsigned)(read_buf * SGB_TC128_BK * SGB_TC128_LDB * 2); \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (SGB_TC128_BK / 16); ks++) {                            \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[4][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 4; fm++) {                                   \
                int srow = k0 + lm_arow_off + lm_r;                            \
                int scol = warpM + fm * 16 + lm_acol_off;                      \
                unsigned addr =                                                \
                    Xs_rd + (unsigned)((srow * SGB_TC128_LDB + scol) * 2);            \
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
                    (unsigned)((srow * SGB_TC128_LDB + warpN + fn * 8) * 2);          \
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
            int r0 = pid_m * SGB_TC128_BM + warpM + fm * 16 + g;                     \
            int c0 = pid_n * SGB_TC128_BN + warpN + fn * 8 + 2 * t;                  \
            _Pragma("unroll")                                                 \
            for (int half = 0; half < 2; half++) {                            \
                int gr = r0 + (half ? 8 : 0);                                 \
                if (gr >= K_out) continue;                                    \
                float* dst = sgb_output_start_if_valid(                        \
                    C, (long long)gr * N, c0, N);                              \
                if (dst == nullptr) continue;                                 \
                if (c0 + 1 < N) {                                             \
                    float x = alpha * acc[fm][fn][2 * half];                   \
                    float y = alpha * acc[fm][fn][2 * half + 1];               \
                    bool packed = packed_epilogue && sgb_is_aligned_8(dst);    \
                    sgb_accumulate_float2_or_scalar(dst, x, y, packed);        \
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

SGB_DEFINE_SGEMM_BI_TN_TC128(bf16, __nv_bfloat16, from_f_bf16, "bf16")
SGB_DEFINE_SGEMM_BI_TN_TC128(f16,  __half,        from_f_f16,  "f16")

// NT staging: dYs[m_local][n chunk] (output rows x reduction) and
// Ws[k_out_local][n chunk] (output cols x reduction).
#define SGB_TC128_STAGE_NT_ASYNC(buf, nIdx)                                      \
    do {                                                                      \
        unsigned _ys = Ys_sbase + (unsigned)((buf) * SGB_TC128_BM * SGB_TC128_LDA * 2);     \
        unsigned _ws = Ws_sbase + (unsigned)((buf) * SGB_TC128_BN * SGB_TC128_LDA * 2);     \
        for (int _i = threadIdx.x; _i < SGB_TC128_BM * (SGB_TC128_BK / 8); _i += 256) {     \
            int _m = _i / (SGB_TC128_BK / 8);                                        \
            int _c = (_i % (SGB_TC128_BK / 8)) * 8;                                  \
            int _gm = pid_m * SGB_TC128_BM + _m;                                     \
            int _gn = (nIdx) + _c;                                            \
            int _elems = sgb_cp_async_valid_elems(_gm < M, N, _gn);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ys + (unsigned)((_m * SGB_TC128_LDA + _c) * 2);         \
            long long _offset = _bytes == 0 ? 0 : (long long)_gm * N + _gn;   \
            const void* _src = sgb_cp_async_source(A, _offset, _bytes);        \
            sgb_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SGB_TC128_BN * (SGB_TC128_BK / 8); _i += 256) {     \
            int _k = _i / (SGB_TC128_BK / 8);                                        \
            int _c = (_i % (SGB_TC128_BK / 8)) * 8;                                  \
            int _gk = pid_n * SGB_TC128_BN + _k;                                     \
            int _gn = (nIdx) + _c;                                            \
            int _elems = sgb_cp_async_valid_elems(_gk < K_out, N, _gn);       \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ws + (unsigned)((_k * SGB_TC128_LDA + _c) * 2);         \
            long long _offset = _bytes == 0 ? 0 : (long long)_gk * N + _gn;   \
            const void* _src = sgb_cp_async_source(B, _offset, _bytes);        \
            sgb_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define SGB_TC128_STAGE_NT_SCALAR(buf, nIdx, TT, FF)                             \
    do {                                                                      \
        TT* _ys = &Ys[buf][0][0];                                             \
        TT* _ws = &Ws[buf][0][0];                                             \
        for (int _i = threadIdx.x; _i < SGB_TC128_BM * SGB_TC128_BK; _i += 256) {           \
            int _m = _i / SGB_TC128_BK;                                              \
            int _c = _i % SGB_TC128_BK;                                              \
            int _gm = pid_m * SGB_TC128_BM + _m;                                     \
            int _gn = (nIdx) + _c;                                            \
            _ys[_m * SGB_TC128_LDA + _c] = (_gm < M && _gn < N)                      \
                                        ? A[(long long)_gm * N + _gn]         \
                                        : FF(0.0f);                           \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SGB_TC128_BN * SGB_TC128_BK; _i += 256) {           \
            int _k = _i / SGB_TC128_BK;                                              \
            int _c = _i % SGB_TC128_BK;                                              \
            int _gk = pid_n * SGB_TC128_BN + _k;                                     \
            int _gn = (nIdx) + _c;                                            \
            _ws[_k * SGB_TC128_LDA + _c] = (_gk < K_out && _gn < N)                  \
                                        ? B[(long long)_gk * N + _gn]         \
                                        : FF(0.0f);                           \
        }                                                                     \
    } while (0)

#define SGB_DEFINE_SGEMM_BI_NT_TC128(SUFFIX, T_ACT, FROM_F, MMA_T)                    \
extern "C" __global__ __launch_bounds__(256, 1)                                \
void sgemm_bi_nt_tc_##SUFFIX(                                                  \
    T_ACT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M, int N, int K_out                                                    \
) {                                                                            \
    extern __shared__ __align__(16) unsigned char sgb_tc_dynsmem[];           \
    T_ACT (*Ys)[SGB_TC128_BM][SGB_TC128_LDA] =                                               \
        reinterpret_cast<T_ACT (*)[SGB_TC128_BM][SGB_TC128_LDA]>(sgb_tc_dynsmem);            \
    T_ACT (*Ws)[SGB_TC128_BN][SGB_TC128_LDA] = reinterpret_cast<T_ACT (*)[SGB_TC128_BN][SGB_TC128_LDA]>(   \
        sgb_tc_dynsmem + 2 * SGB_TC128_BM * SGB_TC128_LDA * (int)sizeof(T_ACT));             \
    int num_pid_n = (K_out + SGB_TC128_BN - 1) / SGB_TC128_BN;                               \
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
    bool fast_stage = sgb_is_aligned_16(A) && sgb_is_aligned_16(B) &&         \
                      ((N & 7) == 0);                                          \
    bool packed_epilogue = sgb_is_aligned_4(C) && ((K_out & 1) == 0);         \
    float acc[4][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 4; fm++)                                             \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++)                                         \
            _Pragma("unroll")                                                  \
            for (int e = 0; e < 4; e++) acc[fm][fn][e] = 0.0f;                 \
    int num_n_tiles = (N + SGB_TC128_BK - 1) / SGB_TC128_BK;                                 \
    if (fast_stage) {                                                          \
        SGB_TC128_STAGE_NT_ASYNC(0, 0);                                           \
    } else {                                                                   \
        SGB_TC128_STAGE_NT_SCALAR(0, 0, T_ACT, FROM_F);                           \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int nt = 0; nt < num_n_tiles; nt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (nt + 1 < num_n_tiles) {                                            \
            if (fast_stage) {                                                  \
                SGB_TC128_STAGE_NT_ASYNC(read_buf ^ 1, (nt + 1) * SGB_TC128_BK);         \
            } else {                                                           \
                SGB_TC128_STAGE_NT_SCALAR(read_buf ^ 1, (nt + 1) * SGB_TC128_BK, T_ACT,  \
                                       FROM_F);                                \
            }                                                                  \
        }                                                                      \
        unsigned Ys_rd = Ys_sbase + (unsigned)(read_buf * SGB_TC128_BM * SGB_TC128_LDA * 2); \
        unsigned Ws_rd = Ws_sbase + (unsigned)(read_buf * SGB_TC128_BN * SGB_TC128_LDA * 2); \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (SGB_TC128_BK / 16); ks++) {                            \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[4][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 4; fm++) {                                   \
                int row = warpM + fm * 16 + lm_row_off + lm_r;                 \
                unsigned addr = Ys_rd +                                        \
                    (unsigned)((row * SGB_TC128_LDA + k0 + lm_col_off) * 2);          \
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
                    (unsigned)((row * SGB_TC128_LDA + k0 + lmb_col_off) * 2);         \
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
            int r0 = pid_m * SGB_TC128_BM + warpM + fm * 16 + g;                     \
            int c0 = pid_n * SGB_TC128_BN + warpN + fn * 8 + 2 * t;                  \
            _Pragma("unroll")                                                 \
            for (int half = 0; half < 2; half++) {                            \
                int gr = r0 + (half ? 8 : 0);                                 \
                if (gr >= M) continue;                                        \
                T_ACT* dst = sgb_output_start_if_valid(                        \
                    C, (long long)gr * K_out, c0, K_out);                      \
                if (dst == nullptr) continue;                                 \
                float v0 = alpha * acc[fm][fn][2 * half];                     \
                float v1 = alpha * acc[fm][fn][2 * half + 1];                 \
                if (packed_epilogue && c0 + 1 < K_out &&                     \
                    sgb_is_aligned_4(dst)) {                                   \
                    sgb_store_pair_rne(dst, v0, v1);                           \
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

SGB_DEFINE_SGEMM_BI_NT_TC128(bf16, __nv_bfloat16, from_f_bf16, "bf16")
SGB_DEFINE_SGEMM_BI_NT_TC128(f16,  __half,        from_f_f16,  "f16")

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
// Tile64/Tile128 routing in gpu/sgemm_bi.rs legal under the strict all-M
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
// RULE (0.4.0 lesson): every constant below is section-local (SGB_TC64_*).
// NEVER reference SGB_TC128_BM/SGB_TC128_BN/SGB_TC128_BK/SGB_TC128_LDA/SGB_TC128_LDB or any other ambient
// define from earlier sections inside this section.

#define SGB_TC64_BM 64
#define SGB_TC64_BN 64
#define SGB_TC64_BK 64
#define SGB_TC64_THREADS 128
#define SGB_TC64_LDA (SGB_TC64_BK + 8) /* 72 halves = 144 B rows, 36 words ≡ 4 mod 8 */
#define SGB_TC64_LDB (SGB_TC64_BN + 8) /* 72 halves = 144 B rows, 36 words ≡ 4 mod 8 */

// NN staging: A 64 rows x 4 chunks + B 32 rows x 8 chunks = 512 cp.async
// over 128 threads (4 per thread), 16B each with zero-fill tails.
#define SGB_TC64_STAGE_ASYNC(buf, bkIdx)                                      \
    do {                                                                      \
        unsigned _as =                                                        \
            As_sbase + (unsigned)((buf) * SGB_TC64_BM * SGB_TC64_LDA * 2);    \
        unsigned _bs =                                                        \
            Bs_sbase + (unsigned)((buf) * SGB_TC64_BK * SGB_TC64_LDB * 2);    \
        for (int _i = threadIdx.x; _i < SGB_TC64_BM * (SGB_TC64_BK / 8);      \
             _i += SGB_TC64_THREADS) {                                        \
            int _m = _i / (SGB_TC64_BK / 8);                                  \
            int _k = (_i % (SGB_TC64_BK / 8)) * 8;                            \
            int _gr = pid_m * SGB_TC64_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            int _elems = sgb_cp_async_valid_elems(_gr < M, K, _gc);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _as + (unsigned)((_m * SGB_TC64_LDA + _k) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gr * lda + _gc; \
            const void* _src = sgb_cp_async_source(A, _offset, _bytes);        \
            sgb_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SGB_TC64_BK * (SGB_TC64_BN / 8);      \
             _i += SGB_TC64_THREADS) {                                        \
            int _k = _i / (SGB_TC64_BN / 8);                                  \
            int _n = (_i % (SGB_TC64_BN / 8)) * 8;                            \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * SGB_TC64_BN + _n;                               \
            int _elems = sgb_cp_async_valid_elems(_gk < K, N, _gn);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _bs + (unsigned)((_k * SGB_TC64_LDB + _n) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gk * ldb + _gn; \
            const void* _src = sgb_cp_async_source(B, _offset, _bytes);        \
            sgb_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

// Scalar staging fallback for misaligned lda/ldb (rare; uniform branch).
#define SGB_TC64_STAGE_SCALAR(buf, bkIdx, TT, FF)                             \
    do {                                                                      \
        TT* _Asw = &As[buf][0][0];                                            \
        TT* _Bsw = &Bs[buf][0][0];                                            \
        for (int _i = threadIdx.x; _i < SGB_TC64_BM * SGB_TC64_BK;            \
             _i += SGB_TC64_THREADS) {                                        \
            int _m = _i / SGB_TC64_BK;                                        \
            int _k = _i % SGB_TC64_BK;                                        \
            int _gr = pid_m * SGB_TC64_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            _Asw[_m * SGB_TC64_LDA + _k] = (_gr < M && _gc < K)               \
                                               ? A[(long long)_gr * lda + _gc]\
                                               : FF(0.0f);                    \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SGB_TC64_BK * SGB_TC64_BN;            \
             _i += SGB_TC64_THREADS) {                                        \
            int _k = _i / SGB_TC64_BN;                                        \
            int _n = _i % SGB_TC64_BN;                                        \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * SGB_TC64_BN + _n;                               \
            _Bsw[_k * SGB_TC64_LDB + _n] = (_gk < K && _gn < N)               \
                                               ? B[(long long)_gk * ldb + _gn]\
                                               : FF(0.0f);                    \
        }                                                                     \
    } while (0)

#define SGB_DEFINE_SGEMM_BI_NN_TC64(SUFFIX, T_ACT, FROM_F, MMA_T)                  \
extern "C" __global__ __launch_bounds__(SGB_TC64_THREADS, 1)                   \
void sgemm_bi_nn_tc64_##SUFFIX(                                                \
    T_ACT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    const float* __restrict__ bias,                                            \
    float alpha, float beta,                                                   \
    int M, int N, int K,                                                       \
    int lda, int ldb, int ldc                                                  \
) {                                                                            \
    assert(alpha == 1.0f || bias == nullptr);                                  \
    __shared__ __align__(16) T_ACT As[2][SGB_TC64_BM][SGB_TC64_LDA];           \
    __shared__ __align__(16) T_ACT Bs[2][SGB_TC64_BK][SGB_TC64_LDB];           \
    int num_pid_n = (N + SGB_TC64_BN - 1) / SGB_TC64_BN;                       \
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
    bool fast_stage = sgb_is_aligned_16(A) && sgb_is_aligned_16(B) &&         \
                      ((lda & 7) == 0) && ((ldb & 7) == 0);                    \
    float acc[2][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            float b0 = 0.0f, b1 = 0.0f;                                        \
            if (bias != nullptr) {                                             \
                int c0 = pid_n * SGB_TC64_BN + warpN + fn * 8 + 2 * t;         \
                b0 = (c0 < N) ? bias[c0] : 0.0f;                               \
                b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                       \
            }                                                                  \
            acc[fm][fn][0] = b0;                                               \
            acc[fm][fn][1] = b1;                                               \
            acc[fm][fn][2] = b0;                                               \
            acc[fm][fn][3] = b1;                                               \
        }                                                                      \
    }                                                                          \
    int num_k_tiles = (K + SGB_TC64_BK - 1) / SGB_TC64_BK;                     \
    if (fast_stage) {                                                          \
        SGB_TC64_STAGE_ASYNC(0, 0);                                            \
    } else {                                                                   \
        SGB_TC64_STAGE_SCALAR(0, 0, T_ACT, FROM_F);                            \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (kt + 1 < num_k_tiles) {                                            \
            if (fast_stage) {                                                  \
                SGB_TC64_STAGE_ASYNC(read_buf ^ 1, (kt + 1) * SGB_TC64_BK);    \
            } else {                                                           \
                SGB_TC64_STAGE_SCALAR(read_buf ^ 1, (kt + 1) * SGB_TC64_BK,    \
                                      T_ACT, FROM_F);                          \
            }                                                                  \
        }                                                                      \
        unsigned As_rd =                                                       \
            As_sbase + (unsigned)(read_buf * SGB_TC64_BM * SGB_TC64_LDA * 2);  \
        unsigned Bs_rd =                                                       \
            Bs_sbase + (unsigned)(read_buf * SGB_TC64_BK * SGB_TC64_LDB * 2);  \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (SGB_TC64_BK / 16); ks++) {                            \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[2][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 2; fm++) {                                   \
                int row = warpM + fm * 16 + lm_row_off + lm_r;                 \
                unsigned addr = As_rd +                                        \
                    (unsigned)((row * SGB_TC64_LDA + k0 + lm_col_off) * 2);    \
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
                    (unsigned)((row * SGB_TC64_LDB + warpN + fn * 8) * 2);     \
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
            int r0 = pid_m * SGB_TC64_BM + warpM + fm * 16 + g;                \
            int c0 = pid_n * SGB_TC64_BN + warpN + fn * 8 + 2 * t;             \
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

SGB_DEFINE_SGEMM_BI_NN_TC64(bf16, __nv_bfloat16, from_f_bf16, "bf16")
SGB_DEFINE_SGEMM_BI_NN_TC64(f16,  __half,        from_f_f16,  "f16")

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
// RULE (0.4.0 lesson): every constant below is section-local (SGB_TC16_*).
#define SGB_TC16_BM 16
#define SGB_TC16_BN 32
#define SGB_TC16_BK 64
#define SGB_TC16_THREADS 128
#define SGB_TC16_STAGES 4
#define SGB_TC16_LDA (SGB_TC16_BK + 8) /* 72 halves = 144 B rows */
#define SGB_TC16_LDB (SGB_TC16_BN + 8) /* 40 halves = 80 B rows, 20 words == 4 mod 8 */

#define SGB_TC16_STAGE_ASYNC(buf, bkIdx)                                      \
    do {                                                                      \
        unsigned _as =                                                        \
            As_sbase + (unsigned)((buf) * SGB_TC16_BM * SGB_TC16_LDA * 2);    \
        unsigned _bs =                                                        \
            Bs_sbase + (unsigned)((buf) * SGB_TC16_BK * SGB_TC16_LDB * 2);    \
        for (int _i = threadIdx.x; _i < SGB_TC16_BM * (SGB_TC16_BK / 8);      \
             _i += SGB_TC16_THREADS) {                                        \
            int _m = _i / (SGB_TC16_BK / 8);                                  \
            int _k = (_i % (SGB_TC16_BK / 8)) * 8;                            \
            int _gr = pid_m * SGB_TC16_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            int _elems = sgb_cp_async_valid_elems(_gr < M, K, _gc);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _as + (unsigned)((_m * SGB_TC16_LDA + _k) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gr * lda + _gc; \
            const void* _src = sgb_cp_async_source(A, _offset, _bytes);        \
            sgb_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SGB_TC16_BK * (SGB_TC16_BN / 8);      \
             _i += SGB_TC16_THREADS) {                                        \
            int _k = _i / (SGB_TC16_BN / 8);                                  \
            int _n = (_i % (SGB_TC16_BN / 8)) * 8;                            \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * SGB_TC16_BN + _n;                               \
            int _elems = sgb_cp_async_valid_elems(_gk < K, N, _gn);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _bs + (unsigned)((_k * SGB_TC16_LDB + _n) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gk * ldb + _gn; \
            const void* _src = sgb_cp_async_source(B, _offset, _bytes);        \
            sgb_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define SGB_TC16_STAGE_SCALAR(buf, bkIdx, TT, FF)                             \
    do {                                                                      \
        TT* _Asw = &As[buf][0][0];                                            \
        TT* _Bsw = &Bs[buf][0][0];                                            \
        for (int _i = threadIdx.x; _i < SGB_TC16_BM * SGB_TC16_BK;            \
             _i += SGB_TC16_THREADS) {                                        \
            int _m = _i / SGB_TC16_BK;                                        \
            int _k = _i % SGB_TC16_BK;                                        \
            int _gr = pid_m * SGB_TC16_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            _Asw[_m * SGB_TC16_LDA + _k] = (_gr < M && _gc < K)               \
                                               ? A[(long long)_gr * lda + _gc]\
                                               : FF(0.0f);                    \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SGB_TC16_BK * SGB_TC16_BN;            \
             _i += SGB_TC16_THREADS) {                                        \
            int _k = _i / SGB_TC16_BN;                                        \
            int _n = _i % SGB_TC16_BN;                                        \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * SGB_TC16_BN + _n;                               \
            _Bsw[_k * SGB_TC16_LDB + _n] = (_gk < K && _gn < N)               \
                                               ? B[(long long)_gk * ldb + _gn]\
                                               : FF(0.0f);                    \
        }                                                                     \
    } while (0)

#define SGB_DEFINE_SGEMM_BI_NN_TC16(SUFFIX, T_ACT, FROM_F, MMA_T)                  \
extern "C" __global__ __launch_bounds__(SGB_TC16_THREADS, 3)                   \
void sgemm_bi_nn_tc16_##SUFFIX(                                                \
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
        T_ACT As[SGB_TC16_STAGES][SGB_TC16_BM][SGB_TC16_LDA];                  \
    __shared__ __align__(16)                                                   \
        T_ACT Bs[SGB_TC16_STAGES][SGB_TC16_BK][SGB_TC16_LDB];                  \
    int num_pid_n = (N + SGB_TC16_BN - 1) / SGB_TC16_BN;                       \
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
    bool fast_stage = sgb_is_aligned_16(A) && sgb_is_aligned_16(B) &&         \
                      ((lda & 7) == 0) && ((ldb & 7) == 0);                    \
    float acc[4];                                                              \
    {                                                                          \
        float b0 = 0.0f, b1 = 0.0f;                                            \
        if (bias != nullptr) {                                                 \
            int c0 = pid_n * SGB_TC16_BN + warpN + 2 * t;                      \
            b0 = (c0 < N) ? bias[c0] : 0.0f;                                   \
            b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                           \
        }                                                                      \
        acc[0] = b0;                                                           \
        acc[1] = b1;                                                           \
        acc[2] = b0;                                                           \
        acc[3] = b1;                                                           \
    }                                                                          \
    int num_k_tiles = (K + SGB_TC16_BK - 1) / SGB_TC16_BK;                     \
    /* Prologue: STAGES-1 commit groups, real or empty - uniform count. */    \
    for (int p = 0; p < SGB_TC16_STAGES - 1; p++) {                            \
        if (p < num_k_tiles) {                                                 \
            if (fast_stage) {                                                  \
                SGB_TC16_STAGE_ASYNC(p, p * SGB_TC16_BK);                      \
            } else {                                                           \
                SGB_TC16_STAGE_SCALAR(p, p * SGB_TC16_BK, T_ACT, FROM_F);      \
                asm volatile("cp.async.commit_group;\n");                     \
            }                                                                  \
        } else {                                                               \
            asm volatile("cp.async.commit_group;\n");                         \
        }                                                                      \
    }                                                                          \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                 \
        asm volatile("cp.async.wait_group %0;\n"                              \
                     :: "n"(SGB_TC16_STAGES - 2));                             \
        __syncthreads();                                                       \
        int next = kt + SGB_TC16_STAGES - 1;                                   \
        if (next < num_k_tiles) {                                              \
            int wbuf = next % SGB_TC16_STAGES;                                 \
            if (fast_stage) {                                                  \
                SGB_TC16_STAGE_ASYNC(wbuf, next * SGB_TC16_BK);                \
            } else {                                                           \
                SGB_TC16_STAGE_SCALAR(wbuf, next * SGB_TC16_BK, T_ACT,         \
                                      FROM_F);                                 \
                asm volatile("cp.async.commit_group;\n");                     \
            }                                                                  \
        } else {                                                               \
            asm volatile("cp.async.commit_group;\n");                         \
        }                                                                      \
        int rbuf = kt % SGB_TC16_STAGES;                                       \
        unsigned As_rd =                                                       \
            As_sbase + (unsigned)(rbuf * SGB_TC16_BM * SGB_TC16_LDA * 2);      \
        unsigned Bs_rd =                                                       \
            Bs_sbase + (unsigned)(rbuf * SGB_TC16_BK * SGB_TC16_LDB * 2);      \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (SGB_TC16_BK / 16); ks++) {                      \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[4];                                                \
            unsigned b_frag[2];                                                \
            {                                                                  \
                int row = lm_row_off + lm_r;                                   \
                unsigned addr = As_rd +                                        \
                    (unsigned)((row * SGB_TC16_LDA + k0 + lm_col_off) * 2);    \
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
                    (unsigned)((row * SGB_TC16_LDB + warpN) * 2);              \
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
        int r0 = pid_m * SGB_TC16_BM + g;                                      \
        int c0 = pid_n * SGB_TC16_BN + warpN + 2 * t;                          \
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

SGB_DEFINE_SGEMM_BI_NN_TC16(bf16, __nv_bfloat16, from_f_bf16, "bf16")
SGB_DEFINE_SGEMM_BI_NN_TC16(f16,  __half,        from_f_f16,  "f16")

// TN (dW) staging: Xs[m_local][k_out chunk], dYs[m_local][n chunk]; both
// rows are the M-reduction dim (SGB_TC64_BK rows per tile, 64-wide rows).
#define SGB_TC64_STAGE_TN_ASYNC(buf, mIdx)                                    \
    do {                                                                      \
        unsigned _xs =                                                        \
            Xs_sbase + (unsigned)((buf) * SGB_TC64_BK * SGB_TC64_LDB * 2);    \
        unsigned _ys =                                                        \
            Ys_sbase + (unsigned)((buf) * SGB_TC64_BK * SGB_TC64_LDB * 2);    \
        for (int _i = threadIdx.x; _i < SGB_TC64_BK * (SGB_TC64_BM / 8);      \
             _i += SGB_TC64_THREADS) {                                        \
            int _r = _i / (SGB_TC64_BM / 8);                                  \
            int _c = (_i % (SGB_TC64_BM / 8)) * 8;                            \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * SGB_TC64_BM + _c;                               \
            int _elems = sgb_cp_async_valid_elems(_gm < M_red, K_out, _gk);   \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _xs + (unsigned)((_r * SGB_TC64_LDB + _c) * 2);   \
            long long _offset =                                                \
                _bytes == 0 ? 0 : (long long)_gm * K_out + _gk;               \
            const void* _src = sgb_cp_async_source(A, _offset, _bytes);        \
            sgb_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SGB_TC64_BK * (SGB_TC64_BN / 8);      \
             _i += SGB_TC64_THREADS) {                                        \
            int _r = _i / (SGB_TC64_BN / 8);                                  \
            int _c = (_i % (SGB_TC64_BN / 8)) * 8;                            \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * SGB_TC64_BN + _c;                               \
            int _elems = sgb_cp_async_valid_elems(_gm < M_red, N, _gn);       \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ys + (unsigned)((_r * SGB_TC64_LDB + _c) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gm * N + _gn;   \
            const void* _src = sgb_cp_async_source(B, _offset, _bytes);        \
            sgb_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define SGB_TC64_STAGE_TN_SCALAR(buf, mIdx, TT, FF)                           \
    do {                                                                      \
        TT* _xs = &Xs[buf][0][0];                                             \
        TT* _ys = &Ys[buf][0][0];                                             \
        for (int _i = threadIdx.x; _i < SGB_TC64_BK * SGB_TC64_BM;            \
             _i += SGB_TC64_THREADS) {                                        \
            int _r = _i / SGB_TC64_BM;                                        \
            int _c = _i % SGB_TC64_BM;                                        \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * SGB_TC64_BM + _c;                               \
            _xs[_r * SGB_TC64_LDB + _c] = (_gm < M_red && _gk < K_out)        \
                                              ? A[(long long)_gm * K_out + _gk]\
                                              : FF(0.0f);                     \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SGB_TC64_BK * SGB_TC64_BN;            \
             _i += SGB_TC64_THREADS) {                                        \
            int _r = _i / SGB_TC64_BN;                                        \
            int _c = _i % SGB_TC64_BN;                                        \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * SGB_TC64_BN + _c;                               \
            _ys[_r * SGB_TC64_LDB + _c] = (_gm < M_red && _gn < N)            \
                                              ? B[(long long)_gm * N + _gn]   \
                                              : FF(0.0f);                     \
        }                                                                     \
    } while (0)

#define SGB_DEFINE_SGEMM_BI_TN_TC64(SUFFIX, T_ACT, FROM_F, MMA_T)                  \
extern "C" __global__ __launch_bounds__(SGB_TC64_THREADS, 1)                   \
void sgemm_bi_tn_tc64_##SUFFIX(                                                \
    float* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M_red, int K_out, int N                                                \
) {                                                                            \
    __shared__ __align__(16) T_ACT Xs[2][SGB_TC64_BK][SGB_TC64_LDB];           \
    __shared__ __align__(16) T_ACT Ys[2][SGB_TC64_BK][SGB_TC64_LDB];           \
    int num_pid_n = (N + SGB_TC64_BN - 1) / SGB_TC64_BN;                       \
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
    bool fast_stage = sgb_is_aligned_16(A) && sgb_is_aligned_16(B) &&         \
                      ((K_out & 7) == 0) && ((N & 7) == 0);                    \
    float acc[2][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++)                                             \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++)                                         \
            _Pragma("unroll")                                                  \
            for (int e = 0; e < 4; e++) acc[fm][fn][e] = 0.0f;                 \
    int num_m_tiles = (M_red + SGB_TC64_BK - 1) / SGB_TC64_BK;                 \
    if (fast_stage) {                                                          \
        SGB_TC64_STAGE_TN_ASYNC(0, 0);                                         \
    } else {                                                                   \
        SGB_TC64_STAGE_TN_SCALAR(0, 0, T_ACT, FROM_F);                         \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int mt = 0; mt < num_m_tiles; mt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (mt + 1 < num_m_tiles) {                                            \
            if (fast_stage) {                                                  \
                SGB_TC64_STAGE_TN_ASYNC(read_buf ^ 1, (mt + 1) * SGB_TC64_BK); \
            } else {                                                           \
                SGB_TC64_STAGE_TN_SCALAR(read_buf ^ 1, (mt + 1) * SGB_TC64_BK, \
                                         T_ACT, FROM_F);                       \
            }                                                                  \
        }                                                                      \
        unsigned Xs_rd =                                                       \
            Xs_sbase + (unsigned)(read_buf * SGB_TC64_BK * SGB_TC64_LDB * 2);  \
        unsigned Ys_rd =                                                       \
            Ys_sbase + (unsigned)(read_buf * SGB_TC64_BK * SGB_TC64_LDB * 2);  \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (SGB_TC64_BK / 16); ks++) {                            \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[2][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 2; fm++) {                                   \
                int srow = k0 + lm_arow_off + lm_r;                            \
                int scol = warpM + fm * 16 + lm_acol_off;                      \
                unsigned addr =                                                \
                    Xs_rd + (unsigned)((srow * SGB_TC64_LDB + scol) * 2);      \
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
                    (unsigned)((srow * SGB_TC64_LDB + warpN + fn * 8) * 2);    \
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
            int r0 = pid_m * SGB_TC64_BM + warpM + fm * 16 + g;                \
            int c0 = pid_n * SGB_TC64_BN + warpN + fn * 8 + 2 * t;             \
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

SGB_DEFINE_SGEMM_BI_TN_TC64(bf16, __nv_bfloat16, from_f_bf16, "bf16")
SGB_DEFINE_SGEMM_BI_TN_TC64(f16,  __half,        from_f_f16,  "f16")

// NT (dX) staging: dYs[m_local][n chunk] (output rows x reduction) and
// Ws[k_out_local][n chunk] (output cols x reduction); rows are BK (32) wide.
#define SGB_TC64_STAGE_NT_ASYNC(buf, nIdx)                                    \
    do {                                                                      \
        unsigned _ys =                                                        \
            Ys_sbase + (unsigned)((buf) * SGB_TC64_BM * SGB_TC64_LDA * 2);    \
        unsigned _ws =                                                        \
            Ws_sbase + (unsigned)((buf) * SGB_TC64_BN * SGB_TC64_LDA * 2);    \
        for (int _i = threadIdx.x; _i < SGB_TC64_BM * (SGB_TC64_BK / 8);      \
             _i += SGB_TC64_THREADS) {                                        \
            int _m = _i / (SGB_TC64_BK / 8);                                  \
            int _c = (_i % (SGB_TC64_BK / 8)) * 8;                            \
            int _gm = pid_m * SGB_TC64_BM + _m;                               \
            int _gn = (nIdx) + _c;                                            \
            int _elems = sgb_cp_async_valid_elems(_gm < M, N, _gn);           \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ys + (unsigned)((_m * SGB_TC64_LDA + _c) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gm * N + _gn;   \
            const void* _src = sgb_cp_async_source(A, _offset, _bytes);        \
            sgb_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SGB_TC64_BN * (SGB_TC64_BK / 8);      \
             _i += SGB_TC64_THREADS) {                                        \
            int _k = _i / (SGB_TC64_BK / 8);                                  \
            int _c = (_i % (SGB_TC64_BK / 8)) * 8;                            \
            int _gk = pid_n * SGB_TC64_BN + _k;                               \
            int _gn = (nIdx) + _c;                                            \
            int _elems = sgb_cp_async_valid_elems(_gk < K_out, N, _gn);       \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ws + (unsigned)((_k * SGB_TC64_LDA + _c) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gk * N + _gn;   \
            const void* _src = sgb_cp_async_source(B, _offset, _bytes);        \
            sgb_cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define SGB_TC64_STAGE_NT_SCALAR(buf, nIdx, TT, FF)                           \
    do {                                                                      \
        TT* _ys = &Ys[buf][0][0];                                             \
        TT* _ws = &Ws[buf][0][0];                                             \
        for (int _i = threadIdx.x; _i < SGB_TC64_BM * SGB_TC64_BK;            \
             _i += SGB_TC64_THREADS) {                                        \
            int _m = _i / SGB_TC64_BK;                                        \
            int _c = _i % SGB_TC64_BK;                                        \
            int _gm = pid_m * SGB_TC64_BM + _m;                               \
            int _gn = (nIdx) + _c;                                            \
            _ys[_m * SGB_TC64_LDA + _c] = (_gm < M && _gn < N)                \
                                              ? A[(long long)_gm * N + _gn]   \
                                              : FF(0.0f);                     \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SGB_TC64_BN * SGB_TC64_BK;            \
             _i += SGB_TC64_THREADS) {                                        \
            int _k = _i / SGB_TC64_BK;                                        \
            int _c = _i % SGB_TC64_BK;                                        \
            int _gk = pid_n * SGB_TC64_BN + _k;                               \
            int _gn = (nIdx) + _c;                                            \
            _ws[_k * SGB_TC64_LDA + _c] = (_gk < K_out && _gn < N)            \
                                              ? B[(long long)_gk * N + _gn]   \
                                              : FF(0.0f);                     \
        }                                                                     \
    } while (0)

#define SGB_DEFINE_SGEMM_BI_NT_TC64(SUFFIX, T_ACT, FROM_F, MMA_T)                  \
extern "C" __global__ __launch_bounds__(SGB_TC64_THREADS, 1)                   \
void sgemm_bi_nt_tc64_##SUFFIX(                                                \
    T_ACT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M, int N, int K_out                                                    \
) {                                                                            \
    __shared__ __align__(16) T_ACT Ys[2][SGB_TC64_BM][SGB_TC64_LDA];           \
    __shared__ __align__(16) T_ACT Ws[2][SGB_TC64_BN][SGB_TC64_LDA];           \
    int num_pid_n = (K_out + SGB_TC64_BN - 1) / SGB_TC64_BN;                   \
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
    bool fast_stage = sgb_is_aligned_16(A) && sgb_is_aligned_16(B) &&         \
                      ((N & 7) == 0);                                          \
    float acc[2][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++)                                             \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++)                                         \
            _Pragma("unroll")                                                  \
            for (int e = 0; e < 4; e++) acc[fm][fn][e] = 0.0f;                 \
    int num_n_tiles = (N + SGB_TC64_BK - 1) / SGB_TC64_BK;                     \
    if (fast_stage) {                                                          \
        SGB_TC64_STAGE_NT_ASYNC(0, 0);                                         \
    } else {                                                                   \
        SGB_TC64_STAGE_NT_SCALAR(0, 0, T_ACT, FROM_F);                         \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int nt = 0; nt < num_n_tiles; nt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (nt + 1 < num_n_tiles) {                                            \
            if (fast_stage) {                                                  \
                SGB_TC64_STAGE_NT_ASYNC(read_buf ^ 1, (nt + 1) * SGB_TC64_BK); \
            } else {                                                           \
                SGB_TC64_STAGE_NT_SCALAR(read_buf ^ 1, (nt + 1) * SGB_TC64_BK, \
                                         T_ACT, FROM_F);                       \
            }                                                                  \
        }                                                                      \
        unsigned Ys_rd =                                                       \
            Ys_sbase + (unsigned)(read_buf * SGB_TC64_BM * SGB_TC64_LDA * 2);  \
        unsigned Ws_rd =                                                       \
            Ws_sbase + (unsigned)(read_buf * SGB_TC64_BN * SGB_TC64_LDA * 2);  \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (SGB_TC64_BK / 16); ks++) {                            \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[2][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 2; fm++) {                                   \
                int row = warpM + fm * 16 + lm_row_off + lm_r;                 \
                unsigned addr = Ys_rd +                                        \
                    (unsigned)((row * SGB_TC64_LDA + k0 + lm_col_off) * 2);    \
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
                    (unsigned)((row * SGB_TC64_LDA + k0 + lmb_col_off) * 2);   \
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
            int r0 = pid_m * SGB_TC64_BM + warpM + fm * 16 + g;                \
            int c0 = pid_n * SGB_TC64_BN + warpN + fn * 8 + 2 * t;             \
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

SGB_DEFINE_SGEMM_BI_NT_TC64(bf16, __nv_bfloat16, from_f_bf16, "bf16")
SGB_DEFINE_SGEMM_BI_NT_TC64(f16,  __half,        from_f_f16,  "f16")

#undef SGB_DEFINE_SGEMM_BI_NN_TC128
#undef SGB_DEFINE_SGEMM_BI_NN_TC16
#undef SGB_DEFINE_SGEMM_BI_NN_TC64
#undef SGB_DEFINE_SGEMM_BI_NT_TC128
#undef SGB_DEFINE_SGEMM_BI_NT_TC64
#undef SGB_DEFINE_SGEMM_BI_TN_TC128
#undef SGB_DEFINE_SGEMM_BI_TN_TC64
#undef SGB_TC128_BK
#undef SGB_TC128_BM
#undef SGB_TC128_BN
#undef SGB_TC128_LDA
#undef SGB_TC128_LDB
#undef SGB_TC128_PAD_A
#undef SGB_TC128_PAD_B
#undef SGB_TC128_STAGE_ASYNC
#undef SGB_TC128_STAGE_NT_ASYNC
#undef SGB_TC128_STAGE_NT_SCALAR
#undef SGB_TC128_STAGE_SCALAR
#undef SGB_TC128_STAGE_TN_ASYNC
#undef SGB_TC128_STAGE_TN_SCALAR
#undef SGB_TC16_BK
#undef SGB_TC16_BM
#undef SGB_TC16_BN
#undef SGB_TC16_LDA
#undef SGB_TC16_LDB
#undef SGB_TC16_STAGES
#undef SGB_TC16_STAGE_ASYNC
#undef SGB_TC16_STAGE_SCALAR
#undef SGB_TC16_THREADS
#undef SGB_TC64_BK
#undef SGB_TC64_BM
#undef SGB_TC64_BN
#undef SGB_TC64_LDA
#undef SGB_TC64_LDB
#undef SGB_TC64_STAGE_ASYNC
#undef SGB_TC64_STAGE_NT_ASYNC
#undef SGB_TC64_STAGE_NT_SCALAR
#undef SGB_TC64_STAGE_SCALAR
#undef SGB_TC64_STAGE_TN_ASYNC
#undef SGB_TC64_STAGE_TN_SCALAR
#undef SGB_TC64_THREADS
