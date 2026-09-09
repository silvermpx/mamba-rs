/*
 * Frozen SM89 half-TN retained families.
 *
 * These two units are byte-derived from the qualified discovery adapters.
 * Only export prefixes are production-normalized. Keep cleanup outside the
 * marked bodies so normalized parity remains exact and the units can compose.
 */

// SM89_HALF_TN_COMPACT_BEGIN
#define GEMM_BI_TC64_BM 64
#define GEMM_BI_TC64_BN 64
#define GEMM_BI_TC64_BK 64
#define GEMM_BI_TC64_THREADS 128
#define GEMM_BI_TC64_LDB 64
#define GEMM_BI_HALF_TN_INDEX(row, col) ((row) * 64 + ((col) ^ (((row) & 7) * 8)))
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
            unsigned _dst = _xs + (unsigned)((GEMM_BI_HALF_TN_INDEX(_r, _c)) * 2);   \
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
            unsigned _dst = _ys + (unsigned)((GEMM_BI_HALF_TN_INDEX(_r, _c)) * 2);   \
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
            _xs[GEMM_BI_HALF_TN_INDEX(_r, _c)] = (_gm < M_red && _gk < K_out)        \
                                              ? A[(long long)_gm * K_out + _gk]\
                                              : FF(0.0f);                     \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BK * GEMM_BI_TC64_BN;            \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _r = _i / GEMM_BI_TC64_BN;                                        \
            int _c = _i % GEMM_BI_TC64_BN;                                        \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_TC64_BN + _c;                               \
            _ys[GEMM_BI_HALF_TN_INDEX(_r, _c)] = (_gm < M_red && _gn < N)            \
                                              ? B[(long long)_gm * N + _gn]   \
                                              : FF(0.0f);                     \
        }                                                                     \
    } while (0)

#define GEMM_BI_DEFINE_GEMM_BI_TN_TC64(SUFFIX, T_ACT, FROM_F, MMA_T)                  \
extern "C" __global__ __launch_bounds__(GEMM_BI_TC64_THREADS, 1)                   \
void gemm_bi_tn_sm89_m64n64_bk64_s2_compact_bxor_v1_##SUFFIX(                                                \
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
                    Xs_rd + (unsigned)((GEMM_BI_HALF_TN_INDEX(srow, scol)) * 2);   \
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
                    (unsigned)((GEMM_BI_HALF_TN_INDEX(srow, warpN + fn * 8)) * 2);\
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


#undef GEMM_BI_HALF_TN_INDEX
// SM89_HALF_TN_COMPACT_END
#undef GEMM_BI_DEFINE_GEMM_BI_TN_TC64
#undef GEMM_BI_TC64_STAGE_TN_SCALAR
#undef GEMM_BI_TC64_STAGE_TN_ASYNC
#undef GEMM_BI_TC64_LDB
#undef GEMM_BI_TC64_THREADS
#undef GEMM_BI_TC64_BK
#undef GEMM_BI_TC64_BN
#undef GEMM_BI_TC64_BM

// SM89_HALF_TN_REGPIPE_VEC2_BEGIN
#define GEMM_BI_TC64_BM 64
#define GEMM_BI_TC64_BN 64
#define GEMM_BI_TC64_BK 64
#define GEMM_BI_TC64_THREADS 128
#define GEMM_BI_TC64_LDB 64
#define GEMM_BI_HALF_TN_INDEX(row, col) ((row) * 64 + ((col) ^ (((row) & 7) * 8)))
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
            unsigned _dst = _xs + (unsigned)((GEMM_BI_HALF_TN_INDEX(_r, _c)) * 2);   \
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
            unsigned _dst = _ys + (unsigned)((GEMM_BI_HALF_TN_INDEX(_r, _c)) * 2);   \
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
            _xs[GEMM_BI_HALF_TN_INDEX(_r, _c)] = (_gm < M_red && _gk < K_out)        \
                                              ? A[(long long)_gm * K_out + _gk]\
                                              : FF(0.0f);                     \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BK * GEMM_BI_TC64_BN;            \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _r = _i / GEMM_BI_TC64_BN;                                        \
            int _c = _i % GEMM_BI_TC64_BN;                                        \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_TC64_BN + _c;                               \
            _ys[GEMM_BI_HALF_TN_INDEX(_r, _c)] = (_gm < M_red && _gn < N)            \
                                              ? B[(long long)_gm * N + _gn]   \
                                              : FF(0.0f);                     \
        }                                                                     \
    } while (0)

#define GEMM_BI_DEFINE_GEMM_BI_TN_TC64(SUFFIX, T_ACT, FROM_F, MMA_T)                  \
extern "C" __global__ __launch_bounds__(GEMM_BI_TC64_THREADS, 1)                   \
void gemm_bi_tn_sm89_m64n64_bk64_s2_regpipe_vec2_v1_##SUFFIX(                                                \
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
        unsigned a_frag[2][2][4]; \
        unsigned b_frag[2][4][2]; \
        { \
            int k0 = 0; \
            _Pragma("unroll") \
            for (int fm = 0; fm < 2; fm++) { \
                int srow = k0 + lm_arow_off + lm_r; \
                int scol = warpM + fm * 16 + lm_acol_off; \
                unsigned addr = \
                    Xs_rd + (unsigned)((GEMM_BI_HALF_TN_INDEX(srow, scol)) * 2); \
                asm volatile( \
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 " \
                    "{%0,%1,%2,%3}, [%4];\n" \
                    : "=r"(a_frag[0][fm][0]), "=r"(a_frag[0][fm][1]), \
                      "=r"(a_frag[0][fm][2]), "=r"(a_frag[0][fm][3]) \
                    : "r"(addr)); \
            } \
            _Pragma("unroll") \
            for (int fn = 0; fn < 4; fn++) { \
                int srow = k0 + lm_brow_off + lm_r; \
                unsigned addr = Ys_rd + \
                    (unsigned)((GEMM_BI_HALF_TN_INDEX(srow, warpN + fn * 8)) * 2); \
                asm volatile( \
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 " \
                    "{%0,%1}, [%2];\n" \
                    : "=r"(b_frag[0][fn][0]), "=r"(b_frag[0][fn][1]) \
                    : "r"(addr)); \
            } \
        } \
        _Pragma("unroll") \
        for (int ks = 0; ks < 4; ++ks) { \
            if (ks + 1 < 4) { \
                int k0 = (ks + 1) * 16; \
            _Pragma("unroll") \
            for (int fm = 0; fm < 2; fm++) { \
                int srow = k0 + lm_arow_off + lm_r; \
                int scol = warpM + fm * 16 + lm_acol_off; \
                unsigned addr = \
                    Xs_rd + (unsigned)((GEMM_BI_HALF_TN_INDEX(srow, scol)) * 2); \
                asm volatile( \
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 " \
                    "{%0,%1,%2,%3}, [%4];\n" \
                    : "=r"(a_frag[(ks + 1) & 1][fm][0]), "=r"(a_frag[(ks + 1) & 1][fm][1]), \
                      "=r"(a_frag[(ks + 1) & 1][fm][2]), "=r"(a_frag[(ks + 1) & 1][fm][3]) \
                    : "r"(addr)); \
            } \
            _Pragma("unroll") \
            for (int fn = 0; fn < 4; fn++) { \
                int srow = k0 + lm_brow_off + lm_r; \
                unsigned addr = Ys_rd + \
                    (unsigned)((GEMM_BI_HALF_TN_INDEX(srow, warpN + fn * 8)) * 2); \
                asm volatile( \
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 " \
                    "{%0,%1}, [%2];\n" \
                    : "=r"(b_frag[(ks + 1) & 1][fn][0]), "=r"(b_frag[(ks + 1) & 1][fn][1]) \
                    : "r"(addr)); \
            } \
            } \
            _Pragma("unroll") \
            for (int fm = 0; fm < 2; fm++) { \
                _Pragma("unroll") \
                for (int fn = 0; fn < 4; fn++) { \
                    asm volatile( \
                        "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "." \
                        MMA_T ".f32 " \
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, " \
                        "{%0,%1,%2,%3};\n" \
                        : "+f"(acc[fm][fn][0]), "+f"(acc[fm][fn][1]), \
                          "+f"(acc[fm][fn][2]), "+f"(acc[fm][fn][3]) \
                        : "r"(a_frag[ks & 1][fm][0]), "r"(a_frag[ks & 1][fm][1]), \
                          "r"(a_frag[ks & 1][fm][2]), "r"(a_frag[ks & 1][fm][3]), \
                          "r"(b_frag[ks & 1][fn][0]), "r"(b_frag[ks & 1][fn][1])); \
                } \
            } \
        } \
        read_buf ^= 1;                                                         \
    }                                                                          \
    /* epilogue: paired f32 accumulate into dW, scalar at the N tail */         \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            int r0 = pid_m * GEMM_BI_TC64_BM + warpM + fm * 16 + g;                \
            int c0 = pid_n * GEMM_BI_TC64_BN + warpN + fn * 8 + 2 * t;             \
            _Pragma("unroll")                                                  \
            for (int half = 0; half < 2; ++half) {                             \
                int gr = r0 + half * 8;                                        \
                int gc = c0;                                                   \
                if (gr >= K_out || gc >= N) continue;                          \
                float* dst = C + (long long)gr * N + gc;                       \
                bool packed = gc + 1 < N && (((unsigned long long)dst & 7ull) == 0); \
                if (packed) {                                                  \
                    gemm_bi_accumulate_float2_or_scalar(                       \
                        dst, alpha * acc[fm][fn][2 * half],                    \
                        alpha * acc[fm][fn][2 * half + 1], true);              \
                } else {                                                       \
                    dst[0] += alpha * acc[fm][fn][2 * half];                   \
                    if (gc + 1 < N)                                            \
                        dst[1] += alpha * acc[fm][fn][2 * half + 1];           \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

GEMM_BI_DEFINE_GEMM_BI_TN_TC64(bf16, __nv_bfloat16, from_f_bf16, "bf16")
GEMM_BI_DEFINE_GEMM_BI_TN_TC64(f16,  __half,        from_f_f16,  "f16")


#undef GEMM_BI_HALF_TN_INDEX
// SM89_HALF_TN_REGPIPE_VEC2_END
#undef GEMM_BI_DEFINE_GEMM_BI_TN_TC64
#undef GEMM_BI_TC64_STAGE_TN_SCALAR
#undef GEMM_BI_TC64_STAGE_TN_ASYNC
#undef GEMM_BI_TC64_LDB
#undef GEMM_BI_TC64_THREADS
#undef GEMM_BI_TC64_BK
#undef GEMM_BI_TC64_BN
#undef GEMM_BI_TC64_BM

