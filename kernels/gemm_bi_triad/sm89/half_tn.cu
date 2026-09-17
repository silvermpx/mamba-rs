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
            int _elems = cp_async_valid_elems(_gm < M_red, K_out, _gk);   \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _xs + (unsigned)((GEMM_BI_HALF_TN_INDEX(_r, _c)) * 2);   \
            long long _offset =                                                \
                _bytes == 0 ? 0 : (long long)_gm * K_out + _gk;               \
            const void* _src = cp_async_source(A, _offset, _bytes);        \
            cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BK * (GEMM_BI_TC64_BN / 8);      \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _r = _i / (GEMM_BI_TC64_BN / 8);                                  \
            int _c = (_i % (GEMM_BI_TC64_BN / 8)) * 8;                            \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_TC64_BN + _c;                               \
            int _elems = cp_async_valid_elems(_gm < M_red, N, _gn);       \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ys + (unsigned)((GEMM_BI_HALF_TN_INDEX(_r, _c)) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gm * N + _gn;   \
            const void* _src = cp_async_source(B, _offset, _bytes);        \
            cp_async_16_zfill(_dst, _src, _bytes);                         \
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
void tn_sm89_m64n64_bk64_s2_compact_bxor_##SUFFIX(                                                \
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
    bool fast_stage = is_aligned_16(A) && is_aligned_16(B) &&         \
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
            int _elems = cp_async_valid_elems(_gm < M_red, K_out, _gk);   \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _xs + (unsigned)((GEMM_BI_HALF_TN_INDEX(_r, _c)) * 2);   \
            long long _offset =                                                \
                _bytes == 0 ? 0 : (long long)_gm * K_out + _gk;               \
            const void* _src = cp_async_source(A, _offset, _bytes);        \
            cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BK * (GEMM_BI_TC64_BN / 8);      \
             _i += GEMM_BI_TC64_THREADS) {                                        \
            int _r = _i / (GEMM_BI_TC64_BN / 8);                                  \
            int _c = (_i % (GEMM_BI_TC64_BN / 8)) * 8;                            \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_TC64_BN + _c;                               \
            int _elems = cp_async_valid_elems(_gm < M_red, N, _gn);       \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ys + (unsigned)((GEMM_BI_HALF_TN_INDEX(_r, _c)) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gm * N + _gn;   \
            const void* _src = cp_async_source(B, _offset, _bytes);        \
            cp_async_16_zfill(_dst, _src, _bytes);                         \
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
void tn_sm89_m64n64_bk64_s2_regpipe_vec2_##SUFFIX(                                                \
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
    bool fast_stage = is_aligned_16(A) && is_aligned_16(B) &&         \
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
                    accumulate_float2_or_scalar(                       \
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

// SM89_HALF_TN_SMALL16_BEGIN
#define GEMM_BI_SMALL16_BK 64
#define GEMM_BI_SMALL16_STAGES 2
#define GEMM_BI_SMALL16_LDB 72
#define GEMM_BI_SMALL16_THREADS 32

__device__ __forceinline__ int small16_valid_elems(bool row_valid, int extent, int start) {
    if (!row_valid || start >= extent) return 0;
    int remaining = extent - start;
    return remaining < 8 ? remaining : 8;
}

template <typename T>
__device__ __forceinline__ const T* small16_async_source(
    const T* base, long long valid_offset, int valid_bytes) {
    return valid_bytes == 0 ? base : base + valid_offset;
}

__device__ __forceinline__ void small16_async_zfill(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
}

__device__ __forceinline__ __nv_bfloat16 small16_from_bf16(float value) {
    return __float2bfloat16_rn(value);
}

__device__ __forceinline__ __half small16_from_f16(float value) {
    return __float2half_rn(value);
}

#define GEMM_BI_SMALL16_STAGE_ASYNC(buf, m_idx, BM, BN)                                      \
    do {                                                                            \
        unsigned xs = Xs_base +                                                     \
            (unsigned)((buf) * GEMM_BI_SMALL16_BK * GEMM_BI_SMALL16_LDB * 2);                          \
        unsigned ys = Ys_base +                                                     \
            (unsigned)((buf) * GEMM_BI_SMALL16_BK * GEMM_BI_SMALL16_LDB * 2);                          \
        for (int i = threadIdx.x; i < GEMM_BI_SMALL16_BK * ((BM) / 8);                       \
             i += GEMM_BI_SMALL16_THREADS) {                                                  \
            int row = i / ((BM) / 8);                                               \
            int column = (i % ((BM) / 8)) * 8;                                     \
            int global_m = (m_idx) + row;                                           \
            int global_k = pid_m * (BM) + column;                                  \
            int elems = small16_valid_elems(global_m < M_red, K_out, global_k);       \
            int bytes = elems * 2;                                                  \
            unsigned dst = xs + (unsigned)((row * GEMM_BI_SMALL16_LDB + column) * 2);         \
            long long offset = bytes == 0 ? 0 :                                    \
                (long long)global_m * K_out + global_k;                             \
            const void* src = small16_async_source(A, offset, bytes);                 \
            small16_async_zfill(dst, src, bytes);                                     \
        }                                                                           \
        for (int i = threadIdx.x; i < GEMM_BI_SMALL16_BK * ((BN) / 8);                       \
             i += GEMM_BI_SMALL16_THREADS) {                                                  \
            int row = i / ((BN) / 8);                                               \
            int column = (i % ((BN) / 8)) * 8;                                     \
            int global_m = (m_idx) + row;                                           \
            int global_n = pid_n * (BN) + column;                                  \
            int elems = small16_valid_elems(global_m < M_red, N, global_n);           \
            int bytes = elems * 2;                                                  \
            unsigned dst = ys + (unsigned)((row * GEMM_BI_SMALL16_LDB + column) * 2);         \
            long long offset = bytes == 0 ? 0 :                                    \
                (long long)global_m * N + global_n;                                 \
            const void* src = small16_async_source(B, offset, bytes);                 \
            small16_async_zfill(dst, src, bytes);                                     \
        }                                                                           \
        asm volatile("cp.async.commit_group;\n");                                 \
    } while (0)

#define GEMM_BI_SMALL16_STAGE_SCALAR(buf, m_idx, T_ACT, FROM_F, BM, BN)                      \
    do {                                                                            \
        T_ACT* xs = &Xs[(buf)][0][0];                                               \
        T_ACT* ys = &Ys[(buf)][0][0];                                               \
        for (int i = threadIdx.x; i < GEMM_BI_SMALL16_BK * (BM); i += GEMM_BI_SMALL16_THREADS) {       \
            int row = i / (BM);                                                     \
            int column = i % (BM);                                                  \
            int global_m = (m_idx) + row;                                           \
            int global_k = pid_m * (BM) + column;                                  \
            xs[row * GEMM_BI_SMALL16_LDB + column] =                                         \
                global_m < M_red && global_k < K_out                               \
                    ? A[(long long)global_m * K_out + global_k]                     \
                    : FROM_F(0.0f);                                                 \
        }                                                                           \
        for (int i = threadIdx.x; i < GEMM_BI_SMALL16_BK * (BN); i += GEMM_BI_SMALL16_THREADS) {       \
            int row = i / (BN);                                                     \
            int column = i % (BN);                                                  \
            int global_m = (m_idx) + row;                                           \
            int global_n = pid_n * (BN) + column;                                  \
            ys[row * GEMM_BI_SMALL16_LDB + column] =                                         \
                global_m < M_red && global_n < N                                   \
                    ? B[(long long)global_m * N + global_n]                         \
                    : FROM_F(0.0f);                                                 \
        }                                                                           \
    } while (0)

#define DEFINE_GEMM_BI_SMALL16_TN(SYMBOL, T_ACT, FROM_F, MMA_T, BM, BN, FM_COUNT, FN_COUNT)  \
extern "C" __global__ __launch_bounds__(GEMM_BI_SMALL16_THREADS, 1)                         \
void SYMBOL(float* __restrict__ C, const T_ACT* __restrict__ A,                    \
            const T_ACT* __restrict__ B, float alpha,                              \
            int M_red, int K_out, int N) {                                         \
    __shared__ __align__(16) T_ACT Xs[GEMM_BI_SMALL16_STAGES][GEMM_BI_SMALL16_BK][GEMM_BI_SMALL16_LDB];          \
    __shared__ __align__(16) T_ACT Ys[GEMM_BI_SMALL16_STAGES][GEMM_BI_SMALL16_BK][GEMM_BI_SMALL16_LDB];          \
    int num_pid_n = (N + (BN) - 1) / (BN);                                         \
    int pid_m = blockIdx.x / num_pid_n;                                             \
    int pid_n = blockIdx.x % num_pid_n;                                             \
    int lane = threadIdx.x;                                                         \
    int group = lane >> 2;                                                          \
    int thread = lane & 3;                                                          \
    int matrix_row = lane & 7;                                                      \
    int matrix_quad = lane >> 3;                                                    \
    int a_row_offset = (matrix_quad & 2) ? 8 : 0;                                  \
    int a_column_offset = (matrix_quad & 1) ? 8 : 0;                               \
    int b_row_offset = (matrix_quad & 1) ? 8 : 0;                                  \
    unsigned Xs_base = (unsigned)__cvta_generic_to_shared(&Xs[0][0][0]);            \
    unsigned Ys_base = (unsigned)__cvta_generic_to_shared(&Ys[0][0][0]);            \
    bool fast_stage = (((unsigned long long)A & 15ULL) == 0ULL) &&                  \
                      (((unsigned long long)B & 15ULL) == 0ULL) &&                  \
                      ((K_out & 7) == 0) && ((N & 7) == 0);                         \
    float acc[FM_COUNT][FN_COUNT][4];                                               \
    _Pragma("unroll")                                                             \
    for (int fm = 0; fm < (FM_COUNT); fm++)                                        \
        _Pragma("unroll")                                                         \
        for (int fn = 0; fn < (FN_COUNT); fn++)                                    \
            _Pragma("unroll")                                                     \
            for (int e = 0; e < 4; e++) acc[fm][fn][e] = 0.0f;                    \
    int num_m_tiles = (M_red + GEMM_BI_SMALL16_BK - 1) / GEMM_BI_SMALL16_BK;                          \
    if (fast_stage) {                                                               \
        GEMM_BI_SMALL16_STAGE_ASYNC(0, 0, BM, BN);                                            \
    } else {                                                                        \
        GEMM_BI_SMALL16_STAGE_SCALAR(0, 0, T_ACT, FROM_F, BM, BN);                           \
    }                                                                               \
    int read_buf = 0;                                                               \
    for (int mt = 0; mt < num_m_tiles; mt++) {                                     \
        if (fast_stage) asm volatile("cp.async.wait_group 0;\n");                 \
        __syncthreads();                                                            \
        if (mt + 1 < num_m_tiles) {                                                 \
            if (fast_stage) {                                                       \
                GEMM_BI_SMALL16_STAGE_ASYNC(read_buf ^ 1, (mt + 1) * GEMM_BI_SMALL16_BK, BM, BN);       \
            } else {                                                                \
                GEMM_BI_SMALL16_STAGE_SCALAR(read_buf ^ 1, (mt + 1) * GEMM_BI_SMALL16_BK,              \
                                   T_ACT, FROM_F, BM, BN);                          \
            }                                                                       \
        }                                                                           \
        unsigned Xs_read = Xs_base +                                                \
            (unsigned)(read_buf * GEMM_BI_SMALL16_BK * GEMM_BI_SMALL16_LDB * 2);                       \
        unsigned Ys_read = Ys_base +                                                \
            (unsigned)(read_buf * GEMM_BI_SMALL16_BK * GEMM_BI_SMALL16_LDB * 2);                       \
        _Pragma("unroll")                                                         \
        for (int ks = 0; ks < (GEMM_BI_SMALL16_BK / 16); ks++) {                             \
            int k0 = ks * 16;                                                       \
            unsigned a_frag[FM_COUNT][4];                                          \
            unsigned b_frag[FN_COUNT][2];                                          \
            _Pragma("unroll")                                                     \
            for (int fm = 0; fm < (FM_COUNT); fm++) {                              \
                int row = k0 + a_row_offset + matrix_row;                          \
                int column = fm * 16 + a_column_offset;                            \
                unsigned addr = Xs_read +                                          \
                    (unsigned)((row * GEMM_BI_SMALL16_LDB + column) * 2);                     \
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "     \
                             "{%0,%1,%2,%3}, [%4];\n"                            \
                             : "=r"(a_frag[fm][0]), "=r"(a_frag[fm][1]),         \
                               "=r"(a_frag[fm][2]), "=r"(a_frag[fm][3])          \
                             : "r"(addr));                                        \
            }                                                                       \
            _Pragma("unroll")                                                     \
            for (int fn = 0; fn < (FN_COUNT); fn++) {                              \
                int row = k0 + b_row_offset + matrix_row;                          \
                unsigned addr = Ys_read +                                          \
                    (unsigned)((row * GEMM_BI_SMALL16_LDB + fn * 8) * 2);                    \
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "     \
                             "{%0,%1}, [%2];\n"                                   \
                             : "=r"(b_frag[fn][0]), "=r"(b_frag[fn][1])          \
                             : "r"(addr));                                        \
            }                                                                       \
            _Pragma("unroll")                                                     \
            for (int fm = 0; fm < (FM_COUNT); fm++) {                              \
                _Pragma("unroll")                                                 \
                for (int fn = 0; fn < (FN_COUNT); fn++) {                          \
                    asm volatile(                                                   \
                        "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "."       \
                        MMA_T ".f32 "                                             \
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "                 \
                        "{%0,%1,%2,%3};\n"                                        \
                        : "+f"(acc[fm][fn][0]), "+f"(acc[fm][fn][1]),           \
                          "+f"(acc[fm][fn][2]), "+f"(acc[fm][fn][3])            \
                        : "r"(a_frag[fm][0]), "r"(a_frag[fm][1]),                \
                          "r"(a_frag[fm][2]), "r"(a_frag[fm][3]),                \
                          "r"(b_frag[fn][0]), "r"(b_frag[fn][1]));               \
                }                                                                   \
            }                                                                       \
        }                                                                           \
        read_buf ^= 1;                                                              \
    }                                                                               \
    _Pragma("unroll")                                                             \
    for (int fm = 0; fm < (FM_COUNT); fm++) {                                      \
        _Pragma("unroll")                                                         \
        for (int fn = 0; fn < (FN_COUNT); fn++) {                                  \
            int row0 = pid_m * (BM) + fm * 16 + group;                             \
            int column0 = pid_n * (BN) + fn * 8 + 2 * thread;                      \
            _Pragma("unroll")                                                     \
            for (int e = 0; e < 4; e++) {                                          \
                int row = row0 + (e >= 2 ? 8 : 0);                                 \
                int column = column0 + (e & 1);                                    \
                if (row >= K_out || column >= N) continue;                         \
                C[(long long)row * N + column] += alpha * acc[fm][fn][e];          \
            }                                                                       \
        }                                                                           \
    }                                                                               \
}

DEFINE_GEMM_BI_SMALL16_TN(tn_sm89_m16n16_bk64_s2_ldb72_bf16,
                __nv_bfloat16, small16_from_bf16, "bf16", 16, 16, 1, 2)
DEFINE_GEMM_BI_SMALL16_TN(tn_sm89_m16n16_bk64_s2_ldb72_f16,
                __half, small16_from_f16, "f16", 16, 16, 1, 2)
// SM89_HALF_TN_SMALL16_END
#undef DEFINE_GEMM_BI_SMALL16_TN
#undef GEMM_BI_SMALL16_STAGE_SCALAR
#undef GEMM_BI_SMALL16_STAGE_ASYNC
#undef GEMM_BI_SMALL16_THREADS
#undef GEMM_BI_SMALL16_LDB
#undef GEMM_BI_SMALL16_STAGES
#undef GEMM_BI_SMALL16_BK
