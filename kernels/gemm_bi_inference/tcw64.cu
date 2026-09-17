// ============================================================================
// GBFW64: the fragment-reuse 128x128 tile (warp tile 64x64).
// ============================================================================
// Same numeric contract as GBF128 and bit-identical to it per output
// element: K walks in the same ascending 64-wide slabs split into the
// same ascending m16n8k16 steps with the same 16-byte-chunk zero-fill
// for tails, so an element's f32 accumulator sees the exact same mma
// chain whichever tile the dispatcher picked. What changes is only who
// computes it: four warps own 64x64 output quadrants instead of eight
// warps owning 64x32 halves, which doubles the work each loaded
// fragment feeds (mma:ldmatrix 2.0 -> 4.0) and cuts shared-memory read
// traffic by a third - the lever the fat training shapes are bound by.
//
// Shared memory drops the +8-half row pads for the standard XOR chunk
// swizzle: the 16-byte chunk at logical (row, c) lives at physical
// chunk c ^ (row & 7), which walks all 32 banks for every 8-lane
// ldmatrix group and every cp.async write group, and shrinks the stage
// pair to 65,536 bytes. The swizzle relocates whole 16-byte chunks and
// never splits one, so the bytes every fragment receives are identical.
//
// Fragments for step ks+1 are loaded before the mma block of step ks
// (an explicit two-deep register buffer): with only four warps per SM,
// one warp must keep its sub-partition's tensor pipe fed, and that
// works only if the next step's operands are already in flight.
//
// A rung joins the dispatch ladder only after an on-box census proves
// per-element byte identity and a measured win.

#define GBFW64_BM 128
#define GBFW64_BN 128
#define GBFW64_BK 64
#define GBFW64_THREADS 128
// 16-byte chunks per staged row: A rows carry BK halves, B rows BN.
#define GBFW64_ACH (GBFW64_BK / 8)
#define GBFW64_BCH (GBFW64_BN / 8)
#define GBFW64_A_STAGE_BYTES (GBFW64_BM * GBFW64_ACH * 16)
#define GBFW64_B_STAGE_BYTES (GBFW64_BK * GBFW64_BCH * 16)
#define GBFW64_SWZ(chunk, row) ((chunk) ^ ((row) & 7))

// Stage one A(128x64) + B(64x128) tile pair into the swizzled layout
// via 16-byte cp.async with zero-fill tails; source pointers form only
// when bytes remain in the object (the same safety rule as every rung).
#define GBFW64_STAGE_ASYNC(buf, bkIdx)                                        \
    do {                                                                      \
        unsigned _as = As_sbase + (unsigned)((buf) * GBFW64_A_STAGE_BYTES);   \
        unsigned _bs = Bs_sbase + (unsigned)((buf) * GBFW64_B_STAGE_BYTES);   \
        for (int _i = threadIdx.x; _i < GBFW64_BM * GBFW64_ACH;               \
             _i += GBFW64_THREADS) {                                          \
            int _m = _i / GBFW64_ACH;                                         \
            int _c = _i % GBFW64_ACH;                                         \
            int _gr = pid_m * GBFW64_BM + _m;                                 \
            int _gc = (bkIdx) + _c * 8;                                       \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as + (unsigned)(_m * GBFW64_ACH * 16 +           \
                                             (GBFW64_SWZ(_c, _m) << 4));      \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&A[(long long)_gr * lda + _gc]                 \
                : (const void*)A;                                            \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBFW64_BK * GBFW64_BCH;               \
             _i += GBFW64_THREADS) {                                          \
            int _k = _i / GBFW64_BCH;                                         \
            int _c = _i % GBFW64_BCH;                                         \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBFW64_BN + _c * 8;                             \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs + (unsigned)(_k * GBFW64_BCH * 16 +           \
                                             (GBFW64_SWZ(_c, _k) << 4));      \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&B[(long long)_gk * ldb + _gn]                 \
                : (const void*)B;                                            \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

// Element-wise fallback for hostile strides/bases, writing the same
// swizzled layout: element (row, k) lands inside chunk k/8 at half
// k%8. Same values as the async path, including the zero-fill.
#define GBFW64_STAGE_SCALAR(buf, bkIdx, TT, FF)                               \
    do {                                                                      \
        unsigned char* _asw = As_bytes + (buf) * GBFW64_A_STAGE_BYTES;        \
        unsigned char* _bsw = Bs_bytes + (buf) * GBFW64_B_STAGE_BYTES;        \
        for (int _i = threadIdx.x; _i < GBFW64_BM * GBFW64_BK;                \
             _i += GBFW64_THREADS) {                                          \
            int _m = _i / GBFW64_BK;                                          \
            int _k = _i % GBFW64_BK;                                          \
            int _gr = pid_m * GBFW64_BM + _m;                                 \
            int _gc = (bkIdx) + _k;                                           \
            TT _v = (_gr < M && _gc < K) ? A[(long long)_gr * lda + _gc]      \
                                         : FF(0.0f);                          \
            *reinterpret_cast<TT*>(                                           \
                _asw + _m * GBFW64_ACH * 16 +                                 \
                (GBFW64_SWZ(_k >> 3, _m) << 4) + (_k & 7) * 2) = _v;          \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBFW64_BK * GBFW64_BN;                \
             _i += GBFW64_THREADS) {                                          \
            int _k = _i / GBFW64_BN;                                          \
            int _n = _i % GBFW64_BN;                                          \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBFW64_BN + _n;                                 \
            TT _v = (_gk < K && _gn < N) ? B[(long long)_gk * ldb + _gn]      \
                                         : FF(0.0f);                          \
            *reinterpret_cast<TT*>(                                           \
                _bsw + _k * GBFW64_BCH * 16 +                                 \
                (GBFW64_SWZ(_n >> 3, _k) << 4) + (_n & 7) * 2) = _v;          \
        }                                                                     \
    } while (0)

// Load step ks's fragments into register buffer `fb`: four .x4 loads
// for A (one per m-fragment) and four .x4.trans for B (each covering
// two adjacent n-fragments).
#define GBFW64_LOAD_FRAGS(fb, ksv)                                            \
    do {                                                                      \
        int _k0 = (ksv) * 16;                                                 \
        _Pragma("unroll")                                                     \
        for (int _fm = 0; _fm < 4; _fm++) {                                   \
            int _row = warpM + _fm * 16 + lm_row_off + lm_r;                  \
            int _chunk = (_k0 + lm_col_off) >> 3;                             \
            unsigned _addr = As_rd +                                          \
                (unsigned)(_row * GBFW64_ACH * 16 +                           \
                           (GBFW64_SWZ(_chunk, _row) << 4));                  \
            asm volatile(                                                     \
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "                   \
                "{%0,%1,%2,%3}, [%4];\n"                                      \
                : "=r"(a_frag[fb][_fm][0]), "=r"(a_frag[fb][_fm][1]),         \
                  "=r"(a_frag[fb][_fm][2]), "=r"(a_frag[fb][_fm][3])          \
                : "r"(_addr));                                                \
        }                                                                     \
        _Pragma("unroll")                                                     \
        for (int _j = 0; _j < 4; _j++) {                                      \
            int _row = _k0 + ((lm_q & 1) ? 8 : 0) + lm_r;                     \
            int _col = warpN + _j * 16 + ((lm_q & 2) ? 8 : 0);                \
            int _chunk = _col >> 3;                                           \
            unsigned _addr = Bs_rd +                                          \
                (unsigned)(_row * GBFW64_BCH * 16 +                           \
                           (GBFW64_SWZ(_chunk, _row) << 4));                  \
            asm volatile(                                                     \
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "             \
                "{%0,%1,%2,%3}, [%4];\n"                                      \
                : "=r"(b_frag[fb][2 * _j][0]), "=r"(b_frag[fb][2 * _j][1]),   \
                  "=r"(b_frag[fb][2 * _j + 1][0]),                            \
                  "=r"(b_frag[fb][2 * _j + 1][1])                             \
                : "r"(_addr));                                                \
        }                                                                     \
    } while (0)

#define DEFINE_GEMM_BI_NN_TCW64(SUFFIX, T_ACT, FROM_F, MMA_T)                 \
extern "C" __global__ __launch_bounds__(GBFW64_THREADS, 1)                    \
void nn_tcw64_##SUFFIX(                                               \
    T_ACT* __restrict__ C,                                                    \
    const T_ACT* __restrict__ A,                                              \
    const T_ACT* __restrict__ B,                                              \
    const float* __restrict__ bias,                                           \
    float alpha, float beta,                                                  \
    int M, int N, int K,                                                      \
    int lda, int ldb, int ldc                                                 \
) {                                                                           \
    assert(alpha == 1.0f || bias == nullptr);                                 \
    extern __shared__ __align__(16) unsigned char gbfw64_dynsmem[];           \
    unsigned char* As_bytes = gbfw64_dynsmem;                                 \
    unsigned char* Bs_bytes = gbfw64_dynsmem + 2 * GBFW64_A_STAGE_BYTES;      \
    int num_pid_n = (N + GBFW64_BN - 1) / GBFW64_BN;                          \
    int pid_m = blockIdx.x / num_pid_n;                                       \
    int pid_n = blockIdx.x % num_pid_n;                                       \
    int warp = threadIdx.x / 32;                                              \
    int lane = threadIdx.x % 32;                                              \
    int warpM = (warp >> 1) * 64;                                             \
    int warpN = (warp & 1) * 64;                                              \
    int g = lane >> 2;                                                        \
    int t = lane & 3;                                                         \
    int lm_r = lane & 7;                                                      \
    int lm_q = lane >> 3;                                                     \
    int lm_row_off = (lm_q & 1) ? 8 : 0;                                      \
    int lm_col_off = (lm_q & 2) ? 8 : 0;                                      \
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(As_bytes);         \
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(Bs_bytes);         \
    bool fast_stage = ((lda & 7) == 0) && ((ldb & 7) == 0) &&                 \
                      gbf_aligned16(A) && gbf_aligned16(B);                   \
    float acc[4][8][4];                                                       \
    _Pragma("unroll")                                                         \
    for (int fm = 0; fm < 4; fm++) {                                          \
        _Pragma("unroll")                                                     \
        for (int fn = 0; fn < 8; fn++) {                                      \
            float b0 = 0.0f, b1 = 0.0f;                                       \
            if (bias != nullptr) {                                            \
                int c0 = pid_n * GBFW64_BN + warpN + fn * 8 + 2 * t;          \
                b0 = (c0 < N) ? bias[c0] : 0.0f;                              \
                b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                      \
            }                                                                 \
            acc[fm][fn][0] = b0;                                              \
            acc[fm][fn][1] = b1;                                              \
            acc[fm][fn][2] = b0;                                              \
            acc[fm][fn][3] = b1;                                              \
        }                                                                     \
    }                                                                         \
    int num_k_tiles = (K + GBFW64_BK - 1) / GBFW64_BK;                        \
    if (fast_stage) {                                                         \
        GBFW64_STAGE_ASYNC(0, 0);                                             \
    } else {                                                                  \
        GBFW64_STAGE_SCALAR(0, 0, T_ACT, FROM_F);                             \
    }                                                                         \
    int read_buf = 0;                                                         \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                \
        if (fast_stage) {                                                     \
            asm volatile("cp.async.wait_group 0;\n");                         \
        }                                                                     \
        __syncthreads();                                                      \
        if (kt + 1 < num_k_tiles) {                                           \
            if (fast_stage) {                                                 \
                GBFW64_STAGE_ASYNC(read_buf ^ 1, (kt + 1) * GBFW64_BK);       \
            } else {                                                          \
                GBFW64_STAGE_SCALAR(read_buf ^ 1, (kt + 1) * GBFW64_BK,       \
                                    T_ACT, FROM_F);                           \
            }                                                                 \
        }                                                                     \
        unsigned As_rd =                                                      \
            As_sbase + (unsigned)(read_buf * GBFW64_A_STAGE_BYTES);           \
        unsigned Bs_rd =                                                      \
            Bs_sbase + (unsigned)(read_buf * GBFW64_B_STAGE_BYTES);           \
        unsigned a_frag[2][4][4];                                             \
        unsigned b_frag[2][8][2];                                             \
        GBFW64_LOAD_FRAGS(0, 0);                                              \
        _Pragma("unroll")                                                     \
        for (int ks = 0; ks < (GBFW64_BK / 16); ks++) {                       \
            int fb = ks & 1;                                                  \
            if (ks + 1 < (GBFW64_BK / 16)) {                                  \
                GBFW64_LOAD_FRAGS(fb ^ 1, ks + 1);                            \
            }                                                                 \
            _Pragma("unroll")                                                 \
            for (int fm = 0; fm < 4; fm++) {                                  \
                _Pragma("unroll")                                             \
                for (int fn = 0; fn < 8; fn++) {                              \
                    asm volatile(                                             \
                        "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "."    \
                        MMA_T ".f32 "                                         \
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "             \
                        "{%0,%1,%2,%3};\n"                                    \
                        : "+f"(acc[fm][fn][0]), "+f"(acc[fm][fn][1]),         \
                          "+f"(acc[fm][fn][2]), "+f"(acc[fm][fn][3])          \
                        : "r"(a_frag[fb][fm][0]), "r"(a_frag[fb][fm][1]),     \
                          "r"(a_frag[fb][fm][2]), "r"(a_frag[fb][fm][3]),     \
                          "r"(b_frag[fb][fn][0]), "r"(b_frag[fb][fn][1]));    \
                }                                                             \
            }                                                                 \
        }                                                                     \
        read_buf ^= 1;                                                        \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int fm = 0; fm < 4; fm++) {                                          \
        _Pragma("unroll")                                                     \
        for (int fn = 0; fn < 8; fn++) {                                      \
            int r0 = pid_m * GBFW64_BM + warpM + fm * 16 + g;                 \
            int c0 = pid_n * GBFW64_BN + warpN + fn * 8 + 2 * t;              \
            _Pragma("unroll")                                                 \
            for (int half = 0; half < 2; half++) {                            \
                int gr = r0 + (half ? 8 : 0);                                 \
                if (gr >= M) continue;                                        \
                float v0 = __fmul_rn(alpha, acc[fm][fn][2 * half]);           \
                float v1 = __fmul_rn(alpha, acc[fm][fn][2 * half + 1]);       \
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
                            val = __fmaf_rn(                                  \
                                beta, to_f(C[(long long)gr * ldc + gc]),      \
                                val);                                         \
                        C[(long long)gr * ldc + gc] = FROM_F(val);            \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

DEFINE_GEMM_BI_NN_TCW64(bf16, __nv_bfloat16, from_f_bf16, "bf16")
DEFINE_GEMM_BI_NN_TCW64(f16,  __half,        from_f_f16,  "f16")

#undef GBFW64_BM
#undef GBFW64_BN
#undef GBFW64_BK
#undef GBFW64_THREADS
#undef GBFW64_ACH
#undef GBFW64_BCH
#undef GBFW64_A_STAGE_BYTES
#undef GBFW64_B_STAGE_BYTES
#undef GBFW64_SWZ
#undef GBFW64_STAGE_ASYNC
#undef GBFW64_STAGE_SCALAR
#undef GBFW64_LOAD_FRAGS
#undef DEFINE_GEMM_BI_NN_TCW64

// ============================================================================
// GBFWN64: the same 64x64 warp tile at CTA 128x256 with 8 warps.
// ============================================================================
// The four-warp constant-area variant measured behind the shipped
// 128-tile everywhere: one warp per sub-partition cannot hide fragment
// latency even with the explicit double buffer. This variant keeps the
// fragment-reuse arithmetic and restores 8 warps per SM by widening the
// CTA to 128x256 (98,304 B of staged operands - inside the opt-in cap
// only because the swizzle is pad-free). It trades wave granularity
// for it, so its wins split by shape - the dispatch rule admits it
// only where the measured grid showed a win.
// Same numeric contract, same per-element mma chain: bit-identical to
// the 128-tile by the same argument, and censused the same way.
#define GBFWN64_BM 128
#define GBFWN64_BN 256
#define GBFWN64_BK 64
#define GBFWN64_THREADS 256
// 16-byte chunks per staged row: A rows carry BK halves, B rows BN.
#define GBFWN64_ACH (GBFWN64_BK / 8)
#define GBFWN64_BCH (GBFWN64_BN / 8)
#define GBFWN64_A_STAGE_BYTES (GBFWN64_BM * GBFWN64_ACH * 16)
#define GBFWN64_B_STAGE_BYTES (GBFWN64_BK * GBFWN64_BCH * 16)
#define GBFWN64_SWZ(chunk, row) ((chunk) ^ ((row) & 7))

// Stage one A(128x64) + B(64x128) tile pair into the swizzled layout
// via 16-byte cp.async with zero-fill tails; source pointers form only
// when bytes remain in the object (the same safety rule as every rung).
#define GBFWN64_STAGE_ASYNC(buf, bkIdx)                                        \
    do {                                                                      \
        unsigned _as = As_sbase + (unsigned)((buf) * GBFWN64_A_STAGE_BYTES);   \
        unsigned _bs = Bs_sbase + (unsigned)((buf) * GBFWN64_B_STAGE_BYTES);   \
        for (int _i = threadIdx.x; _i < GBFWN64_BM * GBFWN64_ACH;               \
             _i += GBFWN64_THREADS) {                                          \
            int _m = _i / GBFWN64_ACH;                                         \
            int _c = _i % GBFWN64_ACH;                                         \
            int _gr = pid_m * GBFWN64_BM + _m;                                 \
            int _gc = (bkIdx) + _c * 8;                                       \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as + (unsigned)(_m * GBFWN64_ACH * 16 +           \
                                             (GBFWN64_SWZ(_c, _m) << 4));      \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&A[(long long)_gr * lda + _gc]                 \
                : (const void*)A;                                            \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBFWN64_BK * GBFWN64_BCH;               \
             _i += GBFWN64_THREADS) {                                          \
            int _k = _i / GBFWN64_BCH;                                         \
            int _c = _i % GBFWN64_BCH;                                         \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBFWN64_BN + _c * 8;                             \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs + (unsigned)(_k * GBFWN64_BCH * 16 +           \
                                             (GBFWN64_SWZ(_c, _k) << 4));      \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&B[(long long)_gk * ldb + _gn]                 \
                : (const void*)B;                                            \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

// Element-wise fallback for hostile strides/bases, writing the same
// swizzled layout: element (row, k) lands inside chunk k/8 at half
// k%8. Same values as the async path, including the zero-fill.
#define GBFWN64_STAGE_SCALAR(buf, bkIdx, TT, FF)                               \
    do {                                                                      \
        unsigned char* _asw = As_bytes + (buf) * GBFWN64_A_STAGE_BYTES;        \
        unsigned char* _bsw = Bs_bytes + (buf) * GBFWN64_B_STAGE_BYTES;        \
        for (int _i = threadIdx.x; _i < GBFWN64_BM * GBFWN64_BK;                \
             _i += GBFWN64_THREADS) {                                          \
            int _m = _i / GBFWN64_BK;                                          \
            int _k = _i % GBFWN64_BK;                                          \
            int _gr = pid_m * GBFWN64_BM + _m;                                 \
            int _gc = (bkIdx) + _k;                                           \
            TT _v = (_gr < M && _gc < K) ? A[(long long)_gr * lda + _gc]      \
                                         : FF(0.0f);                          \
            *reinterpret_cast<TT*>(                                           \
                _asw + _m * GBFWN64_ACH * 16 +                                 \
                (GBFWN64_SWZ(_k >> 3, _m) << 4) + (_k & 7) * 2) = _v;          \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBFWN64_BK * GBFWN64_BN;                \
             _i += GBFWN64_THREADS) {                                          \
            int _k = _i / GBFWN64_BN;                                          \
            int _n = _i % GBFWN64_BN;                                          \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBFWN64_BN + _n;                                 \
            TT _v = (_gk < K && _gn < N) ? B[(long long)_gk * ldb + _gn]      \
                                         : FF(0.0f);                          \
            *reinterpret_cast<TT*>(                                           \
                _bsw + _k * GBFWN64_BCH * 16 +                                 \
                (GBFWN64_SWZ(_n >> 3, _k) << 4) + (_n & 7) * 2) = _v;          \
        }                                                                     \
    } while (0)

// Load step ks's fragments into register buffer `fb`: four .x4 loads
// for A (one per m-fragment) and four .x4.trans for B (each covering
// two adjacent n-fragments).
#define GBFWN64_LOAD_FRAGS(fb, ksv)                                            \
    do {                                                                      \
        int _k0 = (ksv) * 16;                                                 \
        _Pragma("unroll")                                                     \
        for (int _fm = 0; _fm < 4; _fm++) {                                   \
            int _row = warpM + _fm * 16 + lm_row_off + lm_r;                  \
            int _chunk = (_k0 + lm_col_off) >> 3;                             \
            unsigned _addr = As_rd +                                          \
                (unsigned)(_row * GBFWN64_ACH * 16 +                           \
                           (GBFWN64_SWZ(_chunk, _row) << 4));                  \
            asm volatile(                                                     \
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "                   \
                "{%0,%1,%2,%3}, [%4];\n"                                      \
                : "=r"(a_frag[fb][_fm][0]), "=r"(a_frag[fb][_fm][1]),         \
                  "=r"(a_frag[fb][_fm][2]), "=r"(a_frag[fb][_fm][3])          \
                : "r"(_addr));                                                \
        }                                                                     \
        _Pragma("unroll")                                                     \
        for (int _j = 0; _j < 4; _j++) {                                      \
            int _row = _k0 + ((lm_q & 1) ? 8 : 0) + lm_r;                     \
            int _col = warpN + _j * 16 + ((lm_q & 2) ? 8 : 0);                \
            int _chunk = _col >> 3;                                           \
            unsigned _addr = Bs_rd +                                          \
                (unsigned)(_row * GBFWN64_BCH * 16 +                           \
                           (GBFWN64_SWZ(_chunk, _row) << 4));                  \
            asm volatile(                                                     \
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "             \
                "{%0,%1,%2,%3}, [%4];\n"                                      \
                : "=r"(b_frag[fb][2 * _j][0]), "=r"(b_frag[fb][2 * _j][1]),   \
                  "=r"(b_frag[fb][2 * _j + 1][0]),                            \
                  "=r"(b_frag[fb][2 * _j + 1][1])                             \
                : "r"(_addr));                                                \
        }                                                                     \
    } while (0)

#define DEFINE_GEMM_BI_NN_TCWN64(SUFFIX, T_ACT, FROM_F, MMA_T)                 \
extern "C" __global__ __launch_bounds__(GBFWN64_THREADS, 1)                    \
void nn_tcwn64_##SUFFIX(                                               \
    T_ACT* __restrict__ C,                                                    \
    const T_ACT* __restrict__ A,                                              \
    const T_ACT* __restrict__ B,                                              \
    const float* __restrict__ bias,                                           \
    float alpha, float beta,                                                  \
    int M, int N, int K,                                                      \
    int lda, int ldb, int ldc                                                 \
) {                                                                           \
    assert(alpha == 1.0f || bias == nullptr);                                 \
    extern __shared__ __align__(16) unsigned char gbfwn64_dynsmem[];           \
    unsigned char* As_bytes = gbfwn64_dynsmem;                                 \
    unsigned char* Bs_bytes = gbfwn64_dynsmem + 2 * GBFWN64_A_STAGE_BYTES;      \
    int num_pid_n = (N + GBFWN64_BN - 1) / GBFWN64_BN;                          \
    int pid_m = blockIdx.x / num_pid_n;                                       \
    int pid_n = blockIdx.x % num_pid_n;                                       \
    int warp = threadIdx.x / 32;                                              \
    int lane = threadIdx.x % 32;                                              \
    int warpM = (warp >> 2) * 64;                                             \
    int warpN = (warp & 3) * 64;                                              \
    int g = lane >> 2;                                                        \
    int t = lane & 3;                                                         \
    int lm_r = lane & 7;                                                      \
    int lm_q = lane >> 3;                                                     \
    int lm_row_off = (lm_q & 1) ? 8 : 0;                                      \
    int lm_col_off = (lm_q & 2) ? 8 : 0;                                      \
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(As_bytes);         \
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(Bs_bytes);         \
    bool fast_stage = ((lda & 7) == 0) && ((ldb & 7) == 0) &&                 \
                      gbf_aligned16(A) && gbf_aligned16(B);                   \
    float acc[4][8][4];                                                       \
    _Pragma("unroll")                                                         \
    for (int fm = 0; fm < 4; fm++) {                                          \
        _Pragma("unroll")                                                     \
        for (int fn = 0; fn < 8; fn++) {                                      \
            float b0 = 0.0f, b1 = 0.0f;                                       \
            if (bias != nullptr) {                                            \
                int c0 = pid_n * GBFWN64_BN + warpN + fn * 8 + 2 * t;          \
                b0 = (c0 < N) ? bias[c0] : 0.0f;                              \
                b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                      \
            }                                                                 \
            acc[fm][fn][0] = b0;                                              \
            acc[fm][fn][1] = b1;                                              \
            acc[fm][fn][2] = b0;                                              \
            acc[fm][fn][3] = b1;                                              \
        }                                                                     \
    }                                                                         \
    int num_k_tiles = (K + GBFWN64_BK - 1) / GBFWN64_BK;                        \
    if (fast_stage) {                                                         \
        GBFWN64_STAGE_ASYNC(0, 0);                                             \
    } else {                                                                  \
        GBFWN64_STAGE_SCALAR(0, 0, T_ACT, FROM_F);                             \
    }                                                                         \
    int read_buf = 0;                                                         \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                \
        if (fast_stage) {                                                     \
            asm volatile("cp.async.wait_group 0;\n");                         \
        }                                                                     \
        __syncthreads();                                                      \
        if (kt + 1 < num_k_tiles) {                                           \
            if (fast_stage) {                                                 \
                GBFWN64_STAGE_ASYNC(read_buf ^ 1, (kt + 1) * GBFWN64_BK);       \
            } else {                                                          \
                GBFWN64_STAGE_SCALAR(read_buf ^ 1, (kt + 1) * GBFWN64_BK,       \
                                    T_ACT, FROM_F);                           \
            }                                                                 \
        }                                                                     \
        unsigned As_rd =                                                      \
            As_sbase + (unsigned)(read_buf * GBFWN64_A_STAGE_BYTES);           \
        unsigned Bs_rd =                                                      \
            Bs_sbase + (unsigned)(read_buf * GBFWN64_B_STAGE_BYTES);           \
        unsigned a_frag[2][4][4];                                             \
        unsigned b_frag[2][8][2];                                             \
        GBFWN64_LOAD_FRAGS(0, 0);                                              \
        _Pragma("unroll")                                                     \
        for (int ks = 0; ks < (GBFWN64_BK / 16); ks++) {                       \
            int fb = ks & 1;                                                  \
            if (ks + 1 < (GBFWN64_BK / 16)) {                                  \
                GBFWN64_LOAD_FRAGS(fb ^ 1, ks + 1);                            \
            }                                                                 \
            _Pragma("unroll")                                                 \
            for (int fm = 0; fm < 4; fm++) {                                  \
                _Pragma("unroll")                                             \
                for (int fn = 0; fn < 8; fn++) {                              \
                    asm volatile(                                             \
                        "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "."    \
                        MMA_T ".f32 "                                         \
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "             \
                        "{%0,%1,%2,%3};\n"                                    \
                        : "+f"(acc[fm][fn][0]), "+f"(acc[fm][fn][1]),         \
                          "+f"(acc[fm][fn][2]), "+f"(acc[fm][fn][3])          \
                        : "r"(a_frag[fb][fm][0]), "r"(a_frag[fb][fm][1]),     \
                          "r"(a_frag[fb][fm][2]), "r"(a_frag[fb][fm][3]),     \
                          "r"(b_frag[fb][fn][0]), "r"(b_frag[fb][fn][1]));    \
                }                                                             \
            }                                                                 \
        }                                                                     \
        read_buf ^= 1;                                                        \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int fm = 0; fm < 4; fm++) {                                          \
        _Pragma("unroll")                                                     \
        for (int fn = 0; fn < 8; fn++) {                                      \
            int r0 = pid_m * GBFWN64_BM + warpM + fm * 16 + g;                 \
            int c0 = pid_n * GBFWN64_BN + warpN + fn * 8 + 2 * t;              \
            _Pragma("unroll")                                                 \
            for (int half = 0; half < 2; half++) {                            \
                int gr = r0 + (half ? 8 : 0);                                 \
                if (gr >= M) continue;                                        \
                float v0 = __fmul_rn(alpha, acc[fm][fn][2 * half]);           \
                float v1 = __fmul_rn(alpha, acc[fm][fn][2 * half + 1]);       \
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
                            val = __fmaf_rn(                                  \
                                beta, to_f(C[(long long)gr * ldc + gc]),      \
                                val);                                         \
                        C[(long long)gr * ldc + gc] = FROM_F(val);            \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

DEFINE_GEMM_BI_NN_TCWN64(bf16, __nv_bfloat16, from_f_bf16, "bf16")
DEFINE_GEMM_BI_NN_TCWN64(f16,  __half,        from_f_f16,  "f16")

#undef GBFWN64_BM
#undef GBFWN64_BN
#undef GBFWN64_BK
#undef GBFWN64_THREADS
#undef GBFWN64_ACH
#undef GBFWN64_BCH
#undef GBFWN64_A_STAGE_BYTES
#undef GBFWN64_B_STAGE_BYTES
#undef GBFWN64_SWZ
#undef GBFWN64_STAGE_ASYNC
#undef GBFWN64_STAGE_SCALAR
#undef GBFWN64_LOAD_FRAGS
#undef DEFINE_GEMM_BI_NN_TCWN64
