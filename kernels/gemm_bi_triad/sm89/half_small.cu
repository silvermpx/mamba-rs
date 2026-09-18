// Small-tile NN forward and NT input-gradient kernels for the d128 shapes.
//
// At 1024 x 128 outputs the 64x64 tile gives the board 32 CTAs of four
// warps: one warp per tensor pipe on a fifth of the multiprocessors, and the
// whole per-CTA reduction runs on that one warp. A smaller tile raises the
// CTA count to one or two per multiprocessor and shortens every CTA's chain;
// a four-deep cp.async pipeline keeps the short reductions from exposing a
// load latency per slab. The arithmetic is the tiled kernels': ascending
// 64-wide slabs split into ascending m16n8k16 steps with f32 accumulators,
// bias seeded before the chain, zero-filled 16-byte tails, and the same
// epilogue rounding; every output element sees the same chain regardless of
// the tile, so the result is bit for bit the 64x64 kernel's.

namespace sm89_half_small {

#define SMALL_BK 64
#define SMALL_LDK (SMALL_BK + 8)

template <typename T> struct SmallOps;

#define SMALL_OPS(TYPE, FROM, MMA_TYPE)                                       \
template <> struct SmallOps<TYPE> {                                           \
    static __device__ __forceinline__ TYPE from_float(float value) {          \
        return FROM(value);                                                   \
    }                                                                         \
    static __device__ __forceinline__ void mma(                               \
        float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {      \
        asm volatile(                                                         \
            "mma.sync.aligned.m16n8k16.row.col.f32." MMA_TYPE "."             \
            MMA_TYPE ".f32 "                                                  \
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"          \
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                  \
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),                     \
              "r"(b[0]), "r"(b[1]));                                          \
    }                                                                         \
};

SMALL_OPS(__nv_bfloat16, from_f_bf16, "bf16")
SMALL_OPS(__half, from_f_f16, "f16")
#undef SMALL_OPS

// ---------------------------------------------------------------------------
// NT: dX[M, K_out] = dY[M, N] * W[K_out, N]^T. Both operands are staged as
// rows of 64 reduction elements (72-half pitch), read by non-transposing
// ldmatrix; the reduction axis is N.
// ---------------------------------------------------------------------------
template <typename T, int TM, int TN, int WARPS_M, int WARPS_N, int STAGES>
struct SmallNt {
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
    static constexpr int WARP_M = TM / WARPS_M;
    static constexpr int WARP_N = TN / WARPS_N;
    static constexpr int FM = WARP_M / 16;
    static constexpr int FN = WARP_N / 8;
    static constexpr int A_STAGE = TM * SMALL_LDK;
    static constexpr int B_STAGE = TN * SMALL_LDK;
    static constexpr int SHARED_BYTES = STAGES * (A_STAGE + B_STAGE) * (int)sizeof(T);
    static_assert(FM >= 1 && FN >= 1, "warp tile too small");

    static __device__ __forceinline__ void stage_async(
        unsigned ys, unsigned ws, const T* A, const T* B,
        int n_base, int pid_m, int pid_n, int M, int N, int K_out) {
        for (int i = (int)threadIdx.x; i < TM * (SMALL_BK / 8); i += THREADS) {
            int m = i / (SMALL_BK / 8);
            int c = (i % (SMALL_BK / 8)) * 8;
            int gm = pid_m * TM + m;
            int gn = n_base + c;
            int elems = cp_async_valid_elems(gm < M, N, gn);
            int bytes = elems * 2;
            unsigned dst = ys + (unsigned)((m * SMALL_LDK + c) * 2);
            long long offset = bytes == 0 ? 0 : (long long)gm * N + gn;
            const void* src = cp_async_source(A, offset, bytes);
            cp_async_16_zfill(dst, src, bytes);
        }
        for (int i = (int)threadIdx.x; i < TN * (SMALL_BK / 8); i += THREADS) {
            int k = i / (SMALL_BK / 8);
            int c = (i % (SMALL_BK / 8)) * 8;
            int gk = pid_n * TN + k;
            int gn = n_base + c;
            int elems = cp_async_valid_elems(gk < K_out, N, gn);
            int bytes = elems * 2;
            unsigned dst = ws + (unsigned)((k * SMALL_LDK + c) * 2);
            long long offset = bytes == 0 ? 0 : (long long)gk * N + gn;
            const void* src = cp_async_source(B, offset, bytes);
            cp_async_16_zfill(dst, src, bytes);
        }
    }

    static __device__ __forceinline__ void stage_scalar(
        T* ys, T* ws, const T* A, const T* B,
        int n_base, int pid_m, int pid_n, int M, int N, int K_out) {
        for (int i = (int)threadIdx.x; i < TM * SMALL_BK; i += THREADS) {
            int m = i / SMALL_BK;
            int c = i % SMALL_BK;
            int gm = pid_m * TM + m;
            int gn = n_base + c;
            ys[m * SMALL_LDK + c] = (gm < M && gn < N)
                ? A[(long long)gm * N + gn] : SmallOps<T>::from_float(0.0f);
        }
        for (int i = (int)threadIdx.x; i < TN * SMALL_BK; i += THREADS) {
            int k = i / SMALL_BK;
            int c = i % SMALL_BK;
            int gk = pid_n * TN + k;
            int gn = n_base + c;
            ws[k * SMALL_LDK + c] = (gk < K_out && gn < N)
                ? B[(long long)gk * N + gn] : SmallOps<T>::from_float(0.0f);
        }
    }

    static __device__ __forceinline__ void run(
        T* __restrict__ C, const T* __restrict__ A, const T* __restrict__ B,
        float alpha, int M, int N, int K_out) {
        extern __shared__ __align__(16) unsigned char small_shared[];
        T* Ys = reinterpret_cast<T*>(small_shared);
        T* Ws = Ys + STAGES * A_STAGE;
        unsigned Ys_sbase = (unsigned)__cvta_generic_to_shared(Ys);
        unsigned Ws_sbase = (unsigned)__cvta_generic_to_shared(Ws);
        int num_pid_n = (K_out + TN - 1) / TN;
        int pid_m = (int)blockIdx.x / num_pid_n;
        int pid_n = (int)blockIdx.x % num_pid_n;
        int warp = (int)threadIdx.x / 32;
        int lane = (int)threadIdx.x % 32;
        int warpM = (warp / WARPS_N) * WARP_M;
        int warpN = (warp % WARPS_N) * WARP_N;
        int g = lane >> 2;
        int t = lane & 3;
        int lm_r = lane & 7;
        int lm_q = lane >> 3;
        int lm_row_off = (lm_q & 1) ? 8 : 0;
        int lm_col_off = (lm_q & 2) ? 8 : 0;
        int lmb_col_off = (lm_q & 1) ? 8 : 0;
        bool fast_stage = is_aligned_16(A) && is_aligned_16(B) && ((N & 7) == 0);
        float acc[FM][FN][4];
#pragma unroll
        for (int fm = 0; fm < FM; ++fm)
#pragma unroll
            for (int fn = 0; fn < FN; ++fn)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[fm][fn][e] = 0.0f;
        int num_n_tiles = (N + SMALL_BK - 1) / SMALL_BK;
        if (fast_stage) {
#pragma unroll
            for (int s = 0; s < STAGES - 1; ++s) {
                if (s < num_n_tiles) {
                    stage_async(Ys_sbase + (unsigned)(s * A_STAGE * 2),
                        Ws_sbase + (unsigned)(s * B_STAGE * 2),
                        A, B, s * SMALL_BK, pid_m, pid_n, M, N, K_out);
                }
                asm volatile("cp.async.commit_group;\n" ::);
            }
        } else {
            stage_scalar(Ys, Ws, A, B, 0, pid_m, pid_n, M, N, K_out);
        }
        for (int nt = 0; nt < num_n_tiles; ++nt) {
            int read_buf = nt % STAGES;
            if (fast_stage) {
                // Wait until only the stages still being filled are in
                // flight, so the buffer about to be read has landed.
                asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
            }
            __syncthreads();
            int ahead = nt + STAGES - 1;
            if (fast_stage) {
                if (ahead < num_n_tiles) {
                    int write_buf = ahead % STAGES;
                    stage_async(Ys_sbase + (unsigned)(write_buf * A_STAGE * 2),
                        Ws_sbase + (unsigned)(write_buf * B_STAGE * 2),
                        A, B, ahead * SMALL_BK, pid_m, pid_n, M, N, K_out);
                }
                asm volatile("cp.async.commit_group;\n" ::);
            } else if (nt + 1 < num_n_tiles) {
                int write_buf = (nt + 1) % STAGES;
                stage_scalar(Ys + write_buf * A_STAGE, Ws + write_buf * B_STAGE,
                    A, B, (nt + 1) * SMALL_BK, pid_m, pid_n, M, N, K_out);
            }
            unsigned ys_rd = Ys_sbase + (unsigned)(read_buf * A_STAGE * 2);
            unsigned ws_rd = Ws_sbase + (unsigned)(read_buf * B_STAGE * 2);
#pragma unroll
            for (int ks = 0; ks < SMALL_BK / 16; ++ks) {
                int k0 = ks * 16;
                unsigned a_frag[FM][4];
                unsigned b_frag[FN][2];
#pragma unroll
                for (int fm = 0; fm < FM; ++fm) {
                    int row = warpM + fm * 16 + lm_row_off + lm_r;
                    unsigned addr = ys_rd + (unsigned)((row * SMALL_LDK + k0 + lm_col_off) * 2);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(a_frag[fm][0]), "=r"(a_frag[fm][1]),
                          "=r"(a_frag[fm][2]), "=r"(a_frag[fm][3]) : "r"(addr));
                }
#pragma unroll
                for (int fn = 0; fn < FN; ++fn) {
                    int row = warpN + fn * 8 + lm_r;
                    unsigned addr = ws_rd + (unsigned)((row * SMALL_LDK + k0 + lmb_col_off) * 2);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(b_frag[fn][0]), "=r"(b_frag[fn][1]) : "r"(addr));
                }
#pragma unroll
                for (int fm = 0; fm < FM; ++fm)
#pragma unroll
                    for (int fn = 0; fn < FN; ++fn)
                        SmallOps<T>::mma(acc[fm][fn], a_frag[fm], b_frag[fn]);
            }
        }
#pragma unroll
        for (int fm = 0; fm < FM; ++fm) {
#pragma unroll
            for (int fn = 0; fn < FN; ++fn) {
                int r0 = pid_m * TM + warpM + fm * 16 + g;
                int c0 = pid_n * TN + warpN + fn * 8 + 2 * t;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    int gr = r0 + (e >= 2 ? 8 : 0);
                    int gc = c0 + (e & 1);
                    if (gr >= M || gc >= K_out) continue;
                    C[(long long)gr * K_out + gc] = SmallOps<T>::from_float(alpha * acc[fm][fn][e]);
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// NN: C[M, N] = A[M, K] * B[K, N] (+ bias). A is staged as rows of 64
// reduction elements, B as 64 reduction rows of TN columns (TN + 8 pitch),
// read by a transposing ldmatrix.
// ---------------------------------------------------------------------------
template <typename T, int TM, int TN, int WARPS_M, int WARPS_N, int STAGES>
struct SmallNn {
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
    static constexpr int WARP_M = TM / WARPS_M;
    static constexpr int WARP_N = TN / WARPS_N;
    static constexpr int FM = WARP_M / 16;
    static constexpr int FN = WARP_N / 8;
    static constexpr int LDB = TN + 8;
    static constexpr int A_STAGE = TM * SMALL_LDK;
    static constexpr int B_STAGE = SMALL_BK * LDB;
    static constexpr int SHARED_BYTES = STAGES * (A_STAGE + B_STAGE) * (int)sizeof(T);
    static_assert(FM >= 1 && FN >= 1, "warp tile too small");

    static __device__ __forceinline__ void stage_async(
        unsigned as, unsigned bs, const T* A, const T* B,
        int k_base, int pid_m, int pid_n, int M, int N, int K, int lda, int ldb) {
        for (int i = (int)threadIdx.x; i < TM * (SMALL_BK / 8); i += THREADS) {
            int m = i / (SMALL_BK / 8);
            int k = (i % (SMALL_BK / 8)) * 8;
            int gr = pid_m * TM + m;
            int gc = k_base + k;
            int elems = cp_async_valid_elems(gr < M, K, gc);
            int bytes = elems * 2;
            unsigned dst = as + (unsigned)((m * SMALL_LDK + k) * 2);
            long long offset = bytes == 0 ? 0 : (long long)gr * lda + gc;
            const void* src = cp_async_source(A, offset, bytes);
            cp_async_16_zfill(dst, src, bytes);
        }
        for (int i = (int)threadIdx.x; i < SMALL_BK * (TN / 8); i += THREADS) {
            int k = i / (TN / 8);
            int n = (i % (TN / 8)) * 8;
            int gk = k_base + k;
            int gn = pid_n * TN + n;
            int elems = cp_async_valid_elems(gk < K, N, gn);
            int bytes = elems * 2;
            unsigned dst = bs + (unsigned)((k * LDB + n) * 2);
            long long offset = bytes == 0 ? 0 : (long long)gk * ldb + gn;
            const void* src = cp_async_source(B, offset, bytes);
            cp_async_16_zfill(dst, src, bytes);
        }
    }

    static __device__ __forceinline__ void stage_scalar(
        T* as, T* bs, const T* A, const T* B,
        int k_base, int pid_m, int pid_n, int M, int N, int K, int lda, int ldb) {
        for (int i = (int)threadIdx.x; i < TM * SMALL_BK; i += THREADS) {
            int m = i / SMALL_BK;
            int k = i % SMALL_BK;
            int gr = pid_m * TM + m;
            int gc = k_base + k;
            as[m * SMALL_LDK + k] = (gr < M && gc < K)
                ? A[(long long)gr * lda + gc] : SmallOps<T>::from_float(0.0f);
        }
        for (int i = (int)threadIdx.x; i < SMALL_BK * TN; i += THREADS) {
            int k = i / TN;
            int n = i % TN;
            int gk = k_base + k;
            int gn = pid_n * TN + n;
            bs[k * LDB + n] = (gk < K && gn < N)
                ? B[(long long)gk * ldb + gn] : SmallOps<T>::from_float(0.0f);
        }
    }

    static __device__ __forceinline__ void run(
        T* __restrict__ C, const T* __restrict__ A, const T* __restrict__ B,
        const float* __restrict__ bias, float alpha, float beta,
        int M, int N, int K, int lda, int ldb, int ldc) {
        extern __shared__ __align__(16) unsigned char small_shared[];
        T* As = reinterpret_cast<T*>(small_shared);
        T* Bs = As + STAGES * A_STAGE;
        unsigned As_sbase = (unsigned)__cvta_generic_to_shared(As);
        unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(Bs);
        int num_pid_n = (N + TN - 1) / TN;
        int pid_m = (int)blockIdx.x / num_pid_n;
        int pid_n = (int)blockIdx.x % num_pid_n;
        int warp = (int)threadIdx.x / 32;
        int lane = (int)threadIdx.x % 32;
        int warpM = (warp / WARPS_N) * WARP_M;
        int warpN = (warp % WARPS_N) * WARP_N;
        int g = lane >> 2;
        int t = lane & 3;
        int lm_r = lane & 7;
        int lm_q = lane >> 3;
        int lm_row_off = (lm_q & 1) ? 8 : 0;
        int lm_col_off = (lm_q & 2) ? 8 : 0;
        int lmb_row_off = (lm_q & 1) ? 8 : 0;
        bool fast_stage = is_aligned_16(A) && is_aligned_16(B)
            && ((lda & 7) == 0) && ((ldb & 7) == 0) && ((K & 7) == 0) && ((N & 7) == 0);
        float acc[FM][FN][4];
#pragma unroll
        for (int fm = 0; fm < FM; ++fm) {
#pragma unroll
            for (int fn = 0; fn < FN; ++fn) {
                float b0 = 0.0f, b1 = 0.0f;
                if (bias != nullptr) {
                    int c0 = pid_n * TN + warpN + fn * 8 + 2 * t;
                    b0 = (c0 < N) ? bias[c0] : 0.0f;
                    b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;
                }
                acc[fm][fn][0] = b0;
                acc[fm][fn][1] = b1;
                acc[fm][fn][2] = b0;
                acc[fm][fn][3] = b1;
            }
        }
        int num_k_tiles = (K + SMALL_BK - 1) / SMALL_BK;
        if (fast_stage) {
#pragma unroll
            for (int s = 0; s < STAGES - 1; ++s) {
                if (s < num_k_tiles) {
                    stage_async(As_sbase + (unsigned)(s * A_STAGE * 2),
                        Bs_sbase + (unsigned)(s * B_STAGE * 2),
                        A, B, s * SMALL_BK, pid_m, pid_n, M, N, K, lda, ldb);
                }
                asm volatile("cp.async.commit_group;\n" ::);
            }
        } else {
            stage_scalar(As, Bs, A, B, 0, pid_m, pid_n, M, N, K, lda, ldb);
        }
        for (int kt = 0; kt < num_k_tiles; ++kt) {
            int read_buf = kt % STAGES;
            if (fast_stage) {
                // Wait until only the stages still being filled are in
                // flight, so the buffer about to be read has landed.
                asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
            }
            __syncthreads();
            int ahead = kt + STAGES - 1;
            if (fast_stage) {
                if (ahead < num_k_tiles) {
                    int write_buf = ahead % STAGES;
                    stage_async(As_sbase + (unsigned)(write_buf * A_STAGE * 2),
                        Bs_sbase + (unsigned)(write_buf * B_STAGE * 2),
                        A, B, ahead * SMALL_BK, pid_m, pid_n, M, N, K, lda, ldb);
                }
                asm volatile("cp.async.commit_group;\n" ::);
            } else if (kt + 1 < num_k_tiles) {
                int write_buf = (kt + 1) % STAGES;
                stage_scalar(As + write_buf * A_STAGE, Bs + write_buf * B_STAGE,
                    A, B, (kt + 1) * SMALL_BK, pid_m, pid_n, M, N, K, lda, ldb);
            }
            unsigned as_rd = As_sbase + (unsigned)(read_buf * A_STAGE * 2);
            unsigned bs_rd = Bs_sbase + (unsigned)(read_buf * B_STAGE * 2);
#pragma unroll
            for (int ks = 0; ks < SMALL_BK / 16; ++ks) {
                int k0 = ks * 16;
                unsigned a_frag[FM][4];
                unsigned b_frag[FN][2];
#pragma unroll
                for (int fm = 0; fm < FM; ++fm) {
                    int row = warpM + fm * 16 + lm_row_off + lm_r;
                    unsigned addr = as_rd + (unsigned)((row * SMALL_LDK + k0 + lm_col_off) * 2);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(a_frag[fm][0]), "=r"(a_frag[fm][1]),
                          "=r"(a_frag[fm][2]), "=r"(a_frag[fm][3]) : "r"(addr));
                }
#pragma unroll
                for (int fn = 0; fn < FN; ++fn) {
                    int row = k0 + lmb_row_off + lm_r;
                    unsigned addr = bs_rd + (unsigned)((row * LDB + warpN + fn * 8) * 2);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(b_frag[fn][0]), "=r"(b_frag[fn][1]) : "r"(addr));
                }
#pragma unroll
                for (int fm = 0; fm < FM; ++fm)
#pragma unroll
                    for (int fn = 0; fn < FN; ++fn)
                        SmallOps<T>::mma(acc[fm][fn], a_frag[fm], b_frag[fn]);
            }
        }
#pragma unroll
        for (int fm = 0; fm < FM; ++fm) {
#pragma unroll
            for (int fn = 0; fn < FN; ++fn) {
                int r0 = pid_m * TM + warpM + fm * 16 + g;
                int c0 = pid_n * TN + warpN + fn * 8 + 2 * t;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    int gr = r0 + (e >= 2 ? 8 : 0);
                    int gc = c0 + (e & 1);
                    if (gr >= M || gc >= N) continue;
                    float val = alpha * acc[fm][fn][e];
                    if (beta != 0.0f)
                        val += beta * to_f(C[(long long)gr * ldc + gc]);
                    C[(long long)gr * ldc + gc] = SmallOps<T>::from_float(val);
                }
            }
        }
    }
};

} // namespace sm89_half_small

#define SMALL_NT_EXPORT(NAME, TYPE, TM, TN, WM, WN, STAGES)                    \
extern "C" __global__ __launch_bounds__(WM * WN * 32, 2)                      \
void NAME(TYPE* __restrict__ C, const TYPE* __restrict__ A,                   \
          const TYPE* __restrict__ B, float alpha, int M, int N, int K_out) { \
    sm89_half_small::SmallNt<TYPE, TM, TN, WM, WN, STAGES>::run(              \
        C, A, B, alpha, M, N, K_out);                                         \
}

#define SMALL_NN_EXPORT(NAME, TYPE, TM, TN, WM, WN, STAGES)                    \
extern "C" __global__ __launch_bounds__(WM * WN * 32, 2)                      \
void NAME(TYPE* __restrict__ C, const TYPE* __restrict__ A,                   \
          const TYPE* __restrict__ B, const float* __restrict__ bias,         \
          Sm89HalfNnS3Params params) {                                        \
    sm89_half_small::SmallNn<TYPE, TM, TN, WM, WN, STAGES>::run(              \
        C, A, B, bias, params.alpha, params.beta, params.m, params.n,         \
        params.k, params.lda, params.ldb, params.ldc);                        \
}

SMALL_NT_EXPORT(nt_sm89_m16n64_bk64_s4_bf16, __nv_bfloat16, 16, 64, 1, 4, 4)
SMALL_NT_EXPORT(nt_sm89_m16n64_bk64_s4_f16, __half, 16, 64, 1, 4, 4)
SMALL_NN_EXPORT(nn_sm89_m16n64_bk64_s4_bf16, __nv_bfloat16, 16, 64, 1, 4, 4)
SMALL_NN_EXPORT(nn_sm89_m16n64_bk64_s4_f16, __half, 16, 64, 1, 4, 4)

#undef SMALL_NN_EXPORT
#undef SMALL_NT_EXPORT
#undef SMALL_LDK
#undef SMALL_BK
