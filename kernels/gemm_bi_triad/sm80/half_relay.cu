// Half-precision TN weight gradient over a persistent relay schedule.
//
// The tiled 64x64 kernel gives every output tile one CTA that walks the whole
// reduction, so a tile count that is not a multiple of the resident CTA count
// leaves a tail wave: a third of the board idles while the busiest
// multiprocessors set the time. Here the grid is two CTAs per multiprocessor
// and every CTA walks a contiguous range of (tile, slab) units dealt by the
// stream-K formula. The numeric contract stays that of the tiled kernel:
// each output element is one ascending chain of m16n8k16 accumulations over
// the slabs, and when a tile's chain crosses a CTA boundary the earlier CTA
// hands its f32 accumulators over untouched through a global slab and the
// later CTA continues the same chain from them. No partial sums are ever
// added together, so the result is bit for bit the tiled result for any grid.
//
// A CTA processes its range from the highest unit down: the segment that
// starts a later tile needs nothing and is finished first, and the segment
// that finishes an earlier tile waits on the lower CTA only after the CTA's
// own free work, which is at least as long whenever the range holds a whole
// tile's worth of slabs. Waits target lower CTAs only and every CTA of the
// grid is resident, so no wait can starve.
//
// Flags form one word per CTA that is zero before the first launch; the
// consumer clears the flag it waited on, so the array is zero again when the
// kernel exits. The partial slabs hold one CTA's accumulators each.

namespace sm89_half_relay {

#define RELAY_BM 64
#define RELAY_BN 64
#define RELAY_BK 64
// Swizzled slab index: 16-byte chunks of a 64-element-deep row are XORed
// with the row's low three bits so ldmatrix reads of eight consecutive rows
// hit distinct bank groups.
#define RELAY_INDEX(row, col, width) ((row) * (width) + ((col) ^ (((row) & 7) * 8)))

struct RelayRange {
    long long first;
    long long last;
};

// Units [first, last) of CTA `cta`: the first `remainder` CTAs take one unit
// more, so every CTA differs from any other by at most one unit.
static __device__ __forceinline__ RelayRange relay_range(
    long long units, int grid, int cta) {
    long long base = units / grid;
    long long remainder = units % grid;
    long long first = (long long)cta * base + min((long long)cta, remainder);
    long long last = first + base + (cta < remainder ? 1 : 0);
    RelayRange range = {first, last};
    return range;
}

static __device__ __forceinline__ void relay_raise(unsigned* flag) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;\n" :: "l"(flag), "r"(1U) : "memory");
}

static __device__ __forceinline__ void relay_await(const unsigned* flag) {
    unsigned value;
    do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(value) : "l"(flag) : "memory");
    } while (value == 0U);
}

static __device__ __forceinline__ void relay_clear(unsigned* flag) {
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;\n" :: "l"(flag), "r"(0U) : "memory");
}

static __device__ __forceinline__ bool relay_aligned_16(const void* pointer) {
    return ((unsigned long long)pointer & 15ULL) == 0ULL;
}

template <typename T> struct RelayOps;

#define RELAY_OPS(TYPE, FROM, MMA_TYPE)                                       \
template <> struct RelayOps<TYPE> {                                           \
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

RELAY_OPS(__nv_bfloat16, from_f_bf16, "bf16")
RELAY_OPS(__half, from_f_f16, "f16")
#undef RELAY_OPS

// One slab of X (reduction rows x 64 output rows) and one of dY (reduction
// rows x 64 output columns) land in the XOR-swizzled layout the tiled kernel
// uses; out-of-range rows and columns arrive as zeros through the zero-fill
// form of cp.async, exactly as the tiled kernel stages them.
template <int THREADS>
static __device__ __forceinline__ void relay_stage_async(
    unsigned xs, unsigned ys, const void* A, const void* B, int elem_bytes,
    int m_base, int pid_m, int pid_n, int M_red, int K_out, int N) {
    for (int i = (int)threadIdx.x; i < RELAY_BK * (RELAY_BM / 8); i += THREADS) {
        int r = i / (RELAY_BM / 8);
        int c = (i % (RELAY_BM / 8)) * 8;
        int gm = m_base + r;
        int gk = pid_m * RELAY_BM + c;
        int elems = cp_async_valid_elems(gm < M_red, K_out, gk);
        int bytes = elems * elem_bytes;
        unsigned dst = xs + (unsigned)((RELAY_INDEX(r, c, RELAY_BM)) * elem_bytes);
        long long offset = bytes == 0 ? 0 : (long long)gm * K_out + gk;
        const char* src = bytes == 0 ? (const char*)A : (const char*)A + offset * elem_bytes;
        cp_async_16_zfill(dst, src, bytes);
    }
    for (int i = (int)threadIdx.x; i < RELAY_BK * (RELAY_BN / 8); i += THREADS) {
        int r = i / (RELAY_BN / 8);
        int c = (i % (RELAY_BN / 8)) * 8;
        int gm = m_base + r;
        int gn = pid_n * RELAY_BN + c;
        int elems = cp_async_valid_elems(gm < M_red, N, gn);
        int bytes = elems * elem_bytes;
        unsigned dst = ys + (unsigned)((RELAY_INDEX(r, c, RELAY_BN)) * elem_bytes);
        long long offset = bytes == 0 ? 0 : (long long)gm * N + gn;
        const char* src = bytes == 0 ? (const char*)B : (const char*)B + offset * elem_bytes;
        cp_async_16_zfill(dst, src, bytes);
    }
}

template <typename T, int THREADS>
static __device__ __forceinline__ void relay_stage_scalar(
    T* xs, T* ys, const T* A, const T* B,
    int m_base, int pid_m, int pid_n, int M_red, int K_out, int N) {
    for (int i = (int)threadIdx.x; i < RELAY_BK * RELAY_BM; i += THREADS) {
        int r = i / RELAY_BM;
        int c = i % RELAY_BM;
        int gm = m_base + r;
        int gk = pid_m * RELAY_BM + c;
        xs[RELAY_INDEX(r, c, RELAY_BM)] = (gm < M_red && gk < K_out)
            ? A[(long long)gm * K_out + gk] : RelayOps<T>::from_float(0.0f);
    }
    for (int i = (int)threadIdx.x; i < RELAY_BK * RELAY_BN; i += THREADS) {
        int r = i / RELAY_BN;
        int c = i % RELAY_BN;
        int gm = m_base + r;
        int gn = pid_n * RELAY_BN + c;
        ys[RELAY_INDEX(r, c, RELAY_BN)] = (gm < M_red && gn < N)
            ? B[(long long)gm * N + gn] : RelayOps<T>::from_float(0.0f);
    }
}

// Warp geometry: two by two warps of 32x32, each two m16 fragments by four
// n8 fragments, over a three-stage cp.async ring.
template <typename T, int STAGES>
struct RelayKernel {
    static constexpr int WARPS_M = 2;
    static constexpr int WARPS_N = 2;
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
    static constexpr int WARP_M = RELAY_BM / WARPS_M;
    static constexpr int WARP_N = RELAY_BN / WARPS_N;
    static constexpr int FM = WARP_M / 16;
    static constexpr int FN = WARP_N / 8;
    static constexpr int ACC_FLOATS = FM * FN * 4;
    static constexpr int SLAB_FLOATS = THREADS * ACC_FLOATS;
    static constexpr int A_STAGE = RELAY_BK * RELAY_BM;
    static constexpr int B_STAGE = RELAY_BK * RELAY_BN;
    static constexpr int SHARED_BYTES = STAGES * (A_STAGE + B_STAGE) * (int)sizeof(T);

    static __device__ __forceinline__ void load_fragments(
        unsigned xs_rd, unsigned ys_rd, int k0, int warpM, int warpN,
        int lm_r, int lm_arow_off, int lm_acol_off, int lm_brow_off,
        unsigned (&a_frag)[FM][4], unsigned (&b_frag)[FN][2]) {
#pragma unroll
        for (int fm = 0; fm < FM; ++fm) {
            int srow = k0 + lm_arow_off + lm_r;
            int scol = warpM + fm * 16 + lm_acol_off;
            unsigned addr = xs_rd + (unsigned)((RELAY_INDEX(srow, scol, RELAY_BM)) * (int)sizeof(T));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_frag[fm][0]), "=r"(a_frag[fm][1]),
                  "=r"(a_frag[fm][2]), "=r"(a_frag[fm][3])
                : "r"(addr));
        }
#pragma unroll
        for (int fn = 0; fn < FN; ++fn) {
            int srow = k0 + lm_brow_off + lm_r;
            unsigned addr = ys_rd + (unsigned)((RELAY_INDEX(srow, warpN + fn * 8, RELAY_BN)) * (int)sizeof(T));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_frag[fn][0]), "=r"(b_frag[fn][1])
                : "r"(addr));
        }
    }

    static __device__ __forceinline__ void run(
        float* __restrict__ C, const T* __restrict__ A, const T* __restrict__ B,
        float alpha, int M_red, int K_out, int N,
        float* __restrict__ partial, unsigned* __restrict__ flags) {
        extern __shared__ __align__(16) unsigned char relay_shared[];
        T* Xs = reinterpret_cast<T*>(relay_shared);
        T* Ys = Xs + STAGES * A_STAGE;
        unsigned Xs_sbase = (unsigned)__cvta_generic_to_shared(Xs);
        unsigned Ys_sbase = (unsigned)__cvta_generic_to_shared(Ys);

        int num_pid_n = (N + RELAY_BN - 1) / RELAY_BN;
        int num_pid_m = (K_out + RELAY_BM - 1) / RELAY_BM;
        int k_tiles = (M_red + RELAY_BK - 1) / RELAY_BK;
        long long units = (long long)num_pid_m * num_pid_n * k_tiles;
        int grid = (int)gridDim.x;
        int cta = (int)blockIdx.x;
        RelayRange mine = relay_range(units, grid, cta);
        int range_first = (int)mine.first;
        int range_end = (int)mine.last;

        int warp = (int)threadIdx.x / 32;
        int lane = (int)threadIdx.x % 32;
        int warpM = (warp / WARPS_N) * WARP_M;
        int warpN = (warp % WARPS_N) * WARP_N;
        int g = lane >> 2;
        int t = lane & 3;
        int lm_r = lane & 7;
        int lm_q = lane >> 3;
        int lm_arow_off = (lm_q & 2) ? 8 : 0;
        int lm_acol_off = (lm_q & 1) ? 8 : 0;
        int lm_brow_off = (lm_q & 1) ? 8 : 0;
        bool fast_stage = relay_aligned_16(A) && relay_aligned_16(B)
            && ((K_out & 7) == 0) && ((N & 7) == 0);
        float* own_slab = partial + (long long)cta * SLAB_FLOATS
            + (long long)threadIdx.x * ACC_FLOATS;

        float acc[FM][FN][4];
        for (int unit = range_end; unit > range_first;) {
            int tile = (unit - 1) / k_tiles;
            int k_end = unit - tile * k_tiles;
            int k_begin = max(0, k_end - (unit - range_first));
            int pid_m = tile / num_pid_n;
            int pid_n = tile % num_pid_n;
            bool covers_start = k_begin == 0;
            bool covers_end = k_end == k_tiles;

            if (covers_start) {
#pragma unroll
                for (int fm = 0; fm < FM; ++fm)
#pragma unroll
                    for (int fn = 0; fn < FN; ++fn)
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[fm][fn][e] = 0.0f;
            } else {
                // The lower CTA finished the slabs before k_begin of this
                // tile; continue its accumulators exactly where it stopped.
                if (threadIdx.x == 0) relay_await(flags + (cta - 1));
                __syncthreads();
                const float4* theirs = reinterpret_cast<const float4*>(
                    partial + (long long)(cta - 1) * SLAB_FLOATS
                    + (long long)threadIdx.x * ACC_FLOATS);
#pragma unroll
                for (int fm = 0; fm < FM; ++fm) {
#pragma unroll
                    for (int fn = 0; fn < FN; ++fn) {
                        float4 v = __ldcg(theirs + fm * FN + fn);
                        acc[fm][fn][0] = v.x;
                        acc[fm][fn][1] = v.y;
                        acc[fm][fn][2] = v.z;
                        acc[fm][fn][3] = v.w;
                    }
                }
                __syncthreads();
                if (threadIdx.x == 0) relay_clear(flags + (cta - 1));
            }

            // The previous segment's fragment reads are done before its
            // shared stages are restaged.
            __syncthreads();
            int segment_tiles = k_end - k_begin;
            if (fast_stage) {
#pragma unroll
                for (int s = 0; s < STAGES - 1; ++s) {
                    if (s < segment_tiles) {
                        relay_stage_async<THREADS>(
                            Xs_sbase + (unsigned)(s * A_STAGE * (int)sizeof(T)),
                            Ys_sbase + (unsigned)(s * B_STAGE * (int)sizeof(T)),
                            A, B, (int)sizeof(T), (k_begin + s) * RELAY_BK,
                            pid_m, pid_n, M_red, K_out, N);
                    }
                    asm volatile("cp.async.commit_group;\n" ::);
                }
            } else {
                relay_stage_scalar<T, THREADS>(Xs, Ys, A, B, k_begin * RELAY_BK,
                    pid_m, pid_n, M_red, K_out, N);
            }

            for (int step = 0; step < segment_tiles; ++step) {
                int read_buf = step % STAGES;
                if (fast_stage) {
                    // Wait until only the stages still being filled are in
                    // flight, so the buffer about to be read has landed.
                    asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
                }
                __syncthreads();
                int ahead = step + STAGES - 1;
                if (fast_stage) {
                    if (ahead < segment_tiles) {
                        int write_buf = ahead % STAGES;
                        relay_stage_async<THREADS>(
                            Xs_sbase + (unsigned)(write_buf * A_STAGE * (int)sizeof(T)),
                            Ys_sbase + (unsigned)(write_buf * B_STAGE * (int)sizeof(T)),
                            A, B, (int)sizeof(T), (k_begin + ahead) * RELAY_BK,
                            pid_m, pid_n, M_red, K_out, N);
                    }
                    asm volatile("cp.async.commit_group;\n" ::);
                } else if (step + 1 < segment_tiles) {
                    int write_buf = (step + 1) % STAGES;
                    relay_stage_scalar<T, THREADS>(
                        Xs + write_buf * A_STAGE, Ys + write_buf * B_STAGE,
                        A, B, (k_begin + step + 1) * RELAY_BK, pid_m, pid_n, M_red, K_out, N);
                }
                unsigned xs_rd = Xs_sbase + (unsigned)(read_buf * A_STAGE * (int)sizeof(T));
                unsigned ys_rd = Ys_sbase + (unsigned)(read_buf * B_STAGE * (int)sizeof(T));
                unsigned a_frag[2][FM][4];
                unsigned b_frag[2][FN][2];
                load_fragments(xs_rd, ys_rd, 0, warpM, warpN, lm_r, lm_arow_off,
                    lm_acol_off, lm_brow_off, a_frag[0], b_frag[0]);
#pragma unroll
                for (int ks = 0; ks < RELAY_BK / 16; ++ks) {
                    if (ks + 1 < RELAY_BK / 16) {
                        load_fragments(xs_rd, ys_rd, (ks + 1) * 16, warpM, warpN, lm_r,
                            lm_arow_off, lm_acol_off, lm_brow_off,
                            a_frag[(ks + 1) & 1], b_frag[(ks + 1) & 1]);
                    }
#pragma unroll
                    for (int fm = 0; fm < FM; ++fm)
#pragma unroll
                        for (int fn = 0; fn < FN; ++fn)
                            RelayOps<T>::mma(acc[fm][fn], a_frag[ks & 1][fm], b_frag[ks & 1][fn]);
                }
            }

            if (!covers_end) {
                // The chain continues in the next CTA: publish the
                // accumulators as they are.
                float4* destination = reinterpret_cast<float4*>(own_slab);
#pragma unroll
                for (int fm = 0; fm < FM; ++fm) {
#pragma unroll
                    for (int fn = 0; fn < FN; ++fn) {
                        asm volatile("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};\n"
                            :: "l"(destination + fm * FN + fn),
                               "f"(acc[fm][fn][0]), "f"(acc[fm][fn][1]),
                               "f"(acc[fm][fn][2]), "f"(acc[fm][fn][3]) : "memory");
                    }
                }
                __threadfence();
                __syncthreads();
                if (threadIdx.x == 0) relay_raise(flags + cta);
            } else {
                // The tiled kernel's accumulate epilogue.
#pragma unroll
                for (int fm = 0; fm < FM; ++fm) {
#pragma unroll
                    for (int fn = 0; fn < FN; ++fn) {
                        int r0 = pid_m * RELAY_BM + warpM + fm * 16 + g;
                        int c0 = pid_n * RELAY_BN + warpN + fn * 8 + 2 * t;
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            int gr = r0 + (e >= 2 ? 8 : 0);
                            int gc = c0 + (e & 1);
                            if (gr >= K_out || gc >= N) continue;
                            C[(long long)gr * N + gc] += alpha * acc[fm][fn][e];
                        }
                    }
                }
            }
            unit -= segment_tiles;
        }
    }
};

}  // namespace sm89_half_relay

#define HALF_RELAY_EXPORT(NAME, TYPE, STAGES)                                  \
extern "C" __global__ __launch_bounds__(128, 1)                               \
void NAME(float* __restrict__ C, const TYPE* __restrict__ A,                  \
          const TYPE* __restrict__ B, float alpha,                            \
          int M_red, int K_out, int N,                                        \
          float* __restrict__ partial, unsigned* __restrict__ flags) {        \
    sm89_half_relay::RelayKernel<TYPE, STAGES>::run(                          \
        C, A, B, alpha, M_red, K_out, N, partial, flags);                     \
}

HALF_RELAY_EXPORT(tn_sm89_relay_m64n64_bk64_s3_bf16, __nv_bfloat16, 3)
HALF_RELAY_EXPORT(tn_sm89_relay_m64n64_bk64_s3_f16, __half, 3)

#undef HALF_RELAY_EXPORT
#undef RELAY_INDEX
#undef RELAY_BK
#undef RELAY_BN
#undef RELAY_BM
