// The tc64 TN dW body over the stream-K schedule, composed into the portable
// module for every sm80-family target except CC 12.x: those boards run the
// SM120 TMA stream-K kernel, and leaving this fragment out keeps the portable
// module they compile byte-identical to the one their TF32 cohort was frozen
// against. Every constant here is fragment-local (GEMM_BI_SK64_*); the body
// mirrors the tc64 TN kernel of sm80.cu instruction for instruction, so a
// single worker reproduces that kernel bit for bit.

#define GEMM_BI_SK64_BM 64
#define GEMM_BI_SK64_BN 64
#define GEMM_BI_SK64_BK 64
#define GEMM_BI_SK64_THREADS 128
#define GEMM_BI_SK64_LDB (GEMM_BI_SK64_BN + 8) /* 72 halves = 144 B rows, 36 words == 4 mod 8 */

#define GEMM_BI_SK64_STAGE_TN_ASYNC(buf, mIdx)                                    \
    do {                                                                      \
        unsigned _xs =                                                        \
            Xs_sbase + (unsigned)((buf) * GEMM_BI_SK64_BK * GEMM_BI_SK64_LDB * 2);    \
        unsigned _ys =                                                        \
            Ys_sbase + (unsigned)((buf) * GEMM_BI_SK64_BK * GEMM_BI_SK64_LDB * 2);    \
        for (int _i = threadIdx.x; _i < GEMM_BI_SK64_BK * (GEMM_BI_SK64_BM / 8);      \
             _i += GEMM_BI_SK64_THREADS) {                                        \
            int _r = _i / (GEMM_BI_SK64_BM / 8);                                  \
            int _c = (_i % (GEMM_BI_SK64_BM / 8)) * 8;                            \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * GEMM_BI_SK64_BM + _c;                               \
            int _elems = cp_async_valid_elems(_gm < M_red, K_out, _gk);   \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _xs + (unsigned)((_r * GEMM_BI_SK64_LDB + _c) * 2);   \
            long long _offset =                                                \
                _bytes == 0 ? 0 : (long long)_gm * K_out + _gk;               \
            const void* _src = cp_async_source(A, _offset, _bytes);        \
            cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_SK64_BK * (GEMM_BI_SK64_BN / 8);      \
             _i += GEMM_BI_SK64_THREADS) {                                        \
            int _r = _i / (GEMM_BI_SK64_BN / 8);                                  \
            int _c = (_i % (GEMM_BI_SK64_BN / 8)) * 8;                            \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_SK64_BN + _c;                               \
            int _elems = cp_async_valid_elems(_gm < M_red, N, _gn);       \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = _ys + (unsigned)((_r * GEMM_BI_SK64_LDB + _c) * 2);   \
            long long _offset = _bytes == 0 ? 0 : (long long)_gm * N + _gn;   \
            const void* _src = cp_async_source(B, _offset, _bytes);        \
            cp_async_16_zfill(_dst, _src, _bytes);                         \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define GEMM_BI_SK64_STAGE_TN_SCALAR(buf, mIdx, TT, FF)                           \
    do {                                                                      \
        TT* _xs = &Xs[buf][0][0];                                             \
        TT* _ys = &Ys[buf][0][0];                                             \
        for (int _i = threadIdx.x; _i < GEMM_BI_SK64_BK * GEMM_BI_SK64_BM;            \
             _i += GEMM_BI_SK64_THREADS) {                                        \
            int _r = _i / GEMM_BI_SK64_BM;                                        \
            int _c = _i % GEMM_BI_SK64_BM;                                        \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * GEMM_BI_SK64_BM + _c;                               \
            _xs[_r * GEMM_BI_SK64_LDB + _c] = (_gm < M_red && _gk < K_out)        \
                                              ? A[(long long)_gm * K_out + _gk]\
                                              : FF(0.0f);                     \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GEMM_BI_SK64_BK * GEMM_BI_SK64_BN;            \
             _i += GEMM_BI_SK64_THREADS) {                                        \
            int _r = _i / GEMM_BI_SK64_BN;                                        \
            int _c = _i % GEMM_BI_SK64_BN;                                        \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_SK64_BN + _c;                               \
            _ys[_r * GEMM_BI_SK64_LDB + _c] = (_gm < M_red && _gn < N)            \
                                              ? B[(long long)_gm * N + _gn]   \
                                              : FF(0.0f);                     \
        }                                                                     \
    } while (0)

// ----------------------------------------------------------------------------
// TN dW over the stream-K schedule: the tc64 body on a persistent grid.
// ----------------------------------------------------------------------------
// A training batch runs the dW reduction ten thousand rows deep over a few
// dozen 64x64 output tiles, so the tiled kernel above leaves most of the
// device idle (a 384x384 output is 36 CTAs on 142 multiprocessors). Here the
// (tile, m-slab) units are dealt by the formula the SM120 stream-K kernels
// use to a number of workers the host fixes per architecture, never to the
// grid, so every card of an architecture folds alike and gives the same
// bits. Each resident CTA walks the workers in ascending order, every
// grid-th one; a worker computes each segment of its range with the tc64
// mainloop (ascending slabs, the same mma chain per element), stores a
// partial slab where its range ends inside a tile, and the worker holding a
// tile's last unit folds the lower contributors' slabs in ascending worker
// order before the ordinary accumulate epilogue. The output is bit-stable
// for a shape on an architecture but differs from the tiled reduction: the
// kernel carries its own numeric contract and is never dispatched under the
// tiled parity policy. With one worker every tile is one segment walked in
// ascending order and the output equals the tiled kernel bit for bit; the
// tests pin that.
//
// Flags form a (worker, slot) matrix that is zero before the first launch;
// every consumer clears the flags it waited on, so the matrix is zero again
// when the kernel exits. Waits target lower workers only, and every worker
// below a waiting one is done or held by a running CTA, so with the grid no
// larger than the device keeps resident no wait can starve.

struct GemmBiStreamKRange {
    long long first;
    long long last;
};

// Units [first, last) of worker `worker`: the first `remainder` workers take
// one unit more, so every worker differs from any other by at most one unit.
static __device__ __forceinline__ GemmBiStreamKRange streamk_range(
    long long units, int workers, int worker) {
    long long base = units / workers;
    long long remainder = units % workers;
    long long first = (long long)worker * base + min((long long)worker, remainder);
    long long last = first + base + (worker < remainder ? 1 : 0);
    GemmBiStreamKRange range = {first, last};
    return range;
}

// The worker whose range contains `unit`: the inverse of the dealing formula.
static __device__ __forceinline__ int streamk_worker_of(
    long long units, int workers, long long unit) {
    long long base = units / workers;
    long long remainder = units % workers;
    long long wide = remainder * (base + 1);
    if (unit < wide) return (int)(unit / (base + 1));
    return (int)(remainder + (unit - wide) / base);
}

#define GEMM_BI_SK64_STREAMK_SLOTS 2
#define GEMM_BI_SK64_STREAMK_ACCUMULATORS 32

// One slab per (worker, slot): every thread's 2 x 4 x 4 accumulators as eight
// float4 stores straight to L2, where the folding worker reads them.
static __device__ __forceinline__ void tc64_streamk_store_slab(
    float* slab, const float (&acc)[2][4][4]) {
    float4* destination = reinterpret_cast<float4*>(
        slab + (long long)threadIdx.x * GEMM_BI_SK64_STREAMK_ACCUMULATORS);
#pragma unroll
    for (int fm = 0; fm < 2; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            asm volatile("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};\n"
                :: "l"(destination + fm * 4 + fn),
                   "f"(acc[fm][fn][0]), "f"(acc[fm][fn][1]),
                   "f"(acc[fm][fn][2]), "f"(acc[fm][fn][3]) : "memory");
        }
    }
}

static __device__ __forceinline__ void streamk_raise(unsigned* flag) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;\n" :: "l"(flag), "r"(1U) : "memory");
}

static __device__ __forceinline__ void streamk_await(const unsigned* flag) {
    unsigned value;
    do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(value) : "l"(flag) : "memory");
    } while (value == 0U);
}

static __device__ __forceinline__ void streamk_clear(unsigned* flag) {
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;\n" :: "l"(flag), "r"(0U) : "memory");
}

// Folds the lower contributors' slabs into the accumulators: the slabs are
// summed in ascending worker order, then the worker's own value is added
// last, every sum an __fadd_rn. Slabs are read through cache-global loads so
// the eight loads of one slab issue together and bypass the incoherent L1.
static __device__ __forceinline__ void tc64_streamk_fold_slabs(
    const float* partial, long long units, int workers, int first_worker, int worker,
    int tile, int k_tiles, float (&acc)[2][4][4]) {
    const long long slab_floats =
        (long long)GEMM_BI_SK64_THREADS * GEMM_BI_SK64_STREAMK_ACCUMULATORS;
    const long long lane_offset =
        (long long)threadIdx.x * GEMM_BI_SK64_STREAMK_ACCUMULATORS;
    float sum[2][4][4];
#pragma unroll
    for (int fm = 0; fm < 2; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
#pragma unroll
            for (int e = 0; e < 4; ++e) sum[fm][fn][e] = 0.0f;
        }
    }
    bool first = true;
    for (int source = first_worker; source < worker; ++source) {
        GemmBiStreamKRange theirs = streamk_range(units, workers, source);
        int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;
        const float4* slab = reinterpret_cast<const float4*>(
            partial + ((long long)source * GEMM_BI_SK64_STREAMK_SLOTS + slot) * slab_floats
            + lane_offset);
        float4 value[2][4];
#pragma unroll
        for (int fm = 0; fm < 2; ++fm) {
#pragma unroll
            for (int fn = 0; fn < 4; ++fn) {
                value[fm][fn] = __ldcg(slab + fm * 4 + fn);
            }
        }
#pragma unroll
        for (int fm = 0; fm < 2; ++fm) {
#pragma unroll
            for (int fn = 0; fn < 4; ++fn) {
                float4 v = value[fm][fn];
                if (first) {
                    sum[fm][fn][0] = v.x;
                    sum[fm][fn][1] = v.y;
                    sum[fm][fn][2] = v.z;
                    sum[fm][fn][3] = v.w;
                } else {
                    sum[fm][fn][0] = __fadd_rn(sum[fm][fn][0], v.x);
                    sum[fm][fn][1] = __fadd_rn(sum[fm][fn][1], v.y);
                    sum[fm][fn][2] = __fadd_rn(sum[fm][fn][2], v.z);
                    sum[fm][fn][3] = __fadd_rn(sum[fm][fn][3], v.w);
                }
            }
        }
        first = false;
    }
#pragma unroll
    for (int fm = 0; fm < 2; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[fm][fn][e] = __fadd_rn(sum[fm][fn][e], acc[fm][fn][e]);
            }
        }
    }
}

#define GEMM_BI_DEFINE_GEMM_BI_TN_TC64_STREAMK(SUFFIX, T_ACT, FROM_F, MMA_T)          \
extern "C" __global__ __launch_bounds__(GEMM_BI_SK64_THREADS, 1)                   \
void tn_tc64_streamk_##SUFFIX(                                        \
    float* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M_red, int K_out, int N,                                               \
    float* __restrict__ partial,                                               \
    unsigned* __restrict__ flags,                                              \
    int workers                                                                \
) {                                                                            \
    __shared__ __align__(16) T_ACT Xs[2][GEMM_BI_SK64_BK][GEMM_BI_SK64_LDB];           \
    __shared__ __align__(16) T_ACT Ys[2][GEMM_BI_SK64_BK][GEMM_BI_SK64_LDB];           \
    int num_pid_n = (N + GEMM_BI_SK64_BN - 1) / GEMM_BI_SK64_BN;                       \
    int num_pid_m = (K_out + GEMM_BI_SK64_BM - 1) / GEMM_BI_SK64_BM;                   \
    int k_tiles = (M_red + GEMM_BI_SK64_BK - 1) / GEMM_BI_SK64_BK;                     \
    long long units = (long long)num_pid_m * num_pid_n * k_tiles;              \
    const long long slab_floats =                                              \
        (long long)GEMM_BI_SK64_THREADS * GEMM_BI_SK64_STREAMK_ACCUMULATORS;   \
    int warp = threadIdx.x / 32;                                               \
    int lane = threadIdx.x % 32;                                               \
    int warpM = (warp / 2) * 32;                                               \
    int warpN = (warp % 2) * 32;                                               \
    int g = lane >> 2;                                                         \
    int t = lane & 3;                                                          \
    int lm_r = lane & 7;                                                       \
    int lm_q = lane >> 3;                                                      \
    int lm_arow_off = (lm_q & 2) ? 8 : 0;                                      \
    int lm_acol_off = (lm_q & 1) ? 8 : 0;                                      \
    int lm_brow_off = (lm_q & 1) ? 8 : 0;                                      \
    unsigned Xs_sbase = (unsigned)__cvta_generic_to_shared(&Xs[0][0][0]);      \
    unsigned Ys_sbase = (unsigned)__cvta_generic_to_shared(&Ys[0][0][0]);      \
    bool fast_stage = is_aligned_16(A) && is_aligned_16(B) &&         \
                      ((K_out & 7) == 0) && ((N & 7) == 0);                    \
    float acc[2][4][4];                                                        \
    int pid_m = 0;                                                             \
    int pid_n = 0;                                                             \
    for (int worker = (int)blockIdx.x; worker < workers;                       \
         worker += (int)gridDim.x) {                                           \
    GemmBiStreamKRange mine = streamk_range(units, workers, worker);           \
    int range_first = (int)mine.first;                                         \
    int range_end = (int)mine.last;                                            \
    int first_tile = k_tiles > 0 ? range_first / k_tiles : 0;                  \
    for (int unit = range_end; unit > range_first;) {                          \
        int tile = (unit - 1) / k_tiles;                                       \
        int k_end = unit - tile * k_tiles;                                     \
        int k_begin = max(0, k_end - (unit - range_first));                    \
        pid_m = tile / num_pid_n;                                              \
        pid_n = tile % num_pid_n;                                              \
        _Pragma("unroll")                                                      \
        for (int fm = 0; fm < 2; fm++)                                         \
            _Pragma("unroll")                                                  \
            for (int fn = 0; fn < 4; fn++)                                     \
                _Pragma("unroll")                                              \
                for (int e = 0; e < 4; e++) acc[fm][fn][e] = 0.0f;             \
        /* the previous segment's fragment loads are done before its smem */  \
        /* is restaged */                                                      \
        __syncthreads();                                                       \
        if (fast_stage) {                                                      \
            GEMM_BI_SK64_STAGE_TN_ASYNC(0, k_begin * GEMM_BI_SK64_BK);          \
        } else {                                                               \
            GEMM_BI_SK64_STAGE_TN_SCALAR(0, k_begin * GEMM_BI_SK64_BK, T_ACT, FROM_F); \
        }                                                                      \
        int read_buf = 0;                                                      \
        for (int mt = k_begin; mt < k_end; mt++) {                             \
            if (fast_stage) {                                                  \
                asm volatile("cp.async.wait_group 0;\n");                      \
            }                                                                  \
            __syncthreads();                                                   \
            if (mt + 1 < k_end) {                                              \
                if (fast_stage) {                                              \
                    GEMM_BI_SK64_STAGE_TN_ASYNC(read_buf ^ 1, (mt + 1) * GEMM_BI_SK64_BK); \
                } else {                                                       \
                    GEMM_BI_SK64_STAGE_TN_SCALAR(read_buf ^ 1, (mt + 1) * GEMM_BI_SK64_BK, \
                                             T_ACT, FROM_F);                   \
                }                                                              \
            }                                                                  \
            unsigned Xs_rd =                                                   \
                Xs_sbase + (unsigned)(read_buf * GEMM_BI_SK64_BK * GEMM_BI_SK64_LDB * 2); \
            unsigned Ys_rd =                                                   \
                Ys_sbase + (unsigned)(read_buf * GEMM_BI_SK64_BK * GEMM_BI_SK64_LDB * 2); \
            _Pragma("unroll")                                                  \
            for (int ks = 0; ks < (GEMM_BI_SK64_BK / 16); ks++) {              \
                int k0 = ks * 16;                                              \
                unsigned a_frag[2][4];                                         \
                unsigned b_frag[4][2];                                         \
                _Pragma("unroll")                                              \
                for (int fm = 0; fm < 2; fm++) {                               \
                    int srow = k0 + lm_arow_off + lm_r;                        \
                    int scol = warpM + fm * 16 + lm_acol_off;                  \
                    unsigned addr =                                            \
                        Xs_rd + (unsigned)((srow * GEMM_BI_SK64_LDB + scol) * 2); \
                    asm volatile(                                              \
                        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "      \
                        "{%0,%1,%2,%3}, [%4];\n"                               \
                        : "=r"(a_frag[fm][0]), "=r"(a_frag[fm][1]),            \
                          "=r"(a_frag[fm][2]), "=r"(a_frag[fm][3])             \
                        : "r"(addr));                                          \
                }                                                              \
                _Pragma("unroll")                                              \
                for (int fn = 0; fn < 4; fn++) {                               \
                    int srow = k0 + lm_brow_off + lm_r;                        \
                    unsigned addr = Ys_rd +                                    \
                        (unsigned)((srow * GEMM_BI_SK64_LDB + warpN + fn * 8) * 2); \
                    asm volatile(                                              \
                        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "      \
                        "{%0,%1}, [%2];\n"                                     \
                        : "=r"(b_frag[fn][0]), "=r"(b_frag[fn][1])             \
                        : "r"(addr));                                          \
                }                                                              \
                _Pragma("unroll")                                              \
                for (int fm = 0; fm < 2; fm++) {                               \
                    _Pragma("unroll")                                          \
                    for (int fn = 0; fn < 4; fn++) {                           \
                        asm volatile(                                          \
                            "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "." \
                            MMA_T ".f32 "                                      \
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "          \
                            "{%0,%1,%2,%3};\n"                                 \
                            : "+f"(acc[fm][fn][0]), "+f"(acc[fm][fn][1]),      \
                              "+f"(acc[fm][fn][2]), "+f"(acc[fm][fn][3])       \
                            : "r"(a_frag[fm][0]), "r"(a_frag[fm][1]),          \
                              "r"(a_frag[fm][2]), "r"(a_frag[fm][3]),          \
                              "r"(b_frag[fn][0]), "r"(b_frag[fn][1]));         \
                    }                                                          \
                }                                                              \
            }                                                                  \
            read_buf ^= 1;                                                     \
        }                                                                      \
        bool covers_start = k_begin == 0;                                      \
        bool covers_end = k_end == k_tiles;                                    \
        if (!covers_end) {                                                     \
            int slot = tile == first_tile ? 0 : 1;                             \
            tc64_streamk_store_slab(                                   \
                partial + ((long long)worker * GEMM_BI_SK64_STREAMK_SLOTS + slot) * slab_floats, \
                acc);                                                          \
            __threadfence();                                                   \
            __syncthreads();                                                   \
            if (threadIdx.x == 0) {                                            \
                streamk_raise(                                         \
                    flags + (long long)worker * GEMM_BI_SK64_STREAMK_SLOTS + slot); \
            }                                                                  \
            unit -= k_end - k_begin;                                           \
            continue;                                                          \
        }                                                                      \
        if (!covers_start) {                                                   \
            int first_worker =                                                 \
                streamk_worker_of(units, workers, (long long)tile * k_tiles);  \
            int sources = worker - first_worker;                               \
            for (int index = (int)threadIdx.x; index < sources;                \
                 index += GEMM_BI_SK64_THREADS) {                              \
                int source = first_worker + index;                             \
                GemmBiStreamKRange theirs = streamk_range(units, workers, source); \
                int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;     \
                streamk_await(                                         \
                    flags + (long long)source * GEMM_BI_SK64_STREAMK_SLOTS + slot); \
            }                                                                  \
            __syncthreads();                                                   \
            tc64_streamk_fold_slabs(                                   \
                partial, units, workers, first_worker, worker, tile, k_tiles, acc); \
            __syncthreads();                                                   \
            for (int index = (int)threadIdx.x; index < sources;                \
                 index += GEMM_BI_SK64_THREADS) {                              \
                int source = first_worker + index;                             \
                GemmBiStreamKRange theirs = streamk_range(units, workers, source); \
                int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;     \
                streamk_clear(                                         \
                    flags + (long long)source * GEMM_BI_SK64_STREAMK_SLOTS + slot); \
            }                                                                  \
        }                                                                      \
        /* epilogue: f32 accumulate into dW, as the tiled kernel */            \
        _Pragma("unroll")                                                      \
        for (int fm = 0; fm < 2; fm++) {                                       \
            _Pragma("unroll")                                                  \
            for (int fn = 0; fn < 4; fn++) {                                   \
                int r0 = pid_m * GEMM_BI_SK64_BM + warpM + fm * 16 + g;        \
                int c0 = pid_n * GEMM_BI_SK64_BN + warpN + fn * 8 + 2 * t;     \
                _Pragma("unroll")                                              \
                for (int e = 0; e < 4; e++) {                                  \
                    int gr = r0 + (e >= 2 ? 8 : 0);                            \
                    int gc = c0 + (e & 1);                                     \
                    if (gr >= K_out || gc >= N) continue;                      \
                    C[(long long)gr * N + gc] += alpha * acc[fm][fn][e];       \
                }                                                              \
            }                                                                  \
        }                                                                      \
        unit -= k_end - k_begin;                                               \
    }                                                                          \
    }                                                                          \
}

GEMM_BI_DEFINE_GEMM_BI_TN_TC64_STREAMK(bf16, __nv_bfloat16, from_f_bf16, "bf16")
GEMM_BI_DEFINE_GEMM_BI_TN_TC64_STREAMK(f16,  __half,        from_f_f16,  "f16")

#undef GEMM_BI_SK64_STREAMK_SLOTS
#undef GEMM_BI_SK64_STREAMK_ACCUMULATORS

#undef GEMM_BI_SK64_STAGE_TN_ASYNC
#undef GEMM_BI_SK64_STAGE_TN_SCALAR
#undef GEMM_BI_SK64_BM
#undef GEMM_BI_SK64_BN
#undef GEMM_BI_SK64_BK
#undef GEMM_BI_SK64_THREADS
#undef GEMM_BI_SK64_LDB
