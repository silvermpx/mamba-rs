// Shared prelude for templated (multi-dtype) kernels.
//
// Pattern: every activation-touching kernel has 3 extern "C" instantiations:
//   NAME_f32, NAME_bf16, NAME_f16
// Suffix is chosen by Rust dispatch based on activation dtype.
//
// All math happens in f32; dtype conversion is upcast-on-load,
// downcast-on-store (single PTX cvt instruction each).
//
// Storage typing: T_IN / T_OUT are the activation dtype.
// Weights that must remain f32 (a_log/a_neg, D, norm weights, biases)
// are passed as `const float*` explicitly.

#ifndef _MAMBA_TYPED_PRELUDE_CUH
#define _MAMBA_TYPED_PRELUDE_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

// ---- Upcast helpers (load) ------------------------------------------------
__device__ __forceinline__ float to_f(float v)          { return v; }
__device__ __forceinline__ float to_f(__nv_bfloat16 v)  { return __bfloat162float(v); }
__device__ __forceinline__ float to_f(__half v)         { return __half2float(v); }

// ---- Downcast helpers (store) --------------------------------------------
__device__ __forceinline__ float         from_f_f32(float v)  { return v; }
__device__ __forceinline__ __nv_bfloat16 from_f_bf16(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ __half        from_f_f16(float v)  { return __float2half_rn(v); }

// ---- Packed pair upcast (2 elements at once) ------------------------------
// Halves LDS instruction count for warp-uniform smem reads. Used in matvec
// inner loop where smem_a reads are broadcast to all lanes in a warp.
// Address must be 4-byte aligned (2 elements × sizeof(T_IO)).
__device__ __forceinline__ float2 pair_to_f2(const float* p) {
    return {p[0], p[1]};
}
__device__ __forceinline__ float2 pair_to_f2(const __nv_bfloat16* p) {
    __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(p);
    return {__bfloat162float(__low2bfloat16(v)),
            __bfloat162float(__high2bfloat16(v))};
}
__device__ __forceinline__ float2 pair_to_f2(const __half* p) {
    __half2 v = *reinterpret_cast<const __half2*>(p);
    return {__half2float(__low2half(v)),
            __half2float(__high2half(v))};
}

#endif  // _MAMBA_TYPED_PRELUDE_CUH
// Batch-invariant deterministic GEMM — the multi-tile triad.
//
// This contract carries f32, bf16 and f16 on CUDA cores and Tensor Cores.
// The triad is the family with all three operand layouts and therefore the
// only one that can serve a backward. (See BiGemmFamily::Triad.)
//
// Based on siboehm's warptiling kernel (93.7% cuBLAS on A6000).
// Adapted for NVRTC compilation (no templates, no includes).
//
// Dtypes: the f32 kernels are the base contract; the bf16/f16 variants
// (typed section at the end of this file) share the same kernel
// structure with typed I/O and f32 accumulation throughout — a typed
// kernel is bit-identical to upcasting its inputs and running the f32
// kernel. The Tensor-Core variants further down are a SEPARATE numeric
// contract (mma.sync accumulation, not the scalar __fmaf_rn chain):
// deterministic and batch-invariant, but not bit-equal to the scalar
// triad. dW and bias stay f32 in every dtype (master-gradient
// invariant).
//
// Three operand layouts for training:
//   NN (forward):     C[M,N]  = alpha * A[M,K] @ B[K,N] + beta*C + bias
//   TN (backward dW): C[K,N] += alpha * A^T[K,M] @ B[M,N]
//   NT (backward dX): C[M,K]  = alpha * A[M,N] @ B^T[N,K]
//
// Architecture:
//   BM=128, BN=128, BK=16, 256 threads (8 warps)
//   Warp tile: WM=64, WN=32, arranged 2x4 (WMITER=2, WNITER=1 over 8 warps)
//   Thread tile: TM=8, TN=8
//   Per-thread output: 64 elements (16 rows x 4 cols × WMITER=2 = 64)
//   float4 coalesced global loads, A transposed in smem
//   GEMM_BI_GROUP_M per-arch L2 swizzle (8 sm_80, 16 sm_89+), deterministic K-reduction
//
// Source: github.com/siboehm/SGEMM_CUDA (kernel 10, warptiling)

// ============================================================================
// Typed (bf16/f16) variants — the sync-load buckets.
// ============================================================================
// Typed-triad contract:
//   - X / W / Y / dY / dX are T_ACT (typed I/O); loads upcast via to_f at the
//     read site, EXACTLY one RNE downcast (FROM_F) at the final store.
//   - dW and bias stay f32 (master gradients / f32 bias) — never rounded.
//   - All accumulation and the epilogue (alpha*acc + bias + beta*C) stay f32
//     with the same ascending-K __fmaf_rn chains and fixed reduce trees as
//     the f32 kernels: a typed kernel is bit-identical to "upcast inputs to
//     f32, run the f32 kernel" (bf16/f16 products are exact in f32).
//   - to_f / from_f_* come from _typed_prelude.cuh (inlined first in the
//     NVRTC blob; conversions are RNE, no FTZ — see kernels.rs flags).

// ============================================================================
// Tensor-core deterministic NN forward (bi_tensor_cores tier).
// ============================================================================
// mma.sync.aligned.m16n8k16 with f32 accumulators. SEPARATE numeric contract
// from the scalar triad (TC reduction tree, not the ascending-K FMA chain) —
// fully deterministic (fixed K order, fixed fragment/tile assignment, no
// atomics, no split-K) and batch-invariant across ALL M: each output
// element's entire K-reduction lives in one warp, independent of gridDim/M.
//
// cp.async implementation:
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
//     (modules.rs); launch passes the exact per-kernel byte count.
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
// CUDA parameter bundles use only 4-byte scalar fields. Each bundle also
// proves standard layout, member widths, alignment, and total size; together
// those checks prove exact declaration-order offsets without relying on
// offsetof, which NVRTC's standalone environment does not provide.
static_assert(sizeof(float) == 4, "CUDA float width changed");
static_assert(sizeof(int) == 4, "CUDA int width changed");

__device__ __forceinline__ float4 ld_global_L2_128B(const float* p) {
    float4 v;
    asm("ld.global.L2::128B.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(p));
    return v;
}
__device__ __forceinline__ bool gemm_bi_is_aligned_4(const void* ptr) {
    return ((unsigned long long)ptr & 3ULL) == 0;
}

__device__ __forceinline__ bool gemm_bi_is_aligned_8(const void* ptr) {
    return ((unsigned long long)ptr & 7ULL) == 0;
}

__device__ __forceinline__ bool gemm_bi_is_aligned_16(const void* ptr) {
    return ((unsigned long long)ptr & 15ULL) == 0;
}
__device__ __forceinline__ void gemm_bi_store_pair_rne(
    __half* dst, float x, float y) {
    *reinterpret_cast<__half2*>(dst) = __floats2half2_rn(x, y);
}

__device__ __forceinline__ void gemm_bi_store_pair_rne(
    __nv_bfloat16* dst, float x, float y) {
    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(x, y);
}

__device__ __forceinline__ void gemm_bi_accumulate_float2_or_scalar(
    float* dst, float x, float y, bool packed) {
    float current_x;
    float current_y;
    if (packed) {
        float2 current = *reinterpret_cast<const float2*>(dst);
        current_x = current.x;
        current_y = current.y;
    } else {
        current_x = dst[0];
        current_y = dst[1];
    }
    current_x += x;
    current_y += y;
    if (packed) {
        float2 result = {current_x, current_y};
        *reinterpret_cast<float2*>(dst) = result;
    } else {
        dst[0] = current_x;
        dst[1] = current_y;
    }
}

template <typename T>
__device__ __forceinline__ T* gemm_bi_output_start_if_valid(
    T* base, long long row_offset, int column, int extent) {
    if (column >= extent) return nullptr;
    return base + row_offset + column;
}

__device__ __forceinline__ int gemm_bi_cp_async_valid_elems(
    bool row_valid, int extent, int start) {
    if (!row_valid || start >= extent) return 0;
    int remaining = extent - start;
    return remaining < 8 ? remaining : 8;
}

template <typename T>
__device__ __forceinline__ const T* gemm_bi_cp_async_source(
    const T* base, long long valid_offset, int valid_bytes) {
    // PTX ignores the source for a zero-byte lane, but C++ still requires
    // the address expression itself to remain inside the allocation.
    return valid_bytes == 0 ? base : base + valid_offset;
}

__device__ __forceinline__ void gemm_bi_cp_async_16_zfill(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
}

__device__ __forceinline__ void gemm_bi_cp_async_16_zfill_l2(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
}
#include <cuda_awbarrier_primitives.h>
#define GEMM_BI_TC64_BM 64
#define GEMM_BI_TC64_BN 64
#define GEMM_BI_TC64_BK 64
#define GEMM_BI_TC64_THREADS 160
#define GEMM_BI_TC64_LDB 64
#define GEMM_BI_HALF_TN_INDEX(row, col) ((row) * 64 + ((col) ^ (((row) & 7) * 8)))
#define GEMM_BI_TC64_STAGE_TN_ASYNC(buf, mIdx, TT)                            \
    do {                                                                      \
        TT* _xs = &Xs[(buf)][0][0];                                           \
        TT* _ys = &Ys[(buf)][0][0];                                           \
        for (int _i = lane;                                                   \
             _i < GEMM_BI_TC64_BK * (GEMM_BI_TC64_BM / 8); _i += 32) {       \
            int _r = _i / (GEMM_BI_TC64_BM / 8);                              \
            int _c = (_i % (GEMM_BI_TC64_BM / 8)) * 8;                        \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * GEMM_BI_TC64_BM + _c;                           \
            int _elems = gemm_bi_cp_async_valid_elems(_gm < M_red, K_out, _gk); \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = (unsigned)__cvta_generic_to_shared(              \
                _xs + GEMM_BI_HALF_TN_INDEX(_r, _c));                         \
            long long _offset =                                               \
                _bytes == 0 ? 0 : (long long)_gm * K_out + _gk;               \
            const void* _src = gemm_bi_cp_async_source(A, _offset, _bytes);    \
            gemm_bi_cp_async_16_zfill_l2(_dst, _src, _bytes);                 \
        }                                                                     \
        for (int _i = lane;                                                   \
             _i < GEMM_BI_TC64_BK * (GEMM_BI_TC64_BN / 8); _i += 32) {       \
            int _r = _i / (GEMM_BI_TC64_BN / 8);                              \
            int _c = (_i % (GEMM_BI_TC64_BN / 8)) * 8;                        \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_TC64_BN + _c;                           \
            int _elems = gemm_bi_cp_async_valid_elems(_gm < M_red, N, _gn);   \
            int _bytes = _elems * 2;                                          \
            unsigned _dst = (unsigned)__cvta_generic_to_shared(              \
                _ys + GEMM_BI_HALF_TN_INDEX(_r, _c));                         \
            long long _offset = _bytes == 0 ? 0 : (long long)_gm * N + _gn;   \
            const void* _src = gemm_bi_cp_async_source(B, _offset, _bytes);    \
            gemm_bi_cp_async_16_zfill_l2(_dst, _src, _bytes);                 \
        }                                                                     \
        unsigned _filled = (unsigned)__cvta_generic_to_shared(&filled[buf]);  \
        asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];\n"            \
                     :: "r"(_filled) : "memory");                            \
        __mbarrier_arrive(&filled[buf]);                                      \
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
void gemm_bi_tn_test_tc64_bk64_s2_regpipe_vec2_ws5_##SUFFIX(                                                \
    float* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M_red, int K_out, int N                                                \
) {                                                                            \
    __shared__ __align__(16) T_ACT Xs[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB];           \
    __shared__ __align__(16) T_ACT Ys[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB];           \
    __shared__ __align__(8) __mbarrier_t pipeline_barriers[4];                 \
    __mbarrier_t* ready = &pipeline_barriers[0];                               \
    __mbarrier_t* filled = &pipeline_barriers[2];                              \
    int num_pid_n = (N + GEMM_BI_TC64_BN - 1) / GEMM_BI_TC64_BN;                       \
    int pid_m = blockIdx.x / num_pid_n;                                        \
    int pid_n = blockIdx.x % num_pid_n;                                        \
    int physical_warp = threadIdx.x / 32;                                      \
    int lane = threadIdx.x % 32;                                               \
    bool producer = physical_warp == 0;                                        \
    int warp = physical_warp - 1;                                              \
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
    int num_m_tiles = (M_red + GEMM_BI_TC64_BK - 1) / GEMM_BI_TC64_BK;     \
    if (!fast_stage || num_m_tiles == 0) return;                               \
    if (threadIdx.x < 4)                                                       \
        __mbarrier_init(&pipeline_barriers[threadIdx.x], GEMM_BI_TC64_THREADS); \
    __syncthreads();                                                           \
    if (!producer) {                                                           \
        __mbarrier_arrive(&ready[0]);                                          \
        __mbarrier_arrive(&ready[1]);                                          \
    }                                                                          \
    if (producer) {                                                            \
        for (int mt = 0; mt < num_m_tiles; ++mt) {                             \
            int write_buf = mt & 1;                                            \
            __mbarrier_token_t token = __mbarrier_arrive(&ready[write_buf]);   \
            while (!__mbarrier_test_wait(&ready[write_buf], token)) {}         \
            GEMM_BI_TC64_STAGE_TN_ASYNC(                                      \
                write_buf, mt * GEMM_BI_TC64_BK, T_ACT);                      \
        }                                                                      \
    } else {                                                                   \
        for (int mt = 0; mt < num_m_tiles; ++mt) {                             \
            int read_buf = mt & 1;                                             \
            __mbarrier_token_t token = __mbarrier_arrive(&filled[read_buf]);   \
            while (!__mbarrier_test_wait(&filled[read_buf], token)) {}         \
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
            __mbarrier_arrive(&ready[read_buf]);                              \
        }                                                                      \
    }                                                                          \
    if (producer) return;                                                      \
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