// Batch-invariant bf16/f16/f32 GEMM — the single FIXED-TILE family
// (BiGemmFamily::Fixed). Forward-only NN; the triad in gemm_bi_triad.cu
// carries the backward layouts.
//
// Problem: cuBLAS `cublasGemmEx` selects different algorithms per M
// (split-K, tile shape, reduction order). `Y = X @ W` at M=1 vs M=20
// produces sub-ULP differences that amplify through 24 SSM layers
// (observed KL ≈ 0.03, occasional top-1 flip on adversarial prompts).
//
// This kernel is batch-invariant by construction: fixed 64x64x32 tile,
// NO split-K, fixed K-reduction order. The K-reduction tree for
// `C[i, j]` depends ONLY on `A[i, :]` and `B[:, j]`, never on other
// rows of A. Therefore `C[i, j]` is bit-identical whether A has 1, 5,
// or 1024 rows.
//
// Inner GEMM uses Tensor Cores via nvcuda::wmma (m16n16k16 fragments,
// f32 accumulator). The MMA instruction itself is deterministic at
// the hardware level (MMA-Sim arXiv:2511.10909 — 1M random inputs
// bit-identical between simulator and hardware). Non-determinism in
// cuBLAS comes from heuristic algo/Split-K selection, NOT from MMA;
// fixing the tile + Split-K=1 + f32 accumulator is sufficient for
// batch invariance even with Tensor Cores enabled.
//
// Recipe matches vLLM `batch_invariant.bmm_kernel_persistent` and
// Thinking Machines Lab `batch_invariant_ops` (both Triton; the inner
// `tl.dot` lowers to `mma.sync` in PTX). Here transliterated to plain
// CUDA via the WMMA C++ API — mamba-rs has no Python/Triton dep.
//
//   BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32
//   GROUP_M = 8     (L2 swizzle)
//   SPLIT_K = 1     (critical — split-K is the root cause)
//   8 warps/CTA arranged 4 (warp_M) × 2 (warp_N)
//   Per warp: 16M × 32N = 1 frag-M × 2 frag-N (m16n16k16 each)
//   f32 accumulator fragments throughout
//
// Semantics (unchanged from CUDA-core version):
//   A: [M, K]  row-major, element type T_IO  (bf16 or f16)
//   B: [K, N]  row-major, element type T_IO
//   C: [M, N]  row-major, element type T_OUT (bf16 / f16 / f32)
//   bias: nullable [N] f32
//   C = alpha * (A @ B) + beta * C + bias
//
// Launch (unchanged — host dispatcher in src/mamba_ssm/gpu/blas.rs):
//   grid  = ((M/BLOCK_M) * (N/BLOCK_N), 1, 1)  flat — swizzled in-kernel
//   block = (256, 1, 1)
//   smem  = 0 dynamic (all buffers are static __shared__; the TC tile
//           holds 24 KB static - smem_a 4K + smem_b 4K + smem_acc 16K -
//           and the f32 FFMA tile 16 KB). The dynamic K-buffer belongs
//           to matvec_bi_* alone.

#include "_typed_prelude.cuh"
#include <mma.h>
#include <cuda_pipeline.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32
#define GROUP_M 8
#define THREADS 256
#define WARPS_PER_CTA 8     // THREADS / 32
#define WARPS_M 4           // 4 warps along M; 4*16 = 64 = BLOCK_M
#define WARPS_N 2           // 2 warps along N; 2*32 = 64 = BLOCK_N
#define FRAG_M 16
#define FRAG_N 16
#define FRAG_K 16
#define WARP_FRAGS_N 2      // each warp owns 2 N-fragments (covers 32 N cols)
#define K_TILES 2           // BLOCK_K / FRAG_K

using namespace nvcuda;

// --- typed zero helpers (used for OOB padding) --------------------------
__device__ __forceinline__ __nv_bfloat16 zero_bf16() {
    return __float2bfloat16(0.0f);
}
__device__ __forceinline__ __half zero_f16() {
    return __float2half(0.0f);
}
__device__ __forceinline__ float zero_f32() { return 0.0f; }

// --- f32 path: emulate Tensor Core via per-element FMA ------------------
// Tensor Cores on Ada do NOT accept f32 inputs (only bf16/f16/tf32). For
// the f32→f32 instantiation we keep the original CUDA-core inner loop.
// f32 inference was never the regression source (cuBLAS f32 path was also
// CUDA cores) so this path is unchanged in performance vs prior commit.
// --- f32 path: emulate Tensor Core via per-element FMA ------------------
// Tensor Cores on Ada do NOT accept f32 inputs (only bf16/f16/tf32). For
// the f32->f32 instantiation we keep the CUDA-core inner loop - the same
// hardware path cuBLAS takes for f32.
//
// The 64x64x32 tile with 256 threads and 4x4 outputs each is a measured
// optimum for narrow-N inference projections; wider tiles, fewer threads
// and a transposed-A layout were all measured slower at those shapes.
// The launcher's block/thread constants MUST equal the ones here - a
// launch that disagrees fills part of the tile and returns plausible
// garbage. `gemm_bi_fixed_correctness` is the gate.
#define DEFINE_GEMM_BI_FFMA(NAME, T_IO, T_OUT, FROM_F_OUT, ZERO_IO)             \
extern "C" __global__ __launch_bounds__(THREADS, 2) void                        \
NAME(                                                                           \
    T_OUT* __restrict__ c,                                                      \
    const T_IO* __restrict__ a,                                                 \
    const T_IO* __restrict__ b,                                                 \
    const float* __restrict__ bias,                                             \
    float alpha, float beta,                                                    \
    int m, int n, int k,                                                        \
    int lda, int ldb, int ldc                                                   \
) {                                                                             \
    __shared__ T_IO smem_a[BLOCK_M * BLOCK_K];                                  \
    __shared__ T_IO smem_b[BLOCK_K * BLOCK_N];                                  \
                                                                                \
    int num_pid_m = (m + BLOCK_M - 1) / BLOCK_M;                                \
    int num_pid_n = (n + BLOCK_N - 1) / BLOCK_N;                                \
    int num_pid_in_group = GROUP_M * num_pid_n;                                 \
    int group_id = blockIdx.x / num_pid_in_group;                               \
    int first_pid_m = group_id * GROUP_M;                                       \
    int group_size_m = min(num_pid_m - first_pid_m, GROUP_M);                   \
    int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m); \
    int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;                 \
                                                                                \
    int row0 = pid_m * BLOCK_M;                                                 \
    int col0 = pid_n * BLOCK_N;                                                 \
                                                                                \
    int tx = threadIdx.x & 15;                                                  \
    int ty = threadIdx.x >> 4;                                                  \
    int row_base = row0 + ty * 4;                                               \
    int col_base = col0 + tx * 4;                                               \
                                                                                \
    float acc[4][4];                                                            \
    _Pragma("unroll")                                                           \
    for (int i = 0; i < 4; i++) {                                               \
        _Pragma("unroll")                                                       \
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;                           \
    }                                                                           \
                                                                                \
    for (int k_tile = 0; k_tile < k; k_tile += BLOCK_K) {                       \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < 8; i++) {                                           \
            int idx = i * THREADS + threadIdx.x;                                \
            int smem_r = idx / BLOCK_K;                                         \
            int smem_c = idx % BLOCK_K;                                         \
            int g_r = row0 + smem_r;                                            \
            int g_c = k_tile + smem_c;                                          \
            smem_a[smem_r * BLOCK_K + smem_c] =                                 \
                (g_r < m && g_c < k) ? a[g_r * lda + g_c] : ZERO_IO();          \
        }                                                                       \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < 8; i++) {                                           \
            int idx = i * THREADS + threadIdx.x;                                \
            int smem_r = idx / BLOCK_N;                                         \
            int smem_c = idx % BLOCK_N;                                         \
            int g_r = k_tile + smem_r;                                          \
            int g_c = col0 + smem_c;                                            \
            smem_b[smem_r * BLOCK_N + smem_c] =                                 \
                (g_r < k && g_c < n) ? b[g_r * ldb + g_c] : ZERO_IO();          \
        }                                                                       \
        __syncthreads();                                                        \
                                                                                \
        _Pragma("unroll")                                                       \
        for (int kk = 0; kk < BLOCK_K; kk++) {                                  \
            float a_reg[4];                                                     \
            float b_reg[4];                                                     \
            _Pragma("unroll")                                                   \
            for (int i = 0; i < 4; i++)                                         \
                a_reg[i] = to_f(smem_a[(ty * 4 + i) * BLOCK_K + kk]);           \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < 4; j++)                                         \
                b_reg[j] = to_f(smem_b[kk * BLOCK_N + tx * 4 + j]);             \
            _Pragma("unroll")                                                   \
            for (int i = 0; i < 4; i++) {                                       \
                _Pragma("unroll")                                               \
                for (int j = 0; j < 4; j++) acc[i][j] += a_reg[i] * b_reg[j];   \
            }                                                                   \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
                                                                                \
    _Pragma("unroll")                                                           \
    for (int i = 0; i < 4; i++) {                                               \
        int r = row_base + i;                                                   \
        if (r >= m) continue;                                                   \
        _Pragma("unroll")                                                       \
        for (int j = 0; j < 4; j++) {                                           \
            int col = col_base + j;                                             \
            if (col >= n) continue;                                             \
            float val = alpha * acc[i][j];                                      \
            if (bias != nullptr) val += bias[col];                              \
            if (beta != 0.0f) val += beta * to_f(c[r * ldc + col]);             \
            c[r * ldc + col] = FROM_F_OUT(val);                                 \
        }                                                                       \
    }                                                                           \
}

// --- Tensor Core path (bf16 / f16) --------------------------------------
// Per-warp arrangement: 8 warps as 4 (along M) × 2 (along N).
//   warp_id = threadIdx.x / 32
//   warp_m  = warp_id / WARPS_N        (0..3)
//   warp_n  = warp_id % WARPS_N        (0..1)
// Each warp owns:
//   M rows: warp_m * 16  ..  warp_m * 16 + 16   (1 frag-M)
//   N cols: warp_n * 32  ..  warp_n * 32 + 32   (2 frag-N)
// K reduction: 2 inner iterations of 16 each.
//
// Smem A is [BLOCK_M, BLOCK_K] row-major; smem B is [BLOCK_K, BLOCK_N] row-major.
// `wmma::load_matrix_sync` reads with the given leading dimension; B is
// row-major from K's perspective so we use `wmma::row_major` and ldB=BLOCK_N.
#define DEFINE_GEMM_BI_TC(NAME, T_IO, T_OUT, FROM_F_OUT, ZERO_IO)               \
extern "C" __global__ __launch_bounds__(THREADS, 2) void                        \
NAME(                                                                           \
    T_OUT* __restrict__ c,                                                      \
    const T_IO* __restrict__ a,                                                 \
    const T_IO* __restrict__ b,                                                 \
    const float* __restrict__ bias,                                             \
    float alpha, float beta,                                                    \
    int m, int n, int k,                                                        \
    int lda, int ldb, int ldc                                                   \
) {                                                                             \
    __shared__ T_IO smem_a[BLOCK_M * BLOCK_K];                                  \
    __shared__ T_IO smem_b[BLOCK_K * BLOCK_N];                                  \
                                                                                \
    /* GROUP_M swizzle for L2 locality (vLLM recipe). */                        \
    int num_pid_m = (m + BLOCK_M - 1) / BLOCK_M;                                \
    int num_pid_n = (n + BLOCK_N - 1) / BLOCK_N;                                \
    int num_pid_in_group = GROUP_M * num_pid_n;                                 \
    int group_id = blockIdx.x / num_pid_in_group;                               \
    int first_pid_m = group_id * GROUP_M;                                       \
    int group_size_m = min(num_pid_m - first_pid_m, GROUP_M);                   \
    int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m); \
    int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;                 \
                                                                                \
    int row0 = pid_m * BLOCK_M;                                                 \
    int col0 = pid_n * BLOCK_N;                                                 \
                                                                                \
    int warp_id = threadIdx.x / 32;                                             \
    int warp_m = warp_id / WARPS_N;                                             \
    int warp_n = warp_id % WARPS_N;                                             \
                                                                                \
    /* f32 accumulator fragments — one per (warp_m row, warp_n col0/col1). */   \
    wmma::fragment<wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, float> acc_frag[WARP_FRAGS_N]; \
    _Pragma("unroll")                                                           \
    for (int j = 0; j < WARP_FRAGS_N; j++) wmma::fill_fragment(acc_frag[j], 0.0f); \
                                                                                \
    for (int k_tile = 0; k_tile < k; k_tile += BLOCK_K) {                       \
        /* Cooperatively load A tile [BLOCK_M, BLOCK_K] = 2048 elems / 256 threads = 8/thread */ \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < 8; i++) {                                           \
            int idx = i * THREADS + threadIdx.x;                                \
            int smem_r = idx / BLOCK_K;                                         \
            int smem_c = idx % BLOCK_K;                                         \
            int g_r = row0 + smem_r;                                            \
            int g_c = k_tile + smem_c;                                          \
            smem_a[smem_r * BLOCK_K + smem_c] =                                 \
                (g_r < m && g_c < k) ? a[g_r * lda + g_c] : ZERO_IO();          \
        }                                                                       \
        /* Cooperatively load B tile [BLOCK_K, BLOCK_N] = 2048 elems / 256 threads = 8/thread */ \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < 8; i++) {                                           \
            int idx = i * THREADS + threadIdx.x;                                \
            int smem_r = idx / BLOCK_N;                                         \
            int smem_c = idx % BLOCK_N;                                         \
            int g_r = k_tile + smem_r;                                          \
            int g_c = col0 + smem_c;                                            \
            smem_b[smem_r * BLOCK_N + smem_c] =                                 \
                (g_r < k && g_c < n) ? b[g_r * ldb + g_c] : ZERO_IO();          \
        }                                                                       \
        __syncthreads();                                                        \
                                                                                \
        /* Inner K loop: 2 frag-K iterations of 16 each. Tensor Core MMA.   */  \
        /* Reduction order is fixed (kk = 0, 1) and per-output independent  */  \
        /* of M — preserves cross-batch bit-identity.                       */  \
        _Pragma("unroll")                                                       \
        for (int kk = 0; kk < K_TILES; kk++) {                                  \
            wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K,              \
                T_IO, wmma::row_major> a_frag;                                  \
            wmma::load_matrix_sync(                                             \
                a_frag,                                                         \
                &smem_a[(warp_m * FRAG_M) * BLOCK_K + kk * FRAG_K],             \
                BLOCK_K);                                                       \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < WARP_FRAGS_N; j++) {                            \
                wmma::fragment<wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K,          \
                    T_IO, wmma::row_major> b_frag;                              \
                wmma::load_matrix_sync(                                         \
                    b_frag,                                                     \
                    &smem_b[(kk * FRAG_K) * BLOCK_N                             \
                            + warp_n * (FRAG_N * WARP_FRAGS_N) + j * FRAG_N],   \
                    BLOCK_N);                                                   \
                wmma::mma_sync(acc_frag[j], a_frag, b_frag, acc_frag[j]);       \
            }                                                                   \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
                                                                                \
    /* Epilogue. Stage f32 accumulator to a per-warp smem tile, then each   */  \
    /* thread does scalar (alpha, beta, bias) + cast and writes one element */  \
    /* of C. Reusing smem_a (>= 2048 f32 elements when sizeof(T_IO)>=2) for */  \
    /* the staging buffer; only valid when BLOCK_M*BLOCK_K*sizeof(T_IO) >=  */  \
    /* WARPS_PER_CTA * FRAG_M * (FRAG_N * WARP_FRAGS_N) * sizeof(float) =   */  \
    /* 8*16*32*4 = 16 KB. BLOCK_M*BLOCK_K*sizeof(bf16) = 64*32*2 = 4 KB —   */  \
    /* not enough. Use a dedicated f32 staging buffer instead.              */  \
    __shared__ float smem_acc[BLOCK_M * BLOCK_N];                               \
                                                                                \
    /* Each warp stores its 2 N-fragments into the per-warp slot.           */  \
    int warp_row0 = warp_m * FRAG_M;                                            \
    int warp_col0 = warp_n * (FRAG_N * WARP_FRAGS_N);                           \
    _Pragma("unroll")                                                           \
    for (int j = 0; j < WARP_FRAGS_N; j++) {                                    \
        wmma::store_matrix_sync(                                                \
            &smem_acc[warp_row0 * BLOCK_N + warp_col0 + j * FRAG_N],            \
            acc_frag[j],                                                        \
            BLOCK_N,                                                            \
            wmma::mem_row_major);                                               \
    }                                                                           \
    __syncthreads();                                                            \
                                                                                \
    /* Scalar epilogue: 4096 elems / 256 threads = 16 per thread.           */  \
    _Pragma("unroll")                                                           \
    for (int i = 0; i < 16; i++) {                                              \
        int idx = i * THREADS + threadIdx.x;                                    \
        int local_r = idx / BLOCK_N;                                            \
        int local_c = idx % BLOCK_N;                                            \
        int r = row0 + local_r;                                                 \
        int col = col0 + local_c;                                               \
        if (r >= m || col >= n) continue;                                       \
        float val = alpha * smem_acc[local_r * BLOCK_N + local_c];              \
        if (bias != nullptr) val += bias[col];                                  \
        if (beta != 0.0f) val += beta * to_f(c[r * ldc + col]);                 \
        c[r * ldc + col] = FROM_F_OUT(val);                                     \
    }                                                                           \
}

// Tensor-Core instantiations for half-precision paths (the regression source).
DEFINE_GEMM_BI_TC(gemm_bi_bf16_bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16, zero_bf16)
DEFINE_GEMM_BI_TC(gemm_bi_f16_f16,   __half,        __half,        from_f_f16,  zero_f16)
DEFINE_GEMM_BI_TC(gemm_bi_bf16_f32,  __nv_bfloat16, float,         from_f_f32,  zero_bf16)
DEFINE_GEMM_BI_TC(gemm_bi_f16_f32,   __half,        float,         from_f_f32,  zero_f16)

// f32 path stays on CUDA cores (Tensor Cores require fp16/bf16/tf32 inputs;
// converting f32→tf32 would lose 13 mantissa bits — not acceptable for the
// f32 training path that exists specifically because the user wants exact
// f32 math). cuBLAS f32 was never the regression source.
DEFINE_GEMM_BI_FFMA(gemm_bi_f32_f32, float, float, from_f_f32, zero_f32)

// ═════════════════════════════════════════════════════════════════════════
// M=1 specialized batch-invariant matvec — the decode hot path.
//
// Performance design (based on cuBLAS `gemvNSplitK` post-mortem):
// - Memory-bound regime: FLOPs / byte ≈ 1 at M=1, deep under Ada's
//   145 FLOP/byte bf16 TC roofline → HBM bandwidth is the ceiling.
//   Target: ≥ 80% of 960 GB/s = 770 GB/s effective → ≈ 1000 tok/s on
//   mamba-130m bf16 @ 130 GEMMs/tok.
// - **Split-K within block**: each CTA owns `BLOCK_N` cols but
//   partitions the K dimension across `WARPS_PER_BLOCK=4` warps. Each
//   warp does K/4 accumulation; partials are reduced via smem with a
//   fixed tree order (warp 0 += 1 += 2 += 3). This saturates SM
//   occupancy for small-N workloads (mamba-130m N ∈ {768, 1536, 3072}).
// - Each thread owns one output column within its warp's K-range.
// - `a[K]` cooperatively loaded into smem ONCE — reused by all warps.
// - K-loop unrolled by 8 for ILP (issues multiple B loads in flight).
// - B reads: 32 threads of a warp read 32 adjacent cols at same K →
//   coalesced 64-byte transaction (half cache line).
//
// Batch invariance:
//   y[n] = (((Σ₀ + Σ₁) + Σ₂) + Σ₃)  where Σᵢ = Σ_{k∈range_i} a[k]·b[k,n]
//   K partition depends ONLY on K (not M) since M=1 has no batch dim.
//   Fixed-order tree reduction across warps → bit-identical output
//   regardless of launch context.
// SplitK GEMV-over-rows. Each CTA owns one (m_row, col_chunk) tile.
// - BLOCK_N_MV=32 cols per CTA
// - K split across WARPS_PER_BLOCK=8 warps
// - Per-warp partials → smem → fixed-order tree reduce in warp 0
//
// Grid: (ceil(N / BLOCK_N_MV), M, 1) — blockIdx.y = row index.
// Works for any M ≥ 1:
// - Decode (M=1): single row, 72 CTAs on Ada (~1000 tok/s).
// - RL/prefill (M>1): each row computed by its own CTAs. Per-row output
//   is bit-identical to the M=1 case because each row's K-reduction is
//   independent (SPLIT_K=1 across rows, fixed tree across warps within
//   one row). This is what guarantees cross-batch parity.
//
// Trade-off at large M: B is streamed once per row (not shared across M
// rows like cuBLAS's M-tiled GEMM). For small M (RL N_envs up to ~16)
// the bandwidth cost is acceptable given the batch-invariance guarantee.
#define BLOCK_N_MV 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (BLOCK_N_MV * WARPS_PER_BLOCK)
#define DEFINE_MATVEC_BI(NAME, T_IO, T_OUT, FROM_F_OUT)                         \
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 6) void              \
NAME(                                                                           \
    T_OUT* __restrict__ c,                                                      \
    const T_IO* __restrict__ a,                                                 \
    const T_IO* __restrict__ b,                                                 \
    const float* __restrict__ bias,                                             \
    float alpha, float beta,                                                    \
    int m, int n, int k,                                                        \
    int lda /* row-stride of A, == k */,                                        \
    int ldb,                                                                    \
    int ldc /* row-stride of C, == n */                                         \
) {                                                                             \
    const int lane = threadIdx.x & 31;                                          \
    const int warp_id = threadIdx.x >> 5;                                       \
    const int col = blockIdx.x * BLOCK_N_MV + lane;                             \
    const int row = blockIdx.y;  /* which row of A / C we compute */            \
    const bool in_range = (col < n) && (row < m);                               \
                                                                                \
    /* Shift A, C to our row. (row < m assumed by grid launch.)  */             \
    const T_IO* a_row = a + row * lda;                                          \
    T_OUT* c_row = c + row * ldc;                                               \
                                                                                \
    /* Static smem: 2 KB for partials (8 warps × 32 cols × 4 bytes) +       */  \
    /* dynamic smem for a[K]. Separating them avoids offset arithmetic bugs.*/  \
    __shared__ float smem_partials[WARPS_PER_BLOCK * BLOCK_N_MV];               \
    extern __shared__ unsigned char smem_a_raw[];                               \
    T_IO* smem_a = reinterpret_cast<T_IO*>(smem_a_raw);                         \
                                                                                \
    /* Cooperative global->smem load of a_row[0..K). Vectorized 128-bit */ \
    /* cp.async.cg (sm_80+) when EVERY A row is 16-byte aligned (row     */ \
    /* offset = row*k*sizeof(T_IO)); an arbitrary K (e.g. a vision patch */ \
    /* dim) can misalign rows - those take the all-scalar fill. smem     */ \
    /* contents are identical either way, so the per-col K reduction     */ \
    /* (and its bits) never depends on the path taken.                   */ \
    {                                                                       \
        const int VEC = 16 / (int)sizeof(T_IO);                             \
        if ((k * (int)sizeof(T_IO)) % 16 == 0 &&                            \
            (lda * (int)sizeof(T_IO)) % 16 == 0) {                          \
            /* Both gates matter: with lda != k a 16-byte-aligned K can  */ \
            /* still start a_row off-alignment (row * lda), and the      */ \
            /* uint4 reinterpret would fault. Misaligned rows take the   */ \
            /* all-scalar fill; smem contents are identical either way.  */ \
            const int k_vec = k / VEC;                                      \
            const uint4* a_vec =                                            \
                reinterpret_cast<const uint4*>(a_row);                      \
            uint4* smem_a_vec = reinterpret_cast<uint4*>(smem_a);           \
            _Pragma("unroll 1")                                             \
            for (int i = threadIdx.x; i < k_vec; i += THREADS_PER_BLOCK) {  \
                __pipeline_memcpy_async(                                    \
                    &smem_a_vec[i], &a_vec[i], sizeof(uint4));              \
            }                                                               \
            __pipeline_commit();                                            \
            __pipeline_wait_prior(0);                                       \
        } else {                                                            \
            for (int i = threadIdx.x; i < k; i += THREADS_PER_BLOCK) {      \
                smem_a[i] = a_row[i];                                       \
            }                                                               \
        }                                                                   \
    }                                                                       \
    __syncthreads();                                                        \
                                                                            \
    /* K-partition: each warp takes a contiguous range of K, rounded up  */ \
    /* to an EVEN span so every warp's k_start stays pair-aligned for    */ \
    /* the packed smem reads below. Even spans (every historic HF        */ \
    /* d_model) are unchanged - bit-stability preserved; odd spans       */ \
    /* previously FAULTED (misaligned LDS.U32 at an odd element offset). */ \
    int k_per_warp = (k + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;           \
    k_per_warp += (k_per_warp & 1);                                         \
    const int k_start = warp_id * k_per_warp;                               \
    const int k_stop = min(k, k_start + k_per_warp);                        \
                                                                                \
    /* Packed smem_a reads via the pair_to_f2 helper: for bf16/f16 one   */ \
    /* LDS.U32 delivers two elements where two LDS.U16 did one each.     */ \
    /* Address must be 4-byte aligned: kk steps by 8, smem_a starts at 0,  */  \
    /* so kk is always even and pair-aligned.                              */  \
    /* (An earlier __builtin_assume((k & 7) == 0) was removed: it was UB */ \
    /* for arbitrary K - vision patch dims - and the scalar tail below   */ \
    /* already handles k % 8 != 0 correctly at negligible cost.)         */ \
    float acc = 0.0f;                                                           \
    if (in_range && k_start < k_stop) {                                         \
        int kk = k_start;                                                       \
        int k_main = k_start + (((k_stop - k_start) >> 3) << 3);                \
        for (; kk < k_main; kk += 8) {                                          \
            float2 a01 = pair_to_f2(&smem_a[kk    ]);                           \
            float2 a23 = pair_to_f2(&smem_a[kk + 2]);                           \
            float2 a45 = pair_to_f2(&smem_a[kk + 4]);                           \
            float2 a67 = pair_to_f2(&smem_a[kk + 6]);                           \
            float a0 = a01.x, a1 = a01.y;                                       \
            float a2 = a23.x, a3 = a23.y;                                       \
            float a4 = a45.x, a5 = a45.y;                                       \
            float a6 = a67.x, a7 = a67.y;                                       \
            /* __ldcs streaming loads bypass L1, leaving the cache to A   */  \
            /* (reused across decode steps); B is read once per token and */  \
            /* gains nothing from caching.                                */  \
            float b0 = to_f(__ldcs(&b[(kk    ) * ldb + col]));                  \
            float b1 = to_f(__ldcs(&b[(kk + 1) * ldb + col]));                  \
            float b2 = to_f(__ldcs(&b[(kk + 2) * ldb + col]));                  \
            float b3 = to_f(__ldcs(&b[(kk + 3) * ldb + col]));                  \
            float b4 = to_f(__ldcs(&b[(kk + 4) * ldb + col]));                  \
            float b5 = to_f(__ldcs(&b[(kk + 5) * ldb + col]));                  \
            float b6 = to_f(__ldcs(&b[(kk + 6) * ldb + col]));                  \
            float b7 = to_f(__ldcs(&b[(kk + 7) * ldb + col]));                  \
            acc = fmaf(a0, b0, acc);                                            \
            acc = fmaf(a1, b1, acc);                                            \
            acc = fmaf(a2, b2, acc);                                            \
            acc = fmaf(a3, b3, acc);                                            \
            acc = fmaf(a4, b4, acc);                                            \
            acc = fmaf(a5, b5, acc);                                            \
            acc = fmaf(a6, b6, acc);                                            \
            acc = fmaf(a7, b7, acc);                                            \
        }                                                                       \
        for (; kk < k_stop; kk++) {                                             \
            acc = fmaf(to_f(smem_a[kk]), to_f(b[kk * ldb + col]), acc);         \
        }                                                                       \
    }                                                                           \
                                                                                \
    /* Store partial. All threads participate (out-of-range → 0). */            \
    smem_partials[warp_id * BLOCK_N_MV + lane] = acc;                           \
    __syncthreads();                                                            \
                                                                                \
    /* Warp 0 gathers partials for its col and does fixed-order tree reduce. */ \
    if (warp_id == 0 && in_range) {                                             \
        float p0 = smem_partials[0 * BLOCK_N_MV + lane];                        \
        float p1 = smem_partials[1 * BLOCK_N_MV + lane];                        \
        float p2 = smem_partials[2 * BLOCK_N_MV + lane];                        \
        float p3 = smem_partials[3 * BLOCK_N_MV + lane];                        \
        float p4 = smem_partials[4 * BLOCK_N_MV + lane];                        \
        float p5 = smem_partials[5 * BLOCK_N_MV + lane];                        \
        float p6 = smem_partials[6 * BLOCK_N_MV + lane];                        \
        float p7 = smem_partials[7 * BLOCK_N_MV + lane];                        \
        float s01 = p0 + p1;                                                    \
        float s23 = p2 + p3;                                                    \
        float s45 = p4 + p5;                                                    \
        float s67 = p6 + p7;                                                    \
        float s0123 = s01 + s23;                                                \
        float s4567 = s45 + s67;                                                \
        float sum = s0123 + s4567;                                              \
        float val = alpha * sum;                                                \
        if (bias != nullptr) val += bias[col];                                  \
        if (beta != 0.0f) val += beta * to_f(c_row[col]);                       \
        c_row[col] = FROM_F_OUT(val);                                           \
    }                                                                           \
}

DEFINE_MATVEC_BI(matvec_bi_bf16_bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16)
DEFINE_MATVEC_BI(matvec_bi_f16_f16,   __half,        __half,        from_f_f16)
DEFINE_MATVEC_BI(matvec_bi_bf16_f32,  __nv_bfloat16, float,         from_f_f32)
DEFINE_MATVEC_BI(matvec_bi_f16_f32,   __half,        float,         from_f_f32)
DEFINE_MATVEC_BI(matvec_bi_f32_f32,   float,         float,         from_f_f32)

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

// ---------------------------------------------------------------------------
// GBF safety layer (adopted from the audited fix set; every item is
// bit-preserving - the packed store uses the same per-element RNE, the
// cp.async source clamp changes address formation only, and the base-
// alignment gates route misaligned operands to the scalar stage that
// produces identical smem bytes):
//   - a 16-byte cp.async source pointer is FORMED only when bytes > 0
//     (out-of-object address formation is UB even unread);
//   - the fast stage requires 16B-aligned operand BASES, not just
//     8-element strides (a 2-byte-aligned typed subview with an even
//     stride otherwise stages wrong bytes silently);
//   - the packed pair store fires only on a 4-byte-aligned destination
//     (PTX faults or silently masks misaligned 32-bit stores).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ bool gbf_aligned16(const void* p) {
    return (reinterpret_cast<unsigned long long>(p) & 15ull) == 0ull;
}
static __device__ __forceinline__ bool gbf_aligned4(const void* p) {
    return (reinterpret_cast<unsigned long long>(p) & 3ull) == 0ull;
}
static __device__ __forceinline__ void gbf_store_pair_rne(
    __nv_bfloat16* dst, float v0, float v1) {
    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
}
static __device__ __forceinline__ void gbf_store_pair_rne(
    __half* dst, float v0, float v1) {
    *reinterpret_cast<__half2*>(dst) = __floats2half2_rn(v0, v1);
}


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
                float v0 = alpha * acc[fm][fn][2 * half];                     \
                float v1 = alpha * acc[fm][fn][2 * half + 1];                 \
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
                float val = alpha * acc[fm][fn][e];                            \
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
            float val = alpha * acc[e];                                        \
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
