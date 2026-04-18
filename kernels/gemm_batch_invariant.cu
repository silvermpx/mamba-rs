// Batch-invariant bf16/f16/f32 GEMM for Mamba inference + training.
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
//   smem  = (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * sizeof(T_IO) = 8 KB

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
    /* Cooperative global→smem load of a_row[0..K).                         */  \
    /* Vectorized 128-bit async loads via cp.async.cg (sm_80+).             */  \
    /* uint4 = 16 bytes = 8 bf16/f16 or 4 f32. Batch-invariance preserved:  */  \
    /* element values are identical to scalar path; smem write order        */  \
    /* does not affect later per-col K reduction.                           */  \
    {                                                                           \
        const int VEC = 16 / (int)sizeof(T_IO);                                 \
        const int k_vec = k / VEC;                                              \
        const uint4* a_vec =                                                    \
            reinterpret_cast<const uint4*>(a_row);                              \
        uint4* smem_a_vec = reinterpret_cast<uint4*>(smem_a);                   \
        _Pragma("unroll 1")                                                     \
        for (int i = threadIdx.x; i < k_vec; i += THREADS_PER_BLOCK) {          \
            __pipeline_memcpy_async(                                            \
                &smem_a_vec[i], &a_vec[i], sizeof(uint4));                      \
        }                                                                       \
        __pipeline_commit();                                                    \
        /* Scalar tail for k % VEC != 0 (still sync). */                        \
        const int k_tail_start = k_vec * VEC;                                   \
        for (int i = k_tail_start + threadIdx.x; i < k;                         \
             i += THREADS_PER_BLOCK) {                                          \
            smem_a[i] = a_row[i];                                               \
        }                                                                       \
        __pipeline_wait_prior(0);                                               \
    }                                                                           \
    __syncthreads();                                                            \
                                                                                \
    /* K-partition: each warp takes a contiguous range of K. */                 \
    const int k_per_warp = (k + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;         \
    const int k_start = warp_id * k_per_warp;                                   \
    const int k_stop = min(k, k_start + k_per_warp);                            \
                                                                                \
    /* OPT C: packed smem_a reads via pair_to_f2 helper. For bf16/f16 this  */ \
    /* halves LDS instruction count (1 LDS.U32 → 2 elements vs 2× LDS.U16).*/  \
    /* Address must be 4-byte aligned: kk steps by 8, smem_a starts at 0,  */  \
    /* so kk is always even and pair-aligned.                              */  \
    /* OPT E: tell ptxas K is a multiple of 8 (HF Mamba: d_model ∈ {768,   */  \
    /* 1024, 2048, 2560} all div by 8). Lets compiler drop the tail loop   */  \
    /* and keep `kk < k_main` purely as a single-strand bound.             */  \
    __builtin_assume((k & 7) == 0);                                             \
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
            float b0 = to_f(b[(kk    ) * ldb + col]);                           \
            float b1 = to_f(b[(kk + 1) * ldb + col]);                           \
            float b2 = to_f(b[(kk + 2) * ldb + col]);                           \
            float b3 = to_f(b[(kk + 3) * ldb + col]);                           \
            float b4 = to_f(b[(kk + 4) * ldb + col]);                           \
            float b5 = to_f(b[(kk + 5) * ldb + col]);                           \
            float b6 = to_f(b[(kk + 6) * ldb + col]);                           \
            float b7 = to_f(b[(kk + 7) * ldb + col]);                           \
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
