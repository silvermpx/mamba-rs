// Batch-invariant bf16/f16 GEMM for Mamba inference + training.
//
// Problem: cuBLAS `cublasGemmEx` selects different algorithms per M
// (split-K, tile shape, reduction order). `Y = X @ W` at M=1 vs M=20
// produces sub-ULP differences that amplify through 24 SSM layers
// (observed KL ≈ 0.03, occasional top-1 flip on adversarial prompts).
//
// This kernel is batch-invariant by construction: fixed 64x64x32 tile,
// NO split-K, sequential K-reduction in each thread. The K-reduction
// tree for `C[i, j]` depends ONLY on `A[i, :]` and `B[:, j]`, never on
// other rows of A. Therefore `C[i, j]` is bit-identical whether A has
// 1, 5, or 1024 rows.
//
// Recipe: vLLM `batch_invariant.bmm_kernel_persistent` + Thinking
// Machines Lab `batch_invariant_ops` (both Triton; transliterated to
// plain CUDA here — mamba-rs has no Python/Triton dependency).
//
//   BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32
//   GROUP_M = 8    (L2 swizzle)
//   SPLIT_K = 1    (critical — split-K is the root cause)
//   256 threads/block, each owns 4x4 micro-tile of C
//   f32 accumulate (correct by construction — register scalars are f32)
//
// Semantics:
//   A: [M, K]  row-major, element type T_IO  (bf16 or f16)
//   B: [K, N]  row-major, element type T_IO
//   C: [M, N]  row-major, element type T_OUT (bf16 / f16 / f32)
//   bias: nullable [N] f32
//   C = alpha * (A @ B) + beta * C + bias
//
// Launch:
//   grid  = ((M/BLOCK_M) * (N/BLOCK_N), 1, 1)  flat — swizzled in-kernel
//   block = (256, 1, 1)
//   smem  = (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * sizeof(T_IO)  = 8 KB

#include "_typed_prelude.cuh"

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32
#define GROUP_M 8
#define THREADS 256
#define MICRO_M 4
#define MICRO_N 4

// --- typed zero helpers (used for OOB padding) --------------------------
__device__ __forceinline__ __nv_bfloat16 zero_bf16() {
    return __float2bfloat16(0.0f);
}
__device__ __forceinline__ __half zero_f16() {
    return __float2half(0.0f);
}

#define DEFINE_GEMM_BI(NAME, T_IO, T_OUT, FROM_F_OUT, ZERO_IO)                  \
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
    int tx = threadIdx.x & 15;                                                  \
    int ty = threadIdx.x >> 4;                                                  \
    int row_base = row0 + ty * MICRO_M;                                         \
    int col_base = col0 + tx * MICRO_N;                                         \
                                                                                \
    float acc[MICRO_M][MICRO_N];                                                \
    _Pragma("unroll")                                                           \
    for (int i = 0; i < MICRO_M; i++) {                                         \
        _Pragma("unroll")                                                       \
        for (int j = 0; j < MICRO_N; j++) {                                     \
            acc[i][j] = 0.0f;                                                   \
        }                                                                       \
    }                                                                           \
                                                                                \
    for (int k_tile = 0; k_tile < k; k_tile += BLOCK_K) {                       \
        /* Load A tile [BLOCK_M, BLOCK_K] — 2048 elems / 256 threads = 8/thread */ \
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
        /* Load B tile [BLOCK_K, BLOCK_N] — 2048 elems / 256 threads = 8/thread */ \
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
        /* Inner K loop: BLOCK_K=32 iterations. This is the FIXED reduction    \
           order — same at any M. f32 accumulate in registers. */               \
        _Pragma("unroll")                                                       \
        for (int kk = 0; kk < BLOCK_K; kk++) {                                  \
            float a_reg[MICRO_M];                                               \
            float b_reg[MICRO_N];                                               \
            _Pragma("unroll")                                                   \
            for (int i = 0; i < MICRO_M; i++) {                                 \
                a_reg[i] = to_f(smem_a[(ty * MICRO_M + i) * BLOCK_K + kk]);     \
            }                                                                   \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < MICRO_N; j++) {                                 \
                b_reg[j] = to_f(smem_b[kk * BLOCK_N + tx * MICRO_N + j]);       \
            }                                                                   \
            _Pragma("unroll")                                                   \
            for (int i = 0; i < MICRO_M; i++) {                                 \
                _Pragma("unroll")                                               \
                for (int j = 0; j < MICRO_N; j++) {                             \
                    acc[i][j] += a_reg[i] * b_reg[j];                           \
                }                                                               \
            }                                                                   \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
                                                                                \
    /* Epilogue: alpha*acc + beta*C_prev + bias */                              \
    _Pragma("unroll")                                                           \
    for (int i = 0; i < MICRO_M; i++) {                                         \
        int r = row_base + i;                                                   \
        if (r >= m) continue;                                                   \
        _Pragma("unroll")                                                       \
        for (int j = 0; j < MICRO_N; j++) {                                     \
            int col = col_base + j;                                             \
            if (col >= n) continue;                                             \
            float val = alpha * acc[i][j];                                      \
            if (bias != nullptr) val += bias[col];                              \
            if (beta != 0.0f) {                                                 \
                val += beta * to_f(c[r * ldc + col]);                           \
            }                                                                   \
            c[r * ldc + col] = FROM_F_OUT(val);                                 \
        }                                                                       \
    }                                                                           \
}

// Instantiations covering Mamba inference + training:
//   bf16 × bf16 → bf16   (projections in bf16 path)
//   f16  × f16  → f16    (projections in f16 path)
//   bf16 × bf16 → f32    (tied lm_head: logits always f32 for sampling)
//   f16  × f16  → f32    (tied lm_head f16)
//   f32  × f32  → f32    (sgemm path — optional, adds batch-invariance
//                         to training batch-size changes too)
DEFINE_GEMM_BI(gemm_bi_bf16_bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16, zero_bf16)
DEFINE_GEMM_BI(gemm_bi_f16_f16,   __half,        __half,        from_f_f16,  zero_f16)
DEFINE_GEMM_BI(gemm_bi_bf16_f32,  __nv_bfloat16, float,         from_f_f32,  zero_bf16)
DEFINE_GEMM_BI(gemm_bi_f16_f32,   __half,        float,         from_f_f32,  zero_f16)

// f32 path: need a trivial zero helper.
__device__ __forceinline__ float zero_f32() { return 0.0f; }
DEFINE_GEMM_BI(gemm_bi_f32_f32,   float,         float,         from_f_f32,  zero_f32)
