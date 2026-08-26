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
        /* Every step through an explicit intrinsic: bare mul/add    */\
        /* chains here are contraction bait, and a per-target fma     */\
        /* decision would make the epilogue bits arch-dependent.      */\
        float val = __fmul_rn(alpha, sum);                                      \
        if (bias != nullptr) val = __fadd_rn(val, bias[col]);                   \
        if (beta != 0.0f) val = __fmaf_rn(beta, to_f(c_row[col]), val);         \
        c_row[col] = FROM_F_OUT(val);                                           \
    }                                                                           \
}

DEFINE_MATVEC_BI(matvec_bi_bf16_bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16)
DEFINE_MATVEC_BI(matvec_bi_f16_f16,   __half,        __half,        from_f_f16)
DEFINE_MATVEC_BI(matvec_bi_bf16_f32,  __nv_bfloat16, float,         from_f_f32)
DEFINE_MATVEC_BI(matvec_bi_f16_f32,   __half,        float,         from_f_f32)
DEFINE_MATVEC_BI(matvec_bi_f32_f32,   float,         float,         from_f_f32)
