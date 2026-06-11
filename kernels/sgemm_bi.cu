// Batch-invariant deterministic f32 SGEMM — training kernel.
//
// Based on siboehm's warptiling kernel (93.7% cuBLAS on A6000).
// Adapted for NVRTC compilation (no templates, no includes).
//
// Three variants for training:
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
//   SGB_GROUP_M per-arch L2 swizzle (8 sm_80, 16 sm_89+), deterministic K-reduction
//
// Source: github.com/siboehm/SGEMM_CUDA (kernel 10, warptiling)

#define BM 128
#define BN 128
#define BK 16
#define WM 64
#define WN 32
#define WNITER 1
#define TM 8
#define TN 8
#define NUM_THREADS 256
#define WARPSIZE 32
// SGB_GROUP_M is the L2-swizzle row-group size for the persistent-CTA tile walker.
// Trade-off: large groups maximize cross-CTA B-tile reuse in L2, small groups
// reduce the working set so it fits in a smaller L2.
//
// Host-side (kernels.rs::group_m_for_cc) selects a per-architecture value and
// passes `-DSGB_GROUP_M=N` to NVRTC, overriding this default:
//   sm_80 (A100 40MB L2 / sm_86 RTX 30xx 6MB L2): SGB_GROUP_M=8
//   sm_89 (RTX 40xx / 6000 Ada 96MB L2):          SGB_GROUP_M=16
//   sm_90 (H100 60MB L2 / GH200):                 SGB_GROUP_M=16
//   sm_100/120 (Blackwell B200/Ultra ≥100MB L2):  SGB_GROUP_M=16
// Default (host did not pass -D): 16, matching the prior Ada-tuned constant.
#ifndef SGB_GROUP_M
#define SGB_GROUP_M 16
#endif

// SMEM padding breaks 32-way bank conflicts on ld.shared.v4.f32 transposed reads.
// Pad=4 ensures column-stride-TM reads hit different banks instead of colliding.
// Applied across all 23 kernels in this file via (BM + SMEM_A_PAD) and
// (BN + SMEM_B_PAD) in allocation + indexing where Big/Slim tile layout fires.
// Reference: salykova/sgemm.cu (128×128×8 kernel uses ldm=132).
//
// SMEM cost delta:
//   Big  (BM=BN=128, BK=16): +4*16 + 4*16 = +128B per block  → 16384B → 16512B (under 48KB static)
//   Slim (BM=128 BN=64 BK=32): +4*32 + 4*32 = +256B per block → 24576B → 24832B (under 48KB static × 2 blocks)
#define SMEM_A_PAD 4
#define SMEM_B_PAD 4

// Derived constants
#define NUM_WARPS (NUM_THREADS / WARPSIZE)  // 8
#define WMITER ((WM * WN) / (WARPSIZE * TM * TN * WNITER))  // (64*32)/(32*8*8*1) = 2
#define WSUBM (WM / WMITER)   // 32
#define WSUBN (WN / WNITER)   // 32

// ============================================================================
// Phase 2.9: CUTLASS-style cache hints (replaces failed __ldcs experiment).
// ============================================================================
// ld.global.L2::128B prefetches next L2 line alongside .ca caching. sm_75+.
// Used by CUTLASS SGEMM mainloop B loads (cutlass/include/cutlass/arch/memory.h).
// Unlike .cs (evict-first), .ca+L2::128B keeps B resident across CTAs in the
// SGB_GROUP_M swizzle — exactly what our 16× cross-CTA reuse pattern wants.
__device__ __forceinline__ float4 ld_global_L2_128B(const float* p) {
    float4 v;
    asm("ld.global.L2::128B.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(p));
    return v;
}

// __stwt — streaming write. Output tile is write-once, never re-read by the
// same kernel. Marks L1 lines as evict-first so C writes don't evict A staging
// or B working set. Safe only for OVERWRITE epilogues (NT backward dX), NOT
// for NN forward (bias accumulate) or TN backward dW (grad accumulate).

// A load: float4, each thread loads 4 floats along K
// innerRowA = tid / (BK/4) = tid / 4, range 0..31
// innerColA = tid % (BK/4) = tid % 4, range 0..3
// rowStrideA = (NUM_THREADS * 4) / BK = (128*4)/16 = 32
// Loop: 4 iterations to cover BM=128 rows (stride 32)
#define ROW_STRIDE_A ((NUM_THREADS * 4) / BK)  // 32

// B load: float4, each thread loads 4 floats along N
// innerRowB = tid / (BN/4) = tid / 32, range 0..3
// innerColB = tid % (BN/4) = tid % 32, range 0..31
// rowStrideB = NUM_THREADS / (BN/4) = 128/32 = 4
// Loop: 4 iterations to cover BK=16 rows (stride 4)
#define ROW_STRIDE_B (NUM_THREADS / (BN / 4))  // 4

// ============================================================================
// Forward: C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C + bias
// ============================================================================
// __launch_bounds__(128, 2) — target 2 blocks/SM on Ada sm_89.
// With 168 regs × 128 threads × 2 blocks = 43008 regs (< 64K/SM) ✓
// Smem 16KB × 2 = 32KB (< 48KB static) ✓
// ptxas audit (Phase 0.1) confirmed 0 spill stores/loads.
extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_bi_nn(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float alpha, float beta,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    // α=1 contract for bias-IN-FMA seed. With bias seeded into threadResults
    // at K=0 and epilog `α·acc` writing post-α multiply, the math
    // `α·(Σ A·B + bias)` collapses to `Σ A·B + bias` ONLY when α=1 (IEEE 754
    // identity multiply). At α≠1 the pattern would produce `α·sum + α·bias`
    // instead of canonical `α·sum + bias`. Training dispatch always passes
    // α=1; if a future caller passes α≠1 with bias, unify on bias-POST
    // pattern instead. Assert in debug builds only to avoid runtime cost
    // in production.
    assert(alpha == 1.0f || bias == nullptr);
    // T1 v2: 2-stage cp.async pipeline (CUTLASS multistage SM80 pattern).
    // Dynamic smem layout: As[K_PIPE][BK*(BM+SMEM_A_PAD)] || Bs[K_PIPE][BK*(BN+SMEM_B_PAD)].
    // Per-block smem = K_PIPE * (A_STAGE + B_STAGE) * 4B = 2 * (2112 + 2112) * 4 = 33 KB.
    // Requires cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, 33*1024) — caller-side.
    //
    // OOB handling via 4-operand cp.async (PTX ISA 9.7.8.22): src_bytes=0 → hardware
    // zero-fills cp_size bytes in dst. Bit-exact identical to scalar `=0.0f`.
    // No scalar-store path → no ordering hole vs cp.async groups.
    //
    // Determinism preserved: same FMA order per tile, same tile order, same block
    // mapping. cp.async only changes WHEN a load lands; wait_group + sync ensures
    // visibility before any FMA reads from that stage.
    constexpr int K_PIPE = 2;
    extern __shared__ __align__(16) float smem[];
    constexpr int A_STAGE = BK * (BM + SMEM_A_PAD);  // 16 * 132 = 2112 floats
    constexpr int B_STAGE = BK * (BN + SMEM_B_PAD);  // 16 * 132 = 2112 floats
    float* As_buf = smem;                            // [K_PIPE * A_STAGE]
    float* Bs_buf = smem + K_PIPE * A_STAGE;         // [K_PIPE * B_STAGE]

    // SGB_GROUP_M L2 swizzle — count tiles once.
    int num_pid_m = (M + BM - 1) / BM;
    int num_pid_n = (N + BN - 1) / BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    // Warp and thread placement (constant across all tiles a CTA processes).
    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (BN / WN);
    int warpRow = warpIdx / (BN / WN);
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (WSUBN / TN);
    int threadRowInWarp = tidInWarp / (WSUBN / TN);

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);

    // Working-set registers (reset by mainloop per tile).
    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    // 2026-05-12 STAGE-4: persistent CTA loop. Grid_dim ≤ total_tiles; each
    // block walks `tile_id = blockIdx.x, blockIdx.x + gridDim.x, ...`.
    // Determinism preserved: per-tile body is paste-identical to single-tile;
    // tile→block assignment is monotonic stride → tile_id → C[pid_m,pid_n]
    // mapping deterministic; output tiles non-overlapping (no atomics).
    // CUDA Graph compat: grid_dim captured once, stays fixed across replays.
    // Portability: invariant across batch_size / hyperparams / GPU SM count
    // (host passes `grid_dim = min(total_tiles, N_SMs * blocks_per_SM)`).
    // 2026-05-13 — Stage-4 persistent CTA unwrapped. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring pre-Stage-4 register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        // Per-tile accumulators reset.
        // Bias-IN-FMA via accumulator seed at K=0. Mirrors CPU
        // `avx512::sgemm_nn` acc-load pattern (_mm512_loadu_ps(c_ptr) from
        // caller-preseeded C) — bias enters FMA chain as K=0 addend, giving
        // single-rounding bias-fold for the full Σ A·B + bias. Required for
        // α=1 production constraint (training dispatch always passes α=1).
        // Runtime assert at function entry enforces this. Per IEEE 754-2008
        // §5.4.1, fused single-round vs separate FMUL+FADD differs ≤ 1 ULP
        // per accumulation.
        float threadResults[WMITER * TM * WNITER * TN];

        // Resolve (pid_m, pid_n) from tile_id via SGB_GROUP_M swizzle.
        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        // Bias pre-seed. g_col indexing mirrors the epilog write path
        // (see L363-364). g_col is independent of resIdxM / wSubRowIdx,
        // so we compute it once per (wSubColIdx, resIdxN) and broadcast.
        if (bias != nullptr) {
            #pragma unroll
            for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                #pragma unroll
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    int g_col = pid_n * BN + warpCol * WN + wSubColIdx * WSUBN +
                                threadColInWarp * TN + resIdxN;
                    float b_val = (g_col < N) ? bias[g_col] : 0.0f;
                    #pragma unroll
                    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                        #pragma unroll
                        for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                      wSubColIdx * TN + resIdxN;
                            threadResults[idx] = b_val;
                        }
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < WMITER * TM * WNITER * TN; ++i) {
                threadResults[i] = 0.0f;
            }
        }

        // C output pointer for this tile.
        float* C_warp = C + (pid_m * BM + warpRow * WM) * ldc + pid_n * BN + warpCol * WN;

    // Macro-style helper: issue one full A+B tile into the given stage.
    // bkIdx is the K-offset of the tile being loaded. Uses 4-operand cp.async
    // for OOB zero-fill — branch-free, bit-exact, no scalar-store hole.
    #define ISSUE_TILE(stage, bkIdx) do {                                                 \
        /* 2026-05-12 STAGE-3 (NN A coalesce reorder): 32 lanes split as 2 M-rows ×       \
         * BK=16 K-cols, 1 lane = 1 float (4B cp.async). 2 cache lines/warp instr,        \
         * 50% util vs prior 12.5% (8 rows × 16B/row scatter). Same cp.async count,       \
         * same dest As[K][M] layout → bit-exact preserved. 16B cp.async blocked by       \
         * dest stride; full coalesce requires layout swap (deferred).                    \
         */                                                                               \
        {                                                                                 \
            constexpr int WARPS_NN = NUM_THREADS / WARPSIZE;                              \
            constexpr int M_ROWS_PER_WARP_INST_NN = WARPSIZE / BK;                        \
            constexpr int M_ROWS_PER_WARP_NN = BM / WARPS_NN;                             \
            constexpr int INSTR_PER_WARP_NN =                                             \
                M_ROWS_PER_WARP_NN / M_ROWS_PER_WARP_INST_NN;                             \
            static_assert(WARPSIZE % BK == 0,                                             \
                "WARPSIZE must be divisible by BK for NN A coalesce");                    \
            static_assert(BM % WARPS_NN == 0,                                             \
                "BM must be divisible by warp count for NN A coalesce");                  \
            int _warp = threadIdx.x / WARPSIZE;                                           \
            int _lane = threadIdx.x % WARPSIZE;                                           \
            int _m_in_warp = _lane / BK;                                                  \
            int _k_local = _lane % BK;                                                    \
            _Pragma("unroll")                                                             \
            for (int _it = 0; _it < INSTR_PER_WARP_NN; _it++) {                           \
                int _m_local = _warp * M_ROWS_PER_WARP_NN                                 \
                               + _it * M_ROWS_PER_WARP_INST_NN + _m_in_warp;              \
                int _g_row = pid_m * BM + _m_local;                                       \
                int _g_col = (bkIdx) + _k_local;                                          \
                unsigned _dst = As_base + ((stage) * A_STAGE                              \
                    + _k_local * (BM + SMEM_A_PAD) + _m_local)                            \
                    * (unsigned)sizeof(float);                                            \
                int _bytes = (_g_row < M && _g_col < K) ? 4 : 0;                          \
                const float* _src = A + (long long)_g_row * lda + _g_col;                 \
                asm volatile(                                                             \
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"                      \
                    :: "r"(_dst), "l"(_src), "r"(_bytes));                                \
            }                                                                             \
        }                                                                                 \
        for (int _off = 0; _off + ROW_STRIDE_B <= BK; _off += ROW_STRIDE_B) {             \
            int _g_row = (bkIdx) + innerRowB + _off;                                      \
            int _g_col = pid_n * BN + innerColB * 4;                                      \
            unsigned _dst = Bs_base + ((stage) * B_STAGE                                  \
                + (innerRowB + _off) * (BN + SMEM_B_PAD)                                  \
                + innerColB * 4) * (unsigned)sizeof(float);                               \
            const float* _src = B + (long long)_g_row * ldb + _g_col;                     \
            bool _full16 = (_g_row < K) && (_g_col + 3 < N) && ((ldb % 4) == 0);          \
            if (_full16) {                                                                \
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"            \
                             :: "r"(_dst), "l"(_src), "n"(16));                           \
            } else {                                                                      \
                _Pragma("unroll")                                                         \
                for (int _i = 0; _i < 4; _i++) {                                          \
                    unsigned _d = _dst + (unsigned)_i * (unsigned)sizeof(float);          \
                    int _b = (_g_row < K && _g_col + _i < N) ? 4 : 0;                     \
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"         \
                                 :: "r"(_d), "l"(_src + _i), "r"(_b));                    \
                }                                                                         \
            }                                                                             \
        }                                                                                 \
        asm volatile("cp.async.commit_group;\n");                                         \
    } while (0)

    // === Prologue: issue stage 0 ===
    int num_k_tiles = (K + BK - 1) / BK;
    ISSUE_TILE(0, 0);

    // === Mainloop ===
    int read_stage = 0;
    int write_stage = 1;
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        // Wait for the oldest still-in-flight group; for K_PIPE=2 this is the only one.
        asm volatile("cp.async.wait_group %0;\n" :: "n"(K_PIPE - 2));
        __syncthreads();

        // Issue NEXT tile (if not draining).
        int next_tile = tile + 1;
        if (next_tile < num_k_tiles) {
            int next_bkIdx = next_tile * BK;
            ISSUE_TILE(write_stage, next_bkIdx);
        }

        // Compute on read_stage.
        float* As_rd = As_buf + read_stage * A_STAGE;
        float* Bs_rd = Bs_buf + read_stage * B_STAGE;
        // Register fragment double-buffer (salykova/siboehm canonical).
        // Prefetch dotIdx+1 into regM_next/regN_next while FMAs consume
        // regM/regN_curr. FMA order IDENTICAL to single-buffer → bit-exact.
        // Hides smem→reg latency (~20 cycles) behind FMAs (~256 cycles/iter).
        float regM_next[WMITER * TM];
        float regN_next[WNITER * TN];
        // Prime fragment 0.
        #pragma unroll
        for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                regM[wSubRowIdx * TM + i] =
                    As_rd[0 * (BM + SMEM_A_PAD) + warpRow * WM + wSubRowIdx * WSUBM +
                          threadRowInWarp * TM + i];
        #pragma unroll
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
            #pragma unroll
            for (int i = 0; i < TN; ++i)
                regN[wSubColIdx * TN + i] =
                    Bs_rd[0 * (BN + SMEM_B_PAD) + warpCol * WN + wSubColIdx * WSUBN +
                          threadColInWarp * TN + i];

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Prefetch fragment dotIdx+1 into *_next while we FMA on *_curr.
            if (dotIdx + 1 < BK) {
                #pragma unroll
                for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                    #pragma unroll
                    for (int i = 0; i < TM; ++i)
                        regM_next[wSubRowIdx * TM + i] =
                            As_rd[(dotIdx + 1) * (BM + SMEM_A_PAD) + warpRow * WM +
                                  wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int i = 0; i < TN; ++i)
                        regN_next[wSubColIdx * TN + i] =
                            Bs_rd[(dotIdx + 1) * (BN + SMEM_B_PAD) + warpCol * WN +
                                  wSubColIdx * WSUBN + threadColInWarp * TN + i];
            }
            // FMAs on current fragment — IDENTICAL order to single-buffer.
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    #pragma unroll
                    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        #pragma unroll
                        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            // F-NT-FMA pin (2026-05-08): explicit __fmaf_rn for bit-exact
                            // match with CPU `_mm256_fmadd_ps`.
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN)
                                      + wSubColIdx * TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * TM + resIdxM],
                                regN[wSubColIdx * TN + resIdxN],
                                threadResults[idx]);
                        }
                    }
                }
            }
            // Swap: next → curr for the next dotIdx.
            if (dotIdx + 1 < BK) {
                #pragma unroll
                for (int i = 0; i < WMITER * TM; ++i) regM[i] = regM_next[i];
                #pragma unroll
                for (int i = 0; i < WNITER * TN; ++i) regN[i] = regN_next[i];
            }
        }
        // Rotate stages.
        read_stage = (read_stage + 1) % K_PIPE;
        write_stage = (write_stage + 1) % K_PIPE;
    }
    #undef ISSUE_TILE

    // Epilogue: write results with alpha, beta, bias (float4 stores)
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* C_sub = C_warp + wSubRowIdx * WSUBM * ldc + wSubColIdx * WSUBN;
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                int g_row = pid_m * BM + warpRow * WM + wSubRowIdx * WSUBM +
                            threadRowInWarp * TM + resIdxM;
                if (g_row >= M) continue;
                for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    int g_col = pid_n * BN + warpCol * WN + wSubColIdx * WSUBN +
                                threadColInWarp * TN + resIdxN;
                    // Fallback to scalar on column tail OR when ldc % 4 != 0.
                    // STG.128 needs 16-byte aligned address — row-stride in bytes
                    // (ldc * 4) must be a multiple of 16, so ldc must be a multiple
                    // of 4. Otherwise odd rows hit CUDA_ERROR_MISALIGNED_ADDRESS.
                    if (g_col + 3 >= N || (ldc & 3) != 0) {
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                      wSubColIdx * TN + resIdxN + j;
                            // Bias seeded at K=0 (see init above); drop
                            // separate `+ bias` from epilog.
                            float val = alpha * threadResults[idx];
                            if (beta != 0.0f) val += beta * C_sub[(threadRowInWarp * TM + resIdxM) * ldc + threadColInWarp * TN + resIdxN + j];
                            C_sub[(threadRowInWarp * TM + resIdxM) * ldc + threadColInWarp * TN + resIdxN + j] = val;
                        }
                        continue;
                    }
                    float4 tmp;
                    if (beta != 0.0f) {
                        tmp = reinterpret_cast<float4*>(
                            &C_sub[(threadRowInWarp * TM + resIdxM) * ldc +
                                   threadColInWarp * TN + resIdxN])[0];
                    }
                    int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                              wSubColIdx * TN + resIdxN;
                    // Bias seeded at K=0; no separate `+ bias` here.
                    float v0 = alpha * threadResults[idx + 0];
                    float v1 = alpha * threadResults[idx + 1];
                    float v2 = alpha * threadResults[idx + 2];
                    float v3 = alpha * threadResults[idx + 3];
                    if (beta != 0.0f) {
                        v0 += beta * tmp.x;
                        v1 += beta * tmp.y;
                        v2 += beta * tmp.z;
                        v3 += beta * tmp.w;
                    }
                    float4 out = {v0, v1, v2, v3};
                    reinterpret_cast<float4*>(
                        &C_sub[(threadRowInWarp * TM + resIdxM) * ldc +
                               threadColInWarp * TN + resIdxN])[0] = out;
                }
            }
        }
    }
    // Tile boundary — ensure all threads finish epilogue writes before next
    // tile's ISSUE_TILE starts new cp.async into the same shmem buffers.
    __syncthreads();
    } // end persistent CTA loop
}

// ============================================================================
// Backward dW (TN): C[K,N] += alpha * A^T[K,M] @ B[M,N]
// ============================================================================
// A = X_saved [M, K] — read transposed
// B = dY [M, N]
// C = dW [K, N] — accumulated
// Output tile [BM, BN] over (K, N). M is reduction axis.
// __launch_bounds__(128, 2) — target 2 blocks/SM (Phase 0.1 ptxas: 167 regs, 0 spill).
extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_bi_tn(
    float* __restrict__ C,
    const float* __restrict__ A,  // X [M, K_out]
    const float* __restrict__ B,  // dY [M, N]
    float alpha,
    int M_red, int K_out, int N
) {
    // T1 v2: 2-stage cp.async pipeline (CUTLASS multistage SM80).
    constexpr int K_PIPE = 2;
    extern __shared__ __align__(16) float smem[];
    constexpr int A_STAGE = BK * (BM + SMEM_A_PAD);
    constexpr int B_STAGE = BK * (BN + SMEM_B_PAD);
    float* As_buf = smem;
    float* Bs_buf = smem + K_PIPE * A_STAGE;

    int num_pid_m = (K_out + BM - 1) / BM;
    int num_pid_n = (N + BN - 1) / BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (BN / WN);
    int warpRow = warpIdx / (BN / WN);
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (WSUBN / TN);
    int threadRowInWarp = tidInWarp / (WSUBN / TN);

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);

    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    // 2026-05-12 STAGE-4: persistent CTA loop (see sgemm_bi_nn for rationale).
    // 2026-05-13 — Stage-4 persistent CTA unwrapped. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring pre-Stage-4 register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[WMITER * TM * WNITER * TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        float* C_warp = C + (pid_m * BM + warpRow * WM) * N + pid_n * BN + warpCol * WN;

    // ISSUE_TILE_TN(stage, mIdx) — issue one A+B tile for M-reduction position mIdx.
    // 4-operand cp.async zero-fills OOB bytes (PTX 9.7.8.22) — branch-free + bit-exact.
    //
    // 2026-05-12 STAGE-1 (Gemini #1 coalesce): A-load now uses warp-cooperative
    // contiguous-row reads — each warp loads `BK / (NUM_THREADS / WARPSIZE) = 2`
    // X-rows fully (32 lanes × 4 floats = 128-float row = 1 full cache line at
    // 100% utilization). Replaces the prior pattern where each warp at fixed _i
    // hit 8 different X-rows × 4 contiguous cols → 12.5% cache line utilization.
    //
    // Bit-exact: destination As[k][m] layout UNCHANGED; same data written to
    // same shmem cells in the same physical positions — FMA loop's regM/regN
    // sequence and __fmaf_rn accumulation order are byte-identical.
    //
    // Portability: cp.async is sm_80+ (Ampere/Ada/Hopper/Blackwell). No
    // architecture-specific intrinsics — same kernel compiles across all
    // modern NVIDIA GPUs via NVRTC SM-specific codegen.
    #define ISSUE_TILE_TN(stage, mIdx) do {                                               \
        {                                                                                 \
            constexpr int WARPS_TN = NUM_THREADS / WARPSIZE;                              \
            constexpr int ROWS_PER_WARP_TN = BK / WARPS_TN;                               \
            static_assert(BK % WARPS_TN == 0,                                             \
                "BK must be divisible by warp count for coalesced A-load");               \
            static_assert(BM % (WARPSIZE * 4) == 0,                                       \
                "BM must be divisible by 32 lanes * 4 floats for 16B coalesce");          \
            int _warp = threadIdx.x / WARPSIZE;                                           \
            int _lane = threadIdx.x % WARPSIZE;                                           \
            _Pragma("unroll")                                                             \
            for (int _r = 0; _r < ROWS_PER_WARP_TN; _r++) {                               \
                int _k_local = _warp * ROWS_PER_WARP_TN + _r;                             \
                int _m_local = _lane * 4;                                                 \
                int _g_m = (mIdx) + _k_local;                                             \
                int _g_k = pid_m * BM + _m_local;                                         \
                unsigned _dst = As_base + ((stage) * A_STAGE                              \
                    + _k_local * (BM + SMEM_A_PAD) + _m_local)                            \
                    * (unsigned)sizeof(float);                                            \
                bool _full16 = (_g_m < M_red) && (_g_k + 3 < K_out)                       \
                               && ((K_out & 3) == 0);                                     \
                if (_full16) {                                                            \
                    const float* _src = A + (long long)_g_m * K_out + _g_k;               \
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"        \
                                 :: "r"(_dst), "l"(_src), "n"(16));                       \
                } else {                                                                  \
                    _Pragma("unroll")                                                     \
                    for (int _i = 0; _i < 4; _i++) {                                      \
                        int _bytes = (_g_m < M_red && _g_k + _i < K_out) ? 4 : 0;         \
                        const float* _src_e =                                             \
                            A + (long long)_g_m * K_out + _g_k + _i;                      \
                        asm volatile(                                                     \
                            "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"              \
                            :: "r"(_dst + (unsigned)_i * 4),                              \
                               "l"(_src_e), "r"(_bytes));                                 \
                    }                                                                     \
                }                                                                         \
            }                                                                             \
        }                                                                                 \
        for (int _off = 0; _off + ROW_STRIDE_B <= BK; _off += ROW_STRIDE_B) {             \
            int _g_m = (mIdx) + innerRowB + _off;                                         \
            int _g_n = pid_n * BN + innerColB * 4;                                        \
            unsigned _dst = Bs_base + ((stage) * B_STAGE                                  \
                + (innerRowB + _off) * (BN + SMEM_B_PAD) + innerColB * 4)                 \
                * (unsigned)sizeof(float);                                                \
            const float* _src = B + (long long)_g_m * N + _g_n;                           \
            bool _full16 = (_g_m < M_red) && (_g_n + 3 < N) && ((N % 4) == 0);            \
            if (_full16) {                                                                \
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"            \
                             :: "r"(_dst), "l"(_src), "n"(16));                           \
            } else {                                                                      \
                _Pragma("unroll")                                                         \
                for (int _i = 0; _i < 4; _i++) {                                          \
                    unsigned _d = _dst + (unsigned)_i * (unsigned)sizeof(float);          \
                    int _b = (_g_m < M_red && _g_n + _i < N) ? 4 : 0;                     \
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"         \
                                 :: "r"(_d), "l"(_src + _i), "r"(_b));                    \
                }                                                                         \
            }                                                                             \
        }                                                                                 \
        asm volatile("cp.async.commit_group;\n");                                         \
    } while (0)

    int num_k_tiles = (M_red + BK - 1) / BK;
    ISSUE_TILE_TN(0, 0);

    int read_stage = 0;
    int write_stage = 1;
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(K_PIPE - 2));
        __syncthreads();

        int next_tile = tile + 1;
        if (next_tile < num_k_tiles) {
            int next_mIdx = next_tile * BK;
            ISSUE_TILE_TN(write_stage, next_mIdx);
        }

        float* As_rd = As_buf + read_stage * A_STAGE;
        float* Bs_rd = Bs_buf + read_stage * B_STAGE;
        // Register fragment double-buffer.
        float regM_next[WMITER * TM];
        float regN_next[WNITER * TN];
        #pragma unroll
        for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                regM[wSubRowIdx * TM + i] = As_rd[0 * (BM + SMEM_A_PAD) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
        #pragma unroll
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
            #pragma unroll
            for (int i = 0; i < TN; ++i)
                regN[wSubColIdx * TN + i] = Bs_rd[0 * (BN + SMEM_B_PAD) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            if (dotIdx + 1 < BK) {
                #pragma unroll
                for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                    #pragma unroll
                    for (int i = 0; i < TM; ++i)
                        regM_next[wSubRowIdx * TM + i] = As_rd[(dotIdx + 1) * (BM + SMEM_A_PAD) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int i = 0; i < TN; ++i)
                        regN_next[wSubColIdx * TN + i] = Bs_rd[(dotIdx + 1) * (BN + SMEM_B_PAD) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
            }
            // F-NT-FMA pin (2026-05-08): explicit __fmaf_rn for bit-exact
            // match with CPU `_mm256_fmadd_ps`. sgemm_bi_tn (TN GEMM, K-pipelined).
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int resIdxM = 0; resIdxM < TM; ++resIdxM)
                        #pragma unroll
                        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN)
                                      + wSubColIdx * TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * TM + resIdxM],
                                regN[wSubColIdx * TN + resIdxN],
                                threadResults[idx]);
                        }
            if (dotIdx + 1 < BK) {
                #pragma unroll
                for (int i = 0; i < WMITER * TM; ++i) regM[i] = regM_next[i];
                #pragma unroll
                for (int i = 0; i < WNITER * TN; ++i) regN[i] = regN_next[i];
            }
        }
        read_stage = (read_stage + 1) % K_PIPE;
        write_stage = (write_stage + 1) % K_PIPE;
    }
    #undef ISSUE_TILE_TN

    // Epilogue: accumulate into dW
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* C_sub = C_warp + wSubRowIdx * WSUBM * N + wSubColIdx * WSUBN;
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                int g_row = pid_m * BM + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + resIdxM;
                if (g_row >= K_out) continue;
                for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    int g_col = pid_n * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + resIdxN;
                    // Fallback to scalar on tail OR when N (row-stride) % 4 != 0
                    // (STG.128 / LDG.128 need 16-byte aligned address).
                    if (g_col + 3 >= N || (N & 3) != 0) {
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN + j;
                            C_sub[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN + j] += alpha * threadResults[idx];
                        }
                        continue;
                    }
                    float4 old = reinterpret_cast<float4*>(&C_sub[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0];
                    int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
                    old.x += alpha * threadResults[idx + 0];
                    old.y += alpha * threadResults[idx + 1];
                    old.z += alpha * threadResults[idx + 2];
                    old.w += alpha * threadResults[idx + 3];
                    reinterpret_cast<float4*>(&C_sub[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0] = old;
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop
}

// ============================================================================
// Split-M TN backward dW: per-chunk partial of X^T @ dY (CUTLASS parallel-split pattern).
// ============================================================================
// Paired with sgemm_bi_splitm_reduce for fixed-order tree sum across chunks.
// Grid: (K_tiles * N_tiles, 1, F)  where blockIdx.z = fc (chunk index).
// Each block reduces M_CHUNK samples starting at m_begin = fc * M_CHUNK.
// Writes partial[fc, pid_k_out_tile*BM+row, pid_n_tile*BN+col] = raw sum (no alpha).
//
// Invariants:
//   - Inside each chunk: BK-tiled accumulation IDENTICAL to sgemm_bi_tn → bit-exact
//     per-chunk partial.
//   - Each (fc, k, n) slot has exactly ONE writer → no atomics, no race.
//   - Last chunk may be short (M % M_CHUNK != 0) — handled by existing g_m<m_end
//     OOB mask in the cp.async loads.
//
// Gain: inflates grid by F× for Big TN shapes where K_tiles*N_tiles < 2*NUM_SMS.
// ============================================================================
extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_bi_tn_splitm_partial(
    float* __restrict__ partial,       // [F * K_out * N] — unique slot per block
    const float* __restrict__ A,       // X [M, K_out]
    const float* __restrict__ B,       // dY [M, N]
    int M_red, int K_out, int N,
    int M_CHUNK                         // chunk size (multiple of BK)
) {
    __shared__ float As[BK * (BM + SMEM_A_PAD)];
    __shared__ float Bs[BK * (BN + SMEM_B_PAD)];

    int num_pid_m = (K_out + BM - 1) / BM;
    int num_pid_n = (N + BN - 1) / BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;
    int fc = blockIdx.z;
    int m_begin = fc * M_CHUNK;
    int m_end = min(m_begin + M_CHUNK, M_red);
    if (m_begin >= M_red) return;

    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (BN / WN);
    int warpRow = warpIdx / (BN / WN);
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (WSUBN / TN);
    int threadRowInWarp = tidInWarp / (WSUBN / TN);

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);

    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    // 2026-05-12 STAGE-1 coalesce (mirrors sgemm_bi_tn ISSUE_TILE_TN A-loader).
    constexpr int WARPS_SM = NUM_THREADS / WARPSIZE;
    constexpr int ROWS_PER_WARP_SM = BK / WARPS_SM;
    static_assert(BK % WARPS_SM == 0, "BK must be divisible by warp count");
    static_assert(BM % (WARPSIZE * 4) == 0, "BM must be divisible by 32*4 for 16B coalesce");
    int _warp = threadIdx.x / WARPSIZE;
    int _lane = threadIdx.x % WARPSIZE;

    // 2026-05-12 STAGE-4: persistent CTA loop. fc/m_begin/m_end stay kernel-scoped.
    // 2026-05-13 — Stage-4 persistent CTA unwrapped. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring pre-Stage-4 register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[WMITER * TM * WNITER * TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    for (int mIdx = m_begin; mIdx < m_end; mIdx += BK) {
        #pragma unroll
        for (int _r = 0; _r < ROWS_PER_WARP_SM; _r++) {
            int k_local = _warp * ROWS_PER_WARP_SM + _r;
            int m_local = _lane * 4;
            int g_m = mIdx + k_local;
            int g_k = pid_m * BM + m_local;
            unsigned dst = As_base + (k_local * (BM + SMEM_A_PAD) + m_local)
                * (unsigned)sizeof(float);
            bool full16 = (g_m < m_end) && (g_k + 3 < K_out) && ((K_out & 3) == 0);
            if (full16) {
                const float* src = A + (long long)g_m * K_out + g_k;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                #pragma unroll
                for (int _i = 0; _i < 4; _i++) {
                    bool ok = (g_m < m_end) && (g_k + _i < K_out);
                    if (ok) {
                        const float* src_e = A + (long long)g_m * K_out + g_k + _i;
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                                     :: "r"(dst + (unsigned)_i * 4), "l"(src_e));
                    } else {
                        As[k_local * (BM + SMEM_A_PAD) + m_local + _i] = 0.0f;
                    }
                }
            }
        }
        for (int offset = 0; offset + ROW_STRIDE_B <= BK; offset += ROW_STRIDE_B) {
            int g_m = mIdx + innerRowB + offset;
            int g_n = pid_n * BN + innerColB * 4;
            unsigned dst = Bs_base + ((innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
            if (g_m < m_end && g_n + 3 < N && (N % 4 == 0)) {
                const float* src = B + ((long long)g_m) * N + pid_n * BN + innerColB * 4;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 0] = (g_m < m_end && g_n + 0 < N) ? B[g_m * N + pid_n * BN + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 1] = (g_m < m_end && g_n + 1 < N) ? B[g_m * N + pid_n * BN + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 2] = (g_m < m_end && g_n + 2 < N) ? B[g_m * N + pid_n * BN + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 3] = (g_m < m_end && g_n + 3 < N) ? B[g_m * N + pid_n * BN + innerColB * 4 + 3] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                for (int i = 0; i < TM; ++i)
                    regM[wSubRowIdx * TM + i] = As[dotIdx * (BM + SMEM_A_PAD) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
            for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                for (int i = 0; i < TN; ++i)
                    regN[wSubColIdx * TN + i] = Bs[dotIdx * (BN + SMEM_B_PAD) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
            // F-NT-FMA pin (2026-05-08): explicit __fmaf_rn for bit-exact
            // match with CPU `_mm256_fmadd_ps`. Same fix class as F-09a
            // (RoPE backward FMA pin).
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                    for (int resIdxM = 0; resIdxM < TM; ++resIdxM)
                        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN)
                                      + wSubColIdx * TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * TM + resIdxM],
                                regN[wSubColIdx * TN + resIdxN],
                                threadResults[idx]);
                        }
        }
        __syncthreads();
    }

    // Epilogue: OVERWRITE partial[fc, :, :] slot. No alpha, no bias, no accumulate —
    // reducer applies alpha on the final sum.
    float* partial_chunk = partial + (long long)fc * K_out * N;
    float* partial_warp = partial_chunk + (pid_m * BM + warpRow * WM) * N + pid_n * BN + warpCol * WN;

    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* P_sub = partial_warp + wSubRowIdx * WSUBM * N + wSubColIdx * WSUBN;
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                int g_row = pid_m * BM + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + resIdxM;
                if (g_row >= K_out) continue;
                for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    int g_col = pid_n * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + resIdxN;
                    int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
                    // Fallback to scalar on tail OR when N (row-stride) % 4 != 0
                    // (STG.128 requires 16-byte aligned address).
                    if (g_col + 3 >= N || (N & 3) != 0) {
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            P_sub[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN + j] = threadResults[idx + j];
                        }
                        continue;
                    }
                    float4 out;
                    out.x = threadResults[idx + 0];
                    out.y = threadResults[idx + 1];
                    out.z = threadResults[idx + 2];
                    out.w = threadResults[idx + 3];
                    reinterpret_cast<float4*>(&P_sub[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0] = out;
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (tn_splitm_partial)
}

// ============================================================================
// Split-M reducer: dW[K_out, N] += alpha * Σ_fc partial[fc, :, :].
// Fixed ascending-fc order — bit-exact reduction tree per output slot.
// Accumulate (+=) semantic matches sgemm_bi_tn contract.
// Each thread owns one (k, n) output — no atomics, no race.
//
// f64 accumulator (Option B): F-step linear sum lives in double, cast back to
// f32 once at the end. Reduces accumulation error from γ_F·ε_f32 to ~ε_f32
// (single round-down on cast). Kernel is bandwidth-bound on the F partial
// loads, so the f64 add cost is masked by the load latency. Bit-exact
// run-to-run preserved (same operations every launch).
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 8)
// `K_out` here is the leading dim of the per-fc
// partial layout `[F, K_out, N]`, NOT necessarily K of the source GEMM.
// At TN splitm callsites (L900, L3166) `K_out` receives source-GEMM-M; at the
// NT splitn callsite (L1423) `K_out` receives source-GEMM-M while `N` arg
// receives source-K. Treat the param as "leading dim of output [K_out, N]".
void sgemm_bi_splitm_reduce(
    float* __restrict__ dW,
    const float* __restrict__ partial,
    float alpha,
    int K_out, int N, int F
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K_out * N;
    if (idx >= total) return;

    long long kn_stride = (long long)K_out * N;
    double sum = (double)partial[idx];
    for (int fc = 1; fc < F; ++fc) {
        sum += (double)partial[(long long)fc * kn_stride + idx];
    }
    dW[idx] += (float)((double)alpha * sum);
}

// ============================================================================
// Split-K Big NN partial — wave-fill extension for underfilled Big NN shapes.
// Identical per-block FMA order to sgemm_bi_nn on its K-slice → bit-exact.
// Grid: (M_tiles * N_tiles, 1, F). blockIdx.z = fc ∈ [0, F). Each fc owns
// K-chunk [fc*K_chunk, min(K, (fc+1)*K_chunk)) and writes to partial[fc, m, n].
// Caller follows with sgemm_bi_splitm_reduce(out=C, partial, K_out=M, N=N, F=F)
// AFTER cuMemsetAsync(C, 0, M*N*4) — reducer does C += alpha*sum.
//
// Constraints (must hold for bit-exactness):
//   - K_chunk % BK == 0 (enforced by dispatcher; each fc's K-range is BK-aligned
//     except optionally the last fc which handles K-tail via existing cp.async
//     zero-fill pattern, identical to non-split Big NN's K%BK handling)
//   - No alpha / bias / beta in this kernel — raw tile sums only
//   - Dynamic smem 33 KB (same as Big NN): caller MUST cuFuncSetAttribute
//     MAX_DYNAMIC_SHARED_SIZE_BYTES on this CUfunction handle separately
//
// Pattern reference: CUTLASS GemmSplitKParallel (partial kernel overwrites
// unique per-fc slot, reducer fuses Σ ascending-fc in f64). Matches existing
// sgemm_bi_tn_splitm_partial (M-axis split) — this is the K-axis twin.
// ============================================================================
extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_bi_nn_splitk_big_partial(
    float* __restrict__ partial,       // [F * M * N] — unique slot per fc, row-major
    const float* __restrict__ A,       // [M, K_full]
    const float* __restrict__ B,       // [K_full, N]
    int M, int N, int K,
    int lda, int ldb,
    int K_chunk                         // must be multiple of BK
) {
    constexpr int K_PIPE = 2;
    extern __shared__ __align__(16) float smem[];
    constexpr int A_STAGE = BK * (BM + SMEM_A_PAD);
    constexpr int B_STAGE = BK * (BN + SMEM_B_PAD);
    float* As_buf = smem;
    float* Bs_buf = smem + K_PIPE * A_STAGE;

    // Decode fc from z-axis; early exit if fc is out of K-range.
    int fc = blockIdx.z;
    int k_begin = fc * K_chunk;
    if (k_begin >= K) return;
    int k_end = min(K, k_begin + K_chunk);

    // SGB_GROUP_M L2 swizzle (identical to Big NN)
    int num_pid_m = (M + BM - 1) / BM;
    int num_pid_n = (N + BN - 1) / BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (BN / WN);
    int warpRow = warpIdx / (BN / WN);
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (WSUBN / TN);
    int threadRowInWarp = tidInWarp / (WSUBN / TN);

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);

    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    // 2026-05-12 STAGE-4: persistent CTA loop. fc/k_begin/k_end stay kernel-scoped.
    // 2026-05-13 — Stage-4 persistent CTA unwrapped. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring pre-Stage-4 register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[WMITER * TM * WNITER * TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    // ISSUE_TILE — same as Big NN but uses `K` as the global K-bound (for OOB
    // zero-fill). Per-block K range [k_begin, k_end) is enforced by the tile
    // loop; cp.async still zero-fills any lane that reads k >= K.
    #define ISSUE_TILE_SK(stage, bkIdx) do {                                              \
        /* 2026-05-12 STAGE-3 coalesce (mirrors big NN ISSUE_TILE A-load). */             \
        {                                                                                 \
            constexpr int WARPS_SK = NUM_THREADS / WARPSIZE;                              \
            constexpr int M_ROWS_PER_WARP_INST_SK = WARPSIZE / BK;                        \
            constexpr int M_ROWS_PER_WARP_SK = BM / WARPS_SK;                             \
            constexpr int INSTR_PER_WARP_SK =                                             \
                M_ROWS_PER_WARP_SK / M_ROWS_PER_WARP_INST_SK;                             \
            static_assert(WARPSIZE % BK == 0, "WARPSIZE divisible by BK");                \
            static_assert(BM % WARPS_SK == 0, "BM divisible by warp count");              \
            int _warp = threadIdx.x / WARPSIZE;                                           \
            int _lane = threadIdx.x % WARPSIZE;                                           \
            int _m_in_warp = _lane / BK;                                                  \
            int _k_local = _lane % BK;                                                    \
            _Pragma("unroll")                                                             \
            for (int _it = 0; _it < INSTR_PER_WARP_SK; _it++) {                           \
                int _m_local = _warp * M_ROWS_PER_WARP_SK                                 \
                               + _it * M_ROWS_PER_WARP_INST_SK + _m_in_warp;              \
                int _g_row = pid_m * BM + _m_local;                                       \
                int _g_col = (bkIdx) + _k_local;                                          \
                unsigned _dst = As_base + ((stage) * A_STAGE                              \
                    + _k_local * (BM + SMEM_A_PAD) + _m_local)                            \
                    * (unsigned)sizeof(float);                                            \
                int _bytes = (_g_row < M && _g_col < K) ? 4 : 0;                          \
                const float* _src = A + (long long)_g_row * lda + _g_col;                 \
                asm volatile(                                                             \
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"                      \
                    :: "r"(_dst), "l"(_src), "r"(_bytes));                                \
            }                                                                             \
        }                                                                                 \
        for (int _off = 0; _off + ROW_STRIDE_B <= BK; _off += ROW_STRIDE_B) {             \
            int _g_row = (bkIdx) + innerRowB + _off;                                      \
            int _g_col = pid_n * BN + innerColB * 4;                                      \
            unsigned _dst = Bs_base + ((stage) * B_STAGE                                  \
                + (innerRowB + _off) * (BN + SMEM_B_PAD)                                  \
                + innerColB * 4) * (unsigned)sizeof(float);                               \
            const float* _src = B + (long long)_g_row * ldb + _g_col;                     \
            bool _full16 = (_g_row < K) && (_g_col + 3 < N) && ((ldb % 4) == 0);          \
            if (_full16) {                                                                \
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"            \
                             :: "r"(_dst), "l"(_src), "n"(16));                           \
            } else {                                                                      \
                _Pragma("unroll")                                                         \
                for (int _i = 0; _i < 4; _i++) {                                          \
                    unsigned _d = _dst + (unsigned)_i * (unsigned)sizeof(float);          \
                    int _b = (_g_row < K && _g_col + _i < N) ? 4 : 0;                     \
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"         \
                                 :: "r"(_d), "l"(_src + _i), "r"(_b));                    \
                }                                                                         \
            }                                                                             \
        }                                                                                 \
        asm volatile("cp.async.commit_group;\n");                                         \
    } while (0)

    // Tile count for this fc's K-range. k_begin and K_chunk are BK-aligned,
    // so num_k_tiles = ceil((k_end - k_begin) / BK).
    int num_k_tiles = (k_end - k_begin + BK - 1) / BK;

    // Prologue: stage 0 at k_begin
    ISSUE_TILE_SK(0, k_begin);

    // Mainloop — identical structure to Big NN
    int read_stage = 0;
    int write_stage = 1;
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(K_PIPE - 2));
        __syncthreads();

        int next_tile = tile + 1;
        if (next_tile < num_k_tiles) {
            int next_bkIdx = k_begin + next_tile * BK;
            ISSUE_TILE_SK(write_stage, next_bkIdx);
        }

        float* As_rd = As_buf + read_stage * A_STAGE;
        float* Bs_rd = Bs_buf + read_stage * B_STAGE;
        // Register fragment double-buffer — identical FMA order to Big NN.
        float regM_next[WMITER * TM];
        float regN_next[WNITER * TN];
        #pragma unroll
        for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                regM[wSubRowIdx * TM + i] =
                    As_rd[0 * (BM + SMEM_A_PAD) + warpRow * WM + wSubRowIdx * WSUBM +
                          threadRowInWarp * TM + i];
        #pragma unroll
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
            #pragma unroll
            for (int i = 0; i < TN; ++i)
                regN[wSubColIdx * TN + i] =
                    Bs_rd[0 * (BN + SMEM_B_PAD) + warpCol * WN + wSubColIdx * WSUBN +
                          threadColInWarp * TN + i];

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            if (dotIdx + 1 < BK) {
                #pragma unroll
                for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                    #pragma unroll
                    for (int i = 0; i < TM; ++i)
                        regM_next[wSubRowIdx * TM + i] =
                            As_rd[(dotIdx + 1) * (BM + SMEM_A_PAD) + warpRow * WM +
                                  wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int i = 0; i < TN; ++i)
                        regN_next[wSubColIdx * TN + i] =
                            Bs_rd[(dotIdx + 1) * (BN + SMEM_B_PAD) + warpCol * WN +
                                  wSubColIdx * WSUBN + threadColInWarp * TN + i];
            }
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    #pragma unroll
                    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        #pragma unroll
                        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            // F-NT-FMA pin (2026-05-08): explicit __fmaf_rn for bit-exact
                            // match with CPU `_mm256_fmadd_ps`.
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN)
                                      + wSubColIdx * TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * TM + resIdxM],
                                regN[wSubColIdx * TN + resIdxN],
                                threadResults[idx]);
                        }
                    }
                }
            }
            if (dotIdx + 1 < BK) {
                #pragma unroll
                for (int i = 0; i < WMITER * TM; ++i) regM[i] = regM_next[i];
                #pragma unroll
                for (int i = 0; i < WNITER * TN; ++i) regN[i] = regN_next[i];
            }
        }
        read_stage = (read_stage + 1) % K_PIPE;
        write_stage = (write_stage + 1) % K_PIPE;
    }
    #undef ISSUE_TILE_SK

    // Epilogue: OVERWRITE partial[fc, :, :] — raw tile sums. NO alpha, NO bias,
    // NO beta. Reducer applies alpha and folds into C.
    float* partial_chunk = partial + (long long)fc * M * N;
    float* partial_warp = partial_chunk + (pid_m * BM + warpRow * WM) * N + pid_n * BN + warpCol * WN;

    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* P_sub = partial_warp + wSubRowIdx * WSUBM * N + wSubColIdx * WSUBN;
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                int g_row = pid_m * BM + warpRow * WM + wSubRowIdx * WSUBM +
                            threadRowInWarp * TM + resIdxM;
                if (g_row >= M) continue;
                for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    int g_col = pid_n * BN + warpCol * WN + wSubColIdx * WSUBN +
                                threadColInWarp * TN + resIdxN;
                    int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                              wSubColIdx * TN + resIdxN;
                    // Scalar fallback for right-edge AND for non-%4 N
                    // (hardens audit finding M2 — reducer walks per-element, so
                    // partial slot layout must be exact row-stride-N regardless).
                    if (g_col + 3 >= N || (N % 4) != 0) {
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            P_sub[(threadRowInWarp * TM + resIdxM) * N +
                                  threadColInWarp * TN + resIdxN + j] =
                                threadResults[idx + j];
                        }
                        continue;
                    }
                    float4 out;
                    out.x = threadResults[idx + 0];
                    out.y = threadResults[idx + 1];
                    out.z = threadResults[idx + 2];
                    out.w = threadResults[idx + 3];
                    reinterpret_cast<float4*>(
                        &P_sub[(threadRowInWarp * TM + resIdxM) * N +
                               threadColInWarp * TN + resIdxN])[0] = out;
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (nn_splitk_big_partial)
}

// ============================================================================
// Backward dX (NT): C[M,K] = alpha * A[M,N] @ B^T[N,K]
// ============================================================================
// A = dY [M, N]
// B = W [K, N] — read transposed as W^T[N,K]
// C = dX [M, K] — overwrite
// __launch_bounds__(128, 2) — target 2 blocks/SM (Phase 0.1 ptxas: 168 regs, 0 spill).
extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_bi_nt(
    float* __restrict__ C,
    const float* __restrict__ A,  // dY [M, N]
    const float* __restrict__ B,  // W [K, N]
    float alpha,
    int M, int N, int K_out
) {
    // T1 v2: 2-stage cp.async pipeline.
    constexpr int K_PIPE = 2;
    extern __shared__ __align__(16) float smem[];
    constexpr int A_STAGE = BK * (BM + SMEM_A_PAD);
    constexpr int B_STAGE = BK * (BN + SMEM_B_PAD);
    float* As_buf = smem;
    float* Bs_buf = smem + K_PIPE * A_STAGE;

    int num_pid_m = (M + BM - 1) / BM;
    int num_pid_n = (K_out + BN - 1) / BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (BN / WN);
    int warpRow = warpIdx / (BN / WN);
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (WSUBN / TN);
    int threadRowInWarp = tidInWarp / (WSUBN / TN);

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);

    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    // 2026-05-12 STAGE-4: persistent CTA loop (see sgemm_bi_nn for rationale).
    // 2026-05-13 — Stage-4 persistent CTA unwrapped. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring pre-Stage-4 register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[WMITER * TM * WNITER * TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        float* C_warp = C + (pid_m * BM + warpRow * WM) * K_out + pid_n * BN + warpCol * WN;

    // ISSUE_TILE_NT(stage, nIdx) — issue one A+B tile for N-reduction position nIdx.
    //
    // 2026-05-12 STAGE-2 (Gemini #1/coalesce for NT): A-load now uses warp-cooperative
    // contiguous-row reads. NT tile geometry differs from TN: here BK=16 is the
    // N-axis (reduction dim) of dY[M,N] and BM=128 is the M-axis (samples).
    // dY[m, n] is row-major → for fixed m varying n is contiguous.
    //
    // Best coalesce we can get: 32 lanes split as 2 M-rows × BK=16 N-cols per row
    // (each lane reads 1 float, 4B cp.async). Per warp instruction: 2 cache lines,
    // each 64 B utilized of 128 B → 50% utilization. Replaces the prior pattern
    // (32 threads on 8 M-rows × 4 N-cols/row → 8 cache lines × 16 B util = 12.5%).
    // Net gain: **4× cache line utilization** at SAME 8 cp.async/thread count.
    //
    // We do NOT use 16B cp.async here — BK=16 < 32 lanes/row makes a full-warp
    // contiguous N-stripe impossible without layout swap (Gemini #2, deferred).
    //
    // Bit-exact: destination As[n_local][m_local] cells receive identical bytes.
    // FMA loop unchanged.
    #define ISSUE_TILE_NT(stage, nIdx) do {                                               \
        {                                                                                 \
            constexpr int WARPS_NT = NUM_THREADS / WARPSIZE;                              \
            constexpr int M_ROWS_PER_WARP_INST = WARPSIZE / BK;                           \
            constexpr int M_ROWS_PER_WARP = BM / WARPS_NT;                                \
            constexpr int INSTR_PER_WARP_NT = M_ROWS_PER_WARP / M_ROWS_PER_WARP_INST;     \
            static_assert(WARPSIZE % BK == 0,                                             \
                "WARPSIZE must be divisible by BK for NT A coalesce");                    \
            static_assert(BM % WARPS_NT == 0,                                             \
                "BM must be divisible by warp count for NT A coalesce");                  \
            int _warp = threadIdx.x / WARPSIZE;                                           \
            int _lane = threadIdx.x % WARPSIZE;                                           \
            int _m_in_warp = _lane / BK;                                                  \
            int _n_local = _lane % BK;                                                    \
            _Pragma("unroll")                                                             \
            for (int _it = 0; _it < INSTR_PER_WARP_NT; _it++) {                           \
                int _m_local = _warp * M_ROWS_PER_WARP                                    \
                               + _it * M_ROWS_PER_WARP_INST + _m_in_warp;                 \
                int _g_m = pid_m * BM + _m_local;                                         \
                int _g_n = (nIdx) + _n_local;                                             \
                unsigned _dst = As_base + ((stage) * A_STAGE                              \
                    + _n_local * (BM + SMEM_A_PAD) + _m_local)                            \
                    * (unsigned)sizeof(float);                                            \
                int _bytes = (_g_m < M && _g_n < N) ? 4 : 0;                              \
                const float* _src = A + (long long)_g_m * N + _g_n;                       \
                asm volatile(                                                             \
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"                      \
                    :: "r"(_dst), "l"(_src), "r"(_bytes));                                \
            }                                                                             \
        }                                                                                 \
        for (int _off = 0; _off + ROW_STRIDE_B <= BK; _off += ROW_STRIDE_B) {             \
            int _n_local = innerRowB + _off;                                              \
            int _k_base = innerColB * 4;                                                  \
            int _g_n = (nIdx) + _n_local;                                                 \
            int _g_k = pid_n * BN + _k_base;                                              \
            bool _pn = _g_n < N;                                                          \
            _Pragma("unroll")                                                             \
            for (int _i = 0; _i < 4; _i++) {                                              \
                unsigned _dst = Bs_base + ((stage) * B_STAGE                              \
                    + _n_local * (BN + SMEM_B_PAD) + _k_base + _i)                        \
                    * (unsigned)sizeof(float);                                            \
                int _bytes = (_pn && (_g_k + _i) < K_out) ? 4 : 0;                        \
                const float* _src = B + (long long)(_g_k + _i) * N + _g_n;                \
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"             \
                             :: "r"(_dst), "l"(_src), "r"(_bytes));                       \
            }                                                                             \
        }                                                                                 \
        asm volatile("cp.async.commit_group;\n");                                         \
    } while (0)

    int num_k_tiles = (N + BK - 1) / BK;
    ISSUE_TILE_NT(0, 0);

    int read_stage = 0;
    int write_stage = 1;
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(K_PIPE - 2));
        __syncthreads();

        int next_tile = tile + 1;
        if (next_tile < num_k_tiles) {
            int next_nIdx = next_tile * BK;
            ISSUE_TILE_NT(write_stage, next_nIdx);
        }

        float* As_rd = As_buf + read_stage * A_STAGE;
        float* Bs_rd = Bs_buf + read_stage * B_STAGE;
        // Register fragment double-buffer.
        float regM_next[WMITER * TM];
        float regN_next[WNITER * TN];
        #pragma unroll
        for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                regM[wSubRowIdx * TM + i] = As_rd[0 * (BM + SMEM_A_PAD) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
        #pragma unroll
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
            #pragma unroll
            for (int i = 0; i < TN; ++i)
                regN[wSubColIdx * TN + i] = Bs_rd[0 * (BN + SMEM_B_PAD) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            if (dotIdx + 1 < BK) {
                #pragma unroll
                for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                    #pragma unroll
                    for (int i = 0; i < TM; ++i)
                        regM_next[wSubRowIdx * TM + i] = As_rd[(dotIdx + 1) * (BM + SMEM_A_PAD) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int i = 0; i < TN; ++i)
                        regN_next[wSubColIdx * TN + i] = Bs_rd[(dotIdx + 1) * (BN + SMEM_B_PAD) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
            }
            // F-NT-FMA pin (2026-05-08): explicit __fmaf_rn for bit-exact
            // match with CPU `_mm256_fmadd_ps`. sgemm_bi_nt (NT GEMM, K-pipelined).
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int resIdxM = 0; resIdxM < TM; ++resIdxM)
                        #pragma unroll
                        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN)
                                      + wSubColIdx * TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * TM + resIdxM],
                                regN[wSubColIdx * TN + resIdxN],
                                threadResults[idx]);
                        }
            if (dotIdx + 1 < BK) {
                #pragma unroll
                for (int i = 0; i < WMITER * TM; ++i) regM[i] = regM_next[i];
                #pragma unroll
                for (int i = 0; i < WNITER * TN; ++i) regN[i] = regN_next[i];
            }
        }
        read_stage = (read_stage + 1) % K_PIPE;
        write_stage = (write_stage + 1) % K_PIPE;
    }
    #undef ISSUE_TILE_NT

    // Epilogue: overwrite dX
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* C_sub = C_warp + wSubRowIdx * WSUBM * K_out + wSubColIdx * WSUBN;
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                int g_row = pid_m * BM + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + resIdxM;
                if (g_row >= M) continue;
                for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    int g_col = pid_n * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + resIdxN;
                    // float4 write only when K_out is %4-aligned (K_out=257 → scalar).
                    if (g_col + 3 >= K_out || (K_out % 4 != 0)) {
                        int idx_base = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
                        for (int j = 0; j < 4 && g_col + j < K_out; j++) {
                            __stwt(&C_sub[(threadRowInWarp * TM + resIdxM) * K_out + threadColInWarp * TN + resIdxN + j], alpha * threadResults[idx_base + j]);
                        }
                        continue;
                    }
                    int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
                    float4 out = {
                        alpha * threadResults[idx + 0],
                        alpha * threadResults[idx + 1],
                        alpha * threadResults[idx + 2],
                        alpha * threadResults[idx + 3]
                    };
                    // Phase 2.9: __stwt — streaming store. NT backward dX is OVERWRITE (no accumulation),
                    // so marking C lines evict-first is safe and prevents C writes from evicting A staging / B working set.
                    __stwt(reinterpret_cast<float4*>(&C_sub[(threadRowInWarp * TM + resIdxM) * K_out + threadColInWarp * TN + resIdxN]), out);
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop
}

// ============================================================================
// Split-N Big NT partial — wave-fill extension for underfilled Big NT shapes.
// Identical per-block FMA order to sgemm_bi_nt on its N-slice → bit-exact.
// Grid: (M_tiles * K_out_tiles, 1, F). blockIdx.z = fc ∈ [0, F).
// Each fc owns N-chunk [fc*N_chunk, min(N, (fc+1)*N_chunk)) and writes
// partial[fc, m, k_out].
// Caller follows with sgemm_bi_splitm_reduce(out=C, partial, K_out=M, N=K_out, F=F)
// AFTER cuMemsetAsync(C, 0, M*K_out*4) — reducer does C += alpha*sum.
//
// Constraints (must hold for bit-exactness):
//   - N_chunk % BK == 0 (enforced by dispatcher; tail fc handles via cp.async zero-fill)
//   - No alpha — reducer applies it
//   - Dynamic smem 33 KB; caller must cuFuncSetAttribute on this CUfunction handle
//
// Mirrors sgemm_bi_nn_splitk_big_partial (K-axis twin) for dW bwd_dx path.
// ============================================================================
extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_bi_nt_splitn_big_partial(
    float* __restrict__ partial,       // [F * M * K_out] — unique slot per fc
    const float* __restrict__ A,       // dY [M, N]
    const float* __restrict__ B,       // W  [K_out, N]
    int M, int N, int K_out,
    int N_chunk                         // must be multiple of BK
) {
    constexpr int K_PIPE = 2;
    extern __shared__ __align__(16) float smem[];
    constexpr int A_STAGE = BK * (BM + SMEM_A_PAD);
    constexpr int B_STAGE = BK * (BN + SMEM_B_PAD);
    float* As_buf = smem;
    float* Bs_buf = smem + K_PIPE * A_STAGE;

    int fc = blockIdx.z;
    int n_begin = fc * N_chunk;
    if (n_begin >= N) return;
    int n_end = min(N, n_begin + N_chunk);

    // SGB_GROUP_M L2 swizzle (identical to Big NT: pid_m on M, pid_n on K_out)
    int num_pid_m = (M + BM - 1) / BM;
    int num_pid_n = (K_out + BN - 1) / BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (BN / WN);
    int warpRow = warpIdx / (BN / WN);
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (WSUBN / TN);
    int threadRowInWarp = tidInWarp / (WSUBN / TN);

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);

    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    // Allocate As/Bs base pointers once (needed by macro below).

    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    // ISSUE_TILE_NT_SN(stage, nIdx) — identical to Big NT's ISSUE_TILE_NT,
    // but N-bound is `N` (global) for cp.async zero-fill. Per-block N range
    // [n_begin, n_end) is enforced by tile loop.
    #define ISSUE_TILE_NT_SN(stage, nIdx) do {                                            \
        /* 2026-05-12 STAGE-2 coalesce (mirrors big NT ISSUE_TILE_NT A-load). */          \
        {                                                                                 \
            constexpr int WARPS_NTSN = NUM_THREADS / WARPSIZE;                            \
            constexpr int M_ROWS_PER_WARP_INST_NTSN = WARPSIZE / BK;                      \
            constexpr int M_ROWS_PER_WARP_NTSN = BM / WARPS_NTSN;                         \
            constexpr int INSTR_PER_WARP_NTSN =                                           \
                M_ROWS_PER_WARP_NTSN / M_ROWS_PER_WARP_INST_NTSN;                         \
            static_assert(WARPSIZE % BK == 0, "WARPSIZE divisible by BK");                \
            static_assert(BM % WARPS_NTSN == 0, "BM divisible by warp count");            \
            int _warp = threadIdx.x / WARPSIZE;                                           \
            int _lane = threadIdx.x % WARPSIZE;                                           \
            int _m_in_warp = _lane / BK;                                                  \
            int _n_local = _lane % BK;                                                    \
            _Pragma("unroll")                                                             \
            for (int _it = 0; _it < INSTR_PER_WARP_NTSN; _it++) {                         \
                int _m_local = _warp * M_ROWS_PER_WARP_NTSN                               \
                               + _it * M_ROWS_PER_WARP_INST_NTSN + _m_in_warp;            \
                int _g_m = pid_m * BM + _m_local;                                         \
                int _g_n = (nIdx) + _n_local;                                             \
                unsigned _dst = As_base + ((stage) * A_STAGE                              \
                    + _n_local * (BM + SMEM_A_PAD) + _m_local)                            \
                    * (unsigned)sizeof(float);                                            \
                int _bytes = (_g_m < M && _g_n < N) ? 4 : 0;                              \
                const float* _src = A + (long long)_g_m * N + _g_n;                       \
                asm volatile(                                                             \
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"                      \
                    :: "r"(_dst), "l"(_src), "r"(_bytes));                                \
            }                                                                             \
        }                                                                                 \
        for (int _off = 0; _off + ROW_STRIDE_B <= BK; _off += ROW_STRIDE_B) {             \
            int _n_local = innerRowB + _off;                                              \
            int _k_base = innerColB * 4;                                                  \
            int _g_n = (nIdx) + _n_local;                                                 \
            int _g_k = pid_n * BN + _k_base;                                              \
            bool _pn = _g_n < N;                                                          \
            _Pragma("unroll")                                                             \
            for (int _i = 0; _i < 4; _i++) {                                              \
                unsigned _dst = Bs_base + ((stage) * B_STAGE                              \
                    + _n_local * (BN + SMEM_B_PAD) + _k_base + _i)                        \
                    * (unsigned)sizeof(float);                                            \
                int _bytes = (_pn && (_g_k + _i) < K_out) ? 4 : 0;                        \
                const float* _src = B + (long long)(_g_k + _i) * N + _g_n;                \
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"             \
                             :: "r"(_dst), "l"(_src), "r"(_bytes));                       \
            }                                                                             \
        }                                                                                 \
        asm volatile("cp.async.commit_group;\n");                                         \
    } while (0)

    int num_k_tiles = (n_end - n_begin + BK - 1) / BK;

    // 2026-05-12 STAGE-4: persistent CTA loop. fc/n_begin/n_end stay kernel-scoped.
    // 2026-05-13 — Stage-4 persistent CTA unwrapped. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring pre-Stage-4 register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[WMITER * TM * WNITER * TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    ISSUE_TILE_NT_SN(0, n_begin);

    int read_stage = 0;
    int write_stage = 1;
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(K_PIPE - 2));
        __syncthreads();

        int next_tile = tile + 1;
        if (next_tile < num_k_tiles) {
            int next_nIdx = n_begin + next_tile * BK;
            ISSUE_TILE_NT_SN(write_stage, next_nIdx);
        }

        float* As_rd = As_buf + read_stage * A_STAGE;
        float* Bs_rd = Bs_buf + read_stage * B_STAGE;
        float regM_next[WMITER * TM];
        float regN_next[WNITER * TN];
        #pragma unroll
        for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                regM[wSubRowIdx * TM + i] = As_rd[0 * (BM + SMEM_A_PAD) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
        #pragma unroll
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
            #pragma unroll
            for (int i = 0; i < TN; ++i)
                regN[wSubColIdx * TN + i] = Bs_rd[0 * (BN + SMEM_B_PAD) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            if (dotIdx + 1 < BK) {
                #pragma unroll
                for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                    #pragma unroll
                    for (int i = 0; i < TM; ++i)
                        regM_next[wSubRowIdx * TM + i] = As_rd[(dotIdx + 1) * (BM + SMEM_A_PAD) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int i = 0; i < TN; ++i)
                        regN_next[wSubColIdx * TN + i] = Bs_rd[(dotIdx + 1) * (BN + SMEM_B_PAD) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
            }
            // F-NT-FMA pin (2026-05-08): explicit __fmaf_rn for bit-exact
            // match with CPU `_mm256_fmadd_ps`. sgemm_bi_nt_splitn_big_partial.
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int resIdxM = 0; resIdxM < TM; ++resIdxM)
                        #pragma unroll
                        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN)
                                      + wSubColIdx * TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * TM + resIdxM],
                                regN[wSubColIdx * TN + resIdxN],
                                threadResults[idx]);
                        }
            if (dotIdx + 1 < BK) {
                #pragma unroll
                for (int i = 0; i < WMITER * TM; ++i) regM[i] = regM_next[i];
                #pragma unroll
                for (int i = 0; i < WNITER * TN; ++i) regN[i] = regN_next[i];
            }
        }
        read_stage = (read_stage + 1) % K_PIPE;
        write_stage = (write_stage + 1) % K_PIPE;
    }
    #undef ISSUE_TILE_NT_SN

    // Epilogue: OVERWRITE partial[fc, :, :] slot. Output shape is (M, K_out)
    // with row-stride K_out. No __stwt (reducer reads all fc's — not streaming).
    float* partial_chunk = partial + (long long)fc * M * K_out;
    float* partial_warp = partial_chunk + (pid_m * BM + warpRow * WM) * K_out + pid_n * BN + warpCol * WN;

    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* P_sub = partial_warp + wSubRowIdx * WSUBM * K_out + wSubColIdx * WSUBN;
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                int g_row = pid_m * BM + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + resIdxM;
                if (g_row >= M) continue;
                for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    int g_col = pid_n * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + resIdxN;
                    int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
                    // Scalar fallback for right-edge OR non-%4 K_out (handles K_out=257).
                    if (g_col + 3 >= K_out || (K_out % 4) != 0) {
                        for (int j = 0; j < 4 && g_col + j < K_out; j++) {
                            P_sub[(threadRowInWarp * TM + resIdxM) * K_out +
                                  threadColInWarp * TN + resIdxN + j] =
                                threadResults[idx + j];
                        }
                        continue;
                    }
                    float4 out;
                    out.x = threadResults[idx + 0];
                    out.y = threadResults[idx + 1];
                    out.z = threadResults[idx + 2];
                    out.w = threadResults[idx + 3];
                    reinterpret_cast<float4*>(
                        &P_sub[(threadRowInWarp * TM + resIdxM) * K_out +
                               threadColInWarp * TN + resIdxN])[0] = out;
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (nt_splitn_big_partial)
}

// ============================================================================
// Slim-N variant: BM=128, BN=64, BK=32, WM=64, WN=32, WNITER=2
// Used for narrow N (N <= 512): SALE state/action (N=256), Mamba projections
// (N=128/256), SimbaV2 w2 (N=512). BN=64 cuts idle SMs on narrow outputs;
// BK=32 compensates (32-element reduction tile vs 16 for Big).
// Static smem: 2*(32*128+32*64)*4 = 48 KB ← fits 48 KB default static limit.
// ============================================================================
#undef BM
#undef BN
#undef BK
#undef WM
#undef WN
#undef WNITER
#undef TM
#undef TN
#undef WMITER
#undef WSUBM
#undef WSUBN
#undef ROW_STRIDE_A
#undef ROW_STRIDE_B
#undef NUM_THREADS  // Opt1: Big changed to 256; Slim preserves 128

#define NUM_THREADS 128
#define BM 128
#define BN 64
#define BK 32
#define WM 64
#define WN 32
#define WNITER 2
#define TM 8
#define TN 4
#define WMITER ((WM * WN) / (WARPSIZE * TM * TN * WNITER))
#define WSUBM (WM / WMITER)
#define WSUBN (WN / WNITER)
#define ROW_STRIDE_A ((NUM_THREADS * 4) / BK)
#define ROW_STRIDE_B (NUM_THREADS / (BN / 4))

// ============================================================================
// Forward: C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C + bias
// ============================================================================
// __launch_bounds__(128, 3) — target 3 blocks/SM on Ada sm_89.
// 128 regs × 128 threads × 3 blocks = 49152 regs (< 64K/SM) ✓
// Smem 24KB × 2 = 48KB (static limit) — 3 blocks requires 99KB dynamic opt-in (Phase 4.2).
// At (128, 3) without dynamic opt-in, effective occupancy = 2 blocks/SM due to smem limit.
// Phase 0.1 ptxas: 128 regs, 0 spill.
extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_bi_nn_slim(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float alpha, float beta,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    // α=1 contract for bias-IN-FMA seed. Same contract and rationale as
    // sgemm_bi_nn — see the bias-pre-seed block in the Big NN kernel.
    assert(alpha == 1.0f || bias == nullptr);
    // Smem: A transposed [BK * BM], B normal [BK * BN]
    __shared__ float As[BK * (BM + SMEM_A_PAD)];  // 16 * 128 = 2048 floats = 8KB
    __shared__ float Bs[BK * (BN + SMEM_B_PAD)];  // 16 * 128 = 2048 floats = 8KB

    // SGB_GROUP_M L2 swizzle — count tiles once.
    int num_pid_m = (M + BM - 1) / BM;
    int num_pid_n = (N + BN - 1) / BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    // Warp and thread placement
    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (BN / WN);  // 0 or 1
    int warpRow = warpIdx / (BN / WN);  // 0 or 1
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (WSUBN / TN);  // tid % 4
    int threadRowInWarp = tidInWarp / (WSUBN / TN);  // tid / 4

    // A load indices (float4 coalesced along K)
    int innerRowA = threadIdx.x / (BK / 4);  // tid / 4, 0..31
    int innerColA = threadIdx.x % (BK / 4);  // tid % 4, 0..3

    // B load indices (float4 coalesced along N)
    int innerRowB = threadIdx.x / (BN / 4);  // tid / 32, 0..3
    int innerColB = threadIdx.x % (BN / 4);  // tid % 32, 0..31

    // Working-set registers (reset by mainloop per tile).
    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    // 2026-05-12 STAGE-3 slim NN A coalesce (mirrors big NN ISSUE_TILE A-load).
    constexpr int WARPS_NNSLIM = NUM_THREADS / WARPSIZE;
    constexpr int M_ROWS_PER_WARP_INST_NNSLIM = WARPSIZE / BK;
    constexpr int M_ROWS_PER_WARP_NNSLIM = BM / WARPS_NNSLIM;
    constexpr int INSTR_PER_WARP_NNSLIM =
        M_ROWS_PER_WARP_NNSLIM / M_ROWS_PER_WARP_INST_NNSLIM;
    static_assert(WARPSIZE % BK == 0, "WARPSIZE divisible by BK (slim NN)");
    static_assert(BM % WARPS_NNSLIM == 0, "BM divisible by warps (slim NN)");
    int _warp = threadIdx.x / WARPSIZE;
    int _lane = threadIdx.x % WARPSIZE;
    int _m_in_warp_nn = (M_ROWS_PER_WARP_INST_NNSLIM > 0) ? (_lane / BK) : 0;
    int _k_local_lane = _lane % BK;

    // 2026-05-12 STAGE-4: persistent CTA loop for slim NN.
    // 2026-05-13 — Stage-4 persistent CTA unwrapped. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring pre-Stage-4 register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        // Bias-IN-FMA seed at K=0 (see Big NN bias-pre-seed block).
        float threadResults[WMITER * TM * WNITER * TN];

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        // Bias pre-seed. g_col mirrors epilog write (see L1939, L1969).
        if (bias != nullptr) {
            #pragma unroll
            for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                #pragma unroll
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    int g_col = pid_n * BN + warpCol * WN + wSubColIdx * WSUBN +
                                threadColInWarp * TN + resIdxN;
                    float b_val = (g_col < N) ? bias[g_col] : 0.0f;
                    #pragma unroll
                    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                        #pragma unroll
                        for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                      wSubColIdx * TN + resIdxN;
                            threadResults[idx] = b_val;
                        }
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < WMITER * TM * WNITER * TN; ++i) {
                threadResults[i] = 0.0f;
            }
        }

        const float* A_block = A + pid_m * BM * lda;
        const float* B_block = B + pid_n * BN;
        float* C_warp = C + (pid_m * BM + warpRow * WM) * ldc + pid_n * BN + warpCol * WN;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        #pragma unroll
        for (int _it = 0; _it < INSTR_PER_WARP_NNSLIM; _it++) {
            int _m_local = _warp * M_ROWS_PER_WARP_NNSLIM
                           + _it * M_ROWS_PER_WARP_INST_NNSLIM + _m_in_warp_nn;
            int _g_row = pid_m * BM + _m_local;
            int _g_col = bkIdx + _k_local_lane;
            unsigned _dst = As_base
                + (_k_local_lane * (BM + SMEM_A_PAD) + _m_local)
                * (unsigned)sizeof(float);
            if (_g_row < M && _g_col < K) {
                const float* _src = A_block + _m_local * lda + _k_local_lane;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(_dst), "l"(_src));
            } else {
                As[_k_local_lane * (BM + SMEM_A_PAD) + _m_local] = 0.0f;
            }
        }

        // Load B: cp.async.ca.shared.global 16B (contiguous src+dst). Scalar OOB fallback.
        for (int offset = 0; offset + ROW_STRIDE_B <= BK; offset += ROW_STRIDE_B) {
            int g_row = bkIdx + innerRowB + offset;
            int g_col = pid_n * BN + innerColB * 4;
            unsigned dst = Bs_base + ((innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
            if (g_row < K && g_col + 3 < N && (ldb % 4 == 0)) {
                const float* src = B_block + (innerRowB + offset) * ldb + innerColB * 4;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 0] = (g_row < K && g_col + 0 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 1] = (g_row < K && g_col + 1 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 2] = (g_row < K && g_col + 2 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 3] = (g_row < K && g_col + 3 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 3] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        // Compute: warptile matmul from smem
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load A column into registers (transposed smem = contiguous)
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (int i = 0; i < TM; ++i) {
                    regM[wSubRowIdx * TM + i] =
                        As[dotIdx * (BM + SMEM_A_PAD) + warpRow * WM + wSubRowIdx * WSUBM +
                           threadRowInWarp * TM + i];
                }
            }
            // Load B row into registers
            for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (int i = 0; i < TN; ++i) {
                    regN[wSubColIdx * TN + i] =
                        Bs[dotIdx * (BN + SMEM_B_PAD) + warpCol * WN + wSubColIdx * WSUBN +
                           threadColInWarp * TN + i];
                }
            }
            // Outer product: 256 FMA
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            // F-NT-FMA pin (2026-05-08): explicit __fmaf_rn for bit-exact
                            // match with CPU `_mm256_fmadd_ps`.
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN)
                                      + wSubColIdx * TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * TM + resIdxM],
                                regN[wSubColIdx * TN + resIdxN],
                                threadResults[idx]);
                        }
                    }
                }
            }
        }

        A_block += BK;       // move BK columns right
        B_block += BK * ldb; // move BK rows down
        __syncthreads();
    }

    // Epilogue: write results with alpha, beta, bias (float4 stores)
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* C_sub = C_warp + wSubRowIdx * WSUBM * ldc + wSubColIdx * WSUBN;
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                int g_row = pid_m * BM + warpRow * WM + wSubRowIdx * WSUBM +
                            threadRowInWarp * TM + resIdxM;
                if (g_row >= M) continue;
                for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    int g_col = pid_n * BN + warpCol * WN + wSubColIdx * WSUBN +
                                threadColInWarp * TN + resIdxN;
                    // Fallback to scalar on column tail OR when ldc % 4 != 0.
                    // STG.128 needs 16-byte aligned address — row-stride in bytes
                    // (ldc * 4) must be a multiple of 16, so ldc must be a multiple
                    // of 4. Otherwise odd rows hit CUDA_ERROR_MISALIGNED_ADDRESS.
                    if (g_col + 3 >= N || (ldc & 3) != 0) {
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                      wSubColIdx * TN + resIdxN + j;
                            // Bias seeded at K=0 (see init).
                            float val = alpha * threadResults[idx];
                            if (beta != 0.0f) val += beta * C_sub[(threadRowInWarp * TM + resIdxM) * ldc + threadColInWarp * TN + resIdxN + j];
                            C_sub[(threadRowInWarp * TM + resIdxM) * ldc + threadColInWarp * TN + resIdxN + j] = val;
                        }
                        continue;
                    }
                    float4 tmp;
                    if (beta != 0.0f) {
                        tmp = reinterpret_cast<float4*>(
                            &C_sub[(threadRowInWarp * TM + resIdxM) * ldc +
                                   threadColInWarp * TN + resIdxN])[0];
                    }
                    int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                              wSubColIdx * TN + resIdxN;
                    // Bias seeded at K=0 (see init).
                    float v0 = alpha * threadResults[idx + 0];
                    float v1 = alpha * threadResults[idx + 1];
                    float v2 = alpha * threadResults[idx + 2];
                    float v3 = alpha * threadResults[idx + 3];
                    if (beta != 0.0f) {
                        v0 += beta * tmp.x;
                        v1 += beta * tmp.y;
                        v2 += beta * tmp.z;
                        v3 += beta * tmp.w;
                    }
                    float4 out = {v0, v1, v2, v3};
                    reinterpret_cast<float4*>(
                        &C_sub[(threadRowInWarp * TM + resIdxM) * ldc +
                               threadColInWarp * TN + resIdxN])[0] = out;
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (slim NN)
}

// ============================================================================
// Phase 6 v2: Split-K Slim NN partial — wave-fill extension for underfilled
// Slim NN shapes. Identical per-block FMA order to sgemm_bi_nn_slim on its
// K-slice → bit-exact. Grid: (M_tiles * N_tiles, 1, F). blockIdx.z = fc ∈ [0, F).
// Each fc owns K-chunk [fc*K_chunk, min(K, (fc+1)*K_chunk)) and writes to
// partial[fc, m, n].
//
// Caller follows with sgemm_bi_splitk_reduce(y, partial, bias, null_tail,
// null_tail, alpha, M, N, F, 0, 0, 0) — reducer applies alpha + bias,
// overwrites y (x_tail_ptr==null path).
//
// Constraints (must hold for bit-exactness):
//   - K_chunk % BK (=32) == 0 (enforced by dispatcher)
//   - K % 32 == 0 (enforced by dispatcher — no K tail)
//   - No alpha / bias / beta in this kernel — raw tile sums only
//   - Static 16 KB smem (same as Slim NN) — no dynamic-smem attribute needed
//
// Mirror of sgemm_bi_nn_splitk_big_partial (Phase 6 v1) but for BM=128 BN=64
// BK=32 tile. This is the tile that actually fires on b=64 production GEMMs.
// ============================================================================
extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_bi_nn_splitk_slim_partial(
    float* __restrict__ partial,       // [F * M * N] — unique slot per fc
    const float* __restrict__ A,       // [M, K_full]
    const float* __restrict__ B,       // [K_full, N]
    int M, int N, int K,
    int lda, int ldb,
    int K_chunk                         // must be multiple of BK=32
) {
    __shared__ float As[BK * (BM + SMEM_A_PAD)];
    __shared__ float Bs[BK * (BN + SMEM_B_PAD)];

    // Decode fc from z-axis; early exit if fc is out of K-range.
    int fc = blockIdx.z;
    int k_begin = fc * K_chunk;
    if (k_begin >= K) return;
    int k_end = min(K, k_begin + K_chunk);

    // SGB_GROUP_M L2 swizzle (identical to Slim NN)
    int num_pid_m = (M + BM - 1) / BM;
    int num_pid_n = (N + BN - 1) / BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (BN / WN);
    int warpRow = warpIdx / (BN / WN);
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (WSUBN / TN);
    int threadRowInWarp = tidInWarp / (WSUBN / TN);

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);

    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    // 2026-05-12 STAGE-3 slim NN splitk A coalesce.
    constexpr int WARPS_NNSKP = NUM_THREADS / WARPSIZE;
    constexpr int M_ROWS_PER_WARP_INST_NNSKP = WARPSIZE / BK;
    constexpr int M_ROWS_PER_WARP_NNSKP = BM / WARPS_NNSKP;
    constexpr int INSTR_PER_WARP_NNSKP =
        M_ROWS_PER_WARP_NNSKP / M_ROWS_PER_WARP_INST_NNSKP;
    static_assert(WARPSIZE % BK == 0, "WARPSIZE divisible by BK (slim NN splitk)");
    static_assert(BM % WARPS_NNSKP == 0, "BM divisible by warps (slim NN splitk)");
    int _warp = threadIdx.x / WARPSIZE;
    int _lane = threadIdx.x % WARPSIZE;
    int _m_in_warp_nnsk = _lane / BK;
    int _k_local_lane = _lane % BK;

    // 2026-05-12 STAGE-4: persistent CTA loop. fc/k_begin/k_end stay kernel-scoped.
    // 2026-05-13 — Stage-4 persistent CTA unwrapped. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring pre-Stage-4 register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[WMITER * TM * WNITER * TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        const float* A_block = A + pid_m * BM * lda + k_begin;
        const float* B_block = B + (long long)k_begin * ldb + pid_n * BN;

    for (int bkIdx = k_begin; bkIdx < k_end; bkIdx += BK) {
        #pragma unroll
        for (int _it = 0; _it < INSTR_PER_WARP_NNSKP; _it++) {
            int _m_local = _warp * M_ROWS_PER_WARP_NNSKP
                           + _it * M_ROWS_PER_WARP_INST_NNSKP + _m_in_warp_nnsk;
            int _g_row = pid_m * BM + _m_local;
            int _g_col = bkIdx + _k_local_lane;
            unsigned _dst = As_base
                + (_k_local_lane * (BM + SMEM_A_PAD) + _m_local)
                * (unsigned)sizeof(float);
            if (_g_row < M && _g_col < k_end) {
                const float* _src = A_block + _m_local * lda + _k_local_lane;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(_dst), "l"(_src));
            } else {
                As[_k_local_lane * (BM + SMEM_A_PAD) + _m_local] = 0.0f;
            }
        }

        // Load B: contiguous cp.async.16B with scalar OOB fallback.
        for (int offset = 0; offset + ROW_STRIDE_B <= BK; offset += ROW_STRIDE_B) {
            int g_row = bkIdx + innerRowB + offset;
            int g_col = pid_n * BN + innerColB * 4;
            unsigned dst = Bs_base + ((innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
            if (g_row < k_end && g_col + 3 < N && (ldb % 4 == 0)) {
                const float* src = B_block + (innerRowB + offset) * ldb + innerColB * 4;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 0] = (g_row < k_end && g_col + 0 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 1] = (g_row < k_end && g_col + 1 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 2] = (g_row < k_end && g_col + 2 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 3] = (g_row < k_end && g_col + 3 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 3] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        // Compute: same warptile matmul as Slim NN — identical FMA order.
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (int i = 0; i < TM; ++i) {
                    regM[wSubRowIdx * TM + i] =
                        As[dotIdx * (BM + SMEM_A_PAD) + warpRow * WM + wSubRowIdx * WSUBM +
                           threadRowInWarp * TM + i];
                }
            }
            for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (int i = 0; i < TN; ++i) {
                    regN[wSubColIdx * TN + i] =
                        Bs[dotIdx * (BN + SMEM_B_PAD) + warpCol * WN + wSubColIdx * WSUBN +
                           threadColInWarp * TN + i];
                }
            }
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            // F-NT-FMA pin (2026-05-08): explicit __fmaf_rn for bit-exact
                            // match with CPU `_mm256_fmadd_ps`.
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN)
                                      + wSubColIdx * TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * TM + resIdxM],
                                regN[wSubColIdx * TN + resIdxN],
                                threadResults[idx]);
                        }
                    }
                }
            }
        }

        // Advance A and B pointers to next BK-column tile (same as Slim NN).
        A_block += BK;
        B_block += BK * ldb;
        __syncthreads();
    }

    // Epilogue: OVERWRITE partial[fc, :, :]. No alpha, no bias, no beta.
    float* partial_chunk = partial + (long long)fc * M * N;
    float* partial_warp = partial_chunk + (pid_m * BM + warpRow * WM) * N + pid_n * BN + warpCol * WN;

    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* P_sub = partial_warp + wSubRowIdx * WSUBM * N + wSubColIdx * WSUBN;
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                int g_row = pid_m * BM + warpRow * WM + wSubRowIdx * WSUBM +
                            threadRowInWarp * TM + resIdxM;
                if (g_row >= M) continue;
                for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    int g_col = pid_n * BN + warpCol * WN + wSubColIdx * WSUBN +
                                threadColInWarp * TN + resIdxN;
                    int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                              wSubColIdx * TN + resIdxN;
                    // Scalar fallback for right-edge OR non-%4 N (hardens audit M2).
                    if (g_col + 3 >= N || (N % 4) != 0) {
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            P_sub[(threadRowInWarp * TM + resIdxM) * N +
                                  threadColInWarp * TN + resIdxN + j] =
                                threadResults[idx + j];
                        }
                        continue;
                    }
                    float4 out;
                    out.x = threadResults[idx + 0];
                    out.y = threadResults[idx + 1];
                    out.z = threadResults[idx + 2];
                    out.w = threadResults[idx + 3];
                    reinterpret_cast<float4*>(
                        &P_sub[(threadRowInWarp * TM + resIdxM) * N +
                               threadColInWarp * TN + resIdxN])[0] = out;
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (nn_splitk_slim_partial)
}

// ============================================================================
// Backward dW (TN): C[K,N] += alpha * A^T[K,M] @ B[M,N]
// ============================================================================
// A = X_saved [M, K] — read transposed
// B = dY [M, N]
// C = dW [K, N] — accumulated
// Output tile [BM, BN] over (K, N). M is reduction axis.
// __launch_bounds__(128, 2) — target 2 blocks/SM (Phase 0.1 ptxas: 128 regs, 0 spill).
// 2 blocks × 24KB smem = 48KB static limit exactly.
extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_bi_tn_slim(
    float* __restrict__ C,
    const float* __restrict__ A,  // X [M, K]
    const float* __restrict__ B,  // dY [M, N]
    float alpha,
    int M_red,    // batch (reduction axis)
    int K_out,    // output rows
    int N         // output cols
) {
    __shared__ float As[BK * (BM + SMEM_A_PAD)];
    __shared__ float Bs[BK * (BN + SMEM_B_PAD)];

    int num_pid_m = (K_out + BM - 1) / BM;
    int num_pid_n = (N + BN - 1) / BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (BN / WN);
    int warpRow = warpIdx / (BN / WN);
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (WSUBN / TN);
    int threadRowInWarp = tidInWarp / (WSUBN / TN);

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);

    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    // 2026-05-12 STAGE-4: persistent CTA loop for slim TN.
    // 2026-05-13 — Stage-4 persistent CTA unwrapped. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring pre-Stage-4 register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[WMITER * TM * WNITER * TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        float* C_warp = C + (pid_m * BM + warpRow * WM) * N + pid_n * BN + warpCol * WN;

    // Phase B1: cp.async single-buffer load path (A transposed + B).
    // Replaces synchronous scalar loads → async global→shared DMA.
    // Frees register staging, reduces I$ pressure, potentially overlaps DMA latency.
    // .ca = L1 cache (A/B reused across K tiles per block). sm_80+ required.
    // Determinism: identical bytes in identical smem locations as scalar path.
    // OOB: scalar write of 0.0f (matches scalar path's explicit zero-fill).
    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    // 2026-05-12 STAGE-1 slim TN A coalesce (mirrors big TN ISSUE_TILE_TN).
    constexpr int WARPS_TNSLIM = NUM_THREADS / WARPSIZE;
    constexpr int ROWS_PER_WARP_TNSLIM = BK / WARPS_TNSLIM;
    static_assert(BK % WARPS_TNSLIM == 0, "BK divisible by warps (slim TN)");
    static_assert(BM % (WARPSIZE * 4) == 0, "BM divisible by 32*4 (slim TN)");
    int _warp = threadIdx.x / WARPSIZE;
    int _lane = threadIdx.x % WARPSIZE;
    for (int mIdx = 0; mIdx < M_red; mIdx += BK) {
        #pragma unroll
        for (int _r = 0; _r < ROWS_PER_WARP_TNSLIM; _r++) {
            int k_local = _warp * ROWS_PER_WARP_TNSLIM + _r;
            int m_local = _lane * 4;
            int _g_m = mIdx + k_local;
            int _g_k = pid_m * BM + m_local;
            unsigned _dst = As_base
                + (k_local * (BM + SMEM_A_PAD) + m_local)
                * (unsigned)sizeof(float);
            bool _full16 = (_g_m < M_red) && (_g_k + 3 < K_out) && ((K_out & 3) == 0);
            if (_full16) {
                const float* _src = A + (long long)_g_m * K_out + _g_k;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(_dst), "l"(_src));
            } else {
                #pragma unroll
                for (int _i = 0; _i < 4; _i++) {
                    bool ok = (_g_m < M_red) && (_g_k + _i < K_out);
                    if (ok) {
                        const float* _src_e = A + (long long)_g_m * K_out + _g_k + _i;
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                                     :: "r"(_dst + (unsigned)_i * 4), "l"(_src_e));
                    } else {
                        As[k_local * (BM + SMEM_A_PAD) + m_local + _i] = 0.0f;
                    }
                }
            }
        }

        // Load B via cp.async.ca.shared.global 16B (float4, contiguous).
        // Scalar fallback for edge / non-%4 N.
        for (int offset = 0; offset + ROW_STRIDE_B <= BK; offset += ROW_STRIDE_B) {
            int g_m = mIdx + innerRowB + offset;
            int g_n = pid_n * BN + innerColB * 4;
            unsigned dst = Bs_base + ((innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
            if (g_m < M_red && g_n + 3 < N && (N % 4 == 0)) {
                const float* src = B + ((long long)g_m) * N + pid_n * BN + innerColB * 4;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 0] = (g_m < M_red && g_n + 0 < N) ? B[g_m * N + pid_n * BN + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 1] = (g_m < M_red && g_n + 1 < N) ? B[g_m * N + pid_n * BN + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 2] = (g_m < M_red && g_n + 2 < N) ? B[g_m * N + pid_n * BN + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (BN + SMEM_B_PAD) + innerColB * 4 + 3] = (g_m < M_red && g_n + 3 < N) ? B[g_m * N + pid_n * BN + innerColB * 4 + 3] : 0.0f;
            }
        }
        // Commit + wait_all → guarantees all async loads visible before compute.
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                for (int i = 0; i < TM; ++i)
                    regM[wSubRowIdx * TM + i] = As[dotIdx * (BM + SMEM_A_PAD) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
            for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                for (int i = 0; i < TN; ++i)
                    regN[wSubColIdx * TN + i] = Bs[dotIdx * (BN + SMEM_B_PAD) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
            // F-NT-FMA pin (2026-05-08): explicit __fmaf_rn for bit-exact
            // match with CPU `_mm256_fmadd_ps`. Same fix class as F-09a
            // (RoPE backward FMA pin).
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                    for (int resIdxM = 0; resIdxM < TM; ++resIdxM)
                        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN)
                                      + wSubColIdx * TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * TM + resIdxM],
                                regN[wSubColIdx * TN + resIdxN],
                                threadResults[idx]);
                        }
        }
        __syncthreads();
    }

    // Epilogue: accumulate into dW
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* C_sub = C_warp + wSubRowIdx * WSUBM * N + wSubColIdx * WSUBN;
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                int g_row = pid_m * BM + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + resIdxM;
                if (g_row >= K_out) continue;
                for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    int g_col = pid_n * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + resIdxN;
                    // Fallback to scalar on tail OR when N (row-stride) % 4 != 0
                    // (STG.128 / LDG.128 need 16-byte aligned address).
                    if (g_col + 3 >= N || (N & 3) != 0) {
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN + j;
                            C_sub[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN + j] += alpha * threadResults[idx];
                        }
                        continue;
                    }
                    float4 old = reinterpret_cast<float4*>(&C_sub[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0];
                    int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
                    old.x += alpha * threadResults[idx + 0];
                    old.y += alpha * threadResults[idx + 1];
                    old.z += alpha * threadResults[idx + 2];
                    old.w += alpha * threadResults[idx + 3];
                    reinterpret_cast<float4*>(&C_sub[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0] = old;
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (slim TN)
}

// ============================================================================
// Backward dX (NT): C[M,K] = alpha * A[M,N] @ B^T[N,K]
// ============================================================================
// A = dY [M, N]
// B = W [K, N] — read transposed as W^T[N,K]
// C = dX [M, K] — overwrite
// __launch_bounds__(128, 2) — target 2 blocks/SM (Phase 0.1 ptxas: 128 regs, 0 spill).
extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_bi_nt_slim(
    float* __restrict__ C,
    const float* __restrict__ A,  // dY [M, N]
    const float* __restrict__ B,  // W [K, N]
    float alpha,
    int M,        // output rows
    int N,        // reduction axis
    int K_out     // output cols
) {
    __shared__ float As[BK * (BM + SMEM_A_PAD)];
    __shared__ float Bs[BK * (BN + SMEM_B_PAD)];

    int num_pid_m = (M + BM - 1) / BM;
    int num_pid_n = (K_out + BN - 1) / BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (BN / WN);
    int warpRow = warpIdx / (BN / WN);
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (WSUBN / TN);
    int threadRowInWarp = tidInWarp / (WSUBN / TN);

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);

    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    // 2026-05-12 STAGE-4: persistent CTA loop for slim NT.
    // 2026-05-13 — Stage-4 persistent CTA unwrapped. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring pre-Stage-4 register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[WMITER * TM * WNITER * TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        float* C_warp = C + (pid_m * BM + warpRow * WM) * K_out + pid_n * BN + warpCol * WN;

    // Phase B2: cp.async loads for NT backward dX.
    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    // 2026-05-12 STAGE-2 slim NT A coalesce.
    constexpr int WARPS_NTSLIM = NUM_THREADS / WARPSIZE;
    constexpr int M_ROWS_PER_WARP_INST_NTSL = WARPSIZE / BK;
    constexpr int M_ROWS_PER_WARP_NTSL = BM / WARPS_NTSLIM;
    constexpr int INSTR_PER_WARP_NTSL =
        M_ROWS_PER_WARP_NTSL / M_ROWS_PER_WARP_INST_NTSL;
    static_assert(WARPSIZE % BK == 0, "WARPSIZE divisible by BK (slim NT)");
    static_assert(BM % WARPS_NTSLIM == 0, "BM divisible by warps (slim NT)");
    int _warp = threadIdx.x / WARPSIZE;
    int _lane = threadIdx.x % WARPSIZE;
    int _m_in_warp_ntsl = _lane / BK;
    int _n_local_lane = _lane % BK;
    for (int nIdx = 0; nIdx < N; nIdx += BK) {
        #pragma unroll
        for (int _it = 0; _it < INSTR_PER_WARP_NTSL; _it++) {
            int _m_local = _warp * M_ROWS_PER_WARP_NTSL
                           + _it * M_ROWS_PER_WARP_INST_NTSL + _m_in_warp_ntsl;
            int _g_m = pid_m * BM + _m_local;
            int _g_n = nIdx + _n_local_lane;
            unsigned _dst = As_base
                + (_n_local_lane * (BM + SMEM_A_PAD) + _m_local)
                * (unsigned)sizeof(float);
            if (_g_m < M && _g_n < N) {
                const float* _src = A + (long long)_g_m * N + _g_n;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(_dst), "l"(_src));
            } else {
                As[_n_local_lane * (BM + SMEM_A_PAD) + _m_local] = 0.0f;
            }
        }

        // Load B^T (W): source stride N (non-contiguous K-rows), dest contiguous → 4× cp.async.4B.
        for (int offset = 0; offset + ROW_STRIDE_B <= BK; offset += ROW_STRIDE_B) {
            int n_local = innerRowB + offset;
            int k_base = innerColB * 4;
            int g_n = nIdx + n_local;
            int g_k = pid_n * BN + k_base;
            bool pn = g_n < N;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                unsigned dst = Bs_base + (n_local * (BN + SMEM_B_PAD) + k_base + i) * (unsigned)sizeof(float);
                if (pn && (g_k + i) < K_out) {
                    const float* src = B + ((long long)(g_k + i)) * N + g_n;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                                 :: "r"(dst), "l"(src));
                } else {
                    Bs[n_local * (BN + SMEM_B_PAD) + k_base + i] = 0.0f;
                }
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                for (int i = 0; i < TM; ++i)
                    regM[wSubRowIdx * TM + i] = As[dotIdx * (BM + SMEM_A_PAD) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
            for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                for (int i = 0; i < TN; ++i)
                    regN[wSubColIdx * TN + i] = Bs[dotIdx * (BN + SMEM_B_PAD) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
            // F-NT-FMA pin (2026-05-08): explicit __fmaf_rn for bit-exact
            // match with CPU `_mm256_fmadd_ps`. Same fix class as F-09a
            // (RoPE backward FMA pin).
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                    for (int resIdxM = 0; resIdxM < TM; ++resIdxM)
                        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN)
                                      + wSubColIdx * TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * TM + resIdxM],
                                regN[wSubColIdx * TN + resIdxN],
                                threadResults[idx]);
                        }
        }
        __syncthreads();
    }

    // Epilogue: overwrite dX
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* C_sub = C_warp + wSubRowIdx * WSUBM * K_out + wSubColIdx * WSUBN;
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                int g_row = pid_m * BM + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + resIdxM;
                if (g_row >= M) continue;
                for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    int g_col = pid_n * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + resIdxN;
                    // float4 write only when K_out is %4-aligned (K_out=257 → scalar).
                    if (g_col + 3 >= K_out || (K_out % 4 != 0)) {
                        int idx_base = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
                        for (int j = 0; j < 4 && g_col + j < K_out; j++) {
                            __stwt(&C_sub[(threadRowInWarp * TM + resIdxM) * K_out + threadColInWarp * TN + resIdxN + j], alpha * threadResults[idx_base + j]);
                        }
                        continue;
                    }
                    int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
                    float4 out = {
                        alpha * threadResults[idx + 0],
                        alpha * threadResults[idx + 1],
                        alpha * threadResults[idx + 2],
                        alpha * threadResults[idx + 3]
                    };
                    // Phase 2.9: __stwt — streaming store. NT backward dX is OVERWRITE (no accumulation),
                    // so marking C lines evict-first is safe and prevents C writes from evicting A staging / B working set.
                    __stwt(reinterpret_cast<float4*>(&C_sub[(threadRowInWarp * TM + resIdxM) * K_out + threadColInWarp * TN + resIdxN]), out);
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (slim NT)
}

// ============================================================================
// Phase 2.2 — GEMV-N1 NN (forward, N=1): Y[M] = alpha * X[M,K] @ W[K] + beta*Y + bias
// ============================================================================
// ============================================================================
// Ultra-Thin-M NN forward: Y[M,N] = X[M,K] @ W[K,N] + bias
// ============================================================================
// Shape-A fix: covers M ∈ {1..31} (actor inference path, single-env rollout).
// Below all dispatch gates: batch<32 (Split-K min), <128 (Big/Slim min).
// Grid: (ceil(N/32), M, 1). Each block handles one (m, n_tile=32cols).
// 256 threads (8 warps). Warp w handles K-slab [w*K/8, (w+1)*K/8).
// Each lane produces 1 output column. Fixed-order 8-partial tree reduce
// preserves bit-exact determinism (same pattern as matvec_bi, mamba-rs).
//
// Bias-fold rationale: bias is added POST-tree-reduce: `val = α·sum;
// val += bias[col]`. CPU mirror `ultra_thin_sgemm_nn` is called via
// `blas_batch::sgemm_forward` with C pre-seeded to bias and
// `accumulate=true`, producing `prev + sum` where prev = bias. At α=1
// (production constraint), GPU computes `sum + bias` and CPU computes
// `bias + sum` — commutative under IEEE 754 f32 FADD → bit-exact.
// Seeding bias INTO one of the 8 warp accumulators (Big NN / Slim NN
// K=0 pattern) would break the fixed 8-partial tree-reduce structure
// and REGRESS the bit-exact contract. The bias-POST single-add IS the
// canonical unify for this kernel.
extern "C" __global__ __launch_bounds__(256, 4)
void sgemm_bi_nn_ultra_thin(
    float* __restrict__ Y,          // [M, N] output (ldc stride)
    const float* __restrict__ X,    // [M, K] input (lda stride)
    const float* __restrict__ W,    // [K, N] weights (ldb stride)
    const float* __restrict__ bias, // [N] optional
    float alpha, float beta,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;       // 0..7
    const int lane = tid & 31;        // 0..31
    const int n_tile = blockIdx.x;    // which 32-col tile of N
    const int m = blockIdx.y;         // which row
    if (m >= M) return;

    const int col = n_tile * 32 + lane;  // output column for this lane

    // Cooperative smem load of X[m, 0..K).
    extern __shared__ float smem_x[];
    for (int k = tid; k < K; k += blockDim.x) {
        smem_x[k] = X[m * lda + k];
    }
    __syncthreads();

    // Each warp takes a K-slab of size ceil(K/8). Fixed partition (not atomic).
    const int K_per_warp = (K + 7) / 8;
    const int k_start = warp * K_per_warp;
    const int k_end = (k_start + K_per_warp > K) ? K : (k_start + K_per_warp);

    float acc = 0.0f;
    if (col < N) {
        // Per-thread accumulation in fixed k-order. Deterministic per output.
        // F-GEMV-FMA pin: explicit __fmaf_rn matches
        // F-NT-FMA pattern from 22 sibling kernels (2026-05-08 sweep). Under
        // current --fmad=true ptxas contracts to identical FFMA SASS; this
        // pin hardens against future ptxas / NVRTC toolchain choosing to
        // un-fuse under register pressure or contraction-mode change. Per
        // IEEE 754-2008 §5.4.1 fma is single-rounding vs FMUL+FADD two-rounds.
        for (int k = k_start; k < k_end; k++) {
            acc = __fmaf_rn(smem_x[k], W[k * ldb + col], acc);
        }
    }

    // 8 warp partials → smem → fixed-order tree reduce on warp 0.
    __shared__ float smem_partials[8 * 32];
    smem_partials[warp * 32 + lane] = acc;
    __syncthreads();

    if (warp == 0 && col < N) {
        float p0 = smem_partials[0 * 32 + lane];
        float p1 = smem_partials[1 * 32 + lane];
        float p2 = smem_partials[2 * 32 + lane];
        float p3 = smem_partials[3 * 32 + lane];
        float p4 = smem_partials[4 * 32 + lane];
        float p5 = smem_partials[5 * 32 + lane];
        float p6 = smem_partials[6 * 32 + lane];
        float p7 = smem_partials[7 * 32 + lane];
        // Fixed tree: ((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7))
        float s01 = p0 + p1;
        float s23 = p2 + p3;
        float s45 = p4 + p5;
        float s67 = p6 + p7;
        float s0123 = s01 + s23;
        float s4567 = s45 + s67;
        float sum = s0123 + s4567;

        float val = alpha * sum;
        if (bias != nullptr) val += bias[col];
        if (beta != 0.0f) val += beta * Y[m * ldc + col];
        Y[m * ldc + col] = val;
    }
}

// Specialized for output vector (N=1). Replaces cuBLAS fallback for actor heads
// (mean_head.w2, log_std_head.w2) where shape (M, K, 1) bypasses custom Big/Slim
// kernels (N < SGEMM_CUSTOM_MIN=128).
//
// Design: 128 threads/block, 4 warps. Each WARP handles 1 output row.
// Per thread: in-warp K-reduction with fixed k-stride=32. Warp-shuffle butterfly
// reduce (fixed offset 16→8→4→2→1) — deterministic, batch-invariant.
//
// Output Y[row] depends ONLY on X[row,:] and W[:] (no cross-warp reduction).
//
// Works for any M, K, alpha, beta, bias — fully runtime-parametric.
// K can be non-multiple-of-4 (scalar loads). No SMEM, no float4.
//
// Bias-fold rationale: same as `sgemm_bi_nn_ultra_thin` — bias is
// POST-tree-reduce, `val = α·acc; val += bias[0]`. CPU mirror's caller
// pre-seeds Y with bias so CPU computes `bias + sum`, GPU computes
// `sum + bias`; both bit-exact via f32 FADD commutativity at α=1.
// Seeding into one warp's K=0 acc would break the warp-shuffle butterfly
// reduce. No change needed.
extern "C" __global__ __launch_bounds__(128, 4)
void sgemm_bi_nn_gemv(
    float* __restrict__ Y,          // [M] — output, stride ldy in elements (usually 1)
    const float* __restrict__ X,    // [M, K]
    const float* __restrict__ W,    // [K] — weight vector
    const float* __restrict__ bias, // [1] or nullptr
    float alpha, float beta,
    int M, int K,
    int lda,  // stride of X rows, usually = K
    int ldy   // stride between Y[i] elements, usually = 1 (N=1 dense)
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 4 + warp;
    if (row >= M) return;

    // Each thread accumulates X[row, lane + n*32] * W[lane + n*32] for n=0..K/32-1.
    // Fixed in-thread k-order → deterministic per-thread accumulation.
    // F-GEMV-FMA pin: see sgemm_bi_nn_ultra_thin pin.
    float acc = 0.0f;
    const float* X_row = X + row * lda;
    for (int k = lane; k < K; k += 32) {
        acc = __fmaf_rn(X_row[k], W[k], acc);
    }

    // Warp-shuffle butterfly reduce — fixed tree, deterministic.
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (lane == 0) {
        float val = alpha * acc;
        if (bias != nullptr) val += bias[0];
        if (beta != 0.0f) val += beta * Y[row * ldy];
        Y[row * ldy] = val;
    }
}

// ============================================================================
// Phase 2.2b — GEMV-N1 TN (backward dW, N=1): dW[K] += alpha * X^T[K,M] @ dY[M]
// ============================================================================
// Specialized for weight gradient of N=1 output layer. Replaces cuBLAS fallback
// for actor mean_head.w2 / log_std_head.w2 backward pass.
//
// Design: 128 threads/block, 4 warps. Each WARP handles 1 output k.
// Per thread: in-warp M-reduction with fixed m-stride=32. Warp-shuffle butterfly
// reduce (fixed offset 16→8→4→2→1) — deterministic, batch-invariant.
//
// Output dW[k] depends ONLY on X[:,k] and dY[:] (no cross-warp ops).
// Grid blocks cover disjoint k ranges → no race on dW accumulation.
//
// TN semantics: beta=1 (accumulation into existing dW).
extern "C" __global__ __launch_bounds__(128, 4)
void sgemm_bi_tn_gemv(
    float* __restrict__ dW,         // [K_out] — weight gradient (accumulated)
    const float* __restrict__ X,    // [M_red, K_out]
    const float* __restrict__ dY,   // [M_red] — output gradient (N=1)
    float alpha,
    int M_red,   // batch (reduction axis)
    int K_out,   // number of output weights
    int lda,     // stride of X rows, usually = K_out
    int ldy      // stride between dY elements, usually = 1
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int k = blockIdx.x * 4 + warp;
    if (k >= K_out) return;

    // Each thread accumulates X[lane + n*32, k] * dY[lane + n*32] for n=0..M_red/32-1.
    // F-GEMV-FMA pin: see sgemm_bi_nn_ultra_thin pin.
    float acc = 0.0f;
    for (int m = lane; m < M_red; m += 32) {
        acc = __fmaf_rn(X[m * lda + k], dY[m * ldy], acc);
    }

    // Warp-shuffle butterfly reduce.
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (lane == 0) {
        dW[k] += alpha * acc;
    }
}

// ============================================================================
// Phase 2.2c — GEMV-N1 NT (backward dX, N=1): dX[M,K] = alpha * dY[M] @ W^T[K]
// ============================================================================
// Pure element-wise outer product — no reduction. dX[m,k] = alpha * dY[m] * W[k].
// Replaces cuBLAS fallback for input gradient of N=1 output layer.
//
// Trivially deterministic: each dX[m,k] computed by exactly one thread.
//
// NT semantics: beta=0 (overwrite dX).
extern "C" __global__ __launch_bounds__(256)
void sgemm_bi_nt_gemv(
    float* __restrict__ dX,         // [M, K] — output (overwritten)
    const float* __restrict__ dY,   // [M] — upstream gradient (N=1)
    const float* __restrict__ W,    // [K] — weight
    float alpha,
    int M, int K,
    int ldx,  // stride of dX rows, usually = K
    int ldy   // stride between dY elements, usually = 1
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * K;
    if (tid >= total) return;
    const int m = tid / K;
    const int k = tid - m * K;  // tid % K
    dX[m * ldx + k] = alpha * dY[m * ldy] * W[k];
}

// ============================================================================
// Phase 2.1 — Narrow-N NN (forward, N∈9..48): C[M,N] = alpha * A[M,K] @ B[K,N] + beta*C + bias
// ============================================================================
// Specialized for narrow N (9..48). Replaces cuBLAS fallback for critic qhead
// N=25 (TQC quantiles). Tile: BM=64 BN=32 BK=16, 128 threads, 2x2 warps.
//
// Scalar N-epilogue handles non-%4 N (e.g. N=25, last 1..3 cols written scalar).
// Scalar K-fallback for non-%4 K via lda%4 runtime check.
//
// Design mirrors Slim-N but smaller tile → 2x more grid blocks for M=4224, N<64
// (wave underfill protection at narrow N).
#define NBM 64
#define NBN 32
#define NBK 16
#define NWM 32
#define NWN 16
#define NWMITER 1
#define NWNITER 1
#define NTM 4
#define NTN 4
#define NNUM_THREADS 128
#define NWSUBM (NWM / NWMITER)   // 32
#define NWSUBN (NWN / NWNITER)   // 16
#define NROW_STRIDE_A ((NNUM_THREADS * 4) / NBK)  // 32
#define NROW_STRIDE_B (NNUM_THREADS / (NBN / 4))  // 16

extern "C" __global__ __launch_bounds__(NNUM_THREADS, 4)
void sgemm_bi_nn_narrow(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float alpha, float beta,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    int post_op  // reserved for Phase 1.4 fusion; 0 = none (currently unused)
) {
    (void)post_op;
    __shared__ float As[NBK * (NBM + SMEM_A_PAD)];
    __shared__ float Bs[NBK * (NBN + SMEM_B_PAD)];

    int num_pid_m = (M + NBM - 1) / NBM;
    int num_pid_n = (N + NBN - 1) / NBN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (NBN / NWN);  // 0 or 1
    int warpRow = warpIdx / (NBN / NWN);  // 0 or 1
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (NWSUBN / NTN);  // 0..3
    int threadRowInWarp = tidInWarp / (NWSUBN / NTN);  // 0..7

    int innerRowA = threadIdx.x / (NBK / 4);  // tid / 4, 0..31
    int innerColA = threadIdx.x % (NBK / 4);  // tid % 4, 0..3
    int innerRowB = threadIdx.x / (NBN / 4);  // tid / 8, 0..15
    int innerColB = threadIdx.x % (NBN / 4);  // tid % 8, 0..7

    float regM[NWMITER * NTM] = {0.0f};
    float regN[NWNITER * NTN] = {0.0f};

    // Phase B2: cp.async loads for narrow-N NN variant.
    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    // 2026-05-12 STAGE-4: persistent CTA loop for nn_narrow.
    // 2026-05-13 — Stage-4 persistent CTA unwrapped. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring pre-Stage-4 register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        const float* A_block = A + pid_m * NBM * lda;
        const float* B_block = B + pid_n * NBN;
        float* C_warp = C + (pid_m * NBM + warpRow * NWM) * ldc + pid_n * NBN + warpCol * NWN;

        // 2026-05-13 bias-bit-exact fix: pre-seed threadResults with bias[g_col]
        // BEFORE the K-loop so the FMA chain matches CPU's `y = bias; y += x*w`
        // order. Prior version initialized to 0 and added bias as a final
        // epilogue op (`val = alpha*sum + bias`), producing a different f32
        // rounding chain than CPU's `(((bias + x0*w0) + x1*w1) + ...)`.
        // Empirically: predictor head w2 forward (M=128 K=512 N=25 with bias)
        // drifted max_ulp=3007 (2752/3200 dirty); root cause of
        // `forward_parity_matrix::strict_per_tensor_bit_exact_after_step_1`
        // `critic.head4.block0.w2.weight` max_ulp=65541 cascade. Other forward
        // shapes (block w1/w2, input_proj, qh w1) had no bias (SimbaV2
        // HyperLinear is no-bias) so they were already bit-exact.
        // CPU ref: `crates/sqv_uaac/src/native/train/blas_batch.rs:337-346`.
        // Only valid for alpha=1 (all training forward calls use alpha=1).
        float threadResults[NWMITER * NTM * NWNITER * NTN];
        #pragma unroll
        for (int rm = 0; rm < NTM; ++rm) {
            int g_col_base = pid_n * NBN + warpCol * NWN + threadColInWarp * NTN;
            #pragma unroll
            for (int rn = 0; rn < NTN; ++rn) {
                int g_col = g_col_base + rn;
                int idx = rm * NTN + rn;
                threadResults[idx] =
                    (bias != nullptr && g_col < N) ? bias[g_col] : 0.0f;
            }
        }

    for (int bkIdx = 0; bkIdx < K; bkIdx += NBK) {
        // 2026-05-12 STAGE-3 narrow NN coalesce: 4 warps × 8 instr/warp at 50%
        // cache util (vs 12.5% legacy). NBM=64, NBK=16. M_ROWS_PER_WARP_INST=2.
        {
            constexpr int WARPS_NN_NARROW = NNUM_THREADS / WARPSIZE;            // 4
            constexpr int M_ROWS_PER_WARP_INST_NN_NR = WARPSIZE / NBK;          // 2
            constexpr int M_ROWS_PER_WARP_NN_NR = NBM / WARPS_NN_NARROW;        // 16
            constexpr int INSTR_PER_WARP_NN_NR =
                M_ROWS_PER_WARP_NN_NR / M_ROWS_PER_WARP_INST_NN_NR;              // 8
            static_assert(WARPSIZE % NBK == 0, "WARPSIZE divisible by NBK (narrow NN)");
            static_assert(NBM % WARPS_NN_NARROW == 0, "NBM divisible by warps (narrow NN)");
            int _warp = threadIdx.x / WARPSIZE;
            int _lane = threadIdx.x % WARPSIZE;
            int _m_in_warp_nn_nr = _lane / NBK;
            int _k_local_lane_nn_nr = _lane % NBK;
            #pragma unroll
            for (int _it = 0; _it < INSTR_PER_WARP_NN_NR; _it++) {
                int _m_local = _warp * M_ROWS_PER_WARP_NN_NR
                               + _it * M_ROWS_PER_WARP_INST_NN_NR + _m_in_warp_nn_nr;
                int _g_row = pid_m * NBM + _m_local;
                int _g_col = bkIdx + _k_local_lane_nn_nr;
                unsigned _dst = As_base
                    + (_k_local_lane_nn_nr * (NBM + SMEM_A_PAD) + _m_local)
                    * (unsigned)sizeof(float);
                if (_g_row < M && _g_col < K) {
                    const float* _src = A_block + _m_local * lda + _k_local_lane_nn_nr;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                                 :: "r"(_dst), "l"(_src));
                } else {
                    As[_k_local_lane_nn_nr * (NBM + SMEM_A_PAD) + _m_local] = 0.0f;
                }
            }
        }

        // Load B: cp.async.16B contiguous.
        for (int offset = 0; offset + NROW_STRIDE_B <= NBK; offset += NROW_STRIDE_B) {
            int g_row = bkIdx + innerRowB + offset;
            int g_col = pid_n * NBN + innerColB * 4;
            unsigned dst = Bs_base + ((innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
            if (g_row < K && g_col + 3 < N && (ldb % 4 == 0)) {
                const float* src = B_block + (innerRowB + offset) * ldb + innerColB * 4;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                Bs[(innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4 + 0] = (g_row < K && g_col + 0 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4 + 1] = (g_row < K && g_col + 1 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4 + 2] = (g_row < K && g_col + 2 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4 + 3] = (g_row < K && g_col + 3 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 3] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        // Compute.
        for (int dotIdx = 0; dotIdx < NBK; ++dotIdx) {
            for (int i = 0; i < NTM; ++i) {
                regM[i] = As[dotIdx * (NBM + SMEM_A_PAD) + warpRow * NWM + threadRowInWarp * NTM + i];
            }
            for (int i = 0; i < NTN; ++i) {
                regN[i] = Bs[dotIdx * (NBN + SMEM_B_PAD) + warpCol * NWN + threadColInWarp * NTN + i];
            }
            // F-NT-FMA pin (2026-05-08): explicit __fmaf_rn for bit-exact
            // match with CPU `_mm256_fmadd_ps`. sgemm_bi_nn_narrow.
            for (int resIdxM = 0; resIdxM < NTM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < NTN; ++resIdxN) {
                    int idx = resIdxM * NTN + resIdxN;
                    threadResults[idx] = __fmaf_rn(
                        regM[resIdxM], regN[resIdxN], threadResults[idx]);
                }
            }
        }

        A_block += NBK;
        B_block += NBK * ldb;
        __syncthreads();
    }

    // Epilogue: write with alpha, beta. Bias is ABSORBED into threadResults
    // pre-K-loop init above to match CPU FMA-chain order (bias-bit-exact fix
    // 2026-05-13). DO NOT re-add bias here.
    // Scalar N-fallback for non-%4 N (e.g. N=25).
    for (int resIdxM = 0; resIdxM < NTM; ++resIdxM) {
        int g_row = pid_m * NBM + warpRow * NWM + threadRowInWarp * NTM + resIdxM;
        if (g_row >= M) continue;
        for (int resIdxN = 0; resIdxN < NTN; ++resIdxN) {
            int g_col = pid_n * NBN + warpCol * NWN + threadColInWarp * NTN + resIdxN;
            if (g_col >= N) continue;
            int idx = resIdxM * NTN + resIdxN;
            float val = alpha * threadResults[idx];
            if (beta != 0.0f) val += beta * C_warp[(threadRowInWarp * NTM + resIdxM) * ldc + threadColInWarp * NTN + resIdxN];
            C_warp[(threadRowInWarp * NTM + resIdxM) * ldc + threadColInWarp * NTN + resIdxN] = val;
        }
    }
    __syncthreads();
    } // end persistent CTA loop (nn_narrow)
}

#undef NBM
#undef NBN
#undef NBK
#undef NWM
#undef NWN
#undef NWMITER
#undef NWNITER
#undef NTM
#undef NTN
#undef NNUM_THREADS
#undef NWSUBM
#undef NWSUBN
#undef NROW_STRIDE_A
#undef NROW_STRIDE_B

// ============================================================================
// Narrow-N NN small-tile variant — for low-M shapes (batch ≤ 64).
// ============================================================================
// Bit-exact clone of sgemm_bi_nn_narrow with shrunken tile. Per-output FMA
// chain `bias + Σ A[m,k]·B[k,n]` ascending K is identical regardless of tile
// — same single-rounding __fmaf_rn order, same bias pre-seed at K=0, same
// scalar N-tail epilogue. Output is byte-identical to sgemm_bi_nn_narrow on
// any shape; the only difference is GPU CTA grid layout (smaller tile = more
// CTAs = more SMs busy).
//
// Target: TQC qhead w2 (M=64, K=512, N=25, batch ≤ 64). Current narrow_NN
// runs at grid_size=1 (1 CTA on 128-SM Ada, 0.21% SM throughput). With
// NSBM=16, NSBN=16 the grid becomes ceil(64/16) × ceil(25/16) = 4 × 2 = 8
// CTAs → 8× SM utilization. Expected ~2.5-3× speedup, matching cuBLAS f32.
//
// Tile: NSBM=16 NSBN=16 NSBK=16. 64 threads (2 warps).
// Per-warp: WM=8 WN=16 (1 warpRow × 2 warpCol per warp grid → 2 warps).
// Per-thread micro-tile: TM=2 TN=2 (32 threads/warp via (WSUBM/TM)·(WSUBN/TN)
// = (8/2)·(16/2) = 4·8 = 32 = WARPSIZE).
// Smem: As [16·16] + Bs [16·16] = 2 KiB/CTA.
#define NSBM 16
#define NSBN 16
#define NSBK 16
#define NSWM 8
#define NSWN 16
#define NSWMITER 1
#define NSWNITER 1
#define NSTM 2
#define NSTN 2
#define NSNUM_THREADS 64
#define NSWSUBM (NSWM / NSWMITER)   // 8
#define NSWSUBN (NSWN / NSWNITER)   // 16
#define NSROW_STRIDE_A ((NSNUM_THREADS * 4) / NSBK)  // 16
#define NSROW_STRIDE_B (NSNUM_THREADS / (NSBN / 4))  // 16

extern "C" __global__ __launch_bounds__(NSNUM_THREADS, 8)
void sgemm_bi_nn_narrow_small(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float alpha, float beta,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    int post_op
) {
    (void)post_op;
    __shared__ float As[NSBK * (NSBM + SMEM_A_PAD)];
    __shared__ float Bs[NSBK * (NSBN + SMEM_B_PAD)];

    int num_pid_m = (M + NSBM - 1) / NSBM;
    int num_pid_n = (N + NSBN - 1) / NSBN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (NSBN / NSWN);  // 0 or 1
    int warpRow = warpIdx / (NSBN / NSWN);  // 0
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (NSWSUBN / NSTN);  // 0..7
    int threadRowInWarp = tidInWarp / (NSWSUBN / NSTN);  // 0..3

    int innerRowA = threadIdx.x / (NSBK / 4);  // tid / 4, 0..15
    int innerColA = threadIdx.x % (NSBK / 4);  // tid % 4, 0..3
    int innerRowB = threadIdx.x / (NSBN / 4);  // tid / 4, 0..15
    int innerColB = threadIdx.x % (NSBN / 4);  // tid % 4, 0..3

    float regM[NSWMITER * NSTM] = {0.0f};
    float regN[NSWNITER * NSTN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    int tile_id = blockIdx.x;
    {
        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        const float* A_block = A + pid_m * NSBM * lda;
        const float* B_block = B + pid_n * NSBN;
        float* C_warp = C + (pid_m * NSBM + warpRow * NSWM) * ldc + pid_n * NSBN + warpCol * NSWN;

        // Bias pre-seed (matches sgemm_bi_nn_narrow line 2935-2946 verbatim
        // semantic): seed threadResults with bias[g_col] BEFORE K-loop so FMA
        // chain begins with `(bias + A[m,0]*B[0,n])` to match CPU order. The
        // per-output FMA chain is identical to sgemm_bi_nn_narrow regardless
        // of tile size. Only valid for alpha=1 (training forward calls).
        float threadResults[NSWMITER * NSTM * NSWNITER * NSTN];
        #pragma unroll
        for (int rm = 0; rm < NSTM; ++rm) {
            int g_col_base = pid_n * NSBN + warpCol * NSWN + threadColInWarp * NSTN;
            #pragma unroll
            for (int rn = 0; rn < NSTN; ++rn) {
                int g_col = g_col_base + rn;
                int idx = rm * NSTN + rn;
                threadResults[idx] =
                    (bias != nullptr && g_col < N) ? bias[g_col] : 0.0f;
            }
        }

    for (int bkIdx = 0; bkIdx < K; bkIdx += NSBK) {
        // A coalesce mirror: warps load disjoint M-row slabs.
        // WARPS=2, M_ROWS_PER_WARP_INST=WARPSIZE/NSBK=2,
        // M_ROWS_PER_WARP=NSBM/WARPS=8, INSTR_PER_WARP=4.
        {
            constexpr int WARPS_NS_NR = NSNUM_THREADS / WARPSIZE;            // 2
            constexpr int M_ROWS_PER_WARP_INST_NS_NR = WARPSIZE / NSBK;       // 2
            constexpr int M_ROWS_PER_WARP_NS_NR = NSBM / WARPS_NS_NR;         // 8
            constexpr int INSTR_PER_WARP_NS_NR =
                M_ROWS_PER_WARP_NS_NR / M_ROWS_PER_WARP_INST_NS_NR;            // 4
            static_assert(WARPSIZE % NSBK == 0, "WARPSIZE divisible by NSBK (narrow small)");
            static_assert(NSBM % WARPS_NS_NR == 0, "NSBM divisible by warps (narrow small)");
            int _warp = threadIdx.x / WARPSIZE;
            int _lane = threadIdx.x % WARPSIZE;
            int _m_in_warp_ns_nr = _lane / NSBK;
            int _k_local_lane_ns_nr = _lane % NSBK;
            #pragma unroll
            for (int _it = 0; _it < INSTR_PER_WARP_NS_NR; _it++) {
                int _m_local = _warp * M_ROWS_PER_WARP_NS_NR
                               + _it * M_ROWS_PER_WARP_INST_NS_NR + _m_in_warp_ns_nr;
                int _g_row = pid_m * NSBM + _m_local;
                int _g_col = bkIdx + _k_local_lane_ns_nr;
                unsigned _dst = As_base
                    + (_k_local_lane_ns_nr * (NSBM + SMEM_A_PAD) + _m_local)
                    * (unsigned)sizeof(float);
                if (_g_row < M && _g_col < K) {
                    const float* _src = A_block + _m_local * lda + _k_local_lane_ns_nr;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                                 :: "r"(_dst), "l"(_src));
                } else {
                    As[_k_local_lane_ns_nr * (NSBM + SMEM_A_PAD) + _m_local] = 0.0f;
                }
            }
        }

        // Load B: cp.async.16B contiguous. NSROW_STRIDE_B=16 = NSBK → single iter.
        for (int offset = 0; offset + NSROW_STRIDE_B <= NSBK; offset += NSROW_STRIDE_B) {
            int g_row = bkIdx + innerRowB + offset;
            int g_col = pid_n * NSBN + innerColB * 4;
            unsigned dst = Bs_base + ((innerRowB + offset) * (NSBN + SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
            if (g_row < K && g_col + 3 < N && (ldb % 4 == 0)) {
                const float* src = B_block + (innerRowB + offset) * ldb + innerColB * 4;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                Bs[(innerRowB + offset) * (NSBN + SMEM_B_PAD) + innerColB * 4 + 0] = (g_row < K && g_col + 0 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (NSBN + SMEM_B_PAD) + innerColB * 4 + 1] = (g_row < K && g_col + 1 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (NSBN + SMEM_B_PAD) + innerColB * 4 + 2] = (g_row < K && g_col + 2 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (NSBN + SMEM_B_PAD) + innerColB * 4 + 3] = (g_row < K && g_col + 3 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 3] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        // Compute. Identical per-output ascending K __fmaf_rn chain as
        // sgemm_bi_nn_narrow — bit-exact f32 output regardless of tile size.
        for (int dotIdx = 0; dotIdx < NSBK; ++dotIdx) {
            for (int i = 0; i < NSTM; ++i) {
                regM[i] = As[dotIdx * (NSBM + SMEM_A_PAD) + warpRow * NSWM + threadRowInWarp * NSTM + i];
            }
            for (int i = 0; i < NSTN; ++i) {
                regN[i] = Bs[dotIdx * (NSBN + SMEM_B_PAD) + warpCol * NSWN + threadColInWarp * NSTN + i];
            }
            // F-NT-FMA pin: explicit __fmaf_rn for bit-exact match with CPU
            // `f32::mul_add` ascending K — same chain as sgemm_bi_nn_narrow.
            for (int resIdxM = 0; resIdxM < NSTM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < NSTN; ++resIdxN) {
                    int idx = resIdxM * NSTN + resIdxN;
                    threadResults[idx] = __fmaf_rn(
                        regM[resIdxM], regN[resIdxN], threadResults[idx]);
                }
            }
        }

        A_block += NSBK;
        B_block += NSBK * ldb;
        __syncthreads();
    }

    // Epilogue: bias-IN-FMA already absorbed via pre-K-loop seed. Scalar N
    // fallback for non-%4 N (e.g. N=25). Same write path as sgemm_bi_nn_narrow.
    for (int resIdxM = 0; resIdxM < NSTM; ++resIdxM) {
        int g_row = pid_m * NSBM + warpRow * NSWM + threadRowInWarp * NSTM + resIdxM;
        if (g_row >= M) continue;
        for (int resIdxN = 0; resIdxN < NSTN; ++resIdxN) {
            int g_col = pid_n * NSBN + warpCol * NSWN + threadColInWarp * NSTN + resIdxN;
            if (g_col >= N) continue;
            int idx = resIdxM * NSTN + resIdxN;
            float val = alpha * threadResults[idx];
            if (beta != 0.0f) val += beta * C_warp[(threadRowInWarp * NSTM + resIdxM) * ldc + threadColInWarp * NSTN + resIdxN];
            C_warp[(threadRowInWarp * NSTM + resIdxM) * ldc + threadColInWarp * NSTN + resIdxN] = val;
        }
    }
    __syncthreads();
    } // end persistent CTA loop (nn_narrow_small)
}

#undef NSBM
#undef NSBN
#undef NSBK
#undef NSWM
#undef NSWN
#undef NSWMITER
#undef NSWNITER
#undef NSTM
#undef NSTN
#undef NSNUM_THREADS
#undef NSWSUBM
#undef NSWSUBN
#undef NSROW_STRIDE_A
#undef NSROW_STRIDE_B

// ============================================================================
// Phase 2.1b — Narrow-N TN (backward dW, N∈9..48): C[K,N] += alpha * A^T[K,M] @ B[M,N]
// ============================================================================
// A = X_saved [M, K_out] — read transposed into As
// B = dY [M, N]
// C = dW [K_out, N] — accumulated (beta=1)
#define NBM 64
#define NBN 32
#define NBK 16
#define NWM 32
#define NWN 16
#define NTM 4
#define NTN 4
#define NNUM_THREADS 128
#define NWSUBN 16
#define NROW_STRIDE_A ((NNUM_THREADS * 4) / NBK)
#define NROW_STRIDE_B (NNUM_THREADS / (NBN / 4))

extern "C" __global__ __launch_bounds__(NNUM_THREADS, 4)
void sgemm_bi_tn_narrow(
    float* __restrict__ C,         // [K_out, N]
    const float* __restrict__ A,   // [M_red, K_out]
    const float* __restrict__ B,   // [M_red, N]
    float alpha,
    int M_red, int K_out, int N
) {
    __shared__ float As[NBK * (NBM + SMEM_A_PAD)];
    __shared__ float Bs[NBK * (NBN + SMEM_B_PAD)];

    int num_pid_m = (K_out + NBM - 1) / NBM;
    int num_pid_n = (N + NBN - 1) / NBN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (NBN / NWN);
    int warpRow = warpIdx / (NBN / NWN);
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (NWSUBN / NTN);
    int threadRowInWarp = tidInWarp / (NWSUBN / NTN);

    int innerRowA = threadIdx.x / (NBK / 4);  // tid/4 0..31
    int innerColA = threadIdx.x % (NBK / 4);  // tid%4 0..3
    int innerRowB = threadIdx.x / (NBN / 4);  // tid/8 0..15
    int innerColB = threadIdx.x % (NBN / 4);  // tid%8 0..7

    float regM[NTM] = {0.0f};
    float regN[NTN] = {0.0f};

    // 2026-05-12 STAGE-1 narrow TN coalesce: convert A-loader from direct
    // global reads to cp.async with warp-cooperative contiguous loads.
    unsigned As_base = __cvta_generic_to_shared(As);

    // 2026-05-12 STAGE-4: persistent CTA loop for tn_narrow.
    // 2026-05-13 — Stage-4 persistent CTA unwrapped. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring pre-Stage-4 register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[NTM * NTN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        float* C_warp = C + (pid_m * NBM + warpRow * NWM) * N + pid_n * NBN + warpCol * NWN;

    for (int mIdx = 0; mIdx < M_red; mIdx += NBK) {
        // 2026-05-12 STAGE-1 narrow TN A-loader: cp.async coalesced (16B/lane).
        // 4 warps × 16 lanes per row × 2 rows per inst × 2 inst/warp = 4 NBK rows × 4 = 16 NBK rows ✓
        // Source X[m_red, k_out..+3] contig in K_out. Dest As[m_red][k_out..+3] contig in inner.
        // As layout = [NBK outer × NBM inner], 100% cache line util.
        {
            constexpr int WARPS_TN_NR = NNUM_THREADS / WARPSIZE;        // 4
            constexpr int LANES_PER_ROW_TN_NR = NBM / 4;                // 16
            constexpr int ROWS_PER_INST_TN_NR = WARPSIZE / LANES_PER_ROW_TN_NR; // 2
            constexpr int ROWS_PER_WARP_TN_NR = NBK / WARPS_TN_NR;      // 4
            constexpr int INSTR_PER_WARP_TN_NR =
                ROWS_PER_WARP_TN_NR / ROWS_PER_INST_TN_NR;               // 2
            int _warp = threadIdx.x / WARPSIZE;
            int _lane = threadIdx.x % WARPSIZE;
            int _row_in_warp = _lane / LANES_PER_ROW_TN_NR;
            int _col_chunk = (_lane % LANES_PER_ROW_TN_NR) * 4;
            #pragma unroll
            for (int _it = 0; _it < INSTR_PER_WARP_TN_NR; _it++) {
                int _k_outer = _warp * ROWS_PER_WARP_TN_NR
                               + _it * ROWS_PER_INST_TN_NR + _row_in_warp;
                int _m_inner = _col_chunk;
                int _g_m = mIdx + _k_outer;
                int _g_k = pid_m * NBM + _m_inner;
                unsigned _dst = As_base
                    + (_k_outer * (NBM + SMEM_A_PAD) + _m_inner)
                    * (unsigned)sizeof(float);
                bool _full16 = (_g_m < M_red) && (_g_k + 3 < K_out) && ((K_out & 3) == 0);
                if (_full16) {
                    const float* _src = A + (long long)_g_m * K_out + _g_k;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                                 :: "r"(_dst), "l"(_src));
                } else {
                    #pragma unroll
                    for (int _i = 0; _i < 4; _i++) {
                        if ((_g_m < M_red) && (_g_k + _i < K_out)) {
                            const float* _src_e =
                                A + (long long)_g_m * K_out + _g_k + _i;
                            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                                         :: "r"(_dst + (unsigned)_i * 4),
                                            "l"(_src_e));
                        } else {
                            As[_k_outer * (NBM + SMEM_A_PAD) + _m_inner + _i] = 0.0f;
                        }
                    }
                }
            }
        }

        // Load B: dY[mIdx+m_local, pid_n*NBN + n_local]
        for (int offset = 0; offset + NROW_STRIDE_B <= NBK; offset += NROW_STRIDE_B) {
            int g_m = mIdx + innerRowB + offset;
            int g_n = pid_n * NBN + innerColB * 4;
            if (g_m < M_red && g_n + 3 < N && (N % 4 == 0)) {
                reinterpret_cast<float4*>(&Bs[(innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4])[0] =
                    ld_global_L2_128B(&B[g_m * N + pid_n * NBN + innerColB * 4]);
            } else {
                Bs[(innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4 + 0] = (g_m < M_red && g_n + 0 < N) ? B[g_m * N + pid_n * NBN + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4 + 1] = (g_m < M_red && g_n + 1 < N) ? B[g_m * N + pid_n * NBN + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4 + 2] = (g_m < M_red && g_n + 2 < N) ? B[g_m * N + pid_n * NBN + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4 + 3] = (g_m < M_red && g_n + 3 < N) ? B[g_m * N + pid_n * NBN + innerColB * 4 + 3] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        for (int dotIdx = 0; dotIdx < NBK; ++dotIdx) {
            for (int i = 0; i < NTM; ++i) {
                regM[i] = As[dotIdx * (NBM + SMEM_A_PAD) + warpRow * NWM + threadRowInWarp * NTM + i];
            }
            for (int i = 0; i < NTN; ++i) {
                regN[i] = Bs[dotIdx * (NBN + SMEM_B_PAD) + warpCol * NWN + threadColInWarp * NTN + i];
            }
            for (int rm = 0; rm < NTM; ++rm) {
                for (int rn = 0; rn < NTN; ++rn) {
                    // F-NT-FMA pin (2026-05-08): explicit __fmaf_rn for
                    // bit-exact match with CPU `_mm256_fmadd_ps`. nvcc may
                    // emit FMUL+FADD (two roundings) for `+= a*b` depending
                    // on compile flags; explicit FMA forces single-rounding.
                    threadResults[rm * NTN + rn] = __fmaf_rn(
                        regM[rm], regN[rn], threadResults[rm * NTN + rn]);
                }
            }
        }
        __syncthreads();
    }

    // Epilogue — accumulate (beta=1), scalar N-fallback
    for (int rm = 0; rm < NTM; ++rm) {
        int g_row = pid_m * NBM + warpRow * NWM + threadRowInWarp * NTM + rm;
        if (g_row >= K_out) continue;
        for (int rn = 0; rn < NTN; ++rn) {
            int g_col = pid_n * NBN + warpCol * NWN + threadColInWarp * NTN + rn;
            if (g_col >= N) continue;
            C_warp[(threadRowInWarp * NTM + rm) * N + threadColInWarp * NTN + rn] +=
                alpha * threadResults[rm * NTN + rn];
        }
    }
    __syncthreads();
    } // end persistent CTA loop (tn_narrow)
}

// ============================================================================
// Phase C-1.5bc — Narrow-N TN Split-M partial (backward dW for N<128 + large M).
//
// Mirrors sgemm_bi_tn_splitm_partial structure but with narrow-N tile params
// (NBM=64, NBN=32, NBK=16, NWM=32, NWN=16, NTM=4, NTN=4, NNUM_THREADS=128) so
// it covers TQC quantile head (N=25), C-loss head and any N ∈ [2..127].
//
// Why: at v6.5 multi-step + critic quantile head bwd_dw shape (M=batch*learn_len,
// K_out=hidden=256, N=25), plain sgemm_bi_tn_narrow grid = K_tiles*N_tiles =
// ceil(256/64) * ceil(25/32) = 4 * 1 = 4 blocks → ~3% SM utilization on Ada
// (142 SMs). Split-M inflates the grid F× along z by partitioning the M-reduction
// axis into F chunks, each handled by an independent block — same wave-fill
// strategy as the regular Split-M TN at sgemm_bi_tn_splitm_partial.
//
// Bit-exact contract (CLAUDE.md §5.2):
//   - F (split factor) is shape-keyed: caller computes F = div_ceil(2*NUM_SMS,
//     base_blocks) capped by scratch budget → identical F on every call with
//     same (M, K_out, N) → identical reduction tree.
//   - Per-chunk FMA chain inside this kernel uses __fmaf_rn (single-rounding,
//     same as plain sgemm_bi_tn_narrow at line 2643). Per-chunk partial bit-
//     exact identical to the narrow kernel's M-restricted output.
//   - Each (fc, k, n) slot has exactly ONE writer (block coords (k_tile,
//     n_tile, fc)) → no atomics, no race.
//   - OVERWRITE (no accumulate, no alpha, no bias): reducer applies alpha at
//     the final sum and += into the caller's dW slot. Same contract as
//     sgemm_bi_tn_splitm_partial (line 679-681).
//
// Caller uses existing sgemm_bi_splitm_reduce kernel (line 725) — its math is
// shape-agnostic on (K_out, N) and operates per output cell. Reducer applies
// alpha and accumulates into dW (+=) matching the narrow kernel's beta=1 semantic.
// ============================================================================
extern "C" __global__ __launch_bounds__(NNUM_THREADS, 4)
void sgemm_bi_tn_narrow_splitm_partial(
    float* __restrict__ partial,   // [F * K_out * N] — unique slot per (fc, ...)
    const float* __restrict__ A,   // X [M_red, K_out]
    const float* __restrict__ B,   // dY [M_red, N]
    int M_red, int K_out, int N,
    int M_CHUNK                    // chunk size (multiple of NBK)
) {
    __shared__ float As[NBK * (NBM + SMEM_A_PAD)];
    __shared__ float Bs[NBK * (NBN + SMEM_B_PAD)];

    int num_pid_m = (K_out + NBM - 1) / NBM;
    int num_pid_n = (N + NBN - 1) / NBN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;
    int fc = blockIdx.z;
    int m_begin = fc * M_CHUNK;
    int m_end = min(m_begin + M_CHUNK, M_red);
    if (m_begin >= M_red) return;

    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (NBN / NWN);
    int warpRow = warpIdx / (NBN / NWN);
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (NWSUBN / NTN);
    int threadRowInWarp = tidInWarp / (NWSUBN / NTN);

    int innerRowA = threadIdx.x / (NBK / 4);  // 0..31
    int innerColA = threadIdx.x % (NBK / 4);  // 0..3
    int innerRowB = threadIdx.x / (NBN / 4);  // 0..15
    int innerColB = threadIdx.x % (NBN / 4);  // 0..7

    float regM[NTM] = {0.0f};
    float regN[NTN] = {0.0f};

    // 2026-05-12 STAGE-4: persistent CTA loop for tn_narrow_splitm_partial.
    // 2026-05-13 — Stage-4 persistent CTA unwrapped. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring pre-Stage-4 register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[NTM * NTN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    // cp.async (CUTLASS SM80 pattern, mirrors sgemm_bi_tn_splitm_partial lines
    // 613-651). Single-stage with wait_all — adequate at NBK=16 small K-chunk.
    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    for (int mIdx = m_begin; mIdx < m_end; mIdx += NBK) {
        // 2026-05-12 STAGE-1 narrow TN splitm coalesce: 4 warps × 16 lanes per row
        // × 2 rows per warp inst at 100% cache line util (vs 12.5% legacy).
        // NBM=64 K_out cols, NBK=16 M_red rows. Source X[m_red, k_out..+3] contig.
        {
            constexpr int WARPS_TN_NSP = NNUM_THREADS / WARPSIZE;       // 4
            constexpr int LANES_PER_ROW_TN_NSP = NBM / 4;               // 16
            constexpr int ROWS_PER_INST_TN_NSP = WARPSIZE / LANES_PER_ROW_TN_NSP; // 2
            constexpr int ROWS_PER_WARP_TN_NSP = NBK / WARPS_TN_NSP;    // 4
            constexpr int INSTR_PER_WARP_TN_NSP =
                ROWS_PER_WARP_TN_NSP / ROWS_PER_INST_TN_NSP;             // 2
            static_assert(NBM % (WARPSIZE / (WARPSIZE / (NBM / 4))) == 0,
                "narrow TN splitm: lane/row config invariant");
            int _warp = threadIdx.x / WARPSIZE;
            int _lane = threadIdx.x % WARPSIZE;
            int _row_in_warp = _lane / LANES_PER_ROW_TN_NSP;
            int _col_chunk = (_lane % LANES_PER_ROW_TN_NSP) * 4;
            #pragma unroll
            for (int _it = 0; _it < INSTR_PER_WARP_TN_NSP; _it++) {
                int _k_local = _warp * ROWS_PER_WARP_TN_NSP
                               + _it * ROWS_PER_INST_TN_NSP + _row_in_warp;
                int _m_local = _col_chunk;
                int _g_m = mIdx + _k_local;
                int _g_k = pid_m * NBM + _m_local;
                unsigned _dst = As_base
                    + (_k_local * (NBM + SMEM_A_PAD) + _m_local)
                    * (unsigned)sizeof(float);
                bool _full16 = (_g_m < m_end) && (_g_k + 3 < K_out) && ((K_out & 3) == 0);
                if (_full16) {
                    const float* _src = A + (long long)_g_m * K_out + _g_k;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                                 :: "r"(_dst), "l"(_src));
                } else {
                    #pragma unroll
                    for (int _i = 0; _i < 4; _i++) {
                        if ((_g_m < m_end) && (_g_k + _i < K_out)) {
                            const float* _src_e =
                                A + (long long)_g_m * K_out + _g_k + _i;
                            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                                         :: "r"(_dst + (unsigned)_i * 4),
                                            "l"(_src_e));
                        } else {
                            As[_k_local * (NBM + SMEM_A_PAD) + _m_local + _i] = 0.0f;
                        }
                    }
                }
            }
        }
        // Load B tile (cp.async float4 when aligned, scalar fallback otherwise).
        for (int offset = 0; offset + NROW_STRIDE_B <= NBK; offset += NROW_STRIDE_B) {
            int g_m = mIdx + innerRowB + offset;
            int g_n = pid_n * NBN + innerColB * 4;
            unsigned dst = Bs_base + ((innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
            if (g_m < m_end && g_n + 3 < N && (N % 4 == 0)) {
                const float* src = B + ((long long)g_m) * N + pid_n * NBN + innerColB * 4;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                Bs[(innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4 + 0] = (g_m < m_end && g_n + 0 < N) ? B[g_m * N + pid_n * NBN + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4 + 1] = (g_m < m_end && g_n + 1 < N) ? B[g_m * N + pid_n * NBN + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4 + 2] = (g_m < m_end && g_n + 2 < N) ? B[g_m * N + pid_n * NBN + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (NBN + SMEM_B_PAD) + innerColB * 4 + 3] = (g_m < m_end && g_n + 3 < N) ? B[g_m * N + pid_n * NBN + innerColB * 4 + 3] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        for (int dotIdx = 0; dotIdx < NBK; ++dotIdx) {
            for (int i = 0; i < NTM; ++i) {
                regM[i] = As[dotIdx * (NBM + SMEM_A_PAD) + warpRow * NWM + threadRowInWarp * NTM + i];
            }
            for (int i = 0; i < NTN; ++i) {
                regN[i] = Bs[dotIdx * (NBN + SMEM_B_PAD) + warpCol * NWN + threadColInWarp * NTN + i];
            }
            // F-NT-FMA pin: identical to sgemm_bi_tn_narrow line 2643. Per-
            // chunk FMA order matches the narrow kernel exactly → partial[fc]
            // is bit-exact with what narrow would compute on its M-restricted
            // slice. Reducer ascending-fc sum is the only new rounding step,
            // and it's identical for every (M_red, K_out, N) shape.
            for (int rm = 0; rm < NTM; ++rm) {
                for (int rn = 0; rn < NTN; ++rn) {
                    threadResults[rm * NTN + rn] = __fmaf_rn(
                        regM[rm], regN[rn], threadResults[rm * NTN + rn]);
                }
            }
        }
        __syncthreads();
    }

    // Epilogue: OVERWRITE partial[fc, :, :] (no alpha, no bias, no accumulate).
    // Float4 STG.128 fast path when N % 4 == 0 AND 4 lanes stay within N tile
    // — matches Big splitm_partial epilogue (line 695+). At N=25 (TQC quantile
    // head, the primary target shape) N%4 != 0 → always scalar tail.
    float* partial_chunk = partial + (long long)fc * K_out * N;
    float* partial_warp = partial_chunk + (pid_m * NBM + warpRow * NWM) * N + pid_n * NBN + warpCol * NWN;
    const bool n_aligned = (N & 3) == 0;

    for (int rm = 0; rm < NTM; ++rm) {
        int g_row = pid_m * NBM + warpRow * NWM + threadRowInWarp * NTM + rm;
        if (g_row >= K_out) continue;
        int g_col_base = pid_n * NBN + warpCol * NWN + threadColInWarp * NTN;
        // NTN == 4: try float4 store.
        if (n_aligned && g_col_base + 3 < N) {
            float4 out;
            out.x = threadResults[rm * NTN + 0];
            out.y = threadResults[rm * NTN + 1];
            out.z = threadResults[rm * NTN + 2];
            out.w = threadResults[rm * NTN + 3];
            reinterpret_cast<float4*>(
                &partial_warp[(threadRowInWarp * NTM + rm) * N + threadColInWarp * NTN])[0] = out;
        } else {
            #pragma unroll
            for (int rn = 0; rn < NTN; ++rn) {
                int g_col = g_col_base + rn;
                if (g_col >= N) continue;
                partial_warp[(threadRowInWarp * NTM + rm) * N + threadColInWarp * NTN + rn] =
                    threadResults[rm * NTN + rn];
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (tn_narrow_splitm_partial)
}

// ============================================================================
// Phase 2.1c — Narrow-N NT (backward dX, N∈9..48): C[M,K_out] = alpha * A[M,N] @ B^T[N,K_out]
// ============================================================================
// A = dY [M, N]
// B = W [K_out, N] — read transposed as W^T[N, K_out]
// C = dX [M, K_out] — overwrite (beta=0)
extern "C" __global__ __launch_bounds__(NNUM_THREADS, 4)
void sgemm_bi_nt_narrow(
    float* __restrict__ C,
    const float* __restrict__ A,   // dY [M, N]
    const float* __restrict__ B,   // W [K_out, N]
    float alpha,
    int M, int N, int K_out
) {
    __shared__ float As[NBK * (NBM + SMEM_A_PAD)];
    __shared__ float Bs[NBK * (NBN + SMEM_B_PAD)];

    // Grid: over (M, K_out) — "BM" = M tile, "BN" = K_out tile
    int num_pid_m = (M + NBM - 1) / NBM;
    int num_pid_n = (K_out + NBN - 1) / NBN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (NBN / NWN);
    int warpRow = warpIdx / (NBN / NWN);
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (NWSUBN / NTN);
    int threadRowInWarp = tidInWarp / (NWSUBN / NTN);

    int innerRowA = threadIdx.x / (NBK / 4);
    int innerColA = threadIdx.x % (NBK / 4);

    float regM[NTM] = {0.0f};
    float regN[NTN] = {0.0f};

    // 2026-05-12 STAGE-4: persistent CTA loop for nt_narrow.
    // 2026-05-13 — Stage-4 persistent CTA unwrapped. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring pre-Stage-4 register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[NTM * NTN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    for (int nIdx = 0; nIdx < N; nIdx += NBK) {
        // 2026-05-12 STAGE-2 narrow NT coalesce: 4 warps × 8 instr/warp × 2 rows.
        // M_ROWS_PER_WARP_INST = WARPSIZE/NBK = 2. Cache util 50% (vs 12.5% legacy).
        // Each lane: 1 float dY[g_m, g_n] → As[n_local][m_local].
        {
            unsigned As_base_nt_nr = __cvta_generic_to_shared(As);
            constexpr int WARPS_NT_NR = NNUM_THREADS / WARPSIZE;        // 4
            constexpr int M_ROWS_PER_WARP_INST_NT_NR = WARPSIZE / NBK;  // 2
            constexpr int M_ROWS_PER_WARP_NT_NR = NBM / WARPS_NT_NR;    // 16
            constexpr int INSTR_PER_WARP_NT_NR =
                M_ROWS_PER_WARP_NT_NR / M_ROWS_PER_WARP_INST_NT_NR;      // 8
            int _warp = threadIdx.x / WARPSIZE;
            int _lane = threadIdx.x % WARPSIZE;
            int _m_in_warp_nt_nr = _lane / NBK;
            int _n_local_lane_nt_nr = _lane % NBK;
            #pragma unroll
            for (int _it = 0; _it < INSTR_PER_WARP_NT_NR; _it++) {
                int _m_local = _warp * M_ROWS_PER_WARP_NT_NR
                               + _it * M_ROWS_PER_WARP_INST_NT_NR + _m_in_warp_nt_nr;
                int _g_m = pid_m * NBM + _m_local;
                int _g_n = nIdx + _n_local_lane_nt_nr;
                unsigned _dst = As_base_nt_nr
                    + (_n_local_lane_nt_nr * (NBM + SMEM_A_PAD) + _m_local)
                    * (unsigned)sizeof(float);
                if (_g_m < M && _g_n < N) {
                    const float* _src = A + (long long)_g_m * N + _g_n;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                                 :: "r"(_dst), "l"(_src));
                } else {
                    As[_n_local_lane_nt_nr * (NBM + SMEM_A_PAD) + _m_local] = 0.0f;
                }
            }
        }

        // Load B^T: W[pid_n*NBN + k_local, nIdx + n_local] → Bs[n_local * (NBN+PAD) + k_local]
        // Each thread loads 4 B values across N, for a single k = innerRow
        for (int offset = 0; offset + NROW_STRIDE_A <= NBN; offset += NROW_STRIDE_A) {
            int k_base = innerRowA + offset;
            int n_local = innerColA * 4;
            int g_k = pid_n * NBN + k_base;
            int g_n = nIdx + n_local;
            float4 tmp;
            if (g_k < K_out && g_n + 3 < N && (N % 4 == 0)) {
                tmp = ld_global_L2_128B(&B[g_k * N + g_n]);
            } else {
                tmp.x = (g_k < K_out && g_n + 0 < N) ? B[g_k * N + g_n + 0] : 0.0f;
                tmp.y = (g_k < K_out && g_n + 1 < N) ? B[g_k * N + g_n + 1] : 0.0f;
                tmp.z = (g_k < K_out && g_n + 2 < N) ? B[g_k * N + g_n + 2] : 0.0f;
                tmp.w = (g_k < K_out && g_n + 3 < N) ? B[g_k * N + g_n + 3] : 0.0f;
            }
            Bs[(n_local + 0) * (NBN + SMEM_B_PAD) + k_base] = tmp.x;
            Bs[(n_local + 1) * (NBN + SMEM_B_PAD) + k_base] = tmp.y;
            Bs[(n_local + 2) * (NBN + SMEM_B_PAD) + k_base] = tmp.z;
            Bs[(n_local + 3) * (NBN + SMEM_B_PAD) + k_base] = tmp.w;
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        for (int dotIdx = 0; dotIdx < NBK; ++dotIdx) {
            for (int i = 0; i < NTM; ++i) {
                regM[i] = As[dotIdx * (NBM + SMEM_A_PAD) + warpRow * NWM + threadRowInWarp * NTM + i];
            }
            for (int i = 0; i < NTN; ++i) {
                regN[i] = Bs[dotIdx * (NBN + SMEM_B_PAD) + warpCol * NWN + threadColInWarp * NTN + i];
            }
            for (int rm = 0; rm < NTM; ++rm) {
                for (int rn = 0; rn < NTN; ++rn) {
                    // F-NT-FMA pin (2026-05-08): explicit __fmaf_rn for
                    // bit-exact match with CPU `_mm256_fmadd_ps`. nvcc may
                    // emit FMUL+FADD (two roundings) for `+= a*b` depending
                    // on compile flags; explicit FMA forces single-rounding.
                    threadResults[rm * NTN + rn] = __fmaf_rn(
                        regM[rm], regN[rn], threadResults[rm * NTN + rn]);
                }
            }
        }
        __syncthreads();
    }

    float* C_warp = C + (pid_m * NBM + warpRow * NWM) * K_out + pid_n * NBN + warpCol * NWN;

    // Epilogue — overwrite (beta=0), scalar K_out-fallback
    // Phase 2.9: __stwt streaming store (write-once output, evict-first safe).
    for (int rm = 0; rm < NTM; ++rm) {
        int g_row = pid_m * NBM + warpRow * NWM + threadRowInWarp * NTM + rm;
        if (g_row >= M) continue;
        for (int rn = 0; rn < NTN; ++rn) {
            int g_col = pid_n * NBN + warpCol * NWN + threadColInWarp * NTN + rn;
            if (g_col >= K_out) continue;
            __stwt(&C_warp[(threadRowInWarp * NTM + rm) * K_out + threadColInWarp * NTN + rn],
                   alpha * threadResults[rm * NTN + rn]);
        }
    }
    __syncthreads();
    } // end persistent CTA loop (nt_narrow)
}

#undef NBM
#undef NBN
#undef NBK
#undef NWM
#undef NWN
#undef NTM
#undef NTN
#undef NNUM_THREADS
#undef NWSUBN
#undef NROW_STRIDE_A
#undef NROW_STRIDE_B

// ============================================================================
// Split-K Thin-M variant — deterministic f32 via fixed tree reduce.
// ============================================================================
// Designed for small-M shapes (M ∈ [32, 127]) where full K-reduction inside a
// single block leaves the grid underfilled (≤8 output tiles vs 142 SMs on Ada).
// Split K into chunks of size 32, run partial GEMM per (m,n,k_chunk) block,
// then a separate reduce kernel does fixed-order tree sum across chunks.
//
// Tile: SBM=32 SBN=64 SBK=32 (one K-iter per block), WM=16 WN=32 (2×2 warps),
// TM=4 TN=4, WMITER=WNITER=1 → per-thread 16 accum, ~24 regs total.
// __launch_bounds__(128, 4) — 4 blocks/SM × 32-block grid for M=64 N=256 K=128
// fills 32/142 SMs in the first wave with minimal per-block work.
//
// Determinism: each block writes its partial to a unique slot in the
// scratch buffer `partial[K_CHUNKS * M * N]`. The reduce kernel sums in
// ascending chunk order (((p0+p1)+p2)+p3+...), identical on every run.
// No atomicAdd. Batch-invariant by construction.
// ============================================================================

#define SBM 32
#define SBN 64
#define SBK 32
#define SWM 16
#define SWN 32
#define STM 4
#define STN 4
#define SNUM_THREADS 128

// K-bound contract.
// `K_CHUNKS = K_main / SBK` is the count of FULL chunks the kernel processes;
// it covers columns [0..K_main). Anything past K_main (the tail) is handled
// EXTERNALLY by `sgemm_bi_splitk_reduce` via `x_tail_ptr` / `w_tail_ptr` /
// `tail_cnt`. There is NO in-kernel K-bound runtime check here — the
// dispatcher contract guarantees `K_main ≤ K_full` and the kernel reads
// strictly from [0..K_main). If a future dispatcher change ever passes
// `K_CHUNKS · SBK > lda`, the kernel will OOB-read silently. Keep the
// dispatcher (`blas_gpu.rs::gpu_sgemm_forward` Phase 4) honest.
extern "C" __global__ __launch_bounds__(SNUM_THREADS, 4)
void sgemm_bi_nn_splitk32_partial(
    float* __restrict__ partial,  // [K_CHUNKS * M * N]
    const float* __restrict__ A,  // [M, K_full]
    const float* __restrict__ B,  // [K_full, N]
    int M, int N,
    int K_CHUNKS,                  // = K_main / SBK  (K_main = K_CHUNKS * SBK, covers columns [0..K_main))
    int lda                         // actual A row stride in floats (= K_full; can differ from K_main when a tail lives at K ≥ K_main)
) {
    __shared__ float As[SBK * (SBM + SMEM_A_PAD)];
    __shared__ float Bs[SBK * (SBN + SMEM_B_PAD)];

    int num_pid_m = (M + SBM - 1) / SBM;
    int num_pid_n = (N + SBN - 1) / SBN;
    int total_mn = num_pid_m * num_pid_n;
    int total_blocks = K_CHUNKS * total_mn;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;

    int warpIdx = threadIdx.x / WARPSIZE;
    int warpCol = warpIdx % (SBN / SWN);
    int warpRow = warpIdx / (SBN / SWN);
    int tidInWarp = threadIdx.x % WARPSIZE;
    int threadColInWarp = tidInWarp % (SWN / STN);
    int threadRowInWarp = tidInWarp / (SWN / STN);

    int innerRowA = threadIdx.x / (SBK / 4);
    int innerColA = threadIdx.x % (SBK / 4);
    int innerRowB = threadIdx.x / (SBN / 4);
    int innerColB = threadIdx.x % (SBN / 4);

    // Register double-buffer: regM/regN[buf][i] — while FMA consumes buf=0,
    // the next iter's smem→reg load fills buf=1 in parallel, hiding smem
    // latency (~20-30 cyc vs ~8 cyc for 16 FMAs). Bit-exact: same FMA order,
    // same accumulator sequence. +8 regs total (was 8 → 16), fits in ~40 regs
    // per thread (ceiling 255).
    float regM[2][STM] = {{0.0f}};
    float regN[2][STN] = {{0.0f}};

    // 2026-05-12 STAGE-4: persistent CTA loop for splitk32_partial.
    // Grid is K_CHUNKS × total_mn (joint blockIdx.x). Persistent loop iterates
    // tile_id over BOTH K-chunk and (pid_m, pid_n) — pid_k / pid_mn / pid_m / pid_n
    // all derive from tile_id per iteration. Each (pid_k, pid_m, pid_n) tile is
    // independent — partial slot is unique → no race.
    // 2026-05-13 — Stage-4 persistent CTA unwrapped (see sgemm_bi_nn). splitk32
    // case uses total_blocks = K_CHUNKS * total_mn since it iterates across
    // K-chunks too. Bit-exact: each (pid_k, pid_m, pid_n) tile is still
    // independent and gets its own CTA via gridDim = K_CHUNKS * total_mn.
    int tile_id = blockIdx.x;
    {
        float threadResults[STM * STN] = {0.0f};

        int pid_k = tile_id / total_mn;
        int pid_mn = tile_id % total_mn;
        int group_id = pid_mn / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m);
        int pid_n = (pid_mn % num_pid_in_group) / group_size_m;

        int k_offset = pid_k * SBK;
        const float* A_block = A + pid_m * SBM * lda + k_offset;
        const float* B_block = B + k_offset * N + pid_n * SBN;

    // Phase B2: cp.async loads for Split-K thin-M partial kernel.
    // A: 4× cp.async.4B scattered transpose dest (same pattern as NN).
    // B: cp.async.16B contiguous (src+dst). Scalar fallback for non-%4 lda / OOB.
    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    for (int offset = 0; offset < SBM; offset += 16) {
        int g_row = pid_m * SBM + innerRowA + offset;
        bool pr = g_row < M;
        const float* src_base = A_block + (innerRowA + offset) * lda + innerColA * 4;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned dst = As_base + ((innerColA * 4 + i) * (SBM + SMEM_A_PAD) + innerRowA + offset) * (unsigned)sizeof(float);
            if (pr) {
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(dst), "l"(src_base + i));
            } else {
                As[(innerColA * 4 + i) * (SBM + SMEM_A_PAD) + innerRowA + offset] = 0.0f;
            }
        }
    }

    for (int offset = 0; offset < SBK; offset += 8) {
        int g_col = pid_n * SBN + innerColB * 4;
        unsigned dst = Bs_base + ((innerRowB + offset) * (SBN + SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
        if (g_col + 3 < N) {
            const float* src = B_block + (innerRowB + offset) * N + innerColB * 4;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(src));
        } else {
            Bs[(innerRowB + offset) * (SBN + SMEM_B_PAD) + innerColB * 4 + 0] = (g_col + 0 < N) ? B_block[(innerRowB + offset) * N + innerColB * 4 + 0] : 0.0f;
            Bs[(innerRowB + offset) * (SBN + SMEM_B_PAD) + innerColB * 4 + 1] = (g_col + 1 < N) ? B_block[(innerRowB + offset) * N + innerColB * 4 + 1] : 0.0f;
            Bs[(innerRowB + offset) * (SBN + SMEM_B_PAD) + innerColB * 4 + 2] = (g_col + 2 < N) ? B_block[(innerRowB + offset) * N + innerColB * 4 + 2] : 0.0f;
            Bs[(innerRowB + offset) * (SBN + SMEM_B_PAD) + innerColB * 4 + 3] = (g_col + 3 < N) ? B_block[(innerRowB + offset) * N + innerColB * 4 + 3] : 0.0f;
        }
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    // Register double-buffer: prefetch dotIdx=0 into buf=0.
    #pragma unroll
    for (int i = 0; i < STM; ++i)
        regM[0][i] = As[0 * (SBM + SMEM_A_PAD) + warpRow * SWM + threadRowInWarp * STM + i];
    #pragma unroll
    for (int i = 0; i < STN; ++i)
        regN[0][i] = Bs[0 * (SBN + SMEM_B_PAD) + warpCol * SWN + threadColInWarp * STN + i];
    for (int dotIdx = 0; dotIdx < SBK; ++dotIdx) {
        int cur = dotIdx & 1;
        int nxt = cur ^ 1;
        // Prefetch next iteration's fragments (skip on last iter).
        if (dotIdx + 1 < SBK) {
            int next_k = dotIdx + 1;
            #pragma unroll
            for (int i = 0; i < STM; ++i)
                regM[nxt][i] = As[next_k * (SBM + SMEM_A_PAD) + warpRow * SWM + threadRowInWarp * STM + i];
            #pragma unroll
            for (int i = 0; i < STN; ++i)
                regN[nxt][i] = Bs[next_k * (SBN + SMEM_B_PAD) + warpCol * SWN + threadColInWarp * STN + i];
        }
        // FMA consumes current buffer — bit-exact: same rm-major, rn-minor order.
        // F-NT-FMA pin (2026-05-08): explicit __fmaf_rn for bit-exact match
        // with CPU `_mm256_fmadd_ps`. sgemm_bi_nn_splitk32_partial.
        #pragma unroll
        for (int rm = 0; rm < STM; ++rm) {
            #pragma unroll
            for (int rn = 0; rn < STN; ++rn) {
                int idx = rm * STN + rn;
                threadResults[idx] = __fmaf_rn(
                    regM[cur][rm], regN[cur][rn], threadResults[idx]);
            }
        }
    }

    float* partial_base = partial + (long long)pid_k * M * N;
    for (int rm = 0; rm < STM; ++rm) {
        int g_row = pid_m * SBM + warpRow * SWM + threadRowInWarp * STM + rm;
        if (g_row >= M) continue;
        int g_col_base = pid_n * SBN + warpCol * SWN + threadColInWarp * STN;
        if (g_col_base + 3 < N) {
            float4 out = {
                threadResults[rm * STN + 0],
                threadResults[rm * STN + 1],
                threadResults[rm * STN + 2],
                threadResults[rm * STN + 3]
            };
            reinterpret_cast<float4*>(&partial_base[g_row * N + g_col_base])[0] = out;
        } else {
            for (int j = 0; j < STN && g_col_base + j < N; j++) {
                partial_base[g_row * N + g_col_base + j] = threadResults[rm * STN + j];
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (nn_splitk32_partial)
}

// Deterministic tree-reduce. Fixed sum order across K_CHUNKS.
// Supports alpha scale + optional bias + optional K-tail fold (tail_cnt columns
// in [1..31]) + optional output column stride.
//
// Tail fold layout:
//   x_tail_ptr points at X[:, k_main]        — row stride x_tail_stride (= K_full)
//   w_tail_ptr points at W[k_main, :]        — row stride N (contiguous, row-major)
//   For each k in [0, tail_cnt): sum += X[m, k_main+k] * W[k_main+k, n]
//
// Extra args (pass nullptr/0 for no tail fold):
//   tail_cnt           — number of tail columns (0..31), 0 = skip
//   out_col_stride     — if > 0, output row stride = out_col_stride (default = N)
extern "C" __global__ __launch_bounds__(256, 8)
void sgemm_bi_splitk_reduce(
    float* __restrict__ C,
    const float* __restrict__ partial,
    const float* __restrict__ bias,
    const float* __restrict__ x_tail_ptr,
    const float* __restrict__ w_tail_ptr,
    float alpha,
    int M, int N,
    int K_CHUNKS,
    int x_tail_stride,
    int out_col_stride,
    int tail_cnt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int m = idx / N;
    int n = idx % N;
    long long mn_stride = (long long)M * N;

    float sum = partial[idx];
    for (int kc = 1; kc < K_CHUNKS; ++kc) {
        sum += partial[(long long)kc * mn_stride + idx];
    }
    // K-tail fold: sum += Σ_{k=0..tail_cnt-1} X[m, k_main+k] * W[k_main+k, n].
    // Fixed sum order (ascending k) across threads → deterministic.
    // F-NT-FMA pin: explicit __fmaf_rn for bit-exact match with CPU FMA chain.
    if (tail_cnt > 0 && x_tail_ptr != nullptr && w_tail_ptr != nullptr) {
        const float* x_row = x_tail_ptr + (long long)m * x_tail_stride;
        #pragma unroll 4
        for (int k = 0; k < tail_cnt; ++k) {
            sum = __fmaf_rn(x_row[k], w_tail_ptr[(long long)k * N + n], sum);
        }
    }
    // Keep 2-rounding `(α·Σ) + bias` form here. Do NOT
    // fuse to `__fmaf_rn(α, Σ, bias)` even though IEEE 754-2008 §5.4.1 says
    // fused is 1 ULP more accurate. The project's other 4 NN kernels (Big NN,
    // Slim NN, ultra_thin, narrow NN) fold bias differently; fusing here would
    // break cross-kernel bit-exactness in the multi-path dispatcher.
    sum *= alpha;
    if (bias != nullptr) sum += bias[n];

    int write_stride = (out_col_stride > 0) ? out_col_stride : N;
    C[(long long)m * write_stride + n] = sum;
}

// GEMV-style fill for backward_dx K=1 tail column: dX[m, col_idx] = Σ dY[m,n] · W_row[n].
// Used alongside Split-K main (via transpose) to close K_out%4 != 0 shapes
// (e.g. SALE action K_out=257 → main 256 via Split-K NT-via-T + 1 tail here).
// One block per M row-group; each thread handles one m, sequential N reduction.
extern "C" __global__ __launch_bounds__(256, 8)
void sgemm_bi_dx_col_gemv(
    float* __restrict__ dX,           // [M, out_col_stride]
    const float* __restrict__ dY,     // [M, N]
    const float* __restrict__ w_row,  // [N] — W[K_tail_row, :]
    int M, int N,
    int col_idx,                      // column in dX to fill
    int out_col_stride                // dX row stride
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    float sum = 0.0f;
    const float* dy_row = dY + (long long)m * N;
    // F-NT-FMA pin: explicit __fmaf_rn for bit-exact match with CPU FMA.
    for (int n = 0; n < N; ++n) {
        sum = __fmaf_rn(dy_row[n], w_row[n], sum);
    }
    dX[(long long)m * out_col_stride + col_idx] = sum;
}

#undef SBM
#undef SBN
#undef SBK
#undef SWM
#undef SWN
#undef STM
#undef STN
#undef SNUM_THREADS

// ============================================================================
// Transpose [rows, cols] → [cols, rows], f32. Used by backward_dx Split-K path:
// dX[M, K_out] = dY[M, N] @ W^T[N, K_out]  becomes dX = dY @ W_T where
// W_T = transpose(W). Reuses the existing NN Split-K kernel instead of a
// dedicated NT variant (which would require scalar stride-N W-gather, slow).
// 32×32 smem tile with +1 column pad eliminates bank conflicts.
// Source: NVIDIA CUDA C++ Programming Guide §8.7.2 "Matrix Transpose".
// ============================================================================
extern "C" __global__ __launch_bounds__(1024, 2)
void sgemm_transpose_f32_2d(
    float* __restrict__ dst,        // [cols, rows]
    const float* __restrict__ src,  // [rows, cols]
    int rows, int cols
) {
    __shared__ float tile[32][33];  // +1 pad for bank-conflict-free transpose
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_row = blockIdx.y * 32;
    int block_col = blockIdx.x * 32;

    int r = block_row + ty;
    int c = block_col + tx;
    if (r < rows && c < cols) {
        tile[ty][tx] = src[r * cols + c];
    } else {
        tile[ty][tx] = 0.0f;
    }
    __syncthreads();

    int out_row = block_col + ty;  // src col becomes dst row
    int out_col = block_row + tx;  // src row becomes dst col
    if (out_row < cols && out_col < rows) {
        dst[out_row * rows + out_col] = tile[tx][ty];
    }
}

// ============================================================================
// Typed (bf16/f16) variants — Phase 11 stage 2: sync-load buckets.
// ============================================================================
// Contract (per the typed-triad design research):
//   - X / W / Y / dY / dX are T_ACT (typed I/O); loads upcast via to_f at the
//     read site, EXACTLY one RNE downcast (FROM_F) at the final store.
//   - dW and bias stay f32 (master gradients / f32 bias) — never rounded.
//   - All accumulation and the epilogue (alpha*acc + bias + beta*C) stay f32
//     with the same ascending-K __fmaf_rn chains and fixed reduce trees as
//     the f32 kernels: a typed kernel is bit-identical to "upcast inputs to
//     f32, run the f32 kernel" (bf16/f16 products are exact in f32).
//   - to_f / from_f_* come from _typed_prelude.cuh (inlined first in the
//     NVRTC blob; conversions are RNE, no FTZ — see kernels.rs flags).

#define DEFINE_SGEMM_BI_NN_GEMV(SUFFIX, T_ACT, FROM_F)                        \
extern "C" __global__ __launch_bounds__(128, 4)                               \
void sgemm_bi_nn_gemv_##SUFFIX(                                               \
    T_ACT* __restrict__ Y,                                                    \
    const T_ACT* __restrict__ X,                                              \
    const T_ACT* __restrict__ W,                                              \
    const float* __restrict__ bias,                                           \
    float alpha, float beta,                                                  \
    int M, int K,                                                             \
    int lda, int ldy                                                          \
) {                                                                           \
    const int tid = threadIdx.x;                                              \
    const int warp = tid >> 5;                                                \
    const int lane = tid & 31;                                                \
    const int row = blockIdx.x * 4 + warp;                                    \
    if (row >= M) return;                                                     \
    float acc = 0.0f;                                                         \
    const T_ACT* X_row = X + row * lda;                                       \
    for (int k = lane; k < K; k += 32) {                                      \
        acc = __fmaf_rn(to_f(X_row[k]), to_f(W[k]), acc);                     \
    }                                                                         \
    acc += __shfl_xor_sync(0xffffffff, acc, 16);                              \
    acc += __shfl_xor_sync(0xffffffff, acc, 8);                               \
    acc += __shfl_xor_sync(0xffffffff, acc, 4);                               \
    acc += __shfl_xor_sync(0xffffffff, acc, 2);                               \
    acc += __shfl_xor_sync(0xffffffff, acc, 1);                               \
    if (lane == 0) {                                                          \
        float val = alpha * acc;                                              \
        if (bias != nullptr) val += bias[0];                                  \
        if (beta != 0.0f) val += beta * to_f(Y[row * ldy]);                   \
        Y[row * ldy] = FROM_F(val);                                           \
    }                                                                         \
}

DEFINE_SGEMM_BI_NN_GEMV(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SGEMM_BI_NN_GEMV(f16,  __half,        from_f_f16)

// TN GEMV: dW[K] += alpha * X^T[K,M] @ dY[M]. dW stays f32 (master grad).
#define DEFINE_SGEMM_BI_TN_GEMV(SUFFIX, T_ACT, FROM_F)                        \
extern "C" __global__ __launch_bounds__(128, 4)                               \
void sgemm_bi_tn_gemv_##SUFFIX(                                               \
    float* __restrict__ dW,                                                   \
    const T_ACT* __restrict__ X,                                              \
    const T_ACT* __restrict__ dY,                                             \
    float alpha,                                                              \
    int M_red, int K_out,                                                     \
    int lda, int ldy                                                          \
) {                                                                           \
    const int tid = threadIdx.x;                                              \
    const int warp = tid >> 5;                                                \
    const int lane = tid & 31;                                                \
    const int k = blockIdx.x * 4 + warp;                                      \
    if (k >= K_out) return;                                                   \
    float acc = 0.0f;                                                         \
    for (int m = lane; m < M_red; m += 32) {                                  \
        acc = __fmaf_rn(to_f(X[m * lda + k]), to_f(dY[m * ldy]), acc);        \
    }                                                                         \
    acc += __shfl_xor_sync(0xffffffff, acc, 16);                              \
    acc += __shfl_xor_sync(0xffffffff, acc, 8);                               \
    acc += __shfl_xor_sync(0xffffffff, acc, 4);                               \
    acc += __shfl_xor_sync(0xffffffff, acc, 2);                               \
    acc += __shfl_xor_sync(0xffffffff, acc, 1);                               \
    if (lane == 0) {                                                          \
        dW[k] += alpha * acc;                                                 \
    }                                                                         \
    (void)FROM_F;                                                             \
}

DEFINE_SGEMM_BI_TN_GEMV(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SGEMM_BI_TN_GEMV(f16,  __half,        from_f_f16)

// NT GEMV: dX[M,K] = alpha * dY[M] @ W^T[K]. Pure outer product.
#define DEFINE_SGEMM_BI_NT_GEMV(SUFFIX, T_ACT, FROM_F)                        \
extern "C" __global__ __launch_bounds__(256)                                  \
void sgemm_bi_nt_gemv_##SUFFIX(                                               \
    T_ACT* __restrict__ dX,                                                   \
    const T_ACT* __restrict__ dY,                                             \
    const T_ACT* __restrict__ W,                                              \
    float alpha,                                                              \
    int M, int K,                                                             \
    int ldx, int ldy                                                          \
) {                                                                           \
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;                    \
    const int total = M * K;                                                  \
    if (tid >= total) return;                                                 \
    const int m = tid / K;                                                    \
    const int k = tid - m * K;                                                \
    dX[m * ldx + k] = FROM_F(alpha * to_f(dY[m * ldy]) * to_f(W[k]));         \
}

DEFINE_SGEMM_BI_NT_GEMV(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SGEMM_BI_NT_GEMV(f16,  __half,        from_f_f16)

// Ultra-thin-M NN: M in [1, 32), smem-staged X row, 8-warp K-slab partials
// with the fixed 8-way tree reduce. smem_x stays f32 (upcast at stage-in) —
// the FMA chain is then bit-identical to the f32 kernel on upcast inputs.
#define DEFINE_SGEMM_BI_NN_ULTRA_THIN(SUFFIX, T_ACT, FROM_F)                  \
extern "C" __global__ __launch_bounds__(256, 4)                               \
void sgemm_bi_nn_ultra_thin_##SUFFIX(                                         \
    T_ACT* __restrict__ Y,                                                    \
    const T_ACT* __restrict__ X,                                              \
    const T_ACT* __restrict__ W,                                              \
    const float* __restrict__ bias,                                           \
    float alpha, float beta,                                                  \
    int M, int N, int K,                                                      \
    int lda, int ldb, int ldc                                                 \
) {                                                                           \
    const int tid = threadIdx.x;                                              \
    const int warp = tid >> 5;                                                \
    const int lane = tid & 31;                                                \
    const int n_tile = blockIdx.x;                                            \
    const int m = blockIdx.y;                                                 \
    if (m >= M) return;                                                       \
    const int col = n_tile * 32 + lane;                                       \
    extern __shared__ float smem_x[];                                         \
    for (int k = tid; k < K; k += blockDim.x) {                               \
        smem_x[k] = to_f(X[m * lda + k]);                                     \
    }                                                                         \
    __syncthreads();                                                          \
    const int K_per_warp = (K + 7) / 8;                                       \
    const int k_start = warp * K_per_warp;                                    \
    const int k_end = (k_start + K_per_warp > K) ? K : (k_start + K_per_warp);\
    float acc = 0.0f;                                                         \
    if (col < N) {                                                            \
        for (int k = k_start; k < k_end; k++) {                               \
            acc = __fmaf_rn(smem_x[k], to_f(W[k * ldb + col]), acc);          \
        }                                                                     \
    }                                                                         \
    __shared__ float smem_partials[8 * 32];                                   \
    smem_partials[warp * 32 + lane] = acc;                                    \
    __syncthreads();                                                          \
    if (warp == 0 && col < N) {                                               \
        float p0 = smem_partials[0 * 32 + lane];                              \
        float p1 = smem_partials[1 * 32 + lane];                              \
        float p2 = smem_partials[2 * 32 + lane];                              \
        float p3 = smem_partials[3 * 32 + lane];                              \
        float p4 = smem_partials[4 * 32 + lane];                              \
        float p5 = smem_partials[5 * 32 + lane];                              \
        float p6 = smem_partials[6 * 32 + lane];                              \
        float p7 = smem_partials[7 * 32 + lane];                              \
        float s01 = p0 + p1;                                                  \
        float s23 = p2 + p3;                                                  \
        float s45 = p4 + p5;                                                  \
        float s67 = p6 + p7;                                                  \
        float s0123 = s01 + s23;                                              \
        float s4567 = s45 + s67;                                              \
        float sum = s0123 + s4567;                                            \
        float val = alpha * sum;                                              \
        if (bias != nullptr) val += bias[col];                                \
        if (beta != 0.0f) val += beta * to_f(Y[m * ldc + col]);               \
        Y[m * ldc + col] = FROM_F(val);                                       \
    }                                                                         \
}

DEFINE_SGEMM_BI_NN_ULTRA_THIN(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SGEMM_BI_NN_ULTRA_THIN(f16,  __half,        from_f_f16)

// Typed narrow-N NN (generic over tile): A1 route — smem stays f32, typed
// inputs upcast at the SYNC stage-in with the exact zero-fill predication of
// the f32 kernels (the per-tile cp.async there is wait_all-fenced, i.e. not
// pipelined, so sync loads cost ~nothing). FMA mainloop, bias pre-seed at
// K=0 and the scalar-N epilogue are byte-identical to the f32 kernels —
// outputs are bit-identical to "upcast inputs, run f32 kernel".
#define DEFINE_SGEMM_BI_NN_NARROW_T(NAME, T_ACT, FROM_F, BM_, BN_, BK_, WM_, WN_, TM_, TN_, NTHR_, LB_) \
extern "C" __global__ __launch_bounds__(NTHR_, LB_)                           \
void NAME(                                                                    \
    T_ACT* __restrict__ C,                                                    \
    const T_ACT* __restrict__ A,                                              \
    const T_ACT* __restrict__ B,                                              \
    const float* __restrict__ bias,                                           \
    float alpha, float beta,                                                  \
    int M, int N, int K,                                                      \
    int lda, int ldb, int ldc,                                                \
    int post_op                                                               \
) {                                                                           \
    (void)post_op;                                                            \
    __shared__ float As[BK_ * (BM_ + SMEM_A_PAD)];                            \
    __shared__ float Bs[BK_ * (BN_ + SMEM_B_PAD)];                            \
    int num_pid_m = (M + BM_ - 1) / BM_;                                      \
    int num_pid_n = (N + BN_ - 1) / BN_;                                      \
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;                           \
    int warpIdx = threadIdx.x / WARPSIZE;                                     \
    int warpCol = warpIdx % (BN_ / WN_);                                      \
    int warpRow = warpIdx / (BN_ / WN_);                                      \
    int tidInWarp = threadIdx.x % WARPSIZE;                                   \
    int threadColInWarp = tidInWarp % (WN_ / TN_);                            \
    int threadRowInWarp = tidInWarp / (WN_ / TN_);                            \
    float regM[TM_] = {0.0f};                                                 \
    float regN[TN_] = {0.0f};                                                 \
    int tile_id = blockIdx.x;                                                 \
    {                                                                         \
        int group_id = tile_id / num_pid_in_group;                            \
        int first_pid_m = group_id * SGB_GROUP_M;                             \
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);         \
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m); \
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;              \
        const T_ACT* A_block = A + pid_m * BM_ * lda;                         \
        const T_ACT* B_block = B + pid_n * BN_;                               \
        T_ACT* C_warp = C + (pid_m * BM_ + warpRow * WM_) * ldc               \
                        + pid_n * BN_ + warpCol * WN_;                        \
        float threadResults[TM_ * TN_];                                       \
        _Pragma("unroll")                                                     \
        for (int rm = 0; rm < TM_; ++rm) {                                    \
            int g_col_base = pid_n * BN_ + warpCol * WN_ + threadColInWarp * TN_; \
            _Pragma("unroll")                                                 \
            for (int rn = 0; rn < TN_; ++rn) {                                \
                int g_col = g_col_base + rn;                                  \
                threadResults[rm * TN_ + rn] =                                \
                    (bias != nullptr && g_col < N) ? bias[g_col] : 0.0f;      \
            }                                                                 \
        }                                                                     \
        for (int bkIdx = 0; bkIdx < K; bkIdx += BK_) {                        \
            for (int idx = threadIdx.x; idx < BK_ * BM_; idx += NTHR_) {      \
                int _k = idx / BM_;                                           \
                int _m = idx % BM_;                                           \
                int _g_row = pid_m * BM_ + _m;                                \
                int _g_col = bkIdx + _k;                                      \
                As[_k * (BM_ + SMEM_A_PAD) + _m] =                            \
                    (_g_row < M && _g_col < K)                                \
                        ? to_f(A_block[_m * lda + _k]) : 0.0f;                \
            }                                                                 \
            for (int idx = threadIdx.x; idx < BK_ * BN_; idx += NTHR_) {      \
                int _k = idx / BN_;                                           \
                int _n = idx % BN_;                                           \
                int g_row = bkIdx + _k;                                       \
                int g_col = pid_n * BN_ + _n;                                 \
                Bs[_k * (BN_ + SMEM_B_PAD) + _n] =                            \
                    (g_row < K && g_col < N)                                  \
                        ? to_f(B_block[_k * ldb + _n]) : 0.0f;                \
            }                                                                 \
            __syncthreads();                                                  \
            for (int dotIdx = 0; dotIdx < BK_; ++dotIdx) {                    \
                for (int i = 0; i < TM_; ++i) {                               \
                    regM[i] = As[dotIdx * (BM_ + SMEM_A_PAD)                  \
                                 + warpRow * WM_ + threadRowInWarp * TM_ + i]; \
                }                                                             \
                for (int i = 0; i < TN_; ++i) {                               \
                    regN[i] = Bs[dotIdx * (BN_ + SMEM_B_PAD)                  \
                                 + warpCol * WN_ + threadColInWarp * TN_ + i]; \
                }                                                             \
                for (int resIdxM = 0; resIdxM < TM_; ++resIdxM) {             \
                    for (int resIdxN = 0; resIdxN < TN_; ++resIdxN) {         \
                        threadResults[resIdxM * TN_ + resIdxN] = __fmaf_rn(   \
                            regM[resIdxM], regN[resIdxN],                     \
                            threadResults[resIdxM * TN_ + resIdxN]);          \
                    }                                                         \
                }                                                             \
            }                                                                 \
            A_block += BK_;                                                   \
            B_block += BK_ * ldb;                                             \
            __syncthreads();                                                  \
        }                                                                     \
        for (int resIdxM = 0; resIdxM < TM_; ++resIdxM) {                     \
            int g_row = pid_m * BM_ + warpRow * WM_ + threadRowInWarp * TM_ + resIdxM; \
            if (g_row >= M) continue;                                         \
            for (int resIdxN = 0; resIdxN < TN_; ++resIdxN) {                 \
                int g_col = pid_n * BN_ + warpCol * WN_ + threadColInWarp * TN_ + resIdxN; \
                if (g_col >= N) continue;                                     \
                float val = alpha * threadResults[resIdxM * TN_ + resIdxN];   \
                int coff = (threadRowInWarp * TM_ + resIdxM) * ldc            \
                           + threadColInWarp * TN_ + resIdxN;                 \
                if (beta != 0.0f) val += beta * to_f(C_warp[coff]);           \
                C_warp[coff] = FROM_F(val);                                   \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
}

DEFINE_SGEMM_BI_NN_NARROW_T(sgemm_bi_nn_narrow_bf16, __nv_bfloat16, from_f_bf16, 64, 32, 16, 32, 16, 4, 4, 128, 4)
DEFINE_SGEMM_BI_NN_NARROW_T(sgemm_bi_nn_narrow_f16,  __half,        from_f_f16,  64, 32, 16, 32, 16, 4, 4, 128, 4)
DEFINE_SGEMM_BI_NN_NARROW_T(sgemm_bi_nn_narrow_small_bf16, __nv_bfloat16, from_f_bf16, 16, 16, 16, 8, 16, 2, 2, 64, 8)
DEFINE_SGEMM_BI_NN_NARROW_T(sgemm_bi_nn_narrow_small_f16,  __half,        from_f_f16,  16, 16, 16, 8, 16, 2, 2, 64, 8)

// Typed narrow TN (dW): C stays f32 (master grad, += epilogue); A=X and
// B=dY are typed. Same A1 route: f32 smem, sync typed stage-in with the
// f32 kernels' exact zero-fill predication; FMA chain unchanged.
#define DEFINE_SGEMM_BI_TN_NARROW_T(NAME, T_ACT)                              \
extern "C" __global__ __launch_bounds__(128, 4)                               \
void NAME(                                                                    \
    float* __restrict__ C,                                                    \
    const T_ACT* __restrict__ A,                                              \
    const T_ACT* __restrict__ B,                                              \
    float alpha,                                                              \
    int M_red, int K_out, int N                                               \
) {                                                                           \
    __shared__ float As[16 * (64 + SMEM_A_PAD)];                              \
    __shared__ float Bs[16 * (32 + SMEM_B_PAD)];                              \
    int num_pid_m = (K_out + 63) / 64;                                        \
    int num_pid_n = (N + 31) / 32;                                            \
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;                           \
    int warpIdx = threadIdx.x / WARPSIZE;                                     \
    int warpCol = warpIdx % 2;                                                \
    int warpRow = warpIdx / 2;                                                \
    int tidInWarp = threadIdx.x % WARPSIZE;                                   \
    int threadColInWarp = tidInWarp % 4;                                      \
    int threadRowInWarp = tidInWarp / 4;                                      \
    float regM[4] = {0.0f};                                                   \
    float regN[4] = {0.0f};                                                   \
    int tile_id = blockIdx.x;                                                 \
    {                                                                         \
        float threadResults[16] = {0.0f};                                     \
        int group_id = tile_id / num_pid_in_group;                            \
        int first_pid_m = group_id * SGB_GROUP_M;                             \
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);         \
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m); \
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;              \
        float* C_warp = C + (pid_m * 64 + warpRow * 32) * N                   \
                        + pid_n * 32 + warpCol * 16;                          \
        for (int mIdx = 0; mIdx < M_red; mIdx += 16) {                        \
            for (int idx = threadIdx.x; idx < 16 * 64; idx += 128) {          \
                int _k = idx / 64;                                            \
                int _m = idx % 64;                                            \
                int _g_m = mIdx + _k;                                         \
                int _g_k = pid_m * 64 + _m;                                   \
                As[_k * (64 + SMEM_A_PAD) + _m] =                             \
                    (_g_m < M_red && _g_k < K_out)                            \
                        ? to_f(A[(long long)_g_m * K_out + _g_k]) : 0.0f;     \
            }                                                                 \
            for (int idx = threadIdx.x; idx < 16 * 32; idx += 128) {          \
                int _k = idx / 32;                                            \
                int _n = idx % 32;                                            \
                int g_m = mIdx + _k;                                          \
                int g_n = pid_n * 32 + _n;                                    \
                Bs[_k * (32 + SMEM_B_PAD) + _n] =                             \
                    (g_m < M_red && g_n < N)                                  \
                        ? to_f(B[(long long)g_m * N + g_n]) : 0.0f;           \
            }                                                                 \
            __syncthreads();                                                  \
            for (int dotIdx = 0; dotIdx < 16; ++dotIdx) {                     \
                for (int i = 0; i < 4; ++i) {                                 \
                    regM[i] = As[dotIdx * (64 + SMEM_A_PAD)                   \
                                 + warpRow * 32 + threadRowInWarp * 4 + i];   \
                }                                                             \
                for (int i = 0; i < 4; ++i) {                                 \
                    regN[i] = Bs[dotIdx * (32 + SMEM_B_PAD)                   \
                                 + warpCol * 16 + threadColInWarp * 4 + i];   \
                }                                                             \
                for (int rm = 0; rm < 4; ++rm) {                              \
                    for (int rn = 0; rn < 4; ++rn) {                          \
                        threadResults[rm * 4 + rn] = __fmaf_rn(               \
                            regM[rm], regN[rn], threadResults[rm * 4 + rn]);  \
                    }                                                         \
                }                                                             \
            }                                                                 \
            __syncthreads();                                                  \
        }                                                                     \
        for (int rm = 0; rm < 4; ++rm) {                                      \
            int g_row = pid_m * 64 + warpRow * 32 + threadRowInWarp * 4 + rm; \
            if (g_row >= K_out) continue;                                     \
            for (int rn = 0; rn < 4; ++rn) {                                  \
                int g_col = pid_n * 32 + warpCol * 16 + threadColInWarp * 4 + rn; \
                if (g_col >= N) continue;                                     \
                C_warp[(threadRowInWarp * 4 + rm) * N + threadColInWarp * 4 + rn] += \
                    alpha * threadResults[rm * 4 + rn];                       \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
}

DEFINE_SGEMM_BI_TN_NARROW_T(sgemm_bi_tn_narrow_bf16, __nv_bfloat16)
DEFINE_SGEMM_BI_TN_NARROW_T(sgemm_bi_tn_narrow_f16,  __half)

// Typed narrow NT (dX): C=dX typed output (overwrite), A=dY and B=W typed.
// B tile staged TRANSPOSED (rows = reduction n, cols = k_out), exactly as
// the f32 kernel's float4 transposed stores.
#define DEFINE_SGEMM_BI_NT_NARROW_T(NAME, T_ACT, FROM_F)                      \
extern "C" __global__ __launch_bounds__(128, 4)                               \
void NAME(                                                                    \
    T_ACT* __restrict__ C,                                                    \
    const T_ACT* __restrict__ A,                                              \
    const T_ACT* __restrict__ B,                                              \
    float alpha,                                                              \
    int M, int N, int K_out                                                   \
) {                                                                           \
    __shared__ float As[16 * (64 + SMEM_A_PAD)];                              \
    __shared__ float Bs[16 * (32 + SMEM_B_PAD)];                              \
    int num_pid_m = (M + 63) / 64;                                            \
    int num_pid_n = (K_out + 31) / 32;                                        \
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;                           \
    int warpIdx = threadIdx.x / WARPSIZE;                                     \
    int warpCol = warpIdx % 2;                                                \
    int warpRow = warpIdx / 2;                                                \
    int tidInWarp = threadIdx.x % WARPSIZE;                                   \
    int threadColInWarp = tidInWarp % 4;                                      \
    int threadRowInWarp = tidInWarp / 4;                                      \
    float regM[4] = {0.0f};                                                   \
    float regN[4] = {0.0f};                                                   \
    int tile_id = blockIdx.x;                                                 \
    {                                                                         \
        float threadResults[16] = {0.0f};                                     \
        int group_id = tile_id / num_pid_in_group;                            \
        int first_pid_m = group_id * SGB_GROUP_M;                             \
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);         \
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m); \
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;              \
        for (int nIdx = 0; nIdx < N; nIdx += 16) {                            \
            for (int idx = threadIdx.x; idx < 16 * 64; idx += 128) {          \
                int _n = idx / 64;                                            \
                int _m = idx % 64;                                            \
                int _g_m = pid_m * 64 + _m;                                   \
                int _g_n = nIdx + _n;                                         \
                As[_n * (64 + SMEM_A_PAD) + _m] =                             \
                    (_g_m < M && _g_n < N)                                    \
                        ? to_f(A[(long long)_g_m * N + _g_n]) : 0.0f;         \
            }                                                                 \
            for (int idx = threadIdx.x; idx < 16 * 32; idx += 128) {          \
                int _n = idx / 32;                                            \
                int _kb = idx % 32;                                           \
                int g_k = pid_n * 32 + _kb;                                   \
                int g_n = nIdx + _n;                                          \
                Bs[_n * (32 + SMEM_B_PAD) + _kb] =                            \
                    (g_k < K_out && g_n < N)                                  \
                        ? to_f(B[(long long)g_k * N + g_n]) : 0.0f;           \
            }                                                                 \
            __syncthreads();                                                  \
            for (int dotIdx = 0; dotIdx < 16; ++dotIdx) {                     \
                for (int i = 0; i < 4; ++i) {                                 \
                    regM[i] = As[dotIdx * (64 + SMEM_A_PAD)                   \
                                 + warpRow * 32 + threadRowInWarp * 4 + i];   \
                }                                                             \
                for (int i = 0; i < 4; ++i) {                                 \
                    regN[i] = Bs[dotIdx * (32 + SMEM_B_PAD)                   \
                                 + warpCol * 16 + threadColInWarp * 4 + i];   \
                }                                                             \
                for (int rm = 0; rm < 4; ++rm) {                              \
                    for (int rn = 0; rn < 4; ++rn) {                          \
                        threadResults[rm * 4 + rn] = __fmaf_rn(               \
                            regM[rm], regN[rn], threadResults[rm * 4 + rn]);  \
                    }                                                         \
                }                                                             \
            }                                                                 \
            __syncthreads();                                                  \
        }                                                                     \
        T_ACT* C_warp = C + (pid_m * 64 + warpRow * 32) * K_out               \
                        + pid_n * 32 + warpCol * 16;                          \
        for (int rm = 0; rm < 4; ++rm) {                                      \
            int g_row = pid_m * 64 + warpRow * 32 + threadRowInWarp * 4 + rm; \
            if (g_row >= M) continue;                                         \
            for (int rn = 0; rn < 4; ++rn) {                                  \
                int g_col = pid_n * 32 + warpCol * 16 + threadColInWarp * 4 + rn; \
                if (g_col >= K_out) continue;                                 \
                C_warp[(threadRowInWarp * 4 + rm) * K_out                     \
                       + threadColInWarp * 4 + rn] =                          \
                    FROM_F(alpha * threadResults[rm * 4 + rn]);               \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
}

DEFINE_SGEMM_BI_NT_NARROW_T(sgemm_bi_nt_narrow_bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SGEMM_BI_NT_NARROW_T(sgemm_bi_nt_narrow_f16,  __half,        from_f_f16)

// ============================================================================
// Typed Big NN/TN/NT (stage 3 v1): bf16/f16 twins of the Big warptiling
// kernels. Smem stays f32 — fragment loads and the __fmaf_rn chain are
// BYTE-IDENTICAL to the f32 kernels; only the staging instruction differs
// (synchronous ld.global -> to_f -> st.shared replaces cp.async, since
// cp.async cannot copy or convert 2-byte elements into the transposed f32
// As layout). Smem CONTENTS per (stage, cell) are bit-equal to the f32
// kernel's tiles on upcast inputs (incl. zero-fill OOB), so each typed
// kernel is bit-identical to "upcast inputs, run the f32 Big kernel,
// RNE-downcast the output" — the stage-2 contract.
//
// Pipeline: 2-stage rotation retained. One __syncthreads() per K-tile:
//   stage(0); loop { sync; if(next) stage(write); compute(read); rotate; }
// The sync at loop top (a) publishes the previous iteration's staging and
// (b) fences compute(read) of iter i-1 before iter i overwrites that stage
// (K_PIPE=2: write(i) == read(i-1)). Latency hiding falls to warp-level
// parallelism (8 warps/CTA, 2 CTA/SM) instead of async DMA.
// Dynamic smem = 33 KB -> host must set MAX_DYNAMIC_SHARED_SIZE_BYTES
// (34 KB) on these handles, same as the f32 Big kernels.

// Shared compute block: register-fragment double-buffered BK dot-product
// sweep, verbatim semantics of the f32 Big mainloop. Uses As_buf/Bs_buf/
// read_stage/regM/regN/threadResults and the warp placement values from
// the enclosing kernel scope.
#define SGB_T_COMPUTE_TILE()                                                   \
    do {                                                                       \
        float* As_rd = As_buf + read_stage * A_STAGE;                          \
        float* Bs_rd = Bs_buf + read_stage * B_STAGE;                          \
        float regM_next[WMITER * TM];                                          \
        float regN_next[WNITER * TN];                                          \
        _Pragma("unroll")                                                      \
        for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)            \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < TM; ++i)                                       \
                regM[wSubRowIdx * TM + i] =                                    \
                    As_rd[0 * (BM + SMEM_A_PAD) + warpRow * WM +               \
                          wSubRowIdx * WSUBM + threadRowInWarp * TM + i];      \
        _Pragma("unroll")                                                      \
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)            \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < TN; ++i)                                       \
                regN[wSubColIdx * TN + i] =                                    \
                    Bs_rd[0 * (BN + SMEM_B_PAD) + warpCol * WN +               \
                          wSubColIdx * WSUBN + threadColInWarp * TN + i];      \
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {                          \
            if (dotIdx + 1 < BK) {                                             \
                _Pragma("unroll")                                              \
                for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)    \
                    _Pragma("unroll")                                          \
                    for (int i = 0; i < TM; ++i)                               \
                        regM_next[wSubRowIdx * TM + i] =                       \
                            As_rd[(dotIdx + 1) * (BM + SMEM_A_PAD) +           \
                                  warpRow * WM + wSubRowIdx * WSUBM +          \
                                  threadRowInWarp * TM + i];                   \
                _Pragma("unroll")                                              \
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)    \
                    _Pragma("unroll")                                          \
                    for (int i = 0; i < TN; ++i)                               \
                        regN_next[wSubColIdx * TN + i] =                       \
                            Bs_rd[(dotIdx + 1) * (BN + SMEM_B_PAD) +           \
                                  warpCol * WN + wSubColIdx * WSUBN +          \
                                  threadColInWarp * TN + i];                   \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)        \
                _Pragma("unroll")                                              \
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)    \
                    _Pragma("unroll")                                          \
                    for (int resIdxM = 0; resIdxM < TM; ++resIdxM)             \
                        _Pragma("unroll")                                      \
                        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {       \
                            int idx = (wSubRowIdx * TM + resIdxM) *            \
                                          (WNITER * TN) +                      \
                                      wSubColIdx * TN + resIdxN;               \
                            threadResults[idx] = __fmaf_rn(                    \
                                regM[wSubRowIdx * TM + resIdxM],               \
                                regN[wSubColIdx * TN + resIdxN],               \
                                threadResults[idx]);                           \
                        }                                                      \
            if (dotIdx + 1 < BK) {                                             \
                _Pragma("unroll")                                              \
                for (int i = 0; i < WMITER * TM; ++i) regM[i] = regM_next[i];  \
                _Pragma("unroll")                                              \
                for (int i = 0; i < WNITER * TN; ++i) regN[i] = regN_next[i];  \
            }                                                                  \
        }                                                                      \
    } while (0)

// NN staging: As[k][m] = A[(pid_m*BM+m)*lda + bk+k], Bs[k][n] = B[(bk+k)*ldb
// + pid_n*BN+n], zero-fill OOB. Iteration order picked for contiguous global
// reads (A along k, B along n); placement equals the f32 cp.async tiles.
#define SGB_T_STAGE_NN(s, bkIdx)                                               \
    do {                                                                       \
        float* _As_w = As_buf + (s) * A_STAGE;                                 \
        float* _Bs_w = Bs_buf + (s) * B_STAGE;                                 \
        for (int _i = threadIdx.x; _i < BM * BK; _i += NUM_THREADS) {          \
            int _m = _i / BK;                                                  \
            int _k = _i % BK;                                                  \
            int _gr = pid_m * BM + _m;                                         \
            int _gc = (bkIdx) + _k;                                            \
            _As_w[_k * (BM + SMEM_A_PAD) + _m] =                               \
                (_gr < M && _gc < K)                                           \
                    ? to_f(A[(long long)_gr * lda + _gc])                      \
                    : 0.0f;                                                    \
        }                                                                      \
        for (int _i = threadIdx.x; _i < BK * BN; _i += NUM_THREADS) {          \
            int _k = _i / BN;                                                  \
            int _n = _i % BN;                                                  \
            int _gr = (bkIdx) + _k;                                            \
            int _gc = pid_n * BN + _n;                                         \
            _Bs_w[_k * (BN + SMEM_B_PAD) + _n] =                               \
                (_gr < K && _gc < N)                                           \
                    ? to_f(B[(long long)_gr * ldb + _gc])                      \
                    : 0.0f;                                                    \
        }                                                                      \
    } while (0)

// TN staging (A = X[M_red, K_out] read transposed, B = dY[M_red, N]):
// As[r][c] = A[(mIdx+r)*K_out + pid_m*BM+c], Bs[r][c] = B[(mIdx+r)*N +
// pid_n*BN+c]. Contiguous global reads along c.
#define SGB_T_STAGE_TN(s, mIdx)                                                \
    do {                                                                       \
        float* _As_w = As_buf + (s) * A_STAGE;                                 \
        float* _Bs_w = Bs_buf + (s) * B_STAGE;                                 \
        for (int _i = threadIdx.x; _i < BK * BM; _i += NUM_THREADS) {          \
            int _r = _i / BM;                                                  \
            int _c = _i % BM;                                                  \
            int _gm = (mIdx) + _r;                                             \
            int _gk = pid_m * BM + _c;                                         \
            _As_w[_r * (BM + SMEM_A_PAD) + _c] =                               \
                (_gm < M_red && _gk < K_out)                                   \
                    ? to_f(A[(long long)_gm * K_out + _gk])                    \
                    : 0.0f;                                                    \
        }                                                                      \
        for (int _i = threadIdx.x; _i < BK * BN; _i += NUM_THREADS) {          \
            int _r = _i / BN;                                                  \
            int _c = _i % BN;                                                  \
            int _gm = (mIdx) + _r;                                             \
            int _gn = pid_n * BN + _c;                                         \
            _Bs_w[_r * (BN + SMEM_B_PAD) + _c] =                               \
                (_gm < M_red && _gn < N)                                       \
                    ? to_f(B[(long long)_gm * N + _gn])                        \
                    : 0.0f;                                                    \
        }                                                                      \
    } while (0)

// NT staging (A = dY[M, N], B = W[K_out, N], both read along the N
// reduction): As[r][c] = A[(pid_m*BM+c)*N + nIdx+r], Bs[r][c] =
// B[(pid_n*BN+c)*N + nIdx+r]. Contiguous global reads along r.
#define SGB_T_STAGE_NT(s, nIdx)                                                \
    do {                                                                       \
        float* _As_w = As_buf + (s) * A_STAGE;                                 \
        float* _Bs_w = Bs_buf + (s) * B_STAGE;                                 \
        for (int _i = threadIdx.x; _i < BM * BK; _i += NUM_THREADS) {          \
            int _c = _i / BK;                                                  \
            int _r = _i % BK;                                                  \
            int _gm = pid_m * BM + _c;                                         \
            int _gn = (nIdx) + _r;                                             \
            _As_w[_r * (BM + SMEM_A_PAD) + _c] =                               \
                (_gm < M && _gn < N)                                           \
                    ? to_f(A[(long long)_gm * N + _gn])                        \
                    : 0.0f;                                                    \
        }                                                                      \
        for (int _i = threadIdx.x; _i < BN * BK; _i += NUM_THREADS) {          \
            int _c = _i / BK;                                                  \
            int _r = _i % BK;                                                  \
            int _gk = pid_n * BN + _c;                                         \
            int _gn = (nIdx) + _r;                                             \
            _Bs_w[_r * (BN + SMEM_B_PAD) + _c] =                               \
                (_gk < K_out && _gn < N)                                       \
                    ? to_f(B[(long long)_gk * N + _gn])                        \
                    : 0.0f;                                                    \
        }                                                                      \
    } while (0)

#define DEFINE_SGEMM_BI_NN_BIG_T(SUFFIX, T_ACT, FROM_F)                        \
extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)                        \
void sgemm_bi_nn_big_##SUFFIX(                                                 \
    T_ACT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    const float* __restrict__ bias,                                            \
    float alpha, float beta,                                                   \
    int M, int N, int K,                                                       \
    int lda, int ldb, int ldc                                                  \
) {                                                                            \
    assert(alpha == 1.0f || bias == nullptr);                                  \
    constexpr int K_PIPE = 2;                                                  \
    extern __shared__ __align__(16) float smem[];                              \
    constexpr int A_STAGE = BK * (BM + SMEM_A_PAD);                            \
    constexpr int B_STAGE = BK * (BN + SMEM_B_PAD);                            \
    float* As_buf = smem;                                                      \
    float* Bs_buf = smem + K_PIPE * A_STAGE;                                   \
    int num_pid_m = (M + BM - 1) / BM;                                         \
    int num_pid_n = (N + BN - 1) / BN;                                         \
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;                            \
    int warpIdx = threadIdx.x / WARPSIZE;                                      \
    int warpCol = warpIdx % (BN / WN);                                         \
    int warpRow = warpIdx / (BN / WN);                                         \
    int tidInWarp = threadIdx.x % WARPSIZE;                                    \
    int threadColInWarp = tidInWarp % (WSUBN / TN);                            \
    int threadRowInWarp = tidInWarp / (WSUBN / TN);                            \
    float regM[WMITER * TM] = {0.0f};                                          \
    float regN[WNITER * TN] = {0.0f};                                          \
    int tile_id = blockIdx.x;                                                  \
    float threadResults[WMITER * TM * WNITER * TN];                            \
    int group_id = tile_id / num_pid_in_group;                                 \
    int first_pid_m = group_id * SGB_GROUP_M;                                  \
    int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);              \
    int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);   \
    int pid_n = (tile_id % num_pid_in_group) / group_size_m;                   \
    if (bias != nullptr) {                                                     \
        _Pragma("unroll")                                                      \
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {          \
            _Pragma("unroll")                                                  \
            for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {                   \
                int g_col = pid_n * BN + warpCol * WN + wSubColIdx * WSUBN +   \
                            threadColInWarp * TN + resIdxN;                    \
                float b_val = (g_col < N) ? bias[g_col] : 0.0f;                \
                _Pragma("unroll")                                              \
                for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {  \
                    _Pragma("unroll")                                          \
                    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {           \
                        int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN)  \
                                  + wSubColIdx * TN + resIdxN;                 \
                        threadResults[idx] = b_val;                            \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } else {                                                                   \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < WMITER * TM * WNITER * TN; ++i) {                  \
            threadResults[i] = 0.0f;                                           \
        }                                                                      \
    }                                                                          \
    T_ACT* C_warp =                                                            \
        C + (pid_m * BM + warpRow * WM) * ldc + pid_n * BN + warpCol * WN;     \
    int num_k_tiles = (K + BK - 1) / BK;                                       \
    SGB_T_STAGE_NN(0, 0);                                                      \
    int read_stage = 0;                                                        \
    int write_stage = 1;                                                       \
    for (int tile = 0; tile < num_k_tiles; ++tile) {                           \
        __syncthreads();                                                       \
        if (tile + 1 < num_k_tiles) {                                          \
            SGB_T_STAGE_NN(write_stage, (tile + 1) * BK);                      \
        }                                                                      \
        SGB_T_COMPUTE_TILE();                                                  \
        read_stage = (read_stage + 1) % K_PIPE;                                \
        write_stage = (write_stage + 1) % K_PIPE;                              \
    }                                                                          \
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {              \
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {          \
            T_ACT* C_sub = C_warp + wSubRowIdx * WSUBM * ldc +                 \
                           wSubColIdx * WSUBN;                                 \
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {                   \
                int g_row = pid_m * BM + warpRow * WM + wSubRowIdx * WSUBM +   \
                            threadRowInWarp * TM + resIdxM;                    \
                if (g_row >= M) continue;                                      \
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {               \
                    int g_col = pid_n * BN + warpCol * WN +                    \
                                wSubColIdx * WSUBN + threadColInWarp * TN +    \
                                resIdxN;                                       \
                    if (g_col >= N) continue;                                  \
                    int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +    \
                              wSubColIdx * TN + resIdxN;                       \
                    int c_off = (threadRowInWarp * TM + resIdxM) * ldc +       \
                                threadColInWarp * TN + resIdxN;                \
                    float val = alpha * threadResults[idx];                    \
                    if (beta != 0.0f) val += beta * to_f(C_sub[c_off]);        \
                    C_sub[c_off] = FROM_F(val);                                \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

DEFINE_SGEMM_BI_NN_BIG_T(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SGEMM_BI_NN_BIG_T(f16,  __half,        from_f_f16)

#define DEFINE_SGEMM_BI_TN_BIG_T(SUFFIX, T_ACT)                                \
extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)                        \
void sgemm_bi_tn_big_##SUFFIX(                                                 \
    float* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M_red, int K_out, int N                                                \
) {                                                                            \
    constexpr int K_PIPE = 2;                                                  \
    extern __shared__ __align__(16) float smem[];                              \
    constexpr int A_STAGE = BK * (BM + SMEM_A_PAD);                            \
    constexpr int B_STAGE = BK * (BN + SMEM_B_PAD);                            \
    float* As_buf = smem;                                                      \
    float* Bs_buf = smem + K_PIPE * A_STAGE;                                   \
    int num_pid_m = (K_out + BM - 1) / BM;                                     \
    int num_pid_n = (N + BN - 1) / BN;                                         \
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;                            \
    int warpIdx = threadIdx.x / WARPSIZE;                                      \
    int warpCol = warpIdx % (BN / WN);                                         \
    int warpRow = warpIdx / (BN / WN);                                         \
    int tidInWarp = threadIdx.x % WARPSIZE;                                    \
    int threadColInWarp = tidInWarp % (WSUBN / TN);                            \
    int threadRowInWarp = tidInWarp / (WSUBN / TN);                            \
    float regM[WMITER * TM] = {0.0f};                                          \
    float regN[WNITER * TN] = {0.0f};                                          \
    int tile_id = blockIdx.x;                                                  \
    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};                   \
    int group_id = tile_id / num_pid_in_group;                                 \
    int first_pid_m = group_id * SGB_GROUP_M;                                  \
    int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);              \
    int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);   \
    int pid_n = (tile_id % num_pid_in_group) / group_size_m;                   \
    float* C_warp =                                                            \
        C + (pid_m * BM + warpRow * WM) * N + pid_n * BN + warpCol * WN;       \
    int num_m_tiles = (M_red + BK - 1) / BK;                                   \
    SGB_T_STAGE_TN(0, 0);                                                      \
    int read_stage = 0;                                                        \
    int write_stage = 1;                                                       \
    for (int tile = 0; tile < num_m_tiles; ++tile) {                           \
        __syncthreads();                                                       \
        if (tile + 1 < num_m_tiles) {                                          \
            SGB_T_STAGE_TN(write_stage, (tile + 1) * BK);                      \
        }                                                                      \
        SGB_T_COMPUTE_TILE();                                                  \
        read_stage = (read_stage + 1) % K_PIPE;                                \
        write_stage = (write_stage + 1) % K_PIPE;                              \
    }                                                                          \
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {              \
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {          \
            float* C_sub =                                                     \
                C_warp + wSubRowIdx * WSUBM * N + wSubColIdx * WSUBN;          \
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {                   \
                int g_row = pid_m * BM + warpRow * WM + wSubRowIdx * WSUBM +   \
                            threadRowInWarp * TM + resIdxM;                    \
                if (g_row >= K_out) continue;                                  \
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {               \
                    int g_col = pid_n * BN + warpCol * WN +                    \
                                wSubColIdx * WSUBN + threadColInWarp * TN +    \
                                resIdxN;                                       \
                    if (g_col >= N) continue;                                  \
                    int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +    \
                              wSubColIdx * TN + resIdxN;                       \
                    C_sub[(threadRowInWarp * TM + resIdxM) * N +               \
                          threadColInWarp * TN + resIdxN] +=                   \
                        alpha * threadResults[idx];                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

DEFINE_SGEMM_BI_TN_BIG_T(bf16, __nv_bfloat16)
DEFINE_SGEMM_BI_TN_BIG_T(f16,  __half)

#define DEFINE_SGEMM_BI_NT_BIG_T(SUFFIX, T_ACT, FROM_F)                        \
extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)                        \
void sgemm_bi_nt_big_##SUFFIX(                                                 \
    T_ACT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M, int N, int K_out                                                    \
) {                                                                            \
    constexpr int K_PIPE = 2;                                                  \
    extern __shared__ __align__(16) float smem[];                              \
    constexpr int A_STAGE = BK * (BM + SMEM_A_PAD);                            \
    constexpr int B_STAGE = BK * (BN + SMEM_B_PAD);                            \
    float* As_buf = smem;                                                      \
    float* Bs_buf = smem + K_PIPE * A_STAGE;                                   \
    int num_pid_m = (M + BM - 1) / BM;                                         \
    int num_pid_n = (K_out + BN - 1) / BN;                                     \
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;                            \
    int warpIdx = threadIdx.x / WARPSIZE;                                      \
    int warpCol = warpIdx % (BN / WN);                                         \
    int warpRow = warpIdx / (BN / WN);                                         \
    int tidInWarp = threadIdx.x % WARPSIZE;                                    \
    int threadColInWarp = tidInWarp % (WSUBN / TN);                            \
    int threadRowInWarp = tidInWarp / (WSUBN / TN);                            \
    float regM[WMITER * TM] = {0.0f};                                          \
    float regN[WNITER * TN] = {0.0f};                                          \
    int tile_id = blockIdx.x;                                                  \
    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};                   \
    int group_id = tile_id / num_pid_in_group;                                 \
    int first_pid_m = group_id * SGB_GROUP_M;                                  \
    int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);              \
    int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);   \
    int pid_n = (tile_id % num_pid_in_group) / group_size_m;                   \
    T_ACT* C_warp =                                                            \
        C + (pid_m * BM + warpRow * WM) * K_out + pid_n * BN + warpCol * WN;   \
    int num_n_tiles = (N + BK - 1) / BK;                                       \
    SGB_T_STAGE_NT(0, 0);                                                      \
    int read_stage = 0;                                                        \
    int write_stage = 1;                                                       \
    for (int tile = 0; tile < num_n_tiles; ++tile) {                           \
        __syncthreads();                                                       \
        if (tile + 1 < num_n_tiles) {                                          \
            SGB_T_STAGE_NT(write_stage, (tile + 1) * BK);                      \
        }                                                                      \
        SGB_T_COMPUTE_TILE();                                                  \
        read_stage = (read_stage + 1) % K_PIPE;                                \
        write_stage = (write_stage + 1) % K_PIPE;                              \
    }                                                                          \
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {              \
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {          \
            T_ACT* C_sub =                                                     \
                C_warp + wSubRowIdx * WSUBM * K_out + wSubColIdx * WSUBN;      \
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {                   \
                int g_row = pid_m * BM + warpRow * WM + wSubRowIdx * WSUBM +   \
                            threadRowInWarp * TM + resIdxM;                    \
                if (g_row >= M) continue;                                      \
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {               \
                    int g_col = pid_n * BN + warpCol * WN +                    \
                                wSubColIdx * WSUBN + threadColInWarp * TN +    \
                                resIdxN;                                       \
                    if (g_col >= K_out) continue;                              \
                    int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +    \
                              wSubColIdx * TN + resIdxN;                       \
                    C_sub[(threadRowInWarp * TM + resIdxM) * K_out +           \
                          threadColInWarp * TN + resIdxN] =                    \
                        FROM_F(alpha * threadResults[idx]);                    \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

DEFINE_SGEMM_BI_NT_BIG_T(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SGEMM_BI_NT_BIG_T(f16,  __half,        from_f_f16)

// ============================================================================
// Stage 5: tensor-core deterministic NN forward (bi_tensor_cores tier).
// ============================================================================
// mma.sync.aligned.m16n8k16 with f32 accumulators. SEPARATE numeric contract
// from the scalar triad (TC reduction tree, not the ascending-K FMA chain) —
// fully deterministic (fixed K order, fixed fragment/tile assignment, no
// atomics, no split-K) and batch-invariant across ALL M: each output
// element's entire K-reduction lives in one warp, independent of gridDim/M.
//
// v3 (cp.async iteration):
//   - As[m][k] (row-major) AND Bs[k][n] (row-major, global layout) are both
//     16B-chunk contiguous -> 2-stage cp.async pipeline with 4-operand
//     zero-fill for tails (bit-exact vs scalar zero stores). B fragments
//     come from ldmatrix.x2.TRANS of the k-major tile (delivers the
//     col-major k16n8 fragment without a staging transpose).
//   - Pads keep every ldmatrix row chunk in a distinct 4-bank group:
//     A row stride 40 halves (20 words ≡ 4 mod 8), B row stride 136 halves
//     (68 words ≡ 4 mod 8). Row bases are 16B-aligned (80 B / 272 B).
//   - Scalar staging fallback (uniform branch) when lda/ldb % 8 != 0.
//   - Smem: 2 x (10240 + 8704) B = 37.9 KB static.
//
// Geometry: CTA 256 threads = 8 warps as 2x4; BM=BN=128 BK=32; warp tile
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

#define TC_BM 128
#define TC_BN 128
#define TC_BK 32
#define TC_PAD_A 8
#define TC_PAD_B 8
#define TC_LDA (TC_BK + TC_PAD_A)
#define TC_LDB (TC_BN + TC_PAD_B)

// Issue one A+B tile into smem stage `buf` via 16B cp.async with zero-fill
// (fast path; requires lda%8==0 && ldb%8==0, checked by caller-side branch).
// A: 128 rows x 4 chunks; B: 32 rows x 16 chunks; 1024 cp.async / 256 thr.
#define SGB_TC_STAGE_ASYNC(buf, bkIdx)                                        \
    do {                                                                      \
        unsigned _as = As_sbase + (unsigned)((buf) * TC_BM * TC_LDA * 2);     \
        unsigned _bs = Bs_sbase + (unsigned)((buf) * TC_BK * TC_LDB * 2);     \
        for (int _i = threadIdx.x; _i < TC_BM * (TC_BK / 8); _i += 256) {     \
            int _m = _i / (TC_BK / 8);                                        \
            int _c = _i % (TC_BK / 8);                                        \
            int _k = _c * 8;                                                  \
            int _gr = pid_m * TC_BM + _m;                                     \
            int _gc = (bkIdx) + _k;                                           \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as + (unsigned)((_m * TC_LDA + _k) * 2);         \
            const void* _src = &A[(long long)_gr * lda + _gc];               \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < TC_BK * (TC_BN / 8); _i += 256) {     \
            int _k = _i / (TC_BN / 8);                                        \
            int _c = _i % (TC_BN / 8);                                        \
            int _n = _c * 8;                                                  \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * TC_BN + _n;                                     \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs + (unsigned)((_k * TC_LDB + _n) * 2);         \
            const void* _src = &B[(long long)_gk * ldb + _gn];               \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

// Scalar staging fallback for misaligned lda/ldb (rare; uniform branch).
#define SGB_TC_STAGE_SCALAR(buf, bkIdx, TT, FF)                               \
    do {                                                                      \
        TT* _Asw = &As[buf][0][0];                                            \
        TT* _Bsw = &Bs[buf][0][0];                                            \
        for (int _i = threadIdx.x; _i < TC_BM * TC_BK; _i += 256) {           \
            int _m = _i / TC_BK;                                              \
            int _k = _i % TC_BK;                                              \
            int _gr = pid_m * TC_BM + _m;                                     \
            int _gc = (bkIdx) + _k;                                           \
            _Asw[_m * TC_LDA + _k] = (_gr < M && _gc < K)                     \
                                         ? A[(long long)_gr * lda + _gc]      \
                                         : FF(0.0f);                          \
        }                                                                     \
        for (int _i = threadIdx.x; _i < TC_BK * TC_BN; _i += 256) {           \
            int _k = _i / TC_BN;                                              \
            int _n = _i % TC_BN;                                              \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * TC_BN + _n;                                     \
            _Bsw[_k * TC_LDB + _n] = (_gk < K && _gn < N)                     \
                                         ? B[(long long)_gk * ldb + _gn]      \
                                         : FF(0.0f);                          \
        }                                                                     \
    } while (0)

#define DEFINE_SGEMM_BI_NN_TC(SUFFIX, T_ACT, FROM_F, MMA_T)                    \
extern "C" __global__ __launch_bounds__(256, 1)                                \
void sgemm_bi_nn_tc_##SUFFIX(                                                  \
    T_ACT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    const float* __restrict__ bias,                                            \
    float alpha, float beta,                                                   \
    int M, int N, int K,                                                       \
    int lda, int ldb, int ldc                                                  \
) {                                                                            \
    assert(alpha == 1.0f || bias == nullptr);                                  \
    __shared__ __align__(16) T_ACT As[2][TC_BM][TC_LDA];                       \
    __shared__ __align__(16) T_ACT Bs[2][TC_BK][TC_LDB];                       \
    int num_pid_n = (N + TC_BN - 1) / TC_BN;                                   \
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
    bool fast_stage = ((lda & 7) == 0) && ((ldb & 7) == 0);                    \
    float acc[4][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 4; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            float b0 = 0.0f, b1 = 0.0f;                                        \
            if (bias != nullptr) {                                             \
                int c0 = pid_n * TC_BN + warpN + fn * 8 + 2 * t;               \
                b0 = (c0 < N) ? bias[c0] : 0.0f;                               \
                b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                       \
            }                                                                  \
            acc[fm][fn][0] = b0;                                               \
            acc[fm][fn][1] = b1;                                               \
            acc[fm][fn][2] = b0;                                               \
            acc[fm][fn][3] = b1;                                               \
        }                                                                      \
    }                                                                          \
    int num_k_tiles = (K + TC_BK - 1) / TC_BK;                                 \
    if (fast_stage) {                                                          \
        SGB_TC_STAGE_ASYNC(0, 0);                                              \
    } else {                                                                   \
        SGB_TC_STAGE_SCALAR(0, 0, T_ACT, FROM_F);                              \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (kt + 1 < num_k_tiles) {                                            \
            if (fast_stage) {                                                  \
                SGB_TC_STAGE_ASYNC(read_buf ^ 1, (kt + 1) * TC_BK);            \
            } else {                                                           \
                SGB_TC_STAGE_SCALAR(read_buf ^ 1, (kt + 1) * TC_BK, T_ACT,     \
                                    FROM_F);                                   \
            }                                                                  \
        }                                                                      \
        unsigned As_rd = As_sbase + (unsigned)(read_buf * TC_BM * TC_LDA * 2); \
        unsigned Bs_rd = Bs_sbase + (unsigned)(read_buf * TC_BK * TC_LDB * 2); \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < 2; ks++) {                                       \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[4][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 4; fm++) {                                   \
                int row = warpM + fm * 16 + lm_row_off + lm_r;                 \
                unsigned addr = As_rd +                                        \
                    (unsigned)((row * TC_LDA + k0 + lm_col_off) * 2);          \
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
                    (unsigned)((row * TC_LDB + warpN + fn * 8) * 2);           \
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
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 4; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            int r0 = pid_m * TC_BM + warpM + fm * 16 + g;                      \
            int c0 = pid_n * TC_BN + warpN + fn * 8 + 2 * t;                   \
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

DEFINE_SGEMM_BI_NN_TC(bf16, __nv_bfloat16, from_f_bf16, "bf16")
DEFINE_SGEMM_BI_NN_TC(f16,  __half,        from_f_f16,  "f16")
