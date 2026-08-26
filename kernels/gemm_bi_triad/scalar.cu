#define SGB_SCALAR_BM 128
#define SGB_SCALAR_BN 128
#define SGB_SCALAR_BK 16
#define SGB_SCALAR_WM 64
#define SGB_SCALAR_WN 32
#define SGB_SCALAR_WNITER 1
#define SGB_SCALAR_TM 8
#define SGB_SCALAR_TN 8
#define SGB_SCALAR_NUM_THREADS 256
#define SGB_SCALAR_WARP_SIZE 32
// SGB_GROUP_M is the L2-swizzle row-group size for the persistent-CTA tile walker.
// Trade-off: large groups maximize cross-CTA B-tile reuse in L2, small groups
// reduce the working set so it fits in a smaller L2.
//
// Host-side module compilation selects a per-architecture value and
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
// Pad=4 ensures column-stride-SGB_SCALAR_TM reads hit different banks instead of colliding.
// Applied across all 23 kernels in this file via (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) and
// (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) in allocation + indexing where Big/Slim tile layout fires.
// Reference: salykova/sgemm.cu (128×128×8 kernel uses ldm=132).
//
// SMEM cost delta:
//   Big  (SGB_SCALAR_BM=SGB_SCALAR_BN=128, SGB_SCALAR_BK=16): +4*16 + 4*16 = +128B per block  → 16384B → 16512B (under 48KB static)
//   Slim (SGB_SCALAR_BM=128 SGB_SCALAR_BN=64 SGB_SCALAR_BK=32): +4*32 + 4*32 = +256B per block → 24576B → 24832B (under 48KB static × 2 blocks)
#define SGB_SCALAR_SMEM_A_PAD 4
#define SGB_SCALAR_SMEM_B_PAD 4

// Derived constants
#define SGB_SCALAR_NUM_WARPS (SGB_SCALAR_NUM_THREADS / SGB_SCALAR_WARP_SIZE)  // 8
#define SGB_SCALAR_WMITER ((SGB_SCALAR_WM * SGB_SCALAR_WN) / (SGB_SCALAR_WARP_SIZE * SGB_SCALAR_TM * SGB_SCALAR_TN * SGB_SCALAR_WNITER))  // (64*32)/(32*8*8*1) = 2
#define SGB_SCALAR_WSUBM (SGB_SCALAR_WM / SGB_SCALAR_WMITER)   // 32
#define SGB_SCALAR_WSUBN (SGB_SCALAR_WN / SGB_SCALAR_WNITER)   // 32

// ============================================================================
// CUTLASS-style cache hints (replaces failed __ldcs experiment).
// ============================================================================
// ld.global.L2::128B prefetches next L2 line alongside .ca caching. sm_75+.
// Used by CUTLASS SGEMM mainloop B loads (cutlass/include/cutlass/arch/memory.h).
// Unlike .cs (evict-first), .ca+L2::128B keeps B resident across CTAs in the
// SGB_GROUP_M swizzle — exactly what our 16× cross-CTA reuse pattern wants.

// __stwt — streaming write. Output tile is write-once, never re-read by the
// same kernel. Marks L1 lines as evict-first so C writes don't evict A staging
// or B working set. Safe only for OVERWRITE epilogues (NT backward dX), NOT
// for NN forward (bias accumulate) or SGB_SCALAR_TN backward dW (grad accumulate).

// A load: float4, each thread loads 4 floats along K
// innerRowA = tid / (SGB_SCALAR_BK/4) = tid / 4, range 0..31
// innerColA = tid % (SGB_SCALAR_BK/4) = tid % 4, range 0..3
// rowStrideA = (SGB_SCALAR_NUM_THREADS * 4) / SGB_SCALAR_BK = (128*4)/16 = 32
// Loop: 4 iterations to cover SGB_SCALAR_BM=128 rows (stride 32)
#define SGB_SCALAR_ROW_STRIDE_A ((SGB_SCALAR_NUM_THREADS * 4) / SGB_SCALAR_BK)  // 32

// B load: float4, each thread loads 4 floats along N
// innerRowB = tid / (SGB_SCALAR_BN/4) = tid / 32, range 0..3
// innerColB = tid % (SGB_SCALAR_BN/4) = tid % 32, range 0..31
// rowStrideB = SGB_SCALAR_NUM_THREADS / (SGB_SCALAR_BN/4) = 128/32 = 4
// Loop: 4 iterations to cover SGB_SCALAR_BK=16 rows (stride 4)
#define SGB_SCALAR_ROW_STRIDE_B (SGB_SCALAR_NUM_THREADS / (SGB_SCALAR_BN / 4))  // 4

// ============================================================================
// Forward: C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C + bias
// ============================================================================
// __launch_bounds__(128, 2) — target 2 blocks/SM on Ada sm_89.
// With 168 regs × 128 threads × 2 blocks = 43008 regs (< 64K/SM) ✓
// Smem 16KB × 2 = 32KB (< 48KB static) ✓
// ptxas audit  confirmed 0 spill stores/loads.
extern "C" __global__ __launch_bounds__(SGB_SCALAR_NUM_THREADS, 2)
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
    // 2-stage cp.async pipeline (CUTLASS multistage SM80 pattern).
    // Dynamic smem layout: As[K_PIPE][SGB_SCALAR_BK*(SGB_SCALAR_BM+SGB_SCALAR_SMEM_A_PAD)] || Bs[K_PIPE][SGB_SCALAR_BK*(SGB_SCALAR_BN+SGB_SCALAR_SMEM_B_PAD)].
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
    constexpr int A_STAGE = SGB_SCALAR_BK * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD);  // 16 * 132 = 2112 floats
    constexpr int B_STAGE = SGB_SCALAR_BK * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD);  // 16 * 132 = 2112 floats
    float* As_buf = smem;                            // [K_PIPE * A_STAGE]
    float* Bs_buf = smem + K_PIPE * A_STAGE;         // [K_PIPE * B_STAGE]

    // SGB_GROUP_M L2 swizzle — count tiles once.
    int num_pid_m = (M + SGB_SCALAR_BM - 1) / SGB_SCALAR_BM;
    int num_pid_n = (N + SGB_SCALAR_BN - 1) / SGB_SCALAR_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    // Warp and thread placement (constant across all tiles a CTA processes).
    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int warpRow = warpIdx / (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);
    int threadRowInWarp = tidInWarp / (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);

    int innerRowA = threadIdx.x / (SGB_SCALAR_BK / 4);
    int innerColA = threadIdx.x % (SGB_SCALAR_BK / 4);
    int innerRowB = threadIdx.x / (SGB_SCALAR_BN / 4);
    int innerColB = threadIdx.x % (SGB_SCALAR_BN / 4);

    // Working-set registers (reset by mainloop per tile).
    float regM[SGB_SCALAR_WMITER * SGB_SCALAR_TM] = {0.0f};
    float regN[SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    // persistent CTA loop. Grid_dim ≤ total_tiles; each
    // block walks `tile_id = blockIdx.x, blockIdx.x + gridDim.x, ...`.
    // Determinism preserved: per-tile body is paste-identical to single-tile;
    // tile→block assignment is monotonic stride → tile_id → C[pid_m,pid_n]
    // mapping deterministic; output tiles non-overlapping (no atomics).
    // CUDA Graph compat: grid_dim captured once, stays fixed across replays.
    // Portability: invariant across batch_size / hyperparams / GPU SM count
    // (host passes `grid_dim = min(total_tiles, N_SMs * blocks_per_SM)`).
    // Direct CTA mapping avoids the persistent-loop overhead. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring lower register
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
        float threadResults[SGB_SCALAR_WMITER * SGB_SCALAR_TM * SGB_SCALAR_WNITER * SGB_SCALAR_TN];

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
            for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
                #pragma unroll
                for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; ++resIdxN) {
                    int g_col = pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN +
                                threadColInWarp * SGB_SCALAR_TN + resIdxN;
                    float b_val = (g_col < N) ? bias[g_col] : 0.0f;
                    #pragma unroll
                    for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
                        #pragma unroll
                        for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) +
                                      wSubColIdx * SGB_SCALAR_TN + resIdxN;
                            threadResults[idx] = b_val;
                        }
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < SGB_SCALAR_WMITER * SGB_SCALAR_TM * SGB_SCALAR_WNITER * SGB_SCALAR_TN; ++i) {
                threadResults[i] = 0.0f;
            }
        }

        // C output pointer for this tile.
        float* C_warp = C + (pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM) * ldc + pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN;

    // Macro-style helper: issue one full A+B tile into the given stage.
    // bkIdx is the K-offset of the tile being loaded. Uses 4-operand cp.async
    // for OOB zero-fill — branch-free, bit-exact, no scalar-store hole.
    #define ISSUE_TILE(stage, bkIdx) do {                                                 \
        /* 32 lanes split as 2 M-rows ×       \
         * SGB_SCALAR_BK=16 K-cols, 1 lane = 1 float (4B cp.async). 2 cache lines/warp instr,        \
         * 50% util vs prior 12.5% (8 rows × 16B/row scatter). Same cp.async count,       \
         * same dest As[K][M] layout → bit-exact preserved. 16B cp.async blocked by       \
         * dest stride; full coalesce requires layout swap (deferred).                    \
         */                                                                               \
        {                                                                                 \
            constexpr int WARPS_NN = SGB_SCALAR_NUM_THREADS / SGB_SCALAR_WARP_SIZE;                              \
            constexpr int M_ROWS_PER_WARP_INST_NN = SGB_SCALAR_WARP_SIZE / SGB_SCALAR_BK;                        \
            constexpr int M_ROWS_PER_WARP_NN = SGB_SCALAR_BM / WARPS_NN;                             \
            constexpr int INSTR_PER_WARP_NN =                                             \
                M_ROWS_PER_WARP_NN / M_ROWS_PER_WARP_INST_NN;                             \
            static_assert(SGB_SCALAR_WARP_SIZE % SGB_SCALAR_BK == 0,                                             \
                "SGB_SCALAR_WARP_SIZE must be divisible by SGB_SCALAR_BK for NN A coalesce");                    \
            static_assert(SGB_SCALAR_BM % WARPS_NN == 0,                                             \
                "SGB_SCALAR_BM must be divisible by warp count for NN A coalesce");                  \
            int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;                                           \
            int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;                                           \
            int _m_in_warp = _lane / SGB_SCALAR_BK;                                                  \
            int _k_local = _lane % SGB_SCALAR_BK;                                                    \
            _Pragma("unroll")                                                             \
            for (int _it = 0; _it < INSTR_PER_WARP_NN; _it++) {                           \
                int _m_local = _warp * M_ROWS_PER_WARP_NN                                 \
                               + _it * M_ROWS_PER_WARP_INST_NN + _m_in_warp;              \
                int _g_row = pid_m * SGB_SCALAR_BM + _m_local;                                       \
                int _g_col = (bkIdx) + _k_local;                                          \
                unsigned _dst = As_base + ((stage) * A_STAGE                              \
                    + _k_local * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local)                            \
                    * (unsigned)sizeof(float);                                            \
                int _bytes = (_g_row < M && _g_col < K) ? 4 : 0;                          \
                const float* _src = A + (long long)_g_row * lda + _g_col;                 \
                asm volatile(                                                             \
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"                      \
                    :: "r"(_dst), "l"(_src), "r"(_bytes));                                \
            }                                                                             \
        }                                                                                 \
        for (int _off = 0; _off + SGB_SCALAR_ROW_STRIDE_B <= SGB_SCALAR_BK; _off += SGB_SCALAR_ROW_STRIDE_B) {             \
            int _g_row = (bkIdx) + innerRowB + _off;                                      \
            int _g_col = pid_n * SGB_SCALAR_BN + innerColB * 4;                                      \
            unsigned _dst = Bs_base + ((stage) * B_STAGE                                  \
                + (innerRowB + _off) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD)                                  \
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
    int num_k_tiles = (K + SGB_SCALAR_BK - 1) / SGB_SCALAR_BK;
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
            int next_bkIdx = next_tile * SGB_SCALAR_BK;
            ISSUE_TILE(write_stage, next_bkIdx);
        }

        // Compute on read_stage.
        float* As_rd = As_buf + read_stage * A_STAGE;
        float* Bs_rd = Bs_buf + read_stage * B_STAGE;
        // Register fragment double-buffer (salykova/siboehm canonical).
        // Prefetch dotIdx+1 into regM_next/regN_next while FMAs consume
        // regM/regN_curr. FMA order IDENTICAL to single-buffer → bit-exact.
        // Hides smem→reg latency (~20 cycles) behind FMAs (~256 cycles/iter).
        float regM_next[SGB_SCALAR_WMITER * SGB_SCALAR_TM];
        float regN_next[SGB_SCALAR_WNITER * SGB_SCALAR_TN];
        // Prime fragment 0.
        #pragma unroll
        for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
            #pragma unroll
            for (int i = 0; i < SGB_SCALAR_TM; ++i)
                regM[wSubRowIdx * SGB_SCALAR_TM + i] =
                    As_rd[0 * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM +
                          threadRowInWarp * SGB_SCALAR_TM + i];
        #pragma unroll
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
            #pragma unroll
            for (int i = 0; i < SGB_SCALAR_TN; ++i)
                regN[wSubColIdx * SGB_SCALAR_TN + i] =
                    Bs_rd[0 * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN +
                          threadColInWarp * SGB_SCALAR_TN + i];

        for (int dotIdx = 0; dotIdx < SGB_SCALAR_BK; ++dotIdx) {
            // Prefetch fragment dotIdx+1 into *_next while we FMA on *_curr.
            if (dotIdx + 1 < SGB_SCALAR_BK) {
                #pragma unroll
                for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
                    #pragma unroll
                    for (int i = 0; i < SGB_SCALAR_TM; ++i)
                        regM_next[wSubRowIdx * SGB_SCALAR_TM + i] =
                            As_rd[(dotIdx + 1) * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM +
                                  wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + i];
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int i = 0; i < SGB_SCALAR_TN; ++i)
                        regN_next[wSubColIdx * SGB_SCALAR_TN + i] =
                            Bs_rd[(dotIdx + 1) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN +
                                  wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + i];
            }
            // FMAs on current fragment — IDENTICAL order to single-buffer.
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
                    #pragma unroll
                    for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                        #pragma unroll
                        for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; ++resIdxN) {
                            // explicit __fmaf_rn for bit-exact
                            // match with CPU `_mm256_fmadd_ps`.
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN)
                                      + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * SGB_SCALAR_TM + resIdxM],
                                regN[wSubColIdx * SGB_SCALAR_TN + resIdxN],
                                threadResults[idx]);
                        }
                    }
                }
            }
            // Swap: next → curr for the next dotIdx.
            if (dotIdx + 1 < SGB_SCALAR_BK) {
                #pragma unroll
                for (int i = 0; i < SGB_SCALAR_WMITER * SGB_SCALAR_TM; ++i) regM[i] = regM_next[i];
                #pragma unroll
                for (int i = 0; i < SGB_SCALAR_WNITER * SGB_SCALAR_TN; ++i) regN[i] = regN_next[i];
            }
        }
        // Rotate stages.
        read_stage = (read_stage + 1) % K_PIPE;
        write_stage = (write_stage + 1) % K_PIPE;
    }
    #undef ISSUE_TILE

    // Epilogue: write results with alpha, beta, bias (float4 stores)
    for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
            float* C_sub = C_warp + wSubRowIdx * SGB_SCALAR_WSUBM * ldc + wSubColIdx * SGB_SCALAR_WSUBN;
            for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                int g_row = pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM +
                            threadRowInWarp * SGB_SCALAR_TM + resIdxM;
                if (g_row >= M) continue;
                for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; resIdxN += 4) {
                    int g_col = pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN +
                                threadColInWarp * SGB_SCALAR_TN + resIdxN;
                    // Fallback to scalar on column tail OR when ldc % 4 != 0.
                    // STG.128 needs 16-byte aligned address — row-stride in bytes
                    // (ldc * 4) must be a multiple of 16, so ldc must be a multiple
                    // of 4. Otherwise odd rows hit CUDA_ERROR_MISALIGNED_ADDRESS.
                    if (g_col + 3 >= N || (ldc & 3) != 0) {
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) +
                                      wSubColIdx * SGB_SCALAR_TN + resIdxN + j;
                            // Bias seeded at K=0 (see init above); drop
                            // separate `+ bias` from epilog.
                            float val = alpha * threadResults[idx];
                            if (beta != 0.0f) val += beta * C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * ldc + threadColInWarp * SGB_SCALAR_TN + resIdxN + j];
                            C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * ldc + threadColInWarp * SGB_SCALAR_TN + resIdxN + j] = val;
                        }
                        continue;
                    }
                    float4 tmp;
                    if (beta != 0.0f) {
                        tmp = reinterpret_cast<float4*>(
                            &C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * ldc +
                                   threadColInWarp * SGB_SCALAR_TN + resIdxN])[0];
                    }
                    int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) +
                              wSubColIdx * SGB_SCALAR_TN + resIdxN;
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
                        &C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * ldc +
                               threadColInWarp * SGB_SCALAR_TN + resIdxN])[0] = out;
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
// Backward dW (SGB_SCALAR_TN): C[K,N] += alpha * A^T[K,M] @ B[M,N]
// ============================================================================
// A = X_saved [M, K] — read transposed
// B = dY [M, N]
// C = dW [K, N] — accumulated
// Output tile [SGB_SCALAR_BM, SGB_SCALAR_BN] over (K, N). M is reduction axis.
// __launch_bounds__(128, 2) — target 2 blocks/SM (ptxas: 167 regs, 0 spill).
extern "C" __global__ __launch_bounds__(SGB_SCALAR_NUM_THREADS, 2)
void sgemm_bi_tn(
    float* __restrict__ C,
    const float* __restrict__ A,  // X [M, K_out]
    const float* __restrict__ B,  // dY [M, N]
    float alpha,
    int M_red, int K_out, int N
) {
    // 2-stage cp.async pipeline (CUTLASS multistage SM80).
    constexpr int K_PIPE = 2;
    extern __shared__ __align__(16) float smem[];
    constexpr int A_STAGE = SGB_SCALAR_BK * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD);
    constexpr int B_STAGE = SGB_SCALAR_BK * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD);
    float* As_buf = smem;
    float* Bs_buf = smem + K_PIPE * A_STAGE;

    int num_pid_m = (K_out + SGB_SCALAR_BM - 1) / SGB_SCALAR_BM;
    int num_pid_n = (N + SGB_SCALAR_BN - 1) / SGB_SCALAR_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int warpRow = warpIdx / (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);
    int threadRowInWarp = tidInWarp / (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);

    int innerRowA = threadIdx.x / (SGB_SCALAR_BK / 4);
    int innerColA = threadIdx.x % (SGB_SCALAR_BK / 4);
    int innerRowB = threadIdx.x / (SGB_SCALAR_BN / 4);
    int innerColB = threadIdx.x % (SGB_SCALAR_BN / 4);

    float regM[SGB_SCALAR_WMITER * SGB_SCALAR_TM] = {0.0f};
    float regN[SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    // persistent CTA loop (see sgemm_bi_nn for rationale).
    // Direct CTA mapping avoids the persistent-loop overhead. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring lower register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[SGB_SCALAR_WMITER * SGB_SCALAR_TM * SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        float* C_warp = C + (pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM) * N + pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN;

    // ISSUE_TILE_TN(stage, mIdx) — issue one A+B tile for M-reduction position mIdx.
    // 4-operand cp.async zero-fills OOB bytes (PTX 9.7.8.22) — branch-free + bit-exact.
    //
    // A-load now uses warp-cooperative
    // contiguous-row reads — each warp loads `SGB_SCALAR_BK / (SGB_SCALAR_NUM_THREADS / SGB_SCALAR_WARP_SIZE) = 2`
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
            constexpr int WARPS_TN = SGB_SCALAR_NUM_THREADS / SGB_SCALAR_WARP_SIZE;                              \
            constexpr int ROWS_PER_WARP_TN = SGB_SCALAR_BK / WARPS_TN;                               \
            static_assert(SGB_SCALAR_BK % WARPS_TN == 0,                                             \
                "SGB_SCALAR_BK must be divisible by warp count for coalesced A-load");               \
            static_assert(SGB_SCALAR_BM % (SGB_SCALAR_WARP_SIZE * 4) == 0,                                       \
                "SGB_SCALAR_BM must be divisible by 32 lanes * 4 floats for 16B coalesce");          \
            int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;                                           \
            int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;                                           \
            _Pragma("unroll")                                                             \
            for (int _r = 0; _r < ROWS_PER_WARP_TN; _r++) {                               \
                int _k_local = _warp * ROWS_PER_WARP_TN + _r;                             \
                int _m_local = _lane * 4;                                                 \
                int _g_m = (mIdx) + _k_local;                                             \
                int _g_k = pid_m * SGB_SCALAR_BM + _m_local;                                         \
                unsigned _dst = As_base + ((stage) * A_STAGE                              \
                    + _k_local * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local)                            \
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
        for (int _off = 0; _off + SGB_SCALAR_ROW_STRIDE_B <= SGB_SCALAR_BK; _off += SGB_SCALAR_ROW_STRIDE_B) {             \
            int _g_m = (mIdx) + innerRowB + _off;                                         \
            int _g_n = pid_n * SGB_SCALAR_BN + innerColB * 4;                                        \
            unsigned _dst = Bs_base + ((stage) * B_STAGE                                  \
                + (innerRowB + _off) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4)                 \
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

    int num_k_tiles = (M_red + SGB_SCALAR_BK - 1) / SGB_SCALAR_BK;
    ISSUE_TILE_TN(0, 0);

    int read_stage = 0;
    int write_stage = 1;
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(K_PIPE - 2));
        __syncthreads();

        int next_tile = tile + 1;
        if (next_tile < num_k_tiles) {
            int next_mIdx = next_tile * SGB_SCALAR_BK;
            ISSUE_TILE_TN(write_stage, next_mIdx);
        }

        float* As_rd = As_buf + read_stage * A_STAGE;
        float* Bs_rd = Bs_buf + read_stage * B_STAGE;
        // Register fragment double-buffer.
        float regM_next[SGB_SCALAR_WMITER * SGB_SCALAR_TM];
        float regN_next[SGB_SCALAR_WNITER * SGB_SCALAR_TN];
        #pragma unroll
        for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
            #pragma unroll
            for (int i = 0; i < SGB_SCALAR_TM; ++i)
                regM[wSubRowIdx * SGB_SCALAR_TM + i] = As_rd[0 * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + i];
        #pragma unroll
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
            #pragma unroll
            for (int i = 0; i < SGB_SCALAR_TN; ++i)
                regN[wSubColIdx * SGB_SCALAR_TN + i] = Bs_rd[0 * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + i];

        for (int dotIdx = 0; dotIdx < SGB_SCALAR_BK; ++dotIdx) {
            if (dotIdx + 1 < SGB_SCALAR_BK) {
                #pragma unroll
                for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
                    #pragma unroll
                    for (int i = 0; i < SGB_SCALAR_TM; ++i)
                        regM_next[wSubRowIdx * SGB_SCALAR_TM + i] = As_rd[(dotIdx + 1) * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + i];
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int i = 0; i < SGB_SCALAR_TN; ++i)
                        regN_next[wSubColIdx * SGB_SCALAR_TN + i] = Bs_rd[(dotIdx + 1) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + i];
            }
            // explicit __fmaf_rn for bit-exact
            // match with CPU `_mm256_fmadd_ps`. sgemm_bi_tn (SGB_SCALAR_TN GEMM, K-pipelined).
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM)
                        #pragma unroll
                        for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; ++resIdxN) {
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN)
                                      + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * SGB_SCALAR_TM + resIdxM],
                                regN[wSubColIdx * SGB_SCALAR_TN + resIdxN],
                                threadResults[idx]);
                        }
            if (dotIdx + 1 < SGB_SCALAR_BK) {
                #pragma unroll
                for (int i = 0; i < SGB_SCALAR_WMITER * SGB_SCALAR_TM; ++i) regM[i] = regM_next[i];
                #pragma unroll
                for (int i = 0; i < SGB_SCALAR_WNITER * SGB_SCALAR_TN; ++i) regN[i] = regN_next[i];
            }
        }
        read_stage = (read_stage + 1) % K_PIPE;
        write_stage = (write_stage + 1) % K_PIPE;
    }
    #undef ISSUE_TILE_TN

    // Epilogue: accumulate into dW
    for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
            float* C_sub = C_warp + wSubRowIdx * SGB_SCALAR_WSUBM * N + wSubColIdx * SGB_SCALAR_WSUBN;
            for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                int g_row = pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + resIdxM;
                if (g_row >= K_out) continue;
                for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; resIdxN += 4) {
                    int g_col = pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + resIdxN;
                    // Fallback to scalar on tail OR when N (row-stride) % 4 != 0
                    // (STG.128 / LDG.128 need 16-byte aligned address).
                    if (g_col + 3 >= N || (N & 3) != 0) {
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) + wSubColIdx * SGB_SCALAR_TN + resIdxN + j;
                            C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * N + threadColInWarp * SGB_SCALAR_TN + resIdxN + j] += alpha * threadResults[idx];
                        }
                        continue;
                    }
                    float4 old = reinterpret_cast<float4*>(&C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * N + threadColInWarp * SGB_SCALAR_TN + resIdxN])[0];
                    int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                    old.x += alpha * threadResults[idx + 0];
                    old.y += alpha * threadResults[idx + 1];
                    old.z += alpha * threadResults[idx + 2];
                    old.w += alpha * threadResults[idx + 3];
                    reinterpret_cast<float4*>(&C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * N + threadColInWarp * SGB_SCALAR_TN + resIdxN])[0] = old;
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop
}

// ============================================================================
// Split-M SGB_SCALAR_TN backward dW: per-chunk partial of X^T @ dY (CUTLASS parallel-split pattern).
// ============================================================================
// Paired with sgemm_bi_splitm_reduce for fixed-order tree sum across chunks.
// Grid: (K_tiles * N_tiles, 1, F)  where blockIdx.z = fc (chunk index).
// Each block reduces M_CHUNK samples starting at m_begin = fc * M_CHUNK.
// Writes partial[fc, pid_k_out_tile*SGB_SCALAR_BM+row, pid_n_tile*SGB_SCALAR_BN+col] = raw sum (no alpha).
//
// Invariants:
//   - Inside each chunk: SGB_SCALAR_BK-tiled accumulation IDENTICAL to sgemm_bi_tn → bit-exact
//     per-chunk partial.
//   - Each (fc, k, n) slot has exactly ONE writer → no atomics, no race.
//   - Last chunk may be short (M % M_CHUNK != 0) — handled by existing g_m<m_end
//     OOB mask in the cp.async loads.
//
// Gain: inflates grid by F× for Big SGB_SCALAR_TN shapes where K_tiles*N_tiles < 2*NUM_SMS.
// ============================================================================
extern "C" __global__ __launch_bounds__(SGB_SCALAR_NUM_THREADS, 2)
void sgemm_bi_tn_splitm_partial(
    float* __restrict__ partial,       // [F * K_out * N] — unique slot per block
    const float* __restrict__ A,       // X [M, K_out]
    const float* __restrict__ B,       // dY [M, N]
    int M_red, int K_out, int N,
    int M_CHUNK                         // chunk size (multiple of SGB_SCALAR_BK)
) {
    __shared__ float As[SGB_SCALAR_BK * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD)];
    __shared__ float Bs[SGB_SCALAR_BK * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD)];

    int num_pid_m = (K_out + SGB_SCALAR_BM - 1) / SGB_SCALAR_BM;
    int num_pid_n = (N + SGB_SCALAR_BN - 1) / SGB_SCALAR_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;
    int fc = blockIdx.z;
    int m_begin = fc * M_CHUNK;
    int m_end = min(m_begin + M_CHUNK, M_red);
    if (m_begin >= M_red) return;

    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int warpRow = warpIdx / (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);
    int threadRowInWarp = tidInWarp / (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);

    int innerRowA = threadIdx.x / (SGB_SCALAR_BK / 4);
    int innerColA = threadIdx.x % (SGB_SCALAR_BK / 4);
    int innerRowB = threadIdx.x / (SGB_SCALAR_BN / 4);
    int innerColB = threadIdx.x % (SGB_SCALAR_BN / 4);

    float regM[SGB_SCALAR_WMITER * SGB_SCALAR_TM] = {0.0f};
    float regN[SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    // coalesce (mirrors sgemm_bi_tn ISSUE_TILE_TN A-loader).
    constexpr int WARPS_SM = SGB_SCALAR_NUM_THREADS / SGB_SCALAR_WARP_SIZE;
    constexpr int ROWS_PER_WARP_SM = SGB_SCALAR_BK / WARPS_SM;
    static_assert(SGB_SCALAR_BK % WARPS_SM == 0, "SGB_SCALAR_BK must be divisible by warp count");
    static_assert(SGB_SCALAR_BM % (SGB_SCALAR_WARP_SIZE * 4) == 0, "SGB_SCALAR_BM must be divisible by 32*4 for 16B coalesce");
    int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;

    // persistent CTA loop. fc/m_begin/m_end stay kernel-scoped.
    // Direct CTA mapping avoids the persistent-loop overhead. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring lower register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[SGB_SCALAR_WMITER * SGB_SCALAR_TM * SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    for (int mIdx = m_begin; mIdx < m_end; mIdx += SGB_SCALAR_BK) {
        #pragma unroll
        for (int _r = 0; _r < ROWS_PER_WARP_SM; _r++) {
            int k_local = _warp * ROWS_PER_WARP_SM + _r;
            int m_local = _lane * 4;
            int g_m = mIdx + k_local;
            int g_k = pid_m * SGB_SCALAR_BM + m_local;
            unsigned dst = As_base + (k_local * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + m_local)
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
                        As[k_local * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + m_local + _i] = 0.0f;
                    }
                }
            }
        }
        for (int offset = 0; offset + SGB_SCALAR_ROW_STRIDE_B <= SGB_SCALAR_BK; offset += SGB_SCALAR_ROW_STRIDE_B) {
            int g_m = mIdx + innerRowB + offset;
            int g_n = pid_n * SGB_SCALAR_BN + innerColB * 4;
            unsigned dst = Bs_base + ((innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
            if (g_m < m_end && g_n + 3 < N && (N % 4 == 0)) {
                const float* src = B + ((long long)g_m) * N + pid_n * SGB_SCALAR_BN + innerColB * 4;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 0] = (g_m < m_end && g_n + 0 < N) ? B[g_m * N + pid_n * SGB_SCALAR_BN + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 1] = (g_m < m_end && g_n + 1 < N) ? B[g_m * N + pid_n * SGB_SCALAR_BN + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 2] = (g_m < m_end && g_n + 2 < N) ? B[g_m * N + pid_n * SGB_SCALAR_BN + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 3] = (g_m < m_end && g_n + 3 < N) ? B[g_m * N + pid_n * SGB_SCALAR_BN + innerColB * 4 + 3] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        for (int dotIdx = 0; dotIdx < SGB_SCALAR_BK; ++dotIdx) {
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
                for (int i = 0; i < SGB_SCALAR_TM; ++i)
                    regM[wSubRowIdx * SGB_SCALAR_TM + i] = As[dotIdx * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + i];
            for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
                for (int i = 0; i < SGB_SCALAR_TN; ++i)
                    regN[wSubColIdx * SGB_SCALAR_TN + i] = Bs[dotIdx * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + i];
            // explicit __fmaf_rn for bit-exact
            // match with CPU `_mm256_fmadd_ps`. This matches the sibling kernels
            // (RoPE backward FMA pin).
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
                    for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM)
                        for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; ++resIdxN) {
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN)
                                      + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * SGB_SCALAR_TM + resIdxM],
                                regN[wSubColIdx * SGB_SCALAR_TN + resIdxN],
                                threadResults[idx]);
                        }
        }
        __syncthreads();
    }

    // Epilogue: OVERWRITE partial[fc, :, :] slot. No alpha, no bias, no accumulate —
    // reducer applies alpha on the final sum.
    float* partial_chunk = partial + (long long)fc * K_out * N;
    float* partial_warp = partial_chunk + (pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM) * N + pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN;

    for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
            float* P_sub = partial_warp + wSubRowIdx * SGB_SCALAR_WSUBM * N + wSubColIdx * SGB_SCALAR_WSUBN;
            for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                int g_row = pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + resIdxM;
                if (g_row >= K_out) continue;
                for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; resIdxN += 4) {
                    int g_col = pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + resIdxN;
                    int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                    // Fallback to scalar on tail OR when N (row-stride) % 4 != 0
                    // (STG.128 requires 16-byte aligned address).
                    if (g_col + 3 >= N || (N & 3) != 0) {
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            P_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * N + threadColInWarp * SGB_SCALAR_TN + resIdxN + j] = threadResults[idx + j];
                        }
                        continue;
                    }
                    float4 out;
                    out.x = threadResults[idx + 0];
                    out.y = threadResults[idx + 1];
                    out.z = threadResults[idx + 2];
                    out.w = threadResults[idx + 3];
                    reinterpret_cast<float4*>(&P_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * N + threadColInWarp * SGB_SCALAR_TN + resIdxN])[0] = out;
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
// At SGB_SCALAR_TN splitm callsites (L900, L3166) `K_out` receives source-GEMM-M; at the
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
//   - K_chunk % SGB_SCALAR_BK == 0 (enforced by dispatcher; each fc's K-range is SGB_SCALAR_BK-aligned
//     except optionally the last fc which handles K-tail via existing cp.async
//     zero-fill pattern, identical to non-split Big NN's K%SGB_SCALAR_BK handling)
//   - No alpha / bias / beta in this kernel — raw tile sums only
//   - Dynamic smem 33 KB (same as Big NN): caller MUST cuFuncSetAttribute
//     MAX_DYNAMIC_SHARED_SIZE_BYTES on this CUfunction handle separately
//
// Pattern reference: CUTLASS GemmSplitKParallel (partial kernel overwrites
// unique per-fc slot, reducer fuses Σ ascending-fc in f64). Matches existing
// sgemm_bi_tn_splitm_partial (M-axis split) — this is the K-axis twin.
// ============================================================================
extern "C" __global__ __launch_bounds__(SGB_SCALAR_NUM_THREADS, 2)
void sgemm_bi_nn_splitk_big_partial(
    float* __restrict__ partial,       // [F * M * N] — unique slot per fc, row-major
    const float* __restrict__ A,       // [M, K_full]
    const float* __restrict__ B,       // [K_full, N]
    int M, int N, int K,
    int lda, int ldb,
    int K_chunk                         // must be multiple of SGB_SCALAR_BK
) {
    constexpr int K_PIPE = 2;
    extern __shared__ __align__(16) float smem[];
    constexpr int A_STAGE = SGB_SCALAR_BK * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD);
    constexpr int B_STAGE = SGB_SCALAR_BK * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD);
    float* As_buf = smem;
    float* Bs_buf = smem + K_PIPE * A_STAGE;

    // Decode fc from z-axis; early exit if fc is out of K-range.
    int fc = blockIdx.z;
    int k_begin = fc * K_chunk;
    if (k_begin >= K) return;
    int k_end = min(K, k_begin + K_chunk);

    // SGB_GROUP_M L2 swizzle (identical to Big NN)
    int num_pid_m = (M + SGB_SCALAR_BM - 1) / SGB_SCALAR_BM;
    int num_pid_n = (N + SGB_SCALAR_BN - 1) / SGB_SCALAR_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int warpRow = warpIdx / (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);
    int threadRowInWarp = tidInWarp / (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);

    int innerRowA = threadIdx.x / (SGB_SCALAR_BK / 4);
    int innerColA = threadIdx.x % (SGB_SCALAR_BK / 4);
    int innerRowB = threadIdx.x / (SGB_SCALAR_BN / 4);
    int innerColB = threadIdx.x % (SGB_SCALAR_BN / 4);

    float regM[SGB_SCALAR_WMITER * SGB_SCALAR_TM] = {0.0f};
    float regN[SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    // persistent CTA loop. fc/k_begin/k_end stay kernel-scoped.
    // Direct CTA mapping avoids the persistent-loop overhead. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring lower register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[SGB_SCALAR_WMITER * SGB_SCALAR_TM * SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    // ISSUE_TILE — same as Big NN but uses `K` as the global K-bound (for OOB
    // zero-fill). Per-block K range [k_begin, k_end) is enforced by the tile
    // loop; cp.async still zero-fills any lane that reads k >= K.
    #define ISSUE_TILE_SK(stage, bkIdx) do {                                              \
        /* coalesce (mirrors big NN ISSUE_TILE A-load). */             \
        {                                                                                 \
            constexpr int WARPS_SK = SGB_SCALAR_NUM_THREADS / SGB_SCALAR_WARP_SIZE;                              \
            constexpr int M_ROWS_PER_WARP_INST_SK = SGB_SCALAR_WARP_SIZE / SGB_SCALAR_BK;                        \
            constexpr int M_ROWS_PER_WARP_SK = SGB_SCALAR_BM / WARPS_SK;                             \
            constexpr int INSTR_PER_WARP_SK =                                             \
                M_ROWS_PER_WARP_SK / M_ROWS_PER_WARP_INST_SK;                             \
            static_assert(SGB_SCALAR_WARP_SIZE % SGB_SCALAR_BK == 0, "SGB_SCALAR_WARP_SIZE divisible by SGB_SCALAR_BK");                \
            static_assert(SGB_SCALAR_BM % WARPS_SK == 0, "SGB_SCALAR_BM divisible by warp count");              \
            int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;                                           \
            int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;                                           \
            int _m_in_warp = _lane / SGB_SCALAR_BK;                                                  \
            int _k_local = _lane % SGB_SCALAR_BK;                                                    \
            _Pragma("unroll")                                                             \
            for (int _it = 0; _it < INSTR_PER_WARP_SK; _it++) {                           \
                int _m_local = _warp * M_ROWS_PER_WARP_SK                                 \
                               + _it * M_ROWS_PER_WARP_INST_SK + _m_in_warp;              \
                int _g_row = pid_m * SGB_SCALAR_BM + _m_local;                                       \
                int _g_col = (bkIdx) + _k_local;                                          \
                unsigned _dst = As_base + ((stage) * A_STAGE                              \
                    + _k_local * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local)                            \
                    * (unsigned)sizeof(float);                                            \
                int _bytes = (_g_row < M && _g_col < K) ? 4 : 0;                          \
                const float* _src = A + (long long)_g_row * lda + _g_col;                 \
                asm volatile(                                                             \
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"                      \
                    :: "r"(_dst), "l"(_src), "r"(_bytes));                                \
            }                                                                             \
        }                                                                                 \
        for (int _off = 0; _off + SGB_SCALAR_ROW_STRIDE_B <= SGB_SCALAR_BK; _off += SGB_SCALAR_ROW_STRIDE_B) {             \
            int _g_row = (bkIdx) + innerRowB + _off;                                      \
            int _g_col = pid_n * SGB_SCALAR_BN + innerColB * 4;                                      \
            unsigned _dst = Bs_base + ((stage) * B_STAGE                                  \
                + (innerRowB + _off) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD)                                  \
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

    // Tile count for this fc's K-range. k_begin and K_chunk are SGB_SCALAR_BK-aligned,
    // so num_k_tiles = ceil((k_end - k_begin) / SGB_SCALAR_BK).
    int num_k_tiles = (k_end - k_begin + SGB_SCALAR_BK - 1) / SGB_SCALAR_BK;

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
            int next_bkIdx = k_begin + next_tile * SGB_SCALAR_BK;
            ISSUE_TILE_SK(write_stage, next_bkIdx);
        }

        float* As_rd = As_buf + read_stage * A_STAGE;
        float* Bs_rd = Bs_buf + read_stage * B_STAGE;
        // Register fragment double-buffer — identical FMA order to Big NN.
        float regM_next[SGB_SCALAR_WMITER * SGB_SCALAR_TM];
        float regN_next[SGB_SCALAR_WNITER * SGB_SCALAR_TN];
        #pragma unroll
        for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
            #pragma unroll
            for (int i = 0; i < SGB_SCALAR_TM; ++i)
                regM[wSubRowIdx * SGB_SCALAR_TM + i] =
                    As_rd[0 * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM +
                          threadRowInWarp * SGB_SCALAR_TM + i];
        #pragma unroll
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
            #pragma unroll
            for (int i = 0; i < SGB_SCALAR_TN; ++i)
                regN[wSubColIdx * SGB_SCALAR_TN + i] =
                    Bs_rd[0 * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN +
                          threadColInWarp * SGB_SCALAR_TN + i];

        for (int dotIdx = 0; dotIdx < SGB_SCALAR_BK; ++dotIdx) {
            if (dotIdx + 1 < SGB_SCALAR_BK) {
                #pragma unroll
                for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
                    #pragma unroll
                    for (int i = 0; i < SGB_SCALAR_TM; ++i)
                        regM_next[wSubRowIdx * SGB_SCALAR_TM + i] =
                            As_rd[(dotIdx + 1) * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM +
                                  wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + i];
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int i = 0; i < SGB_SCALAR_TN; ++i)
                        regN_next[wSubColIdx * SGB_SCALAR_TN + i] =
                            Bs_rd[(dotIdx + 1) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN +
                                  wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + i];
            }
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
                    #pragma unroll
                    for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                        #pragma unroll
                        for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; ++resIdxN) {
                            // explicit __fmaf_rn for bit-exact
                            // match with CPU `_mm256_fmadd_ps`.
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN)
                                      + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * SGB_SCALAR_TM + resIdxM],
                                regN[wSubColIdx * SGB_SCALAR_TN + resIdxN],
                                threadResults[idx]);
                        }
                    }
                }
            }
            if (dotIdx + 1 < SGB_SCALAR_BK) {
                #pragma unroll
                for (int i = 0; i < SGB_SCALAR_WMITER * SGB_SCALAR_TM; ++i) regM[i] = regM_next[i];
                #pragma unroll
                for (int i = 0; i < SGB_SCALAR_WNITER * SGB_SCALAR_TN; ++i) regN[i] = regN_next[i];
            }
        }
        read_stage = (read_stage + 1) % K_PIPE;
        write_stage = (write_stage + 1) % K_PIPE;
    }
    #undef ISSUE_TILE_SK

    // Epilogue: OVERWRITE partial[fc, :, :] — raw tile sums. NO alpha, NO bias,
    // NO beta. Reducer applies alpha and folds into C.
    float* partial_chunk = partial + (long long)fc * M * N;
    float* partial_warp = partial_chunk + (pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM) * N + pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN;

    for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
            float* P_sub = partial_warp + wSubRowIdx * SGB_SCALAR_WSUBM * N + wSubColIdx * SGB_SCALAR_WSUBN;
            for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                int g_row = pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM +
                            threadRowInWarp * SGB_SCALAR_TM + resIdxM;
                if (g_row >= M) continue;
                for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; resIdxN += 4) {
                    int g_col = pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN +
                                threadColInWarp * SGB_SCALAR_TN + resIdxN;
                    int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) +
                              wSubColIdx * SGB_SCALAR_TN + resIdxN;
                    // Scalar fallback for right-edge AND for non-%4 N
                    // (hardens audit finding M2 — reducer walks per-element, so
                    // partial slot layout must be exact row-stride-N regardless).
                    if (g_col + 3 >= N || (N % 4) != 0) {
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            P_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * N +
                                  threadColInWarp * SGB_SCALAR_TN + resIdxN + j] =
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
                        &P_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * N +
                               threadColInWarp * SGB_SCALAR_TN + resIdxN])[0] = out;
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
// __launch_bounds__(128, 2) — target 2 blocks/SM (ptxas: 168 regs, 0 spill).
extern "C" __global__ __launch_bounds__(SGB_SCALAR_NUM_THREADS, 2)
void sgemm_bi_nt(
    float* __restrict__ C,
    const float* __restrict__ A,  // dY [M, N]
    const float* __restrict__ B,  // W [K, N]
    float alpha,
    int M, int N, int K_out
) {
    // 2-stage cp.async pipeline.
    constexpr int K_PIPE = 2;
    extern __shared__ __align__(16) float smem[];
    constexpr int A_STAGE = SGB_SCALAR_BK * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD);
    constexpr int B_STAGE = SGB_SCALAR_BK * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD);
    float* As_buf = smem;
    float* Bs_buf = smem + K_PIPE * A_STAGE;

    int num_pid_m = (M + SGB_SCALAR_BM - 1) / SGB_SCALAR_BM;
    int num_pid_n = (K_out + SGB_SCALAR_BN - 1) / SGB_SCALAR_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int warpRow = warpIdx / (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);
    int threadRowInWarp = tidInWarp / (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);

    int innerRowA = threadIdx.x / (SGB_SCALAR_BK / 4);
    int innerColA = threadIdx.x % (SGB_SCALAR_BK / 4);
    int innerRowB = threadIdx.x / (SGB_SCALAR_BN / 4);
    int innerColB = threadIdx.x % (SGB_SCALAR_BN / 4);

    float regM[SGB_SCALAR_WMITER * SGB_SCALAR_TM] = {0.0f};
    float regN[SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    // persistent CTA loop (see sgemm_bi_nn for rationale).
    // Direct CTA mapping avoids the persistent-loop overhead. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring lower register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[SGB_SCALAR_WMITER * SGB_SCALAR_TM * SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        float* C_warp = C + (pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM) * K_out + pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN;

    // ISSUE_TILE_NT(stage, nIdx) — issue one A+B tile for N-reduction position nIdx.
    //
    // A-load now uses warp-cooperative
    // contiguous-row reads. NT tile geometry differs from SGB_SCALAR_TN: here SGB_SCALAR_BK=16 is the
    // N-axis (reduction dim) of dY[M,N] and SGB_SCALAR_BM=128 is the M-axis (samples).
    // dY[m, n] is row-major → for fixed m varying n is contiguous.
    //
    // Best coalesce we can get: 32 lanes split as 2 M-rows × SGB_SCALAR_BK=16 N-cols per row
    // (each lane reads 1 float, 4B cp.async). Per warp instruction: 2 cache lines,
    // each 64 B utilized of 128 B → 50% utilization. Replaces the prior pattern
    // (32 threads on 8 M-rows × 4 N-cols/row → 8 cache lines × 16 B util = 12.5%).
    // Net gain: **4× cache line utilization** at SAME 8 cp.async/thread count.
    //
    // We do NOT use 16B cp.async here — SGB_SCALAR_BK=16 < 32 lanes/row makes a full-warp
    // contiguous N-stripe impossible without layout swap.
    //
    // Bit-exact: destination As[n_local][m_local] cells receive identical bytes.
    // FMA loop unchanged.
    #define ISSUE_TILE_NT(stage, nIdx) do {                                               \
        {                                                                                 \
            constexpr int WARPS_NT = SGB_SCALAR_NUM_THREADS / SGB_SCALAR_WARP_SIZE;                              \
            constexpr int M_ROWS_PER_WARP_INST = SGB_SCALAR_WARP_SIZE / SGB_SCALAR_BK;                           \
            constexpr int M_ROWS_PER_WARP = SGB_SCALAR_BM / WARPS_NT;                                \
            constexpr int INSTR_PER_WARP_NT = M_ROWS_PER_WARP / M_ROWS_PER_WARP_INST;     \
            static_assert(SGB_SCALAR_WARP_SIZE % SGB_SCALAR_BK == 0,                                             \
                "SGB_SCALAR_WARP_SIZE must be divisible by SGB_SCALAR_BK for NT A coalesce");                    \
            static_assert(SGB_SCALAR_BM % WARPS_NT == 0,                                             \
                "SGB_SCALAR_BM must be divisible by warp count for NT A coalesce");                  \
            int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;                                           \
            int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;                                           \
            int _m_in_warp = _lane / SGB_SCALAR_BK;                                                  \
            int _n_local = _lane % SGB_SCALAR_BK;                                                    \
            _Pragma("unroll")                                                             \
            for (int _it = 0; _it < INSTR_PER_WARP_NT; _it++) {                           \
                int _m_local = _warp * M_ROWS_PER_WARP                                    \
                               + _it * M_ROWS_PER_WARP_INST + _m_in_warp;                 \
                int _g_m = pid_m * SGB_SCALAR_BM + _m_local;                                         \
                int _g_n = (nIdx) + _n_local;                                             \
                unsigned _dst = As_base + ((stage) * A_STAGE                              \
                    + _n_local * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local)                            \
                    * (unsigned)sizeof(float);                                            \
                int _bytes = (_g_m < M && _g_n < N) ? 4 : 0;                              \
                const float* _src = A + (long long)_g_m * N + _g_n;                       \
                asm volatile(                                                             \
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"                      \
                    :: "r"(_dst), "l"(_src), "r"(_bytes));                                \
            }                                                                             \
        }                                                                                 \
        for (int _off = 0; _off + SGB_SCALAR_ROW_STRIDE_B <= SGB_SCALAR_BK; _off += SGB_SCALAR_ROW_STRIDE_B) {             \
            int _n_local = innerRowB + _off;                                              \
            int _k_base = innerColB * 4;                                                  \
            int _g_n = (nIdx) + _n_local;                                                 \
            int _g_k = pid_n * SGB_SCALAR_BN + _k_base;                                              \
            bool _pn = _g_n < N;                                                          \
            _Pragma("unroll")                                                             \
            for (int _i = 0; _i < 4; _i++) {                                              \
                unsigned _dst = Bs_base + ((stage) * B_STAGE                              \
                    + _n_local * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + _k_base + _i)                        \
                    * (unsigned)sizeof(float);                                            \
                int _bytes = (_pn && (_g_k + _i) < K_out) ? 4 : 0;                        \
                const float* _src = B + (long long)(_g_k + _i) * N + _g_n;                \
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"             \
                             :: "r"(_dst), "l"(_src), "r"(_bytes));                       \
            }                                                                             \
        }                                                                                 \
        asm volatile("cp.async.commit_group;\n");                                         \
    } while (0)

    int num_k_tiles = (N + SGB_SCALAR_BK - 1) / SGB_SCALAR_BK;
    ISSUE_TILE_NT(0, 0);

    int read_stage = 0;
    int write_stage = 1;
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(K_PIPE - 2));
        __syncthreads();

        int next_tile = tile + 1;
        if (next_tile < num_k_tiles) {
            int next_nIdx = next_tile * SGB_SCALAR_BK;
            ISSUE_TILE_NT(write_stage, next_nIdx);
        }

        float* As_rd = As_buf + read_stage * A_STAGE;
        float* Bs_rd = Bs_buf + read_stage * B_STAGE;
        // Register fragment double-buffer.
        float regM_next[SGB_SCALAR_WMITER * SGB_SCALAR_TM];
        float regN_next[SGB_SCALAR_WNITER * SGB_SCALAR_TN];
        #pragma unroll
        for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
            #pragma unroll
            for (int i = 0; i < SGB_SCALAR_TM; ++i)
                regM[wSubRowIdx * SGB_SCALAR_TM + i] = As_rd[0 * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + i];
        #pragma unroll
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
            #pragma unroll
            for (int i = 0; i < SGB_SCALAR_TN; ++i)
                regN[wSubColIdx * SGB_SCALAR_TN + i] = Bs_rd[0 * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + i];

        for (int dotIdx = 0; dotIdx < SGB_SCALAR_BK; ++dotIdx) {
            if (dotIdx + 1 < SGB_SCALAR_BK) {
                #pragma unroll
                for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
                    #pragma unroll
                    for (int i = 0; i < SGB_SCALAR_TM; ++i)
                        regM_next[wSubRowIdx * SGB_SCALAR_TM + i] = As_rd[(dotIdx + 1) * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + i];
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int i = 0; i < SGB_SCALAR_TN; ++i)
                        regN_next[wSubColIdx * SGB_SCALAR_TN + i] = Bs_rd[(dotIdx + 1) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + i];
            }
            // explicit __fmaf_rn for bit-exact
            // match with CPU `_mm256_fmadd_ps`. sgemm_bi_nt (NT GEMM, K-pipelined).
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM)
                        #pragma unroll
                        for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; ++resIdxN) {
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN)
                                      + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * SGB_SCALAR_TM + resIdxM],
                                regN[wSubColIdx * SGB_SCALAR_TN + resIdxN],
                                threadResults[idx]);
                        }
            if (dotIdx + 1 < SGB_SCALAR_BK) {
                #pragma unroll
                for (int i = 0; i < SGB_SCALAR_WMITER * SGB_SCALAR_TM; ++i) regM[i] = regM_next[i];
                #pragma unroll
                for (int i = 0; i < SGB_SCALAR_WNITER * SGB_SCALAR_TN; ++i) regN[i] = regN_next[i];
            }
        }
        read_stage = (read_stage + 1) % K_PIPE;
        write_stage = (write_stage + 1) % K_PIPE;
    }
    #undef ISSUE_TILE_NT

    // Epilogue: overwrite dX
    for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
            float* C_sub = C_warp + wSubRowIdx * SGB_SCALAR_WSUBM * K_out + wSubColIdx * SGB_SCALAR_WSUBN;
            for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                int g_row = pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + resIdxM;
                if (g_row >= M) continue;
                for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; resIdxN += 4) {
                    int g_col = pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + resIdxN;
                    // float4 write only when K_out is %4-aligned (K_out=257 → scalar).
                    if (g_col + 3 >= K_out || (K_out % 4 != 0)) {
                        int idx_base = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                        for (int j = 0; j < 4 && g_col + j < K_out; j++) {
                            __stwt(&C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * K_out + threadColInWarp * SGB_SCALAR_TN + resIdxN + j], alpha * threadResults[idx_base + j]);
                        }
                        continue;
                    }
                    int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                    float4 out = {
                        alpha * threadResults[idx + 0],
                        alpha * threadResults[idx + 1],
                        alpha * threadResults[idx + 2],
                        alpha * threadResults[idx + 3]
                    };
                    // __stwt — streaming store. NT backward dX is OVERWRITE (no accumulation),
                    // so marking C lines evict-first is safe and prevents C writes from evicting A staging / B working set.
                    __stwt(reinterpret_cast<float4*>(&C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * K_out + threadColInWarp * SGB_SCALAR_TN + resIdxN]), out);
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
//   - N_chunk % SGB_SCALAR_BK == 0 (enforced by dispatcher; tail fc handles via cp.async zero-fill)
//   - No alpha — reducer applies it
//   - Dynamic smem 33 KB; caller must cuFuncSetAttribute on this CUfunction handle
//
// Mirrors sgemm_bi_nn_splitk_big_partial (K-axis twin) for dW bwd_dx path.
// ============================================================================
extern "C" __global__ __launch_bounds__(SGB_SCALAR_NUM_THREADS, 2)
void sgemm_bi_nt_splitn_big_partial(
    float* __restrict__ partial,       // [F * M * K_out] — unique slot per fc
    const float* __restrict__ A,       // dY [M, N]
    const float* __restrict__ B,       // W  [K_out, N]
    int M, int N, int K_out,
    int N_chunk                         // must be multiple of SGB_SCALAR_BK
) {
    constexpr int K_PIPE = 2;
    extern __shared__ __align__(16) float smem[];
    constexpr int A_STAGE = SGB_SCALAR_BK * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD);
    constexpr int B_STAGE = SGB_SCALAR_BK * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD);
    float* As_buf = smem;
    float* Bs_buf = smem + K_PIPE * A_STAGE;

    int fc = blockIdx.z;
    int n_begin = fc * N_chunk;
    if (n_begin >= N) return;
    int n_end = min(N, n_begin + N_chunk);

    // SGB_GROUP_M L2 swizzle (identical to Big NT: pid_m on M, pid_n on K_out)
    int num_pid_m = (M + SGB_SCALAR_BM - 1) / SGB_SCALAR_BM;
    int num_pid_n = (K_out + SGB_SCALAR_BN - 1) / SGB_SCALAR_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int warpRow = warpIdx / (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);
    int threadRowInWarp = tidInWarp / (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);

    int innerRowA = threadIdx.x / (SGB_SCALAR_BK / 4);
    int innerColA = threadIdx.x % (SGB_SCALAR_BK / 4);
    int innerRowB = threadIdx.x / (SGB_SCALAR_BN / 4);
    int innerColB = threadIdx.x % (SGB_SCALAR_BN / 4);

    float regM[SGB_SCALAR_WMITER * SGB_SCALAR_TM] = {0.0f};
    float regN[SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

    // Allocate As/Bs base pointers once (needed by macro below).

    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    // ISSUE_TILE_NT_SN(stage, nIdx) — identical to Big NT's ISSUE_TILE_NT,
    // but N-bound is `N` (global) for cp.async zero-fill. Per-block N range
    // [n_begin, n_end) is enforced by tile loop.
    #define ISSUE_TILE_NT_SN(stage, nIdx) do {                                            \
        /* coalesce (mirrors big NT ISSUE_TILE_NT A-load). */          \
        {                                                                                 \
            constexpr int WARPS_NTSN = SGB_SCALAR_NUM_THREADS / SGB_SCALAR_WARP_SIZE;                            \
            constexpr int M_ROWS_PER_WARP_INST_NTSN = SGB_SCALAR_WARP_SIZE / SGB_SCALAR_BK;                      \
            constexpr int M_ROWS_PER_WARP_NTSN = SGB_SCALAR_BM / WARPS_NTSN;                         \
            constexpr int INSTR_PER_WARP_NTSN =                                           \
                M_ROWS_PER_WARP_NTSN / M_ROWS_PER_WARP_INST_NTSN;                         \
            static_assert(SGB_SCALAR_WARP_SIZE % SGB_SCALAR_BK == 0, "SGB_SCALAR_WARP_SIZE divisible by SGB_SCALAR_BK");                \
            static_assert(SGB_SCALAR_BM % WARPS_NTSN == 0, "SGB_SCALAR_BM divisible by warp count");            \
            int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;                                           \
            int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;                                           \
            int _m_in_warp = _lane / SGB_SCALAR_BK;                                                  \
            int _n_local = _lane % SGB_SCALAR_BK;                                                    \
            _Pragma("unroll")                                                             \
            for (int _it = 0; _it < INSTR_PER_WARP_NTSN; _it++) {                         \
                int _m_local = _warp * M_ROWS_PER_WARP_NTSN                               \
                               + _it * M_ROWS_PER_WARP_INST_NTSN + _m_in_warp;            \
                int _g_m = pid_m * SGB_SCALAR_BM + _m_local;                                         \
                int _g_n = (nIdx) + _n_local;                                             \
                unsigned _dst = As_base + ((stage) * A_STAGE                              \
                    + _n_local * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local)                            \
                    * (unsigned)sizeof(float);                                            \
                int _bytes = (_g_m < M && _g_n < N) ? 4 : 0;                              \
                const float* _src = A + (long long)_g_m * N + _g_n;                       \
                asm volatile(                                                             \
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"                      \
                    :: "r"(_dst), "l"(_src), "r"(_bytes));                                \
            }                                                                             \
        }                                                                                 \
        for (int _off = 0; _off + SGB_SCALAR_ROW_STRIDE_B <= SGB_SCALAR_BK; _off += SGB_SCALAR_ROW_STRIDE_B) {             \
            int _n_local = innerRowB + _off;                                              \
            int _k_base = innerColB * 4;                                                  \
            int _g_n = (nIdx) + _n_local;                                                 \
            int _g_k = pid_n * SGB_SCALAR_BN + _k_base;                                              \
            bool _pn = _g_n < N;                                                          \
            _Pragma("unroll")                                                             \
            for (int _i = 0; _i < 4; _i++) {                                              \
                unsigned _dst = Bs_base + ((stage) * B_STAGE                              \
                    + _n_local * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + _k_base + _i)                        \
                    * (unsigned)sizeof(float);                                            \
                int _bytes = (_pn && (_g_k + _i) < K_out) ? 4 : 0;                        \
                const float* _src = B + (long long)(_g_k + _i) * N + _g_n;                \
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"             \
                             :: "r"(_dst), "l"(_src), "r"(_bytes));                       \
            }                                                                             \
        }                                                                                 \
        asm volatile("cp.async.commit_group;\n");                                         \
    } while (0)

    int num_k_tiles = (n_end - n_begin + SGB_SCALAR_BK - 1) / SGB_SCALAR_BK;

    // persistent CTA loop. fc/n_begin/n_end stay kernel-scoped.
    // Direct CTA mapping avoids the persistent-loop overhead. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring lower register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[SGB_SCALAR_WMITER * SGB_SCALAR_TM * SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

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
            int next_nIdx = n_begin + next_tile * SGB_SCALAR_BK;
            ISSUE_TILE_NT_SN(write_stage, next_nIdx);
        }

        float* As_rd = As_buf + read_stage * A_STAGE;
        float* Bs_rd = Bs_buf + read_stage * B_STAGE;
        float regM_next[SGB_SCALAR_WMITER * SGB_SCALAR_TM];
        float regN_next[SGB_SCALAR_WNITER * SGB_SCALAR_TN];
        #pragma unroll
        for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
            #pragma unroll
            for (int i = 0; i < SGB_SCALAR_TM; ++i)
                regM[wSubRowIdx * SGB_SCALAR_TM + i] = As_rd[0 * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + i];
        #pragma unroll
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
            #pragma unroll
            for (int i = 0; i < SGB_SCALAR_TN; ++i)
                regN[wSubColIdx * SGB_SCALAR_TN + i] = Bs_rd[0 * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + i];

        for (int dotIdx = 0; dotIdx < SGB_SCALAR_BK; ++dotIdx) {
            if (dotIdx + 1 < SGB_SCALAR_BK) {
                #pragma unroll
                for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
                    #pragma unroll
                    for (int i = 0; i < SGB_SCALAR_TM; ++i)
                        regM_next[wSubRowIdx * SGB_SCALAR_TM + i] = As_rd[(dotIdx + 1) * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + i];
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int i = 0; i < SGB_SCALAR_TN; ++i)
                        regN_next[wSubColIdx * SGB_SCALAR_TN + i] = Bs_rd[(dotIdx + 1) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + i];
            }
            // explicit __fmaf_rn for bit-exact
            // match with CPU `_mm256_fmadd_ps`. sgemm_bi_nt_splitn_big_partial.
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
                    #pragma unroll
                    for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM)
                        #pragma unroll
                        for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; ++resIdxN) {
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN)
                                      + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * SGB_SCALAR_TM + resIdxM],
                                regN[wSubColIdx * SGB_SCALAR_TN + resIdxN],
                                threadResults[idx]);
                        }
            if (dotIdx + 1 < SGB_SCALAR_BK) {
                #pragma unroll
                for (int i = 0; i < SGB_SCALAR_WMITER * SGB_SCALAR_TM; ++i) regM[i] = regM_next[i];
                #pragma unroll
                for (int i = 0; i < SGB_SCALAR_WNITER * SGB_SCALAR_TN; ++i) regN[i] = regN_next[i];
            }
        }
        read_stage = (read_stage + 1) % K_PIPE;
        write_stage = (write_stage + 1) % K_PIPE;
    }
    #undef ISSUE_TILE_NT_SN

    // Epilogue: OVERWRITE partial[fc, :, :] slot. Output shape is (M, K_out)
    // with row-stride K_out. No __stwt (reducer reads all fc's — not streaming).
    float* partial_chunk = partial + (long long)fc * M * K_out;
    float* partial_warp = partial_chunk + (pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM) * K_out + pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN;

    for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
            float* P_sub = partial_warp + wSubRowIdx * SGB_SCALAR_WSUBM * K_out + wSubColIdx * SGB_SCALAR_WSUBN;
            for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                int g_row = pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + resIdxM;
                if (g_row >= M) continue;
                for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; resIdxN += 4) {
                    int g_col = pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + resIdxN;
                    int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                    // Scalar fallback for right-edge OR non-%4 K_out (handles K_out=257).
                    if (g_col + 3 >= K_out || (K_out % 4) != 0) {
                        for (int j = 0; j < 4 && g_col + j < K_out; j++) {
                            P_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * K_out +
                                  threadColInWarp * SGB_SCALAR_TN + resIdxN + j] =
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
                        &P_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * K_out +
                               threadColInWarp * SGB_SCALAR_TN + resIdxN])[0] = out;
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (nt_splitn_big_partial)
}

// ============================================================================
// Slim-N variant: SGB_SCALAR_BM=128, SGB_SCALAR_BN=64, SGB_SCALAR_BK=32, SGB_SCALAR_WM=64, SGB_SCALAR_WN=32, SGB_SCALAR_WNITER=2
// Used for narrow N (N <= 512): SALE state/action (N=256), Mamba projections
// (N=128/256), SimbaV2 w2 (N=512). SGB_SCALAR_BN=64 cuts idle SMs on narrow outputs;
// SGB_SCALAR_BK=32 compensates (32-element reduction tile vs 16 for Big).
// Static smem: 2*(32*128+32*64)*4 = 48 KB ← fits 48 KB default static limit.
// ============================================================================
#undef SGB_SCALAR_BM
#undef SGB_SCALAR_BN
#undef SGB_SCALAR_BK
#undef SGB_SCALAR_WM
#undef SGB_SCALAR_WN
#undef SGB_SCALAR_WNITER
#undef SGB_SCALAR_TM
#undef SGB_SCALAR_TN
#undef SGB_SCALAR_WMITER
#undef SGB_SCALAR_WSUBM
#undef SGB_SCALAR_WSUBN
#undef SGB_SCALAR_ROW_STRIDE_A
#undef SGB_SCALAR_ROW_STRIDE_B
#undef SGB_SCALAR_NUM_THREADS  // Opt1: Big changed to 256; Slim preserves 128

#define SGB_SCALAR_NUM_THREADS 128
#define SGB_SCALAR_BM 128
#define SGB_SCALAR_BN 64
#define SGB_SCALAR_BK 32
#define SGB_SCALAR_WM 64
#define SGB_SCALAR_WN 32
#define SGB_SCALAR_WNITER 2
#define SGB_SCALAR_TM 8
#define SGB_SCALAR_TN 4
#define SGB_SCALAR_WMITER ((SGB_SCALAR_WM * SGB_SCALAR_WN) / (SGB_SCALAR_WARP_SIZE * SGB_SCALAR_TM * SGB_SCALAR_TN * SGB_SCALAR_WNITER))
#define SGB_SCALAR_WSUBM (SGB_SCALAR_WM / SGB_SCALAR_WMITER)
#define SGB_SCALAR_WSUBN (SGB_SCALAR_WN / SGB_SCALAR_WNITER)
#define SGB_SCALAR_ROW_STRIDE_A ((SGB_SCALAR_NUM_THREADS * 4) / SGB_SCALAR_BK)
#define SGB_SCALAR_ROW_STRIDE_B (SGB_SCALAR_NUM_THREADS / (SGB_SCALAR_BN / 4))

// ============================================================================
// Forward: C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C + bias
// ============================================================================
// __launch_bounds__(128, 3) — target 3 blocks/SM on Ada sm_89.
// 128 regs × 128 threads × 3 blocks = 49152 regs (< 64K/SM) ✓
// Smem 24KB × 2 = 48KB (static limit) — 3 blocks requires 99KB dynamic opt-in .
// At (128, 3) without dynamic opt-in, effective occupancy = 2 blocks/SM due to smem limit.
// ptxas: 128 regs, 0 spill.
extern "C" __global__ __launch_bounds__(SGB_SCALAR_NUM_THREADS, 2)
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
    // Smem: A transposed [SGB_SCALAR_BK * SGB_SCALAR_BM], B normal [SGB_SCALAR_BK * SGB_SCALAR_BN]
    __shared__ float As[SGB_SCALAR_BK * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD)];  // 16 * 128 = 2048 floats = 8KB
    __shared__ float Bs[SGB_SCALAR_BK * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD)];  // 16 * 128 = 2048 floats = 8KB

    // SGB_GROUP_M L2 swizzle — count tiles once.
    int num_pid_m = (M + SGB_SCALAR_BM - 1) / SGB_SCALAR_BM;
    int num_pid_n = (N + SGB_SCALAR_BN - 1) / SGB_SCALAR_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    // Warp and thread placement
    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_SCALAR_BN / SGB_SCALAR_WN);  // 0 or 1
    int warpRow = warpIdx / (SGB_SCALAR_BN / SGB_SCALAR_WN);  // 0 or 1
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);  // tid % 4
    int threadRowInWarp = tidInWarp / (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);  // tid / 4

    // A load indices (float4 coalesced along K)
    int innerRowA = threadIdx.x / (SGB_SCALAR_BK / 4);  // tid / 4, 0..31
    int innerColA = threadIdx.x % (SGB_SCALAR_BK / 4);  // tid % 4, 0..3

    // B load indices (float4 coalesced along N)
    int innerRowB = threadIdx.x / (SGB_SCALAR_BN / 4);  // tid / 32, 0..3
    int innerColB = threadIdx.x % (SGB_SCALAR_BN / 4);  // tid % 32, 0..31

    // Working-set registers (reset by mainloop per tile).
    float regM[SGB_SCALAR_WMITER * SGB_SCALAR_TM] = {0.0f};
    float regN[SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    // slim NN A coalesce (mirrors big NN ISSUE_TILE A-load).
    constexpr int WARPS_NNSLIM = SGB_SCALAR_NUM_THREADS / SGB_SCALAR_WARP_SIZE;
    constexpr int M_ROWS_PER_WARP_INST_NNSLIM = SGB_SCALAR_WARP_SIZE / SGB_SCALAR_BK;
    constexpr int M_ROWS_PER_WARP_NNSLIM = SGB_SCALAR_BM / WARPS_NNSLIM;
    constexpr int INSTR_PER_WARP_NNSLIM =
        M_ROWS_PER_WARP_NNSLIM / M_ROWS_PER_WARP_INST_NNSLIM;
    static_assert(SGB_SCALAR_WARP_SIZE % SGB_SCALAR_BK == 0, "SGB_SCALAR_WARP_SIZE divisible by SGB_SCALAR_BK (slim NN)");
    static_assert(SGB_SCALAR_BM % WARPS_NNSLIM == 0, "SGB_SCALAR_BM divisible by warps (slim NN)");
    int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int _m_in_warp_nn = (M_ROWS_PER_WARP_INST_NNSLIM > 0) ? (_lane / SGB_SCALAR_BK) : 0;
    int _k_local_lane = _lane % SGB_SCALAR_BK;

    // persistent CTA loop for slim NN.
    // Direct CTA mapping avoids the persistent-loop overhead. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring lower register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        // Bias-IN-FMA seed at K=0 (see Big NN bias-pre-seed block).
        float threadResults[SGB_SCALAR_WMITER * SGB_SCALAR_TM * SGB_SCALAR_WNITER * SGB_SCALAR_TN];

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        // Bias pre-seed. g_col mirrors epilog write (see L1939, L1969).
        if (bias != nullptr) {
            #pragma unroll
            for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
                #pragma unroll
                for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; ++resIdxN) {
                    int g_col = pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN +
                                threadColInWarp * SGB_SCALAR_TN + resIdxN;
                    float b_val = (g_col < N) ? bias[g_col] : 0.0f;
                    #pragma unroll
                    for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
                        #pragma unroll
                        for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) +
                                      wSubColIdx * SGB_SCALAR_TN + resIdxN;
                            threadResults[idx] = b_val;
                        }
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < SGB_SCALAR_WMITER * SGB_SCALAR_TM * SGB_SCALAR_WNITER * SGB_SCALAR_TN; ++i) {
                threadResults[i] = 0.0f;
            }
        }

        const float* A_block = A + pid_m * SGB_SCALAR_BM * lda;
        const float* B_block = B + pid_n * SGB_SCALAR_BN;
        float* C_warp = C + (pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM) * ldc + pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN;

    for (int bkIdx = 0; bkIdx < K; bkIdx += SGB_SCALAR_BK) {
        #pragma unroll
        for (int _it = 0; _it < INSTR_PER_WARP_NNSLIM; _it++) {
            int _m_local = _warp * M_ROWS_PER_WARP_NNSLIM
                           + _it * M_ROWS_PER_WARP_INST_NNSLIM + _m_in_warp_nn;
            int _g_row = pid_m * SGB_SCALAR_BM + _m_local;
            int _g_col = bkIdx + _k_local_lane;
            unsigned _dst = As_base
                + (_k_local_lane * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local)
                * (unsigned)sizeof(float);
            if (_g_row < M && _g_col < K) {
                const float* _src = A_block + _m_local * lda + _k_local_lane;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(_dst), "l"(_src));
            } else {
                As[_k_local_lane * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local] = 0.0f;
            }
        }

        // Load B: cp.async.ca.shared.global 16B (contiguous src+dst). Scalar OOB fallback.
        for (int offset = 0; offset + SGB_SCALAR_ROW_STRIDE_B <= SGB_SCALAR_BK; offset += SGB_SCALAR_ROW_STRIDE_B) {
            int g_row = bkIdx + innerRowB + offset;
            int g_col = pid_n * SGB_SCALAR_BN + innerColB * 4;
            unsigned dst = Bs_base + ((innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
            if (g_row < K && g_col + 3 < N && (ldb % 4 == 0)) {
                const float* src = B_block + (innerRowB + offset) * ldb + innerColB * 4;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 0] = (g_row < K && g_col + 0 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 1] = (g_row < K && g_col + 1 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 2] = (g_row < K && g_col + 2 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 3] = (g_row < K && g_col + 3 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 3] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        // Compute: warptile matmul from smem
        for (int dotIdx = 0; dotIdx < SGB_SCALAR_BK; ++dotIdx) {
            // Load A column into registers (transposed smem = contiguous)
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
                for (int i = 0; i < SGB_SCALAR_TM; ++i) {
                    regM[wSubRowIdx * SGB_SCALAR_TM + i] =
                        As[dotIdx * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM +
                           threadRowInWarp * SGB_SCALAR_TM + i];
                }
            }
            // Load B row into registers
            for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
                for (int i = 0; i < SGB_SCALAR_TN; ++i) {
                    regN[wSubColIdx * SGB_SCALAR_TN + i] =
                        Bs[dotIdx * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN +
                           threadColInWarp * SGB_SCALAR_TN + i];
                }
            }
            // Outer product: 256 FMA
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
                    for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                        for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; ++resIdxN) {
                            // explicit __fmaf_rn for bit-exact
                            // match with CPU `_mm256_fmadd_ps`.
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN)
                                      + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * SGB_SCALAR_TM + resIdxM],
                                regN[wSubColIdx * SGB_SCALAR_TN + resIdxN],
                                threadResults[idx]);
                        }
                    }
                }
            }
        }

        A_block += SGB_SCALAR_BK;       // move SGB_SCALAR_BK columns right
        B_block += SGB_SCALAR_BK * ldb; // move SGB_SCALAR_BK rows down
        __syncthreads();
    }

    // Epilogue: write results with alpha, beta, bias (float4 stores)
    for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
            float* C_sub = C_warp + wSubRowIdx * SGB_SCALAR_WSUBM * ldc + wSubColIdx * SGB_SCALAR_WSUBN;
            for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                int g_row = pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM +
                            threadRowInWarp * SGB_SCALAR_TM + resIdxM;
                if (g_row >= M) continue;
                for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; resIdxN += 4) {
                    int g_col = pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN +
                                threadColInWarp * SGB_SCALAR_TN + resIdxN;
                    // Fallback to scalar on column tail OR when ldc % 4 != 0.
                    // STG.128 needs 16-byte aligned address — row-stride in bytes
                    // (ldc * 4) must be a multiple of 16, so ldc must be a multiple
                    // of 4. Otherwise odd rows hit CUDA_ERROR_MISALIGNED_ADDRESS.
                    if (g_col + 3 >= N || (ldc & 3) != 0) {
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) +
                                      wSubColIdx * SGB_SCALAR_TN + resIdxN + j;
                            // Bias seeded at K=0 (see init).
                            float val = alpha * threadResults[idx];
                            if (beta != 0.0f) val += beta * C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * ldc + threadColInWarp * SGB_SCALAR_TN + resIdxN + j];
                            C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * ldc + threadColInWarp * SGB_SCALAR_TN + resIdxN + j] = val;
                        }
                        continue;
                    }
                    float4 tmp;
                    if (beta != 0.0f) {
                        tmp = reinterpret_cast<float4*>(
                            &C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * ldc +
                                   threadColInWarp * SGB_SCALAR_TN + resIdxN])[0];
                    }
                    int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) +
                              wSubColIdx * SGB_SCALAR_TN + resIdxN;
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
                        &C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * ldc +
                               threadColInWarp * SGB_SCALAR_TN + resIdxN])[0] = out;
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (slim NN)
}

// ============================================================================
// Split-K Slim NN partial — wave-fill extension for underfilled
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
//   - K_chunk % SGB_SCALAR_BK (=32) == 0 (enforced by dispatcher)
//   - K % 32 == 0 (enforced by dispatcher — no K tail)
//   - No alpha / bias / beta in this kernel — raw tile sums only
//   - Static 16 KB smem (same as Slim NN) — no dynamic-smem attribute needed
//
// Mirror of sgemm_bi_nn_splitk_big_partial (the big-tile variant) but for SGB_SCALAR_BM=128 SGB_SCALAR_BN=64
// SGB_SCALAR_BK=32 tile. This is the tile that actually fires on b=64 production GEMMs.
// ============================================================================
extern "C" __global__ __launch_bounds__(SGB_SCALAR_NUM_THREADS, 2)
void sgemm_bi_nn_splitk_slim_partial(
    float* __restrict__ partial,       // [F * M * N] — unique slot per fc
    const float* __restrict__ A,       // [M, K_full]
    const float* __restrict__ B,       // [K_full, N]
    int M, int N, int K,
    int lda, int ldb,
    int K_chunk                         // must be multiple of SGB_SCALAR_BK=32
) {
    __shared__ float As[SGB_SCALAR_BK * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD)];
    __shared__ float Bs[SGB_SCALAR_BK * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD)];

    // Decode fc from z-axis; early exit if fc is out of K-range.
    int fc = blockIdx.z;
    int k_begin = fc * K_chunk;
    if (k_begin >= K) return;
    int k_end = min(K, k_begin + K_chunk);

    // SGB_GROUP_M L2 swizzle (identical to Slim NN)
    int num_pid_m = (M + SGB_SCALAR_BM - 1) / SGB_SCALAR_BM;
    int num_pid_n = (N + SGB_SCALAR_BN - 1) / SGB_SCALAR_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int warpRow = warpIdx / (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);
    int threadRowInWarp = tidInWarp / (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);

    int innerRowA = threadIdx.x / (SGB_SCALAR_BK / 4);
    int innerColA = threadIdx.x % (SGB_SCALAR_BK / 4);
    int innerRowB = threadIdx.x / (SGB_SCALAR_BN / 4);
    int innerColB = threadIdx.x % (SGB_SCALAR_BN / 4);

    float regM[SGB_SCALAR_WMITER * SGB_SCALAR_TM] = {0.0f};
    float regN[SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    // slim NN splitk A coalesce.
    constexpr int WARPS_NNSKP = SGB_SCALAR_NUM_THREADS / SGB_SCALAR_WARP_SIZE;
    constexpr int M_ROWS_PER_WARP_INST_NNSKP = SGB_SCALAR_WARP_SIZE / SGB_SCALAR_BK;
    constexpr int M_ROWS_PER_WARP_NNSKP = SGB_SCALAR_BM / WARPS_NNSKP;
    constexpr int INSTR_PER_WARP_NNSKP =
        M_ROWS_PER_WARP_NNSKP / M_ROWS_PER_WARP_INST_NNSKP;
    static_assert(SGB_SCALAR_WARP_SIZE % SGB_SCALAR_BK == 0, "SGB_SCALAR_WARP_SIZE divisible by SGB_SCALAR_BK (slim NN splitk)");
    static_assert(SGB_SCALAR_BM % WARPS_NNSKP == 0, "SGB_SCALAR_BM divisible by warps (slim NN splitk)");
    int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int _m_in_warp_nnsk = _lane / SGB_SCALAR_BK;
    int _k_local_lane = _lane % SGB_SCALAR_BK;

    // persistent CTA loop. fc/k_begin/k_end stay kernel-scoped.
    // Direct CTA mapping avoids the persistent-loop overhead. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring lower register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[SGB_SCALAR_WMITER * SGB_SCALAR_TM * SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        const float* A_block = A + pid_m * SGB_SCALAR_BM * lda + k_begin;
        const float* B_block = B + (long long)k_begin * ldb + pid_n * SGB_SCALAR_BN;

    for (int bkIdx = k_begin; bkIdx < k_end; bkIdx += SGB_SCALAR_BK) {
        #pragma unroll
        for (int _it = 0; _it < INSTR_PER_WARP_NNSKP; _it++) {
            int _m_local = _warp * M_ROWS_PER_WARP_NNSKP
                           + _it * M_ROWS_PER_WARP_INST_NNSKP + _m_in_warp_nnsk;
            int _g_row = pid_m * SGB_SCALAR_BM + _m_local;
            int _g_col = bkIdx + _k_local_lane;
            unsigned _dst = As_base
                + (_k_local_lane * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local)
                * (unsigned)sizeof(float);
            if (_g_row < M && _g_col < k_end) {
                const float* _src = A_block + _m_local * lda + _k_local_lane;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(_dst), "l"(_src));
            } else {
                As[_k_local_lane * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local] = 0.0f;
            }
        }

        // Load B: contiguous cp.async.16B with scalar OOB fallback.
        for (int offset = 0; offset + SGB_SCALAR_ROW_STRIDE_B <= SGB_SCALAR_BK; offset += SGB_SCALAR_ROW_STRIDE_B) {
            int g_row = bkIdx + innerRowB + offset;
            int g_col = pid_n * SGB_SCALAR_BN + innerColB * 4;
            unsigned dst = Bs_base + ((innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
            if (g_row < k_end && g_col + 3 < N && (ldb % 4 == 0)) {
                const float* src = B_block + (innerRowB + offset) * ldb + innerColB * 4;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 0] = (g_row < k_end && g_col + 0 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 1] = (g_row < k_end && g_col + 1 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 2] = (g_row < k_end && g_col + 2 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 3] = (g_row < k_end && g_col + 3 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 3] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        // Compute: same warptile matmul as Slim NN — identical FMA order.
        for (int dotIdx = 0; dotIdx < SGB_SCALAR_BK; ++dotIdx) {
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
                for (int i = 0; i < SGB_SCALAR_TM; ++i) {
                    regM[wSubRowIdx * SGB_SCALAR_TM + i] =
                        As[dotIdx * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM +
                           threadRowInWarp * SGB_SCALAR_TM + i];
                }
            }
            for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
                for (int i = 0; i < SGB_SCALAR_TN; ++i) {
                    regN[wSubColIdx * SGB_SCALAR_TN + i] =
                        Bs[dotIdx * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN +
                           threadColInWarp * SGB_SCALAR_TN + i];
                }
            }
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
                    for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                        for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; ++resIdxN) {
                            // explicit __fmaf_rn for bit-exact
                            // match with CPU `_mm256_fmadd_ps`.
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN)
                                      + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * SGB_SCALAR_TM + resIdxM],
                                regN[wSubColIdx * SGB_SCALAR_TN + resIdxN],
                                threadResults[idx]);
                        }
                    }
                }
            }
        }

        // Advance A and B pointers to next SGB_SCALAR_BK-column tile (same as Slim NN).
        A_block += SGB_SCALAR_BK;
        B_block += SGB_SCALAR_BK * ldb;
        __syncthreads();
    }

    // Epilogue: OVERWRITE partial[fc, :, :]. No alpha, no bias, no beta.
    float* partial_chunk = partial + (long long)fc * M * N;
    float* partial_warp = partial_chunk + (pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM) * N + pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN;

    for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
            float* P_sub = partial_warp + wSubRowIdx * SGB_SCALAR_WSUBM * N + wSubColIdx * SGB_SCALAR_WSUBN;
            for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                int g_row = pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM +
                            threadRowInWarp * SGB_SCALAR_TM + resIdxM;
                if (g_row >= M) continue;
                for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; resIdxN += 4) {
                    int g_col = pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN +
                                threadColInWarp * SGB_SCALAR_TN + resIdxN;
                    int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) +
                              wSubColIdx * SGB_SCALAR_TN + resIdxN;
                    // Scalar fallback for right-edge OR non-%4 N (hardens audit M2).
                    if (g_col + 3 >= N || (N % 4) != 0) {
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            P_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * N +
                                  threadColInWarp * SGB_SCALAR_TN + resIdxN + j] =
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
                        &P_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * N +
                               threadColInWarp * SGB_SCALAR_TN + resIdxN])[0] = out;
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (nn_splitk_slim_partial)
}

// ============================================================================
// Backward dW (SGB_SCALAR_TN): C[K,N] += alpha * A^T[K,M] @ B[M,N]
// ============================================================================
// A = X_saved [M, K] — read transposed
// B = dY [M, N]
// C = dW [K, N] — accumulated
// Output tile [SGB_SCALAR_BM, SGB_SCALAR_BN] over (K, N). M is reduction axis.
// __launch_bounds__(128, 2) — target 2 blocks/SM (ptxas: 128 regs, 0 spill).
// 2 blocks × 24KB smem = 48KB static limit exactly.
extern "C" __global__ __launch_bounds__(SGB_SCALAR_NUM_THREADS, 2)
void sgemm_bi_tn_slim(
    float* __restrict__ C,
    const float* __restrict__ A,  // X [M, K]
    const float* __restrict__ B,  // dY [M, N]
    float alpha,
    int M_red,    // batch (reduction axis)
    int K_out,    // output rows
    int N         // output cols
) {
    __shared__ float As[SGB_SCALAR_BK * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD)];
    __shared__ float Bs[SGB_SCALAR_BK * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD)];

    int num_pid_m = (K_out + SGB_SCALAR_BM - 1) / SGB_SCALAR_BM;
    int num_pid_n = (N + SGB_SCALAR_BN - 1) / SGB_SCALAR_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int warpRow = warpIdx / (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);
    int threadRowInWarp = tidInWarp / (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);

    int innerRowA = threadIdx.x / (SGB_SCALAR_BK / 4);
    int innerColA = threadIdx.x % (SGB_SCALAR_BK / 4);
    int innerRowB = threadIdx.x / (SGB_SCALAR_BN / 4);
    int innerColB = threadIdx.x % (SGB_SCALAR_BN / 4);

    float regM[SGB_SCALAR_WMITER * SGB_SCALAR_TM] = {0.0f};
    float regN[SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

    // persistent CTA loop for slim SGB_SCALAR_TN.
    // Direct CTA mapping avoids the persistent-loop overhead. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring lower register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[SGB_SCALAR_WMITER * SGB_SCALAR_TM * SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        float* C_warp = C + (pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM) * N + pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN;

    // Phase B1: cp.async single-buffer load path (A transposed + B).
    // Replaces synchronous scalar loads → async global→shared DMA.
    // Frees register staging, reduces I$ pressure, potentially overlaps DMA latency.
    // .ca = L1 cache (A/B reused across K tiles per block). sm_80+ required.
    // Determinism: identical bytes in identical smem locations as scalar path.
    // OOB: scalar write of 0.0f (matches scalar path's explicit zero-fill).
    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    // slim SGB_SCALAR_TN A coalesce (mirrors big SGB_SCALAR_TN ISSUE_TILE_TN).
    constexpr int WARPS_TNSLIM = SGB_SCALAR_NUM_THREADS / SGB_SCALAR_WARP_SIZE;
    constexpr int ROWS_PER_WARP_TNSLIM = SGB_SCALAR_BK / WARPS_TNSLIM;
    static_assert(SGB_SCALAR_BK % WARPS_TNSLIM == 0, "SGB_SCALAR_BK divisible by warps (slim SGB_SCALAR_TN)");
    static_assert(SGB_SCALAR_BM % (SGB_SCALAR_WARP_SIZE * 4) == 0, "SGB_SCALAR_BM divisible by 32*4 (slim SGB_SCALAR_TN)");
    int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    for (int mIdx = 0; mIdx < M_red; mIdx += SGB_SCALAR_BK) {
        #pragma unroll
        for (int _r = 0; _r < ROWS_PER_WARP_TNSLIM; _r++) {
            int k_local = _warp * ROWS_PER_WARP_TNSLIM + _r;
            int m_local = _lane * 4;
            int _g_m = mIdx + k_local;
            int _g_k = pid_m * SGB_SCALAR_BM + m_local;
            unsigned _dst = As_base
                + (k_local * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + m_local)
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
                        As[k_local * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + m_local + _i] = 0.0f;
                    }
                }
            }
        }

        // Load B via cp.async.ca.shared.global 16B (float4, contiguous).
        // Scalar fallback for edge / non-%4 N.
        for (int offset = 0; offset + SGB_SCALAR_ROW_STRIDE_B <= SGB_SCALAR_BK; offset += SGB_SCALAR_ROW_STRIDE_B) {
            int g_m = mIdx + innerRowB + offset;
            int g_n = pid_n * SGB_SCALAR_BN + innerColB * 4;
            unsigned dst = Bs_base + ((innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
            if (g_m < M_red && g_n + 3 < N && (N % 4 == 0)) {
                const float* src = B + ((long long)g_m) * N + pid_n * SGB_SCALAR_BN + innerColB * 4;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 0] = (g_m < M_red && g_n + 0 < N) ? B[g_m * N + pid_n * SGB_SCALAR_BN + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 1] = (g_m < M_red && g_n + 1 < N) ? B[g_m * N + pid_n * SGB_SCALAR_BN + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 2] = (g_m < M_red && g_n + 2 < N) ? B[g_m * N + pid_n * SGB_SCALAR_BN + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 3] = (g_m < M_red && g_n + 3 < N) ? B[g_m * N + pid_n * SGB_SCALAR_BN + innerColB * 4 + 3] : 0.0f;
            }
        }
        // Commit + wait_all → guarantees all async loads visible before compute.
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        for (int dotIdx = 0; dotIdx < SGB_SCALAR_BK; ++dotIdx) {
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
                for (int i = 0; i < SGB_SCALAR_TM; ++i)
                    regM[wSubRowIdx * SGB_SCALAR_TM + i] = As[dotIdx * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + i];
            for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
                for (int i = 0; i < SGB_SCALAR_TN; ++i)
                    regN[wSubColIdx * SGB_SCALAR_TN + i] = Bs[dotIdx * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + i];
            // explicit __fmaf_rn for bit-exact
            // match with CPU `_mm256_fmadd_ps`. This matches the sibling kernels
            // (RoPE backward FMA pin).
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
                    for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM)
                        for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; ++resIdxN) {
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN)
                                      + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * SGB_SCALAR_TM + resIdxM],
                                regN[wSubColIdx * SGB_SCALAR_TN + resIdxN],
                                threadResults[idx]);
                        }
        }
        __syncthreads();
    }

    // Epilogue: accumulate into dW
    for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
            float* C_sub = C_warp + wSubRowIdx * SGB_SCALAR_WSUBM * N + wSubColIdx * SGB_SCALAR_WSUBN;
            for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                int g_row = pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + resIdxM;
                if (g_row >= K_out) continue;
                for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; resIdxN += 4) {
                    int g_col = pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + resIdxN;
                    // Fallback to scalar on tail OR when N (row-stride) % 4 != 0
                    // (STG.128 / LDG.128 need 16-byte aligned address).
                    if (g_col + 3 >= N || (N & 3) != 0) {
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) + wSubColIdx * SGB_SCALAR_TN + resIdxN + j;
                            C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * N + threadColInWarp * SGB_SCALAR_TN + resIdxN + j] += alpha * threadResults[idx];
                        }
                        continue;
                    }
                    float4 old = reinterpret_cast<float4*>(&C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * N + threadColInWarp * SGB_SCALAR_TN + resIdxN])[0];
                    int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                    old.x += alpha * threadResults[idx + 0];
                    old.y += alpha * threadResults[idx + 1];
                    old.z += alpha * threadResults[idx + 2];
                    old.w += alpha * threadResults[idx + 3];
                    reinterpret_cast<float4*>(&C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * N + threadColInWarp * SGB_SCALAR_TN + resIdxN])[0] = old;
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (slim SGB_SCALAR_TN)
}

// ============================================================================
// Backward dX (NT): C[M,K] = alpha * A[M,N] @ B^T[N,K]
// ============================================================================
// A = dY [M, N]
// B = W [K, N] — read transposed as W^T[N,K]
// C = dX [M, K] — overwrite
// __launch_bounds__(128, 2) — target 2 blocks/SM (ptxas: 128 regs, 0 spill).
extern "C" __global__ __launch_bounds__(SGB_SCALAR_NUM_THREADS, 2)
void sgemm_bi_nt_slim(
    float* __restrict__ C,
    const float* __restrict__ A,  // dY [M, N]
    const float* __restrict__ B,  // W [K, N]
    float alpha,
    int M,        // output rows
    int N,        // reduction axis
    int K_out     // output cols
) {
    __shared__ float As[SGB_SCALAR_BK * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD)];
    __shared__ float Bs[SGB_SCALAR_BK * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD)];

    int num_pid_m = (M + SGB_SCALAR_BM - 1) / SGB_SCALAR_BM;
    int num_pid_n = (K_out + SGB_SCALAR_BN - 1) / SGB_SCALAR_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int warpRow = warpIdx / (SGB_SCALAR_BN / SGB_SCALAR_WN);
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);
    int threadRowInWarp = tidInWarp / (SGB_SCALAR_WSUBN / SGB_SCALAR_TN);

    int innerRowA = threadIdx.x / (SGB_SCALAR_BK / 4);
    int innerColA = threadIdx.x % (SGB_SCALAR_BK / 4);
    int innerRowB = threadIdx.x / (SGB_SCALAR_BN / 4);
    int innerColB = threadIdx.x % (SGB_SCALAR_BN / 4);

    float regM[SGB_SCALAR_WMITER * SGB_SCALAR_TM] = {0.0f};
    float regN[SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

    // persistent CTA loop for slim NT.
    // Direct CTA mapping avoids the persistent-loop overhead. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring lower register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[SGB_SCALAR_WMITER * SGB_SCALAR_TM * SGB_SCALAR_WNITER * SGB_SCALAR_TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        float* C_warp = C + (pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM) * K_out + pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN;

    // Phase B2: cp.async loads for NT backward dX.
    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    // slim NT A coalesce.
    constexpr int WARPS_NTSLIM = SGB_SCALAR_NUM_THREADS / SGB_SCALAR_WARP_SIZE;
    constexpr int M_ROWS_PER_WARP_INST_NTSL = SGB_SCALAR_WARP_SIZE / SGB_SCALAR_BK;
    constexpr int M_ROWS_PER_WARP_NTSL = SGB_SCALAR_BM / WARPS_NTSLIM;
    constexpr int INSTR_PER_WARP_NTSL =
        M_ROWS_PER_WARP_NTSL / M_ROWS_PER_WARP_INST_NTSL;
    static_assert(SGB_SCALAR_WARP_SIZE % SGB_SCALAR_BK == 0, "SGB_SCALAR_WARP_SIZE divisible by SGB_SCALAR_BK (slim NT)");
    static_assert(SGB_SCALAR_BM % WARPS_NTSLIM == 0, "SGB_SCALAR_BM divisible by warps (slim NT)");
    int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int _m_in_warp_ntsl = _lane / SGB_SCALAR_BK;
    int _n_local_lane = _lane % SGB_SCALAR_BK;
    for (int nIdx = 0; nIdx < N; nIdx += SGB_SCALAR_BK) {
        #pragma unroll
        for (int _it = 0; _it < INSTR_PER_WARP_NTSL; _it++) {
            int _m_local = _warp * M_ROWS_PER_WARP_NTSL
                           + _it * M_ROWS_PER_WARP_INST_NTSL + _m_in_warp_ntsl;
            int _g_m = pid_m * SGB_SCALAR_BM + _m_local;
            int _g_n = nIdx + _n_local_lane;
            unsigned _dst = As_base
                + (_n_local_lane * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local)
                * (unsigned)sizeof(float);
            if (_g_m < M && _g_n < N) {
                const float* _src = A + (long long)_g_m * N + _g_n;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(_dst), "l"(_src));
            } else {
                As[_n_local_lane * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local] = 0.0f;
            }
        }

        // Load B^T (W): source stride N (non-contiguous K-rows), dest contiguous → 4× cp.async.4B.
        for (int offset = 0; offset + SGB_SCALAR_ROW_STRIDE_B <= SGB_SCALAR_BK; offset += SGB_SCALAR_ROW_STRIDE_B) {
            int n_local = innerRowB + offset;
            int k_base = innerColB * 4;
            int g_n = nIdx + n_local;
            int g_k = pid_n * SGB_SCALAR_BN + k_base;
            bool pn = g_n < N;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                unsigned dst = Bs_base + (n_local * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + k_base + i) * (unsigned)sizeof(float);
                if (pn && (g_k + i) < K_out) {
                    const float* src = B + ((long long)(g_k + i)) * N + g_n;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                                 :: "r"(dst), "l"(src));
                } else {
                    Bs[n_local * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + k_base + i] = 0.0f;
                }
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        for (int dotIdx = 0; dotIdx < SGB_SCALAR_BK; ++dotIdx) {
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
                for (int i = 0; i < SGB_SCALAR_TM; ++i)
                    regM[wSubRowIdx * SGB_SCALAR_TM + i] = As[dotIdx * (SGB_SCALAR_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + i];
            for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
                for (int i = 0; i < SGB_SCALAR_TN; ++i)
                    regN[wSubColIdx * SGB_SCALAR_TN + i] = Bs[dotIdx * (SGB_SCALAR_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + i];
            // explicit __fmaf_rn for bit-exact
            // match with CPU `_mm256_fmadd_ps`. This matches the sibling kernels
            // (RoPE backward FMA pin).
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx)
                for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx)
                    for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM)
                        for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; ++resIdxN) {
                            int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN)
                                      + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * SGB_SCALAR_TM + resIdxM],
                                regN[wSubColIdx * SGB_SCALAR_TN + resIdxN],
                                threadResults[idx]);
                        }
        }
        __syncthreads();
    }

    // Epilogue: overwrite dX
    for (int wSubRowIdx = 0; wSubRowIdx < SGB_SCALAR_WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < SGB_SCALAR_WNITER; ++wSubColIdx) {
            float* C_sub = C_warp + wSubRowIdx * SGB_SCALAR_WSUBM * K_out + wSubColIdx * SGB_SCALAR_WSUBN;
            for (int resIdxM = 0; resIdxM < SGB_SCALAR_TM; ++resIdxM) {
                int g_row = pid_m * SGB_SCALAR_BM + warpRow * SGB_SCALAR_WM + wSubRowIdx * SGB_SCALAR_WSUBM + threadRowInWarp * SGB_SCALAR_TM + resIdxM;
                if (g_row >= M) continue;
                for (int resIdxN = 0; resIdxN < SGB_SCALAR_TN; resIdxN += 4) {
                    int g_col = pid_n * SGB_SCALAR_BN + warpCol * SGB_SCALAR_WN + wSubColIdx * SGB_SCALAR_WSUBN + threadColInWarp * SGB_SCALAR_TN + resIdxN;
                    // float4 write only when K_out is %4-aligned (K_out=257 → scalar).
                    if (g_col + 3 >= K_out || (K_out % 4 != 0)) {
                        int idx_base = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                        for (int j = 0; j < 4 && g_col + j < K_out; j++) {
                            __stwt(&C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * K_out + threadColInWarp * SGB_SCALAR_TN + resIdxN + j], alpha * threadResults[idx_base + j]);
                        }
                        continue;
                    }
                    int idx = (wSubRowIdx * SGB_SCALAR_TM + resIdxM) * (SGB_SCALAR_WNITER * SGB_SCALAR_TN) + wSubColIdx * SGB_SCALAR_TN + resIdxN;
                    float4 out = {
                        alpha * threadResults[idx + 0],
                        alpha * threadResults[idx + 1],
                        alpha * threadResults[idx + 2],
                        alpha * threadResults[idx + 3]
                    };
                    // __stwt — streaming store. NT backward dX is OVERWRITE (no accumulation),
                    // so marking C lines evict-first is safe and prevents C writes from evicting A staging / B working set.
                    __stwt(reinterpret_cast<float4*>(&C_sub[(threadRowInWarp * SGB_SCALAR_TM + resIdxM) * K_out + threadColInWarp * SGB_SCALAR_TN + resIdxN]), out);
                }
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (slim NT)
}

// Slim geometry ends here — undef everything so no later-appended section
// can silently inherit SGB_SCALAR_BM=128/SGB_SCALAR_BN=64/SGB_SCALAR_BK=32 (the 0.4.0 geometry-leak class).
// Every section below defines its own prefixed macros and undefs them.
#undef SGB_SCALAR_BM
#undef SGB_SCALAR_BN
#undef SGB_SCALAR_BK
#undef SGB_SCALAR_WM
#undef SGB_SCALAR_WN
#undef SGB_SCALAR_WNITER
#undef SGB_SCALAR_TM
#undef SGB_SCALAR_TN
#undef SGB_SCALAR_WMITER
#undef SGB_SCALAR_WSUBM
#undef SGB_SCALAR_WSUBN
#undef SGB_SCALAR_ROW_STRIDE_A
#undef SGB_SCALAR_ROW_STRIDE_B
#undef SGB_SCALAR_NUM_THREADS

// ============================================================================
// GEMV-N1 NN (forward, N=1): Y[M] = alpha * X[M,K] @ W[K] + beta*Y + bias
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
        // explicit __fmaf_rn matches
        // The same FMA pattern is used by the sibling kernels. Under
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
    // see sgemm_bi_nn_ultra_thin pin.
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
// GEMV-N1 SGB_SCALAR_TN (backward dW, N=1): dW[K] += alpha * X^T[K,M] @ dY[M]
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
// SGB_SCALAR_TN semantics: beta=1 (accumulation into existing dW).
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
    // see sgemm_bi_nn_ultra_thin pin.
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
// GEMV-N1 NT (backward dX, N=1): dX[M,K] = alpha * dY[M] @ W^T[K]
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
// Narrow-N NN (forward, N∈9..48): C[M,N] = alpha * A[M,K] @ B[K,N] + beta*C + bias
// ============================================================================
// Specialized for narrow N (9..48). Replaces cuBLAS fallback for critic qhead
// N=25 (TQC quantiles). Tile: SGB_SCALAR_BM=64 SGB_SCALAR_BN=32 SGB_SCALAR_BK=16, 128 threads, 2x2 warps.
//
// Scalar N-epilogue handles non-%4 N (e.g. N=25, last 1..3 cols written scalar).
// Scalar K-fallback for non-%4 K via lda%4 runtime check.
//
// Design mirrors Slim-N but smaller tile → 2x more grid blocks for M=4224, N<64
// (wave underfill protection at narrow N).
#define SGB_NARROW_BM 64
#define SGB_NARROW_BN 32
#define SGB_NARROW_BK 16
#define SGB_NARROW_WM 32
#define SGB_NARROW_WN 16
#define SGB_NARROW_WMITER 1
#define SGB_NARROW_WNITER 1
#define SGB_NARROW_TM 4
#define SGB_NARROW_TN 4
#define SGB_NARROW_NUM_THREADS 128
#define SGB_NARROW_WSUBM (SGB_NARROW_WM / SGB_NARROW_WMITER)   // 32
#define SGB_NARROW_WSUBN (SGB_NARROW_WN / SGB_NARROW_WNITER)   // 16
#define SGB_NARROW_ROW_STRIDE_A ((SGB_NARROW_NUM_THREADS * 4) / SGB_NARROW_BK)  // 32
#define SGB_NARROW_ROW_STRIDE_B (SGB_NARROW_NUM_THREADS / (SGB_NARROW_BN / 4))  // 16

extern "C" __global__ __launch_bounds__(SGB_NARROW_NUM_THREADS, 4)
void sgemm_bi_nn_narrow(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float alpha, float beta,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    int post_op  // reserved for future epilogue fusion; 0 = none (currently unused)
) {
    (void)post_op;
    __shared__ float As[SGB_NARROW_BK * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD)];
    __shared__ float Bs[SGB_NARROW_BK * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD)];

    int num_pid_m = (M + SGB_NARROW_BM - 1) / SGB_NARROW_BM;
    int num_pid_n = (N + SGB_NARROW_BN - 1) / SGB_NARROW_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_NARROW_BN / SGB_NARROW_WN);  // 0 or 1
    int warpRow = warpIdx / (SGB_NARROW_BN / SGB_NARROW_WN);  // 0 or 1
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_NARROW_WSUBN / SGB_NARROW_TN);  // 0..3
    int threadRowInWarp = tidInWarp / (SGB_NARROW_WSUBN / SGB_NARROW_TN);  // 0..7

    int innerRowA = threadIdx.x / (SGB_NARROW_BK / 4);  // tid / 4, 0..31
    int innerColA = threadIdx.x % (SGB_NARROW_BK / 4);  // tid % 4, 0..3
    int innerRowB = threadIdx.x / (SGB_NARROW_BN / 4);  // tid / 8, 0..15
    int innerColB = threadIdx.x % (SGB_NARROW_BN / 4);  // tid % 8, 0..7

    float regM[SGB_NARROW_WMITER * SGB_NARROW_TM] = {0.0f};
    float regN[SGB_NARROW_WNITER * SGB_NARROW_TN] = {0.0f};

    // Phase B2: cp.async loads for narrow-N NN variant.
    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    // persistent CTA loop for nn_narrow.
    // Direct CTA mapping avoids the persistent-loop overhead. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring lower register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        const float* A_block = A + pid_m * SGB_NARROW_BM * lda;
        const float* B_block = B + pid_n * SGB_NARROW_BN;
        float* C_warp = C + (pid_m * SGB_NARROW_BM + warpRow * SGB_NARROW_WM) * ldc + pid_n * SGB_NARROW_BN + warpCol * SGB_NARROW_WN;

        // Bias is pre-seed threadResults with bias[g_col]
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
        // The CPU reference pre-seeds the same bias before accumulation.
        // Only valid for alpha=1 (all training forward calls use alpha=1).
        float threadResults[SGB_NARROW_WMITER * SGB_NARROW_TM * SGB_NARROW_WNITER * SGB_NARROW_TN];
        #pragma unroll
        for (int rm = 0; rm < SGB_NARROW_TM; ++rm) {
            int g_col_base = pid_n * SGB_NARROW_BN + warpCol * SGB_NARROW_WN + threadColInWarp * SGB_NARROW_TN;
            #pragma unroll
            for (int rn = 0; rn < SGB_NARROW_TN; ++rn) {
                int g_col = g_col_base + rn;
                int idx = rm * SGB_NARROW_TN + rn;
                threadResults[idx] =
                    (bias != nullptr && g_col < N) ? bias[g_col] : 0.0f;
            }
        }

    for (int bkIdx = 0; bkIdx < K; bkIdx += SGB_NARROW_BK) {
        // narrow NN coalesce: 4 warps × 8 instr/warp at 50%
        // cache util (vs 12.5% legacy). SGB_NARROW_BM=64, SGB_NARROW_BK=16. M_ROWS_PER_WARP_INST=2.
        {
            constexpr int WARPS_NN_NARROW = SGB_NARROW_NUM_THREADS / SGB_SCALAR_WARP_SIZE;            // 4
            constexpr int M_ROWS_PER_WARP_INST_NN_NR = SGB_SCALAR_WARP_SIZE / SGB_NARROW_BK;          // 2
            constexpr int M_ROWS_PER_WARP_NN_NR = SGB_NARROW_BM / WARPS_NN_NARROW;        // 16
            constexpr int INSTR_PER_WARP_NN_NR =
                M_ROWS_PER_WARP_NN_NR / M_ROWS_PER_WARP_INST_NN_NR;              // 8
            static_assert(SGB_SCALAR_WARP_SIZE % SGB_NARROW_BK == 0, "SGB_SCALAR_WARP_SIZE divisible by SGB_NARROW_BK (narrow NN)");
            static_assert(SGB_NARROW_BM % WARPS_NN_NARROW == 0, "SGB_NARROW_BM divisible by warps (narrow NN)");
            int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;
            int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;
            int _m_in_warp_nn_nr = _lane / SGB_NARROW_BK;
            int _k_local_lane_nn_nr = _lane % SGB_NARROW_BK;
            #pragma unroll
            for (int _it = 0; _it < INSTR_PER_WARP_NN_NR; _it++) {
                int _m_local = _warp * M_ROWS_PER_WARP_NN_NR
                               + _it * M_ROWS_PER_WARP_INST_NN_NR + _m_in_warp_nn_nr;
                int _g_row = pid_m * SGB_NARROW_BM + _m_local;
                int _g_col = bkIdx + _k_local_lane_nn_nr;
                unsigned _dst = As_base
                    + (_k_local_lane_nn_nr * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local)
                    * (unsigned)sizeof(float);
                if (_g_row < M && _g_col < K) {
                    const float* _src = A_block + _m_local * lda + _k_local_lane_nn_nr;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                                 :: "r"(_dst), "l"(_src));
                } else {
                    As[_k_local_lane_nn_nr * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local] = 0.0f;
                }
            }
        }

        // Load B: cp.async.16B contiguous.
        for (int offset = 0; offset + SGB_NARROW_ROW_STRIDE_B <= SGB_NARROW_BK; offset += SGB_NARROW_ROW_STRIDE_B) {
            int g_row = bkIdx + innerRowB + offset;
            int g_col = pid_n * SGB_NARROW_BN + innerColB * 4;
            unsigned dst = Bs_base + ((innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
            if (g_row < K && g_col + 3 < N && (ldb % 4 == 0)) {
                const float* src = B_block + (innerRowB + offset) * ldb + innerColB * 4;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                Bs[(innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 0] = (g_row < K && g_col + 0 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 1] = (g_row < K && g_col + 1 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 2] = (g_row < K && g_col + 2 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 3] = (g_row < K && g_col + 3 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 3] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        // Compute.
        for (int dotIdx = 0; dotIdx < SGB_NARROW_BK; ++dotIdx) {
            for (int i = 0; i < SGB_NARROW_TM; ++i) {
                regM[i] = As[dotIdx * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_NARROW_WM + threadRowInWarp * SGB_NARROW_TM + i];
            }
            for (int i = 0; i < SGB_NARROW_TN; ++i) {
                regN[i] = Bs[dotIdx * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_NARROW_WN + threadColInWarp * SGB_NARROW_TN + i];
            }
            // explicit __fmaf_rn for bit-exact
            // match with CPU `_mm256_fmadd_ps`. sgemm_bi_nn_narrow.
            for (int resIdxM = 0; resIdxM < SGB_NARROW_TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < SGB_NARROW_TN; ++resIdxN) {
                    int idx = resIdxM * SGB_NARROW_TN + resIdxN;
                    threadResults[idx] = __fmaf_rn(
                        regM[resIdxM], regN[resIdxN], threadResults[idx]);
                }
            }
        }

        A_block += SGB_NARROW_BK;
        B_block += SGB_NARROW_BK * ldb;
        __syncthreads();
    }

    // Epilogue: write with alpha, beta. Bias is ABSORBED into threadResults
    // pre-K-loop init above to match CPU FMA-chain order (bias-bit-exact fix
    // Bias is pre-seeded above. Do not add it again here.
    // Scalar N-fallback for non-%4 N (e.g. N=25).
    for (int resIdxM = 0; resIdxM < SGB_NARROW_TM; ++resIdxM) {
        int g_row = pid_m * SGB_NARROW_BM + warpRow * SGB_NARROW_WM + threadRowInWarp * SGB_NARROW_TM + resIdxM;
        if (g_row >= M) continue;
        for (int resIdxN = 0; resIdxN < SGB_NARROW_TN; ++resIdxN) {
            int g_col = pid_n * SGB_NARROW_BN + warpCol * SGB_NARROW_WN + threadColInWarp * SGB_NARROW_TN + resIdxN;
            if (g_col >= N) continue;
            int idx = resIdxM * SGB_NARROW_TN + resIdxN;
            float val = alpha * threadResults[idx];
            if (beta != 0.0f) val += beta * C_warp[(threadRowInWarp * SGB_NARROW_TM + resIdxM) * ldc + threadColInWarp * SGB_NARROW_TN + resIdxN];
            C_warp[(threadRowInWarp * SGB_NARROW_TM + resIdxM) * ldc + threadColInWarp * SGB_NARROW_TN + resIdxN] = val;
        }
    }
    __syncthreads();
    } // end persistent CTA loop (nn_narrow)
}

#undef SGB_NARROW_BM
#undef SGB_NARROW_BN
#undef SGB_NARROW_BK
#undef SGB_NARROW_WM
#undef SGB_NARROW_WN
#undef SGB_NARROW_WMITER
#undef SGB_NARROW_WNITER
#undef SGB_NARROW_TM
#undef SGB_NARROW_TN
#undef SGB_NARROW_NUM_THREADS
#undef SGB_NARROW_WSUBM
#undef SGB_NARROW_WSUBN
#undef SGB_NARROW_ROW_STRIDE_A
#undef SGB_NARROW_ROW_STRIDE_B

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
// SGB_NARROW_SMALL_BM=16, SGB_NARROW_SMALL_BN=16 the grid becomes ceil(64/16) × ceil(25/16) = 4 × 2 = 8
// CTAs → 8× SM utilization. Expected ~2.5-3× speedup, matching cuBLAS f32.
//
// Tile: SGB_NARROW_SMALL_BM=16 SGB_NARROW_SMALL_BN=16 SGB_NARROW_SMALL_BK=16. 64 threads (2 warps).
// Per-warp: SGB_SCALAR_WM=8 SGB_SCALAR_WN=16 (1 warpRow × 2 warpCol per warp grid → 2 warps).
// Per-thread micro-tile: SGB_SCALAR_TM=2 SGB_SCALAR_TN=2 (32 threads/warp via (SGB_SCALAR_WSUBM/SGB_SCALAR_TM)·(SGB_SCALAR_WSUBN/SGB_SCALAR_TN)
// = (8/2)·(16/2) = 4·8 = 32 = SGB_SCALAR_WARP_SIZE).
// Smem: As [16·16] + Bs [16·16] = 2 KiB/CTA.
#define SGB_NARROW_SMALL_BM 16
#define SGB_NARROW_SMALL_BN 16
#define SGB_NARROW_SMALL_BK 16
#define SGB_NARROW_SMALL_WM 8
#define SGB_NARROW_SMALL_WN 16
#define SGB_NARROW_SMALL_WMITER 1
#define SGB_NARROW_SMALL_WNITER 1
#define SGB_NARROW_SMALL_TM 2
#define SGB_NARROW_SMALL_TN 2
#define SGB_NARROW_SMALL_NUM_THREADS 64
#define SGB_NARROW_SMALL_WSUBM (SGB_NARROW_SMALL_WM / SGB_NARROW_SMALL_WMITER)   // 8
#define SGB_NARROW_SMALL_WSUBN (SGB_NARROW_SMALL_WN / SGB_NARROW_SMALL_WNITER)   // 16
#define SGB_NARROW_SMALL_ROW_STRIDE_A ((SGB_NARROW_SMALL_NUM_THREADS * 4) / SGB_NARROW_SMALL_BK)  // 16
#define SGB_NARROW_SMALL_ROW_STRIDE_B (SGB_NARROW_SMALL_NUM_THREADS / (SGB_NARROW_SMALL_BN / 4))  // 16

extern "C" __global__ __launch_bounds__(SGB_NARROW_SMALL_NUM_THREADS, 8)
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
    __shared__ float As[SGB_NARROW_SMALL_BK * (SGB_NARROW_SMALL_BM + SGB_SCALAR_SMEM_A_PAD)];
    __shared__ float Bs[SGB_NARROW_SMALL_BK * (SGB_NARROW_SMALL_BN + SGB_SCALAR_SMEM_B_PAD)];

    int num_pid_m = (M + SGB_NARROW_SMALL_BM - 1) / SGB_NARROW_SMALL_BM;
    int num_pid_n = (N + SGB_NARROW_SMALL_BN - 1) / SGB_NARROW_SMALL_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_NARROW_SMALL_BN / SGB_NARROW_SMALL_WN);  // 0 or 1
    int warpRow = warpIdx / (SGB_NARROW_SMALL_BN / SGB_NARROW_SMALL_WN);  // 0
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_NARROW_SMALL_WSUBN / SGB_NARROW_SMALL_TN);  // 0..7
    int threadRowInWarp = tidInWarp / (SGB_NARROW_SMALL_WSUBN / SGB_NARROW_SMALL_TN);  // 0..3

    int innerRowA = threadIdx.x / (SGB_NARROW_SMALL_BK / 4);  // tid / 4, 0..15
    int innerColA = threadIdx.x % (SGB_NARROW_SMALL_BK / 4);  // tid % 4, 0..3
    int innerRowB = threadIdx.x / (SGB_NARROW_SMALL_BN / 4);  // tid / 4, 0..15
    int innerColB = threadIdx.x % (SGB_NARROW_SMALL_BN / 4);  // tid % 4, 0..3

    float regM[SGB_NARROW_SMALL_WMITER * SGB_NARROW_SMALL_TM] = {0.0f};
    float regN[SGB_NARROW_SMALL_WNITER * SGB_NARROW_SMALL_TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    int tile_id = blockIdx.x;
    {
        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        const float* A_block = A + pid_m * SGB_NARROW_SMALL_BM * lda;
        const float* B_block = B + pid_n * SGB_NARROW_SMALL_BN;
        float* C_warp = C + (pid_m * SGB_NARROW_SMALL_BM + warpRow * SGB_NARROW_SMALL_WM) * ldc + pid_n * SGB_NARROW_SMALL_BN + warpCol * SGB_NARROW_SMALL_WN;

        // Bias pre-seed matches sgemm_bi_nn_narrow
        // semantic): seed threadResults with bias[g_col] BEFORE K-loop so FMA
        // chain begins with `(bias + A[m,0]*B[0,n])` to match CPU order. The
        // per-output FMA chain is identical to sgemm_bi_nn_narrow regardless
        // of tile size. Only valid for alpha=1 (training forward calls).
        float threadResults[SGB_NARROW_SMALL_WMITER * SGB_NARROW_SMALL_TM * SGB_NARROW_SMALL_WNITER * SGB_NARROW_SMALL_TN];
        #pragma unroll
        for (int rm = 0; rm < SGB_NARROW_SMALL_TM; ++rm) {
            int g_col_base = pid_n * SGB_NARROW_SMALL_BN + warpCol * SGB_NARROW_SMALL_WN + threadColInWarp * SGB_NARROW_SMALL_TN;
            #pragma unroll
            for (int rn = 0; rn < SGB_NARROW_SMALL_TN; ++rn) {
                int g_col = g_col_base + rn;
                int idx = rm * SGB_NARROW_SMALL_TN + rn;
                threadResults[idx] =
                    (bias != nullptr && g_col < N) ? bias[g_col] : 0.0f;
            }
        }

    for (int bkIdx = 0; bkIdx < K; bkIdx += SGB_NARROW_SMALL_BK) {
        // A coalesce mirror: warps load disjoint M-row slabs.
        // WARPS=2, M_ROWS_PER_WARP_INST=SGB_SCALAR_WARP_SIZE/SGB_NARROW_SMALL_BK=2,
        // M_ROWS_PER_WARP=SGB_NARROW_SMALL_BM/WARPS=8, INSTR_PER_WARP=4.
        {
            constexpr int WARPS_NS_NR = SGB_NARROW_SMALL_NUM_THREADS / SGB_SCALAR_WARP_SIZE;            // 2
            constexpr int M_ROWS_PER_WARP_INST_NS_NR = SGB_SCALAR_WARP_SIZE / SGB_NARROW_SMALL_BK;       // 2
            constexpr int M_ROWS_PER_WARP_NS_NR = SGB_NARROW_SMALL_BM / WARPS_NS_NR;         // 8
            constexpr int INSTR_PER_WARP_NS_NR =
                M_ROWS_PER_WARP_NS_NR / M_ROWS_PER_WARP_INST_NS_NR;            // 4
            static_assert(SGB_SCALAR_WARP_SIZE % SGB_NARROW_SMALL_BK == 0, "SGB_SCALAR_WARP_SIZE divisible by SGB_NARROW_SMALL_BK (narrow small)");
            static_assert(SGB_NARROW_SMALL_BM % WARPS_NS_NR == 0, "SGB_NARROW_SMALL_BM divisible by warps (narrow small)");
            int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;
            int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;
            int _m_in_warp_ns_nr = _lane / SGB_NARROW_SMALL_BK;
            int _k_local_lane_ns_nr = _lane % SGB_NARROW_SMALL_BK;
            #pragma unroll
            for (int _it = 0; _it < INSTR_PER_WARP_NS_NR; _it++) {
                int _m_local = _warp * M_ROWS_PER_WARP_NS_NR
                               + _it * M_ROWS_PER_WARP_INST_NS_NR + _m_in_warp_ns_nr;
                int _g_row = pid_m * SGB_NARROW_SMALL_BM + _m_local;
                int _g_col = bkIdx + _k_local_lane_ns_nr;
                unsigned _dst = As_base
                    + (_k_local_lane_ns_nr * (SGB_NARROW_SMALL_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local)
                    * (unsigned)sizeof(float);
                if (_g_row < M && _g_col < K) {
                    const float* _src = A_block + _m_local * lda + _k_local_lane_ns_nr;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                                 :: "r"(_dst), "l"(_src));
                } else {
                    As[_k_local_lane_ns_nr * (SGB_NARROW_SMALL_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local] = 0.0f;
                }
            }
        }

        // Load B: cp.async.16B contiguous. SGB_NARROW_SMALL_ROW_STRIDE_B=16 = SGB_NARROW_SMALL_BK → single iter.
        for (int offset = 0; offset + SGB_NARROW_SMALL_ROW_STRIDE_B <= SGB_NARROW_SMALL_BK; offset += SGB_NARROW_SMALL_ROW_STRIDE_B) {
            int g_row = bkIdx + innerRowB + offset;
            int g_col = pid_n * SGB_NARROW_SMALL_BN + innerColB * 4;
            unsigned dst = Bs_base + ((innerRowB + offset) * (SGB_NARROW_SMALL_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
            if (g_row < K && g_col + 3 < N && (ldb % 4 == 0)) {
                const float* src = B_block + (innerRowB + offset) * ldb + innerColB * 4;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                Bs[(innerRowB + offset) * (SGB_NARROW_SMALL_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 0] = (g_row < K && g_col + 0 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_NARROW_SMALL_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 1] = (g_row < K && g_col + 1 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_NARROW_SMALL_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 2] = (g_row < K && g_col + 2 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_NARROW_SMALL_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 3] = (g_row < K && g_col + 3 < N) ? B_block[(innerRowB + offset) * ldb + innerColB * 4 + 3] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        // Compute. Identical per-output ascending K __fmaf_rn chain as
        // sgemm_bi_nn_narrow — bit-exact f32 output regardless of tile size.
        for (int dotIdx = 0; dotIdx < SGB_NARROW_SMALL_BK; ++dotIdx) {
            for (int i = 0; i < SGB_NARROW_SMALL_TM; ++i) {
                regM[i] = As[dotIdx * (SGB_NARROW_SMALL_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_NARROW_SMALL_WM + threadRowInWarp * SGB_NARROW_SMALL_TM + i];
            }
            for (int i = 0; i < SGB_NARROW_SMALL_TN; ++i) {
                regN[i] = Bs[dotIdx * (SGB_NARROW_SMALL_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_NARROW_SMALL_WN + threadColInWarp * SGB_NARROW_SMALL_TN + i];
            }
            // explicit __fmaf_rn for bit-exact match with CPU
            // `f32::mul_add` ascending K — same chain as sgemm_bi_nn_narrow.
            for (int resIdxM = 0; resIdxM < SGB_NARROW_SMALL_TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < SGB_NARROW_SMALL_TN; ++resIdxN) {
                    int idx = resIdxM * SGB_NARROW_SMALL_TN + resIdxN;
                    threadResults[idx] = __fmaf_rn(
                        regM[resIdxM], regN[resIdxN], threadResults[idx]);
                }
            }
        }

        A_block += SGB_NARROW_SMALL_BK;
        B_block += SGB_NARROW_SMALL_BK * ldb;
        __syncthreads();
    }

    // Epilogue: bias-IN-FMA already absorbed via pre-K-loop seed. Scalar N
    // fallback for non-%4 N (e.g. N=25). Same write path as sgemm_bi_nn_narrow.
    for (int resIdxM = 0; resIdxM < SGB_NARROW_SMALL_TM; ++resIdxM) {
        int g_row = pid_m * SGB_NARROW_SMALL_BM + warpRow * SGB_NARROW_SMALL_WM + threadRowInWarp * SGB_NARROW_SMALL_TM + resIdxM;
        if (g_row >= M) continue;
        for (int resIdxN = 0; resIdxN < SGB_NARROW_SMALL_TN; ++resIdxN) {
            int g_col = pid_n * SGB_NARROW_SMALL_BN + warpCol * SGB_NARROW_SMALL_WN + threadColInWarp * SGB_NARROW_SMALL_TN + resIdxN;
            if (g_col >= N) continue;
            int idx = resIdxM * SGB_NARROW_SMALL_TN + resIdxN;
            float val = alpha * threadResults[idx];
            if (beta != 0.0f) val += beta * C_warp[(threadRowInWarp * SGB_NARROW_SMALL_TM + resIdxM) * ldc + threadColInWarp * SGB_NARROW_SMALL_TN + resIdxN];
            C_warp[(threadRowInWarp * SGB_NARROW_SMALL_TM + resIdxM) * ldc + threadColInWarp * SGB_NARROW_SMALL_TN + resIdxN] = val;
        }
    }
    __syncthreads();
    } // end persistent CTA loop (nn_narrow_small)
}

#undef SGB_NARROW_SMALL_BM
#undef SGB_NARROW_SMALL_BN
#undef SGB_NARROW_SMALL_BK
#undef SGB_NARROW_SMALL_WM
#undef SGB_NARROW_SMALL_WN
#undef SGB_NARROW_SMALL_WMITER
#undef SGB_NARROW_SMALL_WNITER
#undef SGB_NARROW_SMALL_TM
#undef SGB_NARROW_SMALL_TN
#undef SGB_NARROW_SMALL_NUM_THREADS
#undef SGB_NARROW_SMALL_WSUBM
#undef SGB_NARROW_SMALL_WSUBN
#undef SGB_NARROW_SMALL_ROW_STRIDE_A
#undef SGB_NARROW_SMALL_ROW_STRIDE_B

// ============================================================================
// Narrow-N SGB_SCALAR_TN (backward dW, N∈9..48): C[K,N] += alpha * A^T[K,M] @ B[M,N]
// ============================================================================
// A = X_saved [M, K_out] — read transposed into As
// B = dY [M, N]
// C = dW [K_out, N] — accumulated (beta=1)
#define SGB_NARROW_BM 64
#define SGB_NARROW_BN 32
#define SGB_NARROW_BK 16
#define SGB_NARROW_WM 32
#define SGB_NARROW_WN 16
#define SGB_NARROW_TM 4
#define SGB_NARROW_TN 4
#define SGB_NARROW_NUM_THREADS 128
#define SGB_NARROW_WSUBN 16
#define SGB_NARROW_ROW_STRIDE_A ((SGB_NARROW_NUM_THREADS * 4) / SGB_NARROW_BK)
#define SGB_NARROW_ROW_STRIDE_B (SGB_NARROW_NUM_THREADS / (SGB_NARROW_BN / 4))

extern "C" __global__ __launch_bounds__(SGB_NARROW_NUM_THREADS, 4)
void sgemm_bi_tn_narrow(
    float* __restrict__ C,         // [K_out, N]
    const float* __restrict__ A,   // [M_red, K_out]
    const float* __restrict__ B,   // [M_red, N]
    float alpha,
    int M_red, int K_out, int N
) {
    __shared__ float As[SGB_NARROW_BK * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD)];
    __shared__ float Bs[SGB_NARROW_BK * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD)];

    int num_pid_m = (K_out + SGB_NARROW_BM - 1) / SGB_NARROW_BM;
    int num_pid_n = (N + SGB_NARROW_BN - 1) / SGB_NARROW_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_NARROW_BN / SGB_NARROW_WN);
    int warpRow = warpIdx / (SGB_NARROW_BN / SGB_NARROW_WN);
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_NARROW_WSUBN / SGB_NARROW_TN);
    int threadRowInWarp = tidInWarp / (SGB_NARROW_WSUBN / SGB_NARROW_TN);

    int innerRowA = threadIdx.x / (SGB_NARROW_BK / 4);  // tid/4 0..31
    int innerColA = threadIdx.x % (SGB_NARROW_BK / 4);  // tid%4 0..3
    int innerRowB = threadIdx.x / (SGB_NARROW_BN / 4);  // tid/8 0..15
    int innerColB = threadIdx.x % (SGB_NARROW_BN / 4);  // tid%8 0..7

    float regM[SGB_NARROW_TM] = {0.0f};
    float regN[SGB_NARROW_TN] = {0.0f};

    // narrow SGB_SCALAR_TN coalesce: convert A-loader from direct
    // global reads to cp.async with warp-cooperative contiguous loads.
    unsigned As_base = __cvta_generic_to_shared(As);

    // persistent CTA loop for tn_narrow.
    // Direct CTA mapping avoids the persistent-loop overhead. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring lower register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[SGB_NARROW_TM * SGB_NARROW_TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        float* C_warp = C + (pid_m * SGB_NARROW_BM + warpRow * SGB_NARROW_WM) * N + pid_n * SGB_NARROW_BN + warpCol * SGB_NARROW_WN;

    for (int mIdx = 0; mIdx < M_red; mIdx += SGB_NARROW_BK) {
        // narrow SGB_SCALAR_TN A-loader: cp.async coalesced (16B/lane).
        // 4 warps × 16 lanes per row × 2 rows per inst × 2 inst/warp = 4 SGB_NARROW_BK rows × 4 = 16 SGB_NARROW_BK rows ✓
        // Source X[m_red, k_out..+3] contig in K_out. Dest As[m_red][k_out..+3] contig in inner.
        // As layout = [SGB_NARROW_BK outer × SGB_NARROW_BM inner], 100% cache line util.
        {
            constexpr int WARPS_TN_NR = SGB_NARROW_NUM_THREADS / SGB_SCALAR_WARP_SIZE;        // 4
            constexpr int LANES_PER_ROW_TN_NR = SGB_NARROW_BM / 4;                // 16
            constexpr int ROWS_PER_INST_TN_NR = SGB_SCALAR_WARP_SIZE / LANES_PER_ROW_TN_NR; // 2
            constexpr int ROWS_PER_WARP_TN_NR = SGB_NARROW_BK / WARPS_TN_NR;      // 4
            constexpr int INSTR_PER_WARP_TN_NR =
                ROWS_PER_WARP_TN_NR / ROWS_PER_INST_TN_NR;               // 2
            int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;
            int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;
            int _row_in_warp = _lane / LANES_PER_ROW_TN_NR;
            int _col_chunk = (_lane % LANES_PER_ROW_TN_NR) * 4;
            #pragma unroll
            for (int _it = 0; _it < INSTR_PER_WARP_TN_NR; _it++) {
                int _k_outer = _warp * ROWS_PER_WARP_TN_NR
                               + _it * ROWS_PER_INST_TN_NR + _row_in_warp;
                int _m_inner = _col_chunk;
                int _g_m = mIdx + _k_outer;
                int _g_k = pid_m * SGB_NARROW_BM + _m_inner;
                unsigned _dst = As_base
                    + (_k_outer * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD) + _m_inner)
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
                            As[_k_outer * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD) + _m_inner + _i] = 0.0f;
                        }
                    }
                }
            }
        }

        // Load B: dY[mIdx+m_local, pid_n*SGB_NARROW_BN + n_local]
        for (int offset = 0; offset + SGB_NARROW_ROW_STRIDE_B <= SGB_NARROW_BK; offset += SGB_NARROW_ROW_STRIDE_B) {
            int g_m = mIdx + innerRowB + offset;
            int g_n = pid_n * SGB_NARROW_BN + innerColB * 4;
            if (g_m < M_red && g_n + 3 < N && (N % 4 == 0)) {
                reinterpret_cast<float4*>(&Bs[(innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4])[0] =
                    ld_global_L2_128B(&B[g_m * N + pid_n * SGB_NARROW_BN + innerColB * 4]);
            } else {
                Bs[(innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 0] = (g_m < M_red && g_n + 0 < N) ? B[g_m * N + pid_n * SGB_NARROW_BN + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 1] = (g_m < M_red && g_n + 1 < N) ? B[g_m * N + pid_n * SGB_NARROW_BN + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 2] = (g_m < M_red && g_n + 2 < N) ? B[g_m * N + pid_n * SGB_NARROW_BN + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 3] = (g_m < M_red && g_n + 3 < N) ? B[g_m * N + pid_n * SGB_NARROW_BN + innerColB * 4 + 3] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        for (int dotIdx = 0; dotIdx < SGB_NARROW_BK; ++dotIdx) {
            for (int i = 0; i < SGB_NARROW_TM; ++i) {
                regM[i] = As[dotIdx * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_NARROW_WM + threadRowInWarp * SGB_NARROW_TM + i];
            }
            for (int i = 0; i < SGB_NARROW_TN; ++i) {
                regN[i] = Bs[dotIdx * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_NARROW_WN + threadColInWarp * SGB_NARROW_TN + i];
            }
            for (int rm = 0; rm < SGB_NARROW_TM; ++rm) {
                for (int rn = 0; rn < SGB_NARROW_TN; ++rn) {
                    // explicit __fmaf_rn for
                    // bit-exact match with CPU `_mm256_fmadd_ps`. nvcc may
                    // emit FMUL+FADD (two roundings) for `+= a*b` depending
                    // on compile flags; explicit FMA forces single-rounding.
                    threadResults[rm * SGB_NARROW_TN + rn] = __fmaf_rn(
                        regM[rm], regN[rn], threadResults[rm * SGB_NARROW_TN + rn]);
                }
            }
        }
        __syncthreads();
    }

    // Epilogue — accumulate (beta=1), scalar N-fallback
    for (int rm = 0; rm < SGB_NARROW_TM; ++rm) {
        int g_row = pid_m * SGB_NARROW_BM + warpRow * SGB_NARROW_WM + threadRowInWarp * SGB_NARROW_TM + rm;
        if (g_row >= K_out) continue;
        for (int rn = 0; rn < SGB_NARROW_TN; ++rn) {
            int g_col = pid_n * SGB_NARROW_BN + warpCol * SGB_NARROW_WN + threadColInWarp * SGB_NARROW_TN + rn;
            if (g_col >= N) continue;
            C_warp[(threadRowInWarp * SGB_NARROW_TM + rm) * N + threadColInWarp * SGB_NARROW_TN + rn] +=
                alpha * threadResults[rm * SGB_NARROW_TN + rn];
        }
    }
    __syncthreads();
    } // end persistent CTA loop (tn_narrow)
}

// ============================================================================
// Phase C-1.5bc — Narrow-N SGB_SCALAR_TN Split-M partial (backward dW for N<128 + large M).
//
// Mirrors sgemm_bi_tn_splitm_partial structure but with narrow-N tile params
// (SGB_NARROW_BM=64, SGB_NARROW_BN=32, SGB_NARROW_BK=16, SGB_NARROW_WM=32, SGB_NARROW_WN=16, SGB_NARROW_TM=4, SGB_NARROW_TN=4, SGB_NARROW_NUM_THREADS=128) so
// it covers TQC quantile head (N=25), C-loss head and any N ∈ [2..127].
//
// Why: at v6.5 multi-step + critic quantile head bwd_dw shape (M=batch*learn_len,
// K_out=hidden=256, N=25), plain sgemm_bi_tn_narrow grid = K_tiles*N_tiles =
// ceil(256/64) * ceil(25/32) = 4 * 1 = 4 blocks → ~3% SM utilization on Ada
// (142 SMs). Split-M inflates the grid F× along z by partitioning the M-reduction
// axis into F chunks, each handled by an independent block — same wave-fill
// strategy as the regular Split-M SGB_SCALAR_TN at sgemm_bi_tn_splitm_partial.
//
// Bit-exact contract:
//   - F (split factor) is shape-keyed: caller computes F = div_ceil(2*NUM_SMS,
//     base_blocks) capped by scratch budget → identical F on every call with
//     same (M, K_out, N) → identical reduction tree.
//   - Per-chunk FMA chain inside this kernel uses __fmaf_rn (single-rounding,
//     same as plain sgemm_bi_tn_narrow). Per-chunk partial bit-
//     exact identical to the narrow kernel's M-restricted output.
//   - Each (fc, k, n) slot has exactly ONE writer (block coords (k_tile,
//     n_tile, fc)) → no atomics, no race.
//   - OVERWRITE (no accumulate, no alpha, no bias): reducer applies alpha at
//     the final sum and += into the caller's dW slot. Same contract as
//     sgemm_bi_tn_splitm_partial.
//
// Caller uses the existing sgemm_bi_splitm_reduce kernel; its math is
// shape-agnostic on (K_out, N) and operates per output cell. Reducer applies
// alpha and accumulates into dW (+=) matching the narrow kernel's beta=1 semantic.
// ============================================================================
extern "C" __global__ __launch_bounds__(SGB_NARROW_NUM_THREADS, 4)
void sgemm_bi_tn_narrow_splitm_partial(
    float* __restrict__ partial,   // [F * K_out * N] — unique slot per (fc, ...)
    const float* __restrict__ A,   // X [M_red, K_out]
    const float* __restrict__ B,   // dY [M_red, N]
    int M_red, int K_out, int N,
    int M_CHUNK                    // chunk size (multiple of SGB_NARROW_BK)
) {
    __shared__ float As[SGB_NARROW_BK * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD)];
    __shared__ float Bs[SGB_NARROW_BK * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD)];

    int num_pid_m = (K_out + SGB_NARROW_BM - 1) / SGB_NARROW_BM;
    int num_pid_n = (N + SGB_NARROW_BN - 1) / SGB_NARROW_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;
    int fc = blockIdx.z;
    int m_begin = fc * M_CHUNK;
    int m_end = min(m_begin + M_CHUNK, M_red);
    if (m_begin >= M_red) return;

    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_NARROW_BN / SGB_NARROW_WN);
    int warpRow = warpIdx / (SGB_NARROW_BN / SGB_NARROW_WN);
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_NARROW_WSUBN / SGB_NARROW_TN);
    int threadRowInWarp = tidInWarp / (SGB_NARROW_WSUBN / SGB_NARROW_TN);

    int innerRowA = threadIdx.x / (SGB_NARROW_BK / 4);  // 0..31
    int innerColA = threadIdx.x % (SGB_NARROW_BK / 4);  // 0..3
    int innerRowB = threadIdx.x / (SGB_NARROW_BN / 4);  // 0..15
    int innerColB = threadIdx.x % (SGB_NARROW_BN / 4);  // 0..7

    float regM[SGB_NARROW_TM] = {0.0f};
    float regN[SGB_NARROW_TN] = {0.0f};

    // persistent CTA loop for tn_narrow_splitm_partial.
    // Direct CTA mapping avoids the persistent-loop overhead. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring lower register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[SGB_NARROW_TM * SGB_NARROW_TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    // cp.async (CUTLASS SM80 pattern, mirrors sgemm_bi_tn_splitm_partial lines
    // 613-651). Single-stage with wait_all — adequate at SGB_NARROW_BK=16 small K-chunk.
    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    for (int mIdx = m_begin; mIdx < m_end; mIdx += SGB_NARROW_BK) {
        // narrow SGB_SCALAR_TN splitm coalesce: 4 warps × 16 lanes per row
        // × 2 rows per warp inst at 100% cache line util (vs 12.5% legacy).
        // SGB_NARROW_BM=64 K_out cols, SGB_NARROW_BK=16 M_red rows. Source X[m_red, k_out..+3] contig.
        {
            constexpr int WARPS_TN_NSP = SGB_NARROW_NUM_THREADS / SGB_SCALAR_WARP_SIZE;       // 4
            constexpr int LANES_PER_ROW_TN_NSP = SGB_NARROW_BM / 4;               // 16
            constexpr int ROWS_PER_INST_TN_NSP = SGB_SCALAR_WARP_SIZE / LANES_PER_ROW_TN_NSP; // 2
            constexpr int ROWS_PER_WARP_TN_NSP = SGB_NARROW_BK / WARPS_TN_NSP;    // 4
            constexpr int INSTR_PER_WARP_TN_NSP =
                ROWS_PER_WARP_TN_NSP / ROWS_PER_INST_TN_NSP;             // 2
            static_assert(SGB_NARROW_BM % (SGB_SCALAR_WARP_SIZE / (SGB_SCALAR_WARP_SIZE / (SGB_NARROW_BM / 4))) == 0,
                "narrow SGB_SCALAR_TN splitm: lane/row config invariant");
            int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;
            int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;
            int _row_in_warp = _lane / LANES_PER_ROW_TN_NSP;
            int _col_chunk = (_lane % LANES_PER_ROW_TN_NSP) * 4;
            #pragma unroll
            for (int _it = 0; _it < INSTR_PER_WARP_TN_NSP; _it++) {
                int _k_local = _warp * ROWS_PER_WARP_TN_NSP
                               + _it * ROWS_PER_INST_TN_NSP + _row_in_warp;
                int _m_local = _col_chunk;
                int _g_m = mIdx + _k_local;
                int _g_k = pid_m * SGB_NARROW_BM + _m_local;
                unsigned _dst = As_base
                    + (_k_local * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local)
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
                            As[_k_local * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local + _i] = 0.0f;
                        }
                    }
                }
            }
        }
        // Load B tile (cp.async float4 when aligned, scalar fallback otherwise).
        for (int offset = 0; offset + SGB_NARROW_ROW_STRIDE_B <= SGB_NARROW_BK; offset += SGB_NARROW_ROW_STRIDE_B) {
            int g_m = mIdx + innerRowB + offset;
            int g_n = pid_n * SGB_NARROW_BN + innerColB * 4;
            unsigned dst = Bs_base + ((innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
            if (g_m < m_end && g_n + 3 < N && (N % 4 == 0)) {
                const float* src = B + ((long long)g_m) * N + pid_n * SGB_NARROW_BN + innerColB * 4;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                Bs[(innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 0] = (g_m < m_end && g_n + 0 < N) ? B[g_m * N + pid_n * SGB_NARROW_BN + innerColB * 4 + 0] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 1] = (g_m < m_end && g_n + 1 < N) ? B[g_m * N + pid_n * SGB_NARROW_BN + innerColB * 4 + 1] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 2] = (g_m < m_end && g_n + 2 < N) ? B[g_m * N + pid_n * SGB_NARROW_BN + innerColB * 4 + 2] : 0.0f;
                Bs[(innerRowB + offset) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 3] = (g_m < m_end && g_n + 3 < N) ? B[g_m * N + pid_n * SGB_NARROW_BN + innerColB * 4 + 3] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        for (int dotIdx = 0; dotIdx < SGB_NARROW_BK; ++dotIdx) {
            for (int i = 0; i < SGB_NARROW_TM; ++i) {
                regM[i] = As[dotIdx * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_NARROW_WM + threadRowInWarp * SGB_NARROW_TM + i];
            }
            for (int i = 0; i < SGB_NARROW_TN; ++i) {
                regN[i] = Bs[dotIdx * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_NARROW_WN + threadColInWarp * SGB_NARROW_TN + i];
            }
            // identical to sgemm_bi_tn_narrow. Per-
            // chunk FMA order matches the narrow kernel exactly → partial[fc]
            // is bit-exact with what narrow would compute on its M-restricted
            // slice. Reducer ascending-fc sum is the only new rounding step,
            // and it's identical for every (M_red, K_out, N) shape.
            for (int rm = 0; rm < SGB_NARROW_TM; ++rm) {
                for (int rn = 0; rn < SGB_NARROW_TN; ++rn) {
                    threadResults[rm * SGB_NARROW_TN + rn] = __fmaf_rn(
                        regM[rm], regN[rn], threadResults[rm * SGB_NARROW_TN + rn]);
                }
            }
        }
        __syncthreads();
    }

    // Epilogue: OVERWRITE partial[fc, :, :] (no alpha, no bias, no accumulate).
    // Float4 STG.128 fast path when N % 4 == 0 AND 4 lanes stay within N tile
    // — matches the Big splitm_partial epilogue. At N=25 (TQC quantile
    // head, the primary target shape) N%4 != 0 → always scalar tail.
    float* partial_chunk = partial + (long long)fc * K_out * N;
    float* partial_warp = partial_chunk + (pid_m * SGB_NARROW_BM + warpRow * SGB_NARROW_WM) * N + pid_n * SGB_NARROW_BN + warpCol * SGB_NARROW_WN;
    const bool n_aligned = (N & 3) == 0;

    for (int rm = 0; rm < SGB_NARROW_TM; ++rm) {
        int g_row = pid_m * SGB_NARROW_BM + warpRow * SGB_NARROW_WM + threadRowInWarp * SGB_NARROW_TM + rm;
        if (g_row >= K_out) continue;
        int g_col_base = pid_n * SGB_NARROW_BN + warpCol * SGB_NARROW_WN + threadColInWarp * SGB_NARROW_TN;
        // SGB_NARROW_TN == 4: try float4 store.
        if (n_aligned && g_col_base + 3 < N) {
            float4 out;
            out.x = threadResults[rm * SGB_NARROW_TN + 0];
            out.y = threadResults[rm * SGB_NARROW_TN + 1];
            out.z = threadResults[rm * SGB_NARROW_TN + 2];
            out.w = threadResults[rm * SGB_NARROW_TN + 3];
            reinterpret_cast<float4*>(
                &partial_warp[(threadRowInWarp * SGB_NARROW_TM + rm) * N + threadColInWarp * SGB_NARROW_TN])[0] = out;
        } else {
            #pragma unroll
            for (int rn = 0; rn < SGB_NARROW_TN; ++rn) {
                int g_col = g_col_base + rn;
                if (g_col >= N) continue;
                partial_warp[(threadRowInWarp * SGB_NARROW_TM + rm) * N + threadColInWarp * SGB_NARROW_TN + rn] =
                    threadResults[rm * SGB_NARROW_TN + rn];
            }
        }
    }
    __syncthreads();
    } // end persistent CTA loop (tn_narrow_splitm_partial)
}

// ============================================================================
// Narrow-N NT (backward dX, N∈9..48): C[M,K_out] = alpha * A[M,N] @ B^T[N,K_out]
// ============================================================================
// A = dY [M, N]
// B = W [K_out, N] — read transposed as W^T[N, K_out]
// C = dX [M, K_out] — overwrite (beta=0)
extern "C" __global__ __launch_bounds__(SGB_NARROW_NUM_THREADS, 4)
void sgemm_bi_nt_narrow(
    float* __restrict__ C,
    const float* __restrict__ A,   // dY [M, N]
    const float* __restrict__ B,   // W [K_out, N]
    float alpha,
    int M, int N, int K_out
) {
    __shared__ float As[SGB_NARROW_BK * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD)];
    __shared__ float Bs[SGB_NARROW_BK * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD)];

    // Grid: over (M, K_out) — "SGB_SCALAR_BM" = M tile, "SGB_SCALAR_BN" = K_out tile
    int num_pid_m = (M + SGB_NARROW_BM - 1) / SGB_NARROW_BM;
    int num_pid_n = (K_out + SGB_NARROW_BN - 1) / SGB_NARROW_BN;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;
    int total_tiles = num_pid_m * num_pid_n;

    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_NARROW_BN / SGB_NARROW_WN);
    int warpRow = warpIdx / (SGB_NARROW_BN / SGB_NARROW_WN);
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_NARROW_WSUBN / SGB_NARROW_TN);
    int threadRowInWarp = tidInWarp / (SGB_NARROW_WSUBN / SGB_NARROW_TN);

    int innerRowA = threadIdx.x / (SGB_NARROW_BK / 4);
    int innerColA = threadIdx.x % (SGB_NARROW_BK / 4);

    float regM[SGB_NARROW_TM] = {0.0f};
    float regN[SGB_NARROW_TN] = {0.0f};

    // persistent CTA loop for nt_narrow.
    // Direct CTA mapping avoids the persistent-loop overhead. `int tile_id = blockIdx.x;`
    // matches the canonical data-parallel SGEMM (siboehm Kernel 10, CUTLASS
    // Heuristic when total_tiles ≈ sm_count). In our shape regime (total_tiles
    // ≤ 16, sm_count = 80..200 across Ampere/Ada/Hopper/Blackwell) persistent
    // CTA was pure register tax: ptxas held tile_id as a loop-carried induction
    // var, pushing 3 Big kernels to the 128-reg cap of __launch_bounds__(256,2)
    // for +1.0% wall (bisect-confirmed). Constant init lets ptxas SSA-rename
    // tile_id → blockIdx.x at usage points, restoring lower register
    // schedule. Bit-exact: same FMA chain, same per-tile output, same launch
    // semantics (one CTA per output tile via gridDim = total_tiles).
    int tile_id = blockIdx.x;
    {
        float threadResults[SGB_NARROW_TM * SGB_NARROW_TN] = {0.0f};

        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    for (int nIdx = 0; nIdx < N; nIdx += SGB_NARROW_BK) {
        // narrow NT coalesce: 4 warps × 8 instr/warp × 2 rows.
        // M_ROWS_PER_WARP_INST = SGB_SCALAR_WARP_SIZE/SGB_NARROW_BK = 2. Cache util 50% (vs 12.5% legacy).
        // Each lane: 1 float dY[g_m, g_n] → As[n_local][m_local].
        {
            unsigned As_base_nt_nr = __cvta_generic_to_shared(As);
            constexpr int WARPS_NT_NR = SGB_NARROW_NUM_THREADS / SGB_SCALAR_WARP_SIZE;        // 4
            constexpr int M_ROWS_PER_WARP_INST_NT_NR = SGB_SCALAR_WARP_SIZE / SGB_NARROW_BK;  // 2
            constexpr int M_ROWS_PER_WARP_NT_NR = SGB_NARROW_BM / WARPS_NT_NR;    // 16
            constexpr int INSTR_PER_WARP_NT_NR =
                M_ROWS_PER_WARP_NT_NR / M_ROWS_PER_WARP_INST_NT_NR;      // 8
            int _warp = threadIdx.x / SGB_SCALAR_WARP_SIZE;
            int _lane = threadIdx.x % SGB_SCALAR_WARP_SIZE;
            int _m_in_warp_nt_nr = _lane / SGB_NARROW_BK;
            int _n_local_lane_nt_nr = _lane % SGB_NARROW_BK;
            #pragma unroll
            for (int _it = 0; _it < INSTR_PER_WARP_NT_NR; _it++) {
                int _m_local = _warp * M_ROWS_PER_WARP_NT_NR
                               + _it * M_ROWS_PER_WARP_INST_NT_NR + _m_in_warp_nt_nr;
                int _g_m = pid_m * SGB_NARROW_BM + _m_local;
                int _g_n = nIdx + _n_local_lane_nt_nr;
                unsigned _dst = As_base_nt_nr
                    + (_n_local_lane_nt_nr * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local)
                    * (unsigned)sizeof(float);
                if (_g_m < M && _g_n < N) {
                    const float* _src = A + (long long)_g_m * N + _g_n;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                                 :: "r"(_dst), "l"(_src));
                } else {
                    As[_n_local_lane_nt_nr * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD) + _m_local] = 0.0f;
                }
            }
        }

        // Load B^T: W[pid_n*SGB_NARROW_BN + k_local, nIdx + n_local] → Bs[n_local * (SGB_NARROW_BN+PAD) + k_local]
        // Each thread loads 4 B values across N, for a single k = innerRow
        for (int offset = 0; offset + SGB_NARROW_ROW_STRIDE_A <= SGB_NARROW_BN; offset += SGB_NARROW_ROW_STRIDE_A) {
            int k_base = innerRowA + offset;
            int n_local = innerColA * 4;
            int g_k = pid_n * SGB_NARROW_BN + k_base;
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
            Bs[(n_local + 0) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + k_base] = tmp.x;
            Bs[(n_local + 1) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + k_base] = tmp.y;
            Bs[(n_local + 2) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + k_base] = tmp.z;
            Bs[(n_local + 3) * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + k_base] = tmp.w;
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        for (int dotIdx = 0; dotIdx < SGB_NARROW_BK; ++dotIdx) {
            for (int i = 0; i < SGB_NARROW_TM; ++i) {
                regM[i] = As[dotIdx * (SGB_NARROW_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_NARROW_WM + threadRowInWarp * SGB_NARROW_TM + i];
            }
            for (int i = 0; i < SGB_NARROW_TN; ++i) {
                regN[i] = Bs[dotIdx * (SGB_NARROW_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_NARROW_WN + threadColInWarp * SGB_NARROW_TN + i];
            }
            for (int rm = 0; rm < SGB_NARROW_TM; ++rm) {
                for (int rn = 0; rn < SGB_NARROW_TN; ++rn) {
                    // explicit __fmaf_rn for
                    // bit-exact match with CPU `_mm256_fmadd_ps`. nvcc may
                    // emit FMUL+FADD (two roundings) for `+= a*b` depending
                    // on compile flags; explicit FMA forces single-rounding.
                    threadResults[rm * SGB_NARROW_TN + rn] = __fmaf_rn(
                        regM[rm], regN[rn], threadResults[rm * SGB_NARROW_TN + rn]);
                }
            }
        }
        __syncthreads();
    }

    float* C_warp = C + (pid_m * SGB_NARROW_BM + warpRow * SGB_NARROW_WM) * K_out + pid_n * SGB_NARROW_BN + warpCol * SGB_NARROW_WN;

    // Epilogue — overwrite (beta=0), scalar K_out-fallback
    // __stwt streaming store (write-once output, evict-first safe).
    for (int rm = 0; rm < SGB_NARROW_TM; ++rm) {
        int g_row = pid_m * SGB_NARROW_BM + warpRow * SGB_NARROW_WM + threadRowInWarp * SGB_NARROW_TM + rm;
        if (g_row >= M) continue;
        for (int rn = 0; rn < SGB_NARROW_TN; ++rn) {
            int g_col = pid_n * SGB_NARROW_BN + warpCol * SGB_NARROW_WN + threadColInWarp * SGB_NARROW_TN + rn;
            if (g_col >= K_out) continue;
            __stwt(&C_warp[(threadRowInWarp * SGB_NARROW_TM + rm) * K_out + threadColInWarp * SGB_NARROW_TN + rn],
                   alpha * threadResults[rm * SGB_NARROW_TN + rn]);
        }
    }
    __syncthreads();
    } // end persistent CTA loop (nt_narrow)
}

#undef SGB_NARROW_BM
#undef SGB_NARROW_BN
#undef SGB_NARROW_BK
#undef SGB_NARROW_WM
#undef SGB_NARROW_WN
#undef SGB_NARROW_TM
#undef SGB_NARROW_TN
#undef SGB_NARROW_NUM_THREADS
#undef SGB_NARROW_WSUBN
#undef SGB_NARROW_ROW_STRIDE_A
#undef SGB_NARROW_ROW_STRIDE_B

// ============================================================================
// Split-K Thin-M variant — deterministic f32 via fixed tree reduce.
// ============================================================================
// Designed for small-M shapes (M ∈ [32, 127]) where full K-reduction inside a
// single block leaves the grid underfilled (≤8 output tiles vs 142 SMs on Ada).
// Split K into chunks of size 32, run partial GEMM per (m,n,k_chunk) block,
// then a separate reduce kernel does fixed-order tree sum across chunks.
//
// Tile: SGB_SPLITK32_BM=32 SGB_SPLITK32_BN=64 SGB_SPLITK32_BK=32 (one K-iter per block), SGB_SCALAR_WM=16 SGB_SCALAR_WN=32 (2×2 warps),
// SGB_SCALAR_TM=4 SGB_SCALAR_TN=4, SGB_SCALAR_WMITER=SGB_SCALAR_WNITER=1 → per-thread 16 accum, ~24 regs total.
// __launch_bounds__(128, 4) — 4 blocks/SM × 32-block grid for M=64 N=256 K=128
// fills 32/142 SMs in the first wave with minimal per-block work.
//
// Determinism: each block writes its partial to a unique slot in the
// scratch buffer `partial[K_CHUNKS * M * N]`. The reduce kernel sums in
// ascending chunk order (((p0+p1)+p2)+p3+...), identical on every run.
// No atomicAdd. Batch-invariant by construction.
// ============================================================================

#define SGB_SPLITK32_BM 32
#define SGB_SPLITK32_BN 64
#define SGB_SPLITK32_BK 32
#define SGB_SPLITK32_WM 16
#define SGB_SPLITK32_WN 32
#define SGB_SPLITK32_TM 4
#define SGB_SPLITK32_TN 4
#define SGB_SPLITK32_NUM_THREADS 128

// K-bound contract.
// `K_CHUNKS = K_main / SGB_SPLITK32_BK` is the count of FULL chunks the kernel processes;
// it covers columns [0..K_main). Anything past K_main (the tail) is handled
// EXTERNALLY by `sgemm_bi_splitk_reduce` via `x_tail_ptr` / `w_tail_ptr` /
// `tail_cnt`. There is NO in-kernel K-bound runtime check here — the
// dispatcher contract guarantees `K_main ≤ K_full` and the kernel reads
// strictly from [0..K_main). If a future dispatcher change ever passes
// `K_CHUNKS · SGB_SPLITK32_BK > lda`, the kernel will OOB-read silently. Keep the
// dispatcher (`blas_gpu.rs::gpu_sgemm_forward` tier dispatch) honest.
extern "C" __global__ __launch_bounds__(SGB_SPLITK32_NUM_THREADS, 4)
void sgemm_bi_nn_splitk32_partial(
    float* __restrict__ partial,  // [K_CHUNKS * M * N]
    const float* __restrict__ A,  // [M, K_full]
    const float* __restrict__ B,  // [K_full, N]
    int M, int N,
    int K_CHUNKS,                  // = K_main / SGB_SPLITK32_BK  (K_main = K_CHUNKS * SGB_SPLITK32_BK, covers columns [0..K_main))
    int lda                         // actual A row stride in floats (= K_full; can differ from K_main when a tail lives at K ≥ K_main)
) {
    __shared__ float As[SGB_SPLITK32_BK * (SGB_SPLITK32_BM + SGB_SCALAR_SMEM_A_PAD)];
    __shared__ float Bs[SGB_SPLITK32_BK * (SGB_SPLITK32_BN + SGB_SCALAR_SMEM_B_PAD)];

    int num_pid_m = (M + SGB_SPLITK32_BM - 1) / SGB_SPLITK32_BM;
    int num_pid_n = (N + SGB_SPLITK32_BN - 1) / SGB_SPLITK32_BN;
    int total_mn = num_pid_m * num_pid_n;
    int total_blocks = K_CHUNKS * total_mn;
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;

    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;
    int warpCol = warpIdx % (SGB_SPLITK32_BN / SGB_SPLITK32_WN);
    int warpRow = warpIdx / (SGB_SPLITK32_BN / SGB_SPLITK32_WN);
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;
    int threadColInWarp = tidInWarp % (SGB_SPLITK32_WN / SGB_SPLITK32_TN);
    int threadRowInWarp = tidInWarp / (SGB_SPLITK32_WN / SGB_SPLITK32_TN);

    int innerRowA = threadIdx.x / (SGB_SPLITK32_BK / 4);
    int innerColA = threadIdx.x % (SGB_SPLITK32_BK / 4);
    int innerRowB = threadIdx.x / (SGB_SPLITK32_BN / 4);
    int innerColB = threadIdx.x % (SGB_SPLITK32_BN / 4);

    // Register double-buffer: regM/regN[buf][i] — while FMA consumes buf=0,
    // the next iter's smem→reg load fills buf=1 in parallel, hiding smem
    // latency (~20-30 cyc vs ~8 cyc for 16 FMAs). Bit-exact: same FMA order,
    // same accumulator sequence. +8 regs total (was 8 → 16), fits in ~40 regs
    // per thread (ceiling 255).
    float regM[2][SGB_SPLITK32_TM] = {{0.0f}};
    float regN[2][SGB_SPLITK32_TN] = {{0.0f}};

    // persistent CTA loop for splitk32_partial.
    // Grid is K_CHUNKS × total_mn (joint blockIdx.x). Persistent loop iterates
    // tile_id over BOTH K-chunk and (pid_m, pid_n) — pid_k / pid_mn / pid_m / pid_n
    // all derive from tile_id per iteration. Each (pid_k, pid_m, pid_n) tile is
    // independent — partial slot is unique → no race.
    // Direct CTA mapping avoids the persistent-loop overhead (see sgemm_bi_nn). splitk32
    // case uses total_blocks = K_CHUNKS * total_mn since it iterates across
    // K-chunks too. Bit-exact: each (pid_k, pid_m, pid_n) tile is still
    // independent and gets its own CTA via gridDim = K_CHUNKS * total_mn.
    int tile_id = blockIdx.x;
    {
        float threadResults[SGB_SPLITK32_TM * SGB_SPLITK32_TN] = {0.0f};

        int pid_k = tile_id / total_mn;
        int pid_mn = tile_id % total_mn;
        int group_id = pid_mn / num_pid_in_group;
        int first_pid_m = group_id * SGB_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);
        int pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m);
        int pid_n = (pid_mn % num_pid_in_group) / group_size_m;

        int k_offset = pid_k * SGB_SPLITK32_BK;
        const float* A_block = A + pid_m * SGB_SPLITK32_BM * lda + k_offset;
        const float* B_block = B + k_offset * N + pid_n * SGB_SPLITK32_BN;

    // Phase B2: cp.async loads for Split-K thin-M partial kernel.
    // A: 4× cp.async.4B scattered transpose dest (same pattern as NN).
    // B: cp.async.16B contiguous (src+dst). Scalar fallback for non-%4 lda / OOB.
    unsigned As_base = __cvta_generic_to_shared(As);
    unsigned Bs_base = __cvta_generic_to_shared(Bs);

    for (int offset = 0; offset < SGB_SPLITK32_BM; offset += 16) {
        int g_row = pid_m * SGB_SPLITK32_BM + innerRowA + offset;
        bool pr = g_row < M;
        const float* src_base = A_block + (innerRowA + offset) * lda + innerColA * 4;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned dst = As_base + ((innerColA * 4 + i) * (SGB_SPLITK32_BM + SGB_SCALAR_SMEM_A_PAD) + innerRowA + offset) * (unsigned)sizeof(float);
            if (pr) {
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(dst), "l"(src_base + i));
            } else {
                As[(innerColA * 4 + i) * (SGB_SPLITK32_BM + SGB_SCALAR_SMEM_A_PAD) + innerRowA + offset] = 0.0f;
            }
        }
    }

    for (int offset = 0; offset < SGB_SPLITK32_BK; offset += 8) {
        int g_col = pid_n * SGB_SPLITK32_BN + innerColB * 4;
        unsigned dst = Bs_base + ((innerRowB + offset) * (SGB_SPLITK32_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4) * (unsigned)sizeof(float);
        if (g_col + 3 < N) {
            const float* src = B_block + (innerRowB + offset) * N + innerColB * 4;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(src));
        } else {
            Bs[(innerRowB + offset) * (SGB_SPLITK32_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 0] = (g_col + 0 < N) ? B_block[(innerRowB + offset) * N + innerColB * 4 + 0] : 0.0f;
            Bs[(innerRowB + offset) * (SGB_SPLITK32_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 1] = (g_col + 1 < N) ? B_block[(innerRowB + offset) * N + innerColB * 4 + 1] : 0.0f;
            Bs[(innerRowB + offset) * (SGB_SPLITK32_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 2] = (g_col + 2 < N) ? B_block[(innerRowB + offset) * N + innerColB * 4 + 2] : 0.0f;
            Bs[(innerRowB + offset) * (SGB_SPLITK32_BN + SGB_SCALAR_SMEM_B_PAD) + innerColB * 4 + 3] = (g_col + 3 < N) ? B_block[(innerRowB + offset) * N + innerColB * 4 + 3] : 0.0f;
        }
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    // Register double-buffer: prefetch dotIdx=0 into buf=0.
    #pragma unroll
    for (int i = 0; i < SGB_SPLITK32_TM; ++i)
        regM[0][i] = As[0 * (SGB_SPLITK32_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SPLITK32_WM + threadRowInWarp * SGB_SPLITK32_TM + i];
    #pragma unroll
    for (int i = 0; i < SGB_SPLITK32_TN; ++i)
        regN[0][i] = Bs[0 * (SGB_SPLITK32_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SPLITK32_WN + threadColInWarp * SGB_SPLITK32_TN + i];
    for (int dotIdx = 0; dotIdx < SGB_SPLITK32_BK; ++dotIdx) {
        int cur = dotIdx & 1;
        int nxt = cur ^ 1;
        // Prefetch next iteration's fragments (skip on last iter).
        if (dotIdx + 1 < SGB_SPLITK32_BK) {
            int next_k = dotIdx + 1;
            #pragma unroll
            for (int i = 0; i < SGB_SPLITK32_TM; ++i)
                regM[nxt][i] = As[next_k * (SGB_SPLITK32_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_SPLITK32_WM + threadRowInWarp * SGB_SPLITK32_TM + i];
            #pragma unroll
            for (int i = 0; i < SGB_SPLITK32_TN; ++i)
                regN[nxt][i] = Bs[next_k * (SGB_SPLITK32_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_SPLITK32_WN + threadColInWarp * SGB_SPLITK32_TN + i];
        }
        // FMA consumes current buffer — bit-exact: same rm-major, rn-minor order.
        // explicit __fmaf_rn for bit-exact match
        // with CPU `_mm256_fmadd_ps`. sgemm_bi_nn_splitk32_partial.
        #pragma unroll
        for (int rm = 0; rm < SGB_SPLITK32_TM; ++rm) {
            #pragma unroll
            for (int rn = 0; rn < SGB_SPLITK32_TN; ++rn) {
                int idx = rm * SGB_SPLITK32_TN + rn;
                threadResults[idx] = __fmaf_rn(
                    regM[cur][rm], regN[cur][rn], threadResults[idx]);
            }
        }
    }

    float* partial_base = partial + (long long)pid_k * M * N;
    for (int rm = 0; rm < SGB_SPLITK32_TM; ++rm) {
        int g_row = pid_m * SGB_SPLITK32_BM + warpRow * SGB_SPLITK32_WM + threadRowInWarp * SGB_SPLITK32_TM + rm;
        if (g_row >= M) continue;
        int g_col_base = pid_n * SGB_SPLITK32_BN + warpCol * SGB_SPLITK32_WN + threadColInWarp * SGB_SPLITK32_TN;
        if (g_col_base + 3 < N) {
            float4 out = {
                threadResults[rm * SGB_SPLITK32_TN + 0],
                threadResults[rm * SGB_SPLITK32_TN + 1],
                threadResults[rm * SGB_SPLITK32_TN + 2],
                threadResults[rm * SGB_SPLITK32_TN + 3]
            };
            reinterpret_cast<float4*>(&partial_base[g_row * N + g_col_base])[0] = out;
        } else {
            for (int j = 0; j < SGB_SPLITK32_TN && g_col_base + j < N; j++) {
                partial_base[g_row * N + g_col_base + j] = threadResults[rm * SGB_SPLITK32_TN + j];
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
    // explicit __fmaf_rn for bit-exact match with CPU FMA chain.
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
    // explicit __fmaf_rn for bit-exact match with CPU FMA.
    for (int n = 0; n < N; ++n) {
        sum = __fmaf_rn(dy_row[n], w_row[n], sum);
    }
    dX[(long long)m * out_col_stride + col_idx] = sum;
}

#undef SGB_SPLITK32_BM
#undef SGB_SPLITK32_BN
#undef SGB_SPLITK32_BK
#undef SGB_SPLITK32_WM
#undef SGB_SPLITK32_WN
#undef SGB_SPLITK32_TM
#undef SGB_SPLITK32_TN
#undef SGB_SPLITK32_NUM_THREADS

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

#define SGB_DEFINE_SGEMM_BI_NN_GEMV(SUFFIX, T_ACT, FROM_F)                        \
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

SGB_DEFINE_SGEMM_BI_NN_GEMV(bf16, __nv_bfloat16, from_f_bf16)
SGB_DEFINE_SGEMM_BI_NN_GEMV(f16,  __half,        from_f_f16)

// SGB_SCALAR_TN GEMV: dW[K] += alpha * X^T[K,M] @ dY[M]. dW stays f32 (master grad).
#define SGB_DEFINE_SGEMM_BI_TN_GEMV(SUFFIX, T_ACT, FROM_F)                        \
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

SGB_DEFINE_SGEMM_BI_TN_GEMV(bf16, __nv_bfloat16, from_f_bf16)
SGB_DEFINE_SGEMM_BI_TN_GEMV(f16,  __half,        from_f_f16)

// NT GEMV: dX[M,K] = alpha * dY[M] @ W^T[K]. Pure outer product.
#define SGB_DEFINE_SGEMM_BI_NT_GEMV(SUFFIX, T_ACT, FROM_F)                        \
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

SGB_DEFINE_SGEMM_BI_NT_GEMV(bf16, __nv_bfloat16, from_f_bf16)
SGB_DEFINE_SGEMM_BI_NT_GEMV(f16,  __half,        from_f_f16)

// Ultra-thin-M NN: M in [1, 32), smem-staged X row, 8-warp K-slab partials
// with the fixed 8-way tree reduce. smem_x stays f32 (upcast at stage-in) —
// the FMA chain is then bit-identical to the f32 kernel on upcast inputs.
#define SGB_DEFINE_SGEMM_BI_NN_ULTRA_THIN(SUFFIX, T_ACT, FROM_F)                  \
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

SGB_DEFINE_SGEMM_BI_NN_ULTRA_THIN(bf16, __nv_bfloat16, from_f_bf16)
SGB_DEFINE_SGEMM_BI_NN_ULTRA_THIN(f16,  __half,        from_f_f16)

// Typed narrow-N NN (generic over tile): A1 route — smem stays f32, typed
// inputs upcast at the SYNC stage-in with the exact zero-fill predication of
// the f32 kernels (the per-tile cp.async there is wait_all-fenced, i.e. not
// pipelined, so sync loads cost ~nothing). FMA mainloop, bias pre-seed at
// K=0 and the scalar-N epilogue are byte-identical to the f32 kernels —
// outputs are bit-identical to "upcast inputs, run f32 kernel".
#define SGB_DEFINE_SGEMM_BI_NN_NARROW_T(NAME, T_ACT, FROM_F, BM_, BN_, BK_, WM_, WN_, TM_, TN_, NTHR_, LB_) \
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
    __shared__ float As[BK_ * (BM_ + SGB_SCALAR_SMEM_A_PAD)];                            \
    __shared__ float Bs[BK_ * (BN_ + SGB_SCALAR_SMEM_B_PAD)];                            \
    int num_pid_m = (M + BM_ - 1) / BM_;                                      \
    int num_pid_n = (N + BN_ - 1) / BN_;                                      \
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;                           \
    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;                                     \
    int warpCol = warpIdx % (BN_ / WN_);                                      \
    int warpRow = warpIdx / (BN_ / WN_);                                      \
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;                                   \
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
                As[_k * (BM_ + SGB_SCALAR_SMEM_A_PAD) + _m] =                            \
                    (_g_row < M && _g_col < K)                                \
                        ? to_f(A_block[_m * lda + _k]) : 0.0f;                \
            }                                                                 \
            for (int idx = threadIdx.x; idx < BK_ * BN_; idx += NTHR_) {      \
                int _k = idx / BN_;                                           \
                int _n = idx % BN_;                                           \
                int g_row = bkIdx + _k;                                       \
                int g_col = pid_n * BN_ + _n;                                 \
                Bs[_k * (BN_ + SGB_SCALAR_SMEM_B_PAD) + _n] =                            \
                    (g_row < K && g_col < N)                                  \
                        ? to_f(B_block[_k * ldb + _n]) : 0.0f;                \
            }                                                                 \
            __syncthreads();                                                  \
            for (int dotIdx = 0; dotIdx < BK_; ++dotIdx) {                    \
                for (int i = 0; i < TM_; ++i) {                               \
                    regM[i] = As[dotIdx * (BM_ + SGB_SCALAR_SMEM_A_PAD)                  \
                                 + warpRow * WM_ + threadRowInWarp * TM_ + i]; \
                }                                                             \
                for (int i = 0; i < TN_; ++i) {                               \
                    regN[i] = Bs[dotIdx * (BN_ + SGB_SCALAR_SMEM_B_PAD)                  \
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

SGB_DEFINE_SGEMM_BI_NN_NARROW_T(sgemm_bi_nn_narrow_bf16, __nv_bfloat16, from_f_bf16, 64, 32, 16, 32, 16, 4, 4, 128, 4)
SGB_DEFINE_SGEMM_BI_NN_NARROW_T(sgemm_bi_nn_narrow_f16,  __half,        from_f_f16,  64, 32, 16, 32, 16, 4, 4, 128, 4)
SGB_DEFINE_SGEMM_BI_NN_NARROW_T(sgemm_bi_nn_narrow_small_bf16, __nv_bfloat16, from_f_bf16, 16, 16, 16, 8, 16, 2, 2, 64, 8)
SGB_DEFINE_SGEMM_BI_NN_NARROW_T(sgemm_bi_nn_narrow_small_f16,  __half,        from_f_f16,  16, 16, 16, 8, 16, 2, 2, 64, 8)

// Typed narrow SGB_SCALAR_TN (dW): C stays f32 (master grad, += epilogue); A=X and
// B=dY are typed. Same A1 route: f32 smem, sync typed stage-in with the
// f32 kernels' exact zero-fill predication; FMA chain unchanged.
#define SGB_DEFINE_SGEMM_BI_TN_NARROW_T(NAME, T_ACT)                              \
extern "C" __global__ __launch_bounds__(128, 4)                               \
void NAME(                                                                    \
    float* __restrict__ C,                                                    \
    const T_ACT* __restrict__ A,                                              \
    const T_ACT* __restrict__ B,                                              \
    float alpha,                                                              \
    int M_red, int K_out, int N                                               \
) {                                                                           \
    __shared__ float As[16 * (64 + SGB_SCALAR_SMEM_A_PAD)];                              \
    __shared__ float Bs[16 * (32 + SGB_SCALAR_SMEM_B_PAD)];                              \
    int num_pid_m = (K_out + 63) / 64;                                        \
    int num_pid_n = (N + 31) / 32;                                            \
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;                           \
    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;                                     \
    int warpCol = warpIdx % 2;                                                \
    int warpRow = warpIdx / 2;                                                \
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;                                   \
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
                As[_k * (64 + SGB_SCALAR_SMEM_A_PAD) + _m] =                             \
                    (_g_m < M_red && _g_k < K_out)                            \
                        ? to_f(A[(long long)_g_m * K_out + _g_k]) : 0.0f;     \
            }                                                                 \
            for (int idx = threadIdx.x; idx < 16 * 32; idx += 128) {          \
                int _k = idx / 32;                                            \
                int _n = idx % 32;                                            \
                int g_m = mIdx + _k;                                          \
                int g_n = pid_n * 32 + _n;                                    \
                Bs[_k * (32 + SGB_SCALAR_SMEM_B_PAD) + _n] =                             \
                    (g_m < M_red && g_n < N)                                  \
                        ? to_f(B[(long long)g_m * N + g_n]) : 0.0f;           \
            }                                                                 \
            __syncthreads();                                                  \
            for (int dotIdx = 0; dotIdx < 16; ++dotIdx) {                     \
                for (int i = 0; i < 4; ++i) {                                 \
                    regM[i] = As[dotIdx * (64 + SGB_SCALAR_SMEM_A_PAD)                   \
                                 + warpRow * 32 + threadRowInWarp * 4 + i];   \
                }                                                             \
                for (int i = 0; i < 4; ++i) {                                 \
                    regN[i] = Bs[dotIdx * (32 + SGB_SCALAR_SMEM_B_PAD)                   \
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

SGB_DEFINE_SGEMM_BI_TN_NARROW_T(sgemm_bi_tn_narrow_bf16, __nv_bfloat16)
SGB_DEFINE_SGEMM_BI_TN_NARROW_T(sgemm_bi_tn_narrow_f16,  __half)

// Typed narrow NT (dX): C=dX typed output (overwrite), A=dY and B=W typed.
// B tile staged TRANSPOSED (rows = reduction n, cols = k_out), exactly as
// the f32 kernel's float4 transposed stores.
#define SGB_DEFINE_SGEMM_BI_NT_NARROW_T(NAME, T_ACT, FROM_F)                      \
extern "C" __global__ __launch_bounds__(128, 4)                               \
void NAME(                                                                    \
    T_ACT* __restrict__ C,                                                    \
    const T_ACT* __restrict__ A,                                              \
    const T_ACT* __restrict__ B,                                              \
    float alpha,                                                              \
    int M, int N, int K_out                                                   \
) {                                                                           \
    __shared__ float As[16 * (64 + SGB_SCALAR_SMEM_A_PAD)];                              \
    __shared__ float Bs[16 * (32 + SGB_SCALAR_SMEM_B_PAD)];                              \
    int num_pid_m = (M + 63) / 64;                                            \
    int num_pid_n = (K_out + 31) / 32;                                        \
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;                           \
    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;                                     \
    int warpCol = warpIdx % 2;                                                \
    int warpRow = warpIdx / 2;                                                \
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;                                   \
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
                As[_n * (64 + SGB_SCALAR_SMEM_A_PAD) + _m] =                             \
                    (_g_m < M && _g_n < N)                                    \
                        ? to_f(A[(long long)_g_m * N + _g_n]) : 0.0f;         \
            }                                                                 \
            for (int idx = threadIdx.x; idx < 16 * 32; idx += 128) {          \
                int _n = idx / 32;                                            \
                int _kb = idx % 32;                                           \
                int g_k = pid_n * 32 + _kb;                                   \
                int g_n = nIdx + _n;                                          \
                Bs[_n * (32 + SGB_SCALAR_SMEM_B_PAD) + _kb] =                            \
                    (g_k < K_out && g_n < N)                                  \
                        ? to_f(B[(long long)g_k * N + g_n]) : 0.0f;           \
            }                                                                 \
            __syncthreads();                                                  \
            for (int dotIdx = 0; dotIdx < 16; ++dotIdx) {                     \
                for (int i = 0; i < 4; ++i) {                                 \
                    regM[i] = As[dotIdx * (64 + SGB_SCALAR_SMEM_A_PAD)                   \
                                 + warpRow * 32 + threadRowInWarp * 4 + i];   \
                }                                                             \
                for (int i = 0; i < 4; ++i) {                                 \
                    regN[i] = Bs[dotIdx * (32 + SGB_SCALAR_SMEM_B_PAD)                   \
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

SGB_DEFINE_SGEMM_BI_NT_NARROW_T(sgemm_bi_nt_narrow_bf16, __nv_bfloat16, from_f_bf16)
SGB_DEFINE_SGEMM_BI_NT_NARROW_T(sgemm_bi_nt_narrow_f16,  __half,        from_f_f16)

// ============================================================================
// The Slim/narrow sections above redefine the tile constants (SGB_SCALAR_BM/SGB_SCALAR_BN/SGB_SCALAR_BK,
// warp tiling, SGB_SCALAR_NUM_THREADS) and leave them in Slim state. The typed Big
// twins must compile with the BIG geometry regardless of preprocessor
// history, so this section uses its own SGB_T_* constants exclusively:
// 256 threads = 8 warps, 128x128x16 tiles, 64x32 warp tiles, 8x8 thread
// tiles — identical to the f32 Big kernels at the top of this file.
#define SGB_T_NTHREADS 256
#define SGB_T_BM 128
#define SGB_T_BN 128
#define SGB_T_BK 16
#define SGB_T_WM 64
#define SGB_T_WN 32
#define SGB_T_WNITER 1
#define SGB_T_TM 8
#define SGB_T_TN 8
#define SGB_T_WMITER \
    ((SGB_T_WM * SGB_T_WN) / (SGB_SCALAR_WARP_SIZE * SGB_T_TM * SGB_T_TN * SGB_T_WNITER))
#define SGB_T_WSUBM (SGB_T_WM / SGB_T_WMITER)
#define SGB_T_WSUBN (SGB_T_WN / SGB_T_WNITER)

// Typed Big NN/SGB_T_TN/NT (stage 3 v1): bf16/f16 twins of the Big warptiling
// kernels. Smem stays f32 — fragment loads and the __fmaf_rn chain are
// BYTE-IDENTICAL to the f32 kernels; only the staging instruction differs
// (synchronous ld.global -> to_f -> st.shared replaces cp.async, since
// cp.async cannot copy or convert 2-byte elements into the transposed f32
// As layout). Smem CONTENTS per (stage, cell) are bit-equal to the f32
// kernel's tiles on upcast inputs (incl. zero-fill OOB), so each typed
// kernel is bit-identical to "upcast inputs, run the f32 Big kernel,
// RNE-downcast the output" — the typed scalar contract.
//
// Pipeline: 2-stage rotation retained. One __syncthreads() per K-tile:
//   stage(0); loop { sync; if(next) stage(write); compute(read); rotate; }
// The sync at loop top (a) publishes the previous iteration's staging and
// (b) fences compute(read) of iter i-1 before iter i overwrites that stage
// (K_PIPE=2: write(i) == read(i-1)). Latency hiding falls to warp-level
// parallelism (8 warps/CTA, 2 CTA/SM) instead of async DMA.
// Dynamic smem = 33 KB -> host must set MAX_DYNAMIC_SHARED_SIZE_BYTES
// (34 KB) on these handles, same as the f32 Big kernels.

// Shared compute block: register-fragment double-buffered SGB_T_BK dot-product
// sweep, verbatim semantics of the f32 Big mainloop. Uses As_buf/Bs_buf/
// read_stage/regM/regN/threadResults and the warp placement values from
// the enclosing kernel scope.
#define SGB_T_COMPUTE_TILE()                                                   \
    do {                                                                       \
        float* As_rd = As_buf + read_stage * A_STAGE;                          \
        float* Bs_rd = Bs_buf + read_stage * B_STAGE;                          \
        float regM_next[SGB_T_WMITER * SGB_T_TM];                                          \
        float regN_next[SGB_T_WNITER * SGB_T_TN];                                          \
        _Pragma("unroll")                                                      \
        for (int wSubRowIdx = 0; wSubRowIdx < SGB_T_WMITER; ++wSubRowIdx)            \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < SGB_T_TM; ++i)                                       \
                regM[wSubRowIdx * SGB_T_TM + i] =                                    \
                    As_rd[0 * (SGB_T_BM + SGB_SCALAR_SMEM_A_PAD) + warpRow * SGB_T_WM +               \
                          wSubRowIdx * SGB_T_WSUBM + threadRowInWarp * SGB_T_TM + i];      \
        _Pragma("unroll")                                                      \
        for (int wSubColIdx = 0; wSubColIdx < SGB_T_WNITER; ++wSubColIdx)            \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < SGB_T_TN; ++i)                                       \
                regN[wSubColIdx * SGB_T_TN + i] =                                    \
                    Bs_rd[0 * (SGB_T_BN + SGB_SCALAR_SMEM_B_PAD) + warpCol * SGB_T_WN +               \
                          wSubColIdx * SGB_T_WSUBN + threadColInWarp * SGB_T_TN + i];      \
        for (int dotIdx = 0; dotIdx < SGB_T_BK; ++dotIdx) {                          \
            if (dotIdx + 1 < SGB_T_BK) {                                             \
                _Pragma("unroll")                                              \
                for (int wSubRowIdx = 0; wSubRowIdx < SGB_T_WMITER; ++wSubRowIdx)    \
                    _Pragma("unroll")                                          \
                    for (int i = 0; i < SGB_T_TM; ++i)                               \
                        regM_next[wSubRowIdx * SGB_T_TM + i] =                       \
                            As_rd[(dotIdx + 1) * (SGB_T_BM + SGB_SCALAR_SMEM_A_PAD) +           \
                                  warpRow * SGB_T_WM + wSubRowIdx * SGB_T_WSUBM +          \
                                  threadRowInWarp * SGB_T_TM + i];                   \
                _Pragma("unroll")                                              \
                for (int wSubColIdx = 0; wSubColIdx < SGB_T_WNITER; ++wSubColIdx)    \
                    _Pragma("unroll")                                          \
                    for (int i = 0; i < SGB_T_TN; ++i)                               \
                        regN_next[wSubColIdx * SGB_T_TN + i] =                       \
                            Bs_rd[(dotIdx + 1) * (SGB_T_BN + SGB_SCALAR_SMEM_B_PAD) +           \
                                  warpCol * SGB_T_WN + wSubColIdx * SGB_T_WSUBN +          \
                                  threadColInWarp * SGB_T_TN + i];                   \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int wSubRowIdx = 0; wSubRowIdx < SGB_T_WMITER; ++wSubRowIdx)        \
                _Pragma("unroll")                                              \
                for (int wSubColIdx = 0; wSubColIdx < SGB_T_WNITER; ++wSubColIdx)    \
                    _Pragma("unroll")                                          \
                    for (int resIdxM = 0; resIdxM < SGB_T_TM; ++resIdxM)             \
                        _Pragma("unroll")                                      \
                        for (int resIdxN = 0; resIdxN < SGB_T_TN; ++resIdxN) {       \
                            int idx = (wSubRowIdx * SGB_T_TM + resIdxM) *            \
                                          (SGB_T_WNITER * SGB_T_TN) +                      \
                                      wSubColIdx * SGB_T_TN + resIdxN;               \
                            threadResults[idx] = __fmaf_rn(                    \
                                regM[wSubRowIdx * SGB_T_TM + resIdxM],               \
                                regN[wSubColIdx * SGB_T_TN + resIdxN],               \
                                threadResults[idx]);                           \
                        }                                                      \
            if (dotIdx + 1 < SGB_T_BK) {                                             \
                _Pragma("unroll")                                              \
                for (int i = 0; i < SGB_T_WMITER * SGB_T_TM; ++i) regM[i] = regM_next[i];  \
                _Pragma("unroll")                                              \
                for (int i = 0; i < SGB_T_WNITER * SGB_T_TN; ++i) regN[i] = regN_next[i];  \
            }                                                                  \
        }                                                                      \
    } while (0)

// NN staging: As[k][m] = A[(pid_m*SGB_T_BM+m)*lda + bk+k], Bs[k][n] = B[(bk+k)*ldb
// + pid_n*SGB_T_BN+n], zero-fill OOB. Iteration order picked for contiguous global
// reads (A along k, B along n); placement equals the f32 cp.async tiles.
#define SGB_T_STAGE_NN(s, bkIdx)                                               \
    do {                                                                       \
        float* _As_w = As_buf + (s) * A_STAGE;                                 \
        float* _Bs_w = Bs_buf + (s) * B_STAGE;                                 \
        for (int _i = threadIdx.x; _i < SGB_T_BM * SGB_T_BK; _i += SGB_T_NTHREADS) {          \
            int _m = _i / SGB_T_BK;                                                  \
            int _k = _i % SGB_T_BK;                                                  \
            int _gr = pid_m * SGB_T_BM + _m;                                         \
            int _gc = (bkIdx) + _k;                                            \
            _As_w[_k * (SGB_T_BM + SGB_SCALAR_SMEM_A_PAD) + _m] =                               \
                (_gr < M && _gc < K)                                           \
                    ? to_f(A[(long long)_gr * lda + _gc])                      \
                    : 0.0f;                                                    \
        }                                                                      \
        for (int _i = threadIdx.x; _i < SGB_T_BK * SGB_T_BN; _i += SGB_T_NTHREADS) {          \
            int _k = _i / SGB_T_BN;                                                  \
            int _n = _i % SGB_T_BN;                                                  \
            int _gr = (bkIdx) + _k;                                            \
            int _gc = pid_n * SGB_T_BN + _n;                                         \
            _Bs_w[_k * (SGB_T_BN + SGB_SCALAR_SMEM_B_PAD) + _n] =                               \
                (_gr < K && _gc < N)                                           \
                    ? to_f(B[(long long)_gr * ldb + _gc])                      \
                    : 0.0f;                                                    \
        }                                                                      \
    } while (0)

// SGB_T_TN staging (A = X[M_red, K_out] read transposed, B = dY[M_red, N]):
// As[r][c] = A[(mIdx+r)*K_out + pid_m*SGB_T_BM+c], Bs[r][c] = B[(mIdx+r)*N +
// pid_n*SGB_T_BN+c]. Contiguous global reads along c.
#define SGB_T_STAGE_TN(s, mIdx)                                                \
    do {                                                                       \
        float* _As_w = As_buf + (s) * A_STAGE;                                 \
        float* _Bs_w = Bs_buf + (s) * B_STAGE;                                 \
        for (int _i = threadIdx.x; _i < SGB_T_BK * SGB_T_BM; _i += SGB_T_NTHREADS) {          \
            int _r = _i / SGB_T_BM;                                                  \
            int _c = _i % SGB_T_BM;                                                  \
            int _gm = (mIdx) + _r;                                             \
            int _gk = pid_m * SGB_T_BM + _c;                                         \
            _As_w[_r * (SGB_T_BM + SGB_SCALAR_SMEM_A_PAD) + _c] =                               \
                (_gm < M_red && _gk < K_out)                                   \
                    ? to_f(A[(long long)_gm * K_out + _gk])                    \
                    : 0.0f;                                                    \
        }                                                                      \
        for (int _i = threadIdx.x; _i < SGB_T_BK * SGB_T_BN; _i += SGB_T_NTHREADS) {          \
            int _r = _i / SGB_T_BN;                                                  \
            int _c = _i % SGB_T_BN;                                                  \
            int _gm = (mIdx) + _r;                                             \
            int _gn = pid_n * SGB_T_BN + _c;                                         \
            _Bs_w[_r * (SGB_T_BN + SGB_SCALAR_SMEM_B_PAD) + _c] =                               \
                (_gm < M_red && _gn < N)                                       \
                    ? to_f(B[(long long)_gm * N + _gn])                        \
                    : 0.0f;                                                    \
        }                                                                      \
    } while (0)

// NT staging (A = dY[M, N], B = W[K_out, N], both read along the N
// reduction): As[r][c] = A[(pid_m*SGB_T_BM+c)*N + nIdx+r], Bs[r][c] =
// B[(pid_n*SGB_T_BN+c)*N + nIdx+r]. Contiguous global reads along r.
#define SGB_T_STAGE_NT(s, nIdx)                                                \
    do {                                                                       \
        float* _As_w = As_buf + (s) * A_STAGE;                                 \
        float* _Bs_w = Bs_buf + (s) * B_STAGE;                                 \
        for (int _i = threadIdx.x; _i < SGB_T_BM * SGB_T_BK; _i += SGB_T_NTHREADS) {          \
            int _c = _i / SGB_T_BK;                                                  \
            int _r = _i % SGB_T_BK;                                                  \
            int _gm = pid_m * SGB_T_BM + _c;                                         \
            int _gn = (nIdx) + _r;                                             \
            _As_w[_r * (SGB_T_BM + SGB_SCALAR_SMEM_A_PAD) + _c] =                               \
                (_gm < M && _gn < N)                                           \
                    ? to_f(A[(long long)_gm * N + _gn])                        \
                    : 0.0f;                                                    \
        }                                                                      \
        for (int _i = threadIdx.x; _i < SGB_T_BN * SGB_T_BK; _i += SGB_T_NTHREADS) {          \
            int _c = _i / SGB_T_BK;                                                  \
            int _r = _i % SGB_T_BK;                                                  \
            int _gk = pid_n * SGB_T_BN + _c;                                         \
            int _gn = (nIdx) + _r;                                             \
            _Bs_w[_r * (SGB_T_BN + SGB_SCALAR_SMEM_B_PAD) + _c] =                               \
                (_gk < K_out && _gn < N)                                       \
                    ? to_f(B[(long long)_gk * N + _gn])                        \
                    : 0.0f;                                                    \
        }                                                                      \
    } while (0)

#define SGB_DEFINE_SGEMM_BI_NN_BIG_T(SUFFIX, T_ACT, FROM_F)                        \
extern "C" __global__ __launch_bounds__(SGB_T_NTHREADS, 2)                        \
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
    constexpr int A_STAGE = SGB_T_BK * (SGB_T_BM + SGB_SCALAR_SMEM_A_PAD);                            \
    constexpr int B_STAGE = SGB_T_BK * (SGB_T_BN + SGB_SCALAR_SMEM_B_PAD);                            \
    float* As_buf = smem;                                                      \
    float* Bs_buf = smem + K_PIPE * A_STAGE;                                   \
    int num_pid_m = (M + SGB_T_BM - 1) / SGB_T_BM;                                         \
    int num_pid_n = (N + SGB_T_BN - 1) / SGB_T_BN;                                         \
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;                            \
    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;                                      \
    int warpCol = warpIdx % (SGB_T_BN / SGB_T_WN);                                         \
    int warpRow = warpIdx / (SGB_T_BN / SGB_T_WN);                                         \
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;                                    \
    int threadColInWarp = tidInWarp % (SGB_T_WSUBN / SGB_T_TN);                            \
    int threadRowInWarp = tidInWarp / (SGB_T_WSUBN / SGB_T_TN);                            \
    float regM[SGB_T_WMITER * SGB_T_TM] = {0.0f};                                          \
    float regN[SGB_T_WNITER * SGB_T_TN] = {0.0f};                                          \
    int tile_id = blockIdx.x;                                                  \
    float threadResults[SGB_T_WMITER * SGB_T_TM * SGB_T_WNITER * SGB_T_TN];                            \
    int group_id = tile_id / num_pid_in_group;                                 \
    int first_pid_m = group_id * SGB_GROUP_M;                                  \
    int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);              \
    int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);   \
    int pid_n = (tile_id % num_pid_in_group) / group_size_m;                   \
    if (bias != nullptr) {                                                     \
        _Pragma("unroll")                                                      \
        for (int wSubColIdx = 0; wSubColIdx < SGB_T_WNITER; ++wSubColIdx) {          \
            _Pragma("unroll")                                                  \
            for (int resIdxN = 0; resIdxN < SGB_T_TN; ++resIdxN) {                   \
                int g_col = pid_n * SGB_T_BN + warpCol * SGB_T_WN + wSubColIdx * SGB_T_WSUBN +   \
                            threadColInWarp * SGB_T_TN + resIdxN;                    \
                float b_val = (g_col < N) ? bias[g_col] : 0.0f;                \
                _Pragma("unroll")                                              \
                for (int wSubRowIdx = 0; wSubRowIdx < SGB_T_WMITER; ++wSubRowIdx) {  \
                    _Pragma("unroll")                                          \
                    for (int resIdxM = 0; resIdxM < SGB_T_TM; ++resIdxM) {           \
                        int idx = (wSubRowIdx * SGB_T_TM + resIdxM) * (SGB_T_WNITER * SGB_T_TN)  \
                                  + wSubColIdx * SGB_T_TN + resIdxN;                 \
                        threadResults[idx] = b_val;                            \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } else {                                                                   \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < SGB_T_WMITER * SGB_T_TM * SGB_T_WNITER * SGB_T_TN; ++i) {                  \
            threadResults[i] = 0.0f;                                           \
        }                                                                      \
    }                                                                          \
    T_ACT* C_warp =                                                            \
        C + (pid_m * SGB_T_BM + warpRow * SGB_T_WM) * ldc + pid_n * SGB_T_BN + warpCol * SGB_T_WN;     \
    int num_k_tiles = (K + SGB_T_BK - 1) / SGB_T_BK;                                       \
    SGB_T_STAGE_NN(0, 0);                                                      \
    int read_stage = 0;                                                        \
    int write_stage = 1;                                                       \
    for (int tile = 0; tile < num_k_tiles; ++tile) {                           \
        __syncthreads();                                                       \
        if (tile + 1 < num_k_tiles) {                                          \
            SGB_T_STAGE_NN(write_stage, (tile + 1) * SGB_T_BK);                      \
        }                                                                      \
        SGB_T_COMPUTE_TILE();                                                  \
        read_stage = (read_stage + 1) % K_PIPE;                                \
        write_stage = (write_stage + 1) % K_PIPE;                              \
    }                                                                          \
    for (int wSubRowIdx = 0; wSubRowIdx < SGB_T_WMITER; ++wSubRowIdx) {              \
        for (int wSubColIdx = 0; wSubColIdx < SGB_T_WNITER; ++wSubColIdx) {          \
            T_ACT* C_sub = C_warp + wSubRowIdx * SGB_T_WSUBM * ldc +                 \
                           wSubColIdx * SGB_T_WSUBN;                                 \
            for (int resIdxM = 0; resIdxM < SGB_T_TM; ++resIdxM) {                   \
                int g_row = pid_m * SGB_T_BM + warpRow * SGB_T_WM + wSubRowIdx * SGB_T_WSUBM +   \
                            threadRowInWarp * SGB_T_TM + resIdxM;                    \
                if (g_row >= M) continue;                                      \
                for (int resIdxN = 0; resIdxN < SGB_T_TN; ++resIdxN) {               \
                    int g_col = pid_n * SGB_T_BN + warpCol * SGB_T_WN +                    \
                                wSubColIdx * SGB_T_WSUBN + threadColInWarp * SGB_T_TN +    \
                                resIdxN;                                       \
                    if (g_col >= N) continue;                                  \
                    int idx = (wSubRowIdx * SGB_T_TM + resIdxM) * (SGB_T_WNITER * SGB_T_TN) +    \
                              wSubColIdx * SGB_T_TN + resIdxN;                       \
                    int c_off = (threadRowInWarp * SGB_T_TM + resIdxM) * ldc +       \
                                threadColInWarp * SGB_T_TN + resIdxN;                \
                    float val = alpha * threadResults[idx];                    \
                    if (beta != 0.0f) val += beta * to_f(C_sub[c_off]);        \
                    C_sub[c_off] = FROM_F(val);                                \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

SGB_DEFINE_SGEMM_BI_NN_BIG_T(bf16, __nv_bfloat16, from_f_bf16)
SGB_DEFINE_SGEMM_BI_NN_BIG_T(f16,  __half,        from_f_f16)

#define SGB_DEFINE_SGEMM_BI_TN_BIG_T(SUFFIX, T_ACT)                                \
extern "C" __global__ __launch_bounds__(SGB_T_NTHREADS, 2)                        \
void sgemm_bi_tn_big_##SUFFIX(                                                 \
    float* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M_red, int K_out, int N                                                \
) {                                                                            \
    constexpr int K_PIPE = 2;                                                  \
    extern __shared__ __align__(16) float smem[];                              \
    constexpr int A_STAGE = SGB_T_BK * (SGB_T_BM + SGB_SCALAR_SMEM_A_PAD);                            \
    constexpr int B_STAGE = SGB_T_BK * (SGB_T_BN + SGB_SCALAR_SMEM_B_PAD);                            \
    float* As_buf = smem;                                                      \
    float* Bs_buf = smem + K_PIPE * A_STAGE;                                   \
    int num_pid_m = (K_out + SGB_T_BM - 1) / SGB_T_BM;                                     \
    int num_pid_n = (N + SGB_T_BN - 1) / SGB_T_BN;                                         \
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;                            \
    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;                                      \
    int warpCol = warpIdx % (SGB_T_BN / SGB_T_WN);                                         \
    int warpRow = warpIdx / (SGB_T_BN / SGB_T_WN);                                         \
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;                                    \
    int threadColInWarp = tidInWarp % (SGB_T_WSUBN / SGB_T_TN);                            \
    int threadRowInWarp = tidInWarp / (SGB_T_WSUBN / SGB_T_TN);                            \
    float regM[SGB_T_WMITER * SGB_T_TM] = {0.0f};                                          \
    float regN[SGB_T_WNITER * SGB_T_TN] = {0.0f};                                          \
    int tile_id = blockIdx.x;                                                  \
    float threadResults[SGB_T_WMITER * SGB_T_TM * SGB_T_WNITER * SGB_T_TN] = {0.0f};                   \
    int group_id = tile_id / num_pid_in_group;                                 \
    int first_pid_m = group_id * SGB_GROUP_M;                                  \
    int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);              \
    int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);   \
    int pid_n = (tile_id % num_pid_in_group) / group_size_m;                   \
    float* C_warp =                                                            \
        C + (pid_m * SGB_T_BM + warpRow * SGB_T_WM) * N + pid_n * SGB_T_BN + warpCol * SGB_T_WN;       \
    int num_m_tiles = (M_red + SGB_T_BK - 1) / SGB_T_BK;                                   \
    SGB_T_STAGE_TN(0, 0);                                                      \
    int read_stage = 0;                                                        \
    int write_stage = 1;                                                       \
    for (int tile = 0; tile < num_m_tiles; ++tile) {                           \
        __syncthreads();                                                       \
        if (tile + 1 < num_m_tiles) {                                          \
            SGB_T_STAGE_TN(write_stage, (tile + 1) * SGB_T_BK);                      \
        }                                                                      \
        SGB_T_COMPUTE_TILE();                                                  \
        read_stage = (read_stage + 1) % K_PIPE;                                \
        write_stage = (write_stage + 1) % K_PIPE;                              \
    }                                                                          \
    for (int wSubRowIdx = 0; wSubRowIdx < SGB_T_WMITER; ++wSubRowIdx) {              \
        for (int wSubColIdx = 0; wSubColIdx < SGB_T_WNITER; ++wSubColIdx) {          \
            float* C_sub =                                                     \
                C_warp + wSubRowIdx * SGB_T_WSUBM * N + wSubColIdx * SGB_T_WSUBN;          \
            for (int resIdxM = 0; resIdxM < SGB_T_TM; ++resIdxM) {                   \
                int g_row = pid_m * SGB_T_BM + warpRow * SGB_T_WM + wSubRowIdx * SGB_T_WSUBM +   \
                            threadRowInWarp * SGB_T_TM + resIdxM;                    \
                if (g_row >= K_out) continue;                                  \
                for (int resIdxN = 0; resIdxN < SGB_T_TN; ++resIdxN) {               \
                    int g_col = pid_n * SGB_T_BN + warpCol * SGB_T_WN +                    \
                                wSubColIdx * SGB_T_WSUBN + threadColInWarp * SGB_T_TN +    \
                                resIdxN;                                       \
                    if (g_col >= N) continue;                                  \
                    int idx = (wSubRowIdx * SGB_T_TM + resIdxM) * (SGB_T_WNITER * SGB_T_TN) +    \
                              wSubColIdx * SGB_T_TN + resIdxN;                       \
                    C_sub[(threadRowInWarp * SGB_T_TM + resIdxM) * N +               \
                          threadColInWarp * SGB_T_TN + resIdxN] +=                   \
                        alpha * threadResults[idx];                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

SGB_DEFINE_SGEMM_BI_TN_BIG_T(bf16, __nv_bfloat16)
SGB_DEFINE_SGEMM_BI_TN_BIG_T(f16,  __half)

#define SGB_DEFINE_SGEMM_BI_NT_BIG_T(SUFFIX, T_ACT, FROM_F)                        \
extern "C" __global__ __launch_bounds__(SGB_T_NTHREADS, 2)                        \
void sgemm_bi_nt_big_##SUFFIX(                                                 \
    T_ACT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    float alpha,                                                               \
    int M, int N, int K_out                                                    \
) {                                                                            \
    constexpr int K_PIPE = 2;                                                  \
    extern __shared__ __align__(16) float smem[];                              \
    constexpr int A_STAGE = SGB_T_BK * (SGB_T_BM + SGB_SCALAR_SMEM_A_PAD);                            \
    constexpr int B_STAGE = SGB_T_BK * (SGB_T_BN + SGB_SCALAR_SMEM_B_PAD);                            \
    float* As_buf = smem;                                                      \
    float* Bs_buf = smem + K_PIPE * A_STAGE;                                   \
    int num_pid_m = (M + SGB_T_BM - 1) / SGB_T_BM;                                         \
    int num_pid_n = (K_out + SGB_T_BN - 1) / SGB_T_BN;                                     \
    int num_pid_in_group = SGB_GROUP_M * num_pid_n;                            \
    int warpIdx = threadIdx.x / SGB_SCALAR_WARP_SIZE;                                      \
    int warpCol = warpIdx % (SGB_T_BN / SGB_T_WN);                                         \
    int warpRow = warpIdx / (SGB_T_BN / SGB_T_WN);                                         \
    int tidInWarp = threadIdx.x % SGB_SCALAR_WARP_SIZE;                                    \
    int threadColInWarp = tidInWarp % (SGB_T_WSUBN / SGB_T_TN);                            \
    int threadRowInWarp = tidInWarp / (SGB_T_WSUBN / SGB_T_TN);                            \
    float regM[SGB_T_WMITER * SGB_T_TM] = {0.0f};                                          \
    float regN[SGB_T_WNITER * SGB_T_TN] = {0.0f};                                          \
    int tile_id = blockIdx.x;                                                  \
    float threadResults[SGB_T_WMITER * SGB_T_TM * SGB_T_WNITER * SGB_T_TN] = {0.0f};                   \
    int group_id = tile_id / num_pid_in_group;                                 \
    int first_pid_m = group_id * SGB_GROUP_M;                                  \
    int group_size_m = min(num_pid_m - first_pid_m, SGB_GROUP_M);              \
    int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);   \
    int pid_n = (tile_id % num_pid_in_group) / group_size_m;                   \
    T_ACT* C_warp =                                                            \
        C + (pid_m * SGB_T_BM + warpRow * SGB_T_WM) * K_out + pid_n * SGB_T_BN + warpCol * SGB_T_WN;   \
    int num_n_tiles = (N + SGB_T_BK - 1) / SGB_T_BK;                                       \
    SGB_T_STAGE_NT(0, 0);                                                      \
    int read_stage = 0;                                                        \
    int write_stage = 1;                                                       \
    for (int tile = 0; tile < num_n_tiles; ++tile) {                           \
        __syncthreads();                                                       \
        if (tile + 1 < num_n_tiles) {                                          \
            SGB_T_STAGE_NT(write_stage, (tile + 1) * SGB_T_BK);                      \
        }                                                                      \
        SGB_T_COMPUTE_TILE();                                                  \
        read_stage = (read_stage + 1) % K_PIPE;                                \
        write_stage = (write_stage + 1) % K_PIPE;                              \
    }                                                                          \
    for (int wSubRowIdx = 0; wSubRowIdx < SGB_T_WMITER; ++wSubRowIdx) {              \
        for (int wSubColIdx = 0; wSubColIdx < SGB_T_WNITER; ++wSubColIdx) {          \
            T_ACT* C_sub =                                                     \
                C_warp + wSubRowIdx * SGB_T_WSUBM * K_out + wSubColIdx * SGB_T_WSUBN;      \
            for (int resIdxM = 0; resIdxM < SGB_T_TM; ++resIdxM) {                   \
                int g_row = pid_m * SGB_T_BM + warpRow * SGB_T_WM + wSubRowIdx * SGB_T_WSUBM +   \
                            threadRowInWarp * SGB_T_TM + resIdxM;                    \
                if (g_row >= M) continue;                                      \
                for (int resIdxN = 0; resIdxN < SGB_T_TN; ++resIdxN) {               \
                    int g_col = pid_n * SGB_T_BN + warpCol * SGB_T_WN +                    \
                                wSubColIdx * SGB_T_WSUBN + threadColInWarp * SGB_T_TN +    \
                                resIdxN;                                       \
                    if (g_col >= K_out) continue;                              \
                    int idx = (wSubRowIdx * SGB_T_TM + resIdxM) * (SGB_T_WNITER * SGB_T_TN) +    \
                              wSubColIdx * SGB_T_TN + resIdxN;                       \
                    C_sub[(threadRowInWarp * SGB_T_TM + resIdxM) * K_out +           \
                          threadColInWarp * SGB_T_TN + resIdxN] =                    \
                        FROM_F(alpha * threadResults[idx]);                    \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

SGB_DEFINE_SGEMM_BI_NT_BIG_T(bf16, __nv_bfloat16, from_f_bf16)
SGB_DEFINE_SGEMM_BI_NT_BIG_T(f16,  __half,        from_f_f16)


#undef SGB_DEFINE_SGEMM_BI_NN_BIG_T
#undef SGB_DEFINE_SGEMM_BI_NN_GEMV
#undef SGB_DEFINE_SGEMM_BI_NN_NARROW_T
#undef SGB_DEFINE_SGEMM_BI_NN_ULTRA_THIN
#undef SGB_DEFINE_SGEMM_BI_NT_BIG_T
#undef SGB_DEFINE_SGEMM_BI_NT_GEMV
#undef SGB_DEFINE_SGEMM_BI_NT_NARROW_T
#undef SGB_DEFINE_SGEMM_BI_TN_BIG_T
#undef SGB_DEFINE_SGEMM_BI_TN_GEMV
#undef SGB_DEFINE_SGEMM_BI_TN_NARROW_T
#undef SGB_GROUP_M
#undef SGB_NARROW_BK
#undef SGB_NARROW_BM
#undef SGB_NARROW_BN
#undef SGB_NARROW_NUM_THREADS
#undef SGB_NARROW_ROW_STRIDE_A
#undef SGB_NARROW_ROW_STRIDE_B
#undef SGB_NARROW_SMALL_BK
#undef SGB_NARROW_SMALL_BM
#undef SGB_NARROW_SMALL_BN
#undef SGB_NARROW_SMALL_NUM_THREADS
#undef SGB_NARROW_SMALL_ROW_STRIDE_A
#undef SGB_NARROW_SMALL_ROW_STRIDE_B
#undef SGB_NARROW_SMALL_TM
#undef SGB_NARROW_SMALL_TN
#undef SGB_NARROW_SMALL_WM
#undef SGB_NARROW_SMALL_WMITER
#undef SGB_NARROW_SMALL_WN
#undef SGB_NARROW_SMALL_WNITER
#undef SGB_NARROW_SMALL_WSUBM
#undef SGB_NARROW_SMALL_WSUBN
#undef SGB_NARROW_TM
#undef SGB_NARROW_TN
#undef SGB_NARROW_WM
#undef SGB_NARROW_WMITER
#undef SGB_NARROW_WN
#undef SGB_NARROW_WNITER
#undef SGB_NARROW_WSUBM
#undef SGB_NARROW_WSUBN
#undef SGB_SCALAR_BK
#undef SGB_SCALAR_BM
#undef SGB_SCALAR_BN
#undef SGB_SCALAR_NUM_THREADS
#undef SGB_SCALAR_NUM_WARPS
#undef SGB_SCALAR_ROW_STRIDE_A
#undef SGB_SCALAR_ROW_STRIDE_B
#undef SGB_SCALAR_SMEM_A_PAD
#undef SGB_SCALAR_SMEM_B_PAD
#undef SGB_SCALAR_TM
#undef SGB_SCALAR_TN
#undef SGB_SCALAR_WARP_SIZE
#undef SGB_SCALAR_WM
#undef SGB_SCALAR_WMITER
#undef SGB_SCALAR_WN
#undef SGB_SCALAR_WNITER
#undef SGB_SCALAR_WSUBM
#undef SGB_SCALAR_WSUBN
#undef SGB_SPLITK32_BK
#undef SGB_SPLITK32_BM
#undef SGB_SPLITK32_BN
#undef SGB_SPLITK32_NUM_THREADS
#undef SGB_SPLITK32_TM
#undef SGB_SPLITK32_TN
#undef SGB_SPLITK32_WM
#undef SGB_SPLITK32_WN
#undef SGB_T_BK
#undef SGB_T_BM
#undef SGB_T_BN
#undef SGB_T_COMPUTE_TILE
#undef SGB_T_NTHREADS
#undef SGB_T_STAGE_NN
#undef SGB_T_STAGE_NT
#undef SGB_T_STAGE_TN
#undef SGB_T_TM
#undef SGB_T_TN
#undef SGB_T_WM
#undef SGB_T_WMITER
#undef SGB_T_WN
#undef SGB_T_WNITER
#undef SGB_T_WSUBM
#undef SGB_T_WSUBN
