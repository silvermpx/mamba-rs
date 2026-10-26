// The 128 x 64 tile over a cut layer's forward: the 128 x 128 tile's
// mainloop and fold on half its columns and half its threads, each thread
// still eight rows by eight columns. Each piece of `piece_tiles` 16-deep
// tiles is a chain from +0.0 over whole 32-deep groups, added in order to the
// sum a thread keeps in registers, the bias after the pieces - the bits of
// the 128 x 128 and 64 x 64 pieces tiles and of the chunk partials. It puts
// twice the CTAs of the 128 x 128 tile on a layer whose large-tile grid
// leaves part of the card idle.
#define EXACT_NN_M128N64_BM 128
#define EXACT_NN_M128N64_BN 64
#define EXACT_NN_M128N64_BK 16
#define EXACT_NN_M128N64_WM 64
#define EXACT_NN_M128N64_WN 32
#define EXACT_NN_M128N64_WNITER 2
#define EXACT_NN_M128N64_TM 8
#define EXACT_NN_M128N64_TN 4
#define EXACT_NN_M128N64_NUM_THREADS 128
#define EXACT_NN_M128N64_WARP_SIZE 32
#define EXACT_NN_M128N64_SMEM_A_PAD 4
#define EXACT_NN_M128N64_SMEM_B_PAD 4
// The row group of the tile walk, the 128 x 128 tile's on an Ada.
#define EXACT_NN_M128N64_GROUP_M 16
#define EXACT_NN_M128N64_WMITER \
    ((EXACT_NN_M128N64_WM * EXACT_NN_M128N64_WN) \
     / (EXACT_NN_M128N64_WARP_SIZE * EXACT_NN_M128N64_TM * EXACT_NN_M128N64_TN \
        * EXACT_NN_M128N64_WNITER))
#define EXACT_NN_M128N64_WSUBM (EXACT_NN_M128N64_WM / EXACT_NN_M128N64_WMITER)
#define EXACT_NN_M128N64_WSUBN (EXACT_NN_M128N64_WN / EXACT_NN_M128N64_WNITER)
#define EXACT_NN_M128N64_ROW_STRIDE_B \
    (EXACT_NN_M128N64_NUM_THREADS / (EXACT_NN_M128N64_BN / 4))

__device__ __forceinline__ void nn_m128n64_pieces_impl(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float alpha, float beta,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    int piece_tiles
) {
    // The bias follows the pieces unscaled, so a bias comes with alpha 1.
    assert(alpha == 1.0f || bias == nullptr);
    // 2-stage cp.async pipeline (CUTLASS multistage SM80 pattern).
    // Dynamic smem layout: As[K_PIPE][EXACT_NN_M128N64_BK*(EXACT_NN_M128N64_BM+EXACT_NN_M128N64_SMEM_A_PAD)] || Bs[K_PIPE][EXACT_NN_M128N64_BK*(EXACT_NN_M128N64_BN+EXACT_NN_M128N64_SMEM_B_PAD)].
    // Per-block smem = K_PIPE * (A_STAGE + B_STAGE) * 4B = 2 * (2112 + 1088) * 4 = 25 KB.
    //
    // OOB handling via 4-operand cp.async (PTX ISA 9.7.8.22): src_bytes=0 → hardware
    // zero-fills cp_size bytes in dst. Bit-exact identical to scalar `=0.0f`.
    // No scalar-store path → no ordering hole vs cp.async groups.
    //
    // Determinism preserved: same FMA order per tile, same tile order, same block
    // mapping. cp.async only changes WHEN a load lands; wait_group + sync ensures
    // visibility before any FMA reads from that stage.
    constexpr int K_PIPE = 2;
    extern __shared__ __align__(16) float exact_m128n64_smem[];
    constexpr int A_STAGE = EXACT_NN_M128N64_BK * (EXACT_NN_M128N64_BM + EXACT_NN_M128N64_SMEM_A_PAD);  // 16 * 132 = 2112 floats
    constexpr int B_STAGE = EXACT_NN_M128N64_BK * (EXACT_NN_M128N64_BN + EXACT_NN_M128N64_SMEM_B_PAD);  // 16 * 68 = 1088 floats
    float* As_buf = exact_m128n64_smem;                       // [K_PIPE * A_STAGE]
    float* Bs_buf = exact_m128n64_smem + K_PIPE * A_STAGE;    // [K_PIPE * B_STAGE]

    // L2 row-group swizzle.
    int num_pid_m = (M + EXACT_NN_M128N64_BM - 1) / EXACT_NN_M128N64_BM;
    int num_pid_n = (N + EXACT_NN_M128N64_BN - 1) / EXACT_NN_M128N64_BN;
    int num_pid_in_group = EXACT_NN_M128N64_GROUP_M * num_pid_n;

    // Warp and thread placement (constant across all tiles a CTA processes).
    int warpIdx = threadIdx.x / EXACT_NN_M128N64_WARP_SIZE;
    int warpCol = warpIdx % (EXACT_NN_M128N64_BN / EXACT_NN_M128N64_WN);
    int warpRow = warpIdx / (EXACT_NN_M128N64_BN / EXACT_NN_M128N64_WN);
    int tidInWarp = threadIdx.x % EXACT_NN_M128N64_WARP_SIZE;
    int threadColInWarp = tidInWarp % (EXACT_NN_M128N64_WSUBN / EXACT_NN_M128N64_TN);
    int threadRowInWarp = tidInWarp / (EXACT_NN_M128N64_WSUBN / EXACT_NN_M128N64_TN);

    int innerRowB = threadIdx.x / (EXACT_NN_M128N64_BN / 4);
    int innerColB = threadIdx.x % (EXACT_NN_M128N64_BN / 4);

    // Working-set registers (reset by mainloop per tile).
    float regM[EXACT_NN_M128N64_WMITER * EXACT_NN_M128N64_TM] = {0.0f};
    float regN[EXACT_NN_M128N64_WNITER * EXACT_NN_M128N64_TN] = {0.0f};

    unsigned As_base = __cvta_generic_to_shared(As_buf);
    unsigned Bs_base = __cvta_generic_to_shared(Bs_buf);

    // One output tile per CTA.
    int tile_id = blockIdx.x;
    {
        float threadResults[EXACT_NN_M128N64_WMITER * EXACT_NN_M128N64_TM * EXACT_NN_M128N64_WNITER * EXACT_NN_M128N64_TN];
        constexpr int RESULTS = EXACT_NN_M128N64_WMITER * EXACT_NN_M128N64_TM * EXACT_NN_M128N64_WNITER * EXACT_NN_M128N64_TN;
        float kept[RESULTS];
        // The finished pieces' sum starts at -0.0, the one value whose sum
        // with any chain is that chain, so a piece end adds without asking
        // whether it closes the first piece.
        #pragma unroll
        for (int result = 0; result < RESULTS; ++result) {
            kept[result] = -0.0f;
        }

        // Resolve (pid_m, pid_n) from tile_id via the row-group swizzle.
        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * EXACT_NN_M128N64_GROUP_M;
        int group_size_m = min(num_pid_m - first_pid_m, EXACT_NN_M128N64_GROUP_M);
        int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        // A cut layer starts every piece from +0.0 and adds the bias after
        // the pieces.
        #pragma unroll
        for (int i = 0; i < EXACT_NN_M128N64_WMITER * EXACT_NN_M128N64_TM * EXACT_NN_M128N64_WNITER * EXACT_NN_M128N64_TN; ++i) {
            threadResults[i] = 0.0f;
        }

        // C output pointer for this tile.
        float* C_warp = C + (pid_m * EXACT_NN_M128N64_BM + warpRow * EXACT_NN_M128N64_WM) * ldc + pid_n * EXACT_NN_M128N64_BN + warpCol * EXACT_NN_M128N64_WN;

    // Macro-style helper: issue one full A+B tile into the given stage.
    // bkIdx is the K-offset of the tile being loaded. Uses 4-operand cp.async
    // for OOB zero-fill — branch-free, bit-exact, no scalar-store hole.
    #define ISSUE_EXACT_NN_M128N64_TILE(stage, bkIdx) do {                                                 \
        /* 32 lanes split as 2 M-rows ×       \
         * EXACT_NN_M128N64_BK=16 K-cols, 1 lane = 1 float (4B cp.async). 2 cache lines/warp instr,        \
         * 50% util vs prior 12.5% (8 rows × 16B/row scatter). Same cp.async count,       \
         * same dest As[K][M] layout → bit-exact preserved. 16B cp.async blocked by       \
         * dest stride; full coalesce requires layout swap (deferred).                    \
         */                                                                               \
        {                                                                                 \
            constexpr int WARPS_NN = EXACT_NN_M128N64_NUM_THREADS / EXACT_NN_M128N64_WARP_SIZE;                              \
            constexpr int M_ROWS_PER_WARP_INST_NN = EXACT_NN_M128N64_WARP_SIZE / EXACT_NN_M128N64_BK;                        \
            constexpr int M_ROWS_PER_WARP_NN = EXACT_NN_M128N64_BM / WARPS_NN;                             \
            constexpr int INSTR_PER_WARP_NN =                                             \
                M_ROWS_PER_WARP_NN / M_ROWS_PER_WARP_INST_NN;                             \
            static_assert(EXACT_NN_M128N64_WARP_SIZE % EXACT_NN_M128N64_BK == 0,                                             \
                "EXACT_NN_M128N64_WARP_SIZE must be divisible by EXACT_NN_M128N64_BK for NN A coalesce");                    \
            static_assert(EXACT_NN_M128N64_BM % WARPS_NN == 0,                                             \
                "EXACT_NN_M128N64_BM must be divisible by warp count for NN A coalesce");                  \
            int _warp = threadIdx.x / EXACT_NN_M128N64_WARP_SIZE;                                           \
            int _lane = threadIdx.x % EXACT_NN_M128N64_WARP_SIZE;                                           \
            int _m_in_warp = _lane / EXACT_NN_M128N64_BK;                                                  \
            int _k_local = _lane % EXACT_NN_M128N64_BK;                                                    \
            _Pragma("unroll")                                                             \
            for (int _it = 0; _it < INSTR_PER_WARP_NN; _it++) {                           \
                int _m_local = _warp * M_ROWS_PER_WARP_NN                                 \
                               + _it * M_ROWS_PER_WARP_INST_NN + _m_in_warp;              \
                int _g_row = pid_m * EXACT_NN_M128N64_BM + _m_local;                                       \
                int _g_col = (bkIdx) + _k_local;                                          \
                unsigned _dst = As_base + ((stage) * A_STAGE                              \
                    + _k_local * (EXACT_NN_M128N64_BM + EXACT_NN_M128N64_SMEM_A_PAD) + _m_local)                            \
                    * (unsigned)sizeof(float);                                            \
                int _bytes = (_g_row < M && _g_col < K) ? 4 : 0;                          \
                const float* _src = A + (long long)_g_row * lda + _g_col;                 \
                asm volatile(                                                             \
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"                      \
                    :: "r"(_dst), "l"(_src), "r"(_bytes));                                \
            }                                                                             \
        }                                                                                 \
        for (int _off = 0; _off + EXACT_NN_M128N64_ROW_STRIDE_B <= EXACT_NN_M128N64_BK; _off += EXACT_NN_M128N64_ROW_STRIDE_B) {             \
            int _g_row = (bkIdx) + innerRowB + _off;                                      \
            int _g_col = pid_n * EXACT_NN_M128N64_BN + innerColB * 4;                                      \
            unsigned _dst = Bs_base + ((stage) * B_STAGE                                  \
                + (innerRowB + _off) * (EXACT_NN_M128N64_BN + EXACT_NN_M128N64_SMEM_B_PAD)                                  \
                + innerColB * 4) * (unsigned)sizeof(float);                               \
            const float* _src = B + (long long)_g_row * ldb + _g_col;                     \
            bool _full16 = (_g_row < K) && (_g_col + 3 < N) && ((ldb % 4) == 0)          \
                           && is_aligned_16(B);                                       \
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
    // A cut layer walks whole 32-deep groups, as the chunk partials do, so
    // its last piece ends with the same zero products; the plan gives the
    // piece length in tiles at launch.
    int num_k_tiles = (K + 31) / 32 * (32 / EXACT_NN_M128N64_BK);
    // The tile that closes the current piece; one counter keeps the loop free
    // of a division per tile.
    int fold_tile = piece_tiles;
    ISSUE_EXACT_NN_M128N64_TILE(0, 0);

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
            int next_bkIdx = next_tile * EXACT_NN_M128N64_BK;
            ISSUE_EXACT_NN_M128N64_TILE(write_stage, next_bkIdx);
        }

        // Compute on read_stage.
        float* As_rd = As_buf + read_stage * A_STAGE;
        float* Bs_rd = Bs_buf + read_stage * B_STAGE;
        #pragma unroll
        for (int dotIdx = 0; dotIdx < EXACT_NN_M128N64_BK; ++dotIdx) {
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < EXACT_NN_M128N64_WMITER; ++wSubRowIdx)
                scalar_load_fragment<EXACT_NN_M128N64_TM>(&regM[wSubRowIdx * EXACT_NN_M128N64_TM], &As_rd[dotIdx * (EXACT_NN_M128N64_BM + EXACT_NN_M128N64_SMEM_A_PAD) + warpRow * EXACT_NN_M128N64_WM + wSubRowIdx * EXACT_NN_M128N64_WSUBM + threadRowInWarp * EXACT_NN_M128N64_TM]);
            #pragma unroll
            for (int wSubColIdx = 0; wSubColIdx < EXACT_NN_M128N64_WNITER; ++wSubColIdx)
                scalar_load_fragment<EXACT_NN_M128N64_TN>(&regN[wSubColIdx * EXACT_NN_M128N64_TN], &Bs_rd[dotIdx * (EXACT_NN_M128N64_BN + EXACT_NN_M128N64_SMEM_B_PAD) + warpCol * EXACT_NN_M128N64_WN + wSubColIdx * EXACT_NN_M128N64_WSUBN + threadColInWarp * EXACT_NN_M128N64_TN]);
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < EXACT_NN_M128N64_WMITER; ++wSubRowIdx) {
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < EXACT_NN_M128N64_WNITER; ++wSubColIdx) {
                    #pragma unroll
                    for (int resIdxM = 0; resIdxM < EXACT_NN_M128N64_TM; ++resIdxM) {
                        #pragma unroll
                        for (int resIdxN = 0; resIdxN < EXACT_NN_M128N64_TN; ++resIdxN) {
                            // explicit __fmaf_rn for bit-exact
                            // match with CPU `_mm256_fmadd_ps`.
                            int idx = (wSubRowIdx * EXACT_NN_M128N64_TM + resIdxM) * (EXACT_NN_M128N64_WNITER * EXACT_NN_M128N64_TN)
                                      + wSubColIdx * EXACT_NN_M128N64_TN + resIdxN;
                            threadResults[idx] = __fmaf_rn(
                                regM[wSubRowIdx * EXACT_NN_M128N64_TM + resIdxM],
                                regN[wSubColIdx * EXACT_NN_M128N64_TN + resIdxN],
                                threadResults[idx]);
                        }
                    }
                }
            }
        }
        if (tile + 1 == fold_tile && fold_tile < num_k_tiles) {
            // The piece's chain is added to the kept sum in order, and the
            // chain restarts from +0.0.
            #pragma unroll
            for (int result = 0; result < RESULTS; ++result) {
                kept[result] = __fadd_rn(kept[result], threadResults[result]);
                threadResults[result] = 0.0f;
            }
            fold_tile += piece_tiles;
        }
        // Rotate stages.
        read_stage = (read_stage + 1) % K_PIPE;
        write_stage = (write_stage + 1) % K_PIPE;
    }
    #undef ISSUE_EXACT_NN_M128N64_TILE

    // The last piece meets the earlier pieces' sum, then the bias: the order
    // the chunk fold's reduction adds the planes and the bias in. The plan
    // serves a cut layer with alpha 1 and beta 0 only, so the epilogue below
    // stores the result as it is.
    #pragma unroll
    for (int wSubColIdx = 0; wSubColIdx < EXACT_NN_M128N64_WNITER; ++wSubColIdx) {
        #pragma unroll
        for (int resIdxN = 0; resIdxN < EXACT_NN_M128N64_TN; ++resIdxN) {
            int g_col = pid_n * EXACT_NN_M128N64_BN + warpCol * EXACT_NN_M128N64_WN
                + wSubColIdx * EXACT_NN_M128N64_WSUBN + threadColInWarp * EXACT_NN_M128N64_TN
                + resIdxN;
            bool biased = bias != nullptr && g_col < N;
            float add = biased ? bias[g_col] : 0.0f;
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < EXACT_NN_M128N64_WMITER; ++wSubRowIdx) {
                #pragma unroll
                for (int resIdxM = 0; resIdxM < EXACT_NN_M128N64_TM; ++resIdxM) {
                    int idx = (wSubRowIdx * EXACT_NN_M128N64_TM + resIdxM)
                            * (EXACT_NN_M128N64_WNITER * EXACT_NN_M128N64_TN)
                        + wSubColIdx * EXACT_NN_M128N64_TN + resIdxN;
                    float value = __fadd_rn(kept[idx], threadResults[idx]);
                    threadResults[idx] = biased ? __fadd_rn(value, add) : value;
                }
            }
        }
    }

    // Epilogue: write results with alpha and beta (float4 stores)
    #pragma unroll
    for (int wSubRowIdx = 0; wSubRowIdx < EXACT_NN_M128N64_WMITER; ++wSubRowIdx) {
        #pragma unroll
        for (int wSubColIdx = 0; wSubColIdx < EXACT_NN_M128N64_WNITER; ++wSubColIdx) {
            float* C_sub = C_warp + wSubRowIdx * EXACT_NN_M128N64_WSUBM * ldc + wSubColIdx * EXACT_NN_M128N64_WSUBN;
            #pragma unroll
            for (int resIdxM = 0; resIdxM < EXACT_NN_M128N64_TM; ++resIdxM) {
                int g_row = pid_m * EXACT_NN_M128N64_BM + warpRow * EXACT_NN_M128N64_WM + wSubRowIdx * EXACT_NN_M128N64_WSUBM +
                            threadRowInWarp * EXACT_NN_M128N64_TM + resIdxM;
                if (g_row >= M) continue;
                #pragma unroll
                for (int resIdxN = 0; resIdxN < EXACT_NN_M128N64_TN; resIdxN += 4) {
                    int g_col = pid_n * EXACT_NN_M128N64_BN + warpCol * EXACT_NN_M128N64_WN + wSubColIdx * EXACT_NN_M128N64_WSUBN +
                                threadColInWarp * EXACT_NN_M128N64_TN + resIdxN;
                    // Fallback to scalar on column tail OR when ldc % 4 != 0.
                    // STG.128 needs 16-byte aligned address — row-stride in bytes
                    // (ldc * 4) must be a multiple of 16, so ldc must be a multiple
                    // of 4. Otherwise odd rows hit CUDA_ERROR_MISALIGNED_ADDRESS.
                    if (g_col + 3 >= N || (ldc & 3) != 0 || !is_aligned_16(C)) {
                        #pragma unroll
                        for (int j = 0; j < 4 && g_col + j < N; j++) {
                            int idx = (wSubRowIdx * EXACT_NN_M128N64_TM + resIdxM) * (EXACT_NN_M128N64_WNITER * EXACT_NN_M128N64_TN) +
                                      wSubColIdx * EXACT_NN_M128N64_TN + resIdxN + j;
                            float val = alpha * threadResults[idx];
                            if (beta != 0.0f) val += beta * C_sub[(threadRowInWarp * EXACT_NN_M128N64_TM + resIdxM) * ldc + threadColInWarp * EXACT_NN_M128N64_TN + resIdxN + j];
                            C_sub[(threadRowInWarp * EXACT_NN_M128N64_TM + resIdxM) * ldc + threadColInWarp * EXACT_NN_M128N64_TN + resIdxN + j] = val;
                        }
                        continue;
                    }
                    float4 tmp;
                    if (beta != 0.0f) {
                        tmp = reinterpret_cast<float4*>(
                            &C_sub[(threadRowInWarp * EXACT_NN_M128N64_TM + resIdxM) * ldc +
                                   threadColInWarp * EXACT_NN_M128N64_TN + resIdxN])[0];
                    }
                    int idx = (wSubRowIdx * EXACT_NN_M128N64_TM + resIdxM) * (EXACT_NN_M128N64_WNITER * EXACT_NN_M128N64_TN) +
                              wSubColIdx * EXACT_NN_M128N64_TN + resIdxN;
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
                        &C_sub[(threadRowInWarp * EXACT_NN_M128N64_TM + resIdxM) * ldc +
                               threadColInWarp * EXACT_NN_M128N64_TN + resIdxN])[0] = out;
                }
            }
        }
    }
    // The one-tile CTA exits after disjoint global stores; no shared state follows.
    } // end one-tile CTA scope
}

// The second set of 64 accumulators a thread holds for the finished pieces'
// sum leaves registers for one CTA per multiprocessor.
extern "C" __global__ __launch_bounds__(EXACT_NN_M128N64_NUM_THREADS, 1)
void nn_m128n64_pieces_keep(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float alpha, float beta,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    int piece_tiles
) {
    nn_m128n64_pieces_impl(C, A, B, bias, alpha, beta, M, N, K, lda, ldb, ldc, piece_tiles);
}

#undef EXACT_NN_M128N64_ROW_STRIDE_B
#undef EXACT_NN_M128N64_WSUBN
#undef EXACT_NN_M128N64_WSUBM
#undef EXACT_NN_M128N64_WMITER
#undef EXACT_NN_M128N64_GROUP_M
#undef EXACT_NN_M128N64_SMEM_B_PAD
#undef EXACT_NN_M128N64_SMEM_A_PAD
#undef EXACT_NN_M128N64_WARP_SIZE
#undef EXACT_NN_M128N64_NUM_THREADS
#undef EXACT_NN_M128N64_TN
#undef EXACT_NN_M128N64_TM
#undef EXACT_NN_M128N64_WNITER
#undef EXACT_NN_M128N64_WN
#undef EXACT_NN_M128N64_WM
#undef EXACT_NN_M128N64_BK
#undef EXACT_NN_M128N64_BN
#undef EXACT_NN_M128N64_BM
