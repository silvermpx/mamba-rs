// Exact-F32 SM120 routes: the production ascending-k FMA chain behind an 8x8
// register microtile, fed by TMA, for NN, TN and NT, with a deterministic
// reduction split. Compiled into the TriadSm120 module after sm120.cu, which
// declares the tensor-map descriptor type.
//
// Kernel-side terms: C[M][N] = sum_k A(M, k) * B(k, N) over the reduction K.
//   NN: A stored [M][K], B stored [K][N]
//   TN: A stored [K][M], B stored [K][N]
//   NT: A stored [M][K], B stored [N][K]  (TMA 64-byte swizzle on B)
//
// Bits. With one split every output walks its reduction in ascending k with
// one __fmaf_rn per step starting from the bias (or zero), exactly the
// scalar 64x64 NN chain. With S splits the host deals the portable plan's
// pieces: unit (tile, s) accumulates its contiguous range of k tiles from
// zero, splits 0..S-2 publish partial slabs, and split S-1 owns the tile and
// folds the pieces the way the portable reducers do, so a product has the
// same bits as on any other architecture: p0 + ... + p(S-1) with __fadd_rn
// in that order, then alpha, then the bias for NN (splitk_reduce) or the old
// output for TN (splitm_reduce).
//
// Shape of the schedule:
//   - A and B tiles arrive through cp.async.bulk.tensor, one 2d box each
//     per k tile, issued by a single thread and completed on an mbarrier;
//     the compute warps never touch the load/store pipe for operands.
//   - Stages are dense (no padding); the fragment reads are float4 and
//     conflict free in both warp shapes: NN/NT A rows are interleaved and
//     read as float4 along k every four steps, TN A is read as float4 along
//     m every step, NN/TN B as float4 along n every step, NT B through the
//     swizzled rows of its stage.
//   - Two shared stages; the barrier at the top of a k tile is what
//     retires the stage refilled during that tile.
//   - Slabs travel through L2 (st.cg / ld.cg); the owner waits on
//     release/acquire flags that it resets afterwards, so a launch leaves
//     the flag array as it found it.
#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)

#define SM120_FMA_BK 16
#define SM120_FMA_TM 8
#define SM120_FMA_TN 8
#define SM120_FMA_WARP_SIZE 32
#define SM120_FMA_GROUP_M 16
#define SM120_FMA_K_PIPE 2

struct Sm120FmaKernelParams {
    float alpha;
    float beta;
    int m;
    int n;
    int k;
    int ldc;
    int splits;
    int tiles_per_split;
};

static_assert(sizeof(Sm120FmaKernelParams) == 32,
              "SM120 FMA parameter ABI drift");
static_assert(alignof(Sm120FmaKernelParams) == 4,
              "SM120 FMA parameter alignment drift");
static_assert(__is_standard_layout(Sm120FmaKernelParams),
              "SM120 FMA parameters must remain standard layout");

enum { SM120_FMA_OP_NN = 0, SM120_FMA_OP_TN = 1, SM120_FMA_OP_NT = 2 };

template <int Arrivals>
static __device__ __forceinline__ void sm120_fma_init_barrier(unsigned barrier) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(barrier), "n"(Arrivals) : "memory");
}

static __device__ __forceinline__ void sm120_fma_wait_barrier(
    unsigned barrier, unsigned phase) {
    unsigned ready;
    do {
        asm volatile(
            "{ .reg .pred p; "
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
            "p, [%1], %2; "
            "selp.b32 %0, 1, 0, p; }"
            : "=r"(ready) : "r"(barrier), "r"(phase) : "memory");
    } while (!ready);
}

template <int Bytes>
static __device__ __forceinline__ void sm120_fma_expect_transaction(
    unsigned barrier) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
        "_, [%0], %1;"
        :: "r"(barrier), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void sm120_fma_tma_copy(
    unsigned destination, const CUtensorMap* map, int x, int y,
    unsigned barrier) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(destination), "l"(reinterpret_cast<unsigned long long>(map)),
           "r"(x), "r"(y), "r"(barrier)
        : "memory");
}

static __device__ __forceinline__ void sm120_fma_flag_raise(unsigned* flag) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;"
                 :: "l"(flag), "r"(1u) : "memory");
}

static __device__ __forceinline__ unsigned sm120_fma_flag_read(
    const unsigned* flag) {
    unsigned value;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                 : "=r"(value) : "l"(flag) : "memory");
    return value;
}

template <int OP, int BM, int BN, int WM, int WN, int THREADS, bool NT_KVEC>
__device__ __forceinline__ void sm120_fma_body(
    float* __restrict__ C,
    const float* __restrict__ bias,
    float* __restrict__ partials,
    unsigned* __restrict__ flags,
    const Sm120FmaKernelParams& params,
    const CUtensorMap& a_map,
    const CUtensorMap& b_map
) {
    constexpr int BK = SM120_FMA_BK;
    constexpr int TM = SM120_FMA_TM;
    constexpr int TN = SM120_FMA_TN;
    constexpr int K_PIPE = SM120_FMA_K_PIPE;
    constexpr int A_STAGE = BM * BK;
    constexpr int B_STAGE = BK * BN;
    constexpr int STAGE_FLOATS = A_STAGE + B_STAGE;
    constexpr int STAGE_BYTES = STAGE_FLOATS * (int)sizeof(float);
    constexpr int WARPS = THREADS / SM120_FMA_WARP_SIZE;
    constexpr int LANE_COLUMNS = WN / TN;
    constexpr int LANE_ROWS = WM / TM;
    constexpr int SLAB_FLOATS = BM * BN;
    static_assert(WARPS == (BM / WM) * (BN / WN),
                  "warp grid does not cover the block tile");
    static_assert(LANE_COLUMNS * LANE_ROWS == SM120_FMA_WARP_SIZE,
                  "lane grid does not cover the warp tile");
    static_assert(THREADS * TM * TN == SLAB_FLOATS,
                  "the thread microtiles must cover the slab");
    static_assert(BK == 16, "the NT swizzle assumes 64-byte stage rows");
    static_assert(STAGE_BYTES % 1024 == 0,
                  "TMA stages must keep the swizzle span aligned");
    static_assert(!NT_KVEC || OP == SM120_FMA_OP_NT,
                  "the k-vector B fragment only exists for NT");

    extern __shared__ __align__(1024) float sm120_fma_smem[];
    float* stages = sm120_fma_smem;
    unsigned smem_base = __cvta_generic_to_shared(sm120_fma_smem);
    unsigned barrier_base = smem_base + K_PIPE * STAGE_BYTES;

    // A negative split count asks one CTA to walk every piece of its tile:
    // -splits pieces of tiles_per_split k tiles, each a chain from zero,
    // folded in piece order through the tile's slab; k tiles past the
    // pieces are the tail a portable plan adds after the fold, and they
    // continue the folded chain.
    bool fold_pieces = params.splits < 0;
    int pieces = fold_pieces ? -params.splits : 1;
    int splits = fold_pieces ? 1 : params.splits;
    int unit = blockIdx.x;
    int tile_id = unit / splits;
    int split = unit - tile_id * splits;
    bool owner = split == splits - 1;

    int num_pid_m = (params.m + BM - 1) / BM;
    int num_pid_n = (params.n + BN - 1) / BN;
    int num_pid_in_group = SM120_FMA_GROUP_M * num_pid_n;
    int group_id = tile_id / num_pid_in_group;
    int first_pid_m = group_id * SM120_FMA_GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, SM120_FMA_GROUP_M);
    int pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m);
    int pid_n = (tile_id % num_pid_in_group) / group_size_m;

    int warp = threadIdx.x / SM120_FMA_WARP_SIZE;
    int lane = threadIdx.x % SM120_FMA_WARP_SIZE;
    int warp_row = warp / (BN / WN);
    int warp_column = warp % (BN / WN);
    int thread_column = lane % LANE_COLUMNS;
    int thread_row = lane / LANE_COLUMNS;

    // Ownership. NN/NT rows are interleaved (float4 along k per row); TN rows
    // are contiguous (float4 along m per k step). NN/TN columns are 4-wide
    // chunks; NT columns are interleaved scalars read from swizzled rows.
    auto local_row = [&](int i) -> int {
        return OP == SM120_FMA_OP_TN
            ? warp_row * WM + thread_row * TM + i
            : warp_row * WM + thread_row + i * LANE_ROWS;
    };
    auto local_column = [&](int c) -> int {
        return OP == SM120_FMA_OP_NT
            ? warp_column * WN + thread_column + LANE_COLUMNS * c
            : warp_column * WN + (c / 4) * (LANE_COLUMNS * 4)
                + thread_column * 4 + (c % 4);
    };

    float threadResults[TM * TN];
    if (splits == 1 && !fold_pieces && bias != nullptr) {
        #pragma unroll
        for (int c = 0; c < TN; ++c) {
            int global_column = pid_n * BN + local_column(c);
            float bias_value =
                global_column < params.n ? bias[global_column] : 0.0f;
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                threadResults[i * TN + c] = bias_value;
            }
        }
    } else {
        #pragma unroll
        for (int result = 0; result < TM * TN; ++result) {
            threadResults[result] = 0.0f;
        }
    }

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < K_PIPE; ++stage) {
            sm120_fma_init_barrier<1>(barrier_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    // Out-of-range rows and columns arrive as zeros from the tensor maps,
    // which is exactly what the production loader zero-fills.
    auto produce = [&](int stage, int k_tile) {
        unsigned barrier = barrier_base + stage * 8;
        unsigned destination = smem_base + stage * STAGE_BYTES;
        unsigned b_destination = destination + A_STAGE * (int)sizeof(float);
        sm120_fma_expect_transaction<STAGE_BYTES>(barrier);
        if (OP == SM120_FMA_OP_TN) {
            sm120_fma_tma_copy(destination, &a_map, pid_m * BM, k_tile * BK, barrier);
        } else {
            sm120_fma_tma_copy(destination, &a_map, k_tile * BK, pid_m * BM, barrier);
        }
        if (OP == SM120_FMA_OP_NT) {
            sm120_fma_tma_copy(b_destination, &b_map, k_tile * BK, pid_n * BN, barrier);
        } else {
            sm120_fma_tma_copy(b_destination, &b_map, pid_n * BN, k_tile * BK, barrier);
        }
    };

    int num_k_tiles = (params.k + BK - 1) / BK;
    int span = fold_pieces ? num_k_tiles : params.tiles_per_split;
    int begin = split * span;
    int end = min(begin + span, num_k_tiles);
    int range = max(end - begin, 0);
    int piece_end = fold_pieces ? min(pieces * params.tiles_per_split, range) : 0;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < K_PIPE - 1; ++stage) {
            if (stage < range) {
                produce(stage, begin + stage);
            }
        }
    }
    // One k tile of the pipelined chain. A folding CTA runs it in a loop of
    // its own that stops at every piece's end; the plain loop below keeps the
    // codegen of a unit that never folds.
    auto k_step = [&](int step) {
        int stage = step % K_PIPE;
        sm120_fma_wait_barrier(barrier_base + stage * 8,
                          (unsigned)((step / K_PIPE) & 1));
        // The stage refilled below was read one iteration ago; this barrier
        // is what proves every thread has left it.
        __syncthreads();
        int next = step + K_PIPE - 1;
        if (threadIdx.x == 0 && next < range) {
            produce(next % K_PIPE, begin + next);
        }

        const float* As_read = stages + stage * STAGE_FLOATS;
        const float* Bs_read = As_read + A_STAGE;
        float a_fragment[TM][4];
        float b_fragment[TN][4];

        #pragma unroll
        for (int dot_index = 0; dot_index < BK; ++dot_index) {
            if (OP == SM120_FMA_OP_TN) {
                #pragma unroll
                for (int i = 0; i < TM; i += 4) {
                    float4 value = *reinterpret_cast<const float4*>(
                        As_read + dot_index * BM + local_row(0) + i);
                    a_fragment[i + 0][0] = value.x;
                    a_fragment[i + 1][0] = value.y;
                    a_fragment[i + 2][0] = value.z;
                    a_fragment[i + 3][0] = value.w;
                }
            } else if (dot_index % 4 == 0) {
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    float4 value = *reinterpret_cast<const float4*>(
                        As_read + local_row(i) * BK + dot_index);
                    a_fragment[i][0] = value.x;
                    a_fragment[i][1] = value.y;
                    a_fragment[i][2] = value.z;
                    a_fragment[i][3] = value.w;
                }
            }
            if (OP == SM120_FMA_OP_NT) {
                if (NT_KVEC) {
                    if (dot_index % 4 == 0) {
                        #pragma unroll
                        for (int c = 0; c < TN; ++c) {
                            int row = local_column(c);
                            int chunk = (dot_index / 4) ^ ((row >> 1) & 3);
                            float4 value = *reinterpret_cast<const float4*>(
                                Bs_read + row * BK + chunk * 4);
                            b_fragment[c][0] = value.x;
                            b_fragment[c][1] = value.y;
                            b_fragment[c][2] = value.z;
                            b_fragment[c][3] = value.w;
                        }
                    }
                } else {
                    #pragma unroll
                    for (int c = 0; c < TN; ++c) {
                        int row = local_column(c);
                        int chunk = (dot_index / 4) ^ ((row >> 1) & 3);
                        b_fragment[c][0] =
                            Bs_read[row * BK + chunk * 4 + (dot_index % 4)];
                    }
                }
            } else {
                #pragma unroll
                for (int j = 0; j < TN / 4; ++j) {
                    float4 value = *reinterpret_cast<const float4*>(
                        Bs_read + dot_index * BN + local_column(j * 4));
                    b_fragment[j * 4 + 0][0] = value.x;
                    b_fragment[j * 4 + 1][0] = value.y;
                    b_fragment[j * 4 + 2][0] = value.z;
                    b_fragment[j * 4 + 3][0] = value.w;
                }
            }

            // Keep this row-major result nest in ascending reduction order.
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                float a = OP == SM120_FMA_OP_TN
                    ? a_fragment[i][0]
                    : a_fragment[i][dot_index % 4];
                #pragma unroll
                for (int c = 0; c < TN; ++c) {
                    float b = (OP == SM120_FMA_OP_NT && NT_KVEC)
                        ? b_fragment[c][dot_index % 4]
                        : b_fragment[c][0];
                    threadResults[i * TN + c] =
                        __fmaf_rn(a, b, threadResults[i * TN + c]);
                }
            }
        }
    };

    int step = 0;
    if (fold_pieces) {
        // The end of a piece: its chain joins the fold, which waits in this
        // thread's own cells of the tile's slab until the last piece takes
        // it back into the registers, where a tail then continues it.
        float* slab = partials + (size_t)tile_id * SLAB_FLOATS;
        for (int piece_start = 0; piece_start < piece_end;
             piece_start += params.tiles_per_split) {
            int stop = min(piece_start + params.tiles_per_split, piece_end);
            for (; step < stop; ++step) {
                k_step(step);
            }
            bool first = piece_start == 0;
            bool last = stop == piece_end;
            #pragma unroll
            for (int chunk = 0; chunk < TM * TN / 4; ++chunk) {
                float4* cell = reinterpret_cast<float4*>(
                    slab + ((size_t)chunk * THREADS + threadIdx.x) * 4);
                float* own = threadResults + chunk * 4;
                float4 sum = {own[0], own[1], own[2], own[3]};
                if (!first) {
                    float4 held = __ldcg(cell);
                    sum.x = __fadd_rn(held.x, own[0]);
                    sum.y = __fadd_rn(held.y, own[1]);
                    sum.z = __fadd_rn(held.z, own[2]);
                    sum.w = __fadd_rn(held.w, own[3]);
                }
                if (last) {
                    own[0] = sum.x;
                    own[1] = sum.y;
                    own[2] = sum.z;
                    own[3] = sum.w;
                } else {
                    __stcg(cell, sum);
                    own[0] = 0.0f;
                    own[1] = 0.0f;
                    own[2] = 0.0f;
                    own[3] = 0.0f;
                }
            }
        }
    }
    // The whole range of a plain unit, the tail of a folding one.
    for (; step < range; ++step) {
        k_step(step);
    }

    if (fold_pieces && OP == SM120_FMA_OP_TN) {
        #pragma unroll
        for (int result = 0; result < TM * TN; ++result) {
            threadResults[result] = __fmul_rn(params.alpha, threadResults[result]);
        }
    }

    if (splits > 1) {
        // Slab layout is chunk-major so a warp's float4 stores and loads
        // stay contiguous.
        float* slabs = partials + (size_t)tile_id * (splits - 1) * SLAB_FLOATS;
        unsigned* tile_flags = flags + (size_t)tile_id * (splits - 1);
        if (!owner) {
            float* slab = slabs + (size_t)split * SLAB_FLOATS;
            #pragma unroll
            for (int chunk = 0; chunk < TM * TN / 4; ++chunk) {
                float4 value = {
                    threadResults[chunk * 4 + 0],
                    threadResults[chunk * 4 + 1],
                    threadResults[chunk * 4 + 2],
                    threadResults[chunk * 4 + 3]
                };
                __stcg(reinterpret_cast<float4*>(
                           slab + ((size_t)chunk * THREADS + threadIdx.x) * 4),
                       value);
            }
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) {
                sm120_fma_flag_raise(tile_flags + split);
            }
            return;
        }
        if (threadIdx.x == 0) {
            for (int s = 0; s < splits - 1; ++s) {
                while (sm120_fma_flag_read(tile_flags + s) == 0u) {
                }
            }
        }
        __syncthreads();
        // One float4 of the microtile at a time, through every slab in split
        // order and then the owner's own piece.
        #pragma unroll
        for (int chunk = 0; chunk < TM * TN / 4; ++chunk) {
            const float* first = slabs + ((size_t)chunk * THREADS + threadIdx.x) * 4;
            float4 head = __ldcg(reinterpret_cast<const float4*>(first));
            float* own = threadResults + chunk * 4;
            float sum0 = head.x;
            float sum1 = head.y;
            float sum2 = head.z;
            float sum3 = head.w;
            for (int s = 1; s < splits - 1; ++s) {
                float4 value = __ldcg(reinterpret_cast<const float4*>(
                    first + (size_t)s * SLAB_FLOATS));
                sum0 = __fadd_rn(sum0, value.x);
                sum1 = __fadd_rn(sum1, value.y);
                sum2 = __fadd_rn(sum2, value.z);
                sum3 = __fadd_rn(sum3, value.w);
            }
            own[0] = __fadd_rn(sum0, own[0]);
            own[1] = __fadd_rn(sum1, own[1]);
            own[2] = __fadd_rn(sum2, own[2]);
            own[3] = __fadd_rn(sum3, own[3]);
            // A weight gradient scales here, rounded once, so the epilogue
            // only adds it to the old value, as splitm_reduce does.
            if (OP == SM120_FMA_OP_TN) {
                own[0] = __fmul_rn(params.alpha, own[0]);
                own[1] = __fmul_rn(params.alpha, own[1]);
                own[2] = __fmul_rn(params.alpha, own[2]);
                own[3] = __fmul_rn(params.alpha, own[3]);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int s = 0; s < splits - 1; ++s) {
                tile_flags[s] = 0u;
            }
        }
    }

    // A split weight gradient already carries alpha in its folded sum; a
    // split NN product adds its bias after alpha, as the portable reducer
    // does, where one split seeded the chain with it.
    bool folded = splits > 1 || fold_pieces;
    float alpha = OP == SM120_FMA_OP_TN && folded ? 1.0f : params.alpha;
    const float* bias_last = folded ? bias : nullptr;
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_row = pid_m * BM + local_row(i);
        if (global_row >= params.m) {
            continue;
        }
        if (OP == SM120_FMA_OP_NT) {
            #pragma unroll
            for (int c = 0; c < TN; ++c) {
                int global_column = pid_n * BN + local_column(c);
                if (global_column >= params.n) {
                    continue;
                }
                float* destination =
                    C + (long long)global_row * params.ldc + global_column;
                float value = alpha * threadResults[i * TN + c];
                if (params.beta != 0.0f) {
                    value += params.beta * destination[0];
                }
                destination[0] = value;
            }
            continue;
        }
        #pragma unroll
        for (int j = 0; j < TN / 4; ++j) {
            int global_column = pid_n * BN + local_column(j * 4);
            int idx = i * TN + j * 4;
            float* destination =
                C + (long long)global_row * params.ldc + global_column;
            if (global_column + 3 >= params.n || (params.ldc & 3) != 0
                || !is_aligned_16(C)) {
                #pragma unroll
                for (int element = 0;
                     element < 4 && global_column + element < params.n;
                     ++element) {
                    float value = alpha * threadResults[idx + element];
                    if (bias_last != nullptr) {
                        value += bias_last[global_column + element];
                    }
                    if (params.beta != 0.0f) {
                        value += params.beta * destination[element];
                    }
                    destination[element] = value;
                }
                continue;
            }

            float4 previous;
            if (params.beta != 0.0f) {
                previous = reinterpret_cast<float4*>(destination)[0];
            }
            float v0 = alpha * threadResults[idx + 0];
            float v1 = alpha * threadResults[idx + 1];
            float v2 = alpha * threadResults[idx + 2];
            float v3 = alpha * threadResults[idx + 3];
            if (bias_last != nullptr) {
                v0 += bias_last[global_column + 0];
                v1 += bias_last[global_column + 1];
                v2 += bias_last[global_column + 2];
                v3 += bias_last[global_column + 3];
            }
            if (params.beta != 0.0f) {
                v0 += params.beta * previous.x;
                v1 += params.beta * previous.y;
                v2 += params.beta * previous.z;
                v3 += params.beta * previous.w;
            }
            float4 output = {v0, v1, v2, v3};
            reinterpret_cast<float4*>(destination)[0] = output;
        }
    }
}

// The parameter order follows the SM120 stream-K ABI: output, slabs, flags,
// the two tensor maps, bias, then the parameter bundle.
#define SM120_DEFINE_FMA_KERNEL(NAME, OP, BM, BN, WM, WN, THREADS, MIN_BLOCKS, NT_KVEC) \
    extern "C" __global__ __launch_bounds__(THREADS, MIN_BLOCKS)                     \
    void NAME(                                                                        \
        float* __restrict__ C,                                                        \
        float* __restrict__ partials,                                                 \
        unsigned* __restrict__ flags,                                                 \
        const __grid_constant__ CUtensorMap a_map,                                    \
        const __grid_constant__ CUtensorMap b_map,                                    \
        const float* __restrict__ bias,                                               \
        const __grid_constant__ Sm120FmaKernelParams params                           \
    ) {                                                                               \
        assert(params.alpha == 1.0f || bias == nullptr);                              \
        sm120_fma_body<OP, BM, BN, WM, WN, THREADS, NT_KVEC>(                         \
            C, bias, partials, flags, params, a_map, b_map);                          \
    }

// 128x64: four warps stacked, each 32x64. 64x128: four warps side by side,
// each 64x32. 64x64: two warps stacked, each 32x64. The NT k-vector arms
// keep a float4-along-k B fragment and pay for it with two blocks per SM.
SM120_DEFINE_FMA_KERNEL(nn_sm120_tma_fma_m128n64_bk16_s2, SM120_FMA_OP_NN, 128, 64, 32, 64, 128, 3, false)
SM120_DEFINE_FMA_KERNEL(nn_sm120_tma_fma_m64n128_bk16_s2, SM120_FMA_OP_NN, 64, 128, 64, 32, 128, 3, false)
SM120_DEFINE_FMA_KERNEL(nn_sm120_tma_fma_m64n64_bk16_s2, SM120_FMA_OP_NN, 64, 64, 32, 64, 64, 5, false)
SM120_DEFINE_FMA_KERNEL(tn_sm120_tma_fma_m128n64_bk16_s2, SM120_FMA_OP_TN, 128, 64, 32, 64, 128, 3, false)
SM120_DEFINE_FMA_KERNEL(tn_sm120_tma_fma_m64n128_bk16_s2, SM120_FMA_OP_TN, 64, 128, 64, 32, 128, 3, false)
SM120_DEFINE_FMA_KERNEL(tn_sm120_tma_fma_m64n64_bk16_s2, SM120_FMA_OP_TN, 64, 64, 32, 64, 64, 5, false)
SM120_DEFINE_FMA_KERNEL(nt_sm120_tma_fma_m128n64_bk16_s2, SM120_FMA_OP_NT, 128, 64, 32, 64, 128, 3, false)
SM120_DEFINE_FMA_KERNEL(nt_sm120_tma_fma_m64n128_bk16_s2, SM120_FMA_OP_NT, 64, 128, 64, 32, 128, 3, false)
SM120_DEFINE_FMA_KERNEL(nt_sm120_tma_fma_m64n64_bk16_s2, SM120_FMA_OP_NT, 64, 64, 32, 64, 64, 5, false)
SM120_DEFINE_FMA_KERNEL(nt_sm120_tma_fma_m128n64_bk16_s2_kvec, SM120_FMA_OP_NT, 128, 64, 32, 64, 128, 2, true)
SM120_DEFINE_FMA_KERNEL(nt_sm120_tma_fma_m64n128_bk16_s2_kvec, SM120_FMA_OP_NT, 64, 128, 64, 32, 128, 2, true)
SM120_DEFINE_FMA_KERNEL(nt_sm120_tma_fma_m64n64_bk16_s2_kvec, SM120_FMA_OP_NT, 64, 64, 32, 64, 64, 4, true)

#undef SM120_DEFINE_FMA_KERNEL
#undef SM120_FMA_K_PIPE
#undef SM120_FMA_GROUP_M
#undef SM120_FMA_WARP_SIZE
#undef SM120_FMA_TN
#undef SM120_FMA_TM
#undef SM120_FMA_BK

#endif
