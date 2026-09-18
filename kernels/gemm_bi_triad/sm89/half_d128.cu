/*
 * Half-precision TN weight gradient for the d_model-128 classifier shapes:
 * output[KOut x N] += alpha * A^T[KOut x 1024] . B[1024 x N] with bf16 or
 * f16 operands, f32 accumulation and an f32 output, reduced over the 1024
 * rows of both operands.
 *
 * The numeric contract is the tiled 64x64 kernel's: every output element is
 * one ascending chain of m16n8k16 accumulations over 64-wide slabs of the
 * reduction, seeded from zero, scaled by alpha once and added to the output
 * in f32. Only the CTA shape and the pipeline differ: a 32x16 tile gives the
 * tiny outputs four to eight CTAs per multiprocessor instead of a fraction
 * of one, and a four-deep cp.async ring with L2-only caching keeps a chain of
 * sixteen slabs from exposing a load latency per slab. The shared staging is
 * XOR-swizzled so the transposing ldmatrix reads stay bank-conflict free.
 */

namespace tn_sm89_half_d128 {

constexpr int MRed = 1024;

__device__ __forceinline__ bool aligned_16(const void* pointer) {
    return ((unsigned long long)pointer & 15ULL) == 0ULL;
}

template <bool CacheGlobal>
__device__ __forceinline__ void copy_16(unsigned destination, const void* source) {
    if (CacheGlobal) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(destination), "l"(source));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                     :: "r"(destination), "l"(source));
    }
}

__device__ __forceinline__ void commit_group() {
    asm volatile("cp.async.commit_group;\n");
}

template <int Pending>
__device__ __forceinline__ void wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Pending));
}

template <typename T>
struct MmaType;

template <>
struct MmaType<__nv_bfloat16> {
    static __device__ __forceinline__ void mma(
        float (&acc)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};

template <>
struct MmaType<__half> {
    static __device__ __forceinline__ void mma(
        float (&acc)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};

// Shared layout of one operand slab: BK rows of Width halfs, each row a
// run of Width/8 sixteen-byte chunks; the chunk index is xored with a
// function of the row so that eight consecutive rows of one chunk column
// never share a bank group. Rows of 16 halfs: chunk ^ (row/4 % 2); rows of
// 32: chunk ^ (row/2 % 4); rows of 64: chunk ^ (row % 8). A row of 8 halfs
// is a single chunk, and eight consecutive rows already cover all banks.
template <int Width>
struct Swizzle {
    static constexpr int Chunks = Width / 8;
    static_assert(Chunks == 1 || Chunks == 2 || Chunks == 4 || Chunks == 8,
                  "supported row widths are 8, 16, 32, 64");
    static __device__ __forceinline__ int chunk_of(int row, int chunk) {
        return chunk ^ ((row * Chunks / 8) % Chunks);
    }
    // Byte offset of element (row, column) with column a multiple of 8.
    static __device__ __forceinline__ unsigned offset(int row, int column) {
        return (unsigned)((row * Chunks + chunk_of(row, column / 8)) * 16);
    }
};

template <typename T, int KOut, int N, int TileM, int TileN, int BK, int Stages, bool CacheGlobal>
struct Kernel {
    static constexpr int WarpsM = TileM / 16;
    static constexpr int WarpsN = TileN / 8;
    static constexpr int Warps = WarpsM * WarpsN;
    static constexpr int Threads = Warps * 32;
    static constexpr int Tiles = MRed / BK;
    static constexpr int AStageBytes = BK * TileM * 2;
    static constexpr int BStageBytes = BK * TileN * 2;
    static constexpr int StageBytes = AStageBytes + BStageBytes;
    static constexpr int SharedBytes = Stages * StageBytes;
    static constexpr int AChunks = BK * TileM / 8;
    static constexpr int BChunks = BK * TileN / 8;
    static constexpr int StageChunks = AChunks + BChunks;
    static constexpr int ChunksPerThread = (StageChunks + Threads - 1) / Threads;
    static constexpr int RowTiles = KOut / TileM;
    static constexpr int ColumnTiles = N / TileN;
    static constexpr int Grid = RowTiles * ColumnTiles;

    static_assert(MRed % BK == 0, "stages cover the reduction exactly");
    static_assert(BK % 16 == 0, "a stage holds whole k16 steps");
    static_assert(TileM % 16 == 0 && TileN % 8 == 0, "warps own 16x8 blocks");
    static_assert(KOut % TileM == 0 && N % TileN == 0, "tiles cover the output exactly");
    static_assert(Stages >= 2, "the ring needs a landed stage and one in flight");

    using SwA = Swizzle<TileM>;
    using SwB = Swizzle<TileN>;

    struct CopyPlan {
        const T* source[ChunksPerThread];
        unsigned destination[ChunksPerThread];
        int advance[ChunksPerThread];
        bool valid[ChunksPerThread];
    };

    static __device__ __forceinline__ CopyPlan plan_copies(
        const T* __restrict__ a, const T* __restrict__ b,
        int output_row_base, int output_column_base) {
        CopyPlan plan;
        #pragma unroll
        for (int i = 0; i < ChunksPerThread; ++i) {
            int chunk = (int)threadIdx.x + i * Threads;
            plan.valid[i] = chunk < StageChunks;
            if (chunk < AChunks) {
                int row = chunk / (TileM / 8);
                int column = (chunk - row * (TileM / 8)) * 8;
                plan.source[i] = a + (long long)row * KOut + output_row_base + column;
                plan.destination[i] = SwA::offset(row, column);
                plan.advance[i] = BK * KOut;
            } else {
                int w = chunk - AChunks;
                int row = w / (TileN / 8);
                int column = (w - row * (TileN / 8)) * 8;
                plan.source[i] = b + (long long)row * N + output_column_base + column;
                plan.destination[i] = AStageBytes + SwB::offset(row, column);
                plan.advance[i] = BK * N;
            }
        }
        return plan;
    }

    static __device__ __forceinline__ void issue_stage(unsigned stage_base, CopyPlan& plan) {
        #pragma unroll
        for (int i = 0; i < ChunksPerThread; ++i) {
            if (plan.valid[i]) {
                copy_16<CacheGlobal>(stage_base + plan.destination[i], plan.source[i]);
            }
            plan.source[i] += plan.advance[i];
        }
        commit_group();
    }

    static __device__ __forceinline__ void run(
        float* __restrict__ output,
        const T* __restrict__ a,
        const T* __restrict__ b,
        float alpha,
        int m,
        int k,
        int n
    ) {
        if (!output || !a || !b || m != MRed || k != KOut || n != N
            || blockDim.x != Threads || blockDim.y != 1 || blockDim.z != 1
            || gridDim.x != Grid || gridDim.y != 1 || gridDim.z != 1
            || !aligned_16(a) || !aligned_16(b)) {
            return;
        }
        int tile_m = (int)blockIdx.x / ColumnTiles;
        int tile_n = (int)blockIdx.x - tile_m * ColumnTiles;
        int output_row_base = tile_m * TileM;
        int output_column_base = tile_n * TileN;

        int warp = (int)threadIdx.x >> 5;
        int lane = (int)threadIdx.x & 31;
        int warp_m = (warp / WarpsN) * 16;
        int warp_n = (warp - (warp / WarpsN) * WarpsN) * 8;
        int group = lane >> 2;
        int thread = lane & 3;
        int matrix_row = lane & 7;
        int matrix_quad = lane >> 3;
        // ldmatrix quadrants as the tiled TN kernel assigns them:
        // A x4.trans: k offset (quad & 2 ? 8 : 0), m offset (quad & 1 ? 8 : 0);
        // B x2.trans: k offset (quad & 1 ? 8 : 0).
        int a_row_offset = (matrix_quad & 2) ? 8 : 0;
        int a_column_offset = (matrix_quad & 1) ? 8 : 0;
        int b_row_offset = (matrix_quad & 1) ? 8 : 0;

        extern __shared__ __align__(16) unsigned char shared[];
        unsigned shared_base = (unsigned)__cvta_generic_to_shared(shared);
        CopyPlan plan = plan_copies(a, b, output_row_base, output_column_base);

        #pragma unroll
        for (int s = 0; s < Stages - 1; ++s) {
            if (s < Tiles) {
                issue_stage(shared_base + s * StageBytes, plan);
            } else {
                commit_group();
            }
        }

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll 1
        for (int tile = 0; tile < Tiles; ++tile) {
            wait_group<Stages - 2>();
            __syncthreads();
            int prefetch = tile + Stages - 1;
            if (prefetch < Tiles) {
                issue_stage(shared_base + (prefetch % Stages) * StageBytes, plan);
            } else {
                commit_group();
            }
            unsigned a_stage = shared_base + (tile % Stages) * StageBytes;
            unsigned b_stage = a_stage + AStageBytes;
            #pragma unroll
            for (int ks = 0; ks < BK / 16; ++ks) {
                int k0 = ks * 16;
                unsigned a_frag[4];
                unsigned b_frag[2];
                {
                    int row = k0 + a_row_offset + matrix_row;
                    int column = warp_m + a_column_offset;
                    unsigned address = a_stage + SwA::offset(row, column);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                                 "{%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(a_frag[0]), "=r"(a_frag[1]),
                                   "=r"(a_frag[2]), "=r"(a_frag[3])
                                 : "r"(address));
                }
                {
                    int row = k0 + b_row_offset + matrix_row;
                    unsigned address = b_stage + SwB::offset(row, warp_n);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                                 "{%0,%1}, [%2];\n"
                                 : "=r"(b_frag[0]), "=r"(b_frag[1])
                                 : "r"(address));
                }
                MmaType<T>::mma(acc, a_frag, b_frag);
            }
        }

        int row0 = output_row_base + warp_m + group;
        int column0 = output_column_base + warp_n + 2 * thread;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            int row = row0 + (e >= 2 ? 8 : 0);
            int column = column0 + (e & 1);
            long long index = (long long)row * N + column;
            output[index] += alpha * acc[e];
        }
    }
};

} // namespace tn_sm89_half_d128

#define TN_HALF_D128_EXPORT(SYMBOL, T, KOUT, N, TM, TN, BK, STAGES, CG, MINBLOCKS) \
    extern "C" __global__ __launch_bounds__( \
        tn_sm89_half_d128::Kernel<T, KOUT, N, TM, TN, BK, STAGES, CG>::Threads, MINBLOCKS) \
    void SYMBOL(float* output, const T* a, const T* b, float alpha, int m, int k, int n) { \
        tn_sm89_half_d128::Kernel<T, KOUT, N, TM, TN, BK, STAGES, CG>::run( \
            output, a, b, alpha, m, k, n); \
    }

// in_proj: output 128 x 512, 32x16 tiles, four warps, 128 CTAs.
TN_HALF_D128_EXPORT(tn_sm89_half_d128_in_m32n16_bk64_s4_cg_bf16, __nv_bfloat16, 128, 512, 32, 16, 64, 4, true, 3)
TN_HALF_D128_EXPORT(tn_sm89_half_d128_in_m32n16_bk64_s4_cg_f16, __half, 128, 512, 32, 16, 64, 4, true, 3)
// out_proj: output 256 x 128, 32x16 tiles, four warps, 64 CTAs.
TN_HALF_D128_EXPORT(tn_sm89_half_d128_out_m32n16_bk64_s4_cg_bf16, __nv_bfloat16, 256, 128, 32, 16, 64, 4, true, 3)
TN_HALF_D128_EXPORT(tn_sm89_half_d128_out_m32n16_bk64_s4_cg_f16, __half, 256, 128, 32, 16, 64, 4, true, 3)

#undef TN_HALF_D128_EXPORT
