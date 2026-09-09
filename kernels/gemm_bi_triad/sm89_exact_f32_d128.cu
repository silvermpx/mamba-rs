/*
 * Sealed SM89 exact-F32 TN d128 source template.
 *
 * The Rust owner expands this body exactly twice. The physical kernel body is
 * byte-derived from the retained Ada direct-fold discovery implementation;
 * only the namespace, shape, retained alias contract, and export symbol are
 * route parameters.
 */

// SM89_EXACT_F32_D128_ROUTE_BEGIN
namespace __SM89_EXACT_F32_D128_NAMESPACE__ {

constexpr int MRed = 1024;
constexpr int KOut = __SM89_EXACT_F32_D128_K_OUT__;
constexpr int N = __SM89_EXACT_F32_D128_N__;
constexpr int BK = 16;
constexpr int Chunks = MRed / BK;
constexpr int Stages = 2;

static_assert(MRed % BK == 0, "the exact reduction must contain whole chunks");
static_assert(Chunks == 64, "the Split-M reduction tree changed");

__device__ __forceinline__ bool aligned_16(const void* pointer) {
    return ((unsigned long long)pointer & 15ULL) == 0ULL;
}

template <bool CacheGlobal>
struct AsyncCopy;

template <>
struct AsyncCopy<false> {
    static __device__ __forceinline__ void copy(
        unsigned destination, const float* source) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                     :: "r"(destination), "l"(source));
    }
};

template <>
struct AsyncCopy<true> {
    static __device__ __forceinline__ void copy(
        unsigned destination, const float* source) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(destination), "l"(source));
    }
};

template <int TileM, int TileN, int ThreadCount, int APadding,
          bool CacheGlobal, bool FlatGrid>
struct Kernel {
    static constexpr int OutputsPerThread = TileM * TileN / ThreadCount;
    static constexpr int AStride = TileM + APadding;
    static constexpr int AStage = BK * AStride;
    static constexpr int BStage = BK * TileN;
    static constexpr int Stage = AStage + BStage;
    static constexpr int SharedBytes = Stages * Stage * (int)sizeof(float);
    static constexpr int RowTiles = KOut / TileM;
    static constexpr int ColumnTiles = N / TileN;
    static constexpr int AVectors = BK * TileM / 4;
    static constexpr int BVectors = BK * TileN / 4;

    static_assert(TileM * TileN == ThreadCount * OutputsPerThread,
                  "every output must have exactly one owner");
    static_assert(BK * TileM % 4 == 0, "A vectors must be complete");
    static_assert(BK * TileN % 4 == 0, "B vectors must be complete");

    static __device__ __forceinline__ void issue_stage(
        float* stage,
        const float* __restrict__ a,
        const float* __restrict__ b,
        int output_row_base,
        int output_column_base,
        int reduction_base
    ) {
        unsigned shared_base = __cvta_generic_to_shared(stage);

        for (int vector = (int)threadIdx.x; vector < AVectors;
             vector += ThreadCount) {
            int linear = vector << 2;
            int reduction = linear / TileM;
            int local = linear - reduction * TileM;
            int global_reduction = reduction_base + reduction;
            const float* source =
                a + (long long)global_reduction * KOut + output_row_base + local;
            unsigned destination = shared_base
                + (unsigned)(reduction * AStride + local) * sizeof(float);
            AsyncCopy<CacheGlobal>::copy(destination, source);
        }

        constexpr int BSegments = TileN / 4;
        for (int vector = (int)threadIdx.x; vector < BVectors;
             vector += ThreadCount) {
            int reduction = vector / BSegments;
            int local = (vector - reduction * BSegments) << 2;
            int global_reduction = reduction_base + reduction;
            const float* source =
                b + (long long)global_reduction * N + output_column_base + local;
            unsigned destination = shared_base
                + (unsigned)(AStage + reduction * TileN + local) * sizeof(float);
            AsyncCopy<CacheGlobal>::copy(destination, source);
        }
        asm volatile("cp.async.commit_group;\n");
    }

    static __device__ __forceinline__ void run(
        float* __restrict__ output,
        const float* __restrict__ a,
        const float* __restrict__ b,
        float alpha,
        int m,
        int k,
        int n
    ) {
        bool valid_grid = FlatGrid
            ? gridDim.x == RowTiles * ColumnTiles && gridDim.y == 1
            : gridDim.x == RowTiles && gridDim.y == ColumnTiles;
        if (!output || !a || !b || m != MRed || k != KOut || n != N
            || blockDim.x != ThreadCount
            || blockDim.y != 1 || blockDim.z != 1 || !valid_grid
            || gridDim.z != 1
            || !aligned_16(a) || !aligned_16(b)) {
            return;
        }

        int tile_m = FlatGrid ? (int)blockIdx.x % RowTiles : (int)blockIdx.x;
        int tile_n = FlatGrid ? (int)blockIdx.x / RowTiles : (int)blockIdx.y;
        int output_row_base = tile_m * TileM;
        int output_column_base = tile_n * TileN;
        constexpr int ColumnGroups = TileN / OutputsPerThread;
        int local_row = (int)threadIdx.x / ColumnGroups;
        int local_column_base =
            ((int)threadIdx.x - local_row * ColumnGroups) * OutputsPerThread;
        double sums[OutputsPerThread];
        extern __shared__ __align__(16) float shared[];

        issue_stage(shared, a, b, output_row_base, output_column_base, 0);
        int read_stage = 0;

        #pragma unroll 1
        for (int chunk = 0; chunk < Chunks; ++chunk) {
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();

            int next_chunk = chunk + 1;
            if (next_chunk < Chunks) {
                issue_stage(
                    shared + (read_stage ^ 1) * Stage,
                    a,
                    b,
                    output_row_base,
                    output_column_base,
                    next_chunk * BK);
            }

            const float* a_tile = shared + read_stage * Stage;
            const float* b_tile = a_tile + AStage;
            float partial[OutputsPerThread] = {0.0f};

            // Each chunk starts from positive zero, exactly like the old
            // Split-M partial kernel.
            #pragma unroll
            for (int reduction = 0; reduction < BK; ++reduction) {
                float a_value = a_tile[reduction * AStride + local_row];
                #pragma unroll
                for (int owned = 0; owned < OutputsPerThread; ++owned) {
                    partial[owned] = __fmaf_rn(
                        a_value,
                        b_tile[reduction * TileN + local_column_base + owned],
                        partial[owned]);
                }
            }

            // Chunk zero seeds the double accumulator directly. Later chunks
            // follow the reducer's ascending fc=1..15 add order.
            #pragma unroll
            for (int owned = 0; owned < OutputsPerThread; ++owned) {
                double value = (double)partial[owned];
                if (chunk == 0) {
                    sums[owned] = value;
                } else {
                    sums[owned] = __dadd_rn(sums[owned], value);
                }
            }
            read_stage ^= 1;
        }

        #pragma unroll
        for (int owned = 0; owned < OutputsPerThread; ++owned) {
            int row = output_row_base + local_row;
            int column = output_column_base + local_column_base + owned;
            long long index = (long long)row * N + column;
            double scaled = __dmul_rn((double)alpha, sums[owned]);
            output[index] = __fadd_rn(
                output[index], __double2float_rn(scaled));
        }
    }
};

using Retained = Kernel<__SM89_EXACT_F32_D128_TILE_M__, 16, 64, 0, false, true>;

static_assert(Retained::OutputsPerThread == __SM89_EXACT_F32_D128_OUTPUTS_PER_THREAD__,
              "retained thread ownership changed");
static_assert(Retained::SharedBytes == __SM89_EXACT_F32_D128_SHARED_BYTES__,
              "retained shared-memory contract changed");
static_assert(Retained::Stage == __SM89_EXACT_F32_D128_STAGE__,
              "retained stage extent changed");
static_assert(Retained::RowTiles * Retained::ColumnTiles == 256,
              "retained flat-grid contract changed");

} // namespace __SM89_EXACT_F32_D128_NAMESPACE__

extern "C" __global__ __launch_bounds__(64, 4)
void __SM89_EXACT_F32_D128_SYMBOL__(
    float* output,
    const float* a,
    const float* b,
    float alpha,
    int m,
    int k,
    int n
) {
    __SM89_EXACT_F32_D128_NAMESPACE__::Retained::run(
        output, a, b, alpha, m, k, n);
}
// SM89_EXACT_F32_D128_ROUTE_END
