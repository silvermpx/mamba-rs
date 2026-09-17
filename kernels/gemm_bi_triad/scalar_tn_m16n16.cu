struct GemmBiTnM16N16Kernel {
    static constexpr int M_RED = 256;
    static constexpr int K_OUT = 512;
    static constexpr int N = 384;
    static constexpr int BM = 16;
    static constexpr int BN = 16;
    static constexpr int BK = 16;
    static constexpr int THREADS = 64;
    static constexpr int CHUNKS = M_RED / BK;
    static constexpr int STAGES = 2;
    static constexpr int OUTPUTS_PER_THREAD = BM * BN / THREADS;
    static constexpr int A_STAGE = BK * BM;
    static constexpr int B_STAGE = BK * BN;
    static constexpr int STAGE = A_STAGE + B_STAGE;
    static constexpr int SHARED_BYTES = STAGES * STAGE * (int)sizeof(float);
    static constexpr int ROW_TILES = K_OUT / BM;
    static constexpr int COLUMN_TILES = N / BN;
    static constexpr int A_VECTORS = BK * BM / 4;
    static constexpr int B_VECTORS = BK * BN / 4;

    static __device__ __forceinline__ bool aligned_16(const void* pointer) {
        return ((unsigned long long)pointer & 15ULL) == 0ULL;
    }

    static __device__ __forceinline__ void copy_16(
        unsigned destination, const float* source) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                     :: "r"(destination), "l"(source));
    }

    static __device__ __forceinline__ void issue_stage(
        float* stage,
        const float* __restrict__ a,
        const float* __restrict__ b,
        int output_row_base,
        int output_column_base,
        int reduction_base
    ) {
        unsigned shared_base = __cvta_generic_to_shared(stage);

        for (int vector = (int)threadIdx.x; vector < A_VECTORS;
             vector += THREADS) {
            int linear = vector << 2;
            int reduction = linear / BM;
            int local = linear - reduction * BM;
            int global_reduction = reduction_base + reduction;
            const float* source =
                a + (long long)global_reduction * K_OUT + output_row_base + local;
            unsigned destination = shared_base
                + (unsigned)(reduction * BM + local) * sizeof(float);
            copy_16(destination, source);
        }

        constexpr int B_SEGMENTS = BN / 4;
        for (int vector = (int)threadIdx.x; vector < B_VECTORS;
             vector += THREADS) {
            int reduction = vector / B_SEGMENTS;
            int local = (vector - reduction * B_SEGMENTS) << 2;
            int global_reduction = reduction_base + reduction;
            const float* source =
                b + (long long)global_reduction * N + output_column_base + local;
            unsigned destination = shared_base
                + (unsigned)(A_STAGE + reduction * BN + local) * sizeof(float);
            copy_16(destination, source);
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
        if (!output || !a || !b || m != M_RED || k != K_OUT || n != N
            || blockDim.x != THREADS || blockDim.y != 1 || blockDim.z != 1
            || gridDim.x != ROW_TILES * COLUMN_TILES
            || gridDim.y != 1 || gridDim.z != 1
            || !aligned_16(a) || !aligned_16(b)) {
            return;
        }

        int tile_m = (int)blockIdx.x % ROW_TILES;
        int tile_n = (int)blockIdx.x / ROW_TILES;
        int output_row_base = tile_m * BM;
        int output_column_base = tile_n * BN;
        constexpr int COLUMN_GROUPS = BN / OUTPUTS_PER_THREAD;
        int local_row = (int)threadIdx.x / COLUMN_GROUPS;
        int local_column_base =
            ((int)threadIdx.x - local_row * COLUMN_GROUPS) * OUTPUTS_PER_THREAD;
        double sums[OUTPUTS_PER_THREAD];
        extern __shared__ __align__(16) float shared[];

        issue_stage(shared, a, b, output_row_base, output_column_base, 0);
        int read_stage = 0;

#pragma unroll 1
        for (int chunk = 0; chunk < CHUNKS; ++chunk) {
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();

            int next_chunk = chunk + 1;
            if (next_chunk < CHUNKS) {
                issue_stage(
                    shared + (read_stage ^ 1) * STAGE,
                    a,
                    b,
                    output_row_base,
                    output_column_base,
                    next_chunk * BK);
            }

            const float* a_tile = shared + read_stage * STAGE;
            const float* b_tile = a_tile + A_STAGE;
            float partial[OUTPUTS_PER_THREAD] = {0.0f};

            // Each partial follows the old Split-M chunk from positive zero.
#pragma unroll
            for (int reduction = 0; reduction < BK; ++reduction) {
                float a_value = a_tile[reduction * BM + local_row];
#pragma unroll
                for (int owned = 0; owned < OUTPUTS_PER_THREAD; ++owned) {
                    partial[owned] = __fmaf_rn(
                        a_value,
                        b_tile[reduction * BN + local_column_base + owned],
                        partial[owned]);
                }
            }

            // Chunk zero seeds the double sum; the remaining chunks add in
            // ascending Split-M order.
#pragma unroll
            for (int owned = 0; owned < OUTPUTS_PER_THREAD; ++owned) {
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
        for (int owned = 0; owned < OUTPUTS_PER_THREAD; ++owned) {
            int row = output_row_base + local_row;
            int column = output_column_base + local_column_base + owned;
            long long index = (long long)row * N + column;
            double scaled = __dmul_rn((double)alpha, sums[owned]);
            output[index] = __fadd_rn(
                output[index], __double2float_rn(scaled));
        }
    }
};

static_assert(GemmBiTnM16N16Kernel::CHUNKS == 16,
              "M16N16 Split-M tree changed");
static_assert(GemmBiTnM16N16Kernel::OUTPUTS_PER_THREAD == 4,
              "M16N16 output ownership changed");
static_assert(GemmBiTnM16N16Kernel::STAGE == 512,
              "M16N16 stage extent changed");
static_assert(GemmBiTnM16N16Kernel::SHARED_BYTES == 4096,
              "M16N16 shared-memory contract changed");
static_assert(GemmBiTnM16N16Kernel::ROW_TILES
                  * GemmBiTnM16N16Kernel::COLUMN_TILES == 768,
              "M16N16 grid contract changed");

extern "C" __global__ __launch_bounds__(64, 4)
void tn_m16n16_bk16_s2_splitm16(
    float* output,
    const float* a,
    const float* b,
    float alpha,
    int m,
    int k,
    int n
) {
    GemmBiTnM16N16Kernel::run(output, a, b, alpha, m, k, n);
}
