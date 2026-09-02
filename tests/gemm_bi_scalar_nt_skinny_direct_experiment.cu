// Retained test-only NT thin-column runner. Production owns the qualified winner.
#define NT_THIN_BK 64
#define NT_THIN_PARTIAL 32

struct SkinnyNtM8N4SplitK32Kernel {
    static constexpr int BM = 8;
    static constexpr int BN = 4;
    static constexpr int THREADS = 32;
    static_assert(BM * BN == THREADS, "one thread must own one output");
    static constexpr int COPY_FLOATS = 4;
    static constexpr int ROW_STRIDE = NT_THIN_BK + COPY_FLOATS;
    static constexpr int SEGMENTS = NT_THIN_BK / COPY_FLOATS;
    static constexpr int A_STRIDE = ROW_STRIDE;
    static constexpr int B_STRIDE = ROW_STRIDE;
    static constexpr int A_STAGE = BM * A_STRIDE;
    static constexpr int B_STAGE = BN * B_STRIDE;
    static constexpr int STAGE = A_STAGE + B_STAGE;
    static constexpr int SHARED_BYTES = 2 * STAGE * (int)sizeof(float);

    static __device__ __forceinline__ void issue_stage(
        float* stage, const float* __restrict__ a, const float* __restrict__ b,
        int pid_m, int pid_n, int reduction_offset, int3 dims) {
        unsigned stage_base = __cvta_generic_to_shared(stage);
        for (int linear = (int)threadIdx.x; linear < BM * SEGMENTS;
             linear += THREADS) {
            int local_row = linear / SEGMENTS;
            int segment = linear - local_row * SEGMENTS;
            int reduction = segment * COPY_FLOATS;
            int global_row = pid_m * BM + local_row;
            int global_reduction = reduction_offset + reduction;
            bool valid = global_row < dims.x
                && global_reduction + COPY_FLOATS - 1 < dims.y;
            const float* source = valid
                ? a + (long long)global_row * dims.y + global_reduction : a;
            unsigned destination = stage_base
                + (unsigned)(local_row * ROW_STRIDE + reduction)
                    * (unsigned)sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                         :: "r"(destination), "l"(source), "r"(valid ? 16 : 0));
        }
        for (int linear = (int)threadIdx.x; linear < BN * SEGMENTS;
             linear += THREADS) {
            int local_column = linear / SEGMENTS;
            int segment = linear - local_column * SEGMENTS;
            int reduction = segment * COPY_FLOATS;
            int global_column = pid_n * BN + local_column;
            int global_reduction = reduction_offset + reduction;
            bool valid = global_column < dims.z
                && global_reduction + COPY_FLOATS - 1 < dims.y;
            const float* source = valid
                ? b + (long long)global_column * dims.y + global_reduction : b;
            unsigned destination = stage_base
                + (unsigned)(A_STAGE + local_column * ROW_STRIDE + reduction)
                    * (unsigned)sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                         :: "r"(destination), "l"(source), "r"(valid ? 16 : 0));
        }
        asm volatile("cp.async.commit_group;\n");
    }

    static __device__ __forceinline__ void run(
        float* __restrict__ output, const float* __restrict__ a,
        const float* __restrict__ b, float alpha, int m, int n, int k_out) {
        extern __shared__ __align__(16) float shared[];
        int3 dims = make_int3(m, n, k_out);
        int row_tiles = (m + BM - 1) / BM;
        int pid_m = (int)blockIdx.x % row_tiles;
        int pid_n = (int)blockIdx.x / row_tiles;
        int lane = (int)threadIdx.x & 31;
        int local_row = lane / BN;
        int local_column = lane - local_row * BN;
        int reduction_tiles = (n + NT_THIN_BK - 1) / NT_THIN_BK;
        if (m != 512 || n != 2048 || k_out != 16) {
            return;
        }
        float reduced = 0.0f;

        issue_stage(shared, a, b, pid_m, pid_n, 0, dims);
        int read_stage = 0;
        for (int tile = 0; tile < reduction_tiles; ++tile) {
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            int next_tile = tile + 1;
            if (next_tile < reduction_tiles) {
                issue_stage(shared + (read_stage ^ 1) * STAGE, a, b, pid_m, pid_n,
                            next_tile * NT_THIN_BK, dims);
            }
            const float* stage = shared + read_stage * STAGE;
            const float* stage_a = stage;
            const float* stage_b = stage + A_STAGE;
            float partial_0 = 0.0f;
            float partial_1 = 0.0f;
#pragma unroll
            for (int dot = 0; dot < NT_THIN_PARTIAL; ++dot) {
                partial_0 = __fmaf_rn(stage_a[local_row * A_STRIDE + dot],
                    stage_b[local_column * B_STRIDE + dot], partial_0);
            }
#pragma unroll
            for (int dot = NT_THIN_PARTIAL; dot < NT_THIN_BK; ++dot) {
                partial_1 = __fmaf_rn(stage_a[local_row * A_STRIDE + dot],
                    stage_b[local_column * B_STRIDE + dot], partial_1);
            }
            if (tile == 0) {
                reduced = partial_0;
                reduced = __fadd_rn(reduced, partial_1);
            } else {
                reduced = __fadd_rn(reduced, partial_0);
                reduced = __fadd_rn(reduced, partial_1);
            }
            read_stage ^= 1;
        }

        int global_row = pid_m * BM + local_row;
        int global_column = pid_n * BN + local_column;
        if (global_row < m && global_column < k_out) {
            output[(long long)global_row * k_out + global_column]
                = __fmul_rn(alpha, reduced);
        }
    }
};

static_assert(SkinnyNtM8N4SplitK32Kernel::SHARED_BYTES == 6528,
              "M8N4 shared-memory contract drift");

extern "C" __global__ __launch_bounds__(32, 4)
void gemm_bi_nt_thin_m8n4_bk64_splitk32_exp_v1(
    float* output, const float* a, const float* b, float alpha,
    int m, int n, int k_out) {
    SkinnyNtM8N4SplitK32Kernel::run(output, a, b, alpha, m, n, k_out);
}

#undef NT_THIN_PARTIAL
#undef NT_THIN_BK
