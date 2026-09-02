#define GEMM_BI_NT_BK 64
#define GEMM_BI_NT_PARTIALS 64
#define GEMM_BI_NT_ROW_STRIDE 68

struct GemmBiNtM2N16Kernel {
    static constexpr int A_STAGE = 2 * GEMM_BI_NT_ROW_STRIDE;
    static constexpr int B_STAGE = 16 * GEMM_BI_NT_ROW_STRIDE;
    static constexpr int STAGE = A_STAGE + B_STAGE;
    static constexpr int SHARED_BYTES
        = (2 * STAGE + GEMM_BI_NT_PARTIALS * 32) * (int)sizeof(float);

    static __device__ __forceinline__ void issue_stage(
        float* stage, const float* __restrict__ a, const float* __restrict__ b,
        int pid_m, int pid_n, int reduction_offset, int3 dims) {
        unsigned base = __cvta_generic_to_shared(stage);
        for (int linear = (int)threadIdx.x; linear < 2 * 16; linear += 64) {
            int row = linear / 16;
            int reduction = (linear - row * 16) * 4;
            const float* source = a + (long long)(pid_m * 2 + row) * dims.y
                + reduction_offset + reduction;
            unsigned destination = base
                + (unsigned)(row * GEMM_BI_NT_ROW_STRIDE + reduction) * sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(destination), "l"(source));
        }
        for (int linear = (int)threadIdx.x; linear < 16 * 16; linear += 64) {
            int column = linear / 16;
            int reduction = (linear - column * 16) * 4;
            const float* source = b + (long long)(pid_n * 16 + column) * dims.y
                + reduction_offset + reduction;
            unsigned destination = base + (unsigned)(A_STAGE
                + column * GEMM_BI_NT_ROW_STRIDE + reduction) * sizeof(float);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(destination), "l"(source));
        }
        asm volatile("cp.async.commit_group;\n");
    }

    static __device__ __forceinline__ void run(
        float* __restrict__ output, const float* __restrict__ a,
        const float* __restrict__ b, float alpha, int m, int n, int k_out) {
        if (m != 512 || n != 2048 || k_out != 16) return;
        extern __shared__ __align__(16) float shared[];
        int3 dims = make_int3(m, n, k_out);
        int pid_m = (int)blockIdx.x & 255;
        int lane = (int)threadIdx.x & 31;
        int warp = (int)threadIdx.x >> 5;
        int row = lane / 16;
        int column = lane - row * 16;
        float* partials = shared + 2 * STAGE;
        issue_stage(shared, a, b, pid_m, 0, 0, dims);
        int read_stage = 0;
#pragma unroll
        for (int tile = 0; tile < 32; ++tile) {
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            if (tile + 1 < 32) {
                issue_stage(shared + (read_stage ^ 1) * STAGE, a, b, pid_m, 0,
                            (tile + 1) * GEMM_BI_NT_BK, dims);
            }
            const float* stage = shared + read_stage * STAGE;
            const float* stage_a = stage;
            const float* stage_b = stage + A_STAGE;
            float partial = 0.0f;
            int first = warp * 32;
#pragma unroll
            for (int dot = 0; dot < 32; ++dot) {
                partial = __fmaf_rn(
                    stage_a[row * GEMM_BI_NT_ROW_STRIDE + first + dot],
                    stage_b[column * GEMM_BI_NT_ROW_STRIDE + first + dot], partial);
            }
            partials[(2 * tile + warp) * 32 + lane] = partial;
            read_stage ^= 1;
        }
        __syncthreads();
        if (warp != 0) return;
        float reduced = partials[lane];
#pragma unroll
        for (int partial = 1; partial < GEMM_BI_NT_PARTIALS; ++partial) {
            reduced = __fadd_rn(reduced, partials[partial * 32 + lane]);
        }
        output[(long long)(pid_m * 2 + row) * 16 + column]
            = __fmul_rn(alpha, reduced);
    }
};

static_assert(GemmBiNtM2N16Kernel::SHARED_BYTES == 17984,
              "M2N16 shared-memory contract drift");

extern "C" __global__ __launch_bounds__(64, 4)
void gemm_bi_nt_m2n16_bk64_splitk32_v1(
    float* output, const float* a, const float* b, float alpha,
    int m, int n, int k_out) {
    GemmBiNtM2N16Kernel::run(output, a, b, alpha, m, n, k_out);
}

#undef GEMM_BI_NT_ROW_STRIDE
#undef GEMM_BI_NT_PARTIALS
#undef GEMM_BI_NT_BK
