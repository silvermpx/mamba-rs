template <int BM>
__device__ __forceinline__ void gemm_bi_tn_d128_fused_contract_body(
    float* __restrict__ destination,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float alpha,
    int M,
    int K,
    int N
) {
    constexpr int BN = 32;
    constexpr int CHUNK = 16;
    constexpr int CHUNKS = 64;
    constexpr int OUTPUTS_PER_THREAD = BM * BN / 128;
    __shared__ float a_tile[BM][CHUNK];
    __shared__ float b_tile[CHUNK][BN];

    if (!(M == 1024 && K == 128 && N == 512)) return;

    int output_row_base = blockIdx.x * BM;
    int output_column_base = blockIdx.y * 32;
    float partial[OUTPUTS_PER_THREAD];
    double sums[OUTPUTS_PER_THREAD];

    #pragma unroll 1
    for (int chunk = 0; chunk < CHUNKS; ++chunk) {
        int reduction_base = chunk * CHUNK;
        for (int index = threadIdx.x; index < BM * CHUNK; index += 128) {
            int row = index / CHUNK;
            int reduction_offset = index % CHUNK;
            a_tile[row][reduction_offset] =
                a[(long long)(reduction_base + reduction_offset) * K
                    + output_row_base + row];
        }
        for (int index = threadIdx.x; index < CHUNK * BN; index += 128) {
            int reduction_offset = index / BN;
            int column = index % BN;
            b_tile[reduction_offset][column] =
                b[(long long)(reduction_base + reduction_offset) * N
                    + output_column_base + column];
        }
        __syncthreads();

        #pragma unroll
        for (int owned = 0; owned < OUTPUTS_PER_THREAD; ++owned) {
            partial[owned] = 0.0f;
            int linear = threadIdx.x + owned * 128;
            int row = linear / BN;
            int column = linear % BN;
            #pragma unroll
            for (int reduction_offset = 0; reduction_offset < CHUNK; ++reduction_offset) {
                partial[owned] = __fmaf_rn(
                    a_tile[row][reduction_offset],
                    b_tile[reduction_offset][column],
                    partial[owned]);
            }
            if (chunk == 0) {
                sums[owned] = (double)partial[owned];
            } else {
                double sum = sums[owned];
                sum = __dadd_rn(sum, (double)partial[owned]);
                sums[owned] = sum;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int owned = 0; owned < OUTPUTS_PER_THREAD; ++owned) {
        int linear = threadIdx.x + owned * 128;
        int row = output_row_base + linear / BN;
        int column = output_column_base + linear % BN;
        double sum = sums[owned];
        double scaled = __dmul_rn((double)alpha, sum);
        float& output = destination[(long long)row * N + column];
        output += __double2float_rn(scaled);
    }
}

extern "C" __global__ __launch_bounds__(128, 4)
void gemm_bi_tn_d128_fused_m8n32_bk16_exp(
    float* destination,
    const float* a,
    const float* b,
    float alpha,
    int M,
    int K,
    int N
) {
    gemm_bi_tn_d128_fused_contract_body<8>(destination, a, b, alpha, M, K, N);
}

extern "C" __global__ __launch_bounds__(128, 3)
void gemm_bi_tn_d128_fused_m16n32_bk16_exp(
    float* destination,
    const float* a,
    const float* b,
    float alpha,
    int M,
    int K,
    int N
) {
    gemm_bi_tn_d128_fused_contract_body<16>(destination, a, b, alpha, M, K, N);
}
