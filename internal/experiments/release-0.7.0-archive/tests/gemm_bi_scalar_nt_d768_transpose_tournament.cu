template <int BLOCK_ROWS>
__device__ __forceinline__ void transpose_d768_body(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int rows,
    int cols
) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    #pragma unroll
    for (int offset = 0; offset < 32; offset += BLOCK_ROWS) {
        int row = y + offset;
        tile[threadIdx.y + offset][threadIdx.x] =
            row < rows && x < cols ? src[(long long)row * cols + x] : 0.0f;
    }
    __syncthreads();

    int output_column = blockIdx.y * 32 + threadIdx.x;
    int output_row = blockIdx.x * 32 + threadIdx.y;
    #pragma unroll
    for (int offset = 0; offset < 32; offset += BLOCK_ROWS) {
        int row = output_row + offset;
        if (row < cols && output_column < rows) {
            dst[(long long)row * rows + output_column] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

extern "C" __global__ __launch_bounds__(512, 2)
void gemm_bi_transpose_f32_32x16_d768_exp(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int rows,
    int cols
) {
    transpose_d768_body<16>(dst, src, rows, cols);
}
