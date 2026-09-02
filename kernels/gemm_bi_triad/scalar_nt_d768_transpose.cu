__device__ __forceinline__ void transpose_f32_32x16_d768_body(
    float* __restrict__ destination,
    const float* __restrict__ source,
    int rows,
    int columns
) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    #pragma unroll
    for (int offset = 0; offset < 32; offset += 16) {
        int row = y + offset;
        tile[threadIdx.y + offset][threadIdx.x] =
            row < rows && x < columns ? source[(long long)row * columns + x] : 0.0f;
    }
    __syncthreads();

    int output_column = blockIdx.y * 32 + threadIdx.x;
    int output_row = blockIdx.x * 32 + threadIdx.y;
    #pragma unroll
    for (int offset = 0; offset < 32; offset += 16) {
        int row = output_row + offset;
        if (row < columns && output_column < rows) {
            destination[(long long)row * rows + output_column] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

extern "C" __global__ __launch_bounds__(512, 2)
void gemm_bi_transpose_f32_32x16_d768_v1(
    float* __restrict__ destination,
    const float* __restrict__ source,
    int rows,
    int columns
) {
    transpose_f32_32x16_d768_body(destination, source, rows, columns);
}
