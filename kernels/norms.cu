// RMSNorm CUDA kernels (forward + backward).
//
// Each sample processed by one thread block with shared memory reduction.
// Grid: (batch, 1, 1). Block: (min(next_power_of_2(dim), 1024), 1, 1).
// Strided loop handles dim > blockDim.x (e.g., dim=2048 with 1024 threads).
//
// Reference: Zhang & Sennrich (2019), "Root Mean Square Layer Normalization"

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

extern "C" __global__ void rmsnorm_forward(
    float* y, float* rms_out,
    const float* x, const float* scale,
    int batch, int dim, float eps
) {
    int b = blockIdx.x;
    if (b >= batch) return;
    int d = threadIdx.x;

    extern __shared__ float sdata[];

    int off = b * dim;

    // Strided accumulation: each thread sums multiple elements when dim > blockDim.x
    float sum = 0.0f;
    for (int i = d; i < dim; i += blockDim.x) {
        float val = x[off + i];
        sum += val * val;
    }
    sdata[d] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (d < s) {
            sdata[d] += sdata[d + s];
        }
        __syncthreads();
    }
    if (d < 32) {
        float v = sdata[d];
        if (d + 32 < blockDim.x) v += sdata[d + 32];
        v = warp_reduce_sum(v);
        if (d == 0) sdata[0] = v;
    }
    __syncthreads();

    float rms = sqrtf(sdata[0] / (float)dim + eps);
    if (d == 0) {
        rms_out[b] = rms;
    }
    __syncthreads();

    float inv_rms = 1.0f / rms;
    // Strided output write: each thread writes multiple elements when dim > blockDim.x
    for (int i = d; i < dim; i += blockDim.x) {
        y[off + i] = x[off + i] * inv_rms * scale[i];
    }
}

extern "C" __global__ void rmsnorm_backward(
    float* dx, float* d_scale,
    const float* dy, const float* x, const float* scale,
    const float* rms_saved,
    int batch, int dim
) {
    int b = blockIdx.x;
    if (b >= batch) return;
    int d = threadIdx.x;

    extern __shared__ float sdata[];

    int off = b * dim;
    float inv_rms = 1.0f / rms_saved[b];

    // Strided accumulation of dy * y_hat for the reduction
    float sum = 0.0f;
    for (int i = d; i < dim; i += blockDim.x) {
        float x_hat = x[off + i] * inv_rms;
        float dy_val = dy[off + i];
        float y_val = x_hat * scale[i];
        sum += dy_val * y_val;
    }
    sdata[d] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (d < s) {
            sdata[d] += sdata[d + s];
        }
        __syncthreads();
    }
    if (d < 32) {
        float v = sdata[d];
        if (d + 32 < blockDim.x) v += sdata[d + 32];
        v = warp_reduce_sum(v);
        if (d == 0) sdata[0] = v;
    }
    __syncthreads();

    float mean_dy_y = sdata[0] / (float)dim;

    // Strided gradient write: each thread handles multiple elements when dim > blockDim.x
    for (int i = d; i < dim; i += blockDim.x) {
        float x_hat = x[off + i] * inv_rms;
        float dy_val = dy[off + i];
        dx[off + i] = (scale[i] * dy_val - x_hat * mean_dy_y) * inv_rms;
        atomicAdd(&d_scale[i], dy_val * x_hat);
    }
}
