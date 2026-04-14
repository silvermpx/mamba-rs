// RMSNorm CUDA kernels (forward + backward).
//
// Each sample processed by one thread block with shared memory reduction.
// Grid: (batch, 1, 1). Block: (min(next_power_of_2(dim), 1024), 1, 1).
// Strided loop handles dim > blockDim.x (e.g., dim=2048 with 1024 threads).
//
// Forward templated over activation dtype. Reduction always in f32 for
// numerical stability (per CLAUDE.md §5.7 and bf16 mantissa precision).
// Scale weight stays f32 — it's a model parameter, not an activation.
//
// Reference: Zhang & Sennrich (2019), "Root Mean Square Layer Normalization"

#include "_typed_prelude.cuh"

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

// Templated forward: input/output in T_IN, reduction in f32, scale in f32.
#define DEFINE_RMSNORM_FWD(SUFFIX, T, FROM_F)                                \
extern "C" __global__ void rmsnorm_forward_##SUFFIX(                         \
    T* y, float* rms_out,                                                    \
    const T* x, const float* scale,                                          \
    int batch, int dim, float eps                                            \
) {                                                                          \
    int b = blockIdx.x;                                                      \
    if (b >= batch) return;                                                  \
    int d = threadIdx.x;                                                     \
    extern __shared__ float sdata[];                                         \
    int off = b * dim;                                                       \
    float sum = 0.0f;                                                        \
    for (int i = d; i < dim; i += blockDim.x) {                              \
        float v = to_f(x[off + i]);                                          \
        sum += v * v;                                                        \
    }                                                                        \
    sdata[d] = sum;                                                          \
    __syncthreads();                                                         \
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {                 \
        if (d < s) sdata[d] += sdata[d + s];                                 \
        __syncthreads();                                                     \
    }                                                                        \
    if (d < 32) {                                                            \
        float v = sdata[d];                                                  \
        if (d + 32 < blockDim.x) v += sdata[d + 32];                         \
        v = warp_reduce_sum(v);                                              \
        if (d == 0) sdata[0] = v;                                            \
    }                                                                        \
    __syncthreads();                                                         \
    float rms = sqrtf(sdata[0] / (float)dim + eps);                          \
    /* Finite-guard: if an upstream kernel produced NaN or +inf (bf16/f16    \
     * overflow on very deep models, 48+ layers), rms becomes non-finite    \
     * and inv_rms contaminates every subsequent layer. Fall back to 1.0    \
     * so output = x*scale without normalization — still wrong, but avoids  \
     * the silent NaN cascade that breaks the rest of the network. */       \
    if (!isfinite(rms) || rms < 1e-20f) rms = 1.0f;                          \
    if (d == 0) rms_out[b] = rms;                                            \
    __syncthreads();                                                         \
    float inv_rms = 1.0f / rms;                                              \
    for (int i = d; i < dim; i += blockDim.x) {                              \
        y[off + i] = FROM_F(to_f(x[off + i]) * inv_rms * scale[i]);          \
    }                                                                        \
}

DEFINE_RMSNORM_FWD(f32,  float,         from_f_f32)
DEFINE_RMSNORM_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_RMSNORM_FWD(f16,  __half,        from_f_f16)

// Dual-dtype variant: f32 input (residual-path), T_OUT output (bf16/f16).
// Used in end-to-end bf16 inference where residual stays f32 across layers
// but the branch fed into in_proj must be bf16 to match GEMM A dtype.
#define DEFINE_RMSNORM_FWD_F32IN(SUFFIX, T_OUT, FROM_F)                      \
extern "C" __global__ void rmsnorm_forward_f32in_##SUFFIX(                   \
    T_OUT* y, float* rms_out,                                                \
    const float* x, const float* scale,                                      \
    int batch, int dim, float eps                                            \
) {                                                                          \
    int b = blockIdx.x;                                                      \
    if (b >= batch) return;                                                  \
    int d = threadIdx.x;                                                     \
    extern __shared__ float sdata[];                                         \
    int off = b * dim;                                                       \
    float sum = 0.0f;                                                        \
    for (int i = d; i < dim; i += blockDim.x) {                              \
        float v = x[off + i];                                                \
        sum += v * v;                                                        \
    }                                                                        \
    sdata[d] = sum;                                                          \
    __syncthreads();                                                         \
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {                 \
        if (d < s) sdata[d] += sdata[d + s];                                 \
        __syncthreads();                                                     \
    }                                                                        \
    if (d < 32) {                                                            \
        float v = sdata[d];                                                  \
        if (d + 32 < blockDim.x) v += sdata[d + 32];                         \
        v = warp_reduce_sum(v);                                              \
        if (d == 0) sdata[0] = v;                                            \
    }                                                                        \
    __syncthreads();                                                         \
    float rms = sqrtf(sdata[0] / (float)dim + eps);                          \
    /* Finite-guard: if an upstream kernel produced NaN or +inf (bf16/f16    \
     * overflow on very deep models, 48+ layers), rms becomes non-finite    \
     * and inv_rms contaminates every subsequent layer. Fall back to 1.0    \
     * so output = x*scale without normalization — still wrong, but avoids  \
     * the silent NaN cascade that breaks the rest of the network. */       \
    if (!isfinite(rms) || rms < 1e-20f) rms = 1.0f;                          \
    if (d == 0) rms_out[b] = rms;                                            \
    __syncthreads();                                                         \
    float inv_rms = 1.0f / rms;                                              \
    for (int i = d; i < dim; i += blockDim.x) {                              \
        y[off + i] = FROM_F(x[off + i] * inv_rms * scale[i]);                \
    }                                                                        \
}

DEFINE_RMSNORM_FWD_F32IN(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_RMSNORM_FWD_F32IN(f16,  __half,        from_f_f16)

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
