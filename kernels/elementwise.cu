// Element-wise CUDA kernels for Mamba SSM.
//
// Bias broadcast, gating, SSM column gather/scatter, residual add, etc.
// All kernels: 1D grid, 256 threads/block.

extern "C" __global__ void bias_broadcast(
    float* y, const float* bias,
    int batch, int n_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_out;
    if (idx >= total) return;
    int j = idx % n_out;
    y[idx] = bias[j];
}

extern "C" __global__ void colsum_accumulate(
    float* db, const float* dy,
    int batch, int n_out
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    float sum = 0.0f;
    for (int b = 0; b < batch; b++) {
        sum += dy[b * n_out + j];
    }
    db[j] += sum;
}

extern "C" __global__ void vec_add_inplace(
    float* a, const float* b, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] += b[i];
}

extern "C" __global__ void elementwise_mul(
    float* y, const float* a, const float* b, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] * b[i];
}

extern "C" __global__ void exp_negate(
    float* y, const float* x, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = -exp2f(x[i] * 1.4426950408889634f);
}

extern "C" __global__ void gather_cols(
    float* dst, const float* src,
    int batch, int src_stride, int dst_dim, int offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dst_dim;
    if (idx >= total) return;
    int b = idx / dst_dim;
    int d = idx % dst_dim;
    dst[b * dst_dim + d] = src[b * src_stride + offset + d];
}

extern "C" __global__ void scatter_add_cols(
    float* dst, const float* src,
    int batch, int dst_stride, int src_dim, int offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * src_dim;
    if (idx >= total) return;
    int b = idx / src_dim;
    int d = idx % src_dim;
    dst[b * dst_stride + offset + d] += src[b * src_dim + d];
}

extern "C" __global__ void split_gate_silu(
    float* x_branch,       // [batch * d_inner] first half
    float* gate_pre_silu,  // [batch * d_inner] second half (saved for backward)
    float* gate_post_silu, // [batch * d_inner] SiLU(gate)
    const float* proj,     // [batch * 2*d_inner] in_proj output
    int batch, int d_inner
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_inner;
    if (idx >= total) return;
    int b = idx / d_inner;
    int d = idx % d_inner;
    int proj_off = b * 2 * d_inner;
    x_branch[idx] = proj[proj_off + d];
    float g = proj[proj_off + d_inner + d];
    gate_pre_silu[idx] = g;
    gate_post_silu[idx] = g / (1.0f + exp2f(-g * 1.4426950408889634f));
}

extern "C" __global__ void gating_backward(
    float* d_y,            // [n] gradient w.r.t. SSM output
    float* d_gate_pre,     // [n] gradient w.r.t. gate pre-SiLU
    const float* d_gated,  // [n] incoming gradient
    const float* y,        // [n] SSM output (saved)
    const float* gate_pre, // [n] gate pre-SiLU (saved)
    const float* gate_post,// [n] gate post-SiLU (saved)
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dg = d_gated[i];
    d_y[i] = dg * gate_post[i];
    // SiLU derivative: sigma * (1 + x * (1 - sigma))
    float x = gate_pre[i];
    float sigma = 1.0f / (1.0f + exp2f(-x * 1.4426950408889634f));
    d_gate_pre[i] = dg * y[i] * sigma * (1.0f + x * (1.0f - sigma));
}

extern "C" __global__ void concat_halves(
    float* proj,              // [batch * 2*d_inner] output
    const float* first_half,  // [batch * d_inner]
    const float* second_half, // [batch * d_inner]
    int batch, int d_inner
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_inner;
    if (idx >= total) return;
    int b = idx / d_inner;
    int d = idx % d_inner;
    int proj_off = b * 2 * d_inner;
    proj[proj_off + d] = first_half[idx];
    proj[proj_off + d_inner + d] = second_half[idx];
}

extern "C" __global__ void gather_last_timestep(
    float* __restrict__ dst,      // [B * D]
    const float* __restrict__ src, // [B * T * D]
    int B, int T, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D) return;
    int b = idx / D;
    int d = idx % D;
    dst[idx] = src[(b * T + (T - 1)) * D + d];
}

extern "C" __global__ void residual_add(
    float* dst, const float* a, const float* b, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = a[i] + b[i];
}

extern "C" __global__ void gather_bc_cols(
    float* dst_b, float* dst_c, const float* src,
    int batch, int src_stride, int ds, int b_offset, int c_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * ds;
    if (idx >= total) return;
    int b = idx / ds;
    int d = idx % ds;
    int row = b * src_stride;
    dst_b[b * ds + d] = src[row + b_offset + d];
    dst_c[b * ds + d] = src[row + c_offset + d];
}

extern "C" __global__ void softplus_copy(
    float* dst, const float* src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = src[i];
    // Numerically stable softplus using exp2f (matches softplus_forward in activations.cu)
    dst[i] = (x > 20.0f) ? x : logf(1.0f + exp2f(x * 1.4426950408889634f));
}
