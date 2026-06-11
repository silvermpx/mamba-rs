// Element-wise CUDA kernels for Mamba SSM.
//
// Bias broadcast, gating, SSM column gather/scatter, residual add, etc.
// All kernels: 1D grid, 256 threads/block.
//
// Activation-touching kernels are templated via extern "C" wrappers with
// suffixes (_f32, _bf16, _f16). Math in f32, storage in T_IN.

#include "_typed_prelude.cuh"

// ---------------------------------------------------------------------------
// Dtype cast kernels — for mixed-precision inference weight upload.
// f32 -> bf16: used when HF checkpoint is f32 but user requested bf16 storage.
// f32 -> f16:  same, but for f16 storage (rare — bf16 preferred for Mamba).
// ---------------------------------------------------------------------------

extern "C" __global__ void cast_f32_to_bf16(
    __nv_bfloat16* dst, const float* src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Round-to-nearest, matching the intermediate-value downcasts in all
    // typed kernels (from_f_bf16 in _typed_prelude.cuh). The default
    // `__float2bfloat16` is round-toward-zero, which adds a systematic
    // negative bias to every weight and compounds across GEMMs — visible
    // as degenerate greedy decoding on small models (e.g. mamba-130m).
    dst[i] = __float2bfloat16_rn(src[i]);
}

extern "C" __global__ void cast_f32_to_f16(
    __half* dst, const float* src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Same rationale as cast_f32_to_bf16: match the _rn rounding mode used
    // by from_f_f16 in _typed_prelude.cuh.
    dst[i] = __float2half_rn(src[i]);
}

// Step 10 — typed → f32 casts for the M3 mixed-precision backward,
// where some kernels (rmsnorm_bwd, m3_split_bwd's f32 inputs, etc.)
// are pure-f32 and need a typed staging buffer cast back to f32.
extern "C" __global__ void cast_bf16_to_f32(
    float* dst, const __nv_bfloat16* src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __bfloat162float(src[i]);
}

extern "C" __global__ void cast_f16_to_f32(
    float* dst, const __half* src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __half2float(src[i]);
}

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

// Generic 2D reduce-along-axis-0: out[d] = sum_b(partials[b * dim + d]).
// Used as the stage-2 finalizer after Rule-B per-sample partials writes,
// replacing atomicAdd accumulators with a deterministic tree reduction.
// Grid: dim blocks (one per output column). Block: next_pow2(batch).clamp(32, 256).
// Shared memory: block_dim * sizeof(float).
// Deterministic across runs: tree reduce in fixed order, single-thread write.
extern "C" __global__ __launch_bounds__(256, 4)
void reduce_sum_axis0(
    float* __restrict__ out,            // [dim]
    const float* __restrict__ partials, // [batch * dim]
    int batch, int dim,
    int accumulate                      // 0 = overwrite, 1 = +=
) {
    int d = blockIdx.x;
    if (d >= dim) return;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];

    float sum = 0.0f;
    for (int b = tid; b < batch; b += blockDim.x) {
        sum += partials[b * dim + d];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        out[d] = accumulate ? out[d] + sdata[0] : sdata[0];
    }
}

extern "C" __global__ void fill_scalar(
    float* dst, float val, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = val;
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

// gating_backward typed (bf16/f16/f32) for mixed-precision training.
// `y = ssm_out * gate_silu(z)` where z = gate_pre and gate_silu = z*sigma(z).
// All math in f32, activations T_IN (typed). Outputs typed.
// Reference math identical to f32 above; matches state-spaces/mamba
// selective_scan_bwd_kernel.cuh z/gate-branch backward.
#define DEFINE_GATING_BWD(SUFFIX, T, FROM_F)                                   \
extern "C" __global__ void gating_backward_##SUFFIX(                           \
    T* d_y, T* d_gate_pre,                                                     \
    const T* d_gated, const T* y,                                              \
    const T* gate_pre, const T* gate_post,                                     \
    int n                                                                      \
) {                                                                            \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (i >= n) return;                                                        \
    float dg = to_f(d_gated[i]);                                               \
    d_y[i] = FROM_F(dg * to_f(gate_post[i]));                                  \
    float xv = to_f(gate_pre[i]);                                              \
    float sigma = 1.0f / (1.0f + exp2f(-xv * 1.4426950408889634f));            \
    d_gate_pre[i] = FROM_F(dg * to_f(y[i]) * sigma * (1.0f + xv * (1.0f - sigma))); \
}

DEFINE_GATING_BWD(f32,  float,         from_f_f32)
DEFINE_GATING_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_GATING_BWD(f16,  __half,        from_f_f16)

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

// Templated gather_last_timestep — T_out may differ from T_in (e.g., f32 dst
// from bf16 src for mixed prefill when the downstream lm_head expects f32).
// When dst/src share the same dtype, this is a simple typed copy.
#define DEFINE_GATHER_LAST_TIMESTEP(SUFFIX, T)                                \
extern "C" __global__ void gather_last_timestep_##SUFFIX(                     \
    T* __restrict__ dst,                                                      \
    const T* __restrict__ src,                                                \
    int B, int Tlen, int D                                                    \
) {                                                                           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    if (idx >= B * D) return;                                                 \
    int b = idx / D;                                                          \
    int d = idx % D;                                                          \
    dst[idx] = src[(b * Tlen + (Tlen - 1)) * D + d];                          \
}

DEFINE_GATHER_LAST_TIMESTEP(f32,  float)
DEFINE_GATHER_LAST_TIMESTEP(bf16, __nv_bfloat16)
DEFINE_GATHER_LAST_TIMESTEP(f16,  __half)

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
    dst[i] = (x > 20.0f) ? x : log1pf(exp2f(x * 1.4426950408889634f));
}

// ===========================================================================
// Templated variants for activation-touching kernels.
// Suffix _f32/_bf16/_f16 — Rust dispatch selects by ctx.activation_dtype.
// Bias remains f32 (biases are always f32 in production LLMs).
// ===========================================================================

#define DEFINE_BIAS_BROADCAST(SUFFIX, T, FROM_F)                              \
extern "C" __global__ void bias_broadcast_##SUFFIX(                           \
    T* y, const float* bias, int batch, int n_out                             \
) {                                                                           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    int total = batch * n_out;                                                \
    if (idx >= total) return;                                                 \
    int j = idx % n_out;                                                      \
    y[idx] = FROM_F(bias[j]);                                                 \
}

DEFINE_BIAS_BROADCAST(f32,  float,         from_f_f32)
DEFINE_BIAS_BROADCAST(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_BIAS_BROADCAST(f16,  __half,        from_f_f16)

#define DEFINE_ELEMENTWISE_MUL(SUFFIX, T, FROM_F)                             \
extern "C" __global__ void elementwise_mul_##SUFFIX(                          \
    T* y, const T* a, const T* b, int n                                       \
) {                                                                           \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (i >= n) return;                                                       \
    y[i] = FROM_F(to_f(a[i]) * to_f(b[i]));                                   \
}

DEFINE_ELEMENTWISE_MUL(f32,  float,         from_f_f32)
DEFINE_ELEMENTWISE_MUL(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_ELEMENTWISE_MUL(f16,  __half,        from_f_f16)

#define DEFINE_RESIDUAL_ADD(SUFFIX, T, FROM_F)                                \
extern "C" __global__ void residual_add_##SUFFIX(                             \
    T* dst, const T* a, const T* b, int n                                     \
) {                                                                           \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (i >= n) return;                                                       \
    dst[i] = FROM_F(to_f(a[i]) + to_f(b[i]));                                 \
}

DEFINE_RESIDUAL_ADD(f32,  float,         from_f_f32)
DEFINE_RESIDUAL_ADD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_RESIDUAL_ADD(f16,  __half,        from_f_f16)

// Mixed residual add: f32 residual accumulator + bf16/f16 branch output
// → writes f32 (replaces residual in place or into a new f32 dst).
// Used in end-to-end bf16 inference where `residual_in_fp32` keeps the
// cross-layer residual stream f32 while per-layer branch outputs are bf16.
#define DEFINE_RESIDUAL_ADD_F32_T(SUFFIX, T_IN)                               \
extern "C" __global__ void residual_add_f32_##SUFFIX(                         \
    float* dst, const float* a, const T_IN* b, int n                          \
) {                                                                           \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (i >= n) return;                                                       \
    dst[i] = a[i] + to_f(b[i]);                                               \
}

DEFINE_RESIDUAL_ADD_F32_T(bf16, __nv_bfloat16)
DEFINE_RESIDUAL_ADD_F32_T(f16,  __half)

#define DEFINE_GATHER_COLS(SUFFIX, T)                                         \
extern "C" __global__ void gather_cols_##SUFFIX(                              \
    T* dst, const T* src,                                                     \
    int batch, int src_stride, int dst_dim, int offset                        \
) {                                                                           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    int total = batch * dst_dim;                                              \
    if (idx >= total) return;                                                 \
    int b = idx / dst_dim;                                                    \
    int d = idx % dst_dim;                                                    \
    dst[b * dst_dim + d] = src[b * src_stride + offset + d];                  \
}

DEFINE_GATHER_COLS(f32,  float)
DEFINE_GATHER_COLS(bf16, __nv_bfloat16)
DEFINE_GATHER_COLS(f16,  __half)

#define DEFINE_GATHER_BC(SUFFIX, T)                                           \
extern "C" __global__ void gather_bc_cols_##SUFFIX(                           \
    T* dst_b, T* dst_c, const T* src,                                         \
    int batch, int src_stride, int ds, int b_offset, int c_offset             \
) {                                                                           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    int total = batch * ds;                                                   \
    if (idx >= total) return;                                                 \
    int b = idx / ds;                                                         \
    int d = idx % ds;                                                         \
    int row = b * src_stride;                                                 \
    dst_b[b * ds + d] = src[row + b_offset + d];                              \
    dst_c[b * ds + d] = src[row + c_offset + d];                              \
}

DEFINE_GATHER_BC(f32,  float)
DEFINE_GATHER_BC(bf16, __nv_bfloat16)
DEFINE_GATHER_BC(f16,  __half)

#define DEFINE_SPLIT_GATE_SILU(SUFFIX, T, FROM_F)                             \
extern "C" __global__ void split_gate_silu_##SUFFIX(                          \
    T* x_branch, T* gate_pre_silu, T* gate_post_silu,                         \
    const T* proj, int batch, int d_inner                                     \
) {                                                                           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    int total = batch * d_inner;                                              \
    if (idx >= total) return;                                                 \
    int b = idx / d_inner;                                                    \
    int d = idx % d_inner;                                                    \
    int proj_off = b * 2 * d_inner;                                           \
    x_branch[idx] = proj[proj_off + d];                                       \
    float g = to_f(proj[proj_off + d_inner + d]);                             \
    gate_pre_silu[idx] = FROM_F(g);                                           \
    gate_post_silu[idx] = FROM_F(g / (1.0f + exp2f(-g * LOG2E)));             \
}

DEFINE_SPLIT_GATE_SILU(f32,  float,         from_f_f32)
DEFINE_SPLIT_GATE_SILU(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SPLIT_GATE_SILU(f16,  __half,        from_f_f16)

#define DEFINE_SOFTPLUS_COPY(SUFFIX, T, FROM_F)                               \
extern "C" __global__ void softplus_copy_##SUFFIX(                            \
    T* dst, const T* src, int n                                               \
) {                                                                           \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (i >= n) return;                                                       \
    float x = to_f(src[i]);                                                   \
    dst[i] = FROM_F(x > 20.0f ? x : log1pf(exp2f(x * LOG2E)));           \
}

DEFINE_SOFTPLUS_COPY(f32,  float,         from_f_f32)
DEFINE_SOFTPLUS_COPY(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SOFTPLUS_COPY(f16,  __half,        from_f_f16)

// Typed vec_add_inplace — for input_proj / out_proj bias add in M3 pipeline.
// bias is always f32, activations in TY.
#define DEFINE_VEC_ADD_INPLACE(SUFFIX, TY, FROM_F)                            \
extern "C" __global__ void vec_add_inplace_##SUFFIX(                          \
    TY* a, const float* b, int n                                              \
) {                                                                           \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (i >= n) return;                                                       \
    a[i] = FROM_F(to_f(a[i]) + b[i]);                                         \
}

DEFINE_VEC_ADD_INPLACE(f32,  float,         from_f_f32)
DEFINE_VEC_ADD_INPLACE(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_VEC_ADD_INPLACE(f16,  __half,        from_f_f16)

// Typed concat_halves — mirrors f32 `concat_halves` for the mixed backward
// wiring where both `first_half` and `second_half` are typed gradient
// scratches (d_x_branch, d_gate) and the output `proj` feeds a typed dW
// GEMM via in_proj backward. Pure-load/store op, no arithmetic.
#define DEFINE_CONCAT_HALVES(SUFFIX, TY)                                      \
extern "C" __global__ void concat_halves_##SUFFIX(                            \
    TY* proj,                                                                 \
    const TY* first_half,                                                     \
    const TY* second_half,                                                    \
    int batch, int d_inner                                                    \
) {                                                                           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    int total = batch * d_inner;                                              \
    if (idx >= total) return;                                                 \
    int b = idx / d_inner;                                                    \
    int d = idx % d_inner;                                                    \
    int proj_off = b * 2 * d_inner;                                           \
    proj[proj_off + d] = first_half[idx];                                     \
    proj[proj_off + d_inner + d] = second_half[idx];                          \
}

DEFINE_CONCAT_HALVES(f32,  float)
DEFINE_CONCAT_HALVES(bf16, __nv_bfloat16)
DEFINE_CONCAT_HALVES(f16,  __half)

// Typed scatter_add_cols — for mixed backward: d_xdbl (typed) accumulates
// d_delta_raw / d_B / d_C slices (typed). Accumulate in f32 then downcast
// to avoid compounded bf16 round-off during successive scatters.
#define DEFINE_SCATTER_ADD_COLS(SUFFIX, TY, FROM_F)                           \
extern "C" __global__ void scatter_add_cols_##SUFFIX(                         \
    TY* dst, const TY* src,                                                   \
    int batch, int dst_stride, int src_dim, int offset                        \
) {                                                                           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    int total = batch * src_dim;                                              \
    if (idx >= total) return;                                                 \
    int b = idx / src_dim;                                                    \
    int d = idx % src_dim;                                                    \
    int di = b * dst_stride + offset + d;                                     \
    dst[di] = FROM_F(to_f(dst[di]) + to_f(src[idx]));                         \
}

DEFINE_SCATTER_ADD_COLS(f32,  float,         from_f_f32)
DEFINE_SCATTER_ADD_COLS(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SCATTER_ADD_COLS(f16,  __half,        from_f_f16)

// Typed bias reduction: `d_bias[i] += sum over (b, t) of dy[b, t, i]` where
// `dy` is typed and `d_bias` is f32 master grad. Used by mixed dt_proj
// backward to accumulate the bias gradient. One block per bias index i; one
// warp per block handles the B*T reduction. Each block writes to a distinct
// index `i`, so the final `d_bias[i] += sdata[0]` write has no race —
// deterministic without atomics.
#define DEFINE_REDUCE_BIAS(SUFFIX, TY)                                        \
extern "C" __global__ __launch_bounds__(256, 4)                               \
void reduce_bias_##SUFFIX(                                                    \
    float* __restrict__ d_bias, const TY* __restrict__ dy, int bt, int dim    \
) {                                                                           \
    int i = blockIdx.x;                                                       \
    if (i >= dim) return;                                                     \
    float sum = 0.0f;                                                         \
    for (int r = threadIdx.x; r < bt; r += blockDim.x) {                      \
        sum += to_f(dy[r * dim + i]);                                         \
    }                                                                         \
    /* Warp + block reduction via shared memory. */                           \
    extern __shared__ float sdata[];                                          \
    sdata[threadIdx.x] = sum;                                                 \
    __syncthreads();                                                          \
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {                       \
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];    \
        __syncthreads();                                                      \
    }                                                                         \
    /* No race: each block handles a distinct bias index i. */                \
    if (threadIdx.x == 0) d_bias[i] = d_bias[i] + sdata[0];                   \
}

DEFINE_REDUCE_BIAS(f32,  float)
DEFINE_REDUCE_BIAS(bf16, __nv_bfloat16)
DEFINE_REDUCE_BIAS(f16,  __half)
