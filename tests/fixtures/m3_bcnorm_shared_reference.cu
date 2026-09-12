// Frozen shared-memory oracle; the typed prelude precedes this fixture.
#define DEFINE_BCNORM_LEGACY(SUFFIX, T_ACT, FROM_F)                            \
extern "C" __global__ void bcnorm_legacy_##SUFFIX(                              \
    T_ACT* __restrict__ B_normed,                                               \
    T_ACT* __restrict__ C_normed,                                               \
    float* __restrict__ B_rms,                                                  \
    float* __restrict__ C_rms,                                                  \
    const T_ACT* __restrict__ B_raw,                                            \
    const T_ACT* __restrict__ C_raw,                                            \
    const float* __restrict__ B_weight,                                         \
    const float* __restrict__ C_weight,                                         \
    int N, int ng, int ds,                                                      \
    float eps, /* config-driven eps */                                          \
    int src_stride /* row stride of B_raw/C_raw; ng*ds when dense, the    */    \
                   /* projection row width when reading proj in place     */    \
) {                                                                             \
    /* gridDim.y == 2: 0 -> B path, 1 -> C path */                              \
    int which = blockIdx.y;                                                     \
    int block_id = blockIdx.x;                                                  \
    if (block_id >= N * ng) return;                                             \
    int d = threadIdx.x;                                                        \
    if (d >= ds) return;                                                        \
    const T_ACT* raw = (which == 0) ? B_raw : C_raw;                            \
    T_ACT* normed = (which == 0) ? B_normed : C_normed;                         \
    float* rms_out = (which == 0) ? B_rms : C_rms;                              \
    const float* weight = (which == 0) ? B_weight : C_weight;                   \
    int base = block_id * ds;                                                   \
    long long src = (long long)(block_id / ng) * src_stride                     \
                    + (long long)(block_id % ng) * ds;                          \
    float val = to_f(raw[src + d]);                                             \
    extern __shared__ float sdata[];                                            \
    sdata[d] = val * val;                                                       \
    __syncthreads();                                                            \
    int stride = 1;                                                             \
    while (stride < ds) stride <<= 1;                                           \
    stride >>= 1;                                                               \
    for (; stride > 0; stride >>= 1) {                                          \
        if (d < stride && (d + stride) < ds) {                                  \
            sdata[d] += sdata[d + stride];                                      \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
    float rms = sqrtf(sdata[0] / (float)ds + eps);                          \
    if (!isfinite(rms) || rms < 1e-20f) rms = 1.0f;                             \
    if (d == 0) rms_out[block_id] = rms;                                        \
    __syncthreads();                                                            \
    float inv_rms = 1.0f / rms;                                                 \
    normed[base + d] = FROM_F(val * inv_rms * weight[d]);                       \
}

DEFINE_BCNORM_LEGACY(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_BCNORM_LEGACY(f16,  __half,        from_f_f16)
DEFINE_BCNORM_LEGACY(f32,  float,         from_f_f32)
