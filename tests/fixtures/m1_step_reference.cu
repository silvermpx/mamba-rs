// The Mamba-1 decode step as it was released: every state of a channel in
// two register arrays of the compiled capacity, which spilled to local
// memory past about 128 states. The step that goes a slice of states at a
// time is held to it bit for bit. The typed prelude precedes this fixture,
// and the test compiles it with the largest capacity.
#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

__device__ __forceinline__ float ssm_update_decay_fused(float da, float h,
                                                        float du, float B) {
    return __fmaf_rn(da, h, __fmul_rn(du, B));
}
__device__ __forceinline__ float ssm_update_input_fused(float da, float h,
                                                        float du, float B) {
    return __fmaf_rn(du, B, __fmul_rn(da, h));
}

#define DEFINE_SSM_STEP_FWD_FUSED(NAME, IDX, T, FROM_F, UPDATE)            \
extern "C" __global__ void NAME(                                           \
    float* h,                                                              \
    T* y,                                                                  \
    const T* delta_raw,                                                    \
    const T* u,                                                            \
    const T* xdbl,                                                         \
    const T* proj_gate,                                                    \
    int gate_stride,                                                       \
    const float* a_neg,                                                    \
    const float* D,                                                        \
    int batch, int d_inner, int d_state,                                   \
    int xdbl_stride, int b_offset, int c_offset                            \
) {                                                                        \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                  \
    IDX total = (IDX)batch * d_inner;                                      \
    if (idx >= total) return;                                              \
    IDX b = idx / d_inner;                                                 \
    int d = (int)(idx % d_inner);                                          \
    IDX h_base = (b * d_inner + d) * d_state;                              \
    IDX xdbl_base = b * xdbl_stride;                                       \
    float h_local[MAMBA_RS_STATE_CAP];                                     \
    float a_local[MAMBA_RS_STATE_CAP];                                     \
    if (d_state > MAMBA_RS_STATE_CAP) return;                              \
    _Pragma("unroll")                                                      \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {        \
        h_local[n] = h[h_base + n];                                        \
        a_local[n] = a_neg[(IDX)d * d_state + n];                          \
    }                                                                      \
    /* Softplus as the standalone kernel spelled it, then its store. */    \
    float raw = to_f(delta_raw[idx]);                                      \
    float sp = (raw > 20.0f) ? raw : log1pf(exp2f(raw * LOG2E));           \
    float delta_d = to_f(FROM_F(sp));                                      \
    float u_d = to_f(u[idx]);                                              \
    float delta_u_d = delta_d * u_d;                                       \
    float y_d = D[d] * u_d;                                                \
    _Pragma("unroll")                                                      \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {        \
        float da = exp2f(delta_d * a_local[n] * LOG2E);                    \
        float B_n = to_f(xdbl[xdbl_base + b_offset + n]);                  \
        float C_n = to_f(xdbl[xdbl_base + c_offset + n]);                  \
        h_local[n] = UPDATE(da, h_local[n], delta_u_d, B_n);               \
        y_d = __fmaf_rn(h_local[n], C_n, y_d);                             \
    }                                                                      \
    _Pragma("unroll")                                                      \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else          \
        h[h_base + n] = h_local[n];                                        \
    /* The gate's SiLU as the split kernel spelled it, then its store. */  \
    float g = to_f(proj_gate[b * gate_stride + d_inner + d]);              \
    float gp = to_f(FROM_F(g / (1.0f + exp2f(-g * LOG2E))));               \
    y[idx] = FROM_F(y_d * gp);                                             \
}

DEFINE_SSM_STEP_FWD_FUSED(m1_reference_step_f32, int, float, from_f_f32, ssm_update_decay_fused)
DEFINE_SSM_STEP_FWD_FUSED(m1_reference_step_f32_i64, long long, float, from_f_f32, ssm_update_decay_fused)
DEFINE_SSM_STEP_FWD_FUSED(m1_reference_step_bf16, int, __nv_bfloat16, from_f_bf16, ssm_update_input_fused)
DEFINE_SSM_STEP_FWD_FUSED(m1_reference_step_bf16_i64, long long, __nv_bfloat16, from_f_bf16, ssm_update_input_fused)
DEFINE_SSM_STEP_FWD_FUSED(m1_reference_step_f16, int, __half, from_f_f16, ssm_update_input_fused)
DEFINE_SSM_STEP_FWD_FUSED(m1_reference_step_f16_i64, long long, __half, from_f_f16, ssm_update_input_fused)
