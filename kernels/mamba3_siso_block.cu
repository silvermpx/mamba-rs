// Block-tier twins of the Mamba-3 SISO head kernels.
//
// The warp-tier kernels in mamba3_siso.cu run one warp per (batch, head)
// and broadcast the head-shared values (the step coefficients, D and the
// current and previous K and Q rows) from lane 0 by warp shuffle, so a
// head must fit one warp and its width must be a power of two. These twins
// serve every other width: one CTA per (batch, head), one thread per head
// element, the head-shared values staged once in shared memory.
//
// Each thread runs the warp kernel's arithmetic on the same values in the
// same order, so at a width both tiers serve they give the same bits. The
// recurrence is spelled with explicit roundings because the compiler fuses
// the warp kernel's `alpha*h + beta*v*kp + gamma*x*kc` as
// fma(gx, kc, fma(h, alpha, bv*kp)) but would fuse the same source here
// the other way round, one ulp apart. The router still sends those widths
// to the warp tier; the block tier carries the official checkpoints'
// headdim 64 and any other width up to 512.
//
// Composed after mamba3_chunked.cu in the Mamba-3 module.

#include "_typed_prelude.cuh"

#ifndef MAMBA_RS_STATE_CAP
#define MAMBA_RS_STATE_CAP 64
#endif

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

// m3_step_fwd_block: one decode step of a head wider than a warp.
//
// Arguments as m3_step_fwd. Grid: (B, nh). Block: headdim rounded up to
// whole warps. Dynamic shared memory: (4 + 3 * ds) floats, the four step
// coefficients then the current K, previous K and current Q rows.
//
// The previous K is staged by every thread before the barrier, so the
// head's new K may overwrite k_state right after it: no thread reads
// k_state again.
#define DEFINE_M3_STEP_FWD_BLOCK(NAME, TY, FROM_F)                            \
extern "C" __global__ void NAME(                                              \
    float* ssm_state,                                                         \
    float* k_state,                                                           \
    float* v_state,                                                           \
    TY* y,                                                                    \
    const TY* x,                                                              \
    const TY* k_cur,                                                          \
    const TY* q_cur,                                                          \
    const float* dt, const float* a_val, const float* trap,                   \
    const float* D,                                                           \
    int batch, int nh, int hd, int ds                                         \
) {                                                                           \
    extern __shared__ float m3sb_sm[];                                        \
    float* coef = m3sb_sm;                                                    \
    float* kc_sm = coef + 4;                                                  \
    float* kp_sm = kc_sm + ds;                                                \
    float* qc_sm = kp_sm + ds;                                                \
    int b = blockIdx.x;                                                       \
    int h = blockIdx.y;                                                       \
    int p = threadIdx.x;                                                      \
    /* Uniform across the block, so every thread reaches the barrier. */      \
    if (b >= batch || h >= nh || ds > MAMBA_RS_STATE_CAP) return;             \
    int d_inner = nh * hd;                                                    \
    int row = b * nh * ds + h * ds;                                           \
    if (p == 0) {                                                             \
        float dt_v = dt[b * nh + h];                                          \
        float a_v = a_val[b * nh + h];                                        \
        float t_v = trap[b * nh + h];                                         \
        float alpha_h = exp2f((a_v * dt_v) * LOG2E);                          \
        coef[0] = alpha_h;                                                    \
        coef[1] = alpha_h * dt_v * (1.0f - t_v);                              \
        coef[2] = t_v * dt_v;                                                 \
        coef[3] = D[h];                                                       \
    }                                                                         \
    for (int n = p; n < ds; n += blockDim.x) {                                \
        kc_sm[n] = to_f(k_cur[row + n]);                                      \
        kp_sm[n] = k_state[row + n];                                          \
        qc_sm[n] = to_f(q_cur[row + n]);                                      \
    }                                                                         \
    __syncthreads();                                                          \
    for (int n = p; n < ds; n += blockDim.x) k_state[row + n] = kc_sm[n];     \
    if (p >= hd) return;                                                      \
    float alpha_h = coef[0];                                                  \
    float beta_h = coef[1];                                                   \
    float gamma_h = coef[2];                                                  \
    float d_skip = coef[3];                                                   \
    float h_local[MAMBA_RS_STATE_CAP];                                        \
    int h_base = (b * nh * hd + h * hd + p) * ds;                             \
    for (int n = 0; n < ds; n++) h_local[n] = ssm_state[h_base + n];          \
    float x_val = to_f(x[b * d_inner + h * hd + p]);                          \
    float v_prev = v_state[b * nh * hd + h * hd + p];                         \
    float y_val = d_skip * x_val;                                             \
    float bv = beta_h * v_prev;                                               \
    float gx = gamma_h * x_val;                                               \
    for (int n = 0; n < ds; n++) {                                            \
        float kc_n = kc_sm[n];                                                \
        float kp_n = kp_sm[n];                                                \
        float qc_n = qc_sm[n];                                                \
        h_local[n] = __fmaf_rn(gx, kc_n,                                      \
                               __fmaf_rn(h_local[n], alpha_h,                 \
                                         __fmul_rn(bv, kp_n)));               \
        y_val = __fmaf_rn(h_local[n], qc_n, y_val);                           \
    }                                                                         \
    for (int n = 0; n < ds; n++) ssm_state[h_base + n] = h_local[n];          \
    y[b * d_inner + h * hd + p] = FROM_F(y_val);                              \
    v_state[b * nh * hd + h * hd + p] = x_val;                                \
}

DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block, float, from_f_f32)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_f16, __half, from_f_f16)
