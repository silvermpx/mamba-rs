// The Mamba-3 SISO decode steps as they were released: the warp tier's f32
// and typed steps and the block tier's step, each thread holding its
// element's whole state in an array of the compiled capacity that the
// compiler kept in local memory (the state loops index it at run time).
// The steps that take the state 16 at a time in registers are held to them
// bit for bit. The typed prelude precedes this fixture, and the test
// compiles it with the largest capacity.
#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

#define DEFINE_M3_STEP_FWD_F32(NAME, IDX)                                      \
extern "C" __global__ void NAME(                                               \
    /* In/Out: persistent state (mutated) */                                   \
    float* ssm_state,   /* [B * nh * hd * ds] */                               \
    float* k_state,     /* [B * nh * ds] -- previous K (post bias+RoPE) */     \
    float* v_state,     /* [B * nh * hd] -- previous x */                      \
    /* Output */                                                               \
    float* y,           /* [B * d_inner] */                                    \
    /* Inputs (pre-computed by shared ops) */                                  \
    const float* x,     /* [B * d_inner]   (d_inner = nh * hd) */              \
    const float* k_cur, /* [B * nh * ds]   (B after bias+RoPE) */              \
    const float* q_cur, /* [B * nh * ds]   (C after bias+RoPE) */              \
    /* The recurrence coefficients are computed here from the step's */        \
    /* dt, a_val and trap, spelled as the standalone coefficient kernel */     \
    /* spelled them; an f32 stored and reloaded carries the same bits. */      \
    const float* dt,    /* [B * nh] */                                         \
    const float* a_val, /* [B * nh] */                                         \
    const float* trap,  /* [B * nh] */                                         \
    const float* D,     /* [nh] */                                             \
    int batch, int nh, int hd, int ds,                                         \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    IDX b = blockIdx.x;                                                        \
    int h = y_first + blockIdx.y;                                              \
    int p = threadIdx.x;                                                       \
    if (b >= batch || h >= nh || p >= hd) return;                              \
                                                                               \
    int d_inner = nh * hd;                                                     \
                                                                               \
    /* Warp-reduce/broadcast mask: only `hd` lanes launched (block_dim=hd, hd≤32). */ \
    /* Hardcoded 0xFFFFFFFF = UB per CUDA Programming Guide §B.15.1. */        \
    unsigned warp_mask = (hd >= 32) ? 0xFFFFFFFFu : ((1u << hd) - 1u);         \
                                                                               \
    /* Load h_local into registers */                                          \
    float h_local[MAMBA_RS_STATE_CAP];                                         \
    if (ds > MAMBA_RS_STATE_CAP) return;                                       \
    IDX h_base = (b * nh * hd + h * hd + p) * ds;                              \
    for (int n = 0; n < ds; n++)                                               \
        h_local[n] = ssm_state[h_base + n];                                    \
                                                                               \
    /* O1: alpha, beta, gamma computed on p=0 and broadcast */                 \
    float alpha_h = 0.0f;                                                      \
    float beta_h = 0.0f;                                                       \
    float gamma_h = 0.0f;                                                      \
    if (p == 0) {                                                              \
        float dt_v = dt[b * nh + h];                                           \
        float a_v = a_val[b * nh + h];                                         \
        float t_v = trap[b * nh + h];                                          \
        alpha_h = exp2f((a_v * dt_v) * LOG2E);                                 \
        beta_h = alpha_h * dt_v * (1.0f - t_v);                                \
        gamma_h = t_v * dt_v;                                                  \
    }                                                                          \
    alpha_h = __shfl_sync(warp_mask, alpha_h, 0, hd);                          \
    beta_h = __shfl_sync(warp_mask, beta_h, 0, hd);                            \
    gamma_h = __shfl_sync(warp_mask, gamma_h, 0, hd);                          \
                                                                               \
    /* O1: D[h] broadcast from p=0 */                                          \
    float d_skip = 0.0f;                                                       \
    if (p == 0) d_skip = D[h];                                                 \
    d_skip = __shfl_sync(warp_mask, d_skip, 0, hd);                            \
                                                                               \
    /* Per-thread unique values */                                             \
    float x_val = x[b * d_inner + h * hd + p];                                 \
    float v_prev = v_state[b * nh * hd + h * hd + p];                          \
                                                                               \
    float y_val = d_skip * x_val;                                              \
                                                                               \
    for (int n = 0; n < ds; n++) {                                             \
        /* O1: k_cur[n], k_prev[n], q_cur[n] broadcast from p=0 */             \
        float kc_n = 0.0f, kp_n = 0.0f, qc_n = 0.0f;                           \
        if (p == 0) {                                                          \
            kc_n = k_cur[b * nh * ds + h * ds + n];                            \
            kp_n = k_state[b * nh * ds + h * ds + n];                          \
            qc_n = q_cur[b * nh * ds + h * ds + n];                            \
        }                                                                      \
        kc_n = __shfl_sync(warp_mask, kc_n, 0, hd);                            \
        kp_n = __shfl_sync(warp_mask, kp_n, 0, hd);                            \
        qc_n = __shfl_sync(warp_mask, qc_n, 0, hd);                            \
                                                                               \
        /* Trapezoidal recurrence */                                           \
        h_local[n] = alpha_h * h_local[n] + beta_h * v_prev * kp_n + gamma_h * x_val * kc_n; \
        y_val += h_local[n] * qc_n;                                            \
    }                                                                          \
                                                                               \
    /* Write back SSM state */                                                 \
    for (int n = 0; n < ds; n++)                                               \
        ssm_state[h_base + n] = h_local[n];                                    \
                                                                               \
    /* Write output */                                                         \
    y[b * d_inner + h * hd + p] = y_val;                                       \
                                                                               \
    /* Update k_state: only p=0 writes (k_cur is shared across all p in this head) */ \
    if (p == 0) {                                                              \
        for (int n = 0; n < ds; n++)                                           \
            k_state[b * nh * ds + h * ds + n] = k_cur[b * nh * ds + h * ds + n]; \
    }                                                                          \
                                                                               \
    /* Update v_state: each p writes its own x */                              \
    v_state[b * nh * hd + h * hd + p] = x_val;                                 \
}

DEFINE_M3_STEP_FWD_F32(m3_reference_step, int)
DEFINE_M3_STEP_FWD_F32(m3_reference_step_i64, long long)

#define DEFINE_M3_STEP_FWD(NAME, IDX, TY, FROM_F)                           \
extern "C" __global__ void NAME(                                             \
    float* ssm_state,                                                        \
    float* k_state,                                                          \
    float* v_state,                                                          \
    TY* y,                                                                   \
    const TY* x,                                                             \
    const TY* k_cur,                                                         \
    const TY* q_cur,                                                         \
    const float* dt, const float* a_val, const float* trap,                  \
    const float* D,                                                          \
    int batch, int nh, int hd, int ds,                                       \
    int y_first /* this run's first y block */                               \
) {                                                                          \
    IDX b = blockIdx.x;                                                      \
    int h = y_first + blockIdx.y;                                            \
    int p = threadIdx.x;                                                     \
    if (b >= batch || h >= nh || p >= hd) return;                            \
    int d_inner = nh * hd;                                                   \
    unsigned warp_mask = (hd >= 32) ? 0xFFFFFFFFu : ((1u << hd) - 1u);       \
    float h_local[MAMBA_RS_STATE_CAP];                                                       \
    if (ds > MAMBA_RS_STATE_CAP) return;                                                     \
    IDX h_base = (b * nh * hd + h * hd + p) * ds;                            \
    for (int n = 0; n < ds; n++) h_local[n] = ssm_state[h_base + n];          \
    float alpha_h = 0.0f;                                                    \
    float beta_h = 0.0f;                                                     \
    float gamma_h = 0.0f;                                                    \
    if (p == 0) {                                                            \
        float dt_v = dt[b * nh + h];                                         \
        float a_v = a_val[b * nh + h];                                       \
        float t_v = trap[b * nh + h];                                        \
        alpha_h = exp2f((a_v * dt_v) * LOG2E);                               \
        beta_h = alpha_h * dt_v * (1.0f - t_v);                              \
        gamma_h = t_v * dt_v;                                                \
    }                                                                        \
    alpha_h = __shfl_sync(warp_mask, alpha_h, 0, hd);                        \
    beta_h = __shfl_sync(warp_mask, beta_h, 0, hd);                          \
    gamma_h = __shfl_sync(warp_mask, gamma_h, 0, hd);                        \
    float d_skip = 0.0f;                                                     \
    if (p == 0) d_skip = D[h];                                               \
    d_skip = __shfl_sync(warp_mask, d_skip, 0, hd);                         \
    float x_val = to_f(x[b * d_inner + h * hd + p]);                         \
    float v_prev = v_state[b * nh * hd + h * hd + p];                        \
    float y_val = d_skip * x_val;                                            \
    for (int n = 0; n < ds; n++) {                                            \
        float kc_n = 0.0f, kp_n = 0.0f, qc_n = 0.0f;                         \
        if (p == 0) {                                                        \
            kc_n = to_f(k_cur[b * nh * ds + h * ds + n]);                    \
            kp_n = k_state[b * nh * ds + h * ds + n];                        \
            qc_n = to_f(q_cur[b * nh * ds + h * ds + n]);                    \
        }                                                                    \
        kc_n = __shfl_sync(warp_mask, kc_n, 0, hd);                         \
        kp_n = __shfl_sync(warp_mask, kp_n, 0, hd);                         \
        qc_n = __shfl_sync(warp_mask, qc_n, 0, hd);                         \
        h_local[n] = alpha_h * h_local[n] + beta_h * v_prev * kp_n +         \
                     gamma_h * x_val * kc_n;                                 \
        y_val += h_local[n] * qc_n;                                          \
    }                                                                        \
    for (int n = 0; n < ds; n++) ssm_state[h_base + n] = h_local[n];          \
    y[b * d_inner + h * hd + p] = FROM_F(y_val);                             \
    if (p == 0) {                                                            \
        for (int n = 0; n < ds; n++)                                          \
            k_state[b * nh * ds + h * ds + n] =                              \
                to_f(k_cur[b * nh * ds + h * ds + n]);                       \
    }                                                                        \
    v_state[b * nh * hd + h * hd + p] = x_val;                               \
}

DEFINE_M3_STEP_FWD(m3_reference_step_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_M3_STEP_FWD(m3_reference_step_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_M3_STEP_FWD(m3_reference_step_f16, int, __half, from_f_f16)
DEFINE_M3_STEP_FWD(m3_reference_step_f16_i64, long long, __half, from_f_f16)

#define DEFINE_M3_STEP_FWD_BLOCK(NAME, IDX, TY, FROM_F)                       \
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
    int batch, int nh, int hd, int ds,                                        \
    int y_first /* this run's first y block */                                \
) {                                                                           \
    extern __shared__ float m3sb_sm[];                                        \
    float* coef = m3sb_sm;                                                    \
    float* kc_sm = coef + 4;                                                  \
    float* kp_sm = kc_sm + ds;                                                \
    float* qc_sm = kp_sm + ds;                                                \
    IDX b = blockIdx.x;                                                       \
    int h = y_first + blockIdx.y;                                             \
    int tid = threadIdx.x;                                                    \
    /* Uniform across the block, so every thread reaches the barrier. */      \
    if (b >= batch || h >= nh || ds > MAMBA_RS_STATE_CAP) return;             \
    int d_inner = nh * hd;                                                    \
    IDX row = b * nh * ds + h * ds;                                           \
    if (tid == 0) {                                                           \
        float dt_v = dt[b * nh + h];                                          \
        float a_v = a_val[b * nh + h];                                        \
        float t_v = trap[b * nh + h];                                         \
        float alpha_h = exp2f((a_v * dt_v) * LOG2E);                          \
        coef[0] = alpha_h;                                                    \
        coef[1] = alpha_h * dt_v * (1.0f - t_v);                              \
        coef[2] = t_v * dt_v;                                                 \
        coef[3] = D[h];                                                       \
    }                                                                         \
    for (int n = tid; n < ds; n += blockDim.x) {                              \
        kc_sm[n] = to_f(k_cur[row + n]);                                      \
        kp_sm[n] = k_state[row + n];                                          \
        qc_sm[n] = to_f(q_cur[row + n]);                                      \
    }                                                                         \
    __syncthreads();                                                          \
    for (int n = tid; n < ds; n += blockDim.x) k_state[row + n] = kc_sm[n];   \
    float alpha_h = coef[0];                                                  \
    float beta_h = coef[1];                                                   \
    float gamma_h = coef[2];                                                  \
    float d_skip = coef[3];                                                   \
    for (int p = tid; p < hd; p += blockDim.x) {                              \
        float h_local[MAMBA_RS_STATE_CAP];                                    \
        IDX h_base = (b * nh * hd + h * hd + p) * ds;                         \
        for (int n = 0; n < ds; n++) h_local[n] = ssm_state[h_base + n];      \
        float x_val = to_f(x[b * d_inner + h * hd + p]);                      \
        float v_prev = v_state[b * nh * hd + h * hd + p];                     \
        float y_val = d_skip * x_val;                                         \
        float bv = beta_h * v_prev;                                           \
        float gx = gamma_h * x_val;                                           \
        for (int n = 0; n < ds; n++) {                                        \
            float kc_n = kc_sm[n];                                            \
            float kp_n = kp_sm[n];                                            \
            float qc_n = qc_sm[n];                                            \
            h_local[n] = __fmaf_rn(gx, kc_n,                                  \
                                   __fmaf_rn(h_local[n], alpha_h,             \
                                             __fmul_rn(bv, kp_n)));           \
            y_val = __fmaf_rn(h_local[n], qc_n, y_val);                       \
        }                                                                     \
        for (int n = 0; n < ds; n++) ssm_state[h_base + n] = h_local[n];      \
        y[b * d_inner + h * hd + p] = FROM_F(y_val);                          \
        v_state[b * nh * hd + h * hd + p] = x_val;                            \
    }                                                                         \
}

DEFINE_M3_STEP_FWD_BLOCK(m3_reference_step_block, int, float, from_f_f32)
DEFINE_M3_STEP_FWD_BLOCK(m3_reference_step_block_i64, long long, float, from_f_f32)
DEFINE_M3_STEP_FWD_BLOCK(m3_reference_step_block_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_M3_STEP_FWD_BLOCK(m3_reference_step_block_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_M3_STEP_FWD_BLOCK(m3_reference_step_block_f16, int, __half, from_f_f16)
DEFINE_M3_STEP_FWD_BLOCK(m3_reference_step_block_f16_i64, long long, __half, from_f_f16)
