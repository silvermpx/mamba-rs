// Frozen sequential recurrence; the typed prelude precedes this fixture.
#define DEFINE_M3_BURNIN_REFERENCE(SUFFIX, T_ACT, FROM_F)                          \
extern "C" __global__ void m3_burnin_reference_##SUFFIX(                           \
    float* ssm_state,                                                        \
    float* k_state,                                                          \
    float* v_state,                                                          \
    T_ACT* y_out,                                                            \
    float* h_saved,                                                          \
    float* k_prev_saved,                                                     \
    float* v_prev_saved,                                                     \
    const T_ACT* x_flat,                                                     \
    const T_ACT* k_flat,                                                     \
    const T_ACT* q_flat,                                                     \
    const float* alpha_flat,                                                 \
    const float* beta_flat,                                                  \
    const float* gamma_flat,                                                 \
    const float* D,                                                          \
    int batch, int T, int nh, int hd, int ds                                 \
) {                                                                          \
    int b = blockIdx.x;                                                      \
    int h = blockIdx.y;                                                      \
    int p = threadIdx.x;                                                     \
    if (b >= batch || h >= nh || p >= hd) return;                            \
    int d_inner = nh * hd;                                                   \
    int nhd_ds = d_inner * ds;                                               \
    unsigned warp_mask = (hd >= 32) ? 0xFFFFFFFFu : ((1u << hd) - 1u);       \
    float h_local[MAMBA_RS_STATE_CAP];                                                       \
    if (ds > MAMBA_RS_STATE_CAP) return;                                                     \
    int h_base = (b * nh * hd + h * hd + p) * ds;                            \
    for (int n = 0; n < ds; n++) h_local[n] = ssm_state[h_base + n];          \
    float d_skip = 0.0f;                                                     \
    if (p == 0) d_skip = D[h];                                               \
    d_skip = __shfl_sync(warp_mask, d_skip, 0, hd);                          \
    for (int n = 0; n < ds; n++) {                                            \
        int hs_idx = n * d_inner + h * hd + p;                               \
        h_saved[b * (T + 1) * nhd_ds + hs_idx] = h_local[n];                 \
    }                                                                        \
    for (int t = 0; t < T; t++) {                                            \
        if (p == 0) {                                                        \
            for (int n = 0; n < ds; n++)                                      \
                k_prev_saved[(b * T + t) * nh * ds + h * ds + n] =           \
                    k_state[b * nh * ds + h * ds + n];                       \
        }                                                                    \
        v_prev_saved[(b * T + t) * d_inner + h * hd + p] =                   \
            v_state[b * nh * hd + h * hd + p];                               \
        float alpha_h = 0.0f, beta_h = 0.0f, gamma_h = 0.0f;                 \
        if (p == 0) {                                                        \
            alpha_h = alpha_flat[(b * T + t) * nh + h];                      \
            beta_h  = beta_flat[(b * T + t) * nh + h];                       \
            gamma_h = gamma_flat[(b * T + t) * nh + h];                      \
        }                                                                    \
        alpha_h = __shfl_sync(warp_mask, alpha_h, 0, hd);                   \
        beta_h  = __shfl_sync(warp_mask, beta_h,  0, hd);                    \
        gamma_h = __shfl_sync(warp_mask, gamma_h, 0, hd);                   \
        int x_idx = (b * T + t) * d_inner + h * hd + p;                      \
        float x_val = to_f(x_flat[x_idx]);                                   \
        float v_prev = v_state[b * nh * hd + h * hd + p];                    \
        float y_val = d_skip * x_val;                                        \
        for (int n = 0; n < ds; n++) {                                        \
            float kc_n = 0.0f, kp_n = 0.0f, qc_n = 0.0f;                     \
            if (p == 0) {                                                    \
                kc_n = to_f(k_flat[(b * T + t) * nh * ds + h * ds + n]);     \
                kp_n = k_state[b * nh * ds + h * ds + n];                    \
                qc_n = to_f(q_flat[(b * T + t) * nh * ds + h * ds + n]);     \
            }                                                                \
            kc_n = __shfl_sync(warp_mask, kc_n, 0, hd);                     \
            kp_n = __shfl_sync(warp_mask, kp_n, 0, hd);                     \
            qc_n = __shfl_sync(warp_mask, qc_n, 0, hd);                     \
            h_local[n] = alpha_h * h_local[n] + beta_h * v_prev * kp_n       \
                       + gamma_h * x_val * kc_n;                             \
            y_val += h_local[n] * qc_n;                                      \
        }                                                                    \
        y_out[x_idx] = FROM_F(y_val);                                        \
        for (int n = 0; n < ds; n++) {                                        \
            int hs_idx = (t + 1) * nhd_ds + n * d_inner + h * hd + p;        \
            h_saved[b * (T + 1) * nhd_ds + hs_idx] = h_local[n];             \
        }                                                                    \
        if (p == 0) {                                                        \
            for (int n = 0; n < ds; n++)                                      \
                k_state[b * nh * ds + h * ds + n] =                          \
                    to_f(k_flat[(b * T + t) * nh * ds + h * ds + n]);        \
        }                                                                    \
        v_state[b * nh * hd + h * hd + p] = x_val;                           \
    }                                                                        \
    for (int n = 0; n < ds; n++) ssm_state[h_base + n] = h_local[n];          \
}

DEFINE_M3_BURNIN_REFERENCE(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M3_BURNIN_REFERENCE(f16,  __half,        from_f_f16)
