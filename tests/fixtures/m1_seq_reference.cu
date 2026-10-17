// The Mamba-1 sequential kernels as they were released: the burn-in
// forward with and without its saves, and the local backward over f32 and
// typed rows. Each took a channel a thread and held the channel's whole
// state in register arrays of the compiled capacity, which spilled to
// local memory past about 64 states in the backward and 128 in the
// forward. The kernels that give a channel a warp segment of slices are
// held to them bit for bit. The typed prelude precedes this fixture, and
// the test compiles it with the largest capacity.
#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

__device__ __forceinline__ float ssm_update_decay_fused(float da, float h,
                                                        float du, float B) {
    return __fmaf_rn(da, h, __fmul_rn(du, B));
}

#define DEFINE_SSM_BURNIN_NOSAVE(NAME, IDX, TY, FROM_F)                    \
extern "C" __global__ void NAME(                                           \
    float* h,                                                              \
    TY* y_out,                                                             \
    const TY* delta, const TY* u,                                          \
    const TY* B, const TY* C,                                              \
    const float* a_neg, const float* D,                                    \
    const TY* proj_gate,                                                   \
    int gate_stride,                                                       \
    int batch, int T_len, int d_inner, int d_state                         \
) {                                                                        \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                  \
    IDX total = (IDX)batch * d_inner;                                      \
    if (idx >= total) return;                                              \
    IDX b = idx / d_inner;                                                 \
    int d = (int)(idx % d_inner);                                          \
    IDX h_base = (b * d_inner + d) * d_state;                              \
    float h_local[MAMBA_RS_STATE_CAP];                                                     \
    float a_local[MAMBA_RS_STATE_CAP];                                                     \
    if (d_state > MAMBA_RS_STATE_CAP) return;                                              \
    _Pragma("unroll")                                                      \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {        \
        h_local[n] = h[h_base + n];                                        \
        a_local[n] = a_neg[(IDX)d * d_state + n];                          \
    }                                                                      \
    for (int t = 0; t < T_len; t++) {                                      \
        IDX bt_di = (b * T_len + t) * d_inner + d;                         \
        IDX bt_ds = (b * T_len + t) * d_state;                             \
        float delta_d = to_f(delta[bt_di]);                                \
        float u_d = to_f(u[bt_di]);                                        \
        float delta_u_d = delta_d * u_d;                                   \
        float y_d = D[d] * u_d;                                            \
        _Pragma("unroll")                                                  \
        for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {    \
            float da = exp2f(delta_d * a_local[n] * LOG2E);                \
            h_local[n] = ssm_update_decay_fused(da, h_local[n], delta_u_d, \
                                                to_f(B[bt_ds + n]));       \
            y_d = __fmaf_rn(h_local[n], to_f(C[bt_ds + n]), y_d);          \
        }                                                                  \
        TY ty = FROM_F(y_d);                                               \
        if (gate_stride > 0) {                                             \
            /* Round-trip emulation of the replaced typed chain (see    \
             * the parallel nosave twin). */                               \
            float g = to_f(proj_gate[(b * T_len + t) * gate_stride         \
                                     + d_inner + d]);                      \
            TY tg = FROM_F(g / (1.0f + exp2f(-g * 1.4426950408889634f)));  \
            ty = FROM_F(to_f(ty) * to_f(tg));                              \
        }                                                                  \
        y_out[bt_di] = ty;                                                 \
    }                                                                      \
    _Pragma("unroll")                                                      \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else          \
        h[h_base + n] = h_local[n];                                        \
}

DEFINE_SSM_BURNIN_NOSAVE(m1_reference_burnin_nosave_f32, int, float, from_f_f32)
DEFINE_SSM_BURNIN_NOSAVE(m1_reference_burnin_nosave_f32_i64, long long, float, from_f_f32)
DEFINE_SSM_BURNIN_NOSAVE(m1_reference_burnin_nosave_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BURNIN_NOSAVE(m1_reference_burnin_nosave_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BURNIN_NOSAVE(m1_reference_burnin_nosave_f16, int, __half, from_f_f16)
DEFINE_SSM_BURNIN_NOSAVE(m1_reference_burnin_nosave_f16_i64, long long, __half, from_f_f16)

#define DEFINE_SSM_BURNIN(NAME, IDX, TY, FROM_F)                            \
extern "C" __global__ void NAME(                                            \
    float* h, TY* y_out, float* h_saved,                                    \
    const TY* delta_raw, TY* delta_saved, const TY* u,                      \
    const TY* B, const TY* C,                                               \
    const float* a_neg, const float* D,                                     \
    int batch, int T_len, int d_inner, int d_state                          \
) {                                                                         \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                   \
    IDX total = (IDX)batch * d_inner;                                       \
    if (idx >= total) return;                                               \
    IDX b = idx / d_inner;                                                  \
    int d = (int)(idx % d_inner);                                           \
    IDX h_base = (b * d_inner + d) * d_state;                               \
    float h_local[MAMBA_RS_STATE_CAP];                                                      \
    float a_local[MAMBA_RS_STATE_CAP];                                                      \
    if (d_state > MAMBA_RS_STATE_CAP) return;                                               \
    _Pragma("unroll")                                                       \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {         \
        h_local[n] = h[h_base + n];                                         \
        a_local[n] = a_neg[(IDX)d * d_state + n];                           \
    }                                                                       \
    _Pragma("unroll")                                                       \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {         \
        IDX hs_idx =                                                        \
            ((b * (T_len + 1) + 0) * d_inner + d) * d_state + n;            \
        h_saved[hs_idx] = h_local[n];                                       \
    }                                                                       \
    for (int t = 0; t < T_len; t++) {                                       \
        IDX bt_di = (b * T_len + t) * d_inner + d;                          \
        IDX bt_ds = (b * T_len + t) * d_state;                              \
        float raw = to_f(delta_raw[bt_di]);                                 \
        float sp = (raw > 20.0f)                                            \
            ? raw : log1pf(exp2f(raw * 1.4426950408889634f));               \
        TY spt = FROM_F(sp);                                                \
        delta_saved[bt_di] = spt;                                           \
        float delta_d = to_f(spt);                                          \
        float u_d = to_f(u[bt_di]);                                         \
        float delta_u_d = delta_d * u_d;                                    \
        float y_d = D[d] * u_d;                                             \
        _Pragma("unroll")                                                   \
        for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {     \
            float da = exp2f(delta_d * a_local[n] * LOG2E);                 \
            h_local[n] = ssm_update_decay_fused(da, h_local[n], delta_u_d,  \
                                                to_f(B[bt_ds + n]));        \
            y_d = __fmaf_rn(h_local[n], to_f(C[bt_ds + n]), y_d);           \
        }                                                                   \
        y_out[bt_di] = FROM_F(y_d);                                         \
        _Pragma("unroll")                                                   \
        for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {     \
            IDX hs_idx =                                                    \
                ((b * (T_len + 1) + (t + 1)) * d_inner + d) * d_state + n;  \
            h_saved[hs_idx] = h_local[n];                                   \
        }                                                                   \
    }                                                                       \
    _Pragma("unroll")                                                       \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else           \
        h[h_base + n] = h_local[n];                                         \
}

DEFINE_SSM_BURNIN(m1_reference_burnin_f32, int, float, from_f_f32)
DEFINE_SSM_BURNIN(m1_reference_burnin_f32_i64, long long, float, from_f_f32)
DEFINE_SSM_BURNIN(m1_reference_burnin_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BURNIN(m1_reference_burnin_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BURNIN(m1_reference_burnin_f16, int, __half, from_f_f16)
DEFINE_SSM_BURNIN(m1_reference_burnin_f16_i64, long long, __half, from_f_f16)

#define DEFINE_SSM_BACKWARD_LOCAL(NAME, IDX)                                   \
extern "C" __global__ void NAME(                                               \
    /* Inputs (from forward, saved activations) */                             \
    const float* h_saved, /* [batch * (T+1) * d_inner * d_state] saved h BEFORE each step */ \
    const float* delta_saved, /* [batch * T * d_inner] */                      \
    const float* u_saved, /* [batch * T * d_inner] */                          \
    const float* B_saved, /* [batch * T * d_state] */                          \
    const float* C_saved, /* [batch * T * d_state] */                          \
    const float* a_neg, /* [d_inner * d_state] */                              \
    const float* D, /* [d_inner] skip connection weight */                     \
    /* Incoming gradient */                                                    \
    const float* dy, /* [batch * T * d_inner] */                               \
    /* Output gradients (per-sample, need reduction across batch) */           \
    float* d_delta, /* [batch * T * d_inner] per-sample */                     \
    float* d_u, /* [batch * T * d_inner] per-sample (includes skip: += dy*D[d]) */ \
    float* d_B_local, /* [batch * T * d_inner * d_state] per-thread */         \
    float* d_C_local, /* [batch * T * d_inner * d_state] per-thread */         \
    float* d_D_local, /* [batch * d_inner] per-sample (sum across T) */        \
    float* d_a_log_local, /* [batch * d_inner * d_state] per-sample (sum across T) */ \
    /* Dimensions */                                                           \
    int batch, int T, int d_inner, int d_state                                 \
) {                                                                            \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    IDX total = (IDX)batch * d_inner;                                          \
    if (idx >= total) return;                                                  \
                                                                               \
    IDX b = idx / d_inner;                                                     \
    int d = (int)(idx % d_inner);                                              \
                                                                               \
    /* Per-thread accumulators for D and a_log gradients */                    \
    float local_d_D = 0.0f;                                                    \
                                                                               \
    /* Opt A: cache a_neg in registers */                                      \
    float a_local[MAMBA_RS_STATE_CAP];                                         \
    if (d_state > MAMBA_RS_STATE_CAP) return;                                  \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else \
        a_local[n] = a_neg[(IDX)d * d_state + n];                              \
                                                                               \
    /* d_h carries gradient backward through time */                           \
    float d_h[MAMBA_RS_STATE_CAP];                                             \
    /* Register accumulator for d_a_log: the old global `+=` into */           \
    /* d_a_log_local was ~T*d_state dependent global RMWs per thread in */     \
    /* the hottest loop; same adds in the same (t,n) order, written once. */   \
    float d_a_acc[MAMBA_RS_STATE_CAP];                                         \
    /* Register carry for the BPTT state: in the reverse-T walk, */            \
    /* h_curr(t) == h_prev(t+1) — carrying it halves h_saved reads. */         \
    float h_carry[MAMBA_RS_STATE_CAP];                                         \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else { \
        d_h[n] = 0.0f;                                                         \
        d_a_acc[n] = 0.0f;                                                     \
        h_carry[n] = h_saved[((b * (T + 1) + T) * d_inner + d) * d_state + n]; \
    }                                                                          \
                                                                               \
    /* Backward through time (reverse T) */                                    \
    for (int t = T - 1; t >= 0; t--) {                                         \
        IDX bt_di = (b * T + t) * d_inner + d;                                 \
        IDX bt_ds = (b * T + t) * d_state;                                     \
                                                                               \
        float dy_d = dy[bt_di];                                                \
        float delta_d = delta_saved[bt_di];                                    \
        float u_d = u_saved[bt_di];                                            \
                                                                               \
        /* d_D += dy * u (skip connection gradient) */                         \
        local_d_D += dy_d * u_d;                                               \
                                                                               \
        /* d_u from skip connection: dy * D[d] */                              \
        float d_u_val = dy_d * D[d];                                           \
                                                                               \
        float d_delta_val = 0.0f;                                              \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else { \
            /* h_curr = state AFTER step t (carried register: at t=T-1 it */   \
            /* was seeded from h_saved[T]; afterwards it is last round's */    \
            /* h_prev — identical value, one global load saved). */            \
            float h_curr = h_carry[n];                                         \
                                                                               \
            /* Opt B: exp2f instead of expf */                                 \
            float da = exp2f(delta_d * a_local[n] * LOG2E);                    \
                                                                               \
            /* Gradient from output: d_h += dy * C */                          \
            d_h[n] += dy_d * C_saved[bt_ds + n];                               \
                                                                               \
            /* d_C += dy * h_curr (per-thread: indexed by b,t,d,n) */          \
            IDX btdn = ((b * T + t) * d_inner + d) * d_state + n;              \
            d_C_local[btdn] = dy_d * h_curr;                                   \
                                                                               \
            /* h_prev: state BEFORE step t */                                  \
            IDX h_prev_idx = ((b * (T + 1) + t) * d_inner + d) * d_state + n;  \
            float h_prev = h_saved[h_prev_idx];                                \
                                                                               \
            /* d_delta += d_h * (a_dn * da * h_prev + u * B) */                \
            d_delta_val += d_h[n] * (a_local[n] * da * h_prev + u_d * B_saved[bt_ds + n]); \
                                                                               \
            /* d_u += d_h * delta * B */                                       \
            d_u_val += d_h[n] * delta_d * B_saved[bt_ds + n];                  \
                                                                               \
            /* d_B += d_h * delta * u (per-thread: indexed by b,t,d,n) */      \
            d_B_local[btdn] = d_h[n] * delta_d * u_d;                          \
                                                                               \
            /* d_a_log += d_h * da * delta * a_dn * h_prev (register acc) */   \
            d_a_acc[n] += d_h[n] * da * delta_d * a_local[n] * h_prev;         \
                                                                               \
            /* Propagate d_h backward through time: d_h_prev = da * d_h */     \
            d_h[n] = da * d_h[n];                                              \
            h_carry[n] = h_prev;                                               \
        }                                                                      \
                                                                               \
        d_delta[bt_di] = d_delta_val;                                          \
        d_u[bt_di] = d_u_val;                                                  \
    }                                                                          \
                                                                               \
    /* One store per element replaces T global RMWs; full-domain write, */     \
    /* so the per-layer zeroing of d_a_log_local is gone with it. */           \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else \
        d_a_log_local[(b * d_inner + d) * d_state + n] = d_a_acc[n];           \
                                                                               \
    d_D_local[b * d_inner + d] = local_d_D;                                    \
}

DEFINE_SSM_BACKWARD_LOCAL(m1_reference_backward_local, int)
DEFINE_SSM_BACKWARD_LOCAL(m1_reference_backward_local_i64, long long)

#define DEFINE_SSM_BACKWARD_LOCAL_BWD(NAME, IDX, TY, FROM_F)                    \
extern "C" __global__ void NAME(                                                \
    const float* h_saved,                                                       \
    const TY*    delta_saved,                                                   \
    const TY*    u_saved,                                                       \
    const TY*    B_saved,                                                       \
    const TY*    C_saved,                                                       \
    const float* a_neg,                                                         \
    const float* D,                                                             \
    const TY*    dy,                                                            \
    TY*          d_delta,                                                       \
    TY*          d_u,                                                           \
    TY*          d_B_local,                                                     \
    TY*          d_C_local,                                                     \
    float*       d_D_local,                                                     \
    float*       d_a_log_local,                                                 \
    int batch, int T, int d_inner, int d_state                                  \
) {                                                                             \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                       \
    IDX total = (IDX)batch * d_inner;                                           \
    if (idx >= total) return;                                                   \
    IDX b = idx / d_inner;                                                      \
    int d = (int)(idx % d_inner);                                               \
    if (d_state > MAMBA_RS_STATE_CAP) return;                                                   \
                                                                                \
    float local_d_D = 0.0f;                                                     \
    float a_local[MAMBA_RS_STATE_CAP];                                                          \
    float d_h[MAMBA_RS_STATE_CAP];                                                              \
    float d_a_acc[MAMBA_RS_STATE_CAP];                                                          \
    float h_carry[MAMBA_RS_STATE_CAP];                                                          \
    _Pragma("unroll")                                                           \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {             \
        a_local[n] = a_neg[(IDX)d * d_state + n];                               \
        d_h[n] = 0.0f;                                                          \
        d_a_acc[n] = 0.0f;                                                      \
        h_carry[n] = h_saved[((b * (T + 1) + T) * d_inner + d) * d_state + n];  \
    }                                                                           \
                                                                                \
    for (int t = T - 1; t >= 0; t--) {                                          \
        IDX bt_di = (b * T + t) * d_inner + d;                                  \
        IDX bt_ds = (b * T + t) * d_state;                                      \
        float dy_d    = to_f(dy[bt_di]);                                        \
        float delta_d = to_f(delta_saved[bt_di]);                               \
        float u_d     = to_f(u_saved[bt_di]);                                   \
                                                                                \
        local_d_D += dy_d * u_d;                                                \
        float d_u_val = dy_d * D[d];                                            \
        float d_delta_val = 0.0f;                                               \
                                                                                \
        _Pragma("unroll")                                                       \
        for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {         \
            IDX h_prev_idx =                                                    \
                ((b * (T + 1) + t) * d_inner + d) * d_state + n;                \
            float h_curr = h_carry[n];                                          \
            float h_prev = h_saved[h_prev_idx];                                 \
            float B_n    = to_f(B_saved[bt_ds + n]);                            \
            float C_n    = to_f(C_saved[bt_ds + n]);                            \
            float da     = exp2f(delta_d * a_local[n] * LOG2E);                 \
                                                                                \
            d_h[n] += dy_d * C_n;                                               \
            IDX btdn = ((b * T + t) * d_inner + d) * d_state + n;               \
            d_C_local[btdn] = FROM_F(dy_d * h_curr);                            \
                                                                                \
            d_delta_val += d_h[n] * (a_local[n] * da * h_prev + u_d * B_n);     \
            d_u_val     += d_h[n] * delta_d * B_n;                              \
            d_B_local[btdn] = FROM_F(d_h[n] * delta_d * u_d);                   \
                                                                                \
            d_a_acc[n] += d_h[n] * da * delta_d * a_local[n] * h_prev;          \
                                                                                \
            d_h[n] = da * d_h[n];                                               \
            h_carry[n] = h_prev;                                                \
        }                                                                       \
                                                                                \
        d_delta[bt_di] = FROM_F(d_delta_val);                                   \
        d_u[bt_di]     = FROM_F(d_u_val);                                       \
    }                                                                           \
                                                                                \
    _Pragma("unroll")                                                           \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else               \
        d_a_log_local[(b * d_inner + d) * d_state + n] = d_a_acc[n];            \
                                                                                \
    d_D_local[b * d_inner + d] = local_d_D;                                     \
}

DEFINE_SSM_BACKWARD_LOCAL_BWD(m1_reference_backward_local_f32, int, float, from_f_f32)
DEFINE_SSM_BACKWARD_LOCAL_BWD(m1_reference_backward_local_f32_i64, long long, float, from_f_f32)
DEFINE_SSM_BACKWARD_LOCAL_BWD(m1_reference_backward_local_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BACKWARD_LOCAL_BWD(m1_reference_backward_local_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BACKWARD_LOCAL_BWD(m1_reference_backward_local_f16, int, __half, from_f_f16)
DEFINE_SSM_BACKWARD_LOCAL_BWD(m1_reference_backward_local_f16_i64, long long, __half, from_f_f16)
