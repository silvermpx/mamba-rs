// Mamba SSM recurrence CUDA kernels.
//
// Selective Scan: parallel across batch * d_inner, sequential across T.
// For T=1 (collection): single SSM step per (d,n) pair.
// For T>1 (burn-in/training): iterate T steps sequentially.
//
// Discretization: da = exp(delta * A), where A = -exp(a_log) (negative by convention).
// Recurrence: h[d,n] = da * h_prev[d,n] + delta * u * B[n]
// Output: y[d] = sum_n(h[d,n] * C[n]) + D[d] * u[d]
//
// Optimizations applied (from Tri Dao research + our profiling):
// - Opt A: h[d_state] + a_neg[d_state] cached in registers (not global memory)
// - Opt B: exp2f(x * LOG2E) instead of expf(x) (1 PTX instruction vs 2)
// - C2: delta_u_d hoisted from inner loop
//
// Source: CPU reference: train/forward.rs (phases F4d, B3)
// Paper: Gu & Dao 2023 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

// The state capacity the module is compiled for, injected at JIT time
// (-DMAMBA_RS_STATE_CAP=...) from the model config. The decode step and
// the sequential kernels hold slices of 16 states at a time in register
// arrays that index statically, so no state spills whatever d_state; the
// decode step refuses a state past the capacity, which the host checks
// before it launches.
#ifndef MAMBA_RS_STATE_CAP
#define MAMBA_RS_STATE_CAP 64
#endif


#include "_typed_prelude.cuh"

// The state update is one expression, da * h + delta_u * B, and the
// compiler decides which of the two products to fuse into the FMA; the
// choice changes the last bit of every state from the second step on,
// and it changed once between two spellings of the same kernel. Both
// forms are written out here with the rounding intrinsics, and every
// sequential kernel names the one its 0.6.9 build used: the decay product
// fused on the f32 lanes and in every burn-in, the input product fused on
// the typed decode steps. The output accumulates as one FMA per state.
__device__ __forceinline__ float ssm_update_decay_fused(float da, float h,
                                                        float du, float B) {
    return __fmaf_rn(da, h, __fmul_rn(du, B));
}
__device__ __forceinline__ float ssm_update_input_fused(float da, float h,
                                                        float du, float B) {
    return __fmaf_rn(du, B, __fmul_rn(da, h));
}

// The sequential kernels give a channel `lanes` threads of one warp, a
// power of two, each holding a slice of at most M1_SEQ_SLICE states in
// registers: lane k the states from k * ceil(d_state / lanes) on. Lane k
// takes step t at round t + k (the backward walks the steps from the last)
// and hands the step's output chain, with the step's inputs, to lane k + 1
// through a shuffle, and that lane continues the chain over its own states.
// The chain runs over the states in order, as one thread over the whole
// state ran it, so every output keeps that thread's bits, and no state
// leaves the registers whatever d_state.
#define M1_SEQ_SLICE 16

// ======================== FORWARD ========================

// The decode SSM step with its neighbours folded in: softplus on the
// dt_proj output, B and C read straight from xdbl, and the gate read from
// the in_proj output with the SiLU recomputed. One launch replaces the
// softplus, gather, step and gating kernels. Every folded value is
// spelled as the kernel it replaces spelled it, including the rounding
// each deleted store applied (the to_f(FROM_F(..)) round trips), so the
// outputs keep their bits; for f32 those round trips are identities.
// The states go M1_STEP_SLICE at a time (their h and A loaded together,
// updated, stored), the output's chain running on across the slices in
// state order, so the registers hold one slice whatever d_state and no
// state spills to local memory.
//
// proj_gate points at the in_proj output [batch, gate_stride]; the gate of
// channel d sits at column d_inner + d.
#define M1_STEP_SLICE 16
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
    if (d_state > MAMBA_RS_STATE_CAP) return;                              \
    /* Softplus as the standalone kernel spelled it, then its store. */    \
    float raw = to_f(delta_raw[idx]);                                      \
    float sp = (raw > 20.0f) ? raw : log1pf(exp2f(raw * LOG2E));           \
    float delta_d = to_f(FROM_F(sp));                                      \
    float u_d = to_f(u[idx]);                                              \
    float delta_u_d = delta_d * u_d;                                       \
    float y_d = D[d] * u_d;                                                \
    for (int n0 = 0; n0 < d_state; n0 += M1_STEP_SLICE) {                  \
        float h_local[M1_STEP_SLICE];                                      \
        float a_local[M1_STEP_SLICE];                                      \
        _Pragma("unroll")                                                  \
        for (int k = 0; k < M1_STEP_SLICE; k++) if (n0 + k >= d_state) break; else { \
            h_local[k] = h[h_base + n0 + k];                               \
            a_local[k] = a_neg[(IDX)d * d_state + n0 + k];                 \
        }                                                                  \
        _Pragma("unroll")                                                  \
        for (int k = 0; k < M1_STEP_SLICE; k++) if (n0 + k >= d_state) break; else { \
            const int n = n0 + k;                                          \
            float da = exp2f(delta_d * a_local[k] * LOG2E);                \
            float B_n = to_f(xdbl[xdbl_base + b_offset + n]);              \
            float C_n = to_f(xdbl[xdbl_base + c_offset + n]);              \
            h_local[k] = UPDATE(da, h_local[k], delta_u_d, B_n);           \
            y_d = __fmaf_rn(h_local[k], C_n, y_d);                         \
        }                                                                  \
        _Pragma("unroll")                                                  \
        for (int k = 0; k < M1_STEP_SLICE; k++) if (n0 + k >= d_state) break; else      \
            h[h_base + n0 + k] = h_local[k];                               \
    }                                                                      \
    /* The gate's SiLU as the split kernel spelled it, then its store. */  \
    float g = to_f(proj_gate[b * gate_stride + d_inner + d]);              \
    float gp = to_f(FROM_F(g / (1.0f + exp2f(-g * LOG2E))));               \
    y[idx] = FROM_F(y_d * gp);                                             \
}

DEFINE_SSM_STEP_FWD_FUSED(ssm_step_forward_fused_f32, int, float, from_f_f32, ssm_update_decay_fused)
DEFINE_SSM_STEP_FWD_FUSED(ssm_step_forward_fused_f32_i64, long long, float, from_f_f32, ssm_update_decay_fused)
DEFINE_SSM_STEP_FWD_FUSED(ssm_step_forward_fused_bf16, int, __nv_bfloat16, from_f_bf16, ssm_update_input_fused)
DEFINE_SSM_STEP_FWD_FUSED(ssm_step_forward_fused_bf16_i64, long long, __nv_bfloat16, from_f_bf16, ssm_update_input_fused)
DEFINE_SSM_STEP_FWD_FUSED(ssm_step_forward_fused_f16, int, __half, from_f_f16, ssm_update_input_fused)
DEFINE_SSM_STEP_FWD_FUSED(ssm_step_forward_fused_f16_i64, long long, __half, from_f_f16, ssm_update_input_fused)

// Templated SSM burnin nosave — sequence forward for prefill / target net.
// Activations in T_IN, state/a_neg/D stay f32. A channel's lanes as above.
#define DEFINE_SSM_BURNIN_NOSAVE(NAME, IDX, TY, FROM_F)                        \
extern "C" __global__ void NAME(                                               \
    float* h,                                                                  \
    TY* y_out,                                                                 \
    const TY* delta, const TY* u,                                              \
    const TY* B, const TY* C,                                                  \
    const float* a_neg, const float* D,                                        \
    const TY* proj_gate,                                                       \
    int gate_stride,                                                           \
    int batch, int T_len, int d_inner, int d_state, int lanes                  \
) {                                                                            \
    const IDX tid = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                \
    const int k = (int)tid & (lanes - 1);                                      \
    const IDX ch = tid >> (__ffs(lanes) - 1);                                  \
    const bool live = ch < (IDX)batch * d_inner;                               \
    const IDX b = live ? ch / d_inner : 0;                                     \
    const int d = live ? (int)(ch % d_inner) : 0;                              \
    const int per = (d_state + lanes - 1) / lanes;                             \
    const int n0 = k * per;                                                    \
    const int len = live ? max(0, min(per, d_state - n0)) : 0;                 \
    const int last = lanes - 1;                                                \
    const IDX h_base = (b * d_inner + d) * d_state + n0;                       \
    float h_local[M1_SEQ_SLICE];                                               \
    float a_local[M1_SEQ_SLICE];                                               \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < M1_SEQ_SLICE; n++) if (n >= len) break; else {         \
        h_local[n] = h[h_base + n];                                            \
        a_local[n] = a_neg[(IDX)d * d_state + n0 + n];                         \
    }                                                                          \
    float y_pass = 0.0f, delta_pass = 0.0f, u_pass = 0.0f;                     \
    for (int j = 0; j < T_len + last; j++) {                                   \
        const float y_in = __shfl_up_sync(0xffffffffu, y_pass, 1, lanes);      \
        const float delta_in =                                                 \
            __shfl_up_sync(0xffffffffu, delta_pass, 1, lanes);                 \
        const float u_in = __shfl_up_sync(0xffffffffu, u_pass, 1, lanes);      \
        const int t = j - k;                                                   \
        if (!live || t < 0 || t >= T_len) continue;                            \
        IDX bt_di = (b * T_len + t) * d_inner + d;                             \
        IDX bt_ds = (b * T_len + t) * d_state + n0;                            \
        float delta_d, u_d, y_d;                                               \
        if (k == 0) {                                                          \
            delta_d = to_f(delta[bt_di]);                                      \
            u_d = to_f(u[bt_di]);                                              \
            y_d = D[d] * u_d;                                                  \
        } else {                                                               \
            delta_d = delta_in;                                                \
            u_d = u_in;                                                        \
            y_d = y_in;                                                        \
        }                                                                      \
        float delta_u_d = delta_d * u_d;                                       \
        _Pragma("unroll")                                                      \
        for (int n = 0; n < M1_SEQ_SLICE; n++) if (n >= len) break; else {     \
            float da = exp2f(delta_d * a_local[n] * LOG2E);                    \
            h_local[n] = ssm_update_decay_fused(da, h_local[n], delta_u_d,     \
                                                to_f(B[bt_ds + n]));           \
            y_d = __fmaf_rn(h_local[n], to_f(C[bt_ds + n]), y_d);              \
        }                                                                      \
        if (k == last) {                                                       \
            TY ty = FROM_F(y_d);                                               \
            if (gate_stride > 0) {                                             \
                /* Round-trip emulation of the replaced typed chain (see       \
                 * the parallel nosave twin). */                               \
                float g = to_f(proj_gate[(b * T_len + t) * gate_stride         \
                                         + d_inner + d]);                      \
                TY tg = FROM_F(g / (1.0f + exp2f(-g * 1.4426950408889634f)));  \
                ty = FROM_F(to_f(ty) * to_f(tg));                              \
            }                                                                  \
            y_out[bt_di] = ty;                                                 \
        }                                                                      \
        y_pass = y_d;                                                          \
        delta_pass = delta_d;                                                  \
        u_pass = u_d;                                                          \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < M1_SEQ_SLICE; n++) if (n >= len) break; else           \
        h[h_base + n] = h_local[n];                                            \
}

DEFINE_SSM_BURNIN_NOSAVE(ssm_burnin_forward_nosave_f32, int, float, from_f_f32)
DEFINE_SSM_BURNIN_NOSAVE(ssm_burnin_forward_nosave_f32_i64, long long, float, from_f_f32)
DEFINE_SSM_BURNIN_NOSAVE(ssm_burnin_forward_nosave_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BURNIN_NOSAVE(ssm_burnin_forward_nosave_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BURNIN_NOSAVE(ssm_burnin_forward_nosave_f16, int, __half, from_f_f16)
DEFINE_SSM_BURNIN_NOSAVE(ssm_burnin_forward_nosave_f16_i64, long long, __half, from_f_f16)

// Templated SSM burnin WITH saves — full forward sequence for training/backward.
// h_saved and y_out in activation dtype; h_state + a_neg + D in f32. The
// first lane computes and saves the step's softplus and hands it on.
#define DEFINE_SSM_BURNIN(NAME, IDX, TY, FROM_F)                               \
extern "C" __global__ void NAME(                                               \
    float* h, TY* y_out, float* h_saved,                                       \
    const TY* delta_raw, TY* delta_saved, const TY* u,                         \
    const TY* B, const TY* C,                                                  \
    const float* a_neg, const float* D,                                        \
    int batch, int T_len, int d_inner, int d_state, int lanes                  \
) {                                                                            \
    const IDX tid = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                \
    const int k = (int)tid & (lanes - 1);                                      \
    const IDX ch = tid >> (__ffs(lanes) - 1);                                  \
    const bool live = ch < (IDX)batch * d_inner;                               \
    const IDX b = live ? ch / d_inner : 0;                                     \
    const int d = live ? (int)(ch % d_inner) : 0;                              \
    const int per = (d_state + lanes - 1) / lanes;                             \
    const int n0 = k * per;                                                    \
    const int len = live ? max(0, min(per, d_state - n0)) : 0;                 \
    const int last = lanes - 1;                                                \
    const IDX h_base = (b * d_inner + d) * d_state + n0;                       \
    float h_local[M1_SEQ_SLICE];                                               \
    float a_local[M1_SEQ_SLICE];                                               \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < M1_SEQ_SLICE; n++) if (n >= len) break; else {         \
        h_local[n] = h[h_base + n];                                            \
        a_local[n] = a_neg[(IDX)d * d_state + n0 + n];                         \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < M1_SEQ_SLICE; n++) if (n >= len) break; else {         \
        IDX hs_idx =                                                           \
            ((b * (T_len + 1) + 0) * d_inner + d) * d_state + n0 + n;          \
        h_saved[hs_idx] = h_local[n];                                          \
    }                                                                          \
    float y_pass = 0.0f, delta_pass = 0.0f, u_pass = 0.0f;                     \
    for (int j = 0; j < T_len + last; j++) {                                   \
        const float y_in = __shfl_up_sync(0xffffffffu, y_pass, 1, lanes);      \
        const float delta_in =                                                 \
            __shfl_up_sync(0xffffffffu, delta_pass, 1, lanes);                 \
        const float u_in = __shfl_up_sync(0xffffffffu, u_pass, 1, lanes);      \
        const int t = j - k;                                                   \
        if (!live || t < 0 || t >= T_len) continue;                            \
        IDX bt_di = (b * T_len + t) * d_inner + d;                             \
        IDX bt_ds = (b * T_len + t) * d_state + n0;                            \
        float delta_d, u_d, y_d;                                               \
        if (k == 0) {                                                          \
            float raw = to_f(delta_raw[bt_di]);                                \
            float sp = (raw > 20.0f)                                           \
                ? raw : log1pf(exp2f(raw * 1.4426950408889634f));              \
            TY spt = FROM_F(sp);                                               \
            delta_saved[bt_di] = spt;                                          \
            delta_d = to_f(spt);                                               \
            u_d = to_f(u[bt_di]);                                              \
            y_d = D[d] * u_d;                                                  \
        } else {                                                               \
            delta_d = delta_in;                                                \
            u_d = u_in;                                                        \
            y_d = y_in;                                                        \
        }                                                                      \
        float delta_u_d = delta_d * u_d;                                       \
        _Pragma("unroll")                                                      \
        for (int n = 0; n < M1_SEQ_SLICE; n++) if (n >= len) break; else {     \
            float da = exp2f(delta_d * a_local[n] * LOG2E);                    \
            h_local[n] = ssm_update_decay_fused(da, h_local[n], delta_u_d,     \
                                                to_f(B[bt_ds + n]));           \
            y_d = __fmaf_rn(h_local[n], to_f(C[bt_ds + n]), y_d);              \
        }                                                                      \
        if (k == last) y_out[bt_di] = FROM_F(y_d);                             \
        _Pragma("unroll")                                                      \
        for (int n = 0; n < M1_SEQ_SLICE; n++) if (n >= len) break; else {     \
            IDX hs_idx =                                                       \
                ((b * (T_len + 1) + (t + 1)) * d_inner + d) * d_state + n0 + n; \
            h_saved[hs_idx] = h_local[n];                                      \
        }                                                                      \
        y_pass = y_d;                                                          \
        delta_pass = delta_d;                                                  \
        u_pass = u_d;                                                          \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < M1_SEQ_SLICE; n++) if (n >= len) break; else           \
        h[h_base + n] = h_local[n];                                            \
}

DEFINE_SSM_BURNIN(ssm_burnin_forward_f32, int, float, from_f_f32)
DEFINE_SSM_BURNIN(ssm_burnin_forward_f32_i64, long long, float, from_f_f32)
DEFINE_SSM_BURNIN(ssm_burnin_forward_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BURNIN(ssm_burnin_forward_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BURNIN(ssm_burnin_forward_f16, int, __half, from_f_f16)
DEFINE_SSM_BURNIN(ssm_burnin_forward_f16_i64, long long, __half, from_f_f16)

// ======================== BACKWARD ========================

// SSM backward with per-sample LOCAL gradient accumulation (no atomicAdd).
// A channel's lanes (see the forward) accumulate local d_D (the first lane)
// and d_a_log (each lane its slice) and write them to per-sample buffers.
// Separate reduction kernels sum across batch dimension afterward.
// a_neg cached in registers (Opt A). Uses exp2f (Opt B).
//
// Source: CPU reference: train/forward.rs phase B3
// FIX C2: no atomicAdd, local accumulation + reduction
// One lane's slice of a backward step: the whole-state walk's state loop
// over the lane's states, the d_u and d_delta chains carried in d_u_val and
// d_delta_val. The guard holds from state GUARD_FROM on, so a caller that
// knows the lane's first state exists runs it unguarded.
#define M1_BWD_SLICE(IDX, GUARD_FROM)                                          \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < M1_SEQ_SLICE; n++)                                     \
        if (n >= (GUARD_FROM) && n >= len) break; else {                       \
        float h_curr = h_carry[n];                                             \
        float da = exp2f(delta_d * a_local[n] * LOG2E);                        \
        d_h[n] += dy_d * C_saved[bt_ds + n];                                   \
        IDX btdn = ((b * T + t) * d_inner + d) * d_state + n0 + n;             \
        d_C_local[btdn] = dy_d * h_curr;                                       \
        IDX h_prev_idx = ((b * (T + 1) + t) * d_inner + d) * d_state + n0 + n; \
        float h_prev = h_saved[h_prev_idx];                                    \
        d_delta_val += d_h[n] * (a_local[n] * da * h_prev + u_d * B_saved[bt_ds + n]); \
        d_u_val += d_h[n] * delta_d * B_saved[bt_ds + n];                      \
        d_B_local[btdn] = d_h[n] * delta_d * u_d;                              \
        d_a_acc[n] += d_h[n] * da * delta_d * a_local[n] * h_prev;             \
        d_h[n] = da * d_h[n];                                                  \
        h_carry[n] = h_prev;                                                   \
    }

#define DEFINE_SSM_BACKWARD_LOCAL(NAME, IDX)                                   \
extern "C" __global__ void NAME(                                               \
    const float* h_saved, /* [batch * (T+1) * d_inner * d_state] saved h BEFORE each step */ \
    const float* delta_saved, /* [batch * T * d_inner] */                      \
    const float* u_saved, /* [batch * T * d_inner] */                          \
    const float* B_saved, /* [batch * T * d_state] */                          \
    const float* C_saved, /* [batch * T * d_state] */                          \
    const float* a_neg, /* [d_inner * d_state] */                              \
    const float* D, /* [d_inner] skip connection weight */                     \
    const float* dy, /* [batch * T * d_inner] */                               \
    float* d_delta, /* [batch * T * d_inner] per-sample */                     \
    float* d_u, /* [batch * T * d_inner] per-sample (includes skip: += dy*D[d]) */ \
    float* d_B_local, /* [batch * T * d_inner * d_state] per-thread */         \
    float* d_C_local, /* [batch * T * d_inner * d_state] per-thread */         \
    float* d_D_local, /* [batch * d_inner] per-sample (sum across T) */        \
    float* d_a_log_local, /* [batch * d_inner * d_state] per-sample (sum across T) */ \
    int batch, int T, int d_inner, int d_state, int lanes                      \
) {                                                                            \
    const IDX tid = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                \
    const int k = (int)tid & (lanes - 1);                                      \
    const IDX ch = tid >> (__ffs(lanes) - 1);                                  \
    const bool live = ch < (IDX)batch * d_inner;                               \
    const IDX b = live ? ch / d_inner : 0;                                     \
    const int d = live ? (int)(ch % d_inner) : 0;                              \
    const int per = (d_state + lanes - 1) / lanes;                             \
    const int n0 = k * per;                                                    \
    const int len = live ? max(0, min(per, d_state - n0)) : 0;                 \
    const int last = lanes - 1;                                                \
    float local_d_D = 0.0f;                                                    \
    float a_local[M1_SEQ_SLICE];                                               \
    float d_h[M1_SEQ_SLICE];                                                   \
    float d_a_acc[M1_SEQ_SLICE];                                               \
    float h_carry[M1_SEQ_SLICE];                                               \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < M1_SEQ_SLICE; n++) if (n >= len) break; else {         \
        a_local[n] = a_neg[(IDX)d * d_state + n0 + n];                         \
        d_h[n] = 0.0f;                                                         \
        d_a_acc[n] = 0.0f;                                                     \
        h_carry[n] =                                                           \
            h_saved[((b * (T + 1) + T) * d_inner + d) * d_state + n0 + n];     \
    }                                                                          \
    float dy_pass = 0.0f, delta_pass = 0.0f, u_pass = 0.0f;                    \
    float dd_pass = 0.0f, du_pass = 0.0f;                                      \
    for (int j = 0; j < T + last; j++) {                                       \
        const float dy_in = __shfl_up_sync(0xffffffffu, dy_pass, 1, lanes);    \
        const float delta_in =                                                 \
            __shfl_up_sync(0xffffffffu, delta_pass, 1, lanes);                 \
        const float u_in = __shfl_up_sync(0xffffffffu, u_pass, 1, lanes);      \
        const float dd_in = __shfl_up_sync(0xffffffffu, dd_pass, 1, lanes);    \
        const float du_in = __shfl_up_sync(0xffffffffu, du_pass, 1, lanes);    \
        const int t = T - 1 - (j - k);                                         \
        if (!live || t < 0 || t >= T) continue;                                \
        IDX bt_di = (b * T + t) * d_inner + d;                                 \
        IDX bt_ds = (b * T + t) * d_state + n0;                                \
        float dy_d, delta_d, u_d, d_u_val, d_delta_val;                        \
        /* Each branch runs the slice loop itself, the first lane with its */  \
        /* first state unguarded (a live channel has one): its d_u chain */    \
        /* then starts from the skip product in the block of that state, */    \
        /* as the whole-state walk's did, and the compiler fuses the */        \
        /* state's add the same way. */                                        \
        if (k == 0) {                                                          \
            dy_d = dy[bt_di];                                                  \
            delta_d = delta_saved[bt_di];                                      \
            u_d = u_saved[bt_di];                                              \
            local_d_D += dy_d * u_d;                                           \
            d_u_val = dy_d * D[d];                                             \
            d_delta_val = 0.0f;                                                \
            M1_BWD_SLICE(IDX, 1)                                               \
        } else {                                                               \
            dy_d = dy_in;                                                      \
            delta_d = delta_in;                                                \
            u_d = u_in;                                                        \
            d_u_val = du_in;                                                   \
            d_delta_val = dd_in;                                               \
            M1_BWD_SLICE(IDX, 0)                                               \
        }                                                                      \
        if (k == last) {                                                       \
            d_delta[bt_di] = d_delta_val;                                      \
            d_u[bt_di] = d_u_val;                                              \
        }                                                                      \
        dy_pass = dy_d;                                                        \
        delta_pass = delta_d;                                                  \
        u_pass = u_d;                                                          \
        dd_pass = d_delta_val;                                                 \
        du_pass = d_u_val;                                                     \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < M1_SEQ_SLICE; n++) if (n >= len) break; else           \
        d_a_log_local[(b * d_inner + d) * d_state + n0 + n] = d_a_acc[n];      \
    if (live && k == 0) d_D_local[b * d_inner + d] = local_d_D;                \
}

DEFINE_SSM_BACKWARD_LOCAL(ssm_backward_local, int)
DEFINE_SSM_BACKWARD_LOCAL(ssm_backward_local_i64, long long)

// ssm_backward_local typed (bf16/f16/f32) for mixed-precision training.
// HOTTEST + HIGHEST RISK kernel in M1 backward. Validated against:
//   - state-spaces/mamba csrc/selective_scan_bwd_kernel.cuh
//     (scan_t = float2 → BPTT state stays f32)
//   - NVIDIA AMP backward best-practice (typed IO, f32 accumulators)
//
// PRECISION RULES (per validation table):
//   - h_saved, a_neg, D                 → f32 (BPTT state + model params)
//   - delta, u, B, C, dy                → typed (post-promote on load)
//   - d_delta, d_u, d_B_local, d_C_local → typed (downcast on store)
//   - d_D_local, d_a_log_local          → f32 master (T-length += accumulators)
//   - register dh, a_local (a lane's slice), local_d_D → f32 (BPTT precision invariant)
//
// A channel's lanes as for the forward: the first lane reads the step's
// rows and starts the d_delta and d_u chains, the last one stores them.
//
// Math identical to f32 ssm_backward_local above; only IO dtype differs.
// The typed twin of M1_BWD_SLICE, its rows promoted on load and its
// per-state gradients rounded on store.
#define M1_BWD_SLICE_TYPED(IDX, FROM_F, GUARD_FROM)                            \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < M1_SEQ_SLICE; n++)                                     \
        if (n >= (GUARD_FROM) && n >= len) break; else {                       \
        IDX h_prev_idx =                                                       \
            ((b * (T + 1) + t) * d_inner + d) * d_state + n0 + n;              \
        float h_curr = h_carry[n];                                             \
        float h_prev = h_saved[h_prev_idx];                                    \
        float B_n    = to_f(B_saved[bt_ds + n]);                               \
        float C_n    = to_f(C_saved[bt_ds + n]);                               \
        float da     = exp2f(delta_d * a_local[n] * LOG2E);                    \
        d_h[n] += dy_d * C_n;                                                  \
        IDX btdn = ((b * T + t) * d_inner + d) * d_state + n0 + n;             \
        d_C_local[btdn] = FROM_F(dy_d * h_curr);                               \
        d_delta_val += d_h[n] * (a_local[n] * da * h_prev + u_d * B_n);        \
        d_u_val     += d_h[n] * delta_d * B_n;                                 \
        d_B_local[btdn] = FROM_F(d_h[n] * delta_d * u_d);                      \
        d_a_acc[n] += d_h[n] * da * delta_d * a_local[n] * h_prev;             \
        d_h[n] = da * d_h[n];                                                  \
        h_carry[n] = h_prev;                                                   \
    }

#define DEFINE_SSM_BACKWARD_LOCAL_BWD(NAME, IDX, TY, FROM_F)                   \
extern "C" __global__ void NAME(                                               \
    const float* h_saved, /* [batch * (T+1) * d_inner * d_state] saved h BEFORE each step */ \
    const TY* delta_saved, /* [batch * T * d_inner] */                         \
    const TY* u_saved, /* [batch * T * d_inner] */                             \
    const TY* B_saved, /* [batch * T * d_state] */                             \
    const TY* C_saved, /* [batch * T * d_state] */                             \
    const float* a_neg, /* [d_inner * d_state] */                              \
    const float* D, /* [d_inner] skip connection weight */                     \
    const TY* dy, /* [batch * T * d_inner] */                                  \
    TY* d_delta, /* [batch * T * d_inner] per-sample */                        \
    TY* d_u, /* [batch * T * d_inner] per-sample (includes skip: += dy*D[d]) */ \
    TY* d_B_local, /* [batch * T * d_inner * d_state] per-thread */            \
    TY* d_C_local, /* [batch * T * d_inner * d_state] per-thread */            \
    float* d_D_local, /* [batch * d_inner] per-sample (sum across T) */        \
    float* d_a_log_local, /* [batch * d_inner * d_state] per-sample (sum across T) */ \
    int batch, int T, int d_inner, int d_state, int lanes                      \
) {                                                                            \
    const IDX tid = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                \
    const int k = (int)tid & (lanes - 1);                                      \
    const IDX ch = tid >> (__ffs(lanes) - 1);                                  \
    const bool live = ch < (IDX)batch * d_inner;                               \
    const IDX b = live ? ch / d_inner : 0;                                     \
    const int d = live ? (int)(ch % d_inner) : 0;                              \
    const int per = (d_state + lanes - 1) / lanes;                             \
    const int n0 = k * per;                                                    \
    const int len = live ? max(0, min(per, d_state - n0)) : 0;                 \
    const int last = lanes - 1;                                                \
    float local_d_D = 0.0f;                                                    \
    float a_local[M1_SEQ_SLICE];                                               \
    float d_h[M1_SEQ_SLICE];                                                   \
    float d_a_acc[M1_SEQ_SLICE];                                               \
    float h_carry[M1_SEQ_SLICE];                                               \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < M1_SEQ_SLICE; n++) if (n >= len) break; else {         \
        a_local[n] = a_neg[(IDX)d * d_state + n0 + n];                         \
        d_h[n] = 0.0f;                                                         \
        d_a_acc[n] = 0.0f;                                                     \
        h_carry[n] =                                                           \
            h_saved[((b * (T + 1) + T) * d_inner + d) * d_state + n0 + n];     \
    }                                                                          \
    float dy_pass = 0.0f, delta_pass = 0.0f, u_pass = 0.0f;                    \
    float dd_pass = 0.0f, du_pass = 0.0f;                                      \
    for (int j = 0; j < T + last; j++) {                                       \
        const float dy_in = __shfl_up_sync(0xffffffffu, dy_pass, 1, lanes);    \
        const float delta_in =                                                 \
            __shfl_up_sync(0xffffffffu, delta_pass, 1, lanes);                 \
        const float u_in = __shfl_up_sync(0xffffffffu, u_pass, 1, lanes);      \
        const float dd_in = __shfl_up_sync(0xffffffffu, dd_pass, 1, lanes);    \
        const float du_in = __shfl_up_sync(0xffffffffu, du_pass, 1, lanes);    \
        const int t = T - 1 - (j - k);                                         \
        if (!live || t < 0 || t >= T) continue;                                \
        IDX bt_di = (b * T + t) * d_inner + d;                                 \
        IDX bt_ds = (b * T + t) * d_state + n0;                                \
        float dy_d, delta_d, u_d, d_u_val, d_delta_val;                        \
        /* Each branch runs the slice loop itself, the first lane with its */  \
        /* first state unguarded (a live channel has one): its d_u chain */    \
        /* then starts from the skip product in the block of that state, */    \
        /* as the whole-state walk's did, and the compiler fuses the */        \
        /* state's add the same way. */                                        \
        if (k == 0) {                                                          \
            dy_d    = to_f(dy[bt_di]);                                         \
            delta_d = to_f(delta_saved[bt_di]);                                \
            u_d     = to_f(u_saved[bt_di]);                                    \
            local_d_D += dy_d * u_d;                                           \
            d_u_val = dy_d * D[d];                                             \
            d_delta_val = 0.0f;                                                \
            M1_BWD_SLICE_TYPED(IDX, FROM_F, 1)                                 \
        } else {                                                               \
            dy_d = dy_in;                                                      \
            delta_d = delta_in;                                                \
            u_d = u_in;                                                        \
            d_u_val = du_in;                                                   \
            d_delta_val = dd_in;                                               \
            M1_BWD_SLICE_TYPED(IDX, FROM_F, 0)                                 \
        }                                                                      \
        if (k == last) {                                                       \
            d_delta[bt_di] = FROM_F(d_delta_val);                              \
            d_u[bt_di] = FROM_F(d_u_val);                                      \
        }                                                                      \
        dy_pass = dy_d;                                                        \
        delta_pass = delta_d;                                                  \
        u_pass = u_d;                                                          \
        dd_pass = d_delta_val;                                                 \
        du_pass = d_u_val;                                                     \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < M1_SEQ_SLICE; n++) if (n >= len) break; else           \
        d_a_log_local[(b * d_inner + d) * d_state + n0 + n] = d_a_acc[n];      \
    if (live && k == 0) d_D_local[b * d_inner + d] = local_d_D;                \
}

DEFINE_SSM_BACKWARD_LOCAL_BWD(ssm_backward_local_f32, int, float, from_f_f32)
DEFINE_SSM_BACKWARD_LOCAL_BWD(ssm_backward_local_f32_i64, long long, float, from_f_f32)
DEFINE_SSM_BACKWARD_LOCAL_BWD(ssm_backward_local_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BACKWARD_LOCAL_BWD(ssm_backward_local_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BACKWARD_LOCAL_BWD(ssm_backward_local_f16, int, __half, from_f_f16)
DEFINE_SSM_BACKWARD_LOCAL_BWD(ssm_backward_local_f16_i64, long long, __half, from_f_f16)

// Reduction kernels: sum per-sample gradients across batch dimension.

// Fused d_B + d_C reduction across d_inner — one launch instead of two,
// and a full-domain `= (0.0f + sum)` store instead of `+=` onto a
// pre-zeroed buffer, which lets both callers drop their memsets. The two
// inner loops are verbatim copies of the old split reducers (same
// ascending-d order, f32 sums), so every output value is bit-identical
// to the old zero+`+=` pair INCLUDING at sum == -0.0 (0.0f + -0.0f =
// +0.0f — exactly what `+=`-on-zero produced; a bare `= sum` would
// store -0.0 and flip the sign bit downstream).
#define DEFINE_SSM_REDUCE_D_BC_FUSED(SUFFIX, TY)                               \
extern "C" __global__ void ssm_reduce_d_BC_##SUFFIX(                           \
    float* d_B_out,           /* [batch * T * d_state] */                      \
    float* d_C_out,           /* [batch * T * d_state] */                      \
    const TY* d_B_local,      /* [batch * T * d_inner * d_state] */            \
    const TY* d_C_local,      /* [batch * T * d_inner * d_state] */            \
    int batch, int T, int d_inner, int d_state                                 \
) {                                                                            \
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;          \
    long long total = (long long)batch * T * d_state;                          \
    if (idx >= total) return;                                                  \
    long long bt = idx / d_state;                                              \
    int n = (int)(idx % d_state);                                              \
    float sum_b = 0.0f;                                                        \
    for (int d = 0; d < d_inner; d++) {                                        \
        sum_b += to_f(d_B_local[(bt * d_inner + d) * d_state + n]);            \
    }                                                                          \
    d_B_out[idx] = 0.0f + sum_b;                                               \
    float sum_c = 0.0f;                                                        \
    for (int d = 0; d < d_inner; d++) {                                        \
        sum_c += to_f(d_C_local[(bt * d_inner + d) * d_state + n]);            \
    }                                                                          \
    d_C_out[idx] = 0.0f + sum_c;                                               \
}

DEFINE_SSM_REDUCE_D_BC_FUSED(f32,  float)
DEFINE_SSM_REDUCE_D_BC_FUSED(bf16, __nv_bfloat16)
DEFINE_SSM_REDUCE_D_BC_FUSED(f16,  __half)

// T-major twin of ssm_reduce_d_BC_*: reads the PARALLEL route's
// [b][n][d][t] locals (the S2 tape layout). Thread <-> (b, n, t) with t
// innermost so warp reads coalesce; the inner d loop stays ASCENDING
// with the same f32 sum and the same `= (0.0f + sum)` store, so every
// output VALUE is bit-identical to the historical reducer. The output
// layout ([b*T + t]*ds + n) is unchanged — downstream consumers never
// see the tape layout.
#define DEFINE_SSM_REDUCE_D_BC_TMAJOR(SUFFIX, TY)                              \
extern "C" __global__ void ssm_reduce_d_BC_tmajor_##SUFFIX(                    \
    float* d_B_out,           /* [batch * T * d_state] */                      \
    float* d_C_out,           /* [batch * T * d_state] */                      \
    const TY* d_B_local,      /* [batch * d_state * d_inner * T] */            \
    const TY* d_C_local,      /* [batch * d_state * d_inner * T] */            \
    int batch, int T, int d_inner, int d_state                                 \
) {                                                                            \
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;          \
    long long total = (long long)batch * T * d_state;                          \
    if (idx >= total) return;                                                  \
    int t = (int)(idx % T);                                                    \
    long long rem = idx / T;                                                   \
    int n = (int)(rem % d_state);                                              \
    long long b = rem / d_state;                                               \
    long long out_idx = (b * T + t) * d_state + n;                             \
    float sum_b = 0.0f;                                                        \
    for (int d = 0; d < d_inner; d++) {                                        \
        sum_b += to_f(d_B_local[((b * d_state + n) * d_inner + d) * T + t]);   \
    }                                                                          \
    d_B_out[out_idx] = 0.0f + sum_b;                                           \
    float sum_c = 0.0f;                                                        \
    for (int d = 0; d < d_inner; d++) {                                        \
        sum_c += to_f(d_C_local[((b * d_state + n) * d_inner + d) * T + t]);   \
    }                                                                          \
    d_C_out[out_idx] = 0.0f + sum_c;                                           \
}

DEFINE_SSM_REDUCE_D_BC_TMAJOR(f32,  float)
DEFINE_SSM_REDUCE_D_BC_TMAJOR(bf16, __nv_bfloat16)
DEFINE_SSM_REDUCE_D_BC_TMAJOR(f16,  __half)

// Reduce d_D: d_D_out[d] = sum_b(d_D_local[b * d_inner + d])
extern "C" __global__ void ssm_reduce_d_D(
    float* d_D_out,           // [d_inner] accumulated
    const float* d_D_local,   // [batch * d_inner]
    int batch, int d_inner
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= d_inner) return;
    float sum = 0.0f;
    for (int b = 0; b < batch; b++) {
        sum += d_D_local[(long long)b * d_inner + d];
    }
    d_D_out[d] += sum;
}

// (The old split ssm_reduce_d_B / ssm_reduce_d_C kernels and their typed
// twins are gone: the fused ssm_reduce_d_BC_* above covers all three
// dtypes — the typed variants promote each contribution to f32 in the
// inner loop, output stays the f32 master per the AMP convention.
// d_D and d_a_log reducers stay untyped: their inputs are already f32
// per the precision rules in DEFINE_SSM_BACKWARD_LOCAL_BWD.)

// Reduce d_a_log: d_a_log_out[d*ds+n] = sum_b(d_a_log_local[b*di*ds + d*ds + n])
// Chunk-partial variant for the fold backward: partials arrive as
// [batch * n_chunks, d_inner * d_state] rows written in the walk order
// (descending time). The inner fold runs the slots of one sample first
// - reproducing the old per-sample accumulator chain - and only then
// adds across the batch, exactly the association the accumulate-then-
// reduce pair produced.
extern "C" __global__ void ssm_reduce_d_a_log_chunks(
    float* d_a_log_out,         // [d_inner * d_state] accumulated
    const float* partials,      // [batch * n_chunks * d_inner * d_state]
    int batch, int n_chunks, int d_inner, int d_state
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)d_inner * d_state;
    if (idx >= total) return;
    float sum = 0.0f;
    for (int b = 0; b < batch; b++) {
        float acc = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            acc += partials[((long long)b * n_chunks + c) * total + idx];
        }
        sum += acc;
    }
    d_a_log_out[idx] += sum;
}

extern "C" __global__ void ssm_reduce_d_a_log(
    float* d_a_log_out,         // [d_inner * d_state] accumulated
    const float* d_a_log_local, // [batch * d_inner * d_state]
    int batch, int d_inner, int d_state
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)d_inner * d_state;
    if (idx >= total) return;
    float sum = 0.0f;
    for (int b = 0; b < batch; b++) {
        sum += d_a_log_local[b * total + idx];
    }
    d_a_log_out[idx] += sum;
}
