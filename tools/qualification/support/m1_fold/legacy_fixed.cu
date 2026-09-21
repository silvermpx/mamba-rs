#line 1 "kernels/_typed_prelude.cuh"
// Shared prelude for templated (multi-dtype) kernels.
//
// Pattern: every activation-touching kernel has 3 extern "C" instantiations:
//   NAME_f32, NAME_bf16, NAME_f16
// Suffix is chosen by Rust dispatch based on activation dtype.
//
// All math happens in f32; dtype conversion is upcast-on-load,
// downcast-on-store (single PTX cvt instruction each).
//
// Storage typing: T_IN / T_OUT are the activation dtype.
// Weights that must remain f32 (a_log/a_neg, D, norm weights, biases)
// are passed as `const float*` explicitly.

#ifndef _MAMBA_TYPED_PRELUDE_CUH
#define _MAMBA_TYPED_PRELUDE_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

// ---- Upcast helpers (load) ------------------------------------------------
__device__ __forceinline__ float to_f(float v)          { return v; }
__device__ __forceinline__ float to_f(__nv_bfloat16 v)  { return __bfloat162float(v); }
__device__ __forceinline__ float to_f(__half v)         { return __half2float(v); }

// ---- Downcast helpers (store) --------------------------------------------
__device__ __forceinline__ float         from_f_f32(float v)  { return v; }
__device__ __forceinline__ __nv_bfloat16 from_f_bf16(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ __half        from_f_f16(float v)  { return __float2half_rn(v); }

// ---- Packed pair upcast (2 elements at once) ------------------------------
// Halves LDS instruction count for warp-uniform smem reads. Used in matvec
// inner loop where smem_a reads are broadcast to all lanes in a warp.
// Address must be 4-byte aligned (2 elements × sizeof(T_IO)).
__device__ __forceinline__ float2 pair_to_f2(const float* p) {
    return {p[0], p[1]};
}
__device__ __forceinline__ float2 pair_to_f2(const __nv_bfloat16* p) {
    __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(p);
    return {__bfloat162float(__low2bfloat16(v)),
            __bfloat162float(__high2bfloat16(v))};
}
__device__ __forceinline__ float2 pair_to_f2(const __half* p) {
    __half2 v = *reinterpret_cast<const __half2*>(p);
    return {__half2float(__low2half(v)),
            __half2float(__high2half(v))};
}

#endif  // _MAMBA_TYPED_PRELUDE_CUH
#line 1 "kernels/mamba_ssm.cu"
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

// State-dimension capacity of the per-thread register arrays below.
// Injected at JIT time (-DMAMBA_RS_STATE_CAP=...) from the model config
// so any reference-range d_state runs the same code path; 64 covers the
// common shapes at minimum register pressure. Past ~128 the compiler
// spills these arrays to local memory - correct, measurably slower,
// and accepted: capacity is a first-class knob, not a fallback.
// The state loops run to this compiled capacity so the state arrays index
// statically and stay in registers; each loop leaves at the model's
// d_state, so a capacity wider than the state costs nothing per step.
#ifndef MAMBA_RS_STATE_CAP
#define MAMBA_RS_STATE_CAP 64
#endif



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

// ======================== FORWARD ========================

// SSM step forward (T=1): one step per (batch, d_inner) thread.
// Each thread handles all d_state elements for its (b, d) pair.
// h and a_neg cached in registers (Opt A). Uses exp2f (Opt B).
extern "C" __global__ void ssm_step_forward(
    float* h,           // [batch * d_inner * d_state] hidden state (mutated)
    float* y,           // [batch * d_inner] output
    const float* delta, // [batch * d_inner] after softplus
    const float* u,     // [batch * d_inner] gated input
    const float* B,     // [batch * d_state] from x_proj
    const float* C,     // [batch * d_state] from x_proj
    const float* a_neg, // [d_inner * d_state] = -exp(a_log), shared across batch
    const float* D,     // [d_inner] skip connection
    int batch, int d_inner, int d_state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_inner;
    if (idx >= total) return;

    int b = idx / d_inner;
    int d = idx % d_inner;
    int h_base = (b * d_inner + d) * d_state;

    // Opt A: load h and a_neg into registers
    // CONSTRAINT: d_state <= 64. Validated in Rust launch code.
    float h_local[MAMBA_RS_STATE_CAP];
    float a_local[MAMBA_RS_STATE_CAP];
    if (d_state > MAMBA_RS_STATE_CAP) return;
    #pragma unroll
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {
        h_local[n] = h[h_base + n];
        a_local[n] = a_neg[d * d_state + n];
    }

    float delta_d = delta[idx];
    float u_d = u[idx];
    float delta_u_d = delta_d * u_d; // C2: hoisted
    float y_d = D[d] * u_d;

    #pragma unroll
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {
        // Opt B: exp2f instead of expf
        float da = exp2f(delta_d * a_local[n] * LOG2E);
        h_local[n] = ssm_update_decay_fused(da, h_local[n], delta_u_d,
                                            B[b * d_state + n]);
        y_d = __fmaf_rn(h_local[n], C[b * d_state + n], y_d);
    }

    // Opt A: write back h once
    #pragma unroll
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else
        h[h_base + n] = h_local[n];

    y[idx] = y_d;
}

// The decode SSM step with its neighbours folded in: softplus on the
// dt_proj output, B and C read straight from xdbl, and the gate read from
// the in_proj output with the SiLU recomputed. One launch replaces the
// softplus, gather, step and gating kernels. Every folded value is
// spelled as the kernel it replaces spelled it, including the rounding
// each deleted store applied (the to_f(FROM_F(..)) round trips), so the
// outputs keep their bits; for f32 those round trips are identities.
//
// proj_gate points at the in_proj output [batch, gate_stride]; the gate of
// channel d sits at column d_inner + d.
#define DEFINE_SSM_STEP_FWD_FUSED(SUFFIX, T, FROM_F, UPDATE)               \
extern "C" __global__ void ssm_step_forward_fused_##SUFFIX(                 \
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    int total = batch * d_inner;                                           \
    if (idx >= total) return;                                              \
    int b = idx / d_inner;                                                 \
    int d = idx % d_inner;                                                 \
    int h_base = (b * d_inner + d) * d_state;                              \
    int xdbl_base = b * xdbl_stride;                                       \
    float h_local[MAMBA_RS_STATE_CAP];                                     \
    float a_local[MAMBA_RS_STATE_CAP];                                     \
    if (d_state > MAMBA_RS_STATE_CAP) return;                              \
    _Pragma("unroll")                                                      \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {        \
        h_local[n] = h[h_base + n];                                        \
        a_local[n] = a_neg[d * d_state + n];                               \
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

DEFINE_SSM_STEP_FWD_FUSED(f32,  float,         from_f_f32,  ssm_update_decay_fused)
DEFINE_SSM_STEP_FWD_FUSED(bf16, __nv_bfloat16, from_f_bf16, ssm_update_input_fused)
DEFINE_SSM_STEP_FWD_FUSED(f16,  __half,        from_f_f16,  ssm_update_input_fused)

// SSM burn-in forward (T>1): iterate T steps for each (batch, d_inner) thread.
// Saves h_saved[B*(T+1)*d_inner*d_state] for backward BPTT and
// h and a_neg cached in registers (Opt A). Uses exp2f (Opt B).
extern "C" __global__ void ssm_burnin_forward(
    float* h,             // [batch * d_inner * d_state] hidden state (mutated through T steps)
    float* y_out,         // [batch * T * d_inner] output
    float* h_saved,       // [batch * (T+1) * d_inner * d_state] h BEFORE each step
    // Pre-softplus dt: softplus is applied inline (same value the deleted
    // copy pass stored) and the post-softplus save is written here.
    const float* delta_raw, // [batch * T * d_inner]
    float* delta_saved,     // [batch * T * d_inner]
    const float* u,       // [batch * T * d_inner]
    const float* B,       // [batch * T * d_state]
    const float* C,       // [batch * T * d_state]
    const float* a_neg,   // [d_inner * d_state]
    const float* D,       // [d_inner]
    int batch, int T, int d_inner, int d_state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_inner;
    if (idx >= total) return;

    int b = idx / d_inner;
    int d = idx % d_inner;
    int h_base = (b * d_inner + d) * d_state;

    // Opt A: load h and a_neg into registers
    float h_local[MAMBA_RS_STATE_CAP];
    float a_local[MAMBA_RS_STATE_CAP];
    if (d_state > MAMBA_RS_STATE_CAP) return;
    #pragma unroll
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {
        h_local[n] = h[h_base + n];
        a_local[n] = a_neg[d * d_state + n];
    }

    // Save initial h state at time index 0
    #pragma unroll
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {
        int hs_idx = (b * (T + 1) + 0) * d_inner * d_state + d * d_state + n;
        h_saved[hs_idx] = h_local[n];
    }

    for (int t = 0; t < T; t++) {
        int bt_di = (b * T + t) * d_inner + d;
        int bt_ds = (b * T + t) * d_state;

        float raw = delta_raw[bt_di];
        float delta_d =
            (raw > 20.0f) ? raw : log1pf(exp2f(raw * 1.4426950408889634f));
        delta_saved[bt_di] = delta_d;
        float u_d = u[bt_di];
        float delta_u_d = delta_d * u_d; // C2: hoisted
        float y_d = D[d] * u_d;

        #pragma unroll
        for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {
            // Opt B: exp2f instead of expf
            float da = exp2f(delta_d * a_local[n] * LOG2E);

            // No da saved: backward recomputes da from delta and a_neg
            // (cheaper than a global-memory round-trip at d_state=16).

            h_local[n] = ssm_update_decay_fused(da, h_local[n], delta_u_d,
                                                B[bt_ds + n]);
            y_d = __fmaf_rn(h_local[n], C[bt_ds + n], y_d);
        }

        y_out[bt_di] = y_d;

        // Save h AFTER step t = h_saved at time index t+1
        #pragma unroll
        for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {
            int hs_idx = (b * (T + 1) + (t + 1)) * d_inner * d_state + d * d_state + n;
            h_saved[hs_idx] = h_local[n];
        }
    }

    // Opt A: write back final h once
    #pragma unroll
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else
        h[h_base + n] = h_local[n];
}

// SSM burn-in forward NOSAVE variant (target network — no backward needed).
// Identical recurrence to ssm_burnin_forward but skips the h_saved writes.
// Saves ~50% memory bandwidth per layer for target path.
extern "C" __global__ void ssm_burnin_forward_nosave(
    float* h,             // [batch * d_inner * d_state] hidden state (mutated through T steps)
    float* y_out,         // [batch * T * d_inner] output
    const float* delta,   // [batch * T * d_inner]
    const float* u,       // [batch * T * d_inner]
    const float* B,       // [batch * T * d_state]
    const float* C,       // [batch * T * d_state]
    const float* a_neg,   // [d_inner * d_state]
    const float* D,       // [d_inner]
    const float* proj_gate, // in_proj output when gating fuses
    int gate_stride,      // proj row stride; 0 = plain y store
    int batch, int T, int d_inner, int d_state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_inner;
    if (idx >= total) return;

    int b = idx / d_inner;
    int d = idx % d_inner;
    int h_base = (b * d_inner + d) * d_state;

    // Opt A: load h and a_neg into registers
    float h_local[MAMBA_RS_STATE_CAP];
    float a_local[MAMBA_RS_STATE_CAP];
    if (d_state > MAMBA_RS_STATE_CAP) return;
    #pragma unroll
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {
        h_local[n] = h[h_base + n];
        a_local[n] = a_neg[d * d_state + n];
    }

    for (int t = 0; t < T; t++) {
        int bt_di = (b * T + t) * d_inner + d;
        int bt_ds = (b * T + t) * d_state;

        float delta_d = delta[bt_di];
        float u_d = u[bt_di];
        float delta_u_d = delta_d * u_d; // C2: hoisted
        float y_d = D[d] * u_d;

        #pragma unroll
        for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {
            // Opt B: exp2f instead of expf
            float da = exp2f(delta_d * a_local[n] * LOG2E);
            h_local[n] = ssm_update_decay_fused(da, h_local[n], delta_u_d,
                                                B[bt_ds + n]);
            y_d = __fmaf_rn(h_local[n], C[bt_ds + n], y_d);
        }

        if (gate_stride > 0) {
            // Fused gating - same one-rounding product and SiLU formula
            // as the replaced split/mul chain.
            float g = proj_gate[(b * T + t) * gate_stride + d_inner + d];
            y_d *= g / (1.0f + exp2f(-g * 1.4426950408889634f));
        }
        y_out[bt_di] = y_d;
    }

    // Opt A: write back final h once
    #pragma unroll
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else
        h[h_base + n] = h_local[n];
}

// Templated SSM burnin nosave — sequence forward for prefill / target net.
// Activations in T_IN, state/a_neg/D stay f32.
#define DEFINE_SSM_BURNIN_NOSAVE(SUFFIX, TY, FROM_F)                       \
extern "C" __global__ void ssm_burnin_forward_nosave_##SUFFIX(             \
    float* h,                                                              \
    TY* y_out,                                                             \
    const TY* delta, const TY* u,                                          \
    const TY* B, const TY* C,                                              \
    const float* a_neg, const float* D,                                    \
    const TY* proj_gate,                                                   \
    int gate_stride,                                                       \
    int batch, int T_len, int d_inner, int d_state                         \
) {                                                                        \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    int total = batch * d_inner;                                           \
    if (idx >= total) return;                                              \
    int b = idx / d_inner;                                                 \
    int d = idx % d_inner;                                                 \
    int h_base = (b * d_inner + d) * d_state;                              \
    float h_local[MAMBA_RS_STATE_CAP];                                                     \
    float a_local[MAMBA_RS_STATE_CAP];                                                     \
    if (d_state > MAMBA_RS_STATE_CAP) return;                                              \
    _Pragma("unroll")                                                      \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {        \
        h_local[n] = h[h_base + n];                                        \
        a_local[n] = a_neg[d * d_state + n];                               \
    }                                                                      \
    for (int t = 0; t < T_len; t++) {                                      \
        int bt_di = (b * T_len + t) * d_inner + d;                         \
        int bt_ds = (b * T_len + t) * d_state;                             \
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

DEFINE_SSM_BURNIN_NOSAVE(f32,  float,         from_f_f32)
DEFINE_SSM_BURNIN_NOSAVE(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BURNIN_NOSAVE(f16,  __half,        from_f_f16)

// Templated SSM burnin WITH saves — full forward sequence for training/backward.
// h_saved and y_out in activation dtype; h_state + a_neg + D in f32.
#define DEFINE_SSM_BURNIN(SUFFIX, TY, FROM_F)                               \
extern "C" __global__ void ssm_burnin_forward_##SUFFIX(                     \
    float* h, TY* y_out, float* h_saved,                                    \
    const TY* delta_raw, TY* delta_saved, const TY* u,                      \
    const TY* B, const TY* C,                                               \
    const float* a_neg, const float* D,                                     \
    int batch, int T_len, int d_inner, int d_state                          \
) {                                                                         \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                        \
    int total = batch * d_inner;                                            \
    if (idx >= total) return;                                               \
    int b = idx / d_inner;                                                  \
    int d = idx % d_inner;                                                  \
    int h_base = (b * d_inner + d) * d_state;                               \
    float h_local[MAMBA_RS_STATE_CAP];                                                      \
    float a_local[MAMBA_RS_STATE_CAP];                                                      \
    if (d_state > MAMBA_RS_STATE_CAP) return;                                               \
    _Pragma("unroll")                                                       \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {         \
        h_local[n] = h[h_base + n];                                         \
        a_local[n] = a_neg[d * d_state + n];                                \
    }                                                                       \
    _Pragma("unroll")                                                       \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {         \
        int hs_idx = (b * (T_len + 1) + 0) * d_inner * d_state              \
                     + d * d_state + n;                                     \
        h_saved[hs_idx] = h_local[n];                                       \
    }                                                                       \
    for (int t = 0; t < T_len; t++) {                                       \
        int bt_di = (b * T_len + t) * d_inner + d;                          \
        int bt_ds = (b * T_len + t) * d_state;                              \
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
            int hs_idx = (b * (T_len + 1) + (t + 1)) * d_inner * d_state    \
                         + d * d_state + n;                                 \
            h_saved[hs_idx] = h_local[n];                                   \
        }                                                                   \
    }                                                                       \
    _Pragma("unroll")                                                       \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else           \
        h[h_base + n] = h_local[n];                                         \
}

DEFINE_SSM_BURNIN(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BURNIN(f16,  __half,        from_f_f16)

// ======================== BACKWARD ========================

// SSM backward with per-sample LOCAL gradient accumulation (no atomicAdd).
// Each thread (b, d) accumulates local d_D and d_a_log, writes to per-sample buffers.
// Separate reduction kernels sum across batch dimension afterward.
// a_neg cached in registers (Opt A). Uses exp2f (Opt B).
//
// Source: CPU reference: train/forward.rs phase B3
// FIX C2: no atomicAdd, local accumulation + reduction
extern "C" __global__ void ssm_backward_local(
    // Inputs (from forward, saved activations)
    const float* h_saved,    // [batch * (T+1) * d_inner * d_state] saved h BEFORE each step
    const float* delta_saved,// [batch * T * d_inner]
    const float* u_saved,    // [batch * T * d_inner]
    const float* B_saved,    // [batch * T * d_state]
    const float* C_saved,    // [batch * T * d_state]
    const float* a_neg,      // [d_inner * d_state]
    const float* D,          // [d_inner] skip connection weight
    // Incoming gradient
    const float* dy,         // [batch * T * d_inner]
    // Output gradients (per-sample, need reduction across batch)
    float* d_delta,          // [batch * T * d_inner] per-sample
    float* d_u,              // [batch * T * d_inner] per-sample (includes skip: += dy*D[d])
    float* d_B_local,        // [batch * T * d_inner * d_state] per-thread
    float* d_C_local,        // [batch * T * d_inner * d_state] per-thread
    float* d_D_local,        // [batch * d_inner] per-sample (sum across T)
    float* d_a_log_local,    // [batch * d_inner * d_state] per-sample (sum across T)
    // Dimensions
    int batch, int T, int d_inner, int d_state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_inner;
    if (idx >= total) return;

    int b = idx / d_inner;
    int d = idx % d_inner;

    // Per-thread accumulators for D and a_log gradients
    float local_d_D = 0.0f;

    // Opt A: cache a_neg in registers
    float a_local[MAMBA_RS_STATE_CAP];
    if (d_state > MAMBA_RS_STATE_CAP) return;
    #pragma unroll
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else
        a_local[n] = a_neg[d * d_state + n];

    // d_h carries gradient backward through time
    float d_h[MAMBA_RS_STATE_CAP];
    // Register accumulator for d_a_log: the old global `+=` into
    // d_a_log_local was ~T*d_state dependent global RMWs per thread in
    // the hottest loop; same adds in the same (t,n) order, written once.
    float d_a_acc[MAMBA_RS_STATE_CAP];
    // Register carry for the BPTT state: in the reverse-T walk,
    // h_curr(t) == h_prev(t+1) — carrying it halves h_saved reads.
    float h_carry[MAMBA_RS_STATE_CAP];
    #pragma unroll
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {
        d_h[n] = 0.0f;
        d_a_acc[n] = 0.0f;
        h_carry[n] = h_saved[(b * (T + 1) + T) * d_inner * d_state + (d * d_state + n)];
    }

    // Backward through time (reverse T)
    for (int t = T - 1; t >= 0; t--) {
        int bt_di = (b * T + t) * d_inner + d;
        int bt_ds = (b * T + t) * d_state;

        float dy_d = dy[bt_di];
        float delta_d = delta_saved[bt_di];
        float u_d = u_saved[bt_di];

        // d_D += dy * u (skip connection gradient)
        local_d_D += dy_d * u_d;

        // d_u from skip connection: dy * D[d]
        float d_u_val = dy_d * D[d];

        float d_delta_val = 0.0f;

        #pragma unroll
        for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {
            // h_curr = state AFTER step t (carried register: at t=T-1 it
            // was seeded from h_saved[T]; afterwards it is last round's
            // h_prev — identical value, one global load saved).
            float h_curr = h_carry[n];

            // Opt B: exp2f instead of expf
            float da = exp2f(delta_d * a_local[n] * LOG2E);

            // Gradient from output: d_h += dy * C
            d_h[n] += dy_d * C_saved[bt_ds + n];

            // d_C += dy * h_curr (per-thread: indexed by b,t,d,n)
            int btdn = ((b * T + t) * d_inner + d) * d_state + n;
            d_C_local[btdn] = dy_d * h_curr;

            // h_prev: state BEFORE step t
            int h_prev_idx = (b * (T + 1) + t) * d_inner * d_state + (d * d_state + n);
            float h_prev = h_saved[h_prev_idx];

            // d_delta += d_h * (a_dn * da * h_prev + u * B)
            d_delta_val += d_h[n] * (a_local[n] * da * h_prev + u_d * B_saved[bt_ds + n]);

            // d_u += d_h * delta * B
            d_u_val += d_h[n] * delta_d * B_saved[bt_ds + n];

            // d_B += d_h * delta * u (per-thread: indexed by b,t,d,n)
            d_B_local[btdn] = d_h[n] * delta_d * u_d;

            // d_a_log += d_h * da * delta * a_dn * h_prev (register acc)
            d_a_acc[n] += d_h[n] * da * delta_d * a_local[n] * h_prev;

            // Propagate d_h backward through time: d_h_prev = da * d_h
            d_h[n] = da * d_h[n];
            h_carry[n] = h_prev;
        }

        d_delta[bt_di] = d_delta_val;
        d_u[bt_di] = d_u_val;
    }

    // One store per element replaces T global RMWs; full-domain write,
    // so the per-layer zeroing of d_a_log_local is gone with it.
    #pragma unroll
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else
        d_a_log_local[(b * d_inner + d) * d_state + n] = d_a_acc[n];

    d_D_local[b * d_inner + d] = local_d_D;
}

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
//   - register dh, a_local (state-capacity sized), local_d_D → f32 (BPTT precision invariant)
//
// CONSTRAINT: d_state ≤ 64 (compile-time register array). All shipped
// state-spaces/mamba checkpoints use d_state=16.
//
// Math identical to f32 ssm_backward_local above; only IO dtype differs.
#define DEFINE_SSM_BACKWARD_LOCAL_BWD(SUFFIX, TY, FROM_F)                       \
extern "C" __global__ void ssm_backward_local_##SUFFIX(                         \
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                            \
    int total = batch * d_inner;                                                \
    if (idx >= total) return;                                                   \
    int b = idx / d_inner;                                                      \
    int d = idx % d_inner;                                                      \
    if (d_state > MAMBA_RS_STATE_CAP) return;                                                   \
                                                                                \
    float local_d_D = 0.0f;                                                     \
    float a_local[MAMBA_RS_STATE_CAP];                                                          \
    float d_h[MAMBA_RS_STATE_CAP];                                                              \
    float d_a_acc[MAMBA_RS_STATE_CAP];                                                          \
    float h_carry[MAMBA_RS_STATE_CAP];                                                          \
    _Pragma("unroll")                                                           \
    for (int n = 0; n < MAMBA_RS_STATE_CAP; n++) if (n >= d_state) break; else {             \
        a_local[n] = a_neg[d * d_state + n];                                    \
        d_h[n] = 0.0f;                                                          \
        d_a_acc[n] = 0.0f;                                                      \
        h_carry[n] = h_saved[(b * (T + 1) + T) * d_inner * d_state              \
                             + (d * d_state + n)];                              \
    }                                                                           \
                                                                                \
    for (int t = T - 1; t >= 0; t--) {                                          \
        int bt_di = (b * T + t) * d_inner + d;                                  \
        int bt_ds = (b * T + t) * d_state;                                      \
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
            int h_prev_idx = (b * (T + 1) + t)       * d_inner * d_state        \
                             + (d * d_state + n);                               \
            float h_curr = h_carry[n];                                          \
            float h_prev = h_saved[h_prev_idx];                                 \
            float B_n    = to_f(B_saved[bt_ds + n]);                            \
            float C_n    = to_f(C_saved[bt_ds + n]);                            \
            float da     = exp2f(delta_d * a_local[n] * LOG2E);                 \
                                                                                \
            d_h[n] += dy_d * C_n;                                               \
            int btdn = ((b * T + t) * d_inner + d) * d_state + n;               \
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

DEFINE_SSM_BACKWARD_LOCAL_BWD(f32,  float,         from_f_f32)
DEFINE_SSM_BACKWARD_LOCAL_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BACKWARD_LOCAL_BWD(f16,  __half,        from_f_f16)

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int total = batch * T * d_state;                                           \
    if (idx >= total) return;                                                  \
    int bt = idx / d_state;                                                    \
    int n = idx % d_state;                                                     \
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int total = batch * T * d_state;                                           \
    if (idx >= total) return;                                                  \
    int t = idx % T;                                                           \
    int rem = idx / T;                                                         \
    int n = rem % d_state;                                                     \
    int b = rem / d_state;                                                     \
    int out_idx = (b * T + t) * d_state + n;                                   \
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
        sum += d_D_local[b * d_inner + d];
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d_inner * d_state;
    if (idx >= total) return;
    float sum = 0.0f;
    for (int b = 0; b < batch; b++) {
        float acc = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            acc += partials[(b * n_chunks + c) * total + idx];
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d_inner * d_state;
    if (idx >= total) return;
    float sum = 0.0f;
    for (int b = 0; b < batch; b++) {
        sum += d_a_log_local[b * total + idx];
    }
    d_a_log_out[idx] += sum;
}
#line 1 "kernels/mamba_ssm_parallel.cu"
// Parallel prefix scan for Mamba-1 SSM recurrence.
//
// Faithful reimplementation of Tri Dao's selective_scan_fwd_kernel.cuh
// without CUB, PyTorch, or c10 dependencies -- pure NVRTC-compilable CUDA.
//
// Algorithm (identical to original):
//   1. Grid: (batch, d_inner) -- one block per (b, d) pair.
//   2. Block: NTHREADS threads, each owns NITEMS consecutive timesteps.
//      Chunk size = NTHREADS * NITEMS = 1024.
//   3. Outer loop over d_state (sequential, like original).
//   4. For each state index n:
//      a. Load delta, u, B for NITEMS timesteps -> compute (da, delta*u*B) pairs.
//      b. Thread-local sequential scan of NITEMS pairs.
//      c. Block-level inclusive scan via warp shuffle + shared memory raking.
//      d. Inter-chunk carry via smem_running_prefix (exactly like original).
//      e. Single-pass Y accumulation: y[t] += h[t] * C[t] during scan.
//   5. After all d_state iterations, y already contains the full output.
//
// Scan operator: (a1, b1) o (a0, b0) = (a1*a0, a1*b0 + b1)
// Encodes the linear recurrence: h_t = da_t * h_{t-1} + db_t
//
// The running prefix (run_a, run_b) satisfies:
//   h = run_a * h_init + run_b
// where h_init is the initial state h[h_base + n].
//
// Source: Gu & Dao (2023), "Mamba: Linear-Time Sequence Modeling"
//         selective_scan_fwd_kernel.cuh, selective_scan_common.h

// Typed-I/O prelude (to_f / from_f_* upcast/downcast helpers).
// Typed (bf16/f16) mixed-precision parallel scan forward. Following
// state-spaces/mamba's `scan_t = float2` invariant (all scan state in
// f32) and our BPTT precision discipline (h, h_saved,
// a_neg, D, smem_* remain f32). Only the activation I/O tensors
// (delta, u, B, C, y_out) are typed.

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

// Block config: 128 threads x 8 items = 1024 elements per chunk.
// exp(x) = exp2(x * LOG2E) -- we fold LOG2E into a_neg once per (d, n)
// so the inner loop uses a single exp2f() with no extra FMUL.
#define NTHREADS 128
#define NITEMS   8
#define CHUNK_SIZE (NTHREADS * NITEMS)
#define NWARPS   (NTHREADS / 32)
// Resident-block pin scales with the block size (128 -> 3 keeps the
// historical codegen envelope; 256 -> 2 keeps ~128 regs/thread).
#if NTHREADS >= 256
#define SCAN_MINB 2
#else
#define SCAN_MINB 3
#endif

// Must be >= actual d_state. Matches Tri Dao's MAX_DSTATE = 256.
#define MAX_DSTATE 256

// ============================================================================
// Eight consecutive B or C values of one (b, n) row, t0 .. t0+7, as f32.
// The rows are t-contiguous, so a thread's eight items are one 32-byte
// (f32) or 16-byte (half) span. When that span lies inside the row and is
// 16-byte aligned it comes in as vector loads, at 8-byte alignment as
// half-width vectors, otherwise element by element with the tail past T
// zero-filled. The values are the same either way; only the number of
// load instructions changes.
// ============================================================================
template <typename T>
__device__ __forceinline__ void load_row8(
    const T* __restrict__ row, int t0, int T_len, float out[NITEMS]
) {
    const T* p = row + t0;
    unsigned long long addr = (unsigned long long)p;
    bool whole = (t0 + NITEMS <= T_len);
    if (whole && (addr & 15ull) == 0) {
        constexpr int PER16 = 16 / (int)sizeof(T);
        #pragma unroll
        for (int k = 0; k < NITEMS / PER16; k++) {
            uint4 v = reinterpret_cast<const uint4*>(p)[k];
            const T* e = reinterpret_cast<const T*>(&v);
            #pragma unroll
            for (int j = 0; j < PER16; j++) out[k * PER16 + j] = to_f(e[j]);
        }
    } else if (whole && (addr & 7ull) == 0) {
        constexpr int PER8 = 8 / (int)sizeof(T);
        #pragma unroll
        for (int k = 0; k < NITEMS / PER8; k++) {
            uint2 v = reinterpret_cast<const uint2*>(p)[k];
            const T* e = reinterpret_cast<const T*>(&v);
            #pragma unroll
            for (int j = 0; j < PER8; j++) out[k * PER8 + j] = to_f(e[j]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < NITEMS; i++)
            out[i] = (t0 + i < T_len) ? to_f(p[i]) : 0.0f;
    }
}

// ============================================================================
// Warp-level inclusive scan of (a, b) pairs using warp shuffle.
// After return, lane k holds compose(pair_0, ..., pair_k) within its warp.
// ============================================================================
// BUG FIX: accept a mask parameter rather than hardcoding
// 0xffffffff. Step 3 of `block_inclusive_scan_ab` calls this with only
// NWARPS=4 active lanes out of the warp; `__shfl_up_sync(0xffffffff, ...)`
// is UB when mask members don't all execute → silent hang on Ada/sm_89.
// CUDA docs explicitly require the mask to describe the set of actively
// participating threads.
__device__ __forceinline__ void warp_inclusive_scan_ab(
    float &a, float &b, unsigned mask = 0xffffffff
) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float a_prev = __shfl_up_sync(mask, a, offset);
        float b_prev = __shfl_up_sync(mask, b, offset);
        if ((threadIdx.x & 31) >= (unsigned)offset) {
            b = a * b_prev + b;
            a = a * a_prev;
        }
    }
}

// ============================================================================
// Warp-level inclusive REVERSE scan of (a, b) pairs (Tri Dao
// `ThreadReverseScan` from `selective_scan/reverse_scan.cuh`).
//
// Forward scan composes left→right: lane k holds compose(p_0, ..., p_k).
// Reverse scan composes right→left: lane k holds compose(p_k, ..., p_31).
//
// Compose op (same as forward):
//   (a2, b2) ∘ (a1, b1) = (a2*a1, a2*b1 + b2)
//
// In the SSM bwd, this propagates dh_t backward in time:
//   p_t = (delta_A_next[t], dout[t]*B[t]*C[t])
//   reverse_scan_t = compose(p_t, p_{t+1}, ..., p_{T-1})
// so reverse_scan_t.b is dh_t. The "next-step" delta_A is what makes the
// gradient correctly multiply by future-step decay (Tri Dao trick).
// ============================================================================
// `active` = number of participating low lanes (must equal popcount(mask)
// for a contiguous low mask). The update guard must stop at `active`, not
// 32: __shfl_down_sync from a lane outside the mask returns an UNDEFINED
// value, and the old `lane + offset < 32` guard composed those undefined
// values into the scan whenever mask < full warp (the NWARPS-total scan in
// block_inclusive_reverse_scan_ab). The forward twin is immune because its
// shfl_up guard `lane >= offset` only ever reads lower (in-mask) lanes.
__device__ __forceinline__ void warp_inclusive_reverse_scan_ab(
    float &a, float &b, unsigned mask = 0xffffffff, int active = 32
) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float a_next = __shfl_down_sync(mask, a, offset);
        float b_next = __shfl_down_sync(mask, b, offset);
        if ((int)(threadIdx.x & 31) + offset < active) {
            // compose(self, next): (a*a_next, a*b_next + b) NO — careful:
            // reverse semantic: lane k accumulates (p_k ∘ p_{k+1} ∘ ...).
            // op (a2,b2)∘(a1,b1) = (a2*a1, a2*b1 + b2) with self=2nd arg.
            // So acc_k = self ∘ acc_{k+1} where acc_{k+1} arrives via shfl.
            b = a * b_next + b;
            a = a * a_next;
        }
    }
}

// ============================================================================
// Block-level inclusive REVERSE scan of (a, b) pairs.
// Mirror of `block_inclusive_scan_ab` walking right-to-left.
// ============================================================================
__device__ __forceinline__ void block_inclusive_reverse_scan_ab(
    float &a, float &b,
    float *smem_wa, float *smem_wb
) {
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    // Step 1: intra-warp inclusive reverse scan (lane 0 holds full warp tail)
    warp_inclusive_reverse_scan_ab(a, b);

    // Step 2: lane 0 of each warp stores its inclusive total (the full
    // composition of that warp from right-most lane back to lane 0).
    if (lane == 0) {
        smem_wa[warp_id] = a;
        smem_wb[warp_id] = b;
    }
    __syncthreads();

    // Step 3: first warp scans the NWARPS totals in REVERSE.
    // Partial mask AND matching `active` bound: only lanes < NWARPS hold
    // valid totals, so the compose guard must stop at NWARPS (see the
    // helper's comment — stopping at 32 composed undefined shuffle values).
    if (warp_id == 0 && lane < NWARPS) {
        float wa = smem_wa[lane];
        float wb = smem_wb[lane];
        warp_inclusive_reverse_scan_ab(wa, wb, (1u << NWARPS) - 1u, NWARPS);
        smem_wa[lane] = wa;
        smem_wb[lane] = wb;
    }
    __syncthreads();

    // Step 4: threads in warp < NWARPS-1 compose with the NEXT warp's postfix.
    if (warp_id < NWARPS - 1) {
        float na = smem_wa[warp_id + 1];
        float nb = smem_wb[warp_id + 1];
        b = a * nb + b;
        a = a * na;
    }
    // No __syncthreads here — caller syncs before next smem_wa/wb use.
}

// ============================================================================
// Block-level inclusive scan of (a, b) pairs.
// Two-level: warp scan -> inter-warp scan via shared memory -> compose.
// This is CUB's BLOCK_SCAN_WARP_SCANS algorithm.
// ============================================================================
__device__ __forceinline__ void block_inclusive_scan_ab(
    float &a, float &b,
    float *smem_wa, float *smem_wb
) {
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    // Step 1: intra-warp inclusive scan
    warp_inclusive_scan_ab(a, b);

    // Step 2: last lane of each warp stores its inclusive total
    if (lane == 31) {
        smem_wa[warp_id] = a;
        smem_wb[warp_id] = b;
    }
    __syncthreads();

    // Step 3: first warp scans the NWARPS totals. Only lanes 0..NWARPS-1
    // participate — mask must reflect that or the __shfl_up_sync calls
    // inside warp_inclusive_scan_ab deadlock on Ada/sm_89 (mask
    // 0xffffffff requires all 32 lanes to execute the same sync).
    if (warp_id == 0 && lane < NWARPS) {
        float wa = smem_wa[lane];
        float wb = smem_wb[lane];
        warp_inclusive_scan_ab(wa, wb, (1u << NWARPS) - 1u);
        smem_wa[lane] = wa;
        smem_wb[lane] = wb;
    }
    __syncthreads();

    // Step 4: threads in warp > 0 compose with previous warp's prefix
    if (warp_id > 0) {
        float pa = smem_wa[warp_id - 1];
        float pb = smem_wb[warp_id - 1];
        b = a * pb + b;
        a = a * pa;
    }
    // No __syncthreads here -- caller syncs before next smem_wa/wb use.
}

// ============================================================================
// Block-level inclusive scan of (a, b) pairs that also hands every thread
// its exclusive prefix without a third barrier. The inclusive values are
// exactly those of `block_inclusive_scan_ab`. A thread's exclusive prefix
// is the composed inclusive value of the thread before it: a lane takes it
// from the lane above by shuffle, and a warp's first lane rebuilds the
// previous warp's last-lane value from that warp's raw total (kept in the
// raw slots, since the scanned slots overwrite the totals) composed with
// the same prefix that lane composed with, in the same expression.
// ============================================================================
__device__ __forceinline__ void block_scan_ab_with_exclusive(
    float &a, float &b, float &ea, float &eb,
    float *smem_wa, float *smem_wb, float *smem_raw_a, float *smem_raw_b
) {
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    warp_inclusive_scan_ab(a, b);
    if (lane == 31) {
        smem_wa[warp_id] = a;
        smem_wb[warp_id] = b;
        smem_raw_a[warp_id] = a;
        smem_raw_b[warp_id] = b;
    }
    __syncthreads();

    if (warp_id == 0 && lane < NWARPS) {
        float wa = smem_wa[lane];
        float wb = smem_wb[lane];
        warp_inclusive_scan_ab(wa, wb, (1u << NWARPS) - 1u);
        smem_wa[lane] = wa;
        smem_wb[lane] = wb;
    }
    __syncthreads();

    if (warp_id > 0) {
        float pa = smem_wa[warp_id - 1];
        float pb = smem_wb[warp_id - 1];
        b = a * pb + b;
        a = a * pa;
    }
    // Every lane of the warp rebuilds the value its first lane needs, so
    // the branches below depend on the warp index only and never diverge;
    // the first lane then keeps it and the others keep the shuffle.
    float edge_a = 1.0f;
    float edge_b = 0.0f;
    if (warp_id > 0) {
        edge_a = smem_raw_a[warp_id - 1];
        edge_b = smem_raw_b[warp_id - 1];
        if (warp_id > 1) {
            float pa = smem_wa[warp_id - 2];
            float pb = smem_wb[warp_id - 2];
            edge_b = edge_a * pb + edge_b;
            edge_a = edge_a * pa;
        }
    }
    float up_a = __shfl_up_sync(0xffffffffu, a, 1);
    float up_b = __shfl_up_sync(0xffffffffu, b, 1);
    ea = (lane == 0) ? edge_a : up_a;
    eb = (lane == 0) ? edge_b : up_b;
    // No __syncthreads here -- caller syncs before the workspace is reused.
}

// ============================================================================
// Reverse mirror of `block_scan_ab_with_exclusive`: the inclusive values of
// `block_inclusive_reverse_scan_ab`, plus each thread's exclusive postfix
// (the composed value of the thread after it) from the lane below or, on
// a warp's last lane, rebuilt from the next warp's raw total.
// ============================================================================
__device__ __forceinline__ void block_reverse_scan_ab_with_exclusive(
    float &a, float &b, float &na, float &nb,
    float *smem_wa, float *smem_wb, float *smem_raw_a, float *smem_raw_b
) {
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    warp_inclusive_reverse_scan_ab(a, b);
    if (lane == 0) {
        smem_wa[warp_id] = a;
        smem_wb[warp_id] = b;
        smem_raw_a[warp_id] = a;
        smem_raw_b[warp_id] = b;
    }
    __syncthreads();

    if (warp_id == 0 && lane < NWARPS) {
        float wa = smem_wa[lane];
        float wb = smem_wb[lane];
        warp_inclusive_reverse_scan_ab(wa, wb, (1u << NWARPS) - 1u, NWARPS);
        smem_wa[lane] = wa;
        smem_wb[lane] = wb;
    }
    __syncthreads();

    if (warp_id < NWARPS - 1) {
        float pa = smem_wa[warp_id + 1];
        float pb = smem_wb[warp_id + 1];
        b = a * pb + b;
        a = a * pa;
    }
    float edge_a = 1.0f;
    float edge_b = 0.0f;
    if (warp_id < NWARPS - 1) {
        edge_a = smem_raw_a[warp_id + 1];
        edge_b = smem_raw_b[warp_id + 1];
        if (warp_id + 1 < NWARPS - 1) {
            float pa = smem_wa[warp_id + 2];
            float pb = smem_wb[warp_id + 2];
            edge_b = edge_a * pb + edge_b;
            edge_a = edge_a * pa;
        }
    }
    float down_a = __shfl_down_sync(0xffffffffu, a, 1);
    float down_b = __shfl_down_sync(0xffffffffu, b, 1);
    na = (lane == 31) ? edge_a : down_a;
    nb = (lane == 31) ? edge_b : down_b;
    // No __syncthreads here -- caller syncs before the workspace is reused.
}

// ============================================================================
// Shared memory layout of the backward kernels (in extern __shared__ float[]):
//
//   [0                       .. NWARPS)          = smem_wa      (block scan)
//   [NWARPS                  .. 2*NWARPS)        = smem_wb      (block scan)
//   [2*NWARPS                .. 2*NWARPS+MAX_DS) = smem_run_a   (inter-chunk carry)
//   [2*NWARPS+MAX_DS         .. 2*NWARPS+2*MAX)  = smem_run_b   (inter-chunk carry)
//   [2*NWARPS+2*MAX_DS       .. +NTHREADS)       = smem_exch_a  (exclusive prefix)
//   [2*NWARPS+2*MAX_DS+NTHR  .. +NTHREADS)       = smem_exch_b  (exclusive prefix)
//   [2*NWARPS+2*MAX_DS+2*NTHR .. +CHUNK_SIZE)    = smem_stage   (coalesced load staging)
//
// Total: 2*4 + 2*256 + 2*128 + 1024 = 1800 floats = 7200 bytes.
//
// The forward kernels pack a smaller layout at the runtime d_state: the two
// block-scan rows, the two carry rows, and one exchange slot per warp for
// the prefix hand-off (2*NWARPS + 2*d_state + 2*NWARPS floats); they load
// and store directly and keep no staging region. The launcher sizes it.
// ============================================================================
#define SMEM_WA_OFF        0
#define SMEM_WB_OFF        (NWARPS)
#define SMEM_RUN_A_OFF     (2 * NWARPS)
#define SMEM_RUN_B_OFF     (2 * NWARPS + MAX_DSTATE)
#define SMEM_EXCH_A_OFF    (2 * NWARPS + 2 * MAX_DSTATE)
#define SMEM_EXCH_B_OFF    (2 * NWARPS + 2 * MAX_DSTATE + NTHREADS)
#define SMEM_STAGE_OFF     (2 * NWARPS + 2 * MAX_DSTATE + 2 * NTHREADS)
#define SMEM_TOTAL_FLOATS  (2 * NWARPS + 2 * MAX_DSTATE + 2 * NTHREADS + CHUNK_SIZE)

// Extra smem offsets for the backward-pass reverse scan.
// Layout (appended after the forward layout):
//   SMEM_REV_WA/WB     = 2*NWARPS floats   (reverse warp-scan workspace)
//   SMEM_POST_A/B      = 2*MAX_DSTATE      (inter-chunk reverse-scan postfix)
//   SMEM_NEXT_A        = NTHREADS          (next-thread δA exchange buffer)
//   SMEM_DA_LOG_RED    = NTHREADS          (block-reduce of d_a_log per (n))
//   SMEM_CHUNK_FIRST_A = MAX_DSTATE        (per-n boundary from later chunk
//                                           used by earlier chunk's last
//                                           thread as pair.a = a_{t+1})
//
// Total bwd extra: 2*4 + 2*256 + 128 + 128 + 256 = 1032 floats = 4128 B
// added to the 7200 B fwd footprint → 11328 B per block (still < 48 KB so
// no cudaFuncSetAttribute needed at default MAX_DSTATE=256). Rust launch
// code (src/mamba_ssm/gpu/launch.rs::grid_parallel_scan_bwd) matches this.
#define SMEM_REV_WA_OFF        (SMEM_TOTAL_FLOATS)
#define SMEM_REV_WB_OFF        (SMEM_TOTAL_FLOATS + NWARPS)
#define SMEM_POST_A_OFF        (SMEM_TOTAL_FLOATS + 2 * NWARPS)
#define SMEM_POST_B_OFF        (SMEM_TOTAL_FLOATS + 2 * NWARPS + MAX_DSTATE)
#define SMEM_NEXT_A_OFF        (SMEM_TOTAL_FLOATS + 2 * NWARPS + 2 * MAX_DSTATE)
#define SMEM_DA_RED_OFF        (SMEM_TOTAL_FLOATS + 2 * NWARPS + 2 * MAX_DSTATE + NTHREADS)
// Boundary: stores the "first thread's first da" of THIS chunk so that the
// PREVIOUS (earlier-in-time) chunk's last thread can use it as its
// `pair.a = a_{t+1}` boundary when computing reverse-scan dh. Initialized
// to 1.0 (identity) for the very-last chunk in time.
#define SMEM_CHUNK_FIRST_A_OFF (SMEM_TOTAL_FLOATS + 2 * NWARPS + 2 * MAX_DSTATE + 2 * NTHREADS)
#define SMEM_BWD_FLOATS        (SMEM_TOTAL_FLOATS + 2 * NWARPS + 3 * MAX_DSTATE + 2 * NTHREADS)

// ============================================================================
// Forward: parallel prefix scan with activation saves (training path).
//
// Same interface as ssm_burnin_forward. Same saved activations format.
// Single-pass Y accumulation within the d_state loop.
//
// Grid: (batch, d_inner). Block: NTHREADS.
// Shared memory: the forward layout described above, sized by the launcher.
// ============================================================================
extern "C" __global__ __launch_bounds__(NTHREADS, SCAN_MINB) void ssm_parallel_scan_fwd(
    float* __restrict__ h,             // [batch * d_inner * d_state] SSM state (mutated)
    float* __restrict__ y_out,         // [batch * T * d_inner] output
    float* __restrict__ h_saved,       // [batch * (T+1) * d_inner * d_state] saved for backward
    // Pre-softplus dt: the kernel applies softplus inline (through the
    // exact store rounding the deleted softplus_copy pass produced) and
    // WRITES the post-softplus save the backward replays from - one
    // launch and one full read pass fewer per layer.
    const float* __restrict__ delta_raw, // [batch * T * d_inner]
    float* __restrict__ delta_saved,     // [batch * T * d_inner]
    const float* __restrict__ u,       // [batch * T * d_inner]
    const float* __restrict__ B,       // [batch * T * d_state]
    const float* __restrict__ C,       // [batch * T * d_state]
    const float* __restrict__ a_neg,   // [d_inner * d_state]
    const float* __restrict__ D,       // [d_inner]
    int batch, int T, int d_inner, int d_state,
    // Slim tape: [batch*d_inner*d_state*3*n_chunks] rows of
    // (run_a, run_b, h_entry) per chunk. No __restrict__: under slim
    // the launcher passes the SAME buffer for h_saved and run_tape and
    // the kernel touches exactly one of them per launch.
    float* run_tape,
    int slim_tape
) {
    int bid = blockIdx.x;
    int did = blockIdx.y;
    if (bid >= batch || did >= d_inner) return;
    if (d_state > MAX_DSTATE) return;

    extern __shared__ float smem[];
    float *smem_wa     = smem + SMEM_WA_OFF;
    float *smem_wb     = smem + SMEM_WB_OFF;
    /* Runtime d_state stride: run/exchange/stage regions pack at the actual
     * d_state instead of MAX_DSTATE (the launcher shrinks the allocation to
     * match). Region ORDER and contents are unchanged, so every cell holds
     * the same value as the padded layout - address-only. */
    float *smem_run_a  = smem + 2 * NWARPS;
    float *smem_run_b  = smem_run_a + d_state;
    float *smem_exch_a = smem_run_b + d_state;
    float *smem_exch_b = smem_exch_a + NWARPS;

    float D_d = D[did];
    int h_base = (bid * d_inner + did) * d_state;
    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Save initial SSM state (slim: tape row head — the chunk-0 prefix
    // is the identity and the chunk-0 entry state is h_0 itself).
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {
        if (slim_tape) {
            int row = ((bid * d_inner + did) * d_state + n) * 3 * n_chunks;
            run_tape[row + 0] = 1.0f;
            run_tape[row + 1] = 0.0f;
            run_tape[row + 2] = h[h_base + n];
        } else {
            /* T-major tape: [b][d][n][t+1] — lane stride over t is one
             * element, so warp stores/loads coalesce (the old [b][t][d][n]
             * layout put every lane in its own 32-byte sector). Layout is a
             * property of the PARALLEL route; the sequential kernels keep
             * the historical layout. */
            int hs_idx = ((bid * d_inner + did) * d_state + n) * (T + 1) + 0;
            h_saved[hs_idx] = h[h_base + n];
        }
    }

    // Initialize running prefix to identity (1, 0) for each state dimension
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {
        smem_run_a[n] = 1.0f;
        smem_run_b[n] = 0.0f;
    }
    __syncthreads();

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int chunk_start = chunk * CHUNK_SIZE;

        // Direct per-thread loads. In global memory adjacent t are d_inner
        // elements apart, so threads of one block (fixed did) never share
        // a sector whichever thread issues the load; the merging happens
        // across blocks, where neighbouring did hit adjacent addresses at
        // each t. Staging the chunk through shared memory only added
        // barriers.
        float delta_vals[NITEMS];
        float u_vals[NITEMS];
        float delta_u_vals[NITEMS];
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            int t = chunk_start + threadIdx.x * NITEMS + i;
            if (t < T) {
                float raw = delta_raw[(bid * T + t) * d_inner + did];
                float sp = (raw > 20.0f) ? raw
                                         : log1pf(exp2f(raw * 1.4426950408889634f));
                delta_saved[(bid * T + t) * d_inner + did] = sp;
                delta_vals[i] = sp;
            } else {
                delta_vals[i] = 0.0f;
            }
            u_vals[i] = (t < T) ? u[(bid * T + t) * d_inner + did] : 0.0f;
            delta_u_vals[i] = delta_vals[i] * u_vals[i];
        }

        // Initialize output accumulator: y = D * u
        float out_vals[NITEMS];
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            out_vals[i] = D_d * u_vals[i];
        }

        // Outer loop over d_state (sequential, like Tri Dao original)
        for (int n = 0; n < d_state; n++) {
            // Pre-multiply a_neg by LOG2E so we can use exp2f directly,
            // saving one FMUL per (t, d, n) triple.
            float a_dn = a_neg[did * d_state + n] * LOG2E;

            // Build (da, db) pairs for this state dimension. B is read
            // directly (barrier diet — see the delta/u note above).
            float thread_a[NITEMS];
            float thread_b[NITEMS];
            float b_row[NITEMS];
            load_row8(B + (bid * d_state + n) * T,
                      chunk_start + threadIdx.x * NITEMS, T, b_row);

            #pragma unroll
            for (int i = 0; i < NITEMS; i++) {
                int t = chunk_start + threadIdx.x * NITEMS + i;
                if (t < T) {
                    // a_dn already has LOG2E folded in, so exp2f gives exp(delta*a)
                    float da = exp2f(delta_vals[i] * a_dn);
                    float b_t = b_row[i];
                    thread_a[i] = da;
                    thread_b[i] = delta_u_vals[i] * b_t;
                    // No da saved: backward recomputes da from delta
                    // and a_neg (bandwidth win).
                } else {
                    thread_a[i] = 1.0f;  // identity
                    thread_b[i] = 0.0f;
                }
            }

            // Thread-local sequential scan of NITEMS (a, b) pairs
            #pragma unroll
            for (int i = 1; i < NITEMS; i++) {
                thread_b[i] = thread_a[i] * thread_b[i - 1] + thread_b[i];
                thread_a[i] = thread_a[i] * thread_a[i - 1];
            }

            // Block-level inclusive scan of per-thread totals
            float scan_a = thread_a[NITEMS - 1];
            float scan_b = thread_b[NITEMS - 1];

            __syncthreads();
            block_inclusive_scan_ab(scan_a, scan_b, smem_wa, smem_wb);

            // Exclusive prefix of thread t = inclusive value of thread t-1:
            // the lane above hands it down through a shuffle; a warp's first
            // lane takes it from the previous warp's last lane through a
            // per-warp slot; thread 0 takes the identity. The last slot is
            // the block total thread 0 folds into the running prefix. The
            // values are the very registers the full exchange used to copy.
            // Every warp reads the running prefix before the barrier, so
            // thread 0 may overwrite it right after.
            if ((threadIdx.x & 31) == 31) {
                smem_exch_a[threadIdx.x >> 5] = scan_a;
                smem_exch_b[threadIdx.x >> 5] = scan_b;
            }
            float run_a = smem_run_a[n];
            float run_b = smem_run_b[n];
            __syncthreads();
            float excl_a = __shfl_up_sync(0xffffffffu, scan_a, 1);
            float excl_b = __shfl_up_sync(0xffffffffu, scan_b, 1);
            if (threadIdx.x == 0) {
                excl_a = 1.0f;
                excl_b = 0.0f;
            } else if ((threadIdx.x & 31) == 0) {
                excl_a = smem_exch_a[(threadIdx.x >> 5) - 1];
                excl_b = smem_exch_b[(threadIdx.x >> 5) - 1];
            }
            // Slim tape: record this chunk's entry prefix (chunk 0's
            // identity row was written above). Single writer.
            if (slim_tape && chunk > 0 && threadIdx.x == 0) {
                int row =
                    ((bid * d_inner + did) * d_state + n) * 3 * n_chunks;
                run_tape[row + 3 * chunk + 0] = run_a;
                run_tape[row + 3 * chunk + 1] = run_b;
            }

            // Initial state for this (b, d, n) triple
            float h_0 = h[h_base + n];

            // Running prefix for the next chunk: new_run = block_total o old_run
            if (threadIdx.x == 0) {
                float block_a = smem_exch_a[NWARPS - 1];
                float block_b = smem_exch_b[NWARPS - 1];
                smem_run_a[n] = block_a * run_a;
                smem_run_b[n] = block_a * run_b + block_b;
            }

            // Compute h[t] for each element and accumulate y[t] += h[t] * C[t,n]
            float c_row[NITEMS];
            load_row8(C + (bid * d_state + n) * T,
                      chunk_start + threadIdx.x * NITEMS, T, c_row);
            #pragma unroll
            for (int i = 0; i < NITEMS; i++) {
                int t = chunk_start + threadIdx.x * NITEMS + i;
                if (t < T) {
                    // Compose thread-local prefix with exclusive block prefix:
                    //   (comp_a, comp_b) = (thread_a[i], thread_b[i]) o (excl_a, excl_b)
                    float comp_a = thread_a[i] * excl_a;
                    float comp_b = thread_a[i] * excl_b + thread_b[i];

                    // Compose with inter-chunk running prefix:
                    //   (final_a, final_b) = (comp_a, comp_b) o (run_a, run_b)
                    float final_a = comp_a * run_a;
                    float final_b = comp_a * run_b + comp_b;

                    // h[t] = final_a * h_init + final_b
                    float h_t = final_a * h_0 + final_b;

                    // Save h for backward: the full tape stores every
                    // step; slim stores only the NEXT chunk's entry state
                    // (bit-exactly the value the backward's h_prev
                    // boundary read used to load from h_saved).
                    if (slim_tape) {
                        if ((t + 1) % CHUNK_SIZE == 0 && t + 1 < T) {
                            int row = ((bid * d_inner + did) * d_state + n)
                                * 3 * n_chunks;
                            run_tape[row + 3 * (chunk + 1) + 2] = h_t;
                        }
                    } else {
                        int hs_idx = ((bid * d_inner + did) * d_state + n)
                            * (T + 1) + (t + 1);
                        h_saved[hs_idx] = h_t;
                    }

                    // Single-pass Y accumulation (C read directly)
                    float c_t = c_row[i];
                    out_vals[i] += h_t * c_t;
                }
            }
        } // end d_state loop

        // Each thread stores its own items. Within a block did is fixed,
        // so neighbouring t are d_inner apart and every lane lands in its
        // own sector whichever thread issues the store; staging the chunk
        // through shared memory to restripe it bought nothing and cost
        // two barriers per chunk.
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            int t = chunk_start + threadIdx.x * NITEMS + i;
            if (t < T) {
                y_out[(bid * T + t) * d_inner + did] = out_vals[i];
            }
        }
    } // end chunk loop
    // Thread 0 wrote the last chunk's running prefix after the scan
    // barrier; every thread reads its state dimension below.
    __syncthreads();

    // Write final SSM state: h[n] = run_a[n] * h_init[n] + run_b[n]
    // Parallelized across threads.
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {
        float h_0 = h[h_base + n];
        h[h_base + n] = smem_run_a[n] * h_0 + smem_run_b[n];
    }
}

// ============================================================================
// Forward without saves (target network -- no backward needed).
//
// Same interface as ssm_burnin_forward_nosave.
// Same parallel scan algorithm, skips the h_saved writes.
// ============================================================================
extern "C" __global__ __launch_bounds__(NTHREADS, SCAN_MINB) void ssm_parallel_scan_fwd_nosave(
    float* __restrict__ h,             // [batch * d_inner * d_state] SSM state (mutated)
    float* __restrict__ y_out,         // [batch * T * d_inner] output
    const float* __restrict__ delta,   // [batch * T * d_inner]
    const float* __restrict__ u,       // [batch * T * d_inner]
    const float* __restrict__ B,       // [batch * T * d_state]
    const float* __restrict__ C,       // [batch * T * d_state]
    const float* __restrict__ a_neg,   // [d_inner * d_state]
    const float* __restrict__ D,       // [d_inner]
    const float* __restrict__ proj_gate, // in_proj output when gating fuses
    int gate_stride,                   // proj row stride; 0 = plain y store
    int batch, int T, int d_inner, int d_state
) {
    int bid = blockIdx.x;
    int did = blockIdx.y;
    if (bid >= batch || did >= d_inner) return;
    if (d_state > MAX_DSTATE) return;

    extern __shared__ float smem[];
    float *smem_wa     = smem + SMEM_WA_OFF;
    float *smem_wb     = smem + SMEM_WB_OFF;
    /* Runtime d_state stride: run/exchange/stage regions pack at the actual
     * d_state instead of MAX_DSTATE (the launcher shrinks the allocation to
     * match). Region ORDER and contents are unchanged, so every cell holds
     * the same value as the padded layout - address-only. */
    float *smem_run_a  = smem + 2 * NWARPS;
    float *smem_run_b  = smem_run_a + d_state;
    float *smem_exch_a = smem_run_b + d_state;
    float *smem_exch_b = smem_exch_a + NWARPS;

    float D_d = D[did];
    int h_base = (bid * d_inner + did) * d_state;

    // Initialize running prefix to identity
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {
        smem_run_a[n] = 1.0f;
        smem_run_b[n] = 0.0f;
    }
    __syncthreads();

    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int chunk_start = chunk * CHUNK_SIZE;

        // The shared-memory staging of delta, u, B and C was value-neutral
        // (the same elements in the same order) and its lane stride
        // exceeded the 32-byte sector either way, so direct loads drop the
        // staging barriers with no arithmetic change, as in the saving
        // twin. y keeps its staging, the one genuinely coalescing store.
        float delta_vals[NITEMS];
        float u_vals[NITEMS];
        float delta_u_vals[NITEMS];
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            int t = chunk_start + threadIdx.x * NITEMS + i;
            delta_vals[i] = (t < T) ? delta[(bid * T + t) * d_inner + did] : 0.0f;
            u_vals[i] = (t < T) ? u[(bid * T + t) * d_inner + did] : 0.0f;
            delta_u_vals[i] = delta_vals[i] * u_vals[i];
        }

        float out_vals[NITEMS];
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            out_vals[i] = D_d * u_vals[i];
        }

        for (int n = 0; n < d_state; n++) {
            // Pre-multiply a_neg by LOG2E so we can use exp2f directly,
            // saving one FMUL per (t, d, n) triple.
            float a_dn = a_neg[did * d_state + n] * LOG2E;

            float thread_a[NITEMS];
            float thread_b[NITEMS];
            float b_row[NITEMS];
            load_row8(B + (bid * d_state + n) * T,
                      chunk_start + threadIdx.x * NITEMS, T, b_row);

            #pragma unroll
            for (int i = 0; i < NITEMS; i++) {
                int t = chunk_start + threadIdx.x * NITEMS + i;
                if (t < T) {
                    // a_dn already has LOG2E folded in, so exp2f gives exp(delta*a)
                    float da = exp2f(delta_vals[i] * a_dn);
                    float b_t = b_row[i];
                    thread_a[i] = da;
                    thread_b[i] = delta_u_vals[i] * b_t;
                } else {
                    thread_a[i] = 1.0f;
                    thread_b[i] = 0.0f;
                }
            }

            #pragma unroll
            for (int i = 1; i < NITEMS; i++) {
                thread_b[i] = thread_a[i] * thread_b[i - 1] + thread_b[i];
                thread_a[i] = thread_a[i] * thread_a[i - 1];
            }

            float scan_a = thread_a[NITEMS - 1];
            float scan_b = thread_b[NITEMS - 1];

            __syncthreads();
            block_inclusive_scan_ab(scan_a, scan_b, smem_wa, smem_wb);

            // Exclusive prefix through the lane above and a per-warp slot,
            // as in the saving kernel.
            if ((threadIdx.x & 31) == 31) {
                smem_exch_a[threadIdx.x >> 5] = scan_a;
                smem_exch_b[threadIdx.x >> 5] = scan_b;
            }
            float run_a = smem_run_a[n];
            float run_b = smem_run_b[n];
            __syncthreads();
            float excl_a = __shfl_up_sync(0xffffffffu, scan_a, 1);
            float excl_b = __shfl_up_sync(0xffffffffu, scan_b, 1);
            if (threadIdx.x == 0) {
                excl_a = 1.0f;
                excl_b = 0.0f;
            } else if ((threadIdx.x & 31) == 0) {
                excl_a = smem_exch_a[(threadIdx.x >> 5) - 1];
                excl_b = smem_exch_b[(threadIdx.x >> 5) - 1];
            }
            float h_0 = h[h_base + n];

            if (threadIdx.x == 0) {
                float block_a = smem_exch_a[NWARPS - 1];
                float block_b = smem_exch_b[NWARPS - 1];
                smem_run_a[n] = block_a * run_a;
                smem_run_b[n] = block_a * run_b + block_b;
            }

            float c_row[NITEMS];
            load_row8(C + (bid * d_state + n) * T,
                      chunk_start + threadIdx.x * NITEMS, T, c_row);
            #pragma unroll
            for (int i = 0; i < NITEMS; i++) {
                int t = chunk_start + threadIdx.x * NITEMS + i;
                if (t < T) {
                    float comp_a = thread_a[i] * excl_a;
                    float comp_b = thread_a[i] * excl_b + thread_b[i];
                    float final_a = comp_a * run_a;
                    float final_b = comp_a * run_b + comp_b;
                    float h_t = final_a * h_0 + final_b;

                    float c_t = c_row[i];
                    out_vals[i] += h_t * c_t;
                }
            }
        } // end d_state loop

        // Direct per-thread stores, as in the saving kernel.
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            int t = chunk_start + threadIdx.x * NITEMS + i;
            if (t < T) {
                float yv = out_vals[i];
                if (gate_stride > 0) {
                    // Fused gating: the same one-rounding product the
                    // separate elementwise mul performed, on the same
                    // SiLU formula split_gate_silu used - bit-identical
                    // to the three-kernel chain it replaces.
                    float g = proj_gate[(bid * T + t) * gate_stride
                                        + d_inner + did];
                    yv *= g / (1.0f + exp2f(-g * 1.4426950408889634f));
                }
                y_out[(bid * T + t) * d_inner + did] = yv;
            }
        }
    } // end chunk loop
    // Thread 0 wrote the last chunk's running prefix after the scan
    // barrier; every thread reads its state dimension below.
    __syncthreads();

    // Write final SSM state (parallelized across threads)
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {
        float h_0 = h[h_base + n];
        h[h_base + n] = smem_run_a[n] * h_0 + smem_run_b[n];
    }
}

// ============================================================================
// Typed variants (bf16/f16) — the mixed-precision training tier.
//
// Follow `state-spaces/mamba`'s `scan_t = float2` discipline: all scan
// state + running prefix + block scan + registers stay f32. Only the
// activation I/O tensors (delta, u, B, C, y_out) become typed. BPTT
// state (`h`, `h_saved`), model parameters (`a_neg`, `D`),
// and ALL `smem_*` remain f32.
// ============================================================================

#define DEFINE_SSM_PARALLEL_SCAN_FWD(SUFFIX, T_ACT, FROM_F)                   \
extern "C" __global__ __launch_bounds__(NTHREADS, SCAN_MINB) void                          \
ssm_parallel_scan_fwd_##SUFFIX(                                               \
    float* __restrict__ h,                                                    \
    T_ACT* __restrict__ y_out,                                                \
    float* __restrict__ h_saved,                                              \
    const T_ACT* __restrict__ delta_raw,                                      \
    T_ACT* __restrict__ delta_saved,                                          \
    const T_ACT* __restrict__ u,                                              \
    const T_ACT* __restrict__ B,                                              \
    const T_ACT* __restrict__ C,                                              \
    const float* __restrict__ a_neg,                                          \
    const float* __restrict__ D,                                              \
    int batch, int T, int d_inner, int d_state,                               \
    /* Slim tape (no __restrict__: aliases h_saved under slim) */             \
    float* run_tape,                                                          \
    int slim_tape                                                             \
) {                                                                           \
    int bid = blockIdx.x;                                                     \
    int did = blockIdx.y;                                                     \
    if (bid >= batch || did >= d_inner) return;                               \
    if (d_state > MAX_DSTATE) return;                                         \
    extern __shared__ float smem[];                                           \
    float *smem_wa     = smem + SMEM_WA_OFF;                                  \
    float *smem_wb     = smem + SMEM_WB_OFF;                                  \
    /* Runtime d_state stride - see the f32 twin. */                          \
    float *smem_run_a  = smem + 2 * NWARPS;                                   \
    float *smem_run_b  = smem_run_a + d_state;                                \
    float *smem_exch_a = smem_run_b + d_state;                                \
    float *smem_exch_b = smem_exch_a + NWARPS;                                \
    float D_d = D[did];                                                       \
    int h_base = (bid * d_inner + did) * d_state;                             \
    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;                         \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        if (slim_tape) {                                                      \
            int row = ((bid * d_inner + did) * d_state + n) * 3 * n_chunks;   \
            run_tape[row + 0] = 1.0f;                                         \
            run_tape[row + 1] = 0.0f;                                         \
            run_tape[row + 2] = h[h_base + n];                                \
        } else {                                                              \
            /* T-major tape (see the plain fwd note). */                      \
            int hs_idx =                                                      \
                ((bid * d_inner + did) * d_state + n) * (T + 1) + 0;          \
            h_saved[hs_idx] = h[h_base + n];                                  \
        }                                                                     \
    }                                                                         \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        smem_run_a[n] = 1.0f;                                                 \
        smem_run_b[n] = 0.0f;                                                 \
    }                                                                         \
    __syncthreads();                                                          \
    for (int chunk = 0; chunk < n_chunks; chunk++) {                          \
        int chunk_start = chunk * CHUNK_SIZE;                                 \
        /* Barrier diet: the smem staging round trips for delta/u/B/C are     \
         * value-neutral (same elements, same to_f) and their lane stride     \
         * exceeded the 32-byte sector either way - direct loads drop the     \
         * staging barriers with zero arithmetic change. y keeps its          \
         * staging (the one genuinely coalescing store). */                   \
        float delta_vals[NITEMS];                                             \
        float u_vals[NITEMS];                                                 \
        float delta_u_vals[NITEMS];                                           \
        _Pragma("unroll")                                                     \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            if (t < T) {                                                      \
                /* Inline softplus through the exact store rounding the    \
                   deleted copy pass produced; the scan consumes the       \
                   round-tripped value, bit-equal to reading the save. */  \
                float raw = to_f(delta_raw[(bid * T + t) * d_inner + did]); \
                float sp = (raw > 20.0f)                                    \
                    ? raw                                                   \
                    : log1pf(exp2f(raw * 1.4426950408889634f));             \
                T_ACT spt = FROM_F(sp);                                     \
                delta_saved[(bid * T + t) * d_inner + did] = spt;           \
                delta_vals[i] = to_f(spt);                                  \
            } else {                                                        \
                delta_vals[i] = 0.0f;                                       \
            }                                                               \
            u_vals[i] = (t < T) ? to_f(u[(bid * T + t) * d_inner + did])      \
                                : 0.0f;                                       \
            delta_u_vals[i] = delta_vals[i] * u_vals[i];                      \
        }                                                                     \
        float out_vals[NITEMS];                                               \
                                                             \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            out_vals[i] = D_d * u_vals[i];                                    \
        }                                                                     \
        for (int n = 0; n < d_state; n++) {                                   \
            float a_dn = a_neg[did * d_state + n] * LOG2E;                    \
            float thread_a[NITEMS];                                           \
            float thread_b[NITEMS];                                           \
            float b_row[NITEMS];                                              \
            load_row8(B + (bid * d_state + n) * T,                            \
                      chunk_start + threadIdx.x * NITEMS, T, b_row);          \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    float da = exp2f(delta_vals[i] * a_dn);                   \
                    float b_t = b_row[i];                                     \
                    thread_a[i] = da;                                         \
                    thread_b[i] = delta_u_vals[i] * b_t;                      \
                } else {                                                      \
                    thread_a[i] = 1.0f;                                       \
                    thread_b[i] = 0.0f;                                       \
                }                                                             \
            }                                                                 \
                                                             \
            _Pragma("unroll")                                                  \
            for (int i = 1; i < NITEMS; i++) {                                \
                thread_b[i] = thread_a[i] * thread_b[i - 1] + thread_b[i];    \
                thread_a[i] = thread_a[i] * thread_a[i - 1];                  \
            }                                                                 \
            float scan_a = thread_a[NITEMS - 1];                              \
            float scan_b = thread_b[NITEMS - 1];                              \
            __syncthreads();                                                  \
            block_inclusive_scan_ab(scan_a, scan_b, smem_wa, smem_wb);        \
            /* Exclusive prefix through the lane above and a per-warp slot,  \
               as in the f32 kernel; the last slot is the block total. */     \
            if ((threadIdx.x & 31) == 31) {                                   \
                smem_exch_a[threadIdx.x >> 5] = scan_a;                       \
                smem_exch_b[threadIdx.x >> 5] = scan_b;                       \
            }                                                                 \
            float run_a = smem_run_a[n];                                      \
            float run_b = smem_run_b[n];                                      \
            __syncthreads();                                                  \
            float excl_a = __shfl_up_sync(0xffffffffu, scan_a, 1);            \
            float excl_b = __shfl_up_sync(0xffffffffu, scan_b, 1);            \
            if (threadIdx.x == 0) {                                           \
                excl_a = 1.0f;                                                \
                excl_b = 0.0f;                                                \
            } else if ((threadIdx.x & 31) == 0) {                             \
                excl_a = smem_exch_a[(threadIdx.x >> 5) - 1];                 \
                excl_b = smem_exch_b[(threadIdx.x >> 5) - 1];                 \
            }                                                                 \
            /* Slim tape: chunk-entry prefix (single writer). */              \
            if (slim_tape && chunk > 0 && threadIdx.x == 0) {                 \
                int row =                                                     \
                    ((bid * d_inner + did) * d_state + n) * 3 * n_chunks;     \
                run_tape[row + 3 * chunk + 0] = run_a;                        \
                run_tape[row + 3 * chunk + 1] = run_b;                        \
            }                                                                 \
            float h_0 = h[h_base + n];                                        \
            if (threadIdx.x == 0) {                                           \
                float block_a = smem_exch_a[NWARPS - 1];                      \
                float block_b = smem_exch_b[NWARPS - 1];                      \
                smem_run_a[n] = block_a * run_a;                              \
                smem_run_b[n] = block_a * run_b + block_b;                    \
            }                                                                 \
            float c_row[NITEMS];                                              \
            load_row8(C + (bid * d_state + n) * T,                            \
                      chunk_start + threadIdx.x * NITEMS, T, c_row);          \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    float comp_a = thread_a[i] * excl_a;                      \
                    float comp_b = thread_a[i] * excl_b + thread_b[i];        \
                    float final_a = comp_a * run_a;                           \
                    float final_b = comp_a * run_b + comp_b;                  \
                    float h_t = final_a * h_0 + final_b;                      \
                    if (slim_tape) {                                          \
                        if ((t + 1) % CHUNK_SIZE == 0 && t + 1 < T) {         \
                            int row =                                         \
                                ((bid * d_inner + did) * d_state + n)         \
                                * 3 * n_chunks;                               \
                            run_tape[row + 3 * (chunk + 1) + 2] = h_t;        \
                        }                                                     \
                    } else {                                                  \
                        int hs_idx = ((bid * d_inner + did) * d_state + n)    \
                                     * (T + 1) + (t + 1);                     \
                        h_saved[hs_idx] = h_t;                                \
                    }                                                         \
                    float c_t = c_row[i];                                     \
                    out_vals[i] += h_t * c_t;                                 \
                }                                                             \
            }                                                                 \
        }                                                                     \
                                                             \
        /* Direct per-thread stores, as in the f32 kernel. */                 \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            if (t < T) {                                                      \
                y_out[(bid * T + t) * d_inner + did] = FROM_F(out_vals[i]);   \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    /* The last running-prefix write by thread 0 precedes the reads below. */ \
    __syncthreads();                                                          \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        float h_0 = h[h_base + n];                                            \
        h[h_base + n] = smem_run_a[n] * h_0 + smem_run_b[n];                  \
    }                                                                         \
}

DEFINE_SSM_PARALLEL_SCAN_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_PARALLEL_SCAN_FWD(f16,  __half,        from_f_f16)

#define DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(SUFFIX, T_ACT, FROM_F)            \
extern "C" __global__ __launch_bounds__(NTHREADS, SCAN_MINB) void                          \
ssm_parallel_scan_fwd_nosave_##SUFFIX(                                        \
    float* __restrict__ h,                                                    \
    T_ACT* __restrict__ y_out,                                                \
    const T_ACT* __restrict__ delta,                                          \
    const T_ACT* __restrict__ u,                                              \
    const T_ACT* __restrict__ B,                                              \
    const T_ACT* __restrict__ C,                                              \
    const float* __restrict__ a_neg,                                          \
    const float* __restrict__ D,                                              \
    const T_ACT* __restrict__ proj_gate,                                      \
    int gate_stride,                                                          \
    int batch, int T, int d_inner, int d_state                                \
) {                                                                           \
    int bid = blockIdx.x;                                                     \
    int did = blockIdx.y;                                                     \
    if (bid >= batch || did >= d_inner) return;                               \
    if (d_state > MAX_DSTATE) return;                                         \
    extern __shared__ float smem[];                                           \
    float *smem_wa     = smem + SMEM_WA_OFF;                                  \
    float *smem_wb     = smem + SMEM_WB_OFF;                                  \
    /* Runtime d_state stride - see the f32 twin. */                          \
    float *smem_run_a  = smem + 2 * NWARPS;                                   \
    float *smem_run_b  = smem_run_a + d_state;                                \
    float *smem_exch_a = smem_run_b + d_state;                                \
    float *smem_exch_b = smem_exch_a + NWARPS;                                \
    float D_d = D[did];                                                       \
    int h_base = (bid * d_inner + did) * d_state;                             \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        smem_run_a[n] = 1.0f;                                                 \
        smem_run_b[n] = 0.0f;                                                 \
    }                                                                         \
    __syncthreads();                                                          \
    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;                         \
    for (int chunk = 0; chunk < n_chunks; chunk++) {                          \
        int chunk_start = chunk * CHUNK_SIZE;                                 \
        /* Direct loads instead of the staging round trips: the same      \
         * elements through the same to_f, so the values do not change,   \
         * and the staging barriers go with them (as in the saving twin). \
         * y keeps its staging, the one genuinely coalescing store. */     \
        float delta_vals[NITEMS];                                             \
        float u_vals[NITEMS];                                                 \
        float delta_u_vals[NITEMS];                                           \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            delta_vals[i] = (t < T)                                           \
                ? to_f(delta[(bid * T + t) * d_inner + did]) : 0.0f;          \
            u_vals[i] = (t < T) ? to_f(u[(bid * T + t) * d_inner + did])      \
                                : 0.0f;                                       \
            delta_u_vals[i] = delta_vals[i] * u_vals[i];                      \
        }                                                                     \
        float out_vals[NITEMS];                                               \
                                                             \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            out_vals[i] = D_d * u_vals[i];                                    \
        }                                                                     \
        for (int n = 0; n < d_state; n++) {                                   \
            float a_dn = a_neg[did * d_state + n] * LOG2E;                    \
            float thread_a[NITEMS];                                           \
            float thread_b[NITEMS];                                           \
            float b_row[NITEMS];                                              \
            load_row8(B + (bid * d_state + n) * T,                            \
                      chunk_start + threadIdx.x * NITEMS, T, b_row);          \
                                                             \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    float da = exp2f(delta_vals[i] * a_dn);                   \
                    float b_t = b_row[i];                                     \
                    thread_a[i] = da;                                         \
                    thread_b[i] = delta_u_vals[i] * b_t;                      \
                } else {                                                      \
                    thread_a[i] = 1.0f;                                       \
                    thread_b[i] = 0.0f;                                       \
                }                                                             \
            }                                                                 \
                                                             \
            _Pragma("unroll")                                                  \
            for (int i = 1; i < NITEMS; i++) {                                \
                thread_b[i] = thread_a[i] * thread_b[i - 1] + thread_b[i];    \
                thread_a[i] = thread_a[i] * thread_a[i - 1];                  \
            }                                                                 \
            float scan_a = thread_a[NITEMS - 1];                              \
            float scan_b = thread_b[NITEMS - 1];                              \
            __syncthreads();                                                  \
            block_inclusive_scan_ab(scan_a, scan_b, smem_wa, smem_wb);        \
            /* Exclusive prefix through the lane above and a per-warp slot,  \
               as in the f32 kernel; the last slot is the block total. */     \
            if ((threadIdx.x & 31) == 31) {                                   \
                smem_exch_a[threadIdx.x >> 5] = scan_a;                       \
                smem_exch_b[threadIdx.x >> 5] = scan_b;                       \
            }                                                                 \
            float run_a = smem_run_a[n];                                      \
            float run_b = smem_run_b[n];                                      \
            __syncthreads();                                                  \
            float excl_a = __shfl_up_sync(0xffffffffu, scan_a, 1);            \
            float excl_b = __shfl_up_sync(0xffffffffu, scan_b, 1);            \
            if (threadIdx.x == 0) {                                           \
                excl_a = 1.0f;                                                \
                excl_b = 0.0f;                                                \
            } else if ((threadIdx.x & 31) == 0) {                             \
                excl_a = smem_exch_a[(threadIdx.x >> 5) - 1];                 \
                excl_b = smem_exch_b[(threadIdx.x >> 5) - 1];                 \
            }                                                                 \
            float h_0 = h[h_base + n];                                        \
            if (threadIdx.x == 0) {                                           \
                float block_a = smem_exch_a[NWARPS - 1];                      \
                float block_b = smem_exch_b[NWARPS - 1];                      \
                smem_run_a[n] = block_a * run_a;                              \
                smem_run_b[n] = block_a * run_b + block_b;                    \
            }                                                                 \
            float c_row[NITEMS];                                              \
            load_row8(C + (bid * d_state + n) * T,                            \
                      chunk_start + threadIdx.x * NITEMS, T, c_row);          \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    float comp_a = thread_a[i] * excl_a;                      \
                    float comp_b = thread_a[i] * excl_b + thread_b[i];        \
                    float final_a = comp_a * run_a;                           \
                    float final_b = comp_a * run_b + comp_b;                  \
                    float h_t = final_a * h_0 + final_b;                      \
                    float c_t = c_row[i];                                     \
                    out_vals[i] += h_t * c_t;                                 \
                }                                                             \
            }                                                                 \
        }                                                                     \
                                                             \
        /* Direct per-thread stores, as in the f32 kernel. */                 \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            if (t < T) {                                                      \
                T_ACT ty = FROM_F(out_vals[i]);                               \
                if (gate_stride > 0) {                                        \
                    /* Round-trip emulation of the replaced chain: the     \
                     * baseline stored y typed, stored SiLU(gate) typed,   \
                     * then multiplied the reloaded values with one final  \
                     * rounding - reproduce each rounding in place.     */ \
                    float g = to_f(proj_gate[(bid * T + t) * gate_stride    \
                                             + d_inner + did]);              \
                    T_ACT tg = FROM_F(                                       \
                        g / (1.0f + exp2f(-g * 1.4426950408889634f)));       \
                    ty = FROM_F(to_f(ty) * to_f(tg));                        \
                }                                                             \
                y_out[(bid * T + t) * d_inner + did] = ty;                    \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    /* The last running-prefix write by thread 0 precedes the reads below. */ \
    __syncthreads();                                                          \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        float h_0 = h[h_base + n];                                            \
        h[h_base + n] = smem_run_a[n] * h_0 + smem_run_b[n];                  \
    }                                                                         \
}

DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(f16,  __half,        from_f_f16)

// ============================================================================
// Parallel selective-scan BACKWARD pass.
//
// Mirrors state-spaces/mamba `selective_scan_bwd_kernel.cuh`.
// Grid: (batch, d_inner). Block: NTHREADS=128 (1 block per (b, di)).
//
// Algorithm (per (b, di)):
//   For each chunk in REVERSE (n_chunks-1 → 0):
//     Coalesced typed load delta/u/dy → smem → registers (per-thread NITEMS)
//     For each n in [0, d_state):  (sequential outer loop)
//       Coalesced typed load B[*, n], C[*, n] → smem → registers
//       Per-i: da[i] = exp2f(δ·a_neg·LOG2E),  d_local[i] = dy[i]·c[i]
//       Build reverse-scan pair_t = (a_{t+1}, d_local[t]):
//         intra-thread: pair[i].a = da[i+1] for i < NITEMS-1
//         inter-thread: pair[NITEMS-1].a = next thread's da[0] (smem exch)
//         inter-chunk:  last thread's pair.a = postfix-saved next-chunk a
//         globally last: pair.a = 1.0 (no future)
//       Per-thread compose NITEMS pairs → block_inclusive_REVERSE_scan_ab
//       Compose with running_postfix → dh[i] for each timestep
//       Per-i outputs (typed acts loaded from h_saved, b_t, etc.):
//         d_C_local[btdn] = dy * h_saved[t+1]    (typed, store FROM_F)
//         d_B_local[btdn] = dh * δ * u           (typed)
//         d_delta_acc[i] += dh * (a·da·h_prev + u·b)  (register f32)
//         d_u_acc[i]     += dh * δ · b                  (register f32)
//         d_a_per_thread += dh · da · δ · a · h_prev    (register f32)
//       Block-reduce d_a_per_thread → thread 0 += d_a_log_local[bid·di·ds + did·ds + n]
//       Update running_postfix via block_inclusive_reverse_scan_ab tail
//     Store d_delta_acc, d_u_acc to typed HBM (smem coalesced + downcast)
//   Final: d_D_local[bid·d_inner + did] = local_d_D
//
// All scan state, h_saved, registers stay f32 (BPTT scan_t = float2 invariant).
// Outputs follow the existing _local convention so the existing reduction
// kernels (reduce_d_B, reduce_d_C, reduce_d_D, reduce_d_a_log) work unchanged.
// ============================================================================

#define DEFINE_SSM_PARALLEL_SCAN_BWD(SUFFIX, T_ACT, FROM_F)                   \
extern "C" __global__ __launch_bounds__(NTHREADS, SCAN_MINB) void                          \
ssm_parallel_scan_bwd_##SUFFIX(                                               \
    const float* __restrict__ h_saved,    /* [B*(T+1)*di*ds] */               \
    const T_ACT* __restrict__ delta,                                          \
    const T_ACT* __restrict__ u,                                              \
    const T_ACT* __restrict__ B_in,                                           \
    const T_ACT* __restrict__ C_in,                                           \
    const float* __restrict__ a_neg,      /* [di*ds] */                       \
    const float* __restrict__ D,          /* [di] */                          \
    const T_ACT* __restrict__ dy,         /* [B*T*di] */                      \
    T_ACT* __restrict__ d_delta,          /* [B*T*di] */                      \
    T_ACT* __restrict__ d_u,              /* [B*T*di] */                      \
    T_ACT* __restrict__ d_B_local,        /* [B*T*di*ds] */                   \
    T_ACT* __restrict__ d_C_local,        /* [B*T*di*ds] */                   \
    float* __restrict__ d_D_local,        /* [B*di] f32 master */             \
    float* __restrict__ d_a_log_local,    /* [B*di*ds] f32 master */          \
    int batch, int T, int d_inner, int d_state,                               \
    /* Slim tape: (run_a, run_b, h_entry) per (b,d,n,chunk). */               \
    const float* run_tape,                                                    \
    int slim_tape                                                             \
) {                                                                           \
    int bid = blockIdx.x;                                                     \
    int did = blockIdx.y;                                                     \
    if (bid >= batch || did >= d_inner) return;                               \
    if (d_state > MAX_DSTATE) return;                                         \
    extern __shared__ float smem[];                                           \
    /* smem_wa/wb/run_a/run_b are forward-layout (small NWARPS-sized warp     \
       totals + MAX_DSTATE running prefix) — UNUSED in the bwd kernel.       \
       smem_exch_a/b are NTHREADS-sized — we repurpose them as the per-      \
       thread inclusive-reverse-scan postfix tile (read by lane k as        \
       smem_exch_*[k+1] for the exclusive-next postfix). */                  \
    float *smem_rev_wa  = smem + SMEM_REV_WA_OFF;                             \
    float *smem_rev_wb  = smem + SMEM_REV_WB_OFF;                             \
    float *smem_exch_a  = smem + SMEM_EXCH_A_OFF;                             \
    float *smem_exch_b  = smem + SMEM_EXCH_B_OFF;                             \
    float *smem_post_a  = smem + SMEM_POST_A_OFF;                             \
    float *smem_post_b  = smem + SMEM_POST_B_OFF;                             \
    float *smem_next_a  = smem + SMEM_NEXT_A_OFF;                             \
    float *smem_da_red  = smem + SMEM_DA_RED_OFF;                             \
    float *smem_chunk_first_a = smem + SMEM_CHUNK_FIRST_A_OFF;                \
    T_ACT *smem_stage   = (T_ACT *)(smem + SMEM_STAGE_OFF);                   \
    /* Replay scratch (slim tape): the fwd-layout regions are                 \
       unused in this kernel - smem_wa/wb feed the forward                    \
       block_inclusive_scan_ab, the RUN_A slot holds the exclusive            \
       prefix exchange (NTHREADS) plus the chunk-boundary H lane              \
       (NTHREADS more; MAX_DSTATE = 256 fits both), RUN_B the                 \
       b-half of the exchange. */                                             \
    float *smem_fwd_wa  = smem + SMEM_WA_OFF;                                 \
    float *smem_fwd_wb  = smem + SMEM_WB_OFF;                                 \
    float *smem_fexch_a = smem + SMEM_RUN_A_OFF;                              \
    float *smem_fexch_b = smem + SMEM_RUN_B_OFF;                              \
    float *smem_hbound  = smem + SMEM_RUN_A_OFF + NTHREADS;                   \
    float D_d = D[did];                                                       \
    /* T-major tape: per-(b,d) row base; +n*(T+1) selects the state
     * lane's contiguous t-run. */                                            \
    int hsave_row_bd = (bid * d_inner + did) * d_state;                       \
    /* Initialize inter-chunk reverse-scan postfix to identity (1, 0). The   \
       chunk_first_a buffer is set to 1.0 to act as `a_{t+1}=1` for the      \
       very-last timestep of the very-last chunk (no future). */              \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        smem_post_a[n] = 1.0f;                                                \
        smem_post_b[n] = 0.0f;                                                \
        smem_chunk_first_a[n] = 1.0f;                                         \
    }                                                                         \
    __syncthreads();                                                          \
    float local_d_D = 0.0f;                                                   \
    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;                         \
    for (int chunk_loop = 0; chunk_loop < n_chunks; chunk_loop++) {           \
        int chunk = n_chunks - 1 - chunk_loop;  /* walk REVERSE */            \
        int chunk_start = chunk * CHUNK_SIZE;                                 \
        /* Barrier diet: delta/u/dy read directly (value-neutral; the         \
         * staging's lane stride exceeded the 32-byte sector either way). */  \
        float delta_vals[NITEMS];                                             \
        float u_vals[NITEMS];                                                 \
        float dy_vals[NITEMS];                                                \
        _Pragma("unroll")                                                     \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            delta_vals[i] =                                                   \
                (t < T) ? to_f(delta[(bid * T + t) * d_inner + did]) : 0.0f;  \
            u_vals[i] = (t < T) ? to_f(u[(bid * T + t) * d_inner + did])      \
                                : 0.0f;                                       \
            dy_vals[i] = (t < T) ? to_f(dy[(bid * T + t) * d_inner + did])    \
                                 : 0.0f;                                      \
        }                                                                     \
        /* ---- Per-t skip-path contributions accumulate in registers ---- */ \
        float d_u_acc[NITEMS];                                                \
        float d_delta_acc[NITEMS];                                            \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            if (t < T) {                                                      \
                local_d_D += dy_vals[i] * u_vals[i];                          \
                d_u_acc[i] = dy_vals[i] * D_d;                                \
            } else {                                                          \
                d_u_acc[i] = 0.0f;                                            \
            }                                                                 \
            d_delta_acc[i] = 0.0f;                                            \
        }                                                                     \
        /* ---- Outer d_state loop ---- */                                    \
        for (int n = 0; n < d_state; n++) {                                   \
            float a_dn = a_neg[did * d_state + n];                            \
            float a_dn_log2 = a_dn * LOG2E;                                   \
            float b_vals[NITEMS];                                             \
            load_row8(B_in + (bid * d_state + n) * T,                         \
                      chunk_start + threadIdx.x * NITEMS, T, b_vals);         \
            float c_vals[NITEMS];                                             \
            load_row8(C_in + (bid * d_state + n) * T,                         \
                      chunk_start + threadIdx.x * NITEMS, T, c_vals);         \
            /* Per-i: da[i] = exp2(delta * a_neg), d_local[i] = dy * c */     \
            float da_vals[NITEMS];                                            \
            float d_local[NITEMS];                                            \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    da_vals[i] = exp2f(delta_vals[i] * a_dn_log2);            \
                    d_local[i] = dy_vals[i] * c_vals[i];                      \
                } else {                                                      \
                    da_vals[i] = 1.0f;                                        \
                    d_local[i] = 0.0f;                                        \
                }                                                             \
            }                                                                 \
            /* Exchange: each thread publishes its first da into smem so the  \
               left-neighbor thread can read it as its (NITEMS-1).a (the      \
               "next-step a" trick — Tri Dao reverse_scan). */                \
            /* Slim-tape replay: reproduce the forward's h_t for              \
               this chunk BIT-exactly - the same thread-local scan,           \
               the same block_inclusive_scan_ab, the same compose             \
               chain ((comp o run) applied to h_0) on the same                \
               inputs. The full-tape path keeps its h_saved reads. */         \
            float H_vals[NITEMS];                                             \
            float h_prev_boundary = 0.0f;                                     \
            if (slim_tape) {                                                  \
                int row = (hsave_row_bd + n) * 3 * n_chunks;                  \
                float f_run_a = run_tape[row + 3 * chunk + 0];                \
                float f_run_b = run_tape[row + 3 * chunk + 1];                \
                float f_hentry = run_tape[row + 3 * chunk + 2];               \
                float f_h0 = run_tape[row + 2];                               \
                float fwd_a[NITEMS];                                          \
                float fwd_b[NITEMS];                                          \
                _Pragma("unroll")                                             \
                for (int i = 0; i < NITEMS; i++) {                            \
                    int t = chunk_start + threadIdx.x * NITEMS + i;           \
                    if (t < T) {                                              \
                        fwd_a[i] = da_vals[i];                                \
                        fwd_b[i] =                                            \
                            (delta_vals[i] * u_vals[i]) * b_vals[i];          \
                    } else {                                                  \
                        fwd_a[i] = 1.0f;                                      \
                        fwd_b[i] = 0.0f;                                      \
                    }                                                         \
                }                                                             \
                _Pragma("unroll")                                             \
                for (int i = 1; i < NITEMS; i++) {                            \
                    fwd_b[i] = fwd_a[i] * fwd_b[i - 1] + fwd_b[i];            \
                    fwd_a[i] = fwd_a[i] * fwd_a[i - 1];                       \
                }                                                             \
                float fscan_a = fwd_a[NITEMS - 1];                            \
                float fscan_b = fwd_b[NITEMS - 1];                            \
                __syncthreads();                                              \
                block_inclusive_scan_ab(                                      \
                    fscan_a, fscan_b, smem_fwd_wa, smem_fwd_wb);              \
                __syncthreads();                                              \
                smem_fexch_a[threadIdx.x] = fscan_a;                          \
                smem_fexch_b[threadIdx.x] = fscan_b;                          \
                __syncthreads();                                              \
                float fexcl_a, fexcl_b;                                       \
                if (threadIdx.x == 0) {                                       \
                    fexcl_a = 1.0f;                                           \
                    fexcl_b = 0.0f;                                           \
                } else {                                                      \
                    fexcl_a = smem_fexch_a[threadIdx.x - 1];                  \
                    fexcl_b = smem_fexch_b[threadIdx.x - 1];                  \
                }                                                             \
                _Pragma("unroll")                                             \
                for (int i = 0; i < NITEMS; i++) {                            \
                    float comp_a = fwd_a[i] * fexcl_a;                        \
                    float comp_b = fwd_a[i] * fexcl_b + fwd_b[i];             \
                    float final_a = comp_a * f_run_a;                         \
                    float final_b = comp_a * f_run_b + comp_b;                \
                    H_vals[i] = final_a * f_h0 + final_b;                     \
                }                                                             \
                smem_hbound[threadIdx.x] = H_vals[NITEMS - 1];                \
                __syncthreads();                                              \
                h_prev_boundary = (threadIdx.x == 0)                          \
                    ? f_hentry                                                \
                    : smem_hbound[threadIdx.x - 1];                           \
                __syncthreads();                                              \
            }                                                                 \
            smem_next_a[threadIdx.x] = da_vals[0];                            \
            __syncthreads();                                                  \
            /* Build reverse-scan pairs (a_next, d_local).                    \
               pair[i].a = da_vals[i+1] for i in [0, NITEMS-1)                \
               pair[NITEMS-1].a = next thread's da_vals[0] from smem_next_a;  \
               last thread of block uses postfix-saved next-chunk's first a. */\
            float thread_a[NITEMS];                                           \
            float thread_b[NITEMS];                                           \
            for (int i = 0; i < NITEMS - 1; i++) {                            \
                thread_a[i] = da_vals[i + 1];                                 \
                thread_b[i] = d_local[i];                                     \
            }                                                                 \
            float boundary_next_a;                                            \
            if ((int)threadIdx.x < NTHREADS - 1) {                            \
                boundary_next_a = smem_next_a[threadIdx.x + 1];               \
            } else {                                                          \
                /* Last thread of block: need a_{t+1} where t+1 is the FIRST \
                   timestep of the NEXT (later-in-time) chunk. Saved into    \
                   smem_chunk_first_a[n] when that chunk was processed.      \
                   Initialized to 1.0 for the very-last chunk in time. */    \
                boundary_next_a = smem_chunk_first_a[n];                      \
            }                                                                 \
            thread_a[NITEMS - 1] = boundary_next_a;                           \
            thread_b[NITEMS - 1] = d_local[NITEMS - 1];                       \
            __syncthreads();                                                  \
            /* Mask out-of-range elements to identity (a=1, b=0). */          \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t >= T) {                                                 \
                    thread_a[i] = 1.0f;                                       \
                    thread_b[i] = 0.0f;                                       \
                }                                                             \
            }                                                                 \
            /* Intra-thread reverse compose (right→left) the NITEMS pairs.    \
               result.b = a · b_right + b_left where compose(left, right). */ \
            for (int i = NITEMS - 2; i >= 0; i--) {                           \
                thread_b[i] = thread_a[i] * thread_b[i + 1] + thread_b[i];    \
                thread_a[i] = thread_a[i] * thread_a[i + 1];                  \
            }                                                                 \
            float scan_a = thread_a[0];                                       \
            float scan_b = thread_b[0];                                       \
            block_inclusive_reverse_scan_ab(                                  \
                scan_a, scan_b, smem_rev_wa, smem_rev_wb);                    \
            __syncthreads();                                                  \
            /* Reverse scan exclusive-NEXT: lane k needs the postfix from     \
               lane k+1 (excl_next_a/b). Save inclusive scan_a/b into the    \
               NTHREADS-sized exch tiles (smem_wa is only NWARPS floats!).   \
               Then read [threadIdx.x + 1]. */                                \
            smem_exch_a[threadIdx.x] = scan_a;                                \
            smem_exch_b[threadIdx.x] = scan_b;                                \
            __syncthreads();                                                  \
            float next_a, next_b;                                             \
            if ((int)threadIdx.x < NTHREADS - 1) {                            \
                next_a = smem_exch_a[threadIdx.x + 1];                        \
                next_b = smem_exch_b[threadIdx.x + 1];                        \
            } else {                                                          \
                /* Last thread of block: no more lanes within THIS chunk →    \
                   the exclusive-next-thread postfix is identity (1, 0).      \
                   The inter-chunk postfix (smem_post_a/b) is composed in     \
                   separately via (run_a, run_b) below. Audit fix: previous  \
                   code aliased next_a = smem_post_a[n] which then double-   \
                   composed with run_a/b at L1378-1379 → wrong dh for the    \
                   last 8 timesteps of every chunk except the very last     \
                   (manifests at T > CHUNK_SIZE = 1024, n_chunks ≥ 2). */    \
                next_a = 1.0f;                                                \
                next_b = 0.0f;                                                \
            }                                                                 \
            float run_a = smem_post_a[n];                                     \
            float run_b = smem_post_b[n];                                     \
            /* Barrier: all warps must READ the postfix before thread 0       \
               overwrites it (same race as the forward smem_run update). */   \
            __syncthreads();                                                  \
            /* Update postfix carry for the NEXT (earlier) chunk. Block-wide  \
               reverse compose end-to-end: thread 0 holds the full chunk      \
               composition. */                                                \
            if (threadIdx.x == 0) {                                           \
                /* Full chunk composition is at lane 0 after rev-scan. */     \
                float chunk_a = scan_a;                                       \
                float chunk_b = scan_b;                                       \
                /* New postfix = compose(chunk_composition, old_postfix).     \
                   compose order: chunk is to the LEFT (earlier), postfix to  \
                   the RIGHT. op_rev(left, right) = (left.a*right.a,          \
                   left.a*right.b + left.b). */                               \
                smem_post_a[n] = chunk_a * run_a;                             \
                smem_post_b[n] = chunk_a * run_b + chunk_b;                   \
            }                                                                 \
            __syncthreads();                                                  \
            /* Now per-i compute dh[i] for each timestep in this thread.      \
               After intra-thread compose: thread_a/b[i] already contains     \
               compose(pair[i], pair[i+1], ..., pair[NITEMS-1]).              \
               Compose with (next_a, next_b) which represents pairs after     \
               this thread, AND with (run_a, run_b) the postfix from later    \
               chunks. Final per-i pair: compose(thread_state[i], next_then_run). */\
            float post_a = next_a * run_a;                                    \
            float post_b = next_a * run_b + next_b;                           \
            float dh_vals[NITEMS];                                            \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < NITEMS; i++) {                                \
                /* dh[i] = thread_a[i]*post_b + thread_b[i] */                \
                dh_vals[i] = thread_a[i] * post_b + thread_b[i];              \
            }                                                                 \
            /* ---- Per-t output writes (typed) and accumulation ---- */      \
            float d_a_acc = 0.0f;                                             \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t >= T) continue;                                         \
                /* T-major: locals go [b][n][d][t] so this kernel's
                 * lane-over-t stores and the tmajor reducer's
                 * lane-over-t reads both coalesce. */                        \
                int btdn_typed = ((bid * d_state + n) * d_inner + did) * T    \
                                 + t;                                         \
                float h_curr, h_prev;                                         \
                if (slim_tape) {                                              \
                    h_curr = H_vals[i];                                       \
                    h_prev = (i > 0) ? H_vals[i - 1] : h_prev_boundary;       \
                } else {                                                      \
                    int h_row = (hsave_row_bd + n) * (T + 1);                 \
                    h_curr = h_saved[h_row + (t + 1)];                        \
                    h_prev = h_saved[h_row + t];                              \
                }                                                             \
                float dh = dh_vals[i];                                        \
                d_C_local[btdn_typed] = FROM_F(dy_vals[i] * h_curr);          \
                d_B_local[btdn_typed] = FROM_F(dh * delta_vals[i] * u_vals[i]);\
                d_delta_acc[i] += dh * (a_dn * da_vals[i] * h_prev            \
                                        + u_vals[i] * b_vals[i]);             \
                d_u_acc[i] += dh * delta_vals[i] * b_vals[i];                 \
                d_a_acc += dh * da_vals[i] * delta_vals[i] * a_dn * h_prev;   \
            }                                                                 \
            /* Block-reduce d_a_acc → thread 0 → += d_a_log_local */          \
            smem_da_red[threadIdx.x] = d_a_acc;                               \
            __syncthreads();                                                  \
            /* Tree rounds down to a full warp in smem, then the same         \
               pairing continues via shuffles: lane i still adds lane         \
               i+off's value at each halving, so every partial sum is         \
               bit-identical to the all-smem tree. */                         \
            for (int stride = NTHREADS / 2; stride >= 32; stride >>= 1) {     \
                if ((int)threadIdx.x < stride) {                              \
                    smem_da_red[threadIdx.x] += smem_da_red[threadIdx.x +     \
                                                            stride];          \
                }                                                             \
                __syncthreads();                                              \
            }                                                                 \
            float da_warp = 0.0f;                                             \
            if (threadIdx.x < 32) {                                           \
                da_warp = smem_da_red[threadIdx.x];                           \
                for (int off = 16; off > 0; off >>= 1)                        \
                    da_warp += __shfl_down_sync(0xFFFFFFFFu, da_warp,         \
                                                off);                         \
            }                                                                 \
            if (threadIdx.x == 0) {                                           \
                d_a_log_local[(bid * d_inner + did) * d_state + n]            \
                    += da_warp;                                               \
                /* Save THIS chunk's first thread's first da into the         \
                   chunk_first_a[n] slot — the EARLIER chunk (next iter)      \
                   will read this as its boundary `a_{t+1}` for the very-     \
                   last timestep before this chunk starts. */                 \
                smem_chunk_first_a[n] = smem_next_a[0];                       \
            }                                                                 \
            __syncthreads();                                                  \
        }                                                                     \
        /* ---- Store d_delta_acc, d_u_acc to typed HBM via smem ---- */      \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            smem_stage[threadIdx.x * NITEMS + i] = FROM_F(d_delta_acc[i]);    \
        }                                                                     \
        __syncthreads();                                                      \
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                          \
            if (t < T) {                                                      \
                d_delta[(bid * T + t) * d_inner + did] = smem_stage[s];       \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            smem_stage[threadIdx.x * NITEMS + i] = FROM_F(d_u_acc[i]);        \
        }                                                                     \
        __syncthreads();                                                      \
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                          \
            if (t < T) {                                                      \
                d_u[(bid * T + t) * d_inner + did] = smem_stage[s];           \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    /* Final: per-block d_D contribution. local_d_D is per-thread so          \
       reduce within block first. */                                          \
    smem_da_red[threadIdx.x] = local_d_D;                                     \
    __syncthreads();                                                          \
    for (int stride = NTHREADS / 2; stride >= 32; stride >>= 1) {             \
        if ((int)threadIdx.x < stride) {                                      \
            smem_da_red[threadIdx.x] += smem_da_red[threadIdx.x + stride];    \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    if (threadIdx.x < 32) {                                                   \
        float dd_warp = smem_da_red[threadIdx.x];                             \
        for (int off = 16; off > 0; off >>= 1)                                \
            dd_warp += __shfl_down_sync(0xFFFFFFFFu, dd_warp, off);           \
        if (threadIdx.x == 0) {                                               \
            d_D_local[bid * d_inner + did] = dd_warp;                         \
        }                                                                     \
    }                                                                         \
}

DEFINE_SSM_PARALLEL_SCAN_BWD(f32,  float,         from_f_f32)
DEFINE_SSM_PARALLEL_SCAN_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_PARALLEL_SCAN_BWD(f16,  __half,        from_f_f16)

// ============================================================================
// Parallel reverse-scan backward, d-group fold variant.
//
// The plain kernel materializes d_B/d_C locals at [B, ds, di, T] - the
// stores alone were 57% of the kernel by ablation, plus the reducer
// reads it all back. This variant gives each block SCAN_BWD_DGROUP
// consecutive d lanes: per (n, i) it folds the group's dB/dC terms in
// ascending-d order into registers and writes ONE partial row per
// group, shrinking the local tensors and the reducer's depth by
// SCAN_BWD_DGROUP. B/C are read once per (n, chunk) per block, so
// their traffic also drops by the group factor. The d-fold grouping
// is a different dB/dC summation order than the ungrouped kernel
// (deliberate; the partition is a pure function of d_inner). The
// launcher uses this kernel only when d_inner % SCAN_BWD_DGROUP == 0;
// the ungrouped kernel stays as the general-shape path.
// ============================================================================
#define SCAN_BWD_DGROUP 4

// MINB is the minimum blocks per SM the launch bounds promise, HOLD_ROWS
// whether a lane keeps its dt, u and dy rows in registers across the two
// scans, STAGED how many of the block's lanes go through the shared
// memory tile. Staging all four f32 lanes takes 48 KB and leaves the SM
// one block: the f32 kernel stages three, reads the fourth lane's rows
// from global memory into the registers it holds them in anyway, and
// fits two blocks per SM on the 255-register budget that gives it. The
// half kernels stage all four lanes in half the bytes, fit three blocks
// per SM within 168 registers, and read the rows back from the tile at
// each use so nothing spills. The launcher's byte count mirrors STAGED.

#define DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD(SUFFIX, T_ACT, FROM_F, MINB,        \
                                          HOLD_ROWS, STAGED)                  \
extern "C" __global__ __launch_bounds__(NTHREADS, MINB) void                  \
ssm_parallel_scan_bwd_fold_##SUFFIX(                                          \
    const float* __restrict__ h_saved,                                        \
    const T_ACT* __restrict__ delta,                                          \
    const T_ACT* __restrict__ u,                                              \
    const T_ACT* __restrict__ B_in,                                           \
    const T_ACT* __restrict__ C_in,                                           \
    const float* __restrict__ a_neg,                                          \
    const float* __restrict__ D,                                              \
    const T_ACT* __restrict__ dy,                                             \
    /* OUTPUT is the PRE-softplus dt gradient: the epilogue applies the       \
       softplus derivative inline (round-FIRST - the accumulator is           \
       rounded to the activation dtype exactly as the old d_delta store       \
       did, and the derivative multiplies the reloaded value), so the         \
       separate softplus backward launch is gone. */                          \
    T_ACT* __restrict__ d_delta_raw_out,                                      \
    const T_ACT* __restrict__ dt_raw,                                         \
    T_ACT* __restrict__ d_u,                                                  \
    T_ACT* __restrict__ d_B_local, /* [B, ds, di/G, T] group partials */      \
    T_ACT* __restrict__ d_C_local, /* [B, ds, di/G, T] group partials */      \
    float* __restrict__ d_D_local,                                            \
    float* __restrict__ d_a_log_local,                                        \
    int batch, int T, int d_inner, int d_state,                               \
    const float* run_tape,                                                    \
    int slim_tape                                                             \
) {                                                                           \
    const int G = SCAN_BWD_DGROUP;                                            \
    int bid = blockIdx.x;                                                     \
    int gid = blockIdx.y; /* d group */                                       \
    int did0 = gid * G;                                                       \
    if (bid >= batch || did0 >= d_inner) return;                              \
    if (d_state > MAX_DSTATE) return;                                         \
    int n_groups = d_inner / G;                                               \
    extern __shared__ float smem[];                                           \
    /* Layout: rev warp scan (2*NWARPS), fwd-replay warp scan                 \
       (2*NWARPS), the raw warp totals of each scan (2*NWARPS twice; a        \
       warp's edge lane rebuilds its neighbour's composed value from          \
       them), post (2*G*d_state), chunk_first_a (G*d_state), next_a           \
       (NWARPS), da_red (G*NTHREADS), hbound (NWARPS), then the typed         \
       delta/u/dy stage (3*STAGED*CHUNK_SIZE T_ACT slots) and the             \
       CHUNK_SIZE store tile. The launcher's byte count mirrors this. */      \
    static_assert(HOLD_ROWS || STAGED == SCAN_BWD_DGROUP,                     \
                  "a lane that reloads its rows needs every lane staged");    \
    float *smem_rev_wa = smem;                                                \
    float *smem_rev_wb = smem_rev_wa + NWARPS;                                \
    float *smem_fwd_wa = smem_rev_wb + NWARPS;                                \
    float *smem_fwd_wb = smem_fwd_wa + NWARPS;                                \
    float *smem_rev_raw_a = smem_fwd_wb + NWARPS;                             \
    float *smem_rev_raw_b = smem_rev_raw_a + NWARPS;                          \
    float *smem_fwd_raw_a = smem_rev_raw_b + NWARPS;                          \
    float *smem_fwd_raw_b = smem_fwd_raw_a + NWARPS;                          \
    float *smem_post_a = smem_fwd_raw_b + NWARPS;                             \
    float *smem_post_b = smem_post_a + SCAN_BWD_DGROUP * d_state;             \
    float *smem_chunk_first_a = smem_post_b + SCAN_BWD_DGROUP * d_state;      \
    float *smem_next_a = smem_chunk_first_a + SCAN_BWD_DGROUP * d_state;      \
    float *smem_da_red = smem_next_a + NWARPS;                                \
    float *smem_hbound = smem_da_red + SCAN_BWD_DGROUP * NTHREADS;            \
    T_ACT *smem_dio = (T_ACT *)(smem_hbound + NWARPS);                        \
    T_ACT *stage_delta = smem_dio;                                            \
    T_ACT *stage_u = stage_delta + STAGED * CHUNK_SIZE;                       \
    T_ACT *stage_dy = stage_u + STAGED * CHUNK_SIZE;                          \
    T_ACT *stage_bc = stage_dy + STAGED * CHUNK_SIZE;                         \
    unsigned warp_mask = 0xFFFFFFFFu;                                         \
    for (int gg = 0; gg < G; gg++) {                                          \
        for (int n = threadIdx.x; n < d_state; n += NTHREADS) {               \
            smem_post_a[gg * d_state + n] = 1.0f;                             \
            smem_post_b[gg * d_state + n] = 0.0f;                             \
            smem_chunk_first_a[gg * d_state + n] = 1.0f;                      \
        }                                                                     \
    }                                                                         \
    __syncthreads();                                                          \
    float local_d_D[SCAN_BWD_DGROUP];                                         \
    _Pragma("unroll")                                                         \
    for (int gg = 0; gg < G; gg++) local_d_D[gg] = 0.0f;                      \
    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;                         \
    for (int chunk_loop = 0; chunk_loop < n_chunks; chunk_loop++) {           \
        int chunk = n_chunks - 1 - chunk_loop;                                \
        int chunk_start = chunk * CHUNK_SIZE;                                 \
        /* Stage the group's delta/u/dy rows once per chunk: one packed       \
           G-wide load per t covers all four lanes (did0..did0+G-1 is         \
           contiguous and the row base is G-aligned), a quarter of the        \
           load instructions of the per-lane sweep. Smem addressing is        \
           unchanged - same slots, same values, no new bank pattern.    */    \
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                          \
            __align__(16) T_ACT pk_delta[G];                                  \
            __align__(16) T_ACT pk_u[G];                                      \
            __align__(16) T_ACT pk_dy[G];                                     \
            if (t < T) {                                                      \
                int row = (bid * T + t) * d_inner + did0;                     \
                if (sizeof(T_ACT) == 4) {                                     \
                    *reinterpret_cast<uint4 *>(pk_delta) =                    \
                        *reinterpret_cast<const uint4 *>(&delta[row]);        \
                    *reinterpret_cast<uint4 *>(pk_u) =                        \
                        *reinterpret_cast<const uint4 *>(&u[row]);            \
                    *reinterpret_cast<uint4 *>(pk_dy) =                       \
                        *reinterpret_cast<const uint4 *>(&dy[row]);           \
                } else {                                                      \
                    *reinterpret_cast<uint2 *>(pk_delta) =                    \
                        *reinterpret_cast<const uint2 *>(&delta[row]);        \
                    *reinterpret_cast<uint2 *>(pk_u) =                        \
                        *reinterpret_cast<const uint2 *>(&u[row]);            \
                    *reinterpret_cast<uint2 *>(pk_dy) =                       \
                        *reinterpret_cast<const uint2 *>(&dy[row]);           \
                }                                                             \
            } else {                                                          \
                _Pragma("unroll")                                             \
                for (int gg = 0; gg < G; gg++) {                              \
                    pk_delta[gg] = FROM_F(0.0f);                              \
                    pk_u[gg] = FROM_F(0.0f);                                  \
                    pk_dy[gg] = FROM_F(0.0f);                                 \
                }                                                             \
            }                                                                 \
            _Pragma("unroll")                                                 \
            for (int gg = 0; gg < STAGED; gg++) {                             \
                stage_delta[gg * CHUNK_SIZE + s] = pk_delta[gg];              \
                stage_u[gg * CHUNK_SIZE + s] = pk_u[gg];                      \
                stage_dy[gg * CHUNK_SIZE + s] = pk_dy[gg];                    \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
        float d_u_acc[SCAN_BWD_DGROUP][NITEMS];                               \
        float d_delta_acc[SCAN_BWD_DGROUP][NITEMS];                           \
        _Pragma("unroll")                                                     \
        for (int gg = 0; gg < G; gg++) {                                      \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    float dyv, uv;                                            \
                    if (gg < STAGED) {                                        \
                        dyv = to_f(stage_dy[gg * CHUNK_SIZE +                 \
                                            threadIdx.x * NITEMS + i]);       \
                        uv = to_f(stage_u[gg * CHUNK_SIZE +                   \
                                          threadIdx.x * NITEMS + i]);         \
                    } else {                                                  \
                        int row = (bid * T + t) * d_inner + did0 + gg;        \
                        dyv = to_f(dy[row]);                                  \
                        uv = to_f(u[row]);                                    \
                    }                                                         \
                    local_d_D[gg] += dyv * uv;                                \
                    d_u_acc[gg][i] = dyv * D[did0 + gg];                      \
                } else {                                                      \
                    d_u_acc[gg][i] = 0.0f;                                    \
                }                                                             \
                d_delta_acc[gg][i] = 0.0f;                                    \
            }                                                                 \
        }                                                                     \
        for (int n = 0; n < d_state; n++) {                                   \
            float b_vals[NITEMS];                                             \
            load_row8(B_in + (bid * d_state + n) * T,                         \
                      chunk_start + threadIdx.x * NITEMS, T, b_vals);         \
            float c_vals[NITEMS];                                             \
            load_row8(C_in + (bid * d_state + n) * T,                         \
                      chunk_start + threadIdx.x * NITEMS, T, c_vals);         \
            float acc_B[NITEMS];                                              \
            float acc_C[NITEMS];                                              \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < NITEMS; i++) {                                \
                acc_B[i] = 0.0f;                                              \
                acc_C[i] = 0.0f;                                              \
            }                                                                 \
            float da_acc[SCAN_BWD_DGROUP];                                    \
            _Pragma("unroll")                                                 \
            for (int gg = 0; gg < G; gg++) da_acc[gg] = 0.0f;                 \
            /* The lane loop is unrolled so that every accumulator the        \
               lane indexes stays in a register; rolled, the arrays are       \
               indexed dynamically and land in local memory. A lane's dt,     \
               u and dy rows are either held in registers for the whole       \
               lane (HOLD_ROWS) or read back from the staged tile where       \
               they are used; see the note above the macro. */                \
            _Pragma("unroll")                                                 \
            for (int gg = 0; gg < G; gg++) {                                  \
                int did = did0 + gg;                                          \
                float a_dn = a_neg[did * d_state + n];                        \
                float a_dn_log2 = a_dn * LOG2E;                               \
                float delta_h[NITEMS];                                        \
                float u_h[NITEMS];                                            \
                float dy_h[NITEMS];                                           \
                _Pragma("unroll")                                             \
                for (int i = 0; i < NITEMS; i++) {                            \
                    int t = chunk_start + threadIdx.x * NITEMS + i;           \
                    int s = threadIdx.x * NITEMS + i;                         \
                    delta_h[i] = 0.0f;                                        \
                    u_h[i] = 0.0f;                                            \
                    dy_h[i] = 0.0f;                                           \
                    if (HOLD_ROWS && gg < STAGED) {                           \
                        delta_h[i] = to_f(stage_delta[gg * CHUNK_SIZE + s]);  \
                        u_h[i] = to_f(stage_u[gg * CHUNK_SIZE + s]);          \
                        dy_h[i] = to_f(stage_dy[gg * CHUNK_SIZE + s]);        \
                    } else if (HOLD_ROWS && t < T) {                          \
                        /* The lane the block does not stage: its rows come   \
                           straight from global memory, the same values the   \
                           tile would have held. */                           \
                        int row = (bid * T + t) * d_inner + did;              \
                        delta_h[i] = to_f(delta[row]);                        \
                        u_h[i] = to_f(u[row]);                                \
                        dy_h[i] = to_f(dy[row]);                              \
                    }                                                         \
                }                                                             \
                float da_vals[NITEMS];                                        \
                _Pragma("unroll")                                             \
                for (int i = 0; i < NITEMS; i++) {                            \
                    int t = chunk_start + threadIdx.x * NITEMS + i;           \
                    int s = threadIdx.x * NITEMS + i;                         \
                    float delta_v = HOLD_ROWS                                 \
                        ? delta_h[i]                                          \
                        : to_f(stage_delta[gg * CHUNK_SIZE + s]);             \
                    da_vals[i] = (t < T) ? exp2f(delta_v * a_dn_log2) : 1.0f; \
                }                                                             \
                /* The running postfix is read before the scans; thread 0     \
                   folds this chunk into it after the reverse scan, when      \
                   every thread has taken its copy. */                        \
                float run_a = smem_post_a[gg * d_state + n];                  \
                float run_b = smem_post_b[gg * d_state + n];                  \
                /* The reverse pairs need the next thread's first decay:      \
                   the lane below hands it up through a shuffle, a warp's     \
                   last lane takes it from the next warp's first lane         \
                   through a per-warp slot written here, before the replay    \
                   scan's barriers, and the block's last thread takes the     \
                   later chunk's boundary. */                                 \
                if ((threadIdx.x & 31) == 0) {                                \
                    smem_next_a[threadIdx.x >> 5] = da_vals[0];               \
                }                                                             \
                int hsave_row = (bid * d_inner + did) * d_state;              \
                float f_hentry = 0.0f;                                        \
                float H_vals[NITEMS];                                         \
                if (slim_tape) {                                              \
                    /* Slim-tape replay pairs (see the ungrouped kernel). */  \
                    int row = (hsave_row + n) * 3 * n_chunks;                 \
                    float f_run_a = run_tape[row + 3 * chunk + 0];            \
                    float f_run_b = run_tape[row + 3 * chunk + 1];            \
                    f_hentry = run_tape[row + 3 * chunk + 2];                 \
                    float f_h0 = run_tape[row + 2];                           \
                    float fwd_a[NITEMS];                                      \
                    float fwd_b[NITEMS];                                      \
                    _Pragma("unroll")                                         \
                    for (int i = 0; i < NITEMS; i++) {                        \
                        int t = chunk_start + threadIdx.x * NITEMS + i;       \
                        int s = threadIdx.x * NITEMS + i;                     \
                        if (t < T) {                                          \
                            float delta_v = HOLD_ROWS                         \
                                ? delta_h[i]                                  \
                                : to_f(stage_delta[gg * CHUNK_SIZE + s]);     \
                            float u_v = HOLD_ROWS                             \
                                ? u_h[i]                                      \
                                : to_f(stage_u[gg * CHUNK_SIZE + s]);         \
                            fwd_a[i] = da_vals[i];                            \
                            fwd_b[i] = (delta_v * u_v) * b_vals[i];           \
                        } else {                                              \
                            fwd_a[i] = 1.0f;                                  \
                            fwd_b[i] = 0.0f;                                  \
                        }                                                     \
                    }                                                         \
                    _Pragma("unroll")                                         \
                    for (int i = 1; i < NITEMS; i++) {                        \
                        fwd_b[i] = fwd_a[i] * fwd_b[i - 1] + fwd_b[i];        \
                        fwd_a[i] = fwd_a[i] * fwd_a[i - 1];                   \
                    }                                                         \
                    float fscan_a = fwd_a[NITEMS - 1];                        \
                    float fscan_b = fwd_b[NITEMS - 1];                        \
                    float fexcl_a, fexcl_b;                                   \
                    block_scan_ab_with_exclusive(fscan_a, fscan_b, fexcl_a,   \
                                                 fexcl_b, smem_fwd_wa,        \
                                                 smem_fwd_wb, smem_fwd_raw_a, \
                                                 smem_fwd_raw_b);             \
                    _Pragma("unroll")                                         \
                    for (int i = 0; i < NITEMS; i++) {                        \
                        float comp_a = fwd_a[i] * fexcl_a;                    \
                        float comp_b = fwd_a[i] * fexcl_b + fwd_b[i];         \
                        float final_a = comp_a * f_run_a;                     \
                        float final_b = comp_a * f_run_b + comp_b;            \
                        H_vals[i] = final_a * f_h0 + final_b;                 \
                    }                                                         \
                    /* The previous timestep's replayed state crosses a       \
                       warp edge through this slot; the reverse scan's        \
                       barriers retire the write before it is read. */        \
                    if ((threadIdx.x & 31) == 31) {                           \
                        smem_hbound[threadIdx.x >> 5] = H_vals[NITEMS - 1];   \
                    }                                                         \
                } else {                                                      \
                    _Pragma("unroll")                                         \
                    for (int i = 0; i < NITEMS; i++) H_vals[i] = 0.0f;        \
                    __syncthreads();                                          \
                }                                                             \
                /* Warp-uniform slot pick, then a per-lane select: no lane    \
                   walks a branch of its own. */                              \
                float edge_next_a = ((threadIdx.x >> 5) == NWARPS - 1)        \
                    ? smem_chunk_first_a[gg * d_state + n]                    \
                    : smem_next_a[(threadIdx.x >> 5) + 1];                    \
                float down_next_a =                                           \
                    __shfl_down_sync(warp_mask, da_vals[0], 1);               \
                float boundary_next_a =                                       \
                    ((threadIdx.x & 31) == 31) ? edge_next_a : down_next_a;   \
                float thread_a[NITEMS];                                       \
                float thread_b[NITEMS];                                       \
                _Pragma("unroll")                                             \
                for (int i = 0; i < NITEMS - 1; i++) {                        \
                    thread_a[i] = da_vals[i + 1];                             \
                }                                                             \
                thread_a[NITEMS - 1] = boundary_next_a;                       \
                _Pragma("unroll")                                             \
                for (int i = 0; i < NITEMS; i++) {                            \
                    int t = chunk_start + threadIdx.x * NITEMS + i;           \
                    int s = threadIdx.x * NITEMS + i;                         \
                    if (t < T) {                                              \
                        float dy_v = HOLD_ROWS                                \
                            ? dy_h[i]                                         \
                            : to_f(stage_dy[gg * CHUNK_SIZE + s]);            \
                        thread_b[i] = dy_v * c_vals[i];                       \
                    } else {                                                  \
                        thread_a[i] = 1.0f;                                   \
                        thread_b[i] = 0.0f;                                   \
                    }                                                         \
                }                                                             \
                _Pragma("unroll")                                             \
                for (int i = NITEMS - 2; i >= 0; i--) {                       \
                    thread_b[i] = thread_a[i] * thread_b[i + 1] +             \
                                  thread_b[i];                                \
                    thread_a[i] = thread_a[i] * thread_a[i + 1];              \
                }                                                             \
                float scan_a = thread_a[0];                                   \
                float scan_b = thread_b[0];                                   \
                float next_a, next_b;                                         \
                block_reverse_scan_ab_with_exclusive(scan_a, scan_b, next_a,  \
                                                     next_b, smem_rev_wa,     \
                                                     smem_rev_wb,             \
                                                     smem_rev_raw_a,          \
                                                     smem_rev_raw_b);         \
                float edge_h_prev = ((threadIdx.x >> 5) == 0)                 \
                    ? f_hentry                                                \
                    : smem_hbound[(threadIdx.x >> 5) - 1];                    \
                float up_h_prev =                                             \
                    __shfl_up_sync(warp_mask, H_vals[NITEMS - 1], 1);         \
                float h_prev_boundary =                                       \
                    ((threadIdx.x & 31) == 0) ? edge_h_prev : up_h_prev;      \
                if (threadIdx.x == 0) {                                       \
                    smem_post_a[gg * d_state + n] = scan_a * run_a;           \
                    smem_post_b[gg * d_state + n] =                           \
                        scan_a * run_b + scan_b;                              \
                }                                                             \
                float post_a = next_a * run_a;                                \
                float post_b = next_a * run_b + next_b;                       \
                _Pragma("unroll")                                             \
                for (int i = 0; i < NITEMS; i++) {                            \
                    int t = chunk_start + threadIdx.x * NITEMS + i;           \
                    if (t >= T) continue;                                     \
                    int s = threadIdx.x * NITEMS + i;                         \
                    float delta_v = HOLD_ROWS                                 \
                        ? delta_h[i]                                          \
                        : to_f(stage_delta[gg * CHUNK_SIZE + s]);             \
                    float u_v = HOLD_ROWS                                     \
                        ? u_h[i] : to_f(stage_u[gg * CHUNK_SIZE + s]);        \
                    float dy_v = HOLD_ROWS                                    \
                        ? dy_h[i] : to_f(stage_dy[gg * CHUNK_SIZE + s]);      \
                    float dh = thread_a[i] * post_b + thread_b[i];            \
                    float h_curr, h_prev;                                     \
                    if (slim_tape) {                                          \
                        h_curr = H_vals[i];                                   \
                        h_prev =                                              \
                            (i > 0) ? H_vals[i - 1] : h_prev_boundary;        \
                    } else {                                                  \
                        int h_row = (hsave_row + n) * (T + 1);                \
                        h_curr = h_saved[h_row + (t + 1)];                    \
                        h_prev = h_saved[h_row + t];                          \
                    }                                                         \
                    /* ascending-g fold replaces the per-d store */           \
                    acc_C[i] += dy_v * h_curr;                                \
                    acc_B[i] += dh * delta_v * u_v;                           \
                    d_delta_acc[gg][i] += dh * (a_dn * da_vals[i] * h_prev    \
                                                + u_v * b_vals[i]);           \
                    d_u_acc[gg][i] += dh * delta_v * b_vals[i];               \
                    da_acc[gg] += dh * da_vals[i] * delta_v * a_dn *          \
                                  h_prev;                                     \
                }                                                             \
                /* This chunk's first decay is the earlier chunk's            \
                   boundary; the last thread read the old value before        \
                   the reverse scan's barriers. */                            \
                if (threadIdx.x == 0) {                                       \
                    smem_chunk_first_a[gg * d_state + n] = da_vals[0];        \
                }                                                             \
            }                                                                 \
            /* d_a: one block reduction per group, the four sharing their     \
               barriers; each keeps its own lane pairing and add order. */    \
            _Pragma("unroll")                                                 \
            for (int gg = 0; gg < G; gg++) {                                  \
                smem_da_red[gg * NTHREADS + threadIdx.x] = da_acc[gg];        \
            }                                                                 \
            __syncthreads();                                                  \
            for (int stride = NTHREADS / 2; stride >= 32; stride >>= 1) {     \
                if ((int)threadIdx.x < stride) {                              \
                    _Pragma("unroll")                                         \
                    for (int gg = 0; gg < G; gg++) {                          \
                        smem_da_red[gg * NTHREADS + threadIdx.x] +=           \
                            smem_da_red[gg * NTHREADS + threadIdx.x + stride]; \
                    }                                                         \
                }                                                             \
                __syncthreads();                                              \
            }                                                                 \
            if (threadIdx.x < 32) {                                           \
                _Pragma("unroll")                                             \
                for (int gg = 0; gg < G; gg++) {                              \
                    float da_warp = smem_da_red[gg * NTHREADS + threadIdx.x]; \
                    for (int off = 16; off > 0; off >>= 1)                    \
                        da_warp += __shfl_down_sync(warp_mask, da_warp,       \
                                                    off);                     \
                    if (threadIdx.x == 0) {                                   \
                        /* One partial SLOT per chunk instead of a global     \
                           read-modify-write per (chunk, lane, n): slot       \
                           order mirrors the walk (chunk_loop ascends =       \
                           chunks DESCEND in time), and the chunked           \
                           reducer folds the slots in exactly this order      \
                           before adding across the batch - the same          \
                           left-to-right chain the accumulator produced. */   \
                        d_a_log_local[((bid * n_chunks + chunk_loop)          \
                                       * d_inner + did0 + gg) * d_state + n]  \
                            = da_warp;                                        \
                    }                                                         \
                }                                                             \
            }                                                                 \
            /* One partial row per (n, group): [b][n][group][t]. The row is   \
               t-contiguous, but the lane->t mapping is blocked - a direct    \
               store spans 16 sectors per warp instruction. Stage through     \
               the CHUNK tile and store striped: consecutive lanes then       \
               write consecutive addresses. Values unchanged.            */   \
            int row_bc = ((bid * d_state + n) * (d_inner / G) + gid) * T;     \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < NITEMS; i++) {                                \
                stage_bc[threadIdx.x * NITEMS + i] = FROM_F(acc_B[i]);        \
            }                                                                 \
            __syncthreads();                                                  \
            for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {        \
                int t = chunk_start + s;                                      \
                if (t < T) d_B_local[row_bc + t] = stage_bc[s];               \
            }                                                                 \
            __syncthreads();                                                  \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < NITEMS; i++) {                                \
                stage_bc[threadIdx.x * NITEMS + i] = FROM_F(acc_C[i]);        \
            }                                                                 \
            __syncthreads();                                                  \
            for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {        \
                int t = chunk_start + s;                                      \
                if (t < T) d_C_local[row_bc + t] = stage_bc[s];               \
            }                                                                 \
        }                                                                     \
        /* d_delta / d_u: one packed G-wide store per t. Each lane            \
           already holds all G lanes' values for its own t positions,         \
           did0..did0+G-1 is contiguous, and the row base is aligned          \
           because d_inner % G == 0 is the fold's launch precondition -       \
           the smem staging round-trip and its eight barriers per chunk       \
           bought nothing (destination stride between consecutive t is        \
           d_inner elements either way). Values and rounding unchanged.  */   \
        _Pragma("unroll")                                                     \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            if (t < T) {                                                      \
                __align__(16) T_ACT pack_d[G];                                \
                __align__(16) T_ACT pack_u[G];                                \
                __align__(16) T_ACT pack_raw[G];                              \
                int row = (bid * T + t) * d_inner + did0;                     \
                if (sizeof(T_ACT) == 4) {                                     \
                    *reinterpret_cast<uint4 *>(pack_raw) =                    \
                        *reinterpret_cast<const uint4 *>(&dt_raw[row]);       \
                } else {                                                      \
                    *reinterpret_cast<uint2 *>(pack_raw) =                    \
                        *reinterpret_cast<const uint2 *>(&dt_raw[row]);       \
                }                                                             \
                _Pragma("unroll")                                             \
                for (int gg = 0; gg < G; gg++) {                              \
                    /* Round FIRST: the accumulator rounds to the             \
                       activation dtype exactly as the old d_delta store      \
                       did, then the softplus derivative divides the          \
                       reloaded value - the retired kernel's chain,           \
                       rounding for rounding. */                              \
                    float dd = to_f(FROM_F(d_delta_acc[gg][i]));              \
                    float xr = to_f(pack_raw[gg]);                            \
                    pack_d[gg] = FROM_F(                                      \
                        dd / (1.0f + exp2f(-xr * 1.4426950408889634f)));      \
                    pack_u[gg] = FROM_F(d_u_acc[gg][i]);                      \
                }                                                             \
                if (sizeof(T_ACT) == 4) {                                     \
                    *reinterpret_cast<uint4 *>(&d_delta_raw_out[row]) =       \
                        *reinterpret_cast<uint4 *>(pack_d);                   \
                    *reinterpret_cast<uint4 *>(&d_u[row]) =                   \
                        *reinterpret_cast<uint4 *>(pack_u);                   \
                } else {                                                      \
                    *reinterpret_cast<uint2 *>(&d_delta_raw_out[row]) =       \
                        *reinterpret_cast<uint2 *>(pack_d);                   \
                    *reinterpret_cast<uint2 *>(&d_u[row]) =                   \
                        *reinterpret_cast<uint2 *>(pack_u);                   \
                }                                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int gg = 0; gg < G; gg++) {                                          \
        smem_da_red[threadIdx.x] = local_d_D[gg];                             \
        __syncthreads();                                                      \
        for (int stride = NTHREADS / 2; stride >= 32; stride >>= 1) {         \
            if ((int)threadIdx.x < stride) {                                  \
                smem_da_red[threadIdx.x] +=                                   \
                    smem_da_red[threadIdx.x + stride];                        \
            }                                                                 \
            __syncthreads();                                                  \
        }                                                                     \
        if (threadIdx.x < 32) {                                               \
            float dd_warp = smem_da_red[threadIdx.x];                         \
            for (int off = 16; off > 0; off >>= 1)                            \
                dd_warp += __shfl_down_sync(warp_mask, dd_warp, off);         \
            if (threadIdx.x == 0) {                                           \
                d_D_local[bid * d_inner + did0 + gg] = dd_warp;               \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
}

DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD(f32,  float,         from_f_f32,  2, 1, 3)
DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD(bf16, __nv_bfloat16, from_f_bf16, 3, 0, 4)
DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD(f16,  __half,        from_f_f16,  3, 0, 4)


// Clean up macros to avoid polluting subsequent translation units
// (all .cu files are concatenated before NVRTC compilation)
#undef NTHREADS
#undef NITEMS
#undef CHUNK_SIZE
#undef NWARPS
#undef MAX_DSTATE
#undef SMEM_WA_OFF
#undef SMEM_WB_OFF
#undef SMEM_RUN_A_OFF
#undef SMEM_RUN_B_OFF
#undef SMEM_EXCH_A_OFF
#undef SMEM_EXCH_B_OFF
#undef SMEM_STAGE_OFF
#undef SMEM_TOTAL_FLOATS
/* the epilogue stopped at the forward set,
   leaving the backward smem offsets + the DEFINE_* generator names alive
   in the concatenated NVRTC TU. Complete the cleanup. */
#undef SMEM_REV_WA_OFF
#undef SMEM_REV_WB_OFF
#undef SMEM_POST_A_OFF
#undef SMEM_POST_B_OFF
#undef SMEM_NEXT_A_OFF
#undef SMEM_DA_RED_OFF
#undef SMEM_CHUNK_FIRST_A_OFF
#undef SMEM_BWD_FLOATS
#undef DEFINE_SSM_PARALLEL_SCAN_FWD
#undef DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE
#undef DEFINE_SSM_PARALLEL_SCAN_BWD
#undef DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD
#undef SCAN_BWD_DGROUP
#line 1 "kernels/conv1d.cu"
// Depthwise Conv1d CUDA kernels for Mamba.
//
// Shift register pattern: state[d, 0..d_conv] updated each step.
// Depthwise: each channel d independent.
//
// Source: CPU reference: train/backward_ops.rs backward_conv1d_step


// Conv1d step forward (T=1): shift register + depthwise dot product
// state[b, d, 0..d_conv-1] = state[b, d, 1..d_conv]
// state[b, d, d_conv-1] = new_x[b, d]
// out[b, d] = sum_k(state[b, d, k] * weight[d, k]) + bias[d]
//
// Both step kernels below take a registerized d_conv == 4 fast path (the
// production value): the shift happens in registers and global memory
// sees one window load, one writeback and one output store - the generic
// path pays three global read-modify-write shifts plus four reloads of
// what it just wrote, a read-after-write hazard the compiler cannot
// forward through. Value contract: the accumulation is the k-ascending
// chain of the generic loop, term for term; an f32 held in a register
// carries the exact bits a global round-trip would have returned.

// Templated conv1d step with fused SiLU on output. Inference-only fast
// path — replaces the F4 (conv1d_step) + F4b (silu_fwd) launch pair with
// a single kernel. Saves one kernel launch per layer per step (~3-5 µs
// each on Ada). Math identical to running conv1d_step_forward then
// silu_forward in sequence (silu = x / (1 + exp(-x))).
//
// Training kernels (conv1d_burnin_*) keep silu separate so the silu
// backward gets its own activation save buffer — DO NOT fuse those.
#define DEFINE_CONV1D_STEP_FWD_SILU(SUFFIX, T, FROM_F)                     \
extern "C" __global__ void conv1d_step_forward_silu_##SUFFIX(               \
    T* __restrict__ out,                                                   \
    float* __restrict__ state,                                             \
    /* Row stride of new_x: d_inner for a packed input, 2 * d_inner when   \
       the kernel reads the x half of the in_proj output directly. */      \
    const T* __restrict__ new_x,                                           \
    int x_stride,                                                          \
    const float* __restrict__ weight,                                      \
    const float* __restrict__ bias,                                        \
    int batch, int d_inner, int d_conv                                     \
) {                                                                        \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    int total = batch * d_inner;                                           \
    if (idx >= total) return;                                              \
    int b = idx / d_inner;                                                 \
    int d = idx % d_inner;                                                 \
    int x_idx = b * x_stride + d;                                          \
    int state_base = (b * d_inner + d) * d_conv;                           \
    /* Registerized d_conv == 4 fast path - see the note above. */          \
    if (d_conv == 4) {                                                     \
        float s0 = state[state_base + 1];                                  \
        float s1 = state[state_base + 2];                                  \
        float s2 = state[state_base + 3];                                  \
        float s3 = to_f(new_x[x_idx]);                                     \
        state[state_base] = s0;                                            \
        state[state_base + 1] = s1;                                        \
        state[state_base + 2] = s2;                                        \
        state[state_base + 3] = s3;                                        \
        float sum = bias[d];                                               \
        sum += s0 * weight[d * 4];                                         \
        sum += s1 * weight[d * 4 + 1];                                     \
        sum += s2 * weight[d * 4 + 2];                                     \
        sum += s3 * weight[d * 4 + 3];                                     \
        float silu4 = sum / (1.0f + exp2f(-sum * LOG2E));                  \
        out[idx] = FROM_F(silu4);                                          \
        return;                                                            \
    }                                                                      \
    for (int k = 0; k < d_conv - 1; k++) {                                 \
        state[state_base + k] = state[state_base + k + 1];                 \
    }                                                                      \
    state[state_base + d_conv - 1] = to_f(new_x[x_idx]);                   \
    float sum = bias[d];                                                   \
    for (int k = 0; k < d_conv; k++) {                                     \
        sum += state[state_base + k] * weight[d * d_conv + k];             \
    }                                                                      \
    /* SiLU: x * sigmoid(x), via fast exp2f trick */                       \
    float silu = sum / (1.0f + exp2f(-sum * LOG2E));                       \
    out[idx] = FROM_F(silu);                                               \
}

DEFINE_CONV1D_STEP_FWD_SILU(f32,  float,         from_f_f32)
DEFINE_CONV1D_STEP_FWD_SILU(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_CONV1D_STEP_FWD_SILU(f16,  __half,        from_f_f16)

// ======================== BURNIN (T>1) ========================

// conv1d_burnin_backward typed (bf16/f16/f32) for mixed-precision training.
// Pattern matches Dao-AILab/causal-conv1d backward: activations T_IN, weights
// f32 master, recurrent state save (`conv_states`) f32.
//
// Rule B (no atomicAdd): accumulate d_weight/d_bias into thread-local
// registers across the T-loop, then write one per-(b,d) partial at the end.
// Caller MUST follow with two reduce_sum_axis0 launches:
//   reduce_sum_axis0(d_weight, d_weight_partials, batch, d_inner*d_conv, 1)
//   reduce_sum_axis0(d_bias,   d_bias_partials,   batch, d_inner,         1)
//
// Carry register array stays float[8] regardless of T_IN (BUG-M2 fix needs
// f32 precision for the back-propagated gradient through the shift register).
//
// CONSTRAINT: d_conv <= 8 (compile-time register array). All shipped
// state-spaces/mamba checkpoints use d_conv=4.
#define DEFINE_CONV1D_BURNIN_BWD(SUFFIX, T, FROM_F)                            \
extern "C" __global__ __launch_bounds__(256, 4)                                \
void conv1d_burnin_backward_##SUFFIX(                                          \
    T* __restrict__ d_x_branch,                                                \
    float* __restrict__ d_weight_partials,                                     \
    float* __restrict__ d_bias_partials,                                       \
    const T* __restrict__ d_u,                                                 \
    const T* __restrict__ post_conv,                                           \
    const float* __restrict__ conv_states,                                     \
    const float* __restrict__ weight,                                          \
    int batch, int T_, int d_inner, int d_conv                                 \
) {                                                                            \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int total = batch * d_inner;                                               \
    if (idx >= total) return;                                                  \
    int b = idx / d_inner;                                                     \
    int d = idx % d_inner;                                                     \
    float carry[8];                                                            \
    if (d_conv > 8) return;                                                    \
    for (int k = 0; k < d_conv - 1; k++) carry[k] = 0.0f;                      \
    float local_d_weight[8];                                                   \
    for (int k = 0; k < d_conv; k++) local_d_weight[k] = 0.0f;                 \
    float local_d_bias = 0.0f;                                                 \
    for (int t = T_ - 1; t >= 0; t--) {                                        \
        int bt_di = (b * T_ + t) * d_inner + d;                                \
        float x = to_f(post_conv[bt_di]);                                      \
        float sig = 1.0f / (1.0f + exp2f(-x * 1.4426950408889634f));           \
        float silu_grad = sig * (1.0f + x * (1.0f - sig));                     \
        float d_conv_out = to_f(d_u[bt_di]) * silu_grad;                       \
        float dxb = d_conv_out * weight[d * d_conv + d_conv - 1];              \
        int cs_base = ((b * T_ + t) * d_inner + d) * d_conv;                   \
        for (int k = 0; k < d_conv; k++) {                                     \
            local_d_weight[k] += d_conv_out * conv_states[cs_base + k];        \
        }                                                                      \
        local_d_bias += d_conv_out;                                            \
        if (d_conv > 1) {                                                      \
            int carry_len = d_conv - 1;                                        \
            dxb += carry[0];                                                   \
            for (int k = 0; k < carry_len - 1; k++) {                          \
                carry[k] = carry[k + 1]                                        \
                         + d_conv_out * weight[d * d_conv + d_conv - 2 - k];   \
            }                                                                  \
            carry[carry_len - 1] = d_conv_out * weight[d * d_conv];            \
        }                                                                      \
        d_x_branch[bt_di] = FROM_F(dxb);                                       \
    }                                                                          \
    /* Rule B stage-1 output: single per-(b,d) write. */                       \
    int wp_base = (b * d_inner + d) * d_conv;                                  \
    for (int k = 0; k < d_conv; k++) {                                         \
        d_weight_partials[wp_base + k] = local_d_weight[k];                    \
    }                                                                          \
    d_bias_partials[b * d_inner + d] = local_d_bias;                           \
}

DEFINE_CONV1D_BURNIN_BWD(f32,  float,         from_f_f32)
DEFINE_CONV1D_BURNIN_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_CONV1D_BURNIN_BWD(f16,  __half,        from_f_f16)

// ============================================================================
// Tiled conv1d (S-conv phase): the burnin kernels above run ONE thread per
// (b, d) with a T-long serial loop — 24 blocks at the production shape, 146
// idle SMs. The window only reaches d_conv-1 = 3 steps back, so the t-range
// tiles perfectly: tile 0 seeds from the carry-in state, tiles k>0 seed
// their window from x_branch halo loads — the SAME values the serial walk
// would hold, so every output element is bit-identical. Only the tile
// containing t = T-1 writes the carry-out state.
// ============================================================================
#define CONV1D_TILE_T 128

#define DEFINE_CONV1D_BURNIN_TILED(SUFFIX, TY, FROM_F)                        \
extern "C" __global__ void conv1d_burnin_forward_tiled_##SUFFIX(              \
    TY* u_out, float* state, float* conv_states_saved,                        \
    const TY* x_branch, const float* weight, const float* bias,              \
    int batch, int T_len, int d_inner, int d_conv,                            \
    int x_stride /* row stride of x_branch: d_inner, or 2*d_inner when        \
                    reading the in_proj output directly */                    \
) {                                                                          \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                         \
    int total = batch * d_inner;                                             \
    if (idx >= total) return;                                                \
    if (d_conv > 8) return;                                                  \
    int b = idx / d_inner;                                                   \
    int d = idx % d_inner;                                                   \
    int t0 = blockIdx.y * CONV1D_TILE_T;                                     \
    if (t0 >= T_len) return;                                                 \
    int t_end = min(t0 + CONV1D_TILE_T, T_len);                              \
    int state_base = (b * d_inner + d) * d_conv;                             \
    float win[8];                                                            \
    if (t0 == 0) {                                                           \
        for (int k = 0; k < d_conv; k++) {                                   \
            win[k] = state[state_base + k];                                  \
            /* The ONLY conv tape is the carry-in window — the      \
             * backward reconstructs every later window from x_branch. */   \
            conv_states_saved[state_base + k] = win[k];                      \
        }                                                                    \
        /* Carry-out, written by the block that owns the carry-in and      \
         * never by another: after T steps the serial walk holds           \
         * win[k] = x[T - d_conv + k], and the surviving tail of the       \
         * incoming window where that index is negative. Reading the       \
         * whole closed form before storing keeps this thread's own        \
         * carry-in intact. No other block reads or writes state, so       \
         * the launch carries no cross-block order. */                     \
        float carry[8];                                                      \
        for (int k = 0; k < d_conv; k++) {                                   \
            int th = T_len - d_conv + k;                                     \
            carry[k] = (th >= 0)                                             \
                ? to_f(x_branch[(b * T_len + th) * x_stride + d])            \
                : state[state_base + k + T_len];                             \
        }                                                                    \
        for (int k = 0; k < d_conv; k++) state[state_base + k] = carry[k];   \
    } else {                                                                 \
        /* Halo: the serial walk's window after step t0-1 holds           \
         * win[k] = x[t0 - d_conv + k] (win[0] is about to shift out).   \
         * t0 >= CONV1D_TILE_T > d_conv, so the halo never underruns. */ \
        for (int k = 0; k < d_conv; k++) {                                   \
            int th = t0 - d_conv + k;                                        \
            win[k] = to_f(x_branch[(b * T_len + th) * x_stride + d]);         \
        }                                                                    \
    }                                                                        \
    for (int t = t0; t < t_end; t++) {                                       \
        int bt_di = (b * T_len + t) * d_inner + d;                           \
        for (int k = 0; k < d_conv - 1; k++) win[k] = win[k + 1];            \
        win[d_conv - 1] = to_f(x_branch[(b * T_len + t) * x_stride + d]);     \
        float val = bias[d];                                                 \
        for (int k = 0; k < d_conv; k++) {                                   \
            val += win[k] * weight[d * d_conv + k];                          \
        }                                                                    \
        u_out[bt_di] = FROM_F(val / (1.0f + exp2f(-val * 1.4426950408889634f))); \
    }                                                                        \
}

DEFINE_CONV1D_BURNIN_TILED(f32,  float,         from_f_f32)
DEFINE_CONV1D_BURNIN_TILED(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_CONV1D_BURNIN_TILED(f16,  __half,        from_f_f16)

// Nosave tiled twin for the inference prefill: same tile-0 state seeding,
// same x_branch halo, same k-ascending FMA chain - bit-identical to the
// serial nosave walk - but no conv tape and no post_conv (inference keeps
// nothing). Only the tile containing t = T-1 writes the carry-out state,
// so a continued prefill sees exactly the serial walk's final window.
#define DEFINE_CONV1D_BURNIN_NOSAVE_TILED(SUFFIX, TY, FROM_F)                 \
extern "C" __global__ void conv1d_burnin_forward_nosave_tiled_##SUFFIX(       \
    TY* u_out, float* state,                                                 \
    const TY* x_branch, const float* weight, const float* bias,              \
    int batch, int T_len, int d_inner, int d_conv,                           \
    int x_stride /* row stride of x_branch; d_inner, or 2*d_inner when     \
                    reading the in_proj output directly */                   \
) {                                                                          \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                         \
    int total = batch * d_inner;                                             \
    if (idx >= total) return;                                                \
    if (d_conv > 8) return;                                                  \
    int b = idx / d_inner;                                                   \
    int d = idx % d_inner;                                                   \
    int t0 = blockIdx.y * CONV1D_TILE_T;                                     \
    if (t0 >= T_len) return;                                                 \
    int t_end = min(t0 + CONV1D_TILE_T, T_len);                              \
    int state_base = (b * d_inner + d) * d_conv;                             \
    float win[8];                                                            \
    if (t0 == 0) {                                                           \
        for (int k = 0; k < d_conv; k++) {                                   \
            win[k] = state[state_base + k];                                  \
        }                                                                    \
        /* Carry-out, written by the block that owns the carry-in and      \
         * never by another: after T steps the serial walk holds           \
         * win[k] = x[T - d_conv + k], and the surviving tail of the       \
         * incoming window where that index is negative. Reading the       \
         * whole closed form before storing keeps this thread's own        \
         * carry-in intact. No other block reads or writes state, so       \
         * the launch carries no cross-block order. */                     \
        float carry[8];                                                      \
        for (int k = 0; k < d_conv; k++) {                                   \
            int th = T_len - d_conv + k;                                     \
            carry[k] = (th >= 0)                                             \
                ? to_f(x_branch[(b * T_len + th) * x_stride + d])            \
                : state[state_base + k + T_len];                             \
        }                                                                    \
        for (int k = 0; k < d_conv; k++) state[state_base + k] = carry[k];   \
    } else {                                                                 \
        for (int k = 0; k < d_conv; k++) {                                   \
            int th = t0 - d_conv + k;                                        \
            win[k] = to_f(x_branch[(b * T_len + th) * x_stride + d]);         \
        }                                                                    \
    }                                                                        \
    for (int t = t0; t < t_end; t++) {                                       \
        int bt_di = (b * T_len + t) * d_inner + d;                           \
        for (int k = 0; k < d_conv - 1; k++) win[k] = win[k + 1];            \
        win[d_conv - 1] = to_f(x_branch[(b * T_len + t) * x_stride + d]);    \
        float val = bias[d];                                                 \
        for (int k = 0; k < d_conv; k++) {                                   \
            val += win[k] * weight[d * d_conv + k];                          \
        }                                                                    \
        u_out[bt_di] = FROM_F(val / (1.0f + exp2f(-val * 1.4426950408889634f))); \
    }                                                                        \
}

DEFINE_CONV1D_BURNIN_NOSAVE_TILED(f32,  float,         from_f_f32)
DEFINE_CONV1D_BURNIN_NOSAVE_TILED(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_CONV1D_BURNIN_NOSAVE_TILED(f16,  __half,        from_f_f16)

// The conv backward as one kernel per (b, d, tile). The pre-activation
// is recomputed from the x window instead of read from a tape, so the
// forward keeps no pre-activation copy; the SiLU derivative is taken once
// per element instead of once per tap lane; and d_x, the tap gradients and
// the bias gradient walk the tile together. d_x is the anticausal FIR with
// its carries seeded exactly as the serial walk would hold them at the
// tile boundary; the tap and bias accumulators keep the descending-t order
// within the tile and write one partial row each that the fixed-order
// reduction folds as before. The recomputed pre-activation is the
// forward's own chain on the forward's own window values, rounded to the
// activation type as the forward's store rounded it.
#define DEFINE_CONV1D_BWD_TILED(SUFFIX, TY, FROM_F)                           \
extern "C" __global__ void conv1d_bwd_tiled_##SUFFIX(                         \
    TY* __restrict__ d_x_branch,                                              \
    float* __restrict__ d_weight_partials, /* [B*n_tiles, di*d_conv] */       \
    float* __restrict__ d_bias_partials,   /* [B*n_tiles, di] */              \
    const TY* __restrict__ d_u,                                               \
    const TY* __restrict__ x_branch,                                          \
    const float* __restrict__ conv_init, /* [B*di*d_conv] carry-in */         \
    const float* __restrict__ weight,                                         \
    const float* __restrict__ bias,                                           \
    int batch, int T_, int d_inner, int d_conv,                               \
    /* Row stride and column offset of the d_x destination: the x half        \
       of d_proj [bt, 2*d_inner] in production. */                            \
    int out_stride, int out_offset,                                           \
    int x_stride /* row stride of x_branch, as in the forward */              \
) {                                                                           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    int total = batch * d_inner;                                              \
    if (idx >= total) return;                                                 \
    if (d_conv > 8) return;                                                   \
    int b = idx / d_inner;                                                    \
    int d = idx % d_inner;                                                    \
    int n_tiles = (T_ + CONV1D_TILE_T - 1) / CONV1D_TILE_T;                   \
    int tile = blockIdx.y;                                                    \
    int t0 = tile * CONV1D_TILE_T;                                            \
    if (t0 >= T_) return;                                                     \
    int t_end = min(t0 + CONV1D_TILE_T, T_);                                  \
    int row = b * n_tiles + tile;                                             \
    int init_base = (b * d_inner + d) * d_conv;                               \
    int carry_len = d_conv - 1;                                               \
    float carry[8];                                                           \
    float dw[8];                                                              \
    float win[8];                                                             \
    for (int k = 0; k < carry_len; k++) carry[k] = 0.0f;                      \
    for (int k = 0; k < d_conv; k++) dw[k] = 0.0f;                            \
    float local_d_bias = 0.0f;                                                \
    /* The window at step ts holds x[ts - d_conv + 1 + k]; a negative         \
       index reads the carry-in state past its first slot. */                 \
    int ts_top = t_end + carry_len - 1;                                       \
    int ts0 = min(ts_top, T_ - 1);                                            \
    for (int k = 0; k < d_conv; k++) {                                        \
        int tx = ts0 - carry_len + k;                                         \
        win[k] = (tx >= 0)                                                    \
            ? to_f(x_branch[(b * T_ + tx) * x_stride + d])                    \
            : conv_init[init_base + tx + d_conv];                             \
    }                                                                         \
    /* Descending walk: the steps past the tile seed the d_x carries as       \
       the serial walk would hold them (dco = 0 past T), then the tile. */    \
    for (int ts = ts_top; ts >= t0; ts--) {                                   \
        float dco = 0.0f;                                                     \
        float du = 0.0f;                                                      \
        float silu_grad = 0.0f;                                               \
        if (ts < T_) {                                                        \
            if (ts < ts0) {                                                   \
                for (int k = d_conv - 1; k >= 1; k--) win[k] = win[k - 1];    \
                int tx = ts - carry_len;                                      \
                win[0] = (tx >= 0)                                            \
                    ? to_f(x_branch[(b * T_ + tx) * x_stride + d])            \
                    : conv_init[init_base + tx + d_conv];                     \
            }                                                                 \
            float val = bias[d];                                              \
            for (int k = 0; k < d_conv; k++) {                                \
                val += win[k] * weight[d * d_conv + k];                       \
            }                                                                 \
            float x = to_f(FROM_F(val));                                      \
            float sig = 1.0f / (1.0f + exp2f(-x * 1.4426950408889634f));      \
            silu_grad = sig * (1.0f + x * (1.0f - sig));                      \
            du = to_f(d_u[(b * T_ + ts) * d_inner + d]);                      \
            dco = du * silu_grad;                                             \
        }                                                                     \
        if (ts < t_end) {                                                     \
            for (int k = 0; k < d_conv; k++) dw[k] += dco * win[k];           \
            /* explicit two-rounding shape, as the bias gradient always had */ \
            local_d_bias = __fadd_rn(local_d_bias, __fmul_rn(du, silu_grad)); \
            float dxb = dco * weight[d * d_conv + d_conv - 1];                \
            if (d_conv > 1) {                                                 \
                dxb += carry[0];                                              \
                for (int k = 0; k < carry_len - 1; k++) {                     \
                    carry[k] = carry[k + 1]                                   \
                             + dco * weight[d * d_conv + d_conv - 2 - k];     \
                }                                                             \
                carry[carry_len - 1] = dco * weight[d * d_conv];              \
            }                                                                 \
            d_x_branch[(b * T_ + ts) * out_stride + out_offset + d] =         \
                FROM_F(dxb);                                                  \
        } else if (d_conv > 1) {                                              \
            for (int k = 0; k < carry_len - 1; k++) {                         \
                carry[k] = carry[k + 1]                                       \
                         + dco * weight[d * d_conv + d_conv - 2 - k];         \
            }                                                                 \
            carry[carry_len - 1] = dco * weight[d * d_conv];                  \
        }                                                                     \
    }                                                                         \
    for (int k = 0; k < d_conv; k++) {                                        \
        d_weight_partials[row * (d_inner * d_conv) + d * d_conv + k] = dw[k]; \
    }                                                                         \
    d_bias_partials[row * d_inner + d] = local_d_bias;                        \
}                                                                             \

DEFINE_CONV1D_BWD_TILED(f32,  float,         from_f_f32)
DEFINE_CONV1D_BWD_TILED(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_CONV1D_BWD_TILED(f16,  __half,        from_f_f16)
#line 1 "kernels/activations.cu"
// Mamba activation kernels: SiLU + softplus (forward + backward).
//
// Templated over activation dtype via extern "C" wrappers with suffixes:
//   NAME_f32, NAME_bf16, NAME_f16
// Math in f32, storage in T_IN (upcast on load, downcast on store).
// Backward kernels remain f32-only (training path is f32).


// ===================== SiLU forward (templated) =====================

// ===================== Softplus forward (templated) =====================

#define DEFINE_SOFTPLUS_FWD(SUFFIX, T, FROM_F)                            \
extern "C" __global__ void softplus_forward_##SUFFIX(T* x, int n) {       \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (i >= n) return;                                                   \
    float v = to_f(x[i]);                                                 \
    x[i] = FROM_F(v > 20.0f ? v : log1pf(exp2f(v * LOG2E)));         \
}

DEFINE_SOFTPLUS_FWD(f32,  float,         from_f_f32)
DEFINE_SOFTPLUS_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SOFTPLUS_FWD(f16,  __half,        from_f_f16)

// Untyped f32 entry: the training and prefill paths load it as softplus_fwd.
extern "C" __global__ void softplus_forward(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = v > 20.0f ? v : log1pf(exp2f(v * LOG2E));
}

// ===================== Backward (legacy f32 + typed variants) =====================
// Training path stays f32 (atomicAdd on bf16 is sm_90+ only).
// Typed variants exist for future use and parity testing.

extern "C" __global__ void softplus_backward(
    float* dx, const float* x_saved, const float* dy, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dx[i] = dy[i] / (1.0f + exp2f(-x_saved[i] * LOG2E));
}

#define DEFINE_SOFTPLUS_BWD(SUFFIX, T, FROM_F)                            \
extern "C" __global__ void softplus_backward_##SUFFIX(                    \
    T* dx, const T* x_saved, const T* dy, int n                           \
) {                                                                       \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (i >= n) return;                                                   \
    float xs = to_f(x_saved[i]);                                          \
    dx[i] = FROM_F(to_f(dy[i]) / (1.0f + exp2f(-xs * LOG2E)));            \
}

DEFINE_SOFTPLUS_BWD(f32,  float,         from_f_f32)
DEFINE_SOFTPLUS_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SOFTPLUS_BWD(f16,  __half,        from_f_f16)
#line 1 "kernels/norms.cu"
// RMSNorm CUDA kernels (forward + backward).
//
// Each sample processed by one thread block with shared memory reduction.
// Grid: (batch, 1, 1). Block: (min(next_power_of_2(dim), 1024), 1, 1).
// Strided loop handles dim > blockDim.x (e.g., dim=2048 with 1024 threads).
//
// Forward templated over activation dtype. Reduction always in f32 for
// numerical stability - a bf16 mantissa is too short to accumulate a
// row-length sum of squares without visible drift.
// Scale weight stays f32 — it's a model parameter, not an activation.
//
// Reference: Zhang & Sennrich (2019), "Root Mean Square Layer Normalization"


// Register hold depth for the forward kernels: the first RMSN_HOLD strided
// elements per thread stay in registers between the reduction pass and the
// output write, removing the second global read of x whenever
// dim <= RMSN_HOLD * blockDim.x (every shipped d_model). Sum order is
// unchanged (k ascending == i ascending) and out-of-range slots contribute
// +0.0f, which cannot alter the accumulator: it starts at +0.0f and only
// ever adds squares, so it is never -0.0f.
#define RMSN_HOLD 4

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
    float xh[RMSN_HOLD];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < RMSN_HOLD; ++k) {
        int i = d + k * (int)blockDim.x;
        xh[k] = (i < dim) ? x[off + i] : 0.0f;
        sum += xh[k] * xh[k];
    }
    for (int i = d + RMSN_HOLD * (int)blockDim.x; i < dim; i += blockDim.x) {
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
    // Finite-guard: match the typed DEFINE_RMSNORM_FWD variants. Without it,
    // a NaN/Inf anywhere in x produces inv_rms = NaN and contaminates every
    // downstream layer. In f32 training this matters when loss diverges.
    if (!isfinite(rms) || rms < 1e-20f) rms = 1.0f;
    if (d == 0) {
        rms_out[b] = rms;
    }
    __syncthreads();

    float inv_rms = 1.0f / rms;
    // Strided output write from the register-held values; the tail loop
    // re-reads x only past the hold depth.
    #pragma unroll
    for (int k = 0; k < RMSN_HOLD; ++k) {
        int i = d + k * (int)blockDim.x;
        if (i < dim) y[off + i] = xh[k] * inv_rms * scale[i];
    }
    for (int i = d + RMSN_HOLD * (int)blockDim.x; i < dim; i += blockDim.x) {
        y[off + i] = x[off + i] * inv_rms * scale[i];
    }
}

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
    float xh[RMSN_HOLD];                                                     \
    float sum = 0.0f;                                                        \
    _Pragma("unroll")                                                        \
    for (int k = 0; k < RMSN_HOLD; ++k) {                                    \
        int i = d + k * (int)blockDim.x;                                     \
        xh[k] = (i < dim) ? x[off + i] : 0.0f;                               \
        sum += xh[k] * xh[k];                                                \
    }                                                                        \
    for (int i = d + RMSN_HOLD * (int)blockDim.x; i < dim; i += blockDim.x) {\
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
    _Pragma("unroll")                                                        \
    for (int k = 0; k < RMSN_HOLD; ++k) {                                    \
        int i = d + k * (int)blockDim.x;                                     \
        if (i < dim) y[off + i] = FROM_F(xh[k] * inv_rms * scale[i]);        \
    }                                                                        \
    for (int i = d + RMSN_HOLD * (int)blockDim.x; i < dim; i += blockDim.x) {\
        y[off + i] = FROM_F(x[off + i] * inv_rms * scale[i]);                \
    }                                                                        \
}

DEFINE_RMSNORM_FWD_F32IN(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_RMSNORM_FWD_F32IN(f16,  __half,        from_f_f16)

// The residual add of one layer fused into the next layer's RMSNorm: the
// f32 residual takes the branch output (typed) in place, and the norm reads
// the sum from the same registers. The stored sum is `resid + to_f(branch)`
// in that order, exactly what the standalone residual add wrote, and an
// f32 held in a register carries the bits a global round trip would return,
// so the normalized output is the one the two-kernel chain produced. The
// squared-sum expression is spelled as in the plain kernel so it contracts
// the same way. `y` may alias `branch`: every thread reads its branch
// elements before the block's reduction barrier and writes `y` after it.
#define DEFINE_RMSNORM_FWD_RESADD_F32IN(SUFFIX, T_OUT, FROM_F)               \
extern "C" __global__ void rmsnorm_forward_resadd_f32in_##SUFFIX(            \
    T_OUT* y, float* rms_out,                                                \
    float* resid, const T_OUT* branch, const float* scale,                   \
    int batch, int dim, float eps                                            \
) {                                                                          \
    int b = blockIdx.x;                                                      \
    if (b >= batch) return;                                                  \
    int d = threadIdx.x;                                                     \
    extern __shared__ float sdata[];                                         \
    int off = b * dim;                                                       \
    float xh[RMSN_HOLD];                                                     \
    float sum = 0.0f;                                                        \
    _Pragma("unroll")                                                        \
    for (int k = 0; k < RMSN_HOLD; ++k) {                                    \
        int i = d + k * (int)blockDim.x;                                     \
        float v = 0.0f;                                                      \
        if (i < dim) {                                                       \
            v = resid[off + i] + to_f(branch[off + i]);                      \
            resid[off + i] = v;                                              \
        }                                                                    \
        xh[k] = v;                                                           \
        sum += xh[k] * xh[k];                                                \
    }                                                                        \
    for (int i = d + RMSN_HOLD * (int)blockDim.x; i < dim; i += blockDim.x) {\
        float v = resid[off + i] + to_f(branch[off + i]);                    \
        resid[off + i] = v;                                                  \
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
    if (!isfinite(rms) || rms < 1e-20f) rms = 1.0f;                          \
    if (d == 0) rms_out[b] = rms;                                            \
    __syncthreads();                                                         \
    float inv_rms = 1.0f / rms;                                              \
    _Pragma("unroll")                                                        \
    for (int k = 0; k < RMSN_HOLD; ++k) {                                    \
        int i = d + k * (int)blockDim.x;                                     \
        if (i < dim) y[off + i] = FROM_F(xh[k] * inv_rms * scale[i]);        \
    }                                                                        \
    for (int i = d + RMSN_HOLD * (int)blockDim.x; i < dim; i += blockDim.x) {\
        y[off + i] = FROM_F(resid[off + i] * inv_rms * scale[i]);            \
    }                                                                        \
}

DEFINE_RMSNORM_FWD_RESADD_F32IN(f32,  float,         from_f_f32)
DEFINE_RMSNORM_FWD_RESADD_F32IN(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_RMSNORM_FWD_RESADD_F32IN(f16,  __half,        from_f_f16)

// Rule B (no atomicAdd): per-sample per-dim write to `d_scale_partials[b*dim + i]`.
// Caller MUST follow with `reduce_sum_axis0(d_scale, partials, batch, dim, accumulate=1)`
// to finalize the gradient deterministically.
//
// No `__launch_bounds__` — block_dim follows `grid_norm` and can reach 1024 on
// d_model=768/1024/1536/2048/2560 HF checkpoints. The strided accumulation
// loop + shared-memory tree reduce is correct at any power-of-2 block size
// up to 1024.
extern "C" __global__ void rmsnorm_backward(
    float* __restrict__ dx,
    float* __restrict__ d_scale_partials,   // [batch * dim] per-sample per-dim OUTPUT
    const float* __restrict__ dy,
    const float* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ rms_saved,
    int batch, int dim,
    int accumulate, // 1: dx[i] += result (dx = residual grad accumulator)
    // Optional mirror of the dx store for a consumer that would otherwise
    // run a standalone `0.0f + dx` copy pass. nullptr to skip. The f32
    // lane has no such consumer today and always passes nullptr; the
    // parameter keeps the two rmsnorm backward ABIs identical so the one
    // dispatch site can pick either kernel.
    float* __restrict__ dx_typed
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
        float dx_val = (scale[i] * dy_val - x_hat * mean_dy_y) * inv_rms;
        // accumulate=1 folds the old separate vec_add_inplace into this
        // store: same two operands, same per-element order, one launch
        // and one B*T*dm round trip fewer. dx may alias dy when
        // accumulate=0 (norm_f in-place): the sum pass reads all dy
        // before the barrier, and this pass reads dy[off+i] before
        // storing the same element.
        // __fadd_rn pins the two-rounding shape of the old
        // store-then-vec_add pair: without it nvcc contracts the final
        // `* inv_rms` into an FMA with the accumulator (one rounding)
        // and every digest moves.
        float dx_store = accumulate ? __fadd_rn(dx[off + i], dx_val) : dx_val;
        dx[off + i] = dx_store;
        if (dx_typed != nullptr) {
            dx_typed[off + i] = 0.0f + dx_store;
        }
        // Rule B: per-sample per-dim partial (no atomic; reduced externally).
        d_scale_partials[off + i] = dy_val * x_hat;
    }
}

// rmsnorm_backward typed (bf16/f16/f32) for mixed-precision training.
// Pattern matches NVIDIA Apex layer_norm_cuda_kernel.cu and state-spaces/mamba
// reference: load activations cast to f32, reduce mean(dy·ŷ) in f32 shmem,
// store dx as T (downcast).
//
// Rule B (no atomicAdd): per-sample per-dim write to
// `d_scale_partials[b*dim + i]` in f32. Caller MUST follow with
// `reduce_sum_axis0(d_scale, partials, batch, dim, accumulate=1)` to finalize
// the f32 master-grad gradient deterministically.
//
// Inputs:
//   dx                [batch * dim]  T      — output gradient w.r.t. x
//   d_scale_partials  [batch * dim]  float  — per-sample per-dim OUTPUT (f32)
//   dy                [batch * dim]  T      — incoming grad w.r.t. y
//   x                 [batch * dim]  T      — saved input (forward)
//   scale             [dim]          float  — RMS scale weight
//   rms_saved         [batch]        float  — saved RMS scalar per sample
//
// Shared memory: blockDim.x * sizeof(float)  (independent of T — see grid_norm).
//
// No `__launch_bounds__` — see rationale on the f32 `rmsnorm_backward` above.
#define DEFINE_RMSNORM_BWD(SUFFIX, T, FROM_F)                                  \
extern "C" __global__ void rmsnorm_backward_##SUFFIX(                          \
    T* __restrict__ dx, float* __restrict__ d_scale_partials,                  \
    const T* __restrict__ dy, const T* __restrict__ x,                         \
    const float* __restrict__ scale,                                           \
    const float* __restrict__ rms_saved,                                       \
    int batch, int dim                                                         \
) {                                                                            \
    int b = blockIdx.x;                                                        \
    if (b >= batch) return;                                                    \
    int d = threadIdx.x;                                                       \
    extern __shared__ float sdata[];                                           \
    int off = b * dim;                                                         \
    float inv_rms = 1.0f / rms_saved[b];                                       \
    float sum = 0.0f;                                                          \
    for (int i = d; i < dim; i += blockDim.x) {                                \
        float x_hat = to_f(x[off + i]) * inv_rms;                              \
        float dy_val = to_f(dy[off + i]);                                      \
        float y_val = x_hat * scale[i];                                        \
        sum += dy_val * y_val;                                                 \
    }                                                                          \
    sdata[d] = sum;                                                            \
    __syncthreads();                                                           \
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {                   \
        if (d < s) sdata[d] += sdata[d + s];                                   \
        __syncthreads();                                                       \
    }                                                                          \
    if (d < 32) {                                                              \
        float v = sdata[d];                                                    \
        if (d + 32 < blockDim.x) v += sdata[d + 32];                           \
        v = warp_reduce_sum(v);                                                \
        if (d == 0) sdata[0] = v;                                              \
    }                                                                          \
    __syncthreads();                                                           \
    float mean_dy_y = sdata[0] / (float)dim;                                   \
    for (int i = d; i < dim; i += blockDim.x) {                                \
        float x_hat = to_f(x[off + i]) * inv_rms;                              \
        float dy_val = to_f(dy[off + i]);                                      \
        dx[off + i] = FROM_F((scale[i] * dy_val - x_hat * mean_dy_y) * inv_rms); \
        /* Rule B: per-sample per-dim partial (no atomic; reduced externally). */ \
        d_scale_partials[off + i] = dy_val * x_hat;                            \
    }                                                                          \
}

DEFINE_RMSNORM_BWD(f32,  float,         from_f_f32)
DEFINE_RMSNORM_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_RMSNORM_BWD(f16,  __half,        from_f_f16)

// Dual-dtype backward twin of `rmsnorm_forward_f32in_typed`:
//   dy                [batch, dim]   T      — typed upstream gradient
//   x                 [batch, dim]   float  — f32 pre-norm input (residual)
//   scale             [dim]          float  — f32 weight
//   rms_saved         [batch]        float
// Outputs:
//   dx                [batch, dim]   float  — f32 (feeds f32 residual `d_temporal`)
//   d_scale_partials  [batch, dim]   float  — Rule-B per-sample per-dim OUTPUT
//
// Used in the mixed backward per-layer rmsnorm: `d_norm` arrives typed
// from the in_proj dX backward, and must write back into the f32 residual
// stream `d_temporal` via `d_pre_norm` without an intermediate cast kernel.
//
// Rule B (no atomicAdd): caller MUST follow with
// `reduce_sum_axis0(d_scale, partials, batch, dim, accumulate=1)`.
//
// No `__launch_bounds__` — see rationale on the f32 `rmsnorm_backward` above.
#define DEFINE_RMSNORM_BWD_F32IN(SUFFIX, T, FROM_F)                            \
extern "C" __global__ void rmsnorm_backward_f32in_##SUFFIX(                    \
    float* __restrict__ dx, float* __restrict__ d_scale_partials,              \
    const T* __restrict__ dy, const float* __restrict__ x,                     \
    const float* __restrict__ scale,                                           \
    const float* __restrict__ rms_saved,                                       \
    int batch, int dim,                                                        \
    int accumulate,                                                            \
    /* Optional typed mirror of the dx store: the next layer up consumed  \
       dx through a standalone `FROM_F(0.0f + dx)` cast pass, which this   \
       writes inline from the value already in a register. Same           \
       expression on the same value - bit-identical - one launch and one  \
       [B*T*d_model] round trip fewer per layer. Pass nullptr to skip. */  \
    T* __restrict__ dx_typed                                                   \
) {                                                                            \
    int b = blockIdx.x;                                                        \
    if (b >= batch) return;                                                    \
    int d = threadIdx.x;                                                       \
    extern __shared__ float sdata[];                                           \
    int off = b * dim;                                                         \
    float inv_rms = 1.0f / rms_saved[b];                                       \
    float sum = 0.0f;                                                          \
    for (int i = d; i < dim; i += blockDim.x) {                                \
        float x_hat = x[off + i] * inv_rms;                                    \
        float dy_val = to_f(dy[off + i]);                                      \
        float y_val = x_hat * scale[i];                                        \
        sum += dy_val * y_val;                                                 \
    }                                                                          \
    sdata[d] = sum;                                                            \
    __syncthreads();                                                           \
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {                   \
        if (d < s) sdata[d] += sdata[d + s];                                   \
        __syncthreads();                                                       \
    }                                                                          \
    if (d < 32) {                                                              \
        float v = sdata[d];                                                    \
        if (d + 32 < blockDim.x) v += sdata[d + 32];                           \
        v = warp_reduce_sum(v);                                                \
        if (d == 0) sdata[0] = v;                                              \
    }                                                                          \
    __syncthreads();                                                           \
    float mean_dy_y = sdata[0] / (float)dim;                                   \
    for (int i = d; i < dim; i += blockDim.x) {                                \
        float x_hat = x[off + i] * inv_rms;                                    \
        float dy_val = to_f(dy[off + i]);                                      \
        float dx_val = (scale[i] * dy_val - x_hat * mean_dy_y) * inv_rms; \
        float dx_store = accumulate ? __fadd_rn(dx[off + i], dx_val) : dx_val; \
        dx[off + i] = dx_store;                                                \
        if (dx_typed != nullptr) {                                             \
            dx_typed[off + i] = FROM_F(0.0f + dx_store);                       \
        }                                                                      \
        /* Rule B: per-sample per-dim partial (no atomic; reduced externally). */ \
        d_scale_partials[off + i] = dy_val * x_hat;                            \
    }                                                                          \
}

DEFINE_RMSNORM_BWD_F32IN(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_RMSNORM_BWD_F32IN(f16,  __half,        from_f_f16)
#line 1 "kernels/elementwise.cu"
// Element-wise CUDA kernels for Mamba SSM.
//
// Bias broadcast, gating, SSM column gather/scatter, residual add, etc.
// All kernels: 1D grid, 256 threads/block.
//
// Activation-touching kernels are templated via extern "C" wrappers with
// suffixes (_f32, _bf16, _f16). Math in f32, storage in T_IN.


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

// Typed → f32 casts for the M3 mixed-precision backward,
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

// Column sum over the rows. Each column's sum is one serial chain of adds
// in ascending row order held by one owning thread, so the bits are those
// of the plain per-column walk; what changed is how the rows reach it. A
// block covers COLSUM_COLS columns with COLSUM_THREADS threads that stage
// a tile of COLSUM_ROWS rows through shared memory, four rows per thread
// with the next tile's loads in flight while the owners add the current
// one. The walk no longer waits on one load at a time, and a launch of a
// few hundred columns fills a hundred blocks instead of three.
#define COLSUM_COLS 8
#define COLSUM_THREADS 256
#define COLSUM_ROWS 128
extern "C" __global__ __launch_bounds__(COLSUM_THREADS)
void colsum_accumulate(
    float* db, const float* dy,
    int batch, int n_out
) {
    constexpr int TROWS = COLSUM_THREADS / COLSUM_COLS;   // rows a tile step covers
    constexpr int PER_THREAD = COLSUM_ROWS / TROWS;       // rows per thread per tile
    __shared__ float tile[2][COLSUM_ROWS][COLSUM_COLS];
    const int tc = threadIdx.x % COLSUM_COLS;
    const int tr = threadIdx.x / COLSUM_COLS;
    const int j = blockIdx.x * COLSUM_COLS + tc;
    const bool live = j < n_out;
    const int n_tiles = (batch + COLSUM_ROWS - 1) / COLSUM_ROWS;
    float sum = 0.0f;
    _Pragma("unroll")
    for (int q = 0; q < PER_THREAD; q++) {
        int r = tr + q * TROWS;
        tile[0][r][tc] = (live && r < batch)
            ? dy[(long long)r * n_out + j] : 0.0f;
    }
    __syncthreads();
    for (int k = 0; k < n_tiles; k++) {
        const int buf = k & 1;
        // The next tile's rows are requested before this tile's adds and
        // stored after them, into the buffer the adds do not read.
        float next[PER_THREAD];
        _Pragma("unroll")
        for (int q = 0; q < PER_THREAD; q++) {
            int r = (k + 1) * COLSUM_ROWS + tr + q * TROWS;
            next[q] = (k + 1 < n_tiles && live && r < batch)
                ? dy[(long long)r * n_out + j] : 0.0f;
        }
        if (tr == 0) {
            const int rows = min(COLSUM_ROWS, batch - k * COLSUM_ROWS);
            _Pragma("unroll 16")
            for (int r = 0; r < rows; r++) {
                sum += tile[buf][r][tc];
            }
        }
        _Pragma("unroll")
        for (int q = 0; q < PER_THREAD; q++) {
            int r = tr + q * TROWS;
            tile[buf ^ 1][r][tc] = next[q];
        }
        __syncthreads();
    }
    if (tr == 0 && live) db[j] += sum;
}

// Segmented column sum: out[s][j] = sum_t src[s][t][j].
//
// The batched twin of colsum_accumulate for the prefill's pooled route:
// each segment (sample) is summed over its OWN seg_len rows, so samples
// never mix. Per column the accumulation is the identical contract to
// the batch=1 path - ascending t, pure f32 adds seeded at 0.0 - so a
// sample's output is bit-identical whether it rode alone or inside a
// batch. Grid: (ceil(n_out / block), segments).
extern "C" __global__ void colsum_segments(
    float* __restrict__ out,         // [segments * n_out]
    const float* __restrict__ src,   // [segments * seg_len * n_out]
    int segments, int seg_len, int n_out
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (j >= n_out || s >= segments) return;
    const float* base = src + (size_t)s * (size_t)seg_len * (size_t)n_out + j;
    size_t stride = (size_t)n_out;
    float sum = 0.0f;
    /* Same ascending serial chain as colsum_accumulate, loads eight deep. */
    int t = 0;
    for (; t + 8 <= seg_len; t += 8) {
        float s0 = base[(size_t)(t + 0) * stride];
        float s1 = base[(size_t)(t + 1) * stride];
        float s2 = base[(size_t)(t + 2) * stride];
        float s3 = base[(size_t)(t + 3) * stride];
        float s4 = base[(size_t)(t + 4) * stride];
        float s5 = base[(size_t)(t + 5) * stride];
        float s6 = base[(size_t)(t + 6) * stride];
        float s7 = base[(size_t)(t + 7) * stride];
        sum += s0;
        sum += s1;
        sum += s2;
        sum += s3;
        sum += s4;
        sum += s5;
        sum += s6;
        sum += s7;
    }
    for (; t < seg_len; t++) {
        sum += base[(size_t)t * stride];
    }
    out[(size_t)s * (size_t)n_out + j] = sum;
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

extern "C" __global__ void exp_negate(
    float* y, const float* x, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = -exp2f(x[i] * 1.4426950408889634f);
}

// Two-destination exp_negate: the trainer refreshes BOTH a_neg mirrors
// (backward-side and forward-side) from the same master a_log every
// step — one kernel stores the same register twice instead of two
// launches recomputing the same exp.
extern "C" __global__ void exp_negate2(
    float* y0, float* y1, const float* x, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = -exp2f(x[i] * 1.4426950408889634f);
    y0[i] = v;
    y1[i] = v;
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

// Gating forward with the SiLU recomputed from the gate. The SiLU form
// is the DIVISION `g / (1 + exp2(-g*log2e))` the split kernel used - not
// the algebraically equal `g * sigma`, which rounds differently. The gate
// is read through a (row stride, column offset) pair: the gate half of the
// in_proj output in production, so nothing splits it out first.
extern "C" __global__ void gate_mul_silu(
    float* __restrict__ gated,
    const float* __restrict__ y,
    const float* __restrict__ gate_pre,
    int n, int d_inner, int gate_stride, int gate_offset
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = i / d_inner;
    int col = i - row * d_inner;
    float g = gate_pre[row * gate_stride + gate_offset + col];
    gated[i] = y[i] * (g / (1.0f + exp2f(-g * 1.4426950408889634f)));
}

extern "C" __global__ void gating_backward(
    float* d_y,            // [n] gradient w.r.t. SSM output
    float* d_gate_pre,     // [n] gradient w.r.t. gate pre-SiLU
    const float* d_gated,  // [n] incoming gradient
    const float* y,        // [n] SSM output (saved)
    const float* gate_pre, // gate pre-SiLU, read through the same geometry
    int n, int d_inner,
    // Row stride and column offset of the gate half: the gate is read from
    // the in_proj output and its gradient lands directly in the d_proj
    // [bt, 2*d_inner] half, so neither a split nor a concat pass runs.
    int out_stride, int out_offset
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = i / d_inner;
    int col = i - row * d_inner;
    float dg = d_gated[i];
    float x = gate_pre[row * out_stride + out_offset + col];
    // Post-SiLU recomputed in the split kernel's exact form.
    d_y[i] = dg * (x / (1.0f + exp2f(-x * 1.4426950408889634f)));
    // SiLU derivative: sigma * (1 + x * (1 - sigma))
    float sigma = 1.0f / (1.0f + exp2f(-x * 1.4426950408889634f));
    d_gate_pre[row * out_stride + out_offset + col] =
        dg * y[i] * sigma * (1.0f + x * (1.0f - sigma));
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
    const T* gate_pre,                                                         \
    int n, int d_inner, int out_stride, int out_offset                         \
) {                                                                            \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (i >= n) return;                                                        \
    int row = i / d_inner;                                                     \
    int col = i - row * d_inner;                                               \
    float dg = to_f(d_gated[i]);                                               \
    float xv = to_f(gate_pre[row * out_stride + out_offset + col]);            \
    /* Post-SiLU recomputed through the SAME store rounding the deleted    \
       activation carried: FROM_F of the split kernel's division form,     \
       then back to f32 - the exact bits `gate_post[i]` held. */           \
    float gp = to_f(FROM_F(xv / (1.0f + exp2f(-xv * LOG2E))));                 \
    d_y[i] = FROM_F(dg * gp);                                                  \
    float sigma = 1.0f / (1.0f + exp2f(-xv * 1.4426950408889634f));            \
    d_gate_pre[row * out_stride + out_offset + col] =                          \
        FROM_F(dg * to_f(y[i]) * sigma * (1.0f + xv * (1.0f - sigma)));        \
}

DEFINE_GATING_BWD(f32,  float,         from_f_f32)
DEFINE_GATING_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_GATING_BWD(f16,  __half,        from_f_f16)

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

// T-major twin of gather_bc_cols: dst[b][n][t] instead of [b][t][n].
// The parallel scan reads B/C per (d, n) lane over consecutive t; the
// [t][n] layout paid one 32-byte sector per element (61% of the fwd
// kernel by ablation). Pure permutation - identical values.
extern "C" __global__ void gather_bc_cols_tmajor(
    float* dst_b, float* dst_c, const float* src,
    int bt_total, int T, int src_stride, int ds, int b_offset, int c_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bt_total * ds;
    if (idx >= total) return;
    int bt = idx / ds;
    int d = idx % ds;
    int b = bt / T;
    int t = bt % T;
    int row = bt * src_stride;
    dst_b[(b * ds + d) * T + t] = src[row + b_offset + d];
    dst_c[(b * ds + d) * T + t] = src[row + c_offset + d];
}

// Staged-write twin of gather_bc_cols_tmajor. The untiled kernel's writes
// scatter: consecutive lanes vary n for a fixed t, so each dst element in
// the [b][n][t] layout lands one 32-byte sector apart (32 sectors per warp
// for 128 useful bytes). This version stages a GBC_TILE_T x d_state tile
// through shared memory and writes t-contiguous runs per n instead. Pure
// permutation of a copy - identical values, identical bits.
//
// Grid: (ceil(T / GBC_TILE_T), batch). Dynamic smem:
// 2 * ds * (GBC_TILE_T + 1) elements (+1 pad kills bank conflicts on the
// transposed read). The launcher falls back to the untiled kernel when
// that exceeds the default 48 KB static budget (f32 at d_state > 186).
#define GBC_TILE_T 32

extern "C" __global__ void gather_bc_cols_tmajor_tiled(
    float* dst_b, float* dst_c, const float* src,
    int T, int src_stride, int ds, int b_offset, int c_offset
) {
    extern __shared__ float gbc_sm[];
    float* smb = gbc_sm;
    float* smc = gbc_sm + ds * (GBC_TILE_T + 1);
    int b = blockIdx.y;
    int t0 = blockIdx.x * GBC_TILE_T;
    int nt = min(GBC_TILE_T, T - t0);
    int tile_elems = nt * ds;
    for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
        int tt = idx / ds;
        int d = idx % ds;
        int row = (b * T + t0 + tt) * src_stride;
        smb[d * (GBC_TILE_T + 1) + tt] = src[row + b_offset + d];
        smc[d * (GBC_TILE_T + 1) + tt] = src[row + c_offset + d];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
        int d = idx / nt;
        int tt = idx % nt;
        int col = (b * ds + d) * T + t0 + tt;
        dst_b[col] = smb[d * (GBC_TILE_T + 1) + tt];
        dst_c[col] = smc[d * (GBC_TILE_T + 1) + tt];
    }
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

extern "C" __global__ void elementwise_mul(
    float* y, const float* a, const float* b, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] * b[i];
}

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

#define DEFINE_GATHER_BC_TMAJOR(SUFFIX, T_ACT)                                \
extern "C" __global__ void gather_bc_cols_tmajor_##SUFFIX(                    \
    T_ACT* dst_b, T_ACT* dst_c, const T_ACT* src,                             \
    int bt_total, int T, int src_stride, int ds, int b_offset, int c_offset   \
) {                                                                           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    int total = bt_total * ds;                                                \
    if (idx >= total) return;                                                 \
    int bt = idx / ds;                                                        \
    int d = idx % ds;                                                         \
    int b = bt / T;                                                           \
    int t = bt % T;                                                           \
    int row = bt * src_stride;                                                \
    dst_b[(b * ds + d) * T + t] = src[row + b_offset + d];                    \
    dst_c[(b * ds + d) * T + t] = src[row + c_offset + d];                    \
}

DEFINE_GATHER_BC_TMAJOR(f32,  float)
DEFINE_GATHER_BC_TMAJOR(bf16, __nv_bfloat16)
DEFINE_GATHER_BC_TMAJOR(f16,  __half)

/* Typed twin of gather_bc_cols_tmajor_tiled (see the f32 kernel for the
 * staging rationale). Dynamic smem is raw bytes reinterpreted to T_ACT so
 * one extern declaration serves every instantiation. */
#define DEFINE_GATHER_BC_TMAJOR_TILED(SUFFIX, T_ACT)                          \
extern "C" __global__ void gather_bc_cols_tmajor_tiled_##SUFFIX(              \
    T_ACT* dst_b, T_ACT* dst_c, const T_ACT* src,                             \
    int T, int src_stride, int ds, int b_offset, int c_offset                 \
) {                                                                           \
    extern __shared__ unsigned char gbc_sm_raw[];                             \
    T_ACT* smb = (T_ACT*)gbc_sm_raw;                                          \
    T_ACT* smc = smb + ds * (GBC_TILE_T + 1);                                 \
    int b = blockIdx.y;                                                       \
    int t0 = blockIdx.x * GBC_TILE_T;                                         \
    int nt = min(GBC_TILE_T, T - t0);                                         \
    int tile_elems = nt * ds;                                                 \
    for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {        \
        int tt = idx / ds;                                                    \
        int d = idx % ds;                                                     \
        int row = (b * T + t0 + tt) * src_stride;                             \
        smb[d * (GBC_TILE_T + 1) + tt] = src[row + b_offset + d];             \
        smc[d * (GBC_TILE_T + 1) + tt] = src[row + c_offset + d];             \
    }                                                                         \
    __syncthreads();                                                          \
    for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {        \
        int d = idx / nt;                                                     \
        int tt = idx % nt;                                                    \
        int col = (b * ds + d) * T + t0 + tt;                                 \
        dst_b[col] = smb[d * (GBC_TILE_T + 1) + tt];                          \
        dst_c[col] = smc[d * (GBC_TILE_T + 1) + tt];                          \
    }                                                                         \
}

DEFINE_GATHER_BC_TMAJOR_TILED(f32,  float)
DEFINE_GATHER_BC_TMAJOR_TILED(bf16, __nv_bfloat16)
DEFINE_GATHER_BC_TMAJOR_TILED(f16,  __half)

/* Typed gating forward with the SiLU recomputed through its store
 * rounding - reproduces `elementwise_mul(y, gate_post)` exactly. */
#define DEFINE_GATE_MUL_SILU(SUFFIX, T, FROM_F)                               \
extern "C" __global__ void gate_mul_silu_##SUFFIX(                            \
    T* __restrict__ gated, const T* __restrict__ y,                           \
    const T* __restrict__ gate_pre,                                           \
    int n, int d_inner, int gate_stride, int gate_offset                      \
) {                                                                           \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (i >= n) return;                                                       \
    int row = i / d_inner;                                                    \
    int col = i - row * d_inner;                                              \
    float g = to_f(gate_pre[row * gate_stride + gate_offset + col]);          \
    float gp = to_f(FROM_F(g / (1.0f + exp2f(-g * LOG2E))));                   \
    gated[i] = FROM_F(to_f(y[i]) * gp);                                       \
}

DEFINE_GATE_MUL_SILU(f32,  float,         from_f_f32)
DEFINE_GATE_MUL_SILU(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_GATE_MUL_SILU(f16,  __half,        from_f_f16)

// ---------------------------------------------------------------------------
// 16-byte vectorized twins of the hot elementwise kernels.
//
// The class runs at the scalar-copy rate on sm_120 (~2.1 TB/s measured)
// while a uint4-shaped copy of the same bytes reaches ~5.5 TB/s: the
// kernels are instruction-bound on 2-byte accesses, not bandwidth-bound.
// Each thread now moves ONE uint4 per operand - 8 bf16/f16 elements or 4
// f32 - and performs the SAME per-element arithmetic in the SAME order,
// so every output bit is unchanged. The launcher routes here only when
// the element count divides the vector width and every operand pointer is
// 16-byte aligned; otherwise the scalar kernel runs unchanged.
// ---------------------------------------------------------------------------

#define DEFINE_GATE_MUL_SILU_V(SUFFIX, T, FROM_F)                             \
extern "C" __global__ void gate_mul_silu_v_##SUFFIX(                          \
    T* __restrict__ gated, const T* __restrict__ y,                           \
    const T* __restrict__ gate_pre,                                           \
    int n_vec, int d_inner, int gate_stride, int gate_offset                  \
) {                                                                           \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (i >= n_vec) return;                                                   \
    const int NPV = 16 / (int)sizeof(T);                                      \
    /* One vector never straddles a row: the launcher requires the            \
       vector width to divide d_inner. */                                     \
    int e = i * NPV;                                                          \
    int row = e / d_inner;                                                    \
    int col = e - row * d_inner;                                              \
    const T* grow = gate_pre + (size_t)row * gate_stride + gate_offset + col; \
    uint4 yv = reinterpret_cast<const uint4*>(y)[i];                          \
    uint4 gv = *reinterpret_cast<const uint4*>(grow);                         \
    uint4 ov;                                                                 \
    const T* yp = reinterpret_cast<const T*>(&yv);                            \
    const T* gp = reinterpret_cast<const T*>(&gv);                            \
    T* op = reinterpret_cast<T*>(&ov);                                        \
    for (int k = 0; k < NPV; k++) {                                           \
        float g = to_f(gp[k]);                                                \
        float sl = to_f(FROM_F(g / (1.0f + exp2f(-g * LOG2E))));              \
        op[k] = FROM_F(to_f(yp[k]) * sl);                                     \
    }                                                                         \
    reinterpret_cast<uint4*>(gated)[i] = ov;                                  \
}

DEFINE_GATE_MUL_SILU_V(f32,  float,         from_f_f32)
DEFINE_GATE_MUL_SILU_V(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_GATE_MUL_SILU_V(f16,  __half,        from_f_f16)

#define DEFINE_ELEMENTWISE_MUL_V(SUFFIX, T, FROM_F)                           \
extern "C" __global__ void elementwise_mul_v_##SUFFIX(                        \
    T* __restrict__ y, const T* __restrict__ a,                               \
    const T* __restrict__ b, int n_vec                                        \
) {                                                                           \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (i >= n_vec) return;                                                   \
    const int NPV = 16 / (int)sizeof(T);                                      \
    uint4 av = reinterpret_cast<const uint4*>(a)[i];                          \
    uint4 bv = reinterpret_cast<const uint4*>(b)[i];                          \
    uint4 ov;                                                                 \
    const T* ap = reinterpret_cast<const T*>(&av);                            \
    const T* bp = reinterpret_cast<const T*>(&bv);                            \
    T* op = reinterpret_cast<T*>(&ov);                                        \
    for (int k = 0; k < NPV; k++) {                                           \
        op[k] = FROM_F(to_f(ap[k]) * to_f(bp[k]));                            \
    }                                                                         \
    reinterpret_cast<uint4*>(y)[i] = ov;                                      \
}

DEFINE_ELEMENTWISE_MUL_V(f32,  float,         from_f_f32)
DEFINE_ELEMENTWISE_MUL_V(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_ELEMENTWISE_MUL_V(f16,  __half,        from_f_f16)

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

// Typed cast-from-f32 replacing the zero() + vec_add_inplace staging
// idiom (dst = FROM_F(0 + src)). The `0.0f +` is DELIBERATE and must
// stay: it reproduces today's add-on-zeroed-destination bits exactly —
// a bare FROM_F(src[i]) differs at src = -0.0 (+0.0 vs -0.0).
#define DEFINE_VEC_CAST_ZPLUS(SUFFIX, TY, FROM_F)                             \
extern "C" __global__ void vec_cast_zplus_##SUFFIX(                           \
    TY* dst, const float* src, int n                                          \
) {                                                                           \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (i >= n) return;                                                       \
    dst[i] = FROM_F(0.0f + src[i]);                                           \
}

DEFINE_VEC_CAST_ZPLUS(f32,  float,         from_f_f32)
DEFINE_VEC_CAST_ZPLUS(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_VEC_CAST_ZPLUS(f16,  __half,        from_f_f16)

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

// d_xdbl assembly in ONE kernel: the three column ranges [0..dt_rank),
// [dt_rank..dt_rank+ds), [dt_rank+ds..dt_rank+2ds) exactly tile the
// xdbl row, so the old zero + per-range scatter staging (f32 lane:
// 1 memset + 3 scatter_add; mixed lane: 2 memsets + 2 casts + 3
// scatter_add) collapses into a single full-domain store. The dt source
// is already the compute dtype (d_dt_input), B/C sources are the f32
// reduce outputs. Every store keeps the FROM_F(0.0f + x) form —
// bit-identical to the old zero+add chains including the -0.0 class,
// and identical to the old double-round (rounding an already-rounded
// value is the identity).
#define DEFINE_PACK_XDBL_COLS(SUFFIX, TY, FROM_F)                             \
extern "C" __global__ void pack_xdbl_cols_##SUFFIX(                           \
    TY* dst,               /* [batch * (dt_rank + 2*d_state)] */              \
    const TY* dt_src,      /* [batch * dt_rank] */                            \
    const float* b_src,    /* [batch * d_state] */                            \
    const float* c_src,    /* [batch * d_state] */                            \
    int batch, int dt_rank, int d_state                                      \
) {                                                                           \
    int w = dt_rank + 2 * d_state;                                            \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    int total = batch * w;                                                    \
    if (idx >= total) return;                                                 \
    int b = idx / w;                                                          \
    int c = idx % w;                                                          \
    float v;                                                                  \
    if (c < dt_rank) {                                                        \
        v = to_f(dt_src[b * dt_rank + c]);                                    \
    } else if (c < dt_rank + d_state) {                                       \
        v = b_src[b * d_state + (c - dt_rank)];                               \
    } else {                                                                  \
        v = c_src[b * d_state + (c - dt_rank - d_state)];                     \
    }                                                                         \
    dst[idx] = FROM_F(0.0f + v);                                              \
}

DEFINE_PACK_XDBL_COLS(f32,  float,         from_f_f32)
DEFINE_PACK_XDBL_COLS(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_PACK_XDBL_COLS(f16,  __half,        from_f_f16)

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
#line 1 "kernels/loss_scaler.cu"
// Loss-scaling helpers for f16/bf16 mixed-precision training (PyTorch
// GradScaler equivalent — see torch.cuda.amp.GradScaler / NVIDIA Apex AMP).
//
// Workflow:
//   1. CPU side: scaled_loss = loss * scale
//   2. backward(scaled_loss) → produces master grads scaled by `scale`
//   3. GPU: `check_inf_nan_f32` scans every grad buffer, atomicOr into a
//      single device int → 1 if any element is inf/nan
//   4. CPU reads the flag:
//        - overflow → skip optimizer.step(); scaler backs off (scale /= 2)
//        - clean   → `scale_grads_f32(grads, 1/scale)` to unscale, then step;
//                    after `growth_interval` clean steps, scale *= 2
//
// Why f32 only: master gradients are kept in f32 throughout the AMP path
// (bf16/f16 atomicAdd is not supported on ≤sm_89 and reduces precision).
// `scale_grads_f32` accepts a generic multiplier so it can also be used for
// pre-optimizer rescaling (e.g. grad clipping).

extern "C" __global__ void check_inf_nan_f32(
    int* __restrict__ found_overflow,  // [1] device int, atomicOr target
    const float* __restrict__ grads,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int local = 0;
    for (int i = idx; i < n; i += stride) {
        float v = grads[i];
        // Any non-finite triggers overflow. isfinite() returns false for both
        // ±inf and NaN, which is exactly the AMP semantics.
        if (!isfinite(v)) {
            local = 1;
            break;  // one overflow is enough — no need to keep scanning
        }
    }
    // Warp-collapse before HBM atomicOr: worst case
    // every lane has local=1 → 32 atomicOrs per warp → contention.
    // __any_sync collapses to a single ballot and one atomicOr per warp.
    if (__any_sync(0xFFFFFFFFu, local)) {
        if ((threadIdx.x & 31) == 0) {
            atomicOr(found_overflow, 1);
        }
    }
}

// Multiply every gradient in [grads, grads+n) by `scale`. Used both for
// unscaling (scale = 1 / loss_scale) and grad clipping (scale = clip_norm /
// grad_norm). NaN/inf in the input propagates — call check_inf_nan_f32 first
// if you need the AMP-safe path.
extern "C" __global__ void scale_grads_f32(
    float* __restrict__ grads,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        grads[i] *= scale;
    }
}

// Device-scalar twin of `scale_grads_f32`: the factor is read from device
// memory (the clip coefficient the norm kernel just wrote) instead of
// being a launch argument, so the scaling pass can be enqueued without a
// host round trip. Same grid, same element-to-thread mapping, same
// multiply - bit-identical to passing the value by argument.
extern "C" __global__ void scale_grads_dev_f32(
    float* __restrict__ grads,
    const float* __restrict__ scale,
    int n
) {
    float s = scale[0];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        grads[i] *= s;
    }
}

// CUDA-Graph-capturable variant of `scale_grads_f32` that conditionally
// zeros the gradient based on an overflow flag from `check_inf_nan_f32`.
//
//   grads[i] *= (overflow_flag[0] != 0) ? 0.0 : unscale_factor
//
// This lets f16 AMP training capture the full step into a CUDA Graph: the
// graph body always runs the optimizer, but on overflow steps the grads
// are zeroed so AdamW has nothing to apply. CPU reads the flag AFTER
// replay to drive the scaler state machine (backoff vs growth).
//
// ### Divergence from the EAGER f16 path
//
// The EAGER f16 path (trainer.rs::step_f16 non-replay branch) syncs on
// the overflow flag and ACTUALLY skips AdamW + sync_master_to_compute +
// recompute_a_neg when overflow is detected — matching
// `torch.cuda.amp.GradScaler` exactly: on a skipped step master weights,
// m, v, and a_neg all stay at the previous step's values.
//
// The CAPTURED graph path cannot branch mid-graph, so AdamW runs on the
// zeroed grads. The bounded side-effects per overflow step are:
//   * `m_new  = β1 · m_old + (1 − β1) · 0 = β1 · m_old`       (−10% at β1=0.9)
//   * `v_new  = β2 · v_old + (1 − β2) · 0 = β2 · v_old`       (−0.1% at β2=0.999)
//   * `θ_new  = (1 − lr · wd) · θ_old − lr · (m̂ / (√v̂ + ε))` (decoupled
//     weight-decay still fires; with m / v both shrunk by β the Adam term
//     is smaller but non-zero, so θ moves by an amount ~proportional to
//     `m_old · β1`. At a 5–10% overflow rate over 100 k steps this produces
//     a measurable but bounded divergence from the eager path — typically
//     ≤ 0.5 % master-weight drift, never unbounded).
//
// ### a_neg refresh also always runs in the graph
//
// The graph body re-computes `a_neg = -exp(a_log)` after AdamW. On an
// overflow step `a_log` has moved slightly due to the weight-decay-only
// update above, so `a_neg` also moves slightly. The eager path (which
// skips the entire post-forward tail on overflow) does NOT update
// `a_log` or `a_neg` on overflow, so it retains the previous step's
// A-matrix exactly. This is a second source of eager-vs-graph divergence,
// also bounded by the same mechanism.
// `unscale_factor` is read from a 1-element device buffer so the value
// can be updated between graph replays without re-capture. CPU writes
// `1/loss_scale` to `unscale_factor[0]` before each `cuGraphLaunch`.
extern "C" __global__ void scale_grads_skip_f32(
    float* __restrict__ grads,
    const int* __restrict__ overflow_flag,  // [1]
    const float* __restrict__ unscale_factor, // [1] = 1 / loss_scale
    int n
) {
    // Branch on overflow and ASSIGN, not multiply. A multiply-by-zero would
    // propagate NaN: `Inf * 0 = NaN`, `NaN * 0 = NaN`. On overflow the raw
    // grads already contain ±Inf or NaN (that is *why* we overflowed), so
    // `grads[i] *= 0.0f` silently poisons master weights on the next AdamW
    // step. Assignment sanitises the buffer to a clean zero regardless of
    // the current contents, matching PyTorch GradScaler's behaviour of
    // skipping the optimizer step entirely on overflow.
    const int overflow = (overflow_flag[0] != 0);
    const float unscale = unscale_factor[0];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        grads[i] = overflow ? 0.0f : (grads[i] * unscale);
    }
}
#line 1 "kernels/grad_clip.cu"
// Global-norm gradient clipping support.
//
// Deterministic sum-of-squares partial reduction over the flat grad arena:
// FIXED grid (GCLIP_BLOCKS x GCLIP_THREADS) with a fixed-stride grid loop,
// per-thread f64 accumulation, fixed shared-memory tree reduce, one f64
// partial per block. The host performs the final ordered sum of the
// GCLIP_BLOCKS partials — no atomics anywhere, so the norm is bit-stable
// across runs (an atomicAdd norm would be the crate's first determinism
// regression). The actual clip scaling reuses the existing scale_grads_f32
// elementwise kernel.
//
// Section-local geometry constants, #undef'd at end of section: kernel
// sections must own their geometry, never inherit ambient defines.

#define GCLIP_THREADS 256
#define GCLIP_BLOCKS 512

// partials: [GCLIP_BLOCKS] f64, one per block.
// Launch geometry MUST be exactly (GCLIP_BLOCKS, GCLIP_THREADS) — the
// fixed-stride loop and the partial count depend on it.
extern "C" __global__ void grad_sumsq_partial_f32(
    double* __restrict__ partials,
    const float* __restrict__ g,
    int n
) {
    __shared__ double smem[GCLIP_THREADS];
    double acc = 0.0;
    const int stride = GCLIP_BLOCKS * GCLIP_THREADS;
    for (int i = blockIdx.x * GCLIP_THREADS + threadIdx.x; i < n; i += stride) {
        double v = (double)g[i];
        acc += v * v;
    }
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = GCLIP_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = smem[0];
    }
}

// Device-side clip coefficient: folds the ordered f64 partial sum, the
// sqrt and the PyTorch clip_grad_norm_ coefficient into ONE single-thread
// kernel so the host never has to drain the stream between the norm and
// the scaling pass. The fold MUST stay serial and ascending - it is the
// same association the host loop performed, and the norm is part of the
// determinism contract.
//
// Bit contract vs the host path: identical f64 adds in identical order,
// IEEE sqrt, the same f64 divide and the same `coef < 1.0` gate. When the
// gate does not fire the coefficient is exactly 1.0f and the scaling pass
// is a bitwise no-op (x * 1.0f == x for every finite, infinite and NaN
// bit pattern), so always scaling matches the old conditional skip.
// A non-finite norm yields coef = 1.0 (NaN < 1.0 is false), leaving the
// gradients untouched exactly as the host early-return did before the
// caller raises the error.
extern "C" __global__ void grad_clip_coef_f32(
    float* __restrict__ coef_out,        // [1] clip coefficient
    float* __restrict__ norm_out,        // [1] PRE-clip global L2 norm
    const double* __restrict__ partials, // [GCLIP_BLOCKS]
    int n_partials,
    float max_norm
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double sum = 0.0;
    for (int i = 0; i < n_partials; i++) {
        sum += partials[i];
    }
    double norm = sqrt(sum);
    norm_out[0] = (float)norm;
    double coef = (double)max_norm / (norm + 1e-6);
    coef_out[0] = coef < 1.0 ? (float)coef : 1.0f;
}

// Two-block per-layer REGION variants of the pair above: block A is a
// rows x cols stripe with a row stride (a column slice of a row-major
// matrix), block B a contiguous vector; both repeat every layer_stride
// elements for n_layers layers. The region norm rides the same fixed
// geometry, f64 partials and ordered fold as the global one - the
// region clip is part of the same determinism contract.

extern "C" __global__ void grad_region_sumsq_partial_f32(
    double* __restrict__ partials,
    const float* __restrict__ g,
    int n_layers, int layer_stride,
    int a_off, int a_rows, int a_cols, int a_row_stride,
    int b_off, int b_len
) {
    __shared__ double smem[GCLIP_THREADS];
    const int per_layer = a_rows * a_cols + b_len;
    const long long n = (long long)n_layers * per_layer;
    double acc = 0.0;
    const long long stride = (long long)GCLIP_BLOCKS * GCLIP_THREADS;
    for (long long v = (long long)blockIdx.x * GCLIP_THREADS + threadIdx.x; v < n;
         v += stride) {
        int l = (int)(v / per_layer);
        int r = (int)(v % per_layer);
        int off;
        if (r < a_rows * a_cols) {
            int row = r / a_cols;
            int col = r % a_cols;
            off = l * layer_stride + a_off + row * a_row_stride + col;
        } else {
            off = l * layer_stride + b_off + (r - a_rows * a_cols);
        }
        double x = (double)g[off];
        acc += x * x;
    }
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = GCLIP_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = smem[0];
    }
}

// Region scaling by the device-resident coefficient grad_clip_coef_f32
// produced - same index mapping as the region sum, same
// multiply-by-exactly-1.0 no-op contract when the gate did not fire.
extern "C" __global__ void grad_region_scale_dev_f32(
    float* __restrict__ g,
    const float* __restrict__ coef,   // [1]
    int n_layers, int layer_stride,
    int a_off, int a_rows, int a_cols, int a_row_stride,
    int b_off, int b_len
) {
    const float c = coef[0];
    const int per_layer = a_rows * a_cols + b_len;
    const long long n = (long long)n_layers * per_layer;
    const long long stride = (long long)GCLIP_BLOCKS * GCLIP_THREADS;
    for (long long v = (long long)blockIdx.x * GCLIP_THREADS + threadIdx.x; v < n;
         v += stride) {
        int l = (int)(v / per_layer);
        int r = (int)(v % per_layer);
        int off;
        if (r < a_rows * a_cols) {
            int row = r / a_cols;
            int col = r % a_cols;
            off = l * layer_stride + a_off + row * a_row_stride + col;
        } else {
            off = l * layer_stride + b_off + (r - a_rows * a_cols);
        }
        g[off] *= c;
    }
}

#undef GCLIP_THREADS
#undef GCLIP_BLOCKS
#line 1 "kernels/adamw.cu"
// Fused AdamW optimizer step in f32 master precision (PyTorch-AMP convention).
//
// Matches `torch.optim.AdamW` with `capturable=True` numerics and the
// decoupled weight-decay form (Loshchilov & Hutter, "Decoupled Weight Decay
// Regularization", ICLR 2019). Identical update rule to `torch.optim._functional.adamw`
// reference path (PyTorch 2.5, `torch/optim/_functional.py::adamw`):
//
//   m_t = β1·m_{t-1} + (1 - β1)·g_t
//   v_t = β2·v_{t-1} + (1 - β2)·g_t²
//   m̂   = m_t / (1 - β1^t)
//   v̂   = v_t / (1 - β2^t)
//   p_t = p_{t-1} · (1 - lr·wd) - lr · m̂ / (√v̂ + ε)
//
// Bias-correction factors `bias_c1 = 1/(1-β1^t)` and `bias_c2 = 1/(1-β2^t)`
// are computed on the CPU (once per optimizer.step()) and passed in as
// scalars. This avoids per-tensor powf evaluation on the device and is
// what PyTorch's `_single_tensor_adamw` does when `capturable=False`.
//
// Why f32-only: master weights live in f32 (AMP convention — see Micikevicius
// et al., "Mixed Precision Training", ICLR 2018). The optimizer state (m, v)
// must be in f32 to avoid precision collapse after many accumulations; bf16
// Adam accumulators empirically diverge within ~1k steps on SSM-class models.

// CUDA-Graph-capturable variant: bias-correction factors are read from a
// 2-element device buffer instead of taken as scalar kernel args. The CPU
// writes [bc1, bc2] into that buffer (async H2D) BEFORE each graph replay,
// so the captured kernel sees the updated values via a stable device
// pointer. Mirrors PyTorch `torch.optim.AdamW(capturable=True)` semantics
// (PyTorch 2.5, `_multi_tensor_adamw` capturable branch).
extern "C" __global__ void adamw_step_f32_capturable(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    const float* __restrict__ bias_factors,  // [2] = {bc1, bc2}
    int n
) {
    const float bias_c1 = bias_factors[0];
    const float bias_c2 = bias_factors[1];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const float one_minus_b1 = 1.f - beta1;
    const float one_minus_b2 = 1.f - beta2;
    const float decay_factor = 1.f - lr * weight_decay;
    for (int i = idx; i < n; i += stride) {
        float g = grad[i];
        float p = param[i];
        float mi = m[i] * beta1 + one_minus_b1 * g;
        float vi = v[i] * beta2 + one_minus_b2 * g * g;
        m[i] = mi;
        v[i] = vi;
        float m_hat = mi * bias_c1;
        float v_hat = vi * bias_c2;
        param[i] = decay_factor * p - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

extern "C" __global__ void adamw_step_f32(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_c1,   // 1 / (1 - beta1^t)
    float bias_c2,   // 1 / (1 - beta2^t)
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const float one_minus_b1 = 1.f - beta1;
    const float one_minus_b2 = 1.f - beta2;
    const float decay_factor = 1.f - lr * weight_decay;
    for (int i = idx; i < n; i += stride) {
        float g = grad[i];
        float p = param[i];
        float mi = m[i] * beta1 + one_minus_b1 * g;
        float vi = v[i] * beta2 + one_minus_b2 * g * g;
        m[i] = mi;
        v[i] = vi;
        float m_hat = mi * bias_c1;
        float v_hat = vi * bias_c2;
        // Decoupled weight decay: p *= (1 - lr·wd) first, THEN subtract the
        // adam term. This order is what torch.optim.AdamW does — critical
        // so that the effective LR on the regularizer is independent of the
        // adaptive denominator.
        param[i] = decay_factor * p - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}


// ---------------------------------------------------------------------------
// Descriptor-table fused AdamW: ONE launch updates every master tensor.
//
// The per-tensor walk paid 243 kernel launches per optimizer step (plus a
// separate 4-cast-per-layer master->compute sync pass on the mixed lane).
// The chunk table is built ONCE at trainer construction (the flat-arena
// layout is static); each block processes one chunk of one tensor.
//
// Determinism: the update expression per element is IDENTICAL to
// adamw_step_f32_capturable (same operand order, same intrinsics); chunk
// scheduling only changes WHICH block computes an element, never the
// arithmetic — elementwise, no cross-element reduction. The optional
// fused typed write-out stores FROM_F(new_p) of the SAME f32 register
// the master store uses, which is exactly what the old separate cast
// pass produced from memory.
//
// bias_factors is a 3-element device buffer {bc1, bc2, lr}: lr moved
// device-side so a warmup/cosine schedule no longer forfeits the
// captured-graph lane (the old kernels baked lr by value at capture).
// ---------------------------------------------------------------------------

typedef struct {
    unsigned long long param;  // f32*, chunk-offset applied
    unsigned long long grad;   // const f32*
    unsigned long long m;      // f32*
    unsigned long long v;      // f32*
    unsigned long long out;    // typed shadow (0 = none), chunk-offset applied
    unsigned long long n_wd;   // low 32: n elems; high 32: f32 bits of weight_decay
} AdamWChunkDesc;

#define DEFINE_ADAMW_MULTI(SUFFIX, OUT_TY, FROM_F)                             \
extern "C" __global__ void adamw_step_multi_##SUFFIX(                          \
    const AdamWChunkDesc* __restrict__ chunks,                                 \
    float beta1,                                                               \
    float beta2,                                                               \
    float eps,                                                                 \
    const float* __restrict__ bias_factors  /* [3] = {bc1, bc2, lr} */         \
) {                                                                            \
    const AdamWChunkDesc c = chunks[blockIdx.x];                               \
    const float bias_c1 = bias_factors[0];                                     \
    const float bias_c2 = bias_factors[1];                                     \
    const float lr = bias_factors[2];                                          \
    float* param      = (float*)c.param;                                       \
    const float* grad = (const float*)c.grad;                                  \
    float* m          = (float*)c.m;                                           \
    float* v          = (float*)c.v;                                           \
    OUT_TY* out       = (OUT_TY*)c.out;                                        \
    /* Bit 31 of the packed length tags the shadow's dtype: f32-stays-f32 \
       tensors (norms, conv weights/bias, dt bias, D) get their compute   \
       copy written HERE instead of through a per-tensor D2D pass. */     \
    const int n = (int)(c.n_wd & 0x7FFFFFFFULL);                               \
    const bool out_is_f32 = ((c.n_wd >> 31) & 1ULL) != 0ULL;                   \
    const float weight_decay =                                                 \
        __uint_as_float((unsigned int)(c.n_wd >> 32));                         \
    const float one_minus_b1 = 1.f - beta1;                                    \
    const float one_minus_b2 = 1.f - beta2;                                    \
    const float decay_factor = 1.f - lr * weight_decay;                        \
    for (int i = threadIdx.x; i < n; i += blockDim.x) {                        \
        float g = grad[i];                                                     \
        float p = param[i];                                                    \
        float mi = m[i] * beta1 + one_minus_b1 * g;                            \
        float vi = v[i] * beta2 + one_minus_b2 * g * g;                        \
        m[i] = mi;                                                             \
        v[i] = vi;                                                             \
        float m_hat = mi * bias_c1;                                            \
        float v_hat = vi * bias_c2;                                            \
        float np = decay_factor * p - lr * m_hat / (sqrtf(v_hat) + eps);       \
        param[i] = np;                                                         \
        if (out) {                                                             \
            if (out_is_f32) { ((float*)out)[i] = np; }                         \
            else            { out[i] = FROM_F(np); }                           \
        }                                                                      \
    }                                                                          \
}

// FROM_F helpers come from the shared typed prelude (the kernels are
// compiled as one blob with _typed_prelude.cuh inlined first).
DEFINE_ADAMW_MULTI(f32,  float,         from_f_f32)
DEFINE_ADAMW_MULTI(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_ADAMW_MULTI(f16,  __half,        from_f_f16)
#line 1 "kernels/gemm_bi_inference/common.cuh"
// Batch-invariant bf16/f16/f32 GEMM — the single FIXED-TILE family
// (BiGemmFamily::Fixed). Forward-only NN; the triad in gemm_bi_triad.cu
// carries the backward layouts.
//
// Problem: cuBLAS `cublasGemmEx` selects different algorithms per M
// (split-K, tile shape, reduction order). `Y = X @ W` at M=1 vs M=20
// produces sub-ULP differences that amplify through 24 SSM layers
// (observed KL ≈ 0.03, occasional top-1 flip on adversarial prompts).
//
// This kernel is batch-invariant by construction: fixed 64x64x32 tile,
// NO split-K, fixed K-reduction order. The K-reduction tree for
// `C[i, j]` depends ONLY on `A[i, :]` and `B[:, j]`, never on other
// rows of A. Therefore `C[i, j]` is bit-identical whether A has 1, 5,
// or 1024 rows.
//
// Inner GEMM uses Tensor Cores via nvcuda::wmma (m16n16k16 fragments,
// f32 accumulator). The MMA instruction itself is deterministic at
// the hardware level (MMA-Sim arXiv:2511.10909 — 1M random inputs
// bit-identical between simulator and hardware). Non-determinism in
// cuBLAS comes from heuristic algo/Split-K selection, NOT from MMA;
// fixing the tile + Split-K=1 + f32 accumulator is sufficient for
// batch invariance even with Tensor Cores enabled.
//
// The recipe is the standard batch-invariant GEMM shape — one fixed
// tile, grouped launch order, no split-K — written in plain CUDA via
// the WMMA C++ API (no Python/Triton dependency).
//
//   BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32
//   GROUP_M = 8     (L2 swizzle)
//   SPLIT_K = 1     (critical — split-K is the root cause)
//   8 warps/CTA arranged 4 (warp_M) × 2 (warp_N)
//   Per warp: 16M × 32N = 1 frag-M × 2 frag-N (m16n16k16 each)
//   f32 accumulator fragments throughout
//
// Semantics (unchanged from CUDA-core version):
//   A: [M, K]  row-major, element type T_IO  (bf16 or f16)
//   B: [K, N]  row-major, element type T_IO
//   C: [M, N]  row-major, element type T_OUT (bf16 / f16 / f32)
//   bias: nullable [N] f32
//   C = alpha * (A @ B) + beta * C + bias
//
// Launch (unchanged — host dispatcher in src/mamba_ssm/gpu/blas.rs):
//   grid  = ((M/BLOCK_M) * (N/BLOCK_N), 1, 1)  flat — swizzled in-kernel
//   block = (256, 1, 1)
//   smem  = 0 dynamic (all buffers are static __shared__; the TC tile
//           holds 24 KB static - smem_a 4K + smem_b 4K + smem_acc 16K -
//           and the f32 FFMA tile 16 KB). The dynamic K-buffer belongs
//           to matvec_bi_* alone.

#include <mma.h>
#include <cuda_pipeline.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32
#define GROUP_M 8
#define THREADS 256
#define WARPS_PER_CTA 8     // THREADS / 32
#define WARPS_M 4           // 4 warps along M; 4*16 = 64 = BLOCK_M
#define WARPS_N 2           // 2 warps along N; 2*32 = 64 = BLOCK_N
#define FRAG_M 16
#define FRAG_N 16
#define FRAG_K 16
#define WARP_FRAGS_N 2      // each warp owns 2 N-fragments (covers 32 N cols)
#define K_TILES 2           // BLOCK_K / FRAG_K

using namespace nvcuda;

// --- typed zero helpers (used for OOB padding) --------------------------
__device__ __forceinline__ __nv_bfloat16 zero_bf16() {
    return __float2bfloat16(0.0f);
}
__device__ __forceinline__ __half zero_f16() {
    return __float2half(0.0f);
}
__device__ __forceinline__ float zero_f32() { return 0.0f; }

// --- f32 path: emulate Tensor Core via per-element FMA ------------------
// Tensor Cores on Ada do NOT accept f32 inputs (only bf16/f16/tf32). For
// the f32→f32 instantiation we keep the original CUDA-core inner loop.
// f32 inference was never the regression source (cuBLAS f32 path was also
// CUDA cores) so this path is unchanged in performance vs prior commit.
// --- f32 path: emulate Tensor Core via per-element FMA ------------------
// Tensor Cores on Ada do NOT accept f32 inputs (only bf16/f16/tf32). For
// the f32->f32 instantiation we keep the CUDA-core inner loop - the same
// hardware path cuBLAS takes for f32.
//
// The 64x64x32 tile with 256 threads and 4x4 outputs each is a measured
// optimum for narrow-N inference projections; wider tiles, fewer threads
// and a transposed-A layout were all measured slower at those shapes.
// The launcher's block/thread constants MUST equal the ones here - a
// launch that disagrees fills part of the tile and returns plausible
// garbage. `gemm_bi_fixed_correctness` is the gate.
// ---------------------------------------------------------------------------
// GBF safety layer (adopted from the audited fix set; every item is
// bit-preserving - the packed store uses the same per-element RNE, the
// cp.async source clamp changes address formation only, and the base-
// alignment gates route misaligned operands to the scalar stage that
// produces identical smem bytes):
//   - a 16-byte cp.async source pointer is FORMED only when bytes > 0
//     (out-of-object address formation is UB even unread);
//   - the fast stage requires 16B-aligned operand BASES, not just
//     8-element strides (a 2-byte-aligned typed subview with an even
//     stride otherwise stages wrong bytes silently);
//   - packed pair stores check the alignment required by their transport;
//     the f32 overload keeps a scalar fallback for 4-byte subviews.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ bool gbf_aligned16(const void* p) {
    return (reinterpret_cast<unsigned long long>(p) & 15ull) == 0ull;
}
static __device__ __forceinline__ bool gbf_aligned4(const void* p) {
    return (reinterpret_cast<unsigned long long>(p) & 3ull) == 0ull;
}
static __device__ __forceinline__ void gbf_store_pair_rne(
    __nv_bfloat16* dst, float v0, float v1) {
    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
}
static __device__ __forceinline__ void gbf_store_pair_rne(
    __half* dst, float v0, float v1) {
    *reinterpret_cast<__half2*>(dst) = __floats2half2_rn(v0, v1);
}
static __device__ __forceinline__ void gbf_store_pair_rne(
    float* dst, float v0, float v1) {
    if ((reinterpret_cast<unsigned long long>(dst) & 7ull) == 0ull) {
        *reinterpret_cast<float2*>(dst) = make_float2(v0, v1);
    } else {
        dst[0] = v0;
        dst[1] = v1;
    }
}
#line 1 "kernels/gemm_bi_inference/ffma.cu"
#define DEFINE_GEMM_BI_FFMA(NAME, T_IO, T_OUT, FROM_F_OUT, ZERO_IO)             \
extern "C" __global__ __launch_bounds__(THREADS, 2) void                        \
NAME(                                                                           \
    T_OUT* __restrict__ c,                                                      \
    const T_IO* __restrict__ a,                                                 \
    const T_IO* __restrict__ b,                                                 \
    const float* __restrict__ bias,                                             \
    float alpha, float beta,                                                    \
    int m, int n, int k,                                                        \
    int lda, int ldb, int ldc                                                   \
) {                                                                             \
    __shared__ T_IO smem_a[BLOCK_M * BLOCK_K];                                  \
    __shared__ T_IO smem_b[BLOCK_K * BLOCK_N];                                  \
                                                                                \
    int num_pid_m = (m + BLOCK_M - 1) / BLOCK_M;                                \
    int num_pid_n = (n + BLOCK_N - 1) / BLOCK_N;                                \
    int num_pid_in_group = GROUP_M * num_pid_n;                                 \
    int group_id = blockIdx.x / num_pid_in_group;                               \
    int first_pid_m = group_id * GROUP_M;                                       \
    int group_size_m = min(num_pid_m - first_pid_m, GROUP_M);                   \
    int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m); \
    int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;                 \
                                                                                \
    int row0 = pid_m * BLOCK_M;                                                 \
    int col0 = pid_n * BLOCK_N;                                                 \
                                                                                \
    int tx = threadIdx.x & 15;                                                  \
    int ty = threadIdx.x >> 4;                                                  \
    int row_base = row0 + ty * 4;                                               \
    int col_base = col0 + tx * 4;                                               \
                                                                                \
    float acc[4][4];                                                            \
    _Pragma("unroll")                                                           \
    for (int i = 0; i < 4; i++) {                                               \
        _Pragma("unroll")                                                       \
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;                           \
    }                                                                           \
                                                                                \
    for (int k_tile = 0; k_tile < k; k_tile += BLOCK_K) {                       \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < 8; i++) {                                           \
            int idx = i * THREADS + threadIdx.x;                                \
            int smem_r = idx / BLOCK_K;                                         \
            int smem_c = idx % BLOCK_K;                                         \
            int g_r = row0 + smem_r;                                            \
            int g_c = k_tile + smem_c;                                          \
            smem_a[smem_r * BLOCK_K + smem_c] =                                 \
                (g_r < m && g_c < k) ? a[g_r * lda + g_c] : ZERO_IO();          \
        }                                                                       \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < 8; i++) {                                           \
            int idx = i * THREADS + threadIdx.x;                                \
            int smem_r = idx / BLOCK_N;                                         \
            int smem_c = idx % BLOCK_N;                                         \
            int g_r = k_tile + smem_r;                                          \
            int g_c = col0 + smem_c;                                            \
            smem_b[smem_r * BLOCK_N + smem_c] =                                 \
                (g_r < k && g_c < n) ? b[g_r * ldb + g_c] : ZERO_IO();          \
        }                                                                       \
        __syncthreads();                                                        \
                                                                                \
        _Pragma("unroll")                                                       \
        for (int kk = 0; kk < BLOCK_K; kk++) {                                  \
            float a_reg[4];                                                     \
            float b_reg[4];                                                     \
            _Pragma("unroll")                                                   \
            for (int i = 0; i < 4; i++)                                         \
                a_reg[i] = to_f(smem_a[(ty * 4 + i) * BLOCK_K + kk]);           \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < 4; j++)                                         \
                b_reg[j] = to_f(smem_b[kk * BLOCK_N + tx * 4 + j]);             \
            _Pragma("unroll")                                                   \
            for (int i = 0; i < 4; i++) {                                       \
                _Pragma("unroll")                                               \
                for (int j = 0; j < 4; j++) acc[i][j] += a_reg[i] * b_reg[j];   \
            }                                                                   \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
                                                                                \
    _Pragma("unroll")                                                           \
    for (int i = 0; i < 4; i++) {                                               \
        int r = row_base + i;                                                   \
        if (r >= m) continue;                                                   \
        _Pragma("unroll")                                                       \
        for (int j = 0; j < 4; j++) {                                           \
            int col = col_base + j;                                             \
            if (col >= n) continue;                                             \
            float val = __fmul_rn(alpha, acc[i][j]);                    \
            if (bias != nullptr) val = __fadd_rn(val, bias[col]);       \
            if (beta != 0.0f)                                           \
                val = __fmaf_rn(beta, to_f(c[r * ldc + col]), val);     \
            c[r * ldc + col] = FROM_F_OUT(val);                                 \
        }                                                                       \
    }                                                                           \
}

// Experimental exact-f32 mainloop. Every output keeps the legacy tile's
// ascending K order and FFMA chain; staging overlaps the current slab.
#define GBF_F32_THREADS 128
#define GBF_F32_S2_STAGE_ASYNC(BUF, K_TILE)                                    \
    do {                                                                       \
        unsigned _as = (unsigned)__cvta_generic_to_shared(&smem_a[(BUF)][0]);  \
        unsigned _bs = (unsigned)__cvta_generic_to_shared(&smem_b[(BUF)][0]);  \
        for (int _i = threadIdx.x; _i < BLOCK_M * (BLOCK_K / 4);               \
             _i += GBF_F32_THREADS) {                                          \
            int _r = _i / (BLOCK_K / 4);                                      \
            int _c = (_i % (BLOCK_K / 4)) * 4;                                \
            int _gr = row0 + _r;                                               \
            int _gc = (K_TILE) + _c;                                           \
            int _valid = (_gr < m) ? (k - _gc) : 0;                           \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);    \
            unsigned _dst = _as + (unsigned)((_r * BLOCK_K + _c) * 4);        \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&a[(long long)_gr * lda + _gc]                  \
                : (const void*)a;                                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));              \
        }                                                                      \
        for (int _i = threadIdx.x; _i < BLOCK_K * (BLOCK_N / 4);               \
             _i += GBF_F32_THREADS) {                                          \
            int _r = _i / (BLOCK_N / 4);                                      \
            int _c = (_i % (BLOCK_N / 4)) * 4;                                \
            int _gr = (K_TILE) + _r;                                           \
            int _gc = col0 + _c;                                               \
            int _valid = (_gr < k) ? (n - _gc) : 0;                           \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);    \
            unsigned _dst = _bs + (unsigned)((_r * BLOCK_N + _c) * 4);        \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&b[(long long)_gr * ldb + _gc]                  \
                : (const void*)b;                                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));              \
        }                                                                      \
        asm volatile("cp.async.commit_group;\n");                             \
    } while (0)

#define GBF_F32_S2_STAGE_SCALAR(BUF, K_TILE)                                   \
    do {                                                                       \
        for (int _i = threadIdx.x; _i < BLOCK_M * BLOCK_K;                    \
             _i += GBF_F32_THREADS) {                                          \
            int _r = _i / BLOCK_K;                                             \
            int _c = _i % BLOCK_K;                                             \
            int _gr = row0 + _r;                                               \
            int _gc = (K_TILE) + _c;                                           \
            smem_a[(BUF)][_i] = (_gr < m && _gc < k)                          \
                ? a[(long long)_gr * lda + _gc]                                \
                : 0.0f;                                                        \
        }                                                                      \
        for (int _i = threadIdx.x; _i < BLOCK_K * BLOCK_N;                    \
             _i += GBF_F32_THREADS) {                                          \
            int _r = _i / BLOCK_N;                                             \
            int _c = _i % BLOCK_N;                                             \
            int _gr = (K_TILE) + _r;                                           \
            int _gc = col0 + _c;                                               \
            smem_b[(BUF)][_i] = (_gr < k && _gc < n)                           \
                ? b[(long long)_gr * ldb + _gc]                                \
                : 0.0f;                                                        \
        }                                                                      \
    } while (0)

extern "C" __global__ __launch_bounds__(GBF_F32_THREADS, 2) void
f32_f32_s2(
    float* __restrict__ c,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float alpha, float beta,
    int m, int n, int k,
    int lda, int ldb, int ldc
) {
    __align__(16) __shared__ float smem_a[2][BLOCK_M * BLOCK_K];
    __align__(16) __shared__ float smem_b[2][BLOCK_K * BLOCK_N];

    int num_pid_m = (m + BLOCK_M - 1) / BLOCK_M;
    int num_pid_n = (n + BLOCK_N - 1) / BLOCK_N;
    int num_pid_in_group = GROUP_M * num_pid_n;
    int group_id = blockIdx.x / num_pid_in_group;
    int first_pid_m = group_id * GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, GROUP_M);
    int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m);
    int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;
    int row0 = pid_m * BLOCK_M;
    int col0 = pid_n * BLOCK_N;
    int tx = threadIdx.x & 15;
    int ty = threadIdx.x >> 4;
    int row_base = row0 + ty * 8;
    int col_base = col0 + tx * 4;

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    }

    int num_k_tiles = (k + BLOCK_K - 1) / BLOCK_K;
    bool fast_stage = gbf_aligned16(a) && gbf_aligned16(b)
                   && (lda & 3) == 0 && (ldb & 3) == 0;
    if (num_k_tiles > 0) {
        if (fast_stage) {
            GBF_F32_S2_STAGE_ASYNC(0, 0);
        } else {
            GBF_F32_S2_STAGE_SCALAR(0, 0);
        }
    }
    int read_buf = 0;
    for (int kt = 0; kt < num_k_tiles; kt++) {
        if (fast_stage) asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        int next_k = (kt + 1) * BLOCK_K;
        if (kt + 1 < num_k_tiles) {
            if (fast_stage) {
                GBF_F32_S2_STAGE_ASYNC(read_buf ^ 1, next_k);
            } else {
                GBF_F32_S2_STAGE_SCALAR(read_buf ^ 1, next_k);
            }
        }

#pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk++) {
            float a_reg[8];
            float b_reg[4];
#pragma unroll
            for (int i = 0; i < 8; i++)
                a_reg[i] = smem_a[read_buf][(ty * 8 + i) * BLOCK_K + kk];
#pragma unroll
            for (int j = 0; j < 4; j++)
                b_reg[j] = smem_b[read_buf][kk * BLOCK_N + tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] = __fmaf_rn(a_reg[i], b_reg[j], acc[i][j]);
            }
        }
        read_buf ^= 1;
    }

    bool pair_store_fast = row0 <= m - BLOCK_M
        && col0 <= n - BLOCK_N
        && (reinterpret_cast<unsigned long long>(c) & 7ULL) == 0
        && (ldc & 1) == 0;
    if (pair_store_fast) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int r = row_base + i;
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                int col = col_base + j;
                float val0 = __fmul_rn(alpha, acc[i][j]);
                float val1 = __fmul_rn(alpha, acc[i][j + 1]);
                if (bias != nullptr) {
                    val0 = __fadd_rn(val0, bias[col]);
                    val1 = __fadd_rn(val1, bias[col + 1]);
                }
                if (beta != 0.0f) {
                    val0 = __fmaf_rn(beta, c[(long long)r * ldc + col], val0);
                    val1 = __fmaf_rn(beta, c[(long long)r * ldc + col + 1], val1);
                }
                float2 pair = {val0, val1};
                *reinterpret_cast<float2*>(c + (long long)r * ldc + col) = pair;
            }
        }
        return;
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = row_base + i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col_base + j;
            if (col >= n) continue;
            float val = __fmul_rn(alpha, acc[i][j]);
            if (bias != nullptr) val = __fadd_rn(val, bias[col]);
            if (beta != 0.0f)
                val = __fmaf_rn(beta, c[(long long)r * ldc + col], val);
            c[(long long)r * ldc + col] = val;
        }
    }
}

#undef GBF_F32_S2_STAGE_SCALAR
#undef GBF_F32_S2_STAGE_ASYNC
#undef GBF_F32_THREADS

// BEGIN exact-f32 N128 S2 candidate
#define GBF_F32_N128_BM 64
#define GBF_F32_N128_BN 128
#define GBF_F32_N128_BK 32
#define GBF_F32_N128_STAGES 2
#define GBF_F32_N128_THREADS 256

#define GBF_F32_N128_STAGE_ASYNC(BUF, K_TILE)                                  \
    do {                                                                       \
        unsigned _as = (unsigned)__cvta_generic_to_shared(&smem_a[(BUF)][0]);  \
        unsigned _bs = (unsigned)__cvta_generic_to_shared(&smem_b[(BUF)][0]);  \
        for (int _i = threadIdx.x;                                             \
             _i < GBF_F32_N128_BM * (GBF_F32_N128_BK / 4);                    \
             _i += GBF_F32_N128_THREADS) {                                     \
            int _r = _i / (GBF_F32_N128_BK / 4);                              \
            int _c = (_i % (GBF_F32_N128_BK / 4)) * 4;                        \
            int _gr = row0 + _r;                                               \
            int _gc = (K_TILE) + _c;                                           \
            int _valid = (_gr < m) ? (k - _gc) : 0;                           \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);    \
            unsigned _dst = _as                                                \
                + (unsigned)((_r * GBF_F32_N128_BK + _c) * 4);                \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&a[(long long)_gr * lda + _gc]                  \
                : (const void*)a;                                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));              \
        }                                                                      \
        for (int _i = threadIdx.x;                                             \
             _i < GBF_F32_N128_BK * (GBF_F32_N128_BN / 4);                    \
             _i += GBF_F32_N128_THREADS) {                                     \
            int _r = _i / (GBF_F32_N128_BN / 4);                              \
            int _c = (_i % (GBF_F32_N128_BN / 4)) * 4;                        \
            int _gr = (K_TILE) + _r;                                           \
            int _gc = col0 + _c;                                               \
            int _valid = (_gr < k) ? (n - _gc) : 0;                           \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);    \
            unsigned _dst = _bs                                                \
                + (unsigned)((_r * GBF_F32_N128_BN + _c) * 4);                \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&b[(long long)_gr * ldb + _gc]                  \
                : (const void*)b;                                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));              \
        }                                                                      \
        asm volatile("cp.async.commit_group;\n");                             \
    } while (0)

#define GBF_F32_N128_STAGE_SCALAR(BUF, K_TILE)                                 \
    do {                                                                       \
        for (int _i = threadIdx.x;                                             \
             _i < GBF_F32_N128_BM * GBF_F32_N128_BK;                          \
             _i += GBF_F32_N128_THREADS) {                                     \
            int _r = _i / GBF_F32_N128_BK;                                    \
            int _c = _i % GBF_F32_N128_BK;                                    \
            int _gr = row0 + _r;                                               \
            int _gc = (K_TILE) + _c;                                           \
            smem_a[(BUF)][_i] = (_gr < m && _gc < k)                          \
                ? a[(long long)_gr * lda + _gc]                                \
                : 0.0f;                                                        \
        }                                                                      \
        for (int _i = threadIdx.x;                                             \
             _i < GBF_F32_N128_BK * GBF_F32_N128_BN;                          \
             _i += GBF_F32_N128_THREADS) {                                     \
            int _r = _i / GBF_F32_N128_BN;                                    \
            int _c = _i % GBF_F32_N128_BN;                                    \
            int _gr = (K_TILE) + _r;                                           \
            int _gc = col0 + _c;                                               \
            smem_b[(BUF)][_i] = (_gr < k && _gc < n)                          \
                ? b[(long long)_gr * ldb + _gc]                                \
                : 0.0f;                                                        \
        }                                                                      \
    } while (0)

extern "C" __global__ __launch_bounds__(GBF_F32_N128_THREADS, 2) void
f32_f32_n128_s2(
    float* __restrict__ c,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float alpha, float beta,
    int m, int n, int k,
    int lda, int ldb, int ldc
) {
    __align__(16) __shared__ float
        smem_a[GBF_F32_N128_STAGES][GBF_F32_N128_BM * GBF_F32_N128_BK];
    __align__(16) __shared__ float
        smem_b[GBF_F32_N128_STAGES][GBF_F32_N128_BK * GBF_F32_N128_BN];

    int num_pid_m = (m + GBF_F32_N128_BM - 1) / GBF_F32_N128_BM;
    int num_pid_n = (n + GBF_F32_N128_BN - 1) / GBF_F32_N128_BN;
    int num_pid_in_group = GROUP_M * num_pid_n;
    int group_id = blockIdx.x / num_pid_in_group;
    int first_pid_m = group_id * GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, GROUP_M);
    int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m);
    int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;
    int row0 = pid_m * GBF_F32_N128_BM;
    int col0 = pid_n * GBF_F32_N128_BN;
    int tx = threadIdx.x & 31;
    int ty = threadIdx.x >> 5;
    int row_base = row0 + ty * 8;
    int col_base = col0 + tx * 4;

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    }

    int num_k_tiles = (k + GBF_F32_N128_BK - 1) / GBF_F32_N128_BK;
    bool fast_stage = gbf_aligned16(a) && gbf_aligned16(b)
                   && (lda & 3) == 0 && (ldb & 3) == 0;
    if (num_k_tiles > 0) {
        if (fast_stage) {
            GBF_F32_N128_STAGE_ASYNC(0, 0);
        } else {
            GBF_F32_N128_STAGE_SCALAR(0, 0);
        }
    }
    int read_buf = 0;
    for (int kt = 0; kt < num_k_tiles; kt++) {
        if (fast_stage) asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        int next_k = (kt + 1) * GBF_F32_N128_BK;
        if (kt + 1 < num_k_tiles) {
            if (fast_stage) {
                GBF_F32_N128_STAGE_ASYNC(read_buf ^ 1, next_k);
            } else {
                GBF_F32_N128_STAGE_SCALAR(read_buf ^ 1, next_k);
            }
        }

#pragma unroll 8
        for (int kk = 0; kk < GBF_F32_N128_BK; kk++) {
            float a_reg[8];
            float b_reg[4];
#pragma unroll
            for (int i = 0; i < 8; i++)
                a_reg[i] = smem_a[read_buf][(ty * 8 + i) * GBF_F32_N128_BK + kk];
#pragma unroll
            for (int j = 0; j < 4; j++)
                b_reg[j] = smem_b[read_buf][kk * GBF_F32_N128_BN + tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] = __fmaf_rn(a_reg[i], b_reg[j], acc[i][j]);
            }
        }
        read_buf ^= 1;
    }

    bool pair_store_fast = row0 <= m - GBF_F32_N128_BM
        && col0 <= n - GBF_F32_N128_BN
        && (reinterpret_cast<unsigned long long>(c) & 7ULL) == 0
        && (ldc & 1) == 0;
    if (pair_store_fast) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int r = row_base + i;
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                int col = col_base + j;
                float val0 = __fmul_rn(alpha, acc[i][j]);
                float val1 = __fmul_rn(alpha, acc[i][j + 1]);
                if (bias != nullptr) {
                    val0 = __fadd_rn(val0, bias[col]);
                    val1 = __fadd_rn(val1, bias[col + 1]);
                }
                if (beta != 0.0f) {
                    val0 = __fmaf_rn(beta, c[(long long)r * ldc + col], val0);
                    val1 = __fmaf_rn(beta, c[(long long)r * ldc + col + 1], val1);
                }
                float2 pair = {val0, val1};
                *reinterpret_cast<float2*>(c + (long long)r * ldc + col) = pair;
            }
        }
        return;
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = row_base + i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col_base + j;
            if (col >= n) continue;
            float val = __fmul_rn(alpha, acc[i][j]);
            if (bias != nullptr) val = __fadd_rn(val, bias[col]);
            if (beta != 0.0f)
                val = __fmaf_rn(beta, c[(long long)r * ldc + col], val);
            c[(long long)r * ldc + col] = val;
        }
    }
}

#undef GBF_F32_N128_STAGE_SCALAR
#undef GBF_F32_N128_STAGE_ASYNC
#undef GBF_F32_N128_THREADS
#undef GBF_F32_N128_STAGES
#undef GBF_F32_N128_BK
#undef GBF_F32_N128_BN
#undef GBF_F32_N128_BM
// END exact-f32 N128 S2 candidate

// --- Tensor Core path (bf16 / f16) --------------------------------------
// Per-warp arrangement: 8 warps as 4 (along M) × 2 (along N).
//   warp_id = threadIdx.x / 32
//   warp_m  = warp_id / WARPS_N        (0..3)
//   warp_n  = warp_id % WARPS_N        (0..1)
// Each warp owns:
//   M rows: warp_m * 16  ..  warp_m * 16 + 16   (1 frag-M)
//   N cols: warp_n * 32  ..  warp_n * 32 + 32   (2 frag-N)
// K reduction: 2 inner iterations of 16 each.
//
// Smem A is [BLOCK_M, BLOCK_K] row-major; smem B is [BLOCK_K, BLOCK_N] row-major.
// `wmma::load_matrix_sync` reads with the given leading dimension; B is
// row-major from K's perspective so we use `wmma::row_major` and ldB=BLOCK_N.
#line 1 "kernels/gemm_bi_inference/tf32.cu"
// Deterministic TF32 inference GEMM. This is an NN-only numeric mode owned
// by the inference module; its fixed K traversal never changes with M.

struct GbfTf32Params {
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(GbfTf32Params) == 24, "Fixed TF32 parameter ABI drift");
static_assert(alignof(GbfTf32Params) == 4, "Fixed TF32 parameter alignment drift");
static_assert(__is_standard_layout(GbfTf32Params),
              "Fixed TF32 parameters must remain standard layout");

template <int BM, int BN, int Stages, bool CompactXor = false>
struct __align__(16) GbfTf32Storage {
    float a[Stages][BM][CompactXor ? 32 : 36];
    float b[Stages][32][CompactXor ? BN : (BN == 64 ? 72 : 40)];
};

static_assert(sizeof(GbfTf32Storage<128, 64, 2>) == 55296, "M128N64 s2 storage");
static_assert(sizeof(GbfTf32Storage<128, 64, 3>) == 82944, "M128N64 s3 storage");
static_assert(sizeof(GbfTf32Storage<64, 64, 2, true>) == 32768,
              "M64N64 s2 compact XOR storage");
static_assert(sizeof(GbfTf32Storage<64, 64, 3>) == 55296, "M64N64 s3 storage");
static_assert(sizeof(GbfTf32Storage<16, 32, 4>) == 29696, "M16N32 s4 storage");

template <bool CompactXor, int BM, int BN, int Stages>
__device__ __forceinline__ float* gbf_tf32_a_slot(
    GbfTf32Storage<BM, BN, Stages, CompactXor>* storage,
    int stage, int row, int reduction) {
    if constexpr (CompactXor) {
        int chunk = (reduction >> 2) ^ (row & 7);
        return &storage->a[stage][row][chunk * 4 + (reduction & 3)];
    }
    return &storage->a[stage][row][reduction];
}

template <bool CompactXor, int BM, int BN, int Stages>
__device__ __forceinline__ float* gbf_tf32_b_slot(
    GbfTf32Storage<BM, BN, Stages, CompactXor>* storage,
    int stage, int reduction, int column) {
    if constexpr (CompactXor) {
        int chunk = (column >> 2) ^ ((reduction & 3) << 1);
        return &storage->b[stage][reduction][chunk * 4 + (column & 3)];
    }
    return &storage->b[stage][reduction][column];
}

struct GbfTf32Problem {
    float* output;
    const float* a;
    const float* b;
    const float* bias;
    GbfTf32Params params;
    int tile_row;
    int tile_column;
};

template <typename T>
__device__ __forceinline__ const T* gbf_tf32_source(
    const T* base, long long valid_offset, int valid_bytes) {
    return valid_bytes == 0 ? base : base + valid_offset;
}

__device__ __forceinline__ void gbf_tf32_copy_ca(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
}

__device__ __forceinline__ void gbf_tf32_copy_cg(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
                 :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
}

__device__ __forceinline__ unsigned gbf_tf32_rna(float value) {
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ void gbf_tf32_mma_m16n8k8(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

template <bool CompactXor, int BM, int BN, int Stages>
__device__ __forceinline__ void gbf_tf32_stage_scalar(
    GbfTf32Storage<BM, BN, Stages, CompactXor>* storage, int stage,
    const GbfTf32Problem& problem, int reduction_base) {
    for (int linear = (int)threadIdx.x; linear < BM * 32; linear += (int)blockDim.x) {
        int row = linear >> 5;
        int reduction = linear & 31;
        int global_row = problem.tile_row + row;
        int global_reduction = reduction_base + reduction;
        float value = 0.0f;
        if (global_row < problem.params.m && global_reduction < problem.params.k) {
            value = problem.a[(long long)global_row * problem.params.lda + global_reduction];
        }
        *gbf_tf32_a_slot<CompactXor>(storage, stage, row, reduction) = value;
    }
    for (int linear = (int)threadIdx.x; linear < 32 * BN; linear += (int)blockDim.x) {
        int reduction = linear / BN;
        int column = linear - reduction * BN;
        int global_reduction = reduction_base + reduction;
        int global_column = problem.tile_column + column;
        float value = 0.0f;
        if (global_column < problem.params.n && global_reduction < problem.params.k) {
            value = problem.b[(long long)global_reduction * problem.params.ldb + global_column];
        }
        *gbf_tf32_b_slot<CompactXor>(storage, stage, reduction, column) = value;
    }
}

template <int BM>
__device__ __forceinline__ void gbf_tf32_copy_16(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    if constexpr (BM == 16) {
        gbf_tf32_copy_ca(shared_dst, global_src, valid_bytes);
    } else {
        gbf_tf32_copy_cg(shared_dst, global_src, valid_bytes);
    }
}

template <bool CompactXor, int BM, int BN, int Stages>
__device__ __forceinline__ void gbf_tf32_stage_async(
    GbfTf32Storage<BM, BN, Stages, CompactXor>* storage, int stage,
    const GbfTf32Problem& problem, int reduction_base) {
    for (int linear = (int)threadIdx.x; linear < BM * 8; linear += (int)blockDim.x) {
        int row = linear >> 3;
        int reduction = (linear & 7) * 4;
        int global_row = problem.tile_row + row;
        int global_reduction = reduction_base + reduction;
        int valid = global_row < problem.params.m ? problem.params.k - global_reduction : 0;
        valid = valid < 0 ? 0 : (valid > 4 ? 4 : valid);
        int valid_bytes = valid * 4;
        long long offset = (long long)global_row * problem.params.lda + global_reduction;
        const float* source = gbf_tf32_source(
            problem.a, valid_bytes == 0 ? 0 : offset, valid_bytes);
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            gbf_tf32_a_slot<CompactXor>(storage, stage, row, reduction));
        gbf_tf32_copy_16<BM>(destination, source, valid_bytes);
    }
    for (int linear = (int)threadIdx.x; linear < 32 * (BN / 4);
         linear += (int)blockDim.x) {
        int reduction = linear / (BN / 4);
        int column = (linear - reduction * (BN / 4)) * 4;
        int global_reduction = reduction_base + reduction;
        int global_column = problem.tile_column + column;
        int valid = global_reduction < problem.params.k ? problem.params.n - global_column : 0;
        valid = valid < 0 ? 0 : (valid > 4 ? 4 : valid);
        int valid_bytes = valid * 4;
        long long offset = (long long)global_reduction * problem.params.ldb + global_column;
        const float* source = gbf_tf32_source(
            problem.b, valid_bytes == 0 ? 0 : offset, valid_bytes);
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            gbf_tf32_b_slot<CompactXor>(storage, stage, reduction, column));
        gbf_tf32_copy_16<BM>(destination, source, valid_bytes);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int BM, int BN>
__device__ __forceinline__ void gbf_tf32_zero_reduction(
    float* output, const float* bias, const GbfTf32Params& params) {
    int column_tiles = (params.n + BN - 1) / BN;
    int tile_row = (int)blockIdx.x / column_tiles * BM;
    int tile_column = (int)blockIdx.x % column_tiles * BN;
    for (int linear = (int)threadIdx.x; linear < BM * BN; linear += (int)blockDim.x) {
        int row = tile_row + linear / BN;
        int column = tile_column + linear % BN;
        if (row < params.m && column < params.n) {
            output[(long long)row * params.ldc + column] =
                bias == nullptr ? 0.0f : bias[column];
        }
    }
}

template <bool CompactXor, int BM, int BN, int Stages, int MAtoms, int NAtoms>
__device__ __forceinline__ void gbf_tf32_compute_stage(
    GbfTf32Storage<BM, BN, Stages, CompactXor>* storage, int stage,
    int warp_m, int warp_n, int group, int thread,
    float (&accumulators)[MAtoms][NAtoms][4]) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        int k8 = issue * 8;
        unsigned a_fragments[MAtoms][4];
        unsigned b_fragments[NAtoms][2];
#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
            int row = warp_m + m_atom * 16 + group;
            a_fragments[m_atom][0] = gbf_tf32_rna(
                *gbf_tf32_a_slot<CompactXor>(storage, stage, row, k8 + thread));
            a_fragments[m_atom][1] = gbf_tf32_rna(
                *gbf_tf32_a_slot<CompactXor>(storage, stage, row + 8, k8 + thread));
            a_fragments[m_atom][2] = gbf_tf32_rna(
                *gbf_tf32_a_slot<CompactXor>(storage, stage, row, k8 + thread + 4));
            a_fragments[m_atom][3] = gbf_tf32_rna(
                *gbf_tf32_a_slot<CompactXor>(storage, stage, row + 8, k8 + thread + 4));
        }
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
            int column = warp_n + n_atom * 8 + group;
            b_fragments[n_atom][0] = gbf_tf32_rna(
                *gbf_tf32_b_slot<CompactXor>(storage, stage, k8 + thread, column));
            b_fragments[n_atom][1] = gbf_tf32_rna(
                *gbf_tf32_b_slot<CompactXor>(storage, stage, k8 + thread + 4, column));
        }
#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
                gbf_tf32_mma_m16n8k8(
                    accumulators[m_atom][n_atom], a_fragments[m_atom], b_fragments[n_atom]);
            }
        }
    }
}

template <int BM, int BN, int Stages, bool CompactXor = false>
__device__ __forceinline__ void gbf_tf32_kernel(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32Params params) {
    constexpr int MAtoms = BM == 128 ? 4 : (BM == 64 ? 2 : 1);
    constexpr int NAtoms = BM == 16 ? 1 : 4;
    int column_tiles = (params.n + BN - 1) / BN;
    GbfTf32Problem problem = {
        output, a, b, bias, params,
        (int)blockIdx.x / column_tiles * BM,
        (int)blockIdx.x % column_tiles * BN,
    };
    extern __shared__ __align__(16) unsigned char gbf_tf32_shared[];
    auto* storage = reinterpret_cast<GbfTf32Storage<BM, BN, Stages, CompactXor>*>(
        gbf_tf32_shared);

    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    bool compute = BM != 128 || warp < 4;
    int warp_m = BM == 128 ? (warp >> 1) * 64 : (BM == 64 ? (warp >> 1) * 32 : 0);
    int warp_n = BM == 16 ? warp * 8 : (warp & 1) * 32;
    int group = lane >> 2;
    int thread = lane & 3;
    float accumulators[MAtoms][NAtoms][4];

#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = problem.tile_row + warp_m + m_atom * 16
                    + group + (element >= 2 ? 8 : 0);
                int column = problem.tile_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                accumulators[m_atom][n_atom][element] =
                    compute && row < params.m && column < params.n && bias != nullptr
                        ? bias[column]
                        : 0.0f;
            }
        }
    }

    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
    bool fast_stage = gbf_aligned16(a) && gbf_aligned16(b)
        && (params.lda & 3) == 0 && (params.ldb & 3) == 0;
    if (fast_stage) {
#pragma unroll
        for (unsigned tile = 0; tile < Stages - 1; ++tile) {
            if (tile < tile_count) {
                gbf_tf32_stage_async<CompactXor>(
                    storage, static_cast<int>(tile), problem, static_cast<int>(tile * 32U));
            } else {
                asm volatile("cp.async.commit_group;\n" ::);
            }
        }
        for (unsigned tile = 0; tile < tile_count; ++tile) {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(Stages - 2));
            __syncthreads();
            unsigned next = tile + Stages - 1;
            if (next < tile_count) {
                gbf_tf32_stage_async<CompactXor>(
                    storage, static_cast<int>(next % Stages), problem,
                    static_cast<int>(next * 32U));
            } else {
                asm volatile("cp.async.commit_group;\n" ::);
            }
            if (compute) {
                gbf_tf32_compute_stage<CompactXor, BM, BN, Stages, MAtoms, NAtoms>(
                    storage, static_cast<int>(tile % Stages), warp_m, warp_n,
                    group, thread, accumulators);
            }
            __syncthreads();
        }
    } else {
        for (unsigned tile = 0; tile < tile_count; ++tile) {
            int stage = static_cast<int>(tile % Stages);
            gbf_tf32_stage_scalar<CompactXor>(
                storage, stage, problem, static_cast<int>(tile * 32U));
            __syncthreads();
            if (compute) {
                gbf_tf32_compute_stage<CompactXor, BM, BN, Stages, MAtoms, NAtoms>(
                    storage, stage, warp_m, warp_n, group, thread, accumulators);
            }
            __syncthreads();
        }
    }

    if (compute) {
#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
                for (int element = 0; element < 4; ++element) {
                    int row = problem.tile_row + warp_m + m_atom * 16
                        + group + (element >= 2 ? 8 : 0);
                    int column = problem.tile_column + warp_n + n_atom * 8
                        + 2 * thread + (element & 1);
                    if (row < params.m && column < params.n) {
                        output[(long long)row * params.ldc + column] =
                            accumulators[m_atom][n_atom][element];
                    }
                }
            }
        }
    }
}

template <int BM, int BN, int Stages, bool CompactXor = false>
__device__ __forceinline__ void gbf_tf32_entry(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32Params params) {
    if (params.k == 0) {
        gbf_tf32_zero_reduction<BM, BN>(output, bias, params);
        return;
    }
    gbf_tf32_kernel<BM, BN, Stages, CompactXor>(output, a, b, bias, params);
}

#define GBF_TF32_KERNEL(NAME, BM, BN, STAGES, THREADS, MIN_BLOCKS)          \
extern "C" __global__ __launch_bounds__(THREADS, MIN_BLOCKS) void NAME(   \
    float* output, const float* a, const float* b, const float* bias,       \
    GbfTf32Params params) {                                                 \
    gbf_tf32_entry<BM, BN, STAGES>(output, a, b, bias, params);             \
}

GBF_TF32_KERNEL(nn_tf32_m128n64_bk32_s2, 128, 64, 2, 256, 1)
GBF_TF32_KERNEL(nn_tf32_m128n64_bk32_s3, 128, 64, 3, 256, 1)
extern "C" __global__ __launch_bounds__(128, 1)
void nn_tf32_m64n64_bk32_s2(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32Params params) {
    gbf_tf32_entry<64, 64, 2, true>(output, a, b, bias, params);
}
GBF_TF32_KERNEL(nn_tf32_m64n64_bk32_s3, 64, 64, 3, 128, 1)
GBF_TF32_KERNEL(nn_tf32_m16n32_bk32_s4, 16, 32, 4, 128, 3)

#undef GBF_TF32_KERNEL
#line 1 "kernels/gemm_bi_inference/sm120/tf32.cu"
#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)

#if __CUDACC_VER_MAJOR__ >= 13
struct alignas(128) GbfTensorMap {
#else
struct alignas(64) GbfTensorMap {
#endif
    unsigned long long opaque[16];
};

static_assert(sizeof(GbfTensorMap) == 128, "Tensor-map ABI drift");

struct GbfSm120Tf32Params {
    int m;
    int k;
    int n;
    int ldc;
};

static_assert(sizeof(GbfSm120Tf32Params) == 16, "SM120 TF32 parameter ABI drift");
static_assert(__is_standard_layout(GbfSm120Tf32Params),
              "SM120 TF32 parameters must remain standard layout");

template <int M, int N, int Stages>
struct GbfSm120Tf32Storage {
    static constexpr int stage_bytes = (M + N) * 32 * 4;
    static constexpr int dynamic_bytes = 128 + Stages * stage_bytes;
};

template <int M, int N, int WarpN>
struct GbfSm120Tf32Warps {
    static constexpr int threads = (M / 32) * (N / WarpN) * 32;
};

static_assert(GbfSm120Tf32Storage<128, 64, 2>::dynamic_bytes == 49280);
static_assert(GbfSm120Tf32Storage<128, 64, 3>::dynamic_bytes == 73856);
static_assert(GbfSm120Tf32Storage<64, 128, 2>::dynamic_bytes == 49280);
static_assert(GbfSm120Tf32Storage<64, 128, 3>::dynamic_bytes == 73856);
static_assert(GbfSm120Tf32Storage<64, 64, 2>::dynamic_bytes == 32896);

template <int Arrivals>
__device__ __forceinline__ void gbf_sm120_init_barrier(unsigned barrier) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(barrier), "n"(Arrivals) : "memory");
}

__device__ __forceinline__ void gbf_sm120_wait_barrier(
    unsigned barrier, unsigned phase) {
    unsigned ready;
    do {
        asm volatile(
            "{ .reg .pred p; "
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
            "p, [%1], %2; "
            "selp.b32 %0, 1, 0, p; }"
            : "=r"(ready) : "r"(barrier), "r"(phase) : "memory");
    } while (!ready);
}

template <int Bytes>
__device__ __forceinline__ void gbf_sm120_expect_transaction(unsigned barrier) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
        "_, [%0], %1;"
        :: "r"(barrier), "n"(Bytes) : "memory");
}

__device__ __forceinline__ void gbf_sm120_arrive_empty(unsigned barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :: "r"(barrier) : "memory");
}

__device__ __forceinline__ void gbf_sm120_tma_copy(
    unsigned destination, unsigned long long map, int x, int y,
    unsigned barrier) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(destination), "l"(map), "r"(x), "r"(y), "r"(barrier)
        : "memory");
}

__device__ __forceinline__ unsigned gbf_sm120_swizzled_offset(
    unsigned plane_base, unsigned logical_row, unsigned element) {
    unsigned chunk = element / 4U;
    unsigned element_in_vector = element & 3;
    unsigned row_start = plane_base + logical_row * 128U;
    unsigned phase = (row_start / 128U) & 7U;
    unsigned physical_chunk = chunk ^ phase;
    return row_start + physical_chunk * 16U
        + element_in_vector * 4;
}

__device__ __forceinline__ unsigned gbf_sm120_tf32_rna(float value) {
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ void gbf_sm120_tf32_mma(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

template <int M, int N>
__device__ __forceinline__ void gbf_sm120_tf32_zero_reduction(
    float* output, const float* bias, const GbfSm120Tf32Params& params) {
    int column_tiles = (params.n + N - 1) / N;
    int output_row = (int)blockIdx.x / column_tiles * M;
    int output_column = (int)blockIdx.x % column_tiles * N;
    for (int linear = (int)threadIdx.x; linear < M * N; linear += (int)blockDim.x) {
        int row = output_row + linear / N;
        int column = output_column + linear % N;
        if (row < params.m && column < params.n) {
            output[(long long)row * params.ldc + column] =
                bias == nullptr ? 0.0f : bias[column];
        }
    }
}

struct GbfSm120StageContext {
    const GbfTensorMap* a_map;
    const GbfTensorMap* b_map;
    unsigned payload;
    unsigned full_base;
    int output_row;
    int output_column;
};

template <int M, int N, int Stages>
__device__ __forceinline__ void gbf_sm120_produce_stage(
    const GbfSm120StageContext& context, int tile) {
    constexpr int plane_bytes = 32 * 32 * 4;
    constexpr int stage_bytes = GbfSm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned stage_index = (unsigned)(tile % Stages);
    unsigned stage = context.payload + stage_index * stage_bytes;
    unsigned barrier = context.full_base + stage_index * 8;
    unsigned a_destination = stage;
    unsigned b_destination = stage + M * 32 * 4;
    unsigned long long a_descriptor =
        reinterpret_cast<unsigned long long>(context.a_map);
    unsigned long long b_descriptor =
        reinterpret_cast<unsigned long long>(context.b_map);
    int reduction = tile * 32;
    gbf_sm120_expect_transaction<GbfSm120Tf32Storage<M, N, Stages>::stage_bytes>(barrier);
    gbf_sm120_tma_copy(
        a_destination, a_descriptor, reduction, context.output_row, barrier);
#pragma unroll
    for (int plane = 0; plane < N / 32; ++plane) {
        gbf_sm120_tma_copy(
            b_destination + plane * plane_bytes, b_descriptor,
            context.output_column + plane * 32, reduction, barrier);
    }
}

template <int M, int N, int Stages>
__device__ __forceinline__ float gbf_sm120_load_a(
    unsigned char* storage, int stage, int row, int reduction) {
    constexpr int stage_bytes = GbfSm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned plane = (unsigned)(row / 32) * (32 * 32 * 4U);
    unsigned logical_row = (unsigned)(row & 31);
    unsigned offset = gbf_sm120_swizzled_offset(
        plane, logical_row, (unsigned)reduction);
    return *reinterpret_cast<float*>(storage + stage * stage_bytes + offset);
}

template <int M, int N, int Stages>
__device__ __forceinline__ float gbf_sm120_load_b(
    unsigned char* storage, int stage, int reduction, int column) {
    constexpr int stage_bytes = GbfSm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned base = M * 32 * 4;
    unsigned plane = base + (unsigned)(column / 32) * (32 * 32 * 4U);
    unsigned offset = gbf_sm120_swizzled_offset(
        plane, (unsigned)reduction, (unsigned)(column & 31));
    return *reinterpret_cast<float*>(storage + stage * stage_bytes + offset);
}

template <int M, int N, int Stages, int WarpN>
__device__ __forceinline__ void gbf_sm120_load_issue(
    unsigned char* storage, int stage, int warp_m, int warp_n,
    int k8, unsigned (&a_fragments)[2][4],
    unsigned (&b_fragments)[WarpN / 8][2]) {
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
        int row = warp_m + m_atom * 16 + group;
        a_fragments[m_atom][0] = gbf_sm120_tf32_rna(
            gbf_sm120_load_a<M, N, Stages>(storage, stage, row, k8 + thread));
        a_fragments[m_atom][1] = gbf_sm120_tf32_rna(
            gbf_sm120_load_a<M, N, Stages>(
                storage, stage, row + 8, k8 + thread));
        a_fragments[m_atom][2] = gbf_sm120_tf32_rna(
            gbf_sm120_load_a<M, N, Stages>(
                storage, stage, row, k8 + thread + 4));
        a_fragments[m_atom][3] = gbf_sm120_tf32_rna(
            gbf_sm120_load_a<M, N, Stages>(
                storage, stage, row + 8, k8 + thread + 4));
    }
#pragma unroll
    for (int n_atom = 0; n_atom < WarpN / 8; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        b_fragments[n_atom][0] = gbf_sm120_tf32_rna(
            gbf_sm120_load_b<M, N, Stages>(storage, stage, k8 + thread, column));
        b_fragments[n_atom][1] = gbf_sm120_tf32_rna(
            gbf_sm120_load_b<M, N, Stages>(
                storage, stage, k8 + thread + 4, column));
    }
}

template <int M, int N, int Stages, int WarpN>
__device__ __forceinline__ void gbf_sm120_issue_stage(
    unsigned char* storage, int stage, int warp_m, int warp_n,
    float (&accumulator)[2][WarpN / 8][4]) {
    unsigned a_fragments[2][2][4];
    unsigned b_fragments[2][WarpN / 8][2];
    gbf_sm120_load_issue<M, N, Stages, WarpN>(
        storage, stage, warp_m, warp_n, 0, a_fragments[0], b_fragments[0]);
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        int current = issue & 1;
        int next = current ^ 1;
        if (issue + 1 < 4) {
            gbf_sm120_load_issue<M, N, Stages, WarpN>(
                storage, stage, warp_m, warp_n, (issue + 1) * 8,
                a_fragments[next], b_fragments[next]);
        }
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < WarpN / 8; ++n_atom) {
                gbf_sm120_tf32_mma(
                    accumulator[m_atom][n_atom],
                    a_fragments[current][m_atom], b_fragments[current][n_atom]);
            }
        }
    }
}

template <int M, int N, int Stages, int WarpN, bool PairStore,
          bool ProducerWarp>
__device__ __forceinline__ void gbf_sm120_tf32_kernel(
    float* output, const GbfTensorMap& a_map, const GbfTensorMap& b_map,
    const float* bias, const GbfSm120Tf32Params& params) {
    extern __shared__ __align__(1024) unsigned char storage[];
    constexpr int stage_bytes = GbfSm120Tf32Storage<M, N, Stages>::stage_bytes;
    constexpr int compute_warps = GbfSm120Tf32Warps<M, N, WarpN>::threads / 32;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    unsigned payload = shared;
    unsigned full_base = shared + Stages * stage_bytes;
    unsigned empty_base = full_base + 64;
    int column_tiles = (params.n + N - 1) / N;
    int output_row = (int)blockIdx.x / column_tiles * M;
    int output_column = (int)blockIdx.x % column_tiles * N;
    int tile_count = (params.k + 31) / 32;
    const GbfSm120StageContext context = {
        &a_map, &b_map, payload, full_base, output_row, output_column};
    int warp = (int)threadIdx.x >> 5;
    int compute_warp = ProducerWarp ? warp - 1 : warp;
    constexpr int warp_columns = N / WarpN;
    int warp_m = (compute_warp / warp_columns) * 32;
    int warp_n = (compute_warp % warp_columns) * WarpN;
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
    float accumulator[2][WarpN / 8][4];

    if (!ProducerWarp || warp != 0) {
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < WarpN / 8; ++n_atom) {
#pragma unroll
                for (int element = 0; element < 4; ++element) {
                    int column = output_column + warp_n + n_atom * 8
                        + 2 * thread + (element & 1);
                    accumulator[m_atom][n_atom][element] =
                        column < params.n && bias != nullptr ? bias[column] : 0.0f;
                }
            }
        }
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            gbf_sm120_init_barrier<1>(full_base + stage * 8);
            gbf_sm120_init_barrier<compute_warps>(empty_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if constexpr (ProducerWarp) {
        if (warp == 0) {
            if (lane == 0) {
#pragma unroll
                for (int tile = 0; tile < Stages; ++tile) {
                    if (tile < tile_count) {
                        gbf_sm120_produce_stage<M, N, Stages>(context, tile);
                    }
                }
                for (int refill = Stages; refill < tile_count; ++refill) {
                    int consumed = refill - Stages;
                    int stage = consumed % Stages;
                    unsigned generation = (unsigned)(consumed / Stages);
                    gbf_sm120_wait_barrier(
                        empty_base + stage * 8, generation & 1U);
                    gbf_sm120_produce_stage<M, N, Stages>(context, refill);
                }
            }
            return;
        }
    } else {
        if (warp == 0 && lane == 0) {
#pragma unroll
            for (int tile = 0; tile < Stages; ++tile) {
                if (tile < tile_count) {
                    gbf_sm120_produce_stage<M, N, Stages>(context, tile);
                }
            }
        }
        if constexpr (Stages == 2) __syncwarp();
    }

    for (int tile = 0; tile < tile_count; ++tile) {
        int stage = tile % Stages;
        unsigned generation = (unsigned)(tile / Stages);
        gbf_sm120_wait_barrier(full_base + stage * 8, generation & 1U);
        gbf_sm120_issue_stage<M, N, Stages, WarpN>(
            storage, stage, warp_m, warp_n, accumulator);
        if (lane == 0) gbf_sm120_arrive_empty(empty_base + stage * 8);
        if constexpr (!ProducerWarp) {
            if (warp == 0 && lane == 0) {
                int refill = tile + Stages;
                if (refill < tile_count) {
                    gbf_sm120_wait_barrier(empty_base + stage * 8, generation & 1U);
                    gbf_sm120_produce_stage<M, N, Stages>(context, refill);
                }
            }
            if constexpr (Stages == 2) __syncwarp();
        }
    }

    if constexpr (PairStore) {
        bool pair_store_fast = params.m >= M
            && output_row <= params.m - M
            && params.n >= N
            && output_column <= params.n - N
            && (reinterpret_cast<unsigned long long>(output) & 7ULL) == 0
            && (params.ldc & 1) == 0;
        if (pair_store_fast) {
#pragma unroll
            for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
                for (int n_atom = 0; n_atom < WarpN / 8; ++n_atom) {
#pragma unroll
                    for (int element = 0; element < 4; element += 2) {
                        int row = output_row + warp_m + m_atom * 16
                            + group + (element >= 2 ? 8 : 0);
                        int column = output_column + warp_n + n_atom * 8
                            + 2 * thread + (element & 1);
                        float2 pair = {
                            accumulator[m_atom][n_atom][element],
                            accumulator[m_atom][n_atom][element + 1]};
                        *reinterpret_cast<float2*>(
                            output + (long long)row * params.ldc + column) = pair;
                    }
                }
            }
            return;
        }
    }

#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < WarpN / 8; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = output_row + warp_m + m_atom * 16
                    + group + (element >= 2 ? 8 : 0);
                int column = output_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                if (row < params.m && column < params.n) {
                    output[(long long)row * params.ldc + column] =
                        accumulator[m_atom][n_atom][element];
                }
            }
        }
    }
}

template <int M, int N, int Stages, int WarpN, bool PairStore,
          bool ProducerWarp>
__device__ __forceinline__ void gbf_sm120_tf32_entry(
    float* output, const GbfTensorMap& a_map, const GbfTensorMap& b_map,
    const float* bias, const GbfSm120Tf32Params& params) {
    if (params.k == 0) {
        gbf_sm120_tf32_zero_reduction<M, N>(output, bias, params);
        return;
    }
    gbf_sm120_tf32_kernel<M, N, Stages, WarpN, PairStore, ProducerWarp>(
        output, a_map, b_map, bias, params);
}

#define GBF_SM120_TF32_KERNEL(NAME, M, N, STAGES, WARP_N, PAIR_STORE, PRODUCER_WARP) \
extern "C" __global__ __launch_bounds__((M * N) / WARP_N                    \
                                         + (PRODUCER_WARP ? 32 : 0)) void NAME( \
    float* output, const __grid_constant__ GbfTensorMap a_map,                \
    const __grid_constant__ GbfTensorMap b_map, const float* bias,            \
    const __grid_constant__ GbfSm120Tf32Params params) {                      \
    gbf_sm120_tf32_entry<M, N, STAGES, WARP_N, PAIR_STORE, PRODUCER_WARP>(    \
        output, a_map, b_map, bias, params);                                  \
}

GBF_SM120_TF32_KERNEL(
    nn_sm120_tma_tf32_m128n64_bk32_s2, 128, 64, 2, 64, true, false)
GBF_SM120_TF32_KERNEL(
    nn_sm120_tma_tf32_m128n64_bk32_s3, 128, 64, 3, 32, false, false)
GBF_SM120_TF32_KERNEL(
    nn_sm120_tma_tf32_m64n128_bk32_s2, 64, 128, 2, 64, true, false)
GBF_SM120_TF32_KERNEL(
    nn_sm120_tma_tf32_m64n128_bk32_s3, 64, 128, 3, 32, false, false)
GBF_SM120_TF32_KERNEL(
    nn_sm120_tma_tf32_m64n64_bk32_s2_producer_warp,
    64, 64, 2, 32, true, true)
GBF_SM120_TF32_KERNEL(
    nn_sm120_tma_tf32_m64n64_bk32_s2, 64, 64, 2, 32, false, false)
GBF_SM120_TF32_KERNEL(
    nn_sm120_tma_tf32_m64n64_bk32_s2_pair_store,
    64, 64, 2, 32, true, false)

#undef GBF_SM120_TF32_KERNEL
#endif
#line 1 "kernels/gemm_bi_inference/sm120/tma.cu"
// Deterministic SM120 inference GEMM for homogeneous bf16/f16 operands.
// One CTA owns each output tile and walks K in ascending order. There is no
// split-K, inter-CTA reduction, or atomic update anywhere in this file.

#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)

#if __CUDACC_VER_MAJOR__ >= 13
struct alignas(128) GbfSm120HalfTensorMap {
#else
struct alignas(64) GbfSm120HalfTensorMap {
#endif
    unsigned long long opaque[16];
};

static_assert(sizeof(GbfSm120HalfTensorMap) == 128,
              "Fixed SM120 half tensor-map size changed");
#if __CUDACC_VER_MAJOR__ >= 13
static_assert(alignof(GbfSm120HalfTensorMap) == 128,
              "Fixed SM120 half tensor-map alignment changed");
#else
static_assert(alignof(GbfSm120HalfTensorMap) == 64,
              "Fixed SM120 half tensor-map alignment changed");
#endif

struct GbfSm120HalfParams {
    int a_x;
    int a_y;
    int b_x;
    int b_y;
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int ldc;
};

static_assert(sizeof(GbfSm120HalfParams) == 40,
              "Fixed SM120 half parameter size changed");
static_assert(alignof(GbfSm120HalfParams) == 4,
              "Fixed SM120 half parameter alignment changed");
static_assert(__is_standard_layout(GbfSm120HalfParams),
              "Fixed SM120 half parameters must remain standard layout");

#define GBF_SM120_HALF_SWIZZLE_64B 4
#define GBF_SM120_HALF_SWIZZLE_128B 8

template <int M, int N, int BK, int Stages>
struct GbfSm120HalfStorage {
    static constexpr bool wide_m_warp =
        M == 128 && N == 128 && BK == 32;
    static constexpr int compute_warps =
        wide_m_warp ? 8 : (M / 32) * (N / 32);
    static constexpr int threads = compute_warps * 32;
    static constexpr int stage_bytes = (M + N) * BK * 2;
    static constexpr int dynamic_bytes = 128 + Stages * stage_bytes;
};

#define GBF_SM120_HALF_CHECK_STORAGE(M, N, BK, STAGES, BYTES)                \
    static_assert(                                                           \
        GbfSm120HalfStorage<M, N, BK, STAGES>::dynamic_bytes == BYTES,       \
        "Fixed SM120 half dynamic shared size changed")

GBF_SM120_HALF_CHECK_STORAGE(64, 64, 32, 2, 16512);
GBF_SM120_HALF_CHECK_STORAGE(64, 64, 32, 3, 24704);
GBF_SM120_HALF_CHECK_STORAGE(64, 64, 64, 2, 32896);
GBF_SM120_HALF_CHECK_STORAGE(64, 64, 64, 3, 49280);
GBF_SM120_HALF_CHECK_STORAGE(128, 64, 32, 2, 24704);
GBF_SM120_HALF_CHECK_STORAGE(128, 64, 32, 3, 36992);
GBF_SM120_HALF_CHECK_STORAGE(128, 64, 64, 2, 49280);
GBF_SM120_HALF_CHECK_STORAGE(128, 64, 64, 3, 73856);
GBF_SM120_HALF_CHECK_STORAGE(64, 128, 32, 2, 24704);
GBF_SM120_HALF_CHECK_STORAGE(64, 128, 32, 3, 36992);
GBF_SM120_HALF_CHECK_STORAGE(64, 128, 64, 2, 49280);
GBF_SM120_HALF_CHECK_STORAGE(64, 128, 64, 3, 73856);
GBF_SM120_HALF_CHECK_STORAGE(128, 128, 32, 2, 32896);
GBF_SM120_HALF_CHECK_STORAGE(128, 128, 32, 3, 49280);
GBF_SM120_HALF_CHECK_STORAGE(128, 128, 64, 2, 65664);
GBF_SM120_HALF_CHECK_STORAGE(128, 128, 64, 3, 98432);

template <int Arrivals>
static __device__ __forceinline__ void gbf_sm120_half_init_barrier(
    unsigned barrier) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(barrier), "n"(Arrivals) : "memory");
}

static __device__ __forceinline__ void gbf_sm120_half_wait_barrier(
    unsigned barrier, unsigned phase) {
    unsigned ready;
    do {
        asm volatile(
            "{ .reg .pred p; "
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
            "p, [%1], %2; "
            "selp.b32 %0, 1, 0, p; }"
            : "=r"(ready) : "r"(barrier), "r"(phase) : "memory");
    } while (!ready);
}

template <int Bytes>
static __device__ __forceinline__ void gbf_sm120_half_expect_transaction(
    unsigned barrier) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
        "_, [%0], %1;"
        :: "r"(barrier), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void gbf_sm120_half_arrive_empty(
    unsigned barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :: "r"(barrier) : "memory");
}

static __device__ __forceinline__ void gbf_sm120_half_tma_copy(
    unsigned destination, unsigned long long map, int x, int y,
    int origin_x, int origin_y, unsigned barrier) {
    int map_x = x + origin_x;
    int map_y = y + origin_y;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(destination), "l"(map), "r"(map_x), "r"(map_y),
           "r"(barrier)
        : "memory");
}

template <int Groups>
static __device__ __forceinline__ unsigned
gbf_sm120_half_swizzled_address_impl(
    unsigned shared_address, unsigned logical_row, unsigned element) {
    unsigned logical_chunk = element / 8U;
    unsigned row_start = shared_address + logical_row * Groups * 16U;
    unsigned phase = (row_start / 128U) % Groups;
    unsigned physical_chunk = logical_chunk ^ phase;
    return row_start + physical_chunk * 16U;
}

template <int BK>
static __device__ __forceinline__ unsigned gbf_sm120_half_swizzled_address(
    unsigned shared_address, unsigned logical_row, unsigned element) {
    if constexpr (BK == 32) {
        return gbf_sm120_half_swizzled_address_impl<
            GBF_SM120_HALF_SWIZZLE_64B>(
                shared_address, logical_row, element);
    } else {
        static_assert(BK == 64, "Fixed SM120 half BK must be 32 or 64");
        return gbf_sm120_half_swizzled_address_impl<
            GBF_SM120_HALF_SWIZZLE_128B>(
                shared_address, logical_row, element);
    }
}

static __device__ __forceinline__ void gbf_sm120_half_load_x4(
    unsigned address, unsigned (&fragment)[4]) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]),
          "=r"(fragment[3])
        : "r"(address) : "memory");
}

static __device__ __forceinline__ void gbf_sm120_half_load_x2_transpose(
    unsigned address, unsigned (&fragment)[2]) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(fragment[0]), "=r"(fragment[1])
        : "r"(address) : "memory");
}

template <typename T>
struct GbfSm120HalfMma;

template <>
struct GbfSm120HalfMma<__half> {
    static __device__ __forceinline__ void issue(
        float (&accumulator)[4], const unsigned (&a)[4],
        const unsigned (&b)[2]) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};"
            : "+f"(accumulator[0]), "+f"(accumulator[1]),
              "+f"(accumulator[2]), "+f"(accumulator[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]) : "memory");
    }
};

template <>
struct GbfSm120HalfMma<__nv_bfloat16> {
    static __device__ __forceinline__ void issue(
        float (&accumulator)[4], const unsigned (&a)[4],
        const unsigned (&b)[2]) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};"
            : "+f"(accumulator[0]), "+f"(accumulator[1]),
              "+f"(accumulator[2]), "+f"(accumulator[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]) : "memory");
    }
};

struct GbfSm120HalfPipeline {
    unsigned payload;
    unsigned full;
    unsigned empty;
    int output_row;
    int output_column;
};

template <int M, int N, int BK, int Stages>
static __device__ __forceinline__ void gbf_sm120_half_produce_stage(
    const GbfSm120HalfTensorMap& a_map,
    const GbfSm120HalfTensorMap& b_map,
    const GbfSm120HalfParams& params,
    const GbfSm120HalfPipeline& pipeline,
    int tile) {
    constexpr int plane_bytes = BK * BK * 2;
    constexpr int a_bytes = M * BK * 2;
    constexpr int stage_bytes =
        GbfSm120HalfStorage<M, N, BK, Stages>::stage_bytes;
    unsigned stage = pipeline.payload + (tile % Stages) * stage_bytes;
    unsigned a_destination = stage;
    unsigned b_destination = stage + a_bytes;
    unsigned barrier = pipeline.full + (tile % Stages) * 8;
    unsigned long long a_descriptor =
        reinterpret_cast<unsigned long long>(&a_map);
    unsigned long long b_descriptor =
        reinterpret_cast<unsigned long long>(&b_map);
    int reduction = tile * BK;

    gbf_sm120_half_expect_transaction<stage_bytes>(barrier);
    gbf_sm120_half_tma_copy(
        a_destination, a_descriptor, reduction, pipeline.output_row,
        params.a_x, params.a_y, barrier);
    if constexpr (M == 128 && N == 64 && BK == 32) {
        gbf_sm120_half_tma_copy(
            b_destination, b_descriptor, pipeline.output_column, reduction,
            params.b_x, params.b_y, barrier);
    } else {
#pragma unroll
        for (int plane = 0; plane < N / BK; ++plane) {
            int column = pipeline.output_column + plane * BK;
            gbf_sm120_half_tma_copy(
                b_destination + plane * plane_bytes, b_descriptor,
                column, reduction, params.b_x, params.b_y, barrier);
        }
    }
}

template <int M, int N, int BK, int Stages, int MAtoms>
static __device__ __forceinline__ void gbf_sm120_half_load_a_fragments(
    unsigned stage, int warp_m, int slab,
    unsigned (&a_fragment)[MAtoms][4]) {
    constexpr bool compact_addresses =
        M == 128 && N == 128 && BK == 32 && Stages == 2;
    int lane = threadIdx.x & 31;
    int row8 = lane & 7;
    int quadrant = lane >> 3;
    int k0 = slab * 16;

#pragma unroll
    for (int fragment = 0; fragment < MAtoms; ++fragment) {
        int logical_row;
        int element;
        if constexpr (compact_addresses) {
            logical_row = warp_m + fragment * 16 + (lane & 15);
            element = k0 + ((lane & 16) >> 1);
        } else {
            logical_row = warp_m + fragment * 16
                + ((quadrant & 1) ? 8 : 0) + row8;
            element = k0 + ((quadrant & 2) ? 8 : 0);
        }
        unsigned address = gbf_sm120_half_swizzled_address<BK>(
            stage, static_cast<unsigned>(logical_row),
            static_cast<unsigned>(element));
        gbf_sm120_half_load_x4(address, a_fragment[fragment]);
    }
}

template <int M, int N, int BK, int Stages>
static __device__ __forceinline__ void gbf_sm120_half_load_b_fragment(
    unsigned stage, int warp_n, int slab, int fragment,
    unsigned (&b_fragment)[2]) {
    constexpr int plane_bytes = BK * BK * 2;
    constexpr int a_bytes = M * BK * 2;
    constexpr bool compact_addresses =
        (Stages == 2 &&
         ((M == 128 && N == 64) ||
          (M == 128 && N == 128 && BK == 64))) ||
        (Stages == 3 &&
         ((M == 64 && N == 64 && BK == 32) ||
          (M == 64 && N == 128) ||
          (M == 128 && N == 128)));
    unsigned b_base = stage + a_bytes;
    int lane = threadIdx.x & 31;
    int row8 = lane & 7;
    int k0 = slab * 16;
    int logical_row;
    if constexpr (compact_addresses) {
        logical_row = k0 + (lane & 15);
    } else {
        int quadrant = lane >> 3;
        logical_row = k0 + ((quadrant & 1) ? 8 : 0) + row8;
    }
    int output_element = warp_n + fragment * 8;
    unsigned address;
    if constexpr (M == 128 && N == 64 && BK == 32) {
        constexpr int wide_plane_bytes = 64 * BK * 2;
        unsigned plane = b_base + static_cast<unsigned>(
            (output_element / 64) * wide_plane_bytes);
        address = gbf_sm120_half_swizzled_address_impl<
            GBF_SM120_HALF_SWIZZLE_128B>(
                plane, static_cast<unsigned>(logical_row),
                static_cast<unsigned>(output_element % 64));
    } else {
        unsigned plane = b_base + static_cast<unsigned>(
            (output_element / BK) * plane_bytes);
        address = gbf_sm120_half_swizzled_address<BK>(
            plane, static_cast<unsigned>(logical_row),
            static_cast<unsigned>(output_element % BK));
    }
    gbf_sm120_half_load_x2_transpose(address, b_fragment);
}

template <typename T, int M, int N, int BK, int Stages,
          int MAtoms, bool RotatingStage>
static __device__ __forceinline__ void gbf_sm120_half_issue_stage(
    unsigned payload, int stage_or_tile, int warp_m, int warp_n,
    float (&accumulator)[MAtoms][4][4]) {
    constexpr int stage_bytes =
        GbfSm120HalfStorage<M, N, BK, Stages>::stage_bytes;
    constexpr bool lookahead_a = BK == 64 && Stages == 2 && M == 64;
    constexpr bool b_before_next_a = M == 64 && N == 64;
    int stage_index;
    if constexpr (RotatingStage) {
        stage_index = stage_or_tile;
    } else {
        stage_index = stage_or_tile % Stages;
    }
    unsigned stage = payload + stage_index * stage_bytes;
    unsigned a_fragment[lookahead_a ? 2 : 1][MAtoms][4];
    if constexpr (lookahead_a) {
        gbf_sm120_half_load_a_fragments<M, N, BK, Stages, MAtoms>(
            stage, warp_m, 0, a_fragment[0]);
    }

#pragma unroll
    for (int slab = 0; slab < BK / 16; ++slab) {
        int current_a = lookahead_a ? slab & 1 : 0;
        if constexpr (!lookahead_a) {
            gbf_sm120_half_load_a_fragments<M, N, BK, Stages, MAtoms>(
                stage, warp_m, slab, a_fragment[0]);
        }
        if constexpr (lookahead_a && !b_before_next_a) {
            if (slab + 1 < BK / 16) {
                gbf_sm120_half_load_a_fragments<M, N, BK, Stages, MAtoms>(
                    stage, warp_m, slab + 1, a_fragment[current_a ^ 1]);
            }
        }
        if constexpr (N == 64 && BK == 64 && Stages == 2) {
            unsigned b_fragment[2][2];
            gbf_sm120_half_load_b_fragment<M, N, BK, Stages>(
                stage, warp_n, slab, 0, b_fragment[0]);
            if constexpr (lookahead_a && b_before_next_a) {
                if (slab + 1 < BK / 16) {
                    gbf_sm120_half_load_a_fragments<
                        M, N, BK, Stages, MAtoms>(
                            stage, warp_m, slab + 1,
                            a_fragment[current_a ^ 1]);
                }
            }
#pragma unroll
            for (int fragment = 0; fragment < 4; ++fragment) {
                int current = fragment & 1;
                int next = current ^ 1;
                if (fragment + 1 < 4) {
                    gbf_sm120_half_load_b_fragment<M, N, BK, Stages>(
                        stage, warp_n, slab, fragment + 1,
                        b_fragment[next]);
                }
#pragma unroll
                for (int row_fragment = 0;
                     row_fragment < MAtoms;
                     ++row_fragment) {
                    GbfSm120HalfMma<T>::issue(
                        accumulator[row_fragment][fragment],
                        a_fragment[current_a][row_fragment],
                        b_fragment[current]);
                }
            }
        } else {
#pragma unroll
            for (int fragment = 0; fragment < 4; ++fragment) {
                unsigned b_fragment[2];
                gbf_sm120_half_load_b_fragment<M, N, BK, Stages>(
                    stage, warp_n, slab, fragment, b_fragment);
                if constexpr (lookahead_a && b_before_next_a) {
                    if (fragment == 0 && slab + 1 < BK / 16) {
                        gbf_sm120_half_load_a_fragments<
                            M, N, BK, Stages, MAtoms>(
                                stage, warp_m, slab + 1,
                                a_fragment[current_a ^ 1]);
                    }
                }
#pragma unroll
                for (int row_fragment = 0;
                     row_fragment < MAtoms;
                     ++row_fragment) {
                    GbfSm120HalfMma<T>::issue(
                        accumulator[row_fragment][fragment],
                        a_fragment[current_a][row_fragment], b_fragment);
                }
            }
        }
    }
}

template <typename T>
static __device__ __forceinline__ T gbf_sm120_half_from_float(float value);

template <>
__device__ __forceinline__ __half
gbf_sm120_half_from_float<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16
gbf_sm120_half_from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ float
gbf_sm120_half_from_float<float>(float value) {
    return value;
}

static __device__ __forceinline__ float gbf_sm120_half_to_float(__half value) {
    return __half2float(value);
}

static __device__ __forceinline__ float gbf_sm120_half_to_float(
    __nv_bfloat16 value) {
    return __bfloat162float(value);
}

static __device__ __forceinline__ float gbf_sm120_half_to_float(float value) {
    return value;
}

static __device__ __forceinline__ void gbf_sm120_half_store_pair_rne(
    __half* destination, float first, float second) {
    *reinterpret_cast<__half2*>(destination) =
        __floats2half2_rn(first, second);
}

static __device__ __forceinline__ void gbf_sm120_half_store_pair_rne(
    __nv_bfloat16* destination, float first, float second) {
    *reinterpret_cast<__nv_bfloat162*>(destination) =
        __floats2bfloat162_rn(first, second);
}

static __device__ __forceinline__ void gbf_sm120_half_store_pair_rne(
    float* destination, float first, float second) {
    *reinterpret_cast<float2*>(destination) = make_float2(first, second);
}

template <typename T>
struct GbfSm120HalfDirectOutput {
    static constexpr bool value = true;
};

struct GbfSm120HalfOutput {
    void* pointer;
    float alpha;
    float beta;
    int rows;
    int columns;
    int stride;
    int row_tile;
    int column_tile;
    int warp_columns;
};

template <typename T>
static __device__ __forceinline__ void gbf_sm120_half_store_pair(
    const GbfSm120HalfOutput& output, int row, int column,
    float first, float second) {
    long long offset = static_cast<long long>(row) * output.stride + column;
    T* destination = static_cast<T*>(output.pointer) + offset;
    float first_value = first;
    float second_value = second;
    if constexpr (!GbfSm120HalfDirectOutput<T>::value) {
        first_value = __fmul_rn(output.alpha, first);
        second_value = __fmul_rn(output.alpha, second);
        if (output.beta != 0.0f) {
            first_value = __fmaf_rn(
                output.beta, gbf_sm120_half_to_float(destination[0]),
                first_value);
            second_value = __fmaf_rn(
                output.beta, gbf_sm120_half_to_float(destination[1]),
                second_value);
        }
    }
    constexpr unsigned long long pair_alignment = sizeof(T) * 2ULL;
    if ((reinterpret_cast<unsigned long long>(destination)
         & (pair_alignment - 1ULL)) == 0) {
        gbf_sm120_half_store_pair_rne(
            destination, first_value, second_value);
    } else {
        destination[0] = gbf_sm120_half_from_float<T>(first_value);
        destination[1] = gbf_sm120_half_from_float<T>(second_value);
    }
}

template <typename T>
static __device__ __forceinline__ void gbf_sm120_half_store_scalar(
    const GbfSm120HalfOutput& output, int row, int column,
    float accumulator) {
    long long offset = static_cast<long long>(row) * output.stride + column;
    T* destination = static_cast<T*>(output.pointer) + offset;
    float value = accumulator;
    if constexpr (!GbfSm120HalfDirectOutput<T>::value) {
        value = __fmul_rn(output.alpha, accumulator);
        if (output.beta != 0.0f) {
            value = __fmaf_rn(
                output.beta, gbf_sm120_half_to_float(destination[0]), value);
        }
    }
    destination[0] = gbf_sm120_half_from_float<T>(value);
}

template <typename T, int M, int N, int MAtoms>
static __device__ __forceinline__ void gbf_sm120_half_epilogue_full(
    const GbfSm120HalfOutput& output,
    const float (&accumulator)[MAtoms][4][4]) {
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int pair = lane & 3;
    int warp = threadIdx.x >> 5;
    int warp_row = output.row_tile
        + (warp / (N / 32)) * (MAtoms * 16);
    int warp_column = output.column_tile
        + (warp % (N / 32)) * 32 + pair * 2;

#pragma unroll
    for (int row_fragment = 0;
         row_fragment < MAtoms;
         ++row_fragment) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int row = warp_row + row_fragment * 16 + group + half * 8;
            T* row_destination = static_cast<T*>(output.pointer)
                + static_cast<long long>(row) * output.stride
                + warp_column;
            int element = half * 2;
#pragma unroll
            for (int column_fragment = 0;
                 column_fragment < 4;
                 ++column_fragment) {
                T* destination = row_destination + column_fragment * 8;
                float first =
                    accumulator[row_fragment][column_fragment][element];
                float second =
                    accumulator[row_fragment][column_fragment][element + 1];
                if constexpr (!GbfSm120HalfDirectOutput<T>::value) {
                    first = __fmul_rn(output.alpha, first);
                    second = __fmul_rn(output.alpha, second);
                    if (output.beta != 0.0f) {
                        first = __fmaf_rn(
                            output.beta,
                            gbf_sm120_half_to_float(destination[0]), first);
                        second = __fmaf_rn(
                            output.beta,
                            gbf_sm120_half_to_float(destination[1]), second);
                    }
                }
                gbf_sm120_half_store_pair_rne(destination, first, second);
            }
        }
    }
}

template <typename T, int M, int N, int BK, int Stages, int MAtoms>
static __device__ __forceinline__ void gbf_sm120_half_epilogue(
    const GbfSm120HalfOutput& output,
    const float (&accumulator)[MAtoms][4][4]) {
    constexpr bool row_major =
        !(M == 64 && N == 64 && BK == 32 && Stages == 3);
    if constexpr (row_major) {
        constexpr unsigned long long pair_alignment = sizeof(T) * 2ULL;
        bool full = output.row_tile <= output.rows - M
            && output.column_tile <= output.columns - N
            && (reinterpret_cast<unsigned long long>(output.pointer)
                & (pair_alignment - 1ULL)) == 0
            && (output.stride & 1) == 0;
        if (full) {
            gbf_sm120_half_epilogue_full<T, M, N, MAtoms>(
                output, accumulator);
            return;
        }
    }

    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int pair = lane & 3;
    int warp = threadIdx.x >> 5;
#pragma unroll
    for (int row_fragment = 0;
         row_fragment < MAtoms;
         ++row_fragment) {
#pragma unroll
        for (int column_fragment = 0;
             column_fragment < 4;
             ++column_fragment) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = output.row_tile
                    + (warp / output.warp_columns) * (MAtoms * 16)
                    + row_fragment * 16 + group + half * 8;
                int column = output.column_tile
                    + (warp % output.warp_columns) * 32
                    + column_fragment * 8 + pair * 2;
                if (row >= output.rows || column >= output.columns) continue;
                int element = half * 2;
                if (column + 1 < output.columns) {
                    gbf_sm120_half_store_pair<T>(
                        output, row, column,
                        accumulator[row_fragment][column_fragment][element],
                        accumulator[row_fragment][column_fragment][element + 1]);
                } else {
                    gbf_sm120_half_store_scalar<T>(
                        output, row, column,
                        accumulator[row_fragment][column_fragment][element]);
                }
            }
        }
    }
}

template <int MAtoms>
static __device__ __forceinline__ void gbf_sm120_half_initialize_accumulator(
    float (&accumulator)[MAtoms][4][4], const float* bias,
    int columns, int output_column, int warp_n) {
    int pair = threadIdx.x & 3;
#pragma unroll
    for (int row_fragment = 0;
         row_fragment < MAtoms;
         ++row_fragment) {
#pragma unroll
        for (int column_fragment = 0;
             column_fragment < 4;
             ++column_fragment) {
            float first = 0.0f;
            float second = 0.0f;
            if (bias != nullptr) {
                int column = output_column + warp_n
                    + column_fragment * 8 + pair * 2;
                if (column < columns) first = bias[column];
                if (column + 1 < columns) second = bias[column + 1];
            }
            accumulator[row_fragment][column_fragment][0] = first;
            accumulator[row_fragment][column_fragment][1] = second;
            accumulator[row_fragment][column_fragment][2] = first;
            accumulator[row_fragment][column_fragment][3] = second;
        }
    }
}

template <typename TInput, typename TOutput, int M, int N, int BK, int Stages>
static __device__ __forceinline__ void gbf_sm120_half_kernel(
    void* output,
    const GbfSm120HalfTensorMap& a_map,
    const GbfSm120HalfTensorMap& b_map,
    const float* bias,
    const GbfSm120HalfParams& params) {
    constexpr int threads =
        GbfSm120HalfStorage<M, N, BK, Stages>::threads;
    constexpr int warps = threads / 32;
    constexpr int row_fragments =
        GbfSm120HalfStorage<M, N, BK, Stages>::wide_m_warp ? 4 : 2;
    constexpr bool rotating_stage =
        (M == 64 && N == 128) ||
        (M == 128 && N == 64 && Stages == 2) ||
        (M == 128 && N == 128 && BK == 64 && Stages == 3);
    extern __shared__ __align__(128) unsigned char storage[];
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    int column_tiles = 1 + (params.n - 1) / N;
    int output_row = (blockIdx.x / column_tiles) * M;
    int output_column = (blockIdx.x % column_tiles) * N;
    int tile_count = 1 + (params.k - 1) / BK;
    GbfSm120HalfPipeline pipeline = {
        shared + 128,
        shared,
        shared + 64,
        output_row,
        output_column,
    };
    int warp = threadIdx.x >> 5;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            gbf_sm120_half_init_barrier<1>(
                pipeline.full + stage * 8);
            gbf_sm120_half_init_barrier<warps>(
                pipeline.empty + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp == 0 && (threadIdx.x & 31) == 0) {
#pragma unroll
        for (int tile = 0; tile < Stages; ++tile) {
            if (tile < tile_count) {
                gbf_sm120_half_produce_stage<M, N, BK, Stages>(
                    a_map, b_map, params, pipeline, tile);
            }
        }
    }
    __syncwarp();

    int warp_columns = N / 32;
    int warp_m = (warp / warp_columns) * (row_fragments * 16);
    int warp_n = (warp % warp_columns) * 32;
    float accumulator[row_fragments][4][4];
    gbf_sm120_half_initialize_accumulator<row_fragments>(
        accumulator, bias, params.n, output_column, warp_n);
    int rotating_stage_index = 0;
    unsigned rotating_phase = 0;
    for (int tile = 0; tile < tile_count; ++tile) {
        int stage;
        unsigned phase;
        if constexpr (rotating_stage) {
            stage = rotating_stage_index;
            phase = rotating_phase;
        } else {
            stage = tile % Stages;
            phase = static_cast<unsigned>(tile / Stages) & 1U;
        }
        gbf_sm120_half_wait_barrier(
            pipeline.full + stage * 8, phase);
        gbf_sm120_half_issue_stage<
            TInput, M, N, BK, Stages, row_fragments, rotating_stage>(
                pipeline.payload, rotating_stage ? stage : tile,
                warp_m, warp_n, accumulator);
        __syncwarp();
        if ((threadIdx.x & 31) == 0) {
            gbf_sm120_half_arrive_empty(
                pipeline.empty + stage * 8);
        }
        if (warp == 0 && (threadIdx.x & 31) == 0) {
            int refill = tile + Stages;
            if (refill < tile_count) {
                gbf_sm120_half_wait_barrier(
                    pipeline.empty + stage * 8, phase);
                gbf_sm120_half_produce_stage<M, N, BK, Stages>(
                    a_map, b_map, params, pipeline, refill);
            }
        }
        __syncwarp();
        if constexpr (rotating_stage) {
            if (++rotating_stage_index == Stages) {
                rotating_stage_index = 0;
                rotating_phase ^= 1U;
            }
        }
    }

    GbfSm120HalfOutput destination = {
        output,
        params.alpha,
        params.beta,
        params.m,
        params.n,
        params.ldc,
        output_row,
        output_column,
        warp_columns,
    };
    gbf_sm120_half_epilogue<
        TOutput, M, N, BK, Stages, row_fragments>(
            destination, accumulator);
}

#define GBF_SM120_HALF_DEFINE_KERNEL(                                        \
    NAME, INPUT_TYPE, OUTPUT_TYPE, M, N, BK, STAGES)                         \
    extern "C" __global__                                                    \
    __launch_bounds__(GbfSm120HalfStorage<M, N, BK, STAGES>::threads)        \
    void NAME(                                                               \
        void* output,                                                        \
        const __grid_constant__ GbfSm120HalfTensorMap a_map,                 \
        const __grid_constant__ GbfSm120HalfTensorMap b_map,                 \
        const float* bias,                                                   \
        const __grid_constant__ GbfSm120HalfParams params) {                 \
        gbf_sm120_half_kernel<INPUT_TYPE, OUTPUT_TYPE, M, N, BK, STAGES>(    \
            output, a_map, b_map, bias, params);                             \
    }

#define GBF_SM120_HALF_DEFINE_PAIR(M, N, BK, STAGES)                         \
    GBF_SM120_HALF_DEFINE_KERNEL(                                            \
        nn_sm120_tma_##M##x##N##_bk##BK##_s##STAGES##_bf16,          \
        __nv_bfloat16, __nv_bfloat16, M, N, BK, STAGES)                      \
    GBF_SM120_HALF_DEFINE_KERNEL(                                            \
        nn_sm120_tma_##M##x##N##_bk##BK##_s##STAGES##_f16,           \
        __half, __half, M, N, BK, STAGES)                                    \
    GBF_SM120_HALF_DEFINE_KERNEL(                                            \
        nn_sm120_tma_##M##x##N##_bk##BK##_s##STAGES##_f32out_bf16,   \
        __nv_bfloat16, float, M, N, BK, STAGES)                              \
    GBF_SM120_HALF_DEFINE_KERNEL(                                            \
        nn_sm120_tma_##M##x##N##_bk##BK##_s##STAGES##_f32out_f16,    \
        __half, float, M, N, BK, STAGES)

GBF_SM120_HALF_DEFINE_PAIR(64, 64, 64, 2)
GBF_SM120_HALF_DEFINE_PAIR(64, 128, 64, 2)
GBF_SM120_HALF_DEFINE_PAIR(128, 64, 32, 3)
GBF_SM120_HALF_DEFINE_PAIR(128, 128, 32, 2)
GBF_SM120_HALF_DEFINE_PAIR(128, 128, 32, 3)

#undef GBF_SM120_HALF_DEFINE_PAIR
#undef GBF_SM120_HALF_DEFINE_KERNEL
#undef GBF_SM120_HALF_CHECK_STORAGE
#undef GBF_SM120_HALF_SWIZZLE_128B
#undef GBF_SM120_HALF_SWIZZLE_64B

#endif
#line 1 "kernels/gemm_bi_inference/wmma_legacy.cu"
#define DEFINE_GEMM_BI_TC(NAME, T_IO, T_OUT, FROM_F_OUT, ZERO_IO)               \
extern "C" __global__ __launch_bounds__(THREADS, 2) void                        \
NAME(                                                                           \
    T_OUT* __restrict__ c,                                                      \
    const T_IO* __restrict__ a,                                                 \
    const T_IO* __restrict__ b,                                                 \
    const float* __restrict__ bias,                                             \
    float alpha, float beta,                                                    \
    int m, int n, int k,                                                        \
    int lda, int ldb, int ldc                                                   \
) {                                                                             \
    __shared__ T_IO smem_a[BLOCK_M * BLOCK_K];                                  \
    __shared__ T_IO smem_b[BLOCK_K * BLOCK_N];                                  \
                                                                                \
    /* GROUP_M swizzle for L2 locality. */                        \
    int num_pid_m = (m + BLOCK_M - 1) / BLOCK_M;                                \
    int num_pid_n = (n + BLOCK_N - 1) / BLOCK_N;                                \
    int num_pid_in_group = GROUP_M * num_pid_n;                                 \
    int group_id = blockIdx.x / num_pid_in_group;                               \
    int first_pid_m = group_id * GROUP_M;                                       \
    int group_size_m = min(num_pid_m - first_pid_m, GROUP_M);                   \
    int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m); \
    int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;                 \
                                                                                \
    int row0 = pid_m * BLOCK_M;                                                 \
    int col0 = pid_n * BLOCK_N;                                                 \
                                                                                \
    int warp_id = threadIdx.x / 32;                                             \
    int warp_m = warp_id / WARPS_N;                                             \
    int warp_n = warp_id % WARPS_N;                                             \
                                                                                \
    /* f32 accumulator fragments — one per (warp_m row, warp_n col0/col1). */   \
    wmma::fragment<wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, float> acc_frag[WARP_FRAGS_N]; \
    _Pragma("unroll")                                                           \
    for (int j = 0; j < WARP_FRAGS_N; j++) wmma::fill_fragment(acc_frag[j], 0.0f); \
                                                                                \
    for (int k_tile = 0; k_tile < k; k_tile += BLOCK_K) {                       \
        /* Cooperatively load A tile [BLOCK_M, BLOCK_K] = 2048 elems / 256 threads = 8/thread */ \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < 8; i++) {                                           \
            int idx = i * THREADS + threadIdx.x;                                \
            int smem_r = idx / BLOCK_K;                                         \
            int smem_c = idx % BLOCK_K;                                         \
            int g_r = row0 + smem_r;                                            \
            int g_c = k_tile + smem_c;                                          \
            smem_a[smem_r * BLOCK_K + smem_c] =                                 \
                (g_r < m && g_c < k) ? a[g_r * lda + g_c] : ZERO_IO();          \
        }                                                                       \
        /* Cooperatively load B tile [BLOCK_K, BLOCK_N] = 2048 elems / 256 threads = 8/thread */ \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < 8; i++) {                                           \
            int idx = i * THREADS + threadIdx.x;                                \
            int smem_r = idx / BLOCK_N;                                         \
            int smem_c = idx % BLOCK_N;                                         \
            int g_r = k_tile + smem_r;                                          \
            int g_c = col0 + smem_c;                                            \
            smem_b[smem_r * BLOCK_N + smem_c] =                                 \
                (g_r < k && g_c < n) ? b[g_r * ldb + g_c] : ZERO_IO();          \
        }                                                                       \
        __syncthreads();                                                        \
                                                                                \
        /* Inner K loop: 2 frag-K iterations of 16 each. Tensor Core MMA.   */  \
        /* Reduction order is fixed (kk = 0, 1) and per-output independent  */  \
        /* of M — preserves cross-batch bit-identity.                       */  \
        _Pragma("unroll")                                                       \
        for (int kk = 0; kk < K_TILES; kk++) {                                  \
            wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K,              \
                T_IO, wmma::row_major> a_frag;                                  \
            wmma::load_matrix_sync(                                             \
                a_frag,                                                         \
                &smem_a[(warp_m * FRAG_M) * BLOCK_K + kk * FRAG_K],             \
                BLOCK_K);                                                       \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < WARP_FRAGS_N; j++) {                            \
                wmma::fragment<wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K,          \
                    T_IO, wmma::row_major> b_frag;                              \
                wmma::load_matrix_sync(                                         \
                    b_frag,                                                     \
                    &smem_b[(kk * FRAG_K) * BLOCK_N                             \
                            + warp_n * (FRAG_N * WARP_FRAGS_N) + j * FRAG_N],   \
                    BLOCK_N);                                                   \
                wmma::mma_sync(acc_frag[j], a_frag, b_frag, acc_frag[j]);       \
            }                                                                   \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
                                                                                \
    /* Epilogue. Stage f32 accumulator to a per-warp smem tile, then each   */  \
    /* thread does scalar (alpha, beta, bias) + cast and writes one element */  \
    /* of C. Reusing smem_a (>= 2048 f32 elements when sizeof(T_IO)>=2) for */  \
    /* the staging buffer; only valid when BLOCK_M*BLOCK_K*sizeof(T_IO) >=  */  \
    /* WARPS_PER_CTA * FRAG_M * (FRAG_N * WARP_FRAGS_N) * sizeof(float) =   */  \
    /* 8*16*32*4 = 16 KB. BLOCK_M*BLOCK_K*sizeof(bf16) = 64*32*2 = 4 KB —   */  \
    /* not enough. Use a dedicated f32 staging buffer instead.              */  \
    __shared__ float smem_acc[BLOCK_M * BLOCK_N];                               \
                                                                                \
    /* Each warp stores its 2 N-fragments into the per-warp slot.           */  \
    int warp_row0 = warp_m * FRAG_M;                                            \
    int warp_col0 = warp_n * (FRAG_N * WARP_FRAGS_N);                           \
    _Pragma("unroll")                                                           \
    for (int j = 0; j < WARP_FRAGS_N; j++) {                                    \
        wmma::store_matrix_sync(                                                \
            &smem_acc[warp_row0 * BLOCK_N + warp_col0 + j * FRAG_N],            \
            acc_frag[j],                                                        \
            BLOCK_N,                                                            \
            wmma::mem_row_major);                                               \
    }                                                                           \
    __syncthreads();                                                            \
                                                                                \
    /* Scalar epilogue: 4096 elems / 256 threads = 16 per thread.           */  \
    _Pragma("unroll")                                                           \
    for (int i = 0; i < 16; i++) {                                              \
        int idx = i * THREADS + threadIdx.x;                                    \
        int local_r = idx / BLOCK_N;                                            \
        int local_c = idx % BLOCK_N;                                            \
        int r = row0 + local_r;                                                 \
        int col = col0 + local_c;                                               \
        if (r >= m || col >= n) continue;                                       \
        /* Pinned epilogue arithmetic - see the matvec note. */          \
        float val = __fmul_rn(alpha, smem_acc[local_r * BLOCK_N + local_c]);    \
        if (bias != nullptr) val = __fadd_rn(val, bias[col]);                   \
        if (beta != 0.0f) val = __fmaf_rn(beta, to_f(c[r * ldc + col]), val);   \
        c[r * ldc + col] = FROM_F_OUT(val);                                     \
    }                                                                           \
}

// Tensor-Core instantiations for half-precision paths (the regression source).
DEFINE_GEMM_BI_TC(bf16_bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16, zero_bf16)
DEFINE_GEMM_BI_TC(f16_f16,   __half,        __half,        from_f_f16,  zero_f16)
DEFINE_GEMM_BI_TC(bf16_f32,  __nv_bfloat16, float,         from_f_f32,  zero_bf16)
DEFINE_GEMM_BI_TC(f16_f32,   __half,        float,         from_f_f32,  zero_f16)

// f32 path stays on CUDA cores (Tensor Cores require fp16/bf16/tf32 inputs;
// converting f32→tf32 would lose 13 mantissa bits — not acceptable for the
// f32 training path that exists specifically because the user wants exact
// f32 math). cuBLAS f32 was never the regression source.
DEFINE_GEMM_BI_FFMA(f32_f32, float, float, from_f_f32, zero_f32)
#line 1 "kernels/gemm_bi_inference/matvec.cu"
// ═════════════════════════════════════════════════════════════════════════
// M=1 specialized batch-invariant matvec — the decode hot path.
//
// Performance design (based on cuBLAS `gemvNSplitK` post-mortem):
// - Memory-bound regime: FLOPs / byte ≈ 1 at M=1, deep under Ada's
//   145 FLOP/byte bf16 TC roofline → HBM bandwidth is the ceiling.
//   Target: ≥ 80% of 960 GB/s = 770 GB/s effective → ≈ 1000 tok/s on
//   mamba-130m bf16 @ 130 GEMMs/tok.
// - **Split-K within block**: each CTA owns `BLOCK_N` cols but
//   partitions the K dimension across `WARPS_PER_BLOCK=4` warps. Each
//   warp does K/4 accumulation; partials are reduced via smem with a
//   fixed tree order (warp 0 += 1 += 2 += 3). This saturates SM
//   occupancy for small-N workloads (mamba-130m N ∈ {768, 1536, 3072}).
// - Each thread owns one output column within its warp's K-range.
// - `a[K]` cooperatively loaded into smem ONCE — reused by all warps.
// - K-loop unrolled by 8 for ILP (issues multiple B loads in flight).
// - B reads: 32 threads of a warp read 32 adjacent cols at same K →
//   coalesced 64-byte transaction (half cache line).
//
// Batch invariance:
//   y[n] = (((Σ₀ + Σ₁) + Σ₂) + Σ₃)  where Σᵢ = Σ_{k∈range_i} a[k]·b[k,n]
//   K partition depends ONLY on K (not M) since M=1 has no batch dim.
//   Fixed-order tree reduction across warps → bit-identical output
//   regardless of launch context.
// SplitK GEMV-over-rows. Each CTA owns one (m_row, col_chunk) tile.
// - BLOCK_N_MV=32 cols per CTA
// - K split across WARPS_PER_BLOCK=8 warps
// - Per-warp partials → smem → fixed-order tree reduce in warp 0
//
// Grid: (ceil(N / BLOCK_N_MV), M, 1) — blockIdx.y = row index.
// Works for any M ≥ 1:
// - Decode (M=1): single row, 72 CTAs on Ada (~1000 tok/s).
// - RL/prefill (M>1): each row computed by its own CTAs. Per-row output
//   is bit-identical to the M=1 case because each row's K-reduction is
//   independent (SPLIT_K=1 across rows, fixed tree across warps within
//   one row). This is what guarantees cross-batch parity.
//
// Trade-off at large M: B is streamed once per row (not shared across M
// rows like cuBLAS's M-tiled GEMM). For small M (RL N_envs up to ~16)
// the bandwidth cost is acceptable given the batch-invariance guarantee.
#define BLOCK_N_MV 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (BLOCK_N_MV * WARPS_PER_BLOCK)
#define DEFINE_MATVEC_BI(NAME, T_IO, T_OUT, FROM_F_OUT)                         \
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 6) void              \
NAME(                                                                           \
    T_OUT* __restrict__ c,                                                      \
    const T_IO* __restrict__ a,                                                 \
    const T_IO* __restrict__ b,                                                 \
    const float* __restrict__ bias,                                             \
    float alpha, float beta,                                                    \
    int m, int n, int k,                                                        \
    int lda /* row-stride of A, == k */,                                        \
    int ldb,                                                                    \
    int ldc /* row-stride of C, == n */                                         \
) {                                                                             \
    const int lane = threadIdx.x & 31;                                          \
    const int warp_id = threadIdx.x >> 5;                                       \
    const int col = blockIdx.x * BLOCK_N_MV + lane;                             \
    const int row = blockIdx.y;  /* which row of A / C we compute */            \
    const bool in_range = (col < n) && (row < m);                               \
                                                                                \
    /* Shift A, C to our row. (row < m assumed by grid launch.)  */             \
    const T_IO* a_row = a + row * lda;                                          \
    T_OUT* c_row = c + row * ldc;                                               \
                                                                                \
    /* Static smem: 2 KB for partials (8 warps × 32 cols × 4 bytes) +       */  \
    /* dynamic smem for a[K]. Separating them avoids offset arithmetic bugs.*/  \
    __shared__ float smem_partials[WARPS_PER_BLOCK * BLOCK_N_MV];               \
    extern __shared__ unsigned char smem_a_raw[];                               \
    T_IO* smem_a = reinterpret_cast<T_IO*>(smem_a_raw);                         \
                                                                                \
    /* Cooperative global->smem load of a_row[0..K). Vectorized 128-bit */ \
    /* cp.async.cg (sm_80+) when EVERY A row is 16-byte aligned (row     */ \
    /* offset = row*k*sizeof(T_IO)); an arbitrary K (e.g. a vision patch */ \
    /* dim) can misalign rows - those take the all-scalar fill. smem     */ \
    /* contents are identical either way, so the per-col K reduction     */ \
    /* (and its bits) never depends on the path taken.                   */ \
    {                                                                       \
        const int VEC = 16 / (int)sizeof(T_IO);                             \
        if ((k * (int)sizeof(T_IO)) % 16 == 0 &&                            \
            (lda * (int)sizeof(T_IO)) % 16 == 0) {                          \
            /* Both gates matter: with lda != k a 16-byte-aligned K can  */ \
            /* still start a_row off-alignment (row * lda), and the      */ \
            /* uint4 reinterpret would fault. Misaligned rows take the   */ \
            /* all-scalar fill; smem contents are identical either way.  */ \
            const int k_vec = k / VEC;                                      \
            const uint4* a_vec =                                            \
                reinterpret_cast<const uint4*>(a_row);                      \
            uint4* smem_a_vec = reinterpret_cast<uint4*>(smem_a);           \
            _Pragma("unroll 1")                                             \
            for (int i = threadIdx.x; i < k_vec; i += THREADS_PER_BLOCK) {  \
                __pipeline_memcpy_async(                                    \
                    &smem_a_vec[i], &a_vec[i], sizeof(uint4));              \
            }                                                               \
            __pipeline_commit();                                            \
            __pipeline_wait_prior(0);                                       \
        } else {                                                            \
            for (int i = threadIdx.x; i < k; i += THREADS_PER_BLOCK) {      \
                smem_a[i] = a_row[i];                                       \
            }                                                               \
        }                                                                   \
    }                                                                       \
    __syncthreads();                                                        \
                                                                            \
    /* K-partition: each warp takes a contiguous range of K, rounded up  */ \
    /* to an EVEN span so every warp's k_start stays pair-aligned for    */ \
    /* the packed smem reads below. Even spans (every historic HF        */ \
    /* d_model) are unchanged - bit-stability preserved; odd spans       */ \
    /* previously FAULTED (misaligned LDS.U32 at an odd element offset). */ \
    int k_per_warp = (k + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;           \
    k_per_warp += (k_per_warp & 1);                                         \
    const int k_start = warp_id * k_per_warp;                               \
    const int k_stop = min(k, k_start + k_per_warp);                        \
                                                                                \
    /* Packed smem_a reads via the pair_to_f2 helper: for bf16/f16 one   */ \
    /* LDS.U32 delivers two elements where two LDS.U16 did one each.     */ \
    /* Address must be 4-byte aligned: kk steps by 8, smem_a starts at 0,  */  \
    /* so kk is always even and pair-aligned.                              */  \
    /* (An earlier __builtin_assume((k & 7) == 0) was removed: it was UB */ \
    /* for arbitrary K - vision patch dims - and the scalar tail below   */ \
    /* already handles k % 8 != 0 correctly at negligible cost.)         */ \
    float acc = 0.0f;                                                           \
    if (in_range && k_start < k_stop) {                                         \
        int kk = k_start;                                                       \
        int k_main = k_start + (((k_stop - k_start) >> 3) << 3);                \
        for (; kk < k_main; kk += 8) {                                          \
            float2 a01 = pair_to_f2(&smem_a[kk    ]);                           \
            float2 a23 = pair_to_f2(&smem_a[kk + 2]);                           \
            float2 a45 = pair_to_f2(&smem_a[kk + 4]);                           \
            float2 a67 = pair_to_f2(&smem_a[kk + 6]);                           \
            float a0 = a01.x, a1 = a01.y;                                       \
            float a2 = a23.x, a3 = a23.y;                                       \
            float a4 = a45.x, a5 = a45.y;                                       \
            float a6 = a67.x, a7 = a67.y;                                       \
            /* Plain cached loads: an evict-first hint here once looked   */  \
            /* free at M = 1 and cost the reuse every M >= 2 row depends  */  \
            /* on, since the grid re-reads B once per output row.         */  \
            float b0 = to_f(b[(kk    ) * ldb + col]);                  \
            float b1 = to_f(b[(kk + 1) * ldb + col]);                  \
            float b2 = to_f(b[(kk + 2) * ldb + col]);                  \
            float b3 = to_f(b[(kk + 3) * ldb + col]);                  \
            float b4 = to_f(b[(kk + 4) * ldb + col]);                  \
            float b5 = to_f(b[(kk + 5) * ldb + col]);                  \
            float b6 = to_f(b[(kk + 6) * ldb + col]);                  \
            float b7 = to_f(b[(kk + 7) * ldb + col]);                  \
            acc = fmaf(a0, b0, acc);                                            \
            acc = fmaf(a1, b1, acc);                                            \
            acc = fmaf(a2, b2, acc);                                            \
            acc = fmaf(a3, b3, acc);                                            \
            acc = fmaf(a4, b4, acc);                                            \
            acc = fmaf(a5, b5, acc);                                            \
            acc = fmaf(a6, b6, acc);                                            \
            acc = fmaf(a7, b7, acc);                                            \
        }                                                                       \
        for (; kk < k_stop; kk++) {                                             \
            acc = fmaf(to_f(smem_a[kk]), to_f(b[kk * ldb + col]), acc);         \
        }                                                                       \
    }                                                                           \
                                                                                \
    /* Store partial. All threads participate (out-of-range → 0). */            \
    smem_partials[warp_id * BLOCK_N_MV + lane] = acc;                           \
    __syncthreads();                                                            \
                                                                                \
    /* Warp 0 gathers partials for its col and does fixed-order tree reduce. */ \
    if (warp_id == 0 && in_range) {                                             \
        float p0 = smem_partials[0 * BLOCK_N_MV + lane];                        \
        float p1 = smem_partials[1 * BLOCK_N_MV + lane];                        \
        float p2 = smem_partials[2 * BLOCK_N_MV + lane];                        \
        float p3 = smem_partials[3 * BLOCK_N_MV + lane];                        \
        float p4 = smem_partials[4 * BLOCK_N_MV + lane];                        \
        float p5 = smem_partials[5 * BLOCK_N_MV + lane];                        \
        float p6 = smem_partials[6 * BLOCK_N_MV + lane];                        \
        float p7 = smem_partials[7 * BLOCK_N_MV + lane];                        \
        float s01 = p0 + p1;                                                    \
        float s23 = p2 + p3;                                                    \
        float s45 = p4 + p5;                                                    \
        float s67 = p6 + p7;                                                    \
        float s0123 = s01 + s23;                                                \
        float s4567 = s45 + s67;                                                \
        float sum = s0123 + s4567;                                              \
        /* Every step through an explicit intrinsic: bare mul/add    */\
        /* chains here are contraction bait, and a per-target fma     */\
        /* decision would make the epilogue bits arch-dependent.      */\
        float val = __fmul_rn(alpha, sum);                                      \
        if (bias != nullptr) val = __fadd_rn(val, bias[col]);                   \
        if (beta != 0.0f) val = __fmaf_rn(beta, to_f(c_row[col]), val);         \
        c_row[col] = FROM_F_OUT(val);                                           \
    }                                                                           \
}

DEFINE_MATVEC_BI(matvec_bi_bf16_bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16)
DEFINE_MATVEC_BI(matvec_bi_f16_f16,   __half,        __half,        from_f_f16)
DEFINE_MATVEC_BI(matvec_bi_bf16_f32,  __nv_bfloat16, float,         from_f_f32)
DEFINE_MATVEC_BI(matvec_bi_f16_f32,   __half,        float,         from_f_f32)
DEFINE_MATVEC_BI(matvec_bi_f32_f32,   float,         float,         from_f_f32)
#line 1 "kernels/gemm_bi_inference/mma16.cu"
// ============================================================================
// The inference tile ladder - the fixed family's fast NN forward.
// ============================================================================
// Three tensor-core tiles, all mma.sync m16n8k16 over ascending 64-wide
// K-slabs with f32 accumulators: GBF128 (128x128, dynamic smem) for large
// grids, GBF64 (64x64, static smem) for mid-size and underfilled grids,
// GBF16 (16x32) for small-M decode and narrow-N shapes. The three tiles
// produce byte-identical output per element - the reduction chain an
// element sees depends only on its own row, column and the slab order,
// never on the tile geometry - which is what makes a shape-keyed tile
// pick legal without changing bits. A safety layer hardens address
// formation: misaligned typed subviews take the scalar staging path
// (same shared-memory bytes) instead of silently staging wrong data.

// ============================================================================
// GBF128: the 128x128 tensor-core tile.
// ============================================================================
// mma.sync.aligned.m16n8k16 with f32 accumulators. Fully deterministic
// (fixed K order, fixed fragment/tile assignment, no atomics, no split-K)
// and batch-invariant across all M: each output element's entire
// K-reduction lives in one warp, independent of gridDim and M.
//
// Staging:
//   - As[m][k] (row-major) AND Bs[k][n] (row-major, global layout) are both
//     16B-chunk contiguous -> 2-stage cp.async pipeline with 4-operand
//     zero-fill for tails (bit-exact vs scalar zero stores). B fragments
//     come from ldmatrix.x2.TRANS of the k-major tile (delivers the
//     col-major k16n8 fragment without a staging transpose).
//   - Pads keep every ldmatrix row chunk in a distinct 4-bank group:
//     A row stride 72 halves (36 words ≡ 4 mod 8), B row stride 136 halves
//     (68 words ≡ 4 mod 8). Row bases are 16B-aligned (144 B / 272 B).
//   - Scalar staging fallback (uniform branch) when lda/ldb % 8 != 0.
//   - Smem (BK=64): NN 71 680 B / TN 69 632 B / NT 73 728 B — beyond the
//     48 KB static cap, so all three use dynamic smem with the
//     MAX_DYNAMIC_SHARED_SIZE_BYTES opt-in set at module load
//     (kernels.rs); launch passes the exact per-kernel byte count.
//     BK=64 halves the wait_group/__syncthreads boundary count per CTA
//     vs BK=32 (the measured per-boundary cost dominated the gap to
//     cuBLAS-TC).
//
// Geometry: CTA 256 threads = 8 warps as 2x4; BM=BN=128 BK=64; warp tile
// 64x32 = 4 m-frags(16) x 4 n-frags(8); bias pre-seeded into the f32
// accumulators (alpha must be 1.0 with bias); one RNE downcast at store.
//
// Fragment thread maps (PTX ISA m16n8k16, 16-bit A/B, .row.col):
//   lane L: g = L>>2, t = L&3
//   A: a0={(g,2t),(g,2t+1)} a1={(g+8,..)} a2={(g,2t+8),..} a3={(g+8,2t+8),..}
//   B: b0={(2t,g),(2t+1,g)} b1={(2t+8,g),(2t+9,g)}
//   C: c0=(g,2t) c1=(g,2t+1) c2=(g+8,2t) c3=(g+8,2t+1)
// A x4: lanes 0-7/8-15/16-23/24-31 -> (rows 0-7,k0)/(rows 8-15,k0)/
// (rows 0-7,k0+8)/(rows 8-15,k0+8). B x2.trans: lanes 0-7/8-15 -> stored
// rows (k0..k0+7)/(k0+8..k0+15) at column n0; .trans delivers M^T fragments
// = the col-major b-frags.

#define GBF128_BM 128
#define GBF128_BN 128
#define GBF128_BK 64
#define GBF128_PAD_A 8
#define GBF128_PAD_B 8
#define GBF128_LDA (GBF128_BK + GBF128_PAD_A)
#define GBF128_LDB (GBF128_BN + GBF128_PAD_B)

// Issue one A+B tile into smem stage `buf` via 16B cp.async with zero-fill
// (fast path; requires lda%8==0 && ldb%8==0, checked by caller-side branch).
// A: 128 rows x 8 chunks; B: 64 rows x 16 chunks; 2048 cp.async / 256 thr.
#define GBF128_STAGE_ASYNC(buf, bkIdx)                                        \
    do {                                                                      \
        unsigned _as = As_sbase + (unsigned)((buf) * GBF128_BM * GBF128_LDA * 2);     \
        unsigned _bs = Bs_sbase + (unsigned)((buf) * GBF128_BK * GBF128_LDB * 2);     \
        for (int _i = threadIdx.x; _i < GBF128_BM * (GBF128_BK / 8); _i += 256) {     \
            int _m = _i / (GBF128_BK / 8);                                        \
            int _c = _i % (GBF128_BK / 8);                                        \
            int _k = _c * 8;                                                  \
            int _gr = pid_m * GBF128_BM + _m;                                     \
            int _gc = (bkIdx) + _k;                                           \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as + (unsigned)((_m * GBF128_LDA + _k) * 2);         \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&A[(long long)_gr * lda + _gc]                 \
                : (const void*)A;                                             \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBF128_BK * (GBF128_BN / 8); _i += 256) {     \
            int _k = _i / (GBF128_BN / 8);                                        \
            int _c = _i % (GBF128_BN / 8);                                        \
            int _n = _c * 8;                                                  \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBF128_BN + _n;                                     \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs + (unsigned)((_k * GBF128_LDB + _n) * 2);         \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&B[(long long)_gk * ldb + _gn]                 \
                : (const void*)B;                                             \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

// Scalar staging fallback for misaligned lda/ldb (rare; uniform branch).
#define GBF128_STAGE_SCALAR(buf, bkIdx, TT, FF)                               \
    do {                                                                      \
        TT* _Asw = &As[buf][0][0];                                            \
        TT* _Bsw = &Bs[buf][0][0];                                            \
        for (int _i = threadIdx.x; _i < GBF128_BM * GBF128_BK; _i += 256) {           \
            int _m = _i / GBF128_BK;                                              \
            int _k = _i % GBF128_BK;                                              \
            int _gr = pid_m * GBF128_BM + _m;                                     \
            int _gc = (bkIdx) + _k;                                           \
            _Asw[_m * GBF128_LDA + _k] = (_gr < M && _gc < K)                     \
                                         ? A[(long long)_gr * lda + _gc]      \
                                         : FF(0.0f);                          \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBF128_BK * GBF128_BN; _i += 256) {           \
            int _k = _i / GBF128_BN;                                              \
            int _n = _i % GBF128_BN;                                              \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBF128_BN + _n;                                     \
            _Bsw[_k * GBF128_LDB + _n] = (_gk < K && _gn < N)                     \
                                         ? B[(long long)_gk * ldb + _gn]      \
                                         : FF(0.0f);                          \
        }                                                                     \
    } while (0)

#define DEFINE_GEMM_BI_NN_TC128(SUFFIX, T_ACT, T_OUT, FROM_ACT, FROM_OUT, MMA_T) \
extern "C" __global__ __launch_bounds__(256, 1)                                \
void nn_tc128_##SUFFIX(                                                  \
    T_OUT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    const float* __restrict__ bias,                                            \
    float alpha, float beta,                                                   \
    int M, int N, int K,                                                       \
    int lda, int ldb, int ldc                                                  \
) {                                                                            \
    assert(alpha == 1.0f || bias == nullptr);                                  \
    extern __shared__ __align__(16) unsigned char gbf128_dynsmem[];           \
    T_ACT (*As)[GBF128_BM][GBF128_LDA] =                                               \
        reinterpret_cast<T_ACT (*)[GBF128_BM][GBF128_LDA]>(gbf128_dynsmem);            \
    T_ACT (*Bs)[GBF128_BK][GBF128_LDB] = reinterpret_cast<T_ACT (*)[GBF128_BK][GBF128_LDB]>(   \
        gbf128_dynsmem + 2 * GBF128_BM * GBF128_LDA * (int)sizeof(T_ACT));             \
    int num_pid_n = (N + GBF128_BN - 1) / GBF128_BN;                                   \
    int pid_m = blockIdx.x / num_pid_n;                                        \
    int pid_n = blockIdx.x % num_pid_n;                                        \
    int warp = threadIdx.x / 32;                                               \
    int lane = threadIdx.x % 32;                                               \
    int warpRow = warp / 4;                                                    \
    int warpCol = warp % 4;                                                    \
    int warpM = warpRow * 64;                                                  \
    int warpN = warpCol * 32;                                                  \
    int g = lane >> 2;                                                         \
    int t = lane & 3;                                                          \
    int lm_r = lane & 7;                                                       \
    int lm_q = lane >> 3;                                                      \
    int lm_row_off = (lm_q & 1) ? 8 : 0;                                       \
    int lm_col_off = (lm_q & 2) ? 8 : 0;                                       \
    int lmb_row_off = (lm_q & 1) ? 8 : 0;                                      \
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(&As[0][0][0]);      \
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(&Bs[0][0][0]);      \
    bool fast_stage = ((lda & 7) == 0) && ((ldb & 7) == 0) &&                  \
                      gbf_aligned16(A) && gbf_aligned16(B);                    \
    float acc[4][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 4; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            float b0 = 0.0f, b1 = 0.0f;                                        \
            if (bias != nullptr) {                                             \
                int c0 = pid_n * GBF128_BN + warpN + fn * 8 + 2 * t;               \
                b0 = (c0 < N) ? bias[c0] : 0.0f;                               \
                b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                       \
            }                                                                  \
            acc[fm][fn][0] = b0;                                               \
            acc[fm][fn][1] = b1;                                               \
            acc[fm][fn][2] = b0;                                               \
            acc[fm][fn][3] = b1;                                               \
        }                                                                      \
    }                                                                          \
    int num_k_tiles = (K + GBF128_BK - 1) / GBF128_BK;                                 \
    if (fast_stage) {                                                          \
        GBF128_STAGE_ASYNC(0, 0);                                              \
    } else {                                                                   \
        GBF128_STAGE_SCALAR(0, 0, T_ACT, FROM_ACT);                            \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (kt + 1 < num_k_tiles) {                                            \
            if (fast_stage) {                                                  \
                GBF128_STAGE_ASYNC(read_buf ^ 1, (kt + 1) * GBF128_BK);            \
            } else {                                                           \
                GBF128_STAGE_SCALAR(read_buf ^ 1, (kt + 1) * GBF128_BK, T_ACT,     \
                                    FROM_ACT);                                 \
            }                                                                  \
        }                                                                      \
        unsigned As_rd = As_sbase + (unsigned)(read_buf * GBF128_BM * GBF128_LDA * 2); \
        unsigned Bs_rd = Bs_sbase + (unsigned)(read_buf * GBF128_BK * GBF128_LDB * 2); \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (GBF128_BK / 16); ks++) {                        \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[4][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 4; fm++) {                                   \
                int row = warpM + fm * 16 + lm_row_off + lm_r;                 \
                unsigned addr = As_rd +                                        \
                    (unsigned)((row * GBF128_LDA + k0 + lm_col_off) * 2);       \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "                \
                    "{%0,%1,%2,%3}, [%4];\n"                                   \
                    : "=r"(a_frag[fm][0]), "=r"(a_frag[fm][1]),                \
                      "=r"(a_frag[fm][2]), "=r"(a_frag[fm][3])                 \
                    : "r"(addr));                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int fn = 0; fn < 4; fn++) {                                   \
                int row = k0 + lmb_row_off + lm_r;                             \
                unsigned addr = Bs_rd +                                        \
                    (unsigned)((row * GBF128_LDB + warpN + fn * 8) * 2);        \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "          \
                    "{%0,%1}, [%2];\n"                                         \
                    : "=r"(b_frag[fn][0]), "=r"(b_frag[fn][1])                 \
                    : "r"(addr));                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 4; fm++) {                                   \
                _Pragma("unroll")                                              \
                for (int fn = 0; fn < 4; fn++) {                               \
                    asm volatile(                                              \
                        "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "."     \
                        MMA_T ".f32 "                                          \
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "              \
                        "{%0,%1,%2,%3};\n"                                     \
                        : "+f"(acc[fm][fn][0]), "+f"(acc[fm][fn][1]),          \
                          "+f"(acc[fm][fn][2]), "+f"(acc[fm][fn][3])           \
                        : "r"(a_frag[fm][0]), "r"(a_frag[fm][1]),              \
                          "r"(a_frag[fm][2]), "r"(a_frag[fm][3]),              \
                          "r"(b_frag[fn][0]), "r"(b_frag[fn][1]));             \
                }                                                              \
            }                                                                  \
        }                                                                      \
        read_buf ^= 1;                                                         \
    }                                                                          \
    _Pragma("unroll")                                                         \
    for (int fm = 0; fm < 4; fm++) {                                          \
        _Pragma("unroll")                                                     \
        for (int fn = 0; fn < 4; fn++) {                                      \
            int r0 = pid_m * GBF128_BM + warpM + fm * 16 + g;                     \
            int c0 = pid_n * GBF128_BN + warpN + fn * 8 + 2 * t;                  \
            /* c0 is always even (warpN, fn*8, 2t all even), so when          \
               ldc is even too the (c0, c0+1) pair is 4-byte aligned:         \
               pack both RNE results into one 32-bit store (half the          \
               store issue). Values identical to the scalar path. */          \
            _Pragma("unroll")                                                 \
            for (int half = 0; half < 2; half++) {                            \
                int gr = r0 + (half ? 8 : 0);                                 \
                if (gr >= M) continue;                                        \
                /* Explicit unfused multiply: a bare alpha*acc leaves   */ \
                /* ptxas free to contract it per target, which would    */ \
                /* make the epilogue bits architecture-dependent.       */ \
                float v0 = __fmul_rn(alpha, acc[fm][fn][2 * half]);                     \
                float v1 = __fmul_rn(alpha, acc[fm][fn][2 * half + 1]);                 \
                T_OUT* _dst = (c0 < N)                                        \
                    ? &C[(long long)gr * ldc + c0]                            \
                    : (T_OUT*)0;                                              \
                if (sizeof(T_OUT) == 2 && beta == 0.0f && (ldc & 1) == 0 &&   \
                    c0 + 1 < N &&                                              \
                    gbf_aligned4(_dst)) {                                     \
                    gbf_store_pair_rne(_dst, v0, v1);                         \
                } else {                                                      \
                    for (int e = 0; e < 2; e++) {                             \
                        int gc = c0 + e;                                      \
                        if (gc >= N) continue;                                \
                        float val = e ? v1 : v0;                              \
                        if (beta != 0.0f)                                     \
                            val = __fmaf_rn(beta, to_f(C[(long long)gr * ldc + gc]), val);  \
                        C[(long long)gr * ldc + gc] = FROM_OUT(val);          \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

DEFINE_GEMM_BI_NN_TC128(bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16, from_f_bf16, "bf16")
DEFINE_GEMM_BI_NN_TC128(f16,  __half,        __half,        from_f_f16,  from_f_f16,  "f16")
DEFINE_GEMM_BI_NN_TC128(f32out_bf16, __nv_bfloat16, float, from_f_bf16, from_f_f32, "bf16")
DEFINE_GEMM_BI_NN_TC128(f32out_f16,  __half,        float, from_f_f16,  from_f_f32, "f16")

// ============================================================================
// GBF64: the 64x64 tensor-core tile.
// ============================================================================
// Same numeric contract as GBF128, and bit-identical to it per output
// element: both walk K in ascending 64-wide slabs split into ascending
// m16n8k16 mma steps with the same 16B-chunk zero-fill for tails, so an
// element's f32 accumulator sees the exact same mma chain whichever tile
// the dispatcher picked. Do not change the slab width, the mma step
// order, or the tail zero-fill here without changing GBF128 and GBF16 in
// lockstep - the bit-identity is what keeps the tile pick free.
//
// Geometry: CTA 128 threads = 4 warps as 2x2; BM=BN=64, BK=64; warp tile
// 32x32 = 2 m-frags(16) x 4 n-frags(8). Same 2-stage cp.async staging (16B
// chunks, 4-operand zero-fill tails), same ldmatrix x4 / x2(.trans)
// fragment loads, same conflict-free pads (row strides ≡ 4 mod 8 words,
// 16B-aligned row bases): A-layout rows 40 halves (BK wide), B-layout rows
// 72 halves (BN wide). Static smem at BK=64: 36 864 B per kernel (< 48 KB,
// so the Tile64 family stays static while Tile128 goes dynamic).
// Why it wins on small shapes: a 128-tile CTA grid underfills the GPU
// (e.g. d128 in_proj dW = 4 CTAs on a 142-SM Ada); quartering the tile
// quadruples the CTA count at the same total FLOPs.
//
// Every constant below is section-local (GBF64_*).
// NEVER reference ambient tile names from other sections
// define from earlier sections inside this section.

#define GBF64_BM 64
#define GBF64_BN 64
#define GBF64_BK 64
#define GBF64_THREADS 128
#define GBF64_LDA (GBF64_BK + 8) /* 72 halves = 144 B rows, 36 words ≡ 4 mod 8 */
#define GBF64_LDB (GBF64_BN + 8) /* 72 halves = 144 B rows, 36 words ≡ 4 mod 8 */

// NN staging: A 64 rows x 4 chunks + B 32 rows x 8 chunks = 512 cp.async
// over 128 threads (4 per thread), 16B each with zero-fill tails.
#define GBF64_STAGE_ASYNC(buf, bkIdx)                                      \
    do {                                                                      \
        unsigned _as =                                                        \
            As_sbase + (unsigned)((buf) * GBF64_BM * GBF64_LDA * 2);    \
        unsigned _bs =                                                        \
            Bs_sbase + (unsigned)((buf) * GBF64_BK * GBF64_LDB * 2);    \
        for (int _i = threadIdx.x; _i < GBF64_BM * (GBF64_BK / 8);      \
             _i += GBF64_THREADS) {                                        \
            int _m = _i / (GBF64_BK / 8);                                  \
            int _k = (_i % (GBF64_BK / 8)) * 8;                            \
            int _gr = pid_m * GBF64_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as + (unsigned)((_m * GBF64_LDA + _k) * 2);   \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&A[(long long)_gr * lda + _gc]                 \
                : (const void*)A;                                             \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBF64_BK * (GBF64_BN / 8);      \
             _i += GBF64_THREADS) {                                        \
            int _k = _i / (GBF64_BN / 8);                                  \
            int _n = (_i % (GBF64_BN / 8)) * 8;                            \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBF64_BN + _n;                               \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs + (unsigned)((_k * GBF64_LDB + _n) * 2);   \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&B[(long long)_gk * ldb + _gn]                 \
                : (const void*)B;                                             \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

// Scalar staging fallback for misaligned lda/ldb (rare; uniform branch).
#define GBF64_STAGE_SCALAR(buf, bkIdx, TT, FF)                             \
    do {                                                                      \
        TT* _Asw = &As[buf][0][0];                                            \
        TT* _Bsw = &Bs[buf][0][0];                                            \
        for (int _i = threadIdx.x; _i < GBF64_BM * GBF64_BK;            \
             _i += GBF64_THREADS) {                                        \
            int _m = _i / GBF64_BK;                                        \
            int _k = _i % GBF64_BK;                                        \
            int _gr = pid_m * GBF64_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            _Asw[_m * GBF64_LDA + _k] = (_gr < M && _gc < K)               \
                                               ? A[(long long)_gr * lda + _gc]\
                                               : FF(0.0f);                    \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBF64_BK * GBF64_BN;            \
             _i += GBF64_THREADS) {                                        \
            int _k = _i / GBF64_BN;                                        \
            int _n = _i % GBF64_BN;                                        \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBF64_BN + _n;                               \
            _Bsw[_k * GBF64_LDB + _n] = (_gk < K && _gn < N)               \
                                               ? B[(long long)_gk * ldb + _gn]\
                                               : FF(0.0f);                    \
        }                                                                     \
    } while (0)

#define DEFINE_GEMM_BI_NN_TC64(SUFFIX, T_ACT, T_OUT, FROM_ACT, FROM_OUT, MMA_T) \
extern "C" __global__ __launch_bounds__(GBF64_THREADS, 1)                   \
void nn_tc64_##SUFFIX(                                                \
    T_OUT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    const float* __restrict__ bias,                                            \
    float alpha, float beta,                                                   \
    int M, int N, int K,                                                       \
    int lda, int ldb, int ldc                                                  \
) {                                                                            \
    assert(alpha == 1.0f || bias == nullptr);                                  \
    __shared__ __align__(16) T_ACT As[2][GBF64_BM][GBF64_LDA];           \
    __shared__ __align__(16) T_ACT Bs[2][GBF64_BK][GBF64_LDB];           \
    int num_pid_n = (N + GBF64_BN - 1) / GBF64_BN;                       \
    int pid_m = blockIdx.x / num_pid_n;                                        \
    int pid_n = blockIdx.x % num_pid_n;                                        \
    int warp = threadIdx.x / 32;                                               \
    int lane = threadIdx.x % 32;                                               \
    int warpM = (warp / 2) * 32;                                               \
    int warpN = (warp % 2) * 32;                                               \
    int g = lane >> 2;                                                         \
    int t = lane & 3;                                                          \
    int lm_r = lane & 7;                                                       \
    int lm_q = lane >> 3;                                                      \
    int lm_row_off = (lm_q & 1) ? 8 : 0;                                       \
    int lm_col_off = (lm_q & 2) ? 8 : 0;                                       \
    int lmb_row_off = (lm_q & 1) ? 8 : 0;                                      \
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(&As[0][0][0]);      \
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(&Bs[0][0][0]);      \
    bool fast_stage = ((lda & 7) == 0) && ((ldb & 7) == 0) &&                  \
                      gbf_aligned16(A) && gbf_aligned16(B);                    \
    float acc[2][4][4];                                                        \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            float b0 = 0.0f, b1 = 0.0f;                                        \
            if (bias != nullptr) {                                             \
                int c0 = pid_n * GBF64_BN + warpN + fn * 8 + 2 * t;         \
                b0 = (c0 < N) ? bias[c0] : 0.0f;                               \
                b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                       \
            }                                                                  \
            acc[fm][fn][0] = b0;                                               \
            acc[fm][fn][1] = b1;                                               \
            acc[fm][fn][2] = b0;                                               \
            acc[fm][fn][3] = b1;                                               \
        }                                                                      \
    }                                                                          \
    int num_k_tiles = (K + GBF64_BK - 1) / GBF64_BK;                     \
    if (fast_stage) {                                                          \
        GBF64_STAGE_ASYNC(0, 0);                                            \
    } else {                                                                   \
        GBF64_STAGE_SCALAR(0, 0, T_ACT, FROM_ACT);                          \
    }                                                                          \
    int read_buf = 0;                                                          \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                 \
        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \
        __syncthreads();                                                       \
        if (kt + 1 < num_k_tiles) {                                            \
            if (fast_stage) {                                                  \
                GBF64_STAGE_ASYNC(read_buf ^ 1, (kt + 1) * GBF64_BK);    \
            } else {                                                           \
                GBF64_STAGE_SCALAR(read_buf ^ 1, (kt + 1) * GBF64_BK,    \
                                      T_ACT, FROM_ACT);                        \
            }                                                                  \
        }                                                                      \
        unsigned As_rd =                                                       \
            As_sbase + (unsigned)(read_buf * GBF64_BM * GBF64_LDA * 2);  \
        unsigned Bs_rd =                                                       \
            Bs_sbase + (unsigned)(read_buf * GBF64_BK * GBF64_LDB * 2);  \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (GBF64_BK / 16); ks++) {                            \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[2][4];                                             \
            unsigned b_frag[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 2; fm++) {                                   \
                int row = warpM + fm * 16 + lm_row_off + lm_r;                 \
                unsigned addr = As_rd +                                        \
                    (unsigned)((row * GBF64_LDA + k0 + lm_col_off) * 2);    \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "                \
                    "{%0,%1,%2,%3}, [%4];\n"                                   \
                    : "=r"(a_frag[fm][0]), "=r"(a_frag[fm][1]),                \
                      "=r"(a_frag[fm][2]), "=r"(a_frag[fm][3])                 \
                    : "r"(addr));                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int fn = 0; fn < 4; fn++) {                                   \
                int row = k0 + lmb_row_off + lm_r;                             \
                unsigned addr = Bs_rd +                                        \
                    (unsigned)((row * GBF64_LDB + warpN + fn * 8) * 2);     \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "          \
                    "{%0,%1}, [%2];\n"                                         \
                    : "=r"(b_frag[fn][0]), "=r"(b_frag[fn][1])                 \
                    : "r"(addr));                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int fm = 0; fm < 2; fm++) {                                   \
                _Pragma("unroll")                                              \
                for (int fn = 0; fn < 4; fn++) {                               \
                    asm volatile(                                              \
                        "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "."     \
                        MMA_T ".f32 "                                          \
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "              \
                        "{%0,%1,%2,%3};\n"                                     \
                        : "+f"(acc[fm][fn][0]), "+f"(acc[fm][fn][1]),          \
                          "+f"(acc[fm][fn][2]), "+f"(acc[fm][fn][3])           \
                        : "r"(a_frag[fm][0]), "r"(a_frag[fm][1]),              \
                          "r"(a_frag[fm][2]), "r"(a_frag[fm][3]),              \
                          "r"(b_frag[fn][0]), "r"(b_frag[fn][1]));             \
                }                                                              \
            }                                                                  \
        }                                                                      \
        read_buf ^= 1;                                                         \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int fm = 0; fm < 2; fm++) {                                           \
        _Pragma("unroll")                                                      \
        for (int fn = 0; fn < 4; fn++) {                                       \
            int r0 = pid_m * GBF64_BM + warpM + fm * 16 + g;                \
            int c0 = pid_n * GBF64_BN + warpN + fn * 8 + 2 * t;             \
            _Pragma("unroll")                                                  \
            for (int half = 0; half < 2; half++) {                             \
                int gr = r0 + (half ? 8 : 0);                                  \
                if (gr >= M) continue;                                         \
                float v0 = __fmul_rn(alpha, acc[fm][fn][2 * half]);             \
                float v1 = __fmul_rn(alpha, acc[fm][fn][2 * half + 1]);         \
                T_OUT* dst = (c0 < N)                                          \
                    ? &C[(long long)gr * ldc + c0]                             \
                    : (T_OUT*)0;                                               \
                if (beta == 0.0f && (ldc & 1) == 0 && c0 + 1 < N &&            \
                    gbf_aligned4(dst)) {                                       \
                    gbf_store_pair_rne(dst, v0, v1);                           \
                } else {                                                       \
                    for (int e = 0; e < 2; e++) {                              \
                        int gc = c0 + e;                                       \
                        if (gc >= N) continue;                                 \
                        float val = e ? v1 : v0;                               \
                        if (beta != 0.0f)                                      \
                            val = __fmaf_rn(beta, to_f(C[(long long)gr * ldc + gc]), val); \
                        C[(long long)gr * ldc + gc] = FROM_OUT(val);           \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

DEFINE_GEMM_BI_NN_TC64(bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16, from_f_bf16, "bf16")
DEFINE_GEMM_BI_NN_TC64(f16,  __half,        __half,        from_f_f16,  from_f_f16,  "f16")
DEFINE_GEMM_BI_NN_TC64(f32out_bf16, __nv_bfloat16, float, from_f_bf16, from_f_f32, "bf16")
DEFINE_GEMM_BI_NN_TC64(f32out_f16,  __half,        float, from_f_f16,  from_f_f32, "f16")
// ── GBF16: the 16x32 thin tile (decode and narrow-N shapes) ──
//
// Same arithmetic contract as GBF64/GBF128 (ascending m16n8k16 K-slabs,
// f32 accumulators, bias pre-seeded, single RNE downcast), so it is
// byte-identical to them per element. What differs is scheduling: a
// 16-row tile stops wasting 3/4 of the MMA work at M<=16, BN=32 raises
// the CTA count at small M (decode needs CTAs for memory-level
// parallelism, not tile area), and a 4-deep cp.async pipeline keeps
// enough bytes in flight to approach the DRAM floor where a 2-stage
// pipeline stalls on latency.
//
// Pipeline bookkeeping: every iteration commits EXACTLY one group -
// a real stage or an empty commit once the K range is exhausted - so
// the in-flight count is uniform and `wait_group STAGES-2` always
// waits for precisely the stage the mainloop is about to read. Without
// the empty commits a short K (fewer tiles than stages) would make
// wait_group return before the data landed.
//
// Every constant below is section-local (GBF16_*).
#define GBF16_BM 16
#define GBF16_BN 32
#define GBF16_BK 64
#define GBF16_THREADS 128
#define GBF16_STAGES 4
#define GBF16_ACH (GBF16_BK / 8)
#define GBF16_BCH (GBF16_BN / 8)
#define GBF16_A_STAGE_BYTES (GBF16_BM * GBF16_ACH * 16)
#define GBF16_B_STAGE_BYTES (GBF16_BK * GBF16_BCH * 16)
#define GBF16_SLOT(row, chunk, chunks) (((row) * (chunks) + (chunk)) ^ ((row) & 7))

#define GBF16_STAGE_ASYNC(buf, bkIdx)                                      \
    do {                                                                      \
        unsigned _as = As_sbase + (unsigned)((buf) * GBF16_A_STAGE_BYTES);    \
        unsigned _bs = Bs_sbase + (unsigned)((buf) * GBF16_B_STAGE_BYTES);    \
        for (int _i = threadIdx.x; _i < GBF16_BM * GBF16_ACH;                 \
             _i += GBF16_THREADS) {                                        \
            int _m = _i / GBF16_ACH;                                         \
            int _c = _i % GBF16_ACH;                                         \
            int _k = _c * 8;                                                 \
            int _gr = pid_m * GBF16_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as +                                            \
                (unsigned)(GBF16_SLOT(_m, _c, GBF16_ACH) << 4);              \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&A[(long long)_gr * lda + _gc]                 \
                : (const void*)A;                                             \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBF16_BK * GBF16_BCH;                 \
             _i += GBF16_THREADS) {                                        \
            int _k = _i / GBF16_BCH;                                         \
            int _c = _i % GBF16_BCH;                                         \
            int _n = _c * 8;                                                 \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBF16_BN + _n;                               \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs +                                            \
                (unsigned)(GBF16_SLOT(_k, _c, GBF16_BCH) << 4);              \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&B[(long long)_gk * ldb + _gn]                 \
                : (const void*)B;                                             \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define GBF16_STAGE_SCALAR(buf, bkIdx, TT, FF)                             \
    do {                                                                      \
        unsigned char* _Asw = As_bytes + (buf) * GBF16_A_STAGE_BYTES;         \
        unsigned char* _Bsw = Bs_bytes + (buf) * GBF16_B_STAGE_BYTES;         \
        for (int _i = threadIdx.x; _i < GBF16_BM * GBF16_BK;            \
             _i += GBF16_THREADS) {                                        \
            int _m = _i / GBF16_BK;                                        \
            int _k = _i % GBF16_BK;                                        \
            int _gr = pid_m * GBF16_BM + _m;                               \
            int _gc = (bkIdx) + _k;                                           \
            *reinterpret_cast<TT*>(                                           \
                _Asw + (GBF16_SLOT(_m, _k >> 3, GBF16_ACH) << 4) +            \
                (_k & 7) * 2) = (_gr < M && _gc < K)                          \
                                      ? A[(long long)_gr * lda + _gc]         \
                                      : FF(0.0f);                             \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBF16_BK * GBF16_BN;            \
             _i += GBF16_THREADS) {                                        \
            int _k = _i / GBF16_BN;                                        \
            int _n = _i % GBF16_BN;                                        \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBF16_BN + _n;                               \
            *reinterpret_cast<TT*>(                                           \
                _Bsw + (GBF16_SLOT(_k, _n >> 3, GBF16_BCH) << 4) +            \
                (_n & 7) * 2) = (_gk < K && _gn < N)                          \
                                      ? B[(long long)_gk * ldb + _gn]         \
                                      : FF(0.0f);                             \
        }                                                                     \
    } while (0)

#define DEFINE_GEMM_BI_NN_TC16(SUFFIX, T_ACT, T_OUT, FROM_ACT, FROM_OUT, MMA_T) \
extern "C" __global__ __launch_bounds__(GBF16_THREADS, 3)                   \
void nn_tc16_##SUFFIX(                                                \
    T_OUT* __restrict__ C,                                                     \
    const T_ACT* __restrict__ A,                                               \
    const T_ACT* __restrict__ B,                                               \
    const float* __restrict__ bias,                                            \
    float alpha, float beta,                                                   \
    int M, int N, int K,                                                       \
    int lda, int ldb, int ldc                                                  \
) {                                                                            \
    assert(alpha == 1.0f || bias == nullptr);                                  \
    __shared__ __align__(16) unsigned char gbf16_smem[                        \
        GBF16_STAGES * (GBF16_A_STAGE_BYTES + GBF16_B_STAGE_BYTES)];          \
    unsigned char* As_bytes = gbf16_smem;                                     \
    unsigned char* Bs_bytes = gbf16_smem + GBF16_STAGES * GBF16_A_STAGE_BYTES;\
    int num_pid_n = (N + GBF16_BN - 1) / GBF16_BN;                       \
    int pid_m = blockIdx.x / num_pid_n;                                        \
    int pid_n = blockIdx.x % num_pid_n;                                        \
    int warp = threadIdx.x / 32;                                               \
    int lane = threadIdx.x % 32;                                               \
    int warpN = warp * 8;                                                      \
    int g = lane >> 2;                                                         \
    int t = lane & 3;                                                          \
    int lm_r = lane & 7;                                                       \
    int lm_q = lane >> 3;                                                      \
    int lm_row_off = (lm_q & 1) ? 8 : 0;                                       \
    int lm_col_off = (lm_q & 2) ? 8 : 0;                                       \
    int lmb_row_off = (lm_q & 1) ? 8 : 0;                                      \
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(As_bytes);          \
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(Bs_bytes);          \
    bool fast_stage = ((lda & 7) == 0) && ((ldb & 7) == 0) &&                  \
                      gbf_aligned16(A) && gbf_aligned16(B);                    \
    float acc[4];                                                              \
    {                                                                          \
        float b0 = 0.0f, b1 = 0.0f;                                            \
        if (bias != nullptr) {                                                 \
            int c0 = pid_n * GBF16_BN + warpN + 2 * t;                      \
            b0 = (c0 < N) ? bias[c0] : 0.0f;                                   \
            b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                           \
        }                                                                      \
        acc[0] = b0;                                                           \
        acc[1] = b1;                                                           \
        acc[2] = b0;                                                           \
        acc[3] = b1;                                                           \
    }                                                                          \
    int num_k_tiles = (K + GBF16_BK - 1) / GBF16_BK;                     \
    /* Prologue: STAGES-1 commit groups, real or empty - uniform count. */    \
    for (int p = 0; p < GBF16_STAGES - 1; p++) {                            \
        if (p < num_k_tiles) {                                                 \
            if (fast_stage) {                                                  \
                GBF16_STAGE_ASYNC(p, p * GBF16_BK);                      \
            } else {                                                           \
                GBF16_STAGE_SCALAR(p, p * GBF16_BK, T_ACT, FROM_ACT);    \
                asm volatile("cp.async.commit_group;\n");                     \
            }                                                                  \
        } else {                                                               \
            asm volatile("cp.async.commit_group;\n");                         \
        }                                                                      \
    }                                                                          \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                 \
        asm volatile("cp.async.wait_group %0;\n"                              \
                     :: "n"(GBF16_STAGES - 2));                             \
        __syncthreads();                                                       \
        int next = kt + GBF16_STAGES - 1;                                   \
        if (next < num_k_tiles) {                                              \
            int wbuf = next % GBF16_STAGES;                                 \
            if (fast_stage) {                                                  \
                GBF16_STAGE_ASYNC(wbuf, next * GBF16_BK);                \
            } else {                                                           \
                GBF16_STAGE_SCALAR(wbuf, next * GBF16_BK, T_ACT,         \
                                      FROM_ACT);                               \
                asm volatile("cp.async.commit_group;\n");                     \
            }                                                                  \
        } else {                                                               \
            asm volatile("cp.async.commit_group;\n");                         \
        }                                                                      \
        int rbuf = kt % GBF16_STAGES;                                       \
        unsigned As_rd = As_sbase + (unsigned)(rbuf * GBF16_A_STAGE_BYTES);    \
        unsigned Bs_rd = Bs_sbase + (unsigned)(rbuf * GBF16_B_STAGE_BYTES);    \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < (GBF16_BK / 16); ks++) {                      \
            int k0 = ks * 16;                                                  \
            unsigned a_frag[4];                                                \
            unsigned b_frag[2];                                                \
            {                                                                  \
                int row = lm_row_off + lm_r;                                   \
                int chunk = (k0 + lm_col_off) >> 3;                           \
                unsigned addr = As_rd +                                       \
                    (unsigned)(GBF16_SLOT(row, chunk, GBF16_ACH) << 4);        \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "                \
                    "{%0,%1,%2,%3}, [%4];\n"                                  \
                    : "=r"(a_frag[0]), "=r"(a_frag[1]),                        \
                      "=r"(a_frag[2]), "=r"(a_frag[3])                         \
                    : "r"(addr));                                              \
            }                                                                  \
            {                                                                  \
                int row = k0 + lmb_row_off + lm_r;                             \
                int chunk = warpN >> 3;                                       \
                unsigned addr = Bs_rd +                                       \
                    (unsigned)(GBF16_SLOT(row, chunk, GBF16_BCH) << 4);        \
                asm volatile(                                                  \
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "          \
                    "{%0,%1}, [%2];\n"                                        \
                    : "=r"(b_frag[0]), "=r"(b_frag[1])                         \
                    : "r"(addr));                                              \
            }                                                                  \
            asm volatile(                                                      \
                "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "."             \
                MMA_T ".f32 "                                                  \
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "                      \
                "{%0,%1,%2,%3};\n"                                            \
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])       \
                : "r"(a_frag[0]), "r"(a_frag[1]),                              \
                  "r"(a_frag[2]), "r"(a_frag[3]),                              \
                  "r"(b_frag[0]), "r"(b_frag[1]));                             \
        }                                                                      \
    }                                                                          \
    {                                                                          \
        int r0 = pid_m * GBF16_BM + g;                                      \
        int c0 = pid_n * GBF16_BN + warpN + 2 * t;                          \
        _Pragma("unroll")                                                      \
        for (int half = 0; half < 2; half++) {                                 \
            int gr = r0 + (half ? 8 : 0);                                      \
            if (gr >= M) continue;                                             \
            float v0 = __fmul_rn(alpha, acc[2 * half]);                        \
            float v1 = __fmul_rn(alpha, acc[2 * half + 1]);                    \
            T_OUT* dst = (c0 < N)                                              \
                ? &C[(long long)gr * ldc + c0]                                 \
                : (T_OUT*)0;                                                   \
            if (beta == 0.0f && (ldc & 1) == 0 && c0 + 1 < N &&                \
                gbf_aligned4(dst)) {                                           \
                gbf_store_pair_rne(dst, v0, v1);                               \
            } else {                                                           \
                for (int e = 0; e < 2; e++) {                                  \
                    int gc = c0 + e;                                           \
                    if (gc >= N) continue;                                     \
                    float val = e ? v1 : v0;                                   \
                    if (beta != 0.0f)                                          \
                        val = __fmaf_rn(beta, to_f(C[(long long)gr * ldc + gc]), val); \
                    C[(long long)gr * ldc + gc] = FROM_OUT(val);               \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

DEFINE_GEMM_BI_NN_TC16(bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16, from_f_bf16, "bf16")
DEFINE_GEMM_BI_NN_TC16(f16,  __half,        __half,        from_f_f16,  from_f_f16,  "f16")
DEFINE_GEMM_BI_NN_TC16(f32out_bf16, __nv_bfloat16, float, from_f_bf16, from_f_f32, "bf16")
DEFINE_GEMM_BI_NN_TC16(f32out_f16,  __half,        float, from_f_f16,  from_f_f32, "f16")

// GBF hygiene: everything section-local above is undefined so nothing can
// leak into gemm_bi_triad.cu, which concatenates AFTER this file.
#undef GBF128_BM
#undef GBF128_BN
#undef GBF128_BK
#undef GBF128_PAD_A
#undef GBF128_PAD_B
#undef GBF128_LDA
#undef GBF128_LDB
#undef GBF128_STAGE_ASYNC
#undef GBF128_STAGE_SCALAR
#undef DEFINE_GEMM_BI_NN_TC128
#undef GBF64_BM
#undef GBF64_BN
#undef GBF64_BK
#undef GBF64_THREADS
#undef GBF64_LDA
#undef GBF64_LDB
#undef GBF64_STAGE_ASYNC
#undef GBF64_STAGE_SCALAR
#undef DEFINE_GEMM_BI_NN_TC64
#undef GBF16_BM
#undef GBF16_BN
#undef GBF16_BK
#undef GBF16_THREADS
#undef GBF16_STAGES
#undef GBF16_ACH
#undef GBF16_BCH
#undef GBF16_A_STAGE_BYTES
#undef GBF16_B_STAGE_BYTES
#undef GBF16_SLOT
#undef GBF16_STAGE_ASYNC
#undef GBF16_STAGE_SCALAR
#undef DEFINE_GEMM_BI_NN_TC16

// Ambient-define hygiene: this file is concatenated BEFORE gemm_bi_triad.cu
// in the single NVRTC blob. A tile macro leaking downstream would compile
// someone else's kernels with these constants - plausible code that never
// launches correctly. Undefine everything tile-local.
#undef BLOCK_M
#undef BLOCK_N
#undef BLOCK_K
#undef GROUP_M
#undef THREADS
#undef WARPS_PER_CTA
#undef WARPS_M
#undef WARPS_N
#undef FRAG_M
#undef FRAG_N
#undef FRAG_K
#undef WARP_FRAGS_N
#undef BLOCK_N_MV
#undef WARPS_PER_BLOCK
#undef THREADS_PER_BLOCK
#line 1 "kernels/gemm_bi_inference/tcw64.cu"
// ============================================================================
// GBFW64: the fragment-reuse 128x128 tile (warp tile 64x64).
// ============================================================================
// Same numeric contract as GBF128 and bit-identical to it per output
// element: K walks in the same ascending 64-wide slabs split into the
// same ascending m16n8k16 steps with the same 16-byte-chunk zero-fill
// for tails, so an element's f32 accumulator sees the exact same mma
// chain whichever tile the dispatcher picked. What changes is only who
// computes it: four warps own 64x64 output quadrants instead of eight
// warps owning 64x32 halves, which doubles the work each loaded
// fragment feeds (mma:ldmatrix 2.0 -> 4.0) and cuts shared-memory read
// traffic by a third - the lever the fat training shapes are bound by.
//
// Shared memory drops the +8-half row pads for the standard XOR chunk
// swizzle: the 16-byte chunk at logical (row, c) lives at physical
// chunk c ^ (row & 7), which walks all 32 banks for every 8-lane
// ldmatrix group and every cp.async write group, and shrinks the stage
// pair to 65,536 bytes. The swizzle relocates whole 16-byte chunks and
// never splits one, so the bytes every fragment receives are identical.
//
// Fragments for step ks+1 are loaded before the mma block of step ks
// (an explicit two-deep register buffer): with only four warps per SM,
// one warp must keep its sub-partition's tensor pipe fed, and that
// works only if the next step's operands are already in flight.
//
// A rung joins the dispatch ladder only after an on-box census proves
// per-element byte identity and a measured win.

#define GBFW64_BM 128
#define GBFW64_BN 128
#define GBFW64_BK 64
#define GBFW64_THREADS 128
// 16-byte chunks per staged row: A rows carry BK halves, B rows BN.
#define GBFW64_ACH (GBFW64_BK / 8)
#define GBFW64_BCH (GBFW64_BN / 8)
#define GBFW64_A_STAGE_BYTES (GBFW64_BM * GBFW64_ACH * 16)
#define GBFW64_B_STAGE_BYTES (GBFW64_BK * GBFW64_BCH * 16)
#define GBFW64_SWZ(chunk, row) ((chunk) ^ ((row) & 7))

// Stage one A(128x64) + B(64x128) tile pair into the swizzled layout
// via 16-byte cp.async with zero-fill tails; source pointers form only
// when bytes remain in the object (the same safety rule as every rung).
#define GBFW64_STAGE_ASYNC(buf, bkIdx)                                        \
    do {                                                                      \
        unsigned _as = As_sbase + (unsigned)((buf) * GBFW64_A_STAGE_BYTES);   \
        unsigned _bs = Bs_sbase + (unsigned)((buf) * GBFW64_B_STAGE_BYTES);   \
        for (int _i = threadIdx.x; _i < GBFW64_BM * GBFW64_ACH;               \
             _i += GBFW64_THREADS) {                                          \
            int _m = _i / GBFW64_ACH;                                         \
            int _c = _i % GBFW64_ACH;                                         \
            int _gr = pid_m * GBFW64_BM + _m;                                 \
            int _gc = (bkIdx) + _c * 8;                                       \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as + (unsigned)(_m * GBFW64_ACH * 16 +           \
                                             (GBFW64_SWZ(_c, _m) << 4));      \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&A[(long long)_gr * lda + _gc]                 \
                : (const void*)A;                                            \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBFW64_BK * GBFW64_BCH;               \
             _i += GBFW64_THREADS) {                                          \
            int _k = _i / GBFW64_BCH;                                         \
            int _c = _i % GBFW64_BCH;                                         \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBFW64_BN + _c * 8;                             \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs + (unsigned)(_k * GBFW64_BCH * 16 +           \
                                             (GBFW64_SWZ(_c, _k) << 4));      \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&B[(long long)_gk * ldb + _gn]                 \
                : (const void*)B;                                            \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

// Element-wise fallback for hostile strides/bases, writing the same
// swizzled layout: element (row, k) lands inside chunk k/8 at half
// k%8. Same values as the async path, including the zero-fill.
#define GBFW64_STAGE_SCALAR(buf, bkIdx, TT, FF)                               \
    do {                                                                      \
        unsigned char* _asw = As_bytes + (buf) * GBFW64_A_STAGE_BYTES;        \
        unsigned char* _bsw = Bs_bytes + (buf) * GBFW64_B_STAGE_BYTES;        \
        for (int _i = threadIdx.x; _i < GBFW64_BM * GBFW64_BK;                \
             _i += GBFW64_THREADS) {                                          \
            int _m = _i / GBFW64_BK;                                          \
            int _k = _i % GBFW64_BK;                                          \
            int _gr = pid_m * GBFW64_BM + _m;                                 \
            int _gc = (bkIdx) + _k;                                           \
            TT _v = (_gr < M && _gc < K) ? A[(long long)_gr * lda + _gc]      \
                                         : FF(0.0f);                          \
            *reinterpret_cast<TT*>(                                           \
                _asw + _m * GBFW64_ACH * 16 +                                 \
                (GBFW64_SWZ(_k >> 3, _m) << 4) + (_k & 7) * 2) = _v;          \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBFW64_BK * GBFW64_BN;                \
             _i += GBFW64_THREADS) {                                          \
            int _k = _i / GBFW64_BN;                                          \
            int _n = _i % GBFW64_BN;                                          \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBFW64_BN + _n;                                 \
            TT _v = (_gk < K && _gn < N) ? B[(long long)_gk * ldb + _gn]      \
                                         : FF(0.0f);                          \
            *reinterpret_cast<TT*>(                                           \
                _bsw + _k * GBFW64_BCH * 16 +                                 \
                (GBFW64_SWZ(_n >> 3, _k) << 4) + (_n & 7) * 2) = _v;          \
        }                                                                     \
    } while (0)

// Load step ks's fragments into register buffer `fb`: four .x4 loads
// for A (one per m-fragment) and four .x4.trans for B (each covering
// two adjacent n-fragments).
#define GBFW64_LOAD_FRAGS(fb, ksv)                                            \
    do {                                                                      \
        int _k0 = (ksv) * 16;                                                 \
        _Pragma("unroll")                                                     \
        for (int _fm = 0; _fm < 4; _fm++) {                                   \
            int _row = warpM + _fm * 16 + lm_row_off + lm_r;                  \
            int _chunk = (_k0 + lm_col_off) >> 3;                             \
            unsigned _addr = As_rd +                                          \
                (unsigned)(_row * GBFW64_ACH * 16 +                           \
                           (GBFW64_SWZ(_chunk, _row) << 4));                  \
            asm volatile(                                                     \
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "                   \
                "{%0,%1,%2,%3}, [%4];\n"                                      \
                : "=r"(a_frag[fb][_fm][0]), "=r"(a_frag[fb][_fm][1]),         \
                  "=r"(a_frag[fb][_fm][2]), "=r"(a_frag[fb][_fm][3])          \
                : "r"(_addr));                                                \
        }                                                                     \
        _Pragma("unroll")                                                     \
        for (int _j = 0; _j < 4; _j++) {                                      \
            int _row = _k0 + ((lm_q & 1) ? 8 : 0) + lm_r;                     \
            int _col = warpN + _j * 16 + ((lm_q & 2) ? 8 : 0);                \
            int _chunk = _col >> 3;                                           \
            unsigned _addr = Bs_rd +                                          \
                (unsigned)(_row * GBFW64_BCH * 16 +                           \
                           (GBFW64_SWZ(_chunk, _row) << 4));                  \
            asm volatile(                                                     \
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "             \
                "{%0,%1,%2,%3}, [%4];\n"                                      \
                : "=r"(b_frag[fb][2 * _j][0]), "=r"(b_frag[fb][2 * _j][1]),   \
                  "=r"(b_frag[fb][2 * _j + 1][0]),                            \
                  "=r"(b_frag[fb][2 * _j + 1][1])                             \
                : "r"(_addr));                                                \
        }                                                                     \
    } while (0)

#define DEFINE_GEMM_BI_NN_TCW64(SUFFIX, T_ACT, FROM_F, MMA_T)                 \
extern "C" __global__ __launch_bounds__(GBFW64_THREADS, 1)                    \
void nn_tcw64_##SUFFIX(                                               \
    T_ACT* __restrict__ C,                                                    \
    const T_ACT* __restrict__ A,                                              \
    const T_ACT* __restrict__ B,                                              \
    const float* __restrict__ bias,                                           \
    float alpha, float beta,                                                  \
    int M, int N, int K,                                                      \
    int lda, int ldb, int ldc                                                 \
) {                                                                           \
    assert(alpha == 1.0f || bias == nullptr);                                 \
    extern __shared__ __align__(16) unsigned char gbfw64_dynsmem[];           \
    unsigned char* As_bytes = gbfw64_dynsmem;                                 \
    unsigned char* Bs_bytes = gbfw64_dynsmem + 2 * GBFW64_A_STAGE_BYTES;      \
    int num_pid_n = (N + GBFW64_BN - 1) / GBFW64_BN;                          \
    int pid_m = blockIdx.x / num_pid_n;                                       \
    int pid_n = blockIdx.x % num_pid_n;                                       \
    int warp = threadIdx.x / 32;                                              \
    int lane = threadIdx.x % 32;                                              \
    int warpM = (warp >> 1) * 64;                                             \
    int warpN = (warp & 1) * 64;                                              \
    int g = lane >> 2;                                                        \
    int t = lane & 3;                                                         \
    int lm_r = lane & 7;                                                      \
    int lm_q = lane >> 3;                                                     \
    int lm_row_off = (lm_q & 1) ? 8 : 0;                                      \
    int lm_col_off = (lm_q & 2) ? 8 : 0;                                      \
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(As_bytes);         \
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(Bs_bytes);         \
    bool fast_stage = ((lda & 7) == 0) && ((ldb & 7) == 0) &&                 \
                      gbf_aligned16(A) && gbf_aligned16(B);                   \
    float acc[4][8][4];                                                       \
    _Pragma("unroll")                                                         \
    for (int fm = 0; fm < 4; fm++) {                                          \
        _Pragma("unroll")                                                     \
        for (int fn = 0; fn < 8; fn++) {                                      \
            float b0 = 0.0f, b1 = 0.0f;                                       \
            if (bias != nullptr) {                                            \
                int c0 = pid_n * GBFW64_BN + warpN + fn * 8 + 2 * t;          \
                b0 = (c0 < N) ? bias[c0] : 0.0f;                              \
                b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                      \
            }                                                                 \
            acc[fm][fn][0] = b0;                                              \
            acc[fm][fn][1] = b1;                                              \
            acc[fm][fn][2] = b0;                                              \
            acc[fm][fn][3] = b1;                                              \
        }                                                                     \
    }                                                                         \
    int num_k_tiles = (K + GBFW64_BK - 1) / GBFW64_BK;                        \
    if (fast_stage) {                                                         \
        GBFW64_STAGE_ASYNC(0, 0);                                             \
    } else {                                                                  \
        GBFW64_STAGE_SCALAR(0, 0, T_ACT, FROM_F);                             \
    }                                                                         \
    int read_buf = 0;                                                         \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                \
        if (fast_stage) {                                                     \
            asm volatile("cp.async.wait_group 0;\n");                         \
        }                                                                     \
        __syncthreads();                                                      \
        if (kt + 1 < num_k_tiles) {                                           \
            if (fast_stage) {                                                 \
                GBFW64_STAGE_ASYNC(read_buf ^ 1, (kt + 1) * GBFW64_BK);       \
            } else {                                                          \
                GBFW64_STAGE_SCALAR(read_buf ^ 1, (kt + 1) * GBFW64_BK,       \
                                    T_ACT, FROM_F);                           \
            }                                                                 \
        }                                                                     \
        unsigned As_rd =                                                      \
            As_sbase + (unsigned)(read_buf * GBFW64_A_STAGE_BYTES);           \
        unsigned Bs_rd =                                                      \
            Bs_sbase + (unsigned)(read_buf * GBFW64_B_STAGE_BYTES);           \
        unsigned a_frag[2][4][4];                                             \
        unsigned b_frag[2][8][2];                                             \
        GBFW64_LOAD_FRAGS(0, 0);                                              \
        _Pragma("unroll")                                                     \
        for (int ks = 0; ks < (GBFW64_BK / 16); ks++) {                       \
            int fb = ks & 1;                                                  \
            if (ks + 1 < (GBFW64_BK / 16)) {                                  \
                GBFW64_LOAD_FRAGS(fb ^ 1, ks + 1);                            \
            }                                                                 \
            _Pragma("unroll")                                                 \
            for (int fm = 0; fm < 4; fm++) {                                  \
                _Pragma("unroll")                                             \
                for (int fn = 0; fn < 8; fn++) {                              \
                    asm volatile(                                             \
                        "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "."    \
                        MMA_T ".f32 "                                         \
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "             \
                        "{%0,%1,%2,%3};\n"                                    \
                        : "+f"(acc[fm][fn][0]), "+f"(acc[fm][fn][1]),         \
                          "+f"(acc[fm][fn][2]), "+f"(acc[fm][fn][3])          \
                        : "r"(a_frag[fb][fm][0]), "r"(a_frag[fb][fm][1]),     \
                          "r"(a_frag[fb][fm][2]), "r"(a_frag[fb][fm][3]),     \
                          "r"(b_frag[fb][fn][0]), "r"(b_frag[fb][fn][1]));    \
                }                                                             \
            }                                                                 \
        }                                                                     \
        read_buf ^= 1;                                                        \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int fm = 0; fm < 4; fm++) {                                          \
        _Pragma("unroll")                                                     \
        for (int fn = 0; fn < 8; fn++) {                                      \
            int r0 = pid_m * GBFW64_BM + warpM + fm * 16 + g;                 \
            int c0 = pid_n * GBFW64_BN + warpN + fn * 8 + 2 * t;              \
            _Pragma("unroll")                                                 \
            for (int half = 0; half < 2; half++) {                            \
                int gr = r0 + (half ? 8 : 0);                                 \
                if (gr >= M) continue;                                        \
                float v0 = __fmul_rn(alpha, acc[fm][fn][2 * half]);           \
                float v1 = __fmul_rn(alpha, acc[fm][fn][2 * half + 1]);       \
                T_ACT* _dst = (c0 < N)                                        \
                    ? &C[(long long)gr * ldc + c0]                            \
                    : (T_ACT*)0;                                              \
                if (beta == 0.0f && (ldc & 1) == 0 && c0 + 1 < N &&           \
                    gbf_aligned4(_dst)) {                                     \
                    gbf_store_pair_rne(_dst, v0, v1);                         \
                } else {                                                      \
                    for (int e = 0; e < 2; e++) {                             \
                        int gc = c0 + e;                                      \
                        if (gc >= N) continue;                                \
                        float val = e ? v1 : v0;                              \
                        if (beta != 0.0f)                                     \
                            val = __fmaf_rn(                                  \
                                beta, to_f(C[(long long)gr * ldc + gc]),      \
                                val);                                         \
                        C[(long long)gr * ldc + gc] = FROM_F(val);            \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

DEFINE_GEMM_BI_NN_TCW64(bf16, __nv_bfloat16, from_f_bf16, "bf16")
DEFINE_GEMM_BI_NN_TCW64(f16,  __half,        from_f_f16,  "f16")

#undef GBFW64_BM
#undef GBFW64_BN
#undef GBFW64_BK
#undef GBFW64_THREADS
#undef GBFW64_ACH
#undef GBFW64_BCH
#undef GBFW64_A_STAGE_BYTES
#undef GBFW64_B_STAGE_BYTES
#undef GBFW64_SWZ
#undef GBFW64_STAGE_ASYNC
#undef GBFW64_STAGE_SCALAR
#undef GBFW64_LOAD_FRAGS
#undef DEFINE_GEMM_BI_NN_TCW64

// ============================================================================
// GBFWN64: the same 64x64 warp tile at CTA 128x256 with 8 warps.
// ============================================================================
// The four-warp constant-area variant measured behind the shipped
// 128-tile everywhere: one warp per sub-partition cannot hide fragment
// latency even with the explicit double buffer. This variant keeps the
// fragment-reuse arithmetic and restores 8 warps per SM by widening the
// CTA to 128x256 (98,304 B of staged operands - inside the opt-in cap
// only because the swizzle is pad-free). It trades wave granularity
// for it, so its wins split by shape - the dispatch rule admits it
// only where the measured grid showed a win.
// Same numeric contract, same per-element mma chain: bit-identical to
// the 128-tile by the same argument, and censused the same way.
#define GBFWN64_BM 128
#define GBFWN64_BN 256
#define GBFWN64_BK 64
#define GBFWN64_THREADS 256
// 16-byte chunks per staged row: A rows carry BK halves, B rows BN.
#define GBFWN64_ACH (GBFWN64_BK / 8)
#define GBFWN64_BCH (GBFWN64_BN / 8)
#define GBFWN64_A_STAGE_BYTES (GBFWN64_BM * GBFWN64_ACH * 16)
#define GBFWN64_B_STAGE_BYTES (GBFWN64_BK * GBFWN64_BCH * 16)
#define GBFWN64_SWZ(chunk, row) ((chunk) ^ ((row) & 7))

// Stage one A(128x64) + B(64x128) tile pair into the swizzled layout
// via 16-byte cp.async with zero-fill tails; source pointers form only
// when bytes remain in the object (the same safety rule as every rung).
#define GBFWN64_STAGE_ASYNC(buf, bkIdx)                                        \
    do {                                                                      \
        unsigned _as = As_sbase + (unsigned)((buf) * GBFWN64_A_STAGE_BYTES);   \
        unsigned _bs = Bs_sbase + (unsigned)((buf) * GBFWN64_B_STAGE_BYTES);   \
        for (int _i = threadIdx.x; _i < GBFWN64_BM * GBFWN64_ACH;               \
             _i += GBFWN64_THREADS) {                                          \
            int _m = _i / GBFWN64_ACH;                                         \
            int _c = _i % GBFWN64_ACH;                                         \
            int _gr = pid_m * GBFWN64_BM + _m;                                 \
            int _gc = (bkIdx) + _c * 8;                                       \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as + (unsigned)(_m * GBFWN64_ACH * 16 +           \
                                             (GBFWN64_SWZ(_c, _m) << 4));      \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&A[(long long)_gr * lda + _gc]                 \
                : (const void*)A;                                            \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBFWN64_BK * GBFWN64_BCH;               \
             _i += GBFWN64_THREADS) {                                          \
            int _k = _i / GBFWN64_BCH;                                         \
            int _c = _i % GBFWN64_BCH;                                         \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBFWN64_BN + _c * 8;                             \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs + (unsigned)(_k * GBFWN64_BCH * 16 +           \
                                             (GBFWN64_SWZ(_c, _k) << 4));      \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&B[(long long)_gk * ldb + _gn]                 \
                : (const void*)B;                                            \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

// Element-wise fallback for hostile strides/bases, writing the same
// swizzled layout: element (row, k) lands inside chunk k/8 at half
// k%8. Same values as the async path, including the zero-fill.
#define GBFWN64_STAGE_SCALAR(buf, bkIdx, TT, FF)                               \
    do {                                                                      \
        unsigned char* _asw = As_bytes + (buf) * GBFWN64_A_STAGE_BYTES;        \
        unsigned char* _bsw = Bs_bytes + (buf) * GBFWN64_B_STAGE_BYTES;        \
        for (int _i = threadIdx.x; _i < GBFWN64_BM * GBFWN64_BK;                \
             _i += GBFWN64_THREADS) {                                          \
            int _m = _i / GBFWN64_BK;                                          \
            int _k = _i % GBFWN64_BK;                                          \
            int _gr = pid_m * GBFWN64_BM + _m;                                 \
            int _gc = (bkIdx) + _k;                                           \
            TT _v = (_gr < M && _gc < K) ? A[(long long)_gr * lda + _gc]      \
                                         : FF(0.0f);                          \
            *reinterpret_cast<TT*>(                                           \
                _asw + _m * GBFWN64_ACH * 16 +                                 \
                (GBFWN64_SWZ(_k >> 3, _m) << 4) + (_k & 7) * 2) = _v;          \
        }                                                                     \
        for (int _i = threadIdx.x; _i < GBFWN64_BK * GBFWN64_BN;                \
             _i += GBFWN64_THREADS) {                                          \
            int _k = _i / GBFWN64_BN;                                          \
            int _n = _i % GBFWN64_BN;                                          \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * GBFWN64_BN + _n;                                 \
            TT _v = (_gk < K && _gn < N) ? B[(long long)_gk * ldb + _gn]      \
                                         : FF(0.0f);                          \
            *reinterpret_cast<TT*>(                                           \
                _bsw + _k * GBFWN64_BCH * 16 +                                 \
                (GBFWN64_SWZ(_n >> 3, _k) << 4) + (_n & 7) * 2) = _v;          \
        }                                                                     \
    } while (0)

// Load step ks's fragments into register buffer `fb`: four .x4 loads
// for A (one per m-fragment) and four .x4.trans for B (each covering
// two adjacent n-fragments).
#define GBFWN64_LOAD_FRAGS(fb, ksv)                                            \
    do {                                                                      \
        int _k0 = (ksv) * 16;                                                 \
        _Pragma("unroll")                                                     \
        for (int _fm = 0; _fm < 4; _fm++) {                                   \
            int _row = warpM + _fm * 16 + lm_row_off + lm_r;                  \
            int _chunk = (_k0 + lm_col_off) >> 3;                             \
            unsigned _addr = As_rd +                                          \
                (unsigned)(_row * GBFWN64_ACH * 16 +                           \
                           (GBFWN64_SWZ(_chunk, _row) << 4));                  \
            asm volatile(                                                     \
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "                   \
                "{%0,%1,%2,%3}, [%4];\n"                                      \
                : "=r"(a_frag[fb][_fm][0]), "=r"(a_frag[fb][_fm][1]),         \
                  "=r"(a_frag[fb][_fm][2]), "=r"(a_frag[fb][_fm][3])          \
                : "r"(_addr));                                                \
        }                                                                     \
        _Pragma("unroll")                                                     \
        for (int _j = 0; _j < 4; _j++) {                                      \
            int _row = _k0 + ((lm_q & 1) ? 8 : 0) + lm_r;                     \
            int _col = warpN + _j * 16 + ((lm_q & 2) ? 8 : 0);                \
            int _chunk = _col >> 3;                                           \
            unsigned _addr = Bs_rd +                                          \
                (unsigned)(_row * GBFWN64_BCH * 16 +                           \
                           (GBFWN64_SWZ(_chunk, _row) << 4));                  \
            asm volatile(                                                     \
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "             \
                "{%0,%1,%2,%3}, [%4];\n"                                      \
                : "=r"(b_frag[fb][2 * _j][0]), "=r"(b_frag[fb][2 * _j][1]),   \
                  "=r"(b_frag[fb][2 * _j + 1][0]),                            \
                  "=r"(b_frag[fb][2 * _j + 1][1])                             \
                : "r"(_addr));                                                \
        }                                                                     \
    } while (0)

#define DEFINE_GEMM_BI_NN_TCWN64(SUFFIX, T_ACT, FROM_F, MMA_T)                 \
extern "C" __global__ __launch_bounds__(GBFWN64_THREADS, 1)                    \
void nn_tcwn64_##SUFFIX(                                               \
    T_ACT* __restrict__ C,                                                    \
    const T_ACT* __restrict__ A,                                              \
    const T_ACT* __restrict__ B,                                              \
    const float* __restrict__ bias,                                           \
    float alpha, float beta,                                                  \
    int M, int N, int K,                                                      \
    int lda, int ldb, int ldc                                                 \
) {                                                                           \
    assert(alpha == 1.0f || bias == nullptr);                                 \
    extern __shared__ __align__(16) unsigned char gbfwn64_dynsmem[];           \
    unsigned char* As_bytes = gbfwn64_dynsmem;                                 \
    unsigned char* Bs_bytes = gbfwn64_dynsmem + 2 * GBFWN64_A_STAGE_BYTES;      \
    int num_pid_n = (N + GBFWN64_BN - 1) / GBFWN64_BN;                          \
    int pid_m = blockIdx.x / num_pid_n;                                       \
    int pid_n = blockIdx.x % num_pid_n;                                       \
    int warp = threadIdx.x / 32;                                              \
    int lane = threadIdx.x % 32;                                              \
    int warpM = (warp >> 2) * 64;                                             \
    int warpN = (warp & 3) * 64;                                              \
    int g = lane >> 2;                                                        \
    int t = lane & 3;                                                         \
    int lm_r = lane & 7;                                                      \
    int lm_q = lane >> 3;                                                     \
    int lm_row_off = (lm_q & 1) ? 8 : 0;                                      \
    int lm_col_off = (lm_q & 2) ? 8 : 0;                                      \
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(As_bytes);         \
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(Bs_bytes);         \
    bool fast_stage = ((lda & 7) == 0) && ((ldb & 7) == 0) &&                 \
                      gbf_aligned16(A) && gbf_aligned16(B);                   \
    float acc[4][8][4];                                                       \
    _Pragma("unroll")                                                         \
    for (int fm = 0; fm < 4; fm++) {                                          \
        _Pragma("unroll")                                                     \
        for (int fn = 0; fn < 8; fn++) {                                      \
            float b0 = 0.0f, b1 = 0.0f;                                       \
            if (bias != nullptr) {                                            \
                int c0 = pid_n * GBFWN64_BN + warpN + fn * 8 + 2 * t;          \
                b0 = (c0 < N) ? bias[c0] : 0.0f;                              \
                b1 = (c0 + 1 < N) ? bias[c0 + 1] : 0.0f;                      \
            }                                                                 \
            acc[fm][fn][0] = b0;                                              \
            acc[fm][fn][1] = b1;                                              \
            acc[fm][fn][2] = b0;                                              \
            acc[fm][fn][3] = b1;                                              \
        }                                                                     \
    }                                                                         \
    int num_k_tiles = (K + GBFWN64_BK - 1) / GBFWN64_BK;                        \
    if (fast_stage) {                                                         \
        GBFWN64_STAGE_ASYNC(0, 0);                                             \
    } else {                                                                  \
        GBFWN64_STAGE_SCALAR(0, 0, T_ACT, FROM_F);                             \
    }                                                                         \
    int read_buf = 0;                                                         \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                \
        if (fast_stage) {                                                     \
            asm volatile("cp.async.wait_group 0;\n");                         \
        }                                                                     \
        __syncthreads();                                                      \
        if (kt + 1 < num_k_tiles) {                                           \
            if (fast_stage) {                                                 \
                GBFWN64_STAGE_ASYNC(read_buf ^ 1, (kt + 1) * GBFWN64_BK);       \
            } else {                                                          \
                GBFWN64_STAGE_SCALAR(read_buf ^ 1, (kt + 1) * GBFWN64_BK,       \
                                    T_ACT, FROM_F);                           \
            }                                                                 \
        }                                                                     \
        unsigned As_rd =                                                      \
            As_sbase + (unsigned)(read_buf * GBFWN64_A_STAGE_BYTES);           \
        unsigned Bs_rd =                                                      \
            Bs_sbase + (unsigned)(read_buf * GBFWN64_B_STAGE_BYTES);           \
        unsigned a_frag[2][4][4];                                             \
        unsigned b_frag[2][8][2];                                             \
        GBFWN64_LOAD_FRAGS(0, 0);                                              \
        _Pragma("unroll")                                                     \
        for (int ks = 0; ks < (GBFWN64_BK / 16); ks++) {                       \
            int fb = ks & 1;                                                  \
            if (ks + 1 < (GBFWN64_BK / 16)) {                                  \
                GBFWN64_LOAD_FRAGS(fb ^ 1, ks + 1);                            \
            }                                                                 \
            _Pragma("unroll")                                                 \
            for (int fm = 0; fm < 4; fm++) {                                  \
                _Pragma("unroll")                                             \
                for (int fn = 0; fn < 8; fn++) {                              \
                    asm volatile(                                             \
                        "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "."    \
                        MMA_T ".f32 "                                         \
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "             \
                        "{%0,%1,%2,%3};\n"                                    \
                        : "+f"(acc[fm][fn][0]), "+f"(acc[fm][fn][1]),         \
                          "+f"(acc[fm][fn][2]), "+f"(acc[fm][fn][3])          \
                        : "r"(a_frag[fb][fm][0]), "r"(a_frag[fb][fm][1]),     \
                          "r"(a_frag[fb][fm][2]), "r"(a_frag[fb][fm][3]),     \
                          "r"(b_frag[fb][fn][0]), "r"(b_frag[fb][fn][1]));    \
                }                                                             \
            }                                                                 \
        }                                                                     \
        read_buf ^= 1;                                                        \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int fm = 0; fm < 4; fm++) {                                          \
        _Pragma("unroll")                                                     \
        for (int fn = 0; fn < 8; fn++) {                                      \
            int r0 = pid_m * GBFWN64_BM + warpM + fm * 16 + g;                 \
            int c0 = pid_n * GBFWN64_BN + warpN + fn * 8 + 2 * t;              \
            _Pragma("unroll")                                                 \
            for (int half = 0; half < 2; half++) {                            \
                int gr = r0 + (half ? 8 : 0);                                 \
                if (gr >= M) continue;                                        \
                float v0 = __fmul_rn(alpha, acc[fm][fn][2 * half]);           \
                float v1 = __fmul_rn(alpha, acc[fm][fn][2 * half + 1]);       \
                T_ACT* _dst = (c0 < N)                                        \
                    ? &C[(long long)gr * ldc + c0]                            \
                    : (T_ACT*)0;                                              \
                if (beta == 0.0f && (ldc & 1) == 0 && c0 + 1 < N &&           \
                    gbf_aligned4(_dst)) {                                     \
                    gbf_store_pair_rne(_dst, v0, v1);                         \
                } else {                                                      \
                    for (int e = 0; e < 2; e++) {                             \
                        int gc = c0 + e;                                      \
                        if (gc >= N) continue;                                \
                        float val = e ? v1 : v0;                              \
                        if (beta != 0.0f)                                     \
                            val = __fmaf_rn(                                  \
                                beta, to_f(C[(long long)gr * ldc + gc]),      \
                                val);                                         \
                        C[(long long)gr * ldc + gc] = FROM_F(val);            \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

DEFINE_GEMM_BI_NN_TCWN64(bf16, __nv_bfloat16, from_f_bf16, "bf16")
DEFINE_GEMM_BI_NN_TCWN64(f16,  __half,        from_f_f16,  "f16")

#undef GBFWN64_BM
#undef GBFWN64_BN
#undef GBFWN64_BK
#undef GBFWN64_THREADS
#undef GBFWN64_ACH
#undef GBFWN64_BCH
#undef GBFWN64_A_STAGE_BYTES
#undef GBFWN64_B_STAGE_BYTES
#undef GBFWN64_SWZ
#undef GBFWN64_STAGE_ASYNC
#undef GBFWN64_STAGE_SCALAR
#undef GBFWN64_LOAD_FRAGS
#undef DEFINE_GEMM_BI_NN_TCWN64
#line 1 "kernels/gemm_bi_inference/sm90a/wgmma.cu"
// ============================================================================
// Hopper (sm_90a) inference rung: NN forward through wgmma.mma_async.
// ============================================================================
// One warpgroup (128 threads) owns a 64x128 output tile and walks K in
// ascending 64-wide slabs, each slab issued as four ascending
// wgmma.mma_async.m64n128k16 steps into the same f32 accumulators. The
// K-slab order, the tail zero-fill and the single RNE downcast at the
// store mirror the sm_89 ladder's contract. Bias does NOT: it joins in
// the epilogue (after alpha, before beta) rather than pre-seeding the
// accumulators, because the first wgmma group then runs with
// scale-d = 0 and ptxas stops serializing it against a register init
// chain. This is the rung's own numeric contract either way: whether
// its bits EQUAL the mma.sync ladder on the same inputs is a hardware
// question (Hopper's tensor core sums sixteen products per internal
// block where Ada sums eight), answered by the forced-entry census on
// a real sm_90a part before any dispatch cell may route here.
//
// Operands stage through cp.async into shared memory laid out in the
// 128-byte swizzle the wgmma descriptors declare: the 16-byte chunk at
// logical column c of row r lives at physical chunk c ^ (r & 7). The
// swizzle only relocates bytes, never values. cp.async writes land on
// the generic proxy while wgmma reads through the async proxy, so a
// fence.proxy.async makes the staged bytes visible before each slab is
// consumed.
//
// Everything below compiles only for sm_90a: wgmma exists on Hopper
// alone (consumer Blackwell dropped it for mma.sync, datacenter
// Blackwell replaced it with tcgen05), and the loader looks these
// symbols up only when the device resolves to sm_90a.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900

// Shared-memory matrix descriptor: 14-bit fields in 16-byte units, the
// 128-byte-swizzle layout tag in bits 62-63.
static __device__ __forceinline__ unsigned long long
sm90_desc(const void* smem_ptr, unsigned lbo16, unsigned sbo16) {
    unsigned long long addr =
        (unsigned long long)__cvta_generic_to_shared(smem_ptr);
    unsigned long long d = 0;
    d |= (addr >> 4) & 0x3FFFULL;
    d |= ((unsigned long long)(lbo16 & 0x3FFF)) << 16;
    d |= ((unsigned long long)(sbo16 & 0x3FFF)) << 32;
    d |= 1ULL << 62; // 128-byte swizzle
    return d;
}

#define SM90_BM 64
#define SM90_BN 128
#define SM90_BK 64
#define SM90_STAGES 2
// Row strides in elements (half-precision): pad-free, the swizzle owns
// bank-conflict freedom, and the descriptors assume contiguous rows.
#define SM90_LDA SM90_BK
#define SM90_LDB SM90_BN

// Stage one A(64x64) + B(64x128) tile pair into the swizzled layout.
// Every 16-byte chunk lands at (row, chunk ^ (row & 7)); out-of-range
// rows and K/N tails zero-fill so the mma sees exact zeros, matching
// the sm_89 ladder's tail rule. The source pointer is formed only when
// bytes remain in the object.
#define SM90_STAGE_ASYNC(buf, bkIdx)                                          \
    do {                                                                      \
        unsigned _as = As_sbase +                                             \
                       (unsigned)((buf) * SM90_BM * SM90_LDA * 2);            \
        unsigned _bs = Bs_sbase +                                             \
                       (unsigned)((buf) * SM90_BK * SM90_LDB * 2);            \
        for (int _i = threadIdx.x; _i < SM90_BM * (SM90_BK / 8);              \
             _i += blockDim.x) {                                              \
            int _m = _i / (SM90_BK / 8);                                      \
            int _c = _i % (SM90_BK / 8);                                      \
            int _k = _c * 8;                                                  \
            int _gr = pid_m * SM90_BM + _m;                                   \
            int _gc = (bkIdx) + _k;                                           \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as +                                             \
                (unsigned)((_m * SM90_LDA + ((_c ^ (_m & 7)) * 8)) * 2);      \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&A[(long long)_gr * lda + _gc]                 \
                : (const void*)A;                                             \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SM90_BK * (SM90_BN / 8);              \
             _i += blockDim.x) {                                              \
            int _k = _i / (SM90_BN / 8);                                      \
            int _c = _i % (SM90_BN / 8);                                      \
            /* B lives as two complete 64-column blocks 8192 bytes      */    \
            /* apart - the layout the descriptor's leading offset       */    \
            /* (512 x 16 B) and the ks*128 slab advance declare. A      */    \
            /* row-interleaved stage here once contradicted them, which */    \
            /* would have made the first hardware census read garbage.  */    \
            int _h = _c >> 3;                                                 \
            int _cc = _c & 7;                                                 \
            int _n = _c * 8;                                                  \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * SM90_BN + _n;                                   \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs + (unsigned)(_h * 8192) +                     \
                (unsigned)((_k * 64 + ((_cc ^ (_k & 7)) * 8)) * 2);           \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&B[(long long)_gk * ldb + _gn]                 \
                : (const void*)B;                                             \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
    } while (0)

#define DEFINE_GEMM_BI_NN_SM90(SUFFIX, T_ACT, FROM_F, WG_T)                   \
extern "C" __global__ __launch_bounds__(128, 1)                               \
void nn_sm90a_wgmma_wg1_##SUFFIX(                                     \
    T_ACT* __restrict__ C,                                                    \
    const T_ACT* __restrict__ A,                                              \
    const T_ACT* __restrict__ B,                                              \
    const float* __restrict__ bias,                                           \
    float alpha, float beta,                                                  \
    int M, int N, int K,                                                      \
    int lda, int ldb, int ldc                                                 \
) {                                                                           \
    extern __shared__ __align__(1024) unsigned char sm90_dynsmem[];           \
    T_ACT* As = reinterpret_cast<T_ACT*>(sm90_dynsmem);                       \
    T_ACT* Bs = reinterpret_cast<T_ACT*>(                                     \
        sm90_dynsmem + SM90_STAGES * SM90_BM * SM90_LDA * sizeof(T_ACT));     \
    int num_pid_n = (N + SM90_BN - 1) / SM90_BN;                              \
    int pid_m = blockIdx.x / num_pid_n;                                       \
    int pid_n = blockIdx.x % num_pid_n;                                       \
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(As);               \
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(Bs);               \
    int wg_tid = threadIdx.x;                                                 \
    /* Accumulators: m64n128 spreads 64 f32 per thread over the group. */     \
    float acc[64];                                                            \
    int q = wg_tid & 3;                                                       \
    int row8 = (wg_tid >> 2) & 7;                                             \
    int warp = wg_tid >> 5;                                                   \
    /* No accumulator pre-seed: the first wgmma group runs with          */   \
    /* scale-d = 0, which zeroes D regardless of register contents and   */   \
    /* frees ptxas from serializing the first group against an init      */   \
    /* chain. Bias joins in the epilogue - after alpha, before beta -    */   \
    /* which is this rung's own contract (the census decides its family  */   \
    /* membership either way; Hopper's wider internal reduce already     */   \
    /* makes bit-equality with the mma.sync ladder a hardware question). */   \
    _Pragma("unroll")                                                         \
    for (int r = 0; r < 64; r++) {                                            \
        acc[r] = 0.0f;                                                        \
    }                                                                         \
    int num_k_tiles = (K + SM90_BK - 1) / SM90_BK;                            \
    SM90_STAGE_ASYNC(0, 0);                                                   \
    asm volatile("cp.async.commit_group;\n");                                 \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                \
        asm volatile("cp.async.wait_group 0;\n");                             \
        __syncthreads();                                                      \
        /* Staged bytes arrived on the generic proxy; make them visible */    \
        /* to the async proxy the wgmma reads through.                  */    \
        asm volatile("fence.proxy.async.shared::cta;\n");                     \
        int rd = kt & (SM90_STAGES - 1);                                      \
        unsigned long long a_base = sm90_desc(                                \
            As + rd * SM90_BM * SM90_LDA, 1, 64);                             \
        unsigned long long b_base = sm90_desc(                                \
            Bs + rd * SM90_BK * SM90_LDB, 512, 64);                           \
        /* The next stage's copies are issued BEFORE this slab's wgmma  */    \
        /* so they overlap it. WAR-safe: buffer (kt+1)&1 was consumed   */    \
        /* by the wgmma of iteration kt-1, whose wait and barrier have  */    \
        /* both passed.                                                 */    \
        if (kt + 1 < num_k_tiles) {                                           \
            SM90_STAGE_ASYNC((kt + 1) & (SM90_STAGES - 1),                    \
                             (kt + 1) * SM90_BK);                             \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
        asm volatile("wgmma.fence.sync.aligned;\n");                          \
        _Pragma("unroll")                                                     \
        for (int ks = 0; ks < SM90_BK / 16; ks++) {                           \
            unsigned long long da = a_base + (unsigned long long)(ks * 2);    \
            unsigned long long db = b_base + (unsigned long long)(ks * 128);  \
            unsigned scale_d = (kt == 0 && ks == 0) ? 0u : 1u;                \
            asm volatile(                                                     \
                "{.reg .pred p;\n\t"                                          \
                "setp.ne.b32 p, %66, 0;\n\t"                                  \
                "wgmma.mma_async.sync.aligned.m64n128k16.f32." WG_T "." WG_T  \
                " {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"    \
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,"    \
                "%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,"    \
                "%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,"    \
                "%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 0, 1;}\n"       \
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),     \
                  "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]),     \
                  "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]),   \
                  "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]), \
                  "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]), \
                  "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]), \
                  "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]), \
                  "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]), \
                  "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]), \
                  "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]), \
                  "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]), \
                  "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]), \
                  "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]), \
                  "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]), \
                  "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]), \
                  "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])  \
                : "l"(da), "l"(db), "r"(scale_d));                            \
        }                                                                     \
        asm volatile("wgmma.commit_group.sync.aligned;\n");                   \
        asm volatile("wgmma.wait_group.sync.aligned 0;\n");                   \
        __syncthreads();                                                      \
    }                                                                         \
    /* Epilogue: alpha through an explicit unfused multiply (target-  */      \
    /* independent bits), paired store on aligned even destinations,  */      \
    /* scalar RNE tail otherwise - the same rules as the sm_89 tiles. */      \
    _Pragma("unroll")                                                         \
    for (int r = 0; r < 64; r += 2) {                                         \
        int pair_row = (r >> 1) & 1;                                          \
        int n_group = r >> 2;                                                 \
        int row = pid_m * SM90_BM + row8 + 16 * warp + 8 * pair_row;          \
        int col = pid_n * SM90_BN + 2 * q + 8 * n_group;                      \
        if (row >= M) continue;                                               \
        float v0 = __fmul_rn(alpha, acc[r]);                                  \
        float v1 = __fmul_rn(alpha, acc[r + 1]);                              \
        if (bias != nullptr) {                                                \
            if (col < N) v0 = __fadd_rn(v0, bias[col]);                       \
            if (col + 1 < N) v1 = __fadd_rn(v1, bias[col + 1]);               \
        }                                                                     \
        T_ACT* dst = (col < N) ? &C[(long long)row * ldc + col] : (T_ACT*)0;  \
        bool packed = beta == 0.0f && (ldc & 1) == 0 && col + 1 < N &&        \
                      ((reinterpret_cast<unsigned long long>(dst) & 3u)       \
                       == 0u);                                                \
        if (packed) {                                                         \
            gbf_store_pair_rne(dst, v0, v1);                                  \
        } else {                                                              \
            for (int e = 0; e < 2; e++) {                                     \
                int gc = col + e;                                             \
                if (gc >= N) continue;                                        \
                float val = e ? v1 : v0;                                      \
                if (beta != 0.0f)                                             \
                    val = __fmaf_rn(beta, to_f(C[(long long)row * ldc + gc]), val);         \
                C[(long long)row * ldc + gc] = FROM_F(val);                   \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

DEFINE_GEMM_BI_NN_SM90(bf16, __nv_bfloat16, from_f_bf16, "bf16")
DEFINE_GEMM_BI_NN_SM90(f16,  __half,        from_f_f16,  "f16")

#undef SM90_BM
#undef SM90_BN
#undef SM90_BK
#undef SM90_STAGES
#undef SM90_LDA
#undef SM90_LDB
#undef SM90_STAGE_ASYNC
#undef DEFINE_GEMM_BI_NN_SM90

#endif
#line 1 "kernels/gemm_bi_inference/sm100/tcgen05.cu"
// ============================================================================
// Datacenter Blackwell (sm_100 family) inference rung: NN forward through
// tcgen05.mma with f32 accumulation in Tensor Memory.
// ============================================================================
// One 128-thread CTA owns a 128x128 output tile and walks K in ascending
// 64-wide slabs, each slab issued as four ascending tcgen05.mma K16 steps
// into the same TMEM accumulator. The K-slab order, the tail zero-fill
// and the single RNE downcast at the store mirror the mma.sync ladder's
// contract; whether the bits EQUAL that ladder on the same inputs is a
// hardware question (the fifth-generation tensor core sums a different
// internal block width), answered by the forced-entry census on a real
// CC 10.x part before any dispatch cell may route here.
//
// The accumulator does not live in registers: tcgen05 accumulates into
// Tensor Memory, a per-SM 512-column x 128-lane f32 array. The CTA
// allocates 128 columns once (warp-issued, power-of-two, explicitly
// deallocated), seeds them with bias through tcgen05.st when bias is
// present (the first MMA then runs with enable-input-d, preserving the
// bias-before-reduction placement of every other rung), and reads them
// back warp-by-warp with tcgen05.ld: warp w owns TMEM lanes 32w..32w+31,
// so each output element has exactly one store owner and the epilogue
// needs no reduction, no shuffle and no atomics.
//
// Operands stage through cp.async into shared memory laid out in the
// 128-byte swizzle the tcgen05 matrix descriptors declare: the 16-byte
// chunk at logical column c of row r lives at physical chunk c ^ (r & 7).
// A is K-major (a 128-byte row per output row), B is N-major in two
// complete 64-column halves 8192 bytes apart, which is the canonical
// tiling the descriptor's leading-offset field encodes. cp.async writes
// land on the generic proxy while tcgen05 reads through the async proxy,
// so a fence.proxy.async plus the tcgen05 thread-sync fences make the
// staged bytes visible before each slab is consumed. Hardware TMA is the
// natural transport upgrade for this rung and can replace the staging
// loops at qualification time without moving a single output bit: both
// transports deliver the same bytes to the same swizzled addresses.
//
// Stage reuse is gated by the tensor core itself: each slab's MMAs are
// followed by tcgen05.commit, which arrives on an mbarrier only when the
// tracked MMA work no longer reads the stage; every thread waits on that
// barrier's phase before the buffer is refilled.
//
// Everything below compiles only for datacenter-Blackwell feature targets.
// CUDA 12.8 exposes exact targets through __CUDA_ARCH_FEAT_SM*_ALL; CUDA
// 12.9 and newer also expose the public architecture/family macros. Baseline
// targets stay on the portable ladder because they may not issue tcgen05.
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || \
    defined(__CUDA_ARCH_FEAT_SM101_ALL) || \
    defined(__CUDA_ARCH_FEAT_SM103_ALL) || \
    defined(__CUDA_ARCH_FEAT_SM110_ALL) || \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
     (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 || \
      __CUDA_ARCH_FAMILY_SPECIFIC__ == 1010 || \
      __CUDA_ARCH_FAMILY_SPECIFIC__ == 1030 || \
      __CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))

// Shared-memory matrix descriptor for tcgen05: 14-bit address fields in
// 16-byte units, version 1, the 128-byte-swizzle layout tag in bits
// 61-63. Distinct from the wgmma descriptor (version and layout encode
// differently); never share builders between the two families.
static __device__ __forceinline__ unsigned long long
sm100_desc(const void* smem_ptr, unsigned lbo16, unsigned sbo16) {
    unsigned long long addr =
        (unsigned long long)__cvta_generic_to_shared(smem_ptr);
    unsigned long long d = 0;
    d |= (addr >> 4) & 0x3FFFULL;
    d |= ((unsigned long long)(lbo16 & 0x3FFF)) << 16;
    d |= ((unsigned long long)(sbo16 & 0x3FFF)) << 32;
    /* Version sits ABOVE the stride field (bits 46-47), not in the
     * low half - a version bit misplaced at bit 14 presents a
     * Hopper-format descriptor to a tensor core expecting version 1. */
    d |= 1ULL << 46; // descriptor version 1
    d |= 2ULL << 61; // 128-byte swizzle
    return d;
}

#define SM100_BM 128
#define SM100_BN 128
#define SM100_BK 64
#define SM100_STAGES 2
// Row strides in half-precision elements: pad-free, the swizzle owns
// bank-conflict freedom, and the descriptors assume contiguous rows.
#define SM100_LDA SM100_BK
#define SM100_LDB 64
#define SM100_A_BYTES (SM100_BM * SM100_LDA * 2)
#define SM100_STAGE_BYTES (SM100_A_BYTES + SM100_BK * SM100_BN * 2)

// Instruction descriptors are fixed constants encoding f32 D, the A/B
// type, both operands' major mode, M=128 and N=128 for the NN op.
#define SM100_IDESC_F16 0x08210010u
#define SM100_IDESC_BF16 0x08210490u

// Stage one A(128x64) + B(64x128) tile pair into the swizzled layout.
// Every 16-byte chunk lands at (row, chunk ^ (row & 7)); out-of-range
// rows and K/N tails zero-fill so the mma sees exact zeros, matching
// the mma.sync ladder's tail rule. The source pointer is formed only
// when bytes remain in the object. B's two 64-column halves live 8192
// bytes apart, the separation its descriptor declares.
#define SM100_STAGE_ASYNC(buf, bkIdx)                                         \
    do {                                                                      \
        unsigned _as = sm_sbase + (unsigned)((buf) * SM100_STAGE_BYTES);      \
        unsigned _bs = _as + (unsigned)SM100_A_BYTES;                         \
        for (int _i = threadIdx.x; _i < SM100_BM * (SM100_BK / 8);            \
             _i += blockDim.x) {                                              \
            int _m = _i / (SM100_BK / 8);                                     \
            int _c = _i % (SM100_BK / 8);                                     \
            int _gr = pid_m * SM100_BM + _m;                                  \
            int _gc = (bkIdx) + _c * 8;                                       \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as +                                             \
                (unsigned)((_m * SM100_LDA + ((_c ^ (_m & 7)) * 8)) * 2);     \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&A[(long long)_gr * lda + _gc]                 \
                : (const void*)A;                                            \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < 2 * SM100_BK * (64 / 8);              \
             _i += blockDim.x) {                                              \
            int _h = _i / (SM100_BK * 8);                                     \
            int _r = (_i / 8) % SM100_BK;                                     \
            int _c = _i % 8;                                                  \
            int _gk = (bkIdx) + _r;                                           \
            int _gn = pid_n * SM100_BN + _h * 64 + _c * 8;                    \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs + (unsigned)(_h * 8192) +                     \
                (unsigned)((_r * SM100_LDB + ((_c ^ (_r & 7)) * 8)) * 2);     \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&B[(long long)_gk * ldb + _gn]                 \
                : (const void*)B;                                            \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
    } while (0)

#define DEFINE_GEMM_BI_NN_SM100(SUFFIX, T_ACT, FROM_F, IDESC)                 \
extern "C" __global__ __launch_bounds__(128, 1)                               \
void nn_sm100_tcgen_c4_##SUFFIX(                                      \
    T_ACT* __restrict__ C,                                                    \
    const T_ACT* __restrict__ A,                                              \
    const T_ACT* __restrict__ B,                                              \
    const float* __restrict__ bias,                                           \
    float alpha, float beta,                                                  \
    int M, int N, int K,                                                      \
    int lda, int ldb, int ldc                                                 \
) {                                                                           \
    extern __shared__ __align__(1024) unsigned char sm100_dynsmem[];          \
    __shared__ __align__(8) unsigned long long empty_bar;                     \
    __shared__ unsigned tmem_base;                                            \
    unsigned sm_sbase = (unsigned)__cvta_generic_to_shared(sm100_dynsmem);    \
    int num_pid_n = (N + SM100_BN - 1) / SM100_BN;                            \
    int pid_m = blockIdx.x / num_pid_n;                                       \
    int pid_n = blockIdx.x % num_pid_n;                                       \
    int tid = threadIdx.x;                                                    \
    int warp = tid >> 5;                                                      \
    int lane = tid & 31;                                                      \
    /* TMEM allocation and the barrier are warp-zero duties: alloc is   */    \
    /* warp-issued, init is one thread plus the required init fence.    */    \
    if (warp == 0) {                                                          \
        unsigned _tb = (unsigned)__cvta_generic_to_shared(&tmem_base);        \
        asm volatile(                                                         \
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "        \
            "[%0], %1;" :: "r"(_tb), "r"(128u));                              \
        if (lane == 0) {                                                      \
            unsigned _mb = (unsigned)__cvta_generic_to_shared(&empty_bar);    \
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"             \
                         :: "r"(_mb));                                        \
            asm volatile("fence.mbarrier_init.release.cluster;");             \
        }                                                                     \
    }                                                                         \
    __syncthreads();                                                          \
    unsigned taddr = tmem_base;                                               \
    unsigned mbar = (unsigned)__cvta_generic_to_shared(&empty_bar);           \
    int has_bias = bias != nullptr ? 1 : 0;                                   \
    if (has_bias) {                                                           \
        /* Seed the accumulator with bias at its output column, exactly */    \
        /* like every other rung (alpha must be 1.0 with bias). Each    */    \
        /* warp stores its own 32 TMEM lanes, eight columns at a time;  */    \
        /* tail columns store zero.                                     */    \
        unsigned wrow = taddr + ((unsigned)(warp * 32) << 16);                \
        for (int c0 = 0; c0 < SM100_BN; c0 += 8) {                            \
            float bv[8];                                                      \
            _Pragma("unroll")                                                 \
            for (int j = 0; j < 8; j++) {                                     \
                int col = pid_n * SM100_BN + c0 + j;                          \
                bv[j] = col < N ? bias[col] : 0.0f;                           \
            }                                                                 \
            asm volatile(                                                     \
                "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], "                \
                "{%1,%2,%3,%4,%5,%6,%7,%8};"                                  \
                :: "r"(wrow + (unsigned)c0),                                  \
                   "f"(bv[0]), "f"(bv[1]), "f"(bv[2]), "f"(bv[3]),            \
                   "f"(bv[4]), "f"(bv[5]), "f"(bv[6]), "f"(bv[7]));           \
        }                                                                     \
        asm volatile("tcgen05.wait::st.sync.aligned;");                       \
        asm volatile("tcgen05.fence::before_thread_sync;");                   \
    }                                                                         \
    __syncthreads();                                                          \
    int num_k_tiles = (K + SM100_BK - 1) / SM100_BK;                          \
    SM100_STAGE_ASYNC(0, 0);                                                  \
    asm volatile("cp.async.commit_group;\n");                                 \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                \
        asm volatile("cp.async.wait_group 0;\n");                             \
        __syncthreads();                                                      \
        /* Staged bytes arrived on the generic proxy; make them visible */    \
        /* to the async proxy the tensor core reads through.            */    \
        asm volatile("fence.proxy.async.shared::cta;\n");                     \
        int rd = kt & (SM100_STAGES - 1);                                     \
        /* The next stage's copies are issued BEFORE this slab's MMAs   */    \
        /* so they overlap them. WAR-safe: buffer (kt+1)&1 was consumed */    \
        /* by the MMAs of iteration kt-1, whose commit barrier every    */    \
        /* thread has already waited on.                                */    \
        if (kt + 1 < num_k_tiles) {                                           \
            SM100_STAGE_ASYNC((kt + 1) & (SM100_STAGES - 1),                  \
                              (kt + 1) * SM100_BK);                           \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
        if (tid == 0) {                                                       \
            const unsigned char* stage =                                      \
                sm100_dynsmem + rd * SM100_STAGE_BYTES;                       \
            unsigned long long a_base = sm100_desc(stage, 1, 64);             \
            unsigned long long b_base =                                       \
                sm100_desc(stage + SM100_A_BYTES, 512, 64);                   \
            asm volatile("tcgen05.fence::after_thread_sync;");                \
            _Pragma("unroll")                                                 \
            for (int ks = 0; ks < SM100_BK / 16; ks++) {                      \
                unsigned long long da =                                       \
                    a_base + (unsigned long long)(ks * 2);                    \
                unsigned long long db =                                       \
                    b_base + (unsigned long long)(ks * 128);                  \
                unsigned acc_d =                                              \
                    (kt > 0 || ks > 0 || has_bias) ? 1u : 0u;                 \
                asm volatile(                                                 \
                    "{.reg .pred p;\n\t"                                      \
                    "setp.ne.b32 p, %4, 0;\n\t"                               \
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "   \
                    "{%5, %6, %7, %8}, p;}\n\t"                               \
                    :: "r"(taddr), "l"(da), "l"(db), "r"(IDESC),              \
                       "r"(acc_d), "r"(0u), "r"(0u), "r"(0u), "r"(0u));       \
            }                                                                 \
            asm volatile(                                                     \
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"           \
                ".shared::cluster.b64 [%0];" :: "r"(mbar));                   \
        }                                                                     \
        /* The commit arrives only when the tensor core no longer reads */    \
        /* the stage; every thread holds here before refilling it.      */    \
        asm volatile(                                                         \
            "{\n\t.reg .pred p;\n"                                            \
            "WAIT_%=:\n\t"                                                    \
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"       \
            "@!p bra WAIT_%=;\n\t}"                                           \
            :: "r"(mbar), "r"((unsigned)(kt & 1)));                           \
        __syncthreads();                                                      \
    }                                                                         \
    /* Epilogue: warp w reads TMEM lanes 32w..32w+31 (its own quarter), */    \
    /* eight columns per load; alpha through an explicit unfused        */    \
    /* multiply (target-independent bits), paired store on aligned even */    \
    /* destinations, scalar RNE tail otherwise - the mma.sync rules.    */    \
    asm volatile("tcgen05.fence::after_thread_sync;");                        \
    int row = pid_m * SM100_BM + warp * 32 + lane;                            \
    unsigned wrow = taddr + ((unsigned)(warp * 32) << 16);                    \
    for (int c0 = 0; c0 < SM100_BN; c0 += 8) {                                \
        float r[8];                                                           \
        asm volatile(                                                         \
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "                          \
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"                                \
            : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]),                 \
              "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7])                  \
            : "r"(wrow + (unsigned)c0));                                      \
        asm volatile("tcgen05.wait::ld.sync.aligned;");                       \
        /* The ld above is warp-collective; only the stores may diverge. */   \
        if (row < M) {                                                        \
            _Pragma("unroll")                                                 \
            for (int j = 0; j < 8; j += 2) {                                  \
                int col = pid_n * SM100_BN + c0 + j;                          \
                if (col >= N) continue;                                       \
                float v0 = __fmul_rn(alpha, r[j]);                            \
                float v1 = __fmul_rn(alpha, r[j + 1]);                        \
                T_ACT* dst = &C[(long long)row * ldc + col];                  \
                bool packed = beta == 0.0f && (ldc & 1) == 0 &&               \
                              col + 1 < N &&                                  \
                              ((reinterpret_cast<unsigned long long>(dst)     \
                                & 3u) == 0u);                                 \
                if (packed) {                                                 \
                    gbf_store_pair_rne(dst, v0, v1);                          \
                } else {                                                      \
                    for (int e = 0; e < 2; e++) {                             \
                        int gc = col + e;                                     \
                        if (gc >= N) continue;                                \
                        float val = e ? v1 : v0;                              \
                        if (beta != 0.0f)                                     \
                            val = __fmaf_rn(                                  \
                                beta,                                         \
                                to_f(C[(long long)row * ldc + gc]), val);     \
                        C[(long long)row * ldc + gc] = FROM_F(val);           \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    /* All TMEM traffic is finished; release the allocation the same    */    \
    /* warp made. Relinquish before dealloc is the mandated order.      */    \
    __syncthreads();                                                          \
    if (warp == 0) {                                                          \
        asm volatile(                                                         \
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");    \
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"  \
                     :: "r"(taddr), "r"(128u));                               \
    }                                                                         \
}

DEFINE_GEMM_BI_NN_SM100(bf16, __nv_bfloat16, from_f_bf16, SM100_IDESC_BF16)
DEFINE_GEMM_BI_NN_SM100(f16,  __half,        from_f_f16,  SM100_IDESC_F16)

#undef SM100_BM
#undef SM100_BN
#undef SM100_BK
#undef SM100_STAGES
#undef SM100_LDA
#undef SM100_LDB
#undef SM100_A_BYTES
#undef SM100_STAGE_BYTES
#undef SM100_IDESC_F16
#undef SM100_IDESC_BF16
#undef SM100_STAGE_ASYNC
#undef DEFINE_GEMM_BI_NN_SM100

#endif
#line 1 "kernels/gemm_bi_inference/sm80/half_pipeline.cu"
// OWN Ada-only Fixed homogeneous-half inference pipeline.
// Derived from the repository's measured pipe_vec experiment: same
// 128x128/BK64/S2 geometry, ascending k16 MMA chain and bias-seeded F32
// accumulators. The experiment's false scheduling branches are absent.
// Existing portable and CC12 CUDA fragments are deliberately unchanged.
//
// The Rust composer appends this fragment ONLY for ModuleKind::Fixed/sm_89.
// No other architecture is admitted by source composition or this fragment.
// The two exports have five arguments: C, A, B, F32 bias, and the by-value
// 32-byte/align-4 params below. Fixed callers use alpha=1, beta=0.
// block=256; grid=ceil(M/128)*ceil(N/128); dynamic shared=71,680 bytes.
// Loader: zero local memory, <=224 registers, >=1 resident CTA per SM.
//
// Misaligned A/B or lda/ldb not divisible8 use scalar zero-fill staging.
// Vector stores require C16, ldc divisible8, beta0 and a full128-column
// tile; tails/misaligned C use the incumbent scalar/pair RNE epilogue.
// K0 does not read A/B. No scratch, numeric atomics or split reduction.
// Requires the existing Fixed common.cuh typed/alignment/store helpers.

struct FixedSm89HalfParams {
    float alpha;
    float beta;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
};
static_assert(sizeof(FixedSm89HalfParams) == 32, "Fixed SM89 half parameter size");
static_assert(alignof(FixedSm89HalfParams) == 4, "Fixed SM89 half parameter alignment");
static_assert(__is_standard_layout(FixedSm89HalfParams), "Fixed SM89 half standard layout");
// Eight ordered four-byte fields in 32 bytes leave no internal or tail
// padding. Host offset_of assertions independently freeze the field layout.
static_assert(sizeof(float) == 4 && sizeof(int) == 4, "Fixed SM89 half field widths");

namespace fixed_sm89_half_pipeline {

static constexpr int kSharedBytes = 71680;
static constexpr int kOutputStride = 136;
static_assert(2 * (128 * 72 + 64 * 136) * 2 == kSharedBytes, "S2 shared ABI");
static_assert(128 * kOutputStride * 4 <= kSharedBytes, "epilogue aliases S2 storage");
static_assert(sizeof(uint4) == 16, "vector store width");

template <typename T> struct HalfOps;

#define SM89_FHP_OPS(TYPE, FROM, MMA_TYPE)                                    \
template <> struct HalfOps<TYPE> {                                         \
    static __device__ __forceinline__ TYPE from_float(float value) {        \
        return FROM(value);                                                \
    }                                                                      \
    static __device__ __forceinline__ void mma(                             \
        float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {      \
        asm volatile(                                                      \
            "mma.sync.aligned.m16n8k16.row.col.f32." MMA_TYPE "."            \
            MMA_TYPE ".f32 "                                               \
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"       \
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])               \
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),                  \
              "r"(b[0]), "r"(b[1]));                                       \
    }                                                                      \
};

SM89_FHP_OPS(__nv_bfloat16, from_f_bf16, "bf16")
SM89_FHP_OPS(__half, from_f_f16, "f16")
#undef SM89_FHP_OPS

template <typename T>
static __device__ __forceinline__ void stage_scalar(
    T* a_stages, T* b_stages, const T* A, const T* B,
    int stage, int k_base, int pid_m, int pid_n,
    int M, int N, int K, int lda, int ldb) {
    T* a_stage = a_stages + stage * 128 * 72;
    T* b_stage = b_stages + stage * 64 * 136;
    for (int linear = (int)threadIdx.x; linear < 128 * 64; linear += 256) {
        int row = linear / 64;
        int k = linear % 64;
        int global_row = pid_m * 128 + row;
        int global_k = k_base + k;
        a_stage[row * 72 + k] = global_row < M && global_k < K
            ? A[(long long)global_row * lda + global_k] : HalfOps<T>::from_float(0.0f);
    }
    for (int linear = (int)threadIdx.x; linear < 64 * 128; linear += 256) {
        int k = linear / 128;
        int column = linear % 128;
        int global_k = k_base + k;
        int global_column = pid_n * 128 + column;
        b_stage[k * 136 + column] = global_k < K && global_column < N
            ? B[(long long)global_k * ldb + global_column] : HalfOps<T>::from_float(0.0f);
    }
}

struct FragmentOffsets {
    unsigned a[4];
    unsigned b[4];
};

struct Fragments {
    unsigned a[4][4];
    unsigned b[4][2];
};

static __device__ __forceinline__ FragmentOffsets fragment_offsets(int warpM, int warpN) {
    FragmentOffsets offsets;
    int lane = (int)threadIdx.x & 31;
    int row_in_atom = (lane & 7) + ((lane & 8) ? 8 : 0);
    int a_column = (lane & 16) ? 8 : 0;
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        offsets.a[atom] = (unsigned)(((warpM + atom * 16 + row_in_atom) * 72 + a_column) * 2);
        offsets.b[atom] = (unsigned)((row_in_atom * 136 + warpN + atom * 8) * 2);
    }
    return offsets;
}

static __device__ __forceinline__ void load_fragments(
    unsigned a_stage, unsigned b_stage, int issue,
    const FragmentOffsets& offsets, Fragments& fragments) {
    unsigned a_base = a_stage + (unsigned)(issue * 16 * 2);
    unsigned b_base = b_stage + (unsigned)(issue * 16 * 136 * 2);
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        unsigned address = a_base + offsets.a[atom];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(fragments.a[atom][0]), "=r"(fragments.a[atom][1]),
              "=r"(fragments.a[atom][2]), "=r"(fragments.a[atom][3]) : "r"(address));
    }
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        unsigned address = b_base + offsets.b[atom];
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(fragments.b[atom][0]), "=r"(fragments.b[atom][1]) : "r"(address));
    }
}

template <typename T>
static __device__ __forceinline__ void consume_fragments(
    const Fragments& fragments, float (&acc)[4][4][4]) {
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            HalfOps<T>::mma(acc[fm][fn], fragments.a[fm], fragments.b[fn]);
        }
    }
}

struct CopyPlan {
    long long a_offset[4];
    long long b_offset[4];
    unsigned a_destination[4];
    unsigned b_destination[4];
    bool a_row_valid[4];
    int a_k;
    int b_k;
    int b_column_bytes;
};

static __device__ __forceinline__ CopyPlan copy_plan(
    unsigned a_shared, unsigned b_shared, int pid_m, int pid_n,
    int M, int N, int lda, int ldb) {
    CopyPlan plan;
    int thread = (int)threadIdx.x;
    plan.a_k = (thread & 7) * 8;
    plan.b_k = thread >> 4;
    int local_column = (thread & 15) * 8;
    int global_column = pid_n * 128 + local_column;
    int remaining = N - global_column;
    plan.b_column_bytes = remaining >= 8 ? 16 : (remaining > 0 ? remaining * 2 : 0);
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int local_row = (thread >> 3) + slice * 32;
        int global_row = pid_m * 128 + local_row;
        plan.a_row_valid[slice] = global_row < M;
        plan.a_offset[slice] = (long long)(global_row < M ? global_row : 0) * lda + plan.a_k;
        plan.b_offset[slice] = (long long)(plan.b_k + slice * 16) * ldb
            + (remaining > 0 ? global_column : 0);
        plan.a_destination[slice] = a_shared + (unsigned)((local_row * 72 + plan.a_k) * 2);
        plan.b_destination[slice] = b_shared + (unsigned)(((plan.b_k + slice * 16) * 136 + local_column) * 2);
    }
    return plan;
}

template <typename T>
static __device__ __forceinline__ void copy_slice(
    const CopyPlan& plan, const T* A, const T* B,
    int stage, int k_base, long long b_reduction_offset, int K, int slice) {
    int remaining = K - k_base - plan.a_k;
    int a_bytes = plan.a_row_valid[slice]
        ? (remaining >= 8 ? 16 : (remaining > 0 ? remaining * 2 : 0)) : 0;
    int b_bytes = k_base + plan.b_k + slice * 16 < K ? plan.b_column_bytes : 0;
    // Only form an in-object source when at least one element is read.
    const void* a_source = a_bytes > 0 ? (const void*)(A + plan.a_offset[slice] + k_base) : (const void*)A;
    const void* b_source = b_bytes > 0 ? (const void*)(B + plan.b_offset[slice] + b_reduction_offset) : (const void*)B;
    unsigned a_destination = plan.a_destination[slice] + (unsigned)(stage * 128 * 72 * 2);
    unsigned b_destination = plan.b_destination[slice] + (unsigned)(stage * 64 * 136 * 2);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(a_destination), "l"(a_source), "r"(a_bytes));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(b_destination), "l"(b_source), "r"(b_bytes));
}

template <typename T>
static __device__ __forceinline__ void scalar_epilogue(
    T* C, float (&acc)[4][4][4], float alpha, float beta,
    int M, int N, int ldc, int pid_m, int pid_n, int warpM, int warpN) {
    int lane = (int)threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            int r0 = pid_m * 128 + warpM + fm * 16 + g;
            int c0 = pid_n * 128 + warpN + fn * 8 + 2 * t;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = r0 + half * 8;
                if (row >= M) continue;
                float first = __fmul_rn(alpha, acc[fm][fn][2 * half]);
                float second = __fmul_rn(alpha, acc[fm][fn][2 * half + 1]);
                T* destination = c0 < N ? C + (long long)row * ldc + c0 : (T*)0;
                if (beta == 0.0f && (ldc & 1) == 0 && c0 + 1 < N && gbf_aligned4(destination)) {
                    gbf_store_pair_rne(destination, first, second);
                } else {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        int column = c0 + e;
                        if (column >= N) continue;
                        float value = e ? second : first;
                        T* output = C + (long long)row * ldc + column;
                        if (beta != 0.0f) value = __fmaf_rn(beta, to_f(*output), value);
                        *output = HalfOps<T>::from_float(value);
                    }
                }
            }
        }
    }
}

template <typename T>
static __device__ __forceinline__ void vector_epilogue(
    T* C, float* output_tile, float (&acc)[4][4][4], float alpha,
    int M, int ldc, int pid_m, int pid_n, int warpM, int warpN) {
    int lane = (int)threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    // Every warp must finish its last ldmatrix reads before aliasing As/Bs.
    __syncthreads();
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = warpM + fm * 16 + g + half * 8;
                int column = warpN + fn * 8 + 2 * t;
                *reinterpret_cast<float2*>(output_tile + row * kOutputStride + column) =
                    make_float2(acc[fm][fn][2 * half], acc[fm][fn][2 * half + 1]);
            }
        }
    }
    __syncthreads();
    // Each warp writes two adjacent rows in 16-byte chunks. The shared row
    // padding preserves 16-byte addresses for both float4 reads.
#pragma unroll
    for (int linear = (int)threadIdx.x; linear < 128 * 16; linear += 256) {
        int local_row = linear >> 4;
        int row = pid_m * 128 + local_row;
        if (row >= M) continue;
        int column = (linear & 15) * 8;
        const float* source = output_tile + local_row * kOutputStride + column;
        float4 first = *reinterpret_cast<const float4*>(source);
        float4 second = *reinterpret_cast<const float4*>(source + 4);
        uint4 packed;
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.x), __fmul_rn(alpha, first.x), __fmul_rn(alpha, first.y));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.y), __fmul_rn(alpha, first.z), __fmul_rn(alpha, first.w));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.z), __fmul_rn(alpha, second.x), __fmul_rn(alpha, second.y));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.w), __fmul_rn(alpha, second.z), __fmul_rn(alpha, second.w));
        *reinterpret_cast<uint4*>(C + (long long)row * ldc + pid_n * 128 + column) = packed;
    }
}

template <typename T>
static __device__ __forceinline__ void kernel(
    T* C, const T* A, const T* B, const float* bias,
    float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc) {
    static_assert(sizeof(T) == 2, "homogeneous half pipeline only");
    assert(alpha == 1.0f || bias == nullptr);
    extern __shared__ __align__(16) unsigned char fixed_sm89_hp_shared[];
    T (*As)[128][72] = reinterpret_cast<T (*)[128][72]>(fixed_sm89_hp_shared);
    T (*Bs)[64][136] = reinterpret_cast<T (*)[64][136]>(
        fixed_sm89_hp_shared + 2 * 128 * 72 * (int)sizeof(T));
    int num_pid_n = (N + 127) / 128;
    int pid_m = (int)blockIdx.x / num_pid_n;
    int pid_n = (int)blockIdx.x % num_pid_n;
    int warp = (int)threadIdx.x >> 5;
    int warpM = (warp >> 2) * 64;
    int warpN = (warp & 3) * 32;
    int t = (int)threadIdx.x & 3;
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(&As[0][0][0]);
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(&Bs[0][0][0]);
    bool fast_stage = (lda & 7) == 0 && (ldb & 7) == 0 && gbf_aligned16(A) && gbf_aligned16(B);
    float acc[4][4][4];
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            int column = pid_n * 128 + warpN + fn * 8 + 2 * t;
            float first = bias != nullptr && column < N ? bias[column] : 0.0f;
            float second = bias != nullptr && column + 1 < N ? bias[column + 1] : 0.0f;
            acc[fm][fn][0] = first;
            acc[fm][fn][1] = second;
            acc[fm][fn][2] = first;
            acc[fm][fn][3] = second;
        }
    }
    FragmentOffsets offsets = fragment_offsets(warpM, warpN);
    CopyPlan plan;
    if (fast_stage) plan = copy_plan(As_sbase, Bs_sbase, pid_m, pid_n, M, N, lda, ldb);
    int num_k_tiles = (K + 63) / 64;
    if (num_k_tiles > 0) {
        if (fast_stage) {
#pragma unroll
            for (int slice = 0; slice < 4; ++slice) copy_slice(plan, A, B, 0, 0, 0, K, slice);
            asm volatile("cp.async.commit_group;\n" ::);
        } else {
            stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, 0, 0, pid_m, pid_n, M, N, K, lda, ldb);
        }
    }
    int read_buf = 0;
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        if (fast_stage) asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        bool next = kt + 1 < num_k_tiles;
        int next_k = (kt + 1) * 64;
        if (next) {
            if (!fast_stage) {
                stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, read_buf ^ 1, next_k,
                    pid_m, pid_n, M, N, K, lda, ldb);
            }
        }
        unsigned a_read = As_sbase + (unsigned)(read_buf * 128 * 72 * 2);
        unsigned b_read = Bs_sbase + (unsigned)(read_buf * 64 * 136 * 2);
        {
            Fragments fragments[2];
            load_fragments(a_read, b_read, 0, offsets, fragments[0]);
            long long next_b = (long long)next_k * ldb;
#pragma unroll
            for (int issue = 0; issue < 4; ++issue) {
                if (fast_stage && next) copy_slice(plan, A, B, read_buf ^ 1, next_k, next_b, K, issue);
                if (issue < 3) load_fragments(a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
                consume_fragments<T>(fragments[issue & 1], acc);
            }
            if (fast_stage && next) asm volatile("cp.async.commit_group;\n" ::);
        }
        read_buf ^= 1;
    }
    {
        bool vector_output = beta == 0.0f && (ldc & 7) == 0 && gbf_aligned16(C)
            && pid_n * 128 + 128 <= N;
        if (vector_output) {
            vector_epilogue(C, reinterpret_cast<float*>(fixed_sm89_hp_shared), acc, alpha,
                M, ldc, pid_m, pid_n, warpM, warpN);
            return;
        }
    }
    scalar_epilogue(C, acc, alpha, beta, M, N, ldc, pid_m, pid_n, warpM, warpN);
}

} // namespace fixed_sm89_half_pipeline

extern "C" __global__ __launch_bounds__(256, 1)
void nn_sm89_tc128_pipeline_bf16(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B, const float* __restrict__ bias,
    FixedSm89HalfParams params) {
    fixed_sm89_half_pipeline::kernel<__nv_bfloat16>(C, A, B, bias,
        params.alpha, params.beta, params.m, params.n, params.k,
        params.lda, params.ldb, params.ldc);
}

extern "C" __global__ __launch_bounds__(256, 1)
void nn_sm89_tc128_pipeline_f16(
    __half* __restrict__ C, const __half* __restrict__ A,
    const __half* __restrict__ B, const float* __restrict__ bias,
    FixedSm89HalfParams params) {
    fixed_sm89_half_pipeline::kernel<__half>(C, A, B, bias,
        params.alpha, params.beta, params.m, params.n, params.k,
        params.lda, params.ldb, params.ldc);
}
#line 1 "kernels/gemm_bi_inference/sm80/f32_n64_copyplan.cu"
// Ada-only exact F32 N64 copy-plan extension. The immutable standalone
// 713a575 candidate supplies the body; only names and compact ABI unpacking differ.
struct FixedSm89ExactF32Params {
    float alpha, beta;
    int m, n, k, lda, ldb, ldc;
};
static_assert(sizeof(FixedSm89ExactF32Params) == 32, "exact N64 parameter size");
static_assert(alignof(FixedSm89ExactF32Params) == 4, "exact N64 parameter alignment");
// mma16.cu has already undefined the incumbent tile macros at this boundary.
// Private constants preserve the incumbent geometry without leaking ambient macros.
#define SM89_EXACT_N64_CP_BM 64
#define SM89_EXACT_N64_CP_BN 64
#define SM89_EXACT_N64_CP_BK 32
#define SM89_EXACT_N64_CP_GROUP_M 8
// Production N64/BK32/S2 geometry, accumulator ownership, full ascending-K32
// FMA chain and epilogue are retained. The only specialization is producer
// planning for aligned full-K slabs. No tensor operations, split reduction,
// scratch or atomics. Actual registers/spills/residency require parent testing.
// Includes expect the unchanged production Fixed common.cuh helpers/constants.
static_assert(SM89_EXACT_N64_CP_BM == 64 && SM89_EXACT_N64_CP_BN == 64 && SM89_EXACT_N64_CP_BK == 32,
              "N64 copy-plan experiment requires the incumbent geometry");
#define SM89_EXACT_N64_CP_THREADS 128
#define SM89_EXACT_N64_CP_GENERIC_ASYNC(BUF, K_TILE)                                    \
    do {                                                                       \
        unsigned _as = (unsigned)__cvta_generic_to_shared(&smem_a[(BUF)][0]);  \
        unsigned _bs = (unsigned)__cvta_generic_to_shared(&smem_b[(BUF)][0]);  \
        for (int _i = threadIdx.x; _i < SM89_EXACT_N64_CP_BM * (SM89_EXACT_N64_CP_BK / 4);               \
             _i += SM89_EXACT_N64_CP_THREADS) {                                          \
            int _r = _i / (SM89_EXACT_N64_CP_BK / 4);                                      \
            int _c = (_i % (SM89_EXACT_N64_CP_BK / 4)) * 4;                                \
            int _gr = row0 + _r;                                               \
            int _gc = (K_TILE) + _c;                                           \
            int _valid = (_gr < m) ? (k - _gc) : 0;                           \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);    \
            unsigned _dst = _as + (unsigned)((_r * SM89_EXACT_N64_CP_BK + _c) * 4);        \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&a[(long long)_gr * lda + _gc]                  \
                : (const void*)a;                                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));              \
        }                                                                      \
        for (int _i = threadIdx.x; _i < SM89_EXACT_N64_CP_BK * (SM89_EXACT_N64_CP_BN / 4);               \
             _i += SM89_EXACT_N64_CP_THREADS) {                                          \
            int _r = _i / (SM89_EXACT_N64_CP_BN / 4);                                      \
            int _c = (_i % (SM89_EXACT_N64_CP_BN / 4)) * 4;                                \
            int _gr = (K_TILE) + _r;                                           \
            int _gc = col0 + _c;                                               \
            int _valid = (_gr < k) ? (n - _gc) : 0;                           \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);    \
            unsigned _dst = _bs + (unsigned)((_r * SM89_EXACT_N64_CP_BN + _c) * 4);        \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&b[(long long)_gr * ldb + _gc]                  \
                : (const void*)b;                                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));              \
        }                                                                      \
        asm volatile("cp.async.commit_group;\n");                             \
    } while (0)

#define SM89_EXACT_N64_CP_GENERIC_SCALAR(BUF, K_TILE)                                   \
    do {                                                                       \
        for (int _i = threadIdx.x; _i < SM89_EXACT_N64_CP_BM * SM89_EXACT_N64_CP_BK;                    \
             _i += SM89_EXACT_N64_CP_THREADS) {                                          \
            int _r = _i / SM89_EXACT_N64_CP_BK;                                             \
            int _c = _i % SM89_EXACT_N64_CP_BK;                                             \
            int _gr = row0 + _r;                                               \
            int _gc = (K_TILE) + _c;                                           \
            smem_a[(BUF)][_i] = (_gr < m && _gc < k)                          \
                ? a[(long long)_gr * lda + _gc]                                \
                : 0.0f;                                                        \
        }                                                                      \
        for (int _i = threadIdx.x; _i < SM89_EXACT_N64_CP_BK * SM89_EXACT_N64_CP_BN;                    \
             _i += SM89_EXACT_N64_CP_THREADS) {                                          \
            int _r = _i / SM89_EXACT_N64_CP_BN;                                             \
            int _c = _i % SM89_EXACT_N64_CP_BN;                                             \
            int _gr = (K_TILE) + _r;                                           \
            int _gc = col0 + _c;                                               \
            smem_b[(BUF)][_i] = (_gr < k && _gc < n)                           \
                ? b[(long long)_gr * ldb + _gc]                                \
                : 0.0f;                                                        \
        }                                                                      \
    } while (0)

// All plan variables are initialized once inside the full-K branch below.
// Shared/vector offsets and source lengths do not change with the K slab.
// Only two advancing 64-bit source bases are retained; each copy derives its
// own address from a common base plus a vector stride, then discards it.
// Invalid rows/columns select the original allocation base for a zero-byte
// copy, so no out-of-range pointer is submitted even to a zero-fill operation.
#define SM89_EXACT_N64_CP_PLANNED_STAGE(BUF)                                             \
    do {                                                                    \
        unsigned _stage = (unsigned)(BUF) * (SM89_EXACT_N64_CP_BM * SM89_EXACT_N64_CP_BK * 4);           \
        _Pragma("unroll")                                                    \
        for (int _i = 0; _i < 4; ++_i) {                                    \
            unsigned _dst = a_destination + _stage + (unsigned)(_i * 2048);  \
            unsigned long long _address = a_next +                           \
                (unsigned long long)_i * a_vector_stride;                   \
            unsigned long long _source = a_bytes[_i] > 0 ? _address          \
                : reinterpret_cast<unsigned long long>(a);                  \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"  \
                         :: "r"(_dst), "l"(_source), "r"(a_bytes[_i]));       \
        }                                                                   \
        _Pragma("unroll")                                                    \
        for (int _i = 0; _i < 4; ++_i) {                                    \
            unsigned _dst = b_destination + _stage + (unsigned)(_i * 2048);  \
            unsigned long long _address = b_next +                           \
                (unsigned long long)_i * b_vector_stride;                   \
            unsigned long long _source = b_bytes > 0 ? _address             \
                : reinterpret_cast<unsigned long long>(b);                  \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"  \
                         :: "r"(_dst), "l"(_source), "r"(b_bytes));           \
        }                                                                   \
        asm volatile("cp.async.commit_group;\n");                             \
    } while (0)

extern "C" __global__ __launch_bounds__(SM89_EXACT_N64_CP_THREADS, 2) void
nn_sm89_f32_n64_copyplan(
    float* __restrict__ c,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    FixedSm89ExactF32Params params
) {
    const float alpha = params.alpha, beta = params.beta;
    const int m = params.m, n = params.n, k = params.k;
    const int lda = params.lda, ldb = params.ldb, ldc = params.ldc;
    __align__(16) __shared__ float smem_a[2][SM89_EXACT_N64_CP_BM * SM89_EXACT_N64_CP_BK];
    __align__(16) __shared__ float smem_b[2][SM89_EXACT_N64_CP_BK * SM89_EXACT_N64_CP_BN];

    int num_pid_m = (m + SM89_EXACT_N64_CP_BM - 1) / SM89_EXACT_N64_CP_BM;
    int num_pid_n = (n + SM89_EXACT_N64_CP_BN - 1) / SM89_EXACT_N64_CP_BN;
    int num_pid_in_group = SM89_EXACT_N64_CP_GROUP_M * num_pid_n;
    int group_id = blockIdx.x / num_pid_in_group;
    int first_pid_m = group_id * SM89_EXACT_N64_CP_GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, SM89_EXACT_N64_CP_GROUP_M);
    int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m);
    int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;
    int row0 = pid_m * SM89_EXACT_N64_CP_BM;
    int col0 = pid_n * SM89_EXACT_N64_CP_BN;
    int tx = threadIdx.x & 15;
    int ty = threadIdx.x >> 4;
    int row_base = row0 + ty * 8;
    int col_base = col0 + tx * 4;

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    }

    int num_k_tiles = (k + SM89_EXACT_N64_CP_BK - 1) / SM89_EXACT_N64_CP_BK;
    bool fast_stage = gbf_aligned16(a) && gbf_aligned16(b)
                   && (lda & 3) == 0 && (ldb & 3) == 0;
    // K=0 must never construct operand plans: null A/B are legal then.
    // Keep partial-K and unaligned inputs on the exact incumbent staging path.
    if (num_k_tiles > 0 && fast_stage && (k & (SM89_EXACT_N64_CP_BK - 1)) == 0) {
        const int tid = (int)threadIdx.x;
        const int copy_a_row = tid / (SM89_EXACT_N64_CP_BK / 4);
        const int copy_a_col = (tid % (SM89_EXACT_N64_CP_BK / 4)) * 4;
        const int copy_b_row = tid / (SM89_EXACT_N64_CP_BN / 4);
        const int copy_b_col = col0 + (tid % (SM89_EXACT_N64_CP_BN / 4)) * 4;
        int a_bytes[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            a_bytes[i] = row0 + copy_a_row + i * 16 < m ? 16 : 0;
        const int remaining_b = n - copy_b_col;
        const int b_bytes = remaining_b >= 4 ? 16 : (remaining_b > 0 ? remaining_b * 4 : 0);
        const unsigned a_destination =
            (unsigned)__cvta_generic_to_shared(&smem_a[0][0]) + (unsigned)(tid * 16);
        const unsigned b_destination =
            (unsigned)__cvta_generic_to_shared(&smem_b[0][0]) + (unsigned)(tid * 16);
        const unsigned long long a_vector_stride = (unsigned long long)lda * 64ULL;
        const unsigned long long b_vector_stride = (unsigned long long)ldb * 32ULL;
        const unsigned long long b_slab_stride = b_vector_stride * 4ULL;
        // Integer addresses avoid forming C++ pointers outside an allocation
        // for masked M/N lanes. SM89_EXACT_N64_CP_PLANNED_STAGE selects a/b for those lanes.
        unsigned long long a_next = reinterpret_cast<unsigned long long>(a)
            + ((unsigned long long)(row0 + copy_a_row) * (unsigned long long)lda
               + (unsigned long long)copy_a_col) * 4ULL;
        unsigned long long b_next = reinterpret_cast<unsigned long long>(b)
            + ((unsigned long long)copy_b_row * (unsigned long long)ldb
               + (unsigned long long)copy_b_col) * 4ULL;

        SM89_EXACT_N64_CP_PLANNED_STAGE(0);
        a_next += SM89_EXACT_N64_CP_BK * 4ULL;
        b_next += b_slab_stride;
        int read_buf = 0;
        for (int kt = 0; kt < num_k_tiles; ++kt) {
            asm volatile("cp.async.wait_group 0;\n");
            // Both cross-warp copy visibility and previous-stage consumption
            // must finish before a stage is read or reused.
            __syncthreads();
            if (kt + 1 < num_k_tiles) {
                SM89_EXACT_N64_CP_PLANNED_STAGE(read_buf ^ 1);
                a_next += SM89_EXACT_N64_CP_BK * 4ULL;
                b_next += b_slab_stride;
            }

    #pragma unroll
            for (int kk = 0; kk < SM89_EXACT_N64_CP_BK; kk++) {
                float a_reg[8];
                float b_reg[4];
    #pragma unroll
                for (int i = 0; i < 8; i++)
                    a_reg[i] = smem_a[read_buf][(ty * 8 + i) * SM89_EXACT_N64_CP_BK + kk];
    #pragma unroll
                for (int j = 0; j < 4; j++)
                    b_reg[j] = smem_b[read_buf][kk * SM89_EXACT_N64_CP_BN + tx * 4 + j];
    #pragma unroll
                for (int i = 0; i < 8; i++) {
    #pragma unroll
                    for (int j = 0; j < 4; j++)
                        acc[i][j] = __fmaf_rn(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
            read_buf ^= 1;
        }
    } else {
        if (num_k_tiles > 0) {
            if (fast_stage) {
                SM89_EXACT_N64_CP_GENERIC_ASYNC(0, 0);
            } else {
                SM89_EXACT_N64_CP_GENERIC_SCALAR(0, 0);
            }
        }
        int read_buf = 0;
        for (int kt = 0; kt < num_k_tiles; kt++) {
            if (fast_stage) asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            int next_k = (kt + 1) * SM89_EXACT_N64_CP_BK;
            if (kt + 1 < num_k_tiles) {
                if (fast_stage) {
                    SM89_EXACT_N64_CP_GENERIC_ASYNC(read_buf ^ 1, next_k);
                } else {
                    SM89_EXACT_N64_CP_GENERIC_SCALAR(read_buf ^ 1, next_k);
                }
            }

    #pragma unroll
            for (int kk = 0; kk < SM89_EXACT_N64_CP_BK; kk++) {
                float a_reg[8];
                float b_reg[4];
    #pragma unroll
                for (int i = 0; i < 8; i++)
                    a_reg[i] = smem_a[read_buf][(ty * 8 + i) * SM89_EXACT_N64_CP_BK + kk];
    #pragma unroll
                for (int j = 0; j < 4; j++)
                    b_reg[j] = smem_b[read_buf][kk * SM89_EXACT_N64_CP_BN + tx * 4 + j];
    #pragma unroll
                for (int i = 0; i < 8; i++) {
    #pragma unroll
                    for (int j = 0; j < 4; j++)
                        acc[i][j] = __fmaf_rn(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
            read_buf ^= 1;
        }
    }

    bool pair_store_fast = row0 <= m - SM89_EXACT_N64_CP_BM
        && col0 <= n - SM89_EXACT_N64_CP_BN
        && (reinterpret_cast<unsigned long long>(c) & 7ULL) == 0
        && (ldc & 1) == 0;
    if (pair_store_fast) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int r = row_base + i;
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                int col = col_base + j;
                float val0 = __fmul_rn(alpha, acc[i][j]);
                float val1 = __fmul_rn(alpha, acc[i][j + 1]);
                if (bias != nullptr) {
                    val0 = __fadd_rn(val0, bias[col]);
                    val1 = __fadd_rn(val1, bias[col + 1]);
                }
                if (beta != 0.0f) {
                    val0 = __fmaf_rn(beta, c[(long long)r * ldc + col], val0);
                    val1 = __fmaf_rn(beta, c[(long long)r * ldc + col + 1], val1);
                }
                float2 pair = {val0, val1};
                *reinterpret_cast<float2*>(c + (long long)r * ldc + col) = pair;
            }
        }
        return;
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = row_base + i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col_base + j;
            if (col >= n) continue;
            float val = __fmul_rn(alpha, acc[i][j]);
            if (bias != nullptr) val = __fadd_rn(val, bias[col]);
            if (beta != 0.0f)
                val = __fmaf_rn(beta, c[(long long)r * ldc + col], val);
            c[(long long)r * ldc + col] = val;
        }
    }
}

#undef SM89_EXACT_N64_CP_PLANNED_STAGE
#undef SM89_EXACT_N64_CP_GENERIC_SCALAR
#undef SM89_EXACT_N64_CP_GENERIC_ASYNC
#undef SM89_EXACT_N64_CP_THREADS

#undef SM89_EXACT_N64_CP_BM
#undef SM89_EXACT_N64_CP_BN
#undef SM89_EXACT_N64_CP_BK
#undef SM89_EXACT_N64_CP_GROUP_M
#line 1 "kernels/gemm_bi_inference/sm80/tf32_rna_wide.cu"
// Fixed-owned NN wide TF32 candidate. This is the production adaptation of
// internal/experiments/sm89-nn-wide-rna-compatible.cu at SHA256
// c0c9eb735374620eaf8a023345ee86359af9f748df53fc5cf7d07ee7051f65c4.
// The 128x128/BK32/S3 pipeline, ascending k8 MMA sequence, explicit
// cvt.rna.tf32.f32 conversion, bias initialization, tail handling, and
// scalar/vector epilogues are unchanged. Only Fixed-private dependency names,
// the production export, and the self-contained parameter/store helpers differ.
// Compose this fragment after Fixed common.cuh and tf32.cu, on sm_89 only.
//
// Export: nn_rna_wide_tf32_m128n128_bk32_s3
// ABI: (float* output, const float* a, const float* b, const float* bias,
//       GbfTf32WideParams params).
// Params: {float alpha,beta; int m,k,n,lda,ldb,ldc;} (32 bytes, alignment 4).
// Grid=(ceil(m/128)*ceil(n/128),1,1); block=(256,1,1).
// Dynamic shared=98,304 bytes, opt in the function. No scratch/counters.
// Same vector-input admission as the existing wide body: A/B base alignment
// 16 bytes and lda/ldb divisible by four, valid row-major extents/strides,
// dimensions/tile counts within device and signed-int limits. Host skips
// empty output. K=0 uses the unchanged zero-reduction epilogue. Fixed callers
// must pass this 32-byte struct with alpha=1,beta=0, not GbfTf32Params.
//
// Byte-level conversion rationale:
// Fixed's gbf_tf32_rna(float) applies exactly cvt.rna.tf32.f32 to the input
// register and returns its .b32 result. The new helper bit-reinterprets the
// incoming unsigned fragment as float, then uses the same PTX and constraints.
// This preserves all input bits until that instruction, including signed zero,
// subnormals, infinities, signaling/quiet NaNs and every NaN payload. It also
// preserves whatever output bits the conversion instruction produces, without
// an integer mask or a hand-written special-value transformation.
//
// For a hypothetical upper19-bit TF32 layout, finite nearest-away rounding
// equals ((bits+0x1000) & 0xffffe000). That algebra does not prove the PTX
// instruction's complete NaN payload behavior. In the unguarded old add,
// 0x7f801000 -> 0x7f802000; 0x7fffffff -> 0x80000fff; and
// 0xffffffff -> 0x00000fff. A finite-only add would preserve the special input
// bits, but the available evidence does not certify those bits as cvt.rna's
// result for every NaN. Therefore this twin deliberately uses explicit cvt.
//
// NVIDIA PTX ISA 9.3 defines RNA as nearest with ties away from zero and
// describes TF32's internal layout as implementation-defined:
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// This source-level match is not GPU proof of a cross-route batch ladder.
// Main must still compare raw conversion results, exceptional-value GEMMs,
// repeated/batch-prefix bits and performance before any admission.

struct GbfTf32WideParams {
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(GbfTf32WideParams) == 32, "Fixed RNA-wide parameter ABI drift");
static_assert(alignof(GbfTf32WideParams) == 4, "Fixed RNA-wide parameter alignment drift");
static_assert(__is_standard_layout(GbfTf32WideParams),
              "Fixed RNA-wide parameters must remain standard layout");

__device__ __forceinline__ int tf32wrc_a_index(int row, int k) {
    int chunk = (k >> 2) ^ (row & 7);
    return row * 32 + chunk * 4 + (k & 3);
}

__device__ __forceinline__ int tf32wrc_b_index(int k, int column) {
    int chunk = (column >> 2) ^ ((k & 3) << 1);
    return k * 128 + chunk * 4 + (column & 3);
}

// One quarter of a stage's copies: the A and B rows this thread owns in
// slice `slice` (two 16-byte cp.async per thread), so the eight copies of a
// stage interleave with the four k8 steps of the stage being computed.
// Each thread owns eight 16-byte copies per stage (four of A, four of B):
// their global pointers, shared destinations and the row / column bounds
// depend on the tile and the thread only, so they are computed once; a
// stage advances the pointers by its k slab and clamps the copy length by
// the remaining reduction (zero bytes zero-fill the slot; a zero-length
// copy keeps a valid address).
struct Tf32wrcCopyPlan {
    const float* a_source[4];
    const float* b_source[4];
    unsigned a_destination[4];
    unsigned b_destination[4];
    int a_k_offset;
    int b_k_row;
    bool a_row_valid[4];
    int b_column_bytes[4];
};

__device__ __forceinline__ void tf32wrc_copy_plan(
    float* a_stage0, float* b_stage0, const float* a, const float* b,
    const GbfTf32WideParams& params, int tile_row, int tile_column, Tf32wrcCopyPlan& plan) {
    plan.a_k_offset = ((int)threadIdx.x & 7) * 4;
    plan.b_k_row = (int)threadIdx.x >> 5;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int row = linear >> 3;
        int global_row = tile_row + row;
        plan.a_row_valid[slice] = global_row < params.m;
        plan.a_source[slice] = a + (long long)(plan.a_row_valid[slice] ? global_row : 0) * params.lda + plan.a_k_offset;
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(a_stage0 + tf32wrc_a_index(row, plan.a_k_offset));
        int k_row = linear >> 5;
        int column = (linear & 31) * 4;
        int global_column = tile_column + column;
        int columns = params.n - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        plan.b_column_bytes[slice] = columns * 4;
        plan.b_source[slice] = b + (long long)k_row * params.ldb + (columns > 0 ? global_column : 0);
        plan.b_destination[slice] = (unsigned)__cvta_generic_to_shared(b_stage0 + tf32wrc_b_index(k_row, column));
    }
}

// One quarter of a stage's copies. `stage_bytes` is the stage's byte offset
// in the ring, `k_base` the stage's first k; the pointers already point at
// the stage (the caller advances them).
__device__ __forceinline__ void tf32wrc_stage_slice(
    const Tf32wrcCopyPlan& plan, unsigned stage_bytes, int k_base, int reduction, int slice) {
    {
        int remaining = reduction - k_base - plan.a_k_offset;
        remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
        int bytes = plan.a_row_valid[slice] ? remaining * 4 : 0;
        gbf_tf32_copy_cg(plan.a_destination[slice] + stage_bytes, plan.a_source[slice], bytes);
    }
    {
        int bytes = k_base + plan.b_k_row + slice * 8 < reduction ? plan.b_column_bytes[slice] : 0;
        gbf_tf32_copy_cg(plan.b_destination[slice] + stage_bytes, plan.b_source[slice], bytes);
    }
}

__device__ __forceinline__ void tf32wrc_advance_plan(Tf32wrcCopyPlan& plan, long long b_rows) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        plan.a_source[slice] += 32;
        plan.b_source[slice] += b_rows;
    }
}

__device__ __forceinline__ void tf32wrc_stage_async(
    const Tf32wrcCopyPlan& plan, unsigned stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        tf32wrc_stage_slice(plan, stage_bytes, k_base, reduction, slice);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

struct Tf32wrcFragments {
    unsigned a[4][4];
    unsigned b[4][2];
};

// The shared-memory element offsets of this lane's fragments, computed once
// per thread: the swizzle of A depends on the k8 step through its chunk
// index, so the ldmatrix offsets are kept per (atom, step); the swizzle of B
// depends only on the lane's k row within the step, so one offset per
// (atom, half) serves every step with the step's row term added.
struct Tf32wrcFragmentOffsets {
    int a[4][4];
    int b[4][2];
};

__device__ __forceinline__ void tf32wrc_fragment_offsets(
    int warp_m, int warp_n, int group, int thread, int lane, Tf32wrcFragmentOffsets& offsets) {
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            offsets.a[m_atom][step] = tf32wrc_a_index(a_row + m_atom * 16, step * 8 + a_k);
        }
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        offsets.b[n_atom][0] = tf32wrc_b_index(thread, column);
        offsets.b[n_atom][1] = tf32wrc_b_index(thread + 4, column);
    }
}

// Keep the operand conversion instruction identical to gbf_tf32_rna in
// kernels/gemm_bi_inference/tf32.cu. __uint_as_float is a bit reinterpretation;
// it performs no FP arithmetic that could quiet or canonicalize an sNaN
// before the PTX instruction sees the original 32-bit payload.
__device__ __forceinline__ unsigned tf32wrc_round(unsigned bits) {
    float value = __uint_as_float(bits);
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

// The A fragment of one 16 x 8 atom through ldmatrix.x4: lanes 0-7 address
// rows 0-7 at k8, 8-15 rows 8-15 at k8, 16-23 rows 0-7 at k8 + 4 and 24-31
// rows 8-15 at k8 + 4, so the four registers land as a0..a3 of the mma
// (row g / row g + 8 at k t, then the same at k t + 4). Every row address is
// a 16-byte chunk of the swizzled stage.
__device__ __forceinline__ void tf32wrc_load_fragments(
    const float* a_stage, const float* b_stage, int step,
    const Tf32wrcFragmentOffsets& offsets, Tf32wrcFragments& fragments) {
    unsigned a_base = (unsigned)__cvta_generic_to_shared(a_stage);
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
        unsigned address = a_base + (unsigned)offsets.a[m_atom][step] * 4U;
        unsigned raw0, raw1, raw2, raw3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
                     : "r"(address));
        fragments.a[m_atom][0] = tf32wrc_round(raw0);
        fragments.a[m_atom][1] = tf32wrc_round(raw1);
        fragments.a[m_atom][2] = tf32wrc_round(raw2);
        fragments.a[m_atom][3] = tf32wrc_round(raw3);
    }
    const float* b_step = b_stage + step * 8 * 128;
#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        fragments.b[n_atom][0] = tf32wrc_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] = tf32wrc_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}

__device__ __forceinline__ void tf32wrc_mma(const Tf32wrcFragments& fragments, float (&acc)[4][4][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
            gbf_tf32_mma_m16n8k8(acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

__device__ __forceinline__ void tf32wrc_store(
    float* output, int row, int column, float accumulator,
    const GbfTf32WideParams& params) {
    if (row >= params.m || column >= params.n) return;
    float value = params.alpha == 1.0f
        ? accumulator
        : __fmul_rn(params.alpha, accumulator);
    float* destination = output + (long long)row * params.ldc + column;
    if (params.beta != 0.0f) {
        value = __fmaf_rn(params.beta, *destination, value);
    }
    *destination = value;
}

template <int BM, int BN>
__device__ __forceinline__ void tf32wrc_zero_reduction(
    float* output, const float* bias, const GbfTf32WideParams& params) {
    int column_tiles = (params.n + BN - 1) / BN;
    int tile_row = (int)blockIdx.x / column_tiles * BM;
    int tile_column = (int)blockIdx.x % column_tiles * BN;
    for (int linear = (int)threadIdx.x; linear < BM * BN; linear += (int)blockDim.x) {
        int row = tile_row + linear / BN;
        int column = tile_column + linear % BN;
        if (row < params.m && column < params.n) {
            float accumulator = bias == nullptr ? 0.0f : bias[column];
            tf32wrc_store(output, row, column, accumulator, params);
        }
    }
}

template <int Stages>
__device__ __forceinline__ void tf32wrc_nn_kernel(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32WideParams params) {
    extern __shared__ __align__(16) unsigned char tf32wrc_shared[];
    float* a_stages = reinterpret_cast<float*>(tf32wrc_shared);
    float* b_stages = a_stages + Stages * 128 * 32;
    int column_tiles = (params.n + 127) / 128;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * 128;
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 2) * 64;
    int warp_n = (warp & 3) * 32;
    int group = lane >> 2;
    int thread = lane & 3;
    float acc[4][4][4];
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8 + 2 * thread + (element & 1);
                acc[m_atom][n_atom][element] =
                    row < params.m && column < params.n && bias != nullptr ? bias[column] : 0.0f;
            }
        }
    }
    Tf32wrcFragmentOffsets offsets;
    tf32wrc_fragment_offsets(warp_m, warp_n, group, thread, lane, offsets);
    Tf32wrcCopyPlan plan;
    tf32wrc_copy_plan(a_stages, b_stages, a, b, params, tile_row, tile_column, plan);
    long long b_slab_rows = 32LL * params.ldb;
    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
#pragma unroll
    for (unsigned tile = 0; tile < Stages - 1; ++tile) {
        if (tile < tile_count) {
            tf32wrc_stage_async(plan, tile * 128 * 32 * 4U, (int)(tile * 32U), params.k);
            tf32wrc_advance_plan(plan, b_slab_rows);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(Stages - 2));
        // The barrier also retires every warp's reads of the stage the copies
        // below overwrite (the one computed in the previous iteration).
        __syncthreads();
        unsigned next = tile + Stages - 1;
        bool has_next = next < tile_count;
        int write_stage = read_stage == 0 ? Stages - 1 : read_stage - 1;
        unsigned write_bytes = (unsigned)write_stage * 128U * 32U * 4U;
        const float* a_read = a_stages + read_stage * 128 * 32;
        const float* b_read = b_stages + read_stage * 32 * 128;
        Tf32wrcFragments fragments[2];
        tf32wrc_load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
                tf32wrc_stage_slice(plan, write_bytes, (int)(next * 32U), params.k, issue);
            }
            if (issue < 3) {
                tf32wrc_load_fragments(a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
            }
            tf32wrc_mma(fragments[issue & 1], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (has_next) tf32wrc_advance_plan(plan, b_slab_rows);
        if (++read_stage == Stages) read_stage = 0;
    }
    // Epilogue: the tile goes through shared memory (row stride 136 floats:
    // the fragment-shaped float2 writes land in two wavefronts) and out as
    // 16-byte row segments, one warp per 512-byte row, when the output is
    // 16-byte addressable and the tile's columns lie inside n; each element
    // sees the same alpha and beta arithmetic as the scalar store.
    __syncthreads();
    float* tile = reinterpret_cast<float*>(tf32wrc_shared);
    bool vector_rows = tile_column + 128 <= params.n && (params.ldc & 3) == 0
        && gbf_aligned16(output);
    if (vector_rows) {
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = warp_m + m_atom * 16 + group + half * 8;
                    int column = warp_n + n_atom * 8 + 2 * thread;
                    *reinterpret_cast<float2*>(tile + row * 136 + column) = make_float2(
                        acc[m_atom][n_atom][2 * half], acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        __syncthreads();
        bool scale = params.alpha != 1.0f;
        bool blend = params.beta != 0.0f;
#pragma unroll 4
        for (int linear = (int)threadIdx.x; linear < 128 * 32; linear += 256) {
            int row = linear >> 5;
            int chunk = (linear & 31) * 4;
            int global_row = tile_row + row;
            if (global_row >= params.m) continue;
            float4 value = *reinterpret_cast<const float4*>(tile + row * 136 + chunk);
            float* destination = output + (long long)global_row * params.ldc + tile_column + chunk;
            if (scale) {
                value.x = __fmul_rn(params.alpha, value.x);
                value.y = __fmul_rn(params.alpha, value.y);
                value.z = __fmul_rn(params.alpha, value.z);
                value.w = __fmul_rn(params.alpha, value.w);
            }
            if (blend) {
                float4 old = *reinterpret_cast<const float4*>(destination);
                value.x = __fmaf_rn(params.beta, old.x, value.x);
                value.y = __fmaf_rn(params.beta, old.y, value.y);
                value.z = __fmaf_rn(params.beta, old.z, value.z);
                value.w = __fmaf_rn(params.beta, old.w, value.w);
            }
            *reinterpret_cast<float4*>(destination) = value;
        }
        return;
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8 + 2 * thread + (element & 1);
                tf32wrc_store(output, row, column, acc[m_atom][n_atom][element], params);
            }
        }
    }
}


extern "C" __global__ __launch_bounds__(256, 1)
void nn_rna_wide_tf32_m128n128_bk32_s3(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32WideParams params) {
    if (params.k == 0) {
        tf32wrc_zero_reduction<128, 128>(output, bias, params);
        return;
    }
    tf32wrc_nn_kernel<3>(output, a, b, bias, params);
}

template <typename A, typename B> struct GbfRnaSameType { static constexpr bool value = false; };
template <typename A> struct GbfRnaSameType<A, A> { static constexpr bool value = true; };
using GbfRnaWideSignature = void (*)(
    float*, const float*, const float*, const float*, GbfTf32WideParams);
static_assert(GbfRnaSameType<
    decltype(&nn_rna_wide_tf32_m128n128_bk32_s3),
    GbfRnaWideSignature>::value, "Fixed NN RNA-wide ABI");
#line 1 "kernels/gemm_bi_inference/sm80/half_swizzle_layout.cuh"
// Production Fixed SM89 homogeneous-half swizzle layout. Shared by the CUDA twin and
// pure-host address/ldmatrix tests; no CUDA toolkit is needed by the latter.
#pragma once
#if defined(__CUDACC__)
#define SM89_FHS_HD __host__ __device__
#else
#define SM89_FHS_HD
#endif
namespace sm89_fixed_half_swizzle_layout {
constexpr int kStageElements = 8192;
constexpr int kOutputStride = 136;
constexpr int kSharedBytes = 69632;
static_assert(4 * kStageElements * 2 <= kSharedBytes, "S2 staging fits");
static_assert(128 * kOutputStride * 4 == kSharedBytes, "unchanged output scratch fits");
SM89_FHS_HD constexpr int a_index(int row, int k) {
    return row * 64 + (k ^ ((row & 7) * 8));
}
SM89_FHS_HD constexpr int b_index(int k, int column) {
    return k * 128 + (column ^ ((k & 7) * 8));
}
SM89_FHS_HD constexpr unsigned a_copy_offset(int thread, int slice) {
    return unsigned(2 * a_index((thread >> 3) + slice * 32, (thread & 7) * 8));
}
SM89_FHS_HD constexpr unsigned b_copy_offset(int thread, int slice) {
    return unsigned(2 * b_index((thread >> 4) + slice * 16, (thread & 15) * 8));
}
SM89_FHS_HD constexpr unsigned a_fragment_base(int warp_m, int atom, int lane) {
    return unsigned(2 * a_index(warp_m + atom * 16 + (lane & 15), (lane & 16) ? 8 : 0));
}
SM89_FHS_HD constexpr unsigned b_fragment_base(int warp_n, int atom, int lane) {
    return unsigned(2 * b_index(lane & 15, warp_n + atom * 8));
}
SM89_FHS_HD constexpr unsigned a_fragment_issue(unsigned base, int issue) {
    // XOR touches only the low seven byte bits; each A row starts at 128B.
    return base ^ unsigned(issue * 32);
}
SM89_FHS_HD constexpr unsigned b_fragment_issue(unsigned base, int issue) {
    // Advancing K by 16 leaves the low-three-row-bit permutation unchanged.
    return base + unsigned(issue * 16 * 128 * 2);
}
} // namespace sm89_fixed_half_swizzle_layout
#undef SM89_FHS_HD
#line 1 "kernels/gemm_bi_inference/sm80/half_swizzle.cu"
// Production Fixed SM89 homogeneous-half packed/XOR staging twin.
// Only staging addresses change; exact ascending k16 chain, copy issue
// schedule, bias seed, alpha/beta, conversion and 136-float output stride stay.
// Dynamic shared is 69,632 bytes: packed S2 inputs use 65,536, then the
// unchanged vector epilogue aliases 128*136*4 bytes. Threads=256.
// Every scalar and cp.async staging path uses the SAME tested layout helper.
// Actual GPU output, sanitizer and speed qualification remain required.
// Composed only in the Fixed/sm_89 suffix; every other target remains byte-identical.
// Exports: nn_sm89_tc128_swizzle_bf16 and
// nn_sm89_tc128_swizzle_f16.

struct FixedSm89HalfSwizzleParams {
    float alpha;
    float beta;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
};
static_assert(sizeof(FixedSm89HalfSwizzleParams) == 32, "Fixed SM89 swizzle parameter size");
static_assert(alignof(FixedSm89HalfSwizzleParams) == 4, "Fixed SM89 swizzle parameter alignment");
static_assert(__is_standard_layout(FixedSm89HalfSwizzleParams), "Fixed SM89 swizzle standard layout");
static_assert(sizeof(float) == 4 && sizeof(int) == 4, "Fixed SM89 swizzle field widths");

namespace sm89_fixed_half_swizzle {

namespace layout = sm89_fixed_half_swizzle_layout;
static constexpr int kSharedBytes = layout::kSharedBytes;
static constexpr int kOutputStride = 136;
static_assert(2 * (128 * 64 + 64 * 128) * 2 <= kSharedBytes, "packed S2 shared ABI");
static_assert(128 * kOutputStride * 4 <= kSharedBytes, "epilogue aliases S2 storage");
static_assert(sizeof(uint4) == 16, "vector store width");

template <typename T> struct HalfOps;

#define SM89_FHS_OPS(TYPE, FROM, MMA_TYPE)                                    \
template <> struct HalfOps<TYPE> {                                         \
    static __device__ __forceinline__ TYPE from_float(float value) {        \
        return FROM(value);                                                \
    }                                                                      \
    static __device__ __forceinline__ void mma(                             \
        float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {      \
        asm volatile(                                                      \
            "mma.sync.aligned.m16n8k16.row.col.f32." MMA_TYPE "."            \
            MMA_TYPE ".f32 "                                               \
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"       \
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])               \
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),                  \
              "r"(b[0]), "r"(b[1]));                                       \
    }                                                                      \
};

SM89_FHS_OPS(__nv_bfloat16, from_f_bf16, "bf16")
SM89_FHS_OPS(__half, from_f_f16, "f16")
#undef SM89_FHS_OPS

template <typename T>
static __device__ __forceinline__ void stage_scalar(
    T* a_stages, T* b_stages, const T* A, const T* B,
    int stage, int k_base, int pid_m, int pid_n,
    int M, int N, int K, int lda, int ldb) {
    T* a_stage = a_stages + stage * 128 * 64;
    T* b_stage = b_stages + stage * 64 * 128;
    for (int linear = (int)threadIdx.x; linear < 128 * 64; linear += 256) {
        int row = linear / 64;
        int k = linear % 64;
        int global_row = pid_m * 128 + row;
        int global_k = k_base + k;
        a_stage[layout::a_index(row, k)] = global_row < M && global_k < K
            ? A[(long long)global_row * lda + global_k] : HalfOps<T>::from_float(0.0f);
    }
    for (int linear = (int)threadIdx.x; linear < 64 * 128; linear += 256) {
        int k = linear / 128;
        int column = linear % 128;
        int global_k = k_base + k;
        int global_column = pid_n * 128 + column;
        b_stage[layout::b_index(k, column)] = global_k < K && global_column < N
            ? B[(long long)global_k * ldb + global_column] : HalfOps<T>::from_float(0.0f);
    }
}

struct FragmentOffsets {
    unsigned a[4];
    unsigned b[4];
};

struct Fragments {
    unsigned a[4][4];
    unsigned b[4][2];
};

static __device__ __forceinline__ FragmentOffsets fragment_offsets(int warpM, int warpN) {
    FragmentOffsets offsets;
    int lane = (int)threadIdx.x & 31;
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        offsets.a[atom] = layout::a_fragment_base(warpM, atom, lane);
        offsets.b[atom] = layout::b_fragment_base(warpN, atom, lane);
    }
    return offsets;
}

static __device__ __forceinline__ void load_fragments(
    unsigned a_stage, unsigned b_stage, int issue,
    const FragmentOffsets& offsets, Fragments& fragments) {
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        unsigned address = a_stage + layout::a_fragment_issue(offsets.a[atom], issue);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(fragments.a[atom][0]), "=r"(fragments.a[atom][1]),
              "=r"(fragments.a[atom][2]), "=r"(fragments.a[atom][3]) : "r"(address));
    }
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        unsigned address = b_stage + layout::b_fragment_issue(offsets.b[atom], issue);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(fragments.b[atom][0]), "=r"(fragments.b[atom][1]) : "r"(address));
    }
}

template <typename T>
static __device__ __forceinline__ void consume_fragments(
    const Fragments& fragments, float (&acc)[4][4][4]) {
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            HalfOps<T>::mma(acc[fm][fn], fragments.a[fm], fragments.b[fn]);
        }
    }
}

struct CopyPlan {
    long long a_offset[4];
    long long b_offset[4];
    unsigned a_destination[4];
    unsigned b_destination[4];
    bool a_row_valid[4];
    int a_k;
    int b_k;
    int b_column_bytes;
};

static __device__ __forceinline__ CopyPlan copy_plan(
    unsigned a_shared, unsigned b_shared, int pid_m, int pid_n,
    int M, int N, int lda, int ldb) {
    CopyPlan plan;
    int thread = (int)threadIdx.x;
    plan.a_k = (thread & 7) * 8;
    plan.b_k = thread >> 4;
    int local_column = (thread & 15) * 8;
    int global_column = pid_n * 128 + local_column;
    int remaining = N - global_column;
    plan.b_column_bytes = remaining >= 8 ? 16 : (remaining > 0 ? remaining * 2 : 0);
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int local_row = (thread >> 3) + slice * 32;
        int global_row = pid_m * 128 + local_row;
        plan.a_row_valid[slice] = global_row < M;
        plan.a_offset[slice] = (long long)(global_row < M ? global_row : 0) * lda + plan.a_k;
        plan.b_offset[slice] = (long long)(plan.b_k + slice * 16) * ldb
            + (remaining > 0 ? global_column : 0);
        plan.a_destination[slice] = a_shared + layout::a_copy_offset(thread, slice);
        plan.b_destination[slice] = b_shared + layout::b_copy_offset(thread, slice);
    }
    return plan;
}

template <typename T>
static __device__ __forceinline__ void copy_slice(
    const CopyPlan& plan, const T* A, const T* B,
    int stage, int k_base, long long b_reduction_offset, int K, int slice) {
    int remaining = K - k_base - plan.a_k;
    int a_bytes = plan.a_row_valid[slice]
        ? (remaining >= 8 ? 16 : (remaining > 0 ? remaining * 2 : 0)) : 0;
    int b_bytes = k_base + plan.b_k + slice * 16 < K ? plan.b_column_bytes : 0;
    // Only form an in-object source when at least one element is read.
    const void* a_source = a_bytes > 0 ? (const void*)(A + plan.a_offset[slice] + k_base) : (const void*)A;
    const void* b_source = b_bytes > 0 ? (const void*)(B + plan.b_offset[slice] + b_reduction_offset) : (const void*)B;
    unsigned a_destination = plan.a_destination[slice] + (unsigned)(stage * 128 * 64 * 2);
    unsigned b_destination = plan.b_destination[slice] + (unsigned)(stage * 64 * 128 * 2);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(a_destination), "l"(a_source), "r"(a_bytes));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(b_destination), "l"(b_source), "r"(b_bytes));
}

template <typename T>
static __device__ __forceinline__ void scalar_epilogue(
    T* C, float (&acc)[4][4][4], float alpha, float beta,
    int M, int N, int ldc, int pid_m, int pid_n, int warpM, int warpN) {
    int lane = (int)threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            int r0 = pid_m * 128 + warpM + fm * 16 + g;
            int c0 = pid_n * 128 + warpN + fn * 8 + 2 * t;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = r0 + half * 8;
                if (row >= M) continue;
                float first = __fmul_rn(alpha, acc[fm][fn][2 * half]);
                float second = __fmul_rn(alpha, acc[fm][fn][2 * half + 1]);
                T* destination = c0 < N ? C + (long long)row * ldc + c0 : (T*)0;
                if (beta == 0.0f && (ldc & 1) == 0 && c0 + 1 < N && gbf_aligned4(destination)) {
                    gbf_store_pair_rne(destination, first, second);
                } else {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        int column = c0 + e;
                        if (column >= N) continue;
                        float value = e ? second : first;
                        T* output = C + (long long)row * ldc + column;
                        if (beta != 0.0f) value = __fmaf_rn(beta, to_f(*output), value);
                        *output = HalfOps<T>::from_float(value);
                    }
                }
            }
        }
    }
}

template <typename T>
static __device__ __forceinline__ void vector_epilogue(
    T* C, float* output_tile, float (&acc)[4][4][4], float alpha,
    int M, int ldc, int pid_m, int pid_n, int warpM, int warpN) {
    int lane = (int)threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    // Every warp must finish its last ldmatrix reads before aliasing As/Bs.
    __syncthreads();
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = warpM + fm * 16 + g + half * 8;
                int column = warpN + fn * 8 + 2 * t;
                *reinterpret_cast<float2*>(output_tile + row * kOutputStride + column) =
                    make_float2(acc[fm][fn][2 * half], acc[fm][fn][2 * half + 1]);
            }
        }
    }
    __syncthreads();
    // Each warp writes two adjacent rows in 16-byte chunks. The shared row
    // padding preserves 16-byte addresses for both float4 reads.
#pragma unroll
    for (int linear = (int)threadIdx.x; linear < 128 * 16; linear += 256) {
        int local_row = linear >> 4;
        int row = pid_m * 128 + local_row;
        if (row >= M) continue;
        int column = (linear & 15) * 8;
        const float* source = output_tile + local_row * kOutputStride + column;
        float4 first = *reinterpret_cast<const float4*>(source);
        float4 second = *reinterpret_cast<const float4*>(source + 4);
        uint4 packed;
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.x), __fmul_rn(alpha, first.x), __fmul_rn(alpha, first.y));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.y), __fmul_rn(alpha, first.z), __fmul_rn(alpha, first.w));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.z), __fmul_rn(alpha, second.x), __fmul_rn(alpha, second.y));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.w), __fmul_rn(alpha, second.z), __fmul_rn(alpha, second.w));
        *reinterpret_cast<uint4*>(C + (long long)row * ldc + pid_n * 128 + column) = packed;
    }
}

template <typename T>
static __device__ __forceinline__ void kernel(
    T* C, const T* A, const T* B, const float* bias,
    float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc) {
    static_assert(sizeof(T) == 2, "homogeneous half pipeline only");
    assert(alpha == 1.0f || bias == nullptr);
    extern __shared__ __align__(16) unsigned char sm89_fhs_shared[];
    T (*As)[128][64] = reinterpret_cast<T (*)[128][64]>(sm89_fhs_shared);
    T (*Bs)[64][128] = reinterpret_cast<T (*)[64][128]>(
        sm89_fhs_shared + 2 * 128 * 64 * (int)sizeof(T));
    int num_pid_n = (N + 127) / 128;
    int pid_m = (int)blockIdx.x / num_pid_n;
    int pid_n = (int)blockIdx.x % num_pid_n;
    int warp = (int)threadIdx.x >> 5;
    int warpM = (warp >> 2) * 64;
    int warpN = (warp & 3) * 32;
    int t = (int)threadIdx.x & 3;
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(&As[0][0][0]);
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(&Bs[0][0][0]);
    bool fast_stage = (lda & 7) == 0 && (ldb & 7) == 0 && gbf_aligned16(A) && gbf_aligned16(B);
    float acc[4][4][4];
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            int column = pid_n * 128 + warpN + fn * 8 + 2 * t;
            float first = bias != nullptr && column < N ? bias[column] : 0.0f;
            float second = bias != nullptr && column + 1 < N ? bias[column + 1] : 0.0f;
            acc[fm][fn][0] = first;
            acc[fm][fn][1] = second;
            acc[fm][fn][2] = first;
            acc[fm][fn][3] = second;
        }
    }
    FragmentOffsets offsets = fragment_offsets(warpM, warpN);
    CopyPlan plan;
    if (fast_stage) plan = copy_plan(As_sbase, Bs_sbase, pid_m, pid_n, M, N, lda, ldb);
    int num_k_tiles = (K + 63) / 64;
    if (num_k_tiles > 0) {
        if (fast_stage) {
#pragma unroll
            for (int slice = 0; slice < 4; ++slice) copy_slice(plan, A, B, 0, 0, 0, K, slice);
            asm volatile("cp.async.commit_group;\n" ::);
        } else {
            stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, 0, 0, pid_m, pid_n, M, N, K, lda, ldb);
        }
    }
    int read_buf = 0;
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        if (fast_stage) asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        bool next = kt + 1 < num_k_tiles;
        int next_k = (kt + 1) * 64;
        if (next) {
            if (!fast_stage) {
                stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, read_buf ^ 1, next_k,
                    pid_m, pid_n, M, N, K, lda, ldb);
            }
        }
        unsigned a_read = As_sbase + (unsigned)(read_buf * 128 * 64 * 2);
        unsigned b_read = Bs_sbase + (unsigned)(read_buf * 64 * 128 * 2);
        {
            Fragments fragments[2];
            load_fragments(a_read, b_read, 0, offsets, fragments[0]);
            long long next_b = (long long)next_k * ldb;
#pragma unroll
            for (int issue = 0; issue < 4; ++issue) {
                if (fast_stage && next) copy_slice(plan, A, B, read_buf ^ 1, next_k, next_b, K, issue);
                if (issue < 3) load_fragments(a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
                consume_fragments<T>(fragments[issue & 1], acc);
            }
            if (fast_stage && next) asm volatile("cp.async.commit_group;\n" ::);
        }
        read_buf ^= 1;
    }
    {
        bool vector_output = beta == 0.0f && (ldc & 7) == 0 && gbf_aligned16(C)
            && pid_n * 128 + 128 <= N;
        if (vector_output) {
            vector_epilogue(C, reinterpret_cast<float*>(sm89_fhs_shared), acc, alpha,
                M, ldc, pid_m, pid_n, warpM, warpN);
            return;
        }
    }
    scalar_epilogue(C, acc, alpha, beta, M, N, ldc, pid_m, pid_n, warpM, warpN);
}

} // namespace sm89_fixed_half_swizzle

#define SM89_FHS_EXPORT(TYPE, SUFFIX)                                      \
extern "C" __global__ __launch_bounds__(256, 1)                            \
void nn_sm89_tc128_swizzle_##SUFFIX(                      \
    TYPE* __restrict__ C, const TYPE* __restrict__ A,                       \
    const TYPE* __restrict__ B, const float* __restrict__ bias,             \
    FixedSm89HalfSwizzleParams params) {                                    \
    sm89_fixed_half_swizzle::kernel<TYPE>(C, A, B, bias, params.alpha,      \
        params.beta, params.m, params.n, params.k, params.lda, params.ldb,   \
        params.ldc);                                                         \
}
SM89_FHS_EXPORT(__nv_bfloat16, bf16)
SM89_FHS_EXPORT(__half, f16)
#undef SM89_FHS_EXPORT
#line 1 "kernels/gemm_bi_inference/sm80/half_s3.cu"
// Ada homogeneous-half CTA128x128/BK64/S3. The two-stage provider above
// supplies unchanged layout, fragment, arithmetic and epilogue helpers.
namespace sm89_fixed_half_s3 {
static constexpr int kSharedBytes = 98304;
static_assert(3 * (128 * 64 + 64 * 128) * 2 == kSharedBytes, "S3 shared ABI");
template <typename T>
static __device__ __forceinline__ void kernel(
    T* C, const T* A, const T* B, const float* bias,
    float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc) {
    static_assert(sizeof(T) == 2, "homogeneous half pipeline only");
    assert(alpha == 1.0f || bias == nullptr);
    extern __shared__ __align__(16) unsigned char sm89_fhs_shared[];
    T (*As)[128][64] = reinterpret_cast<T (*)[128][64]>(sm89_fhs_shared);
    T (*Bs)[64][128] = reinterpret_cast<T (*)[64][128]>(
        sm89_fhs_shared + 3 * 128 * 64 * (int)sizeof(T));
    int num_pid_n = (N + 127) / 128;
    int pid_m = (int)blockIdx.x / num_pid_n;
    int pid_n = (int)blockIdx.x % num_pid_n;
    int warp = (int)threadIdx.x >> 5;
    int warpM = (warp >> 2) * 64;
    int warpN = (warp & 3) * 32;
    int t = (int)threadIdx.x & 3;
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(&As[0][0][0]);
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(&Bs[0][0][0]);
    bool fast_stage = (lda & 7) == 0 && (ldb & 7) == 0 && gbf_aligned16(A) && gbf_aligned16(B);
    float acc[4][4][4];
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            int column = pid_n * 128 + warpN + fn * 8 + 2 * t;
            float first = bias != nullptr && column < N ? bias[column] : 0.0f;
            float second = bias != nullptr && column + 1 < N ? bias[column + 1] : 0.0f;
            acc[fm][fn][0] = first;
            acc[fm][fn][1] = second;
            acc[fm][fn][2] = first;
            acc[fm][fn][3] = second;
        }
    }
    sm89_fixed_half_swizzle::FragmentOffsets offsets = sm89_fixed_half_swizzle::fragment_offsets(warpM, warpN);
    sm89_fixed_half_swizzle::CopyPlan plan;
    if (fast_stage) plan = sm89_fixed_half_swizzle::copy_plan(As_sbase, Bs_sbase, pid_m, pid_n, M, N, lda, ldb);
    int num_k_tiles = (K + 63) / 64;
    if (num_k_tiles > 0) {
        if (fast_stage) {
#pragma unroll
            for (int slice = 0; slice < 4; ++slice) sm89_fixed_half_swizzle::copy_slice(plan, A, B, 0, 0, 0, K, slice);
            asm volatile("cp.async.commit_group;\n" ::);
            if (num_k_tiles > 1) {
#pragma unroll
                for (int slice = 0; slice < 4; ++slice)
                    sm89_fixed_half_swizzle::copy_slice(plan, A, B, 1, 64, (long long)64 * ldb, K, slice);
                asm volatile("cp.async.commit_group;\n" ::);
            }
        } else {
            sm89_fixed_half_swizzle::stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, 0, 0, pid_m, pid_n, M, N, K, lda, ldb);
        }
    }
    int read_buf = 0;
    if (fast_stage) {
        sm89_fixed_half_swizzle::Fragments fragments[2];
        if (num_k_tiles > 0) {
            if (num_k_tiles > 1) asm volatile("cp.async.wait_group 1;\n" ::);
            else asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            sm89_fixed_half_swizzle::load_fragments(As_sbase, Bs_sbase, 0, offsets, fragments[0]);
        }
        int write_buf = 2;
        for (int kt = 0; kt < num_k_tiles; ++kt) {
            bool next = kt + 1 < num_k_tiles;
            bool refill = kt + 2 < num_k_tiles;
            int next_k = (kt + 2) * 64;
            long long next_b = (long long)next_k * ldb;
            unsigned a_read = As_sbase + (unsigned)(read_buf * 128 * 64 * 2);
            unsigned b_read = Bs_sbase + (unsigned)(read_buf * 64 * 128 * 2);

            if (refill) sm89_fixed_half_swizzle::copy_slice(plan, A, B, write_buf, next_k, next_b, K, 0);
            sm89_fixed_half_swizzle::load_fragments(a_read, b_read, 1, offsets, fragments[1]);
            sm89_fixed_half_swizzle::consume_fragments<T>(fragments[0], acc);

            if (refill) sm89_fixed_half_swizzle::copy_slice(plan, A, B, write_buf, next_k, next_b, K, 1);
            sm89_fixed_half_swizzle::load_fragments(a_read, b_read, 2, offsets, fragments[0]);
            sm89_fixed_half_swizzle::consume_fragments<T>(fragments[1], acc);

            if (refill) {
                sm89_fixed_half_swizzle::copy_slice(plan, A, B, write_buf, next_k, next_b, K, 2);
                sm89_fixed_half_swizzle::copy_slice(plan, A, B, write_buf, next_k, next_b, K, 3);
            }
            sm89_fixed_half_swizzle::load_fragments(a_read, b_read, 3, offsets, fragments[1]);
            sm89_fixed_half_swizzle::consume_fragments<T>(fragments[0], acc);

            if (refill) asm volatile("cp.async.commit_group;\n" ::);
            if (next) {
                if (refill) asm volatile("cp.async.wait_group 1;\n" ::);
                else asm volatile("cp.async.wait_group 0;\n" ::);
                __syncthreads();
                read_buf = read_buf == 2 ? 0 : read_buf + 1;
                write_buf = write_buf == 2 ? 0 : write_buf + 1;
                unsigned next_a_read = As_sbase + (unsigned)(read_buf * 128 * 64 * 2);
                unsigned next_b_read = Bs_sbase + (unsigned)(read_buf * 64 * 128 * 2);
                sm89_fixed_half_swizzle::load_fragments(next_a_read, next_b_read, 0, offsets, fragments[0]);
            }
            sm89_fixed_half_swizzle::consume_fragments<T>(fragments[1], acc);
        }
    } else {
        for (int kt = 0; kt < num_k_tiles; ++kt) {
            __syncthreads();
            bool next = kt + 1 < num_k_tiles;
            int next_k = (kt + 1) * 64;
            if (next) {
                sm89_fixed_half_swizzle::stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, read_buf ^ 1, next_k,
                    pid_m, pid_n, M, N, K, lda, ldb);
            }
            unsigned a_read = As_sbase + (unsigned)(read_buf * 128 * 64 * 2);
            unsigned b_read = Bs_sbase + (unsigned)(read_buf * 64 * 128 * 2);
            sm89_fixed_half_swizzle::Fragments fragments[2];
            sm89_fixed_half_swizzle::load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
            for (int issue = 0; issue < 4; ++issue) {
                if (issue < 3)
                    sm89_fixed_half_swizzle::load_fragments(a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
                sm89_fixed_half_swizzle::consume_fragments<T>(fragments[issue & 1], acc);
            }
            read_buf ^= 1;
        }
    }
    {
        bool vector_output = beta == 0.0f && (ldc & 7) == 0 && gbf_aligned16(C)
            && pid_n * 128 + 128 <= N;
        if (vector_output) {
            sm89_fixed_half_swizzle::vector_epilogue(C, reinterpret_cast<float*>(sm89_fhs_shared), acc, alpha,
                M, ldc, pid_m, pid_n, warpM, warpN);
            return;
        }
    }
    sm89_fixed_half_swizzle::scalar_epilogue(C, acc, alpha, beta, M, N, ldc, pid_m, pid_n, warpM, warpN);
}

} // namespace sm89_fixed_half_s3

extern "C" __global__ __launch_bounds__(256, 1)
void nn_sm89_tc128_s3_bf16(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B, const float* __restrict__ bias,
    FixedSm89HalfSwizzleParams params) {
    sm89_fixed_half_s3::kernel(C, A, B, bias, params.alpha, params.beta,
        params.m, params.n, params.k, params.lda, params.ldb, params.ldc);
}
extern "C" __global__ __launch_bounds__(256, 1)
void nn_sm89_tc128_s3_f16(
    __half* __restrict__ C, const __half* __restrict__ A,
    const __half* __restrict__ B, const float* __restrict__ bias,
    FixedSm89HalfSwizzleParams params) {
    sm89_fixed_half_s3::kernel(C, A, B, bias, params.alpha, params.beta,
        params.m, params.n, params.k, params.lda, params.ldb, params.ldc);
}
#line 1 "kernels/gemm_bi_inference/sm80/tf32_rna_n96.cu"
// Ada Fixed TF32 M128xN96/BK32/S3 finalist. Optional forced route until qualified.

struct GbfTf32N96Params {
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(GbfTf32N96Params) == 32, "N96 parameter ABI");
static_assert(alignof(GbfTf32N96Params) == 4, "N96 parameter alignment");

__device__ __forceinline__ int tf32n96_a_index(int row, int k) {
    int chunk = (k >> 2) ^ (row & 7);
    return row * 32 + chunk * 4 + (k & 3);
}

__device__ __forceinline__ int tf32n96_b_index(int k, int column) {
    int chunk = (column >> 2) ^ ((k & 3) << 1);
    return k * 96 + chunk * 4 + (column & 3);
}

struct Tf32n96CopyPlan {
    const float* a_source[4];
    const float* b_source[3];
    unsigned a_destination[4];
    unsigned b_destination[3];
    int a_k_offset;
    int b_k_row[3];
    bool a_row_valid[4];
    int b_column_bytes[3];
};

__device__ __forceinline__ void tf32n96_copy_plan(
    float* a_stage0, float* b_stage0, const float* a, const float* b,
    const GbfTf32N96Params& params, int tile_row, int tile_column,
    Tf32n96CopyPlan& plan) {
    plan.a_k_offset = ((int)threadIdx.x & 7) * 4;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int row = linear >> 3;
        int global_row = tile_row + row;
        plan.a_row_valid[slice] = global_row < params.m;
        plan.a_source[slice] =
            a + (long long)(plan.a_row_valid[slice] ? global_row : 0) * params.lda
            + plan.a_k_offset;
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(
            a_stage0 + tf32n96_a_index(row, plan.a_k_offset));
    }
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int k_row = linear / 24;
        int column = (linear % 24) * 4;
        int global_column = tile_column + column;
        int columns = params.n - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        plan.b_k_row[slice] = k_row;
        plan.b_column_bytes[slice] = columns * 4;
        plan.b_source[slice] =
            b + (long long)k_row * params.ldb + (columns > 0 ? global_column : 0);
        plan.b_destination[slice] = (unsigned)__cvta_generic_to_shared(
            b_stage0 + tf32n96_b_index(k_row, column));
    }
}

__device__ __forceinline__ void tf32n96_stage_slice(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction, int issue) {
    {
        int remaining = reduction - k_base - plan.a_k_offset;
        remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
        int bytes = plan.a_row_valid[issue] ? remaining * 4 : 0;
        gbf_tf32_copy_cg(
            plan.a_destination[issue] + a_stage_bytes,
            plan.a_source[issue], bytes);
    }
    if (issue < 3) {
        int bytes = k_base + plan.b_k_row[issue] < reduction
            ? plan.b_column_bytes[issue]
            : 0;
        gbf_tf32_copy_cg(
            plan.b_destination[issue] + b_stage_bytes,
            plan.b_source[issue], bytes);
    }
}

__device__ __forceinline__ void tf32n96_advance_plan(
    Tf32n96CopyPlan& plan, long long b_rows) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) plan.a_source[slice] += 32;
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) plan.b_source[slice] += b_rows;
}

__device__ __forceinline__ void tf32n96_stage_async(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        tf32n96_stage_slice(
            plan, a_stage_bytes, b_stage_bytes, k_base, reduction, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

struct Tf32n96Fragments {
    unsigned a[4][4];
    unsigned b[3][2];
};

struct Tf32n96FragmentOffsets {
    int a[4][4];
    int b[3][2];
};

__device__ __forceinline__ void tf32n96_fragment_offsets(
    int warp_m, int warp_n, int group, int thread, int lane,
    Tf32n96FragmentOffsets& offsets) {
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            offsets.a[m_atom][step] =
                tf32n96_a_index(a_row + m_atom * 16, step * 8 + a_k);
        }
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        offsets.b[n_atom][0] = tf32n96_b_index(thread, column);
        offsets.b[n_atom][1] = tf32n96_b_index(thread + 4, column);
    }
}

__device__ __forceinline__ unsigned tf32n96_round(unsigned bits) {
    float value = __uint_as_float(bits);
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ void tf32n96_load_fragments(
    const float* a_stage, const float* b_stage, int step,
    const Tf32n96FragmentOffsets& offsets, Tf32n96Fragments& fragments) {
    unsigned a_base = (unsigned)__cvta_generic_to_shared(a_stage);
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
        unsigned address = a_base + (unsigned)offsets.a[m_atom][step] * 4U;
        unsigned raw0, raw1, raw2, raw3;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
            : "r"(address));
        fragments.a[m_atom][0] = tf32n96_round(raw0);
        fragments.a[m_atom][1] = tf32n96_round(raw1);
        fragments.a[m_atom][2] = tf32n96_round(raw2);
        fragments.a[m_atom][3] = tf32n96_round(raw3);
    }
    const float* b_step = b_stage + step * 8 * 96;
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        fragments.b[n_atom][0] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}

__device__ __forceinline__ void tf32n96_mma(
    const Tf32n96Fragments& fragments, float (&acc)[4][3][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

__device__ __forceinline__ void tf32n96_store(
    float* output, int row, int column, float accumulator,
    const GbfTf32N96Params& params) {
    if (row >= params.m || column >= params.n) return;
    float value = params.alpha == 1.0f
        ? accumulator
        : __fmul_rn(params.alpha, accumulator);
    float* destination = output + (long long)row * params.ldc + column;
    if (params.beta != 0.0f) value = __fmaf_rn(params.beta, *destination, value);
    *destination = value;
}

__device__ __forceinline__ void tf32n96_zero_reduction(
    float* output, const float* bias, const GbfTf32N96Params& params) {
    int column_tiles = (params.n + 95) / 96;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * 96;
    for (int linear = (int)threadIdx.x; linear < 128 * 96;
         linear += (int)blockDim.x) {
        int row = tile_row + linear / 96;
        int column = tile_column + linear % 96;
        if (row < params.m && column < params.n) {
            float accumulator = bias == nullptr ? 0.0f : bias[column];
            tf32n96_store(output, row, column, accumulator, params);
        }
    }
}

__device__ __forceinline__ void tf32n96_kernel(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    float* a_stages = reinterpret_cast<float*>(shared_bytes);
    float* b_stages = a_stages + 3 * 128 * 32;
    int column_tiles = (params.n + 95) / 96;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * 96;
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 2) * 64;
    int warp_n = (warp & 3) * 24;
    int group = lane >> 2;
    int thread = lane & 3;
    float acc[4][3][4];
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group
                    + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                acc[m_atom][n_atom][element] =
                    row < params.m && column < params.n && bias != nullptr
                    ? bias[column]
                    : 0.0f;
            }
        }
    }
    Tf32n96FragmentOffsets offsets;
    tf32n96_fragment_offsets(warp_m, warp_n, group, thread, lane, offsets);
    Tf32n96CopyPlan plan;
    tf32n96_copy_plan(
        a_stages, b_stages, a, b, params, tile_row, tile_column, plan);
    long long b_slab_rows = 32LL * params.ldb;
    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        if (tile < tile_count) {
            tf32n96_stage_async(
                plan, tile * 128U * 32U * 4U, tile * 32U * 96U * 4U,
                (int)(tile * 32U), params.k);
            tf32n96_advance_plan(plan, b_slab_rows);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        unsigned next = tile + 2;
        bool has_next = next < tile_count;
        int write_stage = read_stage == 0 ? 2 : read_stage - 1;
        unsigned write_a_bytes = (unsigned)write_stage * 128U * 32U * 4U;
        unsigned write_b_bytes = (unsigned)write_stage * 32U * 96U * 4U;
        const float* a_read = a_stages + read_stage * 128 * 32;
        const float* b_read = b_stages + read_stage * 32 * 96;
        Tf32n96Fragments fragments[2];
        tf32n96_load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
                tf32n96_stage_slice(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 32U), params.k, issue);
            }
            if (issue < 3) {
                tf32n96_load_fragments(
                    a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
            }
            tf32n96_mma(fragments[issue & 1], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (has_next) tf32n96_advance_plan(plan, b_slab_rows);
        if (++read_stage == 3) read_stage = 0;
    }
    __syncthreads();
    float* tile_output = reinterpret_cast<float*>(shared_bytes);
    bool vector_rows = tile_column + 96 <= params.n && (params.ldc & 3) == 0
        && gbf_aligned16(output);
    if (vector_rows) {
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = warp_m + m_atom * 16 + group + half * 8;
                    int column = warp_n + n_atom * 8 + 2 * thread;
                    *reinterpret_cast<float2*>(tile_output + row * 104 + column) =
                        make_float2(
                            acc[m_atom][n_atom][2 * half],
                            acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        __syncthreads();
        bool scale = params.alpha != 1.0f;
        bool blend = params.beta != 0.0f;
#pragma unroll 4
        for (int linear = (int)threadIdx.x; linear < 128 * 24; linear += 256) {
            int row = linear / 24;
            int chunk = (linear % 24) * 4;
            int global_row = tile_row + row;
            if (global_row >= params.m) continue;
            float4 value = *reinterpret_cast<const float4*>(
                tile_output + row * 104 + chunk);
            float* destination =
                output + (long long)global_row * params.ldc + tile_column + chunk;
            if (scale) {
                value.x = __fmul_rn(params.alpha, value.x);
                value.y = __fmul_rn(params.alpha, value.y);
                value.z = __fmul_rn(params.alpha, value.z);
                value.w = __fmul_rn(params.alpha, value.w);
            }
            if (blend) {
                float4 old = *reinterpret_cast<const float4*>(destination);
                value.x = __fmaf_rn(params.beta, old.x, value.x);
                value.y = __fmaf_rn(params.beta, old.y, value.y);
                value.z = __fmaf_rn(params.beta, old.z, value.z);
                value.w = __fmaf_rn(params.beta, old.w, value.w);
            }
            *reinterpret_cast<float4*>(destination) = value;
        }
        return;
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group
                    + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                tf32n96_store(
                    output, row, column, acc[m_atom][n_atom][element], params);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void nn_sm89_rna_tf32_m128n96_bk32_s3(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {
    if (params.k == 0) {
        tf32n96_zero_reduction(output, bias, params);
        return;
    }
    tf32n96_kernel(output, a, b, bias, params);
}

#line 1 "kernels/gemm_bi_inference/sm80/half_n64.cu"
// Ada Fixed F16 N64 finalists. Optional forced routes until qualified.

namespace sm89_fixed_half_n64 {

template <int BM, int STAGES> struct RectTraits {
    static constexpr int kBn = 64;
    static constexpr int kBk = 64;
    static constexpr int kThreads = 128;
    static constexpr int kMAtoms = BM / 32;
    static constexpr int kASlices = BM * kBk / (kThreads * 8);
    static constexpr int kSharedBytes = STAGES * (BM * kBk + kBk * kBn) * 2;
    static constexpr int kOutputStride = 72;
};

template <int BM>
static __device__ __forceinline__ int rect_a_index(int row, int k) {
    return row * 64 + (k ^ ((row & 7) * 8));
}

static __device__ __forceinline__ int rect_b_index(int k, int column) {
    return k * 64 + (column ^ ((k & 7) * 8));
}

template <int BM, int STAGES, bool M_TAIL, typename T>
static __device__ __forceinline__ void rect_copy_issue(
    unsigned a_base, unsigned b_base, const T* A, const T* B,
    int stage, int tile, int pid_m, int pid_n, int M, int lda, int ldb, int issue) {
    using R = RectTraits<BM, STAGES>;
    int thread = (int)threadIdx.x;
#pragma unroll
    for (int q = 0; q < R::kASlices / 4; ++q) {
        int slice = issue + q * 4;
        int linear = thread + slice * R::kThreads;
        int row = linear >> 3;
        int k = (linear & 7) * 8;
        int global_row = pid_m * BM + row;
        int source_row = M_TAIL && global_row >= M ? 0 : global_row;
        const T* source = A + (long long)source_row * lda + tile * 64 + k;
        unsigned destination = a_base + (unsigned)(stage * BM * 64 * 2)
            + (unsigned)(2 * rect_a_index<BM>(row, k));
        if constexpr (M_TAIL) {
            int source_bytes = global_row < M ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                :: "r"(destination), "l"(source), "r"(source_bytes));
        } else {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(destination), "l"(source));
        }
    }
    int linear = thread + issue * R::kThreads;
    int k = linear >> 3;
    int column = (linear & 7) * 8;
    const T* source = B + (long long)(tile * 64 + k) * ldb + pid_n * 64 + column;
    unsigned destination = b_base + (unsigned)(stage * 64 * 64 * 2)
        + (unsigned)(2 * rect_b_index(k, column));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(destination), "l"(source));
}

template <int MATOMS> struct RectFragments {
    unsigned a[MATOMS][4];
    unsigned b[4][2];
};

template <int BM, int STAGES>
static __device__ __forceinline__ void rect_load(
    unsigned a_stage, unsigned b_stage, int warp_m, int warp_n, int issue,
    RectFragments<RectTraits<BM, STAGES>::kMAtoms>& fragments) {
    using R = RectTraits<BM, STAGES>;
    int lane = (int)threadIdx.x & 31;
#pragma unroll
    for (int atom = 0; atom < R::kMAtoms; ++atom) {
        unsigned address = a_stage + ((unsigned)(2 * rect_a_index<BM>(
            warp_m + atom * 16 + (lane & 15), (lane & 16) ? 8 : 0))
            ^ (unsigned)(issue * 32));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(fragments.a[atom][0]), "=r"(fragments.a[atom][1]),
              "=r"(fragments.a[atom][2]), "=r"(fragments.a[atom][3]) : "r"(address));
    }
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        unsigned address = b_stage + (unsigned)(2 * rect_b_index(
            lane & 15, warp_n + atom * 8)) + (unsigned)(issue * 16 * 64 * 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(fragments.b[atom][0]), "=r"(fragments.b[atom][1]) : "r"(address));
    }
}

template <int BM, int STAGES, typename T>
static __device__ __forceinline__ void rect_consume(
    const RectFragments<RectTraits<BM, STAGES>::kMAtoms>& fragments,
    float (&acc)[RectTraits<BM, STAGES>::kMAtoms][4][4]) {
    using R = RectTraits<BM, STAGES>;
#pragma unroll
    for (int fm = 0; fm < R::kMAtoms; ++fm)
#pragma unroll
        for (int fn = 0; fn < 4; ++fn)
            sm89_fixed_half_swizzle::HalfOps<T>::mma(acc[fm][fn], fragments.a[fm], fragments.b[fn]);
}

template <int BM, int STAGES, typename T>
static __device__ __forceinline__ void rect_epilogue(
    T* C, unsigned char* shared, float (&acc)[RectTraits<BM, STAGES>::kMAtoms][4][4],
    int M, int N, int ldc, int pid_m, int pid_n, int warp_m, int warp_n) {
    using R = RectTraits<BM, STAGES>;
    float* scratch = reinterpret_cast<float*>(shared);
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
    __syncthreads();
#pragma unroll
    for (int fm = 0; fm < R::kMAtoms; ++fm)
#pragma unroll
        for (int fn = 0; fn < 4; ++fn)
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = warp_m + fm * 16 + group + half * 8;
                int column = warp_n + fn * 8 + 2 * thread;
                *reinterpret_cast<float2*>(scratch + row * R::kOutputStride + column) =
                    make_float2(acc[fm][fn][2 * half], acc[fm][fn][2 * half + 1]);
            }
    __syncthreads();
    for (int linear = (int)threadIdx.x; linear < BM * 8; linear += R::kThreads) {
        int local_row = linear >> 3;
        int row = pid_m * BM + local_row;
        if (row >= M) continue;
        int column = (linear & 7) * 8;
        const float* source = scratch + local_row * R::kOutputStride + column;
        float4 first = *reinterpret_cast<const float4*>(source);
        float4 second = *reinterpret_cast<const float4*>(source + 4);
        uint4 packed;
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.x), first.x, first.y);
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.y), first.z, first.w);
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.z), second.x, second.y);
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.w), second.z, second.w);
        *reinterpret_cast<uint4*>(C + (long long)row * ldc + pid_n * 64 + column) = packed;
    }
}

template <int BM, int STAGES, bool S3, bool M_TAIL, typename T>
static __device__ __forceinline__ void rect_kernel(
    T* C, const T* A, const T* B, int M, int N, int K, int lda, int ldb, int ldc) {
    using R = RectTraits<BM, STAGES>;
    static_assert(R::kSharedBytes == 49152, "two-CTA shared budget");
    static_assert(BM * R::kOutputStride * 4 <= R::kSharedBytes, "rectangular epilogue fits");
    extern __shared__ __align__(16) unsigned char half_batch_shared[];
    int num_pid_n = (N + 63) / 64;
    int pid_m = (int)blockIdx.x / num_pid_n;
    int pid_n = (int)blockIdx.x % num_pid_n;
    int warp = (int)threadIdx.x >> 5;
    int warp_m = (warp >> 1) * (BM / 2);
    int warp_n = (warp & 1) * 32;
    unsigned a_base = (unsigned)__cvta_generic_to_shared(half_batch_shared);
    unsigned b_base = a_base + (unsigned)(STAGES * BM * 64 * 2);
    float acc[R::kMAtoms][4][4] = {};
    int tiles = K / 64;

    if constexpr (S3) {
#pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
#pragma unroll
            for (int issue = 0; issue < 4; ++issue)
                rect_copy_issue<BM, STAGES, M_TAIL>(a_base, b_base, A, B, tile, tile,
                    pid_m, pid_n, M, lda, ldb, issue);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        RectFragments<R::kMAtoms> fragments[2];
        rect_load<BM, STAGES>(a_base, b_base, warp_m, warp_n, 0, fragments[0]);
        int read_buf = 0;
        int write_buf = 2;
        for (int tile = 0; tile < tiles; ++tile) {
            bool refill = tile + 2 < tiles;
            bool next = tile + 1 < tiles;
            unsigned a_read = a_base + (unsigned)(read_buf * BM * 64 * 2);
            unsigned b_read = b_base + (unsigned)(read_buf * 64 * 64 * 2);
            if (refill) rect_copy_issue<BM, STAGES, M_TAIL>(a_base, b_base, A, B, write_buf,
                tile + 2, pid_m, pid_n, M, lda, ldb, 0);
            rect_load<BM, STAGES>(a_read, b_read, warp_m, warp_n, 1, fragments[1]);
            rect_consume<BM, STAGES, T>(fragments[0], acc);
            if (refill) rect_copy_issue<BM, STAGES, M_TAIL>(a_base, b_base, A, B, write_buf,
                tile + 2, pid_m, pid_n, M, lda, ldb, 1);
            rect_load<BM, STAGES>(a_read, b_read, warp_m, warp_n, 2, fragments[0]);
            rect_consume<BM, STAGES, T>(fragments[1], acc);
            if (refill) {
                rect_copy_issue<BM, STAGES, M_TAIL>(a_base, b_base, A, B, write_buf,
                    tile + 2, pid_m, pid_n, M, lda, ldb, 2);
                rect_copy_issue<BM, STAGES, M_TAIL>(a_base, b_base, A, B, write_buf,
                    tile + 2, pid_m, pid_n, M, lda, ldb, 3);
            }
            rect_load<BM, STAGES>(a_read, b_read, warp_m, warp_n, 3, fragments[1]);
            rect_consume<BM, STAGES, T>(fragments[0], acc);
            if (refill) asm volatile("cp.async.commit_group;\n" ::);
            if (next) {
                if (refill) asm volatile("cp.async.wait_group 1;\n" ::);
                else asm volatile("cp.async.wait_group 0;\n" ::);
                __syncthreads();
                read_buf = read_buf == 2 ? 0 : read_buf + 1;
                write_buf = write_buf == 2 ? 0 : write_buf + 1;
                rect_load<BM, STAGES>(a_base + (unsigned)(read_buf * BM * 64 * 2),
                    b_base + (unsigned)(read_buf * 64 * 64 * 2), warp_m, warp_n, 0,
                    fragments[0]);
            }
            rect_consume<BM, STAGES, T>(fragments[1], acc);
        }
    } else {
#pragma unroll
        for (int issue = 0; issue < 4; ++issue)
            rect_copy_issue<BM, STAGES, M_TAIL>(a_base, b_base, A, B, 0, 0,
                pid_m, pid_n, M, lda, ldb, issue);
        asm volatile("cp.async.commit_group;\n" ::);
        int read_buf = 0;
        for (int tile = 0; tile < tiles; ++tile) {
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            bool next = tile + 1 < tiles;
            unsigned a_read = a_base + (unsigned)(read_buf * BM * 64 * 2);
            unsigned b_read = b_base + (unsigned)(read_buf * 64 * 64 * 2);
            RectFragments<R::kMAtoms> fragments[2];
            rect_load<BM, STAGES>(a_read, b_read, warp_m, warp_n, 0, fragments[0]);
#pragma unroll
            for (int issue = 0; issue < 4; ++issue) {
                if (next) rect_copy_issue<BM, STAGES, M_TAIL>(a_base, b_base, A, B, read_buf ^ 1,
                    tile + 1, pid_m, pid_n, M, lda, ldb, issue);
                if (issue < 3) rect_load<BM, STAGES>(a_read, b_read, warp_m, warp_n,
                    issue + 1, fragments[(issue + 1) & 1]);
                rect_consume<BM, STAGES, T>(fragments[issue & 1], acc);
            }
            if (next) asm volatile("cp.async.commit_group;\n" ::);
            read_buf ^= 1;
        }
    }
    rect_epilogue<BM, STAGES, T>(C, half_batch_shared, acc,
        M, N, ldc, pid_m, pid_n, warp_m, warp_n);
}

} // namespace sm89_fixed_half_n64

extern "C" __global__ __launch_bounds__(128, 2)
void nn_sm89_m64n64_bk64_s3_f16(
    __half* C, const __half* A, const __half* B, const float* bias,
    FixedSm89HalfSwizzleParams params) {
    if (bias != nullptr || params.alpha != 1.0f || params.beta != 0.0f
        || params.m < 1 || params.m > 2048
        || params.k != 768 || params.n != 2304
        || params.lda != 768 || params.ldb != 2304 || params.ldc != 2304
        || !gbf_aligned16(A) || !gbf_aligned16(B) || !gbf_aligned16(C)) return;
    if (params.m == 2048) {
        sm89_fixed_half_n64::rect_kernel<64, 3, true, false>(C, A, B,
            2048, 2304, 768, 768, 2304, 2304);
    } else {
        sm89_fixed_half_n64::rect_kernel<64, 3, true, true>(C, A, B,
            params.m, 2304, 768, 768, 2304, 2304);
    }
}

extern "C" __global__ __launch_bounds__(128, 2)
void nn_sm89_m128n64_bk64_s2_f16(
    __half* C, const __half* A, const __half* B, const float* bias,
    FixedSm89HalfSwizzleParams params) {
    if (bias != nullptr || params.alpha != 1.0f || params.beta != 0.0f
        || params.m < 1 || params.m > 2048
        || params.k != 2304 || params.n != 768
        || params.lda != 2304 || params.ldb != 768 || params.ldc != 768
        || !gbf_aligned16(A) || !gbf_aligned16(B) || !gbf_aligned16(C)) return;
    if (params.m == 2048) {
        sm89_fixed_half_n64::rect_kernel<128, 2, false, false>(C, A, B,
            2048, 768, 2304, 2304, 768, 768);
    } else {
        sm89_fixed_half_n64::rect_kernel<128, 2, false, true>(C, A, B,
            params.m, 768, 2304, 2304, 768, 768);
    }
}
