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

#include "_typed_prelude.cuh"

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
    float h_local[64];
    float a_local[64];
    if (d_state > 64) return;
    for (int n = 0; n < d_state; n++) {
        h_local[n] = h[h_base + n];
        a_local[n] = a_neg[d * d_state + n];
    }

    float delta_d = delta[idx];
    float u_d = u[idx];
    float delta_u_d = delta_d * u_d; // C2: hoisted
    float y_d = D[d] * u_d;

    for (int n = 0; n < d_state; n++) {
        // Opt B: exp2f instead of expf
        float da = exp2f(delta_d * a_local[n] * LOG2E);
        h_local[n] = da * h_local[n] + delta_u_d * B[b * d_state + n];
        y_d += h_local[n] * C[b * d_state + n];
    }

    // Opt A: write back h once
    for (int n = 0; n < d_state; n++)
        h[h_base + n] = h_local[n];

    y[idx] = y_d;
}

// Templated SSM step forward — activations in T_IN, state/a_neg/D stay f32.
#define DEFINE_SSM_STEP_FWD(SUFFIX, T, FROM_F)                             \
extern "C" __global__ void ssm_step_forward_##SUFFIX(                       \
    float* h,                                                              \
    T* y,                                                                  \
    const T* delta,                                                        \
    const T* u,                                                            \
    const T* B,                                                            \
    const T* C,                                                            \
    const float* a_neg,                                                    \
    const float* D,                                                        \
    int batch, int d_inner, int d_state                                    \
) {                                                                        \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    int total = batch * d_inner;                                           \
    if (idx >= total) return;                                              \
    int b = idx / d_inner;                                                 \
    int d = idx % d_inner;                                                 \
    int h_base = (b * d_inner + d) * d_state;                              \
    float h_local[64];                                                     \
    float a_local[64];                                                     \
    if (d_state > 64) return;                                              \
    for (int n = 0; n < d_state; n++) {                                    \
        h_local[n] = h[h_base + n];                                        \
        a_local[n] = a_neg[d * d_state + n];                               \
    }                                                                      \
    float delta_d = to_f(delta[idx]);                                      \
    float u_d = to_f(u[idx]);                                              \
    float delta_u_d = delta_d * u_d;                                       \
    float y_d = D[d] * u_d;                                                \
    for (int n = 0; n < d_state; n++) {                                    \
        float da = exp2f(delta_d * a_local[n] * LOG2E);                    \
        h_local[n] = da * h_local[n] + delta_u_d * to_f(B[b * d_state + n]);\
        y_d += h_local[n] * to_f(C[b * d_state + n]);                      \
    }                                                                      \
    for (int n = 0; n < d_state; n++)                                      \
        h[h_base + n] = h_local[n];                                        \
    y[idx] = FROM_F(y_d);                                                  \
}

DEFINE_SSM_STEP_FWD(f32,  float,         from_f_f32)
DEFINE_SSM_STEP_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_STEP_FWD(f16,  __half,        from_f_f16)

// SSM step fused with B/C gather from xdbl — eliminates the gather_bc_cols
// kernel by reading B and C directly from xdbl with offsets. Saves one
// launch per layer per step plus the b_buf/c_buf scratch buffers.
//
// Layout: xdbl is [batch * xdbl_dim] where xdbl_dim = dt_rank + 2*d_state.
// B slice starts at offset dt_rank, C slice at dt_rank + d_state.
//
// Math identical to running gather_bc_cols + ssm_step_forward sequentially.
#define DEFINE_SSM_STEP_FWD_GATHER(SUFFIX, T, FROM_F)                      \
extern "C" __global__ void ssm_step_forward_gather_##SUFFIX(                \
    float* h,                                                              \
    T* y,                                                                  \
    const T* delta,                                                        \
    const T* u,                                                            \
    const T* xdbl,                                                         \
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
    float h_local[64];                                                     \
    float a_local[64];                                                     \
    if (d_state > 64) return;                                              \
    for (int n = 0; n < d_state; n++) {                                    \
        h_local[n] = h[h_base + n];                                        \
        a_local[n] = a_neg[d * d_state + n];                               \
    }                                                                      \
    float delta_d = to_f(delta[idx]);                                      \
    float u_d = to_f(u[idx]);                                              \
    float delta_u_d = delta_d * u_d;                                       \
    float y_d = D[d] * u_d;                                                \
    for (int n = 0; n < d_state; n++) {                                    \
        float da = exp2f(delta_d * a_local[n] * LOG2E);                    \
        float B_n = to_f(xdbl[xdbl_base + b_offset + n]);                  \
        float C_n = to_f(xdbl[xdbl_base + c_offset + n]);                  \
        h_local[n] = da * h_local[n] + delta_u_d * B_n;                    \
        y_d += h_local[n] * C_n;                                           \
    }                                                                      \
    for (int n = 0; n < d_state; n++)                                      \
        h[h_base + n] = h_local[n];                                        \
    y[idx] = FROM_F(y_d);                                                  \
}

DEFINE_SSM_STEP_FWD_GATHER(f32,  float,         from_f_f32)
DEFINE_SSM_STEP_FWD_GATHER(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_STEP_FWD_GATHER(f16,  __half,        from_f_f16)

// SSM step with fused B/C gather + final gating multiplication. Replaces
// the (gather_bc + ssm_step + elementwise_mul gate_silu) triplet with a
// single kernel — saves 2 launches per layer per step. Output y already
// includes the y *= gate_silu post-multiplication.
//
// gate_silu must have the same [batch * d_inner] shape and T_ACT dtype.
// Math: same as ssm_step_forward_gather followed by y *= gate_silu in
// elementwise_mul_typed.
#define DEFINE_SSM_STEP_FWD_GATHER_GATE(SUFFIX, T, FROM_F)                 \
extern "C" __global__ void ssm_step_forward_gather_gate_##SUFFIX(           \
    float* h,                                                              \
    T* y,                                                                  \
    const T* delta,                                                        \
    const T* u,                                                            \
    const T* xdbl,                                                         \
    const T* gate_silu,                                                    \
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
    float h_local[64];                                                     \
    float a_local[64];                                                     \
    if (d_state > 64) return;                                              \
    for (int n = 0; n < d_state; n++) {                                    \
        h_local[n] = h[h_base + n];                                        \
        a_local[n] = a_neg[d * d_state + n];                               \
    }                                                                      \
    float delta_d = to_f(delta[idx]);                                      \
    float u_d = to_f(u[idx]);                                              \
    float delta_u_d = delta_d * u_d;                                       \
    float y_d = D[d] * u_d;                                                \
    for (int n = 0; n < d_state; n++) {                                    \
        float da = exp2f(delta_d * a_local[n] * LOG2E);                    \
        float B_n = to_f(xdbl[xdbl_base + b_offset + n]);                  \
        float C_n = to_f(xdbl[xdbl_base + c_offset + n]);                  \
        h_local[n] = da * h_local[n] + delta_u_d * B_n;                    \
        y_d += h_local[n] * C_n;                                           \
    }                                                                      \
    for (int n = 0; n < d_state; n++)                                      \
        h[h_base + n] = h_local[n];                                        \
    /* Fused gating: y *= gate_silu */                                     \
    float gated = y_d * to_f(gate_silu[idx]);                              \
    y[idx] = FROM_F(gated);                                                \
}

DEFINE_SSM_STEP_FWD_GATHER_GATE(f32,  float,         from_f_f32)
DEFINE_SSM_STEP_FWD_GATHER_GATE(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_STEP_FWD_GATHER_GATE(f16,  __half,        from_f_f16)

// SSM burn-in forward (T>1): iterate T steps for each (batch, d_inner) thread.
// Saves h_saved[B*(T+1)*d_inner*d_state] for backward BPTT and
// h and a_neg cached in registers (Opt A). Uses exp2f (Opt B).
extern "C" __global__ void ssm_burnin_forward(
    float* h,             // [batch * d_inner * d_state] hidden state (mutated through T steps)
    float* y_out,         // [batch * T * d_inner] output
    float* h_saved,       // [batch * (T+1) * d_inner * d_state] h BEFORE each step
    const float* delta,   // [batch * T * d_inner]
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
    float h_local[64];
    float a_local[64];
    if (d_state > 64) return;
    for (int n = 0; n < d_state; n++) {
        h_local[n] = h[h_base + n];
        a_local[n] = a_neg[d * d_state + n];
    }

    // Save initial h state at time index 0
    for (int n = 0; n < d_state; n++) {
        int hs_idx = (b * (T + 1) + 0) * d_inner * d_state + d * d_state + n;
        h_saved[hs_idx] = h_local[n];
    }

    for (int t = 0; t < T; t++) {
        int bt_di = (b * T + t) * d_inner + d;
        int bt_ds = (b * T + t) * d_state;

        float delta_d = delta[bt_di];
        float u_d = u[bt_di];
        float delta_u_d = delta_d * u_d; // C2: hoisted
        float y_d = D[d] * u_d;

        for (int n = 0; n < d_state; n++) {
            // Opt B: exp2f instead of expf
            float da = exp2f(delta_d * a_local[n] * LOG2E);

            // No da saved: backward recomputes da from delta and a_neg
            // (cheaper than a global-memory round-trip at d_state=16).

            h_local[n] = da * h_local[n] + delta_u_d * B[bt_ds + n];
            y_d += h_local[n] * C[bt_ds + n];
        }

        y_out[bt_di] = y_d;

        // Save h AFTER step t = h_saved at time index t+1
        for (int n = 0; n < d_state; n++) {
            int hs_idx = (b * (T + 1) + (t + 1)) * d_inner * d_state + d * d_state + n;
            h_saved[hs_idx] = h_local[n];
        }
    }

    // Opt A: write back final h once
    for (int n = 0; n < d_state; n++)
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
    int batch, int T, int d_inner, int d_state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_inner;
    if (idx >= total) return;

    int b = idx / d_inner;
    int d = idx % d_inner;
    int h_base = (b * d_inner + d) * d_state;

    // Opt A: load h and a_neg into registers
    float h_local[64];
    float a_local[64];
    if (d_state > 64) return;
    for (int n = 0; n < d_state; n++) {
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

        for (int n = 0; n < d_state; n++) {
            // Opt B: exp2f instead of expf
            float da = exp2f(delta_d * a_local[n] * LOG2E);
            h_local[n] = da * h_local[n] + delta_u_d * B[bt_ds + n];
            y_d += h_local[n] * C[bt_ds + n];
        }

        y_out[bt_di] = y_d;
    }

    // Opt A: write back final h once
    for (int n = 0; n < d_state; n++)
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
    int batch, int T_len, int d_inner, int d_state                         \
) {                                                                        \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    int total = batch * d_inner;                                           \
    if (idx >= total) return;                                              \
    int b = idx / d_inner;                                                 \
    int d = idx % d_inner;                                                 \
    int h_base = (b * d_inner + d) * d_state;                              \
    float h_local[64];                                                     \
    float a_local[64];                                                     \
    if (d_state > 64) return;                                              \
    for (int n = 0; n < d_state; n++) {                                    \
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
        for (int n = 0; n < d_state; n++) {                                \
            float da = exp2f(delta_d * a_local[n] * LOG2E);                \
            h_local[n] = da * h_local[n] + delta_u_d * to_f(B[bt_ds + n]); \
            y_d += h_local[n] * to_f(C[bt_ds + n]);                        \
        }                                                                  \
        y_out[bt_di] = FROM_F(y_d);                                        \
    }                                                                      \
    for (int n = 0; n < d_state; n++)                                      \
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
    const TY* delta, const TY* u,                                           \
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
    float h_local[64];                                                      \
    float a_local[64];                                                      \
    if (d_state > 64) return;                                               \
    for (int n = 0; n < d_state; n++) {                                     \
        h_local[n] = h[h_base + n];                                         \
        a_local[n] = a_neg[d * d_state + n];                                \
    }                                                                       \
    for (int n = 0; n < d_state; n++) {                                     \
        int hs_idx = (b * (T_len + 1) + 0) * d_inner * d_state              \
                     + d * d_state + n;                                     \
        h_saved[hs_idx] = h_local[n];                                       \
    }                                                                       \
    for (int t = 0; t < T_len; t++) {                                       \
        int bt_di = (b * T_len + t) * d_inner + d;                          \
        int bt_ds = (b * T_len + t) * d_state;                              \
        float delta_d = to_f(delta[bt_di]);                                 \
        float u_d = to_f(u[bt_di]);                                         \
        float delta_u_d = delta_d * u_d;                                    \
        float y_d = D[d] * u_d;                                             \
        for (int n = 0; n < d_state; n++) {                                 \
            float da = exp2f(delta_d * a_local[n] * LOG2E);                 \
            h_local[n] = da * h_local[n] + delta_u_d * to_f(B[bt_ds + n]);  \
            y_d += h_local[n] * to_f(C[bt_ds + n]);                         \
        }                                                                   \
        y_out[bt_di] = FROM_F(y_d);                                         \
        for (int n = 0; n < d_state; n++) {                                 \
            int hs_idx = (b * (T_len + 1) + (t + 1)) * d_inner * d_state    \
                         + d * d_state + n;                                 \
            h_saved[hs_idx] = h_local[n];                                   \
        }                                                                   \
    }                                                                       \
    for (int n = 0; n < d_state; n++)                                       \
        h[h_base + n] = h_local[n];                                         \
}

DEFINE_SSM_BURNIN(f32,  float,         from_f_f32)
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
    float a_local[64];
    if (d_state > 64) return;
    for (int n = 0; n < d_state; n++)
        a_local[n] = a_neg[d * d_state + n];

    // d_h carries gradient backward through time
    float d_h[64];
    for (int n = 0; n < d_state; n++) d_h[n] = 0.0f;

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

        for (int n = 0; n < d_state; n++) {
            // h_curr = state AFTER step t = h_saved at time index t+1
            int h_idx = (b * (T + 1) + (t + 1)) * d_inner * d_state + (d * d_state + n);
            float h_curr = h_saved[h_idx];

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

            // d_a_log += d_h * da * delta * a_dn * h_prev
            d_a_log_local[(b * d_inner + d) * d_state + n] +=
                d_h[n] * da * delta_d * a_local[n] * h_prev;

            // Propagate d_h backward through time: d_h_prev = da * d_h
            d_h[n] = da * d_h[n];
        }

        d_delta[bt_di] = d_delta_val;
        d_u[bt_di] = d_u_val;
    }

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
//   - register dh[64], a_local[64], local_d_D → f32 (BPTT precision invariant)
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
    if (d_state > 64) return;                                                   \
                                                                                \
    float local_d_D = 0.0f;                                                     \
    float a_local[64];                                                          \
    float d_h[64];                                                              \
    for (int n = 0; n < d_state; n++) {                                         \
        a_local[n] = a_neg[d * d_state + n];                                    \
        d_h[n] = 0.0f;                                                          \
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
        for (int n = 0; n < d_state; n++) {                                     \
            int h_idx      = (b * (T + 1) + (t + 1)) * d_inner * d_state        \
                             + (d * d_state + n);                               \
            int h_prev_idx = (b * (T + 1) + t)       * d_inner * d_state        \
                             + (d * d_state + n);                               \
            float h_curr = h_saved[h_idx];                                      \
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
            d_a_log_local[(b * d_inner + d) * d_state + n] +=                   \
                d_h[n] * da * delta_d * a_local[n] * h_prev;                    \
                                                                                \
            d_h[n] = da * d_h[n];                                               \
        }                                                                       \
                                                                                \
        d_delta[bt_di] = FROM_F(d_delta_val);                                   \
        d_u[bt_di]     = FROM_F(d_u_val);                                       \
    }                                                                           \
                                                                                \
    d_D_local[b * d_inner + d] = local_d_D;                                     \
}

DEFINE_SSM_BACKWARD_LOCAL_BWD(f32,  float,         from_f_f32)
DEFINE_SSM_BACKWARD_LOCAL_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_BACKWARD_LOCAL_BWD(f16,  __half,        from_f_f16)

// Reduction kernels: sum per-sample gradients across batch dimension.

// Reduce d_B across d_inner: d_B_out[b*T*ds + t*ds + n] = sum_d(d_B_local[...])
extern "C" __global__ void ssm_reduce_d_B(
    float* d_B_out,           // [batch * T * d_state] accumulated
    const float* d_B_local,   // [batch * T * d_inner * d_state]
    int batch, int T, int d_inner, int d_state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * T * d_state;
    if (idx >= total) return;
    int bt = idx / d_state;
    int n = idx % d_state;
    float sum = 0.0f;
    for (int d = 0; d < d_inner; d++) {
        sum += d_B_local[(bt * d_inner + d) * d_state + n];
    }
    d_B_out[idx] += sum;
}

// Reduce d_C across d_inner (same pattern as d_B)
extern "C" __global__ void ssm_reduce_d_C(
    float* d_C_out,           // [batch * T * d_state] accumulated
    const float* d_C_local,   // [batch * T * d_inner * d_state]
    int batch, int T, int d_inner, int d_state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * T * d_state;
    if (idx >= total) return;
    int bt = idx / d_state;
    int n = idx % d_state;
    float sum = 0.0f;
    for (int d = 0; d < d_inner; d++) {
        sum += d_C_local[(bt * d_inner + d) * d_state + n];
    }
    d_C_out[idx] += sum;
}

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

// Typed reducers for d_B / d_C: when ssm_backward_local writes typed
// (bf16/f16) per-thread d_B_local / d_C_local, the reducer must promote
// each contribution to f32 in the inner sum loop. Output stays f32 master
// (gradient buffer is f32 always per AMP convention; no precision loss
// at write because dst is f32 += f32 sum).
//
// d_D and d_a_log reducers are NOT typed: their inputs are already f32
// per the precision rules in DEFINE_SSM_BACKWARD_LOCAL_BWD (T-length
// accumulators stay f32 regardless of activation dtype).
#define DEFINE_SSM_REDUCE_D_BC_TYPED(SUFFIX, TY, KERNEL_NAME)                  \
extern "C" __global__ void KERNEL_NAME##_##SUFFIX(                             \
    float* dst_out,                                                            \
    const TY* src_local,                                                       \
    int batch, int T, int d_inner, int d_state                                 \
) {                                                                            \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int total = batch * T * d_state;                                           \
    if (idx >= total) return;                                                  \
    int bt = idx / d_state;                                                    \
    int n = idx % d_state;                                                     \
    float sum = 0.0f;                                                          \
    for (int d = 0; d < d_inner; d++) {                                        \
        sum += to_f(src_local[(bt * d_inner + d) * d_state + n]);              \
    }                                                                          \
    dst_out[idx] += sum;                                                       \
}

DEFINE_SSM_REDUCE_D_BC_TYPED(bf16, __nv_bfloat16, ssm_reduce_d_B)
DEFINE_SSM_REDUCE_D_BC_TYPED(f16,  __half,        ssm_reduce_d_B)
DEFINE_SSM_REDUCE_D_BC_TYPED(bf16, __nv_bfloat16, ssm_reduce_d_C)
DEFINE_SSM_REDUCE_D_BC_TYPED(f16,  __half,        ssm_reduce_d_C)

// Reduce d_a_log: d_a_log_out[d*ds+n] = sum_b(d_a_log_local[b*di*ds + d*ds + n])
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
