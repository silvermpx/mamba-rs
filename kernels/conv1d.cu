// Depthwise Conv1d CUDA kernels for Mamba.
//
// Shift register pattern: state[d, 0..d_conv] updated each step.
// Depthwise: each channel d independent.
//
// Source: CPU reference: train/backward_ops.rs backward_conv1d_step

#include "_typed_prelude.cuh"

// Conv1d step forward (T=1): shift register + depthwise dot product
// state[b, d, 0..d_conv-1] = state[b, d, 1..d_conv]
// state[b, d, d_conv-1] = new_x[b, d]
// out[b, d] = sum_k(state[b, d, k] * weight[d, k]) + bias[d]
extern "C" __global__ void conv1d_step_forward(
    float* out,         // [batch * d_inner]
    float* state,       // [batch * d_inner * d_conv] mutated
    const float* new_x, // [batch * d_inner]
    const float* weight, // [d_inner * d_conv]
    const float* bias,  // [d_inner]
    int batch, int d_inner, int d_conv
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_inner;
    if (idx >= total) return;

    int b = idx / d_inner;
    int d = idx % d_inner;
    int state_base = (b * d_inner + d) * d_conv;

    // Shift register left
    for (int k = 0; k < d_conv - 1; k++) {
        state[state_base + k] = state[state_base + k + 1];
    }
    // Insert new value
    state[state_base + d_conv - 1] = new_x[idx];

    // Depthwise dot product
    float sum = bias[d];
    for (int k = 0; k < d_conv; k++) {
        sum += state[state_base + k] * weight[d * d_conv + k];
    }
    out[idx] = sum;
}

// Templated conv1d step — input activations in T_IN, output in T_IN,
// conv state/weight/bias stay f32.
#define DEFINE_CONV1D_STEP_FWD(SUFFIX, T, FROM_F)                          \
extern "C" __global__ void conv1d_step_forward_##SUFFIX(                    \
    T* out,                                                                \
    float* state,                                                          \
    const T* new_x,                                                        \
    const float* weight,                                                   \
    const float* bias,                                                     \
    int batch, int d_inner, int d_conv                                     \
) {                                                                        \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    int total = batch * d_inner;                                           \
    if (idx >= total) return;                                              \
    int b = idx / d_inner;                                                 \
    int d = idx % d_inner;                                                 \
    int state_base = (b * d_inner + d) * d_conv;                           \
    for (int k = 0; k < d_conv - 1; k++) {                                 \
        state[state_base + k] = state[state_base + k + 1];                 \
    }                                                                      \
    state[state_base + d_conv - 1] = to_f(new_x[idx]);                     \
    float sum = bias[d];                                                   \
    for (int k = 0; k < d_conv; k++) {                                     \
        sum += state[state_base + k] * weight[d * d_conv + k];             \
    }                                                                      \
    out[idx] = FROM_F(sum);                                                \
}

DEFINE_CONV1D_STEP_FWD(f32,  float,         from_f_f32)
DEFINE_CONV1D_STEP_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_CONV1D_STEP_FWD(f16,  __half,        from_f_f16)

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
    T* out,                                                                \
    float* state,                                                          \
    const T* new_x,                                                        \
    const float* weight,                                                   \
    const float* bias,                                                     \
    int batch, int d_inner, int d_conv                                     \
) {                                                                        \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    int total = batch * d_inner;                                           \
    if (idx >= total) return;                                              \
    int b = idx / d_inner;                                                 \
    int d = idx % d_inner;                                                 \
    int state_base = (b * d_inner + d) * d_conv;                           \
    for (int k = 0; k < d_conv - 1; k++) {                                 \
        state[state_base + k] = state[state_base + k + 1];                 \
    }                                                                      \
    state[state_base + d_conv - 1] = to_f(new_x[idx]);                     \
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

// Conv1d step backward:
//   d_new_x[b,d] = weight[d, d_conv-1] * dy[b,d]
//   d_weight_partials[b,d,k] = state_saved[b,d,k] * dy[b,d]  (per-sample)
//   d_bias_partials[b,d]     = dy[b,d]                        (per-sample)
//
// Rule B (no atomicAdd): caller MUST follow with two reduce_sum_axis0 launches
// to reduce across batch deterministically.
extern "C" __global__ __launch_bounds__(256, 4)
void conv1d_step_backward(
    float* __restrict__ d_new_x,             // [batch * d_inner]
    float* __restrict__ d_weight_partials,   // [batch * d_inner * d_conv] OUTPUT
    float* __restrict__ d_bias_partials,     // [batch * d_inner] OUTPUT
    const float* __restrict__ dy,            // [batch * d_inner]
    const float* __restrict__ state_saved,   // [batch * d_inner * d_conv]
    const float* __restrict__ weight,        // [d_inner * d_conv]
    int batch, int d_inner, int d_conv
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_inner;
    if (idx >= total) return;

    int b = idx / d_inner;
    int d = idx % d_inner;
    float dy_val = dy[idx];

    // d_new_x: gradient flows through position d_conv-1 only
    d_new_x[idx] = dy_val * weight[d * d_conv + d_conv - 1];

    // Rule B: per-(b,d) write to partials — caller reduces across batch.
    int state_base = (b * d_inner + d) * d_conv;
    int wp_base = (b * d_inner + d) * d_conv;
    for (int k = 0; k < d_conv; k++) {
        d_weight_partials[wp_base + k] = dy_val * state_saved[state_base + k];
    }
    d_bias_partials[b * d_inner + d] = dy_val;
}

// ======================== BURNIN (T>1) ========================

// Conv1d burnin forward (T>1): process all T steps for each (batch, d_inner) thread.
// Fused with SiLU: out[b,t,d] = silu(conv_out[b,t,d]).
// Saves conv_state after each step + pre-SiLU value for backward.
//
// Source: CPU reference: train/forward.rs phase F4a (conv1d + fused SiLU)
extern "C" __global__ void conv1d_burnin_forward(
    float* u_out,          // [batch * T * d_inner] post-SiLU output
    float* post_conv_out,  // [batch * T * d_inner] pre-SiLU (saved for backward)
    float* conv_states_out,// [batch * T * d_inner * d_conv] state after each step (saved for backward)
    float* state,          // [batch * d_inner * d_conv] persistent state (mutated)
    const float* x_branch, // [batch * T * d_inner] input from in_proj split
    const float* weight,   // [d_inner * d_conv]
    const float* bias,     // [d_inner]
    int batch, int T, int d_inner, int d_conv
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_inner;
    if (idx >= total) return;

    int b = idx / d_inner;
    int d = idx % d_inner;
    int state_base = (b * d_inner + d) * d_conv;

    // Register window (see the typed twin's note): same shifts, same
    // values, no per-timestep global RMW chain.
    float win[8];
    if (d_conv > 8) return;
    for (int k = 0; k < d_conv; k++) win[k] = state[state_base + k];
    for (int t = 0; t < T; t++) {
        int bt_di = (b * T + t) * d_inner + d;

        for (int k = 0; k < d_conv - 1; k++) {
            win[k] = win[k + 1];
        }
        win[d_conv - 1] = x_branch[bt_di];

        // Save conv_state after shift (for backward)
        int cs_base = ((b * T + t) * d_inner + d) * d_conv;
        for (int k = 0; k < d_conv; k++) {
            conv_states_out[cs_base + k] = win[k];
        }

        // Depthwise dot product
        float val = bias[d];
        for (int k = 0; k < d_conv; k++) {
            val += win[k] * weight[d * d_conv + k];
        }
        post_conv_out[bt_di] = val;

        // Fused SiLU: u = val * sigmoid(val)
        u_out[bt_di] = val / (1.0f + exp2f(-val * 1.4426950408889634f));
    }
    for (int k = 0; k < d_conv; k++) state[state_base + k] = win[k];
}

// Conv1d burnin forward NOSAVE variant (target network — no backward needed).
// Identical to conv1d_burnin_forward but skips conv_states_out and post_conv_out writes.
// Saves ~50% memory bandwidth for target path.
extern "C" __global__ void conv1d_burnin_forward_nosave(
    float* __restrict__ u_out,          // [batch * T * d_inner] post-SiLU output
    float* __restrict__ state,          // [batch * d_inner * d_conv] persistent state (mutated)
    const float* __restrict__ x_branch, // [batch * T * d_inner] input from in_proj split
    const float* __restrict__ weight,   // [d_inner * d_conv]
    const float* __restrict__ bias,     // [d_inner]
    int batch, int T, int d_inner, int d_conv
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_inner;
    if (idx >= total) return;

    int b = idx / d_inner;
    int d = idx % d_inner;
    int state_base = (b * d_inner + d) * d_conv;

    // Serve-shape fast path (d_conv == 4, the production value): the shift
    // register and the taps live in REGISTERS across the whole T loop, and
    // global memory sees one x_branch read + one u_out write per t plus a
    // single state writeback at the end. Value contract: the accumulation
    // below is the k-ascending chain of the generic loop, term for term —
    // an f32 kept in a register holds the exact bits a global round-trip
    // would have returned, so the emitted FMA chain is unchanged.
    if (d_conv == 4) {
        float s0 = state[state_base];
        float s1 = state[state_base + 1];
        float s2 = state[state_base + 2];
        float s3 = state[state_base + 3];
        const float w0 = weight[d * 4];
        const float w1 = weight[d * 4 + 1];
        const float w2 = weight[d * 4 + 2];
        const float w3 = weight[d * 4 + 3];
        const float bd = bias[d];
        for (int t = 0; t < T; t++) {
            int bt_di = (b * T + t) * d_inner + d;
            s0 = s1;
            s1 = s2;
            s2 = s3;
            s3 = x_branch[bt_di];
            float val = bd;
            val += s0 * w0;
            val += s1 * w1;
            val += s2 * w2;
            val += s3 * w3;
            u_out[bt_di] = val / (1.0f + exp2f(-val * 1.4426950408889634f));
        }
        state[state_base] = s0;
        state[state_base + 1] = s1;
        state[state_base + 2] = s2;
        state[state_base + 3] = s3;
        return;
    }

    for (int t = 0; t < T; t++) {
        int bt_di = (b * T + t) * d_inner + d;

        // Shift register left + insert new value
        for (int k = 0; k < d_conv - 1; k++) {
            state[state_base + k] = state[state_base + k + 1];
        }
        state[state_base + d_conv - 1] = x_branch[bt_di];

        // Depthwise dot product
        float val = bias[d];
        for (int k = 0; k < d_conv; k++) {
            val += state[state_base + k] * weight[d * d_conv + k];
        }

        // Fused SiLU: u = val * sigmoid(val)
        u_out[bt_di] = val / (1.0f + exp2f(-val * 1.4426950408889634f));
    }
}

// Templated conv1d burnin nosave — sequence forward for prefill / target net.
// Activations in T_IN, state/weight/bias stay f32.
#define DEFINE_CONV1D_BURNIN_NOSAVE(SUFFIX, T, FROM_F)                      \
extern "C" __global__ void conv1d_burnin_forward_nosave_##SUFFIX(           \
    T* u_out,                                                               \
    float* state,                                                           \
    const T* x_branch,                                                      \
    const float* weight,                                                    \
    const float* bias,                                                      \
    int batch, int T_len, int d_inner, int d_conv                           \
) {                                                                         \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                        \
    int total = batch * d_inner;                                            \
    if (idx >= total) return;                                               \
    int b = idx / d_inner;                                                  \
    int d = idx % d_inner;                                                  \
    int state_base = (b * d_inner + d) * d_conv;                            \
    for (int t = 0; t < T_len; t++) {                                       \
        int bt_di = (b * T_len + t) * d_inner + d;                          \
        for (int k = 0; k < d_conv - 1; k++) {                              \
            state[state_base + k] = state[state_base + k + 1];              \
        }                                                                   \
        state[state_base + d_conv - 1] = to_f(x_branch[bt_di]);             \
        float val = bias[d];                                                \
        for (int k = 0; k < d_conv; k++) {                                  \
            val += state[state_base + k] * weight[d * d_conv + k];          \
        }                                                                   \
        u_out[bt_di] = FROM_F(val / (1.0f + exp2f(-val * 1.4426950408889634f))); \
    }                                                                       \
}

DEFINE_CONV1D_BURNIN_NOSAVE(f32,  float,         from_f_f32)
DEFINE_CONV1D_BURNIN_NOSAVE(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_CONV1D_BURNIN_NOSAVE(f16,  __half,        from_f_f16)

// Templated conv1d burnin WITH saves — used for training/backward.
// u_out + post_conv + x_branch in T_IN, states/weights in f32.
// IMPORTANT: save order must match f32 `conv1d_burnin_forward` exactly —
// save conv_states AFTER shift+insert so that `conv_states[t]` reflects
// the state WINDOW that was consumed at time t for the dot product (i.e.
// `[x_branch[t-d_conv+1], ..., x_branch[t]]`). Backward `d_weight[k] +=
// conv_states[t][k] * d_conv_out[t]` is only correct with this ordering.
// Saving BEFORE shift+insert produces an off-by-one window and flips the
// sign of `d_conv_weight` (caught by the parity test vs the f32 oracle).
#define DEFINE_CONV1D_BURNIN(SUFFIX, TY, FROM_F)                             \
extern "C" __global__ void conv1d_burnin_forward_##SUFFIX(                   \
    TY* u_out, float* state, float* conv_states_saved, TY* post_conv,        \
    const TY* x_branch, const float* weight, const float* bias,              \
    int batch, int T_len, int d_inner, int d_conv                            \
) {                                                                          \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                         \
    int total = batch * d_inner;                                             \
    if (idx >= total) return;                                                \
    int b = idx / d_inner;                                                   \
    int d = idx % d_inner;                                                   \
    int state_base = (b * d_inner + d) * d_conv;                             \
    /* Register window: the sliding state used to round-trip through     \
     * GLOBAL memory ~7 times per timestep (a 1300-deep dependent RMW    \
     * chain on a 24-block grid). Same shifts, same values - the tape    \
     * saves and every output are bit-identical. */                      \
    float win[8];                                                        \
    if (d_conv > 8) return;                                              \
    for (int k = 0; k < d_conv; k++) win[k] = state[state_base + k];     \
    for (int t = 0; t < T_len; t++) {                                        \
        int bt_di = (b * T_len + t) * d_inner + d;                           \
        for (int k = 0; k < d_conv - 1; k++) {                               \
            win[k] = win[k + 1];                                             \
        }                                                                    \
        win[d_conv - 1] = to_f(x_branch[bt_di]);                             \
        for (int k = 0; k < d_conv; k++) {                                   \
            int save_idx = ((b * T_len + t) * d_inner + d) * d_conv + k;     \
            conv_states_saved[save_idx] = win[k];                            \
        }                                                                    \
        float val = bias[d];                                                 \
        for (int k = 0; k < d_conv; k++) {                                   \
            val += win[k] * weight[d * d_conv + k];                          \
        }                                                                    \
        post_conv[bt_di] = FROM_F(val);                                      \
        u_out[bt_di] = FROM_F(val / (1.0f + exp2f(-val * 1.4426950408889634f))); \
    }                                                                        \
    for (int k = 0; k < d_conv; k++) state[state_base + k] = win[k];         \
}

DEFINE_CONV1D_BURNIN(f32,  float,         from_f_f32)
DEFINE_CONV1D_BURNIN(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_CONV1D_BURNIN(f16,  __half,        from_f_f16)

// Conv1d burnin backward (T>1, reverse): process T steps in reverse for each (b,d) thread.
// Includes BUG-M2 carry fix: gradient propagates through shift register positions.
// Fused with SiLU backward.
//
// Rule B (no atomicAdd): each thread accumulates its local d_weight/d_bias
// contributions across T timesteps into registers, then writes a single
// per-(b,d) partial at the end. Caller MUST follow with two reduce_sum_axis0
// launches to reduce across batch deterministically:
//   reduce_sum_axis0(d_weight, d_weight_partials, batch, d_inner*d_conv, 1)
//   reduce_sum_axis0(d_bias,   d_bias_partials,   batch, d_inner,         1)
//
// Source: CPU reference: train/forward.rs phase B6
extern "C" __global__ __launch_bounds__(256, 4)
void conv1d_burnin_backward(
    float* __restrict__ d_x_branch,       // [batch * T * d_inner] output gradient
    float* __restrict__ d_weight_partials, // [batch * d_inner * d_conv] OUTPUT (no atomicAdd)
    float* __restrict__ d_bias_partials,   // [batch * d_inner] OUTPUT (no atomicAdd)
    const float* __restrict__ d_u,         // [batch * T * d_inner]
    const float* __restrict__ post_conv,   // [batch * T * d_inner] saved pre-SiLU
    const float* __restrict__ conv_states, // [batch * T * d_inner * d_conv]
    const float* __restrict__ weight,      // [d_inner * d_conv]
    int batch, int T, int d_inner, int d_conv
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_inner;
    if (idx >= total) return;

    int b = idx / d_inner;
    int d = idx % d_inner;

    // Per-thread carry for BUG-M2 fix (d_conv-1 elements).
    // CONSTRAINT: d_conv <= 8. Validated in Rust launch code.
    float carry[8];
    if (d_conv > 8) return; // safety guard
    for (int k = 0; k < d_conv - 1; k++) carry[k] = 0.0f;

    // Local T-accumulators for Rule B (avoid T atomicAdds per thread).
    float local_d_weight[8];
    for (int k = 0; k < d_conv; k++) local_d_weight[k] = 0.0f;
    float local_d_bias = 0.0f;

    for (int t = T - 1; t >= 0; t--) {
        int bt_di = (b * T + t) * d_inner + d;

        // SiLU backward: d_conv_out = d_u * silu_grad(post_conv)
        float x = post_conv[bt_di];
        float sig = 1.0f / (1.0f + exp2f(-x * 1.4426950408889634f));
        float silu_grad = sig * (1.0f + x * (1.0f - sig));
        float d_conv_out = d_u[bt_di] * silu_grad;

        // Conv1d backward for this timestep
        // d_x_branch[b,t,d] = weight[d, d_conv-1] * d_conv_out
        d_x_branch[bt_di] = d_conv_out * weight[d * d_conv + d_conv - 1];

        // Accumulate into local d_weight/d_bias (no atomic)
        int cs_base = ((b * T + t) * d_inner + d) * d_conv;
        for (int k = 0; k < d_conv; k++) {
            local_d_weight[k] += d_conv_out * conv_states[cs_base + k];
        }
        local_d_bias += d_conv_out;

        // BUG-M2 carry fix: propagate gradient through shift register
        if (d_conv > 1) {
            int carry_len = d_conv - 1;
            // Add carry from future timesteps
            d_x_branch[bt_di] += carry[0];
            // Shift carry left
            for (int k = 0; k < carry_len - 1; k++) {
                carry[k] = carry[k + 1] + d_conv_out * weight[d * d_conv + d_conv - 2 - k];
            }
            // Last carry position
            carry[carry_len - 1] = d_conv_out * weight[d * d_conv];
        }
    }

    // Rule B stage-1 output: single per-(b,d) write after T-loop.
    int wp_base = (b * d_inner + d) * d_conv;
    for (int k = 0; k < d_conv; k++) {
        d_weight_partials[wp_base + k] = local_d_weight[k];
    }
    d_bias_partials[b * d_inner + d] = local_d_bias;
}

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
// (b, d) with a T-long serial loop — 24 blocks at the campaign shape, 146
// idle SMs. The window only reaches d_conv-1 = 3 steps back, so the t-range
// tiles perfectly: tile 0 seeds from the carry-in state, tiles k>0 seed
// their window from x_branch halo loads — the SAME values the serial walk
// would hold, so every output element is bit-identical. Only the tile
// containing t = T-1 writes the carry-out state.
// ============================================================================
#define CONV1D_TILE_T 128

#define DEFINE_CONV1D_BURNIN_TILED(SUFFIX, TY, FROM_F)                        \
extern "C" __global__ void conv1d_burnin_forward_tiled_##SUFFIX(              \
    TY* u_out, float* state, float* conv_states_saved, TY* post_conv,        \
    const TY* x_branch, const float* weight, const float* bias,              \
    int batch, int T_len, int d_inner, int d_conv                            \
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
            /* LEG-4: the ONLY conv tape is the carry-in window — the      \
             * backward reconstructs every later window from x_branch. */   \
            conv_states_saved[state_base + k] = win[k];                      \
        }                                                                    \
    } else {                                                                 \
        /* Halo: the serial walk's window after step t0-1 holds           \
         * win[k] = x[t0 - d_conv + k] (win[0] is about to shift out).   \
         * t0 >= CONV1D_TILE_T > d_conv, so the halo never underruns. */ \
        for (int k = 0; k < d_conv; k++) {                                   \
            int th = t0 - d_conv + k;                                        \
            win[k] = to_f(x_branch[(b * T_len + th) * d_inner + d]);         \
        }                                                                    \
    }                                                                        \
    for (int t = t0; t < t_end; t++) {                                       \
        int bt_di = (b * T_len + t) * d_inner + d;                           \
        for (int k = 0; k < d_conv - 1; k++) win[k] = win[k + 1];            \
        win[d_conv - 1] = to_f(x_branch[bt_di]);                             \
        float val = bias[d];                                                 \
        for (int k = 0; k < d_conv; k++) {                                   \
            val += win[k] * weight[d * d_conv + k];                          \
        }                                                                    \
        post_conv[bt_di] = FROM_F(val);                                      \
        u_out[bt_di] = FROM_F(val / (1.0f + exp2f(-val * 1.4426950408889634f))); \
    }                                                                        \
    if (t_end == T_len) {                                                    \
        for (int k = 0; k < d_conv; k++) state[state_base + k] = win[k];     \
    }                                                                        \
}

DEFINE_CONV1D_BURNIN_TILED(f32,  float,         from_f_f32)
DEFINE_CONV1D_BURNIN_TILED(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_CONV1D_BURNIN_TILED(f16,  __half,        from_f_f16)

// Tiled d_x half of the conv backward. d_x[t] is a 4-tap anticausal FIR
// of d_conv_out; the reverse carry walk within a tile is seeded with the
// EXACT partial sums the serial walk would hold at the tile boundary
// (same association order), so every d_x element is bit-identical.
// d_weight/d_bias live in the separate dw-only pass below (their
// descending-t accumulation order is the numeric contract and must not
// be tiled).
#define DEFINE_CONV1D_BWD_DX_TILED(SUFFIX, TY, FROM_F)                        \
extern "C" __global__ void conv1d_bwd_dx_tiled_##SUFFIX(                      \
    TY* __restrict__ d_x_branch,                                              \
    const TY* __restrict__ d_u,                                               \
    const TY* __restrict__ post_conv,                                         \
    const float* __restrict__ weight,                                         \
    int batch, int T_, int d_inner, int d_conv                                \
) {                                                                           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    int total = batch * d_inner;                                              \
    if (idx >= total) return;                                                 \
    if (d_conv > 8) return;                                                   \
    int b = idx / d_inner;                                                    \
    int d = idx % d_inner;                                                    \
    int t0 = blockIdx.y * CONV1D_TILE_T;                                      \
    if (t0 >= T_) return;                                                     \
    int t_end = min(t0 + CONV1D_TILE_T, T_);                                  \
    float carry[8];                                                           \
    int carry_len = d_conv - 1;                                               \
    for (int k = 0; k < carry_len; k++) carry[k] = 0.0f;                      \
    /* Seed carries as if the serial reverse walk had processed           \
     * t >= t_end: replay steps t_end+carry_len-1 .. t_end (descending)  \
     * through the same carry recurrence, with dco past T = 0.           */ \
    for (int ts = t_end + carry_len - 1; ts >= t_end; ts--) {                 \
        float dco = 0.0f;                                                     \
        if (ts < T_) {                                                        \
            int bt_di = (b * T_ + ts) * d_inner + d;                          \
            float x = to_f(post_conv[bt_di]);                                 \
            float sig = 1.0f / (1.0f + exp2f(-x * 1.4426950408889634f));      \
            float silu_grad = sig * (1.0f + x * (1.0f - sig));                \
            dco = to_f(d_u[bt_di]) * silu_grad;                               \
        }                                                                     \
        if (d_conv > 1) {                                                     \
            for (int k = 0; k < carry_len - 1; k++) {                         \
                carry[k] = carry[k + 1]                                       \
                         + dco * weight[d * d_conv + d_conv - 2 - k];         \
            }                                                                 \
            carry[carry_len - 1] = dco * weight[d * d_conv];                  \
        }                                                                     \
    }                                                                         \
    for (int t = t_end - 1; t >= t0; t--) {                                   \
        int bt_di = (b * T_ + t) * d_inner + d;                                \
        float x = to_f(post_conv[bt_di]);                                      \
        float sig = 1.0f / (1.0f + exp2f(-x * 1.4426950408889634f));           \
        float silu_grad = sig * (1.0f + x * (1.0f - sig));                     \
        float d_conv_out = to_f(d_u[bt_di]) * silu_grad;                       \
        float dxb = d_conv_out * weight[d * d_conv + d_conv - 1];              \
        if (d_conv > 1) {                                                      \
            dxb += carry[0];                                                   \
            for (int k = 0; k < carry_len - 1; k++) {                          \
                carry[k] = carry[k + 1]                                        \
                         + d_conv_out * weight[d * d_conv + d_conv - 2 - k];   \
            }                                                                  \
            carry[carry_len - 1] = d_conv_out * weight[d * d_conv];            \
        }                                                                      \
        d_x_branch[bt_di] = FROM_F(dxb);                                       \
    }                                                                          \
}

DEFINE_CONV1D_BWD_DX_TILED(f32,  float,         from_f_f32)
DEFINE_CONV1D_BWD_DX_TILED(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_CONV1D_BWD_DX_TILED(f16,  __half,        from_f_f16)

// dw/db-only half: the historical descending-t accumulation, verbatim,
// minus the d_x work (its carries fed only d_x). Rule-B contract with
// the same reduce_sum_axis0 follow-up is unchanged.
#define DEFINE_CONV1D_BWD_DW_ONLY(SUFFIX, TY)                                 \
extern "C" __global__ void conv1d_bwd_dw_only_##SUFFIX(                       \
    float* __restrict__ d_weight_partials,                                     \
    float* __restrict__ d_bias_partials,                                       \
    const TY* __restrict__ d_u,                                                \
    const TY* __restrict__ post_conv,                                          \
    const TY* __restrict__ x_branch,                                           \
    const float* __restrict__ conv_init, /* [B*di*d_conv] carry-in window */   \
    int batch, int T_, int d_inner, int d_conv                                 \
) {                                                                            \
    /* Tap-split: thread role = (b, d, tap) with tap == d_conv meaning the  \
     * bias lane. Each tap's accumulator was ALREADY independent in the     \
     * fused kernel, and every lane keeps the same descending-t add order,  \
     * so all sums are bit-identical - this is pure lane redistribution     \
     * (24 -> ~120 blocks at the campaign shape).                        */ \
    int lanes = d_conv + 1;                                                    \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int total = batch * d_inner * lanes;                                       \
    if (idx >= total) return;                                                  \
    if (d_conv > 8) return;                                                    \
    int tap = idx % lanes;                                                     \
    int bd = idx / lanes;                                                      \
    int b = bd / d_inner;                                                      \
    int d = bd % d_inner;                                                      \
    int init_base = (b * d_inner + d) * d_conv;                                \
    if (tap == d_conv) {                                                       \
        /* bias lane: descending-t sum of d_conv_out, order unchanged */       \
        float local_d_bias = 0.0f;                                             \
        for (int t = T_ - 1; t >= 0; t--) {                                    \
            int bt_di = (b * T_ + t) * d_inner + d;                            \
            float x = to_f(post_conv[bt_di]);                                  \
            float sig = 1.0f / (1.0f + exp2f(-x * 1.4426950408889634f));       \
            float silu_grad = sig * (1.0f + x * (1.0f - sig));                 \
            /* __fmul_rn/__fadd_rn pin the fused kernel's two-rounding       \
             * shape: there d_conv_out materialized (multi-use) before the  \
             * bias add; an inlined product contracts to one FFMA and       \
             * moves every digest. */                                        \
            local_d_bias = __fadd_rn(                                          \
                local_d_bias, __fmul_rn(to_f(d_u[bt_di]), silu_grad));         \
        }                                                                      \
        d_bias_partials[b * d_inner + d] = local_d_bias;                       \
        return;                                                                \
    }                                                                          \
    /* weight-tap lane: window element `tap` at time t is                    \
     * x[t - (d_conv-1) + tap], carry-in fallback at the left edge. */        \
    float local_dw = 0.0f;                                                     \
    for (int t = T_ - 1; t >= 0; t--) {                                        \
        int bt_di = (b * T_ + t) * d_inner + d;                                \
        float x = to_f(post_conv[bt_di]);                                      \
        float sig = 1.0f / (1.0f + exp2f(-x * 1.4426950408889634f));           \
        float silu_grad = sig * (1.0f + x * (1.0f - sig));                     \
        float d_conv_out = to_f(d_u[bt_di]) * silu_grad;                       \
        int tx = t - (d_conv - 1) + tap;                                       \
        float wv = (tx >= 0)                                                   \
            ? to_f(x_branch[(b * T_ + tx) * d_inner + d])                      \
            : conv_init[init_base + tap + t + 1];                              \
        local_dw += d_conv_out * wv;                                           \
    }                                                                          \
    d_weight_partials[init_base + tap] = local_dw;                             \
}

DEFINE_CONV1D_BWD_DW_ONLY(f32,  float)
DEFINE_CONV1D_BWD_DW_ONLY(bf16, __nv_bfloat16)
DEFINE_CONV1D_BWD_DW_ONLY(f16,  __half)
