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

// Conv1d step backward:
//   d_new_x[b,d] = weight[d, d_conv-1] * dy[b,d]
//   d_weight[d,k] += state_saved[b,d,k] * dy[b,d]  (accumulated across batch)
//   d_bias[d] += dy[b,d]  (accumulated across batch)
extern "C" __global__ void conv1d_step_backward(
    float* d_new_x,     // [batch * d_inner]
    float* d_weight,     // [d_inner * d_conv] accumulated
    float* d_bias,       // [d_inner] accumulated
    const float* dy,     // [batch * d_inner]
    const float* state_saved, // [batch * d_inner * d_conv]
    const float* weight, // [d_inner * d_conv]
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

    // d_weight: accumulated across batch
    int state_base = (b * d_inner + d) * d_conv;
    for (int k = 0; k < d_conv; k++) {
        atomicAdd(&d_weight[d * d_conv + k], dy_val * state_saved[state_base + k]);
    }

    // d_bias: accumulated across batch
    atomicAdd(&d_bias[d], dy_val);
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

    for (int t = 0; t < T; t++) {
        int bt_di = (b * T + t) * d_inner + d;

        // Shift register left + insert new value
        for (int k = 0; k < d_conv - 1; k++) {
            state[state_base + k] = state[state_base + k + 1];
        }
        state[state_base + d_conv - 1] = x_branch[bt_di];

        // Save conv_state after shift (for backward)
        int cs_base = ((b * T + t) * d_inner + d) * d_conv;
        for (int k = 0; k < d_conv; k++) {
            conv_states_out[cs_base + k] = state[state_base + k];
        }

        // Depthwise dot product
        float val = bias[d];
        for (int k = 0; k < d_conv; k++) {
            val += state[state_base + k] * weight[d * d_conv + k];
        }
        post_conv_out[bt_di] = val;

        // Fused SiLU: u = val * sigmoid(val)
        u_out[bt_di] = val / (1.0f + exp2f(-val * 1.4426950408889634f));
    }
}

// Conv1d burnin forward NOSAVE variant (target network — no backward needed).
// Identical to conv1d_burnin_forward but skips conv_states_out and post_conv_out writes.
// Saves ~50% memory bandwidth for target path.
extern "C" __global__ void conv1d_burnin_forward_nosave(
    float* u_out,          // [batch * T * d_inner] post-SiLU output
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

// Conv1d burnin backward (T>1, reverse): process T steps in reverse for each (b,d) thread.
// Includes BUG-M2 carry fix: gradient propagates through shift register positions.
// Fused with SiLU backward.
//
// Source: CPU reference: train/forward.rs phase B6
extern "C" __global__ void conv1d_burnin_backward(
    float* d_x_branch,    // [batch * T * d_inner] output gradient
    float* d_weight,       // [d_inner * d_conv] accumulated (atomicAdd across batch*T)
    float* d_bias,         // [d_inner] accumulated (atomicAdd across batch*T)
    const float* d_u,      // [batch * T * d_inner] incoming gradient (after x_proj bwd accumulation)
    const float* post_conv,// [batch * T * d_inner] saved pre-SiLU values
    const float* conv_states, // [batch * T * d_inner * d_conv] saved states
    const float* weight,   // [d_inner * d_conv]
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

        // d_weight[d,k] += conv_states[b,t,d,k] * d_conv_out
        int cs_base = ((b * T + t) * d_inner + d) * d_conv;
        for (int k = 0; k < d_conv; k++) {
            atomicAdd(&d_weight[d * d_conv + k], d_conv_out * conv_states[cs_base + k]);
        }

        // d_bias[d] += d_conv_out
        atomicAdd(&d_bias[d], d_conv_out);

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
}
