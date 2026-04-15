// Mamba-3 SISO chunked parallel scan CUDA kernels.
//
// Replaces sequential T-loop with parallel intra-chunk matmul for Mamba-3
// trapezoidal recurrence. Key difference from Mamba-2 chunked (mamba2_ssd.cu):
// trapezoidal discretization with shifted_gamma/scale K-prescaling.
//
// Chunked forward reformulation:
//   shifted_gamma[t] = DT[t+1] * (1 - sigmoid(trap[t+1]))   // beta of NEXT step
//   gamma[t] = DT[t] * sigmoid(trap[t])                       // gamma of CURRENT step
//   scale[t] = shifted_gamma[t] + gamma[t]                    // combined weight on K[t]
//   K_scaled[t] = K[t] * scale[t]
//   qk_dot[t] = sum(Q[t] * K[t]) * gamma[t]                  // for D skip
//
// Then: Y = Q @ K_scaled^T * causal_decay @ V + state_contrib + (D + qk_dot) * V
//
// Source: CPU reference: native/train/mamba2/forward_m3.rs (chunked path)
//         Triton reference: docs_rs/mamba2/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py
//         Triton backward:  docs_rs/mamba2/mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py
// Paper: Lahoti et al. "Mamba-3: SISO" (ICLR 2026)
//
// Thread/grid conventions follow mamba2_ssd.cu and mamba3_ssd.cu.
// All kernels handle partial last chunk (T not multiple of chunk_size).

#include "_typed_prelude.cuh"

#ifndef FAST_EXP
#define FAST_EXP(x) exp2f((x) * LOG2E)
#endif

// ============================================================================
// 1. m3_preprocess_chunks -- Compute shifted_gamma, scale, qk_dot, K prescaling
// ============================================================================
//
// For each timestep t:
//   gamma[t] = DT[t] * trap_sig[t]
//   shifted_gamma[t] = DT[t+1] * (1 - trap_sig[t+1])    (0 for last timestep)
//   scale[t] = shifted_gamma[t] + gamma[t]
//   qk_dot[t] = sum_n(Q[t,h,n] * K[t,h,n]) * gamma[t]
//   K_scaled[t,h,n] = K[t,h,n] * scale[t]
//
// Input:  K[B*T*nh*ds], Q[B*T*nh*ds], DT[B*T*nh], trap_sig[B*T*nh], D[nh]
// Output: K_scaled[B*T*nh*ds], qk_dot[B*T*nh], scale_out[B*T*nh], gamma_out[B*T*nh]
//
// Grid: (B*n_chunks, nh, 1), Block: (chunk_size, 1, 1)
// Each thread handles one timestep within a chunk, looping over ds for
// the QK dot product and K scaling.
extern "C" __global__ void m3_preprocess_chunks(
    // Outputs
    float* __restrict__ K_scaled,   // [B * T * nh * ds] -- prescaled K
    float* __restrict__ qk_dot,     // [B * T * nh]
    float* __restrict__ scale_out,  // [B * T * nh]
    float* __restrict__ gamma_out,  // [B * T * nh]
    // Inputs
    const float* __restrict__ K,        // [B * T * nh * ds] (post bias+RoPE)
    const float* __restrict__ Q,        // [B * T * nh * ds] (post bias+RoPE)
    const float* __restrict__ DT,       // [B * T * nh] (post-softplus)
    const float* __restrict__ trap_sig, // [B * T * nh] (post-sigmoid)
    int batch, int T, int nh, int ds, int chunk_size
) {
    int n_chunks = (T + chunk_size - 1) / chunk_size;
    int bc = blockIdx.x;         // batch * n_chunks
    int b = bc / n_chunks;
    int chunk = bc % n_chunks;
    int h = blockIdx.y;
    int t_local = threadIdx.x;   // position within chunk

    int chunk_start = chunk * chunk_size;
    int t = chunk_start + t_local;
    if (t >= T) return;

    int th = (b * T + t) * nh + h;     // index into [B*T*nh] arrays

    // gamma[t] = DT[t] * trap_sig[t]
    float dt_cur = DT[th];
    float trap_cur = trap_sig[th];
    float gamma_val = dt_cur * trap_cur;

    // shifted_gamma[t] = DT[t+1] * (1 - trap_sig[t+1]), 0 for last timestep
    float shifted_gamma = 0.0f;
    if (t + 1 < T) {
        int th_next = (b * T + t + 1) * nh + h;
        float dt_next = DT[th_next];
        float trap_next = trap_sig[th_next];
        shifted_gamma = dt_next * (1.0f - trap_next);
    }

    float scale_val = shifted_gamma + gamma_val;

    // Store scale and gamma
    scale_out[th] = scale_val;
    gamma_out[th] = gamma_val;

    // Compute qk_dot = sum_n(Q[t,h,n] * K[t,h,n]) * gamma
    int kq_base = (b * T + t) * nh * ds + h * ds;
    float dot = 0.0f;
    for (int n = 0; n < ds; n++) {
        dot += Q[kq_base + n] * K[kq_base + n];
    }
    qk_dot[th] = dot * gamma_val;

    // K_scaled[t,h,n] = K[t,h,n] * scale
    for (int n = 0; n < ds; n++) {
        K_scaled[kq_base + n] = K[kq_base + n] * scale_val;
    }
}

// ============================================================================
// 2. m3_dA_cumsum -- Inclusive prefix sum of ADT within chunks
// ============================================================================
//
// Computes cumulative sum of adt = A * DT within each chunk, per head.
// Same as Mamba-2 pattern but with Mamba-3's input-dependent ADT.
//
// Input:  adt[B * T * nh]  (A * DT, already computed by m3_split)
// Output: dA_cumsum[B * n_chunks * nh * chunk_size]
//
// Grid: (B, n_chunks, ceil(nh / blockDim.x)), Block: (min(nh, 256))
extern "C" __global__ void m3_dA_cumsum(
    float* __restrict__ dA_cumsum_out,  // [B * n_chunks * nh * chunk_size]
    const float* __restrict__ adt,      // [B * T * nh]
    int batch, int T, int nh, int chunk_size
) {
    int b = blockIdx.x;
    int chunk = blockIdx.y;
    int h = blockIdx.z * blockDim.x + threadIdx.x;
    if (h >= nh) return;

    int n_chunks = gridDim.y;
    int chunk_start = chunk * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > T) chunk_end = T;

    float cumsum = 0.0f;
    int cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;

    for (int t = chunk_start; t < chunk_end; t++) {
        float a_dt = adt[(b * T + t) * nh + h];
        cumsum += a_dt;
        dA_cumsum_out[cs_base + (t - chunk_start)] = cumsum;
    }
}

// ============================================================================
// 3. m3_chunk_state_fwd -- Per-chunk state contribution (trapezoidal K_scaled)
// ============================================================================
//
// Computes per-chunk SSM state using prescaled K (already has trapezoidal scale):
//   chunk_states[c,h,p,n] = sum_t(exp(dA_end - dA_t) * K_scaled[t,h,n] * V[t,h,p])
//
// where V[t,h,p] = x[t, h*hd+p] and K_scaled already includes the scale factor.
//
// Input:  x[B*T*d_inner], K_scaled[B*T*nh*ds], dA_cumsum[B*n_chunks*nh*chunk_size]
// Output: states[B * n_chunks * nh * hd * ds]
//
// Grid: (B * n_chunks, nh, 1), Block: (hd, 1, 1)
// Each thread handles one (b, chunk, h, p), loops over ds and timesteps.
extern "C" __global__ void m3_chunk_state_fwd(
    float* __restrict__ states_out,     // [B * n_chunks * nh * hd * ds]
    const float* __restrict__ x,        // [B * T * d_inner]
    const float* __restrict__ K_scaled, // [B * T * nh * ds]
    const float* __restrict__ dA_cumsum,// [B * n_chunks * nh * chunk_size]
    int batch, int T, int nh, int hd, int ds, int chunk_size
) {
    int d_inner = nh * hd;
    int n_chunks = (T + chunk_size - 1) / chunk_size;
    int bc = blockIdx.x;  // batch * n_chunks
    int b = bc / n_chunks;
    int chunk = bc % n_chunks;
    int h = blockIdx.y;
    int p = threadIdx.x;
    if (p >= hd) return;

    int chunk_start = chunk * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > T) chunk_end = T;
    int chunk_len = chunk_end - chunk_start;

    // dA at end of chunk
    int cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;
    float dA_end = dA_cumsum[cs_base + chunk_len - 1];

    // Output base for this (b, chunk, h, p)
    int state_base = ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds;

    for (int n = 0; n < ds; n++) {
        float acc = 0.0f;
        for (int t = chunk_start; t < chunk_end; t++) {
            int t_local = t - chunk_start;
            float dA_t = dA_cumsum[cs_base + t_local];
            float decay = FAST_EXP(dA_end - dA_t);

            float v_t = x[(b * T + t) * d_inner + h * hd + p];
            float ks_t = K_scaled[(b * T + t) * nh * ds + h * ds + n];

            acc += decay * ks_t * v_t;
        }
        states_out[state_base + n] = acc;
    }
}

// ============================================================================
// 4. m3_state_passing_fwd -- Inter-chunk exclusive prefix scan
// ============================================================================
//
// Sequential scan over n_chunks: out[c] = state entering chunk c.
// Forward recurrence: new_state = exp(dA_chunk_end) * prev_state + chunk_contribution
// Converts states in-place from chunk contributions to prefix-scanned entering states.
// Also computes initial state contribution from k_state/v_state if applicable.
//
// Handles partial last chunk (T not multiple of chunk_size).
//
// Input/Output: states[B * n_chunks * nh * hd * ds] (in-place)
// Output:       final_states[B * nh * hd * ds]
// Input:        dA_cumsum[B * n_chunks * nh * chunk_size]
//
// Grid: (B, nh, ceil(hd*ds / BLOCK)), Block: (min(hd*ds, 256))
extern "C" __global__ void m3_state_passing_fwd(
    float* __restrict__ states,          // [B * n_chunks * nh * hd * ds] in/out
    float* __restrict__ final_states,    // [B * nh * hd * ds] output
    const float* __restrict__ dA_cumsum, // [B * n_chunks * nh * chunk_size]
    int batch, int n_chunks, int nh, int hd, int ds, int chunk_size, int T
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int pd = blockIdx.z * blockDim.x + threadIdx.x;  // flattened (p, n) index
    int dim = hd * ds;
    if (pd >= dim) return;

    // Exclusive prefix scan: states[c] = state ENTERING chunk c
    float state = 0.0f;

    for (int c = 0; c < n_chunks; c++) {
        int state_idx = (b * n_chunks + c) * nh * dim + h * dim + pd;
        float new_contribution = states[state_idx];

        // Write prev_state (state ENTERING this chunk) BEFORE updating
        states[state_idx] = state;

        // dA at end of this chunk (handle partial last chunk)
        int cs_idx = (b * n_chunks + c) * nh + h;
        int chunk_start_c = c * chunk_size;
        int chunk_end_c = chunk_start_c + chunk_size;
        if (chunk_end_c > T) chunk_end_c = T;
        int last_elem = chunk_end_c - chunk_start_c - 1;
        float dA_end = dA_cumsum[cs_idx * chunk_size + last_elem];
        float decay = FAST_EXP(dA_end);

        // Update: state after this chunk = decay * prev_state + chunk_contribution
        state = decay * state + new_contribution;
    }

    // Store final state (state after all chunks)
    int final_idx = b * nh * dim + h * dim + pd;
    final_states[final_idx] = state;
}

// ============================================================================
// 4b. m3_writeback_parallel_states -- Persist SSM/K/V state after parallel scan
// ============================================================================
//
// The parallel chunked scan path does not update the persistent layer state
// buffers (ssm_state, k_state, v_state) unlike the sequential m3_burnin_fwd
// kernel. This kernel writes back the final states so that state continuity
// between forward calls is maintained.
//
// SSM state: copy from final_states[B * nh * hd * ds] (output of K4)
// K state:   extract last timestep (T-1) from k_flat[B * T * nh * ds]
// V state:   extract last timestep (T-1) from x_flat[B * T * d_inner]
//
// Grid: (B, nh, 1), Block: (max(hd, ds), 1, 1)
// Each thread handles one (p) index; loops over the other dimension.
extern "C" __global__ void m3_writeback_parallel_states(
    float* __restrict__ ssm_state,          // [B * nh * hd * ds] out
    float* __restrict__ k_state,            // [B * nh * ds] out
    float* __restrict__ v_state,            // [B * nh * hd] out
    const float* __restrict__ final_states, // [B * nh * hd * ds] in
    const float* __restrict__ k_flat,       // [B * T * nh * ds] in
    const float* __restrict__ x_flat,       // [B * T * d_inner] in
    int batch, int T, int nh, int hd, int ds
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int p = threadIdx.x;
    if (b >= batch || h >= nh) return;

    int d_inner = nh * hd;
    int dim = hd * ds;

    // SSM state: copy final_states -> ssm_state (both [B * nh * hd * ds])
    // Thread p loops over the dimension it doesn't cover
    if (p < hd) {
        for (int n = 0; n < ds; n++) {
            int idx = b * nh * dim + h * dim + p * ds + n;
            ssm_state[idx] = final_states[idx];
        }
    }

    // K state: last timestep from k_flat[B * T * nh * ds]
    // k_flat layout: [b * T * nh * ds + t * nh * ds + h * ds + n]
    if (p < ds) {
        int src_idx = (b * T + (T - 1)) * nh * ds + h * ds + p;
        int dst_idx = b * nh * ds + h * ds + p;
        k_state[dst_idx] = k_flat[src_idx];
    }

    // V state: last timestep from x_flat[B * T * d_inner]
    // x_flat layout: [b * T * d_inner + t * d_inner + h * hd + p]
    if (p < hd) {
        int src_idx = (b * T + (T - 1)) * d_inner + h * hd + p;
        int dst_idx = b * nh * hd + h * hd + p;
        v_state[dst_idx] = x_flat[src_idx];
    }
}

// ============================================================================
// 5. m3_chunk_scan_fwd -- Intra-chunk output computation
// ============================================================================
//
// Computes output for each timestep within a chunk:
//   Y[t,h,p] = sum_n(Q[t,h,n] * state[h,p,n]) * exp(dA[t])        // state contribution
//            + sum_{s<=t}(exp(dA[t]-dA[s]) * QK_scaled_dot * V[s])  // intra-chunk
//            + (D[h] + qk_dot[t,h]) * V[t,h,p]                      // D + qk skip
//
// where QK_scaled_dot = Q[t]^T @ K_scaled[s] (dot product of Q at t with K_scaled at s).
// Q and K_scaled are in the d_state dimension (not headdim).
//
// Grid: (B * n_chunks, nh, 1), Block: (hd, 1, 1)
// Each thread computes output for one (b, chunk, h, p) across all timesteps.
extern "C" __global__ void m3_chunk_scan_fwd(
    float* __restrict__ y_out,          // [B * T * d_inner]
    const float* __restrict__ x,        // [B * T * d_inner]
    const float* __restrict__ Q,        // [B * T * nh * ds] (post bias+RoPE)
    const float* __restrict__ K_scaled, // [B * T * nh * ds] (prescaled by scale)
    const float* __restrict__ qk_dot,   // [B * T * nh] (precomputed D-skip term)
    const float* __restrict__ dA_cumsum,// [B * n_chunks * nh * chunk_size]
    const float* __restrict__ prev_states, // [B * n_chunks * nh * hd * ds] (entering states)
    const float* __restrict__ D,        // [nh]
    int batch, int T, int nh, int hd, int ds, int chunk_size
) {
    int d_inner = nh * hd;
    int n_chunks = (T + chunk_size - 1) / chunk_size;
    int bc = blockIdx.x;
    int b = bc / n_chunks;
    int chunk = bc % n_chunks;
    int h = blockIdx.y;
    int p = threadIdx.x;
    if (p >= hd) return;

    int chunk_start = chunk * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > T) chunk_end = T;
    int chunk_len = chunk_end - chunk_start;

    // dA_cumsum base for this (b, chunk, h)
    int cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;

    // Prev state for this (b, chunk, h, p)
    int state_base = ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds;

    float d_skip = D[h];

    for (int t_local = 0; t_local < chunk_len; t_local++) {
        int t = chunk_start + t_local;
        float dA_t = dA_cumsum[cs_base + t_local];

        // Y_off: state contribution = sum_n(Q[t,n] * prev_state[p,n]) * exp(dA[t])
        float y_off = 0.0f;
        float state_decay = FAST_EXP(dA_t);
        int q_base = (b * T + t) * nh * ds + h * ds;
        for (int n = 0; n < ds; n++) {
            y_off += Q[q_base + n] * prev_states[state_base + n];
        }
        y_off *= state_decay;

        // Y_diag: intra-chunk contribution from positions s <= t (strictly causal + diagonal)
        // For s < t: exp(dA[t] - dA[s]) * Q[t] dot K_scaled[s] * V[s]
        // For s == t: qk_dot handles the diagonal (gamma contribution)
        float y_diag = 0.0f;
        for (int s_local = 0; s_local < t_local; s_local++) {
            int s = chunk_start + s_local;
            float dA_s = dA_cumsum[cs_base + s_local];
            float decay = FAST_EXP(dA_t - dA_s);

            // Q[t] dot K_scaled[s] (in d_state dimension)
            float qk_val = 0.0f;
            int ks_base = (b * T + s) * nh * ds + h * ds;
            for (int n = 0; n < ds; n++) {
                qk_val += Q[q_base + n] * K_scaled[ks_base + n];
            }

            float v_s = x[(b * T + s) * d_inner + h * hd + p];
            y_diag += decay * qk_val * v_s;
        }

        // D + qk_dot skip: (D[h] + qk_dot[t,h]) * V[t,h,p]
        float x_t = x[(b * T + t) * d_inner + h * hd + p];
        int th = (b * T + t) * nh + h;
        float y_skip = (d_skip + qk_dot[th]) * x_t;

        y_out[(b * T + t) * d_inner + h * hd + p] = y_diag + y_off + y_skip;
    }
}

// ============================================================================
// 6. m3_chunk_scan_bwd -- Chunked backward
// ============================================================================
//
// Backward of intra-chunk parallel scan.
// Computes: d_x, d_Q (via atomicAdd on ds dim), d_K_scaled (via atomicAdd on ds dim),
//           d_qk_dot, d_D, d_prev_states, d_dA_cumsum
//
// Forward was:
//   Y[t,h,p] = sum_{s<t}(exp(dA[t]-dA[s]) * QK_s_dot * V[s,h,p])   // intra-chunk
//            + sum_n(Q[t,n]*state[p,n]) * exp(dA[t])                  // state
//            + (D[h] + qk_dot[t,h]) * V[t,h,p]                       // skip
//
// Grid: (B * n_chunks, nh, 1), Block: (hd, 1, 1)
// Each thread handles backward for one (b, chunk, h, p) across all timesteps.
extern "C" __global__ void m3_chunk_scan_bwd(
    // Output gradients
    float* __restrict__ d_x,            // [B * T * d_inner]
    float* __restrict__ d_Q,            // [B * T * nh * ds] (atomicAdd)
    float* __restrict__ d_K_scaled,     // [B * T * nh * ds] (atomicAdd)
    float* __restrict__ d_qk_dot,       // [B * T * nh] (atomicAdd across p)
    float* __restrict__ d_D,            // [nh] (atomicAdd)
    float* __restrict__ d_prev_states,  // [B * n_chunks * nh * hd * ds]
    float* __restrict__ d_dA_cumsum,    // [B * n_chunks * nh * chunk_size] (atomicAdd)
    // Inputs (saved from forward)
    const float* __restrict__ d_y,      // [B * T * d_inner] upstream gradient
    const float* __restrict__ x,        // [B * T * d_inner]
    const float* __restrict__ Q,        // [B * T * nh * ds]
    const float* __restrict__ K_scaled, // [B * T * nh * ds]
    const float* __restrict__ qk_dot_in,// [B * T * nh]
    const float* __restrict__ dA_cumsum,// [B * n_chunks * nh * chunk_size]
    const float* __restrict__ prev_states, // [B * n_chunks * nh * hd * ds]
    const float* __restrict__ D,        // [nh]
    int batch, int T, int nh, int hd, int ds, int chunk_size
) {
    int d_inner = nh * hd;
    int n_chunks = (T + chunk_size - 1) / chunk_size;
    int bc = blockIdx.x;
    int b = bc / n_chunks;
    int chunk = bc % n_chunks;
    int h = blockIdx.y;
    int p = threadIdx.x;
    if (p >= hd) return;

    int chunk_start = chunk * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > T) chunk_end = T;
    int chunk_len = chunk_end - chunk_start;

    int cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;
    int state_base = ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds;

    // Warp-reduce mask: only `hd` lanes are launched (block_dim = hd, hd ≤ 32 by
    // config). Hardcoded 0xFFFFFFFF would name lanes that never executed the
    // intrinsic — documented UB per CUDA Programming Guide §B.15.1.
    unsigned warp_mask = (hd >= 32) ? 0xFFFFFFFFu : ((1u << hd) - 1u);

    float d_skip = D[h];
    float d_D_acc = 0.0f;

    // --- d_prev_states ---
    // d_prev_states[h,p,n] = sum_t(d_y[t,h,p] * Q[t,n] * exp(dA_cumsum[t]))
    for (int n = 0; n < ds; n++) {
        float d_state_n = 0.0f;
        for (int t_local = 0; t_local < chunk_len; t_local++) {
            int t = chunk_start + t_local;
            float dy = d_y[(b * T + t) * d_inner + h * hd + p];
            float q_n = Q[(b * T + t) * nh * ds + h * ds + n];
            float dA_t = dA_cumsum[cs_base + t_local];
            float state_decay = FAST_EXP(dA_t);
            d_state_n += dy * q_n * state_decay;
        }
        d_prev_states[state_base + n] = d_state_n;
    }

    // --- Per-timestep gradients ---
    for (int t_local = 0; t_local < chunk_len; t_local++) {
        int t = chunk_start + t_local;
        int t_idx = b * T + t;
        float dy = d_y[t_idx * d_inner + h * hd + p];
        float dA_t = dA_cumsum[cs_base + t_local];
        float x_t = x[t_idx * d_inner + h * hd + p];
        int th = t_idx * nh + h;

        // d_D: from (D + qk_dot) * V
        d_D_acc += dy * x_t;

        // d_x from D+qk_dot skip: d_x[t] += dy * (D + qk_dot[t])
        float d_x_val = dy * (d_skip + qk_dot_in[th]);

        // d_qk_dot[t,h] = sum_p(dy[t,h,p] * V[t,h,p])
        // Warp reduce over p, atomicAdd
        float d_qk_val = dy * x_t;
        for (int off = hd / 2; off > 0; off >>= 1)
            d_qk_val += __shfl_down_sync(warp_mask, d_qk_val, off, hd);
        if (p == 0)
            atomicAdd(&d_qk_dot[th], d_qk_val);

        // --- Intra-chunk: this timestep t acts as source s for future positions ---
        // d_x[s] += sum_{r>s} d_y[r] * exp(dA[r]-dA[s]) * Q[r] dot K_scaled[s] (scalar for this p)
        // Also computes d_Q and d_K_scaled contributions.
        int q_base_t = t_idx * nh * ds + h * ds;

        // (A) Contributions where this t is the OUTPUT position (receiving from s < t)
        // Already handled: d_x_val above from skip path.
        // d_Q[t,n] from intra-chunk: sum_{s<t} exp(dA[t]-dA[s]) * K_scaled[s,n] * V[s,h,p] * dy[t,h,p]
        // d_Q[t,n] from state:       sum_n(prev_state[p,n]) * exp(dA[t]) * dy[t,h,p]
        // Both require warp reduce over p -> atomicAdd to d_Q

        // State contribution to d_Q
        float state_decay = FAST_EXP(dA_t);
        for (int n = 0; n < ds; n++) {
            float d_q_state = dy * prev_states[state_base + n] * state_decay;
            for (int off = hd / 2; off > 0; off >>= 1)
                d_q_state += __shfl_down_sync(warp_mask, d_q_state, off, hd);
            if (p == 0)
                atomicAdd(&d_Q[q_base_t + n], d_q_state);
        }

        // (B) Iterate over s <= t for intra-chunk contributions
        for (int s_local = 0; s_local < t_local; s_local++) {
            int s = chunk_start + s_local;
            float dA_s = dA_cumsum[cs_base + s_local];
            float decay = FAST_EXP(dA_t - dA_s);

            // QK_scaled dot product for this (t, s) pair
            int ks_base = (b * T + s) * nh * ds + h * ds;
            float qk_val = 0.0f;
            for (int n = 0; n < ds; n++) {
                qk_val += Q[q_base_t + n] * K_scaled[ks_base + n];
            }

            float v_s = x[(b * T + s) * d_inner + h * hd + p];
            float contrib = dy * decay * v_s;

            // d_Q[t,n] += dy * decay * V[s] * K_scaled[s,n] (warp reduce over p)
            for (int n = 0; n < ds; n++) {
                float d_q_val = contrib * K_scaled[ks_base + n];
                for (int off = hd / 2; off > 0; off >>= 1)
                    d_q_val += __shfl_down_sync(warp_mask, d_q_val, off, hd);
                if (p == 0)
                    atomicAdd(&d_Q[q_base_t + n], d_q_val);
            }

            // d_K_scaled[s,n] += dy * decay * V[s] * Q[t,n] (warp reduce over p)
            for (int n = 0; n < ds; n++) {
                float d_ks_val = contrib * Q[q_base_t + n];
                for (int off = hd / 2; off > 0; off >>= 1)
                    d_ks_val += __shfl_down_sync(warp_mask, d_ks_val, off, hd);
                if (p == 0)
                    atomicAdd(&d_K_scaled[ks_base + n], d_ks_val);
            }

            // d_x[s] += dy[t] * decay * qk_val (for this thread's p)
            // We accumulate into s's position. Since multiple t contribute to same s,
            // we use atomicAdd for d_x[s].
            float d_x_s = dy * decay * qk_val;
            atomicAdd(&d_x[(b * T + s) * d_inner + h * hd + p], d_x_s);

            // d_dA_cumsum contributions: decay = exp(dA[t] - dA[s])
            //   d/d(dA[t]) = +val, d/d(dA[s]) = -val
            float dA_grad = dy * decay * qk_val * v_s;
            // Warp reduce dA_grad over p
            for (int off = hd / 2; off > 0; off >>= 1)
                dA_grad += __shfl_down_sync(warp_mask, dA_grad, off, hd);
            if (p == 0) {
                atomicAdd(&d_dA_cumsum[cs_base + t_local], dA_grad);
                atomicAdd(&d_dA_cumsum[cs_base + s_local], -dA_grad);
            }
        }

        // d_dA_cumsum from state contribution: exp(dA[t]) path
        {
            float y_off = 0.0f;
            for (int n = 0; n < ds; n++) {
                y_off += Q[(b * T + t) * nh * ds + h * ds + n] * prev_states[state_base + n];
            }
            float dA_state_grad = dy * y_off * state_decay;
            // Warp reduce over p
            for (int off = hd / 2; off > 0; off >>= 1)
                dA_state_grad += __shfl_down_sync(warp_mask, dA_state_grad, off, hd);
            if (p == 0)
                atomicAdd(&d_dA_cumsum[cs_base + t_local], dA_state_grad);
        }

        // d_x for this timestep from skip path
        d_x[t_idx * d_inner + h * hd + p] += d_x_val;
    }

    // d_D via atomicAdd (shared across batch, p)
    atomicAdd(&d_D[h], d_D_acc);
}

// ============================================================================
// 7. m3_state_passing_bwd -- Inter-chunk backward propagation
// ============================================================================
//
// Backward of m3_state_passing_fwd.
// Forward recurrence: out[c+1] = decay[c] * out[c] + chunk_contrib[c]
// Propagates d_prev_states backward, produces ddA_chunk.
//
// Output semantics (matching Triton _state_passing_bwd):
//   d_prev_states[nchunks-1] = 0
//   d_prev_states[c] = dL/d(out[c+1])_total for c < nchunks-1
//     where total = d_prev_states_orig[c+1] + decay[c+1] * dL/d(out[c+2])_total
//
// Grid: (B, nh, ceil(hd*ds / BLOCK)), Block: (min(hd*ds, 256))
extern "C" __global__ void m3_state_passing_bwd(
    float* __restrict__ d_prev_states,      // [B * n_chunks * nh * dim] in/out (overwritten)
    float* __restrict__ d_dA_cumsum,        // [B * n_chunks * nh * chunk_size] accumulate last elem
    const float* __restrict__ chunk_states, // [B * n_chunks * nh * dim] accumulated states from fwd
    const float* __restrict__ dA_cumsum,    // [B * n_chunks * nh * chunk_size]
    int batch, int n_chunks, int nh, int hd, int ds, int chunk_size, int T
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int pd = blockIdx.z * blockDim.x + threadIdx.x;  // flattened (p, n) index
    int dim = hd * ds;
    if (pd >= dim) return;

    // Initialize dstates = 0 (no dfinal_states gradient in our pipeline)
    float dstates = 0.0f;

    // Save original d_prev_states[nchunks-1] before overwriting with d_final=0
    int last_idx = (b * n_chunks + (n_chunks - 1)) * nh * dim + h * dim + pd;
    float dout_saved = d_prev_states[last_idx];
    d_prev_states[last_idx] = dstates;  // d_prev_states[nchunks-1] = 0

    // Reverse loop: c goes from nchunks-1 down to 1
    for (int c = n_chunks - 1; c >= 1; c--) {
        // dA at end of chunk c (the inter-chunk decay)
        int cs_idx = (b * n_chunks + c) * nh + h;
        int chunk_start_c = c * chunk_size;
        int chunk_end_c = chunk_start_c + chunk_size;
        if (chunk_end_c > T) chunk_end_c = T;
        int last_elem = chunk_end_c - chunk_start_c - 1;
        float dA_end = dA_cumsum[cs_idx * chunk_size + last_elem];
        float decay = FAST_EXP(dA_end);

        // Read accumulated state at chunk c (from forward state_passing output)
        int state_idx = (b * n_chunks + c) * nh * dim + h * dim + pd;
        float out_val = chunk_states[state_idx];

        // ddA = sum over pd of: out[c,pd] * dstates[pd] * decay
        float ddA_contrib = out_val * dstates * decay;
        atomicAdd(&d_dA_cumsum[cs_idx * chunk_size + last_elem], ddA_contrib);

        // Use saved original d_prev_states[c]
        float dout = dout_saved;

        // Propagate: dstates = decay * dstates + d_prev_states_orig[c]
        dstates = decay * dstates + dout;

        // Save original d_prev_states[c-1] before overwriting
        int prev_idx = (b * n_chunks + (c - 1)) * nh * dim + h * dim + pd;
        dout_saved = d_prev_states[prev_idx];

        // Store propagated gradient at chunk c-1
        d_prev_states[prev_idx] = dstates;
    }

    // Chunk 0: ddA = 0 (no initial states)
}

// ============================================================================
// 8. m3_chunk_state_bwd -- Additional gradients from propagated d_states
// ============================================================================
//
// Backward of m3_chunk_state_fwd, using propagated d_prev_states from state_passing_bwd.
//
// Forward: chunk_states[c,h,p,n] = sum_t(exp(dA_end - dA_t) * K_scaled[t,h,n] * V[t,h,p])
//
// This kernel computes ADDITIONAL gradient contributions (on top of m3_chunk_scan_bwd)
// from the inter-chunk state path. Outputs are atomicAdd'd into existing buffers.
//
// Grid: (B * n_chunks, nh, 1), Block: (hd, 1, 1)
// Each thread handles one (b, chunk, h, p), iterating over ds and timesteps.
extern "C" __global__ void m3_chunk_state_bwd(
    // Outputs (additional gradients, atomicAdd)
    float* __restrict__ d_x,            // [B * T * d_inner] accumulate
    float* __restrict__ d_K_scaled,     // [B * T * nh * ds] accumulate (atomicAdd)
    float* __restrict__ d_dA_cumsum,    // [B * n_chunks * nh * chunk_size] accumulate
    // Inputs
    const float* __restrict__ d_chunk_states, // [B * n_chunks * nh * hd * ds] (propagated)
    const float* __restrict__ x,              // [B * T * d_inner]
    const float* __restrict__ K_scaled,       // [B * T * nh * ds]
    const float* __restrict__ dA_cumsum,      // [B * n_chunks * nh * chunk_size]
    int batch, int T, int nh, int hd, int ds, int chunk_size
) {
    int d_inner = nh * hd;
    int n_chunks = (T + chunk_size - 1) / chunk_size;
    int bc = blockIdx.x;
    int b = bc / n_chunks;
    int chunk = bc % n_chunks;
    int h = blockIdx.y;
    int p = threadIdx.x;
    if (p >= hd) return;

    int chunk_start = chunk * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > T) chunk_end = T;
    int chunk_len = chunk_end - chunk_start;

    int cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;
    int state_base = ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds;

    // Warp-reduce mask: only `hd` lanes are launched (block_dim = hd, hd ≤ 32).
    // Hardcoded 0xFFFFFFFF = UB per CUDA Programming Guide §B.15.1.
    unsigned warp_mask = (hd >= 32) ? 0xFFFFFFFFu : ((1u << hd) - 1u);

    // dA at end of this chunk
    float dA_end = dA_cumsum[cs_base + chunk_len - 1];

    for (int t_local = 0; t_local < chunk_len; t_local++) {
        int t = chunk_start + t_local;
        int t_idx = b * T + t;

        float dA_t = dA_cumsum[cs_base + t_local];
        float decay = FAST_EXP(dA_end - dA_t);
        float x_val = x[t_idx * d_inner + h * hd + p];

        float d_x_add = 0.0f;

        for (int n = 0; n < ds; n++) {
            float dcs = d_chunk_states[state_base + n];
            float ks_n = K_scaled[t_idx * nh * ds + h * ds + n];
            float common = dcs * decay;

            // d_x += d_chunk_states * decay * K_scaled
            d_x_add += common * ks_n;

            // d_K_scaled += d_chunk_states * decay * x (atomicAdd across p)
            float d_ks_val = common * x_val;
            for (int off = hd / 2; off > 0; off >>= 1)
                d_ks_val += __shfl_down_sync(warp_mask, d_ks_val, off, hd);
            if (p == 0)
                atomicAdd(&d_K_scaled[t_idx * nh * ds + h * ds + n], d_ks_val);

            // d_dA_cumsum from exp(dA_end - dA_t):
            //   d/d(dA_end) = +val, d/d(dA_t) = -val
            float dA_grad = common * ks_n * x_val;
            // Warp reduce dA_grad over p
            for (int off_r = hd / 2; off_r > 0; off_r >>= 1)
                dA_grad += __shfl_down_sync(warp_mask, dA_grad, off_r, hd);
            if (p == 0 && t_local != chunk_len - 1) {
                atomicAdd(&d_dA_cumsum[cs_base + chunk_len - 1], dA_grad);
                atomicAdd(&d_dA_cumsum[cs_base + t_local], -dA_grad);
            }
        }

        // d_x accumulate (unique per (b,t,h,p) -- add to existing from chunk_scan_bwd)
        d_x[t_idx * d_inner + h * hd + p] += d_x_add;
    }
}

// ============================================================================
// 9. m3_cumsum_bwd -- Reverse cumsum for d_dA → d_adt
// ============================================================================
//
// Converts d_dA_cumsum → d_dA (reverse cumsum per chunk).
// d_dA_cumsum is the accumulated gradient of loss w.r.t. the cumulative sum.
// Reverse cumsum gives gradient w.r.t. each individual dA[t] = adt[t].
//
// Then chain-rules through adt = A * DT to produce d_adt
// (caller decomposes d_adt into d_A and d_dt via separate kernel or fused op).
//
// Input:  d_dA_cumsum[B * n_chunks * nh * chunk_size]
// Output: d_adt[B * T * nh] (accumulated via atomicAdd for overlapping contributions)
//
// Grid: (B, n_chunks, ceil(nh / blockDim.x)), Block: (min(nh, 256))
extern "C" __global__ void m3_cumsum_bwd(
    float* __restrict__ d_adt,              // [B * T * nh] output
    const float* __restrict__ d_dA_cumsum,  // [B * n_chunks * nh * chunk_size]
    int batch, int T, int nh, int chunk_size
) {
    int b = blockIdx.x;
    int chunk = blockIdx.y;
    int h = blockIdx.z * blockDim.x + threadIdx.x;
    if (h >= nh) return;

    int n_chunks = gridDim.y;
    int chunk_start = chunk * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > T) chunk_end = T;
    int chunk_len = chunk_end - chunk_start;
    int cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;

    // Reverse cumsum: d_dA[t] = sum_{s>=t}(d_dA_cumsum[s]) within this chunk
    float rev_sum = 0.0f;
    for (int t_local = chunk_len - 1; t_local >= 0; t_local--) {
        rev_sum += d_dA_cumsum[cs_base + t_local];

        int t = chunk_start + t_local;
        int idx = (b * T + t) * nh + h;
        d_adt[idx] = rev_sum;
    }
}

// ============================================================================
// 10. m3_extract_da_cs_sum -- Extract chunk-end cumsum values
// ============================================================================
// Grid: (B, n_chunks, ceil(nh/256)), Block: (min(nh, 256))
extern "C" __global__ void m3_extract_da_cs_sum(
    float* __restrict__ da_cs_sum,       // [B * n_chunks * nh]
    const float* __restrict__ da_cumsum, // [B * n_chunks * nh * CS]
    int B, int T, int nh, int CS
) {
    int b = blockIdx.x;
    int chunk = blockIdx.y;
    int h = blockIdx.z * blockDim.x + threadIdx.x;
    if (h >= nh) return;
    int n_chunks = (T + CS - 1) / CS;
    int chunk_start = chunk * CS;
    int last_t = min(CS, T - chunk_start) - 1;
    int cs_idx = ((b * n_chunks + chunk) * nh + h) * CS + last_t;
    int out_idx = (b * n_chunks + chunk) * nh + h;
    da_cs_sum[out_idx] = da_cumsum[cs_idx];
}

// ============================================================================
// 11. m3_dqkv -- Monolithic chunked backward (Python Step 2)
// ============================================================================
//
// Translates Python mamba3_siso_bwd_kernel_dqkv.
// Grid: (nh, B), Block: (hd, 1, 1)
// Each block processes ALL chunks in reverse for one (head, batch).
//
// Thread p owns column p of d_ssm_states_acc[hd][ds] (ds register floats).
// Shared memory holds Q, K, V, dO tiles per chunk.
//
// Outputs: dQ_mid, dK_mid, dV, dADT, dQK_dot, dD
extern "C" __global__ void m3_dqkv(
    // Outputs
    float* __restrict__ dQ_mid,       // [B*T*nh*ds]
    float* __restrict__ dK_mid,       // [B*T*nh*ds]
    float* __restrict__ dV,           // [B*T*d_inner]
    float* __restrict__ dADT,         // [B*T*nh]
    float* __restrict__ dQK_dot_out,  // [B*T*nh]
    float* __restrict__ dD_out,       // [nh] (atomicAdd across B)
    // Inputs
    const float* __restrict__ Q_rot,       // [B*T*nh*ds]
    const float* __restrict__ K_scaled,    // [B*T*nh*ds]
    const float* __restrict__ V_in,        // [B*T*d_inner]
    const float* __restrict__ DA_CS,       // [B*n_chunks*nh*CS]
    const float* __restrict__ DA_CS_SUM,   // [B*n_chunks*nh]
    const float* __restrict__ QK_dot_in,   // [B*T*nh]
    const float* __restrict__ SSM_States,  // [B*n_chunks*nh*hd*ds]
    const float* __restrict__ dO,          // [B*T*d_inner]
    const float* __restrict__ D_param,     // [nh]
    int B, int T, int nh_total, int hd, int ds, int CS
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    int p = threadIdx.x;
    if (h >= nh_total || b >= B || p >= hd) return;
    if (ds > 64 || CS > 64) return;  // Safety: d_state[64], dM_rev[64] fixed arrays

    // Warp-reduce mask: only `hd` lanes are launched (block_dim = hd, hd ≤ 32).
    // Hardcoded 0xFFFFFFFF = UB per CUDA Programming Guide §B.15.1.
    unsigned warp_mask = (hd >= 32) ? 0xFFFFFFFFu : ((1u << hd) - 1u);

    int d_inner = nh_total * hd;
    int n_chunks = (T + CS - 1) / CS;
    float D_val = D_param[h];
    float dD_acc = 0.0f;

    // Register state: d_ssm_states_acc — column p of [hd][ds] matrix
    // Each thread holds ds floats
    float d_state[64]; // max ds
    for (int n = 0; n < ds; n++) d_state[n] = 0.0f;

    // Shared memory layout (dynamically sized)
    extern __shared__ float smem[];
    // q_sm[CS][ds], k_sm[CS][ds], v_sm[CS][hd], do_sm[CS][hd]
    // da_cs_sm[CS], qk_dot_sm[CS], ssm_sm[hd][ds] (loaded cooperatively)
    float* q_sm    = smem;
    float* k_sm    = q_sm + CS * ds;
    float* v_sm    = k_sm + CS * ds;
    float* do_sm   = v_sm + CS * hd;
    float* da_cs_sm = do_sm + CS * hd;
    float* qk_sm   = da_cs_sm + CS;
    float* ssm_sm  = qk_sm + CS; // [hd][ds] for SSM_States tile

    for (int chunk_loop = 0; chunk_loop < n_chunks; chunk_loop++) {
        int chunk_idx = n_chunks - 1 - chunk_loop;
        int chunk_start = chunk_idx * CS;
        int chunk_len = min(CS, T - chunk_start);

        // === Cooperative load tiles into shared memory ===
        // V and dO: thread p loads element p of each timestep
        for (int t = 0; t < chunk_len; t++) {
            int gt = chunk_start + t;
            v_sm[t * hd + p] = V_in[(b * T + gt) * d_inner + h * hd + p];
            do_sm[t * hd + p] = dO[(b * T + gt) * d_inner + h * hd + p];
        }
        // Zero padding
        for (int t = chunk_len; t < CS; t++) {
            v_sm[t * hd + p] = 0.0f;
            do_sm[t * hd + p] = 0.0f;
        }
        // Q and K: each thread loads ALL ds entries for indices n stride hd.
        // OLD bug: `if (p < ds)` only worked when ds <= hd; for ds > hd
        // (e.g. ds=16, hd=8) entries n=hd..ds-1 were left as garbage in
        // shared memory, corrupting every dot/dQK computation downstream.
        for (int n = p; n < ds; n += hd) {
            for (int t = 0; t < chunk_len; t++) {
                int gt = chunk_start + t;
                q_sm[t * ds + n] = Q_rot[((b * T + gt) * nh_total + h) * ds + n];
                k_sm[t * ds + n] = K_scaled[((b * T + gt) * nh_total + h) * ds + n];
            }
            for (int t = chunk_len; t < CS; t++) {
                q_sm[t * ds + n] = 0.0f;
                k_sm[t * ds + n] = 0.0f;
            }
        }
        // da_cs and qk_dot: one thread loads
        if (p == 0) {
            for (int t = 0; t < chunk_len; t++) {
                da_cs_sm[t] = DA_CS[((b * n_chunks + chunk_idx) * nh_total + h) * CS + t];
                qk_sm[t] = QK_dot_in[(b * T + chunk_start + t) * nh_total + h];
            }
            for (int t = chunk_len; t < CS; t++) {
                da_cs_sm[t] = 0.0f;
                qk_sm[t] = 0.0f;
            }
        }
        // SSM_States: each thread loads its row (p) of [hd][ds]
        for (int n = 0; n < ds; n++) {
            ssm_sm[p * ds + n] = SSM_States[((b * n_chunks + chunk_idx) * nh_total + h) * hd * ds + p * ds + n];
        }
        __syncthreads();

        float da_cs_chunk_sum = DA_CS_SUM[(b * n_chunks + chunk_idx) * nh_total + h];

        // === Compute per-timestep outputs ===
        // dV, dQK_dot, dD for each timestep
        for (int t = 0; t < chunk_len; t++) {
            int gt = chunk_start + t;
            float dA_t = da_cs_sm[t];
            float exp_rev_t = exp2f((da_cs_chunk_sum - dA_t) * LOG2E);
            float exp_fwd_t = exp2f(dA_t * LOG2E);

            // --- dV[t,h,p] ---
            // Intra: sum_{s: s has t as source, i.e., t<s in causal}
            // P^T[t,s] = sum_n(K[t,n]*Q[s,n]) * exp(dA[s]-dA[t]) for s > t
            float dv_intra = 0.0f;
            for (int s = t + 1; s < chunk_len; s++) {
                float kq = 0.0f;
                for (int n = 0; n < ds; n++)
                    kq += k_sm[t * ds + n] * q_sm[s * ds + n];
                float decay = exp2f((da_cs_sm[s] - dA_t) * LOG2E);
                dv_intra += kq * decay * do_sm[s * hd + p];
            }
            // Inter: K[t] @ d_state^T[p] * exp_rev
            float dv_inter = 0.0f;
            for (int n = 0; n < ds; n++)
                dv_inter += k_sm[t * ds + n] * d_state[n];
            dv_inter *= exp_rev_t;
            // Skip: dO * (D + qk_dot)
            float dv_skip = do_sm[t * hd + p] * (D_val + qk_sm[t]);

            dV[(b * T + gt) * d_inner + h * hd + p] = dv_intra + dv_inter + dv_skip;

            // --- dQK_dot[t,h] and dD ---
            // dQK_dot = sum_p(dO[t,p] * V[t,p])
            float dqk_val = do_sm[t * hd + p] * v_sm[t * hd + p];
            // Warp reduce
            for (int off = hd / 2; off > 0; off >>= 1)
                dqk_val += __shfl_down_sync(warp_mask, dqk_val, off, hd);
            if (p == 0) {
                dQK_dot_out[(b * T + gt) * nh_total + h] = dqk_val;
                dD_acc += dqk_val;
            }
        }
        __syncthreads();

        // === dK_mid and dQ_mid ===
        // These need sum over hd (reduction across threads).
        // dK_mid[t,n] = sum_p(intra_contrib[t,p,n] + inter_contrib[t,p,n])
        // Since each thread has its own p, we compute per-thread then warp reduce.
        if (p < ds) {
            // Use p as n-index (thread p computes d_state dim n=p)
            int n = p;
            for (int t = 0; t < chunk_len; t++) {
                int gt = chunk_start + t;
                float dA_t = da_cs_sm[t];

                // dK_mid[t,n]: intra = sum_{s>t} dO[s]@V[s]^T * exp * Q[s,n]
                // This is dp_t^T[*,t] @ Q[*,n]
                // dp_t[i,j] = sum_p(V[i,p]*dO[j,p]) * mask[i,j]
                // dp_t^T[j,i] = dp_t[i,j]
                // (dp_t^T @ Q)[t, n] = sum_s(dp_t[s,t] * Q[s,n])
                //                    = sum_s(sum_p(V[s,p]*dO[t,p]) * mask[s,t] * Q[s,n])
                // Wait — mask is causal: mask[i,j] = exp(dA[j]-dA[i]) if j>i else 0
                // dp_t[i,j] includes mask. For column t of dp_t: dp_t[s,t] for s where t > s
                // Actually in Python: dp_t = V @ dO^T * mask
                // dp_t[s1, s2] = sum_p(V[s1,p]*dO[s2,p]) * (s2>s1 ? exp(dA[s2]-dA[s1]) : 0)
                // (dp_t @ Q)[s1, n] = sum_s2>s1 (sum_p(V[s1,p]*dO[s2,p]) * exp(dA[s2]-dA[s1]) * Q[s2,n])
                // We want acc_dk[t, n] = (dp_t @ Q)[t, n] — this is dp_t row t dot Q col n
                // But dp_t = V@dO^T ⊙ mask, so dp_t[t, s] = sum_p(V[t,p]*dO[s,p]) * mask[t,s]
                // mask[t,s] = exp(dA[s]-dA[t]) if s>t else 0
                // So acc_dk[t,n] = sum_{s>t} sum_p(V[t,p]*dO[s,p]) * exp(dA[s]-dA[t]) * Q[s,n]

                float dk_intra = 0.0f;
                for (int s = t + 1; s < chunk_len; s++) {
                    float vdo = 0.0f;
                    for (int pp = 0; pp < hd; pp++)
                        vdo += v_sm[t * hd + pp] * do_sm[s * hd + pp];
                    float decay = exp2f((da_cs_sm[s] - dA_t) * LOG2E);
                    dk_intra += vdo * decay * q_sm[s * ds + n];
                }
                // Inter-chunk dK contribution added in second pass below (after d_state → shared memory)
                dK_mid[((b * T + gt) * nh_total + h) * ds + n] = dk_intra;

                // dQ_mid[t,n]: similar structure
                // acc_dq[t,n] = sum_{s<t} sum_p(V[s,p]*dO[t,p]) * exp(dA[t]-dA[s]) * K[s,n]  (wrong?)
                // Actually from Python: s_block^T @ K where s_block = V@dO^T ⊙ mask (same as dp_t)
                // acc_dq = (dp_t^T @ K)[t, n]
                // dp_t^T[t, s] = dp_t[s, t] = sum_p(V[s,p]*dO[t,p]) * mask[s,t]
                // mask[s,t] = exp(dA[t]-dA[s]) if t>s else 0
                // So: acc_dq[t,n] = sum_{s<t} sum_p(V[s,p]*dO[t,p]) * exp(dA[t]-dA[s]) * K[s,n]
                float dq_intra = 0.0f;
                for (int s = 0; s < t; s++) {
                    float vdo = 0.0f;
                    for (int pp = 0; pp < hd; pp++)
                        vdo += v_sm[s * hd + pp] * do_sm[t * hd + pp];
                    float decay = exp2f((dA_t - da_cs_sm[s]) * LOG2E);
                    dq_intra += vdo * decay * k_sm[s * ds + n];
                }
                // Inter: dO[t] @ ssm_states * exp
                float dq_inter = 0.0f;
                for (int pp = 0; pp < hd; pp++)
                    dq_inter += do_sm[t * hd + pp] * ssm_sm[pp * ds + n];
                dq_inter *= exp2f(dA_t * LOG2E);

                dQ_mid[((b * T + gt) * nh_total + h) * ds + n] = dq_intra + dq_inter;
            }
        }
        __syncthreads();

        // === Store d_state to shared for inter-chunk dK_mid contribution ===
        // Each thread writes its d_state[ds] to ssm_sm[p*ds + n]
        for (int n = 0; n < ds; n++)
            ssm_sm[p * ds + n] = d_state[n];
        __syncthreads();

        // Now fix dK_mid inter-chunk: sum_p(V[t,p] * d_state[p][n]) * exp_rev
        if (p < ds) {
            int n = p;
            for (int t = 0; t < chunk_len; t++) {
                float dk_inter = 0.0f;
                float exp_rev_t = exp2f((da_cs_chunk_sum - da_cs_sm[t]) * LOG2E);
                for (int pp = 0; pp < hd; pp++)
                    dk_inter += v_sm[t * hd + pp] * ssm_sm[pp * ds + n];
                dk_inter *= exp_rev_t;
                int gt = chunk_start + t;
                dK_mid[((b * T + gt) * nh_total + h) * ds + n] += dk_inter;
            }
        }
        __syncthreads();

        // === dADT computation (all 4 parts) ===
        // This is complex — computed on thread 0 using shared memory data
        if (p == 0) {
            float dM_rev[64]; // CS max (matches reference chunk_size=64)
            for (int t = 0; t < CS; t++) dM_rev[t] = 0.0f;

            // Part 1: from intra-chunk attention
            // dAinv[i][j] = sum_p(V[i,p]*dO[j,p]) * mask[i,j] * sum_n(K[i,n]*Q[j,n])
            for (int i = 0; i < chunk_len; i++) {
                for (int j = i + 1; j < chunk_len; j++) {
                    float vdo = 0.0f;
                    for (int pp = 0; pp < hd; pp++)
                        vdo += v_sm[i * hd + pp] * do_sm[j * hd + pp];
                    float decay = exp2f((da_cs_sm[j] - da_cs_sm[i]) * LOG2E);
                    float kq = 0.0f;
                    for (int n = 0; n < ds; n++)
                        kq += k_sm[i * ds + n] * q_sm[j * ds + n];
                    float dAinv = vdo * decay * kq;
                    dM_rev[j] += dAinv; // rowsum contribution (j is row in transposed)
                    dM_rev[i] -= dAinv; // colsum contribution
                }
            }

            // Part 2: Q @ ssm_states^T dot dO * exp
            // NOTE: ssm_sm now holds d_state (overwritten at line 1031).
            // Must reload SSM_States from global memory for this part.
            for (int t = 0; t < chunk_len; t++) {
                float qs_do = 0.0f;
                for (int pp = 0; pp < hd; pp++) {
                    float qs = 0.0f;
                    for (int n = 0; n < ds; n++) {
                        float ssm_val = SSM_States[((b * n_chunks + chunk_idx) * nh_total + h) * hd * ds + pp * ds + n];
                        qs += q_sm[t * ds + n] * ssm_val;
                    }
                    qs_do += qs * do_sm[t * hd + pp];
                }
                dM_rev[t] += qs_do * exp2f(da_cs_sm[t] * LOG2E);
            }

            // Part 3: sum(SSM_States * d_state) * exp(cs_sum)
            // ssm_sm holds d_state; reload SSM_States from global memory.
            float dM_scalar = 0.0f;
            for (int pp = 0; pp < hd; pp++) {
                for (int n = 0; n < ds; n++) {
                    float ssm_val = SSM_States[((b * n_chunks + chunk_idx) * nh_total + h) * hd * ds + pp * ds + n];
                    // d_state is in ssm_sm[pp*ds+n] (we stored it there)
                    dM_scalar += ssm_val * ssm_sm[pp * ds + n];
                }
            }
            dM_scalar *= exp2f(da_cs_chunk_sum * LOG2E);

            // Part 4: K @ d_state^T dot V * exp_rev
            float dM_vector[64];
            for (int t = 0; t < chunk_len; t++) {
                float dsk_v = 0.0f;
                for (int pp = 0; pp < hd; pp++) {
                    float dsk = 0.0f;
                    for (int n = 0; n < ds; n++)
                        dsk += k_sm[t * ds + n] * ssm_sm[pp * ds + n]; // d_state[pp][n]
                    dsk_v += dsk * v_sm[t * hd + pp];
                }
                dM_vector[t] = dsk_v * exp2f((da_cs_chunk_sum - da_cs_sm[t]) * LOG2E);
            }

            // Reverse cumsum combine (Python line 590)
            float total_rev = 0.0f;
            for (int t = 0; t < chunk_len; t++) total_rev += dM_rev[t];
            total_rev += dM_scalar;
            float cumsum = 0.0f;
            for (int t = 0; t < chunk_len; t++) {
                cumsum += dM_vector[t] - dM_rev[t];
                dM_rev[t] += total_rev + cumsum - dM_vector[t];
            }

            // Store dADT
            for (int t = 0; t < chunk_len; t++) {
                int gt = chunk_start + t;
                dADT[(b * T + gt) * nh_total + h] = dM_rev[t];
            }
        }
        __syncthreads();

        // === Update d_ssm_states_acc ===
        // d_state[p][n] = exp(cs_sum) * d_state[p][n] + sum_t(dO[t,p]*exp(dA[t]) * Q[t,n])
        for (int n = 0; n < ds; n++) {
            float new_val = exp2f(da_cs_chunk_sum * LOG2E) * d_state[n];
            for (int t = 0; t < chunk_len; t++) {
                new_val += do_sm[t * hd + p] * exp2f(da_cs_sm[t] * LOG2E) * q_sm[t * ds + n];
            }
            d_state[n] = new_val;
        }
        __syncthreads();
    }

    // Store dD
    if (p == 0) atomicAdd(&dD_out[h], dD_acc);
}

// ============================================================================
// 12. m3_dqktheta -- Inverse RoPE + scale + bias gradients (Python Step 3)
// ============================================================================
// Grid: (B*n_chunks, nh), Block: (CS, 1, 1)
// Each thread handles one timestep within a chunk.
extern "C" __global__ void m3_dqktheta(
    // Outputs
    float* __restrict__ dQ_pre,         // [B*T*nh*ds] — pre-RoPE Q gradient
    float* __restrict__ dK_pre,         // [B*T*nh*ds] — pre-RoPE K gradient
    float* __restrict__ dAngles_cumsum, // [B*T*nh*n_angles]
    float* __restrict__ dScale,         // [B*T*nh]
    float* __restrict__ dGamma,         // [B*T*nh]
    float* __restrict__ dQ_bias,        // [nh*ds] (atomicAdd)
    float* __restrict__ dK_bias,        // [nh*ds] (atomicAdd)
    // Inputs
    const float* __restrict__ Q_raw,    // [B*T*nh*ds] — acts.c_biased (pre-RoPE)
    const float* __restrict__ K_raw,    // [B*T*nh*ds] — acts.b_biased (pre-RoPE)
    const float* __restrict__ Scale_in, // [B*T*nh] — recomputed
    const float* __restrict__ Gamma_in, // [B*T*nh] — recomputed
    const float* __restrict__ Angles,   // [B*T*nh*n_angles] — acts.angle_cumsum
    const float* __restrict__ dQ_mid,   // [B*T*nh*ds] — from m3_dqkv
    const float* __restrict__ dK_mid,   // [B*T*nh*ds] — from m3_dqkv
    const float* __restrict__ dQK_dot,  // [B*T*nh] — from m3_dqkv
    int B, int T, int nh, int ds, int n_angles, int CS
) {
    int n_chunks = (T + CS - 1) / CS;
    int bc = blockIdx.x;
    int b_chunk = bc; // b * n_chunks + chunk
    int h = blockIdx.y;
    int t_local = threadIdx.x;

    int b = b_chunk / n_chunks;
    int chunk = b_chunk % n_chunks;
    int gt = chunk * CS + t_local;

    // ds > 64 is a pre-existing kernel limit (q_pre/k_pre/dq_in[64] register
    // arrays). All threads of the block read the same `ds` arg → safe early
    // return (no smem alloc reached, no syncthreads needed).
    if (ds > 64) return;
    if (b >= B || h >= nh) return;

    bool valid = (gt < T);

    // Per-block smem accumulator for dQ_bias / dK_bias to remove HBM-atomic
    // contention (audit Agent 2 M2: previously CS threads serialized on the
    // same h*ds+n slot per block; now one HBM atomic per slot per block).
    __shared__ float dq_bias_smem[64];
    __shared__ float dk_bias_smem[64];
    for (int i = t_local; i < ds; i += blockDim.x) {
        dq_bias_smem[i] = 0.0f;
        dk_bias_smem[i] = 0.0f;
    }
    __syncthreads();

    if (valid) {
        int base = ((b * T + gt) * nh + h) * ds;
        float scale = Scale_in[(b * T + gt) * nh + h];
        float gamma = Gamma_in[(b * T + gt) * nh + h];
        float dqk = dQK_dot[(b * T + gt) * nh + h];

        // Load Q_raw + K_raw (pre-RoPE, post-bias)
        float q_pre[64], k_pre[64]; // max ds
        for (int n = 0; n < ds; n++) {
            q_pre[n] = Q_raw[base + n];
            k_pre[n] = K_raw[base + n];
        }
        float dq_in[64], dk_in[64];
        for (int n = 0; n < ds; n++) {
            dq_in[n] = dQ_mid[base + n];
            dk_in[n] = dK_mid[base + n];
        }

        // Forward RoPE on K_raw to get K_rot (for dScale computation)
        float k_rot[64];
        int angle_base = ((b * T + gt) * nh + h) * n_angles;
        for (int a = 0; a < n_angles && 2 * a + 1 < ds; a++) {
            float theta = Angles[angle_base + a];
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            int i0 = 2 * a, i1 = 2 * a + 1;
            k_rot[i0] = k_pre[i0] * cos_t - k_pre[i1] * sin_t;
            k_rot[i1] = k_pre[i0] * sin_t + k_pre[i1] * cos_t;
        }
        for (int n = 2 * n_angles; n < ds; n++) k_rot[n] = k_pre[n];

        // dScale = sum_n(dK_mid[n] * K_rot[n])
        float d_scale = 0.0f;
        for (int n = 0; n < ds; n++) d_scale += dk_in[n] * k_rot[n];
        dScale[(b * T + gt) * nh + h] = d_scale;

        // dGamma = dQK * sum_n(Q_raw[n] * K_raw[n])
        float qk_raw = 0.0f;
        for (int n = 0; n < ds; n++) qk_raw += q_pre[n] * k_pre[n];
        dGamma[(b * T + gt) * nh + h] = dqk * qk_raw;

        // Scale dK_mid by scale before inverse RoPE
        for (int n = 0; n < ds; n++) dk_in[n] *= scale;

        // Inverse RoPE on dQ_mid and scaled dK_mid
        float dq_pre_out[64], dk_pre_out[64];
        for (int n = 0; n < ds; n++) { dq_pre_out[n] = dq_in[n]; dk_pre_out[n] = dk_in[n]; }
        for (int a = 0; a < n_angles && 2 * a + 1 < ds; a++) {
            float theta = Angles[angle_base + a];
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            int i0 = 2 * a, i1 = 2 * a + 1;
            // Inverse rotation: R^T = [[cos, sin], [-sin, cos]]
            dq_pre_out[i0] = dq_in[i0] * cos_t + dq_in[i1] * sin_t;
            dq_pre_out[i1] = -dq_in[i0] * sin_t + dq_in[i1] * cos_t;
            dk_pre_out[i0] = dk_in[i0] * cos_t + dk_in[i1] * sin_t;
            dk_pre_out[i1] = -dk_in[i0] * sin_t + dk_in[i1] * cos_t;
        }

        // Add dQK path: dQ_pre += dqk * gamma * K_raw, dK_pre += dqk * gamma * Q_raw
        float dqk_gamma = dqk * gamma;
        for (int n = 0; n < ds; n++) {
            dq_pre_out[n] += dqk_gamma * k_pre[n];
            dk_pre_out[n] += dqk_gamma * q_pre[n];
        }

        // Store dQ_pre, dK_pre
        for (int n = 0; n < ds; n++) {
            dQ_pre[base + n] = dq_pre_out[n];
            dK_pre[base + n] = dk_pre_out[n];
        }

        // dQ_bias, dK_bias: accumulate into smem first (block-level atomic
        // is ~10× faster than HBM atomic and removes intra-block contention).
        for (int n = 0; n < ds; n++) {
            atomicAdd(&dq_bias_smem[n], dq_pre_out[n]);
            atomicAdd(&dk_bias_smem[n], dk_pre_out[n]);
        }

        // dAngles_cumsum from rotary gradient
        // dtheta = dQ_in * d(Q_rot)/d(theta) + dK_in_scaled * d(K_rot)/d(theta)
        for (int a = 0; a < n_angles && 2 * a + 1 < ds; a++) {
            float theta = Angles[angle_base + a];
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            int i0 = 2 * a, i1 = 2 * a + 1;
            // d(Q_rot)/dtheta: Q_rot[i0] = Q[i0]*cos - Q[i1]*sin
            //                  Q_rot[i1] = Q[i0]*sin + Q[i1]*cos
            // d/dtheta: [-Q[i0]*sin - Q[i1]*cos, Q[i0]*cos - Q[i1]*sin]
            float dtheta_q = dq_in[i0] * (-q_pre[i0] * sin_t - q_pre[i1] * cos_t)
                           + dq_in[i1] * (q_pre[i0] * cos_t - q_pre[i1] * sin_t);
            float dtheta_k = dk_in[i0] * (-k_pre[i0] * sin_t - k_pre[i1] * cos_t)
                           + dk_in[i1] * (k_pre[i0] * cos_t - k_pre[i1] * sin_t);
            dAngles_cumsum[((b * T + gt) * nh + h) * n_angles + a] = dtheta_q + dtheta_k;
        }
    }

    // One HBM atomic per (h, n) per block instead of CS×ds atomics.
    __syncthreads();
    for (int i = t_local; i < ds; i += blockDim.x) {
        atomicAdd(&dQ_bias[h * ds + i], dq_bias_smem[i]);
        atomicAdd(&dK_bias[h * ds + i], dk_bias_smem[i]);
    }
}

// ============================================================================
// 13. m3_ddt_dtrap -- dScale/dGamma -> dDT, dTrap (Python Step 4)
// ============================================================================
// Grid: (nh, B), Block: (1, 1, 1) — one thread per (head, batch), loops over T
// dDT[t] = (dGamma[t] + dScale[t]) * trap[t] + dScale[t-1] * (1 - trap[t])
// dTrap_presig[t] = ((dGamma[t] + dScale[t]) * dt[t] - dScale[t-1] * dt[t]) * sig*(1-sig)
// Grid: ceil(B*T*nh / 256), Block: 256
// Each thread handles one (b,t,h) element. Reads dScale[t-1] via shifted access.
extern "C" __global__ void m3_ddt_dtrap(
    float* __restrict__ dDT,          // [B*T*nh]
    float* __restrict__ dTrap_presig, // [B*T*nh]
    const float* __restrict__ dScale, // [B*T*nh]
    const float* __restrict__ dGamma, // [B*T*nh]
    const float* __restrict__ DT,     // [B*T*nh] — post-softplus dt
    const float* __restrict__ Trap,   // [B*T*nh] — post-sigmoid trap
    int B, int T, int nh
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * nh;
    if (idx >= total) return;

    // Decompose flat index: idx = (b * T + t) * nh + h
    int h = idx % nh;
    int bt = idx / nh;
    int t = bt % T;
    int b = bt / T;

    float ds_cur = dScale[idx];
    float dg_cur = dGamma[idx];
    float dt_val = DT[idx];
    float trap_val = Trap[idx];

    // dScale from previous timestep (shifted read, 0 for t=0)
    float ds_prev = 0.0f;
    if (t > 0) ds_prev = dScale[idx - nh]; // (b*T + t-1)*nh + h = idx - nh

    float d_dt = (dg_cur + ds_cur) * trap_val + ds_prev * (1.0f - trap_val);

    float d_trap_sig = ((dg_cur + ds_cur) - ds_prev) * dt_val;
    float d_trap_raw = d_trap_sig * trap_val * (1.0f - trap_val);

    dDT[idx] = d_dt;
    dTrap_presig[idx] = d_trap_raw;
}

// ============================================================================
// 14. m3_final_grads -- Combine dADT + dDT + dDT_angle -> d_dd_dt, d_dd_a
// ============================================================================
// Grid: ceil(N/256), Block: 256
// Replaces m3_abg_bwd for the parallel path.
extern "C" __global__ void m3_final_grads(
    float* __restrict__ d_dd_dt_raw,   // [B*T*nh]
    float* __restrict__ d_dd_a_raw,    // [B*T*nh]
    const float* __restrict__ dADT,    // [B*T*nh] — from m3_dqkv
    const float* __restrict__ dDT,     // [B*T*nh] — from m3_ddt_dtrap
    const float* __restrict__ dDT_angle, // [B*T*nh] — from angle_dt_bwd
    const float* __restrict__ a_val,   // [B*T*nh] — saved activation
    const float* __restrict__ DT,      // [B*T*nh] — saved post-softplus dt
    const float* __restrict__ dd_dt_raw_saved, // [B*T*nh] — saved pre-softplus dd_dt (NO bias)
    const float* __restrict__ dd_a_raw_saved,  // [B*T*nh] — saved pre-softplus dd_a
    const float* __restrict__ dt_bias,         // [nh] — bias added before softplus
    float a_floor,
    int N, // = B*T*nh
    int nh
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int h = i % nh;
    float d_adt = dADT[i];
    float d_dt_trap = dDT[i];
    float d_dt_angle_val = dDT_angle[i];

    float dt_val = DT[i];
    float a_v = a_val[i];
    float d_a_val = d_adt * dt_val;
    float d_dt_adt = d_adt * a_v;

    float d_dt_total = d_dt_trap + d_dt_angle_val + d_dt_adt;

    // softplus backward: sigmoid(dd_dt_raw + dt_bias) — must include bias!
    float raw_dt = dd_dt_raw_saved[i] + dt_bias[h];
    float sig_dt = 1.0f / (1.0f + exp2f(-raw_dt * LOG2E));
    d_dd_dt_raw[i] = d_dt_total * sig_dt;

    // d_dd_a_raw: from d_a_val, through clamp and -softplus
    float raw_a = dd_a_raw_saved[i];
    float sp_a = (raw_a > 20.0f) ? raw_a : logf(1.0f + FAST_EXP(raw_a));
    float a_unclamped = -sp_a; // -softplus(raw_a)
    // Was clamped to max(-a_floor)?
    // a_val = max(-softplus(raw), -a_floor) => a_val >= -a_floor (more negative)
    // Actually: a_val = min(-softplus(raw), -a_floor) = clamp(-, max=-a_floor)
    // If unclamped value > -a_floor (i.e., less negative than floor): gradient passes
    // If clamped: gradient = 0
    if (a_unclamped > -a_floor) {
        d_dd_a_raw[i] = 0.0f; // clamped, no gradient
    } else {
        // -softplus backward: d_raw = d_a_val * (-sigmoid(raw))
        float sig_a = 1.0f / (1.0f + exp2f(-raw_a * LOG2E));
        d_dd_a_raw[i] = d_a_val * (-sig_a);
    }
}

// ============================================================================
// Step 8c — typed (bf16/f16) variants of the chunked parallel forward kernels.
//
// Typed I/O: activation tensors (K, Q, K_scaled, x/V, y_out, k_flat, x_flat)
// are T_ACT in storage. All math + state remain f32 (BPTT states, dA_cumsum,
// prev_states, ssm/k/v persistent state, DT/trap_sig/qk_dot/scale/gamma/D).
//
// The following kernels stay f32-only and are NOT typed (per validation
// agents — O(T) compounding scan state mandates float for numerical safety):
//   - m3_dA_cumsum  (prefix-sum scan)
//   - m3_state_passing_fwd  (inter-chunk prefix recurrence)
// ============================================================================

#define DEFINE_M3_PREPROCESS_CHUNKS(SUFFIX, T_ACT, FROM_F)                    \
extern "C" __global__ void                                                    \
m3_preprocess_chunks_##SUFFIX(                                                \
    T_ACT* __restrict__ K_scaled,                                             \
    float* __restrict__ qk_dot,                                               \
    float* __restrict__ scale_out,                                            \
    float* __restrict__ gamma_out,                                            \
    const T_ACT* __restrict__ K,                                              \
    const T_ACT* __restrict__ Q,                                              \
    const float* __restrict__ DT,                                             \
    const float* __restrict__ trap_sig,                                       \
    int batch, int T, int nh, int ds, int chunk_size                          \
) {                                                                           \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                         \
    int bc = blockIdx.x;                                                      \
    int b = bc / n_chunks;                                                    \
    int chunk = bc % n_chunks;                                                \
    int h = blockIdx.y;                                                       \
    int t_local = threadIdx.x;                                                \
    int chunk_start = chunk * chunk_size;                                     \
    int t = chunk_start + t_local;                                            \
    if (t >= T) return;                                                       \
    int th = (b * T + t) * nh + h;                                            \
    float dt_cur = DT[th];                                                    \
    float trap_cur = trap_sig[th];                                            \
    float gamma_val = dt_cur * trap_cur;                                      \
    float shifted_gamma = 0.0f;                                               \
    if (t + 1 < T) {                                                          \
        int th_next = (b * T + t + 1) * nh + h;                               \
        float dt_next = DT[th_next];                                          \
        float trap_next = trap_sig[th_next];                                  \
        shifted_gamma = dt_next * (1.0f - trap_next);                         \
    }                                                                         \
    float scale_val = shifted_gamma + gamma_val;                              \
    scale_out[th] = scale_val;                                                \
    gamma_out[th] = gamma_val;                                                \
    int kq_base = (b * T + t) * nh * ds + h * ds;                             \
    float dot = 0.0f;                                                         \
    for (int n = 0; n < ds; n++) {                                            \
        dot += to_f(Q[kq_base + n]) * to_f(K[kq_base + n]);                   \
    }                                                                         \
    qk_dot[th] = dot * gamma_val;                                             \
    for (int n = 0; n < ds; n++) {                                            \
        float ks = to_f(K[kq_base + n]) * scale_val;                          \
        K_scaled[kq_base + n] = FROM_F(ks);                                   \
    }                                                                         \
}

DEFINE_M3_PREPROCESS_CHUNKS(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M3_PREPROCESS_CHUNKS(f16,  __half,        from_f_f16)

#define DEFINE_M3_CHUNK_STATE_FWD(SUFFIX, T_ACT, FROM_F)                      \
extern "C" __global__ __launch_bounds__(32, 4) void                           \
m3_chunk_state_fwd_##SUFFIX(                                                  \
    float* __restrict__ states_out,                                           \
    const T_ACT* __restrict__ x,                                              \
    const T_ACT* __restrict__ K_scaled,                                       \
    const float* __restrict__ dA_cumsum,                                      \
    int batch, int T, int nh, int hd, int ds, int chunk_size                  \
) {                                                                           \
    int d_inner = nh * hd;                                                    \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                         \
    int bc = blockIdx.x;                                                      \
    int b = bc / n_chunks;                                                    \
    int chunk = bc % n_chunks;                                                \
    int h = blockIdx.y;                                                       \
    int p = threadIdx.x;                                                      \
    if (p >= hd) return;                                                      \
    int chunk_start = chunk * chunk_size;                                     \
    int chunk_end = chunk_start + chunk_size;                                 \
    if (chunk_end > T) chunk_end = T;                                         \
    int chunk_len = chunk_end - chunk_start;                                  \
    int cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;             \
    float dA_end = dA_cumsum[cs_base + chunk_len - 1];                        \
    int state_base = ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds;    \
    for (int n = 0; n < ds; n++) {                                            \
        float acc = 0.0f;                                                     \
        for (int t = chunk_start; t < chunk_end; t++) {                       \
            int t_local = t - chunk_start;                                    \
            float dA_t = dA_cumsum[cs_base + t_local];                        \
            float decay = FAST_EXP(dA_end - dA_t);                            \
            float v_t = to_f(x[(b * T + t) * d_inner + h * hd + p]);          \
            float ks_t = to_f(K_scaled[(b * T + t) * nh * ds + h * ds + n]);  \
            acc += decay * ks_t * v_t;                                        \
        }                                                                     \
        states_out[state_base + n] = acc;                                     \
    }                                                                         \
    (void)FROM_F;                                                             \
}

DEFINE_M3_CHUNK_STATE_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M3_CHUNK_STATE_FWD(f16,  __half,        from_f_f16)

#define DEFINE_M3_WRITEBACK_PARALLEL_STATES(SUFFIX, T_ACT, FROM_F)            \
extern "C" __global__ void                                                    \
m3_writeback_parallel_states_##SUFFIX(                                        \
    float* __restrict__ ssm_state,                                            \
    float* __restrict__ k_state,                                              \
    float* __restrict__ v_state,                                              \
    const float* __restrict__ final_states,                                   \
    const T_ACT* __restrict__ k_flat,                                         \
    const T_ACT* __restrict__ x_flat,                                         \
    int batch, int T, int nh, int hd, int ds                                  \
) {                                                                           \
    int b = blockIdx.x;                                                       \
    int h = blockIdx.y;                                                       \
    int p = threadIdx.x;                                                      \
    if (b >= batch || h >= nh) return;                                        \
    int d_inner = nh * hd;                                                    \
    int dim = hd * ds;                                                        \
    if (p < hd) {                                                             \
        for (int n = 0; n < ds; n++) {                                        \
            int idx = b * nh * dim + h * dim + p * ds + n;                    \
            ssm_state[idx] = final_states[idx];                               \
        }                                                                     \
    }                                                                         \
    if (p < ds) {                                                             \
        int src_idx = (b * T + (T - 1)) * nh * ds + h * ds + p;               \
        int dst_idx = b * nh * ds + h * ds + p;                               \
        k_state[dst_idx] = to_f(k_flat[src_idx]);                             \
    }                                                                         \
    if (p < hd) {                                                             \
        int src_idx = (b * T + (T - 1)) * d_inner + h * hd + p;               \
        int dst_idx = b * nh * hd + h * hd + p;                               \
        v_state[dst_idx] = to_f(x_flat[src_idx]);                             \
    }                                                                         \
    (void)FROM_F;                                                             \
}

DEFINE_M3_WRITEBACK_PARALLEL_STATES(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M3_WRITEBACK_PARALLEL_STATES(f16,  __half,        from_f_f16)

#define DEFINE_M3_CHUNK_SCAN_FWD(SUFFIX, T_ACT, FROM_F)                       \
extern "C" __global__ __launch_bounds__(32, 4) void                           \
m3_chunk_scan_fwd_##SUFFIX(                                                   \
    T_ACT* __restrict__ y_out,                                                \
    const T_ACT* __restrict__ x,                                              \
    const T_ACT* __restrict__ Q,                                              \
    const T_ACT* __restrict__ K_scaled,                                       \
    const float* __restrict__ qk_dot,                                         \
    const float* __restrict__ dA_cumsum,                                      \
    const float* __restrict__ prev_states,                                    \
    const float* __restrict__ D,                                              \
    int batch, int T, int nh, int hd, int ds, int chunk_size                  \
) {                                                                           \
    int d_inner = nh * hd;                                                    \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                         \
    int bc = blockIdx.x;                                                      \
    int b = bc / n_chunks;                                                    \
    int chunk = bc % n_chunks;                                                \
    int h = blockIdx.y;                                                       \
    int p = threadIdx.x;                                                      \
    if (p >= hd) return;                                                      \
    int chunk_start = chunk * chunk_size;                                     \
    int chunk_end = chunk_start + chunk_size;                                 \
    if (chunk_end > T) chunk_end = T;                                         \
    int chunk_len = chunk_end - chunk_start;                                  \
    int cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;             \
    int state_base = ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds;    \
    float d_skip = D[h];                                                      \
    for (int t_local = 0; t_local < chunk_len; t_local++) {                   \
        int t = chunk_start + t_local;                                        \
        float dA_t = dA_cumsum[cs_base + t_local];                            \
        float y_off = 0.0f;                                                   \
        float state_decay = FAST_EXP(dA_t);                                   \
        int q_base = (b * T + t) * nh * ds + h * ds;                          \
        for (int n = 0; n < ds; n++) {                                        \
            y_off += to_f(Q[q_base + n]) * prev_states[state_base + n];       \
        }                                                                     \
        y_off *= state_decay;                                                 \
        float y_diag = 0.0f;                                                  \
        for (int s_local = 0; s_local < t_local; s_local++) {                 \
            int s = chunk_start + s_local;                                    \
            float dA_s = dA_cumsum[cs_base + s_local];                        \
            float decay = FAST_EXP(dA_t - dA_s);                              \
            float qk_val = 0.0f;                                              \
            int ks_base = (b * T + s) * nh * ds + h * ds;                     \
            for (int n = 0; n < ds; n++) {                                    \
                qk_val += to_f(Q[q_base + n]) * to_f(K_scaled[ks_base + n]);  \
            }                                                                 \
            float v_s = to_f(x[(b * T + s) * d_inner + h * hd + p]);          \
            y_diag += decay * qk_val * v_s;                                   \
        }                                                                     \
        float x_t = to_f(x[(b * T + t) * d_inner + h * hd + p]);              \
        int th = (b * T + t) * nh + h;                                        \
        float y_skip = (d_skip + qk_dot[th]) * x_t;                           \
        float y_val = y_diag + y_off + y_skip;                                \
        y_out[(b * T + t) * d_inner + h * hd + p] = FROM_F(y_val);            \
    }                                                                         \
}

DEFINE_M3_CHUNK_SCAN_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M3_CHUNK_SCAN_FWD(f16,  __half,        from_f_f16)

// ============================================================================
// Step 9d — typed (bf16/f16) variants of the chunked parallel backward
// kernels with typed input surface.
//
// All gradient OUTPUTS remain f32 — this is the PyTorch AMP master-grad
// invariant: bf16/f16 atomicAdd is not supported on compute capability ≤ sm_89
// and numerical safety requires f32 accumulation for reduction-style grads.
// Typed I/O is limited to saved-activation inputs (x, Q, K_scaled, d_y).
//
// The following backward kernels stay f32-only (no typed surface):
//   - m3_state_passing_bwd (inter-chunk prefix recurrence — f32 state only)
//   - m3_cumsum_bwd (reverse prefix sum — f32)
//   - m3_extract_da_cs_sum (f32-only utility)
// ============================================================================

#define DEFINE_M3_CHUNK_SCAN_BWD(SUFFIX, T_ACT, FROM_F)                       \
extern "C" __global__ __launch_bounds__(32, 4) void                           \
m3_chunk_scan_bwd_##SUFFIX(                                                   \
    float* __restrict__ d_x,                                                  \
    float* __restrict__ d_Q,                                                  \
    float* __restrict__ d_K_scaled,                                           \
    float* __restrict__ d_qk_dot,                                             \
    float* __restrict__ d_D,                                                  \
    float* __restrict__ d_prev_states,                                        \
    float* __restrict__ d_dA_cumsum,                                          \
    const T_ACT* __restrict__ d_y,                                            \
    const T_ACT* __restrict__ x,                                              \
    const T_ACT* __restrict__ Q,                                              \
    const T_ACT* __restrict__ K_scaled,                                       \
    const float* __restrict__ qk_dot_in,                                      \
    const float* __restrict__ dA_cumsum,                                      \
    const float* __restrict__ prev_states,                                    \
    const float* __restrict__ D,                                              \
    int batch, int T, int nh, int hd, int ds, int chunk_size                  \
) {                                                                           \
    int d_inner = nh * hd;                                                    \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                         \
    int bc = blockIdx.x;                                                      \
    int b = bc / n_chunks;                                                    \
    int chunk = bc % n_chunks;                                                \
    int h = blockIdx.y;                                                       \
    int p = threadIdx.x;                                                      \
    if (p >= hd) return;                                                      \
    int chunk_start = chunk * chunk_size;                                     \
    int chunk_end = chunk_start + chunk_size;                                 \
    if (chunk_end > T) chunk_end = T;                                         \
    int chunk_len = chunk_end - chunk_start;                                  \
    int cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;             \
    int state_base = ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds;    \
    unsigned warp_mask = (hd >= 32) ? 0xFFFFFFFFu : ((1u << hd) - 1u);        \
    float d_skip = D[h];                                                      \
    float d_D_acc = 0.0f;                                                     \
    for (int n = 0; n < ds; n++) {                                            \
        float d_state_n = 0.0f;                                               \
        for (int t_local = 0; t_local < chunk_len; t_local++) {               \
            int t = chunk_start + t_local;                                    \
            float dy = to_f(d_y[(b * T + t) * d_inner + h * hd + p]);         \
            float q_n = to_f(Q[(b * T + t) * nh * ds + h * ds + n]);          \
            float dA_t = dA_cumsum[cs_base + t_local];                        \
            float state_decay = FAST_EXP(dA_t);                               \
            d_state_n += dy * q_n * state_decay;                              \
        }                                                                     \
        d_prev_states[state_base + n] = d_state_n;                            \
    }                                                                         \
    for (int t_local = 0; t_local < chunk_len; t_local++) {                   \
        int t = chunk_start + t_local;                                        \
        int t_idx = b * T + t;                                                \
        float dy = to_f(d_y[t_idx * d_inner + h * hd + p]);                   \
        float dA_t = dA_cumsum[cs_base + t_local];                            \
        float x_t = to_f(x[t_idx * d_inner + h * hd + p]);                    \
        int th = t_idx * nh + h;                                              \
        d_D_acc += dy * x_t;                                                  \
        float d_x_val = dy * (d_skip + qk_dot_in[th]);                        \
        float d_qk_val = dy * x_t;                                            \
        for (int off = hd / 2; off > 0; off >>= 1)                            \
            d_qk_val += __shfl_down_sync(warp_mask, d_qk_val, off, hd);       \
        if (p == 0)                                                           \
            atomicAdd(&d_qk_dot[th], d_qk_val);                               \
        int q_base_t = t_idx * nh * ds + h * ds;                              \
        float state_decay = FAST_EXP(dA_t);                                   \
        for (int n = 0; n < ds; n++) {                                        \
            float d_q_state = dy * prev_states[state_base + n] * state_decay; \
            for (int off = hd / 2; off > 0; off >>= 1)                        \
                d_q_state += __shfl_down_sync(warp_mask, d_q_state, off, hd);\
            if (p == 0)                                                       \
                atomicAdd(&d_Q[q_base_t + n], d_q_state);                     \
        }                                                                     \
        for (int s_local = 0; s_local < t_local; s_local++) {                 \
            int s = chunk_start + s_local;                                    \
            float dA_s = dA_cumsum[cs_base + s_local];                        \
            float decay = FAST_EXP(dA_t - dA_s);                              \
            int ks_base = (b * T + s) * nh * ds + h * ds;                     \
            float qk_val = 0.0f;                                              \
            for (int n = 0; n < ds; n++) {                                    \
                qk_val += to_f(Q[q_base_t + n]) * to_f(K_scaled[ks_base + n]);\
            }                                                                 \
            float v_s = to_f(x[(b * T + s) * d_inner + h * hd + p]);          \
            float contrib = dy * decay * v_s;                                 \
            for (int n = 0; n < ds; n++) {                                    \
                float d_q_val = contrib * to_f(K_scaled[ks_base + n]);        \
                for (int off = hd / 2; off > 0; off >>= 1)                    \
                    d_q_val += __shfl_down_sync(warp_mask, d_q_val, off, hd);\
                if (p == 0)                                                   \
                    atomicAdd(&d_Q[q_base_t + n], d_q_val);                   \
            }                                                                 \
            for (int n = 0; n < ds; n++) {                                    \
                float d_ks_val = contrib * to_f(Q[q_base_t + n]);             \
                for (int off = hd / 2; off > 0; off >>= 1)                    \
                    d_ks_val += __shfl_down_sync(warp_mask, d_ks_val, off, hd);\
                if (p == 0)                                                   \
                    atomicAdd(&d_K_scaled[ks_base + n], d_ks_val);            \
            }                                                                 \
            float d_x_s = dy * decay * qk_val;                                \
            atomicAdd(&d_x[(b * T + s) * d_inner + h * hd + p], d_x_s);       \
            float dA_grad = dy * decay * qk_val * v_s;                        \
            for (int off = hd / 2; off > 0; off >>= 1)                        \
                dA_grad += __shfl_down_sync(warp_mask, dA_grad, off, hd);    \
            if (p == 0) {                                                     \
                atomicAdd(&d_dA_cumsum[cs_base + t_local], dA_grad);          \
                atomicAdd(&d_dA_cumsum[cs_base + s_local], -dA_grad);         \
            }                                                                 \
        }                                                                     \
        {                                                                     \
            float y_off = 0.0f;                                               \
            for (int n = 0; n < ds; n++) {                                    \
                y_off += to_f(Q[(b * T + t) * nh * ds + h * ds + n])          \
                         * prev_states[state_base + n];                       \
            }                                                                 \
            float dA_state_grad = dy * y_off * state_decay;                   \
            for (int off = hd / 2; off > 0; off >>= 1)                        \
                dA_state_grad += __shfl_down_sync(warp_mask,                  \
                                                  dA_state_grad, off, hd);    \
            if (p == 0)                                                       \
                atomicAdd(&d_dA_cumsum[cs_base + t_local], dA_state_grad);    \
        }                                                                     \
        d_x[t_idx * d_inner + h * hd + p] += d_x_val;                         \
    }                                                                         \
    atomicAdd(&d_D[h], d_D_acc);                                              \
    (void)FROM_F;                                                             \
}

DEFINE_M3_CHUNK_SCAN_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M3_CHUNK_SCAN_BWD(f16,  __half,        from_f_f16)

#define DEFINE_M3_CHUNK_STATE_BWD(SUFFIX, T_ACT, FROM_F)                      \
extern "C" __global__ __launch_bounds__(32, 4) void                           \
m3_chunk_state_bwd_##SUFFIX(                                                  \
    float* __restrict__ d_x,                                                  \
    float* __restrict__ d_K_scaled,                                           \
    float* __restrict__ d_dA_cumsum,                                          \
    const float* __restrict__ d_chunk_states,                                 \
    const T_ACT* __restrict__ x,                                              \
    const T_ACT* __restrict__ K_scaled,                                       \
    const float* __restrict__ dA_cumsum,                                      \
    int batch, int T, int nh, int hd, int ds, int chunk_size                  \
) {                                                                           \
    int d_inner = nh * hd;                                                    \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                         \
    int bc = blockIdx.x;                                                      \
    int b = bc / n_chunks;                                                    \
    int chunk = bc % n_chunks;                                                \
    int h = blockIdx.y;                                                       \
    int p = threadIdx.x;                                                      \
    if (p >= hd) return;                                                      \
    int chunk_start = chunk * chunk_size;                                     \
    int chunk_end = chunk_start + chunk_size;                                 \
    if (chunk_end > T) chunk_end = T;                                         \
    int chunk_len = chunk_end - chunk_start;                                  \
    int cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;             \
    int state_base = ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds;    \
    unsigned warp_mask = (hd >= 32) ? 0xFFFFFFFFu : ((1u << hd) - 1u);        \
    float dA_end = dA_cumsum[cs_base + chunk_len - 1];                        \
    for (int t_local = 0; t_local < chunk_len; t_local++) {                   \
        int t = chunk_start + t_local;                                        \
        int t_idx = b * T + t;                                                \
        float dA_t = dA_cumsum[cs_base + t_local];                            \
        float decay = FAST_EXP(dA_end - dA_t);                                \
        float x_val = to_f(x[t_idx * d_inner + h * hd + p]);                  \
        float d_x_add = 0.0f;                                                 \
        for (int n = 0; n < ds; n++) {                                        \
            float dcs = d_chunk_states[state_base + n];                       \
            float ks_n = to_f(K_scaled[t_idx * nh * ds + h * ds + n]);        \
            float common = dcs * decay;                                       \
            d_x_add += common * ks_n;                                         \
            float d_ks_val = common * x_val;                                  \
            for (int off = hd / 2; off > 0; off >>= 1)                        \
                d_ks_val += __shfl_down_sync(warp_mask, d_ks_val, off, hd);   \
            if (p == 0)                                                       \
                atomicAdd(&d_K_scaled[t_idx * nh * ds + h * ds + n],          \
                          d_ks_val);                                          \
            float dA_grad = common * ks_n * x_val;                            \
            for (int off_r = hd / 2; off_r > 0; off_r >>= 1)                  \
                dA_grad += __shfl_down_sync(warp_mask, dA_grad, off_r, hd);  \
            if (p == 0 && t_local != chunk_len - 1) {                         \
                atomicAdd(&d_dA_cumsum[cs_base + chunk_len - 1], dA_grad);    \
                atomicAdd(&d_dA_cumsum[cs_base + t_local], -dA_grad);         \
            }                                                                 \
        }                                                                     \
        d_x[t_idx * d_inner + h * hd + p] += d_x_add;                         \
    }                                                                         \
    (void)FROM_F;                                                             \
}

DEFINE_M3_CHUNK_STATE_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M3_CHUNK_STATE_BWD(f16,  __half,        from_f_f16)

// ============================================================================
// Step 9b — typed (bf16/f16) variants of the HIGHEST-RISK "final grad"
// kernels. Two of the four chunked-bwd tail kernels accept typed activation
// input; the other two operate on f32 scalar grads only.
//
// Typed:
//   - m3_dqkv       : typed Q_rot/K_scaled/V_in/dO; all grad outputs f32
//   - m3_dqktheta   : typed Q_raw/K_raw; all grad outputs f32
// Pure f32 (NO typed variant — operate on f32 scalar grad arrays only):
//   - m3_ddt_dtrap  (scale/gamma → dt/trap grads, f32 scalar math)
//   - m3_final_grads (combines f32 dADT + dDT + dDT_angle into final grads)
// ============================================================================

// __launch_bounds__: hd ≤ 32 per config (block_dim=hd), pin to 4 blocks/SM
// to keep the 64-element register arrays from spilling to local memory under
// nvcc's heuristics (audit Agent 5 M1).
#define DEFINE_M3_DQKV(SUFFIX, T_ACT, FROM_F)                                 \
extern "C" __global__ __launch_bounds__(32, 4) void                           \
m3_dqkv_##SUFFIX(                                                             \
    float* __restrict__ dQ_mid,                                               \
    float* __restrict__ dK_mid,                                               \
    float* __restrict__ dV,                                                   \
    float* __restrict__ dADT,                                                 \
    float* __restrict__ dQK_dot_out,                                          \
    float* __restrict__ dD_out,                                               \
    const T_ACT* __restrict__ Q_rot,                                          \
    const T_ACT* __restrict__ K_scaled,                                       \
    const T_ACT* __restrict__ V_in,                                           \
    const float* __restrict__ DA_CS,                                          \
    const float* __restrict__ DA_CS_SUM,                                      \
    const float* __restrict__ QK_dot_in,                                      \
    const float* __restrict__ SSM_States,                                     \
    const T_ACT* __restrict__ dO,                                             \
    const float* __restrict__ D_param,                                        \
    int B, int T, int nh_total, int hd, int ds, int CS                        \
) {                                                                           \
    int h = blockIdx.x;                                                       \
    int b = blockIdx.y;                                                       \
    int p = threadIdx.x;                                                      \
    if (h >= nh_total || b >= B || p >= hd) return;                           \
    if (ds > 64 || CS > 64) return;                                           \
    unsigned warp_mask = (hd >= 32) ? 0xFFFFFFFFu : ((1u << hd) - 1u);        \
    int d_inner = nh_total * hd;                                              \
    int n_chunks = (T + CS - 1) / CS;                                         \
    float D_val = D_param[h];                                                 \
    float dD_acc = 0.0f;                                                      \
    float d_state[64];                                                        \
    for (int n = 0; n < ds; n++) d_state[n] = 0.0f;                           \
    extern __shared__ float smem[];                                           \
    float* q_sm    = smem;                                                    \
    float* k_sm    = q_sm + CS * ds;                                          \
    float* v_sm    = k_sm + CS * ds;                                          \
    float* do_sm   = v_sm + CS * hd;                                          \
    float* da_cs_sm = do_sm + CS * hd;                                        \
    float* qk_sm   = da_cs_sm + CS;                                           \
    float* ssm_sm  = qk_sm + CS;                                              \
    for (int chunk_loop = 0; chunk_loop < n_chunks; chunk_loop++) {           \
        int chunk_idx = n_chunks - 1 - chunk_loop;                            \
        int chunk_start = chunk_idx * CS;                                     \
        int chunk_len = min(CS, T - chunk_start);                             \
        for (int t = 0; t < chunk_len; t++) {                                 \
            int gt = chunk_start + t;                                         \
            v_sm[t * hd + p] = to_f(V_in[(b * T + gt) * d_inner + h * hd + p]);\
            do_sm[t * hd + p] = to_f(dO[(b * T + gt) * d_inner + h * hd + p]);\
        }                                                                     \
        for (int t = chunk_len; t < CS; t++) {                                \
            v_sm[t * hd + p] = 0.0f;                                          \
            do_sm[t * hd + p] = 0.0f;                                         \
        }                                                                     \
        /* Each thread loads ALL ds entries with stride hd. Old `p < ds`     \
         * filter only worked when ds <= hd; for ds > hd (e.g. ds=16, hd=8)  \
         * entries n=hd..ds-1 stayed garbage in shared memory.               */\
        for (int n = p; n < ds; n += hd) {                                    \
            for (int t = 0; t < chunk_len; t++) {                             \
                int gt = chunk_start + t;                                     \
                q_sm[t * ds + n] = to_f(                                      \
                    Q_rot[((b * T + gt) * nh_total + h) * ds + n]);           \
                k_sm[t * ds + n] = to_f(                                      \
                    K_scaled[((b * T + gt) * nh_total + h) * ds + n]);        \
            }                                                                 \
            for (int t = chunk_len; t < CS; t++) {                            \
                q_sm[t * ds + n] = 0.0f;                                      \
                k_sm[t * ds + n] = 0.0f;                                      \
            }                                                                 \
        }                                                                     \
        if (p == 0) {                                                         \
            for (int t = 0; t < chunk_len; t++) {                             \
                da_cs_sm[t] = DA_CS[                                          \
                    ((b * n_chunks + chunk_idx) * nh_total + h) * CS + t];    \
                qk_sm[t] = QK_dot_in[                                         \
                    (b * T + chunk_start + t) * nh_total + h];                \
            }                                                                 \
            for (int t = chunk_len; t < CS; t++) {                            \
                da_cs_sm[t] = 0.0f;                                           \
                qk_sm[t] = 0.0f;                                              \
            }                                                                 \
        }                                                                     \
        for (int n = 0; n < ds; n++) {                                        \
            ssm_sm[p * ds + n] = SSM_States[                                  \
                ((b * n_chunks + chunk_idx) * nh_total + h) * hd * ds         \
                + p * ds + n];                                                \
        }                                                                     \
        __syncthreads();                                                      \
        float da_cs_chunk_sum = DA_CS_SUM[                                    \
            (b * n_chunks + chunk_idx) * nh_total + h];                       \
        for (int t = 0; t < chunk_len; t++) {                                 \
            int gt = chunk_start + t;                                         \
            float dA_t = da_cs_sm[t];                                         \
            float exp_rev_t = exp2f((da_cs_chunk_sum - dA_t) * LOG2E);        \
            float dv_intra = 0.0f;                                            \
            for (int s = t + 1; s < chunk_len; s++) {                         \
                float kq = 0.0f;                                              \
                for (int n = 0; n < ds; n++)                                  \
                    kq += k_sm[t * ds + n] * q_sm[s * ds + n];                \
                float decay = exp2f((da_cs_sm[s] - dA_t) * LOG2E);            \
                dv_intra += kq * decay * do_sm[s * hd + p];                   \
            }                                                                 \
            float dv_inter = 0.0f;                                            \
            for (int n = 0; n < ds; n++)                                      \
                dv_inter += k_sm[t * ds + n] * d_state[n];                    \
            dv_inter *= exp_rev_t;                                            \
            float dv_skip = do_sm[t * hd + p] * (D_val + qk_sm[t]);           \
            dV[(b * T + gt) * d_inner + h * hd + p] =                         \
                dv_intra + dv_inter + dv_skip;                                \
            float dqk_val = do_sm[t * hd + p] * v_sm[t * hd + p];             \
            for (int off = hd / 2; off > 0; off >>= 1)                        \
                dqk_val += __shfl_down_sync(warp_mask, dqk_val, off, hd);    \
            if (p == 0) {                                                     \
                dQK_dot_out[(b * T + gt) * nh_total + h] = dqk_val;            \
                dD_acc += dqk_val;                                            \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
        if (p < ds) {                                                         \
            int n = p;                                                        \
            for (int t = 0; t < chunk_len; t++) {                             \
                int gt = chunk_start + t;                                     \
                float dA_t = da_cs_sm[t];                                     \
                float dk_intra = 0.0f;                                        \
                for (int s = t + 1; s < chunk_len; s++) {                     \
                    float vdo = 0.0f;                                         \
                    for (int pp = 0; pp < hd; pp++)                           \
                        vdo += v_sm[t * hd + pp] * do_sm[s * hd + pp];        \
                    float decay = exp2f((da_cs_sm[s] - dA_t) * LOG2E);        \
                    dk_intra += vdo * decay * q_sm[s * ds + n];               \
                }                                                             \
                dK_mid[((b * T + gt) * nh_total + h) * ds + n] = dk_intra;    \
                float dq_intra = 0.0f;                                        \
                for (int s = 0; s < t; s++) {                                 \
                    float vdo = 0.0f;                                         \
                    for (int pp = 0; pp < hd; pp++)                           \
                        vdo += v_sm[s * hd + pp] * do_sm[t * hd + pp];        \
                    float decay = exp2f((dA_t - da_cs_sm[s]) * LOG2E);        \
                    dq_intra += vdo * decay * k_sm[s * ds + n];               \
                }                                                             \
                float dq_inter = 0.0f;                                        \
                for (int pp = 0; pp < hd; pp++)                               \
                    dq_inter += do_sm[t * hd + pp] * ssm_sm[pp * ds + n];     \
                dq_inter *= exp2f(dA_t * LOG2E);                              \
                dQ_mid[((b * T + gt) * nh_total + h) * ds + n] =              \
                    dq_intra + dq_inter;                                      \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
        for (int n = 0; n < ds; n++)                                          \
            ssm_sm[p * ds + n] = d_state[n];                                  \
        __syncthreads();                                                      \
        if (p < ds) {                                                         \
            int n = p;                                                        \
            for (int t = 0; t < chunk_len; t++) {                             \
                float dk_inter = 0.0f;                                        \
                float exp_rev_t = exp2f(                                      \
                    (da_cs_chunk_sum - da_cs_sm[t]) * LOG2E);                 \
                for (int pp = 0; pp < hd; pp++)                               \
                    dk_inter += v_sm[t * hd + pp] * ssm_sm[pp * ds + n];      \
                dk_inter *= exp_rev_t;                                        \
                int gt = chunk_start + t;                                     \
                dK_mid[((b * T + gt) * nh_total + h) * ds + n] += dk_inter;   \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
        if (p == 0) {                                                         \
            float dM_rev[64];                                                 \
            for (int t = 0; t < CS; t++) dM_rev[t] = 0.0f;                    \
            for (int i = 0; i < chunk_len; i++) {                             \
                for (int j = i + 1; j < chunk_len; j++) {                     \
                    float vdo = 0.0f;                                         \
                    for (int pp = 0; pp < hd; pp++)                           \
                        vdo += v_sm[i * hd + pp] * do_sm[j * hd + pp];        \
                    float decay = exp2f((da_cs_sm[j] - da_cs_sm[i]) * LOG2E); \
                    float kq = 0.0f;                                          \
                    for (int n = 0; n < ds; n++)                              \
                        kq += k_sm[i * ds + n] * q_sm[j * ds + n];            \
                    float dAinv = vdo * decay * kq;                           \
                    dM_rev[j] += dAinv;                                       \
                    dM_rev[i] -= dAinv;                                       \
                }                                                             \
            }                                                                 \
            for (int t = 0; t < chunk_len; t++) {                             \
                float qs_do = 0.0f;                                           \
                for (int pp = 0; pp < hd; pp++) {                             \
                    float qs = 0.0f;                                          \
                    for (int n = 0; n < ds; n++) {                            \
                        float ssm_val = SSM_States[                           \
                            ((b * n_chunks + chunk_idx) * nh_total + h)       \
                            * hd * ds + pp * ds + n];                         \
                        qs += q_sm[t * ds + n] * ssm_val;                     \
                    }                                                         \
                    qs_do += qs * do_sm[t * hd + pp];                         \
                }                                                             \
                dM_rev[t] += qs_do * exp2f(da_cs_sm[t] * LOG2E);              \
            }                                                                 \
            float dM_scalar = 0.0f;                                           \
            for (int pp = 0; pp < hd; pp++) {                                 \
                for (int n = 0; n < ds; n++) {                                \
                    float ssm_val = SSM_States[                               \
                        ((b * n_chunks + chunk_idx) * nh_total + h)           \
                        * hd * ds + pp * ds + n];                             \
                    dM_scalar += ssm_val * ssm_sm[pp * ds + n];               \
                }                                                             \
            }                                                                 \
            dM_scalar *= exp2f(da_cs_chunk_sum * LOG2E);                      \
            float dM_vector[64];                                              \
            for (int t = 0; t < chunk_len; t++) {                             \
                float dsk_v = 0.0f;                                           \
                for (int pp = 0; pp < hd; pp++) {                             \
                    float dsk = 0.0f;                                         \
                    for (int n = 0; n < ds; n++)                              \
                        dsk += k_sm[t * ds + n] * ssm_sm[pp * ds + n];        \
                    dsk_v += dsk * v_sm[t * hd + pp];                         \
                }                                                             \
                dM_vector[t] = dsk_v                                          \
                    * exp2f((da_cs_chunk_sum - da_cs_sm[t]) * LOG2E);         \
            }                                                                 \
            float total_rev = 0.0f;                                           \
            for (int t = 0; t < chunk_len; t++) total_rev += dM_rev[t];       \
            total_rev += dM_scalar;                                           \
            float cumsum = 0.0f;                                              \
            for (int t = 0; t < chunk_len; t++) {                             \
                cumsum += dM_vector[t] - dM_rev[t];                           \
                dM_rev[t] += total_rev + cumsum - dM_vector[t];               \
            }                                                                 \
            for (int t = 0; t < chunk_len; t++) {                             \
                int gt = chunk_start + t;                                     \
                dADT[(b * T + gt) * nh_total + h] = dM_rev[t];                \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
        for (int n = 0; n < ds; n++) {                                        \
            float new_val = exp2f(da_cs_chunk_sum * LOG2E) * d_state[n];      \
            for (int t = 0; t < chunk_len; t++) {                             \
                new_val += do_sm[t * hd + p]                                  \
                    * exp2f(da_cs_sm[t] * LOG2E) * q_sm[t * ds + n];          \
            }                                                                 \
            d_state[n] = new_val;                                             \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    if (p == 0) atomicAdd(&dD_out[h], dD_acc);                                \
    (void)FROM_F;                                                             \
}

DEFINE_M3_DQKV(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M3_DQKV(f16,  __half,        from_f_f16)

// __launch_bounds__: block_dim=CS ≤ 64. 6× float[64] register arrays per
// thread risk spilling under nvcc heuristics — pin to 4 blocks/SM.
#define DEFINE_M3_DQKTHETA(SUFFIX, T_ACT, FROM_F)                             \
extern "C" __global__ __launch_bounds__(64, 4) void                           \
m3_dqktheta_##SUFFIX(                                                         \
    float* __restrict__ dQ_pre,                                               \
    float* __restrict__ dK_pre,                                               \
    float* __restrict__ dAngles_cumsum,                                       \
    float* __restrict__ dScale,                                               \
    float* __restrict__ dGamma,                                               \
    float* __restrict__ dQ_bias,                                              \
    float* __restrict__ dK_bias,                                              \
    const T_ACT* __restrict__ Q_raw,                                          \
    const T_ACT* __restrict__ K_raw,                                          \
    const float* __restrict__ Scale_in,                                       \
    const float* __restrict__ Gamma_in,                                       \
    const float* __restrict__ Angles,                                         \
    const float* __restrict__ dQ_mid,                                         \
    const float* __restrict__ dK_mid,                                         \
    const float* __restrict__ dQK_dot,                                        \
    int B, int T, int nh, int ds, int n_angles, int CS                        \
) {                                                                           \
    int n_chunks = (T + CS - 1) / CS;                                         \
    int bc = blockIdx.x;                                                      \
    int b_chunk = bc;                                                         \
    int h = blockIdx.y;                                                       \
    int t_local = threadIdx.x;                                                \
    int b = b_chunk / n_chunks;                                               \
    int chunk = b_chunk % n_chunks;                                           \
    int gt = chunk * CS + t_local;                                            \
    if (ds > 64) return;                                                      \
    if (b >= B || h >= nh) return;                                            \
    bool valid = (gt < T);                                                    \
    __shared__ float dq_bias_smem[64];                                        \
    __shared__ float dk_bias_smem[64];                                        \
    for (int i = t_local; i < ds; i += blockDim.x) {                          \
        dq_bias_smem[i] = 0.0f;                                               \
        dk_bias_smem[i] = 0.0f;                                               \
    }                                                                         \
    __syncthreads();                                                          \
    if (valid) {                                                              \
    int base = ((b * T + gt) * nh + h) * ds;                                  \
    float scale = Scale_in[(b * T + gt) * nh + h];                            \
    float gamma = Gamma_in[(b * T + gt) * nh + h];                            \
    float dqk = dQK_dot[(b * T + gt) * nh + h];                               \
    float q_pre[64], k_pre[64];                                               \
    for (int n = 0; n < ds; n++) {                                            \
        q_pre[n] = to_f(Q_raw[base + n]);                                     \
        k_pre[n] = to_f(K_raw[base + n]);                                     \
    }                                                                         \
    float dq_in[64], dk_in[64];                                               \
    for (int n = 0; n < ds; n++) {                                            \
        dq_in[n] = dQ_mid[base + n];                                          \
        dk_in[n] = dK_mid[base + n];                                          \
    }                                                                         \
    float k_rot[64];                                                          \
    int angle_base = ((b * T + gt) * nh + h) * n_angles;                      \
    for (int a = 0; a < n_angles && 2 * a + 1 < ds; a++) {                    \
        float theta = Angles[angle_base + a];                                 \
        float cos_t = cosf(theta);                                            \
        float sin_t = sinf(theta);                                            \
        int i0 = 2 * a, i1 = 2 * a + 1;                                       \
        k_rot[i0] = k_pre[i0] * cos_t - k_pre[i1] * sin_t;                    \
        k_rot[i1] = k_pre[i0] * sin_t + k_pre[i1] * cos_t;                    \
    }                                                                         \
    for (int n = 2 * n_angles; n < ds; n++) k_rot[n] = k_pre[n];              \
    float d_scale = 0.0f;                                                     \
    for (int n = 0; n < ds; n++) d_scale += dk_in[n] * k_rot[n];              \
    dScale[(b * T + gt) * nh + h] = d_scale;                                  \
    float qk_raw = 0.0f;                                                      \
    for (int n = 0; n < ds; n++) qk_raw += q_pre[n] * k_pre[n];               \
    dGamma[(b * T + gt) * nh + h] = dqk * qk_raw;                             \
    for (int n = 0; n < ds; n++) dk_in[n] *= scale;                           \
    float dq_pre_out[64], dk_pre_out[64];                                     \
    for (int n = 0; n < ds; n++) {                                            \
        dq_pre_out[n] = dq_in[n];                                             \
        dk_pre_out[n] = dk_in[n];                                             \
    }                                                                         \
    for (int a = 0; a < n_angles && 2 * a + 1 < ds; a++) {                    \
        float theta = Angles[angle_base + a];                                 \
        float cos_t = cosf(theta);                                            \
        float sin_t = sinf(theta);                                            \
        int i0 = 2 * a, i1 = 2 * a + 1;                                       \
        dq_pre_out[i0] = dq_in[i0] * cos_t + dq_in[i1] * sin_t;               \
        dq_pre_out[i1] = -dq_in[i0] * sin_t + dq_in[i1] * cos_t;              \
        dk_pre_out[i0] = dk_in[i0] * cos_t + dk_in[i1] * sin_t;               \
        dk_pre_out[i1] = -dk_in[i0] * sin_t + dk_in[i1] * cos_t;              \
    }                                                                         \
    float dqk_gamma = dqk * gamma;                                            \
    for (int n = 0; n < ds; n++) {                                            \
        dq_pre_out[n] += dqk_gamma * k_pre[n];                                \
        dk_pre_out[n] += dqk_gamma * q_pre[n];                                \
    }                                                                         \
    for (int n = 0; n < ds; n++) {                                            \
        dQ_pre[base + n] = dq_pre_out[n];                                     \
        dK_pre[base + n] = dk_pre_out[n];                                     \
    }                                                                         \
    for (int n = 0; n < ds; n++) {                                            \
        atomicAdd(&dq_bias_smem[n], dq_pre_out[n]);                           \
        atomicAdd(&dk_bias_smem[n], dk_pre_out[n]);                           \
    }                                                                         \
    for (int a = 0; a < n_angles && 2 * a + 1 < ds; a++) {                    \
        float theta = Angles[angle_base + a];                                 \
        float cos_t = cosf(theta);                                            \
        float sin_t = sinf(theta);                                            \
        int i0 = 2 * a, i1 = 2 * a + 1;                                       \
        float dtheta_q = dq_in[i0]                                            \
            * (-q_pre[i0] * sin_t - q_pre[i1] * cos_t)                        \
            + dq_in[i1] * (q_pre[i0] * cos_t - q_pre[i1] * sin_t);            \
        float dtheta_k = dk_in[i0]                                            \
            * (-k_pre[i0] * sin_t - k_pre[i1] * cos_t)                        \
            + dk_in[i1] * (k_pre[i0] * cos_t - k_pre[i1] * sin_t);            \
        dAngles_cumsum[((b * T + gt) * nh + h) * n_angles + a] =              \
            dtheta_q + dtheta_k;                                              \
    }                                                                         \
    } /* end if (valid) */                                                    \
    __syncthreads();                                                          \
    for (int i = t_local; i < ds; i += blockDim.x) {                          \
        atomicAdd(&dQ_bias[h * ds + i], dq_bias_smem[i]);                     \
        atomicAdd(&dK_bias[h * ds + i], dk_bias_smem[i]);                     \
    }                                                                         \
    (void)FROM_F;                                                             \
}

DEFINE_M3_DQKTHETA(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M3_DQKTHETA(f16,  __half,        from_f_f16)
