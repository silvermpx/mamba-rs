// Mamba-3 SISO trapezoidal SSM recurrence CUDA kernels.
//
// Sequential mode (no chunking) for the trapezoidal recurrence:
//   h[p,n] = alpha * h_prev[p,n] + beta * v_prev[p] * k_prev[n] + gamma * x[p] * k_cur[n]
//   y[p] = sum_n(h[p,n] * q[n]) + D * x[p]
//
// where v_prev = previous x, k_prev = previous B (post bias+RoPE),
// k_cur = current B (post bias+RoPE), q = current C (post bias+RoPE).
//
// alpha, beta, gamma, k_cur, q_cur are PRE-COMPUTED by shared ops kernels
// (m3_split, bcnorm, bias_add, rope). These kernels just consume them.
//
// Thread mapping: one thread per (batch, head, p) where p in [0, headdim).
// Grid: (B, nh, 1), Block: (hd, 1, 1).
// Each thread carries h_local[d_state] in registers (max ds=64).
//
// Key differences from Mamba-2 SSD (mamba2_ssd.cu):
// - Trapezoidal recurrence (3 terms: alpha*h + beta*v*k_prev + gamma*x*k_cur)
//   vs Mamba-2 exponential (2 terms: da*h + delta*x*B)
// - Separate k_state and v_state (delayed one step) vs none
// - alpha/beta/gamma per (batch, head) vs da_exp per (batch, head)
// - No ngroups indexing -- B/C already expanded to per-head by bias+RoPE ops
// - No delta: discretization absorbed into alpha/beta/gamma
//
// Source: CPU reference: native/train/mamba2/forward_m3.rs (phase F5)
//         T=1 step: native/model/mamba2.rs (mamba3_step_f32)
// Paper: Lahoti et al. "Mamba-3: SISO" (ICLR 2026)
//
// O1 Warp Shuffle Optimization (sm_30+):
//   Within each head, all headdim threads read identical alpha, beta, gamma, D,
//   k_cur[n], q_cur[n], k_prev[n]. Only lane p=0 loads from global memory;
//   __shfl_sync broadcasts to all lanes.
//   Precondition: headdim must be a power-of-2 and <= warpSize (32).
//   Default config headdim=16 satisfies this.
//
// O5 Transposed h_saved Layout:
//   h_saved index: b*(T+1)*nhd_ds + t*nhd_ds + n*d_inner + (h*hd+p)
//   where nhd_ds = d_inner * d_state.
//   All headdim threads in a warp access consecutive addresses at each n step
//   -> perfectly coalesced (1 cache line per warp per n step).

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

// ======================== FORWARD ========================

// m3_step_fwd: Mamba-3 T=1 inference/collection step.
//
// One thread per (batch, head, position). Trapezoidal recurrence.
// Grid: (B, nh, 1), Block: (hd, 1, 1).
//
// After SSM update, k_state and v_state are updated for next step.
// k_state update: only p=0 writes (all p share the same k_cur values).
// v_state update: each p writes its own x value.
extern "C" __global__ void m3_step_fwd(
    // In/Out: persistent state (mutated)
    float* ssm_state,   // [B * nh * hd * ds]
    float* k_state,     // [B * nh * ds] -- previous K (post bias+RoPE)
    float* v_state,     // [B * nh * hd] -- previous x
    // Output
    float* y,           // [B * d_inner]
    // Inputs (pre-computed by shared ops)
    const float* x,     // [B * d_inner]   (d_inner = nh * hd)
    const float* k_cur, // [B * nh * ds]   (B after bias+RoPE)
    const float* q_cur, // [B * nh * ds]   (C after bias+RoPE)
    const float* alpha, // [B * nh]
    const float* beta,  // [B * nh]
    const float* gamma, // [B * nh]
    const float* D,     // [nh]
    int batch, int nh, int hd, int ds
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int p = threadIdx.x;
    if (b >= batch || h >= nh || p >= hd) return;

    int d_inner = nh * hd;

    // Load h_local into registers
    float h_local[64];
    if (ds > 64) return;
    int h_base = (b * nh * hd + h * hd + p) * ds;
    for (int n = 0; n < ds; n++)
        h_local[n] = ssm_state[h_base + n];

    // O1: alpha, beta, gamma broadcast from p=0
    float alpha_h = 0.0f;
    if (p == 0) alpha_h = alpha[b * nh + h];
    alpha_h = __shfl_sync(0xFFFFFFFF, alpha_h, 0, hd);

    float beta_h = 0.0f;
    if (p == 0) beta_h = beta[b * nh + h];
    beta_h = __shfl_sync(0xFFFFFFFF, beta_h, 0, hd);

    float gamma_h = 0.0f;
    if (p == 0) gamma_h = gamma[b * nh + h];
    gamma_h = __shfl_sync(0xFFFFFFFF, gamma_h, 0, hd);

    // O1: D[h] broadcast from p=0
    float d_skip = 0.0f;
    if (p == 0) d_skip = D[h];
    d_skip = __shfl_sync(0xFFFFFFFF, d_skip, 0, hd);

    // Per-thread unique values
    float x_val = x[b * d_inner + h * hd + p];
    float v_prev = v_state[b * nh * hd + h * hd + p];

    float y_val = d_skip * x_val;

    for (int n = 0; n < ds; n++) {
        // O1: k_cur[n], k_prev[n], q_cur[n] broadcast from p=0
        float kc_n = 0.0f, kp_n = 0.0f, qc_n = 0.0f;
        if (p == 0) {
            kc_n = k_cur[b * nh * ds + h * ds + n];
            kp_n = k_state[b * nh * ds + h * ds + n];
            qc_n = q_cur[b * nh * ds + h * ds + n];
        }
        kc_n = __shfl_sync(0xFFFFFFFF, kc_n, 0, hd);
        kp_n = __shfl_sync(0xFFFFFFFF, kp_n, 0, hd);
        qc_n = __shfl_sync(0xFFFFFFFF, qc_n, 0, hd);

        // Trapezoidal recurrence
        h_local[n] = alpha_h * h_local[n] + beta_h * v_prev * kp_n + gamma_h * x_val * kc_n;
        y_val += h_local[n] * qc_n;
    }

    // Write back SSM state
    for (int n = 0; n < ds; n++)
        ssm_state[h_base + n] = h_local[n];

    // Write output
    y[b * d_inner + h * hd + p] = y_val;

    // Update k_state: only p=0 writes (k_cur is shared across all p in this head)
    if (p == 0) {
        for (int n = 0; n < ds; n++)
            k_state[b * nh * ds + h * ds + n] = k_cur[b * nh * ds + h * ds + n];
    }

    // Update v_state: each p writes its own x
    v_state[b * nh * hd + h * hd + p] = x_val;
}

// Templated m3_step_fwd — activations in T_IN, state/D stay f32.
// Used for T=1 decode in Mamba-3 LLM inference.
#define DEFINE_M3_STEP_FWD(SUFFIX, TY, FROM_F)                              \
extern "C" __global__ void m3_step_fwd_##SUFFIX(                             \
    float* ssm_state,                                                        \
    float* k_state,                                                          \
    float* v_state,                                                          \
    TY* y,                                                                   \
    const TY* x,                                                             \
    const TY* k_cur,                                                         \
    const TY* q_cur,                                                         \
    const float* alpha, const float* beta, const float* gamma,               \
    const float* D,                                                          \
    int batch, int nh, int hd, int ds                                        \
) {                                                                          \
    int b = blockIdx.x;                                                      \
    int h = blockIdx.y;                                                      \
    int p = threadIdx.x;                                                     \
    if (b >= batch || h >= nh || p >= hd) return;                            \
    int d_inner = nh * hd;                                                   \
    float h_local[64];                                                       \
    if (ds > 64) return;                                                     \
    int h_base = (b * nh * hd + h * hd + p) * ds;                            \
    for (int n = 0; n < ds; n++) h_local[n] = ssm_state[h_base + n];         \
    float alpha_h = 0.0f;                                                    \
    if (p == 0) alpha_h = alpha[b * nh + h];                                 \
    alpha_h = __shfl_sync(0xFFFFFFFF, alpha_h, 0, hd);                       \
    float beta_h = 0.0f;                                                     \
    if (p == 0) beta_h = beta[b * nh + h];                                   \
    beta_h = __shfl_sync(0xFFFFFFFF, beta_h, 0, hd);                         \
    float gamma_h = 0.0f;                                                    \
    if (p == 0) gamma_h = gamma[b * nh + h];                                 \
    gamma_h = __shfl_sync(0xFFFFFFFF, gamma_h, 0, hd);                       \
    float d_skip = 0.0f;                                                     \
    if (p == 0) d_skip = D[h];                                               \
    d_skip = __shfl_sync(0xFFFFFFFF, d_skip, 0, hd);                         \
    float x_val = to_f(x[b * d_inner + h * hd + p]);                         \
    float v_prev = v_state[b * nh * hd + h * hd + p];                        \
    float y_val = d_skip * x_val;                                            \
    for (int n = 0; n < ds; n++) {                                           \
        float kc_n = 0.0f, kp_n = 0.0f, qc_n = 0.0f;                         \
        if (p == 0) {                                                        \
            kc_n = to_f(k_cur[b * nh * ds + h * ds + n]);                    \
            kp_n = k_state[b * nh * ds + h * ds + n];                        \
            qc_n = to_f(q_cur[b * nh * ds + h * ds + n]);                    \
        }                                                                    \
        kc_n = __shfl_sync(0xFFFFFFFF, kc_n, 0, hd);                         \
        kp_n = __shfl_sync(0xFFFFFFFF, kp_n, 0, hd);                         \
        qc_n = __shfl_sync(0xFFFFFFFF, qc_n, 0, hd);                         \
        h_local[n] = alpha_h * h_local[n] + beta_h * v_prev * kp_n +         \
                     gamma_h * x_val * kc_n;                                 \
        y_val += h_local[n] * qc_n;                                          \
    }                                                                        \
    for (int n = 0; n < ds; n++) ssm_state[h_base + n] = h_local[n];         \
    y[b * d_inner + h * hd + p] = FROM_F(y_val);                             \
    if (p == 0) {                                                            \
        for (int n = 0; n < ds; n++)                                         \
            k_state[b * nh * ds + h * ds + n] =                              \
                to_f(k_cur[b * nh * ds + h * ds + n]);                       \
    }                                                                        \
    v_state[b * nh * hd + h * hd + p] = x_val;                               \
}

DEFINE_M3_STEP_FWD(f32,  float,         from_f_f32)
DEFINE_M3_STEP_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M3_STEP_FWD(f16,  __half,        from_f_f16)

// ======================== BURN-IN FORWARD (with activation saves) ========================

// m3_burnin_fwd: Sequential T>1 forward with activation saves for backward BPTT.
//
// Grid: (B, nh, 1), Block: (hd, 1, 1).
// Each thread loops over T sequentially, carrying h_local[ds] in registers.
//
// Saves:
//   h_saved[B * (T+1) * nh * hd * ds] -- SSM state before each step + final state
//     O5 transposed layout: index = b*(T+1)*nhd_ds + t*nhd_ds + n*d_inner + h*hd+p
//   k_prev_saved[B * T * nh * ds] -- k_state entering each timestep
//   v_prev_saved[B * T * nh * hd] -- v_state (= previous x) entering each timestep
extern "C" __global__ void m3_burnin_fwd(
    // In/Out: persistent state (mutated)
    float* ssm_state,       // [B * nh * hd * ds]
    float* k_state,         // [B * nh * ds]
    float* v_state,         // [B * nh * hd]
    // Output
    float* y_out,           // [B * T * d_inner]
    // Saves for backward
    float* h_saved,         // [B * (T+1) * nh * hd * ds] (O5 transposed)
    float* k_prev_saved,    // [B * T * nh * ds]
    float* v_prev_saved,    // [B * T * nh * hd]
    // Inputs (pre-computed by shared ops, flat over T)
    const float* x_flat,      // [B * T * d_inner]
    const float* k_flat,      // [B * T * nh * ds]
    const float* q_flat,      // [B * T * nh * ds]
    const float* alpha_flat,  // [B * T * nh]
    const float* beta_flat,   // [B * T * nh]
    const float* gamma_flat,  // [B * T * nh]
    const float* D,           // [nh]
    int batch, int T, int nh, int hd, int ds
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int p = threadIdx.x;
    if (b >= batch || h >= nh || p >= hd) return;

    int d_inner = nh * hd;
    int nhd_ds = d_inner * ds;

    // Load h_local from persistent state into registers
    float h_local[64];
    if (ds > 64) return;
    int h_base = (b * nh * hd + h * hd + p) * ds;
    for (int n = 0; n < ds; n++)
        h_local[n] = ssm_state[h_base + n];

    // O1: D[h] broadcast once before T loop
    float d_skip = 0.0f;
    if (p == 0) d_skip = D[h];
    d_skip = __shfl_sync(0xFFFFFFFF, d_skip, 0, hd);

    // Save initial h at time 0 (O5 transposed layout)
    for (int n = 0; n < ds; n++) {
        int hs_idx = n * d_inner + h * hd + p;
        h_saved[b * (T + 1) * nhd_ds + hs_idx] = h_local[n];
    }

    for (int t = 0; t < T; t++) {
        // Save k_prev: only p=0 writes (k_state is per-head, shared across p)
        if (p == 0) {
            for (int n = 0; n < ds; n++)
                k_prev_saved[(b * T + t) * nh * ds + h * ds + n] = k_state[b * nh * ds + h * ds + n];
        }

        // Save v_prev: each p saves its own
        v_prev_saved[(b * T + t) * d_inner + h * hd + p] = v_state[b * nh * hd + h * hd + p];

        // O1: alpha, beta, gamma broadcast from p=0
        float alpha_h = 0.0f;
        if (p == 0) alpha_h = alpha_flat[(b * T + t) * nh + h];
        alpha_h = __shfl_sync(0xFFFFFFFF, alpha_h, 0, hd);

        float beta_h = 0.0f;
        if (p == 0) beta_h = beta_flat[(b * T + t) * nh + h];
        beta_h = __shfl_sync(0xFFFFFFFF, beta_h, 0, hd);

        float gamma_h = 0.0f;
        if (p == 0) gamma_h = gamma_flat[(b * T + t) * nh + h];
        gamma_h = __shfl_sync(0xFFFFFFFF, gamma_h, 0, hd);

        // Per-thread unique values
        int x_idx = (b * T + t) * d_inner + h * hd + p;
        float x_val = x_flat[x_idx];
        float v_prev = v_state[b * nh * hd + h * hd + p];

        float y_val = d_skip * x_val;

        for (int n = 0; n < ds; n++) {
            // O1: k_cur[n], k_prev[n], q_cur[n] broadcast from p=0
            float kc_n = 0.0f, kp_n = 0.0f, qc_n = 0.0f;
            if (p == 0) {
                kc_n = k_flat[(b * T + t) * nh * ds + h * ds + n];
                kp_n = k_state[b * nh * ds + h * ds + n];
                qc_n = q_flat[(b * T + t) * nh * ds + h * ds + n];
            }
            kc_n = __shfl_sync(0xFFFFFFFF, kc_n, 0, hd);
            kp_n = __shfl_sync(0xFFFFFFFF, kp_n, 0, hd);
            qc_n = __shfl_sync(0xFFFFFFFF, qc_n, 0, hd);

            // Trapezoidal recurrence
            h_local[n] = alpha_h * h_local[n] + beta_h * v_prev * kp_n + gamma_h * x_val * kc_n;
            y_val += h_local[n] * qc_n;
        }

        // Write output
        y_out[x_idx] = y_val;

        // Save h AFTER step t (O5 transposed layout)
        for (int n = 0; n < ds; n++) {
            int hs_idx = (t + 1) * nhd_ds + n * d_inner + h * hd + p;
            h_saved[b * (T + 1) * nhd_ds + hs_idx] = h_local[n];
        }

        // Update k_state: p=0 writes
        if (p == 0) {
            for (int n = 0; n < ds; n++)
                k_state[b * nh * ds + h * ds + n] = k_flat[(b * T + t) * nh * ds + h * ds + n];
        }

        // Update v_state: each p writes
        v_state[b * nh * hd + h * hd + p] = x_val;
    }

    // Write back final SSM state to persistent buffer
    for (int n = 0; n < ds; n++)
        ssm_state[h_base + n] = h_local[n];
}

// ======================== BURN-IN FORWARD TYPED (bf16/f16) ========================
//
// Mirrors `m3_burnin_fwd` exactly for mixed-precision training. x/k/q and
// y_out are typed (T_ACT); persistent state (ssm_state, k_state, v_state)
// and backward saves (h_saved, k_prev_saved, v_prev_saved) stay f32 per
// BPTT precision invariant; D + alpha/beta/gamma stay f32 (small
// coefficients, non-linear activations). All recurrence math executes in
// f32 internally; typed I/O only at load (`to_f`) and store (`FROM_F`).
#define DEFINE_M3_BURNIN_FWD(SUFFIX, T_ACT, FROM_F)                          \
extern "C" __global__ void m3_burnin_fwd_##SUFFIX(                           \
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
    float h_local[64];                                                       \
    if (ds > 64) return;                                                     \
    int h_base = (b * nh * hd + h * hd + p) * ds;                            \
    for (int n = 0; n < ds; n++) h_local[n] = ssm_state[h_base + n];         \
    float d_skip = 0.0f;                                                     \
    if (p == 0) d_skip = D[h];                                               \
    d_skip = __shfl_sync(0xFFFFFFFF, d_skip, 0, hd);                         \
    for (int n = 0; n < ds; n++) {                                           \
        int hs_idx = n * d_inner + h * hd + p;                               \
        h_saved[b * (T + 1) * nhd_ds + hs_idx] = h_local[n];                 \
    }                                                                        \
    for (int t = 0; t < T; t++) {                                            \
        if (p == 0) {                                                        \
            for (int n = 0; n < ds; n++)                                     \
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
        alpha_h = __shfl_sync(0xFFFFFFFF, alpha_h, 0, hd);                   \
        beta_h  = __shfl_sync(0xFFFFFFFF, beta_h,  0, hd);                   \
        gamma_h = __shfl_sync(0xFFFFFFFF, gamma_h, 0, hd);                   \
        int x_idx = (b * T + t) * d_inner + h * hd + p;                      \
        float x_val = to_f(x_flat[x_idx]);                                   \
        float v_prev = v_state[b * nh * hd + h * hd + p];                    \
        float y_val = d_skip * x_val;                                        \
        for (int n = 0; n < ds; n++) {                                       \
            float kc_n = 0.0f, kp_n = 0.0f, qc_n = 0.0f;                     \
            if (p == 0) {                                                    \
                kc_n = to_f(k_flat[(b * T + t) * nh * ds + h * ds + n]);     \
                kp_n = k_state[b * nh * ds + h * ds + n];                    \
                qc_n = to_f(q_flat[(b * T + t) * nh * ds + h * ds + n]);     \
            }                                                                \
            kc_n = __shfl_sync(0xFFFFFFFF, kc_n, 0, hd);                     \
            kp_n = __shfl_sync(0xFFFFFFFF, kp_n, 0, hd);                     \
            qc_n = __shfl_sync(0xFFFFFFFF, qc_n, 0, hd);                     \
            h_local[n] = alpha_h * h_local[n] + beta_h * v_prev * kp_n       \
                       + gamma_h * x_val * kc_n;                             \
            y_val += h_local[n] * qc_n;                                      \
        }                                                                    \
        y_out[x_idx] = FROM_F(y_val);                                        \
        for (int n = 0; n < ds; n++) {                                       \
            int hs_idx = (t + 1) * nhd_ds + n * d_inner + h * hd + p;        \
            h_saved[b * (T + 1) * nhd_ds + hs_idx] = h_local[n];             \
        }                                                                    \
        if (p == 0) {                                                        \
            for (int n = 0; n < ds; n++)                                     \
                k_state[b * nh * ds + h * ds + n] =                          \
                    to_f(k_flat[(b * T + t) * nh * ds + h * ds + n]);        \
        }                                                                    \
        v_state[b * nh * hd + h * hd + p] = x_val;                           \
    }                                                                        \
    for (int n = 0; n < ds; n++) ssm_state[h_base + n] = h_local[n];         \
}

DEFINE_M3_BURNIN_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M3_BURNIN_FWD(f16,  __half,        from_f_f16)

// ======================== BURN-IN FORWARD NOSAVE (target network) ========================

// m3_burnin_fwd_nosave: Target network forward -- no activation saves for backward.
//
// Identical recurrence to m3_burnin_fwd but skips h_saved/k_prev_saved/v_prev_saved writes.
// Grid: (B, nh, 1), Block: (hd, 1, 1).
extern "C" __global__ void m3_burnin_fwd_nosave(
    // In/Out: persistent state (mutated)
    float* ssm_state,       // [B * nh * hd * ds]
    float* k_state,         // [B * nh * ds]
    float* v_state,         // [B * nh * hd]
    // Output
    float* y_out,           // [B * T * d_inner]
    // Inputs (pre-computed by shared ops, flat over T)
    const float* x_flat,      // [B * T * d_inner]
    const float* k_flat,      // [B * T * nh * ds]
    const float* q_flat,      // [B * T * nh * ds]
    const float* alpha_flat,  // [B * T * nh]
    const float* beta_flat,   // [B * T * nh]
    const float* gamma_flat,  // [B * T * nh]
    const float* D,           // [nh]
    int batch, int T, int nh, int hd, int ds
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int p = threadIdx.x;
    if (b >= batch || h >= nh || p >= hd) return;

    int d_inner = nh * hd;

    // Load h_local from persistent state into registers
    float h_local[64];
    if (ds > 64) return;
    int h_base = (b * nh * hd + h * hd + p) * ds;
    for (int n = 0; n < ds; n++)
        h_local[n] = ssm_state[h_base + n];

    // O1: D[h] broadcast once before T loop
    float d_skip = 0.0f;
    if (p == 0) d_skip = D[h];
    d_skip = __shfl_sync(0xFFFFFFFF, d_skip, 0, hd);

    for (int t = 0; t < T; t++) {
        // O1: alpha, beta, gamma broadcast from p=0
        float alpha_h = 0.0f;
        if (p == 0) alpha_h = alpha_flat[(b * T + t) * nh + h];
        alpha_h = __shfl_sync(0xFFFFFFFF, alpha_h, 0, hd);

        float beta_h = 0.0f;
        if (p == 0) beta_h = beta_flat[(b * T + t) * nh + h];
        beta_h = __shfl_sync(0xFFFFFFFF, beta_h, 0, hd);

        float gamma_h = 0.0f;
        if (p == 0) gamma_h = gamma_flat[(b * T + t) * nh + h];
        gamma_h = __shfl_sync(0xFFFFFFFF, gamma_h, 0, hd);

        // Per-thread unique values
        int x_idx = (b * T + t) * d_inner + h * hd + p;
        float x_val = x_flat[x_idx];
        float v_prev = v_state[b * nh * hd + h * hd + p];

        float y_val = d_skip * x_val;

        for (int n = 0; n < ds; n++) {
            // O1: k_cur[n], k_prev[n], q_cur[n] broadcast from p=0
            float kc_n = 0.0f, kp_n = 0.0f, qc_n = 0.0f;
            if (p == 0) {
                kc_n = k_flat[(b * T + t) * nh * ds + h * ds + n];
                kp_n = k_state[b * nh * ds + h * ds + n];
                qc_n = q_flat[(b * T + t) * nh * ds + h * ds + n];
            }
            kc_n = __shfl_sync(0xFFFFFFFF, kc_n, 0, hd);
            kp_n = __shfl_sync(0xFFFFFFFF, kp_n, 0, hd);
            qc_n = __shfl_sync(0xFFFFFFFF, qc_n, 0, hd);

            // Trapezoidal recurrence
            h_local[n] = alpha_h * h_local[n] + beta_h * v_prev * kp_n + gamma_h * x_val * kc_n;
            y_val += h_local[n] * qc_n;
        }

        // Write output
        y_out[x_idx] = y_val;

        // Update k_state: p=0 writes
        if (p == 0) {
            for (int n = 0; n < ds; n++)
                k_state[b * nh * ds + h * ds + n] = k_flat[(b * T + t) * nh * ds + h * ds + n];
        }

        // Update v_state: each p writes
        v_state[b * nh * hd + h * hd + p] = x_val;
    }

    // Write back final SSM state to persistent buffer
    for (int n = 0; n < ds; n++)
        ssm_state[h_base + n] = h_local[n];
}

// ======================== BACKWARD ========================

// m3_backward_seq: Sequential BPTT for Mamba-3 trapezoidal recurrence.
//
// Grid: (B, nh, 1), Block: (hd, 1, 1).
// Each thread reverse-loops T, carrying d_h[ds] in registers.
// Also carries d_k_carry[ds] and d_v_carry for cross-timestep gradient flow.
//
// Cross-timestep gradient paths:
//   k_prev[t] = k_cur[t-1]  =>  d_k_carry accumulates at t, flushes to d_k[t-1]
//   v_prev[t] = x[t-1]      =>  d_v_carry accumulates at t, adds to d_x[t-1]
//
// Carry flush timing (reverse loop, processing t = T-1, T-2, ..., 0):
//   1. At the START of processing timestep t, flush d_k_carry (from t+1) to d_k[t]
//      because k_prev[t+1] = k_cur[t].
//   2. Accumulate new d_k_carry from k_prev[t] gradient during inner loop.
//   3. After loop: remaining d_k_carry is for k_prev[0] = initial state (discarded).
//
// Gradient outputs:
//   d_x: direct write (unique per thread per timestep)
//   d_k, d_q, d_alpha, d_beta, d_gamma: atomicAdd after warp reduction over p
//   d_D_local: per-thread accumulator, reduced by m3_reduce_d_D kernel
//
// From the forward recurrence:
//   h[p,n] = alpha * h_prev[p,n] + beta * v_prev[p] * k_prev[n] + gamma * x[p] * k_cur[n]
//   y[p] = sum_n(h[p,n] * q[n]) + D * x[p]
extern "C" __global__ void m3_backward_seq(
    // Saved activations from forward
    const float* h_saved,       // [B * (T+1) * nh * hd * ds] (O5 transposed)
    const float* k_prev_saved,  // [B * T * nh * ds]
    const float* v_prev_saved,  // [B * T * nh * hd]
    const float* x_flat,        // [B * T * d_inner]
    const float* k_flat,        // [B * T * nh * ds] (k_cur at each t)
    const float* q_flat,        // [B * T * nh * ds] (q_cur at each t)
    const float* alpha_flat,    // [B * T * nh]
    const float* beta_flat,     // [B * T * nh]
    const float* gamma_flat,    // [B * T * nh]
    const float* D,             // [nh]
    // Incoming gradient
    const float* d_y_flat,      // [B * T * d_inner]
    // Output gradients
    float* d_x,         // [B * T * d_inner] (direct write per thread)
    float* d_k,         // [B * T * nh * ds] (atomicAdd after warp reduce)
    float* d_q,         // [B * T * nh * ds] (atomicAdd after warp reduce)
    float* d_alpha,     // [B * T * nh] (atomicAdd after warp reduce)
    float* d_beta,      // [B * T * nh] (atomicAdd after warp reduce)
    float* d_gamma,     // [B * T * nh] (atomicAdd after warp reduce)
    float* d_D_local,   // [B * d_inner] per-thread, reduced by m3_reduce_d_D
    int batch, int T, int nh, int hd, int ds
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int p = threadIdx.x;
    if (b >= batch || h >= nh || p >= hd) return;

    int d_inner = nh * hd;
    int nhd_ds = d_inner * ds;

    // Warp-reduce mask: only `hd` lanes are launched (block_dim = hd, hd ≤ 32).
    // Hardcoded 0xFFFFFFFF = UB per CUDA Programming Guide §B.15.1.
    unsigned warp_mask = (hd >= 32) ? 0xFFFFFFFFu : ((1u << hd) - 1u);

    // O1: D[h] broadcast once
    float d_skip = 0.0f;
    if (p == 0) d_skip = D[h];
    d_skip = __shfl_sync(warp_mask, d_skip, 0, hd);

    // d_h: BPTT hidden state gradient carried backward through time
    float d_h_reg[64];
    if (ds > 64) return;
    for (int n = 0; n < ds; n++)
        d_h_reg[n] = 0.0f;

    // d_k_carry: gradient for k_prev accumulated at timestep (t+1).
    // Flushed to d_k[t] at the start of processing timestep t.
    // Only lane p=0 accumulates (after warp reduce over p).
    float d_k_carry[64];
    for (int n = 0; n < ds; n++)
        d_k_carry[n] = 0.0f;

    // d_v_carry: gradient for v_prev accumulated at timestep (t+1).
    // Added to d_x[t] since v_prev[t+1] = x[t]. Per-thread (unique per p).
    float d_v_carry = 0.0f;

    // Per-thread d_D accumulator (summed over T, reduced later)
    float sum_d_D = 0.0f;

    int h_b_base = b * (T + 1) * nhd_ds;

    for (int t = T - 1; t >= 0; t--) {
        // --- Flush d_k_carry from previous iteration ---
        // d_k_carry holds gradient for k_prev[t+1] = k_cur[t], so write to d_k[t].
        // Skip at first iteration (t == T-1): carry is zero.
        if (t < T - 1 && p == 0) {
            for (int n = 0; n < ds; n++) {
                if (d_k_carry[n] != 0.0f)
                    atomicAdd(&d_k[(b * T + t) * nh * ds + h * ds + n], d_k_carry[n]);
                d_k_carry[n] = 0.0f;
            }
        }

        int x_idx = (b * T + t) * d_inner + h * hd + p;
        float dy_val = d_y_flat[x_idx];
        float x_val = x_flat[x_idx];
        float v_prev = v_prev_saved[(b * T + t) * d_inner + h * hd + p];

        // O1: alpha, beta, gamma broadcast from p=0
        float alpha_h = 0.0f;
        if (p == 0) alpha_h = alpha_flat[(b * T + t) * nh + h];
        alpha_h = __shfl_sync(0xFFFFFFFF, alpha_h, 0, hd);

        float beta_h = 0.0f;
        if (p == 0) beta_h = beta_flat[(b * T + t) * nh + h];
        beta_h = __shfl_sync(0xFFFFFFFF, beta_h, 0, hd);

        float gamma_h = 0.0f;
        if (p == 0) gamma_h = gamma_flat[(b * T + t) * nh + h];
        gamma_h = __shfl_sync(0xFFFFFFFF, gamma_h, 0, hd);

        // Skip connection: d_D accumulation + d_x from D path
        sum_d_D += dy_val * x_val;
        float d_x_val = d_skip * dy_val;

        // Add v_prev carry from t+1: v_prev[t+1] = x[t]
        d_x_val += d_v_carry;

        // Per-timestep accumulators (reduced across p at end of inner loop)
        float d_alpha_acc = 0.0f;
        float d_beta_acc = 0.0f;
        float d_gamma_acc = 0.0f;
        float d_v_prev_acc = 0.0f;

        for (int n = 0; n < ds; n++) {
            // O1: k_cur[n], k_prev[n], q_cur[n] broadcast from p=0
            float kc_n = 0.0f, kp_n = 0.0f, qc_n = 0.0f;
            if (p == 0) {
                kc_n = k_flat[(b * T + t) * nh * ds + h * ds + n];
                kp_n = k_prev_saved[(b * T + t) * nh * ds + h * ds + n];
                qc_n = q_flat[(b * T + t) * nh * ds + h * ds + n];
            }
            kc_n = __shfl_sync(0xFFFFFFFF, kc_n, 0, hd);
            kp_n = __shfl_sync(0xFFFFFFFF, kp_n, 0, hd);
            qc_n = __shfl_sync(0xFFFFFFFF, qc_n, 0, hd);

            // O5: coalesced h_saved reads
            float h_prev_n = h_saved[h_b_base + t * nhd_ds + n * d_inner + h * hd + p];
            float h_curr_n = h_saved[h_b_base + (t + 1) * nhd_ds + n * d_inner + h * hd + p];

            // From y[p] = sum_n(h[p,n] * q[n]): d_h += d_y * q
            d_h_reg[n] += dy_val * qc_n;

            // d_q[n] = sum_p(d_y[p] * h_curr[p,n]): warp reduce over p
            float d_q_val = dy_val * h_curr_n;
            for (int off = hd / 2; off > 0; off >>= 1)
                d_q_val += __shfl_down_sync(0xFFFFFFFF, d_q_val, off, hd);
            if (p == 0)
                atomicAdd(&d_q[(b * T + t) * nh * ds + h * ds + n], d_q_val);

            // Gradients through trapezoidal recurrence
            float dh_n = d_h_reg[n];

            // d_alpha += dh * h_prev  (scalar, reduce over p and n)
            d_alpha_acc += dh_n * h_prev_n;

            // d_beta += dh * v_prev * k_prev  (scalar, reduce over p and n)
            d_beta_acc += dh_n * v_prev * kp_n;

            // d_gamma += dh * x * k_cur  (scalar, reduce over p and n)
            d_gamma_acc += dh_n * x_val * kc_n;

            // d_v_prev[p] += dh * beta * k_prev[n]  (sum over n, per-p -> d_x[t-1])
            d_v_prev_acc += dh_n * beta_h * kp_n;

            // d_k_prev[n] += dh * beta * v_prev  (sum over p -> warp reduce -> carry)
            float d_kp_val = dh_n * beta_h * v_prev;
            for (int off = hd / 2; off > 0; off >>= 1)
                d_kp_val += __shfl_down_sync(0xFFFFFFFF, d_kp_val, off, hd);
            if (p == 0)
                d_k_carry[n] += d_kp_val;

            // d_x[t,p] += dh * gamma * k_cur[n]  (sum over n, per-p)
            d_x_val += dh_n * gamma_h * kc_n;

            // d_k_cur[n] += dh * gamma * x  (sum over p -> warp reduce -> d_k[t])
            float d_kc_val = dh_n * gamma_h * x_val;
            for (int off = hd / 2; off > 0; off >>= 1)
                d_kc_val += __shfl_down_sync(0xFFFFFFFF, d_kc_val, off, hd);
            if (p == 0)
                atomicAdd(&d_k[(b * T + t) * nh * ds + h * ds + n], d_kc_val);

            // BPTT: propagate d_h backward through alpha
            d_h_reg[n] = alpha_h * dh_n;
        }

        // Write d_x for this timestep (includes D skip + v_prev carry + gamma*k_cur path)
        d_x[x_idx] = d_x_val;

        // d_v_carry for next iteration: v_prev[t] = x[t-1], so this flows to d_x[t-1]
        d_v_carry = d_v_prev_acc;

        // Warp reduce d_alpha, d_beta, d_gamma over p, then atomicAdd
        for (int off = hd / 2; off > 0; off >>= 1)
            d_alpha_acc += __shfl_down_sync(0xFFFFFFFF, d_alpha_acc, off, hd);
        if (p == 0)
            atomicAdd(&d_alpha[(b * T + t) * nh + h], d_alpha_acc);

        for (int off = hd / 2; off > 0; off >>= 1)
            d_beta_acc += __shfl_down_sync(0xFFFFFFFF, d_beta_acc, off, hd);
        if (p == 0)
            atomicAdd(&d_beta[(b * T + t) * nh + h], d_beta_acc);

        for (int off = hd / 2; off > 0; off >>= 1)
            d_gamma_acc += __shfl_down_sync(0xFFFFFFFF, d_gamma_acc, off, hd);
        if (p == 0)
            atomicAdd(&d_gamma[(b * T + t) * nh + h], d_gamma_acc);
    }

    // After loop: d_k_carry holds gradient for k_prev[0] = initial k_state (not trained).
    // d_v_carry holds gradient for v_prev[0] = initial v_state (not trained).
    // Both are discarded.

    // Store per-thread d_D sum for later reduction by m3_reduce_d_D
    d_D_local[b * d_inner + h * hd + p] = sum_d_D;
}

// ======================== REDUCTION KERNELS ========================

// Reduce d_D_local from [B * d_inner] to [nh] by summing across batch and headdim.
// O2 Warp-Parallel: 1 warp (32 threads) per head.
// Launch: grid_dim=(nh, 1, 1), block_dim=(32, 1, 1).
extern "C" __global__ void m3_reduce_d_D(
    float* d_D_out,           // [nh]
    const float* d_D_local,   // [B * d_inner]
    int batch, int nh, int hd
) {
    int head = blockIdx.x;
    if (head >= nh) return;

    int lane = threadIdx.x; // 0..31
    int d_inner = nh * hd;
    int total = batch * hd;

    float sum = 0.0f;
    for (int i = lane; i < total; i += 32) {
        int b_i = i / hd;
        int p_i = i % hd;
        sum += d_D_local[b_i * d_inner + head * hd + p_i];
    }

    // Full warp reduction
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) atomicAdd(&d_D_out[head], sum);
}
