// Frozen three-launch state pass of the SISO chunked forward: the entering
// fold, the inter-chunk pass and the window-final writeback, as they were
// released before one fused launch replaced them. The fused state pass is
// held bit for bit to these. The typed prelude precedes this fixture.

#ifndef FAST_EXP
#define FAST_EXP(x) exp2f((x) * LOG2E)
#endif

// The entering fold of a carried window: the trapezoidal beta term reaches
// one step back across the seam.
//   h_enter[b,h,p,n] += v_state[b,h,p] * k_state[b,h,n] * dt[b,0,h] * (1 - trap[b,0,h])
// Grid: (B, nh, ceil(hd*ds / BLOCK)), Block: (min(hd*ds, 256))
extern "C" __global__ void m3_reference_entering_state(
    float* __restrict__ h_enter,
    const float* __restrict__ k_state,
    const float* __restrict__ v_state,
    const float* __restrict__ dt,
    const float* __restrict__ trap,
    int batch, int nh, int hd, int ds, int T
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int pd = blockIdx.z * blockDim.x + threadIdx.x;
    int dim = hd * ds;
    if (b >= batch || pd >= dim) return;
    int p = pd / ds;
    int n = pd % ds;
    float dt0 = dt[b * T * nh + h];
    float trap0 = trap[b * T * nh + h];
    float v = v_state[(b * nh + h) * hd + p];
    float k = k_state[(b * nh + h) * ds + n];
    h_enter[(b * nh + h) * dim + pd] += v * k * dt0 * (1.0f - trap0);
}

// The exclusive prefix scan over chunks: states[c] becomes the state
// entering chunk c, seeded from init_states when it is not null.
// Grid: (B, nh, ceil(hd*ds / BLOCK)), Block: (min(hd*ds, 256))
extern "C" __global__ void m3_reference_state_passing(
    float* __restrict__ states,
    float* __restrict__ final_states,
    const float* __restrict__ dA_cumsum,
    const float* __restrict__ init_states,
    int batch, int n_chunks, int nh, int hd, int ds, int chunk_size, int T
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int pd = blockIdx.z * blockDim.x + threadIdx.x;
    int dim = hd * ds;
    if (pd >= dim) return;
    float state = init_states ? init_states[b * nh * dim + h * dim + pd] : 0.0f;
    int state_idx = b * n_chunks * nh * dim + h * dim + pd;
    int cs_idx = b * n_chunks * nh + h;
    float contrib = states[state_idx];
    int chunk_end_0 = chunk_size > T ? T : chunk_size;
    float decay = FAST_EXP(__ldg(&dA_cumsum[cs_idx * chunk_size + chunk_end_0 - 1]));
    for (int c = 0; c < n_chunks; c++) {
        float next_contrib = 0.0f;
        float next_decay = 0.0f;
        if (c + 1 < n_chunks) {
            next_contrib = states[state_idx + nh * dim];
            int chunk_start_n = (c + 1) * chunk_size;
            int chunk_end_n = chunk_start_n + chunk_size;
            if (chunk_end_n > T) chunk_end_n = T;
            int last_elem_n = chunk_end_n - chunk_start_n - 1;
            next_decay = FAST_EXP(
                __ldg(&dA_cumsum[(cs_idx + nh) * chunk_size + last_elem_n]));
        }
        states[state_idx] = state;
        state = decay * state + contrib;
        contrib = next_contrib;
        decay = next_decay;
        state_idx += nh * dim;
        cs_idx += nh;
    }
    int final_idx = b * nh * dim + h * dim + pd;
    final_states[final_idx] = state;
}

// The window-final writeback: the SSM state from the pass's final states,
// and K and V from the window's last step.
// Grid: (B, nh, 1), Block: (max(hd, ds), 1, 1)
#define DEFINE_M3_REFERENCE_WRITEBACK(NAME, T_ACT)                             \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ ssm_state,                                             \
    float* __restrict__ k_state,                                               \
    float* __restrict__ v_state,                                               \
    const float* __restrict__ final_states,                                    \
    const T_ACT* __restrict__ k_flat,                                          \
    const T_ACT* __restrict__ x_flat,                                          \
    int batch, int T, int nh, int hd, int ds                                   \
) {                                                                            \
    int b = blockIdx.x;                                                        \
    int h = blockIdx.y;                                                        \
    int p = threadIdx.x;                                                       \
    if (b >= batch || h >= nh) return;                                         \
    int d_inner = nh * hd;                                                     \
    int dim = hd * ds;                                                         \
    if (p < hd) {                                                              \
        for (int n = 0; n < ds; n++) {                                         \
            int idx = b * nh * dim + h * dim + p * ds + n;                     \
            ssm_state[idx] = final_states[idx];                                \
        }                                                                      \
    }                                                                          \
    if (p < ds) {                                                              \
        int src_idx = (b * T + (T - 1)) * nh * ds + h * ds + p;                \
        int dst_idx = b * nh * ds + h * ds + p;                                \
        k_state[dst_idx] = to_f(k_flat[src_idx]);                              \
    }                                                                          \
    if (p < hd) {                                                              \
        int src_idx = (b * T + (T - 1)) * d_inner + h * hd + p;                \
        int dst_idx = b * nh * hd + h * hd + p;                                \
        v_state[dst_idx] = to_f(x_flat[src_idx]);                              \
    }                                                                          \
}

DEFINE_M3_REFERENCE_WRITEBACK(m3_reference_writeback_f32, float)
DEFINE_M3_REFERENCE_WRITEBACK(m3_reference_writeback_bf16, __nv_bfloat16)
DEFINE_M3_REFERENCE_WRITEBACK(m3_reference_writeback_f16, __half)
