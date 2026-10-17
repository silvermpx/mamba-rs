// The angle gradients' reverse sum of the SISO backward as it was released
// before the batched-load kernel replaced it: a thread a (batch, head,
// angle) column, one step's loads at a time. The batched kernel is held to
// it bit for bit. The typed prelude and the packed-row helpers precede
// this fixture.

#ifndef PI
#define PI 3.141592653589793f
#endif

// ============================================================================
// 6b. m3_angle_dt_bwd_seq -- Batched reverse cumsum for angle gradients
// ============================================================================
//
// Batched version of angle_dt_bwd that mirrors m3_angle_dt_fwd_seq.
// Each block handles one batch element; threads loop backward over T.
//
// Forward: angle_cumsum[bt,h,a] = sum_{s=0..t} delta[s,h,a]
//          delta[s,h,a] = tanh(angles_raw[s*n_angles+a]) * PI * dt[s*nh+h]
//
// Backward (reverse cumsum per batch element):
//   d_delta[t,h,a] = sum_{s=t}^{T-1} d_angle_cumsum[s,h,a]
//   d_angles_raw[bt,a] += d_delta * PI * dt_val * sech^2(raw)
//   d_dt_angle[bt,h]   += d_delta * PI * tanh(raw)
//
// Grid: (B, ceil(nh*n_angles/256)). Block: (min(256, nh*n_angles)).
// The no-atomics partials rule: two-pass deterministic variant. Stage 1 per-(b,h,a)
// thread writes per-t contributions to two transposed scratch tensors (no
// atomicAdd):
//   contrib_angles[h, b, t, a] — reduce over h to get d_angles_raw[bt, a]
//   contrib_dt[a, b, t, h]    — reduce over a to get d_dt_angle[bt, h]
// Stage 2 (caller): two reduce_sum_axis0 launches finalise the outputs. In
// a packed row (`seq_idx`) the reverse sum restarts after a sequence's
// first token: its angles never reach the sequence before.
#define DEFINE_M3_ANGLE_DT_BWD_SEQ(NAME, IDX)                                  \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ contrib_angles,      /* [nh * B*T*n_angles] OUTPUT */  \
    float* __restrict__ contrib_dt,          /* [n_angles * B*T*nh] OUTPUT */  \
    const float* __restrict__ d_angle_cumsum,/* [B*T * nh * n_angles] */       \
    const float* __restrict__ angles_raw,    /* [B*T * n_angles] */            \
    const float* __restrict__ dt_arr,        /* [B*T * nh] */                  \
    int B, int T, int nh, int n_angles,                                        \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    IDX b = blockIdx.x;                                                        \
    if (b >= B) return;                                                        \
    int idx = (y_first + blockIdx.y) * blockDim.x + threadIdx.x;               \
    int total_per_env = nh * n_angles;                                         \
    if (idx >= total_per_env) return;                                          \
                                                                               \
    int h = idx / n_angles;                                                    \
    int a = idx % n_angles;                                                    \
                                                                               \
    IDX btn = (IDX)B * T * n_angles;                                           \
    IDX btnh = (IDX)B * T * nh;                                                \
                                                                               \
    /* Reverse cumsum over T for this (batch, head, angle) */                  \
    float running = 0.0f;                                                      \
    for (int t = T - 1; t >= 0; t--) {                                         \
        IDX bt = b * T + t;                                                    \
        if (t + 1 < T && vl_starts(seq_idx, b, T, t + 1)) running = 0.0f;      \
        running += d_angle_cumsum[bt * total_per_env + h * n_angles + a];      \
        float d_delta = running;                                               \
                                                                               \
        float raw = angles_raw[bt * n_angles + a];                             \
        float th = tanhf(raw);                                                 \
        float dt_val = dt_arr[bt * nh + h];                                    \
        float sech2 = 1.0f - th * th;                                          \
                                                                               \
        /* contrib_angles[h, b, t, a] */                                       \
        contrib_angles[h * btn + bt * n_angles + a] = d_delta * PI * dt_val * sech2; \
        /* contrib_dt[a, b, t, h] */                                           \
        contrib_dt[a * btnh + bt * nh + h] = d_delta * PI * th;                \
    }                                                                          \
}

DEFINE_M3_ANGLE_DT_BWD_SEQ(m3_reference_angle_dt_bwd_seq, int)
