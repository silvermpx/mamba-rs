// The angles' sequential sum of the SISO forward as it was released, before
// the batched-load kernel replaced it: a thread a (batch, head, angle)
// column, one step's loads at a time and fmod's wrap every step. The batched
// kernel is held to it bit for bit. The typed prelude and the packed-row
// helpers precede this fixture.
#ifndef PI
#define PI 3.141592653589793f
#endif
// Sequential angle accumulation for training: B envs, T timesteps each.
// Grid: (B, ceil(nh*n_angles/256), 1). Block: (256, 1, 1).
// angles_raw: [B*T*n_angles], dt_arr: [B*T*nh], angle_state: [B*nh*n_angles]
// angle_cumsum: [B*T*nh*n_angles] — saves cumulative state at each timestep.
// A sequence of a packed row (`seq_idx`, see _varlen.cuh) sums its own
// angles from zero.
#define DEFINE_M3_ANGLE_DT_FWD_SEQ(NAME, IDX)                                  \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ angle_cumsum,   /* [B*T * nh * n_angles] */            \
    float* __restrict__ angle_state,    /* [B * nh * n_angles] -- persistent per-env */ \
    const float* __restrict__ angles_raw, /* [B*T * n_angles] */               \
    const float* __restrict__ dt_arr,    /* [B*T * nh] */                      \
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
    double state = (double)angle_state[b * total_per_env + h * n_angles + a];  \
    const double TWO_PI_64 = 6.283185307179586;                                \
                                                                               \
    for (int t = 0; t < T; t++) {                                              \
        IDX bt = b * T + t;                                                    \
        if (vl_starts(seq_idx, b, T, t)) state = 0.0;                          \
        float raw = angles_raw[bt * n_angles + a];                             \
        float dt_val = dt_arr[bt * nh + h];                                    \
        double delta = (double)(tanhf(raw) * PI * dt_val);                     \
        state += delta;                                                        \
        state = fmod(state, TWO_PI_64);                                        \
        if (state < 0.0) state += TWO_PI_64;                                   \
                                                                               \
        angle_cumsum[bt * total_per_env + h * n_angles + a] = (float)state;    \
    }                                                                          \
                                                                               \
    angle_state[b * total_per_env + h * n_angles + a] = (float)state;          \
}

DEFINE_M3_ANGLE_DT_FWD_SEQ(m3_reference_angle_dt_fwd_seq, int)
