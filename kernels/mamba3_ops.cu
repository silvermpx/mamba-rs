// Mamba-3 SISO shared CUDA kernels.
//
// 8 kernels covering the non-SGEMM phases of the Mamba-3 SISO pipeline:
// split, BCNorm (fwd/bwd), bias add, angle accumulation (fwd/bwd), RoPE (fwd/bwd).
//
// CPU reference: mamba3_siso/cpu/forward.rs (phases F3-F5)
// T=1 step: mamba3_siso/cpu/inference.rs
// Paper: Lahoti et al., "Mamba-3", ICLR 2026 (arXiv 2603.15569)

#include "_typed_prelude.cuh"

#ifndef PI
#define PI 3.141592653589793f
#endif
#ifndef TWO_PI
#define TWO_PI 6.283185307179586f
#endif
#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

// Fast exp via exp2f: exp(x) = exp2(x * log2(e))
// Single PTX instruction on NVIDIA GPUs. Source: Tri Dao selective_scan_fwd_kernel.cuh.
#ifndef FAST_EXP
#define FAST_EXP(x) exp2f((x) * LOG2E)
#endif

// ============================================================================
// 1. m3_split -- 8-way split from in_proj + fused softplus/sigmoid
// ============================================================================
//
// Splits the in_proj output into 8 components and applies fused activations:
//   - z, x:        pass-through
//   - B_raw, C_raw: pass-through
//   - dt:          softplus(dd_dt + dt_bias)
//   - a_val:       -heavy_tail(dd_A), clamped to max = -a_floor
//   - trap:        sigmoid(trap_raw)
//   - angles:      pass-through
//
// in_proj layout (contiguous per sample):
//   [z: d_inner | x: d_inner | B: ng*ds | C: ng*ds | dd_dt: nh | dd_A: nh | trap: nh | angles: n_angles]
//   Total = in_proj_dim
//
// Input:  proj[N * in_proj_dim]
// Outputs: z[N*di], x[N*di], B_raw[N*ng*ds], C_raw[N*ng*ds],
//          dt[N*nh] (post-softplus), a_val[N*nh] (neg-heavy-tail clamped),
//          trap[N*nh] (post-sigmoid), angles[N*n_angles] (pass-through),
//          dd_dt_raw[N*nh] (saved for backward), dd_a_raw[N*nh] (saved for backward),
//          trap_raw[N*nh] (saved for backward)
// Params: dt_bias[nh], a_floor (scalar)
// Grid: ceil(N * in_proj_dim / 256) blocks, 256 threads
// ---------------------------------------------------------------------------
// Reference A-activation (state-spaces/mamba `heavy_tail_activation`):
//   heavy_tail(x) = 1 + x        for x >= 0
//                 = 1 / (1 - x)  for x <  0
// Always positive; polynomial decay for negative inputs (softplus saturates
// exponentially — different decay-spectrum distribution and gradient tails,
// i.e. different training dynamics from the published model).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float m3_heavy_tail(float x) {
    return (x >= 0.0f) ? (1.0f + x) : (1.0f / (1.0f - x));
}
__device__ __forceinline__ float m3_heavy_tail_deriv(float x) {
    if (x >= 0.0f) return 1.0f;
    float d = 1.0f - x;
    return 1.0f / (d * d);
}

#define DEFINE_M3_SPLIT_F32(NAME, IDX)                                         \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ z,          /* [N * di] */                             \
    float* __restrict__ x,          /* [N * di] */                             \
    float* __restrict__ B_raw,      /* [N * ng * ds] */                        \
    float* __restrict__ C_raw,      /* [N * ng * ds] */                        \
    float* __restrict__ dt,         /* [N * nh] -- post-softplus */            \
    float* __restrict__ a_val,      /* [N * nh] -- -heavy_tail(dd_A), clamped */ \
    float* __restrict__ trap,       /* [N * nh] -- post-sigmoid */             \
    float* __restrict__ angles,     /* [N * n_angles] */                       \
    float* __restrict__ dd_dt_raw,  /* [N * nh] -- saved for backward */       \
    float* __restrict__ dd_a_raw,   /* [N * nh] -- saved for backward */       \
    float* __restrict__ trap_raw,   /* [N * nh] -- saved for backward */       \
    const float* __restrict__ proj, /* [N * in_proj_dim] */                    \
    const float* __restrict__ dt_bias, /* [nh] */                              \
    float a_floor,                                                             \
    int N, int di, int ng, int ds, int nh, int n_angles                        \
) {                                                                            \
    int in_proj_dim = 2 * di + 2 * ng * ds + 3 * nh + n_angles;                \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    IDX total = (IDX)N * in_proj_dim;                                          \
    if (idx >= total) return;                                                  \
                                                                               \
    IDX sample = idx / in_proj_dim;                                            \
    int col = (int)(idx % in_proj_dim);                                        \
    float val = proj[idx];                                                     \
                                                                               \
    int off0 = di;                     /* end of z */                          \
    int off1 = 2 * di;                 /* end of x */                          \
    int ng_ds = ng * ds;                                                       \
    int off2 = off1 + ng_ds;           /* end of B */                          \
    int off3 = off2 + ng_ds;           /* end of C */                          \
    int off4 = off3 + nh;              /* end of dd_dt */                      \
    int off5 = off4 + nh;              /* end of dd_A */                       \
    int off6 = off5 + nh;              /* end of trap */                       \
                                                                               \
    if (col < off0) {                                                          \
        /* z: pass-through (nullable: the serve prefill reads z inside */      \
        /* the projection in place and skips this copy) */                     \
        if (z) z[sample * di + col] = val;                                     \
    } else if (col < off1) {                                                   \
        /* x: pass-through */                                                  \
        x[sample * di + (col - off0)] = val;                                   \
    } else if (col < off2) {                                                   \
        /* B_raw: pass-through (nullable, as z) */                             \
        if (B_raw) B_raw[sample * ng_ds + (col - off1)] = val;                 \
    } else if (col < off3) {                                                   \
        /* C_raw: pass-through (nullable, as z) */                             \
        if (C_raw) C_raw[sample * ng_ds + (col - off2)] = val;                 \
    } else if (col < off4) {                                                   \
        /* dd_dt: save raw, apply softplus(dd_dt + dt_bias) */                 \
        int h = col - off3;                                                    \
        IDX dt_idx = sample * nh + h;                                          \
        if (dd_dt_raw) dd_dt_raw[dt_idx] = val;                                \
        float biased = val + dt_bias[h];                                       \
        dt[dt_idx] = (biased > 20.0f) ? biased : log1pf(FAST_EXP(biased));     \
    } else if (col < off5) {                                                   \
        /* dd_A: save raw, apply -heavy_tail(dd_A), clamp max = -a_floor */    \
        /* (reference: state-spaces/mamba mamba3.py `heavy_tail_activation`) */ \
        int h = col - off4;                                                    \
        IDX a_idx = sample * nh + h;                                           \
        if (dd_a_raw) dd_a_raw[a_idx] = val;                                   \
        float a = -m3_heavy_tail(val);                                         \
        /* A NaN stays NaN, as the reference's clamp leaves it; fminf would */ \
        /* return the floor and hide a corrupt input behind a finite decay. */ \
        a_val[a_idx] = (a > -a_floor) ? -a_floor : a;                          \
    } else if (col < off6) {                                                   \
        /* trap: save raw, apply sigmoid */                                    \
        int h = col - off5;                                                    \
        IDX t_idx = sample * nh + h;                                           \
        if (trap_raw) trap_raw[t_idx] = val;                                   \
        trap[t_idx] = 1.0f / (1.0f + FAST_EXP(-val));                          \
    } else {                                                                   \
        /* angles: pass-through */                                             \
        int a = col - off6;                                                    \
        if (a < n_angles) {                                                    \
            angles[sample * n_angles + a] = val;                               \
        }                                                                      \
    }                                                                          \
}

DEFINE_M3_SPLIT_F32(m3_split, int)
DEFINE_M3_SPLIT_F32(m3_split_i64, long long)

// ============================================================================
// 3. bcnorm_bwd -- RMSNorm backward for B (or C)
// ============================================================================
//
// Forward: y = x * inv_rms * w, where inv_rms = 1/sqrt(mean(x^2) + eps)
// Backward (standard RMSNorm):
//   c1 = mean(x_hat * w * dy)  where x_hat = x * inv_rms
//   dx = (w * dy - x_hat * c1) * inv_rms
//   dw += dy * x_hat  (per-sample contribution, needs external reduction)
//
// Input:  d_out[N*ng*ds], B_raw[N*ng*ds], rms_val[N*ng], weight[ds]
// Output: d_B[N*ng*ds], d_weight[N*ng*ds] (per-block, reduce across blocks later)
// Grid: N * ng blocks, ds threads per block
#define DEFINE_BCNORM_BWD_F32(NAME, IDX)                                       \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ d_B,        /* [N * ng * ds] -- gradient w.r.t. B_raw */ \
    float* __restrict__ d_weight,   /* [N * ng * ds] -- per-block dw (reduce later) */ \
    const float* __restrict__ d_out,/* [N * ng * ds] -- upstream gradient */   \
    const float* __restrict__ B_raw,/* [N * ng * ds] -- saved input */         \
    const float* __restrict__ rms_val,/* [N * ng] -- saved rms from forward */ \
    const float* __restrict__ weight, /* [ds] */                               \
    int N, int ng, int ds                                                      \
) {                                                                            \
    int block_id = blockIdx.x;                                                 \
    if (block_id >= (IDX)N * ng) return;                                       \
    int d = threadIdx.x;                                                       \
    if (d >= ds) return;                                                       \
                                                                               \
    IDX base = (IDX)block_id * ds;                                             \
    float rms = rms_val[block_id];                                             \
    float inv_rms = 1.0f / fmaxf(rms, 1e-12f);                                 \
                                                                               \
    float x_val = B_raw[base + d];                                             \
    float w = weight[d];                                                       \
    float dy = d_out[base + d];                                                \
    float x_hat = x_val * inv_rms;                                             \
                                                                               \
    /* Shared memory for c1 = mean(x_hat * w * dy) */                          \
    extern __shared__ float sdata[];                                           \
    sdata[d] = x_hat * w * dy;                                                 \
    __syncthreads();                                                           \
                                                                               \
    int stride = 1;                                                            \
    while (stride < ds) stride <<= 1;                                          \
    stride >>= 1;                                                              \
    for (; stride > 0; stride >>= 1) {                                         \
        if (d < stride && (d + stride) < ds) {                                 \
            sdata[d] += sdata[d + stride];                                     \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
    float c1 = sdata[0] / (float)ds;                                           \
                                                                               \
    /* d_B = (w * dy - x_hat * c1) * inv_rms */                                \
    d_B[base + d] = (w * dy - x_hat * c1) * inv_rms;                           \
                                                                               \
    /* d_weight contribution from this block */                                \
    d_weight[base + d] = dy * x_hat;                                           \
}

DEFINE_BCNORM_BWD_F32(bcnorm_bwd, int)
DEFINE_BCNORM_BWD_F32(bcnorm_bwd_i64, long long)

// ============================================================================
// 4. bc_bias_add -- Per-head bias add for B and C (expand groups -> heads)
// ============================================================================
//
// After BCNorm, B_normed is [N * ng * ds] (shared across heads in each group).
// Bias is [nh * ds] (per-head). Output is [N * nh * ds] (expanded to per-head).
//
// For each (sample, head h, state_idx n):
//   g = h / (nh / ng)  -- group index for this head
//   B_biased[sample*nh*ds + h*ds + n] = B_normed[sample*ng*ds + g*ds + n] + bias[h*ds + n]
//
// Input:  B_normed[N * ng * ds], bias[nh * ds]
// Output: B_biased[N * nh * ds]
// Grid: ceil(N * nh * ds / 256) blocks, 256 threads
#define DEFINE_BC_BIAS_ADD(NAME, IDX)                                          \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ B_biased,       /* [N * nh * ds] */                    \
    const float* __restrict__ B_normed, /* [N * ng * ds] */                    \
    const float* __restrict__ bias,     /* [nh * ds] */                        \
    int N, int nh, int ng, int ds                                              \
) {                                                                            \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    IDX total = (IDX)N * nh * ds;                                              \
    if (idx >= total) return;                                                  \
                                                                               \
    int nh_ds = nh * ds;                                                       \
    IDX sample = idx / nh_ds;                                                  \
    int rem = (int)(idx % nh_ds);                                              \
    int h = rem / ds;                                                          \
    int n = rem % ds;                                                          \
                                                                               \
    int heads_per_group = nh / ng;                                             \
    int g = h / heads_per_group;                                               \
                                                                               \
    B_biased[idx] = B_normed[sample * ng * ds + g * ds + n] + bias[h * ds + n]; \
}

DEFINE_BC_BIAS_ADD(bc_bias_add, int)
DEFINE_BC_BIAS_ADD(bc_bias_add_i64, long long)

// Sequential angle accumulation for training: B envs, T timesteps each.
// Grid: (B, ceil(nh*n_angles/256), 1). Block: (256, 1, 1).
// angles_raw: [B*T*n_angles], dt_arr: [B*T*nh], angle_state: [B*nh*n_angles]
// angle_cumsum: [B*T*nh*n_angles] — saves cumulative state at each timestep.
#define DEFINE_M3_ANGLE_DT_FWD_SEQ(NAME, IDX)                                  \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ angle_cumsum,   /* [B*T * nh * n_angles] */            \
    float* __restrict__ angle_state,    /* [B * nh * n_angles] -- persistent per-env */ \
    const float* __restrict__ angles_raw, /* [B*T * n_angles] */               \
    const float* __restrict__ dt_arr,    /* [B*T * nh] */                      \
    int B, int T, int nh, int n_angles,                                        \
    int y_first /* this run's first y block */                                 \
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

DEFINE_M3_ANGLE_DT_FWD_SEQ(m3_angle_dt_fwd_seq, int)
DEFINE_M3_ANGLE_DT_FWD_SEQ(m3_angle_dt_fwd_seq_i64, long long)

// ============================================================================
// Chunked angle accumulation -- the parallel replacement for the
// sequential kernel above on multi-chunk windows.
// ============================================================================
//
// The sequential kernel runs ONE thread per (b, h, angle) through the
// whole window: at production shapes that is a few hundred threads on a
// hundred-SM GPU walking thousands of dependent fp64 adds with a
// per-step fmod, and it dominates the prefill profile. The chunked
// scheme keeps a deterministic fixed association -- every partial sum
// is a serial left fold written by exactly one thread, no atomics --
// while exposing (b, h, angle, chunk) parallelism:
//
//   pass 1: per-chunk raw delta sums (fp64, no wrap inside a chunk;
//           deltas are a few radians each, so 64 of them stay well
//           within exact fp64 range)
//   pass 2: serial carry chain over the chunks per (b, h, angle),
//           seeded from the persistent angle state, wrapped once per
//           chunk boundary
//   pass 3: per-chunk re-walk from the entering carry with the same
//           per-step wrap the sequential kernel applies, writing the
//           f32 cumsum and, from the last chunk, the exit state
//
// This is a different rounding route than the single serial chain (the
// carries defer wrapping to chunk boundaries); the difference is a few
// fp64 ulps, invisible at the f32 output precision. It is a numeric
// route of its own and is validated by its own parity oracle.

#define DEFINE_M3_ANGLE_CHUNK_SUMS(NAME, IDX)                                  \
extern "C" __global__ void NAME(                                               \
    double* __restrict__ chunk_sums,      /* [B * n_chunks * nh * n_angles] */ \
    const float* __restrict__ angles_raw, /* [B*T * n_angles] */               \
    const float* __restrict__ dt_arr,     /* [B*T * nh] */                     \
    int B, int T, int nh, int n_angles, int chunk_size,                        \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                          \
    IDX bc = blockIdx.x;                                                       \
    IDX b = bc / n_chunks;                                                     \
    int chunk = (int)(bc % n_chunks);                                          \
    if (b >= B) return;                                                        \
    int idx = (y_first + blockIdx.y) * blockDim.x + threadIdx.x;               \
    int total_per_env = nh * n_angles;                                         \
    if (idx >= total_per_env) return;                                          \
    int h = idx / n_angles;                                                    \
    int a = idx % n_angles;                                                    \
                                                                               \
    int t_start = chunk * chunk_size;                                          \
    int t_end = t_start + chunk_size;                                          \
    if (t_end > T) t_end = T;                                                  \
                                                                               \
    double sum = 0.0;                                                          \
    for (int t = t_start; t < t_end; t++) {                                    \
        IDX bt = b * T + t;                                                    \
        float raw = angles_raw[bt * n_angles + a];                             \
        float dt_val = dt_arr[bt * nh + h];                                    \
        sum += (double)(tanhf(raw) * PI * dt_val);                             \
    }                                                                          \
    chunk_sums[((b * n_chunks + chunk) * nh + h) * n_angles + a] = sum;        \
}

DEFINE_M3_ANGLE_CHUNK_SUMS(m3_angle_chunk_sums, int)
DEFINE_M3_ANGLE_CHUNK_SUMS(m3_angle_chunk_sums_i64, long long)

/* The wrap into [0, 2*pi). CUDA's fmod is exact (0 ulp), so for a state
 * already in [0, 2*TWO_PI) the answer is either the state itself or one
 * exact subtraction (Sterbenz: TWO_PI <= x < 2*TWO_PI keeps x - TWO_PI
 * exact). That covers every hot iteration - deltas are bounded well
 * below 2*pi - and the general fmod stays as the fallback, so the
 * result is bit-identical to calling fmod every time while removing a
 * multi-instruction fp64 library routine from the densest fp64 loop in
 * the page. */
static __device__ __forceinline__ double m3_angle_wrap(double state) {
    const double TWO_PI_64 = 6.283185307179586;
    if (state >= 0.0 && state < 2.0 * TWO_PI_64) {
        if (state >= TWO_PI_64) state -= TWO_PI_64;
        return state;
    }
    state = fmod(state, TWO_PI_64);
    if (state < 0.0) state += TWO_PI_64;
    return state;
}

#define DEFINE_M3_ANGLE_CHUNK_APPLY(NAME, IDX)                                 \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ angle_cumsum,      /* [B*T * nh * n_angles] */         \
    float* __restrict__ angle_state,       /* [B * nh * n_angles] -- exit state */ \
    const float* __restrict__ entry_state, /* [B * nh * n_angles] -- snapshot */ \
    const double* __restrict__ chunk_sums, /* [B * n_chunks * nh * n_angles] */ \
    const float* __restrict__ angles_raw,  /* [B*T * n_angles] */              \
    const float* __restrict__ dt_arr,      /* [B*T * nh] */                    \
    int B, int T, int nh, int n_angles, int chunk_size,                        \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                          \
    IDX bc = blockIdx.x;                                                       \
    IDX b = bc / n_chunks;                                                     \
    int chunk = (int)(bc % n_chunks);                                          \
    if (b >= B) return;                                                        \
    int idx = (y_first + blockIdx.y) * blockDim.x + threadIdx.x;               \
    int total_per_env = nh * n_angles;                                         \
    if (idx >= total_per_env) return;                                          \
    int h = idx / n_angles;                                                    \
    int a = idx % n_angles;                                                    \
                                                                               \
    int t_start = chunk * chunk_size;                                          \
    int t_end = t_start + chunk_size;                                          \
    if (t_end > T) t_end = T;                                                  \
                                                                               \
    /* The entering carry for this chunk, folded inline: the same */           \
    /* ascending walk over the same stored doubles the dedicated carry */      \
    /* kernel used to run, so the value is bit-identical - and the whole */    \
    /* GPU no longer waits on one 192-thread block to serialize it. The */     \
    /* entry state comes from a snapshot because the last chunk's block */     \
    /* writes the exit state into angle_state concurrently. */                 \
    double state = (double)entry_state[b * total_per_env + idx];               \
    {                                                                          \
        /* Eight independent loads in flight per step: the adds stay a         \
         * serial chain (same order, same bits) but the loads no longer        \
         * serialize on it. */                                                 \
        long long stride = (long long)nh * n_angles;                           \
        const double* base = chunk_sums + (long long)b * n_chunks * stride + idx; \
        int c = 0;                                                             \
        for (; c + 8 <= chunk; c += 8) {                                       \
            double s0 = base[(long long)(c + 0) * stride];                     \
            double s1 = base[(long long)(c + 1) * stride];                     \
            double s2 = base[(long long)(c + 2) * stride];                     \
            double s3 = base[(long long)(c + 3) * stride];                     \
            double s4 = base[(long long)(c + 4) * stride];                     \
            double s5 = base[(long long)(c + 5) * stride];                     \
            double s6 = base[(long long)(c + 6) * stride];                     \
            double s7 = base[(long long)(c + 7) * stride];                     \
            state = m3_angle_wrap(state + s0);                                 \
            state = m3_angle_wrap(state + s1);                                 \
            state = m3_angle_wrap(state + s2);                                 \
            state = m3_angle_wrap(state + s3);                                 \
            state = m3_angle_wrap(state + s4);                                 \
            state = m3_angle_wrap(state + s5);                                 \
            state = m3_angle_wrap(state + s6);                                 \
            state = m3_angle_wrap(state + s7);                                 \
        }                                                                      \
        for (; c < chunk; c++) {                                               \
            state = m3_angle_wrap(state + base[(long long)c * stride]);        \
        }                                                                      \
    }                                                                          \
    for (int t = t_start; t < t_end; t++) {                                    \
        IDX bt = b * T + t;                                                    \
        float raw = angles_raw[bt * n_angles + a];                             \
        float dt_val = dt_arr[bt * nh + h];                                    \
        double delta = (double)(tanhf(raw) * PI * dt_val);                     \
        state += delta;                                                        \
        state = m3_angle_wrap(state);                                          \
        angle_cumsum[bt * total_per_env + h * n_angles + a] = (float)state;    \
    }                                                                          \
    if (chunk == n_chunks - 1) {                                               \
        angle_state[b * total_per_env + h * n_angles + a] = (float)state;      \
    }                                                                          \
}

DEFINE_M3_ANGLE_CHUNK_APPLY(m3_angle_chunk_apply, int)
DEFINE_M3_ANGLE_CHUNK_APPLY(m3_angle_chunk_apply_i64, long long)

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
// Stage 2 (caller): two reduce_sum_axis0 launches finalise the outputs.
#define DEFINE_M3_ANGLE_DT_BWD_SEQ(NAME, IDX)                                  \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ contrib_angles,      /* [nh * B*T*n_angles] OUTPUT */  \
    float* __restrict__ contrib_dt,          /* [n_angles * B*T*nh] OUTPUT */  \
    const float* __restrict__ d_angle_cumsum,/* [B*T * nh * n_angles] */       \
    const float* __restrict__ angles_raw,    /* [B*T * n_angles] */            \
    const float* __restrict__ dt_arr,        /* [B*T * nh] */                  \
    int B, int T, int nh, int n_angles,                                        \
    int y_first /* this run's first y block */                                 \
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

DEFINE_M3_ANGLE_DT_BWD_SEQ(m3_angle_dt_bwd_seq, int)
DEFINE_M3_ANGLE_DT_BWD_SEQ(m3_angle_dt_bwd_seq_i64, long long)

// reduce_sum_axis0 for the angle gradients' partials once their rows hold
// more columns than the 32-bit range: one block per column as there, each
// launch covering the columns from col0 so a row longer than the grid's x
// extent takes several launches, and every column summed in
// reduce_sum_axis0's order (each lane strides over the rows, then the
// shared-memory tree), so a launch of the same block width writes the same
// bits.
extern "C" __global__ __launch_bounds__(256, 4)
void reduce_sum_axis0_wide(
    float* __restrict__ out,            // [dim]
    const float* __restrict__ partials, // [batch * dim]
    int batch, long long dim, long long col0,
    int accumulate                      // 0 = overwrite, 1 = +=
) {
    long long d = col0 + blockIdx.x;
    if (d >= dim) return;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];

    float sum = 0.0f;
    for (int b = tid; b < batch; b += blockDim.x) {
        sum += partials[(long long)b * dim + d];
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

// ============================================================================
// 7. rope_fwd -- Apply cos/sin rotation to B and C pairs
// ============================================================================
//
// For each (sample, head h, angle pair a):
//   cos_a = cos(angle_cumsum[sample * nh * n_angles + h * n_angles + a])
//   sin_a = sin(angle_cumsum[...])
//   i0 = 2*a, i1 = 2*a + 1
//   B_rot[..., i0] = cos_a * B[..., i0] - sin_a * B[..., i1]
//   B_rot[..., i1] = sin_a * B[..., i0] + cos_a * B[..., i1]
//   Same for C.
//   Elements beyond 2*n_angles pass through unchanged.
//
// Input:  B_biased[N * nh * ds], C_biased[N * nh * ds],
//         angle_cumsum[N * nh * n_angles]
// Output: B_rotated[N * nh * ds], C_rotated[N * nh * ds]
// Grid: ceil(N * nh * ds / 256) blocks, 256 threads
//
// Strategy: each thread handles one (sample, head, state_idx) element.
// For rotated pairs (idx < 2*n_angles), we read both elements of the pair
// and write the rotated result.
// For pass-through elements (idx >= 2*n_angles), simple copy.
// To avoid double-writes for pairs, even-indexed threads (i0) write both i0 and i1.
#define DEFINE_ROPE_FWD(NAME, IDX)                                             \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ B_rotated,          /* [N * nh * ds] */                \
    float* __restrict__ C_rotated,          /* [N * nh * ds] */                \
    const float* __restrict__ B_biased,     /* [N * nh * ds] */                \
    const float* __restrict__ C_biased,     /* [N * nh * ds] */                \
    const float* __restrict__ angle_cumsum, /* [N * nh * n_angles] */          \
    int N, int nh, int ds, int n_angles                                        \
) {                                                                            \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    IDX total = (IDX)N * nh * ds;                                              \
    if (idx >= total) return;                                                  \
                                                                               \
    int nh_ds = nh * ds;                                                       \
    IDX sample = idx / nh_ds;                                                  \
    int rem = (int)(idx % nh_ds);                                              \
    int h = rem / ds;                                                          \
    int n = rem % ds;                                                          \
                                                                               \
    IDX base_bc = sample * nh_ds + h * ds;                                     \
    int rope_end = 2 * n_angles;  /* elements [0..rope_end) are rotated in pairs */ \
                                                                               \
    if (n >= rope_end) {                                                       \
        /* Pass-through: no rotation applied */                                \
        B_rotated[idx] = B_biased[idx];                                        \
        C_rotated[idx] = C_biased[idx];                                        \
    } else if ((n & 1) == 0) {                                                 \
        if (n + 1 >= ds) {                                                     \
            /* Degenerate trailing "pair" (config guards 2*n_angles <= ds so */ \
            /* this is unreachable, but never index into the next head). */    \
            B_rotated[idx] = B_biased[idx];                                    \
            C_rotated[idx] = C_biased[idx];                                    \
            return;                                                            \
        }                                                                      \
        /* Even index: this thread handles the pair (n, n+1) */                \
        int a = n / 2;  /* angle index */                                      \
        IDX angle_idx = sample * nh * n_angles + h * n_angles + a;             \
        float cos_a, sin_a;                                                    \
        sincosf(angle_cumsum[angle_idx], &sin_a, &cos_a);                      \
                                                                               \
        IDX i0 = base_bc + n;                                                  \
        IDX i1 = i0 + 1;                                                       \
                                                                               \
        float b0 = B_biased[i0];                                               \
        float b1 = B_biased[i1];                                               \
        B_rotated[i0] = cos_a * b0 - sin_a * b1;                               \
        B_rotated[i1] = sin_a * b0 + cos_a * b1;                               \
                                                                               \
        float c0 = C_biased[i0];                                               \
        float c1 = C_biased[i1];                                               \
        C_rotated[i0] = cos_a * c0 - sin_a * c1;                               \
        C_rotated[i1] = sin_a * c0 + cos_a * c1;                               \
    }                                                                          \
    /* Odd indices within rope range: handled by the even thread of the pair. */ \
    /* No-op here (the even thread already wrote both i0 and i1). */           \
}

DEFINE_ROPE_FWD(rope_fwd, int)
DEFINE_ROPE_FWD(rope_fwd_i64, long long)

// ============================================================================
// 8. rope_bwd -- Inverse rotation for backward (transpose of rotation matrix)
// ============================================================================
//
// Forward rotation matrix R = [[cos, -sin], [sin, cos]]
// Backward (transpose): R^T = [[cos, sin], [-sin, cos]]
//
// For each (sample, head h, angle pair a):
//   d_B_pre[i0] =  cos_a * d_B_rot[i0] + sin_a * d_B_rot[i1]
//   d_B_pre[i1] = -sin_a * d_B_rot[i0] + cos_a * d_B_rot[i1]
//   Same for C.
//
// Plus d_angle contribution from the Jacobian of the rotation:
//   d_angle[a] += d_B_rot[i0] * (-sin_a * B_pre[i0] - cos_a * B_pre[i1])
//              +  d_B_rot[i1] * ( cos_a * B_pre[i0] - sin_a * B_pre[i1])
//   Same contribution from C, summed into d_angle[a].
//
// Input:  d_B_rotated[N*nh*ds], d_C_rotated[N*nh*ds],
//         B_biased[N*nh*ds] (pre-RoPE), C_biased[N*nh*ds] (pre-RoPE),
//         angle_cumsum[N*nh*n_angles]
// Output: d_B_pre_rope[N*nh*ds], d_C_pre_rope[N*nh*ds],
//         d_angle_cumsum[N*nh*n_angles]
// Grid: ceil(N * nh * ds / 256) blocks, 256 threads
#define DEFINE_ROPE_BWD(NAME, IDX)                                             \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ d_B_pre_rope,        /* [N * nh * ds] */               \
    float* __restrict__ d_C_pre_rope,        /* [N * nh * ds] */               \
    float* __restrict__ d_angle_cumsum,      /* [N * nh * n_angles] */         \
    const float* __restrict__ d_B_rotated,   /* [N * nh * ds] */               \
    const float* __restrict__ d_C_rotated,   /* [N * nh * ds] */               \
    const float* __restrict__ B_biased,      /* [N * nh * ds] -- pre-RoPE input */ \
    const float* __restrict__ C_biased,      /* [N * nh * ds] -- pre-RoPE input */ \
    const float* __restrict__ angle_cumsum,  /* [N * nh * n_angles] */         \
    int N, int nh, int ds, int n_angles                                        \
) {                                                                            \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    IDX total = (IDX)N * nh * ds;                                              \
    if (idx >= total) return;                                                  \
                                                                               \
    int nh_ds = nh * ds;                                                       \
    IDX sample = idx / nh_ds;                                                  \
    int rem = (int)(idx % nh_ds);                                              \
    int h = rem / ds;                                                          \
    int n = rem % ds;                                                          \
                                                                               \
    IDX base_bc = sample * nh_ds + h * ds;                                     \
    int rope_end = 2 * n_angles;                                               \
                                                                               \
    if (n >= rope_end) {                                                       \
        /* Pass-through: gradient flows unchanged */                           \
        d_B_pre_rope[idx] = d_B_rotated[idx];                                  \
        d_C_pre_rope[idx] = d_C_rotated[idx];                                  \
    } else if ((n & 1) == 0) {                                                 \
        if (n + 1 >= ds) {                                                     \
            /* Degenerate trailing "pair" — pass through, never cross the */   \
            /* head boundary (unreachable with 2*n_angles <= d_state). */      \
            d_B_pre_rope[idx] = d_B_rotated[idx];                              \
            d_C_pre_rope[idx] = d_C_rotated[idx];                              \
            int a = n / 2;                                                     \
            d_angle_cumsum[sample * nh * n_angles + h * n_angles + a] = 0.0f;  \
            return;                                                            \
        }                                                                      \
        /* Even index: handle pair (n, n+1) */                                 \
        int a = n / 2;                                                         \
        IDX angle_idx = sample * nh * n_angles + h * n_angles + a;             \
        float cos_a, sin_a;                                                    \
        sincosf(angle_cumsum[angle_idx], &sin_a, &cos_a);                      \
                                                                               \
        IDX i0 = base_bc + n;                                                  \
        IDX i1 = i0 + 1;                                                       \
                                                                               \
        /* Inverse rotation (transpose of R) */                                \
        float db0 = d_B_rotated[i0];                                           \
        float db1 = d_B_rotated[i1];                                           \
        d_B_pre_rope[i0] =  cos_a * db0 + sin_a * db1;                         \
        d_B_pre_rope[i1] = -sin_a * db0 + cos_a * db1;                         \
                                                                               \
        float dc0 = d_C_rotated[i0];                                           \
        float dc1 = d_C_rotated[i1];                                           \
        d_C_pre_rope[i0] =  cos_a * dc0 + sin_a * dc1;                         \
        d_C_pre_rope[i1] = -sin_a * dc0 + cos_a * dc1;                         \
                                                                               \
        /* d_angle from Jacobian of rotation applied to B */                   \
        float b_pre0 = B_biased[i0];                                           \
        float b_pre1 = B_biased[i1];                                           \
        float d_angle_b = db0 * (-sin_a * b_pre0 - cos_a * b_pre1)             \
                        + db1 * ( cos_a * b_pre0 - sin_a * b_pre1);            \
                                                                               \
        /* d_angle from Jacobian of rotation applied to C */                   \
        float c_pre0 = C_biased[i0];                                           \
        float c_pre1 = C_biased[i1];                                           \
        float d_angle_c = dc0 * (-sin_a * c_pre0 - cos_a * c_pre1)             \
                        + dc1 * ( cos_a * c_pre0 - sin_a * c_pre1);            \
                                                                               \
        d_angle_cumsum[angle_idx] = d_angle_b + d_angle_c;                     \
    }                                                                          \
    /* Odd indices: handled by even thread of the pair. */                     \
}

DEFINE_ROPE_BWD(rope_bwd, int)
DEFINE_ROPE_BWD(rope_bwd_i64, long long)

// ---------------------------------------------------------------------------
// Compute alpha/beta/gamma from dt, a_val (negative), trap (sigmoided)
// ---------------------------------------------------------------------------
// alpha = exp(a_val * dt)
// beta  = alpha * dt * (1 - trap)
// gamma = trap * dt
#define DEFINE_M3_COMPUTE_ABG(NAME, IDX)                                       \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ alpha,                                                 \
    float* __restrict__ beta,                                                  \
    float* __restrict__ gamma,                                                 \
    const float* __restrict__ dt,                                              \
    const float* __restrict__ a_val,                                           \
    const float* __restrict__ trap,                                            \
    IDX N  /* total elements = batch * T * nheads */                           \
) {                                                                            \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    if (idx >= N) return;                                                      \
    float dt_v = dt[idx];                                                      \
    float a_v = a_val[idx];    /* already negative (-heavy_tail clamped) */    \
    float t_v = trap[idx];     /* already sigmoided */                         \
    float alpha_v = FAST_EXP(a_v * dt_v);                                      \
    alpha[idx] = alpha_v;                                                      \
    beta[idx] = alpha_v * dt_v * (1.0f - t_v);                                 \
    gamma[idx] = t_v * dt_v;                                                   \
}

DEFINE_M3_COMPUTE_ABG(m3_compute_abg, int)
DEFINE_M3_COMPUTE_ABG(m3_compute_abg_i64, long long)

// ============================================================================
// 9. m3_abg_bwd -- Backward through alpha/beta/gamma + softplus/heavy-tail/sigmoid
// ============================================================================
//
// Forward (m3_compute_abg + m3_split activations):
//   dt_val = softplus(dd_dt_raw + dt_bias)
//   a_val  = clamp(-heavy_tail(dd_a_raw), max = -a_floor)
//   trap_sig = sigmoid(trap_raw)
//   alpha  = exp(a_val * dt_val)
//   beta   = alpha * dt_val * (1 - trap_sig)
//   gamma  = trap_sig * dt_val
//
// Backward computes: d_dd_dt_raw, d_dd_a_raw, d_trap_raw
// Also writes: d_dt_bias (per-head, needs external reduction across B*T)
//
// The d_dt_angle contribution (from angle_dt_bwd) is added into d_dd_dt_raw.
//
// Input:  d_alpha[N], d_beta[N], d_gamma[N], d_dt_angle[N],
//         dt[N] (post-softplus), a_val[N] (clamped), alpha[N],
//         dd_dt_raw[N], dd_a_raw[N], trap_raw[N],
//         dt_bias[nh], a_floor (scalar)
// Output: d_dd_dt[N], d_dd_a[N], d_trap_out[N]
// Grid: ceil(N / 256) blocks, 256 threads
#define DEFINE_M3_ABG_BWD(NAME, IDX)                                           \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ d_dd_dt,       /* [N] -- gradient w.r.t. dd_dt_raw (proj component) */ \
    float* __restrict__ d_dd_a,        /* [N] -- gradient w.r.t. dd_a_raw (proj component) */ \
    float* __restrict__ d_trap_out,    /* [N] -- gradient w.r.t. trap_raw (proj component) */ \
    const float* __restrict__ d_alpha, /* [N] */                               \
    const float* __restrict__ d_beta,  /* [N] */                               \
    const float* __restrict__ d_gamma, /* [N] */                               \
    const float* __restrict__ d_dt_angle, /* [N] -- angle contribution to d_dt (from angle_dt_bwd) */ \
    const float* __restrict__ dt,      /* [N] -- post-softplus dt */           \
    const float* __restrict__ a_val,   /* [N] -- clamped -heavy_tail(dd_a) */  \
    const float* __restrict__ alpha,   /* [N] -- exp(a_val * dt) */            \
    const float* __restrict__ dd_dt_raw, /* [N] -- saved raw (pre-softplus) */ \
    const float* __restrict__ dd_a_raw,  /* [N] -- saved raw (pre-heavy-tail) */ \
    const float* __restrict__ trap_raw,  /* [N] -- saved raw (pre-sigmoid) */  \
    const float* __restrict__ dt_bias, /* [nh] */                              \
    float a_floor,                                                             \
    IDX N, int nh                                                              \
) {                                                                            \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    if (idx >= N) return;                                                      \
                                                                               \
    int h = (int)(idx % nh);                                                   \
                                                                               \
    float da = d_alpha[idx];                                                   \
    float db = d_beta[idx];                                                    \
    float dg = d_gamma[idx];                                                   \
    float ddt_angle = d_dt_angle[idx];                                         \
                                                                               \
    float dt_v   = dt[idx];                                                    \
    float a_v    = a_val[idx];    /* already negative, clamped */              \
    float alpha_v = alpha[idx];                                                \
    float tr_raw = trap_raw[idx];                                              \
    float trap_sig = 1.0f / (1.0f + FAST_EXP(-tr_raw));                        \
                                                                               \
    /* d_adt = d_alpha * alpha + d_beta * beta */                              \
    /* (since d(beta)/d(adt) = d(alpha*dt*(1-trap))/d(adt) = alpha*dt*(1-trap) = beta) */ \
    float d_adt = da * alpha_v + db * alpha_v * dt_v * (1.0f - trap_sig);      \
                                                                               \
    /* d_dt contributions */                                                   \
    float d_dt_from_adt   = d_adt * a_v;                                       \
    float d_dt_from_beta  = db * alpha_v * (1.0f - trap_sig);                  \
    float d_dt_from_gamma = dg * trap_sig;                                     \
    float d_dt_total = d_dt_from_adt + d_dt_from_beta + d_dt_from_gamma + ddt_angle; \
                                                                               \
    /* softplus backward for DT: d_dd_dt = d_dt_total * sigmoid(dd_dt_raw + dt_bias) */ \
    float dt_pre = dd_dt_raw[idx] + dt_bias[h];                                \
    float sp_deriv_dt = (dt_pre > 20.0f) ? 1.0f : (1.0f / (1.0f + FAST_EXP(-dt_pre))); \
    d_dd_dt[idx] = d_dt_total * sp_deriv_dt;                                   \
                                                                               \
    /* d_trap_sig from beta and gamma paths */                                 \
    float d_trap_sig = -db * alpha_v * dt_v + dg * dt_v;                       \
    /* sigmoid backward: d_trap_raw = d_trap_sig * sig * (1 - sig) */          \
    d_trap_out[idx] = d_trap_sig * trap_sig * (1.0f - trap_sig);               \
                                                                               \
    /* d_a_val = d_adt * dt_val */                                             \
    float d_a_val = d_adt * dt_v;                                              \
    /* A = -heavy_tail(dd_a_raw), clamped to max = -a_floor */                 \
    /* If clamped (a_val == -a_floor), gradient is zero. */                    \
    float raw_a = dd_a_raw[idx];                                               \
    float a_unclamped = -m3_heavy_tail(raw_a);                                 \
    if (a_unclamped > -a_floor) {                                              \
        /* Was clamped: gradient is zero */                                    \
        d_dd_a[idx] = 0.0f;                                                    \
    } else {                                                                   \
        /* d(-heavy_tail(x))/dx = -heavy_tail'(x) */                           \
        d_dd_a[idx] = d_a_val * (-m3_heavy_tail_deriv(raw_a));                 \
    }                                                                          \
}

DEFINE_M3_ABG_BWD(m3_abg_bwd, int)
DEFINE_M3_ABG_BWD(m3_abg_bwd_i64, long long)

// ============================================================================
// 10. bc_bias_add_bwd -- Reverse of bc_bias_add: head -> group reduction
// ============================================================================
//
// Forward: B_biased[n, h, d] = B_normed[n, g, d] + bias[h, d]
//          where g = h / heads_per_group
//
// Backward:
//   d_bias[h, d] += sum_n d_B_biased[n, h, d]    (via external colsum)
//   d_B_normed[n, g, d] += sum_{h in group g} d_B_biased[n, h, d]
//
// This kernel computes d_B_normed by reducing across heads within each group.
// Grid: ceil(N * ng * ds / 256) blocks, 256 threads
#define DEFINE_BC_BIAS_ADD_BWD_F32(NAME, IDX)                                  \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ d_B_normed,     /* [N * ng * ds] */                    \
    const float* __restrict__ d_B_biased, /* [N * nh * ds] */                  \
    int N, int nh, int ng, int ds                                              \
) {                                                                            \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    IDX total = (IDX)N * ng * ds;                                              \
    if (idx >= total) return;                                                  \
                                                                               \
    int ng_ds = ng * ds;                                                       \
    IDX sample = idx / ng_ds;                                                  \
    int rem = (int)(idx % ng_ds);                                              \
    int g = rem / ds;                                                          \
    int d = rem % ds;                                                          \
                                                                               \
    int heads_per_group = nh / ng;                                             \
    float sum = 0.0f;                                                          \
    IDX base = sample * nh * ds;                                               \
    for (int local_h = 0; local_h < heads_per_group; local_h++) {              \
        int h = g * heads_per_group + local_h;                                 \
        sum += d_B_biased[base + h * ds + d];                                  \
    }                                                                          \
    d_B_normed[idx] = sum;                                                     \
}

DEFINE_BC_BIAS_ADD_BWD_F32(bc_bias_add_bwd, int)
DEFINE_BC_BIAS_ADD_BWD_F32(bc_bias_add_bwd_i64, long long)

// ============================================================================
// 11. m3_split_bwd -- Assemble d_proj from 8 gradient components
// ============================================================================
//
// Reverse of m3_split: writes each gradient component at its correct offset.
// Layout: [d_z: di | d_x: di | d_B_raw: ng*ds | d_C_raw: ng*ds |
//          d_dd_dt: nh | d_dd_A: nh | d_trap: nh | d_angles: n_angles]
//
// Grid: ceil(N * in_proj_dim / 256) blocks, 256 threads
#define DEFINE_M3_SPLIT_BWD_F32(NAME, IDX)                                     \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ d_proj,         /* [N * in_proj_dim] */                \
    const float* __restrict__ d_z,      /* [N * di] */                         \
    const float* __restrict__ d_x,      /* [N * di] */                         \
    const float* __restrict__ d_B_raw,  /* [N * ng * ds] */                    \
    const float* __restrict__ d_C_raw,  /* [N * ng * ds] */                    \
    const float* __restrict__ d_dd_dt,  /* [N * nh] */                         \
    const float* __restrict__ d_dd_a,   /* [N * nh] */                         \
    const float* __restrict__ d_trap,   /* [N * nh] */                         \
    const float* __restrict__ d_angles, /* [N * n_angles] */                   \
    int N, int di, int ng, int ds, int nh, int n_angles                        \
) {                                                                            \
    int in_proj_dim = 2 * di + 2 * ng * ds + 3 * nh + n_angles;                \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    IDX total = (IDX)N * in_proj_dim;                                          \
    if (idx >= total) return;                                                  \
                                                                               \
    IDX sample = idx / in_proj_dim;                                            \
    int col = (int)(idx % in_proj_dim);                                        \
                                                                               \
    int ng_ds = ng * ds;                                                       \
    int off0 = di;                                                             \
    int off1 = 2 * di;                                                         \
    int off2 = off1 + ng_ds;                                                   \
    int off3 = off2 + ng_ds;                                                   \
    int off4 = off3 + nh;                                                      \
    int off5 = off4 + nh;                                                      \
    int off6 = off5 + nh;                                                      \
                                                                               \
    float val;                                                                 \
    if (col < off0) {                                                          \
        val = d_z[sample * di + col];                                          \
    } else if (col < off1) {                                                   \
        val = d_x[sample * di + (col - off0)];                                 \
    } else if (col < off2) {                                                   \
        val = d_B_raw[sample * ng_ds + (col - off1)];                          \
    } else if (col < off3) {                                                   \
        val = d_C_raw[sample * ng_ds + (col - off2)];                          \
    } else if (col < off4) {                                                   \
        val = d_dd_dt[sample * nh + (col - off3)];                             \
    } else if (col < off5) {                                                   \
        val = d_dd_a[sample * nh + (col - off4)];                              \
    } else if (col < off6) {                                                   \
        val = d_trap[sample * nh + (col - off5)];                              \
    } else {                                                                   \
        int a = col - off6;                                                    \
        val = (a < n_angles) ? d_angles[sample * n_angles + a] : 0.0f;         \
    }                                                                          \
    d_proj[idx] = val;                                                         \
}

DEFINE_M3_SPLIT_BWD_F32(m3_split_bwd, int)
DEFINE_M3_SPLIT_BWD_F32(m3_split_bwd_i64, long long)
// RMSNormGated forward: output = RMSNorm(y) * weight * SiLU(z)
// norm_before_gate=True (Mamba-3 style): normalize y first, then gate.
// Source: layernorm_gated.py rms_norm_ref, Mamba-3 default.
// Plain SiLU gating: out[i] = y[i] * silu(z[i])
// Used when is_outproj_norm=false (default).
// Source: Python mamba3_siso_step.py line 219: out = out * silu(z)
#define DEFINE_SILU_GATE_FWD_F32(NAME, IDX)                                    \
extern "C" __global__ void NAME(                                               \
    float* out, const float* y, const float* z, IDX n                          \
) {                                                                            \
    IDX i = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (i >= n) return;                                                        \
    float z_val = z[i];                                                        \
    float silu_z = z_val / (1.0f + exp2f(-z_val * 1.4426950408889634f));       \
    out[i] = y[i] * silu_z;                                                    \
}

DEFINE_SILU_GATE_FWD_F32(silu_gate_fwd, int)
DEFINE_SILU_GATE_FWD_F32(silu_gate_fwd_i64, long long)

// SiLU gating backward: d_y[i] = d_out[i] * silu(z[i]), d_z[i] = d_out[i] * y[i] * silu'(z[i])
#define DEFINE_SILU_GATE_BWD_F32(NAME, IDX)                                    \
extern "C" __global__ void NAME(                                               \
    float* d_y, float* d_z, const float* d_out,                                \
    const float* y, const float* z, IDX n                                      \
) {                                                                            \
    IDX i = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (i >= n) return;                                                        \
    float z_val = z[i];                                                        \
    float sigma = 1.0f / (1.0f + exp2f(-z_val * 1.4426950408889634f));         \
    float silu_z = z_val * sigma;                                              \
    float d_out_val = d_out[i];                                                \
    d_y[i] = d_out_val * silu_z;                                               \
    /* Parity pin: FACTORED silu-grad form, kept in lockstep with the */       \
    /* CPU twin in src/mamba3_siso/cpu/backward.rs - change both or neither. */ \
    d_z[i] = d_out_val * y[i] * sigma * (1.0f + z_val * (1.0f - sigma));       \
}

DEFINE_SILU_GATE_BWD_F32(silu_gate_bwd, int)
DEFINE_SILU_GATE_BWD_F32(silu_gate_bwd_i64, long long)

// Per-group gated RMSNorm, norm before gate (Mamba-3): a group is one head
// of `group_size` channels, out = y * rstd(group) * weight * silu(z).
//
// Grid: (N, ceil(n_groups / groups_per_block)). Block: groups_per_block
// times the group rounded up to a power of two, or 1024 threads for a
// wider group (`gated_norm_cfg` on the host). Dynamic shared memory: one
// float per thread. A group's lanes are its rounded-up width, at most the
// block; a lane takes the group's channels lane, lane + lanes, and so on,
// chains their terms in that order, and the group sums the lanes' chains
// with a pairwise tree, lanes without a channel holding -0.0, which leaves
// every sum as it is. A group of at most 1024 channels gives each lane one
// channel, so a power-of-two group runs the tree the one-block-per-row
// kernel ran, and any other group one fixed order. Every thread reaches
// every barrier, and threads past the groups only feed the tree.
__device__ __forceinline__ int m3g_group_width(int group_size) {
    int width = 1;
    while (width < group_size) width <<= 1;
    return width;
}

__device__ __forceinline__ float m3g_group_sum(float* sm, float v, int local, int width) {
    sm[threadIdx.x] = v;
    __syncthreads();
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
        if (local < stride) sm[threadIdx.x] += sm[threadIdx.x + stride];
        __syncthreads();
    }
    return sm[threadIdx.x - local];
}

// rms_vals: [N * n_groups], the rstd of every group for the backward.
// z_stride: the row stride of z, d_inner when z is dense.
#define DEFINE_RMSNORM_GATED_FWD(NAME, IDX, T_ACT, FROM_F)                     \
extern "C" __global__ void NAME(                                               \
    T_ACT* __restrict__ out,                                                   \
    float* __restrict__ rms_vals,                                              \
    const T_ACT* __restrict__ y,                                               \
    const T_ACT* __restrict__ z,                                               \
    const float* __restrict__ weight,                                          \
    int N, int d_inner, int group_size, float eps, int z_stride,               \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    const int width = m3g_group_width(group_size);                             \
    const int lanes = width < (int)blockDim.x ? width : (int)blockDim.x;       \
    const int local_group = threadIdx.x / lanes;                               \
    const int local = threadIdx.x - local_group * lanes;                       \
    const int n_groups = d_inner / group_size;                                 \
    const int group = (y_first + blockIdx.y) * (blockDim.x / lanes) + local_group; \
    const IDX sample = blockIdx.x;                                             \
    const bool live = sample < N && group < n_groups && local < group_size;    \
    const int d0 = group * group_size + local;                                 \
    const long long base = (long long)sample * d_inner;                        \
    extern __shared__ float m3g_sm[];                                          \
    /* The lane's first channel stays in a register for the output. */        \
    const float y0 = live ? to_f(y[base + d0]) : 0.0f;                         \
    float chain = live ? y0 * y0 : -0.0f;                                      \
    if (live) {                                                                \
        for (int c = local + lanes; c < group_size; c += lanes) {              \
            const float y_val = to_f(y[base + group * group_size + c]);        \
            chain += y_val * y_val;                                            \
        }                                                                      \
    }                                                                          \
    const float sum = m3g_group_sum(m3g_sm, chain, local, lanes);              \
    /* IEEE sqrtf and division, the rstd form of every norm in the tree. */    \
    float rstd = 1.0f / sqrtf(sum / (float)group_size + eps);                  \
    /* A non-finite y would turn rstd into NaN and carry it through the */     \
    /* gate into the residual stream. */                                       \
    if (!isfinite(rstd) || rstd > 1e20f) rstd = 1.0f;                          \
    if (!live) return;                                                         \
    if (local == 0) rms_vals[(long long)sample * n_groups + group] = rstd;     \
    for (int c = local; c < group_size; c += lanes) {                          \
        const int d = group * group_size + c;                                  \
        const float y_val = c == local ? y0 : to_f(y[base + d]);               \
        const float z_val = to_f(z[(long long)sample * z_stride + d]);         \
        const float silu_z = z_val / (1.0f + FAST_EXP(-z_val));                \
        out[base + d] = FROM_F(y_val * rstd * weight[d] * silu_z);             \
    }                                                                          \
}

// The backward, after layernorm_gated.py's NORM_BEFORE_GATE path: d_weight
// is per row ([N * d_inner]), reduced over the rows by the caller. A lane's
// first channel keeps its terms in registers for d_y; the others, present
// only in a group wider than 1024, are read again.
#define DEFINE_RMSNORM_GATED_BWD(NAME, IDX, T_ACT, FROM_F)                     \
extern "C" __global__ void NAME(                                               \
    T_ACT* d_y, T_ACT* d_z, float* d_weight,                                   \
    const T_ACT* d_out, const T_ACT* y, const T_ACT* z,                        \
    const float* weight, const float* rms_vals,                                \
    int N, int d_inner, int group_size,                                        \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    const int width = m3g_group_width(group_size);                             \
    const int lanes = width < (int)blockDim.x ? width : (int)blockDim.x;       \
    const int local_group = threadIdx.x / lanes;                               \
    const int local = threadIdx.x - local_group * lanes;                       \
    const int n_groups = d_inner / group_size;                                 \
    const int group = (y_first + blockIdx.y) * (blockDim.x / lanes) + local_group; \
    const IDX sample = blockIdx.x;                                             \
    const bool live = sample < N && group < n_groups && local < group_size;    \
    const long long base = (long long)sample * d_inner;                        \
    extern __shared__ float m3g_sm[];                                          \
    float rstd = 0.0f, y_hat0 = 0.0f, w0 = 0.0f, dy_scaled0 = 0.0f;            \
    float chain = -0.0f;                                                       \
    if (live) {                                                                \
        rstd = rms_vals[(long long)sample * n_groups + group];                 \
        for (int c = local; c < group_size; c += lanes) {                      \
            const int d = group * group_size + c;                              \
            const float y_val = to_f(y[base + d]);                             \
            const float z_val = to_f(z[base + d]);                             \
            const float w = weight[d];                                         \
            const float d_out_val = to_f(d_out[base + d]);                     \
            const float y_hat = y_val * rstd;                                  \
            const float y_normed = y_hat * w;                                  \
            const float sig_z = 1.0f / (1.0f + FAST_EXP(-z_val));              \
            const float silu_z = z_val * sig_z;                                \
            const float d_silu = sig_z + z_val * sig_z * (1.0f - sig_z);       \
            d_z[base + d] = FROM_F(d_out_val * y_normed * d_silu);             \
            /* The gradient through the gate, then the norm's per-row */       \
            /* weight gradient and the group's c1 sum. */                      \
            const float dy_scaled = d_out_val * silu_z;                        \
            d_weight[base + d] = dy_scaled * y_hat;                            \
            const float contrib = y_hat * w * dy_scaled;                       \
            if (c == local) {                                                  \
                y_hat0 = y_hat;                                                \
                w0 = w;                                                        \
                dy_scaled0 = dy_scaled;                                        \
                chain = contrib;                                               \
            } else {                                                           \
                chain += contrib;                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    const float sum = m3g_group_sum(m3g_sm, chain, local, lanes);              \
    if (!live) return;                                                         \
    const float c1 = sum / (float)group_size;                                  \
    for (int c = local; c < group_size; c += lanes) {                          \
        const int d = group * group_size + c;                                  \
        float y_hat = y_hat0, w = w0, dy_scaled = dy_scaled0;                  \
        if (c != local) {                                                      \
            const float z_val = to_f(z[base + d]);                             \
            w = weight[d];                                                     \
            y_hat = to_f(y[base + d]) * rstd;                                  \
            const float sig_z = 1.0f / (1.0f + FAST_EXP(-z_val));              \
            dy_scaled = to_f(d_out[base + d]) * (z_val * sig_z);               \
        }                                                                      \
        d_y[base + d] = FROM_F((w * dy_scaled - y_hat * c1) * rstd);           \
    }                                                                          \
}

DEFINE_RMSNORM_GATED_FWD(rmsnorm_gated_forward, int, float, from_f_f32)
DEFINE_RMSNORM_GATED_FWD(rmsnorm_gated_forward_i64, long long, float, from_f_f32)
DEFINE_RMSNORM_GATED_FWD(rmsnorm_gated_forward_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_RMSNORM_GATED_FWD(rmsnorm_gated_forward_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_RMSNORM_GATED_FWD(rmsnorm_gated_forward_f16, int, __half, from_f_f16)
DEFINE_RMSNORM_GATED_FWD(rmsnorm_gated_forward_f16_i64, long long, __half, from_f_f16)
DEFINE_RMSNORM_GATED_BWD(rmsnorm_gated_backward, int, float, from_f_f32)
DEFINE_RMSNORM_GATED_BWD(rmsnorm_gated_backward_i64, long long, float, from_f_f32)
DEFINE_RMSNORM_GATED_BWD(rmsnorm_gated_backward_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_RMSNORM_GATED_BWD(rmsnorm_gated_backward_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_RMSNORM_GATED_BWD(rmsnorm_gated_backward_f16, int, __half, from_f_f16)
DEFINE_RMSNORM_GATED_BWD(rmsnorm_gated_backward_f16_i64, long long, __half, from_f_f16)
#undef DEFINE_RMSNORM_GATED_FWD
#undef DEFINE_RMSNORM_GATED_BWD

// ============================================================================
// Templated variants for end-to-end bf16/f16 activations (inference only).
//
// Dtype split for Mamba-3 bf16 path (matches M1 and state-spaces conventions):
// - Activations/linear tensors: T_ACT (bf16/f16)
//   proj, z, x, B_raw, C_raw, B_normed, C_normed, B_biased, C_biased,
//   B_rotated, C_rotated, y, gated, post_norm
// - Coefficient/scalar tensors: f32
//   dt, a_val, trap, alpha, beta, gamma, angles_raw, angle_cumsum,
//   dt_bias, norm weights, rms stats
// - Recurrent state: f32 (see mamba3_siso.cu m3_step_fwd templated variant)
// - Residual stream: f32 (HF residual_in_fp32=True)
//
// Backward/training tensors (dd_*_raw, trap_raw saves) are not used in
// inference — the templated variants still accept them as f32 pointers so
// callers can pass unused scratch without branching.
// ============================================================================

// ------- m3_split typed -------
// proj: T_ACT input; outputs split into bf16 activations + f32 coefficients.
#define DEFINE_M3_SPLIT(NAME, IDX, T_ACT, FROM_F)                             \
extern "C" __global__ void NAME(                                               \
    T_ACT* __restrict__ z,           /* [N * di] */                            \
    T_ACT* __restrict__ x,           /* [N * di] */                            \
    T_ACT* __restrict__ B_raw,       /* [N * ng * ds] */                       \
    T_ACT* __restrict__ C_raw,       /* [N * ng * ds] */                       \
    float* __restrict__ dt,          /* [N * nh] — coefficient, f32 */          \
    float* __restrict__ a_val,       /* [N * nh] — coefficient, f32 */          \
    float* __restrict__ trap,        /* [N * nh] — coefficient, f32 */          \
    float* __restrict__ angles,      /* [N * n_angles] — coefficient, f32 */    \
    float* __restrict__ dd_dt_raw,   /* [N * nh] — backward save (f32) */       \
    float* __restrict__ dd_a_raw,    /* [N * nh] — backward save (f32) */       \
    float* __restrict__ trap_raw,    /* [N * nh] — backward save (f32) */       \
    const T_ACT* __restrict__ proj,  /* [N * in_proj_dim] */                    \
    const float* __restrict__ dt_bias, /* [nh] */                              \
    float a_floor,                                                              \
    int N, int di, int ng, int ds, int nh, int n_angles                         \
) {                                                                             \
    int in_proj_dim = 2 * di + 2 * ng * ds + 3 * nh + n_angles;                 \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                       \
    IDX total = (IDX)N * in_proj_dim;                                           \
    if (idx >= total) return;                                                   \
    IDX sample = idx / in_proj_dim;                                             \
    int col = (int)(idx % in_proj_dim);                                         \
    float val = to_f(proj[idx]);                                                \
    int off0 = di;                                                              \
    int off1 = 2 * di;                                                          \
    int ng_ds = ng * ds;                                                        \
    int off2 = off1 + ng_ds;                                                    \
    int off3 = off2 + ng_ds;                                                    \
    int off4 = off3 + nh;                                                       \
    int off5 = off4 + nh;                                                       \
    int off6 = off5 + nh;                                                       \
    if (col < off0) {                                                           \
        if (z) z[sample * di + col] = FROM_F(val);                                     \
    } else if (col < off1) {                                                    \
        x[sample * di + (col - off0)] = FROM_F(val);                            \
    } else if (col < off2) {                                                    \
        if (B_raw) B_raw[sample * ng_ds + (col - off1)] = FROM_F(val);          \
    } else if (col < off3) {                                                    \
        if (C_raw) C_raw[sample * ng_ds + (col - off2)] = FROM_F(val);          \
    } else if (col < off4) {                                                    \
        int h = col - off3;                                                     \
        IDX dt_idx = sample * nh + h;                                           \
        if (dd_dt_raw) dd_dt_raw[dt_idx] = val;                                 \
        float biased = val + dt_bias[h];                                        \
        dt[dt_idx] = (biased > 20.0f) ? biased : log1pf(FAST_EXP(biased)); \
    } else if (col < off5) {                                                    \
        int h = col - off4;                                                     \
        IDX a_idx = sample * nh + h;                                            \
        if (dd_a_raw) dd_a_raw[a_idx] = val;                                    \
        float a = -m3_heavy_tail(val);                                          \
        a_val[a_idx] = (a > -a_floor) ? -a_floor : a; /* NaN stays NaN */       \
    } else if (col < off6) {                                                    \
        int h = col - off5;                                                     \
        IDX t_idx = sample * nh + h;                                            \
        if (trap_raw) trap_raw[t_idx] = val;                                    \
        trap[t_idx] = 1.0f / (1.0f + FAST_EXP(-val));                           \
    } else {                                                                    \
        int a = col - off6;                                                     \
        if (a < n_angles) {                                                     \
            angles[sample * n_angles + a] = val;                                \
        }                                                                       \
    }                                                                           \
}

DEFINE_M3_SPLIT(m3_split_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_M3_SPLIT(m3_split_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_M3_SPLIT(m3_split_f16, int, __half, from_f_f16)
DEFINE_M3_SPLIT(m3_split_f16_i64, long long, __half, from_f_f16)

// ------- bcnorm_fwd_bc fused (B + C in one launch) -------
// Per-(sample, group) RMSNorm of B and C in one launch: gridDim.y picks
// the operand. Short rows share full warps while retaining the original
// descending reduction tree and the rounded square at its input. Launch
// dimensions must come from bcnorm_fwd_bc_cfg for the packed row mapping.
#define DEFINE_BCNORM_FWD_BC(NAME, IDX, T_ACT, FROM_F)                         \
extern "C" __global__ void NAME(                                                \
    T_ACT* __restrict__ B_normed,                                               \
    T_ACT* __restrict__ C_normed,                                               \
    float* __restrict__ B_rms,                                                  \
    float* __restrict__ C_rms,                                                  \
    const T_ACT* __restrict__ B_raw,                                            \
    const T_ACT* __restrict__ C_raw,                                            \
    const float* __restrict__ B_weight,                                         \
    const float* __restrict__ C_weight,                                         \
    int N, int ng, int ds,                                                      \
    float eps, /* config-driven eps */                                          \
    int src_stride /* row stride of B_raw/C_raw; ng*ds when dense, the    */    \
                   /* projection row width when reading proj in place     */    \
) {                                                                             \
    if (ds <= 32) {                                                            \
        const int width = blockDim.x;                                          \
        const IDX row = (IDX)blockIdx.x * blockDim.y + threadIdx.y;             \
        const int d = threadIdx.x;                                             \
        const bool active = row < (IDX)N * ng && d < ds;                       \
        const T_ACT* raw = (blockIdx.y == 0) ? B_raw : C_raw;                  \
        T_ACT* normed = (blockIdx.y == 0) ? B_normed : C_normed;               \
        float* rms_out = (blockIdx.y == 0) ? B_rms : C_rms;                    \
        const float* weight = (blockIdx.y == 0) ? B_weight : C_weight;         \
        const long long src = (long long)(row / ng) * src_stride               \
                              + (long long)(row % ng) * ds;                   \
        const float val = active ? to_f(raw[src + d]) : 0.0f;                  \
        /* Preserve the rounded square that the shared store materialized. */ \
        float sum = __fmul_rn(val, val);                                       \
        for (int stride = width >> 1; stride > 0; stride >>= 1) {               \
            const float rhs = __shfl_down_sync(0xffffffffu, sum, stride, width); \
            if (d < stride && (d + stride) < ds) sum += rhs;                    \
        }                                                                       \
        sum = __shfl_sync(0xffffffffu, sum, 0, width);                          \
        /* Padded rows and lanes must participate in every shuffle above. */  \
        if (!active) return;                                                   \
        float rms = sqrtf(sum / (float)ds + eps);                              \
        if (!isfinite(rms) || rms < 1e-20f) rms = 1.0f;                         \
        if (d == 0) rms_out[row] = rms;                                        \
        float inv_rms = 1.0f / rms;                                             \
        normed[row * ds + d] = FROM_F(val * inv_rms * weight[d]);               \
        return;                                                                \
    }                                                                           \
    /* gridDim.y == 2: 0 -> B path, 1 -> C path */                              \
    int which = blockIdx.y;                                                     \
    int block_id = blockIdx.x;                                                  \
    if (block_id >= (IDX)N * ng) return;                                        \
    int d = threadIdx.x;                                                        \
    if (d >= ds) return;                                                        \
    const T_ACT* raw = (which == 0) ? B_raw : C_raw;                            \
    T_ACT* normed = (which == 0) ? B_normed : C_normed;                         \
    float* rms_out = (which == 0) ? B_rms : C_rms;                              \
    const float* weight = (which == 0) ? B_weight : C_weight;                   \
    IDX base = (IDX)block_id * ds;                                              \
    long long src = (long long)(block_id / ng) * src_stride                     \
                    + (long long)(block_id % ng) * ds;                          \
    float val = to_f(raw[src + d]);                                             \
    extern __shared__ float sdata[];                                            \
    sdata[d] = val * val;                                                       \
    __syncthreads();                                                            \
    int stride = 1;                                                             \
    while (stride < ds) stride <<= 1;                                           \
    stride >>= 1;                                                               \
    for (; stride > 0; stride >>= 1) {                                          \
        if (d < stride && (d + stride) < ds) {                                  \
            sdata[d] += sdata[d + stride];                                      \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
    float rms = sqrtf(sdata[0] / (float)ds + eps);                          \
    if (!isfinite(rms) || rms < 1e-20f) rms = 1.0f;                             \
    if (d == 0) rms_out[block_id] = rms;                                        \
    __syncthreads();                                                            \
    float inv_rms = 1.0f / rms;                                                 \
    normed[base + d] = FROM_F(val * inv_rms * weight[d]);                       \
}

DEFINE_BCNORM_FWD_BC(bcnorm_fwd_bc_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_BCNORM_FWD_BC(bcnorm_fwd_bc_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_BCNORM_FWD_BC(bcnorm_fwd_bc_f16, int, __half, from_f_f16)
DEFINE_BCNORM_FWD_BC(bcnorm_fwd_bc_f16_i64, long long, __half, from_f_f16)
DEFINE_BCNORM_FWD_BC(bcnorm_fwd_bc_f32, int, float, from_f_f32)
DEFINE_BCNORM_FWD_BC(bcnorm_fwd_bc_f32_i64, long long, float, from_f_f32)

// ------- m3_bias_rope_fwd: bc_bias_add (B + C) + rope_fwd in ONE launch -------
//
// Replaces the F4c/F4d/F4ef launch triple. The biased tensors are still
// materialized (the backward consumes them as saved activations); the
// rotation then uses the register values instead of reloading them - an
// f32 held in a register carries the exact bits the global round trip
// returned, so K/Q match the three-launch chain bit for bit. With
// n_angles == 0 every element passes through (K = B_biased, Q = C_biased),
// which also replaces the two D2D copies of the old no-RoPE branch;
// angle_cumsum is never dereferenced in that case.
#define DEFINE_M3_BIAS_ROPE_FWD_F32(NAME, IDX)                                 \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ B_biased,           /* [N * nh * ds] */                \
    float* __restrict__ C_biased,           /* [N * nh * ds] */                \
    float* __restrict__ K_out,              /* [N * nh * ds] */                \
    float* __restrict__ Q_out,              /* [N * nh * ds] */                \
    const float* __restrict__ B_normed,     /* [N * ng * ds] */                \
    const float* __restrict__ C_normed,     /* [N * ng * ds] */                \
    const float* __restrict__ B_bias,       /* [nh * ds] */                    \
    const float* __restrict__ C_bias,       /* [nh * ds] */                    \
    const float* __restrict__ angle_cumsum, /* [N * nh * n_angles] */          \
    int N, int nh, int ng, int ds, int n_angles,                               \
    /* The decode step advances the angles here instead of in a kernel of */   \
    /* its own: with angle_state set, the lane that owns an angle steps it */  \
    /* from angles_raw and dt_arr, stores it, and rotates with the same */     \
    /* value. Training and prefill pass null and read angle_cumsum. */         \
    float* __restrict__ angle_state,        /* [N * nh * n_angles] or null */  \
    const float* __restrict__ angles_raw,   /* [N * n_angles] */               \
    const float* __restrict__ dt_arr        /* [N * nh] */                     \
) {                                                                            \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    IDX total = (IDX)N * nh * ds;                                              \
    if (idx >= total) return;                                                  \
    int nh_ds = nh * ds;                                                       \
    IDX sample = idx / nh_ds;                                                  \
    int rem = (int)(idx % nh_ds);                                              \
    int h = rem / ds;                                                          \
    int n = rem % ds;                                                          \
    int heads_per_group = nh / ng;                                             \
    int g = h / heads_per_group;                                               \
    IDX src = sample * ng * ds + g * ds + n;                                   \
    int bi = h * ds + n;                                                       \
    float b0 = B_normed[src] + B_bias[bi];                                     \
    float c0 = C_normed[src] + C_bias[bi];                                     \
    if (B_biased) B_biased[idx] = b0;                                          \
    if (C_biased) C_biased[idx] = c0;                                          \
    int rope_end = 2 * n_angles;                                               \
    if (n >= rope_end) {                                                       \
        K_out[idx] = b0;                                                       \
        Q_out[idx] = c0;                                                       \
        return;                                                                \
    }                                                                          \
    if ((n & 1) == 0) {                                                        \
        if (n + 1 >= ds) {                                                     \
            K_out[idx] = b0;                                                   \
            Q_out[idx] = c0;                                                   \
            return;                                                            \
        }                                                                      \
        /* The odd lane writes B/C_biased[idx + 1] itself; this lane only */   \
        /* recomputes the same values for the rotation. */                     \
        float b1 = B_normed[src + 1] + B_bias[bi + 1];                         \
        float c1 = C_normed[src + 1] + C_bias[bi + 1];                         \
        int a = n / 2;                                                         \
        IDX angle_idx = sample * nh * n_angles + h * n_angles + a;             \
        float ang;                                                             \
        if (angle_state != nullptr) {                                          \
            /* The standalone advance, line for line: f64 accumulation,        \
               the wrap into [0, 2*pi), the f32 store. */                      \
            double state = (double)angle_state[angle_idx];                     \
            const double TWO_PI_64 = 6.283185307179586;                        \
            float raw = angles_raw[sample * n_angles + a];                     \
            float dt_val = dt_arr[sample * nh + h];                            \
            double delta = (double)(tanhf(raw) * PI * dt_val);                 \
            state += delta;                                                    \
            state = fmod(state, TWO_PI_64);                                    \
            if (state < 0.0) state += TWO_PI_64;                               \
            ang = (float)state;                                                \
            angle_state[angle_idx] = ang;                                      \
        } else {                                                               \
            ang = angle_cumsum[angle_idx];                                     \
        }                                                                      \
        float cos_a, sin_a;                                                    \
        sincosf(ang, &sin_a, &cos_a);                                          \
        K_out[idx] = cos_a * b0 - sin_a * b1;                                  \
        K_out[idx + 1] = sin_a * b0 + cos_a * b1;                              \
        Q_out[idx] = cos_a * c0 - sin_a * c1;                                  \
        Q_out[idx + 1] = sin_a * c0 + cos_a * c1;                              \
    }                                                                          \
}

DEFINE_M3_BIAS_ROPE_FWD_F32(m3_bias_rope_fwd, int)
DEFINE_M3_BIAS_ROPE_FWD_F32(m3_bias_rope_fwd_i64, long long)

/* Typed twin: the biased stores round through FROM_F exactly like
 * bc_bias_add, and the rotation consumes the ROUND-TRIPPED values
 * (to_f of the stored T_ACT) - the same values rope_fwd reloaded from
 * global in the three-launch chain. */
#define DEFINE_M3_BIAS_ROPE_FWD(NAME, IDX, T_ACT, FROM_F)                      \
extern "C" __global__ void NAME(                                                \
    T_ACT* __restrict__ B_biased,                                               \
    T_ACT* __restrict__ C_biased,                                               \
    T_ACT* __restrict__ K_out,                                                  \
    T_ACT* __restrict__ Q_out,                                                  \
    const T_ACT* __restrict__ B_normed,                                         \
    const T_ACT* __restrict__ C_normed,                                         \
    const float* __restrict__ B_bias,                                           \
    const float* __restrict__ C_bias,                                           \
    const float* __restrict__ angle_cumsum,                                     \
    int N, int nh, int ng, int ds, int n_angles,                                \
    float* __restrict__ angle_state,                                            \
    const float* __restrict__ angles_raw,                                       \
    const float* __restrict__ dt_arr                                            \
) {                                                                             \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                       \
    IDX total = (IDX)N * nh * ds;                                               \
    if (idx >= total) return;                                                   \
    int nh_ds = nh * ds;                                                        \
    IDX sample = idx / nh_ds;                                                   \
    int rem = (int)(idx % nh_ds);                                               \
    int h = rem / ds;                                                           \
    int n = rem % ds;                                                           \
    int heads_per_group = nh / ng;                                              \
    int g = h / heads_per_group;                                                \
    IDX src = sample * ng * ds + g * ds + n;                                    \
    int bi = h * ds + n;                                                        \
    T_ACT tb0 = FROM_F(to_f(B_normed[src]) + B_bias[bi]);                       \
    T_ACT tc0 = FROM_F(to_f(C_normed[src]) + C_bias[bi]);                       \
    if (B_biased) B_biased[idx] = tb0;                                          \
    if (C_biased) C_biased[idx] = tc0;                                          \
    int rope_end = 2 * n_angles;                                                \
    if (n >= rope_end) {                                                        \
        K_out[idx] = tb0;                                                       \
        Q_out[idx] = tc0;                                                       \
        return;                                                                 \
    }                                                                           \
    if ((n & 1) == 0) {                                                         \
        if (n + 1 >= ds) {                                                      \
            K_out[idx] = tb0;                                                   \
            Q_out[idx] = tc0;                                                   \
            return;                                                             \
        }                                                                       \
        T_ACT tb1 = FROM_F(to_f(B_normed[src + 1]) + B_bias[bi + 1]);           \
        T_ACT tc1 = FROM_F(to_f(C_normed[src + 1]) + C_bias[bi + 1]);           \
        float b0 = to_f(tb0);                                                   \
        float b1 = to_f(tb1);                                                   \
        float c0 = to_f(tc0);                                                   \
        float c1 = to_f(tc1);                                                   \
        int a = n / 2;                                                          \
        IDX angle_idx = sample * nh * n_angles + h * n_angles + a;              \
        float ang;                                                              \
        if (angle_state != nullptr) {                                           \
            double state = (double)angle_state[angle_idx];                      \
            const double TWO_PI_64 = 6.283185307179586;                         \
            float raw = angles_raw[sample * n_angles + a];                      \
            float dt_val = dt_arr[sample * nh + h];                             \
            double delta = (double)(tanhf(raw) * PI * dt_val);                  \
            state += delta;                                                     \
            state = fmod(state, TWO_PI_64);                                     \
            if (state < 0.0) state += TWO_PI_64;                                \
            ang = (float)state;                                                 \
            angle_state[angle_idx] = ang;                                       \
        } else {                                                                \
            ang = angle_cumsum[angle_idx];                                      \
        }                                                                       \
        float cos_a, sin_a;                                                     \
        sincosf(ang, &sin_a, &cos_a);                                           \
        K_out[idx] = FROM_F(cos_a * b0 - sin_a * b1);                           \
        K_out[idx + 1] = FROM_F(sin_a * b0 + cos_a * b1);                       \
        Q_out[idx] = FROM_F(cos_a * c0 - sin_a * c1);                           \
        Q_out[idx + 1] = FROM_F(sin_a * c0 + cos_a * c1);                       \
    }                                                                           \
}

DEFINE_M3_BIAS_ROPE_FWD(m3_bias_rope_fwd_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_M3_BIAS_ROPE_FWD(m3_bias_rope_fwd_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_M3_BIAS_ROPE_FWD(m3_bias_rope_fwd_f16, int, __half, from_f_f16)
DEFINE_M3_BIAS_ROPE_FWD(m3_bias_rope_fwd_f16_i64, long long, __half, from_f_f16)

// ------- silu_gate_fwd typed -------
// y, z, out: T_ACT. Simple elementwise.
#define DEFINE_SILU_GATE_FWD(NAME, IDX, T_ACT, FROM_F)                         \
extern "C" __global__ void NAME(                                                \
    T_ACT* __restrict__ out,                                                    \
    const T_ACT* __restrict__ y,                                                \
    const T_ACT* __restrict__ z,                                                \
    IDX n,                                                                      \
    int d_inner,                                                                \
    int z_stride /* row stride of z: d_inner when dense, the projection   */    \
                 /* row width when the gate reads z inside proj in place  */    \
) {                                                                             \
    IDX i = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                         \
    if (i >= n) return;                                                         \
    long long zi = (long long)(i / d_inner) * z_stride + (i % d_inner);         \
    float z_val = to_f(z[zi]);                                                  \
    float silu_z = z_val / (1.0f + exp2f(-z_val * LOG2E));                      \
    out[i] = FROM_F(to_f(y[i]) * silu_z);                                       \
}

DEFINE_SILU_GATE_FWD(silu_gate_fwd_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_SILU_GATE_FWD(silu_gate_fwd_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_SILU_GATE_FWD(silu_gate_fwd_f16, int, __half, from_f_f16)
DEFINE_SILU_GATE_FWD(silu_gate_fwd_f16_i64, long long, __half, from_f_f16)

/* Typed twins of silu_gate_bwd: same argument order and the SAME factored
   d_silu form as the f32 kernel and the CPU twin (parity-pinned there) -
   compute in f32, one RNE downcast at store. */
#define DEFINE_SILU_GATE_BWD(NAME, IDX, T_ACT, FROM_F)                          \
extern "C" __global__ void NAME(                                                \
    T_ACT* __restrict__ d_y,                                                    \
    T_ACT* __restrict__ d_z,                                                    \
    const T_ACT* __restrict__ d_out,                                            \
    const T_ACT* __restrict__ y,                                                \
    const T_ACT* __restrict__ z,                                                \
    IDX n                                                                       \
) {                                                                             \
    IDX i = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                         \
    if (i >= n) return;                                                         \
    float z_val = to_f(z[i]);                                                   \
    float sigma = 1.0f / (1.0f + exp2f(-z_val * LOG2E));                        \
    float silu_z = z_val * sigma;                                               \
    float d_out_val = to_f(d_out[i]);                                           \
    d_y[i] = FROM_F(d_out_val * silu_z);                                        \
    d_z[i] = FROM_F(d_out_val * to_f(y[i]) * sigma * (1.0f + z_val * (1.0f - sigma))); \
}

DEFINE_SILU_GATE_BWD(silu_gate_bwd_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_SILU_GATE_BWD(silu_gate_bwd_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_SILU_GATE_BWD(silu_gate_bwd_f16, int, __half, from_f_f16)
DEFINE_SILU_GATE_BWD(silu_gate_bwd_f16_i64, long long, __half, from_f_f16)


// ============================================================================
// Typed M3 backward kernels (bcnorm + rope + bc_bias + split).
//
// Dtype split follows the same precision discipline as the typed M3 forward:
//   - Activation I/O tensors (B_raw, C_raw, B_normed, C_normed, B_biased,
//     C_biased, k/q-post-rope, z, x, d_proj) are T_ACT.
//   - All per-group RMS saves, angle_cumsum, dt/a_val/trap coefficients,
//     plus every atomicAdd weight-grad target remain f32.
//   - Math runs entirely in f32 internally (promote-on-load via to_f,
//     downcast-on-store via FROM_F).
// ============================================================================

// bcnorm_bwd typed: dy T_ACT, d_B T_ACT, B_raw T_ACT, rms_val f32,
// weight f32, d_weight f32 (atomicAdd master-grad accumulator).
#define DEFINE_BCNORM_BWD(NAME, IDX, T_ACT, FROM_F)                            \
extern "C" __global__ void NAME(                                               \
    T_ACT* __restrict__ d_B,                                                   \
    float* __restrict__ d_weight,                                              \
    const T_ACT* __restrict__ d_out,                                           \
    const T_ACT* __restrict__ B_raw,                                           \
    const float* __restrict__ rms_val,                                         \
    const float* __restrict__ weight,                                          \
    int N, int ng, int ds                                                      \
) {                                                                            \
    int block_id = blockIdx.x;                                                 \
    if (block_id >= (IDX)N * ng) return;                                       \
    int d = threadIdx.x;                                                       \
    if (d >= ds) return;                                                       \
    IDX base = (IDX)block_id * ds;                                             \
    float rms = rms_val[block_id];                                             \
    float inv_rms = 1.0f / fmaxf(rms, 1e-12f);                                 \
    float x_val = to_f(B_raw[base + d]);                                       \
    float w = weight[d];                                                       \
    float dy = to_f(d_out[base + d]);                                          \
    float x_hat = x_val * inv_rms;                                             \
    extern __shared__ float sdata[];                                           \
    sdata[d] = x_hat * w * dy;                                                 \
    __syncthreads();                                                           \
    int stride = 1;                                                            \
    while (stride < ds) stride <<= 1;                                          \
    stride >>= 1;                                                              \
    for (; stride > 0; stride >>= 1) {                                         \
        if (d < stride && (d + stride) < ds) {                                 \
            sdata[d] += sdata[d + stride];                                     \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    float c1 = sdata[0] / (float)ds;                                           \
    d_B[base + d] = FROM_F((w * dy - x_hat * c1) * inv_rms);                   \
    d_weight[base + d] = dy * x_hat;                                           \
}

DEFINE_BCNORM_BWD(bcnorm_bwd_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_BCNORM_BWD(bcnorm_bwd_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_BCNORM_BWD(bcnorm_bwd_f16, int, __half, from_f_f16)
DEFINE_BCNORM_BWD(bcnorm_bwd_f16_i64, long long, __half, from_f_f16)

// bc_bias_add_bwd typed: reduce typed d_B_biased [N*nh*ds] over heads
// within each group → typed d_B_normed [N*ng*ds]. Bias grad is the
// identity reduction over the same axis (handled by a separate
// reduce_bias launch, same pattern as M1 dt_proj_bias).
#define DEFINE_BC_BIAS_ADD_BWD(NAME, IDX, T_ACT, FROM_F)                       \
extern "C" __global__ void NAME(                                               \
    T_ACT* __restrict__ d_B_normed,                                            \
    const T_ACT* __restrict__ d_B_biased,                                      \
    int N, int nh, int ng, int ds                                              \
) {                                                                            \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    IDX total = (IDX)N * ng * ds;                                              \
    if (idx >= total) return;                                                  \
    int ng_ds = ng * ds;                                                       \
    IDX sample = idx / ng_ds;                                                  \
    int rem = (int)(idx % ng_ds);                                              \
    int g = rem / ds;                                                          \
    int d = rem % ds;                                                          \
    int heads_per_group = nh / ng;                                             \
    float sum = 0.0f;                                                          \
    IDX base = sample * nh * ds;                                               \
    for (int local_h = 0; local_h < heads_per_group; local_h++) {              \
        int h = g * heads_per_group + local_h;                                 \
        sum += to_f(d_B_biased[base + h * ds + d]);                            \
    }                                                                          \
    d_B_normed[idx] = FROM_F(sum);                                             \
}

DEFINE_BC_BIAS_ADD_BWD(bc_bias_add_bwd_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_BC_BIAS_ADD_BWD(bc_bias_add_bwd_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_BC_BIAS_ADD_BWD(bc_bias_add_bwd_f16, int, __half, from_f_f16)
DEFINE_BC_BIAS_ADD_BWD(bc_bias_add_bwd_f16_i64, long long, __half, from_f_f16)

// rope_bwd typed: typed d_B/C_rotated, typed d_B/C_pre_rope output,
// typed B/C_biased saved; angle_cumsum + d_angle_cumsum stay f32
// (accumulator precision critical — the rotation angles are summed
// over T timesteps, f32 atomicAdd at the reducer).
#define DEFINE_ROPE_BWD_IO(NAME, IDX, T_GRAD, FROM_F, T_PRE)                   \
extern "C" __global__ void NAME(                                               \
    T_GRAD* __restrict__ d_B_pre_rope,                                         \
    T_GRAD* __restrict__ d_C_pre_rope,                                         \
    float* __restrict__ d_angle_cumsum,                                        \
    const T_GRAD* __restrict__ d_B_rotated,                                    \
    const T_GRAD* __restrict__ d_C_rotated,                                    \
    const T_PRE* __restrict__ B_biased,                                        \
    const T_PRE* __restrict__ C_biased,                                        \
    const float* __restrict__ angle_cumsum,                                    \
    int N, int nh, int ds, int n_angles                                        \
) {                                                                            \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    IDX total = (IDX)N * nh * ds;                                              \
    if (idx >= total) return;                                                  \
    int nh_ds = nh * ds;                                                       \
    IDX sample = idx / nh_ds;                                                  \
    int rem = (int)(idx % nh_ds);                                              \
    int h = rem / ds;                                                          \
    int n = rem % ds;                                                          \
    IDX base_bc = sample * nh_ds + h * ds;                                     \
    int rope_end = 2 * n_angles;                                               \
    if (n >= rope_end) {                                                       \
        d_B_pre_rope[idx] = d_B_rotated[idx];                                  \
        d_C_pre_rope[idx] = d_C_rotated[idx];                                  \
    } else if ((n & 1) == 0) {                                                 \
        if (n + 1 >= ds) {                                                     \
            /* degenerate trailing pair: pass through, stay in head slice */  \
            d_B_pre_rope[idx] = d_B_rotated[idx];                              \
            d_C_pre_rope[idx] = d_C_rotated[idx];                              \
            int a = n / 2;                                                     \
            d_angle_cumsum[sample * nh * n_angles + h * n_angles + a] = 0.0f; \
            return;                                                            \
        }                                                                      \
        int a = n / 2;                                                         \
        IDX angle_idx = sample * nh * n_angles + h * n_angles + a;             \
        float cos_a, sin_a;                                                    \
        sincosf(angle_cumsum[angle_idx], &sin_a, &cos_a);                      \
        IDX i0 = base_bc + n;                                                  \
        IDX i1 = i0 + 1;                                                       \
        float db0 = to_f(d_B_rotated[i0]);                                     \
        float db1 = to_f(d_B_rotated[i1]);                                     \
        d_B_pre_rope[i0] = FROM_F( cos_a * db0 + sin_a * db1);                 \
        d_B_pre_rope[i1] = FROM_F(-sin_a * db0 + cos_a * db1);                 \
        float dc0 = to_f(d_C_rotated[i0]);                                     \
        float dc1 = to_f(d_C_rotated[i1]);                                     \
        d_C_pre_rope[i0] = FROM_F( cos_a * dc0 + sin_a * dc1);                 \
        d_C_pre_rope[i1] = FROM_F(-sin_a * dc0 + cos_a * dc1);                 \
        float b_pre0 = to_f(B_biased[i0]);                                     \
        float b_pre1 = to_f(B_biased[i1]);                                     \
        float d_angle_b = db0 * (-sin_a * b_pre0 - cos_a * b_pre1)             \
                        + db1 * ( cos_a * b_pre0 - sin_a * b_pre1);            \
        float c_pre0 = to_f(C_biased[i0]);                                     \
        float c_pre1 = to_f(C_biased[i1]);                                     \
        float d_angle_c = dc0 * (-sin_a * c_pre0 - cos_a * c_pre1)             \
                        + dc1 * ( cos_a * c_pre0 - sin_a * c_pre1);            \
        d_angle_cumsum[angle_idx] = d_angle_b + d_angle_c;                     \
    }                                                                          \
}

DEFINE_ROPE_BWD_IO(rope_bwd_bf16, int, __nv_bfloat16, from_f_bf16, __nv_bfloat16)
DEFINE_ROPE_BWD_IO(rope_bwd_bf16_i64, long long, __nv_bfloat16, from_f_bf16, __nv_bfloat16)
DEFINE_ROPE_BWD_IO(rope_bwd_f16, int, __half, from_f_f16, __half)
DEFINE_ROPE_BWD_IO(rope_bwd_f16_i64, long long, __half, from_f_f16, __half)
// The sequential mixed backward: f32 gradients, as its backward through
// time writes them, against the bf16 or f16 pre-RoPE B and C.
DEFINE_ROPE_BWD_IO(rope_bwd_mixed_bf16, int, float, from_f_f32, __nv_bfloat16)
DEFINE_ROPE_BWD_IO(rope_bwd_mixed_bf16_i64, long long, float, from_f_f32, __nv_bfloat16)
DEFINE_ROPE_BWD_IO(rope_bwd_mixed_f16, int, float, from_f_f32, __half)
DEFINE_ROPE_BWD_IO(rope_bwd_mixed_f16_i64, long long, float, from_f_f32, __half)
#undef DEFINE_ROPE_BWD_IO

// m3_split_bwd typed: assembles d_proj (T_ACT, feeds in_proj dX GEMM)
// from 8 gradient components. Layout: d_z, d_x, d_B_raw,
// d_C_raw are T_ACT (come from typed backward chain); d_dd_dt,
// d_dd_a, d_trap, d_angles stay f32 (m3_split_typed writes them as
// f32 in forward, and the f32 abg_bwd / angle_dt_bwd paths produce
// them in f32).
#define DEFINE_M3_SPLIT_BWD(NAME, IDX, T_ACT, FROM_F)                          \
extern "C" __global__ void NAME(                                               \
    T_ACT* __restrict__ d_proj,                                                \
    const T_ACT* __restrict__ d_z,                                             \
    const T_ACT* __restrict__ d_x,                                             \
    const T_ACT* __restrict__ d_B_raw,                                         \
    const T_ACT* __restrict__ d_C_raw,                                         \
    const float* __restrict__ d_dd_dt,                                         \
    const float* __restrict__ d_dd_a,                                          \
    const float* __restrict__ d_trap,                                          \
    const float* __restrict__ d_angles,                                        \
    int N, int di, int ng, int ds, int nh, int n_angles                        \
) {                                                                            \
    int in_proj_dim = 2 * di + 2 * ng * ds + 3 * nh + n_angles;                \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    IDX total = (IDX)N * in_proj_dim;                                          \
    if (idx >= total) return;                                                  \
    IDX sample = idx / in_proj_dim;                                            \
    int col = (int)(idx % in_proj_dim);                                        \
    int ng_ds = ng * ds;                                                       \
    int off0 = di;                                                             \
    int off1 = 2 * di;                                                         \
    int off2 = off1 + ng_ds;                                                   \
    int off3 = off2 + ng_ds;                                                   \
    int off4 = off3 + nh;                                                      \
    int off5 = off4 + nh;                                                      \
    int off6 = off5 + nh;                                                      \
    float val;                                                                 \
    if (col < off0) {                                                          \
        val = to_f(d_z[sample * di + col]);                                    \
    } else if (col < off1) {                                                   \
        val = to_f(d_x[sample * di + (col - off0)]);                           \
    } else if (col < off2) {                                                   \
        val = to_f(d_B_raw[sample * ng_ds + (col - off1)]);                    \
    } else if (col < off3) {                                                   \
        val = to_f(d_C_raw[sample * ng_ds + (col - off2)]);                    \
    } else if (col < off4) {                                                   \
        val = d_dd_dt[sample * nh + (col - off3)];                             \
    } else if (col < off5) {                                                   \
        val = d_dd_a[sample * nh + (col - off4)];                              \
    } else if (col < off6) {                                                   \
        val = d_trap[sample * nh + (col - off5)];                              \
    } else {                                                                   \
        int a = col - off6;                                                    \
        val = (a < n_angles) ? d_angles[sample * n_angles + a] : 0.0f;         \
    }                                                                          \
    d_proj[idx] = FROM_F(val);                                                 \
}

DEFINE_M3_SPLIT_BWD(m3_split_bwd_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_M3_SPLIT_BWD(m3_split_bwd_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_M3_SPLIT_BWD(m3_split_bwd_f16, int, __half, from_f_f16)
DEFINE_M3_SPLIT_BWD(m3_split_bwd_f16_i64, long long, __half, from_f_f16)

