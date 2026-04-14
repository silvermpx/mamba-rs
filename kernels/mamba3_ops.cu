// Mamba-3 SISO shared CUDA kernels.
//
// 8 kernels covering the non-SGEMM phases of the Mamba-3 SISO pipeline:
// split, BCNorm (fwd/bwd), bias add, angle accumulation (fwd/bwd), RoPE (fwd/bwd).
//
// CPU reference: mamba3_siso/cpu/forward.rs (phases F3-F5)
// T=1 step: mamba3_siso/cpu/inference.rs
// Paper: Lahoti et al., "Mamba-3: SISO" (ICLR 2026)

#include "_typed_prelude.cuh"

#ifndef RMS_EPS
#define RMS_EPS 1e-5f
#endif
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
//   - a_val:       -softplus(dd_A), clamped to max = -a_floor
//   - trap:        sigmoid(trap_raw)
//   - angles:      pass-through
//
// in_proj layout (contiguous per sample):
//   [z: d_inner | x: d_inner | B: ng*ds | C: ng*ds | dd_dt: nh | dd_A: nh | trap: nh | angles: n_angles]
//   Total = in_proj_dim
//
// Input:  proj[N * in_proj_dim]
// Outputs: z[N*di], x[N*di], B_raw[N*ng*ds], C_raw[N*ng*ds],
//          dt[N*nh] (post-softplus), a_val[N*nh] (neg-softplus clamped),
//          trap[N*nh] (post-sigmoid), angles[N*n_angles] (pass-through),
//          dd_dt_raw[N*nh] (saved for backward), dd_a_raw[N*nh] (saved for backward),
//          trap_raw[N*nh] (saved for backward)
// Params: dt_bias[nh], a_floor (scalar)
// Grid: ceil(N * in_proj_dim / 256) blocks, 256 threads
extern "C" __global__ void m3_split(
    float* __restrict__ z,          // [N * di]
    float* __restrict__ x,          // [N * di]
    float* __restrict__ B_raw,      // [N * ng * ds]
    float* __restrict__ C_raw,      // [N * ng * ds]
    float* __restrict__ dt,         // [N * nh] -- post-softplus
    float* __restrict__ a_val,      // [N * nh] -- -softplus(dd_A), clamped
    float* __restrict__ trap,       // [N * nh] -- post-sigmoid
    float* __restrict__ angles,     // [N * n_angles]
    float* __restrict__ dd_dt_raw,  // [N * nh] -- saved for backward
    float* __restrict__ dd_a_raw,   // [N * nh] -- saved for backward
    float* __restrict__ trap_raw,   // [N * nh] -- saved for backward
    const float* __restrict__ proj, // [N * in_proj_dim]
    const float* __restrict__ dt_bias, // [nh]
    float a_floor,
    int N, int di, int ng, int ds, int nh, int n_angles
) {
    int in_proj_dim = 2 * di + 2 * ng * ds + 3 * nh + n_angles;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * in_proj_dim;
    if (idx >= total) return;

    int sample = idx / in_proj_dim;
    int col = idx % in_proj_dim;
    float val = proj[idx];

    int off0 = di;                     // end of z
    int off1 = 2 * di;                 // end of x
    int ng_ds = ng * ds;
    int off2 = off1 + ng_ds;           // end of B
    int off3 = off2 + ng_ds;           // end of C
    int off4 = off3 + nh;              // end of dd_dt
    int off5 = off4 + nh;              // end of dd_A
    int off6 = off5 + nh;              // end of trap

    if (col < off0) {
        // z: pass-through
        z[sample * di + col] = val;
    } else if (col < off1) {
        // x: pass-through
        x[sample * di + (col - off0)] = val;
    } else if (col < off2) {
        // B_raw: pass-through
        B_raw[sample * ng_ds + (col - off1)] = val;
    } else if (col < off3) {
        // C_raw: pass-through
        C_raw[sample * ng_ds + (col - off2)] = val;
    } else if (col < off4) {
        // dd_dt: save raw, apply softplus(dd_dt + dt_bias)
        int h = col - off3;
        int dt_idx = sample * nh + h;
        dd_dt_raw[dt_idx] = val;
        float biased = val + dt_bias[h];
        dt[dt_idx] = (biased > 20.0f) ? biased : logf(1.0f + FAST_EXP(biased));
    } else if (col < off5) {
        // dd_A: save raw, apply -softplus(dd_A), clamp max = -a_floor
        int h = col - off4;
        int a_idx = sample * nh + h;
        dd_a_raw[a_idx] = val;
        float sp = (val > 20.0f) ? val : logf(1.0f + FAST_EXP(val));
        float a = -sp;
        a_val[a_idx] = fminf(a, -a_floor);
    } else if (col < off6) {
        // trap: save raw, apply sigmoid
        int h = col - off5;
        int t_idx = sample * nh + h;
        trap_raw[t_idx] = val;
        trap[t_idx] = 1.0f / (1.0f + FAST_EXP(-val));
    } else {
        // angles: pass-through
        int a = col - off6;
        if (a < n_angles) {
            angles[sample * n_angles + a] = val;
        }
    }
}

// ============================================================================
// 2. bcnorm_fwd -- RMSNorm on B (or C) with weight, no bias
// ============================================================================
//
// Per-(sample, head_group) RMSNorm: out = x / rms(x) * weight
// where rms(x) = sqrt(mean(x^2) + eps).
// Weight is [ds], tiled across groups (weight[i % ds] for group element i).
//
// Input:  B_raw[N * ng * ds], weight[ds]
// Output: B_normed[N * ng * ds], rms_val[N * ng] (saved for backward)
// Grid: N * ng blocks, ds threads per block
// Shared memory: ds floats for parallel reduction
extern "C" __global__ void bcnorm_fwd(
    float* __restrict__ B_normed,   // [N * ng * ds]
    float* __restrict__ rms_val,    // [N * ng] -- saved rms for backward
    const float* __restrict__ B_raw,// [N * ng * ds]
    const float* __restrict__ weight, // [ds]
    int N, int ng, int ds
) {
    int block_id = blockIdx.x;       // sample * ng + group
    if (block_id >= N * ng) return;
    int d = threadIdx.x;
    if (d >= ds) return;

    int base = block_id * ds;
    float val = B_raw[base + d];

    // Shared memory reduction for sum(x^2)
    extern __shared__ float sdata[];
    sdata[d] = val * val;
    __syncthreads();

    // Tree reduction — start at next power of 2 (safe for non-power-of-2 ds)
    int stride = 1;
    while (stride < ds) stride <<= 1;
    stride >>= 1;
    for (; stride > 0; stride >>= 1) {
        if (d < stride && (d + stride) < ds) {
            sdata[d] += sdata[d + stride];
        }
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / (float)ds + RMS_EPS);
    // Finite-guard: parallel to the fix in norms.cu DEFINE_RMSNORM_FWD —
    // on deep bf16 models a single overflowed bf16 activation can make rms
    // non-finite and NaN-cascade into every subsequent layer.
    if (!isfinite(rms) || rms < 1e-20f) rms = 1.0f;
    if (d == 0) rms_val[block_id] = rms;
    __syncthreads();

    // Normalize and scale
    float inv_rms = 1.0f / rms;
    B_normed[base + d] = val * inv_rms * weight[d];
}

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
extern "C" __global__ void bcnorm_bwd(
    float* __restrict__ d_B,        // [N * ng * ds] -- gradient w.r.t. B_raw
    float* __restrict__ d_weight,   // [N * ng * ds] -- per-block dw (reduce later)
    const float* __restrict__ d_out,// [N * ng * ds] -- upstream gradient
    const float* __restrict__ B_raw,// [N * ng * ds] -- saved input
    const float* __restrict__ rms_val,// [N * ng] -- saved rms from forward
    const float* __restrict__ weight, // [ds]
    int N, int ng, int ds
) {
    int block_id = blockIdx.x;
    if (block_id >= N * ng) return;
    int d = threadIdx.x;
    if (d >= ds) return;

    int base = block_id * ds;
    float rms = rms_val[block_id];
    float inv_rms = 1.0f / fmaxf(rms, 1e-12f);

    float x_val = B_raw[base + d];
    float w = weight[d];
    float dy = d_out[base + d];
    float x_hat = x_val * inv_rms;

    // Shared memory for c1 = mean(x_hat * w * dy)
    extern __shared__ float sdata[];
    sdata[d] = x_hat * w * dy;
    __syncthreads();

    int stride = 1;
    while (stride < ds) stride <<= 1;
    stride >>= 1;
    for (; stride > 0; stride >>= 1) {
        if (d < stride && (d + stride) < ds) {
            sdata[d] += sdata[d + stride];
        }
        __syncthreads();
    }

    float c1 = sdata[0] / (float)ds;

    // d_B = (w * dy - x_hat * c1) * inv_rms
    d_B[base + d] = (w * dy - x_hat * c1) * inv_rms;

    // d_weight contribution from this block
    d_weight[base + d] = dy * x_hat;
}

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
extern "C" __global__ void bc_bias_add(
    float* __restrict__ B_biased,       // [N * nh * ds]
    const float* __restrict__ B_normed, // [N * ng * ds]
    const float* __restrict__ bias,     // [nh * ds]
    int N, int nh, int ng, int ds
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * nh * ds;
    if (idx >= total) return;

    int nh_ds = nh * ds;
    int sample = idx / nh_ds;
    int rem = idx % nh_ds;
    int h = rem / ds;
    int n = rem % ds;

    int heads_per_group = nh / ng;
    int g = h / heads_per_group;

    B_biased[idx] = B_normed[sample * ng * ds + g * ds + n] + bias[h * ds + n];
}

// ============================================================================
// 5. angle_dt_fwd -- Cumulative angle accumulation (sequential over T per head)
// ============================================================================
//
// Per (batch, head, angle): sequential over T timesteps.
// For each timestep t:
//   delta = tanh(angles_raw[t * n_angles + a]) * PI * dt[t * nh + h]
//   angle_state[h * n_angles + a] += delta
//   angle_state[h * n_angles + a] = fmod(angle_state, 2*PI)   -- wrap to [0, 2pi)
//   angle_cumsum[t * nh * n_angles + h * n_angles + a] = angle_state
//
// NOTE: angles_raw is shared across all heads (angles are from in_proj, not per-head).
//       DT is per-head, so the angle delta differs per head.
//
// Input:  angles_raw[T * n_angles] -- raw angles per timestep (shared across heads)
//         dt[T * nh] -- post-softplus DT per (timestep, head)
// In/Out: angle_state[nh * n_angles] -- persistent per-head angle accumulation
// Output: angle_cumsum[T * nh * n_angles] -- per-timestep cumulative angles
// Grid: nh * n_angles threads total (1D). Sequential over T inside each thread.
extern "C" __global__ void angle_dt_fwd(
    float* __restrict__ angle_cumsum,    // [T * nh * n_angles]
    float* __restrict__ angle_state,     // [nh * n_angles] -- in/out
    const float* __restrict__ angles_raw,// [T * n_angles]
    const float* __restrict__ dt_arr,    // [T * nh]
    int T, int nh, int n_angles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = nh * n_angles;
    if (idx >= total_threads) return;

    int h = idx / n_angles;
    int a = idx % n_angles;

    double state = (double)angle_state[h * n_angles + a];
    const double TWO_PI_64 = 6.283185307179586;

    for (int t = 0; t < T; t++) {
        float raw = angles_raw[t * n_angles + a];
        float dt_val = dt_arr[t * nh + h];
        double delta = (double)(tanhf(raw) * PI * dt_val);
        state += delta;
        state = fmod(state, TWO_PI_64);
        if (state < 0.0) state += TWO_PI_64;

        angle_cumsum[t * nh * n_angles + h * n_angles + a] = (float)state;
    }

    angle_state[h * n_angles + a] = (float)state;
}

// ============================================================================
// 5b. angle_dt_fwd_batch -- Batched angle accumulation for N envs in one launch
// ============================================================================
//
// Same as angle_dt_fwd but handles N envs in a single launch instead of N
// separate launches. Each env has T=1 in collection mode.
//
// Grid: (N, ceil(nh * n_angles / blockDim.x), 1), Block: (min(nh*n_angles, 256))
// blockIdx.x = env index, threads handle (h, a) pairs.
//
// Input:  angles_raw[N * n_angles] -- shared across heads per env
//         dt[N * nh] -- per (env, head)
// In/Out: angle_state[N * nh * n_angles] -- persistent per (env, head)
// Output: angle_cumsum[N * nh * n_angles]
extern "C" __global__ void m3_angle_dt_fwd_batch(
    float* __restrict__ angle_cumsum,    // [N * nh * n_angles]
    float* __restrict__ angle_state,     // [N * nh * n_angles] -- in/out
    const float* __restrict__ angles_raw,// [N * n_angles]
    const float* __restrict__ dt_arr,    // [N * nh]
    int N, int nh, int n_angles
) {
    int env = blockIdx.x;
    if (env >= N) return;
    int idx = blockIdx.y * blockDim.x + threadIdx.x;
    int total_per_env = nh * n_angles;
    if (idx >= total_per_env) return;

    int h = idx / n_angles;
    int a = idx % n_angles;

    int state_idx = env * total_per_env + h * n_angles + a;
    double state = (double)angle_state[state_idx];
    const double TWO_PI_64 = 6.283185307179586;

    float raw = angles_raw[env * n_angles + a];
    float dt_val = dt_arr[env * nh + h];
    double delta = (double)(tanhf(raw) * PI * dt_val);
    state += delta;
    state = fmod(state, TWO_PI_64);
    if (state < 0.0) state += TWO_PI_64;

    angle_cumsum[state_idx] = (float)state;
    angle_state[state_idx] = (float)state;
}

// Sequential angle accumulation for training: B envs, T timesteps each.
// Grid: (B, ceil(nh*n_angles/256), 1). Block: (256, 1, 1).
// angles_raw: [B*T*n_angles], dt_arr: [B*T*nh], angle_state: [B*nh*n_angles]
// angle_cumsum: [B*T*nh*n_angles] — saves cumulative state at each timestep.
extern "C" __global__ void m3_angle_dt_fwd_seq(
    float* __restrict__ angle_cumsum,   // [B*T * nh * n_angles]
    float* __restrict__ angle_state,    // [B * nh * n_angles] -- persistent per-env
    const float* __restrict__ angles_raw, // [B*T * n_angles]
    const float* __restrict__ dt_arr,    // [B*T * nh]
    int B, int T, int nh, int n_angles
) {
    int b = blockIdx.x;
    if (b >= B) return;
    int idx = blockIdx.y * blockDim.x + threadIdx.x;
    int total_per_env = nh * n_angles;
    if (idx >= total_per_env) return;

    int h = idx / n_angles;
    int a = idx % n_angles;

    double state = (double)angle_state[b * total_per_env + h * n_angles + a];
    const double TWO_PI_64 = 6.283185307179586;

    for (int t = 0; t < T; t++) {
        int bt = b * T + t;
        float raw = angles_raw[bt * n_angles + a];
        float dt_val = dt_arr[bt * nh + h];
        double delta = (double)(tanhf(raw) * PI * dt_val);
        state += delta;
        state = fmod(state, TWO_PI_64);
        if (state < 0.0) state += TWO_PI_64;

        angle_cumsum[bt * total_per_env + h * n_angles + a] = (float)state;
    }

    angle_state[b * total_per_env + h * n_angles + a] = (float)state;
}

// ============================================================================
// 6. angle_dt_bwd -- Reverse cumsum for angle gradients
// ============================================================================
//
// Forward: angle_cumsum[t,h,a] = sum_{s=0}^{t} delta[s,h,a]
//          delta[s,h,a] = tanh(angles_raw[s*n_angles+a]) * PI * dt[s*nh+h]
//
// Backward (reverse cumsum):
//   For each (h, a), compute reverse cumsum of d_angle_cumsum over T:
//     d_delta[t,h,a] = sum_{s=t}^{T-1} d_angle_cumsum[s,h,a]
//   Then:
//     d_angles_raw[t,a] += d_delta[t,h,a] * PI * dt[t,h] * sech^2(angles_raw[t,a])
//     d_dt[t,h] += d_delta[t,h,a] * PI * tanh(angles_raw[t,a])
//
// NOTE: d_angles_raw is accumulated across all heads (atomicAdd needed).
//       d_dt_angle is per-head (no conflict across angles, sum within thread).
//
// Input:  d_angle_cumsum[T * nh * n_angles], angles_raw[T * n_angles], dt[T * nh]
// Output: d_angles_raw[T * n_angles] (atomicAdd across heads),
//         d_dt_angle[T * nh] (per-head sum across angles)
// Grid: nh * n_angles threads. Sequential over T (reverse) inside each thread.
extern "C" __global__ void angle_dt_bwd(
    float* __restrict__ d_angles_raw,        // [T * n_angles] -- atomicAdd across heads
    float* __restrict__ d_dt_angle,          // [T * nh] -- per-head angle contribution to d_dt
    const float* __restrict__ d_angle_cumsum,// [T * nh * n_angles]
    const float* __restrict__ angles_raw,    // [T * n_angles]
    const float* __restrict__ dt_arr,        // [T * nh]
    int T, int nh, int n_angles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = nh * n_angles;
    if (idx >= total_threads) return;

    int h = idx / n_angles;
    int a = idx % n_angles;

    // Reverse cumsum: d_delta[t] = sum_{s=t}^{T-1} d_angle_cumsum[s]
    float running = 0.0f;
    for (int t = T - 1; t >= 0; t--) {
        running += d_angle_cumsum[t * nh * n_angles + h * n_angles + a];
        float d_delta = running;

        float raw = angles_raw[t * n_angles + a];
        float th = tanhf(raw);
        float dt_val = dt_arr[t * nh + h];

        // d_angles_raw[t, a] += d_delta * PI * dt_val * (1 - tanh^2(raw))
        float sech2 = 1.0f - th * th;
        atomicAdd(&d_angles_raw[t * n_angles + a], d_delta * PI * dt_val * sech2);

        // d_dt_angle[t, h] += d_delta * PI * tanh(raw)
        atomicAdd(&d_dt_angle[t * nh + h], d_delta * PI * th);
    }
}

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
//   d_angles_raw[bt,a] += d_delta * PI * dt_val * sech^2(raw)   (atomicAdd across heads)
//   d_dt_angle[bt,h]   += d_delta * PI * tanh(raw)              (atomicAdd across angles)
//
// Grid: (B, ceil(nh*n_angles/256)). Block: (min(256, nh*n_angles)).
extern "C" __global__ void m3_angle_dt_bwd_seq(
    float* __restrict__ d_angles_raw,        // [B*T * n_angles] -- atomicAdd across heads
    float* __restrict__ d_dt_angle,          // [B*T * nh]
    const float* __restrict__ d_angle_cumsum,// [B*T * nh * n_angles]
    const float* __restrict__ angles_raw,    // [B*T * n_angles]
    const float* __restrict__ dt_arr,        // [B*T * nh]
    int B, int T, int nh, int n_angles
) {
    int b = blockIdx.x;
    if (b >= B) return;
    int idx = blockIdx.y * blockDim.x + threadIdx.x;
    int total_per_env = nh * n_angles;
    if (idx >= total_per_env) return;

    int h = idx / n_angles;
    int a = idx % n_angles;

    // Reverse cumsum over T for this (batch, head, angle)
    float running = 0.0f;
    for (int t = T - 1; t >= 0; t--) {
        int bt = b * T + t;
        running += d_angle_cumsum[bt * total_per_env + h * n_angles + a];
        float d_delta = running;

        float raw = angles_raw[bt * n_angles + a];
        float th = tanhf(raw);
        float dt_val = dt_arr[bt * nh + h];

        // d_angles_raw[bt, a] += d_delta * PI * dt_val * (1 - tanh^2(raw))
        float sech2 = 1.0f - th * th;
        atomicAdd(&d_angles_raw[bt * n_angles + a], d_delta * PI * dt_val * sech2);

        // d_dt_angle[bt, h] += d_delta * PI * tanh(raw)
        atomicAdd(&d_dt_angle[bt * nh + h], d_delta * PI * th);
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
extern "C" __global__ void rope_fwd(
    float* __restrict__ B_rotated,          // [N * nh * ds]
    float* __restrict__ C_rotated,          // [N * nh * ds]
    const float* __restrict__ B_biased,     // [N * nh * ds]
    const float* __restrict__ C_biased,     // [N * nh * ds]
    const float* __restrict__ angle_cumsum, // [N * nh * n_angles]
    int N, int nh, int ds, int n_angles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * nh * ds;
    if (idx >= total) return;

    int nh_ds = nh * ds;
    int sample = idx / nh_ds;
    int rem = idx % nh_ds;
    int h = rem / ds;
    int n = rem % ds;

    int base_bc = sample * nh_ds + h * ds;
    int rope_end = 2 * n_angles;  // elements [0..rope_end) are rotated in pairs

    if (n >= rope_end) {
        // Pass-through: no rotation applied
        B_rotated[idx] = B_biased[idx];
        C_rotated[idx] = C_biased[idx];
    } else if ((n & 1) == 0) {
        // Even index: this thread handles the pair (n, n+1)
        int a = n / 2;  // angle index
        int angle_idx = sample * nh * n_angles + h * n_angles + a;
        float cos_a, sin_a;
        sincosf(angle_cumsum[angle_idx], &sin_a, &cos_a);

        int i0 = base_bc + n;
        int i1 = i0 + 1;

        float b0 = B_biased[i0];
        float b1 = B_biased[i1];
        B_rotated[i0] = cos_a * b0 - sin_a * b1;
        B_rotated[i1] = sin_a * b0 + cos_a * b1;

        float c0 = C_biased[i0];
        float c1 = C_biased[i1];
        C_rotated[i0] = cos_a * c0 - sin_a * c1;
        C_rotated[i1] = sin_a * c0 + cos_a * c1;
    }
    // Odd indices within rope range: handled by the even thread of the pair.
    // No-op here (the even thread already wrote both i0 and i1).
}

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
extern "C" __global__ void rope_bwd(
    float* __restrict__ d_B_pre_rope,        // [N * nh * ds]
    float* __restrict__ d_C_pre_rope,        // [N * nh * ds]
    float* __restrict__ d_angle_cumsum,      // [N * nh * n_angles]
    const float* __restrict__ d_B_rotated,   // [N * nh * ds]
    const float* __restrict__ d_C_rotated,   // [N * nh * ds]
    const float* __restrict__ B_biased,      // [N * nh * ds] -- pre-RoPE input
    const float* __restrict__ C_biased,      // [N * nh * ds] -- pre-RoPE input
    const float* __restrict__ angle_cumsum,  // [N * nh * n_angles]
    int N, int nh, int ds, int n_angles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * nh * ds;
    if (idx >= total) return;

    int nh_ds = nh * ds;
    int sample = idx / nh_ds;
    int rem = idx % nh_ds;
    int h = rem / ds;
    int n = rem % ds;

    int base_bc = sample * nh_ds + h * ds;
    int rope_end = 2 * n_angles;

    if (n >= rope_end) {
        // Pass-through: gradient flows unchanged
        d_B_pre_rope[idx] = d_B_rotated[idx];
        d_C_pre_rope[idx] = d_C_rotated[idx];
    } else if ((n & 1) == 0) {
        // Even index: handle pair (n, n+1)
        int a = n / 2;
        int angle_idx = sample * nh * n_angles + h * n_angles + a;
        float cos_a, sin_a;
        sincosf(angle_cumsum[angle_idx], &sin_a, &cos_a);

        int i0 = base_bc + n;
        int i1 = i0 + 1;

        // Inverse rotation (transpose of R)
        float db0 = d_B_rotated[i0];
        float db1 = d_B_rotated[i1];
        d_B_pre_rope[i0] =  cos_a * db0 + sin_a * db1;
        d_B_pre_rope[i1] = -sin_a * db0 + cos_a * db1;

        float dc0 = d_C_rotated[i0];
        float dc1 = d_C_rotated[i1];
        d_C_pre_rope[i0] =  cos_a * dc0 + sin_a * dc1;
        d_C_pre_rope[i1] = -sin_a * dc0 + cos_a * dc1;

        // d_angle from Jacobian of rotation applied to B
        float b_pre0 = B_biased[i0];
        float b_pre1 = B_biased[i1];
        float d_angle_b = db0 * (-sin_a * b_pre0 - cos_a * b_pre1)
                        + db1 * ( cos_a * b_pre0 - sin_a * b_pre1);

        // d_angle from Jacobian of rotation applied to C
        float c_pre0 = C_biased[i0];
        float c_pre1 = C_biased[i1];
        float d_angle_c = dc0 * (-sin_a * c_pre0 - cos_a * c_pre1)
                        + dc1 * ( cos_a * c_pre0 - sin_a * c_pre1);

        d_angle_cumsum[angle_idx] = d_angle_b + d_angle_c;
    }
    // Odd indices: handled by even thread of the pair.
}

// ---------------------------------------------------------------------------
// Compute alpha/beta/gamma from dt, a_val (negative), trap (sigmoided)
// ---------------------------------------------------------------------------
// alpha = exp(a_val * dt)
// beta  = alpha * dt * (1 - trap)
// gamma = trap * dt
extern "C" __global__ void m3_compute_abg(
    float* __restrict__ alpha,
    float* __restrict__ beta,
    float* __restrict__ gamma,
    const float* __restrict__ dt,
    const float* __restrict__ a_val,
    const float* __restrict__ trap,
    int N  // total elements = batch * T * nheads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float dt_v = dt[idx];
    float a_v = a_val[idx];    // already negative (-softplus clamped)
    float t_v = trap[idx];     // already sigmoided
    float alpha_v = FAST_EXP(a_v * dt_v);
    alpha[idx] = alpha_v;
    beta[idx] = alpha_v * dt_v * (1.0f - t_v);
    gamma[idx] = t_v * dt_v;
}

// ============================================================================
// 9. m3_abg_bwd -- Backward through alpha/beta/gamma + softplus/sigmoid
// ============================================================================
//
// Forward (m3_compute_abg + m3_split activations):
//   dt_val = softplus(dd_dt_raw + dt_bias)
//   a_val  = clamp(-softplus(dd_a_raw), max = -a_floor)
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
extern "C" __global__ void m3_abg_bwd(
    float* __restrict__ d_dd_dt,       // [N] -- gradient w.r.t. dd_dt_raw (proj component)
    float* __restrict__ d_dd_a,        // [N] -- gradient w.r.t. dd_a_raw (proj component)
    float* __restrict__ d_trap_out,    // [N] -- gradient w.r.t. trap_raw (proj component)
    const float* __restrict__ d_alpha, // [N]
    const float* __restrict__ d_beta,  // [N]
    const float* __restrict__ d_gamma, // [N]
    const float* __restrict__ d_dt_angle, // [N] -- angle contribution to d_dt (from angle_dt_bwd)
    const float* __restrict__ dt,      // [N] -- post-softplus dt
    const float* __restrict__ a_val,   // [N] -- clamped -softplus(dd_a)
    const float* __restrict__ alpha,   // [N] -- exp(a_val * dt)
    const float* __restrict__ dd_dt_raw, // [N] -- saved raw (pre-softplus)
    const float* __restrict__ dd_a_raw,  // [N] -- saved raw (pre-softplus)
    const float* __restrict__ trap_raw,  // [N] -- saved raw (pre-sigmoid)
    const float* __restrict__ dt_bias, // [nh]
    float a_floor,
    int N, int nh
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int h = idx % nh;

    float da = d_alpha[idx];
    float db = d_beta[idx];
    float dg = d_gamma[idx];
    float ddt_angle = d_dt_angle[idx];

    float dt_v   = dt[idx];
    float a_v    = a_val[idx];    // already negative, clamped
    float alpha_v = alpha[idx];
    float tr_raw = trap_raw[idx];
    float trap_sig = 1.0f / (1.0f + FAST_EXP(-tr_raw));

    // d_adt = d_alpha * alpha + d_beta * beta
    //   (since d(beta)/d(adt) = d(alpha*dt*(1-trap))/d(adt) = alpha*dt*(1-trap) = beta)
    float d_adt = da * alpha_v + db * alpha_v * dt_v * (1.0f - trap_sig);

    // d_dt contributions
    float d_dt_from_adt   = d_adt * a_v;
    float d_dt_from_beta  = db * alpha_v * (1.0f - trap_sig);
    float d_dt_from_gamma = dg * trap_sig;
    float d_dt_total = d_dt_from_adt + d_dt_from_beta + d_dt_from_gamma + ddt_angle;

    // softplus backward for DT: d_dd_dt = d_dt_total * sigmoid(dd_dt_raw + dt_bias)
    float dt_pre = dd_dt_raw[idx] + dt_bias[h];
    float sp_deriv_dt = (dt_pre > 20.0f) ? 1.0f : (1.0f / (1.0f + FAST_EXP(-dt_pre)));
    d_dd_dt[idx] = d_dt_total * sp_deriv_dt;

    // d_trap_sig from beta and gamma paths
    float d_trap_sig = -db * alpha_v * dt_v + dg * dt_v;
    // sigmoid backward: d_trap_raw = d_trap_sig * sig * (1 - sig)
    d_trap_out[idx] = d_trap_sig * trap_sig * (1.0f - trap_sig);

    // d_a_val = d_adt * dt_val
    float d_a_val = d_adt * dt_v;
    // A = -softplus(dd_a_raw), clamped to max = -a_floor
    // If clamped (a_val == -a_floor), gradient is zero.
    float raw_a = dd_a_raw[idx];
    float sp_a = (raw_a > 20.0f) ? raw_a : logf(1.0f + FAST_EXP(raw_a));
    float a_unclamped = -sp_a;
    if (a_unclamped > -a_floor) {
        // Was clamped: gradient is zero
        d_dd_a[idx] = 0.0f;
    } else {
        float sp_deriv_a = (raw_a > 20.0f) ? 1.0f : (1.0f / (1.0f + FAST_EXP(-raw_a)));
        // d(-softplus(x))/dx = -sigmoid(x)
        d_dd_a[idx] = d_a_val * (-sp_deriv_a);
    }
}

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
extern "C" __global__ void bc_bias_add_bwd(
    float* __restrict__ d_B_normed,     // [N * ng * ds]
    const float* __restrict__ d_B_biased, // [N * nh * ds]
    int N, int nh, int ng, int ds
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * ng * ds;
    if (idx >= total) return;

    int ng_ds = ng * ds;
    int sample = idx / ng_ds;
    int rem = idx % ng_ds;
    int g = rem / ds;
    int d = rem % ds;

    int heads_per_group = nh / ng;
    float sum = 0.0f;
    int base = sample * nh * ds;
    for (int local_h = 0; local_h < heads_per_group; local_h++) {
        int h = g * heads_per_group + local_h;
        sum += d_B_biased[base + h * ds + d];
    }
    d_B_normed[idx] = sum;
}

// ============================================================================
// 11. m3_split_bwd -- Assemble d_proj from 8 gradient components
// ============================================================================
//
// Reverse of m3_split: writes each gradient component at its correct offset.
// Layout: [d_z: di | d_x: di | d_B_raw: ng*ds | d_C_raw: ng*ds |
//          d_dd_dt: nh | d_dd_A: nh | d_trap: nh | d_angles: n_angles]
//
// Grid: ceil(N * in_proj_dim / 256) blocks, 256 threads
extern "C" __global__ void m3_split_bwd(
    float* __restrict__ d_proj,         // [N * in_proj_dim]
    const float* __restrict__ d_z,      // [N * di]
    const float* __restrict__ d_x,      // [N * di]
    const float* __restrict__ d_B_raw,  // [N * ng * ds]
    const float* __restrict__ d_C_raw,  // [N * ng * ds]
    const float* __restrict__ d_dd_dt,  // [N * nh]
    const float* __restrict__ d_dd_a,   // [N * nh]
    const float* __restrict__ d_trap,   // [N * nh]
    const float* __restrict__ d_angles, // [N * n_angles]
    int N, int di, int ng, int ds, int nh, int n_angles
) {
    int in_proj_dim = 2 * di + 2 * ng * ds + 3 * nh + n_angles;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * in_proj_dim;
    if (idx >= total) return;

    int sample = idx / in_proj_dim;
    int col = idx % in_proj_dim;

    int ng_ds = ng * ds;
    int off0 = di;
    int off1 = 2 * di;
    int off2 = off1 + ng_ds;
    int off3 = off2 + ng_ds;
    int off4 = off3 + nh;
    int off5 = off4 + nh;
    int off6 = off5 + nh;

    float val;
    if (col < off0) {
        val = d_z[sample * di + col];
    } else if (col < off1) {
        val = d_x[sample * di + (col - off0)];
    } else if (col < off2) {
        val = d_B_raw[sample * ng_ds + (col - off1)];
    } else if (col < off3) {
        val = d_C_raw[sample * ng_ds + (col - off2)];
    } else if (col < off4) {
        val = d_dd_dt[sample * nh + (col - off3)];
    } else if (col < off5) {
        val = d_dd_a[sample * nh + (col - off4)];
    } else if (col < off6) {
        val = d_trap[sample * nh + (col - off5)];
    } else {
        int a = col - off6;
        val = (a < n_angles) ? d_angles[sample * n_angles + a] : 0.0f;
    }
    d_proj[idx] = val;
}
// RMSNormGated forward: output = RMSNorm(y) * weight * SiLU(z)
// norm_before_gate=True (Mamba-3 style): normalize y first, then gate.
// Source: layernorm_gated.py rms_norm_ref, Mamba-3 default.
// Plain SiLU gating: out[i] = y[i] * silu(z[i])
// Used when is_outproj_norm=false (default).
// Source: Python mamba3_siso_step.py line 219: out = out * silu(z)
extern "C" __global__ void silu_gate_fwd(
    float* out, const float* y, const float* z, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float z_val = z[i];
    float silu_z = z_val / (1.0f + exp2f(-z_val * 1.4426950408889634f));
    out[i] = y[i] * silu_z;
}

// SiLU gating backward: d_y[i] = d_out[i] * silu(z[i]), d_z[i] = d_out[i] * y[i] * silu'(z[i])
extern "C" __global__ void silu_gate_bwd(
    float* d_y, float* d_z, const float* d_out,
    const float* y, const float* z, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float z_val = z[i];
    float sigma = 1.0f / (1.0f + exp2f(-z_val * 1.4426950408889634f));
    float silu_z = z_val * sigma;
    float d_out_val = d_out[i];
    d_y[i] = d_out_val * silu_z;
    d_z[i] = d_out_val * y[i] * sigma * (1.0f + z_val * (1.0f - sigma));
}

// Per-group RMSNormGated forward (group_size = headdim).
// Each block processes one sample. Threads are partitioned into groups of `group_size`.
// Each group computes its own RMS normalization independently.
// rms_vals: [N * n_groups] where n_groups = d_inner / group_size.
//
// Block: d_inner threads (one per channel). Each block processes one sample.
extern "C" __global__ void rmsnorm_gated_forward(
    float* out,           // [N * d_inner]
    float* rms_vals,      // [N * n_groups] — saved rstd per group for backward
    const float* y,       // [N * d_inner]
    const float* z,       // [N * d_inner]
    const float* weight,  // [d_inner]
    int N, int d_inner, int group_size
) {
    if (d_inner > 1024) return; // Guard: shared memory limit
    int sample = blockIdx.x;
    if (sample >= N) return;
    int d = threadIdx.x;
    if (d >= d_inner) return;

    int n_groups = d_inner / group_size;
    int group_id = d / group_size;
    int local_id = d % group_size;

    int base = sample * d_inner;
    float y_val = y[base + d];

    // Shared memory: one region per group, each of size group_size
    // Layout: shared[group_id * group_size + local_id]
    extern __shared__ float shared_sum[];
    shared_sum[group_id * group_size + local_id] = y_val * y_val;
    __syncthreads();

    // Per-group parallel reduction
    int stride = 1;
    while (stride < group_size) stride <<= 1;
    stride >>= 1;
    for (; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            int idx = group_id * group_size + local_id;
            shared_sum[idx] += shared_sum[idx + stride];
        }
        __syncthreads();
    }

    float rstd = rsqrtf(shared_sum[group_id * group_size] / (float)group_size + RMS_EPS);
    // Finite-guard (mirrors norms.cu RMSNorm fix): NaN/Inf in y would produce
    // rstd=NaN and cascade through the output gating into residual stream.
    if (!isfinite(rstd) || rstd > 1e20f) rstd = 1.0f;
    if (local_id == 0) rms_vals[sample * n_groups + group_id] = rstd;
    __syncthreads();

    // output = (y * rstd * weight) * SiLU(z)
    float z_val = z[base + d];
    float silu_z = z_val / (1.0f + FAST_EXP(-z_val));
    out[base + d] = y_val * rstd * weight[d] * silu_z;
}

// Per-group RMSNormGated backward (norm_before_gate=True, Mamba-3 style).
// Forward: out = (y * rstd_group * weight) * SiLU(z)
// Follows layernorm_gated.py _layer_norm_bwd_kernel NORM_BEFORE_GATE=True path.
// Per-group: each group of `group_size` elements has its own rstd.
extern "C" __global__ void rmsnorm_gated_backward(
    float* d_y,           // [N * d_inner] output
    float* d_z,           // [N * d_inner] output
    float* d_weight,      // [N * d_inner] per-sample (needs reduction across N)
    const float* d_out,   // [N * d_inner] incoming gradient
    const float* y,       // [N * d_inner] saved from forward
    const float* z,       // [N * d_inner] saved from forward
    const float* weight,  // [d_inner]
    const float* rms_vals,// [N * n_groups] — saved rstd per group from forward
    int N, int d_inner, int group_size
) {
    if (d_inner > 1024) return; // Guard: shared memory limit
    int sample = blockIdx.x;
    if (sample >= N) return;
    int d = threadIdx.x;
    if (d >= d_inner) return;

    int n_groups = d_inner / group_size;
    int group_id = d / group_size;
    int local_id = d % group_size;

    int base = sample * d_inner;
    float rstd = rms_vals[sample * n_groups + group_id];
    float y_val = y[base + d];
    float z_val = z[base + d];
    float w = weight[d];
    float d_out_val = d_out[base + d];

    // Recompute: y_hat = y * rstd, y_normed = y_hat * w
    float y_hat = y_val * rstd;
    float y_normed = y_hat * w;

    // d_z: gradient through SiLU gate
    // out = y_normed * SiLU(z), so d_z = d_out * y_normed * d_SiLU(z)
    float sig_z = 1.0f / (1.0f + FAST_EXP(-z_val));
    float silu_z = z_val * sig_z;
    float d_silu = sig_z + z_val * sig_z * (1.0f - sig_z);
    d_z[base + d] = d_out_val * y_normed * d_silu;

    // dy_scaled = d_out * SiLU(z) — gradient passed through the gate for RMSNorm backward
    float dy_scaled = d_out_val * silu_z;

    // d_weight (per-sample): d_w = dy_scaled * y_hat
    d_weight[base + d] = dy_scaled * y_hat;

    // RMSNorm backward per group: d_y = (w * dy_scaled - y_hat * c1) * rstd
    // c1 = sum(y_hat * w * dy_scaled) / group_size (within group only)
    extern __shared__ float shared_sum[];
    shared_sum[group_id * group_size + local_id] = y_hat * w * dy_scaled;
    __syncthreads();

    // Per-group parallel reduction
    int stride = 1;
    while (stride < group_size) stride <<= 1;
    stride >>= 1;
    for (; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            int idx = group_id * group_size + local_id;
            shared_sum[idx] += shared_sum[idx + stride];
        }
        __syncthreads();
    }

    float c1 = shared_sum[group_id * group_size] / (float)group_size;
    d_y[base + d] = (w * dy_scaled - y_hat * c1) * rstd;
}

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
// - Recurrent state: f32 (see mamba3_ssd.cu m3_step_fwd templated variant)
// - Residual stream: f32 (HF residual_in_fp32=True)
//
// Backward/training tensors (dd_*_raw, trap_raw saves) are not used in
// inference — the templated variants still accept them as f32 pointers so
// callers can pass unused scratch without branching.
// ============================================================================

// ------- m3_split typed -------
// proj: T_ACT input; outputs split into bf16 activations + f32 coefficients.
#define DEFINE_M3_SPLIT(SUFFIX, T_ACT, FROM_F)                                \
extern "C" __global__ void m3_split_##SUFFIX(                                  \
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                            \
    int total = N * in_proj_dim;                                                \
    if (idx >= total) return;                                                   \
    int sample = idx / in_proj_dim;                                             \
    int col = idx % in_proj_dim;                                                \
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
        z[sample * di + col] = FROM_F(val);                                     \
    } else if (col < off1) {                                                    \
        x[sample * di + (col - off0)] = FROM_F(val);                            \
    } else if (col < off2) {                                                    \
        B_raw[sample * ng_ds + (col - off1)] = FROM_F(val);                     \
    } else if (col < off3) {                                                    \
        C_raw[sample * ng_ds + (col - off2)] = FROM_F(val);                     \
    } else if (col < off4) {                                                    \
        int h = col - off3;                                                     \
        int dt_idx = sample * nh + h;                                           \
        dd_dt_raw[dt_idx] = val;                                                \
        float biased = val + dt_bias[h];                                        \
        dt[dt_idx] = (biased > 20.0f) ? biased : logf(1.0f + FAST_EXP(biased)); \
    } else if (col < off5) {                                                    \
        int h = col - off4;                                                     \
        int a_idx = sample * nh + h;                                            \
        dd_a_raw[a_idx] = val;                                                  \
        float sp = (val > 20.0f) ? val : logf(1.0f + FAST_EXP(val));            \
        float a = -sp;                                                          \
        a_val[a_idx] = fminf(a, -a_floor);                                      \
    } else if (col < off6) {                                                    \
        int h = col - off5;                                                     \
        int t_idx = sample * nh + h;                                            \
        trap_raw[t_idx] = val;                                                  \
        trap[t_idx] = 1.0f / (1.0f + FAST_EXP(-val));                           \
    } else {                                                                    \
        int a = col - off6;                                                     \
        if (a < n_angles) {                                                     \
            angles[sample * n_angles + a] = val;                                \
        }                                                                       \
    }                                                                           \
}

DEFINE_M3_SPLIT(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M3_SPLIT(f16,  __half,        from_f_f16)

// ------- bcnorm_fwd typed -------
// B_raw: T_ACT in; B_normed: T_ACT out; rms_val f32; weight f32.
#define DEFINE_BCNORM_FWD(SUFFIX, T_ACT, FROM_F)                               \
extern "C" __global__ void bcnorm_fwd_##SUFFIX(                                 \
    T_ACT* __restrict__ B_normed,                                               \
    float* __restrict__ rms_val,                                                \
    const T_ACT* __restrict__ B_raw,                                            \
    const float* __restrict__ weight,                                           \
    int N, int ng, int ds                                                       \
) {                                                                             \
    int block_id = blockIdx.x;                                                  \
    if (block_id >= N * ng) return;                                             \
    int d = threadIdx.x;                                                        \
    if (d >= ds) return;                                                        \
    int base = block_id * ds;                                                   \
    float val = to_f(B_raw[base + d]);                                          \
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
    float rms = sqrtf(sdata[0] / (float)ds + RMS_EPS);                          \
    if (d == 0) rms_val[block_id] = rms;                                        \
    __syncthreads();                                                            \
    float inv_rms = 1.0f / rms;                                                 \
    B_normed[base + d] = FROM_F(val * inv_rms * weight[d]);                     \
}

DEFINE_BCNORM_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_BCNORM_FWD(f16,  __half,        from_f_f16)

// ------- bcnorm_fwd_bc fused (B + C in one launch) -------
// Same per-(sample, group) RMSNorm as bcnorm_fwd, but processes B and C
// concurrently via gridDim.y ∈ {0, 1}. Saves a kernel launch per layer
// per step (~3-5 µs each on Ada). Identical math to two sequential bcnorm
// calls; tested via finite-diff parity with the unfused path.
#define DEFINE_BCNORM_FWD_BC(SUFFIX, T_ACT, FROM_F)                            \
extern "C" __global__ void bcnorm_fwd_bc_##SUFFIX(                              \
    T_ACT* __restrict__ B_normed,                                               \
    T_ACT* __restrict__ C_normed,                                               \
    float* __restrict__ B_rms,                                                  \
    float* __restrict__ C_rms,                                                  \
    const T_ACT* __restrict__ B_raw,                                            \
    const T_ACT* __restrict__ C_raw,                                            \
    const float* __restrict__ B_weight,                                         \
    const float* __restrict__ C_weight,                                         \
    int N, int ng, int ds                                                       \
) {                                                                             \
    /* gridDim.y == 2: 0 → B path, 1 → C path */                                \
    int which = blockIdx.y;                                                     \
    int block_id = blockIdx.x;                                                  \
    if (block_id >= N * ng) return;                                             \
    int d = threadIdx.x;                                                        \
    if (d >= ds) return;                                                        \
    const T_ACT* raw = (which == 0) ? B_raw : C_raw;                            \
    T_ACT* normed = (which == 0) ? B_normed : C_normed;                         \
    float* rms_out = (which == 0) ? B_rms : C_rms;                              \
    const float* weight = (which == 0) ? B_weight : C_weight;                   \
    int base = block_id * ds;                                                   \
    float val = to_f(raw[base + d]);                                            \
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
    float rms = sqrtf(sdata[0] / (float)ds + RMS_EPS);                          \
    if (!isfinite(rms) || rms < 1e-20f) rms = 1.0f;                             \
    if (d == 0) rms_out[block_id] = rms;                                        \
    __syncthreads();                                                            \
    float inv_rms = 1.0f / rms;                                                 \
    normed[base + d] = FROM_F(val * inv_rms * weight[d]);                       \
}

DEFINE_BCNORM_FWD_BC(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_BCNORM_FWD_BC(f16,  __half,        from_f_f16)

// ------- bc_bias_add typed -------
// B_normed: T_ACT in; B_biased: T_ACT out; bias f32.
#define DEFINE_BC_BIAS_ADD(SUFFIX, T_ACT, FROM_F)                              \
extern "C" __global__ void bc_bias_add_##SUFFIX(                                \
    T_ACT* __restrict__ B_biased,                                               \
    const T_ACT* __restrict__ B_normed,                                         \
    const float* __restrict__ bias,                                             \
    int N, int nh, int ng, int ds                                               \
) {                                                                             \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                            \
    int total = N * nh * ds;                                                    \
    if (idx >= total) return;                                                   \
    int nh_ds = nh * ds;                                                        \
    int sample = idx / nh_ds;                                                   \
    int rem = idx % nh_ds;                                                      \
    int h = rem / ds;                                                           \
    int n = rem % ds;                                                           \
    int heads_per_group = nh / ng;                                              \
    int g = h / heads_per_group;                                                \
    float nv = to_f(B_normed[sample * ng * ds + g * ds + n]);                   \
    B_biased[idx] = FROM_F(nv + bias[h * ds + n]);                              \
}

DEFINE_BC_BIAS_ADD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_BC_BIAS_ADD(f16,  __half,        from_f_f16)

// ------- bc_bias_add_bc fused (B + C in one launch) -------
// Same per-(sample, head, n) bias add as bc_bias_add, but processes B and C
// concurrently via a 2× grid extension. Saves a launch per layer per step.
// Identical math to two sequential bc_bias_add calls.
#define DEFINE_BC_BIAS_ADD_BC(SUFFIX, T_ACT, FROM_F)                           \
extern "C" __global__ void bc_bias_add_bc_##SUFFIX(                             \
    T_ACT* __restrict__ B_biased,                                               \
    T_ACT* __restrict__ C_biased,                                               \
    const T_ACT* __restrict__ B_normed,                                         \
    const T_ACT* __restrict__ C_normed,                                         \
    const float* __restrict__ B_bias,                                           \
    const float* __restrict__ C_bias,                                           \
    int N, int nh, int ng, int ds                                               \
) {                                                                             \
    /* gridDim.y == 2: 0 → B path, 1 → C path */                                \
    int which = blockIdx.y;                                                     \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                            \
    int total = N * nh * ds;                                                    \
    if (idx >= total) return;                                                   \
    const T_ACT* normed = (which == 0) ? B_normed : C_normed;                   \
    T_ACT* biased = (which == 0) ? B_biased : C_biased;                         \
    const float* bias = (which == 0) ? B_bias : C_bias;                         \
    int nh_ds = nh * ds;                                                        \
    int sample = idx / nh_ds;                                                   \
    int rem = idx % nh_ds;                                                      \
    int h = rem / ds;                                                           \
    int n = rem % ds;                                                           \
    int heads_per_group = nh / ng;                                              \
    int g = h / heads_per_group;                                                \
    float nv = to_f(normed[sample * ng * ds + g * ds + n]);                     \
    biased[idx] = FROM_F(nv + bias[h * ds + n]);                                \
}

DEFINE_BC_BIAS_ADD_BC(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_BC_BIAS_ADD_BC(f16,  __half,        from_f_f16)

// ------- rope_fwd typed -------
// B_biased/C_biased: T_ACT in; B_rotated/C_rotated: T_ACT out;
// angle_cumsum f32 (coefficient, keeps f64-accumulated precision upstream).
#define DEFINE_ROPE_FWD(SUFFIX, T_ACT, FROM_F)                                 \
extern "C" __global__ void rope_fwd_##SUFFIX(                                   \
    T_ACT* __restrict__ B_rotated,                                              \
    T_ACT* __restrict__ C_rotated,                                              \
    const T_ACT* __restrict__ B_biased,                                         \
    const T_ACT* __restrict__ C_biased,                                         \
    const float* __restrict__ angle_cumsum,                                     \
    int N, int nh, int ds, int n_angles                                         \
) {                                                                             \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                            \
    int total = N * nh * ds;                                                    \
    if (idx >= total) return;                                                   \
    int nh_ds = nh * ds;                                                        \
    int sample = idx / nh_ds;                                                   \
    int rem = idx % nh_ds;                                                      \
    int h = rem / ds;                                                           \
    int n = rem % ds;                                                           \
    int base_bc = sample * nh_ds + h * ds;                                      \
    int rope_end = 2 * n_angles;                                                \
    if (n >= rope_end) {                                                        \
        B_rotated[idx] = B_biased[idx];                                         \
        C_rotated[idx] = C_biased[idx];                                         \
    } else if ((n & 1) == 0) {                                                  \
        int a = n / 2;                                                          \
        int angle_idx = sample * nh * n_angles + h * n_angles + a;              \
        float cos_a, sin_a;                                                     \
        sincosf(angle_cumsum[angle_idx], &sin_a, &cos_a);                       \
        int i0 = base_bc + n;                                                   \
        int i1 = i0 + 1;                                                        \
        float b0 = to_f(B_biased[i0]);                                          \
        float b1 = to_f(B_biased[i1]);                                          \
        B_rotated[i0] = FROM_F(cos_a * b0 - sin_a * b1);                        \
        B_rotated[i1] = FROM_F(sin_a * b0 + cos_a * b1);                        \
        float c0 = to_f(C_biased[i0]);                                          \
        float c1 = to_f(C_biased[i1]);                                          \
        C_rotated[i0] = FROM_F(cos_a * c0 - sin_a * c1);                        \
        C_rotated[i1] = FROM_F(sin_a * c0 + cos_a * c1);                        \
    }                                                                           \
}

DEFINE_ROPE_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_ROPE_FWD(f16,  __half,        from_f_f16)

// ------- silu_gate_fwd typed -------
// y, z, out: T_ACT. Simple elementwise.
#define DEFINE_SILU_GATE_FWD(SUFFIX, T_ACT, FROM_F)                            \
extern "C" __global__ void silu_gate_fwd_##SUFFIX(                              \
    T_ACT* __restrict__ out,                                                    \
    const T_ACT* __restrict__ y,                                                \
    const T_ACT* __restrict__ z,                                                \
    int n                                                                       \
) {                                                                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (i >= n) return;                                                         \
    float z_val = to_f(z[i]);                                                   \
    float silu_z = z_val / (1.0f + exp2f(-z_val * LOG2E));                      \
    out[i] = FROM_F(to_f(y[i]) * silu_z);                                       \
}

DEFINE_SILU_GATE_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SILU_GATE_FWD(f16,  __half,        from_f_f16)

// ------- rmsnorm_gated_forward typed -------
// y, z, out: T_ACT; weight, rms_vals: f32. Per-group RMSNorm * SiLU(z).
#define DEFINE_RMSNORM_GATED_FWD(SUFFIX, T_ACT, FROM_F)                        \
extern "C" __global__ void rmsnorm_gated_forward_##SUFFIX(                      \
    T_ACT* __restrict__ out,                                                    \
    float* __restrict__ rms_vals,                                               \
    const T_ACT* __restrict__ y,                                                \
    const T_ACT* __restrict__ z,                                                \
    const float* __restrict__ weight,                                           \
    int N, int d_inner, int group_size                                          \
) {                                                                             \
    if (d_inner > 1024) return;                                                 \
    int sample = blockIdx.x;                                                    \
    if (sample >= N) return;                                                    \
    int d = threadIdx.x;                                                        \
    if (d >= d_inner) return;                                                   \
    int n_groups = d_inner / group_size;                                        \
    int group_id = d / group_size;                                              \
    int local_id = d % group_size;                                              \
    int base = sample * d_inner;                                                \
    float y_val = to_f(y[base + d]);                                            \
    extern __shared__ float shared_sum[];                                       \
    shared_sum[group_id * group_size + local_id] = y_val * y_val;               \
    __syncthreads();                                                            \
    int stride = 1;                                                             \
    while (stride < group_size) stride <<= 1;                                   \
    stride >>= 1;                                                               \
    for (; stride > 0; stride >>= 1) {                                          \
        if (local_id < stride) {                                                \
            int sidx = group_id * group_size + local_id;                        \
            shared_sum[sidx] += shared_sum[sidx + stride];                      \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
    float rstd = rsqrtf(shared_sum[group_id * group_size] / (float)group_size + RMS_EPS); \
    if (local_id == 0) rms_vals[sample * n_groups + group_id] = rstd;           \
    __syncthreads();                                                            \
    float z_val = to_f(z[base + d]);                                            \
    float silu_z = z_val / (1.0f + FAST_EXP(-z_val));                           \
    out[base + d] = FROM_F(y_val * rstd * weight[d] * silu_z);                  \
}

DEFINE_RMSNORM_GATED_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_RMSNORM_GATED_FWD(f16,  __half,        from_f_f16)
