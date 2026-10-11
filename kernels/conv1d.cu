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
//
// Both step kernels below take a registerized d_conv == 4 fast path (the
// production value): the shift happens in registers and global memory
// sees one window load, one writeback and one output store - the generic
// path pays three global read-modify-write shifts plus four reloads of
// what it just wrote, a read-after-write hazard the compiler cannot
// forward through. Value contract: the accumulation is the k-ascending
// chain of the generic loop, term for term; an f32 held in a register
// carries the exact bits a global round-trip would have returned.

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
    T* __restrict__ out,                                                   \
    float* __restrict__ state,                                             \
    /* Row stride of new_x: d_inner for a packed input, 2 * d_inner when   \
       the kernel reads the x half of the in_proj output directly. */      \
    const T* __restrict__ new_x,                                           \
    int x_stride,                                                          \
    const float* __restrict__ weight,                                      \
    const float* __restrict__ bias,                                        \
    int batch, int d_inner, int d_conv                                     \
) {                                                                        \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    int total = batch * d_inner;                                           \
    if (idx >= total) return;                                              \
    int b = idx / d_inner;                                                 \
    int d = idx % d_inner;                                                 \
    int x_idx = b * x_stride + d;                                          \
    int state_base = (b * d_inner + d) * d_conv;                           \
    /* Registerized d_conv == 4 fast path - see the note above. */          \
    if (d_conv == 4) {                                                     \
        float s0 = state[state_base + 1];                                  \
        float s1 = state[state_base + 2];                                  \
        float s2 = state[state_base + 3];                                  \
        float s3 = to_f(new_x[x_idx]);                                     \
        state[state_base] = s0;                                            \
        state[state_base + 1] = s1;                                        \
        state[state_base + 2] = s2;                                        \
        state[state_base + 3] = s3;                                        \
        float sum = bias[d];                                               \
        sum += s0 * weight[d * 4];                                         \
        sum += s1 * weight[d * 4 + 1];                                     \
        sum += s2 * weight[d * 4 + 2];                                     \
        sum += s3 * weight[d * 4 + 3];                                     \
        float silu4 = sum / (1.0f + exp2f(-sum * LOG2E));                  \
        out[idx] = FROM_F(silu4);                                          \
        return;                                                            \
    }                                                                      \
    for (int k = 0; k < d_conv - 1; k++) {                                 \
        state[state_base + k] = state[state_base + k + 1];                 \
    }                                                                      \
    state[state_base + d_conv - 1] = to_f(new_x[x_idx]);                   \
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

// ======================== BURNIN (T>1) ========================

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
// (b, d) with a T-long serial loop — 24 blocks at the production shape, 146
// idle SMs. The window only reaches d_conv-1 = 3 steps back, so the t-range
// tiles perfectly: tile 0 seeds from the carry-in state, tiles k>0 seed
// their window from x_branch halo loads — the SAME values the serial walk
// would hold, so every output element is bit-identical. Only the tile
// containing t = T-1 writes the carry-out state.
//
// The forward tiles are CONV1D_FWD_TILE_T deep; the backward keeps
// CONV1D_TILE_T, which also sets its partial-row count and with it the bits
// of the tap gradients. Within a forward tile the walk loads the inputs of
// CONV1D_FWD_BATCH steps before it computes and stores them: each step's
// load no longer waits for the previous step's store, and the taps and bias
// sit in registers. The usual four-tap window is unrolled at compile time
// so it stays in registers too; any other d_conv walks the same chain with
// a runtime bound. The arithmetic is the serial walk's in every case.
// ============================================================================
#define CONV1D_TILE_T 128
#define CONV1D_FWD_TILE_T 32
#define CONV1D_FWD_BATCH 8

// The walk of one forward tile over [t0, t_end): WIN holds the window
// before t0, W takes the taps and bias_d is the bias of channel d. The
// four-tap walk gets arrays of its own: an array indexed by a runtime bound
// anywhere lives in local memory everywhere.
#define CONV1D_FWD_WALK(DC, FROM_F, WIN, W)                                   \
    for (int k = 0; k < (DC); k++) W[k] = weight[d * d_conv + k];            \
    for (int tb = t0; tb < t_end; tb += CONV1D_FWD_BATCH) {                  \
        float xs[CONV1D_FWD_BATCH];                                          \
        _Pragma("unroll")                                                    \
        for (int j = 0; j < CONV1D_FWD_BATCH; j++) {                         \
            int t = tb + j;                                                  \
            xs[j] = (t < t_end)                                              \
                ? to_f(x_branch[(b * T_len + t) * x_stride + d]) : 0.0f;     \
        }                                                                    \
        _Pragma("unroll")                                                    \
        for (int j = 0; j < CONV1D_FWD_BATCH; j++) {                         \
            int t = tb + j;                                                  \
            if (t < t_end) {                                                 \
                for (int k = 0; k < (DC) - 1; k++) WIN[k] = WIN[k + 1];      \
                WIN[(DC) - 1] = xs[j];                                       \
                float val = bias_d;                                          \
                for (int k = 0; k < (DC); k++) val += WIN[k] * W[k];         \
                u_out[(b * T_len + t) * d_inner + d] =                       \
                    FROM_F(val / (1.0f + exp2f(-val * 1.4426950408889634f))); \
            }                                                                \
        }                                                                    \
    }

#define DEFINE_CONV1D_BURNIN_TILED(SUFFIX, TY, FROM_F)                        \
extern "C" __global__ void conv1d_burnin_forward_tiled_##SUFFIX(              \
    TY* __restrict__ u_out, float* __restrict__ state,                       \
    float* __restrict__ conv_states_saved,                                   \
    const TY* __restrict__ x_branch, const float* __restrict__ weight,       \
    const float* __restrict__ bias,                                          \
    int batch, int T_len, int d_inner, int d_conv,                            \
    int x_stride /* row stride of x_branch: d_inner, or 2*d_inner when        \
                    reading the in_proj output directly */                    \
) {                                                                          \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                         \
    int total = batch * d_inner;                                             \
    if (idx >= total) return;                                                \
    if (d_conv > 8) return;                                                  \
    int b = idx / d_inner;                                                   \
    int d = idx % d_inner;                                                   \
    int t0 = blockIdx.y * CONV1D_FWD_TILE_T;                                 \
    if (t0 >= T_len) return;                                                 \
    int t_end = min(t0 + CONV1D_FWD_TILE_T, T_len);                          \
    int state_base = (b * d_inner + d) * d_conv;                             \
    float win[8];                                                            \
    if (t0 == 0) {                                                           \
        for (int k = 0; k < d_conv; k++) {                                   \
            win[k] = state[state_base + k];                                  \
            /* The ONLY conv tape is the carry-in window — the      \
             * backward reconstructs every later window from x_branch. */   \
            conv_states_saved[state_base + k] = win[k];                      \
        }                                                                    \
        /* Carry-out, written by the block that owns the carry-in and      \
         * never by another: after T steps the serial walk holds           \
         * win[k] = x[T - d_conv + k], and the surviving tail of the       \
         * incoming window where that index is negative. Reading the       \
         * whole closed form before storing keeps this thread's own        \
         * carry-in intact. No other block reads or writes state, so       \
         * the launch carries no cross-block order. */                     \
        float carry[8];                                                      \
        for (int k = 0; k < d_conv; k++) {                                   \
            int th = T_len - d_conv + k;                                     \
            carry[k] = (th >= 0)                                             \
                ? to_f(x_branch[(b * T_len + th) * x_stride + d])            \
                : state[state_base + k + T_len];                             \
        }                                                                    \
        for (int k = 0; k < d_conv; k++) state[state_base + k] = carry[k];   \
    } else {                                                                 \
        /* Halo: the serial walk's window after step t0-1 holds           \
         * win[k] = x[t0 - d_conv + k] (win[0] is about to shift out).   \
         * t0 >= CONV1D_FWD_TILE_T > d_conv, so the halo never underruns. */ \
        for (int k = 0; k < d_conv; k++) {                                   \
            int th = t0 - d_conv + k;                                        \
            win[k] = to_f(x_branch[(b * T_len + th) * x_stride + d]);         \
        }                                                                    \
    }                                                                        \
    float bias_d = bias[d];                                                  \
    if (d_conv == 4) {                                                       \
        float win4[4] = {win[0], win[1], win[2], win[3]};                    \
        float w4[4];                                                         \
        CONV1D_FWD_WALK(4, FROM_F, win4, w4)                                 \
    } else {                                                                 \
        float w[8];                                                          \
        CONV1D_FWD_WALK(d_conv, FROM_F, win, w)                              \
    }                                                                        \
}

DEFINE_CONV1D_BURNIN_TILED(f32,  float,         from_f_f32)
DEFINE_CONV1D_BURNIN_TILED(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_CONV1D_BURNIN_TILED(f16,  __half,        from_f_f16)

// Nosave tiled twin for the inference prefill: same tile-0 state seeding,
// same x_branch halo, same k-ascending FMA chain - bit-identical to the
// serial nosave walk - but no conv tape and no post_conv (inference keeps
// nothing). Only the tile containing t = T-1 writes the carry-out state,
// so a continued prefill sees exactly the serial walk's final window.
#define DEFINE_CONV1D_BURNIN_NOSAVE_TILED(SUFFIX, TY, FROM_F)                 \
extern "C" __global__ void conv1d_burnin_forward_nosave_tiled_##SUFFIX(       \
    TY* __restrict__ u_out, float* __restrict__ state,                       \
    const TY* __restrict__ x_branch, const float* __restrict__ weight,       \
    const float* __restrict__ bias,                                          \
    int batch, int T_len, int d_inner, int d_conv,                           \
    int x_stride /* row stride of x_branch; d_inner, or 2*d_inner when     \
                    reading the in_proj output directly */                   \
) {                                                                          \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                         \
    int total = batch * d_inner;                                             \
    if (idx >= total) return;                                                \
    if (d_conv > 8) return;                                                  \
    int b = idx / d_inner;                                                   \
    int d = idx % d_inner;                                                   \
    int t0 = blockIdx.y * CONV1D_FWD_TILE_T;                                 \
    if (t0 >= T_len) return;                                                 \
    int t_end = min(t0 + CONV1D_FWD_TILE_T, T_len);                          \
    int state_base = (b * d_inner + d) * d_conv;                             \
    float win[8];                                                            \
    if (t0 == 0) {                                                           \
        for (int k = 0; k < d_conv; k++) {                                   \
            win[k] = state[state_base + k];                                  \
        }                                                                    \
        /* Carry-out, written by the block that owns the carry-in and      \
         * never by another: after T steps the serial walk holds           \
         * win[k] = x[T - d_conv + k], and the surviving tail of the       \
         * incoming window where that index is negative. Reading the       \
         * whole closed form before storing keeps this thread's own        \
         * carry-in intact. No other block reads or writes state, so       \
         * the launch carries no cross-block order. */                     \
        float carry[8];                                                      \
        for (int k = 0; k < d_conv; k++) {                                   \
            int th = T_len - d_conv + k;                                     \
            carry[k] = (th >= 0)                                             \
                ? to_f(x_branch[(b * T_len + th) * x_stride + d])            \
                : state[state_base + k + T_len];                             \
        }                                                                    \
        for (int k = 0; k < d_conv; k++) state[state_base + k] = carry[k];   \
    } else {                                                                 \
        for (int k = 0; k < d_conv; k++) {                                   \
            int th = t0 - d_conv + k;                                        \
            win[k] = to_f(x_branch[(b * T_len + th) * x_stride + d]);         \
        }                                                                    \
    }                                                                        \
    float bias_d = bias[d];                                                  \
    if (d_conv == 4) {                                                       \
        float win4[4] = {win[0], win[1], win[2], win[3]};                    \
        float w4[4];                                                         \
        CONV1D_FWD_WALK(4, FROM_F, win4, w4)                                 \
    } else {                                                                 \
        float w[8];                                                          \
        CONV1D_FWD_WALK(d_conv, FROM_F, win, w)                              \
    }                                                                        \
}

DEFINE_CONV1D_BURNIN_NOSAVE_TILED(f32,  float,         from_f_f32)
DEFINE_CONV1D_BURNIN_NOSAVE_TILED(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_CONV1D_BURNIN_NOSAVE_TILED(f16,  __half,        from_f_f16)

// The conv backward as one kernel per (b, d, tile). The pre-activation
// is recomputed from the x window instead of read from a tape, so the
// forward keeps no pre-activation copy; the SiLU derivative is taken once
// per element instead of once per tap lane; and d_x, the tap gradients and
// the bias gradient walk the tile together. d_x is the anticausal FIR with
// its carries seeded exactly as the serial walk would hold them at the
// tile boundary; the tap and bias accumulators keep the descending-t order
// within the tile and write one partial row each that the fixed-order
// reduction folds as before. The recomputed pre-activation is the
// forward's own chain on the forward's own window values, rounded to the
// activation type as the forward's store rounded it.
//
// The walk loads the x and d_u values of CONV1D_BWD_BATCH steps before it
// runs them, so a step's loads no longer wait for the step before it, and
// the usual four-tap walk keeps its window, carries, taps and tap
// accumulators in registers (an array indexed by a runtime bound anywhere
// lives in local memory everywhere). Every step's arithmetic is the one
// the step-by-step walk did, in the same order.
#define CONV1D_BWD_BATCH 8

// The descending walk from ts_top to t0 with DC taps over the arrays WIN,
// CARRY, DW and W (W preloaded with the taps).
#define CONV1D_BWD_WALK(DC, TY, FROM_F, WIN, CARRY, DW, W)                   \
    for (int tb = ts_top; tb >= t0; tb -= CONV1D_BWD_BATCH) {                \
        float xs[CONV1D_BWD_BATCH];                                          \
        float dus[CONV1D_BWD_BATCH];                                         \
        _Pragma("unroll")                                                    \
        for (int j = 0; j < CONV1D_BWD_BATCH; j++) {                         \
            int ts = tb - j;                                                 \
            int tx = ts - ((DC) - 1);                                        \
            bool live = (ts >= t0) && (ts < T_);                             \
            xs[j] = (live && ts < ts0)                                       \
                ? ((tx >= 0)                                                 \
                    ? to_f(x_branch[(b * T_ + tx) * x_stride + d])          \
                    : conv_init[init_base + tx + d_conv])                    \
                : 0.0f;                                                      \
            dus[j] = live ? to_f(d_u[(b * T_ + ts) * d_inner + d]) : 0.0f;   \
        }                                                                    \
        _Pragma("unroll")                                                    \
        for (int j = 0; j < CONV1D_BWD_BATCH; j++) {                         \
            int ts = tb - j;                                                 \
            if (ts < t0) break;                                              \
            float dco = 0.0f;                                                \
            float du = 0.0f;                                                 \
            float silu_grad = 0.0f;                                          \
            if (ts < T_) {                                                   \
                if (ts < ts0) {                                              \
                    for (int k = (DC) - 1; k >= 1; k--) WIN[k] = WIN[k - 1]; \
                    WIN[0] = xs[j];                                          \
                }                                                            \
                float val = bias_d;                                          \
                for (int k = 0; k < (DC); k++) {                             \
                    val += WIN[k] * W[k];                                    \
                }                                                            \
                float x = to_f(FROM_F(val));                                 \
                float sig = 1.0f / (1.0f + exp2f(-x * 1.4426950408889634f)); \
                silu_grad = sig * (1.0f + x * (1.0f - sig));                 \
                du = dus[j];                                                 \
                dco = du * silu_grad;                                        \
            }                                                                \
            if (ts < t_end) {                                                \
                for (int k = 0; k < (DC); k++) DW[k] += dco * WIN[k];        \
                /* explicit two-rounding shape, as the bias gradient       \
                   always had */                                           \
                local_d_bias =                                               \
                    __fadd_rn(local_d_bias, __fmul_rn(du, silu_grad));       \
                /* Two roundings, as the step-by-step walk did with its \
                   runtime-guarded add: a compile-time tap count would let \
                   the compiler fuse them into one. */                     \
                float dxb = __fmul_rn(dco, W[(DC) - 1]);                     \
                if ((DC) > 1) {                                              \
                    dxb = __fadd_rn(dxb, CARRY[0]);                          \
                    for (int k = 0; k < (DC) - 2; k++) {                     \
                        CARRY[k] = CARRY[k + 1] + dco * W[(DC) - 2 - k];     \
                    }                                                        \
                    CARRY[(DC) - 2] = dco * W[0];                            \
                }                                                            \
                d_x_branch[(b * T_ + ts) * out_stride + out_offset + d] =    \
                    FROM_F(dxb);                                             \
            } else if ((DC) > 1) {                                           \
                for (int k = 0; k < (DC) - 2; k++) {                         \
                    CARRY[k] = CARRY[k + 1] + dco * W[(DC) - 2 - k];         \
                }                                                            \
                CARRY[(DC) - 2] = dco * W[0];                                \
            }                                                                \
        }                                                                    \
    }

#define DEFINE_CONV1D_BWD_TILED(SUFFIX, TY, FROM_F)                           \
extern "C" __global__ void conv1d_bwd_tiled_##SUFFIX(                         \
    TY* __restrict__ d_x_branch,                                              \
    float* __restrict__ d_weight_partials, /* [B*n_tiles, di*d_conv] */       \
    float* __restrict__ d_bias_partials,   /* [B*n_tiles, di] */              \
    const TY* __restrict__ d_u,                                               \
    const TY* __restrict__ x_branch,                                          \
    const float* __restrict__ conv_init, /* [B*di*d_conv] carry-in */         \
    const float* __restrict__ weight,                                         \
    const float* __restrict__ bias,                                           \
    int batch, int T_, int d_inner, int d_conv,                               \
    /* Row stride and column offset of the d_x destination: the x half        \
       of d_proj [bt, 2*d_inner] in production. */                            \
    int out_stride, int out_offset,                                           \
    int x_stride /* row stride of x_branch, as in the forward */              \
) {                                                                           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    int total = batch * d_inner;                                              \
    if (idx >= total) return;                                                 \
    if (d_conv > 8) return;                                                   \
    int b = idx / d_inner;                                                    \
    int d = idx % d_inner;                                                    \
    int n_tiles = (T_ + CONV1D_TILE_T - 1) / CONV1D_TILE_T;                   \
    int tile = blockIdx.y;                                                    \
    int t0 = tile * CONV1D_TILE_T;                                            \
    if (t0 >= T_) return;                                                     \
    int t_end = min(t0 + CONV1D_TILE_T, T_);                                  \
    int row = b * n_tiles + tile;                                             \
    int init_base = (b * d_inner + d) * d_conv;                               \
    int carry_len = d_conv - 1;                                               \
    float local_d_bias = 0.0f;                                                \
    float bias_d = bias[d];                                                   \
    /* The window at step ts holds x[ts - d_conv + 1 + k]; a negative         \
       index reads the carry-in state past its first slot. */                 \
    int ts_top = t_end + carry_len - 1;                                       \
    int ts0 = min(ts_top, T_ - 1);                                            \
    if (d_conv == 4) {                                                        \
        float win4[4];                                                        \
        float carry4[3] = {0.0f, 0.0f, 0.0f};                                 \
        float dw4[4] = {0.0f, 0.0f, 0.0f, 0.0f};                              \
        float w4[4];                                                          \
        _Pragma("unroll")                                                     \
        for (int k = 0; k < 4; k++) {                                         \
            w4[k] = weight[d * d_conv + k];                                   \
            int tx = ts0 - 3 + k;                                             \
            win4[k] = (tx >= 0)                                               \
                ? to_f(x_branch[(b * T_ + tx) * x_stride + d])                \
                : conv_init[init_base + tx + d_conv];                         \
        }                                                                     \
        CONV1D_BWD_WALK(4, TY, FROM_F, win4, carry4, dw4, w4)                 \
        _Pragma("unroll")                                                     \
        for (int k = 0; k < 4; k++) {                                         \
            d_weight_partials[row * (d_inner * d_conv) + d * d_conv + k] =    \
                dw4[k];                                                       \
        }                                                                     \
    } else {                                                                  \
        float carry[8];                                                       \
        float dw[8];                                                          \
        float win[8];                                                         \
        float w[8];                                                           \
        for (int k = 0; k < carry_len; k++) carry[k] = 0.0f;                  \
        for (int k = 0; k < d_conv; k++) {                                    \
            dw[k] = 0.0f;                                                     \
            w[k] = weight[d * d_conv + k];                                    \
            int tx = ts0 - carry_len + k;                                     \
            win[k] = (tx >= 0)                                                \
                ? to_f(x_branch[(b * T_ + tx) * x_stride + d])                \
                : conv_init[init_base + tx + d_conv];                         \
        }                                                                     \
        CONV1D_BWD_WALK(d_conv, TY, FROM_F, win, carry, dw, w)                \
        for (int k = 0; k < d_conv; k++) {                                    \
            d_weight_partials[row * (d_inner * d_conv) + d * d_conv + k] =    \
                dw[k];                                                        \
        }                                                                     \
    }                                                                         \
    d_bias_partials[row * d_inner + d] = local_d_bias;                        \
}                                                                             \

DEFINE_CONV1D_BWD_TILED(f32,  float,         from_f_f32)
DEFINE_CONV1D_BWD_TILED(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_CONV1D_BWD_TILED(f16,  __half,        from_f_f16)
