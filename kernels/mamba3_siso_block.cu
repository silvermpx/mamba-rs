// Block-tier twins of the Mamba-3 SISO head kernels.
//
// The warp-tier kernels in mamba3_siso.cu run one warp per (batch, head)
// and broadcast the head-shared values (the step coefficients, D and the
// current and previous K and Q rows) from lane 0 by warp shuffle, so a
// head must fit one warp and its width must be a power of two. These twins
// serve every other width: one CTA per (batch, head), one thread per head
// element, the head-shared values staged once in shared memory.
//
// Each thread runs the warp kernel's arithmetic on the same values in the
// same order, so at a width both tiers serve they give the same bits. The
// recurrence is spelled with explicit roundings because the compiler fuses
// the warp kernel's `alpha*h + beta*v*kp + gamma*x*kc` as
// fma(gx, kc, fma(h, alpha, bv*kp)) but would fuse the same source here
// the other way round, one ulp apart. The router still sends those widths
// to the warp tier; the block tier carries the official checkpoints'
// headdim 64 and every other width. A head wider than a block takes its
// elements in turns, each element's arithmetic unchanged.
//
// Composed after mamba3_chunked.cu in the Mamba-3 module.

#include "_typed_prelude.cuh"

#ifndef MAMBA_RS_STATE_CAP
#define MAMBA_RS_STATE_CAP 64
#endif

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

// m3_step_fwd_block: one decode step of a head wider than a warp.
//
// Arguments as m3_step_fwd. Grid: (B, nh). Block: headdim rounded up to
// whole warps, at most the threads the kernel's registers allow in a block;
// a wider head takes its elements in turns. Dynamic shared memory:
// (4 + 3 * ds) floats, the four step coefficients then the current K,
// previous K and current Q rows.
//
// The previous K is staged by every thread before the barrier, so the
// head's new K may overwrite k_state right after it: no thread reads
// k_state again.
#define DEFINE_M3_STEP_FWD_BLOCK(NAME, TY, FROM_F)                            \
extern "C" __global__ void NAME(                                              \
    float* ssm_state,                                                         \
    float* k_state,                                                           \
    float* v_state,                                                           \
    TY* y,                                                                    \
    const TY* x,                                                              \
    const TY* k_cur,                                                          \
    const TY* q_cur,                                                          \
    const float* dt, const float* a_val, const float* trap,                   \
    const float* D,                                                           \
    int batch, int nh, int hd, int ds                                         \
) {                                                                           \
    extern __shared__ float m3sb_sm[];                                        \
    float* coef = m3sb_sm;                                                    \
    float* kc_sm = coef + 4;                                                  \
    float* kp_sm = kc_sm + ds;                                                \
    float* qc_sm = kp_sm + ds;                                                \
    int b = blockIdx.x;                                                       \
    int h = blockIdx.y;                                                       \
    int tid = threadIdx.x;                                                    \
    /* Uniform across the block, so every thread reaches the barrier. */      \
    if (b >= batch || h >= nh || ds > MAMBA_RS_STATE_CAP) return;             \
    int d_inner = nh * hd;                                                    \
    int row = b * nh * ds + h * ds;                                           \
    if (tid == 0) {                                                           \
        float dt_v = dt[b * nh + h];                                          \
        float a_v = a_val[b * nh + h];                                        \
        float t_v = trap[b * nh + h];                                         \
        float alpha_h = exp2f((a_v * dt_v) * LOG2E);                          \
        coef[0] = alpha_h;                                                    \
        coef[1] = alpha_h * dt_v * (1.0f - t_v);                              \
        coef[2] = t_v * dt_v;                                                 \
        coef[3] = D[h];                                                       \
    }                                                                         \
    for (int n = tid; n < ds; n += blockDim.x) {                              \
        kc_sm[n] = to_f(k_cur[row + n]);                                      \
        kp_sm[n] = k_state[row + n];                                          \
        qc_sm[n] = to_f(q_cur[row + n]);                                      \
    }                                                                         \
    __syncthreads();                                                          \
    for (int n = tid; n < ds; n += blockDim.x) k_state[row + n] = kc_sm[n];   \
    float alpha_h = coef[0];                                                  \
    float beta_h = coef[1];                                                   \
    float gamma_h = coef[2];                                                  \
    float d_skip = coef[3];                                                   \
    for (int p = tid; p < hd; p += blockDim.x) {                              \
        float h_local[MAMBA_RS_STATE_CAP];                                    \
        int h_base = (b * nh * hd + h * hd + p) * ds;                         \
        for (int n = 0; n < ds; n++) h_local[n] = ssm_state[h_base + n];      \
        float x_val = to_f(x[b * d_inner + h * hd + p]);                      \
        float v_prev = v_state[b * nh * hd + h * hd + p];                     \
        float y_val = d_skip * x_val;                                         \
        float bv = beta_h * v_prev;                                           \
        float gx = gamma_h * x_val;                                           \
        for (int n = 0; n < ds; n++) {                                        \
            float kc_n = kc_sm[n];                                            \
            float kp_n = kp_sm[n];                                            \
            float qc_n = qc_sm[n];                                            \
            h_local[n] = __fmaf_rn(gx, kc_n,                                  \
                                   __fmaf_rn(h_local[n], alpha_h,             \
                                             __fmul_rn(bv, kp_n)));           \
            y_val = __fmaf_rn(h_local[n], qc_n, y_val);                       \
        }                                                                     \
        for (int n = 0; n < ds; n++) ssm_state[h_base + n] = h_local[n];      \
        y[b * d_inner + h * hd + p] = FROM_F(y_val);                          \
        v_state[b * nh * hd + h * hd + p] = x_val;                            \
    }                                                                         \
}

DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block, float, from_f_f32)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_f16, __half, from_f_f16)

// ============================================================================
// The sequential scan's training kernels for any head width.
//
// The warp kernels in mamba3_siso.cu broadcast each step's coefficients and
// its K, previous K and Q rows by warp shuffle and reduce the per-step
// gradients over the head with a shuffle tree of width headdim. Here the
// step's values are staged in shared memory, and a gradient summed over the
// head runs the same tree inside every warp, lanes past the head holding
// -0.0, then adds the warps' sums in order. A head of one warp therefore
// gets the warp kernel's bits, and a wider head one fixed order.
// ============================================================================

__device__ __forceinline__ void m3b_store(float* p, float v) { *p = v; }
__device__ __forceinline__ void m3b_store(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16_rn(v);
}
__device__ __forceinline__ void m3b_store(__half* p, float v) { *p = __float2half_rn(v); }

// m3_burnin_fwd_block: the sequential training forward. Arguments as
// m3_burnin_fwd. SAVE false drops the three saves, for the target
// network's forward. Grid: (B, nh). Block: headdim rounded up to whole
// warps, at most 512 threads.
// Dynamic shared memory: (4 + 3 * ds) floats.
//
// A head wider than the block runs in slices of the block's width, each
// over the whole window. A step's previous K is the state's K at the first
// step and the window's K of the step before at every later one, so every
// slice reads the same rows, and the state takes the window's last K once
// all slices are done.
//
// The recurrence is spelled with the roundings the warp kernel compiles to:
// the previous-state term rounded on its own, then two fused steps, and the
// output folded in order. The f32 warp kernels with a constant state width
// fold the skip term into the first output step, fma(D, x, h0 * q0), and
// every other warp kernel rounds D * x on its own first. FOLD_SKIP picks
// the first, so each block-tier entry gives the bits of the warp kernel it
// twins.
template <typename T_ACT, bool SAVE, int DS, bool FOLD_SKIP>
__device__ __forceinline__ void m3_burnin_block_tpl(
    float* ssm_state, float* k_state, float* v_state,
    T_ACT* y_out,
    float* h_saved, float* k_prev_saved, float* v_prev_saved,
    const T_ACT* x_flat, const T_ACT* k_flat, const T_ACT* q_flat,
    const float* alpha_flat, const float* beta_flat, const float* gamma_flat,
    const float* D,
    int batch, int T, int nh, int hd, int ds
) {
    extern __shared__ float m3bb_sm[];
    float* coef = m3bb_sm;
    float* kc_sm = coef + 4;
    float* kp_sm = kc_sm + ds;
    float* qc_sm = kp_sm + ds;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    /* Uniform across the block, so every thread reaches the barriers. */
    if (b >= batch || h >= nh || ds > MAMBA_RS_STATE_CAP) return;
    const int sw = (DS > 0) ? DS : ds;
    const long long d_inner = (long long)nh * hd;
    const long long nhd_ds = d_inner * ds;
    const long long head_ds = (long long)h * ds;
    const long long row_stride = (long long)nh * ds;
    const long long k_row = (long long)b * row_stride + head_ds;
    const float d_skip = D[h];
    for (int p0 = 0; p0 < hd; p0 += blockDim.x) {
        const int p = p0 + tid;
        const bool live = p < hd;
        const long long v_idx = (long long)b * nh * hd + (long long)h * hd + p;
        const long long h_base = v_idx * ds;
        const long long h_tape = (long long)b * (T + 1) * nhd_ds + (long long)h * hd + p;
        float h_local[(DS > 0) ? DS : MAMBA_RS_STATE_CAP];
        if (live) {
            _Pragma("unroll")
            for (int n = 0; n < sw; n++) h_local[n] = ssm_state[h_base + n];
            if (SAVE) {
                _Pragma("unroll")
                for (int n = 0; n < sw; n++) h_saved[h_tape + n * d_inner] = h_local[n];
            }
        }
        for (int t = 0; t < T; t++) {
            const long long bt = (long long)b * T + t;
            const long long kq_row = bt * row_stride + head_ds;
            if (tid == 0) {
                coef[0] = alpha_flat[bt * nh + h];
                coef[1] = beta_flat[bt * nh + h];
                coef[2] = gamma_flat[bt * nh + h];
            }
            for (int n = tid; n < ds; n += blockDim.x) {
                const float kp = (t == 0) ? k_state[k_row + n]
                                          : to_f(k_flat[kq_row - row_stride + n]);
                kp_sm[n] = kp;
                if (SAVE && p0 == 0) k_prev_saved[kq_row + n] = kp;
                kc_sm[n] = to_f(k_flat[kq_row + n]);
                qc_sm[n] = to_f(q_flat[kq_row + n]);
            }
            __syncthreads();
            if (live) {
                const float alpha_h = coef[0];
                const float beta_h = coef[1];
                const float gamma_h = coef[2];
                const long long x_idx = bt * d_inner + (long long)h * hd + p;
                const float x_val = to_f(x_flat[x_idx]);
                const float v_prev = v_state[v_idx];
                if (SAVE) v_prev_saved[x_idx] = v_prev;
                float y_val = __fmul_rn(d_skip, x_val);
                const float bv = __fmul_rn(beta_h, v_prev);
                const float gx = __fmul_rn(gamma_h, x_val);
                _Pragma("unroll")
                for (int n = 0; n < sw; n++) {
                    h_local[n] = __fmaf_rn(gx, kc_sm[n],
                        __fmaf_rn(alpha_h, h_local[n], __fmul_rn(bv, kp_sm[n])));
                    if (FOLD_SKIP && n == 0)
                        y_val = __fmaf_rn(d_skip, x_val, __fmul_rn(qc_sm[0], h_local[0]));
                    else
                        y_val = __fmaf_rn(qc_sm[n], h_local[n], y_val);
                }
                m3b_store(&y_out[x_idx], y_val);
                if (SAVE) {
                    const long long step = h_tape + (long long)(t + 1) * nhd_ds;
                    _Pragma("unroll")
                    for (int n = 0; n < sw; n++) h_saved[step + n * d_inner] = h_local[n];
                }
                v_state[v_idx] = x_val;
            }
            /* The next step restages the rows. */
            __syncthreads();
        }
        if (live) {
            _Pragma("unroll")
            for (int n = 0; n < sw; n++) ssm_state[h_base + n] = h_local[n];
        }
    }
    /* Every slice has read the state's K by now. */
    if (T > 0) {
        const long long last = ((long long)b * T + T - 1) * row_stride + head_ds;
        for (int n = tid; n < ds; n += blockDim.x) k_state[k_row + n] = to_f(k_flat[last + n]);
    }
}

#define M3_BURNIN_BLOCK_ENTRY(NAME, T_ACT, SAVE, FOLD)                         \
extern "C" __global__ __launch_bounds__(512) void NAME(                        \
    float* ssm_state, float* k_state, float* v_state, T_ACT* y_out,           \
    float* h_saved, float* k_prev_saved, float* v_prev_saved,                 \
    const T_ACT* x_flat, const T_ACT* k_flat, const T_ACT* q_flat,            \
    const float* alpha_flat, const float* beta_flat, const float* gamma_flat, \
    const float* D, int batch, int T, int nh, int hd, int ds                  \
) {                                                                           \
    switch (ds) {                                                             \
    case 8: m3_burnin_block_tpl<T_ACT, SAVE, 8, FOLD>(ssm_state, k_state, v_state, y_out, h_saved, k_prev_saved, v_prev_saved, x_flat, k_flat, q_flat, alpha_flat, beta_flat, gamma_flat, D, batch, T, nh, hd, ds); return; \
    case 16: m3_burnin_block_tpl<T_ACT, SAVE, 16, FOLD>(ssm_state, k_state, v_state, y_out, h_saved, k_prev_saved, v_prev_saved, x_flat, k_flat, q_flat, alpha_flat, beta_flat, gamma_flat, D, batch, T, nh, hd, ds); return; \
    case 32: m3_burnin_block_tpl<T_ACT, SAVE, 32, FOLD>(ssm_state, k_state, v_state, y_out, h_saved, k_prev_saved, v_prev_saved, x_flat, k_flat, q_flat, alpha_flat, beta_flat, gamma_flat, D, batch, T, nh, hd, ds); return; \
    case 64: m3_burnin_block_tpl<T_ACT, SAVE, 64, FOLD>(ssm_state, k_state, v_state, y_out, h_saved, k_prev_saved, v_prev_saved, x_flat, k_flat, q_flat, alpha_flat, beta_flat, gamma_flat, D, batch, T, nh, hd, ds); return; \
    default: m3_burnin_block_tpl<T_ACT, SAVE, 0, false>(ssm_state, k_state, v_state, y_out, h_saved, k_prev_saved, v_prev_saved, x_flat, k_flat, q_flat, alpha_flat, beta_flat, gamma_flat, D, batch, T, nh, hd, ds); return; \
    }                                                                         \
}

M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block, float, true, true)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_bf16, __nv_bfloat16, true, false)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_f16, __half, true, false)
#undef M3_BURNIN_BLOCK_ENTRY

// The target network's forward: arguments as m3_burnin_fwd_nosave.
extern "C" __global__ __launch_bounds__(512) void m3_burnin_fwd_nosave_block(
    float* ssm_state, float* k_state, float* v_state, float* y_out,
    const float* x_flat, const float* k_flat, const float* q_flat,
    const float* alpha_flat, const float* beta_flat, const float* gamma_flat,
    const float* D, int batch, int T, int nh, int hd, int ds
) {
    m3_burnin_block_tpl<float, false, 0, false>(ssm_state, k_state, v_state, y_out,
        nullptr, nullptr, nullptr, x_flat, k_flat, q_flat, alpha_flat,
        beta_flat, gamma_flat, D, batch, T, nh, hd, ds);
}

// One warp's shuffle tree over the head, as the warp kernel runs it: width
// is the head for a head under a warp and 32 otherwise.
__device__ __forceinline__ float m3b_warp_tree(float v, int width) {
    _Pragma("unroll")
    for (int step = 0; step < 5; step++) {
        const int off = 16 >> step;
        const float carried = __shfl_down_sync(0xFFFFFFFFu, v, off, 32);
        if (off < width) v += carried;
    }
    return v;
}

// m3_backward_seq_block: the sequential backward through time. Arguments as
// m3_backward_seq, with dO, x, K and Q in the activation type and every
// gradient in f32. Grid: (B, nh). Block: headdim rounded up to whole warps,
// at most 512 threads. Dynamic shared memory, in floats: 4 + 3 * ds for the
// step's values, ds for the carried previous-K gradient, and
// (3 * ds + 3) * warps for the warps' sums.
//
// Each step: the rows are staged, every thread runs its element's terms in
// the warp kernel's source form (each fused add has one product, so it
// compiles to the warp kernel's roundings), every warp reduces its sums,
// and one thread per state element (and one per scalar) adds the warps'
// sums in order and writes the step's dQ, dK, dalpha, dbeta and dgamma.
//
// A head wider than the block runs in slices of the block's width, each
// over the whole window; a slice after the first adds its sums to the
// gradients the slices before it wrote.
template <typename T_ACT, int DS>
__device__ __forceinline__ void m3_backward_seq_block_tpl(
    const float* __restrict__ h_saved, const float* __restrict__ k_prev_saved,
    const float* __restrict__ v_prev_saved, const T_ACT* __restrict__ x_flat,
    const T_ACT* __restrict__ k_flat, const T_ACT* __restrict__ q_flat,
    const float* __restrict__ alpha_flat, const float* __restrict__ beta_flat,
    const float* __restrict__ gamma_flat, const float* __restrict__ D,
    const T_ACT* __restrict__ d_y_flat,
    float* __restrict__ d_x, float* __restrict__ d_k, float* __restrict__ d_q,
    float* __restrict__ d_alpha, float* __restrict__ d_beta,
    float* __restrict__ d_gamma, float* __restrict__ d_D_local,
    int batch, int T, int nh, int hd, int ds
) {
    extern __shared__ float m3bs_sm[];
    float* coef = m3bs_sm;
    float* kc_sm = coef + 4;
    float* kp_sm = kc_sm + ds;
    float* qc_sm = kp_sm + ds;
    float* carry_sm = qc_sm + ds;          /* [ds] previous-K gradient */
    float* red = carry_sm + ds;            /* [(3 * ds + 3) * warps] */
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    /* Uniform across the block, so every thread reaches the barriers. */
    if (b >= batch || h >= nh || ds > MAMBA_RS_STATE_CAP) return;
    const int sw = (DS > 0) ? DS : ds;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warps = blockDim.x >> 5;
    const int width = hd < 32 ? hd : 32;
    const long long d_inner = (long long)nh * hd;
    const long long nhd_ds = d_inner * ds;
    const long long head_ds = (long long)h * ds;
    const float d_skip = D[h];
    for (int p0 = 0; p0 < hd; p0 += blockDim.x) {
        const int p = p0 + tid;
        const bool live = p < hd;
        const bool first = p0 == 0;
        const long long h_tape = (long long)b * (T + 1) * nhd_ds + (long long)h * hd + p;
        for (int n = tid; n < ds; n += blockDim.x) carry_sm[n] = 0.0f;

        float d_h_reg[(DS > 0) ? DS : MAMBA_RS_STATE_CAP];
        _Pragma("unroll")
        for (int n = 0; n < sw; n++) d_h_reg[n] = 0.0f;
        float d_v_carry = 0.0f;
        float sum_d_D = 0.0f;

        for (int t = T - 1; t >= 0; t--) {
            const long long bt = (long long)b * T + t;
            const long long kq_row = bt * nh * ds + head_ds;
            if (tid == 0) {
                coef[0] = alpha_flat[bt * nh + h];
                coef[1] = beta_flat[bt * nh + h];
                coef[2] = gamma_flat[bt * nh + h];
            }
            for (int n = tid; n < ds; n += blockDim.x) {
                kc_sm[n] = to_f(k_flat[kq_row + n]);
                kp_sm[n] = k_prev_saved[kq_row + n];
                qc_sm[n] = to_f(q_flat[kq_row + n]);
            }
            __syncthreads();
            const float alpha_h = coef[0];
            const float beta_h = coef[1];
            const float gamma_h = coef[2];
            const long long x_idx = bt * d_inner + (long long)h * hd + p;
            float dy_val = 0.0f, x_val = 0.0f, v_prev = 0.0f;
            if (live) {
                dy_val = to_f(d_y_flat[x_idx]);
                x_val = to_f(x_flat[x_idx]);
                v_prev = v_prev_saved[x_idx];
            }
            sum_d_D += dy_val * x_val;
            float d_x_val = d_skip * dy_val;
            d_x_val += d_v_carry;
            float d_alpha_acc = 0.0f;
            float d_beta_acc = 0.0f;
            float d_gamma_acc = 0.0f;
            float d_v_prev_acc = 0.0f;
            const long long prev = h_tape + (long long)t * nhd_ds;
            const long long curr = prev + nhd_ds;
            _Pragma("unroll")
            for (int n = 0; n < sw; n++) {
                const float kc_n = kc_sm[n];
                const float kp_n = kp_sm[n];
                const float qc_n = qc_sm[n];
                float d_q_val = -0.0f, d_kp_val = -0.0f, d_kc_val = -0.0f;
                if (live) {
                    const float h_prev_n = h_saved[prev + n * d_inner];
                    const float h_curr_n = h_saved[curr + n * d_inner];
                    d_h_reg[n] += dy_val * qc_n;
                    d_q_val = dy_val * h_curr_n;
                    const float dh_n = d_h_reg[n];
                    d_alpha_acc += dh_n * h_prev_n;
                    d_beta_acc += dh_n * v_prev * kp_n;
                    d_gamma_acc += dh_n * x_val * kc_n;
                    d_v_prev_acc += dh_n * beta_h * kp_n;
                    d_kp_val = dh_n * beta_h * v_prev;
                    d_x_val += dh_n * gamma_h * kc_n;
                    d_kc_val = dh_n * gamma_h * x_val;
                    d_h_reg[n] = alpha_h * dh_n;
                }
                d_q_val = m3b_warp_tree(d_q_val, width);
                d_kp_val = m3b_warp_tree(d_kp_val, width);
                d_kc_val = m3b_warp_tree(d_kc_val, width);
                if (lane == 0) {
                    red[(3 * n) * warps + warp] = d_q_val;
                    red[(3 * n + 1) * warps + warp] = d_kp_val;
                    red[(3 * n + 2) * warps + warp] = d_kc_val;
                }
            }
            if (!live) {
                d_alpha_acc = -0.0f;
                d_beta_acc = -0.0f;
                d_gamma_acc = -0.0f;
            }
            d_alpha_acc = m3b_warp_tree(d_alpha_acc, width);
            d_beta_acc = m3b_warp_tree(d_beta_acc, width);
            d_gamma_acc = m3b_warp_tree(d_gamma_acc, width);
            if (lane == 0) {
                red[(3 * ds) * warps + warp] = d_alpha_acc;
                red[(3 * ds + 1) * warps + warp] = d_beta_acc;
                red[(3 * ds + 2) * warps + warp] = d_gamma_acc;
            }
            if (live) {
                d_x[x_idx] = d_x_val;
                d_v_carry = d_v_prev_acc;
            }
            __syncthreads();
            /* One thread per state element and per scalar adds the warps'
               sums in order. The rows are restaged only after the next
               barrier. */
            for (int i = tid; i < ds + 3; i += blockDim.x) {
                if (i < ds) {
                    const int n = i;
                    float q = red[(3 * n) * warps];
                    float kp = red[(3 * n + 1) * warps];
                    float kc = red[(3 * n + 2) * warps];
                    for (int w = 1; w < warps; w++) {
                        q += red[(3 * n) * warps + w];
                        kp += red[(3 * n + 1) * warps + w];
                        kc += red[(3 * n + 2) * warps + w];
                    }
                    float d_k_write = 0.0f;
                    if (t < T - 1) {
                        d_k_write += carry_sm[n];
                        carry_sm[n] = 0.0f;
                    }
                    carry_sm[n] += kp;
                    d_k_write += kc;
                    if (first) {
                        d_q[kq_row + n] = q;
                        d_k[kq_row + n] = d_k_write;
                    } else {
                        d_q[kq_row + n] += q;
                        d_k[kq_row + n] += d_k_write;
                    }
                } else {
                    const int s = i - ds;
                    float total = red[(3 * ds + s) * warps];
                    for (int w = 1; w < warps; w++) total += red[(3 * ds + s) * warps + w];
                    float* out = (s == 0) ? d_alpha : (s == 1) ? d_beta : d_gamma;
                    if (first)
                        out[bt * nh + h] = total;
                    else
                        out[bt * nh + h] += total;
                }
            }
        }
        if (live) d_D_local[(long long)b * d_inner + (long long)h * hd + p] = sum_d_D;
    }
}

#define M3_BACKWARD_SEQ_BLOCK_ENTRY(NAME, T_ACT)                               \
extern "C" __global__ __launch_bounds__(512) void NAME(                        \
    const float* __restrict__ h_saved, const float* __restrict__ k_prev_saved,\
    const float* __restrict__ v_prev_saved, const T_ACT* __restrict__ x_flat, \
    const T_ACT* __restrict__ k_flat, const T_ACT* __restrict__ q_flat,       \
    const float* __restrict__ alpha_flat, const float* __restrict__ beta_flat,\
    const float* __restrict__ gamma_flat, const float* __restrict__ D,        \
    const T_ACT* __restrict__ d_y_flat,                                       \
    float* __restrict__ d_x, float* __restrict__ d_k, float* __restrict__ d_q,\
    float* __restrict__ d_alpha, float* __restrict__ d_beta,                  \
    float* __restrict__ d_gamma, float* __restrict__ d_D_local,               \
    int batch, int T, int nh, int hd, int ds                                  \
) {                                                                           \
    switch (ds) {                                                             \
    case 8: m3_backward_seq_block_tpl<T_ACT, 8>(h_saved, k_prev_saved, v_prev_saved, x_flat, k_flat, q_flat, alpha_flat, beta_flat, gamma_flat, D, d_y_flat, d_x, d_k, d_q, d_alpha, d_beta, d_gamma, d_D_local, batch, T, nh, hd, ds); return; \
    case 16: m3_backward_seq_block_tpl<T_ACT, 16>(h_saved, k_prev_saved, v_prev_saved, x_flat, k_flat, q_flat, alpha_flat, beta_flat, gamma_flat, D, d_y_flat, d_x, d_k, d_q, d_alpha, d_beta, d_gamma, d_D_local, batch, T, nh, hd, ds); return; \
    case 32: m3_backward_seq_block_tpl<T_ACT, 32>(h_saved, k_prev_saved, v_prev_saved, x_flat, k_flat, q_flat, alpha_flat, beta_flat, gamma_flat, D, d_y_flat, d_x, d_k, d_q, d_alpha, d_beta, d_gamma, d_D_local, batch, T, nh, hd, ds); return; \
    case 64: m3_backward_seq_block_tpl<T_ACT, 64>(h_saved, k_prev_saved, v_prev_saved, x_flat, k_flat, q_flat, alpha_flat, beta_flat, gamma_flat, D, d_y_flat, d_x, d_k, d_q, d_alpha, d_beta, d_gamma, d_D_local, batch, T, nh, hd, ds); return; \
    default: m3_backward_seq_block_tpl<T_ACT, 0>(h_saved, k_prev_saved, v_prev_saved, x_flat, k_flat, q_flat, alpha_flat, beta_flat, gamma_flat, D, d_y_flat, d_x, d_k, d_q, d_alpha, d_beta, d_gamma, d_D_local, batch, T, nh, hd, ds); return; \
    }                                                                         \
}

M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block, float)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_bf16, __nv_bfloat16)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_f16, __half)
#undef M3_BACKWARD_SEQ_BLOCK_ENTRY

// ============================================================================
// The chunked backward for every head and state width.
//
// m3_dqkv keeps a head's whole state in shared memory and in registers, so
// it serves heads of at most one warp and states whose tiles fit beside
// them. The kernels below compute the same outputs for any headdim and any
// d_state in a fixed shared-memory budget: every operand streams through
// tiles of TW columns, and every sum runs in the order m3_dqkv runs it, its
// partial carried exactly as a float from one tile to the next. A tile
// width therefore never changes a bit, and at every width m3_dqkv serves
// the outputs are its bits. Chains run over valid indices only, so padding
// never adds a zero that could turn a -0.0 into +0.0.
// ============================================================================

// Index of the pair (a, b), a < b, in a chunk's strict upper triangle.
__device__ __forceinline__ int m3t_tri(int a, int b, int cs) {
    return a * (2 * cs - a - 1) / 2 + (b - a - 1);
}

// The scalar the chunk's saved state adds to dADT: the flat chain over
// (p, n), p-major, of the saved state times its gradient. One thread per
// (b, chunk, h), since the chain's order is the numeric contract. A kernel
// of its own, so the chain delays no other work.
extern "C" __global__ void m3_dqkv_dm_chain(
    float* __restrict__ dm_out,             // [B*n_chunks*nh]
    const float* __restrict__ SSM_States,   // [B*n_chunks*nh*hd*ds]
    const float* __restrict__ enter_state,  // [B*n_chunks*nh*hd*ds]
    int rows, int hd, int ds
) {
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const long long base = row * hd * ds;
    const int n_el = hd * ds;
    float acc = 0.0f;
    for (int i = 0; i < n_el; i++)
        acc += SSM_States[base + i] * enter_state[base + i];
    dm_out[row] = acc;
}

// The per-chunk terms of the reverse state pass: state_B[p][n] =
// sum over t ascending of dO[t][p] * exp2(dA[t]) * Q[t][n], the chain of
// m3_dqkv_state_terms, for any width. One block of 256 threads per
// (b, h, chunk) walks the [hd][ds] output in 16 x 16 tiles, one output per
// thread. The weighted dO is rounded once into shared memory, the product
// the original forms before its fused add.
#define DEFINE_M3_DQKV_STATE_TERMS_TILED(NAME, T_ACT)                          \
extern "C" __global__ __launch_bounds__(256) void NAME(                        \
    float* __restrict__ state_B,                                               \
    const T_ACT* __restrict__ Q_rot,                                           \
    const float* __restrict__ DA_CS,                                           \
    const T_ACT* __restrict__ dO,                                              \
    int B, int T, int nh_total, int hd, int ds, int CS                         \
) {                                                                            \
    const int h = blockIdx.x;                                                  \
    const int b = blockIdx.y;                                                  \
    const int chunk_idx = blockIdx.z;                                          \
    const int tid = threadIdx.x;                                               \
    if (h >= nh_total || b >= B || CS > 64) return;                            \
    const int n_chunks = (T + CS - 1) / CS;                                    \
    const int chunk_start = chunk_idx * CS;                                    \
    const int len = min(CS, T - chunk_start);                                  \
    const long long d_inner = (long long)nh_total * hd;                        \
    const long long cidx = ((long long)b * n_chunks + chunk_idx) * nh_total + h; \
    __shared__ float w_sm[64 * 17];                                            \
    __shared__ float q_sm[64 * 17];                                            \
    __shared__ float e_sm[64];                                                 \
    for (int t = tid; t < len; t += 256)                                       \
        e_sm[t] = exp2f(DA_CS[cidx * CS + t] * LOG2E);                         \
    const int r = tid / 16;                                                    \
    const int c = tid % 16;                                                    \
    for (int p0 = 0; p0 < hd; p0 += 16) {                                      \
        const int pw = min(16, hd - p0);                                       \
        for (int n0 = 0; n0 < ds; n0 += 16) {                                  \
            const int nw = min(16, ds - n0);                                   \
            __syncthreads();                                                   \
            for (int o = tid; o < len * 16; o += 256) {                        \
                const int t = o / 16;                                          \
                const int j = o % 16;                                          \
                const long long row = (long long)b * T + chunk_start + t;      \
                if (j < pw)                                                    \
                    w_sm[t * 17 + j] =                                         \
                        to_f(dO[row * d_inner + (long long)h * hd + p0 + j])   \
                        * e_sm[t];                                             \
                if (j < nw)                                                    \
                    q_sm[t * 17 + j] =                                         \
                        to_f(Q_rot[(row * nh_total + h) * ds + n0 + j]);       \
            }                                                                  \
            __syncthreads();                                                   \
            if (r < pw && c < nw) {                                            \
                float acc = 0.0f;                                              \
                for (int t = 0; t < len; t++)                                  \
                    acc += w_sm[t * 17 + r] * q_sm[t * 17 + c];                \
                state_B[cidx * hd * ds + (long long)(p0 + r) * ds + n0 + c] =  \
                    acc;                                                       \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

DEFINE_M3_DQKV_STATE_TERMS_TILED(m3_dqkv_state_terms_tiled, float)
DEFINE_M3_DQKV_STATE_TERMS_TILED(m3_dqkv_state_terms_tiled_bf16, __nv_bfloat16)
DEFINE_M3_DQKV_STATE_TERMS_TILED(m3_dqkv_state_terms_tiled_f16, __half)
#undef DEFINE_M3_DQKV_STATE_TERMS_TILED

// m3_dqkv_tiled: m3_dqkv for any headdim and d_state.
//
// One block of 256 threads per (head, sequence, chunk), tiles TW columns
// wide (up to 32, the caller's choice: the bits do not depend on it).
// Shared memory, in floats: nine per-step lanes of 64, the three pair
// triangles (K.Q, V.dO and the decay) of CS (CS - 1) / 2 each, four
// [64][TW + 1] operand tiles and two [TW][TW + 1] state tiles.
//
// The work, in order, each sum in m3_dqkv's order:
// 1. The pair dots over d_state and over headdim, one slot per thread,
//    their partials held in registers across the tiles.
// 2. dQK_dot from global memory, each 32-channel segment through the warp
//    kernel's shuffle tree, the segments added in order.
// 3. Per head tile: K.dS^T and Q.S^T over d_state, then dV and the two
//    per-step chains of the dADT state terms over the tile's channels.
// 4. Per state tile: the intra-chunk dK and dQ, then their state terms
//    over headdim.
// 5. dADT from the triangles and the per-step lanes, its serial combine,
//    and the chunk's dD partial.
#define M3T_MAX_SLOTS 8
#define M3T_MAX_OUT 8
#define DEFINE_M3_DQKV_TILED(NAME, T_ACT)                                      \
extern "C" __global__ __launch_bounds__(256) void NAME(                        \
    float* __restrict__ dQ_mid, float* __restrict__ dK_mid,                    \
    float* __restrict__ dV, float* __restrict__ dADT,                          \
    float* __restrict__ dQK_dot_out, float* __restrict__ dD_partials,          \
    const T_ACT* __restrict__ Q_rot, const T_ACT* __restrict__ K_scaled,       \
    const T_ACT* __restrict__ V_in,                                            \
    const float* __restrict__ DA_CS, const float* __restrict__ DA_CS_SUM,      \
    const float* __restrict__ QK_dot_in,                                       \
    const float* __restrict__ SSM_States,                                      \
    const T_ACT* __restrict__ dO, const float* __restrict__ D_param,           \
    const float* __restrict__ enter_state,                                     \
    const float* __restrict__ dm_chain,                                        \
    int B, int T, int nh_total, int hd, int ds, int CS, int TW                 \
) {                                                                            \
    const int h = blockIdx.x;                                                  \
    const int b = blockIdx.y;                                                  \
    const int chunk_idx = blockIdx.z;                                          \
    const int tid = threadIdx.x;                                               \
    /* Uniform across the block, so every thread reaches the barriers. */     \
    if (h >= nh_total || b >= B || CS > 64 || TW > 32 || TW < 1) return;       \
    const int n_chunks = (T + CS - 1) / CS;                                    \
    const int chunk_start = chunk_idx * CS;                                    \
    const int len = min(CS, T - chunk_start);                                  \
    const long long d_inner = (long long)nh_total * hd;                        \
    const long long cidx = ((long long)b * n_chunks + chunk_idx) * nh_total + h; \
    const long long row0 = (long long)b * T + chunk_start;                     \
    const float* ssm = SSM_States + cidx * hd * ds;                            \
    const float* dst = enter_state + cidx * hd * ds;                           \
    const float D_val = D_param[h];                                            \
    const float cs_sum = DA_CS_SUM[cidx];                                      \
    const int RS = TW + 1;                                                     \
    const int tri_n = CS * (CS - 1) / 2;                                       \
    extern __shared__ float m3t_sm[];                                          \
    float* da_sm = m3t_sm;                                                     \
    float* qk_sm = da_sm + 64;                                                 \
    float* ef_sm = qk_sm + 64;                                                 \
    float* er_sm = ef_sm + 64;                                                 \
    float* dmrev_sm = er_sm + 64;                                              \
    float* dmvec_sm = dmrev_sm + 64;                                           \
    float* dqk_sm = dmvec_sm + 64;                                             \
    float* qsdo_sm = dqk_sm + 64;                                              \
    float* dskv_sm = qsdo_sm + 64;                                             \
    float* kq_tri = dskv_sm + 64;                                              \
    float* vdo_tri = kq_tri + tri_n;                                           \
    float* dec_tri = vdo_tri + tri_n;                                          \
    float* a0 = dec_tri + tri_n; /* [64][RS] K, then K.dS^T */                 \
    float* a1 = a0 + 64 * RS;    /* [64][RS] Q, then Q.S^T */                  \
    float* a2 = a1 + 64 * RS;    /* [64][RS] dO */                             \
    float* a3 = a2 + 64 * RS;    /* [64][RS] V */                              \
    float* s0 = a3 + 64 * RS;    /* [TW][RS] saved state S */                  \
    float* s1 = s0 + TW * RS;    /* [TW][RS] state gradient dS */              \
    for (int t = tid; t < 64; t += 256) {                                      \
        float da = 0.0f, qk = 0.0f;                                            \
        if (t < len) {                                                         \
            da = DA_CS[cidx * CS + t];                                         \
            qk = QK_dot_in[(row0 + t) * nh_total + h];                         \
        }                                                                      \
        da_sm[t] = da;                                                         \
        qk_sm[t] = qk;                                                         \
        ef_sm[t] = exp2f(da * LOG2E);                                          \
        er_sm[t] = exp2f((cs_sum - da) * LOG2E);                               \
    }                                                                          \
    /* The pair dots. Slot idx is pair (aa, bb) of the triangle. */   \
    int pa[M3T_MAX_SLOTS], pb[M3T_MAX_SLOTS];                                  \
    float kq_acc[M3T_MAX_SLOTS], vdo_acc[M3T_MAX_SLOTS];                       \
    _Pragma("unroll")                                                          \
    for (int k = 0; k < M3T_MAX_SLOTS; k++) {                                  \
        const int idx = tid + 256 * k;                                         \
        pa[k] = -1;                                                            \
        pb[k] = -1;                                                            \
        kq_acc[k] = 0.0f;                                                      \
        vdo_acc[k] = 0.0f;                                                     \
        if (idx < tri_n) {                                                     \
            int aa = 0, start = 0;                                             \
            while (start + (CS - 1 - aa) <= idx) {                             \
                start += CS - 1 - aa;                                          \
                aa++;                                                          \
            }                                                                  \
            const int bb = idx - start + aa + 1;                               \
            if (bb < len) {                                                    \
                pa[k] = aa;                                                    \
                pb[k] = bb;                                                    \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    for (int n0 = 0; n0 < ds; n0 += TW) {                                      \
        const int nw = min(TW, ds - n0);                                       \
        __syncthreads();                                                       \
        for (int o = tid; o < len * TW; o += 256) {                            \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            if (j < nw) {                                                      \
                const long long g = ((row0 + t) * nh_total + h) * ds + n0 + j; \
                a0[t * RS + j] = to_f(K_scaled[g]);                            \
                a1[t * RS + j] = to_f(Q_rot[g]);                               \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < M3T_MAX_SLOTS; k++) {                              \
            if (pa[k] >= 0) {                                                  \
                const float* kr = a0 + pa[k] * RS;                             \
                const float* qr = a1 + pb[k] * RS;                             \
                float acc = kq_acc[k];                                         \
                for (int j = 0; j < nw; j++) acc += kr[j] * qr[j];             \
                kq_acc[k] = acc;                                               \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    for (int p0 = 0; p0 < hd; p0 += TW) {                                      \
        const int pw = min(TW, hd - p0);                                       \
        __syncthreads();                                                       \
        for (int o = tid; o < len * TW; o += 256) {                            \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            if (j < pw) {                                                      \
                const long long g = (row0 + t) * d_inner + (long long)h * hd + p0 + j; \
                a3[t * RS + j] = to_f(V_in[g]);                                \
                a2[t * RS + j] = to_f(dO[g]);                                  \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < M3T_MAX_SLOTS; k++) {                              \
            if (pa[k] >= 0) {                                                  \
                const float* vr = a3 + pa[k] * RS;                             \
                const float* dr = a2 + pb[k] * RS;                             \
                float acc = vdo_acc[k];                                        \
                for (int j = 0; j < pw; j++) acc += vr[j] * dr[j];             \
                vdo_acc[k] = acc;                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int k = 0; k < M3T_MAX_SLOTS; k++) {                                  \
        if (pa[k] >= 0) {                                                      \
            const int idx = tid + 256 * k;                                     \
            kq_tri[idx] = kq_acc[k];                                           \
            vdo_tri[idx] = vdo_acc[k];                                         \
            dec_tri[idx] = exp2f((da_sm[pb[k]] - da_sm[pa[k]]) * LOG2E);       \
        }                                                                      \
    }                                                                          \
    /* dQK_dot. A warp takes a step, and lane l holds channel l of */     \
    /* each 32-channel segment. A head under a warp starts its tree at its */  \
    /* power of two, as the warp kernel's width-hd tree does. Lanes past */    \
    /* the head hold -0.0, which leaves every sum as it is. */                 \
    {                                                                          \
        const int warp = tid >> 5;                                             \
        const int lane = tid & 31;                                             \
        int seg_pow = 32;                                                      \
        while (seg_pow / 2 >= hd) seg_pow /= 2;                                \
        for (int t = warp; t < len; t += 8) {                                  \
            float total = 0.0f;                                                \
            for (int s0 = 0; s0 < hd; s0 += 32) {                              \
                const int p = s0 + lane;                                       \
                float val = -0.0f;                                             \
                if (p < hd) {                                                  \
                    const long long g = (row0 + t) * d_inner + (long long)h * hd + p; \
                    val = to_f(dO[g]) * to_f(V_in[g]);                         \
                }                                                              \
                for (int off = seg_pow / 2; off > 0; off >>= 1)                \
                    val += __shfl_down_sync(0xFFFFFFFFu, val, off, 32);        \
                total = (s0 == 0) ? val : total + val;                         \
            }                                                                  \
            if (lane == 0) {                                                   \
                dqk_sm[t] = total;                                             \
                dQK_dot_out[(row0 + t) * nh_total + h] = total;                \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /* Per head tile. Output o of a thread is step t = o / TW, */     \
    /* channel j = o % TW of the tile. */                                      \
    const int n_out = (64 * TW + 255) / 256;                                   \
    for (int p0 = 0; p0 < hd; p0 += TW) {                                      \
        const int pw = min(TW, hd - p0);                                       \
        float ke[M3T_MAX_OUT], qs[M3T_MAX_OUT];                                \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < M3T_MAX_OUT; k++) {                                \
            ke[k] = 0.0f;                                                      \
            qs[k] = 0.0f;                                                      \
        }                                                                      \
        for (int n0 = 0; n0 < ds; n0 += TW) {                                  \
            const int nw = min(TW, ds - n0);                                   \
            __syncthreads();                                                   \
            for (int o = tid; o < len * TW; o += 256) {                        \
                const int t = o / TW;                                          \
                const int j = o % TW;                                          \
                if (j < nw) {                                                  \
                    const long long g = ((row0 + t) * nh_total + h) * ds + n0 + j; \
                    a0[t * RS + j] = to_f(K_scaled[g]);                        \
                    a1[t * RS + j] = to_f(Q_rot[g]);                           \
                }                                                              \
            }                                                                  \
            for (int o = tid; o < pw * TW; o += 256) {                         \
                const int r = o / TW;                                          \
                const int j = o % TW;                                          \
                if (j < nw) {                                                  \
                    const long long g = (long long)(p0 + r) * ds + n0 + j;     \
                    s0[r * RS + j] = ssm[g];                                   \
                    s1[r * RS + j] = dst[g];                                   \
                }                                                              \
            }                                                                  \
            __syncthreads();                                                   \
            _Pragma("unroll")                                                  \
            for (int k = 0; k < M3T_MAX_OUT; k++) {                            \
                const int o = tid + 256 * k;                                   \
                const int t = o / TW;                                          \
                const int j = o % TW;                                          \
                if (k < n_out && t < len && j < pw) {                          \
                    const float* kr = a0 + t * RS;                             \
                    const float* qr = a1 + t * RS;                             \
                    const float* sr = s0 + j * RS;                             \
                    const float* dr = s1 + j * RS;                             \
                    float kacc = ke[k], qacc = qs[k];                          \
                    for (int c = 0; c < nw; c++) {                             \
                        kacc += kr[c] * dr[c];                                 \
                        qacc += qr[c] * sr[c];                                 \
                    }                                                          \
                    ke[k] = kacc;                                              \
                    qs[k] = qacc;                                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        /* The tile's dO and V, and its K.dS^T and Q.S^T where K and Q were. */ \
        for (int o = tid; o < len * TW; o += 256) {                            \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            if (j < pw) {                                                      \
                const long long g = (row0 + t) * d_inner + (long long)h * hd + p0 + j; \
                a2[t * RS + j] = to_f(dO[g]);                                  \
                a3[t * RS + j] = to_f(V_in[g]);                                \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < M3T_MAX_OUT; k++) {                                \
            const int o = tid + 256 * k;                                       \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            if (k < n_out && t < len && j < pw) {                              \
                a0[t * RS + j] = ke[k];                                        \
                a1[t * RS + j] = qs[k];                                        \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < M3T_MAX_OUT; k++) {                                \
            const int o = tid + 256 * k;                                       \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            if (k < n_out && t < len && j < pw) {                              \
                float dv_intra = 0.0f;                                         \
                for (int s = t + 1; s < len; s++) {                            \
                    const int ti = m3t_tri(t, s, CS);                          \
                    float kq = kq_tri[ti];                                     \
                    float decay = dec_tri[ti];                                 \
                    dv_intra += kq * decay * a2[s * RS + j];                   \
                }                                                              \
                float dv_inter = ke[k];                                        \
                dv_inter *= er_sm[t];                                          \
                float dv_skip = a2[t * RS + j] * (D_val + qk_sm[t]);           \
                dV[(row0 + t) * d_inner + (long long)h * hd + p0 + j] =        \
                    dv_intra + dv_inter + dv_skip;                             \
            }                                                                  \
        }                                                                      \
        /* The dADT state terms: per step, chains over the channels in */      \
        /* order, continued from the previous tile. */                         \
        for (int t = tid; t < len; t += 256) {                                 \
            float qs_do = (p0 == 0) ? 0.0f : qsdo_sm[t];                       \
            float dsk_v = (p0 == 0) ? 0.0f : dskv_sm[t];                       \
            for (int j = 0; j < pw; j++) {                                     \
                qs_do += a1[t * RS + j] * a2[t * RS + j];                      \
                dsk_v += a0[t * RS + j] * a3[t * RS + j];                      \
            }                                                                  \
            qsdo_sm[t] = qs_do;                                                \
            dskv_sm[t] = dsk_v;                                                \
        }                                                                      \
    }                                                                          \
    /* Per state tile, dK and dQ. */                                  \
    for (int n0 = 0; n0 < ds; n0 += TW) {                                      \
        const int nw = min(TW, ds - n0);                                       \
        __syncthreads();                                                       \
        for (int o = tid; o < len * TW; o += 256) {                            \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            if (j < nw) {                                                      \
                const long long g = ((row0 + t) * nh_total + h) * ds + n0 + j; \
                a0[t * RS + j] = to_f(K_scaled[g]);                            \
                a1[t * RS + j] = to_f(Q_rot[g]);                               \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        float dki[M3T_MAX_OUT], dqi[M3T_MAX_OUT];                              \
        float dko[M3T_MAX_OUT], dqo[M3T_MAX_OUT];                              \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < M3T_MAX_OUT; k++) {                                \
            const int o = tid + 256 * k;                                       \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            dki[k] = 0.0f;                                                     \
            dqi[k] = 0.0f;                                                     \
            dko[k] = 0.0f;                                                     \
            dqo[k] = 0.0f;                                                     \
            if (k < n_out && t < len && j < nw) {                              \
                float dk_intra = 0.0f;                                         \
                for (int s = t + 1; s < len; s++) {                            \
                    const int ti = m3t_tri(t, s, CS);                          \
                    float vdo = vdo_tri[ti];                                   \
                    float decay = dec_tri[ti];                                 \
                    dk_intra += vdo * decay * a1[s * RS + j];                  \
                }                                                              \
                float dq_intra = 0.0f;                                         \
                for (int s = 0; s < t; s++) {                                  \
                    const int ti = m3t_tri(s, t, CS);                          \
                    float vdo = vdo_tri[ti];                                   \
                    float decay = dec_tri[ti];                                 \
                    dq_intra += vdo * decay * a0[s * RS + j];                  \
                }                                                              \
                dki[k] = dk_intra;                                             \
                dqi[k] = dq_intra;                                             \
            }                                                                  \
        }                                                                      \
        for (int p0 = 0; p0 < hd; p0 += TW) {                                  \
            const int pw = min(TW, hd - p0);                                   \
            __syncthreads();                                                   \
            for (int o = tid; o < len * TW; o += 256) {                        \
                const int t = o / TW;                                          \
                const int j = o % TW;                                          \
                if (j < pw) {                                                  \
                    const long long g = (row0 + t) * d_inner + (long long)h * hd + p0 + j; \
                    a2[t * RS + j] = to_f(dO[g]);                              \
                    a3[t * RS + j] = to_f(V_in[g]);                            \
                }                                                              \
            }                                                                  \
            for (int o = tid; o < pw * TW; o += 256) {                         \
                const int r = o / TW;                                          \
                const int j = o % TW;                                          \
                if (j < nw) {                                                  \
                    const long long g = (long long)(p0 + r) * ds + n0 + j;     \
                    s0[r * RS + j] = ssm[g];                                   \
                    s1[r * RS + j] = dst[g];                                   \
                }                                                              \
            }                                                                  \
            __syncthreads();                                                   \
            _Pragma("unroll")                                                  \
            for (int k = 0; k < M3T_MAX_OUT; k++) {                            \
                const int o = tid + 256 * k;                                   \
                const int t = o / TW;                                          \
                const int j = o % TW;                                          \
                if (k < n_out && t < len && j < nw) {                          \
                    float qacc = dqo[k], kacc = dko[k];                        \
                    for (int c = 0; c < pw; c++) {                             \
                        qacc += a2[t * RS + c] * s0[c * RS + j];               \
                        kacc += a3[t * RS + c] * s1[c * RS + j];               \
                    }                                                          \
                    dqo[k] = qacc;                                             \
                    dko[k] = kacc;                                             \
                }                                                              \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < M3T_MAX_OUT; k++) {                                \
            const int o = tid + 256 * k;                                       \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            if (k < n_out && t < len && j < nw) {                              \
                float dq_inter = dqo[k];                                       \
                dq_inter *= ef_sm[t];                                          \
                float dk_inter = dko[k];                                       \
                dk_inter *= er_sm[t];                                          \
                const long long g = ((row0 + t) * nh_total + h) * ds + n0 + j; \
                dQ_mid[g] = dqi[k] + dq_inter;                                 \
                dK_mid[g] = dki[k] + dk_inter;                                 \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    /* dADT, pairs below the diagonal, then above, then the */        \
    /* entering and exit states, as m3_dqkv orders them. */                    \
    for (int t = tid; t < len; t += 256) {                                     \
        float acc = 0.0f;                                                      \
        for (int i = 0; i < t; i++) {                                          \
            const int ti = m3t_tri(i, t, CS);                                  \
            float vdo = vdo_tri[ti];                                           \
            float decay = dec_tri[ti];                                         \
            float kq = kq_tri[ti];                                             \
            acc += vdo * decay * kq;                                           \
        }                                                                      \
        for (int j = t + 1; j < len; j++) {                                    \
            const int ti = m3t_tri(t, j, CS);                                  \
            float vdo = vdo_tri[ti];                                           \
            float decay = dec_tri[ti];                                         \
            float kq = kq_tri[ti];                                             \
            acc -= vdo * decay * kq;                                           \
        }                                                                      \
        acc += qsdo_sm[t] * ef_sm[t];                                          \
        dmrev_sm[t] = acc;                                                     \
        dmvec_sm[t] = dskv_sm[t] * er_sm[t];                                   \
    }                                                                          \
    __syncthreads();                                                           \
    if (tid == 0) {                                                            \
        float dM_scalar = dm_chain[cidx];                                      \
        dM_scalar *= exp2f(cs_sum * LOG2E);                                    \
        float total_rev = 0.0f;                                                \
        for (int t = 0; t < len; t++) total_rev += dmrev_sm[t];                \
        total_rev += dM_scalar;                                                \
        float cumsum = 0.0f;                                                   \
        for (int t = 0; t < len; t++) {                                       \
            cumsum += dmvec_sm[t] - dmrev_sm[t];                               \
            const float out = dmrev_sm[t] + (total_rev + cumsum - dmvec_sm[t]); \
            dADT[(row0 + t) * nh_total + h] = out;                             \
        }                                                                      \
        float dD_acc = 0.0f;                                                   \
        for (int t = 0; t < len; t++) dD_acc += dqk_sm[t];                     \
        dD_partials[cidx] = dD_acc;                                            \
    }                                                                          \
}

DEFINE_M3_DQKV_TILED(m3_dqkv_tiled, float)
DEFINE_M3_DQKV_TILED(m3_dqkv_tiled_bf16, __nv_bfloat16)
DEFINE_M3_DQKV_TILED(m3_dqkv_tiled_f16, __half)
#undef DEFINE_M3_DQKV_TILED
#undef M3T_MAX_SLOTS
#undef M3T_MAX_OUT
