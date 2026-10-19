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

// The decode steps take the state M3_STEP_SLICE at a time, a slice's
// states in registers (the loops over a slice unroll, so they index it
// statically), where one array of the compiled capacity indexed at run
// time stayed in local memory. The output's chain runs on across the
// slices in state order, so the bits are those of the whole-state loop.
#ifndef M3_STEP_SLICE
#define M3_STEP_SLICE 16
#endif

// The decode steps' slice loop comes in two builds. The plain one (PF 0)
// loads a slice at the top of its turn. The prefetching one (PF 1) loads
// the next slice while the current one computes, which pays when a step
// has few warps on a multiprocessor to hide memory with. Only the loads
// move, so both give the same bits, and the host picks one by occupancy.
#ifndef M3_STEP_PRE_0
#define M3_STEP_PRE_0 float h_local[M3_STEP_SLICE];
#define M3_STEP_PRE_1                                                         \
    float h_local[M3_STEP_SLICE];                                             \
    _Pragma("unroll")                                                         \
    for (int j = 0; j < M3_STEP_SLICE; j++) if (j >= ds) break; else          \
        h_local[j] = ssm_state[h_base + j];
#define M3_STEP_LOAD_0                                                        \
    _Pragma("unroll")                                                         \
    for (int j = 0; j < M3_STEP_SLICE; j++) if (n0 + j >= ds) break; else     \
        h_local[j] = ssm_state[h_base + n0 + j];
#define M3_STEP_LOAD_1                                                        \
    float h_next[M3_STEP_SLICE];                                              \
    _Pragma("unroll")                                                         \
    for (int j = 0; j < M3_STEP_SLICE; j++)                                   \
        h_next[j] = n0 + M3_STEP_SLICE + j < ds                               \
            ? ssm_state[h_base + n0 + M3_STEP_SLICE + j] : 0.0f;
#define M3_STEP_CARRY_0
#define M3_STEP_CARRY_1                                                       \
    _Pragma("unroll")                                                         \
    for (int j = 0; j < M3_STEP_SLICE; j++) h_local[j] = h_next[j];
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
#define DEFINE_M3_STEP_FWD_BLOCK(NAME, IDX, TY, FROM_F, PF)                   \
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
    int batch, int nh, int hd, int ds,                                        \
    int y_first /* this run's first y block */                                \
) {                                                                           \
    extern __shared__ float m3sb_sm[];                                        \
    float* coef = m3sb_sm;                                                    \
    float* kc_sm = coef + 4;                                                  \
    float* kp_sm = kc_sm + ds;                                                \
    float* qc_sm = kp_sm + ds;                                                \
    IDX b = blockIdx.x;                                                       \
    int h = y_first + blockIdx.y;                                             \
    int tid = threadIdx.x;                                                    \
    /* Uniform across the block, so every thread reaches the barrier. */      \
    if (b >= batch || h >= nh || ds > MAMBA_RS_STATE_CAP) return;             \
    int d_inner = nh * hd;                                                    \
    IDX row = b * nh * ds + h * ds;                                           \
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
        IDX h_base = (b * nh * hd + h * hd + p) * ds;                         \
        float x_val = to_f(x[b * d_inner + h * hd + p]);                      \
        float v_prev = v_state[b * nh * hd + h * hd + p];                     \
        float y_val = d_skip * x_val;                                         \
        float bv = beta_h * v_prev;                                           \
        float gx = gamma_h * x_val;                                           \
        M3_STEP_PRE_##PF                                                      \
        for (int n0 = 0; n0 < ds; n0 += M3_STEP_SLICE) {                      \
            M3_STEP_LOAD_##PF                                                 \
            _Pragma("unroll")                                                 \
            for (int j = 0; j < M3_STEP_SLICE; j++) if (n0 + j >= ds) break; else { \
                float kc_n = kc_sm[n0 + j];                                   \
                float kp_n = kp_sm[n0 + j];                                   \
                float qc_n = qc_sm[n0 + j];                                   \
                h_local[j] = __fmaf_rn(gx, kc_n,                              \
                                       __fmaf_rn(h_local[j], alpha_h,         \
                                                 __fmul_rn(bv, kp_n)));       \
                y_val = __fmaf_rn(h_local[j], qc_n, y_val);                   \
            }                                                                 \
            _Pragma("unroll")                                                 \
            for (int j = 0; j < M3_STEP_SLICE; j++) if (n0 + j >= ds) break; else\
                ssm_state[h_base + n0 + j] = h_local[j];                      \
            M3_STEP_CARRY_##PF                                                \
        }                                                                     \
        y[b * d_inner + h * hd + p] = FROM_F(y_val);                          \
        v_state[b * nh * hd + h * hd + p] = x_val;                            \
    }                                                                         \
}

DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block, int, float, from_f_f32, 0)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_i64, long long, float, from_f_f32, 0)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_bf16, int, __nv_bfloat16, from_f_bf16, 0)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_bf16_i64, long long, __nv_bfloat16, from_f_bf16, 0)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_f16, int, __half, from_f_f16, 0)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_f16_i64, long long, __half, from_f_f16, 0)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_pf, int, float, from_f_f32, 1)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_pf_i64, long long, float, from_f_f32, 1)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_pf_bf16, int, __nv_bfloat16, from_f_bf16, 1)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_pf_bf16_i64, long long, __nv_bfloat16, from_f_bf16, 1)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_pf_f16, int, __half, from_f_f16, 1)
DEFINE_M3_STEP_FWD_BLOCK(m3_step_fwd_block_pf_f16_i64, long long, __half, from_f_f16, 1)

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
// twins. In a packed row (`seq_idx`, see _varlen.cuh) a token that starts a
// sequence steps from the empty state, previous key and previous input.
template <typename T_ACT, bool SAVE, int DS, bool FOLD_SKIP>
__device__ __forceinline__ void m3_burnin_block_tpl(
    float* ssm_state, float* k_state, float* v_state,
    T_ACT* y_out,
    float* h_saved, float* k_prev_saved, float* v_prev_saved,
    const T_ACT* x_flat, const T_ACT* k_flat, const T_ACT* q_flat,
    const float* alpha_flat, const float* beta_flat, const float* gamma_flat,
    const float* D,
    int batch, int T, int nh, int hd, int ds,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m3bb_sm[];
    float* coef = m3bb_sm;
    float* kc_sm = coef + 4;
    float* kp_sm = kc_sm + ds;
    float* qc_sm = kp_sm + ds;
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
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
            const bool fresh = vl_starts(seq_idx, b, T, t);
            if (tid == 0) {
                coef[0] = alpha_flat[bt * nh + h];
                coef[1] = beta_flat[bt * nh + h];
                coef[2] = gamma_flat[bt * nh + h];
            }
            for (int n = tid; n < ds; n += blockDim.x) {
                const float kp = fresh    ? 0.0f
                                 : (t == 0) ? k_state[k_row + n]
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
                const float v_prev = fresh ? 0.0f : v_state[v_idx];
                if (SAVE) v_prev_saved[x_idx] = v_prev;
                if (fresh) {
                    _Pragma("unroll")
                    for (int n = 0; n < sw; n++) h_local[n] = 0.0f;
                }
                float y_val = __fmul_rn(d_skip, x_val);
                const float bv = __fmul_rn(beta_h, v_prev);
                const float gx = __fmul_rn(gamma_h, x_val);
                if constexpr (DS > 0 && DS % 4 == 0) {
                    // A constant width reads the three rows four indices at
                    // a time through PTX (see _shared_loads.cuh); the chains
                    // take the same values in the same order.
                    _Pragma("unroll")
                    for (int n4 = 0; n4 < DS; n4 += 4) {
                        const float4 kc4 = lds_f32x4(kc_sm + n4);
                        const float4 kp4 = lds_f32x4(kp_sm + n4);
                        const float4 qc4 = lds_f32x4(qc_sm + n4);
                        const float kc[4] = {kc4.x, kc4.y, kc4.z, kc4.w};
                        const float kp[4] = {kp4.x, kp4.y, kp4.z, kp4.w};
                        const float qc[4] = {qc4.x, qc4.y, qc4.z, qc4.w};
                        _Pragma("unroll")
                        for (int j = 0; j < 4; j++) {
                            const int n = n4 + j;
                            h_local[n] = __fmaf_rn(gx, kc[j],
                                __fmaf_rn(alpha_h, h_local[n], __fmul_rn(bv, kp[j])));
                            if (FOLD_SKIP && n == 0)
                                y_val = __fmaf_rn(d_skip, x_val, __fmul_rn(qc[0], h_local[0]));
                            else
                                y_val = __fmaf_rn(qc[j], h_local[n], y_val);
                        }
                    }
                } else {
                    _Pragma("unroll")
                    for (int n = 0; n < sw; n++) {
                        h_local[n] = __fmaf_rn(gx, kc_sm[n],
                            __fmaf_rn(alpha_h, h_local[n], __fmul_rn(bv, kp_sm[n])));
                        if (FOLD_SKIP && n == 0)
                            y_val = __fmaf_rn(d_skip, x_val, __fmul_rn(qc_sm[0], h_local[0]));
                        else
                            y_val = __fmaf_rn(qc_sm[n], h_local[n], y_val);
                    }
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

// One entry per state width the warp kernels specialise (8, 16, 32 and
// 64) and a general one for every other width, so each takes the
// registers its own state loops need, where one entry that switched
// among them at run time took the widest's for every width. The host
// picks the entry by the model's d_state.
#define M3_BURNIN_BLOCK_ENTRY(NAME, T_ACT, SAVE, DS, FOLD)                     \
extern "C" __global__ __launch_bounds__(512) void NAME(                        \
    float* ssm_state, float* k_state, float* v_state, T_ACT* y_out,            \
    float* h_saved, float* k_prev_saved, float* v_prev_saved,                  \
    const T_ACT* x_flat, const T_ACT* k_flat, const T_ACT* q_flat,             \
    const float* alpha_flat, const float* beta_flat, const float* gamma_flat,  \
    const float* D, int batch, int T, int nh, int hd, int ds,                  \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3_burnin_block_tpl<T_ACT, SAVE, DS, FOLD>(ssm_state, k_state, v_state,    \
        y_out, h_saved, k_prev_saved, v_prev_saved, x_flat, k_flat, q_flat,    \
        alpha_flat, beta_flat, gamma_flat, D, batch, T, nh, hd, ds, y_first,   \
        seq_idx);                                                              \
}

M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_ds8, float, true, 8, true)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_ds8_bf16, __nv_bfloat16, true, 8, false)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_ds8_f16, __half, true, 8, false)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_ds16, float, true, 16, true)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_ds16_bf16, __nv_bfloat16, true, 16, false)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_ds16_f16, __half, true, 16, false)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_ds32, float, true, 32, true)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_ds32_bf16, __nv_bfloat16, true, 32, false)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_ds32_f16, __half, true, 32, false)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_ds64, float, true, 64, true)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_ds64_bf16, __nv_bfloat16, true, 64, false)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_ds64_f16, __half, true, 64, false)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block, float, true, 0, false)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_bf16, __nv_bfloat16, true, 0, false)
M3_BURNIN_BLOCK_ENTRY(m3_burnin_fwd_block_f16, __half, true, 0, false)
#undef M3_BURNIN_BLOCK_ENTRY

// The target network's forward: arguments as m3_burnin_fwd_nosave.
extern "C" __global__ __launch_bounds__(512) void m3_burnin_fwd_nosave_block(
    float* ssm_state, float* k_state, float* v_state, float* y_out,
    const float* x_flat, const float* k_flat, const float* q_flat,
    const float* alpha_flat, const float* beta_flat, const float* gamma_flat,
    const float* D, int batch, int T, int nh, int hd, int ds,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    m3_burnin_block_tpl<float, false, 0, false>(ssm_state, k_state, v_state, y_out,
        nullptr, nullptr, nullptr, x_flat, k_flat, q_flat, alpha_flat,
        beta_flat, gamma_flat, D, batch, T, nh, hd, ds, y_first, seq_idx);
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
// gradients the slices before it wrote. In a packed row a token that starts
// a sequence takes no alpha gradient and passes no state, key or input
// gradient back to the token before it.
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
    int batch, int T, int nh, int hd, int ds,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m3bs_sm[];
    float* coef = m3bs_sm;
    float* kc_sm = coef + 4;
    float* kp_sm = kc_sm + ds;
    float* qc_sm = kp_sm + ds;
    float* carry_sm = qc_sm + ds;          /* [ds] previous-K gradient */
    float* red = carry_sm + ds;            /* [(3 * ds + 3) * warps] */
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
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
            const bool fresh = vl_starts(seq_idx, b, T, t);
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
            // One FMA per step, spelled out as in the warp kernels, so the
            // two tiers keep the same last bit on every architecture.
            sum_d_D = __fmaf_rn(dy_val, x_val, sum_d_D);
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
                    if (!fresh) d_alpha_acc += dh_n * h_prev_n;
                    d_beta_acc += dh_n * v_prev * kp_n;
                    d_gamma_acc += dh_n * x_val * kc_n;
                    d_v_prev_acc += dh_n * beta_h * kp_n;
                    d_kp_val = dh_n * beta_h * v_prev;
                    d_x_val += dh_n * gamma_h * kc_n;
                    d_kc_val = dh_n * gamma_h * x_val;
                    d_h_reg[n] = fresh ? 0.0f : alpha_h * dh_n;
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
                d_v_carry = fresh ? 0.0f : d_v_prev_acc;
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
                    if (!fresh) carry_sm[n] += kp;
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

// Entries per state width as for the forward.
#define M3_BACKWARD_SEQ_BLOCK_ENTRY(NAME, T_ACT, DS)                           \
extern "C" __global__ __launch_bounds__(512) void NAME(                        \
    const float* __restrict__ h_saved, const float* __restrict__ k_prev_saved, \
    const float* __restrict__ v_prev_saved, const T_ACT* __restrict__ x_flat,  \
    const T_ACT* __restrict__ k_flat, const T_ACT* __restrict__ q_flat,        \
    const float* __restrict__ alpha_flat, const float* __restrict__ beta_flat, \
    const float* __restrict__ gamma_flat, const float* __restrict__ D,         \
    const T_ACT* __restrict__ d_y_flat,                                        \
    float* __restrict__ d_x, float* __restrict__ d_k, float* __restrict__ d_q, \
    float* __restrict__ d_alpha, float* __restrict__ d_beta,                   \
    float* __restrict__ d_gamma, float* __restrict__ d_D_local,                \
    int batch, int T, int nh, int hd, int ds,                                  \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3_backward_seq_block_tpl<T_ACT, DS>(h_saved, k_prev_saved, v_prev_saved,  \
        x_flat, k_flat, q_flat, alpha_flat, beta_flat, gamma_flat, D,          \
        d_y_flat, d_x, d_k, d_q, d_alpha, d_beta, d_gamma, d_D_local, batch,   \
        T, nh, hd, ds, y_first, seq_idx);                                      \
}

M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_ds8, float, 8)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_ds8_bf16, __nv_bfloat16, 8)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_ds8_f16, __half, 8)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_ds16, float, 16)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_ds16_bf16, __nv_bfloat16, 16)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_ds16_f16, __half, 16)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_ds32, float, 32)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_ds32_bf16, __nv_bfloat16, 32)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_ds32_f16, __half, 32)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_ds64, float, 64)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_ds64_bf16, __nv_bfloat16, 64)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_ds64_f16, __half, 64)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block, float, 0)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_bf16, __nv_bfloat16, 0)
M3_BACKWARD_SEQ_BLOCK_ENTRY(m3_backward_seq_block_f16, __half, 0)
#undef M3_BACKWARD_SEQ_BLOCK_ENTRY

// ============================================================================
// The sequential backward with the state cut across the block.
//
// The block tier's thread walks every state of its head element each step,
// in an array of the compiled capacity that stays in local memory at a
// state width no entry specialises. Here the state is cut into slices of
// M3S_SLICE states and each slice has threads of its own, a thread a head
// element: a thread's slice of the state gradient stays in registers at any
// width, and a step takes a slice's work instead of the whole state's.
//
// A step's sums over the state (each element's input gradient and its
// previous input's) add the slices' partials in slice order, the first
// slice's carrying the skip term and the gradient carried from the step
// after; its scalars (alpha, beta, gamma) add every warp's tree in warp
// order; each state's sums over the head add the head's warps in order, as
// the block tier does. The order is fixed, so the result does not depend on
// the run; a state of one slice gets the block tier's bits, and a wider one
// rounds the same sums in another order.
//
// Grid (B, nh). Block: the state's slices times the threads of a slice, the
// head rounded up to whole warps and at most 512 / slices of them, rounded
// down to whole warps; a head wider than a slice's threads runs in turns,
// each over the whole window, a turn after the first adding its sums to the
// gradients the turns before it wrote. Dynamic shared memory, in floats:
// 4 + 4 * ds for the step's values and the carried previous-K gradient,
// 3 * ds * pwarps for each state's sums by head warp, 2 * slices * pthreads
// for each element's two sums by slice, and 3 * slices * pwarps for the
// scalars' sums by warp. In a packed row a token that starts a sequence
// takes no alpha gradient and passes no state, key or input gradient back
// to the token before it.
#define M3S_SLICE 16

template <typename T_ACT>
__device__ __forceinline__ void m3_backward_seq_sliced_tpl(
    const float* __restrict__ h_saved, const float* __restrict__ k_prev_saved,
    const float* __restrict__ v_prev_saved, const T_ACT* __restrict__ x_flat,
    const T_ACT* __restrict__ k_flat, const T_ACT* __restrict__ q_flat,
    const float* __restrict__ alpha_flat, const float* __restrict__ beta_flat,
    const float* __restrict__ gamma_flat, const float* __restrict__ D,
    const T_ACT* __restrict__ d_y_flat,
    float* __restrict__ d_x, float* __restrict__ d_k, float* __restrict__ d_q,
    float* __restrict__ d_alpha, float* __restrict__ d_beta,
    float* __restrict__ d_gamma, float* __restrict__ d_D_local,
    int batch, int T, int nh, int hd, int ds,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m3ss_sm[];
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
    /* Uniform across the block, so every thread reaches the barriers. */
    if (b >= batch || h >= nh) return;
    const int slices = (ds + M3S_SLICE - 1) / M3S_SLICE;
    const int pthreads = blockDim.x / slices;  /* whole warps, by the launch */
    const int pwarps = pthreads >> 5;
    const int tid = threadIdx.x;
    const int s = tid / pthreads;
    const int tx = tid - s * pthreads;
    const int lane = tid & 31;
    const int pw = tx >> 5;
    const int width = hd < 32 ? hd : 32;
    float* coef = m3ss_sm;
    float* kc_sm = coef + 4;
    float* kp_sm = kc_sm + ds;
    float* qc_sm = kp_sm + ds;
    float* carry_sm = qc_sm + ds;                     /* [ds] previous-K gradient */
    float* red_n = carry_sm + ds;                     /* [3 * ds * pwarps] */
    float* part = red_n + 3 * ds * pwarps;            /* [2 * slices * pthreads] */
    float* red_s = part + 2 * slices * pthreads;      /* [3 * slices * pwarps] */
    const long long d_inner = (long long)nh * hd;
    const long long nhd_ds = d_inner * ds;
    const long long head_ds = (long long)h * ds;
    const float d_skip = D[h];
    const int n0 = s * M3S_SLICE;
    const int n_end = min(n0 + M3S_SLICE, ds);
    for (int p0 = 0; p0 < hd; p0 += pthreads) {
        const int p = p0 + tx;
        const bool live = p < hd;
        const bool first = p0 == 0;
        const long long h_tape = (long long)b * (T + 1) * nhd_ds + (long long)h * hd + p;
        for (int n = tid; n < ds; n += blockDim.x) carry_sm[n] = 0.0f;

        float d_h_reg[M3S_SLICE];
        _Pragma("unroll")
        for (int j = 0; j < M3S_SLICE; j++) d_h_reg[j] = 0.0f;
        float d_v_carry = 0.0f;
        float sum_d_D = 0.0f;

        for (int t = T - 1; t >= 0; t--) {
            const long long bt = (long long)b * T + t;
            const long long kq_row = bt * nh * ds + head_ds;
            const bool fresh = vl_starts(seq_idx, b, T, t);
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
            // The first slice carries the skip term and the gradient the
            // step after passed back, in the block tier's order, so its sum
            // is the block tier's whenever the state is one slice.
            float d_x_val = 0.0f;
            if (s == 0) {
                sum_d_D = __fmaf_rn(dy_val, x_val, sum_d_D);
                d_x_val = d_skip * dy_val;
                d_x_val += d_v_carry;
            }
            float d_alpha_acc = 0.0f;
            float d_beta_acc = 0.0f;
            float d_gamma_acc = 0.0f;
            float d_v_prev_acc = 0.0f;
            const long long prev = h_tape + (long long)t * nhd_ds;
            const long long curr = prev + nhd_ds;
            _Pragma("unroll")
            for (int j = 0; j < M3S_SLICE; j++) {
                const int n = n0 + j;
                /* The same for a slice's every thread: whole warps leave. */
                if (n >= n_end) break;
                const float kc_n = kc_sm[n];
                const float kp_n = kp_sm[n];
                const float qc_n = qc_sm[n];
                float d_q_val = -0.0f, d_kp_val = -0.0f, d_kc_val = -0.0f;
                if (live) {
                    const float h_prev_n = h_saved[prev + n * d_inner];
                    const float h_curr_n = h_saved[curr + n * d_inner];
                    d_h_reg[j] += dy_val * qc_n;
                    d_q_val = dy_val * h_curr_n;
                    const float dh_n = d_h_reg[j];
                    if (!fresh) d_alpha_acc += dh_n * h_prev_n;
                    d_beta_acc += dh_n * v_prev * kp_n;
                    d_gamma_acc += dh_n * x_val * kc_n;
                    d_v_prev_acc += dh_n * beta_h * kp_n;
                    d_kp_val = dh_n * beta_h * v_prev;
                    d_x_val += dh_n * gamma_h * kc_n;
                    d_kc_val = dh_n * gamma_h * x_val;
                    d_h_reg[j] = fresh ? 0.0f : alpha_h * dh_n;
                }
                d_q_val = m3b_warp_tree(d_q_val, width);
                d_kp_val = m3b_warp_tree(d_kp_val, width);
                d_kc_val = m3b_warp_tree(d_kc_val, width);
                if (lane == 0) {
                    red_n[(3 * n) * pwarps + pw] = d_q_val;
                    red_n[(3 * n + 1) * pwarps + pw] = d_kp_val;
                    red_n[(3 * n + 2) * pwarps + pw] = d_kc_val;
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
            const int warp = s * pwarps + pw;
            const int warps = slices * pwarps;
            if (lane == 0) {
                red_s[warp] = d_alpha_acc;
                red_s[warps + warp] = d_beta_acc;
                red_s[2 * warps + warp] = d_gamma_acc;
            }
            part[s * pthreads + tx] = d_x_val;
            part[(slices + s) * pthreads + tx] = d_v_prev_acc;
            __syncthreads();
            /* The first slice's threads add each element's partials in slice
               order; one thread per state element and per scalar adds the
               warps' sums in order. The rows are restaged only after the
               next barrier. */
            if (s == 0 && live) {
                float x_sum = part[tx];
                float v_sum = part[slices * pthreads + tx];
                for (int r = 1; r < slices; r++) {
                    x_sum += part[r * pthreads + tx];
                    v_sum += part[(slices + r) * pthreads + tx];
                }
                d_x[x_idx] = x_sum;
                d_v_carry = fresh ? 0.0f : v_sum;
            }
            for (int i = tid; i < ds + 3; i += blockDim.x) {
                if (i < ds) {
                    const int n = i;
                    float q = red_n[(3 * n) * pwarps];
                    float kp = red_n[(3 * n + 1) * pwarps];
                    float kc = red_n[(3 * n + 2) * pwarps];
                    for (int w = 1; w < pwarps; w++) {
                        q += red_n[(3 * n) * pwarps + w];
                        kp += red_n[(3 * n + 1) * pwarps + w];
                        kc += red_n[(3 * n + 2) * pwarps + w];
                    }
                    float d_k_write = 0.0f;
                    if (t < T - 1) {
                        d_k_write += carry_sm[n];
                        carry_sm[n] = 0.0f;
                    }
                    if (!fresh) carry_sm[n] += kp;
                    d_k_write += kc;
                    if (first) {
                        d_q[kq_row + n] = q;
                        d_k[kq_row + n] = d_k_write;
                    } else {
                        d_q[kq_row + n] += q;
                        d_k[kq_row + n] += d_k_write;
                    }
                } else {
                    const float* sums = red_s + (i - ds) * warps;
                    float total = sums[0];
                    for (int w = 1; w < warps; w++) total += sums[w];
                    float* out = (i == ds) ? d_alpha : (i == ds + 1) ? d_beta : d_gamma;
                    if (first)
                        out[bt * nh + h] = total;
                    else
                        out[bt * nh + h] += total;
                }
            }
        }
        if (s == 0 && live) d_D_local[(long long)b * d_inner + (long long)h * hd + p] = sum_d_D;
    }
}

#define M3_BACKWARD_SEQ_SLICED_ENTRY(NAME, T_ACT)                              \
extern "C" __global__ __launch_bounds__(512) void NAME(                        \
    const float* __restrict__ h_saved, const float* __restrict__ k_prev_saved, \
    const float* __restrict__ v_prev_saved, const T_ACT* __restrict__ x_flat,  \
    const T_ACT* __restrict__ k_flat, const T_ACT* __restrict__ q_flat,        \
    const float* __restrict__ alpha_flat, const float* __restrict__ beta_flat, \
    const float* __restrict__ gamma_flat, const float* __restrict__ D,         \
    const T_ACT* __restrict__ d_y_flat,                                        \
    float* __restrict__ d_x, float* __restrict__ d_k, float* __restrict__ d_q, \
    float* __restrict__ d_alpha, float* __restrict__ d_beta,                   \
    float* __restrict__ d_gamma, float* __restrict__ d_D_local,                \
    int batch, int T, int nh, int hd, int ds,                                  \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3_backward_seq_sliced_tpl<T_ACT>(h_saved, k_prev_saved, v_prev_saved,     \
        x_flat, k_flat, q_flat, alpha_flat, beta_flat, gamma_flat, D,          \
        d_y_flat, d_x, d_k, d_q, d_alpha, d_beta, d_gamma, d_D_local, batch,   \
        T, nh, hd, ds, y_first, seq_idx);                                      \
}

M3_BACKWARD_SEQ_SLICED_ENTRY(m3_backward_seq_sliced, float)
M3_BACKWARD_SEQ_SLICED_ENTRY(m3_backward_seq_sliced_bf16, __nv_bfloat16)
M3_BACKWARD_SEQ_SLICED_ENTRY(m3_backward_seq_sliced_f16, __half)
#undef M3_BACKWARD_SEQ_SLICED_ENTRY

// The sequential training forward with the state cut across the block, the
// backward's twin: each slice of M3S_SLICE states on threads of its own, a
// thread a head element, the slice's states in registers at any width. A
// step's output adds the slices' partials in slice order, the first
// slice's starting from the skip term, so a state of one slice gets the
// block tier's general entry's bits (the skip term rounded on its own, then
// the state's terms fused in order), and a wider one rounds the same sum in
// another order. The previous input each thread reads is the one it read
// the step before (the state's at the first step), so the slices never
// share the input state while they step. SAVE false drops the three saves,
// for the target network's forward. Grid (B, nh); block as the sliced
// backward's. Dynamic shared memory, in floats: 4 + 3 * ds for the step's
// values and slices * pthreads for the output's partials. In a packed row a
// token that starts a sequence steps from the empty state, previous key and
// previous input.
template <typename T_ACT, bool SAVE>
__device__ __forceinline__ void m3_burnin_sliced_tpl(
    float* ssm_state, float* k_state, float* v_state,
    T_ACT* y_out,
    float* h_saved, float* k_prev_saved, float* v_prev_saved,
    const T_ACT* x_flat, const T_ACT* k_flat, const T_ACT* q_flat,
    const float* alpha_flat, const float* beta_flat, const float* gamma_flat,
    const float* D,
    int batch, int T, int nh, int hd, int ds,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m3sf_sm[];
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
    /* Uniform across the block, so every thread reaches the barriers. */
    if (b >= batch || h >= nh) return;
    const int slices = (ds + M3S_SLICE - 1) / M3S_SLICE;
    const int pthreads = blockDim.x / slices;  /* whole warps, by the launch */
    const int tid = threadIdx.x;
    const int s = tid / pthreads;
    const int tx = tid - s * pthreads;
    float* coef = m3sf_sm;
    float* kc_sm = coef + 4;
    float* kp_sm = kc_sm + ds;
    float* qc_sm = kp_sm + ds;
    float* part = qc_sm + ds;                         /* [slices * pthreads] */
    const long long d_inner = (long long)nh * hd;
    const long long nhd_ds = d_inner * ds;
    const long long head_ds = (long long)h * ds;
    const long long row_stride = (long long)nh * ds;
    const long long k_row = (long long)b * row_stride + head_ds;
    const float d_skip = D[h];
    const int n0 = s * M3S_SLICE;
    const int n_end = min(n0 + M3S_SLICE, ds);
    for (int p0 = 0; p0 < hd; p0 += pthreads) {
        const int p = p0 + tx;
        const bool live = p < hd;
        const long long v_idx = (long long)b * nh * hd + (long long)h * hd + p;
        const long long h_base = v_idx * ds;
        const long long h_tape = (long long)b * (T + 1) * nhd_ds + (long long)h * hd + p;
        float h_local[M3S_SLICE];
        _Pragma("unroll")
        for (int j = 0; j < M3S_SLICE; j++) h_local[j] = 0.0f;
        float x_prev = 0.0f;
        if (live) {
            _Pragma("unroll")
            for (int j = 0; j < M3S_SLICE; j++) {
                const int n = n0 + j;
                if (n >= n_end) break;
                h_local[j] = ssm_state[h_base + n];
                if (SAVE) h_saved[h_tape + n * d_inner] = h_local[j];
            }
            x_prev = v_state[v_idx];
        }
        for (int t = 0; t < T; t++) {
            const long long bt = (long long)b * T + t;
            const long long kq_row = bt * row_stride + head_ds;
            const bool fresh = vl_starts(seq_idx, b, T, t);
            if (tid == 0) {
                coef[0] = alpha_flat[bt * nh + h];
                coef[1] = beta_flat[bt * nh + h];
                coef[2] = gamma_flat[bt * nh + h];
            }
            for (int n = tid; n < ds; n += blockDim.x) {
                const float kp = fresh    ? 0.0f
                                 : (t == 0) ? k_state[k_row + n]
                                            : to_f(k_flat[kq_row - row_stride + n]);
                kp_sm[n] = kp;
                if (SAVE && p0 == 0) k_prev_saved[kq_row + n] = kp;
                kc_sm[n] = to_f(k_flat[kq_row + n]);
                qc_sm[n] = to_f(q_flat[kq_row + n]);
            }
            __syncthreads();
            const long long x_idx = bt * d_inner + (long long)h * hd + p;
            if (live) {
                const float alpha_h = coef[0];
                const float beta_h = coef[1];
                const float gamma_h = coef[2];
                const float x_val = to_f(x_flat[x_idx]);
                const float v_prev = fresh ? 0.0f : x_prev;
                if (SAVE && s == 0) v_prev_saved[x_idx] = v_prev;
                if (fresh) {
                    _Pragma("unroll")
                    for (int j = 0; j < M3S_SLICE; j++) h_local[j] = 0.0f;
                }
                float y_part = (s == 0) ? __fmul_rn(d_skip, x_val) : 0.0f;
                const float bv = __fmul_rn(beta_h, v_prev);
                const float gx = __fmul_rn(gamma_h, x_val);
                const long long step = h_tape + (long long)(t + 1) * nhd_ds;
                _Pragma("unroll")
                for (int j = 0; j < M3S_SLICE; j++) {
                    const int n = n0 + j;
                    if (n >= n_end) break;
                    h_local[j] = __fmaf_rn(gx, kc_sm[n],
                        __fmaf_rn(alpha_h, h_local[j], __fmul_rn(bv, kp_sm[n])));
                    y_part = __fmaf_rn(qc_sm[n], h_local[j], y_part);
                    if (SAVE) h_saved[step + n * d_inner] = h_local[j];
                }
                part[s * pthreads + tx] = y_part;
                x_prev = x_val;
            }
            __syncthreads();
            /* The first slice's threads add each element's partials in slice
               order; the rows and the partials are written again only after
               the next step's first barrier. */
            if (s == 0 && live) {
                float y_val = part[tx];
                for (int r = 1; r < slices; r++) y_val += part[r * pthreads + tx];
                m3b_store(&y_out[x_idx], y_val);
            }
        }
        if (live) {
            _Pragma("unroll")
            for (int j = 0; j < M3S_SLICE; j++) {
                const int n = n0 + j;
                if (n >= n_end) break;
                ssm_state[h_base + n] = h_local[j];
            }
            if (s == 0) v_state[v_idx] = x_prev;
        }
    }
    /* Every turn and slice has read the state's K by now. */
    __syncthreads();
    if (T > 0) {
        const long long last = ((long long)b * T + T - 1) * row_stride + head_ds;
        for (int n = tid; n < ds; n += blockDim.x) k_state[k_row + n] = to_f(k_flat[last + n]);
    }
}

#define M3_BURNIN_SLICED_ENTRY(NAME, T_ACT)                                    \
extern "C" __global__ __launch_bounds__(512) void NAME(                        \
    float* ssm_state, float* k_state, float* v_state, T_ACT* y_out,            \
    float* h_saved, float* k_prev_saved, float* v_prev_saved,                  \
    const T_ACT* x_flat, const T_ACT* k_flat, const T_ACT* q_flat,             \
    const float* alpha_flat, const float* beta_flat, const float* gamma_flat,  \
    const float* D, int batch, int T, int nh, int hd, int ds,                  \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3_burnin_sliced_tpl<T_ACT, true>(ssm_state, k_state, v_state, y_out,      \
        h_saved, k_prev_saved, v_prev_saved, x_flat, k_flat, q_flat,           \
        alpha_flat, beta_flat, gamma_flat, D, batch, T, nh, hd, ds, y_first,   \
        seq_idx);                                                              \
}

M3_BURNIN_SLICED_ENTRY(m3_burnin_fwd_sliced, float)
M3_BURNIN_SLICED_ENTRY(m3_burnin_fwd_sliced_bf16, __nv_bfloat16)
M3_BURNIN_SLICED_ENTRY(m3_burnin_fwd_sliced_f16, __half)
#undef M3_BURNIN_SLICED_ENTRY

// The target network's forward on the sliced tier: arguments as
// m3_burnin_fwd_nosave.
extern "C" __global__ __launch_bounds__(512) void m3_burnin_fwd_nosave_sliced(
    float* ssm_state, float* k_state, float* v_state, float* y_out,
    const float* x_flat, const float* k_flat, const float* q_flat,
    const float* alpha_flat, const float* beta_flat, const float* gamma_flat,
    const float* D, int batch, int T, int nh, int hd, int ds,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    m3_burnin_sliced_tpl<float, false>(ssm_state, k_state, v_state, y_out,
        nullptr, nullptr, nullptr, x_flat, k_flat, q_flat, alpha_flat,
        beta_flat, gamma_flat, D, batch, T, nh, hd, ds, y_first, seq_idx);
}

// ============================================================================
// The chunked backward for every head and state width.
//
// m3_dqkv keeps a head's whole state in shared memory and in registers, so
// it serves heads of at most one warp and states whose tiles fit beside
// them. The kernels below compute the same outputs for any headdim and any
// d_state in a fixed shared-memory budget: every operand streams through
// fixed tiles, and every sum runs in the order m3_dqkv runs it, its partial
// carried exactly as a float from one tile to the next, so at every width
// m3_dqkv serves the outputs are its bits. Chains run over valid indices
// only, so padding never adds a zero that could turn a -0.0 into +0.0.
// ============================================================================

// Index of the pair (a, b), a < b, in a chunk's strict upper triangle.
__device__ __forceinline__ int m3t_tri(int a, int b, int cs) {
    return a * (2 * cs - a - 1) / 2 + (b - a - 1);
}

// The scalar the chunk's saved state adds to dADT: the flat chain over
// (p, n), p-major, of the saved state times its gradient, one thread's
// chain in order, since the order is the numeric contract. A warp takes a
// (b, chunk, h): its lanes stage the two rows M3D_SEG values at a time in
// the warp's slice of shared memory, coalesced, and lane 0 runs the chain
// over them while the next segment's loads are in flight. A kernel of its
// own, so the chain delays no other work.
// Grid: ceil(rows / M3D_WARPS). Block: 32 * M3D_WARPS.
#define M3D_WARPS 8
#define M3D_SEG 128
extern "C" __global__ __launch_bounds__(32 * M3D_WARPS) void m3_dqkv_dm_chain(
    float* __restrict__ dm_out,             // [B*n_chunks*nh]
    const float* __restrict__ SSM_States,   // [B*n_chunks*nh*hd*ds]
    const float* __restrict__ enter_state,  // [B*n_chunks*nh*hd*ds]
    int rows, int hd, int ds
) {
    __shared__ __align__(16) float seg_sm[M3D_WARPS][2][2][M3D_SEG];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const long long row = (long long)blockIdx.x * M3D_WARPS + warp;
    if (row >= rows) return;
    const int n_el = hd * ds;
    const float* s_row = SSM_States + row * n_el;
    const float* e_row = enter_state + row * n_el;
    const bool vec = (n_el & 3) == 0 && M3_QUAD_ALIGNED(SSM_States, float)
                     && M3_QUAD_ALIGNED(enter_state, float);
    // A lane's four values of the segment from `at`, zeros past the row.
    int at = 4 * lane;
    float4 sv = m3_load_quad(s_row + at, n_el - at, vec);
    float4 ev = m3_load_quad(e_row + at, n_el - at, vec);
    reinterpret_cast<float4*>(seg_sm[warp][0][0])[lane] = sv;
    reinterpret_cast<float4*>(seg_sm[warp][0][1])[lane] = ev;
    __syncwarp();
    float acc = 0.0f;
    int cur = 0;
    for (int s0 = 0; s0 < n_el; s0 += M3D_SEG) {
        const bool more = s0 + M3D_SEG < n_el;
        if (more) {
            at = s0 + M3D_SEG + 4 * lane;
            sv = m3_load_quad(s_row + at, n_el - at, vec);
            ev = m3_load_quad(e_row + at, n_el - at, vec);
        }
        if (lane == 0) {
            const float* sr = seg_sm[warp][cur][0];
            const float* er = seg_sm[warp][cur][1];
            const int m = min(M3D_SEG, n_el - s0);
            for (int i = 0; i < m; i++) acc = __fmaf_rn(sr[i], er[i], acc);
        }
        if (more) {
            reinterpret_cast<float4*>(seg_sm[warp][cur ^ 1][0])[lane] = sv;
            reinterpret_cast<float4*>(seg_sm[warp][cur ^ 1][1])[lane] = ev;
        }
        __syncwarp();
        cur ^= 1;
    }
    if (lane == 0) dm_out[row] = acc;
}
#undef M3D_SEG
#undef M3D_WARPS

// The per-chunk terms of the reverse state pass: state_B[p][n] = the
// ascending-t chain of dO[t][p] * exp2(dA[t]) (the product rounded once)
// fused with Q[t][n], from +0 over the tokens before the chunk's first
// sequence start, the chain of m3_dqkv_state_terms, for any width. A block
// takes a tile of M3S_TILE channels by M3S_TILE states of one (h, b, chunk),
// the chunk's weighted dO rows and Q rows staged whole; each of its 256
// threads holds a 4 x 4 block of outputs, so every value it reads feeds
// four chains.
// Grid: (nh * ceil(hd / M3S_TILE) * ceil(ds / M3S_TILE), B, n_chunks).
// Block: 256.
#define M3S_TILE 64
#define M3S_LD (M3S_TILE + 4)
#define DEFINE_M3_DQKV_STATE_TERMS_TILED(NAME, T_ACT)                          \
extern "C" __global__ __launch_bounds__(256) void NAME(                        \
    float* __restrict__ state_B,                                               \
    const T_ACT* __restrict__ Q_rot,                                           \
    const float* __restrict__ DA_CS,                                           \
    const T_ACT* __restrict__ dO,                                              \
    int B, int T, int nh_total, int hd, int ds, int CS,                        \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    const int p_tiles = (hd + M3S_TILE - 1) / M3S_TILE;                        \
    const int n_tiles = (ds + M3S_TILE - 1) / M3S_TILE;                        \
    const int h = blockIdx.x / (p_tiles * n_tiles);                            \
    const int tile = blockIdx.x - h * p_tiles * n_tiles;                       \
    const int p0 = (tile / n_tiles) * M3S_TILE;                                \
    const int n0 = (tile - (tile / n_tiles) * n_tiles) * M3S_TILE;             \
    const int b = y_first + blockIdx.y;                                        \
    const int chunk_idx = z_first + blockIdx.z;                                \
    const int tid = threadIdx.x;                                               \
    if (h >= nh_total || b >= B || CS > 64) return;                            \
    const int n_chunks = (T + CS - 1) / CS;                                    \
    const int chunk_start = chunk_idx * CS;                                    \
    const int len = min(CS, T - chunk_start);                                  \
    /* In a packed row (seq_idx) only the tokens before the chunk's */         \
    /* first start read the state entering it. */                              \
    const int cont = vl_lower_bound(seq_idx, b, T, chunk_start, chunk_start + len, \
                                    vl_id(seq_idx, b, T, chunk_start - 1) + 1) \
                     - chunk_start;                                            \
    const long long d_inner = (long long)nh_total * hd;                        \
    const long long cidx = ((long long)b * n_chunks + chunk_idx) * nh_total + h; \
    const long long row0 = (long long)b * T + chunk_start;                     \
    __shared__ __align__(16) float w_sm[64 * M3S_LD];                          \
    __shared__ __align__(16) float q_sm[64 * M3S_LD];                          \
    __shared__ float e_sm[64];                                                 \
    for (int t = tid; t < cont; t += 256)                                      \
        e_sm[t] = exp2f(DA_CS[cidx * CS + t] * LOG2E);                         \
    __syncthreads();                                                           \
    /* The tile's dO rows times their decay, each product rounded once, */    \
    /* and its Q rows, four values a load where the rows allow it. */          \
    const int pw = min(M3S_TILE, hd - p0);                                     \
    const int nw = min(M3S_TILE, ds - n0);                                     \
    const bool vec_o = (hd & 3) == 0 && M3_QUAD_ALIGNED(dO, T_ACT);            \
    const bool vec_q = (ds & 3) == 0 && M3_QUAD_ALIGNED(Q_rot, T_ACT);         \
    for (int i = tid; i < cont * (M3S_TILE / 4); i += 256) {                   \
        const int t = i / (M3S_TILE / 4);                                      \
        const int c = i - t * (M3S_TILE / 4);                                  \
        const float4 v = m3_load_quad(                                         \
            dO + (row0 + t) * d_inner + (long long)h * hd + p0 + 4 * c,        \
            pw - 4 * c, vec_o);                                                \
        const float e = e_sm[t];                                               \
        float4 w;                                                              \
        w.x = __fmul_rn(v.x, e);                                               \
        w.y = __fmul_rn(v.y, e);                                               \
        w.z = __fmul_rn(v.z, e);                                               \
        w.w = __fmul_rn(v.w, e);                                               \
        *reinterpret_cast<float4*>(w_sm + t * M3S_LD + 4 * c) = w;             \
        *reinterpret_cast<float4*>(q_sm + t * M3S_LD + 4 * c) = m3_load_quad(  \
            Q_rot + ((row0 + t) * nh_total + h) * ds + n0 + 4 * c,             \
            nw - 4 * c, vec_q);                                                \
    }                                                                          \
    __syncthreads();                                                           \
    /* Thread (tp, tn) holds channels p0 + 4 tp .. + 3 by states */            \
    /* n0 + 4 tn .. + 3, each output its own ascending-t chain from +0. */     \
    const int tp = tid >> 4;                                                   \
    const int tn = tid & 15;                                                   \
    if (4 * tp >= pw || 4 * tn >= nw) return;                                  \
    float acc[4][4];                                                           \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 4; i++) {                                              \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;                          \
    }                                                                          \
    for (int t = 0; t < cont; t++) {                                           \
        const float4 w4 = *reinterpret_cast<const float4*>(w_sm + t * M3S_LD + 4 * tp); \
        const float4 q4 = *reinterpret_cast<const float4*>(q_sm + t * M3S_LD + 4 * tn); \
        const float wv[4] = {w4.x, w4.y, w4.z, w4.w};                          \
        const float qv[4] = {q4.x, q4.y, q4.z, q4.w};                          \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 4; i++) {                                          \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 4; j++)                                        \
                acc[i][j] = __fmaf_rn(wv[i], qv[j], acc[i][j]);                \
        }                                                                      \
    }                                                                          \
    const bool vec_s = (ds & 3) == 0 && M3_QUAD_ALIGNED(state_B, float);       \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 4; i++) {                                              \
        if (4 * tp + i >= pw) break;                                           \
        m3_store_quad(state_B + cidx * hd * ds + (long long)(p0 + 4 * tp + i) * ds \
                          + n0 + 4 * tn,                                       \
                      make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]), \
                      nw - 4 * tn, vec_s);                                     \
    }                                                                          \
}

DEFINE_M3_DQKV_STATE_TERMS_TILED(m3_dqkv_state_terms_tiled, float)
DEFINE_M3_DQKV_STATE_TERMS_TILED(m3_dqkv_state_terms_tiled_bf16, __nv_bfloat16)
DEFINE_M3_DQKV_STATE_TERMS_TILED(m3_dqkv_state_terms_tiled_f16, __half)
#undef DEFINE_M3_DQKV_STATE_TERMS_TILED
#undef M3S_LD
#undef M3S_TILE

// m3_dqkv_tiled: m3_dqkv for any headdim and d_state.
//
// m3_dqkv's outputs in its order from passes of exact-order products: every
// output is one thread's chain over its reduction index in m3_dqkv's order,
// from +0 with the same fused multiply-adds, and every epilogue is m3_dqkv's
// expression as its compiler forms it. A thread holds a 4 x 4 block of
// outputs, so each operand it reads from shared memory feeds four chains.
// Operands stream through two tiles of M3R_KT steps of the reduction by 64
// rows or columns, the step major, so a thread reads its four rows and its
// four columns of a step as two float4s. One block of 256 threads per
// (head, sequence, chunk); thread (ty, tx) = (tid / 16, tid % 16) holds rows
// 4 ty .. + 3 by columns 4 tx .. + 3 of a 64 x 64 block.
//
// The passes, in order:
// 1. The pair dots, K.Q over the state and V.dO over the head, a 4 x 4 block
//    of pairs a thread, then the triangles and the pairs' decays.
// 2. dQK_dot, a warp a step, each 32-channel segment through the warp
//    kernel's shuffle tree, the segments added in order.
// 3. Per block of 64 channels: K against the state gradient over the state
//    and each step's chain of its products with V; the intra-chunk dV over
//    the later steps of the step's sequence and dV; Q against the saved
//    state and each step's chain of its products with dO. A step's chain
//    over the channels is handed from thread to thread across the sixteen
//    that hold the step, in channel order.
// 4. Per block of 64 states: dK, the intra-chunk sum over the later steps
//    and the inter term over the head; then dQ over the earlier steps.
// 5. dADT from the triangles and the per-step lanes, its serial combine,
//    and the chunk's dD partial.
//
// Shared memory, in floats: nine per-step lanes of 64, the three pair
// triangles of CS (CS - 1) / 2, then two [M3R_KT][M3R_LD] operand tiles on
// 16 bytes: 43,904 bytes at a chunk of 64.
#define M3R_KT 32
#define M3R_LD 68

// Rows [0, nrows) of a row-major source, steps [k0, k0 + kw) of each, into
// a step-major tile, every value widened to f32: tile[k][r] =
// src[r * ld + k0 + k]; rows past nrows and steps past kw hold zeros. A
// thread loads four steps of a row at once (`vec` when the rows keep their
// alignment).
template <typename T_ACT>
__device__ __forceinline__ void m3r_stage_rows(
    float* __restrict__ tile, const T_ACT* __restrict__ src, long long ld, int nrows, int k0,
    int kw, bool vec
) {
    constexpr int KQ = M3R_KT >> 2;
    for (int i = threadIdx.x; i < 64 * KQ; i += blockDim.x) {
        const int r = i / KQ;
        const int c = i - r * KQ;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (r < nrows) v = m3_load_quad(src + r * ld + k0 + 4 * c, kw - 4 * c, vec);
        tile[(4 * c + 0) * M3R_LD + r] = v.x;
        tile[(4 * c + 1) * M3R_LD + r] = v.y;
        tile[(4 * c + 2) * M3R_LD + r] = v.z;
        tile[(4 * c + 3) * M3R_LD + r] = v.w;
    }
}

// Steps [0, kw) of a step-major source, columns [c0, c0 + ncols) of each,
// into a tile, every value widened to f32: tile[k][c] = src[k * ld + c0 + c];
// steps past kw and columns past ncols hold zeros.
template <typename T_ACT>
__device__ __forceinline__ void m3r_stage_steps(
    float* __restrict__ tile, const T_ACT* __restrict__ src, long long ld, int kw, int c0,
    int ncols, bool vec
) {
    for (int i = threadIdx.x; i < M3R_KT * 16; i += blockDim.x) {
        const int k = i >> 4;
        const int c = i & 15;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (k < kw) v = m3_load_quad(src + k * ld + c0 + 4 * c, ncols - 4 * c, vec);
        *reinterpret_cast<float4*>(tile + k * M3R_LD + 4 * c) = v;
    }
}

// acc[i][j] = fma(a[k][ra + i], b[k][cb + j], acc[i][j]) over the steps
// k < kw in order.
__device__ __forceinline__ void m3r_mma(
    float (&acc)[4][4], const float* __restrict__ a, const float* __restrict__ b, int kw, int ra,
    int cb
) {
    for (int k = 0; k < kw; k++) {
        const float4 av = *reinterpret_cast<const float4*>(a + k * M3R_LD + ra);
        const float4 bv = *reinterpret_cast<const float4*>(b + k * M3R_LD + cb);
        const float ar[4] = {av.x, av.y, av.z, av.w};
        const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(ar[i], br[j], acc[i][j]);
        }
    }
}

__device__ __forceinline__ void m3r_zero(float (&acc)[4][4]) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    }
}

// A step's chain over the columns of a block, carried in `cv` (a value for
// each of the thread's four steps), handed across the sixteen threads that
// hold the step: thread tx adds its columns' terms x[i][j] * y[i][j], j
// ascending, in turn, and every thread reads the chain back from the one
// whose turn it was. Columns past `width` and steps past `len` add nothing.
__device__ __forceinline__ void m3r_row_chain(
    float (&cv)[4], const float (&x)[4][4], const float (&y)[4][4], int ty, int tx, int len,
    int width
) {
    const int base = threadIdx.x & 16;
    for (int step = 0; step < 16; step++) {
        if (tx == step) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (4 * ty + i < len) {
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        if (4 * tx + j < width) cv[i] = __fmaf_rn(x[i][j], y[i][j], cv[i]);
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++) cv[i] = __shfl_sync(0xFFFFFFFFu, cv[i], base + step);
    }
}

#define DEFINE_M3_DQKV_TILED(NAME, T_ACT)                                      \
extern "C" __global__ __launch_bounds__(256, 2) void NAME(                     \
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
    int B, int T, int nh_total, int hd, int ds, int CS,                        \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    const int h = blockIdx.x;                                                  \
    const int b = y_first + blockIdx.y;                                        \
    const int chunk_idx = z_first + blockIdx.z;                                \
    const int tid = threadIdx.x;                                               \
    /* Uniform across the block, so every thread reaches the barriers. */     \
    if (h >= nh_total || b >= B || CS > 64) return;                            \
    const int n_chunks = (T + CS - 1) / CS;                                    \
    const int chunk_start = chunk_idx * CS;                                    \
    const int len = min(CS, T - chunk_start);                                  \
    const long long d_inner = (long long)nh_total * hd;                        \
    const long long kq_ld = (long long)nh_total * ds;                          \
    const long long cidx = ((long long)b * n_chunks + chunk_idx) * nh_total + h; \
    const long long row0 = (long long)b * T + chunk_start;                     \
    const float* ssm = SSM_States + cidx * hd * ds;                            \
    const float* dst = enter_state + cidx * hd * ds;                           \
    const T_ACT* k_rows = K_scaled + (row0 * nh_total + h) * ds;               \
    const T_ACT* q_rows = Q_rot + (row0 * nh_total + h) * ds;                  \
    const T_ACT* v_rows = V_in + row0 * d_inner + (long long)h * hd;           \
    const T_ACT* do_rows = dO + row0 * d_inner + (long long)h * hd;            \
    const float D_val = D_param[h];                                            \
    const float cs_sum = DA_CS_SUM[cidx];                                      \
    const int tri_n = CS * (CS - 1) / 2;                                       \
    const int ty = tid >> 4;                                                   \
    const int tx = tid & 15;                                                   \
    const bool vec_kq = (ds & 3) == 0 && M3_QUAD_ALIGNED(K_scaled, T_ACT)      \
                        && M3_QUAD_ALIGNED(Q_rot, T_ACT);                      \
    const bool vec_vo = (hd & 3) == 0 && M3_QUAD_ALIGNED(V_in, T_ACT)          \
                        && M3_QUAD_ALIGNED(dO, T_ACT);                         \
    const bool vec_st = (ds & 3) == 0 && M3_QUAD_ALIGNED(SSM_States, float)    \
                        && M3_QUAD_ALIGNED(enter_state, float);                \
    extern __shared__ __align__(16) float m3r_sm[];                            \
    float* da_sm = m3r_sm;                                                     \
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
    float* ta = m3r_sm + ((9 * 64 + 3 * tri_n + 3) & ~3);                      \
    float* tb = ta + M3R_KT * M3R_LD;                                          \
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
    /* The sequence spans of the thread's four steps in a packed row. */       \
    VlSpan span[4];                                                            \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 4; i++) {                                              \
        const int t = 4 * ty + i;                                              \
        span[i] = vl_span(seq_idx, b, T, chunk_start, len, t < len ? t : 0);   \
    }                                                                          \
                                                                               \
    /* 1. The pair dots: pair block (ba, bb), ba <= bb, of the upper half */   \
    /* of the chunk's 4 x 4 blocks of steps; a pair (a, c) of it lives */      \
    /* where a < c < len. */                                                   \
    {                                                                          \
        const int nb = (len + 3) >> 2;                                         \
        const int blocks = nb * (nb + 1) / 2;                                  \
        int ba = 0;                                                            \
        int bb = 0;                                                            \
        if (tid < blocks) {                                                    \
            int rem = tid;                                                     \
            while (rem >= nb - ba) {                                           \
                rem -= nb - ba;                                                \
                ba++;                                                          \
            }                                                                  \
            bb = ba + rem;                                                     \
        }                                                                      \
        float kq[4][4];                                                        \
        float vdo[4][4];                                                       \
        m3r_zero(kq);                                                          \
        m3r_zero(vdo);                                                         \
        for (int n0 = 0; n0 < ds; n0 += M3R_KT) {                              \
            const int nw = min(M3R_KT, ds - n0);                               \
            __syncthreads();                                                   \
            m3r_stage_rows(ta, k_rows, kq_ld, len, n0, nw, vec_kq);            \
            m3r_stage_rows(tb, q_rows, kq_ld, len, n0, nw, vec_kq);            \
            __syncthreads();                                                   \
            if (tid < blocks) m3r_mma(kq, ta, tb, nw, 4 * ba, 4 * bb);         \
        }                                                                      \
        for (int p0 = 0; p0 < hd; p0 += M3R_KT) {                              \
            const int pw = min(M3R_KT, hd - p0);                               \
            __syncthreads();                                                   \
            m3r_stage_rows(ta, v_rows, d_inner, len, p0, pw, vec_vo);          \
            m3r_stage_rows(tb, do_rows, d_inner, len, p0, pw, vec_vo);         \
            __syncthreads();                                                   \
            if (tid < blocks) m3r_mma(vdo, ta, tb, pw, 4 * ba, 4 * bb);        \
        }                                                                      \
        if (tid < blocks) {                                                    \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < 4; i++) {                                      \
                _Pragma("unroll")                                              \
                for (int j = 0; j < 4; j++) {                                  \
                    const int pa = 4 * ba + i;                                 \
                    const int pb = 4 * bb + j;                                 \
                    if (pa < pb && pb < len) {                                 \
                        const int idx = m3t_tri(pa, pb, CS);                   \
                        kq_tri[idx] = kq[i][j];                                \
                        vdo_tri[idx] = vdo[i][j];                              \
                        dec_tri[idx] = exp2f((da_sm[pb] - da_sm[pa]) * LOG2E); \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /* 2. dQK_dot. A warp takes a step, and lane l holds channel l of */       \
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
    __syncthreads();                                                           \
                                                                               \
    /* 3. Per block of 64 channels. */                                        \
    for (int p0 = 0; p0 < hd; p0 += 64) {                                      \
        const int pw = min(64, hd - p0);                                       \
        /* K against the state gradient over the state. */                     \
        float ke[4][4];                                                        \
        m3r_zero(ke);                                                          \
        for (int n0 = 0; n0 < ds; n0 += M3R_KT) {                              \
            const int nw = min(M3R_KT, ds - n0);                               \
            __syncthreads();                                                   \
            m3r_stage_rows(ta, k_rows, kq_ld, len, n0, nw, vec_kq);            \
            m3r_stage_rows(tb, dst + (long long)p0 * ds, (long long)ds, pw, n0, nw, vec_st); \
            __syncthreads();                                                   \
            m3r_mma(ke, ta, tb, nw, 4 * ty, 4 * tx);                           \
        }                                                                      \
        /* The thread's V and dO of the block. */                              \
        float vv[4][4];                                                        \
        float dov[4][4];                                                       \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 4; i++) {                                          \
            const int t = 4 * ty + i;                                          \
            const float4 v4 = t < len                                          \
                ? m3_load_quad(v_rows + t * d_inner + p0 + 4 * tx, pw - 4 * tx, vec_vo) \
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);                         \
            const float4 d4 = t < len                                          \
                ? m3_load_quad(do_rows + t * d_inner + p0 + 4 * tx, pw - 4 * tx, vec_vo) \
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);                         \
            vv[i][0] = v4.x; vv[i][1] = v4.y; vv[i][2] = v4.z; vv[i][3] = v4.w; \
            dov[i][0] = d4.x; dov[i][1] = d4.y; dov[i][2] = d4.z; dov[i][3] = d4.w; \
        }                                                                      \
        /* Each step's chain of K.dS times V over the channels, carried */     \
        /* from the block before. */                                           \
        float chain[4];                                                        \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 4; i++) {                                          \
            const int t = 4 * ty + i;                                          \
            chain[i] = (p0 == 0 || t >= len) ? 0.0f : dskv_sm[t];              \
        }                                                                      \
        m3r_row_chain(chain, ke, vv, ty, tx, len, pw);                         \
        __syncthreads();                                                       \
        if (tx == 0) {                                                         \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < 4; i++) {                                      \
                if (4 * ty + i < len) dskv_sm[4 * ty + i] = chain[i];          \
            }                                                                  \
        }                                                                      \
        /* The intra-chunk dV: each output's chain over the later steps of */  \
        /* its step's sequence, in order. */                                   \
        float dvi[4][4];                                                       \
        m3r_zero(dvi);                                                         \
        for (int s0 = 0; s0 < len; s0 += M3R_KT) {                             \
            const int sw = min(M3R_KT, len - s0);                              \
            __syncthreads();                                                   \
            m3r_stage_steps(tb, do_rows + s0 * d_inner, d_inner, sw, p0, pw, vec_vo); \
            __syncthreads();                                                   \
            for (int k = 0; k < sw; k++) {                                     \
                const int s = s0 + k;                                          \
                const float4 bv = *reinterpret_cast<const float4*>(tb + k * M3R_LD + 4 * tx); \
                const float br[4] = {bv.x, bv.y, bv.z, bv.w};                  \
                _Pragma("unroll")                                              \
                for (int i = 0; i < 4; i++) {                                  \
                    const int t = 4 * ty + i;                                  \
                    if (t < len && s > t && s < span[i].end) {                 \
                        const int ti = m3t_tri(t, s, CS);                      \
                        const float w = __fmul_rn(kq_tri[ti], dec_tri[ti]);    \
                        _Pragma("unroll")                                      \
                        for (int j = 0; j < 4; j++)                            \
                            dvi[i][j] = __fmaf_rn(w, br[j], dvi[i][j]);        \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 4; i++) {                                          \
            const int t = 4 * ty + i;                                          \
            if (t >= len) continue;                                            \
            const float skip_w = __fadd_rn(D_val, qk_sm[t]);                   \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 4; j++) {                                      \
                if (4 * tx + j >= pw) continue;                                \
                const float inter = span[i].to_end ? ke[i][j] : 0.0f;          \
                dV[(row0 + t) * d_inner + (long long)h * hd + p0 + 4 * tx + j] = \
                    __fmaf_rn(dov[i][j], skip_w,                               \
                              __fmaf_rn(inter, er_sm[t], dvi[i][j]));          \
            }                                                                  \
        }                                                                      \
        /* Q against the saved state over the state, then each step's */       \
        /* chain of it times dO over the channels. */                          \
        float qs[4][4];                                                        \
        m3r_zero(qs);                                                          \
        for (int n0 = 0; n0 < ds; n0 += M3R_KT) {                              \
            const int nw = min(M3R_KT, ds - n0);                               \
            __syncthreads();                                                   \
            m3r_stage_rows(ta, q_rows, kq_ld, len, n0, nw, vec_kq);            \
            m3r_stage_rows(tb, ssm + (long long)p0 * ds, (long long)ds, pw, n0, nw, vec_st); \
            __syncthreads();                                                   \
            m3r_mma(qs, ta, tb, nw, 4 * ty, 4 * tx);                           \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 4; i++) {                                          \
            const int t = 4 * ty + i;                                          \
            chain[i] = (p0 == 0 || t >= len) ? 0.0f : qsdo_sm[t];              \
        }                                                                      \
        m3r_row_chain(chain, qs, dov, ty, tx, len, pw);                        \
        __syncthreads();                                                       \
        if (tx == 0) {                                                         \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < 4; i++) {                                      \
                if (4 * ty + i < len) qsdo_sm[4 * ty + i] = chain[i];          \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* 4. Per block of 64 states, dK then dQ. */                              \
    for (int n0 = 0; n0 < ds; n0 += 64) {                                      \
        const int nw = min(64, ds - n0);                                       \
        float intra[4][4];                                                     \
        float inter[4][4];                                                     \
        m3r_zero(intra);                                                       \
        m3r_zero(inter);                                                       \
        /* dK: the intra sum over the later steps of the step's sequence. */   \
        for (int s0 = 0; s0 < len; s0 += M3R_KT) {                             \
            const int sw = min(M3R_KT, len - s0);                              \
            __syncthreads();                                                   \
            m3r_stage_steps(tb, q_rows + s0 * kq_ld, kq_ld, sw, n0, nw, vec_kq); \
            __syncthreads();                                                   \
            for (int k = 0; k < sw; k++) {                                     \
                const int s = s0 + k;                                          \
                const float4 bv = *reinterpret_cast<const float4*>(tb + k * M3R_LD + 4 * tx); \
                const float br[4] = {bv.x, bv.y, bv.z, bv.w};                  \
                _Pragma("unroll")                                              \
                for (int i = 0; i < 4; i++) {                                  \
                    const int t = 4 * ty + i;                                  \
                    if (t < len && s > t && s < span[i].end) {                 \
                        const int ti = m3t_tri(t, s, CS);                      \
                        const float w = __fmul_rn(vdo_tri[ti], dec_tri[ti]);   \
                        _Pragma("unroll")                                      \
                        for (int j = 0; j < 4; j++)                            \
                            intra[i][j] = __fmaf_rn(w, br[j], intra[i][j]);    \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* The inter term over the head: V against the state gradient. */      \
        for (int q0 = 0; q0 < hd; q0 += M3R_KT) {                              \
            const int qw = min(M3R_KT, hd - q0);                               \
            __syncthreads();                                                   \
            m3r_stage_rows(ta, v_rows, d_inner, len, q0, qw, vec_vo);          \
            m3r_stage_steps(tb, dst + (long long)q0 * ds, (long long)ds, qw, n0, nw, vec_st); \
            __syncthreads();                                                   \
            m3r_mma(inter, ta, tb, qw, 4 * ty, 4 * tx);                        \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 4; i++) {                                          \
            const int t = 4 * ty + i;                                          \
            if (t >= len) continue;                                            \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 4; j++) {                                      \
                if (4 * tx + j >= nw) continue;                                \
                const long long g = ((row0 + t) * nh_total + h) * ds + n0 + 4 * tx + j; \
                const float sel = span[i].to_end ? inter[i][j] : 0.0f;         \
                dK_mid[g] = __fmaf_rn(sel, er_sm[t], intra[i][j]);             \
            }                                                                  \
        }                                                                      \
        /* dQ: the intra sum over the earlier steps of the step's */           \
        /* sequence, then dO against the saved state over the head. */         \
        m3r_zero(intra);                                                       \
        m3r_zero(inter);                                                       \
        for (int s0 = 0; s0 < len; s0 += M3R_KT) {                             \
            const int sw = min(M3R_KT, len - s0);                              \
            __syncthreads();                                                   \
            m3r_stage_steps(tb, k_rows + s0 * kq_ld, kq_ld, sw, n0, nw, vec_kq); \
            __syncthreads();                                                   \
            for (int k = 0; k < sw; k++) {                                     \
                const int s = s0 + k;                                          \
                const float4 bv = *reinterpret_cast<const float4*>(tb + k * M3R_LD + 4 * tx); \
                const float br[4] = {bv.x, bv.y, bv.z, bv.w};                  \
                _Pragma("unroll")                                              \
                for (int i = 0; i < 4; i++) {                                  \
                    const int t = 4 * ty + i;                                  \
                    if (t < len && s < t && s >= span[i].first) {              \
                        const int ti = m3t_tri(s, t, CS);                      \
                        const float w = __fmul_rn(vdo_tri[ti], dec_tri[ti]);   \
                        _Pragma("unroll")                                      \
                        for (int j = 0; j < 4; j++)                            \
                            intra[i][j] = __fmaf_rn(w, br[j], intra[i][j]);    \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        for (int q0 = 0; q0 < hd; q0 += M3R_KT) {                              \
            const int qw = min(M3R_KT, hd - q0);                               \
            __syncthreads();                                                   \
            m3r_stage_rows(ta, do_rows, d_inner, len, q0, qw, vec_vo);         \
            m3r_stage_steps(tb, ssm + (long long)q0 * ds, (long long)ds, qw, n0, nw, vec_st); \
            __syncthreads();                                                   \
            m3r_mma(inter, ta, tb, qw, 4 * ty, 4 * tx);                        \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 4; i++) {                                          \
            const int t = 4 * ty + i;                                          \
            if (t >= len) continue;                                            \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 4; j++) {                                      \
                if (4 * tx + j >= nw) continue;                                \
                const long long g = ((row0 + t) * nh_total + h) * ds + n0 + 4 * tx + j; \
                const float sel = span[i].from_before ? inter[i][j] : 0.0f;    \
                dQ_mid[g] = __fmaf_rn(sel, ef_sm[t], intra[i][j]);             \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    /* 5. dADT, pairs below the diagonal, then above, then the entering */     \
    /* and exit states, as m3_dqkv orders them. */                             \
    for (int t = tid; t < len; t += 256) {                                     \
        float acc = 0.0f;                                                      \
        const VlSpan sp = vl_span(seq_idx, b, T, chunk_start, len, t);         \
        for (int i = sp.first; i < t; i++) {                                   \
            const int ti = m3t_tri(i, t, CS);                                  \
            float vdo = vdo_tri[ti];                                           \
            float decay = dec_tri[ti];                                         \
            float kq = kq_tri[ti];                                             \
            acc += vdo * decay * kq;                                           \
        }                                                                      \
        for (int j = t + 1; j < sp.end; j++) {                                 \
            const int ti = m3t_tri(t, j, CS);                                  \
            float vdo = vdo_tri[ti];                                           \
            float decay = dec_tri[ti];                                         \
            float kq = kq_tri[ti];                                             \
            acc -= vdo * decay * kq;                                           \
        }                                                                      \
        if (sp.from_before) acc += qsdo_sm[t] * ef_sm[t];                      \
        dmrev_sm[t] = acc;                                                     \
        dmvec_sm[t] = sp.to_end ? dskv_sm[t] * er_sm[t] : 0.0f;                \
    }                                                                          \
    __syncthreads();                                                           \
    if (tid == 0) {                                                            \
        float dM_scalar = dm_chain[cidx];                                      \
        dM_scalar *= exp2f(cs_sum * LOG2E);                                    \
        /* A packed row combines each of its sequences in the chunk on */      \
        /* its own, as m3_dqkv does. */                                        \
        const bool passes = vl_passes(seq_idx, b, T, chunk_start, len);        \
        for (int a = 0; a < len;) {                                            \
            const int e = vl_span(seq_idx, b, T, chunk_start, len, a).end;     \
            float total_rev = 0.0f;                                            \
            for (int t = a; t < e; t++) total_rev += dmrev_sm[t];              \
            if (passes) total_rev += dM_scalar;                                \
            float cumsum = 0.0f;                                               \
            for (int t = a; t < e; t++) {                                      \
                cumsum += dmvec_sm[t] - dmrev_sm[t];                           \
                const float out = dmrev_sm[t] + (total_rev + cumsum - dmvec_sm[t]); \
                dADT[(row0 + t) * nh_total + h] = out;                         \
            }                                                                  \
            a = e;                                                             \
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
#undef M3R_KT
#undef M3R_LD
