// Mamba-2 kernels: the decode step, the grouped gated RMSNorm and the
// mixer's inner MLP gate.
//
// Every value is one thread's chain in a fixed order, spelled with
// explicit round-to-nearest intrinsics: the module is PTX that the driver
// compiles per architecture, and a bare a*b+c may be fused differently by
// ptxas on another board. The decode evaluates the chunked prefill's
// formula at a chunk of one token, so a chunked prefill of whole chunks
// followed by a decode step gives the bits of a chunked prefill one token
// longer (a sequential prefill and its decode are one recurrence already):
//
//   dt   = softplus(raw + dt_bias)            clamped only with a dt limit
//   acs  = 0 + dt * A                         (A = -exp(A_log), host side)
//   y    = (sum_n C[n] h_prev[p][n]) * e^acs + ((C.B) * e^0 * dt) x[p] + D x[p]
//   h[p][n] = e^acs h_prev[p][n] + B[n] (x[p] e^0 dt)
//
// which is, in real arithmetic, the reference's selective_state_update:
// h = h dA + dt B x, y = h C + D x.
//
// Composed after the typed prelude in the Mamba-2 module. Offsets are 64-bit.

#include "_typed_prelude.cuh"

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

__device__ __forceinline__ float m2_exp(float x) {
    return exp2f(__fmul_rn(x, LOG2E));
}

// softplus with the reference's passthrough above 20.
__device__ __forceinline__ float m2_softplus(float v) {
    return v > 20.0f ? v : log1pf(m2_exp(v));
}

// SiLU in the division form.
__device__ __forceinline__ float m2_silu(float z) {
    return __fdiv_rn(z, __fadd_rn(1.0f, m2_exp(-z)));
}

#define M2_FLAG_D_PER_CHANNEL 1
#define M2_FLAG_GATE_IN_SCAN 2
#define M2_FLAG_DT_LIMIT 4

// The step scalars of head h at one token, by thread 0 of the block:
// scal[0] = e^acs, scal[1] = (C.B) e^0 dt, scal[2] = e^0 dt.
template <typename TI>
__device__ __forceinline__ void m2_step_scalars(
    float* scal, const float* b_sm, const float* c_sm, const TI* proj_row,
    int dt_off, int h, const float* dt_bias, const float* a_neg, int ds,
    int flags, float dt_min, float dt_max
) {
    if (threadIdx.x != 0 || threadIdx.y != 0) return;
    float raw = to_f(proj_row[dt_off + h]);
    float dt = m2_softplus(__fadd_rn(raw, dt_bias[h]));
    // A NaN dt stays NaN, as the CPU's clamp leaves it; fminf and fmaxf
    // would return the bound and hide a corrupt step.
    if ((flags & M2_FLAG_DT_LIMIT) && dt == dt) dt = fminf(fmaxf(dt, dt_min), dt_max);
    float acs = __fadd_rn(0.0f, __fmul_rn(dt, a_neg[h]));
    float cb = 0.0f;
    for (int n = 0; n < ds; n++) cb = __fmaf_rn(c_sm[n], b_sm[n], cb);
    float e0 = m2_exp(fminf(__fsub_rn(acs, acs), 0.0f));
    scal[0] = m2_exp(acs);
    scal[1] = __fmul_rn(__fmul_rn(cb, e0), dt);
    scal[2] = __fmul_rn(e0, dt);
}

// One channel's row at one token: the readout from the previous state
// (scaled by e^acs) and the state update h = e^acs h + B (x e^0 dt), both
// over n ascending. Returns C.h_prev e^acs + (C.B dt) x. With B and C on
// 16 bytes the indices go four at a time, B and C one float4 each and the
// row's four values loaded before the chain takes them.
__device__ __forceinline__ float m2_row_step(
    float* __restrict__ trow, const float* __restrict__ b_sm, const float* __restrict__ c_sm,
    float eacs, float w_diag, float x_scale, float xp, int ds
) {
    const float xw = __fmul_rn(xp, x_scale);
    float acc = 0.0f;
    int n = 0;
    const unsigned long long bits =
        reinterpret_cast<unsigned long long>(b_sm) | reinterpret_cast<unsigned long long>(c_sm);
    if (((ds & 3) | (bits & 15)) == 0) {
        for (; n + 4 <= ds; n += 4) {
            const float4 b4 = *reinterpret_cast<const float4*>(b_sm + n);
            const float4 c4 = *reinterpret_cast<const float4*>(c_sm + n);
            const float bb[4] = {b4.x, b4.y, b4.z, b4.w};
            const float cc[4] = {c4.x, c4.y, c4.z, c4.w};
            float hv[4];
#pragma unroll
            for (int j = 0; j < 4; j++) hv[j] = trow[n + j];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                acc = __fmaf_rn(cc[j], hv[j], acc);
                trow[n + j] = __fmaf_rn(eacs, hv[j], __fmaf_rn(bb[j], xw, 0.0f));
            }
        }
    }
    for (; n < ds; n++) {
        float hp = trow[n];
        acc = __fmaf_rn(c_sm[n], hp, acc);
        trow[n] = __fmaf_rn(eacs, hp, __fmaf_rn(b_sm[n], xw, 0.0f));
    }
    acc = __fmul_rn(acc, eacs);
    return __fmaf_rn(w_diag, xp, acc);
}

// A block's [rows, N] state tile between dense global rows and shared rows
// padded to N + 1, every thread of the block taking float4 slices of the
// dense rows, so the tile moves in a few loads per thread.
__device__ __forceinline__ bool m2_tile_vec(const float* gl, int ds) {
    return (ds & 3) == 0 && (reinterpret_cast<unsigned long long>(gl) & 15) == 0;
}

__device__ __forceinline__ void m2_tile_load(
    float* __restrict__ sh, const float* __restrict__ gl, int rows, int ds
) {
    const int lin = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthr = blockDim.x * blockDim.y;
    const int total = rows * ds;
    if (m2_tile_vec(gl, ds)) {
#pragma unroll 4
        for (int i = lin * 4; i < total; i += nthr * 4) {
            const int r = i / ds, n = i - r * ds;
            const float4 v = *reinterpret_cast<const float4*>(gl + i);
            float* d = sh + r * (ds + 1) + n;
            d[0] = v.x;
            d[1] = v.y;
            d[2] = v.z;
            d[3] = v.w;
        }
    } else {
        for (int i = lin; i < total; i += nthr) {
            const int r = i / ds;
            sh[r * (ds + 1) + i - r * ds] = gl[i];
        }
    }
}

__device__ __forceinline__ void m2_tile_store(
    float* __restrict__ gl, const float* __restrict__ sh, int rows, int ds
) {
    const int lin = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthr = blockDim.x * blockDim.y;
    const int total = rows * ds;
    if (m2_tile_vec(gl, ds)) {
#pragma unroll 4
        for (int i = lin * 4; i < total; i += nthr * 4) {
            const int r = i / ds, n = i - r * ds;
            const float* d = sh + r * (ds + 1) + n;
            *reinterpret_cast<float4*>(gl + i) = make_float4(d[0], d[1], d[2], d[3]);
        }
    } else {
        for (int i = lin; i < total; i += nthr) {
            const int r = i / ds;
            gl[i] = sh[r * (ds + 1) + i - r * ds];
        }
    }
}

// The recurrence of every head over a window, token by token: grid
// (B, H, ceil(P / blockDim.x)), block (channels, helpers): thread x of the
// first row owns channel p0 + x, and the helper rows move the state tile
// and bring the next token's B and C into the other half of a double
// buffer while the owners run the current token, so a token costs one
// barrier. Each owner computes the step scalars itself with
// m2_step_scalars' expressions and runs their C.B chain in the same n loop
// as m2_row_step's two, so no thread waits on a scalar phase; every chain
// is the one m2_step_scalars and m2_row_step run. Dynamic shared memory:
// (4 * N + rows * (N + 1)) floats for the two halves of B and C and the
// block's rows of the state, padded by one so a thread's row walk hits
// distinct banks. The state tile moves through shared memory with
// coalesced loads and stores once per window; each thread owns one
// channel's row. Token t of sequence b is row b * seq + t of xbc, proj and
// y. A decode step is the window of one token, so the sequential prefill
// gives the bits of that many decode steps.
//
// y is f32: the norm that follows reads it before any downcast. With
// M2_FLAG_GATE_IN_SCAN (no rmsnorm) the gate is applied here. With CKPT
// the state before tokens 0, stride, 2 stride, ... is also written to
// `ckpt` ([B, S, H, P, N], S = ceil(seq / stride)), what the training
// backward replays from. In a packed row (`seq_idx`, see _varlen.cuh) a
// token that starts a sequence steps from the empty state; a checkpoint
// keeps the state before that reset.
template <typename TI, bool CKPT>
__device__ __forceinline__ void m2_steps_body(
    float* __restrict__ y,
    float* __restrict__ ssm,
    const TI* __restrict__ xbc,
    const TI* __restrict__ proj,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_neg,
    const float* __restrict__ D,
    int batch, int seq, int nh, int ng, int hd, int ds,
    int conv_dim, int d_in_proj, int z_off, int dt_off,
    int flags, float dt_min, float dt_max,
    const int* __restrict__ seq_idx,
    float* __restrict__ ckpt, int stride,
    int y_first, /* this run's first y block */
    int z_first /* this run's first z block */
) {
    extern __shared__ float m2s_sm[];
    float* bc = m2s_sm;            // [2][2 N]: B then C of a token, two halves
    float* tile = m2s_sm + 4 * ds;
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int p0 = (z_first + blockIdx.z) * blockDim.x;
    const int tid = threadIdx.x;
    if (b >= batch || h >= nh || p0 >= hd) return;
    const int rows = min((int)blockDim.x, hd - p0);
    const int lin = threadIdx.y * blockDim.x + tid;
    const int nthr = blockDim.x * blockDim.y;
    const bool owner = threadIdx.y == 0 && tid < rows;
    // The helper rows move the next token's B and C; a block of owners
    // alone moves them itself.
    const bool mover = blockDim.y == 1 || threadIdx.y > 0;
    const int mlin = blockDim.y == 1 ? lin : lin - (int)blockDim.x;
    const int mthr = blockDim.y == 1 ? nthr : nthr - (int)blockDim.x;
    const int g = h / (nh / ng);
    const long long d_ssm = (long long)nh * hd;
    const long long b_col = d_ssm + (long long)g * ds;
    const long long c_col = d_ssm + (long long)(ng + g) * ds;
    const long long base = (((long long)b * nh + h) * hd + p0) * ds;
    const int p = p0 + tid;
    m2_tile_load(tile, ssm + base, rows, ds);
    if (seq > 0) {
        const long long x0 = (long long)b * seq * conv_dim;
        for (int n = lin; n < ds; n += nthr) {
            bc[n] = to_f(xbc[x0 + b_col + n]);
            bc[ds + n] = to_f(xbc[x0 + c_col + n]);
        }
    }
    float dv = 0.0f, bias = 0.0f, an = 0.0f, raw = 0.0f, xp = 0.0f, z = 0.0f;
    const bool gate = (flags & M2_FLAG_GATE_IN_SCAN) != 0;
    if (owner) {
        dv = (flags & M2_FLAG_D_PER_CHANNEL) ? D[(long long)h * hd + p] : D[h];
        bias = dt_bias[h];
        an = a_neg[h];
        if (seq > 0) {
            const long long row = (long long)b * seq;
            raw = to_f(proj[row * d_in_proj + dt_off + h]);
            xp = to_f(xbc[row * conv_dim + (long long)h * hd + p]);
            if (gate) z = to_f(proj[row * d_in_proj + z_off + (long long)h * hd + p]);
        }
    }
    __syncthreads();
    const int nseg = CKPT ? (seq + stride - 1) / stride : 0;
    int id_prev = 0;
    for (int t = 0; t < seq; t++) {
        const long long row = (long long)b * seq + t;
        if (CKPT && t % stride == 0) {
            const long long cbase =
                ((((long long)b * nseg + t / stride) * nh + h) * hd + p0) * ds;
            m2_tile_store(ckpt + cbase, tile, rows, ds);
            __syncthreads();
        }
        const float* cur = bc + (t & 1) * 2 * ds;
        if (mover && t + 1 < seq) {
            float* nxt = bc + ((t + 1) & 1) * 2 * ds;
            const long long xn = (row + 1) * conv_dim;
            for (int n = mlin; n < ds; n += mthr) {
                nxt[n] = to_f(xbc[xn + b_col + n]);
                nxt[ds + n] = to_f(xbc[xn + c_col + n]);
            }
        }
        if (owner) {
            float raw_n = 0.0f, xp_n = 0.0f, z_n = 0.0f;
            if (t + 1 < seq) {
                raw_n = to_f(proj[(row + 1) * d_in_proj + dt_off + h]);
                xp_n = to_f(xbc[(row + 1) * conv_dim + (long long)h * hd + p]);
                if (gate) z_n = to_f(proj[(row + 1) * d_in_proj + z_off + (long long)h * hd + p]);
            }
            float dt = m2_softplus(__fadd_rn(raw, bias));
            if ((flags & M2_FLAG_DT_LIMIT) && dt == dt) dt = fminf(fmaxf(dt, dt_min), dt_max);
            const float acs = __fadd_rn(0.0f, __fmul_rn(dt, an));
            const float e0 = m2_exp(fminf(__fsub_rn(acs, acs), 0.0f));
            const float eacs = m2_exp(acs);
            const float xw = __fmul_rn(xp, __fmul_rn(e0, dt));
            const float* bs = cur;
            const float* cs = cur + ds;
            float* trow = tile + tid * (ds + 1);
            const int id_t = vl_id(seq_idx, b, seq, t);
            if (id_t != id_prev) {
                for (int k = 0; k < ds; k++) trow[k] = 0.0f;
            }
            id_prev = id_t;
            float cb = 0.0f, acc = 0.0f;
            int n = 0;
            // Four state indices at a time, B and C (the same for every
            // channel of the block) one float4 each, the row's four values
            // loaded before the chains take them: the row's own stores would
            // otherwise hold every next load back a round trip.
            if (((ds & 3) | (reinterpret_cast<unsigned long long>(cur) & 15)) == 0) {
                for (; n + 4 <= ds; n += 4) {
                    const float4 b4 = *reinterpret_cast<const float4*>(bs + n);
                    const float4 c4 = *reinterpret_cast<const float4*>(cs + n);
                    const float bb[4] = {b4.x, b4.y, b4.z, b4.w};
                    const float cc[4] = {c4.x, c4.y, c4.z, c4.w};
                    float hv[4];
#pragma unroll
                    for (int j = 0; j < 4; j++) hv[j] = trow[n + j];
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        cb = __fmaf_rn(cc[j], bb[j], cb);
                        acc = __fmaf_rn(cc[j], hv[j], acc);
                        trow[n + j] = __fmaf_rn(eacs, hv[j], __fmaf_rn(bb[j], xw, 0.0f));
                    }
                }
            }
            for (; n < ds; n++) {
                const float bn = bs[n], cn = cs[n], hp = trow[n];
                cb = __fmaf_rn(cn, bn, cb);
                acc = __fmaf_rn(cn, hp, acc);
                trow[n] = __fmaf_rn(eacs, hp, __fmaf_rn(bn, xw, 0.0f));
            }
            const float w_diag = __fmul_rn(__fmul_rn(cb, e0), dt);
            acc = __fmaf_rn(w_diag, xp, __fmul_rn(acc, eacs));
            acc = __fmaf_rn(xp, dv, acc);
            if (gate) acc = __fmul_rn(acc, m2_silu(z));
            y[row * d_ssm + (long long)h * hd + p] = acc;
            raw = raw_n;
            xp = xp_n;
            z = z_n;
        }
        __syncthreads();
    }
    m2_tile_store(ssm + base, tile, rows, ds);
}

// One decode step: the window of one token.
#define DEFINE_M2_STEP_FWD(NAME, TI)                                           \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ y,                                                     \
    float* __restrict__ ssm,                                                   \
    const TI* __restrict__ xbc,                                                \
    const TI* __restrict__ proj,                                               \
    const float* __restrict__ dt_bias,                                         \
    const float* __restrict__ a_neg,                                           \
    const float* __restrict__ D,                                               \
    int batch, int nh, int ng, int hd, int ds,                                 \
    int conv_dim, int d_in_proj, int z_off, int dt_off,                        \
    int flags, float dt_min, float dt_max,                                     \
    int y_first, /* this run's first y block */                                \
    int z_first /* this run's first z block */                                 \
) {                                                                            \
    m2_steps_body<TI, false>(y, ssm, xbc, proj, dt_bias, a_neg, D, batch, 1,   \
        nh, ng, hd, ds, conv_dim, d_in_proj, z_off, dt_off, flags, dt_min,     \
        dt_max, nullptr, nullptr, 1, y_first, z_first);                        \
}

DEFINE_M2_STEP_FWD(m2_step_fwd, float)
DEFINE_M2_STEP_FWD(m2_step_fwd_bf16, __nv_bfloat16)
DEFINE_M2_STEP_FWD(m2_step_fwd_f16, __half)

// The sequential prefill: the window of `seq` tokens.
#define DEFINE_M2_SEQ_FWD(NAME, TI)                                            \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ y,                                                     \
    float* __restrict__ ssm,                                                   \
    const TI* __restrict__ xbc,                                                \
    const TI* __restrict__ proj,                                               \
    const float* __restrict__ dt_bias,                                         \
    const float* __restrict__ a_neg,                                           \
    const float* __restrict__ D,                                               \
    int batch, int seq, int nh, int ng, int hd, int ds,                        \
    int conv_dim, int d_in_proj, int z_off, int dt_off,                        \
    int flags, float dt_min, float dt_max,                                     \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m2_steps_body<TI, false>(y, ssm, xbc, proj, dt_bias, a_neg, D, batch, seq, \
        nh, ng, hd, ds, conv_dim, d_in_proj, z_off, dt_off, flags, dt_min,     \
        dt_max, seq_idx, nullptr, 1, y_first, z_first);                        \
}

DEFINE_M2_SEQ_FWD(m2_seq_fwd, float)
DEFINE_M2_SEQ_FWD(m2_seq_fwd_bf16, __nv_bfloat16)
DEFINE_M2_SEQ_FWD(m2_seq_fwd_f16, __half)

// The training forward's sequential scan: the sequential prefill, also
// writing the replay checkpoints every `stride` tokens.
#define DEFINE_M2_SEQ_FWD_CKPT(NAME, TI)                                       \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ y,                                                     \
    float* __restrict__ ssm,                                                   \
    const TI* __restrict__ xbc,                                                \
    const TI* __restrict__ proj,                                               \
    const float* __restrict__ dt_bias,                                         \
    const float* __restrict__ a_neg,                                           \
    const float* __restrict__ D,                                               \
    int batch, int seq, int nh, int ng, int hd, int ds,                        \
    int conv_dim, int d_in_proj, int z_off, int dt_off,                        \
    int flags, float dt_min, float dt_max,                                     \
    float* __restrict__ ckpt, int stride,                                      \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m2_steps_body<TI, true>(y, ssm, xbc, proj, dt_bias, a_neg, D, batch, seq,  \
        nh, ng, hd, ds, conv_dim, d_in_proj, z_off, dt_off, flags, dt_min,     \
        dt_max, seq_idx, ckpt, stride, y_first, z_first);                      \
}

DEFINE_M2_SEQ_FWD_CKPT(m2_seq_fwd_ckpt, float)
DEFINE_M2_SEQ_FWD_CKPT(m2_seq_fwd_ckpt_bf16, __nv_bfloat16)
DEFINE_M2_SEQ_FWD_CKPT(m2_seq_fwd_ckpt_f16, __half)

// The grouped gated RMSNorm, gate first unless norm_before_gate:
//   u = y silu(z); rstd = 1 / sqrt(sum(u^2) / gs + eps) per group;
//   out = (u rstd) w.
// With norm_before_gate the sum runs over y and out = ((y rstd) w) silu(z).
// Grid (rows, ngroups), one block per row group. The sum is two levels,
// fixed: each head's channels ascending, then the group's heads ascending.
// The group's values are staged in dynamic shared memory, each head's
// row padded by one so the per-head sums (one thread a head) read distinct
// banks (gs + heads floats, plus one per head of the group). `out` is written at column out_col of rows
// ld_out wide.
#define DEFINE_M2_GATED_NORM(NAME, TI, TO, FROM_F)                             \
extern "C" __global__ void NAME(                                               \
    TO* __restrict__ out, long long ld_out, int out_col,                       \
    const float* __restrict__ y,                                               \
    const TI* __restrict__ proj, long long ld_proj, int z_off,                 \
    const float* __restrict__ w,                                               \
    long long rows, int d_ssm, int ng, int hd, float eps,                      \
    int norm_before_gate,                                                      \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    extern __shared__ float m2n_sm[];                                          \
    const long long r = blockIdx.x;                                            \
    const int g = y_first + blockIdx.y;                                        \
    if (r >= rows || g >= ng) return;                                          \
    const int gs = d_ssm / ng;                                                 \
    const int heads = gs / hd;                                                 \
    float* u = m2n_sm;                                                         \
    float* head_sum = u + gs + heads;                                          \
    const long long c0 = (long long)g * gs;                                    \
    for (int i = threadIdx.x; i < gs; i += blockDim.x) {                       \
        float yv = y[r * d_ssm + c0 + i];                                      \
        float zv = to_f(proj[r * ld_proj + z_off + c0 + i]);                   \
        u[i + i / hd] = norm_before_gate ? yv : __fmul_rn(yv, m2_silu(zv));    \
    }                                                                          \
    __syncthreads();                                                           \
    for (int hh = threadIdx.x; hh < heads; hh += blockDim.x) {                 \
        float s = 0.0f;                                                        \
        for (int p = 0; p < hd; p++) {                                         \
            float v = u[hh * (hd + 1) + p];                                    \
            s = __fmaf_rn(v, v, s);                                            \
        }                                                                      \
        head_sum[hh] = s;                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    __shared__ float rstd_sh;                                                  \
    if (threadIdx.x == 0) {                                                    \
        float tot = 0.0f;                                                      \
        for (int hh = 0; hh < heads; hh++) tot = __fadd_rn(tot, head_sum[hh]); \
        float rs = __fdiv_rn(1.0f,                                             \
            sqrtf(__fadd_rn(__fdiv_rn(tot, (float)gs), eps)));                 \
        rstd_sh = isfinite(rs) ? rs : 0.0f;                                    \
    }                                                                          \
    __syncthreads();                                                           \
    const float rs = rstd_sh;                                                  \
    for (int i = threadIdx.x; i < gs; i += blockDim.x) {                       \
        float v = __fmul_rn(__fmul_rn(u[i + i / hd], rs), w[c0 + i]);          \
        if (norm_before_gate) {                                                \
            float zv = to_f(proj[r * ld_proj + z_off + c0 + i]);               \
            v = __fmul_rn(v, m2_silu(zv));                                     \
        }                                                                      \
        out[r * ld_out + out_col + c0 + i] = FROM_F(v);                        \
    }                                                                          \
}

DEFINE_M2_GATED_NORM(m2_gated_norm_fwd, float, float, from_f_f32)
DEFINE_M2_GATED_NORM(m2_gated_norm_fwd_bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16)
DEFINE_M2_GATED_NORM(m2_gated_norm_fwd_f16, __half, __half, from_f_f16)

// The gated norm of a group too wide to stage: the same sums in the same
// order, each value read from global memory where the staged kernel reads
// shared memory, and recomputed with its expressions. A pass takes one head
// a thread into `head_sum` (blockDim floats), and thread 0 adds the pass's
// sums to the group's total heads ascending, so the total, rstd and every
// output are the staged kernel's bits. Grid (rows, ngroups), dynamic
// shared memory one float per thread.
#define DEFINE_M2_GATED_NORM_STREAM(NAME, TI, TO, FROM_F)                      \
extern "C" __global__ void NAME(                                               \
    TO* __restrict__ out, long long ld_out, int out_col,                       \
    const float* __restrict__ y,                                               \
    const TI* __restrict__ proj, long long ld_proj, int z_off,                 \
    const float* __restrict__ w,                                               \
    long long rows, int d_ssm, int ng, int hd, float eps,                      \
    int norm_before_gate,                                                      \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    extern __shared__ float head_sum[];                                        \
    const long long r = blockIdx.x;                                            \
    const int g = y_first + blockIdx.y;                                        \
    if (r >= rows || g >= ng) return;                                          \
    const int gs = d_ssm / ng;                                                 \
    const int heads = gs / hd;                                                 \
    const long long c0 = (long long)g * gs;                                    \
    __shared__ float tot_sh;                                                   \
    if (threadIdx.x == 0) tot_sh = 0.0f;                                       \
    for (int h0 = 0; h0 < heads; h0 += blockDim.x) {                           \
        const int hh = h0 + threadIdx.x;                                       \
        if (hh < heads) {                                                      \
            float s = 0.0f;                                                    \
            for (int p = 0; p < hd; p++) {                                     \
                const long long i = c0 + (long long)hh * hd + p;               \
                float yv = y[r * d_ssm + i];                                   \
                float zv = to_f(proj[r * ld_proj + z_off + i]);                \
                float v = norm_before_gate ? yv : __fmul_rn(yv, m2_silu(zv));  \
                s = __fmaf_rn(v, v, s);                                        \
            }                                                                  \
            head_sum[threadIdx.x] = s;                                         \
        }                                                                      \
        __syncthreads();                                                       \
        if (threadIdx.x == 0) {                                                \
            float tot = tot_sh;                                                \
            const int n = min((int)blockDim.x, heads - h0);                    \
            for (int k = 0; k < n; k++) tot = __fadd_rn(tot, head_sum[k]);     \
            tot_sh = tot;                                                      \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    __shared__ float rstd_sh;                                                  \
    if (threadIdx.x == 0) {                                                    \
        float rs = __fdiv_rn(1.0f,                                             \
            sqrtf(__fadd_rn(__fdiv_rn(tot_sh, (float)gs), eps)));              \
        rstd_sh = isfinite(rs) ? rs : 0.0f;                                    \
    }                                                                          \
    __syncthreads();                                                           \
    const float rs = rstd_sh;                                                  \
    for (int i = threadIdx.x; i < gs; i += blockDim.x) {                       \
        float yv = y[r * d_ssm + c0 + i];                                      \
        float zv = to_f(proj[r * ld_proj + z_off + c0 + i]);                   \
        float u = norm_before_gate ? yv : __fmul_rn(yv, m2_silu(zv));          \
        float v = __fmul_rn(__fmul_rn(u, rs), w[c0 + i]);                      \
        if (norm_before_gate) v = __fmul_rn(v, m2_silu(zv));                   \
        out[r * ld_out + out_col + c0 + i] = FROM_F(v);                        \
    }                                                                          \
}

DEFINE_M2_GATED_NORM_STREAM(m2_gated_norm_fwd_stream, float, float, from_f_f32)
DEFINE_M2_GATED_NORM_STREAM(m2_gated_norm_fwd_stream_bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16)
DEFINE_M2_GATED_NORM_STREAM(m2_gated_norm_fwd_stream_f16, __half, __half, from_f_f16)

// The gate without a norm (rmsnorm off) when the scan did not apply it:
// out = y silu(z), written at out_col. A block's y threads take rows,
// strided by the grid, its x threads the channels: no index is divided.
#define DEFINE_M2_GATE_ONLY(NAME, TI, TO, FROM_F)                              \
extern "C" __global__ void NAME(                                               \
    TO* __restrict__ out, long long ld_out, int out_col,                       \
    const float* __restrict__ y,                                               \
    const TI* __restrict__ proj, long long ld_proj, int z_off,                 \
    long long rows, int d_ssm                                                  \
) {                                                                            \
    for (long long r = (long long)blockIdx.x * blockDim.y + threadIdx.y;       \
         r < rows; r += (long long)gridDim.x * blockDim.y) {                   \
        for (int c = threadIdx.x; c < d_ssm; c += blockDim.x) {                \
            float zv = to_f(proj[r * ld_proj + z_off + c]);                    \
            out[r * ld_out + out_col + c] =                                    \
                FROM_F(__fmul_rn(y[r * d_ssm + c], m2_silu(zv)));              \
        }                                                                      \
    }                                                                          \
}

DEFINE_M2_GATE_ONLY(m2_gate_only_fwd, float, float, from_f_f32)
DEFINE_M2_GATE_ONLY(m2_gate_only_fwd_bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16)
DEFINE_M2_GATE_ONLY(m2_gate_only_fwd_f16, __half, __half, from_f_f16)

// The mixer's inner MLP, the first d_mlp output columns:
// out[r, j] = silu(z0[r, j]) x0[r, j]. A block's y threads take rows,
// strided by the grid, its x threads the columns.
#define DEFINE_M2_MLP_GATE(NAME, TI, TO, FROM_F)                               \
extern "C" __global__ void NAME(                                               \
    TO* __restrict__ out, long long ld_out,                                    \
    const TI* __restrict__ proj, long long ld_proj,                            \
    long long rows, int d_mlp                                                  \
) {                                                                            \
    for (long long r = (long long)blockIdx.x * blockDim.y + threadIdx.y;       \
         r < rows; r += (long long)gridDim.x * blockDim.y) {                   \
        for (int j = threadIdx.x; j < d_mlp; j += blockDim.x) {                \
            float z0 = to_f(proj[r * ld_proj + j]);                            \
            float x0 = to_f(proj[r * ld_proj + d_mlp + j]);                    \
            out[r * ld_out + j] = FROM_F(__fmul_rn(m2_silu(z0), x0));          \
        }                                                                      \
    }                                                                          \
}

DEFINE_M2_MLP_GATE(m2_mlp_gate_fwd, float, float, from_f_f32)
DEFINE_M2_MLP_GATE(m2_mlp_gate_fwd_bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16)
DEFINE_M2_MLP_GATE(m2_mlp_gate_fwd_f16, __half, __half, from_f_f16)
