// Mamba-2 kernels: the decode step, the grouped gated RMSNorm and the
// mixer's inner MLP gate.
//
// Every value is one thread's chain in a fixed order, spelled with
// explicit round-to-nearest intrinsics: the module is PTX that the driver
// compiles per architecture, and a bare a*b+c may be fused differently by
// ptxas on another board. The decode evaluates the chunked prefill's
// formula at a chunk of one token, so a prefill of whole chunks followed by
// a decode step gives the bits of a prefill one token longer:
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
    if (threadIdx.x != 0) return;
    float raw = to_f(proj_row[dt_off + h]);
    float dt = m2_softplus(__fadd_rn(raw, dt_bias[h]));
    if (flags & M2_FLAG_DT_LIMIT) dt = fminf(fmaxf(dt, dt_min), dt_max);
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
// over n ascending. Returns C.h_prev e^acs + (C.B dt) x.
__device__ __forceinline__ float m2_row_step(
    float* trow, const float* b_sm, const float* c_sm,
    float eacs, float w_diag, float x_scale, float xp, int ds
) {
    const float xw = __fmul_rn(xp, x_scale);
    float acc = 0.0f;
    for (int n = 0; n < ds; n++) {
        float hp = trow[n];
        acc = __fmaf_rn(c_sm[n], hp, acc);
        trow[n] = __fmaf_rn(eacs, hp, __fmaf_rn(b_sm[n], xw, 0.0f));
    }
    acc = __fmul_rn(acc, eacs);
    return __fmaf_rn(w_diag, xp, acc);
}

// The recurrence of every head over a window, token by token: grid
// (B, H, ceil(P / blockDim.x)), block a whole number of warps. Dynamic
// shared memory: (2 * N + 4 + rows * (N + 1)) floats for B, C, the step
// scalars and the block's rows of the state, padded by one so a thread's
// row walk hits distinct banks. The state tile moves through shared memory
// with coalesced loads and stores once per window; each thread owns one
// channel's row. Token t of sequence b is row b * seq + t of xbc, proj and
// y. A decode step is the window of one token, so the sequential prefill
// gives the bits of that many decode steps.
//
// y is f32: the norm that follows reads it before any downcast. With
// M2_FLAG_GATE_IN_SCAN (no rmsnorm) the gate is applied here. With CKPT
// the state before tokens 0, stride, 2 stride, ... is also written to
// `ckpt` ([B, S, H, P, N], S = ceil(seq / stride)), what the training
// backward replays from.
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
    float* __restrict__ ckpt, int stride
) {
    extern __shared__ float m2s_sm[];
    float* b_sm = m2s_sm;
    float* c_sm = b_sm + ds;
    float* scal = c_sm + ds;
    float* tile = scal + 4;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int p0 = blockIdx.z * blockDim.x;
    const int tid = threadIdx.x;
    if (b >= batch || h >= nh || p0 >= hd) return;
    const int rows = min((int)blockDim.x, hd - p0);
    const int g = h / (nh / ng);
    const long long d_ssm = (long long)nh * hd;
    const long long base = (((long long)b * nh + h) * hd + p0) * ds;
    for (int r = 0; r < rows; r++) {
        for (int n = tid; n < ds; n += blockDim.x) {
            tile[r * (ds + 1) + n] = ssm[base + (long long)r * ds + n];
        }
    }
    const int nseg = CKPT ? (seq + stride - 1) / stride : 0;
    for (int t = 0; t < seq; t++) {
        const long long row = (long long)b * seq + t;
        const long long xrow = row * conv_dim;
        if (CKPT && t % stride == 0) {
            const long long cbase =
                ((((long long)b * nseg + t / stride) * nh + h) * hd + p0) * ds;
            for (int r = 0; r < rows; r++)
                for (int n = tid; n < ds; n += blockDim.x)
                    ckpt[cbase + (long long)r * ds + n] = tile[r * (ds + 1) + n];
        }
        for (int n = tid; n < ds; n += blockDim.x) {
            b_sm[n] = to_f(xbc[xrow + d_ssm + (long long)g * ds + n]);
            c_sm[n] = to_f(xbc[xrow + d_ssm + (long long)(ng + g) * ds + n]);
        }
        __syncthreads();
        m2_step_scalars(scal, b_sm, c_sm, proj + row * d_in_proj, dt_off, h, dt_bias,
                        a_neg, ds, flags, dt_min, dt_max);
        __syncthreads();
        if (tid < rows) {
            const int p = p0 + tid;
            const float xp = to_f(xbc[xrow + (long long)h * hd + p]);
            float acc = m2_row_step(tile + tid * (ds + 1), b_sm, c_sm, scal[0], scal[1],
                                    scal[2], xp, ds);
            float dv = (flags & M2_FLAG_D_PER_CHANNEL) ? D[(long long)h * hd + p] : D[h];
            acc = __fmaf_rn(xp, dv, acc);
            if (flags & M2_FLAG_GATE_IN_SCAN) {
                float z = to_f(proj[row * d_in_proj + z_off + (long long)h * hd + p]);
                acc = __fmul_rn(acc, m2_silu(z));
            }
            y[row * d_ssm + (long long)h * hd + p] = acc;
        }
        __syncthreads();
    }
    for (int r = 0; r < rows; r++) {
        for (int n = tid; n < ds; n += blockDim.x) {
            ssm[base + (long long)r * ds + n] = tile[r * (ds + 1) + n];
        }
    }
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
    int flags, float dt_min, float dt_max                                      \
) {                                                                            \
    m2_steps_body<TI, false>(y, ssm, xbc, proj, dt_bias, a_neg, D, batch, 1,   \
        nh, ng, hd, ds, conv_dim, d_in_proj, z_off, dt_off, flags, dt_min,     \
        dt_max, nullptr, 1);                                                   \
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
    int flags, float dt_min, float dt_max                                      \
) {                                                                            \
    m2_steps_body<TI, false>(y, ssm, xbc, proj, dt_bias, a_neg, D, batch, seq, \
        nh, ng, hd, ds, conv_dim, d_in_proj, z_off, dt_off, flags, dt_min,     \
        dt_max, nullptr, 1);                                                   \
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
    float* __restrict__ ckpt, int stride                                       \
) {                                                                            \
    m2_steps_body<TI, true>(y, ssm, xbc, proj, dt_bias, a_neg, D, batch, seq,  \
        nh, ng, hd, ds, conv_dim, d_in_proj, z_off, dt_off, flags, dt_min,     \
        dt_max, ckpt, stride);                                                 \
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
// The group's values are staged in dynamic shared memory (gs floats, plus
// one per head of the group). `out` is written at column out_col of rows
// ld_out wide.
#define DEFINE_M2_GATED_NORM(NAME, TI, TO, FROM_F)                             \
extern "C" __global__ void NAME(                                               \
    TO* __restrict__ out, long long ld_out, int out_col,                       \
    const float* __restrict__ y,                                               \
    const TI* __restrict__ proj, long long ld_proj, int z_off,                 \
    const float* __restrict__ w,                                               \
    long long rows, int d_ssm, int ng, int hd, float eps,                      \
    int norm_before_gate                                                       \
) {                                                                            \
    extern __shared__ float m2n_sm[];                                          \
    const long long r = blockIdx.x;                                            \
    const int g = blockIdx.y;                                                  \
    if (r >= rows || g >= ng) return;                                          \
    const int gs = d_ssm / ng;                                                 \
    const int heads = gs / hd;                                                 \
    float* u = m2n_sm;                                                         \
    float* head_sum = u + gs;                                                  \
    const long long c0 = (long long)g * gs;                                    \
    for (int i = threadIdx.x; i < gs; i += blockDim.x) {                       \
        float yv = y[r * d_ssm + c0 + i];                                      \
        float zv = to_f(proj[r * ld_proj + z_off + c0 + i]);                   \
        u[i] = norm_before_gate ? yv : __fmul_rn(yv, m2_silu(zv));             \
    }                                                                          \
    __syncthreads();                                                           \
    for (int hh = threadIdx.x; hh < heads; hh += blockDim.x) {                 \
        float s = 0.0f;                                                        \
        for (int p = 0; p < hd; p++) {                                         \
            float v = u[hh * hd + p];                                          \
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
        float v = __fmul_rn(__fmul_rn(u[i], rs), w[c0 + i]);                   \
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

// The gate without a norm (rmsnorm off) when the scan did not apply it:
// out = y silu(z), written at out_col. Grid 1-D over rows * d_ssm.
#define DEFINE_M2_GATE_ONLY(NAME, TI, TO, FROM_F)                              \
extern "C" __global__ void NAME(                                               \
    TO* __restrict__ out, long long ld_out, int out_col,                       \
    const float* __restrict__ y,                                               \
    const TI* __restrict__ proj, long long ld_proj, int z_off,                 \
    long long rows, int d_ssm                                                  \
) {                                                                            \
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;            \
    if (i >= rows * d_ssm) return;                                             \
    long long r = i / d_ssm;                                                   \
    long long c = i - r * d_ssm;                                               \
    float zv = to_f(proj[r * ld_proj + z_off + c]);                            \
    out[r * ld_out + out_col + c] = FROM_F(__fmul_rn(y[i], m2_silu(zv)));      \
}

DEFINE_M2_GATE_ONLY(m2_gate_only_fwd, float, float, from_f_f32)
DEFINE_M2_GATE_ONLY(m2_gate_only_fwd_bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16)
DEFINE_M2_GATE_ONLY(m2_gate_only_fwd_f16, __half, __half, from_f_f16)

// The mixer's inner MLP, the first d_mlp output columns:
// out[r, j] = silu(z0[r, j]) x0[r, j]. Grid 1-D over rows * d_mlp.
#define DEFINE_M2_MLP_GATE(NAME, TI, TO, FROM_F)                               \
extern "C" __global__ void NAME(                                               \
    TO* __restrict__ out, long long ld_out,                                    \
    const TI* __restrict__ proj, long long ld_proj,                            \
    long long rows, int d_mlp                                                  \
) {                                                                            \
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;            \
    if (i >= rows * d_mlp) return;                                             \
    long long r = i / d_mlp;                                                   \
    long long j = i - r * d_mlp;                                               \
    float z0 = to_f(proj[r * ld_proj + j]);                                    \
    float x0 = to_f(proj[r * ld_proj + d_mlp + j]);                            \
    out[r * ld_out + j] = FROM_F(__fmul_rn(m2_silu(z0), x0));                  \
}

DEFINE_M2_MLP_GATE(m2_mlp_gate_fwd, float, float, from_f_f32)
DEFINE_M2_MLP_GATE(m2_mlp_gate_fwd_bf16, __nv_bfloat16, __nv_bfloat16, from_f_bf16)
DEFINE_M2_MLP_GATE(m2_mlp_gate_fwd_f16, __half, __half, from_f_f16)
