// Mamba-2 prefill on the GPU: the conv over the window and the chunked
// state space dual (chunks of 64 tokens).
//
// Per chunk c of Lc tokens, head h of group g, with dt_t, a_t = dt_t A and
// Acs_t = a_0 + ... + a_t (restarting every chunk):
//
//   CB[t][s]  = sum_n C_t[n] B_s[n]                      (per group, s <= t)
//   W[t][s]   = (CB[t][s] e^min(Acs_t - Acs_s, 0)) dt_s
//   xw[t][p]  = x_t[p] (e^min(Acs_last - Acs_t, 0) dt_t)
//   S[p][n]   = sum_t B_t[n] xw[t][p]                     (t ascending, from 0)
//   h_c       = e^Acs_last h_{c-1} + S
//   y_t[p]    = (sum_n C_t[n] h_{c-1}[p][n]) e^Acs_t + sum_{s<=t} W[t][s] x_s[p] + D x_t[p]
//
// Each value is one thread's chain in that order, spelled with explicit
// round-to-nearest intrinsics, so how the work is tiled never moves a bit,
// and the decode step (the same formulas at one token) continues a prefill
// of whole chunks exactly. Offsets are 64-bit.
//
// Composed after mamba2_ops.cu (which defines m2_exp, m2_softplus and
// m2_silu) in the Mamba-2 module.

#define M2_CHUNK 64

// The conv over the window and the conv state after it. One thread per
// (row, channel): v = bias + sum_k w[c][k] x[t - W + 1 + k][c], the taps
// before the window read from the carried state (`state[b][c][k]`, the last
// d_conv inputs, newest last); out = silu(v). Grid 1-D over B * T * conv_dim.
#define DEFINE_M2_CONV_PREFILL(NAME, TI, FROM_F)                               \
extern "C" __global__ void NAME(                                               \
    TI* __restrict__ out,                                                      \
    const float* __restrict__ state,                                           \
    const TI* __restrict__ proj, long long ld_proj, int xbc_col,              \
    const float* __restrict__ w, const float* __restrict__ bias,               \
    int batch, int seq, int conv_dim, int d_conv                               \
) {                                                                            \
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;            \
    long long total = (long long)batch * seq * conv_dim;                       \
    if (i >= total) return;                                                    \
    const int c = (int)(i % conv_dim);                                         \
    const long long row = i / conv_dim;                                        \
    const int t = (int)(row % seq);                                            \
    const int b = (int)(row / seq);                                            \
    float v = bias[c];                                                         \
    for (int k = 0; k < d_conv; k++) {                                         \
        int src = t - (d_conv - 1) + k;                                        \
        float xv = src >= 0                                                    \
            ? to_f(proj[((long long)b * seq + src) * ld_proj + xbc_col + c])   \
            : state[((long long)b * conv_dim + c) * d_conv + d_conv + src];    \
        v = __fmaf_rn(w[(long long)c * d_conv + k], xv, v);                    \
    }                                                                          \
    out[i] = FROM_F(m2_silu(v));                                               \
}

DEFINE_M2_CONV_PREFILL(m2_conv_prefill, float, from_f_f32)
DEFINE_M2_CONV_PREFILL(m2_conv_prefill_bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M2_CONV_PREFILL(m2_conv_prefill_f16, __half, from_f_f16)

// The decode's conv: one token, the window state `[b][c][k]` shifted and
// the new input appended, then the prefill's chain, bias first and taps
// oldest first, so a decode continues a prefill bit for bit. Grid 1-D over
// B * conv_dim.
#define DEFINE_M2_CONV_STEP(NAME, TI, FROM_F)                                  \
extern "C" __global__ void NAME(                                               \
    TI* __restrict__ out,                                                      \
    float* __restrict__ state,                                                 \
    const TI* __restrict__ proj, long long ld_proj, int xbc_col,              \
    const float* __restrict__ w, const float* __restrict__ bias,               \
    int batch, int conv_dim, int d_conv                                        \
) {                                                                            \
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;            \
    if (i >= (long long)batch * conv_dim) return;                              \
    const int c = (int)(i % conv_dim);                                         \
    const long long b = i / conv_dim;                                          \
    float* st = state + (b * conv_dim + c) * d_conv;                           \
    for (int k = 0; k + 1 < d_conv; k++) st[k] = st[k + 1];                    \
    st[d_conv - 1] = to_f(proj[b * ld_proj + xbc_col + c]);                    \
    float v = bias[c];                                                         \
    for (int k = 0; k < d_conv; k++)                                           \
        v = __fmaf_rn(w[(long long)c * d_conv + k], st[k], v);                 \
    out[b * conv_dim + c] = FROM_F(m2_silu(v));                                \
}

DEFINE_M2_CONV_STEP(m2_conv_step, float, from_f_f32)
DEFINE_M2_CONV_STEP(m2_conv_step_bf16, __nv_bfloat16, from_f_bf16)
DEFINE_M2_CONV_STEP(m2_conv_step_f16, __half, from_f_f16)

// The conv state after the window: the last d_conv raw inputs, from the
// window or, for a window shorter than that, the previous state. Writes a
// separate buffer (the prefill conv above still reads the old state).
// Grid 1-D over B * conv_dim.
#define DEFINE_M2_CONV_STATE(NAME, TI)                                         \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ new_state,                                             \
    const float* __restrict__ state,                                           \
    const TI* __restrict__ proj, long long ld_proj, int xbc_col,              \
    int batch, int seq, int conv_dim, int d_conv                               \
) {                                                                            \
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;            \
    if (i >= (long long)batch * conv_dim) return;                              \
    const int c = (int)(i % conv_dim);                                         \
    const int b = (int)(i / conv_dim);                                         \
    for (int k = 0; k < d_conv; k++) {                                         \
        int src = seq - d_conv + k;                                            \
        float xv = src >= 0                                                    \
            ? to_f(proj[((long long)b * seq + src) * ld_proj + xbc_col + c])   \
            : state[((long long)b * conv_dim + c) * d_conv + d_conv + src];    \
        new_state[((long long)b * conv_dim + c) * d_conv + k] = xv;            \
    }                                                                          \
}

DEFINE_M2_CONV_STATE(m2_conv_state, float)
DEFINE_M2_CONV_STATE(m2_conv_state_bf16, __nv_bfloat16)
DEFINE_M2_CONV_STATE(m2_conv_state_f16, __half)

// dt and the chunk cumsum of a = dt A: one thread per (b, chunk, head),
// the sum in token order. Outputs [B, nc, H, 64]; the tail of a short last
// chunk is written 0 and never read. Grid (B * nc, ceil(H / 32)), block 32.
#define DEFINE_M2_DT_CUMSUM(NAME, TI)                                          \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dt_out, float* __restrict__ acs_out,                   \
    const TI* __restrict__ proj, long long ld_proj, int dt_off,               \
    const float* __restrict__ dt_bias, const float* __restrict__ a_neg,        \
    int batch, int seq, int nh, int has_limit, float dt_min, float dt_max      \
) {                                                                            \
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;                            \
    const long long bc = blockIdx.x;                                           \
    const int h = blockIdx.y * blockDim.x + threadIdx.x;                       \
    if (bc >= (long long)batch * nc || h >= nh) return;                        \
    const int c = (int)(bc % nc);                                              \
    const long long b = bc / nc;                                               \
    const int t0 = c * M2_CHUNK;                                               \
    const int lc = min(M2_CHUNK, seq - t0);                                    \
    const long long o = (bc * nh + h) * M2_CHUNK;                              \
    float acc = 0.0f;                                                          \
    for (int t = 0; t < M2_CHUNK; t++) {                                       \
        if (t < lc) {                                                          \
            float raw = to_f(proj[(b * seq + t0 + t) * ld_proj + dt_off + h]); \
            float dt = m2_softplus(__fadd_rn(raw, dt_bias[h]));                \
            if (has_limit) dt = fminf(fmaxf(dt, dt_min), dt_max);              \
            acc = __fadd_rn(acc, __fmul_rn(dt, a_neg[h]));                     \
            dt_out[o + t] = dt;                                                \
            acs_out[o + t] = acc;                                              \
        } else {                                                               \
            dt_out[o + t] = 0.0f;                                              \
            acs_out[o + t] = 0.0f;                                             \
        }                                                                      \
    }                                                                          \
}

DEFINE_M2_DT_CUMSUM(m2_dt_cumsum, float)
DEFINE_M2_DT_CUMSUM(m2_dt_cumsum_bf16, __nv_bfloat16)
DEFINE_M2_DT_CUMSUM(m2_dt_cumsum_f16, __half)

// CB[t][s] for s <= t of every chunk and group: [B, nc, G, 64 * 65 / 2],
// pair (t, s) at t (t + 1) / 2 + s. One thread per pair, the chain over n
// ascending. Grid (B * nc, G), 1-D blocks striding the pairs.
#define DEFINE_M2_BMM(NAME, TI)                                                \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ cb,                                                    \
    const TI* __restrict__ xbc, long long ld_xbc, int b_col, int c_col,       \
    int batch, int seq, int ng, int ds                                         \
) {                                                                            \
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;                            \
    const long long bc = blockIdx.x;                                           \
    const int g = blockIdx.y;                                                  \
    if (bc >= (long long)batch * nc || g >= ng) return;                        \
    const int c = (int)(bc % nc);                                              \
    const long long b = bc / nc;                                               \
    const int t0 = c * M2_CHUNK;                                               \
    const int lc = min(M2_CHUNK, seq - t0);                                    \
    const int pairs = M2_CHUNK * (M2_CHUNK + 1) / 2;                           \
    const long long o = (bc * ng + g) * pairs;                                 \
    for (int q = threadIdx.x; q < pairs; q += blockDim.x) {                    \
        int t = (int)((sqrtf(8.0f * q + 1.0f) - 1.0f) * 0.5f);                 \
        while ((t + 1) * (t + 2) / 2 <= q) t++;                                \
        while (t * (t + 1) / 2 > q) t--;                                       \
        int s = q - t * (t + 1) / 2;                                           \
        float acc = 0.0f;                                                      \
        if (t < lc) {                                                          \
            const long long rt = (b * seq + t0 + t) * ld_xbc + c_col           \
                                 + (long long)g * ds;                          \
            const long long rs = (b * seq + t0 + s) * ld_xbc + b_col           \
                                 + (long long)g * ds;                          \
            for (int n = 0; n < ds; n++)                                       \
                acc = __fmaf_rn(to_f(xbc[rt + n]), to_f(xbc[rs + n]), acc);    \
        }                                                                      \
        cb[o + q] = acc;                                                       \
    }                                                                          \
}

DEFINE_M2_BMM(m2_bmm_chunk, float)
DEFINE_M2_BMM(m2_bmm_chunk_bf16, __nv_bfloat16)
DEFINE_M2_BMM(m2_bmm_chunk_f16, __half)

// Each chunk's own state S[p][n] = sum_t B_t[n] xw[t][p], from zero, t
// ascending: [B, nc, H, P, N]. One thread per (p, n).
// Grid (B * nc, H, ceil(P * N / blockDim.x)).
#define DEFINE_M2_CHUNK_STATE(NAME, TI)                                        \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ states,                                                \
    const TI* __restrict__ xbc, long long ld_xbc, int b_col,                  \
    const float* __restrict__ dt, const float* __restrict__ acs,               \
    int batch, int seq, int nh, int ng, int hd, int ds                         \
) {                                                                            \
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;                            \
    const long long bc = blockIdx.x;                                           \
    const int h = blockIdx.y;                                                  \
    const long long e = (long long)blockIdx.z * blockDim.x + threadIdx.x;      \
    if (bc >= (long long)batch * nc || h >= nh || e >= (long long)hd * ds)     \
        return;                                                                \
    const int p = (int)(e / ds);                                               \
    const int n = (int)(e % ds);                                               \
    const int g = h / (nh / ng);                                               \
    const int c = (int)(bc % nc);                                              \
    const long long b = bc / nc;                                               \
    const int t0 = c * M2_CHUNK;                                               \
    const int lc = min(M2_CHUNK, seq - t0);                                    \
    const long long sc = (bc * nh + h) * M2_CHUNK;                             \
    const float acs_last = acs[sc + lc - 1];                                   \
    float acc = 0.0f;                                                          \
    for (int t = 0; t < lc; t++) {                                             \
        const long long r = (b * seq + t0 + t) * ld_xbc;                       \
        float x = to_f(xbc[r + (long long)h * hd + p]);                        \
        float decay = m2_exp(fminf(__fsub_rn(acs_last, acs[sc + t]), 0.0f));   \
        float xw = __fmul_rn(x, __fmul_rn(decay, dt[sc + t]));                 \
        acc = __fmaf_rn(to_f(xbc[r + b_col + (long long)g * ds + n]), xw, acc);\
    }                                                                          \
    states[((bc * nh + h) * hd + p) * (long long)ds + n] = acc;                \
}

DEFINE_M2_CHUNK_STATE(m2_chunk_state, float)
DEFINE_M2_CHUNK_STATE(m2_chunk_state_bf16, __nv_bfloat16)
DEFINE_M2_CHUNK_STATE(m2_chunk_state_f16, __half)

// The states across chunks: the state entering chunk c replaces S_c in
// `states`, and the carried state `ssm` ([B, H, P, N]) ends as the state
// after the window: h_c = e^Acs_last h_{c-1} + S_c. One thread per
// (b, h, p, n) walking the chunks. Grid (B * H, ceil(P * N / blockDim.x)).
extern "C" __global__ void m2_state_passing(
    float* __restrict__ states,
    float* __restrict__ ssm,
    const float* __restrict__ acs,
    int batch, int seq, int nh, int hd, int ds
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bh = blockIdx.x;
    const long long e = (long long)blockIdx.y * blockDim.x + threadIdx.x;
    const long long pn = (long long)hd * ds;
    if (bh >= (long long)batch * nh || e >= pn) return;
    const long long b = bh / nh;
    const int h = (int)(bh % nh);
    float st = ssm[bh * pn + e];
    for (int c = 0; c < nc; c++) {
        const int lc = min(M2_CHUNK, seq - c * M2_CHUNK);
        const long long bc = b * nc + c;
        const long long idx = (bc * nh + h) * pn + e;
        const float s_c = states[idx];
        states[idx] = st;
        const float eacs = m2_exp(acs[(bc * nh + h) * M2_CHUNK + lc - 1]);
        st = __fmaf_rn(eacs, st, s_c);
    }
    ssm[bh * pn + e] = st;
}

// The outputs: y_t[p] = (sum_n C_t[n] h_prev[p][n]) e^Acs_t
// + sum_{s<=t} W[t][s] x_s[p] + D x_t[p], f32 [B * T, d_ssm]. One thread
// per (t, p). Grid (B * nc, H, ceil(64 * P / blockDim.x)).
#define DEFINE_M2_CHUNK_SCAN(NAME, TI)                                         \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ y,                                                     \
    const float* __restrict__ states,                                          \
    const float* __restrict__ cb,                                              \
    const TI* __restrict__ xbc, long long ld_xbc, int c_col,                  \
    const float* __restrict__ dt, const float* __restrict__ acs,               \
    const float* __restrict__ D, int d_per_channel,                            \
    int batch, int seq, int nh, int ng, int hd, int ds                         \
) {                                                                            \
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;                            \
    const long long bc = blockIdx.x;                                           \
    const int h = blockIdx.y;                                                  \
    const int e = blockIdx.z * blockDim.x + threadIdx.x;                       \
    if (bc >= (long long)batch * nc || h >= nh || e >= M2_CHUNK * hd) return;  \
    const int t = e / hd;                                                      \
    const int p = e % hd;                                                      \
    const int c = (int)(bc % nc);                                              \
    const long long b = bc / nc;                                               \
    const int t0 = c * M2_CHUNK;                                               \
    const int lc = min(M2_CHUNK, seq - t0);                                    \
    if (t >= lc) return;                                                       \
    const int g = h / (nh / ng);                                               \
    const long long sc = (bc * nh + h) * M2_CHUNK;                             \
    const long long rt = (b * seq + t0 + t) * ld_xbc;                          \
    const float* hprev = states + ((bc * nh + h) * hd + p) * (long long)ds;    \
    float acc = 0.0f;                                                          \
    for (int n = 0; n < ds; n++)                                               \
        acc = __fmaf_rn(to_f(xbc[rt + c_col + (long long)g * ds + n]),         \
                        hprev[n], acc);                                        \
    const float acs_t = acs[sc + t];                                           \
    acc = __fmul_rn(acc, m2_exp(acs_t));                                       \
    const long long cbo = (bc * ng + g) * (M2_CHUNK * (M2_CHUNK + 1) / 2)      \
                          + (long long)t * (t + 1) / 2;                        \
    for (int s = 0; s <= t; s++) {                                             \
        float wts = __fmul_rn(                                                 \
            __fmul_rn(cb[cbo + s],                                             \
                      m2_exp(fminf(__fsub_rn(acs_t, acs[sc + s]), 0.0f))),     \
            dt[sc + s]);                                                       \
        float xs = to_f(xbc[(b * seq + t0 + s) * ld_xbc                        \
                            + (long long)h * hd + p]);                         \
        acc = __fmaf_rn(wts, xs, acc);                                         \
    }                                                                          \
    const float xt = to_f(xbc[rt + (long long)h * hd + p]);                    \
    const float dv = d_per_channel ? D[(long long)h * hd + p] : D[h];          \
    acc = __fmaf_rn(xt, dv, acc);                                              \
    y[(b * seq + t0 + t) * ((long long)nh * hd) + (long long)h * hd + p] = acc;\
}

DEFINE_M2_CHUNK_SCAN(m2_chunk_scan, float)
DEFINE_M2_CHUNK_SCAN(m2_chunk_scan_bf16, __nv_bfloat16)
DEFINE_M2_CHUNK_SCAN(m2_chunk_scan_f16, __half)
