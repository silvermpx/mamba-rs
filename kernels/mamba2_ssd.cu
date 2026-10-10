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

// dt and the chunk cumsum of a = dt A, one block per (b, chunk) and four
// heads: a head's 64 threads compute the dt of its tokens at once, then one
// of them runs the sum in token order over shared memory. Outputs
// [B, nc, H, 64]; the tail of a short last chunk is written 0 and never
// read. Grid (B * nc, ceil(H / 4)), block (64, 4).
#define DEFINE_M2_DT_CUMSUM(NAME, TI)                                          \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dt_out, float* __restrict__ acs_out,                   \
    const TI* __restrict__ proj, long long ld_proj, int dt_off,               \
    const float* __restrict__ dt_bias, const float* __restrict__ a_neg,        \
    int batch, int seq, int nh, int has_limit, float dt_min, float dt_max      \
) {                                                                            \
    __shared__ float dts[4][M2_CHUNK];                                         \
    __shared__ float acss[4][M2_CHUNK];                                        \
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;                            \
    const long long bc = blockIdx.x;                                           \
    if (bc >= (long long)batch * nc) return;                                   \
    const int hl = threadIdx.y;                                                \
    const int h = blockIdx.y * 4 + hl;                                         \
    const int t = threadIdx.x;                                                 \
    const int c = (int)(bc % nc);                                              \
    const long long b = bc / nc;                                               \
    const int t0 = c * M2_CHUNK;                                               \
    const int lc = min(M2_CHUNK, seq - t0);                                    \
    const bool live = h < nh;                                                  \
    float dt = 0.0f;                                                           \
    if (live && t < lc) {                                                      \
        float raw = to_f(proj[(b * seq + t0 + t) * ld_proj + dt_off + h]);     \
        dt = m2_softplus(__fadd_rn(raw, dt_bias[h]));                          \
        /* A NaN dt stays NaN, as the CPU's clamp leaves it. */                \
        if (has_limit && dt == dt) dt = fminf(fmaxf(dt, dt_min), dt_max);     \
    }                                                                          \
    dts[hl][t] = dt;                                                           \
    __syncthreads();                                                           \
    if (live && t == 0) {                                                      \
        const float an = a_neg[h];                                             \
        float acc = 0.0f;                                                      \
        for (int s = 0; s < lc; s++) {                                         \
            acc = __fadd_rn(acc, __fmul_rn(dts[hl][s], an));                   \
            acss[hl][s] = acc;                                                 \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    if (!live) return;                                                         \
    const long long o = (bc * nh + h) * M2_CHUNK;                              \
    dt_out[o + t] = t < lc ? dt : 0.0f;                                        \
    acs_out[o + t] = t < lc ? acss[hl][t] : 0.0f;                              \
}

DEFINE_M2_DT_CUMSUM(m2_dt_cumsum, float)
DEFINE_M2_DT_CUMSUM(m2_dt_cumsum_bf16, __nv_bfloat16)
DEFINE_M2_DT_CUMSUM(m2_dt_cumsum_f16, __half)

// CB[t][s] for s <= t of every chunk and group: [B, nc, G, 64 * 65 / 2],
// pair (t, s) at t (t + 1) / 2 + s, each the chain over n ascending from 0;
// past the chunk's tokens 0. One block per (chunk, group): the C and B
// rows staged transposed [N][68]; thread (tg, sg) of the lower triangle of
// 4 x 4 token tiles runs its 16 chains, the rows staged up to 128 state
// indices a pass. Grid (B * nc, G), block 256; dynamic shared memory
// 2 min(N, 128) 68 floats.
#define DEFINE_M2_BMM(NAME, TI)                                                \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ cb,                                                    \
    const TI* __restrict__ xbc, long long ld_xbc, int b_col, int c_col,       \
    int batch, int seq, int ng, int ds                                         \
) {                                                                            \
    extern __shared__ float m2bmm_sm[];                                        \
    constexpr int LS = M2_CHUNK + 4;                                           \
    const int npass = min(ds, 128);                                            \
    float* ct = m2bmm_sm;                                                      \
    float* bt = ct + npass * LS;                                               \
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;                            \
    const long long bc = blockIdx.x;                                           \
    const int g = blockIdx.y;                                                  \
    if (bc >= (long long)batch * nc || g >= ng) return;                        \
    const int c = (int)(bc % nc);                                              \
    const long long b = bc / nc;                                               \
    const int t0 = c * M2_CHUNK;                                               \
    const int lc = min(M2_CHUNK, seq - t0);                                    \
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;                \
    const int nw = blockDim.x >> 5;                                            \
    const int tg = threadIdx.x >> 4, sg = threadIdx.x & 15;                    \
    float acc[4][4];                                                           \
    _Pragma("unroll") for (int i = 0; i < 4; i++) {                            \
        _Pragma("unroll") for (int j = 0; j < 4; j++) { acc[i][j] = 0.0f; }    \
    }                                                                          \
    for (int m0 = 0; m0 < ds; m0 += npass) {                                   \
    const int mn = min(npass, ds - m0);                                        \
    if (m0 > 0) __syncthreads();                                               \
    for (int r = warp; r < M2_CHUNK; r += nw) {                                \
        const long long row = (b * seq + t0 + r) * ld_xbc + (long long)g * ds + m0; \
        for (int n = lane; n < mn; n += 32) {                                  \
            ct[n * LS + r] = r < lc ? to_f(xbc[row + c_col + n]) : 0.0f;       \
            bt[n * LS + r] = r < lc ? to_f(xbc[row + b_col + n]) : 0.0f;       \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    if (sg <= tg) {                                                            \
    for (int n = 0; n < mn; n++) {                                             \
        const float4 c4 = *reinterpret_cast<const float4*>(ct + n * LS + 4 * tg); \
        const float4 b4 = *reinterpret_cast<const float4*>(bt + n * LS + 4 * sg); \
        const float cv[4] = {c4.x, c4.y, c4.z, c4.w};                          \
        const float bv[4] = {b4.x, b4.y, b4.z, b4.w};                          \
        _Pragma("unroll") for (int i = 0; i < 4; i++) {                        \
            _Pragma("unroll") for (int j = 0; j < 4; j++) {                    \
                acc[i][j] = __fmaf_rn(cv[i], bv[j], acc[i][j]);                \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    }                                                                          \
    }                                                                          \
    if (sg > tg) return;                                                       \
    const int pairs = M2_CHUNK * (M2_CHUNK + 1) / 2;                           \
    const long long o = (bc * ng + g) * pairs;                                 \
    _Pragma("unroll") for (int i = 0; i < 4; i++) {                            \
        const int t = 4 * tg + i;                                              \
        _Pragma("unroll") for (int j = 0; j < 4; j++) {                        \
            const int s_ = 4 * sg + j;                                         \
            if (s_ <= t) cb[o + t * (t + 1) / 2 + s_] = t < lc ? acc[i][j] : 0.0f; \
        }                                                                      \
    }                                                                          \
}

DEFINE_M2_BMM(m2_bmm_chunk, float)
DEFINE_M2_BMM(m2_bmm_chunk_bf16, __nv_bfloat16)
DEFINE_M2_BMM(m2_bmm_chunk_f16, __half)

// Stages `count` values into shared memory through load(i) and put(i, v),
// four indices a thread at a time with every load of the four issued before
// any store, so a thread waits one round trip a batch rather than one a
// value. The values are load's, unchanged.
template <class LF, class PF>
__device__ __forceinline__ void m2_stage4(int count, LF load, PF put) {
    const int nthr = blockDim.x;
    for (int i0 = threadIdx.x; i0 < count; i0 += 4 * nthr) {
        float v[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = i0 + k * nthr;
            v[k] = i < count ? load(i) : 0.0f;
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = i0 + k * nthr;
            if (i < count) put(i, v[k]);
        }
    }
}

// Each chunk's own state S[p][n] = sum_t B_t[n] xw[t][p], from zero, t
// ascending: [B, nc, H, P, N], with xw = x (e^min(Acs_last - Acs_t, 0) dt_t).
// Register tiles: a warp takes 4 channels p and a lane the state indices
// lane + 32 j of the block's tile (JC of them); the per-token weights, xw
// [64][P4] and the B rows [64][32 JC] are staged once per block.
// A warp walks its groups of 4 channels in turn. Grid (B * nc, H,
// ceil(N / (32 JC))), JC from N (1, 2 or 4), block up to 256 threads;
// dynamic shared memory 64 (P4 + 32 JC + 1) floats.
template <typename TI, int JC>
__device__ __forceinline__ void m2_chunk_state_body(
    float* __restrict__ states,
    const TI* __restrict__ xbc, long long ld_xbc, int b_col,
    const float* __restrict__ dt, const float* __restrict__ acs,
    int batch, int seq, int nh, int ng, int hd, int ds
) {
    extern __shared__ float4 m2cs_sm4[];
    constexpr int NT = 32 * JC;
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int n0 = blockIdx.z * NT;
    if (bc >= (long long)batch * nc || h >= nh) return;
    const int g = h / (nh / ng);
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    const long long sc = (bc * nh + h) * M2_CHUNK;
    const int p4 = (hd + 3) & ~3;
    float* xs = reinterpret_cast<float*>(m2cs_sm4);  // [64][p4]
    float* bs = xs + M2_CHUNK * p4;                   // [64][NT]
    float* ws = bs + M2_CHUNK * NT;                   // [64]
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    const float acs_last = acs[sc + lc - 1];
    for (int t = tid; t < lc; t += blockDim.x)
        ws[t] = __fmul_rn(m2_exp(fminf(__fsub_rn(acs_last, acs[sc + t]), 0.0f)), dt[sc + t]);
    __syncthreads();
    m2_stage4(lc * p4,
              [&](int i) {
                  const int t = i / p4, p = i - t * p4;
                  const long long r = (b * seq + t0 + t) * ld_xbc;
                  return p < hd ? __fmul_rn(to_f(xbc[r + (long long)h * hd + p]), ws[t]) : 0.0f;
              },
              [&](int i, float v) { xs[i] = v; });
    m2_stage4(lc * NT,
              [&](int i) {
                  const int t = i / NT, ni = i - t * NT;
                  const int n = n0 + ni;
                  const long long r = (b * seq + t0 + t) * ld_xbc;
                  return n < ds ? to_f(xbc[r + b_col + (long long)g * ds + n]) : 0.0f;
              },
              [&](int i, float v) { bs[i] = v; });
    __syncthreads();
    for (int p0 = 4 * warp; p0 < hd; p0 += 4 * nw) {
        float acc[4][JC];
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < JC; j++) acc[i][j] = 0.0f;
        }
        for (int t = 0; t < lc; t++) {
            const float4 x4 = *reinterpret_cast<const float4*>(xs + t * p4 + p0);
            const float xv[4] = {x4.x, x4.y, x4.z, x4.w};
            float bv[JC];
#pragma unroll
            for (int j = 0; j < JC; j++) bv[j] = bs[t * NT + lane + 32 * j];
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < JC; j++) acc[i][j] = __fmaf_rn(bv[j], xv[i], acc[i][j]);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int p = p0 + i;
            if (p >= hd) continue;
#pragma unroll
            for (int j = 0; j < JC; j++) {
                const int n = n0 + lane + 32 * j;
                if (n < ds) states[((bc * nh + h) * hd + p) * (long long)ds + n] = acc[i][j];
            }
        }
    }
}

#define DEFINE_M2_CHUNK_STATE(NAME, TI)                                        \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ states,                                                \
    const TI* __restrict__ xbc, long long ld_xbc, int b_col,                  \
    const float* __restrict__ dt, const float* __restrict__ acs,               \
    int batch, int seq, int nh, int ng, int hd, int ds                         \
) {                                                                            \
    if (ds <= 32)                                                              \
        m2_chunk_state_body<TI, 1>(states, xbc, ld_xbc, b_col, dt, acs, batch, \
            seq, nh, ng, hd, ds);                                              \
    else if (ds <= 64)                                                         \
        m2_chunk_state_body<TI, 2>(states, xbc, ld_xbc, b_col, dt, acs, batch, \
            seq, nh, ng, hd, ds);                                              \
    else                                                                       \
        m2_chunk_state_body<TI, 4>(states, xbc, ld_xbc, b_col, dt, acs, batch, \
            seq, nh, ng, hd, ds);                                              \
}

DEFINE_M2_CHUNK_STATE(m2_chunk_state, float)
DEFINE_M2_CHUNK_STATE(m2_chunk_state_bf16, __nv_bfloat16)
DEFINE_M2_CHUNK_STATE(m2_chunk_state_f16, __half)

// The states across chunks: the state entering chunk c replaces S_c in
// `states`, and the carried state `ssm` ([B, H, P, N]) ends as the state
// after the window: h_c = e^Acs_last h_{c-1} + S_c. One thread per
// (b, h, p, n) walking the chunks, the chunks' decays computed once per
// block into shared memory (M2_PASS_WINDOW chunks at a time) and the next
// four chunks' S loaded ahead, so a step waits on one fma rather than on
// memory. Grid (B * H, ceil(P * N / blockDim.x)).
#define M2_PASS_WINDOW 1024
extern "C" __global__ void m2_state_passing(
    float* __restrict__ states,
    float* __restrict__ ssm,
    const float* __restrict__ acs,
    int batch, int seq, int nh, int hd, int ds
) {
    __shared__ float decay[M2_PASS_WINDOW];
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bh = blockIdx.x;
    const long long e = (long long)blockIdx.y * blockDim.x + threadIdx.x;
    const long long pn = (long long)hd * ds;
    if (bh >= (long long)batch * nh) return;
    const bool live = e < pn;
    const long long b = bh / nh;
    const int h = (int)(bh % nh);
    const long long cstride = (long long)nh * pn;
    const long long base = (b * nc * nh + h) * pn + e;
    float st = live ? ssm[bh * pn + e] : 0.0f;
    float ring[4];
#pragma unroll
    for (int k = 0; k < 4; k++) ring[k] = live && k < nc ? states[base + k * cstride] : 0.0f;
    for (int c0 = 0; c0 < nc; c0 += M2_PASS_WINDOW) {
        const int cn = min(M2_PASS_WINDOW, nc - c0);
        __syncthreads();
        for (int i = threadIdx.x; i < cn; i += blockDim.x) {
            const int c = c0 + i;
            const int lc = min(M2_CHUNK, seq - c * M2_CHUNK);
            decay[i] = m2_exp(acs[((b * nc + c) * nh + h) * M2_CHUNK + lc - 1]);
        }
        __syncthreads();
        if (!live) continue;
        for (int i = 0; i < cn; i++) {
            const int c = c0 + i;
            const float s_c = ring[0];
            ring[0] = ring[1];
            ring[1] = ring[2];
            ring[2] = ring[3];
            ring[3] = c + 4 < nc ? states[base + (c + 4) * cstride] : 0.0f;
            states[base + c * cstride] = st;
            st = __fmaf_rn(decay[i], st, s_c);
        }
    }
    if (live) ssm[bh * pn + e] = st;
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

// The outputs of m2_chunk_scan as register tiles, for the shapes whose
// staging fits shared memory: one block per (chunk, head, tile of
// M2_SCAN_TT tokens), 256 threads. The tile's entering states h_prev
// [P][N + 1] (rows padded so a warp's channel reads fall in distinct
// banks), its C rows, the weights W[t][s] = cb e^min(Acs_t - Acs_s, 0) dt_s
// and the x rows of every token up to the tile's last are staged; a warp
// then takes 4 tokens at a time and a lane the channels lane, lane + 32.
// Each output keeps m2_chunk_scan's chain: n ascending from zero, times
// e^Acs_t, then s ascending up to t, then D x_t. Dynamic shared memory:
// P (N + 1) + TT N + TT 64 + 64 P floats.
#define M2_SCAN_TT 32
template <typename TI>
__device__ __forceinline__ void m2_chunk_scan_tiles_body(
    float* __restrict__ y,
    const float* __restrict__ states,
    const float* __restrict__ cb,
    const TI* __restrict__ xbc, long long ld_xbc, int c_col,
    const float* __restrict__ dt, const float* __restrict__ acs,
    const float* __restrict__ D, int d_per_channel,
    int batch, int seq, int nh, int ng, int hd, int ds
) {
    extern __shared__ float m2cs_sm[];
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int tile0 = blockIdx.z * M2_SCAN_TT;
    if (bc >= (long long)batch * nc || h >= nh) return;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    if (tile0 >= lc) return;
    const int tn = min(M2_SCAN_TT, lc - tile0);
    const int send = tile0 + tn;
    const int g = h / (nh / ng);
    const long long sc = (bc * nh + h) * M2_CHUNK;
    const long long tri = (bc * ng + g) * (M2_CHUNK * (M2_CHUNK + 1) / 2);
    const int hld = ds + 1;
    float* hs = m2cs_sm;                       // [P][N + 1]
    float* cs = hs + hd * hld;                 // [TT][N]
    float* ws = cs + M2_SCAN_TT * ds;          // [TT][64]
    float* xs = ws + M2_SCAN_TT * M2_CHUNK;    // [64][P]
    const float* hg = states + (bc * nh + h) * (long long)hd * ds;
    m2_stage4(hd * ds, [&](int i) { return hg[i]; },
              [&](int i, float v) {
                  const int p = i / ds;
                  hs[p * hld + i - p * ds] = v;
              });
    m2_stage4(tn * ds,
              [&](int i) {
                  const int t = i / ds, n = i - t * ds;
                  return to_f(
                      xbc[(b * seq + t0 + tile0 + t) * ld_xbc + c_col + (long long)g * ds + n]);
              },
              [&](int i, float v) { cs[i] = v; });
    m2_stage4(tn * M2_CHUNK,
              [&](int i) {
                  const int tl = i / M2_CHUNK, s = i - tl * M2_CHUNK;
                  const int t = tile0 + tl;
                  float w = 0.0f;
                  if (s <= t) {
                      const float acs_t = acs[sc + t];
                      w = __fmul_rn(
                          __fmul_rn(cb[tri + (long long)t * (t + 1) / 2 + s],
                                    m2_exp(fminf(__fsub_rn(acs_t, acs[sc + s]), 0.0f))),
                          dt[sc + s]);
                  }
                  return w;
              },
              [&](int i, float v) { ws[i] = v; });
    m2_stage4(send * hd,
              [&](int i) {
                  const int s = i / hd, p = i - s * hd;
                  return to_f(xbc[(b * seq + t0 + s) * ld_xbc + (long long)h * hd + p]);
              },
              [&](int i, float v) { xs[i] = v; });
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (warp >= nw) return;
    for (int r0 = 4 * warp; r0 < tn; r0 += 4 * nw) {
        const int rows = min(4, tn - r0);
        for (int c0 = 0; c0 < hd; c0 += 64) {
            float acc[4][2];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                acc[i][0] = 0.0f;
                acc[i][1] = 0.0f;
            }
            const int p0 = c0 + lane, p1 = c0 + lane + 32;
            const bool on0 = p0 < hd, on1 = p1 < hd;
            for (int n = 0; n < ds; n++) {
                const float h0 = on0 ? hs[p0 * hld + n] : 0.0f;
                const float h1 = on1 ? hs[p1 * hld + n] : 0.0f;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const float cv = i < rows ? cs[(r0 + i) * ds + n] : 0.0f;
                    acc[i][0] = __fmaf_rn(cv, h0, acc[i][0]);
                    acc[i][1] = __fmaf_rn(cv, h1, acc[i][1]);
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (i < rows) {
                    const float e = m2_exp(acs[sc + tile0 + r0 + i]);
                    acc[i][0] = __fmul_rn(acc[i][0], e);
                    acc[i][1] = __fmul_rn(acc[i][1], e);
                }
            }
            const int last = tile0 + r0 + rows - 1;
            for (int s = 0; s <= last; s++) {
                const float x0 = on0 ? xs[s * hd + p0] : 0.0f;
                const float x1 = on1 ? xs[s * hd + p1] : 0.0f;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (i < rows && s <= tile0 + r0 + i) {
                        const float w = ws[(r0 + i) * M2_CHUNK + s];
                        acc[i][0] = __fmaf_rn(w, x0, acc[i][0]);
                        acc[i][1] = __fmaf_rn(w, x1, acc[i][1]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (i >= rows) continue;
                const int t = tile0 + r0 + i;
                const long long yrow = (b * seq + t0 + t) * ((long long)nh * hd) + (long long)h * hd;
                if (on0) {
                    const float dv = d_per_channel ? D[(long long)h * hd + p0] : D[h];
                    y[yrow + p0] = __fmaf_rn(xs[t * hd + p0], dv, acc[i][0]);
                }
                if (on1) {
                    const float dv = d_per_channel ? D[(long long)h * hd + p1] : D[h];
                    y[yrow + p1] = __fmaf_rn(xs[t * hd + p1], dv, acc[i][1]);
                }
            }
        }
    }
}

#define DEFINE_M2_CHUNK_SCAN_TILES(NAME, TI)                                   \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ y,                                                     \
    const float* __restrict__ states,                                          \
    const float* __restrict__ cb,                                              \
    const TI* __restrict__ xbc, long long ld_xbc, int c_col,                  \
    const float* __restrict__ dt, const float* __restrict__ acs,               \
    const float* __restrict__ D, int d_per_channel,                            \
    int batch, int seq, int nh, int ng, int hd, int ds                         \
) {                                                                            \
    m2_chunk_scan_tiles_body<TI>(y, states, cb, xbc, ld_xbc, c_col, dt, acs,   \
        D, d_per_channel, batch, seq, nh, ng, hd, ds);                         \
}

DEFINE_M2_CHUNK_SCAN_TILES(m2_chunk_scan_tiles, float)
DEFINE_M2_CHUNK_SCAN_TILES(m2_chunk_scan_tiles_bf16, __nv_bfloat16)
DEFINE_M2_CHUNK_SCAN_TILES(m2_chunk_scan_tiles_f16, __half)
