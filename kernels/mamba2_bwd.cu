// Mamba-2 training on the GPU: the backward of the chunked prefill in
// mamba2_ssd.cu, the gated norm, the gates and the conv, all f32.
//
// Per chunk c of Lc tokens, head h of group g, the forward is
//
//   W[t][s] = CB[t][s] E(t, s) dt_s,   E(t, s) = e^min(Acs_t - Acs_s, 0)
//   S_c     = sum_t B_t (x_t E(L-1, t) dt_t)
//   H_c+1   = e^Acs_last H_c + S_c                   (H_c enters chunk c)
//   y_t     = e^Acs_t (C_t . H_c) + sum_{s<=t} W[t][s] x_s + D x_t
//
// and the backward, given dy:
//
//   dH_loc[c] = sum_t e^Acs_t dy_t C_t
//   dS[c]     = d H_c+1, walked back over the chunks from zero:
//               d H_c = dH_loc[c] + e^Acs_last dS[c]
//   acc_s     = E(L-1, s) (B_s . dS[c]) + sum_{t>=s} CB[t][s] E(t, s) dy_t
//   dx_s      = acc_s dt_s + D dy_s,          ddt_x[s] = sum_p acc_s x_s
//   dCB[t][s] = sum_{h in g} (dy_t . x_s) E(t, s) dt_s
//   dB_s      = sum_{h in g} dt_s E(L-1, s) (x_s . dS[c]) + sum_{t>=s} dCB[t][s] C_t
//   dC_t      = sum_{h in g} e^Acs_t (dy_t . H_c) + sum_{s<=t} dCB[t][s] B_s
//
// and the gradient of every decay a_k = dt_k A gathered without
// subtractions (each exponent is a sum of a over a span, so a pair term
// reaches exactly the a_k inside its span):
//
//   da_k = sum_{t>=k, s<k} M[t][s]           M = (dy_t . x_s) E dt_s CB, t > s
//        + sum_{t<k} Q_t                      Q_t = dt_t sum_p x_t (E(L-1,t) B_t . dS)
//        + sum_{t>=k} I_t                     I_t = dy_t . (e^Acs_t C_t . H_c)
//        + e^Acs_last (H_c . dS[c])           (the state passing, every k)
//
// Every value is one thread's chain in a fixed order with explicit
// round-to-nearest intrinsics, and reductions are fixed trees: the result
// does not depend on the tiling, the board's SM count or the run. Offsets
// are 64-bit. The gradients do not reach the states a window starts from.
//
// Kernels that read the projection or conv rows (or write the projection
// gradient) are one template body with f32, bf16 and f16 entry points: the
// half lanes read their stored rows through to_f and round the projection
// gradient once on store, while everything else stays f32. The f32 entry
// points are the same chains.
//
// Composed after mamba2_ops.cu and mamba2_ssd.cu in the Mamba-2 module.

#define M2B_PAIRS (M2_CHUNK * (M2_CHUNK + 1) / 2)

__device__ __forceinline__ int m2b_pair(int t, int s) {
    return t * (t + 1) / 2 + s;
}

// The (t, s) of pair q, s <= t, the forward bmm's decode.
__device__ __forceinline__ void m2b_unpair(int q, int* t_out, int* s_out) {
    int t = (int)((sqrtf(8.0f * q + 1.0f) - 1.0f) * 0.5f);
    while ((t + 1) * (t + 2) / 2 <= q) t++;
    while (t * (t + 1) / 2 > q) t--;
    *t_out = t;
    *s_out = q - t * (t + 1) / 2;
}

__device__ __forceinline__ float m2b_sigmoid(float v) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, m2_exp(-v)));
}

// d silu(z) / dz = s (1 + z (1 - s)), s = sigmoid(z).
__device__ __forceinline__ float m2b_silu_grad(float z) {
    float s = m2b_sigmoid(z);
    return __fmul_rn(s, __fmaf_rn(z, __fsub_rn(1.0f, s), 1.0f));
}

// A store in the activation type, rounded to nearest.
__device__ __forceinline__ void m2b_put(float* p, float v) { *p = v; }
__device__ __forceinline__ void m2b_put(__nv_bfloat16* p, float v) { *p = from_f_bf16(v); }
__device__ __forceinline__ void m2b_put(__half* p, float v) { *p = from_f_f16(v); }

// The host's exp_portable (src/ops/portable_math.rs) operation for
// operation, so A = -exp(a_log) computed here equals the inference
// engine's host-side value bit for bit.
__device__ __forceinline__ float m2_exp_portable(float x) {
    const float lo = -87.33654f;
    const float hi = 88.37626f;
    const float magic = 12582912.0f;
    float xc = x != x ? x : fminf(fmaxf(x, lo), hi);
    float t = __fmaf_rn(xc, 1.4426950408889634f, magic);
    float n = __fsub_rn(t, magic);
    int ni = __float_as_int(t) - __float_as_int(magic);
    float r = __fmaf_rn(n, 2.1219444e-4f, __fmaf_rn(n, -0.6933594f, xc));
    float r2 = __fmul_rn(r, r);
    float p = 1.9875691e-4f;
    p = __fmaf_rn(p, r, 1.3981999e-3f);
    p = __fmaf_rn(p, r, 8.333452e-3f);
    p = __fmaf_rn(p, r, 4.1665796e-2f);
    p = __fmaf_rn(p, r, 1.6666665e-1f);
    p = __fmaf_rn(p, r, 0.5f);
    p = __fadd_rn(__fmaf_rn(p, r2, r), 1.0f);
    return __fmul_rn(p, __int_as_float((int)((unsigned)(ni + 127) << 23)));
}

// A = -exp(a_log) per head. Grid 1-D over n.
extern "C" __global__ void m2_a_neg(
    float* __restrict__ a_neg, const float* __restrict__ a_log, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a_neg[i] = -m2_exp_portable(a_log[i]);
}

// dH_loc[b, c, h, p, n] = sum_t (dy_t[p] e^Acs_t) C_t[n], t ascending.
// Grid (B * nc, H, ceil(P * N / blockDim.x)), the forward chunk state's.
template <typename T>
__device__ __forceinline__ void m2b_dhloc_body(
    float* __restrict__ ds_out,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc, int c_col,
    const float* __restrict__ acs,
    int batch, int seq, int nh, int ng, int hd, int ds
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const long long e = (long long)blockIdx.z * blockDim.x + threadIdx.x;
    if (bc >= (long long)batch * nc || h >= nh || e >= (long long)hd * ds) return;
    const int p = (int)(e / ds);
    const int n = (int)(e % ds);
    const int g = h / (nh / ng);
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    const long long sc = (bc * nh + h) * M2_CHUNK;
    const long long d_ssm = (long long)nh * hd;
    float acc = 0.0f;
    for (int t = 0; t < lc; t++) {
        const long long row = b * seq + t0 + t;
        float dye = __fmul_rn(dy[row * d_ssm + (long long)h * hd + p], m2_exp(acs[sc + t]));
        acc = __fmaf_rn(dye, to_f(xbc[row * ld_xbc + c_col + (long long)g * ds + n]), acc);
    }
    ds_out[((bc * nh + h) * hd + p) * (long long)ds + n] = acc;
}

#define DEFINE_M2B_DHLOC(NAME, T)                                               \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ ds_out,                                                 \
    const float* __restrict__ dy,                                               \
    const T* __restrict__ xbc,                                                  \
    long long ld_xbc,                                                           \
    int c_col,                                                                  \
    const float* __restrict__ acs,                                              \
    int batch,                                                                  \
    int seq,                                                                    \
    int nh,                                                                     \
    int ng,                                                                     \
    int hd,                                                                     \
    int ds                                                                      \
) {                                                                             \
    m2b_dhloc_body<T>(ds_out, dy, xbc, ld_xbc, c_col, acs, batch, seq, nh,      \
        ng, hd, ds);                                                            \
}
DEFINE_M2B_DHLOC(m2b_dhloc, float)
DEFINE_M2B_DHLOC(m2b_dhloc_bf16, __nv_bfloat16)
DEFINE_M2B_DHLOC(m2b_dhloc_f16, __half)

// The state passing backward, in place: dS holds dH_loc on entry and
// dS[c] = d H_c+1 on exit, from zero after the last chunk:
// e <- dH_loc[c] + e^Acs_last e. As m2_state_passing, the decays sit in
// shared memory and the next four chunks are loaded ahead.
// Grid (B * H, ceil(P * N / blockDim.x)).
extern "C" __global__ void m2b_state_passing(
    float* __restrict__ ds,
    const float* __restrict__ acs,
    int batch, int seq, int nh, int hd, int dstate
) {
    __shared__ float decay[M2_PASS_WINDOW];
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bh = blockIdx.x;
    const long long e = (long long)blockIdx.y * blockDim.x + threadIdx.x;
    const long long pn = (long long)hd * dstate;
    if (bh >= (long long)batch * nh) return;
    const bool live = e < pn;
    const long long b = bh / nh;
    const int h = (int)(bh % nh);
    const long long cstride = (long long)nh * pn;
    const long long base = (b * nc * nh + h) * pn + e;
    float ev = 0.0f;
    float ring[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int c = nc - 1 - k;
        ring[k] = live && c >= 0 ? ds[base + c * cstride] : 0.0f;
    }
    for (int c1 = nc; c1 > 0; c1 -= M2_PASS_WINDOW) {
        const int c0 = max(0, c1 - M2_PASS_WINDOW);
        __syncthreads();
        for (int i = threadIdx.x; i < c1 - c0; i += blockDim.x) {
            const int c = c0 + i;
            const int lc = min(M2_CHUNK, seq - c * M2_CHUNK);
            decay[i] = m2_exp(acs[((b * nc + c) * nh + h) * M2_CHUNK + lc - 1]);
        }
        __syncthreads();
        if (!live) continue;
        for (int c = c1 - 1; c >= c0; c--) {
            const float tmp = ring[0];
            ring[0] = ring[1];
            ring[1] = ring[2];
            ring[2] = ring[3];
            ring[3] = c - 4 >= 0 ? ds[base + (c - 4) * cstride] : 0.0f;
            ds[base + c * cstride] = ev;
            ev = __fmaf_rn(decay[c - c0], ev, tmp);
        }
    }
}

// dot[q] = H_c . dS[c] for q = (b * nc + c) * H + h: strided ascending
// partials, then a fixed tree. Grid 1-D over B * nc * H, block 256.
extern "C" __global__ void m2b_chunk_dot(
    float* __restrict__ dot,
    const float* __restrict__ states,
    const float* __restrict__ ds,
    long long count, int pn
) {
    __shared__ float red[256];
    const long long q = blockIdx.x;
    if (q >= count) return;
    const float* hs = states + q * pn;
    const float* gs = ds + q * pn;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < pn; i += 256) acc = __fmaf_rn(hs[i], gs[i], acc);
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (threadIdx.x < w) red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + w]);
        __syncthreads();
    }
    if (threadIdx.x == 0) dot[q] = red[0];
}

// dx and the per-channel terms the decay gradients need, one thread per
// (s, p) of a chunk and head:
//   st = B_s . dS[c][p], ci = C_s . H_c[p]      (n ascending, together)
//   ste = st E(L-1, s); acc = ste + sum_{t>=s} (CB E(t, s)) dy_t[p]
//   dx = acc dt_s + D dy_s[p]
// writing dx into the x columns of dxbc and acc, ste and
// yi = ci e^Acs_s to [B * T, d_ssm] scratch.
// Grid (B * nc, H, ceil(64 * P / blockDim.x)), the chunk scan's.
template <typename T>
__device__ __forceinline__ void m2b_dx_body(
    float* __restrict__ dxbc,
    float* __restrict__ acc_out, float* __restrict__ ste_out, float* __restrict__ yi_out,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc, int b_col, int c_col,
    const float* __restrict__ states, const float* __restrict__ ds_in,
    const float* __restrict__ cb,
    const float* __restrict__ dt, const float* __restrict__ acs,
    const float* __restrict__ D, int d_per_channel,
    int batch, int seq, int nh, int ng, int hd, int dstate
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int e = blockIdx.z * blockDim.x + threadIdx.x;
    if (bc >= (long long)batch * nc || h >= nh || e >= M2_CHUNK * hd) return;
    const int s = e / hd;
    const int p = e % hd;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    if (s >= lc) return;
    const int g = h / (nh / ng);
    const long long d_ssm = (long long)nh * hd;
    const long long sc = (bc * nh + h) * M2_CHUNK;
    const long long rs = b * seq + t0 + s;
    const long long so = ((bc * nh + h) * hd + p) * (long long)dstate;
    float st = 0.0f, ci = 0.0f;
    for (int n = 0; n < dstate; n++) {
        const long long col = (long long)g * dstate + n;
        st = __fmaf_rn(to_f(xbc[rs * ld_xbc + b_col + col]), ds_in[so + n], st);
        ci = __fmaf_rn(to_f(xbc[rs * ld_xbc + c_col + col]), states[so + n], ci);
    }
    const float acs_s = acs[sc + s];
    const float ste = __fmul_rn(st, m2_exp(fminf(__fsub_rn(acs[sc + lc - 1], acs_s), 0.0f)));
    float acc = ste;
    const long long cbo = (bc * ng + g) * M2B_PAIRS;
    for (int t = s; t < lc; t++) {
        float wv = __fmul_rn(cb[cbo + m2b_pair(t, s)],
                             m2_exp(fminf(__fsub_rn(acs[sc + t], acs_s), 0.0f)));
        acc = __fmaf_rn(wv, dy[(b * seq + t0 + t) * d_ssm + (long long)h * hd + p], acc);
    }
    const long long o = rs * d_ssm + (long long)h * hd + p;
    const float dv = d_per_channel ? D[(long long)h * hd + p] : D[h];
    dxbc[rs * ld_xbc + (long long)h * hd + p] = __fmaf_rn(acc, dt[sc + s], __fmul_rn(dv, dy[o]));
    acc_out[o] = acc;
    ste_out[o] = ste;
    yi_out[o] = __fmul_rn(ci, m2_exp(acs_s));
}

#define DEFINE_M2B_DX(NAME, T)                                                  \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ dxbc,                                                   \
    float* __restrict__ acc_out,                                                \
    float* __restrict__ ste_out,                                                \
    float* __restrict__ yi_out,                                                 \
    const float* __restrict__ dy,                                               \
    const T* __restrict__ xbc,                                                  \
    long long ld_xbc,                                                           \
    int b_col,                                                                  \
    int c_col,                                                                  \
    const float* __restrict__ states,                                           \
    const float* __restrict__ ds_in,                                            \
    const float* __restrict__ cb,                                               \
    const float* __restrict__ dt,                                               \
    const float* __restrict__ acs,                                              \
    const float* __restrict__ D,                                                \
    int d_per_channel,                                                          \
    int batch,                                                                  \
    int seq,                                                                    \
    int nh,                                                                     \
    int ng,                                                                     \
    int hd,                                                                     \
    int dstate                                                                  \
) {                                                                             \
    m2b_dx_body<T>(dxbc, acc_out, ste_out, yi_out, dy, xbc, ld_xbc, b_col,      \
        c_col, states, ds_in, cb, dt, acs, D, d_per_channel, batch, seq,        \
        nh, ng, hd, dstate);                                                    \
}
DEFINE_M2B_DX(m2b_dx, float)
DEFINE_M2B_DX(m2b_dx_bf16, __nv_bfloat16)
DEFINE_M2B_DX(m2b_dx_f16, __half)

// The per-token sums over a head's channels, p ascending:
//   ddt_x[t] = sum_p acc x_t,  Q_t = dt_t sum_p ste x_t,  I_t = sum_p dy yi
// stored [B, nc, H, 64] (zero past the chunk). Grid (B * nc, H), block 64.
template <typename T>
__device__ __forceinline__ void m2b_rowsums_body(
    float* __restrict__ ddtx, float* __restrict__ qv, float* __restrict__ inter,
    const float* __restrict__ acc_b, const float* __restrict__ ste_b,
    const float* __restrict__ yi_b, const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc,
    const float* __restrict__ dt,
    int batch, int seq, int nh, int hd
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int t = threadIdx.x;
    if (bc >= (long long)batch * nc || h >= nh || t >= M2_CHUNK) return;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    const long long o = (bc * nh + h) * M2_CHUNK + t;
    if (t >= lc) {
        ddtx[o] = 0.0f;
        qv[o] = 0.0f;
        inter[o] = 0.0f;
        return;
    }
    const long long row = b * seq + t0 + t;
    const long long d_ssm = (long long)nh * hd;
    float a = 0.0f, q = 0.0f, it = 0.0f;
    for (int p = 0; p < hd; p++) {
        const float x = to_f(xbc[row * ld_xbc + (long long)h * hd + p]);
        const long long i = row * d_ssm + (long long)h * hd + p;
        a = __fmaf_rn(acc_b[i], x, a);
        q = __fmaf_rn(ste_b[i], x, q);
        it = __fmaf_rn(dy[i], yi_b[i], it);
    }
    ddtx[o] = a;
    qv[o] = __fmul_rn(q, dt[o]);
    inter[o] = it;
}

#define DEFINE_M2B_ROWSUMS(NAME, T)                                             \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ ddtx,                                                   \
    float* __restrict__ qv,                                                     \
    float* __restrict__ inter,                                                  \
    const float* __restrict__ acc_b,                                            \
    const float* __restrict__ ste_b,                                            \
    const float* __restrict__ yi_b,                                             \
    const float* __restrict__ dy,                                               \
    const T* __restrict__ xbc,                                                  \
    long long ld_xbc,                                                           \
    const float* __restrict__ dt,                                               \
    int batch,                                                                  \
    int seq,                                                                    \
    int nh,                                                                     \
    int hd                                                                      \
) {                                                                             \
    m2b_rowsums_body<T>(ddtx, qv, inter, acc_b, ste_b, yi_b, dy, xbc,           \
        ld_xbc, dt, batch, seq, nh, hd);                                        \
}
DEFINE_M2B_ROWSUMS(m2b_rowsums, float)
DEFINE_M2B_ROWSUMS(m2b_rowsums_bf16, __nv_bfloat16)
DEFINE_M2B_ROWSUMS(m2b_rowsums_f16, __half)

// m2b_dx and m2b_rowsums in one block per chunk and head, as register
// tiles: thread (s-group, p-group) holds 4 tokens s by 4 channels p.
//   st = B_s . dS[c][p] and ci = C_s . H_c[p] run n ascending from 0 over
//   the rows staged `nk` state indices at a time (B or C transposed [n][s],
//   dS or H transposed [n][p]);
//   ste = st E(L-1, s), then acc runs t = s.. ascending over the staged
//   weights wv[t][s] = CB E(t, s) and dy rows; dx = fma(acc, dt_s, D dy_s);
//   acc, ste and yi = ci e^Acs_s go to shared rows, where one thread per
//   token runs the rowsums' p-ascending chains.
// Every value is the two kernels' own chain, so the bits are theirs.
// Shared memory: max(nk (68 + P4 + 4), 3 * 64 (P + 1)) staging floats, then
// wv [64][64], dy [64][P4], acs and dt [64] each; P4 = P rounded up to 4.
// Grid (B * nc, H), block 16 ceil(P / 4) threads.
template <typename T>
__device__ __forceinline__ void m2b_dx_tiles_body(
    float* __restrict__ dxbc,
    float* __restrict__ ddtx, float* __restrict__ qv, float* __restrict__ inter,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc, int b_col, int c_col,
    const float* __restrict__ states, const float* __restrict__ ds_in,
    const float* __restrict__ cb,
    const float* __restrict__ dt, const float* __restrict__ acs,
    const float* __restrict__ D, int d_per_channel,
    int seq, int nh, int ng, int hd, int dstate, int nk
) {
    extern __shared__ float4 m2bt_sm4[];
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    const int g = h / (nh / ng);
    const long long d_ssm = (long long)nh * hd;
    const long long sc = (bc * nh + h) * M2_CHUNK;
    const int pg = (hd + 3) / 4;
    const int p4 = 4 * pg;
    const int lts = M2_CHUNK + 4, rts = p4 + 4, rows = hd + 1;
    const int stage_floats = max(nk * (lts + rts), 3 * M2_CHUNK * rows);
    float* lt = reinterpret_cast<float*>(m2bt_sm4);  // [nk][lts]
    float* rt = lt + nk * lts;                        // [nk][rts]
    float* at = lt;                                   // [64][P + 1] each
    float* et = at + M2_CHUNK * rows;
    float* yt = et + M2_CHUNK * rows;
    float* wv = lt + stage_floats;                    // [64][64]
    float* dyt = wv + M2_CHUNK * M2_CHUNK;            // [64][p4]
    float* acs_s = dyt + M2_CHUNK * p4;               // [64]
    float* dt_s = acs_s + M2_CHUNK;                   // [64]
    const int tid = threadIdx.x;
    const int s0 = 4 * (tid / pg), p0 = 4 * (tid % pg);
    const long long cbo = (bc * ng + g) * M2B_PAIRS;
    for (int i = tid; i < M2_CHUNK; i += blockDim.x) {
        acs_s[i] = i < lc ? acs[sc + i] : 0.0f;
        dt_s[i] = i < lc ? dt[sc + i] : 0.0f;
    }
    m2_stage4(M2_CHUNK * M2_CHUNK, tid, blockDim.x,
              [&](int i) {
                  const int t = i / M2_CHUNK, s = i - t * M2_CHUNK;
                  float w = 0.0f;
                  if (s <= t && t < lc)
                      w = __fmul_rn(cb[cbo + m2b_pair(t, s)],
                                    m2_exp(fminf(__fsub_rn(acs[sc + t], acs[sc + s]), 0.0f)));
                  return w;
              },
              [&](int i, float v) { wv[i] = v; });
    m2_stage4(M2_CHUNK * p4, tid, blockDim.x,
              [&](int i) {
                  const int t = i / p4, p = i - t * p4;
                  return t < lc && p < hd
                      ? dy[(b * seq + t0 + t) * d_ssm + (long long)h * hd + p] : 0.0f;
              },
              [&](int i, float v) { dyt[i] = v; });
    float st[4][4], ci[4][4];
    const float* hs = ds_in + ((bc * nh + h) * hd) * (long long)dstate;
    const float* hc = states + ((bc * nh + h) * hd) * (long long)dstate;
#pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        const int col = pass == 0 ? b_col : c_col;
        const float* src = pass == 0 ? hs : hc;
        float a[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) a[i][j] = 0.0f;
        }
        for (int m0 = 0; m0 < dstate; m0 += nk) {
            const int mm = min(nk, dstate - m0);
            __syncthreads();
            m2_stage4(M2_CHUNK * mm, tid, blockDim.x,
                      [&](int i) {
                          const int s = i / mm, n = i - s * mm;
                          return s < lc ? to_f(xbc[(b * seq + t0 + s) * ld_xbc + col
                                                   + (long long)g * dstate + m0 + n])
                                        : 0.0f;
                      },
                      [&](int i, float v) {
                          const int s = i / mm;
                          lt[(i - s * mm) * lts + s] = v;
                      });
            m2_stage4(p4 * mm, tid, blockDim.x,
                      [&](int i) {
                          const int p = i / mm, n = i - p * mm;
                          return p < hd ? src[(long long)p * dstate + m0 + n] : 0.0f;
                      },
                      [&](int i, float v) {
                          const int p = i / mm;
                          rt[(i - p * mm) * rts + p] = v;
                      });
            __syncthreads();
            for (int n = 0; n < mm; n++) {
                const float4 l4 = *reinterpret_cast<const float4*>(lt + n * lts + s0);
                const float4 r4 = *reinterpret_cast<const float4*>(rt + n * rts + p0);
                const float lv[4] = {l4.x, l4.y, l4.z, l4.w};
                const float rv[4] = {r4.x, r4.y, r4.z, r4.w};
#pragma unroll
                for (int i = 0; i < 4; i++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) a[i][j] = __fmaf_rn(lv[i], rv[j], a[i][j]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                if (pass == 0) st[i][j] = a[i][j];
                else ci[i][j] = a[i][j];
            }
        }
    }
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float el = m2_exp(fminf(__fsub_rn(acs_s[max(lc - 1, 0)], acs_s[s0 + i]), 0.0f));
#pragma unroll
        for (int j = 0; j < 4; j++) {
            st[i][j] = __fmul_rn(st[i][j], el);
            acc[i][j] = st[i][j];
        }
    }
    for (int t = s0; t < lc; t++) {
        const float4 w4 = *reinterpret_cast<const float4*>(wv + t * M2_CHUNK + s0);
        const float4 y4 = *reinterpret_cast<const float4*>(dyt + t * p4 + p0);
        const float wl[4] = {w4.x, w4.y, w4.z, w4.w};
        const float yl[4] = {y4.x, y4.y, y4.z, y4.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (t >= s0 + i) {
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(wl[i], yl[j], acc[i][j]);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int s = s0 + i;
        if (s >= lc) continue;
        const long long rs = b * seq + t0 + s;
        const float ea = m2_exp(acs_s[s]);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int p = p0 + j;
            if (p >= hd) continue;
            const float dv = d_per_channel ? D[(long long)h * hd + p] : D[h];
            dxbc[rs * ld_xbc + (long long)h * hd + p] =
                __fmaf_rn(acc[i][j], dt_s[s], __fmul_rn(dv, dyt[s * p4 + p]));
            at[s * rows + p] = acc[i][j];
            et[s * rows + p] = st[i][j];
            yt[s * rows + p] = __fmul_rn(ci[i][j], ea);
        }
    }
    __syncthreads();
    for (int t = tid; t < M2_CHUNK; t += blockDim.x) {
        const long long o = sc + t;
        if (t >= lc) {
            ddtx[o] = 0.0f;
            qv[o] = 0.0f;
            inter[o] = 0.0f;
            continue;
        }
        const long long row = b * seq + t0 + t;
        float a = 0.0f, q = 0.0f, it = 0.0f;
        for (int p = 0; p < hd; p++) {
            const float x = to_f(xbc[row * ld_xbc + (long long)h * hd + p]);
            a = __fmaf_rn(at[t * rows + p], x, a);
            q = __fmaf_rn(et[t * rows + p], x, q);
            it = __fmaf_rn(dyt[t * p4 + p], yt[t * rows + p], it);
        }
        ddtx[o] = a;
        qv[o] = __fmul_rn(q, dt_s[t]);
        inter[o] = it;
    }
}

#define DEFINE_M2B_DX_TILES(NAME, T)                                            \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ dxbc,                                                   \
    float* __restrict__ ddtx,                                                   \
    float* __restrict__ qv,                                                     \
    float* __restrict__ inter,                                                  \
    const float* __restrict__ dy,                                               \
    const T* __restrict__ xbc,                                                  \
    long long ld_xbc,                                                           \
    int b_col,                                                                  \
    int c_col,                                                                  \
    const float* __restrict__ states,                                           \
    const float* __restrict__ ds_in,                                            \
    const float* __restrict__ cb,                                               \
    const float* __restrict__ dt,                                               \
    const float* __restrict__ acs,                                              \
    const float* __restrict__ D,                                                \
    int d_per_channel,                                                          \
    int seq,                                                                    \
    int nh,                                                                     \
    int ng,                                                                     \
    int hd,                                                                     \
    int dstate,                                                                 \
    int nk                                                                      \
) {                                                                             \
    m2b_dx_tiles_body<T>(dxbc, ddtx, qv, inter, dy, xbc, ld_xbc, b_col, c_col,  \
        states, ds_in, cb, dt, acs, D, d_per_channel, seq, nh, ng, hd, dstate,  \
        nk);                                                                    \
}
DEFINE_M2B_DX_TILES(m2b_dx_tiles, float)
DEFINE_M2B_DX_TILES(m2b_dx_tiles_bf16, __nv_bfloat16)
DEFINE_M2B_DX_TILES(m2b_dx_tiles_f16, __half)

// The pair terms of a chunk and head, one thread per pair s <= t:
//   G = dy_t . x_s (p ascending), dcb = G E(t, s) dt_s,
//   m = dcb CB[t][s] for t > s (0 on the diagonal)
// stored [B, nc, H, pairs]. Grid (B * nc, H, ceil(pairs / blockDim.x)).
template <typename T>
__device__ __forceinline__ void m2b_pairs_body(
    float* __restrict__ dcbh, float* __restrict__ mm,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc,
    const float* __restrict__ cb,
    const float* __restrict__ dt, const float* __restrict__ acs,
    int batch, int seq, int nh, int ng, int hd
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int q = blockIdx.z * blockDim.x + threadIdx.x;
    if (bc >= (long long)batch * nc || h >= nh || q >= M2B_PAIRS) return;
    int t, s;
    m2b_unpair(q, &t, &s);
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    const long long o = (bc * nh + h) * M2B_PAIRS + q;
    if (t >= lc) {
        dcbh[o] = 0.0f;
        mm[o] = 0.0f;
        return;
    }
    const int g = h / (nh / ng);
    const long long d_ssm = (long long)nh * hd;
    const long long rt = b * seq + t0 + t;
    const long long rs = b * seq + t0 + s;
    float gv = 0.0f;
    for (int p = 0; p < hd; p++) {
        gv = __fmaf_rn(dy[rt * d_ssm + (long long)h * hd + p],
                       to_f(xbc[rs * ld_xbc + (long long)h * hd + p]), gv);
    }
    const long long sc = (bc * nh + h) * M2_CHUNK;
    const float ev = m2_exp(fminf(__fsub_rn(acs[sc + t], acs[sc + s]), 0.0f));
    const float dcb = __fmul_rn(__fmul_rn(gv, ev), dt[sc + s]);
    dcbh[o] = dcb;
    mm[o] = t > s ? __fmul_rn(dcb, cb[(bc * ng + g) * M2B_PAIRS + q]) : 0.0f;
}

#define DEFINE_M2B_PAIRS(NAME, T)                                               \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ dcbh,                                                   \
    float* __restrict__ mm,                                                     \
    const float* __restrict__ dy,                                               \
    const T* __restrict__ xbc,                                                  \
    long long ld_xbc,                                                           \
    const float* __restrict__ cb,                                               \
    const float* __restrict__ dt,                                               \
    const float* __restrict__ acs,                                              \
    int batch,                                                                  \
    int seq,                                                                    \
    int nh,                                                                     \
    int ng,                                                                     \
    int hd                                                                      \
) {                                                                             \
    m2b_pairs_body<T>(dcbh, mm, dy, xbc, ld_xbc, cb, dt, acs, batch, seq,       \
        nh, ng, hd);                                                            \
}
DEFINE_M2B_PAIRS(m2b_pairs, float)
DEFINE_M2B_PAIRS(m2b_pairs_bf16, __nv_bfloat16)
DEFINE_M2B_PAIRS(m2b_pairs_f16, __half)

// dB of a chunk and group, one thread per (s, n): the group's heads
// ascending (each x_s . dS[c][:, n], p ascending, times dt_s E(L-1, s)),
// then t >= s ascending of dCB[t][s] C_t[n], where dCB sums the group's
// heads ascending. Written into the B columns of dxbc.
// Grid (B * nc, G, ceil(64 * N / blockDim.x)).
template <typename T>
__device__ __forceinline__ void m2b_db_body(
    float* __restrict__ dxbc,
    const T* __restrict__ xbc, long long ld_xbc, int b_col, int c_col,
    const float* __restrict__ ds_in, const float* __restrict__ dcbh,
    const float* __restrict__ dt, const float* __restrict__ acs,
    int batch, int seq, int nh, int ng, int hd, int dstate
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int g = blockIdx.y;
    const int e = blockIdx.z * blockDim.x + threadIdx.x;
    if (bc >= (long long)batch * nc || g >= ng || e >= M2_CHUNK * dstate) return;
    const int s = e / dstate;
    const int n = e % dstate;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    if (s >= lc) return;
    const int hpg = nh / ng;
    const long long rs = b * seq + t0 + s;
    float acc = 0.0f;
    for (int k = 0; k < hpg; k++) {
        const int h = g * hpg + k;
        const long long sc = (bc * nh + h) * M2_CHUNK;
        const long long so = (bc * nh + h) * hd * (long long)dstate + n;
        float inner = 0.0f;
        for (int p = 0; p < hd; p++) {
            inner = __fmaf_rn(to_f(xbc[rs * ld_xbc + (long long)h * hd + p]),
                              ds_in[so + (long long)p * dstate], inner);
        }
        const float el = m2_exp(fminf(__fsub_rn(acs[sc + lc - 1], acs[sc + s]), 0.0f));
        acc = __fmaf_rn(inner, __fmul_rn(dt[sc + s], el), acc);
    }
    for (int t = s; t < lc; t++) {
        float dcb = 0.0f;
        for (int k = 0; k < hpg; k++) {
            dcb = __fadd_rn(dcb, dcbh[(bc * nh + g * hpg + k) * M2B_PAIRS + m2b_pair(t, s)]);
        }
        const long long ct = (b * seq + t0 + t) * ld_xbc + c_col + (long long)g * dstate + n;
        acc = __fmaf_rn(dcb, to_f(xbc[ct]), acc);
    }
    dxbc[rs * ld_xbc + b_col + (long long)g * dstate + n] = acc;
}

#define DEFINE_M2B_DB(NAME, T)                                                  \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ dxbc,                                                   \
    const T* __restrict__ xbc,                                                  \
    long long ld_xbc,                                                           \
    int b_col,                                                                  \
    int c_col,                                                                  \
    const float* __restrict__ ds_in,                                            \
    const float* __restrict__ dcbh,                                             \
    const float* __restrict__ dt,                                               \
    const float* __restrict__ acs,                                              \
    int batch,                                                                  \
    int seq,                                                                    \
    int nh,                                                                     \
    int ng,                                                                     \
    int hd,                                                                     \
    int dstate                                                                  \
) {                                                                             \
    m2b_db_body<T>(dxbc, xbc, ld_xbc, b_col, c_col, ds_in, dcbh, dt, acs,       \
        batch, seq, nh, ng, hd, dstate);                                        \
}
DEFINE_M2B_DB(m2b_db, float)
DEFINE_M2B_DB(m2b_db_bf16, __nv_bfloat16)
DEFINE_M2B_DB(m2b_db_f16, __half)

// m2b_db and m2b_dc as register tiles, one block per (chunk, group, tile
// of 32 state indices): a warp takes 4 rows (s for dB, t for dC) and a
// lane an index n. For each head of the group ascending, the head's x (dB)
// or dy (dC) rows are staged transposed [P][64] with its dS (dB) or H (dC)
// columns [P][32]; the p-ascending inner sum from 0 then joins the row's
// chain as fma(inner, scale, acc), scale = dt_s E(L-1, s) (dB) or e^Acs_t
// (dC). Then the chain runs over the staged dCB of the group (each pair
// summed over the heads ascending from 0) and the C rows t >= s (dB) or
// the B rows s <= t (dC), ascending: the bits of m2b_db and m2b_dc.
// Shared memory: (68 P + 32 P + 64 * 64 + 64 * 32 + 64) floats (the
// transposed rows padded to 68 so their staging writes spread over banks).
// Grid (B * nc, G, ceil(N / 32)), block 512.
template <typename T, bool DC>
__device__ __forceinline__ void m2b_bc_tiles_body(
    float* __restrict__ dxbc,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc, int b_col, int c_col,
    const float* __restrict__ mat, const float* __restrict__ dcbh,
    const float* __restrict__ dt, const float* __restrict__ acs,
    int seq, int nh, int ng, int hd, int dstate
) {
    extern __shared__ float m2bt_sm[];
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int g = blockIdx.y;
    const int n0 = blockIdx.z * 32;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    const int hpg = nh / ng;
    const long long d_ssm = (long long)nh * hd;
    constexpr int LS = M2_CHUNK + 4;
    float* lt = m2bt_sm;                     // [P][LS]
    float* rt = lt + hd * LS;                // [P][32]
    float* dcbg = rt + hd * 32;              // [64][64], t by s
    float* orow = dcbg + M2_CHUNK * M2_CHUNK;// [64][32]
    float* scale = orow + M2_CHUNK * 32;     // [64]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5, nw = blockDim.x >> 5;
    const int r0 = 4 * warp;
    const int n = n0 + lane;
    const long long row0 = b * seq + t0;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < hpg; k++) {
        const int h = g * hpg + k;
        const long long sc = (bc * nh + h) * M2_CHUNK;
        const float* mh = mat + (bc * nh + h) * hd * (long long)dstate;
        __syncthreads();
        for (int r = warp; r < M2_CHUNK; r += nw) {
            for (int p = lane; p < hd; p += 32) {
                float v = 0.0f;
                if (r < lc) {
                    v = DC ? dy[(row0 + r) * d_ssm + (long long)h * hd + p]
                           : to_f(xbc[(row0 + r) * ld_xbc + (long long)h * hd + p]);
                }
                lt[p * LS + r] = v;
            }
        }
        for (int p = warp; p < hd; p += nw)
            rt[p * 32 + lane] = n < dstate ? mh[(long long)p * dstate + n] : 0.0f;
        for (int r = tid; r < M2_CHUNK; r += blockDim.x) {
            float sv = 0.0f;
            if (r < lc) {
                sv = DC ? m2_exp(acs[sc + r])
                        : __fmul_rn(dt[sc + r],
                                    m2_exp(fminf(__fsub_rn(acs[sc + lc - 1], acs[sc + r]), 0.0f)));
            }
            scale[r] = sv;
        }
        __syncthreads();
        float inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int p = 0; p < hd; p++) {
            const float4 l4 = *reinterpret_cast<const float4*>(lt + p * LS + r0);
            const float rv = rt[p * 32 + lane];
            inner[0] = __fmaf_rn(l4.x, rv, inner[0]);
            inner[1] = __fmaf_rn(l4.y, rv, inner[1]);
            inner[2] = __fmaf_rn(l4.z, rv, inner[2]);
            inner[3] = __fmaf_rn(l4.w, rv, inner[3]);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) acc[i] = __fmaf_rn(inner[i], scale[r0 + i], acc[i]);
    }
    __syncthreads();
    for (int e = tid; e < M2_CHUNK * M2_CHUNK; e += blockDim.x) {
        const int t = e / M2_CHUNK, s_ = e % M2_CHUNK;
        float dcb = 0.0f;
        if (s_ <= t && t < lc) {
            for (int k = 0; k < hpg; k++)
                dcb = __fadd_rn(dcb, dcbh[(bc * nh + g * hpg + k) * M2B_PAIRS + m2b_pair(t, s_)]);
        }
        dcbg[e] = dcb;
    }
    const int ocol = (DC ? b_col : c_col) + g * dstate;
    for (int r = warp; r < M2_CHUNK; r += nw)
        orow[r * 32 + lane] = r < lc && n < dstate ? to_f(xbc[(row0 + r) * ld_xbc + ocol + n]) : 0.0f;
    __syncthreads();
    if (DC) {
        // dC row t: s = 0 .. t ascending.
        for (int s_ = 0; s_ <= r0 + 3 && s_ < lc; s_++) {
            const float bv = orow[s_ * 32 + lane];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (s_ <= r0 + i) acc[i] = __fmaf_rn(dcbg[(r0 + i) * M2_CHUNK + s_], bv, acc[i]);
            }
        }
    } else {
        // dB row s: t = s .. lc - 1 ascending.
        for (int t = r0; t < lc; t++) {
            const float cv = orow[t * 32 + lane];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (t >= r0 + i) acc[i] = __fmaf_rn(dcbg[t * M2_CHUNK + r0 + i], cv, acc[i]);
            }
        }
    }
    if (n >= dstate) return;
    const int wcol = (DC ? c_col : b_col) + g * dstate;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int r = r0 + i;
        if (r < lc) dxbc[(row0 + r) * ld_xbc + wcol + n] = acc[i];
    }
}

#define DEFINE_M2B_BC_TILES(NAME, T, DC)                                        \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ dxbc,                                                   \
    const float* __restrict__ dy,                                               \
    const T* __restrict__ xbc, long long ld_xbc, int b_col, int c_col,          \
    const float* __restrict__ mat, const float* __restrict__ dcbh,              \
    const float* __restrict__ dt, const float* __restrict__ acs,                \
    int seq, int nh, int ng, int hd, int dstate                                 \
) {                                                                             \
    m2b_bc_tiles_body<T, DC>(dxbc, dy, xbc, ld_xbc, b_col, c_col, mat, dcbh,    \
        dt, acs, seq, nh, ng, hd, dstate);                                      \
}
DEFINE_M2B_BC_TILES(m2b_db_tiles, float, false)
DEFINE_M2B_BC_TILES(m2b_db_tiles_bf16, __nv_bfloat16, false)
DEFINE_M2B_BC_TILES(m2b_db_tiles_f16, __half, false)
DEFINE_M2B_BC_TILES(m2b_dc_tiles, float, true)
DEFINE_M2B_BC_TILES(m2b_dc_tiles_bf16, __nv_bfloat16, true)
DEFINE_M2B_BC_TILES(m2b_dc_tiles_f16, __half, true)

// dC of a chunk and group, one thread per (t, n): the group's heads
// ascending (each dy_t . H_c[:, n], p ascending, times e^Acs_t), then
// s <= t ascending of dCB[t][s] B_s[n]. Written into the C columns.
// Grid (B * nc, G, ceil(64 * N / blockDim.x)).
template <typename T>
__device__ __forceinline__ void m2b_dc_body(
    float* __restrict__ dxbc,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc, int b_col, int c_col,
    const float* __restrict__ states, const float* __restrict__ dcbh,
    const float* __restrict__ acs,
    int batch, int seq, int nh, int ng, int hd, int dstate
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int g = blockIdx.y;
    const int e = blockIdx.z * blockDim.x + threadIdx.x;
    if (bc >= (long long)batch * nc || g >= ng || e >= M2_CHUNK * dstate) return;
    const int t = e / dstate;
    const int n = e % dstate;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    if (t >= lc) return;
    const int hpg = nh / ng;
    const long long d_ssm = (long long)nh * hd;
    const long long rt = b * seq + t0 + t;
    float acc = 0.0f;
    for (int k = 0; k < hpg; k++) {
        const int h = g * hpg + k;
        const long long so = (bc * nh + h) * hd * (long long)dstate + n;
        float inner = 0.0f;
        for (int p = 0; p < hd; p++) {
            inner = __fmaf_rn(dy[rt * d_ssm + (long long)h * hd + p],
                              states[so + (long long)p * dstate], inner);
        }
        acc = __fmaf_rn(inner, m2_exp(acs[(bc * nh + h) * M2_CHUNK + t]), acc);
    }
    for (int s = 0; s <= t; s++) {
        float dcb = 0.0f;
        for (int k = 0; k < hpg; k++) {
            dcb = __fadd_rn(dcb, dcbh[(bc * nh + g * hpg + k) * M2B_PAIRS + m2b_pair(t, s)]);
        }
        const long long bs = (b * seq + t0 + s) * ld_xbc + b_col + (long long)g * dstate + n;
        acc = __fmaf_rn(dcb, to_f(xbc[bs]), acc);
    }
    dxbc[rt * ld_xbc + c_col + (long long)g * dstate + n] = acc;
}

#define DEFINE_M2B_DC(NAME, T)                                                  \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ dxbc,                                                   \
    const float* __restrict__ dy,                                               \
    const T* __restrict__ xbc,                                                  \
    long long ld_xbc,                                                           \
    int b_col,                                                                  \
    int c_col,                                                                  \
    const float* __restrict__ states,                                           \
    const float* __restrict__ dcbh,                                             \
    const float* __restrict__ acs,                                              \
    int batch,                                                                  \
    int seq,                                                                    \
    int nh,                                                                     \
    int ng,                                                                     \
    int hd,                                                                     \
    int dstate                                                                  \
) {                                                                             \
    m2b_dc_body<T>(dxbc, dy, xbc, ld_xbc, b_col, c_col, states, dcbh, acs,      \
        batch, seq, nh, ng, hd, dstate);                                        \
}
DEFINE_M2B_DC(m2b_dc, float)
DEFINE_M2B_DC(m2b_dc_bf16, __nv_bfloat16)
DEFINE_M2B_DC(m2b_dc_f16, __half)

// The decay gradients of a chunk and head, one thread per token k:
//   da_k = stable_k + (next_k + prev_k)
//   stable_k = sum_{t=k..L-1} sum_{s<k} M[t][s]     (t outer, s inner, ascending)
//   next_k   = sum_{t<k} Q_t                         (ascending)
//   prev_k   = I'_{L-1} + I_{L-2} + ... + I_k        (descending)
//   I'_{L-1} = I_{L-1} + e^Acs_last (H_c . dS[c])
// then ddt = da A + ddt_x, zero where the dt limit clamped, times the
// softplus slope (1 above 20), written into the dt column of d_proj; and
// per (b, c, h) the sums of da dt and of the dt-column gradient, k
// ascending. Grid (B * nc, H), block 64.
template <typename T>
__device__ __forceinline__ void m2b_combine_body(
    T* __restrict__ d_proj, long long ld_proj, int dt_off,
    float* __restrict__ da_parts, float* __restrict__ dbias_parts,
    const float* __restrict__ mm, const float* __restrict__ qv,
    const float* __restrict__ inter, const float* __restrict__ ddtx,
    const float* __restrict__ chunk_dot,
    const float* __restrict__ dt, const float* __restrict__ acs,
    const T* __restrict__ proj,
    const float* __restrict__ dt_bias, const float* __restrict__ a_neg,
    int batch, int seq, int nh, int has_limit, float dt_min, float dt_max
) {
    __shared__ float s_dadt[M2_CHUNK];
    __shared__ float s_draw[M2_CHUNK];
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int k = threadIdx.x;
    if (bc >= (long long)batch * nc || h >= nh) return;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    const long long sc = (bc * nh + h) * M2_CHUNK;
    const long long mo = (bc * nh + h) * M2B_PAIRS;
    float dadt = 0.0f, draw = 0.0f;
    if (k < lc) {
        float stable = 0.0f;
        for (int t = k; t < lc; t++)
            for (int s = 0; s < k; s++) stable = __fadd_rn(stable, mm[mo + m2b_pair(t, s)]);
        float nxt = 0.0f;
        for (int t = 0; t < k; t++) nxt = __fadd_rn(nxt, qv[sc + t]);
        float prv = __fadd_rn(inter[sc + lc - 1],
                              __fmul_rn(chunk_dot[bc * nh + h], m2_exp(acs[sc + lc - 1])));
        for (int t = lc - 2; t >= k; t--) prv = __fadd_rn(prv, inter[sc + t]);
        const float da = __fadd_rn(stable, __fadd_rn(nxt, prv));
        float ddt = __fmaf_rn(da, a_neg[h], ddtx[sc + k]);
        const long long row = b * seq + t0 + k;
        const float v = __fadd_rn(to_f(proj[row * ld_proj + dt_off + h]), dt_bias[h]);
        if (has_limit) {
            const float sp = m2_softplus(v);
            if (sp < dt_min || sp > dt_max) ddt = 0.0f;
        }
        draw = v > 20.0f ? ddt : __fmul_rn(ddt, m2b_sigmoid(v));
        m2b_put(&d_proj[row * ld_proj + dt_off + h], draw);
        dadt = __fmul_rn(da, dt[sc + k]);
    }
    s_dadt[k] = dadt;
    s_draw[k] = draw;
    __syncthreads();
    if (k == 0) {
        float a = 0.0f, d = 0.0f;
        for (int i = 0; i < lc; i++) {
            a = __fadd_rn(a, s_dadt[i]);
            d = __fadd_rn(d, s_draw[i]);
        }
        da_parts[bc * nh + h] = a;
        dbias_parts[bc * nh + h] = d;
    }
}

#define DEFINE_M2B_COMBINE(NAME, T)                                             \
extern "C" __global__ void NAME(                                                \
    T* __restrict__ d_proj,                                                     \
    long long ld_proj,                                                          \
    int dt_off,                                                                 \
    float* __restrict__ da_parts,                                               \
    float* __restrict__ dbias_parts,                                            \
    const float* __restrict__ mm,                                               \
    const float* __restrict__ qv,                                               \
    const float* __restrict__ inter,                                            \
    const float* __restrict__ ddtx,                                             \
    const float* __restrict__ chunk_dot,                                        \
    const float* __restrict__ dt,                                               \
    const float* __restrict__ acs,                                              \
    const T* __restrict__ proj,                                                 \
    const float* __restrict__ dt_bias,                                          \
    const float* __restrict__ a_neg,                                            \
    int batch,                                                                  \
    int seq,                                                                    \
    int nh,                                                                     \
    int has_limit,                                                              \
    float dt_min,                                                               \
    float dt_max                                                                \
) {                                                                             \
    m2b_combine_body<T>(d_proj, ld_proj, dt_off, da_parts, dbias_parts, mm,     \
        qv, inter, ddtx, chunk_dot, dt, acs, proj, dt_bias, a_neg, batch,       \
        seq, nh, has_limit, dt_min, dt_max);                                    \
}
DEFINE_M2B_COMBINE(m2b_combine, float)
DEFINE_M2B_COMBINE(m2b_combine_bf16, __nv_bfloat16)
DEFINE_M2B_COMBINE(m2b_combine_f16, __half)

// The per-head gradients: dt_bias += sum of the dt-column gradients,
// a_log += (sum of da dt) A, the (b, c) partials ascending.
// Grid 1-D over H.
extern "C" __global__ void m2b_head_grads(
    float* __restrict__ g_dt_bias, float* __restrict__ g_a_log,
    const float* __restrict__ da_parts, const float* __restrict__ dbias_parts,
    const float* __restrict__ a_neg, long long count, int nh
) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= nh) return;
    float a = 0.0f, d = 0.0f;
    for (long long q = 0; q < count; q++) {
        a = __fadd_rn(a, da_parts[q * nh + h]);
        d = __fadd_rn(d, dbias_parts[q * nh + h]);
    }
    g_dt_bias[h] = __fadd_rn(g_dt_bias[h], d);
    g_a_log[h] = __fmaf_rn(a, a_neg[h], g_a_log[h]);
}

// The D skip's partials per (b, c, h, p): sum_t dy x, t ascending.
// Grid (B * nc, H), block 128 striding p.
template <typename T>
__device__ __forceinline__ void m2b_d_parts_body(
    float* __restrict__ parts,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc,
    int batch, int seq, int nh, int hd
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    if (bc >= (long long)batch * nc || h >= nh) return;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    const long long d_ssm = (long long)nh * hd;
    for (int p = threadIdx.x; p < hd; p += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < lc; t++) {
            const long long row = b * seq + t0 + t;
            acc = __fmaf_rn(dy[row * d_ssm + (long long)h * hd + p],
                            to_f(xbc[row * ld_xbc + (long long)h * hd + p]), acc);
        }
        parts[(bc * nh + h) * hd + p] = acc;
    }
}

#define DEFINE_M2B_D_PARTS(NAME, T)                                             \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ parts,                                                  \
    const float* __restrict__ dy,                                               \
    const T* __restrict__ xbc,                                                  \
    long long ld_xbc,                                                           \
    int batch,                                                                  \
    int seq,                                                                    \
    int nh,                                                                     \
    int hd                                                                      \
) {                                                                             \
    m2b_d_parts_body<T>(parts, dy, xbc, ld_xbc, batch, seq, nh, hd);            \
}
DEFINE_M2B_D_PARTS(m2b_d_parts, float)
DEFINE_M2B_D_PARTS(m2b_d_parts_bf16, __nv_bfloat16)
DEFINE_M2B_D_PARTS(m2b_d_parts_f16, __half)

// D += the partials: per channel, the (b, c) partials ascending; per head,
// for each (b, c) the channels ascending, then those ascending.
// Grid 1-D over H * P (per channel) or H (per head).
extern "C" __global__ void m2b_d_reduce(
    float* __restrict__ g_d, const float* __restrict__ parts,
    long long count, int nh, int hd, int d_per_channel
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (d_per_channel ? nh * hd : nh)) return;
    float acc = 0.0f;
    if (d_per_channel) {
        for (long long q = 0; q < count; q++) acc = __fadd_rn(acc, parts[q * nh * hd + i]);
    } else {
        for (long long q = 0; q < count; q++) {
            float s = 0.0f;
            for (int p = 0; p < hd; p++) s = __fadd_rn(s, parts[(q * nh + i) * hd + p]);
            acc = __fadd_rn(acc, s);
        }
    }
    g_d[i] = __fadd_rn(g_d[i], acc);
}

// The grouped gated RMSNorm backward, one block per (row, group). The
// forward's u, its two-level sum and rstd are recomputed in the forward's
// order. With dout' = dout (or dout silu(z) with norm_before_gate),
// gw = w dout', xh = u rstd, c1 = sum(xh gw) / gs (per head, then the
// heads, ascending):
//   du = rstd (gw - xh c1),   dw_parts[r] = dout' xh
// gate first: dy = du silu(z), dz = du y silu'(z);
// norm first: dy = du,         dz = dout (xh w) silu'(z).
// dz is written into the z columns of d_proj. Dynamic shared memory:
// 2 (gs + heads) + heads floats.
template <typename T>
__device__ __forceinline__ void m2b_gated_norm_body(
    float* __restrict__ dy,
    T* __restrict__ d_proj, long long ld_proj, int z_off,
    float* __restrict__ dw_parts,
    const T* __restrict__ dout, long long ld_out, int out_col,
    const float* __restrict__ y,
    const T* __restrict__ proj,
    const float* __restrict__ w,
    long long rows, int d_ssm, int ng, int hd, float eps, int norm_before_gate
) {
    extern __shared__ float m2bn_sm[];
    __shared__ float scal[2];
    const long long r = blockIdx.x;
    const int g = blockIdx.y;
    if (r >= rows || g >= ng) return;
    const int gs = d_ssm / ng;
    const int heads = gs / hd;
    // Each head's row padded by one: the per-head sums (one thread a head)
    // then read distinct banks. Element i of the group sits at i + i / hd.
    float* u = m2bn_sm;
    float* gw = u + gs + heads;
    float* head_sum = gw + gs + heads;
    const long long c0 = (long long)g * gs;
    for (int i = threadIdx.x; i < gs; i += blockDim.x) {
        float yv = y[r * d_ssm + c0 + i];
        float zv = to_f(proj[r * ld_proj + z_off + c0 + i]);
        u[i + i / hd] = norm_before_gate ? yv : __fmul_rn(yv, m2_silu(zv));
    }
    __syncthreads();
    for (int hh = threadIdx.x; hh < heads; hh += blockDim.x) {
        float s = 0.0f;
        for (int p = 0; p < hd; p++) {
            float v = u[hh * (hd + 1) + p];
            s = __fmaf_rn(v, v, s);
        }
        head_sum[hh] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int hh = 0; hh < heads; hh++) tot = __fadd_rn(tot, head_sum[hh]);
        float rs = __fdiv_rn(1.0f, sqrtf(__fadd_rn(__fdiv_rn(tot, (float)gs), eps)));
        scal[0] = isfinite(rs) ? rs : 0.0f;
    }
    __syncthreads();
    const float rs = scal[0];
    for (int i = threadIdx.x; i < gs; i += blockDim.x) {
        const float d = to_f(dout[r * ld_out + out_col + c0 + i]);
        const float wv = w[c0 + i];
        const float xh = __fmul_rn(u[i + i / hd], rs);
        float d1 = d;
        if (norm_before_gate) {
            const float zv = to_f(proj[r * ld_proj + z_off + c0 + i]);
            d1 = __fmul_rn(d, m2_silu(zv));
            m2b_put(&d_proj[r * ld_proj + z_off + c0 + i],
                    __fmul_rn(__fmul_rn(d, __fmul_rn(xh, wv)), m2b_silu_grad(zv)));
        }
        gw[i + i / hd] = __fmul_rn(wv, d1);
        dw_parts[r * d_ssm + c0 + i] = __fmul_rn(d1, xh);
    }
    __syncthreads();
    for (int hh = threadIdx.x; hh < heads; hh += blockDim.x) {
        float s = 0.0f;
        for (int p = 0; p < hd; p++) {
            const int i = hh * (hd + 1) + p;
            s = __fmaf_rn(__fmul_rn(u[i], rs), gw[i], s);
        }
        head_sum[hh] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int hh = 0; hh < heads; hh++) tot = __fadd_rn(tot, head_sum[hh]);
        scal[1] = __fdiv_rn(tot, (float)gs);
    }
    __syncthreads();
    const float c1 = scal[1];
    for (int i = threadIdx.x; i < gs; i += blockDim.x) {
        const float xh = __fmul_rn(u[i + i / hd], rs);
        const float du = __fmul_rn(rs, __fsub_rn(gw[i + i / hd], __fmul_rn(xh, c1)));
        if (norm_before_gate) {
            dy[r * d_ssm + c0 + i] = du;
        } else {
            const float yv = y[r * d_ssm + c0 + i];
            const float zv = to_f(proj[r * ld_proj + z_off + c0 + i]);
            dy[r * d_ssm + c0 + i] = __fmul_rn(du, m2_silu(zv));
            m2b_put(&d_proj[r * ld_proj + z_off + c0 + i],
                    __fmul_rn(__fmul_rn(du, yv), m2b_silu_grad(zv)));
        }
    }
}

#define DEFINE_M2B_GATED_NORM(NAME, T)                                          \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ dy,                                                     \
    T* __restrict__ d_proj,                                                     \
    long long ld_proj,                                                          \
    int z_off,                                                                  \
    float* __restrict__ dw_parts,                                               \
    const T* __restrict__ dout,                                                 \
    long long ld_out,                                                           \
    int out_col,                                                                \
    const float* __restrict__ y,                                                \
    const T* __restrict__ proj,                                                 \
    const float* __restrict__ w,                                                \
    long long rows,                                                             \
    int d_ssm,                                                                  \
    int ng,                                                                     \
    int hd,                                                                     \
    float eps,                                                                  \
    int norm_before_gate                                                        \
) {                                                                             \
    m2b_gated_norm_body<T>(dy, d_proj, ld_proj, z_off, dw_parts, dout,          \
        ld_out, out_col, y, proj, w, rows, d_ssm, ng, hd, eps,                  \
        norm_before_gate);                                                      \
}
DEFINE_M2B_GATED_NORM(m2b_gated_norm, float)
DEFINE_M2B_GATED_NORM(m2b_gated_norm_bf16, __nv_bfloat16)
DEFINE_M2B_GATED_NORM(m2b_gated_norm_f16, __half)

// The gate without a norm: out = y silu(z). Grid 1-D over rows * d_ssm.
template <typename T>
__device__ __forceinline__ void m2b_gate_only_body(
    float* __restrict__ dy,
    T* __restrict__ d_proj, long long ld_proj, int z_off,
    const T* __restrict__ dout, long long ld_out, int out_col,
    const float* __restrict__ y,
    const T* __restrict__ proj,
    long long rows, int d_ssm
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * d_ssm) return;
    long long r = i / d_ssm;
    long long c = i - r * d_ssm;
    const float d = to_f(dout[r * ld_out + out_col + c]);
    const float zv = to_f(proj[r * ld_proj + z_off + c]);
    dy[i] = __fmul_rn(d, m2_silu(zv));
    m2b_put(&d_proj[r * ld_proj + z_off + c], __fmul_rn(__fmul_rn(d, y[i]), m2b_silu_grad(zv)));
}

#define DEFINE_M2B_GATE_ONLY(NAME, T)                                           \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ dy,                                                     \
    T* __restrict__ d_proj,                                                     \
    long long ld_proj,                                                          \
    int z_off,                                                                  \
    const T* __restrict__ dout,                                                 \
    long long ld_out,                                                           \
    int out_col,                                                                \
    const float* __restrict__ y,                                                \
    const T* __restrict__ proj,                                                 \
    long long rows,                                                             \
    int d_ssm                                                                   \
) {                                                                             \
    m2b_gate_only_body<T>(dy, d_proj, ld_proj, z_off, dout, ld_out,             \
        out_col, y, proj, rows, d_ssm);                                         \
}
DEFINE_M2B_GATE_ONLY(m2b_gate_only, float)
DEFINE_M2B_GATE_ONLY(m2b_gate_only_bf16, __nv_bfloat16)
DEFINE_M2B_GATE_ONLY(m2b_gate_only_f16, __half)

// The mixer's inner MLP gate, out = silu(z0) x0: dz0 = dout x0 silu'(z0)
// and dx0 = dout silu(z0) into the first 2 d_mlp columns of d_proj.
// Grid 1-D over rows * d_mlp.
template <typename T>
__device__ __forceinline__ void m2b_mlp_gate_body(
    T* __restrict__ d_proj, long long ld_proj,
    const T* __restrict__ dout, long long ld_out,
    const T* __restrict__ proj,
    long long rows, int d_mlp
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * d_mlp) return;
    long long r = i / d_mlp;
    long long j = i - r * d_mlp;
    const float d = to_f(dout[r * ld_out + j]);
    const float z0 = to_f(proj[r * ld_proj + j]);
    const float x0 = to_f(proj[r * ld_proj + d_mlp + j]);
    m2b_put(&d_proj[r * ld_proj + j], __fmul_rn(__fmul_rn(d, x0), m2b_silu_grad(z0)));
    m2b_put(&d_proj[r * ld_proj + d_mlp + j], __fmul_rn(d, m2_silu(z0)));
}

#define DEFINE_M2B_MLP_GATE(NAME, T)                                            \
extern "C" __global__ void NAME(                                                \
    T* __restrict__ d_proj,                                                     \
    long long ld_proj,                                                          \
    const T* __restrict__ dout,                                                 \
    long long ld_out,                                                           \
    const T* __restrict__ proj,                                                 \
    long long rows,                                                             \
    int d_mlp                                                                   \
) {                                                                             \
    m2b_mlp_gate_body<T>(d_proj, ld_proj, dout, ld_out, proj, rows, d_mlp);     \
}
DEFINE_M2B_MLP_GATE(m2b_mlp_gate, float)
DEFINE_M2B_MLP_GATE(m2b_mlp_gate_bf16, __nv_bfloat16)
DEFINE_M2B_MLP_GATE(m2b_mlp_gate_f16, __half)

// The conv's pre-activation gradient dv = dxbc silu'(v), v recomputed in
// the forward's chain (bias, then the taps oldest first, the taps before
// the window from the state the window started from).
// Grid 1-D over B * T * conv_dim.
template <typename T>
__device__ __forceinline__ void m2b_conv_dv_body(
    float* __restrict__ dv,
    const float* __restrict__ dxbc,
    const float* __restrict__ state0,
    const T* __restrict__ proj, long long ld_proj, int xbc_col,
    const float* __restrict__ w, const float* __restrict__ bias,
    int batch, int seq, int conv_dim, int d_conv
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)batch * seq * conv_dim;
    if (i >= total) return;
    const int c = (int)(i % conv_dim);
    const long long row = i / conv_dim;
    const int t = (int)(row % seq);
    const int b = (int)(row / seq);
    float v = bias[c];
    for (int k = 0; k < d_conv; k++) {
        int src = t - (d_conv - 1) + k;
        float xv = src >= 0
            ? to_f(proj[((long long)b * seq + src) * ld_proj + xbc_col + c])
            : state0[((long long)b * conv_dim + c) * d_conv + d_conv + src];
        v = __fmaf_rn(w[(long long)c * d_conv + k], xv, v);
    }
    dv[i] = __fmul_rn(dxbc[i], m2b_silu_grad(v));
}

#define DEFINE_M2B_CONV_DV(NAME, T)                                             \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ dv,                                                     \
    const float* __restrict__ dxbc,                                             \
    const float* __restrict__ state0,                                           \
    const T* __restrict__ proj,                                                 \
    long long ld_proj,                                                          \
    int xbc_col,                                                                \
    const float* __restrict__ w,                                                \
    const float* __restrict__ bias,                                             \
    int batch,                                                                  \
    int seq,                                                                    \
    int conv_dim,                                                               \
    int d_conv                                                                  \
) {                                                                             \
    m2b_conv_dv_body<T>(dv, dxbc, state0, proj, ld_proj, xbc_col, w, bias,      \
        batch, seq, conv_dim, d_conv);                                          \
}
DEFINE_M2B_CONV_DV(m2b_conv_dv, float)
DEFINE_M2B_CONV_DV(m2b_conv_dv_bf16, __nv_bfloat16)
DEFINE_M2B_CONV_DV(m2b_conv_dv_f16, __half)

// The conv's input gradient into the xBC columns of d_proj:
// d_in[t] = sum_k w[k] dv[t + d_conv - 1 - k] over taps inside the window,
// k ascending. Grid 1-D over B * T * conv_dim.
template <typename T>
__device__ __forceinline__ void m2b_conv_dx_body(
    T* __restrict__ d_proj, long long ld_proj, int xbc_col,
    const float* __restrict__ dv,
    const float* __restrict__ w,
    int batch, int seq, int conv_dim, int d_conv
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)batch * seq * conv_dim;
    if (i >= total) return;
    const int c = (int)(i % conv_dim);
    const long long row = i / conv_dim;
    const int t = (int)(row % seq);
    const long long b = row / seq;
    float acc = 0.0f;
    for (int k = 0; k < d_conv; k++) {
        int dst = t + (d_conv - 1) - k;
        if (dst < seq)
            acc = __fmaf_rn(w[(long long)c * d_conv + k],
                            dv[(b * seq + dst) * conv_dim + c], acc);
    }
    m2b_put(&d_proj[row * ld_proj + xbc_col + c], acc);
}

#define DEFINE_M2B_CONV_DX(NAME, T)                                             \
extern "C" __global__ void NAME(                                                \
    T* __restrict__ d_proj,                                                     \
    long long ld_proj,                                                          \
    int xbc_col,                                                                \
    const float* __restrict__ dv,                                               \
    const float* __restrict__ w,                                                \
    int batch,                                                                  \
    int seq,                                                                    \
    int conv_dim,                                                               \
    int d_conv                                                                  \
) {                                                                             \
    m2b_conv_dx_body<T>(d_proj, ld_proj, xbc_col, dv, w, batch, seq,            \
        conv_dim, d_conv);                                                      \
}
DEFINE_M2B_CONV_DX(m2b_conv_dx, float)
DEFINE_M2B_CONV_DX(m2b_conv_dx_bf16, __nv_bfloat16)
DEFINE_M2B_CONV_DX(m2b_conv_dx_f16, __half)

// The conv weight and bias gradients, one thread per (c, k), k == d_conv
// standing for the bias: rows ascending (sequence, then token). A block
// takes 32 channels (a lane each, a warp per k) and streams the rows in
// tiles of M2B_DW_ROWS through shared memory: dv [rows][32] and the conv
// inputs [rows + d_conv - 1][32] from the row d_conv - 1 before the tile
// (a token's taps inside its sequence are those rows; the taps before a
// sequence's first token come from state0). Grid ceil(conv_dim / 32),
// block (32, d_conv + 1); `g_b` may be null.
#define M2B_DW_ROWS 128
template <typename T>
__device__ __forceinline__ void m2b_conv_dw_body(
    float* __restrict__ g_w, float* __restrict__ g_b,
    const float* __restrict__ dv,
    const float* __restrict__ state0,
    const T* __restrict__ proj, long long ld_proj, int xbc_col,
    int batch, int seq, int conv_dim, int d_conv
) {
    extern __shared__ float m2bc_sm[];
    constexpr int CH = 32;
    const int halo = d_conv - 1;
    float* dvs = m2bc_sm;                    // [M2B_DW_ROWS][CH]
    float* xs = dvs + M2B_DW_ROWS * CH;      // [M2B_DW_ROWS + halo][CH]
    const int c0 = blockIdx.x * CH;
    const int lane = threadIdx.x, k = threadIdx.y;
    const int c = c0 + lane;
    const int tid = threadIdx.y * CH + lane, nthr = CH * blockDim.y;
    const long long rows = (long long)batch * seq;
    const bool live = c < conv_dim && (k < d_conv || g_b != nullptr);
    float acc = 0.0f;
    long long b = 0;
    int t = 0;
    for (long long r0 = 0; r0 < rows; r0 += M2B_DW_ROWS) {
        const int rn = (int)min((long long)M2B_DW_ROWS, rows - r0);
        __syncthreads();
        m2_stage4((rn + halo) * CH, tid, nthr,
                  [&](int e) {
                      const int rr = e / CH, ch = c0 + e % CH;
                      const long long xrow = r0 - halo + rr;
                      return xrow >= 0 && ch < conv_dim
                          ? to_f(proj[xrow * ld_proj + xbc_col + ch]) : 0.0f;
                  },
                  [&](int e, float v) { xs[e] = v; });
        m2_stage4(rn * CH, tid, nthr,
                  [&](int e) {
                      const int rr = e / CH, ch = c0 + e % CH;
                      return ch < conv_dim ? dv[(r0 + rr) * conv_dim + ch] : 0.0f;
                  },
                  [&](int e, float v) { dvs[e] = v; });
        __syncthreads();
        if (live) {
            for (int rr = 0; rr < rn; rr++) {
                const float g = dvs[rr * CH + lane];
                if (k == d_conv) {
                    acc = __fadd_rn(acc, g);
                } else {
                    const int src = t - halo + k;
                    const float xv = src >= 0
                        ? xs[(rr + k) * CH + lane]
                        : state0[(b * conv_dim + c) * d_conv + d_conv + src];
                    acc = __fmaf_rn(xv, g, acc);
                }
                if (++t == seq) {
                    t = 0;
                    b++;
                }
            }
        }
    }
    if (!live) return;
    if (k == d_conv) {
        g_b[c] = __fadd_rn(g_b[c], acc);
    } else {
        g_w[(long long)c * d_conv + k] = __fadd_rn(g_w[(long long)c * d_conv + k], acc);
    }
}

#define DEFINE_M2B_CONV_DW(NAME, T)                                             \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ g_w,                                                    \
    float* __restrict__ g_b,                                                    \
    const float* __restrict__ dv,                                               \
    const float* __restrict__ state0,                                           \
    const T* __restrict__ proj,                                                 \
    long long ld_proj,                                                          \
    int xbc_col,                                                                \
    int batch,                                                                  \
    int seq,                                                                    \
    int conv_dim,                                                               \
    int d_conv                                                                  \
) {                                                                             \
    m2b_conv_dw_body<T>(g_w, g_b, dv, state0, proj, ld_proj, xbc_col,           \
        batch, seq, conv_dim, d_conv);                                          \
}
DEFINE_M2B_CONV_DW(m2b_conv_dw, float)
DEFINE_M2B_CONV_DW(m2b_conv_dw_bf16, __nv_bfloat16)
DEFINE_M2B_CONV_DW(m2b_conv_dw_f16, __half)

// A bias gradient from typed rows: g[j] += sum_r dy[r][j], rows ascending.
// Grid 1-D over n. (The f32 lane uses the Fixed column sum.)
template <typename T>
__device__ __forceinline__ void m2b_bias_grad_body(
    float* __restrict__ g, const T* __restrict__ dy, long long rows, int n
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float acc = 0.0f;
    for (long long r = 0; r < rows; r++) acc = __fadd_rn(acc, to_f(dy[r * n + j]));
    g[j] = __fadd_rn(g[j], acc);
}

#define DEFINE_M2B_BIAS_GRAD(NAME, T)                                           \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ g, const T* __restrict__ dy, long long rows, int n      \
) {                                                                             \
    m2b_bias_grad_body<T>(g, dy, rows, n);                                      \
}
DEFINE_M2B_BIAS_GRAD(m2b_bias_grad_bf16, __nv_bfloat16)
DEFINE_M2B_BIAS_GRAD(m2b_bias_grad_f16, __half)

// The sequential path's backward of the recurrence (ScanMode::Sequential),
// one block per (sequence, head), a whole number of warps covering the
// head's channels. Each segment of `stride` tokens is replayed from the
// checkpoint the forward wrote, through the forward's own step pieces, so
// the replayed states are the forward's bits; the states after each token
// are recorded, then the segment is walked back with the state gradient
// dS in shared memory, starting from zero after the window:
//   dS[p] += dy[p] C,  dC = sum_p dy[p] h_t[p],  dB = dt sum_p dS[p] x[p],
//   dx = dt (dS[p] . B) + D dy,  ddt = sum_p x (dS[p] . B),
//   dalpha = sum dS h_{t-1},  dS <- alpha dS,
// then ddt += dalpha alpha A, the dt limit's mask and the softplus slope
// into the dt column, and per (b, h) the sums of dalpha alpha dt and of
// the dt-column gradient. dB and dC go per head to db_h / dc_h; the
// group's heads are summed by m2s_bc_groups. Dynamic shared memory:
// 2 P (N + 1) + 2 N + 4 + 2 P + blockDim floats.
template <typename T>
__device__ __forceinline__ void m2s_bwd_body(
    float* __restrict__ dxbc,
    T* __restrict__ d_proj, long long ld_proj, int dt_off,
    float* __restrict__ db_h, float* __restrict__ dc_h,
    float* __restrict__ da_parts, float* __restrict__ dbias_parts,
    float* __restrict__ rec,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, const T* __restrict__ proj,
    const float* __restrict__ ckpt,
    const float* __restrict__ dt_bias, const float* __restrict__ a_neg,
    const float* __restrict__ D,
    int seq, int stride, int nh, int ng, int hd, int ds, int conv_dim,
    int flags, float dt_min, float dt_max
) {
    extern __shared__ float m2sb_sm[];
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int g = h / (nh / ng);
    const long long d_ssm = (long long)nh * hd;
    const long long bh = (long long)b * nh + h;
    const int nseg = (seq + stride - 1) / stride;
    const long long pn = (long long)hd * ds;
    float* st = m2sb_sm;
    float* dsr = st + hd * (ds + 1);
    float* b_sm = dsr + hd * (ds + 1);
    float* c_sm = b_sm + ds;
    float* scal = c_sm + ds;
    float* xS = scal + 4;
    float* dyS = xS + hd;
    float* red = dyS + hd;
    const bool own = tid < hd;
    const float dval = own ? ((flags & M2_FLAG_D_PER_CHANNEL) ? D[(long long)h * hd + tid] : D[h]) : 0.0f;
    float da_acc = 0.0f, dbias_acc = 0.0f;
    for (int p = 0; p < hd; p++)
        for (int n = tid; n < ds; n += blockDim.x) dsr[p * (ds + 1) + n] = 0.0f;
    __syncthreads();
    for (int seg = nseg - 1; seg >= 0; seg--) {
        const int t0 = seg * stride;
        const int t1 = min(t0 + stride, seq);
        const float* ck = ckpt + (((long long)b * nseg + seg) * nh + h) * pn;
        m2_tile_load(st, ck, hd, ds);
        __syncthreads();
        for (int t = t0; t < t1; t++) {
            const long long row = (long long)b * seq + t;
            const long long xrow = row * conv_dim;
            for (int n = tid; n < ds; n += blockDim.x) {
                b_sm[n] = to_f(xbc[xrow + d_ssm + (long long)g * ds + n]);
                c_sm[n] = to_f(xbc[xrow + d_ssm + (long long)(ng + g) * ds + n]);
            }
            __syncthreads();
            m2_step_scalars(scal, b_sm, c_sm, proj + row * ld_proj, dt_off, h, dt_bias,
                            a_neg, ds, flags, dt_min, dt_max);
            __syncthreads();
            if (own) {
                const float xp = to_f(xbc[xrow + (long long)h * hd + tid]);
                float* trow = st + tid * (ds + 1);
                m2_row_step(trow, b_sm, c_sm, scal[0], scal[1], scal[2], xp, ds);
            }
            __syncthreads();
            m2_tile_store(rec + (bh * stride + (t - t0)) * pn, st, hd, ds);
        }
        __syncthreads();
        for (int t = t1 - 1; t >= t0; t--) {
            const long long row = (long long)b * seq + t;
            const long long xrow = row * conv_dim;
            const int slot = t - t0;
            const float* h_cur = rec + (bh * stride + slot) * pn;
            const float* h_prev = slot > 0 ? rec + (bh * stride + slot - 1) * pn : ck;
            // The state before the token into the replay tile (unused by the
            // walk back), so each channel's row reads from shared memory.
            m2_tile_load(st, h_prev, hd, ds);
            for (int n = tid; n < ds; n += blockDim.x) {
                b_sm[n] = to_f(xbc[xrow + d_ssm + (long long)g * ds + n]);
                c_sm[n] = to_f(xbc[xrow + d_ssm + (long long)(ng + g) * ds + n]);
            }
            __syncthreads();
            m2_step_scalars(scal, b_sm, c_sm, proj + row * ld_proj, dt_off, h, dt_bias,
                            a_neg, ds, flags, dt_min, dt_max);
            __syncthreads();
            const float alpha = scal[0], dt = scal[2];
            float dyv = 0.0f, xp = 0.0f;
            if (own) {
                dyv = dy[row * d_ssm + (long long)h * hd + tid];
                xp = to_f(xbc[xrow + (long long)h * hd + tid]);
                xS[tid] = xp;
                dyS[tid] = dyv;
                float* drow = dsr + tid * (ds + 1);
                for (int n = 0; n < ds; n++) drow[n] = __fmaf_rn(dyv, c_sm[n], drow[n]);
            }
            __syncthreads();
            for (int n = tid; n < ds; n += blockDim.x) {
                float dc = 0.0f, db = 0.0f;
                for (int p = 0; p < hd; p++) {
                    dc = __fmaf_rn(dyS[p], h_cur[(long long)p * ds + n], dc);
                    db = __fmaf_rn(dsr[p * (ds + 1) + n], xS[p], db);
                }
                dc_h[(row * nh + h) * ds + n] = dc;
                db_h[(row * nh + h) * ds + n] = __fmul_rn(db, dt);
            }
            __syncthreads();
            float ddt_loc = 0.0f, dal_loc = 0.0f;
            if (own) {
                float* drow = dsr + tid * (ds + 1);
                float sx = 0.0f;
                for (int n = 0; n < ds; n++) {
                    const float dsv = drow[n];
                    sx = __fmaf_rn(dsv, b_sm[n], sx);
                    dal_loc = __fmaf_rn(dsv, st[tid * (ds + 1) + n], dal_loc);
                    drow[n] = __fmul_rn(alpha, dsv);
                }
                dxbc[xrow + (long long)h * hd + tid] = __fmaf_rn(dt, sx, __fmul_rn(dval, dyv));
                ddt_loc = __fmul_rn(xp, sx);
            }
            red[tid] = ddt_loc;
            __syncthreads();
            int width = 1;
            while (width < (int)blockDim.x) width <<= 1;
            for (int w = width >> 1; w > 0; w >>= 1) {
                if (tid < w && tid + w < (int)blockDim.x) red[tid] = __fadd_rn(red[tid], red[tid + w]);
                __syncthreads();
            }
            const float ddt_x = red[0];
            __syncthreads();
            red[tid] = dal_loc;
            __syncthreads();
            for (int w = width >> 1; w > 0; w >>= 1) {
                if (tid < w && tid + w < (int)blockDim.x) red[tid] = __fadd_rn(red[tid], red[tid + w]);
                __syncthreads();
            }
            const float dalpha = red[0];
            if (tid == 0) {
                const float da = __fmul_rn(dalpha, alpha);
                float ddt = __fmaf_rn(da, a_neg[h], ddt_x);
                da_acc = __fmaf_rn(da, dt, da_acc);
                const float v = __fadd_rn(to_f(proj[row * ld_proj + dt_off + h]), dt_bias[h]);
                if (flags & M2_FLAG_DT_LIMIT) {
                    const float sp = m2_softplus(v);
                    if (sp < dt_min || sp > dt_max) ddt = 0.0f;
                }
                const float draw = v > 20.0f ? ddt : __fmul_rn(ddt, m2b_sigmoid(v));
                m2b_put(&d_proj[row * ld_proj + dt_off + h], draw);
                dbias_acc = __fadd_rn(dbias_acc, draw);
            }
            __syncthreads();
        }
    }
    if (tid == 0) {
        da_parts[bh] = da_acc;
        dbias_parts[bh] = dbias_acc;
    }
}

#define DEFINE_M2S_BWD(NAME, T)                                                 \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ dxbc,                                                   \
    T* __restrict__ d_proj, long long ld_proj, int dt_off,                      \
    float* __restrict__ db_h, float* __restrict__ dc_h,                         \
    float* __restrict__ da_parts, float* __restrict__ dbias_parts,              \
    float* __restrict__ rec,                                                    \
    const float* __restrict__ dy,                                               \
    const T* __restrict__ xbc, const T* __restrict__ proj,                      \
    const float* __restrict__ ckpt,                                             \
    const float* __restrict__ dt_bias, const float* __restrict__ a_neg,         \
    const float* __restrict__ D,                                                \
    int seq, int stride, int nh, int ng, int hd, int ds, int conv_dim,          \
    int flags, float dt_min, float dt_max                                       \
) {                                                                             \
    m2s_bwd_body<T>(dxbc, d_proj, ld_proj, dt_off, db_h, dc_h, da_parts,        \
        dbias_parts, rec, dy, xbc, proj, ckpt, dt_bias, a_neg, D, seq, stride,  \
        nh, ng, hd, ds, conv_dim, flags, dt_min, dt_max);                       \
}
DEFINE_M2S_BWD(m2s_bwd, float)
DEFINE_M2S_BWD(m2s_bwd_bf16, __nv_bfloat16)
DEFINE_M2S_BWD(m2s_bwd_f16, __half)

// The step scalars of every (row, head) for the sequential backward, with
// m2_step_scalars' own expressions: [rows, H, 3] = (e^acs, (C.B) e^0 dt,
// e^0 dt). One thread per (row, head); grid 1-D over rows * H.
template <typename T>
__device__ __forceinline__ void m2s_scalars_body(
    float* __restrict__ out,
    const T* __restrict__ xbc, const T* __restrict__ proj, long long ld_proj, int dt_off,
    const float* __restrict__ dt_bias, const float* __restrict__ a_neg,
    long long rows, int nh, int ng, int hd, int ds, int conv_dim,
    int flags, float dt_min, float dt_max
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * nh) return;
    const long long row = i / nh;
    const int h = (int)(i % nh);
    const int g = h / (nh / ng);
    const long long base = row * conv_dim + (long long)nh * hd;
    const T* bv = xbc + base + (long long)g * ds;
    const T* cv = xbc + base + (long long)(ng + g) * ds;
    float dt = m2_softplus(__fadd_rn(to_f(proj[row * ld_proj + dt_off + h]), dt_bias[h]));
    if ((flags & M2_FLAG_DT_LIMIT) && dt == dt) dt = fminf(fmaxf(dt, dt_min), dt_max);
    const float acs = __fadd_rn(0.0f, __fmul_rn(dt, a_neg[h]));
    float cb = 0.0f;
    for (int n = 0; n < ds; n++) cb = __fmaf_rn(to_f(cv[n]), to_f(bv[n]), cb);
    const float e0 = m2_exp(fminf(__fsub_rn(acs, acs), 0.0f));
    out[i * 3] = m2_exp(acs);
    out[i * 3 + 1] = __fmul_rn(__fmul_rn(cb, e0), dt);
    out[i * 3 + 2] = __fmul_rn(e0, dt);
}

#define DEFINE_M2S_SCALARS(NAME, T)                                             \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ out,                                                    \
    const T* __restrict__ xbc, const T* __restrict__ proj, long long ld_proj,   \
    int dt_off, const float* __restrict__ dt_bias,                              \
    const float* __restrict__ a_neg, long long rows, int nh, int ng, int hd,    \
    int ds, int conv_dim, int flags, float dt_min, float dt_max                 \
) {                                                                             \
    m2s_scalars_body<T>(out, xbc, proj, ld_proj, dt_off, dt_bias, a_neg, rows,  \
        nh, ng, hd, ds, conv_dim, flags, dt_min, dt_max);                       \
}
DEFINE_M2S_SCALARS(m2s_scalars, float)
DEFINE_M2S_SCALARS(m2s_scalars_bf16, __nv_bfloat16)
DEFINE_M2S_SCALARS(m2s_scalars_f16, __half)

// A 4-byte asynchronous copy into shared memory and its fences.
__device__ __forceinline__ void m2s_cp4(float* dst, const float* src) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(src));
}
__device__ __forceinline__ void m2s_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void m2s_cp_wait() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// A head's state from dense global rows into shared rows padded to N + 1,
// row by row, asynchronously.
__device__ __forceinline__ void m2s_tile_fetch(
    float* __restrict__ sh, const float* __restrict__ gl, int hd, int ds
) {
    for (int p = 0; p < hd; p++)
        for (int n = threadIdx.x; n < ds; n += blockDim.x)
            m2s_cp4(sh + p * (ds + 1) + n, gl + (long long)p * ds + n);
    m2s_cp_commit();
}

// Two sums of the values of the first `n` threads (a whole number of warps)
// through mimo-style fixed trees at once (the tree of m2s_bwd: width the
// power of two at or above n, then red[t] += red[t + w] for w halving), the
// levels above a warp in shared memory and the last five by shuffles in
// warp 0, which pair the same operands in the same order. The sums land in
// thread 0.
__device__ __forceinline__ float2 m2s_block_sum2(float a, float b, float* red, int n) {
    const int t = threadIdx.x;
    if (t < n) {
        red[t] = a;
        red[n + t] = b;
    }
    __syncthreads();
    int width = 1;
    while (width < n) width <<= 1;
    for (int w = width >> 1; w >= 32; w >>= 1) {
        if (t < w && t + w < n) {
            red[t] = __fadd_rn(red[t], red[t + w]);
            red[n + t] = __fadd_rn(red[n + t], red[n + t + w]);
        }
        __syncthreads();
    }
    float2 out = make_float2(0.0f, 0.0f);
    if (t < 32) {
        float x = red[t], y = red[n + t];
#pragma unroll
        for (int w = 16; w > 0; w >>= 1) {
            const float xo = __shfl_down_sync(0xffffffffu, x, w);
            const float yo = __shfl_down_sync(0xffffffffu, y, w);
            if (t < w && t + w < n) {
                x = __fadd_rn(x, xo);
                y = __fadd_rn(y, yo);
            }
        }
        out = make_float2(x, y);
    }
    return out;
}

// m2s_bwd with the token's states in shared memory: three padded tiles
// (the state after the token, the state before it, dS), the scalars read
// from m2s_scalars' rows. The replay runs m2_row_step on the first tile
// and records each state as m2s_bwd does; the walk back reads the state
// after the token from its tile (the replay's last state, then the tile the
// previous step loaded) and fetches the state before the next token
// asynchronously into the tile it just finished with. The block may be
// wider than the channels: the dS update, the dB and dC chains and the
// tile moves spread over all its threads, the row chains stay on the
// channel threads and the sums keep the channel threads' width. Every
// chain and both trees are m2s_bwd's, so the bits are its own. Dynamic
// shared memory: 3 P (N + 1) + 2 N + 2 P + 2 lanes floats (lanes = P
// rounded up to a warp).
template <typename T>
__device__ __forceinline__ void m2s_bwd_tiles_body(
    float* __restrict__ dxbc,
    T* __restrict__ d_proj, long long ld_proj, int dt_off,
    float* __restrict__ db_h, float* __restrict__ dc_h,
    float* __restrict__ da_parts, float* __restrict__ dbias_parts,
    float* __restrict__ rec,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, const T* __restrict__ proj,
    const float* __restrict__ ckpt, const float* __restrict__ scal3,
    const float* __restrict__ dt_bias, const float* __restrict__ a_neg,
    const float* __restrict__ D,
    int seq, int stride, int nh, int ng, int hd, int ds, int conv_dim,
    int flags, float dt_min, float dt_max
) {
    extern __shared__ float m2sb_sm[];
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int g = h / (nh / ng);
    const long long d_ssm = (long long)nh * hd;
    const long long bh = (long long)b * nh + h;
    const int nseg = (seq + stride - 1) / stride;
    const long long pn = (long long)hd * ds;
    const int ld = ds + 1;
    float* tile0 = m2sb_sm;
    float* tile1 = tile0 + hd * ld;
    float* dsr = tile1 + hd * ld;
    float* b_sm = dsr + hd * ld;
    float* c_sm = b_sm + ds;
    float* xS = c_sm + ds;
    float* dyS = xS + hd;
    float* red = dyS + hd;
    const bool own = tid < hd;
    // The channel threads, a warp multiple: the width of the block sums.
    const int lanes = (hd + 31) & ~31;
    const float dval = own ? ((flags & M2_FLAG_D_PER_CHANNEL) ? D[(long long)h * hd + tid] : D[h]) : 0.0f;
    float da_acc = 0.0f, dbias_acc = 0.0f;
    for (int p = 0; p < hd; p++)
        for (int n = tid; n < ds; n += blockDim.x) dsr[p * ld + n] = 0.0f;
    __syncthreads();
    for (int seg = nseg - 1; seg >= 0; seg--) {
        const int t0 = seg * stride;
        const int t1 = min(t0 + stride, seq);
        const float* ck = ckpt + (((long long)b * nseg + seg) * nh + h) * pn;
        float* st = tile0;
        m2_tile_load(st, ck, hd, ds);
        __syncthreads();
        for (int t = t0; t < t1; t++) {
            const long long row = (long long)b * seq + t;
            const long long xrow = row * conv_dim;
            for (int n = tid; n < ds; n += blockDim.x) {
                b_sm[n] = to_f(xbc[xrow + d_ssm + (long long)g * ds + n]);
                c_sm[n] = to_f(xbc[xrow + d_ssm + (long long)(ng + g) * ds + n]);
            }
            __syncthreads();
            if (own) {
                const float* sc = scal3 + (row * nh + h) * 3;
                const float xp = to_f(xbc[xrow + (long long)h * hd + tid]);
                m2_row_step(st + tid * ld, b_sm, c_sm, sc[0], sc[1], sc[2], xp, ds);
            }
            __syncthreads();
            m2_tile_store(rec + (bh * stride + (t - t0)) * pn, st, hd, ds);
        }
        // The state before the last token into the other tile; the replay's
        // last state is the one after it.
        float* cur = tile0;
        float* prev = tile1;
        const int last = t1 - 1 - t0;
        m2s_tile_fetch(prev, last > 0 ? rec + (bh * stride + last - 1) * pn : ck, hd, ds);
        m2s_cp_wait();
        __syncthreads();
        for (int t = t1 - 1; t >= t0; t--) {
            const long long row = (long long)b * seq + t;
            const long long xrow = row * conv_dim;
            const int slot = t - t0;
            for (int n = tid; n < ds; n += blockDim.x) {
                b_sm[n] = to_f(xbc[xrow + d_ssm + (long long)g * ds + n]);
                c_sm[n] = to_f(xbc[xrow + d_ssm + (long long)(ng + g) * ds + n]);
            }
            const float* sc = scal3 + (row * nh + h) * 3;
            const float alpha = sc[0], dt = sc[2];
            __syncthreads();
            float dyv = 0.0f, xp = 0.0f;
            if (own) {
                dyv = dy[row * d_ssm + (long long)h * hd + tid];
                xp = to_f(xbc[xrow + (long long)h * hd + tid]);
                xS[tid] = xp;
                dyS[tid] = dyv;
            }
            __syncthreads();
            // dS[p][n] += dy[p] C[n], each element its own step, over the
            // whole block.
            {
                int p = tid / ds, n = tid - (tid / ds) * ds;
                for (int e = tid; e < hd * ds; e += blockDim.x) {
                    float* d = dsr + p * ld + n;
                    *d = __fmaf_rn(dyS[p], c_sm[n], *d);
                    n += blockDim.x;
                    while (n >= ds) {
                        n -= ds;
                        p++;
                    }
                }
            }
            __syncthreads();
            // dC and dB per state index, the two chains of an index on two
            // threads.
            for (int k = tid; k < 2 * ds; k += blockDim.x) {
                const int n = k < ds ? k : k - ds;
                float acc = 0.0f;
                if (k < ds) {
                    for (int p = 0; p < hd; p++) acc = __fmaf_rn(dyS[p], cur[p * ld + n], acc);
                    dc_h[(row * nh + h) * ds + n] = acc;
                } else {
                    for (int p = 0; p < hd; p++) acc = __fmaf_rn(dsr[p * ld + n], xS[p], acc);
                    db_h[(row * nh + h) * ds + n] = __fmul_rn(acc, dt);
                }
            }
            __syncthreads();
            // The tile of the state after this token is free: the state
            // before the next token (the one before this) comes into it
            // while the rest of this step runs.
            if (t > t0) {
                m2s_tile_fetch(cur, slot > 1 ? rec + (bh * stride + slot - 2) * pn : ck, hd, ds);
            }
            float ddt_loc = 0.0f, dal_loc = 0.0f;
            if (own) {
                float* drow = dsr + tid * ld;
                const float* prow = prev + tid * ld;
                float sx = 0.0f;
                int n = 0;
                // Four indices at a time, B one float4, the rows' values
                // loaded before the chains take them.
                if (((ds & 3) | (reinterpret_cast<unsigned long long>(b_sm) & 15)) == 0) {
                    for (; n + 4 <= ds; n += 4) {
                        const float4 b4 = *reinterpret_cast<const float4*>(b_sm + n);
                        const float bb[4] = {b4.x, b4.y, b4.z, b4.w};
                        float dv[4], pv[4];
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            dv[j] = drow[n + j];
                            pv[j] = prow[n + j];
                        }
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            sx = __fmaf_rn(dv[j], bb[j], sx);
                            dal_loc = __fmaf_rn(dv[j], pv[j], dal_loc);
                            drow[n + j] = __fmul_rn(alpha, dv[j]);
                        }
                    }
                }
                for (; n < ds; n++) {
                    const float dsv = drow[n];
                    sx = __fmaf_rn(dsv, b_sm[n], sx);
                    dal_loc = __fmaf_rn(dsv, prow[n], dal_loc);
                    drow[n] = __fmul_rn(alpha, dsv);
                }
                dxbc[xrow + (long long)h * hd + tid] = __fmaf_rn(dt, sx, __fmul_rn(dval, dyv));
                ddt_loc = __fmul_rn(xp, sx);
            }
            const float2 sums = m2s_block_sum2(ddt_loc, dal_loc, red, lanes);
            if (tid == 0) {
                const float ddt_x = sums.x, dalpha = sums.y;
                const float da = __fmul_rn(dalpha, alpha);
                float ddt = __fmaf_rn(da, a_neg[h], ddt_x);
                da_acc = __fmaf_rn(da, dt, da_acc);
                const float v = __fadd_rn(to_f(proj[row * ld_proj + dt_off + h]), dt_bias[h]);
                if (flags & M2_FLAG_DT_LIMIT) {
                    const float sp = m2_softplus(v);
                    if (sp < dt_min || sp > dt_max) ddt = 0.0f;
                }
                const float draw = v > 20.0f ? ddt : __fmul_rn(ddt, m2b_sigmoid(v));
                m2b_put(&d_proj[row * ld_proj + dt_off + h], draw);
                dbias_acc = __fadd_rn(dbias_acc, draw);
            }
            m2s_cp_wait();
            __syncthreads();
            float* freed = cur;
            cur = prev;
            prev = freed;
        }
    }
    if (tid == 0) {
        da_parts[bh] = da_acc;
        dbias_parts[bh] = dbias_acc;
    }
}

#define DEFINE_M2S_BWD_TILES(NAME, T)                                           \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ dxbc,                                                   \
    T* __restrict__ d_proj, long long ld_proj, int dt_off,                      \
    float* __restrict__ db_h, float* __restrict__ dc_h,                         \
    float* __restrict__ da_parts, float* __restrict__ dbias_parts,              \
    float* __restrict__ rec,                                                    \
    const float* __restrict__ dy,                                               \
    const T* __restrict__ xbc, const T* __restrict__ proj,                      \
    const float* __restrict__ ckpt, const float* __restrict__ scal3,            \
    const float* __restrict__ dt_bias, const float* __restrict__ a_neg,         \
    const float* __restrict__ D,                                                \
    int seq, int stride, int nh, int ng, int hd, int ds, int conv_dim,          \
    int flags, float dt_min, float dt_max                                       \
) {                                                                             \
    m2s_bwd_tiles_body<T>(dxbc, d_proj, ld_proj, dt_off, db_h, dc_h, da_parts,  \
        dbias_parts, rec, dy, xbc, proj, ckpt, scal3, dt_bias, a_neg, D, seq,   \
        stride, nh, ng, hd, ds, conv_dim, flags, dt_min, dt_max);               \
}
DEFINE_M2S_BWD_TILES(m2s_bwd_tiles, float)
DEFINE_M2S_BWD_TILES(m2s_bwd_tiles_bf16, __nv_bfloat16)
DEFINE_M2S_BWD_TILES(m2s_bwd_tiles_f16, __half)

// dB and dC of the sequential path into the B and C columns of dxbc: the
// group's heads summed ascending. Grid 1-D over rows * G * N * 2.
extern "C" __global__ void m2s_bc_groups(
    float* __restrict__ dxbc,
    const float* __restrict__ db_h, const float* __restrict__ dc_h,
    long long rows, int nh, int ng, int ds, long long ld_xbc, int b_col, int c_col
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long per = (long long)ng * ds;
    if (i >= rows * per * 2) return;
    const int which = (int)(i / (rows * per));
    const long long k = i % (rows * per);
    const long long row = k / per;
    const int g = (int)((k % per) / ds);
    const int n = (int)(k % ds);
    const float* src = which ? dc_h : db_h;
    const int hpg = nh / ng;
    float acc = 0.0f;
    for (int j = 0; j < hpg; j++)
        acc = __fadd_rn(acc, src[(row * nh + g * hpg + j) * ds + n]);
    dxbc[row * ld_xbc + (which ? c_col : b_col) + (long long)g * ds + n] = acc;
}
