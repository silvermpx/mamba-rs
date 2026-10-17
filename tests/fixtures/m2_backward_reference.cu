// Mamba-2's per-output backward kernels as they were released: dx with the
// per-channel terms of the token sums, dB and dC, dH_loc and the pair
// terms, a thread an output, each chain read from global memory. They served the heads too wide for the
// register tiles; the tiles that replaced them, sliced over channels, are
// held to them bit for bit. The typed prelude and the packed-row helpers
// precede this fixture.

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif
#define M2_CHUNK 64
#define M2B_PAIRS (M2_CHUNK * (M2_CHUNK + 1) / 2)

__device__ __forceinline__ float m2_exp(float x) {
    return exp2f(__fmul_rn(x, LOG2E));
}

__device__ __forceinline__ int m2b_pair(int t, int s) {
    return t * (t + 1) / 2 + s;
}

// The (t, s) of pair q, s <= t.
__device__ __forceinline__ void m2b_unpair(int q, int* t_out, int* s_out) {
    int t = (int)((sqrtf(8.0f * q + 1.0f) - 1.0f) * 0.5f);
    while ((t + 1) * (t + 2) / 2 <= q) t++;
    while (t * (t + 1) / 2 > q) t--;
    *t_out = t;
    *s_out = q - t * (t + 1) / 2;
}

// dx and the per-channel terms the decay gradients need, one thread per
// (s, p) of a chunk and head:
//   st = B_s . dS[c][p], ci = C_s . H_c[p]      (n ascending, together)
//   ste = st E(L-1, s); acc = ste + sum_{t>=s} (CB E(t, s)) dy_t[p]
//   dx = acc dt_s + D dy_s[p]
// writing dx into the x columns of dxbc and acc, ste and
// yi = ci e^Acs_s to [B * T, d_ssm] scratch. In a packed row ste is 0 off
// the chunk's last sequence, yi 0 past the chunk's first start, and t runs
// over s's own sequence.
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
    int batch, int seq, int nh, int ng, int hd, int dstate,
    int y_first, /* this run's first y block */
    int z_first, /* this run's first z block */
    const int* __restrict__ seq_idx
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int e = (z_first + blockIdx.z) * blockDim.x + threadIdx.x;
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
    // The chunk state holds only its last sequence; the entering state
    // reaches only the tokens before the chunk's first start; s's pairs end
    // where its sequence does.
    const int sid = vl_id(seq_idx, b, seq, t0 + s);
    const bool in_last = sid == vl_id(seq_idx, b, seq, t0 + lc - 1);
    const bool from_prev = sid == vl_id(seq_idx, b, seq, t0 - 1);
    const int t_end = vl_lower_bound(seq_idx, b, seq, t0 + s, t0 + lc, sid + 1) - t0;
    const float ste = in_last
        ? __fmul_rn(st, m2_exp(fminf(__fsub_rn(acs[sc + lc - 1], acs_s), 0.0f)))
        : 0.0f;
    float acc = ste;
    const long long cbo = (bc * ng + g) * M2B_PAIRS;
    for (int t = s; t < t_end; t++) {
        float wv = __fmul_rn(cb[cbo + m2b_pair(t, s)],
                             m2_exp(fminf(__fsub_rn(acs[sc + t], acs_s), 0.0f)));
        acc = __fmaf_rn(wv, dy[(b * seq + t0 + t) * d_ssm + (long long)h * hd + p], acc);
    }
    const long long o = rs * d_ssm + (long long)h * hd + p;
    const float dv = d_per_channel ? D[(long long)h * hd + p] : D[h];
    dxbc[rs * ld_xbc + (long long)h * hd + p] = __fmaf_rn(acc, dt[sc + s], __fmul_rn(dv, dy[o]));
    acc_out[o] = acc;
    ste_out[o] = ste;
    yi_out[o] = from_prev ? __fmul_rn(ci, m2_exp(acs_s)) : 0.0f;
}

#define DEFINE_M2B_DX(NAME, T)                                                 \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dxbc,                                                  \
    float* __restrict__ acc_out,                                               \
    float* __restrict__ ste_out,                                               \
    float* __restrict__ yi_out,                                                \
    const float* __restrict__ dy,                                              \
    const T* __restrict__ xbc,                                                 \
    long long ld_xbc,                                                          \
    int b_col,                                                                 \
    int c_col,                                                                 \
    const float* __restrict__ states,                                          \
    const float* __restrict__ ds_in,                                           \
    const float* __restrict__ cb,                                              \
    const float* __restrict__ dt,                                              \
    const float* __restrict__ acs,                                             \
    const float* __restrict__ D,                                               \
    int d_per_channel,                                                         \
    int batch,                                                                 \
    int seq,                                                                   \
    int nh,                                                                    \
    int ng,                                                                    \
    int hd,                                                                    \
    int dstate,                                                                \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m2b_dx_body<T>(dxbc, acc_out, ste_out, yi_out, dy, xbc, ld_xbc, b_col,     \
        c_col, states, ds_in, cb, dt, acs, D, d_per_channel, batch, seq,       \
        nh, ng, hd, dstate, y_first, z_first, seq_idx);                        \
}
DEFINE_M2B_DX(m2b_reference_dx, float)
DEFINE_M2B_DX(m2b_reference_dx_bf16, __nv_bfloat16)
DEFINE_M2B_DX(m2b_reference_dx_f16, __half)

// dB of a chunk and group, one thread per (s, n): the group's heads
// ascending (each x_s . dS[c][:, n], p ascending, times dt_s E(L-1, s)),
// then t >= s ascending of dCB[t][s] C_t[n], where dCB sums the group's
// heads ascending. Written into the B columns of dxbc. In a packed row the
// first sum is 0 off the chunk's last sequence and t runs over s's own
// sequence. Grid (B * nc, G, ceil(64 * N / blockDim.x)).
template <typename T>
__device__ __forceinline__ void m2b_db_body(
    float* __restrict__ dxbc,
    const T* __restrict__ xbc, long long ld_xbc, int b_col, int c_col,
    const float* __restrict__ ds_in, const float* __restrict__ dcbh,
    const float* __restrict__ dt, const float* __restrict__ acs,
    int batch, int seq, int nh, int ng, int hd, int dstate,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int g = y_first + blockIdx.y;
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
    const int sid = vl_id(seq_idx, b, seq, t0 + s);
    const bool in_last = sid == vl_id(seq_idx, b, seq, t0 + lc - 1);
    const int t_end = vl_lower_bound(seq_idx, b, seq, t0 + s, t0 + lc, sid + 1) - t0;
    float acc = 0.0f;
    for (int k = 0; in_last && k < hpg; k++) {
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
    for (int t = s; t < t_end; t++) {
        float dcb = 0.0f;
        for (int k = 0; k < hpg; k++) {
            dcb = __fadd_rn(dcb, dcbh[(bc * nh + g * hpg + k) * M2B_PAIRS + m2b_pair(t, s)]);
        }
        const long long ct = (b * seq + t0 + t) * ld_xbc + c_col + (long long)g * dstate + n;
        acc = __fmaf_rn(dcb, to_f(xbc[ct]), acc);
    }
    dxbc[rs * ld_xbc + b_col + (long long)g * dstate + n] = acc;
}

#define DEFINE_M2B_DB(NAME, T)                                                 \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dxbc,                                                  \
    const T* __restrict__ xbc,                                                 \
    long long ld_xbc,                                                          \
    int b_col,                                                                 \
    int c_col,                                                                 \
    const float* __restrict__ ds_in,                                           \
    const float* __restrict__ dcbh,                                            \
    const float* __restrict__ dt,                                              \
    const float* __restrict__ acs,                                             \
    int batch,                                                                 \
    int seq,                                                                   \
    int nh,                                                                    \
    int ng,                                                                    \
    int hd,                                                                    \
    int dstate,                                                                \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m2b_db_body<T>(dxbc, xbc, ld_xbc, b_col, c_col, ds_in, dcbh, dt, acs,      \
        batch, seq, nh, ng, hd, dstate, y_first, seq_idx);                     \
}
DEFINE_M2B_DB(m2b_reference_db, float)
DEFINE_M2B_DB(m2b_reference_db_bf16, __nv_bfloat16)
DEFINE_M2B_DB(m2b_reference_db_f16, __half)

// dC of a chunk and group, one thread per (t, n): the group's heads
// ascending (each dy_t . H_c[:, n], p ascending, times e^Acs_t), then
// s <= t ascending of dCB[t][s] B_s[n]. Written into the C columns. In a
// packed row the first sum is 0 past the chunk's first start and s runs
// over t's own sequence. Grid (B * nc, G, ceil(64 * N / blockDim.x)).
template <typename T>
__device__ __forceinline__ void m2b_dc_body(
    float* __restrict__ dxbc,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc, int b_col, int c_col,
    const float* __restrict__ states, const float* __restrict__ dcbh,
    const float* __restrict__ acs,
    int batch, int seq, int nh, int ng, int hd, int dstate,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int g = y_first + blockIdx.y;
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
    const int tid_ = vl_id(seq_idx, b, seq, t0 + t);
    const bool from_prev = tid_ == vl_id(seq_idx, b, seq, t0 - 1);
    const int s_first = vl_lower_bound(seq_idx, b, seq, t0, t0 + t + 1, tid_) - t0;
    float acc = 0.0f;
    for (int k = 0; from_prev && k < hpg; k++) {
        const int h = g * hpg + k;
        const long long so = (bc * nh + h) * hd * (long long)dstate + n;
        float inner = 0.0f;
        for (int p = 0; p < hd; p++) {
            inner = __fmaf_rn(dy[rt * d_ssm + (long long)h * hd + p],
                              states[so + (long long)p * dstate], inner);
        }
        acc = __fmaf_rn(inner, m2_exp(acs[(bc * nh + h) * M2_CHUNK + t]), acc);
    }
    for (int s = s_first; s <= t; s++) {
        float dcb = 0.0f;
        for (int k = 0; k < hpg; k++) {
            dcb = __fadd_rn(dcb, dcbh[(bc * nh + g * hpg + k) * M2B_PAIRS + m2b_pair(t, s)]);
        }
        const long long bs = (b * seq + t0 + s) * ld_xbc + b_col + (long long)g * dstate + n;
        acc = __fmaf_rn(dcb, to_f(xbc[bs]), acc);
    }
    dxbc[rt * ld_xbc + c_col + (long long)g * dstate + n] = acc;
}

#define DEFINE_M2B_DC(NAME, T)                                                 \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dxbc,                                                  \
    const float* __restrict__ dy,                                              \
    const T* __restrict__ xbc,                                                 \
    long long ld_xbc,                                                          \
    int b_col,                                                                 \
    int c_col,                                                                 \
    const float* __restrict__ states,                                          \
    const float* __restrict__ dcbh,                                            \
    const float* __restrict__ acs,                                             \
    int batch,                                                                 \
    int seq,                                                                   \
    int nh,                                                                    \
    int ng,                                                                    \
    int hd,                                                                    \
    int dstate,                                                                \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m2b_dc_body<T>(dxbc, dy, xbc, ld_xbc, b_col, c_col, states, dcbh, acs,     \
        batch, seq, nh, ng, hd, dstate, y_first, seq_idx);                     \
}
DEFINE_M2B_DC(m2b_reference_dc, float)
DEFINE_M2B_DC(m2b_reference_dc_bf16, __nv_bfloat16)
DEFINE_M2B_DC(m2b_reference_dc_f16, __half)

// dH_loc[b, c, h, p, n] = sum_t (dy_t[p] e^Acs_t) C_t[n], t ascending over
// the tokens the entering state reaches. Grid (B * nc, H,
// ceil(P * N / blockDim.x)), the forward chunk state's.
template <typename T>
__device__ __forceinline__ void m2b_dhloc_body(
    float* __restrict__ ds_out,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc, int c_col,
    const float* __restrict__ acs,
    int batch, int seq, int nh, int ng, int hd, int ds,
    int y_first, /* this run's first y block */
    int z_first, /* this run's first z block */
    const int* __restrict__ seq_idx
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const long long e = (long long)(z_first + blockIdx.z) * blockDim.x + threadIdx.x;
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
    const int cont = vl_lower_bound(seq_idx, b, seq, t0, t0 + lc,
                                    vl_id(seq_idx, b, seq, t0 - 1) + 1) - t0;
    float acc = 0.0f;
    for (int t = 0; t < cont; t++) {
        const long long row = b * seq + t0 + t;
        float dye = __fmul_rn(dy[row * d_ssm + (long long)h * hd + p], m2_exp(acs[sc + t]));
        acc = __fmaf_rn(dye, to_f(xbc[row * ld_xbc + c_col + (long long)g * ds + n]), acc);
    }
    ds_out[((bc * nh + h) * hd + p) * (long long)ds + n] = acc;
}

#define DEFINE_M2B_DHLOC(NAME, T)                                              \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ ds_out,                                                \
    const float* __restrict__ dy,                                              \
    const T* __restrict__ xbc,                                                 \
    long long ld_xbc,                                                          \
    int c_col,                                                                 \
    const float* __restrict__ acs,                                             \
    int batch,                                                                 \
    int seq,                                                                   \
    int nh,                                                                    \
    int ng,                                                                    \
    int hd,                                                                    \
    int ds,                                                                    \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m2b_dhloc_body<T>(ds_out, dy, xbc, ld_xbc, c_col, acs, batch, seq, nh,     \
        ng, hd, ds, y_first, z_first, seq_idx);                                \
}
DEFINE_M2B_DHLOC(m2b_reference_dhloc, float)
DEFINE_M2B_DHLOC(m2b_reference_dhloc_bf16, __nv_bfloat16)
DEFINE_M2B_DHLOC(m2b_reference_dhloc_f16, __half)

// The pair terms of a chunk and head, one thread per pair s <= t:
//   G = dy_t . x_s (p ascending), dcb = G E(t, s) dt_s,
//   m = dcb CB[t][s] for t > s (0 on the diagonal)
// stored [B, nc, H, pairs]; both 0 for two sequences of a packed row.
// Grid (B * nc, H, ceil(pairs / blockDim.x)).
template <typename T>
__device__ __forceinline__ void m2b_pairs_body(
    float* __restrict__ dcbh, float* __restrict__ mm,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc,
    const float* __restrict__ cb,
    const float* __restrict__ dt, const float* __restrict__ acs,
    int batch, int seq, int nh, int ng, int hd,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int q = blockIdx.z * blockDim.x + threadIdx.x;
    if (bc >= (long long)batch * nc || h >= nh || q >= M2B_PAIRS) return;
    int t, s;
    m2b_unpair(q, &t, &s);
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    const long long o = (bc * nh + h) * M2B_PAIRS + q;
    if (t >= lc || vl_id(seq_idx, b, seq, t0 + t) != vl_id(seq_idx, b, seq, t0 + s)) {
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

#define DEFINE_M2B_PAIRS(NAME, T)                                              \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dcbh,                                                  \
    float* __restrict__ mm,                                                    \
    const float* __restrict__ dy,                                              \
    const T* __restrict__ xbc,                                                 \
    long long ld_xbc,                                                          \
    const float* __restrict__ cb,                                              \
    const float* __restrict__ dt,                                              \
    const float* __restrict__ acs,                                             \
    int batch,                                                                 \
    int seq,                                                                   \
    int nh,                                                                    \
    int ng,                                                                    \
    int hd,                                                                    \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m2b_pairs_body<T>(dcbh, mm, dy, xbc, ld_xbc, cb, dt, acs, batch, seq,      \
        nh, ng, hd, y_first, seq_idx);                                         \
}
DEFINE_M2B_PAIRS(m2b_reference_pairs, float)
DEFINE_M2B_PAIRS(m2b_reference_pairs_bf16, __nv_bfloat16)
DEFINE_M2B_PAIRS(m2b_reference_pairs_f16, __half)
