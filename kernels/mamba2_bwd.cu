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
// A packed row (`seq_idx`, see _varlen.cuh) masks the backward as its
// forward: the terms of the entering state reach only the tokens before
// the chunk's first start, the chunk state's only its last sequence, a
// pair only two tokens of one sequence, and the state passing skips a
// chunk where a sequence starts. A masked term is dropped, never scaled by
// zero, so an overflow in one sequence leaves the others' gradients whole.
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


// The state passing backward, in place: dS holds dH_loc on entry and
// dS[c] = d H_c+1 on exit, from zero after the last chunk:
// e <- dH_loc[c] + e^Acs_last e, or e <- dH_loc[c] for a chunk where a
// sequence of a packed row starts. As m2_state_passing, the decays sit in
// shared memory and the next four chunks are loaded ahead.
// Grid (B * H, ceil(P * N / blockDim.x)).
extern "C" __global__ void m2b_state_passing(
    float* __restrict__ ds,
    const float* __restrict__ acs,
    int batch, int seq, int nh, int hd, int dstate,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    __shared__ float decay[M2_PASS_WINDOW];
    __shared__ bool reset[M2_PASS_WINDOW];
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bh = blockIdx.x;
    const long long e = (long long)(y_first + blockIdx.y) * blockDim.x + threadIdx.x;
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
            reset[i] = vl_id(seq_idx, b, seq, c * M2_CHUNK + lc - 1)
                       != vl_id(seq_idx, b, seq, c * M2_CHUNK - 1);
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
            ev = reset[c - c0] ? tmp : __fmaf_rn(decay[c - c0], ev, tmp);
        }
    }
}

// dot[q] = H_c . dS[c] for q = (b * nc + c) * H + h: strided ascending
// partials, then a fixed tree; 0 for a chunk where a sequence of a packed
// row starts, whose state owes nothing to H_c. Grid 1-D over B * nc * H,
// block 256.
extern "C" __global__ void m2b_chunk_dot(
    float* __restrict__ dot,
    const float* __restrict__ states,
    const float* __restrict__ ds,
    long long count, int pn,
    int seq, int nh, const int* __restrict__ seq_idx
) {
    __shared__ float red[256];
    const long long q = blockIdx.x;
    if (q >= count) return;
    if (seq_idx != nullptr) {
        const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
        const long long bcq = q / nh;
        const int c = (int)(bcq % nc);
        const long long b = bcq / nc;
        const int lc = min(M2_CHUNK, seq - c * M2_CHUNK);
        if (vl_id(seq_idx, b, seq, c * M2_CHUNK + lc - 1)
            != vl_id(seq_idx, b, seq, c * M2_CHUNK - 1)) {
            if (threadIdx.x == 0) dot[q] = 0.0f;
            return;
        }
    }
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

// The per-token sums over a head's channels, p ascending:
//   ddt_x[t] = sum_p acc x_t,  Q_t = dt_t sum_p ste x_t,  I_t = sum_p dy yi
// stored [B, nc, H, 64] (zero past the chunk). In a packed row Q_t is 0
// off the chunk's last sequence and I_t 0 past its first start, dropped
// rather than summed from zero terms: an overflowed x_t or dt_t times a
// zero would be NaN, and the span sums of the decay gradient carry Q_t
// into the next sequence's tokens. Grid (B * nc, H), block 64.
template <typename T>
__device__ __forceinline__ void m2b_rowsums_body(
    float* __restrict__ ddtx, float* __restrict__ qv, float* __restrict__ inter,
    const float* __restrict__ acc_b, const float* __restrict__ ste_b,
    const float* __restrict__ yi_b, const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc,
    const float* __restrict__ dt,
    int batch, int seq, int nh, int hd,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
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
    const int sid = vl_id(seq_idx, b, seq, t0 + t);
    const bool in_last = sid == vl_id(seq_idx, b, seq, t0 + lc - 1);
    const bool from_prev = sid == vl_id(seq_idx, b, seq, t0 - 1);
    float a = 0.0f, q = 0.0f, it = 0.0f;
    for (int p = 0; p < hd; p++) {
        const float x = to_f(xbc[row * ld_xbc + (long long)h * hd + p]);
        const long long i = row * d_ssm + (long long)h * hd + p;
        a = __fmaf_rn(acc_b[i], x, a);
        q = in_last ? __fmaf_rn(ste_b[i], x, q) : q;
        it = from_prev ? __fmaf_rn(dy[i], yi_b[i], it) : it;
    }
    ddtx[o] = a;
    qv[o] = in_last ? __fmul_rn(q, dt[o]) : 0.0f;
    inter[o] = it;
}

#define DEFINE_M2B_ROWSUMS(NAME, T)                                            \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ ddtx,                                                  \
    float* __restrict__ qv,                                                    \
    float* __restrict__ inter,                                                 \
    const float* __restrict__ acc_b,                                           \
    const float* __restrict__ ste_b,                                           \
    const float* __restrict__ yi_b,                                            \
    const float* __restrict__ dy,                                              \
    const T* __restrict__ xbc,                                                 \
    long long ld_xbc,                                                          \
    const float* __restrict__ dt,                                              \
    int batch,                                                                 \
    int seq,                                                                   \
    int nh,                                                                    \
    int hd,                                                                    \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m2b_rowsums_body<T>(ddtx, qv, inter, acc_b, ste_b, yi_b, dy, xbc,          \
        ld_xbc, dt, batch, seq, nh, hd, y_first, seq_idx);                     \
}
DEFINE_M2B_ROWSUMS(m2b_rowsums, float)
DEFINE_M2B_ROWSUMS(m2b_rowsums_bf16, __nv_bfloat16)
DEFINE_M2B_ROWSUMS(m2b_rowsums_f16, __half)

// dH_loc[b, c, h, p, n] = sum_t (dy_t[p] e^Acs_t) C_t[n], t ascending over
// the tokens the entering state reaches (those before the chunk's first
// start). Register tiles, the forward chunk state's: a warp takes 4
// channels p and a lane the state indices lane + 32 j of the block's tile
// (JC of them). e^Acs_t and the C rows [64][32 JC] are staged once per
// block, dy e^Acs [64][pt] once per tile of pt channels; each output is one
// chain, t ascending from 0, whatever the tile, the per-output kernel's.
// Grid (B * nc, H, ceil(N / (32 JC))), block up to 256 threads, dynamic
// shared memory 64 (pt + 32 JC + 1) floats (chunk_state_geom).
template <typename T, int JC>
__device__ __forceinline__ void m2b_dhloc_body(
    float* __restrict__ ds_out,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc, int c_col,
    const float* __restrict__ acs,
    int batch, int seq, int nh, int ng, int hd, int ds, int pt,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float4 m2bh_sm4[];
    constexpr int NT = 32 * JC;
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int n0 = blockIdx.z * NT;
    if (bc >= (long long)batch * nc || h >= nh) return;
    const int g = h / (nh / ng);
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    const long long sc = (bc * nh + h) * M2_CHUNK;
    const long long d_ssm = (long long)nh * hd;
    float* ys = reinterpret_cast<float*>(m2bh_sm4);  // [64][pt]
    float* cs = ys + M2_CHUNK * pt;                   // [64][NT]
    float* es = cs + M2_CHUNK * NT;                   // [64]
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    const int cont = vl_lower_bound(seq_idx, b, seq, t0, t0 + lc,
                                    vl_id(seq_idx, b, seq, t0 - 1) + 1) - t0;
    for (int t = tid; t < cont; t += blockDim.x) es[t] = m2_exp(acs[sc + t]);
    m2_stage4(cont * NT, tid, blockDim.x,
              [&](int i) {
                  const int t = i / NT, n = n0 + (i - t * NT);
                  return n < ds
                      ? to_f(xbc[(b * seq + t0 + t) * ld_xbc + c_col + (long long)g * ds + n])
                      : 0.0f;
              },
              [&](int i, float v) { cs[i] = v; });
    __syncthreads();
    for (int pt0 = 0; pt0 < hd; pt0 += pt) {
        /* The previous tile's reads are done before its rows are replaced. */
        __syncthreads();
        m2_stage4(cont * pt, tid, blockDim.x,
                  [&](int i) {
                      const int t = i / pt, p = pt0 + (i - t * pt);
                      return p < hd
                          ? __fmul_rn(dy[(b * seq + t0 + t) * d_ssm + (long long)h * hd + p], es[t])
                          : 0.0f;
                  },
                  [&](int i, float v) { ys[i] = v; });
        __syncthreads();
        for (int p0 = 4 * warp; p0 < pt && pt0 + p0 < hd; p0 += 4 * nw) {
            float acc[4][JC];
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < JC; j++) acc[i][j] = 0.0f;
            }
            for (int t = 0; t < cont; t++) {
                const float4 y4 = *reinterpret_cast<const float4*>(ys + t * pt + p0);
                const float yv[4] = {y4.x, y4.y, y4.z, y4.w};
                float cv[JC];
#pragma unroll
                for (int j = 0; j < JC; j++) cv[j] = cs[t * NT + lane + 32 * j];
#pragma unroll
                for (int i = 0; i < 4; i++) {
#pragma unroll
                    for (int j = 0; j < JC; j++) acc[i][j] = __fmaf_rn(yv[i], cv[j], acc[i][j]);
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int p = pt0 + p0 + i;
                if (p >= hd) continue;
#pragma unroll
                for (int j = 0; j < JC; j++) {
                    const int n = n0 + lane + 32 * j;
                    if (n < ds) ds_out[((bc * nh + h) * hd + p) * (long long)ds + n] = acc[i][j];
                }
            }
        }
    }
}

#define DEFINE_M2B_DHLOC(NAME, T)                                              \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ ds_out,                                                \
    const float* __restrict__ dy,                                              \
    const T* __restrict__ xbc,                                                 \
    long long ld_xbc,                                                          \
    int c_col,                                                                 \
    const float* __restrict__ acs,                                             \
    int batch, int seq, int nh, int ng, int hd, int ds, int pt,                \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    if (ds <= 32)                                                              \
        m2b_dhloc_body<T, 1>(ds_out, dy, xbc, ld_xbc, c_col, acs, batch, seq,  \
            nh, ng, hd, ds, pt, y_first, seq_idx);                             \
    else if (ds <= 64)                                                         \
        m2b_dhloc_body<T, 2>(ds_out, dy, xbc, ld_xbc, c_col, acs, batch, seq,  \
            nh, ng, hd, ds, pt, y_first, seq_idx);                             \
    else                                                                       \
        m2b_dhloc_body<T, 4>(ds_out, dy, xbc, ld_xbc, c_col, acs, batch, seq,  \
            nh, ng, hd, ds, pt, y_first, seq_idx);                             \
}
DEFINE_M2B_DHLOC(m2b_dhloc, float)
DEFINE_M2B_DHLOC(m2b_dhloc_bf16, __nv_bfloat16)
DEFINE_M2B_DHLOC(m2b_dhloc_f16, __half)

// dx and the per-channel terms the decay gradients need, as register tiles
// over a chunk and head: thread (s-group, p-group) holds 4 tokens s by 4
// channels p.
//   st = B_s . dS[c][p] and ci = C_s . H_c[p] run n ascending from 0 over
//   the rows staged `nk` state indices at a time (B or C transposed [n][s],
//   dS or H transposed [n][p]);
//   ste = st E(L-1, s), then acc runs t = s.. ascending over the staged
//   weights wv[t][s] = CB E(t, s) and dy rows; dx = fma(acc, dt_s, D dy_s),
//   written into the x columns of dxbc;
//   acc, ste and yi = ci e^Acs_s feed the per-token sums over the head's
//   channels, p ascending:
//     ddt_x[t] = sum_p acc x_t,  Q_t = dt_t sum_p ste x_t,  I_t = sum_p dy yi.
// FUSED: one block holds the whole head, 16 ceil(P / 4) threads, and puts
// acc, ste and yi in shared rows, where a thread per token runs the sums
// into ddtx, qv and inter (o1, o2, o3).
// Otherwise a block holds M2DX_PC channels of the head, 256 threads, grid
// z the channel groups, and writes acc, ste and yi to [B * T, d_ssm]
// scratch (o1, o2, o3), whose sums m2b_rowsums runs.
// Every value is one chain in the order of the per-output kernels these
// replaced, so the bits are theirs; a packed row masks as they do: ste is 0
// off the chunk's last sequence, yi 0 past the chunk's first start, and t
// runs over s's own sequence.
// Shared memory: the staging, nk (68 + P4 + 4) floats (fused, at least
// 3 * 64 (P + 1) for the shared rows), then wv [64][64], dy [64][P4], acs
// and dt [64] each; P4 = the block's channels rounded up to 4.
// Grid (B * nc, H) fused, (B * nc, H, ceil(P / M2DX_PC)) in groups.
#define M2DX_PC 64
template <typename T, bool FUSED>
__device__ __forceinline__ void m2b_dx_tiles_body(
    float* __restrict__ dxbc,
    float* __restrict__ o1, float* __restrict__ o2, float* __restrict__ o3,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc, int b_col, int c_col,
    const float* __restrict__ states, const float* __restrict__ ds_in,
    const float* __restrict__ cb,
    const float* __restrict__ dt, const float* __restrict__ acs,
    const float* __restrict__ D, int d_per_channel,
    int seq, int nh, int ng, int hd, int dstate, int nk,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float4 m2bt_sm4[];
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    const int g = h / (nh / ng);
    const long long d_ssm = (long long)nh * hd;
    const long long sc = (bc * nh + h) * M2_CHUNK;
    // This block's channels: the head (fused) or a group of it.
    const int c0 = FUSED ? 0 : blockIdx.z * M2DX_PC;
    const int cw = FUSED ? hd : min(M2DX_PC, hd - c0);
    const int pg = FUSED ? (hd + 3) / 4 : M2DX_PC / 4;
    const int p4 = 4 * pg;
    const int lts = M2_CHUNK + 4, rts = p4 + 4, rows = hd + 1;
    const int stage_floats =
        FUSED ? max(nk * (lts + rts), 3 * M2_CHUNK * rows) : nk * (lts + rts);
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
                  if (s <= t && t < lc
                      && vl_id(seq_idx, b, seq, t0 + s) == vl_id(seq_idx, b, seq, t0 + t))
                      w = __fmul_rn(cb[cbo + m2b_pair(t, s)],
                                    m2_exp(fminf(__fsub_rn(acs[sc + t], acs[sc + s]), 0.0f)));
                  return w;
              },
              [&](int i, float v) { wv[i] = v; });
    m2_stage4(M2_CHUNK * p4, tid, blockDim.x,
              [&](int i) {
                  const int t = i / p4, p = i - t * p4;
                  return t < lc && p < cw
                      ? dy[(b * seq + t0 + t) * d_ssm + (long long)h * hd + c0 + p] : 0.0f;
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
                          return p < cw ? src[(long long)(c0 + p) * dstate + m0 + n] : 0.0f;
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
    // The chunk state holds only its last sequence; the entering state
    // reaches only the tokens before the chunk's first start; s's pairs end
    // where its sequence does.
    const int id_last = vl_id(seq_idx, b, seq, t0 + lc - 1);
    const int id_prev = vl_id(seq_idx, b, seq, t0 - 1);
    bool in_last[4], from_prev[4];
    int t_end[4];
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int sid = s0 + i < lc ? vl_id(seq_idx, b, seq, t0 + s0 + i) : id_last;
        in_last[i] = sid == id_last;
        from_prev[i] = sid == id_prev;
        t_end[i] = s0 + i < lc
            ? vl_lower_bound(seq_idx, b, seq, t0 + s0 + i, t0 + lc, sid + 1) - t0
            : lc;
        const float el = m2_exp(fminf(__fsub_rn(acs_s[max(lc - 1, 0)], acs_s[s0 + i]), 0.0f));
#pragma unroll
        for (int j = 0; j < 4; j++) {
            st[i][j] = in_last[i] ? __fmul_rn(st[i][j], el) : 0.0f;
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
            if (t >= s0 + i && t < t_end[i]) {
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
            if (p >= cw) continue;
            const long long ch = (long long)h * hd + c0 + p;
            const float dv = d_per_channel ? D[ch] : D[h];
            dxbc[rs * ld_xbc + ch] = __fmaf_rn(acc[i][j], dt_s[s], __fmul_rn(dv, dyt[s * p4 + p]));
            const float yi = from_prev[i] ? __fmul_rn(ci[i][j], ea) : 0.0f;
            if (FUSED) {
                at[s * rows + p] = acc[i][j];
                et[s * rows + p] = st[i][j];
                yt[s * rows + p] = yi;
            } else {
                o1[rs * d_ssm + ch] = acc[i][j];
                o2[rs * d_ssm + ch] = st[i][j];
                o3[rs * d_ssm + ch] = yi;
            }
        }
    }
    if (!FUSED) return;
    __syncthreads();
    for (int t = tid; t < M2_CHUNK; t += blockDim.x) {
        const long long o = sc + t;
        if (t >= lc) {
            o1[o] = 0.0f;
            o2[o] = 0.0f;
            o3[o] = 0.0f;
            continue;
        }
        const long long row = b * seq + t0 + t;
        const int sid = vl_id(seq_idx, b, seq, t0 + t);
        const bool last_t = sid == id_last, prev_t = sid == id_prev;
        float a = 0.0f, q = 0.0f, it = 0.0f;
        for (int p = 0; p < hd; p++) {
            const float x = to_f(xbc[row * ld_xbc + (long long)h * hd + p]);
            a = __fmaf_rn(at[t * rows + p], x, a);
            q = last_t ? __fmaf_rn(et[t * rows + p], x, q) : q;
            it = prev_t ? __fmaf_rn(dyt[t * p4 + p], yt[t * rows + p], it) : it;
        }
        o1[o] = a;
        o2[o] = last_t ? __fmul_rn(q, dt_s[t]) : 0.0f;
        o3[o] = it;
    }
}

#define DEFINE_M2B_DX_TILES(NAME, T)                                           \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dxbc,                                                  \
    float* __restrict__ ddtx,                                                  \
    float* __restrict__ qv,                                                    \
    float* __restrict__ inter,                                                 \
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
    int seq,                                                                   \
    int nh,                                                                    \
    int ng,                                                                    \
    int hd,                                                                    \
    int dstate,                                                                \
    int nk,                                                                    \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m2b_dx_tiles_body<T, true>(dxbc, ddtx, qv, inter, dy, xbc, ld_xbc, b_col, \
        c_col, states, ds_in, cb, dt, acs, D, d_per_channel, seq, nh, ng, hd,  \
        dstate, nk, y_first, seq_idx);                                         \
}
DEFINE_M2B_DX_TILES(m2b_dx_tiles, float)
DEFINE_M2B_DX_TILES(m2b_dx_tiles_bf16, __nv_bfloat16)
DEFINE_M2B_DX_TILES(m2b_dx_tiles_f16, __half)

// The heads too wide for one block: the tiles in groups of M2DX_PC
// channels, acc, ste and yi to scratch for m2b_rowsums; M2DX_NK state
// indices a staging pass, so the shared memory is DX_GROUPS_SMEM_BYTES
// whatever the shape.
#define M2DX_NK 16
#define DEFINE_M2B_DX_GROUPS(NAME, T)                                          \
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
    int seq,                                                                   \
    int nh,                                                                    \
    int ng,                                                                    \
    int hd,                                                                    \
    int dstate,                                                                \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m2b_dx_tiles_body<T, false>(dxbc, acc_out, ste_out, yi_out, dy, xbc,       \
        ld_xbc, b_col, c_col, states, ds_in, cb, dt, acs, D, d_per_channel,    \
        seq, nh, ng, hd, dstate, M2DX_NK, y_first, seq_idx);                   \
}
DEFINE_M2B_DX_GROUPS(m2b_dx_groups, float)
DEFINE_M2B_DX_GROUPS(m2b_dx_groups_bf16, __nv_bfloat16)
DEFINE_M2B_DX_GROUPS(m2b_dx_groups_f16, __half)


// The pair terms of a chunk and head, for every pair s <= t:
//   G = dy_t . x_s (p ascending), dcb = G E(t, s) dt_s,
//   m = dcb CB[t][s] for t > s (0 on the diagonal)
// stored [B, nc, H, pairs]; both 0 for two sequences of a packed row or a
// token past the chunk. One block per chunk and head, 256 threads, a
// thread a 4 x 4 tile of (t, s): dy and x come in transposed, M2BP_PC
// channels at a time, and each G runs on across the passes, p ascending
// from 0, the per-output kernel's chain. Dynamic shared memory: two
// [M2BP_PC][68] tiles (PAIRS_TILES_SMEM_BYTES). Grid (B * nc, H).
#define M2BP_PC 64
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
    extern __shared__ float4 m2bp_sm4[];
    constexpr int LD = M2_CHUNK + 4;
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    if (bc >= (long long)batch * nc || h >= nh) return;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    const int g = h / (nh / ng);
    const long long d_ssm = (long long)nh * hd;
    float* dyt = reinterpret_cast<float*>(m2bp_sm4);  // [PC][68], token across
    float* xt = dyt + M2BP_PC * LD;                    // [PC][68]
    const int tid = threadIdx.x;
    const int tq = 4 * (tid >> 4), sq = 4 * (tid & 15);
    float gv[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) gv[i][j] = 0.0f;
    }
    for (int c0 = 0; c0 < hd; c0 += M2BP_PC) {
        const int pw = min(M2BP_PC, hd - c0);
        __syncthreads();
        for (int i = tid; i < M2_CHUNK * pw; i += blockDim.x) {
            const int r = i / pw, pl = i - r * pw;
            float dv = 0.0f, xv = 0.0f;
            if (r < lc) {
                const long long row = b * seq + t0 + r;
                dv = dy[row * d_ssm + (long long)h * hd + c0 + pl];
                xv = to_f(xbc[row * ld_xbc + (long long)h * hd + c0 + pl]);
            }
            dyt[pl * LD + r] = dv;
            xt[pl * LD + r] = xv;
        }
        __syncthreads();
        if (sq <= tq + 3) {
            for (int pl = 0; pl < pw; pl++) {
                const float4 d4 = *reinterpret_cast<const float4*>(dyt + pl * LD + tq);
                const float4 x4 = *reinterpret_cast<const float4*>(xt + pl * LD + sq);
                const float dv[4] = {d4.x, d4.y, d4.z, d4.w};
                const float xv[4] = {x4.x, x4.y, x4.z, x4.w};
#pragma unroll
                for (int i = 0; i < 4; i++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) gv[i][j] = __fmaf_rn(dv[i], xv[j], gv[i][j]);
                }
            }
        }
    }
    const long long sc = (bc * nh + h) * M2_CHUNK;
    const long long base = (bc * nh + h) * M2B_PAIRS;
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int t = tq + i, s = sq + j;
            if (s > t) continue;
            const long long o = base + m2b_pair(t, s);
            if (t >= lc || vl_id(seq_idx, b, seq, t0 + t) != vl_id(seq_idx, b, seq, t0 + s)) {
                dcbh[o] = 0.0f;
                mm[o] = 0.0f;
                continue;
            }
            const float ev = m2_exp(fminf(__fsub_rn(acs[sc + t], acs[sc + s]), 0.0f));
            const float dcb = __fmul_rn(__fmul_rn(gv[i][j], ev), dt[sc + s]);
            dcbh[o] = dcb;
            mm[o] = t > s ? __fmul_rn(dcb, cb[(bc * ng + g) * M2B_PAIRS + m2b_pair(t, s)]) : 0.0f;
        }
    }
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
DEFINE_M2B_PAIRS(m2b_pairs, float)
DEFINE_M2B_PAIRS(m2b_pairs_bf16, __nv_bfloat16)
DEFINE_M2B_PAIRS(m2b_pairs_f16, __half)

// dB and dC of a chunk and group as register tiles, one block per (chunk,
// group, tile of 32 state indices): a warp takes 4 rows (s for dB, t for
// dC) and a lane an index n.
//   dB_s = sum_{h in g} dt_s E(L-1, s) (x_s . dS[c][:, n])
//          + sum_{t >= s} dCB[t][s] C_t[n]
//   dC_t = sum_{h in g} e^Acs_t (dy_t . H_c[:, n]) + sum_{s <= t} dCB[t][s] B_s[n]
// For each head of the group ascending, the head's x (dB) or dy (dC) rows
// are staged transposed with its dS (dB) or H (dC) columns, M2BT_PC
// channels at a time ([PC][68] and [PC][32]); the p-ascending inner sum
// from 0 runs on across the passes, then joins the row's chain as
// fma(inner, scale, acc), scale = dt_s E(L-1, s) (dB) or e^Acs_t (dC).
// Then the chain runs over the staged dCB of the group (each pair summed
// over the heads ascending from 0) and the C rows t >= s (dB) or the B rows
// s <= t (dC), ascending. In a packed row the first sum takes only the
// chunk's last sequence (dB) or the tokens before the chunk's first start
// (dC), and the pairs stay inside a row's own sequence. Shared memory,
// whatever the head width (BC_TILES_SMEM_BYTES): 68 PC + 32 PC + 64 * 64 +
// 64 * 32 + 64 floats, 37,632 bytes, the transposed rows padded to 68 so
// their staging writes spread over banks.
// Grid (B * nc, G, ceil(N / 32)), block 512.
#define M2BT_PC 32
template <typename T, bool DC>
__device__ __forceinline__ void m2b_bc_tiles_body(
    float* __restrict__ dxbc,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, long long ld_xbc, int b_col, int c_col,
    const float* __restrict__ mat, const float* __restrict__ dcbh,
    const float* __restrict__ dt, const float* __restrict__ acs,
    int seq, int nh, int ng, int hd, int dstate,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m2bt_sm[];
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int g = y_first + blockIdx.y;
    const int n0 = blockIdx.z * 32;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M2_CHUNK;
    const int lc = min(M2_CHUNK, seq - t0);
    const int hpg = nh / ng;
    const long long d_ssm = (long long)nh * hd;
    constexpr int LS = M2_CHUNK + 4;
    float* lt = m2bt_sm;                     // [PC][LS]
    float* rt = lt + M2BT_PC * LS;           // [PC][32]
    float* dcbg = rt + M2BT_PC * 32;         // [64][64], t by s
    float* orow = dcbg + M2_CHUNK * M2_CHUNK;// [64][32]
    float* scale = orow + M2_CHUNK * 32;     // [64]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5, nw = blockDim.x >> 5;
    const int r0 = 4 * warp;
    const int n = n0 + lane;
    const long long row0 = b * seq + t0;
    // A row's first sum takes only the chunk's last sequence (dB) or the
    // tokens before the chunk's first start (dC); its pairs stay inside its
    // own sequence.
    const int id_edge = DC ? vl_id(seq_idx, b, seq, t0 - 1) : vl_id(seq_idx, b, seq, t0 + lc - 1);
    bool keep[4];
    int lo[4], hi[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int r = r0 + i;
        const int sid = r < lc ? vl_id(seq_idx, b, seq, t0 + r) : id_edge;
        keep[i] = sid == id_edge;
        lo[i] = r < lc ? vl_lower_bound(seq_idx, b, seq, t0, t0 + r + 1, sid) - t0 : 0;
        hi[i] = r < lc ? vl_lower_bound(seq_idx, b, seq, t0 + r, t0 + lc, sid + 1) - t0 : lc;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < hpg; k++) {
        const int h = g * hpg + k;
        const long long sc = (bc * nh + h) * M2_CHUNK;
        const float* mh = mat + (bc * nh + h) * hd * (long long)dstate;
        float inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int c0 = 0; c0 < hd; c0 += M2BT_PC) {
            const int pw = min(M2BT_PC, hd - c0);
            __syncthreads();
            if (lane < pw) {
                for (int r = warp; r < M2_CHUNK; r += nw) {
                    float v = 0.0f;
                    if (r < lc) {
                        const long long p = (long long)h * hd + c0 + lane;
                        v = DC ? dy[(row0 + r) * d_ssm + p] : to_f(xbc[(row0 + r) * ld_xbc + p]);
                    }
                    lt[lane * LS + r] = v;
                }
            }
            for (int pl = warp; pl < pw; pl += nw)
                rt[pl * 32 + lane] = n < dstate ? mh[(long long)(c0 + pl) * dstate + n] : 0.0f;
            if (c0 == 0) {
                for (int r = tid; r < M2_CHUNK; r += blockDim.x) {
                    float sv = 0.0f;
                    if (r < lc) {
                        sv = DC ? m2_exp(acs[sc + r])
                                : __fmul_rn(dt[sc + r],
                                            m2_exp(fminf(__fsub_rn(acs[sc + lc - 1], acs[sc + r]),
                                                         0.0f)));
                    }
                    scale[r] = sv;
                }
            }
            __syncthreads();
            for (int pl = 0; pl < pw; pl++) {
                const float4 l4 = *reinterpret_cast<const float4*>(lt + pl * LS + r0);
                const float rv = rt[pl * 32 + lane];
                inner[0] = __fmaf_rn(l4.x, rv, inner[0]);
                inner[1] = __fmaf_rn(l4.y, rv, inner[1]);
                inner[2] = __fmaf_rn(l4.z, rv, inner[2]);
                inner[3] = __fmaf_rn(l4.w, rv, inner[3]);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (keep[i]) acc[i] = __fmaf_rn(inner[i], scale[r0 + i], acc[i]);
        }
    }
    __syncthreads();
    for (int e = tid; e < M2_CHUNK * M2_CHUNK; e += blockDim.x) {
        const int t = e / M2_CHUNK, s_ = e % M2_CHUNK;
        float dcb = 0.0f;
        if (s_ <= t && t < lc
            && vl_id(seq_idx, b, seq, t0 + s_) == vl_id(seq_idx, b, seq, t0 + t)) {
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
                if (s_ <= r0 + i && s_ >= lo[i])
                    acc[i] = __fmaf_rn(dcbg[(r0 + i) * M2_CHUNK + s_], bv, acc[i]);
            }
        }
    } else {
        // dB row s: t = s .. lc - 1 ascending.
        for (int t = r0; t < lc; t++) {
            const float cv = orow[t * 32 + lane];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (t >= r0 + i && t < hi[i])
                    acc[i] = __fmaf_rn(dcbg[t * M2_CHUNK + r0 + i], cv, acc[i]);
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

#define DEFINE_M2B_BC_TILES(NAME, T, DC)                                       \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dxbc,                                                  \
    const float* __restrict__ dy,                                              \
    const T* __restrict__ xbc, long long ld_xbc, int b_col, int c_col,         \
    const float* __restrict__ mat, const float* __restrict__ dcbh,             \
    const float* __restrict__ dt, const float* __restrict__ acs,               \
    int seq, int nh, int ng, int hd, int dstate,                               \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m2b_bc_tiles_body<T, DC>(dxbc, dy, xbc, ld_xbc, b_col, c_col, mat, dcbh,   \
        dt, acs, seq, nh, ng, hd, dstate, y_first, seq_idx);                   \
}
DEFINE_M2B_BC_TILES(m2b_db_tiles, float, false)
DEFINE_M2B_BC_TILES(m2b_db_tiles_bf16, __nv_bfloat16, false)
DEFINE_M2B_BC_TILES(m2b_db_tiles_f16, __half, false)
DEFINE_M2B_BC_TILES(m2b_dc_tiles, float, true)
DEFINE_M2B_BC_TILES(m2b_dc_tiles_bf16, __nv_bfloat16, true)
DEFINE_M2B_BC_TILES(m2b_dc_tiles_f16, __half, true)

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
    int batch, int seq, int nh, int has_limit, float dt_min, float dt_max,
    int y_first /* this run's first y block */
) {
    __shared__ float s_dadt[M2_CHUNK];
    __shared__ float s_draw[M2_CHUNK];
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
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

#define DEFINE_M2B_COMBINE(NAME, T)                                            \
extern "C" __global__ void NAME(                                               \
    T* __restrict__ d_proj,                                                    \
    long long ld_proj,                                                         \
    int dt_off,                                                                \
    float* __restrict__ da_parts,                                              \
    float* __restrict__ dbias_parts,                                           \
    const float* __restrict__ mm,                                              \
    const float* __restrict__ qv,                                              \
    const float* __restrict__ inter,                                           \
    const float* __restrict__ ddtx,                                            \
    const float* __restrict__ chunk_dot,                                       \
    const float* __restrict__ dt,                                              \
    const float* __restrict__ acs,                                             \
    const T* __restrict__ proj,                                                \
    const float* __restrict__ dt_bias,                                         \
    const float* __restrict__ a_neg,                                           \
    int batch,                                                                 \
    int seq,                                                                   \
    int nh,                                                                    \
    int has_limit,                                                             \
    float dt_min,                                                              \
    float dt_max,                                                              \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    m2b_combine_body<T>(d_proj, ld_proj, dt_off, da_parts, dbias_parts, mm,    \
        qv, inter, ddtx, chunk_dot, dt, acs, proj, dt_bias, a_neg, batch,      \
        seq, nh, has_limit, dt_min, dt_max, y_first);                          \
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
    int batch, int seq, int nh, int hd,
    int y_first /* this run's first y block */
) {
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
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

#define DEFINE_M2B_D_PARTS(NAME, T)                                            \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ parts,                                                 \
    const float* __restrict__ dy,                                              \
    const T* __restrict__ xbc,                                                 \
    long long ld_xbc,                                                          \
    int batch,                                                                 \
    int seq,                                                                   \
    int nh,                                                                    \
    int hd,                                                                    \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    m2b_d_parts_body<T>(parts, dy, xbc, ld_xbc, batch, seq, nh, hd, y_first);  \
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
    long long rows, int d_ssm, int ng, int hd, float eps, int norm_before_gate,
    int y_first /* this run's first y block */
) {
    extern __shared__ float m2bn_sm[];
    __shared__ float scal[2];
    const long long r = blockIdx.x;
    const int g = y_first + blockIdx.y;
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

#define DEFINE_M2B_GATED_NORM(NAME, T)                                         \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dy,                                                    \
    T* __restrict__ d_proj,                                                    \
    long long ld_proj,                                                         \
    int z_off,                                                                 \
    float* __restrict__ dw_parts,                                              \
    const T* __restrict__ dout,                                                \
    long long ld_out,                                                          \
    int out_col,                                                               \
    const float* __restrict__ y,                                               \
    const T* __restrict__ proj,                                                \
    const float* __restrict__ w,                                               \
    long long rows,                                                            \
    int d_ssm,                                                                 \
    int ng,                                                                    \
    int hd,                                                                    \
    float eps,                                                                 \
    int norm_before_gate,                                                      \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    m2b_gated_norm_body<T>(dy, d_proj, ld_proj, z_off, dw_parts, dout,         \
        ld_out, out_col, y, proj, w, rows, d_ssm, ng, hd, eps,                 \
        norm_before_gate, y_first);                                            \
}
DEFINE_M2B_GATED_NORM(m2b_gated_norm, float)
DEFINE_M2B_GATED_NORM(m2b_gated_norm_bf16, __nv_bfloat16)
DEFINE_M2B_GATED_NORM(m2b_gated_norm_f16, __half)

// The gated norm backward of a group too wide to stage: the staged
// kernel's sums in its order, every u and gw read again from global memory
// and recomputed with its expressions where the staged kernel keeps them in
// shared memory, so every output carries its bits. A pass of the per-head
// sums takes one head a thread into `head_sum` (blockDim floats), and
// thread 0 adds the pass to the running total heads ascending. Grid
// (rows, ngroups), dynamic shared memory one float per thread.
template <typename T>
__device__ __forceinline__ float m2bs_u(
    const float* __restrict__ y, const T* __restrict__ proj, long long r,
    int d_ssm, long long ld_proj, int z_off, long long i, int norm_before_gate
) {
    const float yv = y[r * d_ssm + i];
    const float zv = to_f(proj[r * ld_proj + z_off + i]);
    return norm_before_gate ? yv : __fmul_rn(yv, m2_silu(zv));
}

template <typename T>
__device__ __forceinline__ float m2bs_gw(
    const T* __restrict__ dout, long long ld_out, int out_col,
    const T* __restrict__ proj, long long ld_proj, int z_off,
    const float* __restrict__ w, long long r, long long i, int norm_before_gate
) {
    const float d = to_f(dout[r * ld_out + out_col + i]);
    float d1 = d;
    if (norm_before_gate) d1 = __fmul_rn(d, m2_silu(to_f(proj[r * ld_proj + z_off + i])));
    return __fmul_rn(w[i], d1);
}

template <typename T>
__device__ __forceinline__ void m2b_gated_norm_stream_body(
    float* __restrict__ dy,
    T* __restrict__ d_proj, long long ld_proj, int z_off,
    float* __restrict__ dw_parts,
    const T* __restrict__ dout, long long ld_out, int out_col,
    const float* __restrict__ y,
    const T* __restrict__ proj,
    const float* __restrict__ w,
    long long rows, int d_ssm, int ng, int hd, float eps, int norm_before_gate,
    int y_first /* this run's first y block */
) {
    extern __shared__ float head_sum[];
    __shared__ float scal[3];
    const long long r = blockIdx.x;
    const int g = y_first + blockIdx.y;
    if (r >= rows || g >= ng) return;
    const int gs = d_ssm / ng;
    const int heads = gs / hd;
    const long long c0 = (long long)g * gs;
    if (threadIdx.x == 0) scal[2] = 0.0f;
    for (int h0 = 0; h0 < heads; h0 += blockDim.x) {
        const int hh = h0 + threadIdx.x;
        if (hh < heads) {
            float s = 0.0f;
            for (int p = 0; p < hd; p++) {
                const float v = m2bs_u(y, proj, r, d_ssm, ld_proj, z_off,
                                       c0 + (long long)hh * hd + p, norm_before_gate);
                s = __fmaf_rn(v, v, s);
            }
            head_sum[threadIdx.x] = s;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float tot = scal[2];
            const int n = min((int)blockDim.x, heads - h0);
            for (int k = 0; k < n; k++) tot = __fadd_rn(tot, head_sum[k]);
            scal[2] = tot;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float rs = __fdiv_rn(1.0f, sqrtf(__fadd_rn(__fdiv_rn(scal[2], (float)gs), eps)));
        scal[0] = isfinite(rs) ? rs : 0.0f;
        scal[2] = 0.0f;
    }
    __syncthreads();
    const float rs = scal[0];
    for (int i = threadIdx.x; i < gs; i += blockDim.x) {
        const long long ci = c0 + i;
        const float d = to_f(dout[r * ld_out + out_col + ci]);
        const float wv = w[ci];
        const float xh = __fmul_rn(
            m2bs_u(y, proj, r, d_ssm, ld_proj, z_off, ci, norm_before_gate), rs);
        float d1 = d;
        if (norm_before_gate) {
            const float zv = to_f(proj[r * ld_proj + z_off + ci]);
            d1 = __fmul_rn(d, m2_silu(zv));
            m2b_put(&d_proj[r * ld_proj + z_off + ci],
                    __fmul_rn(__fmul_rn(d, __fmul_rn(xh, wv)), m2b_silu_grad(zv)));
        }
        dw_parts[r * d_ssm + ci] = __fmul_rn(d1, xh);
    }
    for (int h0 = 0; h0 < heads; h0 += blockDim.x) {
        const int hh = h0 + threadIdx.x;
        if (hh < heads) {
            float s = 0.0f;
            for (int p = 0; p < hd; p++) {
                const long long ci = c0 + (long long)hh * hd + p;
                const float u = m2bs_u(y, proj, r, d_ssm, ld_proj, z_off, ci, norm_before_gate);
                const float gw = m2bs_gw(dout, ld_out, out_col, proj, ld_proj, z_off, w, r,
                                         ci, norm_before_gate);
                s = __fmaf_rn(__fmul_rn(u, rs), gw, s);
            }
            head_sum[threadIdx.x] = s;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float tot = scal[2];
            const int n = min((int)blockDim.x, heads - h0);
            for (int k = 0; k < n; k++) tot = __fadd_rn(tot, head_sum[k]);
            scal[2] = tot;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) scal[1] = __fdiv_rn(scal[2], (float)gs);
    __syncthreads();
    const float c1 = scal[1];
    for (int i = threadIdx.x; i < gs; i += blockDim.x) {
        const long long ci = c0 + i;
        const float xh = __fmul_rn(
            m2bs_u(y, proj, r, d_ssm, ld_proj, z_off, ci, norm_before_gate), rs);
        const float gw = m2bs_gw(dout, ld_out, out_col, proj, ld_proj, z_off, w, r, ci,
                                 norm_before_gate);
        const float du = __fmul_rn(rs, __fsub_rn(gw, __fmul_rn(xh, c1)));
        if (norm_before_gate) {
            dy[r * d_ssm + ci] = du;
        } else {
            const float yv = y[r * d_ssm + ci];
            const float zv = to_f(proj[r * ld_proj + z_off + ci]);
            dy[r * d_ssm + ci] = __fmul_rn(du, m2_silu(zv));
            m2b_put(&d_proj[r * ld_proj + z_off + ci],
                    __fmul_rn(__fmul_rn(du, yv), m2b_silu_grad(zv)));
        }
    }
}

#define DEFINE_M2B_GATED_NORM_STREAM(NAME, T)                                  \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dy,                                                    \
    T* __restrict__ d_proj,                                                    \
    long long ld_proj,                                                         \
    int z_off,                                                                 \
    float* __restrict__ dw_parts,                                              \
    const T* __restrict__ dout,                                                \
    long long ld_out,                                                          \
    int out_col,                                                               \
    const float* __restrict__ y,                                               \
    const T* __restrict__ proj,                                                \
    const float* __restrict__ w,                                               \
    long long rows,                                                            \
    int d_ssm,                                                                 \
    int ng,                                                                    \
    int hd,                                                                    \
    float eps,                                                                 \
    int norm_before_gate,                                                      \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    m2b_gated_norm_stream_body<T>(dy, d_proj, ld_proj, z_off, dw_parts, dout,  \
        ld_out, out_col, y, proj, w, rows, d_ssm, ng, hd, eps,                 \
        norm_before_gate, y_first);                                            \
}
DEFINE_M2B_GATED_NORM_STREAM(m2b_gated_norm_stream, float)
DEFINE_M2B_GATED_NORM_STREAM(m2b_gated_norm_stream_bf16, __nv_bfloat16)
DEFINE_M2B_GATED_NORM_STREAM(m2b_gated_norm_stream_f16, __half)

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
// the window from the state the window started from, a tap of another
// sequence of a packed row 0). Grid 1-D over B * T * conv_dim.
template <typename T>
__device__ __forceinline__ void m2b_conv_dv_body(
    float* __restrict__ dv,
    const float* __restrict__ dxbc,
    const float* __restrict__ state0,
    const T* __restrict__ proj, long long ld_proj, int xbc_col,
    const float* __restrict__ w, const float* __restrict__ bias,
    int batch, int seq, int conv_dim, int d_conv,
    const int* __restrict__ seq_idx
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)batch * seq * conv_dim;
    if (i >= total) return;
    const int c = (int)(i % conv_dim);
    const long long row = i / conv_dim;
    const int t = (int)(row % seq);
    const int b = (int)(row / seq);
    const int sid = vl_id(seq_idx, b, seq, t);
    float v = bias[c];
    for (int k = 0; k < d_conv; k++) {
        int src = t - (d_conv - 1) + k;
        float xv = src >= 0
            ? to_f(proj[((long long)b * seq + src) * ld_proj + xbc_col + c])
            : state0[((long long)b * conv_dim + c) * d_conv + d_conv + src];
        if (vl_id(seq_idx, b, seq, src) != sid) xv = 0.0f;
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
    int d_conv,                                                                 \
    const int* __restrict__ seq_idx                                             \
) {                                                                             \
    m2b_conv_dv_body<T>(dv, dxbc, state0, proj, ld_proj, xbc_col, w, bias,      \
        batch, seq, conv_dim, d_conv, seq_idx);                                 \
}
DEFINE_M2B_CONV_DV(m2b_conv_dv, float)
DEFINE_M2B_CONV_DV(m2b_conv_dv_bf16, __nv_bfloat16)
DEFINE_M2B_CONV_DV(m2b_conv_dv_f16, __half)

// The conv's input gradient into the xBC columns of d_proj:
// d_in[t] = sum_k w[k] dv[t + d_conv - 1 - k] over taps inside the window
// (and, in a packed row, inside t's sequence), k ascending. Grid 1-D over
// B * T * conv_dim.
template <typename T>
__device__ __forceinline__ void m2b_conv_dx_body(
    T* __restrict__ d_proj, long long ld_proj, int xbc_col,
    const float* __restrict__ dv,
    const float* __restrict__ w,
    int batch, int seq, int conv_dim, int d_conv,
    const int* __restrict__ seq_idx
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)batch * seq * conv_dim;
    if (i >= total) return;
    const int c = (int)(i % conv_dim);
    const long long row = i / conv_dim;
    const int t = (int)(row % seq);
    const long long b = row / seq;
    const int sid = vl_id(seq_idx, b, seq, t);
    float acc = 0.0f;
    for (int k = 0; k < d_conv; k++) {
        int dst = t + (d_conv - 1) - k;
        if (dst < seq && vl_id(seq_idx, b, seq, dst) == sid)
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
    int d_conv,                                                                 \
    const int* __restrict__ seq_idx                                             \
) {                                                                             \
    m2b_conv_dx_body<T>(d_proj, ld_proj, xbc_col, dv, w, batch, seq,            \
        conv_dim, d_conv, seq_idx);                                             \
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
// sequence's first token come from state0). Grid (ceil(conv_dim / 32),
// ceil((d_conv + 1) / warps)), block (32, warps) with warps at most 32:
// a block takes the taps blockIdx.y * warps onward and stages the same
// rows as the others. A tap of another sequence of a packed row reads 0,
// as the forward. `g_b` may be null.
#define M2B_DW_ROWS 128
template <typename T>
__device__ __forceinline__ void m2b_conv_dw_body(
    float* __restrict__ g_w, float* __restrict__ g_b,
    const float* __restrict__ dv,
    const float* __restrict__ state0,
    const T* __restrict__ proj, long long ld_proj, int xbc_col,
    int batch, int seq, int conv_dim, int d_conv,
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m2bc_sm[];
    constexpr int CH = 32;
    const int halo = d_conv - 1;
    float* dvs = m2bc_sm;                    // [M2B_DW_ROWS][CH]
    float* xs = dvs + M2B_DW_ROWS * CH;      // [M2B_DW_ROWS + halo][CH]
    const int c0 = blockIdx.x * CH;
    const int lane = threadIdx.x, k = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = c0 + lane;
    const int tid = threadIdx.y * CH + lane, nthr = CH * blockDim.y;
    const long long rows = (long long)batch * seq;
    const bool live = c < conv_dim && (k < d_conv || (k == d_conv && g_b != nullptr));
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
                    float xv = src >= 0
                        ? xs[(rr + k) * CH + lane]
                        : state0[(b * conv_dim + c) * d_conv + d_conv + src];
                    if (vl_id(seq_idx, b, seq, src) != vl_id(seq_idx, b, seq, t)) xv = 0.0f;
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
    int d_conv,                                                                 \
    const int* __restrict__ seq_idx                                             \
) {                                                                             \
    m2b_conv_dw_body<T>(g_w, g_b, dv, state0, proj, ld_proj, xbc_col,           \
        batch, seq, conv_dim, d_conv, seq_idx);                                 \
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
// group's heads are summed by m2s_bc_groups. In a packed row a token that
// starts a sequence replays from the empty state, takes no alpha gradient
// and passes no dS back. Dynamic shared memory:
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
    int flags, float dt_min, float dt_max,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m2sb_sm[];
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
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
                if (vl_starts(seq_idx, b, seq, t)) {
                    for (int n = 0; n < ds; n++) trow[n] = 0.0f;
                }
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
            // A sequence's first token steps from the empty state: no alpha
            // gradient, and nothing of dS reaches the token before.
            const bool fresh = vl_starts(seq_idx, b, seq, t);
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
                    if (!fresh) dal_loc = __fmaf_rn(dsv, st[tid * (ds + 1) + n], dal_loc);
                    drow[n] = fresh ? 0.0f : __fmul_rn(alpha, dsv);
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

#define DEFINE_M2S_BWD(NAME, T)                                                \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dxbc,                                                  \
    T* __restrict__ d_proj, long long ld_proj, int dt_off,                     \
    float* __restrict__ db_h, float* __restrict__ dc_h,                        \
    float* __restrict__ da_parts, float* __restrict__ dbias_parts,             \
    float* __restrict__ rec,                                                   \
    const float* __restrict__ dy,                                              \
    const T* __restrict__ xbc, const T* __restrict__ proj,                     \
    const float* __restrict__ ckpt,                                            \
    const float* __restrict__ dt_bias, const float* __restrict__ a_neg,        \
    const float* __restrict__ D,                                               \
    int seq, int stride, int nh, int ng, int hd, int ds, int conv_dim,         \
    int flags, float dt_min, float dt_max,                                     \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m2s_bwd_body<T>(dxbc, d_proj, ld_proj, dt_off, db_h, dc_h, da_parts,       \
        dbias_parts, rec, dy, xbc, proj, ckpt, dt_bias, a_neg, D, seq, stride, \
        nh, ng, hd, ds, conv_dim, flags, dt_min, dt_max, y_first, seq_idx);    \
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

// The same copy issued by threads first .. first + count - 1 only, element
// by element, so the other threads go on with their own work; every thread
// commits (an empty group for those that issued nothing).
__device__ __forceinline__ void m2s_tile_fetch_by(
    float* __restrict__ sh, const float* __restrict__ gl, int hd, int ds, int first, int count
) {
    const int me = (int)threadIdx.x - first;
    if (me >= 0 && me < count) {
        int p = me / ds, n = me - (me / ds) * ds;
        for (int e = me; e < hd * ds; e += count) {
            m2s_cp4(sh + p * (ds + 1) + n, gl + e);
            n += count;
            while (n >= ds) {
                n -= ds;
                p++;
            }
        }
    }
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
// chain and both trees are m2s_bwd's, so the bits are its own, and so are
// its masks in a packed row. Dynamic shared memory: 3 P (N + 1) + 2 N +
// 2 P + 2 lanes floats (lanes = P rounded up to a warp).
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
    int flags, float dt_min, float dt_max,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m2sb_sm[];
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
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
        // The replay's per-token inputs (this thread's B and C index, and
        // for a channel its x and the step scalars) are loaded a token
        // ahead into registers, so a token starts with no global round trip.
        const bool one_each = ds <= (int)blockDim.x;
        float bq = 0.0f, cq = 0.0f, xq = 0.0f, s0q = 0.0f, s1q = 0.0f, s2q = 0.0f;
        auto fetch_in = [&](int t) {
            const long long row = (long long)b * seq + t;
            const long long xrow = row * conv_dim;
            if (one_each && tid < ds) {
                bq = to_f(xbc[xrow + d_ssm + (long long)g * ds + tid]);
                cq = to_f(xbc[xrow + d_ssm + (long long)(ng + g) * ds + tid]);
            }
            if (own) {
                const float* sc = scal3 + (row * nh + h) * 3;
                xq = to_f(xbc[xrow + (long long)h * hd + tid]);
                s0q = sc[0];
                s1q = sc[1];
                s2q = sc[2];
            }
        };
        fetch_in(t0);
        __syncthreads();
        for (int t = t0; t < t1; t++) {
            const long long row = (long long)b * seq + t;
            const long long xrow = row * conv_dim;
            if (one_each) {
                if (tid < ds) {
                    b_sm[tid] = bq;
                    c_sm[tid] = cq;
                }
            } else {
                for (int n = tid; n < ds; n += blockDim.x) {
                    b_sm[n] = to_f(xbc[xrow + d_ssm + (long long)g * ds + n]);
                    c_sm[n] = to_f(xbc[xrow + d_ssm + (long long)(ng + g) * ds + n]);
                }
            }
            const float xp = xq, a0 = s0q, a1 = s1q, a2 = s2q;
            if (t + 1 < t1) fetch_in(t + 1);
            __syncthreads();
            if (own) {
                if (vl_starts(seq_idx, b, seq, t)) {
                    for (int n = 0; n < ds; n++) st[tid * ld + n] = 0.0f;
                }
                m2_row_step(st + tid * ld, b_sm, c_sm, a0, a1, a2, xp, ds);
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
            // A sequence's first token steps from the empty state: no alpha
            // gradient, and nothing of dS reaches the token before.
            const bool fresh = vl_starts(seq_idx, b, seq, t);
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
            // whole block: a thread keeps one n and takes four rows at a time
            // when the block is a whole number of rows wide.
            if (blockDim.x % ds == 0) {
                const int n = tid % ds, step = blockDim.x / ds;
                const float cn = c_sm[n];
                for (int p0 = tid / ds; p0 < hd; p0 += 4 * step) {
                    float dv[4], yv[4];
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        const int p = min(p0 + u * step, hd - 1);
                        dv[u] = dsr[p * ld + n];
                        yv[u] = dyS[p];
                    }
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        const int p = p0 + u * step;
                        if (p < hd) dsr[p * ld + n] = __fmaf_rn(yv[u], cn, dv[u]);
                    }
                }
            } else {
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
            // while the rest of this step runs, issued by the threads past
            // the channels' two chains when the block has them.
            const bool split = (int)blockDim.x >= lanes + hd;
            const int helpers = (int)blockDim.x - lanes - hd;
            if (t > t0) {
                const float* src = slot > 1 ? rec + (bh * stride + slot - 2) * pn : ck;
                if (split && helpers >= 32)
                    m2s_tile_fetch_by(cur, src, hd, ds, lanes + hd, helpers);
                else
                    m2s_tile_fetch(cur, src, hd, ds);
            }
            // Per channel row the dt chain (sx = dS . B) and the alpha chain
            // (dS . the state before the token), n ascending; with room they
            // run on two threads (the alpha chain's sum goes through its slot
            // of the block sums), then the whole block scales dS by alpha.
            float ddt_loc = 0.0f, dal_loc = 0.0f;
            if (own) {
                const float* drow = dsr + tid * ld;
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
                            pv[j] = split ? 0.0f : prow[n + j];
                        }
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            sx = __fmaf_rn(dv[j], bb[j], sx);
                            if (!split && !fresh) dal_loc = __fmaf_rn(dv[j], pv[j], dal_loc);
                        }
                    }
                }
                for (; n < ds; n++) {
                    const float dsv = drow[n];
                    sx = __fmaf_rn(dsv, b_sm[n], sx);
                    if (!split && !fresh) dal_loc = __fmaf_rn(dsv, prow[n], dal_loc);
                }
                dxbc[xrow + (long long)h * hd + tid] = __fmaf_rn(dt, sx, __fmul_rn(dval, dyv));
                ddt_loc = __fmul_rn(xp, sx);
            } else if (split && tid >= lanes && tid < lanes + hd) {
                const int p = tid - lanes;
                const float* drow = dsr + p * ld;
                const float* prow = prev + p * ld;
                float dal = 0.0f;
                int n = 0;
                for (; n + 4 <= ds; n += 4) {
                    float dv[4], pv[4];
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        dv[j] = drow[n + j];
                        pv[j] = prow[n + j];
                    }
#pragma unroll
                    for (int j = 0; j < 4; j++) dal = __fmaf_rn(dv[j], pv[j], dal);
                }
                for (; n < ds; n++) dal = __fmaf_rn(drow[n], prow[n], dal);
                red[lanes + p] = fresh ? 0.0f : dal;
            }
            __syncthreads();
            if (split && own) dal_loc = red[lanes + tid];
            {
                int p = tid / ds, n = tid - (tid / ds) * ds;
                for (int e = tid; e < hd * ds; e += blockDim.x) {
                    float* d = dsr + p * ld + n;
                    *d = fresh ? 0.0f : __fmul_rn(alpha, *d);
                    n += blockDim.x;
                    while (n >= ds) {
                        n -= ds;
                        p++;
                    }
                }
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

#define DEFINE_M2S_BWD_TILES(NAME, T)                                          \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dxbc,                                                  \
    T* __restrict__ d_proj, long long ld_proj, int dt_off,                     \
    float* __restrict__ db_h, float* __restrict__ dc_h,                        \
    float* __restrict__ da_parts, float* __restrict__ dbias_parts,             \
    float* __restrict__ rec,                                                   \
    const float* __restrict__ dy,                                              \
    const T* __restrict__ xbc, const T* __restrict__ proj,                     \
    const float* __restrict__ ckpt, const float* __restrict__ scal3,           \
    const float* __restrict__ dt_bias, const float* __restrict__ a_neg,        \
    const float* __restrict__ D,                                               \
    int seq, int stride, int nh, int ng, int hd, int ds, int conv_dim,         \
    int flags, float dt_min, float dt_max,                                     \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m2s_bwd_tiles_body<T>(dxbc, d_proj, ld_proj, dt_off, db_h, dc_h, da_parts, \
        dbias_parts, rec, dy, xbc, proj, ckpt, scal3, dt_bias, a_neg, D, seq,  \
        stride, nh, ng, hd, ds, conv_dim, flags, dt_min, dt_max, y_first,      \
        seq_idx);                                                              \
}
DEFINE_M2S_BWD_TILES(m2s_bwd_tiles, float)
DEFINE_M2S_BWD_TILES(m2s_bwd_tiles_bf16, __nv_bfloat16)
DEFINE_M2S_BWD_TILES(m2s_bwd_tiles_f16, __half)

// One token of the replay over the whole block: h[p][n] = e^acs h_prev[p][n]
// + B[n] (x[p] e^0 dt), m2_row_step's update, four elements a step when
// `vec` (dense rows on 16 bytes); a token that starts a sequence of a
// packed row (`fresh`) steps from the empty state. A thread takes the same
// elements at every token, so it reads only what it wrote the token before.
template <typename T>
__device__ __forceinline__ void m2sg_replay(
    float* __restrict__ dst, const float* __restrict__ src,
    const T* __restrict__ xh, const T* __restrict__ bv,
    float eacs, float xs, int hd, int ds, bool vec, bool fresh
) {
    const int tid = threadIdx.x, nt = blockDim.x;
    const long long pn = (long long)hd * ds;
    const int w = vec ? 4 : 1;
    int p = w * tid / ds, n = w * tid - (w * tid / ds) * ds;
    const int dp = w * nt / ds, dn = w * nt - (w * nt / ds) * ds;
    if (vec) {
        for (long long e = 4LL * tid; e < pn; e += 4LL * nt) {
            const float xw = __fmul_rn(to_f(xh[p]), xs);
            const float4 s = fresh ? make_float4(0.0f, 0.0f, 0.0f, 0.0f)
                                   : *reinterpret_cast<const float4*>(src + e);
            float4 o;
            o.x = __fmaf_rn(eacs, s.x, __fmaf_rn(to_f(bv[n]), xw, 0.0f));
            o.y = __fmaf_rn(eacs, s.y, __fmaf_rn(to_f(bv[n + 1]), xw, 0.0f));
            o.z = __fmaf_rn(eacs, s.z, __fmaf_rn(to_f(bv[n + 2]), xw, 0.0f));
            o.w = __fmaf_rn(eacs, s.w, __fmaf_rn(to_f(bv[n + 3]), xw, 0.0f));
            *reinterpret_cast<float4*>(dst + e) = o;
            p += dp;
            n += dn;
            if (n >= ds) {
                n -= ds;
                p++;
            }
        }
    } else {
        for (long long e = tid; e < pn; e += nt) {
            const float xw = __fmul_rn(to_f(xh[p]), xs);
            dst[e] = __fmaf_rn(eacs, fresh ? 0.0f : src[e], __fmaf_rn(to_f(bv[n]), xw, 0.0f));
            p += dp;
            n += dn;
            if (n >= ds) {
                n -= ds;
                p++;
            }
        }
    }
}

// dS[p][n] = dy[p] C[n] + dS[p][n] over the whole block, dS first scaled by
// the alpha of the token after this one when `scale` (every token but the
// window's last), rounded before the fma as m2s_bwd rounds it into dS, or
// dropped when `cut` (the token after this one starts a sequence of a
// packed row). Rows `ld` apart, four elements a step when `vec` (dense
// rows on 16 bytes).
__device__ __forceinline__ void m2sg_ds_update(
    float* __restrict__ dsr, int ld, const float* __restrict__ dyh,
    const float* __restrict__ c_sm, float alpha, bool scale, bool cut, int hd, int ds,
    bool vec
) {
    const int tid = threadIdx.x, nt = blockDim.x;
    const long long pn = (long long)hd * ds;
    const int w = vec ? 4 : 1;
    int p = w * tid / ds, n = w * tid - (w * tid / ds) * ds;
    const int dp = w * nt / ds, dn = w * nt - (w * nt / ds) * ds;
    if (vec) {
        for (long long e = 4LL * tid; e < pn; e += 4LL * nt) {
            float4 v = *reinterpret_cast<const float4*>(dsr + e);
            if (cut) {
                v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            } else if (scale) {
                v.x = __fmul_rn(alpha, v.x);
                v.y = __fmul_rn(alpha, v.y);
                v.z = __fmul_rn(alpha, v.z);
                v.w = __fmul_rn(alpha, v.w);
            }
            const float y = dyh[p];
            v.x = __fmaf_rn(y, c_sm[n], v.x);
            v.y = __fmaf_rn(y, c_sm[n + 1], v.y);
            v.z = __fmaf_rn(y, c_sm[n + 2], v.z);
            v.w = __fmaf_rn(y, c_sm[n + 3], v.w);
            *reinterpret_cast<float4*>(dsr + e) = v;
            p += dp;
            n += dn;
            if (n >= ds) {
                n -= ds;
                p++;
            }
        }
    } else {
        for (long long e = tid; e < pn; e += nt) {
            float* d = dsr + (long long)p * ld + n;
            const float v = cut ? 0.0f : scale ? __fmul_rn(alpha, *d) : *d;
            *d = __fmaf_rn(dyh[p], c_sm[n], v);
            p += dp;
            n += dn;
            if (n >= ds) {
                n -= ds;
                p++;
            }
        }
    }
}

// m2s_bwd for a head whose state tiles do not fit shared memory. The
// replayed states stay in the tape, and dS stays in shared memory when its
// rows fit there (padded by one), else in a global scratch [B, H, P, N].
// The replay runs element by element over the whole block. The walk back
// spreads the dS update over the whole block, puts a state index's dC or
// dB chain on a thread and a channel's two row chains on a thread, and
// leaves the scaling of dS by alpha to the next token's update. A head
// wider than the block (blockDim a power of two then) gives each thread
// the channels p, p + blockDim and so on: the leaves of the two block sums
// past a thread's first go to a global scratch [B, H, 2, lanes], and the
// levels of the tree at or above the block width, which pair positions of
// one thread only, run in that thread before the levels in shared memory.
// Every chain and both trees are m2s_bwd's, so the bits are its own, and so
// are its masks in a packed row. Dynamic shared memory: 2 N + 2 K floats,
// plus P (N + 1) with dS there (lanes = P rounded up to a warp, K = lanes,
// or blockDim for a wider head).
template <typename T>
__device__ __forceinline__ void m2s_bwd_global_body(
    float* __restrict__ dxbc,
    T* __restrict__ d_proj, long long ld_proj, int dt_off,
    float* __restrict__ db_h, float* __restrict__ dc_h,
    float* __restrict__ da_parts, float* __restrict__ dbias_parts,
    float* __restrict__ rec, float* __restrict__ ds_glob, float* __restrict__ leaves,
    const float* __restrict__ dy,
    const T* __restrict__ xbc, const T* __restrict__ proj,
    const float* __restrict__ ckpt, const float* __restrict__ scal3,
    const float* __restrict__ dt_bias, const float* __restrict__ a_neg,
    const float* __restrict__ D,
    int seq, int stride, int nh, int ng, int hd, int ds, int conv_dim,
    int flags, float dt_min, float dt_max,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m2sb_sm[];
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int g = h / (nh / ng);
    const long long d_ssm = (long long)nh * hd;
    const long long bh = (long long)b * nh + h;
    const int nseg = (seq + stride - 1) / stride;
    const long long pn = (long long)hd * ds;
    const int lanes = (hd + 31) & ~31;
    const int kl = lanes <= nt ? lanes : nt;
    const bool shared_ds = ds_glob == nullptr;
    const int ld = shared_ds ? ds + 1 : ds;
    float* dsr = shared_ds ? m2sb_sm : ds_glob + bh * pn;
    float* b_sm = shared_ds ? m2sb_sm + hd * (ds + 1) : m2sb_sm;
    float* c_sm = b_sm + ds;
    float* red = c_sm + ds;
    float* gl = leaves == nullptr ? nullptr : leaves + bh * 2 * lanes;
    float* tape = rec + bh * stride * pn;
    const unsigned long long bases = reinterpret_cast<unsigned long long>(rec)
        | reinterpret_cast<unsigned long long>(ckpt)
        | (shared_ds ? 0ULL : reinterpret_cast<unsigned long long>(ds_glob));
    const bool vec = (ds & 3) == 0 && (bases & 15) == 0;
    if (shared_ds) {
        for (int e = tid; e < hd * ld; e += nt) dsr[e] = 0.0f;
    } else {
        for (long long e = tid; e < pn; e += nt) dsr[e] = 0.0f;
    }
    __syncthreads();
    float da_acc = 0.0f, dbias_acc = 0.0f;
    float alpha_next = 1.0f;
    bool scale = false, cut = false;
    for (int seg = nseg - 1; seg >= 0; seg--) {
        const int t0 = seg * stride;
        const int t1 = min(t0 + stride, seq);
        const float* ck = ckpt + (((long long)b * nseg + seg) * nh + h) * pn;
        for (int t = t0; t < t1; t++) {
            const long long row = (long long)b * seq + t;
            const long long xrow = row * conv_dim;
            const float* sc = scal3 + (row * nh + h) * 3;
            const float* src = t > t0 ? tape + (long long)(t - 1 - t0) * pn : ck;
            m2sg_replay(tape + (long long)(t - t0) * pn, src, xbc + xrow + (long long)h * hd,
                        xbc + xrow + d_ssm + (long long)g * ds, sc[0], sc[2], hd, ds, vec,
                        vl_starts(seq_idx, b, seq, t));
        }
        __syncthreads();
        for (int t = t1 - 1; t >= t0; t--) {
            const long long row = (long long)b * seq + t;
            const long long xrow = row * conv_dim;
            const int slot = t - t0;
            const float* h_cur = tape + (long long)slot * pn;
            const float* h_prev = slot > 0 ? tape + (long long)(slot - 1) * pn : ck;
            for (int n = tid; n < ds; n += nt) {
                b_sm[n] = to_f(xbc[xrow + d_ssm + (long long)g * ds + n]);
                c_sm[n] = to_f(xbc[xrow + d_ssm + (long long)(ng + g) * ds + n]);
            }
            const float* sc = scal3 + (row * nh + h) * 3;
            const float alpha = sc[0], dt = sc[2];
            const float* dyh = dy + row * d_ssm + (long long)h * hd;
            const T* xh = xbc + xrow + (long long)h * hd;
            // A sequence's first token steps from the empty state: no alpha
            // gradient, and nothing of dS reaches the token before.
            const bool fresh = vl_starts(seq_idx, b, seq, t);
            __syncthreads();
            m2sg_ds_update(dsr, ld, dyh, c_sm, alpha_next, scale, cut, hd, ds,
                           vec && !shared_ds);
            __syncthreads();
            // dC and dB per state index, the two chains of an index on two
            // threads, p ascending.
            for (int k = tid; k < 2 * ds; k += nt) {
                float acc = 0.0f;
                if (k < ds) {
#pragma unroll 4
                    for (int p = 0; p < hd; p++)
                        acc = __fmaf_rn(dyh[p], h_cur[(long long)p * ds + k], acc);
                    dc_h[(row * nh + h) * ds + k] = acc;
                } else {
                    const int n = k - ds;
#pragma unroll 4
                    for (int p = 0; p < hd; p++)
                        acc = __fmaf_rn(dsr[(long long)p * ld + n], to_f(xh[p]), acc);
                    db_h[(row * nh + h) * ds + n] = __fmul_rn(acc, dt);
                }
            }
            // Per channel the dt chain (sx = dS . B) and the alpha chain
            // (dS . the state before the token), n ascending, and the
            // channel's dx. The positions from P up to the lanes keep zero
            // leaves, as m2s_bwd's threads past the head do.
            float ddt0 = 0.0f, dal0 = 0.0f;
            for (int p = tid; p < lanes; p += nt) {
                float ddt_loc = 0.0f, dal_loc = 0.0f;
                if (p < hd) {
                    const float* drow = dsr + (long long)p * ld;
                    const float* prow = h_prev + (long long)p * ds;
                    float sx = 0.0f;
                    int n = 0;
                    if (vec) {
                        for (; n + 4 <= ds; n += 4) {
                            float dv[4];
                            if (shared_ds) {
#pragma unroll
                                for (int j = 0; j < 4; j++) dv[j] = drow[n + j];
                            } else {
                                const float4 d4 = *reinterpret_cast<const float4*>(drow + n);
                                dv[0] = d4.x;
                                dv[1] = d4.y;
                                dv[2] = d4.z;
                                dv[3] = d4.w;
                            }
                            const float4 p4 = *reinterpret_cast<const float4*>(prow + n);
                            const float pv[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
                            for (int j = 0; j < 4; j++) {
                                sx = __fmaf_rn(dv[j], b_sm[n + j], sx);
                                if (!fresh) dal_loc = __fmaf_rn(dv[j], pv[j], dal_loc);
                            }
                        }
                    }
                    for (; n < ds; n++) {
                        const float dsv = drow[n];
                        sx = __fmaf_rn(dsv, b_sm[n], sx);
                        if (!fresh) dal_loc = __fmaf_rn(dsv, prow[n], dal_loc);
                    }
                    const float dval =
                        (flags & M2_FLAG_D_PER_CHANNEL) ? D[(long long)h * hd + p] : D[h];
                    dxbc[xrow + (long long)h * hd + p] =
                        __fmaf_rn(dt, sx, __fmul_rn(dval, dyh[p]));
                    ddt_loc = __fmul_rn(to_f(xh[p]), sx);
                }
                if (p == tid) {
                    ddt0 = ddt_loc;
                    dal0 = dal_loc;
                } else {
                    gl[p] = ddt_loc;
                    gl[lanes + p] = dal_loc;
                }
            }
            // The tree's levels at or above the block width, on this
            // thread's own positions: position i + w is i's own when w is a
            // multiple of the block width.
            if (lanes > nt) {
                int width = 1;
                while (width < lanes) width <<= 1;
                for (int w = width >> 1; w >= nt; w >>= 1) {
                    if (tid + w < lanes) {
                        ddt0 = __fadd_rn(ddt0, gl[tid + w]);
                        dal0 = __fadd_rn(dal0, gl[lanes + tid + w]);
                    }
                    for (int i = tid + nt; i < w; i += nt) {
                        if (i + w < lanes) {
                            gl[i] = __fadd_rn(gl[i], gl[i + w]);
                            gl[lanes + i] = __fadd_rn(gl[lanes + i], gl[lanes + i + w]);
                        }
                    }
                }
            }
            const float2 sums = m2s_block_sum2(ddt0, dal0, red, kl);
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
            alpha_next = alpha;
            scale = true;
            cut = fresh;
        }
    }
    if (tid == 0) {
        da_parts[bh] = da_acc;
        dbias_parts[bh] = dbias_acc;
    }
}

#define DEFINE_M2S_BWD_GLOBAL(NAME, T)                                         \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dxbc,                                                  \
    T* __restrict__ d_proj, long long ld_proj, int dt_off,                     \
    float* __restrict__ db_h, float* __restrict__ dc_h,                        \
    float* __restrict__ da_parts, float* __restrict__ dbias_parts,             \
    float* __restrict__ rec, float* __restrict__ ds_glob,                      \
    float* __restrict__ leaves,                                                \
    const float* __restrict__ dy,                                              \
    const T* __restrict__ xbc, const T* __restrict__ proj,                     \
    const float* __restrict__ ckpt, const float* __restrict__ scal3,           \
    const float* __restrict__ dt_bias, const float* __restrict__ a_neg,        \
    const float* __restrict__ D,                                               \
    int seq, int stride, int nh, int ng, int hd, int ds, int conv_dim,         \
    int flags, float dt_min, float dt_max,                                     \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m2s_bwd_global_body<T>(dxbc, d_proj, ld_proj, dt_off, db_h, dc_h, da_parts, \
        dbias_parts, rec, ds_glob, leaves, dy, xbc, proj, ckpt, scal3, dt_bias, \
        a_neg, D, seq, stride, nh, ng, hd, ds, conv_dim, flags, dt_min, dt_max, \
        y_first, seq_idx);                                                     \
}
DEFINE_M2S_BWD_GLOBAL(m2s_bwd_global, float)
DEFINE_M2S_BWD_GLOBAL(m2s_bwd_global_bf16, __nv_bfloat16)
DEFINE_M2S_BWD_GLOBAL(m2s_bwd_global_f16, __half)

// out[j] += sum_r parts[r][j] over `rows` rows of `dim` columns: the
// shared reduce_sum_axis0's per-thread strided sums and halving tree, so
// its bits. That reduction's `width` threads (a power of two up to 256)
// each sum the rows v, v + width, ... of a column; here a block takes `cw`
// neighbouring columns (8, 16 or 32) and runs those sums for all of them,
// a lane a column, so a warp's loads cover whole sectors of a row instead
// of one float of 32 rows. Grid ceil(dim / cw), block 256, shared memory
// width * cw floats.
extern "C" __global__ void m2b_reduce_rows(
    float* __restrict__ out, const float* __restrict__ parts,
    long long rows, long long dim, int width, int cw
) {
    extern __shared__ float m2b_red[];  // [width][cw]
    const int lane = threadIdx.x & 31;
    const int c = lane % cw;
    const int slots = (blockDim.x >> 5) * (32 / cw);
    const int slot = (threadIdx.x >> 5) * (32 / cw) + lane / cw;
    const long long col = (long long)blockIdx.x * cw + c;
    const bool live = col < dim;
    // Eight of the thread's sums at once: their loads are independent, the
    // adds of each stay in row order.
    for (int v0 = slot; v0 < width; v0 += 8 * slots) {
        float sum[8];
#pragma unroll
        for (int u = 0; u < 8; u++) sum[u] = 0.0f;
        for (long long r0 = v0; r0 < rows; r0 += width) {
#pragma unroll
            for (int u = 0; u < 8; u++) {
                const long long r = r0 + (long long)u * slots;
                if (live && v0 + u * slots < width && r < rows) sum[u] += parts[r * dim + col];
            }
        }
#pragma unroll
        for (int u = 0; u < 8; u++) {
            const int v = v0 + u * slots;
            if (v < width) m2b_red[v * cw + c] = sum[u];
        }
    }
    __syncthreads();
    for (int s = width / 2; s > 0; s >>= 1) {
        for (int v = slot; v < s; v += slots) m2b_red[v * cw + c] += m2b_red[(v + s) * cw + c];
        __syncthreads();
    }
    if (slot == 0 && live) out[col] = out[col] + m2b_red[c];
}

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
