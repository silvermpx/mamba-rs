// Mamba-3 MIMO on the GPU, the chunked path: a window in chunks of 64
// tokens, the chunks in parallel.
//
// The trapezoid folds the way SISO's chunked form folds it. Token s's rank
// inputs u_s[r] = Psi_r x_s and rotated keys k_s[r] enter the state with
// gamma_s = lam_s dt_s at their own step and with beta_{s+1} = alpha_{s+1}
// dt_{s+1} lam_c_{s+1} at the next one, so from step s + 1 on they weigh
//   scale_s = gamma_s + dt_{s+1} lam_c_{s+1}     (gamma_s alone at the last token)
// and within a chunk, with Acs the running sum of a dt,
//   S_t = e^Acs_t S_enter + sum_{s<t} e^(Acs_t - Acs_s) scale_s U_s K_s + gamma_t U_t K_t
//   y_t[r] = q_t[r] . S_t + D u_t[r]
// The first token's beta term reads the carried previous keys and input:
// the state entering the window is S_init + dt_0 lam_c_0 sum_r (Psi_r v) kp_r.
// A chunk's own contribution (with every token's scale, the last one's
// reaching into the next chunk) is decayed into the next chunk's entering
// state by the state passing.
//
// The pieces: m3m_prep (step scalars and angles per token, the angles in
// the sequential path's own arithmetic), m3m_chunk_state, m3m_state_pass,
// m3m_chunk_scan (the outputs and the gate), m3m_writeback (the carried
// keys and input). Every value is one thread's chain in a fixed order; the
// bits do not depend on the batch or the grid.
//
// Composed after mamba3_mimo.cu in the MIMO module.

#define M3M_CHUNK 64

// The rotated value at n of a biased key or query rank row (the group's
// normalised row `src` plus the head's `bias`), the half-split pairs
// (a, a + N/2) for a < na turned by the token's angles: the arithmetic of
// mimo_tok_keys and mimo_tok_rotate.
__device__ __forceinline__ float m3m_rot(
    const float* src, const float* bias, int n, int half, int na,
    const float* sr, const float* cr
) {
    const float v = __fadd_rn(src[n], bias[n]);
    if (n < na) {
        const float v1 = __fadd_rn(src[n + half], bias[n + half]);
        return __fsub_rn(__fmul_rn(cr[n], v), __fmul_rn(sr[n], v1));
    }
    if (n >= half && n - half < na) {
        const int a = n - half;
        const float v0 = __fadd_rn(src[a], bias[a]);
        return __fadd_rn(__fmul_rn(sr[a], v0), __fmul_rn(cr[a], v));
    }
    return v;
}

// The step scalars and angles of every token of the window, one block per
// (sequence, head). Thread 0 walks the tokens: dt, gamma = lam dt, the
// chunk-local Acs (a running sum from zero at each chunk start) and scale;
// then the angle threads advance the head's angles over the window (the
// sequential path's double-precision wrap) and record their sines and
// cosines. coef [B, T, H, 4] = (dt, gamma, scale, Acs); sn, cs [B, T, H, na];
// e0 [B, H] = dt_0 lam_c_0. The angle state ends advanced over the window.
// Block max(32, na) threads.
template <typename TI>
__device__ __forceinline__ void m3m_prep_body(
    float* __restrict__ coef, float* __restrict__ sn, float* __restrict__ cs,
    float* __restrict__ e0, float* __restrict__ ang,
    const TI* __restrict__ proj, long long ld_proj, const float* __restrict__ dt_bias,
    int seq, int nh, int heads, int ang_col, int na, float a_floor
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const long long bh = (long long)b * nh + h;
    if (threadIdx.x == 0) {
        float acs = 0.0f, prev_gamma = 0.0f;
        for (int t = 0; t < seq; t++) {
            const long long row = (long long)b * seq + t;
            const TI* pr = proj + row * ld_proj;
            float dd_a = to_f(pr[heads + nh + h]);
            float ht = dd_a >= 0.0f ? __fadd_rn(1.0f, dd_a)
                                    : __fdiv_rn(1.0f, __fsub_rn(1.0f, dd_a));
            float a = -ht;
            a = a > -a_floor ? -a_floor : a;
            float spin = __fadd_rn(to_f(pr[heads + h]), dt_bias[h]);
            float dt = spin > 20.0f ? spin : log1pf(mimo_exp(spin));
            float trap = to_f(pr[heads + 2 * nh + h]);
            float lam = mimo_sigmoid(trap);
            float lam_c = mimo_sigmoid(-trap);
            float gamma = __fmul_rn(lam, dt);
            float lead = __fmul_rn(dt, lam_c);
            if (t % M3M_CHUNK == 0) acs = 0.0f;
            acs = __fadd_rn(acs, __fmul_rn(a, dt));
            float* c = coef + (row * nh + h) * 4;
            c[0] = dt;
            c[1] = gamma;
            c[3] = acs;
            if (t == 0) e0[bh] = lead;
            else coef[((row - 1) * nh + h) * 4 + 2] = __fadd_rn(prev_gamma, lead);
            prev_gamma = gamma;
        }
        coef[(((long long)b * seq + seq - 1) * nh + h) * 4 + 2] = prev_gamma;
    }
    __syncthreads();
    for (int a = threadIdx.x; a < na; a += blockDim.x) {
        float theta = ang[bh * na + a];
        for (int t = 0; t < seq; t++) {
            const long long row = (long long)b * seq + t;
            const TI* pr = proj + row * ld_proj;
            const float dt = coef[(row * nh + h) * 4];
            float tp = mimo_tanh_pi(pr, ang_col, a);
            const double two_pi = 6.283185307179586;
            double acc = (double)theta + (double)__fmul_rn(tp, dt);
            acc -= two_pi * floor(acc / two_pi);
            theta = (float)acc;
            sincosf(theta, &sn[(row * nh + h) * na + a], &cs[(row * nh + h) * na + a]);
        }
        ang[bh * na + a] = theta;
    }
}

#define DEFINE_M3M_PREP(NAME, TI)                                               \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ coef, float* __restrict__ sn, float* __restrict__ cs,   \
    float* __restrict__ e0, float* __restrict__ ang,                            \
    const TI* __restrict__ proj, long long ld_proj,                             \
    const float* __restrict__ dt_bias,                                          \
    int seq, int nh, int heads, int ang_col, int na, float a_floor              \
) {                                                                             \
    m3m_prep_body<TI>(coef, sn, cs, e0, ang, proj, ld_proj, dt_bias, seq, nh,   \
        heads, ang_col, na, a_floor);                                           \
}
DEFINE_M3M_PREP(m3m_prep, float)
DEFINE_M3M_PREP(m3m_prep_bf16, __nv_bfloat16)
DEFINE_M3M_PREP(m3m_prep_f16, __half)

// Each chunk's own contribution to the state, from zero:
//   C[p][n] = sum_s sum_r (e^min(Acs_last - Acs_s, 0) scale_s) u_s[r][p] k_s[r][n]
// (s, then r, ascending). One block per (chunk, head, n-tile); thread
// (p, n-in-tile). The chunk's rotated keys for the tile's n and its inputs
// are staged in shared memory `sl` tokens at a time (the whole chunk when
// it fits): sl * R * nt + sl * P + sl floats. The tile sizes change what a
// pass stages, never the order of a thread's sum.
// Grid (B * nc, H, ceil(N / nt)), block nt * P threads.
template <typename TI>
__device__ __forceinline__ void m3m_chunk_state_body(
    float* __restrict__ contrib,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ coef,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ b_bias, const float* __restrict__ mimo_x,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int nt, int sl
) {
    extern __shared__ float m3m_sm[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int n0 = blockIdx.z * nt;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    float* kt = m3m_sm;                // [sl][R][nt]
    float* xt = kt + sl * rank * nt;   // [sl][P]
    float* wt = xt + sl * hd;          // [sl]
    const int tid = threadIdx.x;
    const int p = tid % hd, nl = tid / hd;
    const bool active = nl < nt && n0 + nl < ds;
    const float acs_last = coef[(((b * seq + t0 + lc - 1) * nh) + h) * 4 + 3];
    float acc = 0.0f;
    for (int s0 = 0; s0 < lc; s0 += sl) {
        const int sn_ = min(sl, lc - s0);
        if (s0 > 0) __syncthreads();
        for (int i = tid; i < sn_ * rank * nt; i += blockDim.x) {
            const int s = i / (rank * nt), r = (i / nt) % rank, ni = i % nt;
            const int n = n0 + ni;
            float v = 0.0f;
            if (n < ds) {
                const long long row = b * seq + t0 + s0 + s;
                v = m3m_rot(bn + row * rgn + ((long long)r * ng + g) * ds,
                            b_bias + ((long long)h * rank + r) * ds, n, half, na,
                            sn + (row * nh + h) * na, cs + (row * nh + h) * na);
            }
            kt[i] = v;
        }
        for (int i = tid; i < sn_ * hd; i += blockDim.x) {
            const int s = i / hd, pp = i % hd;
            xt[i] = to_f(proj[(b * seq + t0 + s0 + s) * ld_proj + di + (long long)h * hd + pp]);
        }
        for (int s = tid; s < sn_; s += blockDim.x) {
            const float* cf = coef + ((b * seq + t0 + s0 + s) * nh + h) * 4;
            wt[s] = __fmul_rn(mimo_exp(fminf(__fsub_rn(acs_last, cf[3]), 0.0f)), cf[2]);
        }
        __syncthreads();
        if (active) {
            for (int s = 0; s < sn_; s++) {
                const float wx = __fmul_rn(wt[s], xt[s * hd + p]);
                for (int r = 0; r < rank; r++) {
                    const float u = __fmul_rn(wx, mimo_x[((long long)h * rank + r) * hd + p]);
                    acc = __fmaf_rn(u, kt[(s * rank + r) * nt + nl], acc);
                }
            }
        }
    }
    if (active) contrib[((bc * nh + h) * hd + p) * (long long)ds + n0 + nl] = acc;
}

#define DEFINE_M3M_CHUNK_STATE(NAME, TI)                                        \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ contrib,                                                \
    const TI* __restrict__ proj, long long ld_proj,                             \
    const float* __restrict__ bn, const float* __restrict__ coef,               \
    const float* __restrict__ sn, const float* __restrict__ cs,                 \
    const float* __restrict__ b_bias, const float* __restrict__ mimo_x,         \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int nt, int sl   \
) {                                                                             \
    m3m_chunk_state_body<TI>(contrib, proj, ld_proj, bn, coef, sn, cs, b_bias,  \
        mimo_x, seq, nh, ng, hd, ds, rank, na, nt, sl);                         \
}
DEFINE_M3M_CHUNK_STATE(m3m_chunk_state, float)
DEFINE_M3M_CHUNK_STATE(m3m_chunk_state_bf16, __nv_bfloat16)
DEFINE_M3M_CHUNK_STATE(m3m_chunk_state_f16, __half)

// The states entering the chunks: S = S_init + e0 sum_r (Psi_r v) kp_r
// (r ascending), then per chunk enter[c] = S, S = e^Acs_last S + C_c. The
// carried state ends as the state after the window. One thread per
// (b, h, p, n). Grid (B * H, ceil(P * N / blockDim.x)).
extern "C" __global__ void m3m_state_pass(
    float* __restrict__ enter,
    float* __restrict__ ssm,
    const float* __restrict__ contrib,
    const float* __restrict__ coef, const float* __restrict__ e0,
    const float* __restrict__ kst, const float* __restrict__ vst,
    const float* __restrict__ mimo_x,
    int batch, int seq, int nh, int hd, int ds, int rank
) {
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bh = blockIdx.x;
    const long long e = (long long)blockIdx.y * blockDim.x + threadIdx.x;
    const long long pn = (long long)hd * ds;
    if (bh >= (long long)batch * nh || e >= pn) return;
    const long long b = bh / nh;
    const int h = (int)(bh % nh);
    const int p = (int)(e / ds), n = (int)(e % ds);
    float adj = 0.0f;
    const float v = vst[bh * hd + p];
    for (int r = 0; r < rank; r++) {
        const float up = __fmul_rn(mimo_x[((long long)h * rank + r) * hd + p], v);
        adj = __fmaf_rn(up, kst[(bh * rank + r) * ds + n], adj);
    }
    float st = __fmaf_rn(e0[bh], adj, ssm[bh * pn + e]);
    for (int c = 0; c < nc; c++) {
        const int lc = min(M3M_CHUNK, seq - c * M3M_CHUNK);
        const long long bc = b * nc + c;
        const long long idx = (bc * nh + h) * pn + e;
        enter[idx] = st;
        const float acs_last = coef[((b * seq + c * M3M_CHUNK + lc - 1) * nh + h) * 4 + 3];
        st = __fmaf_rn(mimo_exp(acs_last), st, contrib[idx]);
    }
    ssm[bh * pn + e] = st;
}

// The outputs. One block per (chunk, head, token tile); thread (t, p) of
// the tile computes y_t[r] for every rank and the gated row:
//   y[r] = e^Acs_t (q_t[r] . S_enter[p])
//        + sum_{s<=t} (G[t,r,s,r'] w(t,s)) u_s[r'][p]        (s, then r')
//        + D u_t[r][p]
// with G[t,r,s,r'] = q_t[r] . k_s[r'] (n ascending) and w(t,s) =
// e^min(Acs_t - Acs_s, 0) scale_s off the diagonal, gamma_t on it. The
// queries of the tile stay staged; the keys come in s-tiles of `st` with
// their G block. Shared memory: tt * R * N (queries) + st * R * (N + 1)
// (keys, rows padded) + tt * R * st * R (G) + st * P (inputs) floats. The tile sizes change what
// a pass stages, never the order of a thread's sum.
// Grid (B * nc, H, ceil(64 / tt)), block tt * P threads. `y_out` (may be
// null) receives the readouts y_t[r][p] with D, [B, T, H, R, P], what the
// training backward's gate reads.
template <typename TI, int RK>
__device__ __forceinline__ void m3m_chunk_scan_body(
    TI* __restrict__ out, float* __restrict__ y_out,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ cn,
    const float* __restrict__ coef,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ enter,
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,
    const float* __restrict__ D,
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,
    const float* __restrict__ mimo_o,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tt, int st
) {
    extern __shared__ float m3m_sm[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int tile0 = blockIdx.z * tt;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    if (tile0 >= lc) return;
    const int tn = min(tt, lc - tile0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int R = MIMO_R(RK, rank);
    const int half = ds / 2;
    const int ST = st;
    // Key rows padded to N + 1: the block's G entries read one key row per
    // thread, which at a row stride of N (a multiple of 32) would put a
    // warp's reads in one bank.
    const int kld = ds + 1;
    float* qs = m3m_sm;                       // [tt][R][N]
    float* ks = qs + tt * rank * ds;          // [ST][R][N + 1]
    float* gs = ks + ST * rank * kld;         // [tt*R][ST*R]
    float* xs = gs + tt * rank * ST * rank;   // [ST][P]
    const int tid = threadIdx.x;
    const int tl = tid / hd, p = tid % hd;
    const bool live = tl < tn;
    // The tile's rotated queries.
    for (int i = tid; i < tn * rank * ds; i += blockDim.x) {
        const int tq = i / (rank * ds), r = (i / ds) % rank, n = i % ds;
        const long long row = b * seq + t0 + tile0 + tq;
        qs[i] = m3m_rot(cn + row * rgn + ((long long)r * ng + g) * ds,
                        c_bias + ((long long)h * rank + r) * ds, n, half, na,
                        sn + (row * nh + h) * na, cs + (row * nh + h) * na);
    }
    __syncthreads();
    float y[MIMO_RA(RK)];
    float u_t[MIMO_RA(RK)];
    float px[MIMO_RA(RK)];
    const int t = tile0 + tl;
    const long long row_t = b * seq + t0 + t;
    float acs_t = 0.0f, gamma_t = 0.0f;
    if (live) {
        const float* cf = coef + (row_t * nh + h) * 4;
        acs_t = cf[3];
        gamma_t = cf[1];
        const float xv = to_f(proj[row_t * ld_proj + di + (long long)h * hd + p]);
        const float* sp = enter + ((bc * nh + h) * hd + p) * (long long)ds;
        const float ef = mimo_exp(acs_t);
        _Pragma("unroll") for (int r = 0; r < R; r++) {
            px[r] = mimo_x[((long long)h * rank + r) * hd + p];
            u_t[r] = __fmul_rn(px[r], xv);
            y[r] = 0.0f;
        }
        // Every rank's chain over n ascending, the entering row read once.
        const float* q = qs + tl * rank * ds;
        for (int n = 0; n < ds; n++) {
            const float sv = sp[n];
            _Pragma("unroll") for (int r = 0; r < R; r++)
                y[r] = __fmaf_rn(q[r * ds + n], sv, y[r]);
        }
        _Pragma("unroll") for (int r = 0; r < R; r++) y[r] = __fmul_rn(y[r], ef);
    }
    const int s_end = tile0 + tn;
    for (int s0 = 0; s0 < s_end; s0 += ST) {
        const int sn_ = min(ST, s_end - s0);
        __syncthreads();
        for (int i = tid; i < sn_ * rank * ds; i += blockDim.x) {
            const int sk = i / (rank * ds), r = (i / ds) % rank, n = i % ds;
            const long long row = b * seq + t0 + s0 + sk;
            ks[(sk * rank + r) * kld + n] = m3m_rot(bn + row * rgn + ((long long)r * ng + g) * ds,
                            b_bias + ((long long)h * rank + r) * ds, n, half, na,
                            sn + (row * nh + h) * na, cs + (row * nh + h) * na);
        }
        for (int i = tid; i < sn_ * hd; i += blockDim.x) {
            const int sk = i / hd, pp = i % hd;
            xs[i] = to_f(proj[(b * seq + t0 + s0 + sk) * ld_proj + di + (long long)h * hd + pp]);
        }
        __syncthreads();
        for (int i = tid; i < tn * rank * sn_ * rank; i += blockDim.x) {
            const int col = i % (sn_ * rank);
            const int rw = i / (sn_ * rank);
            const float* q = qs + rw * ds;
            const float* k = ks + col * kld;
            float acc = 0.0f;
            for (int n = 0; n < ds; n++) acc = __fmaf_rn(q[n], k[n], acc);
            gs[rw * (ST * rank) + col] = acc;
        }
        __syncthreads();
        if (live) {
            for (int sk = 0; sk < sn_; sk++) {
                const int s = s0 + sk;
                if (s > t) break;
                const long long row_s = b * seq + t0 + s;
                const float* cf = coef + (row_s * nh + h) * 4;
                const float w = s < t
                    ? __fmul_rn(mimo_exp(fminf(__fsub_rn(acs_t, cf[3]), 0.0f)), cf[2])
                    : gamma_t;
                const float xsv = xs[sk * hd + p];
                _Pragma("unroll") for (int r = 0; r < R; r++) {
                    const float* grow = gs + (tl * rank + r) * (ST * rank) + sk * rank;
                    _Pragma("unroll") for (int r2 = 0; r2 < R; r2++) {
                        const float us = __fmul_rn(px[r2], xsv);
                        y[r] = __fmaf_rn(__fmul_rn(grow[r2], w), us, y[r]);
                    }
                }
            }
        }
    }
    if (live) {
        const float dval = D[h];
        _Pragma("unroll") for (int r = 0; r < R; r++) y[r] = __fmaf_rn(dval, u_t[r], y[r]);
        if (y_out != nullptr) {
            _Pragma("unroll") for (int r = 0; r < R; r++)
                y_out[((row_t * nh + h) * rank + r) * hd + p] = y[r];
        }
        float pz[MIMO_RA(RK)], po[MIMO_RA(RK)];
        _Pragma("unroll") for (int r = 0; r < R; r++) {
            pz[r] = mimo_z[((long long)h * rank + r) * hd + p];
            po[r] = mimo_o[((long long)h * rank + r) * hd + p];
        }
        const float z = to_f(proj[row_t * ld_proj + (long long)h * hd + p]);
        mimo_put(&out[row_t * di + (long long)h * hd + p], mimo_tok_gate<RK>(y, pz, po, z, rank));
    }
}

#define DEFINE_M3M_CHUNK_SCAN(NAME, TI)                                         \
extern "C" __global__ void NAME(                                                \
    TI* __restrict__ out, float* __restrict__ y_out,                            \
    const TI* __restrict__ proj, long long ld_proj,                             \
    const float* __restrict__ bn, const float* __restrict__ cn,                 \
    const float* __restrict__ coef,                                             \
    const float* __restrict__ sn, const float* __restrict__ cs,                 \
    const float* __restrict__ enter,                                            \
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,         \
    const float* __restrict__ D,                                                \
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,         \
    const float* __restrict__ mimo_o,                                           \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tt, int st   \
) {                                                                             \
    MIMO_RANK_DISPATCH(rank, m3m_chunk_scan_body<TI, RK>(out, y_out, proj,      \
        ld_proj, bn, cn, coef, sn, cs, enter, b_bias, c_bias, D, mimo_x,        \
        mimo_z, mimo_o, seq, nh, ng, hd, ds, rank, na, tt, st))                 \
}
DEFINE_M3M_CHUNK_SCAN(m3m_chunk_scan, float)
DEFINE_M3M_CHUNK_SCAN(m3m_chunk_scan_bf16, __nv_bfloat16)
DEFINE_M3M_CHUNK_SCAN(m3m_chunk_scan_f16, __half)

// The carried keys and input after the window: the last token's rotated
// biased keys and its raw input. Grid (B, H), threads striding R * N + P.
template <typename TI>
__device__ __forceinline__ void m3m_writeback_body(
    float* __restrict__ kst, float* __restrict__ vst,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ b_bias,
    int seq, int nh, int ng, int hd, int ds, int rank, int na
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const long long bh = (long long)b * nh + h;
    const long long row = (long long)b * seq + seq - 1;
    const float* sr = sn + (row * nh + h) * na;
    const float* cr = cs + (row * nh + h) * na;
    for (int i = threadIdx.x; i < rank * ds + hd; i += blockDim.x) {
        if (i < rank * ds) {
            const int r = i / ds, n = i % ds;
            kst[bh * rank * ds + i] =
                m3m_rot(bn + row * rgn + ((long long)r * ng + g) * ds,
                        b_bias + ((long long)h * rank + r) * ds, n, half, na, sr, cr);
        } else {
            const int p = i - rank * ds;
            vst[bh * hd + p] = to_f(proj[row * ld_proj + di + (long long)h * hd + p]);
        }
    }
}

#define DEFINE_M3M_WRITEBACK(NAME, TI)                                          \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ kst, float* __restrict__ vst,                           \
    const TI* __restrict__ proj, long long ld_proj,                             \
    const float* __restrict__ bn,                                               \
    const float* __restrict__ sn, const float* __restrict__ cs,                 \
    const float* __restrict__ b_bias,                                           \
    int seq, int nh, int ng, int hd, int ds, int rank, int na                   \
) {                                                                             \
    m3m_writeback_body<TI>(kst, vst, proj, ld_proj, bn, sn, cs, b_bias, seq,    \
        nh, ng, hd, ds, rank, na);                                              \
}
DEFINE_M3M_WRITEBACK(m3m_writeback, float)
DEFINE_M3M_WRITEBACK(m3m_writeback_bf16, __nv_bfloat16)
DEFINE_M3M_WRITEBACK(m3m_writeback_f16, __half)
