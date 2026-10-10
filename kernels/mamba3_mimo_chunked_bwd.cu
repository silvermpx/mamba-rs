// Mamba-3 MIMO training on the GPU, the chunked path: the backward of
// mamba3_mimo_chunked.cu.
//
// With w(t, s) = e^min(Acs_t - Acs_s, 0) scale_s off the diagonal and
// gamma_t on it, G[t,r,s,r'] = q_t[r] . k_s[r'] and M[t,r,s,r'] =
// dy_t[r] . u_s[r'] (over the channels), the forward's
//   y_t[r] = e^Acs_t (q_t[r] . S_enter) + sum_{s<=t} w(t,s) sum_r' G u_s[r'] + D u_t[r]
//   C_c    = sum_s c_s sum_r u_s[r] k_s[r],  c_s = e^min(Acs_last - Acs_s, 0) scale_s
// give, from dy:
//   dq_t[r] = e^Acs_t dy_t[r] . S_enter + sum_{s<=t, r'} w M k_s[r']
//   dk_s[r'] = c_s dS_c u_s[r'] + sum_{t>=s, r} w M q_t[r]
//   du_s[r'] = D dy_s[r'] + c_s dS_c k_s[r'] + sum_{t>=s, r} w G dy_t[r]
//   dw(t, s) = sum_{r, r'} G M,   dS_loc = sum_t e^Acs_t sum_r dy_t[r] q_t[r]
// then the state passing back over the chunks, the rotation back to the
// biased keys and queries with the angle gradients (cumulative over the
// window), and the decay gradients gathered without subtractions as the
// Mamba-2 backward gathers them:
//   da_k = sum_{t>=k, s<k} dw w + sum_{s<k} dc_s c_s + sum_{t>=k} I_t
//          + e^Acs_last (S_enter . dS_c)
// with I_t = sum_r dq_t[r](inter) . q_t[r] and dc_s = sum_r u_s[r] . (dS_c k_s[r]).
// The trapezoid's weights then take their share: scale_s = gamma_s +
// dt_{s+1} lam_c_{s+1}, gamma = lam dt, and the window's first-token lead
// e0 = dt_0 lam_c_0 from the gradient of the entering state.
//
// Every value is one thread's chain in a fixed order and every reduction a
// fixed tree or one block's fixed loop; the pair terms run in one block per
// chunk and head over (t, s) tiles, so no pair matrix is stored. dq, dk
// and du are f32 [B, T, H, R, N or P]. The gradients stop at the states a
// window starts from (the entering-state lead and Psi's share in it
// included, as the recurrence's first step has them).
//
// Composed after mamba3_mimo_chunked.cu and mamba3_mimo_bwd.cu.

// A block-wide fixed-tree sum; the block need not be a power of two.
__device__ __forceinline__ float m3mb_block_sum(float v, float* red) {
    return mimo_block_sum(v, red);
}

// The gate backward per (row, head, channel): dy_t[r] = dO Phi_r
// silu(Z_r z), du = D dy (the start of du), dz into the z column, and the
// per-row partials of Phi, Z (each [rows, H, R, P]) and of D ([rows, H, P]).
// Grid 1-D over rows * H * P.
template <typename T>
__device__ __forceinline__ void m3mb_gate_body(
    float* __restrict__ dy, float* __restrict__ du, T* __restrict__ d_proj, long long ld_proj,
    float* __restrict__ gphi, float* __restrict__ gz, float* __restrict__ gd,
    const T* __restrict__ d_o, const float* __restrict__ y, const T* __restrict__ proj,
    const float* __restrict__ D, const float* __restrict__ mimo_x,
    const float* __restrict__ mimo_z, const float* __restrict__ mimo_o,
    long long rows, int nh, int hd, int rank
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * nh * hd) return;
    const int p = (int)(i % hd);
    const int h = (int)((i / hd) % nh);
    const long long row = i / ((long long)nh * hd);
    const int di = nh * hd;
    const float z = to_f(proj[row * ld_proj + (long long)h * hd + p]);
    const float x = to_f(proj[row * ld_proj + di + (long long)h * hd + p]);
    const float d_out = to_f(d_o[row * di + (long long)h * hd + p]);
    const float dval = D[h];
    float dz = 0.0f, gdv = 0.0f;
    for (int r = 0; r < rank; r++) {
        const long long w = ((long long)h * rank + r) * hd + p;
        const long long o = ((row * nh + h) * rank + r) * hd + p;
        const float v = __fmul_rn(mimo_z[w], z);
        const float sg = mimo_sigmoid(v);
        const float silu = __fmul_rn(v, sg);
        const float yv = y[o];
        gphi[o] = __fmul_rn(d_out, __fmul_rn(yv, silu));
        const float dyv = __fmul_rn(__fmul_rn(d_out, mimo_o[w]), silu);
        dy[o] = dyv;
        du[o] = __fmul_rn(dval, dyv);
        const float dv = __fmul_rn(__fmul_rn(__fmul_rn(d_out, mimo_o[w]), yv), mimo_silu_grad(v));
        gz[o] = __fmul_rn(dv, z);
        dz = __fmaf_rn(dv, mimo_z[w], dz);
        gdv = __fmaf_rn(dyv, __fmul_rn(mimo_x[w], x), gdv);
    }
    mimo_put(&d_proj[row * ld_proj + (long long)h * hd + p], dz);
    gd[(row * nh + h) * hd + p] = gdv;
}

#define DEFINE_M3MB_GATE(NAME, T)                                               \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ dy, float* __restrict__ du, T* __restrict__ d_proj,     \
    long long ld_proj,                                                          \
    float* __restrict__ gphi, float* __restrict__ gz, float* __restrict__ gd,   \
    const T* __restrict__ d_o, const float* __restrict__ y,                     \
    const T* __restrict__ proj, const float* __restrict__ D,                    \
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,         \
    const float* __restrict__ mimo_o,                                           \
    long long rows, int nh, int hd, int rank                                    \
) {                                                                             \
    m3mb_gate_body<T>(dy, du, d_proj, ld_proj, gphi, gz, gd, d_o, y, proj, D,   \
        mimo_x, mimo_z, mimo_o, rows, nh, hd, rank);                            \
}
DEFINE_M3MB_GATE(m3mb_gate, float)
DEFINE_M3MB_GATE(m3mb_gate_bf16, __nv_bfloat16)
DEFINE_M3MB_GATE(m3mb_gate_f16, __half)

// The inter-chunk part of dq and its decay term, one block per (chunk,
// head, token): dq_t[r][n] = e^Acs_t sum_p dy_t[r][p] S_enter[p][n] (the
// start of dq), and I_t = sum_{r, n} dq_t[r][n] q_t[r][n] by a fixed tree.
// Grid (B * nc, H, 64), block 256.
extern "C" __global__ void m3mb_inter(
    float* __restrict__ dq, float* __restrict__ inter,
    const float* __restrict__ dy, const float* __restrict__ enter,
    const float* __restrict__ cn, const float* __restrict__ c_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    int seq, int nh, int ng, int hd, int ds, int rank, int na
) {
    __shared__ float red[256];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    if (t >= lc) return;
    const int g = h / (nh / ng);
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const long long row = b * seq + t0 + t;
    const float ef = mimo_exp(coef[(row * nh + h) * 4 + 3]);
    const float* sp = enter + (bc * nh + h) * (long long)hd * ds;
    const float* sr = sn + (row * nh + h) * na;
    const float* cr = cs + (row * nh + h) * na;
    float part = 0.0f;
    for (int i = threadIdx.x; i < rank * ds; i += blockDim.x) {
        const int r = i / ds, n = i % ds;
        float acc = 0.0f;
        for (int p = 0; p < hd; p++)
            acc = __fmaf_rn(dy[((row * nh + h) * rank + r) * hd + p], sp[(long long)p * ds + n], acc);
        const float d = __fmul_rn(acc, ef);
        dq[((row * nh + h) * rank + r) * ds + n] = d;
        const float q = m3m_rot(cn + row * rgn + ((long long)r * ng + g) * ds,
                                c_bias + ((long long)h * rank + r) * ds, n, half, na, sr, cr);
        part = __fmaf_rn(d, q, part);
    }
    const float s = m3mb_block_sum(part, red);
    if (threadIdx.x == 0) inter[row * nh + h] = s;
}

// The state gradient each chunk's outputs give its entering state:
// dS_loc[p][n] = sum_t e^Acs_t sum_r dy_t[r][p] q_t[r][n] (t, then r,
// ascending). Laid out as m3m_chunk_state: one block per (chunk, head,
// n-tile), the chunk's rotated queries for the tile staged.
// Grid (B * nc, H, ceil(N / nt)), block nt * P threads.
extern "C" __global__ void m3mb_dsloc(
    float* __restrict__ dsl,
    const float* __restrict__ dy,
    const float* __restrict__ cn, const float* __restrict__ c_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int nt
) {
    extern __shared__ float m3mb_sm[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int n0 = blockIdx.z * nt;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int g = h / (nh / ng);
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    float* qt = m3mb_sm;                    // [L][R][nt]
    float* et = qt + M3M_CHUNK * rank * nt; // [L]
    const int tid = threadIdx.x;
    for (int i = tid; i < lc * rank * nt; i += blockDim.x) {
        const int t = i / (rank * nt), r = (i / nt) % rank, nl = i % nt;
        const int n = n0 + nl;
        const long long row = b * seq + t0 + t;
        qt[i] = n < ds ? m3m_rot(cn + row * rgn + ((long long)r * ng + g) * ds,
                                 c_bias + ((long long)h * rank + r) * ds, n, half, na,
                                 sn + (row * nh + h) * na, cs + (row * nh + h) * na)
                       : 0.0f;
    }
    for (int t = tid; t < lc; t += blockDim.x)
        et[t] = mimo_exp(coef[((b * seq + t0 + t) * nh + h) * 4 + 3]);
    __syncthreads();
    const int p = tid % hd, nl = tid / hd;
    if (nl >= nt || n0 + nl >= ds) return;
    float acc = 0.0f;
    for (int t = 0; t < lc; t++) {
        const long long row = b * seq + t0 + t;
        for (int r = 0; r < rank; r++) {
            const float d = __fmul_rn(dy[((row * nh + h) * rank + r) * hd + p], et[t]);
            acc = __fmaf_rn(d, qt[(t * rank + r) * nt + nl], acc);
        }
    }
    dsl[((bc * nh + h) * hd + p) * (long long)ds + n0 + nl] = acc;
}

// The state passing backward, in place: dS holds dS_loc on entry and the
// gradient of each chunk's exit state on exit, from zero after the window;
// d_init receives the gradient of the state entering the window.
// Grid (B * H, ceil(P * N / blockDim.x)).
extern "C" __global__ void m3mb_pass(
    float* __restrict__ dst, float* __restrict__ d_init,
    const float* __restrict__ coef,
    int batch, int seq, int nh, int hd, int ds
) {
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bh = blockIdx.x;
    const long long e = (long long)blockIdx.y * blockDim.x + threadIdx.x;
    const long long pn = (long long)hd * ds;
    if (bh >= (long long)batch * nh || e >= pn) return;
    const long long b = bh / nh;
    const int h = (int)(bh % nh);
    float ev = 0.0f;
    for (int c = nc - 1; c >= 0; c--) {
        const int lc = min(M3M_CHUNK, seq - c * M3M_CHUNK);
        const long long idx = ((b * nc + c) * nh + h) * pn + e;
        const float tmp = dst[idx];
        dst[idx] = ev;
        const float acs_last = coef[((b * seq + c * M3M_CHUNK + lc - 1) * nh + h) * 4 + 3];
        ev = __fmaf_rn(mimo_exp(acs_last), ev, tmp);
    }
    d_init[bh * pn + e] = ev;
}

// dot[q] = S_enter . dS for q = (b * nc + c) * H + h, a fixed tree.
// Grid 1-D over B * nc * H, block 256.
extern "C" __global__ void m3mb_chunk_dot(
    float* __restrict__ dot, const float* __restrict__ enter, const float* __restrict__ dst,
    long long count, int pn
) {
    __shared__ float red[256];
    const long long q = blockIdx.x;
    if (q >= count) return;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < pn; i += 256)
        acc = __fmaf_rn(enter[q * pn + i], dst[q * pn + i], acc);
    const float s = m3mb_block_sum(acc, red);
    if (threadIdx.x == 0) dot[q] = s;
}

// The state terms of the chunk's tokens, one block per (chunk, head,
// group of `tg` tokens), each token's rotated keys and inputs staged:
//   dk_s[r][n] = c_s sum_p dS_c[p][n] u_s[r][p]   (the start of dk)
//   ste[r][p]  = sum_n dS_c[p][n] k_s[r][n];  du_s[r][p] += c_s ste
//   dc_s       = sum_{r, p} u_s[r][p] ste[r][p]   (a fixed tree)
// With `stage` the chunk's state gradient dS_c is read into shared memory
// once for the group, rows padded by one, else every token reads it from
// global memory. Grid (B * nc, H, ceil(64 / tg)), block 256 (the tree's
// width), dynamic shared memory R * N + R * P floats, plus P * (N + 1)
// with `stage`.
template <typename T>
__device__ __forceinline__ void m3mb_state_terms_body(
    float* __restrict__ dk, float* __restrict__ du, float* __restrict__ dcs,
    const float* __restrict__ dst, const T* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ b_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ mimo_x,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tg, int stage
) {
    extern __shared__ float m3mb_sm[];
    __shared__ float red[256];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int s0 = blockIdx.z * tg;
    if (s0 >= lc) return;
    const int s1 = min(s0 + tg, lc);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const float acs_last = coef[((b * seq + t0 + lc - 1) * nh + h) * 4 + 3];
    const float* dsc = dst + (bc * nh + h) * (long long)hd * ds;
    float* kt = m3mb_sm;                // [R][N]
    float* ut = kt + rank * ds;         // [R][P]
    float* dss = ut + rank * hd;        // [P][N + 1] with stage
    const int dld = stage ? ds + 1 : ds;
    const float* dsr = stage ? dss : dsc;
    if (stage) {
        for (int i = threadIdx.x; i < hd * ds; i += blockDim.x) {
            const int p = i / ds;
            dss[p * (ds + 1) + i - p * ds] = dsc[i];
        }
    }
    for (int s = s0; s < s1; s++) {
        const long long row = b * seq + t0 + s;
        const float* cf = coef + (row * nh + h) * 4;
        const float cfs = __fmul_rn(mimo_exp(fminf(__fsub_rn(acs_last, cf[3]), 0.0f)), cf[2]);
        __syncthreads();
        for (int i = threadIdx.x; i < rank * ds; i += blockDim.x) {
            const int r = i / ds, n = i % ds;
            kt[i] = m3m_rot(bn + row * rgn + ((long long)r * ng + g) * ds,
                            b_bias + ((long long)h * rank + r) * ds, n, half, na,
                            sn + (row * nh + h) * na, cs + (row * nh + h) * na);
        }
        for (int i = threadIdx.x; i < rank * hd; i += blockDim.x) {
            const int r = i / hd, p = i % hd;
            const float x = to_f(proj[row * ld_proj + di + (long long)h * hd + p]);
            ut[i] = __fmul_rn(mimo_x[((long long)h * rank + r) * hd + p], x);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < rank * ds; i += blockDim.x) {
            const int r = i / ds, n = i % ds;
            float acc = 0.0f;
            for (int p = 0; p < hd; p++)
                acc = __fmaf_rn(dsr[(long long)p * dld + n], ut[r * hd + p], acc);
            dk[((row * nh + h) * rank + r) * ds + n] = __fmul_rn(cfs, acc);
        }
        float part = 0.0f;
        for (int i = threadIdx.x; i < rank * hd; i += blockDim.x) {
            const int r = i / hd, p = i % hd;
            float ste = 0.0f;
            for (int n = 0; n < ds; n++)
                ste = __fmaf_rn(dsr[(long long)p * dld + n], kt[r * ds + n], ste);
            const long long o = ((row * nh + h) * rank + r) * hd + p;
            du[o] = __fmaf_rn(cfs, ste, du[o]);
            part = __fmaf_rn(ut[i], ste, part);
        }
        const float sum = m3mb_block_sum(part, red);
        if (threadIdx.x == 0) dcs[row * nh + h] = sum;
    }
}

#define DEFINE_M3MB_STATE_TERMS(NAME, T)                                        \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ dk, float* __restrict__ du, float* __restrict__ dcs,    \
    const float* __restrict__ dst, const T* __restrict__ proj,                  \
    long long ld_proj,                                                          \
    const float* __restrict__ bn, const float* __restrict__ b_bias,             \
    const float* __restrict__ coef, const float* __restrict__ sn,               \
    const float* __restrict__ cs, const float* __restrict__ mimo_x,             \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tg,          \
    int stage                                                                   \
) {                                                                             \
    m3mb_state_terms_body<T>(dk, du, dcs, dst, proj, ld_proj, bn, b_bias, coef, \
        sn, cs, mimo_x, seq, nh, ng, hd, ds, rank, na, tg, stage);              \
}
DEFINE_M3MB_STATE_TERMS(m3mb_state_terms, float)
DEFINE_M3MB_STATE_TERMS(m3mb_state_terms_bf16, __nv_bfloat16)
DEFINE_M3MB_STATE_TERMS(m3mb_state_terms_f16, __half)

// The pair terms of a chunk and head, one block walking (t-tile, s-tile)
// pairs, t-tiles ascending and for each the s-tiles up to it ascending.
// Per tile pair the rotated queries and keys, dy and u are staged; G and M
// come out of n- and p-dots, w(t, s) masks and weighs them, and
//   dw(t, s) = sum_{r, r'} G M      -> dw [B, nc, H, pairs], pair t(t+1)/2 + s
//   dq_t += sum_{s, r'} (w M) k_s    (the t-tile's dq in shared memory)
//   dk_s += sum_{t, r} (w M) q_t      (global, this block's own rows)
//   du_s += sum_{t, r} (w G) dy_t     (global, this block's own rows)
// Every output has one owner: the block, in this loop order; the products
// run as register tiles (m3m_tile_product). Dynamic shared memory:
// tl R N + tl R (N + 1) + tl R P + tl R (P + 1) + 3 (tl R)^2 + tl R N
// floats (queries, keys, dy, u, G, M, wG, the dq tile; key and u rows
// padded by one). Grid (B * nc, H), block 256.
template <typename T>
__device__ __forceinline__ void m3mb_pairs_body(
    float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ du,
    float* __restrict__ dwp,
    const float* __restrict__ dy, const T* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ cn,
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ mimo_x,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tl
) {
    extern __shared__ float m3mb_sm[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const int tr = tl * rank;
    // The key and u rows padded by one: a warp reads one key (u) row per
    // lane at a step of the G (M) product, which at a stride of N or P
    // (multiples of 32) would put the reads in one bank.
    const int kld = ds + 1, uld = hd + 1;
    float* qs = m3mb_sm;            // [tl][R][N]
    float* ks = qs + tr * ds;       // [tl][R][N + 1]
    float* dys = ks + tr * kld;     // [tl][R][P]
    float* us = dys + tr * hd;      // [tl][R][P + 1]
    float* gm = us + tr * uld;      // [tr][tr] G
    float* mm = gm + tr * tr;       // [tr][tr] M, then w M
    float* wg = mm + tr * tr;       // [tr][tr] w G
    float* dqs = wg + tr * tr;      // [tl][R][N]
    const int tid = threadIdx.x;
    const long long pairs = M3M_CHUNK * (M3M_CHUNK + 1) / 2;
    for (int T0 = 0; T0 < lc; T0 += tl) {
        const int tn = min(tl, lc - T0);
        __syncthreads();
#pragma unroll 4
        for (int i = tid; i < tn * rank * ds; i += blockDim.x) {
            const int tq = i / (rank * ds), r = (i / ds) % rank, n = i % ds;
            const long long row = b * seq + t0 + T0 + tq;
            qs[i] =
                m3m_rot(cn + row * rgn + ((long long)r * ng + g) * ds,
                        c_bias + ((long long)h * rank + r) * ds, n, half, na,
                        sn + (row * nh + h) * na, cs + (row * nh + h) * na);
            dqs[i] = dq[((row * nh + h) * rank + r) * ds + n];
        }
#pragma unroll 4
        for (int i = tid; i < tn * rank * hd; i += blockDim.x) {
            const int tq = i / (rank * hd), rp = i % (rank * hd);
            const long long row = b * seq + t0 + T0 + tq;
            dys[i] = dy[(row * nh + h) * rank * hd + rp];
        }
        for (int S0 = 0; S0 <= T0; S0 += tl) {
            const int sn_ = min(tl, lc - S0);
            __syncthreads();
#pragma unroll 4
            for (int i = tid; i < sn_ * rank * ds; i += blockDim.x) {
                const int sk = i / (rank * ds), r = (i / ds) % rank, n = i % ds;
                const long long row = b * seq + t0 + S0 + sk;
                ks[(sk * rank + r) * kld + n] =
                    m3m_rot(bn + row * rgn + ((long long)r * ng + g) * ds,
                            b_bias + ((long long)h * rank + r) * ds, n, half, na,
                            sn + (row * nh + h) * na, cs + (row * nh + h) * na);
            }
#pragma unroll 4
            for (int i = tid; i < sn_ * rank * hd; i += blockDim.x) {
                const int sk = i / (rank * hd), r = (i / hd) % rank, p = i % hd;
                const long long row = b * seq + t0 + S0 + sk;
                const float x = to_f(proj[row * ld_proj + di + (long long)h * hd + p]);
                us[(sk * rank + r) * uld + p] =
                    __fmul_rn(mimo_x[((long long)h * rank + r) * hd + p], x);
            }
            __syncthreads();
            // G and M per (t r, s r'), zero where s > t.
            const int ta = tn * rank, sb = sn_ * rank;
            auto live = [&](int a, int bcol) { return S0 + bcol / rank <= T0 + a / rank; };
            m3m_tile_product<1>(
                ta, sb, ds,
                [&](int a, int n) { return qs[a * ds + n]; },
                [&](int n, int bcol) { return ks[bcol * kld + n]; },
                [&](int, int) { return 0.0f; },
                [&](int a, int bcol, float v) { gm[a * tr + bcol] = live(a, bcol) ? v : 0.0f; });
            m3m_tile_product<1>(
                ta, sb, hd,
                [&](int a, int p) { return dys[a * hd + p]; },
                [&](int p, int bcol) { return us[bcol * uld + p]; },
                [&](int, int) { return 0.0f; },
                [&](int a, int bcol, float v) { mm[a * tr + bcol] = live(a, bcol) ? v : 0.0f; });
            __syncthreads();
            // dw per pair (a fixed loop over (r, r')), then w M and w G.
            for (int i = tid; i < tn * sn_; i += blockDim.x) {
                const int tq = i / sn_, sk = i % sn_;
                const int t = T0 + tq, s = S0 + sk;
                if (s > t) {
                    for (int r = 0; r < rank; r++)
                        for (int r2 = 0; r2 < rank; r2++) {
                            const int at = (tq * rank + r) * tr + sk * rank + r2;
                            wg[at] = 0.0f;
                            mm[at] = 0.0f;
                        }
                    continue;
                }
                const long long row_t = b * seq + t0 + t, row_s = b * seq + t0 + s;
                const float* ct = coef + (row_t * nh + h) * 4;
                const float* cs_ = coef + (row_s * nh + h) * 4;
                const float w = s < t
                    ? __fmul_rn(mimo_exp(fminf(__fsub_rn(ct[3], cs_[3]), 0.0f)), cs_[2])
                    : ct[1];
                float dw = 0.0f;
                for (int r = 0; r < rank; r++) {
                    for (int r2 = 0; r2 < rank; r2++) {
                        const int at = (tq * rank + r) * tr + sk * rank + r2;
                        dw = __fmaf_rn(gm[at], mm[at], dw);
                        wg[at] = __fmul_rn(w, gm[at]);
                        mm[at] = __fmul_rn(w, mm[at]);
                    }
                }
                dwp[(bc * nh + h) * pairs + (long long)t * (t + 1) / 2 + s] = dw;
            }
            __syncthreads();
            // dq for the t-tile (shared), dk and du for the s-tile (global).
            m3m_tile_product<4>(
                ta, ds, sb,
                [&](int a, int bcol) { return mm[a * tr + bcol]; },
                [&](int bcol, int n) { return ks[bcol * kld + n]; },
                [&](int a, int n) { return dqs[a * ds + n]; },
                [&](int a, int n, float v) { dqs[a * ds + n] = v; });
            auto krow = [&](int bcol) {
                const long long row = b * seq + t0 + S0 + bcol / rank;
                return (row * nh + h) * rank + bcol % rank;
            };
            m3m_tile_product<4>(
                sb, ds, ta,
                [&](int bcol, int a) { return mm[a * tr + bcol]; },
                [&](int a, int n) { return qs[a * ds + n]; },
                [&](int bcol, int n) { return dk[krow(bcol) * ds + n]; },
                [&](int bcol, int n, float v) { dk[krow(bcol) * ds + n] = v; });
            m3m_tile_product<2>(
                sb, hd, ta,
                [&](int bcol, int a) { return wg[a * tr + bcol]; },
                [&](int a, int p) { return dys[a * hd + p]; },
                [&](int bcol, int p) { return du[krow(bcol) * hd + p]; },
                [&](int bcol, int p, float v) { du[krow(bcol) * hd + p] = v; });
        }
        __syncthreads();
        for (int i = tid; i < tn * rank * ds; i += blockDim.x) {
            const int tq = i / (rank * ds), rn = i % (rank * ds);
            const long long row = b * seq + t0 + T0 + tq;
            dq[(row * nh + h) * rank * ds + rn] = dqs[i];
        }
    }
}

#define DEFINE_M3MB_PAIRS(NAME, T)                                              \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ du,     \
    float* __restrict__ dwp,                                                    \
    const float* __restrict__ dy, const T* __restrict__ proj,                   \
    long long ld_proj,                                                          \
    const float* __restrict__ bn, const float* __restrict__ cn,                 \
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,         \
    const float* __restrict__ coef, const float* __restrict__ sn,               \
    const float* __restrict__ cs, const float* __restrict__ mimo_x,             \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tl           \
) {                                                                             \
    m3mb_pairs_body<T>(dq, dk, du, dwp, dy, proj, ld_proj, bn, cn, b_bias,      \
        c_bias, coef, sn, cs, mimo_x, seq, nh, ng, hd, ds, rank, na, tl);       \
}
DEFINE_M3MB_PAIRS(m3mb_pairs, float)
DEFINE_M3MB_PAIRS(m3mb_pairs_bf16, __nv_bfloat16)
DEFINE_M3MB_PAIRS(m3mb_pairs_f16, __half)

// The rotation back, one block per (row, head): the rotated-space dq and
// dk turned back to the biased queries and keys (written per head as the
// B/C norm backward reads them, [rows, H, R, N]), and the angles'
// gradient per token, keys before queries, ranks ascending; each token's
// sum starts from zero and m3mb_angles accumulates the totals over the
// window. Block 128.
extern "C" __global__ void m3mb_rot_bwd(
    float* __restrict__ d_bn_h, float* __restrict__ d_cn_h, float* __restrict__ dth,
    const float* __restrict__ dq, const float* __restrict__ dk,
    const float* __restrict__ bn, const float* __restrict__ cn,
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,
    const float* __restrict__ sn, const float* __restrict__ cs,
    int nh, int ng, int ds, int rank, int na
) {
    const long long row = blockIdx.x;
    const int h = blockIdx.y;
    const int g = h / (nh / ng);
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const long long base = (row * nh + h) * rank * ds;
    const float* sr = sn + (row * nh + h) * na;
    const float* cr = cs + (row * nh + h) * na;
    for (int i = threadIdx.x; i < rank * ds; i += blockDim.x) {
        const int n = i % ds;
        const bool rotated = n < na || (n >= half && n - half < na);
        if (!rotated) {
            d_bn_h[base + i] = dk[base + i];
            d_cn_h[base + i] = dq[base + i];
        }
    }
    for (int a = threadIdx.x; a < na; a += blockDim.x) {
        const float s = sr[a], c = cr[a];
        float acc = 0.0f;
        for (int w = 0; w < 2; w++) {
            const float* gv = w == 0 ? dk : dq;
            float* out = w == 0 ? d_bn_h : d_cn_h;
            const float* src = w == 0 ? bn : cn;
            const float* bias = w == 0 ? b_bias : c_bias;
            for (int r = 0; r < rank; r++) {
                const float* srow = src + row * rgn + ((long long)r * ng + g) * ds;
                const float* brow = bias + ((long long)h * rank + r) * ds;
                const float v0 = __fadd_rn(srow[a], brow[a]);
                const float v1 = __fadd_rn(srow[a + half], brow[a + half]);
                const float g0 = gv[base + r * ds + a], g1 = gv[base + r * ds + a + half];
                const float t0v = __fmul_rn(g0, __fsub_rn(__fmul_rn(-s, v0), __fmul_rn(c, v1)));
                const float t1v = __fmul_rn(g1, __fsub_rn(__fmul_rn(c, v0), __fmul_rn(s, v1)));
                acc = __fadd_rn(acc, __fadd_rn(t0v, t1v));
                out[base + r * ds + a] = __fmaf_rn(c, g0, __fmul_rn(s, g1));
                out[base + r * ds + a + half] = __fmaf_rn(-s, g0, __fmul_rn(c, g1));
            }
        }
        dth[(row * nh + h) * na + a] = acc;
    }
}

// The angles over the window, walked back per (sequence, head): the
// gradient of every angle increment is the sum of the angle gradients from
// its token on; d tanh_pi = that sum dt into d_tanh_h (the angle
// projection's backward input), and its share of ddt (sum dt-side, a
// ascending) into ddt_ang [rows, H]. Grid (B, H), block 32: one lane per
// angle, lane 0 sums.
template <typename T>
__device__ __forceinline__ void m3mb_angles_body(
    float* __restrict__ d_tanh_h, float* __restrict__ ddt_ang,
    const float* __restrict__ dth, const float* __restrict__ coef,
    const T* __restrict__ proj, long long ld_proj, int ang_col,
    int seq, int nh, int na
) {
    extern __shared__ float m3mb_sm[];
    float* prod = m3mb_sm;  // [na]
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    float cum[8];
    const int per = (na + (int)blockDim.x - 1) / (int)blockDim.x;
    for (int k = 0; k < per && k < 8; k++) cum[k] = 0.0f;
    for (int t = seq - 1; t >= 0; t--) {
        const long long row = (long long)b * seq + t;
        const float dt = coef[(row * nh + h) * 4];
        const T* pr = proj + row * ld_proj;
        for (int k = 0; k < per && k < 8; k++) {
            const int a = threadIdx.x + k * blockDim.x;
            if (a < na) {
                cum[k] = __fadd_rn(cum[k], dth[(row * nh + h) * na + a]);
                d_tanh_h[(row * nh + h) * na + a] = __fmul_rn(cum[k], dt);
                prod[a] = __fmul_rn(cum[k], mimo_tanh_pi(pr, ang_col, a));
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.0f;
            for (int a = 0; a < na; a++) s = __fadd_rn(s, prod[a]);
            ddt_ang[row * nh + h] = s;
        }
        __syncthreads();
    }
}

#define DEFINE_M3MB_ANGLES(NAME, T)                                             \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ d_tanh_h, float* __restrict__ ddt_ang,                  \
    const float* __restrict__ dth, const float* __restrict__ coef,              \
    const T* __restrict__ proj, long long ld_proj, int ang_col,                 \
    int seq, int nh, int na                                                     \
) {                                                                             \
    m3mb_angles_body<T>(d_tanh_h, ddt_ang, dth, coef, proj, ld_proj, ang_col,   \
        seq, nh, na);                                                           \
}
DEFINE_M3MB_ANGLES(m3mb_angles, float)
DEFINE_M3MB_ANGLES(m3mb_angles_bf16, __nv_bfloat16)
DEFINE_M3MB_ANGLES(m3mb_angles_f16, __half)

// The decay and weight gradients of a chunk and head, one thread per token
// k (Mamba-2's gathering, with the pair values dw w, the state values
// dc c and the inter values I):
//   da_k     = stable_k + (next_k + prev_k)
//   dscale_k = sum_{t>k} dw(t, k) e^min(Acs_t - Acs_k, 0)
//              + dc_k e^min(Acs_last - Acs_k, 0)
//   dgamma_k = dw(k, k)
// written [rows, H, 3]. Grid (B * nc, H), block 64.
extern "C" __global__ void m3mb_combine(
    float* __restrict__ dsc3,
    const float* __restrict__ dwp, const float* __restrict__ dcs,
    const float* __restrict__ inter, const float* __restrict__ dot,
    const float* __restrict__ coef,
    int seq, int nh
) {
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int k = threadIdx.x;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    if (k >= lc) return;
    const long long pairs = M3M_CHUNK * (M3M_CHUNK + 1) / 2;
    const float* dw = dwp + (bc * nh + h) * pairs;
    auto acs = [&](int t) { return coef[((b * seq + t0 + t) * nh + h) * 4 + 3]; };
    auto scale = [&](int t) { return coef[((b * seq + t0 + t) * nh + h) * 4 + 2]; };
    auto wts = [&](int t, int s) {
        return __fmul_rn(mimo_exp(fminf(__fsub_rn(acs(t), acs(s)), 0.0f)), scale(s));
    };
    const float acs_last = acs(lc - 1);
    float stable = 0.0f;
    for (int t = k; t < lc; t++)
        for (int s = 0; s < k; s++) {
            const float wv = wts(t, s);
            stable = __fadd_rn(stable, __fmul_rn(dw[(long long)t * (t + 1) / 2 + s], wv));
        }
    float nxt = 0.0f;
    for (int s = 0; s < k; s++) {
        const float cv = __fmul_rn(mimo_exp(fminf(__fsub_rn(acs_last, acs(s)), 0.0f)), scale(s));
        nxt = __fadd_rn(nxt, __fmul_rn(dcs[(b * seq + t0 + s) * nh + h], cv));
    }
    float prv = __fadd_rn(inter[(b * seq + t0 + lc - 1) * nh + h],
                          __fmul_rn(dot[bc * nh + h], mimo_exp(acs_last)));
    for (int t = lc - 2; t >= k; t--) prv = __fadd_rn(prv, inter[(b * seq + t0 + t) * nh + h]);
    const float da = __fadd_rn(stable, __fadd_rn(nxt, prv));
    float dsc = __fmul_rn(dcs[(b * seq + t0 + k) * nh + h],
                          mimo_exp(fminf(__fsub_rn(acs_last, acs(k)), 0.0f)));
    for (int t = k + 1; t < lc; t++)
        dsc = __fmaf_rn(dw[(long long)t * (t + 1) / 2 + k],
                        mimo_exp(fminf(__fsub_rn(acs(t), acs(k)), 0.0f)), dsc);
    float* o = dsc3 + ((b * seq + t0 + k) * nh + h) * 3;
    o[0] = da;
    o[1] = dsc;
    o[2] = dw[(long long)k * (k + 1) / 2 + k];
}

// The entering state's lead term, one block per (sequence, head): with
// adj[p][n] = sum_r (Psi_r v[p]) kp_r[n] and d_init the gradient of the
// state entering the window, de0 = sum d_init adj (a fixed tree), and Psi
// takes e0 v[p] sum_n d_init[p][n] kp_r[n] per (r, p) into gpsi_init
// [B, H, R, P]. Block a whole number of warps covering P.
extern "C" __global__ void m3mb_lead(
    float* __restrict__ de0, float* __restrict__ gpsi_init,
    const float* __restrict__ d_init, const float* __restrict__ e0,
    const float* __restrict__ kst, const float* __restrict__ vst,
    const float* __restrict__ mimo_x,
    int nh, int hd, int ds, int rank
) {
    extern __shared__ float m3mb_sm[];
    const long long bh = (long long)blockIdx.x * nh + blockIdx.y;
    const int h = blockIdx.y;
    const int p = threadIdx.x;
    float part = 0.0f;
    if (p < hd) {
        const float v = vst[bh * hd + p];
        for (int r = 0; r < rank; r++) {
            float acc = 0.0f;
            for (int n = 0; n < ds; n++)
                acc = __fmaf_rn(d_init[(bh * hd + p) * ds + n], kst[(bh * rank + r) * ds + n], acc);
            const float psi = mimo_x[((long long)h * rank + r) * hd + p];
            gpsi_init[(bh * rank + r) * hd + p] = __fmul_rn(__fmul_rn(e0[bh], v), acc);
            part = __fmaf_rn(__fmul_rn(psi, v), acc, part);
        }
    }
    const float s = mimo_block_sum(part, m3mb_sm);
    if (threadIdx.x == 0) de0[bh] = s;
}

// The per-token scalars, one thread per (row, head): with scale_t =
// gamma_t + lead_{t+1} (lead = dt lam_c, the window's first one e0),
//   dgamma_t += dscale_t, dlead_{t+1} = dscale_t, dlead_0 = de0,
//   ddt = ddt_ang + da a + dgamma lam + dlead lam_c, dA = da dt,
//   dlam = dgamma dt, dlam_c = dlead dt, dtrap = dlam lam (1 - lam) - dlam_c lam_c (1 - lam_c),
// then the clamp mask and heavy tail on A, the softplus slope on dt, into
// the dd_dt, dd_a and trap columns, and the dt-column gradient into
// gdtb [rows, H]. Grid 1-D over rows * H.
template <typename T>
__device__ __forceinline__ void m3mb_scalars_body(
    T* __restrict__ d_proj, float* __restrict__ gdtb,
    const float* __restrict__ dsc3, const float* __restrict__ ddt_ang,
    const float* __restrict__ de0, const float* __restrict__ coef,
    const T* __restrict__ proj, long long ld_proj, const float* __restrict__ dt_bias,
    long long rows, int seq, int nh, int heads, float a_floor
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * nh) return;
    const int h = (int)(i % nh);
    const long long row = i / nh;
    const int t = (int)(row % seq);
    const long long b = row / seq;
    const T* pr = proj + row * ld_proj;
    const float* g3 = dsc3 + (row * nh + h) * 3;
    const float dt = coef[(row * nh + h) * 4];
    float dd_a = to_f(pr[heads + nh + h]);
    float ht = dd_a >= 0.0f ? __fadd_rn(1.0f, dd_a) : __fdiv_rn(1.0f, __fsub_rn(1.0f, dd_a));
    float a = -ht;
    const bool clamped = a > -a_floor;
    a = clamped ? -a_floor : a;
    const float trap = to_f(pr[heads + 2 * nh + h]);
    const float lam = mimo_sigmoid(trap);
    const float lam_c = mimo_sigmoid(-trap);
    // gamma_t weighs its own diagonal and, through scale_t, every later
    // token of the window.
    const float dgamma = __fadd_rn(g3[2], g3[1]);
    const float dlead = t == 0 ? de0[b * nh + h] : dsc3[((row - 1) * nh + h) * 3 + 1];
    float ddt = ddt_ang[row * nh + h];
    ddt = __fmaf_rn(g3[0], a, ddt);
    ddt = __fmaf_rn(dgamma, lam, ddt);
    ddt = __fmaf_rn(dlead, lam_c, ddt);
    const float d_a = __fmul_rn(g3[0], dt);
    const float dlam = __fmul_rn(dgamma, dt);
    const float dlam_c = __fmul_rn(dlead, dt);
    // lam (1 - lam) is lam * lam_c, with lam_c = sigmoid(-trap) keeping its
    // digits where one minus lam would round to zero.
    const float dtrap = __fmul_rn(__fsub_rn(dlam, dlam_c), __fmul_rn(lam, lam_c));
    float d_dd_a = 0.0f;
    if (!clamped) {
        const float slope = dd_a >= 0.0f ? 1.0f
            : __fdiv_rn(1.0f, __fmul_rn(__fsub_rn(1.0f, dd_a), __fsub_rn(1.0f, dd_a)));
        d_dd_a = -__fmul_rn(d_a, slope);
    }
    const float spin = __fadd_rn(to_f(pr[heads + h]), dt_bias[h]);
    const float d_sp = spin > 20.0f ? ddt : __fmul_rn(ddt, mimo_sigmoid(spin));
    mimo_put(&d_proj[row * ld_proj + heads + h], d_sp);
    mimo_put(&d_proj[row * ld_proj + heads + nh + h], d_dd_a);
    mimo_put(&d_proj[row * ld_proj + heads + 2 * nh + h], dtrap);
    gdtb[row * nh + h] = d_sp;
}

#define DEFINE_M3MB_SCALARS(NAME, T)                                            \
extern "C" __global__ void NAME(                                                \
    T* __restrict__ d_proj, float* __restrict__ gdtb,                           \
    const float* __restrict__ dsc3, const float* __restrict__ ddt_ang,          \
    const float* __restrict__ de0, const float* __restrict__ coef,              \
    const T* __restrict__ proj, long long ld_proj,                              \
    const float* __restrict__ dt_bias,                                          \
    long long rows, int seq, int nh, int heads, float a_floor                   \
) {                                                                             \
    m3mb_scalars_body<T>(d_proj, gdtb, dsc3, ddt_ang, de0, coef, proj, ld_proj, \
        dt_bias, rows, seq, nh, heads, a_floor);                                \
}
DEFINE_M3MB_SCALARS(m3mb_scalars, float)
DEFINE_M3MB_SCALARS(m3mb_scalars_bf16, __nv_bfloat16)
DEFINE_M3MB_SCALARS(m3mb_scalars_f16, __half)

// The rank inputs back to x and Psi, per (row, head, channel): dx = sum_r
// du[r] Psi_r into the x column, and du[r] x into gpsi [rows, H, R, P].
// Grid 1-D over rows * H * P.
template <typename T>
__device__ __forceinline__ void m3mb_u_bwd_body(
    T* __restrict__ d_proj, float* __restrict__ gpsi,
    const float* __restrict__ du, const T* __restrict__ proj, long long ld_proj,
    const float* __restrict__ mimo_x,
    long long rows, int nh, int hd, int rank
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * nh * hd) return;
    const int p = (int)(i % hd);
    const int h = (int)((i / hd) % nh);
    const long long row = i / ((long long)nh * hd);
    const int di = nh * hd;
    const float x = to_f(proj[row * ld_proj + di + (long long)h * hd + p]);
    float dx = 0.0f;
    for (int r = 0; r < rank; r++) {
        const long long o = ((row * nh + h) * rank + r) * hd + p;
        dx = __fmaf_rn(du[o], mimo_x[((long long)h * rank + r) * hd + p], dx);
        gpsi[o] = __fmul_rn(du[o], x);
    }
    mimo_put(&d_proj[row * ld_proj + di + (long long)h * hd + p], dx);
}

#define DEFINE_M3MB_U_BWD(NAME, T)                                              \
extern "C" __global__ void NAME(                                                \
    T* __restrict__ d_proj, float* __restrict__ gpsi,                           \
    const float* __restrict__ du, const T* __restrict__ proj,                   \
    long long ld_proj, const float* __restrict__ mimo_x,                        \
    long long rows, int nh, int hd, int rank                                    \
) {                                                                             \
    m3mb_u_bwd_body<T>(d_proj, gpsi, du, proj, ld_proj, mimo_x, rows, nh, hd,   \
        rank);                                                                  \
}
DEFINE_M3MB_U_BWD(m3mb_u_bwd, float)
DEFINE_M3MB_U_BWD(m3mb_u_bwd_bf16, __nv_bfloat16)
DEFINE_M3MB_U_BWD(m3mb_u_bwd_f16, __half)

// g[j] += sum over p of parts[j][p] (per head): D's per-channel partials
// summed over the channels, p ascending. Grid 1-D over n.
extern "C" __global__ void m3mb_sum_inner(
    float* __restrict__ g, const float* __restrict__ parts, int n, int inner
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float acc = 0.0f;
    for (int p = 0; p < inner; p++) acc = __fadd_rn(acc, parts[(long long)j * inner + p]);
    g[j] = __fadd_rn(g[j], acc);
}

// out[j] (+)= sum_r parts[r][j] over `rows` rows of `dim` columns: the same
// per-thread strided sums and halving tree as the shared reduce_sum_axis0,
// so the same bits, with 64-bit indices for windows whose rows * dim pass
// 2^31. Grid (dim), block a power of two up to 256, its floats of shared
// memory.
extern "C" __global__ void m3mb_reduce_rows(
    float* __restrict__ out, const float* __restrict__ parts,
    long long rows, long long dim, int accumulate
) {
    extern __shared__ float m3mb_red[];
    const long long d = blockIdx.x;
    if (d >= dim) return;
    const int tid = threadIdx.x;
    float sum = 0.0f;
    for (long long r = tid; r < rows; r += blockDim.x) sum += parts[r * dim + d];
    m3mb_red[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) m3mb_red[tid] += m3mb_red[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[d] = accumulate ? out[d] + m3mb_red[0] : m3mb_red[0];
}
