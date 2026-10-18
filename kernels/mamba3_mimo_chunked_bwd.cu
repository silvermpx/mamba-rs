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

// The gate backward with the per-rank output norm, one block per (row,
// head): rstd as mimo_gate_normed computes it, then per (r, p), with
// yn = y rstd, a = yn w and s = silu(Z z):
//   gphi = dO a s,  d_a = dO Phi s,  gw += d_a yn,  dyn = d_a w,
//   dv = dO Phi a silu'(Z z): gz = dv z and dz += dv Z (r ascending),
// then per rank c = sum_p dyn yn / P (p ascending) and the readouts'
// gradient dy = rstd (dyn - yn c), du = D dy and the D partial
// sum_r dy u, as m3mb_gate's outputs. The norm weight's partials go to gw
// [rows, H, P]. Grid (rows, H), block a whole number of warps.
template <typename T>
__device__ __forceinline__ void m3mb_gate_normed_body(
    float* __restrict__ dy, float* __restrict__ du, T* __restrict__ d_proj, long long ld_proj,
    float* __restrict__ gphi, float* __restrict__ gz, float* __restrict__ gd,
    float* __restrict__ gw,
    const T* __restrict__ d_o, const float* __restrict__ y, const T* __restrict__ proj,
    const float* __restrict__ D, const float* __restrict__ norm_w,
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,
    const float* __restrict__ mimo_o,
    int nh, int hd, int rank, float eps,
    int y_first /* this run's first y block */
) {
    __shared__ float rstd[MIMO_MAX_RANK];
    __shared__ float cfac[MIMO_MAX_RANK];
    const long long row = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const long long di = (long long)nh * hd;
    const long long base = (row * nh + h) * (long long)rank * hd;
    const float* yr = y + base;
    for (int r = threadIdx.x; r < rank; r += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p < hd; p++) acc = __fmaf_rn(yr[r * hd + p], yr[r * hd + p], acc);
        rstd[r] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(acc, (float)hd), eps)));
    }
    __syncthreads();
    for (int p = threadIdx.x; p < hd; p += blockDim.x) {
        const float z = to_f(proj[row * ld_proj + (long long)h * hd + p]);
        const float d_out = to_f(d_o[row * di + (long long)h * hd + p]);
        const float w = norm_w[(long long)h * hd + p];
        float dz = 0.0f, gwv = 0.0f;
        for (int r = 0; r < rank; r++) {
            const long long q = ((long long)h * rank + r) * hd + p;
            const long long o = base + (long long)r * hd + p;
            const float v = __fmul_rn(mimo_z[q], z);
            const float sg = mimo_sigmoid(v);
            const float silu = __fmul_rn(v, sg);
            const float yn = __fmul_rn(yr[r * hd + p], rstd[r]);
            const float a = __fmul_rn(yn, w);
            gphi[o] = __fmul_rn(d_out, __fmul_rn(a, silu));
            const float d_a = __fmul_rn(__fmul_rn(d_out, mimo_o[q]), silu);
            gwv = __fmaf_rn(d_a, yn, gwv);
            dy[o] = __fmul_rn(d_a, w);
            const float dv = __fmul_rn(__fmul_rn(__fmul_rn(d_out, mimo_o[q]), a), mimo_silu_grad(v));
            gz[o] = __fmul_rn(dv, z);
            dz = __fmaf_rn(dv, mimo_z[q], dz);
        }
        mimo_put(&d_proj[row * ld_proj + (long long)h * hd + p], dz);
        gw[(row * nh + h) * hd + p] = gwv;
    }
    __syncthreads();
    for (int r = threadIdx.x; r < rank; r += blockDim.x) {
        float dot = 0.0f;
        for (int p = 0; p < hd; p++)
            dot = __fadd_rn(dot, __fmul_rn(dy[base + (long long)r * hd + p],
                                           __fmul_rn(yr[r * hd + p], rstd[r])));
        cfac[r] = __fdiv_rn(dot, (float)hd);
    }
    __syncthreads();
    const float dval = D[h];
    for (int p = threadIdx.x; p < hd; p += blockDim.x) {
        const float x = to_f(proj[row * ld_proj + di + (long long)h * hd + p]);
        float gdv = 0.0f;
        for (int r = 0; r < rank; r++) {
            const long long q = ((long long)h * rank + r) * hd + p;
            const long long o = base + (long long)r * hd + p;
            const float yn = __fmul_rn(yr[r * hd + p], rstd[r]);
            const float dyv = __fmul_rn(rstd[r], __fsub_rn(dy[o], __fmul_rn(yn, cfac[r])));
            dy[o] = dyv;
            du[o] = __fmul_rn(dval, dyv);
            gdv = __fmaf_rn(dyv, __fmul_rn(mimo_x[q], x), gdv);
        }
        gd[(row * nh + h) * hd + p] = gdv;
    }
}

#define DEFINE_M3MB_GATE_NORMED(NAME, T)                                       \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dy, float* __restrict__ du, T* __restrict__ d_proj,    \
    long long ld_proj,                                                         \
    float* __restrict__ gphi, float* __restrict__ gz, float* __restrict__ gd,  \
    float* __restrict__ gw,                                                    \
    const T* __restrict__ d_o, const float* __restrict__ y,                    \
    const T* __restrict__ proj, const float* __restrict__ D,                   \
    const float* __restrict__ norm_w,                                          \
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,        \
    const float* __restrict__ mimo_o,                                          \
    int nh, int hd, int rank, float eps,                                       \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    m3mb_gate_normed_body<T>(dy, du, d_proj, ld_proj, gphi, gz, gd, gw, d_o, y, \
        proj, D, norm_w, mimo_x, mimo_z, mimo_o, nh, hd, rank, eps, y_first);  \
}
DEFINE_M3MB_GATE_NORMED(m3mb_gate_normed, float)
DEFINE_M3MB_GATE_NORMED(m3mb_gate_normed_bf16, __nv_bfloat16)
DEFINE_M3MB_GATE_NORMED(m3mb_gate_normed_f16, __half)

// The inter-chunk part of dq and its decay term for a tile of `tt` tokens:
// dq_t[r][n] = e^Acs_t sum_p dy_t[r][p] S_enter[p][n] (the start of dq),
// each element's chain over the channels p ascending from 0, then e^Acs_t;
// and I_t = sum_{r, n} dq_t[r][n] q_t[r][n] as a block of 256 threads sums
// it: virtual thread j takes fma(d, q) over i = j, j + 256, ... (i = r N +
// n), then the halving tree over the 256 partials, every token's tree in
// parallel. Both are zero for a token that does not continue the packed
// sequence the entering state holds. The dq rows (t, r) come out of
// register tiles over a pass of up to 128 state indices (a warp takes 4
// rows, a lane the indices lane + 32 j), with the entering state staged a
// slice of `ps` channels at a time beside dy's slice, the chains running on
// across the slices, so a state of any size runs here. Shared memory: the
// state slice [ps][W] (W the pass's width, at most 128) padded to a float4
// multiple, so the float4 rows of dy's slice [ps][R4 + 4] (transposed, R4 =
// tt R rounded up to 4) start on 16 bytes, then d [tt R][N]; the trees'
// [tt][256] partials reuse the slices' room when it holds them, else follow
// d. Grid (B * nc, H, ceil(64 / tt)), block 8 R4 threads.
__device__ __forceinline__ void m3mb_inter_tiles_body(
    float* __restrict__ dq, float* __restrict__ inter,
    const float* __restrict__ dy, const float* __restrict__ enter,
    const float* __restrict__ cn, const float* __restrict__ c_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tt, int ps,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float4 m3mb_sm4[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int T0 = blockIdx.z * tt;
    if (T0 >= lc) return;
    const int tn = min(tt, lc - T0);
    const int g = h / (nh / ng);
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const int rows = tn * rank;
    const int rs = ((tt * rank + 3) & ~3) + 4;
    const int sts = (ps * min(ds, 128) + 3) & ~3;
    float* st = reinterpret_cast<float*>(m3mb_sm4);  // [ps][W]
    float* dyt = st + sts;                            // [ps][rs]
    float* dd = dyt + ps * rs;                        // [tt R][N]
    float* red = tt * 256 <= sts + ps * rs ? st : dd + tt * rank * ds;  // [tt][256]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5, nw = blockDim.x >> 5;
    const int a0 = 4 * warp;
    const long long row0 = b * seq + t0 + T0;
    const float* sp = enter + (bc * nh + h) * (long long)hd * ds;
    // The tokens from `ce` on start or follow a packed sequence the
    // entering state does not hold: their dq starts at zero and I_t is zero.
    const int ce = vl_cont_end(seq_idx, b, seq, t0, lc);
    for (int nb = 0; nb < ds; nb += 128) {
        const int w = min(128, ds - nb);
        float acc[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
        }
        for (int p0 = 0; p0 < hd; p0 += ps) {
            const int pn = min(ps, hd - p0);
            __syncthreads();
            for (int q = warp; q < pn; q += nw) {
                const float* sr_ = sp + (long long)(p0 + q) * ds + nb;
                for (int n = lane; n < w; n += 32) st[q * w + n] = sr_[n];
            }
            for (int a = warp; a < rs; a += nw) {
                const float* dr =
                    dy + ((row0 + a / rank) * nh + h) * rank * hd + (a % rank) * hd + p0;
                for (int q = lane; q < pn; q += 32) dyt[q * rs + a] = a < rows ? dr[q] : 0.0f;
            }
            __syncthreads();
            for (int q = 0; q < pn; q++) {
                const float4 d4 = *reinterpret_cast<const float4*>(dyt + q * rs + a0);
                const float dv[4] = {d4.x, d4.y, d4.z, d4.w};
                float sv[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int n = lane + 32 * j;
                    sv[j] = n < w ? st[q * w + n] : 0.0f;
                }
#pragma unroll
                for (int i = 0; i < 4; i++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(dv[i], sv[j], acc[i][j]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int a = a0 + i;
            if (a >= rows) continue;
            const long long row = row0 + a / rank;
            const float ef = mimo_exp(coef[(row * nh + h) * 4 + 3]);
            const bool cont = T0 + a / rank < ce;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = nb + lane + 32 * j;
                if (n >= ds) continue;
                const float d = cont ? __fmul_rn(acc[i][j], ef) : 0.0f;
                dq[((row * nh + h) * rank + a % rank) * ds + n] = d;
                dd[a * ds + n] = d;
            }
        }
    }
    __syncthreads();
    const int rn = rank * ds;
    for (int e = tid; e < tn * 256; e += blockDim.x) {
        const int tl = e >> 8, j = e & 255;
        const long long row = row0 + tl;
        const float* sr_ = sn + (row * nh + h) * na;
        const float* cr_ = cs + (row * nh + h) * na;
        // Four terms' loads in flight before the chain takes them in order.
        float part = 0.0f;
        for (int i0 = j; i0 < rn; i0 += 4 * 256) {
            M3mRotIn in[4];
            float dv[4];
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int i = i0 + 256 * u;
                const int r = i / ds, n = i - r * ds;
                in[u] = M3mRotIn{0.0f, 0.0f, 0.0f, 0.0f};
                dv[u] = 0.0f;
                if (i < rn) {
                    in[u] = m3m_rot_load(cn + row * rgn + ((long long)r * ng + g) * ds,
                                         c_bias + ((long long)h * rank + r) * ds, n, half, na,
                                         sr_, cr_);
                    dv[u] = dd[(tl * rank + r) * ds + n];
                }
            }
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int i = i0 + 256 * u;
                const int n = i % ds;
                if (i < rn) part = __fmaf_rn(dv[u], m3m_rot_turn(in[u], n, half, na), part);
            }
        }
        red[e] = part;
    }
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        for (int e = tid; e < tn * w; e += blockDim.x) {
            const int tl = e / w, j = e - tl * w;
            red[tl * 256 + j] = __fadd_rn(red[tl * 256 + j], red[tl * 256 + j + w]);
        }
        __syncthreads();
    }
    for (int tl = tid; tl < tn; tl += blockDim.x)
        inter[(row0 + tl) * nh + h] = T0 + tl < ce ? red[tl * 256] : 0.0f;
}

extern "C" __global__ void __launch_bounds__(512) m3mb_inter_tiles(
    float* __restrict__ dq, float* __restrict__ inter,
    const float* __restrict__ dy, const float* __restrict__ enter,
    const float* __restrict__ cn, const float* __restrict__ c_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tt, int ps,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    m3mb_inter_tiles_body(dq, inter, dy, enter, cn, c_bias, coef, sn, cs, seq, nh, ng, hd, ds,
                          rank, na, tt, ps, y_first, seq_idx);
}

// The state gradient each chunk's outputs give its entering state:
// dS_loc[p][n] = sum_t e^Acs_t sum_r dy_t[r][p] q_t[r][n] (t, then r,
// ascending), t over the tokens that continue the packed sequence the
// entering state holds (all of them without a packing). Laid out as
// m3m_chunk_state: one block per (chunk, head, n-tile, tile of `pt`
// channels), the chunk's rotated queries for the tile staged. Grid
// (B * nc, H, ceil(N / nt) * ceil(P / pt)), the channel tile fastest, block
// nt * pt threads.
extern "C" __global__ void m3mb_dsloc(
    float* __restrict__ dsl,
    const float* __restrict__ dy,
    const float* __restrict__ cn, const float* __restrict__ c_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int nt, int pt,
    int y_first, /* this run's first y block */
    int z_first, /* this run's first z block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m3mb_sm[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int nct = (hd + pt - 1) / pt;
    const int p0 = (int)((z_first + blockIdx.z) % nct) * pt;
    const int pc = min(pt, hd - p0);
    const int n0 = (int)((z_first + blockIdx.z) / nct) * nt;
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
    const int pl = tid % pt, nl = tid / pt;
    const int p = p0 + pl;
    if (nl >= nt || n0 + nl >= ds || pl >= pc) return;
    const int ce = vl_cont_end(seq_idx, b, seq, t0, lc);
    float acc = 0.0f;
    for (int t = 0; t < ce; t++) {
        const long long row = b * seq + t0 + t;
        for (int r = 0; r < rank; r++) {
            const float d = __fmul_rn(dy[((row * nh + h) * rank + r) * hd + p], et[t]);
            acc = __fmaf_rn(d, qt[(t * rank + r) * nt + nl], acc);
        }
    }
    dsl[((bc * nh + h) * hd + p) * (long long)ds + n0 + nl] = acc;
}

// m3mb_dsloc as register tiles, the shape of m3m_chunk_state_tiles: a
// warp takes 4 channels p and a lane the state indices lane + 32 j of the
// block's tile, and a block a group of `pg` channels (the whole head, or
// groups sized to the kernel's registers); d = dy e^Acs_t for its channels
// and the rotated queries are computed once per block and staged `sl`
// tokens at a time, dt [sl R][pg] and qt [sl R][32 JC], with e^Acs [sl].
// Each output keeps the chain above: acc = 0, then fma(d, q, acc) over t
// ascending and r ascending, whatever the group.
// Grid (B * nc, H, ceil(N / (32 JC)) ceil(P / pg)), the state tiles
// innermost, block 8 pg threads.
template <int JC>
__device__ __forceinline__ void m3mb_dsloc_tiles_body(
    float* __restrict__ dsl,
    const float* __restrict__ dy,
    const float* __restrict__ cn, const float* __restrict__ c_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int sl, int pg,
    int y_first, /* this run's first y block */
    int z_first, /* this run's first z block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float4 m3mb_sm4[];
    constexpr int NT = 32 * JC;
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int nz = (ds + NT - 1) / NT;
    const int grp = (z_first + blockIdx.z) / nz;
    const int n0 = ((z_first + blockIdx.z) - grp * nz) * NT;
    const int pb = grp * pg;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int g = h / (nh / ng);
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    float* dt = reinterpret_cast<float*>(m3mb_sm4);  // [sl R][pg]
    float* qt = dt + sl * rank * pg;                 // [sl R][NT]
    float* et = qt + sl * rank * NT;                 // [sl]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5, nw = blockDim.x >> 5;
    const int p0 = 4 * warp;
    float acc[4][JC];
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < JC; j++) acc[i][j] = 0.0f;
    }
    const int ce = vl_cont_end(seq_idx, b, seq, t0, lc);
    for (int s0 = 0; s0 < ce; s0 += sl) {
        const int sn_ = min(sl, ce - s0);
        const long long row0 = b * seq + t0 + s0;
        if (s0 > 0) __syncthreads();
        for (int t = tid; t < sn_; t += blockDim.x)
            et[t] = mimo_exp(coef[((row0 + t) * nh + h) * 4 + 3]);
        for (int tr = warp; tr < sn_ * rank; tr += nw) {
            const int r = tr % rank;
            const long long row = row0 + tr / rank;
            const float* src = cn + row * rgn + ((long long)r * ng + g) * ds;
            const float* bias = c_bias + ((long long)h * rank + r) * ds;
            const float* sr_ = sn + (row * nh + h) * na;
            const float* cr_ = cs + (row * nh + h) * na;
#pragma unroll 4
            for (int ni = lane; ni < NT; ni += 32) {
                const int n = n0 + ni;
                qt[tr * NT + ni] = n < ds ? m3m_rot(src, bias, n, half, na, sr_, cr_) : 0.0f;
            }
        }
        __syncthreads();
        for (int tr = warp; tr < sn_ * rank; tr += nw) {
            const int r = tr % rank;
            const long long row = row0 + tr / rank;
            const float* dyr = dy + ((row * nh + h) * rank + r) * hd + pb;
            const float e = et[tr / rank];
            const int pn = hd - pb;
            for (int pp = lane; pp < pg; pp += 32)
                dt[tr * pg + pp] = pp < pn ? __fmul_rn(dyr[pp], e) : 0.0f;
        }
        __syncthreads();
        const int kd = sn_ * rank;
        for (int k = 0; k < kd; k++) {
            const float4 l4 = *reinterpret_cast<const float4*>(dt + k * pg + p0);
            const float lv[4] = {l4.x, l4.y, l4.z, l4.w};
            float rv[JC];
#pragma unroll
            for (int j = 0; j < JC; j++) rv[j] = qt[k * NT + lane + 32 * j];
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < JC; j++) acc[i][j] = __fmaf_rn(lv[i], rv[j], acc[i][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int p = pb + p0 + i;
        if (p >= hd) continue;
#pragma unroll
        for (int j = 0; j < JC; j++) {
            const int n = n0 + lane + 32 * j;
            if (n < ds) dsl[((bc * nh + h) * hd + p) * (long long)ds + n] = acc[i][j];
        }
    }
}

extern "C" __global__ void m3mb_dsloc_tiles(
    float* __restrict__ dsl,
    const float* __restrict__ dy,
    const float* __restrict__ cn, const float* __restrict__ c_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int jc, int sl, int pg,
    int y_first, /* this run's first y block */
    int z_first, /* this run's first z block */
    const int* __restrict__ seq_idx
) {
    if (jc == 1)
        m3mb_dsloc_tiles_body<1>(dsl, dy, cn, c_bias, coef, sn, cs, seq, nh, ng, hd, ds,
                                 rank, na, sl, pg, y_first, z_first, seq_idx);
    else if (jc == 2)
        m3mb_dsloc_tiles_body<2>(dsl, dy, cn, c_bias, coef, sn, cs, seq, nh, ng, hd, ds,
                                 rank, na, sl, pg, y_first, z_first, seq_idx);
    else
        m3mb_dsloc_tiles_body<4>(dsl, dy, cn, c_bias, coef, sn, cs, seq, nh, ng, hd, ds,
                                 rank, na, sl, pg, y_first, z_first, seq_idx);
}

// The state passing backward, in place: dS holds dS_loc on entry and the
// gradient of each chunk's exit state on exit, from zero after the window;
// d_init receives the gradient of the state entering the window. A chunk
// where a packed sequence starts passes nothing back into the chunk
// before it. Grid (B * H, ceil(P * N / blockDim.x)).
extern "C" __global__ void m3mb_pass(
    float* __restrict__ dst, float* __restrict__ d_init,
    const float* __restrict__ coef,
    int batch, int seq, int nh, int hd, int ds,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bh = blockIdx.x;
    const long long e = (long long)(y_first + blockIdx.y) * blockDim.x + threadIdx.x;
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
        ev = vl_passes(seq_idx, b, seq, c * M3M_CHUNK, lc)
            ? __fmaf_rn(mimo_exp(acs_last), ev, tmp) : tmp;
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

// The state terms of a tile of `tt` tokens, rows (s, r):
//   dk[row][n]  = c_s sum_p u[row][p] dS[p][n]   (p ascending from 0)
//   ste[row][p] = sum_n k[row][n] dS[p][n]       (n ascending from 0)
//   du[row][p]  = fma(c_s, ste, du);  dc_s by a fixed 256-wide tree:
//   virtual thread j sums fma(u, ste) over i = j, j + 256, ... (i = r P +
//   p), then the halving tree, every token's tree in parallel
// for the tokens of the chunk's last packed sequence (every token without a
// packing); an earlier sequence's token reaches no later chunk, so its dk
// starts at zero, its du keeps its value and its dc_s is zero. These are the
// chains of the per-token kernel as it was released (frozen in the tests).
// The products are register tiles (a warp takes 4 rows, a lane the columns
// lane + 32 j), 128 output columns a pass. With `whole` the chunk's dS
// stages once as [P][N + 1], which both products read without bank
// conflicts, beside u and the rotated keys transposed ([P][R4 + 4],
// [N][R4 + 4], their float4 rows on 16 bytes). Otherwise each pass stages
// its operands a slice at a time, the chains running on across the slices:
// dk slices of `ps` channels of u and of dS's 128 columns ([ps][R4 + 4],
// [ps][min(N, 128)]), ste slices of `ns` state indices of the keys and of
// dS's 128 rows ([ns][R4 + 4], [min(P, 128)][LDB], LDB = ns rounded up to
// odd), and the tree recomputes u. ste [tt R][P] follows in shared memory,
// or with `steb_glob` in that block's slice of a global scratch, and the
// trees' [tt][256] last; in the sliced layout they take the slices' room
// when it holds them. Grid (B * nc, H, ceil(64 / tt)), block 8 R4 threads.
template <typename T>
__device__ __forceinline__ void m3mb_state_terms_tiles_body(
    float* __restrict__ dk, float* __restrict__ du, float* __restrict__ dcs,
    const float* __restrict__ dst, const T* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ b_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ mimo_x, float* __restrict__ steb_glob,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tt, int whole, int ps,
    int ns,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float4 m3mb_sm4[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    // Only the chunk's last packed sequence reaches the next chunk.
    const int sf = vl_seq_first(seq_idx, b, seq, t0, t0 + lc - 1) - t0;
    const int S0 = blockIdx.z * tt;
    if (S0 >= lc) return;
    const int tn = min(tt, lc - S0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const int rows = tn * rank;
    const int rs = ((tt * rank + 3) & ~3) + 4;
    const int dld = ds + 1;
    const int aw = min(ds, 128), pw = min(hd, 128);
    const int ldb = (ns & ~1) + 1;
    float* sm = reinterpret_cast<float*>(m3mb_sm4);
    // The staged operands' room: dS, u and the keys whole, or the larger of
    // the two passes' slices.
    const int room = whole ? ((hd * dld + 3) & ~3) + hd * rs + ds * rs
                           : (max(ps * rs + ps * aw, ns * rs + pw * ldb) + 3) & ~3;
    const long long blk = ((long long)bc * nh + h) * gridDim.z + blockIdx.z;
    float* steb = steb_glob != nullptr ? steb_glob + blk * ((long long)tt * rank * hd)
                                       : sm + room;  // [tt R][P]
    float* red = !whole && tt * 256 <= room
        ? sm
        : sm + room + (steb_glob != nullptr ? 0 : tt * rank * hd);  // [tt][256]
    float* ut = sm + ((hd * dld + 3) & ~3);  // [P][rs], whole only
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5, nw = blockDim.x >> 5;
    const int a0 = 4 * warp;
    const long long row0 = b * seq + t0 + S0;
    const float acs_last = coef[((b * seq + t0 + lc - 1) * nh + h) * 4 + 3];
    const float* dsc = dst + (bc * nh + h) * (long long)hd * ds;
    auto key_row = [&](int a) {
        const int r = a % rank;
        const long long row = row0 + a / rank;
        return M3mRotRow{bn + row * rgn + ((long long)r * ng + g) * ds,
                         b_bias + ((long long)h * rank + r) * ds,
                         sn + (row * nh + h) * na, cs + (row * nh + h) * na};
    };
    float cfs[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int a = a0 + i;
        cfs[i] = 0.0f;
        if (a < rows) {
            const float* cf = coef + ((row0 + a / rank) * nh + h) * 4;
            cfs[i] = __fmul_rn(mimo_exp(fminf(__fsub_rn(acs_last, cf[3]), 0.0f)), cf[2]);
        }
    }
    // dk's 128 columns from nb: c_s times the chain over p ascending.
    auto store_dk = [&](int nb, float (&acc)[4][4]) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int a = a0 + i;
            if (a >= rows) continue;
            const long long o = ((row0 + a / rank) * nh + h) * rank + a % rank;
            const bool kept = S0 + a / rank >= sf;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = nb + lane + 32 * j;
                if (n < ds) dk[o * ds + n] = kept ? __fmul_rn(cfs[i], acc[i][j]) : 0.0f;
            }
        }
    };
    // ste's 128 channels from pb: du and the trees' operand.
    auto store_ste = [&](int pb, float (&acc)[4][4]) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int a = a0 + i;
            if (a >= rows) continue;
            const long long o = ((row0 + a / rank) * nh + h) * rank + a % rank;
            const bool kept = S0 + a / rank >= sf;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int p = pb + lane + 32 * j;
                if (p >= hd) continue;
                if (kept) du[o * hd + p] = __fmaf_rn(cfs[i], acc[i][j], du[o * hd + p]);
                steb[a * hd + p] = acc[i][j];
            }
        }
    };
    if (whole) {
        float* dss = sm;           // [P][N + 1]
        float* kt = ut + hd * rs;  // [N][rs]
        // The chunk's dS, four of a warp's rows and four indices a lane in
        // flight together.
        for (int p0 = warp; p0 < hd; p0 += 4 * nw) {
            for (int n0 = 0; n0 < ds; n0 += 128) {
                float dv[4][4];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const int p = p0 + u * nw;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int n = n0 + lane + 32 * j;
                        dv[u][j] = p < hd && n < ds ? dsc[(long long)p * ds + n] : 0.0f;
                    }
                }
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const int p = p0 + u * nw;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int n = n0 + lane + 32 * j;
                        if (p < hd && n < ds) dss[p * dld + n] = dv[u][j];
                    }
                }
            }
        }
        m3m_stage_rot(warp, nw, rows, rs, 0, ds, half, na, key_row,
            [&](int a, int n, float v) { kt[n * rs + a] = v; });
        m3m_stage_u<T>(warp, nw, rows, rs, hd, hd,
            [&](int a) { return proj + (row0 + a / rank) * ld_proj + di + (long long)h * hd; },
            [&](int a) { return mimo_x + ((long long)h * rank + a % rank) * hd; },
            [&](int a, int p, float v) { ut[p * rs + a] = v; });
        __syncthreads();
        for (int nb = 0; nb < ds; nb += 128) {
            float acc[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
            }
            for (int p = 0; p < hd; p++) {
                const float4 u4 = *reinterpret_cast<const float4*>(ut + p * rs + a0);
                const float uv[4] = {u4.x, u4.y, u4.z, u4.w};
                float sv[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int n = nb + lane + 32 * j;
                    sv[j] = n < ds ? dss[p * dld + n] : 0.0f;
                }
#pragma unroll
                for (int i = 0; i < 4; i++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(sv[j], uv[i], acc[i][j]);
                }
            }
            store_dk(nb, acc);
        }
        for (int pb = 0; pb < hd; pb += 128) {
            float acc[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
            }
            for (int n = 0; n < ds; n++) {
                const float4 k4 = *reinterpret_cast<const float4*>(kt + n * rs + a0);
                const float kv[4] = {k4.x, k4.y, k4.z, k4.w};
                float sv[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int p = pb + lane + 32 * j;
                    sv[j] = p < hd ? dss[p * dld + n] : 0.0f;
                }
#pragma unroll
                for (int i = 0; i < 4; i++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(sv[j], kv[i], acc[i][j]);
                }
            }
            store_ste(pb, acc);
        }
    } else {
        // dk: u's channels and dS's rows from p0, at the pass's columns.
        float* us = sm;              // [ps][rs]
        float* as = sm + ps * rs;    // [ps][aw]
        for (int nb = 0; nb < ds; nb += 128) {
            const int nwid = min(128, ds - nb);
            float acc[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
            }
            for (int p0 = 0; p0 < hd; p0 += ps) {
                const int pn = min(ps, hd - p0);
                __syncthreads();
                m3m_stage_u<T>(warp, nw, rows, rs, pn, pn,
                    [&](int a) {
                        return proj + (row0 + a / rank) * ld_proj + di + (long long)h * hd + p0;
                    },
                    [&](int a) { return mimo_x + ((long long)h * rank + a % rank) * hd + p0; },
                    [&](int a, int p, float v) { us[p * rs + a] = v; });
                for (int q0 = warp; q0 < pn; q0 += 4 * nw) {
                    float dv[4][4];
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        const int p = q0 + u * nw;
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            const int n = lane + 32 * j;
                            dv[u][j] = p < pn && n < nwid
                                ? dsc[(long long)(p0 + p) * ds + nb + n] : 0.0f;
                        }
                    }
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        const int p = q0 + u * nw;
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            const int n = lane + 32 * j;
                            if (p < pn && n < nwid) as[p * aw + n] = dv[u][j];
                        }
                    }
                }
                __syncthreads();
                for (int p = 0; p < pn; p++) {
                    const float4 u4 = *reinterpret_cast<const float4*>(us + p * rs + a0);
                    const float uv[4] = {u4.x, u4.y, u4.z, u4.w};
                    float sv[4];
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int n = lane + 32 * j;
                        sv[j] = n < nwid ? as[p * aw + n] : 0.0f;
                    }
#pragma unroll
                    for (int i = 0; i < 4; i++) {
#pragma unroll
                        for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(sv[j], uv[i], acc[i][j]);
                    }
                }
            }
            store_dk(nb, acc);
        }
        // ste: the keys' state indices and dS's columns from n0, at the
        // pass's rows.
        float* ks = sm;              // [ns][rs]
        float* bs = sm + ns * rs;    // [pw][ldb]
        for (int pb = 0; pb < hd; pb += 128) {
            const int pwid = min(128, hd - pb);
            float acc[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
            }
            for (int n0 = 0; n0 < ds; n0 += ns) {
                const int nn = min(ns, ds - n0);
                __syncthreads();
                m3m_stage_rot(warp, nw, rows, rs, n0, nn, half, na, key_row,
                    [&](int a, int n, float v) { ks[n * rs + a] = v; });
                for (int q = warp; q < pwid; q += nw) {
                    const float* src = dsc + (long long)(pb + q) * ds + n0;
                    for (int n = lane; n < nn; n += 32) bs[q * ldb + n] = src[n];
                }
                __syncthreads();
                for (int n = 0; n < nn; n++) {
                    const float4 k4 = *reinterpret_cast<const float4*>(ks + n * rs + a0);
                    const float kv[4] = {k4.x, k4.y, k4.z, k4.w};
                    float sv[4];
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int p = lane + 32 * j;
                        sv[j] = p < pwid ? bs[p * ldb + n] : 0.0f;
                    }
#pragma unroll
                    for (int i = 0; i < 4; i++) {
#pragma unroll
                        for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(sv[j], kv[i], acc[i][j]);
                    }
                }
            }
            store_ste(pb, acc);
        }
    }
    __syncthreads();
    const int rp = rank * hd;
    for (int e = tid; e < tn * 256; e += blockDim.x) {
        const int tl = e >> 8, j = e & 255;
        float part = 0.0f;
        for (int i = j; i < rp; i += 256) {
            const int r = i / hd, p = i - r * hd;
            const int a = tl * rank + r;
            const float uv = whole
                ? ut[p * rs + a]
                : __fmul_rn(mimo_x[((long long)h * rank + r) * hd + p],
                            to_f(proj[(row0 + tl) * ld_proj + di + (long long)h * hd + p]));
            part = __fmaf_rn(uv, steb[a * hd + p], part);
        }
        red[e] = part;
    }
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        for (int e = tid; e < tn * w; e += blockDim.x) {
            const int tl = e / w, j = e - tl * w;
            red[tl * 256 + j] = __fadd_rn(red[tl * 256 + j], red[tl * 256 + j + w]);
        }
        __syncthreads();
    }
    for (int tl = tid; tl < tn; tl += blockDim.x)
        dcs[(row0 + tl) * nh + h] = S0 + tl >= sf ? red[tl * 256] : 0.0f;
}

#define DEFINE_M3MB_STATE_TERMS_TILES(NAME, T)                                 \
extern "C" __global__ void __launch_bounds__(512) NAME(                        \
    float* __restrict__ dk, float* __restrict__ du, float* __restrict__ dcs,   \
    const float* __restrict__ dst, const T* __restrict__ proj,                 \
    long long ld_proj,                                                         \
    const float* __restrict__ bn, const float* __restrict__ b_bias,            \
    const float* __restrict__ coef, const float* __restrict__ sn,              \
    const float* __restrict__ cs, const float* __restrict__ mimo_x,            \
    float* __restrict__ steb_glob,                                             \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tt,         \
    int whole, int ps, int ns,                                                 \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3mb_state_terms_tiles_body<T>(dk, du, dcs, dst, proj, ld_proj, bn, b_bias, \
        coef, sn, cs, mimo_x, steb_glob, seq, nh, ng, hd, ds, rank, na, tt,    \
        whole, ps, ns, y_first, seq_idx);                                      \
}
DEFINE_M3MB_STATE_TERMS_TILES(m3mb_state_terms_tiles, float)
DEFINE_M3MB_STATE_TERMS_TILES(m3mb_state_terms_tiles_bf16, __nv_bfloat16)
DEFINE_M3MB_STATE_TERMS_TILES(m3mb_state_terms_tiles_f16, __half)

// The pair terms of a chunk and head. Per pair (t, s), s <= t in one
// sequence, G = q_t . k_s over the state indices and M = dy_t . u_s over the
// channels for every (r, r'), and the pair's weight w(t, s) give
//   dw(t, s) = sum_{r, r'} G M     -> dw [B, nc, H, pairs], pair t(t+1)/2 + s
//   dq_t += sum_{s, r'} (w M) k_s
//   dk_s += sum_{t, r} (w M) q_t
//   du_s += sum_{t, r} (w G) dy_t
// in two kernels split by the output each block owns, so nothing is read
// back from global memory per pair and the accumulators stay in registers.
// Both run tiles of their own `tl` tokens (tr = tl R rows, TR4 = tr rounded
// up to 4), a power of two, and keep the chains of the pair kernel as it
// was released (frozen in the tests), one block per chunk and head walking
// tiles of `tl_ref` tokens: every chain runs over the same columns in the
// same order, and the masked zeros its tiles put in a chain (the keys past
// a row's token up to the end of its tile, the queries before a key's token
// from the start of its tile) are walked as it walked them, so the bits are
// its own at any tile. A tile past the diagonal that only those zeros reach
// skips its products. A warp takes 4 rows of a product, a lane its columns
// lane + 32 j; a block takes 128 columns of its outputs, the products'
// reductions staged a slice at a time, so any head and state width runs.
//
// dq, one block per (t-tile, 128 state indices): dq_t = its entering
// value, then over the s-tiles ascending, col = (s, r') ascending,
// fma(w M, k_s[r'], dq). M = dy . u (p ascending from 0), zero past the
// diagonal, over slices of `ps` channels (dy transposed staged once when
// the slice is the whole head). Shared memory: dy [ps][TR4 + 4]
// (transposed), u [tr][ps + 1], the keys' 128 indices [tr][W] (W = min(N,
// 128)), padded to a float4 multiple so the float4 rows of w M [tr][TR4 + 4]
// (transposed) start on 16 bytes, w [tl][tl]. With a packing a row's chain
// runs over the keys of its own sequence only, as the released kernel ran
// it. Grid (B * nc, H, ceil(64 / tl) * ceil(N / 128)), block 8 TR4 threads.
template <typename T>
__device__ __forceinline__ void m3mb_pairs_dq_body(
    float* __restrict__ dq,
    const float* __restrict__ dy, const T* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ b_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ mimo_x,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tl, int tl_ref, int ps,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float4 m3mb_sm4[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int nq = (ds + 127) >> 7;
    const int T0 = (int)(blockIdx.z / nq) * tl;
    if (T0 >= lc) return;
    const int nb = (int)(blockIdx.z % nq) * 128;
    const int nw_ = min(128, ds - nb), wmax = min(ds, 128);
    const int tn = min(tl, lc - T0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const int tr = tl * rank, ta = tn * rank;
    const int trs = ((tr + 3) & ~3) + 4;
    const int pss = min(ps, hd);
    const bool whole = pss == hd;
    const int uld = pss + 1;
    float* dyt = reinterpret_cast<float*>(m3mb_sm4);  // [pss][trs]
    float* us = dyt + pss * trs;                       // [tr][uld]
    float* ks = us + tr * uld;                         // [tr][wmax]
    float* mt = us + ((tr * (uld + wmax) + 3) & ~3);   // [tr][trs]
    float* ws = mt + tr * trs;                         // [tl][tl]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5, nw = blockDim.x >> 5;
    const int a0 = 4 * warp;
    const long long trow0 = b * seq + t0 + T0;
    // dy transposed at the channels p0 .. p0 + pn, four of a warp's rows
    // and four channels a lane in flight together.
    auto stage_dy = [&](int p0, int pn) {
        for (int q0 = warp; q0 < trs; q0 += 4 * nw) {
            for (int pq = 0; pq < pn; pq += 128) {
                float dv[4][4];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const int a = q0 + u * nw;
                    const float* row =
                        dy + ((trow0 + a / rank) * nh + h) * rank * hd + (a % rank) * hd + p0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int p = pq + lane + 32 * j;
                        dv[u][j] = a < ta && p < pn ? row[p] : 0.0f;
                    }
                }
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const int a = q0 + u * nw;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int p = pq + lane + 32 * j;
                        if (a < trs && p < pn) dyt[p * trs + a] = dv[u][j];
                    }
                }
            }
        }
    };
    if (whole) stage_dy(0, hd);
    float acc[4][4];
    // Each row's token, the first token of its packed sequence, and without
    // a packing the end of the released kernel's tile of the token, where
    // its chain ends.
    int trw[4], tfst[4], tend[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int a = a0 + i;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = nb + lane + 32 * j;
            acc[i][j] = a < ta && n < ds
                ? dq[(((trow0 + a / rank) * nh + h) * rank + a % rank) * ds + n] : 0.0f;
        }
        trw[i] = a < ta ? T0 + a / rank : -1;
        tfst[i] = a < ta && seq_idx != nullptr
            ? vl_seq_first(seq_idx, b, seq, t0, t0 + trw[i]) - t0 : 0;
        tend[i] = a < ta ? min((trw[i] / tl_ref + 1) * tl_ref, lc) : 0;
    }
    // The s-tiles a row's chain reaches: with a packing up to its own,
    // else to the end of the released kernel's tile of the tile's last
    // token, its zeros included.
    const int s_end = seq_idx == nullptr ? min(((T0 + tn - 1) / tl_ref + 1) * tl_ref, lc) : T0 + tn;
    for (int S0 = 0; S0 < s_end; S0 += tl) {
        const int sn_ = min(tl, lc - S0);
        const int sb = sn_ * rank;
        const long long srow0 = b * seq + t0 + S0;
        // Past the tile's last token every pair is masked: only the zeros
        // of the released kernel's tiles reach here, with the keys they
        // multiply.
        const bool past = S0 > T0 + tn - 1;
        __syncthreads();
        m3m_stage_rot(warp, nw, sb, sb, nb, nw_, half, na,
            [&](int col) {
                const int r = col % rank;
                const long long row = srow0 + col / rank;
                return M3mRotRow{bn + row * rgn + ((long long)r * ng + g) * ds,
                                 b_bias + ((long long)h * rank + r) * ds,
                                 sn + (row * nh + h) * na, cs + (row * nh + h) * na};
            },
            [&](int col, int n, float v) { ks[col * wmax + n] = v; });
        float m[4][2];
#pragma unroll
        for (int i = 0; i < 4; i++) m[i][0] = m[i][1] = 0.0f;
        if (!past) {
            if (whole)
                m3m_stage_u<T>(warp, nw, sb, sb, hd, hd,
                    [&](int col) {
                        return proj + (srow0 + col / rank) * ld_proj + di + (long long)h * hd;
                    },
                    [&](int col) { return mimo_x + ((long long)h * rank + col % rank) * hd; },
                    [&](int col, int p, float v) { us[col * uld + p] = v; });
            for (int i = tid; i < tn * sn_; i += blockDim.x) {
                const int tq = i / sn_, sk = i % sn_;
                const int t = T0 + tq, s = S0 + sk;
                float w = 0.0f;
                if (s <= t && vl_id(seq_idx, b, seq, t0 + s) == vl_id(seq_idx, b, seq, t0 + t)) {
                    const float* ct = coef + ((trow0 + tq) * nh + h) * 4;
                    const float* cs_ = coef + ((srow0 + sk) * nh + h) * 4;
                    w = s < t ? __fmul_rn(mimo_exp(fminf(__fsub_rn(ct[3], cs_[3]), 0.0f)), cs_[2])
                              : ct[1];
                }
                ws[tq * tl + sk] = w;
            }
            for (int p0 = 0; p0 < hd; p0 += pss) {
                const int pn = min(pss, hd - p0);
                if (!whole) {
                    __syncthreads();
                    stage_dy(p0, pn);
                    m3m_stage_u<T>(warp, nw, sb, sb, pn, pn,
                        [&](int col) {
                            return proj + (srow0 + col / rank) * ld_proj + di + (long long)h * hd + p0;
                        },
                        [&](int col) {
                            return mimo_x + ((long long)h * rank + col % rank) * hd + p0;
                        },
                        [&](int col, int p, float v) { us[col * uld + p] = v; });
                }
                __syncthreads();
                for (int p = 0; p < pn; p++) {
                    const float4 d4 = *reinterpret_cast<const float4*>(dyt + p * trs + a0);
                    const float dv[4] = {d4.x, d4.y, d4.z, d4.w};
                    const float u0 = lane < sb ? us[lane * uld + p] : 0.0f;
                    const float u1 = lane + 32 < sb ? us[(lane + 32) * uld + p] : 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        m[i][0] = __fmaf_rn(dv[i], u0, m[i][0]);
                        m[i][1] = __fmaf_rn(dv[i], u1, m[i][1]);
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int a = a0 + i;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int col = lane + 32 * j;
                if (col >= tr) continue;
                float v = 0.0f;
                if (!past && a < ta && col < sb) {
                    const int tq = a / rank, sk = col / rank;
                    if (S0 + sk <= T0 + tq) v = __fmul_rn(ws[tq * tl + sk], m[i][j]);
                }
                mt[col * trs + a] = v;
            }
        }
        __syncthreads();
        if (seq_idx == nullptr && S0 + tl <= T0) {
            for (int col = 0; col < sb; col++) {
                const float4 m4 = *reinterpret_cast<const float4*>(mt + col * trs + a0);
                const float mv[4] = {m4.x, m4.y, m4.z, m4.w};
                float kv[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int n = lane + 32 * j;
                    kv[j] = n < nw_ ? ks[col * wmax + n] : 0.0f;
                }
#pragma unroll
                for (int i = 0; i < 4; i++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(mv[i], kv[j], acc[i][j]);
                }
            }
        } else {
            // A row takes the keys of its own sequence up to itself, or
            // without a packing every key to the end of the released kernel's
            // tile of its token, the masked ones as the zeros it added.
            int lo[4], hi[4];
            int c0 = sb, c1 = 0;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int last = seq_idx == nullptr ? tend[i] : trw[i] + 1;
                lo[i] = trw[i] < 0 ? 0 : max(tfst[i] - S0, 0) * rank;
                hi[i] = trw[i] < 0 ? 0 : max(min(last - S0, sn_), 0) * rank;
                if (lo[i] < hi[i]) {
                    c0 = min(c0, lo[i]);
                    c1 = max(c1, hi[i]);
                }
            }
            for (int col = c0; col < c1; col++) {
                const float4 m4 = *reinterpret_cast<const float4*>(mt + col * trs + a0);
                const float mv[4] = {m4.x, m4.y, m4.z, m4.w};
                float kv[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int n = lane + 32 * j;
                    kv[j] = n < nw_ ? ks[col * wmax + n] : 0.0f;
                }
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (col >= lo[i] && col < hi[i]) {
#pragma unroll
                        for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(mv[i], kv[j], acc[i][j]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int a = a0 + i;
        if (a >= ta) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = nb + lane + 32 * j;
            if (n < ds) dq[(((trow0 + a / rank) * nh + h) * rank + a % rank) * ds + n] = acc[i][j];
        }
    }
}

// dk, du and dw, one block per (s-tile, role): over the t-tiles ascending,
// G = q . k (n ascending from 0) and M = dy . u (p ascending from 0) give
// dw per pair (r, then r', as the released kernel), then
//   dk_s += (w M) q_t and du_s += (w G) dy_t, a = (t, r) ascending,
// each from its value on entry (the state terms). Role j takes dk's state
// indices 128 j onward and du's channels 128 j onward, 128 of each where
// they exist, and role 0 dw: a role forms only the products its outputs
// read. The products stage their operands a slice at a time (`kns` state
// indices, `pss` channels), the s-tile's keys and u transposed once where a
// slice is the whole width. Shared memory: the keys [kns][TR4 + 4] and u
// [pss][TR4 + 4] transposed, the queries [tr][QLD] and dy [tr][DLD] (the
// width plus one where whole, else the larger of the slice and 128, plus
// one), padded to a float4 multiple so the float4 rows of G and M
// [tr][TR4 + 4] start on 16 bytes, then the pairs' w and live flags
// [tl][tl]. A thread a pair walks dw's chain while the others wait, then
// every thread scales its share of G and M into w G and w M in place. With
// a packing a pair of two sequences is no pair (its dw is zero) and a row's
// chains run over the queries of its own sequence only, as the released
// kernel ran them. Grid (B * nc, H, ceil(64 / tl) * max(ceil(N / 128),
// ceil(P / 128))), block 8 TR4 threads.
template <typename T>
__device__ __forceinline__ void m3mb_pairs_dk_body(
    float* __restrict__ dk, float* __restrict__ du, float* __restrict__ dwp,
    const float* __restrict__ dy, const T* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ cn,
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ mimo_x,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tl, int tl_ref, int kns,
    int pss,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float4 m3mb_sm4[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int nk = (ds + 127) >> 7, nu = (hd + 127) >> 7, roles = max(nk, nu);
    const int S0 = (int)(blockIdx.z / roles) * tl;
    if (S0 >= lc) return;
    const int role = (int)(blockIdx.z % roles);
    const bool do_dk = role < nk, do_du = role < nu, do_dw = role == 0;
    const bool need_g = do_du || do_dw, need_m = do_dk || do_dw;
    const int nb = role * 128, pb = role * 128;
    const int sn_ = min(tl, lc - S0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const int tr = tl * rank, sb = sn_ * rank;
    const int trs = ((tr + 3) & ~3) + 4;
    const int kn = min(kns, ds), pn = min(pss, hd);
    const bool kwhole = kn == ds, pwhole = pn == hd;
    const int qld = (kwhole ? ds : max(kn, 128)) + 1;
    const int dld = (pwhole ? hd : max(pn, 128)) + 1;
    float* kt = reinterpret_cast<float*>(m3mb_sm4);  // [kn][trs]
    float* ut = kt + kn * trs;                        // [pn][trs]
    float* qs = ut + pn * trs;                        // [tr][qld]
    float* dys = qs + tr * qld;                       // [tr][dld]
    float* gm = qs + ((tr * (qld + dld) + 3) & ~3);   // [tr][trs] G, then w G
    float* mm = gm + tr * trs;                        // [tr][trs] M, then w M
    float* ws = mm + tr * trs;                        // [tl][tl] w per pair
    int* wl = reinterpret_cast<int*>(ws + tl * tl);   // [tl][tl] live pairs
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5, nw = blockDim.x >> 5;
    const int b0 = 4 * warp;
    const long long srow0 = b * seq + t0 + S0;
    const long long pairs = M3M_CHUNK * (M3M_CHUNK + 1) / 2;
    // The s-tile's rotated keys at n0 .. n0 + kw and its u at p0 .. p0 + pw,
    // transposed.
    auto stage_keys = [&](int n0, int kw) {
        for (int col = warp; col < sb; col += nw) {
            const int r = col % rank;
            const long long row = srow0 + col / rank;
            const float* src = bn + row * rgn + ((long long)r * ng + g) * ds;
            const float* bias = b_bias + ((long long)h * rank + r) * ds;
            const float* sr_ = sn + (row * nh + h) * na;
            const float* cr_ = cs + (row * nh + h) * na;
#pragma unroll 4
            for (int n = lane; n < kw; n += 32)
                kt[n * trs + col] = m3m_rot(src, bias, n0 + n, half, na, sr_, cr_);
        }
    };
    auto stage_u = [&](int p0, int pw) {
        for (int col = warp; col < sb; col += nw) {
            const long long row = srow0 + col / rank;
            const T* xr = proj + row * ld_proj + di + (long long)h * hd + p0;
            const float* mx = mimo_x + ((long long)h * rank + col % rank) * hd + p0;
            for (int p = lane; p < pw; p += 32) ut[p * trs + col] = __fmul_rn(mx[p], to_f(xr[p]));
        }
    };
    // The t-tile's rotated queries at n0 .. n0 + kw and its dy at
    // p0 .. p0 + pw, a row each.
    auto stage_q = [&](long long trow0, int ta, int n0, int kw) {
        for (int a = warp; a < ta; a += nw) {
            const int r = a % rank;
            const long long row = trow0 + a / rank;
            const float* src = cn + row * rgn + ((long long)r * ng + g) * ds;
            const float* bias = c_bias + ((long long)h * rank + r) * ds;
            const float* sr_ = sn + (row * nh + h) * na;
            const float* cr_ = cs + (row * nh + h) * na;
#pragma unroll 4
            for (int n = lane; n < kw; n += 32)
                qs[a * qld + n] = m3m_rot(src, bias, n0 + n, half, na, sr_, cr_);
        }
    };
    auto stage_dy = [&](long long trow0, int ta, int p0, int pw) {
        for (int a = warp; a < ta; a += nw) {
            const long long row = trow0 + a / rank;
            const float* dr = dy + (row * nh + h) * rank * hd + (a % rank) * hd + p0;
            for (int p = lane; p < pw; p += 32) dys[a * dld + p] = dr[p];
        }
    };
    if (kwhole && need_g) stage_keys(0, ds);
    if (pwhole && need_m) stage_u(0, hd);
    float dka[4][4], dua[4][4];
    // Each row's token, the end of its packed sequence in the chunk, and
    // without a packing the start of the released kernel's tile of the
    // token, where its chains start.
    int srw[4], send[4], sbeg[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int col = b0 + i;
        const long long krow = (srow0 + col / rank) * nh * rank + (long long)h * rank + col % rank;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = nb + lane + 32 * j, p = pb + lane + 32 * j;
            dka[i][j] = do_dk && col < sb && n < ds ? dk[krow * ds + n] : 0.0f;
            dua[i][j] = do_du && col < sb && p < hd ? du[krow * hd + p] : 0.0f;
        }
        srw[i] = col < sb ? S0 + col / rank : -1;
        send[i] = col < sb && seq_idx != nullptr
            ? vl_span(seq_idx, b, seq, t0, lc, srw[i]).end : lc;
        sbeg[i] = col < sb ? srw[i] / tl_ref * tl_ref : 0;
    }
    // Without a packing the chains start at the released kernel's tile of
    // the tile's first token, which a smaller tile reaches from before its
    // own.
    const int t_start = seq_idx == nullptr ? S0 / tl_ref * tl_ref : S0;
    for (int T0 = t_start; T0 < lc; T0 += tl) {
        const int tn = min(tl, lc - T0);
        const int ta = tn * rank;
        const long long trow0 = b * seq + t0 + T0;
        // Before the s-tile's first token every pair is masked: only the
        // zeros of the released kernel's tiles reach here, with the queries
        // and dy they multiply.
        const bool before = T0 + tn - 1 < S0;
        // G and M with rows col (s, r') and columns a = (t, r).
        float ga[4][2], ma[4][2];
#pragma unroll
        for (int i = 0; i < 4; i++) ga[i][0] = ga[i][1] = ma[i][0] = ma[i][1] = 0.0f;
        if (need_g && !before) {
            for (int n0 = 0; n0 < ds; n0 += kn) {
                const int kw = min(kn, ds - n0);
                __syncthreads();
                if (!kwhole) stage_keys(n0, kw);
                stage_q(trow0, ta, n0, kw);
                __syncthreads();
                for (int n = 0; n < kw; n++) {
                    const float4 k4 = *reinterpret_cast<const float4*>(kt + n * trs + b0);
                    const float kv[4] = {k4.x, k4.y, k4.z, k4.w};
                    const float q0 = lane < ta ? qs[lane * qld + n] : 0.0f;
                    const float q1 = lane + 32 < ta ? qs[(lane + 32) * qld + n] : 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        ga[i][0] = __fmaf_rn(q0, kv[i], ga[i][0]);
                        ga[i][1] = __fmaf_rn(q1, kv[i], ga[i][1]);
                    }
                }
            }
        }
        if (need_m && !before) {
            for (int p0 = 0; p0 < hd; p0 += pn) {
                const int pw = min(pn, hd - p0);
                __syncthreads();
                if (!pwhole) stage_u(p0, pw);
                stage_dy(trow0, ta, p0, pw);
                __syncthreads();
                for (int p = 0; p < pw; p++) {
                    const float4 u4 = *reinterpret_cast<const float4*>(ut + p * trs + b0);
                    const float uv[4] = {u4.x, u4.y, u4.z, u4.w};
                    const float d0 = lane < ta ? dys[lane * dld + p] : 0.0f;
                    const float d1 = lane + 32 < ta ? dys[(lane + 32) * dld + p] : 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        ma[i][0] = __fmaf_rn(d0, uv[i], ma[i][0]);
                        ma[i][1] = __fmaf_rn(d1, uv[i], ma[i][1]);
                    }
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int col = b0 + i;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int a = lane + 32 * j;
                if (a >= ta || col >= sb) continue;
                const int t = T0 + a / rank, s = S0 + col / rank;
                const bool live = s <= t
                    && vl_id(seq_idx, b, seq, t0 + s) == vl_id(seq_idx, b, seq, t0 + t);
                gm[a * trs + col] = live ? ga[i][j] : 0.0f;
                mm[a * trs + col] = live ? ma[i][j] : 0.0f;
            }
        }
        __syncthreads();
        // dw per pair, a thread a pair walking (r, r') in order over G and M
        // as the products left them, with the pair's weight for the scaling
        // after it; a pair of two sequences or past the diagonal is masked.
        for (int i = tid; i < tn * sn_; i += blockDim.x) {
            const int tq = i / sn_, sk = i % sn_;
            const int t = T0 + tq, s = S0 + sk;
            const bool apart = vl_id(seq_idx, b, seq, t0 + s) != vl_id(seq_idx, b, seq, t0 + t);
            const bool live = s <= t && !apart;
            wl[tq * tl + sk] = live;
            if (!live) {
                if (do_dw && s <= t)
                    dwp[(bc * nh + h) * pairs + (long long)t * (t + 1) / 2 + s] = 0.0f;
                continue;
            }
            const float* ct = coef + ((trow0 + tq) * nh + h) * 4;
            const float* cs_ = coef + ((srow0 + sk) * nh + h) * 4;
            ws[tq * tl + sk] = s < t
                ? __fmul_rn(mimo_exp(fminf(__fsub_rn(ct[3], cs_[3]), 0.0f)), cs_[2])
                : ct[1];
            if (!do_dw) continue;
            float dw = 0.0f;
            for (int r = 0; r < rank; r++) {
                const float* gr = gm + (tq * rank + r) * trs + sk * rank;
                const float* mr = mm + (tq * rank + r) * trs + sk * rank;
#pragma unroll 4
                for (int r2 = 0; r2 < rank; r2++) dw = __fmaf_rn(gr[r2], mr[r2], dw);
            }
            dwp[(bc * nh + h) * pairs + (long long)t * (t + 1) / 2 + s] = dw;
        }
        __syncthreads();
        // w G and w M over the whole tile pair, every thread a share.
        for (int e = tid; e < ta * sb; e += blockDim.x) {
            const int a = e / sb, col = e - a * sb;
            const int pi = a / rank * tl + col / rank;
            const int at = a * trs + col;
            const bool live = wl[pi];
            gm[at] = live ? __fmul_rn(ws[pi], gm[at]) : 0.0f;
            mm[at] = live ? __fmul_rn(ws[pi], mm[at]) : 0.0f;
        }
        // The queries and dy the role's columns read: where a product left
        // the whole width staged they are there, else a slice of 128.
        const bool q_staged = need_g && kwhole && !before;
        const bool dy_staged = need_m && pwhole && !before;
        const int qoff = q_staged ? nb : 0, doff = dy_staged ? pb : 0;
        if ((do_dk && !q_staged) || (do_du && !dy_staged)) {
            __syncthreads();
            if (do_dk && !q_staged) stage_q(trow0, ta, nb, min(128, ds - nb));
            if (do_du && !dy_staged) stage_dy(trow0, ta, pb, min(128, hd - pb));
        }
        __syncthreads();
        int lo[4], hi[4];
        int a_lo = 0, a_hi = ta;
        if (seq_idx != nullptr || T0 <= S0) {
            // A row takes the queries of its own sequence from itself on, or
            // without a packing every query from the start of the released
            // kernel's tile of its token, the masked ones as the zeros it
            // added.
            a_lo = ta;
            a_hi = 0;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int first = seq_idx == nullptr ? sbeg[i] : srw[i];
                lo[i] = srw[i] < 0 ? 0 : max(first - T0, 0) * rank;
                hi[i] = srw[i] < 0 ? 0 : max(min(send[i] - T0, tn), 0) * rank;
                if (lo[i] < hi[i]) {
                    a_lo = min(a_lo, lo[i]);
                    a_hi = max(a_hi, hi[i]);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                lo[i] = 0;
                hi[i] = ta;
            }
        }
        for (int a = a_lo; a < a_hi; a++) {
            float mv[4], wv[4], qv[4], dv[4];
            if (do_dk) {
                const float4 m4 = *reinterpret_cast<const float4*>(mm + a * trs + b0);
                mv[0] = m4.x; mv[1] = m4.y; mv[2] = m4.z; mv[3] = m4.w;
            }
            if (do_du) {
                const float4 w4 = *reinterpret_cast<const float4*>(gm + a * trs + b0);
                wv[0] = w4.x; wv[1] = w4.y; wv[2] = w4.z; wv[3] = w4.w;
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = lane + 32 * j;
                qv[j] = do_dk && nb + n < ds ? qs[a * qld + qoff + n] : 0.0f;
                dv[j] = do_du && pb + n < hd ? dys[a * dld + doff + n] : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (a >= lo[i] && a < hi[i]) {
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        if (do_dk) dka[i][j] = __fmaf_rn(mv[i], qv[j], dka[i][j]);
                        if (do_du) dua[i][j] = __fmaf_rn(wv[i], dv[j], dua[i][j]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int col = b0 + i;
        if (col >= sb) continue;
        const long long krow = (srow0 + col / rank) * nh * rank + (long long)h * rank + col % rank;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = nb + lane + 32 * j, p = pb + lane + 32 * j;
            if (do_dk && n < ds) dk[krow * ds + n] = dka[i][j];
            if (do_du && p < hd) du[krow * hd + p] = dua[i][j];
        }
    }
}

// The split pair kernels come in two builds. The wide one takes blocks of up
// to 512 threads, which caps a thread at 128 registers: dk's accumulators
// and products then spill every token. The narrow one takes at most 256
// threads at one block a multiprocessor, which keeps dk in 218 to 221
// registers without a spill at half the warps. The spills cost more than
// the warps on wide heads, and on narrower heads at wide states and high
// ranks. Every chain keeps its columns and order at any tile, so both give
// the same bits, and the host picks one.
#define M3MB_PAIRS_BOUNDS_WIDE __launch_bounds__(512)
#define M3MB_PAIRS_BOUNDS_NARROW __launch_bounds__(256, 1)

#define DEFINE_M3MB_PAIRS_DQ(NAME, T, BOUNDS)                                  \
extern "C" __global__ void BOUNDS NAME(                                        \
    float* __restrict__ dq,                                                    \
    const float* __restrict__ dy, const T* __restrict__ proj,                  \
    long long ld_proj,                                                         \
    const float* __restrict__ bn, const float* __restrict__ b_bias,            \
    const float* __restrict__ coef, const float* __restrict__ sn,              \
    const float* __restrict__ cs, const float* __restrict__ mimo_x,            \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tl,         \
    int tl_ref, int ps,                                                        \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3mb_pairs_dq_body<T>(dq, dy, proj, ld_proj, bn, b_bias, coef, sn, cs,     \
        mimo_x, seq, nh, ng, hd, ds, rank, na, tl, tl_ref, ps, y_first,        \
        seq_idx);                                                              \
}
DEFINE_M3MB_PAIRS_DQ(m3mb_pairs_dq, float, M3MB_PAIRS_BOUNDS_WIDE)
DEFINE_M3MB_PAIRS_DQ(m3mb_pairs_dq_bf16, __nv_bfloat16, M3MB_PAIRS_BOUNDS_WIDE)
DEFINE_M3MB_PAIRS_DQ(m3mb_pairs_dq_f16, __half, M3MB_PAIRS_BOUNDS_WIDE)
DEFINE_M3MB_PAIRS_DQ(m3mb_pairs_dq_narrow, float, M3MB_PAIRS_BOUNDS_NARROW)
DEFINE_M3MB_PAIRS_DQ(m3mb_pairs_dq_narrow_bf16, __nv_bfloat16, M3MB_PAIRS_BOUNDS_NARROW)
DEFINE_M3MB_PAIRS_DQ(m3mb_pairs_dq_narrow_f16, __half, M3MB_PAIRS_BOUNDS_NARROW)

#define DEFINE_M3MB_PAIRS_DK(NAME, T, BOUNDS)                                  \
extern "C" __global__ void BOUNDS NAME(                                        \
    float* __restrict__ dk, float* __restrict__ du, float* __restrict__ dwp,   \
    const float* __restrict__ dy, const T* __restrict__ proj,                  \
    long long ld_proj,                                                         \
    const float* __restrict__ bn, const float* __restrict__ cn,                \
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,        \
    const float* __restrict__ coef, const float* __restrict__ sn,              \
    const float* __restrict__ cs, const float* __restrict__ mimo_x,            \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tl,         \
    int tl_ref, int kns, int pss,                                              \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3mb_pairs_dk_body<T>(dk, du, dwp, dy, proj, ld_proj, bn, cn, b_bias,      \
        c_bias, coef, sn, cs, mimo_x, seq, nh, ng, hd, ds, rank, na, tl, tl_ref, \
        kns, pss, y_first, seq_idx);                                           \
}
DEFINE_M3MB_PAIRS_DK(m3mb_pairs_dk, float, M3MB_PAIRS_BOUNDS_WIDE)
DEFINE_M3MB_PAIRS_DK(m3mb_pairs_dk_bf16, __nv_bfloat16, M3MB_PAIRS_BOUNDS_WIDE)
DEFINE_M3MB_PAIRS_DK(m3mb_pairs_dk_f16, __half, M3MB_PAIRS_BOUNDS_WIDE)
DEFINE_M3MB_PAIRS_DK(m3mb_pairs_dk_narrow, float, M3MB_PAIRS_BOUNDS_NARROW)
DEFINE_M3MB_PAIRS_DK(m3mb_pairs_dk_narrow_bf16, __nv_bfloat16, M3MB_PAIRS_BOUNDS_NARROW)
DEFINE_M3MB_PAIRS_DK(m3mb_pairs_dk_narrow_f16, __half, M3MB_PAIRS_BOUNDS_NARROW)

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
    int nh, int ng, int ds, int rank, int na,
    int y_first /* this run's first y block */
) {
    const long long row = blockIdx.x;
    const int h = y_first + blockIdx.y;
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
// its token on, to the end of its packed sequence (the angles restart at
// the next one's first token); d tanh_pi = that sum dt into d_tanh_h (the
// angle projection's backward input), and its share of ddt (sum dt-side, a
// ascending) into ddt_ang [rows, H]. Grid (B, H), block 32: one lane per
// angle runs its chain, and every token's ddt sum is one lane's.
//
// The walk takes M3MB_ANG_STEPS tokens at a time. The block first stages
// their angle gradients, their tanh_pi values, their dt and whether each
// starts a sequence, every load in flight at once; then each lane walks its
// angles' chains over the tokens, in time order; then a lane per token adds
// that token's products, a ascending. Every value is the one the
// token-at-a-time walk computed, so the bits are its bits. Shared memory:
// two [M3MB_ANG_STEPS][na] tiles, the products' tile with rows padded by
// one float (a lane per token reads down a column), and two per-token lanes
// (m3mb_angles_smem_bytes).
#define M3MB_ANG_STEPS 16
template <typename T>
__device__ __forceinline__ void m3mb_angles_body(
    float* __restrict__ d_tanh_h, float* __restrict__ ddt_ang,
    const float* __restrict__ dth, const float* __restrict__ coef,
    const T* __restrict__ proj, long long ld_proj, int ang_col,
    int seq, int nh, int na,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m3mb_sm[];
    float* dth_sm = m3mb_sm;                               // [steps][na]
    float* tp_sm = dth_sm + M3MB_ANG_STEPS * na;           // [steps][na]
    float* prod_sm = tp_sm + M3MB_ANG_STEPS * na;          // [steps][na + 1]
    float* dt_sm = prod_sm + M3MB_ANG_STEPS * (na + 1);    // [steps]
    int* start_sm = (int*)(dt_sm + M3MB_ANG_STEPS);        // [steps]
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
    float cum[8];
    const int per = (na + (int)blockDim.x - 1) / (int)blockDim.x;
    for (int k = 0; k < per && k < 8; k++) cum[k] = 0.0f;
    for (int hi = seq - 1; hi >= 0; hi -= M3MB_ANG_STEPS) {
        const int n = min(M3MB_ANG_STEPS, hi + 1);
        // Stage the tokens hi, hi - 1, .., hi - n + 1 (slot j is token hi - j).
        for (int i = threadIdx.x; i < n * na; i += blockDim.x) {
            const int j = i / na;
            const int a = i - j * na;
            const long long row = (long long)b * seq + (hi - j);
            dth_sm[i] = dth[(row * nh + h) * na + a];
            tp_sm[i] = mimo_tanh_pi(proj + row * ld_proj, ang_col, a);
        }
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            const long long row = (long long)b * seq + (hi - j);
            dt_sm[j] = coef[(row * nh + h) * 4];
            start_sm[j] = vl_starts(seq_idx, b, seq, hi - j);
        }
        __syncthreads();
        for (int j = 0; j < n; j++) {
            const long long row = (long long)b * seq + (hi - j);
            for (int k = 0; k < per && k < 8; k++) {
                const int a = threadIdx.x + k * blockDim.x;
                if (a < na) {
                    cum[k] = __fadd_rn(cum[k], dth_sm[j * na + a]);
                    d_tanh_h[(row * nh + h) * na + a] = __fmul_rn(cum[k], dt_sm[j]);
                    prod_sm[j * (na + 1) + a] = __fmul_rn(cum[k], tp_sm[j * na + a]);
                }
            }
            if (start_sm[j]) {
                for (int k = 0; k < per && k < 8; k++) cum[k] = 0.0f;
            }
        }
        __syncthreads();
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            const long long row = (long long)b * seq + (hi - j);
            float s = 0.0f;
            for (int a = 0; a < na; a++) s = __fadd_rn(s, prod_sm[j * (na + 1) + a]);
            ddt_ang[row * nh + h] = s;
        }
        __syncthreads();
    }
}

#define DEFINE_M3MB_ANGLES(NAME, T)                                            \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ d_tanh_h, float* __restrict__ ddt_ang,                 \
    const float* __restrict__ dth, const float* __restrict__ coef,             \
    const T* __restrict__ proj, long long ld_proj, int ang_col,                \
    int seq, int nh, int na,                                                   \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3mb_angles_body<T>(d_tanh_h, ddt_ang, dth, coef, proj, ld_proj, ang_col,  \
        seq, nh, na, y_first, seq_idx);                                        \
}
DEFINE_M3MB_ANGLES(m3mb_angles, float)
DEFINE_M3MB_ANGLES(m3mb_angles_bf16, __nv_bfloat16)
DEFINE_M3MB_ANGLES(m3mb_angles_f16, __half)
#undef M3MB_ANG_STEPS

// The decay and weight gradients of a chunk and head, one thread per token
// k (Mamba-2's gathering, with the pair values dw w, the state values
// dc c and the inter values I):
//   da_k     = stable_k + (next_k + prev_k)
//   dscale_k = sum_{t>k} dw(t, k) e^min(Acs_t - Acs_k, 0)
//              + dc_k e^min(Acs_last - Acs_k, 0)
//   dgamma_k = dw(k, k)
// written [rows, H, 3]. With a packing the pairs are those of k's own
// sequence, the state values those of the chunk's last sequence, and the
// state passing's term only reaches a chunk where no sequence starts (the
// inter values of the other tokens are zero). Grid (B * nc, H), block 64.
extern "C" __global__ void m3mb_combine(
    float* __restrict__ dsc3,
    const float* __restrict__ dwp, const float* __restrict__ dcs,
    const float* __restrict__ inter, const float* __restrict__ dot,
    const float* __restrict__ coef,
    int seq, int nh,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
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
    // k's sequence in the chunk, [k_first, k_end), and the chunk's last
    // sequence from `last`: the whole chunk without a packing.
    const VlSpan ks = vl_span(seq_idx, b, seq, t0, lc, k);
    const int last = vl_seq_first(seq_idx, b, seq, t0, t0 + lc - 1) - t0;
    float stable = 0.0f;
    for (int t = k; t < ks.end; t++)
        for (int s = ks.first; s < k; s++) {
            const float wv = wts(t, s);
            stable = __fadd_rn(stable, __fmul_rn(dw[(long long)t * (t + 1) / 2 + s], wv));
        }
    float nxt = 0.0f;
    for (int s = last; s < k; s++) {
        const float cv = __fmul_rn(mimo_exp(fminf(__fsub_rn(acs_last, acs(s)), 0.0f)), scale(s));
        nxt = __fadd_rn(nxt, __fmul_rn(dcs[(b * seq + t0 + s) * nh + h], cv));
    }
    const float pass = vl_passes(seq_idx, b, seq, t0, lc)
        ? __fmul_rn(dot[bc * nh + h], mimo_exp(acs_last)) : 0.0f;
    float prv = __fadd_rn(inter[(b * seq + t0 + lc - 1) * nh + h], pass);
    for (int t = lc - 2; t >= k; t--) prv = __fadd_rn(prv, inter[(b * seq + t0 + t) * nh + h]);
    const float da = __fadd_rn(stable, __fadd_rn(nxt, prv));
    float dsc = k >= last
        ? __fmul_rn(dcs[(b * seq + t0 + k) * nh + h],
                    mimo_exp(fminf(__fsub_rn(acs_last, acs(k)), 0.0f)))
        : 0.0f;
    for (int t = k + 1; t < ks.end; t++)
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
// [B, H, R, P]. Block a whole number of warps covering P, the tree's leaves
// the channels rounded up to a warp. A head wider than the block (blockDim
// a power of two then) gives each thread the channels p, p + blockDim and
// so on, its leaves past the first in `leaves` [B, H, lanes]
// (mimo_fold_leaves), so the tree is the one a block per channel runs.
// A row whose first token starts a packed sequence drops the carried state,
// so both are zero there.
extern "C" __global__ void m3mb_lead(
    float* __restrict__ de0, float* __restrict__ gpsi_init,
    const float* __restrict__ d_init, const float* __restrict__ e0,
    const float* __restrict__ kst, const float* __restrict__ vst,
    const float* __restrict__ mimo_x, float* __restrict__ leaves,
    int nh, int hd, int ds, int rank,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx, int seq
) {
    extern __shared__ float m3mb_sm[];
    const long long bh = (long long)blockIdx.x * nh + (y_first + blockIdx.y);
    const int h = y_first + blockIdx.y;
    const int tid = threadIdx.x, nt = blockDim.x;
    if (vl_starts(seq_idx, blockIdx.x, seq, 0)) {
        for (int i = tid; i < rank * hd; i += nt) gpsi_init[bh * rank * hd + i] = 0.0f;
        if (tid == 0) de0[bh] = 0.0f;
        return;
    }
    const int lanes = (hd + 31) & ~31;
    float* gl = leaves == nullptr ? nullptr : leaves + bh * lanes;
    float part0 = 0.0f;
    for (int p = tid; p < lanes; p += nt) {
        float part = 0.0f;
        if (p < hd) {
            const float v = vst[bh * hd + p];
            for (int r = 0; r < rank; r++) {
                float acc = 0.0f;
                for (int n = 0; n < ds; n++)
                    acc = __fmaf_rn(d_init[(bh * hd + p) * ds + n], kst[(bh * rank + r) * ds + n],
                                    acc);
                const float psi = mimo_x[((long long)h * rank + r) * hd + p];
                gpsi_init[(bh * rank + r) * hd + p] = __fmul_rn(__fmul_rn(e0[bh], v), acc);
                part = __fmaf_rn(__fmul_rn(psi, v), acc, part);
            }
        }
        if (p == tid)
            part0 = part;
        else
            gl[p] = part;
    }
    part0 = mimo_fold_leaves(part0, gl, lanes);
    const float s = mimo_block_sum_n(part0, m3mb_sm, lanes <= nt ? lanes : nt);
    if (threadIdx.x == 0) de0[bh] = s;
}

// The per-token scalars, one thread per (row, head): with scale_t =
// gamma_t + lead_{t+1} (lead = dt lam_c, the window's first one e0; no lead
// at a packed sequence's first token, whose step keeps nothing before it),
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
    long long rows, int seq, int nh, int heads, float a_floor,
    const int* __restrict__ seq_idx
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
    const float dlead = vl_starts(seq_idx, b, seq, t) ? 0.0f
        : t == 0 ? de0[b * nh + h] : dsc3[((row - 1) * nh + h) * 3 + 1];
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
    long long rows, int seq, int nh, int heads, float a_floor,                  \
    const int* __restrict__ seq_idx                                             \
) {                                                                             \
    m3mb_scalars_body<T>(d_proj, gdtb, dsc3, ddt_ang, de0, coef, proj, ld_proj, \
        dt_bias, rows, seq, nh, heads, a_floor, seq_idx);                       \
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
// 2^31. That reduction's `width` threads (a power of two up to 256) each
// sum the rows v, v + width, ... of a column; here a block takes `cw`
// neighbouring columns (8, 16 or 32) and runs those sums for all of them,
// a lane a column, so a warp's loads cover whole sectors of a row instead
// of one float of 32 rows. Grid ceil(dim / cw), block 256, shared memory
// width * cw floats.
extern "C" __global__ void m3mb_reduce_rows(
    float* __restrict__ out, const float* __restrict__ parts,
    long long rows, long long dim, int width, int cw, int accumulate
) {
    extern __shared__ float m3mb_red[];  // [width][cw]
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
            if (v < width) m3mb_red[v * cw + c] = sum[u];
        }
    }
    __syncthreads();
    for (int s = width / 2; s > 0; s >>= 1) {
        for (int v = slot; v < s; v += slots) m3mb_red[v * cw + c] += m3mb_red[(v + s) * cw + c];
        __syncthreads();
    }
    if (slot == 0 && live) out[col] = accumulate ? out[col] + m3mb_red[c] : m3mb_red[c];
}
