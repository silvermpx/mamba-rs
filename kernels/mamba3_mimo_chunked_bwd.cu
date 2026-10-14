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

// The inter-chunk part of dq and its decay term, one block per (chunk,
// head, token): dq_t[r][n] = e^Acs_t sum_p dy_t[r][p] S_enter[p][n] (the
// start of dq), and I_t = sum_{r, n} dq_t[r][n] q_t[r][n] by a fixed tree;
// both zero for a token that does not continue the packed sequence the
// entering state holds. Grid (B * nc, H, 64), block 256.
extern "C" __global__ void m3mb_inter(
    float* __restrict__ dq, float* __restrict__ inter,
    const float* __restrict__ dy, const float* __restrict__ enter,
    const float* __restrict__ cn, const float* __restrict__ c_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    int seq, int nh, int ng, int hd, int ds, int rank, int na,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    __shared__ float red[256];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
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
    const bool cont = t < vl_cont_end(seq_idx, b, seq, t0, lc);
    float part = 0.0f;
    for (int i = threadIdx.x; i < rank * ds; i += blockDim.x) {
        const int r = i / ds, n = i % ds;
        float d = 0.0f;
        if (cont) {
            float acc = 0.0f;
            for (int p = 0; p < hd; p++)
                acc = __fmaf_rn(dy[((row * nh + h) * rank + r) * hd + p],
                                sp[(long long)p * ds + n], acc);
            d = __fmul_rn(acc, ef);
            const float q = m3m_rot(cn + row * rgn + ((long long)r * ng + g) * ds,
                                    c_bias + ((long long)h * rank + r) * ds, n, half, na, sr, cr);
            part = __fmaf_rn(d, q, part);
        }
        dq[((row * nh + h) * rank + r) * ds + n] = d;
    }
    const float s = m3mb_block_sum(part, red);
    if (threadIdx.x == 0) inter[row * nh + h] = cont ? s : 0.0f;
}

// m3mb_inter for a tile of `tt` tokens at once. dq rows (t, r) by state
// index n come out of register tiles over the channels p ascending from 0
// (a warp takes 4 rows, a lane the indices lane + 32 j), then e^Acs_t, as
// m3mb_inter's per-element chains. I_t replays m3mb_inter's 256-thread
// block: virtual thread j sums fma(d, q) over i = j, j + 256, ... (i = r N
// + n), then the halving tree over the 256 partials, every token's tree in
// parallel. Shared memory: the entering state [P][N] padded to a float4
// multiple, so dy's float4 rows start on 16 bytes, dy transposed
// [P][R4 + 4] (R4 = tt R rounded up to 4), d [tt R][N]; the trees reuse
// the first two when they hold tt 256 floats. Grid (B * nc, H,
// ceil(64 / tt)), block 8 R4 threads.
__device__ __forceinline__ void m3mb_inter_tiles_body(
    float* __restrict__ dq, float* __restrict__ inter,
    const float* __restrict__ dy, const float* __restrict__ enter,
    const float* __restrict__ cn, const float* __restrict__ c_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tt,
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
    const int sts = (hd * ds + 3) & ~3;
    float* st = reinterpret_cast<float*>(m3mb_sm4);  // [P][N]
    float* dyt = st + sts;                            // [P][rs]
    float* dd = dyt + hd * rs;                        // [tt R][N]
    // The trees' partials, [tt][256]: over the state and dy once the
    // products are done, past d when those are too small.
    float* red = tt * 256 <= sts + hd * rs ? st : dd + tt * rank * ds;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5, nw = blockDim.x >> 5;
    const int a0 = 4 * warp;
    const long long row0 = b * seq + t0 + T0;
    const float* sp = enter + (bc * nh + h) * (long long)hd * ds;
    // The tokens from `ce` on start or follow a packed sequence the
    // entering state does not hold: their dq starts at zero and I_t is zero.
    const int ce = vl_cont_end(seq_idx, b, seq, t0, lc);
    for (int p = warp; p < hd; p += nw) {
        for (int n = lane; n < ds; n += 32) st[p * ds + n] = sp[(long long)p * ds + n];
    }
    for (int a = warp; a < rs; a += nw) {
        const float* dr = dy + ((row0 + a / rank) * nh + h) * rank * hd + (a % rank) * hd;
        for (int p = lane; p < hd; p += 32) dyt[p * rs + a] = a < rows ? dr[p] : 0.0f;
    }
    __syncthreads();
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    }
    for (int p = 0; p < hd; p++) {
        const float4 d4 = *reinterpret_cast<const float4*>(dyt + p * rs + a0);
        const float dv[4] = {d4.x, d4.y, d4.z, d4.w};
        float sv[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = lane + 32 * j;
            sv[j] = n < ds ? st[p * ds + n] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(dv[i], sv[j], acc[i][j]);
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
            const int n = lane + 32 * j;
            if (n >= ds) continue;
            const float d = cont ? __fmul_rn(acc[i][j], ef) : 0.0f;
            dq[((row * nh + h) * rank + a % rank) * ds + n] = d;
            dd[a * ds + n] = d;
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
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tt,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    m3mb_inter_tiles_body(dq, inter, dy, enter, cn, c_bias, coef, sn, cs, seq, nh, ng, hd, ds,
                          rank, na, tt, y_first, seq_idx);
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

// The state terms of the chunk's tokens, one block per (chunk, head,
// group of `tg` tokens), each token's rotated keys and inputs staged:
//   dk_s[r][n] = c_s sum_p dS_c[p][n] u_s[r][p]   (the start of dk)
//   ste[r][p]  = sum_n dS_c[p][n] k_s[r][n];  du_s[r][p] += c_s ste
//   dc_s       = sum_{r, p} u_s[r][p] ste[r][p]   (a fixed tree)
// for the tokens of the chunk's last packed sequence (every token without a
// packing); an earlier sequence's token reaches no later chunk, so its dk
// starts at zero, its du keeps its value and its dc_s is zero.
// With `stage` the chunk's state gradient dS_c is read into shared memory
// once for the group, rows padded by one, else every token reads it from
// global memory. The token's keys and inputs stage in shared memory, or
// with `glob` in that block's slice of a global scratch [blocks][R N + R P]
// when they do not fit. Grid (B * nc, H, ceil(64 / tg)), block 256 (the
// tree's width), dynamic shared memory R * N + R * P floats without
// `glob`, plus P * (N + 1) with `stage`.
template <typename T>
__device__ __forceinline__ void m3mb_state_terms_body(
    float* __restrict__ dk, float* __restrict__ du, float* __restrict__ dcs,
    const float* __restrict__ dst, const T* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ b_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ mimo_x, float* __restrict__ glob,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tg, int stage,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m3mb_sm[];
    __shared__ float red[256];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
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
    const int sf = vl_seq_first(seq_idx, b, seq, t0, t0 + lc - 1) - t0;
    const long long blk = ((long long)bc * nh + h) * gridDim.z + blockIdx.z;
    float* kt = glob == nullptr ? m3mb_sm : glob + blk * ((long long)rank * (ds + hd));  // [R][N]
    float* ut = kt + rank * ds;         // [R][P]
    float* dss = glob == nullptr ? ut + rank * hd : m3mb_sm;  // [P][N + 1] with stage
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
        if (s < sf) {
            for (int i = threadIdx.x; i < rank * ds; i += blockDim.x)
                dk[(row * nh + h) * rank * ds + i] = 0.0f;
            if (threadIdx.x == 0) dcs[row * nh + h] = 0.0f;
            continue;
        }
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

// m3mb_state_terms for a tile of `tt` tokens at once, rows (s, r):
//   dk[row][n]  = c_s sum_p u[row][p] dS[p][n]   (p ascending from 0)
//   ste[row][p] = sum_n k[row][n] dS[p][n]       (n ascending from 0)
//   du[row][p]  = fma(c_s, ste, du);  dc_s by m3mb_state_terms' 256-thread
//   tree: virtual thread j sums fma(u, ste) over i = j, j + 256, ...
//   (i = r P + p), then the halving tree, every token's tree in parallel.
// The products are register tiles (a warp takes 4 rows, a lane the columns
// lane + 32 j); dS sits as [P][N + 1], which both products read without
// bank conflicts. Shared memory: dS padded to a float4 multiple, u and
// the rotated keys transposed ([P][R4 + 4], [N][R4 + 4], their float4 rows
// on 16 bytes), ste [tt R][P], the trees' [tt][256].
// Grid (B * nc, H, ceil(64 / tt)), block 8 R4 threads.
template <typename T>
__device__ __forceinline__ void m3mb_state_terms_tiles_body(
    float* __restrict__ dk, float* __restrict__ du, float* __restrict__ dcs,
    const float* __restrict__ dst, const T* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ b_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ mimo_x,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tt,
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
    float* dss = reinterpret_cast<float*>(m3mb_sm4);  // [P][N + 1]
    float* ut = dss + ((hd * dld + 3) & ~3);           // [P][rs]
    float* kt = ut + hd * rs;                          // [N][rs]
    float* steb = kt + ds * rs;                        // [tt R][P]
    float* red = steb + tt * rank * hd;                // [tt][256]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5, nw = blockDim.x >> 5;
    const int a0 = 4 * warp;
    const long long row0 = b * seq + t0 + S0;
    const float acs_last = coef[((b * seq + t0 + lc - 1) * nh + h) * 4 + 3];
    const float* dsc = dst + (bc * nh + h) * (long long)hd * ds;
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
    m3m_stage_rot(warp, nw, rows, rs, 0, ds, half, na,
        [&](int a) {
            const int r = a % rank;
            const long long row = row0 + a / rank;
            return M3mRotRow{bn + row * rgn + ((long long)r * ng + g) * ds,
                             b_bias + ((long long)h * rank + r) * ds,
                             sn + (row * nh + h) * na, cs + (row * nh + h) * na};
        },
        [&](int a, int n, float v) { kt[n * rs + a] = v; });
    m3m_stage_u<T>(warp, nw, rows, rs, hd, hd,
        [&](int a) { return proj + (row0 + a / rank) * ld_proj + di + (long long)h * hd; },
        [&](int a) { return mimo_x + ((long long)h * rank + a % rank) * hd; },
        [&](int a, int p, float v) { ut[p * rs + a] = v; });
    __syncthreads();
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
    // dk: rows by state index, p ascending.
    {
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
                const int n = lane + 32 * j;
                sv[j] = n < ds ? dss[p * dld + n] : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(sv[j], uv[i], acc[i][j]);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int a = a0 + i;
            if (a >= rows) continue;
            const long long o = ((row0 + a / rank) * nh + h) * rank + a % rank;
            const bool kept = S0 + a / rank >= sf;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = lane + 32 * j;
                if (n < ds) dk[o * ds + n] = kept ? __fmul_rn(cfs[i], acc[i][j]) : 0.0f;
            }
        }
    }
    // ste: rows by channel, n ascending; du and the trees' operand.
    {
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
                const int p = lane + 32 * j;
                sv[j] = p < hd ? dss[p * dld + n] : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(sv[j], kv[i], acc[i][j]);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int a = a0 + i;
            if (a >= rows) continue;
            const long long o = ((row0 + a / rank) * nh + h) * rank + a % rank;
            const bool kept = S0 + a / rank >= sf;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int p = lane + 32 * j;
                if (p >= hd) continue;
                if (kept) du[o * hd + p] = __fmaf_rn(cfs[i], acc[i][j], du[o * hd + p]);
                steb[a * hd + p] = acc[i][j];
            }
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
            part = __fmaf_rn(ut[p * rs + a], steb[a * hd + p], part);
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
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tt,         \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3mb_state_terms_tiles_body<T>(dk, du, dcs, dst, proj, ld_proj, bn, b_bias, \
        coef, sn, cs, mimo_x, seq, nh, ng, hd, ds, rank, na, tt, y_first, seq_idx); \
}
DEFINE_M3MB_STATE_TERMS_TILES(m3mb_state_terms_tiles, float)
DEFINE_M3MB_STATE_TERMS_TILES(m3mb_state_terms_tiles_bf16, __nv_bfloat16)
DEFINE_M3MB_STATE_TERMS_TILES(m3mb_state_terms_tiles_f16, __half)

#define DEFINE_M3MB_STATE_TERMS(NAME, T)                                       \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dk, float* __restrict__ du, float* __restrict__ dcs,   \
    const float* __restrict__ dst, const T* __restrict__ proj,                 \
    long long ld_proj,                                                         \
    const float* __restrict__ bn, const float* __restrict__ b_bias,            \
    const float* __restrict__ coef, const float* __restrict__ sn,              \
    const float* __restrict__ cs, const float* __restrict__ mimo_x,            \
    float* __restrict__ glob,                                                  \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tg,         \
    int stage,                                                                 \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3mb_state_terms_body<T>(dk, du, dcs, dst, proj, ld_proj, bn, b_bias, coef, \
        sn, cs, mimo_x, glob, seq, nh, ng, hd, ds, rank, na, tg, stage, y_first, \
        seq_idx);                                                              \
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
// padded by one). With `glob` the queries, keys, dy, u and the dq tile
// sit in that block's slice of a global scratch [B * nc * H][tl R
// (3 N + 2 P + 2)] instead, and only G, M and wG in shared memory: the
// products read the same values in the same order. With a packing a pair
// of two sequences is no pair (its dw is zero), and the products run over
// each row's own sequence only (m3m_tile_product_ranged), so another
// sequence's values never enter a chain. Grid (B * nc, H), block 256.
template <typename T>
__device__ __forceinline__ void m3mb_pairs_body(
    float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ du,
    float* __restrict__ dwp,
    const float* __restrict__ dy, const T* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ cn,
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ mimo_x, float* __restrict__ glob,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tl,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m3mb_sm[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
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
    float* gstage = glob == nullptr
        ? nullptr
        : glob + ((long long)bc * nh + h) * ((long long)tr * (3 * ds + 2 * hd + 2));
    float* qs = gstage == nullptr ? m3mb_sm : gstage;  // [tl][R][N]
    float* ks = qs + tr * ds;       // [tl][R][N + 1]
    float* dys = ks + tr * kld;     // [tl][R][P]
    float* us = dys + tr * hd;      // [tl][R][P + 1]
    float* gm = gstage == nullptr ? us + tr * uld : m3mb_sm;  // [tr][tr] G
    float* mm = gm + tr * tr;       // [tr][tr] M, then w M
    float* wg = mm + tr * tr;       // [tr][tr] w G
    float* dqs = gstage == nullptr ? wg + tr * tr : us + tr * uld;  // [tl][R][N]
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
            // G and M per (t r, s r'), zero where s > t or s and t lie in
            // two packed sequences.
            const int ta = tn * rank, sb = sn_ * rank;
            auto live = [&](int a, int bcol) {
                const int t = T0 + a / rank, s = S0 + bcol / rank;
                return s <= t && vl_id(seq_idx, b, seq, t0 + s) == vl_id(seq_idx, b, seq, t0 + t);
            };
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
                const bool apart = vl_id(seq_idx, b, seq, t0 + s) != vl_id(seq_idx, b, seq, t0 + t);
                if (s > t || apart) {
                    for (int r = 0; r < rank; r++)
                        for (int r2 = 0; r2 < rank; r2++) {
                            const int at = (tq * rank + r) * tr + sk * rank + r2;
                            wg[at] = 0.0f;
                            mm[at] = 0.0f;
                        }
                    if (s <= t) dwp[(bc * nh + h) * pairs + (long long)t * (t + 1) / 2 + s] = 0.0f;
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
            auto krow = [&](int bcol) {
                const long long row = b * seq + t0 + S0 + bcol / rank;
                return (row * nh + h) * rank + bcol % rank;
            };
            if (seq_idx == nullptr) {
                m3m_tile_product<4>(
                    ta, ds, sb,
                    [&](int a, int bcol) { return mm[a * tr + bcol]; },
                    [&](int bcol, int n) { return ks[bcol * kld + n]; },
                    [&](int a, int n) { return dqs[a * ds + n]; },
                    [&](int a, int n, float v) { dqs[a * ds + n] = v; });
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
            } else {
                // A query row takes the keys of its sequence up to itself, a
                // key row the queries of its sequence from itself on.
                auto q_range = [&](int a) {
                    const int t = T0 + a / rank;
                    const int f = vl_seq_first(seq_idx, b, seq, t0, t0 + t) - t0;
                    return make_int2(max(f - S0, 0) * rank, min(t - S0 + 1, sn_) * rank);
                };
                auto k_range = [&](int bcol) {
                    const int s = S0 + bcol / rank;
                    const int e = vl_span(seq_idx, b, seq, t0, lc, s).end;
                    return make_int2(max(s - T0, 0) * rank, min(e - T0, tn) * rank);
                };
                m3m_tile_product_ranged<4>(
                    ta, ds,
                    [&](int a, int bcol) { return mm[a * tr + bcol]; },
                    [&](int bcol, int n) { return ks[bcol * kld + n]; },
                    [&](int a, int n) { return dqs[a * ds + n]; },
                    [&](int a, int n, float v) { dqs[a * ds + n] = v; }, q_range);
                m3m_tile_product_ranged<4>(
                    sb, ds,
                    [&](int bcol, int a) { return mm[a * tr + bcol]; },
                    [&](int a, int n) { return qs[a * ds + n]; },
                    [&](int bcol, int n) { return dk[krow(bcol) * ds + n]; },
                    [&](int bcol, int n, float v) { dk[krow(bcol) * ds + n] = v; }, k_range);
                m3m_tile_product_ranged<2>(
                    sb, hd,
                    [&](int bcol, int a) { return wg[a * tr + bcol]; },
                    [&](int a, int p) { return dys[a * hd + p]; },
                    [&](int bcol, int p) { return du[krow(bcol) * hd + p]; },
                    [&](int bcol, int p, float v) { du[krow(bcol) * hd + p] = v; }, k_range);
            }
        }
        __syncthreads();
        for (int i = tid; i < tn * rank * ds; i += blockDim.x) {
            const int tq = i / (rank * ds), rn = i % (rank * ds);
            const long long row = b * seq + t0 + T0 + tq;
            dq[(row * nh + h) * rank * ds + rn] = dqs[i];
        }
    }
}

#define DEFINE_M3MB_PAIRS(NAME, T)                                             \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ du,    \
    float* __restrict__ dwp,                                                   \
    const float* __restrict__ dy, const T* __restrict__ proj,                  \
    long long ld_proj,                                                         \
    const float* __restrict__ bn, const float* __restrict__ cn,                \
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,        \
    const float* __restrict__ coef, const float* __restrict__ sn,              \
    const float* __restrict__ cs, const float* __restrict__ mimo_x,            \
    float* __restrict__ glob,                                                  \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tl,         \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3mb_pairs_body<T>(dq, dk, du, dwp, dy, proj, ld_proj, bn, cn, b_bias,     \
        c_bias, coef, sn, cs, mimo_x, glob, seq, nh, ng, hd, ds, rank, na, tl, y_first, \
        seq_idx);                                                              \
}
DEFINE_M3MB_PAIRS(m3mb_pairs, float)
DEFINE_M3MB_PAIRS(m3mb_pairs_bf16, __nv_bfloat16)
DEFINE_M3MB_PAIRS(m3mb_pairs_f16, __half)

// The pair terms again, split by the output each block owns so nothing is
// read back from global memory per pair and the accumulators stay in
// registers. Both keep m3mb_pairs' tiles of `tl` tokens (tr = tl R rows,
// TR4 = tr rounded up to 4) and its chains, masked zeros included, so the
// bits are its own. A warp takes 4 rows of a product, a lane its columns
// lane + 32 j.
//
// dq, one block per t-tile: dq_t = its entering value, then over the
// s-tiles up to t ascending, col = (s, r') ascending,
// fma(w M, k_s[r'], dq). M = dy . u (p ascending from 0), zero past the
// diagonal. Shared memory: dy [P][TR4 + 4] (transposed), u [tr][P + 1],
// keys [tr][N], padded to a float4 multiple so the float4 rows of
// w M [tr][TR4 + 4] (transposed) start on 16 bytes, w [tl][tl]. With a
// packing a row's chain runs over the keys of its own sequence only, as
// m3mb_pairs runs it. Grid (B * nc, H, ceil(64 / tl)), block 8 TR4 threads.
template <typename T>
__device__ __forceinline__ void m3mb_pairs_dq_body(
    float* __restrict__ dq,
    const float* __restrict__ dy, const T* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ b_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ mimo_x,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tl,
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
    const int T0 = blockIdx.z * tl;
    if (T0 >= lc) return;
    const int tn = min(tl, lc - T0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const int tr = tl * rank, ta = tn * rank;
    const int trs = ((tr + 3) & ~3) + 4;
    const int uld = hd + 1;
    float* dyt = reinterpret_cast<float*>(m3mb_sm4);  // [P][trs]
    float* us = dyt + hd * trs;                        // [tr][uld]
    float* ks = us + tr * uld;                         // [tr][N]
    float* mt = us + ((tr * (uld + ds) + 3) & ~3);     // [tr][trs]
    float* ws = mt + tr * trs;                         // [tl][tl]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5, nw = blockDim.x >> 5;
    const int a0 = 4 * warp;
    const long long trow0 = b * seq + t0 + T0;
    // dy transposed, four of a warp's rows and four channels a lane in
    // flight together.
    for (int q0 = warp; q0 < trs; q0 += 4 * nw) {
        for (int p0 = 0; p0 < hd; p0 += 128) {
            float dv[4][4];
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int a = q0 + u * nw;
                const float* row = dy + ((trow0 + a / rank) * nh + h) * rank * hd + (a % rank) * hd;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int p = p0 + lane + 32 * j;
                    dv[u][j] = a < ta && p < hd ? row[p] : 0.0f;
                }
            }
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int a = q0 + u * nw;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int p = p0 + lane + 32 * j;
                    if (a < trs && p < hd) dyt[p * trs + a] = dv[u][j];
                }
            }
        }
    }
    float acc[4][4];
    // Each row's token and the first token of its packed sequence.
    int trw[4], tfst[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int a = a0 + i;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = lane + 32 * j;
            acc[i][j] = a < ta && n < ds
                ? dq[(((trow0 + a / rank) * nh + h) * rank + a % rank) * ds + n] : 0.0f;
        }
        trw[i] = a < ta ? T0 + a / rank : -1;
        tfst[i] = a < ta && seq_idx != nullptr
            ? vl_seq_first(seq_idx, b, seq, t0, t0 + trw[i]) - t0 : 0;
    }
    for (int S0 = 0; S0 <= T0; S0 += tl) {
        const int sn_ = min(tl, lc - S0);
        const int sb = sn_ * rank;
        const long long srow0 = b * seq + t0 + S0;
        __syncthreads();
        m3m_stage_rot(warp, nw, sb, sb, 0, ds, half, na,
            [&](int col) {
                const int r = col % rank;
                const long long row = srow0 + col / rank;
                return M3mRotRow{bn + row * rgn + ((long long)r * ng + g) * ds,
                                 b_bias + ((long long)h * rank + r) * ds,
                                 sn + (row * nh + h) * na, cs + (row * nh + h) * na};
            },
            [&](int col, int n, float v) { ks[col * ds + n] = v; });
        m3m_stage_u<T>(warp, nw, sb, sb, hd, hd,
            [&](int col) { return proj + (srow0 + col / rank) * ld_proj + di + (long long)h * hd; },
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
        __syncthreads();
        float m[4][2];
#pragma unroll
        for (int i = 0; i < 4; i++) m[i][0] = m[i][1] = 0.0f;
        for (int p = 0; p < hd; p++) {
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
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int a = a0 + i;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int col = lane + 32 * j;
                if (col >= tr) continue;
                float v = 0.0f;
                if (a < ta && col < sb) {
                    const int tq = a / rank, sk = col / rank;
                    if (S0 + sk <= T0 + tq) v = __fmul_rn(ws[tq * tl + sk], m[i][j]);
                }
                mt[col * trs + a] = v;
            }
        }
        __syncthreads();
        if (seq_idx == nullptr) {
            for (int col = 0; col < sb; col++) {
                const float4 m4 = *reinterpret_cast<const float4*>(mt + col * trs + a0);
                const float mv[4] = {m4.x, m4.y, m4.z, m4.w};
                float kv[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int n = lane + 32 * j;
                    kv[j] = n < ds ? ks[col * ds + n] : 0.0f;
                }
#pragma unroll
                for (int i = 0; i < 4; i++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(mv[i], kv[j], acc[i][j]);
                }
            }
        } else {
            // A row takes the keys of its own sequence up to itself.
            int lo[4], hi[4];
            int c0 = sb, c1 = 0;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                lo[i] = trw[i] < 0 ? 0 : max(tfst[i] - S0, 0) * rank;
                hi[i] = trw[i] < 0 ? 0 : min(trw[i] - S0 + 1, sn_) * rank;
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
                    kv[j] = n < ds ? ks[col * ds + n] : 0.0f;
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
            const int n = lane + 32 * j;
            if (n < ds) dq[(((trow0 + a / rank) * nh + h) * rank + a % rank) * ds + n] = acc[i][j];
        }
    }
}

// dk, du and dw, one block per s-tile: over the t-tiles from it ascending,
// G = q . k (n ascending from 0) and M = dy . u (p ascending from 0) give
// dw per pair (r, then r', as m3mb_pairs), then
//   dk_s += (w M) q_t and du_s += (w G) dy_t, a = (t, r) ascending,
// each from its value on entry (the state terms). Shared memory: keys and
// u of the s-tile transposed [N][TR4 + 4] and [P][TR4 + 4]; per t-tile the
// queries [tr][N + 1] and dy [tr][P + 1], padded to a float4 multiple so
// the float4 rows of G, w M, w G [tr][TR4 + 4] start on 16 bytes; w. With
// a packing a pair of two sequences is no pair (its dw is zero) and a row's
// chains run over the queries of its own sequence only, as m3mb_pairs runs
// them. Grid (B * nc, H, ceil(64 / tl)), block 8 TR4 threads.
template <typename T>
__device__ __forceinline__ void m3mb_pairs_dk_body(
    float* __restrict__ dk, float* __restrict__ du, float* __restrict__ dwp,
    const float* __restrict__ dy, const T* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ cn,
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ mimo_x,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tl,
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
    const int S0 = blockIdx.z * tl;
    if (S0 >= lc) return;
    const int sn_ = min(tl, lc - S0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const int tr = tl * rank, sb = sn_ * rank;
    const int trs = ((tr + 3) & ~3) + 4;
    const int qld = ds + 1, dld = hd + 1;
    float* kt = reinterpret_cast<float*>(m3mb_sm4);  // [N][trs]
    float* ut = kt + ds * trs;                        // [P][trs]
    float* qs = ut + hd * trs;                        // [tr][qld]
    float* dys = qs + tr * qld;                       // [tr][dld]
    float* gm = qs + ((tr * (qld + dld) + 3) & ~3);   // [tr][trs]
    float* mm = gm + tr * trs;                        // [tr][trs]
    float* wg = mm + tr * trs;                        // [tr][trs]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5, nw = blockDim.x >> 5;
    const int b0 = 4 * warp;
    const long long srow0 = b * seq + t0 + S0;
    const long long pairs = M3M_CHUNK * (M3M_CHUNK + 1) / 2;
    for (int col = warp; col < sb; col += nw) {
        const int r = col % rank;
        const long long row = srow0 + col / rank;
        const float* src = bn + row * rgn + ((long long)r * ng + g) * ds;
        const float* bias = b_bias + ((long long)h * rank + r) * ds;
        const float* sr_ = sn + (row * nh + h) * na;
        const float* cr_ = cs + (row * nh + h) * na;
#pragma unroll 4
        for (int n = lane; n < ds; n += 32)
            kt[n * trs + col] = m3m_rot(src, bias, n, half, na, sr_, cr_);
        const T* xr = proj + row * ld_proj + di + (long long)h * hd;
        const float* mx = mimo_x + ((long long)h * rank + r) * hd;
        for (int p = lane; p < hd; p += 32) ut[p * trs + col] = __fmul_rn(mx[p], to_f(xr[p]));
    }
    float dka[4][4], dua[4][4];
    // Each row's token and the end of its packed sequence in the chunk.
    int srw[4], send[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int col = b0 + i;
        const long long krow = (srow0 + col / rank) * nh * rank + (long long)h * rank + col % rank;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = lane + 32 * j;
            dka[i][j] = col < sb && n < ds ? dk[krow * ds + n] : 0.0f;
            dua[i][j] = col < sb && n < hd ? du[krow * hd + n] : 0.0f;
        }
        srw[i] = col < sb ? S0 + col / rank : -1;
        send[i] = col < sb && seq_idx != nullptr
            ? vl_span(seq_idx, b, seq, t0, lc, srw[i]).end : lc;
    }
    for (int T0 = S0; T0 < lc; T0 += tl) {
        const int tn = min(tl, lc - T0);
        const int ta = tn * rank;
        const long long trow0 = b * seq + t0 + T0;
        __syncthreads();
        for (int a = warp; a < ta; a += nw) {
            const int r = a % rank;
            const long long row = trow0 + a / rank;
            const float* src = cn + row * rgn + ((long long)r * ng + g) * ds;
            const float* bias = c_bias + ((long long)h * rank + r) * ds;
            const float* sr_ = sn + (row * nh + h) * na;
            const float* cr_ = cs + (row * nh + h) * na;
#pragma unroll 4
            for (int n = lane; n < ds; n += 32)
                qs[a * qld + n] = m3m_rot(src, bias, n, half, na, sr_, cr_);
            const float* dr = dy + (row * nh + h) * rank * hd + r * hd;
            for (int p = lane; p < hd; p += 32) dys[a * dld + p] = dr[p];
        }
        __syncthreads();
        // G and M with rows col (s, r') and columns a = (t, r).
        float ga[4][2], ma[4][2];
#pragma unroll
        for (int i = 0; i < 4; i++) ga[i][0] = ga[i][1] = ma[i][0] = ma[i][1] = 0.0f;
        for (int n = 0; n < ds; n++) {
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
        for (int p = 0; p < hd; p++) {
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
        for (int i = tid; i < tn * sn_; i += blockDim.x) {
            const int tq = i / sn_, sk = i % sn_;
            const int t = T0 + tq, s = S0 + sk;
            const bool apart = vl_id(seq_idx, b, seq, t0 + s) != vl_id(seq_idx, b, seq, t0 + t);
            if (s > t || apart) {
                for (int r = 0; r < rank; r++)
                    for (int r2 = 0; r2 < rank; r2++) {
                        const int at = (tq * rank + r) * trs + sk * rank + r2;
                        wg[at] = 0.0f;
                        mm[at] = 0.0f;
                    }
                if (s <= t) dwp[(bc * nh + h) * pairs + (long long)t * (t + 1) / 2 + s] = 0.0f;
                continue;
            }
            const float* ct = coef + ((trow0 + tq) * nh + h) * 4;
            const float* cs_ = coef + ((srow0 + sk) * nh + h) * 4;
            const float w = s < t
                ? __fmul_rn(mimo_exp(fminf(__fsub_rn(ct[3], cs_[3]), 0.0f)), cs_[2])
                : ct[1];
            float dw = 0.0f;
            for (int r = 0; r < rank; r++) {
                for (int r2 = 0; r2 < rank; r2++) {
                    const int at = (tq * rank + r) * trs + sk * rank + r2;
                    dw = __fmaf_rn(gm[at], mm[at], dw);
                    wg[at] = __fmul_rn(w, gm[at]);
                    mm[at] = __fmul_rn(w, mm[at]);
                }
            }
            dwp[(bc * nh + h) * pairs + (long long)t * (t + 1) / 2 + s] = dw;
        }
        __syncthreads();
        if (seq_idx == nullptr) {
            for (int a = 0; a < ta; a++) {
                const float4 m4 = *reinterpret_cast<const float4*>(mm + a * trs + b0);
                const float4 w4 = *reinterpret_cast<const float4*>(wg + a * trs + b0);
                const float mv[4] = {m4.x, m4.y, m4.z, m4.w};
                const float wv[4] = {w4.x, w4.y, w4.z, w4.w};
                float qv[4], dv[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int n = lane + 32 * j;
                    qv[j] = n < ds ? qs[a * qld + n] : 0.0f;
                    dv[j] = n < hd ? dys[a * dld + n] : 0.0f;
                }
#pragma unroll
                for (int i = 0; i < 4; i++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        dka[i][j] = __fmaf_rn(mv[i], qv[j], dka[i][j]);
                        dua[i][j] = __fmaf_rn(wv[i], dv[j], dua[i][j]);
                    }
                }
            }
        } else {
            // A row takes the queries of its own sequence from itself on.
            int lo[4], hi[4];
            int a_lo = ta, a_hi = 0;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                lo[i] = srw[i] < 0 ? 0 : max(srw[i] - T0, 0) * rank;
                hi[i] = srw[i] < 0 ? 0 : min(send[i] - T0, tn) * rank;
                if (lo[i] < hi[i]) {
                    a_lo = min(a_lo, lo[i]);
                    a_hi = max(a_hi, hi[i]);
                }
            }
            for (int a = a_lo; a < a_hi; a++) {
                const float4 m4 = *reinterpret_cast<const float4*>(mm + a * trs + b0);
                const float4 w4 = *reinterpret_cast<const float4*>(wg + a * trs + b0);
                const float mv[4] = {m4.x, m4.y, m4.z, m4.w};
                const float wv[4] = {w4.x, w4.y, w4.z, w4.w};
                float qv[4], dv[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int n = lane + 32 * j;
                    qv[j] = n < ds ? qs[a * qld + n] : 0.0f;
                    dv[j] = n < hd ? dys[a * dld + n] : 0.0f;
                }
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (a >= lo[i] && a < hi[i]) {
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            dka[i][j] = __fmaf_rn(mv[i], qv[j], dka[i][j]);
                            dua[i][j] = __fmaf_rn(wv[i], dv[j], dua[i][j]);
                        }
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
            const int n = lane + 32 * j;
            if (n < ds) dk[krow * ds + n] = dka[i][j];
            if (n < hd) du[krow * hd + n] = dua[i][j];
        }
    }
}

#define DEFINE_M3MB_PAIRS_DQ(NAME, T)                                          \
extern "C" __global__ void __launch_bounds__(512) NAME(                        \
    float* __restrict__ dq,                                                    \
    const float* __restrict__ dy, const T* __restrict__ proj,                  \
    long long ld_proj,                                                         \
    const float* __restrict__ bn, const float* __restrict__ b_bias,            \
    const float* __restrict__ coef, const float* __restrict__ sn,              \
    const float* __restrict__ cs, const float* __restrict__ mimo_x,            \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tl,         \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3mb_pairs_dq_body<T>(dq, dy, proj, ld_proj, bn, b_bias, coef, sn, cs,     \
        mimo_x, seq, nh, ng, hd, ds, rank, na, tl, y_first, seq_idx);          \
}
DEFINE_M3MB_PAIRS_DQ(m3mb_pairs_dq, float)
DEFINE_M3MB_PAIRS_DQ(m3mb_pairs_dq_bf16, __nv_bfloat16)
DEFINE_M3MB_PAIRS_DQ(m3mb_pairs_dq_f16, __half)

#define DEFINE_M3MB_PAIRS_DK(NAME, T)                                          \
extern "C" __global__ void __launch_bounds__(512) NAME(                        \
    float* __restrict__ dk, float* __restrict__ du, float* __restrict__ dwp,   \
    const float* __restrict__ dy, const T* __restrict__ proj,                  \
    long long ld_proj,                                                         \
    const float* __restrict__ bn, const float* __restrict__ cn,                \
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,        \
    const float* __restrict__ coef, const float* __restrict__ sn,              \
    const float* __restrict__ cs, const float* __restrict__ mimo_x,            \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tl,         \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3mb_pairs_dk_body<T>(dk, du, dwp, dy, proj, ld_proj, bn, cn, b_bias,      \
        c_bias, coef, sn, cs, mimo_x, seq, nh, ng, hd, ds, rank, na, tl, y_first, \
        seq_idx);                                                              \
}
DEFINE_M3MB_PAIRS_DK(m3mb_pairs_dk, float)
DEFINE_M3MB_PAIRS_DK(m3mb_pairs_dk_bf16, __nv_bfloat16)
DEFINE_M3MB_PAIRS_DK(m3mb_pairs_dk_f16, __half)

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
// angle, lane 0 sums.
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
    float* prod = m3mb_sm;  // [na]
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
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
        if (vl_starts(seq_idx, b, seq, t)) {
            for (int k = 0; k < per && k < 8; k++) cum[k] = 0.0f;
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
