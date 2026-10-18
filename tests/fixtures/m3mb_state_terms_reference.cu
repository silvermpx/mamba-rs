// The MIMO chunked backward's state terms as they were released before the
// token tiles took every shape: one block per chunk, head and group of
// `tg` tokens, a token at a time, its keys and inputs in shared memory or,
// with `glob`, in a global scratch, and the chunk's dS staged once for the
// group with `stage`. The token-tile kernel, whole or with its operands in
// slices, is held to it bit for bit. The typed prelude and the packed-row
// helpers precede this fixture; the helpers below are the module's own,
// unchanged.

#define M3M_CHUNK 64

__device__ __forceinline__ float mimo_exp(float x) {
    return exp2f(__fmul_rn(x, LOG2E));
}

__device__ __forceinline__ float2 mimo_rot_pair(float v0, float v1, float s, float c) {
    return make_float2(__fsub_rn(__fmul_rn(c, v0), __fmul_rn(s, v1)),
                       __fadd_rn(__fmul_rn(s, v0), __fmul_rn(c, v1)));
}

__device__ __forceinline__ float m3m_rot(
    const float* src, const float* bias, int n, int half, int na,
    const float* sr, const float* cr
) {
    const float v = __fadd_rn(src[n], bias[n]);
    if (n < na) {
        const float v1 = __fadd_rn(src[n + half], bias[n + half]);
        return mimo_rot_pair(v, v1, sr[n], cr[n]).x;
    }
    if (n >= half && n - half < na) {
        const int a = n - half;
        const float v0 = __fadd_rn(src[a], bias[a]);
        return mimo_rot_pair(v0, v, sr[a], cr[a]).y;
    }
    return v;
}

__device__ __forceinline__ float mimo_block_sum_n(float v, float* red, int n) {
    const int t = threadIdx.x;
    if (t < n) red[t] = v;
    __syncthreads();
    int width = 1;
    while (width < n) width <<= 1;
    for (int w = width >> 1; w > 0; w >>= 1) {
        if (t < w && t + w < n) red[t] = __fadd_rn(red[t], red[t + w]);
        __syncthreads();
    }
    float out = red[0];
    __syncthreads();
    return out;
}

__device__ __forceinline__ float mimo_block_sum(float v, float* red) {
    return mimo_block_sum_n(v, red, blockDim.x);
}

// The inter-chunk part of dq and its decay term, one block per (chunk,
// head, token): dq_t[r][n] = e^Acs_t sum_p dy_t[r][p] S_enter[p][n] (the
// start of dq), and I_t = sum_{r, n} dq_t[r][n] q_t[r][n] by a fixed tree;
// both zero for a token that does not continue the packed sequence the
// entering state holds. Grid (B * nc, H, 64), block 256.

// A block-wide fixed-tree sum; the block need not be a power of two.
__device__ __forceinline__ float m3mb_block_sum(float v, float* red) {
    return mimo_block_sum(v, red);
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
DEFINE_M3MB_STATE_TERMS(m3mb_reference_state_terms, float)
DEFINE_M3MB_STATE_TERMS(m3mb_reference_state_terms_bf16, __nv_bfloat16)
DEFINE_M3MB_STATE_TERMS(m3mb_reference_state_terms_f16, __half)
