// The MIMO chunked backward's inter-chunk term as it was released before
// the token tiles took every shape: one block per (chunk, head, token), each
// dq element a chain over the channels ascending, then e^Acs_t, and the
// decay term I_t by the block's fixed tree. The token-tile kernel, its
// entering state staged a slice of channels at a time, is held to it bit
// for bit. The typed prelude and the packed-row helpers precede this
// fixture; the helpers below are the module's own, unchanged.

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
extern "C" __global__ void m3mb_reference_inter(
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
    const float s = mimo_block_sum(part, red);
    if (threadIdx.x == 0) inter[row * nh + h] = cont ? s : 0.0f;
}
