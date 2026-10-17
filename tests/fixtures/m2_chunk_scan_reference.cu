// Mamba-2's per-output chunk scan as it was released, the kernel the
// shapes whose whole entering state did not fit a tile ran on: a thread an
// output, its chain read from global memory. The sliced tile kernel that
// replaced it is held to it bit for bit. The typed prelude and the
// packed-row helpers precede this fixture.

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif
#define M2_CHUNK 64

__device__ __forceinline__ float m2_exp(float x) {
    return exp2f(__fmul_rn(x, LOG2E));
}

// The outputs: y_t[p] = (sum_n C_t[n] h_prev[p][n]) e^Acs_t
// + sum_{s<=t} W[t][s] x_s[p] + D x_t[p], f32 [B * T, d_ssm]. In a packed
// row the entering state reaches only the tokens that continue its
// sequence, and s runs over t's own sequence. One thread per (t, p). Grid
// (B * nc, H, ceil(64 * P / blockDim.x)).
#define DEFINE_M2_CHUNK_SCAN(NAME, TI)                                         \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ y,                                                     \
    const float* __restrict__ states,                                          \
    const float* __restrict__ cb,                                              \
    const TI* __restrict__ xbc, long long ld_xbc, int c_col,                   \
    const float* __restrict__ dt, const float* __restrict__ acs,               \
    const float* __restrict__ D, int d_per_channel,                            \
    int batch, int seq, int nh, int ng, int hd, int ds,                        \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    const int nc = (seq + M2_CHUNK - 1) / M2_CHUNK;                            \
    const long long bc = blockIdx.x;                                           \
    const int h = y_first + blockIdx.y;                                        \
    const int e = (z_first + blockIdx.z) * blockDim.x + threadIdx.x;           \
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
    /* The tokens before the chunk's first start continue the entering */      \
    /* state's sequence. */                                                    \
    const int cont = vl_lower_bound(seq_idx, b, seq, t0, t0 + lc,              \
                                    vl_id(seq_idx, b, seq, t0 - 1) + 1) - t0;  \
    const float acs_t = acs[sc + t];                                           \
    float acc = 0.0f;                                                          \
    if (t < cont) {                                                            \
        for (int n = 0; n < ds; n++)                                           \
            acc = __fmaf_rn(to_f(xbc[rt + c_col + (long long)g * ds + n]),     \
                            hprev[n], acc);                                    \
        acc = __fmul_rn(acc, m2_exp(acs_t));                                   \
    }                                                                          \
    const long long cbo = (bc * ng + g) * (M2_CHUNK * (M2_CHUNK + 1) / 2)      \
                          + (long long)t * (t + 1) / 2;                        \
    for (int s = vl_seq_first(seq_idx, b, seq, t0, t0 + t) - t0; s <= t; s++) { \
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
    y[(b * seq + t0 + t) * ((long long)nh * hd) + (long long)h * hd + p] = acc; \
}

DEFINE_M2_CHUNK_SCAN(m2_reference_chunk_scan, float)
DEFINE_M2_CHUNK_SCAN(m2_reference_chunk_scan_bf16, __nv_bfloat16)
DEFINE_M2_CHUNK_SCAN(m2_reference_chunk_scan_f16, __half)
