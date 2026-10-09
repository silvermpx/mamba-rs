// Mamba-3 MIMO on the GPU: the B and C row norms, and the scan of every
// head over a window of tokens, decode being the window of one.
//
// The scan runs one block per (batch, head) and walks the window token by
// token, so a decode step and a prefill share one kernel and a decode
// continues a prefill bit for bit. Per token the block discretises,
// advances the head's angles, builds the R keys and queries (the group's
// normalised rows plus the head's bias, rotated in half-split pairs
// (a, a + N/2)), and each channel's thread then runs its state row
// s = alpha s + sum_r (gamma u_r kc_r + beta up_r kp_r), reads the R
// outputs y_r = s . q_r + D u_r, and mixes them through the gate:
// o = sum_r Phi_r y_r silu(Z_r z). The state rows live in shared memory.
//
// Composed after gated_mlp.cu in the MIMO module (the MLP's gate comes
// from there). Offsets are 64-bit.

#include "_typed_prelude.cuh"

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

#define MIMO_MAX_RANK 16

__device__ __forceinline__ float mimo_exp(float x) {
    return exp2f(__fmul_rn(x, LOG2E));
}

__device__ __forceinline__ float mimo_sigmoid(float x) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, mimo_exp(-x)));
}

// The B and C rows, normalised: out[row][i] = (raw * inv_rms) * w over each
// row of d_state, for B and C of every token (grid.y picks which). One
// block per (token, rank-group row); a fixed tree over the block's
// partial sums. Grid (rows * R * G, 2), block 128.
#define DEFINE_MIMO_BC_NORM(NAME, TI)                                          \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ bn, float* __restrict__ cn,                            \
    const TI* __restrict__ proj, long long ld_proj, int b_col, int c_col,     \
    const float* __restrict__ wb, const float* __restrict__ wc,                \
    long long rows, int rg, int ds, float eps                                  \
) {                                                                            \
    __shared__ float red[128];                                                 \
    const long long q = blockIdx.x;                                            \
    if (q >= rows * rg) return;                                                \
    const long long row = q / rg;                                              \
    const int i = (int)(q % rg);                                               \
    const int which = blockIdx.y;                                              \
    const int col = which ? c_col : b_col;                                     \
    const float* w = which ? wc : wb;                                          \
    float* out = which ? cn : bn;                                              \
    const TI* src = proj + row * ld_proj + col + (long long)i * ds;            \
    float acc = 0.0f;                                                          \
    for (int n = threadIdx.x; n < ds; n += blockDim.x) {                       \
        float v = to_f(src[n]);                                                \
        acc = __fmaf_rn(v, v, acc);                                            \
    }                                                                          \
    red[threadIdx.x] = acc;                                                    \
    __syncthreads();                                                           \
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {                             \
        if (threadIdx.x < s)                                                   \
            red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + s]);\
        __syncthreads();                                                       \
    }                                                                          \
    const float inv = __fdiv_rn(1.0f,                                          \
        sqrtf(__fadd_rn(__fdiv_rn(red[0], (float)ds), eps)));                  \
    float* dst = out + (row * rg + i) * ds;                                    \
    for (int n = threadIdx.x; n < ds; n += blockDim.x)                         \
        dst[n] = __fmul_rn(__fmul_rn(to_f(src[n]), inv), w[n]);                \
}

DEFINE_MIMO_BC_NORM(mimo_bc_norm, float)
DEFINE_MIMO_BC_NORM(mimo_bc_norm_bf16, __nv_bfloat16)
DEFINE_MIMO_BC_NORM(mimo_bc_norm_f16, __half)

// The scan of one head over a window. Grid (batch, nheads), block a whole
// number of warps covering headdim. Dynamic shared memory, in floats:
// hd * (ds + 1) for the state rows, 3 * R * ds for the previous keys, the
// keys and the queries, 2 * na for the angles' sines and cosines, and 8
// for the step scalars.
//
// The state (ssm [B, H, P, N], k [B, H, R, N], v [B, H, P], angle
// [B, H, na]) is read at the start and written back at the end. `out` is
// the gated row of each token, [B * T, d_inner].
#define DEFINE_MIMO_SCAN(NAME, TI, FROM_F)                                     \
extern "C" __global__ void NAME(                                               \
    TI* __restrict__ out,                                                      \
    float* __restrict__ ssm, float* __restrict__ kst,                          \
    float* __restrict__ vst, float* __restrict__ ang,                          \
    const TI* __restrict__ proj, long long ld_proj,                            \
    const float* __restrict__ bn, const float* __restrict__ cn,                \
    const float* __restrict__ dt_bias,                                         \
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,        \
    const float* __restrict__ D,                                               \
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,        \
    const float* __restrict__ mimo_o,                                          \
    int seq, int nh, int ng, int hd, int ds, int rank, int na,                 \
    float a_floor                                                              \
) {                                                                            \
    extern __shared__ float mimo_sm[];                                         \
    const int b = blockIdx.x;                                                  \
    const int h = blockIdx.y;                                                  \
    const int tid = threadIdx.x;                                               \
    const int g = h / (nh / ng);                                               \
    const int di = nh * hd;                                                    \
    const int rgn = rank * ng * ds;                                            \
    float* st = mimo_sm;                                                       \
    float* kp = st + hd * (ds + 1);                                            \
    float* kc = kp + rank * ds;                                                \
    float* qv = kc + rank * ds;                                                \
    float* sn = qv + rank * ds;                                                \
    float* cs = sn + na;                                                       \
    float* scal = cs + na;                                                     \
    const long long bh = (long long)b * nh + h;                                \
    for (int p = 0; p < hd; p++)                                               \
        for (int n = tid; n < ds; n += blockDim.x)                             \
            st[p * (ds + 1) + n] = ssm[(bh * hd + p) * ds + n];                \
    for (int i = tid; i < rank * ds; i += blockDim.x)                          \
        kp[i] = kst[bh * rank * ds + i];                                       \
    const bool own = tid < hd;                                                 \
    float v_prev = own ? vst[bh * hd + tid] : 0.0f;                            \
    float px[MIMO_MAX_RANK], pz[MIMO_MAX_RANK], po[MIMO_MAX_RANK];            \
    for (int r = 0; r < rank; r++) {                                           \
        const long long o = ((long long)h * rank + r) * hd + tid;              \
        px[r] = own ? mimo_x[o] : 0.0f;                                        \
        pz[r] = own ? mimo_z[o] : 0.0f;                                        \
        po[r] = own ? mimo_o[o] : 0.0f;                                        \
    }                                                                          \
    const float dval = D[h];                                                   \
    const int heads = 2 * di + 2 * rgn;                                        \
    __syncthreads();                                                           \
    for (int t = 0; t < seq; t++) {                                            \
        const long long row = (long long)b * seq + t;                          \
        const TI* pr = proj + row * ld_proj;                                   \
        if (tid == 0) {                                                        \
            float dd_a = to_f(pr[heads + nh + h]);                             \
            float ht = dd_a >= 0.0f ? __fadd_rn(1.0f, dd_a)                    \
                                    : __fdiv_rn(1.0f, __fsub_rn(1.0f, dd_a));  \
            float a = -ht;                                                     \
            a = a > -a_floor ? -a_floor : a;                                   \
            float spin = __fadd_rn(to_f(pr[heads + h]), dt_bias[h]);           \
            float dt = spin > 20.0f ? spin : log1pf(mimo_exp(spin));           \
            float trap = to_f(pr[heads + 2 * nh + h]);                         \
            float alpha = mimo_exp(__fmul_rn(a, dt));                          \
            float lam = mimo_sigmoid(trap);                                    \
            float lam_c = mimo_sigmoid(-trap);                                 \
            scal[0] = alpha;                                                   \
            scal[1] = __fmul_rn(__fmul_rn(alpha, dt), lam_c);                  \
            scal[2] = __fmul_rn(lam, dt);                                      \
            scal[3] = dt;                                                      \
        }                                                                      \
        __syncthreads();                                                       \
        const float dt = scal[3];                                              \
        for (int a = tid; a < na; a += blockDim.x) {                           \
            float tp = __fmul_rn(tanhf(to_f(pr[heads + 3 * nh + a])),          \
                                 3.14159265358979f);                           \
            const double two_pi = 6.283185307179586;                          \
            double acc = (double)ang[bh * na + a] + (double)__fmul_rn(tp, dt); \
            acc -= two_pi * floor(acc / two_pi);                               \
            float th = (float)acc;                                             \
            ang[bh * na + a] = th;                                             \
            sincosf(th, &sn[a], &cs[a]);                                       \
        }                                                                      \
        for (int i = tid; i < rank * ds; i += blockDim.x) {                    \
            const int r = i / ds, n = i % ds;                                  \
            const long long src = row * rgn + ((long long)r * ng + g) * ds + n;\
            const long long bias = ((long long)h * rank + r) * ds + n;         \
            kc[i] = __fadd_rn(bn[src], b_bias[bias]);                          \
            qv[i] = __fadd_rn(cn[src], c_bias[bias]);                          \
        }                                                                      \
        __syncthreads();                                                       \
        const int half = ds / 2;                                               \
        for (int i = tid; i < rank * na; i += blockDim.x) {                    \
            const int r = i / na, a = i % na;                                  \
            const float s = sn[a], c = cs[a];                                  \
            float* kr = kc + r * ds;                                           \
            float* qr = qv + r * ds;                                           \
            float k0 = kr[a], k1 = kr[a + half];                               \
            kr[a] = __fsub_rn(__fmul_rn(c, k0), __fmul_rn(s, k1));             \
            kr[a + half] = __fadd_rn(__fmul_rn(s, k0), __fmul_rn(c, k1));      \
            float q0 = qr[a], q1 = qr[a + half];                               \
            qr[a] = __fsub_rn(__fmul_rn(c, q0), __fmul_rn(s, q1));             \
            qr[a + half] = __fadd_rn(__fmul_rn(s, q0), __fmul_rn(c, q1));      \
        }                                                                      \
        __syncthreads();                                                       \
        if (own) {                                                             \
            const float alpha = scal[0], beta = scal[1], gamma = scal[2];      \
            const float x = to_f(pr[di + h * hd + tid]);                       \
            const float z = to_f(pr[h * hd + tid]);                            \
            float u[MIMO_MAX_RANK], bv[MIMO_MAX_RANK], gx[MIMO_MAX_RANK];     \
            float y[MIMO_MAX_RANK];                                            \
            for (int r = 0; r < rank; r++) {                                   \
                u[r] = __fmul_rn(px[r], x);                                    \
                gx[r] = __fmul_rn(gamma, u[r]);                                \
                bv[r] = __fmul_rn(beta, __fmul_rn(px[r], v_prev));             \
                y[r] = 0.0f;                                                   \
            }                                                                  \
            float* srow = st + tid * (ds + 1);                                 \
            for (int n = 0; n < ds; n++) {                                     \
                float inner = __fmul_rn(gx[0], kc[n]);                         \
                inner = __fmaf_rn(bv[0], kp[n], inner);                        \
                for (int r = 1; r < rank; r++) {                               \
                    inner = __fmaf_rn(gx[r], kc[r * ds + n], inner);           \
                    inner = __fmaf_rn(bv[r], kp[r * ds + n], inner);           \
                }                                                              \
                float s = __fmaf_rn(alpha, srow[n], inner);                    \
                srow[n] = s;                                                   \
                for (int r = 0; r < rank; r++)                                 \
                    y[r] = __fmaf_rn(s, qv[r * ds + n], y[r]);                 \
            }                                                                  \
            float o = 0.0f;                                                    \
            for (int r = 0; r < rank; r++) {                                   \
                float yr = __fmaf_rn(dval, u[r], y[r]);                        \
                float v = __fmul_rn(pz[r], z);                                 \
                float sg = __fdiv_rn(v, __fadd_rn(1.0f, mimo_exp(-v)));        \
                float term = __fmul_rn(__fmul_rn(yr, sg), po[r]);              \
                o = r == 0 ? term : __fadd_rn(o, term);                        \
            }                                                                  \
            out[row * di + h * hd + tid] = FROM_F(o);                          \
            v_prev = x;                                                        \
        }                                                                      \
        __syncthreads();                                                       \
        for (int i = tid; i < rank * ds; i += blockDim.x) kp[i] = kc[i];       \
        __syncthreads();                                                       \
    }                                                                          \
    for (int p = 0; p < hd; p++)                                               \
        for (int n = tid; n < ds; n += blockDim.x)                             \
            ssm[(bh * hd + p) * ds + n] = st[p * (ds + 1) + n];                \
    for (int i = tid; i < rank * ds; i += blockDim.x)                          \
        kst[bh * rank * ds + i] = kp[i];                                       \
    if (own) vst[bh * hd + tid] = v_prev;                                      \
}

DEFINE_MIMO_SCAN(mimo_scan, float, from_f_f32)
DEFINE_MIMO_SCAN(mimo_scan_bf16, __nv_bfloat16, from_f_bf16)
DEFINE_MIMO_SCAN(mimo_scan_f16, __half, from_f_f16)
