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

// A body templated on RK runs at rank RK when RK > 0, its rank arrays then
// unrolled into registers; RK = 0 runs at the runtime rank, its arrays in
// local memory. The arithmetic is the same either way.
#define MIMO_R(RK, rank) ((RK) > 0 ? (RK) : (rank))
#define MIMO_RA(RK) ((RK) > 0 ? (RK) : MIMO_MAX_RANK)

// Run the statement with `RK` bound to the runtime rank when it is one of
// the compiled ranks, else to 0.
#define MIMO_RANK_DISPATCH(rank, ...)                                          \
    switch (rank) {                                                            \
    case 1: { constexpr int RK = 1; __VA_ARGS__; } break;                      \
    case 2: { constexpr int RK = 2; __VA_ARGS__; } break;                      \
    case 4: { constexpr int RK = 4; __VA_ARGS__; } break;                      \
    case 8: { constexpr int RK = 8; __VA_ARGS__; } break;                      \
    default: { constexpr int RK = 0; __VA_ARGS__; } break;                     \
    }

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

// The pieces of one token, shared by the scan and by the training
// backward's replay so both compute the same bits. Every thread of the
// block calls each piece; the caller places the barriers.

// The step scalars of head h, by thread 0: alpha, beta = alpha dt lam_c,
// gamma = lam dt, dt, then A, lam and lam_c for the backward.
template <typename TI>
__device__ __forceinline__ void mimo_tok_disc(
    float* scal, const TI* pr, int heads, int nh, int h,
    const float* dt_bias, float a_floor
) {
    if (threadIdx.x != 0) return;
    float dd_a = to_f(pr[heads + nh + h]);
    float ht = dd_a >= 0.0f ? __fadd_rn(1.0f, dd_a)
                            : __fdiv_rn(1.0f, __fsub_rn(1.0f, dd_a));
    float a = -ht;
    a = a > -a_floor ? -a_floor : a;
    float spin = __fadd_rn(to_f(pr[heads + h]), dt_bias[h]);
    float dt = spin > 20.0f ? spin : log1pf(mimo_exp(spin));
    float trap = to_f(pr[heads + 2 * nh + h]);
    float alpha = mimo_exp(__fmul_rn(a, dt));
    float lam = mimo_sigmoid(trap);
    float lam_c = mimo_sigmoid(-trap);
    scal[0] = alpha;
    scal[1] = __fmul_rn(__fmul_rn(alpha, dt), lam_c);
    scal[2] = __fmul_rn(lam, dt);
    scal[3] = dt;
    scal[4] = a;
    scal[5] = lam;
    scal[6] = lam_c;
}

// The token's angle increment tanh(raw) pi for angle a.
template <typename TI>
__device__ __forceinline__ float mimo_tanh_pi(const TI* pr, int ang_col, int a) {
    return __fmul_rn(tanhf(to_f(pr[ang_col + a])), 3.14159265358979f);
}

// One angle step: theta + inc summed in double and wrapped into [0, 2 pi)
// as acc - 2 pi floor(acc / 2 pi), rounded to float. The division only
// matters off the common case. A float sum in (0, 6.28) means the double sum
// lies there too, so the floor is 0 and the double sum rounds to that same
// float (two floats summed through a double round once: 53 >= 2 * 24 + 2).
// A float sum in (-6.28, 0) means the floor is -1, which leaves the single
// wrap add. Zero (whose sign the full formula settles), wider sums and
// non-finite values take the full formula.
__device__ __forceinline__ float mimo_wrap_add(float theta, float inc) {
    const float s = __fadd_rn(theta, inc);
    if (s > 0.0f && s < 6.28f) return s;
    const double two_pi = 6.283185307179586;
    double acc = (double)theta + (double)inc;
    if (s < 0.0f && s > -6.28f) return (float)(acc + two_pi);
    acc -= two_pi * floor(acc / two_pi);
    return (float)acc;
}

// Advance the head's angles `theta` by tanh_pi dt (wrapped into [0, 2 pi))
// and take their sines and cosines.
template <typename TI>
__device__ __forceinline__ void mimo_tok_angles(
    float* sn, float* cs, float* theta, const TI* pr, int ang_col, float dt, int na
) {
    for (int a = threadIdx.x; a < na; a += blockDim.x) {
        float tp = mimo_tanh_pi(pr, ang_col, a);
        float th = mimo_wrap_add(theta[a], __fmul_rn(tp, dt));
        theta[a] = th;
        sincosf(th, &sn[a], &cs[a]);
    }
}

// The keys and queries before rotation: the group's normalised rows of
// this token plus the head's biases.
__device__ __forceinline__ void mimo_tok_keys(
    float* kc, float* qv, const float* bn_row, const float* cn_row,
    const float* b_bias, const float* c_bias, int h, int g, int ng, int rank, int ds
) {
    for (int i = threadIdx.x; i < rank * ds; i += blockDim.x) {
        const int r = i / ds, n = i % ds;
        const long long src = ((long long)r * ng + g) * ds + n;
        const long long bias = ((long long)h * rank + r) * ds + n;
        kc[i] = __fadd_rn(bn_row[src], b_bias[bias]);
        qv[i] = __fadd_rn(cn_row[src], c_bias[bias]);
    }
}

// The half-split rotation of the keys and queries, pairs (a, a + N/2).
__device__ __forceinline__ void mimo_tok_rotate(
    float* kc, float* qv, const float* sn, const float* cs, int rank, int ds, int na
) {
    const int half = ds / 2;
    for (int i = threadIdx.x; i < rank * na; i += blockDim.x) {
        const int r = i / na, a = i % na;
        const float s = sn[a], c = cs[a];
        float* kr = kc + r * ds;
        float* qr = qv + r * ds;
        float k0 = kr[a], k1 = kr[a + half];
        kr[a] = __fsub_rn(__fmul_rn(c, k0), __fmul_rn(s, k1));
        kr[a + half] = __fadd_rn(__fmul_rn(s, k0), __fmul_rn(c, k1));
        float q0 = qr[a], q1 = qr[a + half];
        qr[a] = __fsub_rn(__fmul_rn(c, q0), __fmul_rn(s, q1));
        qr[a + half] = __fadd_rn(__fmul_rn(s, q0), __fmul_rn(c, q1));
    }
}

// One channel's row: u_r = Psi_r x, up_r = Psi_r x_prev, the state row
// s = alpha s + sum_r (gamma u_r kc_r + beta up_r kp_r), and the readouts
// y_r = s . q_r + D u_r.
template <int RK>
__device__ __forceinline__ void mimo_tok_row(
    float* srow, float* y, float* u, float* up, const float* px, float x, float v_prev,
    const float* kc, const float* kp, const float* qv,
    float alpha, float beta, float gamma, float dval, int rank, int ds
) {
    const int R = MIMO_R(RK, rank);
    float bv[MIMO_RA(RK)], gx[MIMO_RA(RK)];
#pragma unroll
    for (int r = 0; r < R; r++) {
        u[r] = __fmul_rn(px[r], x);
        gx[r] = __fmul_rn(gamma, u[r]);
        up[r] = __fmul_rn(px[r], v_prev);
        bv[r] = __fmul_rn(beta, up[r]);
        y[r] = 0.0f;
    }
    for (int n = 0; n < ds; n++) {
        float inner = __fmul_rn(gx[0], kc[n]);
        inner = __fmaf_rn(bv[0], kp[n], inner);
#pragma unroll
        for (int r = 1; r < R; r++) {
            inner = __fmaf_rn(gx[r], kc[r * ds + n], inner);
            inner = __fmaf_rn(bv[r], kp[r * ds + n], inner);
        }
        float s = __fmaf_rn(alpha, srow[n], inner);
        srow[n] = s;
#pragma unroll
        for (int r = 0; r < R; r++) y[r] = __fmaf_rn(s, qv[r * ds + n], y[r]);
    }
#pragma unroll
    for (int r = 0; r < R; r++) y[r] = __fmaf_rn(dval, u[r], y[r]);
}

// The gate: o = sum_r (y_r silu(Z_r z)) Phi_r.
template <int RK>
__device__ __forceinline__ float mimo_tok_gate(
    const float* y, const float* pz, const float* po, float z, int rank
) {
    float o = 0.0f;
#pragma unroll
    for (int r = 0; r < MIMO_R(RK, rank); r++) {
        float v = __fmul_rn(pz[r], z);
        float sg = __fdiv_rn(v, __fadd_rn(1.0f, mimo_exp(-v)));
        float term = __fmul_rn(__fmul_rn(y[r], sg), po[r]);
        o = r == 0 ? term : __fadd_rn(o, term);
    }
    return o;
}

__device__ __forceinline__ void mimo_put(float* p, float v) { *p = v; }
__device__ __forceinline__ void mimo_put(__nv_bfloat16* p, float v) { *p = from_f_bf16(v); }
__device__ __forceinline__ void mimo_put(__half* p, float v) { *p = from_f_f16(v); }

// A head's [P, N] state between dense global rows and shared rows padded
// to N + 1, every thread of the block taking float4 slices of the dense
// rows so a block moves the whole state in a few loads per thread.
__device__ __forceinline__ bool mimo_state_vec(const float* gl, int ds) {
    return (ds & 3) == 0 && (reinterpret_cast<unsigned long long>(gl) & 15) == 0;
}

__device__ __forceinline__ void mimo_state_load(
    float* __restrict__ sh, const float* __restrict__ gl, int hd, int ds
) {
    const int total = hd * ds;
    if (mimo_state_vec(gl, ds)) {
#pragma unroll 4
        for (int i = threadIdx.x * 4; i < total; i += blockDim.x * 4) {
            const int p = i / ds, n = i - p * ds;
            const float4 v = *reinterpret_cast<const float4*>(gl + i);
            float* d = sh + p * (ds + 1) + n;
            d[0] = v.x;
            d[1] = v.y;
            d[2] = v.z;
            d[3] = v.w;
        }
    } else {
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int p = i / ds;
            sh[p * (ds + 1) + i - p * ds] = gl[i];
        }
    }
}

__device__ __forceinline__ void mimo_state_store(
    float* __restrict__ gl, const float* __restrict__ sh, int hd, int ds
) {
    const int total = hd * ds;
    if (mimo_state_vec(gl, ds)) {
#pragma unroll 4
        for (int i = threadIdx.x * 4; i < total; i += blockDim.x * 4) {
            const int p = i / ds, n = i - p * ds;
            const float* d = sh + p * (ds + 1) + n;
            *reinterpret_cast<float4*>(gl + i) = make_float4(d[0], d[1], d[2], d[3]);
        }
    } else {
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int p = i / ds;
            gl[i] = sh[p * (ds + 1) + i - p * ds];
        }
    }
}

// The scan of one head over a window. Grid (batch, nheads), block a whole
// number of warps covering headdim (the threads past it help move the
// state and build the keys). Dynamic shared memory, in floats:
// hd * (ds + 1) for the state rows, 3 * R * ds for the previous keys, the
// keys and the queries, 2 * na for the angles' sines and cosines, and 8
// for the step scalars.
//
// The state (ssm [B, H, P, N], k [B, H, R, N], v [B, H, P], angle
// [B, H, na]) is read at the start and written back at the end. `out` is
// the gated row of each token, [B * T, d_inner]. With CKPT the state
// before tokens 0, stride, 2 stride, ... is also written to the
// checkpoints ([B, S, H, ...], S = ceil(seq / stride)), what the training
// backward replays from.
template <typename TI, bool CKPT, int RK>
__device__ __forceinline__ void mimo_scan_body(
    TI* __restrict__ out,
    float* __restrict__ ssm, float* __restrict__ kst,
    float* __restrict__ vst, float* __restrict__ ang,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ cn,
    const float* __restrict__ dt_bias,
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,
    const float* __restrict__ D,
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,
    const float* __restrict__ mimo_o,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, float a_floor,
    float* __restrict__ ck_ssm, float* __restrict__ ck_k,
    float* __restrict__ ck_v, float* __restrict__ ck_ang, int stride
) {
    extern __shared__ float mimo_sm[];
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    float* st = mimo_sm;
    float* kp = st + hd * (ds + 1);
    float* kc = kp + rank * ds;
    float* qv = kc + rank * ds;
    float* sn = qv + rank * ds;
    float* cs = sn + na;
    float* scal = cs + na;
    const long long bh = (long long)b * nh + h;
    mimo_state_load(st, ssm + bh * hd * ds, hd, ds);
    for (int i = tid; i < rank * ds; i += blockDim.x)
        kp[i] = kst[bh * rank * ds + i];
    const bool own = tid < hd;
    float v_prev = own ? vst[bh * hd + tid] : 0.0f;
    const int R = MIMO_R(RK, rank);
    float px[MIMO_RA(RK)], pz[MIMO_RA(RK)], po[MIMO_RA(RK)];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const long long o = ((long long)h * rank + r) * hd + tid;
        px[r] = own ? mimo_x[o] : 0.0f;
        pz[r] = own ? mimo_z[o] : 0.0f;
        po[r] = own ? mimo_o[o] : 0.0f;
    }
    const float dval = D[h];
    const int heads = 2 * di + 2 * rgn;
    const int nseg = CKPT ? (seq + stride - 1) / stride : 0;
    __syncthreads();
    for (int t = 0; t < seq; t++) {
        const long long row = (long long)b * seq + t;
        const TI* pr = proj + row * ld_proj;
        if (CKPT && t % stride == 0) {
            const long long q = ((long long)b * nseg + t / stride) * nh + h;
            mimo_state_store(ck_ssm + q * hd * ds, st, hd, ds);
            for (int i = tid; i < rank * ds; i += blockDim.x) ck_k[q * rank * ds + i] = kp[i];
            if (own) ck_v[q * hd + tid] = v_prev;
            for (int a = tid; a < na; a += blockDim.x) ck_ang[q * na + a] = ang[bh * na + a];
        }
        mimo_tok_disc(scal, pr, heads, nh, h, dt_bias, a_floor);
        __syncthreads();
        mimo_tok_angles(sn, cs, ang + bh * na, pr, heads + 3 * nh, scal[3], na);
        mimo_tok_keys(kc, qv, bn + row * rgn, cn + row * rgn, b_bias, c_bias, h, g, ng, rank, ds);
        __syncthreads();
        mimo_tok_rotate(kc, qv, sn, cs, rank, ds, na);
        __syncthreads();
        if (own) {
            const float x = to_f(pr[di + h * hd + tid]);
            const float z = to_f(pr[h * hd + tid]);
            float u[MIMO_RA(RK)], up[MIMO_RA(RK)], y[MIMO_RA(RK)];
            mimo_tok_row<RK>(st + tid * (ds + 1), y, u, up, px, x, v_prev, kc, kp, qv,
                             scal[0], scal[1], scal[2], dval, rank, ds);
            mimo_put(&out[row * di + h * hd + tid], mimo_tok_gate<RK>(y, pz, po, z, rank));
            v_prev = x;
        }
        __syncthreads();
        for (int i = tid; i < rank * ds; i += blockDim.x) kp[i] = kc[i];
        __syncthreads();
    }
    mimo_state_store(ssm + bh * hd * ds, st, hd, ds);
    for (int i = tid; i < rank * ds; i += blockDim.x)
        kst[bh * rank * ds + i] = kp[i];
    if (own) vst[bh * hd + tid] = v_prev;
}

#define DEFINE_MIMO_SCAN(NAME, TI)                                             \
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
    MIMO_RANK_DISPATCH(rank, mimo_scan_body<TI, false, RK>(out, ssm, kst, vst, \
        ang, proj, ld_proj, bn, cn, dt_bias, b_bias, c_bias, D, mimo_x,        \
        mimo_z, mimo_o, seq, nh, ng, hd, ds, rank, na, a_floor, nullptr,       \
        nullptr, nullptr, nullptr, 1))                                         \
}

DEFINE_MIMO_SCAN(mimo_scan, float)
DEFINE_MIMO_SCAN(mimo_scan_bf16, __nv_bfloat16)
DEFINE_MIMO_SCAN(mimo_scan_f16, __half)

// The training forward's scan: the scan above, also writing the
// checkpoints every `stride` tokens.
#define DEFINE_MIMO_SCAN_CKPT(NAME, TI)                                        \
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
    float a_floor,                                                             \
    float* __restrict__ ck_ssm, float* __restrict__ ck_k,                      \
    float* __restrict__ ck_v, float* __restrict__ ck_ang, int stride           \
) {                                                                            \
    MIMO_RANK_DISPATCH(rank, mimo_scan_body<TI, true, RK>(out, ssm, kst, vst,  \
        ang, proj, ld_proj, bn, cn, dt_bias, b_bias, c_bias, D, mimo_x,        \
        mimo_z, mimo_o, seq, nh, ng, hd, ds, rank, na, a_floor, ck_ssm, ck_k,  \
        ck_v, ck_ang, stride))                                                 \
}

DEFINE_MIMO_SCAN_CKPT(mimo_scan_ckpt, float)
DEFINE_MIMO_SCAN_CKPT(mimo_scan_ckpt_bf16, __nv_bfloat16)
DEFINE_MIMO_SCAN_CKPT(mimo_scan_ckpt_f16, __half)
