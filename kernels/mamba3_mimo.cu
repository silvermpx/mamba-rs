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

// The state indices a batch of a channel's row loop takes: four at ranks
// up to 4, two at rank 8, so a batch's rank arrays stay in registers; one
// (no batching) at a runtime rank.
#define MIMO_ROW_BATCH(RK) ((RK) == 0 ? 1 : ((RK) <= 4 ? 4 : 2))

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
// backward's replay so both compute the same bits.

// The step scalars of head h at one token: alpha = e^(A dt), beta =
// alpha dt lam_c, gamma = lam dt, and dt, A, lam and lam_c for the
// backward.
struct MimoStep {
    float alpha, beta, gamma, dt, a, lam, lam_c;
};

template <typename TI>
__device__ __forceinline__ MimoStep mimo_step_scalars(
    const TI* pr, int heads, int nh, int h, float dt_bias, float a_floor
) {
    MimoStep st;
    float dd_a = to_f(pr[heads + nh + h]);
    float ht = dd_a >= 0.0f ? __fadd_rn(1.0f, dd_a)
                            : __fdiv_rn(1.0f, __fsub_rn(1.0f, dd_a));
    st.a = -ht;
    st.a = st.a > -a_floor ? -a_floor : st.a;
    const float spin = __fadd_rn(to_f(pr[heads + h]), dt_bias);
    st.dt = spin > 20.0f ? spin : log1pf(mimo_exp(spin));
    const float trap = to_f(pr[heads + 2 * nh + h]);
    st.alpha = mimo_exp(__fmul_rn(st.a, st.dt));
    st.lam = mimo_sigmoid(trap);
    st.lam_c = mimo_sigmoid(-trap);
    st.beta = __fmul_rn(__fmul_rn(st.alpha, st.dt), st.lam_c);
    st.gamma = __fmul_rn(st.lam, st.dt);
    return st;
}

// A key or query pair (a, a + N/2) turned by the angle's sine and cosine.
__device__ __forceinline__ float2 mimo_rot_pair(float v0, float v1, float s, float c) {
    return make_float2(__fsub_rn(__fmul_rn(c, v0), __fmul_rn(s, v1)),
                       __fadd_rn(__fmul_rn(s, v0), __fmul_rn(c, v1)));
}

// The token's angle increment tanh(raw) pi for angle a.
template <typename TI>
__device__ __forceinline__ float mimo_tanh_pi(const TI* pr, int ang_col, int a) {
    return __fmul_rn(tanhf(to_f(pr[ang_col + a])), 3.14159265358979f);
}

// The rare steps of mimo_wrap_add, out of line so the common step stays a
// float add and two compares rather than a select over the double division.
__device__ __noinline__ float mimo_wrap_slow(float theta, float inc, float s) {
    const double two_pi = 6.283185307179586;
    double acc = (double)theta + (double)inc;
    if (s < 0.0f && s > -6.28f) return (float)(acc + two_pi);
    acc -= two_pi * floor(acc / two_pi);
    return (float)acc;
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
    return mimo_wrap_slow(theta, inc, s);
}

// Whether three rank-row buffers of N-wide rows start on 16 bytes and keep
// every row there, so four neighbouring state indices load as one float4.
__device__ __forceinline__ bool mimo_rows_aligned(
    const float* a, const float* b, const float* c, int ds
) {
    const unsigned long long bits = reinterpret_cast<unsigned long long>(a)
        | reinterpret_cast<unsigned long long>(b) | reinterpret_cast<unsigned long long>(c);
    return (ds & 3) == 0 && (bits & 15) == 0;
}

// The four values at src (16-byte aligned) into column r of rows 0..3 of a
// batch, the J-by-rank arrays of the row loops (J is 4 wherever it runs).
template <int J, int RA>
__device__ __forceinline__ void mimo_split4(float (&dst)[J][RA], int r, const float* src) {
    const float4 v = *reinterpret_cast<const float4*>(src);
    const float w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < J; j++) dst[j][r] = w[j];
}

// One channel's row: u_r = Psi_r x, up_r = Psi_r x_prev, the state row
// s = alpha s + sum_r (gamma u_r kc_r + beta up_r kp_r), and the readouts
// y_r = s . q_r + D u_r. At a compiled rank the state indices go in
// batches, every load of a batch issued before its chains run: the row's
// own store would otherwise hold the next index's loads back a round trip.
// The keys and queries of a batch of four, the same for every channel of a
// warp, come as one float4 per rank when their rows are 16-byte aligned.
// The chains are the same, n ascending.
template <int RK>
__device__ __forceinline__ void mimo_tok_row(
    float* __restrict__ srow, float* y, float* u, float* up, const float* px, float x,
    float v_prev, const float* __restrict__ kc, const float* __restrict__ kp,
    const float* __restrict__ qv,
    float alpha, float beta, float gamma, float dval, int rank, int ds
) {
    constexpr int J = MIMO_ROW_BATCH(RK);
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
    const bool vec = J == 4 && mimo_rows_aligned(kc, kp, qv, ds);
    int n = 0;
    for (; J > 1 && n + J <= ds; n += J) {
        float sv[J], kcv[J][MIMO_RA(RK)], kpv[J][MIMO_RA(RK)], qvv[J][MIMO_RA(RK)];
#pragma unroll
        for (int j = 0; j < J; j++) sv[j] = srow[n + j];
        if (vec) {
#pragma unroll
            for (int r = 0; r < R; r++) {
                mimo_split4(kcv, r, kc + r * ds + n);
                mimo_split4(kpv, r, kp + r * ds + n);
                mimo_split4(qvv, r, qv + r * ds + n);
            }
        } else {
#pragma unroll
            for (int j = 0; j < J; j++) {
#pragma unroll
                for (int r = 0; r < R; r++) {
                    kcv[j][r] = kc[r * ds + n + j];
                    kpv[j][r] = kp[r * ds + n + j];
                    qvv[j][r] = qv[r * ds + n + j];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < J; j++) {
            float inner = __fmul_rn(gx[0], kcv[j][0]);
            inner = __fmaf_rn(bv[0], kpv[j][0], inner);
#pragma unroll
            for (int r = 1; r < R; r++) {
                inner = __fmaf_rn(gx[r], kcv[j][r], inner);
                inner = __fmaf_rn(bv[r], kpv[j][r], inner);
            }
            const float s = __fmaf_rn(alpha, sv[j], inner);
            srow[n + j] = s;
#pragma unroll
            for (int r = 0; r < R; r++) y[r] = __fmaf_rn(s, qvv[j][r], y[r]);
        }
    }
    for (; n < ds; n++) {
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
// number of warps covering headdim and at least the angles (the threads
// past headdim help move the state and build the keys). Dynamic shared
// memory, in floats: hd * (ds + 1) for the state rows, 3 * R * ds for the
// keys of the previous and the current token and the queries.
//
// A token costs two barriers. Every thread computes the step scalars
// (mimo_step_scalars); thread a < na keeps angle a in a register, advances
// it (mimo_wrap_add) and turns the key and query pairs (a, a + N/2) of
// every rank from their biased values itself (mimo_rot_pair), the other
// elements are only biased; the keys of the previous and the current token
// swap buffers instead of being copied. The training backward's replay
// runs the same pieces, so both compute the same bits.
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
    float* kp = st + ((hd * (ds + 1) + 3) & ~3);  // the keys start on 16 bytes
    float* kc = kp + rank * ds;
    float* qv = kc + rank * ds;
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
    const int ang_col = heads + 3 * nh;
    const int half = ds / 2;
    const int nseg = CKPT ? (seq + stride - 1) / stride : 0;
    const bool angle_owner = tid < na;
    float theta = angle_owner ? ang[bh * na + tid] : 0.0f;
    const float dtb = dt_bias[h];
    float* kprev = kp;
    float* kcur = kc;
    __syncthreads();
    for (int t = 0; t < seq; t++) {
        const long long row = (long long)b * seq + t;
        const TI* pr = proj + row * ld_proj;
        if (CKPT && t % stride == 0) {
            const long long q = ((long long)b * nseg + t / stride) * nh + h;
            mimo_state_store(ck_ssm + q * hd * ds, st, hd, ds);
            for (int i = tid; i < rank * ds; i += blockDim.x) ck_k[q * rank * ds + i] = kprev[i];
            if (own) ck_v[q * hd + tid] = v_prev;
            if (angle_owner) ck_ang[q * na + tid] = theta;
        }
        const MimoStep sc = mimo_step_scalars(pr, heads, nh, h, dtb, a_floor);
        const float dt = sc.dt;
        const float* bnr = bn + row * rgn;
        const float* cnr = cn + row * rgn;
        if (angle_owner) {
            const int a = tid;
            theta = mimo_wrap_add(theta, __fmul_rn(mimo_tanh_pi(pr, ang_col, a), dt));
            float sa, ca;
            sincosf(theta, &sa, &ca);
            for (int r = 0; r < rank; r++) {
                const long long src = ((long long)r * ng + g) * ds;
                const long long bias = ((long long)h * rank + r) * ds;
                const float k0 = __fadd_rn(bnr[src + a], b_bias[bias + a]);
                const float k1 = __fadd_rn(bnr[src + a + half], b_bias[bias + a + half]);
                const float2 kr = mimo_rot_pair(k0, k1, sa, ca);
                kcur[r * ds + a] = kr.x;
                kcur[r * ds + a + half] = kr.y;
                const float q0 = __fadd_rn(cnr[src + a], c_bias[bias + a]);
                const float q1 = __fadd_rn(cnr[src + a + half], c_bias[bias + a + half]);
                const float2 qr = mimo_rot_pair(q0, q1, sa, ca);
                qv[r * ds + a] = qr.x;
                qv[r * ds + a + half] = qr.y;
            }
        }
        for (int i = tid; i < rank * ds; i += blockDim.x) {
            const int r = i / ds, n = i - r * ds;
            if (n < na || (n >= half && n - half < na)) continue;
            const long long src = ((long long)r * ng + g) * ds + n;
            const long long bias = ((long long)h * rank + r) * ds + n;
            kcur[i] = __fadd_rn(bnr[src], b_bias[bias]);
            qv[i] = __fadd_rn(cnr[src], c_bias[bias]);
        }
        __syncthreads();
        if (own) {
            const float x = to_f(pr[di + h * hd + tid]);
            const float z = to_f(pr[h * hd + tid]);
            float u[MIMO_RA(RK)], up[MIMO_RA(RK)], y[MIMO_RA(RK)];
            mimo_tok_row<RK>(st + tid * (ds + 1), y, u, up, px, x, v_prev, kcur, kprev, qv,
                             sc.alpha, sc.beta, sc.gamma, dval, rank, ds);
            mimo_put(&out[row * di + h * hd + tid], mimo_tok_gate<RK>(y, pz, po, z, rank));
            v_prev = x;
        }
        __syncthreads();
        float* used = kprev;
        kprev = kcur;
        kcur = used;
    }
    mimo_state_store(ssm + bh * hd * ds, st, hd, ds);
    for (int i = tid; i < rank * ds; i += blockDim.x)
        kst[bh * rank * ds + i] = kprev[i];
    if (own) vst[bh * hd + tid] = v_prev;
    if (angle_owner) ang[bh * na + tid] = theta;
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
