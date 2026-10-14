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

// The module is compiled for one rank, MIMO_RANK, which the host defines:
// the bodies templated on RK run at that rank as a constant, their rank
// arrays unrolled into registers.
#ifndef MIMO_RANK
#error "MIMO_RANK, the rank the module serves, is not defined"
#endif
#if MIMO_RANK < 1 || MIMO_RANK > MIMO_MAX_RANK
#error "MIMO_RANK lies outside 1..MIMO_MAX_RANK"
#endif

// The state indices a batch of a channel's row loop takes: four at ranks
// up to 4, two up to 8 and one past that, so a batch's rank arrays stay in
// registers.
#define MIMO_ROW_BATCH(RK) ((RK) <= 4 ? 4 : ((RK) <= 8 ? 2 : 1))

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
// backward. A token that starts a packed sequence (`fresh`) keeps nothing
// of the step before: its alpha and beta are zero.
struct MimoStep {
    float alpha, beta, gamma, dt, a, lam, lam_c;
};

template <typename TI>
__device__ __forceinline__ MimoStep mimo_step_scalars(
    const TI* pr, int heads, int nh, int h, float dt_bias, float a_floor, bool fresh
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
    if (fresh) {
        st.alpha = 0.0f;
        st.beta = 0.0f;
    }
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
// y_r = s . q_r + D u_r. The state indices go in batches of
// MIMO_ROW_BATCH, every load of a batch issued before its chains run: the
// row's own store would otherwise hold the next index's loads back a round
// trip.
// The keys and queries of a batch of four, the same for every channel of a
// warp, come as one float4 per rank when their rows are 16-byte aligned.
// The chains are the same, n ascending.
// FRESH is a token that starts a packed sequence: its state row and the
// previous keys are the empty state, so they read as zeros and the buffers,
// which still hold the sequence before, are never loaded (the caller also
// passes zero alpha, beta and v_prev). The chains are the ones a window
// starting from the empty state runs.
template <int RK, bool FRESH = false>
__device__ __forceinline__ void mimo_tok_row(
    float* __restrict__ srow, float* y, float* u, float* up, const float* px, float x,
    float v_prev, const float* __restrict__ kc, const float* __restrict__ kp,
    const float* __restrict__ qv,
    float alpha, float beta, float gamma, float dval, int rank, int ds
) {
    constexpr int J = MIMO_ROW_BATCH(RK);
    constexpr int R = RK;
    float bv[RK], gx[RK];
#pragma unroll
    for (int r = 0; r < R; r++) {
        u[r] = __fmul_rn(px[r], x);
        gx[r] = __fmul_rn(gamma, u[r]);
        up[r] = __fmul_rn(px[r], v_prev);
        bv[r] = __fmul_rn(beta, up[r]);
        y[r] = 0.0f;
    }
    const bool vec = J == 4 && mimo_rows_aligned(kc, FRESH ? kc : kp, qv, ds);
    int n = 0;
    for (; J > 1 && n + J <= ds; n += J) {
        float sv[J], kcv[J][RK], kpv[J][RK], qvv[J][RK];
#pragma unroll
        for (int j = 0; j < J; j++) sv[j] = FRESH ? 0.0f : srow[n + j];
        if (vec) {
#pragma unroll
            for (int r = 0; r < R; r++) {
                mimo_split4(kcv, r, kc + r * ds + n);
                if constexpr (FRESH) {
#pragma unroll
                    for (int j = 0; j < J; j++) kpv[j][r] = 0.0f;
                } else {
                    mimo_split4(kpv, r, kp + r * ds + n);
                }
                mimo_split4(qvv, r, qv + r * ds + n);
            }
        } else {
#pragma unroll
            for (int j = 0; j < J; j++) {
#pragma unroll
                for (int r = 0; r < R; r++) {
                    kcv[j][r] = kc[r * ds + n + j];
                    kpv[j][r] = FRESH ? 0.0f : kp[r * ds + n + j];
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
        inner = __fmaf_rn(bv[0], FRESH ? 0.0f : kp[n], inner);
#pragma unroll
        for (int r = 1; r < R; r++) {
            inner = __fmaf_rn(gx[r], kc[r * ds + n], inner);
            inner = __fmaf_rn(bv[r], FRESH ? 0.0f : kp[r * ds + n], inner);
        }
        float s = __fmaf_rn(alpha, FRESH ? 0.0f : srow[n], inner);
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
    for (int r = 0; r < RK; r++) {
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

// Whether this block is the last of its (sequence, head) to finish when
// the head's channels run in several blocks (grid z): that block writes
// the keys and angles the head carries out, which every block read at its
// start, so no block reads them after they move. Every block computes the
// same keys and angles, so which block writes them never changes a value.
// `done` holds one counter per (sequence, head), zero between launches.
// Every thread of the block calls it.
__device__ __forceinline__ bool mimo_last_tile(unsigned int* done, long long bh, int tiles) {
    if (tiles == 1) return true;
    __shared__ unsigned int mimo_last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const bool last = atomicAdd(&done[bh], 1u) == (unsigned int)tiles - 1u;
        if (last) {
            done[bh] = 0u;
            __threadfence();
        }
        mimo_last = last ? 1u : 0u;
    }
    __syncthreads();
    return mimo_last != 0u;
}

// The scan of one head over a window, a tile of its channels per block.
// Grid (batch, nheads, ceil(headdim / tile)), block a whole number of
// warps covering the tile's channels and at least the angles (the threads
// past the channels help move the state and build the keys). Dynamic
// shared memory, in floats: tile * (ds + 1) for the state rows, 3 * R * ds
// for the keys of the previous and the current token and the queries.
// Every block of a head builds the same keys and angles: the first writes
// their checkpoints, the last to finish carries them out (mimo_last_tile).
// The channels share nothing else, so a tile never changes a value.
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
//
// With a packing (`seq_idx`, see _varlen.cuh) a token that starts a
// sequence runs from the empty state, previous keys, previous input and
// angles, with zero alpha and beta; a checkpoint keeps the state before
// it, and the replay starts the sequence again the same way.
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
    float* __restrict__ ck_v, float* __restrict__ ck_ang, int stride,
    int tile, unsigned int* __restrict__ done, float* __restrict__ y_out,
    int y_first, /* this run's first y block */
    int z_first, /* this run's first z block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float mimo_sm[];
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int tid = threadIdx.x;
    // This block's channels of the head: p0 .. p0 + pc - 1.
    const int p0 = (z_first + blockIdx.z) * tile;
    const int pc = min(tile, hd - p0);
    const bool lead = (z_first + blockIdx.z) == 0;
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    float* st = mimo_sm;
    float* kp = st + ((pc * (ds + 1) + 3) & ~3);  // the keys start on 16 bytes
    float* kc = kp + rank * ds;
    float* qv = kc + rank * ds;
    const long long bh = (long long)b * nh + h;
    const long long so = bh * hd * ds + (long long)p0 * ds;
    mimo_state_load(st, ssm + so, pc, ds);
    for (int i = tid; i < rank * ds; i += blockDim.x)
        kp[i] = kst[bh * rank * ds + i];
    const bool own = tid < pc;
    const int pch = p0 + tid;
    float v_prev = own ? vst[bh * hd + pch] : 0.0f;
    constexpr int R = RK;
    float px[RK], pz[RK], po[RK];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const long long o = ((long long)h * rank + r) * hd + pch;
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
            mimo_state_store(ck_ssm + q * hd * ds + (long long)p0 * ds, st, pc, ds);
            if (lead)
                for (int i = tid; i < rank * ds; i += blockDim.x) ck_k[q * rank * ds + i] = kprev[i];
            if (own) ck_v[q * hd + pch] = v_prev;
            if (lead && angle_owner) ck_ang[q * na + tid] = theta;
        }
        const bool fresh = vl_starts(seq_idx, b, seq, t);
        const MimoStep sc = mimo_step_scalars(pr, heads, nh, h, dtb, a_floor, fresh);
        const float dt = sc.dt;
        const float* bnr = bn + row * rgn;
        const float* cnr = cn + row * rgn;
        if (angle_owner) {
            const int a = tid;
            if (fresh) theta = 0.0f;
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
            const float x = to_f(pr[di + h * hd + pch]);
            const float z = to_f(pr[h * hd + pch]);
            float u[RK], up[RK], y[RK];
            if (fresh) {
                v_prev = 0.0f;
                mimo_tok_row<RK, true>(st + tid * (ds + 1), y, u, up, px, x, v_prev, kcur,
                                       kprev, qv, sc.alpha, sc.beta, sc.gamma, dval, rank, ds);
            } else {
                mimo_tok_row<RK>(st + tid * (ds + 1), y, u, up, px, x, v_prev, kcur, kprev, qv,
                                 sc.alpha, sc.beta, sc.gamma, dval, rank, ds);
            }
            mimo_put(&out[row * di + h * hd + pch], mimo_tok_gate<RK>(y, pz, po, z, rank));
            if (y_out != nullptr) {
#pragma unroll
                for (int r = 0; r < R; r++)
                    y_out[((row * nh + h) * rank + r) * hd + pch] = y[r];
            }
            v_prev = x;
        }
        __syncthreads();
        float* used = kprev;
        kprev = kcur;
        kcur = used;
    }
    mimo_state_store(ssm + so, st, pc, ds);
    if (own) vst[bh * hd + pch] = v_prev;
    if (mimo_last_tile(done, bh, (hd + tile - 1) / tile)) {
        for (int i = tid; i < rank * ds; i += blockDim.x)
            kst[bh * rank * ds + i] = kprev[i];
        if (angle_owner) ang[bh * na + tid] = theta;
    }
}

// The scan with the keys of the next token built while the channels run
// the current one, when the block has threads past the channels (at least
// one per angle): those threads (the builders) compute token t + 1's step
// scalars, turn its key and query pairs and bias the rest, into the next
// of three key buffers and two query and scalar buffers, while the channel
// threads run token t's row and gate; a token costs one barrier. Every
// value is mimo_scan_body's expression, so the bits are its own. The angle
// of builder a is the angle mimo_scan_body keeps in thread a; with CKPT a
// builder writes the angle before token t when it builds token t.
// Dynamic shared memory: the state rows (padded to 16 bytes), 5 R N floats
// for the keys and queries and 8 for the scalars.
template <typename TI, bool CKPT, int RK>
__device__ __forceinline__ void mimo_scan_pipe_body(
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
    float* __restrict__ ck_v, float* __restrict__ ck_ang, int stride,
    int tile, unsigned int* __restrict__ done, float* __restrict__ y_out,
    int y_first, /* this run's first y block */
    int z_first, /* this run's first z block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float mimo_sm[];
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int tid = threadIdx.x;
    // This block's channels of the head: p0 .. p0 + pc - 1.
    const int p0 = (z_first + blockIdx.z) * tile;
    const int pc = min(tile, hd - p0);
    const bool lead = (z_first + blockIdx.z) == 0;
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int rn = rank * ds;
    float* st = mimo_sm;
    float* kbuf = st + ((pc * (ds + 1) + 3) & ~3);  // [3][R N], 16-byte aligned
    float* qbuf = kbuf + 3 * rn;                     // [2][R N]
    float* scal = qbuf + 2 * rn;                     // [2][4]: alpha, beta, gamma
    const long long bh = (long long)b * nh + h;
    const int lanes = (pc + 31) & ~31;
    const bool own = tid < pc;
    const int pch = p0 + tid;
    const bool builder = tid >= lanes;
    const int btid = tid - lanes, bcount = (int)blockDim.x - lanes;
    const bool angle_owner = builder && btid < na;
    const long long so = bh * hd * ds + (long long)p0 * ds;
    mimo_state_load(st, ssm + so, pc, ds);
    for (int i = tid; i < rn; i += blockDim.x) kbuf[2 * rn + i] = kst[bh * rn + i];
    float v_prev = own ? vst[bh * hd + pch] : 0.0f;
    constexpr int R = RK;
    float px[RK], pz[RK], po[RK];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const long long o = ((long long)h * rank + r) * hd + pch;
        px[r] = own ? mimo_x[o] : 0.0f;
        pz[r] = own ? mimo_z[o] : 0.0f;
        po[r] = own ? mimo_o[o] : 0.0f;
    }
    const float dval = D[h];
    const int heads = 2 * di + 2 * rgn;
    const int ang_col = heads + 3 * nh;
    const int half = ds / 2;
    const int nseg = CKPT ? (seq + stride - 1) / stride : 0;
    float theta = angle_owner ? ang[bh * na + btid] : 0.0f;
    const float dtb = dt_bias[h];
    // Token t's keys, queries and scalars into their buffers (builders).
    auto build = [&](int t) {
        const long long row = (long long)b * seq + t;
        const TI* pr = proj + row * ld_proj;
        const bool fresh = vl_starts(seq_idx, b, seq, t);
        const MimoStep sc = mimo_step_scalars(pr, heads, nh, h, dtb, a_floor, fresh);
        float* kd = kbuf + (t % 3) * rn;
        float* qd = qbuf + (t % 2) * rn;
        if (btid == 0) {
            float* sd = scal + (t % 2) * 4;
            sd[0] = sc.alpha;
            sd[1] = sc.beta;
            sd[2] = sc.gamma;
        }
        const float* bnr = bn + row * rgn;
        const float* cnr = cn + row * rgn;
        if (angle_owner) {
            const int a = btid;
            if (CKPT && lead && t % stride == 0)
                ck_ang[(((long long)b * nseg + t / stride) * nh + h) * na + a] = theta;
            if (fresh) theta = 0.0f;
            theta = mimo_wrap_add(theta, __fmul_rn(mimo_tanh_pi(pr, ang_col, a), sc.dt));
            float sa, ca;
            sincosf(theta, &sa, &ca);
            for (int r = 0; r < rank; r++) {
                const long long src = ((long long)r * ng + g) * ds;
                const long long bias = ((long long)h * rank + r) * ds;
                const float k0 = __fadd_rn(bnr[src + a], b_bias[bias + a]);
                const float k1 = __fadd_rn(bnr[src + a + half], b_bias[bias + a + half]);
                const float2 kr = mimo_rot_pair(k0, k1, sa, ca);
                kd[r * ds + a] = kr.x;
                kd[r * ds + a + half] = kr.y;
                const float q0 = __fadd_rn(cnr[src + a], c_bias[bias + a]);
                const float q1 = __fadd_rn(cnr[src + a + half], c_bias[bias + a + half]);
                const float2 qr = mimo_rot_pair(q0, q1, sa, ca);
                qd[r * ds + a] = qr.x;
                qd[r * ds + a + half] = qr.y;
            }
        }
        for (int i = btid; i < rn; i += bcount) {
            const int r = i / ds, n = i - r * ds;
            if (n < na || (n >= half && n - half < na)) continue;
            const long long src = ((long long)r * ng + g) * ds + n;
            const long long bias = ((long long)h * rank + r) * ds + n;
            kd[i] = __fadd_rn(bnr[src], b_bias[bias]);
            qd[i] = __fadd_rn(cnr[src], c_bias[bias]);
        }
    };
    if (builder && seq > 0) build(0);
    __syncthreads();
    for (int t = 0; t < seq; t++) {
        const long long row = (long long)b * seq + t;
        if (CKPT && t % stride == 0) {
            const long long q = ((long long)b * nseg + t / stride) * nh + h;
            mimo_state_store(ck_ssm + q * hd * ds + (long long)p0 * ds, st, pc, ds);
            const float* kprev = kbuf + ((t + 2) % 3) * rn;
            if (lead)
                for (int i = tid; i < rn; i += blockDim.x) ck_k[q * rn + i] = kprev[i];
            if (own) ck_v[q * hd + pch] = v_prev;
            __syncthreads();
        }
        if (own) {
            const TI* pr = proj + row * ld_proj;
            const float* sd = scal + (t % 2) * 4;
            const float x = to_f(pr[di + h * hd + pch]);
            const float z = to_f(pr[h * hd + pch]);
            float u[RK], up[RK], y[RK];
            if (vl_starts(seq_idx, b, seq, t)) {
                v_prev = 0.0f;
                mimo_tok_row<RK, true>(st + tid * (ds + 1), y, u, up, px, x, v_prev,
                                       kbuf + (t % 3) * rn, kbuf + ((t + 2) % 3) * rn,
                                       qbuf + (t % 2) * rn, sd[0], sd[1], sd[2], dval, rank, ds);
            } else {
                mimo_tok_row<RK>(st + tid * (ds + 1), y, u, up, px, x, v_prev,
                                 kbuf + (t % 3) * rn, kbuf + ((t + 2) % 3) * rn,
                                 qbuf + (t % 2) * rn, sd[0], sd[1], sd[2], dval, rank, ds);
            }
            mimo_put(&out[row * di + h * hd + pch], mimo_tok_gate<RK>(y, pz, po, z, rank));
            if (y_out != nullptr) {
#pragma unroll
                for (int r = 0; r < R; r++)
                    y_out[((row * nh + h) * rank + r) * hd + pch] = y[r];
            }
            v_prev = x;
        } else if (builder && t + 1 < seq) {
            build(t + 1);
        }
        __syncthreads();
    }
    mimo_state_store(ssm + so, st, pc, ds);
    if (own) vst[bh * hd + pch] = v_prev;
    if (mimo_last_tile(done, bh, (hd + tile - 1) / tile)) {
        const float* klast = kbuf + ((seq + 2) % 3) * rn;
        for (int i = tid; i < rn; i += blockDim.x) kst[bh * rn + i] = klast[i];
        if (angle_owner) ang[bh * na + btid] = theta;
    }
}

// The scan of a head's tile of channels: the pipelined body when the
// host's shared memory takes its keys (`allow_pipe`) and the block has a
// builder for every angle past the tile's channels, else the plain one.
template <typename TI, bool CKPT, int RK>
__device__ __forceinline__ void mimo_scan_any(
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
    float* __restrict__ ck_v, float* __restrict__ ck_ang, int stride,
    int tile, int allow_pipe, unsigned int* __restrict__ done, float* __restrict__ y_out,
    int y_first, /* this run's first y block */
    int z_first, /* this run's first z block */
    const int* __restrict__ seq_idx
) {
    const int pc = min(tile, hd - (int)(z_first + blockIdx.z) * tile);
    const int spare = (int)blockDim.x - ((pc + 31) & ~31);
    if (allow_pipe && spare >= 32 && spare >= na)
        mimo_scan_pipe_body<TI, CKPT, RK>(out, ssm, kst, vst, ang, proj, ld_proj, bn, cn,
            dt_bias, b_bias, c_bias, D, mimo_x, mimo_z, mimo_o, seq, nh, ng, hd, ds, rank,
            na, a_floor, ck_ssm, ck_k, ck_v, ck_ang, stride, tile, done, y_out, y_first, z_first,
            seq_idx);
    else
        mimo_scan_body<TI, CKPT, RK>(out, ssm, kst, vst, ang, proj, ld_proj, bn, cn,
            dt_bias, b_bias, c_bias, D, mimo_x, mimo_z, mimo_o, seq, nh, ng, hd, ds, rank,
            na, a_floor, ck_ssm, ck_k, ck_v, ck_ang, stride, tile, done, y_out, y_first, z_first,
            seq_idx);
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
    float a_floor, int tile, int allow_pipe, unsigned int* __restrict__ done,  \
    float* __restrict__ y_out,                                                 \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    mimo_scan_any<TI, false, MIMO_RANK>(out, ssm, kst, vst, ang, proj,         \
        ld_proj, bn, cn, dt_bias, b_bias, c_bias, D, mimo_x, mimo_z, mimo_o,   \
        seq, nh, ng, hd, ds, rank, na, a_floor, nullptr, nullptr, nullptr,     \
        nullptr, 1, tile, allow_pipe, done, y_out, y_first, z_first, seq_idx); \
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
    float* __restrict__ ck_v, float* __restrict__ ck_ang, int stride,          \
    int tile, int allow_pipe, unsigned int* __restrict__ done,                 \
    float* __restrict__ y_out,                                                 \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    mimo_scan_any<TI, true, MIMO_RANK>(out, ssm, kst, vst, ang, proj,          \
        ld_proj, bn, cn, dt_bias, b_bias, c_bias, D, mimo_x, mimo_z, mimo_o,   \
        seq, nh, ng, hd, ds, rank, na, a_floor, ck_ssm, ck_k, ck_v, ck_ang,    \
        stride, tile, allow_pipe, done, y_out, y_first, z_first, seq_idx);     \
}

DEFINE_MIMO_SCAN_CKPT(mimo_scan_ckpt, float)
DEFINE_MIMO_SCAN_CKPT(mimo_scan_ckpt_bf16, __nv_bfloat16)
DEFINE_MIMO_SCAN_CKPT(mimo_scan_ckpt_f16, __half)

// The gate with the per-rank output norm (is_outproj_norm), over the
// readouts y [B, T, H, R, P] (D included) the scans write: each rank's row
// over the head's channels normalised, its sum of squares one chain over p
// ascending as the CPU folds it, scaled by the head's norm weight, then
// gated, o = sum_r (((y_r rstd_r) w) silu(Z_r z)) Phi_r, r ascending.
// Grid (rows, H), block a whole number of warps.
template <typename TI>
__device__ __forceinline__ void mimo_gate_normed_body(
    TI* __restrict__ out, const float* __restrict__ y,
    const TI* __restrict__ proj, long long ld_proj, const float* __restrict__ norm_w,
    const float* __restrict__ mimo_z, const float* __restrict__ mimo_o,
    int nh, int hd, int rank, float eps,
    int y_first /* this run's first y block */
) {
    __shared__ float rstd[MIMO_MAX_RANK];
    const long long row = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const float* yr = y + (row * nh + h) * (long long)rank * hd;
    for (int r = threadIdx.x; r < rank; r += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p < hd; p++) acc = __fmaf_rn(yr[r * hd + p], yr[r * hd + p], acc);
        rstd[r] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(acc, (float)hd), eps)));
    }
    __syncthreads();
    const long long di = (long long)nh * hd;
    for (int p = threadIdx.x; p < hd; p += blockDim.x) {
        const float z = to_f(proj[row * ld_proj + (long long)h * hd + p]);
        const float w = norm_w[(long long)h * hd + p];
        float o = 0.0f;
        for (int r = 0; r < rank; r++) {
            const long long q = ((long long)h * rank + r) * hd + p;
            const float v = __fmul_rn(mimo_z[q], z);
            const float sg = __fdiv_rn(v, __fadd_rn(1.0f, mimo_exp(-v)));
            const float a = __fmul_rn(__fmul_rn(yr[r * hd + p], rstd[r]), w);
            const float t = __fmul_rn(__fmul_rn(a, sg), mimo_o[q]);
            o = r == 0 ? t : __fadd_rn(o, t);
        }
        mimo_put(&out[row * di + (long long)h * hd + p], o);
    }
}

#define DEFINE_MIMO_GATE_NORMED(NAME, TI)                                      \
extern "C" __global__ void NAME(                                               \
    TI* __restrict__ out, const float* __restrict__ y,                         \
    const TI* __restrict__ proj, long long ld_proj,                            \
    const float* __restrict__ norm_w,                                          \
    const float* __restrict__ mimo_z, const float* __restrict__ mimo_o,        \
    int nh, int hd, int rank, float eps,                                       \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    mimo_gate_normed_body<TI>(out, y, proj, ld_proj, norm_w, mimo_z, mimo_o,   \
        nh, hd, rank, eps, y_first);                                           \
}
DEFINE_MIMO_GATE_NORMED(mimo_gate_normed, float)
DEFINE_MIMO_GATE_NORMED(mimo_gate_normed_bf16, __nv_bfloat16)
DEFINE_MIMO_GATE_NORMED(mimo_gate_normed_f16, __half)
