// The MIMO chunked path's step scalars and angles as they were released: one
// block per (sequence, head) walked every token, so a long window kept the
// device to as many multiprocessors as heads and every other warp waited at
// the barrier while one warp walked the angles. The split prep (the step
// scalars over tiles of tokens, the wrap chains a thread per angle, the
// sines and cosines a thread per value) is held to it bit for bit. The
// typed prelude and the packed-row helpers precede this fixture; the test
// defines M3M_CHUNK.

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

__device__ __forceinline__ float mimo_exp(float x) {
    return exp2f(__fmul_rn(x, LOG2E));
}

__device__ __forceinline__ float mimo_sigmoid(float x) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, mimo_exp(-x)));
}

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

// The step scalars and angles of every token of the window, one block per
// (sequence, head). coef [B, T, H, 4] = (dt, gamma, scale, Acs); sn, cs
// [B, T, H, na]; e0 [B, H] = dt_0 lam_c_0. The angle state ends advanced
// over the window. Every value is the one a single thread walking the
// tokens in order computes; only the work is spread:
//   1. per token: dt, gamma = lam dt, scale = gamma + the next token's
//      lead dt lam_c (recomputed rather than waited for; gamma alone when
//      the next token starts a packed sequence) and A dt;
//   2. per chunk: Acs, the running sum of A dt from zero at the chunk
//      start (and at every packed sequence's start), in token order; per
//      (token, angle): the increment tanh_pi dt, parked in sn;
//   3. per angle: the serial wrap chain over the window (from zero at
//      every packed sequence's start), the angles parked in sn;
//   4. per (token, angle): the sines and cosines.
// Block 512 threads.
template <typename TI>
__device__ __forceinline__ void m3m_tok_dt_lead(
    float& dt, float& lead, const TI* pr, const float* dt_bias, int nh, int heads, int h
) {
    float spin = __fadd_rn(to_f(pr[heads + h]), dt_bias[h]);
    dt = spin > 20.0f ? spin : log1pf(mimo_exp(spin));
    lead = __fmul_rn(dt, mimo_sigmoid(-to_f(pr[heads + 2 * nh + h])));
}

// One angle's serial wrap chain over a row's tokens: `col` holds the
// increments (stride `astr`) and leaves holding the angles; the angle the
// row carries in is `theta` and the one it carries out is returned. The
// chain loads the next AB increments while it walks the current ones, so a
// step waits on one add, not on memory. PACKED restarts the angle from
// zero at every token that starts a sequence.
template <bool PACKED>
__device__ __forceinline__ float m3m_angle_chain(
    float theta, float* col, long long astr, int seq,
    const int* __restrict__ seq_idx, long long b
) {
    constexpr int AB = 32;
    float cur[AB], nxt[AB];
    bool fcur[AB], fnxt[AB];
#pragma unroll
    for (int i = 0; i < AB; i++) {
        cur[i] = i < seq ? col[i * astr] : 0.0f;
        fcur[i] = PACKED && i < seq && vl_starts(seq_idx, b, seq, i);
    }
    for (int t0 = 0; t0 < seq; t0 += AB) {
#pragma unroll
        for (int i = 0; i < AB; i++) {
            const int t = t0 + AB + i;
            nxt[i] = t < seq ? col[t * astr] : 0.0f;
            fnxt[i] = PACKED && t < seq && vl_starts(seq_idx, b, seq, t);
        }
#pragma unroll
        for (int i = 0; i < AB; i++) {
            if (t0 + i < seq) {
                theta = mimo_wrap_add(PACKED && fcur[i] ? 0.0f : theta, cur[i]);
                col[(t0 + i) * astr] = theta;
            }
        }
#pragma unroll
        for (int i = 0; i < AB; i++) {
            cur[i] = nxt[i];
            fcur[i] = fnxt[i];
        }
    }
    return theta;
}

// One chunk's running sum of A dt from zero at its start, in token order,
// over the coef rows (stride `cstr`, the sum in column 3 in place). PACKED
// restarts the sum at every token that starts a sequence, so a sequence's
// decays never hold another's terms (an overflow there stays there).
template <bool PACKED>
__device__ __forceinline__ void m3m_acs_chunk(
    float* c_bh, long long cstr, int ch, int seq, const int* __restrict__ seq_idx, long long b
) {
    const int t0 = ch * M3M_CHUNK;
    const int t1 = min(seq, t0 + M3M_CHUNK);
    float acs = 0.0f;
    for (int t = t0; t < t1; t += 8) {
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = t + i < t1 ? c_bh[(t + i) * cstr + 3] : 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (t + i < t1) {
                const bool fresh = PACKED && vl_starts(seq_idx, b, seq, t + i);
                acs = __fadd_rn(fresh ? 0.0f : acs, v[i]);
                c_bh[(t + i) * cstr + 3] = acs;
            }
        }
    }
}

template <typename TI>
__device__ __forceinline__ void m3m_prep_body(
    float* __restrict__ coef, float* __restrict__ sn, float* __restrict__ cs,
    float* __restrict__ e0, float* __restrict__ ang,
    const TI* __restrict__ proj, long long ld_proj, const float* __restrict__ dt_bias,
    int seq, int nh, int heads, int ang_col, int na, float a_floor,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const long long bh = (long long)b * nh + h;
    const long long row0 = (long long)b * seq;
    const long long cstr = (long long)nh * 4;
    const long long astr = (long long)nh * na;
    float* const c_bh = coef + (row0 * nh + h) * 4;
    float* const sn_bh = sn + (row0 * nh + h) * na;
    float* const cs_bh = cs + (row0 * nh + h) * na;
    for (int t = threadIdx.x; t < seq; t += blockDim.x) {
        const TI* pr = proj + (row0 + t) * ld_proj;
        float dd_a = to_f(pr[heads + nh + h]);
        float ht = dd_a >= 0.0f ? __fadd_rn(1.0f, dd_a)
                                : __fdiv_rn(1.0f, __fsub_rn(1.0f, dd_a));
        float a = -ht;
        a = a > -a_floor ? -a_floor : a;
        float dt, lead;
        m3m_tok_dt_lead(dt, lead, pr, dt_bias, nh, heads, h);
        float gamma = __fmul_rn(mimo_sigmoid(to_f(pr[heads + 2 * nh + h])), dt);
        float scale = gamma;
        if (t + 1 < seq && !vl_starts(seq_idx, b, seq, t + 1)) {
            float dt_next, lead_next;
            m3m_tok_dt_lead(dt_next, lead_next, pr + ld_proj, dt_bias, nh, heads, h);
            scale = __fadd_rn(gamma, lead_next);
        }
        float* c = c_bh + t * cstr;
        c[0] = dt;
        c[1] = gamma;
        c[2] = scale;
        c[3] = __fmul_rn(a, dt);
        if (t == 0) e0[bh] = lead;
    }
    __syncthreads();
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    for (int ch = threadIdx.x; ch < nc; ch += blockDim.x) {
        if (seq_idx != nullptr)
            m3m_acs_chunk<true>(c_bh, cstr, ch, seq, seq_idx, b);
        else
            m3m_acs_chunk<false>(c_bh, cstr, ch, seq, seq_idx, b);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int t = warp; t < seq; t += nw) {
        const TI* pr = proj + (row0 + t) * ld_proj;
        const float dt = c_bh[t * cstr];
        for (int a = lane; a < na; a += 32)
            sn_bh[t * astr + a] = __fmul_rn(mimo_tanh_pi(pr, ang_col, a), dt);
    }
    __syncthreads();
    for (int a = threadIdx.x; a < na; a += blockDim.x) {
        const float theta = ang[bh * na + a];
        ang[bh * na + a] = seq_idx != nullptr
            ? m3m_angle_chain<true>(theta, sn_bh + a, astr, seq, seq_idx, b)
            : m3m_angle_chain<false>(theta, sn_bh + a, astr, seq, seq_idx, b);
    }
    __syncthreads();
    for (int t = warp; t < seq; t += nw) {
        for (int a = lane; a < na; a += 32) {
            const long long at = t * astr + a;
            const float th = sn_bh[at];
            sincosf(th, &sn_bh[at], &cs_bh[at]);
        }
    }
}

#define DEFINE_M3M_PREP_REFERENCE(NAME, TI)                                    \
extern "C" __global__ void __launch_bounds__(512) NAME(                        \
    float* __restrict__ coef, float* __restrict__ sn, float* __restrict__ cs,  \
    float* __restrict__ e0, float* __restrict__ ang,                           \
    const TI* __restrict__ proj, long long ld_proj,                            \
    const float* __restrict__ dt_bias,                                         \
    int seq, int nh, int heads, int ang_col, int na, float a_floor,            \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3m_prep_body<TI>(coef, sn, cs, e0, ang, proj, ld_proj, dt_bias, seq, nh,  \
        heads, ang_col, na, a_floor, y_first, seq_idx);                        \
}
DEFINE_M3M_PREP_REFERENCE(m3m_prep_reference, float)
DEFINE_M3M_PREP_REFERENCE(m3m_prep_reference_bf16, __nv_bfloat16)
DEFINE_M3M_PREP_REFERENCE(m3m_prep_reference_f16, __half)
