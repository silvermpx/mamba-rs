// Mamba-3 MIMO on the GPU, the chunked path: a window in chunks of 64
// tokens, the chunks in parallel.
//
// The trapezoid folds the way SISO's chunked form folds it. Token s's rank
// inputs u_s[r] = Psi_r x_s and rotated keys k_s[r] enter the state with
// gamma_s = lam_s dt_s at their own step and with beta_{s+1} = alpha_{s+1}
// dt_{s+1} lam_c_{s+1} at the next one, so from step s + 1 on they weigh
//   scale_s = gamma_s + dt_{s+1} lam_c_{s+1}     (gamma_s alone at the last token)
// and within a chunk, with Acs the running sum of a dt,
//   S_t = e^Acs_t S_enter + sum_{s<t} e^(Acs_t - Acs_s) scale_s U_s K_s + gamma_t U_t K_t
//   y_t[r] = q_t[r] . S_t + D u_t[r]
// The first token's beta term reads the carried previous keys and input:
// the state entering the window is S_init + dt_0 lam_c_0 sum_r (Psi_r v) kp_r.
// A chunk's own contribution (with every token's scale, the last one's
// reaching into the next chunk) is decayed into the next chunk's entering
// state by the state passing.
//
// The pieces: m3m_prep (step scalars and angles per token, the angles in
// the sequential path's own arithmetic), m3m_chunk_state, m3m_state_pass,
// m3m_chunk_scan (the outputs and the gate), m3m_writeback (the carried
// keys and input). Every value is one thread's chain in a fixed order; the
// bits do not depend on the batch or the grid.
//
// Composed after mamba3_mimo.cu in the MIMO module.

#define M3M_CHUNK 64

// The rotated value at n of a biased key or query rank row (the group's
// normalised row `src` plus the head's `bias`), the half-split pairs
// (a, a + N/2) for a < na turned by the token's angles (mimo_rot_pair),
// the scan's arithmetic.
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

// The inputs of one m3m_rot value, loaded apart from the turn so a thread
// can put a whole batch of loads in flight before it turns any: the biased
// value at n, its pair partner at the other half and the pair's sine and
// cosine, the last three only when the pair turns.
struct M3mRotIn {
    float v, w, s, c;
};

__device__ __forceinline__ M3mRotIn m3m_rot_load(
    const float* src, const float* bias, int n, int half, int na,
    const float* sr, const float* cr
) {
    const int a = n < half ? n : n - half;
    const int m = n < half ? n + half : a;
    M3mRotIn in = {__fadd_rn(src[n], bias[n]), 0.0f, 0.0f, 0.0f};
    if (a < na) {
        in.w = __fadd_rn(src[m], bias[m]);
        in.s = sr[a];
        in.c = cr[a];
    }
    return in;
}

// m3m_rot's value from loaded inputs, the same arithmetic.
__device__ __forceinline__ float m3m_rot_turn(const M3mRotIn& in, int n, int half, int na) {
    const int a = n < half ? n : n - half;
    if (a >= na) return in.v;
    return n < half ? mimo_rot_pair(in.v, in.w, in.s, in.c).x
                    : mimo_rot_pair(in.w, in.v, in.s, in.c).y;
}

// A small product out[r][c] = init(r, c) + sum_k L(r, k) R(k, c), k
// ascending, as register tiles: a warp takes 4 rows at a time and a lane
// the columns lane, lane + 32, ... (JC of them), so a step of k loads 4 + JC
// values for 4 JC products and a warp's column loads fall in distinct banks.
// Every output keeps the one chain over k ascending a thread would run
// alone. Only whole warps work: the threads of a last partial warp sit
// out, so every output has one owner whatever the block size (at least 32).
template <int JC, class LF, class RF, class INF, class PUTF>
__device__ __forceinline__ void m3m_tile_product(
    int rows, int cols, int depth, LF lf, RF rf, INF init, PUTF put
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nw = blockDim.x >> 5;
    if (warp >= nw) return;
    for (int r0 = 4 * warp; r0 < rows; r0 += 4 * nw) {
        for (int c0 = 0; c0 < cols; c0 += 32 * JC) {
            float acc[4][JC];
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < JC; j++) {
                    const int r = r0 + i, c = c0 + lane + 32 * j;
                    acc[i][j] = r < rows && c < cols ? init(r, c) : 0.0f;
                }
            }
            for (int k = 0; k < depth; k++) {
                float lv[4], rv[JC];
#pragma unroll
                for (int i = 0; i < 4; i++) lv[i] = r0 + i < rows ? lf(r0 + i, k) : 0.0f;
#pragma unroll
                for (int j = 0; j < JC; j++) {
                    const int c = c0 + lane + 32 * j;
                    rv[j] = c < cols ? rf(k, c) : 0.0f;
                }
#pragma unroll
                for (int i = 0; i < 4; i++) {
#pragma unroll
                    for (int j = 0; j < JC; j++) acc[i][j] = __fmaf_rn(lv[i], rv[j], acc[i][j]);
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < JC; j++) {
                    const int r = r0 + i, c = c0 + lane + 32 * j;
                    if (r < rows && c < cols) put(r, c, acc[i][j]);
                }
            }
        }
    }
}

// The step scalars and angles of every token of the window, one block per
// (sequence, head). coef [B, T, H, 4] = (dt, gamma, scale, Acs); sn, cs
// [B, T, H, na]; e0 [B, H] = dt_0 lam_c_0. The angle state ends advanced
// over the window. Every value is the one a single thread walking the
// tokens in order computes; only the work is spread:
//   1. per token: dt, gamma = lam dt, scale = gamma + the next token's
//      lead dt lam_c (recomputed rather than waited for) and A dt;
//   2. per chunk: Acs, the running sum of A dt from zero at the chunk
//      start, in token order; per (token, angle): the increment tanh_pi dt,
//      parked in sn;
//   3. per angle: the serial wrap chain over the window, the angles parked
//      in sn;
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

template <typename TI>
__device__ __forceinline__ void m3m_prep_body(
    float* __restrict__ coef, float* __restrict__ sn, float* __restrict__ cs,
    float* __restrict__ e0, float* __restrict__ ang,
    const TI* __restrict__ proj, long long ld_proj, const float* __restrict__ dt_bias,
    int seq, int nh, int heads, int ang_col, int na, float a_floor
) {
    constexpr int AB = 32;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
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
        if (t + 1 < seq) {
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
                    acs = __fadd_rn(acs, v[i]);
                    c_bh[(t + i) * cstr + 3] = acs;
                }
            }
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int t = warp; t < seq; t += nw) {
        const TI* pr = proj + (row0 + t) * ld_proj;
        const float dt = c_bh[t * cstr];
        for (int a = lane; a < na; a += 32)
            sn_bh[t * astr + a] = __fmul_rn(mimo_tanh_pi(pr, ang_col, a), dt);
    }
    __syncthreads();
    // The chain loads the next AB increments while it walks the current ones,
    // so a step waits on one add, not on memory.
    for (int a = threadIdx.x; a < na; a += blockDim.x) {
        float theta = ang[bh * na + a];
        float* col = sn_bh + a;
        float cur[AB], nxt[AB];
#pragma unroll
        for (int i = 0; i < AB; i++) cur[i] = i < seq ? col[i * astr] : 0.0f;
        for (int t0 = 0; t0 < seq; t0 += AB) {
#pragma unroll
            for (int i = 0; i < AB; i++) {
                const int t = t0 + AB + i;
                nxt[i] = t < seq ? col[t * astr] : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < AB; i++) {
                if (t0 + i < seq) {
                    theta = mimo_wrap_add(theta, cur[i]);
                    col[(t0 + i) * astr] = theta;
                }
            }
#pragma unroll
            for (int i = 0; i < AB; i++) cur[i] = nxt[i];
        }
        ang[bh * na + a] = theta;
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

#define DEFINE_M3M_PREP(NAME, TI)                                               \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ coef, float* __restrict__ sn, float* __restrict__ cs,   \
    float* __restrict__ e0, float* __restrict__ ang,                            \
    const TI* __restrict__ proj, long long ld_proj,                             \
    const float* __restrict__ dt_bias,                                          \
    int seq, int nh, int heads, int ang_col, int na, float a_floor              \
) {                                                                             \
    m3m_prep_body<TI>(coef, sn, cs, e0, ang, proj, ld_proj, dt_bias, seq, nh,   \
        heads, ang_col, na, a_floor);                                           \
}
DEFINE_M3M_PREP(m3m_prep, float)
DEFINE_M3M_PREP(m3m_prep_bf16, __nv_bfloat16)
DEFINE_M3M_PREP(m3m_prep_f16, __half)

// Each chunk's own contribution to the state, from zero:
//   C[p][n] = sum_s sum_r (e^min(Acs_last - Acs_s, 0) scale_s) u_s[r][p] k_s[r][n]
// (s, then r, ascending). One block per (chunk, head, n-tile); thread
// (p, n-in-tile). The chunk's rotated keys for the tile's n and its inputs
// are staged in shared memory `sl` tokens at a time (the whole chunk when
// it fits): sl * R * nt + sl * P + sl floats. The tile sizes change what a
// pass stages, never the order of a thread's sum.
// Grid (B * nc, H, ceil(N / nt)), block nt * P threads.
template <typename TI>
__device__ __forceinline__ void m3m_chunk_state_body(
    float* __restrict__ contrib,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ coef,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ b_bias, const float* __restrict__ mimo_x,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int nt, int sl
) {
    extern __shared__ float m3m_sm[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int n0 = blockIdx.z * nt;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    float* kt = m3m_sm;                // [sl][R][nt]
    float* xt = kt + sl * rank * nt;   // [sl][P]
    float* wt = xt + sl * hd;          // [sl]
    const int tid = threadIdx.x;
    const int p = tid % hd, nl = tid / hd;
    const bool active = nl < nt && n0 + nl < ds;
    const float acs_last = coef[(((b * seq + t0 + lc - 1) * nh) + h) * 4 + 3];
    float acc = 0.0f;
    for (int s0 = 0; s0 < lc; s0 += sl) {
        const int sn_ = min(sl, lc - s0);
        if (s0 > 0) __syncthreads();
        for (int i = tid; i < sn_ * rank * nt; i += blockDim.x) {
            const int s = i / (rank * nt), r = (i / nt) % rank, ni = i % nt;
            const int n = n0 + ni;
            float v = 0.0f;
            if (n < ds) {
                const long long row = b * seq + t0 + s0 + s;
                v = m3m_rot(bn + row * rgn + ((long long)r * ng + g) * ds,
                            b_bias + ((long long)h * rank + r) * ds, n, half, na,
                            sn + (row * nh + h) * na, cs + (row * nh + h) * na);
            }
            kt[i] = v;
        }
        for (int i = tid; i < sn_ * hd; i += blockDim.x) {
            const int s = i / hd, pp = i % hd;
            xt[i] = to_f(proj[(b * seq + t0 + s0 + s) * ld_proj + di + (long long)h * hd + pp]);
        }
        for (int s = tid; s < sn_; s += blockDim.x) {
            const float* cf = coef + ((b * seq + t0 + s0 + s) * nh + h) * 4;
            wt[s] = __fmul_rn(mimo_exp(fminf(__fsub_rn(acs_last, cf[3]), 0.0f)), cf[2]);
        }
        __syncthreads();
        if (active) {
            for (int s = 0; s < sn_; s++) {
                const float wx = __fmul_rn(wt[s], xt[s * hd + p]);
                for (int r = 0; r < rank; r++) {
                    const float u = __fmul_rn(wx, mimo_x[((long long)h * rank + r) * hd + p]);
                    acc = __fmaf_rn(u, kt[(s * rank + r) * nt + nl], acc);
                }
            }
        }
    }
    if (active) contrib[((bc * nh + h) * hd + p) * (long long)ds + n0 + nl] = acc;
}

#define DEFINE_M3M_CHUNK_STATE(NAME, TI)                                        \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ contrib,                                                \
    const TI* __restrict__ proj, long long ld_proj,                             \
    const float* __restrict__ bn, const float* __restrict__ coef,               \
    const float* __restrict__ sn, const float* __restrict__ cs,                 \
    const float* __restrict__ b_bias, const float* __restrict__ mimo_x,         \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int nt, int sl   \
) {                                                                             \
    m3m_chunk_state_body<TI>(contrib, proj, ld_proj, bn, coef, sn, cs, b_bias,  \
        mimo_x, seq, nh, ng, hd, ds, rank, na, nt, sl);                         \
}
DEFINE_M3M_CHUNK_STATE(m3m_chunk_state, float)
DEFINE_M3M_CHUNK_STATE(m3m_chunk_state_bf16, __nv_bfloat16)
DEFINE_M3M_CHUNK_STATE(m3m_chunk_state_f16, __half)

// The same contributions as register tiles: a warp takes 4 channels p and a
// lane the state indices lane + 32 j of the block's tile (JC of them), so a
// step of (s, r) loads one float4 of u and JC keys for 4 JC products. The
// block computes u = (w_s x_s[p]) Psi_r[p] and the rotated keys once and
// stages them `sl` tokens at a time: ut [sl R][P4] (P rounded up to 4, the
// pad zero), kt [sl R][32 JC], wt [sl]. Every output keeps the chain above,
// acc = 0 then fma(u, k, acc) over s ascending and r ascending, across the
// passes. Grid (B * nc, H, ceil(N / (32 JC))), block 32 ceil(P / 4)
// threads.
template <int JC, typename TI>
__device__ __forceinline__ void m3m_chunk_state_tiles_body(
    float* __restrict__ contrib,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ coef,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ b_bias, const float* __restrict__ mimo_x,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int sl
) {
    extern __shared__ float4 m3m_sm4[];
    constexpr int NT = 32 * JC;
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int n0 = blockIdx.z * NT;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const int p4 = (hd + 3) & ~3;
    float* ut = reinterpret_cast<float*>(m3m_sm4);  // [sl R][p4]
    float* kt = ut + sl * rank * p4;                // [sl R][NT]
    float* wt = kt + sl * rank * NT;                // [sl]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5, nw = blockDim.x >> 5;
    const int p0 = 4 * warp;
    const float acs_last = coef[(((b * seq + t0 + lc - 1) * nh) + h) * 4 + 3];
    float acc[4][JC];
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < JC; j++) acc[i][j] = 0.0f;
    }
    for (int s0 = 0; s0 < lc; s0 += sl) {
        const int sn_ = min(sl, lc - s0);
        if (s0 > 0) __syncthreads();
        for (int s = tid; s < sn_; s += blockDim.x) {
            const float* cf = coef + ((b * seq + t0 + s0 + s) * nh + h) * 4;
            wt[s] = __fmul_rn(mimo_exp(fminf(__fsub_rn(acs_last, cf[3]), 0.0f)), cf[2]);
        }
        for (int sr = warp; sr < sn_ * rank; sr += nw) {
            const int r = sr % rank;
            const long long row = b * seq + t0 + s0 + sr / rank;
            const float* src = bn + row * rgn + ((long long)r * ng + g) * ds;
            const float* bias = b_bias + ((long long)h * rank + r) * ds;
            const float* sr_ = sn + (row * nh + h) * na;
            const float* cr_ = cs + (row * nh + h) * na;
#pragma unroll 4
            for (int ni = lane; ni < NT; ni += 32) {
                const int n = n0 + ni;
                kt[sr * NT + ni] = n < ds ? m3m_rot(src, bias, n, half, na, sr_, cr_) : 0.0f;
            }
        }
        __syncthreads();
        for (int sr = warp; sr < sn_ * rank; sr += nw) {
            const int s = sr / rank, r = sr % rank;
            const TI* xr = proj + (b * seq + t0 + s0 + s) * ld_proj + di + (long long)h * hd;
            const float* mx = mimo_x + ((long long)h * rank + r) * hd;
            const float w = wt[s];
#pragma unroll 4
            for (int pp = lane; pp < p4; pp += 32)
                ut[sr * p4 + pp] = pp < hd ? __fmul_rn(__fmul_rn(w, to_f(xr[pp])), mx[pp]) : 0.0f;
        }
        __syncthreads();
        const int kd = sn_ * rank;
        for (int k = 0; k < kd; k++) {
            const float4 lv = *reinterpret_cast<const float4*>(ut + k * p4 + p0);
            float rv[JC];
#pragma unroll
            for (int j = 0; j < JC; j++) rv[j] = kt[k * NT + lane + 32 * j];
#pragma unroll
            for (int j = 0; j < JC; j++) {
                acc[0][j] = __fmaf_rn(lv.x, rv[j], acc[0][j]);
                acc[1][j] = __fmaf_rn(lv.y, rv[j], acc[1][j]);
                acc[2][j] = __fmaf_rn(lv.z, rv[j], acc[2][j]);
                acc[3][j] = __fmaf_rn(lv.w, rv[j], acc[3][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int p = p0 + i;
        if (p >= hd) continue;
#pragma unroll
        for (int j = 0; j < JC; j++) {
            const int n = n0 + lane + 32 * j;
            if (n < ds) contrib[((bc * nh + h) * hd + p) * (long long)ds + n] = acc[i][j];
        }
    }
}

#define DEFINE_M3M_CHUNK_STATE_TILES(NAME, TI)                                  \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ contrib,                                                \
    const TI* __restrict__ proj, long long ld_proj,                             \
    const float* __restrict__ bn, const float* __restrict__ coef,               \
    const float* __restrict__ sn, const float* __restrict__ cs,                 \
    const float* __restrict__ b_bias, const float* __restrict__ mimo_x,         \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int jc, int sl   \
) {                                                                             \
    if (jc == 1)                                                                \
        m3m_chunk_state_tiles_body<1, TI>(contrib, proj, ld_proj, bn, coef, sn, \
            cs, b_bias, mimo_x, seq, nh, ng, hd, ds, rank, na, sl);             \
    else if (jc == 2)                                                           \
        m3m_chunk_state_tiles_body<2, TI>(contrib, proj, ld_proj, bn, coef, sn, \
            cs, b_bias, mimo_x, seq, nh, ng, hd, ds, rank, na, sl);             \
    else                                                                        \
        m3m_chunk_state_tiles_body<4, TI>(contrib, proj, ld_proj, bn, coef, sn, \
            cs, b_bias, mimo_x, seq, nh, ng, hd, ds, rank, na, sl);             \
}
DEFINE_M3M_CHUNK_STATE_TILES(m3m_chunk_state_tiles, float)
DEFINE_M3M_CHUNK_STATE_TILES(m3m_chunk_state_tiles_bf16, __nv_bfloat16)
DEFINE_M3M_CHUNK_STATE_TILES(m3m_chunk_state_tiles_f16, __half)

// The states entering the chunks: S = S_init + e0 sum_r (Psi_r v) kp_r
// (r ascending), then per chunk enter[c] = S, S = e^Acs_last S + C_c. The
// carried state ends as the state after the window. One thread per
// (b, h, p, n). Grid (B * H, ceil(P * N / blockDim.x)).
extern "C" __global__ void m3m_state_pass(
    float* __restrict__ enter,
    float* __restrict__ ssm,
    const float* __restrict__ contrib,
    const float* __restrict__ coef, const float* __restrict__ e0,
    const float* __restrict__ kst, const float* __restrict__ vst,
    const float* __restrict__ mimo_x,
    int batch, int seq, int nh, int hd, int ds, int rank
) {
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bh = blockIdx.x;
    const long long e = (long long)blockIdx.y * blockDim.x + threadIdx.x;
    const long long pn = (long long)hd * ds;
    if (bh >= (long long)batch * nh || e >= pn) return;
    const long long b = bh / nh;
    const int h = (int)(bh % nh);
    const int p = (int)(e / ds), n = (int)(e % ds);
    float adj = 0.0f;
    const float v = vst[bh * hd + p];
    for (int r = 0; r < rank; r++) {
        const float up = __fmul_rn(mimo_x[((long long)h * rank + r) * hd + p], v);
        adj = __fmaf_rn(up, kst[(bh * rank + r) * ds + n], adj);
    }
    float st = __fmaf_rn(e0[bh], adj, ssm[bh * pn + e]);
    for (int c = 0; c < nc; c++) {
        const int lc = min(M3M_CHUNK, seq - c * M3M_CHUNK);
        const long long bc = b * nc + c;
        const long long idx = (bc * nh + h) * pn + e;
        enter[idx] = st;
        const float acs_last = coef[((b * seq + c * M3M_CHUNK + lc - 1) * nh + h) * 4 + 3];
        st = __fmaf_rn(mimo_exp(acs_last), st, contrib[idx]);
    }
    ssm[bh * pn + e] = st;
}

// The outputs. One block per (chunk, head, token tile); thread (t, p) of
// the tile computes y_t[r] for every rank and the gated row:
//   y[r] = e^Acs_t (q_t[r] . S_enter[p])
//        + sum_{s<=t} (G[t,r,s,r'] w(t,s)) u_s[r'][p]        (s, then r')
//        + D u_t[r][p]
// with G[t,r,s,r'] = q_t[r] . k_s[r'] (n ascending) and w(t,s) =
// e^min(Acs_t - Acs_s, 0) scale_s off the diagonal, gamma_t on it. The
// queries of the tile stay staged; the keys come in s-tiles of `st` with
// their G block. Shared memory: tt * R * N (queries) + st * R * (N + 1)
// (keys, rows padded) + tt * R * st * R (G) + st * P (inputs) floats. The tile sizes change what
// a pass stages, never the order of a thread's sum.
// Grid (B * nc, H, ceil(64 / tt)), block tt * P threads. `y_out` (may be
// null) receives the readouts y_t[r][p] with D, [B, T, H, R, P], what the
// training backward's gate reads.
template <typename TI, int RK>
__device__ __forceinline__ void m3m_chunk_scan_body(
    TI* __restrict__ out, float* __restrict__ y_out,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ cn,
    const float* __restrict__ coef,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ enter,
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,
    const float* __restrict__ D,
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,
    const float* __restrict__ mimo_o,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tt, int st
) {
    extern __shared__ float m3m_sm[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int tile0 = blockIdx.z * tt;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    if (tile0 >= lc) return;
    const int tn = min(tt, lc - tile0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int R = MIMO_R(RK, rank);
    const int half = ds / 2;
    const int ST = st;
    // Key rows padded to N + 1: the block's G entries read one key row per
    // thread, which at a row stride of N (a multiple of 32) would put a
    // warp's reads in one bank.
    const int kld = ds + 1;
    float* qs = m3m_sm;                       // [tt][R][N]
    float* ks = qs + tt * rank * ds;          // [ST][R][N + 1]
    float* gs = ks + ST * rank * kld;         // [tt*R][ST*R]
    float* xs = gs + tt * rank * ST * rank;   // [ST][P]
    const int tid = threadIdx.x;
    const int tl = tid / hd, p = tid % hd;
    const bool live = tl < tn;
    // The tile's rotated queries.
    for (int i = tid; i < tn * rank * ds; i += blockDim.x) {
        const int tq = i / (rank * ds), r = (i / ds) % rank, n = i % ds;
        const long long row = b * seq + t0 + tile0 + tq;
        qs[i] = m3m_rot(cn + row * rgn + ((long long)r * ng + g) * ds,
                        c_bias + ((long long)h * rank + r) * ds, n, half, na,
                        sn + (row * nh + h) * na, cs + (row * nh + h) * na);
    }
    __syncthreads();
    float y[MIMO_RA(RK)];
    float u_t[MIMO_RA(RK)];
    float px[MIMO_RA(RK)];
    const int t = tile0 + tl;
    const long long row_t = b * seq + t0 + t;
    float acs_t = 0.0f, gamma_t = 0.0f;
    if (live) {
        const float* cf = coef + (row_t * nh + h) * 4;
        acs_t = cf[3];
        gamma_t = cf[1];
        const float xv = to_f(proj[row_t * ld_proj + di + (long long)h * hd + p]);
        const float* sp = enter + ((bc * nh + h) * hd + p) * (long long)ds;
        const float ef = mimo_exp(acs_t);
        _Pragma("unroll") for (int r = 0; r < R; r++) {
            px[r] = mimo_x[((long long)h * rank + r) * hd + p];
            u_t[r] = __fmul_rn(px[r], xv);
            y[r] = 0.0f;
        }
        // Every rank's chain over n ascending, the entering row read once.
        const float* q = qs + tl * rank * ds;
        for (int n = 0; n < ds; n++) {
            const float sv = sp[n];
            _Pragma("unroll") for (int r = 0; r < R; r++)
                y[r] = __fmaf_rn(q[r * ds + n], sv, y[r]);
        }
        _Pragma("unroll") for (int r = 0; r < R; r++) y[r] = __fmul_rn(y[r], ef);
    }
    const int s_end = tile0 + tn;
    for (int s0 = 0; s0 < s_end; s0 += ST) {
        const int sn_ = min(ST, s_end - s0);
        __syncthreads();
        for (int i = tid; i < sn_ * rank * ds; i += blockDim.x) {
            const int sk = i / (rank * ds), r = (i / ds) % rank, n = i % ds;
            const long long row = b * seq + t0 + s0 + sk;
            ks[(sk * rank + r) * kld + n] = m3m_rot(bn + row * rgn + ((long long)r * ng + g) * ds,
                            b_bias + ((long long)h * rank + r) * ds, n, half, na,
                            sn + (row * nh + h) * na, cs + (row * nh + h) * na);
        }
        for (int i = tid; i < sn_ * hd; i += blockDim.x) {
            const int sk = i / hd, pp = i % hd;
            xs[i] = to_f(proj[(b * seq + t0 + s0 + sk) * ld_proj + di + (long long)h * hd + pp]);
        }
        __syncthreads();
        m3m_tile_product<1>(
            tn * rank, sn_ * rank, ds,
            [&](int rw, int n) { return qs[rw * ds + n]; },
            [&](int n, int col) { return ks[col * kld + n]; },
            [&](int, int) { return 0.0f; },
            [&](int rw, int col, float v) { gs[rw * (ST * rank) + col] = v; });
        __syncthreads();
        if (live) {
            for (int sk = 0; sk < sn_; sk++) {
                const int s = s0 + sk;
                if (s > t) break;
                const long long row_s = b * seq + t0 + s;
                const float* cf = coef + (row_s * nh + h) * 4;
                const float w = s < t
                    ? __fmul_rn(mimo_exp(fminf(__fsub_rn(acs_t, cf[3]), 0.0f)), cf[2])
                    : gamma_t;
                const float xsv = xs[sk * hd + p];
                _Pragma("unroll") for (int r = 0; r < R; r++) {
                    const float* grow = gs + (tl * rank + r) * (ST * rank) + sk * rank;
                    _Pragma("unroll") for (int r2 = 0; r2 < R; r2++) {
                        const float us = __fmul_rn(px[r2], xsv);
                        y[r] = __fmaf_rn(__fmul_rn(grow[r2], w), us, y[r]);
                    }
                }
            }
        }
    }
    if (live) {
        const float dval = D[h];
        _Pragma("unroll") for (int r = 0; r < R; r++) y[r] = __fmaf_rn(dval, u_t[r], y[r]);
        if (y_out != nullptr) {
            _Pragma("unroll") for (int r = 0; r < R; r++)
                y_out[((row_t * nh + h) * rank + r) * hd + p] = y[r];
        }
        float pz[MIMO_RA(RK)], po[MIMO_RA(RK)];
        _Pragma("unroll") for (int r = 0; r < R; r++) {
            pz[r] = mimo_z[((long long)h * rank + r) * hd + p];
            po[r] = mimo_o[((long long)h * rank + r) * hd + p];
        }
        const float z = to_f(proj[row_t * ld_proj + (long long)h * hd + p]);
        mimo_put(&out[row_t * di + (long long)h * hd + p], mimo_tok_gate<RK>(y, pz, po, z, rank));
    }
}

#define DEFINE_M3M_CHUNK_SCAN(NAME, TI)                                         \
extern "C" __global__ void NAME(                                                \
    TI* __restrict__ out, float* __restrict__ y_out,                            \
    const TI* __restrict__ proj, long long ld_proj,                             \
    const float* __restrict__ bn, const float* __restrict__ cn,                 \
    const float* __restrict__ coef,                                             \
    const float* __restrict__ sn, const float* __restrict__ cs,                 \
    const float* __restrict__ enter,                                            \
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,         \
    const float* __restrict__ D,                                                \
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,         \
    const float* __restrict__ mimo_o,                                           \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tt, int st   \
) {                                                                             \
    MIMO_RANK_DISPATCH(rank, m3m_chunk_scan_body<TI, RK>(out, y_out, proj,      \
        ld_proj, bn, cn, coef, sn, cs, enter, b_bias, c_bias, D, mimo_x,        \
        mimo_z, mimo_o, seq, nh, ng, hd, ds, rank, na, tt, st))                 \
}
DEFINE_M3M_CHUNK_SCAN(m3m_chunk_scan, float)
DEFINE_M3M_CHUNK_SCAN(m3m_chunk_scan_bf16, __nv_bfloat16)
DEFINE_M3M_CHUNK_SCAN(m3m_chunk_scan_f16, __half)

// The same outputs as three register-tile products over the block's query
// tile of `tq` tokens. Rows are (t, r), t of the tile and r ascending; a warp
// takes 8 rows, a lane the channels p = lane + 32 j (JP of them):
//   inter: y = sum_n q[n][row] S_enter[p][n], then y *= e^Acs_t;
//   G:     G[row][col] = sum_n q[n][row] k[n][col] for the key tile's
//          columns col = (s, r'), then Gw = G w(t, s);
//   GU:    y += Gw[col][row] u[col][p], u = Psi_r' x_s, col ascending,
//          skipping s > t;
// then y = fma(D, u_t, y), the readouts, and the gate over r per (t, p).
// Every output keeps the kernel above's chains: n ascending from 0 for the
// inter term and for G, then (s, r') ascending, each term
// fma(fmul(G, w), fmul(Psi, x), y). The keys of a tile of `kt` tokens are
// rotated once per query tile and staged `nk` state indices at a time; the
// inter pass streams the entering state through the same space.
// Shared memory: q [N][QR + 4], keys [nk][68], u [64][32 JP],
// Gw [64][QR + 4], w [tq][kt], the chunk's Acs, scale and gamma [3][64];
// QR = tq R rounded up to 8. A warp holds 4 query rows, so a block of 64
// rows runs 16 warps: one block fills the shared memory of an SM and the
// extra warps hide the staging loads. The heaviest query tile (the last)
// runs first. Grid (B * nc, H, ceil(64 / tq)), block 8 QR threads.
template <typename TI, int JP>
__device__ __forceinline__ void m3m_chunk_scan_tiles_body(
    TI* __restrict__ out, float* __restrict__ y_out,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ cn,
    const float* __restrict__ coef,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ enter,
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,
    const float* __restrict__ D,
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,
    const float* __restrict__ mimo_o,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tq, int kt, int nk
) {
    extern __shared__ float4 m3m_sm4[];
    constexpr int PP = 32 * JP;
    constexpr int KC = 64;
    constexpr int KCS = KC + 4;
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = blockIdx.y;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int nz = (M3M_CHUNK + tq - 1) / tq;
    const int tile0 = (nz - 1 - (int)blockIdx.z) * tq;
    if (tile0 >= lc) return;
    const int tn = min(tq, lc - tile0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const int rows = tn * rank;
    const int qr = (tq * rank + 7) & ~7;
    const int qrs = qr + 4;
    float* qt = reinterpret_cast<float*>(m3m_sm4);  // [N][qrs]
    float* rb = qt + ds * qrs;
    float* kts = rb;                                 // [nk][KCS]
    float* ut = kts + nk * KCS;                      // [KC][PP]
    float* gw = ut + KC * PP;                        // [KC][qrs]
    float* wts = gw + KC * qrs;                      // [tq][kt]
    float* sc = wts + tq * kt;                       // Acs, scale, gamma [3][64]
    const int rb_floats = nk * KCS + KC * PP + KC * qrs + tq * kt;
    float* stg = rb;                                 // inter: [nsp][PP + 1]
    float* yg = rb;                                  // gate: [qr][PP]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int rw0 = 4 * (tid >> 5);
    for (int s = tid; s < lc; s += blockDim.x) {
        const float* cf = coef + ((b * seq + t0 + s) * nh + h) * 4;
        sc[s] = cf[3];
        sc[M3M_CHUNK + s] = cf[2];
        sc[2 * M3M_CHUNK + s] = cf[1];
    }
    const int warp = tid >> 5, nw = blockDim.x >> 5;
    // The queries, rotated: four of a warp's rows and two state indices a
    // lane at a time, every load of the batch in flight before any turn, so
    // a thread waits one round trip a batch rather than one a value.
    for (int qw = warp; qw < qr; qw += 4 * nw) {
        for (int n0 = 0; n0 < ds; n0 += 64) {
            M3mRotIn in[4][2];
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int rw = qw + u * nw;
                const int r = rw % rank;
                const long long row = b * seq + t0 + tile0 + rw / rank;
                const float* src = cn + row * rgn + ((long long)r * ng + g) * ds;
                const float* bias = c_bias + ((long long)h * rank + r) * ds;
                const float* sr_ = sn + (row * nh + h) * na;
                const float* cr_ = cs + (row * nh + h) * na;
#pragma unroll
                for (int v = 0; v < 2; v++) {
                    const int n = n0 + lane + 32 * v;
                    in[u][v] = rw < rows && n < ds
                        ? m3m_rot_load(src, bias, n, half, na, sr_, cr_)
                        : M3mRotIn{0.0f, 0.0f, 0.0f, 0.0f};
                }
            }
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int rw = qw + u * nw;
#pragma unroll
                for (int v = 0; v < 2; v++) {
                    const int n = n0 + lane + 32 * v;
                    if (rw < qr && n < ds)
                        qt[n * qrs + rw] = rw < rows ? m3m_rot_turn(in[u][v], n, half, na) : 0.0f;
                }
            }
        }
    }
    int trow[4];
    int tmax = -1;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int rw = rw0 + i;
        trow[i] = rw < rows ? tile0 + rw / rank : -1;
        tmax = max(tmax, trow[i]);
    }
    float y[4][JP];
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < JP; j++) y[i][j] = 0.0f;
    }
    // The inter term, the entering state streamed `nsp` indices at a time.
    const float* sp = enter + (bc * nh + h) * (long long)hd * ds;
    const int nsp = min(ds, rb_floats / (PP + 1));
    for (int m0 = 0; m0 < ds; m0 += nsp) {
        const int mm = min(nsp, ds - m0);
        __syncthreads();
        for (int p0 = warp; p0 < PP; p0 += 4 * nw) {
            for (int n0 = 0; n0 < mm; n0 += 128) {
                float sv[4][4];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const float* row = sp + (long long)(p0 + u * nw) * ds + m0;
#pragma unroll
                    for (int v = 0; v < 4; v++) {
                        const int n = n0 + lane + 32 * v;
                        sv[u][v] = p0 + u * nw < hd && n < mm ? row[n] : 0.0f;
                    }
                }
#pragma unroll
                for (int u = 0; u < 4; u++) {
#pragma unroll
                    for (int v = 0; v < 4; v++) {
                        const int p = p0 + u * nw, n = n0 + lane + 32 * v;
                        if (p < PP && n < mm) stg[n * (PP + 1) + p] = sv[u][v];
                    }
                }
            }
        }
        __syncthreads();
        for (int n = 0; n < mm; n++) {
            const float4 q0 = *reinterpret_cast<const float4*>(qt + (m0 + n) * qrs + rw0);
            const float qv[4] = {q0.x, q0.y, q0.z, q0.w};
            float sv[JP];
#pragma unroll
            for (int j = 0; j < JP; j++) sv[j] = stg[n * (PP + 1) + lane + 32 * j];
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < JP; j++) y[i][j] = __fmaf_rn(qv[i], sv[j], y[i][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        if (trow[i] < 0) continue;
        const float ef = mimo_exp(sc[trow[i]]);
#pragma unroll
        for (int j = 0; j < JP; j++) y[i][j] = __fmul_rn(y[i][j], ef);
    }
    const int s_end = tile0 + tn;
    for (int s0 = 0; s0 < s_end; s0 += kt) {
        const int cols = min(kt, s_end - s0) * rank;
        __syncthreads();
        // The rank inputs, four of a warp's columns in flight together.
        for (int c0 = warp; c0 < KC; c0 += 4 * nw) {
            float xv[4][JP], mv[4][JP];
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int col = c0 + u * nw;
                const TI* xr = proj + (b * seq + t0 + s0 + col / rank) * ld_proj + di
                    + (long long)h * hd;
                const float* mx = mimo_x + ((long long)h * rank + col % rank) * hd;
#pragma unroll
                for (int j = 0; j < JP; j++) {
                    const int p = lane + 32 * j;
                    const bool live = col < cols && p < hd;
                    xv[u][j] = live ? to_f(xr[p]) : 0.0f;
                    mv[u][j] = live ? mx[p] : 0.0f;
                }
            }
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int col = c0 + u * nw;
                if (col >= KC) continue;
#pragma unroll
                for (int j = 0; j < JP; j++) {
                    const int p = lane + 32 * j;
                    ut[col * PP + p] = col < cols && p < hd ? __fmul_rn(mv[u][j], xv[u][j]) : 0.0f;
                }
            }
        }
        for (int i = tid; i < tq * kt; i += blockDim.x) {
            const int t = tile0 + i / kt, s = s0 + i % kt;
            float w = 0.0f;
            if (t < s_end && s <= t)
                w = s < t ? __fmul_rn(mimo_exp(fminf(__fsub_rn(sc[t], sc[s]), 0.0f)),
                                      sc[M3M_CHUNK + s])
                          : sc[2 * M3M_CHUNK + t];
            wts[i] = w;
        }
        float ga[4][2];
#pragma unroll
        for (int i = 0; i < 4; i++) ga[i][0] = ga[i][1] = 0.0f;
        for (int m0 = 0; m0 < ds; m0 += nk) {
            const int mm = min(nk, ds - m0);
            if (m0 > 0) __syncthreads();
            // The keys, rotated, batched like the queries.
            for (int c0 = warp; c0 < KC; c0 += 4 * nw) {
                for (int n0 = 0; n0 < mm; n0 += 64) {
                    M3mRotIn in[4][2];
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        const int col = c0 + u * nw;
                        const int r = col % rank;
                        const long long row = b * seq + t0 + s0 + col / rank;
                        const float* src = bn + row * rgn + ((long long)r * ng + g) * ds;
                        const float* bias = b_bias + ((long long)h * rank + r) * ds;
                        const float* sr_ = sn + (row * nh + h) * na;
                        const float* cr_ = cs + (row * nh + h) * na;
#pragma unroll
                        for (int v = 0; v < 2; v++) {
                            const int n = n0 + lane + 32 * v;
                            in[u][v] = col < cols && n < mm
                                ? m3m_rot_load(src, bias, m0 + n, half, na, sr_, cr_)
                                : M3mRotIn{0.0f, 0.0f, 0.0f, 0.0f};
                        }
                    }
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        const int col = c0 + u * nw;
#pragma unroll
                        for (int v = 0; v < 2; v++) {
                            const int n = n0 + lane + 32 * v;
                            if (col < KC && n < mm)
                                kts[n * KCS + col] =
                                    col < cols ? m3m_rot_turn(in[u][v], m0 + n, half, na) : 0.0f;
                        }
                    }
                }
            }
            __syncthreads();
            for (int n = 0; n < mm; n++) {
                const float4 q0 = *reinterpret_cast<const float4*>(qt + (m0 + n) * qrs + rw0);
                const float qv[4] = {q0.x, q0.y, q0.z, q0.w};
                const float k0 = kts[n * KCS + lane], k1 = kts[n * KCS + lane + 32];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    ga[i][0] = __fmaf_rn(qv[i], k0, ga[i][0]);
                    ga[i][1] = __fmaf_rn(qv[i], k1, ga[i][1]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int col = lane + 32 * j;
                float v = 0.0f;
                if (trow[i] >= 0 && col < cols)
                    v = __fmul_rn(ga[i][j], wts[(trow[i] - tile0) * kt + col / rank]);
                gw[col * qrs + rw0 + i] = v;
            }
        }
        __syncthreads();
        for (int k = 0; k < cols; k++) {
            const int s = s0 + k / rank;
            if (s > tmax) break;
            const float4 g0 = *reinterpret_cast<const float4*>(gw + k * qrs + rw0);
            const float gv[4] = {g0.x, g0.y, g0.z, g0.w};
            float uv[JP];
#pragma unroll
            for (int j = 0; j < JP; j++) uv[j] = ut[k * PP + lane + 32 * j];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (s <= trow[i]) {
#pragma unroll
                    for (int j = 0; j < JP; j++) y[i][j] = __fmaf_rn(gv[i], uv[j], y[i][j]);
                }
            }
        }
    }
    __syncthreads();
    const float dval = D[h];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        if (trow[i] < 0) continue;
        const int rw = rw0 + i, r = rw % rank;
        const long long row_t = b * seq + t0 + trow[i];
#pragma unroll
        for (int j = 0; j < JP; j++) {
            const int p = lane + 32 * j;
            if (p >= hd) continue;
            const float xv = to_f(proj[row_t * ld_proj + di + (long long)h * hd + p]);
            const float u_t = __fmul_rn(mimo_x[((long long)h * rank + r) * hd + p], xv);
            const float v = __fmaf_rn(dval, u_t, y[i][j]);
            if (y_out != nullptr) y_out[((row_t * nh + h) * rank + r) * hd + p] = v;
            yg[rw * PP + p] = v;
        }
    }
    __syncthreads();
    for (int e = warp * 32 + lane; e < tn * PP; e += blockDim.x) {
        const int tl = e / PP, p = e % PP;
        if (p >= hd) continue;
        const long long row_t = b * seq + t0 + tile0 + tl;
        const float z = to_f(proj[row_t * ld_proj + (long long)h * hd + p]);
        float o = 0.0f;
        for (int r = 0; r < rank; r++) {
            const float v = __fmul_rn(mimo_z[((long long)h * rank + r) * hd + p], z);
            const float sg = __fdiv_rn(v, __fadd_rn(1.0f, mimo_exp(-v)));
            const float term = __fmul_rn(__fmul_rn(yg[(tl * rank + r) * PP + p], sg),
                                         mimo_o[((long long)h * rank + r) * hd + p]);
            o = r == 0 ? term : __fadd_rn(o, term);
        }
        mimo_put(&out[row_t * di + (long long)h * hd + p], o);
    }
}

#define DEFINE_M3M_CHUNK_SCAN_TILES(NAME, TI)                                   \
extern "C" __global__ void __launch_bounds__(512) NAME(                         \
    TI* __restrict__ out, float* __restrict__ y_out,                            \
    const TI* __restrict__ proj, long long ld_proj,                             \
    const float* __restrict__ bn, const float* __restrict__ cn,                 \
    const float* __restrict__ coef,                                             \
    const float* __restrict__ sn, const float* __restrict__ cs,                 \
    const float* __restrict__ enter,                                            \
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,         \
    const float* __restrict__ D,                                                \
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,         \
    const float* __restrict__ mimo_o,                                           \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tq, int kt,  \
    int nk                                                                      \
) {                                                                             \
    if (hd <= 32)                                                               \
        m3m_chunk_scan_tiles_body<TI, 1>(out, y_out, proj, ld_proj, bn, cn,     \
            coef, sn, cs, enter, b_bias, c_bias, D, mimo_x, mimo_z, mimo_o,     \
            seq, nh, ng, hd, ds, rank, na, tq, kt, nk);                         \
    else if (hd <= 64)                                                          \
        m3m_chunk_scan_tiles_body<TI, 2>(out, y_out, proj, ld_proj, bn, cn,     \
            coef, sn, cs, enter, b_bias, c_bias, D, mimo_x, mimo_z, mimo_o,     \
            seq, nh, ng, hd, ds, rank, na, tq, kt, nk);                         \
    else                                                                        \
        m3m_chunk_scan_tiles_body<TI, 4>(out, y_out, proj, ld_proj, bn, cn,     \
            coef, sn, cs, enter, b_bias, c_bias, D, mimo_x, mimo_z, mimo_o,     \
            seq, nh, ng, hd, ds, rank, na, tq, kt, nk);                         \
}
DEFINE_M3M_CHUNK_SCAN_TILES(m3m_chunk_scan_tiles, float)
DEFINE_M3M_CHUNK_SCAN_TILES(m3m_chunk_scan_tiles_bf16, __nv_bfloat16)
DEFINE_M3M_CHUNK_SCAN_TILES(m3m_chunk_scan_tiles_f16, __half)

// The carried keys and input after the window: the last token's rotated
// biased keys and its raw input. Grid (B, H), threads striding R * N + P.
template <typename TI>
__device__ __forceinline__ void m3m_writeback_body(
    float* __restrict__ kst, float* __restrict__ vst,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ b_bias,
    int seq, int nh, int ng, int hd, int ds, int rank, int na
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const long long bh = (long long)b * nh + h;
    const long long row = (long long)b * seq + seq - 1;
    const float* sr = sn + (row * nh + h) * na;
    const float* cr = cs + (row * nh + h) * na;
    for (int i = threadIdx.x; i < rank * ds + hd; i += blockDim.x) {
        if (i < rank * ds) {
            const int r = i / ds, n = i % ds;
            kst[bh * rank * ds + i] =
                m3m_rot(bn + row * rgn + ((long long)r * ng + g) * ds,
                        b_bias + ((long long)h * rank + r) * ds, n, half, na, sr, cr);
        } else {
            const int p = i - rank * ds;
            vst[bh * hd + p] = to_f(proj[row * ld_proj + di + (long long)h * hd + p]);
        }
    }
}

#define DEFINE_M3M_WRITEBACK(NAME, TI)                                          \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ kst, float* __restrict__ vst,                           \
    const TI* __restrict__ proj, long long ld_proj,                             \
    const float* __restrict__ bn,                                               \
    const float* __restrict__ sn, const float* __restrict__ cs,                 \
    const float* __restrict__ b_bias,                                           \
    int seq, int nh, int ng, int hd, int ds, int rank, int na                   \
) {                                                                             \
    m3m_writeback_body<TI>(kst, vst, proj, ld_proj, bn, sn, cs, b_bias, seq,    \
        nh, ng, hd, ds, rank, na);                                              \
}
DEFINE_M3M_WRITEBACK(m3m_writeback, float)
DEFINE_M3M_WRITEBACK(m3m_writeback_bf16, __nv_bfloat16)
DEFINE_M3M_WRITEBACK(m3m_writeback_f16, __half)
