// Mamba-3 MIMO on the GPU, the chunked path: a window in chunks of
// M3M_CHUNK tokens, the chunks in parallel.
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
// The pieces: m3m_prep_tok, m3m_prep_chain and m3m_prep_sincos (step
// scalars and angles per token, the angles in the sequential path's own
// arithmetic), m3m_chunk_state and m3m_state_pass (or
// m3m_chunk_state_chain, the two in one launch),
// m3m_chunk_scan (the outputs and the gate), m3m_writeback (the carried
// keys and input). Every value is one thread's chain in a fixed order; the
// bits do not depend on the batch or the grid.
//
// With a packing (`seq_idx`, see _varlen.cuh) nothing crosses from one
// sequence into the next: a sequence's last token keeps gamma alone as its
// scale, the angles and the chunk's running decay sum restart from zero at
// its first token, a chunk's contribution comes from its last sequence
// only, the state passes into a chunk only when no sequence starts in the
// one before, the entering state reaches only the tokens that continue the
// sequence before the chunk, and a token reads only the earlier tokens of
// its own sequence. Masked terms are skipped or selected, never multiplied
// by zero, so a sequence's overflow stays in it.
//
// Composed after mamba3_mimo.cu in the MIMO module.

// Tokens per chunk: the host compiles the module at the rank's length
// (`chunk_len` in mamba3_mimo/gpu/kernels.rs), 64 when nothing sets it.
#ifndef M3M_CHUNK
#define M3M_CHUNK 64
#endif

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

// A rank row of keys or queries: the group's normalised row, the head's
// bias and the token's sines and cosines, the inputs of m3m_rot.
struct M3mRotRow {
    const float* src;
    const float* bias;
    const float* sr;
    const float* cr;
};

// Rotated rank rows into shared memory: rows i = first, first + step, ...
// below `all` (those from `live` on as zeros) at the state indices m0 + n,
// n below mm, put(i, n, value) storing each. Four rows and two indices a
// lane go at a time, every load of the batch in flight before any is
// turned, so a thread waits one round trip a batch rather than one a
// value.
template <class ROWF, class PUTF>
__device__ __forceinline__ void m3m_stage_rot(
    int first, int step, int live, int all, int m0, int mm, int half, int na,
    ROWF row_of, PUTF put
) {
    const int lane = threadIdx.x & 31;
    for (int i0 = first; i0 < all; i0 += 4 * step) {
        for (int n0 = 0; n0 < mm; n0 += 64) {
            M3mRotIn in[4][2];
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int i = i0 + u * step;
                const M3mRotRow rr = row_of(i);
#pragma unroll
                for (int v = 0; v < 2; v++) {
                    const int n = n0 + lane + 32 * v;
                    in[u][v] = i < live && n < mm
                        ? m3m_rot_load(rr.src, rr.bias, m0 + n, half, na, rr.sr, rr.cr)
                        : M3mRotIn{0.0f, 0.0f, 0.0f, 0.0f};
                }
            }
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int i = i0 + u * step;
#pragma unroll
                for (int v = 0; v < 2; v++) {
                    const int n = n0 + lane + 32 * v;
                    if (i < all && n < mm)
                        put(i, n, i < live ? m3m_rot_turn(in[u][v], m0 + n, half, na) : 0.0f);
                }
            }
        }
    }
}

// The rank inputs u = Psi_r x of rows i (a token and a rank) into shared
// memory, batched like m3m_stage_rot: x_of(i) is the row's input in proj
// and mx_of(i) its Psi_r row; the channels below `width` are stored, those
// from hd on (and the rows from `live` on) as zeros.
template <typename TI, class XF, class MF, class PUTF>
__device__ __forceinline__ void m3m_stage_u(
    int first, int step, int live, int all, int hd, int width, XF x_of, MF mx_of, PUTF put
) {
    const int lane = threadIdx.x & 31;
    for (int i0 = first; i0 < all; i0 += 4 * step) {
        for (int p0 = 0; p0 < width; p0 += 128) {
            float xv[4][4], mv[4][4];
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int i = i0 + u * step;
                const TI* xr = x_of(i);
                const float* mx = mx_of(i);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int p = p0 + lane + 32 * j;
                    const bool on = i < live && p < hd;
                    xv[u][j] = on ? to_f(xr[p]) : 0.0f;
                    mv[u][j] = on ? mx[p] : 0.0f;
                }
            }
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int i = i0 + u * step;
                if (i >= all) continue;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int p = p0 + lane + 32 * j;
                    if (p < width)
                        put(i, p, i < live && p < hd ? __fmul_rn(mv[u][j], xv[u][j]) : 0.0f);
                }
            }
        }
    }
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

// The step scalars and angles of every token of the window, in three
// launches. coef [B, T, H, 4] = (dt, gamma, scale, Acs); sn, cs
// [B, T, H, na]; e0 [B, H] = dt_0 lam_c_0. The angle state ends advanced
// over the window. Every value is the one a single thread walking the
// tokens in order computes; only the work is spread:
//   1. m3m_prep_tok, per token: dt, gamma = lam dt, scale = gamma + the
//      next token's lead dt lam_c (recomputed rather than waited for; gamma
//      alone when the next token starts a packed sequence) and A dt; per
//      chunk: Acs, the running sum of A dt from zero at the chunk start (and
//      at every packed sequence's start), in token order; per (token,
//      angle): the increment tanh_pi dt, parked in sn;
//   2. m3m_prep_chain, per angle: the serial wrap chain over the window
//      (from zero at every packed sequence's start), the angles parked in
//      sn;
//   3. m3m_prep_sincos, per (token, angle): the sines and cosines.
// The serial chains are the only part a long window cannot spread, so the
// rest runs on blocks of its own rather than waiting at a barrier for them.
template <typename TI>
__device__ __forceinline__ void m3m_tok_dt_lead(
    float& dt, float& lead, const TI* pr, const float* dt_bias, int nh, int heads, int h
) {
    float spin = __fadd_rn(to_f(pr[heads + h]), dt_bias[h]);
    dt = spin > 20.0f ? spin : log1pf(mimo_exp(spin));
    lead = __fmul_rn(dt, mimo_sigmoid(-to_f(pr[heads + 2 * nh + h])));
}

// Tokens of a block of m3m_prep_tok: a multiple of every chunk length (a
// power of two up to 64), so a tile holds whole chunks.
#define M3M_PREP_TILE 64

// The step scalars, the chunks' running sums and the angle increments of a
// tile of M3M_PREP_TILE tokens of one (sequence, head). Grid
// (B * ceil(T / M3M_PREP_TILE), H), block 128.
template <typename TI>
__device__ __forceinline__ void m3m_prep_tok_body(
    float* __restrict__ coef, float* __restrict__ sn, float* __restrict__ e0,
    const TI* __restrict__ proj, long long ld_proj, const float* __restrict__ dt_bias,
    int seq, int nh, int heads, int ang_col, int na, float a_floor,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    __shared__ float s_dt[M3M_PREP_TILE];
    __shared__ float s_adt[M3M_PREP_TILE];
    const int ntile = (seq + M3M_PREP_TILE - 1) / M3M_PREP_TILE;
    const long long b = blockIdx.x / ntile;
    const int t0 = (int)(blockIdx.x % ntile) * M3M_PREP_TILE;
    const int tn = min(M3M_PREP_TILE, seq - t0);
    const int h = y_first + blockIdx.y;
    const long long row0 = b * seq + t0;
    const long long cstr = (long long)nh * 4;
    float* const c_rows = coef + (row0 * nh + h) * 4;
    for (int i = threadIdx.x; i < tn; i += blockDim.x) {
        const int t = t0 + i;
        const TI* pr = proj + (row0 + i) * ld_proj;
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
        float* c = c_rows + i * cstr;
        c[0] = dt;
        c[1] = gamma;
        c[2] = scale;
        s_dt[i] = dt;
        s_adt[i] = __fmul_rn(a, dt);
        if (t == 0) e0[b * nh + h] = lead;
    }
    __syncthreads();
    for (int c0 = threadIdx.x * M3M_CHUNK; c0 < tn; c0 += blockDim.x * M3M_CHUNK) {
        const int c1 = min(tn, c0 + M3M_CHUNK);
        float acs = 0.0f;
        for (int i = c0; i < c1; i++) {
            const bool fresh = seq_idx != nullptr && vl_starts(seq_idx, b, seq, t0 + i);
            acs = __fadd_rn(fresh ? 0.0f : acs, s_adt[i]);
            c_rows[i * cstr + 3] = acs;
        }
    }
    const long long astr = (long long)nh * na;
    float* const sn_rows = sn + (row0 * nh + h) * na;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll 4
    for (int i = warp; i < tn; i += nw) {
        const TI* pr = proj + (row0 + i) * ld_proj;
        for (int a = lane; a < na; a += 32)
            sn_rows[i * astr + a] = __fmul_rn(mimo_tanh_pi(pr, ang_col, a), s_dt[i]);
    }
}

#define DEFINE_M3M_PREP_TOK(NAME, TI)                                          \
extern "C" __global__ void __launch_bounds__(128) NAME(                        \
    float* __restrict__ coef, float* __restrict__ sn, float* __restrict__ e0,  \
    const TI* __restrict__ proj, long long ld_proj,                            \
    const float* __restrict__ dt_bias,                                         \
    int seq, int nh, int heads, int ang_col, int na, float a_floor,            \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3m_prep_tok_body<TI>(coef, sn, e0, proj, ld_proj, dt_bias, seq, nh,       \
        heads, ang_col, na, a_floor, y_first, seq_idx);                        \
}
DEFINE_M3M_PREP_TOK(m3m_prep_tok, float)
DEFINE_M3M_PREP_TOK(m3m_prep_tok_bf16, __nv_bfloat16)
DEFINE_M3M_PREP_TOK(m3m_prep_tok_f16, __half)

// One angle's serial wrap chain over a row's tokens: `col` holds the
// increments (stride `astr`) and leaves holding the angles; the angle the
// row carries in is `theta` and the one it carries out is returned. The
// chain loads the next AB increments while it walks the current ones, so a
// step waits on one add, not on memory; a whole batch walks without bounds
// checks, keeping its angles in the batch's registers and storing them after
// it, so a step of a thread walking alone holds the add, the two compares
// and the branch. PACKED restarts the angle from zero at every token that
// starts a sequence.
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
        const float* nx = col + (long long)(t0 + AB) * astr;
        if (t0 + 2 * AB <= seq) {
#pragma unroll
            for (int i = 0; i < AB; i++) {
                nxt[i] = nx[i * astr];
                fnxt[i] = PACKED && vl_starts(seq_idx, b, seq, t0 + AB + i);
            }
        } else {
#pragma unroll
            for (int i = 0; i < AB; i++) {
                const int t = t0 + AB + i;
                nxt[i] = t < seq ? nx[i * astr] : 0.0f;
                fnxt[i] = PACKED && t < seq && vl_starts(seq_idx, b, seq, t);
            }
        }
        float* out = col + (long long)t0 * astr;
        if (t0 + AB <= seq) {
#pragma unroll
            for (int i = 0; i < AB; i++) {
                theta = mimo_wrap_add(PACKED && fcur[i] ? 0.0f : theta, cur[i]);
                cur[i] = theta;
            }
#pragma unroll
            for (int i = 0; i < AB; i++) out[i * astr] = cur[i];
        } else {
#pragma unroll
            for (int i = 0; i < AB; i++) {
                if (t0 + i < seq) {
                    theta = mimo_wrap_add(PACKED && fcur[i] ? 0.0f : theta, cur[i]);
                    out[i * astr] = theta;
                }
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

// The serial wrap chains of the window, a thread per (sequence, head,
// angle) over `chains` = B H na of them: the angle the row carries in from
// `ang`, the increments m3m_prep_tok parked in sn turned into the angles,
// and the angle carried out back into `ang`. A warp walks its chains in
// lockstep and takes a step's double-precision wrap whenever any of them
// needs it, so a block holds few chains (the host's PREP_CHAIN_LANES): the
// chains are serial however they are laid out, and with 32 to a warp the
// wrap of one or another had cost half the steps of every chain. Grid
// ceil(chains / block).
extern "C" __global__ void __launch_bounds__(32) m3m_prep_chain(
    float* __restrict__ sn, float* __restrict__ ang,
    int seq, int nh, int na, long long chains,
    const int* __restrict__ seq_idx
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chains) return;
    const long long bh = i / na;
    const int a = (int)(i - bh * na);
    const long long b = bh / nh;
    const int h = (int)(bh - b * nh);
    const long long astr = (long long)nh * na;
    float* const col = sn + ((b * seq) * nh + h) * na + a;
    const float theta = ang[i];
    ang[i] = seq_idx != nullptr
        ? m3m_angle_chain<true>(theta, col, astr, seq, seq_idx, b)
        : m3m_angle_chain<false>(theta, col, astr, seq, seq_idx, b);
}

// The sines and cosines of the `n` = B T H na angles m3m_prep_chain left in
// sn: sn keeps the sines, cs takes the cosines. Grid-stride, block 256.
extern "C" __global__ void __launch_bounds__(256) m3m_prep_sincos(
    float* __restrict__ sn, float* __restrict__ cs, long long n
) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step) {
        const float th = sn[i];
        sincosf(th, &sn[i], &cs[i]);
    }
}

// Each chunk's own contribution to the state, from zero:
//   C[p][n] = sum_s sum_r (e^min(Acs_last - Acs_s, 0) scale_s) u_s[r][p] k_s[r][n]
// (s, then r, ascending). One block per (chunk, head, n-tile, tile of `pt`
// channels); thread (p, n-in-tile). The chunk's rotated keys for the tile's
// n and its tile's inputs are staged in shared memory `sl` tokens at a time
// (the whole chunk when it fits): sl * R * nt + sl * pt + sl floats. The
// tile sizes change what a pass stages, never the order of a thread's sum.
// Grid (B * nc, H, ceil(N / nt) * ceil(P / pt)), the channel tile fastest,
// block nt * pt threads.
template <typename TI>
__device__ __forceinline__ void m3m_chunk_state_body(
    float* __restrict__ contrib,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ coef,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ b_bias, const float* __restrict__ mimo_x,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int nt, int sl, int pt,
    int y_first, /* this run's first y block */
    int z_first, /* this run's first z block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m3m_sm[];
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
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    float* kt = m3m_sm;                // [sl][R][nt]
    float* xt = kt + sl * rank * nt;   // [sl][pt]
    float* wt = xt + sl * pt;          // [sl]
    const int tid = threadIdx.x;
    const int pl = tid % pt, nl = tid / pt;
    const int p = p0 + pl;
    const bool active = nl < nt && n0 + nl < ds && pl < pc;
    const float acs_last = coef[(((b * seq + t0 + lc - 1) * nh) + h) * 4 + 3];
    // Only the chunk's last sequence reaches the next chunk.
    const int s_first = vl_seq_first(seq_idx, b, seq, t0, t0 + lc - 1) - t0;
    float acc = 0.0f;
    for (int s0 = s_first; s0 < lc; s0 += sl) {
        const int sn_ = min(sl, lc - s0);
        if (s0 > s_first) __syncthreads();
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
        for (int i = tid; i < sn_ * pt; i += blockDim.x) {
            const int s = i / pt, pp = i % pt;
            xt[i] = pp < pc ? to_f(proj[(b * seq + t0 + s0 + s) * ld_proj + di
                                        + (long long)h * hd + p0 + pp])
                            : 0.0f;
        }
        for (int s = tid; s < sn_; s += blockDim.x) {
            const float* cf = coef + ((b * seq + t0 + s0 + s) * nh + h) * 4;
            wt[s] = __fmul_rn(mimo_exp(fminf(__fsub_rn(acs_last, cf[3]), 0.0f)), cf[2]);
        }
        __syncthreads();
        if (active) {
            for (int s = 0; s < sn_; s++) {
                const float wx = __fmul_rn(wt[s], xt[s * pt + pl]);
                for (int r = 0; r < rank; r++) {
                    const float u = __fmul_rn(wx, mimo_x[((long long)h * rank + r) * hd + p]);
                    acc = __fmaf_rn(u, kt[(s * rank + r) * nt + nl], acc);
                }
            }
        }
    }
    if (active) contrib[((bc * nh + h) * hd + p) * (long long)ds + n0 + nl] = acc;
}

#define DEFINE_M3M_CHUNK_STATE(NAME, TI)                                       \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ contrib,                                               \
    const TI* __restrict__ proj, long long ld_proj,                            \
    const float* __restrict__ bn, const float* __restrict__ coef,              \
    const float* __restrict__ sn, const float* __restrict__ cs,                \
    const float* __restrict__ b_bias, const float* __restrict__ mimo_x,        \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int nt, int sl, \
    int pt,                                                                    \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3m_chunk_state_body<TI>(contrib, proj, ld_proj, bn, coef, sn, cs, b_bias, \
        mimo_x, seq, nh, ng, hd, ds, rank, na, nt, sl, pt, y_first, z_first,   \
        seq_idx);                                                              \
}
DEFINE_M3M_CHUNK_STATE(m3m_chunk_state, float)
DEFINE_M3M_CHUNK_STATE(m3m_chunk_state_bf16, __nv_bfloat16)
DEFINE_M3M_CHUNK_STATE(m3m_chunk_state_f16, __half)

// A chunk's contributions to a tile of the state as register tiles: a warp
// takes 4 channels p and a lane the state indices lane + 32 j of the tile
// (JC of them), so a step of (s, r) loads one float4 of u and JC keys for
// 4 JC products. The tile is a group of `pg` channels from pb (a multiple
// of 4: the whole head when the kernel's registers let one block hold it,
// else groups sized so that they do) and the 32 JC state indices from n0.
// The block computes u = (w_s x_s[p]) Psi_r[p] for its channels and the
// rotated keys once and stages them `sl` tokens at a time in `sm`:
// ut [sl R][pg] (the channels past the head zero), kt [sl R][32 JC],
// wt [sl]. Every output keeps the chain above, acc = 0 then fma(u, k, acc)
// over s ascending and r ascending, across the passes, whatever the group.
// The whole block enters; a caller that stages again waits at a barrier
// first, since other warps may still read the last pass.
template <int JC, typename TI>
__device__ __forceinline__ void m3m_state_tile_chunk(
    float (&acc)[4][JC], float* __restrict__ sm,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ coef,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ b_bias, const float* __restrict__ mimo_x,
    long long b, int c, int h, int n0, int pb,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int sl, int pg,
    const int* __restrict__ seq_idx
) {
    constexpr int NT = 32 * JC;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    float* ut = sm;                   // [sl R][pg]
    float* kt = ut + sl * rank * pg;  // [sl R][NT]
    float* wt = kt + sl * rank * NT;  // [sl]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5, nw = blockDim.x >> 5;
    const int p0 = 4 * warp;
    const float acs_last = coef[(((b * seq + t0 + lc - 1) * nh) + h) * 4 + 3];
    // Only the chunk's last sequence reaches the next chunk.
    const int s_first = vl_seq_first(seq_idx, b, seq, t0, t0 + lc - 1) - t0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < JC; j++) acc[i][j] = 0.0f;
    }
    for (int s0 = s_first; s0 < lc; s0 += sl) {
        const int sn_ = min(sl, lc - s0);
        if (s0 > s_first) __syncthreads();
        for (int s = tid; s < sn_; s += blockDim.x) {
            const float* cf = coef + ((b * seq + t0 + s0 + s) * nh + h) * 4;
            wt[s] = __fmul_rn(mimo_exp(fminf(__fsub_rn(acs_last, cf[3]), 0.0f)), cf[2]);
        }
        const int kd = sn_ * rank;
        // The keys, rotated: a warp's rows two at a time, every load of the
        // pair in flight before any value is turned.
        for (int sr0 = warp; sr0 < kd; sr0 += 2 * nw) {
            M3mRotIn in[2][JC];
#pragma unroll
            for (int u = 0; u < 2; u++) {
                const int sr = sr0 + u * nw;
                const int r = sr % rank;
                const long long row = b * seq + t0 + s0 + sr / rank;
                const float* src = bn + row * rgn + ((long long)r * ng + g) * ds;
                const float* bias = b_bias + ((long long)h * rank + r) * ds;
                const float* sr_ = sn + (row * nh + h) * na;
                const float* cr_ = cs + (row * nh + h) * na;
#pragma unroll
                for (int j = 0; j < JC; j++) {
                    const int n = n0 + lane + 32 * j;
                    in[u][j] = sr < kd && n < ds
                        ? m3m_rot_load(src, bias, n, half, na, sr_, cr_)
                        : M3mRotIn{0.0f, 0.0f, 0.0f, 0.0f};
                }
            }
#pragma unroll
            for (int u = 0; u < 2; u++) {
                const int sr = sr0 + u * nw;
                if (sr >= kd) continue;
#pragma unroll
                for (int j = 0; j < JC; j++) {
                    const int n = n0 + lane + 32 * j;
                    kt[sr * NT + lane + 32 * j] = n < ds ? m3m_rot_turn(in[u][j], n, half, na) : 0.0f;
                }
            }
        }
        __syncthreads();
        // The rank inputs, four of a warp's rows at a time, their loads in
        // flight together.
        const int pn = hd - pb;
        for (int sr0 = warp; sr0 < kd; sr0 += 4 * nw) {
            for (int p0s = 0; p0s < pg; p0s += 64) {
                float xv[4][2], mv[4][2];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const int sr = sr0 + u * nw;
                    const int s = sr / rank, r = sr % rank;
                    const TI* xr =
                        proj + (b * seq + t0 + s0 + s) * ld_proj + di + (long long)h * hd + pb;
                    const float* mx = mimo_x + ((long long)h * rank + r) * hd + pb;
#pragma unroll
                    for (int v = 0; v < 2; v++) {
                        const int pp = p0s + lane + 32 * v;
                        const bool on = sr < kd && pp < pg && pp < pn;
                        xv[u][v] = on ? to_f(xr[pp]) : 0.0f;
                        mv[u][v] = on ? mx[pp] : 0.0f;
                    }
                }
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const int sr = sr0 + u * nw;
                    if (sr >= kd) continue;
                    const float w = wt[sr / rank];
#pragma unroll
                    for (int v = 0; v < 2; v++) {
                        const int pp = p0s + lane + 32 * v;
                        if (pp < pg)
                            ut[sr * pg + pp] =
                                pp < pn ? __fmul_rn(__fmul_rn(w, xv[u][v]), mv[u][v]) : 0.0f;
                    }
                }
            }
        }
        __syncthreads();
        for (int k = 0; k < kd; k++) {
            const float4 lv = lds_f32x4(ut + k * pg + p0);
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
}

// The hand-off between the blocks of a chain: the producer's stores are
// fenced before its flag rises, and the waiting block reads them after the
// flag through the cache-global path, past an L1 that is not coherent.
__device__ __forceinline__ void m3m_raise(unsigned* flag) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;\n" :: "l"(flag), "r"(1U) : "memory");
}

__device__ __forceinline__ void m3m_await(const unsigned* flag) {
    unsigned value;
    do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(value) : "l"(flag) : "memory");
    } while (value == 0U);
}

// The chunk states and their passing in one launch, where the register
// tiles fit: a block takes one chunk of one (sequence, head) and a tile of
// its state, forms the chunk's contribution C_c as m3m_state_tile_chunk
// does, then takes the state entering the chunk from the block of the chunk
// before and passes it on as m3m_state_pass does, element by element:
// S = e^Acs_last S + C_c, or S = C_c when a packed sequence starts in chunk
// c, into the next chunk's entering state, or into the carried state `ssm`
// after the last chunk. The first chunk's block forms the window's entering
// state itself, S_init + e0 sum_r (Psi_r v) kp_r (r ascending). Every value
// is the one the two kernels give, while the contributions never leave the
// multiprocessor.
//
// A block takes its place from a ticket (`chain[0]`), so the blocks of a
// chunk run after those of the chunk before, whatever order the hardware
// starts them in: a block only waits on a block that is already running,
// and the chunks' contributions overlap while the states pass. The hand-off
// is the next chunk's entering state in global memory, fenced, then a flag
// per chunk and tile (`chain[1 + c * per + unit]`); the host zeroes `chain`
// before the launch. Grid B * nc * H * tiles, one dimension, the tiles of a
// chunk together; block 8 pg threads.
template <int JC, typename TI>
__device__ __forceinline__ void m3m_chunk_state_chain_body(
    float* __restrict__ enter, float* __restrict__ ssm, unsigned* __restrict__ chain,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ coef,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ b_bias, const float* __restrict__ mimo_x,
    const float* __restrict__ e0, const float* __restrict__ kst,
    const float* __restrict__ vst,
    int batch, int seq, int nh, int ng, int hd, int ds, int rank, int na, int sl, int pg,
    const int* __restrict__ seq_idx
) {
    extern __shared__ float4 m3m_sm4[];
    __shared__ unsigned place;
    constexpr int NT = 32 * JC;
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const int nz = (ds + NT - 1) / NT;
    const int tiles = nz * ((hd + pg - 1) / pg);
    const long long per = (long long)batch * nh * tiles;
    const int tid = threadIdx.x;
    if (tid == 0) place = atomicAdd(chain, 1U);
    __syncthreads();
    const int c = (int)(place / per);
    const long long unit = place - c * per;
    const int zi = (int)(unit % tiles);
    const long long bh = unit / tiles;
    const int h = (int)(bh % nh);
    const long long b = bh / nh;
    const int grp = zi / nz;
    const int n0 = (zi - grp * nz) * NT;
    const int pb = grp * pg;
    float acc[4][JC];
    m3m_state_tile_chunk<JC, TI>(acc, reinterpret_cast<float*>(m3m_sm4), proj, ld_proj, bn,
                                 coef, sn, cs, b_bias, mimo_x, b, c, h, n0, pb, seq, nh, ng,
                                 hd, ds, rank, na, sl, pg, seq_idx);
    const int lane = tid & 31;
    const int p0 = pb + 4 * (tid >> 5);
    const long long pn = (long long)hd * ds;
    unsigned* flags = chain + 1;
    float st[4][JC];
    if (c == 0) {
        float* in = enter + (b * nc * nh + h) * pn;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int p = p0 + i;
            const float v = p < hd ? vst[bh * hd + p] : 0.0f;
#pragma unroll
            for (int j = 0; j < JC; j++) {
                const int n = n0 + lane + 32 * j;
                st[i][j] = 0.0f;
                if (p >= hd || n >= ds) continue;
                float adj = 0.0f;
                for (int r = 0; r < rank; r++) {
                    const float up = __fmul_rn(mimo_x[((long long)h * rank + r) * hd + p], v);
                    adj = __fmaf_rn(up, kst[(bh * rank + r) * ds + n], adj);
                }
                st[i][j] = __fmaf_rn(e0[bh], adj, ssm[bh * pn + (long long)p * ds + n]);
                in[(long long)p * ds + n] = st[i][j];
            }
        }
    } else {
        if (tid == 0) m3m_await(flags + (c - 1) * per + unit);
        __syncthreads();
        const float* in = enter + ((b * nc + c) * nh + h) * pn;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int p = p0 + i;
#pragma unroll
            for (int j = 0; j < JC; j++) {
                const int n = n0 + lane + 32 * j;
                st[i][j] = p < hd && n < ds ? __ldcg(in + (long long)p * ds + n) : 0.0f;
            }
        }
    }
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const float decay = mimo_exp(coef[((b * seq + t0 + lc - 1) * nh + h) * 4 + 3]);
    const bool passes = vl_passes(seq_idx, b, seq, t0, lc);
    float* out = c + 1 < nc ? enter + ((b * nc + c + 1) * nh + h) * pn : ssm + bh * pn;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int p = p0 + i;
#pragma unroll
        for (int j = 0; j < JC; j++) {
            const int n = n0 + lane + 32 * j;
            if (p < hd && n < ds)
                out[(long long)p * ds + n] =
                    passes ? __fmaf_rn(decay, st[i][j], acc[i][j]) : acc[i][j];
        }
    }
    if (c + 1 < nc) {
        __threadfence();
        __syncthreads();
        if (tid == 0) m3m_raise(flags + c * per + unit);
    }
}

#define DEFINE_M3M_CHUNK_STATE_CHAIN(NAME, TI)                                 \
extern "C" __global__ void __launch_bounds__(512) NAME(                        \
    float* __restrict__ enter, float* __restrict__ ssm,                        \
    unsigned* __restrict__ chain,                                              \
    const TI* __restrict__ proj, long long ld_proj,                            \
    const float* __restrict__ bn, const float* __restrict__ coef,              \
    const float* __restrict__ sn, const float* __restrict__ cs,                \
    const float* __restrict__ b_bias, const float* __restrict__ mimo_x,        \
    const float* __restrict__ e0, const float* __restrict__ kst,               \
    const float* __restrict__ vst,                                             \
    int batch, int seq, int nh, int ng, int hd, int ds, int rank, int na,      \
    int jc, int sl, int pg,                                                    \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    if (jc == 1)                                                               \
        m3m_chunk_state_chain_body<1, TI>(enter, ssm, chain, proj, ld_proj, bn, \
            coef, sn, cs, b_bias, mimo_x, e0, kst, vst, batch, seq, nh, ng, hd, \
            ds, rank, na, sl, pg, seq_idx);                                    \
    else if (jc == 2)                                                          \
        m3m_chunk_state_chain_body<2, TI>(enter, ssm, chain, proj, ld_proj, bn, \
            coef, sn, cs, b_bias, mimo_x, e0, kst, vst, batch, seq, nh, ng, hd, \
            ds, rank, na, sl, pg, seq_idx);                                    \
    else                                                                       \
        m3m_chunk_state_chain_body<4, TI>(enter, ssm, chain, proj, ld_proj, bn, \
            coef, sn, cs, b_bias, mimo_x, e0, kst, vst, batch, seq, nh, ng, hd, \
            ds, rank, na, sl, pg, seq_idx);                                    \
}
DEFINE_M3M_CHUNK_STATE_CHAIN(m3m_chunk_state_chain, float)
DEFINE_M3M_CHUNK_STATE_CHAIN(m3m_chunk_state_chain_bf16, __nv_bfloat16)
DEFINE_M3M_CHUNK_STATE_CHAIN(m3m_chunk_state_chain_f16, __half)

// The states entering the chunks: S = S_init + e0 sum_r (Psi_r v) kp_r
// (r ascending), then per chunk enter[c] = S, S = e^Acs_last S + C_c, or
// S = C_c when a packed sequence starts in chunk c. The carried state ends
// as the state after the window. One thread per (b, h, p, n). Grid
// (B * H, ceil(P * N / blockDim.x)).
extern "C" __global__ void m3m_state_pass(
    float* __restrict__ enter,
    float* __restrict__ ssm,
    const float* __restrict__ contrib,
    const float* __restrict__ coef, const float* __restrict__ e0,
    const float* __restrict__ kst, const float* __restrict__ vst,
    const float* __restrict__ mimo_x,
    int batch, int seq, int nh, int hd, int ds, int rank,
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
        st = vl_passes(seq_idx, b, seq, c * M3M_CHUNK, lc)
            ? __fmaf_rn(mimo_exp(acs_last), st, contrib[idx]) : contrib[idx];
    }
    ssm[bh * pn + e] = st;
}

// The outputs. One block per (chunk, head, token tile, tile of `pt`
// channels); thread (t, p) of the tile computes y_t[r] for every rank and
// the gated row:
//   y[r] = e^Acs_t (q_t[r] . S_enter[p])
//        + sum_{s<=t} (G[t,r,s,r'] w(t,s)) u_s[r'][p]        (s, then r')
//        + D u_t[r][p]
// with G[t,r,s,r'] = q_t[r] . k_s[r'] (n ascending) and w(t,s) =
// e^min(Acs_t - Acs_s, 0) scale_s off the diagonal, gamma_t on it. The
// queries of the tile stay staged; the keys come in s-tiles of `st` with
// their G block. Shared memory: tt * R * N (queries) + st * R * (N + 1)
// (keys, rows padded) + tt * R * st * R (G) + st * pt (inputs) floats. The
// tile sizes change what a pass stages, never the order of a thread's sum.
// Grid (B * nc, H, ceil(M3M_CHUNK / tt) * ceil(P / pt)), the channel
// tile fastest, block tt * pt threads. Every channel tile of a token tile
// computes the same G. `y_out` (may be null) receives the readouts y_t[r][p] with D, [B, T, H, R, P], what the
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
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tt, int st, int pt,
    int y_first, /* this run's first y block */
    int z_first, /* this run's first z block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m3m_sm[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int nct = (hd + pt - 1) / pt;
    const int p0 = (int)((z_first + blockIdx.z) % nct) * pt;
    const int pc = min(pt, hd - p0);
    const int tile0 = (int)((z_first + blockIdx.z) / nct) * tt;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    if (tile0 >= lc) return;
    const int tn = min(tt, lc - tile0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    constexpr int R = RK;
    const int half = ds / 2;
    const int ST = st;
    // Key rows padded to N + 1: the block's G entries read one key row per
    // thread, which at a row stride of N (a multiple of 32) would put a
    // warp's reads in one bank.
    const int kld = ds + 1;
    float* qs = m3m_sm;                       // [tt][R][N]
    float* ks = qs + tt * rank * ds;          // [ST][R][N + 1]
    float* gs = ks + ST * rank * kld;         // [tt*R][ST*R]
    float* xs = gs + tt * rank * ST * rank;   // [ST][pt]
    const int tid = threadIdx.x;
    const int tl = tid / pt, pl = tid % pt;
    const int p = p0 + pl;
    const bool live = tl < tn && pl < pc;
    // The tile's rotated queries.
    for (int i = tid; i < tn * rank * ds; i += blockDim.x) {
        const int tq = i / (rank * ds), r = (i / ds) % rank, n = i % ds;
        const long long row = b * seq + t0 + tile0 + tq;
        qs[i] = m3m_rot(cn + row * rgn + ((long long)r * ng + g) * ds,
                        c_bias + ((long long)h * rank + r) * ds, n, half, na,
                        sn + (row * nh + h) * na, cs + (row * nh + h) * na);
    }
    __syncthreads();
    float y[RK];
    float u_t[RK];
    float px[RK];
    const int t = tile0 + tl;
    const long long row_t = b * seq + t0 + t;
    float acs_t = 0.0f, gamma_t = 0.0f;
    // The first token of t's sequence in the chunk: t reads no earlier one.
    int first_t = 0;
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
        const VlSpan span = vl_span(seq_idx, b, seq, t0, lc, t);
        first_t = span.first;
        // Every rank's chain over n ascending, the entering row read once,
        // when t continues the sequence the state entering the chunk holds.
        if (span.from_before) {
            const float* q = qs + tl * rank * ds;
            for (int n = 0; n < ds; n++) {
                const float sv = sp[n];
                _Pragma("unroll") for (int r = 0; r < R; r++)
                    y[r] = __fmaf_rn(q[r * ds + n], sv, y[r]);
            }
            _Pragma("unroll") for (int r = 0; r < R; r++) y[r] = __fmul_rn(y[r], ef);
        }
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
        for (int i = tid; i < sn_ * pt; i += blockDim.x) {
            const int sk = i / pt, pp = i % pt;
            xs[i] = pp < pc ? to_f(proj[(b * seq + t0 + s0 + sk) * ld_proj + di
                                        + (long long)h * hd + p0 + pp])
                            : 0.0f;
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
            for (int sk = max(0, first_t - s0); sk < sn_; sk++) {
                const int s = s0 + sk;
                if (s > t) break;
                const long long row_s = b * seq + t0 + s;
                const float* cf = coef + (row_s * nh + h) * 4;
                const float w = s < t
                    ? __fmul_rn(mimo_exp(fminf(__fsub_rn(acs_t, cf[3]), 0.0f)), cf[2])
                    : gamma_t;
                const float xsv = xs[sk * pt + pl];
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
        float pz[RK], po[RK];
        _Pragma("unroll") for (int r = 0; r < R; r++) {
            pz[r] = mimo_z[((long long)h * rank + r) * hd + p];
            po[r] = mimo_o[((long long)h * rank + r) * hd + p];
        }
        const float z = to_f(proj[row_t * ld_proj + (long long)h * hd + p]);
        mimo_put(&out[row_t * di + (long long)h * hd + p], mimo_tok_gate<RK>(y, pz, po, z, rank));
    }
}

#define DEFINE_M3M_CHUNK_SCAN(NAME, TI)                                        \
extern "C" __global__ void NAME(                                               \
    TI* __restrict__ out, float* __restrict__ y_out,                           \
    const TI* __restrict__ proj, long long ld_proj,                            \
    const float* __restrict__ bn, const float* __restrict__ cn,                \
    const float* __restrict__ coef,                                            \
    const float* __restrict__ sn, const float* __restrict__ cs,                \
    const float* __restrict__ enter,                                           \
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,        \
    const float* __restrict__ D,                                               \
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,        \
    const float* __restrict__ mimo_o,                                          \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tt, int st, \
    int pt,                                                                    \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3m_chunk_scan_body<TI, MIMO_RANK>(out, y_out, proj, ld_proj, bn, cn,      \
        coef, sn, cs, enter, b_bias, c_bias, D, mimo_x, mimo_z, mimo_o, seq,   \
        nh, ng, hd, ds, rank, na, tt, st, pt, y_first, z_first, seq_idx);      \
}
DEFINE_M3M_CHUNK_SCAN(m3m_chunk_scan, float)
DEFINE_M3M_CHUNK_SCAN(m3m_chunk_scan_bf16, __nv_bfloat16)
DEFINE_M3M_CHUNK_SCAN(m3m_chunk_scan_f16, __half)

// The same outputs as three register-tile products over the block's query
// tile of `tq` tokens and its group of 32 JP channels (the whole head up to
// 128 channels, groups of 128 past that). Rows are (t, r), t of the tile and
// r ascending; a warp takes 8 rows, a lane the group's channels
// p = lane + 32 j (JP of them):
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
// Gw [64][QR + 4], w [tq][kt], the chunk's Acs, scale and gamma
// [3][M3M_CHUNK]; QR = tq R rounded up to 8. A warp holds 4 query rows, so a block of 64
// rows runs 16 warps: one block fills the shared memory of an SM and the
// extra warps hide the staging loads. The heaviest query tile (the last)
// runs first. Grid (B * nc, H, ceil(M3M_CHUNK / tq) ceil(P / 32 JP)), the
// query tiles innermost, block 8 QR threads.
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
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tq, int kt, int nk,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float4 m3m_sm4[];
    constexpr int PP = 32 * JP;
    constexpr int KC = 64;
    constexpr int KCS = KC + 4;
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int nz = (M3M_CHUNK + tq - 1) / tq;
    const int grp = (int)blockIdx.z / nz;
    const int pb = grp * PP;
    const int tile0 = (nz - 1 - ((int)blockIdx.z - grp * nz)) * tq;
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
    float* sc = wts + tq * kt;                       // Acs, scale, gamma [3][chunk]
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
    // The queries, rotated.
    m3m_stage_rot(warp, nw, rows, qr, 0, ds, half, na,
        [&](int rw) {
            const int r = rw % rank;
            const long long row = b * seq + t0 + tile0 + rw / rank;
            return M3mRotRow{cn + row * rgn + ((long long)r * ng + g) * ds,
                             c_bias + ((long long)h * rank + r) * ds,
                             sn + (row * nh + h) * na, cs + (row * nh + h) * na};
        },
        [&](int rw, int n, float v) { qt[n * qrs + rw] = v; });
    int trow[4];
    int tmax = -1;
    // Per row, the first token of its sequence in the chunk (it reads no
    // earlier one) and whether it continues the sequence the entering
    // state holds; `mixed` when the warp's rows start at different tokens.
    int tfirst[4];
    bool tcont[4];
    int kfirst = M3M_CHUNK;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int rw = rw0 + i;
        trow[i] = rw < rows ? tile0 + rw / rank : -1;
        tmax = max(tmax, trow[i]);
        tfirst[i] = 0;
        tcont[i] = true;
        if (trow[i] >= 0) {
            const VlSpan span = vl_span(seq_idx, b, seq, t0, lc, trow[i]);
            tfirst[i] = span.first;
            tcont[i] = span.from_before;
            kfirst = min(kfirst, span.first);
        }
    }
    bool mixed = false;
#pragma unroll
    for (int i = 0; i < 4; i++) mixed = mixed || (trow[i] >= 0 && tfirst[i] != kfirst);
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
                    const float* row = sp + (long long)(pb + p0 + u * nw) * ds + m0;
#pragma unroll
                    for (int v = 0; v < 4; v++) {
                        const int n = n0 + lane + 32 * v;
                        sv[u][v] = pb + p0 + u * nw < hd && n < mm ? row[n] : 0.0f;
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
            const float4 q0 = lds_f32x4(qt + (m0 + n) * qrs + rw0);
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
        for (int j = 0; j < JP; j++) y[i][j] = tcont[i] ? __fmul_rn(y[i][j], ef) : 0.0f;
    }
    const int s_end = tile0 + tn;
    for (int s0 = 0; s0 < s_end; s0 += kt) {
        const int cols = min(kt, s_end - s0) * rank;
        __syncthreads();
        // The rank inputs.
        m3m_stage_u<TI>(warp, nw, cols, KC, hd - pb, PP,
            [&](int col) {
                return proj + (b * seq + t0 + s0 + col / rank) * ld_proj + di + (long long)h * hd
                       + pb;
            },
            [&](int col) { return mimo_x + ((long long)h * rank + col % rank) * hd + pb; },
            [&](int col, int p, float v) { ut[col * PP + p] = v; });
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
            // The keys, rotated.
            m3m_stage_rot(warp, nw, cols, KC, m0, mm, half, na,
                [&](int col) {
                    const int r = col % rank;
                    const long long row = b * seq + t0 + s0 + col / rank;
                    return M3mRotRow{bn + row * rgn + ((long long)r * ng + g) * ds,
                                     b_bias + ((long long)h * rank + r) * ds,
                                     sn + (row * nh + h) * na, cs + (row * nh + h) * na};
                },
                [&](int col, int n, float v) { kts[n * KCS + col] = v; });
            __syncthreads();
            for (int n = 0; n < mm; n++) {
                const float4 q0 = lds_f32x4(qt + (m0 + n) * qrs + rw0);
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
        // The keys before the warp's rows' sequences are skipped; a warp
        // whose rows start at different tokens skips per row.
        const int k_lo = max(0, kfirst - s0) * rank;
        if (!mixed) {
            for (int k = k_lo; k < cols; k++) {
                const int s = s0 + k / rank;
                if (s > tmax) break;
                const float4 g0 = lds_f32x4(gw + k * qrs + rw0);
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
        } else {
            for (int k = k_lo; k < cols; k++) {
                const int s = s0 + k / rank;
                if (s > tmax) break;
                const float4 g0 = lds_f32x4(gw + k * qrs + rw0);
                const float gv[4] = {g0.x, g0.y, g0.z, g0.w};
                float uv[JP];
#pragma unroll
                for (int j = 0; j < JP; j++) uv[j] = ut[k * PP + lane + 32 * j];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (s <= trow[i] && s >= tfirst[i]) {
#pragma unroll
                        for (int j = 0; j < JP; j++) y[i][j] = __fmaf_rn(gv[i], uv[j], y[i][j]);
                    }
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
            const int pl = lane + 32 * j;
            const int p = pb + pl;
            if (p >= hd) continue;
            const float xv = to_f(proj[row_t * ld_proj + di + (long long)h * hd + p]);
            const float u_t = __fmul_rn(mimo_x[((long long)h * rank + r) * hd + p], xv);
            const float v = __fmaf_rn(dval, u_t, y[i][j]);
            if (y_out != nullptr) y_out[((row_t * nh + h) * rank + r) * hd + p] = v;
            yg[rw * PP + pl] = v;
        }
    }
    __syncthreads();
    for (int e = warp * 32 + lane; e < tn * PP; e += blockDim.x) {
        const int tl = e / PP, pl = e % PP, p = pb + pl;
        if (p >= hd) continue;
        const long long row_t = b * seq + t0 + tile0 + tl;
        const float z = to_f(proj[row_t * ld_proj + (long long)h * hd + p]);
        float o = 0.0f;
        for (int r = 0; r < rank; r++) {
            const float v = __fmul_rn(mimo_z[((long long)h * rank + r) * hd + p], z);
            const float sg = __fdiv_rn(v, __fadd_rn(1.0f, mimo_exp(-v)));
            const float term = __fmul_rn(__fmul_rn(yg[(tl * rank + r) * PP + pl], sg),
                                         mimo_o[((long long)h * rank + r) * hd + p]);
            o = r == 0 ? term : __fadd_rn(o, term);
        }
        mimo_put(&out[row_t * di + (long long)h * hd + p], o);
    }
}

// One entry for each channel count a lane takes: 1 for a head of up to
// 32 channels, 2 up to 64, 4 past that (`chunk_scan_tiles_for` on the
// host). Each entry takes the registers its own lanes need, where one
// entry that switched among them at run time took the widest's for every
// head.
#define DEFINE_M3M_CHUNK_SCAN_TILES(NAME, TI, JP)                              \
extern "C" __global__ void __launch_bounds__(512) NAME(                        \
    TI* __restrict__ out, float* __restrict__ y_out,                           \
    const TI* __restrict__ proj, long long ld_proj,                            \
    const float* __restrict__ bn, const float* __restrict__ cn,                \
    const float* __restrict__ coef,                                            \
    const float* __restrict__ sn, const float* __restrict__ cs,                \
    const float* __restrict__ enter,                                           \
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,        \
    const float* __restrict__ D,                                               \
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,        \
    const float* __restrict__ mimo_o,                                          \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tq, int kt, \
    int nk,                                                                    \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3m_chunk_scan_tiles_body<TI, JP>(out, y_out, proj, ld_proj, bn, cn,       \
        coef, sn, cs, enter, b_bias, c_bias, D, mimo_x, mimo_z, mimo_o,        \
        seq, nh, ng, hd, ds, rank, na, tq, kt, nk, y_first, seq_idx);          \
}
DEFINE_M3M_CHUNK_SCAN_TILES(m3m_chunk_scan_tiles_jp1, float, 1)
DEFINE_M3M_CHUNK_SCAN_TILES(m3m_chunk_scan_tiles_jp1_bf16, __nv_bfloat16, 1)
DEFINE_M3M_CHUNK_SCAN_TILES(m3m_chunk_scan_tiles_jp1_f16, __half, 1)
DEFINE_M3M_CHUNK_SCAN_TILES(m3m_chunk_scan_tiles_jp2, float, 2)
DEFINE_M3M_CHUNK_SCAN_TILES(m3m_chunk_scan_tiles_jp2_bf16, __nv_bfloat16, 2)
DEFINE_M3M_CHUNK_SCAN_TILES(m3m_chunk_scan_tiles_jp2_f16, __half, 2)
DEFINE_M3M_CHUNK_SCAN_TILES(m3m_chunk_scan_tiles_jp4, float, 4)
DEFINE_M3M_CHUNK_SCAN_TILES(m3m_chunk_scan_tiles_jp4_bf16, __nv_bfloat16, 4)
DEFINE_M3M_CHUNK_SCAN_TILES(m3m_chunk_scan_tiles_jp4_f16, __half, 4)

// The carried keys and input after the window: the last token's rotated
// biased keys and its raw input. Grid (B, H), threads striding R * N + P.
template <typename TI>
__device__ __forceinline__ void m3m_writeback_body(
    float* __restrict__ kst, float* __restrict__ vst,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ b_bias,
    int seq, int nh, int ng, int hd, int ds, int rank, int na,
    int y_first /* this run's first y block */
) {
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
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

#define DEFINE_M3M_WRITEBACK(NAME, TI)                                         \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ kst, float* __restrict__ vst,                          \
    const TI* __restrict__ proj, long long ld_proj,                            \
    const float* __restrict__ bn,                                              \
    const float* __restrict__ sn, const float* __restrict__ cs,                \
    const float* __restrict__ b_bias,                                          \
    int seq, int nh, int ng, int hd, int ds, int rank, int na,                 \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    m3m_writeback_body<TI>(kst, vst, proj, ld_proj, bn, sn, cs, b_bias, seq,   \
        nh, ng, hd, ds, rank, na, y_first);                                    \
}
DEFINE_M3M_WRITEBACK(m3m_writeback, float)
DEFINE_M3M_WRITEBACK(m3m_writeback_bf16, __nv_bfloat16)
DEFINE_M3M_WRITEBACK(m3m_writeback_f16, __half)

// ============================================================================
// The chunk scan on the tensor cores, for the TF32, bf16 and f16 forwards.
// ============================================================================
// m3m_chunk_scan_tiles' outputs from TF32 products with f32 accumulation
// (m16n8k8), each operand rounded to TF32 to nearest, ties away, as it
// enters, as the backward's pair terms take theirs; the f32 forward keeps
// the CUDA-core kernels and their f32 sums. The f16 forward's storage keeps
// the 11 bits TF32 keeps, so one TF32 rounding would cost it as much again
// as its storage does: it takes each operand as two TF32 values, its
// rounding and the rounding of the rest, and each product in three passes,
// the two small terms first, which keeps 22 of the operands' 24 bits. bf16
// rounds eight times coarser than TF32 and the TF32 forward asks for TF32,
// so they take one pass. A block takes all of a chunk's
// rows, at most M3MS_ROWS (the rank's own chunk of 64 / R tokens), and a
// group of M3MS_P channels of the head:
//   inter: Y = Q S_enter^T over the state indices, M3MS_SL at a time, then
//     each row times e^Acs_t where it continues the sequence the entering
//     state holds, zero elsewhere;
//   G = Q K^T over the same slices, then Gw = G w(t, s) for s <= t within
//     the row's own sequence, zero elsewhere;
//   Y += Gw U, U = Psi_r' x_s, over one packed sequence's columns at a time,
//     their groups of eight counted from the sequence's first column, each
//     row keeping only its own sequence's pass (the zero weights of another
//     sequence would carry its infinities and NaNs in);
//   then y = fma(D, u_t, Y), the readouts and the gate over the ranks per
//   (token, channel), as the CUDA-core kernel forms them.
// Every sum keeps one order, so the bits repeat run to run, and an output
// depends only on its own row's and column's operands: a sequence gives the
// same bits wherever a packing puts it in the chunk. A warp holds 16 rows
// by 32 columns of G and of Y. Shared memory (floats): the slices q and k
// [ROWS][SL + 4] and two of s [P][SL + 4] (the next one's copies in flight
// while the current one is taken), then Gw [ROWS][ROWS + 8] and U
// [ROWS][P + 8] in their place, the gate's rows [ROWS][P + 4] in Gw's; the
// chunk's Acs, scale, gamma, sequence ids and continuations [ROWS] each and
// its sequences' first tokens (scan_tc_bytes on the host). Grid (B * nc,
// H, ceil(P / M3MS_P)), block M3MS_THREADS.
#define M3MS_THREADS 256
#define M3MS_ROWS 64
#define M3MS_P 64
#define M3MS_SL 32
#define M3MS_SLD (M3MS_SL + 4)
#define M3MS_GLD (M3MS_ROWS + 8)
#define M3MS_ULD (M3MS_P + 8)
#define M3MS_YLD (M3MS_P + 4)

#if M3M_CHUNK * MIMO_RANK <= M3MS_ROWS

__device__ __forceinline__ unsigned m3ms_tf32(float v) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
    return r;
}

__device__ __forceinline__ void m3ms_mma(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Whether a lane's scan takes its products in three TF32 passes: the f16
// lane's (see the comment above).
template <typename TI> struct m3ms_three { static constexpr bool value = false; };
template <> struct m3ms_three<__half> { static constexpr bool value = true; };

// A fragment's values as TF32: their roundings, and with THREE the
// roundings of what each rounding left.
template <bool THREE, int N>
__device__ __forceinline__ void m3ms_frag(
    const float (&v)[N], unsigned (&hi)[N], unsigned (&lo)[N]) {
#pragma unroll
    for (int i = 0; i < N; i++) {
        hi[i] = m3ms_tf32(v[i]);
        lo[i] = THREE ? m3ms_tf32(__fsub_rn(v[i], __uint_as_float(hi[i]))) : 0u;
    }
}

// d += a b in one TF32 pass, or with THREE in three: a's rest times b's
// rounding, a's rounding times b's rest, then the roundings, in that order.
template <bool THREE>
__device__ __forceinline__ void m3ms_mma3(
    float (&d)[4], const unsigned (&ah)[4], const unsigned (&al)[4],
    const unsigned (&bh)[2], const unsigned (&bl)[2]) {
    if constexpr (THREE) {
        m3ms_mma(d, al, bh);
        m3ms_mma(d, ah, bl);
    }
    m3ms_mma(d, ah, bh);
}

// A float copied into shared memory asynchronously, or a zero where `on`
// is false (no byte of `src` is read then), and the groups' fences.
__device__ __forceinline__ void m3ms_cp4(float* dst, const float* src, bool on) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(d), "l"(src), "r"(on ? 4 : 0));
}
__device__ __forceinline__ void m3ms_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void m3ms_cp_wait_newest() {
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}

// The scan of one chunk, head and group of channels from pb, the block's
// warps 16 rows by CW columns of G and of Y (CW 32 for 8 warps, 16 for 16;
// a warp's tiles move no output's sum). The entering state's rows of the
// head come from `s_src`, `s_ld` floats apart: global memory, copied a
// slice at a time into the arena (SMEM_S false), or a whole copy in shared
// memory (SMEM_S true, its rows zero past the head and the state width).
template <typename TI, int CW, bool SMEM_S>
__device__ __forceinline__ void m3m_chunk_scan_tc_core(
    float* __restrict__ arena,
    TI* __restrict__ out, float* __restrict__ y_out,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ cn,
    const float* __restrict__ coef,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ s_src, long long s_ld,
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,
    const float* __restrict__ D,
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,
    const float* __restrict__ mimo_o,
    int seq, int nh, int ng, int hd, int ds, int na,
    long long b, int c, int h, int pb,
    const int* __restrict__ seq_idx
) {
    constexpr int RK = MIMO_RANK;
    constexpr int J = CW / 8;
    constexpr bool THREE = m3ms_three<TI>::value;
    float* qs = arena;                                  // [ROWS][SLD]
    float* ks = qs + M3MS_ROWS * M3MS_SLD;              // [ROWS][SLD]
    float* ssb = ks + M3MS_ROWS * M3MS_SLD;             // 2 x [P][SLD]
    float* gw = arena;                                  // [ROWS][GLD]
    float* us = gw + M3MS_ROWS * M3MS_GLD;              // [ROWS][ULD]
    float* yg = gw;                                     // [ROWS][YLD]
    float* acs = us + M3MS_ROWS * M3MS_ULD;             // [ROWS]
    float* scl = acs + M3MS_ROWS;
    float* gam = scl + M3MS_ROWS;
    int* sid = reinterpret_cast<int*>(gam + M3MS_ROWS);
    int* cont = sid + M3MS_ROWS;
    int* segs = cont + M3MS_ROWS;                       // [ROWS + 1]
    int* nseg_s = segs + M3MS_ROWS + 1;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int rows = lc * RK;
    const int pw = min(M3MS_P, hd - pb);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = RK * ng * ds;
    const int half = ds / 2;
    const long long row0 = b * seq + t0;
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    for (int t = tid; t < M3MS_ROWS; t += blockDim.x) {
        if (t < lc) {
            const float* cf = coef + ((row0 + t) * nh + h) * 4;
            acs[t] = cf[3];
            scl[t] = cf[2];
            gam[t] = cf[1];
            sid[t] = vl_id(seq_idx, b, seq, t0 + t);
            cont[t] = vl_span(seq_idx, b, seq, t0, lc, t).from_before ? 1 : 0;
        } else {
            acs[t] = scl[t] = gam[t] = 0.0f;
            sid[t] = -1;
            cont[t] = 0;
        }
    }
    __syncthreads();
    // The first token of each sequence in the chunk, then the chunk's end.
    if (tid == 0) {
        int n = 0;
        for (int t = 0; t < lc; t++)
            if (t == 0 || sid[t] != sid[t - 1]) segs[n++] = t;
        segs[n] = lc;
        *nseg_s = n;
    }
    // The query and key slices of rotated rank rows into dst[a * M3MS_SLD +
    // n]: rows a below M3MS_ROWS at the state indices n0 + n for n below
    // M3MS_SL, m3m_rot's value and zero past the chunk or the state width, a
    // thread's elements four at a time, every load of the four queries and
    // four keys in flight before any is turned.
    auto rot_slices = [&](int n0) {
        for (int e0 = tid; e0 < M3MS_ROWS * M3MS_SL; e0 += 4 * blockDim.x) {
            M3mRotIn iq[4], ik[4];
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int e = e0 + u * (int)blockDim.x;
                const int a = e / M3MS_SL, n = n0 + e % M3MS_SL;
                iq[u] = ik[u] = M3mRotIn{0.0f, 0.0f, 0.0f, 0.0f};
                if (e < M3MS_ROWS * M3MS_SL && a < rows && n < ds) {
                    const int r = a % RK;
                    const long long row = row0 + a / RK;
                    const long long rg = row * rgn + ((long long)r * ng + g) * ds;
                    const long long hr = ((long long)h * RK + r) * ds;
                    const float* sr_ = sn + (row * nh + h) * na;
                    const float* cr_ = cs + (row * nh + h) * na;
                    iq[u] = m3m_rot_load(cn + rg, c_bias + hr, n, half, na, sr_, cr_);
                    ik[u] = m3m_rot_load(bn + rg, b_bias + hr, n, half, na, sr_, cr_);
                }
            }
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int e = e0 + u * (int)blockDim.x;
                if (e >= M3MS_ROWS * M3MS_SL) continue;
                const int a = e / M3MS_SL, n = n0 + e % M3MS_SL;
                const bool on = a < rows && n < ds;
                qs[a * M3MS_SLD + e % M3MS_SL] = on ? m3m_rot_turn(iq[u], n, half, na) : 0.0f;
                ks[a * M3MS_SLD + e % M3MS_SL] = on ? m3m_rot_turn(ik[u], n, half, na) : 0.0f;
            }
        }
    };
    // A slice of the entering state [P][SLD] by asynchronous copies, zero
    // past the head's group or the state width, committed as one group.
    auto fetch_s = [&](int n0, float* dst) {
#pragma unroll 4
        for (int e = tid; e < M3MS_P * M3MS_SL; e += blockDim.x) {
            const int p = e / M3MS_SL, n = e % M3MS_SL;
            const bool on = p < pw && n0 + n < ds;
            m3ms_cp4(dst + p * M3MS_SLD + n, on ? s_src + (pb + p) * s_ld + n0 + n : s_src, on);
        }
        m3ms_cp_commit();
    };
    // G and Y: a warp's 16 rows from rb and CW columns from cb.
    const int rb = (warp & 3) * 16, cb = (warp >> 2) * CW;
    float gacc[J][4], yacc[J][4];
#pragma unroll
    for (int j = 0; j < J; j++)
#pragma unroll
        for (int i = 0; i < 4; i++) gacc[j][i] = yacc[j][i] = 0.0f;
    // The entering state's slices land in two buffers in turn, the next one's
    // copies in flight while the current slice's rows are turned and taken;
    // a copy in shared memory is read in place.
    if (!SMEM_S) fetch_s(0, ssb);
    for (int n0 = 0, i = 0; n0 < ds; n0 += M3MS_SL, i++) {
        __syncthreads();
        if (!SMEM_S) {
            if (n0 + M3MS_SL < ds)
                fetch_s(n0 + M3MS_SL, ssb + ((i + 1) & 1) * M3MS_P * M3MS_SLD);
            else
                m3ms_cp_commit();
        }
        rot_slices(n0);
        if (!SMEM_S) m3ms_cp_wait_newest();
        __syncthreads();
        const float* ss = SMEM_S ? s_src + pb * s_ld + n0 : ssb + (i & 1) * M3MS_P * M3MS_SLD;
        const long long sld = SMEM_S ? s_ld : M3MS_SLD;
#pragma unroll
        for (int k8 = 0; k8 < M3MS_SL; k8 += 8) {
            const float qv[4] = {
                qs[(rb + gid) * M3MS_SLD + k8 + tig],
                qs[(rb + gid + 8) * M3MS_SLD + k8 + tig],
                qs[(rb + gid) * M3MS_SLD + k8 + tig + 4],
                qs[(rb + gid + 8) * M3MS_SLD + k8 + tig + 4]};
            unsigned a[4], al[4];
            m3ms_frag<THREE>(qv, a, al);
#pragma unroll
            for (int j = 0; j < J; j++) {
                const int col = cb + 8 * j + gid;
                const float kv[2] = {ks[col * M3MS_SLD + k8 + tig],
                                     ks[col * M3MS_SLD + k8 + tig + 4]};
                unsigned bk[2], bkl[2];
                m3ms_frag<THREE>(kv, bk, bkl);
                m3ms_mma3<THREE>(gacc[j], a, al, bk, bkl);
                const float sv[2] = {ss[col * sld + k8 + tig], ss[col * sld + k8 + tig + 4]};
                unsigned bs[2], bsl[2];
                m3ms_frag<THREE>(sv, bs, bsl);
                m3ms_mma3<THREE>(yacc[j], a, al, bs, bsl);
            }
        }
    }
    // The inter term carries e^Acs_t to the rows that continue the sequence
    // the entering state holds.
#pragma unroll
    for (int j = 0; j < J; j++) {
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int i = rb + gid + (q >> 1) * 8;
            const int t = i / RK;
            yacc[j][q] = i < rows && cont[t] ? __fmul_rn(yacc[j][q], mimo_exp(acs[t])) : 0.0f;
        }
    }
    __syncthreads();
    // Gw: G times w(t, s) for s <= t in the row's own sequence, zero
    // elsewhere; and U = Psi_r' x_s.
#pragma unroll
    for (int j = 0; j < J; j++) {
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int i = rb + gid + (q >> 1) * 8;
            const int col = cb + 8 * j + 2 * tig + (q & 1);
            const int t = i / RK, s = col / RK;
            float v = 0.0f;
            if (i < rows && col < rows && s <= t && sid[s] == sid[t]) {
                const float w = s < t
                    ? __fmul_rn(mimo_exp(fminf(__fsub_rn(acs[t], acs[s]), 0.0f)), scl[s])
                    : gam[t];
                v = __fmul_rn(gacc[j][q], w);
            }
            gw[i * M3MS_GLD + col] = v;
        }
    }
    const TI* xr = proj + row0 * ld_proj + di + (long long)h * hd + pb;
    const float* psi = mimo_x + (long long)h * RK * hd + pb;
#pragma unroll 4
    for (int e = tid; e < M3MS_ROWS * M3MS_P; e += blockDim.x) {
        const int col = e / M3MS_P, p = e % M3MS_P;
        float u = 0.0f;
        if (col < rows && p < pw) {
            const int s = col / RK, r = col - s * RK;
            u = __fmul_rn(psi[r * hd + p], to_f(xr[s * ld_proj + p]));
        }
        us[col * M3MS_ULD + p] = u;
    }
    __syncthreads();
    // Y += Gw U a packed sequence at a time.
    const int nseg = *nseg_s;
    for (int sg = 0; sg < nseg; sg++) {
        const int k0 = segs[sg] * RK, k1 = segs[sg + 1] * RK;
        float held[J][4];
#pragma unroll
        for (int j = 0; j < J; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) held[j][q] = yacc[j][q];
        for (int k8 = k0; k8 < k1; k8 += 8) {
            const int ka = k8 + tig, kb = ka + 4;
            const bool va = ka < k1, vb = kb < k1;
            const float gv[4] = {
                va ? gw[(rb + gid) * M3MS_GLD + ka] : 0.0f,
                va ? gw[(rb + gid + 8) * M3MS_GLD + ka] : 0.0f,
                vb ? gw[(rb + gid) * M3MS_GLD + kb] : 0.0f,
                vb ? gw[(rb + gid + 8) * M3MS_GLD + kb] : 0.0f};
            unsigned a[4], al[4];
            m3ms_frag<THREE>(gv, a, al);
#pragma unroll
            for (int j = 0; j < J; j++) {
                const int p = cb + 8 * j + gid;
                const float uv[2] = {va ? us[ka * M3MS_ULD + p] : 0.0f,
                                     vb ? us[kb * M3MS_ULD + p] : 0.0f};
                unsigned bu[2], bul[2];
                m3ms_frag<THREE>(uv, bu, bul);
                m3ms_mma3<THREE>(yacc[j], a, al, bu, bul);
            }
        }
#pragma unroll
        for (int j = 0; j < J; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int i = rb + gid + (q >> 1) * 8;
                if (i < k0 || i >= k1) yacc[j][q] = held[j][q];
            }
    }
    __syncthreads();
    // D, the readouts and the gate's rows.
    const float dval = D[h];
#pragma unroll
    for (int j = 0; j < J; j++) {
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int i = rb + gid + (q >> 1) * 8;
            const int pl = cb + 8 * j + 2 * tig + (q & 1);
            if (i >= rows || pl >= pw) continue;
            const int t = i / RK, r = i - t * RK;
            const float v = __fmaf_rn(dval, us[i * M3MS_ULD + pl], yacc[j][q]);
            if (y_out != nullptr)
                y_out[(((row0 + t) * nh + h) * RK + r) * hd + pb + pl] = v;
            yg[i * M3MS_YLD + pl] = v;
        }
    }
    __syncthreads();
#pragma unroll 4
    for (int e = tid; e < lc * pw; e += blockDim.x) {
        const int tl = e / pw, pl = e % pw, p = pb + pl;
        const long long row_t = row0 + tl;
        const float z = to_f(proj[row_t * ld_proj + (long long)h * hd + p]);
        float o = 0.0f;
        for (int r = 0; r < RK; r++) {
            const float v = __fmul_rn(mimo_z[((long long)h * RK + r) * hd + p], z);
            const float sg = __fdiv_rn(v, __fadd_rn(1.0f, mimo_exp(-v)));
            const float term = __fmul_rn(__fmul_rn(yg[(tl * RK + r) * M3MS_YLD + pl], sg),
                                         mimo_o[((long long)h * RK + r) * hd + p]);
            o = r == 0 ? term : __fadd_rn(o, term);
        }
        mimo_put(&out[row_t * di + (long long)h * hd + p], o);
    }
}

template <typename TI>
__device__ __forceinline__ void m3m_chunk_scan_tc_body(
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
    int seq, int nh, int ng, int hd, int ds, int na,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float4 m3ms_sm4[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    m3m_chunk_scan_tc_core<TI, 32, false>(
        reinterpret_cast<float*>(m3ms_sm4), out, y_out, proj, ld_proj, bn, cn, coef, sn, cs,
        enter + (bc * nh + h) * (long long)hd * ds, ds, b_bias, c_bias, D, mimo_x, mimo_z,
        mimo_o, seq, nh, ng, hd, ds, na, bc / nc, (int)(bc % nc), h, blockIdx.z * M3MS_P,
        seq_idx);
}

#define DEFINE_M3M_CHUNK_SCAN_TC(NAME, TI)                                     \
extern "C" __global__ void __launch_bounds__(M3MS_THREADS) NAME(               \
    TI* __restrict__ out, float* __restrict__ y_out,                           \
    const TI* __restrict__ proj, long long ld_proj,                            \
    const float* __restrict__ bn, const float* __restrict__ cn,                \
    const float* __restrict__ coef,                                            \
    const float* __restrict__ sn, const float* __restrict__ cs,                \
    const float* __restrict__ enter,                                           \
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,        \
    const float* __restrict__ D,                                               \
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,        \
    const float* __restrict__ mimo_o,                                          \
    int seq, int nh, int ng, int hd, int ds, int na,                           \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3m_chunk_scan_tc_body<TI>(out, y_out, proj, ld_proj, bn, cn, coef, sn,    \
        cs, enter, b_bias, c_bias, D, mimo_x, mimo_z, mimo_o, seq, nh, ng, hd, \
        ds, na, y_first, seq_idx);                                             \
}
DEFINE_M3M_CHUNK_SCAN_TC(m3m_chunk_scan_tc, float)
DEFINE_M3M_CHUNK_SCAN_TC(m3m_chunk_scan_tc_bf16, __nv_bfloat16)
DEFINE_M3M_CHUNK_SCAN_TC(m3m_chunk_scan_tc_f16, __half)

// The state chain and the tensor-core scan in one block per chunk and
// head, for the forwards whose scan runs on the tensor cores and whose head
// one chain tile holds (at most M3MS_P channels and 128 state indices): the
// block forms its chunk's contribution as m3m_chunk_state_chain does, takes
// the entering state from the block of the chunk before and hands the next
// one on, then keeps the entering state in shared memory for its own scan,
// m3m_chunk_scan_tc's arithmetic over 16 warps. Every value is the one the
// two kernels give, but the entering states never round-trip through
// global memory: the hand-off between neighbouring chunks goes through two
// slots per head in `ring` (B H P N floats each), a block reading its slot
// before it raises the flag that lets the next block overwrite the other.
// Shared memory: the entering state [P][M3MCS_NLD], then the scan's arena
// (the contribution's staging in the same room first). Grid B * nc * H,
// one dimension, block 512.
#define M3MCS_NLD (128 + 4)

// The scan part of m3m_chain_scan_tc out of line: its registers are then
// allocated apart from the chain's, which are dead by the call.
template <typename TI>
__device__ __noinline__ void m3m_chain_scan_tc_scan(
    float* __restrict__ arena, TI* __restrict__ out, float* __restrict__ y_out,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ cn,
    const float* __restrict__ coef,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ S,
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,
    const float* __restrict__ D,
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,
    const float* __restrict__ mimo_o,
    int seq, int nh, int ng, int hd, int ds, int na, long long b, int c, int h,
    const int* __restrict__ seq_idx
) {
    m3m_chunk_scan_tc_core<TI, 16, true>(arena, out, y_out, proj, ld_proj, bn, cn, coef, sn, cs,
                                         S, M3MCS_NLD, b_bias, c_bias, D, mimo_x, mimo_z,
                                         mimo_o, seq, nh, ng, hd, ds, na, b, c, h, 0, seq_idx);
}

template <int JC, typename TI>
__device__ __forceinline__ void m3m_chain_scan_tc_body(
    TI* __restrict__ out, float* __restrict__ y_out,
    float* __restrict__ ring, float* __restrict__ ssm, unsigned* __restrict__ chain,
    const TI* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ cn,
    const float* __restrict__ coef,
    const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,
    const float* __restrict__ D,
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,
    const float* __restrict__ mimo_o,
    const float* __restrict__ e0, const float* __restrict__ kst,
    const float* __restrict__ vst,
    int batch, int seq, int nh, int ng, int hd, int ds, int na, int sl,
    const int* __restrict__ seq_idx
) {
    extern __shared__ float4 m3mcs_sm4[];
    __shared__ unsigned place;
    float* S = reinterpret_cast<float*>(m3mcs_sm4);  // [P][NLD]
    float* arena = S + M3MS_P * M3MCS_NLD;
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long per = (long long)batch * nh;
    const int tid = threadIdx.x;
    if (tid == 0) place = atomicAdd(chain, 1U);
    __syncthreads();
    const int c = (int)(place / per);
    const long long bh = place - c * per;
    const int h = (int)(bh % nh);
    const long long b = bh / nh;
    float acc[4][JC];
    m3m_state_tile_chunk<JC, TI>(acc, S, proj, ld_proj, bn, coef, sn, cs, b_bias, mimo_x, b, c,
                                 h, 0, 0, seq, nh, ng, hd, ds, MIMO_RANK, na, sl, M3MS_P,
                                 seq_idx);
    __syncthreads();
    const int lane = tid & 31;
    const int p0 = 4 * (tid >> 5);
    const long long pn = (long long)hd * ds;
    unsigned* flags = chain + 1;
    float st[4][JC];
    if (c == 0) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int p = p0 + i;
            const float v = p < hd ? vst[bh * hd + p] : 0.0f;
#pragma unroll
            for (int j = 0; j < JC; j++) {
                const int n = lane + 32 * j;
                st[i][j] = 0.0f;
                if (p >= hd || n >= ds) continue;
                float adj = 0.0f;
#pragma unroll 1
                for (int r = 0; r < MIMO_RANK; r++) {
                    const float up =
                        __fmul_rn(mimo_x[((long long)h * MIMO_RANK + r) * hd + p], v);
                    adj = __fmaf_rn(up, kst[(bh * MIMO_RANK + r) * ds + n], adj);
                }
                st[i][j] = __fmaf_rn(e0[bh], adj, ssm[bh * pn + (long long)p * ds + n]);
            }
        }
    } else {
        if (tid == 0) m3m_await(flags + (c - 1) * per + bh);
        __syncthreads();
        const float* in = ring + ((long long)(c & 1) * per + bh) * pn;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int p = p0 + i;
#pragma unroll
            for (int j = 0; j < JC; j++) {
                const int n = lane + 32 * j;
                st[i][j] = p < hd && n < ds ? __ldcg(in + (long long)p * ds + n) : 0.0f;
            }
        }
    }
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const float decay = mimo_exp(coef[((b * seq + t0 + lc - 1) * nh + h) * 4 + 3]);
    const bool passes = vl_passes(seq_idx, b, seq, t0, lc);
    float* nxt = c + 1 < nc ? ring + ((long long)((c + 1) & 1) * per + bh) * pn : ssm + bh * pn;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int p = p0 + i;
#pragma unroll
        for (int j = 0; j < JC; j++) {
            const int n = lane + 32 * j;
            if (p < hd && n < ds)
                nxt[(long long)p * ds + n] =
                    passes ? __fmaf_rn(decay, st[i][j], acc[i][j]) : acc[i][j];
        }
    }
    if (c + 1 < nc) {
        __threadfence();
        __syncthreads();
        if (tid == 0) m3m_raise(flags + c * per + bh);
    }
    // The entering state for the scan, zero past the head and the state
    // width.
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < JC; j++) S[(p0 + i) * M3MCS_NLD + lane + 32 * j] = st[i][j];
    }
    m3m_chain_scan_tc_scan<TI>(arena, out, y_out, proj, ld_proj, bn, cn, coef, sn, cs, S, b_bias,
                               c_bias, D, mimo_x, mimo_z, mimo_o, seq, nh, ng, hd, ds, na, b, c,
                               h, seq_idx);
}

#define DEFINE_M3M_CHAIN_SCAN_TC(NAME, TI)                                     \
extern "C" __global__ void __launch_bounds__(512) NAME(                        \
    TI* __restrict__ out, float* __restrict__ y_out,                           \
    float* __restrict__ ring, float* __restrict__ ssm,                         \
    unsigned* __restrict__ chain,                                              \
    const TI* __restrict__ proj, long long ld_proj,                            \
    const float* __restrict__ bn, const float* __restrict__ cn,                \
    const float* __restrict__ coef,                                            \
    const float* __restrict__ sn, const float* __restrict__ cs,                \
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,        \
    const float* __restrict__ D,                                               \
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,        \
    const float* __restrict__ mimo_o,                                          \
    const float* __restrict__ e0, const float* __restrict__ kst,               \
    const float* __restrict__ vst,                                             \
    int batch, int seq, int nh, int ng, int hd, int ds, int na, int jc, int sl, \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    if (jc == 1)                                                               \
        m3m_chain_scan_tc_body<1, TI>(out, y_out, ring, ssm, chain, proj, ld_proj, \
            bn, cn, coef, sn, cs, b_bias, c_bias, D, mimo_x, mimo_z, mimo_o, e0, \
            kst, vst, batch, seq, nh, ng, hd, ds, na, sl, seq_idx);            \
    else if (jc == 2)                                                          \
        m3m_chain_scan_tc_body<2, TI>(out, y_out, ring, ssm, chain, proj, ld_proj, \
            bn, cn, coef, sn, cs, b_bias, c_bias, D, mimo_x, mimo_z, mimo_o, e0, \
            kst, vst, batch, seq, nh, ng, hd, ds, na, sl, seq_idx);            \
    else                                                                       \
        m3m_chain_scan_tc_body<4, TI>(out, y_out, ring, ssm, chain, proj, ld_proj, \
            bn, cn, coef, sn, cs, b_bias, c_bias, D, mimo_x, mimo_z, mimo_o, e0, \
            kst, vst, batch, seq, nh, ng, hd, ds, na, sl, seq_idx);            \
}
DEFINE_M3M_CHAIN_SCAN_TC(m3m_chain_scan_tc, float)
DEFINE_M3M_CHAIN_SCAN_TC(m3m_chain_scan_tc_bf16, __nv_bfloat16)
DEFINE_M3M_CHAIN_SCAN_TC(m3m_chain_scan_tc_f16, __half)

#endif  // M3M_CHUNK * MIMO_RANK <= M3MS_ROWS
