// The MIMO chunked backward's pair terms as they were released before the
// split kernels took every shape: one block per chunk and head walking
// (t-tile, s-tile) pairs over tiles of `tl` tokens, its rows staged in
// shared memory or, with `glob`, in a global scratch. The split kernels,
// at their own tile and with their operands staged a slice at a time, are
// held to it bit for bit at its tile. The typed prelude and the packed-row
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

// m3m_tile_product with each output row's chain over k limited to the
// range [lo, hi) that range(r) (an int2) gives: the terms outside it are
// skipped rather than added as zeros, so whatever the operands hold there
// (another packed sequence's values) never reaches the row. A warp walks
// the union of its four rows' ranges; every output keeps one chain over k
// ascending.
template <int JC, class LF, class RF, class INF, class PUTF, class KRF>
__device__ __forceinline__ void m3m_tile_product_ranged(
    int rows, int cols, LF lf, RF rf, INF init, PUTF put, KRF range
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nw = blockDim.x >> 5;
    if (warp >= nw) return;
    for (int r0 = 4 * warp; r0 < rows; r0 += 4 * nw) {
        int lo[4], hi[4];
        int k0 = 0x7fffffff, k1 = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int2 kr = r0 + i < rows ? range(r0 + i) : make_int2(0, 0);
            lo[i] = kr.x;
            hi[i] = kr.y;
            if (kr.x < kr.y) {
                k0 = min(k0, kr.x);
                k1 = max(k1, kr.y);
            }
        }
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
            for (int k = k0; k < k1; k++) {
                float lv[4], rv[JC];
#pragma unroll
                for (int i = 0; i < 4; i++) lv[i] = k >= lo[i] && k < hi[i] ? lf(r0 + i, k) : 0.0f;
#pragma unroll
                for (int j = 0; j < JC; j++) {
                    const int c = c0 + lane + 32 * j;
                    rv[j] = c < cols ? rf(k, c) : 0.0f;
                }
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (k >= lo[i] && k < hi[i]) {
#pragma unroll
                        for (int j = 0; j < JC; j++) acc[i][j] = __fmaf_rn(lv[i], rv[j], acc[i][j]);
                    }
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

// The pair terms of a chunk and head, one block walking (t-tile, s-tile)
// pairs, t-tiles ascending and for each the s-tiles up to it ascending.
// Per tile pair the rotated queries and keys, dy and u are staged; G and M
// come out of n- and p-dots, w(t, s) masks and weighs them, and
//   dw(t, s) = sum_{r, r'} G M      -> dw [B, nc, H, pairs], pair t(t+1)/2 + s
//   dq_t += sum_{s, r'} (w M) k_s    (the t-tile's dq in shared memory)
//   dk_s += sum_{t, r} (w M) q_t      (global, this block's own rows)
//   du_s += sum_{t, r} (w G) dy_t     (global, this block's own rows)
// Every output has one owner: the block, in this loop order; the products
// run as register tiles (m3m_tile_product). Dynamic shared memory:
// tl R N + tl R (N + 1) + tl R P + tl R (P + 1) + 3 (tl R)^2 + tl R N
// floats (queries, keys, dy, u, G, M, wG, the dq tile; key and u rows
// padded by one). With `glob` the queries, keys, dy, u and the dq tile
// sit in that block's slice of a global scratch [B * nc * H][tl R
// (3 N + 2 P + 2)] instead, and only G, M and wG in shared memory: the
// products read the same values in the same order. With a packing a pair
// of two sequences is no pair (its dw is zero), and the products run over
// each row's own sequence only (m3m_tile_product_ranged), so another
// sequence's values never enter a chain. Grid (B * nc, H), block 256.
template <typename T>
__device__ __forceinline__ void m3mb_pairs_body(
    float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ du,
    float* __restrict__ dwp,
    const float* __restrict__ dy, const T* __restrict__ proj, long long ld_proj,
    const float* __restrict__ bn, const float* __restrict__ cn,
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,
    const float* __restrict__ coef, const float* __restrict__ sn, const float* __restrict__ cs,
    const float* __restrict__ mimo_x, float* __restrict__ glob,
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tl,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m3mb_sm[];
    const int nc = (seq + M3M_CHUNK - 1) / M3M_CHUNK;
    const long long bc = blockIdx.x;
    const int h = y_first + blockIdx.y;
    const int c = (int)(bc % nc);
    const long long b = bc / nc;
    const int t0 = c * M3M_CHUNK;
    const int lc = min(M3M_CHUNK, seq - t0);
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int half = ds / 2;
    const int tr = tl * rank;
    // The key and u rows padded by one: a warp reads one key (u) row per
    // lane at a step of the G (M) product, which at a stride of N or P
    // (multiples of 32) would put the reads in one bank.
    const int kld = ds + 1, uld = hd + 1;
    float* gstage = glob == nullptr
        ? nullptr
        : glob + ((long long)bc * nh + h) * ((long long)tr * (3 * ds + 2 * hd + 2));
    float* qs = gstage == nullptr ? m3mb_sm : gstage;  // [tl][R][N]
    float* ks = qs + tr * ds;       // [tl][R][N + 1]
    float* dys = ks + tr * kld;     // [tl][R][P]
    float* us = dys + tr * hd;      // [tl][R][P + 1]
    float* gm = gstage == nullptr ? us + tr * uld : m3mb_sm;  // [tr][tr] G
    float* mm = gm + tr * tr;       // [tr][tr] M, then w M
    float* wg = mm + tr * tr;       // [tr][tr] w G
    float* dqs = gstage == nullptr ? wg + tr * tr : us + tr * uld;  // [tl][R][N]
    const int tid = threadIdx.x;
    const long long pairs = M3M_CHUNK * (M3M_CHUNK + 1) / 2;
    for (int T0 = 0; T0 < lc; T0 += tl) {
        const int tn = min(tl, lc - T0);
        __syncthreads();
#pragma unroll 4
        for (int i = tid; i < tn * rank * ds; i += blockDim.x) {
            const int tq = i / (rank * ds), r = (i / ds) % rank, n = i % ds;
            const long long row = b * seq + t0 + T0 + tq;
            qs[i] =
                m3m_rot(cn + row * rgn + ((long long)r * ng + g) * ds,
                        c_bias + ((long long)h * rank + r) * ds, n, half, na,
                        sn + (row * nh + h) * na, cs + (row * nh + h) * na);
            dqs[i] = dq[((row * nh + h) * rank + r) * ds + n];
        }
#pragma unroll 4
        for (int i = tid; i < tn * rank * hd; i += blockDim.x) {
            const int tq = i / (rank * hd), rp = i % (rank * hd);
            const long long row = b * seq + t0 + T0 + tq;
            dys[i] = dy[(row * nh + h) * rank * hd + rp];
        }
        for (int S0 = 0; S0 <= T0; S0 += tl) {
            const int sn_ = min(tl, lc - S0);
            __syncthreads();
#pragma unroll 4
            for (int i = tid; i < sn_ * rank * ds; i += blockDim.x) {
                const int sk = i / (rank * ds), r = (i / ds) % rank, n = i % ds;
                const long long row = b * seq + t0 + S0 + sk;
                ks[(sk * rank + r) * kld + n] =
                    m3m_rot(bn + row * rgn + ((long long)r * ng + g) * ds,
                            b_bias + ((long long)h * rank + r) * ds, n, half, na,
                            sn + (row * nh + h) * na, cs + (row * nh + h) * na);
            }
#pragma unroll 4
            for (int i = tid; i < sn_ * rank * hd; i += blockDim.x) {
                const int sk = i / (rank * hd), r = (i / hd) % rank, p = i % hd;
                const long long row = b * seq + t0 + S0 + sk;
                const float x = to_f(proj[row * ld_proj + di + (long long)h * hd + p]);
                us[(sk * rank + r) * uld + p] =
                    __fmul_rn(mimo_x[((long long)h * rank + r) * hd + p], x);
            }
            __syncthreads();
            // G and M per (t r, s r'), zero where s > t or s and t lie in
            // two packed sequences.
            const int ta = tn * rank, sb = sn_ * rank;
            auto live = [&](int a, int bcol) {
                const int t = T0 + a / rank, s = S0 + bcol / rank;
                return s <= t && vl_id(seq_idx, b, seq, t0 + s) == vl_id(seq_idx, b, seq, t0 + t);
            };
            m3m_tile_product<1>(
                ta, sb, ds,
                [&](int a, int n) { return qs[a * ds + n]; },
                [&](int n, int bcol) { return ks[bcol * kld + n]; },
                [&](int, int) { return 0.0f; },
                [&](int a, int bcol, float v) { gm[a * tr + bcol] = live(a, bcol) ? v : 0.0f; });
            m3m_tile_product<1>(
                ta, sb, hd,
                [&](int a, int p) { return dys[a * hd + p]; },
                [&](int p, int bcol) { return us[bcol * uld + p]; },
                [&](int, int) { return 0.0f; },
                [&](int a, int bcol, float v) { mm[a * tr + bcol] = live(a, bcol) ? v : 0.0f; });
            __syncthreads();
            // dw per pair (a fixed loop over (r, r')), then w M and w G.
            for (int i = tid; i < tn * sn_; i += blockDim.x) {
                const int tq = i / sn_, sk = i % sn_;
                const int t = T0 + tq, s = S0 + sk;
                const bool apart = vl_id(seq_idx, b, seq, t0 + s) != vl_id(seq_idx, b, seq, t0 + t);
                if (s > t || apart) {
                    for (int r = 0; r < rank; r++)
                        for (int r2 = 0; r2 < rank; r2++) {
                            const int at = (tq * rank + r) * tr + sk * rank + r2;
                            wg[at] = 0.0f;
                            mm[at] = 0.0f;
                        }
                    if (s <= t) dwp[(bc * nh + h) * pairs + (long long)t * (t + 1) / 2 + s] = 0.0f;
                    continue;
                }
                const long long row_t = b * seq + t0 + t, row_s = b * seq + t0 + s;
                const float* ct = coef + (row_t * nh + h) * 4;
                const float* cs_ = coef + (row_s * nh + h) * 4;
                const float w = s < t
                    ? __fmul_rn(mimo_exp(fminf(__fsub_rn(ct[3], cs_[3]), 0.0f)), cs_[2])
                    : ct[1];
                float dw = 0.0f;
                for (int r = 0; r < rank; r++) {
                    for (int r2 = 0; r2 < rank; r2++) {
                        const int at = (tq * rank + r) * tr + sk * rank + r2;
                        dw = __fmaf_rn(gm[at], mm[at], dw);
                        wg[at] = __fmul_rn(w, gm[at]);
                        mm[at] = __fmul_rn(w, mm[at]);
                    }
                }
                dwp[(bc * nh + h) * pairs + (long long)t * (t + 1) / 2 + s] = dw;
            }
            __syncthreads();
            // dq for the t-tile (shared), dk and du for the s-tile (global).
            auto krow = [&](int bcol) {
                const long long row = b * seq + t0 + S0 + bcol / rank;
                return (row * nh + h) * rank + bcol % rank;
            };
            if (seq_idx == nullptr) {
                m3m_tile_product<4>(
                    ta, ds, sb,
                    [&](int a, int bcol) { return mm[a * tr + bcol]; },
                    [&](int bcol, int n) { return ks[bcol * kld + n]; },
                    [&](int a, int n) { return dqs[a * ds + n]; },
                    [&](int a, int n, float v) { dqs[a * ds + n] = v; });
                m3m_tile_product<4>(
                    sb, ds, ta,
                    [&](int bcol, int a) { return mm[a * tr + bcol]; },
                    [&](int a, int n) { return qs[a * ds + n]; },
                    [&](int bcol, int n) { return dk[krow(bcol) * ds + n]; },
                    [&](int bcol, int n, float v) { dk[krow(bcol) * ds + n] = v; });
                m3m_tile_product<2>(
                    sb, hd, ta,
                    [&](int bcol, int a) { return wg[a * tr + bcol]; },
                    [&](int a, int p) { return dys[a * hd + p]; },
                    [&](int bcol, int p) { return du[krow(bcol) * hd + p]; },
                    [&](int bcol, int p, float v) { du[krow(bcol) * hd + p] = v; });
            } else {
                // A query row takes the keys of its sequence up to itself, a
                // key row the queries of its sequence from itself on.
                auto q_range = [&](int a) {
                    const int t = T0 + a / rank;
                    const int f = vl_seq_first(seq_idx, b, seq, t0, t0 + t) - t0;
                    return make_int2(max(f - S0, 0) * rank, min(t - S0 + 1, sn_) * rank);
                };
                auto k_range = [&](int bcol) {
                    const int s = S0 + bcol / rank;
                    const int e = vl_span(seq_idx, b, seq, t0, lc, s).end;
                    return make_int2(max(s - T0, 0) * rank, min(e - T0, tn) * rank);
                };
                m3m_tile_product_ranged<4>(
                    ta, ds,
                    [&](int a, int bcol) { return mm[a * tr + bcol]; },
                    [&](int bcol, int n) { return ks[bcol * kld + n]; },
                    [&](int a, int n) { return dqs[a * ds + n]; },
                    [&](int a, int n, float v) { dqs[a * ds + n] = v; }, q_range);
                m3m_tile_product_ranged<4>(
                    sb, ds,
                    [&](int bcol, int a) { return mm[a * tr + bcol]; },
                    [&](int a, int n) { return qs[a * ds + n]; },
                    [&](int bcol, int n) { return dk[krow(bcol) * ds + n]; },
                    [&](int bcol, int n, float v) { dk[krow(bcol) * ds + n] = v; }, k_range);
                m3m_tile_product_ranged<2>(
                    sb, hd,
                    [&](int bcol, int a) { return wg[a * tr + bcol]; },
                    [&](int a, int p) { return dys[a * hd + p]; },
                    [&](int bcol, int p) { return du[krow(bcol) * hd + p]; },
                    [&](int bcol, int p, float v) { du[krow(bcol) * hd + p] = v; }, k_range);
            }
        }
        __syncthreads();
        for (int i = tid; i < tn * rank * ds; i += blockDim.x) {
            const int tq = i / (rank * ds), rn = i % (rank * ds);
            const long long row = b * seq + t0 + T0 + tq;
            dq[(row * nh + h) * rank * ds + rn] = dqs[i];
        }
    }
}

#define DEFINE_M3MB_PAIRS(NAME, T)                                             \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ du,    \
    float* __restrict__ dwp,                                                   \
    const float* __restrict__ dy, const T* __restrict__ proj,                  \
    long long ld_proj,                                                         \
    const float* __restrict__ bn, const float* __restrict__ cn,                \
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,        \
    const float* __restrict__ coef, const float* __restrict__ sn,              \
    const float* __restrict__ cs, const float* __restrict__ mimo_x,            \
    float* __restrict__ glob,                                                  \
    int seq, int nh, int ng, int hd, int ds, int rank, int na, int tl,         \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3mb_pairs_body<T>(dq, dk, du, dwp, dy, proj, ld_proj, bn, cn, b_bias,     \
        c_bias, coef, sn, cs, mimo_x, glob, seq, nh, ng, hd, ds, rank, na, tl, y_first, \
        seq_idx);                                                              \
}
DEFINE_M3MB_PAIRS(m3mb_reference_pairs, float)
DEFINE_M3MB_PAIRS(m3mb_reference_pairs_bf16, __nv_bfloat16)
DEFINE_M3MB_PAIRS(m3mb_reference_pairs_f16, __half)
