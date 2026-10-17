// The staged dQKV stage of the SISO chunked backward, as it was released
// before the register-tiled stage replaced it: every operand streamed
// through tiles TW columns wide, an output or a pair slot a thread. The
// register-tiled stage is held to it bit for bit. The typed prelude and the
// packed-row helpers precede this fixture.

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

// Index of the pair (a, b), a < b, in a chunk's strict upper triangle.
__device__ __forceinline__ int m3t_tri(int a, int b, int cs) {
    return a * (2 * cs - a - 1) / 2 + (b - a - 1);
}

// m3_dqkv_tiled: m3_dqkv for any headdim and d_state.
//
// One block of 256 threads per (head, sequence, chunk), tiles TW columns
// wide (up to 32, the caller's choice: the bits do not depend on it).
// Shared memory, in floats: nine per-step lanes of 64, the three pair
// triangles (K.Q, V.dO and the decay) of CS (CS - 1) / 2 each, four
// [64][TW + 1] operand tiles and two [TW][TW + 1] state tiles.
//
// The work, in order, each sum in m3_dqkv's order:
// 1. The pair dots over d_state and over headdim, one slot per thread,
//    their partials held in registers across the tiles.
// 2. dQK_dot from global memory, each 32-channel segment through the warp
//    kernel's shuffle tree, the segments added in order.
// 3. Per head tile: K.dS^T and Q.S^T over d_state, then dV and the two
//    per-step chains of the dADT state terms over the tile's channels.
// 4. Per state tile: the intra-chunk dK and dQ, then their state terms
//    over headdim.
// 5. dADT from the triangles and the per-step lanes, its serial combine,
//    and the chunk's dD partial.
#define M3T_MAX_SLOTS 8
#define M3T_MAX_OUT 8
#define DEFINE_M3_DQKV_TILED(NAME, T_ACT)                                      \
extern "C" __global__ __launch_bounds__(256) void NAME(                        \
    float* __restrict__ dQ_mid, float* __restrict__ dK_mid,                    \
    float* __restrict__ dV, float* __restrict__ dADT,                          \
    float* __restrict__ dQK_dot_out, float* __restrict__ dD_partials,          \
    const T_ACT* __restrict__ Q_rot, const T_ACT* __restrict__ K_scaled,       \
    const T_ACT* __restrict__ V_in,                                            \
    const float* __restrict__ DA_CS, const float* __restrict__ DA_CS_SUM,      \
    const float* __restrict__ QK_dot_in,                                       \
    const float* __restrict__ SSM_States,                                      \
    const T_ACT* __restrict__ dO, const float* __restrict__ D_param,           \
    const float* __restrict__ enter_state,                                     \
    const float* __restrict__ dm_chain,                                        \
    int B, int T, int nh_total, int hd, int ds, int CS, int TW,                \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    const int h = blockIdx.x;                                                  \
    const int b = y_first + blockIdx.y;                                        \
    const int chunk_idx = z_first + blockIdx.z;                                \
    const int tid = threadIdx.x;                                               \
    /* Uniform across the block, so every thread reaches the barriers. */     \
    if (h >= nh_total || b >= B || CS > 64 || TW > 32 || TW < 1) return;       \
    const int n_chunks = (T + CS - 1) / CS;                                    \
    const int chunk_start = chunk_idx * CS;                                    \
    const int len = min(CS, T - chunk_start);                                  \
    const long long d_inner = (long long)nh_total * hd;                        \
    const long long cidx = ((long long)b * n_chunks + chunk_idx) * nh_total + h; \
    const long long row0 = (long long)b * T + chunk_start;                     \
    const float* ssm = SSM_States + cidx * hd * ds;                            \
    const float* dst = enter_state + cidx * hd * ds;                           \
    const float D_val = D_param[h];                                            \
    const float cs_sum = DA_CS_SUM[cidx];                                      \
    const int RS = TW + 1;                                                     \
    const int tri_n = CS * (CS - 1) / 2;                                       \
    extern __shared__ float m3t_sm[];                                          \
    float* da_sm = m3t_sm;                                                     \
    float* qk_sm = da_sm + 64;                                                 \
    float* ef_sm = qk_sm + 64;                                                 \
    float* er_sm = ef_sm + 64;                                                 \
    float* dmrev_sm = er_sm + 64;                                              \
    float* dmvec_sm = dmrev_sm + 64;                                           \
    float* dqk_sm = dmvec_sm + 64;                                             \
    float* qsdo_sm = dqk_sm + 64;                                              \
    float* dskv_sm = qsdo_sm + 64;                                             \
    float* kq_tri = dskv_sm + 64;                                              \
    float* vdo_tri = kq_tri + tri_n;                                           \
    float* dec_tri = vdo_tri + tri_n;                                          \
    float* a0 = dec_tri + tri_n; /* [64][RS] K, then K.dS^T */                 \
    float* a1 = a0 + 64 * RS;    /* [64][RS] Q, then Q.S^T */                  \
    float* a2 = a1 + 64 * RS;    /* [64][RS] dO */                             \
    float* a3 = a2 + 64 * RS;    /* [64][RS] V */                              \
    float* s0 = a3 + 64 * RS;    /* [TW][RS] saved state S */                  \
    float* s1 = s0 + TW * RS;    /* [TW][RS] state gradient dS */              \
    for (int t = tid; t < 64; t += 256) {                                      \
        float da = 0.0f, qk = 0.0f;                                            \
        if (t < len) {                                                         \
            da = DA_CS[cidx * CS + t];                                         \
            qk = QK_dot_in[(row0 + t) * nh_total + h];                         \
        }                                                                      \
        da_sm[t] = da;                                                         \
        qk_sm[t] = qk;                                                         \
        ef_sm[t] = exp2f(da * LOG2E);                                          \
        er_sm[t] = exp2f((cs_sum - da) * LOG2E);                               \
    }                                                                          \
    /* The pair dots. Slot idx is pair (aa, bb) of the triangle. */   \
    int pa[M3T_MAX_SLOTS], pb[M3T_MAX_SLOTS];                                  \
    float kq_acc[M3T_MAX_SLOTS], vdo_acc[M3T_MAX_SLOTS];                       \
    _Pragma("unroll")                                                          \
    for (int k = 0; k < M3T_MAX_SLOTS; k++) {                                  \
        const int idx = tid + 256 * k;                                         \
        pa[k] = -1;                                                            \
        pb[k] = -1;                                                            \
        kq_acc[k] = 0.0f;                                                      \
        vdo_acc[k] = 0.0f;                                                     \
        if (idx < tri_n) {                                                     \
            int aa = 0, start = 0;                                             \
            while (start + (CS - 1 - aa) <= idx) {                             \
                start += CS - 1 - aa;                                          \
                aa++;                                                          \
            }                                                                  \
            const int bb = idx - start + aa + 1;                               \
            if (bb < len) {                                                    \
                pa[k] = aa;                                                    \
                pb[k] = bb;                                                    \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    for (int n0 = 0; n0 < ds; n0 += TW) {                                      \
        const int nw = min(TW, ds - n0);                                       \
        __syncthreads();                                                       \
        for (int o = tid; o < len * TW; o += 256) {                            \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            if (j < nw) {                                                      \
                const long long g = ((row0 + t) * nh_total + h) * ds + n0 + j; \
                a0[t * RS + j] = to_f(K_scaled[g]);                            \
                a1[t * RS + j] = to_f(Q_rot[g]);                               \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < M3T_MAX_SLOTS; k++) {                              \
            if (pa[k] >= 0) {                                                  \
                const float* kr = a0 + pa[k] * RS;                             \
                const float* qr = a1 + pb[k] * RS;                             \
                float acc = kq_acc[k];                                         \
                for (int j = 0; j < nw; j++) acc += kr[j] * qr[j];             \
                kq_acc[k] = acc;                                               \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    for (int p0 = 0; p0 < hd; p0 += TW) {                                      \
        const int pw = min(TW, hd - p0);                                       \
        __syncthreads();                                                       \
        for (int o = tid; o < len * TW; o += 256) {                            \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            if (j < pw) {                                                      \
                const long long g = (row0 + t) * d_inner + (long long)h * hd + p0 + j; \
                a3[t * RS + j] = to_f(V_in[g]);                                \
                a2[t * RS + j] = to_f(dO[g]);                                  \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < M3T_MAX_SLOTS; k++) {                              \
            if (pa[k] >= 0) {                                                  \
                const float* vr = a3 + pa[k] * RS;                             \
                const float* dr = a2 + pb[k] * RS;                             \
                float acc = vdo_acc[k];                                        \
                for (int j = 0; j < pw; j++) acc += vr[j] * dr[j];             \
                vdo_acc[k] = acc;                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int k = 0; k < M3T_MAX_SLOTS; k++) {                                  \
        if (pa[k] >= 0) {                                                      \
            const int idx = tid + 256 * k;                                     \
            kq_tri[idx] = kq_acc[k];                                           \
            vdo_tri[idx] = vdo_acc[k];                                         \
            dec_tri[idx] = exp2f((da_sm[pb[k]] - da_sm[pa[k]]) * LOG2E);       \
        }                                                                      \
    }                                                                          \
    /* dQK_dot. A warp takes a step, and lane l holds channel l of */     \
    /* each 32-channel segment. A head under a warp starts its tree at its */  \
    /* power of two, as the warp kernel's width-hd tree does. Lanes past */    \
    /* the head hold -0.0, which leaves every sum as it is. */                 \
    {                                                                          \
        const int warp = tid >> 5;                                             \
        const int lane = tid & 31;                                             \
        int seg_pow = 32;                                                      \
        while (seg_pow / 2 >= hd) seg_pow /= 2;                                \
        for (int t = warp; t < len; t += 8) {                                  \
            float total = 0.0f;                                                \
            for (int s0 = 0; s0 < hd; s0 += 32) {                              \
                const int p = s0 + lane;                                       \
                float val = -0.0f;                                             \
                if (p < hd) {                                                  \
                    const long long g = (row0 + t) * d_inner + (long long)h * hd + p; \
                    val = to_f(dO[g]) * to_f(V_in[g]);                         \
                }                                                              \
                for (int off = seg_pow / 2; off > 0; off >>= 1)                \
                    val += __shfl_down_sync(0xFFFFFFFFu, val, off, 32);        \
                total = (s0 == 0) ? val : total + val;                         \
            }                                                                  \
            if (lane == 0) {                                                   \
                dqk_sm[t] = total;                                             \
                dQK_dot_out[(row0 + t) * nh_total + h] = total;                \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /* Per head tile. Output o of a thread is step t = o / TW, */     \
    /* channel j = o % TW of the tile. */                                      \
    const int n_out = (64 * TW + 255) / 256;                                   \
    for (int p0 = 0; p0 < hd; p0 += TW) {                                      \
        const int pw = min(TW, hd - p0);                                       \
        float ke[M3T_MAX_OUT], qs[M3T_MAX_OUT];                                \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < M3T_MAX_OUT; k++) {                                \
            ke[k] = 0.0f;                                                      \
            qs[k] = 0.0f;                                                      \
        }                                                                      \
        for (int n0 = 0; n0 < ds; n0 += TW) {                                  \
            const int nw = min(TW, ds - n0);                                   \
            __syncthreads();                                                   \
            for (int o = tid; o < len * TW; o += 256) {                        \
                const int t = o / TW;                                          \
                const int j = o % TW;                                          \
                if (j < nw) {                                                  \
                    const long long g = ((row0 + t) * nh_total + h) * ds + n0 + j; \
                    a0[t * RS + j] = to_f(K_scaled[g]);                        \
                    a1[t * RS + j] = to_f(Q_rot[g]);                           \
                }                                                              \
            }                                                                  \
            for (int o = tid; o < pw * TW; o += 256) {                         \
                const int r = o / TW;                                          \
                const int j = o % TW;                                          \
                if (j < nw) {                                                  \
                    const long long g = (long long)(p0 + r) * ds + n0 + j;     \
                    s0[r * RS + j] = ssm[g];                                   \
                    s1[r * RS + j] = dst[g];                                   \
                }                                                              \
            }                                                                  \
            __syncthreads();                                                   \
            _Pragma("unroll")                                                  \
            for (int k = 0; k < M3T_MAX_OUT; k++) {                            \
                const int o = tid + 256 * k;                                   \
                const int t = o / TW;                                          \
                const int j = o % TW;                                          \
                if (k < n_out && t < len && j < pw) {                          \
                    const float* kr = a0 + t * RS;                             \
                    const float* qr = a1 + t * RS;                             \
                    const float* sr = s0 + j * RS;                             \
                    const float* dr = s1 + j * RS;                             \
                    float kacc = ke[k], qacc = qs[k];                          \
                    for (int c = 0; c < nw; c++) {                             \
                        kacc += kr[c] * dr[c];                                 \
                        qacc += qr[c] * sr[c];                                 \
                    }                                                          \
                    ke[k] = kacc;                                              \
                    qs[k] = qacc;                                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        /* The tile's dO and V, and its K.dS^T and Q.S^T where K and Q were. */ \
        for (int o = tid; o < len * TW; o += 256) {                            \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            if (j < pw) {                                                      \
                const long long g = (row0 + t) * d_inner + (long long)h * hd + p0 + j; \
                a2[t * RS + j] = to_f(dO[g]);                                  \
                a3[t * RS + j] = to_f(V_in[g]);                                \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < M3T_MAX_OUT; k++) {                                \
            const int o = tid + 256 * k;                                       \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            if (k < n_out && t < len && j < pw) {                              \
                a0[t * RS + j] = ke[k];                                        \
                a1[t * RS + j] = qs[k];                                        \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < M3T_MAX_OUT; k++) {                                \
            const int o = tid + 256 * k;                                       \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            if (k < n_out && t < len && j < pw) {                              \
                const VlSpan sp = vl_span(seq_idx, b, T, chunk_start, len, t); \
                float dv_intra = 0.0f;                                         \
                for (int s = t + 1; s < sp.end; s++) {                         \
                    const int ti = m3t_tri(t, s, CS);                          \
                    float kq = kq_tri[ti];                                     \
                    float decay = dec_tri[ti];                                 \
                    dv_intra += kq * decay * a2[s * RS + j];                   \
                }                                                              \
                float dv_inter = ke[k];                                        \
                if (!sp.to_end) dv_inter = 0.0f;                               \
                dv_inter *= er_sm[t];                                          \
                float dv_skip = a2[t * RS + j] * (D_val + qk_sm[t]);           \
                dV[(row0 + t) * d_inner + (long long)h * hd + p0 + j] =        \
                    dv_intra + dv_inter + dv_skip;                             \
            }                                                                  \
        }                                                                      \
        /* The dADT state terms: per step, chains over the channels in */      \
        /* order, continued from the previous tile. */                         \
        for (int t = tid; t < len; t += 256) {                                 \
            float qs_do = (p0 == 0) ? 0.0f : qsdo_sm[t];                       \
            float dsk_v = (p0 == 0) ? 0.0f : dskv_sm[t];                       \
            for (int j = 0; j < pw; j++) {                                     \
                qs_do += a1[t * RS + j] * a2[t * RS + j];                      \
                dsk_v += a0[t * RS + j] * a3[t * RS + j];                      \
            }                                                                  \
            qsdo_sm[t] = qs_do;                                                \
            dskv_sm[t] = dsk_v;                                                \
        }                                                                      \
    }                                                                          \
    /* Per state tile, dK and dQ. */                                  \
    for (int n0 = 0; n0 < ds; n0 += TW) {                                      \
        const int nw = min(TW, ds - n0);                                       \
        __syncthreads();                                                       \
        for (int o = tid; o < len * TW; o += 256) {                            \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            if (j < nw) {                                                      \
                const long long g = ((row0 + t) * nh_total + h) * ds + n0 + j; \
                a0[t * RS + j] = to_f(K_scaled[g]);                            \
                a1[t * RS + j] = to_f(Q_rot[g]);                               \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        float dki[M3T_MAX_OUT], dqi[M3T_MAX_OUT];                              \
        float dko[M3T_MAX_OUT], dqo[M3T_MAX_OUT];                              \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < M3T_MAX_OUT; k++) {                                \
            const int o = tid + 256 * k;                                       \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            dki[k] = 0.0f;                                                     \
            dqi[k] = 0.0f;                                                     \
            dko[k] = 0.0f;                                                     \
            dqo[k] = 0.0f;                                                     \
            if (k < n_out && t < len && j < nw) {                              \
                const VlSpan sp = vl_span(seq_idx, b, T, chunk_start, len, t); \
                float dk_intra = 0.0f;                                         \
                for (int s = t + 1; s < sp.end; s++) {                         \
                    const int ti = m3t_tri(t, s, CS);                          \
                    float vdo = vdo_tri[ti];                                   \
                    float decay = dec_tri[ti];                                 \
                    dk_intra += vdo * decay * a1[s * RS + j];                  \
                }                                                              \
                float dq_intra = 0.0f;                                         \
                for (int s = sp.first; s < t; s++) {                           \
                    const int ti = m3t_tri(s, t, CS);                          \
                    float vdo = vdo_tri[ti];                                   \
                    float decay = dec_tri[ti];                                 \
                    dq_intra += vdo * decay * a0[s * RS + j];                  \
                }                                                              \
                dki[k] = dk_intra;                                             \
                dqi[k] = dq_intra;                                             \
            }                                                                  \
        }                                                                      \
        for (int p0 = 0; p0 < hd; p0 += TW) {                                  \
            const int pw = min(TW, hd - p0);                                   \
            __syncthreads();                                                   \
            for (int o = tid; o < len * TW; o += 256) {                        \
                const int t = o / TW;                                          \
                const int j = o % TW;                                          \
                if (j < pw) {                                                  \
                    const long long g = (row0 + t) * d_inner + (long long)h * hd + p0 + j; \
                    a2[t * RS + j] = to_f(dO[g]);                              \
                    a3[t * RS + j] = to_f(V_in[g]);                            \
                }                                                              \
            }                                                                  \
            for (int o = tid; o < pw * TW; o += 256) {                         \
                const int r = o / TW;                                          \
                const int j = o % TW;                                          \
                if (j < nw) {                                                  \
                    const long long g = (long long)(p0 + r) * ds + n0 + j;     \
                    s0[r * RS + j] = ssm[g];                                   \
                    s1[r * RS + j] = dst[g];                                   \
                }                                                              \
            }                                                                  \
            __syncthreads();                                                   \
            _Pragma("unroll")                                                  \
            for (int k = 0; k < M3T_MAX_OUT; k++) {                            \
                const int o = tid + 256 * k;                                   \
                const int t = o / TW;                                          \
                const int j = o % TW;                                          \
                if (k < n_out && t < len && j < nw) {                          \
                    float qacc = dqo[k], kacc = dko[k];                        \
                    for (int c = 0; c < pw; c++) {                             \
                        qacc += a2[t * RS + c] * s0[c * RS + j];               \
                        kacc += a3[t * RS + c] * s1[c * RS + j];               \
                    }                                                          \
                    dqo[k] = qacc;                                             \
                    dko[k] = kacc;                                             \
                }                                                              \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < M3T_MAX_OUT; k++) {                                \
            const int o = tid + 256 * k;                                       \
            const int t = o / TW;                                              \
            const int j = o % TW;                                              \
            if (k < n_out && t < len && j < nw) {                              \
                const VlSpan sp = vl_span(seq_idx, b, T, chunk_start, len, t); \
                float dq_inter = dqo[k];                                       \
                if (!sp.from_before) dq_inter = 0.0f;                          \
                dq_inter *= ef_sm[t];                                          \
                float dk_inter = dko[k];                                       \
                if (!sp.to_end) dk_inter = 0.0f;                               \
                dk_inter *= er_sm[t];                                          \
                const long long g = ((row0 + t) * nh_total + h) * ds + n0 + j; \
                dQ_mid[g] = dqi[k] + dq_inter;                                 \
                dK_mid[g] = dki[k] + dk_inter;                                 \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    /* dADT, pairs below the diagonal, then above, then the */        \
    /* entering and exit states, as m3_dqkv orders them. */                    \
    for (int t = tid; t < len; t += 256) {                                     \
        float acc = 0.0f;                                                      \
        const VlSpan sp = vl_span(seq_idx, b, T, chunk_start, len, t);         \
        for (int i = sp.first; i < t; i++) {                                   \
            const int ti = m3t_tri(i, t, CS);                                  \
            float vdo = vdo_tri[ti];                                           \
            float decay = dec_tri[ti];                                         \
            float kq = kq_tri[ti];                                             \
            acc += vdo * decay * kq;                                           \
        }                                                                      \
        for (int j = t + 1; j < sp.end; j++) {                                 \
            const int ti = m3t_tri(t, j, CS);                                  \
            float vdo = vdo_tri[ti];                                           \
            float decay = dec_tri[ti];                                         \
            float kq = kq_tri[ti];                                             \
            acc -= vdo * decay * kq;                                           \
        }                                                                      \
        if (sp.from_before) acc += qsdo_sm[t] * ef_sm[t];                      \
        dmrev_sm[t] = acc;                                                     \
        dmvec_sm[t] = sp.to_end ? dskv_sm[t] * er_sm[t] : 0.0f;                \
    }                                                                          \
    __syncthreads();                                                           \
    if (tid == 0) {                                                            \
        float dM_scalar = dm_chain[cidx];                                      \
        dM_scalar *= exp2f(cs_sum * LOG2E);                                    \
        /* A packed row combines each of its sequences in the chunk on */      \
        /* its own, as m3_dqkv does. */                                        \
        const bool passes = vl_passes(seq_idx, b, T, chunk_start, len);        \
        for (int a = 0; a < len;) {                                            \
            const int e = vl_span(seq_idx, b, T, chunk_start, len, a).end;     \
            float total_rev = 0.0f;                                            \
            for (int t = a; t < e; t++) total_rev += dmrev_sm[t];              \
            if (passes) total_rev += dM_scalar;                                \
            float cumsum = 0.0f;                                               \
            for (int t = a; t < e; t++) {                                      \
                cumsum += dmvec_sm[t] - dmrev_sm[t];                           \
                const float out = dmrev_sm[t] + (total_rev + cumsum - dmvec_sm[t]); \
                dADT[(row0 + t) * nh_total + h] = out;                         \
            }                                                                  \
            a = e;                                                             \
        }                                                                      \
        float dD_acc = 0.0f;                                                   \
        for (int t = 0; t < len; t++) dD_acc += dqk_sm[t];                     \
        dD_partials[cidx] = dD_acc;                                            \
    }                                                                          \
}

DEFINE_M3_DQKV_TILED(m3_reference_dqkv_tiled, float)
DEFINE_M3_DQKV_TILED(m3_reference_dqkv_tiled_bf16, __nv_bfloat16)
DEFINE_M3_DQKV_TILED(m3_reference_dqkv_tiled_f16, __half)
#undef DEFINE_M3_DQKV_TILED
#undef M3T_MAX_SLOTS
#undef M3T_MAX_OUT
