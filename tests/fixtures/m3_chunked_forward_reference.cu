// Frozen kernels of the SISO chunked forward, as they were released before
// the sliced kernels replaced them: the pre-state trio (the preprocess, the
// decay cumsum and the chunk state in its scalar and quad layouts) and the
// two-head static-tile intra-chunk scan, in f32 and in the bf16 and f16
// twins. The sliced pre-state and scan are held bit for bit to these, at
// the shapes where the kernels that stage whole rows cannot run as well as
// where they can. The typed prelude and the packed-row helpers precede this
// fixture, and the test compiles it with the module's state capacity.

#ifndef MAMBA_RS_STATE_CAP
#define MAMBA_RS_STATE_CAP 64
#endif

#ifndef FAST_EXP
#define FAST_EXP(x) exp2f((x) * LOG2E)
#endif

// A decay exponent capped at zero; a NaN exponent stays NaN.
__device__ __forceinline__ float m3_nonpos(float x) {
    return x > 0.0f ? 0.0f : x;
}

// ============================================================================
// 1. m3_preprocess_chunks -- Compute shifted_gamma, scale, qk_dot, K prescaling
// ============================================================================
//
// For each timestep t:
//   gamma[t] = DT[t] * trap_sig[t]
//   shifted_gamma[t] = DT[t+1] * (1 - trap_sig[t+1])    (0 for last timestep)
//   scale[t] = shifted_gamma[t] + gamma[t]
//   qk_dot[t] = sum_n(Q[t,h,n] * K[t,h,n]) * gamma[t]
//   K_scaled[t,h,n] = K[t,h,n] * scale[t]
//
// Input:  K[B*T*nh*ds], Q[B*T*nh*ds], DT[B*T*nh], trap_sig[B*T*nh], D[nh]
// Output: K_scaled[B*T*nh*ds], qk_dot[B*T*nh], scale_out[B*T*nh], gamma_out[B*T*nh]
//
// Grid: (B*n_chunks, nh, 1), Block: (chunk_size, 1, 1)
// Each thread handles one timestep within a chunk, looping over ds for
// the QK dot product and K scaling.
#define DEFINE_M3_PREPROCESS_CHUNKS_F32(NAME, IDX)                             \
extern "C" __global__ void NAME(                                               \
    /* Outputs */                                                              \
    float* __restrict__ K_scaled,   /* [B * T * nh * ds] -- prescaled K */     \
    float* __restrict__ qk_dot,     /* [B * T * nh] */                         \
    float* __restrict__ scale_out,  /* [B * T * nh] */                         \
    float* __restrict__ gamma_out,  /* [B * T * nh] */                         \
    /* Inputs */                                                               \
    const float* __restrict__ K,        /* [B * T * nh * ds] (post bias+RoPE) */ \
    const float* __restrict__ Q,        /* [B * T * nh * ds] (post bias+RoPE) */ \
    const float* __restrict__ DT,       /* [B * T * nh] (post-softplus) */     \
    const float* __restrict__ trap_sig, /* [B * T * nh] (post-sigmoid) */      \
    int batch, int T, int nh, int ds, int chunk_size,                          \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                          \
    IDX bc = blockIdx.x;         /* batch * n_chunks */                        \
    IDX b = bc / n_chunks;                                                     \
    int chunk = (int)(bc % n_chunks);                                          \
    int h = y_first + blockIdx.y;                                              \
    int t_local = threadIdx.x;   /* position within chunk */                   \
                                                                               \
    int chunk_start = chunk * chunk_size;                                      \
    int t = chunk_start + t_local;                                             \
    if (t >= T) return;                                                        \
                                                                               \
    IDX th = (b * T + t) * nh + h;     /* index into [B*T*nh] arrays */        \
                                                                               \
    /* gamma[t] = DT[t] * trap_sig[t] */                                       \
    float dt_cur = DT[th];                                                     \
    float trap_cur = trap_sig[th];                                             \
    float gamma_val = dt_cur * trap_cur;                                       \
                                                                               \
    /* shifted_gamma[t] = DT[t+1] * (1 - trap_sig[t+1]), 0 for last timestep */ \
    float shifted_gamma = 0.0f;                                                \
    /* The next step's beta is 0 where it starts a sequence. */                \
    if (t + 1 < T && !vl_starts(seq_idx, b, T, t + 1)) {                       \
        IDX th_next = (b * T + t + 1) * nh + h;                                \
        float dt_next = DT[th_next];                                           \
        float trap_next = trap_sig[th_next];                                   \
        shifted_gamma = dt_next * (1.0f - trap_next);                          \
    }                                                                          \
                                                                               \
    float scale_val = shifted_gamma + gamma_val;                               \
                                                                               \
    /* Store scale and gamma */                                                \
    if (scale_out) scale_out[th] = scale_val;                                  \
    if (gamma_out) gamma_out[th] = gamma_val;                                  \
                                                                               \
    /* Compute qk_dot = sum_n(Q[t,h,n] * K[t,h,n]) * gamma. Each thread */     \
    /* owns one contiguous (t, h) row; the wide path issues float4 loads */    \
    /* and stores (4x fewer transactions) while the dot stays the same */      \
    /* serial n-ascending fold - bit-identical to the scalar path. */          \
    IDX kq_base = (b * T + t) * nh * ds + h * ds;                              \
    float dot = 0.0f;                                                          \
    if (ds % 4 == 0) {                                                         \
        const float4* q4 = reinterpret_cast<const float4*>(Q + kq_base);       \
        const float4* k4 = reinterpret_cast<const float4*>(K + kq_base);       \
        float4* ks4 = reinterpret_cast<float4*>(K_scaled + kq_base);           \
        for (int n4 = 0; n4 < ds / 4; n4++) {                                  \
            float4 qv = q4[n4];                                                \
            float4 kv = k4[n4];                                                \
            dot += qv.x * kv.x;                                                \
            dot += qv.y * kv.y;                                                \
            dot += qv.z * kv.z;                                                \
            dot += qv.w * kv.w;                                                \
            float4 sc;                                                         \
            sc.x = kv.x * scale_val;                                           \
            sc.y = kv.y * scale_val;                                           \
            sc.z = kv.z * scale_val;                                           \
            sc.w = kv.w * scale_val;                                           \
            ks4[n4] = sc;                                                      \
        }                                                                      \
    } else {                                                                   \
        for (int n = 0; n < ds; n++) {                                         \
            dot += Q[kq_base + n] * K[kq_base + n];                            \
        }                                                                      \
        for (int n = 0; n < ds; n++) {                                         \
            K_scaled[kq_base + n] = K[kq_base + n] * scale_val;                \
        }                                                                      \
    }                                                                          \
    qk_dot[th] = dot * gamma_val;                                              \
}

DEFINE_M3_PREPROCESS_CHUNKS_F32(m3_reference_preprocess_chunks, int)

// ============================================================================
// 2. m3_dA_cumsum -- Inclusive prefix sum of ADT within chunks
// ============================================================================
//
// Computes cumulative sum of adt = A * DT within each chunk, per head.
// Same as Mamba-2 pattern but with Mamba-3's input-dependent ADT. In a
// packed row (seq_idx) the sum restarts at every sequence's first token, so
// a sequence's decays never hold another's terms: every decay the chunked
// path reads lies inside one sequence, and an overflow in one sequence's
// a*dt cannot turn another's into inf - inf.
//
// Input:  adt[B * T * nh]  (A * DT, already computed by m3_split)
// Output: dA_cumsum[B * n_chunks * nh * chunk_size]
//
// Grid: (B, n_chunks, ceil(nh / blockDim.x)), Block: (min(nh, 256))
#define DEFINE_M3_DA_CUMSUM(NAME, IDX)                                         \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dA_cumsum_out,  /* [B * n_chunks * nh * chunk_size] */ \
    const float* __restrict__ adt,      /* [B * T * nh] */                     \
    int batch, int T, int nh, int chunk_size,                                  \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    IDX b = blockIdx.x;                                                        \
    int chunk = y_first + blockIdx.y;                                          \
    int h = (z_first + blockIdx.z) * blockDim.x + threadIdx.x;                 \
    if (h >= nh) return;                                                       \
                                                                               \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                          \
    int chunk_start = chunk * chunk_size;                                      \
    int chunk_end = chunk_start + chunk_size;                                  \
    if (chunk_end > T) chunk_end = T;                                          \
                                                                               \
    float cumsum = 0.0f;                                                       \
    IDX cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;              \
                                                                               \
    for (int t = chunk_start; t < chunk_end; t++) {                            \
        float a_dt = adt[(b * T + t) * nh + h];                                \
        cumsum = (vl_starts(seq_idx, b, T, t) ? 0.0f : cumsum) + a_dt;         \
        dA_cumsum_out[cs_base + (t - chunk_start)] = cumsum;                   \
    }                                                                          \
}

DEFINE_M3_DA_CUMSUM(m3_reference_da_cumsum, int)

// ============================================================================
// 3. m3_chunk_state_fwd -- Per-chunk state contribution (trapezoidal K_scaled)
// ============================================================================
//
// Computes per-chunk SSM state using prescaled K (already has trapezoidal scale):
//   chunk_states[c,h,p,n] = sum_t(exp(dA_end - dA_t) * K_scaled[t,h,n] * V[t,h,p])
//
// where V[t,h,p] = x[t, h*hd+p] and K_scaled already includes the scale factor.
//
// Input:  x[B*T*d_inner], K_scaled[B*T*nh*ds], dA_cumsum[B*n_chunks*nh*chunk_size]
// Output: states[B * n_chunks * nh * hd * ds]
//
// Grid: (B * n_chunks, ceil(nh / 2), 1). Block: (lanes, 2, 1) in the scalar
// layout, where a thread folds every state chain of one head element and a
// head wider than its lanes takes its elements in turns, or (hd, ds / 4, 2)
// in the quad layout (see chunk_state_cfg, which takes the lanes from the
// threads the kernel's registers allow).
#define DEFINE_M3_CHUNK_STATE_FWD_F32(NAME, IDX)                               \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ states_out,     /* [B * n_chunks * nh * hd * ds] */    \
    const float* __restrict__ x,        /* [B * T * d_inner] */                \
    const float* __restrict__ K_scaled, /* [B * T * nh * ds] */                \
    const float* __restrict__ dA_cumsum,/* [B * n_chunks * nh * chunk_size] */ \
    int batch, int T, int nh, int hd, int ds, int chunk_size,                  \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    int d_inner = nh * hd;                                                     \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                          \
    IDX bc = blockIdx.x;  /* batch * n_chunks */                               \
    IDX b = bc / n_chunks;                                                     \
    int chunk = (int)(bc % n_chunks);                                          \
    /* Two heads share one block so the 16-lane head dimension fills a */      \
    /* full warp (blockDim = (hd, 2)); each head's work stays private to */    \
    /* its own thread row. */                                                  \
    /* Thread layout, selected by the launcher (single source: */              \
    /* chunk_state_cfg in kernels.rs): */                                      \
    /* legacy: block (hd, heads)          - one thread owns all ds chains */   \
    /* quad:   block (hd, ds/4, heads)    - one thread owns FOUR chains */     \
    /* The quad layout exists for latency hiding (the t-loop is a serial */    \
    /* chain and the legacy grid leaves the SM starved at ~200 threads); */    \
    /* every acc[n] chain is still folded ascending-t by exactly one */        \
    /* thread with the identical expression - work redistribution only. */     \
    /* Unambiguous layout probe: ONLY the quad layout has blockDim.z >= 2 */   \
    /* (legacy is (hd, heads, 1), whose y is the HEAD axis - probing y */      \
    /* would misread it and walk a zero-byte dynamic smem allocation). */      \
    bool quad = blockDim.z >= 2;                                               \
    int h = quad ? ((y_first + blockIdx.y) * blockDim.z + threadIdx.z)         \
                 : ((y_first + blockIdx.y) * blockDim.y + threadIdx.y);        \
    if (ds > MAMBA_RS_STATE_CAP) return;  /* capacity guard for acc[] (uniform) */ \
                                                                               \
    int chunk_start = chunk * chunk_size;                                      \
    int chunk_end = chunk_start + chunk_size;                                  \
    if (chunk_end > T) chunk_end = T;                                          \
    int chunk_len = chunk_end - chunk_start;                                   \
    /* In a packed row (seq_idx) only the chunk's last sequence reaches */     \
    /* its state: the tokens from where that sequence begins. */               \
    int kept = vl_seq_first(seq_idx, b, T, chunk_start, chunk_end - 1) - chunk_start; \
    bool wide = (ds % 4 == 0);                                                 \
                                                                               \
    if (!quad) {                                                               \
        /* The legacy layout holds no barrier, so a head past the last one */  \
        /* leaves at once. */                                                  \
        if (h >= nh) return;                                                   \
        IDX cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;          \
        float dA_end = dA_cumsum[cs_base + chunk_len - 1];                     \
        for (int p = threadIdx.x; p < hd; p += blockDim.x) {                   \
            /* t outer / n inner: the decay exponential and the V load */      \
            /* depend only on t, so the swapped order computes each ONCE */    \
            /* per step instead of once per (step, state) pair. Every */       \
            /* acc[n] still sums its terms in ascending t with the */          \
            /* identical expression, so the bits are unchanged. */             \
            float acc[MAMBA_RS_STATE_CAP];                                     \
            for (int n = 0; n < ds; n++) acc[n] = 0.0f;                        \
            for (int t = chunk_start + kept; t < chunk_end; t++) {             \
                int t_local = t - chunk_start;                                 \
                float dA_t = dA_cumsum[cs_base + t_local];                     \
                float decay = FAST_EXP(m3_nonpos(dA_end - dA_t)); /* The cumsum diff is <=0 mathematically; the cap restores the reference safety net against float-noise positives (decay>1). */ \
                float v_t = x[(b * T + t) * d_inner + h * hd + p];             \
                IDX ks_base = (b * T + t) * nh * ds + h * ds;                  \
                for (int n = 0; n < ds; n++) {                                 \
                    acc[n] += decay * K_scaled[ks_base + n] * v_t;             \
                }                                                              \
            }                                                                  \
            IDX state_base = ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds; \
            if (wide) {                                                        \
                float4* so4 = reinterpret_cast<float4*>(states_out + state_base); \
                for (int n4 = 0; n4 < ds / 4; n4++) {                          \
                    float4 o;                                                  \
                    o.x = acc[n4 * 4 + 0];                                     \
                    o.y = acc[n4 * 4 + 1];                                     \
                    o.z = acc[n4 * 4 + 2];                                     \
                    o.w = acc[n4 * 4 + 3];                                     \
                    so4[n4] = o;                                               \
                }                                                              \
            } else {                                                           \
                for (int n = 0; n < ds; n++) states_out[state_base + n] = acc[n]; \
            }                                                                  \
        }                                                                      \
        return;                                                                \
    }                                                                          \
                                                                               \
    int p = threadIdx.x;                                                       \
    int n_lo = (int)threadIdx.y * 4;                                           \
    int n_hi = n_lo + 4;                                                       \
    /* No early return on the quad path - it holds a __syncthreads and an */   \
    /* out-of-range head (odd nh tail) must still reach the barrier. */        \
    bool live = (p < hd) && (h < nh);                                          \
                                                                               \
    /* Output base for this (b, chunk, h, p) */                                \
    IDX state_base = live ? ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds : 0; \
                                                                               \
    float acc[MAMBA_RS_STATE_CAP];                                             \
    for (int n = n_lo; n < n_hi; n++) acc[n] = 0.0f;                           \
    /* The kernel is L2-throughput bound (measured 76% at the serve shape) */  \
    /* and the quad thread split alone MULTIPLIES redundant global reads */    \
    /* (four threads per p re-read the same x element). So this path */        \
    /* stages the block's x/K/dA chunk slices in shared memory ONCE */         \
    /* (cooperative, coalesced) and the fold reads smem - identical */         \
    /* values, identical per-chain arithmetic, a fraction of the L2 */         \
    /* traffic. */                                                             \
    extern __shared__ float m3cst_sm[];                                        \
    int heads_pb = blockDim.z;                                                 \
    int h0 = (y_first + blockIdx.y) * heads_pb;                                \
    float* sx = m3cst_sm;                       /* [heads][cs_len][hd] */      \
    float* sk = sx + heads_pb * chunk_size * hd; /* [heads][cs_len][ds] */     \
    float* sda = sk + heads_pb * chunk_size * ds; /* [heads][cs_len] */        \
    int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x; \
    int nthr = blockDim.x * blockDim.y * blockDim.z;                           \
    for (int i = tid; i < heads_pb * chunk_len * hd; i += nthr) {              \
        int hz = i / (chunk_len * hd);                                         \
        int rem = i % (chunk_len * hd);                                        \
        int tl = rem / hd;                                                     \
        int pp = rem % hd;                                                     \
        int hg = h0 + hz;                                                      \
        sx[hz * chunk_size * hd + tl * hd + pp] = (hg < nh)                    \
            ? x[(long long)(b * T + chunk_start + tl) * d_inner + hg * hd + pp] \
            : 0.0f;                                                            \
    }                                                                          \
    for (int i = tid; i < heads_pb * chunk_len * ds; i += nthr) {              \
        int hz = i / (chunk_len * ds);                                         \
        int rem = i % (chunk_len * ds);                                        \
        int tl = rem / ds;                                                     \
        int nn = rem % ds;                                                     \
        int hg = h0 + hz;                                                      \
        sk[hz * chunk_size * ds + tl * ds + nn] = (hg < nh)                    \
            ? K_scaled[(long long)(b * T + chunk_start + tl) * nh * ds + hg * ds + nn] \
            : 0.0f;                                                            \
    }                                                                          \
    for (int i = tid; i < heads_pb * chunk_len; i += nthr) {                   \
        int hz = i / chunk_len;                                                \
        int tl = i % chunk_len;                                                \
        int hg = h0 + hz;                                                      \
        sda[hz * chunk_size + tl] = (hg < nh)                                  \
            ? dA_cumsum[((b * n_chunks + chunk) * nh + hg) * chunk_size + tl]  \
            : 0.0f;                                                            \
    }                                                                          \
    __syncthreads();                                                           \
    if (live) {                                                                \
        int hz = threadIdx.z;                                                  \
        const float* sxh = sx + hz * chunk_size * hd;                          \
        const float* skh = sk + hz * chunk_size * ds;                          \
        const float* sdah = sda + hz * chunk_size;                             \
        float dA_end_s = sdah[chunk_len - 1];                                  \
        for (int t_local = kept; t_local < chunk_len; t_local++) {             \
            float dA_t = sdah[t_local];                                        \
            float decay = FAST_EXP(m3_nonpos(dA_end_s - dA_t));                \
            float v_t = sxh[t_local * hd + p];                                 \
            const float4* k4 = reinterpret_cast<const float4*>(skh + t_local * ds); \
            for (int n4 = n_lo / 4; n4 < n_hi / 4; n4++) {                     \
                float4 kv = k4[n4];                                            \
                acc[n4 * 4 + 0] += decay * kv.x * v_t;                         \
                acc[n4 * 4 + 1] += decay * kv.y * v_t;                         \
                acc[n4 * 4 + 2] += decay * kv.z * v_t;                         \
                acc[n4 * 4 + 3] += decay * kv.w * v_t;                         \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    if (!live) return;                                                         \
    if (wide) {                                                                \
        float4* so4 = reinterpret_cast<float4*>(states_out + state_base);      \
        for (int n4 = n_lo / 4; n4 < n_hi / 4; n4++) {                         \
            float4 o;                                                          \
            o.x = acc[n4 * 4 + 0];                                             \
            o.y = acc[n4 * 4 + 1];                                             \
            o.z = acc[n4 * 4 + 2];                                             \
            o.w = acc[n4 * 4 + 3];                                             \
            so4[n4] = o;                                                       \
        }                                                                      \
    } else {                                                                   \
        for (int n = n_lo; n < n_hi; n++) states_out[state_base + n] = acc[n]; \
    }                                                                          \
}

DEFINE_M3_CHUNK_STATE_FWD_F32(m3_reference_chunk_state, int)

// ============================================================================
// 5. m3_chunk_scan_fwd -- Intra-chunk output computation
// ============================================================================
//
// Computes output for each timestep within a chunk:
//   Y[t,h,p] = sum_n(Q[t,h,n] * state[h,p,n]) * exp(dA[t])        // state contribution
//            + sum_{s<=t}(exp(dA[t]-dA[s]) * QK_scaled_dot * V[s])  // intra-chunk
//            + (D[h] + qk_dot[t,h]) * V[t,h,p]                      // D + qk skip
//
// where QK_scaled_dot = Q[t]^T @ K_scaled[s] (dot product of Q at t with K_scaled at s).
// Q and K_scaled are in the d_state dimension (not headdim).
//
// Grid: (B * n_chunks, ceil(nh / 2), 1), Block: (lanes, 2, 1), the lanes
// at most the head and half the threads the kernel's registers allow (see
// chunk_scan_cfg). Each thread computes the output of one (b, chunk, h, p)
// across all timesteps; a head wider than its lanes takes its elements in
// turns.
#define DEFINE_M3_CHUNK_SCAN_FWD_F32(NAME, IDX)                                \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ y_out,          /* [B * T * d_inner] */                \
    const float* __restrict__ x,        /* [B * T * d_inner] */                \
    const float* __restrict__ Q,        /* [B * T * nh * ds] (post bias+RoPE) */ \
    const float* __restrict__ K_scaled, /* [B * T * nh * ds] (prescaled by scale) */ \
    const float* __restrict__ qk_dot,   /* [B * T * nh] (precomputed D-skip term) */ \
    const float* __restrict__ dA_cumsum,/* [B * n_chunks * nh * chunk_size] */ \
    const float* __restrict__ prev_states, /* [B * n_chunks * nh * hd * ds] (entering states) */ \
    const float* __restrict__ D,        /* [nh] */                             \
    int batch, int T, int nh, int hd, int ds, int chunk_size,                  \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    int d_inner = nh * hd;                                                     \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                          \
    IDX bc = blockIdx.x;                                                       \
    IDX b = bc / n_chunks;                                                     \
    int chunk = (int)(bc % n_chunks);                                          \
    /* Two heads share one block (blockDim = (lanes, 2)) so a 16-lane head */  \
    /* dimension fills a full warp; each head row owns its own tile. */        \
    int h = (y_first + blockIdx.y) * blockDim.y + threadIdx.y;                 \
    int head_row = threadIdx.y;                                                \
    bool head_live = h < nh;                                                   \
                                                                               \
    int chunk_start = chunk * chunk_size;                                      \
    int chunk_end = chunk_start + chunk_size;                                  \
    if (chunk_end > T) chunk_end = T;                                          \
    int chunk_len = chunk_end - chunk_start;                                   \
    /* In a packed row (seq_idx) the entering state reaches only the */        \
    /* tokens before the chunk's first start, and a token's pairs stay */      \
    /* inside its own sequence. */                                             \
    int cont = vl_lower_bound(seq_idx, b, T, chunk_start, chunk_end,           \
                              vl_id(seq_idx, b, T, chunk_start - 1) + 1) - chunk_start; \
    if (chunk_size > 64) return;  /* tile rows are sized for the fixed chunk of 64 */ \
    if (ds > MAMBA_RS_STATE_CAP) return;  /* capacity guard for q_reg[] (uniform: all threads agree) */ \
                                                                               \
    /* dA_cumsum base for this (b, chunk, h) */                                \
    IDX cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;              \
                                                                               \
    float d_skip = head_live ? D[h] : 0.0f;                                    \
                                                                               \
    /* The decayed causal attention tile decay(t,s) * (Q[t] . K_scaled[s]) */  \
    /* is identical for every lane p, so it is computed ONCE per */            \
    /* (chunk, head) here instead of once per lane: rows are strided over */   \
    /* the lanes, each row stages Q[t] in registers and dots it against */     \
    /* the K rows below the diagonal. Every tile element is written by */      \
    /* exactly one lane and every dot runs n ascending, so the arithmetic */   \
    /* and its order match the per-lane original. Inactive threads skip */     \
    /* the work but still reach the barrier. */                                \
    __shared__ float qk_tile[2][64 * 64];                                      \
    if (head_live) {                                                           \
        for (int t_local = threadIdx.x; t_local < chunk_len; t_local += blockDim.x) { \
            int t = chunk_start + t_local;                                     \
            float dA_t = dA_cumsum[cs_base + t_local];                         \
            IDX q_base = (b * T + t) * nh * ds + h * ds;                       \
            float q_reg[MAMBA_RS_STATE_CAP];                                   \
            for (int n = 0; n < ds; n++) q_reg[n] = Q[q_base + n];             \
            int lo = vl_seq_first(seq_idx, b, T, chunk_start, t) - chunk_start; \
            for (int s_local = lo; s_local < t_local; s_local++) {             \
                int s = chunk_start + s_local;                                 \
                float dA_s = dA_cumsum[cs_base + s_local];                     \
                float decay = FAST_EXP(m3_nonpos(dA_t - dA_s)); /* Same float-noise safety net as the state-passing kernel above. */ \
                float qk_val = 0.0f;                                           \
                IDX ks_base = (b * T + s) * nh * ds + h * ds;                  \
                for (int n = 0; n < ds; n++) {                                 \
                    qk_val += q_reg[n] * K_scaled[ks_base + n];                \
                }                                                              \
                qk_tile[head_row][t_local * 64 + s_local] = decay * qk_val;    \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    if (!head_live) return;                                                    \
                                                                               \
    for (int p = threadIdx.x; p < hd; p += blockDim.x) {                       \
        /* Prev state for this (b, chunk, h, p) */                             \
        IDX state_base = ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds; \
        for (int t_local = 0; t_local < chunk_len; t_local++) {                \
            int t = chunk_start + t_local;                                     \
            float dA_t = dA_cumsum[cs_base + t_local];                         \
                                                                               \
            /* Y_off: state contribution = sum_n(Q[t,n] * prev_state[p,n]) * exp(dA[t]) */ \
            float y_off = 0.0f;                                                \
            float state_decay = FAST_EXP(dA_t);                                \
            IDX q_base = (b * T + t) * nh * ds + h * ds;                       \
            for (int n = 0; n < ds; n++) {                                     \
                y_off += Q[q_base + n] * prev_states[state_base + n];          \
            }                                                                  \
            if (t_local >= cont) y_off = 0.0f; y_off *= state_decay;           \
                                                                               \
            /* Y_diag: intra-chunk contribution from positions s <= t (strictly */ \
            /* causal + diagonal). For s < t the decayed Q.K factor comes from */ \
            /* the shared tile; s == t rides qk_dot (gamma contribution). */   \
            float y_diag = 0.0f;                                               \
            int lo = vl_seq_first(seq_idx, b, T, chunk_start, t) - chunk_start; \
            for (int s_local = lo; s_local < t_local; s_local++) {             \
                int s = chunk_start + s_local;                                 \
                float v_s = x[(b * T + s) * d_inner + h * hd + p];             \
                y_diag += qk_tile[head_row][t_local * 64 + s_local] * v_s;     \
            }                                                                  \
                                                                               \
            /* D + qk_dot skip: (D[h] + qk_dot[t,h]) * V[t,h,p] */             \
            float x_t = x[(b * T + t) * d_inner + h * hd + p];                 \
            IDX th = (b * T + t) * nh + h;                                     \
            float y_skip = (d_skip + qk_dot[th]) * x_t;                        \
                                                                               \
            y_out[(b * T + t) * d_inner + h * hd + p] = y_diag + y_off + y_skip; \
        }                                                                      \
    }                                                                          \
}

DEFINE_M3_CHUNK_SCAN_FWD_F32(m3_reference_chunk_scan, int)

#define DEFINE_M3_PREPROCESS_CHUNKS(NAME, IDX, T_ACT, FROM_F)                 \
extern "C" __global__ void                                                    \
NAME(                                                                         \
    T_ACT* __restrict__ K_scaled,                                             \
    float* __restrict__ qk_dot,                                               \
    float* __restrict__ scale_out,                                            \
    float* __restrict__ gamma_out,                                            \
    const T_ACT* __restrict__ K,                                              \
    const T_ACT* __restrict__ Q,                                              \
    const float* __restrict__ DT,                                             \
    const float* __restrict__ trap_sig,                                       \
    int batch, int T, int nh, int ds, int chunk_size,                         \
    int y_first, /* this run's first y block */                               \
    const int* __restrict__ seq_idx                                           \
) {                                                                           \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                         \
    IDX bc = blockIdx.x;                                                      \
    IDX b = bc / n_chunks;                                                    \
    int chunk = (int)(bc % n_chunks);                                         \
    int h = y_first + blockIdx.y;                                             \
    int t_local = threadIdx.x;                                                \
    int chunk_start = chunk * chunk_size;                                     \
    int t = chunk_start + t_local;                                            \
    if (t >= T) return;                                                       \
    IDX th = (b * T + t) * nh + h;                                            \
    float dt_cur = DT[th];                                                    \
    float trap_cur = trap_sig[th];                                            \
    float gamma_val = dt_cur * trap_cur;                                      \
    float shifted_gamma = 0.0f;                                               \
    /* The next step's beta is 0 where it starts a sequence. */               \
    if (t + 1 < T && !vl_starts(seq_idx, b, T, t + 1)) {                      \
        IDX th_next = (b * T + t + 1) * nh + h;                               \
        float dt_next = DT[th_next];                                          \
        float trap_next = trap_sig[th_next];                                  \
        shifted_gamma = dt_next * (1.0f - trap_next);                         \
    }                                                                         \
    float scale_val = shifted_gamma + gamma_val;                              \
    if (scale_out) scale_out[th] = scale_val;                                 \
    if (gamma_out) gamma_out[th] = gamma_val;                                 \
    IDX kq_base = (b * T + t) * nh * ds + h * ds;                             \
    float dot = 0.0f;                                                         \
    /* Wide path: one uint4 = 8 typed elements per load/store (the rows  */   \
    /* are contiguous and 16B-aligned when ds % 8 == 0); the fold and    */   \
    /* the conversions run in the same n-ascending order either way.     */   \
    if (ds % 8 == 0) {                                                        \
        for (int n8 = 0; n8 < ds / 8; n8++) {                                 \
            union { uint4 u; T_ACT e[8]; } qq, kk, ss;                        \
            qq.u = *reinterpret_cast<const uint4*>(Q + kq_base + n8 * 8);     \
            kk.u = *reinterpret_cast<const uint4*>(K + kq_base + n8 * 8);     \
            for (int j = 0; j < 8; j++) {                                     \
                dot += to_f(qq.e[j]) * to_f(kk.e[j]);                         \
            }                                                                 \
            for (int j = 0; j < 8; j++) {                                     \
                ss.e[j] = FROM_F(to_f(kk.e[j]) * scale_val);                  \
            }                                                                 \
            *reinterpret_cast<uint4*>(K_scaled + kq_base + n8 * 8) = ss.u;    \
        }                                                                     \
    } else {                                                                  \
        for (int n = 0; n < ds; n++) {                                        \
            dot += to_f(Q[kq_base + n]) * to_f(K[kq_base + n]);               \
        }                                                                     \
        for (int n = 0; n < ds; n++) {                                        \
            float ks = to_f(K[kq_base + n]) * scale_val;                      \
            K_scaled[kq_base + n] = FROM_F(ks);                               \
        }                                                                     \
    }                                                                         \
    qk_dot[th] = dot * gamma_val;                                             \
}

DEFINE_M3_PREPROCESS_CHUNKS(m3_reference_preprocess_chunks_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_M3_PREPROCESS_CHUNKS(m3_reference_preprocess_chunks_f16, int, __half, from_f_f16)

/* The typed twin of m3_chunk_state_fwd: the same layouts and chains, the
 * activations read through to_f. */
#define DEFINE_M3_CHUNK_STATE_FWD(NAME, IDX, T_ACT, FROM_F)                   \
extern "C" __global__ void                                                    \
NAME(                                                                         \
    float* __restrict__ states_out,                                           \
    const T_ACT* __restrict__ x,                                              \
    const T_ACT* __restrict__ K_scaled,                                       \
    const float* __restrict__ dA_cumsum,                                      \
    int batch, int T, int nh, int hd, int ds, int chunk_size,                 \
    int y_first, /* this run's first y block */                               \
    const int* __restrict__ seq_idx                                           \
) {                                                                           \
    int d_inner = nh * hd;                                                    \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                         \
    IDX bc = blockIdx.x;                                                      \
    IDX b = bc / n_chunks;                                                    \
    int chunk = (int)(bc % n_chunks);                                         \
    /* Layout probe by blockDim.z (legacy y is the HEAD axis, see f32). */    \
    bool quad = blockDim.z >= 2;                                              \
    int h = quad ? ((y_first + blockIdx.y) * blockDim.z + threadIdx.z)        \
                 : ((y_first + blockIdx.y) * blockDim.y + threadIdx.y);       \
    if (ds > MAMBA_RS_STATE_CAP) return;                                      \
    int chunk_start = chunk * chunk_size;                                     \
    int chunk_end = chunk_start + chunk_size;                                 \
    if (chunk_end > T) chunk_end = T;                                         \
    int chunk_len = chunk_end - chunk_start;                                  \
    /* In a packed row (seq_idx) only the chunk's last sequence reaches */    \
    /* its state: the tokens from where that sequence begins. */              \
    int kept = vl_seq_first(seq_idx, b, T, chunk_start, chunk_end - 1) - chunk_start; \
    if (!quad) {                                                              \
        if (h >= nh) return;                                                  \
        IDX cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;         \
        float dA_end = dA_cumsum[cs_base + chunk_len - 1];                    \
        for (int p = threadIdx.x; p < hd; p += blockDim.x) {                  \
            float acc[MAMBA_RS_STATE_CAP];                                    \
            for (int n = 0; n < ds; n++) acc[n] = 0.0f;                       \
            for (int t = chunk_start + kept; t < chunk_end; t++) {            \
                int t_local = t - chunk_start;                                \
                float dA_t = dA_cumsum[cs_base + t_local];                    \
                float decay = FAST_EXP(m3_nonpos(dA_end - dA_t));             \
                float v_t = to_f(x[(b * T + t) * d_inner + h * hd + p]);      \
                IDX ks_base = (b * T + t) * nh * ds + h * ds;                 \
                for (int n = 0; n < ds; n++) {                                \
                    acc[n] += decay * to_f(K_scaled[ks_base + n]) * v_t;      \
                }                                                             \
            }                                                                 \
            IDX state_base =                                                  \
                ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds;         \
            for (int n = 0; n < ds; n++) states_out[state_base + n] = acc[n]; \
        }                                                                     \
        return;                                                               \
    }                                                                         \
    int p = threadIdx.x;                                                      \
    int n_lo = (int)threadIdx.y * 4;                                          \
    int n_hi = n_lo + 4;                                                      \
    bool live = (p < hd) && (h < nh);                                         \
    IDX state_base =                                                          \
        live ? ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds : 0;      \
    float acc[MAMBA_RS_STATE_CAP];                                            \
    for (int n = n_lo; n < n_hi; n++) acc[n] = 0.0f;                          \
    /* Staged as f32 via to_f: same values the direct loop read. */           \
    extern __shared__ float m3cst_sm[];                                       \
    int heads_pb = blockDim.z;                                                \
    int h0 = (y_first + blockIdx.y) * heads_pb;                               \
    float* sx = m3cst_sm;                                                     \
    float* sk = sx + heads_pb * chunk_size * hd;                              \
    float* sda = sk + heads_pb * chunk_size * ds;                             \
    int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x           \
              + threadIdx.x;                                                  \
    int nthr = blockDim.x * blockDim.y * blockDim.z;                          \
    for (int i = tid; i < heads_pb * chunk_len * hd; i += nthr) {             \
        int hz = i / (chunk_len * hd);                                        \
        int rem = i % (chunk_len * hd);                                       \
        int tl = rem / hd;                                                    \
        int pp = rem % hd;                                                    \
        int hg = h0 + hz;                                                     \
        sx[hz * chunk_size * hd + tl * hd + pp] = (hg < nh)                   \
            ? to_f(x[(long long)(b * T + chunk_start + tl) * d_inner          \
                     + hg * hd + pp])                                         \
            : 0.0f;                                                           \
    }                                                                         \
    for (int i = tid; i < heads_pb * chunk_len * ds; i += nthr) {             \
        int hz = i / (chunk_len * ds);                                        \
        int rem = i % (chunk_len * ds);                                       \
        int tl = rem / ds;                                                    \
        int nn = rem % ds;                                                    \
        int hg = h0 + hz;                                                     \
        sk[hz * chunk_size * ds + tl * ds + nn] = (hg < nh)                   \
            ? to_f(K_scaled[(long long)(b * T + chunk_start + tl)             \
                            * nh * ds + hg * ds + nn])                        \
            : 0.0f;                                                           \
    }                                                                         \
    for (int i = tid; i < heads_pb * chunk_len; i += nthr) {                  \
        int hz = i / chunk_len;                                               \
        int tl = i % chunk_len;                                               \
        int hg = h0 + hz;                                                     \
        sda[hz * chunk_size + tl] = (hg < nh)                                 \
            ? dA_cumsum[((b * n_chunks + chunk) * nh + hg) * chunk_size       \
                        + tl]                                                 \
            : 0.0f;                                                           \
    }                                                                         \
    __syncthreads();                                                          \
    if (live) {                                                               \
        int hz = threadIdx.z;                                                 \
        const float* sxh = sx + hz * chunk_size * hd;                         \
        const float* skh = sk + hz * chunk_size * ds;                         \
        const float* sdah = sda + hz * chunk_size;                            \
        float dA_end_s = sdah[chunk_len - 1];                                 \
        for (int t_local = kept; t_local < chunk_len; t_local++) {            \
            float dA_t = sdah[t_local];                                       \
            float decay = FAST_EXP(m3_nonpos(dA_end_s - dA_t));               \
            float v_t = sxh[t_local * hd + p];                                \
            const float4* k4 =                                                \
                reinterpret_cast<const float4*>(skh + t_local * ds);          \
            for (int n4 = n_lo / 4; n4 < n_hi / 4; n4++) {                    \
                float4 kv = k4[n4];                                           \
                acc[n4 * 4 + 0] += decay * kv.x * v_t;                        \
                acc[n4 * 4 + 1] += decay * kv.y * v_t;                        \
                acc[n4 * 4 + 2] += decay * kv.z * v_t;                        \
                acc[n4 * 4 + 3] += decay * kv.w * v_t;                        \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    if (!live) return;                                                        \
    for (int n = n_lo; n < n_hi; n++) states_out[state_base + n] = acc[n];    \
    (void)FROM_F;                                                             \
}

DEFINE_M3_CHUNK_STATE_FWD(m3_reference_chunk_state_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_M3_CHUNK_STATE_FWD(m3_reference_chunk_state_f16, int, __half, from_f_f16)

#define DEFINE_M3_CHUNK_SCAN_FWD(NAME, IDX, T_ACT, FROM_F)                    \
/* No __launch_bounds__ pin: the old (64, 2) minimum-blocks hint was          \
 * unreachable behind the 32 KB static tile (smem caps residency at 3)        \
 * and only constrained the register allocator. A head wider than the         \
 * block takes its elements in turns, as in the f32 kernel. */                \
extern "C" __global__ void                                                    \
NAME(                                                                         \
    T_ACT* __restrict__ y_out,                                                \
    const T_ACT* __restrict__ x,                                              \
    const T_ACT* __restrict__ Q,                                              \
    const T_ACT* __restrict__ K_scaled,                                       \
    const float* __restrict__ qk_dot,                                         \
    const float* __restrict__ dA_cumsum,                                      \
    const float* __restrict__ prev_states,                                    \
    const float* __restrict__ D,                                              \
    int batch, int T, int nh, int hd, int ds, int chunk_size,                 \
    int y_first, /* this run's first y block */                               \
    const int* __restrict__ seq_idx                                           \
) {                                                                           \
    int d_inner = nh * hd;                                                    \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                         \
    IDX bc = blockIdx.x;                                                      \
    IDX b = bc / n_chunks;                                                    \
    int chunk = (int)(bc % n_chunks);                                         \
    int h = (y_first + blockIdx.y) * blockDim.y + threadIdx.y;                \
    int head_row = threadIdx.y;                                               \
    bool head_live = h < nh;                                                  \
    int chunk_start = chunk * chunk_size;                                     \
    int chunk_end = chunk_start + chunk_size;                                 \
    if (chunk_end > T) chunk_end = T;                                         \
    int chunk_len = chunk_end - chunk_start;                                  \
    /* In a packed row (seq_idx) the entering state reaches only the */       \
    /* tokens before the chunk's first start, and a token's pairs stay */     \
    /* inside its own sequence. */                                            \
    int cont = vl_lower_bound(seq_idx, b, T, chunk_start, chunk_end,          \
                              vl_id(seq_idx, b, T, chunk_start - 1) + 1) - chunk_start; \
    if (chunk_size > 64) return;                                              \
    if (ds > MAMBA_RS_STATE_CAP) return;                                      \
    IDX cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;             \
    float d_skip = head_live ? D[h] : 0.0f;                                   \
    __shared__ float qk_tile[2][64 * 64];                                     \
    if (head_live) {                                                          \
        for (int t_local = threadIdx.x; t_local < chunk_len;                  \
             t_local += blockDim.x) {                                         \
            int t = chunk_start + t_local;                                    \
            float dA_t = dA_cumsum[cs_base + t_local];                        \
            IDX q_base = (b * T + t) * nh * ds + h * ds;                      \
            float q_reg[MAMBA_RS_STATE_CAP];                                  \
            for (int n = 0; n < ds; n++) q_reg[n] = to_f(Q[q_base + n]);      \
            int lo = vl_seq_first(seq_idx, b, T, chunk_start, t) - chunk_start; \
            for (int s_local = lo; s_local < t_local; s_local++) {            \
                int s = chunk_start + s_local;                                \
                float dA_s = dA_cumsum[cs_base + s_local];                    \
                float decay = FAST_EXP(m3_nonpos(dA_t - dA_s));               \
                float qk_val = 0.0f;                                          \
                IDX ks_base = (b * T + s) * nh * ds + h * ds;                 \
                for (int n = 0; n < ds; n++) {                                \
                    qk_val += q_reg[n] * to_f(K_scaled[ks_base + n]);         \
                }                                                             \
                qk_tile[head_row][t_local * 64 + s_local] = decay * qk_val;   \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    __syncthreads();                                                          \
    if (!head_live) return;                                                   \
    for (int p = threadIdx.x; p < hd; p += blockDim.x) {                      \
        IDX state_base =                                                      \
            ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds;             \
        for (int t_local = 0; t_local < chunk_len; t_local++) {               \
            int t = chunk_start + t_local;                                    \
            float dA_t = dA_cumsum[cs_base + t_local];                        \
            float y_off = 0.0f;                                               \
            float state_decay = FAST_EXP(dA_t);                               \
            IDX q_base = (b * T + t) * nh * ds + h * ds;                      \
            for (int n = 0; n < ds; n++) {                                    \
                y_off += to_f(Q[q_base + n]) * prev_states[state_base + n];   \
            }                                                                 \
            if (t_local >= cont) y_off = 0.0f; y_off *= state_decay;          \
            float y_diag = 0.0f;                                              \
            int lo = vl_seq_first(seq_idx, b, T, chunk_start, t) - chunk_start; \
            for (int s_local = lo; s_local < t_local; s_local++) {            \
                int s = chunk_start + s_local;                                \
                float v_s = to_f(x[(b * T + s) * d_inner + h * hd + p]);      \
                y_diag += qk_tile[head_row][t_local * 64 + s_local] * v_s;    \
            }                                                                 \
            float x_t = to_f(x[(b * T + t) * d_inner + h * hd + p]);          \
            IDX th = (b * T + t) * nh + h;                                    \
            float y_skip = (d_skip + qk_dot[th]) * x_t;                       \
            float y_val = y_diag + y_off + y_skip;                            \
            y_out[(b * T + t) * d_inner + h * hd + p] = FROM_F(y_val);        \
        }                                                                     \
    }                                                                         \
}

DEFINE_M3_CHUNK_SCAN_FWD(m3_reference_chunk_scan_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_M3_CHUNK_SCAN_FWD(m3_reference_chunk_scan_f16, int, __half, from_f_f16)
