// The dQK-theta stage of the SISO chunked backward as it was released
// before the row-streaming stage replaced it: a thread held its whole row in
// state-capacity register arrays, read through six staged tiles or straight
// from global memory, and the half transport form on sm_89 staged four. The
// row-streaming stage is held to each of them bit for bit. The typed prelude
// precedes this fixture, and the test compiles it with the module's state
// capacity.

#ifndef MAMBA_RS_STATE_CAP
#define MAMBA_RS_STATE_CAP 64
#endif

// ============================================================================
// 12. m3_dqktheta -- Inverse RoPE + scale + bias gradients (reference backward stage 3)
// ============================================================================
// Grid: (B*n_chunks, nh), Block: (CS, 1, 1)
// Each thread handles one timestep within a chunk.
#define DEFINE_M3_DQKTHETA_F32(NAME, IDX)                                      \
extern "C" __global__ void NAME(                                               \
    /* Outputs */                                                              \
    float* __restrict__ dQ_pre,         /* [B*T*nh*ds] — pre-RoPE Q gradient */ \
    float* __restrict__ dK_pre,         /* [B*T*nh*ds] — pre-RoPE K gradient */ \
    float* __restrict__ dAngles_cumsum, /* [B*T*nh*n_angles] */                \
    /* NO __restrict__ on dScale/dGamma/Scale_in/Gamma_in: the host passes */  \
    /* the SAME buffer as Scale_in and dScale (resp. Gamma_in/dGamma) — the */ \
    /* value is loaded into a register before the aliased store, which is */   \
    /* well-defined only without the no-alias promise. */                      \
    float* dScale,                      /* [B*T*nh] (may alias Scale_in) */    \
    float* dGamma,                      /* [B*T*nh] (may alias Gamma_in) */    \
    /* dQ_bias/dK_bias accumulators removed from kernel — caller */            \
    /* does colsum_accumulate on dQ_pre/dK_pre (already per-(b,t,h,n) scratch). */ \
    /* Deterministic, no atomicAdd. */                                         \
    /* Inputs */                                                               \
    const float* __restrict__ Q_raw,    /* [B*T*nh*ds] — acts.c_biased (pre-RoPE) */ \
    const float* __restrict__ K_raw,    /* [B*T*nh*ds] — acts.b_biased (pre-RoPE) */ \
    const float* Scale_in,              /* [B*T*nh] (may alias dScale) */      \
    const float* Gamma_in,              /* [B*T*nh] (may alias dGamma) */      \
    const float* __restrict__ Angles,   /* [B*T*nh*n_angles] — acts.angle_cumsum */ \
    const float* __restrict__ dQ_mid,   /* [B*T*nh*ds] — from m3_dqkv */       \
    const float* __restrict__ dK_mid,   /* [B*T*nh*ds] — from m3_dqkv */       \
    const float* __restrict__ dQK_dot,  /* [B*T*nh] — from m3_dqkv */          \
    int B, int T, int nh, int ds, int n_angles, int CS,                        \
    int use_staging,  /* 1 = smem holds the six [CS][ds] I/O tiles */          \
    int y_first /* this run's first y block */                                 \
) {                                                                            \
    int n_chunks = (T + CS - 1) / CS;                                          \
    IDX bc = blockIdx.x;                                                       \
    IDX b_chunk = bc; /* b * n_chunks + chunk */                               \
    int h = y_first + blockIdx.y;                                              \
    int t_local = threadIdx.x;                                                 \
                                                                               \
    IDX b = b_chunk / n_chunks;                                                \
    int chunk = (int)(b_chunk % n_chunks);                                     \
    int gt = chunk * CS + t_local;                                             \
                                                                               \
    /* Capacity guard for the q_pre/k_pre/dq_in register arrays. All */        \
    /* threads of the block read the same `ds` arg -> safe early return */     \
    /* (before any barrier, uniform across the block). */                      \
    if (ds > MAMBA_RS_STATE_CAP) return;                                       \
    if (b >= B || h >= nh) return;                                             \
                                                                               \
    bool valid = (gt < T);                                                     \
                                                                               \
    /* Coalesced staging: the per-thread row loads/stores put adjacent */      \
    /* threads nh*ds floats apart (one 32-byte sector each). Threads */        \
    /* jointly stream the [CS][ds] tiles instead; every thread then */         \
    /* reads ITS OWN row from shared memory — identical values, so the */      \
    /* per-thread arithmetic (and bits) are unchanged. */                      \
    extern __shared__ float dqkt_sm[];                                         \
    float* q_sm     = dqkt_sm;                                                 \
    float* k_sm     = q_sm + CS * ds;                                          \
    float* dqi_sm   = k_sm + CS * ds;                                          \
    float* dki_sm   = dqi_sm + CS * ds;                                        \
    float* out_q_sm = dki_sm + CS * ds;                                        \
    float* out_k_sm = out_q_sm + CS * ds;                                      \
    int t0 = chunk * CS;                                                       \
    int rows = min(CS, T - t0);                                                \
    /* Tiles past the block's shared-memory budget make the launcher */        \
    /* pass use_staging = 0, and every access */                               \
    /* falls back to the direct global form — identical values, only */        \
    /* the coalescing win is lost. */                                          \
    if (use_staging) {                                                         \
        for (int i = t_local; i < rows * ds; i += CS) {                        \
            int tt = i / ds;                                                   \
            int nn = i % ds;                                                   \
            IDX gbase = ((b * T + t0 + tt) * nh + h) * ds + nn;                \
            q_sm[i] = Q_raw[gbase];                                            \
            k_sm[i] = K_raw[gbase];                                            \
            dqi_sm[i] = dQ_mid[gbase];                                         \
            dki_sm[i] = dK_mid[gbase];                                         \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
                                                                               \
    if (valid) {                                                               \
        IDX base = ((b * T + gt) * nh + h) * ds;                               \
        float scale = Scale_in[(b * T + gt) * nh + h];                         \
        float gamma = Gamma_in[(b * T + gt) * nh + h];                         \
        float dqk = dQK_dot[(b * T + gt) * nh + h];                            \
                                                                               \
        /* Load Q_raw + K_raw (pre-RoPE, post-bias) from the staged tiles */   \
        float q_pre[MAMBA_RS_STATE_CAP], k_pre[MAMBA_RS_STATE_CAP]; /* sized by the state capacity */ \
        for (int n = 0; n < ds; n++) {                                         \
            q_pre[n] = use_staging ? q_sm[t_local * ds + n] : Q_raw[base + n]; \
            k_pre[n] = use_staging ? k_sm[t_local * ds + n] : K_raw[base + n]; \
        }                                                                      \
        float dq_in[MAMBA_RS_STATE_CAP], dk_in[MAMBA_RS_STATE_CAP];            \
        for (int n = 0; n < ds; n++) {                                         \
            dq_in[n] = use_staging ? dqi_sm[t_local * ds + n] : dQ_mid[base + n]; \
            dk_in[n] = use_staging ? dki_sm[t_local * ds + n] : dK_mid[base + n]; \
        }                                                                      \
                                                                               \
        /* Forward RoPE on K_raw to get K_rot (for dScale computation) */      \
        float k_rot[MAMBA_RS_STATE_CAP];                                       \
        IDX angle_base = ((b * T + gt) * nh + h) * n_angles;                   \
        /* Each angle's cos/sin was computed three times */                    \
        /* (forward RoPE, inverse RoPE, dtheta) — hoist into registers. */     \
        /* cosf/sinf of the same input is deterministic, so every read */      \
        /* stays bit-identical to the inline forms. */                         \
        float cos_a[MAMBA_RS_STATE_CAP / 2], sin_a[MAMBA_RS_STATE_CAP / 2];    \
        for (int a = 0; a < n_angles && 2 * a + 1 < ds; a++) {                 \
            float theta = Angles[angle_base + a];                              \
            cos_a[a] = cosf(theta);                                            \
            sin_a[a] = sinf(theta);                                            \
        }                                                                      \
        for (int a = 0; a < n_angles && 2 * a + 1 < ds; a++) {                 \
            float cos_t = cos_a[a];                                            \
            float sin_t = sin_a[a];                                            \
            int i0 = 2 * a, i1 = 2 * a + 1;                                    \
            k_rot[i0] = k_pre[i0] * cos_t - k_pre[i1] * sin_t;                 \
            k_rot[i1] = k_pre[i0] * sin_t + k_pre[i1] * cos_t;                 \
        }                                                                      \
        for (int n = 2 * n_angles; n < ds; n++) k_rot[n] = k_pre[n];           \
                                                                               \
        /* dScale = sum_n(dK_mid[n] * K_rot[n]) */                             \
        float d_scale = 0.0f;                                                  \
        for (int n = 0; n < ds; n++) d_scale += dk_in[n] * k_rot[n];           \
        dScale[(b * T + gt) * nh + h] = d_scale;                               \
                                                                               \
        /* dGamma = dQK * sum_n(Q_raw[n] * K_raw[n]) */                        \
        float qk_raw = 0.0f;                                                   \
        for (int n = 0; n < ds; n++) qk_raw += q_pre[n] * k_pre[n];            \
        dGamma[(b * T + gt) * nh + h] = dqk * qk_raw;                          \
                                                                               \
        /* Scale dK_mid by scale before inverse RoPE */                        \
        for (int n = 0; n < ds; n++) dk_in[n] *= scale;                        \
                                                                               \
        /* Inverse RoPE on dQ_mid and scaled dK_mid */                         \
        float dq_pre_out[MAMBA_RS_STATE_CAP], dk_pre_out[MAMBA_RS_STATE_CAP];  \
        for (int n = 0; n < ds; n++) { dq_pre_out[n] = dq_in[n]; dk_pre_out[n] = dk_in[n]; } \
        for (int a = 0; a < n_angles && 2 * a + 1 < ds; a++) {                 \
            float cos_t = cos_a[a];                                            \
            float sin_t = sin_a[a];                                            \
            int i0 = 2 * a, i1 = 2 * a + 1;                                    \
            /* Inverse rotation: R^T = [[cos, sin], [-sin, cos]] */            \
            dq_pre_out[i0] = dq_in[i0] * cos_t + dq_in[i1] * sin_t;            \
            dq_pre_out[i1] = -dq_in[i0] * sin_t + dq_in[i1] * cos_t;           \
            dk_pre_out[i0] = dk_in[i0] * cos_t + dk_in[i1] * sin_t;            \
            dk_pre_out[i1] = -dk_in[i0] * sin_t + dk_in[i1] * cos_t;           \
        }                                                                      \
                                                                               \
        /* Add dQK path: dQ_pre += dqk * gamma * K_raw, dK_pre += dqk * gamma * Q_raw */ \
        float dqk_gamma = dqk * gamma;                                         \
        for (int n = 0; n < ds; n++) {                                         \
            dq_pre_out[n] += dqk_gamma * k_pre[n];                             \
            dk_pre_out[n] += dqk_gamma * q_pre[n];                             \
        }                                                                      \
                                                                               \
        /* Stage dQ_pre, dK_pre rows; the cooperative store below writes */    \
        /* them coalesced. (Caller reduces to dQ_bias/dK_bias via */           \
        /* colsum_accumulate — no atomicAdd here.) */                          \
        if (use_staging) {                                                     \
            for (int n = 0; n < ds; n++) {                                     \
                out_q_sm[t_local * ds + n] = dq_pre_out[n];                    \
                out_k_sm[t_local * ds + n] = dk_pre_out[n];                    \
            }                                                                  \
        } else {                                                               \
            for (int n = 0; n < ds; n++) {                                     \
                dQ_pre[base + n] = dq_pre_out[n];                              \
                dK_pre[base + n] = dk_pre_out[n];                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        /* dAngles_cumsum from rotary gradient */                              \
        /* dtheta = dQ_in * d(Q_rot)/d(theta) + dK_in_scaled * d(K_rot)/d(theta) */ \
        for (int a = 0; a < n_angles && 2 * a + 1 < ds; a++) {                 \
            float cos_t = cos_a[a];                                            \
            float sin_t = sin_a[a];                                            \
            int i0 = 2 * a, i1 = 2 * a + 1;                                    \
            /* d(Q_rot)/dtheta: Q_rot[i0] = Q[i0]*cos - Q[i1]*sin */           \
            /* Q_rot[i1] = Q[i0]*sin + Q[i1]*cos */                            \
            /* d/dtheta: [-Q[i0]*sin - Q[i1]*cos, Q[i0]*cos - Q[i1]*sin] */    \
            float dtheta_q = dq_in[i0] * (-q_pre[i0] * sin_t - q_pre[i1] * cos_t) \
                           + dq_in[i1] * (q_pre[i0] * cos_t - q_pre[i1] * sin_t); \
            float dtheta_k = dk_in[i0] * (-k_pre[i0] * sin_t - k_pre[i1] * cos_t) \
                           + dk_in[i1] * (k_pre[i0] * cos_t - k_pre[i1] * sin_t); \
            dAngles_cumsum[((b * T + gt) * nh + h) * n_angles + a] = dtheta_q + dtheta_k; \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    if (use_staging) {                                                         \
        for (int i = t_local; i < rows * ds; i += CS) {                        \
            int tt = i / ds;                                                   \
            int nn = i % ds;                                                   \
            IDX gbase = ((b * T + t0 + tt) * nh + h) * ds + nn;                \
            dQ_pre[gbase] = out_q_sm[i];                                       \
            dK_pre[gbase] = out_k_sm[i];                                       \
        }                                                                      \
    }                                                                          \
    /* dQ_bias/dK_bias produced via colsum_accumulate on */                    \
    /* dQ_pre/dK_pre by the caller. No atomicAdd here. */                      \
}

DEFINE_M3_DQKTHETA_F32(m3_reference_dqktheta, int)

// __launch_bounds__: block_dim=CS <= 64. Six state-capacity register arrays per
// thread risk spilling under nvcc heuristics — pin to 4 blocks/SM.
// The output tiles become live only after the input tiles are consumed.
// Reusing that storage and permuting its addresses leaves the reductions
// unchanged while reducing shared-memory traffic and bank conflicts.
#define M3_DQKT_Q_TILE_legacy dki_sm + CS * ds
#define M3_DQKT_Q_TILE_transport q_sm
#define M3_DQKT_K_TILE_legacy out_q_sm + CS * ds
#define M3_DQKT_K_TILE_transport k_sm
#define M3_DQKT_COOP_INDEX_legacy(i, t, n, ds) i
#define M3_DQKT_COOP_INDEX_transport(i, t, n, ds) t * ds + (n ^ ((ds == 16) ? ((t >> 1) & 15) : 0))
#define M3_DQKT_THREAD_INDEX_legacy(t, n, ds) t * ds + n
#define M3_DQKT_THREAD_INDEX_transport(t, n, ds) t * ds + (n ^ ((ds == 16) ? ((t >> 1) & 15) : 0))

#define DEFINE_M3_DQKTHETA(NAME, IDX, T_ACT, FROM_F, MODE) \
extern "C" __global__ __launch_bounds__(64, 4) void \
NAME( \
float* __restrict__ dQ_pre, \
float* __restrict__ dK_pre, \
float* __restrict__ dAngles_cumsum, \
float* dScale, \
float* dGamma, \
const T_ACT* __restrict__ Q_raw, \
const T_ACT* __restrict__ K_raw, \
const float* Scale_in, \
const float* Gamma_in, \
const float* __restrict__ Angles, \
const float* __restrict__ dQ_mid, \
const float* __restrict__ dK_mid, \
const float* __restrict__ dQK_dot, \
int B, int T, int nh, int ds, int n_angles, int CS, \
int use_staging, \
    int y_first /* this run's first y block */ \
) { \
    int n_chunks = (T + CS - 1) / CS; \
    IDX bc = blockIdx.x; \
    IDX b_chunk = bc; \
    int h = y_first + blockIdx.y; \
    int t_local = threadIdx.x; \
    IDX b = b_chunk / n_chunks; \
    int chunk = (int)(b_chunk % n_chunks); \
    int gt = chunk * CS + t_local; \
    if (ds > MAMBA_RS_STATE_CAP) return; \
    if (b >= B || h >= nh) return; \
    bool valid = (gt < T); \
    extern __shared__ float dqkt_sm[]; \
    float* q_sm     = dqkt_sm; \
    float* k_sm     = q_sm + CS * ds; \
    float* dqi_sm   = k_sm + CS * ds; \
    float* dki_sm   = dqi_sm + CS * ds; \
    float* out_q_sm = M3_DQKT_Q_TILE_##MODE; \
    float* out_k_sm = M3_DQKT_K_TILE_##MODE; \
    int t0 = chunk * CS; \
    int rows = min(CS, T - t0); \
    if (use_staging) { \
        for (int i = t_local; i < rows * ds; i += CS) { \
            int tt = i / ds; \
            int nn = i % ds; \
            IDX gbase = ((b * T + t0 + tt) * nh + h) * ds + nn; \
            q_sm[M3_DQKT_COOP_INDEX_##MODE(i, tt, nn, ds)] = to_f(Q_raw[gbase]); \
            k_sm[M3_DQKT_COOP_INDEX_##MODE(i, tt, nn, ds)] = to_f(K_raw[gbase]); \
            dqi_sm[M3_DQKT_COOP_INDEX_##MODE(i, tt, nn, ds)] = dQ_mid[gbase]; \
            dki_sm[M3_DQKT_COOP_INDEX_##MODE(i, tt, nn, ds)] = dK_mid[gbase]; \
        } \
    } \
    __syncthreads(); \
    if (valid) { \
        IDX base = ((b * T + gt) * nh + h) * ds; \
        float scale = Scale_in[(b * T + gt) * nh + h]; \
        float gamma = Gamma_in[(b * T + gt) * nh + h]; \
        float dqk = dQK_dot[(b * T + gt) * nh + h]; \
        float q_pre[MAMBA_RS_STATE_CAP], k_pre[MAMBA_RS_STATE_CAP]; \
        for (int n = 0; n < ds; n++) { \
            q_pre[n] = \
            use_staging ? q_sm[M3_DQKT_THREAD_INDEX_##MODE(t_local, n, ds)] : to_f(Q_raw[base + n]); \
            k_pre[n] = \
            use_staging ? k_sm[M3_DQKT_THREAD_INDEX_##MODE(t_local, n, ds)] : to_f(K_raw[base + n]); \
        } \
        float dq_in[MAMBA_RS_STATE_CAP], dk_in[MAMBA_RS_STATE_CAP]; \
        for (int n = 0; n < ds; n++) { \
            dq_in[n] = \
            use_staging ? dqi_sm[M3_DQKT_THREAD_INDEX_##MODE(t_local, n, ds)] : dQ_mid[base + n]; \
            dk_in[n] = \
            use_staging ? dki_sm[M3_DQKT_THREAD_INDEX_##MODE(t_local, n, ds)] : dK_mid[base + n]; \
        } \
        float k_rot[MAMBA_RS_STATE_CAP]; \
        IDX angle_base = ((b * T + gt) * nh + h) * n_angles; \
        float cos_a[MAMBA_RS_STATE_CAP / 2], sin_a[MAMBA_RS_STATE_CAP / 2]; \
        for (int a = 0; a < n_angles && 2 * a + 1 < ds; a++) { \
            float theta = Angles[angle_base + a]; \
            cos_a[a] = cosf(theta); \
            sin_a[a] = sinf(theta); \
        } \
        for (int a = 0; a < n_angles && 2 * a + 1 < ds; a++) { \
            float cos_t = cos_a[a]; \
            float sin_t = sin_a[a]; \
            int i0 = 2 * a, i1 = 2 * a + 1; \
            k_rot[i0] = k_pre[i0] * cos_t - k_pre[i1] * sin_t; \
            k_rot[i1] = k_pre[i0] * sin_t + k_pre[i1] * cos_t; \
        } \
        for (int n = 2 * n_angles; n < ds; n++) k_rot[n] = k_pre[n]; \
        float d_scale = 0.0f; \
        for (int n = 0; n < ds; n++) d_scale += dk_in[n] * k_rot[n]; \
        dScale[(b * T + gt) * nh + h] = d_scale; \
        float qk_raw = 0.0f; \
        for (int n = 0; n < ds; n++) qk_raw += q_pre[n] * k_pre[n]; \
        dGamma[(b * T + gt) * nh + h] = dqk * qk_raw; \
        for (int n = 0; n < ds; n++) dk_in[n] *= scale; \
        float dq_pre_out[MAMBA_RS_STATE_CAP], dk_pre_out[MAMBA_RS_STATE_CAP]; \
        for (int n = 0; n < ds; n++) { \
            dq_pre_out[n] = dq_in[n]; \
            dk_pre_out[n] = dk_in[n]; \
        } \
        for (int a = 0; a < n_angles && 2 * a + 1 < ds; a++) { \
            float cos_t = cos_a[a]; \
            float sin_t = sin_a[a]; \
            int i0 = 2 * a, i1 = 2 * a + 1; \
            dq_pre_out[i0] = dq_in[i0] * cos_t + dq_in[i1] * sin_t; \
            dq_pre_out[i1] = -dq_in[i0] * sin_t + dq_in[i1] * cos_t; \
            dk_pre_out[i0] = dk_in[i0] * cos_t + dk_in[i1] * sin_t; \
            dk_pre_out[i1] = -dk_in[i0] * sin_t + dk_in[i1] * cos_t; \
        } \
        float dqk_gamma = dqk * gamma; \
        for (int n = 0; n < ds; n++) { \
            dq_pre_out[n] += dqk_gamma * k_pre[n]; \
            dk_pre_out[n] += dqk_gamma * q_pre[n]; \
        } \
        if (use_staging) { \
            for (int n = 0; n < ds; n++) { \
                out_q_sm[M3_DQKT_THREAD_INDEX_##MODE(t_local, n, ds)] = dq_pre_out[n]; \
                out_k_sm[M3_DQKT_THREAD_INDEX_##MODE(t_local, n, ds)] = dk_pre_out[n]; \
            } \
        } else { \
            for (int n = 0; n < ds; n++) { \
                dQ_pre[base + n] = dq_pre_out[n]; \
                dK_pre[base + n] = dk_pre_out[n]; \
            } \
        } \
        for (int a = 0; a < n_angles && 2 * a + 1 < ds; a++) { \
            float cos_t = cos_a[a]; \
            float sin_t = sin_a[a]; \
            int i0 = 2 * a, i1 = 2 * a + 1; \
            float dtheta_q = dq_in[i0] \
            * (-q_pre[i0] * sin_t - q_pre[i1] * cos_t) \
            + dq_in[i1] * (q_pre[i0] * cos_t - q_pre[i1] * sin_t); \
            float dtheta_k = dk_in[i0] \
            * (-k_pre[i0] * sin_t - k_pre[i1] * cos_t) \
            + dk_in[i1] * (k_pre[i0] * cos_t - k_pre[i1] * sin_t); \
            dAngles_cumsum[((b * T + gt) * nh + h) * n_angles + a] = \
            dtheta_q + dtheta_k; \
        } \
    } \
    __syncthreads(); \
    if (use_staging) { \
        for (int i = t_local; i < rows * ds; i += CS) { \
            int tt = i / ds; \
            int nn = i % ds; \
            IDX gbase = ((b * T + t0 + tt) * nh + h) * ds + nn; \
            dQ_pre[gbase] = out_q_sm[M3_DQKT_COOP_INDEX_##MODE(i, tt, nn, ds)]; \
            dK_pre[gbase] = out_k_sm[M3_DQKT_COOP_INDEX_##MODE(i, tt, nn, ds)]; \
        } \
    } \
    (void)FROM_F; \
}


DEFINE_M3_DQKTHETA(m3_reference_dqktheta_bf16, int, __nv_bfloat16, from_f_bf16, legacy)
DEFINE_M3_DQKTHETA(m3_reference_dqktheta_f16, int, __half, from_f_f16, legacy)
DEFINE_M3_DQKTHETA(m3_reference_dqktheta_transport_bf16, int, __nv_bfloat16, from_f_bf16, transport)
DEFINE_M3_DQKTHETA(m3_reference_dqktheta_transport_f16, int, __half, from_f_f16, transport)

#undef M3_DQKT_Q_TILE_legacy
#undef M3_DQKT_Q_TILE_transport
#undef M3_DQKT_K_TILE_legacy
#undef M3_DQKT_K_TILE_transport
#undef M3_DQKT_COOP_INDEX_legacy
#undef M3_DQKT_COOP_INDEX_transport
#undef M3_DQKT_THREAD_INDEX_legacy
#undef M3_DQKT_THREAD_INDEX_transport
