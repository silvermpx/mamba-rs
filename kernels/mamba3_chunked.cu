// Mamba-3 SISO chunked parallel scan CUDA kernels.
//
// Replaces sequential T-loop with parallel intra-chunk matmul for Mamba-3
// trapezoidal recurrence. Key difference from the Mamba-2 chunked form:
// trapezoidal discretization with shifted_gamma/scale K-prescaling.
//
// Chunked forward reformulation:
//   shifted_gamma[t] = DT[t+1] * (1 - sigmoid(trap[t+1]))   // beta of NEXT step
//   gamma[t] = DT[t] * sigmoid(trap[t])                       // gamma of CURRENT step
//   scale[t] = shifted_gamma[t] + gamma[t]                    // combined weight on K[t]
//   K_scaled[t] = K[t] * scale[t]
//   qk_dot[t] = sum(Q[t] * K[t]) * gamma[t]                  // for D skip
//
// Then: Y = Q @ K_scaled^T * causal_decay @ V + state_contrib + (D + qk_dot) * V
//
// Paper: Lahoti et al., "Mamba-3", ICLR 2026 (arXiv 2603.15569)
//
// Thread/grid conventions follow mamba3_siso.cu.
// All kernels handle partial last chunk (T not multiple of chunk_size).

// State-dimension capacity of the per-thread register arrays below.
// Injected at JIT time (-DMAMBA_RS_STATE_CAP=...) from the model config
// so any reference-range d_state runs the same code path; 64 covers the
// common shapes at minimum register pressure. Past ~128 the compiler
// spills these arrays to local memory - correct, measurably slower,
// and accepted: capacity is a first-class knob, not a fallback.
#ifndef MAMBA_RS_STATE_CAP
#define MAMBA_RS_STATE_CAP 64
#endif


#include "_typed_prelude.cuh"

#ifndef FAST_EXP
#define FAST_EXP(x) exp2f((x) * LOG2E)
#endif

// A decay exponent capped at zero, the reference's guard against
// float-noise positives (a decay above one). A NaN exponent stays NaN:
// fminf would return the zero and turn a corrupt step into a decay of
// exactly one, hiding it. Finite values get fminf's bits.
__device__ __forceinline__ float m3_nonpos(float x) {
    return x > 0.0f ? 0.0f : x;
}


// The reference A activation and its derivative, for the final gradients
// (see kernels/mamba3_ops.cu, kept in sync).
__device__ __forceinline__ float m3c_heavy_tail(float x) {
    return (x >= 0.0f) ? (1.0f + x) : (1.0f / (1.0f - x));
}
__device__ __forceinline__ float m3c_heavy_tail_deriv(float x) {
    if (x >= 0.0f) return 1.0f;
    float d = 1.0f - x;
    return 1.0f / (d * d);
}




// ============================================================================
// 3. m3_chunk_pre_state_fused -- the pre-state in one kernel, whole rows
// ============================================================================
//
// For each step t of a chunk, as the reference prepares the scan:
//   gamma[t] = DT[t] * trap_sig[t]
//   shifted_gamma[t] = DT[t+1] * (1 - trap_sig[t+1])    (0 at a row's end)
//   scale[t] = shifted_gamma[t] + gamma[t]
//   qk_dot[t] = sum_n(Q[t,h,n] * K[t,h,n]) * gamma[t]
//   K_scaled[t,h,n] = K[t,h,n] * scale[t]
// then the chunk's decay cumsum over the a*dt rows and its state
//   chunk_states[h,p,n] = sum_t(exp(dA_end - dA_t) * K_scaled[t,h,n] * V[t,h,p]).
// Each K_scaled row is computed once, stored to global (the scan consumes
// it) and kept in shared memory for the state fold, with x and each row's
// decay to the chunk's end staged alongside.
//
// Bit contract: phase A is one t per thread; phase B folds four
// ascending-t chains per thread from shared memory. The sliced pre-state
// (3c) keeps those chains for shapes whose rows this tile cannot hold, and
// both are held bit for bit to the frozen pre-state trio in
// tests/fixtures/m3_chunked_forward_reference.cu.
//
// Grid: (B * n_chunks, nh). Block: chunk_size (= 64 at the serve shape).
// Dynamic smem floats: chunk_size * (ds + 4) (K_scaled) + chunk_size * hd
// (x) + chunk_size (the decay) + chunk_size (the decay cumsum).
#define DEFINE_M3_CHUNK_PRE_STATE_FUSED_F32(NAME, IDX)                         \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ K_scaled,                                              \
    float* __restrict__ qk_dot,                                                \
    float* __restrict__ scale_out,                                             \
    float* __restrict__ gamma_out,                                             \
    float* __restrict__ states_out,                                            \
    const float* __restrict__ K,                                               \
    const float* __restrict__ Q,                                               \
    const float* __restrict__ DT,                                              \
    const float* __restrict__ trap_sig,                                        \
    const float* __restrict__ x,                                               \
    const float* __restrict__ adt,       /* [B * T * nh] a*dt */               \
    float* __restrict__ dA_cumsum,       /* [B * n_chunks * nh * chunk_size] out */ \
    int batch, int T, int nh, int hd, int ds, int chunk_size,                  \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    int d_inner = nh * hd;                                                     \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                          \
    IDX bc = blockIdx.x;                                                       \
    IDX b = bc / n_chunks;                                                     \
    int chunk = (int)(bc % n_chunks);                                          \
    int h = y_first + blockIdx.y;                                              \
    if (ds > MAMBA_RS_STATE_CAP) return;                                       \
                                                                               \
    int chunk_start = chunk * chunk_size;                                      \
    int chunk_end = chunk_start + chunk_size;                                  \
    if (chunk_end > T) chunk_end = T;                                          \
    int chunk_len = chunk_end - chunk_start;                                   \
    /* In a packed row (seq_idx) only the chunk's last sequence reaches */     \
    /* its state: the tokens from where that sequence begins. */               \
    int kept = vl_seq_first(seq_idx, b, T, chunk_start, chunk_end - 1) - chunk_start; \
                                                                               \
    extern __shared__ float m3f_sm[];                                          \
    int k_ld = ds + 4; /* padded rows - see the coop kernel's bank note */     \
    float* sm_k = m3f_sm;                       /* [chunk_size][k_ld] */       \
    float* sm_x = sm_k + chunk_size * k_ld;     /* [chunk_size][hd] */         \
    float* sm_dec = sm_x + chunk_size * hd;     /* [chunk_size] */             \
    float* da_chunk = sm_dec + chunk_size;      /* [chunk_size] */             \
                                                                               \
    int tid = threadIdx.x;                                                     \
    int nt = blockDim.x;                                                       \
    /* The chunk's decay cumsum, adds from zero in row order over the */      \
    /* a*dt rows, written for the state pass, the scan and the backward: */    \
    /* one thread adds, as every add waits on the one before it. */            \
    for (int t_local = tid; t_local < chunk_len; t_local += nt) {              \
        da_chunk[t_local] = adt[(b * T + chunk_start + t_local) * nh + h];     \
    }                                                                          \
    __syncthreads();                                                           \
    if (tid == 0) {                                                            \
        float cumsum = 0.0f;                                                   \
        IDX cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;          \
        for (int t_local = 0; t_local < chunk_len; t_local++) {                \
            const bool fresh = vl_starts(seq_idx, b, T, chunk_start + t_local); \
            cumsum = (fresh ? 0.0f : cumsum) + da_chunk[t_local];              \
            da_chunk[t_local] = cumsum;                                        \
            dA_cumsum[cs_base + t_local] = cumsum;                             \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    /* Each row's decay to the chunk's end is staged once with its row: */     \
    /* every fold thread reads the same chunk_len values, and computing */     \
    /* them in the fold repeated one exponential per thread and step. */       \
    /* Same inputs, same expression - the values do not move. */               \
    float dA_end = da_chunk[chunk_len - 1];                                    \
                                                                               \
    /* Phase A: the preprocess arithmetic, one t per thread (verbatim), */     \
    /* with the K_scaled row written to global AND smem; x and the decay */    \
    /* staged. */                                                              \
    for (int t_local = tid; t_local < chunk_len; t_local += nt) {              \
        int t = chunk_start + t_local;                                         \
        IDX th = (b * T + t) * nh + h;                                         \
        float dt_cur = DT[th];                                                 \
        float trap_cur = trap_sig[th];                                         \
        float gamma_val = dt_cur * trap_cur;                                   \
        float shifted_gamma = 0.0f;                                            \
        /* The next step's beta is 0 where it starts a sequence. */            \
        if (t + 1 < T && !vl_starts(seq_idx, b, T, t + 1)) {                   \
            IDX th_next = (b * T + t + 1) * nh + h;                            \
            float dt_next = DT[th_next];                                       \
            float trap_next = trap_sig[th_next];                               \
            shifted_gamma = dt_next * (1.0f - trap_next);                      \
        }                                                                      \
        float scale_val = shifted_gamma + gamma_val;                           \
        if (scale_out) scale_out[th] = scale_val;                              \
        if (gamma_out) gamma_out[th] = gamma_val;                              \
        IDX kq_base = (b * T + t) * nh * ds + h * ds;                          \
        float dot = 0.0f;                                                      \
        if (ds % 4 == 0) {                                                     \
            const float4* q4 = reinterpret_cast<const float4*>(Q + kq_base);   \
            const float4* k4 = reinterpret_cast<const float4*>(K + kq_base);   \
            float4* ks4 = reinterpret_cast<float4*>(K_scaled + kq_base);       \
            float4* sk4 = reinterpret_cast<float4*>(sm_k + t_local * k_ld);    \
            for (int n4 = 0; n4 < ds / 4; n4++) {                              \
                float4 qv = q4[n4];                                            \
                float4 kv = k4[n4];                                            \
                dot += qv.x * kv.x;                                            \
                dot += qv.y * kv.y;                                            \
                dot += qv.z * kv.z;                                            \
                dot += qv.w * kv.w;                                            \
                float4 sc;                                                     \
                sc.x = kv.x * scale_val;                                       \
                sc.y = kv.y * scale_val;                                       \
                sc.z = kv.z * scale_val;                                       \
                sc.w = kv.w * scale_val;                                       \
                ks4[n4] = sc;                                                  \
                sk4[n4] = sc;                                                  \
            }                                                                  \
        } else {                                                               \
            for (int n = 0; n < ds; n++) {                                     \
                dot += Q[kq_base + n] * K[kq_base + n];                        \
            }                                                                  \
            for (int n = 0; n < ds; n++) {                                     \
                float ks = K[kq_base + n] * scale_val;                         \
                K_scaled[kq_base + n] = ks;                                    \
                sm_k[t_local * k_ld + n] = ks;                                 \
            }                                                                  \
        }                                                                      \
        qk_dot[th] = dot * gamma_val;                                          \
        sm_dec[t_local] = FAST_EXP(m3_nonpos(dA_end - da_chunk[t_local]));     \
    }                                                                          \
    if (hd % 4 == 0 && (reinterpret_cast<unsigned long long>(x) & 15) == 0) {  \
        /* Whole float4s when every row starts on 16 bytes: a quarter of */    \
        /* the loads and index math, the same copied values. */                \
        int hq = hd >> 2;                                                      \
        float4* sm_x4 = reinterpret_cast<float4*>(sm_x);                       \
        for (int i = tid; i < chunk_len * hq; i += nt) {                       \
            int tl = i / hq;                                                   \
            int pq = i - tl * hq;                                              \
            sm_x4[i] = __ldg(reinterpret_cast<const float4*>(                  \
                x + (long long)(b * T + chunk_start + tl) * d_inner + h * hd + 4 * pq)); \
        }                                                                      \
    } else {                                                                   \
        for (int i = tid; i < chunk_len * hd; i += nt) {                       \
            int tl = i / hd;                                                   \
            int pp = i % hd;                                                   \
            sm_x[tl * hd + pp] = __ldg(x + (long long)(b * T + chunk_start + tl) * d_inner + h * hd + pp); \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
                                                                               \
    /* Phase B: the chunk_state quad fold (verbatim chains) over smem. */      \
    /* Threads reshape to (p, n-quad); requires blockDim.x >= hd * ds/4 */     \
    /* (the launcher guard). Each acc chain folds ascending t, exactly */      \
    /* the standalone kernel's arithmetic. */                                  \
    int quads = ds / 4;                                                        \
    for (int o = tid; o < hd * quads; o += nt) {                               \
        int p = o / quads;                                                     \
        int nq = o % quads;                                                    \
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;              \
        /* Four steps' reads together ahead of their arithmetic; each chain */ \
        /* takes the same products in the same order as one step at a time. */ \
        int tr = kept;                                                         \
        for (; tr + 4 <= chunk_len; tr += 4) {                                 \
            float dec4[4], vt4[4];                                             \
            float4 kq4[4];                                                     \
            lds_f32_row4(sm_dec + tr, dec4);                                   \
            lds_f32_rows4(sm_x + tr * hd + p, hd, vt4);                        \
            lds_f32x4_rows4(sm_k + tr * k_ld + 4 * nq, k_ld, kq4);             \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 4; j++) {                                      \
                acc0 += dec4[j] * kq4[j].x * vt4[j];                           \
                acc1 += dec4[j] * kq4[j].y * vt4[j];                           \
                acc2 += dec4[j] * kq4[j].z * vt4[j];                           \
                acc3 += dec4[j] * kq4[j].w * vt4[j];                           \
            }                                                                  \
        }                                                                      \
        for (int t_local = tr; t_local < chunk_len; t_local++) {             \
            float decay = sm_dec[t_local];                                     \
            float v_t = sm_x[t_local * hd + p];                                \
            const float* k_row = sm_k + t_local * k_ld;                        \
            float4 kv = lds_f32x4(k_row + 4 * nq);                             \
            acc0 += decay * kv.x * v_t;                                        \
            acc1 += decay * kv.y * v_t;                                        \
            acc2 += decay * kv.z * v_t;                                        \
            acc3 += decay * kv.w * v_t;                                        \
        }                                                                      \
        IDX state_base = ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds + nq * 4; \
        float4 out;                                                            \
        out.x = acc0;                                                          \
        out.y = acc1;                                                          \
        out.z = acc2;                                                          \
        out.w = acc3;                                                          \
        *reinterpret_cast<float4*>(states_out + state_base) = out;             \
    }                                                                          \
}

DEFINE_M3_CHUNK_PRE_STATE_FUSED_F32(m3_chunk_pre_state_fused, int)
DEFINE_M3_CHUNK_PRE_STATE_FUSED_F32(m3_chunk_pre_state_fused_i64, long long)

// ============================================================================
// 4. m3_state_pass_fused -- the inter-chunk state pass of a window
// ============================================================================
//
// One CTA per (b, h), its threads over the hd*ds state elements. Each
// element seeds its state, carries it across the chunks, and stores the
// window's final state, then the block writes the K and V boundary states
// from the window's last step.
//
// The seed: a window that CONTINUES a previous one (a carried prefill)
// enters from the persistent SSM state, with the trapezoidal boundary fold
// applied: the trapezoidal discretization's beta term reaches one step BACK
// across the window seam, so carrying the SSM state alone is wrong at
// exactly the first position of the new window. Following the reference
// (mamba3_siso_fwd.py, HAS_INITIAL_STATES):
//
//   h_enter[b,h,p,n] += v_state[b,h,p] * k_state[b,h,n]
//                       * dt[b,0,h] * (1 - trap[b,0,h])
//
// dt is post-softplus and trap post-sigmoid (both saved that way by the
// split and activation kernels, so no extra sigmoid here, unlike the
// raw-input reference). Training seeds every window this way too, so a
// sequence cut into windows trains as it runs; the backward takes the
// fold's gradient through m3_enter_fold_bwd. A null seed enters at zero,
// the stateless window of the pooled prefill.
//
// The pass: an exclusive prefix scan over the chunks that turns states in
// place from chunk contributions into the state ENTERING each chunk,
//   new_state = exp(dA_chunk_end) * prev_state + chunk_contribution,
// with a partial last chunk when T is not a multiple of the chunk size.
//
// The arithmetic is that of the three launches this kernel replaced (the
// entering fold, the pass and the writeback), in the same order, with the
// folded seed and the final state held in registers instead of round trips
// through global memory; tests/fixtures/m3_state_pass_reference.cu keeps
// them frozen, and the fused pass is held to them bit for bit.
//
// The seed and the persistent state may be one buffer (a carried prefill
// passes the layer's own SSM, K and V states as both): an element is read
// before its own thread writes it, and the K and V boundary states, which
// every element's fold reads, are rewritten only after a block barrier.
// That barrier is why one CTA holds a head's whole state.
//
// Grid: (B, nh). Block: hd*ds rounded up to whole warps, at most 1024; an
// element loop covers larger states.
#define DEFINE_M3_STATE_PASS_FUSED(NAME, IDX, T_ACT)                          \
extern "C" __global__ void                                                    \
NAME(                                                                         \
    float* __restrict__ states,          /* [B*nc*nh*hd*ds] in/out */         \
    float* ssm_state,                    /* [B*nh*hd*ds] out */               \
    float* k_state,                      /* [B*nh*ds] out */                  \
    float* v_state,                      /* [B*nh*hd] out */                  \
    const float* ssm_seed,               /* [B*nh*hd*ds] or nullptr */        \
    const float* k_seed,                 /* [B*nh*ds] */                      \
    const float* v_seed,                 /* [B*nh*hd] */                      \
    const float* __restrict__ dt,        /* [B*T*nh] post-softplus */         \
    const float* __restrict__ trap,      /* [B*T*nh] post-sigmoid */          \
    const float* __restrict__ dA_cumsum, /* [B*nc*nh*chunk_size] */           \
    const T_ACT* __restrict__ k_flat,    /* [B*T*nh*ds] */                    \
    const T_ACT* __restrict__ x_flat,    /* [B*T*d_inner] */                  \
    int batch, int n_chunks, int nh, int hd, int ds, int chunk_size, int T,   \
    int y_first, /* this run's first y block */                               \
    const int* __restrict__ seq_idx                                           \
) {                                                                           \
    IDX b = blockIdx.x;                                                       \
    int h = y_first + blockIdx.y;                                             \
    /* Uniform across the block, so no thread skips the barrier below. */     \
    if (b >= batch || h >= nh) return;                                        \
    int dim = hd * ds;                                                        \
    for (int pd = threadIdx.x; pd < dim; pd += blockDim.x) {                  \
        IDX hi = (b * nh + h) * dim + pd;                                     \
        float state = 0.0f;                                                   \
        if (ssm_seed != nullptr) {                                            \
            int p = pd / ds;                                                  \
            int n = pd % ds;                                                  \
            float dt0 = dt[b * T * nh + h];                                   \
            float trap0 = trap[b * T * nh + h];                               \
            float v = v_seed[(b * nh + h) * hd + p];                          \
            float k = k_seed[(b * nh + h) * ds + n];                          \
            state = ssm_seed[hi];                                             \
            state += v * k * dt0 * (1.0f - trap0);                            \
        }                                                                     \
        /* The pass: the next chunk's loads go out ahead of the */            \
        /* dependent chain, the arithmetic unchanged. */                      \
        /* states is read with plain loads: this kernel also writes it, */   \
        /* and the read-only cache path would let the compiler move a */      \
        /* load past the store that overwrites the same element. */           \
        IDX state_idx = b * n_chunks * nh * dim + h * dim + pd;               \
        IDX cs_idx = b * n_chunks * nh + h;                                   \
        float contrib = states[state_idx];                                    \
        int chunk_end_0 = chunk_size > T ? T : chunk_size;                    \
        float decay =                                                         \
            FAST_EXP(__ldg(&dA_cumsum[cs_idx * chunk_size + chunk_end_0 - 1])); \
        for (int c = 0; c < n_chunks; c++) {                                  \
            float next_contrib = 0.0f;                                        \
            float next_decay = 0.0f;                                          \
            if (c + 1 < n_chunks) {                                           \
                next_contrib = states[state_idx + nh * dim];                  \
                int chunk_start_n = (c + 1) * chunk_size;                     \
                int chunk_end_n = chunk_start_n + chunk_size;                 \
                if (chunk_end_n > T) chunk_end_n = T;                         \
                int last_elem_n = chunk_end_n - chunk_start_n - 1;            \
                next_decay = FAST_EXP(                                        \
                    __ldg(&dA_cumsum[(cs_idx + nh) * chunk_size + last_elem_n])); \
            }                                                                 \
            states[state_idx] = state;                                        \
            /* A chunk where a sequence of a packed row starts passes */      \
            /* nothing of the state entering it. */                           \
            int c_end = (c + 1) * chunk_size < T ? (c + 1) * chunk_size : T;  \
            bool reset = vl_id(seq_idx, b, T, c_end - 1)                      \
                         != vl_id(seq_idx, b, T, c * chunk_size - 1);         \
            state = reset ? contrib : decay * state + contrib;                \
            contrib = next_contrib;                                           \
            decay = next_decay;                                               \
            state_idx += nh * dim;                                            \
            cs_idx += nh;                                                     \
        }                                                                     \
        ssm_state[hi] = state;                                                \
    }                                                                         \
    __syncthreads();                                                          \
    int d_inner = nh * hd;                                                    \
    for (int n = threadIdx.x; n < ds; n += blockDim.x) {                      \
        k_state[(b * nh + h) * ds + n] =                                      \
            to_f(k_flat[(b * T + (T - 1)) * nh * ds + h * ds + n]);           \
    }                                                                         \
    for (int p = threadIdx.x; p < hd; p += blockDim.x) {                      \
        v_state[(b * nh + h) * hd + p] =                                      \
            to_f(x_flat[(b * T + (T - 1)) * d_inner + h * hd + p]);           \
    }                                                                         \
}

DEFINE_M3_STATE_PASS_FUSED(m3_state_pass_fused_f32, int, float)
DEFINE_M3_STATE_PASS_FUSED(m3_state_pass_fused_f32_i64, long long, float)
DEFINE_M3_STATE_PASS_FUSED(m3_state_pass_fused_bf16, int, __nv_bfloat16)
DEFINE_M3_STATE_PASS_FUSED(m3_state_pass_fused_bf16_i64, long long, __nv_bfloat16)
DEFINE_M3_STATE_PASS_FUSED(m3_state_pass_fused_f16, int, __half)
DEFINE_M3_STATE_PASS_FUSED(m3_state_pass_fused_f16_i64, long long, __half)


// ============================================================================
// 5. m3_chunk_scan_fwd_coop -- the intra-chunk scan, whole rows staged
// ============================================================================
//
// Each step's output within its chunk:
//   Y[t,h,p] = sum_n(Q[t,h,n] * state[h,p,n]) * exp(dA[t])        // state
//            + sum_{s<t}(exp(dA[t]-dA[s]) * Q[t].K_scaled[s] * V[s]) // pairs
//            + (D[h] + qk_dot[t,h]) * V[t,h,p]                      // skip
// One head per block of 256 threads: the decayed causal tile is
// triangle-packed (only s < t exists), and Q, K, V, dA, qk_dot and the
// entering state are staged through dynamic shared memory once per block.
// Every tile element is produced by exactly one thread with the n-ascending
// dot, and every output y[t,p] is one thread's serial s-ascending chain;
// the sliced scan (5c) keeps those chains for shapes whose rows this tile
// cannot hold.
//
// Grid: (B * n_chunks, nh). Block: 256. Dynamic smem (floats):
//   tri   chunk_size*(chunk_size-1)/2   (strict lower triangle, t-major rows)
//   q     chunk_size * (ds + 4)
//   k     chunk_size * (ds + 4)
//   v     chunk_size * hd
//   ps    ds * (hd + 4)
//   da    chunk_size
//   qkd   chunk_size
// The Rust launcher routes to this kernel only when that total fits the
// block's shared-memory budget.
#define DEFINE_M3_CHUNK_SCAN_FWD_COOP_F32(NAME, IDX)                           \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ y_out,                                                 \
    const float* __restrict__ x,                                               \
    const float* __restrict__ Q,                                               \
    const float* __restrict__ K_scaled,                                        \
    const float* __restrict__ qk_dot,                                          \
    const float* __restrict__ dA_cumsum,                                       \
    const float* __restrict__ prev_states,                                     \
    const float* __restrict__ D,                                               \
    int batch, int T, int nh, int hd, int ds, int chunk_size,                  \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    int d_inner = nh * hd;                                                     \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                          \
    IDX bc = blockIdx.x;                                                       \
    IDX b = bc / n_chunks;                                                     \
    int chunk = (int)(bc % n_chunks);                                          \
    int h = y_first + blockIdx.y;                                              \
    if (chunk_size > 64) return;                                               \
    if (ds > MAMBA_RS_STATE_CAP) return;                                       \
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
                                                                               \
    IDX cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;              \
    IDX state_head = ((b * n_chunks + chunk) * nh + h) * hd * ds;              \
                                                                               \
    extern __shared__ float m3cs_sm[];                                         \
    /* Rows padded to ds+4 floats (80 B, still 16-byte aligned): at            \
     * ds = 16 an unpadded 64 B stride lands 8 consecutive lanes on the        \
     * same 8 banks (a 4-way conflict on the hottest loop); 80 B walks         \
     * all 32 banks. Address-only - the values and their order do not          \
     * move. */                                                                \
    int qk_ld = ds + 4;                                                        \
    float* sm_tri = m3cs_sm;                                                   \
    float* sm_q   = sm_tri + chunk_size * (chunk_size - 1) / 2;                \
    float* sm_k   = sm_q + chunk_size * qk_ld;                                 \
    float* sm_v   = sm_k + chunk_size * qk_ld;                                 \
    /* The entering state transposed, a row per state index n padded to        \
     * hd + 4 floats: the output pass reads four heads' columns of one row     \
     * as a float4, and a warp's eight column groups then lie in one           \
     * contiguous 128-byte run. Rows per head, ds + 4 floats apart, put        \
     * the groups four rows apart on two bank sets at ds = 16.                 \
     * Address-only: the values and their order do not move. */                \
    int ps_ld = hd + 4;                                                        \
    float* sm_ps  = sm_v + chunk_size * hd;                                    \
    float* sm_da  = sm_ps + ds * ps_ld;                                        \
    float* sm_qkd = sm_da + chunk_size;                                        \
                                                                               \
    int tid = threadIdx.x;                                                     \
    int nt = blockDim.x;                                                       \
                                                                               \
    /* Stage: pure copies of the values the scan reads. */                     \
    for (int i = tid; i < chunk_len * ds; i += nt) {                           \
        int t_local = i / ds;                                                  \
        int n = i % ds;                                                        \
        IDX base = (b * T + chunk_start + t_local) * nh * ds + h * ds;         \
        sm_q[t_local * qk_ld + n] = Q[base + n];                               \
        sm_k[t_local * qk_ld + n] = K_scaled[base + n];                        \
    }                                                                          \
    for (int i = tid; i < chunk_len * hd; i += nt) {                           \
        int t_local = i / hd;                                                  \
        int pp = i % hd;                                                       \
        sm_v[t_local * hd + pp] =                                              \
            x[(b * T + chunk_start + t_local) * d_inner + h * hd + pp];        \
    }                                                                          \
    for (int i = tid; i < hd * ds; i += nt) {                                  \
        sm_ps[(i % ds) * ps_ld + (i / ds)] = prev_states[state_head + i];      \
    }                                                                          \
    for (int i = tid; i < chunk_len; i += nt) {                                \
        sm_da[i] = dA_cumsum[cs_base + i];                                     \
        sm_qkd[i] = qk_dot[(b * T + chunk_start + i) * nh + h];                \
    }                                                                          \
    __syncthreads();                                                           \
                                                                               \
    /* Tile: one thread per (t, s) pair, the n-ascending dot. */               \
    int n_pairs = chunk_len * (chunk_len - 1) / 2;                             \
    /* The kernel is smem-throughput bound (MIO queue, measured 87% L1 */      \
    /* utilization at the serve shape); the wide path issues float4 LDS */     \
    /* where the scalar path issued four scalar LDS. The fold stays the */     \
    /* SAME serial n-ascending FMA chain either way - the vector load */       \
    /* only changes how bytes arrive, never the arithmetic or its order. */    \
    /* Uniform guard: every smem row is 16B-aligned when ds % 4 == 0 */        \
    /* (row strides are multiples of 4 floats; the arena base is 16B). */      \
    bool wide = (ds % 4 == 0) && (hd % 4 == 0);                                \
    for (int e = tid; e < n_pairs; e += nt) {                                  \
        /* Decode the row-major strict-lower-triangle index: row t holds */    \
        /* t entries starting at t*(t-1)/2. The float sqrt gives the row */    \
        /* up to rounding; the two guards pin it exactly. */                   \
        int t_local = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)e)) * 0.5f);    \
        while (t_local * (t_local - 1) / 2 > e) t_local--;                     \
        while ((t_local + 1) * t_local / 2 <= e) t_local++;                    \
        int s_local = e - t_local * (t_local - 1) / 2;                         \
        float decay = FAST_EXP(m3_nonpos(sm_da[t_local] - sm_da[s_local]));    \
        float qk_val = 0.0f;                                                   \
        if (wide) {                                                            \
            const float* q_row = sm_q + t_local * qk_ld;                       \
            const float* k_row = sm_k + s_local * qk_ld;                       \
            for (int n4 = 0; n4 < ds / 4; n4++) {                              \
                float4 qv = lds_f32x4(q_row + 4 * n4);                         \
                float4 kv = lds_f32x4(k_row + 4 * n4);                         \
                qk_val += qv.x * kv.x;                                         \
                qk_val += qv.y * kv.y;                                         \
                qk_val += qv.z * kv.z;                                         \
                qk_val += qv.w * kv.w;                                         \
            }                                                                  \
        } else {                                                               \
            for (int n = 0; n < ds; n++) {                                     \
                qk_val += sm_q[t_local * qk_ld + n] * sm_k[s_local * qk_ld + n]; \
            }                                                                  \
        }                                                                      \
        sm_tri[e] = vl_id(seq_idx, b, T, chunk_start + t_local)                \
                    == vl_id(seq_idx, b, T, chunk_start + s_local)             \
            ? decay * qk_val : 0.0f;                                           \
    }                                                                          \
    __syncthreads();                                                           \
                                                                               \
    float d_skip = D[h];                                                       \
    /* Outputs: the s-ascending y_diag chain and the n-ascending y_off dot */  \
    /* are single serial folds per output element, exactly as before. The */   \
    /* wide path assigns each thread FOUR p-columns of one t so the inner */   \
    /* s-loop issues one float4 LDS of V (plus the sm_tri broadcast) where */  \
    /* the scalar path issued four scalar LDS - four independent per-column */ \
    /* accumulator chains, each still strictly s-ascending. */                 \
    if (wide) {                                                                \
        int quads = hd / 4;                                                    \
        for (int o = tid; o < chunk_len * quads; o += nt) {                    \
            int t_local = o / quads;                                           \
            int p0 = (o % quads) * 4;                                          \
            float state_decay = FAST_EXP(sm_da[t_local]);                      \
            const float* q_row = sm_q + t_local * qk_ld;                       \
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;          \
            for (int n4 = 0; n4 < ds / 4; n4++) {                              \
                float4 qv = lds_f32x4(q_row + 4 * n4);                         \
                /* Row 4 n4 + k holds state index 4 n4 + k for the heads */    \
                /* p0 .. p0 + 3: component j of nk is head p0 + j. */          \
                const float* row = sm_ps + 4 * n4 * ps_ld + p0;                \
                float4 n0 = lds_f32x4(row);                                    \
                float4 n1 = lds_f32x4(row + ps_ld);                            \
                float4 n2 = lds_f32x4(row + 2 * ps_ld);                        \
                float4 n3 = lds_f32x4(row + 3 * ps_ld);                        \
                acc0 += qv.x * n0.x; acc0 += qv.y * n1.x; acc0 += qv.z * n2.x; acc0 += qv.w * n3.x; \
                acc1 += qv.x * n0.y; acc1 += qv.y * n1.y; acc1 += qv.z * n2.y; acc1 += qv.w * n3.y; \
                acc2 += qv.x * n0.z; acc2 += qv.y * n1.z; acc2 += qv.z * n2.z; acc2 += qv.w * n3.z; \
                acc3 += qv.x * n0.w; acc3 += qv.y * n1.w; acc3 += qv.z * n2.w; acc3 += qv.w * n3.w; \
            }                                                                  \
            bool on = t_local < cont;                                          \
            float y_off0 = (on ? acc0 : 0.0f) * state_decay;                   \
            float y_off1 = (on ? acc1 : 0.0f) * state_decay;                   \
            float y_off2 = (on ? acc2 : 0.0f) * state_decay;                   \
            float y_off3 = (on ? acc3 : 0.0f) * state_decay;                   \
            float yd0 = 0.0f, yd1 = 0.0f, yd2 = 0.0f, yd3 = 0.0f;              \
            int tri_row = t_local * (t_local - 1) / 2;                         \
            int lo = vl_seq_first(seq_idx, b, T, chunk_start, chunk_start + t_local) \
                     - chunk_start;                                            \
            /* Four rows' reads together ahead of their arithmetic; each chain */ \
            /* takes the same products in the same order as one row at a time. */ \
            int sr = lo;                                                       \
            for (; sr + 4 <= t_local; sr += 4) {                               \
                float w4[4];                                                   \
                float4 v4[4];                                                  \
                lds_f32_row4(sm_tri + tri_row + sr, w4);                       \
                lds_f32x4_rows4(sm_v + sr * hd + p0, hd, v4);                  \
                _Pragma("unroll")                                              \
                for (int j = 0; j < 4; j++) {                                  \
                    yd0 += w4[j] * v4[j].x;                                    \
                    yd1 += w4[j] * v4[j].y;                                    \
                    yd2 += w4[j] * v4[j].z;                                    \
                    yd3 += w4[j] * v4[j].w;                                    \
                }                                                              \
            }                                                                  \
            for (int s_local = sr; s_local < t_local; s_local++) {             \
                float w = sm_tri[tri_row + s_local];                           \
                float4 v =                                                     \
                    lds_f32x4(sm_v + s_local * hd + p0);                       \
                yd0 += w * v.x;                                                \
                yd1 += w * v.y;                                                \
                yd2 += w * v.z;                                                \
                yd3 += w * v.w;                                                \
            }                                                                  \
            float skip_w = d_skip + sm_qkd[t_local];                           \
            float4 vt = lds_f32x4(sm_v + t_local * hd + p0);                   \
            long long out_base =                                               \
                (long long)(b * T + chunk_start + t_local) * d_inner + h * hd + p0; \
            y_out[out_base + 0] = yd0 + y_off0 + skip_w * vt.x;                \
            y_out[out_base + 1] = yd1 + y_off1 + skip_w * vt.y;                \
            y_out[out_base + 2] = yd2 + y_off2 + skip_w * vt.z;                \
            y_out[out_base + 3] = yd3 + y_off3 + skip_w * vt.w;                \
        }                                                                      \
    } else {                                                                   \
        for (int o = tid; o < chunk_len * hd; o += nt) {                       \
            int t_local = o / hd;                                              \
            int p = o % hd;                                                    \
            float y_off = 0.0f;                                                \
            float state_decay = FAST_EXP(sm_da[t_local]);                      \
            for (int n = 0; n < ds; n++) {                                     \
                y_off += sm_q[t_local * qk_ld + n] * sm_ps[n * ps_ld + p];     \
            }                                                                  \
            if (t_local >= cont) y_off = 0.0f; y_off *= state_decay;           \
            float y_diag = 0.0f;                                               \
            int tri_row = t_local * (t_local - 1) / 2;                         \
            int lo = vl_seq_first(seq_idx, b, T, chunk_start, chunk_start + t_local) \
                     - chunk_start;                                            \
            for (int s_local = lo; s_local < t_local; s_local++) {             \
                y_diag += sm_tri[tri_row + s_local] * sm_v[s_local * hd + p];  \
            }                                                                  \
            float y_skip = (d_skip + sm_qkd[t_local]) * sm_v[t_local * hd + p]; \
            y_out[(b * T + chunk_start + t_local) * d_inner + h * hd + p] =    \
                y_diag + y_off + y_skip;                                       \
        }                                                                      \
    }                                                                          \
}

DEFINE_M3_CHUNK_SCAN_FWD_COOP_F32(m3_chunk_scan_fwd_coop, int)
DEFINE_M3_CHUNK_SCAN_FWD_COOP_F32(m3_chunk_scan_fwd_coop_i64, long long)

// ============================================================================
// Sections 6-9 (m3_chunk_scan_bwd / m3_state_passing_bwd / m3_chunk_state_bwd
// / m3_cumsum_bwd) removed with the monolithic-backward consolidation.
//
// The live chunked backward path now uses the monolithic m3_dqkv + m3_dqktheta
// path (sections 11-12 below) which has no atomicAdd. These four kernels
// were dead code (not referenced from any call site) carrying ~15 atomicAdd
// sites; deleting them simplifies the file and removes a potential
// determinism trap if anyone ever re-wires them.
//
// Forward chunked kernels (sections 1-5, 10) are untouched and still active
// when use_parallel_scan=true.
// ============================================================================

// ============================================================================
// 10. m3_extract_da_cs_sum -- Extract chunk-end cumsum values
// ============================================================================
// Grid: (B, n_chunks, ceil(nh/256)), Block: (min(nh, 256))
#define DEFINE_M3_EXTRACT_DA_CS_SUM(NAME, IDX)                                 \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ da_cs_sum,       /* [B * n_chunks * nh] */             \
    const float* __restrict__ da_cumsum, /* [B * n_chunks * nh * CS] */        \
    int B, int T, int nh, int CS,                                              \
    int y_first, /* this run's first y block */                                \
    int z_first /* this run's first z block */                                 \
) {                                                                            \
    IDX b = blockIdx.x;                                                        \
    int chunk = y_first + blockIdx.y;                                          \
    int h = (z_first + blockIdx.z) * blockDim.x + threadIdx.x;                 \
    if (h >= nh) return;                                                       \
    int n_chunks = (T + CS - 1) / CS;                                          \
    int chunk_start = chunk * CS;                                              \
    int last_t = min(CS, T - chunk_start) - 1;                                 \
    IDX cs_idx = ((b * n_chunks + chunk) * nh + h) * CS + last_t;              \
    IDX out_idx = (b * n_chunks + chunk) * nh + h;                             \
    da_cs_sum[out_idx] = da_cumsum[cs_idx];                                    \
}

DEFINE_M3_EXTRACT_DA_CS_SUM(m3_extract_da_cs_sum, int)
DEFINE_M3_EXTRACT_DA_CS_SUM(m3_extract_da_cs_sum_i64, long long)

// ============================================================================
// 11. m3_dqkv -- Monolithic chunked backward (reference backward stage 2)
// ============================================================================
//
// Translates Python mamba3_siso_bwd_kernel_dqkv.
// Grid: (nh, B), Block: (hd, 1, 1)
// Each block processes ALL chunks in reverse for one (head, batch).
//
// Thread p owns column p of d_ssm_states_acc[hd][ds] (ds register floats).
// Shared memory holds Q, K, V, dO tiles per chunk.
//
// Outputs: dQ_mid, dK_mid, dV, dADT, dQK_dot, dD_partials
// Strict-upper-triangle index for the CS x CS pair matrices: every
// consumer reads (a, b) with a < b only (dv/dk/dq intra + both dADT
// triangles), so only CS*(CS-1)/2 entries exist per matrix.
#define M3_TRI(a, b, CS) ((a) * (2 * (CS) - (a) - 1) / 2 + ((b) - (a) - 1))

// ============================================================================
// Reverse d_state decomposition for the chunk-parallel backward.
//
// The monolithic backward walked chunks serially per (b, h) block only
// because d_state carried across them. Decomposed:
//   state_terms:   B_c[p][n] = sum_t dO[t,p] * exp2(dA[t]) * Q[t,n]
//                  (ascending t within the chunk, zero-seeded);
//   state_passing: E_{c-1} = exp2(dA_sum_c) * E_c + B_c, E_last = 0,
//                  one serial lane per (b, h, p, n);
//   the main kernel then takes E_c as its entering d_state and runs
//   every chunk in parallel (grid z). The B-term grouping differs from
//   the fused accumulate of the serial walk (same release-window
//   family break as the M1 d-fold).
// ============================================================================
#define DEFINE_M3_DQKV_STATE_TERMS_F32(NAME, IDX)                              \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ state_B, /* [B*n_chunks*nh*hd*ds] */                   \
    const float* __restrict__ Q_rot,                                           \
    const float* __restrict__ DA_CS,                                           \
    const float* __restrict__ dO,                                              \
    int B, int T, int nh_total, int hd, int ds, int CS,                        \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    int h = blockIdx.x;                                                        \
    IDX b = y_first + blockIdx.y;                                              \
    int chunk_idx = z_first + blockIdx.z;                                      \
    int p = threadIdx.x;                                                       \
    int ty = threadIdx.y;                                                      \
    int TS = blockDim.y;                                                       \
    int n_chunks = (T + CS - 1) / CS;                                          \
    if (h >= nh_total || b >= B || p >= hd) return;                            \
    if (ds > MAMBA_RS_STATE_CAP || CS > 64) return;                            \
    int d_inner = nh_total * hd;                                               \
    int chunk_start = chunk_idx * CS;                                          \
    int chunk_len = min(CS, T - chunk_start);                                  \
    /* In a packed row (seq_idx) only the tokens before the chunk's */         \
    /* first start read the state entering it. */                              \
    int cont = vl_lower_bound(seq_idx, b, T, chunk_start, chunk_start + chunk_len, \
                              vl_id(seq_idx, b, T, chunk_start - 1) + 1) - chunk_start; \
    extern __shared__ float smem_st[];                                         \
    float* q_sm = smem_st;                 /* [CS*ds] */                       \
    float* do_sm = q_sm + CS * ds;         /* [CS*hd] */                       \
    float* expf_sm = do_sm + CS * hd;      /* [CS] */                          \
    for (int n = p; n < ds; n += hd) {                                         \
        for (int t = ty; t < chunk_len; t += TS) {                             \
            int gt = chunk_start + t;                                          \
            q_sm[t * ds + n] = Q_rot[((b * T + gt) * nh_total + h) * ds + n];  \
        }                                                                      \
    }                                                                          \
    for (int t = ty; t < chunk_len; t += TS) {                                 \
        int gt = chunk_start + t;                                              \
        do_sm[t * hd + p] = dO[(b * T + gt) * d_inner + h * hd + p];           \
    }                                                                          \
    {                                                                          \
        int lane = p + hd * ty;                                                \
        int nlanes = hd * TS;                                                  \
        for (int t = lane; t < chunk_len; t += nlanes) {                       \
            float dA_t =                                                       \
                DA_CS[((b * n_chunks + chunk_idx) * nh_total + h) * CS + t];   \
            expf_sm[t] = exp2f(dA_t * LOG2E);                                  \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    for (int n = ty; n < ds; n += TS) {                                        \
        float acc = 0.0f;                                                      \
        for (int t = 0; t < cont; t++) {                                       \
            acc += do_sm[t * hd + p] * expf_sm[t] * q_sm[t * ds + n];          \
        }                                                                      \
        state_B[((b * n_chunks + chunk_idx) * nh_total + h) * hd * ds          \
                + p * ds + n] = acc;                                           \
    }                                                                          \
}

DEFINE_M3_DQKV_STATE_TERMS_F32(m3_dqkv_state_terms, int)
DEFINE_M3_DQKV_STATE_TERMS_F32(m3_dqkv_state_terms_i64, long long)

#define DEFINE_M3_DQKV_STATE_TERMS(NAME, IDX, T_ACT)                          \
extern "C" __global__ void NAME(                                              \
    float* __restrict__ state_B,                                              \
    const T_ACT* __restrict__ Q_rot,                                          \
    const float* __restrict__ DA_CS,                                          \
    const T_ACT* __restrict__ dO,                                             \
    int B, int T, int nh_total, int hd, int ds, int CS,                       \
    int y_first, /* this run's first y block */                               \
    int z_first, /* this run's first z block */                               \
    const int* __restrict__ seq_idx                                           \
) {                                                                           \
    int h = blockIdx.x;                                                       \
    IDX b = y_first + blockIdx.y;                                             \
    int chunk_idx = z_first + blockIdx.z;                                     \
    int p = threadIdx.x;                                                      \
    int ty = threadIdx.y;                                                     \
    int TS = blockDim.y;                                                      \
    int n_chunks = (T + CS - 1) / CS;                                         \
    if (h >= nh_total || b >= B || p >= hd) return;                           \
    if (ds > MAMBA_RS_STATE_CAP || CS > 64) return;                           \
    int d_inner = nh_total * hd;                                              \
    int chunk_start = chunk_idx * CS;                                         \
    int chunk_len = min(CS, T - chunk_start);                                 \
    /* In a packed row (seq_idx) only the tokens before the chunk's */        \
    /* first start read the state entering it. */                             \
    int cont = vl_lower_bound(seq_idx, b, T, chunk_start, chunk_start + chunk_len, \
                              vl_id(seq_idx, b, T, chunk_start - 1) + 1) - chunk_start; \
    extern __shared__ float smem_st[];                                        \
    float* q_sm = smem_st;                                                    \
    float* do_sm = q_sm + CS * ds;                                            \
    float* expf_sm = do_sm + CS * hd;                                         \
    for (int n = p; n < ds; n += hd) {                                        \
        for (int t = ty; t < chunk_len; t += TS) {                            \
            int gt = chunk_start + t;                                         \
            q_sm[t * ds + n] =                                                \
                to_f(Q_rot[((b * T + gt) * nh_total + h) * ds + n]);          \
        }                                                                     \
    }                                                                         \
    for (int t = ty; t < chunk_len; t += TS) {                                \
        int gt = chunk_start + t;                                             \
        do_sm[t * hd + p] = to_f(dO[(b * T + gt) * d_inner + h * hd + p]);    \
    }                                                                         \
    {                                                                         \
        int lane = p + hd * ty;                                               \
        int nlanes = hd * TS;                                                 \
        for (int t = lane; t < chunk_len; t += nlanes) {                      \
            float dA_t =                                                      \
                DA_CS[((b * n_chunks + chunk_idx) * nh_total + h) * CS + t];  \
            expf_sm[t] = exp2f(dA_t * LOG2E);                                 \
        }                                                                     \
    }                                                                         \
    __syncthreads();                                                          \
    for (int n = ty; n < ds; n += TS) {                                       \
        float acc = 0.0f;                                                     \
        for (int t = 0; t < cont; t++) {                                      \
            acc += do_sm[t * hd + p] * expf_sm[t] * q_sm[t * ds + n];         \
        }                                                                     \
        state_B[((b * n_chunks + chunk_idx) * nh_total + h) * hd * ds         \
                + p * ds + n] = acc;                                          \
    }                                                                         \
}

DEFINE_M3_DQKV_STATE_TERMS(m3_dqkv_state_terms_bf16, int, __nv_bfloat16)
DEFINE_M3_DQKV_STATE_TERMS(m3_dqkv_state_terms_bf16_i64, long long, __nv_bfloat16)
DEFINE_M3_DQKV_STATE_TERMS(m3_dqkv_state_terms_f16, int, __half)
DEFINE_M3_DQKV_STATE_TERMS(m3_dqkv_state_terms_f16_i64, long long, __half)

// One serial lane per (b, h, p, n): folds the per-chunk terms into the
// entering state for every chunk (reverse order, deterministic). The fold
// past the first chunk is the gradient of the state the window entered
// with, which `seed_grad` takes when it is not null.
#define DEFINE_M3_DSTATE_PASSING_BWD(NAME, IDX)                                \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ enter_state, /* [B*n_chunks*nh*hd*ds] */               \
    const float* __restrict__ state_B,                                         \
    const float* __restrict__ DA_CS_SUM, /* [B*n_chunks*nh] */                 \
    int B, int nh_total, int hd, int ds, int n_chunks,                         \
    float* __restrict__ seed_grad, /* [B*nh*hd*ds] or nullptr */               \
    int T, int CS,                                                             \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    IDX total = (IDX)B * nh_total * hd * ds;                                   \
    if (idx >= total) return;                                                  \
    int n = (int)(idx % ds);                                                   \
    IDX rem = idx / ds;                                                        \
    int p = (int)(rem % hd);                                                   \
    rem /= hd;                                                                 \
    int h = (int)(rem % nh_total);                                             \
    IDX b = rem / nh_total;                                                    \
    float e = 0.0f;                                                            \
    for (int c = n_chunks - 1; c >= 0; c--) {                                  \
        IDX base = ((b * n_chunks + c) * nh_total + h) * hd * ds + p * ds + n; \
        enter_state[base] = e;                                                 \
        float a_c = exp2f(DA_CS_SUM[(b * n_chunks + c) * nh_total + h]         \
                          * LOG2E);                                            \
        /* A chunk where a sequence of a packed row starts passes */           \
        /* nothing back to the state entering it. */                           \
        e = vl_passes(seq_idx, b, T, c * CS, min(CS, T - c * CS))              \
            ? a_c * e + state_B[base]                                          \
            : state_B[base];                                                   \
    }                                                                          \
    if (seed_grad != nullptr) seed_grad[idx] = e;                              \
}

DEFINE_M3_DSTATE_PASSING_BWD(m3_dstate_passing_bwd, int)
DEFINE_M3_DSTATE_PASSING_BWD(m3_dstate_passing_bwd_i64, long long)

// The gradient of the trapezoidal fold a window adds to the state it enters
// with (m3_state_pass_fused): the fold scales v_state (x) k_state by
// c = dt0 * (1 - trap0), so
//   dL/dc = sum_{p,n} dL/dh_enter[p][n] * v_state[p] * k_state[n],
// which m3_ddt_dtrap takes as the scale gradient of the step before the
// window's first. A stateless window entered with zero K and V and gets
// zero. One block per (b, h); every thread sums a fixed stride of the
// elements in order and the block adds the partials in a fixed tree, so
// the bits do not depend on the launch. Block: 256 threads.
#define DEFINE_M3_ENTER_FOLD_BWD(NAME, IDX)                                    \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ d_scale_enter,   /* [B*nh] */                          \
    const float* __restrict__ seed_grad, /* [B*nh*hd*ds] */                    \
    const float* __restrict__ k_enter,   /* [B*nh*ds] */                       \
    const float* __restrict__ v_enter,   /* [B*nh*hd] */                       \
    int batch, int nh, int hd, int ds                                          \
) {                                                                            \
    __shared__ float part[256];                                                \
    const IDX bh = blockIdx.x;                                                 \
    if (bh >= (IDX)batch * nh) return;                                         \
    const int dim = hd * ds;                                                   \
    float acc = 0.0f;                                                          \
    for (int pd = threadIdx.x; pd < dim; pd += blockDim.x) {                   \
        const int p = pd / ds;                                                 \
        const int n = pd % ds;                                                 \
        acc = __fmaf_rn(seed_grad[bh * dim + pd],                              \
                        v_enter[bh * hd + p] * k_enter[bh * ds + n], acc);     \
    }                                                                          \
    part[threadIdx.x] = acc;                                                   \
    __syncthreads();                                                           \
    for (int half = blockDim.x / 2; half > 0; half >>= 1) {                    \
        if (threadIdx.x < half) part[threadIdx.x] += part[threadIdx.x + half]; \
        __syncthreads();                                                       \
    }                                                                          \
    if (threadIdx.x == 0) d_scale_enter[bh] = part[0];                         \
}

DEFINE_M3_ENTER_FOLD_BWD(m3_enter_fold_bwd, int)
DEFINE_M3_ENTER_FOLD_BWD(m3_enter_fold_bwd_i64, long long)

#define DEFINE_M3_DQKV_F32(NAME, IDX)                                          \
extern "C" __global__ void NAME(                                               \
    /* Outputs */                                                              \
    float* __restrict__ dQ_mid,       /* [B*T*nh*ds] */                        \
    float* __restrict__ dK_mid,       /* [B*T*nh*ds] */                        \
    float* __restrict__ dV,           /* [B*T*d_inner] */                      \
    float* __restrict__ dADT,         /* [B*T*nh] */                           \
    float* __restrict__ dQK_dot_out,  /* [B*T*nh] */                           \
    float* __restrict__ dD_partials,  /* [B * nh] per-(b,h) OUTPUT (no atomicAdd) */ \
    /* Inputs */                                                               \
    const float* __restrict__ Q_rot,       /* [B*T*nh*ds] */                   \
    const float* __restrict__ K_scaled,    /* [B*T*nh*ds] */                   \
    const float* __restrict__ V_in,        /* [B*T*d_inner] */                 \
    const float* __restrict__ DA_CS,       /* [B*n_chunks*nh*CS] */            \
    const float* __restrict__ DA_CS_SUM,   /* [B*n_chunks*nh] */               \
    const float* __restrict__ QK_dot_in,   /* [B*T*nh] */                      \
    const float* __restrict__ SSM_States,  /* [B*n_chunks*nh*hd*ds] */         \
    const float* __restrict__ dO,          /* [B*T*d_inner] */                 \
    const float* __restrict__ D_param,     /* [nh] */                          \
    const float* __restrict__ enter_state, /* [B*n_chunks*nh*hd*ds] */         \
    int B, int T, int nh_total, int hd, int ds, int CS,                        \
    int use_pair_mats,  /* 1 = smem holds the triangle pair matrices */        \
    int y_first, /* this run's first y block */                                \
    int z_first, /* this run's first z block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    /* t-split: blockDim = (hd, T_SPLIT). One head per block; */               \
    /* threadIdx.y strides the per-timestep loops so the block carries */      \
    /* hd*T_SPLIT live lanes instead of one warp (the 32-thread blocks */      \
    /* ran ~2% occupancy and owned 73% of the M3 step). Every output */        \
    /* element keeps exactly one owning lane with the same inner-loop */       \
    /* order as before, so outputs are bit-identical; only the lane-> */       \
    /* output assignment changed. */                                           \
    int h = blockIdx.x;                                                        \
    IDX b = y_first + blockIdx.y;                                              \
    int p = threadIdx.x;                                                       \
    int ty = threadIdx.y;                                                      \
    int TS = blockDim.y;                                                       \
    int lane = p + hd * ty;                                                    \
    int nlanes = hd * TS;                                                      \
    if (h >= nh_total || b >= B || p >= hd) return;                            \
    if (ds > MAMBA_RS_STATE_CAP || CS > 64) return;                            \
                                                                               \
    /* Warp-reduce mask: with t-split lanes the ty slices of one warp can */   \
    /* run DIFFERENT trip counts (chunk_len < TS or not a multiple), so a */   \
    /* whole-warp mask would name lanes that are not executing the */          \
    /* shuffle (UB per CUDA Programming Guide B.15.1). The mask names */       \
    /* only this hd-lane segment; a segment shares one ty, hence one */        \
    /* trip count, hence convergence. */                                       \
    unsigned warp_mask;                                                        \
    if (hd >= 32) {                                                            \
        warp_mask = 0xFFFFFFFFu;                                               \
    } else {                                                                   \
        unsigned lin = (unsigned)(threadIdx.y * blockDim.x + threadIdx.x);     \
        unsigned seg_base = (lin & 31u) / (unsigned)hd * (unsigned)hd;         \
        warp_mask = (((1u << hd) - 1u) << seg_base);                           \
    }                                                                          \
                                                                               \
    int d_inner = nh_total * hd;                                               \
    int n_chunks = (T + CS - 1) / CS;                                          \
    float D_val = D_param[h];                                                  \
    float dD_acc = 0.0f;                                                       \
                                                                               \
    /* Register state: the chunk's entering d_state column, produced by */     \
    /* m3_dqkv_state_terms + m3_dstate_passing_bwd (chunk-parallel). */        \
    float d_state[MAMBA_RS_STATE_CAP]; /* sized by the state capacity */       \
                                                                               \
    /* Shared memory layout (dynamically sized): ONE head per block, so */     \
    /* the whole dynamic allocation is this head's slice. */                   \
    extern __shared__ float smem_all[];                                        \
    float* smem = smem_all;                                                    \
    /* q_sm[CS][dsp], k_sm[CS][dsp], v_sm[CS][hdp], do_sm[CS][hdp] */          \
    /* da_cs_sm[CS], qk_dot_sm[CS], ssm_sm[hd][ds] (loaded cooperatively) */   \
    /* The four time-major tiles carry one float of padding per row: the */    \
    /* pair-matrix build reads consecutive rows across a warp, and an even */  \
    /* stride lands them on the same two banks. An odd stride only moves */    \
    /* where a value lives, never what it is. */                               \
    const int dsp = ds + 1;                                                    \
    const int hdp = hd + 1;                                                    \
    float* q_sm    = smem;                                                     \
    float* k_sm    = q_sm + CS * dsp;                                          \
    float* v_sm    = k_sm + CS * dsp;                                          \
    float* do_sm   = v_sm + CS * hdp;                                          \
    float* da_cs_sm = do_sm + CS * hdp;                                        \
    float* qk_sm   = da_cs_sm + CS;                                            \
    float* ssm_sm  = qk_sm + CS; /* [hd][ds] for SSM_States tile */            \
    /* Extra staging for the warp-parallel dADT section: the true */           \
    /* SSM_States tile (ssm_sm gets overwritten with d_state before the */     \
    /* section runs) plus the two per-step lanes the combine reads. */         \
    float* ssm2_sm   = ssm_sm + hd * ds; /* [hd][ds] */                        \
    float* dm_rev_sm = ssm2_sm + hd * ds; /* [CS] */                           \
    float* dm_vec_sm = dm_rev_sm + CS;    /* [CS] */                           \
    /* Strict-upper-triangle pair matrices K[a].Q[b] and */                    \
    /* V[a].dO[b] (a < b), computed once per chunk instead of per */           \
    /* consumer lane (hd-fold recompute). Triangle packing keeps the */        \
    /* tile under the block's shared-memory budget. */                         \
    /* When the matrix-inclusive tile would pass that budget */                \
    /* (large d_state), the launcher passes use_pair_mats = 0 with */          \
    /* the legacy tile size and every consumer falls back to the inline */     \
    /* dot — same ascending-index arithmetic, bit-identical values. */         \
    int tri_n = CS * (CS - 1) / 2;                                             \
    float* kq_mat  = dm_vec_sm + CS;      /* [tri_n] (only if use_pair_mats) */ \
    float* vdo_mat = kq_mat + tri_n;      /* [tri_n] */                        \
    /* Decay triangle exp2((da[b]-da[a])*LOG2E) for a < b plus */              \
    /* per-step exp_fwd/exp_rev lanes — the consumers recomputed these */      \
    /* transcendentals inline (~7.4k exp2f per (b,h,chunk) lane). */           \
    float* decay_mat  = vdo_mat + tri_n;   /* [tri_n] */                       \
    float* exp_fwd_sm = decay_mat + tri_n; /* [CS] */                          \
    float* exp_rev_sm = exp_fwd_sm + CS;   /* [CS] */                          \
                                                                               \
    {                                                                          \
        int chunk_idx = z_first + blockIdx.z;                                  \
        int chunk_start = chunk_idx * CS;                                      \
        int chunk_len = min(CS, T - chunk_start);                              \
        for (int n = 0; n < ds; n++) {                                         \
            d_state[n] = enter_state[                                          \
                ((b * n_chunks + chunk_idx) * nh_total + h) * hd * ds          \
                + p * ds + n];                                                 \
        }                                                                      \
                                                                               \
        /* === Cooperative load tiles into shared memory === */                \
        /* V and dO: thread p loads element p, timesteps split over ty. */     \
        for (int t = ty; t < CS; t += TS) {                                    \
            if (t < chunk_len) {                                               \
                int gt = chunk_start + t;                                      \
                v_sm[t * hdp + p] = V_in[(b * T + gt) * d_inner + h * hd + p]; \
                do_sm[t * hdp + p] = dO[(b * T + gt) * d_inner + h * hd + p];  \
            } else {                                                           \
                v_sm[t * hdp + p] = 0.0f;                                      \
                do_sm[t * hdp + p] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
        /* Q and K: each thread loads ALL ds entries for indices n stride hd. */ \
        /* OLD bug: `if (p < ds)` only worked when ds <= hd; for ds > hd */    \
        /* (e.g. ds=16, hd=8) entries n=hd..ds-1 were left as garbage in */    \
        /* shared memory, corrupting every dot/dQK computation downstream. */  \
        for (int n = p; n < ds; n += hd) {                                     \
            for (int t = ty; t < CS; t += TS) {                                \
                if (t < chunk_len) {                                           \
                    int gt = chunk_start + t;                                  \
                    q_sm[t * dsp + n] = Q_rot[((b * T + gt) * nh_total + h) * ds + n]; \
                    k_sm[t * dsp + n] = K_scaled[((b * T + gt) * nh_total + h) * ds + n]; \
                } else {                                                       \
                    q_sm[t * dsp + n] = 0.0f;                                  \
                    k_sm[t * dsp + n] = 0.0f;                                  \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* da_cs and qk_dot: p==0 lanes load, timesteps split over ty. */      \
        if (p == 0) {                                                          \
            for (int t = ty; t < CS; t += TS) {                                \
                if (t < chunk_len) {                                           \
                    da_cs_sm[t] = DA_CS[((b * n_chunks + chunk_idx) * nh_total + h) * CS + t]; \
                    qk_sm[t] = QK_dot_in[(b * T + chunk_start + t) * nh_total + h]; \
                } else {                                                       \
                    da_cs_sm[t] = 0.0f;                                        \
                    qk_sm[t] = 0.0f;                                           \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* SSM_States: ty==0 threads load their row (p) of [hd][ds] */         \
        if (ty == 0) {                                                         \
            for (int n = 0; n < ds; n++) {                                     \
                ssm_sm[p * ds + n] = SSM_States[((b * n_chunks + chunk_idx) * nh_total + h) * hd * ds + p * ds + n]; \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
                                                                               \
        /* Each (a,b) element is computed by exactly one thread with the */    \
        /* same ascending-index dot the inline consumers used, so every */     \
        /* reader sees a bit-identical value. The lanes walk the packed */     \
        /* triangle flat, every lane taking every nlanes-th slot: a walk */    \
        /* by row left most lanes idle on the short rows. A slot's (aa, */     \
        /* bb) is recovered incrementally from the row lengths. */             \
        if (use_pair_mats) {                                                   \
            float cs_sum_pre =                                                 \
                DA_CS_SUM[(b * n_chunks + chunk_idx) * nh_total + h];          \
            int row = 0;                                                       \
            int row_start = 0;                                                 \
            for (int idx = lane; idx < tri_n; idx += nlanes) {                 \
                while (row_start + (CS - 1 - row) <= idx) {                    \
                    row_start += CS - 1 - row;                                 \
                    row++;                                                     \
                }                                                              \
                int aa = row;                                                  \
                int bb = idx - row_start + aa + 1;                             \
                float kq_acc = 0.0f;                                           \
                for (int n = 0; n < ds; n++)                                   \
                    kq_acc += k_sm[aa * dsp + n] * q_sm[bb * dsp + n];         \
                kq_mat[idx] = kq_acc;                                          \
                float vdo_acc = 0.0f;                                          \
                for (int pp = 0; pp < hd; pp++)                                \
                    vdo_acc += v_sm[aa * hdp + pp] * do_sm[bb * hdp + pp];     \
                vdo_mat[idx] = vdo_acc;                                        \
                /* consumer-exact expression */                                \
                decay_mat[idx] =                                               \
                    exp2f((da_cs_sm[bb] - da_cs_sm[aa]) * LOG2E);              \
            }                                                                  \
            for (int tt = lane; tt < CS; tt += nlanes) {                       \
                exp_fwd_sm[tt] = exp2f(da_cs_sm[tt] * LOG2E);                  \
                exp_rev_sm[tt] = exp2f((cs_sum_pre - da_cs_sm[tt]) * LOG2E);   \
            }                                                                  \
            __syncthreads();                                                   \
        }                                                                      \
                                                                               \
        float da_cs_chunk_sum = DA_CS_SUM[(b * n_chunks + chunk_idx) * nh_total + h]; \
                                                                               \
        /* === Compute per-timestep outputs === */                             \
        /* dV, dQK_dot for each timestep; timesteps split over ty. */          \
        for (int t = ty; t < chunk_len; t += TS) {                             \
            int gt = chunk_start + t;                                          \
            float dA_t = da_cs_sm[t];                                          \
            const VlSpan sp = vl_span(seq_idx, b, T, chunk_start, chunk_len, t); \
            float exp_rev_t = use_pair_mats                                    \
                ? exp_rev_sm[t]                                                \
                : exp2f((da_cs_chunk_sum - dA_t) * LOG2E);                     \
                                                                               \
            /* --- dV[t,h,p] --- */                                            \
            /* Intra: sum_{s: s has t as source, i.e., t<s in causal} */       \
            /* P^T[t,s] = sum_n(K[t,n]*Q[s,n]) * exp(dA[s]-dA[t]) for s > t */ \
            float dv_intra = 0.0f;                                             \
            for (int s = t + 1; s < sp.end; s++) {                             \
                float kq;                                                      \
                if (use_pair_mats) {                                           \
                    kq = kq_mat[M3_TRI(t, s, CS)];                             \
                } else {                                                       \
                    kq = 0.0f;                                                 \
                    for (int n = 0; n < ds; n++)                               \
                        kq += k_sm[t * dsp + n] * q_sm[s * dsp + n];           \
                }                                                              \
                float decay = use_pair_mats                                    \
                    ? decay_mat[M3_TRI(t, s, CS)]                              \
                    : exp2f((da_cs_sm[s] - dA_t) * LOG2E);                     \
                dv_intra += kq * decay * do_sm[s * hdp + p];                   \
            }                                                                  \
            /* Inter: K[t] @ d_state^T[p] * exp_rev */                         \
            float dv_inter = 0.0f;                                             \
            for (int n = 0; n < ds; n++)                                       \
                dv_inter += k_sm[t * dsp + n] * d_state[n];                    \
            if (!sp.to_end) dv_inter = 0.0f; dv_inter *= exp_rev_t;            \
            /* Skip: dO * (D + qk_dot) */                                      \
            float dv_skip = do_sm[t * hdp + p] * (D_val + qk_sm[t]);           \
                                                                               \
            dV[(b * T + gt) * d_inner + h * hd + p] = dv_intra + dv_inter + dv_skip; \
                                                                               \
            /* --- dQK_dot[t,h] and dD --- */                                  \
            /* dQK_dot = sum_p(dO[t,p] * V[t,p]) */                            \
            float dqk_val = do_sm[t * hdp + p] * v_sm[t * hdp + p];            \
            /* Warp reduce */                                                  \
            for (int off = hd / 2; off > 0; off >>= 1)                         \
                dqk_val += __shfl_down_sync(warp_mask, dqk_val, off, hd);      \
            if (p == 0) {                                                      \
                dQK_dot_out[(b * T + gt) * nh_total + h] = dqk_val;            \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        /* dD: resum the freshly stored dQK lane in ascending t on ONE */      \
        /* lane — the historical accumulation order (t ascending within */     \
        /* the chunk, chunks in reverse). Summing per-ty partials would */     \
        /* interleave that order and move bits. */                             \
        if (p == 0 && ty == 0) {                                               \
            for (int t = 0; t < chunk_len; t++)                                \
                dD_acc += dQK_dot_out[(b * T + chunk_start + t) * nh_total + h]; \
        }                                                                      \
                                                                               \
        /* === dK_mid and dQ_mid === */                                        \
        /* These need sum over hd (reduction across threads). */               \
        /* dK_mid[t,n] = sum_p(intra_contrib[t,p,n] + inter_contrib[t,p,n]) */ \
        /* Threads stride over state dims: for ds > hd, thread p covers */     \
        /* n = p, p+hd, p+2*hd, ... (mirrors the cooperative Q/K load above; */ \
        /* the old `if (p < ds) { n = p; }` silently zeroed n >= hd). */       \
        for (int n = p; n < ds; n += hd) {                                     \
            for (int t = ty; t < chunk_len; t += TS) {                         \
                int gt = chunk_start + t;                                      \
                float dA_t = da_cs_sm[t];                                      \
                const VlSpan sp = vl_span(seq_idx, b, T, chunk_start, chunk_len, t); \
                                                                               \
                /* dK_mid[t,n]: intra = sum_{s>t} dO[s]@V[s]^T * exp * Q[s,n] */ \
                /* This is dp_t^T[*,t] @ Q[*,n] */                             \
                /* dp_t[i,j] = sum_p(V[i,p]*dO[j,p]) * mask[i,j] */            \
                /* dp_t^T[j,i] = dp_t[i,j] */                                  \
                /* (dp_t^T @ Q)[t, n] = sum_s(dp_t[s,t] * Q[s,n]) */           \
                /* = sum_s(sum_p(V[s,p]*dO[t,p]) * mask[s,t] * Q[s,n]) */      \
                /* Wait — mask is causal: mask[i,j] = exp(dA[j]-dA[i]) if j>i else 0 */ \
                /* dp_t[i,j] includes mask. For column t of dp_t: dp_t[s,t] for s where t > s */ \
                /* Actually in Python: dp_t = V @ dO^T * mask */               \
                /* dp_t[s1, s2] = sum_p(V[s1,p]*dO[s2,p]) * (s2>s1 ? exp(dA[s2]-dA[s1]) : 0) */ \
                /* (dp_t @ Q)[s1, n] = sum_s2>s1 (sum_p(V[s1,p]*dO[s2,p]) * exp(dA[s2]-dA[s1]) * Q[s2,n]) */ \
                /* We want acc_dk[t, n] = (dp_t @ Q)[t, n] — this is dp_t row t dot Q col n */ \
                /* But dp_t = V@dO^T ⊙ mask, so dp_t[t, s] = sum_p(V[t,p]*dO[s,p]) * mask[t,s] */ \
                /* mask[t,s] = exp(dA[s]-dA[t]) if s>t else 0 */               \
                /* So acc_dk[t,n] = sum_{s>t} sum_p(V[t,p]*dO[s,p]) * exp(dA[s]-dA[t]) * Q[s,n] */ \
                                                                               \
                float dk_intra = 0.0f;                                         \
                for (int s = t + 1; s < sp.end; s++) {                         \
                    float vdo;                                                 \
                    if (use_pair_mats) {                                       \
                        vdo = vdo_mat[M3_TRI(t, s, CS)];                       \
                    } else {                                                   \
                        vdo = 0.0f;                                            \
                        for (int pp = 0; pp < hd; pp++)                        \
                            vdo += v_sm[t * hdp + pp] * do_sm[s * hdp + pp];   \
                    }                                                          \
                    float decay = use_pair_mats                                \
                        ? decay_mat[M3_TRI(t, s, CS)]                          \
                        : exp2f((da_cs_sm[s] - dA_t) * LOG2E);                 \
                    dk_intra += vdo * decay * q_sm[s * dsp + n];               \
                }                                                              \
                /* Inter-chunk dK contribution added in second pass below (after d_state → shared memory) */ \
                dK_mid[((b * T + gt) * nh_total + h) * ds + n] = dk_intra;     \
                                                                               \
                /* dQ_mid[t,n]: similar structure */                           \
                /* acc_dq[t,n] = sum_{s<t} sum_p(V[s,p]*dO[t,p]) * exp(dA[t]-dA[s]) * K[s,n]  (wrong?) */ \
                /* Actually from Python: s_block^T @ K where s_block = V@dO^T ⊙ mask (same as dp_t) */ \
                /* acc_dq = (dp_t^T @ K)[t, n] */                              \
                /* dp_t^T[t, s] = dp_t[s, t] = sum_p(V[s,p]*dO[t,p]) * mask[s,t] */ \
                /* mask[s,t] = exp(dA[t]-dA[s]) if t>s else 0 */               \
                /* So: acc_dq[t,n] = sum_{s<t} sum_p(V[s,p]*dO[t,p]) * exp(dA[t]-dA[s]) * K[s,n] */ \
                float dq_intra = 0.0f;                                         \
                for (int s = sp.first; s < t; s++) {                           \
                    float vdo;                                                 \
                    if (use_pair_mats) {                                       \
                        vdo = vdo_mat[M3_TRI(s, t, CS)];                       \
                    } else {                                                   \
                        vdo = 0.0f;                                            \
                        for (int pp = 0; pp < hd; pp++)                        \
                            vdo += v_sm[s * hdp + pp] * do_sm[t * hdp + pp];   \
                    }                                                          \
                    float decay = use_pair_mats                                \
                        ? decay_mat[M3_TRI(s, t, CS)]                          \
                        : exp2f((dA_t - da_cs_sm[s]) * LOG2E);                 \
                    dq_intra += vdo * decay * k_sm[s * dsp + n];               \
                }                                                              \
                /* Inter: dO[t] @ ssm_states * exp */                          \
                float dq_inter = 0.0f;                                         \
                for (int pp = 0; pp < hd; pp++)                                \
                    dq_inter += do_sm[t * hdp + pp] * ssm_sm[pp * ds + n];     \
                if (!sp.from_before) dq_inter = 0.0f;                          \
                dq_inter *=                                                    \
                    use_pair_mats ? exp_fwd_sm[t] : exp2f(dA_t * LOG2E);       \
                                                                               \
                dQ_mid[((b * T + gt) * nh_total + h) * ds + n] = dq_intra + dq_inter; \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
                                                                               \
        /* === Store d_state to shared for inter-chunk dK_mid contribution === */ \
        /* Every ty lane replicates the same d_state bits; ty==0 writes. */    \
        if (ty == 0) {                                                         \
            for (int n = 0; n < ds; n++)                                       \
                ssm_sm[p * ds + n] = d_state[n];                               \
        }                                                                      \
        __syncthreads();                                                       \
                                                                               \
        /* Now fix dK_mid inter-chunk: sum_p(V[t,p] * d_state[p][n]) * exp_rev */ \
        /* Strided over state dims so ds > hd is fully covered (see dK/dQ_mid above). */ \
        for (int n = p; n < ds; n += hd) {                                     \
            for (int t = ty; t < chunk_len; t += TS) {                         \
                float dk_inter = 0.0f;                                         \
                float exp_rev_t = use_pair_mats                                \
                    ? exp_rev_sm[t]                                            \
                    : exp2f((da_cs_chunk_sum - da_cs_sm[t]) * LOG2E);          \
                for (int pp = 0; pp < hd; pp++)                                \
                    dk_inter += v_sm[t * hdp + pp] * ssm_sm[pp * ds + n];      \
                if (vl_id(seq_idx, b, T, chunk_start + t)                      \
                    != vl_id(seq_idx, b, T, chunk_start + chunk_len - 1))      \
                    dk_inter = 0.0f;                                           \
                dk_inter *= exp_rev_t;                                         \
                int gt = chunk_start + t;                                      \
                dK_mid[((b * T + gt) * nh_total + h) * ds + n] += dk_inter;    \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
                                                                               \
        /* === dADT computation (all 4 parts) === */                           \
        /* Formerly a single-lane section (~95% of the kernel serial on */     \
        /* lane 0). Restructured per OUTPUT step: each lane owns a strided */  \
        /* set of steps and computes that step's contributions with a */       \
        /* fixed serial order (below-diagonal pairs ascending, then */         \
        /* above-diagonal), so no cross-lane reduction exists and the */       \
        /* result is deterministic by construction. The true SSM_States */     \
        /* tile is staged in shared memory once (ssm_sm already holds */       \
        /* d_state at this point), replacing the per-element global */         \
        /* re-reads of the old code. */                                        \
        for (int n = lane; n < hd * ds; n += nlanes) {                         \
            ssm2_sm[n] =                                                       \
                SSM_States[((b * n_chunks + chunk_idx) * nh_total + h) * hd * ds + n]; \
        }                                                                      \
        __syncthreads();                                                       \
        for (int t = lane; t < chunk_len; t += nlanes) {                       \
            float acc = 0.0f;                                                  \
            const VlSpan sp = vl_span(seq_idx, b, T, chunk_start, chunk_len, t); \
            /* Below-diagonal: pairs (i, t) with i < t contribute +dAinv. */   \
            for (int i = sp.first; i < t; i++) {                               \
                float vdo;                                                     \
                if (use_pair_mats) {                                           \
                    vdo = vdo_mat[M3_TRI(i, t, CS)];                           \
                } else {                                                       \
                    vdo = 0.0f;                                                \
                    for (int pp = 0; pp < hd; pp++)                            \
                        vdo += v_sm[i * hdp + pp] * do_sm[t * hdp + pp];       \
                }                                                              \
                float decay = use_pair_mats                                    \
                    ? decay_mat[M3_TRI(i, t, CS)]                              \
                    : exp2f((da_cs_sm[t] - da_cs_sm[i]) * LOG2E);              \
                float kq;                                                      \
                if (use_pair_mats) {                                           \
                    kq = kq_mat[M3_TRI(i, t, CS)];                             \
                } else {                                                       \
                    kq = 0.0f;                                                 \
                    for (int n = 0; n < ds; n++)                               \
                        kq += k_sm[i * dsp + n] * q_sm[t * dsp + n];           \
                }                                                              \
                acc += vdo * decay * kq;                                       \
            }                                                                  \
            /* Above-diagonal: pairs (t, j) with j > t contribute -dAinv. */   \
            for (int j = t + 1; j < sp.end; j++) {                             \
                float vdo;                                                     \
                if (use_pair_mats) {                                           \
                    vdo = vdo_mat[M3_TRI(t, j, CS)];                           \
                } else {                                                       \
                    vdo = 0.0f;                                                \
                    for (int pp = 0; pp < hd; pp++)                            \
                        vdo += v_sm[t * hdp + pp] * do_sm[j * hdp + pp];       \
                }                                                              \
                float decay = use_pair_mats                                    \
                    ? decay_mat[M3_TRI(t, j, CS)]                              \
                    : exp2f((da_cs_sm[j] - da_cs_sm[t]) * LOG2E);              \
                float kq;                                                      \
                if (use_pair_mats) {                                           \
                    kq = kq_mat[M3_TRI(t, j, CS)];                             \
                } else {                                                       \
                    kq = 0.0f;                                                 \
                    for (int n = 0; n < ds; n++)                               \
                        kq += k_sm[t * dsp + n] * q_sm[j * dsp + n];           \
                }                                                              \
                acc -= vdo * decay * kq;                                       \
            }                                                                  \
            /* Entering-state term: Q @ ssm_states^T dot dO * exp. */          \
            float qs_do = 0.0f;                                                \
            for (int pp = 0; pp < hd; pp++) {                                  \
                float qs = 0.0f;                                               \
                for (int n = 0; n < ds; n++)                                   \
                    qs += q_sm[t * dsp + n] * ssm2_sm[pp * ds + n];            \
                qs_do += qs * do_sm[t * hdp + pp];                             \
            }                                                                  \
            if (sp.from_before) acc += qs_do                                   \
                * (use_pair_mats ? exp_fwd_sm[t] : exp2f(da_cs_sm[t] * LOG2E)); \
            dm_rev_sm[t] = acc;                                                \
            /* Exit-state term: K @ d_state^T dot V * exp_rev */               \
            /* (ssm_sm holds d_state here). */                                 \
            float dsk_v = 0.0f;                                                \
            for (int pp = 0; pp < hd; pp++) {                                  \
                float dsk = 0.0f;                                              \
                for (int n = 0; n < ds; n++)                                   \
                    dsk += k_sm[t * dsp + n] * ssm_sm[pp * ds + n];            \
                dsk_v += dsk * v_sm[t * hdp + pp];                             \
            }                                                                  \
            dm_vec_sm[t] = !sp.to_end ? 0.0f : dsk_v                           \
                * (use_pair_mats ? exp_rev_sm[t]                               \
                                 : exp2f((da_cs_chunk_sum - da_cs_sm[t]) * LOG2E)); \
        }                                                                      \
        __syncthreads();                                                       \
        if (p == 0 && ty == 0) {                                               \
            /* Scalar term + the serial reverse-cumsum combine (cheap, */      \
            /* and its order is the numeric contract). */                      \
            float dM_scalar = 0.0f;                                            \
            for (int pp = 0; pp < hd; pp++) {                                  \
                for (int n = 0; n < ds; n++)                                   \
                    dM_scalar += ssm2_sm[pp * ds + n] * ssm_sm[pp * ds + n];   \
            }                                                                  \
            dM_scalar *= exp2f(da_cs_chunk_sum * LOG2E);                       \
                                                                               \
            /* A packed row combines each of its sequences in the chunk on */  \
            /* its own: no term of one reaches the decays of another, and */   \
            /* the state pass's scalar belongs to a chunk with no start. */    \
            const bool passes = vl_passes(seq_idx, b, T, chunk_start, chunk_len); \
            for (int a = 0; a < chunk_len;) {                                  \
                const int e = vl_span(seq_idx, b, T, chunk_start, chunk_len, a).end; \
                float total_rev = 0.0f;                                        \
                for (int t = a; t < e; t++) total_rev += dm_rev_sm[t];         \
                if (passes) total_rev += dM_scalar;                            \
                float cumsum = 0.0f;                                           \
                for (int t = a; t < e; t++) {                                  \
                    cumsum += dm_vec_sm[t] - dm_rev_sm[t];                     \
                    /* Same grouping as the historical in-place update */      \
                    /* (dM_rev[t] += total + cumsum - vec): the added term */  \
                    /* is evaluated first, then added once. */                 \
                    float out = dm_rev_sm[t] + (total_rev + cumsum - dm_vec_sm[t]); \
                    int gt = chunk_start + t;                                  \
                    dADT[(b * T + gt) * nh_total + h] = out;                   \
                }                                                              \
                a = e;                                                         \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
                                                                               \
    }                                                                          \
                                                                               \
    /* Store dD per-(b, chunk, h) — caller reduces across rows. */             \
    if (p == 0 && ty == 0)                                                     \
        dD_partials[(b * n_chunks + (z_first + blockIdx.z)) * nh_total + h] = dD_acc; \
}

DEFINE_M3_DQKV_F32(m3_dqkv, int)
DEFINE_M3_DQKV_F32(m3_dqkv_i64, long long)

// ============================================================================
// Row quads for the dQK-theta stage and the sliced chunked kernels
// ============================================================================

// Four consecutive values of a row, the first `valid` of them and zeros
// past them: one vector load when all four are there and the row keeps its
// alignment (`vec`), else one load a value. Widening a half is exact, so
// either way the values are what an element-wise to_f reads.
__device__ __forceinline__ float4 m3_load_quad(const float* p, int valid, bool vec) {
    if (vec && valid >= 4) return *reinterpret_cast<const float4*>(p);
    float4 r = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (valid > 0) r.x = p[0];
    if (valid > 1) r.y = p[1];
    if (valid > 2) r.z = p[2];
    if (valid > 3) r.w = p[3];
    return r;
}
__device__ __forceinline__ float4 m3_load_quad(const __nv_bfloat16* p, int valid, bool vec) {
    if (vec && valid >= 4) {
        const uint2 u = *reinterpret_cast<const uint2*>(p);
        const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        return make_float4(lo.x, lo.y, hi.x, hi.y);
    }
    float4 r = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (valid > 0) r.x = to_f(p[0]);
    if (valid > 1) r.y = to_f(p[1]);
    if (valid > 2) r.z = to_f(p[2]);
    if (valid > 3) r.w = to_f(p[3]);
    return r;
}
__device__ __forceinline__ float4 m3_load_quad(const __half* p, int valid, bool vec) {
    if (vec && valid >= 4) {
        const uint2 u = *reinterpret_cast<const uint2*>(p);
        const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
        return make_float4(lo.x, lo.y, hi.x, hi.y);
    }
    float4 r = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (valid > 0) r.x = to_f(p[0]);
    if (valid > 1) r.y = to_f(p[1]);
    if (valid > 2) r.z = to_f(p[2]);
    if (valid > 3) r.w = to_f(p[3]);
    return r;
}
// The quad loader through the read-only data path (`__ldg`), for inputs a
// kernel never writes. A kernel that also reads shared memory through PTX
// gets no read-only loads from NVVM on its own: the PTX reads count as
// possible writes to any memory.
__device__ __forceinline__ float4 m3_load_quad_ro(const float* p, int valid, bool vec) {
    if (vec && valid >= 4) return __ldg(reinterpret_cast<const float4*>(p));
    float4 r = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (valid > 0) r.x = __ldg(p);
    if (valid > 1) r.y = __ldg(p + 1);
    if (valid > 2) r.z = __ldg(p + 2);
    if (valid > 3) r.w = __ldg(p + 3);
    return r;
}
__device__ __forceinline__ float4 m3_load_quad_ro(const __nv_bfloat16* p, int valid, bool vec) {
    if (vec && valid >= 4) {
        const uint2 u = __ldg(reinterpret_cast<const uint2*>(p));
        const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        return make_float4(lo.x, lo.y, hi.x, hi.y);
    }
    float4 r = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (valid > 0) r.x = to_f(__ldg(p));
    if (valid > 1) r.y = to_f(__ldg(p + 1));
    if (valid > 2) r.z = to_f(__ldg(p + 2));
    if (valid > 3) r.w = to_f(__ldg(p + 3));
    return r;
}
__device__ __forceinline__ float4 m3_load_quad_ro(const __half* p, int valid, bool vec) {
    if (vec && valid >= 4) {
        const uint2 u = __ldg(reinterpret_cast<const uint2*>(p));
        const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
        return make_float4(lo.x, lo.y, hi.x, hi.y);
    }
    float4 r = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (valid > 0) r.x = to_f(__ldg(p));
    if (valid > 1) r.y = to_f(__ldg(p + 1));
    if (valid > 2) r.z = to_f(__ldg(p + 2));
    if (valid > 3) r.w = to_f(__ldg(p + 3));
    return r;
}

// Store the first `valid` of four values of a row, each rounded once to
// the row's dtype as the element-wise store rounds it; one vector store when
// all four go and the row keeps its alignment.
__device__ __forceinline__ void m3_store_quad(float* p, float4 v, int valid, bool vec) {
    if (vec && valid >= 4) {
        *reinterpret_cast<float4*>(p) = v;
        return;
    }
    if (valid > 0) p[0] = v.x;
    if (valid > 1) p[1] = v.y;
    if (valid > 2) p[2] = v.z;
    if (valid > 3) p[3] = v.w;
}
__device__ __forceinline__ void m3_store_quad(__nv_bfloat16* p, float4 v, int valid, bool vec) {
    if (vec && valid >= 4) {
        __nv_bfloat162 lo, hi;
        lo.x = __float2bfloat16_rn(v.x);
        lo.y = __float2bfloat16_rn(v.y);
        hi.x = __float2bfloat16_rn(v.z);
        hi.y = __float2bfloat16_rn(v.w);
        uint2 u;
        u.x = *reinterpret_cast<const unsigned int*>(&lo);
        u.y = *reinterpret_cast<const unsigned int*>(&hi);
        *reinterpret_cast<uint2*>(p) = u;
        return;
    }
    if (valid > 0) p[0] = __float2bfloat16_rn(v.x);
    if (valid > 1) p[1] = __float2bfloat16_rn(v.y);
    if (valid > 2) p[2] = __float2bfloat16_rn(v.z);
    if (valid > 3) p[3] = __float2bfloat16_rn(v.w);
}
__device__ __forceinline__ void m3_store_quad(__half* p, float4 v, int valid, bool vec) {
    if (vec && valid >= 4) {
        __half2 lo, hi;
        lo.x = __float2half_rn(v.x);
        lo.y = __float2half_rn(v.y);
        hi.x = __float2half_rn(v.z);
        hi.y = __float2half_rn(v.w);
        uint2 u;
        u.x = *reinterpret_cast<const unsigned int*>(&lo);
        u.y = *reinterpret_cast<const unsigned int*>(&hi);
        *reinterpret_cast<uint2*>(p) = u;
        return;
    }
    if (valid > 0) p[0] = __float2half_rn(v.x);
    if (valid > 1) p[1] = __float2half_rn(v.y);
    if (valid > 2) p[2] = __float2half_rn(v.z);
    if (valid > 3) p[3] = __float2half_rn(v.w);
}

// Whether a row of `T_ACT` starting at `p` takes vector loads: four values
// in one aligned load.
#define M3_QUAD_ALIGNED(p, T_ACT) \
    ((reinterpret_cast<unsigned long long>(p) & (4 * sizeof(T_ACT) - 1)) == 0)


// ============================================================================
// 12. m3_dqktheta -- Inverse RoPE + scale + bias gradients (reference backward stage 3)
// ============================================================================
// One thread a (b, t, h) row. The rows lie back to back: a row's Q, K and
// their gradients are its ds values at row * ds, its angles the n_angles
// values at row * n_angles, and its scale, gamma, dQK dot, dScale and
// dGamma one value at row. The thread walks its row a quad at a time in
// registers: the two sums (dScale's, and the raw Q.K that dGamma scales)
// take their terms in state order, and every other output of a rotary pair
// needs that pair alone, so no row is ever held whole. Each value is
// written as the stage's earlier whole-row form wrote it, and where that
// form stored a value and read it back (the scaled dK, a rotation the dQK
// term is then added to) the value is rounded there too. The compiler
// fuses a sum of products itself, the first product, but leaves a
// difference to the driver's assembler, so a difference is written as that
// form's source wrote it and gets the same arithmetic on every compiler and
// driver. K's forward rotation is spelled out: that form computed it apart
// from the derivatives that reuse its products, so it must not share them.
// The outputs keep that form's bits.

// What a thread carries along its row.
struct M3DqktRow {
    const float* angles;  // the row's angles
    float* d_angles;      // the row's angle gradients
    int n_angles;
    float scale;          // the row's scale
    float dg;             // gamma times the dQK dot
    float d_scale;        // dK times rotated K, summed in state order
    float qk;             // raw Q times raw K, summed in state order
};

// A state past the rotary pairs, where K is its own rotation.
__device__ __forceinline__ void m3_dqkt_plain(
    M3DqktRow& r, float q, float k, float dq, float dk, float& oq, float& ok
) {
    r.d_scale = __fmaf_rn(dk, k, r.d_scale);
    r.qk = __fmaf_rn(q, k, r.qk);
    oq = __fmaf_rn(r.dg, k, dq);
    ok = __fmaf_rn(r.dg, q, __fmul_rn(r.scale, dk));
}

// States 2p and 2p + 1 of a row: the rotary pair of angle p while p is
// below n_angles, two plain states past them. (q0, q1) and the rest are
// the pair's values; (oq0, oq1) and (ok0, ok1) its pre-RoPE gradients.
__device__ __forceinline__ void m3_dqkt_two(
    M3DqktRow& r, int p, float q0, float q1, float k0, float k1, float dq0, float dq1,
    float dk0, float dk1, float& oq0, float& oq1, float& ok0, float& ok1
) {
    if (p >= r.n_angles) {
        m3_dqkt_plain(r, q0, k0, dq0, dk0, oq0, ok0);
        m3_dqkt_plain(r, q1, k1, dq1, dk1, oq1, ok1);
        return;
    }
    const float theta = r.angles[p];
    const float c = cosf(theta);
    const float s = sinf(theta);
    // K rotated forward, for dScale.
    const float kr0 = k0 * c - k1 * s;
    const float kr1 = __fmaf_rn(k0, s, __fmul_rn(k1, c));
    r.d_scale = __fmaf_rn(dk0, kr0, r.d_scale);
    r.d_scale = __fmaf_rn(dk1, kr1, r.d_scale);
    r.qk = __fmaf_rn(q0, k0, r.qk);
    r.qk = __fmaf_rn(q1, k1, r.qk);
    // dK scaled, both gradients rotated back, then the dQK path added.
    const float sk0 = __fmul_rn(r.scale, dk0);
    const float sk1 = __fmul_rn(r.scale, dk1);
    oq0 = __fmaf_rn(r.dg, k0, dq0 * c + dq1 * s);
    oq1 = __fmaf_rn(r.dg, k1, -dq0 * s + dq1 * c);
    ok0 = __fmaf_rn(r.dg, q0, sk0 * c + sk1 * s);
    ok1 = __fmaf_rn(r.dg, q1, -sk0 * s + sk1 * c);
    // The rotations' derivatives in theta against the gradients that
    // reached them: dQ as it came, dK scaled.
    const float aq = -q0 * s - q1 * c;
    const float bq = q0 * c - q1 * s;
    const float ak = -k0 * s - k1 * c;
    const float bk = k0 * c - k1 * s;
    const float dtheta_q = dq0 * aq + dq1 * bq;
    const float dtheta_k = sk0 * ak + sk1 * bk;
    r.d_angles[p] = dtheta_q + dtheta_k;
}

template <typename IDX, typename T_ACT>
__device__ __forceinline__ void m3_dqktheta_row(
    float* __restrict__ dQ_pre, float* __restrict__ dK_pre, float* __restrict__ dAngles,
    float* dScale, float* dGamma, const T_ACT* __restrict__ Q_raw,
    const T_ACT* __restrict__ K_raw, const float* Scale_in, const float* Gamma_in,
    const float* __restrict__ Angles, const float* __restrict__ dQ_mid,
    const float* __restrict__ dK_mid, const float* __restrict__ dQK_dot, IDX row, int ds,
    int n_angles
) {
    // dScale and dGamma may alias Scale_in and Gamma_in: the row reads both
    // before it stores either.
    const float gamma = Gamma_in[row];
    const float dqk = dQK_dot[row];
    M3DqktRow r;
    r.angles = Angles + row * n_angles;
    r.d_angles = dAngles + row * n_angles;
    r.n_angles = n_angles;
    r.scale = Scale_in[row];
    r.dg = __fmul_rn(gamma, dqk);
    r.d_scale = 0.0f;
    r.qk = 0.0f;
    const IDX base = row * ds;
    // The pointers and ds decide the loads for every row of the launch
    // alike, and vector and element loads read the same values.
    const bool vec = (ds & 3) == 0 && M3_QUAD_ALIGNED(Q_raw, T_ACT)
        && M3_QUAD_ALIGNED(K_raw, T_ACT) && M3_QUAD_ALIGNED(dQ_mid, float)
        && M3_QUAD_ALIGNED(dK_mid, float) && M3_QUAD_ALIGNED(dQ_pre, float)
        && M3_QUAD_ALIGNED(dK_pre, float);
    for (int n = 0; n < ds; n += 4) {
        const int valid = min(4, ds - n);
        const float4 q = m3_load_quad(Q_raw + base + n, valid, vec);
        const float4 k = m3_load_quad(K_raw + base + n, valid, vec);
        const float4 dq = m3_load_quad(dQ_mid + base + n, valid, vec);
        const float4 dk = m3_load_quad(dK_mid + base + n, valid, vec);
        float4 oq = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 ok = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        // A row of odd length ends in one plain state: 2 n_angles <= ds.
        if (valid >= 2) {
            m3_dqkt_two(r, n >> 1, q.x, q.y, k.x, k.y, dq.x, dq.y, dk.x, dk.y, oq.x, oq.y,
                        ok.x, ok.y);
        } else {
            m3_dqkt_plain(r, q.x, k.x, dq.x, dk.x, oq.x, ok.x);
        }
        if (valid == 4) {
            m3_dqkt_two(r, (n >> 1) + 1, q.z, q.w, k.z, k.w, dq.z, dq.w, dk.z, dk.w, oq.z,
                        oq.w, ok.z, ok.w);
        } else if (valid == 3) {
            m3_dqkt_plain(r, q.z, k.z, dq.z, dk.z, oq.z, ok.z);
        }
        m3_store_quad(dQ_pre + base + n, oq, valid, vec);
        m3_store_quad(dK_pre + base + n, ok, valid, vec);
    }
    dScale[row] = r.d_scale;
    dGamma[row] = __fmul_rn(dqk, r.qk);
}

// rows = B * T * nh, a thread a row.
#define DEFINE_M3_DQKTHETA(NAME, IDX, T_ACT)                                   \
extern "C" __global__ __launch_bounds__(256) void NAME(                        \
    float* __restrict__ dQ_pre,          /* [rows * ds] pre-RoPE Q gradient */ \
    float* __restrict__ dK_pre,          /* [rows * ds] pre-RoPE K gradient */ \
    float* __restrict__ dAngles_cumsum,  /* [rows * n_angles] */               \
    float* dScale,                       /* [rows] (may alias Scale_in) */     \
    float* dGamma,                       /* [rows] (may alias Gamma_in) */     \
    const T_ACT* __restrict__ Q_raw,     /* [rows * ds] acts.c_biased */       \
    const T_ACT* __restrict__ K_raw,     /* [rows * ds] acts.b_biased */       \
    const float* Scale_in,               /* [rows] (may alias dScale) */       \
    const float* Gamma_in,               /* [rows] (may alias dGamma) */       \
    const float* __restrict__ Angles,    /* [rows * n_angles] */               \
    const float* __restrict__ dQ_mid,    /* [rows * ds] from m3_dqkv */        \
    const float* __restrict__ dK_mid,    /* [rows * ds] from m3_dqkv */        \
    const float* __restrict__ dQK_dot,   /* [rows] from m3_dqkv */             \
    int rows, int ds, int n_angles                                             \
) {                                                                            \
    const IDX row = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                \
    if (row >= rows) return;                                                   \
    m3_dqktheta_row<IDX, T_ACT>(dQ_pre, dK_pre, dAngles_cumsum, dScale,        \
        dGamma, Q_raw, K_raw, Scale_in, Gamma_in, Angles, dQ_mid, dK_mid,      \
        dQK_dot, row, ds, n_angles);                                           \
}

DEFINE_M3_DQKTHETA(m3_dqktheta, int, float)
DEFINE_M3_DQKTHETA(m3_dqktheta_i64, long long, float)
DEFINE_M3_DQKTHETA(m3_dqktheta_bf16, int, __nv_bfloat16)
DEFINE_M3_DQKTHETA(m3_dqktheta_bf16_i64, long long, __nv_bfloat16)
DEFINE_M3_DQKTHETA(m3_dqktheta_f16, int, __half)
DEFINE_M3_DQKTHETA(m3_dqktheta_f16_i64, long long, __half)
#undef DEFINE_M3_DQKTHETA

// ============================================================================
// 13. m3_ddt_dtrap -- dScale/dGamma -> dDT, dTrap (reference backward stage 4)
// ============================================================================
// Grid: (nh, B), Block: (1, 1, 1) — one thread per (head, batch), loops over T
// dDT[t] = (dGamma[t] + dScale[t]) * trap[t] + dScale[t-1] * (1 - trap[t])
// dTrap_presig[t] = ((dGamma[t] + dScale[t]) * dt[t] - dScale[t-1] * dt[t]) * sig*(1-sig)
// Grid: ceil(B*T*nh / 256), Block: 256
// Each thread handles one (b,t,h) element. Reads dScale[t-1] via shifted
// access; at t = 0 the step before is the previous window's last, whose
// scale gradient is the entering fold's (m3_enter_fold_bwd), or zero
// without one.
#define DEFINE_M3_DDT_DTRAP(NAME, IDX)                                         \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ dDT,          /* [B*T*nh] */                           \
    float* __restrict__ dTrap_presig, /* [B*T*nh] */                           \
    const float* __restrict__ dScale, /* [B*T*nh] */                           \
    const float* __restrict__ dGamma, /* [B*T*nh] */                           \
    const float* __restrict__ DT,     /* [B*T*nh] — post-softplus dt */        \
    const float* __restrict__ Trap,   /* [B*T*nh] — post-sigmoid trap */       \
    int B, int T, int nh,                                                      \
    const float* __restrict__ dScale_enter, /* [B*nh] or nullptr */            \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    IDX idx = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                      \
    IDX total = (IDX)B * T * nh;                                               \
    if (idx >= total) return;                                                  \
                                                                               \
    /* Decompose flat index: idx = (b * T + t) * nh + h */                     \
    int t = (int)((idx / nh) % T);                                             \
                                                                               \
    float ds_cur = dScale[idx];                                                \
    float dg_cur = dGamma[idx];                                                \
    float dt_val = DT[idx];                                                    \
    float trap_val = Trap[idx];                                                \
                                                                               \
    /* dScale from previous timestep (shifted read; the entering fold's */    \
    /* at t = 0) */                                                            \
    float ds_prev = 0.0f;                                                      \
    if (t > 0) ds_prev = dScale[idx - nh]; /* (b*T + t-1)*nh + h = idx - nh */ \
    else if (dScale_enter != nullptr)                                          \
        ds_prev = dScale_enter[(idx / nh / T) * nh + idx % nh];               \
                                                                               \
    /* A token that starts a sequence of a packed row has no beta: the */      \
    /* step before it took no shifted gamma from it. */                        \
    if (vl_starts(seq_idx, idx / nh / T, T, t)) ds_prev = 0.0f;                \
    float d_dt = (dg_cur + ds_cur) * trap_val + ds_prev * (1.0f - trap_val);   \
                                                                               \
    float d_trap_sig = ((dg_cur + ds_cur) - ds_prev) * dt_val;                 \
    float d_trap_raw = d_trap_sig * trap_val * (1.0f - trap_val);              \
                                                                               \
    dDT[idx] = d_dt;                                                           \
    dTrap_presig[idx] = d_trap_raw;                                            \
}

DEFINE_M3_DDT_DTRAP(m3_ddt_dtrap, int)
DEFINE_M3_DDT_DTRAP(m3_ddt_dtrap_i64, long long)

// ============================================================================
// 14. m3_final_grads -- Combine dADT + dDT + dDT_angle -> d_dd_dt, d_dd_a
// ============================================================================
// Grid: ceil(N/256), Block: 256
// Replaces m3_abg_bwd for the parallel path.
#define DEFINE_M3_FINAL_GRADS(NAME, IDX)                                       \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ d_dd_dt_raw,   /* [B*T*nh] */                          \
    float* __restrict__ d_dd_a_raw,    /* [B*T*nh] */                          \
    const float* __restrict__ dADT,    /* [B*T*nh] — from m3_dqkv */           \
    const float* __restrict__ dDT,     /* [B*T*nh] — from m3_ddt_dtrap */      \
    const float* __restrict__ dDT_angle, /* [B*T*nh] — from angle_dt_bwd */    \
    const float* __restrict__ a_val,   /* [B*T*nh] — saved activation */       \
    const float* __restrict__ DT,      /* [B*T*nh] — saved post-softplus dt */ \
    const float* __restrict__ dd_dt_raw_saved, /* [B*T*nh] — saved pre-softplus dd_dt (NO bias) */ \
    const float* __restrict__ dd_a_raw_saved,  /* [B*T*nh] — saved pre-softplus dd_a */ \
    const float* __restrict__ dt_bias,         /* [nh] — bias added before softplus */ \
    float a_floor,                                                             \
    IDX N, /* = B*T*nh */                                                      \
    int nh                                                                     \
) {                                                                            \
    IDX i = (IDX)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (i >= N) return;                                                        \
                                                                               \
    int h = (int)(i % nh);                                                     \
    float d_adt = dADT[i];                                                     \
    float d_dt_trap = dDT[i];                                                  \
    float d_dt_angle_val = dDT_angle[i];                                       \
                                                                               \
    float dt_val = DT[i];                                                      \
    float a_v = a_val[i];                                                      \
    float d_a_val = d_adt * dt_val;                                            \
    float d_dt_adt = d_adt * a_v;                                              \
                                                                               \
    float d_dt_total = d_dt_trap + d_dt_angle_val + d_dt_adt;                  \
                                                                               \
    /* softplus backward: sigmoid(dd_dt_raw + dt_bias) — must include bias! */ \
    float raw_dt = dd_dt_raw_saved[i] + dt_bias[h];                            \
    float sig_dt = (raw_dt > 20.0f) ? 1.0f : 1.0f / (1.0f + exp2f(-raw_dt * LOG2E)); /* Softplus asymptote guard - matches m3_dqkv sp_deriv_dt (mamba3_ops.cu) and the CPU form; without it the chunked backward disagreed with both. */ \
    d_dd_dt_raw[i] = d_dt_total * sig_dt;                                      \
                                                                               \
    /* d_dd_a_raw: from d_a_val, through clamp and -heavy_tail */              \
    float raw_a = dd_a_raw_saved[i];                                           \
    float a_unclamped = -m3c_heavy_tail(raw_a);                                \
    /* a_val = min(-heavy_tail(raw), -a_floor) = clamp(-, max=-a_floor) */     \
    /* If the clamp was active: gradient = 0. */                               \
    if (a_unclamped > -a_floor) {                                              \
        d_dd_a_raw[i] = 0.0f; /* clamped, no gradient */                       \
    } else {                                                                   \
        /* -heavy_tail backward: d_raw = d_a_val * (-heavy_tail'(raw)) */      \
        d_dd_a_raw[i] = d_a_val * (-m3c_heavy_tail_deriv(raw_a));              \
    }                                                                          \
}

DEFINE_M3_FINAL_GRADS(m3_final_grads, int)
DEFINE_M3_FINAL_GRADS(m3_final_grads_i64, long long)

// ============================================================================
// Typed (bf16/f16) variants of the chunked parallel forward kernels.
//
// Typed I/O: activation tensors (K, Q, K_scaled, x/V, y_out, k_flat, x_flat)
// are T_ACT in storage. All math + state remain f32 (BPTT states, dA_cumsum,
// prev_states, ssm/k/v persistent state, DT/trap_sig/qk_dot/scale/gamma/D).
//
// The decay cumsum the pre-state kernels write stays f32: its O(T)
// compounding scan state mandates float for numerical safety. The state
// pass reads typed K and V for the boundary states only; its inter-chunk
// recurrence is f32 too.
// ============================================================================



/* Typed twin of m3_chunk_pre_state_fused. BIT CONTRACT SUBTLETY: the
 * standalone chunk_state reads K_scaled AFTER its typed round trip
 * (preprocess stores T_ACT, chunk_state widens it back), so the fused
 * fold must consume to_f(FROM_F(ks)) - the round-tripped value - never
 * the pre-round f32. x stages through to_f exactly as the standalone
 * reads it. */
#define DEFINE_M3_CHUNK_PRE_STATE_FUSED(NAME, IDX, T_ACT, FROM_F)             \
extern "C" __global__ void NAME(                                              \
    T_ACT* __restrict__ K_scaled,                                             \
    float* __restrict__ qk_dot,                                               \
    float* __restrict__ scale_out,                                            \
    float* __restrict__ gamma_out,                                            \
    float* __restrict__ states_out,                                           \
    const T_ACT* __restrict__ K,                                              \
    const T_ACT* __restrict__ Q,                                              \
    const float* __restrict__ DT,                                             \
    const float* __restrict__ trap_sig,                                       \
    const T_ACT* __restrict__ x,                                              \
    const float* __restrict__ adt,                                            \
    float* __restrict__ dA_cumsum,                                            \
    int batch, int T, int nh, int hd, int ds, int chunk_size,                 \
    int y_first, /* this run's first y block */                               \
    const int* __restrict__ seq_idx                                           \
) {                                                                           \
    int d_inner = nh * hd;                                                    \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                         \
    IDX bc = blockIdx.x;                                                      \
    IDX b = bc / n_chunks;                                                    \
    int chunk = (int)(bc % n_chunks);                                         \
    int h = y_first + blockIdx.y;                                             \
    if (ds > MAMBA_RS_STATE_CAP) return;                                      \
    int chunk_start = chunk * chunk_size;                                     \
    int chunk_end = chunk_start + chunk_size;                                 \
    if (chunk_end > T) chunk_end = T;                                         \
    int chunk_len = chunk_end - chunk_start;                                  \
    /* In a packed row (seq_idx) only the chunk's last sequence reaches */    \
    /* its state: the tokens from where that sequence begins. */              \
    int kept = vl_seq_first(seq_idx, b, T, chunk_start, chunk_end - 1) - chunk_start; \
    extern __shared__ float m3f_sm[];                                         \
    int k_ld = ds + 4;                                                        \
    float* sm_k = m3f_sm;                                                     \
    float* sm_x = sm_k + chunk_size * k_ld;                                   \
    float* sm_dec = sm_x + chunk_size * hd;                                   \
    float* da_chunk = sm_dec + chunk_size;                                    \
    int tid = threadIdx.x;                                                    \
    int nt = blockDim.x;                                                      \
    /* The chunk's decay cumsum and each row's decay - see the f32 twin. */   \
    for (int t_local = tid; t_local < chunk_len; t_local += nt) {             \
        da_chunk[t_local] = adt[(b * T + chunk_start + t_local) * nh + h];    \
    }                                                                         \
    __syncthreads();                                                          \
    if (tid == 0) {                                                           \
        float cumsum = 0.0f;                                                  \
        IDX cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;         \
        for (int t_local = 0; t_local < chunk_len; t_local++) {               \
            const bool fresh = vl_starts(seq_idx, b, T, chunk_start + t_local); \
            cumsum = (fresh ? 0.0f : cumsum) + da_chunk[t_local];             \
            da_chunk[t_local] = cumsum;                                       \
            dA_cumsum[cs_base + t_local] = cumsum;                            \
        }                                                                     \
    }                                                                         \
    __syncthreads();                                                          \
    float dA_end = da_chunk[chunk_len - 1];                                   \
    for (int t_local = tid; t_local < chunk_len; t_local += nt) {             \
        int t = chunk_start + t_local;                                        \
        IDX th = (b * T + t) * nh + h;                                        \
        float dt_cur = DT[th];                                                \
        float trap_cur = trap_sig[th];                                        \
        float gamma_val = dt_cur * trap_cur;                                  \
        float shifted_gamma = 0.0f;                                           \
        /* The next step's beta is 0 where it starts a sequence. */           \
        if (t + 1 < T && !vl_starts(seq_idx, b, T, t + 1)) {                  \
            IDX th_next = (b * T + t + 1) * nh + h;                           \
            float dt_next = DT[th_next];                                      \
            float trap_next = trap_sig[th_next];                              \
            shifted_gamma = dt_next * (1.0f - trap_next);                     \
        }                                                                     \
        float scale_val = shifted_gamma + gamma_val;                          \
        if (scale_out) scale_out[th] = scale_val;                             \
        if (gamma_out) gamma_out[th] = gamma_val;                             \
        IDX kq_base = (b * T + t) * nh * ds + h * ds;                         \
        float dot = 0.0f;                                                     \
        if (ds % 8 == 0) {                                                    \
            for (int n8 = 0; n8 < ds / 8; n8++) {                             \
                union { uint4 u; T_ACT e[8]; } qq, kk, ss;                    \
                qq.u = *reinterpret_cast<const uint4*>(Q + kq_base + n8 * 8); \
                kk.u = *reinterpret_cast<const uint4*>(K + kq_base + n8 * 8); \
                for (int j = 0; j < 8; j++) {                                 \
                    dot += to_f(qq.e[j]) * to_f(kk.e[j]);                     \
                }                                                             \
                for (int j = 0; j < 8; j++) {                                 \
                    T_ACT r = FROM_F(to_f(kk.e[j]) * scale_val);              \
                    ss.e[j] = r;                                              \
                    sm_k[t_local * k_ld + n8 * 8 + j] = to_f(r);                \
                }                                                             \
                *reinterpret_cast<uint4*>(K_scaled + kq_base + n8 * 8) =      \
                    ss.u;                                                     \
            }                                                                 \
        } else {                                                              \
            for (int n = 0; n < ds; n++) {                                    \
                dot += to_f(Q[kq_base + n]) * to_f(K[kq_base + n]);           \
            }                                                                 \
            for (int n = 0; n < ds; n++) {                                    \
                T_ACT r = FROM_F(to_f(K[kq_base + n]) * scale_val);           \
                K_scaled[kq_base + n] = r;                                    \
                sm_k[t_local * k_ld + n] = to_f(r);                             \
            }                                                                 \
        }                                                                     \
        qk_dot[th] = dot * gamma_val;                                         \
        sm_dec[t_local] = FAST_EXP(m3_nonpos(dA_end - da_chunk[t_local]));    \
    }                                                                         \
    if (hd % 4 == 0 && (reinterpret_cast<unsigned long long>(x) & 7) == 0) {  \
        /* Four elements a load when every row starts on 8 bytes, each */     \
        /* widened by to_f exactly as the element-wise copy widens it. */     \
        int hq = hd >> 2;                                                     \
        for (int i = tid; i < chunk_len * hq; i += nt) {                      \
            int tl = i / hq;                                                  \
            int pq = i - tl * hq;                                             \
            union { uint2 u; T_ACT e[4]; } xx;                                \
            xx.u = __ldg(reinterpret_cast<const uint2*>(                      \
                x + (long long)(b * T + chunk_start + tl) * d_inner           \
                  + h * hd + 4 * pq));                                        \
            float4 xf;                                                        \
            xf.x = to_f(xx.e[0]);                                             \
            xf.y = to_f(xx.e[1]);                                             \
            xf.z = to_f(xx.e[2]);                                             \
            xf.w = to_f(xx.e[3]);                                             \
            reinterpret_cast<float4*>(sm_x)[i] = xf;                          \
        }                                                                     \
    } else {                                                                  \
        for (int i = tid; i < chunk_len * hd; i += nt) {                      \
            int tl = i / hd;                                                  \
            int pp = i % hd;                                                  \
            sm_x[tl * hd + pp] = to_f(                                        \
                __ldg(x + (long long)(b * T + chunk_start + tl) * d_inner     \
                  + h * hd + pp));                                            \
        }                                                                     \
    }                                                                         \
    __syncthreads();                                                          \
    int quads = ds / 4;                                                       \
    for (int o = tid; o < hd * quads; o += nt) {                              \
        int p = o / quads;                                                    \
        int nq = o % quads;                                                   \
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;             \
        /* Four steps' reads together ahead of their arithmetic; each chain */ \
        /* takes the same products in the same order as one step at a time. */ \
        int tr = kept;                                                         \
        for (; tr + 4 <= chunk_len; tr += 4) {                                 \
            float dec4[4], vt4[4];                                             \
            float4 kq4[4];                                                     \
            lds_f32_row4(sm_dec + tr, dec4);                                   \
            lds_f32_rows4(sm_x + tr * hd + p, hd, vt4);                        \
            lds_f32x4_rows4(sm_k + tr * k_ld + 4 * nq, k_ld, kq4);             \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 4; j++) {                                      \
                acc0 += dec4[j] * kq4[j].x * vt4[j];                           \
                acc1 += dec4[j] * kq4[j].y * vt4[j];                           \
                acc2 += dec4[j] * kq4[j].z * vt4[j];                           \
                acc3 += dec4[j] * kq4[j].w * vt4[j];                           \
            }                                                                  \
        }                                                                      \
        for (int t_local = tr; t_local < chunk_len; t_local++) {            \
            float decay = sm_dec[t_local];                                    \
            float v_t = sm_x[t_local * hd + p];                               \
            const float* k_row = sm_k + t_local * k_ld;                       \
            float4 kv = lds_f32x4(k_row + 4 * nq);                            \
            acc0 += decay * kv.x * v_t;                                       \
            acc1 += decay * kv.y * v_t;                                       \
            acc2 += decay * kv.z * v_t;                                       \
            acc3 += decay * kv.w * v_t;                                       \
        }                                                                     \
        IDX state_base =                                                      \
            ((b * n_chunks + chunk) * nh + h) * hd * ds + p * ds + nq * 4;    \
        float4 out;                                                           \
        out.x = acc0;                                                         \
        out.y = acc1;                                                         \
        out.z = acc2;                                                         \
        out.w = acc3;                                                         \
        *reinterpret_cast<float4*>(states_out + state_base) = out;            \
    }                                                                         \
}

DEFINE_M3_CHUNK_PRE_STATE_FUSED(m3_chunk_pre_state_fused_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_M3_CHUNK_PRE_STATE_FUSED(m3_chunk_pre_state_fused_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_M3_CHUNK_PRE_STATE_FUSED(m3_chunk_pre_state_fused_f16, int, __half, from_f_f16)
DEFINE_M3_CHUNK_PRE_STATE_FUSED(m3_chunk_pre_state_fused_f16_i64, long long, __half, from_f_f16)

// ============================================================================
// 3c. m3_chunk_pre_state_sliced -- the pre-state for any head and state width
// ============================================================================
//
// The fused pre-state kernel's outputs, bit for bit, and the unfused trio's
// where that kernel cannot run, from a tile that depends on neither the head
// nor the state width:
// - the decay cumsum, scale, gamma and each row's decay to the chunk's end
//   are the fused kernel's expressions, its packed-row restarts included;
// - each row's Q.K dot is one thread's chain over the state, n ascending
//   from +0, its operands staged M3PS_NS states at a time;
// - each scaled K value is rounded once to the activation dtype, as stored,
//   and the state fold reads the stored value;
// - each chunk state value is one thread's chain over the rows of the
//   chunk's last sequence in order, the fused kernel's terms.
// A block takes one (sequence, chunk, head) and M3PS_PB channels of the
// head; a wider head is split across blocks (blockIdx.z), each forming the
// rows' values again, since no forward sum runs over the head's channels.
// The first channel block alone writes what is not per channel: the cumsum,
// scale, gamma, the scaled K rows and the dots.
//
// The tile depends on neither the head nor the state width: M3PS_FLOATS
// floats, 27 KB, so three blocks share an SM on every supported part.
// Grid: (B * n_chunks, nh, ceil(hd / M3PS_PB)). Block: 256.
#define M3PS_NS 16
#define M3PS_PB 64
#define M3PS_CS 64
#define M3PS_LD (M3PS_NS + 4)
#define M3PS_FLOATS (2 * M3PS_CS * M3PS_LD + M3PS_CS * M3PS_PB + 4 * M3PS_CS)

#define DEFINE_M3_CHUNK_PRE_STATE_SLICED(NAME, IDX, T_ACT, FROM_F)            \
extern "C" __global__ __launch_bounds__(256, 3) void NAME(                    \
    T_ACT* __restrict__ K_scaled,                                             \
    float* __restrict__ qk_dot,                                               \
    float* __restrict__ scale_out,                                            \
    float* __restrict__ gamma_out,                                            \
    float* __restrict__ states_out,                                           \
    const T_ACT* __restrict__ K,                                              \
    const T_ACT* __restrict__ Q,                                              \
    const float* __restrict__ DT,                                             \
    const float* __restrict__ trap_sig,                                       \
    const T_ACT* __restrict__ x,                                              \
    const float* __restrict__ adt,       /* [B * T * nh] a*dt */              \
    float* __restrict__ dA_cumsum,       /* [B * n_chunks * nh * chunk_size] out */ \
    int batch, int T, int nh, int hd, int ds, int chunk_size,                 \
    int y_first, /* this run's first y block */                               \
    const int* __restrict__ seq_idx                                           \
) {                                                                           \
    int d_inner = nh * hd;                                                    \
    int n_chunks = (T + chunk_size - 1) / chunk_size;                         \
    IDX bc = blockIdx.x;                                                      \
    IDX b = bc / n_chunks;                                                    \
    int chunk = (int)(bc % n_chunks);                                         \
    int h = y_first + blockIdx.y;                                             \
    int p_lo = blockIdx.z * M3PS_PB;                                          \
    if (chunk_size > M3PS_CS) return;                                         \
    bool lead = blockIdx.z == 0;                                              \
    int chunk_start = chunk * chunk_size;                                     \
    int chunk_end = chunk_start + chunk_size;                                 \
    if (chunk_end > T) chunk_end = T;                                         \
    int chunk_len = chunk_end - chunk_start;                                  \
    int p_len = min(hd - p_lo, M3PS_PB);                                      \
    /* In a packed row (seq_idx) only the chunk's last sequence reaches */    \
    /* its state: the tokens from where that sequence begins. */              \
    int kept = vl_seq_first(seq_idx, b, T, chunk_start, chunk_end - 1) - chunk_start; \
    IDX cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;             \
                                                                              \
    extern __shared__ __align__(16) float m3ps_sm[];                          \
    float* sm_k = m3ps_sm;                         /* [M3PS_CS][M3PS_LD] */   \
    float* sm_q = sm_k + M3PS_CS * M3PS_LD;        /* [M3PS_CS][M3PS_LD] */   \
    float* sm_x = sm_q + M3PS_CS * M3PS_LD;        /* [M3PS_CS][M3PS_PB] */   \
    float* sm_da = sm_x + M3PS_CS * M3PS_PB;       /* [M3PS_CS] */            \
    float* sm_gam = sm_da + M3PS_CS;               /* [M3PS_CS] */            \
    float* sm_dec = sm_gam + M3PS_CS;              /* [M3PS_CS] */            \
    float* sm_scale = sm_dec + M3PS_CS;            /* [M3PS_CS] */            \
    int tid = threadIdx.x;                                                    \
    int nt = blockDim.x;                                                      \
                                                                              \
    /* Each row's own values, the fused kernel's expressions; the lead */     \
    /* block writes the scale and gamma rows. */                              \
    for (int t_local = tid; t_local < chunk_len; t_local += nt) {             \
        int t = chunk_start + t_local;                                        \
        IDX th = (b * T + t) * nh + h;                                        \
        float dt_cur = DT[th];                                                \
        float trap_cur = trap_sig[th];                                        \
        float gamma_val = __fmul_rn(dt_cur, trap_cur);                        \
        float shifted_gamma = 0.0f;                                           \
        /* The next step's beta is 0 where it starts a sequence. */           \
        if (t + 1 < T && !vl_starts(seq_idx, b, T, t + 1)) {                  \
            IDX th_next = (b * T + t + 1) * nh + h;                           \
            float dt_next = DT[th_next];                                      \
            float trap_next = trap_sig[th_next];                              \
            shifted_gamma = __fmul_rn(dt_next, 1.0f - trap_next);             \
        }                                                                     \
        float scale_val = __fadd_rn(shifted_gamma, gamma_val);                \
        if (lead && scale_out) scale_out[th] = scale_val;                     \
        if (lead && gamma_out) gamma_out[th] = gamma_val;                     \
        sm_scale[t_local] = scale_val;                                        \
        sm_gam[t_local] = gamma_val;                                          \
        sm_da[t_local] = adt[th];                                             \
    }                                                                         \
    /* The block's channels of the x rows. */                                 \
    bool vec_x = (hd & 3) == 0 && M3_QUAD_ALIGNED(x, T_ACT);                     \
    for (int i = tid; i < chunk_len * (M3PS_PB / 4); i += nt) {               \
        int tl = i / (M3PS_PB / 4);                                           \
        int pq = i - tl * (M3PS_PB / 4);                                      \
        int valid = p_len - 4 * pq;                                           \
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);                       \
        if (valid > 0) {                                                      \
            v = m3_load_quad(x + (b * T + chunk_start + tl) * d_inner + h * hd   \
                              + p_lo + 4 * pq,                                \
                          valid, vec_x);                                      \
        }                                                                     \
        *reinterpret_cast<float4*>(sm_x + tl * M3PS_PB + 4 * pq) = v;         \
    }                                                                         \
    __syncthreads();                                                          \
    /* The chunk's decay cumsum, the fused kernel's adds from zero in row */  \
    /* order, from zero again where a packed row starts a sequence. */        \
    if (tid == 0) {                                                           \
        float cumsum = 0.0f;                                                  \
        for (int t_local = 0; t_local < chunk_len; t_local++) {               \
            bool fresh = vl_starts(seq_idx, b, T, chunk_start + t_local);     \
            cumsum = __fadd_rn(fresh ? 0.0f : cumsum, sm_da[t_local]);        \
            sm_da[t_local] = cumsum;                                          \
            if (lead) dA_cumsum[cs_base + t_local] = cumsum;                  \
        }                                                                     \
    }                                                                         \
    __syncthreads();                                                          \
    float dA_end = sm_da[chunk_len - 1];                                      \
    for (int t_local = tid; t_local < chunk_len; t_local += nt) {             \
        sm_dec[t_local] = FAST_EXP(m3_nonpos(dA_end - sm_da[t_local]));       \
    }                                                                         \
                                                                              \
    bool vec = (ds & 3) == 0 && M3_QUAD_ALIGNED(K, T_ACT) && M3_QUAD_ALIGNED(Q, T_ACT); \
    bool vec_ks = (ds & 3) == 0 && M3_QUAD_ALIGNED(K_scaled, T_ACT);             \
    bool vec_st = (ds & 3) == 0 && M3_QUAD_ALIGNED(states_out, float);           \
    IDX state_head = ((b * n_chunks + chunk) * nh + h) * hd * ds;             \
    /* Row tid's Q.K chain in the lead block, carried across the slices. */   \
    float dot = 0.0f;                                                         \
    for (int n0 = 0; n0 < ds; n0 += M3PS_NS) {                                \
        int width = min(ds - n0, M3PS_NS);                                    \
        int full = width >> 2;                                                \
        int row = tid >> 2;                                                   \
        int qd = tid & 3;                                                     \
        int valid = width - 4 * qd;                                           \
        IDX base = ((b * T + chunk_start + row) * nh + h) * ds + n0 + 4 * qd; \
        __syncthreads();                                                      \
        {                                                                     \
            /* Four values a thread: row tid / 4, quad tid % 4. */            \
            float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);                  \
            float4 qv = kv;                                                   \
            if (valid > 0 && row < chunk_len) {                               \
                kv = m3_load_quad(K + base, valid, vec);                         \
                if (lead) qv = m3_load_quad(Q + base, valid, vec);               \
            }                                                                 \
            *reinterpret_cast<float4*>(sm_k + row * M3PS_LD + 4 * qd) = kv;   \
            *reinterpret_cast<float4*>(sm_q + row * M3PS_LD + 4 * qd) = qv;   \
        }                                                                     \
        __syncthreads();                                                      \
        /* The dot's terms over the slice, from the rows before K is */       \
        /* scaled. */                                                         \
        if (lead && tid < chunk_len) {                                        \
            const float* qr = sm_q + tid * M3PS_LD;                           \
            const float* kr = sm_k + tid * M3PS_LD;                           \
            for (int q4i = 0; q4i < full; q4i++) {                            \
                float4 a = lds_f32x4(qr + 4 * q4i);                           \
                float4 c = lds_f32x4(kr + 4 * q4i);                           \
                dot = __fmaf_rn(a.x, c.x, dot);                               \
                dot = __fmaf_rn(a.y, c.y, dot);                               \
                dot = __fmaf_rn(a.z, c.z, dot);                               \
                dot = __fmaf_rn(a.w, c.w, dot);                               \
            }                                                                 \
            for (int n = 4 * full; n < width; n++) {                          \
                dot = __fmaf_rn(qr[n], kr[n], dot);                           \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
        /* The slice of K scaled in place, each value rounded once to the */  \
        /* stored dtype; the lead block writes the rows. */                   \
        if (valid > 0 && row < chunk_len) {                                   \
            float scale_val = sm_scale[row];                                  \
            float4* sk = reinterpret_cast<float4*>(sm_k + row * M3PS_LD + 4 * qd); \
            float4 kv = lds_f32x4(sm_k + row * M3PS_LD + 4 * qd);             \
            float4 sc;                                                        \
            sc.x = to_f(FROM_F(__fmul_rn(kv.x, scale_val)));                  \
            sc.y = to_f(FROM_F(__fmul_rn(kv.y, scale_val)));                  \
            sc.z = to_f(FROM_F(__fmul_rn(kv.z, scale_val)));                  \
            sc.w = to_f(FROM_F(__fmul_rn(kv.w, scale_val)));                  \
            if (lead) m3_store_quad(K_scaled + base, sc, valid, vec_ks);           \
            *sk = sc;                                                         \
        }                                                                     \
        __syncthreads();                                                      \
        /* The block's chunk state values over the slice: each one */         \
        /* thread's chain over the rows of the chunk's last sequence. */      \
        for (int o = tid; o < p_len * 4; o += nt) {                           \
            int pl = o >> 2;                                                  \
            int oq = o & 3;                                                   \
            int ovalid = width - 4 * oq;                                      \
            if (ovalid <= 0) continue;                                        \
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;         \
            /* Four steps' reads together ahead of their arithmetic, all of them */ \
            /* through PTX; each chain takes the same products in the same order. */ \
            int tr = kept;                                                     \
            for (; tr + 4 <= chunk_len; tr += 4) {                             \
                float dec4[4], vt4[4];                                         \
                float4 kq4[4];                                                 \
                lds_f32_row4(sm_dec + tr, dec4);                               \
                lds_f32_rows4(sm_x + tr * M3PS_PB + pl, M3PS_PB, vt4);         \
                lds_f32x4_rows4(sm_k + tr * M3PS_LD + 4 * oq, M3PS_LD, kq4);   \
                _Pragma("unroll")                                              \
                for (int j = 0; j < 4; j++) {                                  \
                    acc0 = __fmaf_rn(__fmul_rn(dec4[j], kq4[j].x), vt4[j], acc0); \
                    acc1 = __fmaf_rn(__fmul_rn(dec4[j], kq4[j].y), vt4[j], acc1); \
                    acc2 = __fmaf_rn(__fmul_rn(dec4[j], kq4[j].z), vt4[j], acc2); \
                    acc3 = __fmaf_rn(__fmul_rn(dec4[j], kq4[j].w), vt4[j], acc3); \
                }                                                              \
            }                                                                  \
            for (int t_local = tr; t_local < chunk_len; t_local++) {          \
                float decay = sm_dec[t_local];                                \
                float v_t = sm_x[t_local * M3PS_PB + pl];                     \
                float4 kv = lds_f32x4(sm_k + t_local * M3PS_LD + 4 * oq);     \
                acc0 = __fmaf_rn(__fmul_rn(decay, kv.x), v_t, acc0);          \
                acc1 = __fmaf_rn(__fmul_rn(decay, kv.y), v_t, acc1);          \
                acc2 = __fmaf_rn(__fmul_rn(decay, kv.z), v_t, acc2);          \
                acc3 = __fmaf_rn(__fmul_rn(decay, kv.w), v_t, acc3);          \
            }                                                                 \
            m3_store_quad(states_out + state_head + (IDX)(p_lo + pl) * ds + n0 + 4 * oq, \
                     make_float4(acc0, acc1, acc2, acc3), ovalid, vec_st);    \
        }                                                                     \
    }                                                                         \
    if (lead && tid < chunk_len) {                                            \
        qk_dot[(b * T + chunk_start + tid) * nh + h] = __fmul_rn(dot, sm_gam[tid]); \
    }                                                                         \
}

DEFINE_M3_CHUNK_PRE_STATE_SLICED(m3_chunk_pre_state_sliced, int, float, from_f_f32)
DEFINE_M3_CHUNK_PRE_STATE_SLICED(m3_chunk_pre_state_sliced_i64, long long, float, from_f_f32)
DEFINE_M3_CHUNK_PRE_STATE_SLICED(m3_chunk_pre_state_sliced_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_M3_CHUNK_PRE_STATE_SLICED(m3_chunk_pre_state_sliced_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_M3_CHUNK_PRE_STATE_SLICED(m3_chunk_pre_state_sliced_f16, int, __half, from_f_f16)
DEFINE_M3_CHUNK_PRE_STATE_SLICED(m3_chunk_pre_state_sliced_f16_i64, long long, __half, from_f_f16)
#undef DEFINE_M3_CHUNK_PRE_STATE_SLICED


/* Typed twin of m3_chunk_scan_fwd_coop (see the f32 kernel for the
 * distribution + smem layout). Activations convert through to_f at the
 * staging copy, and y stores through FROM_F. */
#define DEFINE_M3_CHUNK_SCAN_FWD_COOP(NAME, IDX, T_ACT, FROM_F)               \
extern "C" __global__ void NAME(                                              \
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
    int h = y_first + blockIdx.y;                                             \
    if (chunk_size > 64) return;                                              \
    if (ds > MAMBA_RS_STATE_CAP) return;                                      \
    int chunk_start = chunk * chunk_size;                                     \
    int chunk_end = chunk_start + chunk_size;                                 \
    if (chunk_end > T) chunk_end = T;                                         \
    int chunk_len = chunk_end - chunk_start;                                  \
    /* In a packed row (seq_idx) the entering state reaches only the */       \
    /* tokens before the chunk's first start, and a token's pairs stay */     \
    /* inside its own sequence. */                                            \
    int cont = vl_lower_bound(seq_idx, b, T, chunk_start, chunk_end,          \
                              vl_id(seq_idx, b, T, chunk_start - 1) + 1) - chunk_start; \
    IDX cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;             \
    IDX state_head = ((b * n_chunks + chunk) * nh + h) * hd * ds;             \
    extern __shared__ float m3cs_sm[];                                        \
    /* Rows padded to ds+4 floats - see the f32 twin's bank note. */          \
    int qk_ld = ds + 4;                                                       \
    float* sm_tri = m3cs_sm;                                                  \
    float* sm_q   = sm_tri + chunk_size * (chunk_size - 1) / 2;               \
    float* sm_k   = sm_q + chunk_size * qk_ld;                                \
    float* sm_v   = sm_k + chunk_size * qk_ld;                                \
    /* The entering state transposed - see the f32 twin's note. */            \
    int ps_ld = hd + 4;                                                       \
    float* sm_ps  = sm_v + chunk_size * hd;                                   \
    float* sm_da  = sm_ps + ds * ps_ld;                                       \
    float* sm_qkd = sm_da + chunk_size;                                       \
    int tid = threadIdx.x;                                                    \
    int nt = blockDim.x;                                                      \
    for (int i = tid; i < chunk_len * ds; i += nt) {                          \
        int t_local = i / ds;                                                 \
        int n = i % ds;                                                       \
        IDX base = (b * T + chunk_start + t_local) * nh * ds + h * ds;        \
        sm_q[t_local * qk_ld + n] = to_f(Q[base + n]);                           \
        sm_k[t_local * qk_ld + n] = to_f(K_scaled[base + n]);                    \
    }                                                                         \
    for (int i = tid; i < chunk_len * hd; i += nt) {                          \
        int t_local = i / hd;                                                 \
        int pp = i % hd;                                                      \
        sm_v[t_local * hd + pp] =                                             \
            to_f(x[(b * T + chunk_start + t_local) * d_inner + h * hd + pp]); \
    }                                                                         \
    for (int i = tid; i < hd * ds; i += nt) {                                 \
        sm_ps[(i % ds) * ps_ld + (i / ds)] = prev_states[state_head + i];                               \
    }                                                                         \
    for (int i = tid; i < chunk_len; i += nt) {                               \
        sm_da[i] = dA_cumsum[cs_base + i];                                    \
        sm_qkd[i] = qk_dot[(b * T + chunk_start + i) * nh + h];               \
    }                                                                         \
    __syncthreads();                                                          \
    int n_pairs = chunk_len * (chunk_len - 1) / 2;                            \
    /* Wide LDS exactly as the f32 kernel: same serial n-ascending fold, */   \
    /* float4 byte delivery (the smem staging is f32 in both twins). */       \
    bool wide = (ds % 4 == 0) && (hd % 4 == 0);                               \
    for (int e = tid; e < n_pairs; e += nt) {                                 \
        int t_local = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)e)) * 0.5f);   \
        while (t_local * (t_local - 1) / 2 > e) t_local--;                    \
        while ((t_local + 1) * t_local / 2 <= e) t_local++;                   \
        int s_local = e - t_local * (t_local - 1) / 2;                        \
        float decay = FAST_EXP(m3_nonpos(sm_da[t_local] - sm_da[s_local]));   \
        float qk_val = 0.0f;                                                  \
        if (wide) {                                                           \
            const float* q_row = sm_q + t_local * qk_ld;                      \
            const float* k_row = sm_k + s_local * qk_ld;                      \
            for (int n4 = 0; n4 < ds / 4; n4++) {                             \
                float4 qv = lds_f32x4(q_row + 4 * n4);                        \
                float4 kv = lds_f32x4(k_row + 4 * n4);                        \
                qk_val += qv.x * kv.x;                                        \
                qk_val += qv.y * kv.y;                                        \
                qk_val += qv.z * kv.z;                                        \
                qk_val += qv.w * kv.w;                                        \
            }                                                                 \
        } else {                                                              \
            for (int n = 0; n < ds; n++) {                                    \
                qk_val += sm_q[t_local * qk_ld + n] * sm_k[s_local * qk_ld + n];    \
            }                                                                 \
        }                                                                     \
        sm_tri[e] = vl_id(seq_idx, b, T, chunk_start + t_local)               \
                    == vl_id(seq_idx, b, T, chunk_start + s_local)            \
            ? decay * qk_val : 0.0f;                                          \
    }                                                                         \
    __syncthreads();                                                          \
    float d_skip = D[h];                                                      \
    if (wide) {                                                               \
        int quads = hd / 4;                                                   \
        for (int o = tid; o < chunk_len * quads; o += nt) {                   \
            int t_local = o / quads;                                          \
            int p0 = (o % quads) * 4;                                         \
            float state_decay = FAST_EXP(sm_da[t_local]);                     \
            const float* q_row = sm_q + t_local * qk_ld;                      \
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;         \
            for (int n4 = 0; n4 < ds / 4; n4++) {                             \
                float4 qv = lds_f32x4(q_row + 4 * n4);                        \
                const float* row = sm_ps + 4 * n4 * ps_ld + p0;               \
                float4 n0 = lds_f32x4(row);                                   \
                float4 n1 = lds_f32x4(row + ps_ld);                           \
                float4 n2 = lds_f32x4(row + 2 * ps_ld);                       \
                float4 n3 = lds_f32x4(row + 3 * ps_ld);                       \
                acc0 += qv.x * n0.x;                                          \
                acc0 += qv.y * n1.x;                                          \
                acc0 += qv.z * n2.x;                                          \
                acc0 += qv.w * n3.x;                                          \
                acc1 += qv.x * n0.y;                                          \
                acc1 += qv.y * n1.y;                                          \
                acc1 += qv.z * n2.y;                                          \
                acc1 += qv.w * n3.y;                                          \
                acc2 += qv.x * n0.z;                                          \
                acc2 += qv.y * n1.z;                                          \
                acc2 += qv.z * n2.z;                                          \
                acc2 += qv.w * n3.z;                                          \
                acc3 += qv.x * n0.w;                                          \
                acc3 += qv.y * n1.w;                                          \
                acc3 += qv.z * n2.w;                                          \
                acc3 += qv.w * n3.w;                                          \
            }                                                                 \
            bool on = t_local < cont;                                         \
            float y_off0 = (on ? acc0 : 0.0f) * state_decay;                  \
            float y_off1 = (on ? acc1 : 0.0f) * state_decay;                  \
            float y_off2 = (on ? acc2 : 0.0f) * state_decay;                  \
            float y_off3 = (on ? acc3 : 0.0f) * state_decay;                  \
            float yd0 = 0.0f, yd1 = 0.0f, yd2 = 0.0f, yd3 = 0.0f;             \
            int tri_row = t_local * (t_local - 1) / 2;                        \
            int lo = vl_seq_first(seq_idx, b, T, chunk_start, chunk_start + t_local) \
                     - chunk_start;                                           \
            /* Four rows' reads together ahead of their arithmetic; each chain */ \
            /* takes the same products in the same order as one row at a time. */ \
            int sr = lo;                                                       \
            for (; sr + 4 <= t_local; sr += 4) {                               \
                float w4[4];                                                   \
                float4 v4[4];                                                  \
                lds_f32_row4(sm_tri + tri_row + sr, w4);                       \
                lds_f32x4_rows4(sm_v + sr * hd + p0, hd, v4);                  \
                _Pragma("unroll")                                              \
                for (int j = 0; j < 4; j++) {                                  \
                    yd0 += w4[j] * v4[j].x;                                    \
                    yd1 += w4[j] * v4[j].y;                                    \
                    yd2 += w4[j] * v4[j].z;                                    \
                    yd3 += w4[j] * v4[j].w;                                    \
                }                                                              \
            }                                                                  \
            for (int s_local = sr; s_local < t_local; s_local++) {            \
                float w = sm_tri[tri_row + s_local];                          \
                float4 v = lds_f32x4(                                         \
                    sm_v + s_local * hd + p0);                                \
                yd0 += w * v.x;                                               \
                yd1 += w * v.y;                                               \
                yd2 += w * v.z;                                               \
                yd3 += w * v.w;                                               \
            }                                                                 \
            float skip_w = d_skip + sm_qkd[t_local];                          \
            float4 vt =                                                       \
                lds_f32x4(sm_v + t_local * hd + p0);                          \
            long long ob = (long long)(b * T + chunk_start + t_local)         \
                           * d_inner + h * hd + p0;                           \
            y_out[ob + 0] = FROM_F(yd0 + y_off0 + skip_w * vt.x);             \
            y_out[ob + 1] = FROM_F(yd1 + y_off1 + skip_w * vt.y);             \
            y_out[ob + 2] = FROM_F(yd2 + y_off2 + skip_w * vt.z);             \
            y_out[ob + 3] = FROM_F(yd3 + y_off3 + skip_w * vt.w);             \
        }                                                                     \
    } else {                                                                  \
        for (int o = tid; o < chunk_len * hd; o += nt) {                      \
            int t_local = o / hd;                                             \
            int p = o % hd;                                                   \
            float y_off = 0.0f;                                               \
            float state_decay = FAST_EXP(sm_da[t_local]);                     \
            for (int n = 0; n < ds; n++) {                                    \
                y_off += sm_q[t_local * qk_ld + n] * sm_ps[n * ps_ld + p];          \
            }                                                                 \
            if (t_local >= cont) y_off = 0.0f; y_off *= state_decay;          \
            float y_diag = 0.0f;                                              \
            int tri_row = t_local * (t_local - 1) / 2;                        \
            int lo = vl_seq_first(seq_idx, b, T, chunk_start, chunk_start + t_local) \
                     - chunk_start;                                           \
            for (int s_local = lo; s_local < t_local; s_local++) {            \
                y_diag += sm_tri[tri_row + s_local] * sm_v[s_local * hd + p]; \
            }                                                                 \
            float y_skip =                                                    \
                (d_skip + sm_qkd[t_local]) * sm_v[t_local * hd + p];          \
            float y_val = y_diag + y_off + y_skip;                            \
            y_out[(b * T + chunk_start + t_local) * d_inner + h * hd + p] =   \
                FROM_F(y_val);                                                \
        }                                                                     \
    }                                                                         \
}

DEFINE_M3_CHUNK_SCAN_FWD_COOP(m3_chunk_scan_fwd_coop_bf16, int, __nv_bfloat16, from_f_bf16)
DEFINE_M3_CHUNK_SCAN_FWD_COOP(m3_chunk_scan_fwd_coop_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_M3_CHUNK_SCAN_FWD_COOP(m3_chunk_scan_fwd_coop_f16, int, __half, from_f_f16)
DEFINE_M3_CHUNK_SCAN_FWD_COOP(m3_chunk_scan_fwd_coop_f16_i64, long long, __half, from_f_f16)

// ============================================================================
// 5c. m3_chunk_scan_fwd_sliced -- the intra-chunk scan for any head and state
// ============================================================================
//
// The static-tile scan's outputs, bit for bit, through a tile that holds
// neither a whole state row nor the head's whole entering state:
// - each pair's Q.K dot is one thread's chain over the state, n ascending
//   from +0, its operands staged M3SS_NS states at a time;
// - each row's dot with the entering state likewise;
// - each diagonal sum is one thread's chain over the earlier rows of the
//   row's sequence in the chunk, s ascending from +0, over the cooperative
//   kernel's terms;
// - the decay, the masks and the combination are the static-tile kernel's
//   expressions, the contractions its compiler forms written out
//   (M3SS_STATE_TERM).
// Slicing the state moves only where a chain's operands come from between
// its steps, never its order, and a state width that is not a whole number
// of quads ends its chains in scalar steps, so no padding enters a sum. A
// block takes one (sequence, chunk, head) and M3SS_PB channels of the head;
// a wider head is split across blocks (blockIdx.z), each forming the pair
// triangle again, since no forward sum runs over the head's channels. Each
// thread owns a 4 x 4 block of the outputs and of the triangle, so every
// operand it reads from shared memory serves four chains.
//
// The tile depends on neither the head nor the state width: M3SS_FLOATS
// floats, 26 KB, so three blocks share an SM on every supported part.
// Grid: (B * n_chunks, nh, ceil(hd / M3SS_PB)). Block: 256.
#define M3SS_NS 16
#define M3SS_PB 64
#define M3SS_CS 64
#define M3SS_LD (M3SS_NS + 4)
#define M3SS_V_LD (M3SS_PB + 4)
#define M3SS_TRI ((M3SS_CS * (M3SS_CS - 1) / 2 + 3) & ~3)
#define M3SS_SLICES (2 * M3SS_CS * M3SS_LD + M3SS_PB * M3SS_LD)
#define M3SS_VALUES (M3SS_CS * M3SS_V_LD)
#define M3SS_ARENA (M3SS_SLICES > M3SS_VALUES ? M3SS_SLICES : M3SS_VALUES)
#define M3SS_FLOATS (M3SS_TRI + M3SS_ARENA + 5 * M3SS_CS)

// The static-tile kernel left an output's last sums to its compiler, and the
// compiler laid its code out two ways: on Blackwell (sm_100 and up) under
// NVRTC 13.2 and later it fused the entering state's term into the diagonal
// sum; under NVRTC 12.8 and 13.0, and on every earlier part, it rounded the
// term first and added it. The sliced scan forms the term as the static one
// did under the same compiler and part.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 \
    && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
#define M3SS_STATE_TERM(off, decay, yd) __fmaf_rn(off, decay, yd)
#else
#define M3SS_STATE_TERM(off, decay, yd) __fadd_rn(__fmul_rn(off, decay), yd)
#endif

#define DEFINE_M3_CHUNK_SCAN_FWD_SLICED(NAME, IDX, T_ACT)                     \
extern "C" __global__ __launch_bounds__(256, 3) void NAME(                    \
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
    int h = y_first + blockIdx.y;                                             \
    int p_lo = blockIdx.z * M3SS_PB;                                          \
    if (chunk_size > M3SS_CS) return;                                         \
    int chunk_start = chunk * chunk_size;                                     \
    int chunk_end = chunk_start + chunk_size;                                 \
    if (chunk_end > T) chunk_end = T;                                         \
    int chunk_len = chunk_end - chunk_start;                                  \
    int p_len = min(hd - p_lo, M3SS_PB);                                      \
    /* In a packed row (seq_idx) the entering state reaches only the */       \
    /* tokens before the chunk's first start, and a token's pairs stay */     \
    /* inside its own sequence. */                                            \
    int cont = vl_lower_bound(seq_idx, b, T, chunk_start, chunk_end,          \
                              vl_id(seq_idx, b, T, chunk_start - 1) + 1) - chunk_start; \
    IDX cs_base = ((b * n_chunks + chunk) * nh + h) * chunk_size;             \
    IDX state_head = ((b * n_chunks + chunk) * nh + h) * hd * ds;             \
                                                                              \
    extern __shared__ __align__(16) float m3ss_sm[];                          \
    float* sm_tri = m3ss_sm;                                                  \
    float* arena = sm_tri + M3SS_TRI;                                         \
    /* The slices: the chunk's Q and K_scaled rows and the block's */         \
    /* entering-state rows, each over M3SS_NS states; the value rows take */  \
    /* their place once every slice is done. */                               \
    float* sm_q = arena;                                                      \
    float* sm_k = sm_q + M3SS_CS * M3SS_LD;                                   \
    float* sm_ps = sm_k + M3SS_CS * M3SS_LD;                                  \
    float* sm_v = arena;                                                      \
    float* sm_da = arena + M3SS_ARENA;                                        \
    float* sm_qkd = sm_da + M3SS_CS;                                          \
    float* sm_sd = sm_qkd + M3SS_CS;                                          \
    int* sm_id = reinterpret_cast<int*>(sm_sd + M3SS_CS);                     \
    int* sm_lo = sm_id + M3SS_CS;                                             \
    int tid = threadIdx.x;                                                    \
    for (int i = tid; i < chunk_len; i += blockDim.x) {                       \
        sm_da[i] = dA_cumsum[cs_base + i];                                    \
        sm_qkd[i] = qk_dot[(b * T + chunk_start + i) * nh + h];               \
        sm_sd[i] = FAST_EXP(sm_da[i]);                                        \
        sm_id[i] = vl_id(seq_idx, b, T, chunk_start + i);                     \
        sm_lo[i] = vl_seq_first(seq_idx, b, T, chunk_start, chunk_start + i)  \
                   - chunk_start;                                             \
    }                                                                         \
                                                                              \
    /* The output block: rows 4 tr + i of the chunk, channels 4 tc + j of */  \
    /* the block. The triangle block (threads below 136): rows 4 ti + i */    \
    /* and columns 4 si + j, si <= ti, of the chunk's pair matrix. */         \
    int tr = tid >> 4;                                                        \
    int tc = tid & 15;                                                        \
    bool pairs = tid < 136;                                                   \
    int ti = 0;                                                               \
    while ((ti + 1) * (ti + 2) / 2 <= tid) ti++;                              \
    int si = tid - ti * (ti + 1) / 2;                                         \
    float acc_pair[4][4];                                                     \
    float acc_off[4][4];                                                      \
    _Pragma("unroll")                                                         \
    for (int i = 0; i < 4; i++) {                                             \
        _Pragma("unroll")                                                     \
        for (int j = 0; j < 4; j++) {                                         \
            acc_pair[i][j] = 0.0f;                                            \
            acc_off[i][j] = 0.0f;                                             \
        }                                                                     \
    }                                                                         \
    bool vec = (ds & 3) == 0 && M3_QUAD_ALIGNED(Q, T_ACT)                        \
               && M3_QUAD_ALIGNED(K_scaled, T_ACT)                               \
               && M3_QUAD_ALIGNED(prev_states, float);                           \
    for (int n0 = 0; n0 < ds; n0 += M3SS_NS) {                                \
        int width = min(ds - n0, M3SS_NS);                                    \
        __syncthreads();                                                      \
        {                                                                     \
            /* Four values a thread: row tid / 4, quad tid % 4. */            \
            int row = tid >> 2;                                               \
            int qd = tid & 3;                                                 \
            int valid = width - 4 * qd;                                       \
            float4 qv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);                  \
            float4 kv = qv;                                                   \
            float4 pv = qv;                                                   \
            if (valid > 0 && row < chunk_len) {                               \
                IDX base = ((b * T + chunk_start + row) * nh + h) * ds + n0 + 4 * qd; \
                qv = m3_load_quad_ro(Q + base, valid, vec);                   \
                kv = m3_load_quad_ro(K_scaled + base, valid, vec);            \
            }                                                                 \
            if (valid > 0 && row < p_len) {                                   \
                pv = m3_load_quad_ro(prev_states + state_head                 \
                                   + (IDX)(p_lo + row) * ds + n0 + 4 * qd,    \
                               valid, vec);                                   \
            }                                                                 \
            *reinterpret_cast<float4*>(sm_q + row * M3SS_LD + 4 * qd) = qv;   \
            *reinterpret_cast<float4*>(sm_k + row * M3SS_LD + 4 * qd) = kv;   \
            *reinterpret_cast<float4*>(sm_ps + row * M3SS_LD + 4 * qd) = pv;  \
        }                                                                     \
        __syncthreads();                                                      \
        int full = width >> 2;                                                \
        for (int qd = 0; qd < full; qd++) {                                   \
            float4 q4[4];                                                     \
            float4 s4[4];                                                     \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < 4; i++) {                                     \
                q4[i] = lds_f32x4(                                            \
                    sm_q + (4 * tr + i) * M3SS_LD + 4 * qd);                  \
                s4[i] = lds_f32x4(                                            \
                    sm_ps + (4 * tc + i) * M3SS_LD + 4 * qd);                 \
            }                                                                 \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < 4; i++) {                                     \
                _Pragma("unroll")                                             \
                for (int j = 0; j < 4; j++) {                                 \
                    acc_off[i][j] = __fmaf_rn(q4[i].x, s4[j].x, acc_off[i][j]); \
                    acc_off[i][j] = __fmaf_rn(q4[i].y, s4[j].y, acc_off[i][j]); \
                    acc_off[i][j] = __fmaf_rn(q4[i].z, s4[j].z, acc_off[i][j]); \
                    acc_off[i][j] = __fmaf_rn(q4[i].w, s4[j].w, acc_off[i][j]); \
                }                                                             \
            }                                                                 \
            if (pairs) {                                                      \
                _Pragma("unroll")                                             \
                for (int i = 0; i < 4; i++) {                                 \
                    q4[i] = lds_f32x4(                                        \
                        sm_q + (4 * ti + i) * M3SS_LD + 4 * qd);              \
                    s4[i] = lds_f32x4(                                        \
                        sm_k + (4 * si + i) * M3SS_LD + 4 * qd);              \
                }                                                             \
                _Pragma("unroll")                                             \
                for (int i = 0; i < 4; i++) {                                 \
                    _Pragma("unroll")                                         \
                    for (int j = 0; j < 4; j++) {                             \
                        acc_pair[i][j] = __fmaf_rn(q4[i].x, s4[j].x, acc_pair[i][j]); \
                        acc_pair[i][j] = __fmaf_rn(q4[i].y, s4[j].y, acc_pair[i][j]); \
                        acc_pair[i][j] = __fmaf_rn(q4[i].z, s4[j].z, acc_pair[i][j]); \
                        acc_pair[i][j] = __fmaf_rn(q4[i].w, s4[j].w, acc_pair[i][j]); \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
        /* The last states of a width that is not a whole number of quads, */ \
        /* one step each. */                                                  \
        for (int n = 4 * full; n < width; n++) {                              \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < 4; i++) {                                     \
                float q = sm_q[(4 * tr + i) * M3SS_LD + n];                   \
                _Pragma("unroll")                                             \
                for (int j = 0; j < 4; j++) {                                 \
                    acc_off[i][j] = __fmaf_rn(q, sm_ps[(4 * tc + j) * M3SS_LD + n], \
                                              acc_off[i][j]);                 \
                }                                                             \
            }                                                                 \
            if (pairs) {                                                      \
                _Pragma("unroll")                                             \
                for (int i = 0; i < 4; i++) {                                 \
                    float q = sm_q[(4 * ti + i) * M3SS_LD + n];               \
                    _Pragma("unroll")                                         \
                    for (int j = 0; j < 4; j++) {                             \
                        acc_pair[i][j] = __fmaf_rn(                           \
                            q, sm_k[(4 * si + j) * M3SS_LD + n], acc_pair[i][j]); \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    __syncthreads();                                                          \
                                                                              \
    /* The triangle: a pair across two sequences of a packed row is 0. */     \
    if (pairs) {                                                              \
        _Pragma("unroll")                                                     \
        for (int i = 0; i < 4; i++) {                                         \
            int t = 4 * ti + i;                                               \
            _Pragma("unroll")                                                 \
            for (int j = 0; j < 4; j++) {                                     \
                int s = 4 * si + j;                                           \
                if (s < t && t < chunk_len) {                                 \
                    float decay = FAST_EXP(m3_nonpos(sm_da[t] - sm_da[s]));   \
                    sm_tri[t * (t - 1) / 2 + s] =                             \
                        sm_id[t] == sm_id[s] ? __fmul_rn(decay, acc_pair[i][j]) : 0.0f; \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    /* The value rows of the block's channels, in the slices' place. */       \
    bool vec_v = (hd & 3) == 0 && M3_QUAD_ALIGNED(x, T_ACT);                     \
    for (int i = tid; i < chunk_len * (M3SS_PB / 4); i += blockDim.x) {       \
        int s = i / (M3SS_PB / 4);                                            \
        int pq = i - s * (M3SS_PB / 4);                                       \
        int valid = p_len - 4 * pq;                                           \
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);                       \
        if (valid > 0) {                                                      \
            v = m3_load_quad_ro(x + (b * T + chunk_start + s) * d_inner + h * hd \
                              + p_lo + 4 * pq,                                \
                          valid, vec_v);                                      \
        }                                                                     \
        *reinterpret_cast<float4*>(sm_v + s * M3SS_V_LD + 4 * pq) = v;        \
    }                                                                         \
    __syncthreads();                                                          \
                                                                              \
    /* Outputs: the diagonal chain, then the static-tile kernel's sum */      \
    /* y_diag + y_off + skip * v as its compiler contracts it. */             \
    float d_skip = D[h];                                                      \
    bool vec_y = (hd & 3) == 0 && M3_QUAD_ALIGNED(y_out, T_ACT);                 \
    _Pragma("unroll")                                                         \
    for (int i = 0; i < 4; i++) {                                             \
        int t = 4 * tr + i;                                                   \
        if (t >= chunk_len) continue;                                         \
        float state_decay = sm_sd[t];                                         \
        bool on = t < cont;                                                   \
        float yd0 = 0.0f, yd1 = 0.0f, yd2 = 0.0f, yd3 = 0.0f;                 \
        int tri_row = t * (t - 1) / 2;                                        \
        /* Four rows' reads together ahead of their arithmetic; each chain */  \
        /* takes the same products in the same order as one row at a time. */  \
        int sr = sm_lo[t];                                                     \
        for (; sr + 4 <= t; sr += 4) {                                         \
            float w4[4];                                                       \
            float4 v4[4];                                                      \
            lds_f32_row4(sm_tri + tri_row + sr, w4);                           \
            lds_f32x4_rows4(sm_v + sr * M3SS_V_LD + 4 * tc, M3SS_V_LD, v4);    \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 4; j++) {                                      \
                yd0 = __fmaf_rn(w4[j], v4[j].x, yd0);                          \
                yd1 = __fmaf_rn(w4[j], v4[j].y, yd1);                          \
                yd2 = __fmaf_rn(w4[j], v4[j].z, yd2);                          \
                yd3 = __fmaf_rn(w4[j], v4[j].w, yd3);                          \
            }                                                                  \
        }                                                                      \
        for (int s = sr; s < t; s++) {                                  \
            float w = sm_tri[tri_row + s];                                    \
            float4 v = lds_f32x4(sm_v + s * M3SS_V_LD + 4 * tc);              \
            yd0 = __fmaf_rn(w, v.x, yd0);                                     \
            yd1 = __fmaf_rn(w, v.y, yd1);                                     \
            yd2 = __fmaf_rn(w, v.z, yd2);                                     \
            yd3 = __fmaf_rn(w, v.w, yd3);                                     \
        }                                                                     \
        float skip_w = __fadd_rn(d_skip, sm_qkd[t]);                          \
        float4 vt = lds_f32x4(sm_v + t * M3SS_V_LD + 4 * tc);                 \
        float4 yv;                                                            \
        yv.x = __fmaf_rn(skip_w, vt.x,                                        \
                         M3SS_STATE_TERM(on ? acc_off[i][0] : 0.0f, state_decay, yd0)); \
        yv.y = __fmaf_rn(skip_w, vt.y,                                        \
                         M3SS_STATE_TERM(on ? acc_off[i][1] : 0.0f, state_decay, yd1)); \
        yv.z = __fmaf_rn(skip_w, vt.z,                                        \
                         M3SS_STATE_TERM(on ? acc_off[i][2] : 0.0f, state_decay, yd2)); \
        yv.w = __fmaf_rn(skip_w, vt.w,                                        \
                         M3SS_STATE_TERM(on ? acc_off[i][3] : 0.0f, state_decay, yd3)); \
        int valid = p_len - 4 * tc;                                           \
        if (valid > 0) {                                                      \
            m3_store_quad(y_out + (b * T + chunk_start + t) * d_inner + h * hd     \
                         + p_lo + 4 * tc,                                     \
                     yv, valid, vec_y);                                       \
        }                                                                     \
    }                                                                         \
}

DEFINE_M3_CHUNK_SCAN_FWD_SLICED(m3_chunk_scan_fwd_sliced, int, float)
DEFINE_M3_CHUNK_SCAN_FWD_SLICED(m3_chunk_scan_fwd_sliced_i64, long long, float)
DEFINE_M3_CHUNK_SCAN_FWD_SLICED(m3_chunk_scan_fwd_sliced_bf16, int, __nv_bfloat16)
DEFINE_M3_CHUNK_SCAN_FWD_SLICED(m3_chunk_scan_fwd_sliced_bf16_i64, long long, __nv_bfloat16)
DEFINE_M3_CHUNK_SCAN_FWD_SLICED(m3_chunk_scan_fwd_sliced_f16, int, __half)
DEFINE_M3_CHUNK_SCAN_FWD_SLICED(m3_chunk_scan_fwd_sliced_f16_i64, long long, __half)
#undef DEFINE_M3_CHUNK_SCAN_FWD_SLICED
#undef M3SS_STATE_TERM

// ============================================================================
// Typed (bf16/f16) variants of the chunked parallel backward
// kernels with typed input surface.
//
// All gradient OUTPUTS remain f32 — this is the PyTorch AMP master-grad
// invariant: bf16/f16 atomicAdd is not supported on compute capability ≤ sm_89
// and numerical safety requires f32 accumulation for reduction-style grads.
// Typed I/O is limited to saved-activation inputs (x, Q, K_scaled, d_y).
//
// The following backward kernels stay f32-only (no typed surface):
//   - m3_state_passing_bwd (inter-chunk prefix recurrence — f32 state only)
//   - m3_cumsum_bwd (reverse prefix sum — f32)
//   - m3_extract_da_cs_sum (f32-only utility)
// ============================================================================

// typed DEFINE_M3_CHUNK_SCAN_BWD + DEFINE_M3_CHUNK_STATE_BWD
// macros removed (dead — matches f32-section 6/8 removal above). The live
// typed chunked bwd path routes through DEFINE_M3_DQKV below and the
// dQK-theta stage of section 12, just like the f32 path.

// ============================================================================
// Typed (bf16/f16) variants of the HIGHEST-RISK "final grad"
// kernels. Two of the four chunked-bwd tail kernels accept typed activation
// input; the other two operate on f32 scalar grads only.
//
// Typed:
//   - m3_dqkv       : typed Q_rot/K_scaled/V_in/dO; all grad outputs f32
//   - m3_dqktheta   : typed Q_raw/K_raw; all grad outputs f32 (section 12,
//                     one body for every dtype)
// Pure f32 (NO typed variant — operate on f32 scalar grad arrays only):
//   - m3_ddt_dtrap  (scale/gamma → dt/trap grads, f32 scalar math)
//   - m3_final_grads (combines f32 dADT + dDT + dDT_angle into final grads)
// ============================================================================

// No __launch_bounds__ pin: the t-split launch runs (hd, T_SPLIT) blocks
// 512 threads; the 44 KB pair-mats tile bounds residency at <= 2 blocks/SM
// regardless, and the f32 twin has always compiled unpinned.
// Keep each arithmetic schedule in the same typed body. The transport
// schedule stages state once and reuses the separately rounded weighted
// pair products; the legacy schedule keeps its original expressions.
#define M3_DQKV_LOCAL_STATE_legacy \
    float d_state[MAMBA_RS_STATE_CAP];

#define M3_DQKV_LOCAL_STATE_transport

#define M3_DQKV_PAIR_STORAGE_legacy \
    float* decay_mat  = vdo_mat + tri_n; \
    float* exp_fwd_sm = decay_mat + tri_n;

#define M3_DQKV_PAIR_STORAGE_transport \
    float* weighted_kq_mat  = vdo_mat + tri_n; \
    float* exp_fwd_sm = weighted_kq_mat + tri_n;

#define M3_DQKV_LOAD_LOCAL_STATE_legacy \
    for (int n = 0; n < ds; n++) { \
        d_state[n] = enter_state[ \
        ((b * n_chunks + chunk_idx) * nh_total + h) * hd * ds \
        + p * ds + n]; \
    }

#define M3_DQKV_LOAD_LOCAL_STATE_transport

#define M3_DQKV_LOAD_STATE_TILES_legacy \
    if (ty == 0) { \
        for (int n = 0; n < ds; n++) { \
            ssm_sm[p * ds + n] = SSM_States[ \
            ((b * n_chunks + chunk_idx) * nh_total + h) * hd * ds \
            + p * ds + n]; \
        } \
    }

#define M3_DQKV_LOAD_STATE_TILES_transport \
    int state_base = ((b * n_chunks + chunk_idx) * nh_total + h) * hd * ds; \
    for (int n = lane; n < hd * ds; n += nlanes) { \
        ssm_sm[n] = enter_state[state_base + n]; \
        ssm2_sm[n] = SSM_States[state_base + n]; \
    }

#define M3_DQKV_WEIGHT_PAIRS_legacy \
    vdo_mat[idx] = vdo_acc; \
    decay_mat[idx] = \
    exp2f((da_cs_sm[bb] - da_cs_sm[aa]) * LOG2E);

#define M3_DQKV_WEIGHT_PAIRS_transport \
    const float decay = exp2f((da_cs_sm[bb] - da_cs_sm[aa]) * LOG2E); \
    vdo_mat[idx] = __fmul_rn(vdo_acc, decay); \
    weighted_kq_mat[idx] = __fmul_rn(kq_acc, decay);

#define M3_DQKV_DV_TERMS_legacy \
    for (int s = t + 1; s < sp.end; s++) { \
        float kq; \
        if (use_pair_mats) { kq = kq_mat[M3_TRI(t, s, CS)]; } \
        else { kq = 0.0f; for (int n = 0; n < ds; n++) \
            kq += k_sm[t * dsp + n] * q_sm[s * dsp + n]; } \
        float decay = use_pair_mats \
        ? decay_mat[M3_TRI(t, s, CS)] \
        : exp2f((da_cs_sm[s] - dA_t) * LOG2E); \
        dv_intra += kq * decay * do_sm[s * hdp + p]; \
    } \
    float dv_inter = 0.0f; \
    for (int n = 0; n < ds; n++) \
    dv_inter += k_sm[t * dsp + n] * d_state[n];

#define M3_DQKV_DV_TERMS_transport \
    for (int s = t + 1; s < sp.end; s++) { \
        if (use_pair_mats) { \
            const float weighted = weighted_kq_mat[M3_TRI(t, s, CS)]; \
            dv_intra += weighted * do_sm[s * hdp + p]; \
        } else { \
            float kq = 0.0f; \
            for (int n = 0; n < ds; n++) \
            kq += k_sm[t * dsp + n] * q_sm[s * dsp + n]; \
            float decay = exp2f((da_cs_sm[s] - dA_t) * LOG2E); \
            dv_intra += kq * decay * do_sm[s * hdp + p]; \
        } \
    } \
    float dv_inter = 0.0f; \
    for (int n = 0; n < ds; n++) \
    dv_inter += k_sm[t * dsp + n] * ssm_sm[p * ds + n];

#define M3_DQKV_DQ_DK_TERMS_legacy \
    for (int s = t + 1; s < sp.end; s++) { \
        float vdo; \
        if (use_pair_mats) { vdo = vdo_mat[M3_TRI(t, s, CS)]; } \
        else { vdo = 0.0f; for (int pp = 0; pp < hd; pp++) \
            vdo += v_sm[t * hdp + pp] * do_sm[s * hdp + pp]; } \
        float decay = use_pair_mats \
        ? decay_mat[M3_TRI(t, s, CS)] \
        : exp2f((da_cs_sm[s] - dA_t) * LOG2E); \
        dk_intra += vdo * decay * q_sm[s * dsp + n]; \
    } \
    dK_mid[((b * T + gt) * nh_total + h) * ds + n] = dk_intra; \
    float dq_intra = 0.0f; \
    for (int s = sp.first; s < t; s++) { \
        float vdo; \
        if (use_pair_mats) { vdo = vdo_mat[M3_TRI(s, t, CS)]; } \
        else { vdo = 0.0f; for (int pp = 0; pp < hd; pp++) \
            vdo += v_sm[s * hdp + pp] * do_sm[t * hdp + pp]; } \
        float decay = use_pair_mats \
        ? decay_mat[M3_TRI(s, t, CS)] \
        : exp2f((dA_t - da_cs_sm[s]) * LOG2E); \
        dq_intra += vdo * decay * k_sm[s * dsp + n]; \
    } \
    float dq_inter = 0.0f; \
    for (int pp = 0; pp < hd; pp++) \
    dq_inter += do_sm[t * hdp + pp] * ssm_sm[pp * ds + n];

#define M3_DQKV_DQ_DK_TERMS_transport \
    for (int s = t + 1; s < sp.end; s++) { \
        if (use_pair_mats) { \
            const float weighted = vdo_mat[M3_TRI(t, s, CS)]; \
            dk_intra += weighted * q_sm[s * dsp + n]; \
        } else { \
            float vdo = 0.0f; \
            for (int pp = 0; pp < hd; pp++) \
            vdo += v_sm[t * hdp + pp] * do_sm[s * hdp + pp]; \
            float decay = exp2f((da_cs_sm[s] - dA_t) * LOG2E); \
            dk_intra += vdo * decay * q_sm[s * dsp + n]; \
        } \
    } \
    dK_mid[((b * T + gt) * nh_total + h) * ds + n] = dk_intra; \
    float dq_intra = 0.0f; \
    for (int s = sp.first; s < t; s++) { \
        if (use_pair_mats) { \
            const float weighted = vdo_mat[M3_TRI(s, t, CS)]; \
            dq_intra += weighted * k_sm[s * dsp + n]; \
        } else { \
            float vdo = 0.0f; \
            for (int pp = 0; pp < hd; pp++) \
            vdo += v_sm[s * hdp + pp] * do_sm[t * hdp + pp]; \
            float decay = exp2f((dA_t - da_cs_sm[s]) * LOG2E); \
            dq_intra += vdo * decay * k_sm[s * dsp + n]; \
        } \
    } \
    float dq_inter = 0.0f; \
    for (int pp = 0; pp < hd; pp++) \
    dq_inter += do_sm[t * hdp + pp] * ssm2_sm[pp * ds + n];

#define M3_DQKV_RESTAGE_GRADIENT_legacy \
    if (ty == 0) { \
        for (int n = 0; n < ds; n++) \
        ssm_sm[p * ds + n] = d_state[n]; \
    }

#define M3_DQKV_RESTAGE_GRADIENT_transport

#define M3_DQKV_RELOAD_PRIMAL_legacy \
    for (int n = lane; n < hd * ds; n += nlanes) { \
        ssm2_sm[n] = SSM_States[ \
        ((b * n_chunks + chunk_idx) * nh_total + h) * hd * ds + n]; \
    }

#define M3_DQKV_RELOAD_PRIMAL_transport

#define M3_DQKV_DADT_PAIRS_legacy \
    for (int i = sp.first; i < t; i++) { \
        float vdo; \
        if (use_pair_mats) { vdo = vdo_mat[M3_TRI(i, t, CS)]; } \
        else { vdo = 0.0f; for (int pp = 0; pp < hd; pp++) \
            vdo += v_sm[i * hdp + pp] * do_sm[t * hdp + pp]; } \
        float decay = use_pair_mats \
        ? decay_mat[M3_TRI(i, t, CS)] \
        : exp2f((da_cs_sm[t] - da_cs_sm[i]) * LOG2E); \
        float kq; \
        if (use_pair_mats) { kq = kq_mat[M3_TRI(i, t, CS)]; } \
        else { kq = 0.0f; for (int n = 0; n < ds; n++) \
            kq += k_sm[i * dsp + n] * q_sm[t * dsp + n]; } \
        acc += vdo * decay * kq; \
    } \
    for (int j = t + 1; j < sp.end; j++) { \
        float vdo; \
        if (use_pair_mats) { vdo = vdo_mat[M3_TRI(t, j, CS)]; } \
        else { vdo = 0.0f; for (int pp = 0; pp < hd; pp++) \
            vdo += v_sm[t * hdp + pp] * do_sm[j * hdp + pp]; } \
        float decay = use_pair_mats \
        ? decay_mat[M3_TRI(t, j, CS)] \
        : exp2f((da_cs_sm[j] - da_cs_sm[t]) * LOG2E); \
        float kq; \
        if (use_pair_mats) { kq = kq_mat[M3_TRI(t, j, CS)]; } \
        else { kq = 0.0f; for (int n = 0; n < ds; n++) \
            kq += k_sm[t * dsp + n] * q_sm[j * dsp + n]; } \
        acc -= vdo * decay * kq; \
    }

#define M3_DQKV_DADT_PAIRS_transport \
    for (int i = sp.first; i < t; i++) { \
        if (use_pair_mats) { \
            const float weighted = vdo_mat[M3_TRI(i, t, CS)]; \
            const float kq = kq_mat[M3_TRI(i, t, CS)]; \
            acc += weighted * kq; \
        } else { \
            float vdo = 0.0f; \
            for (int pp = 0; pp < hd; pp++) \
            vdo += v_sm[i * hdp + pp] * do_sm[t * hdp + pp]; \
            float decay = exp2f((da_cs_sm[t] - da_cs_sm[i]) * LOG2E); \
            float kq = 0.0f; \
            for (int n = 0; n < ds; n++) \
            kq += k_sm[i * dsp + n] * q_sm[t * dsp + n]; \
            acc += vdo * decay * kq; \
        } \
    } \
    for (int j = t + 1; j < sp.end; j++) { \
        if (use_pair_mats) { \
            const float weighted = vdo_mat[M3_TRI(t, j, CS)]; \
            const float kq = kq_mat[M3_TRI(t, j, CS)]; \
            acc -= weighted * kq; \
        } else { \
            float vdo = 0.0f; \
            for (int pp = 0; pp < hd; pp++) \
            vdo += v_sm[t * hdp + pp] * do_sm[j * hdp + pp]; \
            float decay = exp2f((da_cs_sm[j] - da_cs_sm[t]) * LOG2E); \
            float kq = 0.0f; \
            for (int n = 0; n < ds; n++) \
            kq += k_sm[t * dsp + n] * q_sm[j * dsp + n]; \
            acc -= vdo * decay * kq; \
        } \
    }

#define DEFINE_M3_DQKV(NAME, IDX, T_ACT, FROM_F, MODE) \
extern "C" __global__ void \
NAME( \
float* __restrict__ dQ_mid, \
float* __restrict__ dK_mid, \
float* __restrict__ dV, \
float* __restrict__ dADT, \
float* __restrict__ dQK_dot_out, \
float* __restrict__ dD_partials, \
const T_ACT* __restrict__ Q_rot, \
const T_ACT* __restrict__ K_scaled, \
const T_ACT* __restrict__ V_in, \
const float* __restrict__ DA_CS, \
const float* __restrict__ DA_CS_SUM, \
const float* __restrict__ QK_dot_in, \
const float* __restrict__ SSM_States, \
const T_ACT* __restrict__ dO, \
const float* __restrict__ D_param, \
const float* __restrict__ enter_state, \
int B, int T, int nh_total, int hd, int ds, int CS, \
int use_pair_mats, \
    int y_first, /* this run's first y block */ \
    int z_first, /* this run's first z block */ \
    const int* __restrict__ seq_idx \
) { \
    int h = blockIdx.x; \
    IDX b = y_first + blockIdx.y; \
    int p = threadIdx.x; \
    int ty = threadIdx.y; \
    int TS = blockDim.y; \
    int lane = p + hd * ty; \
    int nlanes = hd * TS; \
    if (h >= nh_total || b >= B || p >= hd) return; \
    if (ds > MAMBA_RS_STATE_CAP || CS > 64) return; \
    unsigned warp_mask; \
    if (hd >= 32) { \
        warp_mask = 0xFFFFFFFFu; \
    } else { \
        unsigned lin = (unsigned)(threadIdx.y * blockDim.x + threadIdx.x); \
        unsigned seg_base = (lin & 31u) / (unsigned)hd * (unsigned)hd; \
        warp_mask = (((1u << hd) - 1u) << seg_base); \
    } \
    int d_inner = nh_total * hd; \
    int n_chunks = (T + CS - 1) / CS; \
    float D_val = D_param[h]; \
    float dD_acc = 0.0f; \
    M3_DQKV_LOCAL_STATE_##MODE \
    extern __shared__ float smem_all[]; \
    float* smem = smem_all; \
    const int dsp = ds + 1; \
    const int hdp = hd + 1; \
    float* q_sm    = smem; \
    float* k_sm    = q_sm + CS * dsp; \
    float* v_sm    = k_sm + CS * dsp; \
    float* do_sm   = v_sm + CS * hdp; \
    float* da_cs_sm = do_sm + CS * hdp; \
    float* qk_sm   = da_cs_sm + CS; \
    float* ssm_sm  = qk_sm + CS; \
    float* ssm2_sm   = ssm_sm + hd * ds; \
    float* dm_rev_sm = ssm2_sm + hd * ds; \
    float* dm_vec_sm = dm_rev_sm + CS; \
    int tri_n = CS * (CS - 1) / 2; \
    float* kq_mat  = dm_vec_sm + CS; \
    float* vdo_mat = kq_mat + tri_n; \
    M3_DQKV_PAIR_STORAGE_##MODE \
    float* exp_rev_sm = exp_fwd_sm + CS; \
    { \
        int chunk_idx = z_first + blockIdx.z; \
        int chunk_start = chunk_idx * CS; \
        int chunk_len = min(CS, T - chunk_start); \
        M3_DQKV_LOAD_LOCAL_STATE_##MODE \
        for (int t = ty; t < CS; t += TS) { \
            if (t < chunk_len) { \
                int gt = chunk_start + t; \
                v_sm[t * hdp + p] = \
                to_f(V_in[(b * T + gt) * d_inner + h * hd + p]); \
                do_sm[t * hdp + p] = \
                to_f(dO[(b * T + gt) * d_inner + h * hd + p]); \
            } else { \
                v_sm[t * hdp + p] = 0.0f; \
                do_sm[t * hdp + p] = 0.0f; \
            } \
        } \
        for (int n = p; n < ds; n += hd) { \
            for (int t = ty; t < CS; t += TS) { \
                if (t < chunk_len) { \
                    int gt = chunk_start + t; \
                    q_sm[t * dsp + n] = to_f( \
                    Q_rot[((b * T + gt) * nh_total + h) * ds + n]); \
                    k_sm[t * dsp + n] = to_f( \
                    K_scaled[((b * T + gt) * nh_total + h) * ds + n]); \
                } else { \
                    q_sm[t * dsp + n] = 0.0f; \
                    k_sm[t * dsp + n] = 0.0f; \
                } \
            } \
        } \
        if (p == 0) { \
            for (int t = ty; t < CS; t += TS) { \
                if (t < chunk_len) { \
                    da_cs_sm[t] = DA_CS[ \
                    ((b * n_chunks + chunk_idx) * nh_total + h) * CS + t]; \
                    qk_sm[t] = QK_dot_in[ \
                    (b * T + chunk_start + t) * nh_total + h]; \
                } else { \
                    da_cs_sm[t] = 0.0f; \
                    qk_sm[t] = 0.0f; \
                } \
            } \
        } \
        M3_DQKV_LOAD_STATE_TILES_##MODE \
        __syncthreads(); \
        if (use_pair_mats) { \
            float cs_sum_pre = DA_CS_SUM[ \
            (b * n_chunks + chunk_idx) * nh_total + h]; \
            int row = 0; \
            int row_start = 0; \
            for (int idx = lane; idx < tri_n; idx += nlanes) { \
                while (row_start + (CS - 1 - row) <= idx) { \
                    row_start += CS - 1 - row; \
                    row++; \
                } \
                int aa = row; \
                int bb = idx - row_start + aa + 1; \
                float kq_acc = 0.0f; \
                for (int n = 0; n < ds; n++) \
                kq_acc += k_sm[aa * dsp + n] * q_sm[bb * dsp + n]; \
                kq_mat[idx] = kq_acc; \
                float vdo_acc = 0.0f; \
                for (int pp = 0; pp < hd; pp++) \
                vdo_acc += v_sm[aa * hdp + pp] * do_sm[bb * hdp + pp]; \
                M3_DQKV_WEIGHT_PAIRS_##MODE \
            } \
            for (int tt = lane; tt < CS; tt += nlanes) { \
                exp_fwd_sm[tt] = exp2f(da_cs_sm[tt] * LOG2E); \
                exp_rev_sm[tt] = \
                exp2f((cs_sum_pre - da_cs_sm[tt]) * LOG2E); \
            } \
            __syncthreads(); \
        } \
        float da_cs_chunk_sum = DA_CS_SUM[ \
        (b * n_chunks + chunk_idx) * nh_total + h]; \
        for (int t = ty; t < chunk_len; t += TS) { \
            int gt = chunk_start + t; \
            float dA_t = da_cs_sm[t]; \
            const VlSpan sp = vl_span(seq_idx, b, T, chunk_start, chunk_len, t); \
            float exp_rev_t = use_pair_mats \
            ? exp_rev_sm[t] \
            : exp2f((da_cs_chunk_sum - dA_t) * LOG2E); \
            float dv_intra = 0.0f; \
            M3_DQKV_DV_TERMS_##MODE \
            if (!sp.to_end) dv_inter = 0.0f; dv_inter *= exp_rev_t; \
            float dv_skip = do_sm[t * hdp + p] * (D_val + qk_sm[t]); \
            dV[(b * T + gt) * d_inner + h * hd + p] = \
            dv_intra + dv_inter + dv_skip; \
            float dqk_val = do_sm[t * hdp + p] * v_sm[t * hdp + p]; \
            for (int off = hd / 2; off > 0; off >>= 1) \
            dqk_val += __shfl_down_sync(warp_mask, dqk_val, off, hd); \
            if (p == 0) { \
                dQK_dot_out[(b * T + gt) * nh_total + h] = dqk_val; \
            } \
        } \
        __syncthreads(); \
        if (p == 0 && ty == 0) { \
            for (int t = 0; t < chunk_len; t++) \
            dD_acc += \
            dQK_dot_out[(b * T + chunk_start + t) * nh_total + h]; \
        } \
        for (int n = p; n < ds; n += hd) { \
            for (int t = ty; t < chunk_len; t += TS) { \
                int gt = chunk_start + t; \
                float dA_t = da_cs_sm[t]; \
                const VlSpan sp = vl_span(seq_idx, b, T, chunk_start, chunk_len, t); \
                float dk_intra = 0.0f; \
                M3_DQKV_DQ_DK_TERMS_##MODE \
                if (!sp.from_before) dq_inter = 0.0f; \
                dq_inter *= use_pair_mats ? exp_fwd_sm[t] : exp2f(dA_t * LOG2E); \
                dQ_mid[((b * T + gt) * nh_total + h) * ds + n] = \
                dq_intra + dq_inter; \
            } \
        } \
        __syncthreads(); \
        M3_DQKV_RESTAGE_GRADIENT_##MODE \
        __syncthreads(); \
        for (int n = p; n < ds; n += hd) { \
            for (int t = ty; t < chunk_len; t += TS) { \
                float dk_inter = 0.0f; \
                float exp_rev_t = use_pair_mats \
                ? exp_rev_sm[t] \
                : exp2f((da_cs_chunk_sum - da_cs_sm[t]) * LOG2E); \
                for (int pp = 0; pp < hd; pp++) \
                dk_inter += v_sm[t * hdp + pp] * ssm_sm[pp * ds + n]; \
                if (vl_id(seq_idx, b, T, chunk_start + t) \
                    != vl_id(seq_idx, b, T, chunk_start + chunk_len - 1)) \
                    dk_inter = 0.0f; \
                dk_inter *= exp_rev_t; \
                int gt = chunk_start + t; \
                dK_mid[((b * T + gt) * nh_total + h) * ds + n] += dk_inter; \
            } \
        } \
        __syncthreads(); \
        M3_DQKV_RELOAD_PRIMAL_##MODE \
        __syncthreads(); \
        for (int t = lane; t < chunk_len; t += nlanes) { \
            float acc = 0.0f; \
            const VlSpan sp = vl_span(seq_idx, b, T, chunk_start, chunk_len, t); \
            M3_DQKV_DADT_PAIRS_##MODE \
            float qs_do = 0.0f; \
            for (int pp = 0; pp < hd; pp++) { \
                float qs = 0.0f; \
                for (int n = 0; n < ds; n++) \
                qs += q_sm[t * dsp + n] * ssm2_sm[pp * ds + n]; \
                qs_do += qs * do_sm[t * hdp + pp]; \
            } \
            if (sp.from_before) acc += qs_do \
            * (use_pair_mats ? exp_fwd_sm[t] \
            : exp2f(da_cs_sm[t] * LOG2E)); \
            dm_rev_sm[t] = acc; \
            float dsk_v = 0.0f; \
            for (int pp = 0; pp < hd; pp++) { \
                float dsk = 0.0f; \
                for (int n = 0; n < ds; n++) \
                dsk += k_sm[t * dsp + n] * ssm_sm[pp * ds + n]; \
                dsk_v += dsk * v_sm[t * hdp + pp]; \
            } \
            dm_vec_sm[t] = !sp.to_end ? 0.0f : dsk_v \
            * (use_pair_mats ? exp_rev_sm[t] \
            : exp2f((da_cs_chunk_sum - da_cs_sm[t]) * LOG2E)); \
        } \
        __syncthreads(); \
        if (p == 0 && ty == 0) { \
            float dM_scalar = 0.0f; \
            for (int pp = 0; pp < hd; pp++) { \
                for (int n = 0; n < ds; n++) \
                dM_scalar += ssm2_sm[pp * ds + n] * ssm_sm[pp * ds + n]; \
            } \
            dM_scalar *= exp2f(da_cs_chunk_sum * LOG2E); \
            /* A packed row combines each of its sequences in the */ \
            /* chunk on its own (see the f32 kernel). */ \
            const bool passes = vl_passes(seq_idx, b, T, chunk_start, chunk_len); \
            for (int a = 0; a < chunk_len;) { \
                const int e = vl_span(seq_idx, b, T, chunk_start, chunk_len, a).end; \
                float total_rev = 0.0f; \
                for (int t = a; t < e; t++) total_rev += dm_rev_sm[t]; \
                if (passes) total_rev += dM_scalar; \
                float cumsum = 0.0f; \
                for (int t = a; t < e; t++) { \
                    cumsum += dm_vec_sm[t] - dm_rev_sm[t]; \
                    float out = \
                    dm_rev_sm[t] + (total_rev + cumsum - dm_vec_sm[t]); \
                    int gt = chunk_start + t; \
                    dADT[(b * T + gt) * nh_total + h] = out; \
                } \
                a = e; \
            } \
        } \
        __syncthreads(); \
    } \
    if (p == 0 && ty == 0) \
    dD_partials[(b * n_chunks + (z_first + blockIdx.z)) * nh_total + h] = dD_acc; \
    (void)FROM_F; \
}


DEFINE_M3_DQKV(m3_dqkv_bf16, int, __nv_bfloat16, from_f_bf16, legacy)
DEFINE_M3_DQKV(m3_dqkv_bf16_i64, long long, __nv_bfloat16, from_f_bf16, legacy)
DEFINE_M3_DQKV(m3_dqkv_f16, int, __half, from_f_f16, legacy)
DEFINE_M3_DQKV(m3_dqkv_f16_i64, long long, __half, from_f_f16, legacy)
DEFINE_M3_DQKV(m3_dqkv_transport_f32, int, float, from_f_f32, transport)
DEFINE_M3_DQKV(m3_dqkv_transport_bf16, int, __nv_bfloat16, from_f_bf16, transport)
DEFINE_M3_DQKV(m3_dqkv_transport_f16, int, __half, from_f_f16, transport)

#undef M3_DQKV_LOCAL_STATE_legacy
#undef M3_DQKV_LOCAL_STATE_transport
#undef M3_DQKV_PAIR_STORAGE_legacy
#undef M3_DQKV_PAIR_STORAGE_transport
#undef M3_DQKV_LOAD_LOCAL_STATE_legacy
#undef M3_DQKV_LOAD_LOCAL_STATE_transport
#undef M3_DQKV_LOAD_STATE_TILES_legacy
#undef M3_DQKV_LOAD_STATE_TILES_transport
#undef M3_DQKV_WEIGHT_PAIRS_legacy
#undef M3_DQKV_WEIGHT_PAIRS_transport
#undef M3_DQKV_DV_TERMS_legacy
#undef M3_DQKV_DV_TERMS_transport
#undef M3_DQKV_DQ_DK_TERMS_legacy
#undef M3_DQKV_DQ_DK_TERMS_transport
#undef M3_DQKV_RESTAGE_GRADIENT_legacy
#undef M3_DQKV_RESTAGE_GRADIENT_transport
#undef M3_DQKV_RELOAD_PRIMAL_legacy
#undef M3_DQKV_RELOAD_PRIMAL_transport
#undef M3_DQKV_DADT_PAIRS_legacy
#undef M3_DQKV_DADT_PAIRS_transport

// Independent output columns share a block while retaining each original
// lane's input stride and every step of the shared-memory reduction tree.
extern "C" __global__ __launch_bounds__(256, 4)
void m3_reduce_sum_axis0_packed(
    float* __restrict__ out,
    const float* __restrict__ partials,
    int batch, int dim, int accumulate
) {
    const int columns = blockDim.x;
    const int lanes = blockDim.y;
    const int column = threadIdx.x;
    const int lane = threadIdx.y;
    const int d = blockIdx.x * columns + column;
    const int slot = lane * columns + column;
    extern __shared__ float sdata[];

    float sum = 0.0f;
    if (d < dim) {
        for (int b = lane; b < batch; b += lanes) {
            sum += partials[b * dim + d];
        }
    }
    sdata[slot] = sum;
    __syncthreads();
    for (int s = lanes / 2; s > 0; s >>= 1) {
        if (lane < s) sdata[slot] += sdata[slot + s * columns];
        __syncthreads();
    }
    if (lane == 0 && d < dim) {
        out[d] = accumulate ? out[d] + sdata[column] : sdata[column];
    }
}
