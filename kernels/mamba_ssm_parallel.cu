// Parallel prefix scan for Mamba-1 SSM recurrence.
//
// Faithful reimplementation of Tri Dao's selective_scan_fwd_kernel.cuh
// without CUB, PyTorch, or c10 dependencies -- pure NVRTC-compilable CUDA.
//
// Algorithm (identical to original):
//   1. Grid: (batch, d_inner) -- one block per (b, d) pair.
//   2. Block: NTHREADS threads, each owns NITEMS consecutive timesteps.
//      Chunk size = NTHREADS * NITEMS = 1024.
//   3. Outer loop over d_state (sequential, like original).
//   4. For each state index n:
//      a. Load delta, u, B for NITEMS timesteps -> compute (da, delta*u*B) pairs.
//      b. Thread-local sequential scan of NITEMS pairs.
//      c. Block-level inclusive scan via warp shuffle + shared memory raking.
//      d. Inter-chunk carry via smem_running_prefix (exactly like original).
//      e. Single-pass Y accumulation: y[t] += h[t] * C[t] during scan.
//   5. After all d_state iterations, y already contains the full output.
//
// Scan operator: (a1, b1) o (a0, b0) = (a1*a0, a1*b0 + b1)
// Encodes the linear recurrence: h_t = da_t * h_{t-1} + db_t
//
// The running prefix (run_a, run_b) satisfies:
//   h = run_a * h_init + run_b
// where h_init is the initial state h[h_base + n].
//
// Source: Gu & Dao (2023), "Mamba: Linear-Time Sequence Modeling"
//         selective_scan_fwd_kernel.cuh, selective_scan_common.h

// Typed-I/O prelude (to_f / from_f_* upcast/downcast helpers).
// Typed (bf16/f16) mixed-precision parallel scan forward. Following
// state-spaces/mamba's `scan_t = float2` invariant (all scan state in
// f32) and our BPTT precision discipline (h, h_saved,
// a_neg, D, smem_* remain f32). Only the activation I/O tensors
// (delta, u, B, C, y_out) are typed.
#include "_typed_prelude.cuh"

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

// Block config: 128 threads x 8 items = 1024 elements per chunk.
// exp(x) = exp2(x * LOG2E) -- we fold LOG2E into a_neg once per (d, n)
// so the inner loop uses a single exp2f() with no extra FMUL.
#define NTHREADS 128
#define NITEMS   8
#define CHUNK_SIZE (NTHREADS * NITEMS)
#define NWARPS   (NTHREADS / 32)
// Resident-block pin scales with the block size (128 -> 3 keeps the
// historical codegen envelope; 256 -> 2 keeps ~128 regs/thread).
#if NTHREADS >= 256
#define SCAN_MINB 2
#else
#define SCAN_MINB 3
#endif

// Must be >= actual d_state. Matches Tri Dao's MAX_DSTATE = 256.
#define MAX_DSTATE 256

// ============================================================================
// Eight consecutive B or C values of one (b, n) row, t0 .. t0+7, as f32.
// The rows are t-contiguous, so a thread's eight items are one 32-byte
// (f32) or 16-byte (half) span. When that span lies inside the row and is
// 16-byte aligned it comes in as vector loads, at 8-byte alignment as
// half-width vectors, otherwise element by element with the tail past T
// zero-filled. The values are the same either way; only the number of
// load instructions changes.
// ============================================================================
template <typename T>
__device__ __forceinline__ void load_row8(
    const T* __restrict__ row, int t0, int T_len, float out[NITEMS]
) {
    const T* p = row + t0;
    unsigned long long addr = (unsigned long long)p;
    bool whole = (t0 + NITEMS <= T_len);
    if (whole && (addr & 15ull) == 0) {
        constexpr int PER16 = 16 / (int)sizeof(T);
        #pragma unroll
        for (int k = 0; k < NITEMS / PER16; k++) {
            uint4 v = reinterpret_cast<const uint4*>(p)[k];
            const T* e = reinterpret_cast<const T*>(&v);
            #pragma unroll
            for (int j = 0; j < PER16; j++) out[k * PER16 + j] = to_f(e[j]);
        }
    } else if (whole && (addr & 7ull) == 0) {
        constexpr int PER8 = 8 / (int)sizeof(T);
        #pragma unroll
        for (int k = 0; k < NITEMS / PER8; k++) {
            uint2 v = reinterpret_cast<const uint2*>(p)[k];
            const T* e = reinterpret_cast<const T*>(&v);
            #pragma unroll
            for (int j = 0; j < PER8; j++) out[k * PER8 + j] = to_f(e[j]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < NITEMS; i++)
            out[i] = (t0 + i < T_len) ? to_f(p[i]) : 0.0f;
    }
}

// ============================================================================
// Warp-level inclusive scan of (a, b) pairs using warp shuffle.
// After return, lane k holds compose(pair_0, ..., pair_k) within its warp.
// ============================================================================
// BUG FIX: accept a mask parameter rather than hardcoding
// 0xffffffff. Step 3 of `block_inclusive_scan_ab` calls this with only
// NWARPS=4 active lanes out of the warp; `__shfl_up_sync(0xffffffff, ...)`
// is UB when mask members don't all execute → silent hang on Ada/sm_89.
// CUDA docs explicitly require the mask to describe the set of actively
// participating threads.
__device__ __forceinline__ void warp_inclusive_scan_ab(
    float &a, float &b, unsigned mask = 0xffffffff
) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float a_prev = __shfl_up_sync(mask, a, offset);
        float b_prev = __shfl_up_sync(mask, b, offset);
        if ((threadIdx.x & 31) >= (unsigned)offset) {
            b = a * b_prev + b;
            a = a * a_prev;
        }
    }
}

// ============================================================================
// Warp-level inclusive REVERSE scan of (a, b) pairs (Tri Dao
// `ThreadReverseScan` from `selective_scan/reverse_scan.cuh`).
//
// Forward scan composes left→right: lane k holds compose(p_0, ..., p_k).
// Reverse scan composes right→left: lane k holds compose(p_k, ..., p_31).
//
// Compose op (same as forward):
//   (a2, b2) ∘ (a1, b1) = (a2*a1, a2*b1 + b2)
//
// In the SSM bwd, this propagates dh_t backward in time:
//   p_t = (delta_A_next[t], dout[t]*B[t]*C[t])
//   reverse_scan_t = compose(p_t, p_{t+1}, ..., p_{T-1})
// so reverse_scan_t.b is dh_t. The "next-step" delta_A is what makes the
// gradient correctly multiply by future-step decay (Tri Dao trick).
// ============================================================================
// `active` = number of participating low lanes (must equal popcount(mask)
// for a contiguous low mask). The update guard must stop at `active`, not
// 32: __shfl_down_sync from a lane outside the mask returns an UNDEFINED
// value, and the old `lane + offset < 32` guard composed those undefined
// values into the scan whenever mask < full warp (the NWARPS-total scan in
// block_inclusive_reverse_scan_ab). The forward twin is immune because its
// shfl_up guard `lane >= offset` only ever reads lower (in-mask) lanes.
__device__ __forceinline__ void warp_inclusive_reverse_scan_ab(
    float &a, float &b, unsigned mask = 0xffffffff, int active = 32
) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float a_next = __shfl_down_sync(mask, a, offset);
        float b_next = __shfl_down_sync(mask, b, offset);
        if ((int)(threadIdx.x & 31) + offset < active) {
            // compose(self, next): (a*a_next, a*b_next + b) NO — careful:
            // reverse semantic: lane k accumulates (p_k ∘ p_{k+1} ∘ ...).
            // op (a2,b2)∘(a1,b1) = (a2*a1, a2*b1 + b2) with self=2nd arg.
            // So acc_k = self ∘ acc_{k+1} where acc_{k+1} arrives via shfl.
            b = a * b_next + b;
            a = a * a_next;
        }
    }
}

// ============================================================================
// Block-level inclusive REVERSE scan of (a, b) pairs.
// Mirror of `block_inclusive_scan_ab` walking right-to-left.
// ============================================================================
__device__ __forceinline__ void block_inclusive_reverse_scan_ab(
    float &a, float &b,
    float *smem_wa, float *smem_wb
) {
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    // Step 1: intra-warp inclusive reverse scan (lane 0 holds full warp tail)
    warp_inclusive_reverse_scan_ab(a, b);

    // Step 2: lane 0 of each warp stores its inclusive total (the full
    // composition of that warp from right-most lane back to lane 0).
    if (lane == 0) {
        smem_wa[warp_id] = a;
        smem_wb[warp_id] = b;
    }
    __syncthreads();

    // Step 3: first warp scans the NWARPS totals in REVERSE.
    // Partial mask AND matching `active` bound: only lanes < NWARPS hold
    // valid totals, so the compose guard must stop at NWARPS (see the
    // helper's comment — stopping at 32 composed undefined shuffle values).
    if (warp_id == 0 && lane < NWARPS) {
        float wa = smem_wa[lane];
        float wb = smem_wb[lane];
        warp_inclusive_reverse_scan_ab(wa, wb, (1u << NWARPS) - 1u, NWARPS);
        smem_wa[lane] = wa;
        smem_wb[lane] = wb;
    }
    __syncthreads();

    // Step 4: threads in warp < NWARPS-1 compose with the NEXT warp's postfix.
    if (warp_id < NWARPS - 1) {
        float na = smem_wa[warp_id + 1];
        float nb = smem_wb[warp_id + 1];
        b = a * nb + b;
        a = a * na;
    }
    // No __syncthreads here — caller syncs before next smem_wa/wb use.
}

// ============================================================================
// Block-level inclusive scan of (a, b) pairs.
// Two-level: warp scan -> inter-warp scan via shared memory -> compose.
// This is CUB's BLOCK_SCAN_WARP_SCANS algorithm.
// ============================================================================
__device__ __forceinline__ void block_inclusive_scan_ab(
    float &a, float &b,
    float *smem_wa, float *smem_wb
) {
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    // Step 1: intra-warp inclusive scan
    warp_inclusive_scan_ab(a, b);

    // Step 2: last lane of each warp stores its inclusive total
    if (lane == 31) {
        smem_wa[warp_id] = a;
        smem_wb[warp_id] = b;
    }
    __syncthreads();

    // Step 3: first warp scans the NWARPS totals. Only lanes 0..NWARPS-1
    // participate — mask must reflect that or the __shfl_up_sync calls
    // inside warp_inclusive_scan_ab deadlock on Ada/sm_89 (mask
    // 0xffffffff requires all 32 lanes to execute the same sync).
    if (warp_id == 0 && lane < NWARPS) {
        float wa = smem_wa[lane];
        float wb = smem_wb[lane];
        warp_inclusive_scan_ab(wa, wb, (1u << NWARPS) - 1u);
        smem_wa[lane] = wa;
        smem_wb[lane] = wb;
    }
    __syncthreads();

    // Step 4: threads in warp > 0 compose with previous warp's prefix
    if (warp_id > 0) {
        float pa = smem_wa[warp_id - 1];
        float pb = smem_wb[warp_id - 1];
        b = a * pb + b;
        a = a * pa;
    }
    // No __syncthreads here -- caller syncs before next smem_wa/wb use.
}

// ============================================================================
// Both block scans at once: the forward scan of (fa, fb) and the reverse
// scan of (ra, rb), each exactly as its own helper computes it, sharing
// the two barriers. The backward runs a forward replay and a reverse scan
// per state dimension that do not depend on each other, so interleaving
// them halves the barrier count without touching a single operation.
// ============================================================================
__device__ __forceinline__ void block_scan_fwd_and_reverse_ab(
    float &fa, float &fb, float &ra, float &rb,
    float *smem_fwd_wa, float *smem_fwd_wb,
    float *smem_rev_wa, float *smem_rev_wb
) {
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    warp_inclusive_scan_ab(fa, fb);
    warp_inclusive_reverse_scan_ab(ra, rb);
    if (lane == 31) {
        smem_fwd_wa[warp_id] = fa;
        smem_fwd_wb[warp_id] = fb;
    }
    if (lane == 0) {
        smem_rev_wa[warp_id] = ra;
        smem_rev_wb[warp_id] = rb;
    }
    __syncthreads();

    if (warp_id == 0 && lane < NWARPS) {
        float wa = smem_fwd_wa[lane];
        float wb = smem_fwd_wb[lane];
        warp_inclusive_scan_ab(wa, wb, (1u << NWARPS) - 1u);
        smem_fwd_wa[lane] = wa;
        smem_fwd_wb[lane] = wb;
        float va = smem_rev_wa[lane];
        float vb = smem_rev_wb[lane];
        warp_inclusive_reverse_scan_ab(va, vb, (1u << NWARPS) - 1u, NWARPS);
        smem_rev_wa[lane] = va;
        smem_rev_wb[lane] = vb;
    }
    __syncthreads();

    if (warp_id > 0) {
        float pa = smem_fwd_wa[warp_id - 1];
        float pb = smem_fwd_wb[warp_id - 1];
        fb = fa * pb + fb;
        fa = fa * pa;
    }
    if (warp_id < NWARPS - 1) {
        float na = smem_rev_wa[warp_id + 1];
        float nb = smem_rev_wb[warp_id + 1];
        rb = ra * nb + rb;
        ra = ra * na;
    }
    // No __syncthreads here -- caller syncs before the workspace is reused.
}

// ============================================================================
// Shared memory layout of the backward kernels (in extern __shared__ float[]):
//
//   [0                       .. NWARPS)          = smem_wa      (block scan)
//   [NWARPS                  .. 2*NWARPS)        = smem_wb      (block scan)
//   [2*NWARPS                .. 2*NWARPS+MAX_DS) = smem_run_a   (inter-chunk carry)
//   [2*NWARPS+MAX_DS         .. 2*NWARPS+2*MAX)  = smem_run_b   (inter-chunk carry)
//   [2*NWARPS+2*MAX_DS       .. +NTHREADS)       = smem_exch_a  (exclusive prefix)
//   [2*NWARPS+2*MAX_DS+NTHR  .. +NTHREADS)       = smem_exch_b  (exclusive prefix)
//   [2*NWARPS+2*MAX_DS+2*NTHR .. +CHUNK_SIZE)    = smem_stage   (coalesced load staging)
//
// Total: 2*4 + 2*256 + 2*128 + 1024 = 1800 floats = 7200 bytes.
//
// The forward kernels pack a smaller layout at the runtime d_state: the two
// block-scan rows, the two carry rows, and one exchange slot per warp for
// the prefix hand-off (2*NWARPS + 2*d_state + 2*NWARPS floats); they load
// and store directly and keep no staging region. The launcher sizes it.
// ============================================================================
#define SMEM_WA_OFF        0
#define SMEM_WB_OFF        (NWARPS)
#define SMEM_RUN_A_OFF     (2 * NWARPS)
#define SMEM_RUN_B_OFF     (2 * NWARPS + MAX_DSTATE)
#define SMEM_EXCH_A_OFF    (2 * NWARPS + 2 * MAX_DSTATE)
#define SMEM_EXCH_B_OFF    (2 * NWARPS + 2 * MAX_DSTATE + NTHREADS)
#define SMEM_STAGE_OFF     (2 * NWARPS + 2 * MAX_DSTATE + 2 * NTHREADS)
#define SMEM_TOTAL_FLOATS  (2 * NWARPS + 2 * MAX_DSTATE + 2 * NTHREADS + CHUNK_SIZE)

// Extra smem offsets for the backward-pass reverse scan.
// Layout (appended after the forward layout):
//   SMEM_REV_WA/WB     = 2*NWARPS floats   (reverse warp-scan workspace)
//   SMEM_POST_A/B      = 2*MAX_DSTATE      (inter-chunk reverse-scan postfix)
//   SMEM_NEXT_A        = NTHREADS          (next-thread δA exchange buffer)
//   SMEM_DA_LOG_RED    = NTHREADS          (block-reduce of d_a_log per (n))
//   SMEM_CHUNK_FIRST_A = MAX_DSTATE        (per-n boundary from later chunk
//                                           used by earlier chunk's last
//                                           thread as pair.a = a_{t+1})
//
// Total bwd extra: 2*4 + 2*256 + 128 + 128 + 256 = 1032 floats = 4128 B
// added to the 7200 B fwd footprint → 11328 B per block (still < 48 KB so
// no cudaFuncSetAttribute needed at default MAX_DSTATE=256). Rust launch
// code (src/mamba_ssm/gpu/launch.rs::grid_parallel_scan_bwd) matches this.
#define SMEM_REV_WA_OFF        (SMEM_TOTAL_FLOATS)
#define SMEM_REV_WB_OFF        (SMEM_TOTAL_FLOATS + NWARPS)
#define SMEM_POST_A_OFF        (SMEM_TOTAL_FLOATS + 2 * NWARPS)
#define SMEM_POST_B_OFF        (SMEM_TOTAL_FLOATS + 2 * NWARPS + MAX_DSTATE)
#define SMEM_NEXT_A_OFF        (SMEM_TOTAL_FLOATS + 2 * NWARPS + 2 * MAX_DSTATE)
#define SMEM_DA_RED_OFF        (SMEM_TOTAL_FLOATS + 2 * NWARPS + 2 * MAX_DSTATE + NTHREADS)
// Boundary: stores the "first thread's first da" of THIS chunk so that the
// PREVIOUS (earlier-in-time) chunk's last thread can use it as its
// `pair.a = a_{t+1}` boundary when computing reverse-scan dh. Initialized
// to 1.0 (identity) for the very-last chunk in time.
#define SMEM_CHUNK_FIRST_A_OFF (SMEM_TOTAL_FLOATS + 2 * NWARPS + 2 * MAX_DSTATE + 2 * NTHREADS)
#define SMEM_BWD_FLOATS        (SMEM_TOTAL_FLOATS + 2 * NWARPS + 3 * MAX_DSTATE + 2 * NTHREADS)

// ============================================================================
// Forward: parallel prefix scan with activation saves (training path).
//
// Same interface as ssm_burnin_forward. Same saved activations format.
// Single-pass Y accumulation within the d_state loop.
//
// Grid: (batch, d_inner). Block: NTHREADS.
// Shared memory: the forward layout described above, sized by the launcher.
// ============================================================================
extern "C" __global__ __launch_bounds__(NTHREADS, SCAN_MINB) void ssm_parallel_scan_fwd(
    float* __restrict__ h,             // [batch * d_inner * d_state] SSM state (mutated)
    float* __restrict__ y_out,         // [batch * T * d_inner] output
    float* __restrict__ h_saved,       // [batch * (T+1) * d_inner * d_state] saved for backward
    // Pre-softplus dt: the kernel applies softplus inline (through the
    // exact store rounding the deleted softplus_copy pass produced) and
    // WRITES the post-softplus save the backward replays from - one
    // launch and one full read pass fewer per layer.
    const float* __restrict__ delta_raw, // [batch * T * d_inner]
    float* __restrict__ delta_saved,     // [batch * T * d_inner]
    const float* __restrict__ u,       // [batch * T * d_inner]
    const float* __restrict__ B,       // [batch * T * d_state]
    const float* __restrict__ C,       // [batch * T * d_state]
    const float* __restrict__ a_neg,   // [d_inner * d_state]
    const float* __restrict__ D,       // [d_inner]
    int batch, int T, int d_inner, int d_state,
    // Slim tape: [batch*d_inner*d_state*3*n_chunks] rows of
    // (run_a, run_b, h_entry) per chunk. No __restrict__: under slim
    // the launcher passes the SAME buffer for h_saved and run_tape and
    // the kernel touches exactly one of them per launch.
    float* run_tape,
    int slim_tape
) {
    int bid = blockIdx.x;
    int did = blockIdx.y;
    if (bid >= batch || did >= d_inner) return;
    if (d_state > MAX_DSTATE) return;

    extern __shared__ float smem[];
    float *smem_wa     = smem + SMEM_WA_OFF;
    float *smem_wb     = smem + SMEM_WB_OFF;
    /* Runtime d_state stride: run/exchange/stage regions pack at the actual
     * d_state instead of MAX_DSTATE (the launcher shrinks the allocation to
     * match). Region ORDER and contents are unchanged, so every cell holds
     * the same value as the padded layout - address-only. */
    float *smem_run_a  = smem + 2 * NWARPS;
    float *smem_run_b  = smem_run_a + d_state;
    float *smem_exch_a = smem_run_b + d_state;
    float *smem_exch_b = smem_exch_a + NWARPS;

    float D_d = D[did];
    int h_base = (bid * d_inner + did) * d_state;
    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Save initial SSM state (slim: tape row head — the chunk-0 prefix
    // is the identity and the chunk-0 entry state is h_0 itself).
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {
        if (slim_tape) {
            int row = ((bid * d_inner + did) * d_state + n) * 3 * n_chunks;
            run_tape[row + 0] = 1.0f;
            run_tape[row + 1] = 0.0f;
            run_tape[row + 2] = h[h_base + n];
        } else {
            /* T-major tape: [b][d][n][t+1] — lane stride over t is one
             * element, so warp stores/loads coalesce (the old [b][t][d][n]
             * layout put every lane in its own 32-byte sector). Layout is a
             * property of the PARALLEL route; the sequential kernels keep
             * the historical layout. */
            int hs_idx = ((bid * d_inner + did) * d_state + n) * (T + 1) + 0;
            h_saved[hs_idx] = h[h_base + n];
        }
    }

    // Initialize running prefix to identity (1, 0) for each state dimension
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {
        smem_run_a[n] = 1.0f;
        smem_run_b[n] = 0.0f;
    }
    __syncthreads();

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int chunk_start = chunk * CHUNK_SIZE;

        // Direct per-thread loads. In global memory adjacent t are d_inner
        // elements apart, so threads of one block (fixed did) never share
        // a sector whichever thread issues the load; the merging happens
        // across blocks, where neighbouring did hit adjacent addresses at
        // each t. Staging the chunk through shared memory only added
        // barriers.
        float delta_vals[NITEMS];
        float u_vals[NITEMS];
        float delta_u_vals[NITEMS];
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            int t = chunk_start + threadIdx.x * NITEMS + i;
            if (t < T) {
                float raw = delta_raw[(bid * T + t) * d_inner + did];
                float sp = (raw > 20.0f) ? raw
                                         : log1pf(exp2f(raw * 1.4426950408889634f));
                delta_saved[(bid * T + t) * d_inner + did] = sp;
                delta_vals[i] = sp;
            } else {
                delta_vals[i] = 0.0f;
            }
            u_vals[i] = (t < T) ? u[(bid * T + t) * d_inner + did] : 0.0f;
            delta_u_vals[i] = delta_vals[i] * u_vals[i];
        }

        // Initialize output accumulator: y = D * u
        float out_vals[NITEMS];
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            out_vals[i] = D_d * u_vals[i];
        }

        // Outer loop over d_state (sequential, like Tri Dao original)
        for (int n = 0; n < d_state; n++) {
            // Pre-multiply a_neg by LOG2E so we can use exp2f directly,
            // saving one FMUL per (t, d, n) triple.
            float a_dn = a_neg[did * d_state + n] * LOG2E;

            // Build (da, db) pairs for this state dimension. B is read
            // directly (barrier diet — see the delta/u note above).
            float thread_a[NITEMS];
            float thread_b[NITEMS];
            float b_row[NITEMS];
            load_row8(B + (bid * d_state + n) * T,
                      chunk_start + threadIdx.x * NITEMS, T, b_row);

            #pragma unroll
            for (int i = 0; i < NITEMS; i++) {
                int t = chunk_start + threadIdx.x * NITEMS + i;
                if (t < T) {
                    // a_dn already has LOG2E folded in, so exp2f gives exp(delta*a)
                    float da = exp2f(delta_vals[i] * a_dn);
                    float b_t = b_row[i];
                    thread_a[i] = da;
                    thread_b[i] = delta_u_vals[i] * b_t;
                    // No da saved: backward recomputes da from delta
                    // and a_neg (bandwidth win).
                } else {
                    thread_a[i] = 1.0f;  // identity
                    thread_b[i] = 0.0f;
                }
            }

            // Thread-local sequential scan of NITEMS (a, b) pairs
            #pragma unroll
            for (int i = 1; i < NITEMS; i++) {
                thread_b[i] = thread_a[i] * thread_b[i - 1] + thread_b[i];
                thread_a[i] = thread_a[i] * thread_a[i - 1];
            }

            // Block-level inclusive scan of per-thread totals
            float scan_a = thread_a[NITEMS - 1];
            float scan_b = thread_b[NITEMS - 1];

            __syncthreads();
            block_inclusive_scan_ab(scan_a, scan_b, smem_wa, smem_wb);

            // Exclusive prefix of thread t = inclusive value of thread t-1:
            // the lane above hands it down through a shuffle; a warp's first
            // lane takes it from the previous warp's last lane through a
            // per-warp slot; thread 0 takes the identity. The last slot is
            // the block total thread 0 folds into the running prefix. The
            // values are the very registers the full exchange used to copy.
            // Every warp reads the running prefix before the barrier, so
            // thread 0 may overwrite it right after.
            if ((threadIdx.x & 31) == 31) {
                smem_exch_a[threadIdx.x >> 5] = scan_a;
                smem_exch_b[threadIdx.x >> 5] = scan_b;
            }
            float run_a = smem_run_a[n];
            float run_b = smem_run_b[n];
            __syncthreads();
            float excl_a = __shfl_up_sync(0xffffffffu, scan_a, 1);
            float excl_b = __shfl_up_sync(0xffffffffu, scan_b, 1);
            if (threadIdx.x == 0) {
                excl_a = 1.0f;
                excl_b = 0.0f;
            } else if ((threadIdx.x & 31) == 0) {
                excl_a = smem_exch_a[(threadIdx.x >> 5) - 1];
                excl_b = smem_exch_b[(threadIdx.x >> 5) - 1];
            }
            // Slim tape: record this chunk's entry prefix (chunk 0's
            // identity row was written above). Single writer.
            if (slim_tape && chunk > 0 && threadIdx.x == 0) {
                int row =
                    ((bid * d_inner + did) * d_state + n) * 3 * n_chunks;
                run_tape[row + 3 * chunk + 0] = run_a;
                run_tape[row + 3 * chunk + 1] = run_b;
            }

            // Initial state for this (b, d, n) triple
            float h_0 = h[h_base + n];

            // Running prefix for the next chunk: new_run = block_total o old_run
            if (threadIdx.x == 0) {
                float block_a = smem_exch_a[NWARPS - 1];
                float block_b = smem_exch_b[NWARPS - 1];
                smem_run_a[n] = block_a * run_a;
                smem_run_b[n] = block_a * run_b + block_b;
            }

            // Compute h[t] for each element and accumulate y[t] += h[t] * C[t,n]
            float c_row[NITEMS];
            load_row8(C + (bid * d_state + n) * T,
                      chunk_start + threadIdx.x * NITEMS, T, c_row);
            #pragma unroll
            for (int i = 0; i < NITEMS; i++) {
                int t = chunk_start + threadIdx.x * NITEMS + i;
                if (t < T) {
                    // Compose thread-local prefix with exclusive block prefix:
                    //   (comp_a, comp_b) = (thread_a[i], thread_b[i]) o (excl_a, excl_b)
                    float comp_a = thread_a[i] * excl_a;
                    float comp_b = thread_a[i] * excl_b + thread_b[i];

                    // Compose with inter-chunk running prefix:
                    //   (final_a, final_b) = (comp_a, comp_b) o (run_a, run_b)
                    float final_a = comp_a * run_a;
                    float final_b = comp_a * run_b + comp_b;

                    // h[t] = final_a * h_init + final_b
                    float h_t = final_a * h_0 + final_b;

                    // Save h for backward: the full tape stores every
                    // step; slim stores only the NEXT chunk's entry state
                    // (bit-exactly the value the backward's h_prev
                    // boundary read used to load from h_saved).
                    if (slim_tape) {
                        if ((t + 1) % CHUNK_SIZE == 0 && t + 1 < T) {
                            int row = ((bid * d_inner + did) * d_state + n)
                                * 3 * n_chunks;
                            run_tape[row + 3 * (chunk + 1) + 2] = h_t;
                        }
                    } else {
                        int hs_idx = ((bid * d_inner + did) * d_state + n)
                            * (T + 1) + (t + 1);
                        h_saved[hs_idx] = h_t;
                    }

                    // Single-pass Y accumulation (C read directly)
                    float c_t = c_row[i];
                    out_vals[i] += h_t * c_t;
                }
            }
        } // end d_state loop

        // Each thread stores its own items. Within a block did is fixed,
        // so neighbouring t are d_inner apart and every lane lands in its
        // own sector whichever thread issues the store; staging the chunk
        // through shared memory to restripe it bought nothing and cost
        // two barriers per chunk.
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            int t = chunk_start + threadIdx.x * NITEMS + i;
            if (t < T) {
                y_out[(bid * T + t) * d_inner + did] = out_vals[i];
            }
        }
    } // end chunk loop
    // Thread 0 wrote the last chunk's running prefix after the scan
    // barrier; every thread reads its state dimension below.
    __syncthreads();

    // Write final SSM state: h[n] = run_a[n] * h_init[n] + run_b[n]
    // Parallelized across threads.
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {
        float h_0 = h[h_base + n];
        h[h_base + n] = smem_run_a[n] * h_0 + smem_run_b[n];
    }
}

// ============================================================================
// Forward without saves (target network -- no backward needed).
//
// Same interface as ssm_burnin_forward_nosave.
// Same parallel scan algorithm, skips the h_saved writes.
// ============================================================================
extern "C" __global__ __launch_bounds__(NTHREADS, SCAN_MINB) void ssm_parallel_scan_fwd_nosave(
    float* __restrict__ h,             // [batch * d_inner * d_state] SSM state (mutated)
    float* __restrict__ y_out,         // [batch * T * d_inner] output
    const float* __restrict__ delta,   // [batch * T * d_inner]
    const float* __restrict__ u,       // [batch * T * d_inner]
    const float* __restrict__ B,       // [batch * T * d_state]
    const float* __restrict__ C,       // [batch * T * d_state]
    const float* __restrict__ a_neg,   // [d_inner * d_state]
    const float* __restrict__ D,       // [d_inner]
    const float* __restrict__ proj_gate, // in_proj output when gating fuses
    int gate_stride,                   // proj row stride; 0 = plain y store
    int batch, int T, int d_inner, int d_state
) {
    int bid = blockIdx.x;
    int did = blockIdx.y;
    if (bid >= batch || did >= d_inner) return;
    if (d_state > MAX_DSTATE) return;

    extern __shared__ float smem[];
    float *smem_wa     = smem + SMEM_WA_OFF;
    float *smem_wb     = smem + SMEM_WB_OFF;
    /* Runtime d_state stride: run/exchange/stage regions pack at the actual
     * d_state instead of MAX_DSTATE (the launcher shrinks the allocation to
     * match). Region ORDER and contents are unchanged, so every cell holds
     * the same value as the padded layout - address-only. */
    float *smem_run_a  = smem + 2 * NWARPS;
    float *smem_run_b  = smem_run_a + d_state;
    float *smem_exch_a = smem_run_b + d_state;
    float *smem_exch_b = smem_exch_a + NWARPS;

    float D_d = D[did];
    int h_base = (bid * d_inner + did) * d_state;

    // Initialize running prefix to identity
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {
        smem_run_a[n] = 1.0f;
        smem_run_b[n] = 0.0f;
    }
    __syncthreads();

    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int chunk_start = chunk * CHUNK_SIZE;

        // The shared-memory staging of delta, u, B and C was value-neutral
        // (the same elements in the same order) and its lane stride
        // exceeded the 32-byte sector either way, so direct loads drop the
        // staging barriers with no arithmetic change, as in the saving
        // twin. y keeps its staging, the one genuinely coalescing store.
        float delta_vals[NITEMS];
        float u_vals[NITEMS];
        float delta_u_vals[NITEMS];
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            int t = chunk_start + threadIdx.x * NITEMS + i;
            delta_vals[i] = (t < T) ? delta[(bid * T + t) * d_inner + did] : 0.0f;
            u_vals[i] = (t < T) ? u[(bid * T + t) * d_inner + did] : 0.0f;
            delta_u_vals[i] = delta_vals[i] * u_vals[i];
        }

        float out_vals[NITEMS];
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            out_vals[i] = D_d * u_vals[i];
        }

        for (int n = 0; n < d_state; n++) {
            // Pre-multiply a_neg by LOG2E so we can use exp2f directly,
            // saving one FMUL per (t, d, n) triple.
            float a_dn = a_neg[did * d_state + n] * LOG2E;

            float thread_a[NITEMS];
            float thread_b[NITEMS];
            float b_row[NITEMS];
            load_row8(B + (bid * d_state + n) * T,
                      chunk_start + threadIdx.x * NITEMS, T, b_row);

            #pragma unroll
            for (int i = 0; i < NITEMS; i++) {
                int t = chunk_start + threadIdx.x * NITEMS + i;
                if (t < T) {
                    // a_dn already has LOG2E folded in, so exp2f gives exp(delta*a)
                    float da = exp2f(delta_vals[i] * a_dn);
                    float b_t = b_row[i];
                    thread_a[i] = da;
                    thread_b[i] = delta_u_vals[i] * b_t;
                } else {
                    thread_a[i] = 1.0f;
                    thread_b[i] = 0.0f;
                }
            }

            #pragma unroll
            for (int i = 1; i < NITEMS; i++) {
                thread_b[i] = thread_a[i] * thread_b[i - 1] + thread_b[i];
                thread_a[i] = thread_a[i] * thread_a[i - 1];
            }

            float scan_a = thread_a[NITEMS - 1];
            float scan_b = thread_b[NITEMS - 1];

            __syncthreads();
            block_inclusive_scan_ab(scan_a, scan_b, smem_wa, smem_wb);

            // Exclusive prefix through the lane above and a per-warp slot,
            // as in the saving kernel.
            if ((threadIdx.x & 31) == 31) {
                smem_exch_a[threadIdx.x >> 5] = scan_a;
                smem_exch_b[threadIdx.x >> 5] = scan_b;
            }
            float run_a = smem_run_a[n];
            float run_b = smem_run_b[n];
            __syncthreads();
            float excl_a = __shfl_up_sync(0xffffffffu, scan_a, 1);
            float excl_b = __shfl_up_sync(0xffffffffu, scan_b, 1);
            if (threadIdx.x == 0) {
                excl_a = 1.0f;
                excl_b = 0.0f;
            } else if ((threadIdx.x & 31) == 0) {
                excl_a = smem_exch_a[(threadIdx.x >> 5) - 1];
                excl_b = smem_exch_b[(threadIdx.x >> 5) - 1];
            }
            float h_0 = h[h_base + n];

            if (threadIdx.x == 0) {
                float block_a = smem_exch_a[NWARPS - 1];
                float block_b = smem_exch_b[NWARPS - 1];
                smem_run_a[n] = block_a * run_a;
                smem_run_b[n] = block_a * run_b + block_b;
            }

            float c_row[NITEMS];
            load_row8(C + (bid * d_state + n) * T,
                      chunk_start + threadIdx.x * NITEMS, T, c_row);
            #pragma unroll
            for (int i = 0; i < NITEMS; i++) {
                int t = chunk_start + threadIdx.x * NITEMS + i;
                if (t < T) {
                    float comp_a = thread_a[i] * excl_a;
                    float comp_b = thread_a[i] * excl_b + thread_b[i];
                    float final_a = comp_a * run_a;
                    float final_b = comp_a * run_b + comp_b;
                    float h_t = final_a * h_0 + final_b;

                    float c_t = c_row[i];
                    out_vals[i] += h_t * c_t;
                }
            }
        } // end d_state loop

        // Direct per-thread stores, as in the saving kernel.
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            int t = chunk_start + threadIdx.x * NITEMS + i;
            if (t < T) {
                float yv = out_vals[i];
                if (gate_stride > 0) {
                    // Fused gating: the same one-rounding product the
                    // separate elementwise mul performed, on the same
                    // SiLU formula split_gate_silu used - bit-identical
                    // to the three-kernel chain it replaces.
                    float g = proj_gate[(bid * T + t) * gate_stride
                                        + d_inner + did];
                    yv *= g / (1.0f + exp2f(-g * 1.4426950408889634f));
                }
                y_out[(bid * T + t) * d_inner + did] = yv;
            }
        }
    } // end chunk loop
    // Thread 0 wrote the last chunk's running prefix after the scan
    // barrier; every thread reads its state dimension below.
    __syncthreads();

    // Write final SSM state (parallelized across threads)
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {
        float h_0 = h[h_base + n];
        h[h_base + n] = smem_run_a[n] * h_0 + smem_run_b[n];
    }
}

// ============================================================================
// Typed variants (bf16/f16) — the mixed-precision training tier.
//
// Follow `state-spaces/mamba`'s `scan_t = float2` discipline: all scan
// state + running prefix + block scan + registers stay f32. Only the
// activation I/O tensors (delta, u, B, C, y_out) become typed. BPTT
// state (`h`, `h_saved`), model parameters (`a_neg`, `D`),
// and ALL `smem_*` remain f32.
// ============================================================================

#define DEFINE_SSM_PARALLEL_SCAN_FWD(SUFFIX, T_ACT, FROM_F)                   \
extern "C" __global__ __launch_bounds__(NTHREADS, SCAN_MINB) void                          \
ssm_parallel_scan_fwd_##SUFFIX(                                               \
    float* __restrict__ h,                                                    \
    T_ACT* __restrict__ y_out,                                                \
    float* __restrict__ h_saved,                                              \
    const T_ACT* __restrict__ delta_raw,                                      \
    T_ACT* __restrict__ delta_saved,                                          \
    const T_ACT* __restrict__ u,                                              \
    const T_ACT* __restrict__ B,                                              \
    const T_ACT* __restrict__ C,                                              \
    const float* __restrict__ a_neg,                                          \
    const float* __restrict__ D,                                              \
    int batch, int T, int d_inner, int d_state,                               \
    /* Slim tape (no __restrict__: aliases h_saved under slim) */             \
    float* run_tape,                                                          \
    int slim_tape                                                             \
) {                                                                           \
    int bid = blockIdx.x;                                                     \
    int did = blockIdx.y;                                                     \
    if (bid >= batch || did >= d_inner) return;                               \
    if (d_state > MAX_DSTATE) return;                                         \
    extern __shared__ float smem[];                                           \
    float *smem_wa     = smem + SMEM_WA_OFF;                                  \
    float *smem_wb     = smem + SMEM_WB_OFF;                                  \
    /* Runtime d_state stride - see the f32 twin. */                          \
    float *smem_run_a  = smem + 2 * NWARPS;                                   \
    float *smem_run_b  = smem_run_a + d_state;                                \
    float *smem_exch_a = smem_run_b + d_state;                                \
    float *smem_exch_b = smem_exch_a + NWARPS;                                \
    float D_d = D[did];                                                       \
    int h_base = (bid * d_inner + did) * d_state;                             \
    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;                         \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        if (slim_tape) {                                                      \
            int row = ((bid * d_inner + did) * d_state + n) * 3 * n_chunks;   \
            run_tape[row + 0] = 1.0f;                                         \
            run_tape[row + 1] = 0.0f;                                         \
            run_tape[row + 2] = h[h_base + n];                                \
        } else {                                                              \
            /* T-major tape (see the plain fwd note). */                      \
            int hs_idx =                                                      \
                ((bid * d_inner + did) * d_state + n) * (T + 1) + 0;          \
            h_saved[hs_idx] = h[h_base + n];                                  \
        }                                                                     \
    }                                                                         \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        smem_run_a[n] = 1.0f;                                                 \
        smem_run_b[n] = 0.0f;                                                 \
    }                                                                         \
    __syncthreads();                                                          \
    for (int chunk = 0; chunk < n_chunks; chunk++) {                          \
        int chunk_start = chunk * CHUNK_SIZE;                                 \
        /* Barrier diet: the smem staging round trips for delta/u/B/C are     \
         * value-neutral (same elements, same to_f) and their lane stride     \
         * exceeded the 32-byte sector either way - direct loads drop the     \
         * staging barriers with zero arithmetic change. y keeps its          \
         * staging (the one genuinely coalescing store). */                   \
        float delta_vals[NITEMS];                                             \
        float u_vals[NITEMS];                                                 \
        float delta_u_vals[NITEMS];                                           \
        _Pragma("unroll")                                                     \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            if (t < T) {                                                      \
                /* Inline softplus through the exact store rounding the    \
                   deleted copy pass produced; the scan consumes the       \
                   round-tripped value, bit-equal to reading the save. */  \
                float raw = to_f(delta_raw[(bid * T + t) * d_inner + did]); \
                float sp = (raw > 20.0f)                                    \
                    ? raw                                                   \
                    : log1pf(exp2f(raw * 1.4426950408889634f));             \
                T_ACT spt = FROM_F(sp);                                     \
                delta_saved[(bid * T + t) * d_inner + did] = spt;           \
                delta_vals[i] = to_f(spt);                                  \
            } else {                                                        \
                delta_vals[i] = 0.0f;                                       \
            }                                                               \
            u_vals[i] = (t < T) ? to_f(u[(bid * T + t) * d_inner + did])      \
                                : 0.0f;                                       \
            delta_u_vals[i] = delta_vals[i] * u_vals[i];                      \
        }                                                                     \
        float out_vals[NITEMS];                                               \
                                                             \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            out_vals[i] = D_d * u_vals[i];                                    \
        }                                                                     \
        for (int n = 0; n < d_state; n++) {                                   \
            float a_dn = a_neg[did * d_state + n] * LOG2E;                    \
            float thread_a[NITEMS];                                           \
            float thread_b[NITEMS];                                           \
            float b_row[NITEMS];                                              \
            load_row8(B + (bid * d_state + n) * T,                            \
                      chunk_start + threadIdx.x * NITEMS, T, b_row);          \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    float da = exp2f(delta_vals[i] * a_dn);                   \
                    float b_t = b_row[i];                                     \
                    thread_a[i] = da;                                         \
                    thread_b[i] = delta_u_vals[i] * b_t;                      \
                } else {                                                      \
                    thread_a[i] = 1.0f;                                       \
                    thread_b[i] = 0.0f;                                       \
                }                                                             \
            }                                                                 \
                                                             \
            _Pragma("unroll")                                                  \
            for (int i = 1; i < NITEMS; i++) {                                \
                thread_b[i] = thread_a[i] * thread_b[i - 1] + thread_b[i];    \
                thread_a[i] = thread_a[i] * thread_a[i - 1];                  \
            }                                                                 \
            float scan_a = thread_a[NITEMS - 1];                              \
            float scan_b = thread_b[NITEMS - 1];                              \
            __syncthreads();                                                  \
            block_inclusive_scan_ab(scan_a, scan_b, smem_wa, smem_wb);        \
            /* Exclusive prefix through the lane above and a per-warp slot,  \
               as in the f32 kernel; the last slot is the block total. */     \
            if ((threadIdx.x & 31) == 31) {                                   \
                smem_exch_a[threadIdx.x >> 5] = scan_a;                       \
                smem_exch_b[threadIdx.x >> 5] = scan_b;                       \
            }                                                                 \
            float run_a = smem_run_a[n];                                      \
            float run_b = smem_run_b[n];                                      \
            __syncthreads();                                                  \
            float excl_a = __shfl_up_sync(0xffffffffu, scan_a, 1);            \
            float excl_b = __shfl_up_sync(0xffffffffu, scan_b, 1);            \
            if (threadIdx.x == 0) {                                           \
                excl_a = 1.0f;                                                \
                excl_b = 0.0f;                                                \
            } else if ((threadIdx.x & 31) == 0) {                             \
                excl_a = smem_exch_a[(threadIdx.x >> 5) - 1];                 \
                excl_b = smem_exch_b[(threadIdx.x >> 5) - 1];                 \
            }                                                                 \
            /* Slim tape: chunk-entry prefix (single writer). */              \
            if (slim_tape && chunk > 0 && threadIdx.x == 0) {                 \
                int row =                                                     \
                    ((bid * d_inner + did) * d_state + n) * 3 * n_chunks;     \
                run_tape[row + 3 * chunk + 0] = run_a;                        \
                run_tape[row + 3 * chunk + 1] = run_b;                        \
            }                                                                 \
            float h_0 = h[h_base + n];                                        \
            if (threadIdx.x == 0) {                                           \
                float block_a = smem_exch_a[NWARPS - 1];                      \
                float block_b = smem_exch_b[NWARPS - 1];                      \
                smem_run_a[n] = block_a * run_a;                              \
                smem_run_b[n] = block_a * run_b + block_b;                    \
            }                                                                 \
            float c_row[NITEMS];                                              \
            load_row8(C + (bid * d_state + n) * T,                            \
                      chunk_start + threadIdx.x * NITEMS, T, c_row);          \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    float comp_a = thread_a[i] * excl_a;                      \
                    float comp_b = thread_a[i] * excl_b + thread_b[i];        \
                    float final_a = comp_a * run_a;                           \
                    float final_b = comp_a * run_b + comp_b;                  \
                    float h_t = final_a * h_0 + final_b;                      \
                    if (slim_tape) {                                          \
                        if ((t + 1) % CHUNK_SIZE == 0 && t + 1 < T) {         \
                            int row =                                         \
                                ((bid * d_inner + did) * d_state + n)         \
                                * 3 * n_chunks;                               \
                            run_tape[row + 3 * (chunk + 1) + 2] = h_t;        \
                        }                                                     \
                    } else {                                                  \
                        int hs_idx = ((bid * d_inner + did) * d_state + n)    \
                                     * (T + 1) + (t + 1);                     \
                        h_saved[hs_idx] = h_t;                                \
                    }                                                         \
                    float c_t = c_row[i];                                     \
                    out_vals[i] += h_t * c_t;                                 \
                }                                                             \
            }                                                                 \
        }                                                                     \
                                                             \
        /* Direct per-thread stores, as in the f32 kernel. */                 \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            if (t < T) {                                                      \
                y_out[(bid * T + t) * d_inner + did] = FROM_F(out_vals[i]);   \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    /* The last running-prefix write by thread 0 precedes the reads below. */ \
    __syncthreads();                                                          \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        float h_0 = h[h_base + n];                                            \
        h[h_base + n] = smem_run_a[n] * h_0 + smem_run_b[n];                  \
    }                                                                         \
}

DEFINE_SSM_PARALLEL_SCAN_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_PARALLEL_SCAN_FWD(f16,  __half,        from_f_f16)

#define DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(SUFFIX, T_ACT, FROM_F)            \
extern "C" __global__ __launch_bounds__(NTHREADS, SCAN_MINB) void                          \
ssm_parallel_scan_fwd_nosave_##SUFFIX(                                        \
    float* __restrict__ h,                                                    \
    T_ACT* __restrict__ y_out,                                                \
    const T_ACT* __restrict__ delta,                                          \
    const T_ACT* __restrict__ u,                                              \
    const T_ACT* __restrict__ B,                                              \
    const T_ACT* __restrict__ C,                                              \
    const float* __restrict__ a_neg,                                          \
    const float* __restrict__ D,                                              \
    const T_ACT* __restrict__ proj_gate,                                      \
    int gate_stride,                                                          \
    int batch, int T, int d_inner, int d_state                                \
) {                                                                           \
    int bid = blockIdx.x;                                                     \
    int did = blockIdx.y;                                                     \
    if (bid >= batch || did >= d_inner) return;                               \
    if (d_state > MAX_DSTATE) return;                                         \
    extern __shared__ float smem[];                                           \
    float *smem_wa     = smem + SMEM_WA_OFF;                                  \
    float *smem_wb     = smem + SMEM_WB_OFF;                                  \
    /* Runtime d_state stride - see the f32 twin. */                          \
    float *smem_run_a  = smem + 2 * NWARPS;                                   \
    float *smem_run_b  = smem_run_a + d_state;                                \
    float *smem_exch_a = smem_run_b + d_state;                                \
    float *smem_exch_b = smem_exch_a + NWARPS;                                \
    float D_d = D[did];                                                       \
    int h_base = (bid * d_inner + did) * d_state;                             \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        smem_run_a[n] = 1.0f;                                                 \
        smem_run_b[n] = 0.0f;                                                 \
    }                                                                         \
    __syncthreads();                                                          \
    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;                         \
    for (int chunk = 0; chunk < n_chunks; chunk++) {                          \
        int chunk_start = chunk * CHUNK_SIZE;                                 \
        /* Direct loads instead of the staging round trips: the same      \
         * elements through the same to_f, so the values do not change,   \
         * and the staging barriers go with them (as in the saving twin). \
         * y keeps its staging, the one genuinely coalescing store. */     \
        float delta_vals[NITEMS];                                             \
        float u_vals[NITEMS];                                                 \
        float delta_u_vals[NITEMS];                                           \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            delta_vals[i] = (t < T)                                           \
                ? to_f(delta[(bid * T + t) * d_inner + did]) : 0.0f;          \
            u_vals[i] = (t < T) ? to_f(u[(bid * T + t) * d_inner + did])      \
                                : 0.0f;                                       \
            delta_u_vals[i] = delta_vals[i] * u_vals[i];                      \
        }                                                                     \
        float out_vals[NITEMS];                                               \
                                                             \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            out_vals[i] = D_d * u_vals[i];                                    \
        }                                                                     \
        for (int n = 0; n < d_state; n++) {                                   \
            float a_dn = a_neg[did * d_state + n] * LOG2E;                    \
            float thread_a[NITEMS];                                           \
            float thread_b[NITEMS];                                           \
            float b_row[NITEMS];                                              \
            load_row8(B + (bid * d_state + n) * T,                            \
                      chunk_start + threadIdx.x * NITEMS, T, b_row);          \
                                                             \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    float da = exp2f(delta_vals[i] * a_dn);                   \
                    float b_t = b_row[i];                                     \
                    thread_a[i] = da;                                         \
                    thread_b[i] = delta_u_vals[i] * b_t;                      \
                } else {                                                      \
                    thread_a[i] = 1.0f;                                       \
                    thread_b[i] = 0.0f;                                       \
                }                                                             \
            }                                                                 \
                                                             \
            _Pragma("unroll")                                                  \
            for (int i = 1; i < NITEMS; i++) {                                \
                thread_b[i] = thread_a[i] * thread_b[i - 1] + thread_b[i];    \
                thread_a[i] = thread_a[i] * thread_a[i - 1];                  \
            }                                                                 \
            float scan_a = thread_a[NITEMS - 1];                              \
            float scan_b = thread_b[NITEMS - 1];                              \
            __syncthreads();                                                  \
            block_inclusive_scan_ab(scan_a, scan_b, smem_wa, smem_wb);        \
            /* Exclusive prefix through the lane above and a per-warp slot,  \
               as in the f32 kernel; the last slot is the block total. */     \
            if ((threadIdx.x & 31) == 31) {                                   \
                smem_exch_a[threadIdx.x >> 5] = scan_a;                       \
                smem_exch_b[threadIdx.x >> 5] = scan_b;                       \
            }                                                                 \
            float run_a = smem_run_a[n];                                      \
            float run_b = smem_run_b[n];                                      \
            __syncthreads();                                                  \
            float excl_a = __shfl_up_sync(0xffffffffu, scan_a, 1);            \
            float excl_b = __shfl_up_sync(0xffffffffu, scan_b, 1);            \
            if (threadIdx.x == 0) {                                           \
                excl_a = 1.0f;                                                \
                excl_b = 0.0f;                                                \
            } else if ((threadIdx.x & 31) == 0) {                             \
                excl_a = smem_exch_a[(threadIdx.x >> 5) - 1];                 \
                excl_b = smem_exch_b[(threadIdx.x >> 5) - 1];                 \
            }                                                                 \
            float h_0 = h[h_base + n];                                        \
            if (threadIdx.x == 0) {                                           \
                float block_a = smem_exch_a[NWARPS - 1];                      \
                float block_b = smem_exch_b[NWARPS - 1];                      \
                smem_run_a[n] = block_a * run_a;                              \
                smem_run_b[n] = block_a * run_b + block_b;                    \
            }                                                                 \
            float c_row[NITEMS];                                              \
            load_row8(C + (bid * d_state + n) * T,                            \
                      chunk_start + threadIdx.x * NITEMS, T, c_row);          \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    float comp_a = thread_a[i] * excl_a;                      \
                    float comp_b = thread_a[i] * excl_b + thread_b[i];        \
                    float final_a = comp_a * run_a;                           \
                    float final_b = comp_a * run_b + comp_b;                  \
                    float h_t = final_a * h_0 + final_b;                      \
                    float c_t = c_row[i];                                     \
                    out_vals[i] += h_t * c_t;                                 \
                }                                                             \
            }                                                                 \
        }                                                                     \
                                                             \
        /* Direct per-thread stores, as in the f32 kernel. */                 \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            if (t < T) {                                                      \
                T_ACT ty = FROM_F(out_vals[i]);                               \
                if (gate_stride > 0) {                                        \
                    /* Round-trip emulation of the replaced chain: the     \
                     * baseline stored y typed, stored SiLU(gate) typed,   \
                     * then multiplied the reloaded values with one final  \
                     * rounding - reproduce each rounding in place.     */ \
                    float g = to_f(proj_gate[(bid * T + t) * gate_stride    \
                                             + d_inner + did]);              \
                    T_ACT tg = FROM_F(                                       \
                        g / (1.0f + exp2f(-g * 1.4426950408889634f)));       \
                    ty = FROM_F(to_f(ty) * to_f(tg));                        \
                }                                                             \
                y_out[(bid * T + t) * d_inner + did] = ty;                    \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    /* The last running-prefix write by thread 0 precedes the reads below. */ \
    __syncthreads();                                                          \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        float h_0 = h[h_base + n];                                            \
        h[h_base + n] = smem_run_a[n] * h_0 + smem_run_b[n];                  \
    }                                                                         \
}

DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(f16,  __half,        from_f_f16)

// ============================================================================
// Parallel selective-scan BACKWARD pass.
//
// Mirrors state-spaces/mamba `selective_scan_bwd_kernel.cuh`.
// Grid: (batch, d_inner). Block: NTHREADS=128 (1 block per (b, di)).
//
// Algorithm (per (b, di)):
//   For each chunk in REVERSE (n_chunks-1 → 0):
//     Coalesced typed load delta/u/dy → smem → registers (per-thread NITEMS)
//     For each n in [0, d_state):  (sequential outer loop)
//       Coalesced typed load B[*, n], C[*, n] → smem → registers
//       Per-i: da[i] = exp2f(δ·a_neg·LOG2E),  d_local[i] = dy[i]·c[i]
//       Build reverse-scan pair_t = (a_{t+1}, d_local[t]):
//         intra-thread: pair[i].a = da[i+1] for i < NITEMS-1
//         inter-thread: pair[NITEMS-1].a = next thread's da[0] (smem exch)
//         inter-chunk:  last thread's pair.a = postfix-saved next-chunk a
//         globally last: pair.a = 1.0 (no future)
//       Per-thread compose NITEMS pairs → block_inclusive_REVERSE_scan_ab
//       Compose with running_postfix → dh[i] for each timestep
//       Per-i outputs (typed acts loaded from h_saved, b_t, etc.):
//         d_C_local[btdn] = dy * h_saved[t+1]    (typed, store FROM_F)
//         d_B_local[btdn] = dh * δ * u           (typed)
//         d_delta_acc[i] += dh * (a·da·h_prev + u·b)  (register f32)
//         d_u_acc[i]     += dh * δ · b                  (register f32)
//         d_a_per_thread += dh · da · δ · a · h_prev    (register f32)
//       Block-reduce d_a_per_thread → thread 0 += d_a_log_local[bid·di·ds + did·ds + n]
//       Update running_postfix via block_inclusive_reverse_scan_ab tail
//     Store d_delta_acc, d_u_acc to typed HBM (smem coalesced + downcast)
//   Final: d_D_local[bid·d_inner + did] = local_d_D
//
// All scan state, h_saved, registers stay f32 (BPTT scan_t = float2 invariant).
// Outputs follow the existing _local convention so the existing reduction
// kernels (reduce_d_B, reduce_d_C, reduce_d_D, reduce_d_a_log) work unchanged.
// ============================================================================

#define DEFINE_SSM_PARALLEL_SCAN_BWD(SUFFIX, T_ACT, FROM_F)                   \
extern "C" __global__ __launch_bounds__(NTHREADS, SCAN_MINB) void                          \
ssm_parallel_scan_bwd_##SUFFIX(                                               \
    const float* __restrict__ h_saved,    /* [B*(T+1)*di*ds] */               \
    const T_ACT* __restrict__ delta,                                          \
    const T_ACT* __restrict__ u,                                              \
    const T_ACT* __restrict__ B_in,                                           \
    const T_ACT* __restrict__ C_in,                                           \
    const float* __restrict__ a_neg,      /* [di*ds] */                       \
    const float* __restrict__ D,          /* [di] */                          \
    const T_ACT* __restrict__ dy,         /* [B*T*di] */                      \
    T_ACT* __restrict__ d_delta,          /* [B*T*di] */                      \
    T_ACT* __restrict__ d_u,              /* [B*T*di] */                      \
    T_ACT* __restrict__ d_B_local,        /* [B*T*di*ds] */                   \
    T_ACT* __restrict__ d_C_local,        /* [B*T*di*ds] */                   \
    float* __restrict__ d_D_local,        /* [B*di] f32 master */             \
    float* __restrict__ d_a_log_local,    /* [B*di*ds] f32 master */          \
    int batch, int T, int d_inner, int d_state,                               \
    /* Slim tape: (run_a, run_b, h_entry) per (b,d,n,chunk). */               \
    const float* run_tape,                                                    \
    int slim_tape                                                             \
) {                                                                           \
    int bid = blockIdx.x;                                                     \
    int did = blockIdx.y;                                                     \
    if (bid >= batch || did >= d_inner) return;                               \
    if (d_state > MAX_DSTATE) return;                                         \
    extern __shared__ float smem[];                                           \
    /* smem_wa/wb/run_a/run_b are forward-layout (small NWARPS-sized warp     \
       totals + MAX_DSTATE running prefix) — UNUSED in the bwd kernel.       \
       smem_exch_a/b are NTHREADS-sized — we repurpose them as the per-      \
       thread inclusive-reverse-scan postfix tile (read by lane k as        \
       smem_exch_*[k+1] for the exclusive-next postfix). */                  \
    float *smem_rev_wa  = smem + SMEM_REV_WA_OFF;                             \
    float *smem_rev_wb  = smem + SMEM_REV_WB_OFF;                             \
    float *smem_exch_a  = smem + SMEM_EXCH_A_OFF;                             \
    float *smem_exch_b  = smem + SMEM_EXCH_B_OFF;                             \
    float *smem_post_a  = smem + SMEM_POST_A_OFF;                             \
    float *smem_post_b  = smem + SMEM_POST_B_OFF;                             \
    float *smem_next_a  = smem + SMEM_NEXT_A_OFF;                             \
    float *smem_da_red  = smem + SMEM_DA_RED_OFF;                             \
    float *smem_chunk_first_a = smem + SMEM_CHUNK_FIRST_A_OFF;                \
    T_ACT *smem_stage   = (T_ACT *)(smem + SMEM_STAGE_OFF);                   \
    /* Replay scratch (slim tape): the fwd-layout regions are                 \
       unused in this kernel - smem_wa/wb feed the forward                    \
       block_inclusive_scan_ab, the RUN_A slot holds the exclusive            \
       prefix exchange (NTHREADS) plus the chunk-boundary H lane              \
       (NTHREADS more; MAX_DSTATE = 256 fits both), RUN_B the                 \
       b-half of the exchange. */                                             \
    float *smem_fwd_wa  = smem + SMEM_WA_OFF;                                 \
    float *smem_fwd_wb  = smem + SMEM_WB_OFF;                                 \
    float *smem_fexch_a = smem + SMEM_RUN_A_OFF;                              \
    float *smem_fexch_b = smem + SMEM_RUN_B_OFF;                              \
    float *smem_hbound  = smem + SMEM_RUN_A_OFF + NTHREADS;                   \
    float D_d = D[did];                                                       \
    /* T-major tape: per-(b,d) row base; +n*(T+1) selects the state
     * lane's contiguous t-run. */                                            \
    int hsave_row_bd = (bid * d_inner + did) * d_state;                       \
    /* Initialize inter-chunk reverse-scan postfix to identity (1, 0). The   \
       chunk_first_a buffer is set to 1.0 to act as `a_{t+1}=1` for the      \
       very-last timestep of the very-last chunk (no future). */              \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        smem_post_a[n] = 1.0f;                                                \
        smem_post_b[n] = 0.0f;                                                \
        smem_chunk_first_a[n] = 1.0f;                                         \
    }                                                                         \
    __syncthreads();                                                          \
    float local_d_D = 0.0f;                                                   \
    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;                         \
    for (int chunk_loop = 0; chunk_loop < n_chunks; chunk_loop++) {           \
        int chunk = n_chunks - 1 - chunk_loop;  /* walk REVERSE */            \
        int chunk_start = chunk * CHUNK_SIZE;                                 \
        /* Barrier diet: delta/u/dy read directly (value-neutral; the         \
         * staging's lane stride exceeded the 32-byte sector either way). */  \
        float delta_vals[NITEMS];                                             \
        float u_vals[NITEMS];                                                 \
        float dy_vals[NITEMS];                                                \
        _Pragma("unroll")                                                     \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            delta_vals[i] =                                                   \
                (t < T) ? to_f(delta[(bid * T + t) * d_inner + did]) : 0.0f;  \
            u_vals[i] = (t < T) ? to_f(u[(bid * T + t) * d_inner + did])      \
                                : 0.0f;                                       \
            dy_vals[i] = (t < T) ? to_f(dy[(bid * T + t) * d_inner + did])    \
                                 : 0.0f;                                      \
        }                                                                     \
        /* ---- Per-t skip-path contributions accumulate in registers ---- */ \
        float d_u_acc[NITEMS];                                                \
        float d_delta_acc[NITEMS];                                            \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            if (t < T) {                                                      \
                local_d_D += dy_vals[i] * u_vals[i];                          \
                d_u_acc[i] = dy_vals[i] * D_d;                                \
            } else {                                                          \
                d_u_acc[i] = 0.0f;                                            \
            }                                                                 \
            d_delta_acc[i] = 0.0f;                                            \
        }                                                                     \
        /* ---- Outer d_state loop ---- */                                    \
        for (int n = 0; n < d_state; n++) {                                   \
            float a_dn = a_neg[did * d_state + n];                            \
            float a_dn_log2 = a_dn * LOG2E;                                   \
            float b_vals[NITEMS];                                             \
            load_row8(B_in + (bid * d_state + n) * T,                         \
                      chunk_start + threadIdx.x * NITEMS, T, b_vals);         \
            float c_vals[NITEMS];                                             \
            load_row8(C_in + (bid * d_state + n) * T,                         \
                      chunk_start + threadIdx.x * NITEMS, T, c_vals);         \
            /* Per-i: da[i] = exp2(delta * a_neg), d_local[i] = dy * c */     \
            float da_vals[NITEMS];                                            \
            float d_local[NITEMS];                                            \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    da_vals[i] = exp2f(delta_vals[i] * a_dn_log2);            \
                    d_local[i] = dy_vals[i] * c_vals[i];                      \
                } else {                                                      \
                    da_vals[i] = 1.0f;                                        \
                    d_local[i] = 0.0f;                                        \
                }                                                             \
            }                                                                 \
            /* Exchange: each thread publishes its first da into smem so the  \
               left-neighbor thread can read it as its (NITEMS-1).a (the      \
               "next-step a" trick — Tri Dao reverse_scan). */                \
            /* Slim-tape replay: reproduce the forward's h_t for              \
               this chunk BIT-exactly - the same thread-local scan,           \
               the same block_inclusive_scan_ab, the same compose             \
               chain ((comp o run) applied to h_0) on the same                \
               inputs. The full-tape path keeps its h_saved reads. */         \
            float H_vals[NITEMS];                                             \
            float h_prev_boundary = 0.0f;                                     \
            if (slim_tape) {                                                  \
                int row = (hsave_row_bd + n) * 3 * n_chunks;                  \
                float f_run_a = run_tape[row + 3 * chunk + 0];                \
                float f_run_b = run_tape[row + 3 * chunk + 1];                \
                float f_hentry = run_tape[row + 3 * chunk + 2];               \
                float f_h0 = run_tape[row + 2];                               \
                float fwd_a[NITEMS];                                          \
                float fwd_b[NITEMS];                                          \
                _Pragma("unroll")                                             \
                for (int i = 0; i < NITEMS; i++) {                            \
                    int t = chunk_start + threadIdx.x * NITEMS + i;           \
                    if (t < T) {                                              \
                        fwd_a[i] = da_vals[i];                                \
                        fwd_b[i] =                                            \
                            (delta_vals[i] * u_vals[i]) * b_vals[i];          \
                    } else {                                                  \
                        fwd_a[i] = 1.0f;                                      \
                        fwd_b[i] = 0.0f;                                      \
                    }                                                         \
                }                                                             \
                _Pragma("unroll")                                             \
                for (int i = 1; i < NITEMS; i++) {                            \
                    fwd_b[i] = fwd_a[i] * fwd_b[i - 1] + fwd_b[i];            \
                    fwd_a[i] = fwd_a[i] * fwd_a[i - 1];                       \
                }                                                             \
                float fscan_a = fwd_a[NITEMS - 1];                            \
                float fscan_b = fwd_b[NITEMS - 1];                            \
                __syncthreads();                                              \
                block_inclusive_scan_ab(                                      \
                    fscan_a, fscan_b, smem_fwd_wa, smem_fwd_wb);              \
                __syncthreads();                                              \
                smem_fexch_a[threadIdx.x] = fscan_a;                          \
                smem_fexch_b[threadIdx.x] = fscan_b;                          \
                __syncthreads();                                              \
                float fexcl_a, fexcl_b;                                       \
                if (threadIdx.x == 0) {                                       \
                    fexcl_a = 1.0f;                                           \
                    fexcl_b = 0.0f;                                           \
                } else {                                                      \
                    fexcl_a = smem_fexch_a[threadIdx.x - 1];                  \
                    fexcl_b = smem_fexch_b[threadIdx.x - 1];                  \
                }                                                             \
                _Pragma("unroll")                                             \
                for (int i = 0; i < NITEMS; i++) {                            \
                    float comp_a = fwd_a[i] * fexcl_a;                        \
                    float comp_b = fwd_a[i] * fexcl_b + fwd_b[i];             \
                    float final_a = comp_a * f_run_a;                         \
                    float final_b = comp_a * f_run_b + comp_b;                \
                    H_vals[i] = final_a * f_h0 + final_b;                     \
                }                                                             \
                smem_hbound[threadIdx.x] = H_vals[NITEMS - 1];                \
                __syncthreads();                                              \
                h_prev_boundary = (threadIdx.x == 0)                          \
                    ? f_hentry                                                \
                    : smem_hbound[threadIdx.x - 1];                           \
                __syncthreads();                                              \
            }                                                                 \
            smem_next_a[threadIdx.x] = da_vals[0];                            \
            __syncthreads();                                                  \
            /* Build reverse-scan pairs (a_next, d_local).                    \
               pair[i].a = da_vals[i+1] for i in [0, NITEMS-1)                \
               pair[NITEMS-1].a = next thread's da_vals[0] from smem_next_a;  \
               last thread of block uses postfix-saved next-chunk's first a. */\
            float thread_a[NITEMS];                                           \
            float thread_b[NITEMS];                                           \
            for (int i = 0; i < NITEMS - 1; i++) {                            \
                thread_a[i] = da_vals[i + 1];                                 \
                thread_b[i] = d_local[i];                                     \
            }                                                                 \
            float boundary_next_a;                                            \
            if ((int)threadIdx.x < NTHREADS - 1) {                            \
                boundary_next_a = smem_next_a[threadIdx.x + 1];               \
            } else {                                                          \
                /* Last thread of block: need a_{t+1} where t+1 is the FIRST \
                   timestep of the NEXT (later-in-time) chunk. Saved into    \
                   smem_chunk_first_a[n] when that chunk was processed.      \
                   Initialized to 1.0 for the very-last chunk in time. */    \
                boundary_next_a = smem_chunk_first_a[n];                      \
            }                                                                 \
            thread_a[NITEMS - 1] = boundary_next_a;                           \
            thread_b[NITEMS - 1] = d_local[NITEMS - 1];                       \
            __syncthreads();                                                  \
            /* Mask out-of-range elements to identity (a=1, b=0). */          \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t >= T) {                                                 \
                    thread_a[i] = 1.0f;                                       \
                    thread_b[i] = 0.0f;                                       \
                }                                                             \
            }                                                                 \
            /* Intra-thread reverse compose (right→left) the NITEMS pairs.    \
               result.b = a · b_right + b_left where compose(left, right). */ \
            for (int i = NITEMS - 2; i >= 0; i--) {                           \
                thread_b[i] = thread_a[i] * thread_b[i + 1] + thread_b[i];    \
                thread_a[i] = thread_a[i] * thread_a[i + 1];                  \
            }                                                                 \
            float scan_a = thread_a[0];                                       \
            float scan_b = thread_b[0];                                       \
            block_inclusive_reverse_scan_ab(                                  \
                scan_a, scan_b, smem_rev_wa, smem_rev_wb);                    \
            __syncthreads();                                                  \
            /* Reverse scan exclusive-NEXT: lane k needs the postfix from     \
               lane k+1 (excl_next_a/b). Save inclusive scan_a/b into the    \
               NTHREADS-sized exch tiles (smem_wa is only NWARPS floats!).   \
               Then read [threadIdx.x + 1]. */                                \
            smem_exch_a[threadIdx.x] = scan_a;                                \
            smem_exch_b[threadIdx.x] = scan_b;                                \
            __syncthreads();                                                  \
            float next_a, next_b;                                             \
            if ((int)threadIdx.x < NTHREADS - 1) {                            \
                next_a = smem_exch_a[threadIdx.x + 1];                        \
                next_b = smem_exch_b[threadIdx.x + 1];                        \
            } else {                                                          \
                /* Last thread of block: no more lanes within THIS chunk →    \
                   the exclusive-next-thread postfix is identity (1, 0).      \
                   The inter-chunk postfix (smem_post_a/b) is composed in     \
                   separately via (run_a, run_b) below. Audit fix: previous  \
                   code aliased next_a = smem_post_a[n] which then double-   \
                   composed with run_a/b at L1378-1379 → wrong dh for the    \
                   last 8 timesteps of every chunk except the very last     \
                   (manifests at T > CHUNK_SIZE = 1024, n_chunks ≥ 2). */    \
                next_a = 1.0f;                                                \
                next_b = 0.0f;                                                \
            }                                                                 \
            float run_a = smem_post_a[n];                                     \
            float run_b = smem_post_b[n];                                     \
            /* Barrier: all warps must READ the postfix before thread 0       \
               overwrites it (same race as the forward smem_run update). */   \
            __syncthreads();                                                  \
            /* Update postfix carry for the NEXT (earlier) chunk. Block-wide  \
               reverse compose end-to-end: thread 0 holds the full chunk      \
               composition. */                                                \
            if (threadIdx.x == 0) {                                           \
                /* Full chunk composition is at lane 0 after rev-scan. */     \
                float chunk_a = scan_a;                                       \
                float chunk_b = scan_b;                                       \
                /* New postfix = compose(chunk_composition, old_postfix).     \
                   compose order: chunk is to the LEFT (earlier), postfix to  \
                   the RIGHT. op_rev(left, right) = (left.a*right.a,          \
                   left.a*right.b + left.b). */                               \
                smem_post_a[n] = chunk_a * run_a;                             \
                smem_post_b[n] = chunk_a * run_b + chunk_b;                   \
            }                                                                 \
            __syncthreads();                                                  \
            /* Now per-i compute dh[i] for each timestep in this thread.      \
               After intra-thread compose: thread_a/b[i] already contains     \
               compose(pair[i], pair[i+1], ..., pair[NITEMS-1]).              \
               Compose with (next_a, next_b) which represents pairs after     \
               this thread, AND with (run_a, run_b) the postfix from later    \
               chunks. Final per-i pair: compose(thread_state[i], next_then_run). */\
            float post_a = next_a * run_a;                                    \
            float post_b = next_a * run_b + next_b;                           \
            float dh_vals[NITEMS];                                            \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < NITEMS; i++) {                                \
                /* dh[i] = thread_a[i]*post_b + thread_b[i] */                \
                dh_vals[i] = thread_a[i] * post_b + thread_b[i];              \
            }                                                                 \
            /* ---- Per-t output writes (typed) and accumulation ---- */      \
            float d_a_acc = 0.0f;                                             \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t >= T) continue;                                         \
                /* T-major: locals go [b][n][d][t] so this kernel's
                 * lane-over-t stores and the tmajor reducer's
                 * lane-over-t reads both coalesce. */                        \
                int btdn_typed = ((bid * d_state + n) * d_inner + did) * T    \
                                 + t;                                         \
                float h_curr, h_prev;                                         \
                if (slim_tape) {                                              \
                    h_curr = H_vals[i];                                       \
                    h_prev = (i > 0) ? H_vals[i - 1] : h_prev_boundary;       \
                } else {                                                      \
                    int h_row = (hsave_row_bd + n) * (T + 1);                 \
                    h_curr = h_saved[h_row + (t + 1)];                        \
                    h_prev = h_saved[h_row + t];                              \
                }                                                             \
                float dh = dh_vals[i];                                        \
                d_C_local[btdn_typed] = FROM_F(dy_vals[i] * h_curr);          \
                d_B_local[btdn_typed] = FROM_F(dh * delta_vals[i] * u_vals[i]);\
                d_delta_acc[i] += dh * (a_dn * da_vals[i] * h_prev            \
                                        + u_vals[i] * b_vals[i]);             \
                d_u_acc[i] += dh * delta_vals[i] * b_vals[i];                 \
                d_a_acc += dh * da_vals[i] * delta_vals[i] * a_dn * h_prev;   \
            }                                                                 \
            /* Block-reduce d_a_acc → thread 0 → += d_a_log_local */          \
            smem_da_red[threadIdx.x] = d_a_acc;                               \
            __syncthreads();                                                  \
            /* Tree rounds down to a full warp in smem, then the same         \
               pairing continues via shuffles: lane i still adds lane         \
               i+off's value at each halving, so every partial sum is         \
               bit-identical to the all-smem tree. */                         \
            for (int stride = NTHREADS / 2; stride >= 32; stride >>= 1) {     \
                if ((int)threadIdx.x < stride) {                              \
                    smem_da_red[threadIdx.x] += smem_da_red[threadIdx.x +     \
                                                            stride];          \
                }                                                             \
                __syncthreads();                                              \
            }                                                                 \
            float da_warp = 0.0f;                                             \
            if (threadIdx.x < 32) {                                           \
                da_warp = smem_da_red[threadIdx.x];                           \
                for (int off = 16; off > 0; off >>= 1)                        \
                    da_warp += __shfl_down_sync(0xFFFFFFFFu, da_warp,         \
                                                off);                         \
            }                                                                 \
            if (threadIdx.x == 0) {                                           \
                d_a_log_local[(bid * d_inner + did) * d_state + n]            \
                    += da_warp;                                               \
                /* Save THIS chunk's first thread's first da into the         \
                   chunk_first_a[n] slot — the EARLIER chunk (next iter)      \
                   will read this as its boundary `a_{t+1}` for the very-     \
                   last timestep before this chunk starts. */                 \
                smem_chunk_first_a[n] = smem_next_a[0];                       \
            }                                                                 \
            __syncthreads();                                                  \
        }                                                                     \
        /* ---- Store d_delta_acc, d_u_acc to typed HBM via smem ---- */      \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            smem_stage[threadIdx.x * NITEMS + i] = FROM_F(d_delta_acc[i]);    \
        }                                                                     \
        __syncthreads();                                                      \
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                          \
            if (t < T) {                                                      \
                d_delta[(bid * T + t) * d_inner + did] = smem_stage[s];       \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < NITEMS; i++) {                                    \
            smem_stage[threadIdx.x * NITEMS + i] = FROM_F(d_u_acc[i]);        \
        }                                                                     \
        __syncthreads();                                                      \
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                          \
            if (t < T) {                                                      \
                d_u[(bid * T + t) * d_inner + did] = smem_stage[s];           \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    /* Final: per-block d_D contribution. local_d_D is per-thread so          \
       reduce within block first. */                                          \
    smem_da_red[threadIdx.x] = local_d_D;                                     \
    __syncthreads();                                                          \
    for (int stride = NTHREADS / 2; stride >= 32; stride >>= 1) {             \
        if ((int)threadIdx.x < stride) {                                      \
            smem_da_red[threadIdx.x] += smem_da_red[threadIdx.x + stride];    \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    if (threadIdx.x < 32) {                                                   \
        float dd_warp = smem_da_red[threadIdx.x];                             \
        for (int off = 16; off > 0; off >>= 1)                                \
            dd_warp += __shfl_down_sync(0xFFFFFFFFu, dd_warp, off);           \
        if (threadIdx.x == 0) {                                               \
            d_D_local[bid * d_inner + did] = dd_warp;                         \
        }                                                                     \
    }                                                                         \
}

DEFINE_SSM_PARALLEL_SCAN_BWD(f32,  float,         from_f_f32)
DEFINE_SSM_PARALLEL_SCAN_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_PARALLEL_SCAN_BWD(f16,  __half,        from_f_f16)

// ============================================================================
// Parallel reverse-scan backward, d-group fold variant.
//
// The plain kernel materializes d_B/d_C locals at [B, ds, di, T] - the
// stores alone were 57% of the kernel by ablation, plus the reducer
// reads it all back. This variant gives each block SCAN_BWD_DGROUP
// consecutive d lanes: per (n, i) it folds the group's dB/dC terms in
// ascending-d order into registers and writes ONE partial row per
// group, shrinking the local tensors and the reducer's depth by
// SCAN_BWD_DGROUP. B/C are read once per (n, chunk) per block, so
// their traffic also drops by the group factor. The d-fold grouping
// is a different dB/dC summation order than the ungrouped kernel
// (deliberate; the partition is a pure function of d_inner). The
// launcher uses this kernel only when d_inner % SCAN_BWD_DGROUP == 0;
// the ungrouped kernel stays as the general-shape path.
// ============================================================================
#define SCAN_BWD_DGROUP 4

#define DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD(SUFFIX, T_ACT, FROM_F)              \
extern "C" __global__ __launch_bounds__(NTHREADS, 3) void                     \
ssm_parallel_scan_bwd_fold_##SUFFIX(                                          \
    const float* __restrict__ h_saved,                                        \
    const T_ACT* __restrict__ delta,                                          \
    const T_ACT* __restrict__ u,                                              \
    const T_ACT* __restrict__ B_in,                                           \
    const T_ACT* __restrict__ C_in,                                           \
    const float* __restrict__ a_neg,                                          \
    const float* __restrict__ D,                                              \
    const T_ACT* __restrict__ dy,                                             \
    /* OUTPUT is the PRE-softplus dt gradient: the epilogue applies the   \
       softplus derivative inline (round-FIRST - the accumulator is       \
       rounded to the activation dtype exactly as the old d_delta store   \
       did, and the derivative multiplies the reloaded value), so the     \
       separate softplus backward launch is gone. */                      \
    T_ACT* __restrict__ d_delta_raw_out,                                      \
    const T_ACT* __restrict__ dt_raw,                                         \
    T_ACT* __restrict__ d_u,                                                  \
    T_ACT* __restrict__ d_B_local, /* [B, ds, di/G, T] group partials */      \
    T_ACT* __restrict__ d_C_local, /* [B, ds, di/G, T] group partials */      \
    float* __restrict__ d_D_local,                                            \
    float* __restrict__ d_a_log_local,                                        \
    int batch, int T, int d_inner, int d_state,                               \
    const float* run_tape,                                                    \
    int slim_tape                                                             \
) {                                                                           \
    const int G = SCAN_BWD_DGROUP;                                            \
    int bid = blockIdx.x;                                                     \
    int gid = blockIdx.y; /* d group */                                       \
    int did0 = gid * G;                                                       \
    if (bid >= batch || did0 >= d_inner) return;                              \
    if (d_state > MAX_DSTATE) return;                                         \
    int n_groups = d_inner / G;                                               \
    extern __shared__ float smem[];                                           \
    /* Layout: rev warp scan (2*NWARPS), fwd-replay warp scan                 \
       (2*NWARPS), one slot per warp for the reverse postfix hand-off         \
       (2*NWARPS), the same for the replay prefix (2*NWARPS), post            \
       (2*G*d_state), chunk_first_a (G*d_state), next_a (NWARPS),             \
       da_red (G*NTHREADS), hbound (NWARPS), then the typed                   \
       delta/u/dy stage (3*G*CHUNK_SIZE T_ACT slots). */                      \
    float *smem_rev_wa = smem;                                                \
    float *smem_rev_wb = smem_rev_wa + NWARPS;                                \
    float *smem_fwd_wa = smem_rev_wb + NWARPS;                                \
    float *smem_fwd_wb = smem_fwd_wa + NWARPS;                                \
    float *smem_exch_a = smem_fwd_wb + NWARPS;                                \
    float *smem_exch_b = smem_exch_a + NWARPS;                                \
    float *smem_fexch_a = smem_exch_b + NWARPS;                               \
    float *smem_fexch_b = smem_fexch_a + NWARPS;                              \
    float *smem_post_a = smem_fexch_b + NWARPS;                               \
    float *smem_post_b = smem_post_a + SCAN_BWD_DGROUP * d_state;             \
    float *smem_chunk_first_a = smem_post_b + SCAN_BWD_DGROUP * d_state;      \
    float *smem_next_a = smem_chunk_first_a + SCAN_BWD_DGROUP * d_state;      \
    float *smem_da_red = smem_next_a + NWARPS;                                \
    float *smem_hbound = smem_da_red + SCAN_BWD_DGROUP * NTHREADS;            \
    T_ACT *smem_dio = (T_ACT *)(smem_hbound + NTHREADS);                      \
    T_ACT *stage_delta = smem_dio;                                            \
    T_ACT *stage_u = stage_delta + SCAN_BWD_DGROUP * CHUNK_SIZE;              \
    T_ACT *stage_dy = stage_u + SCAN_BWD_DGROUP * CHUNK_SIZE;                 \
    T_ACT *stage_bc = stage_dy + SCAN_BWD_DGROUP * CHUNK_SIZE;                \
    unsigned warp_mask = 0xFFFFFFFFu;                                         \
    for (int gg = 0; gg < G; gg++) {                                          \
        for (int n = threadIdx.x; n < d_state; n += NTHREADS) {               \
            smem_post_a[gg * d_state + n] = 1.0f;                          \
            smem_post_b[gg * d_state + n] = 0.0f;                          \
            smem_chunk_first_a[gg * d_state + n] = 1.0f;                   \
        }                                                                     \
    }                                                                         \
    __syncthreads();                                                          \
    float local_d_D[SCAN_BWD_DGROUP];                                         \
    _Pragma("unroll")                                                         \
    for (int gg = 0; gg < G; gg++) local_d_D[gg] = 0.0f;                      \
    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;                         \
    for (int chunk_loop = 0; chunk_loop < n_chunks; chunk_loop++) {           \
        int chunk = n_chunks - 1 - chunk_loop;                                \
        int chunk_start = chunk * CHUNK_SIZE;                                 \
        /* Stage the group's delta/u/dy rows once per chunk: one packed   \
           G-wide load per t covers all four lanes (did0..did0+G-1 is       \
           contiguous and the row base is G-aligned), a quarter of the      \
           load instructions of the per-lane sweep. Smem addressing is      \
           unchanged - same slots, same values, no new bank pattern.    */  \
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                         \
            __align__(16) T_ACT pk_delta[G];                                 \
            __align__(16) T_ACT pk_u[G];                                     \
            __align__(16) T_ACT pk_dy[G];                                    \
            if (t < T) {                                                     \
                int row = (bid * T + t) * d_inner + did0;                    \
                if (sizeof(T_ACT) == 4) {                                    \
                    *reinterpret_cast<uint4 *>(pk_delta) =                   \
                        *reinterpret_cast<const uint4 *>(&delta[row]);       \
                    *reinterpret_cast<uint4 *>(pk_u) =                       \
                        *reinterpret_cast<const uint4 *>(&u[row]);           \
                    *reinterpret_cast<uint4 *>(pk_dy) =                      \
                        *reinterpret_cast<const uint4 *>(&dy[row]);          \
                } else {                                                     \
                    *reinterpret_cast<uint2 *>(pk_delta) =                   \
                        *reinterpret_cast<const uint2 *>(&delta[row]);       \
                    *reinterpret_cast<uint2 *>(pk_u) =                       \
                        *reinterpret_cast<const uint2 *>(&u[row]);           \
                    *reinterpret_cast<uint2 *>(pk_dy) =                      \
                        *reinterpret_cast<const uint2 *>(&dy[row]);          \
                }                                                            \
            } else {                                                         \
                _Pragma("unroll")                                            \
                for (int gg = 0; gg < G; gg++) {                             \
                    pk_delta[gg] = FROM_F(0.0f);                             \
                    pk_u[gg] = FROM_F(0.0f);                                 \
                    pk_dy[gg] = FROM_F(0.0f);                                \
                }                                                            \
            }                                                                 \
            _Pragma("unroll")                                                \
            for (int gg = 0; gg < G; gg++) {                                 \
                stage_delta[gg * CHUNK_SIZE + s] = pk_delta[gg];             \
                stage_u[gg * CHUNK_SIZE + s] = pk_u[gg];                     \
                stage_dy[gg * CHUNK_SIZE + s] = pk_dy[gg];                   \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
        float d_u_acc[SCAN_BWD_DGROUP][NITEMS];                               \
        float d_delta_acc[SCAN_BWD_DGROUP][NITEMS];                           \
        _Pragma("unroll")                                                     \
        for (int gg = 0; gg < G; gg++) {                                      \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                float dyv = to_f(stage_dy[gg * CHUNK_SIZE +                   \
                                          threadIdx.x * NITEMS + i]);         \
                float uv = to_f(stage_u[gg * CHUNK_SIZE +                     \
                                        threadIdx.x * NITEMS + i]);           \
                if (t < T) {                                                  \
                    local_d_D[gg] += dyv * uv;                                \
                    d_u_acc[gg][i] = dyv * D[did0 + gg];                      \
                } else {                                                      \
                    d_u_acc[gg][i] = 0.0f;                                    \
                }                                                             \
                d_delta_acc[gg][i] = 0.0f;                                    \
            }                                                                 \
        }                                                                     \
        for (int n = 0; n < d_state; n++) {                                   \
            float b_vals[NITEMS];                                             \
            load_row8(B_in + (bid * d_state + n) * T,                         \
                      chunk_start + threadIdx.x * NITEMS, T, b_vals);         \
            float c_vals[NITEMS];                                             \
            load_row8(C_in + (bid * d_state + n) * T,                         \
                      chunk_start + threadIdx.x * NITEMS, T, c_vals);         \
            float acc_B[NITEMS];                                              \
            float acc_C[NITEMS];                                              \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < NITEMS; i++) {                                \
                acc_B[i] = 0.0f;                                              \
                acc_C[i] = 0.0f;                                              \
            }                                                                 \
            float da_acc[SCAN_BWD_DGROUP];                                    \
            _Pragma("unroll")                                                 \
            for (int gg = 0; gg < G; gg++) da_acc[gg] = 0.0f;                 \
            for (int gg = 0; gg < G; gg++) {                                  \
                int did = did0 + gg;                                          \
                float a_dn = a_neg[did * d_state + n];                        \
                float a_dn_log2 = a_dn * LOG2E;                               \
                float delta_vals[NITEMS];                                     \
                float u_vals[NITEMS];                                         \
                float dy_vals[NITEMS];                                        \
                _Pragma("unroll")                                             \
                for (int i = 0; i < NITEMS; i++) {                            \
                    int s = threadIdx.x * NITEMS + i;                         \
                    delta_vals[i] = to_f(stage_delta[gg * CHUNK_SIZE + s]);   \
                    u_vals[i] = to_f(stage_u[gg * CHUNK_SIZE + s]);           \
                    dy_vals[i] = to_f(stage_dy[gg * CHUNK_SIZE + s]);         \
                }                                                             \
                float da_vals[NITEMS];                                        \
                float d_local[NITEMS];                                        \
                _Pragma("unroll")                                             \
                for (int i = 0; i < NITEMS; i++) {                            \
                    int t = chunk_start + threadIdx.x * NITEMS + i;           \
                    if (t < T) {                                              \
                        da_vals[i] = exp2f(delta_vals[i] * a_dn_log2);        \
                        d_local[i] = dy_vals[i] * c_vals[i];                  \
                    } else {                                                  \
                        da_vals[i] = 1.0f;                                    \
                        d_local[i] = 0.0f;                                    \
                    }                                                         \
                }                                                             \
                /* Slim-tape replay pairs (see the ungrouped kernel); the     \
                   full tape scans identities and reads h_saved instead. */   \
                int hsave_row = (bid * d_inner + did) * d_state;              \
                float f_run_a = 1.0f;                                         \
                float f_run_b = 0.0f;                                         \
                float f_hentry = 0.0f;                                        \
                float f_h0 = 0.0f;                                            \
                float fwd_a[NITEMS];                                          \
                float fwd_b[NITEMS];                                          \
                if (slim_tape) {                                              \
                    int row = (hsave_row + n) * 3 * n_chunks;                 \
                    f_run_a = run_tape[row + 3 * chunk + 0];                  \
                    f_run_b = run_tape[row + 3 * chunk + 1];                  \
                    f_hentry = run_tape[row + 3 * chunk + 2];                 \
                    f_h0 = run_tape[row + 2];                                 \
                    _Pragma("unroll")                                         \
                    for (int i = 0; i < NITEMS; i++) {                        \
                        int t = chunk_start + threadIdx.x * NITEMS + i;       \
                        if (t < T) {                                          \
                            fwd_a[i] = da_vals[i];                            \
                            fwd_b[i] =                                        \
                                (delta_vals[i] * u_vals[i]) * b_vals[i];      \
                        } else {                                              \
                            fwd_a[i] = 1.0f;                                  \
                            fwd_b[i] = 0.0f;                                  \
                        }                                                     \
                    }                                                         \
                    _Pragma("unroll")                                         \
                    for (int i = 1; i < NITEMS; i++) {                        \
                        fwd_b[i] = fwd_a[i] * fwd_b[i - 1] + fwd_b[i];        \
                        fwd_a[i] = fwd_a[i] * fwd_a[i - 1];                   \
                    }                                                         \
                } else {                                                      \
                    _Pragma("unroll")                                         \
                    for (int i = 0; i < NITEMS; i++) {                        \
                        fwd_a[i] = 1.0f;                                      \
                        fwd_b[i] = 0.0f;                                      \
                    }                                                         \
                }                                                             \
                /* The reverse pairs need the next thread's first decay:      \
                   the lane below hands it up through a shuffle, a warp's     \
                   last lane takes it from the next warp's first lane         \
                   through a per-warp slot, and the block's last thread       \
                   takes the later chunk's boundary. This barrier also        \
                   retires the previous state dimension's use of every        \
                   slot and of the scan workspace. */                         \
                if ((threadIdx.x & 31) == 0) {                                \
                    smem_next_a[threadIdx.x >> 5] = da_vals[0];               \
                }                                                             \
                __syncthreads();                                              \
                float boundary_next_a =                                       \
                    __shfl_down_sync(warp_mask, da_vals[0], 1);               \
                if ((int)threadIdx.x == NTHREADS - 1) {                       \
                    boundary_next_a = smem_chunk_first_a[gg * d_state + n];   \
                } else if ((threadIdx.x & 31) == 31) {                        \
                    boundary_next_a = smem_next_a[(threadIdx.x >> 5) + 1];    \
                }                                                             \
                float thread_a[NITEMS];                                       \
                float thread_b[NITEMS];                                       \
                for (int i = 0; i < NITEMS - 1; i++) {                        \
                    thread_a[i] = da_vals[i + 1];                             \
                    thread_b[i] = d_local[i];                                 \
                }                                                             \
                thread_a[NITEMS - 1] = boundary_next_a;                       \
                thread_b[NITEMS - 1] = d_local[NITEMS - 1];                   \
                _Pragma("unroll")                                             \
                for (int i = 0; i < NITEMS; i++) {                            \
                    int t = chunk_start + threadIdx.x * NITEMS + i;           \
                    if (t >= T) {                                             \
                        thread_a[i] = 1.0f;                                   \
                        thread_b[i] = 0.0f;                                   \
                    }                                                         \
                }                                                             \
                for (int i = NITEMS - 2; i >= 0; i--) {                       \
                    thread_b[i] = thread_a[i] * thread_b[i + 1] +             \
                                  thread_b[i];                                \
                    thread_a[i] = thread_a[i] * thread_a[i + 1];              \
                }                                                             \
                float fscan_a = fwd_a[NITEMS - 1];                            \
                float fscan_b = fwd_b[NITEMS - 1];                            \
                float scan_a = thread_a[0];                                   \
                float scan_b = thread_b[0];                                   \
                block_scan_fwd_and_reverse_ab(fscan_a, fscan_b, scan_a,       \
                                              scan_b, smem_fwd_wa,            \
                                              smem_fwd_wb, smem_rev_wa,       \
                                              smem_rev_wb);                   \
                /* Replay prefix from the lane above, reverse postfix from    \
                   the lane below, per-warp slots between warps. The          \
                   running postfix is read before the barrier, so thread 0    \
                   may fold this chunk into it right after. */                \
                if ((threadIdx.x & 31) == 31) {                               \
                    smem_fexch_a[threadIdx.x >> 5] = fscan_a;                 \
                    smem_fexch_b[threadIdx.x >> 5] = fscan_b;                 \
                }                                                             \
                if ((threadIdx.x & 31) == 0) {                                \
                    smem_exch_a[threadIdx.x >> 5] = scan_a;                   \
                    smem_exch_b[threadIdx.x >> 5] = scan_b;                   \
                }                                                             \
                float run_a = smem_post_a[gg * d_state + n];                  \
                float run_b = smem_post_b[gg * d_state + n];                  \
                __syncthreads();                                              \
                float fexcl_a = __shfl_up_sync(warp_mask, fscan_a, 1);        \
                float fexcl_b = __shfl_up_sync(warp_mask, fscan_b, 1);        \
                if (threadIdx.x == 0) {                                       \
                    fexcl_a = 1.0f;                                           \
                    fexcl_b = 0.0f;                                           \
                } else if ((threadIdx.x & 31) == 0) {                         \
                    fexcl_a = smem_fexch_a[(threadIdx.x >> 5) - 1];           \
                    fexcl_b = smem_fexch_b[(threadIdx.x >> 5) - 1];           \
                }                                                             \
                float next_a = __shfl_down_sync(warp_mask, scan_a, 1);        \
                float next_b = __shfl_down_sync(warp_mask, scan_b, 1);        \
                if ((int)threadIdx.x == NTHREADS - 1) {                       \
                    next_a = 1.0f;                                            \
                    next_b = 0.0f;                                            \
                } else if ((threadIdx.x & 31) == 31) {                        \
                    next_a = smem_exch_a[(threadIdx.x >> 5) + 1];             \
                    next_b = smem_exch_b[(threadIdx.x >> 5) + 1];             \
                }                                                             \
                if (threadIdx.x == 0) {                                       \
                    smem_post_a[gg * d_state + n] = scan_a * run_a;           \
                    smem_post_b[gg * d_state + n] =                           \
                        scan_a * run_b + scan_b;                              \
                }                                                             \
                float H_vals[NITEMS];                                         \
                _Pragma("unroll")                                             \
                for (int i = 0; i < NITEMS; i++) {                            \
                    float comp_a = fwd_a[i] * fexcl_a;                        \
                    float comp_b = fwd_a[i] * fexcl_b + fwd_b[i];             \
                    float final_a = comp_a * f_run_a;                         \
                    float final_b = comp_a * f_run_b + comp_b;                \
                    H_vals[i] = final_a * f_h0 + final_b;                     \
                }                                                             \
                /* The previous timestep's replayed state comes from the      \
                   lane above the same way. */                                \
                if ((threadIdx.x & 31) == 31) {                               \
                    smem_hbound[threadIdx.x >> 5] = H_vals[NITEMS - 1];       \
                }                                                             \
                __syncthreads();                                              \
                float h_prev_boundary =                                       \
                    __shfl_up_sync(warp_mask, H_vals[NITEMS - 1], 1);         \
                if (threadIdx.x == 0) {                                       \
                    h_prev_boundary = f_hentry;                               \
                } else if ((threadIdx.x & 31) == 0) {                         \
                    h_prev_boundary = smem_hbound[(threadIdx.x >> 5) - 1];    \
                }                                                             \
                float post_a = next_a * run_a;                                \
                float post_b = next_a * run_b + next_b;                       \
                _Pragma("unroll")                                             \
                for (int i = 0; i < NITEMS; i++) {                            \
                    int t = chunk_start + threadIdx.x * NITEMS + i;           \
                    if (t >= T) continue;                                     \
                    float dh = thread_a[i] * post_b + thread_b[i];            \
                    float h_curr, h_prev;                                     \
                    if (slim_tape) {                                          \
                        h_curr = H_vals[i];                                   \
                        h_prev =                                              \
                            (i > 0) ? H_vals[i - 1] : h_prev_boundary;        \
                    } else {                                                  \
                        int h_row = (hsave_row + n) * (T + 1);                \
                        h_curr = h_saved[h_row + (t + 1)];                    \
                        h_prev = h_saved[h_row + t];                          \
                    }                                                         \
                    /* ascending-g fold replaces the per-d store */           \
                    acc_C[i] += dy_vals[i] * h_curr;                          \
                    acc_B[i] += dh * delta_vals[i] * u_vals[i];               \
                    d_delta_acc[gg][i] += dh * (a_dn * da_vals[i] * h_prev    \
                                                + u_vals[i] * b_vals[i]);     \
                    d_u_acc[gg][i] += dh * delta_vals[i] * b_vals[i];         \
                    da_acc[gg] += dh * da_vals[i] * delta_vals[i] * a_dn *    \
                                  h_prev;                                     \
                }                                                             \
                /* This chunk's first decay is the earlier chunk's            \
                   boundary; the last thread read the old value before        \
                   the scan barrier above. */                                 \
                if (threadIdx.x == 0) {                                       \
                    smem_chunk_first_a[gg * d_state + n] = da_vals[0];        \
                }                                                             \
            }                                                                 \
            /* d_a: one block reduction per group, the four sharing their     \
               barriers; each keeps its own lane pairing and add order. */    \
            _Pragma("unroll")                                                 \
            for (int gg = 0; gg < G; gg++) {                                  \
                smem_da_red[gg * NTHREADS + threadIdx.x] = da_acc[gg];        \
            }                                                                 \
            __syncthreads();                                                  \
            for (int stride = NTHREADS / 2; stride >= 32; stride >>= 1) {     \
                if ((int)threadIdx.x < stride) {                              \
                    _Pragma("unroll")                                         \
                    for (int gg = 0; gg < G; gg++) {                          \
                        smem_da_red[gg * NTHREADS + threadIdx.x] +=           \
                            smem_da_red[gg * NTHREADS + threadIdx.x + stride]; \
                    }                                                         \
                }                                                             \
                __syncthreads();                                              \
            }                                                                 \
            if (threadIdx.x < 32) {                                           \
                _Pragma("unroll")                                             \
                for (int gg = 0; gg < G; gg++) {                              \
                    float da_warp = smem_da_red[gg * NTHREADS + threadIdx.x]; \
                    for (int off = 16; off > 0; off >>= 1)                    \
                        da_warp += __shfl_down_sync(warp_mask, da_warp,       \
                                                    off);                     \
                    if (threadIdx.x == 0) {                                   \
                        /* One partial SLOT per chunk instead of a global     \
                           read-modify-write per (chunk, lane, n): slot       \
                           order mirrors the walk (chunk_loop ascends =       \
                           chunks DESCEND in time), and the chunked           \
                           reducer folds the slots in exactly this order      \
                           before adding across the batch - the same          \
                           left-to-right chain the accumulator produced. */   \
                        d_a_log_local[((bid * n_chunks + chunk_loop)          \
                                       * d_inner + did0 + gg) * d_state + n]  \
                            = da_warp;                                        \
                    }                                                         \
                }                                                             \
            }                                                                 \
            /* One partial row per (n, group): [b][n][group][t]. The row is   \
               t-contiguous, but the lane->t mapping is blocked - a direct  \
               store spans 16 sectors per warp instruction. Stage through   \
               the CHUNK tile and store striped: consecutive lanes then     \
               write consecutive addresses. Values unchanged.            */ \
            int row_bc = ((bid * d_state + n) * (d_inner / G) + gid) * T;     \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < NITEMS; i++) {                                \
                stage_bc[threadIdx.x * NITEMS + i] = FROM_F(acc_B[i]);        \
            }                                                                 \
            __syncthreads();                                                  \
            for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {        \
                int t = chunk_start + s;                                      \
                if (t < T) d_B_local[row_bc + t] = stage_bc[s];               \
            }                                                                 \
            __syncthreads();                                                  \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < NITEMS; i++) {                                \
                stage_bc[threadIdx.x * NITEMS + i] = FROM_F(acc_C[i]);        \
            }                                                                 \
            __syncthreads();                                                  \
            for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {        \
                int t = chunk_start + s;                                      \
                if (t < T) d_C_local[row_bc + t] = stage_bc[s];               \
            }                                                                 \
        }                                                                     \
        /* d_delta / d_u: one packed G-wide store per t. Each lane        \
           already holds all G lanes' values for its own t positions,        \
           did0..did0+G-1 is contiguous, and the row base is aligned         \
           because d_inner % G == 0 is the fold's launch precondition -     \
           the smem staging round-trip and its eight barriers per chunk     \
           bought nothing (destination stride between consecutive t is      \
           d_inner elements either way). Values and rounding unchanged.  */ \
        _Pragma("unroll")                                                     \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            if (t < T) {                                                      \
                __align__(16) T_ACT pack_d[G];                                \
                __align__(16) T_ACT pack_u[G];                                \
                __align__(16) T_ACT pack_raw[G];                              \
                int row = (bid * T + t) * d_inner + did0;                     \
                if (sizeof(T_ACT) == 4) {                                     \
                    *reinterpret_cast<uint4 *>(pack_raw) =                    \
                        *reinterpret_cast<const uint4 *>(&dt_raw[row]);       \
                } else {                                                      \
                    *reinterpret_cast<uint2 *>(pack_raw) =                    \
                        *reinterpret_cast<const uint2 *>(&dt_raw[row]);       \
                }                                                             \
                _Pragma("unroll")                                             \
                for (int gg = 0; gg < G; gg++) {                              \
                    /* Round FIRST: the accumulator rounds to the         \
                       activation dtype exactly as the old d_delta store  \
                       did, then the softplus derivative divides the      \
                       reloaded value - the retired kernel's chain,       \
                       rounding for rounding. */                          \
                    float dd = to_f(FROM_F(d_delta_acc[gg][i]));              \
                    float xr = to_f(pack_raw[gg]);                            \
                    pack_d[gg] = FROM_F(                                      \
                        dd / (1.0f + exp2f(-xr * 1.4426950408889634f)));      \
                    pack_u[gg] = FROM_F(d_u_acc[gg][i]);                      \
                }                                                             \
                if (sizeof(T_ACT) == 4) {                                     \
                    *reinterpret_cast<uint4 *>(&d_delta_raw_out[row]) =      \
                        *reinterpret_cast<uint4 *>(pack_d);                   \
                    *reinterpret_cast<uint4 *>(&d_u[row]) =                   \
                        *reinterpret_cast<uint4 *>(pack_u);                   \
                } else {                                                      \
                    *reinterpret_cast<uint2 *>(&d_delta_raw_out[row]) =      \
                        *reinterpret_cast<uint2 *>(pack_d);                   \
                    *reinterpret_cast<uint2 *>(&d_u[row]) =                   \
                        *reinterpret_cast<uint2 *>(pack_u);                   \
                }                                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int gg = 0; gg < G; gg++) {                                          \
        smem_da_red[threadIdx.x] = local_d_D[gg];                             \
        __syncthreads();                                                      \
        for (int stride = NTHREADS / 2; stride >= 32; stride >>= 1) {         \
            if ((int)threadIdx.x < stride) {                                  \
                smem_da_red[threadIdx.x] +=                                   \
                    smem_da_red[threadIdx.x + stride];                        \
            }                                                                 \
            __syncthreads();                                                  \
        }                                                                     \
        if (threadIdx.x < 32) {                                               \
            float dd_warp = smem_da_red[threadIdx.x];                         \
            for (int off = 16; off > 0; off >>= 1)                            \
                dd_warp += __shfl_down_sync(warp_mask, dd_warp, off);         \
            if (threadIdx.x == 0) {                                           \
                d_D_local[bid * d_inner + did0 + gg] = dd_warp;               \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
}

DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD(f32,  float,         from_f_f32)
DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD(f16,  __half,        from_f_f16)


// Clean up macros to avoid polluting subsequent translation units
// (all .cu files are concatenated before NVRTC compilation)
#undef NTHREADS
#undef NITEMS
#undef CHUNK_SIZE
#undef NWARPS
#undef MAX_DSTATE
#undef SMEM_WA_OFF
#undef SMEM_WB_OFF
#undef SMEM_RUN_A_OFF
#undef SMEM_RUN_B_OFF
#undef SMEM_EXCH_A_OFF
#undef SMEM_EXCH_B_OFF
#undef SMEM_STAGE_OFF
#undef SMEM_TOTAL_FLOATS
/* the epilogue stopped at the forward set,
   leaving the backward smem offsets + the DEFINE_* generator names alive
   in the concatenated NVRTC TU. Complete the cleanup. */
#undef SMEM_REV_WA_OFF
#undef SMEM_REV_WB_OFF
#undef SMEM_POST_A_OFF
#undef SMEM_POST_B_OFF
#undef SMEM_NEXT_A_OFF
#undef SMEM_DA_RED_OFF
#undef SMEM_CHUNK_FIRST_A_OFF
#undef SMEM_BWD_FLOATS
#undef DEFINE_SSM_PARALLEL_SCAN_FWD
#undef DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE
#undef DEFINE_SSM_PARALLEL_SCAN_BWD
#undef DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD
#undef SCAN_BWD_DGROUP
