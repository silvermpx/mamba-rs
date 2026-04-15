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
// Step 8b: bf16/f16 mixed-precision parallel scan forward. Following
// state-spaces/mamba's `scan_t = float2` invariant (all scan state in
// f32) and our Step 5 BPTT precision discipline (h, h_saved, da_exp_out,
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

// Must be >= actual d_state. Matches Tri Dao's MAX_DSTATE = 256.
#define MAX_DSTATE 256

// ============================================================================
// Warp-level inclusive scan of (a, b) pairs using warp shuffle.
// After return, lane k holds compose(pair_0, ..., pair_k) within its warp.
// ============================================================================
// BUG FIX (Step 8b): accept a mask parameter rather than hardcoding
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
__device__ __forceinline__ void warp_inclusive_reverse_scan_ab(
    float &a, float &b, unsigned mask = 0xffffffff
) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float a_next = __shfl_down_sync(mask, a, offset);
        float b_next = __shfl_down_sync(mask, b, offset);
        if ((threadIdx.x & 31) + (unsigned)offset < 32) {
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
    // Same partial-mask correctness fix as forward variant.
    if (warp_id == 0 && lane < NWARPS) {
        float wa = smem_wa[lane];
        float wb = smem_wb[lane];
        warp_inclusive_reverse_scan_ab(wa, wb, (1u << NWARPS) - 1u);
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
// Shared memory layout (in extern __shared__ float[]):
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
// ============================================================================
#define SMEM_WA_OFF        0
#define SMEM_WB_OFF        (NWARPS)
#define SMEM_RUN_A_OFF     (2 * NWARPS)
#define SMEM_RUN_B_OFF     (2 * NWARPS + MAX_DSTATE)
#define SMEM_EXCH_A_OFF    (2 * NWARPS + 2 * MAX_DSTATE)
#define SMEM_EXCH_B_OFF    (2 * NWARPS + 2 * MAX_DSTATE + NTHREADS)
#define SMEM_STAGE_OFF     (2 * NWARPS + 2 * MAX_DSTATE + 2 * NTHREADS)
#define SMEM_TOTAL_FLOATS  (2 * NWARPS + 2 * MAX_DSTATE + 2 * NTHREADS + CHUNK_SIZE)

// Step 8e — extra smem offsets for the backward-pass reverse scan.
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
// Shared memory: SMEM_TOTAL_FLOATS * sizeof(float).
// ============================================================================
extern "C" __global__ __launch_bounds__(128, 3) void ssm_parallel_scan_fwd(
    float* __restrict__ h,             // [batch * d_inner * d_state] SSM state (mutated)
    float* __restrict__ y_out,         // [batch * T * d_inner] output
    float* __restrict__ h_saved,       // [batch * (T+1) * d_inner * d_state] saved for backward
    float* __restrict__ da_exp_out,    // [batch * T * d_inner * d_state] saved exp(delta*A)
    const float* __restrict__ delta,   // [batch * T * d_inner]
    const float* __restrict__ u,       // [batch * T * d_inner]
    const float* __restrict__ B,       // [batch * T * d_state]
    const float* __restrict__ C,       // [batch * T * d_state]
    const float* __restrict__ a_neg,   // [d_inner * d_state]
    const float* __restrict__ D,       // [d_inner]
    int batch, int T, int d_inner, int d_state
) {
    int bid = blockIdx.x;
    int did = blockIdx.y;
    if (bid >= batch || did >= d_inner) return;
    if (d_state > MAX_DSTATE) return;

    extern __shared__ float smem[];
    float *smem_wa     = smem + SMEM_WA_OFF;
    float *smem_wb     = smem + SMEM_WB_OFF;
    float *smem_run_a  = smem + SMEM_RUN_A_OFF;
    float *smem_run_b  = smem + SMEM_RUN_B_OFF;
    float *smem_exch_a = smem + SMEM_EXCH_A_OFF;
    float *smem_exch_b = smem + SMEM_EXCH_B_OFF;
    float *smem_stage  = smem + SMEM_STAGE_OFF;

    float D_d = D[did];
    int h_base = (bid * d_inner + did) * d_state;

    // Save initial SSM state at time index 0 (parallelized across threads)
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {
        int hs_idx = (bid * (T + 1) + 0) * d_inner * d_state + did * d_state + n;
        h_saved[hs_idx] = h[h_base + n];
    }

    // Initialize running prefix to identity (1, 0) for each state dimension
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {
        smem_run_a[n] = 1.0f;
        smem_run_b[n] = 0.0f;
    }
    __syncthreads();

    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int chunk_start = chunk * CHUNK_SIZE;

        // ================================================================
        // Coalesced delta load via shared memory staging.
        // Striped load: thread k loads indices k, k+NTHREADS, k+2*NTHREADS, ...
        // Then blocked read from smem_stage for per-thread NITEMS.
        // Layout in global: delta[(bid*T + t) * d_inner + did] -- stride d_inner
        // between adjacent t, so adjacent threads reading adjacent t gives
        // coalesced access when d_inner is the innermost dim.
        // ================================================================
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {
            int t = chunk_start + s;
            if (t < T) {
                smem_stage[s] = delta[(bid * T + t) * d_inner + did];
            } else {
                smem_stage[s] = 0.0f;
            }
        }
        __syncthreads();

        float delta_vals[NITEMS];
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            delta_vals[i] = smem_stage[threadIdx.x * NITEMS + i];
        }
        __syncthreads();

        // ================================================================
        // Coalesced u load via shared memory staging (same pattern).
        // ================================================================
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {
            int t = chunk_start + s;
            if (t < T) {
                smem_stage[s] = u[(bid * T + t) * d_inner + did];
            } else {
                smem_stage[s] = 0.0f;
            }
        }
        __syncthreads();

        float u_vals[NITEMS];
        float delta_u_vals[NITEMS];
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            u_vals[i] = smem_stage[threadIdx.x * NITEMS + i];
            delta_u_vals[i] = delta_vals[i] * u_vals[i];
        }
        __syncthreads();

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

            // ============================================================
            // Coalesced B load via shared memory staging.
            // B layout: [batch, T, d_state] -- stride d_state between
            // adjacent t, so we stage through smem for coalescing.
            // ============================================================
            for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {
                int t = chunk_start + s;
                if (t < T) {
                    smem_stage[s] = B[bid * T * d_state + t * d_state + n];
                } else {
                    smem_stage[s] = 0.0f;
                }
            }
            __syncthreads();

            // Build (da, db) pairs for this state dimension
            float thread_a[NITEMS];
            float thread_b[NITEMS];

            #pragma unroll
            for (int i = 0; i < NITEMS; i++) {
                int t = chunk_start + threadIdx.x * NITEMS + i;
                if (t < T) {
                    // a_dn already has LOG2E folded in, so exp2f gives exp(delta*a)
                    float da = exp2f(delta_vals[i] * a_dn);
                    float b_t = smem_stage[threadIdx.x * NITEMS + i];
                    thread_a[i] = da;
                    thread_b[i] = delta_u_vals[i] * b_t;
                    // da_exp_out write removed: backward recomputes da from
                    // delta and a_neg. Saves bandwidth; buffer kept in
                    // interface for ABI stability.
                } else {
                    thread_a[i] = 1.0f;  // identity
                    thread_b[i] = 0.0f;
                }
            }
            __syncthreads();

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

            // Store inclusive scan results for exclusive prefix extraction
            __syncthreads();
            smem_exch_a[threadIdx.x] = scan_a;
            smem_exch_b[threadIdx.x] = scan_b;
            __syncthreads();

            // Exclusive prefix: inclusive result of thread (t-1), or identity for thread 0
            float excl_a, excl_b;
            if (threadIdx.x == 0) {
                excl_a = 1.0f;
                excl_b = 0.0f;
            } else {
                excl_a = smem_exch_a[threadIdx.x - 1];
                excl_b = smem_exch_b[threadIdx.x - 1];
            }

            // Read inter-chunk running prefix for this state dimension
            float run_a = smem_run_a[n];
            float run_b = smem_run_b[n];

            // Initial state for this (b, d, n) triple
            float h_0 = h[h_base + n];

            // Update running prefix for next chunk (thread 0 only -- data dependency):
            //   new_run = block_total o old_run
            if (threadIdx.x == 0) {
                float block_a = smem_exch_a[NTHREADS - 1];
                float block_b = smem_exch_b[NTHREADS - 1];
                smem_run_a[n] = block_a * run_a;
                smem_run_b[n] = block_a * run_b + block_b;
            }

            // ============================================================
            // Coalesced C load via shared memory staging.
            // Same layout as B: [batch, T, d_state].
            // ============================================================
            __syncthreads();
            for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {
                int t = chunk_start + s;
                if (t < T) {
                    smem_stage[s] = C[bid * T * d_state + t * d_state + n];
                } else {
                    smem_stage[s] = 0.0f;
                }
            }
            __syncthreads();

            // Compute h[t] for each element and accumulate y[t] += h[t] * C[t,n]
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

                    // Save h for backward (at time index t+1)
                    int hs_idx = (bid * (T + 1) + (t + 1)) * d_inner * d_state + did * d_state + n;
                    h_saved[hs_idx] = h_t;

                    // Single-pass Y accumulation
                    float c_t = smem_stage[threadIdx.x * NITEMS + i];
                    out_vals[i] += h_t * c_t;
                }
            }

            __syncthreads();
        } // end d_state loop

        // ================================================================
        // Coalesced y_out write via shared memory staging.
        // Write blocked into smem_stage, sync, write striped to global.
        // ================================================================
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            smem_stage[threadIdx.x * NITEMS + i] = out_vals[i];
        }
        __syncthreads();

        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {
            int t = chunk_start + s;
            if (t < T) {
                y_out[(bid * T + t) * d_inner + did] = smem_stage[s];
            }
        }
        __syncthreads();
    } // end chunk loop

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
// Same parallel scan algorithm, skips h_saved and da_exp writes.
// ============================================================================
extern "C" __global__ __launch_bounds__(128, 3) void ssm_parallel_scan_fwd_nosave(
    float* __restrict__ h,             // [batch * d_inner * d_state] SSM state (mutated)
    float* __restrict__ y_out,         // [batch * T * d_inner] output
    const float* __restrict__ delta,   // [batch * T * d_inner]
    const float* __restrict__ u,       // [batch * T * d_inner]
    const float* __restrict__ B,       // [batch * T * d_state]
    const float* __restrict__ C,       // [batch * T * d_state]
    const float* __restrict__ a_neg,   // [d_inner * d_state]
    const float* __restrict__ D,       // [d_inner]
    int batch, int T, int d_inner, int d_state
) {
    int bid = blockIdx.x;
    int did = blockIdx.y;
    if (bid >= batch || did >= d_inner) return;
    if (d_state > MAX_DSTATE) return;

    extern __shared__ float smem[];
    float *smem_wa     = smem + SMEM_WA_OFF;
    float *smem_wb     = smem + SMEM_WB_OFF;
    float *smem_run_a  = smem + SMEM_RUN_A_OFF;
    float *smem_run_b  = smem + SMEM_RUN_B_OFF;
    float *smem_exch_a = smem + SMEM_EXCH_A_OFF;
    float *smem_exch_b = smem + SMEM_EXCH_B_OFF;
    float *smem_stage  = smem + SMEM_STAGE_OFF;

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

        // ================================================================
        // Coalesced delta load via shared memory staging.
        // ================================================================
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {
            int t = chunk_start + s;
            if (t < T) {
                smem_stage[s] = delta[(bid * T + t) * d_inner + did];
            } else {
                smem_stage[s] = 0.0f;
            }
        }
        __syncthreads();

        float delta_vals[NITEMS];
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            delta_vals[i] = smem_stage[threadIdx.x * NITEMS + i];
        }
        __syncthreads();

        // ================================================================
        // Coalesced u load via shared memory staging.
        // ================================================================
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {
            int t = chunk_start + s;
            if (t < T) {
                smem_stage[s] = u[(bid * T + t) * d_inner + did];
            } else {
                smem_stage[s] = 0.0f;
            }
        }
        __syncthreads();

        float u_vals[NITEMS];
        float delta_u_vals[NITEMS];
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            u_vals[i] = smem_stage[threadIdx.x * NITEMS + i];
            delta_u_vals[i] = delta_vals[i] * u_vals[i];
        }
        __syncthreads();

        float out_vals[NITEMS];
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            out_vals[i] = D_d * u_vals[i];
        }

        for (int n = 0; n < d_state; n++) {
            // Pre-multiply a_neg by LOG2E so we can use exp2f directly,
            // saving one FMUL per (t, d, n) triple.
            float a_dn = a_neg[did * d_state + n] * LOG2E;

            // ============================================================
            // Coalesced B load via shared memory staging.
            // ============================================================
            for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {
                int t = chunk_start + s;
                if (t < T) {
                    smem_stage[s] = B[bid * T * d_state + t * d_state + n];
                } else {
                    smem_stage[s] = 0.0f;
                }
            }
            __syncthreads();

            float thread_a[NITEMS];
            float thread_b[NITEMS];

            #pragma unroll
            for (int i = 0; i < NITEMS; i++) {
                int t = chunk_start + threadIdx.x * NITEMS + i;
                if (t < T) {
                    // a_dn already has LOG2E folded in, so exp2f gives exp(delta*a)
                    float da = exp2f(delta_vals[i] * a_dn);
                    float b_t = smem_stage[threadIdx.x * NITEMS + i];
                    thread_a[i] = da;
                    thread_b[i] = delta_u_vals[i] * b_t;
                } else {
                    thread_a[i] = 1.0f;
                    thread_b[i] = 0.0f;
                }
            }
            __syncthreads();

            #pragma unroll
            for (int i = 1; i < NITEMS; i++) {
                thread_b[i] = thread_a[i] * thread_b[i - 1] + thread_b[i];
                thread_a[i] = thread_a[i] * thread_a[i - 1];
            }

            float scan_a = thread_a[NITEMS - 1];
            float scan_b = thread_b[NITEMS - 1];

            __syncthreads();
            block_inclusive_scan_ab(scan_a, scan_b, smem_wa, smem_wb);

            __syncthreads();
            smem_exch_a[threadIdx.x] = scan_a;
            smem_exch_b[threadIdx.x] = scan_b;
            __syncthreads();

            float excl_a, excl_b;
            if (threadIdx.x == 0) {
                excl_a = 1.0f;
                excl_b = 0.0f;
            } else {
                excl_a = smem_exch_a[threadIdx.x - 1];
                excl_b = smem_exch_b[threadIdx.x - 1];
            }

            float run_a = smem_run_a[n];
            float run_b = smem_run_b[n];
            float h_0 = h[h_base + n];

            if (threadIdx.x == 0) {
                float block_a = smem_exch_a[NTHREADS - 1];
                float block_b = smem_exch_b[NTHREADS - 1];
                smem_run_a[n] = block_a * run_a;
                smem_run_b[n] = block_a * run_b + block_b;
            }

            // ============================================================
            // Coalesced C load via shared memory staging.
            // ============================================================
            __syncthreads();
            for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {
                int t = chunk_start + s;
                if (t < T) {
                    smem_stage[s] = C[bid * T * d_state + t * d_state + n];
                } else {
                    smem_stage[s] = 0.0f;
                }
            }
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < NITEMS; i++) {
                int t = chunk_start + threadIdx.x * NITEMS + i;
                if (t < T) {
                    float comp_a = thread_a[i] * excl_a;
                    float comp_b = thread_a[i] * excl_b + thread_b[i];
                    float final_a = comp_a * run_a;
                    float final_b = comp_a * run_b + comp_b;
                    float h_t = final_a * h_0 + final_b;

                    float c_t = smem_stage[threadIdx.x * NITEMS + i];
                    out_vals[i] += h_t * c_t;
                }
            }

            __syncthreads();
        } // end d_state loop

        // ================================================================
        // Coalesced y_out write via shared memory staging.
        // ================================================================
        #pragma unroll
        for (int i = 0; i < NITEMS; i++) {
            smem_stage[threadIdx.x * NITEMS + i] = out_vals[i];
        }
        __syncthreads();

        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {
            int t = chunk_start + s;
            if (t < T) {
                y_out[(bid * T + t) * d_inner + did] = smem_stage[s];
            }
        }
        __syncthreads();
    } // end chunk loop

    // Write final SSM state (parallelized across threads)
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {
        float h_0 = h[h_base + n];
        h[h_base + n] = smem_run_a[n] * h_0 + smem_run_b[n];
    }
}

// ============================================================================
// Typed variants (bf16/f16) — Step 8b of v0.2.2 mixed-precision training.
//
// Follow `state-spaces/mamba`'s `scan_t = float2` discipline: all scan
// state + running prefix + block scan + registers stay f32. Only the
// activation I/O tensors (delta, u, B, C, y_out) become typed. BPTT
// state (`h`, `h_saved`, `da_exp_out`), model parameters (`a_neg`, `D`),
// and ALL `smem_*` (including smem_stage) remain f32.
// ============================================================================

#define DEFINE_SSM_PARALLEL_SCAN_FWD(SUFFIX, T_ACT, FROM_F)                   \
extern "C" __global__ __launch_bounds__(128, 4) void                          \
ssm_parallel_scan_fwd_##SUFFIX(                                               \
    float* __restrict__ h,                                                    \
    T_ACT* __restrict__ y_out,                                                \
    float* __restrict__ h_saved,                                              \
    float* __restrict__ da_exp_out,                                           \
    const T_ACT* __restrict__ delta,                                          \
    const T_ACT* __restrict__ u,                                              \
    const T_ACT* __restrict__ B,                                              \
    const T_ACT* __restrict__ C,                                              \
    const float* __restrict__ a_neg,                                          \
    const float* __restrict__ D,                                              \
    int batch, int T, int d_inner, int d_state                                \
) {                                                                           \
    int bid = blockIdx.x;                                                     \
    int did = blockIdx.y;                                                     \
    if (bid >= batch || did >= d_inner) return;                               \
    if (d_state > MAX_DSTATE) return;                                         \
    extern __shared__ float smem[];                                           \
    float *smem_wa     = smem + SMEM_WA_OFF;                                  \
    float *smem_wb     = smem + SMEM_WB_OFF;                                  \
    float *smem_run_a  = smem + SMEM_RUN_A_OFF;                               \
    float *smem_run_b  = smem + SMEM_RUN_B_OFF;                               \
    float *smem_exch_a = smem + SMEM_EXCH_A_OFF;                              \
    float *smem_exch_b = smem + SMEM_EXCH_B_OFF;                              \
    /* Typed smem stage: 2-byte slots reuse the f32 stage region. The         \
       typed launch helper only allocates CHUNK_SIZE * sizeof(T_ACT) bytes    \
       for this region (vs CHUNK_SIZE * 4 for the f32 path), saving 2 KB     \
       per block → enables an extra resident block on Ada (audit Agent 5     \
       optimization #1). Load stores T_ACT directly; upcast happens only     \
       inside the compute loop via to_f(). */                                 \
    T_ACT *smem_stage = (T_ACT *)(smem + SMEM_STAGE_OFF);                     \
    float D_d = D[did];                                                       \
    int h_base = (bid * d_inner + did) * d_state;                             \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        int hs_idx = (bid * (T + 1) + 0) * d_inner * d_state                  \
                     + did * d_state + n;                                     \
        h_saved[hs_idx] = h[h_base + n];                                      \
    }                                                                         \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        smem_run_a[n] = 1.0f;                                                 \
        smem_run_b[n] = 0.0f;                                                 \
    }                                                                         \
    __syncthreads();                                                          \
    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;                         \
    for (int chunk = 0; chunk < n_chunks; chunk++) {                          \
        int chunk_start = chunk * CHUNK_SIZE;                                 \
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                          \
            smem_stage[s] = (t < T)                                           \
                ? delta[(bid * T + t) * d_inner + did]                        \
                : FROM_F(0.0f);                                               \
        }                                                                     \
        __syncthreads();                                                      \
        float delta_vals[NITEMS];                                             \
                                                             \
        for (int i = 0; i < NITEMS; i++) {                                    \
            delta_vals[i] = to_f(smem_stage[threadIdx.x * NITEMS + i]);       \
        }                                                                     \
        __syncthreads();                                                      \
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                          \
            smem_stage[s] = (t < T)                                           \
                ? u[(bid * T + t) * d_inner + did]                            \
                : FROM_F(0.0f);                                               \
        }                                                                     \
        __syncthreads();                                                      \
        float u_vals[NITEMS];                                                 \
        float delta_u_vals[NITEMS];                                           \
                                                             \
        for (int i = 0; i < NITEMS; i++) {                                    \
            u_vals[i] = to_f(smem_stage[threadIdx.x * NITEMS + i]);           \
            delta_u_vals[i] = delta_vals[i] * u_vals[i];                      \
        }                                                                     \
        __syncthreads();                                                      \
        float out_vals[NITEMS];                                               \
                                                             \
        for (int i = 0; i < NITEMS; i++) {                                    \
            out_vals[i] = D_d * u_vals[i];                                    \
        }                                                                     \
        for (int n = 0; n < d_state; n++) {                                   \
            float a_dn = a_neg[did * d_state + n] * LOG2E;                    \
            for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {        \
                int t = chunk_start + s;                                      \
                smem_stage[s] = (t < T)                                       \
                    ? B[bid * T * d_state + t * d_state + n]                  \
                    : FROM_F(0.0f);                                           \
            }                                                                 \
            __syncthreads();                                                  \
            float thread_a[NITEMS];                                           \
            float thread_b[NITEMS];                                           \
                                                             \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    float da = exp2f(delta_vals[i] * a_dn);                   \
                    float b_t = to_f(smem_stage[threadIdx.x * NITEMS + i]);   \
                    thread_a[i] = da;                                         \
                    thread_b[i] = delta_u_vals[i] * b_t;                      \
                } else {                                                      \
                    thread_a[i] = 1.0f;                                       \
                    thread_b[i] = 0.0f;                                       \
                }                                                             \
            }                                                                 \
            __syncthreads();                                                  \
                                                             \
            for (int i = 1; i < NITEMS; i++) {                                \
                thread_b[i] = thread_a[i] * thread_b[i - 1] + thread_b[i];    \
                thread_a[i] = thread_a[i] * thread_a[i - 1];                  \
            }                                                                 \
            float scan_a = thread_a[NITEMS - 1];                              \
            float scan_b = thread_b[NITEMS - 1];                              \
            __syncthreads();                                                  \
            block_inclusive_scan_ab(scan_a, scan_b, smem_wa, smem_wb);        \
            __syncthreads();                                                  \
            smem_exch_a[threadIdx.x] = scan_a;                                \
            smem_exch_b[threadIdx.x] = scan_b;                                \
            __syncthreads();                                                  \
            float excl_a, excl_b;                                             \
            if (threadIdx.x == 0) {                                           \
                excl_a = 1.0f;                                                \
                excl_b = 0.0f;                                                \
            } else {                                                          \
                excl_a = smem_exch_a[threadIdx.x - 1];                        \
                excl_b = smem_exch_b[threadIdx.x - 1];                        \
            }                                                                 \
            float run_a = smem_run_a[n];                                      \
            float run_b = smem_run_b[n];                                      \
            float h_0 = h[h_base + n];                                        \
            if (threadIdx.x == 0) {                                           \
                float block_a = smem_exch_a[NTHREADS - 1];                    \
                float block_b = smem_exch_b[NTHREADS - 1];                    \
                smem_run_a[n] = block_a * run_a;                              \
                smem_run_b[n] = block_a * run_b + block_b;                    \
            }                                                                 \
            __syncthreads();                                                  \
            for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {        \
                int t = chunk_start + s;                                      \
                smem_stage[s] = (t < T)                                       \
                    ? C[bid * T * d_state + t * d_state + n]                  \
                    : FROM_F(0.0f);                                           \
            }                                                                 \
            __syncthreads();                                                  \
                                                             \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    float comp_a = thread_a[i] * excl_a;                      \
                    float comp_b = thread_a[i] * excl_b + thread_b[i];        \
                    float final_a = comp_a * run_a;                           \
                    float final_b = comp_a * run_b + comp_b;                  \
                    float h_t = final_a * h_0 + final_b;                      \
                    int hs_idx = (bid * (T + 1) + (t + 1)) * d_inner * d_state\
                                 + did * d_state + n;                         \
                    h_saved[hs_idx] = h_t;                                    \
                    float c_t = to_f(smem_stage[threadIdx.x * NITEMS + i]);   \
                    out_vals[i] += h_t * c_t;                                 \
                }                                                             \
            }                                                                 \
            __syncthreads();                                                  \
        }                                                                     \
                                                             \
        for (int i = 0; i < NITEMS; i++) {                                    \
            smem_stage[threadIdx.x * NITEMS + i] = FROM_F(out_vals[i]);       \
        }                                                                     \
        __syncthreads();                                                      \
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                          \
            if (t < T) {                                                      \
                y_out[(bid * T + t) * d_inner + did] = smem_stage[s];         \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        float h_0 = h[h_base + n];                                            \
        h[h_base + n] = smem_run_a[n] * h_0 + smem_run_b[n];                  \
    }                                                                         \
    (void)da_exp_out;                                                         \
}

DEFINE_SSM_PARALLEL_SCAN_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_PARALLEL_SCAN_FWD(f16,  __half,        from_f_f16)

#define DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(SUFFIX, T_ACT, FROM_F)            \
extern "C" __global__ __launch_bounds__(128, 4) void                          \
ssm_parallel_scan_fwd_nosave_##SUFFIX(                                        \
    float* __restrict__ h,                                                    \
    T_ACT* __restrict__ y_out,                                                \
    const T_ACT* __restrict__ delta,                                          \
    const T_ACT* __restrict__ u,                                              \
    const T_ACT* __restrict__ B,                                              \
    const T_ACT* __restrict__ C,                                              \
    const float* __restrict__ a_neg,                                          \
    const float* __restrict__ D,                                              \
    int batch, int T, int d_inner, int d_state                                \
) {                                                                           \
    int bid = blockIdx.x;                                                     \
    int did = blockIdx.y;                                                     \
    if (bid >= batch || did >= d_inner) return;                               \
    if (d_state > MAX_DSTATE) return;                                         \
    extern __shared__ float smem[];                                           \
    float *smem_wa     = smem + SMEM_WA_OFF;                                  \
    float *smem_wb     = smem + SMEM_WB_OFF;                                  \
    float *smem_run_a  = smem + SMEM_RUN_A_OFF;                               \
    float *smem_run_b  = smem + SMEM_RUN_B_OFF;                               \
    float *smem_exch_a = smem + SMEM_EXCH_A_OFF;                              \
    float *smem_exch_b = smem + SMEM_EXCH_B_OFF;                              \
    /* Typed smem stage (audit Agent 5 #1): 2-byte slots vs 4-byte f32. */    \
    T_ACT *smem_stage = (T_ACT *)(smem + SMEM_STAGE_OFF);                     \
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
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                          \
            smem_stage[s] = (t < T)                                           \
                ? delta[(bid * T + t) * d_inner + did]                        \
                : FROM_F(0.0f);                                               \
        }                                                                     \
        __syncthreads();                                                      \
        float delta_vals[NITEMS];                                             \
                                                             \
        for (int i = 0; i < NITEMS; i++) {                                    \
            delta_vals[i] = to_f(smem_stage[threadIdx.x * NITEMS + i]);       \
        }                                                                     \
        __syncthreads();                                                      \
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                          \
            smem_stage[s] = (t < T)                                           \
                ? u[(bid * T + t) * d_inner + did]                            \
                : FROM_F(0.0f);                                               \
        }                                                                     \
        __syncthreads();                                                      \
        float u_vals[NITEMS];                                                 \
        float delta_u_vals[NITEMS];                                           \
                                                             \
        for (int i = 0; i < NITEMS; i++) {                                    \
            u_vals[i] = to_f(smem_stage[threadIdx.x * NITEMS + i]);           \
            delta_u_vals[i] = delta_vals[i] * u_vals[i];                      \
        }                                                                     \
        __syncthreads();                                                      \
        float out_vals[NITEMS];                                               \
                                                             \
        for (int i = 0; i < NITEMS; i++) {                                    \
            out_vals[i] = D_d * u_vals[i];                                    \
        }                                                                     \
        for (int n = 0; n < d_state; n++) {                                   \
            float a_dn = a_neg[did * d_state + n] * LOG2E;                    \
            for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {        \
                int t = chunk_start + s;                                      \
                smem_stage[s] = (t < T)                                       \
                    ? B[bid * T * d_state + t * d_state + n]                  \
                    : FROM_F(0.0f);                                           \
            }                                                                 \
            __syncthreads();                                                  \
            float thread_a[NITEMS];                                           \
            float thread_b[NITEMS];                                           \
                                                             \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    float da = exp2f(delta_vals[i] * a_dn);                   \
                    float b_t = to_f(smem_stage[threadIdx.x * NITEMS + i]);   \
                    thread_a[i] = da;                                         \
                    thread_b[i] = delta_u_vals[i] * b_t;                      \
                } else {                                                      \
                    thread_a[i] = 1.0f;                                       \
                    thread_b[i] = 0.0f;                                       \
                }                                                             \
            }                                                                 \
            __syncthreads();                                                  \
                                                             \
            for (int i = 1; i < NITEMS; i++) {                                \
                thread_b[i] = thread_a[i] * thread_b[i - 1] + thread_b[i];    \
                thread_a[i] = thread_a[i] * thread_a[i - 1];                  \
            }                                                                 \
            float scan_a = thread_a[NITEMS - 1];                              \
            float scan_b = thread_b[NITEMS - 1];                              \
            __syncthreads();                                                  \
            block_inclusive_scan_ab(scan_a, scan_b, smem_wa, smem_wb);        \
            __syncthreads();                                                  \
            smem_exch_a[threadIdx.x] = scan_a;                                \
            smem_exch_b[threadIdx.x] = scan_b;                                \
            __syncthreads();                                                  \
            float excl_a, excl_b;                                             \
            if (threadIdx.x == 0) {                                           \
                excl_a = 1.0f;                                                \
                excl_b = 0.0f;                                                \
            } else {                                                          \
                excl_a = smem_exch_a[threadIdx.x - 1];                        \
                excl_b = smem_exch_b[threadIdx.x - 1];                        \
            }                                                                 \
            float run_a = smem_run_a[n];                                      \
            float run_b = smem_run_b[n];                                      \
            float h_0 = h[h_base + n];                                        \
            if (threadIdx.x == 0) {                                           \
                float block_a = smem_exch_a[NTHREADS - 1];                    \
                float block_b = smem_exch_b[NTHREADS - 1];                    \
                smem_run_a[n] = block_a * run_a;                              \
                smem_run_b[n] = block_a * run_b + block_b;                    \
            }                                                                 \
            __syncthreads();                                                  \
            for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {        \
                int t = chunk_start + s;                                      \
                smem_stage[s] = (t < T)                                       \
                    ? C[bid * T * d_state + t * d_state + n]                  \
                    : FROM_F(0.0f);                                           \
            }                                                                 \
            __syncthreads();                                                  \
                                                             \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    float comp_a = thread_a[i] * excl_a;                      \
                    float comp_b = thread_a[i] * excl_b + thread_b[i];        \
                    float final_a = comp_a * run_a;                           \
                    float final_b = comp_a * run_b + comp_b;                  \
                    float h_t = final_a * h_0 + final_b;                      \
                    float c_t = to_f(smem_stage[threadIdx.x * NITEMS + i]);   \
                    out_vals[i] += h_t * c_t;                                 \
                }                                                             \
            }                                                                 \
            __syncthreads();                                                  \
        }                                                                     \
                                                             \
        for (int i = 0; i < NITEMS; i++) {                                    \
            smem_stage[threadIdx.x * NITEMS + i] = FROM_F(out_vals[i]);       \
        }                                                                     \
        __syncthreads();                                                      \
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                          \
            if (t < T) {                                                      \
                y_out[(bid * T + t) * d_inner + did] = smem_stage[s];         \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        float h_0 = h[h_base + n];                                            \
        h[h_base + n] = smem_run_a[n] * h_0 + smem_run_b[n];                  \
    }                                                                         \
}

DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(f16,  __half,        from_f_f16)

// ============================================================================
// Step 8e — parallel selective-scan BACKWARD pass.
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
extern "C" __global__ __launch_bounds__(128, 3) void                          \
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
    int batch, int T, int d_inner, int d_state                                \
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
    float D_d = D[did];                                                       \
    int hsave_base_b = bid * (T + 1) * d_inner * d_state;                     \
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
        /* ---- Load typed delta into smem and upcast per-thread ---- */      \
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                          \
            smem_stage[s] = (t < T)                                           \
                ? delta[(bid * T + t) * d_inner + did]                        \
                : FROM_F(0.0f);                                               \
        }                                                                     \
        __syncthreads();                                                      \
        float delta_vals[NITEMS];                                             \
        for (int i = 0; i < NITEMS; i++) {                                    \
            delta_vals[i] = to_f(smem_stage[threadIdx.x * NITEMS + i]);       \
        }                                                                     \
        __syncthreads();                                                      \
        /* ---- Load typed u ---- */                                          \
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                          \
            smem_stage[s] = (t < T)                                           \
                ? u[(bid * T + t) * d_inner + did]                            \
                : FROM_F(0.0f);                                               \
        }                                                                     \
        __syncthreads();                                                      \
        float u_vals[NITEMS];                                                 \
        for (int i = 0; i < NITEMS; i++) {                                    \
            u_vals[i] = to_f(smem_stage[threadIdx.x * NITEMS + i]);           \
        }                                                                     \
        __syncthreads();                                                      \
        /* ---- Load typed dy ---- */                                         \
        for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {            \
            int t = chunk_start + s;                                          \
            smem_stage[s] = (t < T)                                           \
                ? dy[(bid * T + t) * d_inner + did]                           \
                : FROM_F(0.0f);                                               \
        }                                                                     \
        __syncthreads();                                                      \
        float dy_vals[NITEMS];                                                \
        for (int i = 0; i < NITEMS; i++) {                                    \
            dy_vals[i] = to_f(smem_stage[threadIdx.x * NITEMS + i]);          \
        }                                                                     \
        __syncthreads();                                                      \
        /* ---- Per-t skip-path contributions accumulate in registers ---- */ \
        float d_u_acc[NITEMS];                                                \
        float d_delta_acc[NITEMS];                                            \
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
            /* Load typed B[chunk, n] */                                      \
            for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {        \
                int t = chunk_start + s;                                      \
                smem_stage[s] = (t < T)                                       \
                    ? B_in[bid * T * d_state + t * d_state + n]               \
                    : FROM_F(0.0f);                                           \
            }                                                                 \
            __syncthreads();                                                  \
            float b_vals[NITEMS];                                             \
            for (int i = 0; i < NITEMS; i++) {                                \
                b_vals[i] = to_f(smem_stage[threadIdx.x * NITEMS + i]);       \
            }                                                                 \
            __syncthreads();                                                  \
            /* Load typed C[chunk, n] */                                      \
            for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {        \
                int t = chunk_start + s;                                      \
                smem_stage[s] = (t < T)                                       \
                    ? C_in[bid * T * d_state + t * d_state + n]               \
                    : FROM_F(0.0f);                                           \
            }                                                                 \
            __syncthreads();                                                  \
            float c_vals[NITEMS];                                             \
            for (int i = 0; i < NITEMS; i++) {                                \
                c_vals[i] = to_f(smem_stage[threadIdx.x * NITEMS + i]);       \
            }                                                                 \
            __syncthreads();                                                  \
            /* Per-i: da[i] = exp2(delta * a_neg), d_local[i] = dy * c */     \
            float da_vals[NITEMS];                                            \
            float d_local[NITEMS];                                            \
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
            for (int i = 0; i < NITEMS; i++) {                                \
                /* dh[i] = thread_a[i]*post_b + thread_b[i] */                \
                dh_vals[i] = thread_a[i] * post_b + thread_b[i];              \
            }                                                                 \
            /* ---- Per-t output writes (typed) and accumulation ---- */      \
            float d_a_acc = 0.0f;                                             \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t >= T) continue;                                         \
                int btdn_typed = ((bid * T + t) * d_inner + did) * d_state    \
                                 + n;                                         \
                int h_curr_idx = hsave_base_b                                 \
                    + (t + 1) * d_inner * d_state                             \
                    + did * d_state + n;                                      \
                int h_prev_idx = hsave_base_b                                 \
                    + t * d_inner * d_state                                   \
                    + did * d_state + n;                                      \
                float h_curr = h_saved[h_curr_idx];                           \
                float h_prev = h_saved[h_prev_idx];                           \
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
            for (int stride = NTHREADS / 2; stride > 0; stride >>= 1) {       \
                if ((int)threadIdx.x < stride) {                              \
                    smem_da_red[threadIdx.x] += smem_da_red[threadIdx.x +     \
                                                            stride];          \
                }                                                             \
                __syncthreads();                                              \
            }                                                                 \
            if (threadIdx.x == 0) {                                           \
                d_a_log_local[(bid * d_inner + did) * d_state + n]            \
                    += smem_da_red[0];                                        \
                /* Save THIS chunk's first thread's first da into the         \
                   chunk_first_a[n] slot — the EARLIER chunk (next iter)      \
                   will read this as its boundary `a_{t+1}` for the very-     \
                   last timestep before this chunk starts. */                 \
                smem_chunk_first_a[n] = smem_next_a[0];                       \
            }                                                                 \
            __syncthreads();                                                  \
        }                                                                     \
        /* ---- Store d_delta_acc, d_u_acc to typed HBM via smem ---- */      \
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
    for (int stride = NTHREADS / 2; stride > 0; stride >>= 1) {               \
        if ((int)threadIdx.x < stride) {                                      \
            smem_da_red[threadIdx.x] += smem_da_red[threadIdx.x + stride];    \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    if (threadIdx.x == 0) {                                                   \
        d_D_local[bid * d_inner + did] = smem_da_red[0];                      \
    }                                                                         \
}

DEFINE_SSM_PARALLEL_SCAN_BWD(f32,  float,         from_f_f32)
DEFINE_SSM_PARALLEL_SCAN_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_PARALLEL_SCAN_BWD(f16,  __half,        from_f_f16)

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
