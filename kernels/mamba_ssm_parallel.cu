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
__device__ __forceinline__ void warp_inclusive_scan_ab(float &a, float &b) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float a_prev = __shfl_up_sync(0xffffffff, a, offset);
        float b_prev = __shfl_up_sync(0xffffffff, b, offset);
        if ((threadIdx.x & 31) >= (unsigned)offset) {
            b = a * b_prev + b;
            a = a * a_prev;
        }
    }
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

    // Step 3: first warp scans the NWARPS totals
    if (warp_id == 0 && lane < NWARPS) {
        float wa = smem_wa[lane];
        float wb = smem_wb[lane];
        warp_inclusive_scan_ab(wa, wb);
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
