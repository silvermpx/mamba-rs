// The Mamba-1 parallel scan forwards as they were released, with and
// without their saves: a block of 128 threads took one channel, eight steps
// a thread, so a window shorter than the block's 1024-step chunk left all
// but a few of its threads stepping the identity. The forwards that hold a
// group of channels in a block, each in the fewest threads that cover its
// window, are held to these bit for bit. The typed prelude precedes this
// fixture.
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

// The store twin of load_row8: a thread's eight values at t0 .. t0+7 of one
// t-major row, as 16-byte vectors when the span lies inside the row and is
// 16-byte aligned, element by element otherwise.
template <typename T>
__device__ __forceinline__ void store_row8(
    T* __restrict__ row, int t0, int T_len, const T vals[NITEMS]
) {
    T* p = row + t0;
    if (t0 + NITEMS <= T_len && ((unsigned long long)p & 15ull) == 0) {
        constexpr int PER16 = 16 / (int)sizeof(T);
        #pragma unroll
        for (int k = 0; k < NITEMS / PER16; k++) {
            uint4 v;
            T* e = reinterpret_cast<T*>(&v);
            #pragma unroll
            for (int j = 0; j < PER16; j++) e[j] = vals[k * PER16 + j];
            reinterpret_cast<uint4*>(p)[k] = v;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < NITEMS; i++)
            if (t0 + i < T_len) p[i] = vals[i];
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
// Block-level inclusive scan of (a, b) pairs that also hands every thread
// its exclusive prefix without a third barrier. The inclusive values are
// exactly those of `block_inclusive_scan_ab`. A thread's exclusive prefix
// is the composed inclusive value of the thread before it: a lane takes it
// from the lane above by shuffle, and a warp's first lane rebuilds the
// previous warp's last-lane value from that warp's raw total (kept in the
// raw slots, since the scanned slots overwrite the totals) composed with
// the same prefix that lane composed with, in the same expression.
// ============================================================================
__device__ __forceinline__ void block_scan_ab_with_exclusive(
    float &a, float &b, float &ea, float &eb,
    float *smem_wa, float *smem_wb, float *smem_raw_a, float *smem_raw_b
) {
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    warp_inclusive_scan_ab(a, b);
    if (lane == 31) {
        smem_wa[warp_id] = a;
        smem_wb[warp_id] = b;
        smem_raw_a[warp_id] = a;
        smem_raw_b[warp_id] = b;
    }
    __syncthreads();

    if (warp_id == 0 && lane < NWARPS) {
        float wa = smem_wa[lane];
        float wb = smem_wb[lane];
        warp_inclusive_scan_ab(wa, wb, (1u << NWARPS) - 1u);
        smem_wa[lane] = wa;
        smem_wb[lane] = wb;
    }
    __syncthreads();

    if (warp_id > 0) {
        float pa = smem_wa[warp_id - 1];
        float pb = smem_wb[warp_id - 1];
        b = a * pb + b;
        a = a * pa;
    }
    // Every lane of the warp rebuilds the value its first lane needs, so
    // the branches below depend on the warp index only and never diverge;
    // the first lane then keeps it and the others keep the shuffle.
    float edge_a = 1.0f;
    float edge_b = 0.0f;
    if (warp_id > 0) {
        edge_a = smem_raw_a[warp_id - 1];
        edge_b = smem_raw_b[warp_id - 1];
        if (warp_id > 1) {
            float pa = smem_wa[warp_id - 2];
            float pb = smem_wb[warp_id - 2];
            edge_b = edge_a * pb + edge_b;
            edge_a = edge_a * pa;
        }
    }
    float up_a = __shfl_up_sync(0xffffffffu, a, 1);
    float up_b = __shfl_up_sync(0xffffffffu, b, 1);
    ea = (lane == 0) ? edge_a : up_a;
    eb = (lane == 0) ? edge_b : up_b;
    // No __syncthreads here -- caller syncs before the workspace is reused.
}

// ============================================================================
// Reverse mirror of `block_scan_ab_with_exclusive`: the inclusive values of
// `block_inclusive_reverse_scan_ab`, plus each thread's exclusive postfix
// (the composed value of the thread after it) from the lane below or, on
// a warp's last lane, rebuilt from the next warp's raw total.
// ============================================================================
__device__ __forceinline__ void block_reverse_scan_ab_with_exclusive(
    float &a, float &b, float &na, float &nb,
    float *smem_wa, float *smem_wb, float *smem_raw_a, float *smem_raw_b
) {
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    warp_inclusive_reverse_scan_ab(a, b);
    if (lane == 0) {
        smem_wa[warp_id] = a;
        smem_wb[warp_id] = b;
        smem_raw_a[warp_id] = a;
        smem_raw_b[warp_id] = b;
    }
    __syncthreads();

    if (warp_id == 0 && lane < NWARPS) {
        float wa = smem_wa[lane];
        float wb = smem_wb[lane];
        warp_inclusive_reverse_scan_ab(wa, wb, (1u << NWARPS) - 1u, NWARPS);
        smem_wa[lane] = wa;
        smem_wb[lane] = wb;
    }
    __syncthreads();

    if (warp_id < NWARPS - 1) {
        float pa = smem_wa[warp_id + 1];
        float pb = smem_wb[warp_id + 1];
        b = a * pb + b;
        a = a * pa;
    }
    float edge_a = 1.0f;
    float edge_b = 0.0f;
    if (warp_id < NWARPS - 1) {
        edge_a = smem_raw_a[warp_id + 1];
        edge_b = smem_raw_b[warp_id + 1];
        if (warp_id + 1 < NWARPS - 1) {
            float pa = smem_wa[warp_id + 2];
            float pb = smem_wb[warp_id + 2];
            edge_b = edge_a * pb + edge_b;
            edge_a = edge_a * pa;
        }
    }
    float down_a = __shfl_down_sync(0xffffffffu, a, 1);
    float down_b = __shfl_down_sync(0xffffffffu, b, 1);
    na = (lane == 31) ? edge_a : down_a;
    nb = (lane == 31) ? edge_b : down_b;
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
// Forward: parallel prefix scan with activation saves (training path), and
// its twin without saves (target network and prefill).
//
// Same interface and saved activations as ssm_burnin_forward. Y accumulates
// in one pass inside the d_state loop. Grid: (batch, d_inner). Block:
// NTHREADS. Shared memory: the forward layout described above, sized by the
// launcher.
//
// One body serves every activation type. The scan state, the running
// prefix, the block scan and every register stay f32, the scan_t = float2
// of state-spaces/mamba: only the activations (delta, u, B, C, y_out) take
// T_ACT, and the state, its tape, the parameters and all of shared memory
// stay f32. With T_ACT = float the conversions are the identity, and the f32
// build compiles to the instructions of a kernel written for f32 alone.
//
// Each type is built twice: with int offsets under the kernel's own name,
// and with long long offsets (IDX) under the same name with `_i64`
// appended. The 64-bit offsets cost the scan enough registers on sm_89 to
// lose a resident block per SM, so the launcher takes the int build
// whenever every offset of the launch fits the 32-bit range, and the `_i64`
// build only past it. Both builds do the same arithmetic on the same values,
// so their results agree bit for bit. The body is a macro, not a template
// inlined into two entry points: inlined, its address arithmetic compiled
// differently on sm_120, and the int build stopped being the 32-bit code.
// ============================================================================

#define DEFINE_SSM_PARALLEL_SCAN_FWD(NAME, IDX, T_ACT, FROM_F)                \
extern "C" __global__ __launch_bounds__(NTHREADS, SCAN_MINB) void             \
NAME(                                                                         \
    float* __restrict__ h,               /* [batch, d_inner, d_state] */      \
    T_ACT* __restrict__ y_out,           /* [batch, T, d_inner] */            \
    float* __restrict__ h_saved,         /* [batch, d_inner, d_state, T+1] */ \
    /* Pre-softplus dt: the kernel applies the softplus and writes the        \
       post-softplus save the backward replays from. */                       \
    const T_ACT* __restrict__ delta_raw, /* [batch, T, d_inner] */            \
    T_ACT* __restrict__ delta_saved,     /* [batch, T, d_inner] */            \
    const T_ACT* __restrict__ u,         /* [batch, T, d_inner] */            \
    const T_ACT* __restrict__ B,         /* [batch, d_state, T] */            \
    const T_ACT* __restrict__ C,         /* [batch, d_state, T] */            \
    const float* __restrict__ a_neg,     /* [d_inner, d_state] */             \
    const float* __restrict__ D,         /* [d_inner] */                      \
    int batch, int T, int d_inner, int d_state,                               \
    /* Slim tape: rows of (run_a, run_b, h_entry) per chunk,                  \
       [batch, d_inner, d_state, 3 * n_chunks]. No __restrict__: under        \
       slim the launcher passes the buffer of h_saved, and a launch           \
       touches only one of the two. */                                        \
    float* run_tape,                                                          \
    int slim_tape,                                                            \
    int y_first /* this run's first y block */                                \
) {                                                                           \
    IDX bid = blockIdx.x;                                                     \
    int did = y_first + blockIdx.y;                                           \
    if (bid >= batch || did >= d_inner) return;                               \
    if (d_state > MAX_DSTATE) return;                                         \
    extern __shared__ float smem[];                                           \
    float *smem_wa     = smem + SMEM_WA_OFF;                                  \
    float *smem_wb     = smem + SMEM_WB_OFF;                                  \
    /* The run and exchange rows pack at the runtime d_state, not at          \
       MAX_DSTATE, and the launcher sizes the allocation to match. */         \
    float *smem_run_a  = smem + 2 * NWARPS;                                   \
    float *smem_run_b  = smem_run_a + d_state;                                \
    float *smem_exch_a = smem_run_b + d_state;                                \
    float *smem_exch_b = smem_exch_a + NWARPS;                                \
    float D_d = D[did];                                                       \
    IDX h_base = (bid * d_inner + did) * d_state;                             \
    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;                         \
    /* The initial state: a slim tape row opens with the identity prefix      \
       and h_0 itself, the entry state of chunk 0. */                         \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        if (slim_tape) {                                                      \
            IDX row = ((bid * d_inner + did) * d_state + n) * 3 * n_chunks;   \
            run_tape[row + 0] = 1.0f;                                         \
            run_tape[row + 1] = 0.0f;                                         \
            run_tape[row + 2] = h[h_base + n];                                \
        } else {                                                              \
            /* The full tape is t-major, [b][d][n][t+1]: the lane stride      \
               over t is one element, so warp stores and loads coalesce. */   \
            IDX hs_idx =                                                      \
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
        /* Direct per-thread loads. Adjacent t are d_inner elements apart,    \
         * so the threads of one block never share a sector whichever of      \
         * them loads, and staging the chunk through shared memory only       \
         * added barriers. */                                                 \
        float delta_vals[NITEMS];                                             \
        float u_vals[NITEMS];                                                 \
        float delta_u_vals[NITEMS];                                           \
        _Pragma("unroll")                                                     \
        for (int i = 0; i < NITEMS; i++) {                                    \
            int t = chunk_start + threadIdx.x * NITEMS + i;                   \
            if (t < T) {                                                      \
                /* The scan consumes the softplus after the rounding of       \
                   its save, the value the backward reads back. */            \
                float raw = to_f(delta_raw[(bid * T + t) * d_inner + did]);   \
                float sp = (raw > 20.0f)                                      \
                    ? raw                                                     \
                    : log1pf(exp2f(raw * 1.4426950408889634f));               \
                T_ACT spt = FROM_F(sp);                                       \
                delta_saved[(bid * T + t) * d_inner + did] = spt;             \
                delta_vals[i] = to_f(spt);                                    \
            } else {                                                          \
                delta_vals[i] = 0.0f;                                         \
            }                                                                 \
            u_vals[i] = (t < T) ? to_f(u[(bid * T + t) * d_inner + did])      \
                                : 0.0f;                                       \
            delta_u_vals[i] = delta_vals[i] * u_vals[i];                      \
        }                                                                     \
        float out_vals[NITEMS];                                               \
                                                                              \
        _Pragma("unroll")                                                     \
        for (int i = 0; i < NITEMS; i++) {                                    \
            out_vals[i] = D_d * u_vals[i];                                    \
        }                                                                     \
        for (int n = 0; n < d_state; n++) {                                   \
            float a_dn = a_neg[(IDX)did * d_state + n] * LOG2E;               \
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
            _Pragma("unroll")                                                 \
            for (int i = 1; i < NITEMS; i++) {                                \
                thread_b[i] = thread_a[i] * thread_b[i - 1] + thread_b[i];    \
                thread_a[i] = thread_a[i] * thread_a[i - 1];                  \
            }                                                                 \
            float scan_a = thread_a[NITEMS - 1];                              \
            float scan_b = thread_b[NITEMS - 1];                              \
            __syncthreads();                                                  \
            block_inclusive_scan_ab(scan_a, scan_b, smem_wa, smem_wb);        \
            /* Exclusive prefix of thread t, the inclusive value of thread    \
               t - 1: the lane above hands it down through a shuffle, a       \
               warp's first lane takes it from the previous warp's last       \
               lane through a per-warp slot, and thread 0 takes the           \
               identity. The last slot is the block total thread 0 folds      \
               into the running prefix. Every warp reads the running          \
               prefix before the barrier, so thread 0 may overwrite it        \
               right after. */                                                \
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
            /* Slim tape: this chunk's entry prefix, one writer. */           \
            if (slim_tape && chunk > 0 && threadIdx.x == 0) {                 \
                IDX row =                                                     \
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
            _Pragma("unroll")                                                 \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    float comp_a = thread_a[i] * excl_a;                      \
                    float comp_b = thread_a[i] * excl_b + thread_b[i];        \
                    float final_a = comp_a * run_a;                           \
                    float final_b = comp_a * run_b + comp_b;                  \
                    float h_t = final_a * h_0 + final_b;                      \
                    /* The full tape keeps every step. The slim one keeps     \
                       the next chunk's entry state, the value the            \
                       backward's boundary read loads. */                     \
                    if (slim_tape) {                                          \
                        if ((t + 1) % CHUNK_SIZE == 0 && t + 1 < T) {         \
                            IDX row =                                         \
                                ((bid * d_inner + did) * d_state + n)         \
                                * 3 * n_chunks;                               \
                            run_tape[row + 3 * (chunk + 1) + 2] = h_t;        \
                        }                                                     \
                    } else {                                                  \
                        IDX hs_idx = ((bid * d_inner + did) * d_state + n)    \
                                     * (T + 1) + (t + 1);                     \
                        h_saved[hs_idx] = h_t;                                \
                    }                                                         \
                    float c_t = c_row[i];                                     \
                    out_vals[i] += h_t * c_t;                                 \
                }                                                             \
            }                                                                 \
        }                                                                     \
                                                                              \
        /* Each thread stores its own items. Within a block did is            \
           fixed, so neighbouring t land in separate sectors whichever        \
           thread stores them. */                                             \
        _Pragma("unroll")                                                     \
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

DEFINE_SSM_PARALLEL_SCAN_FWD(ssm_parallel_scan_fwd,          int,       float,         from_f_f32)
DEFINE_SSM_PARALLEL_SCAN_FWD(ssm_parallel_scan_fwd_i64,      long long, float,         from_f_f32)
DEFINE_SSM_PARALLEL_SCAN_FWD(ssm_parallel_scan_fwd_bf16,     int,       __nv_bfloat16, from_f_bf16)
DEFINE_SSM_PARALLEL_SCAN_FWD(ssm_parallel_scan_fwd_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_PARALLEL_SCAN_FWD(ssm_parallel_scan_fwd_f16,      int,       __half,        from_f_f16)
DEFINE_SSM_PARALLEL_SCAN_FWD(ssm_parallel_scan_fwd_f16_i64,  long long, __half,        from_f_f16)

#define DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(NAME, IDX, T_ACT, FROM_F)         \
extern "C" __global__ __launch_bounds__(NTHREADS, SCAN_MINB) void             \
NAME(                                                                         \
    float* __restrict__ h,               /* [batch, d_inner, d_state] */      \
    T_ACT* __restrict__ y_out,                                                \
    const T_ACT* __restrict__ delta,     /* post-softplus */                  \
    const T_ACT* __restrict__ u,                                              \
    const T_ACT* __restrict__ B,         /* [batch, d_state, T] */            \
    const T_ACT* __restrict__ C,         /* [batch, d_state, T] */            \
    const float* __restrict__ a_neg,     /* [d_inner, d_state] */             \
    const float* __restrict__ D,         /* [d_inner] */                      \
    const T_ACT* __restrict__ proj_gate, /* in_proj rows, gate half */        \
    int gate_stride,                     /* 0 stores y ungated */             \
    int batch, int T, int d_inner, int d_state,                               \
    /* > 0: delta, u and y in t-major rows [batch, d_inner, tstride],         \
       y stored ungated for gate_from_tmajor. 0: row-major                    \
       [batch, T, d_inner]. */                                                \
    int tstride,                                                              \
    int y_first /* this run's first y block */                                \
) {                                                                           \
    IDX bid = blockIdx.x;                                                     \
    int did = y_first + blockIdx.y;                                           \
    if (bid >= batch || did >= d_inner) return;                               \
    if (d_state > MAX_DSTATE) return;                                         \
    extern __shared__ float smem[];                                           \
    float *smem_wa     = smem + SMEM_WA_OFF;                                  \
    float *smem_wb     = smem + SMEM_WB_OFF;                                  \
    /* The run and exchange rows pack at the runtime d_state, not at          \
       MAX_DSTATE, and the launcher sizes the allocation to match. */         \
    float *smem_run_a  = smem + 2 * NWARPS;                                   \
    float *smem_run_b  = smem_run_a + d_state;                                \
    float *smem_exch_a = smem_run_b + d_state;                                \
    float *smem_exch_b = smem_exch_a + NWARPS;                                \
    float D_d = D[did];                                                       \
    IDX h_base = (bid * d_inner + did) * d_state;                             \
    for (int n = threadIdx.x; n < d_state; n += NTHREADS) {                   \
        smem_run_a[n] = 1.0f;                                                 \
        smem_run_b[n] = 0.0f;                                                 \
    }                                                                         \
    __syncthreads();                                                          \
    int n_chunks = (T + CHUNK_SIZE - 1) / CHUNK_SIZE;                         \
    for (int chunk = 0; chunk < n_chunks; chunk++) {                          \
        int chunk_start = chunk * CHUNK_SIZE;                                 \
        /* Direct per-thread loads, as in the saving forward. In t-major      \
         * rows a thread's eight times are one span and come in as            \
         * vectors, the way B and C do. */                                    \
        float delta_vals[NITEMS];                                             \
        float u_vals[NITEMS];                                                 \
        float delta_u_vals[NITEMS];                                           \
        long long row = ((long long)bid * d_inner + did) * tstride;           \
        if (tstride > 0) {                                                    \
            load_row8(delta + row, chunk_start + threadIdx.x * NITEMS, T,     \
                      delta_vals);                                            \
            load_row8(u + row, chunk_start + threadIdx.x * NITEMS, T, u_vals); \
        } else {                                                              \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                delta_vals[i] = (t < T)                                       \
                    ? to_f(delta[(bid * T + t) * d_inner + did]) : 0.0f;      \
                u_vals[i] = (t < T) ? to_f(u[(bid * T + t) * d_inner + did])  \
                                    : 0.0f;                                   \
            }                                                                 \
        }                                                                     \
        _Pragma("unroll")                                                     \
        for (int i = 0; i < NITEMS; i++) {                                    \
            delta_u_vals[i] = delta_vals[i] * u_vals[i];                      \
        }                                                                     \
        float out_vals[NITEMS];                                               \
                                                                              \
        _Pragma("unroll")                                                     \
        for (int i = 0; i < NITEMS; i++) {                                    \
            out_vals[i] = D_d * u_vals[i];                                    \
        }                                                                     \
        for (int n = 0; n < d_state; n++) {                                   \
            float a_dn = a_neg[(IDX)did * d_state + n] * LOG2E;               \
            float thread_a[NITEMS];                                           \
            float thread_b[NITEMS];                                           \
            float b_row[NITEMS];                                              \
            load_row8(B + (bid * d_state + n) * T,                            \
                      chunk_start + threadIdx.x * NITEMS, T, b_row);          \
                                                                              \
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
            _Pragma("unroll")                                                 \
            for (int i = 1; i < NITEMS; i++) {                                \
                thread_b[i] = thread_a[i] * thread_b[i - 1] + thread_b[i];    \
                thread_a[i] = thread_a[i] * thread_a[i - 1];                  \
            }                                                                 \
            float scan_a = thread_a[NITEMS - 1];                              \
            float scan_b = thread_b[NITEMS - 1];                              \
            __syncthreads();                                                  \
            block_inclusive_scan_ab(scan_a, scan_b, smem_wa, smem_wb);        \
            /* Exclusive prefix through the lane above and a per-warp         \
               slot, as in the saving forward. The last slot is the block     \
               total. */                                                      \
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
            _Pragma("unroll")                                                 \
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
        /* Direct per-thread stores, as in the saving forward. On the         \
           t-major route a thread's eight times are one span of its           \
           channel's row. */                                                  \
        if (tstride > 0) {                                                    \
            T_ACT y8[NITEMS];                                                 \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < NITEMS; i++) y8[i] = FROM_F(out_vals[i]);     \
            store_row8(y_out + row, chunk_start + threadIdx.x * NITEMS, T, y8); \
        } else {                                                              \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < NITEMS; i++) {                                \
                int t = chunk_start + threadIdx.x * NITEMS + i;               \
                if (t < T) {                                                  \
                    T_ACT ty = FROM_F(out_vals[i]);                           \
                    if (gate_stride > 0) {                                    \
                        /* Each rounding of the chain this store              \
                         * replaced, in place: it stored y and SiLU(gate)     \
                         * in T_ACT, then multiplied the reloaded values      \
                         * with one final rounding. In f32 the stores         \
                         * round nothing. */                                  \
                        float g = to_f(proj_gate[(bid * T + t) * gate_stride  \
                                                 + d_inner + did]);           \
                        T_ACT tg = FROM_F(                                    \
                            g / (1.0f + exp2f(-g * 1.4426950408889634f)));    \
                        ty = FROM_F(to_f(ty) * to_f(tg));                     \
                    }                                                         \
                    y_out[(bid * T + t) * d_inner + did] = ty;                \
                }                                                             \
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

DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(ssm_parallel_scan_fwd_nosave,          int,       float,         from_f_f32)
DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(ssm_parallel_scan_fwd_nosave_i64,      long long, float,         from_f_f32)
DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(ssm_parallel_scan_fwd_nosave_bf16,     int,       __nv_bfloat16, from_f_bf16)
DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(ssm_parallel_scan_fwd_nosave_bf16_i64, long long, __nv_bfloat16, from_f_bf16)
DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(ssm_parallel_scan_fwd_nosave_f16,      int,       __half,        from_f_f16)
DEFINE_SSM_PARALLEL_SCAN_FWD_NOSAVE(ssm_parallel_scan_fwd_nosave_f16_i64,  long long, __half,        from_f_f16)
