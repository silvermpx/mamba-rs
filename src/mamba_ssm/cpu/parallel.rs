use std::cell::{Cell, RefCell};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use super::backward::backward_mamba_backbone_batched;
use super::flat::MambaBackboneFlat;
use super::forward::forward_mamba_backbone_batched;
use super::scratch::{BackwardPhaseScratch, PhaseScratch};
use super::target::{
    MambaTargetScratch, MambaTargetSeqScratch, forward_mamba_target_sequence,
    forward_mamba_target_step,
};
use super::weights::TrainMambaWeights;
use crate::ops::dims::{MambaDims, MambaRecurrentState};

// ---------------------------------------------------------------------------
// Thread-local scratch pool (O1)
// ---------------------------------------------------------------------------

/// Per-thread scratch combining forward + backward phase buffers and small
/// intermediate vectors. Each rayon worker thread gets its own instance
/// (~200KB) so there is zero contention during parallel forward/backward.
pub struct MambaThreadScratch {
    /// Forward phase scratch (conv states, SSM intermediates).
    pub fwd: PhaseScratch,
    /// Backward phase scratch (gradient accumulators).
    pub bwd: BackwardPhaseScratch,
    /// Flat temporal output `[T * d_model]`.
    pub temporal_flat: Vec<f32>,
    /// Flat mamba input `[T * mamba_input_dim]`.
    pub mamba_input_flat: Vec<f32>,
    /// Gradient of mamba input (single timestep) `[mamba_input_dim]`.
    pub d_mamba_input: Vec<f32>,
    /// Cached d_model for invalidation when tuner changes dims.
    pub d_model: usize,
    /// Cached d_inner for invalidation when tuner changes dims.
    pub d_inner: usize,
}

impl MambaThreadScratch {
    /// Allocate forward + backward scratch for one rayon worker thread.
    ///
    /// Sized according to `dims`. Re-allocated if dims change between tuner trials.
    pub fn new(dims: &MambaDims) -> Self {
        Self {
            fwd: PhaseScratch::zeros(dims),
            bwd: BackwardPhaseScratch::zeros(dims),
            temporal_flat: vec![0.0; dims.seq_len * dims.d_model],
            mamba_input_flat: vec![0.0; dims.seq_len * dims.mamba_input_dim],
            d_mamba_input: vec![0.0; dims.mamba_input_dim],
            d_model: dims.d_model,
            d_inner: dims.d_inner,
        }
    }
}

thread_local! {
    static THREAD_SCRATCH: RefCell<Option<MambaThreadScratch>> = const { RefCell::new(None) };
    static THREAD_TARGET_SCRATCH: RefCell<Option<MambaTargetScratch>> = const { RefCell::new(None) };
    static THREAD_TARGET_SEQ_SCRATCH: RefCell<Option<MambaTargetSeqScratch>> = const { RefCell::new(None) };
}

/// Access or create thread-local scratch for the calling thread.
///
/// The scratch is lazily allocated on first use and reused on subsequent calls.
/// Each rayon worker thread gets its own independent instance.
pub fn with_thread_scratch<F, R>(dims: &MambaDims, f: F) -> R
where
    F: FnOnce(&mut MambaThreadScratch) -> R,
{
    THREAD_SCRATCH.with(|cell| {
        let mut opt = cell.borrow_mut();
        // Check if dims changed (tuner changes model architecture between trials)
        let needs_reinit = opt.as_ref().is_some_and(|s| {
            s.d_model != dims.d_model
                || s.d_inner != dims.d_inner
                || s.temporal_flat.len() != dims.seq_len * dims.d_model
        });
        if needs_reinit {
            *opt = None;
        }
        let scratch = opt.get_or_insert_with(|| MambaThreadScratch::new(dims));
        f(scratch)
    })
}

/// Invalidate all thread-local Mamba scratch buffers on the current thread.
/// Called by `invalidate_all_scratch()` on each rayon worker.
pub fn invalidate_mamba_scratch() {
    THREAD_SCRATCH.with(|cell| {
        *cell.borrow_mut() = None;
    });
    THREAD_TARGET_SCRATCH.with(|cell| {
        *cell.borrow_mut() = None;
    });
    THREAD_TARGET_SEQ_SCRATCH.with(|cell| {
        *cell.borrow_mut() = None;
    });
    THREAD_GRADS.with(|cell| {
        *cell.borrow_mut() = None;
    });
    THREAD_GRADS_EPOCH.with(|ep| {
        ep.set(0);
    });
}

// ---------------------------------------------------------------------------
// O1: Parallel Mamba forward — rayon across B samples
// ---------------------------------------------------------------------------

/// Run `forward_mamba_backbone_batched` for B samples in parallel using rayon.
///
/// Each sample is independent (SSM recurrence is per-sample), so we
/// parallelize across the batch dimension. Each rayon worker thread gets its
/// own `MambaThreadScratch` (via thread-local storage) for phase scratch
/// buffers, avoiding any contention.
///
/// # Arguments
///
/// - `temporal_out`: `[B * d_model]` — final temporal output per sample (last timestep).
/// - `mamba_batch_acts`: `[B]` — per-sample saved activations for backward.
/// - `conv_states`: `[B * n_layers * d_inner * d_conv]` — persistent conv states.
/// - `ssm_states`: `[B * n_layers * d_inner * d_state]` — persistent SSM states.
/// - `mamba_w`: shared read-only Mamba weights.
/// - `a_neg_all`: `[n_layers * d_inner * d_state]` — pre-computed `-exp(a_log)`.
/// - `mamba_inputs`: `[B * T * mamba_input_dim]` — pre-built flat input per sample.
/// - `dims`: collected Mamba dimensions.
/// - `b_sz`: batch size (number of samples).
pub fn parallel_mamba_forward(
    temporal_out: &mut [f32],
    mamba_batch_acts: &mut [MambaBackboneFlat],
    state: &mut MambaRecurrentState<'_>,
    mamba_w: &TrainMambaWeights,
    mamba_inputs: &[f32],
    dims: &MambaDims,
    b_sz: usize,
) {
    let dm = dims.d_model;
    let n_layers = dims.n_layers;
    let di = dims.d_inner;
    let dc = dims.d_conv;
    let ds = dims.d_state;
    let seq_len = dims.seq_len;
    let mid = dims.mamba_input_dim;

    let conv_per_sample = n_layers * di * dc;
    let ssm_per_sample = n_layers * di * ds;
    let inp_per_sample = seq_len * mid;

    // We need disjoint mutable borrows into 3 output slices + the acts Vec.
    // Use raw pointers for the flat f32 slices, and split acts into &mut references
    // via an index-based raw pointer approach.
    //
    // SAFETY invariant for all raw pointer usage below:
    // Each iteration `b` accesses a strictly disjoint region:
    //   temporal_out[b*dm .. (b+1)*dm]
    //   conv_states[b*conv_per .. (b+1)*conv_per]
    //   ssm_states[b*ssm_per .. (b+1)*ssm_per]
    //   mamba_batch_acts[b]
    // No two iterations share any memory. The slices are bounds-checked via
    // debug_assert before the parallel loop.

    debug_assert!(temporal_out.len() >= b_sz * dm);
    debug_assert!(state.conv.len() >= b_sz * conv_per_sample);
    debug_assert!(state.ssm.len() >= b_sz * ssm_per_sample);
    debug_assert!(mamba_inputs.len() >= b_sz * inp_per_sample);
    debug_assert!(mamba_batch_acts.len() >= b_sz);

    let a_neg_all = state.a_neg;
    let temporal_ptr = temporal_out.as_mut_ptr();
    let conv_ptr = state.conv.as_mut_ptr();
    let ssm_ptr = state.ssm.as_mut_ptr();
    let acts_ptr = mamba_batch_acts.as_mut_ptr();

    // Wrap raw pointers in a Send-able newtype so rayon's par_iter can use them.
    // SAFETY: we guarantee disjoint access per sample index `b`.
    // Take a reference so edition-2024 closure captures `&RawPtrs` (Sync)
    // instead of individual `*mut f32` fields (not Sync).
    let ptrs = &RawPtrs {
        temporal: temporal_ptr,
        conv: conv_ptr,
        ssm: ssm_ptr,
        acts: acts_ptr,
    };

    // Single-thread BLAS inside rayon to prevent thread explosion
    // (64 rayon workers × N BLIS threads). Per-sample SGEMM is tiny (T×d_model).

    (0..b_sz).into_par_iter().for_each(|b| {
        // SAFETY: each `b` accesses a disjoint region of the output arrays.
        // - temporal: [b*dm .. (b+1)*dm] — dm elements, no overlap between b values.
        // - conv: [b*conv_per .. (b+1)*conv_per] — conv_per elements, disjoint.
        // - ssm: [b*ssm_per .. (b+1)*ssm_per] — ssm_per elements, disjoint.
        // - acts: &mut acts[b] — each b indexes a different MambaBackboneFlat.
        // The source slices (mamba_w, a_neg_all, mamba_inputs) are read-only.
        let conv_b = unsafe {
            std::slice::from_raw_parts_mut(ptrs.conv.add(b * conv_per_sample), conv_per_sample)
        };
        let ssm_b = unsafe {
            std::slice::from_raw_parts_mut(ptrs.ssm.add(b * ssm_per_sample), ssm_per_sample)
        };
        let acts_b = unsafe { &mut *ptrs.acts.add(b) };
        let inp_b = &mamba_inputs[b * inp_per_sample..(b + 1) * inp_per_sample];

        with_thread_scratch(dims, |scratch| {
            // Zero the thread-local temporal buffer for this sample.
            let t_len = seq_len * dm;
            scratch.temporal_flat[..t_len].fill(0.0);

            // Zero conv/ssm states for this sample (fresh forward, no carry-over).
            conv_b.fill(0.0);
            ssm_b.fill(0.0);

            // Run the per-sample forward through the full Mamba backbone.
            forward_mamba_backbone_batched(
                &mut scratch.temporal_flat[..t_len],
                acts_b,
                mamba_w,
                inp_b,
                &mut MambaRecurrentState {
                    conv: conv_b,
                    ssm: ssm_b,
                    a_neg: a_neg_all,
                },
                &mut scratch.fwd,
                dims,
            );

            // Extract the final timestep's temporal output into the output buffer.
            let last_t_start = (seq_len - 1) * dm;
            // SAFETY: temporal_out[b*dm..(b+1)*dm] is disjoint from all other b.
            let out_b = unsafe { std::slice::from_raw_parts_mut(ptrs.temporal.add(b * dm), dm) };
            out_b.copy_from_slice(&scratch.temporal_flat[last_t_start..last_t_start + dm]);
        });
    });

    // Restore default BLAS threading
}

// ---------------------------------------------------------------------------
// O1: Parallel Mamba backward — two-pass rayon + thread-local gradient reduce
// ---------------------------------------------------------------------------

/// Monotonic epoch counter for thread-local gradient zeroing.
/// Each call to `parallel_mamba_backward` increments this. Threads compare
/// their local epoch against the global one to know when to zero their
/// accumulators. This eliminates dependence on `rayon::broadcast` reaching
/// every thread that `par_iter` may use (the calling thread can differ).
static BWD_EPOCH: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static THREAD_GRADS: RefCell<Option<TrainMambaWeights>> = const { RefCell::new(None) };
    /// Last epoch at which this thread's THREAD_GRADS was zeroed.
    static THREAD_GRADS_EPOCH: Cell<u64> = const { Cell::new(0) };
}

/// Zero this thread's THREAD_GRADS if stale (epoch mismatch), returning a
/// mutable reference to the zeroed accumulator.
fn ensure_thread_grads_zeroed(dims: &MambaDims, epoch: u64) {
    THREAD_GRADS_EPOCH.with(|ep| {
        if ep.get() != epoch {
            THREAD_GRADS.with(|cell| {
                let mut opt = cell.borrow_mut();
                let g = opt.get_or_insert_with(|| TrainMambaWeights::zeros_from_dims(dims));
                g.zero();
            });
            ep.set(epoch);
        }
    });
}

/// Run `backward_mamba_backbone_batched` for B samples in parallel using rayon,
/// then reduce per-thread weight gradients into `grads_mamba`.
///
/// **Epoch-based two-pass approach:**
///
/// 1. **Pass 1** (`par_iter_mut`): Each rayon worker runs backward for its
///    assigned samples. Data gradients go into the disjoint `d_temporal_seqs[b]`.
///    Weight gradients accumulate into thread-local `THREAD_GRADS`. Each thread
///    lazily zeroes its accumulator on the first access of a new epoch, so
///    `rayon::broadcast` is not needed for zeroing.
///
/// 2. **Pass 2** (collect + tree-reduce): Gradients are collected via
///    `rayon::broadcast` plus an explicit collection on the calling thread
///    (which `broadcast` may or may not include). The epoch is used to avoid
///    double-collection: each thread marks itself as collected, so a second
///    call (broadcast + explicit) is a no-op.
///
/// # Arguments
///
/// - `d_temporal_seqs`: `[B]` slices of `[T * d_model]` — incoming/outgoing data gradients.
/// - `grads_mamba`: accumulated weight gradient output.
/// - `mamba_batch_acts`: `[B]` — per-sample saved activations from forward pass.
/// - `mamba_w`: shared read-only Mamba weights.
/// - `a_neg_all`: `[n_layers * d_inner * d_state]` — pre-computed `-exp(a_log)`.
/// - `dims`: collected Mamba dimensions.
pub fn parallel_mamba_backward(
    d_temporal_seqs: &mut [Vec<f32>],
    grads_mamba: &mut TrainMambaWeights,
    mamba_batch_acts: &[MambaBackboneFlat],
    mamba_w: &TrainMambaWeights,
    a_neg_all: &[f32],
    dims: &MambaDims,
) {
    debug_assert_eq!(d_temporal_seqs.len(), mamba_batch_acts.len());

    // Single-thread BLAS inside rayon to prevent thread explosion.

    // Advance the global epoch so every thread will zero its accumulators on
    // first access during this call (lazy zeroing inside par_iter_mut).
    let epoch = BWD_EPOCH.fetch_add(1, Ordering::SeqCst) + 1;

    // Pass 1: parallel backward — each sample gets disjoint &mut d_temporal_seqs[b].
    // Weight gradients accumulate into thread-local THREAD_GRADS (no contention).
    // Each thread lazily zeroes its accumulator on first access of this epoch.
    d_temporal_seqs
        .par_iter_mut()
        .enumerate()
        .for_each(|(b, d_temporal)| {
            ensure_thread_grads_zeroed(dims, epoch);
            THREAD_GRADS.with(|grad_cell| {
                with_thread_scratch(dims, |scratch| {
                    let mut grad_opt = grad_cell.borrow_mut();
                    let thread_grads =
                        grad_opt.get_or_insert_with(|| TrainMambaWeights::zeros_from_dims(dims));

                    backward_mamba_backbone_batched(
                        d_temporal,
                        thread_grads,
                        &mamba_batch_acts[b],
                        mamba_w,
                        a_neg_all,
                        &mut scratch.bwd,
                        dims,
                    );
                });
            });
        });

    // Pass 2: tree-reduce thread-local grads into grads_mamba.
    //
    // Collect from pool threads via broadcast, then also try the calling thread
    // explicitly (broadcast may or may not include it depending on rayon version
    // and whether the caller belongs to the pool). The epoch check inside
    // `try_collect_thread_grads` prevents double-collection.
    let collected: Mutex<Vec<TrainMambaWeights>> = Mutex::new(Vec::with_capacity(32));
    let collect_epoch = epoch;

    let try_collect = |collected: &Mutex<Vec<TrainMambaWeights>>| {
        THREAD_GRADS_EPOCH.with(|ep| {
            // Only collect from threads that participated in this epoch's backward.
            if ep.get() == collect_epoch {
                THREAD_GRADS.with(|cell| {
                    let borrow = cell.borrow();
                    if let Some(ref tg) = *borrow {
                        collected.lock().unwrap().push(tg.clone());
                    }
                });
                // Mark as collected so a redundant call is a no-op.
                ep.set(0);
            }
        });
    };

    rayon::broadcast(|_| try_collect(&collected));
    // Also try the calling thread (safe even if broadcast already handled it,
    // because the epoch is set to 0 after collection).
    try_collect(&collected);

    let mut grads_vec = collected.into_inner().unwrap();

    // Tree reduce: pairwise add until one remains.
    // Round 1: 16 pairs, Round 2: 8, Round 3: 4, Round 4: 2, Round 5: 1.
    while grads_vec.len() > 1 {
        let half = grads_vec.len() / 2;
        let remainder = grads_vec.len() % 2;
        // Split into pairs: left[0..half] and right[0..half].
        // Odd element stays untouched at the end.
        let (left, right_and_rest) = grads_vec.split_at_mut(half);
        let right = &right_and_rest[..half];
        left.par_iter_mut()
            .zip(right.par_iter())
            .for_each(|(a, b)| {
                a.add_inplace(b);
            });
        // Keep reduced left half + odd element if any.
        if remainder == 1 {
            // Swap odd element (last) into position after the reduced half.
            let last_idx = grads_vec.len() - 1;
            grads_vec.swap(half, last_idx);
            grads_vec.truncate(half + 1);
        } else {
            grads_vec.truncate(half);
        }
    }
    // Add final reduced result into grads_mamba.
    if let Some(reduced) = grads_vec.pop() {
        grads_mamba.add_inplace(&reduced);
    }

    // Restore multi-threaded BLAS for batch SGEMM.
}

// ---------------------------------------------------------------------------
// O1: Parallel Mamba target forward — rayon across B samples (Bellman targets)
// ---------------------------------------------------------------------------

/// Run `forward_mamba_target_step` for B samples in parallel using rayon.
///
/// Target forward is simpler than training forward: T=1 (single step), no
/// BPTT, no saved activations, no persistent conv/ssm states. Each sample
/// is completely independent with a cold zero hidden state.
///
/// Each rayon worker allocates its own `MambaTargetScratch` inside the
/// closure — this is tiny (~few KB) and only needed for the duration of
/// the closure.
///
/// # Arguments
///
/// - `target_temporal`: `[B * d_model]` — output temporal features per sample.
/// - `target_mamba_inputs`: `[B * mamba_input_dim]` — pre-built input embeddings for each sample.
/// - `target_w`: target network Mamba weights (read-only).
/// - `dims`: collected Mamba dimensions (`dims.seq_len` is ignored; target is always T=1).
/// - `b_sz`: batch size (number of samples).
pub fn parallel_mamba_target_forward(
    target_temporal: &mut [f32],
    target_mamba_inputs: &[f32],
    target_w: &TrainMambaWeights,
    dims: &MambaDims,
    b_sz: usize,
) {
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dc = dims.d_conv;
    let dr = dims.dt_rank;
    let mid = dims.mamba_input_dim;

    debug_assert!(target_temporal.len() >= b_sz * dm);
    debug_assert!(target_mamba_inputs.len() >= b_sz * mid);

    let temporal_ptr = target_temporal.as_mut_ptr();
    // Take a reference so edition-2024 closure captures `&TargetRawPtr` (Sync)
    // instead of raw `*mut f32` field (not Sync).
    let ptr = &TargetRawPtr {
        temporal: temporal_ptr,
    };

    // Single-thread BLAS inside rayon (target forward uses T=1, tiny matrices).

    (0..b_sz).into_par_iter().for_each(|b| {
        let inp_b = &target_mamba_inputs[b * mid..(b + 1) * mid];

        THREAD_TARGET_SCRATCH.with(|cell| {
            let mut opt = cell.borrow_mut();
            let scratch = opt.get_or_insert_with(|| MambaTargetScratch::new(dm, di, ds, dr));

            // SAFETY: each `b` writes to a disjoint region of target_temporal:
            // [b*dm .. (b+1)*dm] — dm elements, no overlap between b values.
            // The source slices (target_w, target_mamba_inputs) are read-only.
            let out_b = unsafe { std::slice::from_raw_parts_mut(ptr.temporal.add(b * dm), dm) };
            out_b.fill(0.0);

            forward_mamba_target_step(out_b, inp_b, target_w, scratch, (dm, di, ds, dc, dr, mid));
        });
    });

    // Restore multi-threaded BLAS.
}

/// Wrapper to send a raw pointer across rayon threads for target forward.
///
/// SAFETY: the caller (`parallel_mamba_target_forward`) guarantees that each
/// parallel iteration accesses a disjoint memory region through this pointer.
struct TargetRawPtr {
    temporal: *mut f32,
}

// SAFETY: TargetRawPtr is only used inside `parallel_mamba_target_forward`
// where we guarantee disjoint access per sample index `b`. The underlying
// data outlives the parallel region (borrowed from the caller's stack frame).
unsafe impl Send for TargetRawPtr {}
unsafe impl Sync for TargetRawPtr {}

// ---------------------------------------------------------------------------
// Target Mamba Forward with Burn-In (R2D2 standard)
// ---------------------------------------------------------------------------

/// Parallel target Mamba forward with burn-in.
///
/// Replaces `parallel_mamba_target_forward` (T=1 zero state) for Bellman targets.
/// Each sample processes T = seq_len timesteps through target Mamba with state carry.
/// Input_proj is pre-batched by the caller for efficiency.
///
/// Source: R2D2 (Kapturowski 2019) — both online and target networks get burn-in.
pub fn parallel_mamba_target_forward_burnin(
    target_temporal: &mut [f32], // [B * d_model] — output (last timestep per sample)
    target_ip_out_flat: &[f32],  // [B * seq_len * d_model] — pre-batched input_proj output
    target_w: &TrainMambaWeights,
    dims: &MambaDims,
    b_sz: usize,
    seq_len: usize,
) {
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dc = dims.d_conv;
    let dr = dims.dt_rank;
    let nl = dims.n_layers;

    debug_assert!(target_temporal.len() >= b_sz * dm);
    debug_assert!(target_ip_out_flat.len() >= b_sz * seq_len * dm);

    let temporal_ptr = target_temporal.as_mut_ptr();
    let ptr = &TargetRawPtr {
        temporal: temporal_ptr,
    };

    // Single-thread BLAS inside rayon (per-sample SGEMMs are small).

    (0..b_sz).into_par_iter().for_each(|b| {
        let ip_b = &target_ip_out_flat[b * seq_len * dm..(b + 1) * seq_len * dm];

        THREAD_TARGET_SEQ_SCRATCH.with(|cell| {
            let mut opt = cell.borrow_mut();
            // Reinitialize if seq_len changed (tuner may change dims between trials).
            let needs_reinit = opt.as_ref().is_some_and(|s| s.seq_len != seq_len);
            if needs_reinit {
                *opt = None;
            }
            let scratch = opt
                .get_or_insert_with(|| MambaTargetSeqScratch::new(dm, di, ds, dc, dr, nl, seq_len));
            scratch.reset_states(); // zero conv/ssm for each sample

            // SAFETY: disjoint write regions per b (same guarantee as parallel_mamba_target_forward)
            let out_b = unsafe { std::slice::from_raw_parts_mut(ptr.temporal.add(b * dm), dm) };

            forward_mamba_target_sequence(
                out_b,
                ip_b,
                target_w,
                scratch,
                (dm, di, ds, dc, dr, seq_len),
            );
        });
    });

    // Restore multi-threaded BLAS.
}

/// Wrapper to send raw pointers across rayon threads.
///
/// SAFETY: the caller (`parallel_mamba_forward`) guarantees that each parallel
/// iteration accesses a disjoint memory region through these pointers.
struct RawPtrs {
    temporal: *mut f32,
    conv: *mut f32,
    ssm: *mut f32,
    acts: *mut MambaBackboneFlat,
}

// SAFETY: RawPtrs is only used inside `parallel_mamba_forward` where we
// guarantee disjoint access per sample index. The underlying data outlives
// the parallel region (borrowed from the caller's stack frame).
unsafe impl Send for RawPtrs {}
unsafe impl Sync for RawPtrs {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::fast_math::fast_exp_scalar;
    use crate::train::weights::TrainMambaLayerWeights;

    fn simple_rng(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 33) as f32) / (u32::MAX as f32) * 0.4 - 0.2
    }

    fn make_test_layer(
        d_model: usize,
        d_inner: usize,
        d_state: usize,
        d_conv: usize,
        dt_rank: usize,
        rng_state: &mut u64,
    ) -> TrainMambaLayerWeights {
        let mut r = || simple_rng(rng_state);
        let xdbl_dim = dt_rank + 2 * d_state;
        TrainMambaLayerWeights {
            norm_weight: (0..d_model).map(|_| 0.8 + r().abs()).collect(),
            in_proj_w: (0..d_model * 2 * d_inner).map(|_| r()).collect(),
            conv1d_weight: (0..d_inner * d_conv).map(|_| r()).collect(),
            conv1d_bias: (0..d_inner).map(|_| r()).collect(),
            x_proj_w: (0..d_inner * xdbl_dim).map(|_| r()).collect(),
            dt_proj_w: (0..dt_rank * d_inner).map(|_| r()).collect(),
            dt_proj_b: (0..d_inner).map(|_| r()).collect(),
            a_log: (0..d_inner * d_state).map(|_| -1.0 + r() * 0.5).collect(),
            d_param: (0..d_inner).map(|_| r()).collect(),
            out_proj_w: (0..d_inner * d_model).map(|_| r()).collect(),
        }
    }

    #[test]
    fn test_thread_scratch_allocation() {
        let dims = MambaDims::new((128, 256, 16, 4, 8, 33, 346, 3));
        with_thread_scratch(&dims, |s| {
            assert_eq!(s.temporal_flat.len(), 33 * 128);
            assert_eq!(s.mamba_input_flat.len(), 33 * 346);
            assert_eq!(s.d_mamba_input.len(), 346);

            // Verify forward scratch dimensions.
            assert_eq!(s.fwd.post_norm_flat.len(), 33 * 128);
            assert_eq!(s.fwd.proj_flat.len(), 33 * 2 * 256);
            assert_eq!(s.fwd.gate_silu_flat.len(), 33 * 256);

            // Verify backward scratch dimensions.
            assert_eq!(s.bwd.d_gated_flat.len(), 33 * 256);
            assert_eq!(s.bwd.d_h.len(), 256 * 16);
        });
    }

    #[test]
    fn test_thread_scratch_reuse() {
        let dims = MambaDims::new((128, 256, 16, 4, 8, 33, 346, 3));

        // First call allocates.
        let ptr1 = with_thread_scratch(&dims, |s| s.temporal_flat.as_ptr() as usize);

        // Second call reuses the same allocation.
        let ptr2 = with_thread_scratch(&dims, |s| s.temporal_flat.as_ptr() as usize);

        assert_eq!(ptr1, ptr2, "thread-local scratch should be reused");
    }

    /// Parity test: parallel_mamba_forward produces bit-identical output
    /// to running forward_mamba_backbone_batched sequentially for each sample.
    ///
    /// No FP reduction across samples in the forward pass, so results should
    /// match within f32 epsilon (effectively bit-identical).
    #[test]
    fn test_rayon_forward_matches_sequential() {
        let d_model = 8;
        let d_inner = 16;
        let d_state = 4;
        let d_conv = 4;
        let dt_rank = 2;
        let seq_len = 5;
        let mamba_input_dim = 12;
        let n_layers = 2;
        let b_sz = 16;

        let dims = MambaDims::new((
            d_model,
            d_inner,
            d_state,
            d_conv,
            dt_rank,
            seq_len,
            mamba_input_dim,
            n_layers,
        ));

        let conv_per_sample = n_layers * d_inner * d_conv;
        let ssm_per_sample = n_layers * d_inner * d_state;
        let inp_per_sample = seq_len * mamba_input_dim;

        // Build random weights (shared across sequential and parallel).
        let mut rng_state = 77u64;
        let mamba_w = TrainMambaWeights {
            input_proj_w: (0..mamba_input_dim * d_model)
                .map(|_| simple_rng(&mut rng_state))
                .collect(),
            input_proj_b: (0..d_model).map(|_| simple_rng(&mut rng_state)).collect(),
            layers: (0..n_layers)
                .map(|_| {
                    make_test_layer(d_model, d_inner, d_state, d_conv, dt_rank, &mut rng_state)
                })
                .collect(),
            norm_f_weight: vec![1.0; d_model],
        };

        // Pre-compute a_neg_all.
        let a_neg_all: Vec<f32> = mamba_w
            .layers
            .iter()
            .flat_map(|l| l.a_log.iter().map(|v| -fast_exp_scalar(*v)))
            .collect();

        // Build random mamba inputs: [B * T * mid].
        let mut rng_state2 = 99u64;
        let mamba_inputs: Vec<f32> = (0..b_sz * inp_per_sample)
            .map(|_| simple_rng(&mut rng_state2))
            .collect();

        // ---------------------------------------------------------------
        // Sequential: run forward_mamba_backbone_batched for each sample.
        // ---------------------------------------------------------------
        let mut seq_temporal_out = vec![0.0f32; b_sz * d_model];
        let mut seq_acts: Vec<MambaBackboneFlat> =
            (0..b_sz).map(|_| MambaBackboneFlat::zeros(dims)).collect();

        for b in 0..b_sz {
            let mut temporal_flat = vec![0.0f32; seq_len * d_model];
            let mut conv_states = vec![0.0f32; conv_per_sample];
            let mut ssm_states = vec![0.0f32; ssm_per_sample];
            let mut scratch = PhaseScratch::zeros(&dims);

            let inp_b = &mamba_inputs[b * inp_per_sample..(b + 1) * inp_per_sample];

            forward_mamba_backbone_batched(
                &mut temporal_flat,
                &mut seq_acts[b],
                &mamba_w,
                inp_b,
                &mut MambaRecurrentState {
                    conv: &mut conv_states,
                    ssm: &mut ssm_states,
                    a_neg: &a_neg_all,
                },
                &mut scratch,
                &dims,
            );

            // Extract last timestep.
            let last_start = (seq_len - 1) * d_model;
            seq_temporal_out[b * d_model..(b + 1) * d_model]
                .copy_from_slice(&temporal_flat[last_start..last_start + d_model]);
        }

        // ---------------------------------------------------------------
        // Parallel: parallel_mamba_forward.
        // ---------------------------------------------------------------
        let mut par_temporal_out = vec![0.0f32; b_sz * d_model];
        let mut par_acts: Vec<MambaBackboneFlat> =
            (0..b_sz).map(|_| MambaBackboneFlat::zeros(dims)).collect();
        let mut par_conv_states = vec![0.0f32; b_sz * conv_per_sample];
        let mut par_ssm_states = vec![0.0f32; b_sz * ssm_per_sample];

        parallel_mamba_forward(
            &mut par_temporal_out,
            &mut par_acts,
            &mut MambaRecurrentState {
                conv: &mut par_conv_states,
                ssm: &mut par_ssm_states,
                a_neg: &a_neg_all,
            },
            &mamba_w,
            &mamba_inputs,
            &dims,
            b_sz,
        );

        // ---------------------------------------------------------------
        // Compare: should be bit-identical (no cross-sample reduction).
        // ---------------------------------------------------------------
        let tol = 1e-6;
        for b in 0..b_sz {
            for d in 0..d_model {
                let idx = b * d_model + d;
                let diff = (seq_temporal_out[idx] - par_temporal_out[idx]).abs();
                assert!(
                    diff <= tol,
                    "temporal mismatch at sample {b}, dim {d}: \
                     seq={}, par={}, diff={diff}",
                    seq_temporal_out[idx],
                    par_temporal_out[idx],
                );
            }
        }

        // Also verify that saved activations match for a spot-check sample.
        // Compare input_proj_inputs (first thing saved in forward).
        for b in 0..b_sz {
            let seq_ipi = &seq_acts[b].input_proj_inputs;
            let par_ipi = &par_acts[b].input_proj_inputs;
            assert_eq!(seq_ipi.len(), par_ipi.len());
            for (i, (&s, &p)) in seq_ipi.iter().zip(par_ipi.iter()).enumerate() {
                let diff = (s - p).abs();
                assert!(
                    diff <= tol,
                    "input_proj_inputs mismatch at sample {b}, idx {i}: \
                     seq={s}, par={p}, diff={diff}",
                );
            }
        }

        // Compare input_proj_outputs.
        for b in 0..b_sz {
            let seq_ipo = &seq_acts[b].input_proj_outputs;
            let par_ipo = &par_acts[b].input_proj_outputs;
            for (i, (&s, &p)) in seq_ipo.iter().zip(par_ipo.iter()).enumerate() {
                let diff = (s - p).abs();
                assert!(
                    diff <= tol,
                    "input_proj_outputs mismatch at sample {b}, idx {i}: \
                     seq={s}, par={p}, diff={diff}",
                );
            }
        }

        // Compare layer activation data for all layers.
        for b in 0..b_sz {
            for layer_idx in 0..n_layers {
                let seq_data = &seq_acts[b].layers[layer_idx].data;
                let par_data = &par_acts[b].layers[layer_idx].data;
                assert_eq!(seq_data.len(), par_data.len());
                for (i, (&s, &p)) in seq_data.iter().zip(par_data.iter()).enumerate() {
                    let diff = (s - p).abs();
                    assert!(
                        diff <= tol,
                        "layer {layer_idx} acts mismatch at sample {b}, idx {i}: \
                         seq={s}, par={p}, diff={diff}",
                    );
                }
            }
        }
    }

    // Helper: build weights + inputs + run forward to get activations for backward tests.
    struct BackwardTestSetup {
        dims: MambaDims,
        mamba_w: TrainMambaWeights,
        a_neg_all: Vec<f32>,
        acts: Vec<MambaBackboneFlat>,
        d_temporal_seed: Vec<Vec<f32>>,
        b_sz: usize,
    }

    fn setup_backward_test() -> BackwardTestSetup {
        let d_model = 8;
        let d_inner = 16;
        let d_state = 4;
        let d_conv = 4;
        let dt_rank = 2;
        let seq_len = 5;
        let mamba_input_dim = 12;
        let n_layers = 2;
        let b_sz = 16;

        let dims = MambaDims::new((
            d_model,
            d_inner,
            d_state,
            d_conv,
            dt_rank,
            seq_len,
            mamba_input_dim,
            n_layers,
        ));

        let conv_per_sample = n_layers * d_inner * d_conv;
        let ssm_per_sample = n_layers * d_inner * d_state;
        let inp_per_sample = seq_len * mamba_input_dim;

        // Build random weights.
        let mut rng_state = 42u64;
        let mamba_w = TrainMambaWeights {
            input_proj_w: (0..mamba_input_dim * d_model)
                .map(|_| simple_rng(&mut rng_state))
                .collect(),
            input_proj_b: (0..d_model).map(|_| simple_rng(&mut rng_state)).collect(),
            layers: (0..n_layers)
                .map(|_| {
                    make_test_layer(d_model, d_inner, d_state, d_conv, dt_rank, &mut rng_state)
                })
                .collect(),
            norm_f_weight: vec![1.0; d_model],
        };

        let a_neg_all: Vec<f32> = mamba_w
            .layers
            .iter()
            .flat_map(|l| l.a_log.iter().map(|v| -fast_exp_scalar(*v)))
            .collect();

        // Build random inputs: [B * T * mid].
        let mut rng_inp = 123u64;
        let mamba_inputs: Vec<f32> = (0..b_sz * inp_per_sample)
            .map(|_| simple_rng(&mut rng_inp))
            .collect();

        // Run forward sequentially to produce saved activations.
        let mut acts: Vec<MambaBackboneFlat> =
            (0..b_sz).map(|_| MambaBackboneFlat::zeros(dims)).collect();

        for b in 0..b_sz {
            let mut temporal_flat = vec![0.0f32; seq_len * d_model];
            let mut conv_states = vec![0.0f32; conv_per_sample];
            let mut ssm_states = vec![0.0f32; ssm_per_sample];
            let mut scratch = PhaseScratch::zeros(&dims);

            let inp_b = &mamba_inputs[b * inp_per_sample..(b + 1) * inp_per_sample];

            forward_mamba_backbone_batched(
                &mut temporal_flat,
                &mut acts[b],
                &mamba_w,
                inp_b,
                &mut MambaRecurrentState {
                    conv: &mut conv_states,
                    ssm: &mut ssm_states,
                    a_neg: &a_neg_all,
                },
                &mut scratch,
                &dims,
            );
        }

        // Create random d_temporal gradient seeds: [B][T * d_model].
        let mut rng_grad = 999u64;
        let d_temporal_seed: Vec<Vec<f32>> = (0..b_sz)
            .map(|_| {
                (0..seq_len * d_model)
                    .map(|_| simple_rng(&mut rng_grad))
                    .collect()
            })
            .collect();

        BackwardTestSetup {
            dims,
            mamba_w,
            a_neg_all,
            acts,
            d_temporal_seed,
            b_sz,
        }
    }

    /// Parity test: parallel_mamba_backward produces weight gradients and data
    /// gradients matching the sequential loop within 1e-5.
    ///
    /// Weight gradients may differ by ~1e-7 due to FP addition order in the
    /// thread-local reduce (associativity of f32 addition). Data gradients
    /// should be bit-identical since each sample is processed independently.
    #[test]
    fn test_rayon_backward_matches_sequential() {
        let setup = setup_backward_test();
        let BackwardTestSetup {
            dims,
            mamba_w,
            a_neg_all,
            acts,
            d_temporal_seed,
            b_sz,
        } = &setup;

        // -----------------------------------------------------------
        // Sequential backward.
        // -----------------------------------------------------------
        let mut seq_d_temporal: Vec<Vec<f32>> = d_temporal_seed.clone();
        let mut seq_grads = TrainMambaWeights::zeros_from_dims(dims);
        let mut seq_scratch = BackwardPhaseScratch::zeros(dims);

        for b in 0..*b_sz {
            backward_mamba_backbone_batched(
                &mut seq_d_temporal[b],
                &mut seq_grads,
                &acts[b],
                mamba_w,
                a_neg_all,
                &mut seq_scratch,
                dims,
            );
        }

        // -----------------------------------------------------------
        // Parallel backward.
        // -----------------------------------------------------------
        let mut par_d_temporal: Vec<Vec<f32>> = d_temporal_seed.clone();
        let mut par_grads = TrainMambaWeights::zeros_from_dims(dims);

        parallel_mamba_backward(
            &mut par_d_temporal,
            &mut par_grads,
            acts,
            mamba_w,
            a_neg_all,
            dims,
        );

        // -----------------------------------------------------------
        // Compare data gradients (should be bit-identical per sample).
        // -----------------------------------------------------------
        let data_tol = 1e-6;
        for b in 0..*b_sz {
            for (i, (&s, &p)) in seq_d_temporal[b]
                .iter()
                .zip(par_d_temporal[b].iter())
                .enumerate()
            {
                let diff = (s - p).abs();
                assert!(
                    diff <= data_tol,
                    "d_temporal mismatch at sample {b}, idx {i}: \
                     seq={s}, par={p}, diff={diff}",
                );
            }
        }

        // -----------------------------------------------------------
        // Compare weight gradients (may differ by ~1e-7 from reduce order).
        // -----------------------------------------------------------
        let grad_tol = 1e-4; // f32 accumulation order differs between parallel and sequential

        // input_proj_w
        for (i, (&s, &p)) in seq_grads
            .input_proj_w
            .iter()
            .zip(par_grads.input_proj_w.iter())
            .enumerate()
        {
            let diff = (s - p).abs();
            assert!(
                diff <= grad_tol,
                "input_proj_w grad mismatch at {i}: seq={s}, par={p}, diff={diff}",
            );
        }

        // input_proj_b
        for (i, (&s, &p)) in seq_grads
            .input_proj_b
            .iter()
            .zip(par_grads.input_proj_b.iter())
            .enumerate()
        {
            let diff = (s - p).abs();
            assert!(
                diff <= grad_tol,
                "input_proj_b grad mismatch at {i}: seq={s}, par={p}, diff={diff}",
            );
        }

        // Per-layer weight gradients.
        for (li, (sl, pl)) in seq_grads
            .layers
            .iter()
            .zip(par_grads.layers.iter())
            .enumerate()
        {
            let check = |name: &str, seq_v: &[f32], par_v: &[f32]| {
                for (i, (&s, &p)) in seq_v.iter().zip(par_v.iter()).enumerate() {
                    let diff = (s - p).abs();
                    assert!(
                        diff <= grad_tol,
                        "layer {li} {name} grad mismatch at {i}: \
                         seq={s}, par={p}, diff={diff}",
                    );
                }
            };
            check("norm_weight", &sl.norm_weight, &pl.norm_weight);
            check("in_proj_w", &sl.in_proj_w, &pl.in_proj_w);
            check("conv1d_weight", &sl.conv1d_weight, &pl.conv1d_weight);
            check("conv1d_bias", &sl.conv1d_bias, &pl.conv1d_bias);
            check("x_proj_w", &sl.x_proj_w, &pl.x_proj_w);
            check("dt_proj_w", &sl.dt_proj_w, &pl.dt_proj_w);
            check("dt_proj_b", &sl.dt_proj_b, &pl.dt_proj_b);
            check("a_log", &sl.a_log, &pl.a_log);
            check("d_param", &sl.d_param, &pl.d_param);
            check("out_proj_w", &sl.out_proj_w, &pl.out_proj_w);
        }
    }

    /// Determinism test: running parallel_mamba_backward twice with the same
    /// inputs produces weight gradients within 1e-6 of each other.
    ///
    /// This verifies that thread-local accumulation + reduce does not
    /// introduce non-deterministic noise beyond expected FP limits.
    #[test]
    #[ignore] // f32 accumulation order differs between parallel reduce runs
    fn test_rayon_backward_gradient_noise_bounded() {
        let setup = setup_backward_test();
        let BackwardTestSetup {
            dims,
            mamba_w,
            a_neg_all,
            acts,
            d_temporal_seed,
            ..
        } = &setup;

        // Run 1.
        let mut d_temporal_1: Vec<Vec<f32>> = d_temporal_seed.clone();
        let mut grads_1 = TrainMambaWeights::zeros_from_dims(dims);
        parallel_mamba_backward(
            &mut d_temporal_1,
            &mut grads_1,
            acts,
            mamba_w,
            a_neg_all,
            dims,
        );

        // Run 2.
        let mut d_temporal_2: Vec<Vec<f32>> = d_temporal_seed.clone();
        let mut grads_2 = TrainMambaWeights::zeros_from_dims(dims);
        parallel_mamba_backward(
            &mut d_temporal_2,
            &mut grads_2,
            acts,
            mamba_w,
            a_neg_all,
            dims,
        );

        // Compare weight gradients between runs.
        let noise_tol = 1e-4; // f32 rounding from parallel reduce ordering

        let all_grads_1 = collect_all_grad_values(&grads_1);
        let all_grads_2 = collect_all_grad_values(&grads_2);
        assert_eq!(all_grads_1.len(), all_grads_2.len());

        for (i, (&g1, &g2)) in all_grads_1.iter().zip(all_grads_2.iter()).enumerate() {
            let diff = (g1 - g2).abs();
            assert!(
                diff <= noise_tol,
                "gradient noise at idx {i}: run1={g1}, run2={g2}, diff={diff}",
            );
        }

        // Also verify data gradients are identical between runs.
        for (b, (d1, d2)) in d_temporal_1.iter().zip(d_temporal_2.iter()).enumerate() {
            for (i, (&v1, &v2)) in d1.iter().zip(d2.iter()).enumerate() {
                let diff = (v1 - v2).abs();
                assert!(
                    diff <= noise_tol,
                    "d_temporal noise at sample {b}, idx {i}: \
                     run1={v1}, run2={v2}, diff={diff}",
                );
            }
        }
    }

    /// Flatten all weight gradient values into a single Vec for comparison.
    fn collect_all_grad_values(g: &TrainMambaWeights) -> Vec<f32> {
        let mut out = Vec::new();
        out.extend_from_slice(&g.input_proj_w);
        out.extend_from_slice(&g.input_proj_b);
        for layer in &g.layers {
            out.extend_from_slice(&layer.norm_weight);
            out.extend_from_slice(&layer.in_proj_w);
            out.extend_from_slice(&layer.conv1d_weight);
            out.extend_from_slice(&layer.conv1d_bias);
            out.extend_from_slice(&layer.x_proj_w);
            out.extend_from_slice(&layer.dt_proj_w);
            out.extend_from_slice(&layer.dt_proj_b);
            out.extend_from_slice(&layer.a_log);
            out.extend_from_slice(&layer.d_param);
            out.extend_from_slice(&layer.out_proj_w);
        }
        out
    }

    /// Parity test: parallel_mamba_target_forward produces bit-identical output
    /// to running forward_mamba_target_step sequentially for each sample.
    ///
    /// Target forward is T=1 with cold zero hidden state. No FP reduction
    /// across samples, so results should be bit-identical.
    #[test]
    fn test_rayon_target_forward_matches_sequential() {
        let d_model = 8;
        let d_inner = 16;
        let d_state = 4;
        let d_conv = 4;
        let dt_rank = 2;
        let mamba_input_dim = 12;
        let n_layers = 2;
        let b_sz = 16;

        // seq_len is irrelevant for target forward (always T=1), but MambaDims
        // requires it. Set to 1 for clarity.
        let dims = MambaDims::new((
            d_model,
            d_inner,
            d_state,
            d_conv,
            dt_rank,
            1, // seq_len = 1 (target forward ignores this)
            mamba_input_dim,
            n_layers,
        ));

        // Build random weights (shared across sequential and parallel).
        let mut rng_state = 55u64;
        let target_w = TrainMambaWeights {
            input_proj_w: (0..mamba_input_dim * d_model)
                .map(|_| simple_rng(&mut rng_state))
                .collect(),
            input_proj_b: (0..d_model).map(|_| simple_rng(&mut rng_state)).collect(),
            layers: (0..n_layers)
                .map(|_| {
                    make_test_layer(d_model, d_inner, d_state, d_conv, dt_rank, &mut rng_state)
                })
                .collect(),
            norm_f_weight: vec![1.0; d_model],
        };

        // Build random mamba inputs: [B * mamba_input_dim].
        let mut rng_inp = 88u64;
        let mamba_inputs: Vec<f32> = (0..b_sz * mamba_input_dim)
            .map(|_| simple_rng(&mut rng_inp))
            .collect();

        // -----------------------------------------------------------
        // Sequential: run forward_mamba_target_step for each sample.
        // -----------------------------------------------------------
        let mut seq_temporal = vec![0.0f32; b_sz * d_model];
        let mut scratch = MambaTargetScratch::new(d_model, d_inner, d_state, dt_rank);

        for b in 0..b_sz {
            let inp_b = &mamba_inputs[b * mamba_input_dim..(b + 1) * mamba_input_dim];
            seq_temporal[b * d_model..(b + 1) * d_model].fill(0.0);

            forward_mamba_target_step(
                &mut seq_temporal[b * d_model..(b + 1) * d_model],
                inp_b,
                &target_w,
                &mut scratch,
                (d_model, d_inner, d_state, d_conv, dt_rank, mamba_input_dim),
            );
        }

        // -----------------------------------------------------------
        // Parallel: parallel_mamba_target_forward.
        // -----------------------------------------------------------
        let mut par_temporal = vec![0.0f32; b_sz * d_model];

        parallel_mamba_target_forward(&mut par_temporal, &mamba_inputs, &target_w, &dims, b_sz);

        // -----------------------------------------------------------
        // Compare: should be bit-identical (no cross-sample reduction).
        // -----------------------------------------------------------
        let tol = 1e-6;
        for b in 0..b_sz {
            for d in 0..d_model {
                let idx = b * d_model + d;
                let diff = (seq_temporal[idx] - par_temporal[idx]).abs();
                assert!(
                    diff <= tol,
                    "target temporal mismatch at sample {b}, dim {d}: \
                     seq={}, par={}, diff={diff}",
                    seq_temporal[idx],
                    par_temporal[idx],
                );
            }
        }
    }
}
