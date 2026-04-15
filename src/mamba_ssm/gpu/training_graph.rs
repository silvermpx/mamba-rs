//! CUDA-Graph-captured training step for Mamba-1 mixed-precision (bf16).
//!
//! Captures **forward + backward + AdamW + master→compute sync** as a
//! single CUDA Graph. On replay, the entire training step launches with
//! a single `cuGraphLaunch` instead of ~150 individual kernel launches —
//! cuts launch overhead from ~50–100 µs per step to ~5 µs.
//!
//! ## Scope
//! - **bf16 only**: the f16 path needs the inf/nan overflow check after
//!   backward, which forces a CPU readback (`OverflowFlag::read`) and
//!   thus breaks graph capture. f16 needs a separate "polling kernel"
//!   refactor — out of scope for Step 14.
//! - **Per (batch, seq_len) shape**: a captured graph bakes in tensor
//!   sizes via the kernel launch configs. A new shape requires a new
//!   capture; the [`GpuMambaTrainingStepGraph`] holder records its dims
//!   and can be invalidated explicitly.
//!
//! ## What's inside the graph
//!   1. `grads.flat.zero()` — async memset on stream
//!   2. `gpu_forward_mamba_backbone_mixed`
//!   3. `gpu_backward_mamba_backbone_mixed`
//!   4. `step_m1_capturable` (per-tensor AdamW, bias factors via device buf)
//!   5. `sync_master_to_compute` (per-tensor cast f32 → bf16 / D2D copy)
//!
//! ## What stays outside the graph
//!   - H2D upload of `mamba_input` (per-step input data)
//!   - H2D upload of `d_temporal` (per-step loss gradient)
//!   - H2D upload of `(bc1, bc2)` into [`AdamWBiasFactors`] — caller
//!     calls `bias.write(stream, bc1, bc2)` BEFORE each replay, after
//!     `adam.advance()` to bump the step counter.
//!   - State management (zeroing or carrying recurrent state)
//!
//! ## Pointer-stability invariant
//! Captured pointers are stored at capture time. On replay, the holder
//! asserts that every input buffer's `cached_ptr()` still matches what
//! was captured. Any reallocation panics — matches the audit's "the
//! 130m race lesson" rule that mutating buffers between capture and
//! replay silently corrupts outputs.
//!
//! ## Reference
//! Same pattern as `GpuInferenceEngine::capture_graph` (inference.rs:307),
//! extended to backward + optimizer. Mirrors PyTorch's
//! `torch.cuda.CUDAGraph` + `capturable=True` optimizer mode (PyTorch
//! 2.5, `torch/cuda/graphs.py` and `_multi_tensor_adamw`).

use cudarc::driver::{CudaGraph, PushKernelArg};

use crate::mamba_ssm::gpu::adamw::{AdamWBiasFactors, GpuAdamW, step_m1_capturable};
use crate::mamba_ssm::gpu::backward::gpu_backward_mamba_backbone;
use crate::mamba_ssm::gpu::backward_mixed::gpu_backward_mamba_backbone_mixed;
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::forward::{
    GpuMambaBackboneActs, GpuMambaScratch, GpuRecurrentState, gpu_forward_mamba_backbone,
};
use crate::mamba_ssm::gpu::forward_mixed::{
    GpuMambaBackboneMixedActs, GpuMambaMixedTrainScratch, gpu_forward_mamba_backbone_mixed,
};
use crate::mamba_ssm::gpu::graph_capture::capture_into_graph;
use crate::mamba_ssm::gpu::launch::grid_1d;
use crate::mamba_ssm::gpu::weights::{GpuMambaGrads, GpuMambaTrainLayerWeights, GpuMambaTrainWeights};
use crate::mamba_ssm::gpu::weights_mixed_train::GpuMambaTrainMixedWeights;

/// Capture-side variant of the trainer's `recompute_a_neg_all` helper.
/// Launches `exp_negate` per layer from `master_layers[l].a_log` into
/// BOTH `a_neg_all[l*per_layer..]` (backward) and
/// `state_a_neg_all[l*per_layer..]` (forward). These launches must be
/// inside the captured body so replay picks up the freshly-updated
/// `a_log` from AdamW. Without them, every replay runs the SSM with the
/// A-matrix from graph-capture time, effectively freezing the decay.
fn recompute_a_neg_captured(
    ctx: &GpuCtx,
    master_layers: &[GpuMambaTrainLayerWeights],
    a_neg_all: &GpuBuffer,
    state_a_neg_all: &GpuBuffer,
    d_inner: usize,
    d_state: usize,
) -> Result<(), String> {
    let per_layer = d_inner * d_state;
    if per_layer == 0 {
        return Ok(());
    }
    let n_i32 = per_layer as i32;
    for (li, mw) in master_layers.iter().enumerate() {
        let src = mw.a_log.cached_ptr();
        let dst_a = a_neg_all.inner_at(li * per_layer);
        let mut b1 = ctx.stream.launch_builder(&ctx.kernels.exp_negate);
        b1.arg(&dst_a);
        b1.arg(&src);
        b1.arg(&n_i32);
        unsafe { b1.launch(grid_1d(per_layer)) }
            .map_err(|e| format!("exp_negate captured a_neg_all L{li}: {e:?}"))?;
        let dst_s = state_a_neg_all.inner_at(li * per_layer);
        let mut b2 = ctx.stream.launch_builder(&ctx.kernels.exp_negate);
        b2.arg(&dst_s);
        b2.arg(&src);
        b2.arg(&n_i32);
        unsafe { b2.launch(grid_1d(per_layer)) }
            .map_err(|e| format!("exp_negate captured state.a_neg_all L{li}: {e:?}"))?;
    }
    Ok(())
}

/// CUDA-Graph holder for a single Mamba-1 training step (bf16).
pub struct GpuMambaTrainingStepGraph {
    pub graph: CudaGraph,
    pub batch: usize,
    pub seq_len: usize,
    pub dtype: WeightDtype,

    // Captured-pointer assertions. If any of these change between
    // capture and replay, the graph is silently writing to the original
    // address — panic immediately.
    captured_input_ptr: u64,
    captured_d_temporal_ptr: u64,
    captured_grads_flat_ptr: u64,
    captured_adam_m_ptr: u64,
    captured_adam_v_ptr: u64,
    captured_bias_factors_ptr: u64,
    // ALL three GpuRecurrentState fields are accessed by forward (conv_states
    // and a_neg_all are read directly from `state.*`, ssm_states is the
    // recurrent SSM working buffer). Each pointer is baked into the graph;
    // any reallocation between capture and replay corrupts silently.
    captured_state_ssm_states_ptr: u64,
    captured_state_conv_states_ptr: u64,
    captured_state_a_neg_all_ptr: u64,
    // The backward-side `a_neg_all` is a SEPARATE parameter (not the
    // `state.a_neg_all` field above — they happen to share a name but are
    // distinct buffers in the public API).
    captured_a_neg_all_ptr: u64,
    // Weight-stability proxy: snapshot BOTH the first-allocated and the
    // last-allocated master tensor, plus their compute slices. If the
    // weight set is rebuilt (e.g. checkpoint reload), at least one of the
    // four is virtually guaranteed to land at a different address (the
    // allocator can't reuse all original slots simultaneously).
    captured_master_input_proj_w_ptr: u64,
    captured_master_norm_f_ptr: u64,
    captured_compute_input_proj_w_ptr: u64,
    captured_compute_norm_f_ptr: u64,
    // Defensive lazy-grow guard: today the typed mixed forward path uses
    // `gpu_gemm_typed_forward_raw` which doesn't call `ensure_half_staging`,
    // so this guard is currently inert. Kept (with cheap presize) so that
    // any future refactor that routes a captured kernel through
    // `gpu_gemm_forward_dispatch` doesn't silently bake a freed pointer
    // (CUDA_ERROR_ILLEGAL_ADDRESS).
    captured_half_staging_ptr: u64,
}

impl GpuMambaTrainingStepGraph {
    /// Capture the training step into a CUDA Graph.
    ///
    /// Caller owns all the input/output buffers. The graph records the
    /// kernel launches; pointers are baked in. After capture, [`Self::replay`]
    /// re-runs the entire step with a single `cuGraphLaunch`.
    ///
    /// ## Required ordering before this call
    ///   1. Allocate `acts`, `scratch`, `grads`, `adam` (m/v), `bias`,
    ///      `state`, `mamba_input`, `d_temporal`, all weight tensors.
    ///   2. Run a warmup forward+backward+adamw+sync ONCE without graph
    ///      capture — this primes cuBLAS workspace selection and any
    ///      lazy CUDA module loading.
    ///   3. Then call this function. The pre-capture stream sync inside
    ///      ensures all warmup + allocations are complete (the "130m
    ///      race-fix invariant" — see `GpuMambaBackboneMixedActs::new`).
    ///
    /// `bias` must already hold the bias-correction factors for step **1**
    /// (or whatever step number you'll start replaying at). The captured
    /// AdamW reads from `bias`'s device pointer; CPU-side `adam.advance()`
    /// + `bias.write()` between replays drives the per-step values.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        ctx: &GpuCtx,
        cfg: &crate::config::MambaConfig,
        train_w: &mut GpuMambaTrainMixedWeights,
        adam: &GpuAdamW,
        bias: &AdamWBiasFactors,
        grads: &mut GpuMambaGrads,
        acts: &mut GpuMambaBackboneMixedActs,
        scratch: &mut GpuMambaMixedTrainScratch,
        a_neg_all: &GpuBuffer,
        mamba_input: &GpuBuffer,
        d_temporal: &mut GpuBuffer,
        state: &mut GpuRecurrentState,
        batch: usize,
        seq_len: usize,
    ) -> Result<Self, String> {
        assert!(
            matches!(train_w.dtype, WeightDtype::Bf16),
            "Step 14 graph capture supports bf16 only (f16 needs in-graph overflow check)"
        );
        assert_eq!(acts.dtype, WeightDtype::Bf16);

        // CRITICAL: presize the half-precision staging buffer BEFORE capture
        // so `ensure_half_staging` inside the body is a no-op. Otherwise a
        // lazy grow during the captured forward bakes a freed pointer into
        // the graph (CUDA_ERROR_ILLEGAL_ADDRESS on replay).
        ctx.presize_half_staging_for_train(cfg, batch, seq_len, train_w.dtype)?;

        // Snapshot pointers BEFORE capture so we can stash them after the
        // helper consumes the &mut borrows.
        let snap_input = mamba_input.cached_ptr();
        let snap_d_temporal = d_temporal.cached_ptr();
        let snap_grads_flat = grads.flat.cached_ptr();
        let snap_adam_m = adam.m.cached_ptr();
        let snap_adam_v = adam.v.cached_ptr();
        let snap_bias = bias.ptr();
        let snap_state_ssm = state.ssm_states.cached_ptr();
        let snap_state_conv = state.conv_states.cached_ptr();
        let snap_state_a_neg = state.a_neg_all.cached_ptr();
        let snap_a_neg = a_neg_all.cached_ptr();
        let snap_master_input = train_w.master.input_proj_w.cached_ptr();
        let snap_master_norm_f = train_w.master.norm_f_weight.cached_ptr();
        let snap_compute_input = train_w.compute.input_proj_w.ptr();
        let snap_compute_norm_f = train_w.compute.norm_f_weight.ptr();
        let snap_half_staging = ctx.half_staging_ptr();

        let graph = capture_into_graph(&ctx.stream, || {
            grads.zero(&ctx.stream)?;
            gpu_forward_mamba_backbone_mixed(
                ctx,
                acts,
                &train_w.compute,
                mamba_input,
                state,
                scratch,
            )?;
            gpu_backward_mamba_backbone_mixed(
                ctx,
                d_temporal,
                grads,
                acts,
                &train_w.compute,
                a_neg_all,
                scratch,
            )?;
            step_m1_capturable(
                ctx,
                &ctx.kernels.adamw_step_f32_capturable,
                adam,
                bias.ptr(),
                &mut train_w.master,
                grads,
            )?;
            train_w.sync_master_to_compute(ctx)?;
            // Recompute a_neg = -exp(a_log) into BOTH a_neg_all buffers
            // used by forward and backward. Without these launches baked
            // into the captured body the graph replays a stale A-matrix
            // forever — a_log moves per AdamW step but the SSM kernel
            // reads the initial a_neg values until re-capture.
            recompute_a_neg_captured(
                ctx,
                &train_w.master.layers,
                a_neg_all,
                &state.a_neg_all,
                cfg.d_inner(),
                cfg.d_state,
            )?;
            Ok(())
        })?;

        Ok(Self {
            graph,
            batch,
            seq_len,
            dtype: train_w.dtype,
            captured_input_ptr: snap_input,
            captured_d_temporal_ptr: snap_d_temporal,
            captured_grads_flat_ptr: snap_grads_flat,
            captured_adam_m_ptr: snap_adam_m,
            captured_adam_v_ptr: snap_adam_v,
            captured_bias_factors_ptr: snap_bias,
            captured_state_ssm_states_ptr: snap_state_ssm,
            captured_state_conv_states_ptr: snap_state_conv,
            captured_state_a_neg_all_ptr: snap_state_a_neg,
            captured_a_neg_all_ptr: snap_a_neg,
            captured_master_input_proj_w_ptr: snap_master_input,
            captured_master_norm_f_ptr: snap_master_norm_f,
            captured_compute_input_proj_w_ptr: snap_compute_input,
            captured_compute_norm_f_ptr: snap_compute_norm_f,
            captured_half_staging_ptr: snap_half_staging,
        })
    }

    /// Replay the captured graph. Caller must have already:
    ///   1. Uploaded fresh `mamba_input` content
    ///   2. Computed loss + uploaded fresh `d_temporal`
    ///   3. Called `adam.advance()` and `bias.write(stream, bc1, bc2)`
    ///      with the new step number's bias factors
    ///   4. Optionally zeroed `state` (or carried forward; user choice)
    ///
    /// Asserts every captured pointer still matches the live buffer's
    /// `cached_ptr()` — any reallocation since capture is a panic.
    #[allow(clippy::too_many_arguments)]
    pub fn replay(
        &self,
        ctx: &GpuCtx,
        train_w: &GpuMambaTrainMixedWeights,
        adam: &GpuAdamW,
        bias: &AdamWBiasFactors,
        grads: &GpuMambaGrads,
        a_neg_all: &GpuBuffer,
        mamba_input: &GpuBuffer,
        d_temporal: &GpuBuffer,
        state: &GpuRecurrentState,
    ) -> Result<(), String> {
        assert_eq!(
            mamba_input.cached_ptr(),
            self.captured_input_ptr,
            "training_graph replay: mamba_input pointer changed since capture"
        );
        assert_eq!(
            d_temporal.cached_ptr(),
            self.captured_d_temporal_ptr,
            "training_graph replay: d_temporal pointer changed since capture"
        );
        assert_eq!(
            grads.flat.cached_ptr(),
            self.captured_grads_flat_ptr,
            "training_graph replay: grads.flat pointer changed since capture"
        );
        assert_eq!(
            adam.m.cached_ptr(),
            self.captured_adam_m_ptr,
            "training_graph replay: adam.m pointer changed since capture"
        );
        assert_eq!(
            adam.v.cached_ptr(),
            self.captured_adam_v_ptr,
            "training_graph replay: adam.v pointer changed since capture"
        );
        assert_eq!(
            bias.ptr(),
            self.captured_bias_factors_ptr,
            "training_graph replay: bias_factors pointer changed since capture"
        );
        assert_eq!(
            state.ssm_states.cached_ptr(),
            self.captured_state_ssm_states_ptr,
            "training_graph replay: state.ssm_states pointer changed since capture"
        );
        assert_eq!(
            state.conv_states.cached_ptr(),
            self.captured_state_conv_states_ptr,
            "training_graph replay: state.conv_states pointer changed since capture"
        );
        assert_eq!(
            state.a_neg_all.cached_ptr(),
            self.captured_state_a_neg_all_ptr,
            "training_graph replay: state.a_neg_all pointer changed since capture"
        );
        assert_eq!(
            a_neg_all.cached_ptr(),
            self.captured_a_neg_all_ptr,
            "training_graph replay: standalone a_neg_all pointer changed since capture"
        );
        assert_eq!(
            train_w.master.input_proj_w.cached_ptr(),
            self.captured_master_input_proj_w_ptr,
            "training_graph replay: master input_proj_w pointer changed since capture"
        );
        assert_eq!(
            train_w.master.norm_f_weight.cached_ptr(),
            self.captured_master_norm_f_ptr,
            "training_graph replay: master norm_f_weight pointer changed since capture"
        );
        assert_eq!(
            train_w.compute.input_proj_w.ptr(),
            self.captured_compute_input_proj_w_ptr,
            "training_graph replay: compute input_proj_w pointer changed since capture"
        );
        assert_eq!(
            train_w.compute.norm_f_weight.ptr(),
            self.captured_compute_norm_f_ptr,
            "training_graph replay: compute norm_f_weight pointer changed since capture"
        );
        assert_eq!(
            ctx.half_staging_ptr(),
            self.captured_half_staging_ptr,
            "training_graph replay: half_staging pointer changed since capture \
             (lazy grow during a previous step?)"
        );
        self.graph
            .launch()
            .map_err(|e| format!("training_graph launch: {e:?}"))
    }
}

// ════════════════════════════════════════════════════════════════════════
// f32 training step graph (no master/compute split, no half_staging).
// ════════════════════════════════════════════════════════════════════════

/// CUDA-Graph holder for a single Mamba-1 f32 training step. Captures
/// `grads.zero + forward + backward + AdamW`. There's no
/// `sync_master_to_compute` because f32 training has no compute shadow —
/// weights are read directly during the next step's forward.
pub struct GpuMambaF32TrainingStepGraph {
    pub graph: CudaGraph,
    pub batch: usize,
    pub seq_len: usize,

    captured_input_ptr: u64,
    captured_d_temporal_ptr: u64,
    captured_grads_flat_ptr: u64,
    captured_adam_m_ptr: u64,
    captured_adam_v_ptr: u64,
    captured_bias_factors_ptr: u64,
    // ALL three GpuRecurrentState fields are accessed by forward.
    captured_state_ssm_states_ptr: u64,
    captured_state_conv_states_ptr: u64,
    captured_state_a_neg_all_ptr: u64,
    // `temporal` is written by forward; standalone `a_neg_all` (separate
    // from `state.a_neg_all` above) is read by backward.
    captured_temporal_ptr: u64,
    captured_a_neg_all_ptr: u64,
    captured_weights_input_proj_w_ptr: u64,
    captured_weights_norm_f_ptr: u64,
}

impl GpuMambaF32TrainingStepGraph {
    /// Capture the f32 training step. Caller responsible for the same
    /// warmup contract as the mixed variant: run one eager step before
    /// calling this so cuBLAS has selected its kernels and any lazy
    /// resources have settled.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        ctx: &GpuCtx,
        cfg: &crate::config::MambaConfig,
        weights: &mut GpuMambaTrainWeights,
        adam: &GpuAdamW,
        bias: &AdamWBiasFactors,
        grads: &mut GpuMambaGrads,
        acts: &mut GpuMambaBackboneActs,
        scratch: &mut GpuMambaScratch,
        a_neg_all: &GpuBuffer,
        temporal: &mut GpuBuffer,
        mamba_input: &GpuBuffer,
        d_temporal: &mut GpuBuffer,
        state: &mut GpuRecurrentState,
        batch: usize,
        seq_len: usize,
    ) -> Result<Self, String> {
        let snap_input = mamba_input.cached_ptr();
        let snap_d_temporal = d_temporal.cached_ptr();
        let snap_grads_flat = grads.flat.cached_ptr();
        let snap_adam_m = adam.m.cached_ptr();
        let snap_adam_v = adam.v.cached_ptr();
        let snap_bias = bias.ptr();
        let snap_state_ssm = state.ssm_states.cached_ptr();
        let snap_state_conv = state.conv_states.cached_ptr();
        let snap_state_a_neg = state.a_neg_all.cached_ptr();
        let snap_temporal = temporal.cached_ptr();
        let snap_a_neg = a_neg_all.cached_ptr();
        let snap_input_proj = weights.input_proj_w.cached_ptr();
        let snap_norm_f = weights.norm_f_weight.cached_ptr();

        let cfg_local = *cfg;
        let graph = capture_into_graph(&ctx.stream, || {
            grads.zero(&ctx.stream)?;
            gpu_forward_mamba_backbone(ctx, temporal, acts, weights, mamba_input, state, scratch)?;
            gpu_backward_mamba_backbone(ctx, d_temporal, grads, acts, weights, a_neg_all, scratch)?;
            crate::mamba_ssm::gpu::adamw::step_m1_capturable(
                ctx,
                &ctx.kernels.adamw_step_f32_capturable,
                adam,
                bias.ptr(),
                weights,
                grads,
            )?;
            // Recompute a_neg after AdamW — see mixed graph above for
            // rationale. Without this the f32 SSM runs on a stale A-matrix
            // across every replay.
            recompute_a_neg_captured(
                ctx,
                &weights.layers,
                a_neg_all,
                &state.a_neg_all,
                cfg_local.d_inner(),
                cfg_local.d_state,
            )?;
            Ok(())
        })?;

        Ok(Self {
            graph,
            batch,
            seq_len,
            captured_input_ptr: snap_input,
            captured_d_temporal_ptr: snap_d_temporal,
            captured_grads_flat_ptr: snap_grads_flat,
            captured_adam_m_ptr: snap_adam_m,
            captured_adam_v_ptr: snap_adam_v,
            captured_bias_factors_ptr: snap_bias,
            captured_state_ssm_states_ptr: snap_state_ssm,
            captured_state_conv_states_ptr: snap_state_conv,
            captured_state_a_neg_all_ptr: snap_state_a_neg,
            captured_temporal_ptr: snap_temporal,
            captured_a_neg_all_ptr: snap_a_neg,
            captured_weights_input_proj_w_ptr: snap_input_proj,
            captured_weights_norm_f_ptr: snap_norm_f,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn replay(
        &self,
        weights: &GpuMambaTrainWeights,
        adam: &GpuAdamW,
        bias: &AdamWBiasFactors,
        grads: &GpuMambaGrads,
        temporal: &GpuBuffer,
        a_neg_all: &GpuBuffer,
        mamba_input: &GpuBuffer,
        d_temporal: &GpuBuffer,
        state: &GpuRecurrentState,
    ) -> Result<(), String> {
        assert_eq!(
            mamba_input.cached_ptr(),
            self.captured_input_ptr,
            "f32 training_graph replay: mamba_input pointer changed since capture"
        );
        assert_eq!(
            d_temporal.cached_ptr(),
            self.captured_d_temporal_ptr,
            "f32 training_graph replay: d_temporal pointer changed since capture"
        );
        assert_eq!(
            grads.flat.cached_ptr(),
            self.captured_grads_flat_ptr,
            "f32 training_graph replay: grads.flat pointer changed since capture"
        );
        assert_eq!(
            adam.m.cached_ptr(),
            self.captured_adam_m_ptr,
            "f32 training_graph replay: adam.m pointer changed since capture"
        );
        assert_eq!(
            adam.v.cached_ptr(),
            self.captured_adam_v_ptr,
            "f32 training_graph replay: adam.v pointer changed since capture"
        );
        assert_eq!(
            bias.ptr(),
            self.captured_bias_factors_ptr,
            "f32 training_graph replay: bias_factors pointer changed since capture"
        );
        assert_eq!(
            state.ssm_states.cached_ptr(),
            self.captured_state_ssm_states_ptr,
            "f32 training_graph replay: state.ssm_states pointer changed since capture"
        );
        assert_eq!(
            state.conv_states.cached_ptr(),
            self.captured_state_conv_states_ptr,
            "f32 training_graph replay: state.conv_states pointer changed since capture"
        );
        assert_eq!(
            state.a_neg_all.cached_ptr(),
            self.captured_state_a_neg_all_ptr,
            "f32 training_graph replay: state.a_neg_all pointer changed since capture"
        );
        assert_eq!(
            temporal.cached_ptr(),
            self.captured_temporal_ptr,
            "f32 training_graph replay: temporal pointer changed since capture"
        );
        assert_eq!(
            a_neg_all.cached_ptr(),
            self.captured_a_neg_all_ptr,
            "f32 training_graph replay: standalone a_neg_all pointer changed since capture"
        );
        assert_eq!(
            weights.input_proj_w.cached_ptr(),
            self.captured_weights_input_proj_w_ptr,
            "f32 training_graph replay: input_proj_w pointer changed since capture"
        );
        assert_eq!(
            weights.norm_f_weight.cached_ptr(),
            self.captured_weights_norm_f_ptr,
            "f32 training_graph replay: norm_f_weight pointer changed since capture"
        );
        self.graph
            .launch()
            .map_err(|e| format!("f32 training_graph launch: {e:?}"))
    }
}
