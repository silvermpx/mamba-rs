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

use cudarc::driver::CudaGraph;

use crate::mamba_ssm::gpu::adamw::{AdamWBiasFactors, GpuAdamW, step_m1_capturable};
use crate::mamba_ssm::gpu::backward_mixed::gpu_backward_mamba_backbone_mixed;
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::forward::GpuRecurrentState;
use crate::mamba_ssm::gpu::forward_mixed::{
    GpuMambaBackboneMixedActs, GpuMambaMixedTrainScratch, gpu_forward_mamba_backbone_mixed,
};
use crate::mamba_ssm::gpu::graph_capture::capture_into_graph;
use crate::mamba_ssm::gpu::weights::GpuMambaGrads;
use crate::mamba_ssm::gpu::weights_mixed_train::GpuMambaTrainMixedWeights;

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
    captured_state_ptr: u64,
    captured_master_input_proj_w_ptr: u64,
    captured_compute_input_proj_w_ptr: u64,
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

        // Snapshot pointers BEFORE capture so we can stash them after the
        // helper consumes the &mut borrows.
        let snap_input = mamba_input.cached_ptr();
        let snap_d_temporal = d_temporal.cached_ptr();
        let snap_grads_flat = grads.flat.cached_ptr();
        let snap_adam_m = adam.m.cached_ptr();
        let snap_adam_v = adam.v.cached_ptr();
        let snap_bias = bias.ptr();
        let snap_state = state.ssm_states.cached_ptr();
        let snap_master = train_w.master.input_proj_w.cached_ptr();
        let snap_compute = train_w.compute.input_proj_w.ptr();

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
            captured_state_ptr: snap_state,
            captured_master_input_proj_w_ptr: snap_master,
            captured_compute_input_proj_w_ptr: snap_compute,
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
        train_w: &GpuMambaTrainMixedWeights,
        adam: &GpuAdamW,
        bias: &AdamWBiasFactors,
        grads: &GpuMambaGrads,
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
            self.captured_state_ptr,
            "training_graph replay: state pointer changed since capture"
        );
        assert_eq!(
            train_w.master.input_proj_w.cached_ptr(),
            self.captured_master_input_proj_w_ptr,
            "training_graph replay: master weight pointer changed since capture"
        );
        assert_eq!(
            train_w.compute.input_proj_w.ptr(),
            self.captured_compute_input_proj_w_ptr,
            "training_graph replay: compute weight pointer changed since capture"
        );
        self.graph
            .launch()
            .map_err(|e| format!("training_graph launch: {e:?}"))
    }
}
