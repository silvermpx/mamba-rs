//! CUDA-Graph-captured training step for Mamba-3 mixed-precision (bf16).
//!
//! Mirrors the M1 [`crate::mamba_ssm::gpu::training_graph`] design. Captures
//! `grads.zero + forward_mixed + backward_mixed + adamw_capturable +
//! sync_master_to_compute` as one CUDA Graph; replay launches the full
//! training step with a single `cuGraphLaunch`.
//!
//! Same constraints as the M1 variant: bf16 only (f16 needs in-graph
//! overflow handling), per (batch, seq_len) shape, all scratch
//! pre-allocated, pointer-stability invariant enforced on replay.

use cudarc::driver::CudaGraph;

use crate::mamba_ssm::gpu::adamw::{AdamWBiasFactors, GpuAdamW, step_m3_capturable};
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::graph_capture::capture_into_graph;
use crate::mamba3_siso::gpu::backward_mixed::gpu_backward_mamba3_backbone_mixed;
use crate::mamba3_siso::gpu::forward_mixed::{
    GpuMamba3BackboneMixedActs, GpuMamba3MixedScratch, gpu_forward_mamba3_backbone_mixed,
};
use crate::mamba3_siso::gpu::kernels::Mamba3Kernels;
use crate::mamba3_siso::gpu::mamba3_gpu::GpuMamba3Dims;
use crate::mamba3_siso::gpu::state::GpuMamba3Scratch;
use crate::mamba3_siso::gpu::weights::GpuMamba3Grads;
use crate::mamba3_siso::gpu::weights_mixed_train::GpuMamba3TrainMixedWeights;

/// CUDA-Graph holder for a single Mamba-3 training step (bf16).
pub struct GpuMamba3TrainingStepGraph {
    pub graph: CudaGraph,
    pub batch: usize,
    pub seq_len: usize,
    pub dtype: WeightDtype,

    captured_input_ptr: u64,
    captured_d_temporal_ptr: u64,
    captured_grads_flat_ptr: u64,
    captured_adam_m_ptr: u64,
    captured_adam_v_ptr: u64,
    captured_bias_factors_ptr: u64,
    // ALL four state buffers are written by forward and read by backward —
    // any of them reallocating between capture and replay silently corrupts.
    captured_ssm_states_ptr: u64,
    captured_k_states_ptr: u64,
    captured_v_states_ptr: u64,
    captured_angle_states_ptr: u64,
    // Weight-stability proxies: BOTH first- and last-allocated master/compute
    // tensors. See M1 training_graph.rs for the rationale.
    captured_master_input_proj_w_ptr: u64,
    captured_master_norm_f_ptr: u64,
    captured_compute_input_proj_w_ptr: u64,
    captured_compute_norm_f_ptr: u64,
    // Lazy-grow guard: forward_mixed's `ensure_half_staging` must not fire
    // during the captured body. Capture pre-sizes via
    // `presize_half_staging_for_train_m3`, then snapshots the resulting
    // pointer here; replay asserts no grow has happened since.
    captured_half_staging_ptr: u64,
}

impl GpuMamba3TrainingStepGraph {
    /// See M1 [`crate::mamba_ssm::gpu::training_graph::GpuMambaTrainingStepGraph::capture`]
    /// for the warmup / caller-ordering contract — identical here.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        ctx: &GpuCtx,
        cfg: &crate::mamba3_siso::config::Mamba3Config,
        m3k: &Mamba3Kernels,
        train_w: &mut GpuMamba3TrainMixedWeights,
        adam: &GpuAdamW,
        bias: &AdamWBiasFactors,
        grads: &mut GpuMamba3Grads,
        acts: &mut GpuMamba3BackboneMixedActs,
        f32_scratch: &mut GpuMamba3Scratch,
        mixed_scratch: &mut GpuMamba3MixedScratch,
        temporal_f32: &mut GpuBuffer,
        mamba_input: &GpuBuffer,
        d_temporal: &mut GpuBuffer,
        ssm_states: &mut GpuBuffer,
        k_states: &mut GpuBuffer,
        v_states: &mut GpuBuffer,
        angle_states: &mut GpuBuffer,
        dims: &GpuMamba3Dims,
    ) -> Result<Self, String> {
        assert!(
            matches!(train_w.dtype, WeightDtype::Bf16),
            "M3 training graph capture supports bf16 only"
        );
        assert_eq!(acts.dtype, WeightDtype::Bf16);

        // Presize half-staging BEFORE capture (see M1 training_graph for the
        // CUDA_ERROR_ILLEGAL_ADDRESS rationale).
        ctx.presize_half_staging_for_train_m3(cfg, dims.batch, dims.seq_len, train_w.dtype)?;

        let snap_input = mamba_input.cached_ptr();
        let snap_d_temporal = d_temporal.cached_ptr();
        let snap_grads_flat = grads.flat.cached_ptr();
        let snap_adam_m = adam.m.cached_ptr();
        let snap_adam_v = adam.v.cached_ptr();
        let snap_bias = bias.ptr();
        let snap_ssm_states = ssm_states.cached_ptr();
        let snap_k_states = k_states.cached_ptr();
        let snap_v_states = v_states.cached_ptr();
        let snap_angle_states = angle_states.cached_ptr();
        let snap_master_input = train_w.master.input_proj_w.cached_ptr();
        let snap_master_norm_f = train_w.master.norm_f_weight.cached_ptr();
        let snap_compute_input = train_w.compute.input_proj_w.ptr();
        let snap_compute_norm_f = train_w.compute.norm_f_weight.ptr();
        let snap_half_staging = ctx.half_staging_ptr();

        let graph = capture_into_graph(&ctx.stream, || {
            grads.zero(&ctx.stream)?;
            gpu_forward_mamba3_backbone_mixed(
                ctx,
                m3k,
                temporal_f32,
                acts,
                train_w,
                mamba_input,
                ssm_states,
                k_states,
                v_states,
                angle_states,
                mixed_scratch,
                dims,
            )?;
            gpu_backward_mamba3_backbone_mixed(
                ctx,
                m3k,
                d_temporal,
                acts,
                train_w,
                grads,
                f32_scratch,
                mixed_scratch,
                dims,
            )?;
            step_m3_capturable(
                ctx,
                &m3k.adamw_step_f32_capturable,
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
            batch: dims.batch,
            seq_len: dims.seq_len,
            dtype: train_w.dtype,
            captured_input_ptr: snap_input,
            captured_d_temporal_ptr: snap_d_temporal,
            captured_grads_flat_ptr: snap_grads_flat,
            captured_adam_m_ptr: snap_adam_m,
            captured_adam_v_ptr: snap_adam_v,
            captured_bias_factors_ptr: snap_bias,
            captured_ssm_states_ptr: snap_ssm_states,
            captured_k_states_ptr: snap_k_states,
            captured_v_states_ptr: snap_v_states,
            captured_angle_states_ptr: snap_angle_states,
            captured_master_input_proj_w_ptr: snap_master_input,
            captured_master_norm_f_ptr: snap_master_norm_f,
            captured_compute_input_proj_w_ptr: snap_compute_input,
            captured_compute_norm_f_ptr: snap_compute_norm_f,
            captured_half_staging_ptr: snap_half_staging,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn replay(
        &self,
        ctx: &GpuCtx,
        train_w: &GpuMamba3TrainMixedWeights,
        adam: &GpuAdamW,
        bias: &AdamWBiasFactors,
        grads: &GpuMamba3Grads,
        mamba_input: &GpuBuffer,
        d_temporal: &GpuBuffer,
        ssm_states: &GpuBuffer,
        k_states: &GpuBuffer,
        v_states: &GpuBuffer,
        angle_states: &GpuBuffer,
    ) -> Result<(), String> {
        assert_eq!(
            mamba_input.cached_ptr(),
            self.captured_input_ptr,
            "M3 training_graph replay: mamba_input pointer changed since capture"
        );
        assert_eq!(
            d_temporal.cached_ptr(),
            self.captured_d_temporal_ptr,
            "M3 training_graph replay: d_temporal pointer changed since capture"
        );
        assert_eq!(
            grads.flat.cached_ptr(),
            self.captured_grads_flat_ptr,
            "M3 training_graph replay: grads.flat pointer changed since capture"
        );
        assert_eq!(
            adam.m.cached_ptr(),
            self.captured_adam_m_ptr,
            "M3 training_graph replay: adam.m pointer changed since capture"
        );
        assert_eq!(
            adam.v.cached_ptr(),
            self.captured_adam_v_ptr,
            "M3 training_graph replay: adam.v pointer changed since capture"
        );
        assert_eq!(
            bias.ptr(),
            self.captured_bias_factors_ptr,
            "M3 training_graph replay: bias_factors pointer changed since capture"
        );
        assert_eq!(
            ssm_states.cached_ptr(),
            self.captured_ssm_states_ptr,
            "M3 training_graph replay: ssm_states pointer changed since capture"
        );
        assert_eq!(
            k_states.cached_ptr(),
            self.captured_k_states_ptr,
            "M3 training_graph replay: k_states pointer changed since capture"
        );
        assert_eq!(
            v_states.cached_ptr(),
            self.captured_v_states_ptr,
            "M3 training_graph replay: v_states pointer changed since capture"
        );
        assert_eq!(
            angle_states.cached_ptr(),
            self.captured_angle_states_ptr,
            "M3 training_graph replay: angle_states pointer changed since capture"
        );
        assert_eq!(
            train_w.master.input_proj_w.cached_ptr(),
            self.captured_master_input_proj_w_ptr,
            "M3 training_graph replay: master input_proj_w pointer changed since capture"
        );
        assert_eq!(
            train_w.master.norm_f_weight.cached_ptr(),
            self.captured_master_norm_f_ptr,
            "M3 training_graph replay: master norm_f_weight pointer changed since capture"
        );
        assert_eq!(
            train_w.compute.input_proj_w.ptr(),
            self.captured_compute_input_proj_w_ptr,
            "M3 training_graph replay: compute input_proj_w pointer changed since capture"
        );
        assert_eq!(
            train_w.compute.norm_f_weight.ptr(),
            self.captured_compute_norm_f_ptr,
            "M3 training_graph replay: compute norm_f_weight pointer changed since capture"
        );
        assert_eq!(
            ctx.half_staging_ptr(),
            self.captured_half_staging_ptr,
            "M3 training_graph replay: half_staging pointer changed since capture \
             (lazy grow during a previous step?)"
        );
        self.graph
            .launch()
            .map_err(|e| format!("M3 training_graph launch: {e:?}"))
    }
}

// ════════════════════════════════════════════════════════════════════════
// f32 M3 training step graph (no master/compute split, no half_staging).
// ════════════════════════════════════════════════════════════════════════

use crate::mamba3_siso::gpu::backward::gpu_backward_mamba3_backbone;
use crate::mamba3_siso::gpu::forward::gpu_forward_mamba3_backbone;
use crate::mamba3_siso::gpu::state::GpuMamba3BackboneActs;
use crate::mamba3_siso::gpu::weights::GpuMamba3Weights;

/// CUDA-Graph holder for a single Mamba-3 f32 training step. Captures
/// `grads.zero + forward + backward + AdamW`. No `sync_master_to_compute`
/// — f32 training has no compute shadow.
pub struct GpuMamba3F32TrainingStepGraph {
    pub graph: CudaGraph,
    pub batch: usize,
    pub seq_len: usize,

    captured_input_ptr: u64,
    captured_d_temporal_ptr: u64,
    captured_grads_flat_ptr: u64,
    captured_adam_m_ptr: u64,
    captured_adam_v_ptr: u64,
    captured_bias_factors_ptr: u64,
    captured_ssm_states_ptr: u64,
    captured_k_states_ptr: u64,
    captured_v_states_ptr: u64,
    captured_angle_states_ptr: u64,
    captured_weights_input_proj_w_ptr: u64,
    captured_weights_norm_f_ptr: u64,
}

impl GpuMamba3F32TrainingStepGraph {
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        ctx: &GpuCtx,
        m3k: &Mamba3Kernels,
        weights: &mut GpuMamba3Weights,
        adam: &GpuAdamW,
        bias: &AdamWBiasFactors,
        grads: &mut GpuMamba3Grads,
        acts: &mut GpuMamba3BackboneActs,
        scratch: &mut GpuMamba3Scratch,
        temporal: &mut GpuBuffer,
        mamba_input: &GpuBuffer,
        d_temporal: &mut GpuBuffer,
        ssm_states: &mut GpuBuffer,
        k_states: &mut GpuBuffer,
        v_states: &mut GpuBuffer,
        angle_states: &mut GpuBuffer,
        dims: &GpuMamba3Dims,
    ) -> Result<Self, String> {
        let snap_input = mamba_input.cached_ptr();
        let snap_d_temporal = d_temporal.cached_ptr();
        let snap_grads_flat = grads.flat.cached_ptr();
        let snap_adam_m = adam.m.cached_ptr();
        let snap_adam_v = adam.v.cached_ptr();
        let snap_bias = bias.ptr();
        let snap_ssm = ssm_states.cached_ptr();
        let snap_k = k_states.cached_ptr();
        let snap_v = v_states.cached_ptr();
        let snap_angle = angle_states.cached_ptr();
        let snap_input_proj = weights.input_proj_w.cached_ptr();
        let snap_norm_f = weights.norm_f_weight.cached_ptr();

        let graph = capture_into_graph(&ctx.stream, || {
            grads.zero(&ctx.stream)?;
            gpu_forward_mamba3_backbone(
                ctx,
                m3k,
                temporal,
                acts,
                weights,
                mamba_input,
                ssm_states,
                k_states,
                v_states,
                angle_states,
                scratch,
                dims,
            )?;
            gpu_backward_mamba3_backbone(
                ctx, m3k, d_temporal, acts, weights, grads, scratch, dims,
            )?;
            step_m3_capturable(
                ctx,
                &m3k.adamw_step_f32_capturable,
                adam,
                bias.ptr(),
                weights,
                grads,
            )?;
            Ok(())
        })?;

        Ok(Self {
            graph,
            batch: dims.batch,
            seq_len: dims.seq_len,
            captured_input_ptr: snap_input,
            captured_d_temporal_ptr: snap_d_temporal,
            captured_grads_flat_ptr: snap_grads_flat,
            captured_adam_m_ptr: snap_adam_m,
            captured_adam_v_ptr: snap_adam_v,
            captured_bias_factors_ptr: snap_bias,
            captured_ssm_states_ptr: snap_ssm,
            captured_k_states_ptr: snap_k,
            captured_v_states_ptr: snap_v,
            captured_angle_states_ptr: snap_angle,
            captured_weights_input_proj_w_ptr: snap_input_proj,
            captured_weights_norm_f_ptr: snap_norm_f,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn replay(
        &self,
        weights: &GpuMamba3Weights,
        adam: &GpuAdamW,
        bias: &AdamWBiasFactors,
        grads: &GpuMamba3Grads,
        mamba_input: &GpuBuffer,
        d_temporal: &GpuBuffer,
        ssm_states: &GpuBuffer,
        k_states: &GpuBuffer,
        v_states: &GpuBuffer,
        angle_states: &GpuBuffer,
    ) -> Result<(), String> {
        macro_rules! check {
            ($live:expr, $snap:expr, $name:literal) => {
                assert_eq!(
                    $live, $snap,
                    concat!(
                        "M3 f32 training_graph replay: ",
                        $name,
                        " pointer changed since capture"
                    )
                );
            };
        }
        check!(
            mamba_input.cached_ptr(),
            self.captured_input_ptr,
            "mamba_input"
        );
        check!(
            d_temporal.cached_ptr(),
            self.captured_d_temporal_ptr,
            "d_temporal"
        );
        check!(
            grads.flat.cached_ptr(),
            self.captured_grads_flat_ptr,
            "grads.flat"
        );
        check!(adam.m.cached_ptr(), self.captured_adam_m_ptr, "adam.m");
        check!(adam.v.cached_ptr(), self.captured_adam_v_ptr, "adam.v");
        check!(bias.ptr(), self.captured_bias_factors_ptr, "bias_factors");
        check!(
            ssm_states.cached_ptr(),
            self.captured_ssm_states_ptr,
            "ssm_states"
        );
        check!(
            k_states.cached_ptr(),
            self.captured_k_states_ptr,
            "k_states"
        );
        check!(
            v_states.cached_ptr(),
            self.captured_v_states_ptr,
            "v_states"
        );
        check!(
            angle_states.cached_ptr(),
            self.captured_angle_states_ptr,
            "angle_states"
        );
        check!(
            weights.input_proj_w.cached_ptr(),
            self.captured_weights_input_proj_w_ptr,
            "input_proj_w"
        );
        check!(
            weights.norm_f_weight.cached_ptr(),
            self.captured_weights_norm_f_ptr,
            "norm_f_weight"
        );
        self.graph
            .launch()
            .map_err(|e| format!("M3 f32 training_graph launch: {e:?}"))
    }
}
