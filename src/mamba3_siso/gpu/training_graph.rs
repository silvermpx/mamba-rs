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
    captured_ssm_states_ptr: u64,
    captured_master_norm_f_ptr: u64,
    captured_compute_norm_f_ptr: u64,
}

impl GpuMamba3TrainingStepGraph {
    /// See M1 [`crate::mamba_ssm::gpu::training_graph::GpuMambaTrainingStepGraph::capture`]
    /// for the warmup / caller-ordering contract — identical here.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        ctx: &GpuCtx,
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

        let snap_input = mamba_input.cached_ptr();
        let snap_d_temporal = d_temporal.cached_ptr();
        let snap_grads_flat = grads.flat.cached_ptr();
        let snap_adam_m = adam.m.cached_ptr();
        let snap_adam_v = adam.v.cached_ptr();
        let snap_bias = bias.ptr();
        let snap_ssm_states = ssm_states.cached_ptr();
        let snap_master_norm_f = train_w.master.norm_f_weight.cached_ptr();
        let snap_compute_norm_f = train_w.compute.norm_f_weight.ptr();

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
            captured_master_norm_f_ptr: snap_master_norm_f,
            captured_compute_norm_f_ptr: snap_compute_norm_f,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn replay(
        &self,
        train_w: &GpuMamba3TrainMixedWeights,
        adam: &GpuAdamW,
        bias: &AdamWBiasFactors,
        grads: &GpuMamba3Grads,
        mamba_input: &GpuBuffer,
        d_temporal: &GpuBuffer,
        ssm_states: &GpuBuffer,
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
            train_w.master.norm_f_weight.cached_ptr(),
            self.captured_master_norm_f_ptr,
            "M3 training_graph replay: master weight pointer changed since capture"
        );
        assert_eq!(
            train_w.compute.norm_f_weight.ptr(),
            self.captured_compute_norm_f_ptr,
            "M3 training_graph replay: compute weight pointer changed since capture"
        );
        self.graph
            .launch()
            .map_err(|e| format!("M3 training_graph launch: {e:?}"))
    }
}
