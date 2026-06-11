//! CUDA-Graph-captured training step for Mamba-3 mixed-precision (bf16).
//!
//! Mirrors the M1 [`crate::mamba_ssm::gpu::training_graph`] design. Captures
//! `grads.zero + forward_mixed + backward_mixed + adamw_capturable +
//! sync_master_to_compute` as one CUDA Graph; replay launches the full
//! training step with a single `cuGraphLaunch`.
//!
//! Same constraints as the M1 variant: this module's structs are bf16-only
//! by assert (the trainer's `capture_graph_f16` adds the in-graph
//! loss-scaler handling for f16), per (batch, seq_len) shape, all scratch
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
use crate::mamba3_siso::gpu::state::{GpuMamba3Scratch, GpuMamba3StateBufs, M3Exec};
use crate::mamba3_siso::gpu::weights::GpuMamba3Grads;
use crate::mamba3_siso::gpu::weights_mixed_train::GpuMamba3TrainMixedWeights;

/// Buffers and parameter state borrowed into the captured M3 bf16
/// training-step body (forward + backward + AdamW + master→compute sync).
pub struct Mamba3MixedCapture<'a> {
    pub train_w: &'a mut GpuMamba3TrainMixedWeights,
    pub adam: &'a GpuAdamW,
    pub bias: &'a AdamWBiasFactors,
    pub grads: &'a mut GpuMamba3Grads,
    pub acts: &'a mut GpuMamba3BackboneMixedActs,
    pub f32_scratch: &'a mut GpuMamba3Scratch,
    pub mixed_scratch: &'a mut GpuMamba3MixedScratch,
    /// Written by the captured forward — pointer baked in.
    pub temporal_f32: &'a mut GpuBuffer,
    pub mamba_input: &'a GpuBuffer,
    pub d_temporal: &'a mut GpuBuffer,
    /// ALL four state buffers are written by forward and read by backward.
    pub states: GpuMamba3StateBufs<'a>,
}

/// Live buffers checked against the captured pointers on every M3 bf16
/// graph replay.
#[derive(Clone, Copy)]
pub struct Mamba3MixedReplay<'a> {
    pub train_w: &'a GpuMamba3TrainMixedWeights,
    pub adam: &'a GpuAdamW,
    pub bias: &'a AdamWBiasFactors,
    pub grads: &'a GpuMamba3Grads,
    pub temporal_f32: &'a GpuBuffer,
    pub mamba_input: &'a GpuBuffer,
    pub d_temporal: &'a GpuBuffer,
    pub ssm_states: &'a GpuBuffer,
    pub k_states: &'a GpuBuffer,
    pub v_states: &'a GpuBuffer,
    pub angle_states: &'a GpuBuffer,
}

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
    // `temporal_f32` is written by the captured forward — pointer baked in.
    captured_temporal_ptr: u64,
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
    // Same guard for the batch-invariant typed-GEMM upcast scratch triple
    // (see the M1 mixed graph) — M3 bf16 bi GEMMs route through it too.
    captured_bi_upcast_ptrs: [u64; 3],
}

impl GpuMamba3TrainingStepGraph {
    /// See M1 [`crate::mamba_ssm::gpu::training_graph::GpuMambaTrainingStepGraph::capture`]
    /// for the warmup / caller-ordering contract — identical here.
    pub fn capture(
        exec: &M3Exec<'_>,
        cfg: &crate::mamba3_siso::config::Mamba3Config,
        cap: Mamba3MixedCapture<'_>,
    ) -> Result<Self, String> {
        let M3Exec {
            ctx,
            kernels: m3k,
            dims,
        } = *exec;
        let Mamba3MixedCapture {
            train_w,
            adam,
            bias,
            grads,
            acts,
            f32_scratch,
            mixed_scratch,
            temporal_f32,
            mamba_input,
            d_temporal,
            mut states,
        } = cap;
        assert!(
            matches!(train_w.dtype, WeightDtype::Bf16),
            "M3 training graph capture supports bf16 only"
        );
        assert_eq!(acts.dtype, WeightDtype::Bf16);

        // Presize half-staging BEFORE capture (see M1 training_graph for the
        // CUDA_ERROR_ILLEGAL_ADDRESS rationale).
        ctx.presize_half_staging_for_train_m3(cfg, dims.batch, dims.seq_len, train_w.dtype)?;
        ctx.presize_bi_upcast_scratch_for_train_m3(cfg, dims.batch, dims.seq_len, train_w.dtype)?;

        let snap_input = mamba_input.cached_ptr();
        let snap_d_temporal = d_temporal.cached_ptr();
        let snap_grads_flat = grads.flat.cached_ptr();
        let snap_adam_m = adam.m.cached_ptr();
        let snap_adam_v = adam.v.cached_ptr();
        let snap_bias = bias.ptr();
        let snap_ssm_states = states.ssm.cached_ptr();
        let snap_k_states = states.k.cached_ptr();
        let snap_v_states = states.v.cached_ptr();
        let snap_angle_states = states.angle.cached_ptr();
        let snap_temporal = temporal_f32.cached_ptr();
        let snap_master_input = train_w.master.input_proj_w.cached_ptr();
        let snap_master_norm_f = train_w.master.norm_f_weight.cached_ptr();
        let snap_compute_input = train_w.compute.input_proj_w.ptr();
        let snap_compute_norm_f = train_w.compute.norm_f_weight.ptr();
        let snap_half_staging = ctx.half_staging_ptr();
        let snap_bi_upcast = ctx.bi_upcast_scratch_ptrs();

        let graph = capture_into_graph(&ctx.stream, || {
            grads.zero(&ctx.stream)?;
            gpu_forward_mamba3_backbone_mixed(
                exec,
                temporal_f32,
                acts,
                train_w,
                mamba_input,
                states.reborrow(),
                mixed_scratch,
            )?;
            gpu_backward_mamba3_backbone_mixed(
                exec,
                d_temporal,
                acts,
                train_w,
                grads,
                f32_scratch,
                mixed_scratch,
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
            captured_temporal_ptr: snap_temporal,
            captured_master_input_proj_w_ptr: snap_master_input,
            captured_master_norm_f_ptr: snap_master_norm_f,
            captured_compute_input_proj_w_ptr: snap_compute_input,
            captured_compute_norm_f_ptr: snap_compute_norm_f,
            captured_half_staging_ptr: snap_half_staging,
            captured_bi_upcast_ptrs: snap_bi_upcast,
        })
    }

    pub fn replay(&self, ctx: &GpuCtx, rp: &Mamba3MixedReplay<'_>) -> Result<(), String> {
        let Mamba3MixedReplay {
            train_w,
            adam,
            bias,
            grads,
            temporal_f32,
            mamba_input,
            d_temporal,
            ssm_states,
            k_states,
            v_states,
            angle_states,
        } = *rp;
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
            temporal_f32.cached_ptr(),
            self.captured_temporal_ptr,
            "M3 training_graph replay: temporal_f32 pointer changed since capture"
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
        assert_eq!(
            ctx.bi_upcast_scratch_ptrs(),
            self.captured_bi_upcast_ptrs,
            "M3 training_graph replay: bi_upcast_scratch pointer changed since \
             capture (a larger typed bi GEMM regrew the scratch after this \
             graph was captured — re-capture or presize for the larger shape)"
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

/// Buffers and parameter state borrowed into the captured M3 f32
/// training-step body (`grads.zero + forward + backward + AdamW`).
pub struct Mamba3F32Capture<'a> {
    pub weights: &'a mut GpuMamba3Weights,
    pub adam: &'a GpuAdamW,
    pub bias: &'a AdamWBiasFactors,
    pub grads: &'a mut GpuMamba3Grads,
    pub acts: &'a mut GpuMamba3BackboneActs,
    pub scratch: &'a mut GpuMamba3Scratch,
    /// Written by the captured forward — pointer baked in.
    pub temporal: &'a mut GpuBuffer,
    pub mamba_input: &'a GpuBuffer,
    pub d_temporal: &'a mut GpuBuffer,
    pub states: GpuMamba3StateBufs<'a>,
}

/// Live buffers checked against the captured pointers on every M3 f32
/// graph replay.
#[derive(Clone, Copy)]
pub struct Mamba3F32Replay<'a> {
    pub weights: &'a GpuMamba3Weights,
    pub adam: &'a GpuAdamW,
    pub bias: &'a AdamWBiasFactors,
    pub grads: &'a GpuMamba3Grads,
    pub temporal: &'a GpuBuffer,
    pub mamba_input: &'a GpuBuffer,
    pub d_temporal: &'a GpuBuffer,
    pub ssm_states: &'a GpuBuffer,
    pub k_states: &'a GpuBuffer,
    pub v_states: &'a GpuBuffer,
    pub angle_states: &'a GpuBuffer,
}

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
    // `temporal` is written by the captured forward — pointer baked in.
    captured_temporal_ptr: u64,
    captured_weights_input_proj_w_ptr: u64,
    captured_weights_norm_f_ptr: u64,
}

impl GpuMamba3F32TrainingStepGraph {
    pub fn capture(exec: &M3Exec<'_>, cap: Mamba3F32Capture<'_>) -> Result<Self, String> {
        let M3Exec {
            ctx,
            kernels: m3k,
            dims,
        } = *exec;
        let Mamba3F32Capture {
            weights,
            adam,
            bias,
            grads,
            acts,
            scratch,
            temporal,
            mamba_input,
            d_temporal,
            mut states,
        } = cap;
        let snap_input = mamba_input.cached_ptr();
        let snap_d_temporal = d_temporal.cached_ptr();
        let snap_grads_flat = grads.flat.cached_ptr();
        let snap_adam_m = adam.m.cached_ptr();
        let snap_adam_v = adam.v.cached_ptr();
        let snap_bias = bias.ptr();
        let snap_ssm = states.ssm.cached_ptr();
        let snap_k = states.k.cached_ptr();
        let snap_v = states.v.cached_ptr();
        let snap_angle = states.angle.cached_ptr();
        let snap_temporal = temporal.cached_ptr();
        let snap_input_proj = weights.input_proj_w.cached_ptr();
        let snap_norm_f = weights.norm_f_weight.cached_ptr();

        let graph = capture_into_graph(&ctx.stream, || {
            grads.zero(&ctx.stream)?;
            gpu_forward_mamba3_backbone(
                exec,
                temporal,
                acts,
                weights,
                mamba_input,
                states.reborrow(),
                scratch,
            )?;
            gpu_backward_mamba3_backbone(exec, d_temporal, acts, weights, grads, scratch)?;
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
            captured_temporal_ptr: snap_temporal,
            captured_weights_input_proj_w_ptr: snap_input_proj,
            captured_weights_norm_f_ptr: snap_norm_f,
        })
    }

    pub fn replay(&self, rp: &Mamba3F32Replay<'_>) -> Result<(), String> {
        let Mamba3F32Replay {
            weights,
            adam,
            bias,
            grads,
            temporal,
            mamba_input,
            d_temporal,
            ssm_states,
            k_states,
            v_states,
            angle_states,
        } = *rp;
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
            temporal.cached_ptr(),
            self.captured_temporal_ptr,
            "temporal"
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
