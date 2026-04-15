//! High-level Mamba-3 training API.
//!
//! M3 analogue of [`crate::mamba_ssm::gpu::trainer::MambaTrainer`]. Owns
//! mixed-precision weights + grads + Adam + acts + scratch + state + graph.
//!
//! Currently supports bf16 mixed-precision only. See the M1 trainer for
//! the precision-support rationale.

use crate::mamba_ssm::gpu::adamw::{AdamWBiasFactors, GpuAdamW, step_m3_capturable};
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::device::GpuDevice;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::graph_capture::capture_into_graph;
use crate::mamba_ssm::gpu::loss_scaler::{
    DynamicLossScaler, OverflowFlag, UnscaleFactor, check_inf_nan_gpu, scale_grads_skip_gpu,
};
use crate::mamba_ssm::gpu::trainer::StepMetrics;
use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::gpu::backward::gpu_backward_mamba3_backbone;
use crate::mamba3_siso::gpu::backward_mixed::gpu_backward_mamba3_backbone_mixed;
use crate::mamba3_siso::gpu::forward::gpu_forward_mamba3_backbone;
use crate::mamba3_siso::gpu::forward_mixed::{
    GpuMamba3BackboneMixedActs, GpuMamba3MixedScratch, gpu_forward_mamba3_backbone_mixed,
};
use crate::mamba3_siso::gpu::kernels::Mamba3Kernels;
use crate::mamba3_siso::gpu::state::{GpuMamba3BackboneActs, GpuMamba3Dims, GpuMamba3Scratch};
use crate::mamba3_siso::gpu::training_graph::{
    GpuMamba3F32TrainingStepGraph, GpuMamba3TrainingStepGraph,
};
use crate::mamba3_siso::gpu::weights::{GpuMamba3Grads, GpuMamba3Weights};
use crate::mamba3_siso::gpu::weights_mixed_train::GpuMamba3TrainMixedWeights;
use crate::mamba3_siso::weights::Mamba3Weights;

/// Internal precision-dispatch enum (mirrors `inference::M3BackboneEngine`).
enum Trainer3Inner {
    F32(Box<Mamba3TrainerF32>),
    Mixed(Box<Mamba3TrainerMixed>),
}

/// High-level Mamba-3 training wrapper. Same shape as
/// [`crate::mamba3_siso::gpu::inference::GpuMamba3Backbone`]: one public
/// struct, one method per operation, dtype dispatch happens internally.
pub struct Mamba3Trainer {
    inner: Trainer3Inner,
}

impl Mamba3Trainer {
    pub fn new_with_dtype(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        seq_len: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        Self::new_full(
            gpu_ordinal,
            cpu_weights,
            cfg,
            input_dim,
            batch,
            seq_len,
            dtype,
            1e-3,
            1e-2,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        seq_len: usize,
        dtype: WeightDtype,
        lr: f32,
        weight_decay: f32,
    ) -> Result<Self, String> {
        let inner = match dtype {
            WeightDtype::F32 => Trainer3Inner::F32(Box::new(Mamba3TrainerF32::new_full(
                gpu_ordinal,
                cpu_weights,
                cfg,
                input_dim,
                batch,
                seq_len,
                lr,
                weight_decay,
            )?)),
            WeightDtype::Bf16 | WeightDtype::F16 => {
                Trainer3Inner::Mixed(Box::new(Mamba3TrainerMixed::new_full(
                    gpu_ordinal,
                    cpu_weights,
                    cfg,
                    input_dim,
                    batch,
                    seq_len,
                    dtype,
                    lr,
                    weight_decay,
                )?))
            }
        };
        Ok(Self { inner })
    }

    pub fn dtype(&self) -> WeightDtype {
        match &self.inner {
            Trainer3Inner::F32(_) => WeightDtype::F32,
            Trainer3Inner::Mixed(t) => t.dtype,
        }
    }
    pub fn batch(&self) -> usize {
        match &self.inner {
            Trainer3Inner::F32(t) => t.dims.batch,
            Trainer3Inner::Mixed(t) => t.dims.batch,
        }
    }
    pub fn seq_len(&self) -> usize {
        match &self.inner {
            Trainer3Inner::F32(t) => t.dims.seq_len,
            Trainer3Inner::Mixed(t) => t.dims.seq_len,
        }
    }
    pub fn ctx(&self) -> &GpuCtx {
        match &self.inner {
            Trainer3Inner::F32(t) => &t.ctx,
            Trainer3Inner::Mixed(t) => &t.ctx,
        }
    }
    pub fn has_graph(&self) -> bool {
        match &self.inner {
            Trainer3Inner::F32(t) => t.graph.is_some(),
            Trainer3Inner::Mixed(t) => t.has_graph(),
        }
    }
    pub fn reset_state(&mut self) -> Result<(), String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.reset_state(),
            Trainer3Inner::Mixed(t) => t.reset_state(),
        }
    }
    pub fn capture_graph(&mut self) -> Result<(), String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.capture_graph(),
            Trainer3Inner::Mixed(t) => t.capture_graph(),
        }
    }
    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.step(input, d_temporal),
            Trainer3Inner::Mixed(t) => t.step(input, d_temporal),
        }
    }
    pub fn snapshot_master(&self) -> Result<Mamba3Weights, String> {
        match &self.inner {
            Trainer3Inner::F32(t) => t.snapshot_master(),
            Trainer3Inner::Mixed(t) => t.snapshot_master(),
        }
    }
}

pub struct Mamba3TrainerMixed {
    ctx: GpuCtx,
    m3k: Mamba3Kernels,
    cfg: Mamba3Config,
    dims: GpuMamba3Dims,
    dtype: WeightDtype,

    pub weights: GpuMamba3TrainMixedWeights,
    pub grads: GpuMamba3Grads,
    pub adam: GpuAdamW,
    bias: AdamWBiasFactors,

    acts: GpuMamba3BackboneMixedActs,
    f32_scratch: GpuMamba3Scratch,
    mixed_scratch: GpuMamba3MixedScratch,

    temporal: GpuBuffer,
    mamba_input: GpuBuffer,
    d_temporal: GpuBuffer,
    ssm_states: GpuBuffer,
    k_states: GpuBuffer,
    v_states: GpuBuffer,
    angle_states: GpuBuffer,

    graph: Option<GpuMamba3TrainingStepGraph>,

    // f16 AMP loss scaler (None for bf16). See M1 trainer for the protocol.
    scaler: Option<DynamicLossScaler>,
    overflow_flag: Option<OverflowFlag>,
    d_temporal_scaled: Option<GpuBuffer>,
    /// f16 CUDA Graph (Step 22, M3 analogue of M1's `graph_f16`).
    graph_f16: Option<cudarc::driver::CudaGraph>,
    /// 1-element device buffer of `1/loss_scale` (Step 22).
    unscale_factor: Option<UnscaleFactor>,
}

impl Mamba3TrainerMixed {
    #[allow(clippy::too_many_arguments)]
    fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        seq_len: usize,
        dtype: WeightDtype,
        lr: f32,
        weight_decay: f32,
    ) -> Result<Self, String> {
        assert!(
            matches!(dtype, WeightDtype::Bf16 | WeightDtype::F16),
            "Mamba3TrainerMixed accepts Bf16 or F16; got {dtype:?}"
        );

        let device = GpuDevice::new(gpu_ordinal)?;
        let ctx = GpuCtx::new(&device)?;
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let m3k = Mamba3Kernels::compile(device.context(), arch)?;

        let weights =
            GpuMamba3TrainMixedWeights::from_cpu(&ctx.stream, cpu_weights, &cfg, input_dim, dtype)?;

        let dims = GpuMamba3Dims {
            batch,
            d_model: cfg.d_model,
            d_inner: cfg.d_inner(),
            d_state: cfg.d_state,
            nheads: cfg.nheads(),
            headdim: cfg.headdim,
            ngroups: cfg.ngroups,
            in_proj_dim: cfg.in_proj_out_dim(),
            seq_len,
            mamba_input_dim: input_dim,
            n_layers: cfg.n_layers,
            n_angles: cfg.num_rope_angles(),
            a_floor: cfg.a_floor,
            is_outproj_norm: cfg.is_outproj_norm,
            use_parallel_scan: true,
        };

        let acts =
            GpuMamba3BackboneMixedActs::new(&ctx.stream, &cfg, batch, seq_len, input_dim, dtype)?;
        let f32_scratch = GpuMamba3Scratch::new(&ctx.stream, &dims)?;
        let mixed_scratch = GpuMamba3MixedScratch::new(&ctx.stream, &cfg, batch, seq_len, dtype)?;

        let bt = batch * seq_len;
        let nh = cfg.nheads();
        let hd = cfg.headdim;
        let ds = cfg.d_state;
        let na = cfg.num_rope_angles().max(1);
        let nl = cfg.n_layers;

        let temporal = GpuBuffer::zeros(&ctx.stream, bt * cfg.d_model)?;
        let mamba_input = GpuBuffer::zeros(&ctx.stream, bt * input_dim)?;
        let d_temporal = GpuBuffer::zeros(&ctx.stream, bt * cfg.d_model)?;
        let ssm_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * hd * ds)?;
        let k_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * ds)?;
        let v_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * hd)?;
        let angle_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * na)?;
        let grads = GpuMamba3Grads::new(&ctx.stream, &cfg, input_dim)?;

        let adam = GpuAdamW::new(&ctx.stream, grads.flat.len())?
            .with_lr(lr)
            .with_weight_decay(weight_decay);
        let bias = AdamWBiasFactors::new(&ctx.stream)?;

        let (scaler, overflow_flag, d_temporal_scaled, unscale_factor) =
            if matches!(dtype, WeightDtype::F16) {
                let s = DynamicLossScaler::new();
                let f = OverflowFlag::new(&ctx.stream)?;
                let scaled = GpuBuffer::zeros(&ctx.stream, batch * seq_len * cfg.d_model)?;
                let u = UnscaleFactor::new(&ctx.stream)?;
                (Some(s), Some(f), Some(scaled), Some(u))
            } else {
                (None, None, None, None)
            };

        ctx.stream
            .synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;

        Ok(Self {
            ctx,
            m3k,
            cfg,
            dims,
            dtype,
            weights,
            grads,
            adam,
            bias,
            acts,
            f32_scratch,
            mixed_scratch,
            temporal,
            mamba_input,
            d_temporal,
            ssm_states,
            k_states,
            v_states,
            angle_states,
            graph: None,
            scaler,
            overflow_flag,
            d_temporal_scaled,
            graph_f16: None,
            unscale_factor,
        })
    }

    pub fn dtype(&self) -> WeightDtype {
        self.dtype
    }

    pub fn batch(&self) -> usize {
        self.dims.batch
    }

    pub fn seq_len(&self) -> usize {
        self.dims.seq_len
    }

    pub fn ctx(&self) -> &GpuCtx {
        &self.ctx
    }

    pub fn has_graph(&self) -> bool {
        self.graph.is_some() || self.graph_f16.is_some()
    }

    /// Reset SSM / K / V / angle states to zero (keeps weights untouched).
    pub fn reset_state(&mut self) -> Result<(), String> {
        self.ssm_states.zero(&self.ctx.stream)?;
        self.k_states.zero(&self.ctx.stream)?;
        self.v_states.zero(&self.ctx.stream)?;
        self.angle_states.zero(&self.ctx.stream)?;
        Ok(())
    }

    /// Capture the training-step CUDA Graph. Call once after at least one
    /// warmup [`Self::step`].
    pub fn capture_graph(&mut self) -> Result<(), String> {
        if matches!(self.dtype, WeightDtype::F16) {
            return self.capture_graph_f16();
        }
        self.bias.write(&self.ctx.stream, 1.0, 1.0)?;

        let g = GpuMamba3TrainingStepGraph::capture(
            &self.ctx,
            &self.cfg,
            &self.m3k,
            &mut self.weights,
            &self.adam,
            &self.bias,
            &mut self.grads,
            &mut self.acts,
            &mut self.f32_scratch,
            &mut self.mixed_scratch,
            &mut self.temporal,
            &self.mamba_input,
            &mut self.d_temporal,
            &mut self.ssm_states,
            &mut self.k_states,
            &mut self.v_states,
            &mut self.angle_states,
            &self.dims,
        )?;
        self.graph = Some(g);
        Ok(())
    }

    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        assert_eq!(input.len(), self.mamba_input.len(), "input shape mismatch");
        assert_eq!(
            d_temporal.len(),
            self.d_temporal.len(),
            "d_temporal shape mismatch"
        );

        if matches!(self.dtype, WeightDtype::F16) {
            return self.step_f16(input, d_temporal);
        }

        self.mamba_input.upload(&self.ctx.stream, input)?;
        self.d_temporal.upload(&self.ctx.stream, d_temporal)?;

        let (step, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2)?;

        let replayed = if let Some(ref g) = self.graph {
            g.replay(
                &self.ctx,
                &self.weights,
                &self.adam,
                &self.bias,
                &self.grads,
                &self.temporal,
                &self.mamba_input,
                &self.d_temporal,
                &self.ssm_states,
                &self.k_states,
                &self.v_states,
                &self.angle_states,
            )?;
            true
        } else {
            self.step_eager()?;
            false
        };

        Ok(StepMetrics::plain(step, replayed))
    }

    /// f16 step. See M1 [`crate::mamba_ssm::gpu::trainer::MambaTrainerMixed::step_f16`]
    /// for the full protocol; this is the M3 mirror.
    fn step_f16(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        let scale = self.scaler.as_ref().expect("f16 scaler").scale();
        self.mamba_input.upload(&self.ctx.stream, input)?;
        let scaled: Vec<f32> = d_temporal.iter().map(|v| v * scale).collect();
        let dt_scaled = self.d_temporal_scaled.as_mut().expect("f16 dt_scaled");
        dt_scaled.upload(&self.ctx.stream, &scaled)?;

        if let Some(ref mut u) = self.unscale_factor {
            u.write(&self.ctx.stream, 1.0 / scale)?;
        }
        let prev_step = self.adam.step;
        let (next_step, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2)?;

        let flag = self.overflow_flag.as_mut().expect("f16 overflow flag");
        flag.zero(&self.ctx.stream)?;

        let (step, overflow, replayed) = if let Some(ref g) = self.graph_f16 {
            g.launch()
                .map_err(|e| format!("M3 f16 graph launch: {e:?}"))?;
            let overflow = flag.read(&self.ctx.stream)? != 0;
            self.scaler.as_mut().expect("f16 scaler").update(overflow);
            (next_step, overflow, true)
        } else {
            self.grads.zero(&self.ctx.stream)?;
            gpu_forward_mamba3_backbone_mixed(
                &self.ctx,
                &self.m3k,
                &mut self.temporal,
                &mut self.acts,
                &self.weights,
                &self.mamba_input,
                &mut self.ssm_states,
                &mut self.k_states,
                &mut self.v_states,
                &mut self.angle_states,
                &mut self.mixed_scratch,
                &self.dims,
            )?;
            gpu_backward_mamba3_backbone_mixed(
                &self.ctx,
                &self.m3k,
                dt_scaled,
                &self.acts,
                &self.weights,
                &self.grads,
                &mut self.f32_scratch,
                &mut self.mixed_scratch,
                &self.dims,
            )?;
            check_inf_nan_gpu(&self.ctx, &self.ctx.kernels, flag, &self.grads.flat)?;
            let unscale = self.unscale_factor.as_ref().unwrap();
            scale_grads_skip_gpu(
                &self.ctx,
                &self.ctx.kernels,
                self.overflow_flag.as_mut().unwrap(),
                &mut self.grads.flat,
                unscale,
            )?;
            step_m3_capturable(
                &self.ctx,
                &self.m3k.adamw_step_f32_capturable,
                &self.adam,
                self.bias.ptr(),
                &mut self.weights.master,
                &self.grads,
            )?;
            self.weights.sync_master_to_compute(&self.ctx)?;
            let overflow = self
                .overflow_flag
                .as_ref()
                .unwrap()
                .read(&self.ctx.stream)?
                != 0;
            self.scaler.as_mut().expect("f16 scaler").update(overflow);
            (next_step, overflow, false)
        };

        let final_step = if overflow {
            self.adam.step = prev_step;
            prev_step
        } else {
            step
        };

        Ok(StepMetrics {
            step: final_step,
            graph_replayed: replayed,
            loss_scale: Some(scale),
            overflow_skipped: Some(overflow),
        })
    }

    /// Capture the M3 f16 training step (Step 22 — M3 mirror).
    fn capture_graph_f16(&mut self) -> Result<(), String> {
        self.bias.write(&self.ctx.stream, 1.0, 1.0)?;
        let init_unscale = 1.0 / self.scaler.as_ref().expect("f16 scaler").scale();
        self.unscale_factor
            .as_mut()
            .expect("unscale buf")
            .write(&self.ctx.stream, init_unscale)?;
        self.overflow_flag
            .as_mut()
            .expect("overflow flag")
            .zero(&self.ctx.stream)?;
        let dummy = vec![0.0f32; self.d_temporal.len()];
        self.d_temporal_scaled
            .as_mut()
            .expect("dt_scaled")
            .upload(&self.ctx.stream, &dummy)?;

        self.ctx.presize_half_staging_for_train_m3(
            &self.cfg,
            self.dims.batch,
            self.dims.seq_len,
            self.dtype,
        )?;

        let stream = self.ctx.stream.clone();
        let g = capture_into_graph(&stream, || {
            self.grads.zero(&self.ctx.stream)?;
            gpu_forward_mamba3_backbone_mixed(
                &self.ctx,
                &self.m3k,
                &mut self.temporal,
                &mut self.acts,
                &self.weights,
                &self.mamba_input,
                &mut self.ssm_states,
                &mut self.k_states,
                &mut self.v_states,
                &mut self.angle_states,
                &mut self.mixed_scratch,
                &self.dims,
            )?;
            gpu_backward_mamba3_backbone_mixed(
                &self.ctx,
                &self.m3k,
                self.d_temporal_scaled.as_mut().unwrap(),
                &self.acts,
                &self.weights,
                &self.grads,
                &mut self.f32_scratch,
                &mut self.mixed_scratch,
                &self.dims,
            )?;
            check_inf_nan_gpu(
                &self.ctx,
                &self.ctx.kernels,
                self.overflow_flag.as_mut().unwrap(),
                &self.grads.flat,
            )?;
            scale_grads_skip_gpu(
                &self.ctx,
                &self.ctx.kernels,
                self.overflow_flag.as_mut().unwrap(),
                &mut self.grads.flat,
                self.unscale_factor.as_ref().unwrap(),
            )?;
            step_m3_capturable(
                &self.ctx,
                &self.m3k.adamw_step_f32_capturable,
                &self.adam,
                self.bias.ptr(),
                &mut self.weights.master,
                &self.grads,
            )?;
            self.weights.sync_master_to_compute(&self.ctx)?;
            Ok(())
        })?;
        self.graph_f16 = Some(g);
        Ok(())
    }

    fn step_eager(&mut self) -> Result<(), String> {
        self.grads.zero(&self.ctx.stream)?;
        gpu_forward_mamba3_backbone_mixed(
            &self.ctx,
            &self.m3k,
            &mut self.temporal,
            &mut self.acts,
            &self.weights,
            &self.mamba_input,
            &mut self.ssm_states,
            &mut self.k_states,
            &mut self.v_states,
            &mut self.angle_states,
            &mut self.mixed_scratch,
            &self.dims,
        )?;
        gpu_backward_mamba3_backbone_mixed(
            &self.ctx,
            &self.m3k,
            &mut self.d_temporal,
            &self.acts,
            &self.weights,
            &self.grads,
            &mut self.f32_scratch,
            &mut self.mixed_scratch,
            &self.dims,
        )?;
        step_m3_capturable(
            &self.ctx,
            &self.m3k.adamw_step_f32_capturable,
            &self.adam,
            self.bias.ptr(),
            &mut self.weights.master,
            &self.grads,
        )?;
        self.weights.sync_master_to_compute(&self.ctx)?;
        Ok(())
    }

    /// Download master weights to CPU for checkpointing.
    pub fn snapshot_master(&self) -> Result<Mamba3Weights, String> {
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("pre-snapshot sync: {e:?}"))?;
        let m = &self.weights.master;
        let input_dim = self.mamba_input.len() / (self.dims.batch * self.dims.seq_len);
        let mut out = Mamba3Weights::zeros(&self.cfg, input_dim);
        out.input_proj_w = m.input_proj_w.to_cpu(&self.ctx.stream)?;
        out.input_proj_b = m.input_proj_b.to_cpu(&self.ctx.stream)?;
        for (i, lw) in out.layers.iter_mut().enumerate() {
            let g = &m.layers[i];
            lw.norm_weight = g.norm_weight.to_cpu(&self.ctx.stream)?;
            lw.in_proj_w = g.in_proj_w.to_cpu(&self.ctx.stream)?;
            lw.dt_bias = g.dt_bias.to_cpu(&self.ctx.stream)?;
            lw.b_norm_weight = g.b_norm_weight.to_cpu(&self.ctx.stream)?;
            lw.c_norm_weight = g.c_norm_weight.to_cpu(&self.ctx.stream)?;
            lw.b_bias = g.b_bias.to_cpu(&self.ctx.stream)?;
            lw.c_bias = g.c_bias.to_cpu(&self.ctx.stream)?;
            lw.d_param = g.d_param.to_cpu(&self.ctx.stream)?;
            lw.norm_gate_weight = g.norm_gate_weight.to_cpu(&self.ctx.stream)?;
            lw.out_proj_w = g.out_proj_w.to_cpu(&self.ctx.stream)?;
        }
        out.norm_f_weight = m.norm_f_weight.to_cpu(&self.ctx.stream)?;
        Ok(out)
    }
}

// ════════════════════════════════════════════════════════════════════════
// f32 M3 training wrapper.
// ════════════════════════════════════════════════════════════════════════

pub struct Mamba3TrainerF32 {
    pub ctx: GpuCtx,
    pub m3k: Mamba3Kernels,
    pub cfg: Mamba3Config,
    pub dims: GpuMamba3Dims,
    pub weights: GpuMamba3Weights,
    pub grads: GpuMamba3Grads,
    pub adam: GpuAdamW,
    bias: AdamWBiasFactors,
    acts: GpuMamba3BackboneActs,
    scratch: GpuMamba3Scratch,
    temporal: GpuBuffer,
    mamba_input: GpuBuffer,
    d_temporal: GpuBuffer,
    ssm_states: GpuBuffer,
    k_states: GpuBuffer,
    v_states: GpuBuffer,
    angle_states: GpuBuffer,
    graph: Option<GpuMamba3F32TrainingStepGraph>,
}

impl Mamba3TrainerF32 {
    #[allow(clippy::too_many_arguments)]
    fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        seq_len: usize,
        lr: f32,
        weight_decay: f32,
    ) -> Result<Self, String> {
        let device = GpuDevice::new(gpu_ordinal)?;
        let ctx = GpuCtx::new(&device)?;
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let m3k = Mamba3Kernels::compile(device.context(), arch)?;

        let weights = GpuMamba3Weights::from_cpu(&ctx.stream, cpu_weights, &cfg, input_dim)?;

        let dims = GpuMamba3Dims {
            batch,
            d_model: cfg.d_model,
            d_inner: cfg.d_inner(),
            d_state: cfg.d_state,
            nheads: cfg.nheads(),
            headdim: cfg.headdim,
            ngroups: cfg.ngroups,
            in_proj_dim: cfg.in_proj_out_dim(),
            seq_len,
            mamba_input_dim: input_dim,
            n_layers: cfg.n_layers,
            n_angles: cfg.num_rope_angles(),
            a_floor: cfg.a_floor,
            is_outproj_norm: cfg.is_outproj_norm,
            use_parallel_scan: true,
        };

        let acts = GpuMamba3BackboneActs::new(&ctx.stream, &dims)?;
        let scratch = GpuMamba3Scratch::new(&ctx.stream, &dims)?;

        let bt = batch * seq_len;
        let nh = cfg.nheads();
        let hd = cfg.headdim;
        let ds = cfg.d_state;
        let na = cfg.num_rope_angles().max(1);
        let nl = cfg.n_layers;

        let temporal = GpuBuffer::zeros(&ctx.stream, bt * cfg.d_model)?;
        let mamba_input = GpuBuffer::zeros(&ctx.stream, bt * input_dim)?;
        let d_temporal = GpuBuffer::zeros(&ctx.stream, bt * cfg.d_model)?;
        let ssm_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * hd * ds)?;
        let k_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * ds)?;
        let v_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * hd)?;
        let angle_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * na)?;
        let grads = GpuMamba3Grads::new(&ctx.stream, &cfg, input_dim)?;

        let adam = GpuAdamW::new(&ctx.stream, grads.flat.len())?
            .with_lr(lr)
            .with_weight_decay(weight_decay);
        let bias = AdamWBiasFactors::new(&ctx.stream)?;

        ctx.stream
            .synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;

        Ok(Self {
            ctx,
            m3k,
            cfg,
            dims,
            weights,
            grads,
            adam,
            bias,
            acts,
            scratch,
            temporal,
            mamba_input,
            d_temporal,
            ssm_states,
            k_states,
            v_states,
            angle_states,
            graph: None,
        })
    }

    pub fn reset_state(&mut self) -> Result<(), String> {
        self.ssm_states.zero(&self.ctx.stream)?;
        self.k_states.zero(&self.ctx.stream)?;
        self.v_states.zero(&self.ctx.stream)?;
        self.angle_states.zero(&self.ctx.stream)?;
        Ok(())
    }

    pub fn capture_graph(&mut self) -> Result<(), String> {
        self.bias.write(&self.ctx.stream, 1.0, 1.0)?;
        let g = GpuMamba3F32TrainingStepGraph::capture(
            &self.ctx,
            &self.m3k,
            &mut self.weights,
            &self.adam,
            &self.bias,
            &mut self.grads,
            &mut self.acts,
            &mut self.scratch,
            &mut self.temporal,
            &self.mamba_input,
            &mut self.d_temporal,
            &mut self.ssm_states,
            &mut self.k_states,
            &mut self.v_states,
            &mut self.angle_states,
            &self.dims,
        )?;
        self.graph = Some(g);
        Ok(())
    }

    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        assert_eq!(input.len(), self.mamba_input.len(), "input shape mismatch");
        assert_eq!(
            d_temporal.len(),
            self.d_temporal.len(),
            "d_temporal shape mismatch"
        );
        self.mamba_input.upload(&self.ctx.stream, input)?;
        self.d_temporal.upload(&self.ctx.stream, d_temporal)?;
        let (step, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2)?;
        let replayed = if let Some(ref g) = self.graph {
            g.replay(
                &self.weights,
                &self.adam,
                &self.bias,
                &self.grads,
                &self.temporal,
                &self.mamba_input,
                &self.d_temporal,
                &self.ssm_states,
                &self.k_states,
                &self.v_states,
                &self.angle_states,
            )?;
            true
        } else {
            self.step_eager()?;
            false
        };
        Ok(StepMetrics::plain(step, replayed))
    }

    fn step_eager(&mut self) -> Result<(), String> {
        self.grads.zero(&self.ctx.stream)?;
        gpu_forward_mamba3_backbone(
            &self.ctx,
            &self.m3k,
            &mut self.temporal,
            &mut self.acts,
            &self.weights,
            &self.mamba_input,
            &mut self.ssm_states,
            &mut self.k_states,
            &mut self.v_states,
            &mut self.angle_states,
            &mut self.scratch,
            &self.dims,
        )?;
        gpu_backward_mamba3_backbone(
            &self.ctx,
            &self.m3k,
            &mut self.d_temporal,
            &self.acts,
            &self.weights,
            &self.grads,
            &mut self.scratch,
            &self.dims,
        )?;
        step_m3_capturable(
            &self.ctx,
            &self.m3k.adamw_step_f32_capturable,
            &self.adam,
            self.bias.ptr(),
            &mut self.weights,
            &self.grads,
        )?;
        Ok(())
    }

    pub fn snapshot_master(&self) -> Result<Mamba3Weights, String> {
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("pre-snapshot sync: {e:?}"))?;
        let w = &self.weights;
        let input_dim = self.mamba_input.len() / (self.dims.batch * self.dims.seq_len);
        let mut out = Mamba3Weights::zeros(&self.cfg, input_dim);
        out.input_proj_w = w.input_proj_w.to_cpu(&self.ctx.stream)?;
        out.input_proj_b = w.input_proj_b.to_cpu(&self.ctx.stream)?;
        for (i, lw) in out.layers.iter_mut().enumerate() {
            let g = &w.layers[i];
            lw.norm_weight = g.norm_weight.to_cpu(&self.ctx.stream)?;
            lw.in_proj_w = g.in_proj_w.to_cpu(&self.ctx.stream)?;
            lw.dt_bias = g.dt_bias.to_cpu(&self.ctx.stream)?;
            lw.b_norm_weight = g.b_norm_weight.to_cpu(&self.ctx.stream)?;
            lw.c_norm_weight = g.c_norm_weight.to_cpu(&self.ctx.stream)?;
            lw.b_bias = g.b_bias.to_cpu(&self.ctx.stream)?;
            lw.c_bias = g.c_bias.to_cpu(&self.ctx.stream)?;
            lw.d_param = g.d_param.to_cpu(&self.ctx.stream)?;
            lw.norm_gate_weight = g.norm_gate_weight.to_cpu(&self.ctx.stream)?;
            lw.out_proj_w = g.out_proj_w.to_cpu(&self.ctx.stream)?;
        }
        out.norm_f_weight = w.norm_f_weight.to_cpu(&self.ctx.stream)?;
        Ok(out)
    }
}
