//! High-level Mamba-3 training API.
//!
//! M3 analogue of [`crate::mamba_ssm::gpu::trainer::MambaTrainer`]. Owns
//! mixed-precision weights + grads + Adam + acts + scratch + state + graph.
//!
//! Supports f32, bf16, and f16 (with dynamic loss scaler), including CUDA
//! Graph capture for all three. See the M1 trainer for the
//! precision-support rationale.

use cudarc::driver::PushKernelArg;

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
pub use crate::mamba_ssm::gpu::trainer::TrainSessionCfg;
use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::gpu::backward::gpu_backward_mamba3_backbone;
use crate::mamba3_siso::gpu::backward_mixed::gpu_backward_mamba3_backbone_mixed;
use crate::mamba3_siso::gpu::forward::gpu_forward_mamba3_backbone;
use crate::mamba3_siso::gpu::forward_mixed::{
    GpuMamba3BackboneMixedActs, GpuMamba3MixedScratch, gpu_forward_mamba3_backbone_mixed,
};
use crate::mamba3_siso::gpu::kernels::Mamba3Kernels;
use crate::mamba3_siso::gpu::state::{
    GpuMamba3BackboneActs, GpuMamba3Dims, GpuMamba3Scratch, GpuMamba3StateBufs, M3Exec,
};
use crate::mamba3_siso::gpu::training_graph::{
    GpuMamba3F32TrainingStepGraph, GpuMamba3TrainingStepGraph, Mamba3F32Capture, Mamba3F32Replay,
    Mamba3MixedCapture, Mamba3MixedReplay,
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
            TrainSessionCfg {
                input_dim,
                batch,
                seq_len,
                lr: 1e-3,
                weight_decay: 1e-2,
            },
            dtype,
        )
    }

    pub fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        session: TrainSessionCfg,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        crate::mamba_ssm::gpu::launch::validate_kernel_arg_capacity(
            session.batch,
            session.seq_len,
            cfg.d_inner(),
            cfg.d_state,
        )?;
        let inner = match dtype {
            WeightDtype::F32 => Trainer3Inner::F32(Box::new(Mamba3TrainerF32::new_full(
                gpu_ordinal,
                cpu_weights,
                cfg,
                session,
            )?)),
            WeightDtype::Bf16 | WeightDtype::F16 => Trainer3Inner::Mixed(Box::new(
                Mamba3TrainerMixed::new_full(gpu_ordinal, cpu_weights, cfg, session, dtype)?,
            )),
        };
        Ok(Self { inner })
    }

    /// Weight storage dtype the trainer was constructed with.
    pub fn dtype(&self) -> WeightDtype {
        match &self.inner {
            Trainer3Inner::F32(_) => WeightDtype::F32,
            Trainer3Inner::Mixed(t) => t.dtype,
        }
    }

    /// Batch dimension fixed at construction.
    pub fn batch(&self) -> usize {
        match &self.inner {
            Trainer3Inner::F32(t) => t.dims.batch,
            Trainer3Inner::Mixed(t) => t.dims.batch,
        }
    }

    /// Sequence length fixed at construction.
    pub fn seq_len(&self) -> usize {
        match &self.inner {
            Trainer3Inner::F32(t) => t.dims.seq_len,
            Trainer3Inner::Mixed(t) => t.dims.seq_len,
        }
    }

    /// CUDA context the trainer runs on.
    pub fn ctx(&self) -> &GpuCtx {
        match &self.inner {
            Trainer3Inner::F32(t) => &t.ctx,
            Trainer3Inner::Mixed(t) => &t.ctx,
        }
    }

    /// `true` once [`Self::capture_graph`] has been called.
    pub fn has_graph(&self) -> bool {
        match &self.inner {
            Trainer3Inner::F32(t) => t.graph.is_some(),
            Trainer3Inner::Mixed(t) => t.has_graph(),
        }
    }

    /// Reset the recurrent SSM, K, V, and angle states to zero.
    pub fn reset_state(&mut self) -> Result<(), String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.reset_state(),
            Trainer3Inner::Mixed(t) => t.reset_state(),
        }
    }

    /// Record the full training step into a CUDA Graph. Run at least one
    /// warmup [`Self::step`] first. See [`MambaTrainer::capture_graph`]
    /// for the pointer-stability contract — same rules apply here.
    pub fn capture_graph(&mut self) -> Result<(), String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.capture_graph(),
            Trainer3Inner::Mixed(t) => t.capture_graph(),
        }
    }

    /// Run one training step on `(input, d_temporal)`. Shapes match
    /// [`MambaTrainer::step`]. Returns [`StepMetrics`].
    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.step(input, d_temporal),
            Trainer3Inner::Mixed(t) => t.step(input, d_temporal),
        }
    }

    /// Download f32 master weights for checkpointing.
    pub fn snapshot_master(&self) -> Result<Mamba3Weights, String> {
        match &self.inner {
            Trainer3Inner::F32(t) => t.snapshot_master(),
            Trainer3Inner::Mixed(t) => t.snapshot_master(),
        }
    }

    /// Serialize the dynamic loss scaler state for checkpoint resume.
    /// Returns `Some((scale, growth_tracker))` only for f16 training where
    /// the scaler is active; `None` for bf16 / f32. Mirrors
    /// [`MambaTrainer::scaler_state`].
    pub fn scaler_state(&self) -> Option<(f32, u32)> {
        match &self.inner {
            Trainer3Inner::F32(_) => None,
            Trainer3Inner::Mixed(t) => t.scaler_state(),
        }
    }

    /// Restore the dynamic loss scaler state saved via [`Self::scaler_state`].
    /// No-op for non-f16 trainers.
    pub fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) {
        if let Trainer3Inner::Mixed(ref mut t) = self.inner {
            t.load_scaler_state(scale, growth_tracker);
        }
    }
}

pub(crate) struct Mamba3TrainerMixed {
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
    /// Pointer-stability snapshots for the f16 graph.
    captured_f16_bias_ptr: u64,
    captured_f16_unscale_ptr: u64,
    captured_f16_overflow_ptr: u64,
    captured_f16_grads_ptr: u64,
    captured_f16_dt_scaled_ptr: u64,
}

impl Mamba3TrainerMixed {
    fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        session: TrainSessionCfg,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        // Fail at construction, not on the first step(): the mixed backward
        // currently supports only the RMSNormGated output path and the
        // identity-input_proj branch. The reference default
        // (is_outproj_norm = false) would otherwise construct successfully
        // — allocating every buffer and compiling 50+ kernels — and then
        // error on step().
        if !cfg.is_outproj_norm {
            return Err("Mamba3TrainerMixed: the silu_gate output path \
                 (is_outproj_norm = false) has no mixed-precision backward yet — \
                 set cfg.is_outproj_norm = true or train with WeightDtype::F32"
                .into());
        }
        if !cpu_weights.input_proj_w.is_empty() {
            return Err(
                "Mamba3TrainerMixed: non-identity input_proj is not yet supported in the \
                 mixed-precision pipeline — clear input_proj_w/input_proj_b (identity \
                 D2D branch) or train with WeightDtype::F32"
                    .into(),
            );
        }
        cfg.validate()?;
        let TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr,
            weight_decay,
        } = session;
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
            captured_f16_bias_ptr: 0,
            captured_f16_unscale_ptr: 0,
            captured_f16_overflow_ptr: 0,
            captured_f16_grads_ptr: 0,
            captured_f16_dt_scaled_ptr: 0,
        })
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
            &M3Exec {
                ctx: &self.ctx,
                kernels: &self.m3k,
                dims: &self.dims,
            },
            &self.cfg,
            Mamba3MixedCapture {
                train_w: &mut self.weights,
                adam: &self.adam,
                bias: &self.bias,
                grads: &mut self.grads,
                acts: &mut self.acts,
                f32_scratch: &mut self.f32_scratch,
                mixed_scratch: &mut self.mixed_scratch,
                temporal_f32: &mut self.temporal,
                mamba_input: &self.mamba_input,
                d_temporal: &mut self.d_temporal,
                states: GpuMamba3StateBufs {
                    ssm: &mut self.ssm_states,
                    k: &mut self.k_states,
                    v: &mut self.v_states,
                    angle: &mut self.angle_states,
                },
            },
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
                &Mamba3MixedReplay {
                    train_w: &self.weights,
                    adam: &self.adam,
                    bias: &self.bias,
                    grads: &self.grads,
                    temporal_f32: &self.temporal,
                    mamba_input: &self.mamba_input,
                    d_temporal: &self.d_temporal,
                    ssm_states: &self.ssm_states,
                    k_states: &self.k_states,
                    v_states: &self.v_states,
                    angle_states: &self.angle_states,
                },
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
        // Scale d_temporal on-device — the old path built a scaled Vec<f32>
        // on the host every step (B*T*d_model alloc + traversal).
        let dt_scaled = self.d_temporal_scaled.as_mut().expect("f16 dt_scaled");
        dt_scaled.upload(&self.ctx.stream, d_temporal)?;
        {
            let n = d_temporal.len() as i32;
            let mut builder = self
                .ctx
                .stream
                .launch_builder(&self.ctx.kernels.scale_grads_f32);
            builder.arg(dt_scaled.inner_mut());
            builder.arg(&scale);
            builder.arg(&n);
            unsafe { builder.launch(crate::mamba_ssm::gpu::launch::grid_1d(d_temporal.len())) }
                .map_err(|e| format!("scale d_temporal (m3 f16): {e:?}"))?;
        }

        if let Some(ref mut u) = self.unscale_factor {
            u.write(&self.ctx.stream, 1.0 / scale)?;
        }
        let prev_step = self.adam.step;
        let (next_step, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2)?;

        self.overflow_flag
            .as_mut()
            .expect("f16 overflow flag")
            .zero(&self.ctx.stream)?;

        let (step, overflow, replayed) = if let Some(ref g) = self.graph_f16 {
            assert_eq!(
                self.bias.ptr(),
                self.captured_f16_bias_ptr,
                "M3 f16 graph replay: bias pointer changed since capture"
            );
            assert_eq!(
                self.unscale_factor.as_ref().unwrap().ptr(),
                self.captured_f16_unscale_ptr,
                "M3 f16 graph replay: unscale_factor pointer changed since capture"
            );
            assert_eq!(
                self.overflow_flag
                    .as_ref()
                    .unwrap()
                    .stable_ptr(&self.ctx.stream),
                self.captured_f16_overflow_ptr,
                "M3 f16 graph replay: overflow_flag pointer changed since capture"
            );
            assert_eq!(
                self.grads.flat.cached_ptr(),
                self.captured_f16_grads_ptr,
                "M3 f16 graph replay: grads.flat pointer changed since capture"
            );
            assert_eq!(
                self.d_temporal_scaled.as_ref().unwrap().cached_ptr(),
                self.captured_f16_dt_scaled_ptr,
                "M3 f16 graph replay: d_temporal_scaled pointer changed since capture"
            );

            g.launch()
                .map_err(|e| format!("M3 f16 graph launch: {e:?}"))?;
            let overflow = self
                .overflow_flag
                .as_ref()
                .unwrap()
                .read(&self.ctx.stream)?
                != 0;
            self.scaler.as_mut().expect("f16 scaler").update(overflow);
            (next_step, overflow, true)
        } else {
            self.grads.zero(&self.ctx.stream)?;
            let exec = M3Exec {
                ctx: &self.ctx,
                kernels: &self.m3k,
                dims: &self.dims,
            };
            gpu_forward_mamba3_backbone_mixed(
                &exec,
                &mut self.temporal,
                &mut self.acts,
                &self.weights,
                &self.mamba_input,
                GpuMamba3StateBufs {
                    ssm: &mut self.ssm_states,
                    k: &mut self.k_states,
                    v: &mut self.v_states,
                    angle: &mut self.angle_states,
                },
                &mut self.mixed_scratch,
            )?;
            gpu_backward_mamba3_backbone_mixed(
                &exec,
                dt_scaled,
                &self.acts,
                &self.weights,
                &self.grads,
                &mut self.f32_scratch,
                &mut self.mixed_scratch,
            )?;
            check_inf_nan_gpu(
                &self.ctx,
                &self.ctx.kernels,
                self.overflow_flag.as_mut().unwrap(),
                &self.grads.flat,
            )?;
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

        // Snapshot every device buffer baked into the captured kernels.
        let snap_bias = self.bias.ptr();
        let snap_unscale = self.unscale_factor.as_ref().unwrap().ptr();
        let snap_overflow = self
            .overflow_flag
            .as_ref()
            .unwrap()
            .stable_ptr(&self.ctx.stream);
        let snap_grads = self.grads.flat.cached_ptr();
        let snap_dt_scaled = self.d_temporal_scaled.as_ref().unwrap().cached_ptr();

        let stream = self.ctx.stream.clone();
        let g = capture_into_graph(&stream, || {
            self.grads.zero(&self.ctx.stream)?;
            let exec = M3Exec {
                ctx: &self.ctx,
                kernels: &self.m3k,
                dims: &self.dims,
            };
            gpu_forward_mamba3_backbone_mixed(
                &exec,
                &mut self.temporal,
                &mut self.acts,
                &self.weights,
                &self.mamba_input,
                GpuMamba3StateBufs {
                    ssm: &mut self.ssm_states,
                    k: &mut self.k_states,
                    v: &mut self.v_states,
                    angle: &mut self.angle_states,
                },
                &mut self.mixed_scratch,
            )?;
            gpu_backward_mamba3_backbone_mixed(
                &exec,
                self.d_temporal_scaled.as_mut().unwrap(),
                &self.acts,
                &self.weights,
                &self.grads,
                &mut self.f32_scratch,
                &mut self.mixed_scratch,
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
        self.captured_f16_bias_ptr = snap_bias;
        self.captured_f16_unscale_ptr = snap_unscale;
        self.captured_f16_overflow_ptr = snap_overflow;
        self.captured_f16_grads_ptr = snap_grads;
        self.captured_f16_dt_scaled_ptr = snap_dt_scaled;
        Ok(())
    }

    fn step_eager(&mut self) -> Result<(), String> {
        self.grads.zero(&self.ctx.stream)?;
        let exec = M3Exec {
            ctx: &self.ctx,
            kernels: &self.m3k,
            dims: &self.dims,
        };
        gpu_forward_mamba3_backbone_mixed(
            &exec,
            &mut self.temporal,
            &mut self.acts,
            &self.weights,
            &self.mamba_input,
            GpuMamba3StateBufs {
                ssm: &mut self.ssm_states,
                k: &mut self.k_states,
                v: &mut self.v_states,
                angle: &mut self.angle_states,
            },
            &mut self.mixed_scratch,
        )?;
        gpu_backward_mamba3_backbone_mixed(
            &exec,
            &mut self.d_temporal,
            &self.acts,
            &self.weights,
            &self.grads,
            &mut self.f32_scratch,
            &mut self.mixed_scratch,
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

    /// Serialize dynamic loss scaler state. `None` when scaler is disabled
    /// (bf16). Mirrors `MambaTrainerMixed::scaler_state`.
    pub fn scaler_state(&self) -> Option<(f32, u32)> {
        self.scaler.as_ref().map(|s| s.state())
    }

    /// Restore scaler state from a prior `scaler_state()`. No-op when the
    /// scaler is disabled (bf16). Also rewrites the on-device unscale
    /// factor so the next f16 step uses the restored scale.
    pub fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) {
        if let Some(ref mut s) = self.scaler {
            s.load_state(scale, growth_tracker);
            if let Some(ref mut uf) = self.unscale_factor {
                let unscale = 1.0 / s.scale();
                let _ = uf.write(&self.ctx.stream, unscale);
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// f32 M3 training wrapper.
// ════════════════════════════════════════════════════════════════════════

pub(crate) struct Mamba3TrainerF32 {
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
    fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        session: TrainSessionCfg,
    ) -> Result<Self, String> {
        let TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr,
            weight_decay,
        } = session;
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
            &M3Exec {
                ctx: &self.ctx,
                kernels: &self.m3k,
                dims: &self.dims,
            },
            Mamba3F32Capture {
                weights: &mut self.weights,
                adam: &self.adam,
                bias: &self.bias,
                grads: &mut self.grads,
                acts: &mut self.acts,
                scratch: &mut self.scratch,
                temporal: &mut self.temporal,
                mamba_input: &self.mamba_input,
                d_temporal: &mut self.d_temporal,
                states: GpuMamba3StateBufs {
                    ssm: &mut self.ssm_states,
                    k: &mut self.k_states,
                    v: &mut self.v_states,
                    angle: &mut self.angle_states,
                },
            },
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
            g.replay(&Mamba3F32Replay {
                weights: &self.weights,
                adam: &self.adam,
                bias: &self.bias,
                grads: &self.grads,
                temporal: &self.temporal,
                mamba_input: &self.mamba_input,
                d_temporal: &self.d_temporal,
                ssm_states: &self.ssm_states,
                k_states: &self.k_states,
                v_states: &self.v_states,
                angle_states: &self.angle_states,
            })?;
            true
        } else {
            self.step_eager()?;
            false
        };
        Ok(StepMetrics::plain(step, replayed))
    }

    fn step_eager(&mut self) -> Result<(), String> {
        self.grads.zero(&self.ctx.stream)?;
        let exec = M3Exec {
            ctx: &self.ctx,
            kernels: &self.m3k,
            dims: &self.dims,
        };
        gpu_forward_mamba3_backbone(
            &exec,
            &mut self.temporal,
            &mut self.acts,
            &self.weights,
            &self.mamba_input,
            GpuMamba3StateBufs {
                ssm: &mut self.ssm_states,
                k: &mut self.k_states,
                v: &mut self.v_states,
                angle: &mut self.angle_states,
            },
            &mut self.scratch,
        )?;
        gpu_backward_mamba3_backbone(
            &exec,
            &mut self.d_temporal,
            &self.acts,
            &self.weights,
            &self.grads,
            &mut self.scratch,
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
