//! High-level Mamba-1 training API.
//!
//! Mirrors the shape of [`super::inference::GpuMambaBackbone`] but on the
//! training side. A single [`MambaTrainer`] owns EVERY piece of state
//! needed to run a full training step:
//!
//!   * mixed-precision weights (master + compute shadow)
//!   * gradient arena
//!   * AdamW optimizer state + bias-factor device buffer
//!   * saved activations (`acts`) + scratch
//!   * recurrent state buffers
//!   * pre-allocated input / d_temporal upload buffers
//!   * the CUDA Graph holder (lazily captured)
//!
//! ## Usage
//! ```ignore
//! let mut trainer = MambaTrainer::new_with_dtype(
//!     0, &cpu_weights, cfg, input_dim=dm, batch=8, seq_len=128,
//!     WeightDtype::Bf16,
//! )?;
//!
//! // First few steps run eagerly so cuBLAS / lazy CUDA resources settle.
//! for _ in 0..3 {
//!     trainer.step(&input, &d_temporal)?;
//! }
//! trainer.capture_graph()?;  // subsequent steps are graph-accelerated
//!
//! for _ in 0..n_steps {
//!     trainer.step(&input, &d_temporal)?;
//! }
//! ```
//!
//! ## Precision support
//! - `WeightDtype::Bf16`: full graph capture, sync_master_to_compute, AMP-style
//!   master weights in f32.
//! - `WeightDtype::F32`: full graph capture, no compute shadow (weights stay
//!   in f32 throughout). See [`MambaTrainerF32`].
//! - `WeightDtype::F16`: not supported — the overflow check required by f16
//!   training would force a CPU readback inside the captured body. Planned
//!   for a future step via a device-side skip-on-overflow kernel.

use crate::config::MambaConfig;
use crate::mamba_ssm::gpu::adamw::{AdamWBiasFactors, GpuAdamW, step_m1, step_m1_capturable};
use crate::mamba_ssm::gpu::backward::gpu_backward_mamba_backbone;
use crate::mamba_ssm::gpu::backward_mixed::gpu_backward_mamba_backbone_mixed;
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::device::GpuDevice;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::forward::{
    GpuMambaBackboneActs, GpuMambaDims, GpuMambaScratch, GpuRecurrentState,
    gpu_forward_mamba_backbone,
};
use crate::mamba_ssm::gpu::forward_mixed::{
    GpuMambaBackboneMixedActs, GpuMambaMixedTrainScratch, gpu_forward_mamba_backbone_train_mixed,
};
use crate::mamba_ssm::gpu::loss_scaler::{
    DynamicLossScaler, OverflowFlag, check_inf_nan_gpu, scale_grads_gpu,
};
use crate::mamba_ssm::gpu::training_graph::{
    GpuMambaF32TrainingStepGraph, GpuMambaTrainingStepGraph,
};
use crate::mamba_ssm::gpu::weights::{GpuMambaGrads, GpuMambaTrainWeights};
use crate::mamba_ssm::gpu::weights_mixed_train::GpuMambaTrainMixedWeights;
use crate::weights::MambaWeights;

/// Per-step metrics returned by [`MambaTrainer::step`].
#[derive(Debug, Clone)]
pub struct StepMetrics {
    /// 1-indexed step counter (matches `adam.step`).
    pub step: u64,
    /// Whether the step was executed via captured-graph replay (true) or
    /// the eager kernel-by-kernel path (false).
    pub graph_replayed: bool,
    /// `Some(scale)` when the f16 loss-scaler is active. `None` for bf16 /
    /// f32 where no scaling is applied.
    pub loss_scale: Option<f32>,
    /// `Some(true)` if the f16 loss-scaler detected an inf/nan in the
    /// grad arena and the optimizer step was skipped. `None` for bf16/f32.
    pub overflow_skipped: Option<bool>,
}

impl StepMetrics {
    /// Convenience constructor for paths without loss-scaler activity.
    pub fn plain(step: u64, graph_replayed: bool) -> Self {
        Self {
            step,
            graph_replayed,
            loss_scale: None,
            overflow_skipped: None,
        }
    }
}

/// Internal precision-dispatch enum. Hidden behind [`MambaTrainer`] so the
/// public API is a single struct with a single set of method names — caller
/// never matches `F32`/`Mixed` directly. Mirrors `inference::BackboneEngine`.
enum TrainerInner {
    F32(Box<MambaTrainerF32>),
    Mixed(Box<MambaTrainerMixed>),
}

/// High-level Mamba-1 training wrapper. Same shape as
/// [`super::inference::GpuMambaBackbone`]: one public struct, one method
/// per operation, dtype dispatch happens internally on the private enum.
pub struct MambaTrainer {
    inner: TrainerInner,
}

impl MambaTrainer {
    /// Construct with default Adam hyperparams (lr=1e-3, wd=1e-2).
    pub fn new_with_dtype(
        gpu_ordinal: usize,
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
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
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
        input_dim: usize,
        batch: usize,
        seq_len: usize,
        dtype: WeightDtype,
        lr: f32,
        weight_decay: f32,
    ) -> Result<Self, String> {
        let inner = match dtype {
            WeightDtype::F32 => TrainerInner::F32(Box::new(MambaTrainerF32::new_full(
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
                TrainerInner::Mixed(Box::new(MambaTrainerMixed::new_full(
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
            TrainerInner::F32(_) => WeightDtype::F32,
            TrainerInner::Mixed(t) => t.dtype,
        }
    }

    pub fn batch(&self) -> usize {
        match &self.inner {
            TrainerInner::F32(t) => t.batch,
            TrainerInner::Mixed(t) => t.batch,
        }
    }

    pub fn seq_len(&self) -> usize {
        match &self.inner {
            TrainerInner::F32(t) => t.seq_len,
            TrainerInner::Mixed(t) => t.seq_len,
        }
    }

    pub fn ctx(&self) -> &GpuCtx {
        match &self.inner {
            TrainerInner::F32(t) => &t.ctx,
            TrainerInner::Mixed(t) => &t.ctx,
        }
    }

    pub fn has_graph(&self) -> bool {
        match &self.inner {
            TrainerInner::F32(t) => t.graph.is_some(),
            TrainerInner::Mixed(t) => t.graph.is_some(),
        }
    }

    pub fn reset_state(&mut self) -> Result<(), String> {
        match &mut self.inner {
            TrainerInner::F32(t) => t.reset_state(),
            TrainerInner::Mixed(t) => t.reset_state(),
        }
    }

    pub fn capture_graph(&mut self) -> Result<(), String> {
        match &mut self.inner {
            TrainerInner::F32(t) => t.capture_graph(),
            TrainerInner::Mixed(t) => t.capture_graph(),
        }
    }

    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        match &mut self.inner {
            TrainerInner::F32(t) => t.step(input, d_temporal),
            TrainerInner::Mixed(t) => t.step(input, d_temporal),
        }
    }

    pub fn snapshot_master(&self) -> Result<MambaWeights, String> {
        match &self.inner {
            TrainerInner::F32(t) => t.snapshot_master(),
            TrainerInner::Mixed(t) => t.snapshot_master(),
        }
    }
}

/// bf16 mixed-precision training inner (master f32 + compute bf16 shadow +
/// sync_master_to_compute each step).
pub struct MambaTrainerMixed {
    ctx: GpuCtx,
    cfg: MambaConfig,
    batch: usize,
    seq_len: usize,
    dtype: WeightDtype,

    // Weights + optimizer state.
    pub weights: GpuMambaTrainMixedWeights,
    pub grads: GpuMambaGrads,
    pub adam: GpuAdamW,
    bias: AdamWBiasFactors,

    // Activations + scratch (forward saves → backward reads).
    acts: GpuMambaBackboneMixedActs,
    scratch: GpuMambaMixedTrainScratch,

    // Recurrent state + per-training standalone a_neg_all.
    state: GpuRecurrentState,
    a_neg_all: GpuBuffer,

    // Upload buffers (stable pointers — reused every step).
    mamba_input: GpuBuffer,
    d_temporal: GpuBuffer,

    // Lazily-populated CUDA Graph. None → eager path; Some → replayed.
    // Always None for f16 (loss-scaler overflow check requires CPU readback,
    // which breaks graph capture).
    graph: Option<GpuMambaTrainingStepGraph>,

    // f16 AMP loss scaler — populated for `WeightDtype::F16`, None otherwise.
    // When present, every step scales d_temporal by `scaler.scale()` before
    // backward, then checks the grad arena for inf/nan and either unscales
    // and runs AdamW or skips the step and backs off the scale.
    scaler: Option<DynamicLossScaler>,
    overflow_flag: Option<OverflowFlag>,
    /// Persistent device buffer for the scaled d_temporal (kept here so its
    /// pointer is stable across steps).
    d_temporal_scaled: Option<GpuBuffer>,
}

impl MambaTrainerMixed {
    #[allow(clippy::too_many_arguments)]
    fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
        input_dim: usize,
        batch: usize,
        seq_len: usize,
        dtype: WeightDtype,
        lr: f32,
        weight_decay: f32,
    ) -> Result<Self, String> {
        assert!(
            matches!(dtype, WeightDtype::Bf16 | WeightDtype::F16),
            "MambaTrainerMixed accepts Bf16 or F16; got {dtype:?}"
        );

        let device = GpuDevice::new(gpu_ordinal)?;
        let ctx = GpuCtx::new(&device)?;

        let weights = GpuMambaTrainMixedWeights::from_cpu(&ctx.stream, cpu_weights, &cfg, dtype)?;

        let d_inner = cfg.d_inner();
        let d_state = cfg.d_state;
        let d_conv = cfg.d_conv;
        let n_layers = cfg.n_layers;

        let dims = GpuMambaDims {
            batch,
            d_model: cfg.d_model,
            d_inner,
            d_state,
            d_conv,
            dt_rank: cfg.dt_rank(),
            xdbl_dim: cfg.xdbl_dim(),
            seq_len,
            mamba_input_dim: input_dim,
            n_layers,
        };

        let acts = GpuMambaBackboneMixedActs::new(&ctx.stream, &dims, dtype)?;
        let scratch = GpuMambaMixedTrainScratch::new(&ctx.stream, &dims, dtype)?;

        // Seed recurrent state: conv/ssm zero, a_neg = -exp(a_log).
        let mut a_neg_flat = vec![0.0f32; n_layers * d_inner * d_state];
        for (l, lw) in cpu_weights.layers.iter().enumerate() {
            for i in 0..d_inner * d_state {
                a_neg_flat[l * d_inner * d_state + i] = -lw.a_log[i].exp();
            }
        }
        let mut a_neg_all = GpuBuffer::zeros(&ctx.stream, n_layers * d_inner * d_state)?;
        a_neg_all.upload(&ctx.stream, &a_neg_flat)?;

        let mut state = GpuRecurrentState {
            conv_states: GpuBuffer::zeros(&ctx.stream, n_layers * d_inner * d_conv)?,
            ssm_states: GpuBuffer::zeros(&ctx.stream, n_layers * d_inner * d_state)?,
            a_neg_all: GpuBuffer::zeros(&ctx.stream, n_layers * d_inner * d_state)?,
        };
        state.a_neg_all.upload(&ctx.stream, &a_neg_flat)?;

        let mamba_input = GpuBuffer::zeros(&ctx.stream, batch * seq_len * input_dim)?;
        let d_temporal = GpuBuffer::zeros(&ctx.stream, batch * seq_len * cfg.d_model)?;
        let grads = GpuMambaGrads::new(&ctx.stream, &cfg, input_dim)?;

        let adam = GpuAdamW::new(&ctx.stream, grads.flat.len())?
            .with_lr(lr)
            .with_weight_decay(weight_decay);
        let bias = AdamWBiasFactors::new(&ctx.stream)?;

        // f16 needs the dynamic loss scaler + a separate scratch buffer for
        // the scaled d_temporal (so the original caller-provided values stay
        // untouched). bf16 has the same dynamic range as f32 and skips both.
        let (scaler, overflow_flag, d_temporal_scaled) = if matches!(dtype, WeightDtype::F16) {
            let s = DynamicLossScaler::new();
            let f = OverflowFlag::new(&ctx.stream)?;
            let scaled = GpuBuffer::zeros(&ctx.stream, batch * seq_len * cfg.d_model)?;
            (Some(s), Some(f), Some(scaled))
        } else {
            (None, None, None)
        };

        ctx.stream
            .synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;

        Ok(Self {
            ctx,
            cfg,
            batch,
            seq_len,
            dtype,
            weights,
            grads,
            adam,
            bias,
            acts,
            scratch,
            state,
            a_neg_all,
            mamba_input,
            d_temporal,
            graph: None,
            scaler,
            overflow_flag,
            d_temporal_scaled,
        })
    }

    pub fn dtype(&self) -> WeightDtype {
        self.dtype
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn ctx(&self) -> &GpuCtx {
        &self.ctx
    }

    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Reset recurrent state (conv_states + ssm_states) to zero. Keeps
    /// `a_neg_all` populated — it's a fixed function of the current
    /// weights and must survive resets.
    pub fn reset_state(&mut self) -> Result<(), String> {
        self.state.conv_states.zero(&self.ctx.stream)?;
        self.state.ssm_states.zero(&self.ctx.stream)?;
        Ok(())
    }

    /// Capture the training-step CUDA Graph. Call once after at least one
    /// warmup [`Self::step`] so cuBLAS has selected its kernels and lazy
    /// resources have settled.
    pub fn capture_graph(&mut self) -> Result<(), String> {
        if matches!(self.dtype, WeightDtype::F16) {
            return Err(
                "f16 training cannot use a captured graph (overflow check requires CPU \
                 readback). Step 22 will add a device-side skip-on-overflow kernel."
                    .into(),
            );
        }
        // Make sure the bias buffer holds something finite — capture_into_graph
        // will record the AdamW kernel reading from it. Real values are
        // overwritten per step by `step()`.
        self.bias.write(&self.ctx.stream, 1.0, 1.0)?;

        let g = GpuMambaTrainingStepGraph::capture(
            &self.ctx,
            &self.cfg,
            &mut self.weights,
            &self.adam,
            &self.bias,
            &mut self.grads,
            &mut self.acts,
            &mut self.scratch,
            &self.a_neg_all,
            &self.mamba_input,
            &mut self.d_temporal,
            &mut self.state,
            self.batch,
            self.seq_len,
        )?;
        self.graph = Some(g);
        Ok(())
    }

    /// Run one training step. For bf16 this is the existing
    /// forward+backward+AdamW+sync path (graph-accelerated when captured).
    /// For f16 the path runs eager only and goes through the dynamic loss
    /// scaler (scale d_temporal → backward → check overflow → unscale +
    /// step OR skip + back off).
    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        assert_eq!(
            input.len(),
            self.mamba_input.len(),
            "input shape mismatch: expected {} got {}",
            self.mamba_input.len(),
            input.len()
        );
        assert_eq!(
            d_temporal.len(),
            self.d_temporal.len(),
            "d_temporal shape mismatch: expected {} got {}",
            self.d_temporal.len(),
            d_temporal.len()
        );

        if matches!(self.dtype, WeightDtype::F16) {
            return self.step_f16(input, d_temporal);
        }

        // bf16 path: existing graph / eager dispatch.
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
                &self.a_neg_all,
                &self.mamba_input,
                &self.d_temporal,
                &self.state,
            )?;
            true
        } else {
            self.step_eager()?;
            false
        };

        Ok(StepMetrics::plain(step, replayed))
    }

    /// f16 step (eager, no graph). Mirrors PyTorch GradScaler protocol:
    ///   1. Upload d_temporal scaled by `scaler.scale()`
    ///   2. forward + backward → grads (also scaled)
    ///   3. check_inf_nan over the grad arena, CPU readback of the flag
    ///   4. clean: unscale grads (`*= 1/scale`), AdamW + sync, scaler.update(false)
    ///      overflow: skip AdamW + sync, scaler.update(true) → scale halves
    fn step_f16(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        let scale = self.scaler.as_ref().expect("f16 scaler").scale();

        // 1. Upload input + scaled d_temporal.
        self.mamba_input.upload(&self.ctx.stream, input)?;
        // d_temporal_scaled = d_temporal * scale (CPU-side multiply — small
        // dN buffer, < 1ms even at max shapes; avoids an extra GPU kernel).
        let scaled: Vec<f32> = d_temporal.iter().map(|v| v * scale).collect();
        let dt_scaled = self.d_temporal_scaled.as_mut().expect("f16 dt_scaled");
        dt_scaled.upload(&self.ctx.stream, &scaled)?;

        // 2. forward + backward into self.grads (which will be the scaled
        //    gradient arena).
        self.grads.zero(&self.ctx.stream)?;
        gpu_forward_mamba_backbone_train_mixed(
            &self.ctx,
            &mut self.acts,
            &self.weights,
            &self.mamba_input,
            &mut self.state,
            &mut self.scratch,
        )?;
        gpu_backward_mamba_backbone_mixed(
            &self.ctx,
            dt_scaled,
            &self.grads,
            &self.acts,
            &self.weights.compute,
            &self.a_neg_all,
            &mut self.scratch,
        )?;

        // 3. Check for inf/nan in the grad arena.
        let flag = self.overflow_flag.as_mut().expect("f16 overflow flag");
        flag.zero(&self.ctx.stream)?;
        check_inf_nan_gpu(&self.ctx, &self.ctx.kernels, flag, &self.grads.flat)?;
        let overflow = flag.read(&self.ctx.stream)? != 0;

        // 4. Either unscale + step or skip + back off.
        let step;
        if overflow {
            // Skip optimizer entirely. Adam step counter is NOT bumped — Adam
            // bias correction must reflect actual updates, not skipped ones
            // (matches PyTorch GradScaler.step semantics).
            step = self.adam.step;
            self.scaler.as_mut().expect("f16 scaler").update(true);
        } else {
            scale_grads_gpu(
                &self.ctx,
                &self.ctx.kernels,
                &mut self.grads.flat,
                1.0 / scale,
            )?;
            let (s, bc1, bc2) = self.adam.advance();
            step = s;
            self.bias.write(&self.ctx.stream, bc1, bc2)?;
            step_m1_capturable(
                &self.ctx,
                &self.ctx.kernels.adamw_step_f32_capturable,
                &self.adam,
                self.bias.ptr(),
                &mut self.weights.master,
                &self.grads,
            )?;
            self.weights.sync_master_to_compute(&self.ctx)?;
            self.scaler.as_mut().expect("f16 scaler").update(false);
        }

        Ok(StepMetrics {
            step,
            graph_replayed: false,
            loss_scale: Some(scale),
            overflow_skipped: Some(overflow),
        })
    }

    /// Eager fallback (used before [`Self::capture_graph`] is called and
    /// shared as the body of capture). Mirrors the exact op sequence the
    /// captured graph records.
    fn step_eager(&mut self) -> Result<(), String> {
        self.grads.zero(&self.ctx.stream)?;
        gpu_forward_mamba_backbone_train_mixed(
            &self.ctx,
            &mut self.acts,
            &self.weights,
            &self.mamba_input,
            &mut self.state,
            &mut self.scratch,
        )?;
        gpu_backward_mamba_backbone_mixed(
            &self.ctx,
            &mut self.d_temporal,
            &self.grads,
            &self.acts,
            &self.weights.compute,
            &self.a_neg_all,
            &mut self.scratch,
        )?;
        // Use the capturable kernel here too so the graph's and eager path's
        // numerics are bit-identical (the kernel reads bias_factors from a
        // device pointer that the `bias.write` above populated).
        step_m1_capturable(
            &self.ctx,
            &self.ctx.kernels.adamw_step_f32_capturable,
            &self.adam,
            self.bias.ptr(),
            &mut self.weights.master,
            &self.grads,
        )?;
        self.weights.sync_master_to_compute(&self.ctx)?;
        Ok(())
    }

    /// Alternative non-capturable step that uses the scalar-arg AdamW
    /// kernel (the original `step_m1`). Produces the same math as
    /// [`Self::step_eager`] but does not exercise the graph's bias-factor
    /// device buffer path — useful for one-off debugging.
    pub fn step_debug_scalar_adamw(&mut self) -> Result<StepMetrics, String> {
        assert!(
            self.graph.is_none(),
            "step_debug_scalar_adamw cannot run while a graph is captured"
        );
        let (step, _, _) = self.adam.advance();
        self.grads.zero(&self.ctx.stream)?;
        gpu_forward_mamba_backbone_train_mixed(
            &self.ctx,
            &mut self.acts,
            &self.weights,
            &self.mamba_input,
            &mut self.state,
            &mut self.scratch,
        )?;
        gpu_backward_mamba_backbone_mixed(
            &self.ctx,
            &mut self.d_temporal,
            &self.grads,
            &self.acts,
            &self.weights.compute,
            &self.a_neg_all,
            &mut self.scratch,
        )?;
        // step_m1 calls adam.advance() internally, so we need a fresh
        // GpuAdamW with step=step-1 to match. Easier: skip this path's
        // step bookkeeping and don't use it in the hot loop.
        step_m1(
            &self.ctx,
            &self.ctx.kernels.adamw_step_f32,
            &mut self.adam,
            &mut self.weights.master,
            &self.grads,
        )?;
        self.weights.sync_master_to_compute(&self.ctx)?;
        Ok(StepMetrics::plain(step, false))
    }

    /// Download the master weights to a CPU-side `MambaWeights` for
    /// checkpointing. Includes a stream sync.
    pub fn snapshot_master(&self) -> Result<MambaWeights, String> {
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("pre-snapshot sync: {e:?}"))?;
        let master = &self.weights.master;
        let mut out = MambaWeights::zeros(
            &self.cfg,
            self.mamba_input.len() / (self.batch * self.seq_len),
        );
        out.input_proj_w = master.input_proj_w.to_cpu(&self.ctx.stream)?;
        out.input_proj_b = master.input_proj_b.to_cpu(&self.ctx.stream)?;
        for (i, lw) in out.layers.iter_mut().enumerate() {
            let g = &master.layers[i];
            lw.norm_weight = g.norm_weight.to_cpu(&self.ctx.stream)?;
            lw.in_proj_w = g.in_proj_w.to_cpu(&self.ctx.stream)?;
            lw.conv1d_weight = g.conv1d_weight.to_cpu(&self.ctx.stream)?;
            lw.conv1d_bias = g.conv1d_bias.to_cpu(&self.ctx.stream)?;
            lw.x_proj_w = g.x_proj_w.to_cpu(&self.ctx.stream)?;
            lw.dt_proj_w = g.dt_proj_w.to_cpu(&self.ctx.stream)?;
            lw.dt_proj_b = g.dt_proj_b.to_cpu(&self.ctx.stream)?;
            lw.a_log = g.a_log.to_cpu(&self.ctx.stream)?;
            lw.d_param = g.d_param.to_cpu(&self.ctx.stream)?;
            lw.out_proj_w = g.out_proj_w.to_cpu(&self.ctx.stream)?;
            lw.a_neg = lw.a_log.iter().map(|v| -v.exp()).collect();
        }
        out.norm_f_weight = master.norm_f_weight.to_cpu(&self.ctx.stream)?;
        Ok(out)
    }
}

// ════════════════════════════════════════════════════════════════════════
// f32 training wrapper (no master/compute split, no half_staging).
// ════════════════════════════════════════════════════════════════════════

/// f32 training inner. Weights stay in f32 throughout — no compute shadow,
/// no master→compute sync step in the training loop.
pub struct MambaTrainerF32 {
    pub ctx: GpuCtx,
    pub cfg: MambaConfig,
    pub batch: usize,
    pub seq_len: usize,
    pub weights: GpuMambaTrainWeights,
    pub grads: GpuMambaGrads,
    pub adam: GpuAdamW,
    bias: AdamWBiasFactors,
    acts: GpuMambaBackboneActs,
    scratch: GpuMambaScratch,
    state: GpuRecurrentState,
    a_neg_all: GpuBuffer,
    temporal: GpuBuffer,
    mamba_input: GpuBuffer,
    d_temporal: GpuBuffer,
    graph: Option<GpuMambaF32TrainingStepGraph>,
}

impl MambaTrainerF32 {
    #[allow(clippy::too_many_arguments)]
    fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
        input_dim: usize,
        batch: usize,
        seq_len: usize,
        lr: f32,
        weight_decay: f32,
    ) -> Result<Self, String> {
        let device = GpuDevice::new(gpu_ordinal)?;
        let ctx = GpuCtx::new(&device)?;

        let weights = GpuMambaTrainWeights::from_cpu(&ctx.stream, cpu_weights)?;

        let d_inner = cfg.d_inner();
        let d_state = cfg.d_state;
        let d_conv = cfg.d_conv;
        let n_layers = cfg.n_layers;

        let dims = GpuMambaDims {
            batch,
            d_model: cfg.d_model,
            d_inner,
            d_state,
            d_conv,
            dt_rank: cfg.dt_rank(),
            xdbl_dim: cfg.xdbl_dim(),
            seq_len,
            mamba_input_dim: input_dim,
            n_layers,
        };

        let acts = GpuMambaBackboneActs::new(&ctx.stream, &dims)?;
        let scratch = GpuMambaScratch::new(&ctx.stream, &dims)?;

        let mut a_neg_flat = vec![0.0f32; n_layers * d_inner * d_state];
        for (l, lw) in cpu_weights.layers.iter().enumerate() {
            for i in 0..d_inner * d_state {
                a_neg_flat[l * d_inner * d_state + i] = -lw.a_log[i].exp();
            }
        }
        let mut a_neg_all = GpuBuffer::zeros(&ctx.stream, n_layers * d_inner * d_state)?;
        a_neg_all.upload(&ctx.stream, &a_neg_flat)?;

        let mut state = GpuRecurrentState {
            conv_states: GpuBuffer::zeros(&ctx.stream, n_layers * d_inner * d_conv)?,
            ssm_states: GpuBuffer::zeros(&ctx.stream, n_layers * d_inner * d_state)?,
            a_neg_all: GpuBuffer::zeros(&ctx.stream, n_layers * d_inner * d_state)?,
        };
        state.a_neg_all.upload(&ctx.stream, &a_neg_flat)?;

        let temporal = GpuBuffer::zeros(&ctx.stream, batch * seq_len * cfg.d_model)?;
        let mamba_input = GpuBuffer::zeros(&ctx.stream, batch * seq_len * input_dim)?;
        let d_temporal = GpuBuffer::zeros(&ctx.stream, batch * seq_len * cfg.d_model)?;
        let grads = GpuMambaGrads::new(&ctx.stream, &cfg, input_dim)?;

        let adam = GpuAdamW::new(&ctx.stream, grads.flat.len())?
            .with_lr(lr)
            .with_weight_decay(weight_decay);
        let bias = AdamWBiasFactors::new(&ctx.stream)?;

        ctx.stream
            .synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;

        Ok(Self {
            ctx,
            cfg,
            batch,
            seq_len,
            weights,
            grads,
            adam,
            bias,
            acts,
            scratch,
            state,
            a_neg_all,
            temporal,
            mamba_input,
            d_temporal,
            graph: None,
        })
    }

    pub fn reset_state(&mut self) -> Result<(), String> {
        self.state.conv_states.zero(&self.ctx.stream)?;
        self.state.ssm_states.zero(&self.ctx.stream)?;
        Ok(())
    }

    pub fn capture_graph(&mut self) -> Result<(), String> {
        self.bias.write(&self.ctx.stream, 1.0, 1.0)?;
        let g = GpuMambaF32TrainingStepGraph::capture(
            &self.ctx,
            &mut self.weights,
            &self.adam,
            &self.bias,
            &mut self.grads,
            &mut self.acts,
            &mut self.scratch,
            &self.a_neg_all,
            &mut self.temporal,
            &self.mamba_input,
            &mut self.d_temporal,
            &mut self.state,
            self.batch,
            self.seq_len,
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
                &self.a_neg_all,
                &self.mamba_input,
                &self.d_temporal,
                &self.state,
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
        gpu_forward_mamba_backbone(
            &self.ctx,
            &mut self.temporal,
            &mut self.acts,
            &self.weights,
            &self.mamba_input,
            &mut self.state,
            &mut self.scratch,
        )?;
        gpu_backward_mamba_backbone(
            &self.ctx,
            &mut self.d_temporal,
            &self.grads,
            &self.acts,
            &self.weights,
            &self.a_neg_all,
            &mut self.scratch,
        )?;
        step_m1_capturable(
            &self.ctx,
            &self.ctx.kernels.adamw_step_f32_capturable,
            &self.adam,
            self.bias.ptr(),
            &mut self.weights,
            &self.grads,
        )?;
        Ok(())
    }

    pub fn snapshot_master(&self) -> Result<MambaWeights, String> {
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("pre-snapshot sync: {e:?}"))?;
        let w = &self.weights;
        let input_dim = self.mamba_input.len() / (self.batch * self.seq_len);
        let mut out = MambaWeights::zeros(&self.cfg, input_dim);
        out.input_proj_w = w.input_proj_w.to_cpu(&self.ctx.stream)?;
        out.input_proj_b = w.input_proj_b.to_cpu(&self.ctx.stream)?;
        for (i, lw) in out.layers.iter_mut().enumerate() {
            let g = &w.layers[i];
            lw.norm_weight = g.norm_weight.to_cpu(&self.ctx.stream)?;
            lw.in_proj_w = g.in_proj_w.to_cpu(&self.ctx.stream)?;
            lw.conv1d_weight = g.conv1d_weight.to_cpu(&self.ctx.stream)?;
            lw.conv1d_bias = g.conv1d_bias.to_cpu(&self.ctx.stream)?;
            lw.x_proj_w = g.x_proj_w.to_cpu(&self.ctx.stream)?;
            lw.dt_proj_w = g.dt_proj_w.to_cpu(&self.ctx.stream)?;
            lw.dt_proj_b = g.dt_proj_b.to_cpu(&self.ctx.stream)?;
            lw.a_log = g.a_log.to_cpu(&self.ctx.stream)?;
            lw.d_param = g.d_param.to_cpu(&self.ctx.stream)?;
            lw.out_proj_w = g.out_proj_w.to_cpu(&self.ctx.stream)?;
            lw.a_neg = lw.a_log.iter().map(|v| -v.exp()).collect();
        }
        out.norm_f_weight = w.norm_f_weight.to_cpu(&self.ctx.stream)?;
        Ok(out)
    }
}
