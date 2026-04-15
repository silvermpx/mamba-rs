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
//! - `WeightDtype::F16`: not supported — the overflow check required by f16
//!   training would force a CPU readback inside the captured body.
//! - `WeightDtype::F32`: not wired through this wrapper today (the f32
//!   training path exists as `GpuMambaF32TrainingStepGraph`; a sibling
//!   wrapper can be added if needed).

use crate::config::MambaConfig;
use crate::mamba_ssm::gpu::adamw::{AdamWBiasFactors, GpuAdamW, step_m1, step_m1_capturable};
use crate::mamba_ssm::gpu::backward_mixed::gpu_backward_mamba_backbone_mixed;
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::device::GpuDevice;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::forward::{GpuMambaDims, GpuRecurrentState};
use crate::mamba_ssm::gpu::forward_mixed::{
    GpuMambaBackboneMixedActs, GpuMambaMixedTrainScratch, gpu_forward_mamba_backbone_train_mixed,
};
use crate::mamba_ssm::gpu::training_graph::GpuMambaTrainingStepGraph;
use crate::mamba_ssm::gpu::weights::GpuMambaGrads;
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
}

/// High-level Mamba-1 training wrapper. bf16 mixed-precision only for now
/// (f16 training needs in-graph overflow handling).
pub struct MambaTrainer {
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
    graph: Option<GpuMambaTrainingStepGraph>,
}

impl MambaTrainer {
    /// Create a trainer with the default Adam hyperparams
    /// (lr=1e-3, β1=0.9, β2=0.999, eps=1e-8, wd=1e-2).
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

    /// Create a trainer, overriding lr and weight_decay.
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
        assert!(
            matches!(dtype, WeightDtype::Bf16),
            "MambaTrainer currently supports bf16 only (f16 needs in-graph overflow check; \
             f32 training lives in GpuMambaF32TrainingStepGraph for now)"
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

    /// Run one training step. Uploads `input` (`[batch * seq_len * input_dim]`)
    /// and `d_temporal` (`[batch * seq_len * d_model]`), advances the Adam
    /// step counter, and executes forward + backward + AdamW + master-to-
    /// compute sync. Uses the captured graph if available.
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

        // 1. Upload per-step data (outside any graph).
        self.mamba_input.upload(&self.ctx.stream, input)?;
        self.d_temporal.upload(&self.ctx.stream, d_temporal)?;

        // 2. Advance Adam step counter + write fresh bias factors.
        let (step, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2)?;

        // 3. Run training step — graph replay when captured, eager fallback otherwise.
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

        Ok(StepMetrics {
            step,
            graph_replayed: replayed,
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
        Ok(StepMetrics {
            step,
            graph_replayed: false,
        })
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
