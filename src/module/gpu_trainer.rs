//! One surface over the GPU trainers of the four families.
//!
//! Every family's trainer has these methods of its own; [`GpuTrainer`]
//! names them for a training loop that takes any family. The master
//! weights come back through each trainer's `snapshot_master`, typed by its
//! family.

use crate::config::ModelFamily;
use crate::dist::DistContext;
use crate::gpu::adamw::AdamWStateBlob;
use crate::gpu::context::GpuCtx;
use crate::gpu::dtype::WeightDtype;
use crate::gpu::train::{BackwardMetrics, BackwardOpts, StepMetrics};
use crate::gpu::train_state::TrainerState;
use crate::mamba1::gpu::trainer::MambaGpuTrainer;
use crate::mamba2::gpu::Mamba2GpuTrainer;
use crate::mamba3_mimo::gpu::Mamba3MimoGpuTrainer;
use crate::mamba3_siso::gpu::trainer::Mamba3GpuTrainer;

/// What the GPU trainer of every family does.
///
/// ```rust,no_run
/// use mamba_rs::{GpuOptions, GpuTrainer, Mamba2Config, Mamba2GpuTrainer, Mamba2Weights};
/// use mamba_rs::TrainSessionCfg;
///
/// fn train(t: &mut impl GpuTrainer, input: &[f32], target: &[f32]) -> f32 {
///     let mut out = vec![0.0f32; t.batch() * t.seq_len() * t.d_model()];
///     t.forward(input, &mut out).unwrap();
///     let n = out.len() as f32;
///     let d: Vec<f32> = out.iter().zip(target).map(|(o, y)| 2.0 * (o - y) / n).collect();
///     t.backward_step(&d, Default::default()).unwrap();
///     out.iter().zip(target).map(|(o, y)| (o - y) * (o - y)).sum::<f32>() / n
/// }
///
/// let cfg = Mamba2Config::default();
/// let weights = Mamba2Weights::init(&cfg, 16, 7);
/// let session = TrainSessionCfg::new(16, 2, 64);
/// let mut trainer = Mamba2GpuTrainer::new(&weights, cfg, session, GpuOptions::default()).unwrap();
/// let input = vec![0.1f32; 2 * 64 * 16];
/// let target = vec![0.0f32; 2 * 64 * cfg.d_model];
/// let loss = train(&mut trainer, &input, &target);
/// ```
pub trait GpuTrainer {
    /// The training forward over one window: `input` is `[batch, seq_len,
    /// input_dim]`, and `temporal_out` (`[batch, seq_len, d_model]`)
    /// receives every position's final-norm rows, f32 in every storage
    /// precision. The recurrent state carries from the last window.
    fn forward(&mut self, input: &[f32], temporal_out: &mut [f32]) -> Result<(), String>;

    /// The backward of the last forward for `d_temporal` (d loss / d its
    /// rows), then the optimizer tail `opts` asks for: the global-norm
    /// clip, the skip of a spiking window, an accumulating call.
    fn backward_step(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
    ) -> Result<BackwardMetrics, String>;

    /// [`Self::backward_step`] that also writes d loss / d the window's
    /// input into `d_input` (`[batch, seq_len, input_dim]`), for a model
    /// that trains something upstream of the backbone. In f16 the gradient
    /// comes back unscaled.
    fn backward_step_input_grad(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        d_input: &mut [f32],
    ) -> Result<BackwardMetrics, String>;

    /// Forward, backward and the optimizer step in one call, with the
    /// step options (see [`GpuTrainer::set_step_opts`]); a captured graph
    /// replays the same step.
    fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String>;

    /// Set the options [`GpuTrainer::step`] applies: the control-channel
    /// clip, the global clip and the spike skip, which the eager step and
    /// a captured graph both apply. Accumulation is refused (the fused
    /// step applies its window), and so is a change under a captured
    /// graph, which baked the options in: `drop_graph` first.
    fn set_step_opts(&mut self, opts: BackwardOpts) -> Result<(), String>;

    /// The options [`GpuTrainer::step`] applies.
    fn step_opts(&self) -> BackwardOpts;

    /// One micro-batch of an accumulation window in one call: the forward
    /// of `input` and the backward of `d_temporal`, the gradient added to
    /// the window (the first call opens it); AdamW waits for
    /// [`GpuTrainer::apply_window`]. A captured graph replays it.
    fn accumulate(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<BackwardMetrics, String>;

    /// Apply the open accumulation window with the step options, as an
    /// applying `backward_step` with those options closes a window. A
    /// captured graph replays it.
    fn apply_window(&mut self) -> Result<StepMetrics, String>;

    /// Zero the recurrent state, the start of new sequences.
    fn reset_state(&mut self) -> Result<(), String>;

    /// Set the AdamW learning rate for the next steps.
    fn set_lr(&mut self, lr: f32) -> Result<(), String>;

    /// The AdamW learning rate.
    fn lr(&self) -> f32;

    /// Sequences per window.
    fn batch(&self) -> usize;

    /// Tokens per window.
    fn seq_len(&self) -> usize;

    /// Width of an input row.
    fn input_dim(&self) -> usize;

    /// Width of the model, the rows `forward` writes.
    fn d_model(&self) -> usize;

    /// Storage precision of the weights.
    fn dtype(&self) -> WeightDtype;

    /// The CUDA context: the GEMM mode and route live here.
    fn ctx(&self) -> &GpuCtx;

    /// The f16 loss scale and its growth counter; `None` in the other
    /// precisions.
    fn scaler_state(&self) -> Option<(f32, u32)>;

    /// Restore the f16 loss scale and its growth counter; no effect in the
    /// other precisions. A scale that is not finite and positive is an
    /// error.
    fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) -> Result<(), String>;

    /// The full optimizer state (the Adam moments, the step counter and the
    /// update settings) for a bit-continuous resume; the learning rate is
    /// the caller's schedule and not part of it.
    fn optimizer_state(&self) -> Result<AdamWStateBlob, String>;

    /// Upload a previously exported optimizer state; a blob of another
    /// parameterization is an error.
    fn load_optimizer_state(&mut self, blob: &AdamWStateBlob) -> Result<(), String>;

    /// Everything beside the weights a bit-continuous resume needs: the
    /// optimizer state, the f16 loss scale and the carried recurrence.
    /// [`TrainerState::save`] writes it next to the weight checkpoint.
    fn trainer_state(&self) -> Result<TrainerState, String>;

    /// Load a [`Self::trainer_state`] of a trainer of the same family and
    /// shape; another family's state is an error.
    fn load_trainer_state(&mut self, state: &TrainerState) -> Result<(), String>;

    /// Capture the training step as one CUDA graph, which later
    /// [`Self::step`] calls replay with the eager step's bits. Needs one
    /// eager step first.
    fn capture_graph(&mut self) -> Result<(), String>;

    /// Whether [`Self::step`] replays a captured graph.
    fn has_graph(&self) -> bool;

    /// Drop the captured graph: the next steps run eagerly and may be
    /// captured again.
    fn drop_graph(&mut self);

    /// Data-parallel applying backward: the local backward joins the
    /// window, the window's gradient is reduced to the ranks' mean through
    /// `dist`, then the optimizer tail runs. An accumulating call, or a
    /// world of one, is a plain [`Self::backward_step`].
    fn backward_step_dist(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        dist: &DistContext,
    ) -> Result<BackwardMetrics, String>;

    /// The open window's gradient on the host, for a reducer of the
    /// caller's own.
    fn download_grad_arena(&self) -> Result<Vec<f32>, String>;

    /// Overwrite the open window's gradient with `values`, a reduced arena
    /// of the same length.
    fn upload_grad_arena(&mut self, values: &[f32]) -> Result<(), String>;

    /// Only the optimizer tail over the open window's gradient: the
    /// control-channel clip, the global clip, the skip of a spiking window
    /// and the AdamW step. After accumulating calls it gives the bits of an
    /// applying [`Self::backward_step`].
    fn apply_step_full(
        &mut self,
        clip_max_norm: Option<f32>,
        step_skip_above: Option<f32>,
        control_clip_max_norm: Option<f32>,
    ) -> Result<BackwardMetrics, String>;
}

// Each family's trainer has these methods of its own (inherent methods win
// method resolution), so the trait methods hand straight to them.
macro_rules! gpu_trainer_by_inherent_methods {
    ($($ty:ty),+ $(,)?) => {$(
        impl GpuTrainer for $ty {
            fn forward(&mut self, input: &[f32], temporal_out: &mut [f32]) -> Result<(), String> {
                <$ty>::forward(self, input, temporal_out)
            }
            fn backward_step(
                &mut self,
                d_temporal: &[f32],
                opts: BackwardOpts,
            ) -> Result<BackwardMetrics, String> {
                <$ty>::backward_step(self, d_temporal, opts)
            }
            fn backward_step_input_grad(
                &mut self,
                d_temporal: &[f32],
                opts: BackwardOpts,
                d_input: &mut [f32],
            ) -> Result<BackwardMetrics, String> {
                <$ty>::backward_step_input_grad(self, d_temporal, opts, d_input)
            }
            fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
                <$ty>::step(self, input, d_temporal)
            }
            fn set_step_opts(&mut self, opts: BackwardOpts) -> Result<(), String> {
                <$ty>::set_step_opts(self, opts)
            }
            fn step_opts(&self) -> BackwardOpts {
                <$ty>::step_opts(self)
            }
            fn accumulate(
                &mut self,
                input: &[f32],
                d_temporal: &[f32],
            ) -> Result<BackwardMetrics, String> {
                <$ty>::accumulate(self, input, d_temporal)
            }
            fn apply_window(&mut self) -> Result<StepMetrics, String> {
                <$ty>::apply_window(self)
            }
            fn reset_state(&mut self) -> Result<(), String> {
                <$ty>::reset_state(self)
            }
            fn set_lr(&mut self, lr: f32) -> Result<(), String> {
                <$ty>::set_lr(self, lr)
            }
            fn lr(&self) -> f32 {
                <$ty>::lr(self)
            }
            fn batch(&self) -> usize {
                <$ty>::batch(self)
            }
            fn seq_len(&self) -> usize {
                <$ty>::seq_len(self)
            }
            fn input_dim(&self) -> usize {
                <$ty>::input_dim(self)
            }
            fn d_model(&self) -> usize {
                <$ty>::d_model(self)
            }
            fn dtype(&self) -> WeightDtype {
                <$ty>::dtype(self)
            }
            fn ctx(&self) -> &GpuCtx {
                <$ty>::ctx(self)
            }
            fn scaler_state(&self) -> Option<(f32, u32)> {
                <$ty>::scaler_state(self)
            }
            fn load_scaler_state(
                &mut self,
                scale: f32,
                growth_tracker: u32,
            ) -> Result<(), String> {
                <$ty>::load_scaler_state(self, scale, growth_tracker)
            }
            fn optimizer_state(&self) -> Result<AdamWStateBlob, String> {
                <$ty>::optimizer_state(self)
            }
            fn load_optimizer_state(&mut self, blob: &AdamWStateBlob) -> Result<(), String> {
                <$ty>::load_optimizer_state(self, blob)
            }
            fn trainer_state(&self) -> Result<TrainerState, String> {
                <$ty>::trainer_state(self)
            }
            fn load_trainer_state(&mut self, state: &TrainerState) -> Result<(), String> {
                <$ty>::load_trainer_state(self, state)
            }
            fn capture_graph(&mut self) -> Result<(), String> {
                <$ty>::capture_graph(self)
            }
            fn has_graph(&self) -> bool {
                <$ty>::has_graph(self)
            }
            fn drop_graph(&mut self) {
                <$ty>::drop_graph(self)
            }
            fn backward_step_dist(
                &mut self,
                d_temporal: &[f32],
                opts: BackwardOpts,
                dist: &DistContext,
            ) -> Result<BackwardMetrics, String> {
                <$ty>::backward_step_dist(self, d_temporal, opts, dist)
            }
            fn download_grad_arena(&self) -> Result<Vec<f32>, String> {
                <$ty>::download_grad_arena(self)
            }
            fn upload_grad_arena(&mut self, values: &[f32]) -> Result<(), String> {
                <$ty>::upload_grad_arena(self, values)
            }
            fn apply_step_full(
                &mut self,
                clip_max_norm: Option<f32>,
                step_skip_above: Option<f32>,
                control_clip_max_norm: Option<f32>,
            ) -> Result<BackwardMetrics, String> {
                <$ty>::apply_step_full(self, clip_max_norm, step_skip_above, control_clip_max_norm)
            }
        }
    )+};
}

gpu_trainer_by_inherent_methods!(
    MambaGpuTrainer,
    Mamba2GpuTrainer,
    Mamba3GpuTrainer,
    Mamba3MimoGpuTrainer,
);

/// A GPU trainer of any family, chosen at run time, behind the shared
/// surface.
pub enum AnyGpuTrainer {
    M1(Box<MambaGpuTrainer>),
    M2(Box<Mamba2GpuTrainer>),
    M3(Box<Mamba3GpuTrainer>),
    M3Mimo(Box<Mamba3MimoGpuTrainer>),
}

impl AnyGpuTrainer {
    /// The family of the trainer held.
    pub fn family(&self) -> ModelFamily {
        match self {
            Self::M1(_) => ModelFamily::Mamba1,
            Self::M2(_) => ModelFamily::Mamba2,
            Self::M3(_) => ModelFamily::Mamba3,
            Self::M3Mimo(_) => ModelFamily::Mamba3Mimo,
        }
    }

    fn inner(&self) -> &dyn GpuTrainer {
        match self {
            Self::M1(t) => t.as_ref(),
            Self::M2(t) => t.as_ref(),
            Self::M3(t) => t.as_ref(),
            Self::M3Mimo(t) => t.as_ref(),
        }
    }

    fn inner_mut(&mut self) -> &mut dyn GpuTrainer {
        match self {
            Self::M1(t) => t.as_mut(),
            Self::M2(t) => t.as_mut(),
            Self::M3(t) => t.as_mut(),
            Self::M3Mimo(t) => t.as_mut(),
        }
    }
}

impl From<MambaGpuTrainer> for AnyGpuTrainer {
    fn from(t: MambaGpuTrainer) -> Self {
        Self::M1(Box::new(t))
    }
}

impl From<Mamba2GpuTrainer> for AnyGpuTrainer {
    fn from(t: Mamba2GpuTrainer) -> Self {
        Self::M2(Box::new(t))
    }
}

impl From<Mamba3GpuTrainer> for AnyGpuTrainer {
    fn from(t: Mamba3GpuTrainer) -> Self {
        Self::M3(Box::new(t))
    }
}

impl From<Mamba3MimoGpuTrainer> for AnyGpuTrainer {
    fn from(t: Mamba3MimoGpuTrainer) -> Self {
        Self::M3Mimo(Box::new(t))
    }
}

impl GpuTrainer for AnyGpuTrainer {
    fn forward(&mut self, input: &[f32], temporal_out: &mut [f32]) -> Result<(), String> {
        self.inner_mut().forward(input, temporal_out)
    }
    fn backward_step(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
    ) -> Result<BackwardMetrics, String> {
        self.inner_mut().backward_step(d_temporal, opts)
    }
    fn backward_step_input_grad(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        d_input: &mut [f32],
    ) -> Result<BackwardMetrics, String> {
        self.inner_mut()
            .backward_step_input_grad(d_temporal, opts, d_input)
    }
    fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        self.inner_mut().step(input, d_temporal)
    }
    fn set_step_opts(&mut self, opts: BackwardOpts) -> Result<(), String> {
        self.inner_mut().set_step_opts(opts)
    }
    fn step_opts(&self) -> BackwardOpts {
        self.inner().step_opts()
    }
    fn accumulate(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<BackwardMetrics, String> {
        self.inner_mut().accumulate(input, d_temporal)
    }
    fn apply_window(&mut self) -> Result<StepMetrics, String> {
        self.inner_mut().apply_window()
    }
    fn reset_state(&mut self) -> Result<(), String> {
        self.inner_mut().reset_state()
    }
    fn set_lr(&mut self, lr: f32) -> Result<(), String> {
        self.inner_mut().set_lr(lr)
    }
    fn lr(&self) -> f32 {
        self.inner().lr()
    }
    fn batch(&self) -> usize {
        self.inner().batch()
    }
    fn seq_len(&self) -> usize {
        self.inner().seq_len()
    }
    fn input_dim(&self) -> usize {
        self.inner().input_dim()
    }
    fn d_model(&self) -> usize {
        self.inner().d_model()
    }
    fn dtype(&self) -> WeightDtype {
        self.inner().dtype()
    }
    fn ctx(&self) -> &GpuCtx {
        self.inner().ctx()
    }
    fn scaler_state(&self) -> Option<(f32, u32)> {
        self.inner().scaler_state()
    }
    fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) -> Result<(), String> {
        self.inner_mut().load_scaler_state(scale, growth_tracker)
    }
    fn optimizer_state(&self) -> Result<AdamWStateBlob, String> {
        self.inner().optimizer_state()
    }
    fn load_optimizer_state(&mut self, blob: &AdamWStateBlob) -> Result<(), String> {
        self.inner_mut().load_optimizer_state(blob)
    }
    fn trainer_state(&self) -> Result<TrainerState, String> {
        self.inner().trainer_state()
    }
    fn load_trainer_state(&mut self, state: &TrainerState) -> Result<(), String> {
        self.inner_mut().load_trainer_state(state)
    }
    fn capture_graph(&mut self) -> Result<(), String> {
        self.inner_mut().capture_graph()
    }
    fn has_graph(&self) -> bool {
        self.inner().has_graph()
    }
    fn drop_graph(&mut self) {
        self.inner_mut().drop_graph()
    }
    fn backward_step_dist(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        dist: &DistContext,
    ) -> Result<BackwardMetrics, String> {
        self.inner_mut().backward_step_dist(d_temporal, opts, dist)
    }
    fn download_grad_arena(&self) -> Result<Vec<f32>, String> {
        self.inner().download_grad_arena()
    }
    fn upload_grad_arena(&mut self, values: &[f32]) -> Result<(), String> {
        self.inner_mut().upload_grad_arena(values)
    }
    fn apply_step_full(
        &mut self,
        clip_max_norm: Option<f32>,
        step_skip_above: Option<f32>,
        control_clip_max_norm: Option<f32>,
    ) -> Result<BackwardMetrics, String> {
        self.inner_mut()
            .apply_step_full(clip_max_norm, step_skip_above, control_clip_max_norm)
    }
}
