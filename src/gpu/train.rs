//! What every family's GPU trainer takes and reports: the session it is
//! built for, the options of a backward step, and the metrics of a step.

/// The session every family's GPU trainer is built for: the window shape,
/// fixed at construction, and the AdamW settings.
#[derive(Clone, Copy, Debug)]
pub struct TrainSessionCfg {
    /// Input feature dimension fed to the backbone.
    pub input_dim: usize,
    /// Batch dimension fixed at construction; CUDA Graph capture binds
    /// device pointers for this exact `batch * seq_len` shape.
    pub batch: usize,
    /// Sequence length fixed at construction.
    pub seq_len: usize,
    /// AdamW learning rate.
    pub lr: f32,
    /// AdamW decoupled weight decay.
    pub weight_decay: f32,
}

impl TrainSessionCfg {
    /// Session settings with the default optimizer: AdamW at learning
    /// rate 1e-3 and weight decay 1e-2. Struct update syntax changes them:
    /// `TrainSessionCfg { lr: 3e-4, ..TrainSessionCfg::new(d, b, t) }`.
    pub fn new(input_dim: usize, batch: usize, seq_len: usize) -> Self {
        Self {
            input_dim,
            batch,
            seq_len,
            lr: 1e-3,
            weight_decay: 1e-2,
        }
    }
}

/// Per-step metrics returned by [`GpuTrainer::step`](crate::module::GpuTrainer::step).
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

/// Options for [`GpuTrainer::backward_step`](crate::module::GpuTrainer::backward_step).
///
/// `#[non_exhaustive]` + `with_*` builders: cross-crate struct literals are
/// blocked on purpose so future fields stay semver-minor. The derived
/// `Default` is the safe fused-step behavior (zero the arena, run backward,
/// run the full optimizer tail).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default)]
pub struct BackwardOpts {
    /// `Some(c)`: after backward (and, for f16, after unscale) scale the
    /// gradients by `min(1, c / (||g||_2 + 1e-6))` before AdamW —
    /// `torch.nn.utils.clip_grad_norm_` semantics. The norm is computed by
    /// a deterministic fixed-order f64 reduction and returned in
    /// [`BackwardMetrics::grad_norm`]. A non-finite norm is an error.
    /// Incompatible with `accumulate_only` (the norm is only defined over
    /// the complete accumulated gradient — clip on the applying call).
    pub clip_max_norm: Option<f32>,
    /// `true`: accumulate only — the gradient arena is NOT zeroed by the
    /// next `backward_step`, Adam does not advance, and AdamW /
    /// master→compute sync / the `a_neg` refresh are all skipped. Loss
    /// averaging stays caller-side (scale `d_temporal` by `1/n_micro`).
    /// Unsupported for f16 (the loss-scale freeze window across
    /// micro-batches has no defined semantics).
    pub accumulate_only: bool,
    /// `Some(t)`: on the applying call, if the PRE-clip global gradient
    /// norm exceeds `t`, SKIP the optimizer step entirely and discard
    /// the window (the arena re-zeroes on the next backward, Adam does
    /// not advance). A rare pathological window then costs one skipped
    /// update instead of a poisoned Adam second moment and a destroyed
    /// model - the small-batch analogue of the averaging a huge batch
    /// provides. Without `clip_max_norm` the same fold measures the norm
    /// and leaves the gradients as they are.
    /// [`BackwardMetrics::optimizer_stepped`] reports the skip.
    pub step_skip_above: Option<f32>,
    /// `Some(c)`: before the global clip, clip the Mamba-3 CONTROL
    /// channels (the dd_dt/dd_A/trap/angle columns of every layer's
    /// in_proj gradient plus dt_bias) to their own max norm `c`. Those
    /// 80-of-1648 columns carry the only gradients whose magnitude
    /// scales with the sequence length; without a separate bound one
    /// resonant page rescales the ENTIRE arena through the global clip
    /// and starves the representational columns. Honored by the M3 f32
    /// and mixed lanes; the M1 lanes ignore it (their dt path has no
    /// such route).
    /// Read by the Mamba-3 trainer only: Mamba-1 has no control region
    /// and leaves this unset regardless of what a caller passes.
    pub control_clip_max_norm: Option<f32>,
}

impl BackwardOpts {
    /// Request global-norm gradient clipping at `c`.
    pub fn with_clip_max_norm(mut self, c: f32) -> Self {
        self.clip_max_norm = Some(c);
        self
    }

    /// Toggle accumulate-only mode (see the field docs).
    pub fn with_accumulate_only(mut self, on: bool) -> Self {
        self.accumulate_only = on;
        self
    }

    /// Skip the optimizer step (discarding the window) when the pre-clip
    /// global norm exceeds `t`. See [`BackwardOpts::step_skip_above`].
    pub fn with_step_skip_above(mut self, t: f32) -> Self {
        self.step_skip_above = Some(t);
        self
    }

    /// Clip the Mamba-3 control-channel gradients to their own max norm
    /// before the global clip. See
    /// [`BackwardOpts::control_clip_max_norm`].
    pub fn with_control_clip_max_norm(mut self, c: f32) -> Self {
        self.control_clip_max_norm = Some(c);
        self
    }
}

/// Metrics returned by [`GpuTrainer::backward_step`](crate::module::GpuTrainer::backward_step).
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BackwardMetrics {
    /// Adam step counter AFTER this call; unchanged on an accumulate-only
    /// call and on an f16 overflow-skipped step.
    pub step: u64,
    /// Whether AdamW actually ran (false on accumulate-only / f16 overflow).
    pub optimizer_stepped: bool,
    /// Pre-clip, post-unscale global gradient L2 norm; `Some` iff clipping
    /// was requested. Deliberately never added to [`StepMetrics`] — the
    /// fused step must not grow a hidden device sync.
    pub grad_norm: Option<f32>,
    /// `Some(scale)` when the f16 loss scaler is active.
    pub loss_scale: Option<f32>,
    /// `Some(true)` when the f16 scaler detected inf/nan and skipped AdamW.
    pub overflow_skipped: Option<bool>,
}
