//! What every family's trainer, on the CPU and on the GPU, takes and
//! reports: the session it is built for, the options of a backward step,
//! and the metrics of a step.

/// The session every family's trainer is built for: the window shape,
/// fixed at construction, and the AdamW settings.
#[derive(Clone, Copy, Debug)]
pub struct TrainSessionCfg {
    /// Input feature dimension fed to the backbone.
    pub input_dim: usize,
    /// Batch dimension fixed at construction; a GPU trainer's CUDA graph
    /// binds device pointers for this exact `batch * seq_len` shape.
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

/// Per-step metrics a trainer's `step` returns.
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
    /// Whether AdamW took the step: false after an f16 overflow or when
    /// the spike skip of the trainer's step options discarded the window.
    pub optimizer_stepped: bool,
    /// The pre-clip global gradient norm when the step options measure it
    /// (a global clip or a spike skip), else `None`.
    pub grad_norm: Option<f32>,
}

impl StepMetrics {
    /// Convenience constructor for paths without loss-scaler activity.
    pub fn plain(step: u64, graph_replayed: bool) -> Self {
        Self {
            step,
            graph_replayed,
            loss_scale: None,
            overflow_skipped: None,
            optimizer_stepped: true,
            grad_norm: None,
        }
    }

    /// The fused step's metrics from those of its applying backward.
    pub(crate) fn from_backward(m: BackwardMetrics, graph_replayed: bool) -> Self {
        Self {
            step: m.step,
            graph_replayed,
            loss_scale: m.loss_scale,
            overflow_skipped: m.overflow_skipped,
            optimizer_stepped: m.optimizer_stepped,
            grad_norm: m.grad_norm,
        }
    }
}

/// Options of a trainer's `backward_step`.
///
/// `#[non_exhaustive]` + `with_*` builders: cross-crate struct literals are
/// blocked on purpose so future fields stay semver-minor. The derived
/// `Default` is the safe fused-step behavior (zero the arena, run backward,
/// run the full optimizer tail).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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
    /// In f16 the loss scale holds for the whole window: every micro-batch
    /// adds gradients of the same scale, and the applying call checks the
    /// window's sum for overflow, skipping the whole window and backing the
    /// scale off when any micro-batch overflowed.
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
    /// `Some(c)`: before the global clip, clip the CONTROL channels to
    /// their own max norm `c`: the gradients of the step size and the
    /// decay, the only ones whose magnitude scales with the sequence
    /// length. Without a separate bound one resonant page rescales the
    /// ENTIRE arena through the global clip and starves the
    /// representational columns. For Mamba-3 SISO and MIMO they are the
    /// dd_dt/dd_A/trap/angle columns of every layer's in_proj gradient
    /// plus dt_bias; for Mamba-1 the step-size columns of x_proj, dt_proj
    /// and A_log; for Mamba-2 the dt columns of in_proj (and their in_proj
    /// bias entries where the model has the bias), dt_bias and A_log.
    /// Every family honors it on the applying call; on an accumulating
    /// call it is an error, as the global clip is.
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

    /// Clip the control-channel gradients to their own max norm before
    /// the global clip. See
    /// [`BackwardOpts::control_clip_max_norm`].
    pub fn with_control_clip_max_norm(mut self, c: f32) -> Self {
        self.control_clip_max_norm = Some(c);
        self
    }
}

/// Metrics a trainer's `backward_step` returns.
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
