use crate::config::MambaConfig;
use crate::inference::{MambaStepScratch, mamba_step, mamba_step_no_proj};
use crate::state::MambaState;
use crate::weights::{MambaLayerWeights, MambaWeights};

/// Complete Mamba backbone: input_proj -> N layers -> norm_f.
///
/// Owns all weights. Provides both single-step recurrent inference
/// and access to raw weights for training integration.
///
/// ```rust
/// use mamba_rs::module::MambaBackbone;
/// use mamba_rs::MambaConfig;
///
/// let cfg = MambaConfig::default();
/// let backbone = MambaBackbone::init(cfg, 128, 42);
///
/// let mut state = backbone.alloc_state();
/// let mut scratch = backbone.alloc_scratch();
/// let mut output = vec![0.0f32; backbone.config().d_model];
///
/// let input = vec![0.1f32; 128];
/// backbone.forward_step(&input, &mut output, &mut state, &mut scratch);
/// ```
pub struct MambaBackbone {
    weights: MambaWeights,
    cfg: MambaConfig,
    input_dim: usize,
    identity_proj: bool,
}

impl MambaBackbone {
    /// Create a backbone with Mamba-specific weight initialization.
    ///
    /// Uses Kaiming uniform for projections, log-space init for A,
    /// inverse-softplus init for dt_proj bias (Gu & Dao, Section 3.5).
    pub fn init(cfg: MambaConfig, input_dim: usize, seed: u64) -> Self {
        let weights = MambaWeights::init(&cfg, input_dim, seed);
        Self {
            weights,
            cfg,
            input_dim,
            identity_proj: false,
        }
    }

    /// Create a backbone from pre-loaded weights.
    ///
    /// Validates dimensions against config. Returns `Err` on mismatch.
    /// Recomputes `a_neg = -exp(a_log)` for every layer — inference reads
    /// `a_neg` exclusively and `from_weights` must accept weight bundles
    /// whose `a_neg` field may be stale (e.g. fresh-constructed, post
    /// serialize/deserialize, or loaded from a non-HF source).
    pub fn from_weights(cfg: MambaConfig, mut weights: MambaWeights) -> Result<Self, String> {
        let input_dim = weights.input_proj_w.len() / cfg.d_model;
        weights.validate(&cfg, input_dim)?;
        for lw in &mut weights.layers {
            lw.compute_a_neg();
        }
        Ok(Self {
            weights,
            cfg,
            input_dim,
            identity_proj: false,
        })
    }

    /// Create a backbone from HF weights that have no input_proj.
    ///
    /// Skips the standard `validate()` (which rejects empty `input_proj_w`).
    /// Performs targeted layer-weight validation and calls `compute_a_neg()`
    /// on every layer (critical — SSM reads `a_neg`, never `a_log` at inference time).
    #[cfg(feature = "hf")]
    pub fn from_weights_no_proj(
        cfg: MambaConfig,
        mut weights: MambaWeights,
    ) -> Result<Self, String> {
        if weights.layers.len() != cfg.n_layers {
            return Err(format!(
                "expected {} layers, got {}",
                cfg.n_layers,
                weights.layers.len()
            ));
        }
        let d = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let dc = cfg.d_conv;
        let dr = cfg.dt_rank();
        let xd = cfg.xdbl_dim();
        for (i, lw) in weights.layers.iter().enumerate() {
            let check = |name: &str, actual: usize, expected: usize| -> Result<(), String> {
                if actual != expected {
                    return Err(format!(
                        "layer[{i}].{name}: expected {expected}, got {actual}"
                    ));
                }
                Ok(())
            };
            check("norm_weight", lw.norm_weight.len(), d)?;
            check("in_proj_w", lw.in_proj_w.len(), d * 2 * di)?;
            check("conv1d_weight", lw.conv1d_weight.len(), di * dc)?;
            check("conv1d_bias", lw.conv1d_bias.len(), di)?;
            check("x_proj_w", lw.x_proj_w.len(), di * xd)?;
            check("dt_proj_w", lw.dt_proj_w.len(), dr * di)?;
            check("dt_proj_b", lw.dt_proj_b.len(), di)?;
            check("a_log", lw.a_log.len(), di * ds)?;
            check("d_param", lw.d_param.len(), di)?;
            check("out_proj_w", lw.out_proj_w.len(), di * d)?;
        }
        if weights.norm_f_weight.len() != d {
            return Err(format!(
                "norm_f_weight: expected {d}, got {}",
                weights.norm_f_weight.len()
            ));
        }
        for lw in &mut weights.layers {
            lw.compute_a_neg();
        }
        Ok(Self {
            weights,
            cfg,
            input_dim: d,
            identity_proj: true,
        })
    }

    /// Extract owned weights (consuming self).
    pub fn into_weights(self) -> MambaWeights {
        self.weights
    }

    /// Read-only weight access.
    pub fn weights(&self) -> &MambaWeights {
        &self.weights
    }

    /// Mutable weight access (for optimizer updates).
    pub fn weights_mut(&mut self) -> &mut MambaWeights {
        &mut self.weights
    }

    /// Read-only access to a specific layer's weights.
    pub fn layer(&self, index: usize) -> &MambaLayerWeights {
        &self.weights.layers[index]
    }

    /// Mutable access to a specific layer's weights.
    pub fn layer_mut(&mut self, index: usize) -> &mut MambaLayerWeights {
        &mut self.weights.layers[index]
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        self.weights.param_count(self.input_dim, &self.cfg)
    }

    /// The config this backbone was built with.
    pub fn config(&self) -> &MambaConfig {
        &self.cfg
    }

    /// External input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Single-step recurrent forward through the full backbone.
    ///
    /// `input_proj(input) -> N x layer_step -> norm_f -> output`
    ///
    /// Zero allocations per call. Delegates to [`mamba_step`].
    pub fn forward_step(
        &self,
        input: &[f32],
        output: &mut [f32],
        state: &mut MambaState,
        scratch: &mut MambaStepScratch,
    ) {
        if self.identity_proj {
            mamba_step_no_proj(
                input,
                output,
                &self.weights,
                &mut state.layers,
                scratch,
                &self.cfg,
            );
        } else {
            mamba_step(
                input,
                output,
                &self.weights,
                &mut state.layers,
                scratch,
                &self.cfg,
                self.input_dim,
            );
        }
    }

    /// Run T inference steps sequentially, collecting all outputs.
    ///
    /// `inputs`: `[T * input_dim]` — T sequential inputs.
    /// `outputs`: `[T * d_model]` — T sequential outputs (written in-place).
    /// State carries across all T steps (warm-up, offline eval, etc.).
    pub fn forward_sequence(
        &self,
        inputs: &[f32],
        outputs: &mut [f32],
        state: &mut MambaState,
        scratch: &mut MambaStepScratch,
        seq_len: usize,
    ) {
        let dm = self.cfg.d_model;
        debug_assert_eq!(inputs.len(), seq_len * self.input_dim);
        debug_assert_eq!(outputs.len(), seq_len * dm);
        for t in 0..seq_len {
            let inp = &inputs[t * self.input_dim..(t + 1) * self.input_dim];
            let out = &mut outputs[t * dm..(t + 1) * dm];
            self.forward_step(inp, out, state, scratch);
        }
    }

    /// Batched single-step forward through the backbone.
    ///
    /// Processes B independent samples with the same weights.
    /// `inputs`: `[B * input_dim]`, `outputs`: `[B * d_model]`.
    pub fn forward_step_batch(
        &self,
        inputs: &[f32],
        outputs: &mut [f32],
        states: &mut [MambaState],
        scratches: &mut [MambaStepScratch],
    ) {
        crate::inference::mamba_step_batch(
            inputs,
            outputs,
            &self.weights,
            states,
            scratches,
            &self.cfg,
            self.input_dim,
        );
    }

    /// Allocate zeroed recurrent state matching this backbone.
    pub fn alloc_state(&self) -> MambaState {
        MambaState::zeros(
            self.cfg.n_layers,
            self.cfg.d_inner(),
            self.cfg.d_state,
            self.cfg.d_conv,
        )
    }

    /// Allocate inference scratch buffers matching this backbone.
    pub fn alloc_scratch(&self) -> MambaStepScratch {
        MambaStepScratch::new(&self.cfg)
    }

    /// Allocate a full-sequence prefill scratch for sequences of `seq_len`.
    pub fn alloc_prefill_scratch(
        &self,
        seq_len: usize,
    ) -> crate::mamba_ssm::cpu::prefill::PrefillScratch {
        let dims = crate::ops::dims::MambaDims::from_config(&self.cfg, seq_len, self.input_dim);
        crate::mamba_ssm::cpu::prefill::PrefillScratch::new(&dims)
    }

    /// Full-sequence prefill: run `seq_len` inputs through the batched
    /// SGEMM pipeline (orders of magnitude faster than the per-step
    /// [`Self::forward_sequence`] loop), writing the post-norm_f output at
    /// EVERY position and carrying the recurrent state so
    /// [`Self::forward_step`] continues seamlessly (prefill-then-decode).
    ///
    /// `inputs`: `[seq_len * input_dim]`; `outputs`: `[seq_len * d_model]`.
    pub fn forward_prefill(
        &self,
        inputs: &[f32],
        outputs: &mut [f32],
        state: &mut MambaState,
        scratch: &mut crate::mamba_ssm::cpu::prefill::PrefillScratch,
        seq_len: usize,
        mode: crate::mamba_ssm::cpu::prefill::PrefillMode,
    ) {
        let dims = crate::ops::dims::MambaDims::from_config(&self.cfg, seq_len, self.input_dim);
        crate::mamba_ssm::cpu::prefill::forward_mamba_backbone_prefill_mode(
            outputs,
            inputs,
            &self.weights,
            state,
            scratch,
            &dims,
            mode,
        );
    }
}
