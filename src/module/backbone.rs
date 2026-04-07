use crate::config::MambaConfig;
use crate::inference::{MambaStepScratch, mamba_step};
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
        }
    }

    /// Create a backbone from pre-loaded weights.
    ///
    /// Validates dimensions against config. Returns `Err` on mismatch.
    pub fn from_weights(cfg: MambaConfig, weights: MambaWeights) -> Result<Self, String> {
        let input_dim = weights.input_proj_w.len() / cfg.d_model;
        weights.validate(&cfg, input_dim)?;
        Ok(Self {
            weights,
            cfg,
            input_dim,
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
}
