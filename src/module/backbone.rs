use crate::inference::{
    MambaPackedLinears, MambaStepScratch, mamba_step, mamba_step_no_proj,
    mamba_step_no_proj_packed, mamba_step_packed,
};
use crate::mamba_ssm::config::MambaConfig;
use crate::mamba_ssm::state::MambaState;
use crate::mamba_ssm::weights::{MambaLayerWeights, MambaWeights};

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
    /// The projections packed for decode by [`Self::pack_for_decode`]: a
    /// copy of `weights`, dropped by every mutable access to them so a step
    /// never reads stale panels.
    packed: Option<MambaPackedLinears>,
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
            packed: None,
        }
    }

    /// Wrap loaded weights for input rows `input_dim` wide, after checking
    /// every tensor against `cfg`. Weights without an input projection (an
    /// empty `input_proj_w`, as the checkpoint loader gives) read rows
    /// `d_model` wide unchanged, so `input_dim` is then `d_model`.
    ///
    /// Recomputes `a_neg = -exp(a_log)` for every layer: inference reads
    /// `a_neg` alone, and loaded weights may carry a stale one.
    pub fn new(weights: MambaWeights, cfg: MambaConfig, input_dim: usize) -> Result<Self, String> {
        if weights.input_proj_w.is_empty() {
            if input_dim != cfg.d_model {
                return Err(format!(
                    "weights without an input projection read rows d_model ({}) wide, \
                     not {input_dim}",
                    cfg.d_model
                ));
            }
            return Self::without_projection(cfg, weights);
        }
        Self::with_projection(cfg, weights, input_dim)
    }

    #[deprecated(
        since = "0.8.0",
        note = "use `MambaBackbone::new(weights, cfg, input_dim)`"
    )]
    pub fn from_weights(cfg: MambaConfig, weights: MambaWeights) -> Result<Self, String> {
        let input_dim = weights.input_proj_w.len() / cfg.d_model;
        Self::with_projection(cfg, weights, input_dim)
    }

    fn with_projection(
        cfg: MambaConfig,
        mut weights: MambaWeights,
        input_dim: usize,
    ) -> Result<Self, String> {
        weights.validate(&cfg, input_dim)?;
        for lw in &mut weights.layers {
            lw.compute_a_neg();
        }
        Ok(Self {
            weights,
            cfg,
            input_dim,
            identity_proj: false,
            packed: None,
        })
    }

    #[cfg(feature = "hf")]
    #[deprecated(
        since = "0.8.0",
        note = "use `MambaBackbone::new(weights, cfg, cfg.d_model)`"
    )]
    pub fn from_weights_no_proj(cfg: MambaConfig, weights: MambaWeights) -> Result<Self, String> {
        Self::without_projection(cfg, weights)
    }

    /// Weights without an input projection. The checks of `validate()`
    /// except the projection's, which it requires, and the `a_neg`
    /// refresh the SSM reads.
    fn without_projection(cfg: MambaConfig, mut weights: MambaWeights) -> Result<Self, String> {
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
            packed: None,
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

    /// Mutable weight access (for optimizer updates). Drops the packed
    /// projections; pack again with [`Self::pack_for_decode`].
    pub fn weights_mut(&mut self) -> &mut MambaWeights {
        self.packed = None;
        &mut self.weights
    }

    /// Read-only access to a specific layer's weights.
    pub fn layer(&self, index: usize) -> &MambaLayerWeights {
        &self.weights.layers[index]
    }

    /// Mutable access to a specific layer's weights. Drops the packed
    /// projections; pack again with [`Self::pack_for_decode`].
    pub fn layer_mut(&mut self, index: usize) -> &mut MambaLayerWeights {
        self.packed = None;
        &mut self.weights.layers[index]
    }

    /// Pack the projections in panels for the single-step decode
    /// ([`Self::forward_step`] and [`Self::forward_sequence`]): the same
    /// outputs as the row-major product without a BLAS, faster on large
    /// layers, at the memory of a second copy of the projections. A mutable
    /// access to the weights drops the copy.
    pub fn pack_for_decode(&mut self) {
        self.packed = Some(MambaPackedLinears::pack(
            &self.weights,
            &self.cfg,
            self.input_dim,
        ));
    }

    /// Whether the projections are packed for decode.
    pub fn is_packed(&self) -> bool {
        self.packed.is_some()
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

    /// Width of an output row.
    pub fn d_model(&self) -> usize {
        self.cfg.d_model
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
        match (&self.packed, self.identity_proj) {
            (Some(p), true) => mamba_step_no_proj_packed(
                input,
                output,
                (&self.weights, p),
                &mut state.layers,
                scratch,
                &self.cfg,
            ),
            (Some(p), false) => mamba_step_packed(
                input,
                output,
                (&self.weights, p),
                &mut state.layers,
                scratch,
                &self.cfg,
                self.input_dim,
            ),
            (None, true) => mamba_step_no_proj(
                input,
                output,
                &self.weights,
                &mut state.layers,
                scratch,
                &self.cfg,
            ),
            (None, false) => mamba_step(
                input,
                output,
                &self.weights,
                &mut state.layers,
                scratch,
                &self.cfg,
                self.input_dim,
            ),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn det(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0
            })
            .collect()
    }

    /// Packed projections give the row-major step's bits over a chain of
    /// steps (without a BLAS), and a mutable access to the weights drops
    /// them, so a step never reads panels of weights that changed.
    #[test]
    fn packed_decode_keeps_the_bits_and_drops_on_mutation() {
        let cfg = MambaConfig {
            d_model: 96,
            n_layers: 2,
            ..MambaConfig::default()
        };
        let input_dim = 40;
        let plain = MambaBackbone::init(cfg, input_dim, 7);
        let mut packed = MambaBackbone::init(cfg, input_dim, 7);
        packed.pack_for_decode();
        assert!(packed.is_packed());
        let (mut s0, mut s1) = (plain.alloc_state(), packed.alloc_state());
        let (mut c0, mut c1) = (plain.alloc_scratch(), packed.alloc_scratch());
        let (mut o0, mut o1) = (vec![0f32; cfg.d_model], vec![0f32; cfg.d_model]);
        for t in 0..12u64 {
            let x = det(input_dim, 100 + t);
            plain.forward_step(&x, &mut o0, &mut s0, &mut c0);
            packed.forward_step(&x, &mut o1, &mut s1, &mut c1);
            if cfg!(not(any(
                all(feature = "accelerate", target_os = "macos"),
                feature = "gemm-blas"
            ))) {
                assert_eq!(o0, o1, "step {t}");
            }
        }
        packed.weights_mut().layers[0].in_proj_w[0] += 1.0;
        assert!(!packed.is_packed());
        packed.pack_for_decode();
        packed.layer_mut(1).out_proj_w[0] += 1.0;
        assert!(!packed.is_packed());
    }
}
