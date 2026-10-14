//! The gated MLP that follows the mixer in every layer of models trained
//! with one, the official Mamba-3 checkpoints among them.
//!
//! Each layer adds two residual branches (mamba_ssm/modules/block.py):
//! `x += mixer(norm(x)); x += mlp(norm2(x))`. The MLP is the reference
//! `GatedMLP` (mamba_ssm/modules/mlp.py): one up projection whose first
//! half is the value and second half the gate, `value * silu(gate)`, and a
//! down projection, both without bias. Its hidden width is the configured
//! `d_intermediate` rounded up to a multiple of 128.

pub mod cpu;

use crate::config::BlockNorm;

/// The rounding the reference applies to the configured hidden width.
pub const HIDDEN_MULTIPLE: usize = 128;

/// The hidden width of a gated MLP configured with `d_intermediate`
/// (0 means the model has no MLP).
pub fn mlp_hidden(d_intermediate: usize) -> usize {
    d_intermediate.div_ceil(HIDDEN_MULTIPLE) * HIDDEN_MULTIPLE
}

/// One layer's gated MLP, in the `[n_in, n_out]` row-major layout every
/// projection here uses.
#[derive(Clone, Debug, PartialEq)]
pub struct GatedMlpLayerWeights {
    /// The pre-MLP norm's scale (`norm2`): `[d_model]`.
    pub norm_weight: Vec<f32>,
    /// The pre-MLP LayerNorm's bias: `[d_model]`, empty for RMSNorm.
    pub norm_bias: Vec<f32>,
    /// The up projection `[d_model, 2 * hidden]`: value columns first,
    /// then gate columns.
    pub fc1_w: Vec<f32>,
    /// The down projection `[hidden, d_model]`.
    pub fc2_w: Vec<f32>,
}

impl GatedMlpLayerWeights {
    /// Zeroed projections and a unit norm of kind `norm` for a `d_model`
    /// by `hidden` MLP.
    pub fn zeros(d_model: usize, hidden: usize, norm: BlockNorm) -> Self {
        Self {
            norm_weight: vec![1.0; d_model],
            norm_bias: vec![0.0; norm.bias_len(d_model)],
            fc1_w: vec![0.0; d_model * 2 * hidden],
            fc2_w: vec![0.0; hidden * d_model],
        }
    }

    /// The hidden width these weights carry.
    pub fn hidden(&self) -> usize {
        if self.norm_weight.is_empty() {
            0
        } else {
            self.fc2_w.len() / self.norm_weight.len()
        }
    }

    /// Check the tensor sizes against a `d_model` by `hidden` MLP with a
    /// norm of kind `norm`.
    pub fn validate(
        &self,
        d_model: usize,
        hidden: usize,
        norm: BlockNorm,
        label: &str,
    ) -> Result<(), String> {
        for (name, actual, expected) in [
            ("norm2", self.norm_weight.len(), d_model),
            ("norm2 bias", self.norm_bias.len(), norm.bias_len(d_model)),
            ("fc1", self.fc1_w.len(), d_model * 2 * hidden),
            ("fc2", self.fc2_w.len(), hidden * d_model),
        ] {
            if actual != expected {
                return Err(format!("{label}.{name}: expected {expected}, got {actual}"));
            }
        }
        Ok(())
    }
}
