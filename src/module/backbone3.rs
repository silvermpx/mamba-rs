//! Mamba-3 SISO backbone wrapper (CPU). Mirrors [`MambaBackbone`] for the
//! Mamba-3 SISO architecture (Lahoti et al., ICLR 2026): no conv1d,
//! input-dependent A, trapezoidal integration, RoPE, BCNorm, output gate.
//!
//! This is the **inference-only T=1** entrypoint. Training (forward,
//! BPTT, parallel rayon) lives in [`crate::mamba3_siso::cpu`] and operates
//! on `TrainMamba3Weights` instead.
//!
//! [`MambaBackbone`]: crate::module::MambaBackbone

use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::cpu::inference::{Mamba3StepScratch, mamba3_step};
use crate::mamba3_siso::state::{Mamba3LayerState, Mamba3State};
use crate::mamba3_siso::weights::Mamba3Weights;

/// Mamba-3 SISO backbone for CPU inference. Holds the full set of
/// per-layer weights and the architecture config; the recurrent state
/// is owned by the caller (`Mamba3State`) so multiple sessions can run
/// concurrently against shared weights.
///
/// For GPU inference see `crate::mamba3_siso::gpu::GpuMamba3Backbone`;
/// for LM-style generation with sampling see
/// `crate::module::gpu_lm3::GpuMamba3LM` (feature `cuda` + `hf`).
pub struct Mamba3Backbone {
    weights: Mamba3Weights,
    cfg: Mamba3Config,
}

impl Mamba3Backbone {
    /// Initialize fresh weights from `cfg`. Uses the Mamba-3 init scheme
    /// (zeros for biases, ones for B/C bias and norm scales, the
    /// Mamba-paper dt_proj scheme for the discretization MLP).
    pub fn init(cfg: Mamba3Config, input_dim: usize, seed: u64) -> Self {
        let weights = Mamba3Weights::init(&cfg, input_dim, seed);
        Self { weights, cfg }
    }

    /// Borrow the architecture config.
    pub fn config(&self) -> &Mamba3Config {
        &self.cfg
    }

    /// Backbone hidden width (`d_model` in the paper).
    pub fn d_model(&self) -> usize {
        self.cfg.d_model
    }

    /// Single-step (T=1) forward. Reads `input[..d_model]`, writes
    /// `temporal[..d_model]`, advances each layer's recurrent state in
    /// `states` in place. `scratch` is reusable across calls.
    pub fn forward_step(
        &self,
        input: &[f32],
        temporal: &mut [f32],
        scratch: &mut Mamba3StepScratch,
        states: &mut [Mamba3LayerState],
    ) {
        mamba3_step(temporal, input, scratch, &self.weights, states, &self.cfg);
    }

    /// Allocate a zero-initialized full-stack recurrent state (SSM + K +
    /// V + angle accumulators per layer). Reset between independent
    /// sequences.
    pub fn alloc_state(&self) -> Mamba3State {
        Mamba3State::zeros(&self.cfg)
    }

    /// Allocate a per-step scratch buffer (split, BCNorm, RoPE, gating,
    /// proj). Reusable across calls; never aliased across threads.
    pub fn alloc_scratch(&self) -> Mamba3StepScratch {
        Mamba3StepScratch::new(&self.cfg)
    }
}
