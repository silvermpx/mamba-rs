//! Mamba-2 backbone wrapper (CPU), the Mamba-2 peer of
//! [`crate::module::MambaBackbone`]: inference, single steps and windows.

use crate::mamba2::config::Mamba2Config;
use crate::mamba2::cpu::inference::{Mamba2StepScratch, mamba2_step};
use crate::mamba2::cpu::prefill::{Mamba2PrefillScratch, forward_mamba2_backbone_prefill};
use crate::mamba2::state::{Mamba2LayerState, Mamba2State};
use crate::mamba2::weights::Mamba2Weights;

/// Mamba-2 backbone for CPU inference. The recurrent state is owned by the
/// caller, so several sessions can share one set of weights.
pub struct Mamba2Backbone {
    weights: Mamba2Weights,
    cfg: Mamba2Config,
}

impl Mamba2Backbone {
    /// Fresh weights with the reference init.
    pub fn init(cfg: Mamba2Config, input_dim: usize, seed: u64) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            weights: Mamba2Weights::init(&cfg, input_dim, seed),
            cfg,
        })
    }

    /// Wrap loaded weights after checking every tensor against `cfg`.
    pub fn from_weights(
        weights: Mamba2Weights,
        cfg: Mamba2Config,
        input_dim: usize,
    ) -> Result<Self, String> {
        weights.validate(&cfg, input_dim)?;
        Ok(Self { weights, cfg })
    }

    pub fn weights(&self) -> &Mamba2Weights {
        &self.weights
    }

    pub fn config(&self) -> &Mamba2Config {
        &self.cfg
    }

    pub fn d_model(&self) -> usize {
        self.cfg.d_model
    }

    /// One token: reads `input`, writes `temporal[..d_model]` and advances
    /// every layer's state.
    pub fn forward_step(
        &self,
        input: &[f32],
        temporal: &mut [f32],
        scratch: &mut Mamba2StepScratch,
        states: &mut [Mamba2LayerState],
    ) {
        mamba2_step(temporal, input, scratch, &self.weights, states, &self.cfg);
    }

    /// A window of `seq_len` tokens in one pass, the decode chain's bits on
    /// every row.
    pub fn forward_sequence(
        &self,
        (inputs, outputs, seq_len): (&[f32], &mut [f32], usize),
        scratch: &mut Mamba2PrefillScratch,
        states: &mut [Mamba2LayerState],
    ) {
        forward_mamba2_backbone_prefill(
            outputs,
            inputs,
            &self.weights,
            states,
            scratch,
            (&self.cfg, seq_len),
        );
    }

    pub fn alloc_state(&self) -> Mamba2State {
        Mamba2State::zeros(&self.cfg)
    }

    /// Buffers for single steps and for windows.
    pub fn alloc_scratch(&self) -> Mamba2Scratch {
        Mamba2Scratch {
            step: Mamba2StepScratch::new(&self.cfg),
            prefill: Mamba2PrefillScratch::new(),
        }
    }
}

/// The step and window buffers of one Mamba-2 session.
pub struct Mamba2Scratch {
    pub step: Mamba2StepScratch,
    pub prefill: Mamba2PrefillScratch,
}
