//! Mamba-3 MIMO backbone wrapper (CPU), the MIMO peer of
//! [`crate::module::Mamba3Backbone`]: inference only, T=1 steps.

use crate::mamba3_mimo::config::Mamba3MimoConfig;
use crate::mamba3_mimo::cpu::inference::{Mamba3MimoStepScratch, mamba3_mimo_step};
use crate::mamba3_mimo::cpu::prefill::{
    Mamba3MimoPrefillScratch, forward_mamba3_mimo_backbone_prefill,
};
use crate::mamba3_mimo::state::{Mamba3MimoLayerState, Mamba3MimoState};
use crate::mamba3_mimo::weights::Mamba3MimoWeights;

/// Mamba-3 MIMO backbone for CPU inference. The recurrent state is owned
/// by the caller, so several sessions can share one set of weights.
pub struct Mamba3MimoBackbone {
    weights: Mamba3MimoWeights,
    cfg: Mamba3MimoConfig,
}

impl Mamba3MimoBackbone {
    /// Fresh weights with the reference init.
    pub fn init(cfg: Mamba3MimoConfig, input_dim: usize, seed: u64) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            weights: Mamba3MimoWeights::init(&cfg, input_dim, seed),
            cfg,
        })
    }

    /// Wrap loaded weights after checking every tensor against `cfg`.
    pub fn from_weights(
        weights: Mamba3MimoWeights,
        cfg: Mamba3MimoConfig,
        input_dim: usize,
    ) -> Result<Self, String> {
        weights.validate(&cfg, input_dim)?;
        Ok(Self { weights, cfg })
    }

    pub fn weights(&self) -> &Mamba3MimoWeights {
        &self.weights
    }

    pub fn config(&self) -> &Mamba3MimoConfig {
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
        scratch: &mut Mamba3MimoStepScratch,
        states: &mut [Mamba3MimoLayerState],
    ) {
        mamba3_mimo_step(temporal, input, scratch, &self.weights, states, &self.cfg);
    }

    /// A window of `seq_len` tokens in one pass (`inputs` is
    /// `[seq_len, input_dim]`, `outputs` `[seq_len, d_model]`), the decode
    /// chain's bits on every row.
    pub fn forward_sequence(
        &self,
        (inputs, outputs, seq_len): (&[f32], &mut [f32], usize),
        scratch: &mut Mamba3MimoPrefillScratch,
        states: &mut [Mamba3MimoLayerState],
    ) {
        forward_mamba3_mimo_backbone_prefill(
            outputs,
            inputs,
            &self.weights,
            states,
            scratch,
            (&self.cfg, seq_len),
        );
    }

    pub fn alloc_state(&self) -> Mamba3MimoState {
        Mamba3MimoState::zeros(&self.cfg)
    }

    /// Buffers for single steps and for windows.
    pub fn alloc_scratch(&self) -> Mamba3MimoScratch {
        Mamba3MimoScratch {
            step: Mamba3MimoStepScratch::new(&self.cfg),
            prefill: Mamba3MimoPrefillScratch::new(&self.cfg),
        }
    }
}

/// The step and window buffers of one MIMO session.
pub struct Mamba3MimoScratch {
    pub step: Mamba3MimoStepScratch,
    pub prefill: Mamba3MimoPrefillScratch,
}
