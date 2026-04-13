//! Mamba-3 SISO backbone wrapper (mirrors MambaBackbone API).

use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::cpu::inference::{Mamba3StepScratch, mamba3_step};
use crate::mamba3_siso::state::{Mamba3LayerState, Mamba3State};
use crate::mamba3_siso::weights::Mamba3Weights;

pub struct Mamba3Backbone {
    weights: Mamba3Weights,
    cfg: Mamba3Config,
}

impl Mamba3Backbone {
    pub fn init(cfg: Mamba3Config, input_dim: usize, seed: u64) -> Self {
        let weights = Mamba3Weights::init(&cfg, input_dim, seed);
        Self { weights, cfg }
    }

    pub fn config(&self) -> &Mamba3Config {
        &self.cfg
    }

    pub fn d_model(&self) -> usize {
        self.cfg.d_model
    }

    pub fn forward_step(
        &self,
        input: &[f32],
        temporal: &mut [f32],
        scratch: &mut Mamba3StepScratch,
        states: &mut [Mamba3LayerState],
    ) {
        mamba3_step(temporal, input, scratch, &self.weights, states, &self.cfg);
    }

    pub fn alloc_state(&self) -> Mamba3State {
        Mamba3State::zeros(&self.cfg)
    }

    pub fn alloc_scratch(&self) -> Mamba3StepScratch {
        Mamba3StepScratch::new(&self.cfg)
    }
}
