//! Recurrent state of a Mamba-3 MIMO backbone.

use super::config::Mamba3MimoConfig;

/// One layer's decode state.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba3MimoLayerState {
    /// The SSM state `[nheads, headdim, d_state]`.
    pub ssm_state: Vec<f32>,
    /// The last token's keys, biased and rotated: `[nheads, mimo_rank,
    /// d_state]`, head-major so a head's keys are contiguous.
    pub k_state: Vec<f32>,
    /// The last token's raw input x `[nheads, headdim]`; the rank
    /// projection is applied when it is read.
    pub v_state: Vec<f32>,
    /// The accumulated rotation angles `[nheads, num_rope_angles]`, kept in
    /// [0, 2*pi).
    pub angle_state: Vec<f32>,
}

impl Mamba3MimoLayerState {
    pub fn zeros(cfg: &Mamba3MimoConfig) -> Self {
        let (nh, p, n) = (cfg.nheads(), cfg.headdim, cfg.d_state);
        Self {
            ssm_state: vec![0.0; nh * p * n],
            k_state: vec![0.0; nh * cfg.mimo_rank * n],
            v_state: vec![0.0; nh * p],
            angle_state: vec![0.0; nh * cfg.num_rope_angles()],
        }
    }

    /// Back to the empty state, keeping the allocations.
    pub fn reset(&mut self) {
        self.ssm_state.fill(0.0);
        self.k_state.fill(0.0);
        self.v_state.fill(0.0);
        self.angle_state.fill(0.0);
    }
}

/// The decode state of every layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba3MimoState {
    pub layers: Vec<Mamba3MimoLayerState>,
}

impl Mamba3MimoState {
    pub fn zeros(cfg: &Mamba3MimoConfig) -> Self {
        Self {
            layers: (0..cfg.n_layers)
                .map(|_| Mamba3MimoLayerState::zeros(cfg))
                .collect(),
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}
