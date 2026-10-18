use super::config::MambaConfig;

/// Persistent recurrent state for one Mamba layer.
///
/// Holds the conv1d shift register and SSM hidden state.
/// Reset on episode/sequence boundaries.
#[derive(Clone)]
pub struct MambaLayerState {
    /// Conv1d shift register `[(d_conv - 1) * d_inner]`
    pub conv_state: Vec<f32>,
    /// SSM hidden state `[d_inner * d_state]`
    pub ssm_state: Vec<f32>,
}

impl MambaLayerState {
    /// A zeroed layer state, the start of a sequence.
    pub fn zeros(cfg: &MambaConfig) -> Self {
        let (di, dc) = (cfg.d_inner(), cfg.d_conv.max(1));
        Self {
            conv_state: vec![0.0; (dc - 1) * di],
            ssm_state: vec![0.0; di * cfg.d_state],
        }
    }

    /// Reset all state to zero (new sequence boundary).
    pub fn reset(&mut self) {
        self.conv_state.fill(0.0);
        self.ssm_state.fill(0.0);
    }
}

/// Full recurrent state for the Mamba backbone (all layers).
#[derive(Clone)]
pub struct MambaState {
    pub layers: Vec<MambaLayerState>,
}

impl MambaState {
    /// A zeroed state of every layer, the start of a sequence.
    pub fn zeros(cfg: &MambaConfig) -> Self {
        Self {
            layers: (0..cfg.n_layers)
                .map(|_| MambaLayerState::zeros(cfg))
                .collect(),
        }
    }

    /// Reset all layers (new sequence boundary).
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}
