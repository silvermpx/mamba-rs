/// Persistent recurrent state for one Mamba layer.
///
/// Holds the conv1d shift register and SSM hidden state.
/// Reset on episode/sequence boundaries.
pub struct MambaLayerState {
    /// Conv1d shift register `[(d_conv - 1) * d_inner]`
    pub conv_state: Vec<f32>,
    /// SSM hidden state `[d_inner * d_state]`
    pub ssm_state: Vec<f32>,
}

impl MambaLayerState {
    /// Create a zeroed layer state for a given Mamba configuration.
    ///
    /// - `d_inner`: expanded model dimension (`expand * d_model`)
    /// - `d_state`: SSM state dimension (memory capacity per channel)
    /// - `d_conv`: convolution kernel width
    pub fn zeros(d_inner: usize, d_state: usize, d_conv: usize) -> Self {
        assert!(d_conv > 0, "d_conv must be > 0");
        Self {
            conv_state: vec![0.0; (d_conv - 1) * d_inner],
            ssm_state: vec![0.0; d_inner * d_state],
        }
    }

    /// Reset all state to zero (new sequence boundary).
    pub fn reset(&mut self) {
        self.conv_state.fill(0.0);
        self.ssm_state.fill(0.0);
    }
}

/// Full recurrent state for the Mamba backbone (all layers).
pub struct MambaState {
    pub layers: Vec<MambaLayerState>,
}

impl MambaState {
    /// Create zeroed state for all Mamba layers.
    ///
    /// - `n_layers`: number of stacked Mamba layers
    /// - `d_inner`: expanded model dimension (`expand * d_model`)
    /// - `d_state`: SSM state dimension
    /// - `d_conv`: convolution kernel width
    pub fn zeros(n_layers: usize, d_inner: usize, d_state: usize, d_conv: usize) -> Self {
        Self {
            layers: (0..n_layers)
                .map(|_| MambaLayerState::zeros(d_inner, d_state, d_conv))
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
