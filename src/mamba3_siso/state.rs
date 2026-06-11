//! Persistent recurrent state for Mamba-3 SISO layers.
//!
//! Four state components per layer (no conv state — Mamba-3 has no conv1d):
//! - SSM hidden state: `[nheads * headdim * d_state]`
//! - K state (previous B after RoPE): `[nheads * d_state]`
//! - V state (previous x): `[nheads * headdim]`
//! - Angle state (RoPE cumulative angles): `[nheads * num_rope_angles]`

use super::config::Mamba3Config;

/// Persistent recurrent state for one Mamba-3 SISO layer.
#[derive(Clone)]
pub struct Mamba3LayerState {
    /// SSM hidden state: `[nheads * headdim * d_state]`.
    pub ssm_state: Vec<f32>,
    /// K state (post-RoPE, post-bias B from previous step): `[nheads * d_state]`.
    pub k_state: Vec<f32>,
    /// V state (x from previous step): `[nheads * headdim]`.
    pub v_state: Vec<f32>,
    /// RoPE cumulative angle state: `[nheads * num_rope_angles]`.
    /// Wraps to `[0, 2*pi)` after each step.
    pub angle_state: Vec<f32>,
}

impl Mamba3LayerState {
    /// Create a zeroed layer state.
    pub fn zeros(nheads: usize, headdim: usize, d_state: usize, num_rope_angles: usize) -> Self {
        Self {
            ssm_state: vec![0.0; nheads * headdim * d_state],
            k_state: vec![0.0; nheads * d_state],
            v_state: vec![0.0; nheads * headdim],
            angle_state: vec![0.0; nheads * num_rope_angles.max(1)],
        }
    }

    /// Reset all state to zero (new sequence boundary).
    pub fn reset(&mut self) {
        self.ssm_state.fill(0.0);
        self.k_state.fill(0.0);
        self.v_state.fill(0.0);
        self.angle_state.fill(0.0);
    }
}

/// Full recurrent state for the Mamba-3 SISO backbone (all layers).
#[derive(Clone)]
pub struct Mamba3State {
    pub layers: Vec<Mamba3LayerState>,
}

impl Mamba3State {
    /// Create zeroed state for all layers from config.
    pub fn zeros(cfg: &Mamba3Config) -> Self {
        let nh = cfg.nheads();
        let hd = cfg.headdim;
        let ds = cfg.d_state;
        let na = cfg.num_rope_angles();
        Self {
            layers: (0..cfg.n_layers)
                .map(|_| Mamba3LayerState::zeros(nh, hd, ds, na))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_zeros() {
        let cfg = Mamba3Config::default();
        let state = Mamba3State::zeros(&cfg);
        assert_eq!(state.layers.len(), cfg.n_layers);
        let l = &state.layers[0];
        assert_eq!(l.ssm_state.len(), cfg.nheads() * cfg.headdim * cfg.d_state);
        assert_eq!(l.k_state.len(), cfg.nheads() * cfg.d_state);
        assert_eq!(l.v_state.len(), cfg.nheads() * cfg.headdim);
        assert_eq!(
            l.angle_state.len(),
            cfg.nheads() * cfg.num_rope_angles().max(1)
        );
        assert!(l.ssm_state.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_state_reset() {
        let cfg = Mamba3Config::default();
        let mut state = Mamba3State::zeros(&cfg);
        state.layers[0].ssm_state[0] = 42.0;
        state.layers[0].angle_state[0] = std::f32::consts::PI;
        state.reset();
        assert_eq!(state.layers[0].ssm_state[0], 0.0);
        assert_eq!(state.layers[0].angle_state[0], 0.0);
    }
}
