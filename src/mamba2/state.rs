//! Recurrent state of a Mamba-2 backbone.

use super::config::Mamba2Config;

/// One layer's decode state.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba2LayerState {
    /// The SSM state `[nheads, d_state, headdim]`: for each head, one row
    /// of `headdim` channels per state entry, so the recurrence runs along
    /// contiguous channels with no reduction.
    pub ssm: Vec<f32>,
    /// The last `d_conv - 1` raw conv inputs `[d_conv - 1, conv_dim]`,
    /// oldest first.
    pub conv_hist: Vec<f32>,
}

impl Mamba2LayerState {
    pub fn zeros(cfg: &Mamba2Config) -> Self {
        Self {
            ssm: vec![0.0; cfg.nheads() * cfg.d_state * cfg.headdim],
            conv_hist: vec![0.0; (cfg.d_conv - 1) * cfg.conv_dim()],
        }
    }

    pub fn reset(&mut self) {
        self.ssm.fill(0.0);
        self.conv_hist.fill(0.0);
    }

    /// The SSM state in the reference's `(nheads, headdim, d_state)` layout.
    pub fn ssm_to_reference(&self, cfg: &Mamba2Config, out: &mut [f32]) {
        let (n, p) = (cfg.d_state, cfg.headdim);
        for (src, dst) in self
            .ssm
            .chunks_exact(n * p)
            .zip(out.chunks_exact_mut(n * p))
        {
            for (ni, row) in src.chunks_exact(p).enumerate() {
                for (pi, &v) in row.iter().enumerate() {
                    dst[pi * n + ni] = v;
                }
            }
        }
    }
}

/// The decode state of every layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba2State {
    pub layers: Vec<Mamba2LayerState>,
}

impl Mamba2State {
    pub fn zeros(cfg: &Mamba2Config) -> Self {
        Self {
            layers: (0..cfg.n_layers)
                .map(|_| Mamba2LayerState::zeros(cfg))
                .collect(),
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}
