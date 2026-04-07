//! Collected Mamba dimensions to avoid passing 8+ separate `usize` params.
//!
//! Used by all optimized forward/backward functions (O1-O5 flat layout).

use crate::config::MambaConfig;

/// Collected Mamba dimensions to avoid passing 8+ separate usize params.
/// Used by all optimized forward/backward functions.
#[derive(Debug, Clone, Copy)]
pub struct MambaDims {
    /// Model dimension (features projected to this size).
    pub d_model: usize,
    /// Expanded inner dimension (`expand * d_model`).
    pub d_inner: usize,
    /// SSM state dimension (memory capacity per channel).
    pub d_state: usize,
    /// Convolution kernel width.
    pub d_conv: usize,
    /// Delta projection bottleneck rank (`ceil(d_model / 16)`).
    pub dt_rank: usize,
    /// x_proj output size: `dt_rank + 2 * d_state` (delta_raw, B, C concatenated).
    pub xdbl_dim: usize,
    /// Sequence length (timesteps per sample).
    pub seq_len: usize,
    /// Dimension of the external input fed into input_proj.
    pub mamba_input_dim: usize,
    /// Number of stacked Mamba layers.
    pub n_layers: usize,
}

impl MambaDims {
    /// Build from individual dimension values.
    ///
    /// Accepts a tuple to stay within clippy's 7-argument limit.
    /// Order: `(d_model, d_inner, d_state, d_conv, dt_rank, seq_len, mamba_input_dim, n_layers)`.
    pub fn new(
        (d_model, d_inner, d_state, d_conv, dt_rank, seq_len, mamba_input_dim, n_layers): (
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
        ),
    ) -> Self {
        Self {
            d_model,
            d_inner,
            d_state,
            d_conv,
            dt_rank,
            xdbl_dim: dt_rank + 2 * d_state,
            seq_len,
            mamba_input_dim,
            n_layers,
        }
    }

    /// Construct from [`MambaConfig`] with explicit input dimension.
    pub fn from_config(config: &MambaConfig, seq_len: usize, input_dim: usize) -> Self {
        Self::new((
            config.d_model,
            config.d_inner(),
            config.d_state,
            config.d_conv,
            config.dt_rank(),
            seq_len,
            input_dim,
            config.n_layers,
        ))
    }
}

/// Mutable recurrent state for Mamba forward/backward.
///
/// Bundles the three arrays that always travel together:
/// conv states, SSM hidden states, and pre-computed `-exp(a_log)`.
pub struct MambaRecurrentState<'a> {
    /// Conv1d shift register `[n_layers * d_inner * d_conv]`.
    pub conv: &'a mut [f32],
    /// SSM hidden state `[n_layers * d_inner * d_state]`.
    pub ssm: &'a mut [f32],
    /// Pre-computed `-exp(a_log)` `[n_layers * d_inner * d_state]`.
    pub a_neg: &'a [f32],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_dims_defaults() {
        let dims = MambaDims::new((128, 256, 16, 4, 8, 33, 346, 3));
        assert_eq!(dims.d_model, 128);
        assert_eq!(dims.d_inner, 256);
        assert_eq!(dims.d_state, 16);
        assert_eq!(dims.d_conv, 4);
        assert_eq!(dims.dt_rank, 8);
        assert_eq!(dims.xdbl_dim, 40); // 8 + 2*16
        assert_eq!(dims.seq_len, 33);
        assert_eq!(dims.mamba_input_dim, 346);
        assert_eq!(dims.n_layers, 3);
    }

    #[test]
    fn test_mamba_dims_from_config() {
        let config = MambaConfig::default();
        let dims = MambaDims::from_config(&config, 33, 346);
        assert_eq!(dims.d_model, 128);
        assert_eq!(dims.d_inner, 256); // expand=2 * d_model=128
        assert_eq!(dims.d_state, 16);
        assert_eq!(dims.d_conv, 4);
        assert_eq!(dims.dt_rank, 8); // ceil(128/16)
        assert_eq!(dims.xdbl_dim, 40); // 8 + 2*16
        assert_eq!(dims.seq_len, 33);
        assert_eq!(dims.mamba_input_dim, 346);
        assert_eq!(dims.n_layers, 3);
    }
}
