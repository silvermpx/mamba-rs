//! Mamba-3 SISO dimension calculator.
//!
//! Collected dimensions to avoid passing 10+ separate `usize` params
//! through forward/backward/parallel functions.

use crate::mamba3_siso::config::Mamba3Config;

/// Collected Mamba-3 SISO dimensions for a given sequence length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mamba3Dims {
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub headdim: usize,
    pub nheads: usize,
    pub ngroups: usize,
    /// in_proj output: 2*d_inner + 2*ngroups*d_state + 3*nheads + num_rope_angles.
    pub in_proj_dim: usize,
    pub seq_len: usize,
    pub n_layers: usize,
    /// Number of RoPE angle pairs.
    pub num_rope_angles: usize,
    /// Input-dependent A floor clamp.
    pub a_floor: f32,
    /// Whether to use output RMSNormGated.
    pub is_outproj_norm: bool,
}

impl Mamba3Dims {
    /// Construct from [`Mamba3Config`] and sequence length.
    pub fn from_config(config: &Mamba3Config, seq_len: usize) -> Self {
        config.validate().expect("invalid Mamba3Config");
        Self {
            d_model: config.d_model,
            d_inner: config.d_inner(),
            d_state: config.d_state,
            headdim: config.headdim,
            nheads: config.nheads(),
            ngroups: config.ngroups,
            in_proj_dim: config.in_proj_out_dim(),
            seq_len,
            n_layers: config.n_layers,
            num_rope_angles: config.num_rope_angles(),
            a_floor: config.a_floor,
            is_outproj_norm: config.is_outproj_norm,
        }
    }

    /// Batch * sequence length (total timesteps for batched SGEMM).
    #[inline]
    pub fn bt(&self, batch: usize) -> usize {
        batch * self.seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dims_from_default_config() {
        let config = Mamba3Config::default();
        let dims = Mamba3Dims::from_config(&config, 32);

        assert_eq!(dims.d_model, 128);
        assert_eq!(dims.d_inner, 256); // 128 * 2
        assert_eq!(dims.d_state, 16);
        assert_eq!(dims.headdim, 16);
        assert_eq!(dims.nheads, 16); // 256 / 16
        assert_eq!(dims.ngroups, 1);
        assert_eq!(dims.num_rope_angles, 4); // ceil(16 * 0.5 / 2)
        assert_eq!(dims.seq_len, 32);
        assert_eq!(dims.n_layers, 4);
        // in_proj: 2*256 + 2*1*16 + 3*16 + 4 = 512 + 32 + 48 + 4 = 596
        assert_eq!(dims.in_proj_dim, 596);
    }

    #[test]
    fn test_dims_bt() {
        let config = Mamba3Config::default();
        let dims = Mamba3Dims::from_config(&config, 32);
        assert_eq!(dims.bt(4), 128);
        assert_eq!(dims.bt(1), 32);
    }
}
