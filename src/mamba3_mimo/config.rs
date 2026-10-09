//! Mamba-3 MIMO configuration.
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026 (arXiv 2603.15569); the
//! reference module is state-spaces/mamba `mamba_ssm/modules/mamba3.py`
//! with `is_mimo=True`.

/// Configuration for a Mamba-3 MIMO backbone.
///
/// MIMO keeps one state per head but feeds it `mimo_rank` key/value pairs
/// per token and reads `mimo_rank` outputs back, mixing them through the
/// per-head projections `mimo_x`, `mimo_z` and `mimo_o`. It also rotates
/// its keys and queries in half-split pairs `(a, a + d_state/2)`, where
/// SISO rotates adjacent pairs, so a rank-1 MIMO model is not a SISO model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mamba3MimoConfig {
    /// Model (embedding) dimension.
    pub d_model: usize,
    /// SSM state dimension per head.
    pub d_state: usize,
    /// Inner dimension expansion factor. d_inner = d_model * expand.
    pub expand: usize,
    /// Per-head dimension; any width that divides `d_inner`.
    pub headdim: usize,
    /// Number of B/C groups (nheads must be divisible by ngroups).
    pub ngroups: usize,
    /// Number of layers.
    pub n_layers: usize,
    /// The MIMO rank R: key/query vectors per group and token.
    pub mimo_rank: usize,
    /// Fraction of d_state used for RoPE angles (0.5 or 1.0).
    pub rope_fraction: f32,
    /// Input-dependent A floor clamp. A <= -a_floor.
    pub a_floor: f32,
    /// RMSNormGated on each rank's output before `mimo_o` (default false,
    /// as the released checkpoints).
    pub is_outproj_norm: bool,
    /// Epsilon of the block norms (`norm`, `norm2`, `norm_f`); the
    /// reference's `norm_epsilon`.
    pub rms_norm_eps: f32,
    /// Epsilon of the mixer's B/C norms and output-gate norm, which the
    /// reference fixes at 1e-5 whatever `norm_epsilon` says.
    pub ssm_norm_eps: f32,
    /// The gated MLP after the mixer in every layer; 0 means no MLP. The
    /// hidden width is this value rounded up to a multiple of 128.
    pub d_intermediate: usize,
}

/// The largest rank: each state row keeps one readout accumulator per
/// rank in registers. The reference warns outside {1, 2, 4, 8}.
pub const MAX_MIMO_RANK: usize = 16;

/// The largest d_state, the reference's own maximum.
pub const MAX_D_STATE: usize = 256;

impl Mamba3MimoConfig {
    /// The gated MLP's hidden width, 0 when the model has none.
    pub fn mlp_hidden(&self) -> usize {
        crate::mlp::mlp_hidden(self.d_intermediate)
    }

    /// Inner dimension: d_model * expand.
    pub fn d_inner(&self) -> usize {
        self.d_model * self.expand
    }

    /// Number of heads: d_inner / headdim.
    pub fn nheads(&self) -> usize {
        self.d_inner() / self.headdim
    }

    /// Number of RoPE angles: `int(d_state * rope_fraction)` rounded down
    /// to even, halved, as the reference computes it.
    pub fn num_rope_angles(&self) -> usize {
        (self.d_state as f32 * self.rope_fraction) as usize / 2
    }

    /// Width of one token's B (or C) block: `mimo_rank * ngroups * d_state`.
    pub fn bc_width(&self) -> usize {
        self.mimo_rank * self.ngroups * self.d_state
    }

    /// in_proj output dimension, split as
    /// `[z:di | x:di | B:R*G*N | C:R*G*N | dd_dt:H | dd_A:H | trap:H | angles:na]`.
    pub fn in_proj_out_dim(&self) -> usize {
        2 * self.d_inner() + 2 * self.bc_width() + 3 * self.nheads() + self.num_rope_angles()
    }

    /// Validate the model envelope: the shapes the reference accepts.
    pub fn validate(&self) -> Result<(), String> {
        if self.headdim < 1 {
            return Err("headdim must be >= 1".into());
        }
        if self.d_model < 1 || self.expand < 1 {
            return Err("d_model and expand must be >= 1".into());
        }
        if !self.d_inner().is_multiple_of(self.headdim) {
            return Err(format!(
                "d_inner ({}) must be divisible by headdim ({})",
                self.d_inner(),
                self.headdim
            ));
        }
        if !(1..=MAX_D_STATE).contains(&self.d_state) {
            return Err(format!(
                "d_state ({}) must be in 1..={MAX_D_STATE} (the reference range)",
                self.d_state
            ));
        }
        // The half-split rotation pairs a with a + d_state/2.
        if self.num_rope_angles() > 0 && !self.d_state.is_multiple_of(2) {
            return Err(format!(
                "d_state ({}) must be even for the half-split rotation",
                self.d_state
            ));
        }
        if self.ngroups < 1 || !self.nheads().is_multiple_of(self.ngroups) {
            return Err(format!(
                "nheads ({}) must be a multiple of ngroups ({})",
                self.nheads(),
                self.ngroups
            ));
        }
        if !(1..=MAX_MIMO_RANK).contains(&self.mimo_rank) {
            return Err(format!(
                "mimo_rank ({}) must be in 1..={MAX_MIMO_RANK}",
                self.mimo_rank
            ));
        }
        if !(self.rope_fraction == 0.5 || self.rope_fraction == 1.0) {
            return Err(format!(
                "rope_fraction must be 0.5 or 1.0, got {}",
                self.rope_fraction
            ));
        }
        if self.a_floor.is_nan() || self.a_floor <= 0.0 {
            return Err(format!("a_floor must be positive, got {}", self.a_floor));
        }
        if self.n_layers < 1 {
            return Err("n_layers must be >= 1".into());
        }
        if self.d_intermediate > (1 << 20) {
            return Err(format!(
                "d_intermediate ({}) must be <= 2^20",
                self.d_intermediate
            ));
        }
        for (name, eps) in [
            ("rms_norm_eps", self.rms_norm_eps),
            ("ssm_norm_eps", self.ssm_norm_eps),
        ] {
            if !eps.is_finite() || eps <= 0.0 {
                return Err(format!("{name} must be positive, got {eps}"));
            }
        }
        Ok(())
    }
}

impl Default for Mamba3MimoConfig {
    /// A small model in the released checkpoints' proportions: rank 4,
    /// rope fraction 0.5, no output norm.
    fn default() -> Self {
        Self {
            d_model: 128,
            d_state: 16,
            expand: 2,
            headdim: 16,
            ngroups: 1,
            n_layers: 4,
            mimo_rank: 4,
            rope_fraction: 0.5,
            a_floor: 1e-4,
            is_outproj_norm: false,
            rms_norm_eps: 1e-5,
            ssm_norm_eps: 1e-5,
            d_intermediate: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released mimo-187m shape: its in_proj is 4200 wide.
    #[test]
    fn released_shape_is_accepted() {
        let cfg = Mamba3MimoConfig {
            d_model: 768,
            d_state: 128,
            headdim: 64,
            n_layers: 12,
            d_intermediate: 1264,
            ..Mamba3MimoConfig::default()
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.in_proj_out_dim(), 4200);
        assert_eq!(cfg.mlp_hidden(), 1280);
    }

    #[test]
    fn invalid_shapes_are_refused() {
        let base = Mamba3MimoConfig::default();
        for bad in [
            Mamba3MimoConfig {
                mimo_rank: 0,
                ..base
            },
            Mamba3MimoConfig {
                mimo_rank: MAX_MIMO_RANK + 1,
                ..base
            },
            Mamba3MimoConfig {
                d_state: 15,
                ..base
            },
            Mamba3MimoConfig {
                headdim: 24,
                ..base
            },
            Mamba3MimoConfig {
                ssm_norm_eps: f32::NAN,
                ..base
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }
}
