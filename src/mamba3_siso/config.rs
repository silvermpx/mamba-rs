//! Mamba-3 SISO configuration.
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026 (arXiv 2603.15569).

/// Configuration for a Mamba-3 SISO backbone.
#[derive(Debug, Clone)]
pub struct Mamba3Config {
    /// Model (embedding) dimension.
    pub d_model: usize,
    /// SSM state dimension per head.
    pub d_state: usize,
    /// Inner dimension expansion factor. d_inner = d_model * expand.
    pub expand: usize,
    /// Per-head dimension (must be power-of-2, <= 32 for warp shuffle).
    pub headdim: usize,
    /// Number of B/C groups (nheads must be divisible by ngroups).
    pub ngroups: usize,
    /// Number of Mamba-3 layers.
    pub n_layers: usize,
    /// Fraction of d_state used for RoPE angles (0.5 or 1.0).
    pub rope_fraction: f32,
    /// Input-dependent A floor clamp. A <= -a_floor.
    pub a_floor: f32,
    /// Enable RMSNormGated before out_proj (default: false per reference).
    pub is_outproj_norm: bool,
}

impl Mamba3Config {
    /// Inner dimension: d_model * expand.
    pub fn d_inner(&self) -> usize {
        self.d_model * self.expand
    }

    /// Number of heads: d_inner / headdim.
    pub fn nheads(&self) -> usize {
        self.d_inner() / self.headdim
    }

    /// Number of RoPE angle pairs: floor(d_state * rope_fraction) / 2.
    ///
    /// Floor semantics match the reference (`int(d_state * rope_fraction)
    /// // 2` in state-spaces/mamba). The previous ceil variant rotated one
    /// pair too many for odd `d_state * rope_fraction`, indexing past the
    /// head slice (out of bounds in the GPU rope kernels).
    pub fn num_rope_angles(&self) -> usize {
        (self.d_state as f32 * self.rope_fraction) as usize / 2
    }

    /// in_proj output dimension (8-way split).
    /// [z:d_inner | x:d_inner | B:ng*ds | C:ng*ds | dd_dt:nh | dd_A:nh | trap:nh | angles:n_rope]
    pub fn in_proj_out_dim(&self) -> usize {
        let di = self.d_inner();
        let nh = self.nheads();
        let ng = self.ngroups;
        let ds = self.d_state;
        let na = self.num_rope_angles();
        2 * di + 2 * ng * ds + 3 * nh + na
    }

    /// Validate all constraints. Panics on invalid configuration.
    pub fn validate(&self) {
        assert!(
            self.headdim <= 32 && self.headdim.is_power_of_two(),
            "headdim ({}) must be <= 32 and power of 2 (warp shuffle)",
            self.headdim
        );
        assert!(
            self.d_inner().is_multiple_of(self.headdim),
            "d_inner ({}) must be divisible by headdim ({})",
            self.d_inner(),
            self.headdim
        );
        assert!(
            self.d_state >= 1 && self.d_state <= 64,
            "d_state ({}) must be in 1..=64 (CUDA register limit)",
            self.d_state
        );
        assert!(
            2 * self.num_rope_angles() <= self.d_state,
            "2 * num_rope_angles ({}) must be <= d_state ({}) — rotation \
             pairs may not cross the head boundary",
            2 * self.num_rope_angles(),
            self.d_state
        );
        assert!(
            self.headdim * self.d_state <= 1024,
            "headdim*d_state ({}) must be <= 1024 (CUDA register budget)",
            self.headdim * self.d_state
        );
        assert!(self.ngroups >= 1, "ngroups must be >= 1");
        assert!(
            self.nheads().is_multiple_of(self.ngroups),
            "nheads ({}) must be divisible by ngroups ({})",
            self.nheads(),
            self.ngroups
        );
        assert!(
            self.rope_fraction == 0.5 || self.rope_fraction == 1.0,
            "rope_fraction must be 0.5 or 1.0, got {}",
            self.rope_fraction
        );
        assert!(
            self.a_floor > 0.0,
            "a_floor must be positive, got {}",
            self.a_floor
        );
        assert!(self.n_layers >= 1, "n_layers must be >= 1");
        assert!(self.d_model >= 1, "d_model must be >= 1");
        assert!(self.expand >= 1, "expand must be >= 1");
    }
}

impl Default for Mamba3Config {
    fn default() -> Self {
        Self {
            d_model: 128,
            d_state: 16,
            expand: 2,
            headdim: 16,
            ngroups: 1,
            n_layers: 4,
            rope_fraction: 0.5,
            // state-spaces/mamba mamba3.py default: A_floor=1e-4. Earlier
            // mamba-rs used 0.0625 (≈625× stronger clamp) which forced much
            // faster state decay than reference. Switched to 1e-4 for training
            // parity with upstream.
            a_floor: 1e-4,
            is_outproj_norm: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let cfg = Mamba3Config::default();
        cfg.validate();
        assert_eq!(cfg.d_inner(), 256);
        assert_eq!(cfg.nheads(), 16);
        assert_eq!(cfg.num_rope_angles(), 4); // ceil(16 * 0.5 / 2) = 4
        assert_eq!(cfg.in_proj_out_dim(), 2 * 256 + 2 * 16 + 3 * 16 + 4);
    }

    #[test]
    #[should_panic(expected = "headdim")]
    fn test_invalid_headdim() {
        Mamba3Config {
            headdim: 7,
            ..Mamba3Config::default()
        }
        .validate();
    }

    #[test]
    #[should_panic(expected = "d_state")]
    fn test_invalid_d_state() {
        Mamba3Config {
            d_state: 128,
            ..Mamba3Config::default()
        }
        .validate();
    }

    #[test]
    #[should_panic(expected = "rope_fraction")]
    fn test_invalid_rope_fraction() {
        Mamba3Config {
            rope_fraction: 0.25,
            ..Mamba3Config::default()
        }
        .validate();
    }
}
