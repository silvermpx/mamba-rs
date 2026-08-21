//! Mamba-3 SISO configuration.
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026 (arXiv 2603.15569).

use crate::config::ScanMode;

/// Configuration for a Mamba-3 SISO backbone.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// GPU SSM scan mode for the TRAINING forward/backward. `Auto` resolves
    /// to the chunked parallel path (see [`Mamba3Config::train_use_parallel_scan`]);
    /// CPU paths and the T=1 decode step are unaffected.
    pub scan_mode: ScanMode,
    /// RMSNorm/BCNorm epsilon used by every norm site in this backbone
    /// (layer RMSNorm, B/C BCNorm, RMSNormGated, final norm). A checkpoint
    /// trained with a different eps loads a DIFFERENT model — the value
    /// rides checkpoint metadata (see `serialize.rs`).
    pub rms_norm_eps: f32,
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

    /// Validate all constraints.
    ///
    /// Returns `Err` with a description on invalid configuration — same
    /// contract as [`crate::config::MambaConfig::validate`] (the old
    /// panicking variant made library constructors abort instead of
    /// surfacing a recoverable error).
    pub fn validate(&self) -> Result<(), String> {
        if !(self.headdim <= 32 && self.headdim.is_power_of_two()) {
            return Err(format!(
                "headdim ({}) must be <= 32 and power of 2 (warp shuffle)",
                self.headdim
            ));
        }
        if !self.d_inner().is_multiple_of(self.headdim) {
            return Err(format!(
                "d_inner ({}) must be divisible by headdim ({})",
                self.d_inner(),
                self.headdim
            ));
        }
        // 256 is the reference implementations' own maximum. The GPU
        // kernels size their per-thread state arrays at JIT time from
        // this value (past ~128 the compiler spills to local memory —
        // correct, slower, accepted as a first-class capacity knob).
        if !(self.d_state >= 1 && self.d_state <= 256) {
            return Err(format!(
                "d_state ({}) must be in 1..=256 (the reference range)",
                self.d_state
            ));
        }
        if 2 * self.num_rope_angles() > self.d_state {
            return Err(format!(
                "2 * num_rope_angles ({}) must be <= d_state ({}) — rotation \
                 pairs may not cross the head boundary",
                2 * self.num_rope_angles(),
                self.d_state
            ));
        }
        // The chunked-backward kernel stages two chunk-by-d_state operand
        // tiles (chunk size 64) plus smaller pieces in dynamic shared
        // memory; the kernel loader opts in to the device's extended
        // budget, and 99 KB is the opt-in floor across every supported
        // part since Ampere (some parts offer more; the floor keeps one
        // portable bound). Shapes above it need the state-tiled redesign
        // of that kernel and are refused loudly until it lands. (This
        // replaces
        // the cruder headdim*d_state <= 1024 product bound: the register
        // arrays it protected spill gracefully under compiler control,
        // while shared memory is a hard launch limit.)
        // Only the chunked training path runs that kernel: an explicit
        // Sequential scan mode trains without it and skips this bound.
        let cs = 64usize;
        let chunked_bwd_smem = (2 * cs * self.d_state
            + 2 * cs * self.headdim
            + 4 * cs
            + 2 * self.headdim * self.d_state)
            * 4;
        if self.train_use_parallel_scan() && chunked_bwd_smem > 99 * 1024 {
            return Err(format!(
                "headdim {} with d_state {} needs {} KB of shared memory in the \
                 chunked backward — over the 99 KB opt-in budget; this ceiling \
                 falls with the state-tiled redesign of that kernel",
                self.headdim,
                self.d_state,
                chunked_bwd_smem / 1024
            ));
        }
        if self.ngroups < 1 {
            return Err("ngroups must be >= 1".into());
        }
        if !self.nheads().is_multiple_of(self.ngroups) {
            return Err(format!(
                "nheads ({}) must be divisible by ngroups ({})",
                self.nheads(),
                self.ngroups
            ));
        }
        if !(self.rope_fraction == 0.5 || self.rope_fraction == 1.0) {
            return Err(format!(
                "rope_fraction must be 0.5 or 1.0, got {}",
                self.rope_fraction
            ));
        }
        if self.a_floor <= 0.0 {
            return Err(format!("a_floor must be positive, got {}", self.a_floor));
        }
        if self.n_layers < 1 {
            return Err("n_layers must be >= 1".into());
        }
        if self.d_model < 1 {
            return Err("d_model must be >= 1".into());
        }
        if self.expand < 1 {
            return Err("expand must be >= 1".into());
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(format!(
                "rms_norm_eps must be positive, got {}",
                self.rms_norm_eps
            ));
        }
        if self.is_outproj_norm && self.d_inner() > 1024 {
            return Err(format!(
                "is_outproj_norm requires d_inner ({}) <= 1024: the \
                 rmsnorm_gated kernels launch one thread per channel and \
                 SILENTLY no-op past 1024 — reject loudly here \
                 until the kernels tile the channel dim",
                self.d_inner()
            ));
        }
        Ok(())
    }

    /// Resolve the training-time scan choice from `scan_mode`.
    ///
    /// M3 semantics differ from Mamba-1's length-threshold `Auto`: the
    /// chunked path is the only one with a mixed-precision backward (the
    /// sequential mixed backward is a documented non-goal — its activation
    /// tape costs CHUNK_SIZE x VRAM), and the trainers have always run
    /// chunked. `Auto` therefore resolves to the chunked parallel path
    /// regardless of T; `Sequential` is honored for the f32 lane only
    /// (the mixed forward/backward reject it loudly downstream).
    pub fn train_use_parallel_scan(&self) -> bool {
        !matches!(self.scan_mode, ScanMode::Sequential)
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
            scan_mode: ScanMode::Auto,
            // Reference default (state-spaces/mamba RMSNorm eps). Rides
            // checkpoint metadata; changing it re-defines the model.
            rms_norm_eps: 1e-5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let cfg = Mamba3Config::default();
        cfg.validate().unwrap();
        assert_eq!(cfg.d_inner(), 256);
        assert_eq!(cfg.nheads(), 16);
        assert_eq!(cfg.num_rope_angles(), 4); // ceil(16 * 0.5 / 2) = 4
        assert_eq!(cfg.in_proj_out_dim(), 2 * 256 + 2 * 16 + 3 * 16 + 4);
    }

    #[test]
    fn test_invalid_headdim() {
        let err = Mamba3Config {
            headdim: 7,
            ..Mamba3Config::default()
        }
        .validate()
        .unwrap_err();
        assert!(err.contains("headdim"), "{err}");
    }

    /// Large d_state is first-class up to the reference range: the
    /// kernels size their state arrays at JIT time, so 65 and 128 are
    /// plainly valid; past 256 there is no upstream precedent and the
    /// config refuses loudly.
    #[test]
    fn test_d_state_reference_range() {
        for ds in [65usize, 128] {
            Mamba3Config {
                d_state: ds,
                ..Mamba3Config::default()
            }
            .validate()
            .unwrap_or_else(|e| panic!("d_state {ds} must validate: {e}"));
        }
        let err = Mamba3Config {
            d_state: 257,
            ..Mamba3Config::default()
        }
        .validate()
        .unwrap_err();
        assert!(err.contains("d_state"), "{err}");
    }

    /// Shapes whose chunked-backward shared-memory tiles exceed the
    /// opt-in budget are refused loudly, naming the resource — silent
    /// wrong math past a capacity was the failure mode all these guards
    /// exist for.
    #[test]
    fn test_chunked_backward_smem_budget_rejected_loudly() {
        let err = Mamba3Config {
            d_state: 256,
            headdim: 32,
            ..Mamba3Config::default()
        }
        .validate()
        .unwrap_err();
        assert!(err.contains("shared memory"), "{err}");
    }

    /// Scan-mode resolver pins — Auto resolves to the chunked
    /// parallel path for TRAINING (the only route with a mixed backward);
    /// explicit Sequential is honored (f32 lane).
    #[test]
    fn test_scan_mode_resolver_pins() {
        let auto = Mamba3Config::default();
        assert_eq!(auto.scan_mode, ScanMode::Auto);
        assert!(auto.train_use_parallel_scan());
        assert!(
            Mamba3Config {
                scan_mode: ScanMode::Parallel,
                ..Mamba3Config::default()
            }
            .train_use_parallel_scan()
        );
        assert!(
            !Mamba3Config {
                scan_mode: ScanMode::Sequential,
                ..Mamba3Config::default()
            }
            .train_use_parallel_scan()
        );
    }

    /// The rmsnorm_gated kernels silently no-op at d_inner > 1024
    /// — the config now rejects that combination loudly. d_inner > 1024
    /// WITHOUT the gated norm stays valid (those kernels never launch).
    #[test]
    fn test_gated_norm_d_inner_cap_rejected_loudly() {
        let big_gated = Mamba3Config {
            d_model: 1024,
            expand: 2,
            headdim: 32,
            is_outproj_norm: true,
            ..Mamba3Config::default()
        };
        let err = big_gated.validate().unwrap_err();
        assert!(err.contains("is_outproj_norm"), "{err}");

        let big_silu = Mamba3Config {
            is_outproj_norm: false,
            ..big_gated
        };
        big_silu.validate().unwrap();
    }

    #[test]
    fn test_invalid_rms_norm_eps() {
        for bad in [0.0_f32, -1e-5, f32::NAN] {
            let err = Mamba3Config {
                rms_norm_eps: bad,
                ..Mamba3Config::default()
            }
            .validate()
            .unwrap_err();
            assert!(err.contains("rms_norm_eps"), "{err}");
        }
    }

    #[test]
    fn test_invalid_rope_fraction() {
        let err = Mamba3Config {
            rope_fraction: 0.25,
            ..Mamba3Config::default()
        }
        .validate()
        .unwrap_err();
        assert!(err.contains("rope_fraction"), "{err}");
    }
}
