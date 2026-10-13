//! Mamba-3 SISO configuration.
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026 (arXiv 2603.15569).

use crate::config::{ScanMode, envelope};

/// Configuration for a Mamba-3 SISO backbone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mamba3Config {
    /// Model (embedding) dimension.
    pub d_model: usize,
    /// SSM state dimension per head.
    pub d_state: usize,
    /// Inner dimension expansion factor. d_inner = d_model * expand.
    pub expand: usize,
    /// Per-head dimension. Any width that divides `d_inner`, as in the
    /// reference; the GPU kernels serve a narrower set (see
    /// [`crate::mamba3_siso::route`]).
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
    /// Epsilon of the block norms (`norm`, `norm2`, `norm_f`); the
    /// reference's `norm_epsilon`. A checkpoint trained with a different
    /// eps loads a DIFFERENT model — the value rides checkpoint metadata
    /// (see `serialize.rs`).
    pub rms_norm_eps: f32,
    /// Epsilon of the mixer's B/C norms and output-gate norm, which the
    /// reference fixes at 1e-5 whatever `norm_epsilon` says.
    pub ssm_norm_eps: f32,
    /// The gated MLP after the mixer in every layer (`d_intermediate` in
    /// the reference config); 0 means no MLP. The hidden width is this
    /// value rounded up to a multiple of 128 (see [`Self::mlp_hidden`]).
    pub d_intermediate: usize,
}

impl Mamba3Config {
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

    /// Validate the model envelope: every shape the reference accepts.
    ///
    /// The CPU runs every config that passes. The GPU kernels have narrower
    /// limits of their own (head width, shared memory, the fused output
    /// norm), checked by [`crate::mamba3_siso::route`] at each GPU entry
    /// point.
    ///
    /// Returns `Err` with a description on invalid configuration — same
    /// contract as [`crate::mamba1::config::MambaConfig::validate`] (the old
    /// panicking variant made library constructors abort instead of
    /// surfacing a recoverable error).
    pub fn validate(&self) -> Result<(), String> {
        // Checked first: nheads() divides by it.
        if self.headdim < 1 {
            return Err("headdim must be >= 1".into());
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
        if self.ngroups < 1 {
            return Err("ngroups must be >= 1".into());
        }
        if self.d_model > envelope::MAX_WIDTH {
            return Err(format!("d_model ({}) must be <= 2^20", self.d_model));
        }
        if self.d_intermediate > envelope::MAX_WIDTH {
            return Err(format!(
                "d_intermediate ({}) must be <= 2^20",
                self.d_intermediate
            ));
        }
        if self.n_layers > envelope::MAX_LAYERS {
            return Err(format!("n_layers ({}) must be <= 4096", self.n_layers));
        }
        self.d_model
            .checked_mul(self.expand)
            .ok_or("expand * d_model overflows usize")?;
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
        // The reference clamps A at -A_floor; zero is legal and never
        // clamps, as for MIMO.
        if !self.a_floor.is_finite() || self.a_floor < 0.0 {
            return Err(format!(
                "a_floor must be finite and non-negative, got {}",
                self.a_floor
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
        envelope::check_param_bytes(self.layer_params(), self.n_layers)
    }

    /// One layer's parameters, exact in u128 (`None` past it): the norms,
    /// the in and out projections, dt bias, the B and C norms and biases,
    /// D, the gate norm and the gated MLP.
    pub(crate) fn layer_params(&self) -> Option<u128> {
        let w = |v: usize| v as u128;
        let (d, ds, ng) = (w(self.d_model), w(self.d_state), w(self.ngroups));
        let di = d.checked_mul(w(self.expand))?;
        let nh = di.checked_div(w(self.headdim))?;
        let na = w(self.num_rope_angles());
        let ip = envelope::sum_of_products(&[&[2, di], &[2, ng, ds], &[3, nh], &[na]])?;
        let h = w(self.mlp_hidden());
        envelope::sum_of_products(&[
            &[d],
            &[d, ip],
            &[2, nh],
            &[2, ds],
            &[2, nh, ds],
            &[di],
            &[di, d],
            &[if h > 0 { d } else { 0 }],
            &[3, d, h],
        ])
    }

    /// Resolve the training-time scan choice from `scan_mode`.
    ///
    /// M3 semantics differ from Mamba-1's length-threshold `Auto`: training
    /// takes the chunked parallel scan regardless of T, because the
    /// sequential scan's activation tape costs about the chunk length times
    /// more memory. `Sequential` runs the sequential scan in every lane.
    pub fn train_use_parallel_scan(&self) -> bool {
        !matches!(self.scan_mode, ScanMode::Sequential)
    }

    /// Resolve the prompt window's scan choice from `scan_mode`, as for
    /// training: the chunked scan in every mode but `Sequential`, so `Auto`
    /// keeps the chunked prompt at every length.
    pub fn prefill_use_parallel_scan(&self) -> bool {
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
            ssm_norm_eps: 1e-5,
            d_intermediate: 0,
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

    /// The official checkpoints' head width is a valid model: the envelope
    /// asks only that headdim divide d_inner, as the reference does.
    #[test]
    fn test_official_headdim_valid() {
        for (hd, ds) in [(64usize, 128usize), (48, 64), (128, 256), (1, 16)] {
            Mamba3Config {
                d_model: 768,
                headdim: hd,
                d_state: ds,
                ..Mamba3Config::default()
            }
            .validate()
            .unwrap_or_else(|e| panic!("headdim {hd} d_state {ds} must validate: {e}"));
        }
        let err = Mamba3Config {
            headdim: 0,
            ..Mamba3Config::default()
        }
        .validate()
        .unwrap_err();
        assert!(err.contains("headdim"), "{err}");
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
    /// `validate` never panics, and every config it passes derives its
    /// widths and tensor sizes exactly, each tensor within one allocation.
    #[test]
    fn validate_passes_only_exact_widths() {
        use crate::config::envelope_sweep::{FLOATS, sweep};
        let fields: [fn(&mut Mamba3Config, usize); 7] = [
            |c, v| c.d_model = v,
            |c, v| c.d_state = v,
            |c, v| c.expand = v,
            |c, v| c.headdim = v,
            |c, v| c.ngroups = v,
            |c, v| c.n_layers = v,
            |c, v| c.d_intermediate = v,
        ];
        sweep(Mamba3Config::default(), &fields, |c| {
            if c.validate().is_err() {
                return;
            }
            let (d, di, ip) = (c.d_model, c.d_inner(), c.in_proj_out_dim());
            let _ = (c.nheads(), c.num_rope_angles());
            assert!(c.layer_params().is_some());
            for (a, b) in [
                (d, ip),
                (di, d),
                (c.nheads(), c.d_state),
                (d, 2 * c.mlp_hidden()),
            ] {
                let n = a.saturating_mul(b);
                assert!(n <= isize::MAX as usize / 4, "{c:?}");
            }
        });
        for v in FLOATS {
            let a_floor = Mamba3Config {
                a_floor: v,
                ..Mamba3Config::default()
            };
            assert_eq!(a_floor.validate().is_ok(), v.is_finite() && v >= 0.0, "{v}");
            for eps in [
                Mamba3Config {
                    rms_norm_eps: v,
                    ..Mamba3Config::default()
                },
                Mamba3Config {
                    ssm_norm_eps: v,
                    ..Mamba3Config::default()
                },
            ] {
                assert_eq!(eps.validate().is_ok(), v.is_finite() && v > 0.0, "{v}");
            }
            let rope = Mamba3Config {
                rope_fraction: v,
                ..Mamba3Config::default()
            };
            assert_eq!(rope.validate().is_ok(), v == 0.5 || v == 1.0, "{v}");
        }
    }

    /// The parameters `validate` bounds are the ones the weights hold.
    #[test]
    fn layer_params_count_every_tensor() {
        let odd = Mamba3Config {
            d_model: 40,
            d_state: 6,
            headdim: 20,
            ngroups: 2,
            n_layers: 2,
            rope_fraction: 1.0,
            d_intermediate: 100,
            ..Mamba3Config::default()
        };
        for cfg in [Mamba3Config::default(), odd] {
            let input_dim = 5;
            let w = crate::mamba3_siso::weights::Mamba3Weights::zeros(&cfg, input_dim);
            let (list, _) = crate::mamba3_siso::serialize::checkpoint_parts(&w, &cfg, input_dim)
                .expect("the weights match their config");
            let outer = (input_dim * cfg.d_model + 2 * cfg.d_model) as u128;
            let layers = cfg.n_layers as u128 * cfg.layer_params().expect("a small model");
            assert_eq!(list.elements() as u128, outer + layers, "{cfg:?}");
        }
    }
}
