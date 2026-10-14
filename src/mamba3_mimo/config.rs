//! Mamba-3 MIMO configuration.
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026 (arXiv 2603.15569); the
//! reference module is state-spaces/mamba `mamba_ssm/modules/mamba3.py`
//! with `is_mimo=True`.

use crate::config::{BlockNorm, ScanMode, envelope};

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
    /// The kind of the block norms (`norm`, `norm2`, `norm_f`): RMSNorm,
    /// or LayerNorm with a bias for a model built with `rms_norm=False`.
    pub block_norm: BlockNorm,
    /// Whether the backbone ends in the final norm (`norm_f`), as every
    /// model and every official checkpoint does. A segment of a stack that
    /// continues in layers of the caller's own, such as attention between
    /// Mamba segments, is built without it: its output is the residual
    /// stream after its last block, which the next segment, built with an
    /// identity input, carries on.
    pub final_norm: bool,
    /// Whether the layers read the sequence in both directions, in pairs:
    /// layer 2k reads it forward and layer 2k+1 reads every sequence
    /// reversed, both from the pair's input, and the two mixers' outputs
    /// join the residual stream together, as the bidirectional stacks of
    /// Vim, Caduceus and burn-mamba's `BidiLayers` do. A pair's gated MLPs,
    /// when the model has them, follow its two mixers in layer order. A
    /// bidirectional model reads whole sequences: it has no decode step and
    /// carries no state from one window or prompt to the next. It needs an
    /// even number of layers.
    pub bidirectional: bool,
    /// Epsilon of the block norms (`norm`, `norm2`, `norm_f`), of either
    /// kind; the reference's `norm_epsilon`.
    pub rms_norm_eps: f32,
    /// Epsilon of the mixer's B/C norms and output-gate norm, which the
    /// reference fixes at 1e-5 whatever `norm_epsilon` says.
    pub ssm_norm_eps: f32,
    /// The gated MLP after the mixer in every layer; 0 means no MLP. The
    /// hidden width is this value rounded up to a multiple of 128.
    pub d_intermediate: usize,
    /// Which scan the GPU runs over a window, as for Mamba-1 and Mamba-3
    /// SISO: the sequential recurrence (the decode step token by token) or
    /// the chunked scan. See [`Self::prefill_parallel`] and
    /// [`Self::train_use_parallel_scan`]. The CPU always runs the
    /// recurrence.
    pub scan_mode: ScanMode,
}

/// The largest rank: each state row keeps one readout accumulator per
/// rank in registers. The reference warns outside {1, 2, 4, 8}.
pub const MAX_MIMO_RANK: usize = 16;

/// The largest d_state, the reference's own maximum.
pub const MAX_D_STATE: usize = 256;

impl Mamba3MimoConfig {
    /// Whether a prompt window of `seq_len` tokens runs the chunked scan:
    /// `Sequential` and `Parallel` as named, `Auto` the chunked scan above
    /// [`ScanMode::PARALLEL_SCAN_THRESHOLD`] tokens.
    pub fn prefill_parallel(&self, seq_len: usize) -> bool {
        match self.scan_mode {
            ScanMode::Sequential => false,
            ScanMode::Parallel => true,
            ScanMode::Auto => seq_len > ScanMode::PARALLEL_SCAN_THRESHOLD,
        }
    }

    /// Whether training runs the chunked scan: every mode but `Sequential`,
    /// as Mamba-3 SISO resolves it.
    pub fn train_use_parallel_scan(&self) -> bool {
        !matches!(self.scan_mode, ScanMode::Sequential)
    }

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

    /// Validate the model envelope: the shapes the reference accepts,
    /// inside the bounds every family keeps (`d_model` and
    /// `d_intermediate` at most 2^20, at most 4096 layers, the layers' f32
    /// parameters within one allocation). Every bound on a field is checked
    /// before a width is derived from it.
    pub fn validate(&self) -> Result<(), String> {
        if self.headdim < 1 {
            return Err("headdim must be >= 1".into());
        }
        if self.d_model < 1 || self.expand < 1 {
            return Err("d_model and expand must be >= 1".into());
        }
        if self.n_layers < 1 {
            return Err("n_layers must be >= 1".into());
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
        if self.bidirectional && !self.n_layers.is_multiple_of(2) {
            return Err(format!(
                "a bidirectional model's layers work in pairs: n_layers ({}) must be even",
                self.n_layers
            ));
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
        // The reference clamps A at -A_floor; zero is legal and never clamps.
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

    /// One layer's parameters, exact in u128 (`None` past it): the norms
    /// with a LayerNorm's bias, the in and out projections, dt bias, the B
    /// and C norms and biases, D, the rank projections, the gate norm and
    /// the gated MLP.
    pub(crate) fn layer_params(&self) -> Option<u128> {
        let w = |v: usize| v as u128;
        let (d, ds, r) = (w(self.d_model), w(self.d_state), w(self.mimo_rank));
        let di = d.checked_mul(w(self.expand))?;
        let nh = di.checked_div(w(self.headdim))?;
        let na = w(self.num_rope_angles());
        let ip =
            envelope::sum_of_products(&[&[2, di], &[2, r, w(self.ngroups), ds], &[3, nh], &[na]])?;
        let h = w(self.mlp_hidden());
        let norm_bias = |n: u128| if self.block_norm.rms_norm() { 0 } else { n };
        envelope::sum_of_products(&[
            &[d],
            &[norm_bias(d)],
            &[d, ip],
            &[2, nh],
            &[2, ds],
            &[2, nh, r, ds],
            &[3, r, di],
            &[di],
            &[di, d],
            &[if h > 0 { d } else { 0 }],
            &[if h > 0 { norm_bias(d) } else { 0 }],
            &[3, d, h],
        ])
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
            block_norm: BlockNorm::Rms,
            final_norm: true,
            bidirectional: false,
            rms_norm_eps: 1e-5,
            ssm_norm_eps: 1e-5,
            d_intermediate: 0,
            scan_mode: ScanMode::Auto,
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
    /// `validate` never panics, and every config it passes derives its
    /// widths and tensor sizes exactly, each tensor within one allocation.
    #[test]
    fn validate_passes_only_exact_widths() {
        use crate::config::envelope_sweep::{FLOATS, sweep};
        let fields: [fn(&mut Mamba3MimoConfig, usize); 8] = [
            |c, v| c.d_model = v,
            |c, v| c.d_state = v,
            |c, v| c.expand = v,
            |c, v| c.headdim = v,
            |c, v| c.ngroups = v,
            |c, v| c.n_layers = v,
            |c, v| c.mimo_rank = v,
            |c, v| c.d_intermediate = v,
        ];
        sweep(Mamba3MimoConfig::default(), &fields, |c| {
            if c.validate().is_err() {
                return;
            }
            let (d, di, ip) = (c.d_model, c.d_inner(), c.in_proj_out_dim());
            let _ = (c.nheads(), c.num_rope_angles(), c.bc_width());
            assert!(c.layer_params().is_some());
            for (a, b) in [(d, ip), (di, d), (c.mimo_rank, di), (d, 2 * c.mlp_hidden())] {
                let n = a.saturating_mul(b);
                assert!(n <= isize::MAX as usize / 4, "{c:?}");
            }
        });
        for v in FLOATS {
            let a_floor = Mamba3MimoConfig {
                a_floor: v,
                ..Mamba3MimoConfig::default()
            };
            assert_eq!(a_floor.validate().is_ok(), v.is_finite() && v >= 0.0, "{v}");
            for eps in [
                Mamba3MimoConfig {
                    rms_norm_eps: v,
                    ..Mamba3MimoConfig::default()
                },
                Mamba3MimoConfig {
                    ssm_norm_eps: v,
                    ..Mamba3MimoConfig::default()
                },
            ] {
                assert_eq!(eps.validate().is_ok(), v.is_finite() && v > 0.0, "{v}");
            }
        }
    }

    /// The parameters `validate` bounds are the ones the weights hold.
    #[test]
    fn layer_params_count_every_tensor() {
        let odd = Mamba3MimoConfig {
            d_model: 40,
            d_state: 6,
            headdim: 20,
            ngroups: 2,
            n_layers: 2,
            mimo_rank: 3,
            rope_fraction: 1.0,
            d_intermediate: 100,
            ..Mamba3MimoConfig::default()
        };
        let layer_norm = Mamba3MimoConfig {
            block_norm: BlockNorm::Layer,
            ..odd
        };
        for cfg in [Mamba3MimoConfig::default(), odd, layer_norm] {
            let input_dim = 5;
            let w = crate::mamba3_mimo::weights::Mamba3MimoWeights::zeros(&cfg, input_dim);
            let (list, _) = crate::mamba3_mimo::serialize::checkpoint_parts(&w, &cfg, input_dim)
                .expect("the weights match their config");
            let norm_f = cfg.d_model + cfg.block_norm.bias_len(cfg.d_model);
            let outer = (input_dim * cfg.d_model + cfg.d_model + norm_f) as u128;
            let layers = cfg.n_layers as u128 * cfg.layer_params().expect("a small model");
            assert_eq!(list.elements() as u128, outer + layers, "{cfg:?}");
        }
    }
}
