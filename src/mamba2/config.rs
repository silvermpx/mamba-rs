//! Mamba-2 configuration.
//!
//! Source: Dao & Gu, "Transformers are SSMs", ICML 2024 (arXiv 2405.21060);
//! the reference is state-spaces/mamba `mamba_ssm/modules/mamba2.py`.

use crate::config::ScanMode;

/// Configuration for a Mamba-2 backbone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mamba2Config {
    /// Model (embedding) dimension.
    pub d_model: usize,
    /// SSM state dimension per head.
    pub d_state: usize,
    /// Causal conv1d width.
    pub d_conv: usize,
    /// Inner dimension expansion factor. d_inner = d_model * expand.
    pub expand: usize,
    /// Per-head dimension; any width that divides `d_ssm`.
    pub headdim: usize,
    /// Channels the SSM runs on; 0 means all of `d_inner`. The rest of
    /// `d_inner` is a gated MLP inside the mixer, `silu(z0) * x0`.
    pub d_ssm: usize,
    /// Number of B/C groups (nheads must be divisible by ngroups); also the
    /// number of groups of the gated norm.
    pub ngroups: usize,
    /// Number of layers.
    pub n_layers: usize,
    /// D per channel (`d_ssm` values) instead of per head.
    pub d_has_hdim: bool,
    /// Gated RMSNorm before out_proj; without it the output is `y * silu(z)`.
    pub rmsnorm: bool,
    /// Normalise y before the gate instead of after it.
    pub norm_before_gate: bool,
    /// dt is clamped to [lo, hi] after the softplus.
    pub dt_limit: (f32, f32),
    /// in_proj and out_proj carry a bias.
    pub proj_bias: bool,
    /// The conv carries a bias.
    pub conv_bias: bool,
    /// Epsilon of the block norms (`norm`, `norm2`, `norm_f`).
    pub rms_norm_eps: f32,
    /// Epsilon of the mixer's gated norm, which the reference fixes at 1e-5.
    pub gated_norm_eps: f32,
    /// The gated MLP after the mixer in every layer; 0 means no MLP.
    pub d_intermediate: usize,
    /// Which scan the GPU runs over a window, as for Mamba-1 and Mamba-3:
    /// the sequential recurrence (the decode step's formulas token by
    /// token) or the chunked state space dual. See
    /// [`Self::prefill_parallel`] and [`Self::train_use_parallel_scan`].
    /// The CPU always runs the recurrence.
    pub scan_mode: ScanMode,
}

impl Mamba2Config {
    /// Whether a prompt window of `seq_len` tokens runs the chunked scan:
    /// `Sequential` and `Parallel` always as named (the sequential kernels
    /// serve every d_state), `Auto` the chunked scan above
    /// [`ScanMode::PARALLEL_SCAN_THRESHOLD`] tokens.
    pub fn prefill_parallel(&self, seq_len: usize) -> bool {
        match self.scan_mode {
            ScanMode::Sequential => false,
            ScanMode::Parallel => true,
            ScanMode::Auto => seq_len > ScanMode::PARALLEL_SCAN_THRESHOLD,
        }
    }

    /// Whether training runs the chunked scan: every mode but `Sequential`,
    /// as Mamba-3 resolves it (training windows are long).
    pub fn train_use_parallel_scan(&self) -> bool {
        !matches!(self.scan_mode, ScanMode::Sequential)
    }

    pub fn d_inner(&self) -> usize {
        self.d_model * self.expand
    }

    /// The SSM channels: `d_ssm`, or all of `d_inner` when it is 0.
    pub fn ssm_dim(&self) -> usize {
        if self.d_ssm == 0 {
            self.d_inner()
        } else {
            self.d_ssm
        }
    }

    /// Width of each half of the mixer's own MLP, `d_inner - d_ssm`.
    pub fn d_mlp(&self) -> usize {
        self.d_inner() - self.ssm_dim()
    }

    pub fn nheads(&self) -> usize {
        self.ssm_dim() / self.headdim
    }

    /// Conv channels: x, B and C.
    pub fn conv_dim(&self) -> usize {
        self.ssm_dim() + 2 * self.ngroups * self.d_state
    }

    /// in_proj output, split as `[z0 | x0 | z | xBC | dt]`.
    pub fn in_proj_out_dim(&self) -> usize {
        2 * self.d_mlp() + self.ssm_dim() + self.conv_dim() + self.nheads()
    }

    /// The gated MLP's hidden width after the mixer, 0 without one.
    pub fn mlp_hidden(&self) -> usize {
        crate::mlp::mlp_hidden(self.d_intermediate)
    }

    /// Validate the model envelope: the shapes the reference accepts.
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model < 1 || self.expand < 1 || self.n_layers < 1 {
            return Err("d_model, expand and n_layers must be >= 1".into());
        }
        if self.ssm_dim() > self.d_inner() {
            return Err(format!(
                "d_ssm ({}) must not exceed d_inner ({})",
                self.d_ssm,
                self.d_inner()
            ));
        }
        if self.headdim < 1 || !self.ssm_dim().is_multiple_of(self.headdim) {
            return Err(format!(
                "d_ssm ({}) must be a multiple of headdim ({})",
                self.ssm_dim(),
                self.headdim
            ));
        }
        if self.ngroups < 1 || !self.nheads().is_multiple_of(self.ngroups) {
            return Err(format!(
                "nheads ({}) must be a multiple of ngroups ({})",
                self.nheads(),
                self.ngroups
            ));
        }
        if self.d_state < 1 || self.d_state > 1024 {
            return Err(format!("d_state ({}) must be in 1..=1024", self.d_state));
        }
        if self.d_conv < 1 || self.d_conv > 64 {
            return Err(format!("d_conv ({}) must be in 1..=64", self.d_conv));
        }
        let (lo, hi) = self.dt_limit;
        if lo.is_nan() || hi.is_nan() || lo < 0.0 || lo > hi {
            return Err(format!("dt_limit ({lo}, {hi}) must satisfy 0 <= lo <= hi"));
        }
        for (name, eps) in [
            ("rms_norm_eps", self.rms_norm_eps),
            ("gated_norm_eps", self.gated_norm_eps),
        ] {
            if !eps.is_finite() || eps <= 0.0 {
                return Err(format!("{name} must be positive, got {eps}"));
            }
        }
        if self.d_intermediate > (1 << 20) || self.d_model > (1 << 20) {
            return Err("d_model and d_intermediate must be <= 2^20".into());
        }
        Ok(())
    }
}

impl Default for Mamba2Config {
    /// A small model with the reference's defaults: rmsnorm with the gate
    /// first, a conv bias, no projection bias, an open dt limit.
    fn default() -> Self {
        Self {
            d_model: 128,
            d_state: 32,
            d_conv: 4,
            expand: 2,
            headdim: 32,
            d_ssm: 0,
            ngroups: 1,
            n_layers: 4,
            d_has_hdim: false,
            rmsnorm: true,
            norm_before_gate: false,
            dt_limit: (0.0, f32::INFINITY),
            proj_bias: false,
            conv_bias: true,
            rms_norm_eps: 1e-5,
            gated_norm_eps: 1e-5,
            d_intermediate: 0,
            scan_mode: ScanMode::Auto,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released mamba2-130m shape: in_proj 3352 wide, conv 1792.
    #[test]
    fn released_shape() {
        let cfg = Mamba2Config {
            d_model: 768,
            d_state: 128,
            headdim: 64,
            n_layers: 24,
            ..Mamba2Config::default()
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.in_proj_out_dim(), 3352);
        assert_eq!(cfg.conv_dim(), 1792);
        assert_eq!(cfg.nheads(), 24);
    }

    #[test]
    fn invalid_shapes_are_refused() {
        let base = Mamba2Config::default();
        for bad in [
            Mamba2Config {
                headdim: 24,
                ..base
            },
            Mamba2Config { ngroups: 3, ..base },
            Mamba2Config { d_ssm: 512, ..base },
            Mamba2Config {
                dt_limit: (0.2, 0.1),
                ..base
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }
}
