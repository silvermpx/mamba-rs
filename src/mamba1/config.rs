//! Mamba-1 model configuration.

use crate::config::{ScanMode, envelope};

/// Configuration for a Mamba SSM model.
///
/// Source: Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
/// Paper: <https://arxiv.org/abs/2312.00752>
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MambaConfig {
    /// Model dimension — input features projected to this size.
    pub d_model: usize,

    /// SSM state dimension. Controls memory capacity per channel.
    /// Paper default: 16.
    pub d_state: usize,

    /// Local convolution width before SSM. Paper default: 4.
    pub d_conv: usize,

    /// Expansion factor. `d_inner = expand * d_model`.
    /// Paper default: 2.
    pub expand: usize,

    /// Number of stacked Mamba layers. Paper default: varies by model size.
    pub n_layers: usize,

    /// GPU SSM scan mode. Default: Auto (see [`ScanMode`]).
    /// Only affects GPU training forward/backward. CPU always uses sequential.
    pub scan_mode: ScanMode,

    /// RMSNorm epsilon for every layer norm and the final norm_f.
    /// Default 1e-5 (state-spaces/mamba). FalconMamba checkpoints use 1e-6 —
    /// the HF loader plumbs the checkpoint's value through here.
    pub rms_norm_eps: f32,
}

impl MambaConfig {
    /// Inner dimension (expanded). Used for in_proj, SSM, conv1d.
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    /// dt_rank = ceil(d_model / 16). Controls delta projection bottleneck.
    pub fn dt_rank(&self) -> usize {
        self.d_model.div_ceil(16)
    }

    /// x_proj output dimension: dt_rank + 2 * d_state (for delta, B, C).
    pub fn xdbl_dim(&self) -> usize {
        self.dt_rank() + 2 * self.d_state
    }
}

impl MambaConfig {
    /// Validate configuration constraints.
    ///
    /// Returns `Err` for a zero dimension, one past the bounds every family
    /// keeps (`d_model` at most 2^20, at most 4096 layers, the layers'
    /// f32 parameters within one allocation), or one past what the kernels
    /// or the reference serve, before any width is derived from it.
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model == 0 {
            return Err("d_model must be > 0".into());
        }
        if self.d_state == 0 {
            return Err("d_state must be > 0".into());
        }
        if self.d_conv == 0 {
            return Err("d_conv must be > 0".into());
        }
        if self.expand == 0 {
            return Err("expand must be > 0".into());
        }
        if self.n_layers == 0 {
            return Err("n_layers must be > 0".into());
        }
        if self.d_model > envelope::MAX_WIDTH {
            return Err(format!("d_model ({}) must be <= 2^20", self.d_model));
        }
        if self.n_layers > envelope::MAX_LAYERS {
            return Err(format!("n_layers ({}) must be <= 4096", self.n_layers));
        }
        self.expand
            .checked_mul(self.d_model)
            .ok_or("expand * d_model overflows usize")?;
        // 256 is the reference implementations' own maximum. The
        // sequential SSM kernels size their per-thread state arrays at
        // JIT time from the model config (past ~128 the compiler spills
        // to local memory, correct and slower). `ScanMode::Auto` takes
        // the parallel scan above 64 states, the faster route there.
        if self.d_state > 256 {
            return Err(format!(
                "d_state ({}) must be <= 256 (CUDA parallel scan MAX_DSTATE limit)",
                self.d_state
            ));
        }
        // The bound the Mamba-2 config keeps too.
        if self.d_conv > 64 {
            return Err(format!("d_conv ({}) must be <= 64", self.d_conv));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(format!(
                "rms_norm_eps must be positive, got {}",
                self.rms_norm_eps
            ));
        }
        envelope::check_param_bytes(self.layer_params(), self.n_layers)
    }

    /// One layer's parameters, exact in u128 (`None` past it): the norm,
    /// the in, x, dt and out projections, the conv, A and D.
    pub(crate) fn layer_params(&self) -> Option<u128> {
        let w = |v: usize| v as u128;
        let (d, ds, dc) = (w(self.d_model), w(self.d_state), w(self.d_conv));
        let di = d.checked_mul(w(self.expand))?;
        let dr = w(self.d_model.div_ceil(16));
        let xd = dr.checked_add(ds.checked_mul(2)?)?;
        envelope::sum_of_products(&[
            &[d],
            &[d, 2, di],
            &[di, dc],
            &[di],
            &[di, xd],
            &[dr, di],
            &[di],
            &[di, ds],
            &[di],
            &[di, d],
        ])
    }
}

/// Paper defaults: d_model=128, d_state=16, d_conv=4, expand=2, n_layers=3.
impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            d_model: 128,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            n_layers: 3,
            scan_mode: ScanMode::Auto,
            rms_norm_eps: crate::ops::fast_math::RMS_NORM_EPS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::envelope_sweep::{FLOATS, sweep};

    /// `validate` never panics, and every config it passes derives its
    /// widths and tensor sizes exactly, each tensor within one allocation.
    #[test]
    fn validate_passes_only_exact_widths() {
        let fields: [fn(&mut MambaConfig, usize); 5] = [
            |c, v| c.d_model = v,
            |c, v| c.d_state = v,
            |c, v| c.d_conv = v,
            |c, v| c.expand = v,
            |c, v| c.n_layers = v,
        ];
        sweep(MambaConfig::default(), &fields, |c| {
            if c.validate().is_err() {
                return;
            }
            let (d, di, xd) = (c.d_model, c.d_inner(), c.xdbl_dim());
            assert!(c.layer_params().is_some());
            for (a, b) in [
                (2 * di, d),
                (di, xd),
                (c.dt_rank(), di),
                (di, c.d_state),
                (di, c.d_conv),
            ] {
                let n = a.saturating_mul(b);
                assert!(n <= isize::MAX as usize / 4, "{c:?}");
            }
        });
        for eps in FLOATS {
            let c = MambaConfig {
                rms_norm_eps: eps,
                ..MambaConfig::default()
            };
            assert_eq!(c.validate().is_ok(), eps.is_finite() && eps > 0.0, "{eps}");
        }
    }

    /// The parameters `validate` bounds are the ones the weights hold.
    #[test]
    fn layer_params_count_every_tensor() {
        let odd = MambaConfig {
            d_model: 33,
            d_state: 7,
            d_conv: 3,
            expand: 3,
            n_layers: 2,
            ..MambaConfig::default()
        };
        for cfg in [MambaConfig::default(), odd] {
            let input_dim = 5;
            let w = crate::mamba1::weights::MambaWeights::zeros(&cfg, input_dim);
            let (list, _) = crate::mamba1::serialize::checkpoint_parts(&w, &cfg, input_dim)
                .expect("the weights match their config");
            let outer = (input_dim * cfg.d_model + 2 * cfg.d_model) as u128;
            let layers = cfg.n_layers as u128 * cfg.layer_params().expect("a small model");
            assert_eq!(list.elements() as u128, outer + layers, "{cfg:?}");
        }
    }
}
