//! Mamba-1 model configuration.

use crate::config::{BlockNorm, ScanMode, envelope};

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

    /// Epsilon of the block norms (`norm`, `norm2`, `norm_f`), of either
    /// kind. Default 1e-5 (state-spaces/mamba). FalconMamba checkpoints use
    /// 1e-6 — the HF loader plumbs the checkpoint's value through here.
    pub rms_norm_eps: f32,

    /// FalconMamba's mixer norms: the dt, B and C parts of every x_proj row
    /// each divided by its root mean square, `v / sqrt(mean(v^2) + eps)`,
    /// with no weight, before dt_proj and the scan read them. `None` is
    /// Mamba-1; FalconMamba checkpoints carry 1e-6 (`mixer_rms_eps`).
    pub bcdt_rms_eps: Option<f32>,

    /// The rank of the dt projection: dt_proj is `[dt_rank, d_inner]` and
    /// x_proj's first `dt_rank` columns feed it. `None` is the reference's
    /// "auto", `ceil(d_model / 16)`.
    pub dt_rank: Option<usize>,

    /// Biases on in_proj and out_proj (`bias` in the reference, `use_bias`
    /// in transformers). Every official checkpoint has none.
    pub proj_bias: bool,

    /// A bias on the depthwise conv (`conv_bias`, `use_conv_bias`), as
    /// every official checkpoint has. Without one the conv has no bias to
    /// apply or train.
    pub conv_bias: bool,

    /// The gated MLP after the mixer in every layer (`d_intermediate` in
    /// the reference config); 0 means no MLP. The hidden width is this
    /// value rounded up to a multiple of 128 (see [`Self::mlp_hidden`]).
    pub d_intermediate: usize,
}

impl MambaConfig {
    /// Inner dimension (expanded). Used for in_proj, SSM, conv1d.
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    /// The gated MLP's hidden width, 0 when the model has none.
    pub fn mlp_hidden(&self) -> usize {
        crate::mlp::mlp_hidden(self.d_intermediate)
    }

    /// The dt projection's rank: [`Self::dt_rank`] when given, else
    /// `ceil(d_model / 16)`.
    pub fn dt_rank(&self) -> usize {
        self.dt_rank.unwrap_or(self.d_model.div_ceil(16))
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
        if self.d_intermediate > envelope::MAX_WIDTH {
            return Err(format!(
                "d_intermediate ({}) must be <= 2^20",
                self.d_intermediate
            ));
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
        if let Some(eps) = self.bcdt_rms_eps
            && (!eps.is_finite() || eps <= 0.0)
        {
            return Err(format!("bcdt_rms_eps must be positive, got {eps}"));
        }
        if let Some(rank) = self.dt_rank
            && (rank == 0 || rank > envelope::MAX_WIDTH)
        {
            return Err(format!("dt_rank ({rank}) must be in 1..=2^20"));
        }
        envelope::check_param_bytes(self.layer_params(), self.n_layers)
    }

    /// One layer's parameters, exact in u128 (`None` past it): the norm
    /// and a LayerNorm's bias, the in, x, dt and out projections and their
    /// biases, the conv, A, D and the gated MLP with its norm.
    pub(crate) fn layer_params(&self) -> Option<u128> {
        let w = |v: usize| v as u128;
        let (d, ds, dc) = (w(self.d_model), w(self.d_state), w(self.d_conv));
        let di = d.checked_mul(w(self.expand))?;
        let dr = w(self.dt_rank());
        let xd = dr.checked_add(ds.checked_mul(2)?)?;
        let (proj_bias, conv_bias) = (
            w(usize::from(self.proj_bias)),
            w(usize::from(self.conv_bias)),
        );
        let h = w(self.mlp_hidden());
        let norm_bias = |n: u128| if self.block_norm.rms_norm() { 0 } else { n };
        envelope::sum_of_products(&[
            &[d],
            &[norm_bias(d)],
            &[d, 2, di],
            &[proj_bias, 2, di],
            &[di, dc],
            &[conv_bias, di],
            &[di, xd],
            &[dr, di],
            &[di],
            &[di, ds],
            &[di],
            &[di, d],
            &[proj_bias, d],
            &[if h > 0 { d } else { 0 }],
            &[if h > 0 { norm_bias(d) } else { 0 }],
            &[3, d, h],
        ])
    }

    /// The length of a layer's in_proj bias: `2 d_inner` with
    /// [`Self::proj_bias`], else none.
    pub fn in_proj_bias_len(&self) -> usize {
        if self.proj_bias {
            2 * self.d_inner()
        } else {
            0
        }
    }

    /// The length of a layer's out_proj bias: `d_model` with
    /// [`Self::proj_bias`], else none.
    pub fn out_proj_bias_len(&self) -> usize {
        if self.proj_bias { self.d_model } else { 0 }
    }

    /// The length of a layer's conv bias: `d_inner` with
    /// [`Self::conv_bias`], else none.
    pub fn conv_bias_len(&self) -> usize {
        if self.conv_bias { self.d_inner() } else { 0 }
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
            block_norm: BlockNorm::Rms,
            final_norm: true,
            rms_norm_eps: crate::ops::fast_math::RMS_NORM_EPS,
            bcdt_rms_eps: None,
            dt_rank: None,
            proj_bias: false,
            conv_bias: true,
            d_intermediate: 0,
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
        let fields: [fn(&mut MambaConfig, usize); 7] = [
            |c, v| c.d_model = v,
            |c, v| c.d_state = v,
            |c, v| c.d_conv = v,
            |c, v| c.expand = v,
            |c, v| c.n_layers = v,
            |c, v| c.dt_rank = Some(v),
            |c, v| c.d_intermediate = v,
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
                (d, 2 * c.mlp_hidden()),
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
        let variant = MambaConfig {
            dt_rank: Some(9),
            proj_bias: true,
            conv_bias: false,
            d_intermediate: 100,
            ..odd
        };
        let layer_norm = MambaConfig {
            block_norm: BlockNorm::Layer,
            ..variant
        };
        for cfg in [MambaConfig::default(), odd, variant, layer_norm] {
            let input_dim = 5;
            let w = crate::mamba1::weights::MambaWeights::zeros(&cfg, input_dim);
            let (list, _) = crate::mamba1::serialize::checkpoint_parts(&w, &cfg, input_dim)
                .expect("the weights match their config");
            let d = cfg.d_model;
            let outer = (input_dim * d + 2 * d + cfg.block_norm.bias_len(d)) as u128;
            let layers = cfg.n_layers as u128 * cfg.layer_params().expect("a small model");
            assert_eq!(list.elements() as u128, outer + layers, "{cfg:?}");
        }
    }
}
