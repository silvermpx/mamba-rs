//! Weight storage for Mamba-2 layers: flat row-major `Vec<f32>` tensors,
//! projections in the `[in, out]` layout every backbone here uses.

use super::config::Mamba2Config;
use crate::mlp::GatedMlpLayerWeights;
use crate::ops::init::{SimpleRng, dt_bias_init, linear_default_uniform};

/// Weights of one Mamba-2 layer's mixer.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba2LayerWeights {
    /// The block norm's scale before the mixer: `[d_model]`.
    pub norm_weight: Vec<f32>,
    /// The block LayerNorm's bias: `[d_model]`, empty for RMSNorm.
    pub norm_bias: Vec<f32>,
    /// in_proj `[d_model, in_proj_dim]`, split `[z0 | x0 | z | xBC | dt]`.
    pub in_proj_w: Vec<f32>,
    /// in_proj bias `[in_proj_dim]`, empty without `proj_bias`.
    pub in_proj_b: Vec<f32>,
    /// Depthwise conv weights `[conv_dim, d_conv]`, oldest tap first.
    pub conv_w: Vec<f32>,
    /// Conv bias `[conv_dim]`, empty without `conv_bias`.
    pub conv_b: Vec<f32>,
    /// dt bias `[nheads]`, added before the softplus.
    pub dt_bias: Vec<f32>,
    /// A = -exp(a_log), `[nheads]`.
    pub a_log: Vec<f32>,
    /// D skip, `[nheads]` or `[d_ssm]` with `d_has_hdim`.
    pub d_param: Vec<f32>,
    /// Gated RMSNorm scale `[d_ssm]`, empty without `rmsnorm`.
    pub gate_norm_weight: Vec<f32>,
    /// out_proj `[d_inner, d_model]`.
    pub out_proj_w: Vec<f32>,
    /// out_proj bias `[d_model]`, empty without `proj_bias`.
    pub out_proj_b: Vec<f32>,
}

/// Weights of a whole Mamba-2 backbone.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba2Weights {
    /// Input projection `[input_dim, d_model]`; empty with an empty bias
    /// means the identity (a language model feeds its embedding in).
    pub input_proj_w: Vec<f32>,
    pub input_proj_b: Vec<f32>,
    pub layers: Vec<Mamba2LayerWeights>,
    /// The final norm's scale `[d_model]`.
    pub norm_f_weight: Vec<f32>,
    /// The final LayerNorm's bias `[d_model]`, empty for RMSNorm.
    pub norm_f_bias: Vec<f32>,
    /// The gated MLP of each layer; empty without one.
    pub mlp: Vec<GatedMlpLayerWeights>,
}

impl Mamba2LayerWeights {
    /// Zeroed projections and the reference's constants: unit norms with
    /// zero biases (`nn.LayerNorm`'s init) and D, A = -1.
    pub fn zeros(cfg: &Mamba2Config) -> Self {
        let (d, di, ip, cd, nh) = (
            cfg.d_model,
            cfg.d_inner(),
            cfg.in_proj_out_dim(),
            cfg.conv_dim(),
            cfg.nheads(),
        );
        let bias = |n: usize, on: bool| if on { vec![0.0; n] } else { Vec::new() };
        Self {
            norm_weight: vec![1.0; d],
            norm_bias: vec![0.0; cfg.block_norm.bias_len(d)],
            in_proj_w: vec![0.0; d * ip],
            in_proj_b: bias(ip, cfg.proj_bias),
            conv_w: vec![0.0; cd * cfg.d_conv],
            conv_b: bias(cd, cfg.conv_bias),
            dt_bias: vec![0.0; nh],
            a_log: vec![0.0; nh],
            d_param: vec![1.0; if cfg.d_has_hdim { cfg.ssm_dim() } else { nh }],
            gate_norm_weight: if cfg.rmsnorm {
                vec![1.0; cfg.ssm_dim()]
            } else {
                Vec::new()
            },
            out_proj_w: vec![0.0; di * d],
            out_proj_b: bias(d, cfg.proj_bias),
        }
    }
}

impl Mamba2Weights {
    /// Round in place the matrices a reduced precision keeps as the GPU
    /// does: in_proj, out_proj and the MLP's projections in the half
    /// storage, and the input projection too as TF32 operands in the TF32
    /// lane (the half lanes keep it f32). f32 changes nothing.
    pub(crate) fn round_bulk(&mut self, dtype: crate::precision::WeightDtype) {
        use crate::ops::lane::round_matrix;
        for lw in &mut self.layers {
            round_matrix(dtype, &mut lw.in_proj_w);
            round_matrix(dtype, &mut lw.out_proj_w);
        }
        for m in &mut self.mlp {
            round_matrix(dtype, &mut m.fc1_w);
            round_matrix(dtype, &mut m.fc2_w);
        }
        if dtype == crate::precision::WeightDtype::Tf32 {
            round_matrix(dtype, &mut self.input_proj_w);
        }
    }

    pub fn zeros(cfg: &Mamba2Config, input_dim: usize) -> Self {
        let d = cfg.d_model;
        Self {
            input_proj_w: vec![0.0; input_dim * d],
            input_proj_b: vec![0.0; d],
            layers: (0..cfg.n_layers)
                .map(|_| Mamba2LayerWeights::zeros(cfg))
                .collect(),
            norm_f_weight: vec![1.0; d],
            norm_f_bias: vec![0.0; cfg.block_norm.bias_len(d)],
            mlp: match cfg.mlp_hidden() {
                0 => Vec::new(),
                hidden => (0..cfg.n_layers)
                    .map(|_| GatedMlpLayerWeights::zeros(d, hidden, cfg.block_norm))
                    .collect(),
            },
        }
    }

    /// The reference init: `nn.Linear` and `nn.Conv1d` defaults (bound
    /// 1/sqrt(fan_in), the conv's fan_in being its width), the
    /// prenorm-residual rescale on out_proj and fc2, dt_bias from the
    /// reference's log-uniform dt, A uniform in [1, 16].
    pub fn init(cfg: &Mamba2Config, input_dim: usize, seed: u64) -> Self {
        let mut w = Self::zeros(cfg, input_dim);
        let mut rng = SimpleRng::new(seed);
        let (d, di) = (cfg.d_model, cfg.d_inner());
        let residuals_per_layer = if w.mlp.is_empty() { 1 } else { 2 };
        let rescale = 1.0 / ((residuals_per_layer * cfg.n_layers) as f64).sqrt() as f32;

        linear_default_uniform(&mut w.input_proj_w, input_dim, &mut rng);
        for lw in &mut w.layers {
            linear_default_uniform(&mut lw.in_proj_w, d, &mut rng);
            // The reference model's own init zeroes every linear bias.
            lw.in_proj_b.fill(0.0);
            linear_default_uniform(&mut lw.conv_w, cfg.d_conv, &mut rng);
            linear_default_uniform(&mut lw.conv_b, cfg.d_conv, &mut rng);
            dt_bias_init(&mut lw.dt_bias, &mut rng);
            for a in &mut lw.a_log {
                *a = (1.0 + 15.0 * rng.next_f32()).ln();
            }
            linear_default_uniform(&mut lw.out_proj_w, di, &mut rng);
            for v in &mut lw.out_proj_w {
                *v *= rescale;
            }
            lw.out_proj_b.fill(0.0);
        }
        let hidden = cfg.mlp_hidden();
        for m in &mut w.mlp {
            linear_default_uniform(&mut m.fc1_w, d, &mut rng);
            linear_default_uniform(&mut m.fc2_w, hidden, &mut rng);
            for v in &mut m.fc2_w {
                *v *= rescale;
            }
        }
        w
    }

    /// Check every tensor's size against the config.
    pub fn validate(&self, cfg: &Mamba2Config, input_dim: usize) -> Result<(), String> {
        cfg.validate()?;
        let (d, di, ip, cd, nh) = (
            cfg.d_model,
            cfg.d_inner(),
            cfg.in_proj_out_dim(),
            cfg.conv_dim(),
            cfg.nheads(),
        );
        let ck = |name: &str, actual: usize, expected: usize| -> Result<(), String> {
            if actual == expected {
                Ok(())
            } else {
                Err(format!("{name}: expected {expected}, got {actual}"))
            }
        };
        let on = |n: usize, flag: bool| if flag { n } else { 0 };
        if self.input_proj_w.is_empty() && self.input_proj_b.is_empty() {
            ck("identity input width", input_dim, d)?;
        } else {
            ck(
                "input_proj_w",
                self.input_proj_w.len(),
                input_dim.saturating_mul(d),
            )?;
            ck("input_proj_b", self.input_proj_b.len(), d)?;
        }
        ck("norm_f_weight", self.norm_f_weight.len(), d)?;
        ck(
            "norm_f_bias",
            self.norm_f_bias.len(),
            cfg.block_norm.bias_len(d),
        )?;
        ck("n_layers", self.layers.len(), cfg.n_layers)?;
        for (i, lw) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("layer[{i}].{s}");
            ck(&n("norm_weight"), lw.norm_weight.len(), d)?;
            ck(
                &n("norm_bias"),
                lw.norm_bias.len(),
                cfg.block_norm.bias_len(d),
            )?;
            ck(&n("in_proj_w"), lw.in_proj_w.len(), d * ip)?;
            ck(&n("in_proj_b"), lw.in_proj_b.len(), on(ip, cfg.proj_bias))?;
            ck(&n("conv_w"), lw.conv_w.len(), cd * cfg.d_conv)?;
            ck(&n("conv_b"), lw.conv_b.len(), on(cd, cfg.conv_bias))?;
            ck(&n("dt_bias"), lw.dt_bias.len(), nh)?;
            ck(&n("a_log"), lw.a_log.len(), nh)?;
            let d_len = if cfg.d_has_hdim { cfg.ssm_dim() } else { nh };
            ck(&n("d_param"), lw.d_param.len(), d_len)?;
            ck(
                &n("gate_norm_weight"),
                lw.gate_norm_weight.len(),
                on(cfg.ssm_dim(), cfg.rmsnorm),
            )?;
            ck(&n("out_proj_w"), lw.out_proj_w.len(), di * d)?;
            ck(&n("out_proj_b"), lw.out_proj_b.len(), on(d, cfg.proj_bias))?;
        }
        match cfg.mlp_hidden() {
            0 => ck("mlp layers", self.mlp.len(), 0)?,
            hidden => {
                ck("mlp layers", self.mlp.len(), cfg.n_layers)?;
                for (i, m) in self.mlp.iter().enumerate() {
                    m.validate(d, hidden, cfg.block_norm, &format!("mlp[{i}]"))?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_valid_and_deterministic() {
        for cfg in [
            Mamba2Config::default(),
            Mamba2Config {
                proj_bias: true,
                conv_bias: false,
                d_has_hdim: true,
                rmsnorm: false,
                d_ssm: 192,
                d_intermediate: 64,
                block_norm: crate::config::BlockNorm::Layer,
                ..Mamba2Config::default()
            },
        ] {
            let a = Mamba2Weights::init(&cfg, 48, 9);
            a.validate(&cfg, 48).unwrap();
            assert_eq!(a, Mamba2Weights::init(&cfg, 48, 9));
            assert!(
                a.layers[0]
                    .a_log
                    .iter()
                    .all(|&v| (0.0..=16f32.ln()).contains(&v))
            );
        }
    }
}
