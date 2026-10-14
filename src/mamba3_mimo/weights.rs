//! Weight storage for Mamba-3 MIMO layers.
//!
//! Flat row-major `Vec<f32>` tensors, projections in the `[in, out]`
//! layout every backbone here uses.

use super::config::Mamba3MimoConfig;
use crate::mlp::GatedMlpLayerWeights;
use crate::ops::init::{SimpleRng, dt_bias_init, linear_default_uniform};

/// Weights of one Mamba-3 MIMO layer's mixer.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba3MimoLayerWeights {
    /// The block norm's scale before the mixer: `[d_model]`.
    pub norm_weight: Vec<f32>,
    /// The block LayerNorm's bias: `[d_model]`, empty for RMSNorm.
    pub norm_bias: Vec<f32>,
    /// in_proj weight `[d_model, in_proj_dim]`, split as
    /// `[z, x, B, C, dd_dt, dd_A, trap, angles]`.
    pub in_proj_w: Vec<f32>,
    /// DT bias: `[nheads]`, added before softplus.
    pub dt_bias: Vec<f32>,
    /// B norm scale `[d_state]`, shared by every rank and group.
    pub b_norm_weight: Vec<f32>,
    /// C norm scale `[d_state]`, shared by every rank and group.
    pub c_norm_weight: Vec<f32>,
    /// B bias `[nheads, mimo_rank, d_state]`, added after the norm.
    pub b_bias: Vec<f32>,
    /// C bias `[nheads, mimo_rank, d_state]`, added after the norm.
    pub c_bias: Vec<f32>,
    /// D skip `[nheads]`, applied to the rank-expanded input.
    pub d_param: Vec<f32>,
    /// The input's per-rank projection `[nheads, mimo_rank, headdim]`.
    pub mimo_x: Vec<f32>,
    /// The gate's per-rank projection `[nheads, mimo_rank, headdim]`.
    pub mimo_z: Vec<f32>,
    /// The output's per-rank projection `[nheads, mimo_rank, headdim]`.
    pub mimo_o: Vec<f32>,
    /// RMSNormGated scale `[d_inner]`, used only with `is_outproj_norm`.
    pub norm_gate_weight: Vec<f32>,
    /// out_proj weight `[d_inner, d_model]`.
    pub out_proj_w: Vec<f32>,
}

/// Weights of a whole Mamba-3 MIMO backbone.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba3MimoWeights {
    /// Input projection `[input_dim, d_model]`; empty with an empty bias
    /// means the identity (a language model feeds its embedding in).
    pub input_proj_w: Vec<f32>,
    /// Input projection bias `[d_model]`.
    pub input_proj_b: Vec<f32>,
    pub layers: Vec<Mamba3MimoLayerWeights>,
    /// The final norm's scale `[d_model]`.
    pub norm_f_weight: Vec<f32>,
    /// The final LayerNorm's bias `[d_model]`, empty for RMSNorm.
    pub norm_f_bias: Vec<f32>,
    /// The gated MLP of each layer; empty without one.
    pub mlp: Vec<GatedMlpLayerWeights>,
}

impl Mamba3MimoLayerWeights {
    /// Zeroed projections with the reference's constant inits: unit norms,
    /// D and gate projection, biases of one, and the input and output
    /// projections at 1/R so a fresh layer averages its ranks.
    pub fn zeros(cfg: &Mamba3MimoConfig) -> Self {
        let (d, di, ds, nh, r, p) = (
            cfg.d_model,
            cfg.d_inner(),
            cfg.d_state,
            cfg.nheads(),
            cfg.mimo_rank,
            cfg.headdim,
        );
        let inv_r = 1.0 / r as f32;
        Self {
            norm_weight: vec![1.0; d],
            norm_bias: vec![0.0; cfg.block_norm.bias_len(d)],
            in_proj_w: vec![0.0; d * cfg.in_proj_out_dim()],
            dt_bias: vec![0.0; nh],
            b_norm_weight: vec![1.0; ds],
            c_norm_weight: vec![1.0; ds],
            b_bias: vec![1.0; nh * r * ds],
            c_bias: vec![1.0; nh * r * ds],
            d_param: vec![1.0; nh],
            mimo_x: vec![inv_r; nh * r * p],
            mimo_z: vec![1.0; nh * r * p],
            mimo_o: vec![inv_r; nh * r * p],
            norm_gate_weight: vec![1.0; di],
            out_proj_w: vec![0.0; di * d],
        }
    }
}

impl Mamba3MimoWeights {
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

    /// Allocate weights with the constant inits and zeroed projections.
    pub fn zeros(cfg: &Mamba3MimoConfig, input_dim: usize) -> Self {
        let d = cfg.d_model;
        Self {
            input_proj_w: vec![0.0; input_dim * d],
            input_proj_b: vec![0.0; d],
            layers: (0..cfg.n_layers)
                .map(|_| Mamba3MimoLayerWeights::zeros(cfg))
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

    /// The reference init: `nn.Linear` defaults for the projections, the
    /// prenorm-residual rescale on out_proj and fc2, dt_bias from the
    /// reference's log-uniform dt, and the constants of [`Self::zeros`].
    pub fn init(cfg: &Mamba3MimoConfig, input_dim: usize, seed: u64) -> Self {
        let mut w = Self::zeros(cfg, input_dim);
        let mut rng = SimpleRng::new(seed);
        let (d, di) = (cfg.d_model, cfg.d_inner());
        let residuals_per_layer = if w.mlp.is_empty() { 1 } else { 2 };
        let rescale = 1.0 / ((residuals_per_layer * cfg.n_layers) as f64).sqrt() as f32;

        linear_default_uniform(&mut w.input_proj_w, input_dim, &mut rng);
        for lw in &mut w.layers {
            linear_default_uniform(&mut lw.in_proj_w, d, &mut rng);
            linear_default_uniform(&mut lw.out_proj_w, di, &mut rng);
            for v in &mut lw.out_proj_w {
                *v *= rescale;
            }
            dt_bias_init(&mut lw.dt_bias, &mut rng);
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
    pub fn validate(&self, cfg: &Mamba3MimoConfig, input_dim: usize) -> Result<(), String> {
        cfg.validate()?;
        let (d, di, ds, nh, r, p) = (
            cfg.d_model,
            cfg.d_inner(),
            cfg.d_state,
            cfg.nheads(),
            cfg.mimo_rank,
            cfg.headdim,
        );
        let ck = |name: &str, actual: usize, expected: usize| -> Result<(), String> {
            if actual == expected {
                Ok(())
            } else {
                Err(format!("{name}: expected {expected}, got {actual}"))
            }
        };
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
        let norm_bias = cfg.block_norm.bias_len(d);
        ck("norm_f_weight", self.norm_f_weight.len(), d)?;
        ck("norm_f_bias", self.norm_f_bias.len(), norm_bias)?;
        ck("n_layers", self.layers.len(), cfg.n_layers)?;
        for (i, lw) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("layer[{i}].{s}");
            ck(&n("norm_weight"), lw.norm_weight.len(), d)?;
            ck(&n("norm_bias"), lw.norm_bias.len(), norm_bias)?;
            ck(
                &n("in_proj_w"),
                lw.in_proj_w.len(),
                d * cfg.in_proj_out_dim(),
            )?;
            ck(&n("dt_bias"), lw.dt_bias.len(), nh)?;
            ck(&n("b_norm_weight"), lw.b_norm_weight.len(), ds)?;
            ck(&n("c_norm_weight"), lw.c_norm_weight.len(), ds)?;
            ck(&n("b_bias"), lw.b_bias.len(), nh * r * ds)?;
            ck(&n("c_bias"), lw.c_bias.len(), nh * r * ds)?;
            ck(&n("d_param"), lw.d_param.len(), nh)?;
            ck(&n("mimo_x"), lw.mimo_x.len(), nh * r * p)?;
            ck(&n("mimo_z"), lw.mimo_z.len(), nh * r * p)?;
            ck(&n("mimo_o"), lw.mimo_o.len(), nh * r * p)?;
            ck(&n("norm_gate_weight"), lw.norm_gate_weight.len(), di)?;
            ck(&n("out_proj_w"), lw.out_proj_w.len(), di * d)?;
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
        let cfg = Mamba3MimoConfig {
            d_intermediate: 100,
            ..Mamba3MimoConfig::default()
        };
        let a = Mamba3MimoWeights::init(&cfg, 64, 3);
        a.validate(&cfg, 64).unwrap();
        assert_eq!(a, Mamba3MimoWeights::init(&cfg, 64, 3));
        assert!(a.layers[0].dt_bias.iter().all(|v| v.is_finite()));
        assert_eq!(a.layers[0].mimo_x[0], 0.25);
    }
}
