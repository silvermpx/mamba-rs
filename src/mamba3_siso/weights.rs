//! Weight storage for Mamba-3 SISO layers.
//!
//! All tensors stored as flat `Vec<f32>` in row-major order.
//! Source: Lahoti et al., "Mamba-3", ICLR 2026 (arXiv 2603.15569).

use super::config::Mamba3Config;
use crate::ops::init::{SimpleRng, dt_bias_init, linear_default_uniform};

/// Weights for a single Mamba-3 SISO layer (inference format).
///
/// No conv1d (removed), no fixed A (input-dependent).
/// Added: BCNorm weights, per-head biases, RoPE via in_proj.
#[derive(Clone)]
pub struct Mamba3LayerWeights {
    /// The block norm's scale: `[d_model]`.
    pub norm_weight: Vec<f32>,
    /// The block LayerNorm's bias: `[d_model]`, empty for RMSNorm.
    pub norm_bias: Vec<f32>,
    /// in_proj weight: `[d_model, in_proj_dim]`, no bias.
    /// Split output: `[z, x, B, C, dd_dt, dd_A, trap, angles]`.
    pub in_proj_w: Vec<f32>,
    /// DT bias: `[nheads]`. Added before softplus.
    pub dt_bias: Vec<f32>,
    /// BCNorm weight for B: `[d_state]`. RMSNorm scale, init ones.
    pub b_norm_weight: Vec<f32>,
    /// BCNorm weight for C: `[d_state]`. RMSNorm scale, init ones.
    pub c_norm_weight: Vec<f32>,
    /// B bias: `[nheads * d_state]`. Added after BCNorm, before RoPE.
    pub b_bias: Vec<f32>,
    /// C bias: `[nheads * d_state]`. Added after BCNorm, before RoPE.
    pub c_bias: Vec<f32>,
    /// D skip connection: `[nheads]`. Per-head (not per-channel like Mamba SSM).
    pub d_param: Vec<f32>,
    /// RMSNormGated weight: `[d_inner]`. Only used if `is_outproj_norm=true`.
    pub norm_gate_weight: Vec<f32>,
    /// out_proj weight: `[d_inner, d_model]`, no bias.
    pub out_proj_w: Vec<f32>,
}

/// Weights for the complete Mamba-3 SISO backbone.
#[derive(Clone)]
pub struct Mamba3Weights {
    /// Input projection: `[input_dim, d_model]`.
    pub input_proj_w: Vec<f32>,
    /// Input projection bias: `[d_model]`.
    pub input_proj_b: Vec<f32>,
    /// Per-layer weights.
    pub layers: Vec<Mamba3LayerWeights>,
    /// The final norm's scale after all layers: `[d_model]`.
    pub norm_f_weight: Vec<f32>,
    /// The final LayerNorm's bias: `[d_model]`, empty for RMSNorm.
    pub norm_f_bias: Vec<f32>,
    /// The gated MLP of each layer, after its mixer; empty when the config
    /// has `d_intermediate == 0`.
    pub mlp: Vec<crate::mlp::GatedMlpLayerWeights>,
}

impl Mamba3Weights {
    /// Round in place the matrices a reduced precision keeps as the GPU
    /// does: the input projection, in_proj, out_proj and the MLP's
    /// projections, in the half storage or as TF32 operands. f32 changes
    /// nothing.
    pub(crate) fn round_bulk(&mut self, dtype: crate::precision::WeightDtype) {
        use crate::ops::lane::round_matrix;
        round_matrix(dtype, &mut self.input_proj_w);
        for lw in &mut self.layers {
            round_matrix(dtype, &mut lw.in_proj_w);
            round_matrix(dtype, &mut lw.out_proj_w);
        }
        for m in &mut self.mlp {
            round_matrix(dtype, &mut m.fc1_w);
            round_matrix(dtype, &mut m.fc2_w);
        }
    }

    /// Allocate zeroed weights matching the config dimensions.
    pub fn zeros(cfg: &Mamba3Config, input_dim: usize) -> Self {
        let d = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let nh = cfg.nheads();
        let ip = cfg.in_proj_out_dim();

        Self {
            input_proj_w: vec![0.0; input_dim * d],
            input_proj_b: vec![0.0; d],
            layers: (0..cfg.n_layers)
                .map(|_| Mamba3LayerWeights {
                    norm_weight: vec![1.0; d],
                    norm_bias: vec![0.0; cfg.block_norm.bias_len(d)],
                    in_proj_w: vec![0.0; d * ip],
                    dt_bias: vec![0.0; nh],
                    b_norm_weight: vec![1.0; ds],
                    c_norm_weight: vec![1.0; ds],
                    // Reference `state-spaces/mamba/mamba_ssm/modules/mamba3.py`:
                    //     self.B_bias = nn.Parameter(1 + torch.zeros(...))
                    //     self.C_bias = nn.Parameter(1 + torch.zeros(...))
                    // Init to ones (not zeros — a prior version's comment here
                    // misread the reference as plain `zeros(...)`).
                    b_bias: vec![1.0; nh * ds],
                    c_bias: vec![1.0; nh * ds],
                    d_param: vec![1.0; nh],
                    norm_gate_weight: vec![1.0; di],
                    out_proj_w: vec![0.0; di * d],
                })
                .collect(),
            // A segment without the final norm has neither tensor.
            norm_f_weight: vec![1.0; cfg.block_norm.final_lens(d, cfg.final_norm).0],
            norm_f_bias: vec![0.0; cfg.block_norm.final_lens(d, cfg.final_norm).1],
            mlp: match cfg.mlp_hidden() {
                0 => Vec::new(),
                hidden => (0..cfg.n_layers)
                    .map(|_| crate::mlp::GatedMlpLayerWeights::zeros(d, hidden, cfg.block_norm))
                    .collect(),
            },
        }
    }

    /// Initialize weights with Mamba-3 specific scheme.
    ///
    /// - Linear layers: the PyTorch `nn.Linear` default uniform (the
    ///   shipped reference init — see `linear_default_uniform`)
    /// - out_proj (and the MLP's fc2) additionally carries the GPT-2
    ///   prenorm-residual rescale, dividing by
    ///   sqrt(n_residuals_per_layer * n_layers) per the official
    ///   `mixer_seq_simple.py` `_init_weights`: one residual per layer in a
    ///   pure-Mamba stack, two with an MLP
    /// - dt_bias: inverse softplus of log-uniform(0.001, 0.1)
    /// - D, norm weights: ones
    /// - B/C biases: ones (per state-spaces/mamba `mamba3.py`:
    ///   `B_bias = 1 + torch.zeros(...)`, `C_bias = 1 + torch.zeros(...)`)
    pub fn init(cfg: &Mamba3Config, input_dim: usize, seed: u64) -> Self {
        cfg.validate().expect("invalid Mamba3Config");
        let mut w = Self::zeros(cfg, input_dim);
        let mut rng = SimpleRng::new(seed);
        let d = cfg.d_model;
        let di = cfg.d_inner();
        let residuals_per_layer = if w.mlp.is_empty() { 1 } else { 2 };
        let residual_rescale = 1.0 / ((residuals_per_layer * cfg.n_layers) as f64).sqrt() as f32;

        linear_default_uniform(&mut w.input_proj_w, input_dim, &mut rng);

        for lw in &mut w.layers {
            linear_default_uniform(&mut lw.in_proj_w, d, &mut rng);
            linear_default_uniform(&mut lw.out_proj_w, di, &mut rng);
            for v in &mut lw.out_proj_w {
                *v *= residual_rescale;
            }

            // dt_bias: inv_softplus(log-uniform(0.001, 0.1)), floored at
            // dt_init_floor per the reference init.
            dt_bias_init(&mut lw.dt_bias, &mut rng);

            // D=ones, norm_weight=ones, b_bias=ones, c_bias=ones,
            // b_norm_weight=ones, c_norm_weight=ones — all set by zeros()
            // above (zeros() initializes these to 1.0 per reference), a
            // LayerNorm's bias to zero as nn.LayerNorm does.
        }

        // The MLP draws after every mixer weight, so a model without one
        // keeps the stream it always had.
        let hidden = cfg.mlp_hidden();
        for m in &mut w.mlp {
            linear_default_uniform(&mut m.fc1_w, d, &mut rng);
            linear_default_uniform(&mut m.fc2_w, hidden, &mut rng);
            for v in &mut m.fc2_w {
                *v *= residual_rescale;
            }
        }

        w
    }

    /// Verify weight dimensions match the config.
    pub fn validate(&self, cfg: &Mamba3Config, input_dim: usize) -> Result<(), String> {
        cfg.validate()?;
        let d = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let nh = cfg.nheads();
        let ip = cfg.in_proj_out_dim();

        let ck = |name: &str, actual: usize, expected: usize| -> Result<(), String> {
            if actual != expected {
                Err(format!("{name}: expected {expected}, got {actual}"))
            } else {
                Ok(())
            }
        };

        // An empty input projection is the identity: the input is already
        // d_model wide (an HF language model feeds its embedding straight
        // in), and both the CPU step and the GPU engines skip the GEMM.
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
        let (nf, nfb) = cfg.block_norm.final_lens(d, cfg.final_norm);
        ck("norm_f_weight", self.norm_f_weight.len(), nf)?;
        ck("norm_f_bias", self.norm_f_bias.len(), nfb)?;
        ck("n_layers", self.layers.len(), cfg.n_layers)?;

        for (i, lw) in self.layers.iter().enumerate() {
            let p = |n: &str| format!("layer[{i}].{n}");
            ck(&p("norm_weight"), lw.norm_weight.len(), d)?;
            ck(&p("norm_bias"), lw.norm_bias.len(), norm_bias)?;
            ck(&p("in_proj_w"), lw.in_proj_w.len(), d * ip)?;
            ck(&p("dt_bias"), lw.dt_bias.len(), nh)?;
            ck(&p("b_norm_weight"), lw.b_norm_weight.len(), ds)?;
            ck(&p("c_norm_weight"), lw.c_norm_weight.len(), ds)?;
            ck(&p("b_bias"), lw.b_bias.len(), nh * ds)?;
            ck(&p("c_bias"), lw.c_bias.len(), nh * ds)?;
            ck(&p("d_param"), lw.d_param.len(), nh)?;
            ck(&p("norm_gate_weight"), lw.norm_gate_weight.len(), di)?;
            ck(&p("out_proj_w"), lw.out_proj_w.len(), di * d)?;
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
    fn test_zeros_valid() {
        let cfg = Mamba3Config::default();
        let w = Mamba3Weights::zeros(&cfg, 128);
        assert!(w.validate(&cfg, 128).is_ok());
    }

    #[test]
    fn test_init_valid() {
        let cfg = Mamba3Config::default();
        let w = Mamba3Weights::init(&cfg, 128, 42);
        assert!(w.validate(&cfg, 128).is_ok());
        // dt_bias should not be zero after init
        assert!(w.layers[0].dt_bias.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_init_deterministic() {
        let cfg = Mamba3Config::default();
        let w1 = Mamba3Weights::init(&cfg, 128, 42);
        let w2 = Mamba3Weights::init(&cfg, 128, 42);
        assert_eq!(w1.layers[0].in_proj_w, w2.layers[0].in_proj_w);
        assert_eq!(w1.layers[0].dt_bias, w2.layers[0].dt_bias);
    }
}
