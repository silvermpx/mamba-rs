//! Weight storage for Mamba-3 SISO layers.
//!
//! All tensors stored as flat `Vec<f32>` in row-major order.
//! Source: Lahoti et al., "Mamba-3", ICLR 2026 (arXiv 2603.15569).

use super::config::Mamba3Config;

/// Weights for a single Mamba-3 SISO layer (inference format).
///
/// No conv1d (removed), no fixed A (input-dependent).
/// Added: BCNorm weights, per-head biases, RoPE via in_proj.
#[derive(Clone)]
pub struct Mamba3LayerWeights {
    /// RMSNorm scale: `[d_model]`.
    pub norm_weight: Vec<f32>,
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
    /// Final RMSNorm after all layers: `[d_model]`.
    pub norm_f_weight: Vec<f32>,
}

/// Simple xorshift64 RNG for weight initialization (deterministic, no deps).
struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFFFF) as f32 / 16777216.0
    }
}

fn kaiming_uniform(buf: &mut [f32], fan_in: usize, rng: &mut SimpleRng) {
    let bound = (3.0 / fan_in as f64).sqrt() as f32;
    for v in buf.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * bound;
    }
}

fn inv_softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (x.exp() - 1.0).ln() }
}

impl Mamba3Weights {
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
            norm_f_weight: vec![1.0; d],
        }
    }

    /// Initialize weights with Mamba-3 specific scheme.
    ///
    /// - Linear layers: Kaiming uniform (fan_in)
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

        kaiming_uniform(&mut w.input_proj_w, input_dim, &mut rng);

        for lw in &mut w.layers {
            kaiming_uniform(&mut lw.in_proj_w, d, &mut rng);
            kaiming_uniform(&mut lw.out_proj_w, di, &mut rng);

            // dt_bias: inv_softplus(log-uniform(0.001, 0.1))
            let log_dt_min = 0.001_f32.ln();
            let log_dt_max = 0.1_f32.ln();
            for b in &mut lw.dt_bias {
                let dt = (rng.next_f32() * (log_dt_max - log_dt_min) + log_dt_min).exp();
                *b = inv_softplus(dt);
            }

            // D=ones, norm_weight=ones, b_bias=ones, c_bias=ones,
            // b_norm_weight=ones, c_norm_weight=ones — all set by zeros()
            // above (zeros() initializes these to 1.0 per reference).
        }

        w
    }

    /// Verify weight dimensions match the config.
    pub fn validate(&self, cfg: &Mamba3Config, input_dim: usize) -> Result<(), String> {
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

        ck("input_proj_w", self.input_proj_w.len(), input_dim * d)?;
        ck("input_proj_b", self.input_proj_b.len(), d)?;
        ck("norm_f_weight", self.norm_f_weight.len(), d)?;
        ck("n_layers", self.layers.len(), cfg.n_layers)?;

        for (i, lw) in self.layers.iter().enumerate() {
            let p = |n: &str| format!("layer[{i}].{n}");
            ck(&p("norm_weight"), lw.norm_weight.len(), d)?;
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
