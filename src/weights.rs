use crate::config::MambaConfig;

/// Weight storage for a single Mamba layer.
///
/// Matches the parameter layout from Gu & Dao (2023).
/// All tensors stored as flat `Vec<f32>` in row-major order.
#[derive(Clone)]
pub struct MambaLayerWeights {
    /// RMSNorm scale `[d_model]`
    pub norm_weight: Vec<f32>,

    /// Input projection `[d_model, 2 * d_inner]` (no bias)
    pub in_proj_w: Vec<f32>,

    /// Depthwise conv1d `[d_inner, d_conv]`
    pub conv1d_weight: Vec<f32>,
    /// Conv1d bias `[d_inner]`
    pub conv1d_bias: Vec<f32>,

    /// SSM input projection `[d_inner, dt_rank + 2*d_state]` (no bias)
    pub x_proj_w: Vec<f32>,

    /// Delta projection `[dt_rank, d_inner]`
    pub dt_proj_w: Vec<f32>,
    /// Delta projection bias `[d_inner]`
    pub dt_proj_b: Vec<f32>,

    /// SSM transition matrix (log-space) `[d_inner * d_state]`
    pub a_log: Vec<f32>,

    /// Pre-computed `a_neg = -exp(a_log)` `[d_inner * d_state]`.
    /// Computed once at load time. Inference reads this instead of recomputing per step.
    pub a_neg: Vec<f32>,

    /// SSM skip connection `[d_inner]`
    pub d_param: Vec<f32>,

    /// Output projection `[d_inner, d_model]` (no bias)
    pub out_proj_w: Vec<f32>,
}

impl MambaLayerWeights {
    /// Recompute `a_neg = -exp(a_log)` from current `a_log` values.
    /// Call after loading weights from safetensors or modifying `a_log`.
    pub fn compute_a_neg(&mut self) {
        for i in 0..self.a_log.len() {
            self.a_neg[i] = -self.a_log[i].exp();
        }
    }
}

/// Weight storage for the complete Mamba backbone.
#[derive(Clone)]
pub struct MambaWeights {
    /// Input projection `[input_dim, d_model]`
    pub input_proj_w: Vec<f32>,
    /// Input projection bias `[d_model]`
    pub input_proj_b: Vec<f32>,

    /// Per-layer weights
    pub layers: Vec<MambaLayerWeights>,

    /// Final RMSNorm after all layers `[d_model]`
    pub norm_f_weight: Vec<f32>,
}

impl MambaWeights {
    /// Allocate zeroed weights matching the config dimensions.
    pub fn zeros(cfg: &MambaConfig, input_dim: usize) -> Self {
        let d = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let dc = cfg.d_conv;
        let dr = cfg.dt_rank();
        let xd = cfg.xdbl_dim();

        Self {
            input_proj_w: vec![0.0; input_dim * d],
            input_proj_b: vec![0.0; d],
            layers: (0..cfg.n_layers)
                .map(|_| MambaLayerWeights {
                    norm_weight: vec![1.0; d], // RMSNorm init = ones
                    in_proj_w: vec![0.0; d * 2 * di],
                    conv1d_weight: vec![0.0; di * dc],
                    conv1d_bias: vec![0.0; di],
                    x_proj_w: vec![0.0; di * xd],
                    dt_proj_w: vec![0.0; dr * di],
                    dt_proj_b: vec![0.0; di],
                    a_log: vec![0.0; di * ds],
                    a_neg: vec![0.0; di * ds], // computed from a_log after init
                    d_param: vec![1.0; di],    // D init = ones (paper)
                    out_proj_w: vec![0.0; di * d],
                })
                .collect(),
            norm_f_weight: vec![1.0; d], // RMSNorm init = ones
        }
    }

    /// Initialize weights with Mamba-specific scheme from the paper.
    ///
    /// - Linear layers: the PyTorch default, uniform with bound 1/sqrt(fan_in)
    /// - dt_proj weight: uniform with bound 1/sqrt(dt_rank), which is the
    ///   reference's `dt_rank^-0.5 * dt_scale` at the shipped scale 1.0
    /// - out_proj: the same draw, then divided by sqrt(n_layers) — the
    ///   GPT-2 prenorm-residual rule the reference applies to this one
    ///   weight (one residual per layer for Mamba-1)
    /// - A_log: log(1..=d_state) repeated across d_inner (Section 3.5)
    /// - dt_proj bias: inverse softplus of log-uniform(dt_min, dt_max) (Section 3.5)
    /// - D: ones
    /// - RMSNorm: ones
    /// - conv1d weight and bias: the PyTorch default with fan_in = d_conv
    pub fn init(cfg: &MambaConfig, input_dim: usize, seed: u64) -> Self {
        /// The reference clamps the log-uniform dt draw from below before
        /// inverting the softplus; inside the shipped [0.001, 0.1] range the
        /// clamp never fires, and it keeps a narrower range faithful.
        const DT_INIT_FLOOR: f32 = 1e-4;
        let mut w = Self::zeros(cfg, input_dim);
        let mut rng = SimpleRng::new(seed);
        let d = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let dc = cfg.d_conv;
        let dr = cfg.dt_rank();

        // input_proj: the Linear default on fan_in = input_dim
        linear_default_uniform(&mut w.input_proj_w, input_dim, &mut rng);
        // input_proj bias: zero (default)

        let residual_rescale = 1.0 / (cfg.n_layers as f64).sqrt() as f32;

        for lw in &mut w.layers {
            // in_proj: the Linear default on fan_in = d_model
            linear_default_uniform(&mut lw.in_proj_w, d, &mut rng);

            // conv1d weight and bias: the Conv1d default, fan_in = d_conv
            // (in_channels/groups * kernel_size = 1 * d_conv). `_init_weights`
            // zeroes Linear biases only, so the conv bias keeps this draw.
            linear_default_uniform(&mut lw.conv1d_weight, dc, &mut rng);
            linear_default_uniform(&mut lw.conv1d_bias, dc, &mut rng);

            // x_proj: the Linear default on fan_in = d_inner
            linear_default_uniform(&mut lw.x_proj_w, di, &mut rng);

            // dt_proj weight: the reference draws uniform(-s, s) with
            // s = dt_rank^-0.5 * dt_scale to preserve the variance of the
            // dt branch; at the shipped dt_scale = 1.0 that is exactly the
            // Linear default on fan_in = dt_rank.
            linear_default_uniform(&mut lw.dt_proj_w, dr, &mut rng);
            // dt_proj bias: inv_softplus(uniform(0.001, 0.1))
            // dt_proj bias: inv_softplus(log-uniform(dt_min, dt_max))
            // Log-uniform sampling matches official Python init (Section 3.5)
            let log_dt_min = 0.001_f32.ln();
            let log_dt_max = 0.1_f32.ln();
            for b in &mut lw.dt_proj_b {
                let dt = (rng.next_f32() * (log_dt_max - log_dt_min) + log_dt_min)
                    .exp()
                    .max(DT_INIT_FLOOR);
                *b = inv_softplus(dt);
            }

            // A_log: log(1..=d_state) repeated across d_inner
            for d_idx in 0..di {
                for n in 0..ds {
                    lw.a_log[d_idx * ds + n] = ((n + 1) as f32).ln();
                }
            }

            // out_proj: the Linear default, then the GPT-2 prenorm-residual
            // rescale. Without it every block's residual contribution starts
            // n_layers times too large in variance.
            linear_default_uniform(&mut lw.out_proj_w, di, &mut rng);
            for v in &mut lw.out_proj_w {
                *v *= residual_rescale;
            }

            // D = ones, norm_weight = ones (already set in zeros())

            // Pre-compute a_neg from a_log
            lw.compute_a_neg();
        }

        w
    }

    /// Verify weight dimensions match the config.
    ///
    /// Returns `Err` if any tensor has the wrong size.
    pub fn validate(&self, cfg: &MambaConfig, input_dim: usize) -> Result<(), String> {
        let d = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let dc = cfg.d_conv;
        let dr = cfg.dt_rank();
        let xd = cfg.xdbl_dim();

        let check = |name: &str, actual: usize, expected: usize| -> Result<(), String> {
            if actual != expected {
                return Err(format!("{name}: expected {expected}, got {actual}"));
            }
            Ok(())
        };

        check("input_proj_w", self.input_proj_w.len(), input_dim * d)?;
        check("input_proj_b", self.input_proj_b.len(), d)?;
        check("norm_f_weight", self.norm_f_weight.len(), d)?;

        if self.layers.len() != cfg.n_layers {
            return Err(format!(
                "expected {} layers, got {}",
                cfg.n_layers,
                self.layers.len()
            ));
        }

        for (i, lw) in self.layers.iter().enumerate() {
            let p = |n: &str| format!("layer[{i}].{n}");
            check(&p("norm_weight"), lw.norm_weight.len(), d)?;
            check(&p("in_proj_w"), lw.in_proj_w.len(), d * 2 * di)?;
            check(&p("conv1d_weight"), lw.conv1d_weight.len(), di * dc)?;
            check(&p("conv1d_bias"), lw.conv1d_bias.len(), di)?;
            check(&p("x_proj_w"), lw.x_proj_w.len(), di * xd)?;
            check(&p("dt_proj_w"), lw.dt_proj_w.len(), dr * di)?;
            check(&p("dt_proj_b"), lw.dt_proj_b.len(), di)?;
            check(&p("a_log"), lw.a_log.len(), di * ds)?;
            check(&p("d_param"), lw.d_param.len(), di)?;
            check(&p("out_proj_w"), lw.out_proj_w.len(), di * d)?;
        }

        Ok(())
    }

    /// Total parameter count.
    pub fn param_count(&self, input_dim: usize, cfg: &MambaConfig) -> usize {
        let d = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let dc = cfg.d_conv;
        let dr = cfg.dt_rank();
        let xd = cfg.xdbl_dim();
        let per_layer =
            d + d * 2 * di + di * dc + di + di * xd + dr * di + di + di * ds + di + di * d;
        input_dim * d + d + cfg.n_layers * per_layer + d
    }
}

// Minimal RNG (no external dependency)
struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// The reference Linear init: `nn.Linear`'s default `reset_parameters`
/// is `kaiming_uniform_(a=sqrt(5))`, i.e. gain 1/sqrt(3) and bound
/// 1/sqrt(fan_in), not the gain-1 bound sqrt(3/fan_in). The official
/// `_init_weights` re-touches only biases and the embedding, so the
/// PyTorch default IS the shipped Mamba-1 init. `nn.Conv1d` draws its
/// weight and its bias from the same rule with fan_in = d_conv.
/// The Mamba-3 twin carries the identical helper.
fn linear_default_uniform(w: &mut [f32], fan_in: usize, rng: &mut SimpleRng) {
    let bound = (1.0 / fan_in as f64).sqrt() as f32;
    for v in w.iter_mut() {
        *v = -bound + 2.0 * bound * rng.next_f32();
    }
}

/// Inverse softplus. `exp_m1` keeps the small-x branch cancellation-free;
/// the naive `(y.exp() - 1.0).ln()` loses about half the mantissa at the
/// low end of the init range.
fn inv_softplus(y: f32) -> f32 {
    if y > 20.0 { y } else { y.exp_m1().ln() }
}

#[cfg(test)]
mod init_tests {
    use super::*;

    fn cfg(n_layers: usize) -> MambaConfig {
        MambaConfig {
            d_model: 64,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            n_layers,
            ..MambaConfig::default()
        }
    }

    fn bound_of(w: &[f32]) -> f32 {
        w.iter().fold(0.0_f32, |m, v| m.max(v.abs()))
    }

    #[test]
    fn linear_weights_draw_the_pytorch_default_bound() {
        // nn.Linear's reset_parameters is kaiming_uniform_(a=sqrt(5)),
        // which is uniform with bound 1/sqrt(fan_in). The gain-1 bound
        // sqrt(3/fan_in) that this crate used gave every projection three
        // times the reference variance.
        let c = cfg(2);
        let w = MambaWeights::init(&c, c.d_model, 11);
        let lw = &w.layers[0];
        let expect = |fan_in: usize| 1.0 / (fan_in as f32).sqrt();
        for (name, buf, fan_in) in [
            ("in_proj", &lw.in_proj_w, c.d_model),
            ("x_proj", &lw.x_proj_w, c.d_inner()),
            ("dt_proj", &lw.dt_proj_w, c.dt_rank()),
            ("conv1d", &lw.conv1d_weight, c.d_conv),
        ] {
            let b = bound_of(buf);
            let e = expect(fan_in);
            assert!(
                b <= e && b > 0.80 * e,
                "{name}: max |w| = {b}, expected just under {e}"
            );
        }
    }

    #[test]
    fn out_proj_carries_the_residual_rescale() {
        // The reference re-draws out_proj and divides it by
        // sqrt(n_residuals_per_layer * n_layer); Mamba-1 has one residual
        // per layer, so the factor is 1/sqrt(n_layer).
        let c = cfg(16);
        let w = MambaWeights::init(&c, c.d_model, 12);
        let b = bound_of(&w.layers[0].out_proj_w);
        let e = (1.0 / (c.d_inner() as f32).sqrt()) / (c.n_layers as f32).sqrt();
        assert!(
            b <= e && b > 0.80 * e,
            "out_proj: max |w| = {b}, expected just under {e}"
        );
    }

    #[test]
    fn conv1d_bias_is_drawn_not_zeroed() {
        // _init_weights zeroes nn.Linear biases only; nn.Conv1d keeps its
        // own uniform(-1/sqrt(d_conv), +1/sqrt(d_conv)) draw, and the bias
        // enters before the SiLU, so a zero bias starts every channel on
        // the symmetric part of the activation.
        let c = cfg(2);
        let w = MambaWeights::init(&c, c.d_model, 13);
        let bias = &w.layers[0].conv1d_bias;
        assert!(
            bias.iter().any(|v| *v != 0.0),
            "conv1d bias must not be all zeros"
        );
        let b = bound_of(bias);
        let e = 1.0 / (c.d_conv as f32).sqrt();
        assert!(b <= e, "conv1d bias: max |b| = {b}, bound {e}");
    }

    #[test]
    fn a_log_is_the_s4d_real_ladder() {
        let c = cfg(1);
        let w = MambaWeights::init(&c, c.d_model, 14);
        let lw = &w.layers[0];
        for d in 0..c.d_inner() {
            for n in 0..c.d_state {
                let got = lw.a_log[d * c.d_state + n];
                let want = ((n + 1) as f32).ln();
                assert!(
                    (got - want).abs() < 1e-6,
                    "A_log[{d},{n}] = {got}, want {want}"
                );
            }
        }
    }
}
