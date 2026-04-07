use crate::config::MambaConfig;

/// Weight storage for a single Mamba layer.
///
/// Matches the parameter layout from Gu & Dao (2023).
/// All tensors stored as flat `Vec<f32>` in row-major order.
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
    /// - Linear layers: Kaiming uniform (fan_in)
    /// - A_log: log(1..=d_state) repeated across d_inner (Section 3.5)
    /// - dt_proj bias: inverse softplus of uniform(dt_min, dt_max) (Section 3.5)
    /// - D: ones
    /// - RMSNorm: ones
    /// - conv1d: Kaiming uniform
    pub fn init(cfg: &MambaConfig, input_dim: usize, seed: u64) -> Self {
        let mut w = Self::zeros(cfg, input_dim);
        let mut rng = SimpleRng::new(seed);
        let d = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let dc = cfg.d_conv;
        let dr = cfg.dt_rank();

        // input_proj: Kaiming uniform(fan_in=input_dim)
        kaiming_uniform(&mut w.input_proj_w, input_dim, &mut rng);
        // input_proj bias: zero (default)

        for lw in &mut w.layers {
            // in_proj: Kaiming uniform(fan_in=d_model)
            kaiming_uniform(&mut lw.in_proj_w, d, &mut rng);

            // conv1d: Kaiming uniform(fan_in=d_conv)
            kaiming_uniform(&mut lw.conv1d_weight, dc, &mut rng);
            // conv1d bias: zero

            // x_proj: Kaiming uniform(fan_in=d_inner)
            kaiming_uniform(&mut lw.x_proj_w, di, &mut rng);

            // dt_proj: special init (Section 3.5)
            kaiming_uniform(&mut lw.dt_proj_w, dr, &mut rng);
            // dt_proj bias: inv_softplus(uniform(0.001, 0.1))
            // dt_proj bias: inv_softplus(log-uniform(dt_min, dt_max))
            // Log-uniform sampling matches official Python init (Section 3.5)
            let log_dt_min = 0.001_f32.ln();
            let log_dt_max = 0.1_f32.ln();
            for b in &mut lw.dt_proj_b {
                let dt = (rng.next_f32() * (log_dt_max - log_dt_min) + log_dt_min).exp();
                *b = inv_softplus(dt);
            }

            // A_log: log(1..=d_state) repeated across d_inner
            for d_idx in 0..di {
                for n in 0..ds {
                    lw.a_log[d_idx * ds + n] = ((n + 1) as f32).ln();
                }
            }

            // out_proj: Kaiming uniform(fan_in=d_inner)
            kaiming_uniform(&mut lw.out_proj_w, di, &mut rng);

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

fn kaiming_uniform(w: &mut [f32], fan_in: usize, rng: &mut SimpleRng) {
    let bound = (3.0 / fan_in as f32).sqrt();
    for v in w.iter_mut() {
        *v = -bound + 2.0 * bound * rng.next_f32();
    }
}

fn inv_softplus(y: f32) -> f32 {
    if y > 20.0 { y } else { (y.exp() - 1.0).ln() }
}
