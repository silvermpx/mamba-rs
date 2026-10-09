//! Weight-initialisation helpers shared by the backbones that draw their
//! own reference init.

/// Simple xorshift64 RNG for weight initialization (deterministic, no deps).
pub(crate) struct SimpleRng(u64);

impl SimpleRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFFFF) as f32 / 16777216.0
    }
}

/// The reference Linear init: `nn.Linear`'s default `reset_parameters`
/// is `kaiming_uniform_(a=sqrt(5))`, i.e. gain 1/sqrt(3) and bound
/// 1/sqrt(fan_in) — NOT the gain-1 bound sqrt(3/fan_in). The official
/// `_init_weights` never re-touches these projections, so the PyTorch
/// default IS the shipped Mamba-3 init; our previous gain-1 bound gave
/// every projection 3x the reference variance.
pub(crate) fn linear_default_uniform(buf: &mut [f32], fan_in: usize, rng: &mut SimpleRng) {
    let bound = (1.0 / fan_in as f64).sqrt() as f32;
    for v in buf.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * bound;
    }
}

/// Inverse softplus. `exp_m1` keeps the small-x branch cancellation-free
/// — the naive `(x.exp() - 1.0).ln()` loses ~half the mantissa at the
/// dt_init_floor end of the init range.
pub(crate) fn inv_softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp_m1().ln() }
}

/// The reference dt_bias init (state-spaces/mamba `mamba3.py`): dt drawn
/// log-uniform in [0.001, 0.1], floored at dt_init_floor = 1e-4, stored as
/// its inverse softplus.
pub(crate) fn dt_bias_init(buf: &mut [f32], rng: &mut SimpleRng) {
    let log_dt_min = 0.001_f32.ln();
    let log_dt_max = 0.1_f32.ln();
    const DT_INIT_FLOOR: f32 = 1e-4;
    for b in buf.iter_mut() {
        let dt = (rng.next_f32() * (log_dt_max - log_dt_min) + log_dt_min)
            .exp()
            .max(DT_INIT_FLOOR);
        *b = inv_softplus(dt);
    }
}
