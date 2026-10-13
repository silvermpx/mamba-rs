//! Dynamic loss scaler for mixed-precision (especially f16) training: the
//! host state machine every trainer drives, on the GPU and on the CPU.
//!
//! Mirrors `torch.cuda.amp.GradScaler` (PyTorch since 1.6, prior NVIDIA Apex):
//!   * Scale the loss by `S` before the backward: the gradients come out as
//!     `S * dL/dθ`.
//!   * Scan the f32 master gradients for inf/nan (the GPU's `OverflowFlag`,
//!     a host scan on the CPU).
//!   * On overflow: skip the optimizer step, decrease `S` by `backoff_factor`.
//!   * On a clean step: unscale the gradients (`grads *= 1 / S`), step the
//!     optimizer.
//!   * After `growth_interval` consecutive clean steps, increase `S` by
//!     `growth_factor` (capped at `max_scale`).
//!
//! ## When to use
//! - **f16 training**: REQUIRED. f16 dynamic range (~6e-5..6e4) is too narrow
//!   for un-scaled SSM/transformer gradients; small grads underflow to 0.
//! - **bf16 training**: NOT NEEDED. bf16 has the same dynamic range as f32,
//!   only mantissa is reduced. Setting [`DynamicLossScaler::disabled`] keeps
//!   the API uniform for code that targets both dtypes.
//! - **f32 training**: NOT NEEDED. Use `disabled()`.
//!
//! ## Defaults (matching torch.cuda.amp.GradScaler)
//! - `init_scale = 2^16 = 65 536`
//! - `growth_factor = 2.0`
//! - `backoff_factor = 0.5`
//! - `growth_interval = 2 000`
//! - `max_scale = 2^24 = 16 777 216`  (avoid runaway towards f32 inf)
//! - `min_scale = 1.0` (don't undershoot — under 1.0 the scaler is doing
//!   nothing useful)
//!
//! ## Example
//! ```ignore
//! let mut scaler = DynamicLossScaler::new();
//! loop {
//!     let scaled_loss = compute_loss() * scaler.scale();
//!     backward(scaled_loss, &mut grads)?;
//!     let has_overflow = grads.iter().any(|g| !g.is_finite());
//!     if !has_overflow {
//!         grads.iter_mut().for_each(|g| *g *= 1.0 / scaler.scale());
//!         optimizer.step(&mut weights, &grads);
//!     }
//!     scaler.update(has_overflow);
//! }
//! ```
//!
//! Source: `torch.cuda.amp.grad_scaler.GradScaler` (PyTorch 2.5,
//! `torch/cuda/amp/grad_scaler.py`), NVIDIA Apex `apex/amp/scaler.py`.

/// CPU-side state machine for dynamic loss scaling.
#[derive(Clone, Debug)]
pub struct DynamicLossScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: u32,
    growth_tracker: u32,
    max_scale: f32,
    min_scale: f32,
    enabled: bool,
}

impl Default for DynamicLossScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicLossScaler {
    /// Default scaler suitable for f16 training.
    pub fn new() -> Self {
        Self {
            scale: 65_536.0, // 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2_000,
            growth_tracker: 0,
            max_scale: 16_777_216.0, // 2^24
            min_scale: 1.0,
            enabled: true,
        }
    }

    /// Disabled scaler — `scale()` always returns 1.0, `update()` no-op.
    /// Use for bf16 or f32 training where scaling is unnecessary, while
    /// keeping the same call sites as f16.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            scale: 1.0,
            ..Self::new()
        }
    }

    /// Override the initial scale. Useful when resuming from a checkpoint
    /// with a known stable scale, or when training a small model that
    /// converges to a low scale.
    ///
    /// Panics if `init_scale` is not finite or ≤ 0 (would silently zero or
    /// invert all grads).
    #[must_use]
    pub fn with_init_scale(mut self, init_scale: f32) -> Self {
        assert!(
            init_scale.is_finite() && init_scale > 0.0,
            "init_scale must be finite and positive, got {init_scale}"
        );
        self.scale = init_scale;
        self
    }

    /// Override the growth interval (default 2000 clean steps).
    ///
    /// Panics if `n == 0` (would grow every step, defeating the dynamic-
    /// discovery purpose).
    #[must_use]
    pub fn with_growth_interval(mut self, n: u32) -> Self {
        assert!(n > 0, "growth_interval must be > 0");
        self.growth_interval = n;
        self
    }

    /// Override max scale (default 2^24). Higher → more headroom for tiny
    /// grads, but increases inf risk on already-large grads.
    ///
    /// Panics if `s` is not finite or ≤ 0, or if `s < min_scale`.
    #[must_use]
    pub fn with_max_scale(mut self, s: f32) -> Self {
        assert!(
            s.is_finite() && s > 0.0,
            "max_scale must be finite and positive, got {s}"
        );
        assert!(
            s >= self.min_scale,
            "max_scale ({s}) must be >= min_scale ({})",
            self.min_scale
        );
        self.max_scale = s;
        self
    }

    /// Override the minimum scale floor (default 1.0). Set to a small
    /// value (e.g. `f32::MIN_POSITIVE`) to disable the floor and match
    /// PyTorch GradScaler semantics where a chronically-broken model is
    /// allowed to drive scale below 1.0 (signals to the user that
    /// training is failing — see the scaler notes).
    ///
    /// Panics if `s` is not finite, ≤ 0, or > max_scale.
    #[must_use]
    pub fn with_min_scale(mut self, s: f32) -> Self {
        assert!(
            s.is_finite() && s > 0.0,
            "min_scale must be finite and positive, got {s}"
        );
        assert!(
            s <= self.max_scale,
            "min_scale ({s}) must be <= max_scale ({})",
            self.max_scale
        );
        self.min_scale = s;
        self
    }

    /// Serialize state for checkpoint resume — returns `(scale, growth_tracker)`.
    /// Pass back into [`Self::load_state`] after loading config to resume
    /// without re-paying the ~2000-step scale-discovery phase. Mirrors
    /// `torch.cuda.amp.GradScaler.state_dict()`.
    pub fn state(&self) -> (f32, u32) {
        (self.scale, self.growth_tracker)
    }

    /// Restore state from a previous [`Self::state`]. A scale that is not
    /// finite and positive is an error; one outside the scaler's bounds is
    /// clamped to them.
    pub fn load_state(&mut self, scale: f32, growth_tracker: u32) -> Result<(), String> {
        Self::check_scale(scale)?;
        self.scale = scale.clamp(self.min_scale, self.max_scale);
        self.growth_tracker = growth_tracker.min(self.growth_interval.saturating_sub(1));
        Ok(())
    }

    /// Whether `scale` can be loaded: finite and positive.
    pub fn check_scale(scale: f32) -> Result<(), String> {
        if scale.is_finite() && scale > 0.0 {
            Ok(())
        } else {
            Err(format!(
                "a loaded loss scale must be finite and positive, got {scale}"
            ))
        }
    }

    /// Current scale value to multiply the loss by.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Number of consecutive clean steps since last overflow.
    pub fn clean_step_count(&self) -> u32 {
        self.growth_tracker
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Apply the result of the inf/nan check after backward.
    ///
    /// `had_overflow = true`  → divide scale by `backoff_factor`, reset
    /// the clean-step counter. The caller MUST skip the optimizer step.
    ///
    /// `had_overflow = false` → increment the clean-step counter; after
    /// `growth_interval` consecutive clean steps multiply scale by
    /// `growth_factor` (capped at `max_scale`) and reset the counter.
    pub fn update(&mut self, had_overflow: bool) {
        if !self.enabled {
            return;
        }
        if had_overflow {
            self.scale = (self.scale * self.backoff_factor).max(self.min_scale);
            self.growth_tracker = 0;
        } else {
            self.growth_tracker = self.growth_tracker.saturating_add(1);
            if self.growth_tracker >= self.growth_interval {
                self.scale = (self.scale * self.growth_factor).min(self.max_scale);
                self.growth_tracker = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_state_machine_clean_growth() {
        let mut s = DynamicLossScaler::new()
            .with_init_scale(8.0)
            .with_growth_interval(3);
        assert_eq!(s.scale(), 8.0);
        s.update(false); // 1 clean
        s.update(false); // 2 clean
        assert_eq!(s.scale(), 8.0);
        s.update(false); // 3 clean → grow
        assert_eq!(s.scale(), 16.0);
        assert_eq!(s.clean_step_count(), 0);
    }

    #[test]
    fn cpu_state_machine_overflow_backoff() {
        let mut s = DynamicLossScaler::new().with_init_scale(8.0);
        s.update(false);
        s.update(false);
        s.update(true); // overflow halves scale
        assert_eq!(s.scale(), 4.0);
        assert_eq!(s.clean_step_count(), 0);
    }

    #[test]
    fn cpu_state_machine_min_scale_floor() {
        let mut s = DynamicLossScaler::new().with_init_scale(2.0);
        // Without floor: 2 → 1 → 0.5 → 0.25 ...
        // With min=1.0: stays at 1.0
        s.update(true);
        s.update(true);
        s.update(true);
        assert_eq!(s.scale(), 1.0);
    }

    #[test]
    fn cpu_state_machine_max_scale_cap() {
        let mut s = DynamicLossScaler::new()
            .with_init_scale(8.0)
            .with_growth_interval(1)
            .with_max_scale(16.0);
        s.update(false); // 8 → 16
        assert_eq!(s.scale(), 16.0);
        s.update(false); // capped at 16
        assert_eq!(s.scale(), 16.0);
    }

    #[test]
    fn disabled_scaler_is_identity() {
        let mut s = DynamicLossScaler::disabled();
        assert_eq!(s.scale(), 1.0);
        assert!(!s.enabled());
        s.update(true);
        assert_eq!(s.scale(), 1.0);
    }

    #[test]
    fn state_dict_round_trip() {
        let mut s = DynamicLossScaler::new()
            .with_init_scale(8.0)
            .with_growth_interval(10);
        s.update(false);
        s.update(false);
        s.update(false);
        let (saved_scale, saved_tracker) = s.state();
        assert_eq!(saved_scale, 8.0);
        assert_eq!(saved_tracker, 3);

        let mut s2 = DynamicLossScaler::new().with_growth_interval(10);
        s2.load_state(saved_scale, saved_tracker).unwrap();
        assert_eq!(s2.scale(), 8.0);
        assert_eq!(s2.clean_step_count(), 3);
    }

    #[test]
    fn load_state_clamps_to_bounds() {
        let mut s = DynamicLossScaler::new()
            .with_init_scale(8.0)
            .with_max_scale(100.0);
        s.load_state(1e9, 999_999).unwrap(); // way over max + tracker over interval
        assert_eq!(s.scale(), 100.0); // clamped to max
        // tracker clamped under growth_interval so it doesn't immediately grow
        assert!(s.clean_step_count() < s.growth_interval);
    }

    #[test]
    #[should_panic(expected = "init_scale must be finite and positive")]
    fn init_scale_zero_panics() {
        let _ = DynamicLossScaler::new().with_init_scale(0.0);
    }

    #[test]
    #[should_panic(expected = "init_scale must be finite and positive")]
    fn init_scale_neg_panics() {
        let _ = DynamicLossScaler::new().with_init_scale(-1.0);
    }

    #[test]
    #[should_panic(expected = "init_scale must be finite and positive")]
    fn init_scale_inf_panics() {
        let _ = DynamicLossScaler::new().with_init_scale(f32::INFINITY);
    }

    #[test]
    #[should_panic(expected = "init_scale must be finite and positive")]
    fn init_scale_nan_panics() {
        let _ = DynamicLossScaler::new().with_init_scale(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "growth_interval must be > 0")]
    fn growth_interval_zero_panics() {
        let _ = DynamicLossScaler::new().with_growth_interval(0);
    }

    #[test]
    #[should_panic(expected = "max_scale")]
    fn max_below_min_panics() {
        // default min_scale=1.0 → max=0.5 < 1.0 → panic
        let _ = DynamicLossScaler::new().with_max_scale(0.5);
    }

    #[test]
    #[should_panic(expected = "min_scale")]
    fn min_above_max_panics() {
        // default max_scale=2^24, set min above it
        let _ = DynamicLossScaler::new().with_min_scale(1e10);
    }

    #[test]
    fn min_scale_pytorch_equivalent() {
        // PyTorch GradScaler has no floor — disable ours by setting a tiny min.
        let mut s = DynamicLossScaler::new()
            .with_init_scale(2.0)
            .with_min_scale(f32::MIN_POSITIVE);
        s.update(true); // 2 → 1 (above floor)
        s.update(true); // 1 → 0.5
        s.update(true); // 0.5 → 0.25
        assert!((s.scale() - 0.25).abs() < 1e-6);
    }
}
