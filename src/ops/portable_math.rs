//! Transcendentals with one result on every machine.
//!
//! Built only from IEEE-rounded operations, explicit fused multiply-adds
//! and integer bit operations, never libm: the platform libm and the
//! instruction set no longer decide the bits. The Mamba-2 CPU path uses
//! these, so its outputs are the same on NEON, AVX2 and the scalar
//! fallback, on macOS and on Linux.
//!
//! The polynomials are Cephes (S. L. Moshier) `expf` and `logf`, the same
//! coefficients as [`crate::ops::fast_math`]; what differs is that every
//! product-add here is one fused operation, in a fixed order.

/// exp(-87.33654) = 2^-126, the smallest normal: below it the result
/// saturates rather than going subnormal.
const EXP_LO: f32 = -87.33654;
/// The largest input whose exponent still fits (n rounds to 127).
const EXP_HI: f32 = 88.37626;
const LOG2E: f32 = std::f32::consts::LOG2_E;
/// 1.5 * 2^23: adding it rounds a value below 2^22 to an integer, which
/// then sits in the low bits of the sum.
const ROUND_MAGIC: f32 = 12_582_912.0;
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;
const EXP_P: [f32; 6] = [
    1.987_569_1e-4,
    1.398_199_9e-3,
    8.333_452e-3,
    4.166_579_6e-2,
    1.666_666_5e-1,
    0.5,
];
const LOG_P: [f32; 9] = [
    7.037_683_6e-2,
    -1.151_461e-1,
    1.167_699_9e-1,
    -1.242_014_1e-1,
    1.424_932_3e-1,
    -1.666_805_8e-1,
    2.000_071_5e-1,
    -2.499_999_4e-1,
    3.333_333e-1,
];

/// e^x, within 2 ulp of the exact value over the normal range. Inputs
/// below -87.34 give 2^-126, above 88.38 give e^88.38; a NaN stays NaN.
#[inline(always)]
pub fn exp_portable(x: f32) -> f32 {
    // clamp keeps a NaN a NaN.
    let xc = x.clamp(EXP_LO, EXP_HI);
    let t = xc.mul_add(LOG2E, ROUND_MAGIC);
    let n = t - ROUND_MAGIC;
    let ni = t.to_bits().wrapping_sub(ROUND_MAGIC.to_bits()) as i32;
    let r = n.mul_add(-LN2_LO, n.mul_add(-LN2_HI, xc));
    let r2 = r * r;
    let mut p = EXP_P[0];
    for &c in &EXP_P[1..] {
        p = p.mul_add(r, c);
    }
    let p = p.mul_add(r2, r) + 1.0;
    p * f32::from_bits(((ni + 127) as u32) << 23)
}

/// ln(1 + y) for y in [0, 1], within 2 ulp. The reduction keeps the
/// mantissa in [sqrt(1/2), sqrt(2)), and the last term restores what the
/// rounding of 1 + y dropped.
#[inline(always)]
pub fn log1p_portable(y: f32) -> f32 {
    let u = 1.0 + y;
    if u == 1.0 {
        return y;
    }
    let (e, m) = if u >= std::f32::consts::SQRT_2 {
        (1.0f32, u * 0.5)
    } else {
        (0.0f32, u)
    };
    // Exact: m is within a factor of two of 1.
    let f = m - 1.0;
    let z = f * f;
    let mut p = LOG_P[0];
    for &c in &LOG_P[1..] {
        p = p.mul_add(f, c);
    }
    let log_m = (-0.5f32).mul_add(z, (p * f) * z) + f;
    let log_u = e.mul_add(LN2_HI, e.mul_add(LN2_LO, log_m));
    log_u + (y - (u - 1.0)) / u
}

/// softplus(a) = ln(1 + e^a), the identity above 20 as the reference's
/// threshold has it; written `max(a, 0) + ln(1 + e^-|a|)` so neither branch
/// overflows.
#[inline(always)]
pub fn softplus_portable(a: f32) -> f32 {
    if a > 20.0 {
        a
    } else {
        let pos = if a > 0.0 { a } else { 0.0 };
        pos + log1p_portable(exp_portable(-a.abs()))
    }
}

/// 1 / (1 + e^-x).
#[inline(always)]
pub fn sigmoid_portable(x: f32) -> f32 {
    1.0 / (1.0 + exp_portable(-x))
}

/// v / (1 + e^-v), SiLU in the division form the GPU kernels use.
#[inline(always)]
pub fn silu_portable(v: f32) -> f32 {
    v / (1.0 + exp_portable(-v))
}

/// d/dv silu(v) = sigma(v) (1 + v (1 - sigma(v))).
#[inline(always)]
pub fn silu_grad_portable(v: f32) -> f32 {
    let s = sigmoid_portable(v);
    s * v.mul_add(1.0 - s, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ulps(a: f32, b: f64) -> f64 {
        let b32 = b as f32;
        let step = f64::from((b32.abs().max(f32::MIN_POSITIVE)).next_up()) - f64::from(b32.abs());
        (f64::from(a) - b).abs() / step
    }

    fn sweep(lo: f32, hi: f32, n: usize) -> impl Iterator<Item = f32> {
        (0..=n).map(move |i| lo + (hi - lo) * (i as f32 / n as f32))
    }

    #[test]
    fn exp_is_within_two_ulps() {
        let mut worst = 0f64;
        for x in sweep(-87.3, 88.3, 400_000) {
            worst = worst.max(ulps(exp_portable(x), f64::from(x).exp()));
        }
        assert!(worst <= 2.0, "worst {worst} ulp");
        assert_eq!(exp_portable(0.0), 1.0);
        assert!(exp_portable(f32::NAN).is_nan());
        assert!(exp_portable(-1000.0).is_normal());
        assert!(exp_portable(1000.0).is_finite());
    }

    #[test]
    fn log1p_is_within_two_ulps() {
        let mut worst = 0f64;
        for y in sweep(0.0, 1.0, 400_000).chain([1e-30, 1e-10, 1e-7, 3e-7]) {
            worst = worst.max(ulps(log1p_portable(y), f64::from(y).ln_1p()));
        }
        assert!(worst <= 2.0, "worst {worst} ulp");
    }

    #[test]
    fn softplus_matches_f64() {
        for a in sweep(-30.0, 30.0, 100_000) {
            let want = if a > 20.0 {
                f64::from(a)
            } else {
                f64::from(a).exp().ln_1p()
            };
            let rel = (f64::from(softplus_portable(a)) - want).abs() / want.max(1e-30);
            assert!(rel <= 3e-7, "a={a}: rel {rel:.3e}");
        }
        assert!(softplus_portable(f32::NAN).is_nan());
    }

    #[test]
    fn silu_and_its_gradient_match_f64() {
        for v in sweep(-20.0, 20.0, 50_000) {
            let s = 1.0 / (1.0 + (-f64::from(v)).exp());
            let want = f64::from(v) * s;
            assert!((f64::from(silu_portable(v)) - want).abs() <= 1e-6 * want.abs().max(1e-3));
            let dwant = s * (1.0 + f64::from(v) * (1.0 - s));
            assert!((f64::from(silu_grad_portable(v)) - dwant).abs() <= 2e-6);
        }
    }
}
