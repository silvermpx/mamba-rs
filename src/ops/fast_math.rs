//! Vectorized f32 exp() using the Cephes degree-7 polynomial (Moshier).
//!
//! Provides SIMD-accelerated exp() for NEON (AArch64) and AVX2 (x86_64),
//! plus a scalar fallback. Max relative error: ~1.7e-7 (~1.4 ULP).
//!
//! Algorithm (identical to Cephes expf.c, Eigen pexp_float, sse_mathfun):
//!   1. Cody-Waite range reduction: x = n * ln(2) + r
//!   2. Degree-7 polynomial: P(r)*r^2 + r + 1.0
//!   3. Reconstruction: exp(x) = poly * 2^n (IEEE 754 exponent bit shift)
//!
//! Sources:
//!   - Cephes expf.c (Stephen Moshier): coefficients + evaluation structure
//!   - Eigen GenericPacketMathFunctions.h pexp_float: SIMD reference
//!   - sse_mathfun.h (Julien Pommier): SSE reference

/// RMSNorm epsilon constant.
pub const RMS_NORM_EPS: f32 = 1e-5;

// Cody-Waite range reduction constants (same across Cephes/Eigen/sse_mathfun)
const EXP_HI: f32 = 88.37626; // ln(FLT_MAX)
const EXP_LO: f32 = -88.37626;
const LOG2EF: f32 = std::f32::consts::LOG2_E;
const LN2_HI: f32 = 0.6933594; // upper bits of ln(2), exactly representable
const LN2_LO: f32 = -2.1219444e-4; // lower bits, LN2_HI + LN2_LO = ln(2) to 12 digits

// Cephes minimax polynomial coefficients for exp() on [-ln2/2, ln2/2].
// Structure: P(r) * r^2 + r + 1.0 where P is degree-5 Horner in r.
// Effective degree: 7 (p0*r^7 + p1*r^6 + ... + p5*r^2 + r + 1.0).
const CEPH_P0: f32 = 1.987569e-4; // ~1/5040 (x^7/7!)
const CEPH_P1: f32 = 1.398_2e-3; // ~1/720  (x^6/6!)
const CEPH_P2: f32 = 8.333452e-3; // ~1/120  (x^5/5!)
const CEPH_P3: f32 = 4.166_58e-2; // ~1/24   (x^4/4!)
const CEPH_P4: f32 = 1.666667e-1; // ~1/6    (x^3/3!)
const CEPH_P5: f32 = 0.5; // 1/2 (x^2/2!)

/// IEEE 754 round-to-nearest-even (banker's rounding).
/// Matches NEON `vrndnq_f32` and AVX2 `_mm256_round_ps(NEAREST)`.
/// Rust's `f32::round()` rounds ties away from zero — NOT the same.
#[inline(always)]
fn roundeven(x: f32) -> f32 {
    // The magic constant trick: adding and subtracting 2^23 forces rounding
    // to the nearest integer using the current FP rounding mode (default = nearest-even).
    // Must use sign-preserving variant for negative numbers.
    const MAGIC: f32 = 8_388_608.0; // 2^23
    if x >= 0.0 {
        if x >= MAGIC { x } else { (x + MAGIC) - MAGIC }
    } else {
        if -x >= MAGIC { x } else { (x - MAGIC) + MAGIC }
    }
}

/// Scalar fast exp (Cephes degree-7). Max relative error ~1.7e-7 (~1.4 ULP).
#[inline(always)]
pub fn fast_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(EXP_LO, EXP_HI);
    // Cody-Waite range reduction: x = n * ln(2) + r
    // Round to nearest even (IEEE 754 default) to match NEON vrndnq / AVX2 roundps.
    // f32::round() rounds ties away from zero; we need ties-to-even for SIMD parity.
    let n = roundeven(x * LOG2EF);
    let r = x - n * LN2_HI - n * LN2_LO;
    // Cephes degree-7: P(r) * r^2 + r + 1.0
    let r2 = r * r;
    let p = CEPH_P0;
    let p = p * r + CEPH_P1;
    let p = p * r + CEPH_P2;
    let p = p * r + CEPH_P3;
    let p = p * r + CEPH_P4;
    let p = p * r + CEPH_P5;
    let p = p * r2 + r + 1.0;
    // Reconstruct: exp(x) = p * 2^n via IEEE 754 exponent bit manipulation
    let n_i = n as i32;
    let pow2n = f32::from_bits(((n_i + 127) as u32) << 23);
    p * pow2n
}

/// Apply fast exp() to each element in a slice (in-place).
/// Dispatches to SIMD when available: NEON (AArch64, 4-wide), AVX2+FMA (x86_64, 8-wide).
#[inline]
pub fn fast_exp_inplace(buf: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        fast_exp_neon(buf);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { fast_exp_avx2(buf) };
            return;
        }
        for v in buf.iter_mut() {
            *v = fast_exp_scalar(*v);
        }
    }
}

// ---------------------------------------------------------------------------
// AArch64 NEON implementation (4-wide float32x4_t)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn fast_exp_neon(buf: &mut [f32]) {
    use std::arch::aarch64::*;

    let len = buf.len();
    let chunks = len / 4;
    let remainder = len % 4;

    unsafe {
        let v_log2ef = vdupq_n_f32(LOG2EF);
        // v_half no longer needed — using vrndnq_f32 (round to nearest even)
        let v_ln2_hi = vdupq_n_f32(LN2_HI);
        let v_ln2_lo = vdupq_n_f32(LN2_LO);
        let v_p0 = vdupq_n_f32(CEPH_P0);
        let v_p1 = vdupq_n_f32(CEPH_P1);
        let v_p2 = vdupq_n_f32(CEPH_P2);
        let v_p3 = vdupq_n_f32(CEPH_P3);
        let v_p4 = vdupq_n_f32(CEPH_P4);
        let v_p5 = vdupq_n_f32(CEPH_P5);
        let v_one = vdupq_n_f32(1.0);
        let v_hi = vdupq_n_f32(EXP_HI);
        let v_lo = vdupq_n_f32(EXP_LO);
        let v_127 = vdupq_n_s32(127);

        let ptr = buf.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(ptr.add(off));

            // Clamp
            let x = vminq_f32(vmaxq_f32(x, v_lo), v_hi);

            // n = round(x * log2e) — using vrndnq (round to nearest even)
            // avoids double-rounding hazard of floor(x * log2e + 0.5)
            let n_f = vrndnq_f32(vmulq_f32(x, v_log2ef));

            // r = x - n * ln2_hi - n * ln2_lo (Cody-Waite)
            let r = vfmsq_f32(vfmsq_f32(x, n_f, v_ln2_hi), n_f, v_ln2_lo);

            // Cephes degree-7: P(r) * r^2 + r + 1.0
            let r2 = vmulq_f32(r, r);
            let p = vfmaq_f32(v_p1, v_p0, r);
            let p = vfmaq_f32(v_p2, p, r);
            let p = vfmaq_f32(v_p3, p, r);
            let p = vfmaq_f32(v_p4, p, r);
            let p = vfmaq_f32(v_p5, p, r);
            let p = vfmaq_f32(r, p, r2);
            let p = vaddq_f32(p, v_one);

            // 2^n via exponent bit shift
            let n_i = vcvtq_s32_f32(n_f);
            let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n_i, v_127)));

            let result = vmulq_f32(p, pow2n);
            vst1q_f32(ptr.add(off), result);
        }
    }

    // Scalar tail
    let tail_start = chunks * 4;
    for v in &mut buf[tail_start..tail_start + remainder] {
        *v = fast_exp_scalar(*v);
    }
}

// ---------------------------------------------------------------------------
// x86_64 AVX2+FMA implementation (8-wide __m256)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fast_exp_avx2(buf: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let len = buf.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let v_log2ef = _mm256_set1_ps(LOG2EF);
        // v_half no longer needed — using _mm256_round_ps (round to nearest even)
        let v_ln2_hi = _mm256_set1_ps(LN2_HI);
        let v_ln2_lo = _mm256_set1_ps(LN2_LO);
        let v_p0 = _mm256_set1_ps(CEPH_P0);
        let v_p1 = _mm256_set1_ps(CEPH_P1);
        let v_p2 = _mm256_set1_ps(CEPH_P2);
        let v_p3 = _mm256_set1_ps(CEPH_P3);
        let v_p4 = _mm256_set1_ps(CEPH_P4);
        let v_p5 = _mm256_set1_ps(CEPH_P5);
        let v_one = _mm256_set1_ps(1.0);
        let v_hi = _mm256_set1_ps(EXP_HI);
        let v_lo = _mm256_set1_ps(EXP_LO);
        let v_127 = _mm256_set1_epi32(127);

        let ptr = buf.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 8;
            let x = _mm256_loadu_ps(ptr.add(off));

            // Clamp
            let x = _mm256_min_ps(_mm256_max_ps(x, v_lo), v_hi);

            // n = round(x * log2e) — using roundps (round to nearest even)
            // avoids double-rounding hazard of floor(x * log2e + 0.5)
            let n_f = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
                _mm256_mul_ps(x, v_log2ef),
            );

            // r = x - n * ln2_hi - n * ln2_lo (Cody-Waite)
            let r = _mm256_sub_ps(
                _mm256_sub_ps(x, _mm256_mul_ps(n_f, v_ln2_hi)),
                _mm256_mul_ps(n_f, v_ln2_lo),
            );

            // Cephes degree-7: P(r) * r^2 + r + 1.0
            let r2 = _mm256_mul_ps(r, r);
            let p = _mm256_fmadd_ps(v_p0, r, v_p1);
            let p = _mm256_fmadd_ps(p, r, v_p2);
            let p = _mm256_fmadd_ps(p, r, v_p3);
            let p = _mm256_fmadd_ps(p, r, v_p4);
            let p = _mm256_fmadd_ps(p, r, v_p5);
            let p = _mm256_fmadd_ps(p, r2, r);
            let p = _mm256_add_ps(p, v_one);

            // 2^n via exponent bit shift
            let n_i = _mm256_cvtps_epi32(n_f);
            let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n_i, v_127), 23));

            let result = _mm256_mul_ps(p, pow2n);
            _mm256_storeu_ps(ptr.add(off), result);
        }

        // Scalar tail
        let tail_start = chunks * 8;
        for v in &mut buf[tail_start..tail_start + remainder] {
            *v = fast_exp_scalar(*v);
        }
    } // unsafe
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_exp_scalar_accuracy() {
        let test_vals: Vec<f32> = (-800..=200).map(|i| i as f32 * 0.1).collect();
        let mut max_rel_err = 0.0_f64;

        for &x in &test_vals {
            let fast = fast_exp_scalar(x) as f64;
            let exact = (x as f64).exp();
            if exact > 1e-30 {
                let rel_err = ((fast - exact) / exact).abs();
                max_rel_err = max_rel_err.max(rel_err);
            }
        }

        assert!(
            max_rel_err < 2e-7,
            "scalar exp max relative error {max_rel_err:.2e} exceeds 2e-7"
        );
    }

    #[test]
    fn test_fast_exp_inplace_accuracy() {
        let mut buf: Vec<f32> = (-400..=100).map(|i| i as f32 * 0.1).collect();
        let expected: Vec<f32> = buf.iter().map(|&x| x.exp()).collect();

        fast_exp_inplace(&mut buf);

        for (i, (&fast, &exact)) in buf.iter().zip(expected.iter()).enumerate() {
            if exact > 1e-30 {
                let rel_err = ((fast - exact) / exact).abs();
                assert!(
                    rel_err < 2e-7,
                    "fast_exp_inplace[{i}]: fast={fast}, exact={exact}, rel_err={rel_err:.2e}",
                );
            }
        }
    }

    #[test]
    fn test_fast_exp_edge_cases() {
        assert!(fast_exp_scalar(-100.0) >= 0.0);
        assert!(fast_exp_scalar(-100.0).is_finite());
        assert!(fast_exp_scalar(100.0).is_finite());
        assert!((fast_exp_scalar(0.0) - 1.0).abs() < 1e-8);
        assert!((fast_exp_scalar(-0.01) - (-0.01_f32).exp()).abs() < 1e-8);
        assert!((fast_exp_scalar(-0.5) - (-0.5_f32).exp()).abs() < 1e-7);
        assert!((fast_exp_scalar(-1.0) - (-1.0_f32).exp()).abs() < 1e-7);
    }

    #[test]
    fn test_fast_exp_ssm_range() {
        // SSM typical range: delta * a_neg where delta in [0.01, 5.0], a_neg in [-16, -1]
        for i in 0..1000 {
            let x = -(i as f32) * 0.08;
            let fast = fast_exp_scalar(x);
            let exact = x.exp();
            if exact > 1e-30 {
                let rel_err = ((fast as f64 - exact as f64) / exact as f64).abs();
                assert!(
                    rel_err < 2e-7,
                    "SSM range: x={x}, fast={fast}, exact={exact}, rel_err={rel_err:.2e}"
                );
            }
        }
    }
}
