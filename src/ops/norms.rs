#![allow(clippy::needless_range_loop)]
//! Shared normalization operations for Mamba SSM and Mamba-3 SISO.
//!
//! - `rms_norm` — standard RMSNorm (scale only, no bias)
//! - `rms_norm_weighted` — RMSNorm with learnable weight
//! - `bcnorm` — per-group RMSNorm for B/C (Mamba-3)
//! - `rmsnorm_gated` — RMSNormGated: norm(y) * weight * SiLU(z) (Mamba-3)

use super::fast_math::fast_exp_scalar;

/// RMSNorm: `out[i] = x[i] / rms * weight[i]`
/// Returns inv_rms for backward.
#[inline]
pub fn rms_norm_weighted(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) -> f32 {
    let n = x.len();
    debug_assert_eq!(n, weight.len());
    debug_assert_eq!(n, out.len());

    let mut sum_sq = 0.0_f32;
    for &v in x {
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    for i in 0..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
    inv_rms
}

/// RMSNorm in-place: `x[i] = x[i] / rms * weight[i]`
#[inline]
pub fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) -> f32 {
    let n = x.len();
    let mut sum_sq = 0.0_f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    for i in 0..n {
        x[i] *= inv_rms * weight[i];
    }
    inv_rms
}

/// BCNorm: per-group RMSNorm(x) * weight for B/C in Mamba-3.
/// x is `[ngroups * d_state]`, weight is `[d_state]` (shared across groups).
/// Returns per-group inv_rms values.
#[inline]
pub fn bcnorm(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    ngroups: usize,
    d_state: usize,
    eps: f32,
    inv_rms_out: &mut [f32],
) {
    debug_assert_eq!(x.len(), ngroups * d_state);
    debug_assert_eq!(out.len(), ngroups * d_state);
    debug_assert_eq!(weight.len(), d_state);
    debug_assert!(inv_rms_out.len() >= ngroups);

    for g in 0..ngroups {
        let start = g * d_state;
        let end = start + d_state;
        let group = &x[start..end];

        let mut sum_sq = 0.0_f32;
        for &v in group {
            sum_sq += v * v;
        }
        let inv_rms = 1.0 / (sum_sq / d_state as f32 + eps).sqrt();
        inv_rms_out[g] = inv_rms;

        for i in 0..d_state {
            out[start + i] = group[i] * inv_rms * weight[i];
        }
    }
}

/// RMSNormGated: per-group RMSNorm(y) * weight * SiLU(z).
/// group_size = headdim (Mamba-3 convention).
#[inline]
pub fn rmsnorm_gated(
    out: &mut [f32],
    y: &[f32],
    z: &[f32],
    weight: &[f32],
    group_size: usize,
    eps: f32,
) {
    let dim = y.len();
    debug_assert_eq!(dim, z.len());
    debug_assert_eq!(dim, weight.len());
    debug_assert_eq!(dim, out.len());

    for g_start in (0..dim).step_by(group_size) {
        let g_end = (g_start + group_size).min(dim);
        let g_len = g_end - g_start;

        let mut sum_sq = 0.0_f32;
        for i in g_start..g_end {
            sum_sq += y[i] * y[i];
        }
        let inv_rms = 1.0 / (sum_sq / g_len as f32 + eps).sqrt();

        for i in g_start..g_end {
            let zi = z[i];
            let silu_z = zi / (1.0 + fast_exp_scalar(-zi));
            out[i] = y[i] * inv_rms * weight[i] * silu_z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_weighted_unit_weight() {
        let x = vec![3.0, 4.0];
        let w = vec![1.0, 1.0];
        let mut out = vec![0.0; 2];
        let inv_rms = rms_norm_weighted(&mut out, &x, &w, 1e-5);

        let rms = (12.5_f32).sqrt(); // (9+16)/2 = 12.5
        let expected_inv = 1.0 / rms;
        assert!((inv_rms - expected_inv).abs() < 1e-5);
        assert!((out[0] - 3.0 * expected_inv).abs() < 1e-5);
        assert!((out[1] - 4.0 * expected_inv).abs() < 1e-5);
    }

    #[test]
    fn test_bcnorm_per_group() {
        let x = vec![1.0, 2.0, 3.0, 4.0]; // 2 groups of 2
        let w = vec![1.0, 1.0];
        let mut out = vec![0.0; 4];
        let mut inv_rms = vec![0.0; 2];
        bcnorm(&mut out, &x, &w, 2, 2, 1e-5, &mut inv_rms);

        // Group 0: rms = sqrt((1+4)/2) = sqrt(2.5)
        // Group 1: rms = sqrt((9+16)/2) = sqrt(12.5)
        assert!(inv_rms[0] > 0.0);
        assert!(inv_rms[1] > 0.0);
        assert!(inv_rms[0] > inv_rms[1]); // smaller values → larger inv_rms
    }
}
