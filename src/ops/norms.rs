//! Shared normalization operations for Mamba SSM and Mamba-3 SISO.
//!
//! - `rms_norm` — standard RMSNorm (scale only, no bias)
//! - `rms_norm_weighted` — RMSNorm with learnable weight
//! - `bcnorm` — per-group RMSNorm for B/C (Mamba-3)
//! - `rmsnorm_gated` — RMSNormGated: norm(y) * weight * SiLU(z) (Mamba-3)
//! - `layer_norm_row` — LayerNorm with a bias, the block norm of a model
//!   built with `rms_norm = false`

use super::fast_math::fast_exp_scalar;
use super::fold16::{dot_fold16, sum_sq_fold16};

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

    for (g, inv_rms_slot) in inv_rms_out[..ngroups].iter_mut().enumerate() {
        let start = g * d_state;
        let end = start + d_state;
        let group = &x[start..end];

        let mut sum_sq = 0.0_f32;
        for &v in group {
            sum_sq += v * v;
        }
        let inv_rms = 1.0 / (sum_sq / d_state as f32 + eps).sqrt();
        *inv_rms_slot = inv_rms;

        for ((o, &v), &w) in out[start..end].iter_mut().zip(group).zip(weight) {
            *o = v * inv_rms * w;
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
        for &yv in &y[g_start..g_end] {
            sum_sq += yv * yv;
        }
        let inv_rms = 1.0 / (sum_sq / g_len as f32 + eps).sqrt();

        for i in g_start..g_end {
            let zi = z[i];
            let silu_z = zi / (1.0 + fast_exp_scalar(-zi));
            out[i] = y[i] * inv_rms * weight[i] * silu_z;
        }
    }
}

/// FalconMamba's weightless RMS norm of one part of an x_proj row, in
/// place: `v / sqrt(mean(v^2) + eps)`, the sum of squares in the
/// sixteen-lane order. Returns the reciprocal scale the backward takes.
#[inline]
pub fn weightless_rms_norm(v: &mut [f32], eps: f32) -> f32 {
    let r = 1.0 / (sum_sq_fold16(v) / v.len() as f32 + eps).sqrt();
    for x in v.iter_mut() {
        *x *= r;
    }
    r
}

/// The backward of [`weightless_rms_norm`] in place: `dy`, the gradient of
/// the normalized part `y`, becomes the gradient of the part before the
/// norm, `r (dy - y mean(dy y))`.
#[inline]
pub fn weightless_rms_norm_backward(dy: &mut [f32], y: &[f32], r: f32) {
    let m = dot_fold16(dy, y) / y.len() as f32;
    for (g, &yv) in dy.iter_mut().zip(y) {
        *g = r * (*g - yv * m);
    }
}

/// FalconMamba's norms over an x_proj row `[dt | B | C]` of part widths
/// `dt_rank`, `d_state` and `d_state`, in place. Returns each part's
/// reciprocal scale.
#[inline]
pub fn bcdt_rms_norm(row: &mut [f32], (dt_rank, d_state): (usize, usize), eps: f32) -> [f32; 3] {
    let (dt, bc) = row.split_at_mut(dt_rank);
    let (b, c) = bc.split_at_mut(d_state);
    [
        weightless_rms_norm(dt, eps),
        weightless_rms_norm(b, eps),
        weightless_rms_norm(&mut c[..d_state], eps),
    ]
}

/// The backward of [`bcdt_rms_norm`] in place: `d_row` takes the gradient
/// of the row before the norms, from the normalized `row` and the scales.
#[inline]
pub fn bcdt_rms_norm_backward(
    d_row: &mut [f32],
    row: &[f32],
    (dt_rank, d_state): (usize, usize),
    r: &[f32],
) {
    let (d_dt, d_bc) = d_row.split_at_mut(dt_rank);
    let (d_b, d_c) = d_bc.split_at_mut(d_state);
    let (dt, bc) = row.split_at(dt_rank);
    let (b, c) = bc.split_at(d_state);
    weightless_rms_norm_backward(d_dt, dt, r[0]);
    weightless_rms_norm_backward(d_b, b, r[1]);
    weightless_rms_norm_backward(&mut d_c[..d_state], &c[..d_state], r[2]);
}

/// LayerNorm of one row with the biased variance (`nn.LayerNorm`), the
/// block norm of a model built with `rms_norm = false`: `out = (x - mean)
/// / std * weight + bias`, `std = sqrt(var + eps)`. The mean, then the
/// centred sum of squares, are left-to-right f32 sums (two passes, as the
/// reference's kernel, so a large mean costs no precision). Returns
/// `[std, mean]` for the backward.
#[inline]
pub fn layer_norm_row(
    out: &mut [f32],
    x: &[f32],
    (weight, bias): (&[f32], &[f32]),
    eps: f32,
) -> [f32; 2] {
    debug_assert_eq!(x.len(), out.len());
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), bias.len());
    let [sd, mean] = layer_norm_stats(x, eps);
    let inv = 1.0 / sd;
    for (((o, &v), &w), &b) in out.iter_mut().zip(x).zip(weight).zip(bias) {
        *o = (v - mean) * inv * w + b;
    }
    [sd, mean]
}

/// [`layer_norm_row`] in place, the same expressions element by element.
/// Returns `[std, mean]`.
#[inline]
pub fn layer_norm_inplace(x: &mut [f32], (weight, bias): (&[f32], &[f32]), eps: f32) -> [f32; 2] {
    let [sd, mean] = layer_norm_stats(x, eps);
    let inv = 1.0 / sd;
    for ((v, &w), &b) in x.iter_mut().zip(weight).zip(bias) {
        *v = (*v - mean) * inv * w + b;
    }
    [sd, mean]
}

/// The statistics [`layer_norm_row`] normalizes a row with, `[std, mean]`,
/// for a backward that recomputes them: the same sums in the same order.
#[inline]
pub fn layer_norm_stats(x: &[f32], eps: f32) -> [f32; 2] {
    let n = x.len() as f32;
    let mut sum = 0.0_f32;
    for &v in x {
        sum += v;
    }
    let mean = sum / n;
    let mut sq = 0.0_f32;
    for &v in x {
        let c = v - mean;
        sq += c * c;
    }
    [(sq / n + eps).sqrt(), mean]
}

/// The backward of [`layer_norm_row`] for one row. With `x_hat = (x -
/// mean) / std` and `g = weight * dy`, the input gradient
/// `(g - mean(g) - x_hat * mean(g * x_hat)) / std` is written into `dx`,
/// or added onto it when `accumulate`; `dy * x_hat` and `dy` accumulate
/// into the weight's and the bias's gradients.
#[inline]
pub fn layer_norm_row_backward(
    dx: &mut [f32],
    (d_weight, d_bias): (&mut [f32], &mut [f32]),
    (dy, x): (&[f32], &[f32]),
    weight: &[f32],
    [sd, mean]: [f32; 2],
    accumulate: bool,
) {
    let n = x.len();
    debug_assert_eq!(n, dx.len());
    debug_assert_eq!(n, dy.len());
    let inv = 1.0 / sd;
    let (mut s_g, mut s_gx) = (0.0_f32, 0.0_f32);
    for ((&v, &d), &w) in x.iter().zip(dy).zip(weight) {
        let x_hat = (v - mean) * inv;
        let g = w * d;
        s_g += g;
        s_gx += g * x_hat;
    }
    let (mean_g, mean_gx) = (s_g / n as f32, s_gx / n as f32);
    for i in 0..n {
        let x_hat = (x[i] - mean) * inv;
        let d = (weight[i] * dy[i] - mean_g - x_hat * mean_gx) * inv;
        dx[i] = if accumulate { dx[i] + d } else { d };
        d_weight[i] += dy[i] * x_hat;
        d_bias[i] += dy[i];
    }
}

/// [`layer_norm_row_backward`] in place, overwriting: `d` holds the
/// gradient of the normalized row and leaves holding the input's, the same
/// expressions element by element.
#[inline]
pub fn layer_norm_row_backward_in_place(
    d: &mut [f32],
    (d_weight, d_bias): (&mut [f32], &mut [f32]),
    x: &[f32],
    weight: &[f32],
    [sd, mean]: [f32; 2],
) {
    let n = x.len();
    debug_assert_eq!(n, d.len());
    let inv = 1.0 / sd;
    let (mut s_g, mut s_gx) = (0.0_f32, 0.0_f32);
    for ((&v, &dy), &w) in x.iter().zip(d.iter()).zip(weight) {
        let x_hat = (v - mean) * inv;
        let g = w * dy;
        s_g += g;
        s_gx += g * x_hat;
    }
    let (mean_g, mean_gx) = (s_g / n as f32, s_gx / n as f32);
    for i in 0..n {
        let x_hat = (x[i] - mean) * inv;
        let dy = d[i];
        d_weight[i] += dy * x_hat;
        d_bias[i] += dy;
        d[i] = (weight[i] * dy - mean_g - x_hat * mean_gx) * inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The backward is the derivative of the norm: against central
    /// differences of `sum(g y)` at every input, each part of a row alike.
    #[test]
    fn bcdt_rms_norm_backward_is_its_derivative() {
        let (dt_rank, d_state, eps) = (5usize, 7usize, 1e-6f32);
        let n = dt_rank + 2 * d_state;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.3 + 0.05)
            .collect();
        let g: Vec<f32> = (0..n).map(|i| ((i * 13 % 7) as f32 - 3.0) * 0.2).collect();
        let loss = |x: &[f32]| {
            let mut y = x.to_vec();
            bcdt_rms_norm(&mut y, (dt_rank, d_state), eps);
            y.iter()
                .zip(&g)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>()
        };
        let mut y = x.clone();
        let r = bcdt_rms_norm(&mut y, (dt_rank, d_state), eps);
        let mut dx = g.clone();
        bcdt_rms_norm_backward(&mut dx, &y, (dt_rank, d_state), &r);
        let h = 1e-3f32;
        for i in 0..n {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[i] += h;
            xm[i] -= h;
            let num = (loss(&xp) - loss(&xm)) / (2.0 * f64::from(h));
            assert!(
                (num - f64::from(dx[i])).abs() < 2e-3,
                "{i}: {num} vs {}",
                dx[i]
            );
        }
        // Each part comes out with a unit root mean square.
        for part in [
            &y[..dt_rank],
            &y[dt_rank..dt_rank + d_state],
            &y[dt_rank + d_state..],
        ] {
            let ms = part.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / part.len() as f64;
            assert!((ms - 1.0).abs() < 1e-4, "{ms}");
        }
    }

    /// LayerNorm against an f64 transcription of `nn.LayerNorm`, and its
    /// backward against central differences of `sum(g y)` taken through
    /// that f64 transcription, for every input, weight and bias element; a
    /// row with a large mean checks the two-pass statistics.
    #[test]
    fn layer_norm_row_matches_the_reference_and_its_derivative() {
        let eps = 1e-5f64;
        for (n, shift) in [(13usize, 0.25f32), (64, 40.0)] {
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.3 + shift)
                .collect();
            let w: Vec<f32> = (0..n)
                .map(|i| 1.0 + ((i * 7 % 5) as f32 - 2.0) * 0.1)
                .collect();
            let b: Vec<f32> = (0..n).map(|i| ((i * 3 % 4) as f32 - 1.5) * 0.05).collect();
            let g: Vec<f32> = (0..n).map(|i| ((i * 13 % 7) as f32 - 3.0) * 0.2).collect();
            let reference = |x: &[f64], w: &[f64], b: &[f64]| -> Vec<f64> {
                let m = x.iter().sum::<f64>() / n as f64;
                let var = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64;
                let sd = (var + eps).sqrt();
                (0..n).map(|i| (x[i] - m) / sd * w[i] + b[i]).collect()
            };
            let wide = |v: &[f32]| v.iter().map(|&a| f64::from(a)).collect::<Vec<f64>>();
            let (xd, wd, bd, gd) = (wide(&x), wide(&w), wide(&b), wide(&g));
            let mut y = vec![0.0f32; n];
            let stats = layer_norm_row(&mut y, &x, (&w, &b), eps as f32);
            for (i, (&got, want)) in y.iter().zip(reference(&xd, &wd, &bd)).enumerate() {
                assert!(
                    (f64::from(got) - want).abs() < 1e-5,
                    "y[{i}]: {got} vs {want}"
                );
            }
            let loss = |x: &[f64], w: &[f64], b: &[f64]| -> f64 {
                reference(x, w, b).iter().zip(&gd).map(|(a, c)| a * c).sum()
            };
            let mut dx = vec![0.5f32; n];
            let (mut dw, mut db) = (vec![0.0f32; n], vec![0.0f32; n]);
            layer_norm_row_backward(&mut dx, (&mut dw, &mut db), (&g, &x), &w, stats, true);
            // The in-place form gives the same bits, and the stats a backward
            // recomputes are the forward's.
            assert_eq!(layer_norm_stats(&x, eps as f32), stats);
            let mut d_in = g.clone();
            let (mut dw_in, mut db_in) = (vec![0.0f32; n], vec![0.0f32; n]);
            layer_norm_row_backward_in_place(&mut d_in, (&mut dw_in, &mut db_in), &x, &w, stats);
            let mut d_out = vec![0.0f32; n];
            let (mut dw_out, mut db_out) = (vec![0.0f32; n], vec![0.0f32; n]);
            layer_norm_row_backward(
                &mut d_out,
                (&mut dw_out, &mut db_out),
                (&g, &x),
                &w,
                stats,
                false,
            );
            assert_eq!((&d_in, &dw_in, &db_in), (&d_out, &dw_out, &db_out));
            let h = 1e-6f64;
            let fd = |which: usize, i: usize| -> f64 {
                let (mut p, mut m) = (
                    [xd.clone(), wd.clone(), bd.clone()],
                    [xd.clone(), wd.clone(), bd.clone()],
                );
                p[which][i] += h;
                m[which][i] -= h;
                (loss(&p[0], &p[1], &p[2]) - loss(&m[0], &m[1], &m[2])) / (2.0 * h)
            };
            for i in 0..n {
                for (which, got) in [
                    (0, f64::from(dx[i]) - 0.5),
                    (1, f64::from(dw[i])),
                    (2, f64::from(db[i])),
                ] {
                    let want = fd(which, i);
                    assert!(
                        (got - want).abs() < 1e-4 * want.abs().max(1.0),
                        "n {n} part {which}[{i}]: {got} vs {want}"
                    );
                }
            }
        }
    }

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
