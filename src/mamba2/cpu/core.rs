//! The Mamba-2 CPU math shared by decode and prefill.
//!
//! Every expression is fixed element by element: fused multiply-adds in a
//! set order, the portable transcendentals of [`crate::ops::portable_math`]
//! and the sixteen-lane sums of [`crate::ops::fold16`]. The scan runs along
//! a head's channels and the conv along the channels of a token, so no sum
//! crosses SIMD lanes, and the result does not depend on the instruction
//! set. The order follows the reference's step: conv bias first, SiLU as
//! `v / (1 + e^-v)`, `B * (x dt)`, the state as `s dA + B x dt`, D last.
//! The dt clamp after the softplus comes from the reference's chunked
//! forward (its step has none), so decode and prefill clamp alike.

use pulp::{Arch, Simd, WithSimd};

use super::super::config::Mamba2Config;
use super::super::weights::Mamba2LayerWeights;
use crate::ops::fold16::sum_sq_fold16;
use crate::ops::norms::layer_norm_row;
use crate::ops::portable_math::{exp_portable, silu_portable, softplus_portable};

/// `dst = (src * r) * w` with `r = 1 / sqrt(sum(src^2) / n + eps)`.
#[inline(always)]
pub(crate) fn rms_row(dst: &mut [f32], src: &[f32], w: &[f32], eps: f32) {
    let r = 1.0 / (sum_sq_fold16(src) / src.len() as f32 + eps).sqrt();
    for ((d, &s), &wv) in dst.iter_mut().zip(src).zip(w) {
        *d = (s * r) * wv;
    }
}

/// A block norm of one row: [`rms_row`], or LayerNorm for a model whose
/// norms carry a bias (`b` then holds the row's width).
#[inline(always)]
pub(crate) fn block_norm_row(dst: &mut [f32], src: &[f32], (w, b): (&[f32], &[f32]), eps: f32) {
    if b.is_empty() {
        rms_row(dst, src, w, eps);
    } else {
        layer_norm_row(dst, src, (w, b), eps);
    }
}

/// One token's conv and SiLU over every channel:
/// `out[c] = silu(bias[c] + sum_k w[c][k] x[t - W + 1 + k][c])`, the sum
/// started from the bias and taken oldest tap first. `hist` holds the
/// previous `W - 1` inputs, oldest first.
#[inline(always)]
pub(crate) fn conv_silu_token(out: &mut [f32], cur: &[f32], hist: &[f32], lw: &Mamba2LayerWeights) {
    let cd = cur.len();
    let width = lw.conv_w.len() / cd;
    for (c, (o, &xc)) in out.iter_mut().zip(cur).enumerate() {
        let taps = &lw.conv_w[c * width..(c + 1) * width];
        let mut v = lw.conv_b.get(c).copied().unwrap_or(0.0);
        for (k, &wk) in taps[..width - 1].iter().enumerate() {
            v = wk.mul_add(hist[k * cd + c], v);
        }
        v = taps[width - 1].mul_add(xc, v);
        *o = silu_portable(v);
    }
}

/// Append this token's raw conv input to the history, dropping the oldest.
#[inline(always)]
pub(crate) fn conv_hist_push(hist: &mut [f32], cur: &[f32]) {
    let cd = cur.len();
    if hist.is_empty() {
        return;
    }
    hist.copy_within(cd.., 0);
    let last = hist.len() - cd;
    hist[last..].copy_from_slice(cur);
}

/// dt and dA of one head: `dt = clamp(softplus(raw + bias), lo, hi)` (a
/// NaN passes through), `dA = exp(dt * -exp(a_log))`.
#[inline(always)]
pub(crate) fn dt_da(raw: f32, bias: f32, a_log: f32, (lo, hi): (f32, f32)) -> (f32, f32) {
    // clamp keeps a NaN a NaN; the config guarantees lo <= hi.
    let dt = softplus_portable(raw + bias).clamp(lo, hi);
    (dt, exp_portable(dt * -exp_portable(a_log)))
}

/// One head's scan for one token. The state is `[N, P]`; for each channel
/// p, with `xdt = x[p] dt`, the state entries update in ascending n as
/// `s = fma(s, dA, B[n] xdt)` and the readout accumulates
/// `y = fma(s, C[n], y)` from zero; D is added last, `y = fma(x, D, y)`.
pub(crate) struct HeadScan<'a> {
    pub state: &'a mut [f32],
    pub x: &'a [f32],
    pub b: &'a [f32],
    pub c: &'a [f32],
    pub dt: f32,
    pub da: f32,
    /// D for this head: one value, or one per channel.
    pub d: DSkip<'a>,
    pub y: &'a mut [f32],
}

#[derive(Clone, Copy)]
pub(crate) enum DSkip<'a> {
    Head(f32),
    Channel(&'a [f32]),
}

impl WithSimd for HeadScan<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let p_len = self.x.len();
        let lanes = S::F32_LANES;
        let whole = p_len / lanes * lanes;
        let vdt = simd.splat_f32s(self.dt);
        let vda = simd.splat_f32s(self.da);
        let d_at = |p: usize| match self.d {
            DSkip::Head(d) => d,
            DSkip::Channel(d) => d[p],
        };
        // Whole vectors of channels: y stays in a register across n.
        for p0 in (0..whole).step_by(lanes) {
            let (xv, _) = S::as_simd_f32s(&self.x[p0..p0 + lanes]);
            let xdt = simd.mul_f32s(xv[0], vdt);
            let mut y = simd.splat_f32s(0.0);
            for (n, (&bn, &cn)) in self.b.iter().zip(self.c).enumerate() {
                let row = &mut self.state[n * p_len + p0..n * p_len + p0 + lanes];
                let (sv, _) = S::as_mut_simd_f32s(row);
                let s = simd.mul_add_f32s(sv[0], vda, simd.mul_f32s(simd.splat_f32s(bn), xdt));
                sv[0] = s;
                y = simd.mul_add_f32s(s, simd.splat_f32s(cn), y);
            }
            let dv = match self.d {
                DSkip::Head(d) => simd.splat_f32s(d),
                DSkip::Channel(d) => S::as_simd_f32s(&d[p0..p0 + lanes]).0[0],
            };
            let (yo, _) = S::as_mut_simd_f32s(&mut self.y[p0..p0 + lanes]);
            yo[0] = simd.mul_add_f32s(xv[0], dv, y);
        }
        // The channels past the last whole vector, the same chain each.
        for p in whole..p_len {
            let xdt = self.x[p] * self.dt;
            let mut y = 0.0f32;
            for (n, (&bn, &cn)) in self.b.iter().zip(self.c).enumerate() {
                let s = &mut self.state[n * p_len + p];
                *s = s.mul_add(self.da, bn * xdt);
                y = s.mul_add(cn, y);
            }
            self.y[p] = self.x[p].mul_add(d_at(p), y);
        }
    }
}

#[inline(always)]
pub(crate) fn head_scan(scan: HeadScan<'_>) {
    Arch::new().dispatch(scan);
}

/// The mixer's output before out_proj, `[d_mlp | d_ssm]`: the inner MLP
/// `silu(z0) * x0`, then y gated and normalised per group.
#[inline(always)]
pub(crate) fn gate_token(
    out: &mut [f32],
    y: &[f32],
    proj: &[f32],
    lw: &Mamba2LayerWeights,
    cfg: &Mamba2Config,
) {
    let (d_mlp, d_ssm) = (cfg.d_mlp(), cfg.ssm_dim());
    let (z0, rest) = proj.split_at(d_mlp);
    let (x0, rest) = rest.split_at(d_mlp);
    let z = &rest[..d_ssm];
    let (o_mlp, o_ssm) = out.split_at_mut(d_mlp);
    for ((o, &zv), &xv) in o_mlp.iter_mut().zip(z0).zip(x0) {
        *o = silu_portable(zv) * xv;
    }
    if !cfg.rmsnorm {
        for ((o, &yv), &zv) in o_ssm.iter_mut().zip(y).zip(z) {
            *o = yv * silu_portable(zv);
        }
        return;
    }
    let gs = d_ssm / cfg.ngroups;
    let eps = cfg.gated_norm_eps;
    for (((o_g, y_g), z_g), w_g) in o_ssm
        .chunks_exact_mut(gs)
        .zip(y.chunks_exact(gs))
        .zip(z.chunks_exact(gs))
        .zip(lw.gate_norm_weight.chunks_exact(gs))
    {
        if cfg.norm_before_gate {
            let r = 1.0 / (sum_sq_fold16(y_g) / gs as f32 + eps).sqrt();
            for (((o, &yv), &zv), &wv) in o_g.iter_mut().zip(y_g).zip(z_g).zip(w_g) {
                *o = ((yv * r) * wv) * silu_portable(zv);
            }
        } else {
            for ((o, &yv), &zv) in o_g.iter_mut().zip(y_g).zip(z_g) {
                *o = yv * silu_portable(zv);
            }
            let r = 1.0 / (sum_sq_fold16(o_g) / gs as f32 + eps).sqrt();
            for (o, &wv) in o_g.iter_mut().zip(w_g) {
                *o = (*o * r) * wv;
            }
        }
    }
}

/// One token's SSM over every head: dt and dA, then the head's scan into
/// `y` (`[d_ssm]`). `xbc` is the token's conv output.
#[inline(always)]
pub(crate) fn ssm_token(
    (state, y): (&mut [f32], &mut [f32]),
    xbc: &[f32],
    dt_raw: &[f32],
    lw: &Mamba2LayerWeights,
    cfg: &Mamba2Config,
) {
    let (n, p_len, g_n, nh) = (cfg.d_state, cfg.headdim, cfg.ngroups, cfg.nheads());
    let d_ssm = cfg.ssm_dim();
    let (x, bc) = xbc.split_at(d_ssm);
    let (b_all, c_all) = bc.split_at(g_n * n);
    for (h, (st, y_h)) in state
        .chunks_exact_mut(n * p_len)
        .zip(y.chunks_exact_mut(p_len))
        .enumerate()
    {
        let g = h / (nh / g_n);
        let (dt, da) = dt_da(dt_raw[h], lw.dt_bias[h], lw.a_log[h], cfg.dt_limit);
        head_scan(HeadScan {
            state: st,
            x: &x[h * p_len..(h + 1) * p_len],
            b: &b_all[g * n..(g + 1) * n],
            c: &c_all[g * n..(g + 1) * n],
            dt,
            da,
            d: if cfg.d_has_hdim {
                DSkip::Channel(&lw.d_param[h * p_len..(h + 1) * p_len])
            } else {
                DSkip::Head(lw.d_param[h])
            },
            y: y_h,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SIMD scan equals the same chain run one channel at a time.
    #[test]
    fn scan_vectors_and_tail_agree_with_the_scalar_chain() {
        for p_len in [1usize, 3, 4, 8, 13, 64] {
            let n = 7usize;
            let mut s_vec: Vec<f32> = (0..n * p_len).map(|i| (i as f32 * 0.37).sin()).collect();
            let mut s_ref = s_vec.clone();
            let x: Vec<f32> = (0..p_len).map(|i| (i as f32 * 0.91).cos()).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 * 1.3).sin()).collect();
            let c: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos()).collect();
            let (dt, da, d) = (0.07f32, 0.93f32, 0.5f32);
            let mut y = vec![0f32; p_len];
            head_scan(HeadScan {
                state: &mut s_vec,
                x: &x,
                b: &b,
                c: &c,
                dt,
                da,
                d: DSkip::Head(d),
                y: &mut y,
            });
            for p in 0..p_len {
                let xdt = x[p] * dt;
                let mut yr = 0f32;
                for k in 0..n {
                    let s = &mut s_ref[k * p_len + p];
                    *s = s.mul_add(da, b[k] * xdt);
                    yr = s.mul_add(c[k], yr);
                }
                assert_eq!(y[p].to_bits(), x[p].mul_add(d, yr).to_bits(), "p={p}");
            }
            assert_eq!(s_vec, s_ref);
        }
    }
}
