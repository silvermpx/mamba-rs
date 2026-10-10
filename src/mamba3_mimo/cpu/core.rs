//! The Mamba-3 MIMO CPU math shared by every path: discretisation, the
//! half-split rotation, the SSM rows and the output gate.
//!
//! The rows are generic over the rank so the released ranks get unrolled
//! per-rank loops; a constant and a runtime rank run the same operations
//! in the same order and give the same bits. At rank 1 the row is SISO's
//! row operation for operation.

use pulp::{Arch, Simd, WithSimd};

use super::super::config::{MAX_MIMO_RANK, Mamba3MimoConfig};
use super::super::weights::Mamba3MimoLayerWeights;
use crate::mamba3_siso::cpu::forward::{
    clamp_a, heavy_tail, simd_rms_scale, simd_sum_sq, softplus,
};
use crate::ops::fast_math::fast_exp_scalar;

/// A rank count: [`Const`] for the ranks the released models use, so the
/// per-rank loops unroll, [`Dyn`] for any other.
pub(crate) trait Ranks: Copy + Send + Sync {
    fn get(self) -> usize;
}

#[derive(Clone, Copy)]
pub(crate) struct Const<const R: usize>;

impl<const R: usize> Ranks for Const<R> {
    #[inline(always)]
    fn get(self) -> usize {
        R
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Dyn(pub usize);

impl Ranks for Dyn {
    #[inline(always)]
    fn get(self) -> usize {
        self.0
    }
}

/// The step coefficients of one head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Disc {
    pub dt: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
}

/// A, dt and the trapezoid weights of one head at one step.
///
/// `A = min(-heavy_tail(dd_a), -a_floor)`, `dt = softplus(dd_dt + bias)`,
/// `alpha = exp(A dt)`, `lambda = sigmoid(trap)`, `gamma = lambda dt`,
/// `beta = alpha dt (1 - lambda)`. The `1 - lambda` is computed as
/// `sigmoid(-trap)`, as the reference's chunked kernel does: `1 - sigmoid`
/// cancels to nothing for a large trap where `sigmoid(-trap)` stays exact.
#[inline]
pub(crate) fn discretize(dd_a: f32, dd_dt: f32, dt_bias: f32, trap: f32, a_floor: f32) -> Disc {
    let a = clamp_a(-heavy_tail(dd_a), a_floor);
    let dt = softplus(dd_dt + dt_bias);
    let alpha = fast_exp_scalar(a * dt);
    let lam = 1.0 / (1.0 + fast_exp_scalar(-trap));
    let lam_c = 1.0 / (1.0 + fast_exp_scalar(trap));
    Disc {
        dt,
        alpha,
        beta: (alpha * dt) * lam_c,
        gamma: lam * dt,
    }
}

/// Advance one head's angles by `tanh(raw) * pi * dt` and return their
/// sines and cosines. The sum runs in f64 and wraps into [0, 2*pi), as the
/// SISO path does, so a long sequence keeps full angle precision.
#[inline]
pub(crate) fn advance_angles(theta: &mut [f32], tanh_pi: &[f32], dt: f32, sc: &mut [(f32, f32)]) {
    const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
    for ((th, &tp), out) in theta.iter_mut().zip(tanh_pi).zip(sc.iter_mut()) {
        let mut acc = f64::from(*th) + f64::from(tp * dt);
        acc -= TWO_PI * (acc / TWO_PI).floor();
        *th = acc as f32;
        *out = th.sin_cos();
    }
}

/// Rotate the pairs `(a, a + half)` of one vector by the angles whose
/// `(sin, cos)` are `sc`; entries past the angles pass through.
#[inline]
pub(crate) fn rope_half_split(v: &mut [f32], sc: &[(f32, f32)]) {
    let half = v.len() / 2;
    let (lo, hi) = v.split_at_mut(half);
    for ((v0, v1), &(s, c)) in lo.iter_mut().zip(hi.iter_mut()).zip(sc) {
        let (a, b) = (*v0, *v1);
        *v0 = c * a - s * b;
        *v1 = s * a + c * b;
    }
}

/// The SSM rows of one head at one step, every row under one dispatch.
///
/// For each channel p the state row takes, per state entry n,
/// `s = alpha s + sum_r (gamma u_r[p] kc_r[n] + beta up_r[p] kp_r[n])`
/// and reads out `y_r[p] = sum_n s[n] q_r[n]` for every rank.
pub(crate) struct MimoRows<'a, K: Ranks> {
    pub rk: K,
    /// The head's state, `headdim` rows of `d_state`.
    pub state: &'a mut [f32],
    /// The previous token's keys `[R, N]`.
    pub k_prev: &'a [f32],
    /// This token's keys `[R, N]`.
    pub k_cur: &'a [f32],
    /// This token's queries `[R, N]`.
    pub q: &'a [f32],
    /// `beta * up_r[p]` laid out `[P, R]`.
    pub bv: &'a [f32],
    /// `gamma * u_r[p]` laid out `[P, R]`.
    pub gx: &'a [f32],
    pub alpha: f32,
    /// The readouts, `[P, R]`.
    pub y: &'a mut [f32],
}

impl<K: Ranks> WithSimd for MimoRows<'_, K> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let r = self.rk.get();
        let n = self.k_cur.len() / r;
        let va = simd.splat_f32s(self.alpha);
        let zero = simd.splat_f32s(0.0);
        // Each rank's vectors, split into whole SIMD vectors and a tail
        // the same way for every rank (they share the length n).
        let mut kp = [(&[][..], &[][..]); MAX_MIMO_RANK];
        let mut kc = [(&[][..], &[][..]); MAX_MIMO_RANK];
        for i in 0..r {
            let span = i * n..(i + 1) * n;
            kp[i] = S::as_simd_f32s(&self.k_prev[span.clone()]);
            kc[i] = S::as_simd_f32s(&self.k_cur[span]);
        }
        for ((row, (bvp, gxp)), yp) in self
            .state
            .chunks_exact_mut(n)
            .zip(self.bv.chunks_exact(r).zip(self.gx.chunks_exact(r)))
            .zip(self.y.chunks_exact_mut(r))
        {
            let mut vbv = [zero; MAX_MIMO_RANK];
            let mut vgx = [zero; MAX_MIMO_RANK];
            for i in 0..r {
                vbv[i] = simd.splat_f32s(bvp[i]);
                vgx[i] = simd.splat_f32s(gxp[i]);
            }
            let (s_h, s_t) = S::as_mut_simd_f32s(row);
            for (j, s) in s_h.iter_mut().enumerate() {
                let mut inner = simd.mul_f32s(vgx[0], kc[0].0[j]);
                inner = simd.mul_add_f32s(vbv[0], kp[0].0[j], inner);
                for i in 1..r {
                    inner = simd.mul_add_f32s(vgx[i], kc[i].0[j], inner);
                    inner = simd.mul_add_f32s(vbv[i], kp[i].0[j], inner);
                }
                *s = simd.mul_add_f32s(va, *s, inner);
            }
            for (j, s) in s_t.iter_mut().enumerate() {
                let mut inner = gxp[0] * kc[0].1[j];
                inner = bvp[0].mul_add(kp[0].1[j], inner);
                for i in 1..r {
                    inner = gxp[i].mul_add(kc[i].1[j], inner);
                    inner = bvp[i].mul_add(kp[i].1[j], inner);
                }
                *s = self.alpha.mul_add(*s, inner);
            }
            // Each rank's readout in the sixteen-lane order of
            // crate::ops::fold16: the same bits on every instruction set.
            for (y, q) in yp.iter_mut().zip(self.q.chunks_exact(n)) {
                *y = crate::ops::fold16::dot16(simd, row, q);
            }
        }
    }
}

/// Run one head's rows, with the rank as a constant for the released ranks.
#[inline]
pub(crate) fn mimo_rows(rank: usize, io: MimoRowsIo<'_>) {
    fn go<K: Ranks>(rk: K, io: MimoRowsIo<'_>) {
        Arch::new().dispatch(MimoRows {
            rk,
            state: io.state,
            k_prev: io.k_prev,
            k_cur: io.k_cur,
            q: io.q,
            bv: io.bv,
            gx: io.gx,
            alpha: io.alpha,
            y: io.y,
        });
    }
    match rank {
        1 => go(Const::<1>, io),
        2 => go(Const::<2>, io),
        4 => go(Const::<4>, io),
        8 => go(Const::<8>, io),
        r => go(Dyn(r), io),
    }
}

/// The operands of [`mimo_rows`], as [`MimoRows`] without the rank.
pub(crate) struct MimoRowsIo<'a> {
    pub state: &'a mut [f32],
    pub k_prev: &'a [f32],
    pub k_cur: &'a [f32],
    pub q: &'a [f32],
    pub bv: &'a [f32],
    pub gx: &'a [f32],
    pub alpha: f32,
    pub y: &'a mut [f32],
}

/// The output gate without a norm: per channel p,
/// `o[p] = sum_r mimo_o[r][p] * y_r[p] * silu(mimo_z[r][p] * z[p])`,
/// folded over r in ascending order. `y` is `[P, R]`; the projections are
/// the head's `[R, P]` rows.
#[inline]
pub(crate) fn gate_plain(o: &mut [f32], y: &[f32], z: &[f32], mimo_z: &[f32], mimo_o: &[f32]) {
    let p_len = o.len();
    let r = y.len() / p_len;
    for (p, (op, yp)) in o.iter_mut().zip(y.chunks_exact(r)).enumerate() {
        let mut acc = 0.0f32;
        for (i, &yr) in yp.iter().enumerate() {
            let v = mimo_z[i * p_len + p] * z[p];
            let s = v / (1.0 + fast_exp_scalar(-v));
            let t = (yr * s) * mimo_o[i * p_len + p];
            acc = if i == 0 { t } else { acc + t };
        }
        *op = acc;
    }
}

/// The output gate with the per-rank RMSNorm (`is_outproj_norm`): each
/// rank's readout row is normalised over the head's channels and scaled by
/// the head's norm weight before the gate and the output projection.
#[inline]
pub(crate) fn gate_normed(o: &mut [f32], y: &[f32], z: &[f32], proj: GateProj<'_>, eps: f32) {
    let p_len = o.len();
    let r = y.len() / p_len;
    let mut rstd = [0f32; MAX_MIMO_RANK];
    for (i, rs) in rstd[..r].iter_mut().enumerate() {
        let sum_sq = y
            .iter()
            .skip(i)
            .step_by(r)
            .fold(0.0f32, |acc, &v| v.mul_add(v, acc));
        *rs = 1.0 / (sum_sq / p_len as f32 + eps).sqrt();
    }
    for (p, (op, yp)) in o.iter_mut().zip(y.chunks_exact(r)).enumerate() {
        let mut acc = 0.0f32;
        for (i, &yr) in yp.iter().enumerate() {
            let v = proj.mimo_z[i * p_len + p] * z[p];
            let s = v / (1.0 + fast_exp_scalar(-v));
            let t = (((yr * rstd[i]) * proj.norm_weight[p]) * s) * proj.mimo_o[i * p_len + p];
            acc = if i == 0 { t } else { acc + t };
        }
        *op = acc;
    }
}

/// The head's gate weights for [`gate_normed`].
pub(crate) struct GateProj<'a> {
    pub mimo_z: &'a [f32],
    pub mimo_o: &'a [f32],
    pub norm_weight: &'a [f32],
}

/// One token's mixer inputs: its in_proj row split, B and C normalised,
/// the angle increments' `tanh(raw) * pi`.
pub(crate) struct TokenIn<'a> {
    /// `[d_inner]` each.
    pub z: &'a [f32],
    pub x: &'a [f32],
    /// Normalised B and C, `[R, G, N]` each.
    pub bn: &'a [f32],
    pub cn: &'a [f32],
    /// `[nheads]` each.
    pub dd_dt: &'a [f32],
    pub dd_a: &'a [f32],
    pub trap: &'a [f32],
    /// `[num_rope_angles]`.
    pub tanh_pi: &'a [f32],
}

/// Split one token's in_proj row and prepare the shared inputs: B and C
/// normalised per (rank, group) row into `bn` and `cn`, and the angle
/// increments into `tanh_pi`. Returns the token's view.
pub(crate) fn prepare_token<'a>(
    cfg: &Mamba3MimoConfig,
    lw: &Mamba3MimoLayerWeights,
    proj: &'a [f32],
    (bn, cn, tanh_pi): (&'a mut [f32], &'a mut [f32], &'a mut [f32]),
) -> TokenIn<'a> {
    let (di, nh, n, bc) = (cfg.d_inner(), cfg.nheads(), cfg.d_state, cfg.bc_width());
    let (z, rest) = proj.split_at(di);
    let (x, rest) = rest.split_at(di);
    let (b_raw, rest) = rest.split_at(bc);
    let (c_raw, rest) = rest.split_at(bc);
    let (dd_dt, rest) = rest.split_at(nh);
    let (dd_a, rest) = rest.split_at(nh);
    let (trap, angles_raw) = rest.split_at(nh);
    for (d_row, s_row) in bn.chunks_exact_mut(n).zip(b_raw.chunks_exact(n)) {
        row_norm(d_row, s_row, &lw.b_norm_weight, cfg.ssm_norm_eps);
    }
    for (d_row, s_row) in cn.chunks_exact_mut(n).zip(c_raw.chunks_exact(n)) {
        row_norm(d_row, s_row, &lw.c_norm_weight, cfg.ssm_norm_eps);
    }
    let pi = std::f32::consts::PI;
    for (tp, &raw) in tanh_pi.iter_mut().zip(angles_raw) {
        *tp = raw.tanh() * pi;
    }
    TokenIn {
        z,
        x,
        bn,
        cn,
        dd_dt,
        dd_a,
        trap,
        tanh_pi,
    }
}

/// `dst = (src * inv_rms) * w`, the row norm every MIMO path uses.
#[inline]
pub(crate) fn row_norm(dst: &mut [f32], src: &[f32], w: &[f32], eps: f32) {
    let inv_rms = 1.0 / (simd_sum_sq(src) / src.len() as f32 + eps).sqrt();
    simd_rms_scale(dst, src, w, inv_rms);
}

/// One head's recurrent state: `[P, N]` state, `[R, N]` keys, `[P]` raw
/// previous input, `[na]` angles.
pub(crate) struct HeadState<'a> {
    pub ssm: &'a mut [f32],
    pub k: &'a mut [f32],
    pub v: &'a mut [f32],
    pub angle: &'a mut [f32],
}

/// One head's working buffers.
pub(crate) struct HeadScratch {
    kc: Vec<f32>,
    q: Vec<f32>,
    u: Vec<f32>,
    bv: Vec<f32>,
    gx: Vec<f32>,
    y: Vec<f32>,
    sc: Vec<(f32, f32)>,
}

impl HeadScratch {
    pub(crate) fn new(cfg: &Mamba3MimoConfig) -> Self {
        let (r, n, p) = (cfg.mimo_rank, cfg.d_state, cfg.headdim);
        Self {
            kc: vec![0.0; r * n],
            q: vec![0.0; r * n],
            u: vec![0.0; p * r],
            bv: vec![0.0; p * r],
            gx: vec![0.0; p * r],
            y: vec![0.0; p * r],
            sc: vec![(0.0, 1.0); cfg.num_rope_angles()],
        }
    }
}

/// One token through one head: discretise, advance the angles, build and
/// rotate the R keys and queries, run the state rows, add D, gate and mix
/// the ranks into `o` (`[headdim]`), and leave this token's keys and raw
/// input in the head's state. Decode and prefill both run it, so they
/// agree bit for bit.
pub(crate) fn mimo_head_step(
    (cfg, lw, h): (&Mamba3MimoConfig, &Mamba3MimoLayerWeights, usize),
    tok: &TokenIn<'_>,
    st: HeadState<'_>,
    hs: &mut HeadScratch,
    o: &mut [f32],
) {
    let (n, p_len, g_n, r) = (cfg.d_state, cfg.headdim, cfg.ngroups, cfg.mimo_rank);
    let g = h / (cfg.nheads() / g_n);
    let disc = discretize(
        tok.dd_a[h],
        tok.dd_dt[h],
        lw.dt_bias[h],
        tok.trap[h],
        cfg.a_floor,
    );
    advance_angles(st.angle, tok.tanh_pi, disc.dt, &mut hs.sc);

    // Each rank's key and query: the normalised group row plus the head's
    // bias, then rotated.
    let bias = h * r * n..(h + 1) * r * n;
    for (i, ((kc, q), (kb, qb))) in hs
        .kc
        .chunks_exact_mut(n)
        .zip(hs.q.chunks_exact_mut(n))
        .zip(
            lw.b_bias[bias.clone()]
                .chunks_exact(n)
                .zip(lw.c_bias[bias].chunks_exact(n)),
        )
        .enumerate()
    {
        let row = (i * g_n + g) * n..(i * g_n + g + 1) * n;
        for ((k, &b), &kb) in kc.iter_mut().zip(&tok.bn[row.clone()]).zip(kb) {
            *k = b + kb;
        }
        for ((qv, &c), &qb) in q.iter_mut().zip(&tok.cn[row]).zip(qb) {
            *qv = c + qb;
        }
        rope_half_split(kc, &hs.sc);
        rope_half_split(q, &hs.sc);
    }

    // u_r = mimo_x_r * x and its previous-token twin, laid out [P, R].
    let proj_rows = h * r * p_len..(h + 1) * r * p_len;
    let mimo_x = &lw.mimo_x[proj_rows.clone()];
    let x_h = &tok.x[h * p_len..(h + 1) * p_len];
    for (p, ((u, bv), gx)) in
        hs.u.chunks_exact_mut(r)
            .zip(hs.bv.chunks_exact_mut(r))
            .zip(hs.gx.chunks_exact_mut(r))
            .enumerate()
    {
        for i in 0..r {
            let psi = mimo_x[i * p_len + p];
            u[i] = psi * x_h[p];
            gx[i] = disc.gamma * u[i];
            bv[i] = disc.beta * (psi * st.v[p]);
        }
    }

    mimo_rows(
        r,
        MimoRowsIo {
            state: st.ssm,
            k_prev: st.k,
            k_cur: &hs.kc,
            q: &hs.q,
            bv: &hs.bv,
            gx: &hs.gx,
            alpha: disc.alpha,
            y: &mut hs.y,
        },
    );
    let d = lw.d_param[h];
    for (y, &u) in hs.y.iter_mut().zip(&hs.u) {
        *y = d.mul_add(u, *y);
    }

    let z_h = &tok.z[h * p_len..(h + 1) * p_len];
    if cfg.is_outproj_norm {
        gate_normed(
            o,
            &hs.y,
            z_h,
            GateProj {
                mimo_z: &lw.mimo_z[proj_rows.clone()],
                mimo_o: &lw.mimo_o[proj_rows],
                norm_weight: &lw.norm_gate_weight[h * p_len..(h + 1) * p_len],
            },
            cfg.ssm_norm_eps,
        );
    } else {
        gate_plain(
            o,
            &hs.y,
            z_h,
            &lw.mimo_z[proj_rows.clone()],
            &lw.mimo_o[proj_rows],
        );
    }
    st.k.copy_from_slice(&hs.kc);
    st.v.copy_from_slice(x_h);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba3_siso::cpu::forward::{SsmHeadStep, simd_ssm_head};

    fn det(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0
            })
            .collect()
    }

    /// At rank 1 the MIMO rows are SISO's rows: the same state bits and
    /// readouts for every width, SIMD tail included.
    #[test]
    fn rank_one_rows_match_the_siso_rows_bitwise() {
        for n in [1usize, 3, 8, 13, 64, 128, 256] {
            for p_len in [1usize, 5, 64] {
                let state0 = det(p_len * n, 1 + n as u64);
                let (kp, kc, q) = (det(n, 2), det(n, 3), det(n, 4));
                let (x, v_prev) = (det(p_len, 5), det(p_len, 6));
                let (alpha, beta, gamma, d) = (0.93f32, 0.07f32, 0.11f32, 0.5f32);

                let mut s_siso = state0.clone();
                let mut y_siso = vec![0f32; p_len];
                simd_ssm_head(SsmHeadStep {
                    state: &mut s_siso,
                    k_prev: &kp,
                    k_cur: &kc,
                    q: &q,
                    x: &x,
                    x_stride: 1,
                    v_prev: &v_prev,
                    y: &mut y_siso,
                    y_stride: 1,
                    d,
                    alpha,
                    beta,
                    gamma,
                });

                let mut s_mimo = state0.clone();
                let bv: Vec<f32> = v_prev.iter().map(|&v| beta * v).collect();
                let gx: Vec<f32> = x.iter().map(|&v| gamma * v).collect();
                let mut y_mimo = vec![0f32; p_len];
                mimo_rows(
                    1,
                    MimoRowsIo {
                        state: &mut s_mimo,
                        k_prev: &kp,
                        k_cur: &kc,
                        q: &q,
                        bv: &bv,
                        gx: &gx,
                        alpha,
                        y: &mut y_mimo,
                    },
                );
                for (yv, &xv) in y_mimo.iter_mut().zip(&x) {
                    *yv = d.mul_add(xv, *yv);
                }
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&s_siso), bits(&s_mimo), "state n={n} p={p_len}");
                assert_eq!(bits(&y_siso), bits(&y_mimo), "y n={n} p={p_len}");
            }
        }
    }

    /// A constant rank and the same rank at runtime give the same bits.
    #[test]
    fn constant_and_runtime_ranks_agree_bitwise() {
        fn run<K: Ranks>(rk: K, n: usize, p_len: usize) -> (Vec<f32>, Vec<f32>) {
            let r = rk.get();
            let mut state = det(p_len * n, 7);
            let mut y = vec![0f32; p_len * r];
            Arch::new().dispatch(MimoRows {
                rk,
                state: &mut state,
                k_prev: &det(r * n, 8),
                k_cur: &det(r * n, 9),
                q: &det(r * n, 10),
                bv: &det(p_len * r, 11),
                gx: &det(p_len * r, 12),
                alpha: 0.9,
                y: &mut y,
            });
            (state, y)
        }
        for (n, p_len) in [(13usize, 5usize), (128, 64)] {
            assert_eq!(run(Const::<4>, n, p_len), run(Dyn(4), n, p_len));
            assert_eq!(run(Const::<2>, n, p_len), run(Dyn(2), n, p_len));
        }
    }

    /// Rotating by an angle and then by its negative restores the vector,
    /// and the entries past the angles are untouched.
    #[test]
    fn half_split_rotation_round_trips() {
        let v0 = det(16, 13);
        let mut v = v0.clone();
        let sc: Vec<(f32, f32)> = [0.3f32, -1.2, 2.0].iter().map(|a| a.sin_cos()).collect();
        rope_half_split(&mut v, &sc);
        assert_eq!(v[3..8], v0[3..8]);
        assert_eq!(v[11..], v0[11..]);
        let back: Vec<(f32, f32)> = sc.iter().map(|&(s, c)| (-s, c)).collect();
        rope_half_split(&mut v, &back);
        for (a, b) in v.iter().zip(&v0) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
