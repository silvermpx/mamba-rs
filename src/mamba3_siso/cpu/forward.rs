//! Mamba-3 SISO batched training forward pass (7 phases).
//!
//! F1: RMSNorm (loop T) — save residual, rms_val, post_norm
//! F2: in_proj batched SGEMM [T*d_model → T*in_proj_dim]
//! F3: 8-way split (z, x, B, C, dd_dt, dd_A, trap, angles)
//! F4: BCNorm + input-dependent A/DT
//! F5: Per-head trapezoidal SSM (sequential T, state carry)
//! F6: Output gating (y * SiLU(z) or RMSNormGated)
//! F7: out_proj batched SGEMM + residual
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026.

use super::dims::Mamba3Dims;
use super::flat::Mamba3LayerFlat;
use super::scratch::Mamba3Scratch;
use super::weights::TrainMamba3LayerWeights;
use crate::mlp::cpu::RowsGemm;
use crate::ops::blas::sgemm_forward;
use crate::ops::fast_math::fast_exp_scalar;
use crate::ops::rounding::store;

// Stack-array limits (must match config validation)
// Sized to the config validator's reference-range maximum; a
// 256-float stack scratch per head is ~1 KB - negligible on CPU.
const MAX_DS: usize = 256;
const MAX_ANGLES: usize = MAX_DS / 2;

// ── SIMD helpers ──

use pulp::{Arch, Simd, WithSimd};

/// sum x^2 in the sixteen-lane order of [`crate::ops::fold16`]: the same
/// bits on every instruction set.
#[inline]
pub(crate) fn simd_sum_sq(x: &[f32]) -> f32 {
    crate::ops::fold16::sum_sq_fold16_simd(x)
}

#[inline]
pub(crate) fn simd_rms_scale(out: &mut [f32], x: &[f32], weight: &[f32], inv_rms: f32) {
    struct Scale<'a> {
        out: &'a mut [f32],
        x: &'a [f32],
        w: &'a [f32],
        s: f32,
    }
    impl WithSimd for Scale<'_> {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let vs = simd.splat_f32s(self.s);
            let (o_h, o_t) = S::as_mut_simd_f32s(self.out);
            let (x_h, x_t) = S::as_simd_f32s(self.x);
            let (w_h, w_t) = S::as_simd_f32s(self.w);
            for ((o, &x), &w) in o_h.iter_mut().zip(x_h).zip(w_h) {
                *o = simd.mul_f32s(simd.mul_f32s(x, vs), w);
            }
            for ((o, &x), &w) in o_t.iter_mut().zip(x_t).zip(w_t) {
                *o = x * self.s * w;
            }
        }
    }
    Arch::new().dispatch(Scale {
        out,
        x,
        w: weight,
        s: inv_rms,
    });
}

/// One state row of the SSM recurrence under a chosen SIMD level:
/// state[n] = alpha*state[n] + beta_v*k_prev[n] + gamma_x*k_cur[n],
/// returning y = sum(state[n] * q[n]) in the sixteen-lane order of
/// [`crate::ops::fold16`], the same bits on every instruction set.
#[inline(always)]
fn ssm_row<S: Simd>(
    simd: S,
    state: &mut [f32],
    (kp, kc, q): (&[f32], &[f32], &[f32]),
    (a, bv, gx): (f32, f32, f32),
) -> f32 {
    let va = simd.splat_f32s(a);
    let vbv = simd.splat_f32s(bv);
    let vgx = simd.splat_f32s(gx);
    let (s_h, s_t) = S::as_mut_simd_f32s(state);
    let (kp_h, kp_t) = S::as_simd_f32s(kp);
    let (kc_h, kc_t) = S::as_simd_f32s(kc);
    for ((s, &kp), &kc) in s_h.iter_mut().zip(kp_h).zip(kc_h) {
        *s = simd.mul_add_f32s(va, *s, simd.mul_add_f32s(vbv, kp, simd.mul_f32s(vgx, kc)));
    }
    for ((s, &kp), &kc) in s_t.iter_mut().zip(kp_t).zip(kc_t) {
        *s = a.mul_add(*s, bv.mul_add(kp, gx * kc));
    }
    crate::ops::fold16::dot16(simd, state, q)
}

/// One head's channels at one step of the SSM recurrence: for every
/// channel p, its state row takes [`ssm_row`] with beta * v_prev[p] and
/// gamma * x[p], and y[p] = d * x[p] + the row's readout.
pub(crate) struct SsmHeadStep<'a> {
    /// The head's states, `headdim` rows of `d_state`.
    pub state: &'a mut [f32],
    pub k_prev: &'a [f32],
    pub k_cur: &'a [f32],
    pub q: &'a [f32],
    /// The step's inputs, channel p at `x[p * x_stride]`.
    pub x: &'a [f32],
    pub x_stride: usize,
    /// The previous step's inputs, one per channel.
    pub v_prev: &'a [f32],
    /// The outputs, channel p at `y[p * y_stride]`.
    pub y: &'a mut [f32],
    pub y_stride: usize,
    pub d: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
}

impl WithSimd for SsmHeadStep<'_> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let ds = self.k_cur.len();
        for (p, row) in self.state.chunks_exact_mut(ds).enumerate() {
            let x_val = self.x[p * self.x_stride];
            let y_val = ssm_row(
                simd,
                row,
                (self.k_prev, self.k_cur, self.q),
                (self.alpha, self.beta * self.v_prev[p], self.gamma * x_val),
            );
            self.y[p * self.y_stride] = self.d.mul_add(x_val, y_val);
        }
    }
}

/// The SSM recurrence of one head at one step, every channel under one
/// SIMD dispatch: a dispatch per channel spent more on choosing the
/// instruction set than on a `d_state`-wide row. The rows run the same
/// instructions in the same order either way.
#[inline]
pub(crate) fn simd_ssm_head(step: SsmHeadStep<'_>) {
    Arch::new().dispatch(step);
}

#[inline(always)]
/// Reference A-activation (state-spaces/mamba `heavy_tail_activation`):
/// `1 + x` for x >= 0, `1 / (1 - x)` for x < 0. Always positive; polynomial
/// decay for negative inputs where softplus saturates exponentially — using
/// softplus here changed the decay-spectrum distribution and gradient tails
/// relative to the published model.
pub(crate) fn heavy_tail(x: f32) -> f32 {
    if x >= 0.0 { 1.0 + x } else { 1.0 / (1.0 - x) }
}

/// A capped at `-a_floor`, the reference's `clamp(max = -a_floor)`. A NaN
/// stays NaN, as `torch.clamp` leaves it: `f32::min` would return the floor
/// and hide a corrupt input behind a finite decay. Finite values get the
/// same result as `min`.
#[inline(always)]
pub(crate) fn clamp_a(a: f32, a_floor: f32) -> f32 {
    if a > -a_floor { -a_floor } else { a }
}

/// d/dx of [`heavy_tail`]: `1` for x >= 0, `1/(1-x)^2` for x < 0.
pub(crate) fn heavy_tail_deriv(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        let d = 1.0 - x;
        1.0 / (d * d)
    }
}

pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        fast_exp_scalar(x).ln_1p()
    }
}

// ═══════════════════════════════════════════════════════════════════
// FORWARD: 7 phases
// ═══════════════════════════════════════════════════════════════════

/// Mutable per-layer recurrent state slices: SSM + K + V + RoPE angle.
///
/// Groups the four state buffers a Mamba-3 layer mutates so layer-level
/// entry points stay within the 7-argument budget.
pub struct Mamba3LayerStateMut<'a> {
    pub ssm: &'a mut [f32],
    pub k: &'a mut [f32],
    pub v: &'a mut [f32],
    pub angle: &'a mut [f32],
}
/// Mamba-3 SISO single-layer batched forward pass.
///
/// Processes `seq_len` timesteps, saves all activations in `acts` for backward.
/// Mutates `ssm_state`, `k_state`, `v_state`, `angle_state` (persistent).
///
/// In a reduced precision (the scratch's) the values between the kernels
/// are stored ones, as the GPU trainer keeps them: the normalized rows,
/// both projections' outputs, the normalized B and C, each head's B and C
/// with their bias and again after the rotation, the scan output and the
/// gated rows. TF32 rounds every GEMM operand instead. The states, the
/// head scalars and the residual stream stay f32.
pub fn forward_mamba3_layer_batched(
    temporal_flat: &mut [f32],
    acts: &mut Mamba3LayerFlat,
    layer_w: &TrainMamba3LayerWeights,
    state: Mamba3LayerStateMut<'_>,
    scratch: &mut Mamba3Scratch,
    dims: &Mamba3Dims,
) {
    let Mamba3LayerStateMut {
        ssm: ssm_state,
        k: k_state,
        v: v_state,
        angle: angle_state,
    } = state;
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let nh = dims.nheads;
    let hd = dims.headdim;
    let ng = dims.ngroups;
    let ip = dims.in_proj_dim;
    let seq_len = dims.seq_len;
    let n_angles = dims.num_rope_angles;
    let a_floor = dims.a_floor;
    let is_outproj_norm = dims.is_outproj_norm;
    let dtype = scratch.dtype;
    let lane = (sgemm_forward as RowsGemm, dtype);

    // The angles the window starts from, for the backward's rotations.
    let na_init = nh * n_angles;
    acts.angle_init[..na_init].copy_from_slice(&angle_state[..na_init]);

    let o = &acts.offsets;

    // ═══ F1: RMSNorm (loop T) ═══
    for t in 0..seq_len {
        let off = t * dm;
        let src = &temporal_flat[off..off + dm];
        let base_t = acts.base(t);

        acts.data[base_t + o.residual..base_t + o.residual + dm].copy_from_slice(src);

        let sum_sq = simd_sum_sq(src);
        let rms = (sum_sq / dm as f32 + dims.rms_norm_eps).sqrt();
        acts.data[base_t + o.rms_val] = rms;
        let inv_rms = 1.0 / rms;

        let pn = &mut scratch.post_norm_flat[off..off + dm];
        simd_rms_scale(pn, src, &layer_w.norm_weight[..dm], inv_rms);
        store(dtype, pn);
        acts.data[base_t + o.post_norm..base_t + o.post_norm + dm].copy_from_slice(pn);
    }

    // ═══ F2: Batch in_proj SGEMM ═══
    scratch.gemm.forward(
        lane,
        (&mut scratch.proj_flat, &scratch.post_norm_flat),
        (&layer_w.in_proj_w, None),
        (seq_len, dm, ip),
    );

    // ═══ F3: 8-way split (loop T) ═══
    for t in 0..seq_len {
        let proj = &scratch.proj_flat[t * ip..(t + 1) * ip];
        let base_t = acts.base(t);
        let mut off = 0;

        acts.data[base_t + o.z..base_t + o.z + di].copy_from_slice(&proj[off..off + di]);
        off += di;
        acts.data[base_t + o.x..base_t + o.x + di].copy_from_slice(&proj[off..off + di]);
        off += di;
        let bsize = ng * ds;
        acts.data[base_t + o.b_raw..base_t + o.b_raw + bsize]
            .copy_from_slice(&proj[off..off + bsize]);
        off += bsize;
        acts.data[base_t + o.c_raw..base_t + o.c_raw + bsize]
            .copy_from_slice(&proj[off..off + bsize]);
        off += bsize;
        acts.data[base_t + o.dd_dt_raw..base_t + o.dd_dt_raw + nh]
            .copy_from_slice(&proj[off..off + nh]);
        off += nh;
        acts.data[base_t + o.dd_a_raw..base_t + o.dd_a_raw + nh]
            .copy_from_slice(&proj[off..off + nh]);
        off += nh;
        acts.data[base_t + o.trap_raw..base_t + o.trap_raw + nh]
            .copy_from_slice(&proj[off..off + nh]);
        off += nh;
        if n_angles > 0 {
            acts.data[base_t + o.angles_raw..base_t + o.angles_raw + n_angles]
                .copy_from_slice(&proj[off..off + n_angles]);
        }
    }

    // ═══ F4: BCNorm + input-dependent A/DT ═══
    for t in 0..seq_len {
        let base_t = acts.base(t);
        // BCNorm for B
        let b_raw_start = base_t + o.b_raw;
        let bn_start = base_t + o.b_normed;
        for g in 0..ng {
            let gs = g * ds;
            let sum_sq = simd_sum_sq(&acts.data[b_raw_start + gs..b_raw_start + gs + ds]);
            let rms = (sum_sq / ds as f32 + dims.rms_norm_eps).sqrt();
            acts.data[base_t + o.bcnorm_rms_b + g] = rms;
            let inv_rms = 1.0 / rms;
            for i in 0..ds {
                acts.data[bn_start + gs + i] =
                    acts.data[b_raw_start + gs + i] * inv_rms * layer_w.b_norm_weight[i];
            }
            store(dtype, &mut acts.data[bn_start + gs..bn_start + gs + ds]);
        }
        // BCNorm for C
        let c_raw_start = base_t + o.c_raw;
        let cn_start = base_t + o.c_normed;
        for g in 0..ng {
            let gs = g * ds;
            let sum_sq = simd_sum_sq(&acts.data[c_raw_start + gs..c_raw_start + gs + ds]);
            let rms = (sum_sq / ds as f32 + dims.rms_norm_eps).sqrt();
            acts.data[base_t + o.bcnorm_rms_c + g] = rms;
            let inv_rms = 1.0 / rms;
            for i in 0..ds {
                acts.data[cn_start + gs + i] =
                    acts.data[c_raw_start + gs + i] * inv_rms * layer_w.c_norm_weight[i];
            }
            store(dtype, &mut acts.data[cn_start + gs..cn_start + gs + ds]);
        }
    }

    // ═══ F5: Per-head trapezoidal SSM (sequential T) ═══
    let h_state_len = nh * hd * ds;
    // Each head's B and C rows with bias, staged once per call: every head
    // writes the first d_state entries before reading them, so the rows
    // need no zeroing per head and step.
    let mut k_local = [0.0_f32; MAX_DS];
    let mut q_local = [0.0_f32; MAX_DS];

    for t in 0..seq_len {
        let base_t = acts.base(t);

        // Save h_prev
        let hp = base_t + o.h_prev;
        acts.data[hp..hp + h_state_len].copy_from_slice(&ssm_state[..h_state_len]);
        // Save k_prev, v_prev
        let kp = base_t + o.k_prev;
        acts.data[kp..kp + nh * ds].copy_from_slice(&k_state[..nh * ds]);
        let vp = base_t + o.v_prev;
        acts.data[vp..vp + nh * hd].copy_from_slice(&v_state[..nh * hd]);

        // tanh(angles_raw) * pi depends only on (t, a) — hoist out of the
        // head loop (it was recomputed nh times per timestep).
        let mut tanh_pi = [0.0_f32; MAX_ANGLES];
        if n_angles > 0 {
            let pi = std::f32::consts::PI;
            for (a, tp) in tanh_pi[..n_angles].iter_mut().enumerate() {
                *tp = acts.data[base_t + o.angles_raw + a].tanh() * pi;
            }
        }

        for h in 0..nh {
            let g = h / (nh / ng);

            // A = -heavy_tail(dd_A), clamp (reference activation)
            let a_val = clamp_a(-heavy_tail(acts.data[base_t + o.dd_a_raw + h]), a_floor);
            let dt_val = softplus(acts.data[base_t + o.dd_dt_raw + h] + layer_w.dt_bias[h]);
            acts.data[base_t + o.a_val + h] = a_val;
            acts.data[base_t + o.dt_val + h] = dt_val;

            // Per-head B/C with bias
            for n in 0..ds {
                k_local[n] =
                    acts.data[base_t + o.b_normed + g * ds + n] + layer_w.b_bias[h * ds + n];
                q_local[n] =
                    acts.data[base_t + o.c_normed + g * ds + n] + layer_w.c_bias[h * ds + n];
            }
            store(dtype, &mut k_local[..ds]);
            store(dtype, &mut q_local[..ds]);

            // RoPE
            if n_angles > 0 {
                let ab = h * n_angles;
                for a in 0..n_angles {
                    let delta = tanh_pi[a] * dt_val;
                    let mut acc = angle_state[ab + a] as f64 + delta as f64;
                    let two_pi_64 = 2.0 * std::f64::consts::PI;
                    acc -= two_pi_64 * (acc / two_pi_64).floor();
                    angle_state[ab + a] = acc as f32;
                }
                for a in 0..n_angles {
                    let (sin_a, cos_a) = angle_state[h * n_angles + a].sin_cos();
                    let i0 = 2 * a;
                    let i1 = 2 * a + 1;
                    let b0 = k_local[i0];
                    let b1 = k_local[i1];
                    k_local[i0] = cos_a * b0 - sin_a * b1;
                    k_local[i1] = sin_a * b0 + cos_a * b1;
                    let c0 = q_local[i0];
                    let c1 = q_local[i1];
                    q_local[i0] = cos_a * c0 - sin_a * c1;
                    q_local[i1] = sin_a * c0 + cos_a * c1;
                }
                store(dtype, &mut k_local[..2 * n_angles]);
                store(dtype, &mut q_local[..2 * n_angles]);
            }

            // Save per-step angle velocity tanh(raw)*pi (backward reads it
            // to rebuild cumulative angles without recomputing tanh).
            if h == nh - 1 && n_angles > 0 {
                acts.data[base_t + o.angle_cumsum..base_t + o.angle_cumsum + n_angles]
                    .copy_from_slice(&tanh_pi[..n_angles]);
            }

            // Alpha, beta, gamma
            let alpha = fast_exp_scalar(a_val * dt_val);
            let trap_sig = 1.0 / (1.0 + fast_exp_scalar(-acts.data[base_t + o.trap_raw + h]));
            let beta = alpha * dt_val * (1.0 - trap_sig);
            let gamma = trap_sig * dt_val;
            acts.data[base_t + o.alpha + h] = alpha;
            acts.data[base_t + o.beta + h] = beta;
            acts.data[base_t + o.gamma + h] = gamma;

            // SSM recurrence; x sits before y in a step's fields.
            let (lo, hi) = acts.data.split_at_mut(base_t + o.y);
            let x_head = base_t + o.x + h * hd;
            simd_ssm_head(SsmHeadStep {
                state: &mut ssm_state[h * hd * ds..(h + 1) * hd * ds],
                k_prev: &k_state[h * ds..h * ds + ds],
                k_cur: &k_local[..ds],
                q: &q_local[..ds],
                x: &lo[x_head..x_head + hd],
                x_stride: 1,
                v_prev: &v_state[h * hd..(h + 1) * hd],
                y: &mut hi[h * hd..(h + 1) * hd],
                y_stride: 1,
                d: layer_w.d_param[h],
                alpha,
                beta,
                gamma,
            });
            store(dtype, &mut hi[h * hd..(h + 1) * hd]);

            // Update states
            k_state[h * ds..h * ds + ds].copy_from_slice(&k_local[..ds]);
        }
        // Each head read only its own previous inputs above, so the step's
        // inputs become the next step's in one copy.
        let xs = base_t + o.x;
        v_state[..nh * hd].copy_from_slice(&acts.data[xs..xs + nh * hd]);

        // Save h_curr
        let hc = base_t + o.h_curr;
        acts.data[hc..hc + h_state_len].copy_from_slice(&ssm_state[..h_state_len]);
    }

    // ═══ F6: Output gating (loop T) ═══
    for t in 0..seq_len {
        let base_t = acts.base(t);
        let ys = base_t + o.y;
        let zs = base_t + o.z;
        let gs = base_t + o.gated;

        if is_outproj_norm {
            for g_start in (0..di).step_by(hd) {
                let g_end = (g_start + hd).min(di);
                let g_len = g_end - g_start;
                let sum_sq = simd_sum_sq(&acts.data[ys + g_start..ys + g_end]);
                let rstd = 1.0 / (sum_sq / g_len as f32 + dims.rms_norm_eps).sqrt();
                for d in g_start..g_end {
                    let z = acts.data[zs + d];
                    let silu = z / (1.0 + fast_exp_scalar(-z));
                    acts.data[gs + d] =
                        acts.data[ys + d] * rstd * layer_w.norm_gate_weight[d] * silu;
                }
            }
        } else {
            for d in 0..di {
                let z = acts.data[zs + d];
                let silu = z / (1.0 + fast_exp_scalar(-z));
                acts.data[gs + d] = acts.data[ys + d] * silu;
            }
        }
        store(dtype, &mut acts.data[gs..gs + di]);

        scratch.gated_flat[t * di..(t + 1) * di].copy_from_slice(&acts.data[gs..gs + di]);
    }

    // ═══ F7: Batch out_proj SGEMM + residual ═══
    scratch.gemm.forward(
        lane,
        (&mut scratch.out_flat, &scratch.gated_flat),
        (&layer_w.out_proj_w, None),
        (seq_len, di, dm),
    );

    for t in 0..seq_len {
        let off = t * dm;
        let base_t = acts.base(t);
        for d in 0..dm {
            temporal_flat[off + d] = acts.data[base_t + o.residual + d] + scratch.out_flat[off + d];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba3_siso::config::Mamba3Config;

    fn test_cfg() -> Mamba3Config {
        Mamba3Config {
            d_model: 16,
            d_state: 8,
            expand: 2,
            headdim: 4,
            ngroups: 1,
            n_layers: 1,
            rope_fraction: 0.5,
            a_floor: 0.0625,
            is_outproj_norm: false,
            ..Mamba3Config::default()
        }
    }

    #[test]
    fn test_forward_no_panic() {
        let cfg = test_cfg();
        let dims = Mamba3Dims::from_config(&cfg, 4);
        let w = TrainMamba3LayerWeights::zeros(&dims);
        // Init some weights so output isn't all zeros
        let mut w = w;
        w.norm_weight.fill(1.0);
        w.d_param.fill(1.0);

        let mut acts = Mamba3LayerFlat::zeros(dims);
        let mut scratch = Mamba3Scratch::zeros(&dims);
        let mut ssm = vec![0.0; dims.nheads * dims.headdim * dims.d_state];
        let mut k_st = vec![0.0; dims.nheads * dims.d_state];
        let mut v_st = vec![0.0; dims.nheads * dims.headdim];
        let mut a_st = vec![0.0; dims.nheads * dims.num_rope_angles.max(1)];
        let mut temporal = vec![1.0_f32; dims.seq_len * dims.d_model];

        forward_mamba3_layer_batched(
            &mut temporal,
            &mut acts,
            &w,
            Mamba3LayerStateMut {
                ssm: &mut ssm,
                k: &mut k_st,
                v: &mut v_st,
                angle: &mut a_st,
            },
            &mut scratch,
            &dims,
        );

        assert!(
            temporal.iter().all(|v| v.is_finite()),
            "output must be finite"
        );
    }
}
