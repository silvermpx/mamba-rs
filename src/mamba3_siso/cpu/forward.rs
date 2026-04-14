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
use crate::ops::blas::sgemm_forward;
use crate::ops::fast_math::{RMS_NORM_EPS, fast_exp_scalar};

// Stack-array limits (must match config validation)
const MAX_DS: usize = 64;
const MAX_ANGLES: usize = MAX_DS / 2;

// ── SIMD helpers ──

use pulp::{Arch, Simd, WithSimd};

#[inline]
pub(crate) fn simd_sum_sq(x: &[f32]) -> f32 {
    struct SumSq<'a>(&'a [f32]);
    impl WithSimd for SumSq<'_> {
        type Output = f32;
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> f32 {
            let (head, tail) = S::as_simd_f32s(self.0);
            let mut acc = simd.splat_f32s(0.0);
            for &v in head {
                acc = simd.mul_add_f32s(v, v, acc);
            }
            let mut s = simd.reduce_sum_f32s(acc);
            for &v in tail {
                s += v * v;
            }
            s
        }
    }
    Arch::new().dispatch(SumSq(x))
}

#[inline]
fn simd_rms_scale(out: &mut [f32], x: &[f32], weight: &[f32], inv_rms: f32) {
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

/// SIMD SSM recurrence: state[n] = alpha*state[n] + beta_v*k_prev[n] + gamma_x*k_cur[n]
/// Returns y = sum(state[n] * q[n]).
#[inline]
pub(crate) fn simd_ssm_recurrence(
    state: &mut [f32],
    k_prev: &[f32],
    k_cur: &[f32],
    q: &[f32],
    alpha: f32,
    beta_v: f32,
    gamma_x: f32,
) -> f32 {
    struct Rec<'a> {
        state: &'a mut [f32],
        kp: &'a [f32],
        kc: &'a [f32],
        q: &'a [f32],
        a: f32,
        bv: f32,
        gx: f32,
    }
    impl WithSimd for Rec<'_> {
        type Output = f32;
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> f32 {
            let va = simd.splat_f32s(self.a);
            let vbv = simd.splat_f32s(self.bv);
            let vgx = simd.splat_f32s(self.gx);
            let (s_h, s_t) = S::as_mut_simd_f32s(self.state);
            let (kp_h, kp_t) = S::as_simd_f32s(self.kp);
            let (kc_h, kc_t) = S::as_simd_f32s(self.kc);
            let (q_h, q_t) = S::as_simd_f32s(self.q);
            let mut yacc = simd.splat_f32s(0.0);
            for (((s, &kp), &kc), &q) in s_h.iter_mut().zip(kp_h).zip(kc_h).zip(q_h) {
                let ns =
                    simd.mul_add_f32s(va, *s, simd.mul_add_f32s(vbv, kp, simd.mul_f32s(vgx, kc)));
                *s = ns;
                yacc = simd.mul_add_f32s(ns, q, yacc);
            }
            let mut y = simd.reduce_sum_f32s(yacc);
            for (((s, &kp), &kc), &q) in s_t.iter_mut().zip(kp_t).zip(kc_t).zip(q_t) {
                *s = self.a.mul_add(*s, self.bv.mul_add(kp, self.gx * kc));
                y = (*s).mul_add(q, y);
            }
            y
        }
    }
    Arch::new().dispatch(Rec {
        state,
        kp: k_prev,
        kc: k_cur,
        q,
        a: alpha,
        bv: beta_v,
        gx: gamma_x,
    })
}

#[inline(always)]
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0_f32 + fast_exp_scalar(x)).ln()
    }
}

// ═══════════════════════════════════════════════════════════════════
// FORWARD: 7 phases
// ═══════════════════════════════════════════════════════════════════

/// Mamba-3 SISO single-layer batched forward pass.
///
/// Processes `seq_len` timesteps, saves all activations in `acts` for backward.
/// Mutates `ssm_state`, `k_state`, `v_state`, `angle_state` (persistent).
pub fn forward_mamba3_layer_batched(
    temporal_flat: &mut [f32],
    acts: &mut Mamba3LayerFlat,
    layer_w: &TrainMamba3LayerWeights,
    ssm_state: &mut [f32],
    k_state: &mut [f32],
    v_state: &mut [f32],
    angle_state: &mut [f32],
    scratch: &mut Mamba3Scratch,
    dims: &Mamba3Dims,
) {
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

    let o = &acts.offsets;

    // ═══ F1: RMSNorm (loop T) ═══
    for t in 0..seq_len {
        let off = t * dm;
        let src = &temporal_flat[off..off + dm];
        let base_t = acts.base(t);

        acts.data[base_t + o.residual..base_t + o.residual + dm].copy_from_slice(src);

        let sum_sq = simd_sum_sq(src);
        let rms = (sum_sq / dm as f32 + RMS_NORM_EPS).sqrt();
        acts.data[base_t + o.rms_val] = rms;
        let inv_rms = 1.0 / rms;

        let pn = &mut scratch.post_norm_flat[off..off + dm];
        simd_rms_scale(pn, src, &layer_w.norm_weight[..dm], inv_rms);
        acts.data[base_t + o.post_norm..base_t + o.post_norm + dm].copy_from_slice(pn);
    }

    // ═══ F2: Batch in_proj SGEMM ═══
    sgemm_forward(
        &mut scratch.proj_flat,
        &scratch.post_norm_flat,
        &layer_w.in_proj_w,
        None,
        seq_len,
        dm,
        ip,
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
            let rms = (sum_sq / ds as f32 + RMS_NORM_EPS).sqrt();
            acts.data[base_t + o.bcnorm_rms_b + g] = rms;
            let inv_rms = 1.0 / rms;
            for i in 0..ds {
                acts.data[bn_start + gs + i] =
                    acts.data[b_raw_start + gs + i] * inv_rms * layer_w.b_norm_weight[i];
            }
        }
        // BCNorm for C
        let c_raw_start = base_t + o.c_raw;
        let cn_start = base_t + o.c_normed;
        for g in 0..ng {
            let gs = g * ds;
            let sum_sq = simd_sum_sq(&acts.data[c_raw_start + gs..c_raw_start + gs + ds]);
            let rms = (sum_sq / ds as f32 + RMS_NORM_EPS).sqrt();
            acts.data[base_t + o.bcnorm_rms_c + g] = rms;
            let inv_rms = 1.0 / rms;
            for i in 0..ds {
                acts.data[cn_start + gs + i] =
                    acts.data[c_raw_start + gs + i] * inv_rms * layer_w.c_norm_weight[i];
            }
        }
    }

    // ═══ F5: Per-head trapezoidal SSM (sequential T) ═══
    let h_state_len = nh * hd * ds;

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

        for h in 0..nh {
            let g = h / (nh / ng);

            // A = -softplus(dd_A), clamp
            let a_val = (-softplus(acts.data[base_t + o.dd_a_raw + h])).min(-a_floor);
            let dt_val = softplus(acts.data[base_t + o.dd_dt_raw + h] + layer_w.dt_bias[h]);
            acts.data[base_t + o.a_val + h] = a_val;
            acts.data[base_t + o.dt_val + h] = dt_val;

            // Per-head B/C with bias
            let mut k_local = [0.0_f32; MAX_DS];
            let mut q_local = [0.0_f32; MAX_DS];
            for n in 0..ds {
                k_local[n] =
                    acts.data[base_t + o.b_normed + g * ds + n] + layer_w.b_bias[h * ds + n];
                q_local[n] =
                    acts.data[base_t + o.c_normed + g * ds + n] + layer_w.c_bias[h * ds + n];
            }

            // RoPE
            if n_angles > 0 {
                let ab = h * n_angles;
                let pi = std::f32::consts::PI;
                for a in 0..n_angles {
                    let raw = acts.data[base_t + o.angles_raw + a];
                    let delta = raw.tanh() * pi * dt_val;
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
            }

            // Save angle_cumsum
            if h == nh - 1 && n_angles > 0 {
                let pi = std::f32::consts::PI;
                for a in 0..n_angles {
                    acts.data[base_t + o.angle_cumsum + a] =
                        acts.data[base_t + o.angles_raw + a].tanh() * pi;
                }
            }

            // Alpha, beta, gamma
            let alpha = fast_exp_scalar(a_val * dt_val);
            let trap_sig = 1.0 / (1.0 + fast_exp_scalar(-acts.data[base_t + o.trap_raw + h]));
            let beta = alpha * dt_val * (1.0 - trap_sig);
            let gamma = trap_sig * dt_val;
            acts.data[base_t + o.alpha + h] = alpha;
            acts.data[base_t + o.beta + h] = beta;
            acts.data[base_t + o.gamma + h] = gamma;

            // SSM recurrence
            let kp_slice = &k_state[h * ds..h * ds + ds];
            for p in 0..hd {
                let x_val = acts.data[base_t + o.x + h * hd + p];
                let v_prev = v_state[h * hd + p];
                let s_off = (h * hd + p) * ds;
                let y_val = simd_ssm_recurrence(
                    &mut ssm_state[s_off..s_off + ds],
                    kp_slice,
                    &k_local[..ds],
                    &q_local[..ds],
                    alpha,
                    beta * v_prev,
                    gamma * x_val,
                );
                acts.data[base_t + o.y + h * hd + p] = layer_w.d_param[h].mul_add(x_val, y_val);
            }

            // Update states
            k_state[h * ds..h * ds + ds].copy_from_slice(&k_local[..ds]);
            for p in 0..hd {
                v_state[h * hd + p] = acts.data[base_t + o.x + h * hd + p];
            }
        }

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
                let rstd = 1.0 / (sum_sq / g_len as f32 + RMS_NORM_EPS).sqrt();
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

        scratch.gated_flat[t * di..(t + 1) * di].copy_from_slice(&acts.data[gs..gs + di]);
    }

    // ═══ F7: Batch out_proj SGEMM + residual ═══
    sgemm_forward(
        &mut scratch.out_flat,
        &scratch.gated_flat,
        &layer_w.out_proj_w,
        None,
        seq_len,
        di,
        dm,
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
        }
    }

    #[test]
    fn test_forward_no_panic() {
        let cfg = test_cfg();
        let dims = Mamba3Dims::from_config(&cfg, 4);
        let w = TrainMamba3LayerWeights::zeros(&dims);
        // Init some weights so output isn't all zeros
        let mut w = w;
        for v in &mut w.norm_weight {
            *v = 1.0;
        }
        for v in &mut w.d_param {
            *v = 1.0;
        }

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
            &mut ssm,
            &mut k_st,
            &mut v_st,
            &mut a_st,
            &mut scratch,
            &dims,
        );

        assert!(
            temporal.iter().all(|v| v.is_finite()),
            "output must be finite"
        );
    }
}
