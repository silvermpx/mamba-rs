//! Mamba-3 SISO batched training backward pass (8 phases).
//!
//! B1: out_proj SGEMM backward
//! B2: Output gating backward
//! B3+B4+B5 (fused): Trapezoidal BPTT + RoPE backward + BCNorm bias accumulation
//! B5 (continued): BCNorm backward (per-group RMSNorm backward)
//! B6: Discretization backward (alpha/beta/gamma → dd_A, dd_dt, trap, angles)
//! B7: in_proj SGEMM backward
//! B8: RMSNorm backward + residual
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026.

use super::dims::Mamba3Dims;
use super::flat::Mamba3LayerFlat;
use super::forward::{simd_sum_sq, softplus};
use super::scratch::Mamba3Scratch;
use super::weights::TrainMamba3LayerWeights;
use crate::ops::blas::sgemm_backward;
use crate::ops::fast_math::{RMS_NORM_EPS, fast_exp_scalar};

const MAX_DS: usize = 64;
// MAX_NH_DS removed — d_k_carry now uses Vec to support all valid configs
const MAX_ANGLES: usize = MAX_DS / 2;

/// Mamba-3 SISO single-layer batched backward pass.
///
/// Computes gradients for all layer weights into `d_layer` and
/// propagates `d_temporal_flat` backward through the layer.
/// `angle_state_init`: snapshot of angle_state BEFORE the forward pass of this window.
/// Required for correct RoPE gradient when angle_state carries from a prior window (burn-in).
/// Pass `None` or a zero slice if training starts from zero state (no burn-in).
pub fn backward_mamba3_layer_batched(
    d_temporal_flat: &mut [f32],
    acts: &Mamba3LayerFlat,
    layer_w: &TrainMamba3LayerWeights,
    d_layer: &mut TrainMamba3LayerWeights,
    scratch: &mut Mamba3Scratch,
    dims: &Mamba3Dims,
    angle_state_init: Option<&[f32]>,
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
    let is_outproj_norm = dims.is_outproj_norm;

    let o = &acts.offsets;
    let bsize = ng * ds;

    // ═══ B1: out_proj SGEMM backward ═══
    for t in 0..seq_len {
        let base_t = acts.base(t);
        let gs = base_t + o.gated;
        scratch.gated_buf[t * di..(t + 1) * di].copy_from_slice(&acts.data[gs..gs + di]);
    }

    sgemm_backward(
        &mut scratch.d_gated_flat,
        &mut d_layer.out_proj_w,
        None,
        d_temporal_flat,
        &scratch.gated_buf,
        &layer_w.out_proj_w,
        (seq_len, di, dm),
    );

    // ═══ B2: Output gating backward ═══
    for t in 0..seq_len {
        let base_t = acts.base(t);
        let ys = base_t + o.y;
        let zs = base_t + o.z;

        let dg = &scratch.d_gated_flat[t * di..(t + 1) * di];
        let dy = &mut scratch.d_y_flat[t * di..(t + 1) * di];
        let dz = &mut scratch.d_z_flat[t * di..(t + 1) * di];

        if is_outproj_norm {
            for g_start in (0..di).step_by(hd) {
                let g_end = (g_start + hd).min(di);
                let g_len = g_end - g_start;
                let sum_sq = simd_sum_sq(&acts.data[ys + g_start..ys + g_end]);
                let rstd = 1.0 / (sum_sq / g_len as f32 + RMS_NORM_EPS).sqrt();

                for d in g_start..g_end {
                    let z = acts.data[zs + d];
                    let sig = 1.0 / (1.0 + fast_exp_scalar(-z));
                    let silu = z * sig;
                    let y_hat = acts.data[ys + d] * rstd;
                    let y_normed = y_hat * layer_w.norm_gate_weight[d];

                    dz[d] = dg[d] * y_normed * (sig + z * sig * (1.0 - sig));
                    dy[d] = dg[d] * silu;
                    d_layer.norm_gate_weight[d] += dg[d] * silu * y_hat;
                }

                let mut c1 = 0.0_f32;
                for d in g_start..g_end {
                    let y_hat = acts.data[ys + d] * rstd;
                    c1 += y_hat * layer_w.norm_gate_weight[d] * dy[d];
                }
                c1 /= g_len as f32;

                for d in g_start..g_end {
                    let y_hat = acts.data[ys + d] * rstd;
                    let wdy = layer_w.norm_gate_weight[d] * dy[d];
                    dy[d] = (wdy - y_hat * c1) * rstd;
                }
            }
        } else {
            for d in 0..di {
                let z = acts.data[zs + d];
                let sig = 1.0 / (1.0 + fast_exp_scalar(-z));
                let silu = z * sig;
                let d_silu = sig + z * sig * (1.0 - sig);
                dy[d] = dg[d] * silu;
                dz[d] = dg[d] * acts.data[ys + d] * d_silu;
            }
        }
    }

    let mut d_d_param = vec![0.0_f32; nh];

    // Pre-compute cumulative angles for backward RoPE reconstruction.
    // If angle_state_init is provided (burn-in case), initialize running from it
    // so that cumulative angles match what forward actually used.
    let cum_angles = if n_angles > 0 {
        let mut ca = vec![0.0_f32; seq_len * nh * n_angles];
        let pi = std::f32::consts::PI;
        let two_pi = 2.0 * pi;
        for h in 0..nh {
            let mut running = [0.0_f32; MAX_ANGLES];
            if let Some(init) = angle_state_init {
                let base = h * n_angles;
                running[..n_angles].copy_from_slice(&init[base..base + n_angles]);
            }
            for t in 0..seq_len {
                let base_s = acts.base(t);
                let dt_h = acts.data[base_s + o.dt_val + h];
                for (a, r) in running[..n_angles].iter_mut().enumerate() {
                    let raw = acts.data[base_s + o.angles_raw + a];
                    *r += raw.tanh() * pi * dt_h;
                    *r -= two_pi * (*r / two_pi).floor();
                }
                let off = t * nh * n_angles + h * n_angles;
                ca[off..off + n_angles].copy_from_slice(&running[..n_angles]);
            }
        }
        ca
    } else {
        Vec::new()
    };

    // ═══ B3+B4+B5 (fused): BPTT + RoPE backward + bias accumulation ═══
    scratch.d_h.fill(0.0);
    scratch.d_alpha_flat.fill(0.0);
    scratch.d_beta_flat.fill(0.0);
    scratch.d_gamma_flat.fill(0.0);
    scratch.d_x_flat.fill(0.0);
    scratch.d_angle_cumsum_flat.fill(0.0);
    scratch.d_b_pre_rope_flat.fill(0.0);
    scratch.d_c_pre_rope_flat.fill(0.0);
    d_d_param.fill(0.0);

    let mut d_k_carry = vec![0.0_f32; nh * ds];
    let mut d_k_carry_next = vec![0.0_f32; nh * ds];

    for t in (0..seq_len).rev() {
        let base_t = acts.base(t);
        let d_y = &scratch.d_y_flat[t * di..(t + 1) * di];

        d_k_carry_next[..nh * ds].fill(0.0);

        for h in 0..nh {
            let g = h / (nh / ng);
            let alpha_h = acts.data[base_t + o.alpha + h];
            let beta_h = acts.data[base_t + o.beta + h];
            let gamma_h = acts.data[base_t + o.gamma + h];

            // Reconstruct k_local, q_local (post-norm + bias + RoPE)
            let mut k_pre = [0.0_f32; MAX_DS];
            let mut q_pre = [0.0_f32; MAX_DS];
            let mut k_local = [0.0_f32; MAX_DS];
            let mut q_local = [0.0_f32; MAX_DS];
            for n in 0..ds {
                k_pre[n] = acts.data[base_t + o.b_normed + g * ds + n] + layer_w.b_bias[h * ds + n];
                q_pre[n] = acts.data[base_t + o.c_normed + g * ds + n] + layer_w.c_bias[h * ds + n];
                k_local[n] = k_pre[n];
                q_local[n] = q_pre[n];
            }

            let mut cum_angle_h = [0.0_f32; MAX_ANGLES];
            if n_angles > 0 {
                let ca_off = t * nh * n_angles + h * n_angles;
                cum_angle_h[..n_angles].copy_from_slice(&cum_angles[ca_off..ca_off + n_angles]);
                for a in 0..n_angles {
                    let (sin_a, cos_a) = cum_angle_h[a].sin_cos();
                    let (i0, i1) = (2 * a, 2 * a + 1);
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

            let mut d_k_h = [0.0_f32; MAX_DS];
            d_k_h[..ds].copy_from_slice(&d_k_carry[h * ds..h * ds + ds]);
            let mut d_c_h = [0.0_f32; MAX_DS];

            for p in 0..hd {
                let x_val = acts.data[base_t + o.x + h * hd + p];
                let dy_val = d_y[h * hd + p];
                let v_prev = acts.data[base_t + o.v_prev + h * hd + p];

                d_d_param[h] += dy_val * x_val;

                for n in 0..ds {
                    let idx = (h * hd + p) * ds + n;
                    scratch.d_h[idx] += dy_val * q_local[n];
                    d_c_h[n] += dy_val * acts.data[base_t + o.h_curr + idx];
                }

                let mut d_x_val = layer_w.d_param[h] * dy_val;

                for n in 0..ds {
                    let idx = (h * hd + p) * ds + n;
                    let dh = scratch.d_h[idx];
                    let h_prev_val = acts.data[base_t + o.h_prev + idx];
                    let k_prev_n = acts.data[base_t + o.k_prev + h * ds + n];

                    scratch.d_alpha_flat[t * nh + h] += dh * h_prev_val;
                    scratch.d_beta_flat[t * nh + h] += dh * v_prev * k_prev_n;
                    scratch.d_gamma_flat[t * nh + h] += dh * x_val * k_local[n];

                    if t > 0 {
                        scratch.d_x_flat[(t - 1) * di + h * hd + p] += dh * beta_h * k_prev_n;
                    }
                    d_k_carry_next[h * ds + n] += dh * beta_h * v_prev;
                    d_x_val += dh * gamma_h * k_local[n];
                    d_k_h[n] += dh * gamma_h * x_val;

                    scratch.d_h[idx] = alpha_h * dh;
                }

                scratch.d_x_flat[t * di + h * hd + p] += d_x_val;
            }

            // B4: Inverse RoPE
            let mut d_k_pre_rope = d_k_h;
            let mut d_c_pre_rope = d_c_h;

            if n_angles > 0 {
                for a in 0..n_angles {
                    let (sin_a, cos_a) = cum_angle_h[a].sin_cos();
                    let (i0, i1) = (2 * a, 2 * a + 1);
                    d_k_pre_rope[i0] = cos_a * d_k_h[i0] + sin_a * d_k_h[i1];
                    d_k_pre_rope[i1] = -sin_a * d_k_h[i0] + cos_a * d_k_h[i1];
                    d_c_pre_rope[i0] = cos_a * d_c_h[i0] + sin_a * d_c_h[i1];
                    d_c_pre_rope[i1] = -sin_a * d_c_h[i0] + cos_a * d_c_h[i1];

                    let d_angle_b = d_k_h[i0] * (-sin_a * k_pre[i0] - cos_a * k_pre[i1])
                        + d_k_h[i1] * (cos_a * k_pre[i0] - sin_a * k_pre[i1]);
                    let d_angle_c = d_c_h[i0] * (-sin_a * q_pre[i0] - cos_a * q_pre[i1])
                        + d_c_h[i1] * (cos_a * q_pre[i0] - sin_a * q_pre[i1]);

                    scratch.d_angle_cumsum_flat[t * nh * n_angles + h * n_angles + a] +=
                        d_angle_b + d_angle_c;
                }
            }

            // B5: Accumulate bias gradients
            for n in 0..ds {
                d_layer.b_bias[h * ds + n] += d_k_pre_rope[n];
                d_layer.c_bias[h * ds + n] += d_c_pre_rope[n];
                scratch.d_b_pre_rope_flat[t * bsize + g * ds + n] += d_k_pre_rope[n];
                scratch.d_c_pre_rope_flat[t * bsize + g * ds + n] += d_c_pre_rope[n];
            }
        }

        d_k_carry[..nh * ds].copy_from_slice(&d_k_carry_next[..nh * ds]);
    }

    for (dl, &dd) in d_layer.d_param[..nh].iter_mut().zip(&d_d_param[..nh]) {
        *dl += dd;
    }

    // ═══ B5 (continued): BCNorm backward ═══
    scratch.d_b_raw_flat.fill(0.0);
    scratch.d_c_raw_flat.fill(0.0);

    for t in 0..seq_len {
        let base_t = acts.base(t);

        // B backward
        let d_b_normed = &scratch.d_b_pre_rope_flat[t * bsize..(t + 1) * bsize];
        let d_b_raw = &mut scratch.d_b_raw_flat[t * bsize..(t + 1) * bsize];
        for g in 0..ng {
            let gs = g * ds;
            let rms_b = acts.data[base_t + o.bcnorm_rms_b + g];
            let inv_rms = 1.0 / rms_b.max(1e-12);
            let mut c1 = 0.0_f32;
            for i in 0..ds {
                let b_hat = acts.data[base_t + o.b_raw + gs + i] * inv_rms;
                c1 += b_hat * layer_w.b_norm_weight[i] * d_b_normed[gs + i];
            }
            c1 /= ds as f32;
            for i in 0..ds {
                let b_hat = acts.data[base_t + o.b_raw + gs + i] * inv_rms;
                d_b_raw[gs + i] =
                    (layer_w.b_norm_weight[i] * d_b_normed[gs + i] - b_hat * c1) * inv_rms;
                d_layer.b_norm_weight[i] += d_b_normed[gs + i] * b_hat;
            }
        }

        // C backward
        let d_c_normed = &scratch.d_c_pre_rope_flat[t * bsize..(t + 1) * bsize];
        let d_c_raw = &mut scratch.d_c_raw_flat[t * bsize..(t + 1) * bsize];
        for g in 0..ng {
            let gs = g * ds;
            let rms_c = acts.data[base_t + o.bcnorm_rms_c + g];
            let inv_rms = 1.0 / rms_c.max(1e-12);
            let mut c1 = 0.0_f32;
            for i in 0..ds {
                let c_hat = acts.data[base_t + o.c_raw + gs + i] * inv_rms;
                c1 += c_hat * layer_w.c_norm_weight[i] * d_c_normed[gs + i];
            }
            c1 /= ds as f32;
            for i in 0..ds {
                let c_hat = acts.data[base_t + o.c_raw + gs + i] * inv_rms;
                d_c_raw[gs + i] =
                    (layer_w.c_norm_weight[i] * d_c_normed[gs + i] - c_hat * c1) * inv_rms;
                d_layer.c_norm_weight[i] += d_c_normed[gs + i] * c_hat;
            }
        }
    }

    // ═══ B6: Discretization backward ═══
    scratch.d_dd_dt_flat.fill(0.0);
    scratch.d_dd_a_flat.fill(0.0);
    scratch.d_trap_flat.fill(0.0);
    scratch.d_angles_flat.fill(0.0);

    // Reverse cumsum of d_angle_cumsum
    if n_angles > 0 {
        for h in 0..nh {
            for a in 0..n_angles {
                for t in (0..seq_len.saturating_sub(1)).rev() {
                    let cur = t * nh * n_angles + h * n_angles + a;
                    let nxt = (t + 1) * nh * n_angles + h * n_angles + a;
                    scratch.d_angle_cumsum_flat[cur] += scratch.d_angle_cumsum_flat[nxt];
                }
            }
        }
    }

    for t in 0..seq_len {
        let base_t = acts.base(t);
        let d_alpha = &scratch.d_alpha_flat[t * nh..(t + 1) * nh];
        let d_beta = &scratch.d_beta_flat[t * nh..(t + 1) * nh];
        let d_gamma = &scratch.d_gamma_flat[t * nh..(t + 1) * nh];

        for h in 0..nh {
            let a_val = acts.data[base_t + o.a_val + h];
            let dt_val = acts.data[base_t + o.dt_val + h];
            let alpha_h = acts.data[base_t + o.alpha + h];
            let trap_raw = acts.data[base_t + o.trap_raw + h];
            let trap_sig = 1.0 / (1.0 + fast_exp_scalar(-trap_raw));

            let d_adt = d_alpha[h] * alpha_h + d_beta[h] * alpha_h * dt_val * (1.0 - trap_sig);

            let d_dt_from_adt = d_adt * a_val;
            let d_dt_from_beta = d_beta[h] * alpha_h * (1.0 - trap_sig);
            let d_dt_from_gamma = d_gamma[h] * trap_sig;

            let mut d_dt_from_angles = 0.0_f32;
            if n_angles > 0 {
                let pi = std::f32::consts::PI;
                for a in 0..n_angles {
                    let raw = acts.data[base_t + o.angles_raw + a];
                    let tanh_raw = raw.tanh();
                    let d_delta = scratch.d_angle_cumsum_flat[t * nh * n_angles + h * n_angles + a];
                    d_dt_from_angles += d_delta * tanh_raw * pi;
                    scratch.d_angles_flat[t * n_angles + a] +=
                        d_delta * pi * dt_val * (1.0 - tanh_raw * tanh_raw);
                }
            }

            let d_dt_total = d_dt_from_adt + d_dt_from_beta + d_dt_from_gamma + d_dt_from_angles;

            let dt_pre = acts.data[base_t + o.dd_dt_raw + h] + layer_w.dt_bias[h];
            let sp_deriv_dt = if dt_pre > 20.0 {
                1.0
            } else {
                1.0 / (1.0 + fast_exp_scalar(-dt_pre))
            };
            scratch.d_dd_dt_flat[t * nh + h] = d_dt_total * sp_deriv_dt;
            d_layer.dt_bias[h] += d_dt_total * sp_deriv_dt;

            let d_trap_sig = -d_beta[h] * alpha_h * dt_val + d_gamma[h] * dt_val;
            scratch.d_trap_flat[t * nh + h] = d_trap_sig * trap_sig * (1.0 - trap_sig);

            let d_a_val = d_adt * dt_val;
            let dd_a_raw = acts.data[base_t + o.dd_a_raw + h];
            let a_unclamped = -softplus(dd_a_raw);
            let was_clamped = a_unclamped > -dims.a_floor;
            if !was_clamped {
                let sp_deriv_a = if dd_a_raw > 20.0 {
                    1.0
                } else {
                    1.0 / (1.0 + fast_exp_scalar(-dd_a_raw))
                };
                scratch.d_dd_a_flat[t * nh + h] = d_a_val * (-sp_deriv_a);
            }
        }
    }

    // ═══ B7: in_proj SGEMM backward ═══
    let n_angles_alloc = n_angles.max(1);
    for t in 0..seq_len {
        let dp = &mut scratch.d_proj_flat[t * ip..(t + 1) * ip];
        let mut off = 0;
        dp[off..off + di].copy_from_slice(&scratch.d_z_flat[t * di..(t + 1) * di]);
        off += di;
        dp[off..off + di].copy_from_slice(&scratch.d_x_flat[t * di..(t + 1) * di]);
        off += di;
        dp[off..off + bsize].copy_from_slice(&scratch.d_b_raw_flat[t * bsize..(t + 1) * bsize]);
        off += bsize;
        dp[off..off + bsize].copy_from_slice(&scratch.d_c_raw_flat[t * bsize..(t + 1) * bsize]);
        off += bsize;
        dp[off..off + nh].copy_from_slice(&scratch.d_dd_dt_flat[t * nh..(t + 1) * nh]);
        off += nh;
        dp[off..off + nh].copy_from_slice(&scratch.d_dd_a_flat[t * nh..(t + 1) * nh]);
        off += nh;
        dp[off..off + nh].copy_from_slice(&scratch.d_trap_flat[t * nh..(t + 1) * nh]);
        off += nh;
        if n_angles > 0 {
            dp[off..off + n_angles]
                .copy_from_slice(&scratch.d_angles_flat[t * n_angles..t * n_angles + n_angles]);
        } else {
            dp[off..off + n_angles_alloc].fill(0.0);
        }
    }

    for t in 0..seq_len {
        let base_t = acts.base(t);
        scratch.post_norm_buf[t * dm..(t + 1) * dm]
            .copy_from_slice(&acts.data[base_t + o.post_norm..base_t + o.post_norm + dm]);
    }

    sgemm_backward(
        &mut scratch.d_post_norm_flat,
        &mut d_layer.in_proj_w,
        None,
        &scratch.d_proj_flat,
        &scratch.post_norm_buf,
        &layer_w.in_proj_w,
        (seq_len, dm, ip),
    );

    // ═══ B8: RMSNorm backward + residual ═══
    scratch.d_residual_buf[..seq_len * dm].copy_from_slice(&d_temporal_flat[..seq_len * dm]);

    let inv_dim = 1.0 / dm as f32;
    for t in 0..seq_len {
        let base_t = acts.base(t);
        let off = t * dm;
        let rms = acts.data[base_t + o.rms_val];
        let inv_rms = 1.0 / rms.max(1e-12);

        let mut rowdot = 0.0_f32;
        for d in 0..dm {
            let x_hat = acts.data[base_t + o.residual + d] * inv_rms;
            let y_d = x_hat * layer_w.norm_weight[d];
            rowdot += scratch.d_post_norm_flat[off + d] * y_d;
        }
        let mean_rowdot = rowdot * inv_dim;

        for d in 0..dm {
            let x_hat = acts.data[base_t + o.residual + d] * inv_rms;
            d_layer.norm_weight[d] += scratch.d_post_norm_flat[off + d] * x_hat;
            d_temporal_flat[off + d] = (layer_w.norm_weight[d] * scratch.d_post_norm_flat[off + d]
                - x_hat * mean_rowdot)
                * inv_rms;
        }
    }

    // Add residual pass-through
    for (dt, &dr) in d_temporal_flat[..seq_len * dm]
        .iter_mut()
        .zip(&scratch.d_residual_buf[..seq_len * dm])
    {
        *dt += dr;
    }
}

#[cfg(test)]
mod tests {
    use super::super::forward::forward_mamba3_layer_batched;
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
    fn test_backward_no_panic() {
        let cfg = test_cfg();
        let dims = Mamba3Dims::from_config(&cfg, 4);
        let mut w = TrainMamba3LayerWeights::zeros(&dims);
        for v in &mut w.norm_weight {
            *v = 1.0;
        }
        for v in &mut w.d_param {
            *v = 1.0;
        }
        for v in &mut w.b_norm_weight {
            *v = 1.0;
        }
        for v in &mut w.c_norm_weight {
            *v = 1.0;
        }

        let mut acts = Mamba3LayerFlat::zeros(dims);
        let mut scratch = Mamba3Scratch::zeros(&dims);
        let mut ssm = vec![0.0; dims.nheads * dims.headdim * dims.d_state];
        let mut k_st = vec![0.0; dims.nheads * dims.d_state];
        let mut v_st = vec![0.0; dims.nheads * dims.headdim];
        let mut a_st = vec![0.0; dims.nheads * dims.num_rope_angles.max(1)];
        let mut temporal = vec![1.0_f32; dims.seq_len * dims.d_model];

        // Forward
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

        // Backward
        let mut d_temporal = vec![1.0_f32; dims.seq_len * dims.d_model];
        let mut d_w = TrainMamba3LayerWeights::zeros(&dims);

        backward_mamba3_layer_batched(
            &mut d_temporal,
            &acts,
            &w,
            &mut d_w,
            &mut scratch,
            &dims,
            None,
        );

        assert!(
            d_temporal.iter().all(|v| v.is_finite()),
            "d_temporal must be finite"
        );
        assert!(
            d_w.in_proj_w.iter().all(|v| v.is_finite()),
            "d_in_proj_w must be finite"
        );
        assert!(
            d_w.out_proj_w.iter().all(|v| v.is_finite()),
            "d_out_proj_w must be finite"
        );
    }

    #[test]
    fn test_gradients_finite() {
        let cfg = test_cfg();
        let dims = Mamba3Dims::from_config(&cfg, 4);
        let mut w = TrainMamba3LayerWeights::zeros(&dims);
        for v in &mut w.norm_weight {
            *v = 1.0;
        }
        for v in &mut w.d_param {
            *v = 1.0;
        }
        for v in &mut w.b_norm_weight {
            *v = 1.0;
        }
        for v in &mut w.c_norm_weight {
            *v = 1.0;
        }
        // Add some nonzero in_proj weights
        for (i, v) in w.in_proj_w.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.01;
        }

        let mut acts = Mamba3LayerFlat::zeros(dims);
        let mut scratch = Mamba3Scratch::zeros(&dims);
        let mut ssm = vec![0.0; dims.nheads * dims.headdim * dims.d_state];
        let mut k_st = vec![0.0; dims.nheads * dims.d_state];
        let mut v_st = vec![0.0; dims.nheads * dims.headdim];
        let mut a_st = vec![0.0; dims.nheads * dims.num_rope_angles.max(1)];
        let mut temporal = vec![0.5_f32; dims.seq_len * dims.d_model];

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

        let mut d_temporal = vec![1.0_f32; dims.seq_len * dims.d_model];
        let mut d_w = TrainMamba3LayerWeights::zeros(&dims);

        backward_mamba3_layer_batched(
            &mut d_temporal,
            &acts,
            &w,
            &mut d_w,
            &mut scratch,
            &dims,
            None,
        );

        // Check ALL gradient tensors are finite
        let check = |name: &str, v: &[f32]| {
            assert!(
                v.iter().all(|x| x.is_finite()),
                "{name} has non-finite gradients"
            );
        };
        check("d_temporal", &d_temporal);
        check("d_norm_weight", &d_w.norm_weight);
        check("d_in_proj_w", &d_w.in_proj_w);
        check("d_dt_bias", &d_w.dt_bias);
        check("d_b_norm_weight", &d_w.b_norm_weight);
        check("d_c_norm_weight", &d_w.c_norm_weight);
        check("d_b_bias", &d_w.b_bias);
        check("d_c_bias", &d_w.c_bias);
        check("d_d_param", &d_w.d_param);
        check("d_out_proj_w", &d_w.out_proj_w);
    }
}
