//! A float64 transcription of the reference Mamba-2 step, shared by the
//! Mamba-2 tests: the decode is held to it, and the gradient test takes
//! central differences on it.

use mamba_rs::mamba2::{Mamba2Config, Mamba2LayerWeights, Mamba2Weights};

// The float64 step oracle, transcribed from the reference's `Mamba2.step`
// (mamba_ssm/modules/mamba2.py), `selective_state_update`
// (ops/triton/selective_state_update.py) and `RMSNormGated`
// (ops/triton/layernorm_gated.py) at state-spaces/mamba e9594ce: the conv
// window rolled and weighted, softplus, the state as state dA + dt B x,
// y = state C + D x, the grouped gated norm. The dt clamp is the chunked
// forward's (ssd_chunk_state.py); the reference step has none.

#[derive(Clone)]
pub struct Layer64 {
    pub conv: Vec<Vec<f64>>,
    pub ssm: Vec<f64>,
}

fn matvec64(x: &[f64], w: &[f32], b: &[f32], n_out: usize) -> Vec<f64> {
    let mut y: Vec<f64> = if b.is_empty() {
        vec![0.0; n_out]
    } else {
        b.iter().map(|&v| f64::from(v)).collect()
    };
    for (i, &xv) in x.iter().enumerate() {
        for (o, yv) in y.iter_mut().enumerate() {
            *yv += xv * f64::from(w[i * n_out + o]);
        }
    }
    y
}

fn rms64(x: &[f64], w: &[f32], eps: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, &w)| v * inv * f64::from(w))
        .collect()
}

fn silu64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn softplus64(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn mixer64(cfg: &Mamba2Config, lw: &Mamba2LayerWeights, st: &mut Layer64, u: &[f64]) -> Vec<f64> {
    let (d_mlp, d_ssm, n, p_len, g_n, nh, cd, w) = (
        cfg.d_mlp(),
        cfg.ssm_dim(),
        cfg.d_state,
        cfg.headdim,
        cfg.ngroups,
        cfg.nheads(),
        cfg.conv_dim(),
        cfg.d_conv,
    );
    let normed = rms64(u, &lw.norm_weight, f64::from(cfg.rms_norm_eps));
    let proj = matvec64(&normed, &lw.in_proj_w, &lw.in_proj_b, cfg.in_proj_out_dim());
    let z0 = &proj[..d_mlp];
    let x0 = &proj[d_mlp..2 * d_mlp];
    let z = &proj[2 * d_mlp..2 * d_mlp + d_ssm];
    let xbc_raw = &proj[2 * d_mlp + d_ssm..2 * d_mlp + d_ssm + cd];
    let dt_raw = &proj[2 * d_mlp + d_ssm + cd..];

    st.conv.remove(0);
    st.conv.push(xbc_raw.to_vec());
    let xbc: Vec<f64> = (0..cd)
        .map(|c| {
            let mut v: f64 = (0..w)
                .map(|k| st.conv[k][c] * f64::from(lw.conv_w[c * w + k]))
                .sum();
            if let Some(&b) = lw.conv_b.get(c) {
                v += f64::from(b);
            }
            silu64(v)
        })
        .collect();
    let x = &xbc[..d_ssm];
    let b = &xbc[d_ssm..d_ssm + g_n * n];
    let c = &xbc[d_ssm + g_n * n..];
    let mut y = vec![0f64; d_ssm];
    for h in 0..nh {
        let g = h / (nh / g_n);
        let a = -f64::from(lw.a_log[h]).exp();
        let (lo, hi) = cfg.dt_limit;
        let dt =
            softplus64(dt_raw[h] + f64::from(lw.dt_bias[h])).clamp(f64::from(lo), f64::from(hi));
        let da = (dt * a).exp();
        for p in 0..p_len {
            let xp = x[h * p_len + p];
            let mut acc = 0f64;
            for k in 0..n {
                let s = &mut st.ssm[(h * p_len + p) * n + k];
                *s = *s * da + dt * b[g * n + k] * xp;
                acc += *s * c[g * n + k];
            }
            let d = if cfg.d_has_hdim {
                lw.d_param[h * p_len + p]
            } else {
                lw.d_param[h]
            };
            y[h * p_len + p] = acc + f64::from(d) * xp;
        }
    }
    let mut out_ssm = vec![0f64; d_ssm];
    if cfg.rmsnorm {
        let gs = d_ssm / g_n;
        for g in 0..g_n {
            let sl = g * gs..(g + 1) * gs;
            let w_g = &lw.gate_norm_weight[sl.clone()];
            let gated: Vec<f64> = if cfg.norm_before_gate {
                rms64(&y[sl.clone()], w_g, f64::from(cfg.gated_norm_eps))
                    .iter()
                    .zip(&z[sl.clone()])
                    .map(|(v, &zv)| v * silu64(zv))
                    .collect()
            } else {
                let u: Vec<f64> = y[sl.clone()]
                    .iter()
                    .zip(&z[sl.clone()])
                    .map(|(v, &zv)| v * silu64(zv))
                    .collect();
                rms64(&u, w_g, f64::from(cfg.gated_norm_eps))
            };
            out_ssm[sl].copy_from_slice(&gated);
        }
    } else {
        for i in 0..d_ssm {
            out_ssm[i] = y[i] * silu64(z[i]);
        }
    }
    let mut gated: Vec<f64> = z0.iter().zip(x0).map(|(&a, &b)| silu64(a) * b).collect();
    gated.extend(out_ssm);
    let mixed = matvec64(&gated, &lw.out_proj_w, &lw.out_proj_b, cfg.d_model);
    u.iter().zip(&mixed).map(|(a, b)| a + b).collect()
}

fn mlp64(m: &mamba_rs::mlp::GatedMlpLayerWeights, x: &[f64], eps: f64) -> Vec<f64> {
    let normed = rms64(x, &m.norm_weight, eps);
    let hidden = m.hidden();
    let up = matvec64(&normed, &m.fc1_w, &[], 2 * hidden);
    let act: Vec<f64> = (0..hidden)
        .map(|j| up[j] * silu64(up[hidden + j]))
        .collect();
    let down = matvec64(&act, &m.fc2_w, &[], x.len());
    x.iter().zip(&down).map(|(a, b)| a + b).collect()
}

pub fn step64(
    cfg: &Mamba2Config,
    w: &Mamba2Weights,
    st: &mut [Layer64],
    input: &[f32],
) -> Vec<f64> {
    let inp: Vec<f64> = input.iter().map(|&v| f64::from(v)).collect();
    let mut x = if w.input_proj_w.is_empty() {
        inp
    } else {
        matvec64(&inp, &w.input_proj_w, &w.input_proj_b, cfg.d_model)
    };
    for (l, (lw, s)) in w.layers.iter().zip(st.iter_mut()).enumerate() {
        x = mixer64(cfg, lw, s, &x);
        if let Some(m) = w.mlp.get(l) {
            x = mlp64(m, &x, f64::from(cfg.rms_norm_eps));
        }
    }
    rms64(&x, &w.norm_f_weight, f64::from(cfg.rms_norm_eps))
}

pub fn zeros64(cfg: &Mamba2Config) -> Vec<Layer64> {
    (0..cfg.n_layers)
        .map(|_| Layer64 {
            conv: vec![vec![0.0; cfg.conv_dim()]; cfg.d_conv],
            ssm: vec![0.0; cfg.nheads() * cfg.headdim * cfg.d_state],
        })
        .collect()
}
