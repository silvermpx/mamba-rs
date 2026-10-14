//! A float64 transcription of the reference Mamba-3 SISO step, for the SISO
//! gradient tests to take central differences on: the mixer, and the whole
//! model with its gated MLPs and final norm.

use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::cpu::weights::TrainMamba3LayerWeights;
use mamba_rs::mamba3_siso::weights::{Mamba3LayerWeights, Mamba3Weights};
use mamba_rs::mlp::GatedMlpLayerWeights;

// The float64 step oracle, transcribed from the reference's `Mamba3.step`
// and `_postprocess` (mamba_ssm/modules/mamba3.py) at state-spaces/mamba
// e9594ce: heavy-tail A clamped at -a_floor, softplus dt, BCNorm per group,
// the per-head B/C biases, the accumulated RoPE angle rotating interleaved
// pairs, the trapezoidal update, D skip, and the SiLU gate (after the
// grouped RMS norm when the output norm is on). Plain loops in f64,
// written from the reference, not from the crate's kernels.

#[derive(Clone)]
pub struct Layer64 {
    pub ssm: Vec<f64>,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    pub angle: Vec<f64>,
}

/// A block norm: LayerNorm (biased variance, as `nn.LayerNorm`) when it has
/// a bias, RMSNorm otherwise.
fn block_norm64(x: &[f64], (w, b): (&[f32], &[f32]), eps: f64) -> Vec<f64> {
    if b.is_empty() {
        return rms64(x, w, eps);
    }
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(w.iter().zip(b))
        .map(|(v, (&w, &b))| (v - mean) * inv * f64::from(w) + f64::from(b))
        .collect()
}

fn rms64(x: &[f64], w: &[f32], eps: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, &w)| v * inv * f64::from(w))
        .collect()
}

fn matvec64(x: &[f64], w: &[f32], n_out: usize) -> Vec<f64> {
    let mut y = vec![0f64; n_out];
    for (i, &xv) in x.iter().enumerate() {
        for (o, yv) in y.iter_mut().enumerate() {
            *yv += xv * f64::from(w[i * n_out + o]);
        }
    }
    y
}

fn silu64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// One token through one SISO layer: `u` is the residual stream in, the
/// return value the stream out.
pub fn layer_step64(
    cfg: &Mamba3Config,
    lw: &TrainMamba3LayerWeights,
    st: &mut Layer64,
    u: &[f64],
) -> Vec<f64> {
    let (dm, di, n, nh, p_len, g_n) = (
        cfg.d_model,
        cfg.d_inner(),
        cfg.d_state,
        cfg.nheads(),
        cfg.headdim,
        cfg.ngroups,
    );
    let na = cfg.num_rope_angles();
    let eps = f64::from(cfg.rms_norm_eps);
    // The reference's B/C and gate norms keep an epsilon of their own.
    let ssm_eps = f64::from(cfg.ssm_norm_eps);
    let normed = block_norm64(u, (&lw.norm_weight, &lw.norm_bias), eps);
    let proj = matvec64(&normed, &lw.in_proj_w, cfg.in_proj_out_dim());
    let mut o = 0;
    let mut take = |len: usize| {
        let s = proj[o..o + len].to_vec();
        o += len;
        s
    };
    let z = take(di);
    let x = take(di);
    let b = take(g_n * n);
    let c = take(g_n * n);
    let dd_dt = take(nh);
    let dd_a = take(nh);
    let trap_raw = take(nh);
    let ang = take(na);
    let bn: Vec<f64> = b
        .chunks(n)
        .flat_map(|row| rms64(row, &lw.b_norm_weight, ssm_eps))
        .collect();
    let cn: Vec<f64> = c
        .chunks(n)
        .flat_map(|row| rms64(row, &lw.c_norm_weight, ssm_eps))
        .collect();
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut y = vec![0f64; di];
    for h in 0..nh {
        let g = h / (nh / g_n);
        let pre = dd_a[h];
        let heavy = if pre >= 0.0 {
            1.0 + pre
        } else {
            1.0 / (1.0 - pre)
        };
        let a = (-heavy).min(-f64::from(cfg.a_floor));
        let sp_in = dd_dt[h] + f64::from(lw.dt_bias[h]);
        let dt = if sp_in > 20.0 {
            sp_in
        } else {
            sp_in.exp().ln_1p()
        };
        let mut k_rot: Vec<f64> = (0..n)
            .map(|m| bn[g * n + m] + f64::from(lw.b_bias[h * n + m]))
            .collect();
        let mut q_rot: Vec<f64> = (0..n)
            .map(|m| cn[g * n + m] + f64::from(lw.c_bias[h * n + m]))
            .collect();
        for (i, th) in st.angle[h * na..(h + 1) * na].iter_mut().enumerate() {
            *th += ang[i].tanh() * std::f64::consts::PI * dt;
            *th -= two_pi * (*th / two_pi).floor();
            let (s, co) = th.sin_cos();
            for v in [&mut k_rot, &mut q_rot] {
                let (v0, v1) = (v[2 * i], v[2 * i + 1]);
                v[2 * i] = v0 * co - v1 * s;
                v[2 * i + 1] = v0 * s + v1 * co;
            }
        }
        let trap = 1.0 / (1.0 + (-trap_raw[h]).exp());
        let alpha = (a * dt).exp();
        let beta = (1.0 - trap) * dt * alpha;
        let gamma = trap * dt;
        for p in 0..p_len {
            let xv = x[h * p_len + p];
            let vprev = st.v[h * p_len + p];
            let mut acc = 0f64;
            for m in 0..n {
                let s_idx = (h * p_len + p) * n + m;
                let s =
                    alpha * st.ssm[s_idx] + beta * st.k[h * n + m] * vprev + gamma * k_rot[m] * xv;
                st.ssm[s_idx] = s;
                acc += s * q_rot[m];
            }
            y[h * p_len + p] = acc + f64::from(lw.d_param[h]) * xv;
        }
        st.k[h * n..(h + 1) * n].copy_from_slice(&k_rot);
    }
    st.v.copy_from_slice(&x);
    for (h, row) in y.chunks_mut(p_len).enumerate() {
        if cfg.is_outproj_norm {
            let ms = row.iter().map(|v| v * v).sum::<f64>() / p_len as f64;
            let inv = 1.0 / (ms + ssm_eps).sqrt();
            for (p, v) in row.iter_mut().enumerate() {
                *v *= inv * f64::from(lw.norm_gate_weight[h * p_len + p]);
            }
        }
        for (p, v) in row.iter_mut().enumerate() {
            *v *= silu64(z[h * p_len + p]);
        }
    }
    let mixed = matvec64(&y, &lw.out_proj_w, dm);
    u.iter().zip(&mixed).map(|(a, b)| a + b).collect()
}

/// One layer's weights in the training layout [`layer_step64`] reads.
pub fn train_layer(lw: &Mamba3LayerWeights) -> TrainMamba3LayerWeights {
    TrainMamba3LayerWeights {
        norm_weight: lw.norm_weight.clone(),
        norm_bias: lw.norm_bias.clone(),
        in_proj_w: lw.in_proj_w.clone(),
        dt_bias: lw.dt_bias.clone(),
        b_norm_weight: lw.b_norm_weight.clone(),
        c_norm_weight: lw.c_norm_weight.clone(),
        b_bias: lw.b_bias.clone(),
        c_bias: lw.c_bias.clone(),
        d_param: lw.d_param.clone(),
        norm_gate_weight: lw.norm_gate_weight.clone(),
        out_proj_w: lw.out_proj_w.clone(),
    }
}

/// The gated MLP on one row: `x + fc2(value * silu(gate))` with
/// `[value | gate] = fc1(norm2(x))`.
fn mlp64(m: &GatedMlpLayerWeights, x: &[f64], eps: f64) -> Vec<f64> {
    let normed = block_norm64(x, (&m.norm_weight, &m.norm_bias), eps);
    let hidden = m.hidden();
    let up = matvec64(&normed, &m.fc1_w, 2 * hidden);
    let act: Vec<f64> = (0..hidden)
        .map(|j| up[j] * silu64(up[hidden + j]))
        .collect();
    let down = matvec64(&act, &m.fc2_w, x.len());
    x.iter().zip(&down).map(|(a, b)| a + b).collect()
}

/// One token through the whole model: the input projection (none for an
/// identity one), every layer's mixer and gated MLP, and the final norm.
/// `layers` are `w`'s layers through [`train_layer`].
pub fn backbone_step64(
    cfg: &Mamba3Config,
    (w, layers): (&Mamba3Weights, &[TrainMamba3LayerWeights]),
    st: &mut [Layer64],
    input: &[f32],
) -> Vec<f64> {
    let eps = f64::from(cfg.rms_norm_eps);
    let input64: Vec<f64> = input.iter().map(|&v| f64::from(v)).collect();
    let mut x = if w.input_proj_w.is_empty() {
        input64
    } else {
        let mut x = matvec64(&input64, &w.input_proj_w, cfg.d_model);
        for (xv, &b) in x.iter_mut().zip(&w.input_proj_b) {
            *xv += f64::from(b);
        }
        x
    };
    for (l, (lw, s)) in layers.iter().zip(st.iter_mut()).enumerate() {
        x = layer_step64(cfg, lw, s, &x);
        if let Some(m) = w.mlp.get(l) {
            x = mlp64(m, &x, eps);
        }
    }
    block_norm64(&x, (&w.norm_f_weight, &w.norm_f_bias), eps)
}

pub fn zeros64(cfg: &Mamba3Config) -> Vec<Layer64> {
    let (nh, p, n, na) = (
        cfg.nheads(),
        cfg.headdim,
        cfg.d_state,
        cfg.num_rope_angles(),
    );
    (0..cfg.n_layers)
        .map(|_| Layer64 {
            ssm: vec![0.0; nh * p * n],
            k: vec![0.0; nh * n],
            v: vec![0.0; nh * p],
            angle: vec![0.0; nh * na],
        })
        .collect()
}
