//! A float64 transcription of the reference Mamba-3 MIMO step, shared by the
//! MIMO tests: the decode is held to it, and the gradient test takes
//! central differences on it.

use mamba_rs::mamba3_mimo::{Mamba3MimoConfig, Mamba3MimoLayerWeights, Mamba3MimoWeights};

// The float64 step oracle, transcribed from the reference's
// `mamba3_MIMO_step_ref` (tests/ops/tilelang/test_mamba3_mimo.py) and
// `Mamba3.forward` / `_postprocess` (mamba_ssm/modules/mamba3.py) at
// state-spaces/mamba e9594ce, with the half-split rotation the module uses
// for MIMO. Plain loops in f64, written from the reference, not from the
// crate's kernels.

#[derive(Clone)]
pub struct Layer64 {
    pub ssm: Vec<f64>,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    pub angle: Vec<f64>,
}

fn f64v(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// A block norm: RMSNorm, or `nn.LayerNorm` (biased variance) when the
/// bias is not empty.
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

fn mixer_step64(
    cfg: &Mamba3MimoConfig,
    lw: &Mamba3MimoLayerWeights,
    st: &mut Layer64,
    u: &[f64],
) -> Vec<f64> {
    let (dm, di, n, nh, p_len, g_n, r) = (
        cfg.d_model,
        cfg.d_inner(),
        cfg.d_state,
        cfg.nheads(),
        cfg.headdim,
        cfg.ngroups,
        cfg.mimo_rank,
    );
    let na = cfg.num_rope_angles();
    let normed = block_norm64(
        u,
        (&lw.norm_weight, &lw.norm_bias),
        f64::from(cfg.rms_norm_eps),
    );
    let proj = matvec64(&normed, &lw.in_proj_w, cfg.in_proj_out_dim());
    let mut o = 0;
    let mut take = |len: usize| {
        let s = proj[o..o + len].to_vec();
        o += len;
        s
    };
    let z = take(di);
    let x = take(di);
    let b = take(r * g_n * n);
    let c = take(r * g_n * n);
    let dd_dt = take(nh);
    let dd_a = take(nh);
    let trap_raw = take(nh);
    let ang = take(na);
    let eps = f64::from(cfg.ssm_norm_eps);
    let bn: Vec<f64> = b
        .chunks(n)
        .flat_map(|row| rms64(row, &lw.b_norm_weight, eps))
        .collect();
    let cn: Vec<f64> = c
        .chunks(n)
        .flat_map(|row| rms64(row, &lw.c_norm_weight, eps))
        .collect();
    let mut out = vec![0f64; di];
    for h in 0..nh {
        let g = h / (nh / g_n);
        let pre = dd_a[h];
        let ht = if pre >= 0.0 {
            1.0 + pre
        } else {
            1.0 / (1.0 - pre)
        };
        let a = (-ht).min(-f64::from(cfg.a_floor));
        let sp_in = dd_dt[h] + f64::from(lw.dt_bias[h]);
        let dt = if sp_in > 20.0 {
            sp_in
        } else {
            sp_in.exp().ln_1p()
        };
        let lam = 1.0 / (1.0 + (-trap_raw[h]).exp());
        let alpha = (a * dt).exp();
        let beta = (1.0 - lam) * dt * alpha;
        let gamma = lam * dt;
        for (i, th) in st.angle[h * na..(h + 1) * na].iter_mut().enumerate() {
            *th += ang[i].tanh() * std::f64::consts::PI * dt;
        }
        let rotate = |v: &mut [f64], angle: &[f64]| {
            let half = n / 2;
            for (i, &th) in angle.iter().enumerate() {
                let (s, c) = th.sin_cos();
                let (v0, v1) = (v[i], v[i + half]);
                v[i] = v0 * c - v1 * s;
                v[i + half] = v0 * s + v1 * c;
            }
        };
        let mut k_cur = vec![0f64; r * n];
        let mut q = vec![0f64; r * n];
        for i in 0..r {
            for m in 0..n {
                let src = i * g_n * n + g * n + m;
                let bias = (h * r + i) * n + m;
                k_cur[i * n + m] = bn[src] + f64::from(lw.b_bias[bias]);
                q[i * n + m] = cn[src] + f64::from(lw.c_bias[bias]);
            }
            let angle = st.angle[h * na..(h + 1) * na].to_vec();
            rotate(&mut k_cur[i * n..(i + 1) * n], &angle);
            rotate(&mut q[i * n..(i + 1) * n], &angle);
        }
        let px = |i: usize, p: usize| f64::from(lw.mimo_x[(h * r + i) * p_len + p]);
        let mut y = vec![0f64; r * p_len];
        for p in 0..p_len {
            let xv = x[h * p_len + p];
            let vprev = st.v[h * p_len + p];
            for m in 0..n {
                let s_idx = (h * p_len + p) * n + m;
                let mut s = alpha * st.ssm[s_idx];
                for i in 0..r {
                    s += beta * st.k[(h * r + i) * n + m] * (vprev * px(i, p));
                    s += gamma * k_cur[i * n + m] * (xv * px(i, p));
                }
                st.ssm[s_idx] = s;
            }
            for i in 0..r {
                let mut acc = 0f64;
                for m in 0..n {
                    acc += st.ssm[(h * p_len + p) * n + m] * q[i * n + m];
                }
                y[i * p_len + p] = acc + f64::from(lw.d_param[h]) * xv * px(i, p);
            }
        }
        for i in 0..r {
            let row = &mut y[i * p_len..(i + 1) * p_len];
            if cfg.is_outproj_norm {
                let ms = row.iter().map(|v| v * v).sum::<f64>() / p_len as f64;
                let inv = 1.0 / (ms + eps).sqrt();
                for (p, v) in row.iter_mut().enumerate() {
                    *v *= inv * f64::from(lw.norm_gate_weight[h * p_len + p]);
                }
            }
            for (p, v) in row.iter_mut().enumerate() {
                let zr = z[h * p_len + p] * f64::from(lw.mimo_z[(h * r + i) * p_len + p]);
                *v *= silu64(zr);
            }
        }
        for p in 0..p_len {
            out[h * p_len + p] = (0..r)
                .map(|i| y[i * p_len + p] * f64::from(lw.mimo_o[(h * r + i) * p_len + p]))
                .sum();
        }
        st.k[h * r * n..(h + 1) * r * n].copy_from_slice(&k_cur);
        st.v[h * p_len..(h + 1) * p_len].copy_from_slice(&x[h * p_len..(h + 1) * p_len]);
    }
    let mixed = matvec64(&out, &lw.out_proj_w, dm);
    u.iter().zip(&mixed).map(|(a, b)| a + b).collect()
}

pub fn backbone_step64(
    cfg: &Mamba3MimoConfig,
    w: &Mamba3MimoWeights,
    st: &mut [Layer64],
    input: &[f32],
) -> Vec<f64> {
    let mut x = if w.input_proj_w.is_empty() {
        f64v(input)
    } else {
        let mut x = matvec64(&f64v(input), &w.input_proj_w, cfg.d_model);
        for (xv, &b) in x.iter_mut().zip(&w.input_proj_b) {
            *xv += f64::from(b);
        }
        x
    };
    for (l, (lw, s)) in w.layers.iter().zip(st.iter_mut()).enumerate() {
        x = mixer_step64(cfg, lw, s, &x);
        if let Some(m) = w.mlp.get(l) {
            x = mlp64(m, &x, f64::from(cfg.rms_norm_eps));
        }
    }
    block_norm64(
        &x,
        (&w.norm_f_weight, &w.norm_f_bias),
        f64::from(cfg.rms_norm_eps),
    )
}

pub fn zeros64(cfg: &Mamba3MimoConfig) -> Vec<Layer64> {
    let (nh, p, n, r, na) = (
        cfg.nheads(),
        cfg.headdim,
        cfg.d_state,
        cfg.mimo_rank,
        cfg.num_rope_angles(),
    );
    (0..cfg.n_layers)
        .map(|_| Layer64 {
            ssm: vec![0.0; nh * p * n],
            k: vec![0.0; nh * r * n],
            v: vec![0.0; nh * p],
            angle: vec![0.0; nh * na],
        })
        .collect()
}

fn mlp64(m: &mamba_rs::mlp::GatedMlpLayerWeights, x: &[f64], eps: f64) -> Vec<f64> {
    let normed = block_norm64(x, (&m.norm_weight, &m.norm_bias), eps);
    let hidden = m.hidden();
    let up = matvec64(&normed, &m.fc1_w, 2 * hidden);
    let act: Vec<f64> = (0..hidden)
        .map(|j| up[j] * silu64(up[hidden + j]))
        .collect();
    let down = matvec64(&act, &m.fc2_w, x.len());
    x.iter().zip(&down).map(|(a, b)| a + b).collect()
}
