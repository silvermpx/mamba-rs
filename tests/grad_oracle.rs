//! f64 shadow-forward gradient oracle for the Mamba-1 CPU training path.
//!
//! Every finite-difference test in the crate differences an f32 forward,
//! whose accumulation noise floor (~1e-5) forces coarse steps (h=1e-2) and a
//! 30% per-probe tolerance — good enough to catch sign/axis bugs, unable to
//! certify a gradient formula, and unable to touch high-fan-out parameters
//! at all. This oracle replaces that: a naive, un-optimized f64 reference
//! forward (plain `f64::exp`, sequential recurrence) drops the differencing
//! floor to ~1e-13, so a balanced central-difference step (h ~ 1e-5) yields
//! ~1e-9 numeric accuracy. The crate's f32 ANALYTIC backward is then checked
//! against the f64 numeric gradient at rel_err < 5e-3 per probed element —
//! loose enough to absorb the f32-vs-f64 evaluation gap, tight enough that a
//! wrong term or a missing chain-rule factor (which shows at >> 5e-3) cannot
//! pass. Same philosophy as torch.autograd.gradcheck (f64 reference).
//!
//! Design pins:
//! - The loss is a fixed random unit projection `L = <temporal, proj>` and
//!   the analytic backward is seeded with `d_temporal = proj`, so analytic
//!   and numeric measure the gradient of exactly the same scalar, and no
//!   parameter sits in a flat/near-cancelling direction (the `sum(y^2)`
//!   RMSNorm pathology).
//! - Probes span the |grad| distribution of each tensor (max, quartiles,
//!   minimum) — not only the top elements, which was an f32 noise-floor
//!   workaround this oracle removes.
//! - `a_log` gradients flow through the precomputed `a_neg = -exp(a_log)`;
//!   the shadow recomputes `a_neg` from the perturbed `a_log` on every
//!   probe, guarding the a_neg-refresh coupling at the math level.
//! - Two configs: a rectangular `input_dim != d_model` (the vision patch
//!   embed shape) and the square `input_dim == d_model` case.

use mamba_rs::config::MambaConfig;
use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::backward::backward_mamba_backbone_batched;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
use mamba_rs::train::weights::TrainMambaWeights;
use mamba_rs::weights::MambaWeights;

/// Deterministic xorshift pseudo-random fill (same idiom as the parity tests).
fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * scale
        })
        .collect()
}

fn train_weights_from(w: &MambaWeights) -> TrainMambaWeights {
    use mamba_rs::train::weights::TrainMambaLayerWeights;
    TrainMambaWeights {
        input_proj_w: w.input_proj_w.clone(),
        input_proj_b: w.input_proj_b.clone(),
        layers: w
            .layers
            .iter()
            .map(|lw| TrainMambaLayerWeights {
                norm_weight: lw.norm_weight.clone(),
                in_proj_w: lw.in_proj_w.clone(),
                conv1d_weight: lw.conv1d_weight.clone(),
                conv1d_bias: lw.conv1d_bias.clone(),
                x_proj_w: lw.x_proj_w.clone(),
                dt_proj_w: lw.dt_proj_w.clone(),
                dt_proj_b: lw.dt_proj_b.clone(),
                a_log: lw.a_log.clone(),
                d_param: lw.d_param.clone(),
                out_proj_w: lw.out_proj_w.clone(),
            })
            .collect(),
        norm_f_weight: w.norm_f_weight.clone(),
    }
}

// ---------------------------------------------------------------------------
// f64 shadow weights + forward
// ---------------------------------------------------------------------------

struct F64Layer {
    norm_weight: Vec<f64>,
    in_proj_w: Vec<f64>,
    conv1d_weight: Vec<f64>,
    conv1d_bias: Vec<f64>,
    x_proj_w: Vec<f64>,
    dt_proj_w: Vec<f64>,
    dt_proj_b: Vec<f64>,
    a_log: Vec<f64>,
    d_param: Vec<f64>,
    out_proj_w: Vec<f64>,
}

struct F64Weights {
    input_proj_w: Vec<f64>,
    input_proj_b: Vec<f64>,
    layers: Vec<F64Layer>,
    norm_f_weight: Vec<f64>,
}

fn up(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

impl F64Weights {
    fn from_train(w: &TrainMambaWeights) -> Self {
        Self {
            input_proj_w: up(&w.input_proj_w),
            input_proj_b: up(&w.input_proj_b),
            layers: w
                .layers
                .iter()
                .map(|l| F64Layer {
                    norm_weight: up(&l.norm_weight),
                    in_proj_w: up(&l.in_proj_w),
                    conv1d_weight: up(&l.conv1d_weight),
                    conv1d_bias: up(&l.conv1d_bias),
                    x_proj_w: up(&l.x_proj_w),
                    dt_proj_w: up(&l.dt_proj_w),
                    dt_proj_b: up(&l.dt_proj_b),
                    a_log: up(&l.a_log),
                    d_param: up(&l.d_param),
                    out_proj_w: up(&l.out_proj_w),
                })
                .collect(),
            norm_f_weight: up(&w.norm_f_weight),
        }
    }

    /// Mutable access to one tensor by (name, layer). Layer-scoped names use
    /// `Some(idx)`; backbone-global tensors use `None`.
    fn tensor_mut(&mut self, name: &str, layer: Option<usize>) -> &mut Vec<f64> {
        match (name, layer) {
            ("input_proj_w", None) => &mut self.input_proj_w,
            ("input_proj_b", None) => &mut self.input_proj_b,
            ("norm_f_weight", None) => &mut self.norm_f_weight,
            (n, Some(l)) => {
                let lw = &mut self.layers[l];
                match n {
                    "norm_weight" => &mut lw.norm_weight,
                    "in_proj_w" => &mut lw.in_proj_w,
                    "conv1d_weight" => &mut lw.conv1d_weight,
                    "conv1d_bias" => &mut lw.conv1d_bias,
                    "x_proj_w" => &mut lw.x_proj_w,
                    "dt_proj_w" => &mut lw.dt_proj_w,
                    "dt_proj_b" => &mut lw.dt_proj_b,
                    "a_log" => &mut lw.a_log,
                    "d_param" => &mut lw.d_param,
                    "out_proj_w" => &mut lw.out_proj_w,
                    other => panic!("unknown layer tensor {other}"),
                }
            }
            (n, None) => panic!("unknown global tensor {n}"),
        }
    }
}

/// out[m,n] = x[m,k] @ w[k,n] (+ bias[n]) — mirrors `sgemm_forward` layout.
fn gemm64(
    out: &mut [f64],
    x: &[f64],
    w: &[f64],
    bias: Option<&[f64]>,
    m: usize,
    k: usize,
    n: usize,
) {
    for row in 0..m {
        for col in 0..n {
            let mut acc = bias.map_or(0.0, |b| b[col]);
            for i in 0..k {
                acc += x[row * k + i] * w[i * n + col];
            }
            out[row * n + col] = acc;
        }
    }
}

fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Naive f64 reference forward of the full M1 backbone. Mirrors
/// `forward_mamba_backbone_batched` term for term (zero-initialized
/// recurrent states, softplus threshold at 20.0, `a_neg` recomputed from
/// `a_log`), with plain `f64::exp` everywhere.
fn forward_f64(w: &F64Weights, input: &[f64], dims: &MambaDims) -> Vec<f64> {
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dc = dims.d_conv;
    let dr = dims.dt_rank;
    let xd = dims.xdbl_dim;
    let mid = dims.mamba_input_dim;
    let t_len = dims.seq_len;
    let eps = dims.rms_norm_eps as f64;

    // Input projection.
    let mut temporal = vec![0.0f64; t_len * dm];
    gemm64(
        &mut temporal,
        input,
        &w.input_proj_w,
        Some(&w.input_proj_b),
        t_len,
        mid,
        dm,
    );

    for lw in &w.layers {
        let a_neg: Vec<f64> = lw.a_log.iter().map(|&a| -a.exp()).collect();
        let mut conv_state = vec![0.0f64; di * dc];
        let mut ssm_state = vec![0.0f64; di * ds];

        // F1: RMSNorm.
        let residual = temporal.clone();
        let mut post_norm = vec![0.0f64; t_len * dm];
        for t in 0..t_len {
            let src = &residual[t * dm..(t + 1) * dm];
            let mean_sq = src.iter().map(|v| v * v).sum::<f64>() / dm as f64;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            for d in 0..dm {
                post_norm[t * dm + d] = src[d] * inv_rms * lw.norm_weight[d];
            }
        }

        // F2/F3: in_proj + split + gate SiLU.
        let mut proj = vec![0.0f64; t_len * 2 * di];
        gemm64(
            &mut proj,
            &post_norm,
            &lw.in_proj_w,
            None,
            t_len,
            dm,
            2 * di,
        );

        // F4a: conv1d shift register + fused SiLU.
        let mut u_all = vec![0.0f64; t_len * di];
        for t in 0..t_len {
            for d in 0..di {
                let base = d * dc;
                for k in 0..dc - 1 {
                    conv_state[base + k] = conv_state[base + k + 1];
                }
                conv_state[base + dc - 1] = proj[t * 2 * di + d];
                let mut val = lw.conv1d_bias[d];
                for k in 0..dc {
                    val += conv_state[base + k] * lw.conv1d_weight[base + k];
                }
                u_all[t * di + d] = val * sigmoid64(val);
            }
        }

        // F4b/F4c: x_proj, dt_proj + softplus.
        let mut xdbl = vec![0.0f64; t_len * xd];
        gemm64(&mut xdbl, &u_all, &lw.x_proj_w, None, t_len, di, xd);
        let mut dt_in = vec![0.0f64; t_len * dr];
        for t in 0..t_len {
            dt_in[t * dr..(t + 1) * dr].copy_from_slice(&xdbl[t * xd..t * xd + dr]);
        }
        let mut delta = vec![0.0f64; t_len * di];
        gemm64(
            &mut delta,
            &dt_in,
            &lw.dt_proj_w,
            Some(&lw.dt_proj_b),
            t_len,
            dr,
            di,
        );
        for v in delta.iter_mut() {
            if *v <= 20.0 {
                *v = v.exp().ln_1p();
            }
        }

        // F4d/F4e: SSM recurrence + gating; F5: out_proj; F6: residual.
        let mut gated = vec![0.0f64; t_len * di];
        for t in 0..t_len {
            for d in 0..di {
                let delta_d = delta[t * di + d];
                let u_d = u_all[t * di + d];
                let a_base = d * ds;
                let mut y_d = 0.0f64;
                for n in 0..ds {
                    let idx = a_base + n;
                    let b_n = xdbl[t * xd + dr + n];
                    let c_n = xdbl[t * xd + dr + ds + n];
                    let da = (delta_d * a_neg[idx]).exp();
                    ssm_state[idx] = da * ssm_state[idx] + delta_d * u_d * b_n;
                    y_d += ssm_state[idx] * c_n;
                }
                y_d += lw.d_param[d] * u_d;
                let g = proj[t * 2 * di + di + d];
                gated[t * di + d] = y_d * (g * sigmoid64(g));
            }
        }
        let mut out = vec![0.0f64; t_len * dm];
        gemm64(&mut out, &gated, &lw.out_proj_w, None, t_len, di, dm);
        for i in 0..t_len * dm {
            temporal[i] = residual[i] + out[i];
        }
    }

    // norm_f.
    for t in 0..t_len {
        let off = t * dm;
        let mean_sq = temporal[off..off + dm].iter().map(|v| v * v).sum::<f64>() / dm as f64;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for d in 0..dm {
            temporal[off + d] *= inv_rms * w.norm_f_weight[d];
        }
    }
    temporal
}

// ---------------------------------------------------------------------------
// The oracle harness
// ---------------------------------------------------------------------------

/// Pick probe indices spanning the |grad| distribution: max, quartiles, min.
fn probe_indices(grad: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..grad.len()).collect();
    order.sort_by(|&a, &b| grad[a].abs().partial_cmp(&grad[b].abs()).unwrap());
    let n = order.len();
    let picks = [n - 1, (3 * n) / 4, n / 2, n / 4, 0];
    let mut out: Vec<usize> = picks.iter().map(|&p| order[p.min(n - 1)]).collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn run_oracle(input_dim: usize) {
    let cfg = MambaConfig {
        d_model: 32,
        n_layers: 2,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    };
    let seq_len = 4;
    let dims = MambaDims::from_config(&cfg, seq_len, input_dim);
    let dm = cfg.d_model;
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let dc = cfg.d_conv;
    let nl = cfg.n_layers;

    let weights = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    let tw = train_weights_from(&weights);
    let input = det(seq_len * input_dim, 0xAA, 0.5);

    // Fixed unit-norm random projection: the loss is L = <temporal, proj>,
    // and the analytic backward is seeded with d_temporal = proj so both
    // sides measure the gradient of the same scalar.
    let mut proj = det(seq_len * dm, 0xBB, 1.0);
    let norm = proj.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt() as f32;
    for v in proj.iter_mut() {
        *v /= norm;
    }

    // Analytic side: f32 forward + backward through the crate path.
    let mut a_neg_flat = vec![0.0f32; nl * di * ds];
    for (l, lw) in tw.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg_flat[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }
    let mut acts = MambaBackboneFlat::zeros(dims);
    let mut fwd_scratch = PhaseScratch::zeros(&dims);
    let mut conv_state = vec![0.0f32; nl * di * dc];
    let mut ssm_state = vec![0.0f32; nl * di * ds];
    let mut state = MambaRecurrentState {
        conv: &mut conv_state,
        ssm: &mut ssm_state,
        a_neg: &a_neg_flat,
    };
    let mut temporal = vec![0.0f32; seq_len * dm];
    forward_mamba_backbone_batched(
        &mut temporal,
        &mut acts,
        &tw,
        &input,
        &mut state,
        &mut fwd_scratch,
        &dims,
    );

    let mut grads = TrainMambaWeights::zeros_from_dims(&dims);
    let mut d_temporal = proj.clone();
    let mut bwd_scratch = BackwardPhaseScratch::zeros(&dims);
    backward_mamba_backbone_batched(
        &mut d_temporal,
        &mut grads,
        &acts,
        &tw,
        &a_neg_flat,
        &mut bwd_scratch,
        &dims,
    );

    // Cross-check the f64 shadow forward itself against the f32 forward
    // before trusting its finite differences.
    let mut w64 = F64Weights::from_train(&tw);
    let input64 = up(&input);
    let base64 = forward_f64(&w64, &input64, &dims);
    // Tolerance: the f32 path uses fast_exp (~1.4 ULP Cephes) and f32
    // accumulation through the recurrence; ~1e-3 relative with a small
    // absolute floor is the honest f32-vs-f64 evaluation gap, far below
    // anything a formula error would produce.
    for (i, (&a, &b)) in temporal.iter().zip(base64.iter()).enumerate() {
        let diff = (a as f64 - b).abs();
        let denom = (a as f64).abs().max(b.abs()).max(1e-2);
        assert!(
            diff / denom < 1e-3,
            "f64 shadow forward diverges from f32 forward at {i}: {a} vs {b}"
        );
    }

    let proj64 = up(&proj);
    let loss = |temporal64: &[f64]| -> f64 {
        temporal64
            .iter()
            .zip(proj64.iter())
            .map(|(&t, &p)| t * p)
            .sum()
    };

    let h = 1e-5f64;
    // (name, layer, analytic grad slice)
    let mut checks: Vec<(&str, Option<usize>, &[f32])> = vec![
        ("input_proj_w", None, &grads.input_proj_w),
        ("input_proj_b", None, &grads.input_proj_b),
        ("norm_f_weight", None, &grads.norm_f_weight),
    ];
    for (l, lg) in grads.layers.iter().enumerate() {
        checks.extend([
            ("norm_weight", Some(l), lg.norm_weight.as_slice()),
            ("in_proj_w", Some(l), lg.in_proj_w.as_slice()),
            ("conv1d_weight", Some(l), lg.conv1d_weight.as_slice()),
            ("conv1d_bias", Some(l), lg.conv1d_bias.as_slice()),
            ("x_proj_w", Some(l), lg.x_proj_w.as_slice()),
            ("dt_proj_w", Some(l), lg.dt_proj_w.as_slice()),
            ("dt_proj_b", Some(l), lg.dt_proj_b.as_slice()),
            ("a_log", Some(l), lg.a_log.as_slice()),
            ("d_param", Some(l), lg.d_param.as_slice()),
            ("out_proj_w", Some(l), lg.out_proj_w.as_slice()),
        ]);
    }

    let mut probed = 0usize;
    for (name, layer, analytic) in checks {
        for idx in probe_indices(analytic) {
            let a = analytic[idx] as f64;
            let field = w64.tensor_mut(name, layer);
            let orig = field[idx];
            field[idx] = orig + h;
            let lp = loss(&forward_f64(&w64, &input64, &dims));
            w64.tensor_mut(name, layer)[idx] = orig - h;
            let lm = loss(&forward_f64(&w64, &input64, &dims));
            w64.tensor_mut(name, layer)[idx] = orig;
            let numeric = (lp - lm) / (2.0 * h);

            let denom = a.abs().max(numeric.abs());
            if denom < 1e-6 {
                // Zero-gradient probe: both sides must agree it is zero.
                continue;
            }
            let rel = (a - numeric).abs() / denom;
            assert!(
                rel < 5e-3,
                "{name}{} [{idx}]: analytic {a:e} vs f64 numeric {numeric:e} (rel {rel:e})",
                layer.map_or(String::new(), |l| format!(" L{l}"))
            );
            probed += 1;
        }
    }
    // The harness must actually have exercised the parameter space; a probe
    // count collapse means the picker or the gradients degenerated.
    assert!(probed > 60, "only {probed} non-degenerate probes ran");
}

/// Rectangular input_proj — the vision patch-embed shape (input_dim != d_model).
#[test]
fn m1_gradients_match_f64_oracle_rect_input() {
    run_oracle(20);
}

/// Square input_proj — regression against the existing parity-test shape.
#[test]
fn m1_gradients_match_f64_oracle_square_input() {
    run_oracle(32);
}
