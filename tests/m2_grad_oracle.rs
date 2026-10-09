//! Mamba-2 training gradients on the CPU against central differences of the
//! float64 reference step, and the training forward against the prefill.

use mamba_rs::mamba_ssm::cpu::prefill::PrefillMode;
use mamba_rs::mamba2::{
    Mamba2Config, Mamba2PrefillScratch, Mamba2State, Mamba2Tape, Mamba2TrainScratch, Mamba2Weights,
    backward_mamba2_train, forward_mamba2_backbone_prefill_mode, forward_mamba2_train,
};

#[path = "common/m2_oracle.rs"]
mod m2_oracle;
use m2_oracle::{step64, zeros64};

fn det(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Weights with every parameter drawn and large enough projections that
/// each tensor's gradient is well above the f64 differences' noise.
fn drawn(cfg: &Mamba2Config, input_dim: usize, seed: u64) -> Mamba2Weights {
    let mut w = Mamba2Weights::init(cfg, input_dim, seed);
    for (l, lw) in w.layers.iter_mut().enumerate() {
        let s = seed * 100 + l as u64 * 20;
        let jitter = |v: &mut Vec<f32>, k: u64, base: f32, scale: f32| {
            let d = det(v.len(), s + k, scale);
            for (x, dv) in v.iter_mut().zip(d) {
                *x = base + dv;
            }
        };
        jitter(&mut lw.d_param, 1, 1.0, 0.5);
        jitter(&mut lw.gate_norm_weight, 2, 1.0, 0.3);
        jitter(&mut lw.norm_weight, 3, 1.0, 0.2);
        jitter(&mut lw.in_proj_b, 4, 0.0, 0.2);
        jitter(&mut lw.out_proj_b, 5, 0.0, 0.2);
        jitter(&mut lw.conv_b, 6, 0.0, 0.3);
        // Larger out_proj so later layers see a real signal.
        for v in &mut lw.out_proj_w {
            *v *= 3.0;
        }
    }
    w.input_proj_b = det(cfg.d_model, seed + 7, 0.1);
    for (i, m) in w.mlp.iter_mut().enumerate() {
        let d = det(m.norm_weight.len(), seed + 30 + i as u64, 0.2);
        for (x, dv) in m.norm_weight.iter_mut().zip(d) {
            *x = 1.0 + dv;
        }
    }
    w
}

/// L = sum over the window of out * probe, in float64 through the oracle.
fn loss64(
    cfg: &Mamba2Config,
    w: &Mamba2Weights,
    inputs: &[f32],
    probe: &[f32],
    input_dim: usize,
) -> f64 {
    let mut st = zeros64(cfg);
    let dm = cfg.d_model;
    inputs
        .chunks_exact(input_dim)
        .zip(probe.chunks_exact(dm))
        .map(|(x, pr)| {
            let out = step64(cfg, w, &mut st, x);
            out.iter()
                .zip(pr)
                .map(|(o, &p)| o * f64::from(p))
                .sum::<f64>()
        })
        .sum()
}

/// Every tensor of the weights, by name, for probing.
fn tensors(w: &mut Mamba2Weights) -> Vec<(String, &mut Vec<f32>)> {
    let mut out: Vec<(String, &mut Vec<f32>)> = vec![
        ("input_proj_w".into(), &mut w.input_proj_w),
        ("input_proj_b".into(), &mut w.input_proj_b),
        ("norm_f".into(), &mut w.norm_f_weight),
    ];
    for (l, lw) in w.layers.iter_mut().enumerate() {
        for (name, v) in [
            ("norm", &mut lw.norm_weight),
            ("in_proj_w", &mut lw.in_proj_w),
            ("in_proj_b", &mut lw.in_proj_b),
            ("conv_w", &mut lw.conv_w),
            ("conv_b", &mut lw.conv_b),
            ("dt_bias", &mut lw.dt_bias),
            ("a_log", &mut lw.a_log),
            ("d", &mut lw.d_param),
            ("gate_norm", &mut lw.gate_norm_weight),
            ("out_proj_w", &mut lw.out_proj_w),
            ("out_proj_b", &mut lw.out_proj_b),
        ] {
            out.push((format!("layer{l}.{name}"), v));
        }
    }
    for (l, m) in w.mlp.iter_mut().enumerate() {
        out.push((format!("mlp{l}.norm"), &mut m.norm_weight));
        out.push((format!("mlp{l}.fc1"), &mut m.fc1_w));
        out.push((format!("mlp{l}.fc2"), &mut m.fc2_w));
    }
    out.retain(|(_, v)| !v.is_empty());
    out
}

fn cases() -> Vec<(&'static str, Mamba2Config)> {
    let base = Mamba2Config {
        d_model: 16,
        d_state: 8,
        headdim: 8,
        expand: 2,
        n_layers: 2,
        d_conv: 3,
        ..Mamba2Config::default()
    };
    vec![
        (
            "groups 2, D per channel, dt clamped, biases",
            Mamba2Config {
                ngroups: 2,
                d_has_hdim: true,
                dt_limit: (0.02, 0.25),
                proj_bias: true,
                ..base
            },
        ),
        (
            "norm before gate, no conv bias",
            Mamba2Config {
                norm_before_gate: true,
                conv_bias: false,
                d_conv: 4,
                ..base
            },
        ),
        (
            "no norm, inner MLP, MLP after",
            Mamba2Config {
                rmsnorm: false,
                d_ssm: 16,
                d_intermediate: 40,
                ..base
            },
        ),
    ]
}

/// The analytic f32 gradient of every tensor, probed at its largest
/// entries and a spread of others, against float64 central differences.
#[test]
fn m2_gradients_match_f64_central_differences() {
    let (input_dim, seq, stride) = (12usize, 9usize, 4usize);
    for (label, cfg) in cases() {
        cfg.validate().unwrap();
        let mut w = drawn(&cfg, input_dim, 31);
        let dm = cfg.d_model;
        let inputs = det(seq * input_dim, 77, 1.0);
        let probe = det(seq * dm, 78, 1.0);

        let mut tape = Mamba2Tape::new(&cfg, seq, input_dim, stride);
        let mut s = Mamba2TrainScratch::default();
        let mut st = Mamba2State::zeros(&cfg);
        let mut out = vec![0f32; seq * dm];
        forward_mamba2_train(
            &mut out,
            &inputs,
            &w,
            &mut st.layers,
            &mut tape,
            (&cfg, &mut s),
        );
        let mut grads = w.zeros_like();
        let mut d_out = probe.clone();
        let mut d_in = vec![0f32; seq * input_dim];
        backward_mamba2_train(
            &mut d_out,
            &mut d_in,
            (&w, &mut grads),
            &tape,
            (&cfg, &mut s),
        );

        let mut checked = 0usize;
        let mut worst = 0f64;
        let grad_tensors: Vec<(String, Vec<f32>)> = tensors(&mut grads)
            .into_iter()
            .map(|(n, v)| (n, v.clone()))
            .collect();
        for (name, g) in &grad_tensors {
            let scale = g.iter().fold(0f32, |a, &v| a.max(v.abs()));
            assert!(scale > 0.0, "{label}: {name} has no gradient");
            // The largest entry, then a spread across the tensor.
            let mut idx: Vec<usize> = vec![
                g.iter()
                    .enumerate()
                    .fold((0, 0f32), |(bi, bv), (i, &v)| {
                        if v.abs() > bv { (i, v.abs()) } else { (bi, bv) }
                    })
                    .0,
            ];
            idx.extend((1..4).map(|k| (k * g.len() / 4 + 1) % g.len()));
            for i in idx {
                let mut plus = w.clone();
                let mut minus = w.clone();
                let base;
                {
                    let mut tp = tensors(&mut plus);
                    let v = &mut tp.iter_mut().find(|(n, _)| n == name).unwrap().1;
                    base = v[i];
                    v[i] = base + 1e-3 * base.abs().max(1.0);
                }
                {
                    let mut tm = tensors(&mut minus);
                    let v = &mut tm.iter_mut().find(|(n, _)| n == name).unwrap().1;
                    v[i] = base - 1e-3 * base.abs().max(1.0);
                }
                let hp = {
                    let mut tp = tensors(&mut plus);
                    f64::from(tp.iter_mut().find(|(n, _)| n == name).unwrap().1[i])
                };
                let hm = {
                    let mut tm = tensors(&mut minus);
                    f64::from(tm.iter_mut().find(|(n, _)| n == name).unwrap().1[i])
                };
                let fd = (loss64(&cfg, &plus, &inputs, &probe, input_dim)
                    - loss64(&cfg, &minus, &inputs, &probe, input_dim))
                    / (hp - hm);
                let err = (f64::from(g[i]) - fd).abs() / fd.abs().max(0.05 * f64::from(scale));
                worst = worst.max(err);
                assert!(
                    err < 5e-3,
                    "{label}: {name}[{i}] analytic {} vs fd {fd:.6e} (rel {err:.3e})",
                    g[i]
                );
                checked += 1;
            }
        }
        // The input gradient, at a spread of entries.
        for i in (0..d_in.len()).step_by(7) {
            let mut xp = inputs.clone();
            let mut xm = inputs.clone();
            xp[i] += 1e-3;
            xm[i] -= 1e-3;
            let fd = (loss64(&cfg, &w, &xp, &probe, input_dim)
                - loss64(&cfg, &w, &xm, &probe, input_dim))
                / f64::from(xp[i] - xm[i]);
            let err = (f64::from(d_in[i]) - fd).abs() / fd.abs().max(1e-2);
            assert!(
                err < 5e-3,
                "{label}: d_input[{i}] {} vs fd {fd:.6e}",
                d_in[i]
            );
        }
        eprintln!("{label}: {checked} weight probes, worst rel {worst:.2e}");
        let _ = &mut w;
    }
}

/// The training forward gives the prefill's outputs and states bit for bit
/// (default build), and the gradients do not depend on the checkpoint
/// stride or on repeating the run.
#[test]
fn m2_train_forward_is_the_prefill_and_stride_free() {
    let (input_dim, seq) = (12usize, 13usize);
    for (label, cfg) in cases() {
        let w = drawn(&cfg, input_dim, 41);
        let dm = cfg.d_model;
        let inputs = det(seq * input_dim, 91, 1.0);
        let probe = det(seq * dm, 92, 1.0);

        let mut pst = Mamba2State::zeros(&cfg);
        let mut ps = Mamba2PrefillScratch::new();
        let mut want = vec![0f32; seq * dm];
        forward_mamba2_backbone_prefill_mode(
            &mut want,
            &inputs,
            &w,
            &mut pst.layers,
            &mut ps,
            (&cfg, seq, PrefillMode::Single),
        );

        let run = |stride: usize| -> (Vec<f32>, Mamba2State, Mamba2Weights) {
            let mut tape = Mamba2Tape::new(&cfg, seq, input_dim, stride);
            let mut s = Mamba2TrainScratch::default();
            let mut st = Mamba2State::zeros(&cfg);
            let mut out = vec![0f32; seq * dm];
            forward_mamba2_train(
                &mut out,
                &inputs,
                &w,
                &mut st.layers,
                &mut tape,
                (&cfg, &mut s),
            );
            let mut grads = w.zeros_like();
            let mut d_out = probe.clone();
            backward_mamba2_train(&mut d_out, &mut [], (&w, &mut grads), &tape, (&cfg, &mut s));
            (out, st, grads)
        };
        let (out, st, g1) = run(4);
        if !cfg!(any(feature = "accelerate", feature = "gemm-blas")) {
            assert_eq!(out, want, "{label}: training forward rows");
            assert_eq!(st, pst, "{label}: training forward states");
        }
        for stride in [1usize, 5, seq] {
            let (_, _, g) = run(stride);
            assert!(g == g1, "{label}: gradients moved at stride {stride}");
        }
        let (_, _, again) = run(4);
        assert!(again == g1, "{label}: gradients not repeatable");
    }
}
