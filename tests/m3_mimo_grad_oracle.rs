//! Mamba-3 MIMO training gradients on the CPU against central differences
//! of the float64 reference step, and the training forward against the
//! prefill.

use mamba_rs::mamba_ssm::cpu::prefill::PrefillMode;
use mamba_rs::mamba3_mimo::{
    Mamba3MimoConfig, Mamba3MimoPrefillScratch, Mamba3MimoState, Mamba3MimoTape,
    Mamba3MimoTrainScratch, Mamba3MimoWeights, backward_mamba3_mimo_train,
    forward_mamba3_mimo_backbone_prefill_mode, forward_mamba3_mimo_train,
};

#[path = "common/m3_mimo_oracle.rs"]
mod m3_mimo_oracle;
use m3_mimo_oracle::{backbone_step64, zeros64};

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

/// Weights with every parameter drawn.
fn drawn(cfg: &Mamba3MimoConfig, input_dim: usize, seed: u64) -> Mamba3MimoWeights {
    let mut w = Mamba3MimoWeights::init(cfg, input_dim, seed);
    for (l, lw) in w.layers.iter_mut().enumerate() {
        let s = seed * 100 + l as u64 * 20;
        for (v, k, base, scale) in [
            (&mut lw.b_norm_weight, 1, 1.0, 0.3),
            (&mut lw.c_norm_weight, 2, 1.0, 0.3),
            (&mut lw.b_bias, 3, 0.0, 0.5),
            (&mut lw.c_bias, 4, 0.0, 0.5),
            (&mut lw.d_param, 5, 1.0, 0.5),
            (&mut lw.mimo_x, 6, 0.0, 0.8),
            (&mut lw.mimo_z, 7, 1.0, 0.5),
            (&mut lw.mimo_o, 8, 0.0, 0.8),
            (&mut lw.norm_weight, 9, 1.0, 0.2),
        ] {
            let d = det(v.len(), s + k, scale);
            for (x, dv) in v.iter_mut().zip(d) {
                *x = base + dv;
            }
        }
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

fn loss64(
    cfg: &Mamba3MimoConfig,
    w: &Mamba3MimoWeights,
    inputs: &[f32],
    probe: &[f32],
    input_dim: usize,
) -> f64 {
    let mut st = zeros64(cfg);
    inputs
        .chunks_exact(input_dim)
        .zip(probe.chunks_exact(cfg.d_model))
        .map(|(x, pr)| {
            let out = backbone_step64(cfg, w, &mut st, x);
            out.iter()
                .zip(pr)
                .map(|(o, &p)| o * f64::from(p))
                .sum::<f64>()
        })
        .sum()
}

fn tensors(w: &mut Mamba3MimoWeights) -> Vec<(String, &mut Vec<f32>)> {
    let mut out: Vec<(String, &mut Vec<f32>)> = vec![
        ("input_proj_w".into(), &mut w.input_proj_w),
        ("input_proj_b".into(), &mut w.input_proj_b),
        ("norm_f".into(), &mut w.norm_f_weight),
    ];
    for (l, lw) in w.layers.iter_mut().enumerate() {
        for (name, v) in [
            ("norm", &mut lw.norm_weight),
            ("in_proj_w", &mut lw.in_proj_w),
            ("dt_bias", &mut lw.dt_bias),
            ("b_norm", &mut lw.b_norm_weight),
            ("c_norm", &mut lw.c_norm_weight),
            ("b_bias", &mut lw.b_bias),
            ("c_bias", &mut lw.c_bias),
            ("d", &mut lw.d_param),
            ("mimo_x", &mut lw.mimo_x),
            ("mimo_z", &mut lw.mimo_z),
            ("mimo_o", &mut lw.mimo_o),
            ("out_proj_w", &mut lw.out_proj_w),
        ] {
            out.push((format!("layer{l}.{name}"), v));
        }
    }
    for (l, m) in w.mlp.iter_mut().enumerate() {
        out.push((format!("mlp{l}.norm"), &mut m.norm_weight));
        out.push((format!("mlp{l}.fc1"), &mut m.fc1_w));
        out.push((format!("mlp{l}.fc2"), &mut m.fc2_w));
    }
    out
}

fn cases() -> Vec<(&'static str, Mamba3MimoConfig)> {
    let base = Mamba3MimoConfig {
        d_model: 16,
        d_state: 8,
        headdim: 8,
        expand: 2,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    };
    vec![
        (
            "rank 2, groups 2",
            Mamba3MimoConfig {
                mimo_rank: 2,
                ngroups: 2,
                ..base
            },
        ),
        (
            "rank 4, full rope, MLP after",
            Mamba3MimoConfig {
                rope_fraction: 1.0,
                d_intermediate: 40,
                ..base
            },
        ),
        (
            "rank 3",
            Mamba3MimoConfig {
                mimo_rank: 3,
                ..base
            },
        ),
    ]
}

/// The analytic f32 gradient of every tensor, probed at its largest entry
/// and a spread of others, and of the inputs, against float64 central
/// differences.
#[test]
fn m3_mimo_gradients_match_f64_central_differences() {
    let (input_dim, seq, stride) = (12usize, 9usize, 4usize);
    for (label, cfg) in cases() {
        cfg.validate().unwrap();
        let w = drawn(&cfg, input_dim, 31);
        let dm = cfg.d_model;
        let inputs = det(seq * input_dim, 77, 1.0);
        let probe = det(seq * dm, 78, 1.0);

        let mut tape = Mamba3MimoTape::new(&cfg, seq, input_dim, stride);
        let mut s = Mamba3MimoTrainScratch::default();
        let mut st = Mamba3MimoState::zeros(&cfg);
        let mut out = vec![0f32; seq * dm];
        forward_mamba3_mimo_train(
            &mut out,
            &inputs,
            &w,
            &mut st.layers,
            &mut tape,
            (&cfg, &mut s),
        )
        .unwrap();
        let mut grads = w.zeros_like();
        let mut d_out = probe.clone();
        let mut d_in = vec![0f32; seq * input_dim];
        backward_mamba3_mimo_train(
            &mut d_out,
            &mut d_in,
            (&w, &mut grads),
            &tape,
            (&cfg, &mut s),
        );

        let grad_tensors: Vec<(String, Vec<f32>)> = tensors(&mut grads)
            .into_iter()
            .map(|(n, v)| (n, v.clone()))
            .collect();
        let mut checked = 0usize;
        let mut worst = 0f64;
        for (name, g) in &grad_tensors {
            let scale = g.iter().fold(0f32, |a, &v| a.max(v.abs()));
            assert!(scale > 0.0, "{label}: {name} has no gradient");
            let mut idx = vec![
                g.iter()
                    .enumerate()
                    .fold((0, 0f32), |(bi, bv), (i, &v)| {
                        if v.abs() > bv { (i, v.abs()) } else { (bi, bv) }
                    })
                    .0,
            ];
            idx.extend((1..4).map(|k| (k * g.len() / 4 + 1) % g.len()));
            for i in idx {
                let perturb = |sign: f32| -> (Mamba3MimoWeights, f64) {
                    let mut wp = w.clone();
                    let mut tp = tensors(&mut wp);
                    let v = &mut tp.iter_mut().find(|(n, _)| n == name).unwrap().1;
                    v[i] += sign * 1e-3 * v[i].abs().max(1.0);
                    let at = f64::from(v[i]);
                    drop(tp);
                    (wp, at)
                };
                let (plus, hp) = perturb(1.0);
                let (minus, hm) = perturb(-1.0);
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
    }
}

/// The training forward gives the prefill's outputs and states bit for bit
/// (default build), and the gradients do not depend on the checkpoint
/// stride or on repeating the run.
#[test]
fn m3_mimo_train_forward_is_the_prefill_and_stride_free() {
    let (input_dim, seq) = (12usize, 13usize);
    for (label, cfg) in cases() {
        let w = drawn(&cfg, input_dim, 41);
        let dm = cfg.d_model;
        let inputs = det(seq * input_dim, 91, 1.0);
        let probe = det(seq * dm, 92, 1.0);

        let mut pst = Mamba3MimoState::zeros(&cfg);
        let mut ps = Mamba3MimoPrefillScratch::new(&cfg);
        let mut want = vec![0f32; seq * dm];
        forward_mamba3_mimo_backbone_prefill_mode(
            &mut want,
            &inputs,
            &w,
            &mut pst.layers,
            &mut ps,
            (&cfg, seq, PrefillMode::Single),
        );
        let run = |stride: usize| -> (Vec<f32>, Mamba3MimoState, Mamba3MimoWeights) {
            let mut tape = Mamba3MimoTape::new(&cfg, seq, input_dim, stride);
            let mut s = Mamba3MimoTrainScratch::default();
            let mut st = Mamba3MimoState::zeros(&cfg);
            let mut out = vec![0f32; seq * dm];
            forward_mamba3_mimo_train(
                &mut out,
                &inputs,
                &w,
                &mut st.layers,
                &mut tape,
                (&cfg, &mut s),
            )
            .unwrap();
            let mut grads = w.zeros_like();
            let mut d_out = probe.clone();
            backward_mamba3_mimo_train(
                &mut d_out,
                &mut [],
                (&w, &mut grads),
                &tape,
                (&cfg, &mut s),
            );
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
