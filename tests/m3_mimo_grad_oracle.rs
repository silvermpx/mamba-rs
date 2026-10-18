//! Mamba-3 MIMO training gradients on the CPU against central differences
//! of the float64 reference step, and the training forward against the
//! prefill.

use mamba_rs::mamba1::cpu::prefill::PrefillMode;
use mamba_rs::mamba3_mimo::cpu::{
    Mamba3MimoPrefillScratch, Mamba3MimoTape, Mamba3MimoTrainScratch, backward_mamba3_mimo_train,
    forward_mamba3_mimo_backbone_prefill_mode, forward_mamba3_mimo_train,
};
use mamba_rs::mamba3_mimo::{Mamba3MimoConfig, Mamba3MimoState, Mamba3MimoWeights};

#[path = "common/m3_mimo_oracle.rs"]
mod m3_mimo_oracle;
use m3_mimo_oracle::{Layer64, backbone_step64, zeros64};

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
            (&mut lw.norm_bias, 11, 0.0, 0.2),
        ] {
            let d = det(v.len(), s + k, scale);
            for (x, dv) in v.iter_mut().zip(d) {
                *x = base + dv;
            }
        }
        for v in &mut lw.out_proj_w {
            *v *= 3.0;
        }
        let d = det(lw.norm_gate_weight.len(), s + 10, 0.3);
        for (x, dv) in lw.norm_gate_weight.iter_mut().zip(d) {
            *x = 1.0 + dv;
        }
    }
    w.input_proj_b = det(cfg.d_model, seed + 7, 0.1);
    for (i, m) in w.mlp.iter_mut().enumerate() {
        let d = det(m.norm_weight.len(), seed + 30 + i as u64, 0.2);
        for (x, dv) in m.norm_weight.iter_mut().zip(d) {
            *x = 1.0 + dv;
        }
        let d = det(m.norm_bias.len(), seed + 40 + i as u64, 0.2);
        for (x, dv) in m.norm_bias.iter_mut().zip(d) {
            *x = dv;
        }
    }
    let d = det(w.norm_f_bias.len(), seed + 50, 0.2);
    for (x, dv) in w.norm_f_bias.iter_mut().zip(d) {
        *x = dv;
    }
    w
}

/// The f64 copy of a carried state; the layouts are the same.
fn state64(st: &Mamba3MimoState) -> Vec<Layer64> {
    let f = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();
    st.layers
        .iter()
        .map(|l| Layer64 {
            ssm: f(&l.ssm_state),
            k: f(&l.k_state),
            v: f(&l.v_state),
            angle: f(&l.angle_state),
        })
        .collect()
}

/// The probed loss of a window that starts from `start`, which is an input
/// of the window like the tokens are (the gradients stop there).
fn loss64(
    cfg: &Mamba3MimoConfig,
    w: &Mamba3MimoWeights,
    (inputs, probe): (&[f32], &[f32]),
    start: &[Layer64],
    input_dim: usize,
) -> f64 {
    let mut st = start.to_vec();
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

/// Every trained tensor of `w` with its name. The output norm's weight is
/// in the model whether or not it runs, so it counts only `with_norm`.
fn tensors(w: &mut Mamba3MimoWeights, with_norm: bool) -> Vec<(String, &mut Vec<f32>)> {
    let mut out: Vec<(String, &mut Vec<f32>)> = vec![
        ("input_proj_w".into(), &mut w.input_proj_w),
        ("input_proj_b".into(), &mut w.input_proj_b),
        ("norm_f".into(), &mut w.norm_f_weight),
        ("norm_f_bias".into(), &mut w.norm_f_bias),
    ];
    for (l, lw) in w.layers.iter_mut().enumerate() {
        for (name, v) in [
            ("norm", &mut lw.norm_weight),
            ("norm_bias", &mut lw.norm_bias),
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
            ("norm_gate", &mut lw.norm_gate_weight),
            ("out_proj_w", &mut lw.out_proj_w),
        ] {
            if name != "norm_gate" || with_norm {
                out.push((format!("layer{l}.{name}"), v));
            }
        }
    }
    for (l, m) in w.mlp.iter_mut().enumerate() {
        out.push((format!("mlp{l}.norm"), &mut m.norm_weight));
        out.push((format!("mlp{l}.norm_bias"), &mut m.norm_bias));
        out.push((format!("mlp{l}.fc1"), &mut m.fc1_w));
        out.push((format!("mlp{l}.fc2"), &mut m.fc2_w));
    }
    // A LayerNorm's biases are empty in an RMSNorm model.
    out.retain(|(_, v)| !v.is_empty());
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
            "LayerNorm block norms, MLP after",
            Mamba3MimoConfig {
                block_norm: mamba_rs::BlockNorm::Layer,
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
        (
            "rank 1, groups 4",
            Mamba3MimoConfig {
                mimo_rank: 1,
                ngroups: 4,
                ..base
            },
        ),
        (
            "rank 8",
            Mamba3MimoConfig {
                mimo_rank: 8,
                ..base
            },
        ),
        (
            "clamped A, saturated dt and trap",
            Mamba3MimoConfig {
                mimo_rank: 2,
                a_floor: 0.5,
                ..base
            },
        ),
        (
            "rank 2, the gated output norm",
            Mamba3MimoConfig {
                mimo_rank: 2,
                is_outproj_norm: true,
                ..base
            },
        ),
    ]
}

/// Weights for a case: the drawn ones, and for the saturation case a head
/// whose dt sits on both sides of the softplus cutoff and trap columns wide
/// enough to reach |trap| ~ 10 (the A clamp is active through a_floor).
fn case_weights(
    label: &str,
    cfg: &Mamba3MimoConfig,
    input_dim: usize,
    seed: u64,
) -> Mamba3MimoWeights {
    let mut w = drawn(cfg, input_dim, seed);
    if label.starts_with("clamped") {
        let (out, nh) = (cfg.in_proj_out_dim(), cfg.nheads());
        let trap0 = 2 * cfg.d_inner() + 2 * cfg.bc_width() + 2 * nh;
        for lw in &mut w.layers {
            lw.dt_bias[0] = 20.0;
            for row in lw.in_proj_w.chunks_exact_mut(out) {
                for v in &mut row[trap0..trap0 + nh] {
                    *v *= 25.0;
                }
            }
        }
    }
    w
}

/// The probed entries of a gradient: its largest, a spread that does not
/// alias onto one column, and for the structured tensors one entry of every
/// in_proj column group and of the last rank.
fn probe_indices(cfg: &Mamba3MimoConfig, name: &str, g: &[f32]) -> Vec<usize> {
    let len = g.len();
    let argmax = g
        .iter()
        .enumerate()
        .fold((0, 0f32), |(bi, bv), (i, &v)| {
            if v.abs() > bv { (i, v.abs()) } else { (bi, bv) }
        })
        .0;
    let mut idx = vec![argmax];
    idx.extend((1..5).map(|k| (k * (len / 5) + 7 * k + 3) % len));
    let (di, bc, nh, na) = (
        cfg.d_inner(),
        cfg.bc_width(),
        cfg.nheads(),
        cfg.num_rope_angles(),
    );
    let (r, p, n) = (cfg.mimo_rank, cfg.headdim, cfg.d_state);
    if name.ends_with("in_proj_w") {
        let out = cfg.in_proj_out_dim();
        let heads = 2 * di + 2 * bc;
        let rows = len / out;
        let groups = [
            1,
            di + 1,
            2 * di + 1,
            2 * di + bc + 1,
            heads,
            heads + nh,
            heads + 2 * nh,
            heads + 3 * nh,
        ];
        for (k, &col) in groups.iter().enumerate() {
            if col < out && (col < heads + 3 * nh || na > 0) {
                idx.push(((3 * k + 1) % rows) * out + col);
            }
        }
    }
    let h = 1 % nh;
    if ["mimo_x", "mimo_z", "mimo_o"]
        .iter()
        .any(|t| name.ends_with(t))
    {
        idx.push((h * r + r - 1) * p + p / 2);
    }
    if name.ends_with("b_bias") || name.ends_with("c_bias") {
        idx.push((h * r + r - 1) * n + n - 1);
    }
    idx
}

/// The analytic f32 gradient of every tensor (at its largest entry, a
/// spread, every in_proj column group and the last rank) and of the inputs
/// against float64 central differences, for windows that start from zero
/// and from a state carried in from earlier tokens.
#[test]
fn m3_mimo_gradients_match_f64_central_differences() {
    let (input_dim, seq, stride, warm) = (12usize, 9usize, 4usize, 5usize);
    for (label, cfg) in cases() {
        cfg.validate().unwrap();
        let w = case_weights(label, &cfg, input_dim, 31);
        let dm = cfg.d_model;
        let inputs = det(seq * input_dim, 77, 1.0);
        let probe = det(seq * dm, 78, 1.0);
        let mut s = Mamba3MimoTrainScratch::default();
        let mut carried = Mamba3MimoState::zeros(&cfg);
        {
            let mut tape = Mamba3MimoTape::new(&cfg, warm, input_dim, stride);
            let mut out = vec![0f32; warm * dm];
            forward_mamba3_mimo_train(
                &mut out,
                &det(warm * input_dim, 79, 1.0),
                &w,
                &mut carried.layers,
                &mut tape,
                (&cfg, &mut s),
            )
            .unwrap();
        }
        let carried64 = state64(&carried);
        for (start_label, start, start64) in [
            ("zero", Mamba3MimoState::zeros(&cfg), zeros64(&cfg)),
            ("carried", carried, carried64),
        ] {
            let mut tape = Mamba3MimoTape::new(&cfg, seq, input_dim, stride);
            let mut st = start.clone();
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
            )
            .unwrap();

            let grad_tensors: Vec<(String, Vec<f32>)> = tensors(&mut grads, cfg.is_outproj_norm)
                .into_iter()
                .map(|(n, v)| (n, v.clone()))
                .collect();
            let loss = |wv: &Mamba3MimoWeights, xs: &[f32]| {
                loss64(&cfg, wv, (xs, &probe), &start64, input_dim)
            };
            let mut checked = 0usize;
            let mut worst = 0f64;
            for (name, g) in &grad_tensors {
                let scale = g.iter().fold(0f32, |a, &v| a.max(v.abs()));
                assert!(scale > 0.0, "{label}: {name} has no gradient");
                for i in probe_indices(&cfg, name, g) {
                    let perturb = |sign: f32| -> (Mamba3MimoWeights, f64) {
                        let mut wp = w.clone();
                        let mut tp = tensors(&mut wp, cfg.is_outproj_norm);
                        let v = &mut tp.iter_mut().find(|(n, _)| n == name).unwrap().1;
                        v[i] += sign * 1e-3 * v[i].abs().max(1.0);
                        let at = f64::from(v[i]);
                        drop(tp);
                        (wp, at)
                    };
                    let (plus, hp) = perturb(1.0);
                    let (minus, hm) = perturb(-1.0);
                    let fd = (loss(&plus, &inputs) - loss(&minus, &inputs)) / (hp - hm);
                    let err = (f64::from(g[i]) - fd).abs() / fd.abs().max(0.05 * f64::from(scale));
                    worst = worst.max(err);
                    assert!(
                        err < 5e-3,
                        "{label}, {start_label} start: {name}[{i}] analytic {} vs fd {fd:.6e} \
                         (rel {err:.3e})",
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
                let fd = (loss(&w, &xp) - loss(&w, &xm)) / f64::from(xp[i] - xm[i]);
                let err = (f64::from(d_in[i]) - fd).abs() / fd.abs().max(1e-2);
                assert!(
                    err < 5e-3,
                    "{label}, {start_label} start: d_input[{i}] {} vs fd {fd:.6e}",
                    d_in[i]
                );
            }
            eprintln!(
                "{label}, {start_label} start: {checked} weight probes, worst rel {worst:.2e}"
            );
        }
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
            )
            .unwrap();
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
