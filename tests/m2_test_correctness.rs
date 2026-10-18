//! Mamba-2 on the CPU: against a float64 transcription of the reference
//! step, prefill against the decode chain, and the decode's invariants.

use mamba_rs::mamba1::cpu::prefill::PrefillMode;
use mamba_rs::mamba2::cpu::{
    Mamba2PrefillScratch, Mamba2StepScratch, forward_mamba2_backbone_prefill_mode, mamba2_step,
};
use mamba_rs::mamba2::{Mamba2Config, Mamba2State, Mamba2Weights};

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

fn rel_max(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0f64, |a, &w| a.max(w.abs())).max(1e-3);
    got.iter()
        .zip(want)
        .fold(0f64, |a, (&g, &w)| a.max((f64::from(g) - w).abs()))
        / scale
}

/// Weights with every parameter drawn, biases and norm scales included.
fn drawn(cfg: &Mamba2Config, input_dim: usize, seed: u64) -> Mamba2Weights {
    let mut w = Mamba2Weights::init(cfg, input_dim, seed);
    for (l, lw) in w.layers.iter_mut().enumerate() {
        let s = seed * 100 + l as u64 * 10;
        let jitter = |v: &mut Vec<f32>, k: u64, base: f32, scale: f32| {
            let d = det(v.len(), s + k, scale);
            for (x, dv) in v.iter_mut().zip(d) {
                *x = base + dv;
            }
        };
        jitter(&mut lw.d_param, 1, 1.0, 0.5);
        jitter(&mut lw.gate_norm_weight, 2, 1.0, 0.3);
        jitter(&mut lw.norm_weight, 3, 1.0, 0.2);
        jitter(&mut lw.in_proj_b, 4, 0.0, 0.1);
        jitter(&mut lw.out_proj_b, 5, 0.0, 0.1);
        jitter(&mut lw.norm_bias, 6, 0.0, 0.2);
    }
    for (l, m) in w.mlp.iter_mut().enumerate() {
        let s = seed * 100 + l as u64 * 10 + 7;
        for (k, (v, base)) in [(&mut m.norm_weight, 1.0), (&mut m.norm_bias, 0.0)]
            .into_iter()
            .enumerate()
        {
            let d = det(v.len(), s + k as u64, 0.2);
            for (x, dv) in v.iter_mut().zip(d) {
                *x = base + dv;
            }
        }
    }
    for (x, dv) in w
        .norm_f_bias
        .iter_mut()
        .zip(det(cfg.d_model, seed + 9, 0.2))
    {
        *x = dv;
    }
    w.input_proj_b = det(cfg.d_model, seed + 7, 0.1);
    w
}

#[path = "common/m2_oracle.rs"]
mod m2_oracle;
use m2_oracle::{step64, zeros64};

fn cases() -> Vec<(&'static str, Mamba2Config)> {
    let base = Mamba2Config {
        d_model: 64,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba2Config::default()
    };
    vec![
        ("default", base),
        (
            "groups 2, D per channel, dt clamped",
            Mamba2Config {
                d_model: 96,
                headdim: 24,
                d_state: 20,
                ngroups: 2,
                d_conv: 3,
                d_has_hdim: true,
                dt_limit: (0.01, 0.2),
                ..base
            },
        ),
        (
            "norm before gate, biases, no conv bias",
            Mamba2Config {
                norm_before_gate: true,
                proj_bias: true,
                conv_bias: false,
                ..base
            },
        ),
        (
            "no norm, inner MLP, MLP after",
            Mamba2Config {
                rmsnorm: false,
                d_ssm: 64,
                d_intermediate: 96,
                ..base
            },
        ),
        (
            "official head",
            Mamba2Config {
                headdim: 64,
                d_state: 128,
                ..base
            },
        ),
        (
            "LayerNorm block norms, MLP after",
            Mamba2Config {
                block_norm: mamba_rs::BlockNorm::Layer,
                d_intermediate: 96,
                ..base
            },
        ),
    ]
}

/// The f32 decode against the float64 oracle over 40 steps.
#[test]
fn m2_decode_matches_the_f64_step_oracle() {
    let input_dim = 40usize;
    for (label, cfg) in cases() {
        cfg.validate().unwrap();
        let w = drawn(&cfg, input_dim, 11);
        let mut st = Mamba2State::zeros(&cfg);
        let mut scratch = Mamba2StepScratch::new(&cfg);
        let mut st64 = zeros64(&cfg);
        let mut out = vec![0f32; cfg.d_model];
        let mut worst = 0f64;
        for t in 0..40u64 {
            let input = det(input_dim, 70 + t, 1.0);
            mamba2_step(&mut out, &input, &mut scratch, &w, &mut st.layers, &cfg);
            let want = step64(&cfg, &w, &mut st64, &input);
            worst = worst.max(rel_max(&out, &want));
        }
        let mut reference = vec![0f32; st.layers[1].ssm.len()];
        st.layers[1].ssm_to_reference(&cfg, &mut reference);
        let state_rel = rel_max(&reference, &st64[1].ssm);
        eprintln!("{label}: outputs {worst:.2e}, last state {state_rel:.2e}");
        assert!(worst < 5e-5, "{label}: outputs rel {worst:.3e}");
        assert!(state_rel < 5e-5, "{label}: state rel {state_rel:.3e}");
    }
}

fn assert_same(label: &str, got: &[f32], want: &[f32]) {
    if cfg!(any(feature = "accelerate", feature = "gemm-blas")) {
        let w64: Vec<f64> = want.iter().map(|&v| f64::from(v)).collect();
        let rel = rel_max(got, &w64);
        assert!(rel < 1e-5, "{label}: rel {rel:.3e}");
    } else {
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(got), bits(want), "{label}");
    }
}

/// A window prefilled in one pass gives the decode chain's outputs and
/// states in both modes (bit for bit in the default build), split windows
/// give the whole window's, and decoding on after the window continues the
/// same chain.
#[test]
fn m2_prefill_matches_the_decode_chain() {
    let input_dim = 40usize;
    let seq = 37usize;
    for (label, cfg) in cases() {
        let w = drawn(&cfg, input_dim, 23);
        let dm = cfg.d_model;
        let inputs = det(seq * input_dim, 404, 1.0);
        let tail: Vec<Vec<f32>> = (0..3u64).map(|t| det(input_dim, 500 + t, 1.0)).collect();

        let mut dec = Mamba2State::zeros(&cfg);
        let mut step = Mamba2StepScratch::new(&cfg);
        let mut dec_rows = vec![0f32; seq * dm];
        for (row, x) in dec_rows
            .chunks_exact_mut(dm)
            .zip(inputs.chunks_exact(input_dim))
        {
            mamba2_step(row, x, &mut step, &w, &mut dec.layers, &cfg);
        }
        let window_state = dec.clone();
        let mut dec_tail = vec![0f32; 3 * dm];
        for (row, x) in dec_tail.chunks_exact_mut(dm).zip(&tail) {
            mamba2_step(row, x, &mut step, &w, &mut dec.layers, &cfg);
        }

        for mode in [PrefillMode::Single, PrefillMode::Parallel] {
            let mut st = Mamba2State::zeros(&cfg);
            let mut ps = Mamba2PrefillScratch::new();
            let mut rows = vec![0f32; seq * dm];
            forward_mamba2_backbone_prefill_mode(
                &mut rows,
                &inputs,
                &w,
                &mut st.layers,
                &mut ps,
                (&cfg, seq, mode),
            );
            assert_same(&format!("{label} {mode:?} rows"), &rows, &dec_rows);
            for (a, b) in st.layers.iter().zip(&window_state.layers) {
                assert_same(&format!("{label} {mode:?} ssm"), &a.ssm, &b.ssm);
                assert_same(
                    &format!("{label} {mode:?} conv"),
                    &a.conv_hist,
                    &b.conv_hist,
                );
            }
            let mut out = vec![0f32; 3 * dm];
            for (row, x) in out.chunks_exact_mut(dm).zip(&tail) {
                mamba2_step(row, x, &mut step, &w, &mut st.layers, &cfg);
            }
            assert_same(&format!("{label} {mode:?} decode after"), &out, &dec_tail);

            // 2 + 35 rows: the first window shorter than the conv width.
            let mut st2 = Mamba2State::zeros(&cfg);
            let mut split = vec![0f32; seq * dm];
            let (first, second) = split.split_at_mut(2 * dm);
            forward_mamba2_backbone_prefill_mode(
                first,
                &inputs[..2 * input_dim],
                &w,
                &mut st2.layers,
                &mut ps,
                (&cfg, 2, mode),
            );
            forward_mamba2_backbone_prefill_mode(
                second,
                &inputs[2 * input_dim..],
                &w,
                &mut st2.layers,
                &mut ps,
                (&cfg, seq - 2, mode),
            );
            assert_same(&format!("{label} {mode:?} split"), &split, &dec_rows);
        }
    }
}

/// The same inputs give the same bits, a reset state replays them, and a
/// long sequence stays finite.
#[test]
fn m2_decode_is_deterministic_and_stable() {
    let cfg = Mamba2Config {
        n_layers: 2,
        ..Mamba2Config::default()
    };
    let w = drawn(&cfg, cfg.d_model, 5);
    let mut st = Mamba2State::zeros(&cfg);
    let mut scratch = Mamba2StepScratch::new(&cfg);
    let mut out = vec![0f32; cfg.d_model];
    let mut first = Vec::new();
    for t in 0..8u64 {
        mamba2_step(
            &mut out,
            &det(cfg.d_model, 40 + t, 1.0),
            &mut scratch,
            &w,
            &mut st.layers,
            &cfg,
        );
        first.extend(out.iter().map(|v| v.to_bits()));
    }
    st.reset();
    let mut again = Vec::new();
    for t in 0..8u64 {
        mamba2_step(
            &mut out,
            &det(cfg.d_model, 40 + t, 1.0),
            &mut scratch,
            &w,
            &mut st.layers,
            &cfg,
        );
        again.extend(out.iter().map(|v| v.to_bits()));
    }
    assert_eq!(first, again);
    for t in 0..2048u64 {
        mamba2_step(
            &mut out,
            &det(cfg.d_model, 9000 + t, 3.0),
            &mut scratch,
            &w,
            &mut st.layers,
            &cfg,
        );
        assert!(out.iter().all(|v| v.is_finite()), "step {t}");
    }
}
