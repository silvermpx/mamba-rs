//! Mamba-3 MIMO decode on the CPU: against a float64 transcription of the
//! reference step, against SISO through the rank-1 equivalence, and its
//! own invariants (determinism, reset, long-sequence stability).

use mamba_rs::mamba3_mimo::{
    Mamba3MimoConfig, Mamba3MimoState, Mamba3MimoStepScratch, Mamba3MimoWeights, mamba3_mimo_step,
};
use mamba_rs::mamba3_siso::{
    Mamba3Config, Mamba3State, Mamba3StepScratch, Mamba3Weights, mamba3_step,
};

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
    let scale = want.iter().fold(0f64, |a, &w| a.max(w.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .fold(0f64, |a, (&g, &w)| a.max((f64::from(g) - w).abs()))
        / scale
}

/// Weights with every parameter drawn, so no constant init hides a path.
fn drawn_weights(cfg: &Mamba3MimoConfig, seed: u64) -> Mamba3MimoWeights {
    let mut w = Mamba3MimoWeights::init(cfg, cfg.d_model, seed);
    for (l, lw) in w.layers.iter_mut().enumerate() {
        let s = seed * 100 + l as u64 * 10;
        let jitter = |v: &mut Vec<f32>, k: u64, base: f32, scale: f32| {
            let d = det(v.len(), s + k, scale);
            for (x, dv) in v.iter_mut().zip(d) {
                *x = base + dv;
            }
        };
        jitter(&mut lw.b_norm_weight, 1, 1.0, 0.3);
        jitter(&mut lw.c_norm_weight, 2, 1.0, 0.3);
        jitter(&mut lw.b_bias, 3, 0.0, 0.5);
        jitter(&mut lw.c_bias, 4, 0.0, 0.5);
        jitter(&mut lw.d_param, 5, 1.0, 0.5);
        jitter(&mut lw.mimo_x, 6, 0.0, 0.8);
        jitter(&mut lw.mimo_z, 7, 1.0, 0.5);
        jitter(&mut lw.mimo_o, 8, 0.0, 0.8);
        jitter(&mut lw.norm_gate_weight, 9, 1.0, 0.3);
        jitter(&mut lw.norm_weight, 10, 1.0, 0.2);
    }
    w
}

#[path = "common/m3_mimo_oracle.rs"]
mod m3_mimo_oracle;
use m3_mimo_oracle::{backbone_step64, zeros64};

/// The f32 decode against the float64 oracle over 24 steps: every rank the
/// fast path and the runtime path serve, grouped B/C, the output norm, and
/// the full rope fraction.
#[test]
fn m3_mimo_decode_matches_the_f64_step_oracle() {
    let base = Mamba3MimoConfig {
        d_model: 64,
        d_state: 16,
        expand: 2,
        headdim: 16,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    };
    let cases = [
        (
            "rank 1",
            Mamba3MimoConfig {
                mimo_rank: 1,
                ..base
            },
        ),
        (
            "rank 2 groups 2",
            Mamba3MimoConfig {
                mimo_rank: 2,
                ngroups: 2,
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
        ("rank 4", base),
        (
            "rank 4 out-norm",
            Mamba3MimoConfig {
                is_outproj_norm: true,
                ..base
            },
        ),
        (
            "rank 4 rope 1.0",
            Mamba3MimoConfig {
                rope_fraction: 1.0,
                ..base
            },
        ),
        (
            "rank 8 headdim 8",
            Mamba3MimoConfig {
                mimo_rank: 8,
                headdim: 8,
                d_state: 24,
                ..base
            },
        ),
    ];
    for (label, cfg) in cases {
        cfg.validate().unwrap();
        let w = drawn_weights(&cfg, 7);
        let mut st32 = Mamba3MimoState::zeros(&cfg);
        let mut scratch = Mamba3MimoStepScratch::new(&cfg);
        let mut st64 = zeros64(&cfg);
        let mut out = vec![0f32; cfg.d_model];
        let mut worst = 0f64;
        for t in 0..24u64 {
            let input = det(cfg.d_model, 300 + t, 1.0);
            mamba3_mimo_step(&mut out, &input, &mut scratch, &w, &mut st32.layers, &cfg);
            let want = backbone_step64(&cfg, &w, &mut st64, &input);
            worst = worst.max(rel_max(&out, &want));
        }
        let state_rel = rel_max(&st32.layers[1].ssm_state, &st64[1].ssm);
        eprintln!("{label}: outputs {worst:.2e}, last state {state_rel:.2e}");
        assert!(worst < 2e-5, "{label}: outputs rel {worst:.3e}");
        assert!(state_rel < 2e-5, "{label}: state rel {state_rel:.3e}");
    }
}

/// The MIMO slot `m` of a SISO state vector: half-split pairs (a, a + N/2)
/// are SISO's adjacent pairs (2a, 2a + 1).
fn siso_slot(m: usize, n: usize) -> usize {
    if m < n / 2 {
        2 * m
    } else {
        2 * (m - n / 2) + 1
    }
}

/// A rank-1 MIMO model with unit projections is a SISO model with its
/// state columns permuted: the same outputs and, column for column, the
/// same states.
#[test]
fn m3_mimo_rank_one_unit_projections_equal_permuted_siso() {
    let siso_cfg = Mamba3Config {
        d_model: 64,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 2,
        n_layers: 2,
        ..Mamba3Config::default()
    };
    siso_cfg.validate().unwrap();
    let mut sw = Mamba3Weights::init(&siso_cfg, 64, 21);
    for (l, lw) in sw.layers.iter_mut().enumerate() {
        let s = 50 + l as u64 * 10;
        for (v, k, base, scale) in [
            (&mut lw.b_norm_weight, 1, 1.0, 0.3),
            (&mut lw.c_norm_weight, 2, 1.0, 0.3),
            (&mut lw.b_bias, 3, 0.0, 0.5),
            (&mut lw.c_bias, 4, 0.0, 0.5),
            (&mut lw.d_param, 5, 1.0, 0.5),
        ] {
            let drawn = det(v.len(), s + k, scale);
            for (x, d) in v.iter_mut().zip(drawn) {
                *x = base + d;
            }
        }
    }
    let cfg = Mamba3MimoConfig {
        d_model: 64,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 2,
        n_layers: 2,
        mimo_rank: 1,
        ..Mamba3MimoConfig::default()
    };
    let (n, g_n, di, nh) = (cfg.d_state, cfg.ngroups, cfg.d_inner(), cfg.nheads());
    let ip = cfg.in_proj_out_dim();
    assert_eq!(ip, siso_cfg.in_proj_out_dim());
    let perm = |v: &[f32]| -> Vec<f32> { (0..n).map(|m| v[siso_slot(m, n)]).collect() };
    let mut w = Mamba3MimoWeights::zeros(&cfg, 64);
    w.input_proj_w.clone_from(&sw.input_proj_w);
    w.input_proj_b.clone_from(&sw.input_proj_b);
    w.norm_f_weight.clone_from(&sw.norm_f_weight);
    for (lw, sl) in w.layers.iter_mut().zip(&sw.layers) {
        lw.norm_weight.clone_from(&sl.norm_weight);
        lw.dt_bias.clone_from(&sl.dt_bias);
        lw.d_param.clone_from(&sl.d_param);
        lw.out_proj_w.clone_from(&sl.out_proj_w);
        lw.mimo_x.fill(1.0);
        lw.mimo_z.fill(1.0);
        lw.mimo_o.fill(1.0);
        lw.b_norm_weight = perm(&sl.b_norm_weight);
        lw.c_norm_weight = perm(&sl.c_norm_weight);
        for h in 0..nh {
            lw.b_bias[h * n..(h + 1) * n].copy_from_slice(&perm(&sl.b_bias[h * n..(h + 1) * n]));
            lw.c_bias[h * n..(h + 1) * n].copy_from_slice(&perm(&sl.c_bias[h * n..(h + 1) * n]));
        }
        // in_proj columns: B and C columns permuted within each group,
        // every other column as SISO has it.
        let (b_off, c_off) = (2 * di, 2 * di + g_n * n);
        for row in 0..cfg.d_model {
            let src = &sl.in_proj_w[row * ip..(row + 1) * ip];
            let dst = &mut lw.in_proj_w[row * ip..(row + 1) * ip];
            dst.copy_from_slice(src);
            for g in 0..g_n {
                for m in 0..n {
                    dst[b_off + g * n + m] = src[b_off + g * n + siso_slot(m, n)];
                    dst[c_off + g * n + m] = src[c_off + g * n + siso_slot(m, n)];
                }
            }
        }
    }
    w.validate(&cfg, 64).unwrap();

    let mut s_state = Mamba3State::zeros(&siso_cfg);
    let mut s_scratch = Mamba3StepScratch::new(&siso_cfg);
    let mut m_state = Mamba3MimoState::zeros(&cfg);
    let mut m_scratch = Mamba3MimoStepScratch::new(&cfg);
    let (mut s_out, mut m_out) = (vec![0f32; 64], vec![0f32; 64]);
    for t in 0..16u64 {
        let input = det(64, 900 + t, 1.0);
        mamba3_step(
            &mut s_out,
            &input,
            &mut s_scratch,
            &sw,
            &mut s_state.layers,
            &siso_cfg,
        );
        mamba3_mimo_step(
            &mut m_out,
            &input,
            &mut m_scratch,
            &w,
            &mut m_state.layers,
            &cfg,
        );
        let want: Vec<f64> = s_out.iter().map(|&v| f64::from(v)).collect();
        let rel = rel_max(&m_out, &want);
        assert!(rel < 1e-5, "step {t}: outputs rel {rel:.3e}");
    }
    for (ms, ss) in m_state.layers.iter().zip(&s_state.layers) {
        let permuted: Vec<f64> = ms
            .ssm_state
            .chunks_exact(n)
            .flat_map(|row| {
                (0..n).map(move |s| row[(0..n).find(|&m| siso_slot(m, n) == s).unwrap()])
            })
            .map(f64::from)
            .collect();
        let rel = rel_max(&ss.ssm_state, &permuted);
        assert!(rel < 1e-5, "states rel {rel:.3e}");
    }
}

/// The same inputs give the same bits, and a reset state replays them.
#[test]
fn m3_mimo_decode_is_deterministic_and_reset_replays() {
    let cfg = Mamba3MimoConfig {
        d_intermediate: 96,
        ..Mamba3MimoConfig::default()
    };
    let w = drawn_weights(&cfg, 3);
    let mut scratch = Mamba3MimoStepScratch::new(&cfg);
    let run = |state: &mut Mamba3MimoState, scratch: &mut Mamba3MimoStepScratch| -> Vec<u32> {
        let mut out = vec![0f32; cfg.d_model];
        let mut bits = Vec::new();
        for t in 0..8u64 {
            mamba3_mimo_step(
                &mut out,
                &det(cfg.d_model, 40 + t, 1.0),
                scratch,
                &w,
                &mut state.layers,
                &cfg,
            );
            bits.extend(out.iter().map(|v| v.to_bits()));
        }
        bits
    };
    let mut state = Mamba3MimoState::zeros(&cfg);
    let first = run(&mut state, &mut scratch);
    state.reset();
    assert_eq!(run(&mut state, &mut scratch), first);
    let mut fresh = Mamba3MimoState::zeros(&cfg);
    assert_eq!(
        run(&mut fresh, &mut Mamba3MimoStepScratch::new(&cfg)),
        first
    );
}

/// A long sequence stays finite and keeps its angles in [0, 2*pi).
#[test]
fn m3_mimo_long_sequence_stays_finite() {
    let cfg = Mamba3MimoConfig {
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    };
    let w = drawn_weights(&cfg, 5);
    let mut state = Mamba3MimoState::zeros(&cfg);
    let mut scratch = Mamba3MimoStepScratch::new(&cfg);
    let mut out = vec![0f32; cfg.d_model];
    for t in 0..2048u64 {
        mamba3_mimo_step(
            &mut out,
            &det(cfg.d_model, 7000 + t, 2.0),
            &mut scratch,
            &w,
            &mut state.layers,
            &cfg,
        );
        assert!(out.iter().all(|v| v.is_finite()), "step {t}");
    }
    for layer in &state.layers {
        assert!(layer.ssm_state.iter().all(|v| v.is_finite()));
        let two_pi = 2.0 * std::f32::consts::PI;
        assert!(
            layer
                .angle_state
                .iter()
                .all(|&a| (0.0..two_pi).contains(&a))
        );
    }
}

/// Bits must match in the default build, whose GEMM adds every output in
/// ascending k whatever the number of rows; a BLAS build tiles differently
/// and is held to a tolerance instead.
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
/// states in both modes, split windows give the whole window's, and decode
/// steps after the window continue the same chain.
#[test]
fn m3_mimo_prefill_matches_the_decode_chain() {
    use mamba_rs::mamba1::cpu::prefill::PrefillMode;
    use mamba_rs::mamba3_mimo::{
        Mamba3MimoPrefillScratch, forward_mamba3_mimo_backbone_prefill_mode,
    };

    let base = Mamba3MimoConfig {
        d_model: 64,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    };
    let cases = [
        (
            "rank 4 groups 2 mlp",
            Mamba3MimoConfig {
                ngroups: 2,
                d_intermediate: 96,
                ..base
            },
        ),
        (
            "rank 2 out-norm",
            Mamba3MimoConfig {
                mimo_rank: 2,
                is_outproj_norm: true,
                ..base
            },
        ),
        (
            "rank 3 no angles",
            Mamba3MimoConfig {
                mimo_rank: 3,
                d_state: 2,
                ..base
            },
        ),
    ];
    let input_dim = 48usize;
    let seq = 37usize;
    for (label, cfg) in cases {
        cfg.validate().unwrap();
        let w = drawn_weights_with_input(&cfg, input_dim, 17);
        let dm = cfg.d_model;
        let inputs = det(seq * input_dim, 404, 1.0);
        let tail: Vec<Vec<f32>> = (0..3u64).map(|t| det(input_dim, 500 + t, 1.0)).collect();

        // The decode chain over the window and three more tokens.
        let mut dec_state = Mamba3MimoState::zeros(&cfg);
        let mut step = Mamba3MimoStepScratch::new(&cfg);
        let mut dec_rows = vec![0f32; seq * dm];
        for (row, x) in dec_rows
            .chunks_exact_mut(dm)
            .zip(inputs.chunks_exact(input_dim))
        {
            mamba3_mimo_step(row, x, &mut step, &w, &mut dec_state.layers, &cfg);
        }
        let window_state = dec_state.clone();
        let mut dec_tail = vec![0f32; 3 * dm];
        for (row, x) in dec_tail.chunks_exact_mut(dm).zip(&tail) {
            mamba3_mimo_step(row, x, &mut step, &w, &mut dec_state.layers, &cfg);
        }

        for mode in [PrefillMode::Single, PrefillMode::Parallel] {
            let mut st = Mamba3MimoState::zeros(&cfg);
            let mut ps = Mamba3MimoPrefillScratch::new(&cfg);
            let mut rows = vec![0f32; seq * dm];
            forward_mamba3_mimo_backbone_prefill_mode(
                &mut rows,
                &inputs,
                &w,
                &mut st.layers,
                &mut ps,
                (&cfg, seq, mode),
            );
            assert_same(&format!("{label} {mode:?} rows"), &rows, &dec_rows);
            for (a, b) in st.layers.iter().zip(&window_state.layers) {
                assert_same(&format!("{label} {mode:?} ssm"), &a.ssm_state, &b.ssm_state);
                assert_same(&format!("{label} {mode:?} k"), &a.k_state, &b.k_state);
                assert_same(&format!("{label} {mode:?} v"), &a.v_state, &b.v_state);
                assert_same(
                    &format!("{label} {mode:?} angle"),
                    &a.angle_state,
                    &b.angle_state,
                );
            }
            let mut out = vec![0f32; 3 * dm];
            for (row, x) in out.chunks_exact_mut(dm).zip(&tail) {
                mamba3_mimo_step(row, x, &mut step, &w, &mut st.layers, &cfg);
            }
            assert_same(&format!("{label} {mode:?} decode after"), &out, &dec_tail);

            // The window in two pieces, 16 then 21 rows, through one scratch.
            let mut st2 = Mamba3MimoState::zeros(&cfg);
            let mut split = vec![0f32; seq * dm];
            let (first, second) = split.split_at_mut(16 * dm);
            forward_mamba3_mimo_backbone_prefill_mode(
                first,
                &inputs[..16 * input_dim],
                &w,
                &mut st2.layers,
                &mut ps,
                (&cfg, 16, mode),
            );
            forward_mamba3_mimo_backbone_prefill_mode(
                second,
                &inputs[16 * input_dim..],
                &w,
                &mut st2.layers,
                &mut ps,
                (&cfg, seq - 16, mode),
            );
            assert_same(&format!("{label} {mode:?} split"), &split, &dec_rows);
        }
    }
}

fn drawn_weights_with_input(
    cfg: &Mamba3MimoConfig,
    input_dim: usize,
    seed: u64,
) -> Mamba3MimoWeights {
    let mut w = drawn_weights(cfg, seed);
    let fresh = Mamba3MimoWeights::init(cfg, input_dim, seed);
    w.input_proj_w = fresh.input_proj_w;
    w.input_proj_b = det(cfg.d_model, seed + 1, 0.1);
    w
}
