//! Mamba-2 training on the GPU against the CPU trainer: the training
//! forward's rows, every weight gradient and the input gradient over
//! groups, per-channel D, an active dt clamp, biases, both norm orders, no
//! norm with the inner MLP, the released head shape, a conv of 40 taps,
//! heads past what shared memory holds and the gated MLP after the mixer,
//! on windows that end inside a chunk,
//! in f32 and in the tf32, bf16 and f16 lanes. Then AdamW steps that lower a
//! loss in every lane.
#![cfg(feature = "cuda")]

use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::WeightDtype;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::train::TrainSessionCfg;
use mamba_rs::mamba2::cpu::{
    Mamba2Tape, Mamba2TrainScratch, backward_mamba2_train, forward_mamba2_train,
};
use mamba_rs::mamba2::gpu::{Mamba2GpuEngine, Mamba2GpuTrainer};
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

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        num += (f64::from(g) - f64::from(w)).powi(2);
        den += f64::from(w).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

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
            "groups 2, D per channel, dt clamped, biases",
            Mamba2Config {
                d_model: 96,
                headdim: 24,
                d_state: 20,
                ngroups: 2,
                d_conv: 3,
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
                ..base
            },
        ),
        (
            "no norm, inner MLP",
            Mamba2Config {
                rmsnorm: false,
                d_ssm: 64,
                ..base
            },
        ),
        (
            "released head",
            Mamba2Config {
                headdim: 64,
                d_state: 128,
                ..base
            },
        ),
        (
            "a conv of 40 taps over two blocks",
            Mamba2Config { d_conv: 40, ..base },
        ),
        (
            "a head past the chunk state's tile",
            Mamba2Config {
                d_model: 384,
                headdim: 384,
                d_state: 128,
                ..base
            },
        ),
        (
            "the gated MLP after the mixer",
            Mamba2Config {
                d_intermediate: 96,
                ..base
            },
        ),
        (
            "LayerNorm block norms with the gated MLP",
            Mamba2Config {
                block_norm: mamba_rs::BlockNorm::Layer,
                d_intermediate: 96,
                ..base
            },
        ),
    ]
}

/// Heads the sequential backward serves from the replay tape: dS of the
/// head in shared memory, and heads wider than the block.
fn tape_heads() -> Vec<(&'static str, Mamba2Config)> {
    let base = cases()[0].1;
    vec![
        (
            "dS of a 128 by 128 head in shared memory",
            Mamba2Config {
                d_model: 128,
                headdim: 128,
                d_state: 128,
                ..base
            },
        ),
        (
            "heads wider than a block",
            // Two heads and D per channel: a per-head gradient of a head
            // this wide is one f32 sum over every row and channel, and the
            // draw can cancel it to a small fraction of its terms (dt_bias
            // and D did here, past the tolerance on the chunked path too).
            // A second head and D per channel keep each compared tensor
            // from resting on one such sum.
            Mamba2Config {
                d_model: 640,
                expand: 4,
                headdim: 1280,
                d_state: 16,
                d_has_hdim: true,
                ..base
            },
        ),
    ]
}

fn drawn(cfg: &Mamba2Config, input_dim: usize, seed: u64) -> Mamba2Weights {
    let mut w = Mamba2Weights::init(cfg, input_dim, seed);
    for (l, lw) in w.layers.iter_mut().enumerate() {
        let s = seed * 100 + l as u64 * 20;
        for (v, k, base, scale) in [
            (&mut lw.d_param, 1, 1.0, 0.5),
            (&mut lw.gate_norm_weight, 2, 1.0, 0.3),
            (&mut lw.norm_weight, 3, 1.0, 0.2),
            (&mut lw.in_proj_b, 4, 0.0, 0.2),
            (&mut lw.out_proj_b, 5, 0.0, 0.2),
            (&mut lw.conv_b, 6, 0.0, 0.3),
            (&mut lw.norm_bias, 7, 0.0, 0.2),
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
    for (l, m) in w.mlp.iter_mut().enumerate() {
        let d = det(m.norm_weight.len(), seed * 100 + l as u64 * 20 + 9, 0.2);
        for (x, dv) in m.norm_weight.iter_mut().zip(d) {
            *x = 1.0 + dv;
        }
        let d = det(m.norm_bias.len(), seed * 100 + l as u64 * 20 + 11, 0.2);
        for (x, dv) in m.norm_bias.iter_mut().zip(d) {
            *x = dv;
        }
    }
    let d = det(w.norm_f_bias.len(), seed + 13, 0.2);
    for (x, dv) in w.norm_f_bias.iter_mut().zip(d) {
        *x = dv;
    }
    w.input_proj_b = det(cfg.d_model, seed + 7, 0.1);
    w
}

/// Every tensor of `w` with its name, in a fixed order.
fn named(w: &Mamba2Weights) -> Vec<(String, &[f32])> {
    let mut out = vec![
        ("input_proj_w".to_string(), w.input_proj_w.as_slice()),
        ("input_proj_b".to_string(), w.input_proj_b.as_slice()),
        ("norm_f".to_string(), w.norm_f_weight.as_slice()),
        ("norm_f_bias".to_string(), w.norm_f_bias.as_slice()),
    ];
    for (l, lw) in w.layers.iter().enumerate() {
        for (name, v) in [
            ("norm", &lw.norm_weight),
            ("norm_bias", &lw.norm_bias),
            ("in_proj_w", &lw.in_proj_w),
            ("in_proj_b", &lw.in_proj_b),
            ("conv_w", &lw.conv_w),
            ("conv_b", &lw.conv_b),
            ("dt_bias", &lw.dt_bias),
            ("a_log", &lw.a_log),
            ("d", &lw.d_param),
            ("gate_norm", &lw.gate_norm_weight),
            ("out_proj_w", &lw.out_proj_w),
            ("out_proj_b", &lw.out_proj_b),
        ] {
            out.push((format!("layer{l}.{name}"), v.as_slice()));
        }
    }
    for (l, m) in w.mlp.iter().enumerate() {
        for (name, v) in [
            ("norm", &m.norm_weight),
            ("norm_bias", &m.norm_bias),
            ("fc1", &m.fc1_w),
            ("fc2", &m.fc2_w),
        ] {
            out.push((format!("mlp{l}.{name}"), v.as_slice()));
        }
    }
    out
}

/// The CPU trainer over each sequence from zero states, its gradients
/// summed over the sequences: (rows, gradients, input gradient).
fn cpu_reference(
    cfg: &Mamba2Config,
    w: &Mamba2Weights,
    (inputs, d_out): (&[f32], &[f32]),
    (input_dim, batch, seq): (usize, usize, usize),
) -> (Vec<f32>, Mamba2Weights, Vec<f32>) {
    let dm = cfg.d_model;
    let mut rows = vec![0f32; batch * seq * dm];
    let mut d_in = vec![0f32; batch * seq * input_dim];
    let mut grads = w.zeros_like();
    let mut s = Mamba2TrainScratch::default();
    for b in 0..batch {
        let mut tape = Mamba2Tape::new(cfg, seq, input_dim, 16);
        let mut st = Mamba2State::zeros(cfg);
        let x = &inputs[b * seq * input_dim..(b + 1) * seq * input_dim];
        forward_mamba2_train(
            &mut rows[b * seq * dm..(b + 1) * seq * dm],
            x,
            w,
            &mut st.layers,
            &mut tape,
            (cfg, &mut s),
        );
        let mut d = d_out[b * seq * dm..(b + 1) * seq * dm].to_vec();
        backward_mamba2_train(
            &mut d,
            &mut d_in[b * seq * input_dim..(b + 1) * seq * input_dim],
            (w, &mut grads),
            &tape,
            (cfg, &mut s),
        );
    }
    (rows, grads, d_in)
}

// The GPU's exp2f/log1pf and the CPU's portable functions differ in the last
// bits, and the chunked sums run in another order than the CPU's per-token
// scan: measured on these shapes the forward agrees to 6e-7, the input
// gradient to 1e-6 and every weight gradient to 6e-6. The bounds keep
// about ten times that; a wrong index or term fails by orders of magnitude.
#[test]
fn m2_gpu_train_matches_the_cpu_trainer() {
    let (input_dim, batch, seq) = (40usize, 2usize, 150usize);
    for (label, cfg) in cases() {
        cfg.validate().unwrap();
        let w = drawn(&cfg, input_dim, 11);
        let dm = cfg.d_model;
        let inputs = det(batch * seq * input_dim, 500, 1.0);
        let d_out = det(batch * seq * dm, 501, 1.0);
        let (want_rows, want_grads, want_d_in) =
            cpu_reference(&cfg, &w, (&inputs, &d_out), (input_dim, batch, seq));

        let mut tr = Mamba2GpuTrainer::new(
            &w,
            cfg,
            TrainSessionCfg::new(input_dim, batch, seq),
            GpuOptions::default(),
        )
        .unwrap();
        let mut rows = vec![0f32; batch * seq * dm];
        tr.forward(&inputs, &mut rows).unwrap();
        let fwd = rel_l2(&rows, &want_rows);
        let mut d_in = vec![0f32; batch * seq * input_dim];
        tr.backward(&d_out, Some(&mut d_in)).unwrap();
        let grads = tr.grads().unwrap();

        let mut worst = (0f64, String::new());
        for ((name, got), (_, want)) in named(&grads).into_iter().zip(named(&want_grads)) {
            if want.is_empty() {
                continue;
            }
            let e = rel_l2(got, want);
            if e > worst.0 {
                worst = (e, name.clone());
            }
            assert!(e < 5e-5, "{label}: gradient {name} rel_l2 {e:.3e}");
        }
        let din = rel_l2(&d_in, &want_d_in);
        eprintln!(
            "{label}: forward {fwd:.2e}, input gradient {din:.2e}, worst weight gradient \
             {:.2e} ({})",
            worst.0, worst.1
        );
        assert!(fwd < 5e-6, "{label}: forward rel_l2 {fwd:.3e}");
        assert!(din < 1e-5, "{label}: input gradient rel_l2 {din:.3e}");
    }
}

/// The training forward against the inference prefill on the same window.
#[test]
fn m2_gpu_train_forward_tracks_the_prefill() {
    let device = GpuDevice::new(0).unwrap();
    let (input_dim, batch, seq) = (40usize, 2usize, 130usize);
    for (label, cfg) in cases() {
        let w = drawn(&cfg, input_dim, 21);
        let dm = cfg.d_model;
        let inputs = det(batch * seq * input_dim, 700, 1.0);
        let mut tr = Mamba2GpuTrainer::new(
            &w,
            cfg,
            TrainSessionCfg::new(input_dim, batch, seq),
            GpuOptions::default(),
        )
        .unwrap();
        let mut got = vec![0f32; batch * seq * dm];
        tr.forward(&inputs, &mut got).unwrap();
        let engine =
            Mamba2GpuEngine::new(&device, &w, cfg, input_dim, batch, WeightDtype::F32).unwrap();
        let mut st = engine.alloc_state().unwrap();
        let mut pf = engine.alloc_prefill(seq).unwrap();
        let mut want = vec![0f32; batch * seq * dm];
        engine
            .prefill(&inputs, &mut want, &mut st, &mut pf)
            .unwrap();
        let e = rel_l2(&got, &want);
        eprintln!(
            "{label}: training forward vs prefill rel_l2 {e:.2e} ({})",
            if got == want {
                "bitwise"
            } else {
                "not bitwise"
            }
        );
        assert!(e < 1e-5, "{label}: rel_l2 {e:.3e}");
    }
}

/// AdamW on L = 0.5 |out - target|^2 lowers the loss, the gradients of
/// a repeated backward are bit for bit the same, and the weights move.
#[test]
fn m2_gpu_train_adamw_lowers_a_loss() {
    let (input_dim, batch, seq) = (40usize, 2usize, 70usize);
    let cfg = cases()[1].1;
    let w = drawn(&cfg, input_dim, 31);
    let dm = cfg.d_model;
    let inputs = det(batch * seq * input_dim, 900, 1.0);
    let target = det(batch * seq * dm, 901, 1.0);
    let mut tr = Mamba2GpuTrainer::new(
        &w,
        cfg,
        TrainSessionCfg::new(input_dim, batch, seq),
        GpuOptions::default(),
    )
    .unwrap();
    tr.adamw_mut().lr = 3e-3;
    let mut out = vec![0f32; batch * seq * dm];
    let mut losses = Vec::new();
    for step in 0..12 {
        tr.reset_state().unwrap();
        tr.forward(&inputs, &mut out).unwrap();
        let d: Vec<f32> = out.iter().zip(&target).map(|(o, t)| o - t).collect();
        losses.push(d.iter().map(|v| 0.5 * f64::from(*v).powi(2)).sum::<f64>());
        tr.backward(&d, None).unwrap();
        if step == 0 {
            let first = tr.grads().unwrap();
            tr.backward(&d, None).unwrap();
            assert!(tr.grads().unwrap() == first, "a repeated backward moved");
        }
        tr.apply_adamw().unwrap();
    }
    eprintln!("AdamW losses: {:.4} -> {:.4}", losses[0], losses[11]);
    assert!(
        losses[11] < 0.9 * losses[0],
        "loss {:.4} -> {:.4}",
        losses[0],
        losses[11]
    );
    assert!(
        tr.snapshot_master().unwrap() != w,
        "the weights did not move"
    );
}

/// tf32, bf16 and f16 training against the f32 CPU trainer: the forward
/// rows, the input gradient and every weight gradient, measured per lane;
/// a repeated backward gives the same bits.
#[test]
fn m2_gpu_train_typed_lanes_track_the_cpu() {
    use mamba_rs::gpu::dtype::WeightDtype;
    let (input_dim, batch, seq) = (40usize, 2usize, 150usize);
    for (dtype, fwd_tol, grad_tol) in [
        (WeightDtype::Tf32, 3e-3, 3e-2),
        (WeightDtype::F16, 5e-3, 5e-2),
        (WeightDtype::Bf16, 3e-2, 2e-1),
    ] {
        for (label, cfg) in cases() {
            let w = drawn(&cfg, input_dim, 11);
            let dm = cfg.d_model;
            let inputs = det(batch * seq * input_dim, 500, 1.0);
            let d_out = det(batch * seq * dm, 501, 1.0);
            let (want_rows, want_grads, want_d_in) =
                cpu_reference(&cfg, &w, (&inputs, &d_out), (input_dim, batch, seq));
            let mut tr = Mamba2GpuTrainer::new(
                &w,
                cfg,
                TrainSessionCfg::new(input_dim, batch, seq),
                GpuOptions::default().dtype(dtype),
            )
            .unwrap();
            // A scale these gradients do not overflow in f16 (the dynamic
            // scale's first steps would back off from 2^16 and skip).
            tr.load_scaler_state(256.0, 0).unwrap();
            let mut rows = vec![0f32; batch * seq * dm];
            tr.forward(&inputs, &mut rows).unwrap();
            let mut d_in = vec![0f32; batch * seq * input_dim];
            tr.backward(&d_out, Some(&mut d_in)).unwrap();
            assert!(
                !tr.overflowed(),
                "{dtype:?} {label}: the loss scale overflowed"
            );
            let grads = tr.grads().unwrap();
            tr.backward(&d_out, None).unwrap();
            assert!(
                tr.grads().unwrap() == grads,
                "{dtype:?} {label}: a repeated backward moved"
            );
            let fwd = rel_l2(&rows, &want_rows);
            let din = rel_l2(&d_in, &want_d_in);
            let mut worst = (0f64, String::new());
            for ((name, got), (_, want)) in named(&grads).into_iter().zip(named(&want_grads)) {
                if want.is_empty() {
                    continue;
                }
                let e = rel_l2(got, want);
                if e > worst.0 {
                    worst = (e, name);
                }
            }
            eprintln!(
                "{dtype:?} {label}: forward {fwd:.2e}, input gradient {din:.2e}, worst weight \
                 gradient {:.2e} ({})",
                worst.0, worst.1
            );
            assert!(fwd < fwd_tol, "{dtype:?} {label}: forward {fwd:.3e}");
            assert!(
                din < grad_tol,
                "{dtype:?} {label}: input gradient {din:.3e}"
            );
            assert!(
                worst.0 < grad_tol,
                "{dtype:?} {label}: gradient {} {:.3e}",
                worst.1,
                worst.0
            );
        }
    }
}

/// bf16 and f16 AdamW lower the same loss as the f32 test.
#[test]
fn m2_gpu_train_half_lanes_lower_a_loss() {
    use mamba_rs::gpu::dtype::WeightDtype;
    let (input_dim, batch, seq) = (40usize, 2usize, 70usize);
    let cfg = cases()[1].1;
    let w = drawn(&cfg, input_dim, 31);
    let dm = cfg.d_model;
    let inputs = det(batch * seq * input_dim, 900, 1.0);
    let target = det(batch * seq * dm, 901, 1.0);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let mut tr = Mamba2GpuTrainer::new(
            &w,
            cfg,
            TrainSessionCfg::new(input_dim, batch, seq),
            GpuOptions::default().dtype(dtype),
        )
        .unwrap();
        tr.adamw_mut().lr = 3e-3;
        let mut out = vec![0f32; batch * seq * dm];
        let (mut losses, mut applied) = (Vec::new(), 0);
        for _ in 0..12 {
            tr.reset_state().unwrap();
            tr.forward(&inputs, &mut out).unwrap();
            let d: Vec<f32> = out.iter().zip(&target).map(|(o, t)| o - t).collect();
            losses.push(d.iter().map(|v| 0.5 * f64::from(*v).powi(2)).sum::<f64>());
            tr.backward(&d, None).unwrap();
            applied += usize::from(tr.apply_adamw().unwrap());
        }
        eprintln!(
            "{dtype:?} AdamW losses: {:.4} -> {:.4}, {applied} of 12 steps applied, scaler {:?}",
            losses[0],
            losses[11],
            tr.scaler_state()
        );
        // The f16 scale starts at 2^16 and backs off on the first overflows,
        // skipping those steps (as the reference GradScaler does); a few
        // skips at the start are the protocol, not a failure.
        assert!(applied >= 6, "{dtype:?}: only {applied} steps applied");
        assert!(
            losses[11] < 0.9 * losses[0],
            "{dtype:?}: loss {:.4} -> {:.4}",
            losses[0],
            losses[11]
        );
        // The GEMMs read half copies of the matrices: after the steps they
        // must be the stepped weights, so the trainer's forward tracks a CPU
        // forward of the weights it reports.
        let moved = tr.snapshot_master().unwrap();
        tr.reset_state().unwrap();
        tr.forward(&inputs, &mut out).unwrap();
        let (want, _, _) = cpu_reference(&cfg, &moved, (&inputs, &target), (input_dim, batch, seq));
        let drift = rel_l2(&out, &want);
        let tol = if dtype == WeightDtype::F16 {
            5e-3
        } else {
            3e-2
        };
        eprintln!("{dtype:?} after AdamW: forward vs CPU of the stepped weights {drift:.2e}");
        assert!(
            drift < tol,
            "{dtype:?}: forward after the steps {drift:.3e}"
        );
    }
}

/// The sequential path (ScanMode::Sequential: the recurrence replayed
/// segment by segment) against the CPU trainer, in every lane, over every
/// case and the heads whose backward reads the replay tape. A repeated
/// backward gives the same bits.
#[test]
fn m2_gpu_train_sequential_matches_the_cpu_trainer() {
    use mamba_rs::config::ScanMode;
    use mamba_rs::gpu::dtype::WeightDtype;
    let (input_dim, batch, seq) = (40usize, 2usize, 150usize);
    for (dtype, fwd_tol, grad_tol) in [
        (WeightDtype::F32, 5e-6, 5e-5),
        (WeightDtype::Tf32, 3e-3, 3e-2),
        (WeightDtype::F16, 5e-3, 5e-2),
        (WeightDtype::Bf16, 3e-2, 2e-1),
    ] {
        for (label, base) in cases().into_iter().chain(tape_heads()) {
            let cfg = Mamba2Config {
                scan_mode: ScanMode::Sequential,
                ..base
            };
            let w = drawn(&cfg, input_dim, 11);
            let dm = cfg.d_model;
            let inputs = det(batch * seq * input_dim, 500, 1.0);
            let d_out = det(batch * seq * dm, 501, 1.0);
            let (want_rows, want_grads, want_d_in) =
                cpu_reference(&cfg, &w, (&inputs, &d_out), (input_dim, batch, seq));
            let mut tr = Mamba2GpuTrainer::new(
                &w,
                cfg,
                TrainSessionCfg::new(input_dim, batch, seq),
                GpuOptions::default().dtype(dtype),
            )
            .unwrap();
            tr.load_scaler_state(256.0, 0).unwrap();
            let mut rows = vec![0f32; batch * seq * dm];
            tr.forward(&inputs, &mut rows).unwrap();
            let mut d_in = vec![0f32; batch * seq * input_dim];
            tr.backward(&d_out, Some(&mut d_in)).unwrap();
            assert!(!tr.overflowed(), "{dtype:?} {label}: overflow");
            let grads = tr.grads().unwrap();
            tr.backward(&d_out, None).unwrap();
            assert!(
                tr.grads().unwrap() == grads,
                "{dtype:?} {label}: a repeated backward moved"
            );
            let fwd = rel_l2(&rows, &want_rows);
            let din = rel_l2(&d_in, &want_d_in);
            let mut worst = (0f64, String::new());
            for ((name, got), (_, want)) in named(&grads).into_iter().zip(named(&want_grads)) {
                if want.is_empty() {
                    continue;
                }
                let e = rel_l2(got, want);
                if e > worst.0 {
                    worst = (e, name);
                }
            }
            eprintln!(
                "sequential {dtype:?} {label}: forward {fwd:.2e}, input gradient {din:.2e}, \
                 worst weight gradient {:.2e} ({})",
                worst.0, worst.1
            );
            assert!(fwd < fwd_tol, "{dtype:?} {label}: forward {fwd:.3e}");
            assert!(
                din < grad_tol,
                "{dtype:?} {label}: input gradient {din:.3e}"
            );
            assert!(
                worst.0 < grad_tol,
                "{dtype:?} {label}: gradient {} {:.3e}",
                worst.1,
                worst.0
            );
        }
    }
}

fn worst_gradient(got: &Mamba2Weights, want: &Mamba2Weights) -> (f64, String) {
    let mut worst = (0f64, String::new());
    for ((name, g), (_, w)) in named(got).into_iter().zip(named(want)) {
        if w.is_empty() {
            continue;
        }
        let e = rel_l2(g, w);
        if e > worst.0 {
            worst = (e, name);
        }
    }
    worst
}

/// Two windows in a row without a reset, on both paths: the second starts
/// from the conv history and SSM state the first left, so the entering
/// state, the first token's previous state and the conv history all carry
/// values. One shape runs groups with the norm before the gate over exactly
/// two chunks; the other the gated norm with an inner MLP, a head whose dt
/// sits past the softplus cutoff, over three chunks. The GPU matches the
/// CPU trainer carrying the same state.
#[test]
fn m2_gpu_train_continues_a_carried_state() {
    use mamba_rs::config::ScanMode;
    use mamba_rs::gpu::dtype::WeightDtype;
    let (input_dim, batch) = (40usize, 2usize);
    let base = cases()[0].1;
    let shapes = [
        (
            "groups 2, norm before gate, 128 tokens",
            Mamba2Config {
                ngroups: 2,
                norm_before_gate: true,
                d_conv: 3,
                ..base
            },
            128usize,
        ),
        (
            "norm with an inner MLP, softplus past its cutoff, 150 tokens",
            Mamba2Config { d_ssm: 64, ..base },
            150usize,
        ),
    ];
    for mode in [ScanMode::Auto, ScanMode::Sequential] {
        for (label, shape, seq) in shapes {
            let cfg = Mamba2Config {
                scan_mode: mode,
                ..shape
            };
            let mut w = drawn(&cfg, input_dim, 17);
            if label.contains("softplus") {
                for lw in &mut w.layers {
                    lw.dt_bias[0] = 20.0;
                }
            }
            let dm = cfg.d_model;
            let x1 = det(batch * seq * input_dim, 600, 1.0);
            let x2 = det(batch * seq * input_dim, 601, 1.0);
            let d_out = det(batch * seq * dm, 602, 1.0);

            let mut want_rows = vec![0f32; batch * seq * dm];
            let mut want_d_in = vec![0f32; batch * seq * input_dim];
            let mut want_grads = w.zeros_like();
            let mut s = Mamba2TrainScratch::default();
            for b in 0..batch {
                let span = |v: &[f32], n: usize| v[b * seq * n..(b + 1) * seq * n].to_vec();
                let mut st = Mamba2State::zeros(&cfg);
                let mut first = Mamba2Tape::new(&cfg, seq, input_dim, 16);
                let mut rows1 = vec![0f32; seq * dm];
                forward_mamba2_train(
                    &mut rows1,
                    &span(&x1, input_dim),
                    &w,
                    &mut st.layers,
                    &mut first,
                    (&cfg, &mut s),
                );
                let mut tape = Mamba2Tape::new(&cfg, seq, input_dim, 16);
                forward_mamba2_train(
                    &mut want_rows[b * seq * dm..(b + 1) * seq * dm],
                    &span(&x2, input_dim),
                    &w,
                    &mut st.layers,
                    &mut tape,
                    (&cfg, &mut s),
                );
                let mut d = span(&d_out, dm);
                backward_mamba2_train(
                    &mut d,
                    &mut want_d_in[b * seq * input_dim..(b + 1) * seq * input_dim],
                    (&w, &mut want_grads),
                    &tape,
                    (&cfg, &mut s),
                );
            }
            let mut tr = Mamba2GpuTrainer::new(
                &w,
                cfg,
                TrainSessionCfg::new(input_dim, batch, seq),
                GpuOptions::default().dtype(WeightDtype::F32),
            )
            .unwrap();
            let mut rows = vec![0f32; batch * seq * dm];
            tr.forward(&x1, &mut rows).unwrap();
            tr.forward(&x2, &mut rows).unwrap();
            let mut d_in = vec![0f32; batch * seq * input_dim];
            tr.backward(&d_out, Some(&mut d_in)).unwrap();
            let grads = tr.grads().unwrap();
            let fwd = rel_l2(&rows, &want_rows);
            let din = rel_l2(&d_in, &want_d_in);
            let worst = worst_gradient(&grads, &want_grads);
            eprintln!(
                "{mode:?} {label}: carried window forward {fwd:.2e}, input gradient {din:.2e}, \
                 worst weight gradient {:.2e} ({})",
                worst.0, worst.1
            );
            assert!(fwd < 5e-5, "{mode:?} {label}: forward {fwd:.3e}");
            assert!(din < 5e-4, "{mode:?} {label}: input gradient {din:.3e}");
            assert!(
                worst.0 < 5e-4,
                "{mode:?} {label}: gradient {} {:.3e}",
                worst.1,
                worst.0
            );
        }
    }
}

/// The identity input (no input projection, what the checkpoint loader
/// builds) on both paths, in f32 and bf16, against the CPU trainer.
#[test]
fn m2_gpu_train_identity_input() {
    use mamba_rs::config::ScanMode;
    use mamba_rs::gpu::dtype::WeightDtype;
    let base = cases()[2].1;
    let (batch, seq) = (2usize, 100usize);
    let input_dim = base.d_model;
    for mode in [ScanMode::Auto, ScanMode::Sequential] {
        for (dtype, fwd_tol, grad_tol) in [
            (WeightDtype::F32, 5e-5, 5e-4),
            (WeightDtype::Bf16, 3e-2, 2e-1),
        ] {
            let cfg = Mamba2Config {
                scan_mode: mode,
                ..base
            };
            let mut w = drawn(&cfg, input_dim, 19);
            w.input_proj_w.clear();
            w.input_proj_b.clear();
            let dm = cfg.d_model;
            let inputs = det(batch * seq * input_dim, 540, 1.0);
            let d_out = det(batch * seq * dm, 541, 1.0);
            let (want_rows, want_grads, want_d_in) =
                cpu_reference(&cfg, &w, (&inputs, &d_out), (input_dim, batch, seq));
            let mut tr = Mamba2GpuTrainer::new(
                &w,
                cfg,
                TrainSessionCfg::new(input_dim, batch, seq),
                GpuOptions::default().dtype(dtype),
            )
            .unwrap();
            let mut rows = vec![0f32; batch * seq * dm];
            tr.forward(&inputs, &mut rows).unwrap();
            let mut d_in = vec![0f32; batch * seq * input_dim];
            tr.backward(&d_out, Some(&mut d_in)).unwrap();
            let grads = tr.grads().unwrap();
            let fwd = rel_l2(&rows, &want_rows);
            let din = rel_l2(&d_in, &want_d_in);
            let worst = worst_gradient(&grads, &want_grads);
            eprintln!(
                "{mode:?} {dtype:?} identity input: forward {fwd:.2e}, input gradient {din:.2e}, \
                 worst weight gradient {:.2e} ({})",
                worst.0, worst.1
            );
            assert!(fwd < fwd_tol, "{mode:?} {dtype:?}: forward {fwd:.3e}");
            assert!(
                din < grad_tol,
                "{mode:?} {dtype:?}: input gradient {din:.3e}"
            );
            assert!(
                worst.0 < grad_tol,
                "{mode:?} {dtype:?}: gradient {:.3e}",
                worst.0
            );
        }
    }
}

/// An f16 window whose loss scale overflows: the step is skipped (weights
/// untouched), the scale halves, a second step on the same gradients is
/// refused, and the next clean backward steps. A backward with no forward
/// and slices of the wrong length are refused, not run.
#[test]
fn m2_gpu_train_f16_overflow_skips_the_step() {
    use mamba_rs::gpu::dtype::WeightDtype;
    let (input_dim, batch, seq) = (40usize, 2usize, 70usize);
    let cfg = cases()[0].1;
    let w = drawn(&cfg, input_dim, 33);
    let dm = cfg.d_model;
    let inputs = det(batch * seq * input_dim, 910, 1.0);
    let d_out = det(batch * seq * dm, 911, 1.0);
    let mut tr = Mamba2GpuTrainer::new(
        &w,
        cfg,
        TrainSessionCfg::new(input_dim, batch, seq),
        GpuOptions::default().dtype(WeightDtype::F16),
    )
    .unwrap();
    assert!(
        tr.backward(&d_out, None).is_err(),
        "a backward with no forward ran"
    );
    let mut out = vec![0f32; batch * seq * dm];
    assert!(
        tr.forward(&inputs[1..], &mut out).is_err(),
        "a short input ran"
    );
    let before = tr.snapshot_master().unwrap();
    // The scaler's ceiling, 2^24, takes an output gradient of order one
    // past f16's 65504.
    tr.load_scaler_state(2f32.powi(24), 0).unwrap();
    let loaded = tr.scaler_state().map(|s| s.0);
    tr.forward(&inputs, &mut out).unwrap();
    tr.backward(&d_out, None).unwrap();
    assert!(tr.overflowed(), "a 2^24 loss scale did not overflow f16");
    assert!(!tr.apply_adamw().unwrap(), "an overflowed step was applied");
    assert!(
        tr.snapshot_master().unwrap() == before,
        "a skipped step moved the weights"
    );
    assert_eq!(tr.scaler_state().map(|s| s.0), loaded.map(|s| s / 2.0));
    assert!(
        tr.apply_adamw().is_err(),
        "the same gradients stepped twice"
    );
    tr.load_scaler_state(256.0, 0).unwrap();
    tr.forward(&inputs, &mut out).unwrap();
    tr.backward(&d_out, None).unwrap();
    assert!(!tr.overflowed());
    assert!(tr.apply_adamw().unwrap(), "a clean step was skipped");
    assert!(
        tr.snapshot_master().unwrap() != before,
        "a clean step did not move the weights"
    );
}
