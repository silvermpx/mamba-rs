//! Mamba-2 training on the GPU against the CPU trainer: the training
//! forward's rows, every weight gradient and the input gradient over
//! groups, per-channel D, an active dt clamp, biases, both norm orders, no
//! norm with the inner MLP and the released head shape, on windows that
//! end inside a chunk, in f32 and in the tf32, bf16 and f16 lanes; then
//! AdamW steps that lower a loss in every lane.
#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba2::gpu::{Mamba2GpuInferenceEngine, Mamba2GpuTrainer};
use mamba_rs::mamba2::{
    Mamba2Config, Mamba2State, Mamba2Tape, Mamba2TrainScratch, Mamba2Weights,
    backward_mamba2_train, forward_mamba2_train,
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
    w
}

/// Every tensor of `w` with its name, in a fixed order.
fn named(w: &Mamba2Weights) -> Vec<(String, &[f32])> {
    let mut out = vec![
        ("input_proj_w".to_string(), w.input_proj_w.as_slice()),
        ("input_proj_b".to_string(), w.input_proj_b.as_slice()),
        ("norm_f".to_string(), w.norm_f_weight.as_slice()),
    ];
    for (l, lw) in w.layers.iter().enumerate() {
        for (name, v) in [
            ("norm", &lw.norm_weight),
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
    let device = GpuDevice::new(0).unwrap();
    let (input_dim, batch, seq) = (40usize, 2usize, 150usize);
    for (label, cfg) in cases() {
        cfg.validate().unwrap();
        let w = drawn(&cfg, input_dim, 11);
        let dm = cfg.d_model;
        let inputs = det(batch * seq * input_dim, 500, 1.0);
        let d_out = det(batch * seq * dm, 501, 1.0);
        let (want_rows, want_grads, want_d_in) =
            cpu_reference(&cfg, &w, (&inputs, &d_out), (input_dim, batch, seq));

        let mut tr = Mamba2GpuTrainer::new(&device, &w, (cfg, input_dim, batch, seq)).unwrap();
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
        let mut tr = Mamba2GpuTrainer::new(&device, &w, (cfg, input_dim, batch, seq)).unwrap();
        let mut got = vec![0f32; batch * seq * dm];
        tr.forward(&inputs, &mut got).unwrap();
        let engine = Mamba2GpuInferenceEngine::new(&device, &w, cfg, input_dim, batch).unwrap();
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
    let device = GpuDevice::new(0).unwrap();
    let (input_dim, batch, seq) = (40usize, 2usize, 70usize);
    let cfg = cases()[1].1;
    let w = drawn(&cfg, input_dim, 31);
    let dm = cfg.d_model;
    let inputs = det(batch * seq * input_dim, 900, 1.0);
    let target = det(batch * seq * dm, 901, 1.0);
    let mut tr = Mamba2GpuTrainer::new(&device, &w, (cfg, input_dim, batch, seq)).unwrap();
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
    assert!(tr.weights().unwrap() != w, "the weights did not move");
}

/// tf32, bf16 and f16 training against the f32 CPU trainer: the forward
/// rows, the input gradient and every weight gradient, measured per lane;
/// a repeated backward gives the same bits.
#[test]
fn m2_gpu_train_typed_lanes_track_the_cpu() {
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    let device = GpuDevice::new(0).unwrap();
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
            let mut tr =
                Mamba2GpuTrainer::new_with_dtype(&device, &w, (cfg, input_dim, batch, seq), dtype)
                    .unwrap();
            // A scale these gradients do not overflow in f16 (the dynamic
            // scale's first steps would back off from 2^16 and skip).
            tr.load_scaler_state(256.0, 0);
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
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    let device = GpuDevice::new(0).unwrap();
    let (input_dim, batch, seq) = (40usize, 2usize, 70usize);
    let cfg = cases()[1].1;
    let w = drawn(&cfg, input_dim, 31);
    let dm = cfg.d_model;
    let inputs = det(batch * seq * input_dim, 900, 1.0);
    let target = det(batch * seq * dm, 901, 1.0);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let mut tr =
            Mamba2GpuTrainer::new_with_dtype(&device, &w, (cfg, input_dim, batch, seq), dtype)
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
    }
}

/// The sequential path (ScanMode::Sequential: the recurrence replayed
/// segment by segment) against the CPU trainer, in every lane; a repeated
/// backward gives the same bits.
#[test]
fn m2_gpu_train_sequential_matches_the_cpu_trainer() {
    use mamba_rs::config::ScanMode;
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    let device = GpuDevice::new(0).unwrap();
    let (input_dim, batch, seq) = (40usize, 2usize, 150usize);
    for (dtype, fwd_tol, grad_tol) in [
        (WeightDtype::F32, 5e-6, 5e-5),
        (WeightDtype::Tf32, 3e-3, 3e-2),
        (WeightDtype::F16, 5e-3, 5e-2),
        (WeightDtype::Bf16, 3e-2, 2e-1),
    ] {
        for (label, base) in cases() {
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
            let mut tr =
                Mamba2GpuTrainer::new_with_dtype(&device, &w, (cfg, input_dim, batch, seq), dtype)
                    .unwrap();
            tr.load_scaler_state(256.0, 0);
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
