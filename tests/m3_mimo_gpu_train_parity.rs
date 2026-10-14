//! Mamba-3 MIMO training on the GPU, the chunked path and the sequential
//! one, against the CPU trainer: the training forward's rows, every weight
//! gradient and the input gradient over ranks, groups, rope fractions, the
//! MLP after the mixer, the gated output norm, a head that is not a whole
//! number of warps, the released head shape and heads past what one block
//! holds, in f32 and in the tf32, bf16 and f16 lanes. The gradients do not depend on the
//! checkpoint stride, a repeated backward gives the same bits, and AdamW
//! lowers a loss in every lane.
#![cfg(feature = "cuda")]

use mamba_rs::config::ScanMode;
use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::train::TrainSessionCfg;
use mamba_rs::mamba3_mimo::gpu::Mamba3MimoGpuTrainer;
use mamba_rs::mamba3_mimo::{
    Mamba3MimoConfig, Mamba3MimoState, Mamba3MimoTape, Mamba3MimoTrainScratch, Mamba3MimoWeights,
    backward_mamba3_mimo_train, forward_mamba3_mimo_train,
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

fn cases() -> Vec<(&'static str, Mamba3MimoConfig)> {
    let base = Mamba3MimoConfig {
        d_model: 64,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    };
    vec![
        (
            "rank 2, groups 2",
            Mamba3MimoConfig {
                mimo_rank: 2,
                ngroups: 2,
                d_state: 24,
                ..base
            },
        ),
        (
            "rank 1, full rope, MLP after",
            Mamba3MimoConfig {
                mimo_rank: 1,
                rope_fraction: 1.0,
                d_intermediate: 160,
                ..base
            },
        ),
        (
            "rank 3, wide head off the warp",
            Mamba3MimoConfig {
                d_model: 120,
                headdim: 40,
                mimo_rank: 3,
                ..base
            },
        ),
        (
            "released head, MLP after",
            Mamba3MimoConfig {
                headdim: 64,
                d_state: 128,
                d_intermediate: 128,
                ..base
            },
        ),
        (
            "rank 4, the gated output norm",
            Mamba3MimoConfig {
                is_outproj_norm: true,
                ..base
            },
        ),
        (
            "rank 4, LayerNorm block norms, MLP after",
            Mamba3MimoConfig {
                block_norm: mamba_rs::BlockNorm::Layer,
                d_intermediate: 128,
                ..base
            },
        ),
    ]
}

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
            (&mut lw.norm_gate_weight, 10, 1.0, 0.3),
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

/// Every tensor of `w` with its name, in a fixed order.
fn named(w: &Mamba3MimoWeights) -> Vec<(String, &[f32])> {
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
            ("dt_bias", &lw.dt_bias),
            ("b_norm", &lw.b_norm_weight),
            ("c_norm", &lw.c_norm_weight),
            ("b_bias", &lw.b_bias),
            ("c_bias", &lw.c_bias),
            ("d", &lw.d_param),
            ("mimo_x", &lw.mimo_x),
            ("mimo_z", &lw.mimo_z),
            ("mimo_o", &lw.mimo_o),
            ("norm_gate", &lw.norm_gate_weight),
            ("out_proj_w", &lw.out_proj_w),
        ] {
            out.push((format!("layer{l}.{name}"), v.as_slice()));
        }
    }
    for (l, m) in w.mlp.iter().enumerate() {
        out.push((format!("mlp{l}.norm"), m.norm_weight.as_slice()));
        out.push((format!("mlp{l}.norm_bias"), m.norm_bias.as_slice()));
        out.push((format!("mlp{l}.fc1"), m.fc1_w.as_slice()));
        out.push((format!("mlp{l}.fc2"), m.fc2_w.as_slice()));
    }
    out
}

/// The CPU trainer over each sequence from zero states, its gradients
/// summed over the sequences: (rows, gradients, input gradient).
fn cpu_reference(
    cfg: &Mamba3MimoConfig,
    w: &Mamba3MimoWeights,
    (inputs, d_out): (&[f32], &[f32]),
    (input_dim, batch, seq): (usize, usize, usize),
) -> (Vec<f32>, Mamba3MimoWeights, Vec<f32>) {
    let dm = cfg.d_model;
    let mut rows = vec![0f32; batch * seq * dm];
    let mut d_in = vec![0f32; batch * seq * input_dim];
    let mut grads = w.zeros_like();
    let mut s = Mamba3MimoTrainScratch::default();
    for b in 0..batch {
        let mut tape = Mamba3MimoTape::new(cfg, seq, input_dim, 8);
        let mut st = Mamba3MimoState::zeros(cfg);
        let x = &inputs[b * seq * input_dim..(b + 1) * seq * input_dim];
        forward_mamba3_mimo_train(
            &mut rows[b * seq * dm..(b + 1) * seq * dm],
            x,
            w,
            &mut st.layers,
            &mut tape,
            (cfg, &mut s),
        )
        .unwrap();
        let mut d = d_out[b * seq * dm..(b + 1) * seq * dm].to_vec();
        backward_mamba3_mimo_train(
            &mut d,
            &mut d_in[b * seq * input_dim..(b + 1) * seq * input_dim],
            (w, &mut grads),
            &tape,
            (cfg, &mut s),
        )
        .unwrap();
    }
    (rows, grads, d_in)
}

fn worst_gradient(got: &Mamba3MimoWeights, want: &Mamba3MimoWeights) -> (f64, String) {
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

// The GPU's exp2f/log1pf/sincosf/tanhf and the CPU's fast exp differ in the
// last bits and the reductions run in other orders; the bounds hold about
// ten times what these shapes measure, and a wrong index or term fails by
// orders of magnitude. The half lanes round the stored rows.
#[test]
fn m3_mimo_gpu_train_matches_the_cpu_trainer() {
    // Two chunks on the chunked path, the second partial.
    let (input_dim, batch, seq) = (40usize, 2usize, 100usize);
    for mode in [ScanMode::Auto, ScanMode::Sequential] {
        for (dtype, fwd_tol, grad_tol) in [
            (WeightDtype::F32, 5e-5, 5e-4),
            (WeightDtype::Tf32, 3e-3, 3e-2),
            (WeightDtype::F16, 5e-3, 5e-2),
            (WeightDtype::Bf16, 3e-2, 2e-1),
        ] {
            for (label, base) in cases() {
                let cfg = Mamba3MimoConfig {
                    scan_mode: mode,
                    ..base
                };
                cfg.validate().unwrap();
                let w = drawn(&cfg, input_dim, 11);
                let dm = cfg.d_model;
                let inputs = det(batch * seq * input_dim, 500, 1.0);
                let d_out = det(batch * seq * dm, 501, 1.0);
                let (want_rows, want_grads, want_d_in) =
                    cpu_reference(&cfg, &w, (&inputs, &d_out), (input_dim, batch, seq));
                let mut tr = Mamba3MimoGpuTrainer::new_with_stride(
                    &w,
                    cfg,
                    TrainSessionCfg::new(input_dim, batch, seq),
                    GpuOptions::default().dtype(dtype),
                    8,
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
                    "{mode:?} {dtype:?} {label}: a repeated backward moved"
                );
                let fwd = rel_l2(&rows, &want_rows);
                let din = rel_l2(&d_in, &want_d_in);
                let worst = worst_gradient(&grads, &want_grads);
                eprintln!(
                    "{mode:?} {dtype:?} {label}: forward {fwd:.2e}, input gradient {din:.2e}, worst \
                 weight gradient {:.2e} ({})",
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
}

/// The sequential path's gradients are the same bits for every checkpoint
/// stride, and a repeated backward gives the same bits on both paths.
#[test]
fn m3_mimo_gpu_train_is_stride_free_and_repeatable() {
    let (input_dim, batch, seq) = (40usize, 2usize, 29usize);
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        for (label, base) in cases() {
            let cfg = Mamba3MimoConfig {
                scan_mode: ScanMode::Sequential,
                ..base
            };
            let w = drawn(&cfg, input_dim, 21);
            let dm = cfg.d_model;
            let inputs = det(batch * seq * input_dim, 700, 1.0);
            let d_out = det(batch * seq * dm, 701, 1.0);
            let run = |stride: usize| -> (Vec<f32>, Mamba3MimoWeights) {
                let mut tr = Mamba3MimoGpuTrainer::new_with_stride(
                    &w,
                    cfg,
                    TrainSessionCfg::new(input_dim, batch, seq),
                    GpuOptions::default().dtype(dtype),
                    stride,
                )
                .unwrap();
                let mut rows = vec![0f32; batch * seq * dm];
                tr.forward(&inputs, &mut rows).unwrap();
                tr.backward(&d_out, None).unwrap();
                let g = tr.grads().unwrap();
                tr.backward(&d_out, None).unwrap();
                assert!(
                    tr.grads().unwrap() == g,
                    "{dtype:?} {label}: a repeated backward moved"
                );
                (rows, g)
            };
            let (rows4, g4) = run(4);
            for stride in [1usize, 7, seq] {
                let (rows, g) = run(stride);
                assert_eq!(rows, rows4, "{dtype:?} {label}: rows at stride {stride}");
                assert!(
                    g == g4,
                    "{dtype:?} {label}: gradients moved at stride {stride}"
                );
            }
        }
    }
}

/// AdamW on L = 0.5 |out - target|^2 lowers the loss in every lane.
#[test]
fn m3_mimo_gpu_train_adamw_lowers_a_loss() {
    let (input_dim, batch, seq) = (40usize, 2usize, 24usize);
    let cfg = cases()[1].1;
    let w = drawn(&cfg, input_dim, 31);
    let dm = cfg.d_model;
    let inputs = det(batch * seq * input_dim, 900, 1.0);
    let target = det(batch * seq * dm, 901, 1.0);
    for dtype in [
        WeightDtype::F32,
        WeightDtype::Tf32,
        WeightDtype::Bf16,
        WeightDtype::F16,
    ] {
        let mut tr = Mamba3MimoGpuTrainer::new_with_stride(
            &w,
            cfg,
            TrainSessionCfg::new(input_dim, batch, seq),
            GpuOptions::default().dtype(dtype),
            8,
        )
        .unwrap();
        tr.adamw_mut().lr = 3e-3;
        let mut out = vec![0f32; batch * seq * dm];
        let (mut losses, mut applied) = (Vec::new(), 0usize);
        for _ in 0..12 {
            tr.reset_state().unwrap();
            tr.forward(&inputs, &mut out).unwrap();
            let d: Vec<f32> = out.iter().zip(&target).map(|(o, t)| o - t).collect();
            losses.push(d.iter().map(|v| 0.5 * f64::from(*v).powi(2)).sum::<f64>());
            tr.backward(&d, None).unwrap();
            applied += usize::from(tr.apply_adamw().unwrap());
        }
        eprintln!(
            "{dtype:?} AdamW losses: {:.4} -> {:.4}, {applied} of 12 steps applied",
            losses[0], losses[11]
        );
        // The f16 scale starts at 2^16 and backs off on the first overflows.
        assert!(applied >= 6, "{dtype:?}: only {applied} steps applied");
        assert!(
            losses[11] < 0.9 * losses[0],
            "{dtype:?}: loss {:.4} -> {:.4}",
            losses[0],
            losses[11]
        );
        let moved = tr.snapshot_master().unwrap();
        assert!(moved != w, "{dtype:?}: the weights did not move");
        // The GEMMs read half copies in the half lanes: after the steps they
        // must be the stepped weights, so the trainer's forward tracks a CPU
        // forward of the weights it reports.
        tr.reset_state().unwrap();
        tr.forward(&inputs, &mut out).unwrap();
        let (want, _, _) = cpu_reference(&cfg, &moved, (&inputs, &target), (input_dim, batch, seq));
        let drift = rel_l2(&out, &want);
        let tol = match dtype {
            WeightDtype::F32 => 5e-5,
            WeightDtype::Tf32 => 3e-3,
            WeightDtype::F16 => 5e-3,
            _ => 3e-2,
        };
        eprintln!("{dtype:?} after AdamW: forward vs CPU of the stepped weights {drift:.2e}");
        assert!(
            drift < tol,
            "{dtype:?}: forward after the steps {drift:.3e}"
        );
    }
}

/// A rank-16 head of 16 channels on both paths: the chunked backward's
/// local state gradient stages 65 KB (past the default 48 KB) and the
/// chunked scan takes a smaller key tile. Both paths still match the CPU
/// trainer, and a repeated backward gives the same bits.
#[test]
fn m3_mimo_gpu_train_fits_a_wide_rank() {
    let base = Mamba3MimoConfig {
        d_model: 64,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        mimo_rank: 16,
        ..Mamba3MimoConfig::default()
    };
    let (input_dim, batch, seq) = (40usize, 2usize, 100usize);
    for mode in [ScanMode::Auto, ScanMode::Sequential] {
        for (dtype, fwd_tol, grad_tol) in [
            (WeightDtype::F32, 5e-5, 5e-4),
            (WeightDtype::Bf16, 3e-2, 2e-1),
        ] {
            let cfg = Mamba3MimoConfig {
                scan_mode: mode,
                ..base
            };
            let w = drawn(&cfg, input_dim, 13);
            let dm = cfg.d_model;
            let inputs = det(batch * seq * input_dim, 520, 1.0);
            let d_out = det(batch * seq * dm, 521, 1.0);
            let (want_rows, want_grads, want_d_in) =
                cpu_reference(&cfg, &w, (&inputs, &d_out), (input_dim, batch, seq));
            let mut tr = Mamba3MimoGpuTrainer::new_with_stride(
                &w,
                cfg,
                TrainSessionCfg::new(input_dim, batch, seq),
                GpuOptions::default().dtype(dtype),
                8,
            )
            .unwrap();
            let mut rows = vec![0f32; batch * seq * dm];
            tr.forward(&inputs, &mut rows).unwrap();
            let mut d_in = vec![0f32; batch * seq * input_dim];
            tr.backward(&d_out, Some(&mut d_in)).unwrap();
            let grads = tr.grads().unwrap();
            tr.backward(&d_out, None).unwrap();
            assert!(
                tr.grads().unwrap() == grads,
                "{mode:?} {dtype:?}: a repeated backward moved"
            );
            let fwd = rel_l2(&rows, &want_rows);
            let din = rel_l2(&d_in, &want_d_in);
            let worst = worst_gradient(&grads, &want_grads);
            eprintln!(
                "{mode:?} {dtype:?} rank 16: forward {fwd:.2e}, input gradient {din:.2e}, worst \
                 weight gradient {:.2e} ({})",
                worst.0, worst.1
            );
            assert!(fwd < fwd_tol, "{mode:?} {dtype:?}: forward {fwd:.3e}");
            assert!(
                din < grad_tol,
                "{mode:?} {dtype:?}: input gradient {din:.3e}"
            );
            assert!(
                worst.0 < grad_tol,
                "{mode:?} {dtype:?}: gradient {} {:.3e}",
                worst.1,
                worst.0
            );
        }
    }
}

/// Heads whose shared-memory blocks do not end on a float4: the chunked
/// backward's token-tile kernels read dy, u and the keys as float4 rows
/// past a state of hd N floats (hd (N + 1) for the state terms), and the
/// split pair kernels w M and G past blocks of tr (P + 1 + N) and
/// tr (N + P + 2) floats. One channel of two states, three of six, and
/// on a 99 KB part, rank 9 over 128 states and 128 or 127 channels, whose
/// pair tiles of two tokens put 18 rows there, start those rows off 16
/// bytes unless the blocks before them are padded. In f32 each matches the
/// CPU trainer tensor by tensor; in bf16 the rows and the input gradient do
/// (a one-head layer's dt_bias gradient is one value summed over the
/// window with cancellation, which bf16 does not resolve against an f32
/// host). A repeated backward gives the same bits.
#[test]
fn m3_mimo_gpu_train_pads_every_float4_block() {
    let base = Mamba3MimoConfig {
        n_layers: 2,
        rope_fraction: 1.0,
        ..Mamba3MimoConfig::default()
    };
    let shapes = [
        (16, 2, 1, 2, 1),
        (24, 2, 3, 6, 2),
        (64, 2, 128, 128, 9),
        (127, 1, 127, 128, 9),
    ];
    let (input_dim, batch, seq) = (24usize, 2usize, 100usize);
    for (d_model, expand, headdim, d_state, mimo_rank) in shapes {
        for (dtype, fwd_tol, grad_tol) in [
            (WeightDtype::F32, 5e-5, 5e-4),
            (WeightDtype::Bf16, 3e-2, 2e-1),
        ] {
            let cfg = Mamba3MimoConfig {
                d_model,
                expand,
                headdim,
                d_state,
                mimo_rank,
                ..base
            };
            cfg.validate().unwrap();
            let label = format!("{dtype:?} P {headdim} N {d_state} R {mimo_rank}");
            let w = drawn(&cfg, input_dim, 17);
            let inputs = det(batch * seq * input_dim, 530, 1.0);
            let d_out = det(batch * seq * d_model, 531, 1.0);
            let (want_rows, want_grads, want_d_in) =
                cpu_reference(&cfg, &w, (&inputs, &d_out), (input_dim, batch, seq));
            let mut tr = Mamba3MimoGpuTrainer::new_with_stride(
                &w,
                cfg,
                TrainSessionCfg::new(input_dim, batch, seq),
                GpuOptions::default().dtype(dtype),
                8,
            )
            .unwrap();
            let mut rows = vec![0f32; batch * seq * d_model];
            tr.forward(&inputs, &mut rows).unwrap();
            let mut d_in = vec![0f32; batch * seq * input_dim];
            tr.backward(&d_out, Some(&mut d_in)).unwrap();
            let grads = tr.grads().unwrap();
            tr.backward(&d_out, None).unwrap();
            assert!(
                tr.grads().unwrap() == grads,
                "{label}: a repeated backward moved"
            );
            let fwd = rel_l2(&rows, &want_rows);
            let din = rel_l2(&d_in, &want_d_in);
            let worst = worst_gradient(&grads, &want_grads);
            eprintln!(
                "{label}: forward {fwd:.2e}, input gradient {din:.2e}, worst weight gradient \
                 {:.2e} ({})",
                worst.0, worst.1
            );
            assert!(fwd < fwd_tol, "{label}: forward {fwd:.3e}");
            assert!(din < grad_tol, "{label}: input gradient {din:.3e}");
            if dtype == WeightDtype::F32 {
                assert!(
                    worst.0 < grad_tol,
                    "{label}: gradient {} {:.3e}",
                    worst.1,
                    worst.0
                );
            }
        }
    }
}

/// Heads past what one block holds, against the CPU trainer on both
/// paths: two heads wider than a block, rank 16 over 256 states, whose
/// keys and state rows pass shared memory together, and two rank-16 heads
/// wider than a block under the output norm. The sequential path
/// runs the global backward and the forward in tiles of channels, the
/// chunked one its kernels in channel tiles.
#[test]
fn m3_mimo_gpu_train_wide_heads_match_the_cpu_trainer() {
    let base = Mamba3MimoConfig {
        d_model: 64,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    };
    let heads = [
        (
            "heads wider than a block",
            Mamba3MimoConfig {
                d_model: 640,
                expand: 4,
                headdim: 1280,
                mimo_rank: 2,
                ..base
            },
        ),
        (
            "rank 16 over 256 states",
            Mamba3MimoConfig {
                mimo_rank: 16,
                d_state: 256,
                headdim: 64,
                ..base
            },
        ),
        (
            "rank 16 heads wider than a block, the output norm",
            Mamba3MimoConfig {
                d_model: 640,
                expand: 4,
                headdim: 1280,
                mimo_rank: 16,
                is_outproj_norm: true,
                ..base
            },
        ),
    ];
    let (input_dim, batch, seq) = (40usize, 2usize, 100usize);
    for (label, base) in heads {
        for (mode, dtype, fwd_tol, grad_tol) in [
            (ScanMode::Sequential, WeightDtype::F32, 5e-5, 5e-4),
            (ScanMode::Sequential, WeightDtype::Bf16, 3e-2, 2e-1),
            (ScanMode::Auto, WeightDtype::F32, 5e-5, 5e-4),
        ] {
            let cfg = Mamba3MimoConfig {
                scan_mode: mode,
                ..base
            };
            cfg.validate().unwrap();
            let w = drawn(&cfg, input_dim, 17);
            let dm = cfg.d_model;
            let inputs = det(batch * seq * input_dim, 530, 1.0);
            let d_out = det(batch * seq * dm, 531, 1.0);
            let (want_rows, want_grads, want_d_in) =
                cpu_reference(&cfg, &w, (&inputs, &d_out), (input_dim, batch, seq));
            let mut tr = Mamba3MimoGpuTrainer::new_with_stride(
                &w,
                cfg,
                TrainSessionCfg::new(input_dim, batch, seq),
                GpuOptions::default().dtype(dtype),
                8,
            )
            .unwrap();
            let mut rows = vec![0f32; batch * seq * dm];
            tr.forward(&inputs, &mut rows).unwrap();
            let mut d_in = vec![0f32; batch * seq * input_dim];
            tr.backward(&d_out, Some(&mut d_in)).unwrap();
            let grads = tr.grads().unwrap();
            tr.backward(&d_out, None).unwrap();
            assert!(
                tr.grads().unwrap() == grads,
                "{mode:?} {dtype:?} {label}: a repeated backward moved"
            );
            let fwd = rel_l2(&rows, &want_rows);
            let din = rel_l2(&d_in, &want_d_in);
            let worst = worst_gradient(&grads, &want_grads);
            eprintln!(
                "{mode:?} {dtype:?} {label}: forward {fwd:.2e}, input gradient {din:.2e}, worst \
                 weight gradient {:.2e} ({})",
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

/// Two windows in a row without a reset, on both paths: the second starts
/// from the states the first left (the SSM state, the previous keys and
/// input, the angles), so the entering lead and the carried state run on
/// real values; three chunks put a middle chunk in the state passing. The
/// second case clamps A (a_floor 0.5) and widens trap to about +-10. The
/// GPU matches the CPU trainer carrying the same state.
#[test]
fn m3_mimo_gpu_train_continues_a_carried_state() {
    let (input_dim, batch, seq) = (40usize, 2usize, 150usize);
    let all = cases();
    for mode in [ScanMode::Auto, ScanMode::Sequential] {
        for (label, base, saturate) in [(all[0].0, all[0].1, false), (all[1].0, all[1].1, true)] {
            let cfg = Mamba3MimoConfig {
                scan_mode: mode,
                a_floor: if saturate { 0.5 } else { base.a_floor },
                ..base
            };
            let mut w = drawn(&cfg, input_dim, 17);
            if saturate {
                let (out, nh) = (cfg.in_proj_out_dim(), cfg.nheads());
                let trap0 = 2 * cfg.d_inner() + 2 * cfg.bc_width() + 2 * nh;
                for lw in &mut w.layers {
                    for row in lw.in_proj_w.chunks_exact_mut(out) {
                        for v in &mut row[trap0..trap0 + nh] {
                            *v *= 25.0;
                        }
                    }
                }
            }
            let dm = cfg.d_model;
            let x1 = det(batch * seq * input_dim, 600, 1.0);
            let x2 = det(batch * seq * input_dim, 601, 1.0);
            let d_out = det(batch * seq * dm, 602, 1.0);

            let mut want_rows = vec![0f32; batch * seq * dm];
            let mut want_d_in = vec![0f32; batch * seq * input_dim];
            let mut want_grads = w.zeros_like();
            let mut s = Mamba3MimoTrainScratch::default();
            for b in 0..batch {
                let mut st = Mamba3MimoState::zeros(&cfg);
                let mut first = Mamba3MimoTape::new(&cfg, seq, input_dim, 8);
                let mut rows1 = vec![0f32; seq * dm];
                let span = |v: &[f32], n: usize| v[b * seq * n..(b + 1) * seq * n].to_vec();
                forward_mamba3_mimo_train(
                    &mut rows1,
                    &span(&x1, input_dim),
                    &w,
                    &mut st.layers,
                    &mut first,
                    (&cfg, &mut s),
                )
                .unwrap();
                let mut tape = Mamba3MimoTape::new(&cfg, seq, input_dim, 8);
                forward_mamba3_mimo_train(
                    &mut want_rows[b * seq * dm..(b + 1) * seq * dm],
                    &span(&x2, input_dim),
                    &w,
                    &mut st.layers,
                    &mut tape,
                    (&cfg, &mut s),
                )
                .unwrap();
                let mut d = span(&d_out, dm);
                backward_mamba3_mimo_train(
                    &mut d,
                    &mut want_d_in[b * seq * input_dim..(b + 1) * seq * input_dim],
                    (&w, &mut want_grads),
                    &tape,
                    (&cfg, &mut s),
                )
                .unwrap();
            }

            let mut tr = Mamba3MimoGpuTrainer::new_with_stride(
                &w,
                cfg,
                TrainSessionCfg::new(input_dim, batch, seq),
                GpuOptions::default().dtype(WeightDtype::F32),
                8,
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
                "{mode:?} {label}{}: carried window forward {fwd:.2e}, input gradient {din:.2e}, \
                 worst weight gradient {:.2e} ({})",
                if saturate {
                    ", clamped A, wide trap"
                } else {
                    ""
                },
                worst.0,
                worst.1
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
fn m3_mimo_gpu_train_identity_input() {
    let base = cases()[1].1;
    let (batch, seq) = (2usize, 100usize);
    let input_dim = base.d_model;
    for mode in [ScanMode::Auto, ScanMode::Sequential] {
        for (dtype, fwd_tol, grad_tol) in [
            (WeightDtype::F32, 5e-5, 5e-4),
            (WeightDtype::Bf16, 3e-2, 2e-1),
        ] {
            let cfg = Mamba3MimoConfig {
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
            let mut tr = Mamba3MimoGpuTrainer::new_with_stride(
                &w,
                cfg,
                TrainSessionCfg::new(input_dim, batch, seq),
                GpuOptions::default().dtype(dtype),
                8,
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
fn m3_mimo_gpu_train_f16_overflow_skips_the_step() {
    let (input_dim, batch, seq) = (40usize, 2usize, 70usize);
    let cfg = cases()[0].1;
    let w = drawn(&cfg, input_dim, 33);
    let dm = cfg.d_model;
    let inputs = det(batch * seq * input_dim, 910, 1.0);
    let d_out = det(batch * seq * dm, 911, 1.0);
    let mut tr = Mamba3MimoGpuTrainer::new_with_stride(
        &w,
        cfg,
        TrainSessionCfg::new(input_dim, batch, seq),
        GpuOptions::default().dtype(WeightDtype::F16),
        8,
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
