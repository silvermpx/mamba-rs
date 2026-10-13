//! Mamba-1 on the GPU against the CPU at inner widths that are not a
//! multiple of four, odd ones too, and at conv widths from 2 to 64: a
//! prompt through two prefills and decode steps after them, and a training
//! window's rows and every gradient, in every storage precision, both scan
//! modes and the cuBLAS modes. The parallel backward folds four channels at
//! a time, so the odd inner widths take its ungrouped kernel and the flat
//! reducers; a conv wider than eight taps takes the kernels that read their
//! window from the rows, and one wider than 32 deepens the forward tiles.
#![cfg(feature = "cuda")]

use mamba_rs::MambaBackbone;
use mamba_rs::config::ScanMode;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::train::{BackwardOpts, TrainSessionCfg};
use mamba_rs::gpu::{GemmMode, GpuOptions};
use mamba_rs::mamba1::gpu::inference::MambaGpuBackbone;
use mamba_rs::mamba1::gpu::trainer::MambaGpuTrainer;
use mamba_rs::mamba1::{MambaConfig, MambaWeights};
use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::backward::backward_mamba_backbone_batched;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};

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

fn cases() -> Vec<(&'static str, MambaConfig)> {
    let base = MambaConfig {
        d_model: 5,
        n_layers: 2,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
        dt_rank: None,
        proj_bias: false,
        conv_bias: true,
        d_intermediate: 0,
    };
    vec![
        ("d_inner 10", base),
        // FalconMamba's cases keep dt_rank past 1: a one-element dt part
        // normalizes to its sign, and the gradient of that norm, large only
        // where the element crosses zero, is past what a half precision
        // can follow against the f32 host.
        (
            "FalconMamba's dt, B and C norms",
            MambaConfig {
                d_model: 40,
                bcdt_rms_eps: Some(1e-6),
                ..base
            },
        ),
        (
            "FalconMamba's norms, odd d_inner 33 over 128 states",
            MambaConfig {
                d_model: 33,
                expand: 1,
                d_state: 128,
                bcdt_rms_eps: Some(1e-6),
                ..base
            },
        ),
        (
            "in_proj and out_proj biases at odd widths",
            MambaConfig {
                d_model: 7,
                expand: 3,
                proj_bias: true,
                ..base
            },
        ),
        (
            "no conv bias",
            MambaConfig {
                d_model: 8,
                conv_bias: false,
                ..base
            },
        ),
        (
            "dt_rank 6 against the default 1",
            MambaConfig {
                d_model: 10,
                dt_rank: Some(6),
                ..base
            },
        ),
        (
            "gated MLP 100 wide (128 hidden)",
            MambaConfig {
                d_model: 12,
                d_intermediate: 100,
                ..base
            },
        ),
        (
            "every variant with FalconMamba's norms",
            MambaConfig {
                d_model: 24,
                dt_rank: Some(4),
                proj_bias: true,
                conv_bias: false,
                d_intermediate: 100,
                bcdt_rms_eps: Some(1e-6),
                ..base
            },
        ),
        (
            "odd d_inner 33",
            MambaConfig {
                d_model: 33,
                expand: 1,
                d_state: 8,
                ..base
            },
        ),
        (
            "d_inner 18 over 128 states",
            MambaConfig {
                d_model: 9,
                d_state: 128,
                ..base
            },
        ),
        (
            "d_conv 2",
            MambaConfig {
                d_model: 8,
                d_conv: 2,
                ..base
            },
        ),
        (
            "d_conv 9, past the window the tiled conv holds",
            MambaConfig {
                d_model: 6,
                d_conv: 9,
                ..base
            },
        ),
        (
            "d_conv 40, deeper than a forward tile",
            MambaConfig {
                d_model: 12,
                d_conv: 40,
                ..base
            },
        ),
        (
            "d_conv 64 at an odd inner width",
            MambaConfig {
                d_model: 7,
                expand: 3,
                d_conv: 64,
                ..base
            },
        ),
    ]
}

/// Weights with the norms, the biases and D drawn away from their init.
fn drawn(cfg: &MambaConfig, input_dim: usize, seed: u64) -> MambaWeights {
    let mut w = MambaWeights::init(cfg, input_dim, seed);
    for (l, lw) in w.layers.iter_mut().enumerate() {
        let s = seed * 100 + l as u64 * 10;
        for (v, k, base, scale) in [
            (&mut lw.norm_weight, 1, 1.0, 0.2),
            (&mut lw.conv1d_bias, 2, 0.0, 0.3),
            (&mut lw.d_param, 3, 1.0, 0.5),
            (&mut lw.in_proj_b, 4, 0.0, 0.3),
            (&mut lw.out_proj_b, 5, 0.0, 0.3),
        ] {
            let d = det(v.len(), s + k, scale);
            for (x, dv) in v.iter_mut().zip(d) {
                *x = base + dv;
            }
        }
    }
    let d = det(w.norm_f_weight.len(), seed + 7, 0.2);
    for (x, dv) in w.norm_f_weight.iter_mut().zip(d) {
        *x = 1.0 + dv;
    }
    for (l, m) in w.mlp.iter_mut().enumerate() {
        let d = det(m.norm_weight.len(), seed * 100 + l as u64 * 10 + 6, 0.2);
        for (x, dv) in m.norm_weight.iter_mut().zip(d) {
            *x = 1.0 + dv;
        }
    }
    w
}

// The GPU's exp2f/log1pf and the CPU's fast exp differ in the last bits;
// over a prompt and a few steps of a small random model that stays well
// under these bounds, and a wrong index or order fails by orders of
// magnitude. The half lanes round the stored weights and rows.
const LANES: [(WeightDtype, f64); 4] = [
    (WeightDtype::F32, 2e-4),
    (WeightDtype::Tf32, 3e-3),
    (WeightDtype::F16, 5e-3),
    (WeightDtype::Bf16, 3e-2),
];

/// A prompt of two sequences through two prefills, the second from the
/// state the first left, then decode steps, against the CPU walking every
/// token of each sequence one by one. Both prefills are shorter than the
/// widest conv, so its carry-out keeps part of the window it came in with.
#[test]
fn m1_gpu_prefill_and_decode_match_cpu_at_any_width() {
    let (input_dim, batch, seq, split) = (12usize, 2usize, 37usize, 17usize);
    for (label, base) in cases() {
        for scan_mode in [ScanMode::Sequential, ScanMode::Parallel] {
            let cfg = MambaConfig { scan_mode, ..base };
            cfg.validate().unwrap();
            let dm = cfg.d_model;
            let w = drawn(&cfg, input_dim, 11);
            let cpu = MambaBackbone::new(w.clone(), cfg, input_dim).unwrap();
            let prompt = det(batch * seq * input_dim, 300, 1.0);
            let steps: Vec<Vec<f32>> = (0..6)
                .map(|t| det(batch * input_dim, 400 + t, 1.0))
                .collect();
            // The prompt rows of every sequence over [a, z).
            let part = |a: usize, z: usize| -> Vec<f32> {
                (0..batch)
                    .flat_map(|b| {
                        prompt[(b * seq + a) * input_dim..(b * seq + z) * input_dim]
                            .iter()
                            .copied()
                    })
                    .collect()
            };
            // The CPU's last row of each prefill and its decode rows, per
            // sequence.
            let mut want = vec![vec![0f32; batch * dm]; steps.len() + 2];
            for b in 0..batch {
                let (mut st, mut sc) = (cpu.alloc_state(), cpu.alloc_scratch());
                let mut c = vec![0f32; dm];
                for t in 0..seq {
                    let at = (b * seq + t) * input_dim;
                    cpu.forward_step(&prompt[at..at + input_dim], &mut c, &mut st, &mut sc);
                    if t + 1 == split {
                        want[0][b * dm..(b + 1) * dm].copy_from_slice(&c);
                    }
                }
                want[1][b * dm..(b + 1) * dm].copy_from_slice(&c);
                for (s, x) in steps.iter().enumerate() {
                    let x = &x[b * input_dim..(b + 1) * input_dim];
                    cpu.forward_step(x, &mut c, &mut st, &mut sc);
                    want[s + 2][b * dm..(b + 1) * dm].copy_from_slice(&c);
                }
            }
            let mut runs: Vec<(WeightDtype, GemmMode, f64)> = LANES
                .iter()
                .map(|&(dtype, tol)| (dtype, GemmMode::Deterministic, tol))
                .collect();
            for mode in [GemmMode::CublasFast, GemmMode::CublasPedantic] {
                runs.push((WeightDtype::F32, mode, 3e-3));
                runs.push((WeightDtype::Bf16, mode, 3e-2));
            }
            for (dtype, mode, tol) in runs {
                let opts = GpuOptions::default().dtype(dtype).mode(mode);
                let mut gpu = MambaGpuBackbone::new(&w, cfg, input_dim, batch, opts).unwrap();
                let mut g = vec![0f32; batch * dm];
                let mut worst = 0f64;
                for (i, (a, z)) in [(0, split), (split, seq)].into_iter().enumerate() {
                    gpu.prefill(&part(a, z), z - a).unwrap();
                    gpu.download_temporal(&mut g).unwrap();
                    worst = worst.max(rel_l2(&g, &want[i]));
                }
                for (s, x) in steps.iter().enumerate() {
                    gpu.step(x, &mut g).unwrap();
                    worst = worst.max(rel_l2(&g, &want[s + 2]));
                }
                eprintln!(
                    "{label} {scan_mode:?} {dtype:?} {mode:?}: prefill + decode vs CPU {worst:.2e}"
                );
                assert!(
                    worst < tol,
                    "{label} {scan_mode:?} {dtype:?} {mode:?}: rel_l2 {worst:.3e} over {tol:.0e}"
                );
            }
        }
    }
}

/// Each reduced-precision backbone against the CPU backbone of the same
/// precision, which rounds the weights and the values between the kernels
/// where the GPU stores them, its prefill where the GPU's prefill does and
/// its decode where the GPU's decode does: a prompt's last row and the
/// steps after it land closer than the f32 CPU's.
#[test]
fn m1_gpu_typed_lanes_track_the_cpu_lane() {
    use mamba_rs::CpuOptions;
    use mamba_rs::mamba1::cpu::prefill::PrefillMode;
    let (input_dim, seq) = (12usize, 37usize);
    for dtype in [WeightDtype::Tf32, WeightDtype::F16, WeightDtype::Bf16] {
        for (label, cfg) in cases() {
            let dm = cfg.d_model;
            let w = drawn(&cfg, input_dim, 13);
            let prompt = det(seq * input_dim, 500, 1.0);
            let steps: Vec<Vec<f32>> = (0..6).map(|t| det(input_dim, 600 + t, 1.0)).collect();
            let opts = GpuOptions::default().dtype(dtype);
            let mut gpu = MambaGpuBackbone::new(&w, cfg, input_dim, 1, opts).unwrap();
            let mut got = vec![vec![0f32; dm]; steps.len() + 1];
            gpu.prefill(&prompt, seq).unwrap();
            gpu.download_temporal(&mut got[0]).unwrap();
            for (s, x) in steps.iter().enumerate() {
                gpu.step(x, &mut got[s + 1]).unwrap();
            }
            // The host in f32 and in the backbone's precision.
            let dist = |o: CpuOptions| {
                let cpu = MambaBackbone::new_with(w.clone(), cfg, input_dim, o).unwrap();
                let (mut st, mut sc) = (cpu.alloc_state(), cpu.alloc_scratch());
                let mut pf = cpu.alloc_prefill_scratch(seq);
                let mut rows = vec![0f32; seq * dm];
                cpu.forward_prefill(
                    &prompt,
                    &mut rows,
                    &mut st,
                    &mut pf,
                    seq,
                    PrefillMode::Single,
                );
                let mut worst = rel_l2(&got[0], &rows[(seq - 1) * dm..]);
                let mut c = vec![0f32; dm];
                for (s, x) in steps.iter().enumerate() {
                    cpu.forward_step(x, &mut c, &mut st, &mut sc);
                    worst = worst.max(rel_l2(&got[s + 1], &c));
                }
                worst
            };
            let f32_host = dist(CpuOptions::default());
            let lane_host = dist(CpuOptions::default().dtype(dtype));
            eprintln!(
                "{dtype:?} {label}: GPU vs CPU f32 {f32_host:.2e}, vs CPU {dtype:?} {lane_host:.2e}"
            );
            assert!(
                lane_host < f32_host,
                "{dtype:?} {label}: the CPU lane ({lane_host:.3e}) is no closer than f32 \
                 ({f32_host:.3e})"
            );
        }
    }
}

fn train_weights_from(w: &MambaWeights) -> TrainMambaWeights {
    TrainMambaWeights {
        input_proj_w: w.input_proj_w.clone(),
        input_proj_b: w.input_proj_b.clone(),
        layers: w
            .layers
            .iter()
            .map(|lw| TrainMambaLayerWeights {
                norm_weight: lw.norm_weight.clone(),
                in_proj_w: lw.in_proj_w.clone(),
                in_proj_b: lw.in_proj_b.clone(),
                conv1d_weight: lw.conv1d_weight.clone(),
                conv1d_bias: lw.conv1d_bias.clone(),
                x_proj_w: lw.x_proj_w.clone(),
                dt_proj_w: lw.dt_proj_w.clone(),
                dt_proj_b: lw.dt_proj_b.clone(),
                a_log: lw.a_log.clone(),
                d_param: lw.d_param.clone(),
                out_proj_w: lw.out_proj_w.clone(),
                out_proj_b: lw.out_proj_b.clone(),
            })
            .collect(),
        norm_f_weight: w.norm_f_weight.clone(),
        mlp: w.mlp.clone(),
    }
}

/// Every gradient tensor of `g` with its name, in the GPU arena's order.
fn named(g: &TrainMambaWeights) -> Vec<(String, &[f32])> {
    let mut out = vec![
        ("input_proj_w".to_string(), g.input_proj_w.as_slice()),
        ("input_proj_b".to_string(), g.input_proj_b.as_slice()),
    ];
    for (l, lw) in g.layers.iter().enumerate() {
        for (name, v) in [
            ("norm", &lw.norm_weight),
            ("in_proj_w", &lw.in_proj_w),
            ("in_proj_b", &lw.in_proj_b),
            ("conv1d_w", &lw.conv1d_weight),
            ("conv1d_b", &lw.conv1d_bias),
            ("x_proj_w", &lw.x_proj_w),
            ("dt_proj_w", &lw.dt_proj_w),
            ("dt_proj_b", &lw.dt_proj_b),
            ("a_log", &lw.a_log),
            ("d", &lw.d_param),
            ("out_proj_w", &lw.out_proj_w),
            ("out_proj_b", &lw.out_proj_b),
        ] {
            out.push((format!("layer{l}.{name}"), v.as_slice()));
        }
    }
    // The gated MLPs sit between the layers and norm_f in the arena, the
    // CPU trainer's order.
    for (l, m) in g.mlp.iter().enumerate() {
        for (name, v) in [
            ("mlp_norm", &m.norm_weight),
            ("mlp_fc1", &m.fc1_w),
            ("mlp_fc2", &m.fc2_w),
        ] {
            out.push((format!("layer{l}.{name}"), v.as_slice()));
        }
    }
    out.push(("norm_f".to_string(), g.norm_f_weight.as_slice()));
    out
}

/// The CPU trainer over each sequence from zero states: the rows, and the
/// gradients summed over the sequences in sequence order, flat in the GPU
/// arena's order, with the tensors' names and lengths.
fn cpu_reference(
    cfg: &MambaConfig,
    w: &MambaWeights,
    (inputs, d_out): (&[f32], &[f32]),
    (input_dim, batch, seq): (usize, usize, usize),
) -> (Vec<f32>, Vec<f32>, Vec<(String, usize)>) {
    let tw = train_weights_from(w);
    let dims = MambaDims::from_config(cfg, seq, input_dim);
    let (dm, di, ds, dc, nl) = (
        cfg.d_model,
        cfg.d_inner(),
        cfg.d_state,
        cfg.d_conv,
        cfg.n_layers,
    );
    let mut a_neg = vec![0f32; nl * di * ds];
    for (l, lw) in w.layers.iter().enumerate() {
        for (a, &v) in a_neg[l * di * ds..(l + 1) * di * ds]
            .iter_mut()
            .zip(&lw.a_log)
        {
            *a = -v.exp();
        }
    }
    let mut rows = vec![0f32; batch * seq * dm];
    let mut total: Vec<f32> = Vec::new();
    let mut tensors = Vec::new();
    for b in 0..batch {
        let mut acts = MambaBackboneFlat::zeros(dims);
        let mut fwd = PhaseScratch::zeros(&dims);
        let (mut conv, mut ssm) = (vec![0f32; nl * di * dc], vec![0f32; nl * di * ds]);
        let mut state = MambaRecurrentState {
            conv: &mut conv,
            ssm: &mut ssm,
            a_neg: &a_neg,
        };
        forward_mamba_backbone_batched(
            &mut rows[b * seq * dm..(b + 1) * seq * dm],
            &mut acts,
            &tw,
            &inputs[b * seq * input_dim..(b + 1) * seq * input_dim],
            &mut state,
            &mut fwd,
            &dims,
        );
        let mut grads = TrainMambaWeights::zeros_from_dims(&dims);
        let mut d_t = d_out[b * seq * dm..(b + 1) * seq * dm].to_vec();
        let mut bwd = BackwardPhaseScratch::zeros(&dims);
        backward_mamba_backbone_batched(&mut d_t, &mut grads, &acts, &tw, &a_neg, &mut bwd, &dims);
        let parts = named(&grads);
        let flat: Vec<f32> = parts.iter().flat_map(|(_, v)| v.iter().copied()).collect();
        if b == 0 {
            tensors = parts.iter().map(|(n, v)| (n.clone(), v.len())).collect();
            total = flat;
        } else {
            for (t, v) in total.iter_mut().zip(flat) {
                *t += v;
            }
        }
    }
    (rows, total, tensors)
}

/// The training window's rows and every gradient against the CPU trainer:
/// both scan modes, every storage precision. 150 steps take several
/// forward tiles at every conv width and two backward tiles.
#[test]
fn m1_gpu_train_matches_cpu_at_any_width() {
    let (input_dim, batch, seq) = (12usize, 2usize, 150usize);
    for (label, base) in cases() {
        for scan_mode in [ScanMode::Sequential, ScanMode::Parallel] {
            let cfg = MambaConfig { scan_mode, ..base };
            let dm = cfg.d_model;
            let w = drawn(&cfg, input_dim, 13);
            let inputs = det(batch * seq * input_dim, 500, 1.0);
            let d_out = det(batch * seq * dm, 501, 1.0);
            let (want_rows, want_grads, tensors) =
                cpu_reference(&cfg, &w, (&inputs, &d_out), (input_dim, batch, seq));
            for (dtype, fwd_tol, grad_tol) in [
                (WeightDtype::F32, 5e-5, 5e-4),
                (WeightDtype::Tf32, 3e-3, 3e-2),
                (WeightDtype::F16, 5e-3, 5e-2),
                (WeightDtype::Bf16, 3e-2, 2e-1),
            ] {
                let mut tr = MambaGpuTrainer::new(
                    &w,
                    cfg,
                    TrainSessionCfg::new(input_dim, batch, seq),
                    GpuOptions::default().dtype(dtype),
                )
                .unwrap();
                if dtype == WeightDtype::F16 {
                    tr.load_scaler_state(256.0, 0).unwrap();
                }
                let mut rows = vec![0f32; batch * seq * dm];
                tr.forward(&inputs, &mut rows).unwrap();
                // The step leaves the window's gradients in the arena.
                let m = tr.backward_step(&d_out, BackwardOpts::default()).unwrap();
                assert!(
                    m.overflow_skipped != Some(true),
                    "{label} {scan_mode:?} {dtype:?}: the step was skipped"
                );
                let stream = tr.ctx().stream.clone();
                let arena = tr.grad_arena().to_cpu(&stream).unwrap();
                assert_eq!(arena.len(), want_grads.len(), "{label}: arena length");
                let (mut at, mut worst) = (0usize, (0f64, String::new()));
                for (name, len) in &tensors {
                    let e = rel_l2(&arena[at..at + len], &want_grads[at..at + len]);
                    at += len;
                    if e > worst.0 {
                        worst = (e, name.clone());
                    }
                }
                let fwd = rel_l2(&rows, &want_rows);
                eprintln!(
                    "{label} {scan_mode:?} {dtype:?}: forward {fwd:.2e}, worst gradient {:.2e} ({})",
                    worst.0, worst.1
                );
                assert!(
                    fwd < fwd_tol,
                    "{label} {scan_mode:?} {dtype:?}: forward {fwd:.3e}"
                );
                assert!(
                    worst.0 < grad_tol,
                    "{label} {scan_mode:?} {dtype:?}: gradient {} {:.3e}",
                    worst.1,
                    worst.0
                );
            }
        }
    }
}
