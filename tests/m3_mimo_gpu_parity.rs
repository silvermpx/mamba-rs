//! Mamba-3 MIMO on the GPU against the CPU: decode and windows over ranks,
//! groups, rope fractions, the MLP after the mixer and head widths that are
//! not a whole number of warps; batch rows against the same sequence alone;
//! the bit-exact seams (a decode step continues a window, and split windows
//! give the whole window); and the half and tf32 storage lanes.
#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::cpu::prefill::PrefillMode;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_mimo::gpu::Mamba3MimoGpuEngine;
use mamba_rs::mamba3_mimo::{
    Mamba3MimoConfig, Mamba3MimoPrefillScratch, Mamba3MimoState, Mamba3MimoStepScratch,
    Mamba3MimoWeights, forward_mamba3_mimo_backbone_prefill_mode, mamba3_mimo_step,
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

// The GPU's exp2f/log1pf/sincosf and the CPU's portable functions differ in
// the last bits; over a few dozen steps of a small random model that stays
// well under 2e-4, and a wrong index or order fails by orders of magnitude.
const TOL: f64 = 2e-4;

fn cases() -> Vec<(&'static str, Mamba3MimoConfig)> {
    let base = Mamba3MimoConfig {
        d_model: 64,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    };
    vec![
        ("rank 4", base),
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
            "released head",
            Mamba3MimoConfig {
                headdim: 64,
                d_state: 128,
                d_intermediate: 128,
                ..base
            },
        ),
    ]
}

/// Weights with every small parameter drawn away from its init.
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

fn engine(
    device: &GpuDevice,
    w: &Mamba3MimoWeights,
    (cfg, input_dim, batch): (Mamba3MimoConfig, usize, usize),
    dtype: WeightDtype,
) -> Mamba3MimoGpuEngine {
    Mamba3MimoGpuEngine::new(device, w, (cfg, input_dim, batch), dtype).unwrap()
}

#[test]
fn m3_mimo_gpu_decode_matches_cpu() {
    let device = GpuDevice::new(0).unwrap();
    let input_dim = 40usize;
    for (label, cfg) in cases() {
        cfg.validate().unwrap();
        let w = drawn(&cfg, input_dim, 13);
        let dm = cfg.d_model;
        let e = engine(&device, &w, (cfg, input_dim, 1), WeightDtype::F32);
        let mut gst = e.alloc_state().unwrap();
        let mut win = e.alloc_window(1).unwrap();
        let mut cst = Mamba3MimoState::zeros(&cfg);
        let mut csc = Mamba3MimoStepScratch::new(&cfg);
        let (mut cpu, mut gpu) = (vec![0f32; dm], vec![0f32; dm]);
        let mut worst = 0f64;
        for t in 0..24u64 {
            let x = det(input_dim, 200 + t, 1.0);
            mamba3_mimo_step(&mut cpu, &x, &mut csc, &w, &mut cst.layers, &cfg);
            e.run(&x, &mut gpu, &mut gst, &mut win).unwrap();
            worst = worst.max(rel_l2(&gpu, &cpu));
        }
        eprintln!("{label}: GPU decode vs CPU rel_l2 {worst:.2e}");
        assert!(worst < TOL, "{label}: rel_l2 {worst:.3e}");
    }
}

/// A window of two sequences against the CPU prefill of each, then decode
/// steps continuing both against the CPU chain.
#[test]
fn m3_mimo_gpu_window_matches_cpu() {
    let device = GpuDevice::new(0).unwrap();
    let input_dim = 40usize;
    let (batch, seq) = (2usize, 45usize);
    for (label, cfg) in cases() {
        let w = drawn(&cfg, input_dim, 19);
        let dm = cfg.d_model;
        let inputs = det(batch * seq * input_dim, 600, 1.0);
        let e = engine(&device, &w, (cfg, input_dim, batch), WeightDtype::F32);
        let mut gst = e.alloc_state().unwrap();
        let mut win = e.alloc_window(seq).unwrap();
        let mut rows = vec![0f32; batch * seq * dm];
        e.run(&inputs, &mut rows, &mut gst, &mut win).unwrap();
        let mut worst = 0f64;
        let mut cpu_states = Vec::new();
        for b in 0..batch {
            let mut st = Mamba3MimoState::zeros(&cfg);
            let mut ps = Mamba3MimoPrefillScratch::new(&cfg);
            let mut want = vec![0f32; seq * dm];
            forward_mamba3_mimo_backbone_prefill_mode(
                &mut want,
                &inputs[b * seq * input_dim..(b + 1) * seq * input_dim],
                &w,
                &mut st.layers,
                &mut ps,
                (&cfg, seq, PrefillMode::Single),
            );
            worst = worst.max(rel_l2(&rows[b * seq * dm..(b + 1) * seq * dm], &want));
            cpu_states.push(st);
        }
        let mut step_win = e.alloc_window(1).unwrap();
        let mut csc = Mamba3MimoStepScratch::new(&cfg);
        let mut g = vec![0f32; batch * dm];
        let mut c = vec![0f32; dm];
        for t in 0..6u64 {
            let xs = det(batch * input_dim, 700 + t, 1.0);
            e.run(&xs, &mut g, &mut gst, &mut step_win).unwrap();
            for (b, st) in cpu_states.iter_mut().enumerate() {
                let x = &xs[b * input_dim..(b + 1) * input_dim];
                mamba3_mimo_step(&mut c, x, &mut csc, &w, &mut st.layers, &cfg);
                worst = worst.max(rel_l2(&g[b * dm..(b + 1) * dm], &c));
            }
        }
        eprintln!("{label}: GPU window + decode vs CPU rel_l2 {worst:.2e}");
        assert!(worst < TOL, "{label}: rel_l2 {worst:.3e}");
    }
}

/// A row of a batch gets the same bits as that sequence alone, and a decode
/// step after a window gives the bits of one window over both; a window
/// split in two gives the whole window's bits.
#[test]
fn m3_mimo_gpu_seams_and_batch_rows_are_bit_exact() {
    let device = GpuDevice::new(0).unwrap();
    let input_dim = 40usize;
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        for (label, cfg) in cases() {
            let w = drawn(&cfg, input_dim, 17);
            let dm = cfg.d_model;
            let (batch, seq) = (3usize, 33usize);
            let inputs = det(batch * (seq + 1) * input_dim, 800, 1.0);
            // The inputs of sequence b, positions [a, z).
            let part = |b: usize, a: usize, z: usize| -> Vec<f32> {
                let base = b * (seq + 1) * input_dim;
                inputs[base + a * input_dim..base + z * input_dim].to_vec()
            };
            let gather = |a: usize, z: usize| -> Vec<f32> {
                (0..batch).flat_map(|b| part(b, a, z)).collect()
            };

            let e = engine(&device, &w, (cfg, input_dim, batch), dtype);
            let mut whole_st = e.alloc_state().unwrap();
            let mut whole = vec![0f32; batch * (seq + 1) * dm];
            let mut w_all = e.alloc_window(seq + 1).unwrap();
            e.run(&gather(0, seq + 1), &mut whole, &mut whole_st, &mut w_all)
                .unwrap();

            let mut st = e.alloc_state().unwrap();
            let mut head = vec![0f32; batch * seq * dm];
            let mut w_seq = e.alloc_window(seq).unwrap();
            e.run(&gather(0, seq), &mut head, &mut st, &mut w_seq)
                .unwrap();
            let mut last = vec![0f32; batch * dm];
            let mut w_one = e.alloc_window(1).unwrap();
            e.run(&gather(seq, seq + 1), &mut last, &mut st, &mut w_one)
                .unwrap();
            for b in 0..batch {
                let at = |t: usize| (b * (seq + 1) + t) * dm;
                assert_eq!(
                    &head[b * seq * dm..(b + 1) * seq * dm],
                    &whole[at(0)..at(seq)],
                    "{dtype:?} {label}: window rows of sequence {b}"
                );
                assert_eq!(
                    &last[b * dm..(b + 1) * dm],
                    &whole[at(seq)..at(seq + 1)],
                    "{dtype:?} {label}: decode after the window, sequence {b}"
                );
            }

            let split = 14usize;
            let mut sst = e.alloc_state().unwrap();
            let mut a_rows = vec![0f32; batch * split * dm];
            let mut b_rows = vec![0f32; batch * (seq + 1 - split) * dm];
            let mut wa = e.alloc_window(split).unwrap();
            let mut wb = e.alloc_window(seq + 1 - split).unwrap();
            e.run(&gather(0, split), &mut a_rows, &mut sst, &mut wa)
                .unwrap();
            e.run(&gather(split, seq + 1), &mut b_rows, &mut sst, &mut wb)
                .unwrap();
            let tail = seq + 1 - split;
            for b in 0..batch {
                let at = |t: usize| (b * (seq + 1) + t) * dm;
                assert_eq!(
                    &b_rows[b * tail * dm..(b + 1) * tail * dm],
                    &whole[at(split)..at(seq + 1)],
                    "{dtype:?} {label}: split window, sequence {b}"
                );
            }

            let single = engine(&device, &w, (cfg, input_dim, 1), dtype);
            let mut one_st = single.alloc_state().unwrap();
            let mut one = vec![0f32; (seq + 1) * dm];
            let mut w1 = single.alloc_window(seq + 1).unwrap();
            single
                .run(&part(1, 0, seq + 1), &mut one, &mut one_st, &mut w1)
                .unwrap();
            assert_eq!(
                &one[..],
                &whole[(seq + 1) * dm..2 * (seq + 1) * dm],
                "{dtype:?} {label}: batch row 1 against the sequence alone"
            );
        }
    }
}

/// tf32, f16 and bf16 storage against the f32 CPU over a window and decode.
#[test]
fn m3_mimo_gpu_typed_lanes_track_cpu() {
    let device = GpuDevice::new(0).unwrap();
    let input_dim = 40usize;
    let seq = 50usize;
    for (dtype, tol) in [
        (WeightDtype::Tf32, 3e-3),
        (WeightDtype::F16, 5e-3),
        (WeightDtype::Bf16, 3e-2),
    ] {
        for (label, cfg) in cases() {
            let w = drawn(&cfg, input_dim, 23);
            let dm = cfg.d_model;
            let inputs = det(seq * input_dim, 900, 1.0);
            let mut st = Mamba3MimoState::zeros(&cfg);
            let mut ps = Mamba3MimoPrefillScratch::new(&cfg);
            let mut want = vec![0f32; seq * dm];
            forward_mamba3_mimo_backbone_prefill_mode(
                &mut want,
                &inputs,
                &w,
                &mut st.layers,
                &mut ps,
                (&cfg, seq, PrefillMode::Single),
            );
            let e = engine(&device, &w, (cfg, input_dim, 1), dtype);
            let mut gst = e.alloc_state().unwrap();
            let mut win = e.alloc_window(seq).unwrap();
            let mut rows = vec![0f32; seq * dm];
            e.run(&inputs, &mut rows, &mut gst, &mut win).unwrap();
            let mut worst = rel_l2(&rows, &want);
            let mut one = e.alloc_window(1).unwrap();
            let mut csc = Mamba3MimoStepScratch::new(&cfg);
            let (mut g, mut c) = (vec![0f32; dm], vec![0f32; dm]);
            for t in 0..6u64 {
                let x = det(input_dim, 950 + t, 1.0);
                e.run(&x, &mut g, &mut gst, &mut one).unwrap();
                mamba3_mimo_step(&mut c, &x, &mut csc, &w, &mut st.layers, &cfg);
                worst = worst.max(rel_l2(&g, &c));
            }
            eprintln!("{dtype:?} {label}: GPU vs CPU rel_l2 {worst:.2e}");
            assert!(
                worst < tol,
                "{dtype:?} {label}: rel_l2 {worst:.3e} over {tol:.0e}"
            );
        }
    }
}

/// The gated output norm is refused rather than run wrong.
#[test]
fn m3_mimo_gpu_refuses_the_output_norm() {
    let device = GpuDevice::new(0).unwrap();
    let cfg = Mamba3MimoConfig {
        is_outproj_norm: true,
        ..cases()[0].1
    };
    let w = drawn(&cfg, 40, 3);
    let err = Mamba3MimoGpuEngine::new(&device, &w, (cfg, 40, 1), WeightDtype::F32)
        .err()
        .unwrap();
    assert!(err.contains("output norm"), "{err}");
}
