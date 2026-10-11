//! Mamba-2 on the GPU against the CPU: decode and the chunked prefill over
//! every head width the kernels tile differently, groups, per-channel D, the
//! dt clamp, both norm orders, no norm and the mixer's inner MLP; batch rows
//! against the same sequence alone; and the prefill's bit-exact seams (a
//! decode step continues a prefill of whole chunks, and windows split at
//! chunk boundaries give the whole window).
#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::cpu::prefill::PrefillMode;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba2::gpu::Mamba2GpuEngine;
use mamba_rs::mamba2::{
    Mamba2Config, Mamba2PrefillScratch, Mamba2State, Mamba2StepScratch, Mamba2Weights,
    forward_mamba2_backbone_prefill_mode, mamba2_step,
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

// The GPU's exp2f/log1pf and the CPU's portable functions differ in the
// last bits; over 24 steps of a small random model that stays under 2e-4,
// and a wrong index or order fails by orders of magnitude.
const TOL: f64 = 2e-4;

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
                dt_limit: (0.01, 0.2),
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
            "official head",
            Mamba2Config {
                headdim: 64,
                d_state: 128,
                ..base
            },
        ),
        (
            "wide head, large state",
            Mamba2Config {
                d_model: 128,
                headdim: 80,
                d_ssm: 160,
                d_state: 256,
                ..base
            },
        ),
    ]
}

fn drawn(cfg: &Mamba2Config, input_dim: usize, seed: u64) -> Mamba2Weights {
    let mut w = Mamba2Weights::init(cfg, input_dim, seed);
    for (l, lw) in w.layers.iter_mut().enumerate() {
        let s = seed * 100 + l as u64 * 10;
        for (v, k, base, scale) in [
            (&mut lw.d_param, 1, 1.0, 0.5),
            (&mut lw.gate_norm_weight, 2, 1.0, 0.3),
            (&mut lw.in_proj_b, 3, 0.0, 0.1),
            (&mut lw.out_proj_b, 4, 0.0, 0.1),
        ] {
            let d = det(v.len(), s + k, scale);
            for (x, dv) in v.iter_mut().zip(d) {
                *x = base + dv;
            }
        }
    }
    w.input_proj_b = det(cfg.d_model, seed + 7, 0.1);
    w
}

#[test]
fn m2_gpu_decode_matches_cpu() {
    let device = GpuDevice::new(0).unwrap();
    let input_dim = 40usize;
    for (label, cfg) in cases() {
        cfg.validate().unwrap();
        let w = drawn(&cfg, input_dim, 13);
        let dm = cfg.d_model;
        let engine = Mamba2GpuEngine::new(&device, &w, cfg, input_dim, 1).unwrap();
        let mut gst = engine.alloc_state().unwrap();
        let mut gsc = engine.alloc_scratch().unwrap();
        let mut cst = Mamba2State::zeros(&cfg);
        let mut csc = Mamba2StepScratch::new(&cfg);
        let (mut cpu, mut gpu) = (vec![0f32; dm], vec![0f32; dm]);
        let mut worst = 0f64;
        for t in 0..24u64 {
            let x = det(input_dim, 200 + t, 1.0);
            mamba2_step(&mut cpu, &x, &mut csc, &w, &mut cst.layers, &cfg);
            engine.step(&x, &mut gpu, &mut gst, &mut gsc).unwrap();
            worst = worst.max(rel_l2(&gpu, &cpu));
        }
        eprintln!("{label}: GPU vs CPU rel_l2 {worst:.2e}");
        assert!(worst < TOL, "{label}: rel_l2 {worst:.3e}");
    }
}

/// A row of a batch gets the same bits as that sequence decoded alone.
#[test]
fn m2_gpu_batch_rows_match_single_bits() {
    let device = GpuDevice::new(0).unwrap();
    let input_dim = 40usize;
    let cfg = cases()[1].1;
    let w = drawn(&cfg, input_dim, 17);
    let dm = cfg.d_model;
    let batch = 5usize;
    let engine = Mamba2GpuEngine::new(&device, &w, cfg, input_dim, batch).unwrap();
    let single = Mamba2GpuEngine::new(&device, &w, cfg, input_dim, 1).unwrap();
    let mut bst = engine.alloc_state().unwrap();
    let mut bsc = engine.alloc_scratch().unwrap();
    let mut sst = single.alloc_state().unwrap();
    let mut ssc = single.alloc_scratch().unwrap();
    let mut out_b = vec![0f32; batch * dm];
    let mut out_s = vec![0f32; dm];
    for t in 0..8u64 {
        let xs = det(batch * input_dim, 300 + t, 1.0);
        engine.step(&xs, &mut out_b, &mut bst, &mut bsc).unwrap();
        // Sequence 2 alone.
        single
            .step(
                &xs[2 * input_dim..3 * input_dim],
                &mut out_s,
                &mut sst,
                &mut ssc,
            )
            .unwrap();
        assert_eq!(&out_b[2 * dm..3 * dm], &out_s[..], "step {t}");
    }
}

/// The prefill rows of two sequences against the CPU prefill, then decode
/// steps continuing both against the CPU chain.
#[test]
fn m2_gpu_prefill_matches_cpu() {
    let device = GpuDevice::new(0).unwrap();
    let input_dim = 40usize;
    let (batch, seq) = (2usize, 150usize);
    for (label, cfg) in cases() {
        let w = drawn(&cfg, input_dim, 19);
        let dm = cfg.d_model;
        let inputs = det(batch * seq * input_dim, 600, 1.0);
        let engine = Mamba2GpuEngine::new(&device, &w, cfg, input_dim, batch).unwrap();
        let mut gst = engine.alloc_state().unwrap();
        let mut pf = engine.alloc_prefill(seq).unwrap();
        let mut rows = vec![0f32; batch * seq * dm];
        engine
            .prefill(&inputs, &mut rows, &mut gst, &mut pf)
            .unwrap();
        let mut worst = 0f64;
        let mut cpu_states = Vec::new();
        for b in 0..batch {
            let mut st = Mamba2State::zeros(&cfg);
            let mut ps = Mamba2PrefillScratch::new();
            let mut want = vec![0f32; seq * dm];
            forward_mamba2_backbone_prefill_mode(
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
        let mut gsc = engine.alloc_scratch().unwrap();
        let mut csc = Mamba2StepScratch::new(&cfg);
        let mut gout = vec![0f32; batch * dm];
        let mut cout = vec![0f32; dm];
        for t in 0..6u64 {
            let xs = det(batch * input_dim, 700 + t, 1.0);
            engine.step(&xs, &mut gout, &mut gst, &mut gsc).unwrap();
            for (b, st) in cpu_states.iter_mut().enumerate() {
                mamba2_step(
                    &mut cout,
                    &xs[b * input_dim..(b + 1) * input_dim],
                    &mut csc,
                    &w,
                    &mut st.layers,
                    &cfg,
                );
                worst = worst.max(rel_l2(&gout[b * dm..(b + 1) * dm], &cout));
            }
        }
        eprintln!("{label}: GPU prefill and decode vs CPU rel_l2 {worst:.2e}");
        assert!(worst < TOL, "{label}: rel_l2 {worst:.3e}");
    }
}

/// On the GPU a decode step after a prefill of whole chunks gives the bits
/// of a prefill one token longer, and a window split at a chunk boundary
/// gives the whole window's rows and state.
#[test]
fn m2_gpu_prefill_seams_are_bit_exact() {
    let device = GpuDevice::new(0).unwrap();
    let input_dim = 40usize;
    for (label, base) in [cases()[1], cases()[4]] {
        // The chunked scan for every window, the 64-token ones included.
        let cfg = Mamba2Config {
            scan_mode: mamba_rs::config::ScanMode::Parallel,
            ..base
        };
        let w = drawn(&cfg, input_dim, 29);
        let dm = cfg.d_model;
        let engine = Mamba2GpuEngine::new(&device, &w, cfg, input_dim, 1).unwrap();
        let stream = &engine.ctx().stream;
        let inputs = det(129 * input_dim, 800, 1.0);
        let download = |b: &mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer| {
            let mut v = vec![0f32; b.len()];
            b.download(stream, &mut v).unwrap();
            stream.synchronize().unwrap();
            v
        };

        // The whole window of 129.
        let mut st_a = engine.alloc_state().unwrap();
        let mut pf129 = engine.alloc_prefill(129).unwrap();
        let mut rows_a = vec![0f32; 129 * dm];
        engine
            .prefill(&inputs, &mut rows_a, &mut st_a, &mut pf129)
            .unwrap();

        // 128 by prefill, the last by a decode step.
        let mut st_b = engine.alloc_state().unwrap();
        let mut pf128 = engine.alloc_prefill(128).unwrap();
        let mut rows_b = vec![0f32; 128 * dm];
        engine
            .prefill(
                &inputs[..128 * input_dim],
                &mut rows_b,
                &mut st_b,
                &mut pf128,
            )
            .unwrap();
        let mut sc = engine.alloc_scratch().unwrap();
        let mut last = vec![0f32; dm];
        engine
            .step(&inputs[128 * input_dim..], &mut last, &mut st_b, &mut sc)
            .unwrap();
        assert_eq!(&rows_b[..], &rows_a[..128 * dm], "{label}: first 128 rows");
        assert_eq!(
            &last[..],
            &rows_a[128 * dm..],
            "{label}: decode after prefill"
        );
        assert_eq!(
            download(&st_a.ssm),
            download(&st_b.ssm),
            "{label}: ssm state"
        );
        assert_eq!(
            download(&st_a.conv),
            download(&st_b.conv),
            "{label}: conv state"
        );

        // 64 then 65.
        let mut st_c = engine.alloc_state().unwrap();
        let mut pf64 = engine.alloc_prefill(64).unwrap();
        let mut pf65 = engine.alloc_prefill(65).unwrap();
        let mut rows_c = vec![0f32; 129 * dm];
        let (first, second) = rows_c.split_at_mut(64 * dm);
        engine
            .prefill(&inputs[..64 * input_dim], first, &mut st_c, &mut pf64)
            .unwrap();
        engine
            .prefill(&inputs[64 * input_dim..], second, &mut st_c, &mut pf65)
            .unwrap();
        assert_eq!(rows_c, rows_a, "{label}: split window rows");
        assert_eq!(
            download(&st_a.ssm),
            download(&st_c.ssm),
            "{label}: split ssm state"
        );
    }
}

/// The tf32, f16 and bf16 engines, prefill and then decode, against the
/// CPU. Their bounds follow the storage: tf32 rounds the GEMM inputs to 10
/// mantissa bits, f16 stores activations with 10 and bf16 with 7.
#[test]
fn m2_gpu_typed_lanes_track_cpu() {
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    let device = GpuDevice::new(0).unwrap();
    let input_dim = 40usize;
    let seq = 70usize;
    for (dtype, tol) in [
        (WeightDtype::Tf32, 3e-3),
        (WeightDtype::F16, 5e-3),
        (WeightDtype::Bf16, 3e-2),
    ] {
        for (label, cfg) in cases() {
            let w = drawn(&cfg, input_dim, 23);
            let dm = cfg.d_model;
            let inputs = det(seq * input_dim, 900, 1.0);
            let mut st = Mamba2State::zeros(&cfg);
            let mut ps = Mamba2PrefillScratch::new();
            let mut want = vec![0f32; seq * dm];
            forward_mamba2_backbone_prefill_mode(
                &mut want,
                &inputs,
                &w,
                &mut st.layers,
                &mut ps,
                (&cfg, seq, PrefillMode::Single),
            );
            let engine =
                Mamba2GpuEngine::new_with_dtype(&device, &w, (cfg, input_dim, 1), dtype).unwrap();
            let mut gst = engine.alloc_state().unwrap();
            let mut pf = engine.alloc_prefill(seq).unwrap();
            let mut rows = vec![0f32; seq * dm];
            engine
                .prefill(&inputs, &mut rows, &mut gst, &mut pf)
                .unwrap();
            let mut worst = rel_l2(&rows, &want);
            let mut gsc = engine.alloc_scratch().unwrap();
            let mut csc = Mamba2StepScratch::new(&cfg);
            let (mut g, mut c) = (vec![0f32; dm], vec![0f32; dm]);
            for t in 0..6u64 {
                let x = det(input_dim, 950 + t, 1.0);
                engine.step(&x, &mut g, &mut gst, &mut gsc).unwrap();
                mamba2_step(&mut c, &x, &mut csc, &w, &mut st.layers, &cfg);
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

/// The sequential prefill (ScanMode::Sequential) gives the bits of as many
/// decode steps, rows and states, and tracks the chunked prefill.
#[test]
fn m2_gpu_sequential_prefill_is_decode_steps() {
    use mamba_rs::config::ScanMode;
    let device = GpuDevice::new(0).unwrap();
    let input_dim = 40usize;
    let (batch, seq) = (2usize, 77usize);
    for (label, base) in cases() {
        let cfg = Mamba2Config {
            scan_mode: ScanMode::Sequential,
            ..base
        };
        let w = drawn(&cfg, input_dim, 29);
        let dm = cfg.d_model;
        let inputs = det(batch * seq * input_dim, 610, 1.0);
        let engine = Mamba2GpuEngine::new(&device, &w, cfg, input_dim, batch).unwrap();
        let mut st = engine.alloc_state().unwrap();
        let mut pf = engine.alloc_prefill(seq).unwrap();
        let mut rows = vec![0f32; batch * seq * dm];
        engine
            .prefill(&inputs, &mut rows, &mut st, &mut pf)
            .unwrap();

        let mut dst = engine.alloc_state().unwrap();
        let mut dsc = engine.alloc_scratch().unwrap();
        let mut out = vec![0f32; batch * dm];
        for t in 0..seq {
            let x: Vec<f32> = (0..batch)
                .flat_map(|b| {
                    let o = (b * seq + t) * input_dim;
                    inputs[o..o + input_dim].to_vec()
                })
                .collect();
            engine.step(&x, &mut out, &mut dst, &mut dsc).unwrap();
            for b in 0..batch {
                assert_eq!(
                    &rows[(b * seq + t) * dm..(b * seq + t + 1) * dm],
                    &out[b * dm..(b + 1) * dm],
                    "{label}: token {t} of sequence {b}"
                );
            }
        }
        let s = &engine.ctx().stream;
        assert_eq!(
            st.ssm.to_cpu(s).unwrap(),
            dst.ssm.to_cpu(s).unwrap(),
            "{label}: SSM states"
        );
        assert_eq!(
            st.conv.to_cpu(s).unwrap(),
            dst.conv.to_cpu(s).unwrap(),
            "{label}: conv states"
        );

        let chunked_cfg = Mamba2Config {
            scan_mode: ScanMode::Parallel,
            ..base
        };
        let chunked = Mamba2GpuEngine::new(&device, &w, chunked_cfg, input_dim, batch).unwrap();
        let mut cst = chunked.alloc_state().unwrap();
        let mut cpf = chunked.alloc_prefill(seq).unwrap();
        let mut crow = vec![0f32; batch * seq * dm];
        chunked
            .prefill(&inputs, &mut crow, &mut cst, &mut cpf)
            .unwrap();
        let e = rel_l2(&rows, &crow);
        eprintln!("{label}: sequential vs chunked prefill rel_l2 {e:.2e}");
        assert!(e < TOL, "{label}: rel_l2 {e:.3e}");
    }
}

/// A captured decode graph replays the eager step's bits in every
/// precision, continues a prefill window run between replays, and refuses a
/// state it was not captured with until the graph is cleared.
#[test]
fn m2_gpu_graph_replay_is_the_eager_step() {
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    let device = GpuDevice::new(0).unwrap();
    let input_dim = 40usize;
    let (batch, seq) = (3usize, 70usize);
    let cfg = cases()[1].1;
    let w = drawn(&cfg, input_dim, 29);
    let dm = cfg.d_model;
    let prompt = det(batch * seq * input_dim, 40, 1.0);
    for dtype in [
        WeightDtype::F32,
        WeightDtype::Tf32,
        WeightDtype::Bf16,
        WeightDtype::F16,
    ] {
        let new = || {
            Mamba2GpuEngine::new_with_dtype(&device, &w, (cfg, input_dim, batch), dtype).unwrap()
        };
        let (eager, mut graphed) = (new(), new());
        let (mut est, mut esc) = (eager.alloc_state().unwrap(), eager.alloc_scratch().unwrap());
        let (mut gst, mut gsc) = (
            graphed.alloc_state().unwrap(),
            graphed.alloc_scratch().unwrap(),
        );
        let mut epf = eager.alloc_prefill(seq).unwrap();
        let mut gpf = graphed.alloc_prefill(seq).unwrap();
        let (mut erows, mut grows) = (vec![0f32; batch * seq * dm], vec![0f32; batch * seq * dm]);
        let (mut eo, mut go) = (vec![0f32; batch * dm], vec![0f32; batch * dm]);
        assert!(
            unsafe { graphed.capture_graph(&mut gst, &mut gsc) }.is_err(),
            "{dtype:?}: a capture needs an eager step first"
        );
        eager
            .prefill(&prompt, &mut erows, &mut est, &mut epf)
            .unwrap();
        graphed
            .prefill(&prompt, &mut grows, &mut gst, &mut gpf)
            .unwrap();
        for t in 0..12u64 {
            let x = det(batch * input_dim, 500 + t, 1.0);
            eager.step(&x, &mut eo, &mut est, &mut esc).unwrap();
            graphed.step(&x, &mut go, &mut gst, &mut gsc).unwrap();
            assert_eq!(eo, go, "{dtype:?} step {t}");
            if t == 0 {
                unsafe { graphed.capture_graph(&mut gst, &mut gsc) }.unwrap();
                assert!(graphed.has_graph());
            }
            if t == 6 {
                eager
                    .prefill(&prompt, &mut erows, &mut est, &mut epf)
                    .unwrap();
                graphed
                    .prefill(&prompt, &mut grows, &mut gst, &mut gpf)
                    .unwrap();
                assert_eq!(erows, grows, "{dtype:?}: prefill between replays");
            }
        }
        let x = det(batch * input_dim, 600, 1.0);
        let mut other = graphed.alloc_state().unwrap();
        assert!(
            graphed.step(&x, &mut go, &mut other, &mut gsc).is_err(),
            "{dtype:?}: a replay on another state"
        );
        graphed.clear_graph().unwrap();
        assert!(!graphed.has_graph());
        let mut fresh = eager.alloc_state().unwrap();
        eager.step(&x, &mut eo, &mut fresh, &mut esc).unwrap();
        graphed.step(&x, &mut go, &mut other, &mut gsc).unwrap();
        assert_eq!(eo, go, "{dtype:?}: eager again after the clear");
    }
}
