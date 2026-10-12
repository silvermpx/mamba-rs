//! A Mamba-3 SISO model with the gated MLP after every mixer, the layout of
//! the official checkpoints, trains on the GPU in every lane and on both
//! scans: AdamW lowers a loss and moves every MLP tensor, and the weights
//! the forward reads follow the master weights, so after the steps the
//! trainer's forward is, bit for bit, that of a trainer built from the
//! weights it reports. In the half lanes that holds only when the optimizer
//! writes every half copy the GEMMs read and every f32 copy the norms read.
#![cfg(feature = "cuda")]

use mamba_rs::config::ScanMode;
use mamba_rs::{
    BackwardOpts, GpuOptions, Mamba3Config, Mamba3GpuTrainer, Mamba3Weights, TrainSessionCfg,
    WeightDtype,
};

const BATCH: usize = 2;
const SEQ_LEN: usize = 48;
const STEPS: usize = 12;

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((x >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.5
        })
        .collect()
}

fn trainer(w: &Mamba3Weights, cfg: Mamba3Config, dtype: WeightDtype) -> Mamba3GpuTrainer {
    let session = TrainSessionCfg {
        input_dim: cfg.d_model,
        batch: BATCH,
        seq_len: SEQ_LEN,
        lr: 1e-2,
        weight_decay: 0.1,
    };
    Mamba3GpuTrainer::new(w, cfg, session, GpuOptions::default().dtype(dtype)).unwrap()
}

#[test]
fn m3_gpu_mlp_training_lowers_a_loss_and_serves_its_weights() {
    for (scan_mode, is_outproj_norm) in [(ScanMode::Auto, true), (ScanMode::Sequential, false)] {
        let cfg = Mamba3Config {
            d_model: 64,
            d_state: 16,
            expand: 2,
            headdim: 16,
            n_layers: 2,
            d_intermediate: 160,
            is_outproj_norm,
            scan_mode,
            ..Mamba3Config::default()
        };
        let dm = cfg.d_model;
        let mut w = Mamba3Weights::init(&cfg, dm, 23);
        w.input_proj_w.clear();
        w.input_proj_b.clear();
        let input = det(BATCH * SEQ_LEN * dm, 1);
        let target = det(BATCH * SEQ_LEN * dm, 2);
        for dtype in [
            WeightDtype::F32,
            WeightDtype::Tf32,
            WeightDtype::Bf16,
            WeightDtype::F16,
        ] {
            let label = format!("{scan_mode:?} {dtype:?}");
            let mut t = trainer(&w, cfg, dtype);
            let mut out = vec![0f32; BATCH * SEQ_LEN * dm];
            let (mut losses, mut applied) = (Vec::new(), 0usize);
            for _ in 0..STEPS {
                t.reset_state().unwrap();
                t.forward(&input, &mut out).unwrap();
                let d: Vec<f32> = out.iter().zip(&target).map(|(o, y)| o - y).collect();
                losses.push(d.iter().map(|v| 0.5 * f64::from(*v).powi(2)).sum::<f64>());
                let m = t.backward_step(&d, BackwardOpts::default()).unwrap();
                applied += usize::from(m.optimizer_stepped);
            }
            let (first, last) = (losses[0], losses[STEPS - 1]);
            eprintln!("{label}: loss {first:.3} -> {last:.3}, {applied} of {STEPS} steps applied");
            // The f16 loss scale starts high and backs off on the first
            // overflows.
            assert!(
                applied >= STEPS / 2,
                "{label}: only {applied} steps applied"
            );
            assert!(last < 0.9 * first, "{label}: loss {first:.3} -> {last:.3}");

            let moved = t.snapshot_master().unwrap();
            for (l, (a, b)) in moved.mlp.iter().zip(&w.mlp).enumerate() {
                for (name, x, y) in [
                    ("norm", &a.norm_weight, &b.norm_weight),
                    ("fc1", &a.fc1_w, &b.fc1_w),
                    ("fc2", &a.fc2_w, &b.fc2_w),
                ] {
                    assert!(x != y, "{label}: the MLP {name} of layer {l} did not move");
                }
            }

            t.reset_state().unwrap();
            t.forward(&input, &mut out).unwrap();
            let mut want = vec![0f32; out.len()];
            trainer(&moved, cfg, dtype)
                .forward(&input, &mut want)
                .unwrap();
            let differ = out
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differ, 0,
                "{label}: {differ} outputs of the stepped trainer differ from a trainer \
                 built from the weights it reports"
            );
        }
    }
}
