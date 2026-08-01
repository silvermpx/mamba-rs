//! Run-to-run bit determinism of TRAINING in the parallel-scan regime.
//!
//! The existing determinism pins run small-T shapes that dispatch the
//! sequential kernels; this suite pins the same law for the parallel scan
//! (T > PARALLEL_SCAN_THRESHOLD, ScanMode::Auto): two identical training
//! runs from identical init must produce bit-identical master weights.

#![cfg(feature = "cuda")]

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::forward::PARALLEL_SCAN_THRESHOLD;
use mamba_rs::mamba_ssm::gpu::trainer::{BackwardOpts, MambaTrainer};
use mamba_rs::weights::MambaWeights;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * scale
        })
        .collect()
}

fn flatten(w: &MambaWeights) -> Vec<f32> {
    let mut out = Vec::new();
    out.extend_from_slice(&w.input_proj_w);
    out.extend_from_slice(&w.input_proj_b);
    for l in &w.layers {
        out.extend_from_slice(&l.norm_weight);
        out.extend_from_slice(&l.in_proj_w);
        out.extend_from_slice(&l.conv1d_weight);
        out.extend_from_slice(&l.conv1d_bias);
        out.extend_from_slice(&l.x_proj_w);
        out.extend_from_slice(&l.dt_proj_w);
        out.extend_from_slice(&l.dt_proj_b);
        out.extend_from_slice(&l.a_log);
    }
    out.extend_from_slice(&w.norm_f_weight);
    out
}

fn run_once(steps: usize) -> Vec<f32> {
    let (batch, input_dim) = (1usize, 48usize);
    let seq_len = PARALLEL_SCAN_THRESHOLD + 44;
    let cfg = MambaConfig {
        d_model: 64,
        n_layers: 2,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
    };
    let mut cpu = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }
    let mut trainer =
        MambaTrainer::new_with_dtype(0, &cpu, cfg, input_dim, batch, seq_len, WeightDtype::Bf16)
            .expect("trainer");
    let mut temporal = vec![0.0f32; batch * seq_len * cfg.d_model];
    for s in 0..steps {
        let input = det(batch * seq_len * input_dim, 0xA5 + s as u32, 0.05);
        let d_temporal = det(batch * seq_len * cfg.d_model, 0xB6 + s as u32, 0.01);
        trainer.forward(&input, &mut temporal).expect("fwd");
        trainer
            .backward_step(&d_temporal, BackwardOpts::default())
            .expect("bwd");
    }
    flatten(&trainer.snapshot_master().expect("snapshot"))
}

#[test]
fn parallel_training_is_bit_reproducible_run_to_run() {
    let a = run_once(4);
    let b = run_once(4);
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "master weight [{i}] diverged across identical parallel-mode runs: {x} vs {y}"
        );
    }
}
