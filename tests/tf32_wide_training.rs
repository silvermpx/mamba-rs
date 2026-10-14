//! Tf32 storage trains at the widths of the published Mamba-1 models.
//!
//! The deterministic Tf32 GEMMs pick a measured cell for each shape and, for
//! a shape no cell measured, the nearest one. A nearest cell can carry a
//! split-K plan whose workspace the shape at hand outgrows. The step must
//! then run on a plan that fits, never fail.
#![cfg(feature = "cuda")]

use mamba_rs::config::ScanMode;
use mamba_rs::{
    GpuOptions, MambaConfig, MambaGpuTrainer, MambaWeights, TrainSessionCfg, WeightDtype,
};

fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((x >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.05
        })
        .collect()
}

fn one_step(d_model: usize, batch: usize, seq_len: usize) {
    let cfg = MambaConfig {
        d_model,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        n_layers: 1,
        scan_mode: ScanMode::Auto,
        block_norm: mamba_rs::BlockNorm::Rms,
        final_norm: true,
        bidirectional: false,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
        dt_rank: None,
        proj_bias: false,
        conv_bias: true,
        d_intermediate: 0,
    };
    let w = MambaWeights::init(&cfg, d_model, 5);
    let session = TrainSessionCfg {
        input_dim: d_model,
        batch,
        seq_len,
        lr: 1e-4,
        weight_decay: 0.0,
    };
    let mut t = MambaGpuTrainer::new(
        &w,
        cfg,
        session,
        GpuOptions::default().dtype(WeightDtype::Tf32),
    )
    .unwrap();
    let n = batch * seq_len * d_model;
    let m = t.step(&values(n, 1), &values(n, 2));
    assert!(m.is_ok(), "d_model {d_model} at {batch} x {seq_len}: {m:?}");
}

#[test]
fn tf32_trains_mamba_130m_width_at_8192_rows() {
    one_step(768, 8, 1024);
}

#[test]
fn tf32_trains_mamba_370m_width_at_16384_rows() {
    one_step(1024, 16, 1024);
}
