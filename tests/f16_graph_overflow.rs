//! A captured f16 training step whose gradients overflow moves nothing.
//!
//! The eager f16 step reads the overflow flag on the host and skips the
//! optimizer. A captured step cannot branch on the host, so the fused AdamW
//! reads the flag on the device instead. Either way a step that overflows
//! leaves the master weights and the Adam moments exactly as they were, and
//! the step after it runs as if the overflow never happened.
#![cfg(feature = "cuda")]

use mamba_rs::config::ScanMode;
use mamba_rs::gpu::adamw::AdamWStateBlob;
use mamba_rs::{
    GpuOptions, Mamba3Config, Mamba3GpuTrainer, Mamba3Weights, MambaConfig, MambaGpuTrainer,
    MambaWeights, TrainSessionCfg, WeightDtype,
};

const BATCH: usize = 2;
const SEQ_LEN: usize = 32;
const D_MODEL: usize = 32;

fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((x >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.1
        })
        .collect()
}

fn session() -> TrainSessionCfg {
    TrainSessionCfg {
        input_dim: D_MODEL,
        batch: BATCH,
        seq_len: SEQ_LEN,
        lr: 1e-3,
        weight_decay: 0.1,
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn m1_flat(w: &MambaWeights) -> Vec<f32> {
    let mut v = [w.input_proj_w.as_slice(), &w.input_proj_b].concat();
    for l in &w.layers {
        for t in [
            &l.norm_weight,
            &l.in_proj_w,
            &l.conv1d_weight,
            &l.conv1d_bias,
            &l.x_proj_w,
            &l.dt_proj_w,
            &l.dt_proj_b,
            &l.a_log,
            &l.d_param,
            &l.out_proj_w,
        ] {
            v.extend_from_slice(t);
        }
    }
    v.extend_from_slice(&w.norm_f_weight);
    v
}

fn m3_flat(w: &Mamba3Weights) -> Vec<f32> {
    let mut v = [w.input_proj_w.as_slice(), &w.input_proj_b].concat();
    for l in &w.layers {
        for t in [
            &l.norm_weight,
            &l.in_proj_w,
            &l.dt_bias,
            &l.b_norm_weight,
            &l.c_norm_weight,
            &l.b_bias,
            &l.c_bias,
            &l.d_param,
            &l.norm_gate_weight,
            &l.out_proj_w,
        ] {
            v.extend_from_slice(t);
        }
    }
    v.extend_from_slice(&w.norm_f_weight);
    v
}

fn moments(blob: &AdamWStateBlob) -> (Vec<u32>, Vec<u32>, u64) {
    (bits(&blob.m), bits(&blob.v), blob.step)
}

/// A gradient no loss scale survives: every element overflows in f16.
fn overflowing_gradient() -> Vec<f32> {
    vec![f32::MAX; BATCH * SEQ_LEN * D_MODEL]
}

/// The loss scale the tests train at. The scaler starts at 65536, where
/// ordinary gradients of this model can overflow too and the first steps
/// would be skipped while it backs off. A fixed modest scale keeps every
/// ordinary step clean, so the only overflow is the one a test makes.
const SCALE: f32 = 256.0;

#[test]
fn m1_f16_graph_step_that_overflows_moves_nothing() {
    let cfg = MambaConfig {
        d_model: D_MODEL,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: ScanMode::Auto,
        block_norm: mamba_rs::BlockNorm::Rms,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
        dt_rank: None,
        proj_bias: false,
        conv_bias: true,
        d_intermediate: 0,
    };
    let weights = MambaWeights::init(&cfg, D_MODEL, 71);
    let mut t = MambaGpuTrainer::new(
        &weights,
        cfg,
        session(),
        GpuOptions::default().dtype(WeightDtype::F16),
    )
    .unwrap();
    t.load_scaler_state(SCALE, 0).unwrap();
    let input = values(BATCH * SEQ_LEN * D_MODEL, 1);
    let d_out = values(BATCH * SEQ_LEN * D_MODEL, 2);
    let eager = t.step(&input, &d_out).unwrap();
    assert_eq!(eager.overflow_skipped, Some(false));
    t.capture_graph().unwrap();
    let applied = t.step(&input, &d_out).unwrap();
    assert!(applied.graph_replayed);
    assert_eq!(applied.overflow_skipped, Some(false));

    let master = t.snapshot_master().unwrap();
    let adam = moments(&t.optimizer_state().unwrap());
    let skipped = t.step(&input, &overflowing_gradient()).unwrap();
    assert!(skipped.graph_replayed);
    assert_eq!(skipped.overflow_skipped, Some(true));
    let after = t.snapshot_master().unwrap();
    assert_eq!(
        bits(&m1_flat(&after)),
        bits(&m1_flat(&master)),
        "the master weights moved on an overflowing step"
    );
    assert_eq!(
        moments(&t.optimizer_state().unwrap()),
        adam,
        "the Adam moments moved on an overflowing step"
    );
}

#[test]
fn m3_f16_graph_step_that_overflows_moves_nothing() {
    let cfg = Mamba3Config {
        d_model: D_MODEL,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        ..Mamba3Config::default()
    };
    let weights = Mamba3Weights::init(&cfg, D_MODEL, 72);
    let mut t = Mamba3GpuTrainer::new(
        &weights,
        cfg,
        session(),
        GpuOptions::default().dtype(WeightDtype::F16),
    )
    .unwrap();
    t.load_scaler_state(SCALE, 0).unwrap();
    let input = values(BATCH * SEQ_LEN * D_MODEL, 3);
    let d_out = values(BATCH * SEQ_LEN * D_MODEL, 4);
    let eager = t.step(&input, &d_out).unwrap();
    assert_eq!(eager.overflow_skipped, Some(false));
    t.capture_graph().unwrap();
    let applied = t.step(&input, &d_out).unwrap();
    assert!(applied.graph_replayed);
    assert_eq!(applied.overflow_skipped, Some(false));

    let master = t.snapshot_master().unwrap();
    let adam = moments(&t.optimizer_state().unwrap());
    let skipped = t.step(&input, &overflowing_gradient()).unwrap();
    assert!(skipped.graph_replayed);
    assert_eq!(skipped.overflow_skipped, Some(true));
    let after = t.snapshot_master().unwrap();
    assert_eq!(
        bits(&m3_flat(&after)),
        bits(&m3_flat(&master)),
        "the master weights moved on an overflowing step"
    );
    assert_eq!(
        moments(&t.optimizer_state().unwrap()),
        adam,
        "the Adam moments moved on an overflowing step"
    );
}
