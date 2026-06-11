//! Step 15 smoke tests — `MambaTrainer` (M1) and `Mamba3Trainer` (M3)
//! high-level APIs. Validates:
//!   1. Eager steps run without panic.
//!   2. `capture_graph()` succeeds.
//!   3. Post-capture `step()` returns `graph_replayed=true`.
//!   4. Master weights actually change across steps (non-trivial training).
//!   5. `snapshot_master()` round-trips through `Mamba*Weights`.

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & 0xFFFF) as f32 / 65536.0 - 0.5
        })
        .collect()
}

#[test]
fn m1_trainer_bf16_smoke() {
    use mamba_rs::config::MambaConfig;
    use mamba_rs::mamba_ssm::gpu::trainer::MambaTrainer;
    use mamba_rs::weights::MambaWeights;

    let cfg = MambaConfig {
        d_model: 32,
        n_layers: 1,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    };
    let input_dim = cfg.d_model;
    let batch = 1;
    let seq_len = 4;
    let n = batch * seq_len * input_dim;

    let mut cpu = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    cpu.input_proj_w.clear();
    cpu.input_proj_b.clear();
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    let mut trainer =
        MambaTrainer::new_with_dtype(0, &cpu, cfg, input_dim, batch, seq_len, WeightDtype::Bf16)
            .expect("construct");

    let before = trainer.snapshot_master().unwrap();

    // Two warmup eager steps.
    for s in 0..2 {
        let m = trainer.step(&det(n, 0xA0 + s), &det(n, 0xB0 + s)).unwrap();
        assert!(!m.graph_replayed, "pre-capture must run eager");
        assert_eq!(m.step, s as u64 + 1);
    }

    // Capture + replay.
    trainer.capture_graph().unwrap();
    assert!(trainer.has_graph());
    for s in 0..3 {
        let m = trainer.step(&det(n, 0xC0 + s), &det(n, 0xD0 + s)).unwrap();
        assert!(m.graph_replayed, "post-capture must replay");
    }

    let after = trainer.snapshot_master().unwrap();

    // Weights must have moved.
    let max_diff = before.layers[0]
        .in_proj_w
        .iter()
        .zip(&after.layers[0].in_proj_w)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("M1 trainer: weight max_diff after 5 steps = {max_diff:.3e}");
    assert!(max_diff > 0.0, "weights did not change");
    assert!(max_diff.is_finite());
}

/// Multi-layer (n_layers=3) trainer smoke — validates the full
/// forward+backward+AdamW+sync path through the high-level API at a
/// realistic depth. Single-layer tests above don't exercise the per-
/// layer residual stream + d_temporal hand-off.
#[test]
fn m1_trainer_multi_layer_bf16() {
    use mamba_rs::config::MambaConfig;
    use mamba_rs::mamba_ssm::gpu::trainer::{MambaTrainer, TrainSessionCfg};
    use mamba_rs::weights::MambaWeights;

    let cfg = MambaConfig {
        d_model: 32,
        n_layers: 3,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    };
    let input_dim = cfg.d_model;
    let batch = 1;
    let seq_len = 4;
    let n = batch * seq_len * input_dim;

    let mut cpu = MambaWeights::init(&cfg, input_dim, 0xCAFE_3133);
    cpu.input_proj_w.clear();
    cpu.input_proj_b.clear();
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    let mut trainer = MambaTrainer::new_full(
        0,
        &cpu,
        cfg,
        TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr: 1e-5,
            weight_decay: 0.0,
        },
        WeightDtype::Bf16,
    )
    .expect("construct multi-layer bf16");

    // Eager warmup
    for s in 0..2 {
        let m = trainer.step(&det(n, 0xA0 + s), &det(n, 0xB0 + s)).unwrap();
        assert!(!m.graph_replayed);
    }
    // Graph capture + replay
    trainer.capture_graph().unwrap();
    for s in 0..3 {
        let m = trainer.step(&det(n, 0xC0 + s), &det(n, 0xD0 + s)).unwrap();
        assert!(m.graph_replayed);
    }
    let snap = trainer.snapshot_master().unwrap();
    assert_eq!(snap.layers.len(), 3);
    for (i, lw) in snap.layers.iter().enumerate() {
        assert!(
            lw.in_proj_w.iter().all(|v| v.is_finite()),
            "L{i} in_proj_w non-finite"
        );
        assert!(
            lw.out_proj_w.iter().all(|v| v.is_finite()),
            "L{i} out_proj_w non-finite"
        );
    }
}

#[test]
fn m3_trainer_multi_layer_bf16() {
    use mamba_rs::mamba3_siso::config::Mamba3Config;
    use mamba_rs::mamba3_siso::gpu::trainer::{Mamba3Trainer, TrainSessionCfg};
    use mamba_rs::mamba3_siso::weights::Mamba3Weights;

    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 3,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
    };
    let input_dim = cfg.d_model;
    let batch = 1;
    let seq_len = 64;
    let n = batch * seq_len * input_dim;

    let mut cpu = Mamba3Weights::init(&cfg, input_dim, 0x000C_AFE3);
    cpu.input_proj_w.clear();
    cpu.input_proj_b.clear();

    let mut trainer = Mamba3Trainer::new_full(
        0,
        &cpu,
        cfg,
        TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr: 1e-5,
            weight_decay: 0.0,
        },
        WeightDtype::Bf16,
    )
    .expect("construct multi-layer M3 bf16");

    for s in 0..2 {
        let m = trainer.step(&det(n, 0xE0 + s), &det(n, 0xF0 + s)).unwrap();
        assert!(!m.graph_replayed);
    }
    trainer.capture_graph().unwrap();
    for s in 0..3 {
        let m = trainer.step(&det(n, 0x20 + s), &det(n, 0x30 + s)).unwrap();
        assert!(m.graph_replayed);
    }
    let snap = trainer.snapshot_master().unwrap();
    assert_eq!(snap.layers.len(), 3);
    for (i, lw) in snap.layers.iter().enumerate() {
        assert!(
            lw.in_proj_w.iter().all(|v| v.is_finite()),
            "L{i} in_proj_w non-finite"
        );
        assert!(
            lw.out_proj_w.iter().all(|v| v.is_finite()),
            "L{i} out_proj_w non-finite"
        );
    }
}

#[test]
fn m1_trainer_f16_smoke_eager_with_loss_scaler() {
    use mamba_rs::config::MambaConfig;
    use mamba_rs::mamba_ssm::gpu::trainer::{MambaTrainer, TrainSessionCfg};
    use mamba_rs::weights::MambaWeights;

    let cfg = MambaConfig {
        d_model: 32,
        n_layers: 1,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    };
    let input_dim = cfg.d_model;
    let batch = 1;
    let seq_len = 4;
    let n = batch * seq_len * input_dim;

    let mut cpu = MambaWeights::init(&cfg, input_dim, 0xF16_C0FF);
    cpu.input_proj_w.clear();
    cpu.input_proj_b.clear();
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    // Tiny lr — synthetic random d_temporal at default lr blows up f16
    // weights within a few steps; this test cares about the API path,
    // not convergence.
    let mut trainer = MambaTrainer::new_full(
        0,
        &cpu,
        cfg,
        TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr: 1e-7,
            weight_decay: 0.0,
        },
        WeightDtype::F16,
    )
    .expect("construct f16");
    assert_eq!(trainer.dtype(), WeightDtype::F16);

    // f16 must report loss_scale + overflow flag every step.
    for s in 0..3 {
        let m = trainer.step(&det(n, 0xA0 + s), &det(n, 0xB0 + s)).unwrap();
        assert!(!m.graph_replayed, "f16 must run eager (no graph)");
        assert!(m.loss_scale.is_some(), "f16 must report loss_scale");
        assert!(
            m.overflow_skipped.is_some(),
            "f16 must report overflow_skipped"
        );
    }

    // f16 graph capture must work (Step 22).
    trainer.capture_graph().expect("f16 capture");
    assert!(trainer.has_graph(), "f16 graph captured");
    for s in 0..3 {
        let m = trainer.step(&det(n, 0xC0 + s), &det(n, 0xD0 + s)).unwrap();
        assert!(m.graph_replayed, "post-capture must replay");
        assert!(m.loss_scale.is_some());
        assert!(m.overflow_skipped.is_some());
    }

    // Just check we can roundtrip — divergence at synthetic gradients is OK.
    let _ = trainer.snapshot_master().unwrap();
}

#[test]
fn m3_trainer_f16_smoke_eager_with_loss_scaler() {
    use mamba_rs::mamba3_siso::config::Mamba3Config;
    use mamba_rs::mamba3_siso::gpu::trainer::{Mamba3Trainer, TrainSessionCfg};
    use mamba_rs::mamba3_siso::weights::Mamba3Weights;

    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
    };
    let input_dim = cfg.d_model;
    let batch = 1;
    let seq_len = 64;
    let n = batch * seq_len * input_dim;

    let mut cpu = Mamba3Weights::init(&cfg, input_dim, 0x000F_16DE);
    cpu.input_proj_w.clear();
    cpu.input_proj_b.clear();

    let mut trainer = Mamba3Trainer::new_full(
        0,
        &cpu,
        cfg,
        TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr: 1e-7,
            weight_decay: 0.0,
        },
        WeightDtype::F16,
    )
    .expect("construct M3 f16");
    assert_eq!(trainer.dtype(), WeightDtype::F16);

    for s in 0..3 {
        let m = trainer.step(&det(n, 0xE0 + s), &det(n, 0xF0 + s)).unwrap();
        assert!(!m.graph_replayed);
        assert!(m.loss_scale.is_some());
        assert!(m.overflow_skipped.is_some());
    }

    trainer.capture_graph().expect("M3 f16 capture");
    assert!(trainer.has_graph());
    for s in 0..3 {
        let m = trainer.step(&det(n, 0x20 + s), &det(n, 0x30 + s)).unwrap();
        assert!(m.graph_replayed);
        assert!(m.loss_scale.is_some());
        assert!(m.overflow_skipped.is_some());
    }

    let _ = trainer.snapshot_master().unwrap();
}

#[test]
fn m1_trainer_f32_smoke() {
    use mamba_rs::config::MambaConfig;
    use mamba_rs::mamba_ssm::gpu::trainer::MambaTrainer;
    use mamba_rs::weights::MambaWeights;

    let cfg = MambaConfig {
        d_model: 32,
        n_layers: 1,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    };
    let input_dim = cfg.d_model;
    let batch = 1;
    let seq_len = 4;
    let n = batch * seq_len * input_dim;

    // f32 forward has no identity-proj branch; keep input_proj populated.
    let mut cpu = MambaWeights::init(&cfg, input_dim, 0xF32_C0FF);
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    let mut trainer =
        MambaTrainer::new_with_dtype(0, &cpu, cfg, input_dim, batch, seq_len, WeightDtype::F32)
            .expect("construct f32");
    assert_eq!(trainer.dtype(), WeightDtype::F32);

    for s in 0..2 {
        let m = trainer.step(&det(n, 0xA0 + s), &det(n, 0xB0 + s)).unwrap();
        assert!(!m.graph_replayed);
    }
    trainer.capture_graph().unwrap();
    assert!(trainer.has_graph());
    for s in 0..3 {
        let m = trainer.step(&det(n, 0xC0 + s), &det(n, 0xD0 + s)).unwrap();
        assert!(m.graph_replayed);
    }
    let snap = trainer.snapshot_master().unwrap();
    assert!(snap.layers[0].in_proj_w.iter().all(|v| v.is_finite()));
}

#[test]
fn m3_trainer_bf16_smoke() {
    use mamba_rs::mamba3_siso::config::Mamba3Config;
    use mamba_rs::mamba3_siso::gpu::trainer::Mamba3Trainer;
    use mamba_rs::mamba3_siso::weights::Mamba3Weights;

    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
    };
    let input_dim = cfg.d_model;
    let batch = 1;
    let seq_len = 64;
    let n = batch * seq_len * input_dim;

    let mut cpu = Mamba3Weights::init(&cfg, input_dim, 0xDECA11);
    cpu.input_proj_w.clear();
    cpu.input_proj_b.clear();

    let mut trainer =
        Mamba3Trainer::new_with_dtype(0, &cpu, cfg, input_dim, batch, seq_len, WeightDtype::Bf16)
            .expect("construct");

    let before = trainer.snapshot_master().unwrap();

    for s in 0..2 {
        let m = trainer.step(&det(n, 0xE0 + s), &det(n, 0xF0 + s)).unwrap();
        assert!(!m.graph_replayed);
        assert_eq!(m.step, s as u64 + 1);
    }

    trainer.capture_graph().unwrap();
    assert!(trainer.has_graph());
    for s in 0..3 {
        let m = trainer.step(&det(n, 0x20 + s), &det(n, 0x30 + s)).unwrap();
        assert!(m.graph_replayed);
    }

    let after = trainer.snapshot_master().unwrap();
    let max_diff = before.layers[0]
        .in_proj_w
        .iter()
        .zip(&after.layers[0].in_proj_w)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("M3 trainer: weight max_diff after 5 steps = {max_diff:.3e}");
    assert!(max_diff > 0.0);
    assert!(max_diff.is_finite());
}

#[test]
fn m3_trainer_f32_smoke() {
    use mamba_rs::mamba3_siso::config::Mamba3Config;
    use mamba_rs::mamba3_siso::gpu::trainer::Mamba3Trainer;
    use mamba_rs::mamba3_siso::weights::Mamba3Weights;

    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
    };
    let input_dim = cfg.d_model;
    let batch = 1;
    let seq_len = 64;
    let n = batch * seq_len * input_dim;

    // f32 M3 forward needs eye(dm) input_proj.
    let mut cpu = Mamba3Weights::init(&cfg, input_dim, 0x000F_3F32);
    cpu.input_proj_w = (0..input_dim * cfg.d_model)
        .map(|i| {
            if i / cfg.d_model == i % cfg.d_model {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    cpu.input_proj_b = vec![0.0; cfg.d_model];

    let mut trainer =
        Mamba3Trainer::new_with_dtype(0, &cpu, cfg, input_dim, batch, seq_len, WeightDtype::F32)
            .expect("construct f32");
    assert_eq!(trainer.dtype(), WeightDtype::F32);

    for s in 0..2 {
        let m = trainer.step(&det(n, 0xE0 + s), &det(n, 0xF0 + s)).unwrap();
        assert!(!m.graph_replayed);
    }
    trainer.capture_graph().unwrap();
    assert!(trainer.has_graph());
    for s in 0..3 {
        let m = trainer.step(&det(n, 0x20 + s), &det(n, 0x30 + s)).unwrap();
        assert!(m.graph_replayed);
    }
    let snap = trainer.snapshot_master().unwrap();
    assert!(snap.layers[0].in_proj_w.iter().all(|v| v.is_finite()));
}
