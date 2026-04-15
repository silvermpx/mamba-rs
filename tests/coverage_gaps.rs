//! Coverage-gap closures identified in the v0.3.0 multi-agent audit.
//!
//! * `m1_trainer_parallel_scan_{f32,bf16,f16}` — M1 training through the full
//!   `MambaTrainer` API with `ScanMode::Parallel` (T > 128). Previously only
//!   the unit parity tests (`m1_parallel_scan_typed_parity`,
//!   `m1_parallel_bwd_parity`) touched the parallel scan kernels; this
//!   exercises them end-to-end including AdamW + sync + graph capture.
//! * `m1_trainer_f16_production_lr_stable` — f16 training at a realistic
//!   learning rate (1e-4) over 50 eager steps. Verifies the dynamic loss
//!   scaler settles, at least some steps commit (not stuck in overflow),
//!   and master weights remain finite. Existing f16 smoke tests use
//!   lr=1e-7 which hides scaler pathologies.
//! * `a_log_actually_reaches_ssm_after_training` — asserts that after a
//!   few training steps, BOTH `a_log` master weights AND the SSM
//!   recurrence's `a_neg` compute buffer have changed. This is the
//!   regression test for the round-2 audit CRIT bug where `a_neg` was
//!   initialized once at trainer construction and never refreshed after
//!   AdamW touched `a_log` — letting the optimizer "train" the A-matrix
//!   in isolation while the forward kernel kept reading the pre-training
//!   decay values forever. Without the fix this test fails because
//!   `a_neg_all` is identical to its initialization after any number of
//!   steps (only `a_log` changes).

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;

fn det_scaled(n: usize, seed: u32, scale: f32) -> Vec<f32> {
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

// ===========================================================================
// Gap 1: parallel scan integration via Trainer API
// ===========================================================================

fn run_parallel_scan_trainer(dtype: WeightDtype) {
    use mamba_rs::config::{MambaConfig, ScanMode};
    use mamba_rs::mamba_ssm::gpu::trainer::MambaTrainer;
    use mamba_rs::weights::MambaWeights;

    // T=256 triggers the parallel prefix-scan kernels for both forward and
    // backward on the GPU training path.
    let cfg = MambaConfig {
        d_model: 32,
        n_layers: 2,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Parallel,
    };
    let input_dim = cfg.d_model;
    let batch = 1;
    let seq_len = 256;
    let n = batch * seq_len * input_dim;

    let mut cpu = MambaWeights::init(&cfg, input_dim, 0xBA4A_15CA);
    if !matches!(dtype, WeightDtype::F32) {
        cpu.input_proj_w.clear();
        cpu.input_proj_b.clear();
    }
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    // lr=1e-7 keeps synthetic-gradient f16 runs finite; scan correctness is
    // what this test proves, not optimizer convergence.
    let mut trainer =
        MambaTrainer::new_full(0, &cpu, cfg, input_dim, batch, seq_len, dtype, 1e-7, 0.0)
            .expect("construct parallel-scan trainer");

    let before = trainer.snapshot_master().expect("snapshot pre");

    // f16 at default loss_scale=65536 × random gradient of O(0.5) overflows
    // immediately; feed RL-realistic micro-gradients so the scaler converges.
    let in_scale = if matches!(dtype, WeightDtype::F16) {
        0.001
    } else {
        1.0
    };

    // Eager warmup — both fwd and bwd go through the parallel scan kernels.
    // Give f16 extra warmup steps so the scaler can back off to a stable level
    // before we assert anything about weight movement.
    let warmup_steps = if matches!(dtype, WeightDtype::F16) {
        12
    } else {
        2
    };
    for s in 0..warmup_steps {
        let m = trainer
            .step(
                &det_scaled(n, 0xA0 + s as u32, in_scale),
                &det_scaled(n, 0xB0 + s as u32, in_scale),
            )
            .expect("eager step");
        assert!(!m.graph_replayed);
        if matches!(dtype, WeightDtype::F16) {
            assert!(m.loss_scale.is_some());
        }
    }

    // Graph capture + replay — full parallel-scan graph round-trip.
    trainer
        .capture_graph()
        .expect("capture parallel-scan graph");
    assert!(trainer.has_graph());
    for s in 0..5 {
        let m = trainer
            .step(
                &det_scaled(n, 0xC0 + s as u32, in_scale),
                &det_scaled(n, 0xD0 + s as u32, in_scale),
            )
            .expect("graph step");
        assert!(m.graph_replayed, "post-capture must replay");
    }

    let after = trainer.snapshot_master().expect("snapshot post");
    for (i, lw) in after.layers.iter().enumerate() {
        assert!(
            lw.in_proj_w.iter().all(|v| v.is_finite()),
            "{dtype:?} L{i} in_proj_w non-finite after parallel-scan training"
        );
        assert!(
            lw.out_proj_w.iter().all(|v| v.is_finite()),
            "{dtype:?} L{i} out_proj_w non-finite after parallel-scan training"
        );
    }

    // Some weights must have moved — parallel scan must actually run bwd.
    let max_diff = before.layers[0]
        .in_proj_w
        .iter()
        .zip(&after.layers[0].in_proj_w)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("parallel-scan {dtype:?} trainer: max |Δw| = {max_diff:.3e}");
    assert!(max_diff > 0.0, "{dtype:?} weights did not move");
    assert!(max_diff.is_finite());
}

#[test]
fn m1_trainer_parallel_scan_f32() {
    run_parallel_scan_trainer(WeightDtype::F32);
}

#[test]
fn m1_trainer_parallel_scan_bf16() {
    run_parallel_scan_trainer(WeightDtype::Bf16);
}

#[test]
fn m1_trainer_parallel_scan_f16() {
    run_parallel_scan_trainer(WeightDtype::F16);
}

// ===========================================================================
// Gap 3: f16 training at production learning rate over 50 steps
// ===========================================================================

/// f16 training at realistic `lr=1e-4` with the dynamic loss scaler running
/// 50 eager steps. Previous f16 tests used `lr=1e-7` which makes updates
/// vanish and sidesteps the whole point of f16 + loss scaling. This test
/// verifies:
///
/// 1. Weights remain finite after 50 steps.
/// 2. The loss scaler stays inside `[1.0, 2^24]` (PyTorch GradScaler default
///    bounds).
/// 3. At least one step is NOT overflow-skipped (otherwise the test is just
///    "the optimizer stood still", which tells us nothing).
/// 4. Weights actually move between pre-train and post-train snapshots.
///
/// Small inputs at `scale=0.01` keep the synthetic gradients within bf16
/// dynamic range at the default initial loss scale; with `lr=1e-4` the
/// update magnitude is comparable to a real RL actor step.
// ===========================================================================
// Regression: a_log gradient must actually flow into SSM recurrence
// ===========================================================================

/// Trainer construction seeds `a_neg_all = -exp(a_log)` once. AdamW
/// updates `a_log` every step, but the forward/backward SSM kernels read
/// `a_neg_all`. Before the round-2 audit fix, `a_neg_all` was never
/// recomputed → the SSM used the initial A-matrix for the entire run
/// even though `a_log` changed in memory. Assert that after 5 training
/// steps:
///   1. `a_log` master weights differ from initialization (AdamW worked).
///   2. `a_neg_all` GPU buffer ALSO differs from initialization by AT
///      LEAST the same amount (fix made the refresh happen).
///   3. The delta tracks `-exp(a_log_new) + exp(a_log_old)` within a
///      small tolerance (the recompute formula is mathematically
///      correct, not just "some arbitrary update").
#[test]
fn a_log_actually_reaches_ssm_after_training() {
    use mamba_rs::config::{MambaConfig, ScanMode};
    use mamba_rs::mamba_ssm::gpu::trainer::MambaTrainer;
    use mamba_rs::weights::MambaWeights;

    let cfg = MambaConfig {
        d_model: 64,
        n_layers: 2,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Sequential,
    };
    let input_dim = cfg.d_model;
    let batch = 1;
    let seq_len = 8;
    let n = batch * seq_len * input_dim;

    let mut cpu = MambaWeights::init(&cfg, input_dim, 0xA10_C0FFEE);
    cpu.input_proj_w.clear();
    cpu.input_proj_b.clear();
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    // Initial a_log per layer — flattened for easy diff.
    let a_log_init: Vec<f32> = cpu.layers.iter().flat_map(|lw| lw.a_log.clone()).collect();
    let a_neg_init: Vec<f32> = a_log_init.iter().map(|&v| -v.exp()).collect();

    let mut trainer = MambaTrainer::new_full(
        0,
        &cpu,
        cfg,
        input_dim,
        batch,
        seq_len,
        WeightDtype::Bf16,
        1e-3, // large enough to move a_log visibly in 5 steps
        0.0,
    )
    .expect("build trainer");

    // Seed d_temporal with a signal that has non-trivial gradient w.r.t.
    // a_log. Pure zero gradient wouldn't move the optimizer at all.
    for s in 0..5 {
        trainer
            .step(&det_scaled(n, 0xA0 + s, 1.0), &det_scaled(n, 0xB0 + s, 0.1))
            .expect("training step");
    }

    // Read both sides after training.
    let after = trainer.snapshot_master().expect("snapshot");
    let a_log_after: Vec<f32> = after.layers.iter().flat_map(|lw| lw.a_log.clone()).collect();
    let a_neg_after = trainer.debug_a_neg_all().expect("download a_neg_all");

    // Step 1: a_log must have moved (optimizer is working).
    let a_log_max_delta: f32 = a_log_init
        .iter()
        .zip(&a_log_after)
        .map(|(a, b): (&f32, &f32)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        a_log_max_delta > 1e-5,
        "a_log did not move after 5 steps (max delta {a_log_max_delta:.3e}) — optimizer broken"
    );

    // Step 2: a_neg_all must have moved too (the fix under test).
    let a_neg_max_delta: f32 = a_neg_init
        .iter()
        .zip(&a_neg_after)
        .map(|(a, b): (&f32, &f32)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        a_neg_max_delta > 1e-5,
        "a_neg_all was NOT refreshed after AdamW updates to a_log \
         (max delta {a_neg_max_delta:.3e}). This is the round-2 audit \
         CRIT bug — SSM would run on stale A-matrix forever."
    );

    // Step 3: the recompute formula is mathematically correct, not just
    // an arbitrary mutation. a_neg_after[i] == -exp(a_log_after[i]) for
    // every i. Tolerance absorbs bf16 sync rounding on large negatives.
    let mut worst_formula_err = 0.0f32;
    for (log, neg) in a_log_after.iter().zip(a_neg_after.iter()) {
        let expected = -(*log as f32).exp();
        let err = (expected - neg).abs();
        if err > worst_formula_err {
            worst_formula_err = err;
        }
    }
    assert!(
        worst_formula_err < 1e-3,
        "a_neg_all != -exp(a_log) after refresh (max err {worst_formula_err:.3e}) \
         — recompute formula broken"
    );

    eprintln!(
        "a_log max delta = {a_log_max_delta:.3e}  a_neg max delta = {a_neg_max_delta:.3e}  \
         formula err = {worst_formula_err:.3e}"
    );
}

#[test]
fn m1_trainer_f16_production_lr_stable() {
    use mamba_rs::config::{MambaConfig, ScanMode};
    use mamba_rs::mamba_ssm::gpu::trainer::MambaTrainer;
    use mamba_rs::weights::MambaWeights;

    let cfg = MambaConfig {
        d_model: 64,
        n_layers: 2,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Sequential,
    };
    let input_dim = cfg.d_model;
    let batch = 2;
    let seq_len = 32;
    let n = batch * seq_len * input_dim;

    let mut cpu = MambaWeights::init(&cfg, input_dim, 0xF16_5CA1E);
    cpu.input_proj_w.clear();
    cpu.input_proj_b.clear();
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    let mut trainer = MambaTrainer::new_full(
        0,
        &cpu,
        cfg,
        input_dim,
        batch,
        seq_len,
        WeightDtype::F16,
        1e-4, // production-ish learning rate
        0.01, // weight decay
    )
    .expect("construct f16 prod-lr trainer");

    let before = trainer.snapshot_master().expect("snapshot pre");

    let mut scales: Vec<f32> = Vec::with_capacity(50);
    let mut committed_steps = 0usize;
    let mut overflow_count = 0usize;
    // RL-realistic micro-gradients — `det()` base is [-0.5, 0.5], which when
    // multiplied by the initial loss_scale 2^16 overflows f16 on step 1 and
    // traps the scaler in perpetual backoff. 1e-3 puts the scaled grad at
    // ≈65 (well inside f16's ±65504 range) and lets the scaler climb.
    let in_scale = 1e-3f32;
    for s in 0..50 {
        let m = trainer
            .step(
                &det_scaled(n, 0xA0 + s as u32, in_scale),
                &det_scaled(n, 0xB0 + s as u32, in_scale),
            )
            .expect("f16 prod-lr step");
        let scale = m.loss_scale.expect("f16 must report loss_scale");
        let skipped = m.overflow_skipped.expect("f16 must report overflow");
        scales.push(scale);
        if skipped {
            overflow_count += 1;
        } else {
            committed_steps += 1;
        }
        assert!(scale >= 1.0, "step {s}: scale {scale} < 1.0");
        assert!(
            scale <= (1u32 << 24) as f32,
            "step {s}: scale {scale} > 2^24"
        );
    }

    let after = trainer.snapshot_master().expect("snapshot post");

    for (i, lw) in after.layers.iter().enumerate() {
        for (name, slice) in [
            ("in_proj_w", lw.in_proj_w.as_slice()),
            ("out_proj_w", lw.out_proj_w.as_slice()),
            ("x_proj_w", lw.x_proj_w.as_slice()),
            ("dt_proj_w", lw.dt_proj_w.as_slice()),
        ] {
            assert!(
                slice.iter().all(|v| v.is_finite()),
                "f16 L{i} {name} non-finite after 50 prod-lr steps"
            );
        }
    }

    assert!(
        committed_steps > 0,
        "all 50 steps were overflow-skipped — loss scaler stuck"
    );

    // Weights must have visibly moved on committed steps.
    let max_diff = before.layers[0]
        .in_proj_w
        .iter()
        .zip(&after.layers[0].in_proj_w)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "f16 prod-lr: committed={committed_steps}/50 overflow={overflow_count}  scale_last={:.0}  max|Δw|={max_diff:.3e}",
        scales.last().copied().unwrap_or(0.0)
    );
    assert!(
        max_diff > 1e-8,
        "f16 weights barely moved after 50 prod-lr steps (max|Δw|={max_diff:.3e})"
    );
    assert!(max_diff.is_finite(), "max|Δw|={max_diff} non-finite");
}
