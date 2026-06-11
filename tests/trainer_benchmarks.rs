//! Step 23 — workload-style training benchmarks via the high-level
//! `MambaTrainer` / `Mamba3Trainer` API, sweeping all three precisions
//! (f32 / bf16 / f16) × eager vs CUDA Graph.
//!
//! Two workloads:
//!   * **LM** — Mamba-1 at mamba-130m-ish shape (d_model=768, n_layers=24,
//!     batch=2, seq_len=64) — full forward+backward+AdamW+sync per step
//!   * **RL** — Mamba-3 at SQV-RS actor shape (d_model=128, n_layers=4,
//!     headdim=16, ngroups=1, batch=64, seq_len=32)
//!
//! All tests are `#[ignore]` — opt-in:
//!   cargo test --release --features cuda --test trainer_benchmarks -- --ignored --nocapture
//!
//! The outputs are eager-vs-graph timing tables for each dtype, written
//! to stderr. Use these to track the real cost of a training step at
//! release-realistic shapes.

#![cfg(feature = "cuda")]

use std::time::Instant;

use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;

const WARMUP: usize = 3;
const STEPS_EAGER: usize = 10;
const STEPS_GRAPH: usize = 30;

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

// ═══════════════════════════════════════════════════════════════════════
// LM workload — Mamba-1 at 130m-ish shape, all 3 dtypes
// ═══════════════════════════════════════════════════════════════════════

fn lm_cfg() -> mamba_rs::config::MambaConfig {
    mamba_rs::config::MambaConfig {
        d_model: 768,
        n_layers: 24,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
    }
}

fn run_lm_for_dtype(dtype: WeightDtype) -> Result<(), String> {
    use mamba_rs::mamba_ssm::gpu::trainer::{MambaTrainer, TrainSessionCfg};
    use mamba_rs::weights::MambaWeights;

    let cfg = lm_cfg();
    let input_dim = cfg.d_model;
    let batch = 2;
    let seq_len = 64;
    let n = batch * seq_len * input_dim;
    let label = format!("{dtype:?}");

    let mut cpu = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    if !matches!(dtype, WeightDtype::F32) {
        cpu.input_proj_w.clear();
        cpu.input_proj_b.clear();
    }
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    // Conservative lr so f16 doesn't NaN out before timing finishes.
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
        dtype,
    )?;

    // Warmup
    for s in 0..WARMUP {
        trainer.step(
            &det(n, 0xA0 + s as u32, 0.01),
            &det(n, 0xB0 + s as u32, 0.01),
        )?;
    }
    trainer
        .ctx()
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;

    // Eager timing
    let t0 = Instant::now();
    for s in 0..STEPS_EAGER {
        trainer.step(
            &det(n, 0xC0 + s as u32, 0.01),
            &det(n, 0xD0 + s as u32, 0.01),
        )?;
    }
    trainer
        .ctx()
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let eager_ms = t0.elapsed().as_secs_f64() * 1000.0 / STEPS_EAGER as f64;

    // Capture + graph timing
    trainer.capture_graph()?;
    assert!(trainer.has_graph());
    let t1 = Instant::now();
    for s in 0..STEPS_GRAPH {
        let m = trainer.step(
            &det(n, 0xE0 + s as u32, 0.01),
            &det(n, 0xF0 + s as u32, 0.01),
        )?;
        assert!(m.graph_replayed);
    }
    trainer
        .ctx()
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let graph_ms = t1.elapsed().as_secs_f64() * 1000.0 / STEPS_GRAPH as f64;

    eprintln!(
        "LM {label:5}  eager={eager_ms:7.3} ms/step  graph={graph_ms:7.3} ms/step  speedup={:.2}x",
        eager_ms / graph_ms
    );
    Ok(())
}

#[test]
#[ignore]
fn bench_lm_train_f32() {
    run_lm_for_dtype(WeightDtype::F32).unwrap();
}

#[test]
#[ignore]
fn bench_lm_train_bf16() {
    run_lm_for_dtype(WeightDtype::Bf16).unwrap();
}

#[test]
#[ignore]
fn bench_lm_train_f16() {
    run_lm_for_dtype(WeightDtype::F16).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════
// RL workload — Mamba-3 actor shape (SQV-RS-ish), all 3 dtypes
// ═══════════════════════════════════════════════════════════════════════

fn rl_cfg() -> mamba_rs::mamba3_siso::config::Mamba3Config {
    // Note: `is_outproj_norm: true` to enable the M3 mixed backward path
    // (the no-norm gating variant isn't wired for mixed bwd). For pure
    // f32 training either flag works.
    mamba_rs::mamba3_siso::config::Mamba3Config {
        d_model: 128,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 1,
        n_layers: 4,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
    }
}

fn run_rl_for_dtype(dtype: WeightDtype) -> Result<(), String> {
    use mamba_rs::mamba3_siso::gpu::trainer::{Mamba3Trainer, TrainSessionCfg};
    use mamba_rs::mamba3_siso::weights::Mamba3Weights;

    let cfg = rl_cfg();
    let input_dim = cfg.d_model;
    let batch = 64;
    let seq_len = 32;
    let n = batch * seq_len * input_dim;
    let label = format!("{dtype:?}");

    let mut cpu = Mamba3Weights::init(&cfg, input_dim, 0xDECADE);
    if matches!(dtype, WeightDtype::F32) {
        // f32 M3 forward needs a real input_proj (no identity branch).
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
    } else {
        cpu.input_proj_w.clear();
        cpu.input_proj_b.clear();
    }

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
        dtype,
    )?;

    for s in 0..WARMUP {
        trainer.step(
            &det(n, 0xA0 + s as u32, 0.01),
            &det(n, 0xB0 + s as u32, 0.01),
        )?;
    }
    trainer
        .ctx()
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;

    let t0 = Instant::now();
    for s in 0..STEPS_EAGER {
        trainer.step(
            &det(n, 0xC0 + s as u32, 0.01),
            &det(n, 0xD0 + s as u32, 0.01),
        )?;
    }
    trainer
        .ctx()
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let eager_ms = t0.elapsed().as_secs_f64() * 1000.0 / STEPS_EAGER as f64;

    trainer.capture_graph()?;
    let t1 = Instant::now();
    for s in 0..STEPS_GRAPH {
        let m = trainer.step(
            &det(n, 0xE0 + s as u32, 0.01),
            &det(n, 0xF0 + s as u32, 0.01),
        )?;
        assert!(m.graph_replayed);
    }
    trainer
        .ctx()
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let graph_ms = t1.elapsed().as_secs_f64() * 1000.0 / STEPS_GRAPH as f64;

    eprintln!(
        "RL {label:5}  eager={eager_ms:7.3} ms/step  graph={graph_ms:7.3} ms/step  speedup={:.2}x",
        eager_ms / graph_ms
    );
    Ok(())
}

#[test]
#[ignore]
fn bench_rl_train_f32() {
    run_rl_for_dtype(WeightDtype::F32).unwrap();
}

#[test]
#[ignore]
fn bench_rl_train_bf16() {
    run_rl_for_dtype(WeightDtype::Bf16).unwrap();
}

#[test]
#[ignore]
fn bench_rl_train_f16() {
    run_rl_for_dtype(WeightDtype::F16).unwrap();
}
