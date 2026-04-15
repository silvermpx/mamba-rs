//! Mamba-3 bf16 mixed-precision training with CUDA Graph capture.
//!
//! ```bash
//! cargo run --release --example mamba3_gpu_training_bf16 --features cuda
//! ```
//!
//! M3 analogue of `gpu_training_bf16.rs`. Uses [`Mamba3Trainer`] to run
//! full forward+backward+AdamW+sync_master_to_compute per step, captures
//! the whole thing as one CUDA Graph, and reports eager-vs-graph timing.

#![cfg(feature = "cuda")]

use std::time::Instant;

use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::trainer::Mamba3Trainer;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

/// Deterministic pseudo-random input. `scale` caps the magnitude — with
/// synthetic random d_temporal (no real loss signal), un-scaled gradients
/// blow up master weights to NaN within ~100 steps.
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

fn main() -> Result<(), String> {
    let cfg = Mamba3Config {
        d_model: 256,
        d_state: 16,
        expand: 2,
        headdim: 32,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
    };
    let input_dim = cfg.d_model;
    let batch = 4;
    let seq_len = 128;
    let n = batch * seq_len * input_dim;
    let warmup_steps = 3;
    let eager_steps = 20;
    let graph_steps = 100;

    println!(
        "M3 bf16 trainer demo: d_model={} n_layers={} headdim={} nheads={} batch={} seq_len={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.headdim,
        cfg.nheads(),
        batch,
        seq_len,
    );

    let mut cpu = Mamba3Weights::init(&cfg, input_dim, 0xDECADE);
    // Identity input_proj (HF-style) so the M3 mixed forward takes the
    // D2D fast path instead of a GEMM with an unused input projection.
    cpu.input_proj_w.clear();
    cpu.input_proj_b.clear();

    // Modest lr for synthetic-gradient stability (see gpu_training_bf16.rs).
    let lr = 1e-5_f32;
    let wd = 1e-2_f32;
    let mut trainer = Mamba3Trainer::new_full(
        0,
        &cpu,
        cfg,
        input_dim,
        batch,
        seq_len,
        WeightDtype::Bf16,
        lr,
        wd,
    )?;

    // Warmup eager.
    for s in 0..warmup_steps {
        let m = trainer.step(&det(n, 0xA0 + s as u32, 0.1), &det(n, 0xB0 + s as u32, 0.1))?;
        assert!(!m.graph_replayed);
    }

    let t0 = Instant::now();
    for s in 0..eager_steps {
        trainer.step(&det(n, 0xC0 + s as u32, 0.1), &det(n, 0xD0 + s as u32, 0.1))?;
    }
    trainer
        .ctx()
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let eager_ms = t0.elapsed().as_secs_f64() * 1000.0 / eager_steps as f64;
    println!("eager step: {eager_ms:.3} ms/step ({eager_steps} steps)");

    trainer.capture_graph()?;
    assert!(trainer.has_graph());

    let t1 = Instant::now();
    for s in 0..graph_steps {
        let m = trainer.step(&det(n, 0xE0 + s as u32, 0.1), &det(n, 0xF0 + s as u32, 0.1))?;
        assert!(m.graph_replayed);
    }
    trainer
        .ctx()
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let graph_ms = t1.elapsed().as_secs_f64() * 1000.0 / graph_steps as f64;
    println!("graph step: {graph_ms:.3} ms/step ({graph_steps} steps)");
    println!("speedup: {:.2}x", eager_ms / graph_ms);

    let snap = trainer.snapshot_master()?;
    let l0_norm: f32 = snap.layers[0]
        .in_proj_w
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();
    println!(
        "snapshot OK: L0 in_proj_w len={} ||w||={:.4} (finite={})",
        snap.layers[0].in_proj_w.len(),
        l0_norm,
        l0_norm.is_finite()
    );
    // NB: synthetic random d_temporal provides no real loss signal — over
    // many steps the bf16 SSM recurrence may diverge. Expected for a
    // graph-capture benchmark.

    Ok(())
}
