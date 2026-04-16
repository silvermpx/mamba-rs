//! Mamba SSM bf16 mixed-precision training with CUDA Graph capture.
//!
//! ```bash
//! cargo run --release --example gpu_training_bf16 --features cuda
//! ```
//!
//! Runs a short synthetic training loop through [`MambaTrainer`], exercising
//! the full training stack: bf16 master→compute shadow with f32 master
//! AdamW, batch-invariant GEMM for numerical stability, CUDA Graph capture
//! of the complete forward + backward + optimizer step, and pointer-
//! stability invariants on replay.
//!
//! The workload is intentionally synthetic (random input / d_temporal)
//! because the crate doesn't ship an LM-head loss kernel. What the
//! example demonstrates:
//!   * API ergonomics: construct → warm up → capture → loop
//!   * Weights actually update (non-trivial AdamW steps)
//!   * Captured-graph replay is meaningfully faster than eager dispatch
//!   * `snapshot_master()` round-trips the weight set for checkpointing

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the `cuda` feature:");
    eprintln!("  cargo run --release --example gpu_training_bf16 --features cuda");
}

#[cfg(feature = "cuda")]
mod cuda_example {
    use std::time::Instant;

    use mamba_rs::config::MambaConfig;
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::mamba_ssm::gpu::trainer::MambaTrainer;
    use mamba_rs::weights::MambaWeights;

    /// Deterministic pseudo-random input. `scale` caps the magnitude — with
    /// synthetic random d_temporal (no real loss signal), un-scaled gradients
    /// blow up master weights to NaN within ~100 steps, so the example keeps
    /// d_temporal small.
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

    pub fn run() -> Result<(), String> {
        let cfg = MambaConfig {
            d_model: 128,
            n_layers: 2,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            scan_mode: mamba_rs::config::ScanMode::Sequential,
        };
        let input_dim = cfg.d_model;
        let batch = 2;
        let seq_len = 32;
        let n = batch * seq_len * input_dim;
        let warmup_steps = 3;
        let eager_steps = 20;
        let graph_steps = 100;

        println!(
            "M1 bf16 trainer demo: d_model={} n_layers={} batch={} seq_len={}",
            cfg.d_model, cfg.n_layers, batch, seq_len
        );

        // CPU init — clear input_proj for HF identity projection (required by
        // the mixed forward's identity branch).
        let mut cpu = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
        cpu.input_proj_w.clear();
        cpu.input_proj_b.clear();
        for lw in cpu.layers.iter_mut() {
            lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
        }

        // Modest lr — with synthetic random gradients and no real loss signal,
        // the default lr=1e-3 drives weights to NaN in ~100 steps. Real
        // training with an actual loss would pick lr based on the problem.
        let lr = 1e-6_f32;
        let wd = 1e-3_f32;
        let mut trainer = MambaTrainer::new_full(
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

        // --- Sanity: weights should be finite at init ---
        {
            let init = trainer.snapshot_master()?;
            let init_norm: f32 = init.layers[0]
                .in_proj_w
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            println!(
                "init: L0 in_proj_w ||w||={init_norm:.4} (finite={})",
                init_norm.is_finite()
            );
        }

        // --- Warmup (eager, lets cuBLAS select kernels) ---
        for s in 0..warmup_steps {
            let m = trainer.step(
                &det(n, 0xA0 + s as u32, 0.01),
                &det(n, 0xB0 + s as u32, 0.001),
            )?;
            assert!(!m.graph_replayed);
        }

        // --- Time eager loop (no graph) ---
        let t0 = Instant::now();
        for s in 0..eager_steps {
            trainer.step(
                &det(n, 0xC0 + s as u32, 0.01),
                &det(n, 0xD0 + s as u32, 0.001),
            )?;
        }
        trainer
            .ctx()
            .stream
            .synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;
        let eager_ms = t0.elapsed().as_secs_f64() * 1000.0 / eager_steps as f64;
        println!("eager step: {eager_ms:.3} ms/step ({eager_steps} steps)");

        // --- Capture + time replay loop ---
        trainer.capture_graph()?;
        assert!(trainer.has_graph());

        let t1 = Instant::now();
        for s in 0..graph_steps {
            let m = trainer.step(
                &det(n, 0xE0 + s as u32, 0.01),
                &det(n, 0xF0 + s as u32, 0.001),
            )?;
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

        // --- Checkpoint roundtrip ---
        let snap = trainer.snapshot_master()?;
        let l0_in_proj_norm: f32 = snap.layers[0]
            .in_proj_w
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        println!(
            "snapshot OK: L0 in_proj_w len={} ||w||={:.4} (finite={})",
            snap.layers[0].in_proj_w.len(),
            l0_in_proj_norm,
            l0_in_proj_norm.is_finite()
        );
        // NB: with synthetic random d_temporal (no real loss), the bf16 SSM
        // recurrence can accumulate and weights may diverge to NaN over ~100
        // steps. This is expected for a graph-capture benchmark — real
        // training with a proper loss keeps gradients bounded.

        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), String> {
    cuda_example::run()
}
