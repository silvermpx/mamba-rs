//! Training on the GPU with a loss of your own, through the split forward
//! and backward.
//!
//! ```bash
//! cargo run --release --example custom_loss --features cuda
//! ```
//!
//! The fused `MambaGpuTrainer::step(input, d_temporal)` needs the loss
//! gradient up front, but a loss needs the forward's output first. The
//! split calls take it in three steps:
//!
//!   1. `forward()` runs the training forward and returns the whole
//!      `batch * seq_len * d_model` output after the final norm, on the host;
//!   2. you compute any loss and its gradient with respect to that output in
//!      plain Rust (a classification head, distillation KL, contrastive);
//!   3. `backward_step()` takes that gradient back through the saved
//!      activations and runs AdamW, with global-norm clipping if asked, or
//!      `accumulate_only` to sum gradients over micro-batches.
//!
//! The demo trains a tiny backbone to push its mean-pooled output toward a
//! fixed target vector (MSE), a stand-in for a real head and loss. The
//! trainer asks for `GemmMode::Deterministic` explicitly; `GpuOptions`
//! without a mode reads `MAMBA_RS_GEMM_MODE` instead.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the `cuda` feature:");
    eprintln!("  cargo run --release --example custom_loss --features cuda");
}

#[cfg(feature = "cuda")]
mod cuda_example {
    use mamba_rs::GpuOptions;
    use mamba_rs::MambaConfig;
    use mamba_rs::gpu::GemmMode;
    use mamba_rs::gpu::dtype::WeightDtype;
    use mamba_rs::gpu::train::{BackwardOpts, TrainSessionCfg};
    use mamba_rs::mamba1::MambaWeights;
    use mamba_rs::mamba1::gpu::trainer::MambaGpuTrainer;

    pub fn run() -> Result<(), String> {
        let cfg = MambaConfig {
            d_model: 64,
            n_layers: 2,
            ..MambaConfig::default()
        };
        let input_dim = 32;
        let (batch, seq_len) = (2usize, 16usize);
        let dm = cfg.d_model;
        let n_in = batch * seq_len * input_dim;
        let n_out = batch * seq_len * dm;

        let weights = MambaWeights::init(&cfg, input_dim, 42);
        // A trainer context runs the Triad family in the deterministic mode;
        // the split forward/backward path below always executes eagerly.
        let session = TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr: 3e-3,
            weight_decay: 1e-2,
        };
        let opts = GpuOptions::default()
            .dtype(WeightDtype::Bf16)
            .mode(GemmMode::Deterministic);
        let mut trainer = MambaGpuTrainer::new(&weights, cfg, session, opts)?;

        // Synthetic input and a fixed target for the mean-pooled feature.
        let input: Vec<f32> = (0..n_in)
            .map(|i| ((i % 97) as f32 / 97.0 - 0.5) * 0.2)
            .collect();
        let target: Vec<f32> = (0..dm)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();

        let mut temporal = vec![0.0f32; n_out];
        let mut d_temporal = vec![0.0f32; n_out];

        for step in 0..30 {
            trainer.reset_state()?;

            // 1. Forward: full temporal readback (f32 on every dtype).
            trainer.forward(&input, &mut temporal)?;

            // 2. Host-side loss: L = mean_b || meanpool_t(y) - target ||^2.
            //    dL/dy[b,t,d] = 2 * (pool[b,d] - target[d]) / (T * B).
            let mut loss = 0.0f64;
            for b in 0..batch {
                let sample = &temporal[b * seq_len * dm..(b + 1) * seq_len * dm];
                let mut pool = vec![0.0f32; dm];
                for row in sample.chunks(dm) {
                    for (p, &v) in pool.iter_mut().zip(row) {
                        *p += v;
                    }
                }
                for p in pool.iter_mut() {
                    *p /= seq_len as f32;
                }
                for (d, (&p, &tv)) in pool.iter().zip(&target).enumerate() {
                    let diff = p - tv;
                    loss += (diff * diff) as f64 / batch as f64;
                    let g = 2.0 * diff / (seq_len as f32 * batch as f32);
                    for t in 0..seq_len {
                        d_temporal[b * seq_len * dm + t * dm + d] = g;
                    }
                }
            }

            // 3. Backward + AdamW, with global-norm clipping at 1.0.
            let m = trainer
                .backward_step(&d_temporal, BackwardOpts::default().with_clip_max_norm(1.0))?;
            if step % 5 == 0 {
                println!(
                    "step {:2}  loss {:.5}  grad_norm {:.4}",
                    step,
                    loss,
                    m.grad_norm.unwrap_or(f32::NAN)
                );
            }

            // A cosine-ish LR decay via set_lr (illegal under a captured
            // graph; the split path is always eager, so it just works).
            trainer.set_lr(3e-3 * (1.0 - step as f32 / 40.0))?;
        }

        // Gradient accumulation: two micro-batches, one optimizer step.
        // Scale d_temporal by 1/n_micro caller-side for loss averaging.
        trainer.reset_state()?;
        trainer.forward(&input, &mut temporal)?;
        let half: Vec<f32> = d_temporal.iter().map(|g| g * 0.5).collect();
        trainer.backward_step(&half, BackwardOpts::default().with_accumulate_only(true))?;
        trainer.reset_state()?;
        trainer.forward(&input, &mut temporal)?;
        let m = trainer.backward_step(&half, BackwardOpts::default())?;
        println!("accumulated step -> adam step {}", m.step);

        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn main() {
    if let Err(e) = cuda_example::run() {
        eprintln!("custom_loss example failed: {e}");
        std::process::exit(1);
    }
}
