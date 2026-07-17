//! Bring-your-own-loss GPU training with the forward/backward split.
//!
//! ```bash
//! cargo run --release --example custom_loss --features cuda
//! ```
//!
//! The fused `MambaTrainer::step(input, d_temporal)` requires the caller to
//! already HOLD the loss gradient — but computing a loss needs the forward
//! activations first. The split API resolves that:
//!
//!   1. `forward()` — runs the training forward, returns the full
//!      `batch * seq_len * d_model` post-norm_f temporal output on the host;
//!   2. the caller computes ANY loss + its gradient w.r.t. that output
//!      (classification head, distillation KL, contrastive — plain Rust);
//!   3. `backward_step()` — backprops that gradient through the saved
//!      activations and runs AdamW (optionally with global-norm clipping,
//!      or `accumulate_only` for gradient accumulation across micro-batches).
//!
//! The demo trains a tiny backbone to push its mean-pooled output toward a
//! fixed target vector (MSE) — a stand-in for any real head/loss.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the `cuda` feature:");
    eprintln!("  cargo run --release --example custom_loss --features cuda");
}

#[cfg(feature = "cuda")]
mod cuda_example {
    use mamba_rs::MambaConfig;
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::mamba_ssm::gpu::trainer::{BackwardOpts, MambaTrainer, TrainSessionCfg};
    use mamba_rs::weights::MambaWeights;

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
        let mut trainer = MambaTrainer::new_full(
            0,
            &weights,
            cfg,
            TrainSessionCfg {
                input_dim,
                batch,
                seq_len,
                lr: 3e-3,
                weight_decay: 1e-2,
            },
            WeightDtype::Bf16,
        )?;

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
