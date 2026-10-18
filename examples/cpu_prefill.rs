//! A prompt in one pass on the CPU, then decode token by token.
//!
//! ```bash
//! cargo run --release --example cpu_prefill --features gemm-blas
//! ```
//!
//! A decode step runs one token through the model. A whole prompt of T
//! tokens that way pays T trips through every layer. `forward_sequence`
//! takes the window at once instead: its matrix products run batched over
//! the T rows on the thread pool, it writes the output at every position,
//! and it carries the recurrent state, so `forward_step` continues the
//! sequence after it. Every family's CPU backbone has the same two calls.
//!
//! A build without `gemm-blas` (or `accelerate` on macOS) uses the
//! pure-Rust scalar GEMM, several times slower.

use std::time::Instant;

use mamba_rs::{MambaBackbone, MambaConfig};

fn main() -> Result<(), String> {
    let cfg = MambaConfig {
        d_model: 256,
        n_layers: 8,
        ..MambaConfig::default()
    };
    let input_dim = 128;
    let seq_len = 512;
    let backbone = MambaBackbone::init(cfg, input_dim, 42)?;
    let dm = backbone.d_model();

    // A synthetic prompt: T feature vectors.
    let prompt: Vec<f32> = (0..seq_len * input_dim)
        .map(|i| ((i % 251) as f32 / 251.0 - 0.5) * 0.1)
        .collect();

    // The whole prompt in one call.
    let mut state = backbone.alloc_state();
    let mut scratch = backbone.alloc_scratch();
    let mut rows = vec![0.0f32; seq_len * dm];
    let start = Instant::now();
    backbone.forward_sequence(&prompt, &mut rows, &mut state, &mut scratch, seq_len);
    let window_ms = start.elapsed().as_secs_f64() * 1000.0;

    // The same tokens one step at a time, for comparison.
    let mut step_state = backbone.alloc_state();
    let mut step_out = vec![0.0f32; dm];
    let start = Instant::now();
    for token in prompt.chunks(input_dim) {
        backbone.forward_step(token, &mut step_out, &mut step_state, &mut scratch);
    }
    let steps_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("window (T={seq_len}):    {window_ms:8.2} ms");
    println!(
        "step loop (T={seq_len}): {steps_ms:8.2} ms  ({:.1}x)",
        steps_ms / window_ms
    );

    // Decode continues from the state the window left.
    let next_input = vec![0.05f32; input_dim];
    let mut next_out = vec![0.0f32; dm];
    backbone.forward_step(&next_input, &mut next_out, &mut state, &mut scratch);
    println!("decode after the window: out[0..4] = {:?}", &next_out[..4]);

    // Every position of the window carries the final norm, so a pooling
    // head reads them all, here a mean pool for classification.
    let mut pooled = vec![0.0f32; dm];
    for row in rows.chunks(dm) {
        for (p, &v) in pooled.iter_mut().zip(row) {
            *p += v;
        }
    }
    for p in pooled.iter_mut() {
        *p /= seq_len as f32;
    }
    println!("mean-pooled feature: pooled[0..4] = {:?}", &pooled[..4]);
    Ok(())
}
