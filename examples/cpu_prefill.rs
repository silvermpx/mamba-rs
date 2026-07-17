//! Full-sequence CPU prefill, then per-step decode (prefill-then-decode).
//!
//! ```bash
//! cargo run --release --example cpu_prefill --features gemm-blas
//! ```
//!
//! The per-step inference path costs one matvec pipeline per token — a
//! whole prompt/page of T tokens pays T dispatches. `forward_prefill` runs
//! the training forward's batched-SGEMM pipeline instead (no activation
//! tape), writes the post-norm_f output at EVERY position, and carries the
//! recurrent state so `forward_step` continues seamlessly.
//!
//! Featureless builds work but use the pure-Rust scalar GEMM (5-20x
//! slower) — real deployments enable `gemm-blas` (or `accelerate` on
//! macOS). `PrefillMode::Parallel` additionally parallelizes every phase
//! for single-sequence latency; batches of independent sequences should
//! prefer `prefill_batch` (one core per sequence).

use std::time::Instant;

use mamba_rs::MambaConfig;
use mamba_rs::inference::PrefillMode;
use mamba_rs::module::MambaBackbone;

fn main() {
    let cfg = MambaConfig {
        d_model: 256,
        n_layers: 8,
        ..MambaConfig::default()
    };
    let input_dim = 128;
    let seq_len = 512;
    let backbone = MambaBackbone::init(cfg, input_dim, 42);
    let dm = backbone.config().d_model;

    // A synthetic "prompt": T feature vectors.
    let prompt: Vec<f32> = (0..seq_len * input_dim)
        .map(|i| ((i % 251) as f32 / 251.0 - 0.5) * 0.1)
        .collect();

    // Prefill the whole prompt in one call.
    let mut state = backbone.alloc_state();
    let mut prefill_scratch = backbone.alloc_prefill_scratch(seq_len);
    let mut prefill_out = vec![0.0f32; seq_len * dm];
    let start = Instant::now();
    backbone.forward_prefill(
        &prompt,
        &mut prefill_out,
        &mut state,
        &mut prefill_scratch,
        seq_len,
        PrefillMode::Parallel,
    );
    let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;

    // The same tokens through the per-step loop, for comparison.
    let mut step_state = backbone.alloc_state();
    let mut step_scratch = backbone.alloc_scratch();
    let mut step_out = vec![0.0f32; seq_len * dm];
    let start = Instant::now();
    backbone.forward_sequence(
        &prompt,
        &mut step_out,
        &mut step_state,
        &mut step_scratch,
        seq_len,
    );
    let steps_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("prefill (T={seq_len}):  {prefill_ms:8.2} ms");
    println!(
        "step loop (T={seq_len}): {steps_ms:8.2} ms  ({:.1}x)",
        steps_ms / prefill_ms
    );

    // Decode continues FROM the prefilled state — feed the next token.
    let next_input = vec![0.05f32; input_dim];
    let mut next_out = vec![0.0f32; dm];
    backbone.forward_step(&next_input, &mut next_out, &mut state, &mut step_scratch);
    println!("decode after prefill: out[0..4] = {:?}", &next_out[..4]);

    // Pooling consumers read every position of `prefill_out` (norm_f is
    // applied at each one), e.g. a mean-pool for classification:
    let mut pooled = vec![0.0f32; dm];
    for row in prefill_out.chunks(dm) {
        for (p, &v) in pooled.iter_mut().zip(row) {
            *p += v;
        }
    }
    for p in pooled.iter_mut() {
        *p /= seq_len as f32;
    }
    println!("mean-pooled feature: pooled[0..4] = {:?}", &pooled[..4]);
}
