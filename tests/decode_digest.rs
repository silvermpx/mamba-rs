//! Decode-lane bit gate. KL-tolerance coverage alone is not enough for
//! the step (autoregressive) path — tolerance cannot distinguish a
//! bit-identical fusion from lucky rounding. This records FNV hashes of
//! 16 chained decode steps (output after every step + the carried conv
//! and SSM state at the end) on synthetic weights, per tier and graph
//! mode. Record on a good build; compare after any step-path edit.
//!
//!   cargo test --release --features cuda --test decode_digest -- --ignored --nocapture

#![cfg(feature = "cuda")]

mod common;

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::inference::GpuMambaInference;
use mamba_rs::weights::MambaWeights;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

#[test]
#[ignore = "bit-gate recorder"]
fn decode_run_digest() {
    let cfg = MambaConfig {
        d_model: 384,
        n_layers: 24,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
    };
    let input_dim = 384usize;
    let batch = 1usize;
    let mut w = MambaWeights::init(&cfg, input_dim, 0xDEC0DE);
    for lw in w.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }
    let device = GpuDevice::new(0).unwrap();

    for graph in [false, true] {
        let engine = GpuMambaInference::new(&device, &w, cfg, input_dim, batch).unwrap();
        let mut state = engine.alloc_state().unwrap();
        let mut scratch = engine.alloc_scratch().unwrap();
        let mut out = vec![0f32; batch * cfg.d_model];
        let mut engine = engine;
        if graph {
            // One warm step before capture, per the capture contract.
            engine
                .step(
                    &det(batch * input_dim, 900, 0.1),
                    &mut out,
                    &mut state,
                    &mut scratch,
                )
                .unwrap();
            // State and scratch stay alive until the engine graph is cleared.
            unsafe { engine.capture_graph(&mut state, &mut scratch) }.unwrap();
            // Reset the carried state so both modes hash the same run.
            state.reset(&engine.ctx().stream).unwrap();
        }
        // Sequential absorption, not an XOR fold: XOR is linear, so a
        // pair of correlated step changes could cancel; a running FNV
        // state makes cancellation impossible and the per-step lines
        // say WHICH step moved when the chain does.
        let mut chain = common::digest::Digest::new();
        for step_i in 0..16u32 {
            let input = det(batch * input_dim, 1000 + step_i, 0.1);
            engine
                .step(&input, &mut out, &mut state, &mut scratch)
                .unwrap();
            eprintln!(
                "DECODE-DIGEST step={step_i} {:016x}",
                common::bench::fnv1a_f32(&out)
            );
            chain.absorb_label("step");
            chain.absorb_f32(&out);
        }
        let chained = chain.finish();
        let mut conv_v = vec![0f32; cfg.n_layers * cfg.d_inner() * cfg.d_conv];
        let mut ssm_v = vec![0f32; cfg.n_layers * cfg.d_inner() * cfg.d_state];
        state
            .conv
            .download(&engine.ctx().stream, &mut conv_v)
            .unwrap();
        state
            .ssm
            .download(&engine.ctx().stream, &mut ssm_v)
            .unwrap();
        engine.ctx().stream.synchronize().unwrap();
        let h_conv = common::bench::fnv1a_f32(&conv_v);
        let h_ssm = common::bench::fnv1a_f32(&ssm_v);
        eprintln!(
            "DECODE-DIGEST graph={graph} steps=16 d384 L24: out_chain={chained:016x} \
             conv={h_conv:016x} ssm={h_ssm:016x}"
        );
        let arm = if graph { "graph" } else { "eager" };
        common::evidence::record_digest("decode_digest", arm, "out_chain", chained);
        common::evidence::record_digest("decode_digest", arm, "conv", h_conv);
        common::evidence::record_digest("decode_digest", arm, "ssm", h_ssm);
    }
}
