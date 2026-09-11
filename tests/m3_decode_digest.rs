//! Decode-lane bit gate for Mamba-3, the twin of `decode_digest`:
//! tolerance checks cannot tell a bit-identical fusion from lucky
//! rounding, so this records FNV hashes of 16 chained decode steps (the
//! output after every step and the carried states at the end) on
//! synthetic weights, eager and graph, f32 and bf16. Record on a good
//! build; compare after any step-path edit.
//!
//!   cargo test --release --features cuda --test m3_decode_digest -- --ignored --nocapture

#![cfg(feature = "cuda")]

#[path = "common/digest.rs"]
mod digest;
#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/evidence_digest.rs"]
mod evidence_digest;

use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::inference::{Mamba3GpuInferenceEngine, Mamba3GpuInferenceMixed};
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn config() -> Mamba3Config {
    Mamba3Config {
        d_model: 384,
        n_layers: 24,
        d_state: 16,
        headdim: 16,
        ngroups: 1,
        ..Mamba3Config::default()
    }
}

fn report(lane: &str, arm: &str, outs: &[Vec<f32>], states: &[(&str, Vec<f32>)]) {
    // Sequential absorption, not an XOR fold: a running FNV state cannot
    // cancel, and the per-step lines say which step moved.
    let mut chain = digest::Digest::new();
    for (step_i, out) in outs.iter().enumerate() {
        eprintln!(
            "DECODE-DIGEST m3 {lane} {arm} step={step_i} {:016x}",
            digest::fnv1a_f32(out)
        );
        chain.absorb_bytes(b"step");
        chain.absorb_f32(out);
    }
    let chained = chain.finish();
    let mut line =
        format!("DECODE-DIGEST m3 {lane} {arm} steps=16 d384 L24: out_chain={chained:016x}");
    evidence_digest::record_digest(
        "m3_decode_digest",
        &format!("{lane}/{arm}"),
        "out_chain",
        chained,
    )
    .expect("acceptance evidence");
    for (name, v) in states {
        let h = digest::fnv1a_f32(v);
        line.push_str(&format!(" {name}={h:016x}"));
        evidence_digest::record_digest("m3_decode_digest", &format!("{lane}/{arm}"), name, h)
            .expect("acceptance evidence");
    }
    eprintln!("{line}");
}

#[test]
#[ignore = "bit-gate recorder"]
fn m3_decode_run_digest_f32() {
    let cfg = config();
    let input_dim = cfg.d_model;
    let batch = 1usize;
    // The mixed engine serves the LLM path only, where the embedding is
    // already d_model wide: no input projection on either lane, so both
    // digest the same model.
    let mut w = Mamba3Weights::init(&cfg, input_dim, 0xDEC0DE3);
    w.input_proj_w.clear();
    w.input_proj_b.clear();
    let device = GpuDevice::new(0).unwrap();
    for graph in [false, true] {
        let mut engine = Mamba3GpuInferenceEngine::new(&device, &w, cfg, input_dim, batch).unwrap();
        let mut state = engine.alloc_state().unwrap();
        let mut scratch = engine.alloc_scratch().unwrap();
        let mut out = vec![0f32; batch * cfg.d_model];
        if graph {
            engine
                .step(
                    &det(batch * input_dim, 900, 0.1),
                    &mut out,
                    &mut state,
                    &mut scratch,
                )
                .unwrap();
            unsafe { engine.capture_graph(&mut state, &mut scratch) }.unwrap();
            state.reset(&engine.ctx().stream).unwrap();
        }
        let mut outs = Vec::new();
        for step_i in 0..16u32 {
            let input = det(batch * input_dim, 1000 + step_i, 0.1);
            engine
                .step(&input, &mut out, &mut state, &mut scratch)
                .unwrap();
            outs.push(out.clone());
        }
        let stream = &engine.ctx().stream;
        let mut states = Vec::new();
        for (name, buf) in [
            ("ssm", &state.ssm_state),
            ("k", &state.k_state),
            ("v", &state.v_state),
            ("angle", &state.angle_state),
        ] {
            let mut v = vec![0f32; buf.len()];
            buf.download(stream, &mut v).unwrap();
            stream.synchronize().unwrap();
            states.push((name, v));
        }
        report("f32", if graph { "graph" } else { "eager" }, &outs, &states);
    }
}

#[test]
#[ignore = "bit-gate recorder"]
fn m3_decode_run_digest_bf16() {
    let cfg = config();
    let input_dim = cfg.d_model;
    let batch = 1usize;
    let mut w = Mamba3Weights::init(&cfg, input_dim, 0xDEC0DE3);
    w.input_proj_w.clear();
    w.input_proj_b.clear();
    let device = GpuDevice::new(0).unwrap();
    for graph in [false, true] {
        let mut engine =
            Mamba3GpuInferenceMixed::new(&device, &w, cfg, input_dim, batch, WeightDtype::Bf16)
                .unwrap();
        let mut state = engine.alloc_state().unwrap();
        let mut scratch = engine.alloc_mixed_scratch().unwrap();
        let mut out = vec![0f32; batch * cfg.d_model];
        if graph {
            engine
                .step_mixed_native(
                    &det(batch * input_dim, 900, 0.1),
                    &mut out,
                    &mut state,
                    &mut scratch,
                )
                .unwrap();
            unsafe { engine.capture_graph_mixed_native(&mut state, &mut scratch) }.unwrap();
            state.reset(engine.ctx_stream()).unwrap();
        }
        let mut outs = Vec::new();
        for step_i in 0..16u32 {
            let input = det(batch * input_dim, 1000 + step_i, 0.1);
            engine
                .step_mixed_native(&input, &mut out, &mut state, &mut scratch)
                .unwrap();
            outs.push(out.clone());
        }
        let stream = engine.ctx_stream();
        let mut states = Vec::new();
        for (name, buf) in [
            ("ssm", &state.ssm_state),
            ("k", &state.k_state),
            ("v", &state.v_state),
            ("angle", &state.angle_state),
        ] {
            let mut v = vec![0f32; buf.len()];
            buf.download(stream, &mut v).unwrap();
            stream.synchronize().unwrap();
            states.push((name, v));
        }
        report(
            "bf16",
            if graph { "graph" } else { "eager" },
            &outs,
            &states,
        );
    }
}
